"""The phase 3 CLI surface, exercised against a real database.

Every test here runs the fake provider end to end and then reads the result
back through the command a user or a CI job would actually invoke, so a
regression in the materialization path, the storage round trip or the JSON
envelope shows up here rather than only in a unit test of one layer.
"""

from __future__ import annotations

import csv
import io
import json
from typing import TYPE_CHECKING, Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from llm_eval_lab.cli.main import ExitCode, app
from llm_eval_lab.pricing.loader import load_price_table
from llm_eval_lab.redaction import redact_url
from llm_eval_lab.reporting.formatters import markup_safe, neutralise_formula
from llm_eval_lab.settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

SMOKE_CASES = 6
CANARY = "sk-canary-DO-NOT-LEAK-0123456789"
"""The same literal `tests/integration/test_secret_leak.py` plants and hunts for."""

DSN_WITH_CANARY = f"postgresql+asyncpg://admin:{CANARY}@db.example.test:5432/evals"
"""A PostgreSQL DSN carrying the canary where a key-name scrubber cannot see it."""

REDACTED_DSN = "postgresql+asyncpg://admin:[redacted]@db.example.test:5432/evals"
"""What the DSN above must render as: everything but the password survives."""

REQUIRED_METRIC_KEYS = (
    "pass_rate",
    "pass_rate_ci",
    "error_rate",
    "error_policy",
    "n_score_only",
    "token_usage_coverage",
    "cost_coverage",
    "cost_per_case",
    "cost_per_successful_evaluation",
    "by_category",
    "by_tag",
    "by_evaluator",
)
"""Keys the documented `metrics --json` contract promises."""


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, db_url: str) -> None:
    """Point every CLI invocation at a test-private database."""
    monkeypatch.setenv("LLM_EVAL_DATABASE_URL", db_url)
    get_settings.cache_clear()


def execute_run(
    repo_root: Path,
    *extra: str,
    expected_exit: ExitCode = ExitCode.SUCCESS,
) -> str:
    """Run the smoke suite against the fake provider and return the run id.

    A run with an errored case exits 5 rather than 0, so the expected code is a
    parameter: a helper that asserted success would have to skip the assertion
    entirely for the partial-run fixtures.
    """
    result = runner.invoke(
        app,
        [
            "run",
            str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
            "--provider",
            "fake",
            "--model",
            "fake-1",
            "--fake-mode",
            "expected",
            "--json",
            *extra,
        ],
    )
    assert result.exit_code == expected_exit, result.output
    run_id: str = json.loads(result.stdout)["run_id"]
    return run_id


@pytest.fixture
def run_id(repo_root: Path, migrated_db: str) -> str:
    """A completed run over the six-case smoke suite."""
    del migrated_db
    return execute_run(repo_root)


def metrics_of(run_id: str, *extra: str) -> dict[str, Any]:
    """Read one run's rollup through the CLI and return the metrics document."""
    result = runner.invoke(app, ["metrics", run_id, "--json", *extra])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    document: dict[str, Any] = json.loads(result.stdout)
    return document


# --- the documented metrics envelope --------------------------------------


def test_metrics_json_carries_every_documented_key(run_id: str) -> None:
    metrics = metrics_of(run_id)["metrics"]
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    assert not missing, f"metrics --json is missing {missing}"

    assert "latency" in metrics
    for key in ("p50_ms", "p95_ms", "p99_ms", "low_confidence"):
        assert key in metrics["latency"], key


def test_metrics_are_materialized_at_run_finalization(run_id: str) -> None:
    # The rollup exists without anybody asking for it, which is what stops a
    # listing of fifty runs re-deriving fifty rollups.
    assert metrics_of(run_id)["materialized"] is True


def test_metrics_report_the_smoke_suite_passing_completely(run_id: str) -> None:
    metrics = metrics_of(run_id)["metrics"]
    assert metrics["n_cases"] == SMOKE_CASES
    assert metrics["n_completed"] == SMOKE_CASES
    assert metrics["pass_rate"] == 1.0
    assert metrics["n_passed"] == SMOKE_CASES
    assert metrics["n_pass_denominator"] == SMOKE_CASES
    assert metrics["error_rate"] == 0.0
    assert metrics["error_policy"] == "exclude"


def test_the_pass_rate_interval_is_reported_next_to_the_rate(run_id: str) -> None:
    interval = metrics_of(run_id)["metrics"]["pass_rate_ci"]
    lower, upper = interval
    assert 0.0 <= lower <= 1.0
    assert upper == 1.0
    assert lower < 1.0, "a Wilson interval over six cases must not collapse onto the point estimate"


def test_the_breakdowns_are_populated(run_id: str) -> None:
    metrics = metrics_of(run_id)["metrics"]
    assert set(metrics["by_category"]) == {"arithmetic", "formatting", "geography", "science"}
    assert "smoke" in metrics["by_tag"]
    assert {"exact_match", "exact_match_ci", "contains", "regex", "numeric_tolerance"} <= set(
        metrics["by_evaluator"]
    )


def test_a_category_carrying_brackets_survives_the_human_breakdown(
    tmp_path: Path, migrated_db: str
) -> None:
    """Rich drops an unresolvable style tag, so an unescaped cell is deleted, not restyled."""
    del migrated_db
    suite = tmp_path / "bracketed.yaml"
    suite.write_text(
        "schema_version: 1\n"
        "name: bracketed\n"
        'version: "1"\n'
        "cases:\n"
        "  - id: bracketed-case\n"
        '    category: "[urgent] triage"\n'
        "    input: Say alpha\n"
        "    expected: alpha\n"
        "    evaluators:\n"
        "      - type: exact_match\n",
        encoding="utf-8",
    )
    started = runner.invoke(
        app,
        [
            "run",
            str(suite),
            "--provider",
            "fake",
            "--model",
            "fake-1",
            "--fake-mode",
            "expected",
            "--json",
        ],
    )
    assert started.exit_code == ExitCode.SUCCESS, started.output
    reported = runner.invoke(app, ["metrics", json.loads(started.stdout)["run_id"]])

    assert reported.exit_code == ExitCode.SUCCESS, reported.output
    assert "[urgent] triage" in reported.output


def test_per_evaluator_scores_are_reported_separately(run_id: str) -> None:
    by_evaluator = metrics_of(run_id)["metrics"]["by_evaluator"]
    for bucket in by_evaluator.values():
        assert "mean_score" in bucket
        assert "median_score" in bucket
    assert "blended_score" not in metrics_of(run_id)["metrics"]


# --- the low-confidence gates ---------------------------------------------


def test_a_small_run_marks_its_untrustworthy_percentiles_and_still_reports_them(
    run_id: str,
) -> None:
    latency = metrics_of(run_id)["metrics"]["latency"]

    assert latency["low_confidence"] == ["p90_ms", "p95_ms", "p99_ms"]
    assert "p50_ms" not in latency["low_confidence"], "six samples meets the p50 gate"
    assert isinstance(latency["p50_ms"], float)
    assert isinstance(latency["p95_ms"], float)
    assert isinstance(latency["p99_ms"], float)
    assert latency["n"] == SMOKE_CASES


def test_the_human_output_explains_the_low_confidence_marker(run_id: str) -> None:
    result = runner.invoke(app, ["metrics", run_id])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    assert "low-n" in result.output
    assert "insufficient" in result.output.lower() or "trustworthy" in result.output.lower()


# --- cost -----------------------------------------------------------------


def test_cost_figures_carry_their_coverage(run_id: str) -> None:
    metrics = metrics_of(run_id)["metrics"]
    assert metrics["cost_coverage"] == 1.0
    assert metrics["token_usage_coverage"] == 1.0
    assert metrics["cost"]["price_table_id"] == "builtin"
    assert metrics["cost"]["priced"] is True


def test_money_is_rendered_as_exact_text_not_a_float(run_id: str) -> None:
    metrics = metrics_of(run_id)["metrics"]
    assert isinstance(metrics["cost"]["total_cost"], str)
    assert isinstance(metrics["cost_per_case"], str)


# --- recompute and missing rollups ----------------------------------------


def test_recompute_reproduces_the_materialized_figures(run_id: str) -> None:
    stored = metrics_of(run_id)["metrics"]
    recomputed = metrics_of(run_id, "--recompute")["metrics"]

    for key in ("pass_rate", "n_passed", "n_pass_denominator", "error_rate", "cost_per_case"):
        assert recomputed[key] == stored[key], key


def test_metrics_on_an_unknown_run_exits_with_a_usage_code(migrated_db: str) -> None:
    del migrated_db
    result = runner.invoke(app, ["metrics", "no-such-run", "--json"])
    assert result.exit_code == ExitCode.USAGE
    assert json.loads(result.stdout)["error"] == "not_found"


# --- a run with errored cases ---------------------------------------------


@pytest.fixture
def partial_run_id(repo_root: Path, migrated_db: str) -> str:
    """A run in which one case fails on every attempt."""
    del migrated_db
    return execute_run(
        repo_root,
        "--provider-option",
        'fail_case_ids=["capital-exact"]',
        expected_exit=ExitCode.RUN_INCOMPLETE,
    )


def test_a_run_with_errored_cases_is_partial(partial_run_id: str) -> None:
    assert metrics_of(partial_run_id)["status"] == "partial"


def test_an_errored_case_leaves_the_denominator_and_is_counted_as_an_error(
    partial_run_id: str,
) -> None:
    metrics = metrics_of(partial_run_id)["metrics"]

    assert metrics["n_errors"] == 1
    assert metrics["n_completed"] == SMOKE_CASES
    assert metrics["n_pass_denominator"] == SMOKE_CASES - 1, "the errored case is excluded"
    assert metrics["n_passed"] == SMOKE_CASES - 1
    assert metrics["pass_rate"] == 1.0


def test_the_error_rate_is_over_total_attempted(partial_run_id: str) -> None:
    metrics = metrics_of(partial_run_id)["metrics"]
    assert metrics["error_rate"] == pytest.approx(1 / SMOKE_CASES)
    assert metrics["error_breakdown"] == {"server": 1}


def test_latency_excludes_the_errored_case(partial_run_id: str) -> None:
    latency = metrics_of(partial_run_id)["metrics"]["latency"]
    assert latency["n"] == SMOKE_CASES - 1


def test_an_errored_case_lowers_the_cost_coverage_rather_than_being_absorbed(
    partial_run_id: str,
) -> None:
    # The failed case produced no response, so it could not be priced. The
    # coverage ratio is where that shows, instead of the total quietly being
    # divided by six as though every case had a known cost.
    metrics = metrics_of(partial_run_id)["metrics"]
    assert metrics["cost_coverage"] == pytest.approx((SMOKE_CASES - 1) / SMOKE_CASES)
    assert metrics["cost"]["priced"] is False
    assert metrics["cost"]["unpriced_reason"] == "partial_coverage"


def test_the_fail_policy_counts_errored_cases_in_the_denominator(
    repo_root: Path, migrated_db: str
) -> None:
    del migrated_db
    failing = execute_run(
        repo_root,
        "--error-policy",
        "fail",
        "--provider-option",
        'fail_case_ids=["capital-exact"]',
        expected_exit=ExitCode.RUN_INCOMPLETE,
    )
    metrics = metrics_of(failing)["metrics"]

    assert metrics["error_policy"] == "fail"
    assert metrics["n_pass_denominator"] == SMOKE_CASES
    assert metrics["n_passed"] == SMOKE_CASES - 1
    assert metrics["pass_rate"] == pytest.approx((SMOKE_CASES - 1) / SMOKE_CASES)
    assert metrics["error_rate"] == pytest.approx(1 / SMOKE_CASES), (
        "the error rate is over total attempted under either policy"
    )


# --- the runs listing -----------------------------------------------------


def test_runs_lists_the_run_with_its_derived_figures(run_id: str) -> None:
    result = runner.invoke(app, ["runs", "--limit", "5", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    payload = json.loads(result.stdout)

    assert payload["limit"] == 5
    listed = {item["id"]: item for item in payload["runs"]}
    assert run_id in listed
    row = listed[run_id]
    assert row["pass_rate"] == 1.0
    assert row["p95_ms"] is not None
    assert row["total_cost"] is not None, "a completed run's rollup must reach the listing"


def test_the_runs_listing_reports_a_total_cost_not_a_per_case_cost(run_id: str) -> None:
    listing = runner.invoke(app, ["runs", "--json"])
    row = next(item for item in json.loads(listing.stdout)["runs"] if item["id"] == run_id)
    metrics = metrics_of(run_id)["metrics"]
    assert row["total_cost"] == metrics["cost"]["total_cost"]


def test_the_runs_listing_escapes_a_model_name(
    tmp_path: Path, migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model names are caller-supplied, and the runs table interpolates them.

    Rich drops an unresolvable style tag, so `fake-1[beta]` listed as `fake-1`:
    two different models rendered as the same row.

    `COLUMNS` is widened because Rich truncates a column to the terminal width,
    and a cell elided by width would make this pass whether or not the escaping
    works.
    """
    del migrated_db
    monkeypatch.setenv("COLUMNS", "300")
    suite = tmp_path / "one.yaml"
    suite.write_text(
        "schema_version: 1\n"
        "name: one\n"
        'version: "1"\n'
        "cases:\n"
        "  - id: only\n"
        "    input: Say alpha\n"
        "    expected: alpha\n"
        "    evaluators:\n"
        "      - type: exact_match\n",
        encoding="utf-8",
    )
    started = runner.invoke(
        app,
        [
            "run",
            str(suite),
            "--provider",
            "fake",
            "--model",
            "fake-1[beta]",
            "--fake-mode",
            "expected",
            "--json",
        ],
    )
    assert started.exit_code == ExitCode.SUCCESS, started.output

    listed = runner.invoke(app, ["runs"])
    assert listed.exit_code == ExitCode.SUCCESS, listed.output
    assert "fake-1[beta]" in listed.output


def test_the_runs_listing_pages(repo_root: Path, migrated_db: str) -> None:
    del migrated_db
    ids = [execute_run(repo_root, "--label", f"run-{index}") for index in range(3)]

    page = json.loads(
        runner.invoke(app, ["runs", "--limit", "2", "--offset", "2", "--json"]).stdout
    )
    assert page["total"] == len(ids)
    assert page["limit"] == 2
    assert page["offset"] == 2
    assert len(page["runs"]) == 1

    # Newest first, so the third page item is the oldest run.
    assert page["runs"][0]["id"] == ids[0]


# --- export escaping ------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    ["=1+1", "+1", "-1+1", "@SUM(A1)", "\tcmd", "\rcmd"],
)
def test_every_formula_trigger_is_neutralised(hostile: str) -> None:
    guarded = neutralise_formula(hostile)
    assert guarded.startswith("'")
    assert guarded[1:] == hostile, "the value is neutralised, not altered"


@pytest.mark.parametrize("benign", ["geography", "Q3 results", "0.5", "case-1", ""])
def test_a_benign_value_is_left_exactly_as_it_is(benign: str) -> None:
    assert neutralise_formula(benign) == benign


def test_a_csv_export_neutralises_a_formula_planted_in_a_benchmark(
    tmp_path: Path, migrated_db: str
) -> None:
    """A benchmark file is shared, untrusted data and `category` has no pattern.

    Left unescaped, the cell executes when the export is opened in a spreadsheet.
    """
    del migrated_db
    suite = tmp_path / "hostile.yaml"
    suite.write_text(
        "schema_version: 1\n"
        "name: hostile\n"
        'version: "1"\n'
        "cases:\n"
        "  - id: injected\n"
        '    category: \'=HYPERLINK("http://attacker.example","Q3 results")\'\n'
        "    input: Say alpha\n"
        "    expected: alpha\n"
        "    evaluators:\n"
        "      - type: exact_match\n",
        encoding="utf-8",
    )
    started = runner.invoke(
        app,
        [
            "run",
            str(suite),
            "--provider",
            "fake",
            "--model",
            "fake-1",
            "--fake-mode",
            "expected",
            "--json",
        ],
    )
    assert started.exit_code == ExitCode.SUCCESS, started.output
    injected_run = json.loads(started.stdout)["run_id"]

    exported = runner.invoke(app, ["export", injected_run, "--format", "csv"])
    assert exported.exit_code == ExitCode.SUCCESS, exported.output

    header = exported.stdout.splitlines()[0].split(",")
    cells = next(csv.reader(exported.stdout.splitlines()[1:]))
    category = cells[header.index("category")]

    assert not category.startswith("="), category
    assert category.startswith("'="), "the standard mitigation is a leading apostrophe"
    assert "HYPERLINK" in category, "the value is neutralised, not destroyed"


# --- models ----------------------------------------------------------------


def test_models_lists_the_fake_provider_model_and_every_price_entry() -> None:
    """The fake entry is present, and every shipped price-table entry is listed.

    Not asserted as equality against a fixed table: the shipped
    `pricing/data/prices.yaml` legitimately grows over time (new model
    snapshots, new vendor entries), and a test pinned to today's exact
    contents breaks on every such addition without that being a defect in
    the code under test. What the CLI promises is that no priced model goes
    unlisted, which is what this checks directly against the real table.
    """
    result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    payload = json.loads(result.stdout)

    rows = payload["models"]
    listed = {(row["provider"], row["model"]) for row in rows}
    assert ("fake", "fake-1") in listed
    assert "fake-1" in {row["model"] for row in rows if row["source"] == "provider"}

    price_table_rows = {
        (row["provider"], row["model"]) for row in rows if row["source"] == "price_table"
    }
    assert ("fake", "fake-") in price_table_rows

    every_priced_model = {(entry.provider, entry.model) for entry in load_price_table().models}
    assert every_priced_model <= listed, f"missing from the listing: {every_priced_model - listed}"
    assert payload["price_table_id"] == "builtin"


def test_models_reports_prices_and_availability() -> None:
    rows = json.loads(runner.invoke(app, ["models", "--json"]).stdout)["models"]
    fake = next(row for row in rows if row["model"] == "fake-1")
    assert fake["available"] is True
    assert fake["input_per_mtok"] == "0"
    assert fake["credential_env"] is None


def test_models_never_prints_a_credential_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EVAL_CANARY_KEY", CANARY)
    result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == ExitCode.SUCCESS
    assert CANARY not in result.output


# --- config show -----------------------------------------------------------


def test_config_show_prints_the_effective_configuration() -> None:
    result = runner.invoke(app, ["config", "show", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    payload = json.loads(result.stdout)

    assert payload["precedence"] == ["cli", "environment", "config-file", "default"]
    settings = {row["setting"]: row for row in payload["settings"]}
    assert settings["concurrency"]["value"] == 8
    assert settings["log_level"]["env_var"] == "LLM_EVAL_LOG_LEVEL"


def test_config_show_never_prints_a_credential_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EVAL_CANARY_KEY", CANARY)
    monkeypatch.setenv("LLM_EVAL_API_TOKEN", CANARY)
    get_settings.cache_clear()

    for argv in (["config", "show"], ["config", "show", "--json"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == ExitCode.SUCCESS, result.output
        assert CANARY not in result.output, argv
        # Not even a fragment: a partial mask is still a disclosure.
        assert "sk-canary" not in result.output, argv
        assert "0123456789" not in result.output, argv


def test_config_show_never_prints_a_password_inside_the_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A credential also hides in a connection string, where key-name matching never looks.

    `Settings.database_url` is a plain `str`, so the secret-key scrubber cannot
    see it. A PostgreSQL DSN carries its password in the userinfo and this
    command printed the value three times: the top-level field, the settings row
    and the human table.
    """
    monkeypatch.setenv("LLM_EVAL_DATABASE_URL", DSN_WITH_CANARY)
    get_settings.cache_clear()

    for argv in (["config", "show"], ["config", "show", "--json"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == ExitCode.SUCCESS, result.output
        assert CANARY not in result.output, argv
        assert "sk-canary" not in result.output, argv
        assert "0123456789" not in result.output, argv

    payload = json.loads(runner.invoke(app, ["config", "show", "--json"]).stdout)
    row = next(item for item in payload["settings"] if item["setting"] == "database_url")
    # Redacted, not deleted: everything but the password survives, because that
    # is what makes a printed configuration worth printing.
    assert row["value"] == REDACTED_DSN
    assert payload["database_url"] == REDACTED_DSN


def test_the_human_form_shows_the_redaction_marker_rather_than_swallowing_it() -> None:
    """Rich parses `[redacted]` as a style tag and drops what it cannot resolve.

    Unescaped, the marker vanished and the output read
    `postgresql://admin:@host/db`, which says "this database has an empty
    password" rather than "a password was removed here". The value was safe
    either way; the statement was wrong.
    """
    result = runner.invoke(app, ["--db-url", DSN_WITH_CANARY, "config", "show"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    assert CANARY not in result.output
    assert "[redacted]" in result.output
    assert "admin:@db.example.test" not in result.output


def test_a_password_passed_as_a_cli_flag_is_redacted_too() -> None:
    result = runner.invoke(app, ["--db-url", DSN_WITH_CANARY, "config", "show", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    assert CANARY not in result.output
    assert json.loads(result.stdout)["database_url"] == REDACTED_DSN


def test_db_upgrade_escapes_the_url_it_echoes(tmp_path: Path) -> None:
    """Redacting was not enough on its own: Rich also has to be told not to parse it.

    `db upgrade` echoes the URL back so an operator can confirm which database
    was touched, and that line interpolates it into Rich markup. Rich drops a
    style tag it cannot resolve, so an unescaped URL loses every bracketed run -
    which is how the redaction marker itself disappeared and made a removed
    password read as an empty one.

    A bracketed directory name exercises the same line on the success path,
    where a real database is genuinely created. The name is `db[beta]` rather
    than `db[1]` on purpose: Rich only consumes a bracket run that could be a
    style name, so `[1]` survives unescaped and would make this test pass
    whether or not the fix is present.
    """
    bracketed = tmp_path / "db[beta]"
    bracketed.mkdir()
    url = f"sqlite+aiosqlite:///{bracketed / 'lab.db'}"

    result = runner.invoke(app, ["--db-url", url, "db", "upgrade"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    assert "db[beta]" in result.output, "Rich swallowed the bracketed path segment"


def test_db_upgrade_redacts_a_secret_query_parameter(tmp_path: Path) -> None:
    """The same line, with a real credential in it, on a database that really opens."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'lab.db'}?password={CANARY}"

    result = runner.invoke(app, ["--db-url", url, "db", "upgrade"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    assert CANARY not in result.output
    assert "password=***" in result.output, "the redaction must survive Rich's markup parser"


def test_db_commands_never_print_a_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PostgreSQL DSN reaches no driver here, so this pins the failure path too.

    The command cannot connect (asyncpg is an optional extra), so it exits
    non-zero. Neither the mapped error nor the traceback may carry the password.
    """
    monkeypatch.setenv("LLM_EVAL_DATABASE_URL", DSN_WITH_CANARY)
    get_settings.cache_clear()

    for argv in (["db", "upgrade"], ["db", "revision", "--json"]):
        result = runner.invoke(app, argv)
        assert CANARY not in result.output, argv
        assert "sk-canary" not in result.output, argv


def test_the_db_line_renders_the_redaction_marker_visibly() -> None:
    """The userinfo half of the composition, rendered through a real Rich console.

    A SQLite URL cannot carry userinfo and SQLAlchemy refuses one, so no working
    database can drive `db upgrade` down the `[redacted]` branch. What that line
    actually does - redact, then escape - is asserted here instead, against the
    same Rich renderer the command writes to.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, width=200)
    console.print(f"database at [bold]{markup_safe(redact_url(DSN_WITH_CANARY))}[/bold]")

    rendered = buffer.getvalue()
    assert CANARY not in rendered
    assert "[redacted]" in rendered, "the marker must reach the reader, not be parsed away"
    assert "admin:@db.example.test" not in rendered


def test_a_sqlite_url_is_not_mangled_by_the_redactor(db_url: str) -> None:
    # Redaction must be a no-op for the URL people actually read.
    payload = json.loads(runner.invoke(app, ["config", "show", "--json"]).stdout)
    assert payload["database_url"] == db_url


# --- precedence -----------------------------------------------------------


def test_the_environment_beats_a_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pydantic-settings ranks init arguments above the environment.

    Passing the config file through as init arguments therefore inverted two of
    the four documented layers: a file value silently beat `LLM_EVAL_*` while
    `config show` printed the opposite order.
    """
    config = tmp_path / "settings.yaml"
    config.write_text("log_level: INFO\n", encoding="utf-8")
    monkeypatch.setenv("LLM_EVAL_LOG_LEVEL", "DEBUG")
    get_settings.cache_clear()

    payload = json.loads(
        runner.invoke(app, ["--config", str(config), "config", "show", "--json"]).stdout
    )
    row = next(item for item in payload["settings"] if item["setting"] == "log_level")
    assert row["value"] == "DEBUG"
    assert row["source"] == "environment"


def test_a_cli_flag_beats_both_the_environment_and_a_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("log_level: INFO\n", encoding="utf-8")
    monkeypatch.setenv("LLM_EVAL_LOG_LEVEL", "DEBUG")
    get_settings.cache_clear()

    payload = json.loads(
        runner.invoke(
            app,
            ["--config", str(config), "--log-level", "warning", "config", "show", "--json"],
        ).stdout
    )
    row = next(item for item in payload["settings"] if item["setting"] == "log_level")
    assert row["value"] == "WARNING"
    assert row["source"] == "cli"


def test_a_config_file_beats_a_default(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("concurrency: 3\n", encoding="utf-8")
    payload = json.loads(
        runner.invoke(app, ["--config", str(config), "config", "show", "--json"]).stdout
    )
    row = next(item for item in payload["settings"] if item["setting"] == "concurrency")
    assert row["value"] == 3
    assert row["source"] == "config-file"


def test_an_untouched_field_reports_the_default_as_its_source() -> None:
    payload = json.loads(runner.invoke(app, ["config", "show", "--json"]).stdout)
    row = next(item for item in payload["settings"] if item["setting"] == "judge_concurrency")
    assert row["source"] == "default"
    assert row["value"] == 4


def test_the_printed_precedence_matches_the_order_actually_applied() -> None:
    payload = json.loads(runner.invoke(app, ["config", "show", "--json"]).stdout)
    assert payload["precedence"] == ["cli", "environment", "config-file", "default"]


def test_config_show_reports_a_secret_as_a_variable_name_and_a_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EVAL_API_TOKEN", CANARY)
    get_settings.cache_clear()

    payload = json.loads(runner.invoke(app, ["config", "show", "--json"]).stdout)
    api_token = next(row for row in payload["settings"] if row["setting"] == "api_token")
    assert api_token["is_secret"] is True
    assert api_token["value"] is None
    assert api_token["set"] is True
    assert api_token["env_var"] == "LLM_EVAL_API_TOKEN"


def test_config_show_reports_a_cli_flag_as_its_source(tmp_path: Path) -> None:
    alternate = f"sqlite+aiosqlite:///{tmp_path / 'alt.db'}"
    result = runner.invoke(app, ["--db-url", alternate, "config", "show", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output

    payload = json.loads(result.stdout)
    row = next(item for item in payload["settings"] if item["setting"] == "database_url")
    assert row["source"] == "cli"
    assert payload["database_url"] == alternate


def test_config_show_reports_a_config_file_as_its_source(tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("concurrency: 3\n", encoding="utf-8")
    result = runner.invoke(app, ["--config", str(config), "config", "show", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output

    payload = json.loads(result.stdout)
    row = next(item for item in payload["settings"] if item["setting"] == "concurrency")
    assert row["source"] == "config-file"
    assert row["value"] == 3


# --- pricing ---------------------------------------------------------------


def test_pricing_show_reports_the_table_and_its_content_hash() -> None:
    result = runner.invoke(app, ["pricing", "show", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output

    table = json.loads(result.stdout)["price_table"]
    assert table["id"] == "builtin"
    assert table["content_hash"].startswith("sha256:")
    assert table["models"]


def test_pricing_validate_accepts_the_shipped_table() -> None:
    result = runner.invoke(app, ["pricing", "validate", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["errors"] == []


def test_pricing_validate_refuses_a_reused_id_and_version_with_new_content(
    tmp_path: Path,
) -> None:
    """A conflicting file reuses the SHIPPED table's exact `id` and `version`.

    Read off the real shipped table rather than hardcoded, so this test keeps
    testing the conflict it names rather than silently degrading into "any
    file validates" the next time `pricing/data/prices.yaml`'s `version` is
    bumped for an unrelated reason - which is exactly what happened to the
    hardcoded date this replaces.
    """
    shipped = load_price_table()
    conflicting = tmp_path / "prices.yaml"
    conflicting.write_text(
        f"id: {shipped.id}\n"
        f'version: "{shipped.version}"\n'
        "currency: USD\n"
        "models:\n"
        "  - provider: fake\n"
        "    model: fake-\n"
        "    match: prefix\n"
        '    input_per_mtok: "99"\n'
        '    output_per_mtok: "99"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["pricing", "validate", "--prices", str(conflicting), "--json"])

    assert result.exit_code == ExitCode.VALIDATION_FAILED, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("already used" in message for message in payload["errors"])


def test_pricing_validate_refuses_a_duplicate_entry(tmp_path: Path) -> None:
    duplicated = tmp_path / "prices.yaml"
    duplicated.write_text(
        "id: local\n"
        'version: "1"\n'
        "currency: USD\n"
        "models:\n"
        "  - provider: fake\n"
        "    model: fake-1\n"
        '    input_per_mtok: "1"\n'
        '    output_per_mtok: "2"\n'
        "  - provider: fake\n"
        "    model: fake-1\n"
        '    input_per_mtok: "3"\n'
        '    output_per_mtok: "4"\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["pricing", "validate", "--prices", str(duplicated), "--json"])

    assert result.exit_code == ExitCode.VALIDATION_FAILED, result.output
    assert any("declared 2 times" in item for item in json.loads(result.stdout)["errors"])


def test_pricing_validate_exits_three_on_an_unreadable_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["pricing", "validate", "--prices", str(tmp_path / "missing.yaml"), "--json"]
    )
    assert result.exit_code == ExitCode.VALIDATION_FAILED
    assert json.loads(result.stdout)["ok"] is False


# --- export ----------------------------------------------------------------


def test_export_json_carries_one_row_per_case(run_id: str) -> None:
    result = runner.invoke(app, ["export", run_id, "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output

    payload = json.loads(result.stdout)
    assert payload["n_cases"] == SMOKE_CASES
    assert len(payload["cases"]) == SMOKE_CASES
    assert "metrics" in payload
    assert {row["case_id"] for row in payload["cases"]}


def test_export_csv_writes_a_header_and_a_row_per_case(run_id: str) -> None:
    result = runner.invoke(app, ["export", run_id, "--format", "csv"])
    assert result.exit_code == ExitCode.SUCCESS, result.output

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == SMOKE_CASES + 1
    assert lines[0].startswith("run_id,case_id,")
    assert "\r" not in result.stdout


def test_export_csv_writes_an_empty_cell_for_an_unknown_value(run_id: str) -> None:
    result = runner.invoke(app, ["export", run_id, "--format", "csv"])
    header = result.stdout.splitlines()[0].split(",")
    row = result.stdout.splitlines()[1].split(",")
    judge_cost = row[header.index("judge_cost")]
    assert judge_cost == "", "an unknown cost must be an empty cell, never a zero"


def test_export_writes_to_a_file_when_asked(run_id: str, tmp_path: Path) -> None:
    target = tmp_path / "cases.csv"
    result = runner.invoke(app, ["export", run_id, "--format", "csv", "--output", str(target)])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    assert target.read_text(encoding="utf-8").count("\n") == SMOKE_CASES + 1


def test_export_rejects_an_unknown_format(run_id: str) -> None:
    result = runner.invoke(app, ["export", run_id, "--format", "parquet"])
    assert result.exit_code == ExitCode.USAGE


def test_export_on_an_unknown_run_exits_with_a_usage_code(migrated_db: str) -> None:
    del migrated_db
    result = runner.invoke(app, ["export", "no-such-run", "--json"])
    assert result.exit_code == ExitCode.USAGE


# --- no schema drift -------------------------------------------------------


def test_the_metrics_row_exists_without_a_new_migration(run_id: str, db_url: str) -> None:
    """Phase 3 ships no migration: the table was created by the initial one."""
    import sqlite3  # noqa: PLC0415 - a direct schema check, not a storage dependency

    path = db_url.removeprefix("sqlite+aiosqlite:///")
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "select count(*) from run_metrics where run_id = ?", (run_id,)
        ).fetchone()
    assert rows[0] == 1, "exactly one rollup row per run, by construction"
