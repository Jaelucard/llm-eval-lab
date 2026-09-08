"""The CLI surface: exit codes, the JSON envelope, and the dry-run guarantee."""

import json
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_eval_lab.cli.main import ExitCode, app
from llm_eval_lab.cli.output import json_default
from llm_eval_lab.models import ProviderConfig, ProviderError
from llm_eval_lab.settings import get_settings

runner = CliRunner()

SMOKE_CASES = 6


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, db_url: str) -> None:
    """Point every CLI invocation at a test-private database.

    `get_settings` caches, so the cache is cleared around each test; otherwise
    the first test in a session would fix the database location for all of them.
    """
    monkeypatch.setenv("LLM_EVAL_DATABASE_URL", db_url)
    get_settings.cache_clear()


@pytest.mark.integration
def test_validate_accepts_the_smoke_suite(repo_root: Path) -> None:
    result = runner.invoke(
        app, ["validate", str(repo_root / "examples" / "benchmarks" / "smoke.yaml"), "--json"]
    )
    assert result.exit_code == ExitCode.SUCCESS
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["n_cases"] == SMOKE_CASES


@pytest.mark.integration
def test_validate_exits_three_and_names_the_field(fixtures_dir: Path) -> None:
    path = fixtures_dir / "suites" / "malformed_missing_id.yaml"
    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == ExitCode.VALIDATION_FAILED
    combined = result.output
    assert "cases[0].id" in combined
    assert str(path) in combined


@pytest.mark.integration
def test_human_validation_output_keeps_the_failing_case_id(fixtures_dir: Path) -> None:
    """Rich parses `[case has-neither]` as a style tag unless the text is escaped.

    Without the escape the case id - the single most useful thing in the message
    - is silently eaten, and only in the human path, so `--json` looks fine.
    """
    path = fixtures_dir / "suites" / "malformed_missing_id.yaml"
    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == ExitCode.VALIDATION_FAILED
    assert "[case has-neither]" in result.output
    assert "[case has-unknown-field]" in result.output


@pytest.mark.integration
def test_validate_json_lists_every_field_error(fixtures_dir: Path) -> None:
    path = fixtures_dir / "suites" / "malformed_bad_evaluator.yaml"
    result = runner.invoke(app, ["validate", str(path), "--json"])

    assert result.exit_code == ExitCode.VALIDATION_FAILED
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    locations = {item["location"] for item in payload["errors"]}
    assert "cases[0].evaluators[0].type" in locations
    assert "cases[1].evaluators[0].params.pattern" in locations


@pytest.mark.integration
def test_dry_run_issues_zero_provider_calls_and_writes_no_run(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    migrated_db: str,
) -> None:
    from llm_eval_lab.providers.registry import (  # noqa: PLC0415 - reads next to its use
        ProviderRegistry,
    )

    def _explode(_self: ProviderRegistry, config: ProviderConfig) -> None:
        msg = f"a dry run must never construct a provider (asked for {config.provider})"
        raise ProviderError(msg)

    monkeypatch.setattr(ProviderRegistry, "create", _explode)

    result = runner.invoke(
        app,
        [
            "run",
            str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
            "--provider",
            "fake",
            "--model",
            "fake-1",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == ExitCode.SUCCESS, result.output
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["n_cases"] == SMOKE_CASES
    assert payload["config"]["provider"]["model"] == "fake-1"
    assert "estimated_cost" in payload
    assert payload["assumptions"]

    del migrated_db
    with sqlite3.connect(tmp_path / "test.db") as connection:
        assert connection.execute("select count(*) from runs").fetchone()[0] == 0


@pytest.mark.integration
def test_dry_run_prints_the_effective_configuration_for_a_human(repo_root: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
            "--provider",
            "fake",
            "--model",
            "fake-1",
            "--dry-run",
        ],
    )
    assert result.exit_code == ExitCode.SUCCESS
    assert "effective configuration" in result.output
    assert "cases selected" in result.output
    assert "estimated cost" in result.output


@pytest.mark.integration
def test_a_secret_shaped_provider_option_is_a_clean_validation_failure(
    repo_root: Path, migrated_db: str
) -> None:
    """A `--provider-option` with a secret-shaped key must not crash the CLI.

    Before this fix, `ProviderConfig`'s own rejection of the option surfaced as
    an unhandled `pydantic.ValidationError` - a raw multi-frame traceback and
    exit code 1 ("bug"), rather than the same clean exit-3 usage failure every
    other malformed-input case in this file gets. The planted value must never
    appear anywhere in the output, on top of everything else.
    """
    del migrated_db
    result = runner.invoke(
        app,
        [
            "run",
            str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
            "--provider",
            "fake",
            "--model",
            "fake-1",
            "--provider-option",
            'api_key="sk-canary-leak-me-not"',
        ],
    )

    assert result.exit_code == ExitCode.VALIDATION_FAILED, result.output
    assert "Traceback" not in result.output
    assert "sk-canary-leak-me-not" not in result.output
    assert "options" in result.output
    assert "api_key" in result.output


@pytest.mark.integration
def test_a_secret_shaped_provider_option_is_a_clean_validation_failure_as_json(
    repo_root: Path, migrated_db: str
) -> None:
    del migrated_db
    result = runner.invoke(
        app,
        [
            "run",
            str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
            "--provider",
            "fake",
            "--model",
            "fake-1",
            "--provider-option",
            'password="hunter2secret"',
            "--json",
        ],
    )

    assert result.exit_code == ExitCode.VALIDATION_FAILED, result.output
    assert "Traceback" not in result.output
    assert "hunter2secret" not in result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "validation_failed"
    assert any("password" in item["message"] for item in payload["errors"])
    assert all(item["input"] is None for item in payload["errors"])


@pytest.mark.integration
def test_db_upgrade_reports_the_revision(db_url: str) -> None:
    result = runner.invoke(app, ["db", "upgrade", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    payload = json.loads(result.stdout)
    assert payload["revision"] == "0001"
    assert payload["database_url"] == db_url


@pytest.mark.integration
def test_db_url_option_targets_a_second_database(tmp_path: Path) -> None:
    alternate = tmp_path / "alt.db"
    result = runner.invoke(
        app, ["--db-url", f"sqlite+aiosqlite:///{alternate}", "db", "upgrade", "--json"]
    )
    assert result.exit_code == ExitCode.SUCCESS, result.output
    assert alternate.exists()
    assert not (tmp_path / "test.db").exists(), "the default database must be untouched"


@pytest.mark.integration
def test_providers_and_evaluators_are_listable() -> None:
    providers = runner.invoke(app, ["providers", "--json"])
    evaluators = runner.invoke(app, ["evaluators", "--json"])

    assert providers.exit_code == ExitCode.SUCCESS
    assert evaluators.exit_code == ExitCode.SUCCESS
    assert "fake" in {item["name"] for item in json.loads(providers.stdout)["providers"]}
    assert {"exact_match", "exact_match_ci", "contains", "regex", "numeric_tolerance"} <= {
        item["type"] for item in json.loads(evaluators.stdout)["evaluators"]
    }


@pytest.mark.integration
def test_providers_reports_sdk_and_credential_status_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every extra installed and no credential set, the two facts diverge.

    `openai`, `anthropic` and `google` all have their optional extra installed
    in this environment, so their SDK is genuinely importable; none of their
    credential variables are set here. `sdk_available` must say `true` for all
    three and `credential_present` must say `false` - collapsing the two into
    one `available` flag would make that distinction unreadable. `fake` needs
    neither, so both are `true` for it.
    """
    for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    result = runner.invoke(app, ["providers", "--json"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    by_name = {item["name"]: item for item in json.loads(result.stdout)["providers"]}

    for name in ("openai", "anthropic", "google"):
        row = by_name[name]
        assert row["sdk_available"] is True, name
        assert row["credential_present"] is False, name
        assert row["credential_env"], name
        assert row["required_extra"], name

    fake = by_name["fake"]
    assert fake["sdk_available"] is True
    assert fake["credential_present"] is True
    assert fake["credential_env"] is None
    assert fake["required_extra"] is None


@pytest.mark.integration
def test_show_on_an_unknown_run_exits_with_a_usage_code(migrated_db: str) -> None:
    del migrated_db
    result = runner.invoke(app, ["show", "no-such-run", "--json"])
    assert result.exit_code == ExitCode.USAGE
    assert json.loads(result.stdout)["error"] == "not_found"


@pytest.mark.integration
def test_an_invalid_max_cost_is_a_usage_error(repo_root: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
            "--provider",
            "fake",
            "--model",
            "fake-1",
            "--max-cost",
            "not-a-number",
            "--dry-run",
        ],
    )
    assert result.exit_code == ExitCode.USAGE


@pytest.mark.integration
def test_a_budget_triggered_cancellation_exits_run_incomplete_not_interrupted(
    fixtures_dir: Path, migrated_db: str
) -> None:
    """`--max-cost` cancels the run with no signal ever delivered to this process.

    Before this fix, a budget-triggered cancellation shared exit 130 with a real
    `SIGINT` - a CI script keying off "130 means the operator hit Ctrl-C" would
    be surprised to see it from an automatic budget cap. It must exit 5, the
    same "did not get to finish, not the run's fault" code `PARTIAL` and
    `FAILED` already use.
    """
    del migrated_db
    result = runner.invoke(
        app,
        [
            "run",
            str(fixtures_dir / "suites" / "smoke.yaml"),
            "--provider",
            "fake",
            "--model",
            "fake-budget",  # priced at 1.00 USD/token by the test price table
            "--provider-option",
            'mode="expected"',
            "--prices",
            str(fixtures_dir / "pricing" / "test_prices.yaml"),
            "--concurrency",
            "1",
            "--max-cost",
            "0.000001",  # far below the cost of even one case
            "--json",
        ],
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "cancelled", payload
    assert payload["error"] == "budget_exceeded", payload
    assert result.exit_code == ExitCode.RUN_INCOMPLETE, result.output


@pytest.mark.integration
def test_json_timestamps_are_rfc_3339(repo_root: Path, migrated_db: str) -> None:
    """`str(datetime)` uses a space where RFC 3339 requires a `T`.

    Python parses both; browsers and most other parsers do not, and this
    envelope is what the HTTP API and the dashboard consume.

    The UTC offset is spelled `Z`, and that spelling is checked against a
    timestamp nested inside a dumped model rather than only in isolation.
    Pydantic emits `Z` for every nested timestamp, so a top-level `+00:00`
    would put two spellings of the same instant in one document and leave a
    consumer comparing them as strings to find them unequal.
    """
    del migrated_db
    run = runner.invoke(
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
        ],
    )
    assert run.exit_code == ExitCode.SUCCESS, run.output
    run_id = json.loads(run.stdout)["run_id"]

    shown = runner.invoke(app, ["show", run_id, "--json"])
    assert shown.exit_code == ExitCode.SUCCESS
    created_at = json.loads(shown.stdout)["created_at"]

    assert "T" in created_at, created_at
    assert " " not in created_at, created_at
    assert created_at.endswith("Z"), created_at
    assert "+00:00" not in created_at, created_at
    assert datetime.fromisoformat(created_at).tzinfo is not None

    # `metrics.computed_at` is rendered by pydantic's model dump; `created_at`
    # above is rendered by the CLI's own encoder hook. The two paths must agree,
    # which is the whole point of normalising the hook.
    reported = runner.invoke(app, ["metrics", run_id, "--json"])
    assert reported.exit_code == ExitCode.SUCCESS, reported.output
    computed_at = json.loads(reported.stdout)["metrics"]["computed_at"]
    assert computed_at.endswith("Z"), computed_at
    assert "+00:00" not in computed_at, computed_at
    assert datetime.fromisoformat(computed_at).tzinfo is not None


@pytest.mark.integration
def test_the_json_encoder_refuses_an_unknown_type() -> None:
    """A blanket `str` would silently ship a `repr` to a consumer; this must be loud."""

    class Unknown:
        pass

    with pytest.raises(TypeError, match="not JSON-serializable"):
        json_default(Unknown())


@pytest.mark.integration
def test_the_json_encoder_keeps_decimals_exact() -> None:
    assert json_default(Decimal("0.0000012345")) == "0.0000012345"
