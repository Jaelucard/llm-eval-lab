"""The headline claim, proved end to end: a real regression is detected.

Two real runs against the deterministic fake provider, compared through the
installed entry point as a process, because the exit code is the thing under
test and it is only observable from outside. A pipeline keys off exit 4; a
comparison that reported the regression correctly in its JSON and still exited 0
would be worthless, and only a subprocess can tell the two apart.

The degradation is deterministic. The fake provider seeds its own generator from
the request content, so `--fake-mode mutate --mutate-rate 0.2` degrades the same
case on every machine and every rerun, which is what lets these tests assert the
case ids by name rather than asserting that "something" broke.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from llm_eval_lab.models import RegressionReport

RUN_TIMEOUT_S = 180
SMOKE_CASES = 6
MUTATE_RATE = "0.2"

REGRESSION_EXIT_CODE = 4
"""``ExitCode.REGRESSION``. Spelled out here so the test pins the contract."""

INCOMPARABLE_EXIT_CODE = 6
"""``ExitCode.INCOMPARABLE``."""

USAGE_EXIT_CODE = 2
"""``ExitCode.USAGE``."""


def _cli(
    repo_root: Path,
    db_url: str,
    *args: str,
    timeout: int = RUN_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    """Invoke the installed CLI as a subprocess with an isolated database."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-controlled input
        [sys.executable, "-m", "llm_eval_lab.cli", *args],
        cwd=repo_root,
        env={**os.environ, "LLM_EVAL_DATABASE_URL": db_url},
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _run(repo_root: Path, db_url: str, *extra: str, label: str) -> str:
    """Execute the smoke suite once and return the run id."""
    completed = _cli(
        repo_root,
        db_url,
        "run",
        str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
        "--provider",
        "fake",
        "--model",
        "fake-1",
        "--label",
        label,
        *extra,
        "--json",
    )
    assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
    payload: dict[str, Any] = json.loads(completed.stdout)
    assert payload["n_cases"] == SMOKE_CASES
    run_id: str = payload["run_id"]
    return run_id


@pytest.fixture
def thresholds_path(repo_root: Path) -> Path:
    """Return the CI threshold policy the documented pipeline recipe names."""
    return repo_root / "examples" / "thresholds" / "ci-thresholds.yaml"


@pytest.fixture
def clean_run(repo_root: Path, migrated_db: str) -> str:
    """An all-passing baseline: the fake provider returns each case's own answer."""
    return _run(repo_root, migrated_db, "--fake-mode", "expected", label="clean")


@pytest.fixture
def degraded_run(repo_root: Path, migrated_db: str) -> str:
    """A deliberately degraded candidate, degraded by a fixed fraction of cases."""
    return _run(
        repo_root,
        migrated_db,
        "--fake-mode",
        "mutate",
        "--mutate-rate",
        MUTATE_RATE,
        label="degraded",
    )


def _compare(
    repo_root: Path,
    db_url: str,
    baseline: str,
    candidate: str,
    thresholds: Path,
    *extra: str,
) -> tuple[int, RegressionReport, str]:
    """Compare two runs and parse the report the command wrote to stdout."""
    completed = _cli(
        repo_root,
        db_url,
        "compare",
        baseline,
        candidate,
        "--thresholds",
        str(thresholds),
        *extra,
        "--json",
    )
    report = RegressionReport.model_validate_json(completed.stdout)
    return completed.returncode, report, completed.stderr


@pytest.mark.e2e
def test_an_intentional_regression_exits_four_and_names_the_failing_cases(
    repo_root: Path,
    migrated_db: str,
    thresholds_path: Path,
    clean_run: str,
    degraded_run: str,
) -> None:
    code, report, _ = _compare(repo_root, migrated_db, clean_run, degraded_run, thresholds_path)

    assert code == REGRESSION_EXIT_CODE
    assert report.verdict.value == "fail"
    assert report.mode == "paired"
    assert report.paired_case_count == SMOKE_CASES
    assert report.comparable is True
    assert report.suite_hash_match is True
    # The degraded cases are named, not merely counted.
    assert report.summary.newly_failing
    assert report.summary.newly_passing == ()
    assert report.summary.n_flipped == len(report.summary.newly_failing)
    # And the overall pass-rate gate is what failed, on the point estimate.
    overall = next(check for check in report.checks if check.metric == "pass_rate")
    assert overall.status.value == "failed"
    assert overall.violated_bound == "max_absolute_decrease"
    assert overall.candidate is not None
    assert overall.baseline is not None
    assert overall.candidate < overall.baseline


@pytest.mark.e2e
def test_the_named_failing_cases_are_exactly_the_ones_that_stopped_passing(
    repo_root: Path,
    migrated_db: str,
    thresholds_path: Path,
    clean_run: str,
    degraded_run: str,
) -> None:
    _, report, _ = _compare(repo_root, migrated_db, clean_run, degraded_run, thresholds_path)

    exported = _cli(repo_root, migrated_db, "export", degraded_run, "--json")
    assert exported.returncode == 0, exported.stderr
    cases = json.loads(exported.stdout)["cases"]
    actually_failing = {case["case_id"] for case in cases if case["passed"] is False}

    assert set(report.summary.newly_failing) == actually_failing


@pytest.mark.e2e
def test_the_reverse_comparison_passes_and_names_the_recovered_cases(
    repo_root: Path,
    migrated_db: str,
    thresholds_path: Path,
    clean_run: str,
    degraded_run: str,
) -> None:
    code, report, _ = _compare(repo_root, migrated_db, degraded_run, clean_run, thresholds_path)

    assert code == 0
    assert report.verdict.value == "pass"
    assert report.summary.newly_passing
    assert report.summary.newly_failing == ()


@pytest.mark.e2e
def test_identical_runs_compare_clean(
    repo_root: Path,
    migrated_db: str,
    thresholds_path: Path,
    clean_run: str,
) -> None:
    code, report, _ = _compare(repo_root, migrated_db, clean_run, clean_run, thresholds_path)

    assert code == 0
    assert report.verdict.value == "pass"
    assert report.summary.n_flipped == 0
    assert report.changed_cases == ()
    assert not [check for check in report.checks if check.status.value in ("failed", "warning")]


@pytest.mark.e2e
def test_compare_json_writes_only_the_report_to_stdout(
    repo_root: Path,
    migrated_db: str,
    thresholds_path: Path,
    clean_run: str,
    degraded_run: str,
) -> None:
    completed = _cli(
        repo_root,
        migrated_db,
        "compare",
        clean_run,
        degraded_run,
        "--thresholds",
        str(thresholds_path),
        "--json",
    )

    # Exactly one document, and it is the report itself rather than the command
    # envelope, so `compare ... --json > report.json` is directly consumable.
    document = json.loads(completed.stdout)
    assert document["schema_version"] == 1
    assert document["mode"] == "paired"
    assert set(document) >= {"verdict", "checks", "summary", "thresholds_id"}
    assert "command" not in document
    # Every human-facing line went to stderr.
    assert "verdict" in completed.stderr.lower()


@pytest.mark.e2e
def test_an_unknown_run_id_is_a_usage_error(
    repo_root: Path,
    migrated_db: str,
    thresholds_path: Path,
    clean_run: str,
) -> None:
    completed = _cli(
        repo_root,
        migrated_db,
        "compare",
        clean_run,
        "not-a-run",
        "--thresholds",
        str(thresholds_path),
    )

    assert completed.returncode == USAGE_EXIT_CODE
    assert "not-a-run" in completed.stderr


@pytest.mark.e2e
def test_runs_of_different_suites_are_incomparable(
    repo_root: Path,
    migrated_db: str,
    thresholds_path: Path,
    fixtures_dir: Path,
    clean_run: str,
) -> None:
    other = _cli(
        repo_root,
        migrated_db,
        "run",
        str(fixtures_dir / "suites" / "minimal.yaml"),
        "--provider",
        "fake",
        "--model",
        "fake-1",
        "--fake-mode",
        "expected",
        "--json",
    )
    assert other.returncode == 0, other.stderr
    other_id = json.loads(other.stdout)["run_id"]

    code, report, stderr = _compare(repo_root, migrated_db, clean_run, other_id, thresholds_path)

    assert code == INCOMPARABLE_EXIT_CODE
    assert report.verdict.value == "incomparable"
    assert report.comparable is False
    assert report.suite_hash_match is False
    assert report.checks == ()
    assert "different benchmark suites" in stderr


@pytest.mark.e2e
def test_the_markdown_report_carries_everything_a_reviewer_needs(  # noqa: PLR0913, PLR0917 - one pytest fixture per parameter
    repo_root: Path,
    migrated_db: str,
    tmp_path: Path,
    thresholds_path: Path,
    clean_run: str,
    degraded_run: str,
) -> None:
    output = tmp_path / "report.md"
    completed = _cli(
        repo_root,
        migrated_db,
        "report",
        degraded_run,
        "--format",
        "md",
        "--compare-to",
        clean_run,
        "--thresholds",
        str(thresholds_path),
        "--out",
        str(output),
    )
    assert completed.returncode == 0, completed.stderr

    rendered = output.read_text(encoding="utf-8")
    assert "**Verdict: FAIL**" in rendered
    assert f"Paired cases: **{SMOKE_CASES}**" in rendered
    assert "### Newly failing" in rendered
    # One row per check, carrying baseline, candidate, delta and threshold.
    header = "| check | metric | baseline | candidate | delta | threshold |"
    assert header in rendered
    rows = [
        line
        for line in rendered.splitlines()
        if line.startswith("| overall pass rate | pass_rate |")
    ]
    assert len(rows) == 1
    assert "max_absolute_decrease" in rows[0]
    _, report, _ = _compare(repo_root, migrated_db, clean_run, degraded_run, thresholds_path)
    for case_id in report.summary.newly_failing:
        assert f"- `{case_id}`" in rendered


@pytest.mark.e2e
def test_the_json_report_is_the_same_payload_the_compare_command_emits(  # noqa: PLR0913, PLR0917 - one pytest fixture per parameter
    repo_root: Path,
    migrated_db: str,
    tmp_path: Path,
    thresholds_path: Path,
    clean_run: str,
    degraded_run: str,
) -> None:
    output = tmp_path / "report.json"
    completed = _cli(
        repo_root,
        migrated_db,
        "report",
        degraded_run,
        "--format",
        "json",
        "--compare-to",
        clean_run,
        "--thresholds",
        str(thresholds_path),
        "--out",
        str(output),
    )
    assert completed.returncode == 0, completed.stderr

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["mode"] == "paired"
    RegressionReport.model_validate(document)

    _, report, _ = _compare(repo_root, migrated_db, clean_run, degraded_run, thresholds_path)
    from_file = {key: value for key, value in document.items() if key != "generated_at"}
    from_compare = {
        key: value
        for key, value in json.loads(report.model_dump_json()).items()
        if key != "generated_at"
    }
    assert from_file == from_compare


WARNING_EXIT_CODE = 7
"""``ExitCode.REGRESSION_WARNING``."""


@pytest.mark.e2e
def test_a_warning_severity_breach_blocks_only_when_the_caller_asks(
    repo_root: Path,
    migrated_db: str,
    tmp_path: Path,
    clean_run: str,
    degraded_run: str,
) -> None:
    policy = tmp_path / "warn-only.yaml"
    policy.write_text(
        "id: warn-only\n"
        "version: '1'\n"
        "checks:\n"
        "  - metric: pass_rate\n"
        "    label: overall pass rate\n"
        "    direction: higher_is_better\n"
        "    max_absolute_decrease: 0.02\n"
        "    severity: warning\n",
        encoding="utf-8",
    )

    permissive, report, _ = _compare(repo_root, migrated_db, clean_run, degraded_run, policy)
    assert permissive == 0
    assert report.verdict.value == "warn"
    assert report.checks[0].status.value == "warning"

    strict, _, _ = _compare(
        repo_root, migrated_db, clean_run, degraded_run, policy, "--fail-on-warning"
    )
    assert strict == WARNING_EXIT_CODE


VALIDATION_EXIT_CODE = 3
"""``ExitCode.VALIDATION_FAILED``."""


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        # An unaddressable metric.
        (
            (
                "id: broken\nversion: '1'\nchecks:\n"
                "  - metric: pass_rat\n    direction: higher_is_better\n"
                "    max_absolute_decrease: 0.02\n"
            ),
            "unknown threshold metric",
        ),
        # The reserved flag, rejected by the frozen contract model itself.
        (
            (
                "id: reserved\nversion: '1'\nchecks:\n"
                "  - metric: pass_rate\n    direction: higher_is_better\n"
                "    max_absolute_decrease: 0.02\n    require_significant: true\n"
            ),
            "require_significant is reserved",
        ),
        # A check that constrains nothing.
        (
            (
                "id: toothless\nversion: '1'\nchecks:\n"
                "  - metric: pass_rate\n    direction: higher_is_better\n"
            ),
            "configures no bound",
        ),
    ],
    ids=["unknown-metric", "require-significant", "no-bound"],
)
def test_an_unusable_threshold_policy_exits_three(  # noqa: PLR0913, PLR0917 - one pytest fixture per parameter
    repo_root: Path,
    migrated_db: str,
    tmp_path: Path,
    clean_run: str,
    policy: str,
    expected: str,
) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(policy, encoding="utf-8")

    completed = _cli(
        repo_root,
        migrated_db,
        "compare",
        clean_run,
        clean_run,
        "--thresholds",
        str(path),
        "--json",
    )

    # Exit 3, not 1: a mistyped metric in the caller's own YAML is a validation
    # failure, and exit 1 is reserved for a defect in the tool.
    assert completed.returncode == VALIDATION_EXIT_CODE, completed.stderr
    assert expected in completed.stderr
    # And stdout stays empty, so a pipeline never finds a second document shape
    # at the path it redirected the report to.
    assert completed.stdout == ""


@pytest.mark.e2e
def test_a_missing_policy_file_exits_three(
    repo_root: Path,
    migrated_db: str,
    tmp_path: Path,
    clean_run: str,
) -> None:
    completed = _cli(
        repo_root,
        migrated_db,
        "compare",
        clean_run,
        clean_run,
        "--thresholds",
        str(tmp_path / "absent.yaml"),
    )

    assert completed.returncode == VALIDATION_EXIT_CODE, completed.stderr
    assert "cannot read threshold policy" in completed.stderr


@pytest.mark.e2e
def test_an_unknown_run_id_leaves_stdout_empty_under_json(
    repo_root: Path,
    migrated_db: str,
    thresholds_path: Path,
    clean_run: str,
) -> None:
    completed = _cli(
        repo_root,
        migrated_db,
        "compare",
        clean_run,
        "not-a-run",
        "--thresholds",
        str(thresholds_path),
        "--json",
    )

    assert completed.returncode == USAGE_EXIT_CODE
    assert completed.stdout == ""
    assert "not-a-run" in completed.stderr


@pytest.mark.e2e
def test_html_output_is_not_offered_and_not_accepted(
    repo_root: Path,
    migrated_db: str,
    clean_run: str,
) -> None:
    rejected = _cli(repo_root, migrated_db, "report", clean_run, "--format", "html")

    assert rejected.returncode == USAGE_EXIT_CODE
    assert "--format must be one of" in rejected.stderr

    listed = _cli(repo_root, migrated_db, "report", "--help")
    assert listed.returncode == 0
    # The help mentions HTML only to say it is unsupported; it never offers it
    # as a value.
    assert "HTML is not supported" in listed.stdout
    assert "md or json" in listed.stdout
