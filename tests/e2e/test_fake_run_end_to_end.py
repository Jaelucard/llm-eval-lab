"""The walking skeleton, exercised the way a user exercises it: as a process.

These tests shell out to the installed `llm-eval` entry point rather than
calling functions, because two of the guarantees under test are only observable
from outside the process: that stdout carries JSON and nothing else, and that a
SIGINT mid-run leaves a CANCELLED run whose completed cases are still queryable.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

SMOKE_CASES = 6
READINESS_POLL_S = 0.05
READINESS_DEADLINE_S = 30.0
RUN_TIMEOUT_S = 120
SIGINT_EXIT_CODE = 130
CASE_LATENCY_MS = 500


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


@pytest.fixture
def cli_module(repo_root: Path) -> Path:
    """Ensure the package is runnable as `python -m llm_eval_lab.cli`."""
    return repo_root / "src" / "llm_eval_lab" / "cli" / "__main__.py"


@pytest.mark.e2e
def test_run_then_show_reports_a_perfect_pass_rate(
    repo_root: Path, migrated_db: str, cli_module: Path
) -> None:
    del cli_module
    run = _cli(
        repo_root,
        migrated_db,
        "run",
        str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
        "--provider",
        "fake",
        "--model",
        "fake-1",
        "--fake-mode",
        "expected",
        "--json",
    )
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"

    payload: dict[str, Any] = json.loads(run.stdout)
    assert payload["schema_version"] == 1
    assert payload["run_id"]

    shown = _cli(repo_root, migrated_db, "show", payload["run_id"], "--json")
    assert shown.returncode == 0, f"{shown.stdout}\n{shown.stderr}"
    view: dict[str, Any] = json.loads(shown.stdout)

    assert view["n_cases"] == SMOKE_CASES
    assert view["n_completed"] == SMOKE_CASES
    assert view["pass_rate"] == 1.0


@pytest.mark.e2e
def test_stdout_carries_json_alone(repo_root: Path, migrated_db: str) -> None:
    """`llm-eval run ... --json 2>/dev/null | jq .` must always work."""
    run = _cli(
        repo_root,
        migrated_db,
        "run",
        str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
        "--provider",
        "fake",
        "--model",
        "fake-1",
        "--fake-mode",
        "expected",
        "--json",
    )
    assert run.returncode == 0
    # Parses as ONE document with nothing before or after it.
    assert json.loads(run.stdout)["ok"] is True
    assert run.stdout.strip().count("\n") == 0


@pytest.mark.e2e
def test_the_database_holds_rows_in_every_table(
    repo_root: Path, migrated_db: str, tmp_path: Path
) -> None:
    import sqlite3  # noqa: PLC0415 - reads next to its use

    run = _cli(
        repo_root,
        migrated_db,
        "run",
        str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
        "--provider",
        "fake",
        "--model",
        "fake-1",
        "--fake-mode",
        "expected",
        "--json",
    )
    assert run.returncode == 0

    with sqlite3.connect(tmp_path / "test.db") as connection:
        counts = {
            # S608 suppressed: the table names come from the fixed literal tuple
            # on the next line, not from input of any kind.
            table: connection.execute(f"select count(*) from {table}").fetchone()[0]  # noqa: S608
            for table in ("runs", "case_results", "model_responses", "evaluations")
        }
    assert counts == {
        "runs": 1,
        "case_results": SMOKE_CASES,
        "model_responses": SMOKE_CASES,
        "evaluations": SMOKE_CASES,
    }


@pytest.mark.e2e
def test_runs_listing_finds_the_run(repo_root: Path, migrated_db: str) -> None:
    run = _cli(
        repo_root,
        migrated_db,
        "run",
        str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
        "--provider",
        "fake",
        "--model",
        "fake-1",
        "--fake-mode",
        "expected",
        "--json",
    )
    assert run.returncode == 0
    run_id = json.loads(run.stdout)["run_id"]

    listing = _cli(repo_root, migrated_db, "runs", "--limit", "1", "--json")
    assert listing.returncode == 0
    assert json.loads(listing.stdout)["runs"][0]["id"] == run_id


def _wait_for_first_persisted_case(database: Path, *, deadline_s: float) -> None:
    """Block until the child has durably finished its first case.

    A fixed `time.sleep` was racing CLI startup: measured startup is 0.37-0.43 s,
    so on a loaded machine the signal could arrive before the run had installed
    its SIGINT handler. Python's default handler would then raise
    `KeyboardInterrupt`, the process would exit 130 having printed no JSON at
    all, and the test would fail on an opaque `JSONDecodeError` rather than on
    the thing that actually went wrong.

    Polling for a persisted case result is the real readiness signal: it proves
    the handler is installed, the run is executing, and at least one case is
    already durable, which is exactly the precondition the assertions need.
    """
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if database.is_file():
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                rows = connection.execute("select count(*) from case_results").fetchone()[0]
            if rows >= 1:
                return
        time.sleep(READINESS_POLL_S)
    msg = f"the run did not persist a case within {deadline_s}s"
    raise AssertionError(msg)


@pytest.mark.e2e
@pytest.mark.slow
def test_sigint_leaves_a_cancelled_run_whose_cases_are_queryable(
    repo_root: Path, migrated_db: str, tmp_path: Path
) -> None:
    """Exit criterion 5: interrupting a run must not lose the work already done."""
    database = tmp_path / "test.db"
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell, test-controlled input
        [
            sys.executable,
            "-m",
            "llm_eval_lab.cli",
            "run",
            str(repo_root / "examples" / "benchmarks" / "smoke.yaml"),
            "--provider",
            "fake",
            "--model",
            "fake-1",
            "--fake-mode",
            "expected",
            "--concurrency",
            "1",
            "--provider-option",
            "sleep=true",
            "--provider-option",
            f"latency_min_ms={CASE_LATENCY_MS}",
            "--provider-option",
            f"latency_max_ms={CASE_LATENCY_MS}",
            "--json",
        ],
        cwd=repo_root,
        env={**os.environ, "LLM_EVAL_DATABASE_URL": migrated_db},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_first_persisted_case(database, deadline_s=READINESS_DEADLINE_S)
    except AssertionError:
        process.kill()
        raise
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=RUN_TIMEOUT_S)

    assert process.returncode == SIGINT_EXIT_CODE, f"{stdout}\n{stderr}"
    assert stdout.strip(), (
        f"the run exited 130 without emitting JSON, so the signal most likely "
        f"arrived before the handler was installed. stderr:\n{stderr}"
    )
    payload = json.loads(stdout)
    assert payload["status"] == "cancelled"
    assert 1 <= payload["n_completed"] < SMOKE_CASES

    shown = _cli(repo_root, migrated_db, "show", payload["run_id"], "--cases", "--json")
    assert shown.returncode == 0, f"{shown.stdout}\n{shown.stderr}"
    view = json.loads(shown.stdout)
    assert view["status"] == "cancelled"
    assert view["n_completed"] == payload["n_completed"]
    assert len(view["cases"]) == payload["n_completed"]
