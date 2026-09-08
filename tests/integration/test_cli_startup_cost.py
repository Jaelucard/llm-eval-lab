"""Every CLI command must not pay for the web stack.

`cli/main.py` registers every command at import time, so a module-scope import of
`llm_eval_lab.api` anywhere under `cli/commands/` loads FastAPI, Starlette and
uvicorn into every invocation - `llm-eval --help` included, which is the one
command whose whole job is to be fast. The import contract permits the
`cli -> api` edge that `serve` needs; this is the cost it is not allowed to
impose on anyone else.

A subprocess, necessarily: the test process has already imported FastAPI for the
API tests, so an in-process check of `sys.modules` would pass no matter what.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

WEB_STACK = ("fastapi", "starlette", "uvicorn")

PROBE = """
import sys

from typer.testing import CliRunner

from llm_eval_lab.cli.main import app

result = CliRunner().invoke(app, ["--help"])
if result.exit_code != 0:
    raise SystemExit("--help exited " + str(result.exit_code) + ": " + result.output)
watched = ("fastapi", "starlette", "uvicorn")
print(",".join(sorted(name for name in watched if name in sys.modules)))
"""
"""Invoked in a fresh interpreter. `watched` mirrors `WEB_STACK` below."""


def _run_probe() -> subprocess.CompletedProcess[str]:
    """Invoke `llm-eval --help` in a fresh interpreter and report what it loaded."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-controlled input
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.integration
def test_the_help_output_does_not_load_the_web_stack() -> None:
    completed = _run_probe()
    assert completed.returncode == 0, completed.stderr
    loaded = completed.stdout.strip()
    assert loaded == "", f"`llm-eval --help` imported {loaded}"


@pytest.mark.integration
def test_the_serve_command_is_still_registered() -> None:
    """The deferral must not have been achieved by dropping the command."""
    completed = subprocess.run(
        [sys.executable, "-m", "llm_eval_lab.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "serve" in completed.stdout
