"""The import-linter exemption for CLI plugin loading is exactly as narrow as it claims.

`pyproject.toml`'s "api and cli reach persistence and providers only through
services" contract carries one named `ignore_imports` entry:
``llm_eval_lab.cli.main -> llm_eval_lab.evaluators.plugins``. That is the sole
legitimate `cli -> evaluators` edge (decision D-CUSTOM: `cli/main.py` loads
opt-in evaluator plugins once, at process start). Everything else in `cli`,
and everything in `api`, must still be forbidden from reaching `evaluators`
directly.

`tests/unit/test_plugins.py` already proves this by static analysis (an AST
scan of every file under `api/` and `services/`) and by proving importing the
API module doesn't even import the plugin module. This test proves the
complementary, mechanical half: that `lint-imports` ITSELF - the tool the
exemption lives in - still catches the same import if it is planted somewhere
the exemption does not name. A probe file is written under `api/`, actually
importing `evaluators.plugins`, and `lint-imports` is run for real against it.
The probe is removed in a `finally`, whether the assertion holds or not, so
this test can never leave stray source behind.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LINT_IMPORTS = shutil.which("lint-imports")
_CONTRACT_NAME = "api and cli reach persistence and providers only through services"
_PROBE_PATH = _REPO_ROOT / "src" / "llm_eval_lab" / "api" / "_lint_boundary_probe.py"


def _run_lint_imports() -> subprocess.CompletedProcess[str]:
    assert _LINT_IMPORTS is not None, "callers skip when lint-imports is not on PATH"
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [_LINT_IMPORTS],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(_LINT_IMPORTS is None, reason="lint-imports is not on PATH")
def test_lint_imports_passes_cleanly_with_no_probe_planted() -> None:
    """Sanity precondition: the real tree, unmodified, keeps every contract."""
    completed = _run_lint_imports()
    assert completed.returncode == 0, completed.stdout
    assert "Contracts: 11 kept, 0 broken." in completed.stdout


@pytest.mark.skipif(_LINT_IMPORTS is None, reason="lint-imports is not on PATH")
def test_the_same_plugin_import_planted_in_api_is_still_caught() -> None:
    """The exemption names `cli.main` only; the identical edge from `api/` must still fail."""
    assert not _PROBE_PATH.exists(), f"{_PROBE_PATH} already exists; a previous run left it behind"
    _PROBE_PATH.write_text(
        '"""Throwaway probe planted and removed by test_cli_import_boundary.py."""\n\n'
        "from llm_eval_lab.evaluators.plugins import load_plugins\n\n"
        '__all__ = ["load_plugins"]\n',
        encoding="utf-8",
    )
    try:
        completed = _run_lint_imports()
    finally:
        _PROBE_PATH.unlink(missing_ok=True)

    assert completed.returncode != 0, completed.stdout
    assert f"{_CONTRACT_NAME} BROKEN" in completed.stdout, completed.stdout
    assert "llm_eval_lab.api._lint_boundary_probe -> llm_eval_lab.evaluators" in completed.stdout, (
        completed.stdout
    )


_HELP_PROBE = textwrap.dedent("""
    import contextlib
    import sys

    sys.argv = ["llm-eval", "--help"]
    from llm_eval_lab.cli.main import app

    with contextlib.suppress(SystemExit):
        app()

    print("plugins_imported:", "llm_eval_lab.evaluators.plugins" in sys.modules)
""")


def test_cli_help_does_not_import_the_plugin_module() -> None:
    """`--help` must not pay for loading the evaluator plugin machinery.

    Click's eager `--help` handling short-circuits before the Typer callback
    body (`cli.main.main`, where `load_evaluator_plugins` lives) ever runs, so
    `evaluators.plugins` is never imported for a `--help` invocation - a
    subprocess, so no earlier test's import of the module can make this pass
    by accident.
    """
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [sys.executable, "-c", _HELP_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "plugins_imported: False" in completed.stdout, completed.stdout
