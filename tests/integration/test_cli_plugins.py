"""`cli/main.py` is the one call site for evaluator plugin loading (decision D-CUSTOM).

`tests/unit/test_plugins.py` already covers the four gates and the refusals in
isolation, against a throwaway `EvaluatorRegistry()`. What is new here, and
untested elsewhere, is the WIRING: that `llm-eval` actually calls the loader at
process start, that a loaded plugin's evaluator type is genuinely usable by a
real run, and that what was loaded is recorded on the persisted
`RunConfig.plugins`.

Run as a SUBPROCESS rather than through an in-process `CliRunner`, on purpose.
`llm_eval_lab.evaluators.default_registry()` is a process-wide, `lru_cache`d
singleton (by design: a plugin loaded once at CLI start must stay visible to
every later `evaluator_registry()` call in the same process). Registering a
probe evaluator into it from an in-process test would leak into every other
test in this pytest session that touches evaluators - "evaluators --json"
listing exactly 10 built-ins, for one. A subprocess gets a fresh, empty
registry that no other test can ever observe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from llm_eval_lab.cli.main import ExitCode, app
from llm_eval_lab.settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

_SCRIPT = textwrap.dedent(r"""
    import builtins
    import json
    import sys
    import types
    from importlib.metadata import EntryPoint
    from pathlib import Path
    from typing import ClassVar

    from pydantic import BaseModel, ConfigDict

    from llm_eval_lab.evaluators import plugins as plugins_module
    from llm_eval_lab.evaluators.base import BaseEvaluator

    PROBE_MODULE = "cli_plugin_probe"
    PROBE_NAME = "cli_probe_evals"
    PROBE_TYPE = "cli_probe_always_passes"

    class ProbeParams(BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid")

    class ProbeEvaluator(BaseEvaluator[ProbeParams]):
        # `builtins.type` because the `type` ClassVar below shadows the builtin.
        type: ClassVar[str] = PROBE_TYPE
        params_model: ClassVar[builtins.type[BaseModel]] = ProbeParams

        async def evaluate(self, case, response, context):
            del case, response, context
            return self.verdict(passed=True)

    module = types.ModuleType(PROBE_MODULE)
    module.ProbeEvaluator = ProbeEvaluator
    sys.modules[PROBE_MODULE] = module

    point = EntryPoint(
        name=PROBE_NAME,
        value=f"{PROBE_MODULE}:ProbeEvaluator",
        group=plugins_module.ENTRY_POINT_GROUP,
    )
    # A never-advertised second allowlist entry, so the CLI's unloaded-names
    # warning path runs too (checked below via the persisted run succeeding
    # regardless - an unloaded, unadvertised name must never abort startup).
    plugins_module.entry_points = lambda **_: [point]

    suite_path, db_url = sys.argv[1], sys.argv[2]
    Path(suite_path).write_text(
        "schema_version: 1\n"
        "name: plugin-probe\n"
        'version: "1"\n'
        "cases:\n"
        "  - id: only\n"
        "    input: hi\n"
        "    evaluators:\n"
        f"      - type: {PROBE_TYPE}\n",
        encoding="utf-8",
    )

    from typer.testing import CliRunner

    from llm_eval_lab.cli.main import app

    runner = CliRunner()

    upgrade = runner.invoke(app, ["--db-url", db_url, "db", "upgrade", "--json"])
    assert upgrade.exit_code == 0, upgrade.output

    listed = runner.invoke(app, ["evaluators", "--json"])
    assert listed.exit_code == 0, listed.output
    types_listed = {row["type"] for row in json.loads(listed.stdout)["evaluators"]}
    assert PROBE_TYPE in types_listed, types_listed

    run_args = [
        "--db-url", db_url, "run", suite_path,
        "--provider", "fake", "--model", "fake-1", "--json",
    ]
    run_result = runner.invoke(app, run_args)
    assert run_result.exit_code == 0, run_result.output
    run_id = json.loads(run_result.stdout)["run_id"]

    show_result = runner.invoke(app, ["--db-url", db_url, "show", run_id, "--json"])
    assert show_result.exit_code == 0, show_result.output
    config = json.loads(show_result.stdout)["config"]
    records = config["plugins"]
    assert len(records) == 1, records
    assert records[0]["entry_point"] == PROBE_NAME
    assert records[0]["evaluator_types"] == [PROBE_TYPE]

    print("SUBPROCESS_OK")
""")


def test_a_loaded_plugin_is_usable_in_a_run_and_recorded_on_run_config(tmp_path: Path) -> None:
    suite_path = tmp_path / "probe.yaml"
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'probe.db'}"
    env = dict(os.environ)
    env["LLM_EVAL_PLUGINS_ENABLED"] = "true"
    # Includes a name nothing advertises, so `unloaded_names` has something to
    # report - non-fatal, and the run below must still succeed regardless.
    env["LLM_EVAL_PLUGINS_ALLOWED"] = '["cli_probe_evals", "never_installed"]'

    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [sys.executable, "-c", _SCRIPT, str(suite_path), db_url],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
    assert "SUBPROCESS_OK" in completed.stdout


def test_plugins_disabled_by_default_leaves_run_config_plugins_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, migrated_db: str, repo_root: Path
) -> None:
    """The default path: no env vars set, so `RunConfig.plugins` stays empty."""
    del tmp_path
    monkeypatch.delenv("LLM_EVAL_PLUGINS_ENABLED", raising=False)
    monkeypatch.delenv("LLM_EVAL_PLUGINS_ALLOWED", raising=False)
    monkeypatch.setenv("LLM_EVAL_DATABASE_URL", migrated_db)
    get_settings.cache_clear()

    runner = CliRunner()
    started = runner.invoke(
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
    assert started.exit_code == ExitCode.SUCCESS, started.output
    run_id = json.loads(started.stdout)["run_id"]

    shown = runner.invoke(app, ["show", run_id, "--json"])
    assert shown.exit_code == ExitCode.SUCCESS, shown.output
    assert json.loads(shown.stdout)["config"]["plugins"] == []
