"""The contract layer must be importable with only pydantic installed.

`llm_eval_lab.models` is the bottom of the dependency graph: every other layer
imports it and it imports nothing back. If a web framework, an ORM or a vendor
SDK ever leaks into it, the whole layering argument collapses quietly, and the
first symptom is a test suite that needs a database to construct a value
object. This runs the import in a fresh interpreter, so the result cannot be
contaminated by whatever the rest of the session has already imported.
"""

from __future__ import annotations

import json
import subprocess
import sys

FORBIDDEN_MODULES = (
    "sqlalchemy",
    "fastapi",
    "typer",
    "httpx",
    "openai",
    "anthropic",
    "google",
    "uvicorn",
    "starlette",
    "aiosqlite",
    "alembic",
    "yaml",
    "jsonschema",
    "rich",
    "structlog",
    # The contract layer must not reach configuration either: a model that read
    # settings would be untestable without an environment.
    "pydantic_settings",
    "dotenv",
)

_PROBE = """
import json, sys
import llm_eval_lab.models  # noqa: F401
forbidden = json.loads(sys.argv[1])
print(json.dumps(sorted(m for m in forbidden if m in sys.modules)))
"""


def _modules_pulled_in_by_importing_the_contract_layer() -> list[str]:
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [sys.executable, "-c", _PROBE, json.dumps(list(FORBIDDEN_MODULES))],
        capture_output=True,
        text=True,
        check=True,
    )
    result: list[str] = json.loads(completed.stdout)
    return result


def test_importing_the_contract_layer_pulls_in_no_heavy_dependency() -> None:
    leaked = _modules_pulled_in_by_importing_the_contract_layer()
    assert leaked == [], f"llm_eval_lab.models imported forbidden modules: {leaked}"


def test_the_probe_would_notice_a_leak() -> None:
    """Guard the guard: a module that IS imported has to be detected.

    Without this, a probe that silently reported nothing would look identical
    to a clean import, and the purity test would pass forever by accident.
    """
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [sys.executable, "-c", _PROBE, json.dumps(["json", "sys"])],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == ["json", "sys"]
