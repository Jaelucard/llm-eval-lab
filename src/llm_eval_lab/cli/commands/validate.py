# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
# FBT is a good rule for library APIs and simply does not apply to a command
# signature that a CLI framework reflects over.
"""``llm-eval validate`` - check a benchmark file before spending anything on it.

Reports EVERY problem in the file, not the first one, and exits 3 when the file
is invalid so a CI job can gate on it.
"""

from pathlib import Path
from typing import Annotated

import typer

from llm_eval_lab.cli.output import emit_json, key_value_table, stderr_console
from llm_eval_lab.services import build_catalog_service


def validate(
    ctx: typer.Context,
    suite: Annotated[Path, typer.Argument(help="Path to the benchmark YAML or JSON file.")],
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Validate a benchmark file and report every problem it contains."""
    from llm_eval_lab.cli.main import app_context, guard  # noqa: PLC0415 - avoids a cycle

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    def action() -> None:
        service = build_catalog_service(context.settings)
        report = service.validate(suite)
        if as_json:
            emit_json(
                "validate",
                {
                    "ok": True,
                    "path": report.path,
                    "suite_name": report.suite_name,
                    "suite_version": report.suite_version,
                    "suite_hash": report.suite_hash,
                    "n_cases": report.n_cases,
                    "evaluator_types": list(report.evaluator_types),
                },
            )
            return
        console.print(
            key_value_table(
                f"{report.suite_name} is valid",
                {
                    "path": report.path,
                    "version": report.suite_version,
                    "suite_hash": report.suite_hash,
                    "cases": report.n_cases,
                    "evaluators": ", ".join(report.evaluator_types) or "-",
                },
            )
        )

    guard(context, "validate", as_json=as_json, action=action)
