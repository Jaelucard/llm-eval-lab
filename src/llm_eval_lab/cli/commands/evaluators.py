# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
# FBT is a good rule for library APIs and simply does not apply to a command
# signature that a CLI framework reflects over.
"""``llm-eval evaluators`` - list the evaluator types a benchmark may use."""

from typing import Annotated

import typer
from rich.table import Table

from llm_eval_lab.cli.output import emit_json, stderr_console
from llm_eval_lab.reporting.formatters import markup_safe
from llm_eval_lab.services import build_catalog_service


def evaluators(
    ctx: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
    schema: Annotated[
        bool,
        typer.Option("--schema", help="Include each evaluator's parameter JSON Schema."),
    ] = False,
) -> None:
    """List every registered evaluator type."""
    from llm_eval_lab.cli.main import app_context, guard  # noqa: PLC0415 - avoids a cycle

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    def action() -> None:
        entries = build_catalog_service(context.settings).evaluators()
        if as_json:
            emit_json(
                "evaluators",
                {
                    "ok": True,
                    "evaluators": [
                        {
                            "type": entry.type,
                            "needs_expected": entry.needs_expected,
                            "is_model_graded": entry.is_model_graded,
                            "summary": entry.summary,
                            **({"params_schema": entry.params_schema()} if schema else {}),
                        }
                        for entry in entries
                    ],
                },
            )
            return
        table = Table(title="evaluators")
        table.add_column("type", style="bold")
        table.add_column("needs expected")
        table.add_column("model graded")
        table.add_column("summary")
        for entry in entries:
            table.add_row(
                markup_safe(entry.type),
                "yes" if entry.needs_expected else "no",
                "yes" if entry.is_model_graded else "no",
                markup_safe(entry.summary),
            )
        console.print(table)

    guard(context, "evaluators", as_json=as_json, action=action)
