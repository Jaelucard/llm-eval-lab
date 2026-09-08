# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
"""``llm-eval pricing`` - inspect and validate the external price table.

Two subcommands, and only two. ``show`` prints a table with the content hash a
run records; ``validate`` refuses a file that would make a historical cost
figure unrecoverable. Repricing an existing run is deliberately out of scope:
it is what would turn ``run_metrics`` from one row per run into a history, and
keeping it out is what lets the metrics rollup ship without a schema change.
"""

from pathlib import Path
from typing import Annotated

import typer

from llm_eval_lab.cli.output import emit_json, key_value_table, stderr_console
from llm_eval_lab.models import PricingError
from llm_eval_lab.reporting.formatters import (
    markup_safe,
    price_entry_table,
    price_table_document,
)
from llm_eval_lab.services import build_catalog_service

app = typer.Typer(
    name="pricing",
    help="Inspect and validate the price table.",
    no_args_is_help=True,
)


@app.command("show")
def show(
    ctx: typer.Context,
    prices: Annotated[
        Path | None,
        typer.Option("--prices", help="Price table to read. Defaults to the shipped table."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Print a price table, its version and its content hash."""
    from llm_eval_lab.cli.main import app_context, guard  # noqa: PLC0415 - avoids a cycle

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    def action() -> None:
        table = build_catalog_service(context.settings).price_table(prices)
        if as_json:
            emit_json("pricing show", {"ok": True, "price_table": price_table_document(table)})
            return
        console.print(
            key_value_table(
                f"price table {table.id}",
                {
                    "version": table.version,
                    "currency": table.currency,
                    "content hash": table.content_hash,
                    "source": table.source,
                    "entries": len(table.models),
                },
            )
        )
        console.print(price_entry_table(table.models))

    guard(context, "pricing show", as_json=as_json, action=action)


@app.command("validate")
def validate(
    ctx: typer.Context,
    prices: Annotated[
        Path | None,
        typer.Option("--prices", help="Price table to check. Defaults to the shipped table."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Check a price file, and refuse one that breaks cost reproducibility.

    Exits 3 on any problem, including an unreadable or malformed file, which is
    the same code ``llm-eval validate`` uses for a bad benchmark. A CI job
    gating on a price file gets one code to check rather than two.
    """
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        ExitCode,
        app_context,
        guard,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    def action() -> None:
        service = build_catalog_service(context.settings)
        try:
            report = service.validate_prices(prices)
        except PricingError as exc:
            if as_json:
                emit_json(
                    "pricing validate",
                    {
                        "ok": False,
                        "path": str(prices) if prices is not None else None,
                        "errors": [str(exc)],
                    },
                )
            else:
                console.print(f"[red]price table is invalid:[/red] {markup_safe(exc)}")
            raise typer.Exit(int(ExitCode.VALIDATION_FAILED)) from exc

        problems = [*report.conflicts, *report.duplicates]
        if as_json:
            emit_json(
                "pricing validate",
                {
                    "ok": report.ok,
                    "path": report.path,
                    "price_table_id": report.table.id,
                    "price_table_version": report.table.version,
                    "price_table_hash": report.table.content_hash,
                    "n_entries": len(report.table.models),
                    "errors": problems,
                },
            )
        else:
            for problem in problems:
                console.print(f"[red]error:[/red] {markup_safe(problem)}")
            if report.ok:
                console.print(
                    f"[green]{markup_safe(report.path)} is valid[/green]: "
                    f"{markup_safe(report.table.id)}@{markup_safe(report.table.version)} with "
                    f"{len(report.table.models)} entries, "
                    f"hash {markup_safe(report.table.content_hash)}"
                )
        if not report.ok:
            raise typer.Exit(int(ExitCode.VALIDATION_FAILED))

    guard(context, "pricing validate", as_json=as_json, action=action)


__all__ = ["app", "show", "validate"]
