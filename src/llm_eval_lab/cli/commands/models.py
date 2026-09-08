# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
"""``llm-eval models`` - every model this build can reach, and what it costs.

Two sources are merged. A provider advertises the models it serves; a price
table quotes unit prices for models it knows. A model only the table knows is
still listed, because a table quoting something this build cannot reach is
exactly the mismatch worth seeing.

A row that is not usable says why in terms an operator can act on: the name of
the optional extra to install, and the NAME of the environment variable to set.
The credential value is never read and never printed.
"""

from pathlib import Path
from typing import Annotated

import typer

from llm_eval_lab.cli.output import emit_json, stderr_console
from llm_eval_lab.reporting.formatters import markup_safe, models_table
from llm_eval_lab.services import build_catalog_service


def models(
    ctx: typer.Context,
    prices: Annotated[
        Path | None,
        typer.Option("--prices", help="Price table to read. Defaults to the shipped table."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """List every model a provider serves or the price table quotes."""
    from llm_eval_lab.cli.main import app_context, guard  # noqa: PLC0415 - avoids a cycle

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    def action() -> None:
        service = build_catalog_service(context.settings)
        table = service.price_table(prices)
        rows = service.models(table)
        documents = [
            {
                "provider": row.provider,
                "model": row.model,
                "source": row.source,
                "match": row.match,
                "input_per_mtok": row.input_per_mtok,
                "output_per_mtok": row.output_per_mtok,
                "available": row.available,
                "extra": row.extra,
                "credential_env": row.credential_env,
                "credential_resolved": row.credential_resolved,
            }
            for row in rows
        ]
        if as_json:
            emit_json(
                "models",
                {
                    "ok": True,
                    "price_table_id": table.id,
                    "price_table_version": table.version,
                    "price_table_hash": table.content_hash,
                    "currency": table.currency,
                    "models": documents,
                },
            )
            return
        console.print(models_table(tuple(documents)))
        console.print(
            f"price table {markup_safe(table.id)}@{markup_safe(table.version)} "
            f"({markup_safe(table.content_hash)}), "
            f"prices per million tokens in {markup_safe(table.currency)}"
        )

    guard(context, "models", as_json=as_json, action=action)


__all__ = ["models"]
