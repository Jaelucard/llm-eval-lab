# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
# FBT is a good rule for library APIs and simply does not apply to a command
# signature that a CLI framework reflects over.
"""``llm-eval providers`` - list the provider integrations this build supports.

``sdk_available`` and ``credential_present`` are reported SEPARATELY, never
folded into one opaque boolean. A provider whose SDK is not installed and a
provider whose credential is not set are two different operator actions -
``pip install`` versus setting an environment variable - and collapsing them
into a single ``available: false`` would leave the operator guessing which one
applies. ``credential_env`` is a variable NAME, never a value: this command
never has the value to leak in the first place.
"""

from typing import Annotated

import typer
from rich.table import Table

from llm_eval_lab.cli.output import emit_json, stderr_console
from llm_eval_lab.reporting.formatters import markup_safe
from llm_eval_lab.services import build_catalog_service


def providers(
    ctx: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """List every registered provider, its required extra and its credential status."""
    from llm_eval_lab.cli.main import app_context, guard  # noqa: PLC0415 - avoids a cycle

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    def action() -> None:
        catalog = build_catalog_service(context.settings)
        rows = [
            {
                "name": entry.name,
                "requires_credential": entry.requires_credential,
                "required_extra": entry.extra,
                "sdk_available": entry.available,
                "credential_env": entry.credential_env,
                # `is not False`, not `is True`: a provider that needs no
                # credential (`None`) has nothing standing in its way either,
                # and must read the same as one whose credential resolved.
                "credential_present": catalog.credential_status(entry.name) is not False,
                "summary": entry.summary,
            }
            for entry in catalog.providers()
        ]
        if as_json:
            emit_json("providers", {"ok": True, "providers": rows})
            return
        table = Table(title="providers")
        table.add_column("name", style="bold")
        table.add_column("sdk")
        table.add_column("credential")
        table.add_column("extra")
        table.add_column("summary")
        for row in rows:
            if row["requires_credential"]:
                credential = "set" if row["credential_present"] else "MISSING"
            else:
                credential = "none"
            table.add_row(
                markup_safe(str(row["name"])),
                "installed" if row["sdk_available"] else "MISSING",
                credential,
                markup_safe(str(row["required_extra"] or "-")),
                markup_safe(str(row["summary"])),
            )
        console.print(table)

    guard(context, "providers", as_json=as_json, action=action)
