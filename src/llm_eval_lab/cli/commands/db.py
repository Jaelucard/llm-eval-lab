# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
# FBT is a good rule for library APIs and simply does not apply to a command
# signature that a CLI framework reflects over.
"""``llm-eval db`` - database lifecycle commands.

The database URL is echoed back so an operator can confirm which database was
touched, and it goes through the shared URL redactor first: a PostgreSQL DSN
carries its password in the userinfo, and this command would otherwise print it
to a terminal and into a CI log.

The redacted URL is then ESCAPED before it reaches Rich. Redacting alone was not
enough: Rich reads ``[redacted]`` as a style tag it cannot resolve and drops it,
so the human line read ``admin:@host`` - which states that the database has an
empty password rather than that one was removed.
"""

from typing import Annotated

import typer

from llm_eval_lab.cli.output import emit_json, stderr_console
from llm_eval_lab.redaction import redact_url
from llm_eval_lab.reporting.formatters import markup_safe
from llm_eval_lab.services import database_revision, upgrade_database

app = typer.Typer(help="Create and migrate the database.", no_args_is_help=True)


@app.command()
def upgrade(
    ctx: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Bring the database up to the latest migration, creating it if needed."""
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        app_context,
        guard,
        run_async,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    def action() -> None:
        revision = run_async(upgrade_database(context.database_url))
        shown = redact_url(context.database_url)
        if as_json:
            emit_json(
                "db.upgrade",
                {"ok": True, "database_url": shown, "revision": revision},
            )
            return
        console.print(
            f"database at [bold]{markup_safe(shown)}[/bold] is at revision {markup_safe(revision)}"
        )

    guard(context, "db.upgrade", as_json=as_json, action=action)


@app.command()
def revision(
    ctx: typer.Context,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Report the migration revision the database is currently at."""
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        app_context,
        guard,
        run_async,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    def action() -> None:
        current = run_async(database_revision(context.database_url))
        if as_json:
            emit_json(
                "db.revision",
                {"ok": True, "database_url": redact_url(context.database_url), "revision": current},
            )
            return
        console.print(markup_safe(current) if current else "no migration applied")

    guard(context, "db.revision", as_json=as_json, action=action)
