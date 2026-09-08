# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
"""``llm-eval runs`` - list stored runs, newest first.

Suite and model names are author-supplied, so every cell goes through
``markup_safe``: Rich drops a style tag it cannot resolve, and a model called
``fake-1[beta]`` would otherwise be listed as ``fake-1``.
"""

from typing import Annotated

import typer
from rich.table import Table

from llm_eval_lab.cli.output import emit_json, stderr_console
from llm_eval_lab.models import RunQuery, RunStatus
from llm_eval_lab.observability.logging import get_logger
from llm_eval_lab.reporting.formatters import markup_safe
from llm_eval_lab.services import build_run_service, open_unit_of_work_factory


def runs(  # noqa: PLR0913, PLR0917 - one parameter per documented CLI option
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", help="Maximum runs to list.")] = 20,
    offset: Annotated[int, typer.Option("--offset", help="Runs to skip.")] = 0,
    status: Annotated[
        str | None, typer.Option("--status", help="Only runs in this status.")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Only runs against this provider.")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Only runs against this model.")
    ] = None,
    suite: Annotated[
        str | None, typer.Option("--suite", help="Only runs of this suite name.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """List stored runs, newest first."""
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        app_context,
        guard,
        run_async,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    if status is not None and status not in set(RunStatus):
        msg = f"--status must be one of {', '.join(sorted(RunStatus))}, got {status!r}"
        raise typer.BadParameter(msg)

    query = RunQuery(
        status=RunStatus(status) if status is not None else None,
        provider=provider,
        model=model,
        suite_name=suite,
        limit=limit,
        offset=offset,
    )

    async def fetch() -> dict[str, object]:
        logger = get_logger("llm_eval_lab.cli.runs")
        async with open_unit_of_work_factory(context.database_url) as uow_factory:
            service = build_run_service(context.settings, uow_factory, logger)
            page = await service.list_runs(query)
        return {
            "ok": True,
            "total": page.total,
            "limit": page.limit,
            "offset": page.offset,
            "runs": [item.model_dump(mode="json") for item in page.items],
        }

    def action() -> None:
        payload = run_async(fetch())
        if as_json:
            emit_json("runs", payload)
            return
        table = Table(title="runs")
        for column in ("id", "status", "suite", "model", "cases", "pass rate", "created"):
            table.add_column(column)
        listed = payload["runs"]
        rows = listed if isinstance(listed, list) else []
        for item in rows:
            rate = item.get("pass_rate")
            table.add_row(
                markup_safe(item.get("id")),
                markup_safe(item.get("status")),
                markup_safe(item.get("suite_name")),
                markup_safe(item.get("model")),
                f"{item.get('n_completed')}/{item.get('n_cases')}",
                "-" if rate is None else f"{float(rate):.3f}",
                markup_safe(item.get("created_at")),
            )
        console.print(table)

    guard(context, "runs", as_json=as_json, action=action)
