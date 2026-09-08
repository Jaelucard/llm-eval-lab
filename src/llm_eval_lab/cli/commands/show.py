# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
"""``llm-eval show`` - inspect one run and the cases inside it.

``pass_rate`` is computed from the persisted case results under the run's own
``error_policy``, and is ``null`` rather than ``0.0`` when no case was eligible.
"nothing could be scored" and "everything failed" are different facts and this
command never conflates them.
"""

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from llm_eval_lab.cli.output import emit_json, key_value_table, stderr_console
from llm_eval_lab.models import CaseQuery, CaseStatus
from llm_eval_lab.observability.logging import get_logger
from llm_eval_lab.reporting.formatters import markup_safe
from llm_eval_lab.services import build_run_service, open_unit_of_work_factory


def render_run(console: Console, payload: dict[str, Any]) -> None:
    """Render one run's summary, and its cases when they were requested."""
    rate = payload["pass_rate"]
    console.print(
        key_value_table(
            f"run {payload['run_id']}",
            {
                "status": payload["status"],
                "error": payload["error"],
                "cases": payload["n_cases"],
                "completed": payload["n_completed"],
                "errors": payload["n_errors"],
                "passed": payload["n_passed"],
                "pass denominator": payload["n_pass_denominator"],
                "pass rate": "-" if rate is None else f"{rate:.4f}",
                "created": payload["created_at"],
                "completed at": payload["completed_at"],
            },
        )
    )
    listed = payload.get("cases")
    if not isinstance(listed, list):
        return
    table = Table(title="cases")
    for column in ("case", "status", "passed", "score", "attempts"):
        table.add_column(column)
    for item in listed:
        table.add_row(
            markup_safe(item.get("case_id")),
            markup_safe(item.get("status")),
            "-" if item.get("passed") is None else markup_safe(item.get("passed")),
            "-" if item.get("score") is None else f"{float(item['score']):.3f}",
            markup_safe(item.get("attempts")),
        )
    console.print(table)


def show(  # noqa: PLR0913, PLR0917 - one parameter per documented CLI option
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="The run id to inspect.")],
    cases: Annotated[
        bool, typer.Option("--cases", help="Include per-case results in the output.")
    ] = False,
    failed_only: Annotated[
        bool, typer.Option("--failed", help="With --cases, list only cases that did not pass.")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", help="Maximum cases to list.")] = 50,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Show one run: its status, its totals and, optionally, its cases."""
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        ExitCode,
        app_context,
        guard,
        run_async,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    async def fetch() -> dict[str, Any] | None:
        logger = get_logger("llm_eval_lab.cli.show")
        async with open_unit_of_work_factory(context.database_url) as uow_factory:
            service = build_run_service(context.settings, uow_factory, logger)
            view = await service.get_run(run_id)
            if view is None:
                return None
            payload: dict[str, Any] = {
                "ok": True,
                "run_id": view.run.id,
                "label": view.run.label,
                "status": view.run.status.value,
                "error": view.run.error,
                "created_at": view.run.created_at,
                "completed_at": view.run.completed_at,
                "config": view.run.config.model_dump(mode="json"),
                "n_cases": view.n_cases,
                "n_completed": view.n_completed,
                "n_errors": view.n_errors,
                "n_passed": view.n_passed,
                "n_pass_denominator": view.n_pass_denominator,
                "pass_rate": view.pass_rate,
            }
            if cases:
                page = await service.list_cases(
                    run_id,
                    CaseQuery(limit=limit, passed=False if failed_only else None),
                )
                payload["cases"] = [item.model_dump(mode="json") for item in page.items]
                payload["cases_total"] = page.total
            return payload

    def action() -> None:
        payload = run_async(fetch())
        if payload is None:
            if as_json:
                emit_json("show", {"ok": False, "error": "not_found", "run_id": run_id})
            else:
                console.print(f"[red]no run with id {markup_safe(run_id)}[/red]")
            raise typer.Exit(int(ExitCode.USAGE))
        if as_json:
            emit_json("show", payload)
            return
        render_run(console, payload)

    guard(context, "show", as_json=as_json, action=action)


__all__ = ["CaseStatus", "show"]
