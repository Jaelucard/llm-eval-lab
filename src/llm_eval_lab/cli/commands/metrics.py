# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
"""``llm-eval metrics`` - the aggregate rollup for one run.

The rollup is read from the row the run engine materialized at finalization.
A run that never reached finalization has no row, so the figures are computed
from its persisted cases and the output says ``materialized: false``: a rollup
over a half-finished run is a different claim from one written when the run
completed, and the flag is what keeps the two distinguishable.
"""

from typing import Annotated

import typer
from rich.console import Console

from llm_eval_lab.cli.output import emit_json, stderr_console
from llm_eval_lab.observability.logging import get_logger
from llm_eval_lab.reporting.formatters import (
    category_table,
    evaluator_table,
    latency_table,
    markup_safe,
    metrics_document,
    metrics_table,
)
from llm_eval_lab.reporting.formatters.tables import LOW_CONFIDENCE_NOTE
from llm_eval_lab.services import MetricsView, build_metrics_service, open_unit_of_work_factory


def render(console: Console, view: MetricsView, *, breakdowns: bool) -> None:
    """Render one run's rollup for a human."""
    console.print(metrics_table(view.metrics))
    console.print(latency_table(view.metrics.latency))
    if view.metrics.latency.low_confidence:
        console.print(f"[yellow]{LOW_CONFIDENCE_NOTE}[/yellow]")
    if not view.materialized:
        console.print(
            "[yellow]computed on demand: this run has no stored rollup, so these "
            "figures reflect the cases persisted so far.[/yellow]"
        )
    if not breakdowns:
        return
    if view.metrics.by_category:
        console.print(category_table(view.metrics.by_category, "by category"))
    if view.metrics.by_tag:
        console.print(category_table(view.metrics.by_tag, "by tag"))
    if view.metrics.by_evaluator:
        console.print(evaluator_table(view.metrics.by_evaluator))


def metrics(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="The run id to report on.")],
    recompute: Annotated[
        bool,
        typer.Option("--recompute", help="Rebuild the rollup from the persisted case results."),
    ] = False,
    breakdowns: Annotated[
        bool,
        typer.Option("--breakdowns/--no-breakdowns", help="Include the per-bucket tables."),
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Report the aggregate metrics for one run."""
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        ExitCode,
        app_context,
        guard,
        run_async,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    async def fetch() -> MetricsView | None:
        logger = get_logger("llm_eval_lab.cli.metrics")
        async with open_unit_of_work_factory(context.database_url) as uow_factory:
            service = build_metrics_service(uow_factory, logger)
            return await (service.recompute(run_id) if recompute else service.get(run_id))

    def action() -> None:
        view = run_async(fetch())
        if view is None:
            if as_json:
                emit_json("metrics", {"ok": False, "error": "not_found", "run_id": run_id})
            else:
                console.print(f"[red]no run with id {markup_safe(run_id)}[/red]")
            raise typer.Exit(int(ExitCode.USAGE))
        if as_json:
            emit_json(
                "metrics",
                {
                    "ok": True,
                    "run_id": view.run.id,
                    "status": view.run.status.value,
                    "materialized": view.materialized,
                    "metrics": metrics_document(view.metrics),
                },
            )
            return
        render(console, view, breakdowns=breakdowns)

    guard(context, "metrics", as_json=as_json, action=action)


__all__ = ["metrics"]
