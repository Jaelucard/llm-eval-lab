"""``llm-eval report`` - a readable report for one run, or for a comparison.

Two formats, and deliberately only two. ``--format md`` renders Markdown, which
a reviewer already reads in a pull request, a pager and a chat window.
``--format json`` renders the same facts as data: for a comparison that is the
:class:`~llm_eval_lab.models.RegressionReport` itself, byte for byte what the
API returns and what ``llm-eval compare --json`` writes, so a consumer never has
to know which command produced the file.

**There is no HTML formatter and ``--format html`` is not accepted.** A third
rendering path would have to be tested and secured against the untrusted model
output and benchmark text it embeds, and it would buy presentation polish over a
format every reviewer can already read.

This command RENDERS; it does not gate. It exits 0 whatever the verdict is, and
``llm-eval compare`` is the command whose exit code a pipeline keys off.
"""

from pathlib import Path
from typing import Annotated

import typer

from llm_eval_lab.cli.output import json_document, stderr_console
from llm_eval_lab.models import EvaluatorConfigError
from llm_eval_lab.observability.logging import get_logger
from llm_eval_lab.reporting.formatters import (
    comparison_markdown,
    markup_safe,
    metrics_document,
    metrics_markdown,
)
from llm_eval_lab.services import (
    ComparisonView,
    MetricsView,
    RunNotFoundError,
    build_comparison_service,
    open_unit_of_work_factory,
)

FORMATS: dict[str, str] = {"md": "md", "markdown": "md", "json": "json"}
"""Accepted ``--format`` values, mapped onto the two renderers behind them."""


def _resolve_format(value: str) -> str:
    """Return the renderer for a ``--format`` value.

    Raises:
        typer.BadParameter: for anything else, naming what is accepted. ``html``
            lands here on purpose: it is out of scope for v1 and an unhelpful
            "unknown format" would leave a reader guessing whether it is
            supported elsewhere.
    """
    chosen = FORMATS.get(value.lower())
    if chosen is None:
        accepted = ", ".join(sorted(FORMATS))
        msg = f"--format must be one of {accepted}, got {value!r}"
        raise typer.BadParameter(msg)
    return chosen


def render_comparison(view: ComparisonView, chosen: str) -> str:
    """Render a comparison in the requested format."""
    if chosen == "json":
        return view.report.model_dump_json()
    return comparison_markdown(view.report)


def render_run(view: MetricsView, chosen: str) -> str:
    """Render one run's rollup in the requested format."""
    if chosen == "json":
        return json_document(
            "report",
            {
                "ok": True,
                "run_id": view.run.id,
                "status": view.run.status.value,
                "materialized": view.materialized,
                "metrics": metrics_document(view.metrics),
            },
        )
    return metrics_markdown(view.run, view.metrics, materialized=view.materialized)


def report(  # noqa: PLR0913, PLR0917 - one parameter per documented CLI option
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="The run id to report on.")],
    compare_to: Annotated[
        str | None,
        typer.Option(
            "--compare-to",
            help="Baseline run id. Produces a regression report instead of a run report.",
        ),
    ] = None,
    report_format: Annotated[
        str,
        typer.Option("--format", help="md or json. HTML is not supported."),
    ] = "md",
    thresholds: Annotated[
        Path | None,
        typer.Option(
            "--thresholds",
            help="Threshold policy for --compare-to. Defaults to the shipped policy.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write to this file instead of stdout."),
    ] = None,
) -> None:
    """Render a report for one run, or for a comparison against a baseline."""
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        ExitCode,
        app_context,
        guard,
        run_async,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)
    chosen = _resolve_format(report_format)

    async def fetch() -> str:
        logger = get_logger("llm_eval_lab.cli.report")
        async with open_unit_of_work_factory(context.database_url) as uow_factory:
            service = build_comparison_service(uow_factory, logger)
            if compare_to is None:
                return render_run(await service.metrics(run_id), chosen)
            comparison = await service.compare(
                compare_to,
                run_id,
                thresholds=service.thresholds(thresholds),
            )
            for warning in comparison.warnings:
                console.print(f"[yellow]warning: {markup_safe(warning)}[/yellow]")
            return render_comparison(comparison, chosen)

    def action() -> None:
        try:
            rendered = run_async(fetch())
        except RunNotFoundError as exc:
            console.print(f"[red]no run with id {markup_safe(exc.run_id)}[/red]")
            raise typer.Exit(int(ExitCode.USAGE)) from exc
        except EvaluatorConfigError as exc:
            # Exit 3 for the same reason `compare` does it: an unusable threshold
            # file is the caller's YAML, not a defect in the tool.
            console.print(f"[red]threshold policy is unusable:[/red] {markup_safe(exc)}")
            raise typer.Exit(int(ExitCode.VALIDATION_FAILED)) from exc
        if out is None:
            # Stdout, so `report --format json | jq` works exactly like every
            # other JSON-emitting command here.
            print(rendered)  # noqa: T201 - stdout is this command's output channel
            return
        out.write_text(rendered + "\n", encoding="utf-8")
        console.print(f"wrote {chosen} report to {markup_safe(out)}")

    # `as_json=False`: an error is reported to a human on stderr, because the
    # success payload is a document rather than the command envelope and a
    # failure that returned the envelope instead would be a second shape for a
    # consumer to parse.
    guard(context, "report", as_json=False, action=action)


__all__ = ["report"]
