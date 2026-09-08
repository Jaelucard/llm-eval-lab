# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
"""``llm-eval export`` - one run's case results as JSON or CSV.

The row shape and the CSV column order are declared once, in
``reporting.formatters``, and the JSON envelope around them once in
``services.export_document`` - which ``GET /api/runs/{id}/export`` calls too - so
an export written today lines up column for column with one written after the
next release, and with one taken over HTTP.

An unknown value is an empty CSV cell and a JSON ``null``, never a zero. A
spreadsheet reads an empty numeric cell as absent and a zero as a measurement,
and that is exactly the distinction between a cost that is unknown and one that
is free.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from llm_eval_lab.cli.output import emit_json, json_document, stderr_console
from llm_eval_lab.models import CaseResult
from llm_eval_lab.observability.logging import get_logger
from llm_eval_lab.reporting.formatters import markup_safe, write_case_csv
from llm_eval_lab.services import (
    MetricsView,
    build_metrics_service,
    export_document,
    open_unit_of_work_factory,
)

FORMATS = frozenset({"json", "csv"})
"""The export formats this command accepts."""


@dataclass(frozen=True)
class _Export:
    """Everything one export needs, read inside a single unit of work."""

    view: MetricsView
    results: tuple[CaseResult, ...]


def _write_csv(exported: _Export, output: Path | None, console: Console) -> None:
    """Write the case rows as CSV, to a file or to stdout."""
    if output is None:
        write_case_csv(exported.results, sys.stdout)
        return
    with output.open("w", encoding="utf-8", newline="") as handle:
        write_case_csv(exported.results, handle)
    console.print(f"wrote {len(exported.results)} rows to {markup_safe(output)}")


def _write_json(
    run_id: str,
    exported: _Export,
    output: Path | None,
    console: Console,
    *,
    include_metrics: bool,
) -> None:
    """Emit the case rows, and optionally the rollup, as the standard envelope."""
    # The envelope is built in `services.export_document`, which the API's export
    # route also calls, so the two surfaces cannot drift into two shapes of the
    # same document.
    payload: dict[str, Any] = {
        "ok": True,
        **export_document(
            run_id,
            exported.results,
            exported.view.metrics if include_metrics else None,
        ),
    }
    if output is None:
        emit_json("export", payload)
        return
    output.write_text(json_document("export", payload) + "\n", encoding="utf-8")
    console.print(f"wrote {len(exported.results)} cases to {markup_safe(output)}")


def export(  # noqa: PLR0913, PLR0917 - one parameter per documented CLI option
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="The run id to export.")],
    export_format: Annotated[
        str,
        typer.Option("--format", help="json or csv."),
    ] = "json",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of stdout."),
    ] = None,
    include_metrics: Annotated[
        bool,
        typer.Option("--metrics/--no-metrics", help="Include the run's rollup in JSON output."),
    ] = True,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the JSON envelope on stdout. Implies --format json."),
    ] = False,
) -> None:
    """Export one run's case results.

    ``--format csv`` writes bare CSV rather than a JSON envelope, because a CSV
    wrapped in JSON is useful to nobody. ``--json`` and ``--format json`` both
    produce the standard envelope.
    """
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        ExitCode,
        app_context,
        guard,
        run_async,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)
    chosen = "json" if as_json else export_format.lower()
    if chosen not in FORMATS:
        msg = f"--format must be json or csv, got {export_format!r}"
        raise typer.BadParameter(msg)

    async def fetch() -> _Export | None:
        logger = get_logger("llm_eval_lab.cli.export")
        async with open_unit_of_work_factory(context.database_url) as uow_factory:
            service = build_metrics_service(uow_factory, logger)
            view = await service.get(run_id)
            if view is None:
                return None
            return _Export(view=view, results=await service.cases(run_id))

    def action() -> None:
        exported = run_async(fetch())
        if exported is None:
            if chosen == "json":
                emit_json("export", {"ok": False, "error": "not_found", "run_id": run_id})
            else:
                console.print(f"[red]no run with id {markup_safe(run_id)}[/red]")
            raise typer.Exit(int(ExitCode.USAGE))
        if chosen == "csv":
            _write_csv(exported, output, console)
            return
        _write_json(run_id, exported, output, console, include_metrics=include_metrics)

    guard(context, "export", as_json=chosen == "json", action=action)


__all__ = ["export"]
