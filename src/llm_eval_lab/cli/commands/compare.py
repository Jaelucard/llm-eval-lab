# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
"""``llm-eval compare`` - the CI gate.

This command exists to be run by a pipeline, so its EXIT CODE is its primary
output:

===== ==============================================================
0     no regression (verdict ``pass``, or ``warn`` without --fail-on-warning)
2     one of the two run ids does not exist
4     a regression was detected (verdict ``fail``)
6     the two runs are not comparable (verdict ``incomparable``)
7     warning-severity breaches only, and --fail-on-warning was passed
===== ==============================================================

``--json`` writes the :class:`~llm_eval_lab.models.RegressionReport` itself to
stdout and NOTHING else - not the usual command envelope. The report already
carries its own ``schema_version``, and wrapping it would make the file a
pipeline reads a different shape from the object the API returns for the same
comparison. Every human-facing line still goes to stderr, so
``compare ... --json > report.json`` produces a file that validates against the
contract.

That guarantee is absolute, so it holds on the FAILURE paths too: an unknown run
id or an unusable policy writes nothing at all to stdout and reports itself on
stderr with a distinguishing exit code. A pipeline therefore never finds a file
of a second shape at the path it expects, and an empty file plus a non-zero exit
is unambiguous.

The verdict is decided by point estimates alone (decision D-GATE), so the same
two runs always produce the same exit code. The intervals and the McNemar
p-value in the report are advisory context for whoever reads the failure.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from llm_eval_lab.cli.output import stderr_console
from llm_eval_lab.models import (
    CheckStatus,
    EvaluatorConfigError,
    RegressionReport,
    Verdict,
)
from llm_eval_lab.observability.logging import get_logger
from llm_eval_lab.reporting.formatters import markup_safe
from llm_eval_lab.services import (
    ComparisonView,
    RunNotFoundError,
    build_comparison_service,
    open_unit_of_work_factory,
)

_STATUS_STYLES: dict[CheckStatus, str] = {
    CheckStatus.PASSED: "green",
    CheckStatus.FAILED: "red",
    CheckStatus.WARNING: "yellow",
    CheckStatus.INSUFFICIENT_DATA: "cyan",
    CheckStatus.MISSING_METRIC: "magenta",
}
"""Colour per status. The status word itself is always printed, colour or not."""

_VERDICT_STYLES: dict[Verdict, str] = {
    Verdict.PASS: "green",
    Verdict.FAIL: "red",
    Verdict.WARN: "yellow",
    Verdict.INCOMPARABLE: "magenta",
}
"""Colour per verdict."""


def exit_code_for(verdict: Verdict, *, fail_on_warning: bool) -> int:
    """Map a verdict onto this command's exit code.

    A warning is not a failure unless the caller asked for it to be. A pipeline
    that wants advisory checks to block opts in with ``--fail-on-warning``
    rather than discovering that they always did.
    """
    from llm_eval_lab.cli.main import ExitCode  # noqa: PLC0415 - avoids a cycle

    if verdict is Verdict.FAIL:
        return int(ExitCode.REGRESSION)
    if verdict is Verdict.INCOMPARABLE:
        return int(ExitCode.INCOMPARABLE)
    if verdict is Verdict.WARN and fail_on_warning:
        return int(ExitCode.REGRESSION_WARNING)
    return int(ExitCode.SUCCESS)


def _number(value: float | None, places: int = 4) -> str:
    """Render a float, or ``-`` when it is genuinely unknown."""
    return "-" if value is None else f"{value:.{places}f}"


def _signed(value: float | None, places: int = 4) -> str:
    """Render a delta with an explicit sign."""
    return "-" if value is None else f"{value:+.{places}f}"


def check_table(report: RegressionReport) -> Table:
    """Render every threshold outcome, one row each.

    Every cell goes through :func:`~llm_eval_lab.reporting.formatters.markup_safe`:
    a check label and a metric path both interpolate names authored in an
    untrusted benchmark or policy file, and Rich silently DELETES a bracketed
    span it cannot resolve as a style.
    """
    table = Table(
        title=markup_safe(f"threshold checks ({report.thresholds_id}@{report.thresholds_version})"),
        box=None,
    )
    for column in (
        "check",
        "metric",
        "baseline",
        "candidate",
        "delta",
        "threshold",
        "n base/cand",
        "status",
    ):
        table.add_column(column)
    for check in report.checks:
        style = _STATUS_STYLES[check.status]
        table.add_row(
            markup_safe(check.label),
            markup_safe(check.metric),
            _number(check.baseline),
            _number(check.candidate),
            _signed(check.delta),
            _number(check.threshold_value),
            f"{check.n_baseline}/{check.n_candidate}",
            f"[{style}]{check.status.value.upper()}[/{style}]",
        )
    return table


def render(console: Console, view: ComparisonView) -> None:
    """Render a comparison for a human, on stderr."""
    report = view.report
    style = _VERDICT_STYLES[report.verdict]
    console.print(f"[{style}]verdict: {report.verdict.value.upper()}[/{style}]")
    console.print(
        f"mode: {report.mode}  paired cases: {report.paired_case_count}  "
        f"changed cases: {len(report.changed_cases)}"
    )
    if report.incomparable_reason is not None:
        console.print(f"[magenta]{markup_safe(report.incomparable_reason)}[/magenta]")
    for warning in view.warnings:
        console.print(f"[yellow]warning: {markup_safe(warning)}[/yellow]")
    if report.checks:
        console.print(check_table(report))
    for check in report.checks:
        if check.note:
            console.print(f"  {markup_safe(check.label)}: {markup_safe(check.note)}")
    if report.summary.newly_failing:
        listed = ", ".join(report.summary.newly_failing)
        console.print(
            f"[red]newly failing ({len(report.summary.newly_failing)}):[/red] {markup_safe(listed)}"
        )
    if report.summary.newly_passing:
        listed = ", ".join(report.summary.newly_passing)
        console.print(
            f"[green]newly passing ({len(report.summary.newly_passing)}):[/green] "
            f"{markup_safe(listed)}"
        )


def emit_report(report: RegressionReport) -> None:
    """Write the report itself to stdout, with no envelope around it."""
    sys.stdout.write(report.model_dump_json())
    sys.stdout.write("\n")
    sys.stdout.flush()


def compare(  # noqa: PLR0913, PLR0917 - one parameter per documented CLI option
    ctx: typer.Context,
    baseline_run_id: Annotated[str, typer.Argument(help="The run id to compare against.")],
    candidate_run_id: Annotated[str, typer.Argument(help="The run id under test.")],
    thresholds: Annotated[
        Path | None,
        typer.Option(
            "--thresholds",
            help="Threshold policy file. Defaults to the policy shipped with the package.",
        ),
    ] = None,
    fail_on_warning: Annotated[
        bool,
        typer.Option(
            "--fail-on-warning",
            help="Exit 7 when the only breaches are warning-severity.",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Write the regression report to stdout, and nothing else."),
    ] = False,
) -> None:
    """Compare a candidate run against a baseline and gate on the result."""
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        ExitCode,
        app_context,
        guard,
        run_async,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)

    async def fetch() -> ComparisonView:
        logger = get_logger("llm_eval_lab.cli.compare")
        async with open_unit_of_work_factory(context.database_url) as uow_factory:
            service = build_comparison_service(uow_factory, logger)
            return await service.compare(
                baseline_run_id,
                candidate_run_id,
                thresholds=service.thresholds(thresholds),
            )

    def action() -> None:
        try:
            view = run_async(fetch())
        except RunNotFoundError as exc:
            console.print(f"[red]no run with id {markup_safe(exc.run_id)}[/red]")
            raise typer.Exit(int(ExitCode.USAGE)) from exc
        except EvaluatorConfigError as exc:
            # Exit 3, not 1. A mistyped metric in the caller's own threshold file
            # is a validation failure, and exit 1 is reserved for "this is a bug
            # in the tool". Handled here rather than by widening `guard`, which
            # is shared with commands where a different mapping may be right.
            console.print(f"[red]threshold policy is unusable:[/red] {markup_safe(exc)}")
            raise typer.Exit(int(ExitCode.VALIDATION_FAILED)) from exc

        if as_json:
            emit_report(view.report)
        render(console, view)
        raise typer.Exit(exit_code_for(view.verdict, fail_on_warning=fail_on_warning))

    # `as_json=False`: `guard`'s own error path writes the command envelope to
    # stdout, which would break the one-document promise above. Every failure
    # this command can produce is reported to a human on stderr instead.
    guard(context, "compare", as_json=False, action=action)


__all__ = ["compare", "exit_code_for"]
