# ruff: noqa: FBT002
# Typer derives a `--flag/--no-flag` option from a boolean PARAMETER, so every
# CLI flag in this module is necessarily a boolean argument with a default.
"""``llm-eval run`` - execute a benchmark against a model.

Two spend guards live here, and both are load-bearing.

``--dry-run`` reports the effective configuration, the selected case count and a
cost estimate, then exits 0 WITHOUT constructing a provider client or issuing a
single request. Nothing is written to the database either, so a dry run against
the wrong model costs nothing and leaves no trace.

``--max-cost`` is a running budget rather than a pre-flight check. The runner
adds each completed case's KNOWN cost to a total and cancels the run when the
total passes the cap, leaving the cases that already completed persisted and
inspectable. Unpriced cases contribute nothing and are reported separately, so a
budget is never enforced against a number the system does not know.

Interruption: the first ``SIGINT`` requests a graceful cancellation and lets the
in-flight cases finish, so their results are persisted; a second one is left to
Python's default handler, which is how an operator forces the issue.
"""

import asyncio
import json
import signal
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console

from llm_eval_lab.cli.output import emit_json, key_value_table, stderr_console
from llm_eval_lab.models import ProgressCallback, ProgressEvent, ProgressEventType, RunStatus
from llm_eval_lab.observability.logging import get_logger
from llm_eval_lab.reporting.formatters import markup_safe
from llm_eval_lab.services import (
    BUDGET_EXCEEDED,
    DryRunPlan,
    RunRequest,
    build_run_service,
    open_unit_of_work_factory,
    run_summary_payload,
)

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from llm_eval_lab.cli.main import AppContext

INTERRUPTED_STATUSES = frozenset({RunStatus.CANCELLED, RunStatus.INTERRUPTED})
"""Run statuses that make `llm-eval run` exit 130 rather than 0.

`RunStatus.CANCELLED` is the one exception: a budget-triggered cancellation
carries this same status but no signal was ever delivered to this process, so
`exit_code_for` checks `Run.error` FIRST and reports that case as exit 5
instead - see its own docstring.
"""

INCOMPLETE_STATUSES = frozenset({RunStatus.PARTIAL, RunStatus.FAILED})
"""Run statuses that make `llm-eval run` exit 5, the CI 'run incomplete' code."""


def parse_max_cost(raw: str | None) -> Decimal | None:
    """Parse ``--max-cost`` into an exact Decimal.

    Parsed from the string form rather than through a float, so
    ``--max-cost 0.10`` means ten cents and not 0.1000000000000000055511151231.

    Raises:
        typer.BadParameter: when the value is not a non-negative decimal.
    """
    if raw is None:
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        msg = f"--max-cost must be a decimal number, got {raw!r}"
        raise typer.BadParameter(msg) from exc
    if value < 0:
        msg = f"--max-cost must not be negative, got {raw!r}"
        raise typer.BadParameter(msg)
    return value


def parse_provider_option(raw: str) -> tuple[str, Any]:
    """Parse one ``--provider-option key=value`` pair.

    The value is read as JSON when it parses as JSON, so ``retries=3`` is an
    integer and ``sleep=true`` is a boolean, and is otherwise kept as the literal
    string. Without that, every option would arrive as text and a provider's
    typed options model would reject perfectly reasonable input.

    Raises:
        typer.BadParameter: when the pair has no ``=``.
    """
    key, separator, value = raw.partition("=")
    if not separator or not key:
        msg = f"--provider-option must be KEY=VALUE, got {raw!r}"
        raise typer.BadParameter(msg)
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def build_provider_options(
    fake_mode: str | None,
    fake_seed: int | None,
    pairs: list[str] | None = None,
    mutate_rate: float | None = None,
) -> dict[str, Any]:
    """Collect the provider-specific options the CLI exposes.

    An explicit ``--provider-option`` still wins: it is applied last, so a
    caller can reach any option the fake provider has, including the three that
    also have named flags.
    """
    options: dict[str, Any] = {}
    if fake_mode is not None:
        options["mode"] = fake_mode
    if fake_seed is not None:
        options["seed"] = fake_seed
    if mutate_rate is not None:
        options["mutate_rate"] = mutate_rate
    for raw in pairs or ():
        key, value = parse_provider_option(raw)
        options[key] = value
    return options


def print_dry_run(plan: DryRunPlan, *, color: bool) -> None:
    """Render a dry-run plan for a human."""
    console = stderr_console(color=color)
    config = plan.config
    console.print(
        key_value_table(
            "effective configuration (dry run)",
            {
                "suite": f"{config.suite_name} v{config.suite_version}",
                "suite_hash": config.suite_hash,
                "provider": config.provider.provider,
                "model": config.provider.model,
                "api_key_env": config.provider.api_key_env,
                "cases selected": plan.n_cases,
                "concurrency": config.concurrency,
                "case_timeout_s": config.case_timeout_s,
                "max_cost": config.max_cost,
                "error_policy": config.error_policy,
                "price_table": f"{config.price_table_id} v{config.price_table_version}",
                "estimated input tokens": plan.estimated_input_tokens,
                "estimated output tokens": plan.estimated_output_tokens,
                "estimated cost": plan.estimated_cost
                if plan.estimated_cost is not None
                else f"unpriced ({plan.estimate_unpriced_reason})",
            },
        )
    )
    for assumption in plan.assumptions:
        console.print(f"  [dim]assumption:[/dim] {markup_safe(assumption)}")


def dry_run_payload(plan: DryRunPlan) -> dict[str, Any]:
    """Build the JSON document a dry run emits."""
    return {
        "ok": True,
        "dry_run": True,
        "config": plan.config.model_dump(mode="json"),
        "n_cases": plan.n_cases,
        "case_ids": list(plan.case_ids),
        "estimated_input_tokens": plan.estimated_input_tokens,
        "estimated_output_tokens": plan.estimated_output_tokens,
        "estimated_cost": plan.estimated_cost,
        "estimate_unpriced_reason": plan.estimate_unpriced_reason,
        "assumptions": list(plan.assumptions),
    }


def install_sigint_handler(cancel: asyncio.Event) -> None:
    """Make the first SIGINT request a graceful cancellation.

    A second SIGINT is not handled here: the handler removes itself, so the
    default behaviour returns and an operator who really wants out gets out.
    """
    loop = asyncio.get_running_loop()
    console = stderr_console()

    def _request_cancel() -> None:
        console.print("[yellow]cancelling: letting in-flight cases finish[/yellow]")
        cancel.set()
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - platform dependent
            return

    try:
        loop.add_signal_handler(signal.SIGINT, _request_cancel)
    except (NotImplementedError, RuntimeError):  # pragma: no cover - platform dependent
        return


def make_progress_printer(console: Console) -> ProgressCallback:
    """Build the progress callback that renders events on stderr, never on stdout."""

    def on_progress(event: ProgressEvent) -> None:
        if event.type is ProgressEventType.RUN_STARTED:
            console.print(f"[dim]running {event.total} cases[/dim]")
        elif event.type in {ProgressEventType.CASE_COMPLETED, ProgressEventType.CASE_FAILED}:
            state = "failed" if event.type is ProgressEventType.CASE_FAILED else "ok"
            console.print(
                f"[dim]{event.completed}/{event.total}[/dim] {markup_safe(event.case_id)} ({state})"
            )
        elif event.type is ProgressEventType.CASE_RETRY:
            delay = "" if event.retry_in_s is None else f" in {event.retry_in_s:.2f}s"
            console.print(
                f"[yellow]retry[/yellow] {markup_safe(event.case_id)} "
                f"attempt {event.attempt}{delay}"
            )

    return on_progress


def exit_code_for(status: RunStatus, *, error: str | None) -> int:
    """Map a finished run's status onto the CLI exit-code table.

    `error` breaks the one ambiguous case: `RunStatus.CANCELLED` is shared by
    two different events. A ``SIGINT`` cancellation means this process
    received a signal, which is exactly what exit 130 documents. A
    ``--max-cost`` cancellation (`Run.error == BUDGET_EXCEEDED`) received no
    signal at all - the run simply did not get to finish, for a reason that is
    not the run's fault, the same shape as `PARTIAL` or `FAILED` - so it is
    reported the same way those are, at exit 5, and 130 is reserved for the
    one case that is actually true of it.
    """
    from llm_eval_lab.cli.main import ExitCode  # noqa: PLC0415 - avoids a cycle

    if status is RunStatus.CANCELLED and error == BUDGET_EXCEEDED:
        return int(ExitCode.RUN_INCOMPLETE)
    if status in INTERRUPTED_STATUSES:
        return int(ExitCode.INTERRUPTED)
    if status in INCOMPLETE_STATUSES:
        return int(ExitCode.RUN_INCOMPLETE)
    return int(ExitCode.SUCCESS)


async def execute_run(
    context: "AppContext",
    request: RunRequest,
    *,
    as_json: bool,
    console: Console,
) -> int:
    """Open persistence, execute the run, report it, and return the exit code."""
    logger = get_logger("llm_eval_lab.cli.run")
    async with open_unit_of_work_factory(
        context.database_url, max_raw_bytes=context.settings.max_raw_bytes
    ) as uow_factory:
        service = build_run_service(context.settings, uow_factory, logger)
        cancel = asyncio.Event()
        install_sigint_handler(cancel)
        outcome = await service.execute(
            request, cancel_event=cancel, progress=make_progress_printer(console)
        )

    payload = dict(run_summary_payload(outcome))
    if as_json:
        emit_json("run", payload)
    else:
        console.print(key_value_table(f"run {outcome.run.id}", dict(payload)))
    return exit_code_for(outcome.run.status, error=outcome.run.error)


def execute_dry_run(context: "AppContext", request: RunRequest, *, as_json: bool) -> None:
    """Report what a run would do, without opening the database or a provider."""
    service = build_run_service(
        context.settings,
        refuse_persistence,
        get_logger("llm_eval_lab.cli.run"),
    )
    plan = service.dry_run(request)
    if as_json:
        emit_json("run", dry_run_payload(plan))
    else:
        print_dry_run(plan, color=context.color)


def run(  # noqa: PLR0913, PLR0917 - one parameter per documented CLI option - one parameter per documented CLI option
    ctx: typer.Context,
    suite: Annotated[Path, typer.Argument(help="Path to the benchmark YAML or JSON file.")],
    provider: Annotated[str, typer.Option("--provider", help="Provider name.")],
    model: Annotated[str, typer.Option("--model", help="Model identifier.")],
    api_key_env: Annotated[
        str | None,
        typer.Option("--api-key-env", help="Name of the environment variable holding the key."),
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Override the provider endpoint.")
    ] = None,
    fake_mode: Annotated[
        str | None,
        typer.Option("--fake-mode", help="derived, scripted, echo, expected or mutate."),
    ] = None,
    fake_seed: Annotated[
        int | None, typer.Option("--fake-seed", help="Seed for the fake provider.")
    ] = None,
    mutate_rate: Annotated[
        float | None,
        typer.Option(
            "--mutate-rate",
            help="Fraction of cases the fake provider's `mutate` mode degrades.",
        ),
    ] = None,
    provider_option: Annotated[
        list[str] | None,
        typer.Option(
            "--provider-option",
            help="Provider-specific KEY=VALUE option. Repeatable. Values are read as JSON.",
        ),
    ] = None,
    label: Annotated[str | None, typer.Option("--label", help="Human label for this run.")] = None,
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Tag to attach to the run. Repeatable.")
    ] = None,
    select_tag: Annotated[
        list[str] | None,
        typer.Option("--select-tag", help="Only run cases carrying this tag. Repeatable."),
    ] = None,
    case: Annotated[
        list[str] | None, typer.Option("--case", help="Only run this case id. Repeatable.")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Run at most N cases.")] = None,
    sample: Annotated[int | None, typer.Option("--sample", help="Sample N cases.")] = None,
    sample_seed: Annotated[
        int | None, typer.Option("--sample-seed", help="Seed for --sample.")
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", help="In-flight cases.")
    ] = None,
    timeout: Annotated[
        float | None, typer.Option("--timeout", help="Per-case wall-clock budget in seconds.")
    ] = None,
    max_cost: Annotated[
        str | None,
        typer.Option("--max-cost", help="Cancel the run once known spend exceeds this."),
    ] = None,
    error_policy: Annotated[
        str, typer.Option("--error-policy", help="exclude or fail.")
    ] = "exclude",
    seed: Annotated[int | None, typer.Option("--seed", help="Recorded run seed.")] = None,
    prices: Annotated[
        Path | None, typer.Option("--prices", help="Price table to cost this run against.")
    ] = None,
    notes: Annotated[str | None, typer.Option("--notes", help="Free-text note on the run.")] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report what would run, issue nothing, write nothing."),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON on stdout.")] = False,
) -> None:
    """Run a benchmark suite against a model."""
    from llm_eval_lab.cli.main import (  # noqa: PLC0415 - avoids a cycle
        ExitCode,
        app_context,
        guard,
        run_async,
    )

    context = app_context(ctx)
    console = stderr_console(color=context.color)
    if error_policy not in {"exclude", "fail"}:
        msg = f"--error-policy must be 'exclude' or 'fail', got {error_policy!r}"
        raise typer.BadParameter(msg)

    request = RunRequest(
        suite_path=suite,
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        provider_options=build_provider_options(fake_mode, fake_seed, provider_option, mutate_rate),
        label=label,
        tags=tuple(tag or ()),
        select_tags=tuple(select_tag or ()),
        case_ids=tuple(case or ()),
        limit=limit,
        sample=sample,
        sample_seed=sample_seed,
        concurrency=concurrency,
        case_timeout_s=timeout,
        max_cost=parse_max_cost(max_cost),
        error_policy="fail" if error_policy == "fail" else "exclude",
        seed=seed,
        price_table_path=prices,
        notes=notes,
        plugins=context.plugins,
    )

    def action() -> None:
        if dry_run:
            execute_dry_run(context, request, as_json=as_json)
            return
        code = run_async(execute_run(context, request, as_json=as_json, console=console))
        if code != int(ExitCode.SUCCESS):
            raise typer.Exit(code)

    guard(context, "run", as_json=as_json, action=action)


def refuse_persistence() -> Any:
    """Stand in for the unit-of-work factory during a dry run.

    A dry run must not touch the database, and the clearest way to guarantee
    that is to hand the service a factory that raises if anything calls it,
    rather than a real one nobody is supposed to use.

    Raises:
        RuntimeError: always. Reaching this is a bug in the dry-run path.
    """
    msg = "a dry run must not open the database"
    raise RuntimeError(msg)
