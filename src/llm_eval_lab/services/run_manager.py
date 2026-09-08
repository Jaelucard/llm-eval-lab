"""Background run ownership for the HTTP API: one process, one event loop.

`POST /api/runs` cannot block for the length of a benchmark run, so the API
launches the run as an ``asyncio`` task and answers ``202`` immediately with a
run id the client polls. This module owns those tasks (decision D-RUN).

**The single-worker assumption is load-bearing and is not an implementation
detail.** A launched run lives in the memory of the process that launched it:
its cancel event, its task handle and its progress snapshot are all local. A
second worker process would answer ``POST /api/runs/{id}/cancel`` for a run it
has never heard of, and would answer it with a lie. ``llm-eval serve``
therefore passes ``workers=1``, and :meth:`RunManager.cancel` reports
:data:`CancelOutcome.NOT_MANAGED` rather than pretending, for any run this
process does not hold. Introducing a queue and a second process is exactly the
change this class exists to localise.

**Durability does not depend on this process staying alive.** The engine
persists each case as it finishes and heartbeats ``Run.updated_at`` alongside
it, so a killed process leaves a partially written run rather than nothing. The
lifespan startup sweep marks any run left ``RUNNING`` by a dead process as
``INTERRUPTED``; nothing here needs to reconstruct in-flight state.

**Cancellation is cooperative.** :meth:`cancel` sets the run's
``asyncio.Event``; the engine checks it before starting each remaining case and
finishes the ones already in flight. The task is never hard-cancelled by a
client request, because a hard cancel mid-write is how a half-written case row
happens. Shutdown does hard-cancel, but only after asking politely first and
waiting.

**A third broad catch, alongside the runner's per-case isolation and the
API's opaque-500 boundary.** :meth:`_execute` runs a launched run inside a
detached ``asyncio`` task nobody awaits, so an exception that escaped it would
surface nowhere - no traceback, no caller to report it to, and the run row
stuck ``RUNNING`` forever. Its own ``except Exception`` logs the full
traceback and writes the failure onto the run row instead, which is strictly
more information than letting it vanish; see its docstring for the detail.
Three instances of the same deliberate, bounded exception to this project's
no-broad-catch rule now exist, each independently justified and each
recording everything it catches rather than swallowing anything.
"""

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

import structlog

from llm_eval_lab.models import (
    JSONValue,
    LLMEvalError,
    ProgressEvent,
    Run,
    RunStatus,
    StorageError,
)
from llm_eval_lab.services.catalog_service import CatalogService
from llm_eval_lab.services.run_service import PreparedRun, RunRequest, RunService

SHUTDOWN_GRACE_S = 5.0
"""How long a cooperative shutdown waits before hard-cancelling the tasks."""

MAX_CONCURRENT_RUNS = 16
"""How many runs this process will execute or hold a reservation for, at once.

Not a performance tuning knob. Each launched run is a background task that
spends the operator's money against a vendor, and `POST /api/runs` in a loop
would create them without bound - so the cap exists to make a client's mistake,
or a stolen bearer token, cost a bounded amount. It is per PROCESS because the
tasks are (decision D-RUN), and it sits here rather than in `Settings` because
`settings.py` is a frozen contract file; a deployment-configurable ceiling is a
contract change to propose, not one to make in passing.
"""


class TooManyRunsError(LLMEvalError):
    """This process is already executing as many runs as it will hold at once."""

    def __init__(self, limit: int) -> None:
        """Name the limit that was reached, so the caller knows what to wait for."""
        self.limit = limit
        super().__init__(
            f"this process is already executing {limit} runs, which is its limit; "
            f"wait for one to finish before starting another"
        )


TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.INTERRUPTED,
    }
)
"""Statuses a run does not leave. A cancel request against one is refused."""


class CancelOutcome(StrEnum):
    """What a cancellation request could actually do."""

    REQUESTED = "requested"
    """The cancel event was set; the run will stop after its in-flight cases."""

    ALREADY_TERMINAL = "already_terminal"
    """The run had already finished; there was nothing to stop."""

    NOT_MANAGED = "not_managed"
    """The run exists but this process is not executing it, so it cannot be stopped."""

    NOT_FOUND = "not_found"
    """No run has that id."""


@dataclass(frozen=True)
class LaunchRequest:
    """One API-launched run, described the way an HTTP client is allowed to describe it.

    ``suite`` is a NAME, resolved under the configured benchmark root by
    :class:`~llm_eval_lab.services.catalog_service.CatalogService`. The API
    never accepts a server-side file path, so this is deliberately not a
    ``Path``: the difference between the two types is the whole traversal
    boundary.
    """

    suite: str
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    provider_options: dict[str, JSONValue] = field(default_factory=dict)
    label: str | None = None
    tags: tuple[str, ...] = ()
    select_tags: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    limit: int | None = None
    sample: int | None = None
    sample_seed: int | None = None
    concurrency: int | None = None
    case_timeout_s: float | None = None
    max_cost: Decimal | None = None
    error_policy: Literal["exclude", "fail"] = "exclude"
    seed: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class RunProgress:
    """What ``GET /api/runs/{id}/status`` reports, read from the persisted row.

    The persisted row rather than the in-memory snapshot, because the row is
    the only answer that is still correct for a run this process did not launch
    - one from a previous process, or one launched by the CLI. ``managed`` says
    whether this process holds the task, which is exactly what decides whether
    a cancel request can succeed.
    """

    run_id: str
    status: RunStatus
    completed: int
    total: int
    passed: int
    errors: int
    cancelled: int
    started_at: datetime | None
    updated_at: datetime
    error: str | None
    managed: bool
    last_case_id: str | None
    """The case named by the most recent progress event this process saw.

    Progress events exist only in the memory of the process executing the run,
    so this is ``None`` for any run ``managed`` reports as false, and for a
    managed run whose latest event was run-level rather than case-level. It is a
    display aid for a polling client, never a count: with concurrency above one,
    several cases are in flight and this names the one that most recently
    changed state.
    """


@dataclass
class _Tracked:
    """One run this process is executing, and everything needed to steer it."""

    run_id: str
    cancel: asyncio.Event
    task: asyncio.Task[None] | None = None
    last_event: ProgressEvent | None = None

    def observe(self, event: ProgressEvent) -> None:
        """Record the most recent progress event.

        Synchronous and non-blocking by contract: the runner calls this from
        inside its ``TaskGroup`` and a callback that awaited anything would
        stall the run it is reporting on.
        """
        self.last_event = event


class RunManager:
    """Owns the background tasks executing API-launched runs, and their cancellation."""

    def __init__(
        self,
        *,
        runs: RunService,
        catalog: CatalogService,
        logger: structlog.BoundLogger,
    ) -> None:
        """Bind the manager to the services it composes."""
        self._runs = runs
        self._catalog = catalog
        self._log = logger
        self._tracked: dict[str, _Tracked] = {}
        self._reserved: set[str] = set()

    # -- launching -------------------------------------------------------

    def _request_for(self, launch: LaunchRequest) -> RunRequest:
        """Turn a name-addressed launch request into a path-addressed run request.

        Raises:
            SuiteNameError: when the name escapes the configured root.
            SuiteNotFoundError: when no such suite exists.
        """
        return RunRequest(
            suite_path=self._catalog.suite_path(launch.suite),
            provider=launch.provider,
            model=launch.model,
            api_key_env=launch.api_key_env,
            base_url=launch.base_url,
            provider_options=dict(launch.provider_options),
            label=launch.label,
            tags=launch.tags,
            select_tags=launch.select_tags,
            case_ids=launch.case_ids,
            limit=launch.limit,
            sample=launch.sample,
            sample_seed=launch.sample_seed,
            concurrency=launch.concurrency,
            case_timeout_s=launch.case_timeout_s,
            max_cost=launch.max_cost,
            error_policy=launch.error_policy,
            seed=launch.seed,
            notes=launch.notes,
        )

    async def launch(self, launch: LaunchRequest) -> Run:
        """Validate, persist and start one run, returning before it executes.

        Everything that can be decided synchronously IS decided synchronously,
        before the ``202``: the suite is loaded and validated, the provider name
        is resolved and the price table is read. A caller therefore learns about
        a malformed suite from the response to its own request rather than from
        a status poll thirty seconds later.

        Raises:
            SuiteNameError | SuiteNotFoundError: when the suite name is unusable.
            BenchmarkValidationError: when the suite file is invalid.
            PricingError: when the price table cannot be loaded.
            ProviderNotInstalledError: when the provider name is not registered
                or its optional extra is not installed.
            MissingCredentialError: when the named credential variable is unset.
            TooManyRunsError: when this process is already at its concurrency cap.
            StorageError: when the PENDING run row cannot be written.
        """
        ticket = self._reserve()
        try:
            request = self._request_for(launch)
            # `prepare` reads and parses a YAML file and hashes it, all
            # synchronously. On the event loop that stalls every concurrent status
            # poll for as long as the suite takes to load, so it runs in a worker
            # thread.
            prepared = await asyncio.to_thread(self._runs.prepare, request)
            # Before the row is written, not after: resolving the credential here
            # is what turns an unset variable into a 424 on this request instead
            # of a 202 followed by a run that quietly fails, with the variable's
            # name reaching only the server log.
            self._runs.ensure_credential(prepared.config.provider)
            run = await self._runs.create_run(
                prepared.config,
                prepared.suite,
                prepared.cases,
                label=launch.label,
                tags=launch.tags,
            )
            tracked = _Tracked(run_id=run.id, cancel=asyncio.Event())
            self._tracked[run.id] = tracked
            tracked.task = asyncio.create_task(
                self._execute(run, prepared, tracked), name=f"llm-eval-run:{run.id}"
            )
        finally:
            # Released only once the run is either tracked or definitively not
            # happening, so the slot is accounted for continuously and never twice.
            self._reserved.discard(ticket)
        self._log.info(
            "run_launched",
            run_id=run.id,
            suite=launch.suite,
            provider=launch.provider,
            model=launch.model,
            n_cases=len(prepared.cases),
        )
        return run

    def _reserve(self) -> str:
        """Claim one of this process's run slots, or refuse.

        Synchronous, and the check and the claim are one uninterrupted step. The
        cap used to be tested against ``self._tracked`` alone, and the entry that
        would have grown it was only added three awaits later - after a file was
        read in a worker thread, a credential was resolved and a row was written.
        Forty simultaneous requests all saw an empty dictionary, all passed, and
        all launched: a classic check-then-act race, and one that spends money.

        A reservation is a ticket rather than a counter so that a leak is visible
        in a debugger, and `discard` makes double release harmless.

        Raises:
            TooManyRunsError: when the process is already at its cap.
        """
        if len(self._tracked) + len(self._reserved) >= MAX_CONCURRENT_RUNS:
            raise TooManyRunsError(MAX_CONCURRENT_RUNS)
        ticket = uuid.uuid4().hex
        self._reserved.add(ticket)
        return ticket

    async def _execute(self, run: Run, prepared: PreparedRun, tracked: _Tracked) -> None:
        """Execute one run to completion inside its own task.

        The broad ``except Exception`` here is a deliberate, bounded exception to
        the project's no-broad-catch rule, and it swallows nothing. This is a
        detached task: an exception that escapes it lands in a ``Task`` object
        nobody awaits, so the operator sees no traceback and the run row stays
        ``RUNNING`` until the next process restart sweeps it. The clause logs the
        full traceback and writes the failure onto the run row, which is strictly
        more information than letting it escape, and then stops - the caller that
        would have received the exception returned its ``202`` long ago.
        """
        logger = self._log.bind(run_id=run.id)
        try:
            engine = self._runs.engine(prepared.prices, logger)
            await engine.execute(
                run,
                prepared.cases,
                cancel_event=tracked.cancel,
                progress=tracked.observe,
            )
        except asyncio.CancelledError:
            # A hard cancel, which only shutdown performs. The row is left for the
            # next process's startup sweep to mark INTERRUPTED, which is the
            # honest description of what happened to it.
            logger.warning("run_task_cancelled")
            raise
        except StorageError:
            # The engine already marked the run FAILED on a best-effort basis, and
            # the thing that failed is the database, so a second write would fail
            # too.
            logger.exception("run_storage_failure")
        except LLMEvalError as exc:
            logger.exception("run_failed")
            await self._record_failure(run.id, type(exc).__name__)
        except Exception as exc:
            logger.exception("run_task_crashed")
            await self._record_failure(run.id, type(exc).__name__)
        finally:
            self._tracked.pop(run.id, None)

    async def _record_failure(self, run_id: str, error: str) -> None:
        """Write a failed terminal status, tolerating a database that is also gone."""
        try:
            await self._runs.fail_run(run_id, error=error)
        except StorageError:
            self._log.exception("could_not_record_run_failure", run_id=run_id)

    # -- observing -------------------------------------------------------

    def in_flight(self) -> int:
        """Runs this process is executing or has reserved a slot for."""
        return len(self._tracked) + len(self._reserved)

    def is_managed(self, run_id: str) -> bool:
        """Report whether this process holds the task executing `run_id`."""
        return run_id in self._tracked

    async def status(self, run_id: str) -> RunProgress | None:
        """Report one run's progress, or ``None`` when no run has that id.

        Read from the persisted row. The engine heartbeats ``Run.totals`` and
        ``Run.updated_at`` in the same unit of work that writes each case, so the
        row is never behind the cases it describes, and it is equally correct for
        a run this process did not launch.

        Raises:
            StorageError: when the run cannot be read.
        """
        run = await self._runs.fetch_run(run_id)
        if run is None:
            return None
        tracked = self._tracked.get(run_id)
        latest = tracked.last_event if tracked is not None else None
        return RunProgress(
            run_id=run.id,
            status=run.status,
            completed=run.totals.n_completed,
            total=run.totals.n_cases,
            passed=run.totals.n_passed,
            errors=run.totals.n_errors,
            cancelled=run.totals.n_cancelled,
            started_at=run.started_at,
            updated_at=run.updated_at,
            error=run.error,
            managed=self.is_managed(run_id),
            last_case_id=None if latest is None else latest.case_id,
        )

    # -- cancelling ------------------------------------------------------

    async def cancel(self, run_id: str) -> CancelOutcome:
        """Ask one run to stop after its in-flight cases finish.

        Raises:
            StorageError: when the run cannot be read.
        """
        run = await self._runs.fetch_run(run_id)
        if run is None:
            return CancelOutcome.NOT_FOUND
        if run.status in TERMINAL_STATUSES:
            return CancelOutcome.ALREADY_TERMINAL
        tracked = self._tracked.get(run_id)
        if tracked is None:
            return CancelOutcome.NOT_MANAGED
        tracked.cancel.set()
        self._log.info("run_cancel_requested", run_id=run_id)
        return CancelOutcome.REQUESTED

    async def shutdown(self) -> None:
        """Stop every managed run, cooperatively first and forcibly second.

        Every cancel event is set, then the tasks are given
        :data:`SHUTDOWN_GRACE_S` to finish the cases already in flight and write
        their terminal status. Anything still running after that is hard
        cancelled: the process is going away regardless, and a run left mid-write
        is recovered by the next process's startup sweep.

        ``_reserved`` is cleared on EVERY path, including the early return
        below. A launch in progress claims a reservation before anything is
        tracked (see :meth:`_reserve`); a shutdown landing in that window once
        saw no pending TASK, returned immediately, and left the ticket in
        ``_reserved`` forever - a slot leaked for the rest of the process's
        life every time shutdown raced a launch.
        """
        tasks = self._pending_tasks()
        if not tasks:
            self._reserved.clear()
            return
        for tracked in list(self._tracked.values()):
            tracked.cancel.set()
        self._log.info("run_manager_shutdown", n_runs=len(tasks))
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(SHUTDOWN_GRACE_S):
                await asyncio.gather(*tasks, return_exceptions=True)
        remaining = self._pending_tasks()
        for task in remaining:
            task.cancel()
        if remaining:
            self._log.warning("run_manager_forced_cancel", n_runs=len(remaining))
            await asyncio.gather(*remaining, return_exceptions=True)
        self._tracked.clear()
        self._reserved.clear()

    def _pending_tasks(self) -> tuple[asyncio.Task[None], ...]:
        """Return every tracked task that has not finished."""
        return tuple(
            item.task
            for item in list(self._tracked.values())
            if item.task is not None and not item.task.done()
        )


__all__ = [
    "SHUTDOWN_GRACE_S",
    "CancelOutcome",
    "LaunchRequest",
    "RunManager",
    "RunProgress",
]
