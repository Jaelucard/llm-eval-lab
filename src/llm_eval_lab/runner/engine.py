"""The run engine: one `TaskGroup`, bounded concurrency, per-case isolation.

Everything about how a run behaves under failure is decided here.

**Partial failure is the normal case.** A benchmark run against a real vendor
will have cases that time out, cases that get throttled and cases that come back
malformed. One bad case must not cost the other ninety-nine, so every case runs
inside :meth:`RunEngine._run_case_isolated`, which catches in a fixed order:
``asyncio.CancelledError`` is re-raised (a cancellation is not a failure),
then ``TimeoutError``, then ``ProviderError``, then ``EvaluatorError``, and
finally a broad ``except Exception`` that logs with a traceback, records
``ProviderErrorKind.UNKNOWN`` plus the exception's class name, and returns a
``CaseResult``. That last clause is the only broad catch in the PER-CASE
request/execution path, and it is permitted because it records everything it
catches rather than swallowing it. (A structurally identical, separately
documented exception exists for the API's opaque-500 boundary and for
:meth:`~llm_eval_lab.services.run_manager.RunManager._execute`'s detached
background task - see that module's own docstring - which is why this is
scoped to "this path" rather than claimed as the only one anywhere.)

**`StorageError` is deliberately NOT isolated.** A database that cannot be
written to is not a bad case, it is a broken run: it aborts the whole run with
``RunStatus.FAILED``. Isolating it would produce a run that looks complete and
has no rows behind it.

**Persistence is incremental.** Each case is written as it finishes, in its own
unit of work, so a crash leaves a resumable run rather than nothing at all.

**The budget is a running total of KNOWN spend.** Unpriced cases contribute
nothing and are counted separately, so ``--max-cost`` can never be enforced
against a number the system does not actually know.
"""

import asyncio
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from decimal import Decimal

import structlog

from llm_eval_lab.evaluators.base import apply_on_error
from llm_eval_lab.evaluators.registry import EvaluatorRegistry
from llm_eval_lab.models import (
    CaseResult,
    CaseStatus,
    CostBreakdown,
    EvaluationContext,
    EvaluationRecord,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorError,
    EvaluatorSettings,
    ModelResponse,
    PriceTable,
    ProgressCallback,
    Provider,
    ProviderConfig,
    ProviderError,
    ProviderErrorInfo,
    ProviderErrorKind,
    ProviderRequest,
    ResolvedCase,
    Run,
    RunStatus,
    RunTotals,
    StorageError,
    UnitOfWorkFactory,
)
from llm_eval_lab.pricing.calculator import UNPRICED_NO_RESPONSE, billable_total, compute_cost
from llm_eval_lab.providers import (
    ATTEMPT_METADATA_KEY,
    CASE_ID_METADATA_KEY,
    EXPECTED_METADATA_KEY,
    RUN_ID_METADATA_KEY,
)
from llm_eval_lab.reporting.aggregate import metrics_for_run
from llm_eval_lab.runner.progress import ProgressEmitter
from llm_eval_lab.runner.ratelimit import RateLimitGate
from llm_eval_lab.runner.retry import next_retry
from llm_eval_lab.utils.json import canonical_json
from llm_eval_lab.utils.time import monotonic_ms, utc_now

type ProviderFactory = Callable[[ProviderConfig], AbstractAsyncContextManager[Provider]]
"""Opens a provider for a configuration and closes it when the scope exits."""

BUDGET_EXCEEDED = "budget_exceeded"
"""The exact value written to `Run.error` when a run is stopped by `--max-cost`."""

MAX_ERROR_RATE_EXCEEDED = "max_error_rate_exceeded"
"""The exact value written to `Run.error` when too many cases failed."""

SELF_PREFERENCE_RISK = "self_preference_risk"
"""`Run.warnings` entry: some evaluation's judge model equals the candidate model."""

HIGH_JUDGE_VARIANCE = "high_judge_variance"
"""`Run.warnings` entry: some evaluation's judge dispersion exceeded its threshold."""

JUDGE_DISAGREEMENT = "judge_disagreement"
"""`Run.warnings` entry: some evaluation's judges (a panel) disagreed with each other."""


def _first_storage_error(group: ExceptionGroup[StorageError]) -> StorageError:
    """Pull one `StorageError` out of a `TaskGroup`'s exception group.

    Several cases can fail the same write at once; they are the same fault seen
    from several tasks, so the first is reported and the rest are its siblings.
    """
    for item in group.exceptions:
        if isinstance(item, StorageError):
            return item
        if isinstance(item, ExceptionGroup):
            return _first_storage_error(item)
    # Unreachable: `except*` only routes a group here when it holds a match.
    msg = "exception group contained no StorageError"
    raise AssertionError(msg)


@dataclass
class _Attempts:
    """Mutable attempt bookkeeping, so a raised error still reports its attempt count."""

    count: int = 1
    total_delay_s: float = 0.0


@dataclass
class _Budget:
    """Running total of KNOWN spend, plus how many cases could not be priced."""

    cap: Decimal | None
    spent: Decimal = Decimal(0)
    unpriced: int = 0

    def add(self, cost: CostBreakdown | None) -> None:
        """Fold one case's cost into the running total."""
        if cost is None or not cost.priced:
            self.unpriced += 1
            return
        self.spent += billable_total(cost)

    @property
    def exceeded(self) -> bool:
        """Report whether known spend has passed the cap."""
        return self.cap is not None and self.spent > self.cap


@dataclass
class _Counters:
    """Run-level counters maintained as cases complete.

    ``n_completed`` is the run's total, seeded from already-persisted cases when
    resuming, and is what reaches ``RunTotals``. ``n_attempted`` counts only the
    cases THIS process ran, and is the denominator of the error rate: dividing
    by ``n_completed`` on a resumed run would spread this process's errors over
    cases it never attempted.
    """

    n_cases: int
    n_completed: int = 0
    n_attempted: int = 0
    n_passed: int = 0
    n_errors: int = 0
    n_cancelled: int = 0
    statuses: list[CaseStatus] = field(default_factory=list)

    def totals(self) -> RunTotals:
        """Project the counters into the contract's `RunTotals`."""
        return RunTotals(
            n_cases=self.n_cases,
            n_completed=self.n_completed,
            n_passed=self.n_passed,
            n_errors=self.n_errors,
            n_cancelled=self.n_cancelled,
        )


@dataclass(frozen=True)
class RunOutcome:
    """What a completed run produced, returned to the caller as one value."""

    run: Run
    results: tuple[CaseResult, ...]
    budget_spent: Decimal
    unpriced_cases: int


def case_status_for(evaluations: Sequence[EvaluationResult]) -> CaseStatus:
    """Decide a case's status once a response has arrived.

    A case is ERRORED when any of its evaluations could not reach a conclusion,
    not only when the provider call failed. Without this, a suite whose every
    evaluator timed out reported ``n_errors: 0``, finished ``COMPLETED`` and
    exited 0 - so a CI gate keyed on the exit code passed a benchmark in which
    nothing was scored. The evaluator failing is exactly as much a failure of
    the run as the model call failing; only the blame differs, and ``status``
    versus ``passed`` is what records the difference.

    An errored case is excluded from the pass-rate denominator under the default
    ``error_policy="exclude"`` and counted in ``n_errors``, which is over total
    attempted.
    """
    if any(item.status is EvaluationStatus.ERROR for item in evaluations):
        return CaseStatus.ERROR
    return CaseStatus.OK


def aggregate_case_verdict(
    evaluations: Sequence[EvaluationResult],
) -> tuple[bool | None, float | None]:
    """Reduce a case's evaluations to one pass/fail and one score.

    Pass/fail keys off ``passed``, never off ``status``: a score-only evaluator
    reports ``status=PASSED, passed=None`` and must not be read as a failure.
    A case whose evaluators all produced ``passed=None`` therefore has
    ``passed=None`` too, which is the honest answer and is what the aggregate
    counts as score-only.

    The score is the weight-weighted mean of every evaluation that produced one,
    including non-required evaluators: a score is information regardless of
    whether it gates.
    """
    gating = [item for item in evaluations if item.required and item.passed is not None]
    passed = all(item.passed for item in gating) if gating else None

    scored = [item for item in evaluations if item.score is not None]
    total_weight = sum(item.weight for item in scored)
    if not scored or total_weight <= 0:
        return passed, None
    weighted = sum((item.score or 0.0) * item.weight for item in scored)
    return passed, weighted / total_weight


def run_warnings(results: Sequence[CaseResult]) -> tuple[str, ...]:
    """Derive `Run.warnings` from every persisted evaluation's judge advisories.

    Per the frozen-contract addendum, ``self_preference_risk`` and
    ``high_judge_variance`` must be "visible, never silent, never blocking" at
    the RUN level, not only inside each case's own `JudgeProvenance` - a reader
    looking only at `Run.warnings` must never see a clean run when even one
    evaluation anywhere in it flagged a risk. ``judge_disagreement`` lives in
    `EvaluationResult.metadata` rather than on `JudgeProvenance` itself (a
    frozen model that cannot grow fields), so it is read from there.

    Scans ``results`` rather than only the cases this process attempted, so a
    resumed run's warnings reflect ALL of its persisted evaluations, the same
    way its metrics rollup does.
    """
    seen = dict.fromkeys((SELF_PREFERENCE_RISK, HIGH_JUDGE_VARIANCE, JUDGE_DISAGREEMENT), False)
    for result in results:
        for evaluation in result.evaluations:
            provenance = evaluation.judge
            if provenance is not None:
                if provenance.self_preference_risk:
                    seen[SELF_PREFERENCE_RISK] = True
                if provenance.high_judge_variance:
                    seen[HIGH_JUDGE_VARIANCE] = True
            if evaluation.metadata.get("judge_disagreement") is True:
                seen[JUDGE_DISAGREEMENT] = True
    return tuple(warning for warning, flagged in seen.items() if flagged)


class RunEngine:
    """Executes one resolved suite against one provider, persisting as it goes."""

    def __init__(  # noqa: PLR0913 - every collaborator is injected; none is discovered
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        provider_factory: ProviderFactory,
        evaluators: EvaluatorRegistry,
        price_table: PriceTable,
        evaluator_settings: EvaluatorSettings,
        logger: structlog.BoundLogger,
    ) -> None:
        """Bind the engine to its collaborators."""
        self._uow_factory = uow_factory
        self._provider_factory = provider_factory
        self._evaluators = evaluators
        self._price_table = price_table
        self._evaluator_settings = evaluator_settings
        self._log = logger

    # -- request construction --------------------------------------------

    @staticmethod
    def _expected_text(case: ResolvedCase) -> str | None:
        """Render a case's expected answer as text for the request metadata.

        Only the fake provider reads this. A provider talking to a vendor must
        never transmit request metadata; see ``providers.base``.
        """
        if case.expected is None:
            return None
        if isinstance(case.expected, str):
            return case.expected
        if isinstance(case.expected, bool):
            return "true" if case.expected else "false"
        if isinstance(case.expected, (int, float)):
            return str(case.expected)
        return canonical_json(case.expected)

    def _request_for(self, run: Run, case: ResolvedCase, attempt: int) -> ProviderRequest:
        """Build the provider request for one attempt at one case."""
        metadata = {
            CASE_ID_METADATA_KEY: case.id,
            RUN_ID_METADATA_KEY: run.id,
            ATTEMPT_METADATA_KEY: str(attempt),
        }
        expected = self._expected_text(case)
        if expected is not None:
            metadata[EXPECTED_METADATA_KEY] = expected
        return ProviderRequest(
            messages=case.messages,
            system=case.system,
            params=case.params,
            request_id=case.id,
            metadata=metadata,
        )

    # -- generation ------------------------------------------------------

    async def _generate(  # noqa: PLR0913 - the retry loop needs all of its collaborators
        self,
        run: Run,
        case: ResolvedCase,
        provider: Provider,
        *,
        attempts: _Attempts,
        gate: RateLimitGate,
        emitter: ProgressEmitter,
    ) -> ModelResponse:
        """Generate one response, retrying according to the run's retry policy.

        Raises:
            ProviderError: when the attempt budget or the retry budget runs out,
                or the failure is not retryable. `attempts` carries the count
                that was reached, so the caller can record it accurately.
        """
        policy = run.config.retry
        started_ms = monotonic_ms()
        while True:
            request = self._request_for(run, case, attempts.count)
            try:
                response = await provider.generate(request)
            except ProviderError as exc:
                if exc.kind is ProviderErrorKind.RATE_LIMIT:
                    cooldown = gate.close_for(exc.retry_after_s)
                    self._log.warning(
                        "rate_limited", case_id=case.id, cooldown_s=cooldown, provider=exc.provider
                    )
                decision = next_retry(
                    policy,
                    exc,
                    attempt=attempts.count,
                    elapsed_delay_s=attempts.total_delay_s,
                )
                if not decision.retry:
                    raise
                emitter.case_retry(
                    case.id,
                    attempt=attempts.count,
                    retry_in_s=decision.delay_s,
                    reason=exc.kind.value,
                )
                self._log.info(
                    "retrying_case",
                    case_id=case.id,
                    attempt=attempts.count,
                    retry_in_s=decision.delay_s,
                    kind=exc.kind.value,
                )
                await asyncio.sleep(decision.delay_s)
                attempts.total_delay_s += decision.delay_s
                attempts.count += 1
                await gate.wait()
                continue
            return response.model_copy(
                update={
                    "attempts": attempts.count,
                    "total_latency_ms": monotonic_ms() - started_ms,
                }
            )

    # -- evaluation ------------------------------------------------------

    async def _evaluate(
        self,
        run: Run,
        case: ResolvedCase,
        response: ModelResponse,
        emitter: ProgressEmitter,
    ) -> tuple[EvaluationResult, ...]:
        """Run every evaluator the case declares, in declaration order.

        Raises:
            EvaluatorError: when an evaluator cannot be constructed at all. That
                is a configuration fault, not a scoring outcome, and it is
                recorded against the case by the isolation boundary.
        """
        deadline = asyncio.get_running_loop().time() + run.config.case_timeout_s
        context = EvaluationContext(
            run_id=run.id,
            suite_hash=run.config.suite_hash,
            case_hash=case.case_hash,
            provider_factory=self._provider_factory,
            deadline=deadline,
            logger=self._log,
            settings=self._evaluator_settings,
        )
        results: list[EvaluationResult] = []
        for spec in case.evaluators:
            evaluator = self._evaluators.create(spec)
            result = apply_on_error(await evaluator.evaluate(case, response, context), spec)
            results.append(result)
            emitter.evaluation_completed(case.id, result.evaluator_id)
        return tuple(results)

    # -- one case --------------------------------------------------------

    def _judge_cost(self, evaluations: Sequence[EvaluationResult]) -> Decimal | None:
        """Price every judge's recorded token usage against the run's price table.

        Judge spend is a distinct line item from candidate spend (Phase 3
        ruling; see ``evaluators.judge.evaluator``, whose docstring pins the
        pricing responsibility to this layer). Priced per judge PROVENANCE,
        against the model it actually used: ``JudgeProvenance.usage`` is
        already summed across every call one evaluator made, so a provenance
        naming more than one judge model - a panel - cannot be attributed to a
        single price and contributes nothing, the same "unknown adds nothing"
        rule :func:`~llm_eval_lab.pricing.calculator.billable_total` applies to
        candidate spend, rather than guessing which model's rate to use.

        Returns ``None`` when no evaluation carries judge provenance, or when
        none of it could be priced, so a non-judge case's cost is unaffected.
        """
        provenances = [item.judge for item in evaluations if item.judge is not None]
        if not provenances:
            return None
        total = Decimal(0)
        priced_any = False
        for provenance in provenances:
            if len(provenance.judge_models) != 1:
                continue
            provider, _, model = provenance.judge_models[0].partition(":")
            priced = compute_cost(
                self._price_table, provider=provider, model=model, usage=provenance.usage
            )
            if not priced.priced or priced.total_cost is None:
                continue
            total += priced.total_cost
            priced_any = True
        return total if priced_any else None

    def _cost_for(
        self,
        run: Run,
        response: ModelResponse | None,
        evaluations: Sequence[EvaluationResult] = (),
    ) -> CostBreakdown:
        """Price one case, or record why it could not be priced.

        Judge usage found in ``evaluations`` is priced too and recorded on
        ``CostBreakdown.judge_cost`` - never folded into ``total_cost``, which
        stays candidate-model spend alone.
        """
        judge_cost = self._judge_cost(evaluations)
        if response is None:
            return CostBreakdown(
                currency=self._price_table.currency,
                input_cost=None,
                output_cost=None,
                total_cost=None,
                judge_cost=judge_cost,
                priced=False,
                unpriced_reason=UNPRICED_NO_RESPONSE,
                price_table_id=self._price_table.id,
                price_table_version=self._price_table.version,
            )
        return compute_cost(
            self._price_table,
            provider=run.config.provider.provider,
            model=run.config.provider.model,
            usage=response.usage,
            judge_cost=judge_cost,
        )

    def _error_result(  # noqa: PLR0913 - one parameter per recorded field
        self,
        run: Run,
        case: ResolvedCase,
        *,
        kind: ProviderErrorKind,
        message: str,
        exception_type: str | None,
        status: CaseStatus,
        attempts: int,
        started_at: object,
    ) -> CaseResult:
        """Build the CaseResult recorded when a case failed to produce a response."""
        info = ProviderErrorInfo(
            kind=kind,
            message=message,
            provider=run.config.provider.provider,
            attempts=attempts,
            exception_type=exception_type,
        )
        return CaseResult.model_validate(
            {
                "run_id": run.id,
                "case_id": case.id,
                "case_hash": case.case_hash,
                "status": status,
                "response": None,
                "evaluations": (),
                "passed": None,
                "score": None,
                "cost": self._cost_for(run, None),
                "attempts": attempts,
                "started_at": started_at,
                "completed_at": utc_now(),
                "error": info,
                "category": case.category,
                "tags": case.tags,
            }
        )

    async def _execute_case(  # noqa: PLR0913 - the case body needs all of its collaborators
        self,
        run: Run,
        case: ResolvedCase,
        provider: Provider,
        *,
        attempts: _Attempts,
        gate: RateLimitGate,
        emitter: ProgressEmitter,
    ) -> CaseResult:
        """Generate and evaluate one case, under one wall-clock budget.

        The timeout covers generation AND evaluation, because a case that spent
        its whole budget inside a judge call has still used the whole budget.
        """
        started_at = utc_now()
        async with asyncio.timeout(run.config.case_timeout_s):
            response = await self._generate(
                run, case, provider, attempts=attempts, gate=gate, emitter=emitter
            )
            evaluations = await self._evaluate(run, case, response, emitter)

        passed, score = aggregate_case_verdict(evaluations)
        return CaseResult.model_validate(
            {
                "run_id": run.id,
                "case_id": case.id,
                "case_hash": case.case_hash,
                "status": case_status_for(evaluations),
                "response": response,
                "evaluations": evaluations,
                "passed": passed,
                "score": score,
                "cost": self._cost_for(run, response, evaluations),
                "attempts": attempts.count,
                "started_at": started_at,
                "completed_at": utc_now(),
                "error": None,
                "category": case.category,
                "tags": case.tags,
            }
        )

    async def _run_case_isolated(
        self,
        run: Run,
        case: ResolvedCase,
        provider: Provider,
        *,
        gate: RateLimitGate,
        emitter: ProgressEmitter,
    ) -> CaseResult:
        """Run one case, converting every failure into a recorded `CaseResult`.

        The catch order is fixed and each clause is deliberate. `StorageError`
        is absent on purpose: it propagates and aborts the run.
        """
        attempts = _Attempts()
        started_at = utc_now()
        try:
            return await self._execute_case(
                run, case, provider, attempts=attempts, gate=gate, emitter=emitter
            )
        except asyncio.CancelledError:
            # A cancellation is a request to stop, not a failure to record.
            raise
        except TimeoutError:
            self._log.warning("case_timeout", case_id=case.id, timeout_s=run.config.case_timeout_s)
            return self._error_result(
                run,
                case,
                kind=ProviderErrorKind.TIMEOUT,
                message=f"case exceeded its {run.config.case_timeout_s}s budget",
                exception_type="TimeoutError",
                status=CaseStatus.TIMEOUT,
                attempts=attempts.count,
                started_at=started_at,
            )
        except ProviderError as exc:
            self._log.warning("case_provider_error", case_id=case.id, kind=exc.kind.value)
            return self._error_result(
                run,
                case,
                kind=exc.kind,
                message=exc.message,
                exception_type=type(exc).__name__,
                status=CaseStatus.ERROR,
                attempts=attempts.count,
                started_at=started_at,
            )
        except EvaluatorError as exc:
            self._log.warning("case_evaluator_error", case_id=case.id, error=str(exc))
            return self._error_result(
                run,
                case,
                kind=ProviderErrorKind.UNKNOWN,
                message=str(exc),
                exception_type=type(exc).__name__,
                status=CaseStatus.ERROR,
                attempts=attempts.count,
                started_at=started_at,
            )
        except Exception as exc:  # noqa: BLE001 - the one permitted broad catch; see the docstring
            # Per-case isolation with FULL recording: the traceback is logged,
            # the exception class name is persisted, and the case becomes an
            # errored result rather than an aborted run. Nothing is swallowed.
            self._log.exception("case_failed_unexpectedly", case_id=case.id, exc_info=True)
            return self._error_result(
                run,
                case,
                kind=ProviderErrorKind.UNKNOWN,
                message=f"unexpected {type(exc).__name__}: {exc}",
                exception_type=type(exc).__name__,
                status=CaseStatus.ERROR,
                attempts=attempts.count,
                started_at=started_at,
            )

    # -- persistence -----------------------------------------------------

    async def _persist_case(self, run: Run, result: CaseResult, counters: _Counters) -> None:
        """Write one finished case, and heartbeat the run, in one unit of work.

        The response is written by the response repository only; the case row
        does not carry it. That single write path is what stops the same
        response existing in two places with two different values.
        """
        async with self._uow_factory() as uow:
            if result.response is not None:
                await uow.responses.add(run.id, result.case_id, result.response)
            await uow.cases.add(result)
            await uow.evaluations.add_many(
                [
                    EvaluationRecord(run_id=run.id, case_id=result.case_id, result=evaluation)
                    for evaluation in result.evaluations
                ]
            )
            await uow.runs.update_status(
                run.id,
                RunStatus.RUNNING,
                updated_at=utc_now(),
                totals=counters.totals(),
            )
            await uow.commit()

    async def _load_completed(self, run_id: str) -> frozenset[str]:
        """Read the case ids already persisted for this run, for resume support."""
        async with self._uow_factory() as uow:
            return await uow.cases.completed_case_ids(run_id)

    async def _begin(self, run: Run, counters: _Counters) -> None:
        """Mark the run RUNNING and stamp its start time."""
        started_at = utc_now()
        async with self._uow_factory() as uow:
            await uow.runs.update_status(
                run.id,
                RunStatus.RUNNING,
                started_at=started_at,
                updated_at=started_at,
                totals=counters.totals(),
            )
            await uow.commit()

    async def _record(
        self,
        run: Run,
        result: CaseResult,
        *,
        counters: _Counters,
        budget: _Budget,
        emitter: ProgressEmitter,
    ) -> None:
        """Fold one finished case into the counters, persist it, and announce it.

        The order matters: the counters advance first so the heartbeat written
        alongside the case is already current, and the terminal progress event is
        emitted last so a consumer never sees a case reported complete before it
        is durable.
        """
        counters.n_completed += 1
        counters.n_attempted += 1
        counters.statuses.append(result.status)
        if result.status is not CaseStatus.OK:
            counters.n_errors += 1
        if result.passed:
            counters.n_passed += 1
        budget.add(result.cost)

        await self._persist_case(run, result, counters)

        if result.status is CaseStatus.OK:
            emitter.case_completed(result.case_id, passed=result.passed)
        else:
            emitter.case_failed(
                result.case_id,
                message=None if result.error is None else result.error.kind.value,
            )

    async def _mark_failed(
        self,
        run: Run,
        counters: _Counters,
        error: StorageError,
    ) -> None:
        """Record a run as FAILED after a persistence failure. Best effort, by nature.

        The thing that failed IS the database, so writing the verdict into it may
        fail too. That second failure is caught and logged rather than raised,
        because it must not displace the original one: the operator needs to see
        why the run died, not why the epitaph could not be written.
        """
        now = utc_now()
        try:
            async with self._uow_factory() as uow:
                await uow.runs.update_status(
                    run.id,
                    RunStatus.FAILED,
                    completed_at=now,
                    updated_at=now,
                    totals=counters.totals(),
                    error=str(error),
                )
                await uow.commit()
        except StorageError:
            self._log.exception("could_not_record_run_failure", run_id=run.id)

    async def _finish(
        self,
        run: Run,
        *,
        status: RunStatus,
        counters: _Counters,
        error: str | None,
    ) -> Run:
        """Write the run's terminal status and its metrics rollup, in ONE unit of work.

        The two writes share a transaction so a reader can never observe the
        run reporting a terminal status - COMPLETED or PARTIAL - with no
        rollup behind it yet. Writing them in separate transactions left
        exactly that window open: `GET /api/runs/{id}/metrics` could land
        between the status commit and the rollup commit and report
        `X-Metrics-Materialized: false` for a run that, by the time the
        response left, already had one.

        The rollup is computed from the PERSISTED case results rather than
        from the results this process happens to hold: a resumed run's earlier
        cases were written by a previous process, and a rollup over only this
        process's share would report a pass rate over part of the run. The
        same persisted results are what `Run.warnings` is derived from, for
        the same reason.

        A ``StorageError`` here propagates like every other one. A run whose
        terminal status and rollup could not be written is not a finished run
        with a missing convenience; it is a run whose database stopped
        accepting writes, and the caller needs to hear that.
        """
        completed_at = utc_now()
        async with self._uow_factory() as uow:
            results = [result async for result in uow.cases.iter_for_run(run.id)]
            await uow.runs.update_status(
                run.id,
                status,
                completed_at=completed_at,
                updated_at=completed_at,
                totals=counters.totals(),
                error=error,
                warnings=run_warnings(results),
            )
            stored = await uow.runs.get(run.id)
            finished = stored if stored is not None else run
            await uow.metrics.put(metrics_for_run(finished, results))
            await uow.commit()
        return finished

    # -- the run ---------------------------------------------------------

    def _final_status(
        self,
        run: Run,
        counters: _Counters,
        *,
        cancelled: bool,
        budget: _Budget,
    ) -> tuple[RunStatus, str | None]:
        """Decide the run's terminal status and the error string that explains it."""
        if cancelled:
            return RunStatus.CANCELLED, BUDGET_EXCEEDED if budget.exceeded else "cancelled"
        attempted = counters.n_attempted
        error_rate = counters.n_errors / attempted if attempted else 0.0
        if error_rate > run.config.max_error_rate:
            return RunStatus.FAILED, MAX_ERROR_RATE_EXCEEDED
        if counters.n_errors:
            return RunStatus.PARTIAL, None
        return RunStatus.COMPLETED, None

    async def execute(
        self,
        run: Run,
        cases: Sequence[ResolvedCase],
        *,
        cancel_event: asyncio.Event | None = None,
        progress: ProgressCallback | None = None,
        resume: bool = False,
    ) -> RunOutcome:
        """Execute every case of a run, persisting each as it finishes.

        Raises:
            StorageError: when persistence fails. The run is marked FAILED on a
                best-effort basis first, so the database records what happened
                rather than leaving a row stuck in RUNNING forever, and a SINGLE
                mapped exception is raised rather than the ``TaskGroup``'s
                ``ExceptionGroup`` - callers bound their handling to this
                project's exception hierarchy and cannot be expected to unwrap a
                group to find it.
        """
        counters = _Counters(n_cases=len(cases))
        budget = _Budget(cap=run.config.max_cost)
        cancel = cancel_event or asyncio.Event()
        emitter = ProgressEmitter(run.id, len(cases), progress)
        results: list[CaseResult] = []

        pending = list(cases)
        if resume:
            already = await self._load_completed(run.id)
            counters.n_completed = len(already)
            pending = [case for case in cases if case.id not in already]

        await self._begin(run, counters)
        emitter.run_started()

        semaphore = asyncio.Semaphore(run.config.concurrency)

        async def worker(case: ResolvedCase, provider: Provider, gate: RateLimitGate) -> None:
            """Run, record and persist one case."""
            async with semaphore:
                if cancel.is_set():
                    return
                await gate.wait()
                emitter.case_started(case.id)
                result = await self._run_case_isolated(
                    run, case, provider, gate=gate, emitter=emitter
                )
            results.append(result)
            await self._record(run, result, counters=counters, budget=budget, emitter=emitter)
            if budget.exceeded and not cancel.is_set():
                self._log.warning("budget_exceeded", spent=str(budget.spent), cap=str(budget.cap))
                cancel.set()

        try:
            async with (
                self._provider_factory(run.config.provider) as provider,
                RateLimitGate() as gate,
                asyncio.TaskGroup() as group,
            ):
                for case in pending:
                    group.create_task(worker(case, provider, gate))
        except* StorageError as group_error:
            # StorageError is deliberately NOT isolated per case: a database that
            # cannot be written to is a broken run, not a bad case. Record that
            # verdict where an operator will see it, then re-raise the original
            # exception on its own.
            failure = _first_storage_error(group_error)
            await self._mark_failed(run, counters, failure)
            raise failure from None

        counters.n_cancelled = len(pending) - len(results)
        status, error = self._final_status(run, counters, cancelled=cancel.is_set(), budget=budget)
        finished = await self._finish(run, status=status, counters=counters, error=error)
        emitter.run_completed(message=error)
        return RunOutcome(
            run=finished,
            results=tuple(results),
            budget_spent=budget.spent,
            unpriced_cases=budget.unpriced,
        )


__all__ = [
    "BUDGET_EXCEEDED",
    "HIGH_JUDGE_VARIANCE",
    "JUDGE_DISAGREEMENT",
    "MAX_ERROR_RATE_EXCEEDED",
    "SELF_PREFERENCE_RISK",
    "EvaluationStatus",
    "ProviderFactory",
    "RunEngine",
    "RunOutcome",
    "aggregate_case_verdict",
    "run_warnings",
]
