"""Reading a run's aggregate rollup, and recomputing one when it is missing.

The rollup is materialized by the run engine at finalization, so the normal
path here is a single-row read. Two situations leave a run without one: a run
interrupted by a dead process, which never reached finalization, and a run
whose rollup predates a change to the aggregation rules. Both are answered by
recomputing from the persisted case results and storing the result, so the
gap closes permanently rather than being papered over on every read.

The recomputation is not a silent fallback. :class:`MetricsView` reports
whether the figures came from the stored row or were derived on the spot, and
the CLI prints that, because a rollup computed from a half-finished run is a
different claim from one written when the run completed.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from statistics import median

import structlog

from llm_eval_lab.models import (
    AggregateMetrics,
    CaseResult,
    JSONValue,
    Run,
    RunQuery,
    RunStatus,
    RunSummary,
    UnitOfWorkFactory,
)
from llm_eval_lab.reporting.aggregate import metrics_for_run
from llm_eval_lab.reporting.formatters import case_rows, metrics_document
from llm_eval_lab.reporting.statistics import wilson_interval

TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.INTERRUPTED,
    }
)
"""Statuses a run does not leave. Only a terminal run's recomputed rollup is stored."""

MAX_SUMMARISED_RUNS = 2000
"""Cap on how many runs one leaderboard walks, so the endpoint stays bounded."""

RUN_PAGE_SIZE = 500
"""Page size used to walk the runs listing. Matches the repository's own cap."""


@dataclass(frozen=True)
class MetricsView:
    """A run's rollup, plus where it came from."""

    run: Run
    metrics: AggregateMetrics
    materialized: bool
    """True when the figures were read from the stored row rather than derived now."""


@dataclass(frozen=True)
class ModelSummary:
    """One leaderboard row: every stored run of one provider and model, pooled.

    Only runs that HAVE a stored rollup contribute. A run still executing has no
    final figures, and pooling a moving target into a leaderboard would report a
    pass rate over a population that changes between two reads of the same page.

    ``pass_rate`` and ``pass_rate_ci`` are computed from the pooled numerator and
    denominator rather than by averaging per-run rates, so a 200-case run is not
    given the same weight as a 2-case one. The interval is Wilson at 95%.

    ``median_run_p95_ms`` is NAMED for what it is: the median of the per-run p95
    values, not a pooled percentile over cases. Per-case latencies are not loaded
    here, so a true pooled p95 is not available, and a field called ``p95_ms``
    would be read as one by every dashboard author who never opened its
    description. ``n_runs_with_latency`` says how many numbers went into it.

    ``cost_per_case`` is the summed cost over the summed COMPLETED cases, which
    is the definition phase 3 pinned for the same field name on a single run:
    ``reporting/aggregate.py`` computes it as ``run_cost / n_completed`` and
    documents the identity ``cost_per_case * n_completed == cost.total_cost``.
    The runs listing publishes the same number under the same name.

    Dividing by the priced cases instead would be a more useful figure and a
    worse one: two endpoints would then publish different numbers under one name,
    and the identity a reader can check by hand would stop holding. The coverage
    caveat travels beside it in ``cost_coverage`` rather than being folded into
    the value, which is the same choice the rest of this codebase makes about
    unknown data.
    """

    provider: str
    model: str
    n_runs: int
    n_cases: int
    n_passed: int
    n_pass_denominator: int
    pass_rate: float | None
    pass_rate_ci: tuple[float, float] | None
    median_run_p95_ms: float | None
    n_runs_with_latency: int
    total_cost: Decimal | None
    cost_per_case: Decimal | None
    cost_coverage: float | None
    last_run_at: datetime | None


@dataclass
class _Accumulator:
    """Mutable running totals for one provider-and-model bucket."""

    provider: str
    model: str
    n_runs: int = 0
    n_cases: int = 0
    n_passed: int = 0
    n_pass_denominator: int = 0
    p95_values: list[float] = field(default_factory=list)
    total_cost: Decimal | None = None
    n_cost_cases: int = 0
    coverage_weight: float = 0.0
    coverage_cases: int = 0
    last_run_at: datetime | None = None

    def add(self, summary: RunSummary, metrics: AggregateMetrics) -> None:
        """Fold one run's stored rollup into this bucket."""
        self.n_runs += 1
        self.n_cases += metrics.n_completed
        self.n_passed += metrics.n_passed
        self.n_pass_denominator += metrics.n_pass_denominator
        if metrics.latency.p95_ms is not None:
            self.p95_values.append(metrics.latency.p95_ms)
        if metrics.cost.total_cost is not None:
            self.total_cost = (self.total_cost or Decimal(0)) + metrics.cost.total_cost
            # Every COMPLETED case of a run that reported a cost, matching what
            # `reporting.aggregate` divides by for a single run. Not the priced
            # subset: that would publish a different number under the same field
            # name the runs listing already uses, and break the identity
            # `cost_per_case * n_completed == total_cost` that a reader can check.
            self.n_cost_cases += metrics.n_completed
        if metrics.cost_coverage is not None:
            self.coverage_weight += metrics.cost_coverage * metrics.n_completed
            self.coverage_cases += metrics.n_completed
        stamp = summary.completed_at or summary.created_at
        if self.last_run_at is None or stamp > self.last_run_at:
            self.last_run_at = stamp

    def finish(self) -> ModelSummary:
        """Reduce the running totals into one immutable leaderboard row."""
        rate: float | None = None
        interval: tuple[float, float] | None = None
        if self.n_pass_denominator > 0:
            point, low, high = wilson_interval(self.n_passed, self.n_pass_denominator)
            rate, interval = point, (low, high)
        per_case: Decimal | None = None
        if self.total_cost is not None and self.n_cost_cases > 0:
            per_case = self.total_cost / Decimal(self.n_cost_cases)
        return ModelSummary(
            provider=self.provider,
            model=self.model,
            n_runs=self.n_runs,
            n_cases=self.n_cases,
            n_passed=self.n_passed,
            n_pass_denominator=self.n_pass_denominator,
            pass_rate=rate,
            pass_rate_ci=interval,
            median_run_p95_ms=median(self.p95_values) if self.p95_values else None,
            n_runs_with_latency=len(self.p95_values),
            total_cost=self.total_cost,
            cost_per_case=per_case,
            cost_coverage=(
                self.coverage_weight / self.coverage_cases if self.coverage_cases else None
            ),
            last_run_at=self.last_run_at,
        )


class MetricsService:
    """Reads and, where necessary, rebuilds a run's aggregate metrics."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        logger: structlog.BoundLogger,
    ) -> None:
        """Bind the service to its collaborators.

        No ``Settings``: nothing here reads one. An injected dependency that is
        never used reads as a promise that this service is configurable, and the
        first person to act on that promise finds it is not.
        """
        self._uow_factory = uow_factory
        self._log = logger

    async def get(self, run_id: str) -> MetricsView | None:
        """Return one run's rollup, computing it first if the row is absent.

        A rollup recomputed for a run that is still RUNNING is returned but NOT
        stored: the run is still writing cases, and persisting a snapshot of a
        moving target would leave a stale row behind that later reads would
        trust. Only a terminal run's recomputation is written.

        Returns:
            The view, or ``None`` when no run has that id.
        """
        async with self._uow_factory() as uow:
            run = await uow.runs.get(run_id)
            if run is None:
                return None
            stored = await uow.metrics.get(run_id)
            if stored is not None:
                return MetricsView(run=run, metrics=stored, materialized=True)

            results = [result async for result in uow.cases.iter_for_run(run_id)]
            computed = metrics_for_run(run, results)
            if run.status in TERMINAL_STATUSES:
                await uow.metrics.put(computed)
                await uow.commit()
                self._log.info(
                    "metrics_backfilled",
                    run_id=run_id,
                    status=run.status.value,
                    n_results=len(results),
                )
            return MetricsView(run=run, metrics=computed, materialized=False)

    async def recompute(self, run_id: str) -> MetricsView | None:
        """Rebuild and store one run's rollup from its persisted case results.

        Used when the aggregation rules themselves have changed. Storing an
        unconditional recomputation of a RUNNING run would race the engine, so
        a non-terminal run is recomputed and returned without being written.

        Returns:
            The view, or ``None`` when no run has that id.
        """
        async with self._uow_factory() as uow:
            run = await uow.runs.get(run_id)
            if run is None:
                return None
            results = [result async for result in uow.cases.iter_for_run(run_id)]
            computed = metrics_for_run(run, results)
            if run.status in TERMINAL_STATUSES:
                await uow.metrics.put(computed)
                await uow.commit()
            return MetricsView(run=run, metrics=computed, materialized=False)

    async def model_summaries(self) -> tuple[ModelSummary, ...]:
        """Pool every stored run rollup into one row per provider and model.

        Runs without a stored rollup are skipped rather than recomputed: a
        leaderboard is a read, and recomputing a rollup here would make a page
        load do the work of a whole run's aggregation for every unfinished run
        in the database.
        """
        buckets: dict[tuple[str, str], _Accumulator] = {}
        async with self._uow_factory() as uow:
            summaries: list[RunSummary] = []
            offset = 0
            while len(summaries) < MAX_SUMMARISED_RUNS:
                page = await uow.runs.list(RunQuery(limit=RUN_PAGE_SIZE, offset=offset))
                summaries.extend(page.items)
                offset += len(page.items)
                if not page.items or offset >= page.total:
                    break
            stored = await uow.metrics.get_many([summary.id for summary in summaries])

        for summary in summaries:
            metrics = stored.get(summary.id)
            if metrics is None:
                continue
            key = (summary.provider, summary.model)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = _Accumulator(provider=summary.provider, model=summary.model)
                buckets[key] = bucket
            bucket.add(summary, metrics)

        return tuple(buckets[key].finish() for key in sorted(buckets))

    async def cases(self, run_id: str) -> tuple[CaseResult, ...]:
        """Return every persisted case result of one run, for export.

        Drained inside the owning unit of work: the repository streams through
        a session that closes when the scope exits, so a lazily returned
        iterator would be consumed after its session was gone.
        """
        async with self._uow_factory() as uow:
            return tuple([result async for result in uow.cases.iter_for_run(run_id)])


def export_document(
    run_id: str,
    results: Sequence[CaseResult],
    metrics: AggregateMetrics | None,
) -> dict[str, JSONValue]:
    """Build the JSON export envelope both surfaces emit.

    Here rather than in the API router or the CLI command, because both write it
    and two constructions of one document drift. The ROW shapes already come
    from `reporting.formatters`; this is the wrapper around them, and it was the
    half still duplicated.
    """
    rows: list[JSONValue] = list(case_rows(results))
    document: dict[str, JSONValue] = {
        "run_id": run_id,
        "n_cases": len(results),
        "cases": rows,
    }
    if metrics is not None:
        document["metrics"] = metrics_document(metrics)
    return document


def build_metrics_service(
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
) -> MetricsService:
    """Build a metrics service over one database."""
    return MetricsService(uow_factory=uow_factory, logger=logger)
