"""Composing a comparison: fetch two runs, roll them up, apply the policy.

The regression engine in :mod:`llm_eval_lab.reporting.regression` knows nothing
about the database, which is what lets it be unit-tested against fixtures. This
service is the seam: it reads the two runs, their rollups and their persisted
case results, hands them to the engine, and returns the structured report the
CLI and the API both render.

A run with no stored rollup is rolled up here from its persisted cases rather
than being refused. An interrupted run is a legitimate baseline to compare
against, and :class:`ComparisonView` reports which side was derived on the spot
so a reader can weigh it.
"""

from dataclasses import dataclass
from pathlib import Path

import structlog

from llm_eval_lab.models import (
    RegressionReport,
    RegressionThresholds,
    UnitOfWorkFactory,
    Verdict,
)
from llm_eval_lab.reporting.regression import (
    RunComparisonInput,
    compare_runs,
    load_thresholds,
)
from llm_eval_lab.services.metrics_service import MetricsService, MetricsView


@dataclass(frozen=True)
class ComparisonView:
    """A finished comparison, plus what a reader needs to weigh it.

    ``warnings`` carries the conditions that do not change the verdict but do
    change how much it is worth: a rollup computed on the spot, or two runs of
    different suites compared because the policy allowed it.
    """

    report: RegressionReport
    baseline: MetricsView
    candidate: MetricsView
    warnings: tuple[str, ...] = ()

    @property
    def verdict(self) -> Verdict:
        """The verdict the exit code is derived from."""
        return self.report.verdict


class RunNotFoundError(Exception):
    """Raised when a comparison names a run id that no run has.

    A dedicated exception rather than ``None``, because the caller has to tell
    WHICH of the two ids was wrong and a bare ``None`` cannot say.
    """

    def __init__(self, run_id: str) -> None:
        """Name the run that could not be found."""
        self.run_id = run_id
        super().__init__(f"no run with id {run_id}")


class ComparisonService:
    """Compares two persisted runs against a threshold policy."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        logger: structlog.BoundLogger,
    ) -> None:
        """Bind the service to its collaborators."""
        self._metrics = MetricsService(uow_factory=uow_factory, logger=logger)
        self._log = logger

    def thresholds(self, path: Path | None = None) -> RegressionThresholds:
        """Load the policy for this comparison, or the one shipped with the package.

        Raises:
            EvaluatorConfigError: when the policy file is missing or invalid.
        """
        return load_thresholds(path)

    async def _side(self, run_id: str) -> tuple[MetricsView, RunComparisonInput]:
        """Read one run, its rollup and its cases into a comparison input.

        Raises:
            RunNotFoundError: when no run has that id.
        """
        view = await self._metrics.get(run_id)
        if view is None:
            raise RunNotFoundError(run_id)
        results = await self._metrics.cases(run_id)
        return view, RunComparisonInput(
            run_id=view.run.id,
            label=view.run.label,
            suite_hash=view.run.config.suite_hash,
            metrics=view.metrics,
            results=results,
            error_policy=view.run.config.error_policy,
        )

    async def compare(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        *,
        thresholds: RegressionThresholds,
    ) -> ComparisonView:
        """Compare a candidate run against a baseline under one policy.

        Raises:
            RunNotFoundError: when either run id is unknown.
            RegressionInputError: when a rollup does not belong to its run.
            EvaluatorConfigError: when the policy names an unaddressable metric.
        """
        baseline_view, baseline = await self._side(baseline_run_id)
        candidate_view, candidate = await self._side(candidate_run_id)
        report = compare_runs(baseline, candidate, thresholds)

        warnings: list[str] = []
        for name, view in (("baseline", baseline_view), ("candidate", candidate_view)):
            if not view.materialized:
                warnings.append(
                    f"the {name} run has no stored rollup, so its figures were "
                    f"computed from the cases persisted so far"
                )
        if not report.suite_hash_match and report.comparable:
            warnings.append(
                "the two runs used DIFFERENT benchmark suites and were compared "
                "anyway because the policy sets require_same_suite: false"
            )
        if baseline.error_policy != candidate.error_policy:
            warnings.append(
                f"the two runs counted errors differently ({baseline.error_policy} "
                f"versus {candidate.error_policy}), so their pass rates are over "
                f"different populations"
            )

        self._log.info(
            "comparison_completed",
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            verdict=report.verdict.value,
            mode=report.mode,
            paired_case_count=report.paired_case_count,
            thresholds_id=report.thresholds_id,
        )
        return ComparisonView(
            report=report,
            baseline=baseline_view,
            candidate=candidate_view,
            warnings=tuple(warnings),
        )

    async def metrics(self, run_id: str) -> MetricsView:
        """Return one run's rollup, for a single-run report.

        Raises:
            RunNotFoundError: when no run has that id.
        """
        view = await self._metrics.get(run_id)
        if view is None:
            raise RunNotFoundError(run_id)
        return view


def build_comparison_service(
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
) -> ComparisonService:
    """Build a comparison service over one database."""
    return ComparisonService(uow_factory=uow_factory, logger=logger)


__all__ = [
    "ComparisonService",
    "ComparisonView",
    "RunNotFoundError",
    "build_comparison_service",
]
