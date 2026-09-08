"""Rolling persisted case results up into one run's :class:`AggregateMetrics`.

The rollup is materialized once, at run finalization, and stored in the
``run_metrics`` row. Recomputing it on every read would make a listing of fifty
runs re-derive fifty rollups, and would let two surfaces disagree the moment
one of them cached.

Four rules decide almost everything here.

**One pass-rate rule.** The numerator and denominator come from
:mod:`llm_eval_lab.scoring`, which is the single definition in the project.
This module never re-implements it, at run level or per category or per
evaluator. Errored cases leave the denominator under ``error_policy="exclude"``
and join it as failures under ``"fail"``; score-only results leave it under
both. ``error_rate`` is over total attempted under either policy.

**One population per figure, stated.** Latency and score statistics are over
SUCCESSFULLY COMPLETED cases: a case that errored has no latency worth
reporting and no score anybody should average. Token and cost aggregates are
over EVERY attempted case, because an errored case can still have consumed
tokens and cost real money, and a coverage ratio that quietly dropped it would
overstate how much of the spend is accounted for. The two coverage fractions
say what fraction of that population actually carried the figure.

**Unknown is null, never zero.** No pass rate over an empty denominator, no
cost for an unpriced model, no mean over an empty series. Each of those is
``None``, and every one of them would otherwise read as a real measurement of
zero.

**A number below its sample-size gate is still reported, and marked.**
``LatencyStats.low_confidence`` names the percentile fields whose sample count
is under the gate. The values are computed and present; what the tuple adds is
that they should not be trusted.

**A known limitation of ``mean_score``, recorded rather than fixed.** The
run-level ``mean_score`` and ``median_score`` average ``CaseResult.score``,
which the runner already produced as a weight-weighted mean across a case's
evaluators. When a suite mixes evaluator types, that per-case blend combines
measurements on different scales before this module ever sees them, so the
run-level mean inherits the blend. The honest figures for a mixed suite are the
per-evaluator means in ``by_evaluator``, which are never combined. The formula
is deliberately unchanged: ``CaseResult.score`` is a frozen contract field with
a defined meaning, and re-deriving a different per-case score here would give
the same name two values. A consumer comparing evaluators reads
``by_evaluator``; ``mean_score`` is a headline for a single-evaluator suite.
"""

from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from statistics import fmean, median
from typing import Literal

from llm_eval_lab.models import (
    AggregateMetrics,
    CaseResult,
    CaseStatus,
    CategoryMetrics,
    CostBreakdown,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorMetrics,
    LatencyStats,
    Run,
    TokenTotals,
    TokenUsage,
)
from llm_eval_lab.pricing.calculator import aggregate_costs, per_unit_cost
from llm_eval_lab.reporting.statistics import (
    low_confidence_percentiles,
    percentile,
    wilson_interval,
)
from llm_eval_lab.scoring import CaseVerdict, PassRate, pass_rate, verdict_of
from llm_eval_lab.utils.time import utc_now


def _succeeded(results: Iterable[CaseResult]) -> list[CaseResult]:
    """Return the cases that completed successfully.

    The population behind every latency and score figure. ``CaseStatus.OK`` is
    the same condition :func:`llm_eval_lab.scoring.verdict_of` treats as
    non-errored, so the exclusion is defined once.
    """
    return [result for result in results if result.status is CaseStatus.OK]


def _interval(rate: PassRate) -> tuple[float, float] | None:
    """Return the Wilson interval for a pass rate, or None when nothing was eligible."""
    if rate.n_denominator <= 0:
        return None
    _, lower, upper = wilson_interval(rate.n_passed, rate.n_denominator)
    return lower, upper


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Return the mean, or None for an empty series rather than a fabricated zero."""
    return fmean(values) if values else None


def _median_or_none(values: Sequence[float]) -> float | None:
    """Return the median, or None for an empty series."""
    return float(median(values)) if values else None


def latency_stats(results: Sequence[CaseResult]) -> LatencyStats:
    """Summarize latency over successfully completed cases only.

    The series is ``ModelResponse.latency_ms``: how long the SUCCESSFUL attempt
    took, which is the quantity that compares across models. Retry waiting is
    deliberately excluded and lives in ``total_latency_ms`` and ``attempts``
    instead, because a p95 inflated by backoff measures a rate limit rather than
    a model. A case that errored or timed out is excluded from the series even
    when it did record elapsed time before failing, and ``n`` is what makes that
    exclusion visible next to every percentile.

    Percentiles use linear interpolation. Each one below its minimum-sample gate
    is still computed and is named in ``low_confidence``.
    """
    samples = [
        result.response.latency_ms for result in _succeeded(results) if result.response is not None
    ]
    n = len(samples)
    if not samples:
        return LatencyStats(
            n=0,
            mean_ms=None,
            p50_ms=None,
            p90_ms=None,
            p95_ms=None,
            p99_ms=None,
            max_ms=None,
            low_confidence=low_confidence_percentiles(0),
        )
    return LatencyStats(
        n=n,
        mean_ms=fmean(samples),
        p50_ms=percentile(samples, 50),
        p90_ms=percentile(samples, 90),
        p95_ms=percentile(samples, 95),
        p99_ms=percentile(samples, 99),
        max_ms=max(samples),
        low_confidence=low_confidence_percentiles(n),
    )


def token_totals(results: Sequence[CaseResult]) -> TokenTotals:
    """Sum token usage over every attempted case, counting the gaps.

    Usage counts only when the vendor reported BOTH halves. A lone total is
    never split, so a response carrying only one side is missing usage rather
    than half-present, and it lands in ``n_missing_usage`` where the coverage
    ratio can see it.
    """
    usages: list[TokenUsage] = [
        result.response.usage for result in results if result.response is not None
    ]
    complete = [usage for usage in usages if usage.is_complete]
    n_with_usage = len(complete)
    n_missing_usage = len(results) - n_with_usage
    if not complete:
        return TokenTotals(
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            mean_input_tokens=None,
            mean_output_tokens=None,
            n_with_usage=0,
            n_missing_usage=n_missing_usage,
        )
    input_tokens = sum(usage.input_tokens or 0 for usage in complete)
    output_tokens = sum(usage.output_tokens or 0 for usage in complete)
    return TokenTotals(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=sum(usage.total_tokens or 0 for usage in complete),
        mean_input_tokens=input_tokens / n_with_usage,
        mean_output_tokens=output_tokens / n_with_usage,
        n_with_usage=n_with_usage,
        n_missing_usage=n_missing_usage,
    )


def error_breakdown(results: Sequence[CaseResult]) -> dict[str, int]:
    """Tally why cases failed, keyed by a stable code.

    A provider failure is keyed by its ``ProviderErrorKind``. A case whose
    provider call succeeded but whose evaluator could not reach a conclusion is
    keyed ``evaluation_error``, because blaming the model for a broken evaluator
    is exactly the confusion ``status`` versus ``passed`` exists to prevent.
    """
    tally: dict[str, int] = {}
    for result in results:
        if result.status is CaseStatus.OK:
            continue
        if result.error is not None:
            key = result.error.kind.value
        elif result.status is CaseStatus.ERROR:
            key = "evaluation_error"
        else:
            key = result.status.value
        tally[key] = tally.get(key, 0) + 1
    return dict(sorted(tally.items()))


def _bucket_metrics(bucket: Sequence[CaseResult], *, error_policy: str) -> CategoryMetrics:
    """Summarize one category or tag bucket under the run's error policy.

    ``n`` is every case in the bucket, including the errored ones the pass-rate
    denominator drops. A bucket of ten cases with a pass rate over four is a
    different statement from a bucket of four, and only reporting both says so.
    """
    rate = pass_rate(
        (verdict_of(item.status, passed=item.passed) for item in bucket),
        error_policy=error_policy,
    )
    scores = [item.score for item in _succeeded(bucket) if item.score is not None]
    return CategoryMetrics(
        n=len(bucket),
        n_passed=rate.n_passed,
        pass_rate=rate.rate,
        pass_rate_ci=_interval(rate),
        mean_score=_mean_or_none(scores),
    )


def by_category(results: Sequence[CaseResult], *, error_policy: str) -> dict[str, CategoryMetrics]:
    """Group results by category. Cases with no category form no bucket.

    An "uncategorized" bucket would be an invented category, and a reader
    comparing buckets would have no way to tell it from one the suite declared.
    """
    buckets: dict[str, list[CaseResult]] = {}
    for result in results:
        if result.category is not None:
            buckets.setdefault(result.category, []).append(result)
    return {
        key: _bucket_metrics(bucket, error_policy=error_policy)
        for key, bucket in sorted(buckets.items())
    }


def by_tag(results: Sequence[CaseResult], *, error_policy: str) -> dict[str, CategoryMetrics]:
    """Group results by tag. A case carrying several tags is counted under each.

    The buckets therefore overlap and their sizes do not sum to the run's case
    count, which is a property of tags rather than a defect.
    """
    buckets: dict[str, list[CaseResult]] = {}
    for result in results:
        for tag in result.tags:
            buckets.setdefault(tag, []).append(result)
    return {
        key: _bucket_metrics(bucket, error_policy=error_policy)
        for key, bucket in sorted(buckets.items())
    }


def _is_score_only(evaluation: EvaluationResult) -> bool:
    """Report whether one evaluation produced a score but no pass/fail verdict.

    The contract names exactly one pairing: ``status=PASSED`` with
    ``passed=None`` and a populated score. An errored or skipped evaluation also
    has ``passed=None`` and is not score-only; it produced nothing at all.
    """
    return (
        evaluation.status is EvaluationStatus.PASSED
        and evaluation.passed is None
        and evaluation.score is not None
    )


def _evaluator_metrics(
    evaluations: Sequence[EvaluationResult],
    *,
    error_policy: str,
) -> EvaluatorMetrics:
    """Roll one evaluator's results up across every case of a run."""
    rate = pass_rate(
        (
            CaseVerdict(
                errored=item.status is EvaluationStatus.ERROR,
                passed=item.passed,
            )
            for item in evaluations
        ),
        error_policy=error_policy,
    )
    scores = [item.score for item in evaluations if item.score is not None]
    return EvaluatorMetrics(
        evaluator_id=evaluations[0].evaluator_id,
        evaluator_type=evaluations[0].evaluator_type,
        n=len(evaluations),
        n_passed=rate.n_passed,
        n_errors=sum(1 for item in evaluations if item.status is EvaluationStatus.ERROR),
        pass_rate=rate.rate,
        mean_score=_mean_or_none(scores),
        median_score=_median_or_none(scores),
        n_score_only=sum(1 for item in evaluations if _is_score_only(item)),
        # The denominator behind `pass_rate`, carried rather than discarded: it
        # is the `n` any interval on that rate needs, and re-deriving it from
        # `n`, `n_errors` and `n_score_only` in a consumer would be a second
        # implementation of the exclusion rule `scoring` owns.
        n_pass_denominator=rate.n_denominator,
    )


def by_evaluator(
    results: Sequence[CaseResult],
    *,
    error_policy: str,
) -> dict[str, EvaluatorMetrics]:
    """Group every evaluation of a run by evaluator id.

    Per-evaluator means are reported separately and are never blended into one
    number. Evaluators of different types score on different scales, and an
    average across them would be arithmetic performed on incompatible units.

    The pass rate here is over EVALUATIONS, not cases, so its denominator is a
    different one from the run's. It goes through the same
    :func:`llm_eval_lab.scoring.pass_rate` rule, with each evaluation presented
    as its own verdict, which is what keeps the score-only and errored
    exclusions identical at both levels.
    """
    buckets: dict[str, list[EvaluationResult]] = {}
    for result in results:
        for evaluation in result.evaluations:
            buckets.setdefault(evaluation.evaluator_id, []).append(evaluation)
    return {
        key: _evaluator_metrics(bucket, error_policy=error_policy)
        for key, bucket in sorted(buckets.items())
    }


def _run_cost(cost: CostBreakdown) -> Decimal | None:
    """Return the run's candidate-model spend, or None when nothing was priced.

    ``total_cost`` means CANDIDATE-MODEL COST, here and on every other surface.
    Judge spend is a separate line item in ``CostBreakdown.judge_cost`` and is
    never folded in, so ``cost_per_case * n_completed`` reconciles against
    ``cost.total_cost`` and the runs listing publishes the same number under the
    same name. Folding judge cost into one of the two figures and not the other
    is what made them disagree, which is the cross-surface drift
    :mod:`llm_eval_lab.scoring` exists to prevent for pass rates.
    """
    return cost.total_cost


def compute_metrics(  # noqa: PLR0913 - one parameter per independent input
    *,
    run_id: str,
    n_cases: int,
    results: Sequence[CaseResult],
    error_policy: str,
    price_table_id: str,
    price_table_version: str,
    currency: Literal["USD"] = "USD",
    computed_at: datetime | None = None,
) -> AggregateMetrics:
    """Roll a run's persisted case results up into its aggregate metrics.

    Args:
        run_id: The run these results belong to.
        n_cases: How many cases the run selected, which can exceed the number of
            results when a run was cancelled before finishing.
        results: Every persisted case result of the run.
        error_policy: ``"exclude"`` or ``"fail"``, from the run's configuration.
        price_table_id: Identity of the price table the run was costed against.
        price_table_version: Version of that table.
        currency: The table's currency.
        computed_at: Timestamp for the rollup. Defaults to now.

    Returns:
        The complete rollup, ready to persist.
    """
    n_completed = len(results)
    succeeded = _succeeded(results)
    # `status is not OK` is the same condition `verdict_of` calls errored and
    # the same one `RunTotals.n_errors` counts, so the run row, the pass-rate
    # rule and this rollup cannot report three different error counts.
    n_errors = n_completed - len(succeeded)

    rate = pass_rate(
        (verdict_of(item.status, passed=item.passed) for item in results),
        error_policy=error_policy,
    )
    scores = [item.score for item in succeeded if item.score is not None]

    costs = [item.cost for item in results]
    rollup = aggregate_costs(
        costs,
        price_table_id=price_table_id,
        price_table_version=price_table_version,
        currency=currency,
        n_units=n_completed,
    )
    run_cost = _run_cost(rollup.cost)
    tokens = token_totals(results)

    return AggregateMetrics(
        run_id=run_id,
        computed_at=computed_at or utc_now(),
        n_cases=n_cases,
        n_completed=n_completed,
        n_errors=n_errors,
        n_timeouts=sum(1 for item in results if item.status is CaseStatus.TIMEOUT),
        n_skipped=sum(1 for item in results if item.status is CaseStatus.SKIPPED),
        # Always over total ATTEMPTED, under either error policy. Cases a
        # cancelled run never reached are not in the denominator: the system has
        # no evidence about them and counting them as successes would flatter
        # the rate.
        error_rate=(n_errors / n_completed if n_completed else 0.0),
        error_breakdown=error_breakdown(results),
        error_policy=error_policy,
        pass_rate=rate.rate,
        pass_rate_ci=_interval(rate),
        n_passed=rate.n_passed,
        n_pass_denominator=rate.n_denominator,
        n_scored=len(scores),
        mean_score=_mean_or_none(scores),
        median_score=_median_or_none(scores),
        n_score_only=sum(
            1 for item in results for evaluation in item.evaluations if _is_score_only(evaluation)
        ),
        latency=latency_stats(results),
        tokens=tokens,
        token_usage_coverage=(tokens.n_with_usage / n_completed if n_completed else None),
        cost=rollup.cost,
        cost_coverage=rollup.coverage,
        cost_per_case=per_unit_cost(run_cost, n_completed),
        cost_per_successful_evaluation=per_unit_cost(run_cost, rate.n_passed),
        by_category=by_category(results, error_policy=error_policy),
        by_tag=by_tag(results, error_policy=error_policy),
        by_evaluator=by_evaluator(results, error_policy=error_policy),
    )


def metrics_for_run(run: Run, results: Sequence[CaseResult]) -> AggregateMetrics:
    """Compute a run's rollup, reading every setting off the run itself.

    The price table identity comes from ``RunConfig``, which recorded it when
    the run started. A historical rollup therefore keeps naming the table its
    costs were computed with, whatever the shipped price file has since become.
    """
    return compute_metrics(
        run_id=run.id,
        n_cases=run.totals.n_cases,
        results=results,
        error_policy=run.config.error_policy,
        price_table_id=run.config.price_table_id,
        price_table_version=run.config.price_table_version,
    )


__all__ = [
    "by_category",
    "by_evaluator",
    "by_tag",
    "compute_metrics",
    "error_breakdown",
    "latency_stats",
    "metrics_for_run",
    "token_totals",
]
