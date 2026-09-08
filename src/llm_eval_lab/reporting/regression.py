"""The regression engine: baseline versus candidate, decided on point estimates.

Decision D-GATE in one sentence: **a threshold compares two point estimates and
resolves deterministically**, so the same two runs always produce the same
verdict. Wilson intervals, the Newcombe Method-10 interval on the paired
pass-rate delta and the McNemar exact p-value are computed here and travel on
the :class:`~llm_eval_lab.models.CheckOutcome` as advisory context. They never
participate in pass/fail logic. A CI gate that flipped because a stochastic
provider landed near an interval boundary would be worse than no gate at all.

Four rules decide almost everything below.

**Bounds are exact, and asymmetric on purpose.** ``min_value`` is an INCLUSIVE
floor: a candidate exactly equal to it passes. ``max_value`` is an EXCLUSIVE
ceiling: a candidate exactly equal to it FAILS. That asymmetry is what makes
the specification's "error rate must remain below 1%" expressible as
``max_value: 0.01``. Every comparison goes through :func:`_close`, so a bound
sitting exactly on a value that floating-point subtraction produced as
``-0.020000000000000018`` is still read as the boundary it is.

**Nothing is skipped silently.** A metric below its sample-size gate, an
instance present in only one run, a relative bound whose baseline is zero and a
policy naming a metric the runs do not carry each produce a VISIBLE outcome
with a stated reason. ``INSUFFICIENT_DATA`` is the honest answer to "we cannot
say"; a silent pass is not. Sample counts that are unavailable are treated as
zero and therefore as insufficient, because "we did not measure the population"
and "the population was too small" are the same claim about trust.

**Pairing is on ``(case_id, case_hash)``.** A case whose id matches but whose
hash differs is a DIFFERENT case wearing the same name. It lands in
``changed_cases`` and is excluded from every paired statistic rather than
silently compared. In ``paired`` mode the overall pass rate is computed over
that intersection, which is what McNemar and Method 10 describe; every other
check reads each run's own aggregate figures.

**Modes change the advisory context, never the verdict.** ``paired`` mode
attaches the McNemar p-value and the Method-10 interval on the delta.
``unpaired`` mode attaches each run's own Wilson interval and whether the two
overlap. The thresholds evaluate the same point estimates either way, and the
mode is stated explicitly in the report so downstream tooling cannot read an
unpaired comparison as a matched one.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from llm_eval_lab.models import (
    AggregateMetrics,
    CaseDelta,
    CaseResult,
    CategoryMetrics,
    CheckOutcome,
    CheckStatus,
    EvaluatorConfigError,
    EvaluatorMetrics,
    MetricCheck,
    RegressionInputError,
    RegressionReport,
    RegressionSummary,
    RegressionThresholds,
    Verdict,
)
from llm_eval_lab.reporting.statistics import (
    mcnemar_exact,
    newcombe_method10,
    wilson_interval,
    z_for_alpha,
)
from llm_eval_lab.scoring import normalize_error_policy, verdict_of
from llm_eval_lab.utils.time import utc_now

DEFAULT_THRESHOLDS_PATH = Path(__file__).parent / "data" / "thresholds.yaml"
"""The policy shipped inside the package. Overridable per invocation."""

MAX_SCORE_DROPS = 5
"""How many of the largest per-case score drops the summary carries."""

_REL_TOL = 1e-9
"""Relative tolerance for reading a bound comparison at its exact boundary."""

_ABS_TOL = 1e-12
"""Absolute tolerance for the same, so a bound of zero is still comparable."""

_ROOT_METRICS: frozenset[str] = frozenset({"pass_rate", "error_rate", "mean_score", "median_score"})
"""Metric paths that name one figure of a run's rollup directly."""

_LATENCY_FIELDS: frozenset[str] = frozenset(
    {"mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"}
)
"""The percentile and summary fields of ``LatencyStats``, addressed as ``latency.<field>``."""

_NAMESPACES: frozenset[str] = frozenset({"category", "tag", "evaluator", "evaluator_type"})
"""Metric path prefixes that select a SET of instances rather than one figure."""

_BUCKET_LEAVES: frozenset[str] = frozenset({"pass_rate", "mean_score"})
"""The figures addressable on one category, tag or evaluator instance."""

_INSTANCE_PATH_SEGMENTS = 3
"""``<namespace>.<selector>.<leaf>``."""

_LATENCY_PATH_SEGMENTS = 2
"""``latency.<field>``."""

MISSING_METRIC_NOTE = (
    "One run reports this metric and the other does not, so there is nothing to compare it against."
)
"""Why a check resolved to ``MISSING_METRIC``: an ASYMMETRIC absence.

This is the mismatch ``fail_on_missing_metric`` exists to catch. A metric one run
measured and the other did not usually means the benchmark or the evaluator set
moved underneath the policy.
"""

BOTH_ABSENT_NOTE = (
    "Neither run reports this metric, so the policy could not be applied to this comparison at all."
)
"""Why a check resolved to ``INSUFFICIENT_DATA`` rather than ``MISSING_METRIC``.

A metric absent from BOTH runs says nothing about the candidate. A judge-only
suite has no ``pass_rate`` in either run by design, and reporting that as a
regression would fail a build for a candidate byte-identical to its baseline. The
symmetric case is therefore "we cannot say", which never fails a build; only the
asymmetric case is a mismatch worth failing on.
"""

LATENCY_NOTE = (
    "Latency percentiles are over successful model attempts and exclude retry backoff, "
    "so this gate measures the model rather than a rate limit."
)
"""Printed beside any latency gate, because wall-clock latency is a different quantity."""


# ---------------------------------------------------------------------------
# Bound arithmetic
# ---------------------------------------------------------------------------


def _close(left: float, right: float) -> bool:
    """Report whether two values are the same number to within reading tolerance.

    Exists because the boundary cases are the ones that matter. A pass rate that
    fell by exactly the tolerated two points arrives from float subtraction as
    ``-0.020000000000000018``, and a bare ``<`` would call that a regression.
    """
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _below(value: float, bound: float) -> bool:
    """Report whether `value` is strictly below `bound`, boundary excluded."""
    return value < bound and not _close(value, bound)


def _above(value: float, bound: float) -> bool:
    """Report whether `value` is strictly above `bound`, boundary excluded."""
    return value > bound and not _close(value, bound)


@dataclass(frozen=True)
class _Bound:
    """One configured bound, resolved against a concrete baseline and candidate.

    ``threshold`` is in the metric's own units: the number the candidate was
    compared against, which is what a report reader needs. ``delta_bound`` is
    the same statement expressed on candidate-minus-baseline, and is ``None``
    for an absolute bound that does not reference the baseline. It is what the
    advisory interval is compared against, since that interval is on the delta.
    """

    name: str
    threshold: float
    delta_bound: float | None
    violated: bool


def _absolute_bounds(check: MetricCheck, candidate: float) -> list[_Bound]:
    """Resolve the bounds that read the candidate alone."""
    bounds: list[_Bound] = []
    if check.min_value is not None:
        # INCLUSIVE floor: equality passes.
        bounds.append(
            _Bound(
                name="min_value",
                threshold=check.min_value,
                delta_bound=None,
                violated=_below(candidate, check.min_value),
            )
        )
    if check.max_value is not None:
        # EXCLUSIVE ceiling: equality FAILS. This is the asymmetry that makes
        # "must remain below 1%" expressible as max_value: 0.01.
        bounds.append(
            _Bound(
                name="max_value",
                threshold=check.max_value,
                delta_bound=None,
                violated=_above(candidate, check.max_value) or _close(candidate, check.max_value),
            )
        )
    return bounds


def _movement_bounds(check: MetricCheck, baseline: float, candidate: float) -> list[_Bound]:
    """Resolve the bounds that compare the candidate against the baseline."""
    bounds: list[_Bound] = []
    delta = candidate - baseline
    if check.max_absolute_decrease is not None:
        limit = -check.max_absolute_decrease
        bounds.append(
            _Bound(
                name="max_absolute_decrease",
                threshold=baseline + limit,
                delta_bound=limit,
                violated=_below(delta, limit),
            )
        )
    if check.max_absolute_increase is not None:
        limit = check.max_absolute_increase
        bounds.append(
            _Bound(
                name="max_absolute_increase",
                threshold=baseline + limit,
                delta_bound=limit,
                violated=_above(delta, limit),
            )
        )
    if check.max_relative_decrease is not None:
        floor = baseline * (1.0 - check.max_relative_decrease)
        bounds.append(
            _Bound(
                name="max_relative_decrease",
                threshold=floor,
                delta_bound=floor - baseline,
                violated=_below(candidate, floor),
            )
        )
    if check.max_relative_increase is not None:
        ceiling = baseline * (1.0 + check.max_relative_increase)
        bounds.append(
            _Bound(
                name="max_relative_increase",
                threshold=ceiling,
                delta_bound=ceiling - baseline,
                violated=_above(candidate, ceiling),
            )
        )
    return bounds


def _references_baseline(check: MetricCheck) -> bool:
    """Report whether any configured bound needs a baseline value to resolve."""
    return any(
        value is not None
        for value in (
            check.max_absolute_decrease,
            check.max_absolute_increase,
            check.max_relative_decrease,
            check.max_relative_increase,
        )
    )


def _is_relative(check: MetricCheck) -> bool:
    """Report whether any configured bound is expressed as a fraction of the baseline."""
    return check.max_relative_decrease is not None or check.max_relative_increase is not None


def _has_bound(check: MetricCheck) -> bool:
    """Report whether the check constrains anything at all."""
    return check.min_value is not None or check.max_value is not None or _references_baseline(check)


# ---------------------------------------------------------------------------
# Evaluating one check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Advisory:
    """The statistics attached to a check for a human, never for the gate.

    Every field here is context. Under decision D-GATE none of it can change a
    :class:`~llm_eval_lab.models.CheckStatus`; what it can do is put a sentence
    in ``note`` saying that a verdict resolved close to its own threshold.
    """

    n_paired: int | None = None
    """Cases behind the paired statistics in this advisory, or None when there are none.

    Set only where a paired statistic was actually computed, which is the overall
    pass-rate check in paired mode. A check whose two arms were measured over
    their own populations reports None rather than the run-level intersection,
    which would otherwise put the run's case count beside a category rate
    computed over a tenth of it.
    """
    confidence_interval: tuple[float, float] | None = None
    baseline_ci: tuple[float, float] | None = None
    candidate_ci: tuple[float, float] | None = None
    p_value: float | None = None
    alpha: float = 0.05
    extra_notes: tuple[str, ...] = ()


def _intervals_overlap(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
) -> bool | None:
    """Report whether two intervals share any point, or None when one is absent."""
    if left is None or right is None:
        return None
    return left[0] <= right[1] and right[0] <= left[1]


def _straddle_note(
    interval: tuple[float, float] | None,
    bounds: Sequence[_Bound],
    status: CheckStatus,
) -> str | None:
    """Say so when the advisory interval on the delta contains a threshold.

    Tested against EVERY resolved bound rather than only the reported one. The
    reported bound is chosen for the threshold column and absolute bounds resolve
    first, so a check carrying both a ``max_value`` and a movement bound would
    otherwise report the absolute bound, whose statement is not on the delta, and
    suppress a straddle the movement bound really does have.

    The verdict does not move. What this sentence buys is that a reader can see
    the difference between a regression the data support and one that a rerun
    might not reproduce.
    """
    if interval is None:
        return None
    straddled = any(
        bound.delta_bound is not None and interval[0] < bound.delta_bound < interval[1]
        for bound in bounds
    )
    if not straddled:
        return None
    if status is CheckStatus.PASSED:
        return (
            "Advisory: the 95% interval on the delta extends past this threshold, "
            "so the pass is not comfortable."
        )
    if status in (CheckStatus.FAILED, CheckStatus.WARNING):
        return (
            "Advisory: the 95% interval on the delta also covers values on the "
            "passing side of this threshold, so the breach is not firmly established."
        )
    return None


def _status_for_violation(check: MetricCheck) -> CheckStatus:
    """Map a violated bound onto its configured severity."""
    return CheckStatus.WARNING if check.severity == "warning" else CheckStatus.FAILED


def _resolve_bounds(
    check: MetricCheck,
    baseline_value: float | None,
    candidate_value: float,
) -> tuple[_Bound | None, _Bound, list[_Bound]]:
    """Return the first violated bound, the bound to report, and all of them.

    A passing check still reports the bound it was measured against: a report row
    with an empty threshold column says nothing about how close it came. The full
    list travels too, because the advisory straddle test is a statement about
    every bound expressed on the delta rather than only the reported one.
    """
    bounds = _absolute_bounds(check, candidate_value)
    if baseline_value is not None:
        bounds.extend(_movement_bounds(check, baseline_value, candidate_value))
    breach = next((bound for bound in bounds if bound.violated), None)
    return breach, breach or bounds[0], bounds


def _blocking_gate(  # noqa: PLR0913, PLR0917 - one parameter per independent input
    check: MetricCheck,
    baseline_value: float | None,
    candidate_value: float | None,
    n_baseline: int,
    n_candidate: int,
    unavailable_reason: str | None,
) -> tuple[CheckStatus, str] | None:
    """Return the status that stops a check short, with the reason, or None.

    The five ways a threshold cannot be resolved, in the order they are tested.
    Each one produces a status a reader can see; none of them is a silent pass.
    """
    if unavailable_reason is not None:
        return CheckStatus.INSUFFICIENT_DATA, unavailable_reason
    if candidate_value is None and baseline_value is None:
        # Symmetric absence is tested FIRST. A metric neither run reports is not
        # evidence about the candidate, and calling it a regression fails a build
        # for a candidate identical to its baseline.
        return CheckStatus.INSUFFICIENT_DATA, BOTH_ABSENT_NOTE
    if candidate_value is None or (baseline_value is None and _references_baseline(check)):
        return CheckStatus.MISSING_METRIC, MISSING_METRIC_NOTE
    if min(n_baseline, n_candidate) < check.min_samples:
        return (
            CheckStatus.INSUFFICIENT_DATA,
            (
                f"Needs at least {check.min_samples} samples in both runs; this "
                f"comparison has {n_baseline} and {n_candidate}."
            ),
        )
    if _is_relative(check) and baseline_value == 0.0:
        return (
            CheckStatus.INSUFFICIENT_DATA,
            (
                "A relative bound has no meaning against a baseline of zero, so this "
                "threshold was not resolved rather than being read as always passing."
            ),
        )
    return None


def evaluate_check(  # noqa: PLR0913 - one parameter per independent input
    check: MetricCheck,
    baseline_value: float | None,
    candidate_value: float | None,
    n_baseline: int | None,
    n_candidate: int | None,
    *,
    metric: str | None = None,
    label: str | None = None,
    advisory: Advisory | None = None,
    unavailable_reason: str | None = None,
) -> CheckOutcome:
    """Resolve one threshold against one pair of point estimates.

    Evaluation lives here rather than on :class:`~llm_eval_lab.models.MetricCheck`
    deliberately: gate semantics are engine behaviour, and a frozen contract
    model carrying them would force every later correction through the
    frozen-file protocol.

    The resolution order is fixed, and each step produces a VISIBLE outcome
    rather than a silent skip. An unavailable instance, then a missing metric,
    then the sample-size gate, then a relative bound with nothing to be relative
    to, and only then the bounds themselves.

    Args:
        check: The threshold to apply.
        baseline_value: The baseline point estimate, or None when absent.
        candidate_value: The candidate point estimate, or None when absent.
        n_baseline: Samples behind the baseline value. None is treated as zero,
            and therefore as insufficient: an unmeasured population and an
            undersized one are the same claim about how far to trust the number.
        n_candidate: Samples behind the candidate value, on the same terms.
        metric: The RESOLVED metric path, when it differs from ``check.metric``
            because a wildcard was expanded.
        label: The resolved display label.
        advisory: Statistics to attach for a human. Never affects the status.
        unavailable_reason: Why this check could not be evaluated at all, when
            the caller already knows. Produces ``INSUFFICIENT_DATA``.

    Returns:
        The outcome, always populated: there is no "no result" case.

    Raises:
        EvaluatorConfigError: when the check constrains nothing.
    """
    if not _has_bound(check):
        msg = f"threshold for metric {check.metric!r} configures no bound to check against"
        raise EvaluatorConfigError(msg)

    context = advisory or Advisory()
    resolved_metric = metric or check.metric
    resolved_label = label or check.label or resolved_metric
    counted_baseline = n_baseline or 0
    counted_candidate = n_candidate or 0
    notes: list[str] = []

    delta: float | None = None
    relative_delta: float | None = None
    if baseline_value is not None and candidate_value is not None:
        delta = candidate_value - baseline_value
        if baseline_value != 0.0:
            relative_delta = delta / baseline_value

    overlap = _intervals_overlap(context.baseline_ci, context.candidate_ci)

    def build(
        status: CheckStatus,
        *,
        violated_bound: str | None = None,
        threshold_value: float | None = None,
    ) -> CheckOutcome:
        """Assemble the outcome, so every early return carries the same fields."""
        significant = None if context.p_value is None else context.p_value < context.alpha
        text = " ".join(notes) if notes else None
        return CheckOutcome(
            metric=resolved_metric,
            label=resolved_label,
            direction=check.direction,
            status=status,
            baseline=baseline_value,
            candidate=candidate_value,
            delta=delta,
            relative_delta=relative_delta,
            violated_bound=violated_bound,
            threshold_value=threshold_value,
            n_baseline=counted_baseline,
            n_candidate=counted_candidate,
            n_paired=context.n_paired,
            confidence_interval=context.confidence_interval,
            baseline_ci=context.baseline_ci,
            candidate_ci=context.candidate_ci,
            intervals_overlap=overlap,
            p_value=context.p_value,
            significant=significant,
            note=text,
        )

    notes.extend(context.extra_notes)

    blocked = _blocking_gate(
        check,
        baseline_value,
        candidate_value,
        counted_baseline,
        counted_candidate,
        unavailable_reason,
    )
    if blocked is not None:
        status, reason = blocked
        notes.append(reason)
        return build(status)

    if candidate_value is None:
        # `_blocking_gate` has already answered this case. Restating it here is
        # what narrows the type for the arithmetic below, and it reuses the same
        # note constant so the two paths cannot describe it differently.
        notes.append(MISSING_METRIC_NOTE)
        return build(CheckStatus.MISSING_METRIC)

    breach, reported, bounds = _resolve_bounds(check, baseline_value, candidate_value)
    status = _status_for_violation(check) if breach is not None else CheckStatus.PASSED

    straddle = _straddle_note(context.confidence_interval, bounds, status)
    if straddle is not None:
        notes.append(straddle)
    if overlap:
        notes.append(
            "Advisory: the two runs' 95% intervals overlap, so this comparison is "
            "inconclusive even where the point estimates differ."
        )

    return build(
        status,
        violated_bound=breach.name if breach is not None else None,
        threshold_value=reported.threshold,
    )


# ---------------------------------------------------------------------------
# Metric paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricValue:
    """One run's figure for one metric, with the population and interval behind it."""

    value: float | None
    n: int | None
    interval: tuple[float, float] | None = None


ABSENT = MetricValue(value=None, n=None, interval=None)
"""What a run reports for a metric it does not carry."""


def validate_metric_path(path: str) -> None:
    """Reject a metric path no rollup can answer.

    Raised at policy-load time rather than left to surface as a puzzling
    ``MISSING_METRIC`` at compare time, because a typo in a YAML file is a
    configuration defect and should read as one.

    Raises:
        EvaluatorConfigError: when the path is not addressable.
    """
    segments = path.split(".")
    if len(segments) == 1 and segments[0] in _ROOT_METRICS:
        return
    if (
        len(segments) == _LATENCY_PATH_SEGMENTS
        and segments[0] == "latency"
        and segments[1] in _LATENCY_FIELDS
    ):
        return
    if (
        len(segments) == _INSTANCE_PATH_SEGMENTS
        and segments[0] in _NAMESPACES
        and segments[2] in _BUCKET_LEAVES
    ):
        return
    known = (
        f"{sorted(_ROOT_METRICS)}, latency.<{'|'.join(sorted(_LATENCY_FIELDS))}>, "
        f"<{'|'.join(sorted(_NAMESPACES))}>.<name or *>.<{'|'.join(sorted(_BUCKET_LEAVES))}>"
    )
    msg = f"unknown threshold metric {path!r}; addressable metrics are {known}"
    raise EvaluatorConfigError(msg)


def _root_value(metrics: AggregateMetrics, name: str) -> MetricValue:
    """Resolve one of the rollup's headline figures."""
    if name == "pass_rate":
        return MetricValue(
            value=metrics.pass_rate,
            n=metrics.n_pass_denominator,
            interval=metrics.pass_rate_ci,
        )
    if name == "error_rate":
        # The one interval computed here rather than read off the rollup: the
        # rollup stores the rate and the two counts but not an interval for it.
        interval = (
            None
            if metrics.n_completed <= 0
            else wilson_interval(metrics.n_errors, metrics.n_completed)[1:]
        )
        return MetricValue(value=metrics.error_rate, n=metrics.n_completed, interval=interval)
    if name == "mean_score":
        return MetricValue(value=metrics.mean_score, n=metrics.n_scored)
    return MetricValue(value=metrics.median_score, n=metrics.n_scored)


def _latency_value(metrics: AggregateMetrics, field: str) -> MetricValue:
    """Resolve one latency field, over the successful-attempt population."""
    value: float | None = getattr(metrics.latency, field)
    return MetricValue(value=value, n=metrics.latency.n)


def _bucket_value(bucket: CategoryMetrics, leaf: str) -> MetricValue:
    """Resolve one figure of a category or tag bucket."""
    if leaf == "pass_rate":
        return MetricValue(value=bucket.pass_rate, n=bucket.n, interval=bucket.pass_rate_ci)
    return MetricValue(value=bucket.mean_score, n=bucket.n)


def _evaluator_value(bucket: EvaluatorMetrics, leaf: str) -> MetricValue:
    """Resolve one figure of an evaluator rollup.

    The pass rate is gated on, and interval-ed against, ``n_pass_denominator`` -
    evaluations actually eligible for pass/fail - rather than ``n``, which counts
    every evaluation including the errored ones. Those are different populations,
    and measuring a ``min_samples`` gate against one while the rate it gates is
    over the other is wrong the moment anybody raises the gate above one. The
    rollup carries the denominator, so this module never re-derives the exclusion
    rule :mod:`llm_eval_lab.scoring` owns.

    A rollup persisted before ``n_pass_denominator`` existed reports zero, which reads
    as an unknown population and therefore as insufficient. That is the honest
    answer: recovering the denominator from the rate is float-lossy and undefined
    at a rate of zero.

    ``mean_score`` keeps ``n``. The count of scored evaluations is not recorded,
    and a mean has no Wilson interval to attach in any case.
    """
    if leaf == "pass_rate":
        interval = (
            None
            if bucket.n_pass_denominator <= 0
            else wilson_interval(bucket.n_passed, bucket.n_pass_denominator)[1:]
        )
        return MetricValue(value=bucket.pass_rate, n=bucket.n_pass_denominator, interval=interval)
    return MetricValue(value=bucket.mean_score, n=bucket.n)


def _select_buckets(
    buckets: Mapping[str, CategoryMetrics], selector: str, leaf: str
) -> dict[str, MetricValue]:
    """Pick the category or tag buckets a selector names."""
    keys = sorted(buckets) if selector == "*" else [selector] if selector in buckets else []
    return {key: _bucket_value(buckets[key], leaf) for key in keys}


def _select_evaluators(
    buckets: Mapping[str, EvaluatorMetrics],
    namespace: str,
    selector: str,
    leaf: str,
) -> dict[str, MetricValue]:
    """Pick evaluator instances by id, or by evaluator TYPE.

    ``evaluator_type.json_valid.pass_rate`` selects every evaluator instance of
    that type and produces one outcome per instance. It never sums them into a
    single rate: the per-evaluator denominators are not recorded, and inventing
    a combined one would be arithmetic on numbers this module does not own.
    """
    if namespace == "evaluator":
        keys = sorted(buckets) if selector == "*" else [selector] if selector in buckets else []
    else:
        keys = sorted(
            key for key, bucket in buckets.items() if selector in ("*", bucket.evaluator_type)
        )
    return {key: _evaluator_value(buckets[key], leaf) for key in keys}


def resolve_metric(metrics: AggregateMetrics, path: str) -> MetricValue:
    """Resolve a non-instance metric path against one run's rollup.

    Raises:
        EvaluatorConfigError: when the path is not addressable.
    """
    validate_metric_path(path)
    segments = path.split(".")
    if len(segments) == 1:
        return _root_value(metrics, segments[0])
    if len(segments) == _LATENCY_PATH_SEGMENTS:
        return _latency_value(metrics, segments[1])
    msg = f"metric {path!r} names a set of instances; use expand_check instead"
    raise EvaluatorConfigError(msg)


def _namespace_instances(
    metrics: AggregateMetrics, namespace: str, selector: str, leaf: str
) -> dict[str, MetricValue]:
    """Return every instance of one namespace that the selector matches."""
    if namespace == "category":
        return _select_buckets(metrics.by_category, selector, leaf)
    if namespace == "tag":
        return _select_buckets(metrics.by_tag, selector, leaf)
    return _select_evaluators(metrics.by_evaluator, namespace, selector, leaf)


@dataclass(frozen=True)
class ResolvedCheck:
    """One check bound to one concrete pair of values, ready to evaluate."""

    check: MetricCheck
    metric: str
    label: str
    baseline: MetricValue
    candidate: MetricValue
    unavailable_reason: str | None = None
    notes: tuple[str, ...] = ()


def _instance_label(check: MetricCheck, key: str, metric: str) -> str:
    """Label one expanded instance so a report row names which instance it is."""
    return f"{check.label} ({key})" if check.label else metric


def expand_check(
    check: MetricCheck,
    baseline: AggregateMetrics,
    candidate: AggregateMetrics,
) -> list[ResolvedCheck]:
    """Bind one policy entry to the concrete metrics of two runs.

    A path naming one figure yields exactly one resolved check. A path naming a
    SET of instances (a category, a tag, an evaluator, an evaluator type) yields
    one per instance present in either run, so a category that appeared or
    disappeared is reported rather than quietly dropped.

    An instance selector that matches nothing in either run still yields one
    resolved check, marked unavailable. The alternative is a policy line that
    looks enforced and is not, which is the silence this engine exists to avoid.

    Raises:
        EvaluatorConfigError: when the metric path is not addressable.
    """
    validate_metric_path(check.metric)
    segments = check.metric.split(".")
    if len(segments) < _INSTANCE_PATH_SEGMENTS:
        notes = (LATENCY_NOTE,) if segments[0] == "latency" else ()
        return [
            ResolvedCheck(
                check=check,
                metric=check.metric,
                label=check.label or check.metric,
                baseline=resolve_metric(baseline, check.metric),
                candidate=resolve_metric(candidate, check.metric),
                notes=notes,
            )
        ]

    namespace, selector, leaf = segments
    baseline_instances = _namespace_instances(baseline, namespace, selector, leaf)
    candidate_instances = _namespace_instances(candidate, namespace, selector, leaf)
    keys = sorted(set(baseline_instances) | set(candidate_instances))
    if not keys:
        target = "any instance" if selector == "*" else f"{selector!r}"
        return [
            ResolvedCheck(
                check=check,
                metric=check.metric,
                label=check.label or check.metric,
                baseline=ABSENT,
                candidate=ABSENT,
                unavailable_reason=(
                    f"Not applicable: neither run reports {target} under {namespace!r}."
                ),
            )
        ]

    # An `evaluator_type` selector matches evaluator INSTANCES, so the resolved
    # path is emitted under the namespace that actually addresses an instance id.
    # `evaluator_type.<instance id>.<leaf>` would validate and then resolve to
    # nothing, so a reader tightening a policy by copying the metric string out of
    # a failing report would get a check that looks enforced and enforces nothing.
    # The label still names the instance for a human.
    emitted = "evaluator" if namespace == "evaluator_type" else namespace
    resolved: list[ResolvedCheck] = []
    for key in keys:
        metric = f"{emitted}.{key}.{leaf}"
        in_baseline = key in baseline_instances
        in_candidate = key in candidate_instances
        reason = None
        if not (in_baseline and in_candidate):
            sides = ("candidate", "baseline") if in_candidate else ("baseline", "candidate")
            present, absent = sides
            reason = (
                f"Present only in the {present} run, so there is nothing in the "
                f"{absent} run to compare it against."
            )
        resolved.append(
            ResolvedCheck(
                check=check,
                metric=metric,
                label=_instance_label(check, key, metric),
                baseline=baseline_instances.get(key, ABSENT),
                candidate=candidate_instances.get(key, ABSENT),
                unavailable_reason=reason,
            )
        )
    return resolved


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def build_thresholds(document: Mapping[str, Any]) -> RegressionThresholds:
    """Validate a parsed policy document into a threshold policy.

    Every metric path is checked here, so a mistyped metric is a configuration
    error at load rather than a ``MISSING_METRIC`` row at compare time.

    Raises:
        EvaluatorConfigError: when the document does not describe a usable
            policy, names an unaddressable metric, sets the reserved
            ``require_significant`` flag, or configures a check with no bound.
    """
    try:
        thresholds = RegressionThresholds.model_validate(dict(document))
    except ValidationError as exc:
        msg = f"threshold policy is invalid: {exc}"
        raise EvaluatorConfigError(msg) from exc
    for check in thresholds.checks:
        validate_metric_path(check.metric)
        if not _has_bound(check):
            msg = (
                f"threshold for metric {check.metric!r} configures no bound, so it "
                f"would report a pass without checking anything"
            )
            raise EvaluatorConfigError(msg)
    return thresholds


def load_thresholds(path: Path | str | None = None) -> RegressionThresholds:
    """Load a threshold policy from `path`, or the one shipped with the package.

    Raises:
        EvaluatorConfigError: when the file is missing, unparseable or invalid.
    """
    resolved = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read threshold policy {resolved}: {exc.strerror or exc}"
        raise EvaluatorConfigError(msg) from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"threshold policy {resolved} is not valid YAML: {exc}"
        raise EvaluatorConfigError(msg) from exc
    if not isinstance(document, dict):
        kind = type(document).__name__
        msg = f"threshold policy {resolved} must be a mapping, got {kind}"
        raise EvaluatorConfigError(msg)
    return build_thresholds(document)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunComparisonInput:
    """One side of a comparison: who it is, what it measured, and every case.

    Deliberately not a :class:`~llm_eval_lab.models.Run`. The reporting layer
    must not need a persisted row to compare two sets of results, which is what
    lets this engine be unit-tested against fixtures rather than a database.
    """

    run_id: str
    label: str | None
    suite_hash: str
    metrics: AggregateMetrics
    results: Sequence[CaseResult]
    error_policy: str = "exclude"


@dataclass(frozen=True)
class Pairing:
    """How the two runs' case sets line up."""

    paired_ids: tuple[str, ...]
    changed_cases: tuple[str, ...]
    only_in_baseline: tuple[str, ...]
    only_in_candidate: tuple[str, ...]


def pair_cases(
    baseline: Sequence[CaseResult],
    candidate: Sequence[CaseResult],
) -> Pairing:
    """Line the two runs' cases up on ``(case_id, case_hash)``.

    A case whose id matches but whose hash differs is a different case wearing
    the same name: the prompt, the expected answer or the evaluators changed
    underneath it. It goes to ``changed_cases`` and is excluded from every
    paired statistic. Comparing it would report a benchmark edit as a model
    regression.
    """
    baseline_by_id = {result.case_id: result for result in baseline}
    candidate_by_id = {result.case_id: result for result in candidate}
    shared = sorted(set(baseline_by_id) & set(candidate_by_id))
    paired = tuple(
        case_id
        for case_id in shared
        if baseline_by_id[case_id].case_hash == candidate_by_id[case_id].case_hash
    )
    # Hoisted out of the generator, where this set was rebuilt once per shared
    # case and made pairing quadratic: half a second at five thousand cases.
    paired_ids = set(paired)
    changed = tuple(case_id for case_id in shared if case_id not in paired_ids)
    return Pairing(
        paired_ids=paired,
        changed_cases=changed,
        only_in_baseline=tuple(sorted(set(baseline_by_id) - set(candidate_by_id))),
        only_in_candidate=tuple(sorted(set(candidate_by_id) - set(baseline_by_id))),
    )


def case_outcome(result: CaseResult, *, error_policy: str) -> bool | None:
    """Reduce one case to the boolean a concordance table needs, or None.

    Goes through :func:`llm_eval_lab.scoring.verdict_of` and the same exclusion
    rule every pass rate in the project uses, so the paired table cannot count a
    case the run's own pass rate excluded.
    """
    policy = normalize_error_policy(error_policy)
    verdict = verdict_of(result.status, passed=result.passed)
    if verdict.errored:
        return False if policy == "fail" else None
    return verdict.passed


@dataclass(frozen=True)
class Concordance:
    """The paired 2x2 table, and the two pass rates it implies."""

    a: int
    b: int
    c: int
    d: int

    @property
    def n(self) -> int:
        """Cases contributing to the table."""
        return self.a + self.b + self.c + self.d

    @property
    def baseline_rate(self) -> float | None:
        """The baseline pass rate over the paired cases."""
        return (self.a + self.b) / self.n if self.n else None

    @property
    def candidate_rate(self) -> float | None:
        """The candidate pass rate over the paired cases."""
        return (self.a + self.c) / self.n if self.n else None


def concordance(
    baseline: Sequence[CaseResult],
    candidate: Sequence[CaseResult],
    paired_ids: Sequence[str],
    *,
    baseline_policy: str = "exclude",
    candidate_policy: str = "exclude",
) -> Concordance:
    """Build the paired 2x2 table over cases both runs reached a verdict on.

    Each side is reduced under ITS OWN error policy, because that is the policy
    its own reported pass rate used. A case either run excluded contributes to
    neither cell: a table that counted it on one side only would not be paired.
    """
    baseline_by_id = {result.case_id: result for result in baseline}
    candidate_by_id = {result.case_id: result for result in candidate}
    cells = {"a": 0, "b": 0, "c": 0, "d": 0}
    for case_id in paired_ids:
        left = case_outcome(baseline_by_id[case_id], error_policy=baseline_policy)
        right = case_outcome(candidate_by_id[case_id], error_policy=candidate_policy)
        if left is None or right is None:
            continue
        if left and right:
            cells["a"] += 1
        elif left:
            cells["b"] += 1
        elif right:
            cells["c"] += 1
        else:
            cells["d"] += 1
    return Concordance(a=cells["a"], b=cells["b"], c=cells["c"], d=cells["d"])


def _case_delta(baseline: CaseResult, candidate: CaseResult, policy: tuple[str, str]) -> CaseDelta:
    """Describe how one paired case moved."""
    drop = (
        candidate.score - baseline.score
        if baseline.score is not None and candidate.score is not None
        else None
    )
    return CaseDelta(
        case_id=baseline.case_id,
        baseline_score=baseline.score,
        candidate_score=candidate.score,
        delta=drop,
        baseline_passed=case_outcome(baseline, error_policy=policy[0]),
        candidate_passed=case_outcome(candidate, error_policy=policy[1]),
        category=candidate.category if candidate.category is not None else baseline.category,
    )


def summarize(
    baseline: RunComparisonInput,
    candidate: RunComparisonInput,
    paired_ids: Sequence[str],
) -> RegressionSummary:
    """Say which paired cases flipped, and which lost the most score.

    Computed over the PAIRED ids only. A case whose hash changed, or that only
    one run ran, cannot be said to have newly failed; naming it here would be a
    claim the data does not support.
    """
    baseline_by_id = {result.case_id: result for result in baseline.results}
    candidate_by_id = {result.case_id: result for result in candidate.results}
    policy = (baseline.error_policy, candidate.error_policy)

    newly_failing: list[str] = []
    newly_passing: list[str] = []
    still_failing: list[str] = []
    deltas: list[CaseDelta] = []
    for case_id in paired_ids:
        moved = _case_delta(baseline_by_id[case_id], candidate_by_id[case_id], policy)
        if moved.delta is not None and moved.delta < 0:
            deltas.append(moved)
        if moved.baseline_passed is None or moved.candidate_passed is None:
            continue
        if moved.baseline_passed and not moved.candidate_passed:
            newly_failing.append(case_id)
        elif not moved.baseline_passed and moved.candidate_passed:
            newly_passing.append(case_id)
        elif not moved.baseline_passed and not moved.candidate_passed:
            still_failing.append(case_id)
    deltas.sort(key=lambda item: (item.delta or 0.0, item.case_id))
    return RegressionSummary(
        newly_failing=tuple(newly_failing),
        newly_passing=tuple(newly_passing),
        still_failing=tuple(still_failing),
        n_flipped=len(newly_failing) + len(newly_passing),
        largest_score_drops=tuple(deltas[:MAX_SCORE_DROPS]),
    )


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _verdict(checks: Sequence[CheckOutcome], thresholds: RegressionThresholds) -> Verdict:
    """Roll the check outcomes up into the verdict a CI pipeline keys off.

    ``INSUFFICIENT_DATA`` never fails a build. It is the engine saying it cannot
    answer, and turning "we do not know" into "you regressed" would make the
    gate untrustworthy in exactly the small-sample case it is most often used in.
    """
    statuses = {outcome.status for outcome in checks}
    if CheckStatus.FAILED in statuses:
        return Verdict.FAIL
    if CheckStatus.MISSING_METRIC in statuses and thresholds.fail_on_missing_metric:
        return Verdict.FAIL
    if CheckStatus.WARNING in statuses:
        return Verdict.WARN
    return Verdict.PASS


def _mode_for(
    thresholds: RegressionThresholds,
    pairing: Pairing,
) -> Literal["paired", "unpaired"]:
    """Choose the comparison mode from the policy and the case sets.

    ``paired`` needs enough cases both runs ran UNCHANGED. Where every shared id
    has a changed hash there is no matched population to run McNemar over, so the
    mode is chosen on the ``(case_id, case_hash)`` intersection rather than the
    id intersection alone.
    """
    if thresholds.paired and len(pairing.paired_ids) >= max(1, thresholds.min_paired_cases):
        return "paired"
    return "unpaired"


def _incomparable_report(
    baseline: RunComparisonInput,
    candidate: RunComparisonInput,
    thresholds: RegressionThresholds,
    pairing: Pairing,
    reason: str,
) -> RegressionReport:
    """Build the report for two runs that must not be compared at all.

    The mode is derived the same way a real comparison derives it, rather than
    hardcoded. Two runs of different suites can still share unchanged case ids,
    and a report stating a positive ``paired_case_count`` beside ``unpaired``
    reads as though the two fields disagree.
    """
    return RegressionReport(
        generated_at=utc_now(),
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        verdict=Verdict.INCOMPARABLE,
        mode=_mode_for(thresholds, pairing),
        intervals_overlap=None,
        comparable=False,
        incomparable_reason=reason,
        suite_hash_match=baseline.suite_hash == candidate.suite_hash,
        baseline_suite_hash=baseline.suite_hash,
        candidate_suite_hash=candidate.suite_hash,
        paired_case_count=len(pairing.paired_ids),
        only_in_baseline=pairing.only_in_baseline,
        only_in_candidate=pairing.only_in_candidate,
        changed_cases=pairing.changed_cases,
        thresholds_id=thresholds.id,
        thresholds_version=thresholds.version,
        checks=(),
        summary=RegressionSummary(
            newly_failing=(),
            newly_passing=(),
            still_failing=(),
            n_flipped=0,
            largest_score_drops=(),
        ),
    )


def _paired_advisory(
    table: Concordance,
    alpha: float,
    z: float,
    notes: tuple[str, ...],
) -> Advisory:
    """Attach McNemar and Method 10 to the overall pass-rate check."""
    return Advisory(
        n_paired=table.n,
        confidence_interval=newcombe_method10(table.a, table.b, table.c, table.d, z),
        p_value=mcnemar_exact(table.b, table.c),
        alpha=alpha,
        extra_notes=notes,
    )


def _policy_note(baseline: RunComparisonInput, candidate: RunComparisonInput) -> tuple[str, ...]:
    """Warn on the pass-rate check when the two runs counted errors differently."""
    if baseline.error_policy == candidate.error_policy:
        return ()
    return (
        (
            f"The two runs used different error policies ({baseline.error_policy} "
            f"versus {candidate.error_policy}), so their pass-rate denominators are "
            f"not the same population."
        ),
    )


def _resolve_checks(  # noqa: PLR0913, PLR0917 - one parameter per independent input
    thresholds: RegressionThresholds,
    baseline: RunComparisonInput,
    candidate: RunComparisonInput,
    mode: Literal["paired", "unpaired"],
    table: Concordance | None,
    alpha: float,
) -> list[CheckOutcome]:
    """Evaluate every check of the policy against the two runs."""
    z = z_for_alpha(alpha)
    outcomes: list[CheckOutcome] = []
    for check in thresholds.checks:
        for resolved in expand_check(check, baseline.metrics, candidate.metrics):
            is_overall_pass_rate = resolved.metric == "pass_rate"
            baseline_value = resolved.baseline
            candidate_value = resolved.candidate
            notes = resolved.notes
            if is_overall_pass_rate:
                notes = (*notes, *_policy_note(baseline, candidate))

            # The overall pass rate is the ONE check whose two arms are replaced
            # by the paired intersection's own rates, because that is the
            # population McNemar and Method 10 are statements about. Comparing
            # the runs' whole-population rates and then attaching a paired
            # interval to them would put a figure and its interval on two
            # different denominators.
            substituted = (
                mode == "paired" and is_overall_pass_rate and table is not None and table.n > 0
            )
            if substituted and table is not None:
                advisory = _paired_advisory(
                    table,
                    alpha,
                    z,
                    (*notes, *_paired_population_note(baseline, candidate, table)),
                )
                baseline_value = MetricValue(value=table.baseline_rate, n=table.n)
                candidate_value = MetricValue(value=table.candidate_rate, n=table.n)
            else:
                # Every other check carries each arm's own Wilson interval, in
                # BOTH modes. Methodology 2.5 asks for an interval beside every
                # point estimate, and a category rate under a red FAILED badge
                # with no uncertainty beside it is the manufactured precision that
                # rule exists to prevent. The marginals are withheld only where
                # Method 10 replaces them: on a paired difference the two
                # marginals ignore the correlation and invite the "they overlap so
                # there is no difference" fallacy, and the delta interval is the
                # correct object.
                #
                # `n_paired` stays None here. It is the sample size BEHIND a
                # paired statistic, and this branch computes none: reporting the
                # run-level intersection would put 30 next to a ten-case category
                # check whose two arms were measured over ten.
                advisory = Advisory(
                    baseline_ci=baseline_value.interval,
                    candidate_ci=candidate_value.interval,
                    alpha=alpha,
                    extra_notes=notes,
                )

            outcomes.append(
                evaluate_check(
                    resolved.check,
                    baseline_value.value,
                    candidate_value.value,
                    baseline_value.n,
                    candidate_value.n,
                    metric=resolved.metric,
                    label=resolved.label,
                    advisory=advisory,
                    unavailable_reason=resolved.unavailable_reason,
                )
            )
    return outcomes


def _paired_population_note(
    baseline: RunComparisonInput,
    candidate: RunComparisonInput,
    table: Concordance,
) -> tuple[str, ...]:
    """Say so when the paired pass rate is over fewer cases than either run ran."""
    if (
        table.n == baseline.metrics.n_pass_denominator
        and table.n == candidate.metrics.n_pass_denominator
    ):
        return ()
    return (
        (
            f"Computed over the {table.n} cases both runs ran unchanged and both "
            f"reached a verdict on, not over each run's full population "
            f"({baseline.metrics.n_pass_denominator} and "
            f"{candidate.metrics.n_pass_denominator})."
        ),
    )


def compare_runs(
    baseline: RunComparisonInput,
    candidate: RunComparisonInput,
    thresholds: RegressionThresholds,
) -> RegressionReport:
    """Compare a candidate run against a baseline and return the structured verdict.

    Raises:
        RegressionInputError: when a rollup does not belong to the run it is
            presented with, which would make every figure in the report wrong.
        EvaluatorConfigError: when the policy names an unaddressable metric.
    """
    for side in (baseline, candidate):
        if side.metrics.run_id != side.run_id:
            msg = f"metrics for run {side.metrics.run_id!r} were supplied for run {side.run_id!r}"
            raise RegressionInputError(msg)

    pairing = pair_cases(baseline.results, candidate.results)
    if baseline.suite_hash != candidate.suite_hash and thresholds.require_same_suite:
        reason = (
            f"the two runs used different benchmark suites "
            f"({baseline.suite_hash} versus {candidate.suite_hash}); pass "
            f"require_same_suite: false to compare them anyway"
        )
        return _incomparable_report(baseline, candidate, thresholds, pairing, reason)

    mode = _mode_for(thresholds, pairing)
    table = (
        concordance(
            baseline.results,
            candidate.results,
            pairing.paired_ids,
            baseline_policy=baseline.error_policy,
            candidate_policy=candidate.error_policy,
        )
        if mode == "paired"
        else None
    )
    alpha = round(1.0 - thresholds.confidence_level, 10)
    checks = _resolve_checks(thresholds, baseline, candidate, mode, table, alpha)
    overall = next((outcome for outcome in checks if outcome.metric == "pass_rate"), None)

    return RegressionReport(
        generated_at=utc_now(),
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        verdict=_verdict(checks, thresholds),
        mode=mode,
        intervals_overlap=None if overall is None else overall.intervals_overlap,
        comparable=True,
        incomparable_reason=None,
        suite_hash_match=baseline.suite_hash == candidate.suite_hash,
        baseline_suite_hash=baseline.suite_hash,
        candidate_suite_hash=candidate.suite_hash,
        paired_case_count=len(pairing.paired_ids),
        only_in_baseline=pairing.only_in_baseline,
        only_in_candidate=pairing.only_in_candidate,
        changed_cases=pairing.changed_cases,
        thresholds_id=thresholds.id,
        thresholds_version=thresholds.version,
        checks=tuple(checks),
        summary=summarize(baseline, candidate, pairing.paired_ids),
    )


__all__ = [
    "ABSENT",
    "DEFAULT_THRESHOLDS_PATH",
    "LATENCY_NOTE",
    "MAX_SCORE_DROPS",
    "Advisory",
    "Concordance",
    "MetricValue",
    "Pairing",
    "ResolvedCheck",
    "RunComparisonInput",
    "build_thresholds",
    "case_outcome",
    "compare_runs",
    "concordance",
    "evaluate_check",
    "expand_check",
    "load_thresholds",
    "pair_cases",
    "resolve_metric",
    "summarize",
    "validate_metric_path",
]
