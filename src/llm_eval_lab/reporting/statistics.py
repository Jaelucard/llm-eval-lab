"""The statistics behind every reported figure. Standard library only.

Six functions, and no other module in the project computes any of them:
:func:`wilson_interval`, :func:`percentile`, :func:`mcnemar_exact`,
:func:`newcombe_method10`, :func:`required_n` and
:func:`minimum_detectable_difference`. Wilson intervals, linear-interpolation
percentiles, an exact McNemar test summed with Python's arbitrary-precision
integers, Newcombe's Method 10 and one bisection are together a few dozen lines
of arithmetic, so this module depends on ``math`` and ``statistics`` and on
nothing else. No numpy, no scipy.

Three conventions that are decisions rather than details.

**Sign.** :func:`newcombe_method10` returns the interval for CANDIDATE MINUS
BASELINE, matching the sign of ``CheckOutcome.delta``. The published derivation
of Method 10 is written for baseline minus candidate; this is its
negated-and-swapped form. A transcription that keeps the magnitudes and flips
the signs still looks plausible, so the unit tests assert the interval's
midpoint agrees in sign with the point difference.

**Pooled sample size.** :func:`required_n` uses the pooled two-proportion
form. The unpooled variant gives a materially different number, so this is not
a stylistic preference.

**No invented inputs.** Every function refuses a domain violation with a
``ValueError`` rather than clamping its way to a plausible-looking answer. A
caller with no samples asks whether it has any before asking for a percentile.
"""

import math
from collections.abc import Sequence
from statistics import NormalDist

Z_95 = 1.959963984540054
"""Two-sided 95% standard normal quantile. The default for every interval here."""

Z_POWER_80 = 0.8416212335729143
"""Standard normal quantile for 80% power, used by the sample-size formulas."""

DEFAULT_ALPHA = 0.05
"""Two-sided significance level behind :data:`Z_95`."""

DEFAULT_POWER = 0.80
"""Statistical power behind :data:`Z_POWER_80`."""

P50_MIN_N = 5
"""Fewest successful samples before a median is reported without a low-confidence mark."""

P90_MIN_N = 10
"""Fewest successful samples before a 90th percentile is reported without a mark.

Ten samples is the point at which the 90th percentile stops being an
interpolation between the two largest observations, which is what a smaller
sample makes it.
"""

P95_MIN_N = 20
"""Fewest successful samples before a 95th percentile is reported without a mark."""

P99_MIN_N = 100
"""Fewest successful samples before a 99th percentile is reported without a mark."""

_MDD_SEARCH_LOW = 1e-6
"""Lower end of the bisection interval for :func:`minimum_detectable_difference`."""

_MDD_SEARCH_HIGH = 1.0 - 1e-6
"""Upper end of that bisection interval."""

_MDD_TOLERANCE = 1e-6
"""Width the bisection narrows to before returning its midpoint."""

_PERCENT_MAX = 100.0
"""Upper end of the percentile argument's domain."""

_PINNED_QUANTILES: dict[float, float] = {
    1.0 - DEFAULT_ALPHA / 2: Z_95,
    DEFAULT_POWER: Z_POWER_80,
}
"""Quantiles pinned to the digits the project's fixtures were computed with.

``NormalDist().inv_cdf`` reproduces both to within one unit in the last place,
which is irrelevant at any tolerance anyone tests against but leaves the shipped
constants and the computed ones as different floats. The fixture table is the
authority for the two standard settings, so those two values are read from it
and every other alpha or power is computed.
"""


def _normal_quantile(p: float) -> float:
    """Return the standard normal quantile at `p`, honouring the pinned values."""
    pinned = _PINNED_QUANTILES.get(p)
    return pinned if pinned is not None else NormalDist().inv_cdf(p)


def z_for_alpha(alpha: float = DEFAULT_ALPHA) -> float:
    """Return the two-sided normal quantile for significance level `alpha`.

    Args:
        alpha: Two-sided significance level, strictly between 0 and 1.

    Returns:
        The quantile at ``1 - alpha / 2``.

    Raises:
        ValueError: when `alpha` is outside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be strictly between 0 and 1, got {alpha}"
        raise ValueError(msg)
    return _normal_quantile(1.0 - alpha / 2.0)


def z_for_power(power: float = DEFAULT_POWER) -> float:
    """Return the one-sided normal quantile for statistical power `power`.

    Args:
        power: Desired power, strictly between 0 and 1.

    Returns:
        The quantile at `power`.

    Raises:
        ValueError: when `power` is outside ``(0, 1)``.
    """
    if not 0.0 < power < 1.0:
        msg = f"power must be strictly between 0 and 1, got {power}"
        raise ValueError(msg)
    return _normal_quantile(power)


def wilson_interval(x: int, n: int, z: float = Z_95) -> tuple[float, float, float]:
    """Return the point estimate and Wilson score interval for `x` successes of `n`.

    The Wilson interval is used rather than the normal approximation because it
    is well behaved at the sample sizes a benchmark actually has and at the
    boundaries a benchmark actually hits. At ``0/20`` the normal approximation
    collapses to the degenerate ``(0, 0)``; Wilson reports ``(0, 0.1611)``,
    which is the honest statement. Both bounds are clamped into ``[0, 1]``: the
    interval is a proportion and a bound outside that range would be reported
    as one.

    Args:
        x: Number of successes.
        n: Number of trials, strictly positive.
        z: Standard normal quantile for the desired confidence level.

    Returns:
        ``(p_hat, lower, upper)``.

    Raises:
        ValueError: when `n` is not positive or `x` is outside ``[0, n]``.
    """
    if n <= 0:
        msg = f"wilson_interval needs at least one trial, got n={n}"
        raise ValueError(msg)
    if not 0 <= x <= n:
        msg = f"wilson_interval needs 0 <= x <= n, got x={x} and n={n}"
        raise ValueError(msg)

    p_hat = x / n
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    centre = (p_hat + z_squared / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1.0 - p_hat) / n + z_squared / (4 * n * n))
    return p_hat, max(0.0, centre - margin), min(1.0, centre + margin)


def percentile(values: Sequence[float], p: float) -> float:
    """Return the `p`-th percentile of `values` by linear interpolation.

    The index is ``(p / 100) * (n - 1)`` over the sorted samples, interpolating
    between the two neighbouring order statistics. This is numpy's default
    method, and it is chosen over nearest-rank deliberately: at twenty samples
    nearest-rank puts p99 exactly on the sample maximum, which reports the worst
    single observation as if it were a distribution tail.

    Args:
        values: The samples. Order does not matter; they are sorted here.
        p: The percentile to compute, between 0 and 100 inclusive.

    Returns:
        The interpolated percentile.

    Raises:
        ValueError: when `values` is empty or `p` is outside ``[0, 100]``.
    """
    if not values:
        msg = "percentile needs at least one value"
        raise ValueError(msg)
    if not 0.0 <= p <= _PERCENT_MAX:
        msg = f"percentile needs 0 <= p <= 100, got {p}"
        raise ValueError(msg)

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    index = (p / _PERCENT_MAX) * (len(ordered) - 1)
    lower_index = math.floor(index)
    upper_index = math.ceil(index)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    weight = index - lower_index
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    return lower + weight * (upper - lower)


def mcnemar_exact(b: int, c: int) -> float:
    """Return the two-sided exact McNemar p-value for a paired comparison.

    Only the discordant pairs carry information: `b` cases the baseline passed
    and the candidate failed, `c` the reverse. Under the null they are a
    binomial with ``p = 0.5`` over ``b + c`` trials, and the two-sided p-value
    is twice the smaller tail, capped at 1.

    The exact test is used rather than the continuity-corrected chi-squared
    because a benchmark's discordant count is routinely in the single digits,
    where the asymptotic approximation is not trustworthy. Python's
    arbitrary-precision integers make the sum exact at any size a benchmark
    reaches, so there is nothing to trade away.

    Args:
        b: Discordant pairs where the baseline passed and the candidate failed.
        c: Discordant pairs where the candidate passed and the baseline failed.

    Returns:
        The two-sided p-value. ``1.0`` when there are no discordant pairs, which
        is the correct statement that the data contain no evidence of a change.

    Raises:
        ValueError: when either count is negative.
    """
    if b < 0 or c < 0:
        msg = f"mcnemar_exact needs non-negative counts, got b={b} and c={c}"
        raise ValueError(msg)
    n = b + c
    if n == 0:
        return 1.0
    smaller = min(b, c)
    tail = sum(math.comb(n, k) for k in range(smaller + 1))
    # `1 << n` rather than `2**n`: identical value, exact at any size Python's
    # integers reach, and typed as `int` where `int ** int` widens to `Any`.
    return min(1.0, 2.0 * tail / (1 << n))


def phi_coefficient(a: int, b: int, c: int, d: int) -> float:
    """Return the phi correlation of a paired 2x2 concordance table.

    The cells are ``a`` both passed, ``b`` baseline only, ``c`` candidate only,
    ``d`` neither. Phi is the correlation between the two runs' per-case
    outcomes, and it is what makes :func:`newcombe_method10` narrower than a
    naive unpaired interval: two runs over the same cases agree on most of them,
    and an interval that ignores that agreement overstates the uncertainty.

    Args:
        a: Cases both runs passed.
        b: Cases the baseline passed and the candidate failed.
        c: Cases the candidate passed and the baseline failed.
        d: Cases neither run passed.

    Returns:
        The phi coefficient, or ``0.0`` when a marginal is empty and the
        correlation is undefined.

    Raises:
        ValueError: when any count is negative.
    """
    if min(a, b, c, d) < 0:
        msg = f"phi_coefficient needs non-negative counts, got ({a}, {b}, {c}, {d})"
        raise ValueError(msg)
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    if denominator == 0:
        return 0.0
    return (a * d - b * c) / math.sqrt(denominator)


def newcombe_method10(
    a: int,
    b: int,
    c: int,
    d: int,
    z: float = Z_95,
) -> tuple[float, float]:
    """Return the Newcombe Method-10 interval for a paired pass-rate difference.

    **Orientation.** The returned interval is for CANDIDATE MINUS BASELINE, so
    a candidate that got worse yields a negative point difference and an
    interval centred below zero. Published derivations of Method 10 are written
    for baseline minus candidate; this is the negated-and-swapped form, which
    is what makes the interval directly comparable with
    ``CheckOutcome.delta``.

    Method 10 combines each marginal's Wilson interval with the phi correlation
    between the two runs, which is why it is used instead of a Wald interval on
    the paired difference: the naive interval at the project's canonical fixture
    is both wider and badly centred.

    The interval is advisory. Under decision D-GATE the regression verdict is
    decided by point estimates alone, so this value is displayed next to a
    verdict and never participates in it.

    Args:
        a: Cases both runs passed.
        b: Cases the baseline passed and the candidate failed.
        c: Cases the candidate passed and the baseline failed.
        d: Cases neither run passed.
        z: Standard normal quantile for the desired confidence level.

    Returns:
        ``(lower, upper)`` for candidate minus baseline.

    Raises:
        ValueError: when any count is negative or the table is empty.
    """
    if min(a, b, c, d) < 0:
        msg = f"newcombe_method10 needs non-negative counts, got ({a}, {b}, {c}, {d})"
        raise ValueError(msg)
    n = a + b + c + d
    if n == 0:
        msg = "newcombe_method10 needs at least one paired case"
        raise ValueError(msg)

    baseline = (a + b) / n
    candidate = (a + c) / n
    _, baseline_low, baseline_high = wilson_interval(a + b, n, z)
    _, candidate_low, candidate_high = wilson_interval(a + c, n, z)
    phi = phi_coefficient(a, b, c, d)

    # Written in the published baseline-minus-candidate orientation, then
    # negated and swapped once at the return. Deriving it directly in the
    # candidate-minus-baseline orientation would mean transcribing four
    # subtractions in reverse, which is precisely the error the sign assertion
    # in the unit tests exists to catch.
    delta = baseline - candidate
    lower_span = math.sqrt(
        max(
            0.0,
            (baseline - baseline_low) ** 2
            - 2.0 * phi * (baseline - baseline_low) * (candidate_high - candidate)
            + (candidate_high - candidate) ** 2,
        )
    )
    upper_span = math.sqrt(
        max(
            0.0,
            (baseline_high - baseline) ** 2
            - 2.0 * phi * (baseline_high - baseline) * (candidate - candidate_low)
            + (candidate - candidate_low) ** 2,
        )
    )
    return -(delta + upper_span), -(delta - lower_span)


def _required_n_continuous(delta: float, p1: float, z: float, z_beta: float) -> float:
    """Return the pooled two-proportion sample size, before any rounding.

    Kept separate from :func:`required_n` because
    :func:`minimum_detectable_difference` has to invert a continuous function.
    ``required_n`` returns an ``int`` and is therefore a step function, which a
    bisection converges to a step boundary of rather than to a root.

    ``p2`` is clamped into ``[0, 1]``: the bisection sweeps `delta` across the
    whole unit interval and would otherwise take the square root of a negative
    variance for any `delta` above `p1`. The clamp only ever engages far outside
    the region where the root lies.
    """
    p2 = min(1.0, max(0.0, p1 - delta))
    p_bar = (p1 + p2) / 2.0
    numerator = z * math.sqrt(2.0 * p_bar * (1.0 - p_bar)) + z_beta * math.sqrt(
        p1 * (1.0 - p1) + p2 * (1.0 - p2)
    )
    return (numerator / delta) ** 2


def required_n(
    delta: float,
    p1: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Return the per-arm sample size needed to detect a `delta` drop from `p1`.

    The pooled two-proportion form, two-sided, rounded UP: a fractional sample
    size that is rounded down is a study that does not have the power it claims.
    Rounding happens here and only here.

    The figure is for an independent two-proportion comparison and is a
    conservative upper bound for a paired benchmark, where the same cases run
    against both models are positively correlated and the design is usually more
    sensitive at the same `n`.

    Args:
        delta: The absolute pass-rate drop to detect, strictly between 0 and 1.
        p1: The baseline pass rate, between 0 and 1.
        alpha: Two-sided significance level.
        power: Desired statistical power.

    Returns:
        The number of cases needed per arm.

    Raises:
        ValueError: when `delta` or `p1` is outside its domain, or when `delta`
            is larger than `p1` and the drop is therefore infeasible (a pass
            rate cannot fall by more than it is). A baseline of ``0.0`` always
            raises, since no drop is possible from a pass rate that is already
            zero.
    """
    if not 0.0 < delta < 1.0:
        msg = f"required_n needs 0 < delta < 1, got {delta}"
        raise ValueError(msg)
    if not 0.0 <= p1 <= 1.0:
        msg = f"required_n needs 0 <= p1 <= 1, got {p1}"
        raise ValueError(msg)
    if delta > p1:
        msg = (
            f"required_n needs delta <= p1 (a drop of {delta} from a baseline of "
            f"{p1} is not possible)"
        )
        raise ValueError(msg)
    exact = _required_n_continuous(delta, p1, z_for_alpha(alpha), z_for_power(power))
    return math.ceil(exact)


def minimum_detectable_difference(
    n: int,
    p1: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """Return the smallest pass-rate drop `n` cases per arm can detect.

    This is the honest counterweight to a small benchmark. At fifty cases per
    arm with a baseline of 0.80 the answer is roughly twenty-six percentage
    points, which is what a reader needs alongside any claim that a two-point
    movement means something.

    Inverts the CONTINUOUS sample-size expression by bisection over `delta`,
    not :func:`required_n`, which returns an ``int``: a bisection against a step
    function converges to a step boundary rather than to a root. Because the
    result comes out of a numeric inversion, it reproduces the project's
    fixtures to ``1e-3`` rather than the ``1e-4`` every closed form here meets.

    The result is capped at `p1`: a drop larger than the baseline itself is not
    a possible pass-rate movement. When `n` is too small to detect even a drop
    all the way to a zero pass rate, the unconstrained root this bisects toward
    sits above `p1`, and the honest answer is the largest drop that is still
    physically possible: this returns (approximately) `p1` itself in that
    case, rather than a number larger than the baseline that no real pass-rate
    drop could ever equal.

    Args:
        n: Cases per arm, strictly positive.
        p1: The baseline pass rate, between 0 and 1.
        alpha: Two-sided significance level.
        power: Desired statistical power.

    Returns:
        The minimum detectable absolute difference in pass rate, never more
        than `p1`.

    Raises:
        ValueError: when `n` is not positive, `p1` is outside ``[0, 1]``, or
            `p1` is ``0.0``. A baseline of zero has no feasible drop at all
            (there is nothing below zero to fall to), so there is no
            detectable difference to report.
    """
    if n <= 0:
        msg = f"minimum_detectable_difference needs n > 0, got {n}"
        raise ValueError(msg)
    if not 0.0 <= p1 <= 1.0:
        msg = f"minimum_detectable_difference needs 0 <= p1 <= 1, got {p1}"
        raise ValueError(msg)
    if p1 == 0.0:
        msg = (
            "minimum_detectable_difference needs p1 > 0 (no drop is possible from a baseline of 0)"
        )
        raise ValueError(msg)

    z = z_for_alpha(alpha)
    z_beta = z_for_power(power)
    # The required sample size falls as the difference to detect grows, so the
    # bracket is ordered largest-n first and the midpoint moves the low end up
    # while it still demands more cases than are available. The bracket itself
    # is left at its original width (rather than narrowed to p1) so that every
    # already-feasible case takes the same bisection path and reproduces the
    # same float, bit for bit, as before this function learned about p1's
    # ceiling.
    low, high = _MDD_SEARCH_LOW, _MDD_SEARCH_HIGH
    while high - low > _MDD_TOLERANCE:
        middle = (low + high) / 2.0
        if _required_n_continuous(middle, p1, z, z_beta) > n:
            low = middle
        else:
            high = middle
    # A drop larger than the baseline itself is not a possible movement, so the
    # result is capped at p1. This is the only place that ceiling is enforced:
    # when n is too small to detect even a drop all the way to a zero pass
    # rate, the unconstrained root the bisection converges toward sits above
    # p1, and capping it here reports the largest drop that is still feasible
    # instead of a number no real pass rate could ever fall by.
    return min((low + high) / 2.0, p1)


def low_confidence_percentiles(n: int) -> tuple[str, ...]:
    """Return the percentile field names whose sample count is below their gate.

    The value at a gated percentile is still computed and still reported. What
    this marks is that it should not be read as trustworthy, which the CLI and
    the dashboard render as an "insufficient samples" badge. Silently omitting
    the number, or presenting it unmarked, are the two failures this exists to
    prevent.

    Args:
        n: The number of samples the percentiles were computed over.

    Returns:
        The names of the gated fields, in ascending percentile order.
    """
    gated: list[str] = []
    if n < P50_MIN_N:
        gated.append("p50_ms")
    if n < P90_MIN_N:
        gated.append("p90_ms")
    if n < P95_MIN_N:
        gated.append("p95_ms")
    if n < P99_MIN_N:
        gated.append("p99_ms")
    return tuple(gated)


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_POWER",
    "P50_MIN_N",
    "P90_MIN_N",
    "P95_MIN_N",
    "P99_MIN_N",
    "Z_95",
    "Z_POWER_80",
    "low_confidence_percentiles",
    "mcnemar_exact",
    "minimum_detectable_difference",
    "newcombe_method10",
    "percentile",
    "phi_coefficient",
    "required_n",
    "wilson_interval",
    "z_for_alpha",
    "z_for_power",
]
