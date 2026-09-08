"""Appendix B, transcribed as executable expectations.

Every value here was computed independently of the implementation, with
``z = 1.959963984540054`` and ``z_beta = 0.8416212335729143``, and the
implementation must reproduce it. The tolerance is ``1e-4`` everywhere except
the two minimum-detectable-difference cases, which come out of a bisection
inversion rather than a closed form and use ``1e-3``.

Two rows of the appendix are deliberately absent: the continuity-corrected
McNemar chi-squared and the naive Wald paired-difference interval. They are
reference values showing what the chosen methods are being compared against.
Neither is implemented, so neither is tested.
"""

from __future__ import annotations

import math

import pytest

from llm_eval_lab.reporting.statistics import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    P90_MIN_N,
    Z_95,
    Z_POWER_80,
    low_confidence_percentiles,
    mcnemar_exact,
    minimum_detectable_difference,
    newcombe_method10,
    percentile,
    phi_coefficient,
    required_n,
    wilson_interval,
    z_for_alpha,
    z_for_power,
)

TOLERANCE = 1e-4
"""Every closed-form fixture in Appendix B is met to this."""

MDD_TOLERANCE = 1e-3
"""The bisection inversion is a numeric result and gets its own, looser bound."""

LATENCY_FIXTURE = [
    102,
    98,
    115,
    121,
    99,
    105,
    110,
    130,
    95,
    108,
    112,
    119,
    101,
    125,
    140,
    97,
    103,
    117,
    108,
    300,
]
"""Appendix B.2, unsorted exactly as published."""

PAIRED_FIXTURE = (40, 5, 2, 3)
"""Appendix B.3: a=40 both passed, b=5 baseline only, c=2 candidate only, d=3 neither."""


def close(actual: float, expected: float, tolerance: float = TOLERANCE) -> bool:
    """Report whether `actual` matches `expected` within `tolerance`."""
    return abs(actual - expected) < tolerance


# --- B.1 Wilson score interval -------------------------------------------


@pytest.mark.parametrize(
    ("x", "n", "expected"),
    [
        (45, 50, (0.9000, 0.7864, 0.9565)),
        (18, 20, (0.9000, 0.6990, 0.9721)),
        (40, 50, (0.8000, 0.6696, 0.8876)),
        (43, 50, (0.8600, 0.7381, 0.9305)),
        (38, 50, (0.7600, 0.6259, 0.8570)),
        (0, 20, (0.0000, 0.0000, 0.1611)),
        (20, 20, (1.0000, 0.8389, 1.0000)),
    ],
)
def test_wilson_interval_matches_appendix_b1(
    x: int, n: int, expected: tuple[float, float, float]
) -> None:
    p_hat, lower, upper = wilson_interval(x, n)
    assert close(p_hat, expected[0]), f"p_hat for {x}/{n}"
    assert close(lower, expected[1]), f"lower bound for {x}/{n}"
    assert close(upper, expected[2]), f"upper bound for {x}/{n}"


@pytest.mark.parametrize(("x", "n"), [(0, 20), (20, 20), (1, 3), (7, 7), (0, 1)])
def test_wilson_bounds_never_leave_the_unit_interval(x: int, n: int) -> None:
    _, lower, upper = wilson_interval(x, n)
    assert 0.0 <= lower <= upper <= 1.0


def test_wilson_interval_covers_the_point_estimate() -> None:
    p_hat, lower, upper = wilson_interval(43, 50)
    assert lower < p_hat < upper


@pytest.mark.parametrize(("x", "n"), [(1, 0), (-1, 10), (11, 10), (0, -5)])
def test_wilson_interval_refuses_an_impossible_count(x: int, n: int) -> None:
    with pytest.raises(ValueError, match="wilson_interval"):
        wilson_interval(x, n)


# --- B.2 Percentiles ------------------------------------------------------


def test_percentile_fixture_mean_matches_appendix_b2() -> None:
    mean = sum(LATENCY_FIXTURE) / len(LATENCY_FIXTURE)
    assert close(mean, 120.25)


@pytest.mark.parametrize(("p", "expected"), [(50, 109.0), (95, 148.0), (99, 269.6)])
def test_percentile_matches_appendix_b2(p: float, expected: float) -> None:
    assert close(percentile(LATENCY_FIXTURE, p), expected)


def test_percentile_rejects_nearest_rank() -> None:
    # Nearest-rank would collapse p99 onto the sample maximum at n=20, and p95
    # onto the second largest. Both are the rejected method, not this one.
    assert percentile(LATENCY_FIXTURE, 99) != 300
    assert percentile(LATENCY_FIXTURE, 95) != 140


def test_percentile_endpoints_are_the_extremes() -> None:
    assert percentile(LATENCY_FIXTURE, 0) == 95.0
    assert percentile(LATENCY_FIXTURE, 100) == 300.0


def test_percentile_of_a_single_sample_is_that_sample() -> None:
    assert percentile([42.0], 99) == 42.0


def test_percentile_does_not_depend_on_input_order() -> None:
    assert percentile(sorted(LATENCY_FIXTURE), 95) == percentile(LATENCY_FIXTURE, 95)


def test_percentile_refuses_an_empty_series() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        percentile([], 50)


@pytest.mark.parametrize("p", [-1.0, 100.1])
def test_percentile_refuses_a_percentile_outside_the_range(p: float) -> None:
    with pytest.raises(ValueError, match="0 <= p <= 100"):
        percentile(LATENCY_FIXTURE, p)


# --- B.3 Paired comparison ------------------------------------------------


def test_mcnemar_exact_matches_appendix_b3() -> None:
    assert close(mcnemar_exact(b=5, c=2), 0.453125)


def test_mcnemar_exact_is_symmetric() -> None:
    assert mcnemar_exact(b=5, c=2) == mcnemar_exact(b=2, c=5)


def test_mcnemar_with_no_discordant_pairs_reports_no_evidence() -> None:
    assert mcnemar_exact(b=0, c=0) == 1.0


def test_mcnemar_is_never_above_one() -> None:
    assert mcnemar_exact(b=3, c=3) <= 1.0


def test_mcnemar_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_exact(b=-1, c=2)


def test_phi_matches_appendix_b3() -> None:
    assert close(phi_coefficient(*PAIRED_FIXTURE), 0.4001)


def test_phi_of_an_empty_marginal_is_zero() -> None:
    assert phi_coefficient(0, 0, 0, 10) == 0.0


def test_newcombe_method10_matches_appendix_b3() -> None:
    lower, upper = newcombe_method10(*PAIRED_FIXTURE)
    assert close(lower, -0.1749), "lower bound, candidate minus baseline"
    assert close(upper, 0.0487), "upper bound, candidate minus baseline"


def test_newcombe_method10_is_oriented_candidate_minus_baseline() -> None:
    # The published derivation is written for baseline minus candidate. A
    # transcription that keeps the magnitudes and flips the signs would return
    # (-0.0487, 0.1749) and still look plausible, so the sign is asserted
    # against the point difference rather than only the magnitudes.
    a, b, c, d = PAIRED_FIXTURE
    n = a + b + c + d
    delta = (a + c) / n - (a + b) / n
    assert delta == pytest.approx(-0.06)

    lower, upper = newcombe_method10(a, b, c, d)
    assert lower < delta < upper
    assert (lower + upper) / 2 < 0


def test_newcombe_method10_straddles_zero_for_the_canonical_fixture() -> None:
    # A six-point drop at n=50 is not distinguishable from noise, which is the
    # advisory statement the interval exists to make next to a failing gate.
    lower, upper = newcombe_method10(*PAIRED_FIXTURE)
    assert lower < 0 < upper


def test_newcombe_method10_marginals_match_the_appendix_wilson_rows() -> None:
    a, b, c, d = PAIRED_FIXTURE
    n = a + b + c + d
    baseline = wilson_interval(a + b, n)
    candidate = wilson_interval(a + c, n)
    assert close(baseline[0], 0.9000)
    assert close(baseline[1], 0.7864)
    assert close(baseline[2], 0.9565)
    assert close(candidate[0], 0.8400)
    assert close(candidate[1], 0.7149)
    assert close(candidate[2], 0.9166)


def test_newcombe_method10_of_identical_runs_is_centred_on_zero() -> None:
    lower, upper = newcombe_method10(40, 0, 0, 10)
    assert lower < 0 < upper
    assert close((lower + upper) / 2, 0.0, tolerance=1e-9)


def test_newcombe_method10_refuses_an_empty_table() -> None:
    with pytest.raises(ValueError, match="at least one paired case"):
        newcombe_method10(0, 0, 0, 0)


def test_newcombe_method10_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        newcombe_method10(-1, 0, 0, 1)


# --- B.4 Sample size and minimum detectable difference --------------------


def test_minimum_detectable_difference_matches_appendix_b4() -> None:
    assert close(minimum_detectable_difference(n=100, p1=0.80), 0.1790, MDD_TOLERANCE)


def test_required_n_matches_appendix_b4() -> None:
    assert required_n(delta=0.179, p1=0.80) == 100


def test_required_n_returns_an_integer() -> None:
    value = required_n(delta=0.179, p1=0.80)
    assert isinstance(value, int)


@pytest.mark.parametrize(
    ("n", "p1", "expected_pp"),
    [
        (20, 0.80, 42.3),
        (20, 0.90, 39.3),
        (20, 0.50, 39.5),
        (50, 0.80, 26.0),
        (50, 0.90, 22.7),
        (50, 0.50, 26.7),
        (100, 0.80, 17.9),
        (100, 0.90, 15.0),
        (100, 0.50, 19.3),
        (200, 0.80, 12.3),
        (200, 0.90, 10.0),
        (200, 0.50, 13.8),
        (500, 0.80, 7.5),
        (500, 0.90, 5.9),
        (500, 0.50, 8.8),
    ],
)
def test_minimum_detectable_difference_reproduces_the_guidance_table(
    n: int, p1: float, expected_pp: float
) -> None:
    # The published table is quoted to one decimal place in percentage points,
    # so the comparison is made there rather than pretending to more precision.
    assert close(minimum_detectable_difference(n=n, p1=p1) * 100, expected_pp, 0.05)


def test_required_n_and_mdd_are_consistent_round_trip() -> None:
    delta = minimum_detectable_difference(n=100, p1=0.80)
    assert required_n(delta=delta, p1=0.80) == 100


def test_required_n_falls_as_the_difference_to_detect_grows() -> None:
    sizes = [required_n(delta=d, p1=0.80) for d in (0.05, 0.10, 0.20, 0.40)]
    assert sizes == sorted(sizes, reverse=True)


def test_minimum_detectable_difference_falls_as_samples_grow() -> None:
    values = [minimum_detectable_difference(n=n, p1=0.80) for n in (20, 50, 100, 200, 500)]
    assert values == sorted(values, reverse=True)


def test_minimum_detectable_difference_uses_the_pooled_form() -> None:
    # The unpooled variant substitutes sqrt(p1(1-p1) + p2(1-p2)) for the pooled
    # sqrt(2 * p_bar * (1 - p_bar)) in the alpha term. At n=100, p1=0.80 it
    # gives roughly 0.174, which is outside the 1e-3 tolerance of the pooled
    # 0.1790 the appendix publishes. So the choice is load-bearing, and this
    # asserts the shipped value is not the unpooled one.
    unpooled = 0.174
    assert not close(minimum_detectable_difference(n=100, p1=0.80), unpooled, MDD_TOLERANCE)


@pytest.mark.parametrize(("delta", "p1"), [(0.0, 0.8), (1.0, 0.8), (0.1, 1.5), (0.1, -0.1)])
def test_required_n_refuses_a_domain_violation(delta: float, p1: float) -> None:
    with pytest.raises(ValueError, match="required_n"):
        required_n(delta=delta, p1=p1)


@pytest.mark.parametrize(("n", "p1"), [(0, 0.8), (-10, 0.8), (100, 1.5)])
def test_minimum_detectable_difference_refuses_a_domain_violation(n: int, p1: float) -> None:
    with pytest.raises(ValueError, match="minimum_detectable_difference"):
        minimum_detectable_difference(n=n, p1=p1)


# --- constants and quantiles ----------------------------------------------


def test_default_quantiles_are_the_published_constants() -> None:
    assert z_for_alpha(DEFAULT_ALPHA) == Z_95
    assert z_for_power(DEFAULT_POWER) == Z_POWER_80


def test_quantiles_are_computed_for_non_default_settings() -> None:
    assert z_for_alpha(0.01) > Z_95
    assert z_for_power(0.90) > Z_POWER_80


def test_published_constants_agree_with_the_standard_normal_to_machine_precision() -> None:
    from statistics import NormalDist  # noqa: PLC0415 - a one-line check, not a module dependency

    assert math.isclose(Z_95, NormalDist().inv_cdf(0.975), rel_tol=1e-12)
    assert math.isclose(Z_POWER_80, NormalDist().inv_cdf(0.80), rel_tol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.5])
def test_z_for_alpha_refuses_a_domain_violation(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        z_for_alpha(alpha)


@pytest.mark.parametrize("power", [0.0, 1.0, 2.0])
def test_z_for_power_refuses_a_domain_violation(power: float) -> None:
    with pytest.raises(ValueError, match="power"):
        z_for_power(power)


# --- percentile confidence gates ------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, ("p50_ms", "p90_ms", "p95_ms", "p99_ms")),
        (4, ("p50_ms", "p90_ms", "p95_ms", "p99_ms")),
        (5, ("p90_ms", "p95_ms", "p99_ms")),
        (9, ("p90_ms", "p95_ms", "p99_ms")),
        (10, ("p95_ms", "p99_ms")),
        (19, ("p95_ms", "p99_ms")),
        (20, ("p99_ms",)),
        (99, ("p99_ms",)),
        (100, ()),
        (500, ()),
    ],
)
def test_low_confidence_percentiles_apply_the_documented_gates(
    n: int, expected: tuple[str, ...]
) -> None:
    assert low_confidence_percentiles(n) == expected


def test_the_p90_gate_is_ten_samples() -> None:
    assert P90_MIN_N == 10
    assert "p90_ms" in low_confidence_percentiles(P90_MIN_N - 1)
    assert "p90_ms" not in low_confidence_percentiles(P90_MIN_N)


def test_gated_names_are_reported_in_ascending_percentile_order() -> None:
    assert low_confidence_percentiles(0) == ("p50_ms", "p90_ms", "p95_ms", "p99_ms")
