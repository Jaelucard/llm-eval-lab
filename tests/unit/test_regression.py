"""The regression engine: bounds, pairing, modes, and the advisory statistics.

The canonical paired fixture is a=40, b=5, c=2, d=3 over fifty cases. It is the
same table the statistics unit tests use, so a disagreement between this
module's report and that module's arithmetic shows up as a failure here rather
than as two plausible numbers on two screens.
"""

import json
from datetime import UTC, datetime

import pytest

from llm_eval_lab.models import (
    AggregateMetrics,
    CaseResult,
    CaseStatus,
    CheckStatus,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorConfigError,
    FinishReason,
    MetricCheck,
    ModelResponse,
    RegressionInputError,
    RegressionReport,
    RegressionThresholds,
    ThresholdDirection,
    Verdict,
)
from llm_eval_lab.reporting.aggregate import compute_metrics
from llm_eval_lab.reporting.formatters import comparison_markdown
from llm_eval_lab.reporting.regression import (
    Advisory,
    RunComparisonInput,
    build_thresholds,
    compare_runs,
    concordance,
    evaluate_check,
    expand_check,
    load_thresholds,
    pair_cases,
    resolve_metric,
    validate_metric_path,
)
from llm_eval_lab.reporting.statistics import wilson_interval

# The canonical paired concordance table.
FIXTURE_A = 40
FIXTURE_B = 5
FIXTURE_C = 2
FIXTURE_D = 3
FIXTURE_N = FIXTURE_A + FIXTURE_B + FIXTURE_C + FIXTURE_D
FIXTURE_DELTA = -0.06
FIXTURE_P_VALUE = 0.453125
FIXTURE_CI = (-0.1749, 0.0487)
CI_TOLERANCE = 1e-4

AT = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)


def make_response(latency_ms: float) -> ModelResponse:
    """Build the minimal successful response a case result carries."""
    return ModelResponse(
        output_text="answer",
        provider="fake",
        model="fake-1",
        requested_model="fake-1",
        finish_reason=FinishReason.STOP,
        latency_ms=latency_ms,
        total_latency_ms=latency_ms,
        started_at=AT,
        completed_at=AT,
    )


def make_case(  # noqa: PLR0913 - one parameter per independent input
    run_id: str,
    case_id: str,
    *,
    passed: bool | None = True,
    score: float | None = 1.0,
    latency_ms: float = 100.0,
    category: str | None = None,
    status: CaseStatus = CaseStatus.OK,
    case_hash: str | None = None,
    evaluator_id: str = "exact_match",
    evaluator_type: str = "exact_match",
) -> CaseResult:
    """Build one persisted case result.

    ``passed=None`` with a score is the score-only pairing the contract names;
    ``passed=None`` with no score is a skipped evaluation, which is the only way
    to produce a run whose ``mean_score`` is genuinely absent.
    """
    if passed is None:
        status_value = EvaluationStatus.PASSED if score is not None else EvaluationStatus.SKIPPED
    else:
        status_value = EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED
    evaluations = (
        EvaluationResult(
            evaluator_id=evaluator_id,
            evaluator_type=evaluator_type,
            status=status_value,
            passed=passed,
            score=score,
        ),
    )
    return CaseResult(
        run_id=run_id,
        case_id=case_id,
        case_hash=case_hash or f"sha256:{case_id}",
        status=status,
        response=make_response(latency_ms) if status is CaseStatus.OK else None,
        evaluations=evaluations if status is CaseStatus.OK else (),
        passed=passed if status is CaseStatus.OK else None,
        score=score if status is CaseStatus.OK else None,
        cost=None,
        started_at=AT,
        completed_at=AT,
        category=category,
    )


def make_metrics(run_id: str, results: list[CaseResult]) -> AggregateMetrics:
    """Roll a set of case results up exactly as a finished run would."""
    return compute_metrics(
        run_id=run_id,
        n_cases=len(results),
        results=results,
        error_policy="exclude",
        price_table_id="test",
        price_table_version="1",
        computed_at=AT,
    )


def make_side(
    run_id: str,
    results: list[CaseResult],
    *,
    suite_hash: str = "sha256:suite",
    label: str | None = None,
) -> RunComparisonInput:
    """Assemble one side of a comparison from its case results."""
    return RunComparisonInput(
        run_id=run_id,
        label=label,
        suite_hash=suite_hash,
        metrics=make_metrics(run_id, results),
        results=results,
    )


def concordance_cases(
    run_id: str,
    *,
    baseline_side: bool,
    latency_ms: float = 100.0,
) -> list[CaseResult]:
    """Build the fifty cases of the a=40, b=5, c=2, d=3 fixture for one side.

    Cells in order: `a` both pass, `b` baseline only, `c` candidate only, `d`
    neither. Case ids are stable across the two sides so the pairing is exact.
    """
    cells: tuple[tuple[str, int, bool], ...] = (
        ("a", FIXTURE_A, True),
        ("b", FIXTURE_B, baseline_side),
        ("c", FIXTURE_C, not baseline_side),
        ("d", FIXTURE_D, False),
    )
    return [
        make_case(run_id, f"{prefix}{index}", passed=passed, latency_ms=latency_ms)
        for prefix, count, passed in cells
        for index in range(count)
    ]


PASS_RATE_ONLY = RegressionThresholds(
    checks=(
        MetricCheck(
            metric="pass_rate",
            label="overall pass rate",
            direction=ThresholdDirection.HIGHER_IS_BETTER,
            max_absolute_decrease=0.02,
        ),
    )
)


# ---------------------------------------------------------------------------
# The canonical paired fixture
# ---------------------------------------------------------------------------


def test_canonical_paired_fixture_fails_on_the_point_estimate() -> None:
    baseline = make_side("base", concordance_cases("base", baseline_side=True))
    candidate = make_side("cand", concordance_cases("cand", baseline_side=False))

    report = compare_runs(baseline, candidate, PASS_RATE_ONLY)

    assert report.mode == "paired"
    assert report.paired_case_count == FIXTURE_N
    check = report.checks[0]
    assert check.metric == "pass_rate"
    assert check.baseline == pytest.approx(0.90)
    assert check.candidate == pytest.approx(0.84)
    assert check.delta == pytest.approx(FIXTURE_DELTA)
    assert check.n_paired == FIXTURE_N
    assert check.p_value == pytest.approx(FIXTURE_P_VALUE)
    assert check.confidence_interval is not None
    assert check.confidence_interval[0] == pytest.approx(FIXTURE_CI[0], abs=CI_TOLERANCE)
    assert check.confidence_interval[1] == pytest.approx(FIXTURE_CI[1], abs=CI_TOLERANCE)
    assert check.status is CheckStatus.FAILED
    assert check.violated_bound == "max_absolute_decrease"
    assert report.verdict is Verdict.FAIL


def test_advisory_interval_straddling_the_threshold_does_not_rescue_the_verdict() -> None:
    baseline = make_side("base", concordance_cases("base", baseline_side=True))
    candidate = make_side("cand", concordance_cases("cand", baseline_side=False))

    report = compare_runs(baseline, candidate, PASS_RATE_ONLY)
    check = report.checks[0]

    # The interval covers the -0.02 bound on both sides, and the gate is still a
    # failure because the point estimate is what decides it.
    assert check.confidence_interval is not None
    assert check.confidence_interval[0] < -0.02 < check.confidence_interval[1]
    assert check.status is CheckStatus.FAILED
    assert check.note is not None
    assert "not firmly established" in check.note
    # McNemar says the discordant pattern is unremarkable, and it changes nothing.
    assert check.significant is False
    assert report.verdict is Verdict.FAIL


def test_concordance_table_matches_the_fixture_cells() -> None:
    baseline = concordance_cases("base", baseline_side=True)
    candidate = concordance_cases("cand", baseline_side=False)
    pairing = pair_cases(baseline, candidate)

    table = concordance(baseline, candidate, pairing.paired_ids)

    assert (table.a, table.b, table.c, table.d) == (
        FIXTURE_A,
        FIXTURE_B,
        FIXTURE_C,
        FIXTURE_D,
    )


# ---------------------------------------------------------------------------
# The uneventful comparison
# ---------------------------------------------------------------------------


def test_identical_runs_pass_every_check() -> None:
    results = [make_case("base", f"case{index}") for index in range(20)]
    same = [make_case("cand", f"case{index}") for index in range(20)]
    thresholds = RegressionThresholds(
        checks=(
            MetricCheck(
                metric="pass_rate",
                direction=ThresholdDirection.HIGHER_IS_BETTER,
                max_absolute_decrease=0.02,
            ),
            MetricCheck(
                metric="error_rate",
                direction=ThresholdDirection.LOWER_IS_BETTER,
                max_value=0.01,
            ),
        )
    )

    report = compare_runs(make_side("base", results), make_side("cand", same), thresholds)

    assert report.verdict is Verdict.PASS
    assert [check.status for check in report.checks] == [
        CheckStatus.PASSED,
        CheckStatus.PASSED,
    ]
    assert report.summary.newly_failing == ()
    assert report.summary.newly_passing == ()
    assert report.summary.n_flipped == 0


def test_identical_runs_pass_under_the_shipped_default_policy() -> None:
    results = [make_case("base", f"case{index}", category="core") for index in range(24)]
    same = [make_case("cand", f"case{index}", category="core") for index in range(24)]

    report = compare_runs(make_side("base", results), make_side("cand", same), load_thresholds())

    assert report.verdict is Verdict.PASS
    assert not [
        check
        for check in report.checks
        if check.status in (CheckStatus.FAILED, CheckStatus.WARNING)
    ]


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------


def test_differing_suite_hash_is_incomparable_when_the_policy_requires_one_suite() -> None:
    baseline = make_side("base", [make_case("base", "c0")], suite_hash="sha256:one")
    candidate = make_side("cand", [make_case("cand", "c0")], suite_hash="sha256:two")

    report = compare_runs(baseline, candidate, PASS_RATE_ONLY)

    assert report.verdict is Verdict.INCOMPARABLE
    assert report.comparable is False
    assert report.suite_hash_match is False
    assert report.incomparable_reason is not None
    assert "different benchmark suites" in report.incomparable_reason
    assert report.checks == ()


def test_differing_suite_hash_is_compared_when_the_policy_allows_it() -> None:
    thresholds = PASS_RATE_ONLY.model_copy(update={"require_same_suite": False})
    baseline = make_side("base", [make_case("base", "c0")], suite_hash="sha256:one")
    candidate = make_side("cand", [make_case("cand", "c0")], suite_hash="sha256:two")

    report = compare_runs(baseline, candidate, thresholds)

    assert report.verdict is Verdict.PASS
    assert report.comparable is True
    assert report.suite_hash_match is False


def test_a_case_whose_hash_changed_is_reported_and_never_compared() -> None:
    baseline = [make_case("base", "c0"), make_case("base", "c1")]
    candidate = [
        make_case("cand", "c0"),
        make_case("cand", "c1", passed=False, case_hash="sha256:edited"),
    ]

    report = compare_runs(make_side("base", baseline), make_side("cand", candidate), PASS_RATE_ONLY)

    assert report.changed_cases == ("c1",)
    assert report.paired_case_count == 1
    # `c1` regressed, and it is excluded because it is a different case now.
    assert report.summary.newly_failing == ()


def test_metrics_belonging_to_another_run_are_refused() -> None:
    results = [make_case("base", "c0")]
    side = RunComparisonInput(
        run_id="base",
        label=None,
        suite_hash="sha256:suite",
        metrics=make_metrics("someone-else", results),
        results=results,
    )

    with pytest.raises(RegressionInputError, match="were supplied for run"):
        compare_runs(side, side, PASS_RATE_ONLY)


# ---------------------------------------------------------------------------
# Scope gates
# ---------------------------------------------------------------------------


def test_a_category_below_its_sample_gate_reports_insufficient_data() -> None:
    baseline = [make_case("base", f"c{index}", category="small") for index in range(6)]
    candidate = [make_case("cand", f"c{index}", category="small") for index in range(6)]
    thresholds = RegressionThresholds(
        checks=(
            MetricCheck(
                metric="category.*.pass_rate",
                label="category pass rate",
                direction=ThresholdDirection.HIGHER_IS_BETTER,
                max_absolute_decrease=0.02,
                min_samples=10,
            ),
        )
    )

    report = compare_runs(make_side("base", baseline), make_side("cand", candidate), thresholds)

    check = report.checks[0]
    assert check.metric == "category.small.pass_rate"
    assert check.status is CheckStatus.INSUFFICIENT_DATA
    assert check.note is not None
    assert "at least 10 samples" in check.note
    # An undersized bucket is not a pass and is not a failure.
    assert report.verdict is Verdict.PASS


def test_p95_latency_below_its_sample_gate_reports_insufficient_data() -> None:
    baseline = [make_case("base", f"c{index}", latency_ms=100.0) for index in range(30)]
    candidate = [make_case("cand", f"c{index}", latency_ms=900.0) for index in range(12)]
    thresholds = RegressionThresholds(
        checks=(
            MetricCheck(
                metric="latency.p95_ms",
                label="p95 model latency",
                direction=ThresholdDirection.LOWER_IS_BETTER,
                max_relative_increase=0.15,
                min_samples=20,
            ),
        )
    )

    report = compare_runs(make_side("base", baseline), make_side("cand", candidate), thresholds)

    check = report.checks[0]
    assert check.status is CheckStatus.INSUFFICIENT_DATA
    assert check.n_candidate == 12
    assert check.note is not None
    assert "model latency" in check.label
    assert "exclude retry backoff" in check.note
    assert report.verdict is Verdict.PASS


def test_an_unavailable_sample_count_is_treated_as_insufficient() -> None:
    check = MetricCheck(
        metric="pass_rate",
        direction=ThresholdDirection.HIGHER_IS_BETTER,
        max_absolute_decrease=0.02,
        min_samples=1,
    )

    outcome = evaluate_check(check, 0.9, 0.5, None, None)

    assert outcome.status is CheckStatus.INSUFFICIENT_DATA
    assert outcome.n_baseline == 0
    assert outcome.n_candidate == 0


def test_a_relative_bound_against_a_zero_baseline_is_not_silently_dropped() -> None:
    check = MetricCheck(
        metric="latency.p95_ms",
        direction=ThresholdDirection.LOWER_IS_BETTER,
        max_relative_increase=0.15,
    )

    outcome = evaluate_check(check, 0.0, 4000.0, 50, 50)

    assert outcome.status is CheckStatus.INSUFFICIENT_DATA
    assert outcome.note is not None
    assert "baseline of zero" in outcome.note


MEAN_SCORE_ONLY = RegressionThresholds(
    checks=(
        MetricCheck(
            metric="mean_score",
            direction=ThresholdDirection.HIGHER_IS_BETTER,
            max_absolute_decrease=0.05,
        ),
    )
)


def test_a_metric_neither_run_reports_is_insufficient_data_not_a_regression() -> None:
    baseline = [make_case("base", "c0", score=None, passed=None)]
    candidate = [make_case("cand", "c0", score=None, passed=None)]

    report = compare_runs(
        make_side("base", baseline), make_side("cand", candidate), MEAN_SCORE_ONLY
    )

    check = report.checks[0]
    assert check.status is CheckStatus.INSUFFICIENT_DATA
    assert check.note is not None
    assert "Neither run reports this metric" in check.note
    # The candidate is identical to the baseline, so calling this a regression
    # would fail a build for a change nobody made.
    assert report.verdict is Verdict.PASS


def test_a_metric_only_one_run_reports_is_a_missing_metric() -> None:
    baseline = [make_case("base", "c0", score=0.9)]
    candidate = [make_case("cand", "c0", score=None, passed=None)]

    report = compare_runs(
        make_side("base", baseline), make_side("cand", candidate), MEAN_SCORE_ONLY
    )

    check = report.checks[0]
    assert check.status is CheckStatus.MISSING_METRIC
    assert check.note is not None
    assert "One run reports this metric and the other does not" in check.note
    assert report.verdict is Verdict.FAIL

    lenient = MEAN_SCORE_ONLY.model_copy(update={"fail_on_missing_metric": False})
    assert (
        compare_runs(make_side("base", baseline), make_side("cand", candidate), lenient).verdict
        is Verdict.PASS
    )


def test_two_identical_judge_only_runs_pass_under_the_shipped_policy() -> None:
    def judged(run_id: str) -> list[CaseResult]:
        return [
            make_case(
                run_id,
                f"c{index}",
                passed=None,
                score=0.8,
                category="judged",
                evaluator_id="judge",
                evaluator_type="judge",
            )
            for index in range(12)
        ]

    baseline = make_side("base", judged("base"))
    candidate = make_side("cand", judged("cand"))
    # A score-only suite has no pass rate in EITHER run, by design.
    assert baseline.metrics.pass_rate is None
    assert baseline.metrics.n_score_only == 12

    report = compare_runs(baseline, candidate, load_thresholds())

    assert report.verdict is Verdict.PASS
    statuses = {check.metric: check.status for check in report.checks}
    assert statuses["pass_rate"] is CheckStatus.INSUFFICIENT_DATA
    assert statuses["category.judged.pass_rate"] is CheckStatus.INSUFFICIENT_DATA
    assert statuses["evaluator.judge.mean_score"] is CheckStatus.PASSED


def test_an_evaluator_type_no_run_uses_is_reported_as_not_applicable() -> None:
    baseline = [make_case("base", "c0")]
    candidate = [make_case("cand", "c0")]
    thresholds = RegressionThresholds(
        checks=(
            MetricCheck(
                metric="evaluator_type.json_valid.pass_rate",
                label="JSON validity",
                direction=ThresholdDirection.HIGHER_IS_BETTER,
                max_absolute_decrease=0.0,
            ),
        )
    )

    report = compare_runs(make_side("base", baseline), make_side("cand", candidate), thresholds)

    check = report.checks[0]
    assert check.status is CheckStatus.INSUFFICIENT_DATA
    assert check.note is not None
    assert "Not applicable" in check.note
    assert report.verdict is Verdict.PASS


def test_an_instance_present_in_only_one_run_is_not_compared() -> None:
    baseline = [make_case("base", f"c{index}", category="old") for index in range(12)]
    candidate = [make_case("cand", f"c{index}", category="new") for index in range(12)]
    thresholds = RegressionThresholds(
        checks=(
            MetricCheck(
                metric="category.*.pass_rate",
                direction=ThresholdDirection.HIGHER_IS_BETTER,
                max_absolute_decrease=0.02,
                min_samples=10,
            ),
        )
    )

    report = compare_runs(make_side("base", baseline), make_side("cand", candidate), thresholds)

    statuses = {check.metric: check.status for check in report.checks}
    assert statuses == {
        "category.new.pass_rate": CheckStatus.INSUFFICIENT_DATA,
        "category.old.pass_rate": CheckStatus.INSUFFICIENT_DATA,
    }
    assert report.verdict is Verdict.PASS


# ---------------------------------------------------------------------------
# Bound semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (0.01, CheckStatus.FAILED),
        (0.00999, CheckStatus.PASSED),
        (0.0101, CheckStatus.FAILED),
    ],
)
def test_max_value_is_an_exclusive_ceiling(candidate: float, expected: CheckStatus) -> None:
    check = MetricCheck(
        metric="error_rate",
        direction=ThresholdDirection.LOWER_IS_BETTER,
        max_value=0.01,
    )

    outcome = evaluate_check(check, 0.0, candidate, 100, 100)

    assert outcome.status is expected
    assert outcome.threshold_value == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (0.90, CheckStatus.PASSED),
        (0.8999, CheckStatus.FAILED),
        (0.91, CheckStatus.PASSED),
    ],
)
def test_min_value_is_an_inclusive_floor(candidate: float, expected: CheckStatus) -> None:
    check = MetricCheck(
        metric="pass_rate",
        direction=ThresholdDirection.HIGHER_IS_BETTER,
        min_value=0.90,
    )

    outcome = evaluate_check(check, 0.95, candidate, 100, 100)

    assert outcome.status is expected


def test_a_decrease_of_exactly_the_tolerance_passes() -> None:
    check = MetricCheck(
        metric="pass_rate",
        direction=ThresholdDirection.HIGHER_IS_BETTER,
        max_absolute_decrease=0.02,
    )

    # 0.90 - 0.88 is -0.020000000000000018 in binary floating point.
    assert evaluate_check(check, 0.90, 0.88, 100, 100).status is CheckStatus.PASSED
    assert evaluate_check(check, 0.90, 0.8799, 100, 100).status is CheckStatus.FAILED


def test_a_relative_increase_bound_reports_the_ceiling_it_compared_against() -> None:
    check = MetricCheck(
        metric="latency.p95_ms",
        direction=ThresholdDirection.LOWER_IS_BETTER,
        max_relative_increase=0.15,
    )

    outcome = evaluate_check(check, 1000.0, 1200.0, 50, 50)

    assert outcome.status is CheckStatus.FAILED
    assert outcome.threshold_value == pytest.approx(1150.0)
    assert outcome.relative_delta == pytest.approx(0.2)


def test_a_warning_severity_breach_warns_rather_than_failing() -> None:
    thresholds = RegressionThresholds(
        checks=(
            MetricCheck(
                metric="pass_rate",
                direction=ThresholdDirection.HIGHER_IS_BETTER,
                max_absolute_decrease=0.02,
                severity="warning",
            ),
        )
    )
    baseline = [make_case("base", f"c{index}") for index in range(10)]
    candidate = [make_case("cand", f"c{index}", passed=index < 5) for index in range(10)]

    report = compare_runs(make_side("base", baseline), make_side("cand", candidate), thresholds)

    assert report.checks[0].status is CheckStatus.WARNING
    assert report.verdict is Verdict.WARN


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_require_significant_is_rejected_as_reserved() -> None:
    with pytest.raises(EvaluatorConfigError, match="reserved for a future release"):
        MetricCheck(
            metric="pass_rate",
            direction=ThresholdDirection.HIGHER_IS_BETTER,
            max_absolute_decrease=0.02,
            require_significant=True,
        )


def test_a_check_with_no_bound_is_a_configuration_error() -> None:
    with pytest.raises(EvaluatorConfigError, match="configures no bound"):
        build_thresholds(
            {
                "checks": [
                    {"metric": "pass_rate", "direction": "higher_is_better"},
                ]
            }
        )


def test_an_unaddressable_metric_is_a_configuration_error() -> None:
    with pytest.raises(EvaluatorConfigError, match="unknown threshold metric"):
        validate_metric_path("pass_rat")
    with pytest.raises(EvaluatorConfigError, match="unknown threshold metric"):
        build_thresholds(
            {
                "checks": [
                    {
                        "metric": "latency.p42_ms",
                        "direction": "lower_is_better",
                        "max_relative_increase": 0.15,
                    }
                ]
            }
        )


def test_the_shipped_policy_carries_the_documented_defaults() -> None:
    thresholds = load_thresholds()
    by_metric = {check.metric: check for check in thresholds.checks}

    assert by_metric["pass_rate"].max_absolute_decrease == pytest.approx(0.02)
    assert by_metric["category.*.pass_rate"].max_absolute_decrease == pytest.approx(0.02)
    assert by_metric["category.*.pass_rate"].min_samples == 10
    assert by_metric["evaluator_type.json_valid.pass_rate"].max_absolute_decrease == 0.0
    assert by_metric["evaluator_type.json_schema.pass_rate"].max_absolute_decrease == 0.0
    assert by_metric["latency.p95_ms"].max_relative_increase == pytest.approx(0.15)
    assert by_metric["latency.p95_ms"].min_samples == 20
    assert by_metric["latency.p95_ms"].label == "p95 model latency"
    assert by_metric["error_rate"].max_value == pytest.approx(0.01)
    assert by_metric["evaluator.*.mean_score"].max_absolute_decrease == pytest.approx(0.05)


def test_the_shipped_policy_and_the_ci_example_agree_on_every_bound(repo_root: object) -> None:
    from pathlib import Path  # noqa: PLC0415 - local to this one comparison

    example = Path(str(repo_root)) / "examples" / "thresholds" / "ci-thresholds.yaml"
    shipped = load_thresholds()
    published = load_thresholds(example)

    assert [check.model_dump() for check in published.checks] == [
        check.model_dump() for check in shipped.checks
    ]


def test_an_unreadable_policy_file_is_a_configuration_error() -> None:
    with pytest.raises(EvaluatorConfigError, match="cannot read threshold policy"):
        load_thresholds("/nonexistent/thresholds.yaml")


# ---------------------------------------------------------------------------
# Paired and unpaired modes
# ---------------------------------------------------------------------------


def test_runs_with_no_shared_cases_compare_in_unpaired_mode() -> None:
    baseline = [make_case("base", f"b{index}", passed=index < 18) for index in range(20)]
    candidate = [make_case("cand", f"c{index}", passed=index < 16) for index in range(20)]

    report = compare_runs(make_side("base", baseline), make_side("cand", candidate), PASS_RATE_ONLY)

    assert report.mode == "unpaired"
    assert report.paired_case_count == 0
    check = report.checks[0]
    assert check.p_value is None
    assert check.confidence_interval is None
    assert check.n_paired is None
    assert check.baseline_ci is not None
    assert check.candidate_ci is not None
    # The per-check flag says which pair overlaps, and it agrees with the two
    # intervals it was derived from.
    expected = check.baseline_ci[0] <= check.candidate_ci[1] and (
        check.candidate_ci[0] <= check.baseline_ci[1]
    )
    assert check.intervals_overlap is expected
    assert report.intervals_overlap is check.intervals_overlap
    # The gate still resolves, from the point estimates alone.
    assert check.baseline == pytest.approx(0.90)
    assert check.candidate == pytest.approx(0.80)
    assert check.status is CheckStatus.FAILED
    assert report.verdict is Verdict.FAIL


def test_paired_mode_gives_the_overall_rate_a_delta_interval_and_the_rest_marginals() -> None:
    baseline = make_side("base", concordance_cases("base", baseline_side=True))
    candidate = make_side("cand", concordance_cases("cand", baseline_side=False))

    report = compare_runs(baseline, candidate, load_thresholds())

    assert report.mode == "paired"

    # The overall pass rate is the ONE substituted check: Method 10 on the delta
    # replaces the two marginals, which on a paired difference ignore the
    # correlation and invite the "they overlap so nothing changed" fallacy.
    overall = next(check for check in report.checks if check.metric == "pass_rate")
    assert overall.confidence_interval is not None
    assert overall.p_value is not None
    assert overall.n_paired == FIXTURE_N
    assert overall.baseline_ci is None
    assert overall.candidate_ci is None
    assert overall.intervals_overlap is None
    assert report.intervals_overlap is None

    # Every other check carrying a rate carries each arm's own interval, so no
    # point estimate is rendered bare beneath a status badge. It reports no
    # `n_paired`, because no paired statistic was computed for it.
    error_rate = next(check for check in report.checks if check.metric == "error_rate")
    assert error_rate.baseline_ci is not None
    assert error_rate.candidate_ci is not None
    assert error_rate.intervals_overlap is not None
    assert error_rate.confidence_interval is None
    assert error_rate.p_value is None
    assert error_rate.n_paired is None


def test_paired_false_forces_unpaired_mode_even_when_case_ids_intersect() -> None:
    thresholds = PASS_RATE_ONLY.model_copy(update={"paired": False})
    baseline = make_side("base", concordance_cases("base", baseline_side=True))
    candidate = make_side("cand", concordance_cases("cand", baseline_side=False))

    report = compare_runs(baseline, candidate, thresholds)

    assert report.mode == "unpaired"
    # The intersection is still reported; it just is not used for statistics.
    assert report.paired_case_count == FIXTURE_N
    check = report.checks[0]
    assert check.p_value is None
    assert check.confidence_interval is None
    assert check.n_paired is None
    assert check.baseline_ci is not None
    assert check.status is CheckStatus.FAILED


def test_the_summary_names_the_cases_that_flipped() -> None:
    baseline = [make_case("base", f"c{index}", passed=index != 3) for index in range(6)]
    candidate = [
        make_case("cand", f"c{index}", passed=index not in (0, 1), score=0.0 if index < 2 else 1.0)
        for index in range(6)
    ]

    report = compare_runs(make_side("base", baseline), make_side("cand", candidate), PASS_RATE_ONLY)

    assert report.summary.newly_failing == ("c0", "c1")
    assert report.summary.newly_passing == ("c3",)
    assert report.summary.n_flipped == 3
    assert [drop.case_id for drop in report.summary.largest_score_drops] == ["c0", "c1"]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_the_report_round_trips_through_json_with_its_schema_version_and_mode() -> None:
    baseline = make_side("base", concordance_cases("base", baseline_side=True), label="v1")
    candidate = make_side("cand", concordance_cases("cand", baseline_side=False), label="v2")
    report = compare_runs(baseline, candidate, load_thresholds())

    text = report.model_dump_json()
    document = json.loads(text)

    assert document["schema_version"] == 1
    assert document["mode"] == "paired"
    assert document["verdict"] == "fail"
    restored = RegressionReport.model_validate_json(text)
    assert restored == report


def test_resolve_metric_reads_the_rollups_own_interval() -> None:
    results = [make_case("base", f"c{index}", passed=index < 18) for index in range(20)]
    metrics = make_metrics("base", results)

    value = resolve_metric(metrics, "pass_rate")

    assert value.value == pytest.approx(0.90)
    assert value.n == 20
    assert value.interval == metrics.pass_rate_ci


# ---------------------------------------------------------------------------
# Uncertainty is reported beside every point estimate
# ---------------------------------------------------------------------------


CATEGORY_ONLY = RegressionThresholds(
    checks=(
        MetricCheck(
            metric="category.*.pass_rate",
            label="category pass rate",
            direction=ThresholdDirection.HIGHER_IS_BETTER,
            max_absolute_decrease=0.02,
            min_samples=10,
        ),
    )
)

CATEGORY_CASES = 30
CATEGORY_PASSING_IN_CANDIDATE = 24


def _category_comparison() -> tuple[RunComparisonInput, RunComparisonInput]:
    """Thirty paired cases in one category, with a twenty-point drop in it."""
    baseline = [make_case("base", f"c{i}", category="core") for i in range(CATEGORY_CASES)]
    candidate = [
        make_case("cand", f"c{i}", category="core", passed=i < CATEGORY_PASSING_IN_CANDIDATE)
        for i in range(CATEGORY_CASES)
    ]
    return make_side("base", baseline), make_side("cand", candidate)


def test_a_paired_category_check_reports_both_arms_intervals() -> None:
    baseline, candidate = _category_comparison()

    report = compare_runs(baseline, candidate, CATEGORY_ONLY)

    assert report.mode == "paired"
    check = report.checks[0]
    assert check.metric == "category.core.pass_rate"
    assert check.status is CheckStatus.FAILED
    # At thirty cases the minimum detectable difference is around thirty points,
    # so a twenty-point drop must not be rendered as a bare number under a red
    # badge with no uncertainty anywhere in the document.
    assert check.baseline_ci is not None
    assert check.candidate_ci is not None
    assert check.intervals_overlap is not None
    assert check.baseline_ci == baseline.metrics.by_category["core"].pass_rate_ci
    assert check.candidate_ci == candidate.metrics.by_category["core"].pass_rate_ci
    # No paired statistic was computed for this check, so it claims no paired
    # sample size. Reporting the run-level intersection here would put the run's
    # case count beside a rate measured over one category.
    assert check.n_paired is None
    assert check.confidence_interval is None
    assert check.p_value is None


def test_a_paired_category_checks_intervals_reach_both_renderings() -> None:
    baseline, candidate = _category_comparison()
    report = compare_runs(baseline, candidate, CATEGORY_ONLY)
    check = report.checks[0]
    assert check.baseline_ci is not None
    assert check.candidate_ci is not None

    document = json.loads(report.model_dump_json())
    rendered = document["checks"][0]
    assert rendered["baseline_ci"] == list(check.baseline_ci)
    assert rendered["candidate_ci"] == list(check.candidate_ci)
    assert rendered["intervals_overlap"] == check.intervals_overlap

    markdown = comparison_markdown(report)
    advisory = [
        line
        for line in markdown.splitlines()
        if line.startswith("| category pass rate (core) | higher is better |")
    ]
    assert len(advisory) == 1
    assert f"[{check.baseline_ci[0]:.4f}, {check.baseline_ci[1]:.4f}]" in advisory[0]
    assert f"[{check.candidate_ci[0]:.4f}, {check.candidate_ci[1]:.4f}]" in advisory[0]

    # The two intervals overlap at thirty cases, and the report says so in words
    # rather than leaving a red badge to imply a settled twenty-point regression.
    assert check.intervals_overlap is True
    assert check.note is not None
    assert "intervals overlap" in check.note


EVALUATOR_TOTAL = 12
EVALUATOR_ELIGIBLE = 8


def _mixed_evaluator_cases(run_id: str) -> list[CaseResult]:
    """Twelve evaluations of one evaluator, four of them score-only.

    A score-only evaluation counts toward `n` and is excluded from the pass-rate
    denominator, so the two populations genuinely differ: `n` is twelve and
    `n_pass_denominator` is eight.
    """
    return [
        make_case(
            run_id,
            f"c{index}",
            passed=True if index < EVALUATOR_ELIGIBLE else None,
            score=0.9,
            evaluator_id="strict_json",
            evaluator_type="json_valid",
        )
        for index in range(EVALUATOR_TOTAL)
    ]


def test_an_evaluator_pass_rate_is_gated_on_its_denominator_not_its_case_count() -> None:
    baseline = make_side("base", _mixed_evaluator_cases("base"))
    candidate = make_side("cand", _mixed_evaluator_cases("cand"))
    bucket = baseline.metrics.by_evaluator["strict_json"]
    assert bucket.n == EVALUATOR_TOTAL
    assert bucket.n_pass_denominator == EVALUATOR_ELIGIBLE

    thresholds = RegressionThresholds(
        checks=(
            MetricCheck(
                metric="evaluator_type.json_valid.pass_rate",
                label="JSON validity",
                direction=ThresholdDirection.HIGHER_IS_BETTER,
                max_absolute_decrease=0.0,
                min_samples=10,
            ),
        )
    )
    report = compare_runs(baseline, candidate, thresholds)

    check = report.checks[0]
    # Gated on eight eligible evaluations, not on twelve. Reading `n` here would
    # have passed a gate the rate it guards does not actually meet.
    assert check.n_baseline == EVALUATOR_ELIGIBLE
    assert check.n_candidate == EVALUATOR_ELIGIBLE
    assert check.status is CheckStatus.INSUFFICIENT_DATA

    relaxed = thresholds.model_copy(
        update={
            "checks": (thresholds.checks[0].model_copy(update={"min_samples": 8}),),
        }
    )
    resolved = compare_runs(baseline, candidate, relaxed).checks[0]
    assert resolved.status is CheckStatus.PASSED
    # And the interval is over the same eight, so the figure and its uncertainty
    # share a denominator.
    assert resolved.baseline_ci is not None
    assert resolved.candidate_ci is not None
    assert resolved.baseline_ci == wilson_interval(EVALUATOR_ELIGIBLE, EVALUATOR_ELIGIBLE)[1:]


def test_an_evaluator_rollup_with_an_unknown_denominator_gates_out() -> None:
    baseline = make_side("base", [make_case("base", "c0")])
    stale = baseline.metrics.by_evaluator["exact_match"].model_copy(
        update={"n_pass_denominator": 0}
    )
    aged = baseline.metrics.model_copy(update={"by_evaluator": {"exact_match": stale}})
    thresholds = RegressionThresholds(
        checks=(
            MetricCheck(
                metric="evaluator.*.pass_rate",
                direction=ThresholdDirection.HIGHER_IS_BETTER,
                max_absolute_decrease=0.0,
            ),
        )
    )

    report = compare_runs(
        RunComparisonInput(
            run_id="base",
            label=None,
            suite_hash="sha256:suite",
            metrics=aged,
            results=baseline.results,
        ),
        baseline,
        thresholds,
    )

    # Zero means UNKNOWN, so the gate reports it rather than guessing.
    assert report.checks[0].status is CheckStatus.INSUFFICIENT_DATA
    assert report.verdict is Verdict.PASS


# ---------------------------------------------------------------------------
# Emitted metric paths are addressable
# ---------------------------------------------------------------------------


def test_an_evaluator_type_check_emits_a_path_that_addresses_the_instance() -> None:
    def cases(run_id: str) -> list[CaseResult]:
        return [make_case(run_id, "c0", evaluator_id="strict_json", evaluator_type="json_valid")]

    baseline = make_side("base", cases("base"))
    candidate = make_side("cand", cases("cand"))
    thresholds = RegressionThresholds(
        checks=(
            MetricCheck(
                metric="evaluator_type.json_valid.pass_rate",
                label="JSON validity",
                direction=ThresholdDirection.HIGHER_IS_BETTER,
                max_absolute_decrease=0.0,
            ),
        )
    )

    report = compare_runs(baseline, candidate, thresholds)
    check = report.checks[0]

    # The key is an evaluator INSTANCE id, so the path is emitted under the
    # namespace that actually addresses one.
    assert check.metric == "evaluator.strict_json.pass_rate"
    assert check.label == "JSON validity (strict_json)"

    # And copying that string back into a policy yields the same check, rather
    # than one that looks enforced and resolves to nothing.
    copied = MetricCheck(
        metric=check.metric,
        direction=ThresholdDirection.HIGHER_IS_BETTER,
        max_absolute_decrease=0.0,
    )
    resolved = expand_check(copied, baseline.metrics, candidate.metrics)
    assert len(resolved) == 1
    assert resolved[0].metric == check.metric
    assert resolved[0].unavailable_reason is None
    assert resolved[0].candidate.value == check.candidate


def test_every_emitted_metric_path_round_trips_through_the_policy_loader() -> None:
    baseline = make_side(
        "base",
        [make_case("base", f"c{index}", category="core") for index in range(12)],
    )
    candidate = make_side(
        "cand",
        [make_case("cand", f"c{index}", category="core") for index in range(12)],
    )
    report = compare_runs(baseline, candidate, load_thresholds())

    for check in report.checks:
        if check.status is CheckStatus.INSUFFICIENT_DATA and check.candidate is None:
            # A policy line matching no instance keeps its own selector, which is
            # the policy string rather than a resolved one.
            continue
        loaded = build_thresholds(
            {
                "checks": [
                    {
                        "metric": check.metric,
                        "direction": check.direction.value,
                        "max_absolute_decrease": 0.02,
                    }
                ]
            }
        )
        resolved = expand_check(loaded.checks[0], baseline.metrics, candidate.metrics)
        assert len(resolved) == 1, check.metric
        assert resolved[0].metric == check.metric
        assert resolved[0].unavailable_reason is None, check.metric


# ---------------------------------------------------------------------------
# Advisory notes and the incomparable report
# ---------------------------------------------------------------------------


def test_the_straddle_note_sees_a_movement_bound_behind_an_absolute_one() -> None:
    # Absolute bounds resolve first, so the reported bound is `max_value`, whose
    # statement is not on the delta. The straddle test still has to see the
    # movement bound behind it.
    check = MetricCheck(
        metric="error_rate",
        direction=ThresholdDirection.LOWER_IS_BETTER,
        max_value=0.5,
        max_absolute_increase=0.02,
    )

    outcome = evaluate_check(
        check,
        0.10,
        0.11,
        100,
        100,
        advisory=Advisory(confidence_interval=(-0.05, 0.05), n_paired=100),
    )

    assert outcome.status is CheckStatus.PASSED
    assert outcome.threshold_value == pytest.approx(0.5)
    assert outcome.note is not None
    assert "not comfortable" in outcome.note


def test_an_incomparable_report_states_the_mode_it_actually_has() -> None:
    baseline = make_side(
        "base", concordance_cases("base", baseline_side=True), suite_hash="sha256:one"
    )
    candidate = make_side(
        "cand", concordance_cases("cand", baseline_side=False), suite_hash="sha256:two"
    )

    report = compare_runs(baseline, candidate, PASS_RATE_ONLY)

    assert report.verdict is Verdict.INCOMPARABLE
    # The two runs share unchanged case ids, so reporting `unpaired` beside a
    # positive paired case count would read as two fields disagreeing.
    assert report.paired_case_count == FIXTURE_N
    assert report.mode == "paired"
