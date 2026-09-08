"""The rollup: denominators, populations, coverage and the low-confidence gates.

Almost every assertion here is about which cases went into which denominator.
That is where a metrics layer lies most easily: a plausible number over the
wrong population is much harder to spot than a missing one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from llm_eval_lab.models import (
    AggregateMetrics,
    CaseResult,
    CaseStatus,
    CostBreakdown,
    EvaluationResult,
    EvaluationStatus,
    FinishReason,
    ModelResponse,
    ProviderErrorInfo,
    ProviderErrorKind,
    TokenUsage,
)
from llm_eval_lab.reporting.aggregate import (
    by_category,
    by_evaluator,
    by_tag,
    compute_metrics,
    error_breakdown,
    latency_stats,
    token_totals,
)
from llm_eval_lab.reporting.statistics import wilson_interval

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
RUN_ID = "7b1f9c0e-0000-4000-8000-000000000001"
PRICE_TABLE_ID = "builtin"
PRICE_TABLE_VERSION = "2026-09-06"


def response(
    *,
    latency_ms: float = 10.0,
    input_tokens: int | None = 100,
    output_tokens: int | None = 20,
) -> ModelResponse:
    """Build a successful response with the given latency and usage."""
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    return ModelResponse(
        output_text="4",
        provider="fake",
        model="fake-1",
        requested_model="fake-1",
        finish_reason=FinishReason.STOP,
        usage=usage,
        latency_ms=latency_ms,
        total_latency_ms=latency_ms * 2,
        started_at=AT,
        completed_at=AT,
    )


def evaluation(
    *,
    evaluator_id: str = "exact_match",
    status: EvaluationStatus | None = None,
    passed: bool | None = True,
    score: float | None = 1.0,
) -> EvaluationResult:
    """Build one evaluation result, deriving `status` from `passed` when unset.

    The contract refuses a status that contradicts `passed`, so the default
    follows it rather than fighting it: `passed=False` is FAILED, and True or
    None is PASSED, the latter being the score-only pairing.
    """
    if status is None:
        status = EvaluationStatus.FAILED if passed is False else EvaluationStatus.PASSED
    error = None
    if status is EvaluationStatus.ERROR:
        error = {"kind": "runtime", "message": "evaluator blew up"}
    return EvaluationResult.model_validate(
        {
            "evaluator_id": evaluator_id,
            "evaluator_type": evaluator_id,
            "status": status,
            "passed": passed,
            "score": score,
            "error": error,
        }
    )


def priced(total: str = "0.001") -> CostBreakdown:
    """Build a priced cost breakdown."""
    return CostBreakdown(
        input_cost=Decimal(total) / 2,
        output_cost=Decimal(total) / 2,
        total_cost=Decimal(total),
        priced=True,
        price_table_id=PRICE_TABLE_ID,
        price_table_version=PRICE_TABLE_VERSION,
    )


def unpriced(reason: str = "no_price_entry") -> CostBreakdown:
    """Build an unpriced cost breakdown."""
    return CostBreakdown(
        input_cost=None,
        output_cost=None,
        total_cost=None,
        priced=False,
        unpriced_reason=reason,
        price_table_id=PRICE_TABLE_ID,
        price_table_version=PRICE_TABLE_VERSION,
    )


def case(  # noqa: PLR0913 - a fixture builder mirrors the model's fields
    case_id: str,
    *,
    status: CaseStatus = CaseStatus.OK,
    passed: bool | None = True,
    score: float | None = 1.0,
    latency_ms: float = 10.0,
    usage: bool = True,
    cost: CostBreakdown | None = None,
    category: str | None = None,
    tags: tuple[str, ...] = (),
    evaluations: tuple[EvaluationResult, ...] | None = None,
    error_kind: ProviderErrorKind | None = None,
) -> CaseResult:
    """Build one case result."""
    body = (
        None
        if error_kind is not None
        else response(
            latency_ms=latency_ms,
            input_tokens=100 if usage else None,
            output_tokens=20 if usage else None,
        )
    )
    error = (
        None
        if error_kind is None
        else ProviderErrorInfo(kind=error_kind, message="failed", provider="fake")
    )
    if evaluations is None:
        if status is CaseStatus.OK:
            evaluations = (evaluation(passed=passed, score=score),)
        elif error_kind is not None:
            # The provider call failed, so no evaluator ever ran.
            evaluations = ()
        else:
            evaluations = (evaluation(status=EvaluationStatus.ERROR, passed=None, score=None),)
    return CaseResult(
        run_id=RUN_ID,
        case_id=case_id,
        case_hash="sha256:" + "a" * 64,
        status=status,
        response=body,
        evaluations=evaluations,
        passed=passed,
        score=score,
        cost=cost if cost is not None else priced(),
        started_at=AT,
        completed_at=AT,
        error=error,
        category=category,
        tags=tags,
    )


def metrics_over(
    results: list[CaseResult],
    *,
    error_policy: str = "exclude",
    n_cases: int = 0,
) -> AggregateMetrics:
    """Roll a list of case results up with the fixture's price table identity."""
    return compute_metrics(
        run_id=RUN_ID,
        n_cases=n_cases or len(results),
        results=results,
        error_policy=error_policy,
        price_table_id=PRICE_TABLE_ID,
        price_table_version=PRICE_TABLE_VERSION,
        computed_at=AT,
    )


# --- pass rate, error rate and the two policies ---------------------------


def fifty_case_run() -> list[CaseResult]:
    """The brief's worked example: 50 cases, 5 errored, 36 of the rest passing."""
    results = [
        case(f"err-{index}", status=CaseStatus.ERROR, passed=None, score=None) for index in range(5)
    ]
    results += [case(f"pass-{index}", passed=True, score=1.0) for index in range(36)]
    results += [case(f"fail-{index}", passed=False, score=0.0) for index in range(9)]
    return results


def test_pass_rate_under_exclude_drops_errored_cases_from_the_denominator() -> None:
    metrics = metrics_over(fifty_case_run(), error_policy="exclude")
    assert metrics.n_passed == 36
    assert metrics.n_pass_denominator == 45
    assert metrics.pass_rate == pytest.approx(36 / 45)


def test_pass_rate_under_fail_counts_errored_cases_as_failures() -> None:
    metrics = metrics_over(fifty_case_run(), error_policy="fail")
    assert metrics.n_passed == 36
    assert metrics.n_pass_denominator == 50
    assert metrics.pass_rate == pytest.approx(36 / 50)


@pytest.mark.parametrize("policy", ["exclude", "fail"])
def test_error_rate_is_over_total_attempted_under_either_policy(policy: str) -> None:
    metrics = metrics_over(fifty_case_run(), error_policy=policy)
    assert metrics.n_errors == 5
    assert metrics.error_rate == pytest.approx(5 / 50)


def test_pass_rate_interval_uses_the_pass_rate_denominator() -> None:
    metrics = metrics_over(fifty_case_run())
    _, lower, upper = wilson_interval(36, 45)
    assert metrics.pass_rate_ci == (lower, upper)


def test_a_score_only_case_leaves_both_sides_of_the_pass_rate() -> None:
    results = [
        case("scored", passed=None, score=0.7, evaluations=(evaluation(passed=None, score=0.7),)),
        case("passed", passed=True, score=1.0),
    ]
    metrics = metrics_over(results)
    assert metrics.n_pass_denominator == 1
    assert metrics.n_passed == 1
    assert metrics.n_score_only == 1


def test_an_empty_denominator_reports_a_null_pass_rate_not_zero() -> None:
    results = [
        case("scored", passed=None, score=0.5, evaluations=(evaluation(passed=None, score=0.5),))
    ]
    metrics = metrics_over(results)
    assert metrics.pass_rate is None, "nothing eligible is not the same fact as everything failing"
    assert metrics.pass_rate_ci is None


def test_a_run_with_no_results_reports_nulls_rather_than_zeros() -> None:
    metrics = metrics_over([], n_cases=10)
    assert metrics.pass_rate is None
    assert metrics.mean_score is None
    assert metrics.error_rate == 0.0
    assert metrics.token_usage_coverage is None
    assert metrics.cost_coverage is None
    assert metrics.cost_per_case is None


# --- error accounting ------------------------------------------------------


def test_error_breakdown_separates_a_provider_failure_from_an_evaluator_one() -> None:
    results = [
        case("rate", status=CaseStatus.ERROR, passed=None, error_kind=ProviderErrorKind.RATE_LIMIT),
        case(
            "evaluator",
            status=CaseStatus.ERROR,
            passed=None,
            score=None,
            evaluations=(evaluation(status=EvaluationStatus.ERROR, passed=None, score=None),),
        ),
        case("ok"),
    ]
    assert error_breakdown(results) == {"evaluation_error": 1, "rate_limit": 1}


def test_timeouts_are_counted_separately_and_still_count_as_errors() -> None:
    results = [
        case("slow", status=CaseStatus.TIMEOUT, passed=None, error_kind=ProviderErrorKind.TIMEOUT),
        case("ok"),
    ]
    metrics = metrics_over(results)
    assert metrics.n_timeouts == 1
    assert metrics.n_errors == 1
    assert metrics.n_pass_denominator == 1


# --- latency ---------------------------------------------------------------


def test_latency_uses_successful_cases_only() -> None:
    results = [
        case("ok-1", latency_ms=10.0),
        case("ok-2", latency_ms=20.0),
        case("failed", status=CaseStatus.ERROR, passed=None, error_kind=ProviderErrorKind.SERVER),
    ]
    stats = latency_stats(results)
    assert stats.n == 2
    assert stats.mean_ms == pytest.approx(15.0)
    assert stats.max_ms == pytest.approx(20.0)


def test_latency_percentiles_below_their_gate_are_present_and_marked() -> None:
    results = [case(f"c-{index}", latency_ms=float(index + 1)) for index in range(10)]
    stats = latency_stats(results)

    assert stats.low_confidence == ("p95_ms", "p99_ms")
    assert "p50_ms" not in stats.low_confidence
    assert "p90_ms" not in stats.low_confidence, "ten samples meets the p90 gate exactly"
    assert stats.p50_ms is not None
    assert stats.p95_ms is not None
    assert stats.p99_ms is not None


def test_latency_over_no_successful_cases_is_null_and_fully_marked() -> None:
    results = [
        case("failed", status=CaseStatus.ERROR, passed=None, error_kind=ProviderErrorKind.SERVER)
    ]
    stats = latency_stats(results)
    assert stats.n == 0
    assert stats.p50_ms is None
    assert stats.low_confidence == ("p50_ms", "p90_ms", "p95_ms", "p99_ms")


def test_latency_uses_the_successful_attempt_not_the_retry_wall_clock() -> None:
    # The fixture's `total_latency_ms` is twice `latency_ms`, so a series built
    # from the wrong field is off by exactly a factor of two.
    stats = latency_stats([case("c", latency_ms=10.0)])
    assert stats.mean_ms == pytest.approx(10.0)


# --- tokens and coverage ---------------------------------------------------


def test_token_totals_count_the_cases_that_reported_usage() -> None:
    results = [case("with", usage=True), case("without", usage=False)]
    totals = token_totals(results)
    assert totals.n_with_usage == 1
    assert totals.n_missing_usage == 1
    assert totals.input_tokens == 100
    assert totals.mean_input_tokens == pytest.approx(100.0)


def test_token_totals_with_no_usage_anywhere_are_null_not_zero() -> None:
    totals = token_totals([case("a", usage=False), case("b", usage=False)])
    assert totals.input_tokens is None
    assert totals.total_tokens is None
    assert totals.n_missing_usage == 2


def test_token_usage_coverage_is_the_fraction_of_attempted_cases_with_usage() -> None:
    metrics = metrics_over([case("with", usage=True), case("without", usage=False)])
    assert metrics.token_usage_coverage == pytest.approx(0.5)


# --- cost ------------------------------------------------------------------


def test_cost_sums_only_the_priced_cases_and_reports_partial_coverage() -> None:
    results = [case("a", cost=priced("0.002")), case("b", cost=unpriced())]
    metrics = metrics_over(results)
    assert metrics.cost.total_cost == Decimal("0.002")
    assert metrics.cost.priced is False, "a partial total is not a complete one"
    assert metrics.cost.unpriced_reason == "partial_coverage"
    assert metrics.cost_coverage == pytest.approx(0.5)


def test_a_fully_unpriced_run_reports_null_cost_and_never_zero() -> None:
    metrics = metrics_over([case("a", cost=unpriced()), case("b", cost=unpriced())])
    assert metrics.cost.total_cost is None
    assert metrics.cost.priced is False
    assert metrics.cost.unpriced_reason == "no_price_entry"
    assert metrics.cost_per_case is None
    assert metrics.cost_per_successful_evaluation is None
    assert metrics.cost_coverage == 0.0


def test_cost_per_case_divides_the_known_total_by_the_attempted_cases() -> None:
    results = [case("a", cost=priced("0.004")), case("b", cost=priced("0.006"))]
    metrics = metrics_over(results)
    assert metrics.cost.total_cost == Decimal("0.010")
    assert metrics.cost_per_case == Decimal("0.0050000000")


def test_cost_per_successful_evaluation_is_null_when_nothing_passed() -> None:
    results = [case("a", passed=False, score=0.0, cost=priced("0.004"))]
    metrics = metrics_over(results)
    assert metrics.n_passed == 0
    assert metrics.cost_per_successful_evaluation is None
    assert metrics.cost_per_case == Decimal("0.0040000000")


def test_the_price_table_identity_is_stamped_onto_the_rollup() -> None:
    metrics = metrics_over([case("a")])
    assert metrics.cost.price_table_id == PRICE_TABLE_ID
    assert metrics.cost.price_table_version == PRICE_TABLE_VERSION


def test_cost_is_decimal_end_to_end() -> None:
    metrics = metrics_over([case("a", cost=priced("0.0000012345"))])
    assert isinstance(metrics.cost.total_cost, Decimal)
    assert metrics.cost.total_cost == Decimal("0.0000012345")


# --- breakdowns ------------------------------------------------------------


def test_category_buckets_report_their_own_counts_and_intervals() -> None:
    results = [
        case("a", category="math", passed=True),
        case("b", category="math", passed=False, score=0.0),
        case("c", category="prose", passed=True),
    ]
    buckets = by_category(results, error_policy="exclude")
    assert set(buckets) == {"math", "prose"}
    assert buckets["math"].n == 2
    assert buckets["math"].n_passed == 1
    assert buckets["math"].pass_rate == pytest.approx(0.5)
    assert buckets["math"].pass_rate_ci is not None


def test_a_case_with_no_category_forms_no_bucket() -> None:
    assert by_category([case("a", category=None)], error_policy="exclude") == {}


def test_a_case_with_several_tags_is_counted_under_each() -> None:
    buckets = by_tag([case("a", tags=("smoke", "math"))], error_policy="exclude")
    assert set(buckets) == {"math", "smoke"}
    assert buckets["smoke"].n == 1
    assert buckets["math"].n == 1


def test_a_bucket_counts_errored_cases_in_n_but_not_in_the_denominator() -> None:
    results = [
        case("ok", category="math", passed=True),
        case(
            "bad",
            category="math",
            status=CaseStatus.ERROR,
            passed=None,
            error_kind=ProviderErrorKind.SERVER,
        ),
    ]
    bucket = by_category(results, error_policy="exclude")["math"]
    assert bucket.n == 2
    assert bucket.pass_rate == pytest.approx(1.0)


def test_evaluator_metrics_are_reported_separately_and_never_blended() -> None:
    results = [
        case(
            "a",
            evaluations=(
                evaluation(evaluator_id="exact_match", passed=True, score=1.0),
                evaluation(evaluator_id="lexical_similarity", passed=False, score=0.2),
            ),
            passed=False,
            score=0.6,
        )
    ]
    buckets = by_evaluator(results, error_policy="exclude")
    assert set(buckets) == {"exact_match", "lexical_similarity"}
    assert buckets["exact_match"].mean_score == pytest.approx(1.0)
    assert buckets["lexical_similarity"].mean_score == pytest.approx(0.2)


def test_an_errored_evaluation_leaves_the_evaluator_denominator_and_is_counted() -> None:
    results = [
        case(
            "a",
            status=CaseStatus.ERROR,
            passed=None,
            score=None,
            evaluations=(
                evaluation(
                    evaluator_id="judge", status=EvaluationStatus.ERROR, passed=None, score=None
                ),
            ),
        ),
        case("b", evaluations=(evaluation(evaluator_id="judge", passed=True, score=1.0),)),
    ]
    bucket = by_evaluator(results, error_policy="exclude")["judge"]
    assert bucket.n == 2
    assert bucket.n_errors == 1
    assert bucket.pass_rate == pytest.approx(1.0)


def test_score_only_evaluations_are_counted_at_both_levels() -> None:
    results = [
        case(
            "a",
            passed=None,
            score=0.8,
            evaluations=(evaluation(evaluator_id="judge", passed=None, score=0.8),),
        )
    ]
    metrics = metrics_over(results)
    assert metrics.n_score_only == 1
    assert metrics.by_evaluator["judge"].n_score_only == 1
    assert metrics.by_evaluator["judge"].pass_rate is None


def test_a_skipped_evaluation_is_not_counted_as_score_only() -> None:
    results = [
        case(
            "a",
            passed=None,
            score=None,
            evaluations=(
                evaluation(
                    evaluator_id="judge",
                    status=EvaluationStatus.SKIPPED,
                    passed=None,
                    score=None,
                ),
            ),
        )
    ]
    assert metrics_over(results).n_score_only == 0, "a skipped evaluation produced nothing at all"


# --- scores ----------------------------------------------------------------


def test_scores_are_averaged_over_successful_cases_only() -> None:
    results = [
        case("a", score=1.0),
        case("b", score=0.0, passed=False),
        case(
            "errored",
            status=CaseStatus.ERROR,
            passed=None,
            score=0.5,
            error_kind=ProviderErrorKind.SERVER,
        ),
    ]
    metrics = metrics_over(results)
    assert metrics.n_scored == 2
    assert metrics.mean_score == pytest.approx(0.5)
    assert metrics.median_score == pytest.approx(0.5)


def test_the_rollup_carries_the_run_case_count_even_when_cases_are_missing() -> None:
    metrics = metrics_over([case("a")], n_cases=10)
    assert metrics.n_cases == 10
    assert metrics.n_completed == 1


# --- the error policy is validated, never silently defaulted --------------


def test_an_unrecognised_error_policy_is_refused() -> None:
    # Falling back to "exclude" would make a typo in a config file change every
    # pass rate in the report while looking like it had been honoured.
    with pytest.raises(ValueError, match="error_policy"):
        metrics_over([case("a")], error_policy="excludee")


@pytest.mark.parametrize("policy", ["exclude", "fail"])
def test_both_documented_policies_are_accepted(policy: str) -> None:
    assert metrics_over([case("a")], error_policy=policy).error_policy == policy
