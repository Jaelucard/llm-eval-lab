"""The five deterministic evaluators, including the regex ReDoS bound."""

import time
from typing import Any

import pytest

from llm_eval_lab.datasets.resolve import resolve_suite
from llm_eval_lab.evaluators import default_registry
from llm_eval_lab.models import (
    BenchmarkSuite,
    BenchmarkValidationError,
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorSpec,
    FinishReason,
    JSONValue,
    ModelResponse,
    ResolvedCase,
)
from llm_eval_lab.utils.time import utc_now

TWO_OF_THREE = 2 / 3
REDOS_BUDGET_S = 5.0
CATASTROPHIC_PATTERN = "(a+)+$"


def _case(expected: JSONValue, spec: dict[str, Any]) -> ResolvedCase:
    suite = BenchmarkSuite.model_validate(
        {
            "schema_version": 1,
            "name": "evaluators",
            "version": "1",
            "cases": [{"id": "c1", "input": "prompt", "expected": expected, "evaluators": [spec]}],
        }
    )
    return resolve_suite(suite).cases[0]


def _response(text: str) -> ModelResponse:
    now = utc_now()
    return ModelResponse(
        output_text=text,
        provider="fake",
        model="fake-1",
        requested_model="fake-1",
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        total_latency_ms=1.0,
        started_at=now,
        completed_at=now,
    )


async def _evaluate(
    expected: JSONValue,
    spec: dict[str, Any],
    output: str,
    context: EvaluationContext,
) -> EvaluationResult:
    case = _case(expected, spec)
    evaluator = default_registry().create(EvaluatorSpec.model_validate(spec))
    return await evaluator.evaluate(case, _response(output), context)


# --- exact_match ----------------------------------------------------------


async def test_exact_match_passes_on_a_match(evaluation_context: EvaluationContext) -> None:
    result = await _evaluate("Paris", {"type": "exact_match"}, "Paris", evaluation_context)
    assert result.status is EvaluationStatus.PASSED
    assert result.passed is True
    assert result.score == 1.0


async def test_exact_match_fails_otherwise(evaluation_context: EvaluationContext) -> None:
    result = await _evaluate("Paris", {"type": "exact_match"}, "Lyon", evaluation_context)
    assert result.status is EvaluationStatus.FAILED
    assert result.passed is False
    assert result.score == 0.0


async def test_exact_match_strips_by_default(evaluation_context: EvaluationContext) -> None:
    result = await _evaluate("Paris", {"type": "exact_match"}, "  Paris\n", evaluation_context)
    assert result.passed is True


async def test_exact_match_can_keep_surrounding_whitespace(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "exact_match", "params": {"strip": False}}
    result = await _evaluate("Paris", spec, "  Paris ", evaluation_context)
    assert result.passed is False


async def test_exact_match_normalizes_whitespace_when_asked(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "exact_match", "params": {"normalize_whitespace": True}}
    result = await _evaluate("hello world", spec, "hello   \n  world", evaluation_context)
    assert result.passed is True


async def test_exact_match_without_an_expected_value_is_an_error(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate(None, {"type": "exact_match"}, "anything", evaluation_context)
    assert result.status is EvaluationStatus.ERROR
    assert result.passed is None
    assert result.metadata["error_code"] == "missing_expected"


# --- exact_match_ci -------------------------------------------------------


async def test_exact_match_ci_ignores_case(evaluation_context: EvaluationContext) -> None:
    passing = await _evaluate("paris", {"type": "exact_match_ci"}, "Paris", evaluation_context)
    failing = await _evaluate("paris", {"type": "exact_match_ci"}, "Lyon", evaluation_context)
    assert passing.passed is True
    assert failing.passed is False


async def test_case_sensitive_exact_match_fails_the_same_pair(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate("paris", {"type": "exact_match"}, "Paris", evaluation_context)
    assert result.passed is False


# --- contains -------------------------------------------------------------


async def test_contains_all_scores_the_fraction_found(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "contains", "params": {"values": ["alpha", "beta", "gamma"], "mode": "all"}}
    result = await _evaluate(None, spec, "alpha and beta only", evaluation_context)
    assert result.passed is False
    assert result.score == pytest.approx(TWO_OF_THREE)
    assert result.metadata["missing"] == ["gamma"]


async def test_contains_any_passes_when_one_is_present(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "contains", "params": {"values": ["alpha", "beta", "gamma"], "mode": "any"}}
    result = await _evaluate(None, spec, "alpha and beta only", evaluation_context)
    assert result.passed is True
    assert result.score == pytest.approx(TWO_OF_THREE)


async def test_contains_case_sensitivity_toggles_both_modes(
    evaluation_context: EvaluationContext,
) -> None:
    sensitive = {"type": "contains", "params": {"values": ["Alpha"], "mode": "all"}}
    insensitive = {
        "type": "contains",
        "params": {"values": ["Alpha"], "mode": "all", "case_sensitive": False},
    }
    assert (await _evaluate(None, sensitive, "alpha", evaluation_context)).passed is False
    assert (await _evaluate(None, insensitive, "alpha", evaluation_context)).passed is True


async def test_contains_requires_at_least_one_value() -> None:
    with pytest.raises(Exception, match="at least 1 item"):
        default_registry().create(
            EvaluatorSpec.model_validate({"type": "contains", "params": {"values": []}})
        )


# --- regex ----------------------------------------------------------------


async def test_regex_must_match_true(evaluation_context: EvaluationContext) -> None:
    spec = {"type": "regex", "params": {"pattern": r"^\d+$"}}
    assert (await _evaluate(None, spec, "42", evaluation_context)).passed is True
    assert (await _evaluate(None, spec, "forty two", evaluation_context)).passed is False


async def test_regex_must_match_false_inverts_the_polarity(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "regex", "params": {"pattern": r"\d", "must_match": False}}
    assert (await _evaluate(None, spec, "no digits here", evaluation_context)).passed is True
    assert (await _evaluate(None, spec, "one 1 digit", evaluation_context)).passed is False


async def test_regex_group_captures_into_metadata(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "regex", "params": {"pattern": r"order (\d+)", "group": 1}}
    result = await _evaluate(None, spec, "your order 1234 shipped", evaluation_context)
    assert result.passed is True
    assert result.metadata["captured"] == "1234"


def test_an_invalid_regex_is_a_load_time_error(fixtures_dir: Any) -> None:
    from llm_eval_lab.datasets.loader import load_suite  # noqa: PLC0415 - reads next to its use

    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(
            fixtures_dir / "suites" / "malformed_bad_evaluator.yaml",
            validators=(default_registry().validate_specs,),
        )
    locations = {item.location for item in caught.value.errors}
    assert "cases[1].evaluators[0].params.pattern" in locations


def test_a_group_index_the_pattern_lacks_is_a_load_time_error() -> None:
    with pytest.raises(Exception, match="does not exist"):
        default_registry().create(
            EvaluatorSpec.model_validate(
                {"type": "regex", "params": {"pattern": "abc", "group": 2}}
            )
        )


async def test_a_catastrophic_regex_times_out_without_hanging_the_run(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {
        "type": "regex",
        "params": {"pattern": CATASTROPHIC_PATTERN, "match_timeout_s": 1.0},
    }
    subject = "a" * 40 + "b"

    started = time.monotonic()
    result = await _evaluate(None, spec, subject, evaluation_context)
    elapsed = time.monotonic() - started

    assert elapsed < REDOS_BUDGET_S
    assert result.status is EvaluationStatus.ERROR
    assert result.error is not None
    assert result.error.kind == "timeout"
    assert result.metadata["error_code"] == "regex_timeout"


# --- numeric_tolerance ----------------------------------------------------


async def test_numeric_tolerance_absolute(evaluation_context: EvaluationContext) -> None:
    spec = {"type": "numeric_tolerance", "params": {"abs_tol": 0.01}}
    passing = await _evaluate(3.14, spec, "3.141", evaluation_context)
    failing = await _evaluate(3.14, spec, "3.2", evaluation_context)
    assert passing.passed is True
    assert passing.raw_score == pytest.approx(3.141)
    assert failing.passed is False


async def test_numeric_tolerance_relative_is_independent(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "numeric_tolerance", "params": {"rel_tol": 0.05}}
    assert (await _evaluate(100.0, spec, "103", evaluation_context)).passed is True
    assert (await _evaluate(100.0, spec, "120", evaluation_context)).passed is False


async def test_numeric_tolerance_extract_last(evaluation_context: EvaluationContext) -> None:
    spec = {"type": "numeric_tolerance", "params": {"abs_tol": 0.01, "extract": "last"}}
    result = await _evaluate(
        7.0, spec, "First I considered 3, then 5, and the answer is 7", evaluation_context
    )
    assert result.passed is True
    assert result.raw_score == pytest.approx(7.0)


async def test_numeric_tolerance_extract_first_by_default(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "numeric_tolerance", "params": {"abs_tol": 0.01}}
    result = await _evaluate(3.0, spec, "3 then 9", evaluation_context)
    assert result.passed is True
    assert result.raw_score == pytest.approx(3.0)


async def test_unparseable_output_is_failed_not_errored(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "numeric_tolerance", "params": {"abs_tol": 0.01}}
    result = await _evaluate(3.14, spec, "no numbers at all", evaluation_context)
    assert result.status is EvaluationStatus.FAILED
    assert result.passed is False
    assert result.error is None
    assert result.metadata["error_code"] == "no_number_extracted"


async def test_numeric_tolerance_without_an_expected_number_errors(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "numeric_tolerance", "params": {"abs_tol": 0.01}}
    result = await _evaluate("not a number", spec, "3.14", evaluation_context)
    assert result.status is EvaluationStatus.ERROR
    assert result.metadata["error_code"] == "missing_expected"


# --- on_error policy ------------------------------------------------------


async def test_on_error_fail_turns_an_error_into_a_failure(
    evaluation_context: EvaluationContext,
) -> None:
    from llm_eval_lab.evaluators import apply_on_error  # noqa: PLC0415 - reads next to its use

    spec = EvaluatorSpec(type="exact_match", on_error="fail")
    errored = await _evaluate(None, {"type": "exact_match"}, "anything", evaluation_context)
    reshaped = apply_on_error(errored, spec)
    assert reshaped.status is EvaluationStatus.FAILED
    assert reshaped.passed is False
    assert reshaped.metadata["original_status"] == "error"


async def test_on_error_skip_drops_the_evaluation_from_scoring(
    evaluation_context: EvaluationContext,
) -> None:
    from llm_eval_lab.evaluators import apply_on_error  # noqa: PLC0415 - reads next to its use

    spec = EvaluatorSpec(type="exact_match", on_error="skip")
    errored = await _evaluate(None, {"type": "exact_match"}, "anything", evaluation_context)
    reshaped = apply_on_error(errored, spec)
    assert reshaped.status is EvaluationStatus.SKIPPED
    assert reshaped.passed is None
