"""The two structural evaluators: `json_valid` and `json_schema`."""

import json
import time
from typing import Any

import pytest

from llm_eval_lab.datasets.resolve import resolve_suite
from llm_eval_lab.evaluators import default_registry
from llm_eval_lab.models import (
    BenchmarkSuite,
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorSpec,
    FinishReason,
    ModelResponse,
    ResolvedCase,
)
from llm_eval_lab.utils.time import utc_now

THREE_VIOLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "a": {"type": "integer"},
        "b": {"type": "integer"},
        "c": {"type": "integer"},
    },
    "required": ["a", "b", "c"],
}
THREE_VIOLATION_SCORE = 0.25
ONE_VIOLATION_SCORE = 0.5
N_VIOLATIONS = 3


def _case(spec: dict[str, Any]) -> ResolvedCase:
    suite = BenchmarkSuite.model_validate(
        {
            "schema_version": 1,
            "name": "json-evaluators",
            "version": "1",
            "cases": [{"id": "c1", "input": "prompt", "evaluators": [spec]}],
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
    spec: dict[str, Any],
    output: str,
    context: EvaluationContext,
) -> EvaluationResult:
    case = _case(spec)
    evaluator = default_registry().create(EvaluatorSpec.model_validate(spec))
    return await evaluator.evaluate(case, _response(output), context)


# --- json_valid -----------------------------------------------------------


async def test_json_valid_passes_on_an_object(evaluation_context: EvaluationContext) -> None:
    result = await _evaluate({"type": "json_valid"}, '{"a": 1}', evaluation_context)
    assert result.status is EvaluationStatus.PASSED
    assert result.passed is True
    assert result.score == 1.0
    assert result.metadata["top_level_type"] == "dict"


async def test_json_valid_accepts_any_json_value_by_default(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate({"type": "json_valid"}, "[1, 2, 3]", evaluation_context)
    assert result.passed is True


async def test_json_valid_can_require_an_object(evaluation_context: EvaluationContext) -> None:
    result = await _evaluate(
        {"type": "json_valid", "params": {"require_object": True}},
        "[1, 2, 3]",
        evaluation_context,
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.passed is False
    assert result.metadata["top_level_type"] == "list"


async def test_prose_is_a_failed_evaluation_not_an_error(
    evaluation_context: EvaluationContext,
) -> None:
    """A model that answered with prose answered wrongly; the evaluator worked."""
    result = await _evaluate({"type": "json_valid"}, "Sure! Here you go.", evaluation_context)
    assert result.status is EvaluationStatus.FAILED
    assert result.passed is False
    assert result.error is None


# --- markdown fences ------------------------------------------------------

FENCED = '```json\n{"a": 1, "b": 2, "c": 3}\n```'


async def test_a_fenced_block_parses_when_the_fence_is_allowed(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate({"type": "json_valid"}, FENCED, evaluation_context)
    assert result.passed is True
    assert result.metadata["fenced"] is True


async def test_a_fenced_block_fails_when_the_fence_is_not_allowed(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate(
        {"type": "json_valid", "params": {"allow_markdown_fence": False}},
        FENCED,
        evaluation_context,
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.passed is False


async def test_json_schema_honours_the_fence_setting_too(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "json_schema", "params": {"schema": THREE_VIOLATION_SCHEMA}}
    allowed = await _evaluate(spec, FENCED, evaluation_context)
    assert allowed.passed is True
    assert allowed.metadata["fenced"] is True

    refused = await _evaluate(
        {
            "type": "json_schema",
            "params": {"schema": THREE_VIOLATION_SCHEMA, "allow_markdown_fence": False},
        },
        FENCED,
        evaluation_context,
    )
    assert refused.passed is False


# --- json_schema partial credit ------------------------------------------


async def test_three_violations_score_one_quarter_and_are_all_listed(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate(
        {"type": "json_schema", "params": {"schema": THREE_VIOLATION_SCHEMA}},
        '{"a": "x", "b": "y", "c": "z"}',
        evaluation_context,
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.passed is False
    assert result.score == THREE_VIOLATION_SCORE
    assert result.raw_score == float(N_VIOLATIONS)
    assert result.metadata["n_errors"] == N_VIOLATIONS
    errors = result.metadata["errors"]
    assert isinstance(errors, list)
    assert len(errors) == N_VIOLATIONS
    assert {str(message).split(":")[0] for message in errors} == {"a", "b", "c"}


async def test_one_violation_scores_a_half(evaluation_context: EvaluationContext) -> None:
    result = await _evaluate(
        {"type": "json_schema", "params": {"schema": THREE_VIOLATION_SCHEMA}},
        '{"a": "x", "b": 2, "c": 3}',
        evaluation_context,
    )
    assert result.score == ONE_VIOLATION_SCORE
    assert result.metadata["n_errors"] == 1


async def test_a_conforming_response_scores_one(evaluation_context: EvaluationContext) -> None:
    result = await _evaluate(
        {"type": "json_schema", "params": {"schema": THREE_VIOLATION_SCHEMA}},
        '{"a": 1, "b": 2, "c": 3}',
        evaluation_context,
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.metadata["errors"] == []


async def test_unparseable_json_scores_zero_against_a_schema(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate(
        {"type": "json_schema", "params": {"schema": THREE_VIOLATION_SCHEMA}},
        "not json at all",
        evaluation_context,
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.score == 0.0
    assert result.error is None


# --- load-time schema refusals -------------------------------------------


def test_a_malformed_schema_is_refused_when_the_evaluator_is_built() -> None:
    from llm_eval_lab.models import EvaluatorConfigError  # noqa: PLC0415

    with pytest.raises(EvaluatorConfigError, match="invalid JSON Schema"):
        default_registry().create(
            EvaluatorSpec.model_validate(
                {"type": "json_schema", "params": {"schema": {"type": "not-a-type"}}}
            )
        )


def test_a_remote_ref_is_refused_by_name() -> None:
    """A benchmark file must not make validation fetch a document it chose."""
    from llm_eval_lab.models import EvaluatorConfigError  # noqa: PLC0415

    with pytest.raises(EvaluatorConfigError, match="non-local reference"):
        default_registry().create(
            EvaluatorSpec.model_validate(
                {
                    "type": "json_schema",
                    "params": {
                        "schema": {
                            "type": "object",
                            "properties": {"a": {"$ref": "https://example.invalid/s.json"}},
                        }
                    },
                }
            )
        )


def test_a_local_ref_is_accepted() -> None:
    evaluator = default_registry().create(
        EvaluatorSpec.model_validate(
            {
                "type": "json_schema",
                "params": {
                    "schema": {
                        "$defs": {"count": {"type": "integer"}},
                        "type": "object",
                        "properties": {"a": {"$ref": "#/$defs/count"}},
                    }
                },
            }
        )
    )
    assert evaluator.type == "json_schema"


def test_a_missing_schema_is_a_load_time_field_error() -> None:
    suite = BenchmarkSuite.model_validate(
        {
            "schema_version": 1,
            "name": "json-evaluators",
            "version": "1",
            "cases": [
                {"id": "needs-schema", "input": "x", "evaluators": [{"type": "json_schema"}]}
            ],
        }
    )
    errors = default_registry().validate_specs(suite, "suite.yaml")
    assert len(errors) == 1
    assert errors[0].case_id == "needs-schema"
    assert errors[0].location.endswith(".params.schema")


# --- the fence fact survives a parse failure ------------------------------

BAD_FENCED = "```json\n{not valid json at all}\n```"


async def test_a_fenced_block_of_bad_json_still_reports_that_it_was_fenced(
    evaluation_context: EvaluationContext,
) -> None:
    """Reporting `fenced: False` here described the response wrongly."""
    result = await _evaluate({"type": "json_valid"}, BAD_FENCED, evaluation_context)
    assert result.status is EvaluationStatus.FAILED
    assert result.metadata["fenced"] is True
    assert result.metadata["parse_error"]


async def test_json_schema_reports_the_fence_and_a_violation_count_on_bad_json(
    evaluation_context: EvaluationContext,
) -> None:
    """`raw_score` means violation count everywhere else, so it is set here too."""
    result = await _evaluate(
        {"type": "json_schema", "params": {"schema": THREE_VIOLATION_SCHEMA}},
        BAD_FENCED,
        evaluation_context,
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.score == 0.0
    assert result.raw_score == 1.0
    assert result.metadata["fenced"] is True
    assert result.metadata["n_errors"] == 1
    errors = result.metadata["errors"]
    assert isinstance(errors, list)
    assert len(errors) == 1


async def test_unfenced_bad_json_reports_no_fence(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate({"type": "json_valid"}, "{not json}", evaluation_context)
    assert result.metadata["fenced"] is False


# --- HIGH-1: a benchmark-supplied schema `pattern` is bounded -------------

import asyncio  # noqa: E402 - grouped with the bounded-execution tests it serves

from llm_eval_lab.evaluators.json_evaluators import (  # noqa: E402
    MAX_SCHEMA_BYTES,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_ERRORS,
    JsonSchemaEvaluator,
    scan_schema,
)

CATASTROPHIC_PATTERN = "^(a+)+$"
REDOS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "string", "pattern": CATASTROPHIC_PATTERN}},
}
REDOS_BUDGET_S = 5.0
REDOS_SUBJECT_CHARS = 40
MIN_LOOP_TICKS = 3


async def _tick(state: dict[str, int]) -> None:
    """Count how many times the event loop got a turn."""
    while True:
        state["ticks"] += 1
        await asyncio.sleep(0.05)


async def test_a_catastrophic_schema_pattern_is_abandoned_without_freezing_the_loop(
    evaluation_context: EvaluationContext,
) -> None:
    """The `regex` evaluator's guard, reached through JSON Schema's `pattern`.

    A CPython regex match holds the GIL for its whole duration, so an unbounded
    one does not merely fail this case: it stops the event loop, and with it
    every concurrent case and the API server hosting the run. The ticker asserts
    the loop kept running, which is the property a wall-clock timeout alone
    would not have given.
    """
    state = {"ticks": 0}
    ticker = asyncio.create_task(_tick(state))
    started = time.monotonic()
    try:
        result = await _evaluate(
            {"type": "json_schema", "params": {"schema": REDOS_SCHEMA}},
            '{"a": "' + "a" * REDOS_SUBJECT_CHARS + '!"}',
            evaluation_context,
        )
    finally:
        ticker.cancel()
    elapsed = time.monotonic() - started

    assert elapsed < REDOS_BUDGET_S
    assert result.status is EvaluationStatus.ERROR
    assert result.passed is None
    assert result.score is None
    assert result.error is not None
    assert result.error.kind == "timeout"
    assert result.metadata["error_code"] == "schema_timeout"
    assert result.metadata["bounded_in_subprocess"] is True
    assert state["ticks"] >= MIN_LOOP_TICKS


async def test_a_pattern_bearing_schema_still_scores_correctly(
    evaluation_context: EvaluationContext,
) -> None:
    """The subprocess path must produce the same verdicts as the in-process one."""
    spec = {
        "type": "json_schema",
        "params": {"schema": {"type": "object", "properties": {"a": {"pattern": "^[0-9]+$"}}}},
    }
    good = await _evaluate(spec, '{"a": "123"}', evaluation_context)
    assert good.passed is True
    assert good.score == 1.0
    assert good.metadata["bounded_in_subprocess"] is True

    bad = await _evaluate(spec, '{"a": "abc"}', evaluation_context)
    assert bad.passed is False
    assert bad.metadata["n_errors"] == 1
    assert "a:" in str(bad.metadata["errors"])


async def test_a_schema_without_a_pattern_stays_in_process(
    evaluation_context: EvaluationContext,
) -> None:
    """An interpreter start per case would be a real cost for no benefit."""
    result = await _evaluate(
        {"type": "json_schema", "params": {"schema": THREE_VIOLATION_SCHEMA}},
        '{"a": 1, "b": 2, "c": 3}',
        evaluation_context,
    )
    assert result.metadata["bounded_in_subprocess"] is False


def test_the_schema_scan_finds_every_regex_keyword() -> None:
    assert scan_schema({"type": "string"}).uses_regex is False
    assert scan_schema({"pattern": "^a$"}).uses_regex is True
    assert scan_schema({"patternProperties": {"^x": {"type": "string"}}}).uses_regex is True
    nested: dict[str, Any] = {
        "type": "array",
        "items": {"anyOf": [{"type": "string", "pattern": "^a$"}]},
    }
    assert scan_schema(nested).uses_regex is True


def test_format_is_an_annotation_and_never_compiles_a_regex() -> None:
    """No format_checker is ever passed, so `format` asserts nothing.

    If that ever changes, `format` becomes a third regex keyword and has to join
    `SCHEMA_REGEX_KEYWORDS`, or the bound is silently lost again.
    """
    spec = EvaluatorSpec.model_validate(
        {
            "type": "json_schema",
            "params": {"schema": {"type": "object", "properties": {"a": {"format": "email"}}}},
        }
    )
    evaluator = default_registry().create(spec)
    assert isinstance(evaluator, JsonSchemaEvaluator)
    assert evaluator._needs_worker is False
    assert evaluator._validator.FORMAT_CHECKER is not None
    assert evaluator._validator.format_checker is None


async def test_a_format_keyword_does_not_fail_a_non_conforming_value(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate(
        {
            "type": "json_schema",
            "params": {"schema": {"type": "object", "properties": {"a": {"format": "email"}}}},
        },
        '{"a": "not-an-email"}',
        evaluation_context,
    )
    assert result.passed is True


# --- MED-1: violations are capped before they are materialised ------------

WIDE_BRANCHES = 60
WIDE_ITEMS = 200


async def test_a_wide_anyof_against_a_long_array_stops_at_the_cap(
    evaluation_context: EvaluationContext,
) -> None:
    """`sorted()` over the full generator drained hundreds of megabytes first."""
    schema: dict[str, Any] = {
        "type": "array",
        "items": {"anyOf": [{"const": f"never-{index}"} for index in range(WIDE_BRANCHES)]},
    }
    output = json.dumps(["nope"] * WIDE_ITEMS)
    result = await _evaluate(
        {"type": "json_schema", "params": {"schema": schema}}, output, evaluation_context
    )

    assert result.status is EvaluationStatus.FAILED
    assert result.metadata["n_errors"] == MAX_SCHEMA_ERRORS
    assert result.metadata["errors_truncated"] is True
    errors = result.metadata["errors"]
    assert isinstance(errors, list)
    assert len(errors) == MAX_SCHEMA_ERRORS


async def test_violations_with_mixed_index_and_name_paths_sort_without_crashing(
    evaluation_context: EvaluationContext,
) -> None:
    """Array indices are ints and property names are strs; comparing them raises."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "integer"}},
            "name": {"type": "integer"},
        },
    }
    result = await _evaluate(
        {"type": "json_schema", "params": {"schema": schema}},
        '{"items": ["x", "y"], "name": "z"}',
        evaluation_context,
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.metadata["n_errors"] == N_VIOLATIONS


# --- MED-2: size and depth are refused at load time -----------------------


def _nested(depth: int) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object"}
    for _ in range(depth):
        schema = {"type": "object", "properties": {"a": schema}}
    return schema


# Deep enough that the pre-fix walk raised `RecursionError`, shallow enough that
# pydantic still accepts the surrounding `dict[str, JSONValue]`. Past roughly 150
# pydantic refuses the params first, which is safe but tests a different path.
DEEP_BUT_ACCEPTED_DEPTH = 100


def test_a_deeply_nested_schema_is_a_field_error_not_a_recursion_error() -> None:
    from llm_eval_lab.models import EvaluatorConfigError  # noqa: PLC0415

    with pytest.raises(EvaluatorConfigError, match="nests deeper than"):
        default_registry().create(
            EvaluatorSpec.model_validate(
                {"type": "json_schema", "params": {"schema": _nested(DEEP_BUT_ACCEPTED_DEPTH)}}
            )
        )


def test_a_deeply_nested_schema_names_the_case_at_suite_load_time() -> None:
    suite = BenchmarkSuite.model_validate(
        {
            "schema_version": 1,
            "name": "json-evaluators",
            "version": "1",
            "cases": [
                {
                    "id": "too-deep",
                    "input": "x",
                    "evaluators": [
                        {
                            "type": "json_schema",
                            "params": {"schema": _nested(DEEP_BUT_ACCEPTED_DEPTH)},
                        }
                    ],
                }
            ],
        }
    )
    errors = default_registry().validate_specs(suite, "suite.yaml")
    assert errors
    assert errors[0].case_id == "too-deep"
    assert errors[0].location.endswith(".params.schema")


def test_a_reasonably_nested_schema_is_accepted() -> None:
    evaluator = default_registry().create(
        EvaluatorSpec.model_validate(
            {"type": "json_schema", "params": {"schema": _nested(MAX_SCHEMA_DEPTH // 4)}}
        )
    )
    assert evaluator.type == "json_schema"


def test_an_oversized_schema_is_refused_at_load_time() -> None:
    from llm_eval_lab.models import EvaluatorConfigError  # noqa: PLC0415

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {f"field_{index}": {"type": "string"} for index in range(4000)},
    }
    assert len(json.dumps(schema)) > MAX_SCHEMA_BYTES
    with pytest.raises(EvaluatorConfigError, match="over the"):
        default_registry().create(
            EvaluatorSpec.model_validate({"type": "json_schema", "params": {"schema": schema}})
        )


def test_an_absurdly_nested_schema_is_refused_without_a_recursion_error() -> None:
    """Past the depth pydantic itself accepts, the refusal comes from pydantic.

    Both bands must refuse rather than raise `RecursionError`, which is what
    escaped the load-time validation pass and surfaced as a traceback from
    `llm-eval validate` and an opaque 500 from the validate endpoint.
    """
    from pydantic import ValidationError  # noqa: PLC0415

    with pytest.raises(ValidationError):
        BenchmarkSuite.model_validate(
            {
                "schema_version": 1,
                "name": "json-evaluators",
                "version": "1",
                "cases": [
                    {
                        "id": "absurd",
                        "input": "x",
                        "evaluators": [{"type": "json_schema", "params": {"schema": _nested(400)}}],
                    }
                ],
            }
        )
