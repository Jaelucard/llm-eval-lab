"""Behavioural tests for the frozen contract layer.

Two things are proved here. First, that every contract model survives a
JSON round trip unchanged, because these objects are persisted as JSON and
returned over an HTTP API, so a field that cannot make the trip is a data-loss
bug waiting to happen. Second, that the validators which exist to prevent a
specific accident actually fire: a credential smuggled into a config, a
response that is neither a success nor a failure, a token total invented from
nothing, a threshold policy with no thresholds.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ValidationError

import llm_eval_lab.models as contracts
from llm_eval_lab.models import (
    RETRYABLE_KINDS,
    AggregateMetrics,
    BenchmarkCase,
    BenchmarkSnapshot,
    BenchmarkSuite,
    CaseDelta,
    CaseQuery,
    CaseResult,
    CaseSelection,
    CaseStatus,
    CategoryMetrics,
    ChatMessage,
    CheckOutcome,
    CheckStatus,
    CostBreakdown,
    EvaluationErrorInfo,
    EvaluationRecord,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorConfigError,
    EvaluatorMetrics,
    EvaluatorSettings,
    EvaluatorSpec,
    FieldError,
    FinishReason,
    GenerationParams,
    JudgeConfig,
    JudgeCriterion,
    JudgeProvenance,
    JudgeScale,
    JudgeVerdict,
    LatencyStats,
    MetricCheck,
    ModelResponse,
    Page,
    PluginRecord,
    PriceEntry,
    PriceTable,
    ProgressEvent,
    ProgressEventType,
    ProviderConfig,
    ProviderErrorInfo,
    ProviderErrorKind,
    ProviderRequest,
    RegressionReport,
    RegressionSummary,
    RegressionThresholds,
    ResolvedCase,
    ResolvedSuite,
    RetryPolicy,
    Run,
    RunConfig,
    RunQuery,
    RunStatus,
    RunSummary,
    RunTotals,
    SuiteDefaults,
    ThresholdDirection,
    TokenTotals,
    TokenUsage,
    Verdict,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

_MESSAGE = ChatMessage(role="user", content="What is 2 + 2?")
_PARAMS = GenerationParams(temperature=0.0, max_output_tokens=64, stop=("\n\n",))
_PROVIDER_CONFIG = ProviderConfig(
    provider="fake",
    model="fake-1",
    api_key_env="LLM_EVAL_FAKE_KEY",
    base_url="https://api.example.test/v1",
)
_EVALUATOR_SPEC = EvaluatorSpec(type="exact_match", id="exact_match", params={"strip": True})
_USAGE = TokenUsage(input_tokens=11, output_tokens=7)
_ERROR_INFO = ProviderErrorInfo(
    kind=ProviderErrorKind.RATE_LIMIT,
    message="429 from the vendor",
    provider="fake",
    attempts=3,
    status_code=429,
    retry_after_s=1.5,
    exception_type="ProviderRateLimitError",
)
_RESPONSE = ModelResponse(
    output_text="4",
    provider="fake",
    model="fake-1",
    requested_model="fake-1",
    finish_reason=FinishReason.STOP,
    usage=_USAGE,
    latency_ms=12.5,
    total_latency_ms=12.5,
    started_at=_AT,
    completed_at=_AT,
    raw={"id": "resp_1", "choices": [{"index": 0}]},
)
_JUDGE_SCALE = JudgeScale(labels={"1": "Completely wrong", "5": "Fully correct"})
_JUDGE_VERDICT = JudgeVerdict(
    score=4.0,
    per_criterion={"accuracy": 4.0},
    reasoning="Correct and complete.",
    confidence=0.8,
)
_JUDGE_PROVENANCE = JudgeProvenance(
    template_id="single_answer_grading_v2",
    rubric_hash="sha256:" + "a" * 64,
    scale=_JUDGE_SCALE,
    aggregation="median",
    verdicts=(_JUDGE_VERDICT,),
    judge_models=("fake:fake-judge",),
    raw_outputs=('{"score": 4}',),
    rendered_prompts=("Grade the answer.",),
    usage=_USAGE,
    cost_usd=Decimal("0.0004"),
    dispersion=0.0,
    agreement=1.0,
)
_EVALUATION = EvaluationResult(
    evaluator_id="exact_match",
    evaluator_type="exact_match",
    status=EvaluationStatus.PASSED,
    passed=True,
    score=1.0,
    metadata={"method": "strict"},
)
_COST = CostBreakdown(
    input_cost=Decimal("0.000011"),
    output_cost=Decimal("0.000021"),
    total_cost=Decimal("0.000032"),
    priced=True,
    price_table_id="builtin",
    price_table_version="2026-09-01",
    input_per_mtok=Decimal("1.00"),
    output_per_mtok=Decimal("3.00"),
)
_LATENCY = LatencyStats(
    n=3,
    mean_ms=12.0,
    p50_ms=12.0,
    p90_ms=13.0,
    p95_ms=13.0,
    p99_ms=13.0,
    max_ms=13.0,
    low_confidence=("p50_ms", "p95_ms", "p99_ms"),
)
_TOKENS = TokenTotals(
    input_tokens=33,
    output_tokens=21,
    total_tokens=54,
    mean_input_tokens=11.0,
    mean_output_tokens=7.0,
    n_with_usage=3,
    n_missing_usage=0,
)
_CATEGORY = CategoryMetrics(
    n=3, n_passed=3, pass_rate=1.0, pass_rate_ci=(0.44, 1.0), mean_score=1.0
)
_RUN_CONFIG = RunConfig(
    suite_name="arithmetic",
    suite_version="1",
    suite_hash="sha256:" + "b" * 64,
    suite_source="benchmarks/arithmetic.yaml",
    case_selection=CaseSelection(n_selected=3),
    provider=_PROVIDER_CONFIG,
    params=_PARAMS,
    evaluators=(_EVALUATOR_SPEC,),
    price_table_id="builtin",
    price_table_version="2026-09-01",
    price_table_hash="sha256:" + "c" * 64,
    library_version="0.1.0",
    python_version="3.13.0",
)
_RUN_SUMMARY = RunSummary(
    id="7b1f9c0e-0000-4000-8000-000000000001",
    label="nightly",
    status=RunStatus.COMPLETED,
    created_at=_AT,
    completed_at=_AT,
    provider="fake",
    model="fake-1",
    suite_name="arithmetic",
    suite_version="1",
    suite_hash="sha256:" + "b" * 64,
    n_cases=3,
    n_completed=3,
    n_errors=0,
    pass_rate=1.0,
    total_cost=Decimal("0.000096"),
    p95_ms=13.0,
)
_CHECK_OUTCOME = CheckOutcome(
    metric="pass_rate",
    label="pass rate",
    direction=ThresholdDirection.HIGHER_IS_BETTER,
    status=CheckStatus.PASSED,
    baseline=0.90,
    candidate=0.92,
    delta=0.02,
    relative_delta=0.0222,
    violated_bound=None,
    threshold_value=None,
    n_baseline=50,
    n_candidate=50,
    n_paired=50,
    confidence_interval=(-0.03, 0.07),
    p_value=0.5,
    significant=False,
    note=None,
)

# One constructed instance of every pydantic contract model. A model missing
# from this list is caught by `test_every_exported_model_has_a_round_trip_case`,
# so the round-trip guarantee cannot silently develop a hole.
CONTRACT_INSTANCES: tuple[BaseModel, ...] = (
    _MESSAGE,
    _PARAMS,
    _PROVIDER_CONFIG,
    ProviderRequest(messages=(_MESSAGE,), params=_PARAMS, request_id="run:case"),
    _USAGE,
    _ERROR_INFO,
    _RESPONSE,
    _EVALUATOR_SPEC,
    BenchmarkCase(id="add-1", input="What is 2 + 2?", expected="4", category="arithmetic"),
    SuiteDefaults(system="Answer with digits only.", params=_PARAMS),
    BenchmarkSuite(
        name="arithmetic",
        cases=(BenchmarkCase(id="add-1", input="What is 2 + 2?", expected="4"),),
    ),
    ResolvedCase(
        id="add-1",
        case_hash="sha256:" + "d" * 64,
        messages=(_MESSAGE,),
        system=None,
        expected="4",
        category="arithmetic",
        tags=("smoke",),
        weight=1.0,
        metadata={},
        params=_PARAMS,
        evaluators=(_EVALUATOR_SPEC,),
    ),
    ResolvedSuite(
        name="arithmetic",
        version="1",
        suite_hash="sha256:" + "b" * 64,
        cases=(),
        metadata={},
    ),
    BenchmarkSnapshot(
        suite_hash="sha256:" + "b" * 64,
        name="arithmetic",
        version="1",
        n_cases=1,
        created_at=_AT,
        source_path="benchmarks/arithmetic.yaml",
        body={"name": "arithmetic"},
    ),
    EvaluationErrorInfo(kind="parse", message="judge returned unparseable text"),
    _JUDGE_SCALE,
    JudgeCriterion(id="accuracy", description="Is the answer factually right?"),
    JudgeConfig(provider=_PROVIDER_CONFIG, rubric="Grade 1-5 for correctness."),
    _JUDGE_VERDICT,
    _JUDGE_PROVENANCE,
    _EVALUATION,
    CaseSelection(tags=("smoke",), limit=10, sample_seed=7, n_selected=3),
    PluginRecord(
        entry_point="acme_evals",
        distribution="acme-evals",
        version="0.3.1",
        evaluator_types=("acme_rubric",),
    ),
    RunTotals(n_cases=3, n_completed=3, n_passed=3),
    RetryPolicy(),
    _RUN_CONFIG,
    Run(
        id="7b1f9c0e-0000-4000-8000-000000000001",
        label="nightly",
        status=RunStatus.COMPLETED,
        created_at=_AT,
        started_at=_AT,
        completed_at=_AT,
        config=_RUN_CONFIG,
        totals=RunTotals(n_cases=3, n_completed=3, n_passed=3),
        updated_at=_AT,
        warnings=("self_preference_risk",),
    ),
    CaseResult(
        run_id="7b1f9c0e-0000-4000-8000-000000000001",
        case_id="add-1",
        case_hash="sha256:" + "d" * 64,
        status=CaseStatus.OK,
        response=_RESPONSE,
        evaluations=(_EVALUATION,),
        passed=True,
        score=1.0,
        cost=_COST,
        started_at=_AT,
        completed_at=_AT,
    ),
    _LATENCY,
    _TOKENS,
    _CATEGORY,
    EvaluatorMetrics(
        evaluator_id="exact_match",
        evaluator_type="exact_match",
        n=3,
        n_passed=3,
        n_errors=0,
        pass_rate=1.0,
        mean_score=1.0,
        median_score=1.0,
    ),
    AggregateMetrics(
        run_id="7b1f9c0e-0000-4000-8000-000000000001",
        computed_at=_AT,
        n_cases=3,
        n_completed=3,
        n_errors=0,
        n_timeouts=0,
        n_skipped=0,
        error_rate=0.0,
        error_breakdown={},
        error_policy="exclude",
        pass_rate=1.0,
        pass_rate_ci=(0.44, 1.0),
        n_scored=3,
        mean_score=1.0,
        median_score=1.0,
        latency=_LATENCY,
        tokens=_TOKENS,
        token_usage_coverage=1.0,
        cost=_COST,
        cost_coverage=1.0,
        cost_per_case=Decimal("0.000032"),
        cost_per_successful_evaluation=Decimal("0.000032"),
        by_category={"arithmetic": _CATEGORY},
        by_tag={"smoke": _CATEGORY},
        by_evaluator={},
    ),
    ProgressEvent(
        type=ProgressEventType.CASE_COMPLETED,
        run_id="7b1f9c0e-0000-4000-8000-000000000001",
        case_id="add-1",
        completed=1,
        total=3,
        at=_AT,
    ),
    Page[RunSummary](items=(_RUN_SUMMARY,), total=1, limit=50, offset=0),
    RunQuery(status=RunStatus.COMPLETED, provider="fake"),
    CaseQuery(passed=True, tag="smoke"),
    _RUN_SUMMARY,
    EvaluationRecord(
        run_id="7b1f9c0e-0000-4000-8000-000000000001",
        case_id="add-1",
        result=_EVALUATION,
    ),
    PriceEntry(
        provider="fake",
        model="fake-1",
        input_per_mtok=Decimal("1.00"),
        output_per_mtok=Decimal("3.00"),
    ),
    PriceTable(
        id="builtin",
        version="2026-09-01",
        models=(
            PriceEntry(
                provider="fake",
                model="fake-1",
                input_per_mtok=Decimal("1.00"),
                output_per_mtok=Decimal("3.00"),
            ),
        ),
        content_hash="sha256:" + "e" * 64,
    ),
    _COST,
    MetricCheck(
        metric="pass_rate",
        direction=ThresholdDirection.HIGHER_IS_BETTER,
        max_absolute_decrease=0.02,
    ),
    RegressionThresholds(
        checks=(MetricCheck(metric="pass_rate", direction=ThresholdDirection.HIGHER_IS_BETTER),),
    ),
    _CHECK_OUTCOME,
    CaseDelta(
        case_id="add-1",
        baseline_score=1.0,
        candidate_score=0.5,
        delta=-0.5,
        baseline_passed=True,
        candidate_passed=False,
    ),
    RegressionSummary(
        newly_failing=("add-1",),
        newly_passing=(),
        still_failing=(),
        n_flipped=1,
        largest_score_drops=(),
    ),
    RegressionReport(
        generated_at=_AT,
        baseline_run_id="7b1f9c0e-0000-4000-8000-000000000001",
        candidate_run_id="7b1f9c0e-0000-4000-8000-000000000002",
        baseline_label="nightly",
        candidate_label="candidate",
        verdict=Verdict.PASS,
        mode="paired",
        comparable=True,
        incomparable_reason=None,
        suite_hash_match=True,
        baseline_suite_hash="sha256:" + "b" * 64,
        candidate_suite_hash="sha256:" + "b" * 64,
        paired_case_count=3,
        only_in_baseline=(),
        only_in_candidate=(),
        changed_cases=(),
        thresholds_id="default",
        thresholds_version="1",
        checks=(_CHECK_OUTCOME,),
        summary=RegressionSummary(
            newly_failing=(),
            newly_passing=(),
            still_failing=(),
            n_flipped=0,
            largest_score_drops=(),
        ),
    ),
    EvaluatorSettings(),
)


def _model_response_kwargs() -> dict[str, object]:
    """A minimal valid ModelResponse payload, for tests that vary one field."""
    return {
        "output_text": "4",
        "provider": "fake",
        "model": "fake-1",
        "requested_model": "fake-1",
        "finish_reason": FinishReason.STOP,
        "latency_ms": 1.0,
        "total_latency_ms": 1.0,
        "started_at": _AT,
        "completed_at": _AT,
    }


def _run_kwargs() -> dict[str, object]:
    """A minimal valid Run payload, for tests that vary one field."""
    return {
        "id": "7b1f9c0e-0000-4000-8000-000000000001",
        "label": None,
        "status": RunStatus.COMPLETED,
        "created_at": _AT,
        "started_at": _AT,
        "completed_at": _AT,
        "config": _RUN_CONFIG,
        "totals": RunTotals(),
        "updated_at": _AT,
    }


def _model_class(instance: BaseModel) -> type[BaseModel]:
    """Return the defining class of an instance, unwrapping generic aliases."""
    origin = type(instance).__pydantic_generic_metadata__["origin"]
    return origin if origin is not None else type(instance)


def _exported_models() -> Iterator[type[BaseModel]]:
    """Yield every pydantic model exported from `llm_eval_lab.models`."""
    for name in contracts.__all__:
        obj = getattr(contracts, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            yield obj


@pytest.mark.parametrize(
    "instance",
    CONTRACT_INSTANCES,
    ids=[type(i).__name__ for i in CONTRACT_INSTANCES],
)
def test_contract_model_round_trips_through_json(instance: BaseModel) -> None:
    restored = type(instance).model_validate_json(instance.model_dump_json())
    assert restored == instance


def test_every_exported_model_has_a_round_trip_case() -> None:
    covered = {_model_class(instance) for instance in CONTRACT_INSTANCES}
    missing = sorted(model.__name__ for model in _exported_models() if model not in covered)
    assert missing == [], f"exported contract models with no round-trip case: {missing}"


def test_provider_config_rejects_a_secret_shaped_option_key() -> None:
    with pytest.raises(ValidationError, match="api_key"):
        ProviderConfig(provider="fake", model="fake-1", options={"api_key": "sk-not-a-real-key"})


def test_provider_config_rejects_credentials_embedded_in_base_url() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig(provider="fake", model="fake-1", base_url="https://u:p@host/v1")


def test_model_response_rejects_an_error_alongside_output_text() -> None:
    with pytest.raises(ValidationError):
        ModelResponse(
            output_text="4",
            provider="fake",
            model="fake-1",
            requested_model="fake-1",
            finish_reason=FinishReason.ERROR,
            latency_ms=1.0,
            total_latency_ms=1.0,
            started_at=_AT,
            completed_at=_AT,
            error=_ERROR_INFO,
        )


def test_model_response_rejects_absent_output_with_no_error() -> None:
    with pytest.raises(ValidationError):
        ModelResponse(
            output_text=None,
            provider="fake",
            model="fake-1",
            requested_model="fake-1",
            finish_reason=FinishReason.STOP,
            latency_ms=1.0,
            total_latency_ms=1.0,
            started_at=_AT,
            completed_at=_AT,
        )


def test_token_usage_never_fabricates_the_halves_from_a_total() -> None:
    usage = TokenUsage(total_tokens=100)
    assert usage.input_tokens is None
    assert usage.output_tokens is None


def test_token_usage_derives_the_total_from_both_halves() -> None:
    assert TokenUsage(input_tokens=10, output_tokens=5).total_tokens == 15


def test_benchmark_case_rejects_both_prompt_forms() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase(id="add-1", input="2 + 2?", messages=(_MESSAGE,))


def test_benchmark_case_rejects_neither_prompt_form() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCase(id="add-1")


def test_regression_thresholds_rejects_an_empty_check_list() -> None:
    with pytest.raises(ValidationError):
        RegressionThresholds(checks=())


def test_require_significant_is_reserved_and_refused() -> None:
    with pytest.raises(EvaluatorConfigError, match="reserved for a future release"):
        MetricCheck(
            metric="pass_rate",
            direction=ThresholdDirection.HIGHER_IS_BETTER,
            require_significant=True,
        )


def test_metric_check_documents_its_bound_semantics_on_the_fields() -> None:
    """The exclusive ceiling has to be written down where phase 4 will read it.

    `MetricCheck` is pure data: evaluation lives in the phase 4 engine. The
    asymmetry between an inclusive floor and an exclusive ceiling is therefore
    carried in the field descriptions, and this is what stops it being dropped.
    """
    check = MetricCheck(
        metric="error_rate",
        direction=ThresholdDirection.LOWER_IS_BETTER,
        max_value=0.01,
    )
    assert check.max_value == 0.01
    assert check.min_value is None

    max_doc = MetricCheck.model_fields["max_value"].description or ""
    min_doc = MetricCheck.model_fields["min_value"].description or ""
    assert "EXCLUSIVE" in max_doc
    assert "FAILS" in max_doc
    assert "INCLUSIVE" in min_doc


def test_metric_check_carries_no_evaluation_logic() -> None:
    """Gate semantics belong to the phase 4 engine, not to a frozen model."""
    assert not hasattr(MetricCheck, "evaluate")
    assert not hasattr(MetricCheck, "_first_violated_bound")


def test_provider_config_rejects_a_credential_nested_in_options() -> None:
    with pytest.raises(ValidationError, match=r"options\.headers\.Authorization"):
        ProviderConfig(
            provider="fake",
            model="fake-1",
            options={"headers": {"Authorization": "Bearer sk-not-a-real-key"}},
        )


def test_provider_config_rejects_a_credential_nested_inside_a_list() -> None:
    with pytest.raises(ValidationError, match=r"options\.h\[0\]\.api_key"):
        ProviderConfig(
            provider="fake",
            model="fake-1",
            options={"h": [{"api_key": "sk-not-a-real-key"}]},
        )


def test_provider_config_accepts_a_benign_nested_options_tree() -> None:
    """Negative control: the recursive walk must not refuse ordinary nesting."""
    config = ProviderConfig(
        provider="fake",
        model="fake-1",
        options={"retry": {"modes": ["fast", "slow"]}, "region": "eu"},
    )
    assert config.options["region"] == "eu"


def test_model_response_rejects_error_finish_reason_with_no_error() -> None:
    """The third branch of the invariant: `succeeded` must never be True here."""
    with pytest.raises(ValidationError, match="must carry an error"):
        ModelResponse(
            output_text="partial",
            provider="fake",
            model="fake-1",
            requested_model="fake-1",
            finish_reason=FinishReason.ERROR,
            latency_ms=1.0,
            total_latency_ms=1.0,
            started_at=_AT,
            completed_at=_AT,
        )


@pytest.mark.parametrize("kind", list(ProviderErrorKind))
def test_provider_error_info_derives_retryability_from_the_kind(kind: ProviderErrorKind) -> None:
    info = ProviderErrorInfo(kind=kind, message="x", provider="fake")
    assert info.retryable is (kind in RETRYABLE_KINDS)


@pytest.mark.parametrize("kind", list(ProviderErrorKind))
def test_provider_error_info_refuses_a_retryable_flag_contradicting_its_kind(
    kind: ProviderErrorKind,
) -> None:
    with pytest.raises(ValidationError, match="contradicts kind"):
        ProviderErrorInfo(
            kind=kind,
            message="x",
            provider="fake",
            retryable=kind not in RETRYABLE_KINDS,
        )


def test_retryable_kinds_is_exactly_the_contract_table() -> None:
    assert {k.value for k in RETRYABLE_KINDS} == {
        "rate_limit",
        "timeout",
        "connection",
        "server",
    }


@pytest.mark.parametrize(
    ("model_cls", "field"),
    [(Run, "created_at"), (ModelResponse, "started_at")],
)
def test_timestamp_fields_reject_naive_datetimes(model_cls: type[BaseModel], field: str) -> None:
    """A naive timestamp round-trips faithfully, which is why it must never be accepted."""
    naive = datetime(2026, 1, 2, 3, 4, 5)  # noqa: DTZ001 - the point of the test
    payload = (_run_kwargs() if model_cls is Run else _model_response_kwargs()) | {field: naive}
    with pytest.raises(ValidationError, match="aware"):
        model_cls(**payload)


def test_generation_params_merge_inherits_unmentioned_fields() -> None:
    base = GenerationParams(temperature=0.7, max_output_tokens=100, extra={"a": 1})
    override = GenerationParams(max_output_tokens=200, extra={"b": 2})
    merged = base.merged_with(override)

    assert merged.temperature == 0.7, "an unmentioned field inherits rather than resetting"
    assert merged.max_output_tokens == 200
    assert merged.extra == {"a": 1, "b": 2}, "extra merges key by key"


def test_generation_params_merge_with_none_returns_self() -> None:
    base = GenerationParams(temperature=0.5)
    assert base.merged_with(None) is base


def test_generation_params_merge_can_set_a_field_back_to_none() -> None:
    """An explicit None in the override wins; `model_fields_set` is what makes that work."""
    base = GenerationParams(temperature=0.7)
    assert base.merged_with(GenerationParams(temperature=None)).temperature is None


def _price_table() -> PriceTable:
    return PriceTable(
        id="t",
        version="1",
        models=(
            PriceEntry(
                provider="fake",
                model="fake-1",
                input_per_mtok=Decimal(1),
                output_per_mtok=Decimal(2),
            ),
            PriceEntry(
                provider="fake",
                model="fake",
                match="prefix",
                input_per_mtok=Decimal(3),
                output_per_mtok=Decimal(4),
            ),
            PriceEntry(
                provider="fake",
                model="fake-2",
                match="prefix",
                input_per_mtok=Decimal(5),
                output_per_mtok=Decimal(6),
            ),
            PriceEntry(
                provider="fake",
                model="^zz",
                match="regex",
                input_per_mtok=Decimal(7),
                output_per_mtok=Decimal(8),
            ),
        ),
        content_hash="sha256:" + "f" * 64,
    )


def test_price_lookup_prefers_an_exact_match() -> None:
    entry = _price_table().lookup("fake", "fake-1")
    assert entry is not None
    assert entry.input_per_mtok == Decimal(1)


def test_price_lookup_falls_back_to_the_longest_prefix() -> None:
    entry = _price_table().lookup("fake", "fake-2-turbo")
    assert entry is not None
    assert entry.input_per_mtok == Decimal(5), "the longer prefix 'fake-2' must win over 'fake'"


def test_price_lookup_falls_back_to_a_regex() -> None:
    entry = _price_table().lookup("fake", "zz-9")
    assert entry is not None
    assert entry.input_per_mtok == Decimal(7)


def test_price_lookup_returns_none_on_a_miss_and_never_assumes_zero() -> None:
    assert _price_table().lookup("other", "fake-1") is None
    assert _price_table().lookup("fake", "nothing-matches-this") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1.0, 0.0), (5.0, 1.0), (3.0, 0.5), (0.0, 0.0), (9.0, 1.0)],
)
def test_judge_scale_normalizes_and_clamps(raw: float, expected: float) -> None:
    assert JudgeScale().normalize(raw) == expected


@pytest.mark.parametrize(("minimum", "maximum"), [(1.0, 1.0), (5.0, 1.0)])
def test_judge_scale_rejects_a_degenerate_or_inverted_scale(minimum: float, maximum: float) -> None:
    """A zero span would otherwise score every judgment 0.0, silently."""
    with pytest.raises(ValidationError, match="maximum > minimum"):
        JudgeScale(minimum=minimum, maximum=maximum)


def test_benchmark_suite_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValidationError, match="dup"):
        BenchmarkSuite(
            name="s",
            cases=(
                BenchmarkCase(id="dup", input="a"),
                BenchmarkCase(id="dup", input="b"),
            ),
        )


def _evaluation_kwargs() -> dict[str, object]:
    return {"evaluator_id": "e", "evaluator_type": "t", "status": EvaluationStatus.PASSED}


@pytest.mark.parametrize(
    ("status", "passed", "error"),
    [
        (EvaluationStatus.PASSED, True, None),
        (EvaluationStatus.PASSED, None, None),
        (EvaluationStatus.FAILED, False, None),
        (EvaluationStatus.SKIPPED, None, None),
        (EvaluationStatus.ERROR, None, EvaluationErrorInfo(kind="runtime", message="x")),
    ],
)
def test_evaluation_result_accepts_a_consistent_status_and_passed_pair(
    *,
    status: EvaluationStatus,
    passed: bool | None,
    error: EvaluationErrorInfo | None,
) -> None:
    result = EvaluationResult(
        **_evaluation_kwargs() | {"status": status, "passed": passed, "error": error}
    )
    assert result.status is status


@pytest.mark.parametrize(
    ("status", "passed", "error"),
    [
        (EvaluationStatus.PASSED, False, None),
        (EvaluationStatus.FAILED, True, None),
        (EvaluationStatus.FAILED, None, None),
        (EvaluationStatus.SKIPPED, True, None),
        (EvaluationStatus.ERROR, False, EvaluationErrorInfo(kind="runtime", message="x")),
        (EvaluationStatus.ERROR, None, None),
    ],
)
def test_evaluation_result_rejects_a_contradictory_status_and_passed_pair(
    *,
    status: EvaluationStatus,
    passed: bool | None,
    error: EvaluationErrorInfo | None,
) -> None:
    with pytest.raises(ValidationError):
        EvaluationResult(
            **_evaluation_kwargs() | {"status": status, "passed": passed, "error": error}
        )


def test_score_only_judgment_is_passed_status_with_no_boolean() -> None:
    """The one pairing `n_score_only` counts, pinned so three lanes agree on it."""
    result = EvaluationResult(
        evaluator_id="judge",
        evaluator_type="llm_judge",
        status=EvaluationStatus.PASSED,
        passed=None,
        score=0.75,
    )
    assert result.passed is None
    assert result.score == 0.75


def test_field_error_serializes_for_the_validate_endpoint() -> None:
    """`FieldError` is a dataclass, so it sits outside the round-trip test."""
    error = FieldError(
        file="suite.yaml",
        location="cases[7].evaluators[0].params.pattern",
        case_id="add-1",
        message="not a valid regex",
        input_repr="42",
    )
    payload = dataclasses.asdict(error)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["case_id"] == "add-1"


def test_run_config_has_no_spend_cap_by_default() -> None:
    """A cap is opt-in: an unset `max_cost` must not silently cancel a run."""
    assert _RUN_CONFIG.max_cost is None


def test_run_config_accepts_a_spend_cap_and_round_trips_it() -> None:
    config = RunConfig(**_RUN_CONFIG.model_dump() | {"max_cost": Decimal("2.50")})
    assert config.max_cost == Decimal("2.50")
    assert RunConfig.model_validate_json(config.model_dump_json()).max_cost == Decimal("2.50")


def test_run_config_rejects_a_negative_spend_cap() -> None:
    with pytest.raises(ValidationError):
        RunConfig(**_RUN_CONFIG.model_dump() | {"max_cost": Decimal(-1)})


def test_run_config_documents_the_budget_exceeded_cancellation() -> None:
    """The cap's meaning has to reach the runner, which will not read this test."""
    description = RunConfig.model_fields["max_cost"].description or ""
    assert "budget_exceeded" in description
