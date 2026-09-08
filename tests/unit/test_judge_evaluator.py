"""Judge aggregation, score-only accounting, cost separation and bias flags."""

import asyncio
import json
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import pytest
import structlog
from structlog.testing import capture_logs

from llm_eval_lab.datasets.resolve import resolve_suite
from llm_eval_lab.evaluators import default_registry
from llm_eval_lab.models import (
    BenchmarkSuite,
    CaseResult,
    CaseStatus,
    CostBreakdown,
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorConfigError,
    EvaluatorSettings,
    EvaluatorSpec,
    FinishReason,
    JSONValue,
    ModelResponse,
    ProviderConfig,
    ProviderError,
    ProviderErrorKind,
    ProviderRequest,
    ResolvedCase,
    TokenUsage,
)
from llm_eval_lab.reporting.aggregate import compute_metrics
from llm_eval_lab.utils.time import utc_now

JUDGE_PROVIDER: dict[str, JSONValue] = {"provider": "fake", "model": "fake-judge-1"}
CANDIDATE_PROVIDER = "fake"
CANDIDATE_MODEL = "fake-1"
RUBRIC = "Grade the answer for correctness and completeness."
SCALE: dict[str, JSONValue] = {"kind": "integer", "minimum": 1, "maximum": 5}

JUDGE_INPUT_TOKENS = 100
JUDGE_OUTPUT_TOKENS = 20
CANDIDATE_INPUT_TOKENS = 10
CANDIDATE_OUTPUT_TOKENS = 5

THREE_REPEATS = 3
TWO_CALLS_ISSUED = 2
TWO_JUDGES = 2
FOUR_CALLS = 4
MEDIAN_OF_2_4_5 = 4.0
DISPERSION_OF_2_4_5 = 3.0
AGREEMENT_OF_2_4_5 = 0.25
PANEL_MEDIAN = 3.0


def _judge_params(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build judge params with no criteria, so a scripted overall score is the score."""
    params: dict[str, JSONValue] = {
        "rubric": RUBRIC,
        "scale": dict(SCALE),
        "provider": dict(JUDGE_PROVIDER),
    }
    params.update(overrides)
    return params


def _verdict(score: float, *, rationale: str = "a judgment") -> str:
    return json.dumps({"overall_score": score, "rationale": rationale, "confidence": 0.7})


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class ScriptedJudge:
    """Returns canned judge responses in order, one per call."""

    name: ClassVar[str] = "fake"

    def __init__(self, config: ProviderConfig, outputs: Sequence[str]) -> None:
        self._config = config
        self._outputs = list(outputs)
        self.requests: list[ProviderRequest] = []

    @property
    def config(self) -> ProviderConfig:
        return self._config

    async def generate(self, request: ProviderRequest) -> ModelResponse:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._outputs) - 1)
        now = utc_now()
        return ModelResponse(
            output_text=self._outputs[index],
            provider=self._config.provider,
            model=self._config.model,
            requested_model=self._config.model,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                input_tokens=JUDGE_INPUT_TOKENS,
                output_tokens=JUDGE_OUTPUT_TOKENS,
            ),
            latency_ms=1.0,
            total_latency_ms=1.0,
            started_at=now,
            completed_at=now,
        )

    async def aclose(self) -> None:
        return


class FailingJudge:
    """Raises a provider error on every call."""

    name: ClassVar[str] = "fake"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self.calls = 0

    @property
    def config(self) -> ProviderConfig:
        return self._config

    async def generate(self, request: ProviderRequest) -> ModelResponse:
        del request
        self.calls += 1
        msg = "the judge endpoint is unavailable"
        raise ProviderError(msg, kind=ProviderErrorKind.SERVER, provider=self._config.provider)

    async def aclose(self) -> None:
        return


class PanelFactory:
    """Opens a different scripted judge per model, so a panel can disagree."""

    def __init__(self, judges: dict[str, Any]) -> None:
        self.judges = judges
        self.opened: list[str] = []

    def __call__(self, config: ProviderConfig) -> Any:
        self.opened.append(config.model)
        return self._open(self.judges[config.model])

    @asynccontextmanager
    async def _open(self, judge: Any) -> Any:
        yield judge


def _context(
    factory: Any,
    *,
    judge_concurrency: int = 4,
    deadline: float | None = None,
) -> EvaluationContext:
    logger: structlog.BoundLogger = structlog.get_logger("tests")
    return EvaluationContext(
        run_id="test-run",
        suite_hash="sha256:test",
        case_hash="sha256:case",
        provider_factory=factory,
        deadline=deadline,
        logger=logger,
        settings=EvaluatorSettings(judge_concurrency=judge_concurrency),
    )


def _case(spec: dict[str, Any]) -> ResolvedCase:
    suite = BenchmarkSuite.model_validate(
        {
            "schema_version": 1,
            "name": "judge",
            "version": "1",
            "cases": [
                {
                    "id": "c1",
                    "input": "What is the capital of France?",
                    "expected": "Paris",
                    "evaluators": [spec],
                }
            ],
        }
    )
    return resolve_suite(suite).cases[0]


def _response() -> ModelResponse:
    now = utc_now()
    return ModelResponse(
        output_text="Paris",
        provider=CANDIDATE_PROVIDER,
        model=CANDIDATE_MODEL,
        requested_model=CANDIDATE_MODEL,
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(
            input_tokens=CANDIDATE_INPUT_TOKENS,
            output_tokens=CANDIDATE_OUTPUT_TOKENS,
        ),
        latency_ms=1.0,
        total_latency_ms=1.0,
        started_at=now,
        completed_at=now,
    )


async def _run(
    params: dict[str, JSONValue],
    factory: Any,
    *,
    evaluator_id: str = "rubric",
    context: EvaluationContext | None = None,
) -> EvaluationResult:
    spec = {"type": "llm_judge", "id": evaluator_id, "params": params}
    evaluator = default_registry().create(EvaluatorSpec.model_validate(spec))
    return await evaluator.evaluate(_case(spec), _response(), context or _context(factory))


def _single(outputs: Sequence[str]) -> PanelFactory:
    judge = ScriptedJudge(ProviderConfig(**JUDGE_PROVIDER), outputs)
    return PanelFactory({"fake-judge-1": judge})


# ---------------------------------------------------------------------------
# score-only accounting (R-JUDGE-09)
# ---------------------------------------------------------------------------


async def test_without_a_pass_threshold_the_judge_scores_but_does_not_judge_pass() -> None:
    result = await _run(_judge_params(), _single([_verdict(4)]))
    assert result.status is EvaluationStatus.PASSED
    assert result.passed is None
    assert result.score is not None
    assert result.raw_score == pytest.approx(4.0)
    assert result.metadata["pass_threshold"] is None


async def test_with_a_pass_threshold_the_judge_returns_a_real_verdict() -> None:
    passing = await _run(_judge_params(pass_threshold=4.0), _single([_verdict(4)]))
    assert passing.status is EvaluationStatus.PASSED
    assert passing.passed is True

    failing = await _run(_judge_params(pass_threshold=4.0), _single([_verdict(3)]))
    assert failing.status is EvaluationStatus.FAILED
    assert failing.passed is False


def test_a_pass_threshold_outside_the_scale_is_refused_at_load_time() -> None:
    """0.8 on a 1-5 scale is the normalized/raw confusion, caught before any call."""
    with pytest.raises(EvaluatorConfigError, match="outside the declared scale"):
        default_registry().create(
            EvaluatorSpec.model_validate(
                {"type": "llm_judge", "params": _judge_params(pass_threshold=0.8)}
            )
        )


async def test_a_score_only_judgment_is_counted_and_excluded_from_the_pass_rate() -> None:
    result = await _run(_judge_params(), _single([_verdict(4)]))
    metrics = compute_metrics(
        run_id="test-run",
        n_cases=1,
        results=[_case_result(result)],
        error_policy="exclude",
        price_table_id="test",
        price_table_version="1",
    )
    assert metrics.n_score_only == 1
    assert metrics.n_pass_denominator == 0
    assert metrics.pass_rate is None

    per_evaluator = metrics.by_evaluator["rubric"]
    assert per_evaluator.n_score_only == 1
    assert per_evaluator.n_pass_denominator == 0
    assert per_evaluator.pass_rate is None
    assert per_evaluator.mean_score is not None


def _case_result(evaluation: EvaluationResult) -> CaseResult:
    now = utc_now()
    return CaseResult(
        run_id="test-run",
        case_id="c1",
        case_hash="sha256:case",
        status=CaseStatus.OK,
        response=_response(),
        evaluations=(evaluation,),
        passed=None,
        score=evaluation.score,
        cost=CostBreakdown(
            currency="USD",
            input_cost=None,
            output_cost=None,
            total_cost=None,
            priced=False,
            unpriced_reason="no_price_entry",
            price_table_id="test",
            price_table_version="1",
        ),
        started_at=now,
        completed_at=now,
    )


# ---------------------------------------------------------------------------
# repeats, dispersion and variance
# ---------------------------------------------------------------------------


async def test_three_repeats_take_the_median_and_report_the_spread() -> None:
    factory = _single([_verdict(2), _verdict(4), _verdict(5)])
    result = await _run(_judge_params(repetitions=THREE_REPEATS), factory)

    assert result.raw_score == pytest.approx(MEDIAN_OF_2_4_5)
    assert result.judge is not None
    assert result.judge.dispersion == pytest.approx(DISPERSION_OF_2_4_5)
    assert result.judge.agreement == pytest.approx(AGREEMENT_OF_2_4_5)
    assert result.judge.high_judge_variance is True
    assert len(result.judge.verdicts) == THREE_REPEATS
    assert len(result.judge.raw_outputs) == THREE_REPEATS
    assert result.metadata["raw_scores"] == [2.0, 4.0, 5.0]


async def test_agreeing_repeats_are_not_flagged_as_high_variance() -> None:
    factory = _single([_verdict(4), _verdict(4), _verdict(4)])
    result = await _run(_judge_params(repetitions=THREE_REPEATS), factory)
    assert result.judge is not None
    assert result.judge.dispersion == pytest.approx(0.0)
    assert result.judge.agreement == pytest.approx(1.0)
    assert result.judge.high_judge_variance is False


async def test_the_variance_threshold_is_configurable_on_the_raw_scale() -> None:
    factory = _single([_verdict(3), _verdict(4)])
    lenient = await _run(
        _judge_params(repetitions=2, high_variance_threshold=2.0),
        factory,
    )
    assert lenient.judge is not None
    assert lenient.judge.high_judge_variance is False

    strict = await _run(
        _judge_params(repetitions=2, high_variance_threshold=0.5),
        _single([_verdict(3), _verdict(4)]),
    )
    assert strict.judge is not None
    assert strict.judge.high_judge_variance is True


async def test_median_aggregation_survives_one_wild_judgment() -> None:
    """The reason the default is median: one outlier must not move the number."""
    steady = await _run(
        _judge_params(repetitions=THREE_REPEATS),
        _single([_verdict(4), _verdict(4), _verdict(1)]),
    )
    assert steady.raw_score == pytest.approx(4.0)

    averaged = await _run(
        _judge_params(repetitions=THREE_REPEATS, aggregation="mean"),
        _single([_verdict(4), _verdict(4), _verdict(1)]),
    )
    assert averaged.raw_score == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


async def test_a_panel_reports_the_median_of_the_judges_and_their_range() -> None:
    judges = {
        "judge-a": ScriptedJudge(ProviderConfig(provider="fake", model="judge-a"), [_verdict(2)]),
        "judge-b": ScriptedJudge(ProviderConfig(provider="fake", model="judge-b"), [_verdict(3)]),
        "judge-c": ScriptedJudge(ProviderConfig(provider="fake", model="judge-c"), [_verdict(5)]),
    }
    params = _judge_params(
        provider=None,
        judges=[
            {"provider": "fake", "model": "judge-a"},
            {"provider": "fake", "model": "judge-b"},
            {"provider": "fake", "model": "judge-c"},
        ],
    )
    result = await _run(params, PanelFactory(judges))

    assert result.raw_score == pytest.approx(PANEL_MEDIAN)
    assert result.metadata["judge_min"] == pytest.approx(2.0)
    assert result.metadata["judge_max"] == pytest.approx(5.0)
    assert result.metadata["judge_median_range"] == pytest.approx(3.0)
    assert result.judge is not None
    assert result.judge.judge_models == ("fake:judge-a", "fake:judge-b", "fake:judge-c")
    assert result.judge.high_judge_variance is True


# ---------------------------------------------------------------------------
# cost and token separation
# ---------------------------------------------------------------------------


async def test_judge_tokens_are_recorded_separately_from_the_candidate_response() -> None:
    result = await _run(_judge_params(repetitions=2), _single([_verdict(4), _verdict(4)]))
    assert result.judge is not None
    assert result.judge.usage.input_tokens == 2 * JUDGE_INPUT_TOKENS
    assert result.judge.usage.output_tokens == 2 * JUDGE_OUTPUT_TOKENS

    candidate = _response()
    assert candidate.usage.input_tokens == CANDIDATE_INPUT_TOKENS
    assert candidate.usage.output_tokens == CANDIDATE_OUTPUT_TOKENS


async def test_the_candidate_cost_breakdown_never_absorbs_judge_spend() -> None:
    """`total_cost` means candidate-model cost; judge spend is its own line item."""
    result = await _run(_judge_params(), _single([_verdict(4)]))
    breakdown = _case_result(result).cost
    assert breakdown is not None
    assert breakdown.judge_cost is None
    assert breakdown.total_cost is None
    assert result.judge is not None
    assert result.judge.usage.input_tokens == JUDGE_INPUT_TOKENS


# ---------------------------------------------------------------------------
# bias flags
# ---------------------------------------------------------------------------


async def test_a_judge_grading_its_own_model_is_flagged_but_not_blocked() -> None:
    same: dict[str, JSONValue] = {"provider": CANDIDATE_PROVIDER, "model": CANDIDATE_MODEL}
    judges = {
        CANDIDATE_MODEL: ScriptedJudge(ProviderConfig(**same), [_verdict(5)]),
    }
    result = await _run(_judge_params(provider=same), PanelFactory(judges))

    assert result.status is EvaluationStatus.PASSED
    assert result.judge is not None
    assert result.judge.self_preference_risk is True
    assert result.metadata["self_preference_risk"] is True


async def test_a_different_judge_model_is_not_flagged() -> None:
    result = await _run(_judge_params(), _single([_verdict(4)]))
    assert result.judge is not None
    assert result.judge.self_preference_risk is False


async def test_the_candidate_length_is_logged_beside_the_score() -> None:
    """Verbosity bias is not corrected for, so it has to be visible in the data."""
    result = await _run(_judge_params(), _single([_verdict(4)]))
    assert result.metadata["candidate_chars"] == len("Paris")
    assert result.metadata["candidate_truncated"] is False


async def test_judge_calls_default_to_temperature_zero() -> None:
    factory = _single([_verdict(4)])
    result = await _run(_judge_params(), factory)
    assert result.metadata["temperature"] == 0.0
    judge = factory.judges["fake-judge-1"]
    assert judge.requests[0].params.temperature == 0.0


# ---------------------------------------------------------------------------
# provenance and failure handling
# ---------------------------------------------------------------------------


async def test_the_provenance_records_the_prompt_the_judge_actually_saw() -> None:
    factory = _single([_verdict(4)])
    result = await _run(_judge_params(), factory)
    assert result.judge is not None
    assert len(result.judge.rendered_prompts) == 1
    rendered = result.judge.rendered_prompts[0]
    assert RUBRIC in rendered
    assert "Paris" in rendered
    assert result.judge.template_id == "single_answer_grading_v2"
    assert result.judge.rubric_hash.startswith("sha256:")


async def test_a_provider_failure_is_an_error_not_a_zero() -> None:
    judge = FailingJudge(ProviderConfig(**JUDGE_PROVIDER))
    result = await _run(_judge_params(), PanelFactory({"fake-judge-1": judge}))

    assert result.status is EvaluationStatus.ERROR
    assert result.passed is None
    assert result.score is None
    assert result.error is not None
    assert result.error.kind == "provider"
    assert result.metadata["error_code"] == "judge_provider_error"
    # One call, then one repair attempt is not made: there was no response to repair.
    assert judge.calls == 1


async def test_a_judge_with_no_provider_is_refused_at_load_time() -> None:
    with pytest.raises(EvaluatorConfigError, match="requires a judge model"):
        default_registry().create(
            EvaluatorSpec.model_validate({"type": "llm_judge", "params": {"rubric": RUBRIC}})
        )


def test_zero_total_criterion_weight_is_refused_at_load_time() -> None:
    with pytest.raises(EvaluatorConfigError, match="weights sum to zero"):
        default_registry().create(
            EvaluatorSpec.model_validate(
                {
                    "type": "llm_judge",
                    "params": _judge_params(
                        criteria=[
                            {"id": "a", "description": "a", "weight": 0.0},
                            {"id": "b", "description": "b", "weight": 0.0},
                        ]
                    ),
                }
            )
        )


def test_duplicate_criterion_ids_are_refused_at_load_time() -> None:
    with pytest.raises(EvaluatorConfigError, match="duplicate criterion"):
        default_registry().create(
            EvaluatorSpec.model_validate(
                {
                    "type": "llm_judge",
                    "params": _judge_params(
                        criteria=[
                            {"id": "a", "description": "a", "weight": 1.0},
                            {"id": "a", "description": "again", "weight": 1.0},
                        ]
                    ),
                }
            )
        )


def test_an_unknown_template_is_refused_at_load_time() -> None:
    with pytest.raises(EvaluatorConfigError, match="unknown judge template"):
        default_registry().create(
            EvaluatorSpec.model_validate(
                {"type": "llm_judge", "params": _judge_params(template_id="made_up_v9")}
            )
        )


def test_the_judge_is_advertised_as_model_graded() -> None:
    info = default_registry().info("llm_judge")
    assert info.is_model_graded is True
    assert info.needs_expected is False


# ---------------------------------------------------------------------------
# a deadline must not destroy the evidence for the calls that finished
# ---------------------------------------------------------------------------


class StallingJudge:
    """Answers the first call, then blocks forever on the second."""

    name: ClassVar[str] = "fake"

    def __init__(self, config: ProviderConfig, first: str) -> None:
        self._config = config
        self._first = first
        self.calls = 0

    @property
    def config(self) -> ProviderConfig:
        return self._config

    async def generate(self, request: ProviderRequest) -> ModelResponse:
        del request
        self.calls += 1
        if self.calls > 1:
            await asyncio.Event().wait()
        now = utc_now()
        return ModelResponse(
            output_text=self._first,
            provider=self._config.provider,
            model=self._config.model,
            requested_model=self._config.model,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                input_tokens=JUDGE_INPUT_TOKENS,
                output_tokens=JUDGE_OUTPUT_TOKENS,
            ),
            latency_ms=1.0,
            total_latency_ms=1.0,
            started_at=now,
            completed_at=now,
        )

    async def aclose(self) -> None:
        return


async def test_a_deadline_keeps_the_evidence_for_every_completed_judge_call() -> None:
    """A timeout must not take the calls that already finished down with it."""
    judge = StallingJudge(ProviderConfig(**JUDGE_PROVIDER), _verdict(4))
    factory = PanelFactory({"fake-judge-1": judge})
    deadline = asyncio.get_running_loop().time() + 0.2
    result = await _run(
        _judge_params(repetitions=THREE_REPEATS),
        factory,
        context=_context(factory, deadline=deadline),
    )

    assert result.status is EvaluationStatus.ERROR
    assert result.passed is None
    assert result.score is None
    assert result.error is not None
    assert result.error.kind == "timeout"
    assert result.metadata["error_code"] == "judge_timeout"

    assert judge.calls == TWO_CALLS_ISSUED
    provenance = result.judge
    assert provenance is not None
    assert len(provenance.raw_outputs) == 1
    assert "overall_score" in provenance.raw_outputs[0]
    assert len(provenance.rendered_prompts) == 1
    assert provenance.judge_models == ("fake:fake-judge-1",)
    assert provenance.usage.input_tokens == JUDGE_INPUT_TOKENS
    assert provenance.usage.output_tokens == JUDGE_OUTPUT_TOKENS
    assert result.metadata["n_calls"] == 1
    assert result.metadata["n_judgments"] == 1


async def test_a_deadline_that_has_not_expired_does_not_interfere() -> None:
    factory = _single([_verdict(4)])
    deadline = asyncio.get_running_loop().time() + 30.0
    result = await _run(_judge_params(), factory, context=_context(factory, deadline=deadline))
    assert result.status is EvaluationStatus.PASSED
    assert result.raw_score == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# judge_disagreement is a different question from high_judge_variance
# ---------------------------------------------------------------------------


def _panel(scores_by_model: dict[str, list[float]]) -> tuple[dict[str, JSONValue], PanelFactory]:
    """Build a panel whose judges return the given scores, one per repeat."""
    judges = {
        model: ScriptedJudge(
            ProviderConfig(provider="fake", model=model),
            [_verdict(score) for score in scores],
        )
        for model, scores in scores_by_model.items()
    }
    params = _judge_params(
        provider=None,
        judges=[{"provider": "fake", "model": model} for model in scores_by_model],
        repetitions=max(len(scores) for scores in scores_by_model.values()),
    )
    return params, PanelFactory(judges)


async def test_judges_that_disagree_set_judge_disagreement() -> None:
    params, factory = _panel({"judge-a": [2.0], "judge-b": [5.0]})
    result = await _run(params, factory)
    assert result.metadata["judge_disagreement"] is True
    assert result.metadata["n_judges"] == TWO_JUDGES
    assert result.metadata["judge_median_range"] == pytest.approx(3.0)


async def test_one_unstable_judge_sets_variance_but_not_disagreement() -> None:
    """A single judge cannot disagree with anyone, however much it wobbles."""
    result = await _run(
        _judge_params(repetitions=2),
        _single([_verdict(2), _verdict(5)]),
    )
    assert result.metadata["high_judge_variance"] is True
    assert result.metadata["judge_disagreement"] is False
    assert result.metadata["n_judges"] == 1


async def test_two_judges_that_wobble_identically_agree_with_each_other() -> None:
    """Both flags are computed, and they answer different questions."""
    params, factory = _panel({"judge-a": [2.0, 5.0], "judge-b": [2.0, 5.0]})
    result = await _run(params, factory)
    assert result.metadata["judge_median_range"] == pytest.approx(0.0)
    assert result.metadata["judge_disagreement"] is False
    assert result.metadata["high_judge_variance"] is True


async def test_agreeing_judges_set_neither_flag() -> None:
    params, factory = _panel({"judge-a": [4.0], "judge-b": [4.0]})
    result = await _run(params, factory)
    assert result.metadata["judge_disagreement"] is False
    assert result.metadata["high_judge_variance"] is False


# ---------------------------------------------------------------------------
# renormalised criterion weights are announced, not applied in silence
# ---------------------------------------------------------------------------

WEIGHTED_CRITERIA: list[JSONValue] = [
    {"id": "correctness", "description": "Is it right?", "weight": 0.6},
    {"id": "completeness", "description": "Is it whole?", "weight": 0.5},
]
NORMALISED_CRITERIA: list[JSONValue] = [
    {"id": "correctness", "description": "Is it right?", "weight": 0.6},
    {"id": "completeness", "description": "Is it whole?", "weight": 0.4},
]
WEIGHT_SUM = 1.1


def _criteria_verdict(correctness: int, completeness: int, overall: float) -> str:
    return json.dumps(
        {
            "criteria": [
                {"name": "correctness", "score": correctness},
                {"name": "completeness", "score": completeness},
            ],
            "overall_score": overall,
            "rationale": "a judgment",
        }
    )


async def test_weights_that_do_not_sum_to_one_are_warned_about_and_recorded() -> None:
    factory = _single([_criteria_verdict(4, 5, 4.5)])
    with capture_logs() as logs:
        result = await _run(_judge_params(criteria=WEIGHTED_CRITERIA), factory)

    assert result.metadata["weights_renormalised"] is True
    assert result.metadata["criterion_weight_sum"] == pytest.approx(WEIGHT_SUM)

    warnings = [entry for entry in logs if entry["log_level"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["event"] == "judge_criterion_weights_renormalised"
    assert warnings[0]["weight_sum"] == pytest.approx(WEIGHT_SUM)


async def test_weights_that_already_sum_to_one_are_not_warned_about() -> None:
    factory = _single([_criteria_verdict(4, 5, 4.4)])
    with capture_logs() as logs:
        result = await _run(_judge_params(criteria=NORMALISED_CRITERIA), factory)

    assert result.metadata["weights_renormalised"] is False
    assert result.metadata["criterion_weight_sum"] is None
    assert [entry for entry in logs if entry["log_level"] == "warning"] == []


async def test_renormalised_weights_still_produce_the_rescaled_score() -> None:
    """0.6 and 0.5 rescale to 0.545 and 0.455, so 4 and 5 give 4.4545..."""
    factory = _single([_criteria_verdict(4, 5, 4.5)])
    result = await _run(_judge_params(criteria=WEIGHTED_CRITERIA), factory)
    assert result.raw_score == pytest.approx((0.6 * 4 + 0.5 * 5) / WEIGHT_SUM)


# ---------------------------------------------------------------------------
# per-call evidence: latency, repeat index, judge identity, generation params
# ---------------------------------------------------------------------------


async def test_every_judge_call_records_its_latency_repeat_and_model() -> None:
    params, factory = _panel({"judge-a": [4.0, 4.0], "judge-b": [4.0, 4.0]})
    result = await _run(params, factory)

    assert result.metadata["judge_call_repeats"] == [0, 1, 0, 1]
    assert result.metadata["judge_call_models"] == [
        "fake:judge-a",
        "fake:judge-a",
        "fake:judge-b",
        "fake:judge-b",
    ]
    latencies = result.metadata["judge_call_latency_ms"]
    assert isinstance(latencies, list)
    assert len(latencies) == FOUR_CALLS
    assert all(isinstance(value, float) and value >= 0.0 for value in latencies)

    provenance = result.judge
    assert provenance is not None
    # Index-aligned with the raw outputs, so a verdict can be attributed.
    assert len(provenance.raw_outputs) == len(latencies)


async def test_the_full_judge_generation_params_are_recorded() -> None:
    result = await _run(_judge_params(), _single([_verdict(4)]))
    params = result.metadata["judge_params"]
    assert isinstance(params, dict)
    assert params["temperature"] == 0.0
    assert params["response_format"] == "json_object"
    # Present as explicit nulls: "not set" and "not recorded" are different facts.
    assert params["max_output_tokens"] is None
    assert params["seed"] is None


async def test_the_per_call_evidence_survives_a_failed_judgment(fixtures_dir: Any) -> None:
    malformed = (fixtures_dir / "judge_outputs" / "malformed.txt").read_text(encoding="utf-8")
    result = await _run(_judge_params(), _single([malformed]))

    assert result.status is EvaluationStatus.ERROR
    assert result.metadata["judge_call_repeats"] == [0]
    assert result.metadata["judge_call_models"] == ["fake:fake-judge-1"]
    assert isinstance(result.metadata["judge_call_latency_ms"], list)
    assert isinstance(result.metadata["judge_params"], dict)
    assert result.metadata["candidate_chars"] == len("Paris")


# ---------------------------------------------------------------------------
# the terminal cause of a failure is the one reported
# ---------------------------------------------------------------------------


class RepairFailsJudge:
    """Returns unusable prose, then fails the repair attempt with a provider error."""

    name: ClassVar[str] = "fake"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self.calls = 0

    @property
    def config(self) -> ProviderConfig:
        return self._config

    async def generate(self, request: ProviderRequest) -> ModelResponse:
        del request
        self.calls += 1
        if self.calls > 1:
            msg = "the judge endpoint went away mid-repair"
            raise ProviderError(msg, kind=ProviderErrorKind.SERVER, provider=self._config.provider)
        now = utc_now()
        return ModelResponse(
            output_text="Here is my assessment in prose, which is not JSON.",
            provider=self._config.provider,
            model=self._config.model,
            requested_model=self._config.model,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(input_tokens=JUDGE_INPUT_TOKENS, output_tokens=JUDGE_OUTPUT_TOKENS),
            latency_ms=1.0,
            total_latency_ms=1.0,
            started_at=now,
            completed_at=now,
        )

    async def aclose(self) -> None:
        return


async def test_a_repair_killed_by_an_outage_is_reported_as_a_provider_failure() -> None:
    """Labelling an outage a parsing problem sends the operator to the wrong place."""
    judge = RepairFailsJudge(ProviderConfig(**JUDGE_PROVIDER))
    result = await _run(_judge_params(), PanelFactory({"fake-judge-1": judge}))

    assert judge.calls == TWO_CALLS_ISSUED
    assert result.status is EvaluationStatus.ERROR
    assert result.error is not None
    assert result.error.kind == "provider"
    assert result.metadata["error_code"] == "judge_provider_error"
    assert "went away mid-repair" in result.error.message


# ---------------------------------------------------------------------------
# persisted judge text is capped and scrubbed
# ---------------------------------------------------------------------------

SMALL_OUTPUT_CAP = 200
LEAKY_URL = "postgresql://admin:hunter2@db.internal/eval"
SIGNED_URL = "https://api.example.com/v1/grade?api_key=sk-live-abcdef123456&model=x"


def test_persistable_text_redacts_credentials_hiding_in_urls() -> None:
    from llm_eval_lab.evaluators.judge.evaluator import persistable_text  # noqa: PLC0415

    cleaned = persistable_text(f"see {LEAKY_URL} and {SIGNED_URL} for details", max_chars=8000)
    assert "hunter2" not in cleaned
    assert "sk-live-abcdef123456" not in cleaned
    assert "[redacted]" in cleaned
    # Everything that is not the secret survives, or the record is useless.
    assert "db.internal" in cleaned
    assert "api.example.com" in cleaned


def test_persistable_text_caps_and_says_that_it_capped() -> None:
    from llm_eval_lab.evaluators.judge.evaluator import persistable_text  # noqa: PLC0415

    capped = persistable_text("x" * 5000, max_chars=SMALL_OUTPUT_CAP)
    assert capped.startswith("x" * SMALL_OUTPUT_CAP)
    assert "[truncated: 5000 characters]" in capped
    assert len(capped) < 5000


def test_short_text_passes_through_unchanged() -> None:
    from llm_eval_lab.evaluators.judge.evaluator import persistable_text  # noqa: PLC0415

    assert persistable_text("a plain verdict", max_chars=8000) == "a plain verdict"


async def test_a_long_judge_response_is_capped_before_it_reaches_provenance() -> None:
    """`raw_outputs` grows with repetitions times judges and rode in unbounded."""
    padded = json.dumps({"overall_score": 4, "rationale": "y" * 5000, "confidence": 0.7})
    factory = _single([padded])
    context = _context(factory)
    context = EvaluationContext(
        run_id=context.run_id,
        suite_hash=context.suite_hash,
        case_hash=context.case_hash,
        provider_factory=factory,
        deadline=None,
        logger=context.logger,
        settings=EvaluatorSettings(max_output_chars=SMALL_OUTPUT_CAP),
    )
    result = await _run(_judge_params(), factory, context=context)

    assert result.status is EvaluationStatus.PASSED
    assert result.raw_score == pytest.approx(4.0)
    provenance = result.judge
    assert provenance is not None
    assert "[truncated:" in provenance.raw_outputs[0]
    assert len(provenance.raw_outputs[0]) < len(padded)
    assert "[truncated:" in provenance.rendered_prompts[0]


async def test_a_credential_in_a_judge_response_is_scrubbed_before_persistence() -> None:
    leaky = json.dumps(
        {
            "overall_score": 4,
            "rationale": f"the answer cited {LEAKY_URL} which is wrong",
            "confidence": 0.7,
        }
    )
    result = await _run(_judge_params(), _single([leaky]))

    provenance = result.judge
    assert provenance is not None
    assert "hunter2" not in provenance.raw_outputs[0]
    assert "db.internal" in provenance.raw_outputs[0]
    # The parsed verdict keeps the reasoning text, which the judge wrote and the
    # scrubber does not touch: only what is PERSISTED as raw text is scrubbed.
    assert result.explanation is not None


async def test_the_judge_still_sees_the_untouched_response_for_its_repair_turn(
    fixtures_dir: Any,
) -> None:
    """Capping must happen at the persistence boundary, not in working memory."""
    malformed = (fixtures_dir / "judge_outputs" / "malformed.txt").read_text(encoding="utf-8")
    valid = (fixtures_dir / "judge_outputs" / "valid.json").read_text(encoding="utf-8")
    factory = _single([malformed, valid])
    context = EvaluationContext(
        run_id="test-run",
        suite_hash="sha256:test",
        case_hash="sha256:case",
        provider_factory=factory,
        deadline=None,
        logger=structlog.get_logger("tests"),
        settings=EvaluatorSettings(max_output_chars=SMALL_OUTPUT_CAP),
    )
    result = await _run(_judge_params(criteria=NORMALISED_CRITERIA), factory, context=context)

    judge = factory.judges["fake-judge-1"]
    assert len(judge.requests) == TWO_CALLS_ISSUED
    # The repair turn echoed the real previous answer, not a truncated one.
    echoed = judge.requests[1].messages[1].content
    assert "roughly 4.4 overall" in echoed
    assert result.status is EvaluationStatus.PASSED


# ---------------------------------------------------------------------------
# redaction has to survive every kind of whitespace, not just the space
# ---------------------------------------------------------------------------

SECRET = "hunter2"  # noqa: S105 - a fixture value, not a credential
QUERY_SECRET = "sk-live-abcdef123456"  # noqa: S105 - a fixture value, not a credential


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("", ""),
        (" ", " "),
        ("\n", "\n"),
        ("\n\n", "\n\n"),
        ("\t", "\t"),
        ("\r\n", "\r\n"),
        ("   ", "   "),
        ("line one\n", "\nline three"),
        ("| ", " |"),
    ],
    ids=[
        "start-and-end-of-string",
        "spaces",
        "newlines",
        "blank-line",
        "tabs",
        "crlf",
        "run-of-spaces",
        "multi-line-document",
        "table-cell",
    ],
)
def test_a_url_is_redacted_whatever_whitespace_surrounds_it(before: str, after: str) -> None:
    """A rendered judge prompt is a multi-line document, so this is the common case.

    Splitting on the space character alone left a URL welded to its newlines,
    so `looks_like_url` saw one token that was not a URL and the credential was
    persisted verbatim.
    """
    from llm_eval_lab.evaluators.judge.evaluator import persistable_text  # noqa: PLC0415

    text = f"{before}postgresql://admin:{SECRET}@db.internal/eval{after}"
    cleaned = persistable_text(text, max_chars=8000)
    assert SECRET not in cleaned
    assert "[redacted]" in cleaned
    assert "db.internal" in cleaned
    # The surrounding layout is reproduced exactly.
    assert cleaned.startswith(before)
    assert cleaned.endswith(after)


def test_a_signed_url_on_its_own_line_is_redacted() -> None:
    from llm_eval_lab.evaluators.judge.evaluator import persistable_text  # noqa: PLC0415

    text = (
        "# The candidate answer to grade\n\n"
        f"\thttps://api.example.com/grade?api_key={QUERY_SECRET}&model=x\n\n"
        "Grade the text between the candidate markers."
    )
    cleaned = persistable_text(text, max_chars=8000)
    assert QUERY_SECRET not in cleaned
    assert "api.example.com" in cleaned
    assert cleaned.count("\n") == text.count("\n")
    assert "\t" in cleaned


def test_text_with_nothing_to_redact_is_returned_byte_identical() -> None:
    """The split has to be reversible, or every stored prompt is subtly reflowed."""
    from llm_eval_lab.evaluators.judge.evaluator import persistable_text  # noqa: PLC0415

    text = "line one\n\n\tindented\r\n  trailing spaces   \nend"
    assert persistable_text(text, max_chars=8000) == text


async def test_a_credential_on_its_own_line_of_a_judge_prompt_is_not_persisted() -> None:
    """End to end: the rendered prompt is multi-line and reaches the database."""
    leaky = json.dumps(
        {
            "overall_score": 4,
            "rationale": (
                f"the candidate cited\npostgresql://admin:{SECRET}@db.internal/eval\nwhich is wrong"
            ),
            "confidence": 0.7,
        }
    )
    result = await _run(_judge_params(), _single([leaky]))
    provenance = result.judge
    assert provenance is not None
    assert SECRET not in provenance.raw_outputs[0]
    assert "db.internal" in provenance.raw_outputs[0]


# ---------------------------------------------------------------------------
# the renormalisation warning is about the configuration, not about a response
# ---------------------------------------------------------------------------


async def test_the_weight_warning_fires_once_per_evaluator_not_once_per_case() -> None:
    """A thousand-case suite must not put the same line in the log a thousand times."""
    factory = _single([_criteria_verdict(4, 5, 4.5)])
    spec = {
        "type": "llm_judge",
        "id": "rubric",
        "params": _judge_params(criteria=WEIGHTED_CRITERIA),
    }
    evaluator = default_registry().create(EvaluatorSpec.model_validate(spec))
    context = _context(factory)

    with capture_logs() as logs:
        results = [await evaluator.evaluate(_case(spec), _response(), context) for _ in range(3)]

    warnings = [entry for entry in logs if entry["log_level"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["event"] == "judge_criterion_weights_renormalised"

    # The per-case RECORD stays per case: a log line is gone by the time anyone
    # reads the run, and the stored data is not.
    assert len(results) == THREE_REPEATS
    for result in results:
        assert result.metadata["weights_renormalised"] is True
        assert result.metadata["criterion_weight_sum"] == pytest.approx(WEIGHT_SUM)


@pytest.mark.parametrize("escape", ["\\n", "\\t", "\\r"])
def test_a_url_beside_an_escaped_newline_is_redacted(escape: str) -> None:
    """`raw_outputs` holds the judge's JSON verbatim, where a line break is two characters.

    Without covering the escaped spelling, a connection string on its own line of
    a judge's reasoning stayed welded to `cited\\npostgresql://...`, which has no
    valid scheme, so `redact_url` found nothing and the credential was persisted.
    """
    from llm_eval_lab.evaluators.judge.evaluator import persistable_text  # noqa: PLC0415

    text = f"cited{escape}postgresql://admin:{SECRET}@db.internal/eval{escape}which is wrong"
    cleaned = persistable_text(text, max_chars=8000)
    assert SECRET not in cleaned
    assert "[redacted]" in cleaned
    assert "db.internal" in cleaned
    assert cleaned.startswith(f"cited{escape}")
    assert cleaned.endswith(f"{escape}which is wrong")


def test_escaped_whitespace_with_nothing_to_redact_is_returned_byte_identical() -> None:
    from llm_eval_lab.evaluators.judge.evaluator import persistable_text  # noqa: PLC0415

    text = "a JSON rationale\\nwith escaped\\tbreaks\\rand a literal backslash \\ in it"
    assert persistable_text(text, max_chars=8000) == text
