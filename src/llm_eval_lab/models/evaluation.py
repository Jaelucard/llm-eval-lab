"""Evaluation and LLM-judge contracts.

The distinction this module exists to preserve: a FAILED evaluation means the
model's answer was wrong, an ERROR evaluation means the evaluator itself could
not reach a conclusion. Collapsing the two makes a broken judge look like a
failing model. :class:`JudgeProvenance` records everything needed to audit or
reproduce a judgment, including the advisory flags that mark a judgment as
worth a second look.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_eval_lab.models.common import GenerationParams, JSONValue, ProviderConfig
from llm_eval_lab.models.enums import EvaluationStatus
from llm_eval_lab.models.response import ProviderErrorInfo, TokenUsage


class EvaluationErrorInfo(BaseModel):
    """Why an evaluator could not produce a verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["config", "runtime", "provider", "parse", "timeout"]
    message: str
    exception_type: str | None = None
    provider_error: ProviderErrorInfo | None = None


class JudgeScale(BaseModel):
    """The raw scale a judge scores on, and how it maps onto 0.0..1.0."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["integer", "continuous", "binary"] = "integer"
    minimum: float = 1.0
    maximum: float = 5.0
    labels: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _scale_has_positive_span(self) -> "JudgeScale":
        """Refuse a degenerate or inverted scale.

        With `minimum >= maximum` there is no meaningful mapping onto 0..1, and
        the alternative to refusing it is returning a silent `0.0` for every
        judgment - the exact class of invented number this project exists to
        avoid. Caught at configuration time, where a human can fix it.
        """
        if self.maximum <= self.minimum:
            msg = (
                f"JudgeScale requires maximum > minimum, got minimum={self.minimum} "
                f"and maximum={self.maximum}"
            )
            raise ValueError(msg)
        return self

    def normalize(self, raw: float) -> float:
        """Map a raw judge score onto 0.0..1.0, clamped to the scale's ends.

        The validator guarantees a positive span, so this cannot divide by zero.
        A raw score outside the scale is clamped rather than refused: judges do
        occasionally return an out-of-range number, and clamping keeps that a
        recoverable parse rather than a lost evaluation.
        """
        span = self.maximum - self.minimum
        return min(1.0, max(0.0, (raw - self.minimum) / span))


class JudgeCriterion(BaseModel):
    """One named dimension a judge scores separately."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    description: str
    weight: float = Field(default=1.0, ge=0.0)


class JudgeConfig(BaseModel):
    """How to run a model-graded evaluation.

    There is no default rubric: ``rubric`` is required, because a judge run
    against an unstated standard produces numbers nobody can interpret.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderConfig | None = None
    judges: tuple[ProviderConfig, ...] = ()
    template_id: str = "single_answer_grading_v2"
    rubric: str
    criteria: tuple[JudgeCriterion, ...] = ()
    scale: JudgeScale = JudgeScale()
    use_reference: bool = True
    reference: str | None = None
    pass_threshold: float | None = None
    repetitions: int = Field(default=1, ge=1, le=9)
    aggregation: Literal["mean", "median", "majority", "min"] = "median"
    params: GenerationParams = GenerationParams(temperature=0.0, response_format="json_object")
    max_output_chars: int = Field(default=8000, ge=1)
    high_variance_threshold: float = 1.0
    randomize_criteria_order: bool = False
    swap_positions: bool = False


class JudgeVerdict(BaseModel):
    """The structured output the judge model is required to produce."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float
    per_criterion: dict[str, float] = Field(default_factory=dict)
    reasoning: str
    violations: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class JudgeProvenance(BaseModel):
    """Everything needed to audit or reproduce a judgment.

    ``self_preference_risk`` and ``high_judge_variance`` are advisory: they are
    always visible and never silently suppress or gate a result.
    """

    model_config = ConfigDict(frozen=True)

    template_id: str
    rubric_hash: str
    scale: JudgeScale
    aggregation: str
    verdicts: tuple[JudgeVerdict, ...]
    judge_models: tuple[str, ...]
    raw_outputs: tuple[str, ...]
    rendered_prompts: tuple[str, ...]
    usage: TokenUsage
    cost_usd: Decimal | None
    dispersion: float | None
    agreement: float | None
    parse_failures: int = 0
    self_preference_risk: bool = False
    high_judge_variance: bool = False


class EvaluationResult(BaseModel):
    """The outcome of applying one evaluator to one model response.

    The rule, because three lanes need the same one: ``status`` reports whether
    the **evaluator** reached a conclusion, ``passed`` reports whether the
    **case** met the bar. A score-only evaluator - a judge with no
    ``pass_threshold`` - returns ``status=PASSED`` with ``passed=None`` and a
    populated ``score``. That pairing, and only that pairing, is what
    ``n_score_only`` counts. Pass/fail counting keys off ``passed``, never off
    ``status``, so a score-only judgment is never silently read as a failure.
    """

    model_config = ConfigDict(frozen=True)

    evaluator_id: str
    evaluator_type: str
    status: EvaluationStatus
    passed: bool | None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_score: float | None = None
    explanation: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    error: EvaluationErrorInfo | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    judge: JudgeProvenance | None = None
    weight: float = 1.0
    required: bool = True

    @model_validator(mode="after")
    def _status_matches_passed(self) -> "EvaluationResult":
        """Refuse a `status` and `passed` pairing that contradicts itself."""
        if self.status is EvaluationStatus.ERROR:
            if self.passed is not None or self.error is None:
                msg = "status=ERROR requires passed=None and a populated error"
                raise ValueError(msg)
        elif self.status is EvaluationStatus.SKIPPED:
            if self.passed is not None:
                msg = "status=SKIPPED requires passed=None"
                raise ValueError(msg)
        elif self.status is EvaluationStatus.FAILED:
            if self.passed is not False:
                msg = "status=FAILED requires passed=False"
                raise ValueError(msg)
        elif self.passed is False:
            msg = "status=PASSED requires passed=True, or passed=None for a score-only evaluator"
            raise ValueError(msg)
        return self
