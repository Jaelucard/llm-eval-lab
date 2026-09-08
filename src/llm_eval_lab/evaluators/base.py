"""The evaluator base class and the shared result-construction helpers.

The one rule every evaluator in this package obeys, restated because it is the
distinction the whole reporting layer rests on: ``status`` reports whether the
EVALUATOR reached a conclusion, ``passed`` reports whether the CASE met the
bar. A model that answered wrongly is ``status=FAILED, passed=False``. An
evaluator that could not run at all is ``status=ERROR, passed=None``. Collapsing
the two makes a broken evaluator look like a failing model, which is the exact
dishonesty this project exists to avoid.

Consequently :meth:`BaseEvaluator.evaluate` never raises for an ordinary
scoring failure. It returns an ERROR result instead, and the runner records it
and moves on.
"""

import builtins
from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from llm_eval_lab.models import (
    EvaluationContext,
    EvaluationErrorInfo,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorSpec,
    JSONValue,
    ModelResponse,
    ResolvedCase,
)


class NoParams(BaseModel):
    """Parameter model for an evaluator that takes no parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class BaseEvaluator[ParamsT: BaseModel](ABC):
    """Common construction, result shaping and error shaping for evaluators.

    Generic in its parameter model so ``self.params`` is precisely typed in a
    subclass without a cast at every use site.
    """

    type: ClassVar[str]
    # `builtins.type` because the `type` ClassVar above shadows the builtin
    # inside this class body. Matches the frozen `Evaluator` protocol.
    params_model: ClassVar[builtins.type[BaseModel]]
    needs_expected: ClassVar[bool] = False
    is_model_graded: ClassVar[bool] = False

    def __init__(self, spec: EvaluatorSpec, params: ParamsT) -> None:
        """Bind an already-validated parameter object to its authoring spec.

        Parameters are validated by the registry at LOAD time, not here, so an
        unusable evaluator configuration is a benchmark validation error the
        author sees before a single request is issued.
        """
        self.spec = spec
        self.params = params

    @property
    def evaluator_id(self) -> str:
        """Return the id this evaluator reports results under."""
        return self.spec.id or self.type

    @abstractmethod
    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Score one response. Must not raise except `asyncio.CancelledError`."""

    def result(  # noqa: PLR0913 - one parameter per field of the result being built
        self,
        *,
        status: EvaluationStatus,
        passed: bool | None,
        score: float | None = None,
        raw_score: float | None = None,
        explanation: str | None = None,
        metadata: dict[str, JSONValue] | None = None,
        duration_ms: float = 0.0,
    ) -> EvaluationResult:
        """Build a scoring result carrying this evaluator's identity and weight."""
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            evaluator_type=self.type,
            status=status,
            passed=passed,
            score=score,
            raw_score=raw_score,
            explanation=explanation,
            metadata=metadata or {},
            error=None,
            duration_ms=duration_ms,
            judge=None,
            weight=self.spec.weight,
            required=self.spec.required,
        )

    def verdict(  # noqa: PLR0913 - one parameter per field of the result being built
        self,
        *,
        passed: bool,
        score: float | None = None,
        raw_score: float | None = None,
        explanation: str | None = None,
        metadata: dict[str, JSONValue] | None = None,
        duration_ms: float = 0.0,
    ) -> EvaluationResult:
        """Build a pass/fail result, deriving `status` from `passed`.

        One place decides that pairing, so a subclass cannot accidentally emit
        the ``status=PASSED, passed=False`` shape the contract refuses.
        """
        return self.result(
            status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
            passed=passed,
            score=(1.0 if passed else 0.0) if score is None else score,
            raw_score=raw_score,
            explanation=explanation,
            metadata=metadata,
            duration_ms=duration_ms,
        )

    def error(  # noqa: PLR0913 - one parameter per field of the error result being built
        self,
        *,
        kind: str,
        message: str,
        error_code: str,
        exception_type: str | None = None,
        metadata: dict[str, JSONValue] | None = None,
        duration_ms: float = 0.0,
    ) -> EvaluationResult:
        """Build an ERROR result: the evaluator could not reach a conclusion."""
        info = EvaluationErrorInfo.model_validate(
            {"kind": kind, "message": message, "exception_type": exception_type}
        )
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            evaluator_type=self.type,
            status=EvaluationStatus.ERROR,
            passed=None,
            score=None,
            raw_score=None,
            explanation=None,
            metadata={**(metadata or {}), "error_code": error_code},
            error=info,
            duration_ms=duration_ms,
            judge=None,
            weight=self.spec.weight,
            required=self.spec.required,
        )


def apply_on_error(result: EvaluationResult, spec: EvaluatorSpec) -> EvaluationResult:
    """Reshape an ERROR result according to the spec's `on_error` policy.

    ``error`` keeps it as an error, which is the honest default. ``fail`` turns
    it into a failing case, for a suite that treats an unusable evaluator as a
    failed answer. ``skip`` drops it out of pass/fail accounting entirely. The
    original error is preserved in ``metadata`` in both non-default cases, so
    the reshaping is always visible rather than silent.
    """
    if result.status is not EvaluationStatus.ERROR or spec.on_error == "error":
        return result

    detail: dict[str, JSONValue] = {
        "on_error": spec.on_error,
        "original_status": EvaluationStatus.ERROR.value,
        "original_error": result.error.message if result.error else None,
    }
    metadata: dict[str, JSONValue] = {**result.metadata, **detail}
    if spec.on_error == "fail":
        return result.model_copy(
            update={
                "status": EvaluationStatus.FAILED,
                "passed": False,
                "score": 0.0,
                "error": None,
                "metadata": metadata,
            }
        )
    return result.model_copy(
        update={
            "status": EvaluationStatus.SKIPPED,
            "passed": None,
            "score": None,
            "error": None,
            "metadata": metadata,
        }
    )
