"""The public contract surface, checked in both directions.

`EXPECTED_SURFACE` is the literal list of every name the implementation plan
promises to later phases. Checking it forwards turns "every symbol phases 2
through 8 reference exists and is importable" from a reading exercise into a
command. Checking it backwards is what stops the contract growing quietly: a
name added to `llm_eval_lab.models.__all__` without a corresponding plan entry
fails here, which is the signal to go through the frozen-file protocol rather
than around it.
"""

from __future__ import annotations

import pytest

import llm_eval_lab.models as contracts

EXPECTED_SURFACE = (
    "AggregateMetrics",
    "BenchmarkCase",
    "BenchmarkRepository",
    "BenchmarkSnapshot",
    "BenchmarkSuite",
    "BenchmarkValidationError",
    "CASE_ID_PATTERN",
    "CaseDelta",
    "CaseQuery",
    "CaseResult",
    "CaseResultRepository",
    "CaseSelection",
    "CaseStatus",
    "CategoryMetrics",
    "ChatMessage",
    "CheckOutcome",
    "CheckStatus",
    "CostBreakdown",
    "CredentialResolver",
    "EmbeddingProvider",
    "EvaluationContext",
    "EvaluationErrorInfo",
    "EvaluationRecord",
    "EvaluationRepository",
    "EvaluationResult",
    "EvaluationStatus",
    "Evaluator",
    "EvaluatorConfigError",
    "EvaluatorError",
    "EvaluatorMetrics",
    "EvaluatorSettings",
    "EvaluatorSpec",
    "FieldError",
    "FinishReason",
    "GenerationParams",
    "JSONValue",
    "JudgeConfig",
    "JudgeCriterion",
    "JudgeProvenance",
    "JudgeScale",
    "JudgeVerdict",
    "LLMEvalError",
    "LatencyStats",
    "MetricCheck",
    "MetricsRepository",
    "MissingCredentialError",
    "ModelResponse",
    "Page",
    "PluginRecord",
    "PriceEntry",
    "PriceTable",
    "PricingError",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressEventType",
    "Provider",
    "ProviderAuthError",
    "ProviderConfig",
    "ProviderError",
    "ProviderErrorInfo",
    "ProviderErrorKind",
    "ProviderInvalidRequestError",
    "ProviderNotInstalledError",
    "ProviderRateLimitError",
    "ProviderRequest",
    "ProviderServerError",
    "ProviderTimeoutError",
    "RETRYABLE_KINDS",
    "RegressionInputError",
    "RegressionReport",
    "RegressionSummary",
    "RegressionThresholds",
    "ResolvedCase",
    "ResolvedSuite",
    "ResponseRepository",
    "RetryPolicy",
    "Role",
    "Run",
    "RunConfig",
    "RunQuery",
    "RunRepository",
    "RunStatus",
    "RunSummary",
    "RunTotals",
    "StorageError",
    "SuiteDefaults",
    "ThresholdDirection",
    "TokenTotals",
    "TokenUsage",
    "UnitOfWork",
    "UnitOfWorkFactory",
    "Verdict",
)


@pytest.mark.parametrize("name", EXPECTED_SURFACE)
def test_expected_contract_name_is_importable(name: str) -> None:
    # `hasattr`, not `getattr(...) is not None`: a name legitimately bound to
    # None would otherwise be reported as missing.
    assert hasattr(contracts, name), (
        f"llm_eval_lab.models does not export {name!r}, which later phases depend on"
    )


def test_all_matches_the_expected_surface_exactly() -> None:
    """Set equality, not containment, in both directions.

    Containment alone would miss a name silently DROPPED from `__all__`: the
    module-level `from ... import` still binds it, so the forward `hasattr`
    check keeps passing while the declared public surface has quietly shrunk.
    """
    declared = set(contracts.__all__)
    expected = set(EXPECTED_SURFACE)
    assert declared == expected, (
        f"missing from __all__: {sorted(expected - declared)}; "
        f"exported without a plan entry: {sorted(declared - expected)}. "
        "Changing the frozen contract goes through the frozen-file protocol."
    )


def test_the_expected_surface_has_no_duplicates() -> None:
    duplicates = sorted({n for n in EXPECTED_SURFACE if EXPECTED_SURFACE.count(n) > 1})
    assert duplicates == []
