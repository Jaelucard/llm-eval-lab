"""Run configuration, per-case results, aggregate metrics and progress events.

:class:`RunConfig` is persisted verbatim so a run is reproducible: it records
the resolved suite digest, the exact provider configuration (credential names
only, never values), the price table identity and the library and interpreter
versions. Aggregate metrics are deliberately honest about their own gaps -
coverage ratios, ``n_score_only`` and ``LatencyStats.low_confidence`` all exist
so a number is never presented without the sample size behind it.
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_eval_lab.models.benchmark import EvaluatorSpec
from llm_eval_lab.models.common import GenerationParams, ProviderConfig, UtcDatetime
from llm_eval_lab.models.enums import CaseStatus, ProgressEventType, RunStatus
from llm_eval_lab.models.evaluation import EvaluationResult
from llm_eval_lab.models.pricing import CostBreakdown
from llm_eval_lab.models.response import ModelResponse, ProviderErrorInfo


class CaseSelection(BaseModel):
    """Exactly which cases the run executed, and how they were chosen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tags: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    limit: int | None = None
    sample: int | None = None
    sample_seed: int | None = None
    n_selected: int


class PluginRecord(BaseModel):
    """Provenance of one opt-in evaluator plugin that was loaded for this run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_point: str
    distribution: str
    version: str
    evaluator_types: tuple[str, ...]


class RunTotals(BaseModel):
    """Running counters maintained as cases complete."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_cases: int = 0
    n_completed: int = 0
    n_passed: int = 0
    n_errors: int = 0
    n_cancelled: int = 0


class RetryPolicy(BaseModel):
    """Bounded exponential backoff with jitter, owned by the runner.

    Providers perform exactly one attempt per call (decision D4), so this
    policy is the single place retry behaviour is described.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=4, ge=1, le=11)
    base_delay_s: float = Field(default=0.5, gt=0.0)
    max_delay_s: float = Field(default=30.0, gt=0.0)
    jitter: Literal["full", "equal", "none"] = "full"
    respect_retry_after: bool = True
    max_total_delay_s: float = 120.0


class RunConfig(BaseModel):
    """The effective configuration of a run, persisted verbatim.

    Contains no secrets: :class:`~llm_eval_lab.models.common.ProviderConfig`
    records the NAME of the credential environment variable, never its value.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    suite_name: str
    suite_version: str
    suite_hash: str
    suite_source: str | None
    case_selection: CaseSelection
    provider: ProviderConfig
    params: GenerationParams
    evaluators: tuple[EvaluatorSpec, ...]
    concurrency: int = Field(default=8, ge=1, le=256)
    case_timeout_s: float = Field(default=120.0, gt=0.0)
    retry: RetryPolicy = RetryPolicy()
    judge_concurrency: int = Field(default=4, ge=1, le=64)
    max_error_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    error_policy: Literal["exclude", "fail"] = "exclude"
    max_cost: Decimal | None = Field(
        default=None,
        ge=0,
        description=(
            "Running spend cap in the price table's currency; the runner cancels "
            "the run with error 'budget_exceeded' when accumulated known cost "
            "exceeds it. None disables the cap."
        ),
    )
    price_table_id: str
    price_table_version: str
    price_table_hash: str
    library_version: str
    python_version: str
    plugins: tuple[PluginRecord, ...] = ()
    seed: int | None = None
    notes: str | None = None


class Run(BaseModel):
    """One execution of a suite against one model.

    ``id`` is a canonical lowercase uuid4 string. Sortability is supplied by
    the ``runs(created_at desc)`` index, not by the id.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    label: str | None
    status: RunStatus
    created_at: UtcDatetime
    started_at: UtcDatetime | None
    completed_at: UtcDatetime | None
    config: RunConfig
    totals: RunTotals
    updated_at: UtcDatetime
    baseline_run_id: str | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


class CaseResult(BaseModel):
    """Everything recorded about one case in one run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    case_id: str
    case_hash: str
    status: CaseStatus
    response: ModelResponse | None
    evaluations: tuple[EvaluationResult, ...] = ()
    passed: bool | None
    score: float | None
    cost: CostBreakdown | None
    attempts: int = 1
    started_at: UtcDatetime | None
    completed_at: UtcDatetime | None
    error: ProviderErrorInfo | None = None
    category: str | None = None
    tags: tuple[str, ...] = ()


class LatencyStats(BaseModel):
    """Latency distribution over SUCCESSFULLY COMPLETED cases only.

    Errored cases with no measured latency are excluded, and ``n`` is what
    makes that exclusion visible next to every percentile.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    mean_ms: float | None
    p50_ms: float | None
    p90_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    low_confidence: tuple[str, ...] = ()


class TokenTotals(BaseModel):
    """Token totals, with explicit counts of how many cases reported usage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    n_with_usage: int
    n_missing_usage: int


class CategoryMetrics(BaseModel):
    """Pass rate and mean score for one category or tag bucket."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    n_passed: int
    pass_rate: float | None
    pass_rate_ci: tuple[float, float] | None
    mean_score: float | None


class EvaluatorMetrics(BaseModel):
    """Per-evaluator rollup for one run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluator_id: str
    evaluator_type: str
    n: int
    """Every evaluation this evaluator produced, including errored ones."""
    n_passed: int
    n_errors: int
    pass_rate: float | None
    mean_score: float | None
    median_score: float | None
    n_score_only: int = 0
    n_pass_denominator: int = 0
    """Evaluations eligible for pass/fail under the run's `error_policy`.

    The denominator of `pass_rate`, and the `n` behind any confidence interval
    computed for it. Named exactly as `AggregateMetrics.n_pass_denominator` is at
    run level, because one concept carrying two names is the drift addendum 5
    exists to prevent. Recorded because it is not derivable from `n`, `n_errors` and
    `n_score_only` without re-deriving the exclusion rule that
    `llm_eval_lab.scoring` owns, and because a consumer comparing an evaluator
    across two runs needs both `k` and `n` to say anything about uncertainty.

    Defaulted to zero so a rollup persisted before this field existed still
    loads. Zero there means UNKNOWN, not "nothing was eligible", and a consumer
    that gates on a sample size treats it as insufficient rather than guessing.
    """


class AggregateMetrics(BaseModel):
    """The computed rollup for one run.

    ``error_rate`` is always over total attempted, whatever ``error_policy``
    excludes from the pass-rate denominator, and that denominator is recorded
    explicitly in ``n_pass_denominator`` rather than left to be re-derived. The
    two coverage ratios say what fraction of cases actually carried token usage
    and a price, so a cost figure is never read as complete when it is not.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    computed_at: UtcDatetime
    n_cases: int
    n_completed: int
    n_errors: int
    n_timeouts: int
    n_skipped: int
    error_rate: float
    error_breakdown: dict[str, int]
    error_policy: str
    pass_rate: float | None
    pass_rate_ci: tuple[float, float] | None
    n_passed: int = 0
    """Cases counted as passing. The numerator of `pass_rate`."""
    n_pass_denominator: int = 0
    """Cases eligible for pass/fail after `error_policy` and score-only exclusions.

    The denominator of `pass_rate`, and the `n` behind `pass_rate_ci`. Recorded
    because the denominator is not derivable from `n_completed`, `n_errors`,
    `n_skipped` and `n_score_only` without re-deriving the exclusion rule in a
    second module, and because a Wilson interval needs both `k` and `n`.
    """
    n_scored: int
    mean_score: float | None
    median_score: float | None
    n_score_only: int = 0
    latency: LatencyStats
    tokens: TokenTotals
    token_usage_coverage: float | None
    cost: CostBreakdown
    cost_coverage: float | None
    cost_per_case: Decimal | None
    cost_per_successful_evaluation: Decimal | None
    by_category: dict[str, CategoryMetrics]
    by_tag: dict[str, CategoryMetrics]
    by_evaluator: dict[str, EvaluatorMetrics]


class ProgressEvent(BaseModel):
    """One observation emitted by the runner while a run executes.

    Three consumers share this shape: the runner emits it, the CLI renders it
    as a progress bar, and the API's run manager snapshots it for the run
    status endpoint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ProgressEventType
    run_id: str
    case_id: str | None = None
    completed: int = 0
    total: int = 0
    passed: int = 0
    errors: int = 0
    attempt: int | None = None
    retry_in_s: float | None = None
    message: str | None = None
    at: UtcDatetime


type ProgressCallback = Callable[[ProgressEvent], None]
"""Synchronous by design and MUST NOT block: it is called from inside the run's TaskGroup.

A consumer that needs to do I/O queues the event and returns.
"""


class Page[T](BaseModel):
    """One page of a listing, with the total available behind it."""

    model_config = ConfigDict(frozen=True)

    items: tuple[T, ...]
    total: int
    limit: int
    offset: int


class RunQuery(BaseModel):
    """Filters and paging for a run listing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: RunStatus | None = None
    provider: str | None = None
    model: str | None = None
    suite_name: str | None = None
    suite_hash: str | None = None
    since: UtcDatetime | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    order: Literal["created_at", "-created_at"] = "-created_at"


class CaseQuery(BaseModel):
    """Filters and paging for the cases within one run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: CaseStatus | None = None
    passed: bool | None = None
    tag: str | None = None
    category: str | None = None
    q: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class RunSummary(BaseModel):
    """The listing-sized projection of a run, denormalized for the runs table."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str | None
    status: RunStatus
    created_at: UtcDatetime
    completed_at: UtcDatetime | None
    provider: str
    model: str
    suite_name: str
    suite_version: str
    suite_hash: str
    n_cases: int
    n_completed: int
    n_errors: int
    pass_rate: float | None
    total_cost: Decimal | None
    p95_ms: float | None


class EvaluationRecord(BaseModel):
    """Persistence-facing row: an EvaluationResult keyed to its run and case."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    case_id: str
    result: EvaluationResult
