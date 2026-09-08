"""Enumerated value sets shared by every layer of the frozen contract.

Every ``StrEnum`` in the contract lives here so that the value strings have
exactly one definition, and so the topical model modules import them rather
than redefining them. The string values are the wire format: they appear in
persisted rows, in CLI JSON output and in the HTTP API, so they are part of
the contract and are never renamed casually.
"""

from enum import StrEnum


class FinishReason(StrEnum):
    """Why a provider stopped producing output."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_USE = "tool_use"
    ERROR = "error"
    UNKNOWN = "unknown"


class ProviderErrorKind(StrEnum):
    """Vendor-neutral classification of a single failed provider attempt."""

    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    CONTEXT_LENGTH = "context_length"
    CONTENT_FILTER = "content_filter"
    UNSUPPORTED = "unsupported"
    NOT_INSTALLED = "not_installed"
    QUOTA = "quota"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER = "server"


RETRYABLE_KINDS: frozenset[ProviderErrorKind] = frozenset(
    {
        ProviderErrorKind.RATE_LIMIT,
        ProviderErrorKind.TIMEOUT,
        ProviderErrorKind.CONNECTION,
        ProviderErrorKind.SERVER,
    }
)
"""The only error kinds a retry may be attempted for. Everything else is terminal."""


class EvaluationStatus(StrEnum):
    """Outcome of one evaluator applied to one model response."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class RunStatus(StrEnum):
    """Lifecycle state of a benchmark run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class CaseStatus(StrEnum):
    """Lifecycle state of a single case within a run."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ProgressEventType(StrEnum):
    """Kind of progress event emitted by the runner while a run executes."""

    RUN_STARTED = "run_started"
    CASE_STARTED = "case_started"
    CASE_RETRY = "case_retry"
    CASE_COMPLETED = "case_completed"
    CASE_FAILED = "case_failed"
    EVALUATION_COMPLETED = "evaluation_completed"
    RUN_COMPLETED = "run_completed"


class ThresholdDirection(StrEnum):
    """Which way a metric has to move for the change to be an improvement."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class CheckStatus(StrEnum):
    """Outcome of a single regression threshold check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    INSUFFICIENT_DATA = "insufficient_data"
    MISSING_METRIC = "missing_metric"


class Verdict(StrEnum):
    """Overall outcome of a regression comparison, and the CI exit-code source."""

    PASS = "pass"  # noqa: S105 - a verdict value, not a credential
    WARN = "warn"
    FAIL = "fail"
    INCOMPARABLE = "incomparable"
