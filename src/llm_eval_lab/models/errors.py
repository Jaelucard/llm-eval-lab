"""Exception hierarchy and the loader's structured field-error shape.

Every exception this library raises descends from :class:`LLMEvalError`, so a
caller can bound its ``except`` clause to this project without resorting to a
bare ``except Exception``. The provider subclasses exist purely for ``except``
ergonomics: each one fixes a :class:`~llm_eval_lab.models.enums.ProviderErrorKind`
as its default, and retryability is derived from that kind through
:data:`~llm_eval_lab.models.enums.RETRYABLE_KINDS` rather than being decided
independently at each raise site.
"""

from dataclasses import dataclass
from typing import ClassVar

from llm_eval_lab.models.enums import RETRYABLE_KINDS, ProviderErrorKind


@dataclass(frozen=True)
class FieldError:
    """One precise, human-addressable problem found while loading a file.

    Loaders collect these and report all of them at once rather than aborting
    on the first, so a malformed suite is fixed in one pass instead of ten.
    """

    file: str
    location: str
    case_id: str | None
    message: str
    input_repr: str | None


class LLMEvalError(Exception):
    """Root of every exception raised by this library."""


class ProviderError(LLMEvalError):
    """One provider attempt failed.

    Providers perform exactly one attempt per call and raise this (decision
    D4); retries, backoff and rate-limit pacing belong to the runner. The
    ``kind`` decides retryability, so a provider that maps a vendor failure to
    the right kind gets correct retry behaviour for free.
    """

    default_kind: ClassVar[ProviderErrorKind] = ProviderErrorKind.UNKNOWN

    message: str
    kind: ProviderErrorKind
    provider: str | None
    status_code: int | None
    retry_after_s: float | None
    vendor_code: str | None

    def __init__(  # noqa: PLR0913 - mirrors the ProviderErrorInfo payload the runner records
        self,
        message: str,
        *,
        kind: ProviderErrorKind | None = None,
        provider: str | None = None,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        vendor_code: str | None = None,
    ) -> None:
        """Record a scrubbed message plus the normalized failure classification."""
        super().__init__(message)
        self.message = message
        self.kind = self.default_kind if kind is None else kind
        self.provider = provider
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.vendor_code = vendor_code

    @property
    def retryable(self) -> bool:
        """Report whether the runner may attempt this call again."""
        return self.kind in RETRYABLE_KINDS


class ProviderAuthError(ProviderError):
    """Credentials were missing, malformed or rejected."""

    default_kind: ClassVar[ProviderErrorKind] = ProviderErrorKind.AUTHENTICATION


class ProviderRateLimitError(ProviderError):
    """The vendor throttled the request. Retryable."""

    default_kind: ClassVar[ProviderErrorKind] = ProviderErrorKind.RATE_LIMIT


class ProviderTimeoutError(ProviderError):
    """The attempt exceeded its time budget. Retryable."""

    default_kind: ClassVar[ProviderErrorKind] = ProviderErrorKind.TIMEOUT


class ProviderInvalidRequestError(ProviderError):
    """The vendor rejected the request shape. Not retryable."""

    default_kind: ClassVar[ProviderErrorKind] = ProviderErrorKind.INVALID_REQUEST


class ProviderServerError(ProviderError):
    """The vendor returned a server-side failure. Retryable."""

    default_kind: ClassVar[ProviderErrorKind] = ProviderErrorKind.SERVER


class ProviderNotInstalledError(ProviderError):
    """The optional SDK extra backing this provider is not installed."""

    default_kind: ClassVar[ProviderErrorKind] = ProviderErrorKind.NOT_INSTALLED


class MissingCredentialError(ProviderAuthError):
    """A required credential environment variable is unset.

    The message names the variable. It never contains, echoes or hints at the
    value, because this text reaches logs, CLI output and API responses.
    """


class BenchmarkValidationError(LLMEvalError):
    """A benchmark file is structurally or semantically invalid.

    Carries every :class:`FieldError` the loader found, so the CLI and
    ``POST /api/benchmarks/validate`` can report all problems at once.
    """

    errors: tuple[FieldError, ...]

    def __init__(self, message: str, *, errors: tuple[FieldError, ...] = ()) -> None:
        """Record the summary message plus the individual field errors."""
        super().__init__(message)
        self.errors = errors


class EvaluatorError(LLMEvalError):
    """An evaluator failed in a way that is not an ordinary scoring failure."""


class EvaluatorConfigError(EvaluatorError):
    """An evaluator or threshold was configured with unusable parameters."""


class PricingError(LLMEvalError):
    """A price table could not be loaded, parsed or applied."""


class StorageError(LLMEvalError):
    """A persistence operation failed."""


class RegressionInputError(LLMEvalError):
    """Two runs cannot be compared, or a threshold policy is unusable."""
