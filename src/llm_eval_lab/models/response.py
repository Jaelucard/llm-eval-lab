"""Normalized provider response shapes.

Every provider maps its vendor payload onto these types, so downstream layers
never branch on which vendor produced a result. Token counts are reported only
when the vendor actually reported them: absent usage stays absent rather than
being estimated, because an invented token count becomes an invented cost.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_eval_lab.models.common import JSONValue, UtcDatetime
from llm_eval_lab.models.enums import RETRYABLE_KINDS, FinishReason, ProviderErrorKind


class TokenUsage(BaseModel):
    """Token counts as reported by the vendor.

    ``total_tokens`` is derived when both halves are known and the vendor did
    not supply it. The reverse is never done: a lone total is never split into
    a fabricated input/output pair.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _derive_total_only_upwards(cls, data: Any) -> Any:
        """Fill in `total_tokens` from a known input/output pair, never the reverse."""
        if not isinstance(data, dict):
            return data
        if data.get("total_tokens") is not None:
            return data
        supplied_input = data.get("input_tokens")
        supplied_output = data.get("output_tokens")
        if isinstance(supplied_input, int) and isinstance(supplied_output, int):
            return {**data, "total_tokens": supplied_input + supplied_output}
        return data

    @property
    def is_complete(self) -> bool:
        """Report whether input, output and total counts are all present."""
        return (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
        )


class ProviderErrorInfo(BaseModel):
    """Structured, persistable record of a failed provider call.

    ``message`` has already been scrubbed by the provider layer, and
    ``exception_type`` carries a class name only; traceback text never reaches
    this object because it reaches storage, logs and the HTTP API.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ProviderErrorKind
    message: str
    provider: str
    retryable: bool = False
    attempts: int = Field(default=1, ge=1)
    status_code: int | None = None
    retry_after_s: float | None = Field(default=None, ge=0.0)
    vendor_code: str | None = None
    exception_type: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_retryable_from_kind(cls, data: Any) -> Any:
        """Derive `retryable` from `kind` unless it was given explicitly."""
        if not isinstance(data, dict) or "retryable" in data:
            return data
        kind = data.get("kind")
        if not isinstance(kind, str):
            return data
        try:
            resolved = ProviderErrorKind(kind)
        except ValueError:
            return data
        return {**data, "retryable": resolved in RETRYABLE_KINDS}

    @model_validator(mode="after")
    def _retryable_matches_kind(self) -> "ProviderErrorInfo":
        """Refuse a `retryable` flag that contradicts its `kind`.

        Retryability is a property of the kind, recorded once in
        `RETRYABLE_KINDS`. Without this the field is a second, durable source of
        truth: a persisted `kind=authentication, retryable=True` would make the
        runner retry a credential failure forever.
        """
        expected = self.kind in RETRYABLE_KINDS
        if self.retryable != expected:
            msg = (
                f"retryable={self.retryable} contradicts kind={self.kind.value!r}; "
                f"retryability is derived from RETRYABLE_KINDS, not chosen per error"
            )
            raise ValueError(msg)
        return self


class ModelResponse(BaseModel):
    """The normalized result of one case's generation, successful or failed.

    The error invariant is enforced here rather than by convention: a response
    either carries output text and no error, or an error and no output text.
    There is no third shape, so no consumer has to guess.
    """

    model_config = ConfigDict(frozen=True)

    output_text: str | None
    provider: str
    model: str
    requested_model: str
    finish_reason: FinishReason
    usage: TokenUsage = TokenUsage()
    latency_ms: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
    started_at: UtcDatetime
    completed_at: UtcDatetime
    attempts: int = Field(default=1, ge=1)
    truncated: bool = False
    error: ProviderErrorInfo | None = None
    provider_request_id: str | None = None
    raw: dict[str, JSONValue] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def _error_and_output_are_mutually_exclusive(self) -> "ModelResponse":
        """Require exactly one of `error` and `output_text` to be populated."""
        if self.error is not None:
            if self.output_text is not None:
                msg = "a ModelResponse carrying an error must have output_text=None"
                raise ValueError(msg)
            if self.finish_reason is not FinishReason.ERROR:
                msg = (
                    f"a ModelResponse carrying an error must have "
                    f"finish_reason={FinishReason.ERROR.value!r}, got {self.finish_reason.value!r}"
                )
                raise ValueError(msg)
        elif self.output_text is None:
            msg = "a ModelResponse with output_text=None must carry an error"
            raise ValueError(msg)
        elif self.finish_reason is FinishReason.ERROR:
            msg = "a ModelResponse with finish_reason=ERROR must carry an error"
            raise ValueError(msg)
        return self

    @property
    def succeeded(self) -> bool:
        """Report whether this response carries output rather than an error."""
        return self.error is None
