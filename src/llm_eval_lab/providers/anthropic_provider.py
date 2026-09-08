"""Anthropic, through the Messages API.

Three facts about this API shape the adapter, all confirmed against the current
documentation and the pinned SDK rather than recalled:

**``max_tokens`` is required.** There is no server-side default, so a benchmark
that does not set ``max_output_tokens`` would fail every request. This adapter
supplies :data:`DEFAULT_MAX_TOKENS` in that case and records the value it used
in the request, so the cap is visible rather than mysterious.

**Text lives in a content block list**, not in a single string field. A reply is
``content: [{"type": "text", "text": ...}, ...]`` and may interleave text with
tool-use and thinking blocks, so the output text is the concatenation of the
``text`` blocks and nothing else.

**``temperature`` is documented as deprecated on newer models**, so it is sent
only when a benchmark set it explicitly. ``GenerationParams.temperature``
defaults to ``None``, which means "say nothing about temperature", and this
adapter preserves that distinction instead of substituting a default of its own.

Documentation consulted, with access date and pinned version:
``docs/providers-sources.md``.
"""

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_eval_lab.models import (
    FinishReason,
    ModelResponse,
    ProviderConfig,
    ProviderError,
    ProviderErrorKind,
    ProviderInvalidRequestError,
    ProviderRequest,
    TokenUsage,
)
from llm_eval_lab.providers.remote import (
    AttemptTiming,
    RemoteProvider,
    build_timeout,
    capped_raw,
    header_value,
    identifier_at,
    optional_count,
    parse_retry_after,
    scrub_message,
    sequence_at,
    status_kind,
    text_at,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; the real import is lazy
    from anthropic import AsyncAnthropic

ANTHROPIC_EXTRA = "anthropic"
"""The install extra backing this provider: ``pip install 'llm-eval-lab[anthropic]'``."""

ANTHROPIC_IMPORT_NAME = "anthropic"
"""The module whose presence decides whether this provider is available."""

ANTHROPIC_CREDENTIAL_ENV = "ANTHROPIC_API_KEY"
"""Default environment variable NAME holding the credential. Never its value."""

DEFAULT_MAX_TOKENS = 4096
"""Output cap sent when a benchmark sets none, because the API requires one.

Chosen to be large enough for ordinary benchmark answers and small enough that a
runaway generation cannot quietly cost a fortune. Override it per suite with
``GenerationParams.max_output_tokens``, or per provider with
``options.default_output_cap``.
"""

_STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "model_context_window_exceeded": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_USE,
    "refusal": FinishReason.CONTENT_FILTER,
    "pause_turn": FinishReason.UNKNOWN,
}
"""The full ``stop_reason`` set of the pinned SDK, mapped to the neutral contract.

``model_context_window_exceeded`` is grouped with ``max_tokens`` because both
mean the same thing to a consumer: the answer was cut short by a limit rather
than finished. ``pause_turn`` means a long-running server tool paused the turn
and the caller is expected to continue it; this project makes one attempt per
case and does not continue, so the honest report is UNKNOWN rather than STOP.
"""


class AnthropicOptions(BaseModel):
    """Vendor knobs with no neutral spelling, read from ``ProviderConfig.options``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_output_cap: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    """Output cap used when the benchmark sets none. The API requires a value.

    Spelled without the word "token" deliberately:
    :class:`~llm_eval_lab.models.ProviderConfig` refuses any option key
    containing it, because that is one of the shapes a credential arrives in,
    and this object is persisted and logged verbatim.
    """


def _import_anthropic() -> Any:
    """Import the Anthropic SDK, or explain exactly which extra is missing.

    Raises:
        ProviderExtraRequiredError: when the ``anthropic`` extra is not installed.
    """
    from llm_eval_lab.providers.registry import (  # noqa: PLC0415 - avoids an import cycle
        ProviderExtraRequiredError,
    )

    try:
        import anthropic  # noqa: PLC0415 - lazy by design; see the module docstring
    except ImportError as exc:
        msg = (
            "the anthropic provider needs the 'anthropic' extra; install it with "
            "pip install 'llm-eval-lab[anthropic]'"
        )
        raise ProviderExtraRequiredError(msg, provider="anthropic", extra=ANTHROPIC_EXTRA) from exc
    return anthropic


class AnthropicProvider(RemoteProvider[dict[str, Any]]):
    """One Messages API call per :meth:`generate`, normalized and timed."""

    name: ClassVar[str] = "anthropic"
    credential_env: ClassVar[str | None] = ANTHROPIC_CREDENTIAL_ENV

    def __init__(self, config: ProviderConfig, credential: Any = None) -> None:
        """Validate the options block and bind the credential. Imports no SDK.

        Raises:
            ProviderInvalidRequestError: for an unusable `options` mapping.
        """
        super().__init__(config, credential)
        try:
            self.options = AnthropicOptions.model_validate(config.options)
        except ValidationError as exc:
            msg = f"invalid options for the anthropic provider: {exc}"
            raise ProviderInvalidRequestError(msg, provider=self.name) from exc
        self._client: AsyncAnthropic | None = None

    def client(self) -> "AsyncAnthropic":
        """Return the vendor client, constructing it on first use.

        ``max_retries=0`` disables the SDK's own retry loop so the runner owns
        backoff and the recorded attempt count is the number of requests that
        actually left the machine (decision D4).

        The credential is required up front. This client constructs happily with
        ``api_key=None`` and only fails deep inside the first request, with a
        ``TypeError`` that no SDK error class covers and that would therefore be
        reported as an unexplained ``UNKNOWN`` rather than as the configuration
        mistake it is. It also reads ``ANTHROPIC_API_KEY`` from the environment
        itself when handed ``None``, which decision D8 reserves for
        ``settings.read_credential``.

        An ``httpx.Timeout`` rather than a bare float, so a configured
        ``connect_timeout_s`` is honoured here exactly as it is on every other
        provider.
        """
        if self._client is None:
            anthropic = _import_anthropic()
            self._client = anthropic.AsyncAnthropic(
                api_key=self._require_api_key(),
                base_url=self._config.base_url,
                timeout=build_timeout(self._config),
                max_retries=0,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP transport, if one was ever opened."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -- request -----------------------------------------------------------

    def _build_kwargs(self, request: ProviderRequest) -> dict[str, Any]:
        """Translate a neutral request into Messages API parameters.

        A ``system`` turn inside ``messages`` is lifted into the top-level
        ``system`` parameter, because the Messages API rejects a system role in
        the message list. ``response_format`` has no equivalent on this API and
        is dropped; ask for JSON in the prompt instead.
        """
        params = request.params
        system_parts = [] if request.system is None else [request.system]
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                system_parts.append(message.content)
                continue
            messages.append({"role": message.role, "content": message.content})

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": params.max_output_tokens or self.options.default_output_cap,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.stop:
            kwargs["stop_sequences"] = list(params.stop)
        if params.extra:
            kwargs["extra_body"] = dict(params.extra)
        return kwargs

    async def _invoke(self, request: ProviderRequest) -> dict[str, Any]:
        """Perform one Messages API call and return its payload as a plain dict."""
        message = await self.client().messages.create(**self._build_kwargs(request))
        dumped: dict[str, Any] = message.model_dump(mode="json")
        return dumped

    # -- normalization -----------------------------------------------------

    def _normalize(self, payload: dict[str, Any], timing: AttemptTiming) -> ModelResponse:
        """Map a Messages payload onto the normalized response contract."""
        finish_reason = _STOP_REASONS.get(
            text_at(payload, "stop_reason") or "", FinishReason.UNKNOWN
        )
        return ModelResponse(
            output_text=_output_text(payload),
            provider=self.name,
            model=text_at(payload, "model") or self._config.model,
            requested_model=self._config.model,
            finish_reason=finish_reason,
            usage=_usage(payload),
            latency_ms=timing.latency_ms,
            total_latency_ms=timing.latency_ms,
            started_at=timing.started_at,
            completed_at=timing.completed_at,
            attempts=1,
            truncated=finish_reason is FinishReason.LENGTH,
            error=None,
            provider_request_id=identifier_at(payload, "id"),
            raw=capped_raw(payload),
        )

    # -- errors ------------------------------------------------------------

    def _classify_error(self, exc: Exception) -> ProviderError:
        """Map an Anthropic SDK exception onto a typed provider error."""
        anthropic = _import_anthropic()
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            status_code = None
        code = _vendor_code(exc)
        message = scrub_message(getattr(exc, "message", None) or exc)

        for class_name, kind in _EXCEPTION_KINDS:
            sdk_class = getattr(anthropic, class_name, None)
            if sdk_class is not None and isinstance(exc, sdk_class):
                retry_after = _retry_after(exc) if kind is ProviderErrorKind.RATE_LIMIT else None
                return ProviderError(
                    message,
                    kind=kind,
                    provider=self.name,
                    status_code=status_code,
                    retry_after_s=retry_after,
                    vendor_code=code,
                )

        bad_request = getattr(anthropic, "BadRequestError", None)
        if bad_request is not None and isinstance(exc, bad_request):
            return ProviderError(
                message,
                kind=_bad_request_kind(code),
                provider=self.name,
                status_code=status_code,
                vendor_code=code,
            )

        api_status_error = getattr(anthropic, "APIStatusError", None)
        if api_status_error is not None and isinstance(exc, api_status_error):
            return ProviderError(
                message,
                kind=status_kind(status_code),
                provider=self.name,
                status_code=status_code,
                vendor_code=code,
            )
        return ProviderError(
            message,
            kind=_transport_kind(exc),
            provider=self.name,
            status_code=status_code,
            vendor_code=code,
        )


_EXCEPTION_KINDS: tuple[tuple[str, ProviderErrorKind], ...] = (
    # Ordered most specific first, because several of these are subclasses of
    # the ones below them: APITimeoutError extends APIConnectionError, and every
    # named status error extends APIStatusError.
    ("APITimeoutError", ProviderErrorKind.TIMEOUT),
    ("DeadlineExceededError", ProviderErrorKind.TIMEOUT),
    ("APIConnectionError", ProviderErrorKind.CONNECTION),
    ("RateLimitError", ProviderErrorKind.RATE_LIMIT),
    ("AuthenticationError", ProviderErrorKind.AUTHENTICATION),
    ("PermissionDeniedError", ProviderErrorKind.PERMISSION),
    ("NotFoundError", ProviderErrorKind.NOT_FOUND),
    ("RequestTooLargeError", ProviderErrorKind.CONTEXT_LENGTH),
    ("UnprocessableEntityError", ProviderErrorKind.INVALID_REQUEST),
    ("ConflictError", ProviderErrorKind.INVALID_REQUEST),
    ("OverloadedError", ProviderErrorKind.SERVER),
    ("ServiceUnavailableError", ProviderErrorKind.SERVER),
    ("InternalServerError", ProviderErrorKind.SERVER),
)
"""SDK exception class names and the kind each maps to, checked in this order.

``BadRequestError`` is absent on purpose: a 400 carries a vendor ``type`` that
distinguishes a context overflow from an ordinary malformed request, and
:func:`_bad_request_kind` reads it.
"""

_BAD_REQUEST_TYPES: tuple[tuple[str, ProviderErrorKind], ...] = (
    ("context_window", ProviderErrorKind.CONTEXT_LENGTH),
    ("prompt_too_long", ProviderErrorKind.CONTEXT_LENGTH),
    ("max_tokens", ProviderErrorKind.INVALID_REQUEST),
)
"""Vendor error types on a 400 that mean something more precise than "bad request"."""


def _bad_request_kind(code: str | None) -> ProviderErrorKind:
    """Refine a 400 using the vendor's error type, falling back to INVALID_REQUEST."""
    if code is not None:
        lowered = code.lower()
        for needle, kind in _BAD_REQUEST_TYPES:
            if needle in lowered:
                return kind
    return ProviderErrorKind.INVALID_REQUEST


def _transport_kind(exc: Exception) -> ProviderErrorKind:
    """Classify an exception raised below the SDK, such as a raw httpx failure."""
    import httpx  # noqa: PLC0415 - a core dependency, imported where it is used

    if isinstance(exc, httpx.TimeoutException):
        return ProviderErrorKind.TIMEOUT
    if isinstance(exc, httpx.TransportError):
        return ProviderErrorKind.CONNECTION
    return ProviderErrorKind.UNKNOWN


def _vendor_code(exc: Exception) -> str | None:
    """Read the vendor's own error type off an SDK exception, when it carries one."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            found = error.get("type")
            if isinstance(found, str) and found:
                return found
    return None


def _retry_after(exc: Exception) -> float | None:
    """Read a ``Retry-After`` header off a rate-limit exception, when present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    return parse_retry_after(header_value(getattr(response, "headers", None), "retry-after"))


def _output_text(payload: dict[str, Any]) -> str:
    """Concatenate every ``text`` content block, ignoring tool-use and thinking blocks."""
    return "".join(
        block["text"]
        for block in sequence_at(payload, "content")
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _usage(payload: dict[str, Any]) -> TokenUsage:
    """Map ``usage``, deriving the total the API does not report.

    The Messages API reports input and output counts but no total, so the
    total is derived from the pair by :class:`~llm_eval_lab.models.TokenUsage`.
    ``cache_read_input_tokens`` is the cache-hit count and is the figure a
    cached-input price applies to; ``cache_creation_input_tokens`` is a write
    and is left in ``raw`` rather than being folded into a field that means
    something else.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=optional_count(usage.get("input_tokens")),
        output_tokens=optional_count(usage.get("output_tokens")),
        cached_input_tokens=optional_count(usage.get("cache_read_input_tokens")),
    )


__all__ = [
    "ANTHROPIC_CREDENTIAL_ENV",
    "ANTHROPIC_EXTRA",
    "ANTHROPIC_IMPORT_NAME",
    "DEFAULT_MAX_TOKENS",
    "AnthropicOptions",
    "AnthropicProvider",
]
