"""OpenAI, through the Responses API.

Responses rather than Chat Completions, because OpenAI documents Responses as
the default for new projects and states that Chat Completions receives no new
feature development. The consequences are recorded here rather than discovered
later: on the pinned SDK, ``responses.create`` has no ``stop`` and no ``seed``
parameter, so :class:`~llm_eval_lab.models.GenerationParams` fields of those
names are dropped for this provider. That is the documented behaviour of the
parameter contract - a provider translates what it can and drops what it cannot
- and it is written down here so a benchmark author reading a suite that sets
``stop`` is not left wondering why it had no effect.

The SDK is imported lazily, on first request, so installing this package
without the ``openai`` extra costs nothing and importing
``llm_eval_lab.providers`` never pulls a vendor SDK into a ``llm-eval --help``.

Documentation consulted, with access date and pinned version:
``docs/providers-sources.md``.
"""

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from llm_eval_lab.models import (
    FinishReason,
    ModelResponse,
    ProviderAuthError,
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
    dict_at,
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
    from openai import AsyncOpenAI

OPENAI_EXTRA = "openai"
"""The install extra backing this provider: ``pip install 'llm-eval-lab[openai]'``."""

OPENAI_IMPORT_NAME = "openai"
"""The module whose presence decides whether this provider is available."""

OPENAI_CREDENTIAL_ENV = "OPENAI_API_KEY"
"""Default environment variable NAME holding the credential. Never its value."""

_TOOL_CALL_ITEM_TYPES: frozenset[str] = frozenset(
    {
        "function_call",
        "custom_tool_call",
        "computer_call",
        "file_search_call",
        "web_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "local_shell_call",
        "mcp_call",
    }
)
"""Output item types that mean the model stopped to call a tool rather than to answer."""

_INCOMPLETE_REASONS: dict[str, FinishReason] = {
    "max_output_tokens": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}
"""``incomplete_details.reason`` values the pinned SDK types. Anything else is UNKNOWN."""


class OpenAIOptions(BaseModel):
    """Vendor knobs with no neutral spelling, read from ``ProviderConfig.options``.

    ``extra="forbid"`` on purpose: a misspelled key in a benchmark file is a
    mistake the author wants to hear about, not a setting that silently did
    nothing for a whole run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    store: bool = False
    """Ask OpenAI to retain the response. Off by default: benchmark prompts are user data."""

    truncation: Literal["auto", "disabled"] | None = None
    """Server-side context truncation strategy. ``None`` leaves the API default in place."""

    service_tier: Literal["auto", "default", "flex", "priority"] | None = None
    """Latency/price tier. ``None`` leaves the account default in place."""


def _parse_options(config: ProviderConfig, provider: str) -> OpenAIOptions:
    """Validate the provider-specific options block.

    Raises:
        ProviderInvalidRequestError: when `options` does not describe a usable
            configuration, so the runner classifies it as a non-retryable
            failure like any other bad request.
    """
    try:
        return OpenAIOptions.model_validate(config.options)
    except ValidationError as exc:
        msg = f"invalid options for the {provider} provider: {exc}"
        raise ProviderInvalidRequestError(msg, provider=provider) from exc


def import_openai(provider: str = "openai") -> Any:
    """Import the OpenAI SDK, or explain exactly which extra is missing.

    `provider` names the adapter that needs it, so the ``openai_compatible``
    adapter - which uses the same client against a third-party endpoint -
    reports its own name rather than OpenAI's.

    Raises:
        ProviderExtraRequiredError: when the ``openai`` extra is not installed.
    """
    from llm_eval_lab.providers.registry import (  # noqa: PLC0415 - avoids an import cycle
        ProviderExtraRequiredError,
    )

    try:
        import openai  # noqa: PLC0415 - lazy by design; see the module docstring
    except ImportError as exc:
        msg = (
            f"the {provider} provider needs the 'openai' extra; install it with "
            f"pip install 'llm-eval-lab[openai]'"
        )
        raise ProviderExtraRequiredError(msg, provider=provider, extra=OPENAI_EXTRA) from exc
    return openai


class OpenAIProvider(RemoteProvider[dict[str, Any]]):
    """One Responses API call per :meth:`generate`, normalized and timed."""

    name: ClassVar[str] = "openai"
    credential_env: ClassVar[str | None] = OPENAI_CREDENTIAL_ENV

    def __init__(self, config: ProviderConfig, credential: Any = None) -> None:
        """Validate the options block and bind the credential. Imports no SDK."""
        super().__init__(config, credential)
        self.options = _parse_options(config, self.name)
        self._client: AsyncOpenAI | None = None

    def client(self) -> "AsyncOpenAI":
        """Return the vendor client, constructing it on first use.

        ``max_retries=0`` is not optional. The SDK retries connection errors,
        408, 409, 429 and 5xx twice by default, which would make three requests
        leave the machine while this project recorded one attempt, and would
        double-count against the runner's own bounded backoff (decision D4).
        """
        if self._client is None:
            openai = import_openai(self.name)
            api_key = self._require_api_key()
            try:
                self._client = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url=self._config.base_url,
                    organization=self._config.organization,
                    timeout=build_timeout(self._config),
                    max_retries=0,
                )
            except openai.OpenAIError as exc:
                # `_require_api_key` has already ruled out the usual cause. This
                # is the second net: any other constructor refusal is still a
                # configuration failure, and reporting it as UNKNOWN would tell
                # the operator nothing about what to change.
                msg = f"the openai client could not be constructed: {scrub_message(exc)}"
                raise ProviderAuthError(msg, provider=self.name) from exc
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP transport, if one was ever opened."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -- request -----------------------------------------------------------

    def _build_kwargs(self, request: ProviderRequest) -> dict[str, Any]:
        """Translate a neutral request into Responses API parameters.

        ``system`` becomes ``instructions``; ``max_output_tokens`` keeps its
        name; ``response_format="json_object"`` becomes ``text.format``.
        ``stop`` and ``seed`` have no parameter on this API at the pinned
        version and are dropped, which the module docstring states plainly.
        """
        params = request.params
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "input": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "store": self.options.store,
        }
        if request.system is not None:
            kwargs["instructions"] = request.system
        if params.max_output_tokens is not None:
            kwargs["max_output_tokens"] = params.max_output_tokens
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.response_format == "json_object":
            kwargs["text"] = {"format": {"type": "json_object"}}
        if self.options.truncation is not None:
            kwargs["truncation"] = self.options.truncation
        if self.options.service_tier is not None:
            kwargs["service_tier"] = self.options.service_tier
        if params.extra:
            kwargs["extra_body"] = dict(params.extra)
        return kwargs

    async def _invoke(self, request: ProviderRequest) -> dict[str, Any]:
        """Perform one Responses API call and return its payload as a plain dict."""
        response = await self.client().responses.create(**self._build_kwargs(request))
        dumped: dict[str, Any] = response.model_dump(mode="json")
        return dumped

    # -- normalization -----------------------------------------------------

    def _normalize(self, payload: dict[str, Any], timing: AttemptTiming) -> ModelResponse:
        """Map a Responses payload onto the normalized response contract.

        Raises:
            ProviderError: when the payload reports a terminal server-side
                state (``failed`` or ``cancelled``) rather than a response. A
                :class:`~llm_eval_lab.models.ModelResponse` cannot represent
                "no output and no error", so those two states are raised.
        """
        self._raise_on_terminal_status(payload)
        finish_reason = _finish_reason(payload)
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

    def _raise_on_terminal_status(self, payload: dict[str, Any]) -> None:
        """Raise for a payload that carries no usable output.

        Raises:
            ProviderError: for ``failed``, ``cancelled``, and for the
                background-mode statuses this provider never requests.
        """
        status = text_at(payload, "status")
        if status == "failed":
            error = dict_at(payload, "error")
            code = text_at(error, "code")
            message = text_at(error, "message") or "the response failed"
            msg = f"OpenAI reported a failed response: {message}"
            raise self._error(msg, kind=_failed_kind(code), vendor_code=code)
        if status == "cancelled":
            msg = "OpenAI reported a cancelled response"
            raise self._error(msg, kind=ProviderErrorKind.CANCELLED)
        if status in {"in_progress", "queued"}:
            msg = (
                f"OpenAI returned a response still in state {status!r}; "
                f"this provider does not use background mode"
            )
            raise self._error(msg, kind=ProviderErrorKind.UNKNOWN)

    # -- errors ------------------------------------------------------------

    def _classify_error(self, exc: Exception) -> ProviderError:
        """Map an OpenAI SDK exception onto a typed provider error."""
        return classify_openai_error(exc, provider=self.name)


def _failed_kind(code: str | None) -> ProviderErrorKind:
    """Classify the ``error.code`` on a ``status="failed"`` response."""
    if code is None:
        return ProviderErrorKind.SERVER
    lowered = code.lower()
    if "content_filter" in lowered or "content_policy" in lowered:
        return ProviderErrorKind.CONTENT_FILTER
    if "context_length" in lowered or "too_long" in lowered:
        return ProviderErrorKind.CONTEXT_LENGTH
    if "rate_limit" in lowered:
        return ProviderErrorKind.RATE_LIMIT
    return ProviderErrorKind.SERVER


def _output_text(payload: dict[str, Any]) -> str:
    """Concatenate every ``output_text`` block in the response's output items.

    An empty string is a real, representable answer: a model that produced no
    text produced no text. It is not ``None``, which the response contract
    reserves for "this attempt failed".
    """
    chunks: list[str] = []
    for item in sequence_at(payload, "output"):
        if not isinstance(item, dict):
            continue
        for block in sequence_at(item, "content"):
            if isinstance(block, dict) and block.get("type") == "output_text":
                text = text_at(block, "text")
                if text is not None:
                    chunks.append(text)
    return "".join(chunks)


def _has_refusal(payload: dict[str, Any]) -> bool:
    """Report whether any output block is a refusal."""
    return any(
        isinstance(block, dict) and block.get("type") == "refusal"
        for item in sequence_at(payload, "output")
        if isinstance(item, dict)
        for block in sequence_at(item, "content")
    )


def _finish_reason(payload: dict[str, Any]) -> FinishReason:
    """Derive the finish reason from ``status`` and ``incomplete_details``."""
    status = text_at(payload, "status")
    if status == "incomplete":
        reason = text_at(dict_at(payload, "incomplete_details"), "reason")
        return _INCOMPLETE_REASONS.get(reason or "", FinishReason.UNKNOWN)
    if status != "completed":
        return FinishReason.UNKNOWN
    if _has_refusal(payload):
        return FinishReason.CONTENT_FILTER
    if any(
        isinstance(item, dict) and item.get("type") in _TOOL_CALL_ITEM_TYPES
        for item in sequence_at(payload, "output")
    ):
        return FinishReason.TOOL_USE
    return FinishReason.STOP


def _usage(payload: dict[str, Any]) -> TokenUsage:
    """Map the ``usage`` object, leaving every count the vendor omitted as ``None``."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=optional_count(usage.get("input_tokens")),
        output_tokens=optional_count(usage.get("output_tokens")),
        total_tokens=optional_count(usage.get("total_tokens")),
        cached_input_tokens=optional_count(
            dict_at(usage, "input_tokens_details").get("cached_tokens")
        ),
        reasoning_tokens=optional_count(
            dict_at(usage, "output_tokens_details").get("reasoning_tokens")
        ),
    )


_STATUS_EXCEPTION_KINDS: tuple[tuple[str, ProviderErrorKind], ...] = (
    # Ordered most specific first; the SDK's own class hierarchy is the mapping.
    ("APITimeoutError", ProviderErrorKind.TIMEOUT),
    ("APIConnectionError", ProviderErrorKind.CONNECTION),
    ("RateLimitError", ProviderErrorKind.RATE_LIMIT),
    ("AuthenticationError", ProviderErrorKind.AUTHENTICATION),
    ("PermissionDeniedError", ProviderErrorKind.PERMISSION),
    ("NotFoundError", ProviderErrorKind.NOT_FOUND),
    ("UnprocessableEntityError", ProviderErrorKind.INVALID_REQUEST),
    ("ConflictError", ProviderErrorKind.INVALID_REQUEST),
    ("InternalServerError", ProviderErrorKind.SERVER),
)
"""SDK exception class names and the kind each maps to, checked in this order.

``BadRequestError`` is deliberately absent: a 400 needs its ``code`` inspected
to tell a context-length overflow and a content-policy refusal apart from an
ordinary malformed request, which :func:`classify_openai_error` does.
"""

_BAD_REQUEST_CODES: tuple[tuple[str, ProviderErrorKind], ...] = (
    ("context_length_exceeded", ProviderErrorKind.CONTEXT_LENGTH),
    ("string_above_max_length", ProviderErrorKind.CONTEXT_LENGTH),
    ("content_filter", ProviderErrorKind.CONTENT_FILTER),
    ("content_policy_violation", ProviderErrorKind.CONTENT_FILTER),
    ("insufficient_quota", ProviderErrorKind.QUOTA),
    ("billing_hard_limit_reached", ProviderErrorKind.QUOTA),
    ("model_not_found", ProviderErrorKind.NOT_FOUND),
    ("unsupported_parameter", ProviderErrorKind.UNSUPPORTED),
    ("unsupported_value", ProviderErrorKind.UNSUPPORTED),
)
"""``error.code`` values on a 400 that mean something more precise than "bad request"."""


def classify_openai_error(exc: Exception, *, provider: str) -> ProviderError:
    """Map an OpenAI SDK exception onto a typed provider error.

    Shared by the ``openai`` and ``openai_compatible`` adapters, which use the
    same client and therefore raise the same exception classes. The mapping is
    driven by the SDK's exception classes and by ``error.code``, never by
    matching words in a message: vendors reword messages without notice, and a
    classifier that reads prose silently changes behaviour when they do.
    """
    openai = import_openai(provider)
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None
    code = _vendor_code(exc)
    message = scrub_message(getattr(exc, "message", None) or exc)

    def build(kind: ProviderErrorKind, *, retry_after_s: float | None = None) -> ProviderError:
        return ProviderError(
            message,
            kind=kind,
            provider=provider,
            status_code=status_code,
            retry_after_s=retry_after_s,
            vendor_code=code,
        )

    for class_name, kind in _STATUS_EXCEPTION_KINDS:
        sdk_class = getattr(openai, class_name, None)
        if sdk_class is not None and isinstance(exc, sdk_class):
            if kind is ProviderErrorKind.RATE_LIMIT:
                return build(kind, retry_after_s=_retry_after(exc))
            return build(kind)

    bad_request = getattr(openai, "BadRequestError", None)
    if bad_request is not None and isinstance(exc, bad_request):
        return build(_bad_request_kind(code))

    api_status_error = getattr(openai, "APIStatusError", None)
    if api_status_error is not None and isinstance(exc, api_status_error):
        return build(status_kind(status_code))

    validation_error = getattr(openai, "APIResponseValidationError", None)
    if validation_error is not None and isinstance(exc, validation_error):
        # The vendor answered with a body the SDK could not parse into its own
        # types. There is no PARSE kind in the contract, and calling this a
        # server error would make the runner retry a response that will not
        # change, so it is UNKNOWN and terminal.
        return build(ProviderErrorKind.UNKNOWN)

    return build(_transport_kind(exc))


def _bad_request_kind(code: str | None) -> ProviderErrorKind:
    """Refine a 400 using its ``error.code``, falling back to INVALID_REQUEST."""
    if code is not None:
        lowered = code.lower()
        for needle, kind in _BAD_REQUEST_CODES:
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
    """Read the vendor's own error code off an SDK exception, when it carries one."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            nested = error.get("code")
            if isinstance(nested, str) and nested:
                return nested
    return None


def _retry_after(exc: Exception) -> float | None:
    """Read a ``Retry-After`` header off a rate-limit exception, when present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    return parse_retry_after(header_value(getattr(response, "headers", None), "retry-after"))


__all__ = [
    "OPENAI_CREDENTIAL_ENV",
    "OPENAI_EXTRA",
    "OPENAI_IMPORT_NAME",
    "OpenAIOptions",
    "OpenAIProvider",
    "classify_openai_error",
    "import_openai",
]
