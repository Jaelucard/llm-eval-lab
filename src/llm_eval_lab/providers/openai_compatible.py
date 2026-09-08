"""Any third-party or self-hosted endpoint that speaks the OpenAI wire format.

``AsyncOpenAI(base_url=...)`` is the pattern OpenAI documents for pointing its
client at another server, so this adapter is the OpenAI client aimed elsewhere
rather than a second HTTP client with its own bugs.

**Chat Completions, not Responses.** Chat Completions is the surface third-party
servers actually implement; the Responses API is an OpenAI-hosted surface a
compatible endpoint is not expected to provide. Sending Responses requests to a
vLLM, LM Studio, Together or OpenRouter deployment would fail on the first call
against most of them, so this adapter calls ``chat.completions.create`` and the
``openai`` adapter keeps Responses. That is the one behavioural difference
between the two, and it is why they are separate modules rather than one class
with a flag.

**Fields that go missing, and what happens then.** Compatible servers implement
the schema to varying depths, so this adapter treats every field below the
response text as optional and degrades to ``None`` instead of inventing a value:

``usage``
    Frequently absent entirely. Every token count then reports ``None``, which
    makes the run's cost coverage fall rather than making the run look free.
``usage.prompt_tokens_details.cached_tokens`` and ``completion_tokens_details.reasoning_tokens``
    Almost always absent outside OpenAI itself.
``choices[].finish_reason``
    Sometimes ``null`` or a value outside OpenAI's set; anything unrecognised
    becomes ``FinishReason.UNKNOWN`` rather than being guessed at as ``stop``.
``id``
    Sometimes absent, in which case ``provider_request_id`` is ``None``.
``model``
    Often echoed back in a different form from the id that was requested, so
    both are recorded: ``model`` is what the server said it served, and
    ``requested_model`` is what the benchmark asked for.

**Credentials are optional here**, unlike every other real provider: a local
server usually wants none. The OpenAI client requires *some* api_key string, so
a non-secret placeholder is sent when the configuration names no credential
variable. Servers that ignore authentication ignore it; servers that check it
reject the placeholder with a 401, which is the correct and legible outcome.

Documentation consulted, with access date and pinned version:
``docs/providers-sources.md``.
"""

from typing import TYPE_CHECKING, Any, ClassVar

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
from llm_eval_lab.providers.openai_provider import (
    OPENAI_EXTRA,
    OPENAI_IMPORT_NAME,
    classify_openai_error,
    import_openai,
)
from llm_eval_lab.providers.remote import (
    AttemptTiming,
    RemoteProvider,
    build_timeout,
    capped_raw,
    dict_at,
    identifier_at,
    optional_count,
    sequence_at,
    text_at,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; the real import is lazy
    from openai import AsyncOpenAI

COMPATIBLE_EXTRA = OPENAI_EXTRA
"""Backed by the same extra as the OpenAI adapter, because it is the same client."""

COMPATIBLE_IMPORT_NAME = OPENAI_IMPORT_NAME
"""The module whose presence decides whether this provider is available."""

PLACEHOLDER_API_KEY = "not-required"
"""Sent when no credential variable is configured, because the client demands a string.

It is deliberately not secret-shaped, so it can never be mistaken for a real key
in a log line, and a server that does check authentication rejects it plainly.
"""

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
    "tool_calls": FinishReason.TOOL_USE,
    "function_call": FinishReason.TOOL_USE,
}
"""OpenAI's documented ``finish_reason`` set. Anything else, or ``null``, is UNKNOWN."""


class OpenAICompatibleProvider(RemoteProvider[dict[str, Any]]):
    """One Chat Completions call per :meth:`generate`, against a configured base URL."""

    name: ClassVar[str] = "openai_compatible"

    def __init__(self, config: ProviderConfig, credential: Any = None) -> None:
        """Bind the configuration, requiring a base URL. Imports no SDK.

        Raises:
            ProviderInvalidRequestError: when no ``base_url`` is configured.
                Defaulting to OpenAI's own endpoint would send a benchmark
                somewhere the operator did not name, and bill them for it.
        """
        super().__init__(config, credential)
        if not config.base_url:
            msg = (
                "the openai_compatible provider requires base_url naming the endpoint "
                "to call; it has no default"
            )
            raise ProviderInvalidRequestError(msg, provider=self.name)
        self._client: AsyncOpenAI | None = None

    def client(self) -> "AsyncOpenAI":
        """Return the vendor client, constructing it on first use with retries disabled."""
        if self._client is None:
            openai = import_openai(self.name)
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key() or PLACEHOLDER_API_KEY,
                base_url=self._config.base_url,
                organization=self._config.organization,
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
        """Translate a neutral request into Chat Completions parameters.

        The system instruction becomes a leading ``system`` message, which is
        how Chat Completions carries it; ``max_output_tokens`` becomes
        ``max_completion_tokens``; ``response_format`` maps across directly.
        """
        params = request.params
        messages: list[dict[str, Any]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.extend(
            {"role": message.role, "content": message.content} for message in request.messages
        )
        kwargs: dict[str, Any] = {"model": self._config.model, "messages": messages}
        if params.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = params.max_output_tokens
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.stop:
            kwargs["stop"] = list(params.stop)
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if params.extra:
            kwargs["extra_body"] = dict(params.extra)
        return kwargs

    async def _invoke(self, request: ProviderRequest) -> dict[str, Any]:
        """Perform one Chat Completions call and return its payload as a plain dict."""
        completion = await self.client().chat.completions.create(**self._build_kwargs(request))
        dumped: dict[str, Any] = completion.model_dump(mode="json")
        return dumped

    # -- normalization -----------------------------------------------------

    def _normalize(self, payload: dict[str, Any], timing: AttemptTiming) -> ModelResponse:
        """Map a Chat Completions payload onto the normalized response contract.

        Raises:
            ProviderError: when the payload carries no ``choices`` at all. That
                is not a response with empty output; it is a server that did
                not answer the question, and the response contract has no shape
                for "no output and no error".
        """
        choices = sequence_at(payload, "choices")
        if not choices or not isinstance(choices[0], dict):
            msg = "the endpoint returned no choices"
            raise self._error(msg, kind=ProviderErrorKind.UNKNOWN)
        choice: dict[str, Any] = choices[0]
        finish_reason = _FINISH_REASONS.get(
            text_at(choice, "finish_reason") or "", FinishReason.UNKNOWN
        )
        return ModelResponse(
            output_text=_content(choice),
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
        """Map an OpenAI SDK exception onto a typed provider error."""
        return classify_openai_error(exc, provider=self.name)


def _content(choice: dict[str, Any]) -> str:
    """Return the assistant message text, or an empty string when there is none.

    A tool-call-only reply legitimately has ``content: null``. That is empty
    output, not a failure, and it is reported as an empty string with a
    ``tool_use`` finish reason rather than as ``None``, which the response
    contract reserves for a failed attempt.
    """
    message = dict_at(choice, "message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some servers return the multimodal content-part array shape here.
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _usage(payload: dict[str, Any]) -> TokenUsage:
    """Map ``usage``, leaving every count the endpoint omitted as ``None``."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=optional_count(usage.get("prompt_tokens")),
        output_tokens=optional_count(usage.get("completion_tokens")),
        total_tokens=optional_count(usage.get("total_tokens")),
        cached_input_tokens=optional_count(
            dict_at(usage, "prompt_tokens_details").get("cached_tokens")
        ),
        reasoning_tokens=optional_count(
            dict_at(usage, "completion_tokens_details").get("reasoning_tokens")
        ),
    )


__all__ = [
    "COMPATIBLE_EXTRA",
    "COMPATIBLE_IMPORT_NAME",
    "PLACEHOLDER_API_KEY",
    "OpenAICompatibleProvider",
]
