"""Ollama, through its native ``POST /api/chat`` endpoint over plain httpx.

There is no SDK extra for this provider: ``httpx`` is already a core dependency,
Ollama's chat endpoint is a single JSON POST, and the endpoint is reachable
without credentials on a local machine. So this provider is always available and
needs no install extra, which is what makes it the one real provider a developer
can exercise end to end without an account.

Native ``/api/chat`` rather than Ollama's OpenAI-compatible shim, because the
native reply carries ``done_reason``, ``prompt_eval_count`` and ``eval_count``,
and the shim discards some of them. Point the ``openai_compatible`` provider at
``http://localhost:11434/v1`` if the compatibility layer is what is wanted.

**Absent token counts are ``None``, never zero.** The documented reply shape
omits ``prompt_eval_count`` and ``eval_count`` entirely in some cases - the
``load`` and ``unload`` replies in the official documentation carry neither - and
a missing count is not a count of zero. Reporting zero would say the request
consumed no tokens, which a cost calculation would then faithfully price at
nothing. :func:`~llm_eval_lab.providers.remote.optional_count` is what keeps that
distinction, and there is a test for it.

Documentation consulted, with access date and pinned version:
``docs/providers-sources.md``.
"""

import json
from typing import Any, ClassVar

import httpx
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
    capped_raw,
    dict_at,
    optional_count,
    scrub_message,
    status_kind,
    text_at,
)

OLLAMA_EXTRA: str | None = None
"""No install extra: this provider is built on httpx, which is a core dependency."""

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
"""Where Ollama listens by default. Override with ``ProviderConfig.base_url``."""

CHAT_PATH = "/api/chat"
"""The native chat endpoint, relative to the base URL."""

_DONE_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "load": FinishReason.UNKNOWN,
    "unload": FinishReason.UNKNOWN,
}
"""``done_reason`` values, mapped to the neutral contract.

`stop`, `load` and `unload` are the three the official `api.md` enumerates.
`length` is emitted by the server when the `num_predict` cap cuts generation
short but is NOT enumerated in that document, so it is mapped here as observed
behaviour; `docs/providers-sources.md` records the distinction rather than
letting the two blur together.

``load`` and ``unload`` are lifecycle replies rather than generations - the
server acknowledging that it loaded or released a model - so they carry no
answer and no token counts, and UNKNOWN is the honest report. An unrecognised or
absent value falls through to UNKNOWN too, rather than being assumed to be a
clean stop.
"""


class OllamaOptions(BaseModel):
    """Vendor knobs with no neutral spelling, read from ``ProviderConfig.options``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keep_alive: str | None = None
    """How long the server keeps the model loaded after this request, e.g. ``"5m"``."""

    num_ctx: int | None = Field(default=None, ge=1)
    """Context window to load the model with. The server default applies when unset."""

    think: bool | None = None
    """Enable thinking on models that support it. ``None`` leaves the server default."""


class OllamaProvider(RemoteProvider[dict[str, Any]]):
    """One ``/api/chat`` request per :meth:`generate`, normalized and timed."""

    name: ClassVar[str] = "ollama"

    def __init__(self, config: ProviderConfig, credential: Any = None) -> None:
        """Validate the options block and bind the credential.

        A credential is optional and unusual here: a local server wants none.
        When one is configured it is sent as a bearer token, which is what the
        common reverse-proxy deployments in front of Ollama expect.

        Raises:
            ProviderInvalidRequestError: for an unusable `options` mapping.
        """
        super().__init__(config, credential)
        try:
            self.options = OllamaOptions.model_validate(config.options)
        except ValidationError as exc:
            msg = f"invalid options for the ollama provider: {exc}"
            raise ProviderInvalidRequestError(msg, provider=self.name) from exc
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        """Return the configured server address, or Ollama's default."""
        return (self._config.base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")

    def client(self) -> httpx.AsyncClient:
        """Return the HTTP client, constructing it on first use.

        There is no retry configuration to disable: httpx does not retry
        requests, so one call here is one request on the wire, which is exactly
        the contract the runner's own backoff is built on (decision D4).
        """
        if self._client is None:
            headers = {}
            credential = self._api_key()
            if credential is not None:
                headers["Authorization"] = f"Bearer {credential}"
            connect = self._config.connect_timeout_s or self._config.timeout_s
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self._config.timeout_s, connect=connect),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP transport, if one was ever opened."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- request -----------------------------------------------------------

    def _build_options(self, request: ProviderRequest) -> dict[str, Any]:
        """Translate the decoding parameters into Ollama's ``options`` vocabulary.

        ``max_output_tokens`` becomes ``num_predict``; everything else keeps a
        recognisable name. ``params.extra`` carries a sampler knob that has no
        neutral spelling, such as ``top_k`` or ``repeat_penalty``.

        ``extra`` is applied FIRST and the translated neutral parameters
        overwrite it. Ollama has one flat ``options`` bag rather than the
        separate ``extra_body`` the other adapters use, so the two would
        otherwise collide silently: ``extra={"num_predict": 8}`` would override
        the suite's own ``max_output_tokens`` with nothing to show for it in any
        report. This way a neutral parameter always means what it says, and an
        `extra` key that shadows one is simply inert rather than quietly
        authoritative.
        """
        params = request.params
        options: dict[str, Any] = dict(params.extra)
        if params.temperature is not None:
            options["temperature"] = params.temperature
        if params.top_p is not None:
            options["top_p"] = params.top_p
        if params.max_output_tokens is not None:
            options["num_predict"] = params.max_output_tokens
        if params.stop:
            options["stop"] = list(params.stop)
        if params.seed is not None:
            options["seed"] = params.seed
        if self.options.num_ctx is not None:
            options["num_ctx"] = self.options.num_ctx
        return options

    def _build_body(self, request: ProviderRequest) -> dict[str, Any]:
        """Translate a neutral request into an ``/api/chat`` request body.

        ``stream`` is always false, because this project wants one complete
        reply per attempt rather than a token stream to reassemble.
        """
        messages: list[dict[str, Any]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.extend(
            {"role": message.role, "content": message.content} for message in request.messages
        )

        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
        }
        options = self._build_options(request)
        if options:
            body["options"] = options
        if request.params.response_format == "json_object":
            body["format"] = "json"
        if self.options.keep_alive is not None:
            body["keep_alive"] = self.options.keep_alive
        if self.options.think is not None:
            body["think"] = self.options.think
        return body

    async def _invoke(self, request: ProviderRequest) -> dict[str, Any]:
        """Perform one ``/api/chat`` request and return the decoded reply.

        Raises:
            ProviderError: for a non-2xx status, or for a body that is not a
                JSON object. Both are explicit; neither is swallowed.
        """
        response = await self.client().post(CHAT_PATH, json=self._build_body(request))
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise self._status_error(response)
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            msg = f"the Ollama server returned a body that is not JSON: {exc}"
            raise self._error(
                msg, kind=ProviderErrorKind.UNKNOWN, status_code=response.status_code
            ) from exc
        if not isinstance(payload, dict):
            msg = f"the Ollama server returned {type(payload).__name__}, expected a JSON object"
            raise self._error(msg, kind=ProviderErrorKind.UNKNOWN, status_code=response.status_code)
        return payload

    def _status_error(self, response: httpx.Response) -> ProviderError:
        """Build the typed error for a non-2xx reply.

        Ollama reports a model it has not pulled as a 404 with an ``error``
        member naming it, which is worth surfacing verbatim: "model not found,
        try pulling it first" is the entire fix, and burying it under a generic
        404 would cost the operator a debugging session.
        """
        detail = _error_detail(response)
        status = response.status_code
        message = f"the Ollama server returned HTTP {status}"
        if detail:
            message = f"{message}: {detail}"
        return self._error(message, kind=status_kind(status), status_code=status)

    # -- normalization -----------------------------------------------------

    def _normalize(self, payload: dict[str, Any], timing: AttemptTiming) -> ModelResponse:
        """Map an ``/api/chat`` reply onto the normalized response contract."""
        finish_reason = _DONE_REASONS.get(
            text_at(payload, "done_reason") or "", FinishReason.UNKNOWN
        )
        content = text_at(dict_at(payload, "message"), "content") or ""
        return ModelResponse(
            output_text=content,
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
            provider_request_id=None,
            raw=capped_raw(payload),
        )

    # -- errors ------------------------------------------------------------

    def _classify_error(self, exc: Exception) -> ProviderError:
        """Map an httpx exception onto a typed provider error."""
        if isinstance(exc, httpx.TimeoutException):
            kind = ProviderErrorKind.TIMEOUT
        elif isinstance(exc, httpx.TransportError):
            # Covers ConnectError, ReadError, RemoteProtocolError and the rest:
            # the server was unreachable or the connection failed mid-flight,
            # which is exactly what the runner is allowed to retry.
            kind = ProviderErrorKind.CONNECTION
        elif isinstance(exc, httpx.HTTPError):
            kind = ProviderErrorKind.UNKNOWN
        else:
            kind = ProviderErrorKind.UNKNOWN
        return ProviderError(scrub_message(exc), kind=kind, provider=self.name)


def _error_detail(response: httpx.Response) -> str | None:
    """Read Ollama's ``error`` member out of a failure body, defensively.

    The body of a failed request is not guaranteed to be JSON - a proxy in front
    of the server may return HTML - so an unparseable body yields ``None`` and
    the status code carries the message on its own.
    """
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(body, dict):
        return text_at(body, "error")
    return None


def _usage(payload: dict[str, Any]) -> TokenUsage:
    """Map the token counters, leaving every one the server omitted as ``None``.

    ``prompt_eval_cached_count`` is the count of prompt tokens served from the
    server's cache, which is the same thing every other provider here calls
    cached input, so it maps to the same field.
    """
    return TokenUsage(
        input_tokens=optional_count(payload.get("prompt_eval_count")),
        output_tokens=optional_count(payload.get("eval_count")),
        cached_input_tokens=optional_count(payload.get("prompt_eval_cached_count")),
    )


__all__ = [
    "CHAT_PATH",
    "OLLAMA_DEFAULT_BASE_URL",
    "OLLAMA_EXTRA",
    "OllamaOptions",
    "OllamaProvider",
]
