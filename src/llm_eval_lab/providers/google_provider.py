"""Google Gemini, through the ``google-genai`` SDK.

The package is ``google-genai``, imported as ``google.genai``. The legacy
``google-generativeai`` package is not used anywhere in this repository and must
not be introduced: the two have different client objects, different
configuration types and different async surfaces, and code written against one
fails against the other in ways that look like a model problem.

Two response shapes matter and they are not the same shape:

**A generated answer** comes back as ``candidates[0].content.parts[].text`` with
a ``finish_reason``. A candidate stopped by a safety filter still *is* a
candidate; it simply carries little or no text. That is reported as a response
with ``finish_reason=content_filter`` and whatever text there was, because the
model did answer the request, just not usefully.

**A blocked prompt** comes back with **no candidates at all** and
``prompt_feedback.block_reason`` set. Nothing was generated, so there is no
response to report and this adapter raises a
:class:`~llm_eval_lab.models.ProviderError` of kind ``content_filter``. The
alternative - a response with ``output_text=None`` - is not representable, and
inventing an empty string would record a model that answered with silence when
in fact it was never asked.

Documentation consulted, with access date and pinned version:
``docs/providers-sources.md``.
"""

from typing import TYPE_CHECKING, Any, ClassVar

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
    identifier_at,
    optional_count,
    scrub_message,
    sequence_at,
    status_kind,
    text_at,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; the real import is lazy
    from google.genai import Client

GOOGLE_EXTRA = "google"
"""The install extra backing this provider: ``pip install 'llm-eval-lab[google]'``."""

GOOGLE_IMPORT_NAME = "google.genai"
"""The module whose presence decides whether this provider is available."""

GOOGLE_CREDENTIAL_ENV = "GOOGLE_API_KEY"
"""Default environment variable NAME holding the credential. Never its value."""

_MILLISECONDS_PER_SECOND = 1000

_FINISH_REASONS: dict[str, FinishReason] = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,
    "IMAGE_SAFETY": FinishReason.CONTENT_FILTER,
    "IMAGE_PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "IMAGE_RECITATION": FinishReason.CONTENT_FILTER,
    "MALFORMED_FUNCTION_CALL": FinishReason.TOOL_USE,
    "UNEXPECTED_TOOL_CALL": FinishReason.TOOL_USE,
}
"""The pinned SDK's 17-value ``FinishReason`` set, minus the ones that mean UNKNOWN.

``FINISH_REASON_UNSPECIFIED``, ``OTHER``, ``LANGUAGE``, ``NO_IMAGE`` and
``IMAGE_OTHER`` are deliberately absent and fall through to UNKNOWN. ``LANGUAGE``
in particular is not a content filter - it means the model declined an
unsupported language - and filing it under ``content_filter`` would make a
capability gap look like a safety event in every report that counts them.
"""


class GoogleOptions(BaseModel):
    """Vendor knobs with no neutral spelling, read from ``ProviderConfig.options``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_version: str | None = None
    """Pin the API version, for example ``v1`` instead of the SDK's default."""


def _import_genai() -> Any:
    """Import the Google GenAI SDK, or explain exactly which extra is missing.

    Raises:
        ProviderExtraRequiredError: when the ``google`` extra is not installed.
    """
    from llm_eval_lab.providers.registry import (  # noqa: PLC0415 - avoids an import cycle
        ProviderExtraRequiredError,
    )

    try:
        from google import genai  # noqa: PLC0415 - lazy by design; see the module docstring
    except ImportError as exc:
        msg = (
            "the google provider needs the 'google' extra; install it with "
            "pip install 'llm-eval-lab[google]'"
        )
        raise ProviderExtraRequiredError(msg, provider="google", extra=GOOGLE_EXTRA) from exc
    return genai


class GoogleProvider(RemoteProvider[dict[str, Any]]):
    """One ``generate_content`` call per :meth:`generate`, normalized and timed."""

    name: ClassVar[str] = "google"
    credential_env: ClassVar[str | None] = GOOGLE_CREDENTIAL_ENV

    def __init__(self, config: ProviderConfig, credential: Any = None) -> None:
        """Validate the options block and bind the credential. Imports no SDK.

        Raises:
            ProviderInvalidRequestError: for an unusable `options` mapping.
        """
        super().__init__(config, credential)
        try:
            self.options = GoogleOptions.model_validate(config.options)
        except ValidationError as exc:
            msg = f"invalid options for the google provider: {exc}"
            raise ProviderInvalidRequestError(msg, provider=self.name) from exc
        self._client: Client | None = None

    def client(self) -> "Client":
        """Return the vendor client, constructing it on first use.

        ``HttpRetryOptions(attempts=1)`` is how this SDK's own retry loop is
        disabled; there is no ``max_retries`` here. Without it the SDK would
        make several requests per recorded attempt and the runner's backoff
        would be layered on top of a hidden one (decision D4). ``timeout`` on
        ``HttpOptions`` is in MILLISECONDS, unlike every other client here.

        A separate ``connect_timeout_s`` cannot be expressed in that single
        millisecond value, so it is passed as an ``httpx.Timeout`` through
        ``async_client_args``, which this SDK splats into its
        ``httpx.AsyncClient``. Without it a configured connect budget would be
        honoured on three providers out of five.

        Raises:
            ProviderAuthError: when no credential was resolved, or when the SDK
                refuses the configuration for any other reason.
        """
        if self._client is None:
            genai = _import_genai()
            types = genai.types
            api_key = self._require_api_key()
            http_options = types.HttpOptions(
                timeout=int(self._config.timeout_s * _MILLISECONDS_PER_SECOND),
                retry_options=types.HttpRetryOptions(attempts=1),
                async_client_args={"timeout": build_timeout(self._config)},
            )
            if self._config.base_url is not None:
                http_options.base_url = self._config.base_url
            if self.options.api_version is not None:
                http_options.api_version = self.options.api_version
            try:
                self._client = genai.Client(api_key=api_key, http_options=http_options)
            except ValueError as exc:
                # This SDK signals an unusable credential with a bare ValueError,
                # which no error class covers and which would otherwise be
                # reported as an unexplained UNKNOWN.
                msg = f"the google client could not be constructed: {scrub_message(exc)}"
                raise ProviderAuthError(msg, provider=self.name) from exc
        return self._client

    async def aclose(self) -> None:
        """Close the async transport, if one was ever opened.

        ``Client.close()`` closes only the SYNCHRONOUS transport and documents
        that it leaves the async one open; ``Client.aio.aclose()`` is the one
        that releases what this provider actually used.
        """
        if self._client is not None:
            await self._client.aio.aclose()
            self._client = None

    # -- request -----------------------------------------------------------

    def _build_config(self, request: ProviderRequest, types: Any) -> Any:
        """Build the ``GenerateContentConfig`` for one request.

        The system instruction, the output cap and the decoding parameters all
        live on this object rather than on the call, which is the single
        biggest shape difference between this SDK and the other two.

        ``params.extra`` reaches the request through
        ``GenerateContentConfig.http_options.extra_body``, which is this SDK's
        equivalent of the ``extra_body`` the other adapters use. It is merged
        into the request body by the SDK, so a benchmark can reach a knob with
        no neutral spelling here exactly as it can everywhere else.
        """
        params = request.params
        system_parts = [] if request.system is None else [request.system]
        system_parts.extend(
            message.content for message in request.messages if message.role == "system"
        )
        config: dict[str, Any] = {}
        if system_parts:
            config["system_instruction"] = "\n\n".join(system_parts)
        if params.max_output_tokens is not None:
            config["max_output_tokens"] = params.max_output_tokens
        if params.temperature is not None:
            config["temperature"] = params.temperature
        if params.top_p is not None:
            config["top_p"] = params.top_p
        if params.stop:
            config["stop_sequences"] = list(params.stop)
        if params.seed is not None:
            config["seed"] = params.seed
        if params.response_format == "json_object":
            config["response_mime_type"] = "application/json"
        if params.extra:
            config["http_options"] = types.HttpOptions(extra_body=dict(params.extra))
        return types.GenerateContentConfig(**config)

    def _build_contents(self, request: ProviderRequest, types: Any) -> list[Any]:
        """Map the neutral message list onto ``Content`` turns.

        Gemini names the assistant role ``model``, and it has no ``system`` turn
        at all: a system message is lifted into ``system_instruction`` by
        :meth:`_build_config` and skipped here.
        """
        contents: list[Any] = []
        for message in request.messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=message.content)])
            )
        return contents

    async def _invoke(self, request: ProviderRequest) -> dict[str, Any]:
        """Perform one ``generate_content`` call and return its payload as a plain dict."""
        genai = _import_genai()
        types = genai.types
        response = await self.client().aio.models.generate_content(
            model=self._config.model,
            contents=self._build_contents(request, types),
            config=self._build_config(request, types),
        )
        dumped: dict[str, Any] = response.model_dump(mode="json", exclude_none=True)
        return dumped

    # -- normalization -----------------------------------------------------

    def _normalize(self, payload: dict[str, Any], timing: AttemptTiming) -> ModelResponse:
        """Map a ``GenerateContentResponse`` payload onto the response contract.

        Raises:
            ProviderError: when the payload carries no candidate, which is what
                a prompt-level safety block and a malformed reply both look
                like. Neither is a response with empty output.
        """
        candidates = sequence_at(payload, "candidates")
        if not candidates or not isinstance(candidates[0], dict):
            raise self._no_candidate_error(payload)
        candidate: dict[str, Any] = candidates[0]
        finish_reason = _FINISH_REASONS.get(
            text_at(candidate, "finish_reason") or "", FinishReason.UNKNOWN
        )
        return ModelResponse(
            output_text=_candidate_text(candidate),
            provider=self.name,
            model=text_at(payload, "model_version") or self._config.model,
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
            provider_request_id=identifier_at(payload, "response_id"),
            raw=capped_raw(payload),
        )

    def _no_candidate_error(self, payload: dict[str, Any]) -> ProviderError:
        """Build the error for a reply that contains no candidate.

        A ``block_reason`` makes this a content-filter refusal of the PROMPT,
        which is terminal: retrying the identical prompt gets the identical
        refusal. Without one it is an unexplained empty reply, reported as
        UNKNOWN rather than as a server error, for the same reason - a retry
        would not change it.
        """
        feedback = dict_at(payload, "prompt_feedback")
        block_reason = text_at(feedback, "block_reason")
        if block_reason is not None:
            message = text_at(feedback, "block_reason_message")
            detail = f": {message}" if message else ""
            return self._error(
                f"the prompt was blocked before generation, block_reason={block_reason}{detail}",
                kind=ProviderErrorKind.CONTENT_FILTER,
                vendor_code=block_reason,
            )
        return self._error(
            "the response contained no candidates and no block reason",
            kind=ProviderErrorKind.UNKNOWN,
        )

    # -- errors ------------------------------------------------------------

    def _classify_error(self, exc: Exception) -> ProviderError:
        """Map a ``google.genai`` exception onto a typed provider error.

        This SDK has a much flatter hierarchy than the other two: one
        :class:`APIError` with ``ClientError`` and ``ServerError`` subclasses,
        each carrying ``.code`` (the HTTP status) and ``.status`` (the vendor's
        own status string, such as ``RESOURCE_EXHAUSTED``). The status code is
        therefore the primary signal, refined by the status string where that
        distinguishes a quota exhaustion from ordinary throttling.
        """
        genai = _import_genai()
        api_error = getattr(genai.errors, "APIError", None)
        if api_error is not None and isinstance(exc, api_error):
            status_code = getattr(exc, "code", None)
            if not isinstance(status_code, int):
                status_code = None
            vendor_status = getattr(exc, "status", None)
            vendor_code = vendor_status if isinstance(vendor_status, str) else None
            return ProviderError(
                scrub_message(getattr(exc, "message", None) or exc),
                kind=_api_error_kind(status_code, vendor_code),
                provider=self.name,
                status_code=status_code,
                vendor_code=vendor_code,
            )
        return ProviderError(
            scrub_message(exc),
            kind=_transport_kind(exc),
            provider=self.name,
        )


_VENDOR_STATUS_KINDS: dict[str, ProviderErrorKind] = {
    "UNAUTHENTICATED": ProviderErrorKind.AUTHENTICATION,
    "PERMISSION_DENIED": ProviderErrorKind.PERMISSION,
    "NOT_FOUND": ProviderErrorKind.NOT_FOUND,
    "INVALID_ARGUMENT": ProviderErrorKind.INVALID_REQUEST,
    "FAILED_PRECONDITION": ProviderErrorKind.INVALID_REQUEST,
    "RESOURCE_EXHAUSTED": ProviderErrorKind.RATE_LIMIT,
    "DEADLINE_EXCEEDED": ProviderErrorKind.TIMEOUT,
    "UNAVAILABLE": ProviderErrorKind.SERVER,
    "INTERNAL": ProviderErrorKind.SERVER,
    "CANCELLED": ProviderErrorKind.CANCELLED,
    "UNIMPLEMENTED": ProviderErrorKind.UNSUPPORTED,
}
"""Google's canonical status strings, mapped to the neutral kinds.

``RESOURCE_EXHAUSTED`` is deliberately RATE_LIMIT rather than QUOTA: Gemini
returns it for ordinary per-minute throttling, which is retryable, and treating
it as an exhausted billing quota would abandon a run that a short wait would
have completed.
"""


def _api_error_kind(status_code: int | None, vendor_code: str | None) -> ProviderErrorKind:
    """Classify an ``APIError`` from its HTTP status, refined by the vendor status."""
    if vendor_code is not None:
        kind = _VENDOR_STATUS_KINDS.get(vendor_code.upper())
        if kind is not None:
            return kind
    return status_kind(status_code)


def _transport_kind(exc: Exception) -> ProviderErrorKind:
    """Classify an exception raised below the SDK, such as a raw httpx failure."""
    import httpx  # noqa: PLC0415 - a core dependency, imported where it is used

    if isinstance(exc, httpx.TimeoutException):
        return ProviderErrorKind.TIMEOUT
    if isinstance(exc, httpx.TransportError):
        return ProviderErrorKind.CONNECTION
    return ProviderErrorKind.UNKNOWN


def _candidate_text(candidate: dict[str, Any]) -> str:
    """Concatenate the text of every non-thought part of a candidate.

    Parts flagged ``thought`` are the model's internal reasoning and are not the
    answer, so they are excluded from the output text while remaining visible in
    ``raw``. Including them would make an evaluator score the reasoning.
    """
    return "".join(
        part["text"]
        for part in sequence_at(dict_at(candidate, "content"), "parts")
        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
    )


def _usage(payload: dict[str, Any]) -> TokenUsage:
    """Map ``usage_metadata``, leaving every count the vendor omitted as ``None``.

    ``candidates_token_count`` is absent from a reply whose only output was
    reasoning or whose candidate was cut off immediately, so it stays ``None``
    rather than becoming a zero-token answer that costs nothing.
    """
    usage = payload.get("usage_metadata")
    if not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=optional_count(usage.get("prompt_token_count")),
        output_tokens=optional_count(usage.get("candidates_token_count")),
        total_tokens=optional_count(usage.get("total_token_count")),
        cached_input_tokens=optional_count(usage.get("cached_content_token_count")),
        reasoning_tokens=optional_count(usage.get("thoughts_token_count")),
    )


__all__ = [
    "GOOGLE_CREDENTIAL_ENV",
    "GOOGLE_EXTRA",
    "GOOGLE_IMPORT_NAME",
    "GoogleOptions",
    "GoogleProvider",
]
