"""Shared machinery for providers that talk to a real endpoint.

Five adapters need the same four things: one timed attempt, an explicit
exception-to-:class:`~llm_eval_lab.models.ProviderErrorKind` mapping, a vendor
payload that is scrubbed and size-capped before it travels any further, and
token counts that stay ``None`` when the vendor reported none. Writing those
four things five times would give this project five subtly different answers to
"what does a missing token count mean", which is the one question it exists to
answer honestly.

**One attempt, measured around the call.** :meth:`RemoteProvider.generate` is a
template method: it stamps ``started_at``, runs the subclass's ``_invoke``
exactly once, stamps ``completed_at``, and hands the payload to the subclass's
``_normalize``. It never loops. Retries, backoff and rate-limit pacing belong to
the runner (decision D4), and every adapter additionally disables its SDK's own
retry loop, so the ``attempts`` a run records is the number of requests that
actually left the machine.

**Latency is measured on a monotonic clock**, so an NTP step mid-run cannot
produce a negative or wildly inflated duration, while ``started_at`` and
``completed_at`` remain wall-clock UTC because those are what a human reads.

**The payload crossing ``_invoke`` -> ``_normalize`` is a plain dict**, obtained
from the SDK response object with ``model_dump(mode="json")`` (or, for Ollama,
straight off the wire). That is deliberate and it is what makes normalization
testable: a recorded JSON fixture *is* a payload, so the whole mapping from
vendor fields to :class:`~llm_eval_lab.models.ModelResponse` can be exercised
with no SDK object, no client, no credential and no socket. It is also exactly
what ends up in ``raw``, so what a test asserts on is what a run stores.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import Any, ClassVar

from pydantic import SecretStr

from llm_eval_lab.models import (
    JSONValue,
    ModelResponse,
    ProviderAuthError,
    ProviderConfig,
    ProviderError,
    ProviderErrorKind,
    ProviderRequest,
)
from llm_eval_lab.providers.base import BaseProvider
from llm_eval_lab.redaction import scrub_value
from llm_eval_lab.utils.json import canonical_json_bytes
from llm_eval_lab.utils.time import monotonic_ms, utc_now

MAX_RAW_BYTES = 65536
"""Provider-side cap on the vendor payload carried in ``ModelResponse.raw``.

Matches ``settings.max_raw_bytes``. The storage mapper caps again on the way to
a database row and that cap remains authoritative; this one exists so a vendor
reply carrying a large embedded blob is not held in memory for every in-flight
case of a run before storage ever sees it.
"""

MAX_ERROR_MESSAGE_CHARS = 2000
"""Cap on a vendor error message. Long vendor bodies are truncated, not stored whole."""

RAW_TRUNCATED_KEY = "_truncated"
"""Marker key set on a payload that was replaced because it exceeded the cap."""

_SECRET_TEXT_PATTERN = re.compile(
    "|".join(
        (
            # Assembled from fragments so the file itself contains no string a
            # credential scanner would flag, and so the repository-wide grep for
            # leaked keys stays free of false positives from this module.
            "sk" + r"-[A-Za-z0-9_\-]{16,}",
            "AIz" + r"a[0-9A-Za-z_\-]{30,}",
            r"\bBearer\s+[A-Za-z0-9._\-]{8,}",
            r"\bx-api-key:\s*\S+",
        )
    ),
    re.IGNORECASE,
)
"""Credential shapes that occasionally appear inside a vendor's own error text.

A provider error message reaches structured logs, the CLI, a stored row and an
HTTP problem detail. A vendor that echoes the offending ``Authorization`` header
back in a 401 body would otherwise put a live key in all four.
"""


def scrub_message(text: object) -> str:
    """Return a vendor message safe to log, persist and return over the API.

    Credential-shaped substrings are replaced and the result is truncated. The
    text is never parsed for meaning: classification is done from exception
    types and status codes, never from message wording.
    """
    rendered = text if isinstance(text, str) else str(text)
    redacted = _SECRET_TEXT_PATTERN.sub("[redacted]", rendered)
    if len(redacted) <= MAX_ERROR_MESSAGE_CHARS:
        return redacted
    return redacted[:MAX_ERROR_MESSAGE_CHARS] + "... [truncated]"


def capped_raw(payload: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Return `payload` scrubbed of secret-shaped keys and capped in size.

    An oversized payload is replaced wholesale by a marker recording its size,
    rather than being trimmed to a prefix: half a JSON document is not a JSON
    document, and a consumer that could not tell the difference would be
    parsing garbage.
    """
    scrubbed: dict[str, JSONValue] = scrub_value(dict(payload))
    try:
        size = len(canonical_json_bytes(scrubbed))
    except (TypeError, ValueError):
        return {RAW_TRUNCATED_KEY: True, "reason": "payload is not JSON-serializable"}
    if size <= MAX_RAW_BYTES:
        return scrubbed
    return {
        RAW_TRUNCATED_KEY: True,
        "reason": f"payload exceeded {MAX_RAW_BYTES} bytes",
        "bytes": size,
    }


def optional_count(value: object) -> int | None:
    """Return a non-negative token count, or ``None`` when the vendor reported none.

    The whole point of this function is the ``None``. A vendor that omits
    ``prompt_eval_count`` has told us nothing about the prompt; defaulting that
    to ``0`` would turn silence into the claim that the request was free, and
    that claim would then be multiplied by a unit price and reported as a cost.
    Booleans are rejected rather than coerced, because ``True`` is an ``int``
    in Python and would otherwise arrive as a token count of one.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def parse_retry_after(value: object, *, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header into seconds, or return ``None``.

    RFC 9110 permits both a delay in seconds and an HTTP date; vendors send
    both, so both are read. An unparseable or past-dated value yields ``None``
    rather than a guess, which leaves the runner's own backoff in charge.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        pass
    else:
        return seconds if seconds >= 0.0 else None
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        return None
    delta = (when - (now or utc_now())).total_seconds()
    return max(0.0, delta)


def header_value(headers: object, name: str) -> str | None:
    """Read one header from an SDK response's header mapping, defensively.

    The header container differs between SDKs and between an error raised
    before a response existed and one raised after, so this reads what is there
    and returns ``None`` for everything else rather than raising a second,
    less informative exception on the failure path.
    """
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    try:
        found = getter(name)
    except (TypeError, ValueError, AttributeError):
        return None
    return None if found is None else str(found)


def identifier_at(payload: Mapping[str, Any], key: str) -> str | None:
    """Return a vendor request id, treating an empty string as no id at all.

    Some OpenAI-compatible servers send ``"id": ""`` rather than omitting the
    field. Storing that would put an empty string where the contract means
    "this vendor gave us nothing to correlate against", and every consumer
    would then need its own truthiness check.
    """
    return text_at(payload, key) or None


def dict_at(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    """Follow a chain of keys through nested mappings, yielding ``{}`` on any miss.

    Vendor payloads nest their token details two and three levels deep and omit
    whole branches when a feature was not used, so every read of a nested field
    would otherwise need its own ``None`` check.
    """
    current: Mapping[str, Any] = payload
    for key in keys:
        found = current.get(key)
        if not isinstance(found, Mapping):
            return {}
        current = found
    return current


def sequence_at(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    """Return a list-valued field, or an empty sequence when it is absent."""
    found = payload.get(key)
    return found if isinstance(found, list) else ()


def text_at(payload: Mapping[str, Any], key: str) -> str | None:
    """Return a string-valued field, or ``None`` when it is absent or not a string."""
    found = payload.get(key)
    return found if isinstance(found, str) else None


def build_timeout(config: ProviderConfig) -> Any:
    """Build an httpx timeout honouring a separate connect budget when one is set.

    Lives here rather than in one adapter because every client this project
    builds has to read ``connect_timeout_s`` the same way. A provider that
    passed a bare float instead would silently ignore the field, giving one
    configuration two meanings depending on which vendor served it.
    """
    import httpx  # noqa: PLC0415 - a core dependency, imported where it is used

    connect = config.connect_timeout_s or config.timeout_s
    return httpx.Timeout(config.timeout_s, connect=connect)


@dataclass(frozen=True)
class AttemptTiming:
    """When one provider attempt started and finished, and how long it took.

    ``latency_ms`` comes from a monotonic clock rather than from subtracting the
    two wall-clock stamps, so it is a duration even when the system clock moves.
    """

    started_at: datetime
    completed_at: datetime
    latency_ms: float


class RemoteProvider[PayloadT: Mapping[str, Any]](BaseProvider, ABC):
    """A provider that performs exactly one timed request against a real endpoint.

    Subclasses implement three methods and nothing else:

    ``_invoke``
        Perform one request and return the vendor payload as a plain mapping.
    ``_normalize``
        Map that payload onto a :class:`~llm_eval_lab.models.ModelResponse`.
    ``_classify_error``
        Map one vendor exception onto a typed
        :class:`~llm_eval_lab.models.ProviderError`.
    """

    name: ClassVar[str]

    credential_env: ClassVar[str | None] = None
    """The environment variable NAME this provider defaults to. Never a value.

    Used only to name the variable in a failure message, so an operator who has
    not configured ``api_key_env`` is still told exactly what to set.
    """

    def __init__(self, config: ProviderConfig, credential: SecretStr | None = None) -> None:
        """Bind the configuration and the resolved credential.

        The credential is held in a :class:`~pydantic.SecretStr` and is read
        exactly once, when the vendor client is constructed. Constructing a
        provider deliberately imports no vendor SDK and opens no connection, so
        the normalization logic is reachable in a test without either.
        """
        super().__init__(config)
        self._credential = credential

    def _api_key(self) -> str | None:
        """Return the credential value for handing to a vendor client constructor."""
        return None if self._credential is None else self._credential.get_secret_value()

    def _require_api_key(self) -> str:
        """Return the credential, refusing to build a client without one.

        Two things depend on this. First, a missing key becomes an
        ``AUTHENTICATION`` failure naming the variable to set, rather than
        whatever the vendor SDK happens to raise - which is a ``ValueError`` on
        one SDK and a ``TypeError`` from inside the request on another, and both
        of which would be reported as an unexplained ``UNKNOWN``.

        Second, and less obviously, every vendor SDK here reads the environment
        ITSELF when handed ``api_key=None``: verified at the pinned versions,
        all three pick up their own variable from ``os.environ``. That would put
        a second, undeclared environment reader in the project beside
        ``settings.read_credential`` (decision D8), and would let a run succeed
        against a credential the recorded ``ProviderConfig`` never named.
        Refusing ``None`` here is what makes that path unreachable.

        Raises:
            ProviderAuthError: when no credential was resolved for this provider.
        """
        key = self._api_key()
        if key is None:
            named = self._config.api_key_env or self.credential_env
            where = f"set {named} in the environment" if named else "configure api_key_env"
            msg = (
                f"no credential is configured for the {self.name} provider; {where}, "
                f"or name a different variable with api_key_env"
            )
            raise ProviderAuthError(msg, provider=self.name)
        return key

    async def generate(self, request: ProviderRequest) -> ModelResponse:
        """Perform exactly one attempt and return the normalized response.

        Raises:
            ProviderError: when the attempt failed. Every vendor exception is
                mapped to a typed kind by ``_classify_error``; nothing is
                swallowed and nothing is retried here.
        """
        started_at = utc_now()
        start_ms = monotonic_ms()
        try:
            payload = await self._invoke(request)
        except ProviderError:
            # Already classified, by this provider or by the credential layer.
            raise
        except Exception as exc:
            # Deliberately broad, and deliberately not a swallow: every vendor
            # exception is re-raised as a typed ProviderError so the runner sees
            # one error type with a known retryability, and `_classify_error`
            # has an explicit UNKNOWN fallback rather than a silent pass.
            raise self._classify_error(exc) from exc
        timing = AttemptTiming(
            started_at=started_at,
            completed_at=utc_now(),
            latency_ms=max(0.0, monotonic_ms() - start_ms),
        )
        return self._normalize(payload, timing)

    def _error(
        self,
        message: str,
        *,
        kind: ProviderErrorKind,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        vendor_code: str | None = None,
    ) -> ProviderError:
        """Build a typed provider error carrying a scrubbed message."""
        return ProviderError(
            scrub_message(message),
            kind=kind,
            provider=self.name,
            status_code=status_code,
            retry_after_s=retry_after_s,
            vendor_code=vendor_code,
        )

    @abstractmethod
    async def _invoke(self, request: ProviderRequest) -> PayloadT:
        """Perform one request and return the vendor payload as a plain mapping."""

    @abstractmethod
    def _normalize(self, payload: PayloadT, timing: AttemptTiming) -> ModelResponse:
        """Map one vendor payload onto the normalized response contract."""

    @abstractmethod
    def _classify_error(self, exc: Exception) -> ProviderError:
        """Map one vendor exception onto a typed provider error."""


_STATUS_KINDS: dict[int, ProviderErrorKind] = {
    HTTPStatus.UNAUTHORIZED: ProviderErrorKind.AUTHENTICATION,
    HTTPStatus.FORBIDDEN: ProviderErrorKind.PERMISSION,
    HTTPStatus.NOT_FOUND: ProviderErrorKind.NOT_FOUND,
    HTTPStatus.REQUEST_TIMEOUT: ProviderErrorKind.TIMEOUT,
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: ProviderErrorKind.CONTEXT_LENGTH,
    HTTPStatus.TOO_MANY_REQUESTS: ProviderErrorKind.RATE_LIMIT,
}
"""HTTP statuses whose meaning is more specific than "client error" or "server error"."""


def status_kind(status_code: int | None) -> ProviderErrorKind:
    """Classify an HTTP status code when no more specific mapping applies.

    The last resort, used for a status an SDK did not give its own exception
    class. Every adapter prefers its SDK's typed exceptions and reaches this
    only for the residue.
    """
    if status_code is None:
        return ProviderErrorKind.UNKNOWN
    specific = _STATUS_KINDS.get(status_code)
    if specific is not None:
        return specific
    if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return ProviderErrorKind.SERVER
    if status_code >= HTTPStatus.BAD_REQUEST:
        return ProviderErrorKind.INVALID_REQUEST
    return ProviderErrorKind.UNKNOWN


__all__ = [
    "MAX_ERROR_MESSAGE_CHARS",
    "MAX_RAW_BYTES",
    "RAW_TRUNCATED_KEY",
    "AttemptTiming",
    "RemoteProvider",
    "build_timeout",
    "capped_raw",
    "dict_at",
    "header_value",
    "identifier_at",
    "optional_count",
    "parse_retry_after",
    "scrub_message",
    "sequence_at",
    "status_kind",
    "text_at",
]
