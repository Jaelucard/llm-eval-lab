"""Provider-facing request primitives shared by every layer.

Nothing here knows about HTTP, SQL or any vendor SDK. :class:`ProviderConfig`
in particular is designed so that persisting, logging or returning it over the
API verbatim can never leak a credential.
"""

from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
"""Any value expressible in JSON. Used by the `raw`, `metadata` and `params` fields."""

type UtcDatetime = Annotated[datetime, AwareDatetime]
"""A timezone-aware timestamp. Naive datetimes are refused at the boundary.

Every timestamp in the contract is documented as UTC, and this is what makes that
documentation enforceable. It matters beyond tidiness: PostgreSQL `TIMESTAMPTZ`
returns aware datetimes while SQLite returns naive ones, so a contract that accepted
both would let the same code produce different values on the two backends, and
comparing a stored naive value against an aware `utc_now()` raises `TypeError` at
runtime. Refusing naive input here gives the storage mapper an explicit obligation to
re-attach UTC on read. Deliberately not exported: it is a spelling, not a contract
symbol, so `__all__` and `EXPECTED_SURFACE` are unaffected.
"""

Role = Literal["system", "user", "assistant"]

_SECRETISH: tuple[str, ...] = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
)

_ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _find_secretish_key(value: JSONValue, path: str) -> tuple[str, str] | None:
    """Find the first secret-shaped key anywhere inside a JSON value.

    Walks nested dicts and lists to any depth, because `options` is a
    `dict[str, JSONValue]` and its values nest by construction. A top-level-only
    check would accept `{"headers": {"Authorization": "Bearer ..."}}`, which is
    the single most likely way a real credential reaches this object.

    Returns the dotted path to the offending key and the token that matched it,
    or `None` when the whole structure is clean.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            here = f"{path}.{key}"
            lowered = key.lower()
            token = next((t for t in _SECRETISH if t in lowered), None)
            if token is not None:
                return (here, token)
            found = _find_secretish_key(nested, here)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_secretish_key(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


class ChatMessage(BaseModel):
    """One turn of a conversation, in provider-neutral form."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str
    name: str | None = None


class GenerationParams(BaseModel):
    """Decoding parameters, normalized across providers.

    A provider translates these into its own vocabulary and drops what it does
    not support; `extra` carries vendor-specific knobs that have no neutral
    spelling.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: tuple[str, ...] = ()
    seed: int | None = None
    response_format: Literal["text", "json_object"] = "text"
    extra: dict[str, JSONValue] = Field(default_factory=dict)

    def merged_with(self, override: "GenerationParams | None") -> "GenerationParams":
        """Return a copy with every field the override explicitly set applied.

        Only fields present in ``override.model_fields_set`` win, so a case
        that never mentions ``temperature`` inherits the suite default rather
        than silently resetting it to ``None``. ``extra`` is merged key by key
        rather than replaced wholesale.
        """
        if override is None:
            return self
        data = self.model_dump()
        for name in override.model_fields_set:
            data[name] = getattr(override, name)
        if "extra" in override.model_fields_set:
            data["extra"] = {**self.extra, **override.extra}
        return GenerationParams.model_validate(data)


class ProviderConfig(BaseModel):
    """Fully describes how to reach a model.

    Contains NO secret material, by construction: only the NAME of the
    environment variable holding the credential is recorded, so this object is
    safe to persist, log and return over the API verbatim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)
    connect_timeout_s: float | None = Field(default=None, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    organization: str | None = None
    label: str | None = None
    options: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("options")
    @classmethod
    def _reject_secretish_option_keys(cls, value: dict[str, JSONValue]) -> dict[str, JSONValue]:
        """Reject option keys that look like a place someone put a credential.

        The search is recursive. A nested `{"headers": {"Authorization": ...}}`
        is exactly the shape a user reaches for against a self-hosted or
        OpenAI-compatible endpoint, and a top-level-only check would persist it.
        """
        found = _find_secretish_key(value, "options")
        if found is not None:
            location, token = found
            msg = (
                f"{location} contains {token!r}; ProviderConfig is persisted and logged "
                f"verbatim and must never hold secret material. Record the environment "
                f"variable NAME in api_key_env instead."
            )
            raise ValueError(msg)
        return value

    @field_validator("base_url")
    @classmethod
    def _reject_unusable_base_url(cls, value: str | None) -> str | None:
        """Require an http(s) URL that embeds no userinfo credentials."""
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in _ALLOWED_URL_SCHEMES:
            msg = f"base_url must use http or https, got {parsed.scheme or 'no'} scheme"
            raise ValueError(msg)
        if not parsed.netloc:
            msg = "base_url must include a host"
            raise ValueError(msg)
        if parsed.username or parsed.password:
            msg = (
                "base_url must not embed userinfo credentials; ProviderConfig is persisted and "
                "logged verbatim. Record the environment variable NAME in api_key_env instead."
            )
            raise ValueError(msg)
        return value


class ProviderRequest(BaseModel):
    """One fully-specified generation request handed to a provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    system: str | None = None
    params: GenerationParams = GenerationParams()
    request_id: str
    metadata: dict[str, str] = Field(default_factory=dict)
