"""Secret scrubbing for free-form dictionaries. One rule, two consumers.

The logging processor scrubs every event before it reaches a sink; the
persistence mappers scrub ``GenerationParams.extra``, ``EvaluatorSpec.params``
and a provider's ``raw`` payload before they reach a database file. Those two
live at opposite ends of the layered dependency contract - ``observability`` at
the bottom, ``storage`` near the top - and the contract gives them no common
package to share code through: ``storage`` may not import ``observability``,
and the layers contract ranks ``observability`` and ``utils`` as independent
siblings that may not import each other either.

So this module sits at the package root, OUTSIDE the layer stack, next to
``settings.py`` and for the same reason: it is a cross-cutting policy rather
than a layer. Duplicating the rule in two places was the alternative, and a
security control with two implementations has, in practice, one implementation
and one bug.

Why this is done in the persistence and logging paths rather than in a model
validator: keys like ``api_key_env`` are LEGITIMATE contract fields - recording
the NAME of a credential variable is the whole design - so a validator that
rejected secret-shaped keys would reject the correct configuration along with
the dangerous one.

The second policy here is URL redaction. A key-shaped VALUE also hides inside a
connection string: ``postgresql+asyncpg://admin:hunter2@host/db`` is a plain
``str`` on ``Settings``, so key-name matching never sees it. :func:`redact_url`
is the one implementation, and :func:`redact_setting_value` is the display rule
every configuration surface goes through, so a future API endpoint cannot
re-derive a weaker version of it.

A connection string hides a credential in TWO places, not one. The userinfo is
the obvious half; the query string is the other, and several drivers put it
there rather than in the userinfo - ``?password=``, libpq's ``?sslpassword=``,
and the ``?token=`` or ``?api_key=`` style used by hosted databases.
:func:`redact_url` covers both, and it decides which query keys are secret with
the very same :func:`is_secret_key` used above, so there is one list of
secret-shaped names in this project rather than two that drift apart.
"""

import re
from collections.abc import MutableMapping
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from pydantic import SecretStr

REDACTED = "[redacted]"
"""What a scrubbed value is replaced with. One spelling, so tests can assert on it."""

MAX_SCRUB_DEPTH = 12
"""Recursion bound, so a deeply nested structure cannot turn scrubbing into a hang."""

SECRET_KEY_PATTERN = re.compile(
    r"""
    api[-_]?key                                  # api_key, apiKey, x-api-key
    | apikey
    | secret
    | password | passwd
    | authorization | bearer | credential
    | (?:access|auth|bearer|refresh|session|id|api)[-_]?token   # access_token, api-token
    | ^token$                                                   # the bare key "token"
    """,
    re.IGNORECASE | re.VERBOSE,
)
"""Key names whose values never reach a log sink or a database row.

Most alternatives match as a SUBSTRING on purpose: `openai_api_key`,
`x-api-key` and `headers.Authorization` all have to be caught, and an
exact-name list misses every one of them.

`token` is the exception, and it is why this pattern is spelled out rather than
being a short list of scary words. A bare substring match on `token` also
matches `max_output_tokens`, `input_tokens`, `total_tokens` and
`token_usage_coverage` - every token COUNT in the contract. Redacting those is
not a harmless over-reach: the counts are persisted, so the next read fails
validation and the run becomes unloadable. `token` therefore matches only as the
whole key, or with a credential-ish qualifier in front of it.
"""

_SAFE_KEY_SUFFIXES: tuple[str, ...] = ("_env", "_env_var", "_name")
"""Suffixes marking a key that NAMES a credential rather than holding one.

`api_key_env` is the point of `ProviderConfig`: it records an environment
variable name, which is safe, persisted deliberately, and often the single most
useful field in a log line about a failed authentication.
"""


def is_secret_key(key: str) -> bool:
    """Report whether a key's value must be redacted."""
    lowered = key.lower()
    if lowered.endswith(_SAFE_KEY_SUFFIXES):
        return False
    return SECRET_KEY_PATTERN.search(lowered) is not None


def scrub_value(value: Any, *, depth: int = 0) -> Any:
    """Return `value` with every secret-shaped key's value replaced.

    Walks mappings, lists and tuples to :data:`MAX_SCRUB_DEPTH`. Anything deeper
    is replaced wholesale rather than trusted: an unbounded walk over an
    attacker-influenced structure is its own denial of service.
    """
    if depth >= MAX_SCRUB_DEPTH:
        return REDACTED
    if isinstance(value, MutableMapping):
        return {
            key: (
                REDACTED
                if isinstance(key, str) and is_secret_key(key)
                else scrub_value(item, depth=depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_value(item, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_value(item, depth=depth + 1) for item in value)
    return value


def scrub_mapping(value: dict[str, Any]) -> dict[str, Any]:
    """Return a scrubbed copy of a mapping, typed as a mapping for callers."""
    scrubbed = scrub_value(dict(value))
    if not isinstance(scrubbed, dict):  # pragma: no cover - scrub_value preserves mappings
        return {}
    return scrubbed


QUERY_REDACTED = "***"
"""What a redacted QUERY-STRING value is replaced with.

Deliberately not :data:`REDACTED`. A query value has to survive URL encoding,
and ``[redacted]`` would either be written with invalid unencoded brackets or
come back as the unreadable ``%5Bredacted%5D``. ``***`` needs no encoding, so
the redacted URL stays a valid, readable URL. The userinfo has no such
constraint and keeps the project's usual marker.
"""


def _redact_userinfo(netloc: str, *, redact_username: bool = False) -> str | None:
    """Return `netloc` with its credentials replaced, or None when there is none.

    Split on the LAST ``@`` rather than the first, so an unencoded ``@`` inside
    a password does not truncate the redaction and leak its tail. The host's
    letter case and the port pass through untouched.

    With `redact_username`, the whole userinfo - name included - becomes
    ``[redacted]@``. That is a stricter answer than the default, which keeps a
    bare username on the theory that a printed configuration should still say
    who it connects as; a caller building an error message from a URL that
    failed to even PARSE has no such theory available and asks for the
    stricter one instead.
    """
    if "@" not in netloc:
        return None
    userinfo, _, hostpart = netloc.rpartition("@")
    if redact_username:
        if not userinfo:
            return None
        return f"{REDACTED}@{hostpart}"
    if ":" not in userinfo:
        # A user with no password. Nothing to hide.
        return None
    user, _, _password = userinfo.partition(":")
    return f"{user}:{REDACTED}@{hostpart}"


def _redact_query(query: str) -> str | None:
    """Return `query` with every secret-shaped parameter's value replaced.

    Rewrites the raw query string pair by pair rather than parsing and
    re-encoding it, so every parameter that is NOT a secret survives byte for
    byte and a printed URL stays the one the operator configured.

    Returns ``None`` when nothing matched, which is what lets :func:`redact_url`
    hand back the original string untouched instead of a re-joined equivalent.
    """
    if not query:
        return None
    redacted: list[str] = []
    changed = False
    for pair in query.split("&"):
        key, separator, _value = pair.partition("=")
        if separator and is_secret_key(unquote(key)):
            redacted.append(f"{key}={QUERY_REDACTED}")
            changed = True
        else:
            redacted.append(pair)
    return "&".join(redacted) if changed else None


def redact_url(value: str, *, redact_username: bool = False) -> str:
    """Return `value` with every credential in it replaced.

    Two hiding places, both covered. The userinfo
    (``postgresql+asyncpg://admin:hunter2@host/db``) becomes
    ``postgresql+asyncpg://admin:[redacted]@host/db``. A secret-shaped query
    parameter (``?password=``, ``?sslpassword=``, ``?token=``, ``?api_key=``,
    and anything else :func:`is_secret_key` recognises, matched
    case-insensitively) has its value replaced with :data:`QUERY_REDACTED`.
    Which query keys count is decided by the same function the logging and
    persistence scrubbers use, so this project has one list of secret-shaped
    names rather than two.

    The scheme, user, host, port, path and every non-secret parameter survive
    intact, because they are what makes a printed configuration useful and none
    of them is the secret.

    `redact_username`, when set, replaces the whole userinfo - the name along
    with the password - with ``[redacted]@``. Default ``False`` keeps the
    bare username, which is what every existing caller (``llm-eval config
    show``, ``llm-eval db upgrade``) wants for a URL that opened fine. A
    caller building a message around a URL that just FAILED to open - the
    engine's own error path - has no such use for the name and asks for the
    stricter form instead.

    A string that is not a URL, or a URL with nothing to hide, is returned
    UNCHANGED rather than re-joined: a SQLite path carries no credential and
    mangling it would make the one setting people actually need to read
    unreadable.

    Raises nothing. A URL that will not parse is replaced wholesale with
    :data:`REDACTED`: failing closed is the only safe direction for a function
    whose job is to stop a value being printed.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        return REDACTED
    if parts.netloc and "@" in f"{parts.path}{parts.query}{parts.fragment}":
        # An authority followed by an `@` somewhere after it means the split
        # may have landed in the wrong place: an unencoded `/`, `?` or `#`
        # inside a credential ends the netloc early and pushes the rest of the
        # credential, together with the real `@host`, past it, where the
        # userinfo redaction below cannot reach it. Nothing after the scheme
        # can be trusted at that point, so fail closed rather than print the
        # tail. The truncated authority can look like anything - a bare token
        # used as the username leaves no colon behind - so the only reliable
        # signal is that an authority exists at all. A SQLite URL has none, so
        # a path like `/tmp/a@b.db` passes through. The cost is that an
        # ordinary URL carrying an `@` after its host, such as an email
        # address in a query string, is redacted wholesale too; that is the
        # safe direction for a function whose only job is redaction.
        return f"{parts.scheme}://{REDACTED}"
    netloc = _redact_userinfo(parts.netloc, redact_username=redact_username)
    query = _redact_query(parts.query)
    if netloc is None and query is None:
        return value
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc if netloc is None else netloc,
            parts.path,
            parts.query if query is None else query,
            parts.fragment,
        )
    )


_URL_TOKEN_PATTERN = re.compile(r"""[A-Za-z][A-Za-z0-9+.\-]*://[^\s"'<>]*""")
"""Matches a URL-shaped token anywhere inside free-form text.

Deliberately looser than a real URL grammar: it exists to catch a connection
string a DRIVER re-formatted into its own error message (a different
argument order, a filled-in default port), not to validate one. Overmatching
a little at the edges is the safe direction for something whose only job is
redaction.
"""


def redact_error_text(text: str, *, url: str | None = None) -> str:
    """Return `text` with every credential belonging to `url` scrubbed out of it.

    Sanitizes a WRAPPED exception's own message, which can carry the very
    credential the caller is trying to keep out of a raised error
    independently of anything the caller writes: a database driver's failure
    text often echoes back exactly what it failed to connect to.

    Two passes. First, when the offending `url` is known, its password and
    username are cut out of `text` by exact substring match - byte for byte,
    regardless of how the driver worded the rest of the message - and so is
    the URL as a whole, in case the driver quoted it verbatim. Second,
    whatever URL-shaped token remains anywhere in `text` (the driver's own
    reformatting, or an unrelated URL) is passed through :func:`redact_url`
    with `redact_username` on, since a scrubbed error message has no use for
    a name attached to a database that could not be reached.

    Empty or `None` for `url` skips the first pass and still runs the second,
    so this is safe to call even when the caller has no URL to key off.

    The exact-match pass is a plain substring replacement, so a very short or
    very common password or username (``"a"``, ``"test"``, a value that also
    occurs inside the host name) will also blank unrelated fragments of the
    driver's text. That costs some readability and never leaks anything,
    which is the right trade for a function whose only job is redaction.
    """
    sanitized = text
    if url:
        sanitized = sanitized.replace(url, REDACTED)
        try:
            parts = urlsplit(url)
        except ValueError:
            parts = None
        if parts is not None and "@" in parts.netloc:
            userinfo = parts.netloc.rpartition("@")[0]
            user, _, password = userinfo.partition(":")
            if password:
                sanitized = sanitized.replace(password, REDACTED)
            if user:
                sanitized = sanitized.replace(user, REDACTED)
    return _URL_TOKEN_PATTERN.sub(
        lambda match: redact_url(match.group(0), redact_username=True), sanitized
    )


def looks_like_url(value: str) -> bool:
    """Report whether a string is shaped like a URL carrying a netloc."""
    return "://" in value


def redact_setting_value(value: Any) -> Any:
    """Return one configuration value rendered safely for display.

    The single display rule for every surface that prints configuration.

    A :class:`~pydantic.SecretStr` becomes ``None``: the caller reports the
    environment variable's NAME and whether it is set, and never a value, not
    even a partially masked one. A mask still leaks the length, the prefix and
    usually enough of the tail to identify a key.

    A string that looks like a URL goes through :func:`redact_url`. A sequence
    is rendered element by element under the same rule, because ``cors_origins``
    is a list of URLs. Everything else is returned as it is.
    """
    if isinstance(value, SecretStr):
        return None
    if isinstance(value, str):
        return redact_url(value) if looks_like_url(value) else value
    if isinstance(value, (list, tuple, set, frozenset)):
        return sorted(str(redact_setting_value(item)) for item in value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return redact_setting_value(str(value))
