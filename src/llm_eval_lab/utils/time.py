"""Time helpers, so timezone handling has exactly one implementation.

Every timestamp in this project is timezone-aware UTC. SQLite hands back naive
datetimes regardless of what was written, so :func:`ensure_utc` exists to give
the storage mappers one obvious place to re-attach the timezone on read rather
than each of them inventing its own.
"""

import time
from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Return `value` as an aware UTC datetime.

    A naive value is assumed to already be UTC and is stamped as such, which is
    true for everything this project writes. An aware value in another zone is
    converted.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def ensure_utc_optional(value: datetime | None) -> datetime | None:
    """Return :func:`ensure_utc` of `value`, passing ``None`` through."""
    return None if value is None else ensure_utc(value)


def isoformat_z(value: datetime) -> str:
    """Render an instant as RFC 3339 with a trailing ``Z``.

    One spelling of a UTC timestamp across the whole project. Pydantic already
    emits ``Z`` for every timestamp nested inside a dumped model, so a surface
    that reached for ``isoformat()`` directly would put ``+00:00`` next to
    ``Z`` in one document and leave a consumer comparing them as strings to
    find them unequal.
    """
    return ensure_utc(value).isoformat().replace("+00:00", "Z")


def isoformat_z_optional(value: datetime | None) -> str | None:
    """Return :func:`isoformat_z` of `value`, passing ``None`` through."""
    return None if value is None else isoformat_z(value)


def monotonic_ms() -> float:
    """Return a monotonic clock reading in milliseconds.

    Monotonic rather than wall clock, so a measured duration is never distorted
    by an NTP step or a daylight-saving transition mid-run.
    """
    return time.monotonic() * 1000.0
