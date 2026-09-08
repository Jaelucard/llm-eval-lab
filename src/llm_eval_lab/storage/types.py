"""Column types that keep the schema portable between SQLite and PostgreSQL.

Two problems this module solves.

**Money must stay exact.** A cost of ``Decimal("0.0000012345")`` written to a
``Float`` column comes back as something else. :class:`Money` stores the
decimal's exact textual form in a ``String`` on SQLite and uses a real
``Numeric(20, 10)`` on PostgreSQL, so the value that comes back is the value
that went in on both backends.

**JSON should be indexable where the backend can do it.** ``JSON_TYPE`` is
plain ``JSON`` on SQLite and ``JSONB`` on PostgreSQL, which costs nothing today
and leaves the door open to a JSON index later without a type migration.
"""

from decimal import Decimal, InvalidOperation
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.interfaces import Dialect

MONEY_STRING_LENGTH = 40
"""Wide enough for any plausible cost, including a long fractional tail."""

MONEY_PRECISION = 20
MONEY_SCALE = 10
"""PostgreSQL `Numeric` precision and scale. Ten decimal places bounds per-token pricing."""


class Money(sa.types.TypeDecorator[Decimal]):
    """A `Decimal` column that is exact on every supported backend.

    On SQLite the value is stored as its own textual form, because SQLite has
    no decimal type and its ``REAL`` is a binary double. On PostgreSQL the
    variant below swaps in a native ``NUMERIC``.
    """

    impl = sa.String(MONEY_STRING_LENGTH)
    cache_ok = True

    def process_bind_param(self, value: Decimal | None, dialect: Dialect) -> str | None:
        """Render a Decimal for storage, preserving every significant digit."""
        del dialect
        return None if value is None else str(value)

    def process_result_value(self, value: Any, dialect: Dialect) -> Decimal | None:
        """Reconstruct a Decimal from storage.

        Raises:
            ValueError: when the stored text is not a decimal. A corrupt money
                column is a bug worth surfacing, not a value worth guessing at.
        """
        del dialect
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except InvalidOperation as exc:
            msg = f"stored money value {value!r} is not a decimal"
            raise ValueError(msg) from exc


MONEY_TYPE = Money().with_variant(
    sa.Numeric(MONEY_PRECISION, MONEY_SCALE, asdecimal=True), "postgresql"
)
"""Exact decimal money: text on SQLite, native `NUMERIC` on PostgreSQL."""

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB, "postgresql")
"""Structured payloads: `JSON` on SQLite, `JSONB` on PostgreSQL."""

UTC_DATETIME = sa.DateTime(timezone=True)
"""Every timestamp column. SQLite drops the zone, so mappers re-attach UTC on read."""
