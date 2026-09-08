"""Price table contract and the sole definition of :class:`CostBreakdown`.

Prices live in an external, versioned data file rather than in code, and every
cost figure records the exact unit prices that produced it. A run therefore
stays interpretable after the vendor changes its pricing page, and a price
lookup that misses records ``priced=False`` instead of quietly assuming zero.
"""

import re
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict


class PriceEntry(BaseModel):
    """Unit prices for one provider/model pairing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    match: Literal["exact", "prefix", "regex"] = "exact"
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cached_input_per_mtok: Decimal | None = None
    effective_from: date | None = None
    notes: str | None = None


class PriceTable(BaseModel):
    """A versioned, content-addressed set of price entries."""

    model_config = ConfigDict(frozen=True)

    id: str
    version: str
    currency: Literal["USD"] = "USD"
    source: str | None = None
    models: tuple[PriceEntry, ...]
    content_hash: str

    def lookup(self, provider: str, model: str) -> PriceEntry | None:
        """Find the price entry for a model.

        Exact match wins; then the longest matching prefix; then the first
        matching regex, in declaration order. Returns ``None`` on a miss.
        NEVER assumes zero - the caller records ``priced=False``.

        Two properties the price-table author has to know. Regex entries are
        matched with ``re.search``, not ``re.fullmatch``, so ``model: "gpt"``
        also matches ``my-gpt-clone``; anchor the pattern to avoid that.
        ``effective_from`` is not consulted here, so a table holding two entries
        for one model at different dates resolves by declaration order - the
        loader is responsible for filtering by date before building the table.
        """
        candidates = [entry for entry in self.models if entry.provider == provider]

        exact = next(
            (e for e in candidates if e.match == "exact" and e.model == model),
            None,
        )
        if exact is not None:
            return exact

        prefixes = [e for e in candidates if e.match == "prefix" and model.startswith(e.model)]
        if prefixes:
            return max(prefixes, key=lambda e: len(e.model))

        return next(
            (e for e in candidates if e.match == "regex" and re.search(e.model, model) is not None),
            None,
        )


class CostBreakdown(BaseModel):
    """What one unit of work cost, and the exact prices that produced it.

    ``judge_cost`` is tracked separately and is never folded into
    ``total_cost``: model-graded evaluation is a distinct line item, and
    hiding it inside the generation cost makes a judge look free.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    currency: Literal["USD"] = "USD"
    input_cost: Decimal | None
    output_cost: Decimal | None
    total_cost: Decimal | None
    judge_cost: Decimal | None = None
    priced: bool
    unpriced_reason: str | None = None
    price_table_id: str
    price_table_version: str
    input_per_mtok: Decimal | None = None
    output_per_mtok: Decimal | None = None
