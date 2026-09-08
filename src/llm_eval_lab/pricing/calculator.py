"""Turning token usage and a price table into a cost, or into an honest gap.

The rule this module exists to enforce: a missing price or a missing token
count produces ``priced=False`` with a stated reason, never a silent zero. A
fabricated zero is indistinguishable from a genuinely free call, and it
propagates into every total downstream.

``unpriced_reason`` carries a STABLE CODE, not a sentence. The value is
persisted on every case, is read back by the dashboard and the CLI, and is
grouped on to answer "how many cases went unpriced, and why". A prose message
containing the provider and model would make every case its own group and would
change meaning the first time somebody reworded it. The codes are the
``UNPRICED_*`` constants below; the surface that shows one to a human is
responsible for expanding it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Literal

from llm_eval_lab.models import CostBreakdown, PriceEntry, PriceTable, TokenUsage

TOKENS_PER_MTOK = Decimal(1_000_000)
"""Prices are quoted per million tokens."""

MONEY_QUANTUM = Decimal("0.0000000001")
"""Ten decimal places: the scale of the `Money` column, applied to every derived figure.

A cost that is SUMMED keeps whatever precision its inputs had. A cost that is
DIVIDED - a per-case mean - is quantized here, because the storage column is
``Numeric(20, 10)`` on PostgreSQL and would round it there and not on SQLite.
Rounding once, in the domain, is what keeps the two backends returning the same
number.
"""

UNPRICED_NO_PRICE_ENTRY = "no_price_entry"
"""The price table has no entry for this provider and model."""

UNPRICED_NO_TOKEN_USAGE = "no_token_usage"  # noqa: S105 - a billing reason code, not a credential
"""The provider reported no token counts, so there is nothing to multiply."""

UNPRICED_NO_RESPONSE = "no_response"
"""The case produced no response at all, so no request was billed."""

UNPRICED_PARTIAL_COVERAGE = "partial_coverage"
"""An aggregate over a mix of priced and unpriced units. See `CostRollup.coverage`."""


def _unpriced(
    table: PriceTable,
    reason: str,
    judge_cost: Decimal | None = None,
) -> CostBreakdown:
    """Build a cost breakdown that records why no cost could be computed.

    ``judge_cost`` is carried through rather than dropped. A judge's spend is
    known independently of whether the CANDIDATE model has a price entry, so
    discarding it here would silently zero out judge cost for exactly the models
    whose pricing is already unknown.
    """
    return CostBreakdown(
        currency=table.currency,
        input_cost=None,
        output_cost=None,
        total_cost=None,
        judge_cost=judge_cost,
        priced=False,
        unpriced_reason=reason,
        price_table_id=table.id,
        price_table_version=table.version,
        input_per_mtok=None,
        output_per_mtok=None,
    )


def _input_cost(entry: PriceEntry, usage: TokenUsage) -> Decimal:
    """Cost the input side, billing cached tokens at their own rate when quoted.

    ``cached_input_tokens`` is treated as a SUBSET of ``input_tokens``, which is
    how every vendor this project targets reports it: the cached count is part
    of the input total, not an extra charge alongside it. The uncached remainder
    is clamped at zero so a provider that reported a cached count larger than
    its input total produces a smaller bill rather than a negative one.
    """
    total_input = Decimal(usage.input_tokens or 0)
    cached = Decimal(usage.cached_input_tokens or 0)
    if entry.cached_input_per_mtok is None or cached == 0:
        return total_input / TOKENS_PER_MTOK * entry.input_per_mtok
    uncached = max(total_input - cached, Decimal(0))
    return (
        uncached / TOKENS_PER_MTOK * entry.input_per_mtok
        + cached / TOKENS_PER_MTOK * entry.cached_input_per_mtok
    )


def compute_cost(
    table: PriceTable,
    *,
    provider: str,
    model: str,
    usage: TokenUsage,
    judge_cost: Decimal | None = None,
) -> CostBreakdown:
    """Price one unit of work, or explain why it could not be priced.

    ``judge_cost`` is carried alongside and is never folded into
    ``total_cost``: model-graded evaluation is its own line item, and hiding it
    inside the generation cost makes a judge look free.
    """
    entry = table.lookup(provider, model)
    if entry is None:
        return _unpriced(table, UNPRICED_NO_PRICE_ENTRY, judge_cost)
    if usage.input_tokens is None or usage.output_tokens is None:
        return _unpriced(table, UNPRICED_NO_TOKEN_USAGE, judge_cost)

    input_cost = _input_cost(entry, usage)
    output_cost = Decimal(usage.output_tokens) / TOKENS_PER_MTOK * entry.output_per_mtok
    return CostBreakdown(
        currency=table.currency,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        judge_cost=judge_cost,
        priced=True,
        unpriced_reason=None,
        price_table_id=table.id,
        price_table_version=table.version,
        input_per_mtok=entry.input_per_mtok,
        output_per_mtok=entry.output_per_mtok,
    )


def billable_total(cost: CostBreakdown | None) -> Decimal:
    """Return the known spend a cost breakdown contributes to a running budget.

    An unpriced case contributes nothing, so a budget is never enforced against
    a number the system does not actually know. The caller reports how many
    cases were unpriced alongside the total.
    """
    if cost is None or not cost.priced:
        return Decimal(0)
    return (cost.total_cost or Decimal(0)) + (cost.judge_cost or Decimal(0))


@dataclass(frozen=True)
class CostRollup:
    """A summed cost together with how much of the population it actually covers.

    The coverage fraction is not decoration. A total over ninety priced cases
    out of a hundred is a real number and a useful one, but reading it as the
    run's cost overstates nothing and understates by ten percent, and only the
    fraction says which. Every surface that renders ``cost`` renders
    ``coverage`` beside it.
    """

    cost: CostBreakdown
    n_priced: int
    n_units: int

    @property
    def coverage(self) -> float | None:
        """Return the fraction of units that carried a known price, or None for no units."""
        return self.n_priced / self.n_units if self.n_units else None


def _sum_optional(values: Sequence[Decimal | None]) -> Decimal | None:
    """Sum the known values, or return None when none of them is known.

    ``None`` rather than ``Decimal(0)`` for an all-unknown set, for the same
    reason a missing price is not a zero: "nothing was priced" and "everything
    was free" are different facts.
    """
    known = [value for value in values if value is not None]
    return sum(known, Decimal(0)) if known else None


def aggregate_costs(
    costs: Sequence[CostBreakdown | None],
    *,
    price_table_id: str,
    price_table_version: str,
    currency: Literal["USD"] = "USD",
    n_units: int | None = None,
) -> CostRollup:
    """Sum per-unit costs into one breakdown, carrying its coverage.

    Only priced units contribute. An unpriced unit adds nothing rather than
    adding a zero, and is counted in the coverage denominator so the gap is
    visible instead of being absorbed.

    Args:
        costs: One entry per unit of work. ``None`` counts as unpriced.
        price_table_id: The run's price table identity, stamped onto the result.
        price_table_version: The version of that table.
        currency: The table's currency. Every entry must already agree with it.
        n_units: The coverage denominator. Defaults to ``len(costs)``; pass it
            explicitly when some units produced no cost object at all.

    Returns:
        The summed breakdown and the counts behind it.
    """
    priced = [cost for cost in costs if cost is not None and cost.priced]
    units = len(costs) if n_units is None else n_units
    fully_priced = bool(priced) and len(priced) == units

    if not priced:
        reason = _dominant_unpriced_reason(costs)
        summed = CostBreakdown(
            currency=currency,
            input_cost=None,
            output_cost=None,
            total_cost=None,
            judge_cost=None,
            priced=False,
            unpriced_reason=reason,
            price_table_id=price_table_id,
            price_table_version=price_table_version,
        )
        return CostRollup(cost=summed, n_priced=0, n_units=units)

    summed = CostBreakdown(
        currency=currency,
        input_cost=_sum_optional([cost.input_cost for cost in priced]),
        output_cost=_sum_optional([cost.output_cost for cost in priced]),
        total_cost=_sum_optional([cost.total_cost for cost in priced]),
        judge_cost=_sum_optional([cost.judge_cost for cost in priced]),
        priced=fully_priced,
        unpriced_reason=None if fully_priced else UNPRICED_PARTIAL_COVERAGE,
        price_table_id=price_table_id,
        price_table_version=price_table_version,
    )
    return CostRollup(cost=summed, n_priced=len(priced), n_units=units)


def _dominant_unpriced_reason(costs: Sequence[CostBreakdown | None]) -> str:
    """Return the most common reason a set of units could not be priced.

    Most common rather than first, because the first case of a run is not a
    more informative sample than the other ninety-nine, and because a single
    odd case should not name the whole run's gap.
    """
    tally: dict[str, int] = {}
    for cost in costs:
        reason = UNPRICED_NO_RESPONSE if cost is None else cost.unpriced_reason
        if reason is not None:
            tally[reason] = tally.get(reason, 0) + 1
    if not tally:
        return UNPRICED_NO_PRICE_ENTRY
    return max(sorted(tally), key=lambda reason: tally[reason])


def per_unit_cost(total: Decimal | None, units: int) -> Decimal | None:
    """Divide a known total by a unit count, quantized to the money scale.

    Returns ``None`` when the total is unknown or there are no units. A cost per
    zero cases is not zero; it is a question with no answer, and reporting it as
    ``0`` would put a free-looking number next to an empty run.
    """
    if total is None or units <= 0:
        return None
    quantized = (total / Decimal(units)).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_EVEN)
    # A quantized zero carries the exponent and renders as "0E-10", which reads
    # like a defect in a report. Its plain form is the same exact value.
    return Decimal(0) if quantized == 0 else quantized
