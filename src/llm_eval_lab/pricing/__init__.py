"""Price table loading and cost calculation.

Prices are external data with a version and a content hash, and a lookup that
misses records ``priced=False`` with a reason rather than assuming zero.
"""

from llm_eval_lab.pricing.calculator import (
    MONEY_QUANTUM,
    UNPRICED_NO_PRICE_ENTRY,
    UNPRICED_NO_RESPONSE,
    UNPRICED_NO_TOKEN_USAGE,
    UNPRICED_PARTIAL_COVERAGE,
    CostRollup,
    aggregate_costs,
    billable_total,
    compute_cost,
    per_unit_cost,
)
from llm_eval_lab.pricing.loader import (
    DEFAULT_PRICE_TABLE_PATH,
    build_price_table,
    load_price_table,
)

__all__ = [
    "DEFAULT_PRICE_TABLE_PATH",
    "MONEY_QUANTUM",
    "UNPRICED_NO_PRICE_ENTRY",
    "UNPRICED_NO_RESPONSE",
    "UNPRICED_NO_TOKEN_USAGE",
    "UNPRICED_PARTIAL_COVERAGE",
    "CostRollup",
    "aggregate_costs",
    "billable_total",
    "build_price_table",
    "compute_cost",
    "load_price_table",
    "per_unit_cost",
]
