"""JSON documents for the CLI envelope and the HTTP API.

Every document here is built by dumping a frozen contract model in JSON mode,
so the key names in the output are the field names in the contract and there is
no second, hand-maintained mapping to fall out of step. ``Decimal`` survives as
its exact textual form rather than becoming a float, which is the same reason
the money column stores text.
"""

from collections.abc import Sequence
from decimal import Decimal

from llm_eval_lab.models import (
    AggregateMetrics,
    CaseResult,
    JSONValue,
    PriceTable,
)
from llm_eval_lab.utils.time import isoformat_z_optional


def metrics_document(metrics: AggregateMetrics) -> dict[str, JSONValue]:
    """Render one run's rollup as the JSON document the CLI and the API emit.

    A straight dump of the contract model. The keys a consumer relies on -
    ``pass_rate``, ``pass_rate_ci``, ``error_rate``, ``error_policy``,
    ``n_score_only``, the nested ``latency`` object with its ``low_confidence``
    tuple, both coverage ratios, the two cost-per figures and the three
    breakdowns - are therefore exactly the field names, and stay in step with
    the contract by construction.
    """
    document: dict[str, JSONValue] = metrics.model_dump(mode="json")
    return document


def _decimal_text(value: Decimal | None) -> str | None:
    """Render a money value as exact text, or None when it is unknown."""
    return None if value is None else str(value)


def case_row(result: CaseResult) -> dict[str, JSONValue]:
    """Flatten one case result into the row shape an export carries.

    Flat rather than nested, because this row is what becomes a CSV line and a
    spreadsheet column. The response body itself is deliberately absent: an
    export of a thousand cases is a summary table, and whole model outputs
    belong behind ``llm-eval show``.
    """
    response = result.response
    usage = response.usage if response is not None else None
    cost = result.cost
    return {
        "run_id": result.run_id,
        "case_id": result.case_id,
        "case_hash": result.case_hash,
        "status": result.status.value,
        "passed": result.passed,
        "score": result.score,
        "attempts": result.attempts,
        "category": result.category,
        "tags": ",".join(result.tags),
        "latency_ms": None if response is None else response.latency_ms,
        "total_latency_ms": None if response is None else response.total_latency_ms,
        "input_tokens": None if usage is None else usage.input_tokens,
        "output_tokens": None if usage is None else usage.output_tokens,
        "total_tokens": None if usage is None else usage.total_tokens,
        "total_cost": None if cost is None else _decimal_text(cost.total_cost),
        "judge_cost": None if cost is None else _decimal_text(cost.judge_cost),
        "priced": None if cost is None else cost.priced,
        "unpriced_reason": None if cost is None else cost.unpriced_reason,
        "error_kind": None if result.error is None else result.error.kind.value,
        "started_at": isoformat_z_optional(result.started_at),
        "completed_at": isoformat_z_optional(result.completed_at),
    }


def case_rows(results: Sequence[CaseResult]) -> list[dict[str, JSONValue]]:
    """Flatten every case result into export rows, in the order given."""
    return [case_row(result) for result in results]


def price_table_document(table: PriceTable) -> dict[str, JSONValue]:
    """Render a price table, including the content hash a run records.

    The hash is part of the document rather than a detail, because it is what
    lets somebody confirm that a historical run's costs came from this exact
    file and not from a later edit that kept the same version string.
    """
    document: dict[str, JSONValue] = table.model_dump(mode="json")
    return document


__all__ = ["case_row", "case_rows", "metrics_document", "price_table_document"]
