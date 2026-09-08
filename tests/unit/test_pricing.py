"""Cost calculation, and the rule that an unknown price is never a zero.

A fabricated zero is the worst possible failure here, because it is
indistinguishable from a genuinely free call and it propagates into every total
downstream. Several of these tests exist only to assert the absence of
``Decimal("0")``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from llm_eval_lab.models import CostBreakdown, PriceEntry, PriceTable, PricingError, TokenUsage
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

if TYPE_CHECKING:
    from pathlib import Path

TABLE_ID = "test"
TABLE_VERSION = "1"


def table() -> PriceTable:
    """A small price table: one exact entry, one prefix entry, one cached rate."""
    return PriceTable(
        id=TABLE_ID,
        version=TABLE_VERSION,
        models=(
            PriceEntry(
                provider="fake",
                model="fake-1",
                input_per_mtok=Decimal("1.50"),
                output_per_mtok=Decimal("6.00"),
            ),
            PriceEntry(
                provider="fake",
                model="cached-",
                match="prefix",
                input_per_mtok=Decimal("10.00"),
                output_per_mtok=Decimal("20.00"),
                cached_input_per_mtok=Decimal("1.00"),
            ),
        ),
        content_hash="sha256:" + "f" * 64,
    )


def usage(input_tokens: int | None = 1_000_000, output_tokens: int | None = 500_000) -> TokenUsage:
    """Build token usage at a round million so the arithmetic is readable."""
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


# --- the unknown-is-null rule ---------------------------------------------


def test_a_missing_price_entry_is_unpriced_and_never_zero() -> None:
    cost = compute_cost(table(), provider="fake", model="unknown-model", usage=usage())

    assert cost.priced is False
    assert cost.total_cost is None
    assert cost.unpriced_reason == UNPRICED_NO_PRICE_ENTRY
    assert cost.total_cost != Decimal(0), "an unknown price must never read as free"
    assert cost.input_cost is None
    assert cost.output_cost is None


def test_a_missing_provider_is_unpriced() -> None:
    cost = compute_cost(table(), provider="nobody", model="fake-1", usage=usage())
    assert cost.priced is False
    assert cost.unpriced_reason == UNPRICED_NO_PRICE_ENTRY


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [(None, 100), (100, None), (None, None)],
)
def test_missing_token_usage_is_unpriced_and_never_zero(
    input_tokens: int | None, output_tokens: int | None
) -> None:
    cost = compute_cost(
        table(),
        provider="fake",
        model="fake-1",
        usage=usage(input_tokens, output_tokens),
    )
    assert cost.priced is False
    assert cost.total_cost is None
    assert cost.unpriced_reason == UNPRICED_NO_TOKEN_USAGE


def test_the_unpriced_reason_is_a_stable_code_not_a_sentence() -> None:
    # The reason is persisted on every case and grouped on to answer "how many
    # cases went unpriced, and why". A message carrying the model name would
    # make every case its own group.
    cost = compute_cost(table(), provider="fake", model="unknown-model", usage=usage())
    assert cost.unpriced_reason == "no_price_entry"
    assert "unknown-model" not in (cost.unpriced_reason or "")


# --- the arithmetic --------------------------------------------------------


def test_cost_is_tokens_times_the_per_million_rate() -> None:
    cost = compute_cost(table(), provider="fake", model="fake-1", usage=usage())

    assert cost.priced is True
    assert cost.input_cost == Decimal("1.50")
    assert cost.output_cost == Decimal("3.00")
    assert cost.total_cost == Decimal("4.50")


def test_the_resolved_unit_prices_travel_on_the_breakdown() -> None:
    cost = compute_cost(table(), provider="fake", model="fake-1", usage=usage())
    assert cost.input_per_mtok == Decimal("1.50")
    assert cost.output_per_mtok == Decimal("6.00")
    assert cost.price_table_id == TABLE_ID
    assert cost.price_table_version == TABLE_VERSION


def test_cached_input_tokens_are_billed_at_their_own_rate() -> None:
    cached = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=400_000,
    )
    cost = compute_cost(table(), provider="fake", model="cached-x", usage=cached)
    # 600k uncached at 10/Mtok plus 400k cached at 1/Mtok.
    assert cost.input_cost == Decimal("6.40")


def test_money_stays_decimal_and_exact() -> None:
    tiny = PriceTable(
        id=TABLE_ID,
        version=TABLE_VERSION,
        models=(
            PriceEntry(
                provider="fake",
                model="fake-1",
                input_per_mtok=Decimal("0.0000012345"),
                output_per_mtok=Decimal(0),
            ),
        ),
        content_hash="sha256:" + "f" * 64,
    )
    cost = compute_cost(tiny, provider="fake", model="fake-1", usage=usage(1_000_000, 0))
    assert isinstance(cost.total_cost, Decimal)
    assert cost.total_cost == Decimal("0.0000012345")


def test_judge_cost_is_carried_separately_and_never_folded_in() -> None:
    cost = compute_cost(
        table(),
        provider="fake",
        model="fake-1",
        usage=usage(),
        judge_cost=Decimal("0.25"),
    )
    assert cost.total_cost == Decimal("4.50"), "judge spend must not inflate the generation cost"
    assert cost.judge_cost == Decimal("0.25")
    assert billable_total(cost) == Decimal("4.75")


def test_an_unpriced_candidate_model_still_reports_a_known_judge_cost() -> None:
    # A judge's spend is known independently of whether the CANDIDATE model has
    # a price entry. Dropping it here would zero out judge cost for exactly the
    # models whose pricing is already unknown.
    cost = compute_cost(
        table(),
        provider="fake",
        model="unknown-model",
        usage=usage(),
        judge_cost=Decimal("0.25"),
    )
    assert cost.priced is False
    assert cost.total_cost is None
    assert cost.judge_cost == Decimal("0.25")


def test_an_unpriced_case_with_no_usage_still_reports_a_known_judge_cost() -> None:
    cost = compute_cost(
        table(),
        provider="fake",
        model="fake-1",
        usage=usage(None, None),
        judge_cost=Decimal("0.25"),
    )
    assert cost.unpriced_reason == UNPRICED_NO_TOKEN_USAGE
    assert cost.judge_cost == Decimal("0.25")


def test_an_unpriced_case_contributes_nothing_to_a_budget() -> None:
    cost = compute_cost(table(), provider="fake", model="unknown", usage=usage())
    assert billable_total(cost) == Decimal(0)
    assert billable_total(None) == Decimal(0)


# --- aggregation -----------------------------------------------------------


def priced_breakdown(total: str) -> CostBreakdown:
    """A priced breakdown for the aggregation tests."""
    return CostBreakdown(
        input_cost=Decimal(total),
        output_cost=Decimal(0),
        total_cost=Decimal(total),
        priced=True,
        price_table_id=TABLE_ID,
        price_table_version=TABLE_VERSION,
    )


def unpriced_breakdown(reason: str = UNPRICED_NO_PRICE_ENTRY) -> CostBreakdown:
    """An unpriced breakdown for the aggregation tests."""
    return CostBreakdown(
        input_cost=None,
        output_cost=None,
        total_cost=None,
        priced=False,
        unpriced_reason=reason,
        price_table_id=TABLE_ID,
        price_table_version=TABLE_VERSION,
    )


def rollup(costs: list[CostBreakdown | None]) -> CostRollup:
    """Aggregate costs with the fixture's table identity."""
    return aggregate_costs(
        costs,
        price_table_id=TABLE_ID,
        price_table_version=TABLE_VERSION,
    )


def test_aggregating_priced_costs_sums_them_and_reports_full_coverage() -> None:
    result = rollup([priced_breakdown("1.25"), priced_breakdown("2.75")])
    assert result.cost.total_cost == Decimal("4.00")
    assert result.cost.priced is True
    assert result.cost.unpriced_reason is None
    assert result.coverage == pytest.approx(1.0)


def test_aggregating_a_mix_reports_a_partial_total_and_says_so() -> None:
    result = rollup([priced_breakdown("1.00"), unpriced_breakdown()])
    assert result.cost.total_cost == Decimal("1.00")
    assert result.cost.priced is False
    assert result.cost.unpriced_reason == UNPRICED_PARTIAL_COVERAGE
    assert result.coverage == pytest.approx(0.5)


def test_aggregating_nothing_priced_is_null_and_keeps_the_dominant_reason() -> None:
    result = rollup(
        [
            unpriced_breakdown(UNPRICED_NO_PRICE_ENTRY),
            unpriced_breakdown(UNPRICED_NO_PRICE_ENTRY),
            unpriced_breakdown(UNPRICED_NO_TOKEN_USAGE),
        ]
    )
    assert result.cost.total_cost is None
    assert result.cost.priced is False
    assert result.cost.unpriced_reason == UNPRICED_NO_PRICE_ENTRY
    assert result.coverage == 0.0


def test_a_case_with_no_cost_object_at_all_counts_as_unpriced() -> None:
    result = rollup([None, None])
    assert result.cost.total_cost is None
    assert result.cost.unpriced_reason == UNPRICED_NO_RESPONSE
    assert result.n_priced == 0


def test_aggregating_no_units_reports_null_coverage_rather_than_zero() -> None:
    result = rollup([])
    assert result.coverage is None, "no units is not the same fact as no coverage"
    assert result.cost.total_cost is None


def test_judge_cost_is_summed_alongside_the_generation_cost() -> None:
    with_judge = CostBreakdown(
        input_cost=Decimal("1.00"),
        output_cost=Decimal(0),
        total_cost=Decimal("1.00"),
        judge_cost=Decimal("0.50"),
        priced=True,
        price_table_id=TABLE_ID,
        price_table_version=TABLE_VERSION,
    )
    result = rollup([with_judge, with_judge])
    assert result.cost.total_cost == Decimal("2.00")
    assert result.cost.judge_cost == Decimal("1.00")


# --- per-unit division -----------------------------------------------------


def test_per_unit_cost_quantizes_to_the_money_scale() -> None:
    value = per_unit_cost(Decimal(1), 3)
    assert value is not None
    assert value == Decimal("0.3333333333")
    assert value.as_tuple().exponent == MONEY_QUANTUM.as_tuple().exponent


def test_per_unit_cost_of_an_unknown_total_is_null() -> None:
    assert per_unit_cost(None, 10) is None


def test_per_unit_cost_over_no_units_is_null_not_zero() -> None:
    assert per_unit_cost(Decimal("1.00"), 0) is None, "a cost per zero cases has no answer"


# --- the shipped table and its loader --------------------------------------


def test_the_shipped_price_table_loads_and_is_content_addressed() -> None:
    shipped = load_price_table(None)
    assert shipped.id == "builtin"
    assert shipped.content_hash.startswith("sha256:")
    assert shipped.models


def test_the_shipped_table_prices_the_fake_provider() -> None:
    entry = load_price_table(DEFAULT_PRICE_TABLE_PATH).lookup("fake", "fake-1")
    assert entry is not None
    assert entry.input_per_mtok == Decimal(0)


def test_prices_written_as_bare_numbers_stay_exact() -> None:
    built = build_price_table(
        {
            "id": "t",
            "version": "1",
            "models": [
                {
                    "provider": "fake",
                    "model": "fake-1",
                    "input_per_mtok": 0.15,
                    "output_per_mtok": "0.60",
                }
            ],
        }
    )
    assert built.models[0].input_per_mtok == Decimal("0.15")


def test_the_content_hash_changes_when_a_price_changes() -> None:
    document = {
        "id": "t",
        "version": "1",
        "models": [
            {"provider": "fake", "model": "m", "input_per_mtok": "1", "output_per_mtok": "2"}
        ],
    }
    first = build_price_table(document)
    changed = {
        **document,
        "models": [{**document["models"][0], "input_per_mtok": "9"}],  # type: ignore[dict-item]
    }
    assert build_price_table(changed).content_hash != first.content_hash


def test_an_unreadable_price_file_raises_a_pricing_error(tmp_path: Path) -> None:
    with pytest.raises(PricingError, match="cannot read"):
        load_price_table(tmp_path / "does-not-exist.yaml")


def test_a_price_file_that_is_not_a_mapping_raises_a_pricing_error(tmp_path: Path) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(PricingError, match="must be a mapping"):
        load_price_table(path)
