"""`--max-cost`: a RUNNING budget, enforced only against spend the system knows."""

from decimal import Decimal
from pathlib import Path

import pytest
import structlog

from llm_eval_lab.models import RunStatus, UnitOfWorkFactory
from llm_eval_lab.runner.engine import BUDGET_EXCEEDED
from llm_eval_lab.services import RunRequest, build_run_service
from llm_eval_lab.settings import Settings

SMOKE_CASES = 6
BUDGET_MODEL = "fake-budget"
"""Priced at 1.00 USD per token by the test price table, so the arithmetic is exact."""


def _request(
    fixtures_dir: Path,
    *,
    model: str = BUDGET_MODEL,
    max_cost: Decimal | None = None,
) -> RunRequest:
    return RunRequest(
        suite_path=fixtures_dir / "suites" / "smoke.yaml",
        provider="fake",
        model=model,
        provider_options={"mode": "expected"},
        concurrency=1,
        max_cost=max_cost,
        price_table_path=fixtures_dir / "pricing" / "test_prices.yaml",
    )


@pytest.mark.integration
async def test_a_budget_below_the_projected_total_cancels_the_run(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)

    # Learn what the suite actually costs rather than hard-coding a figure that
    # would silently stop testing anything if the fixture prompts ever changed.
    baseline = await service.execute(_request(fixtures_dir))
    assert baseline.run.status is RunStatus.COMPLETED
    assert baseline.budget_spent > 0
    assert baseline.unpriced_cases == 0

    cap = baseline.budget_spent / 3
    capped = await service.execute(_request(fixtures_dir, max_cost=cap))

    assert capped.run.status is RunStatus.CANCELLED
    assert capped.run.error == BUDGET_EXCEEDED
    assert len(capped.results) >= 1
    assert len(capped.results) < SMOKE_CASES

    view = await service.get_run(capped.run.id)
    assert view is not None
    assert view.n_completed == len(capped.results)


@pytest.mark.integration
async def test_a_budget_above_the_projected_total_completes_normally(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    baseline = await service.execute(_request(fixtures_dir))
    generous = await service.execute(_request(fixtures_dir, max_cost=baseline.budget_spent * 2))

    assert generous.run.status is RunStatus.COMPLETED
    assert generous.run.error is None
    assert len(generous.results) == SMOKE_CASES


@pytest.mark.integration
async def test_unpriced_cases_contribute_nothing_and_are_counted(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    # A model the test price table does not list at all.
    outcome = await service.execute(
        _request(fixtures_dir, model="unpriced-model", max_cost=Decimal("0.01"))
    )

    assert outcome.run.status is RunStatus.COMPLETED, (
        "a budget must never be enforced against spend the system does not know"
    )
    assert outcome.budget_spent == Decimal(0)
    assert outcome.unpriced_cases == SMOKE_CASES


@pytest.mark.integration
async def test_the_cap_is_recorded_in_the_persisted_configuration(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    cap = Decimal("12.5")
    outcome = await service.execute(_request(fixtures_dir, max_cost=cap))

    view = await service.get_run(outcome.run.id)
    assert view is not None
    assert view.run.config.max_cost == cap
