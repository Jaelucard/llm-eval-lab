"""A judge-backed suite executed by the real runner, against the fake provider.

The unit tests drive the judge evaluator directly with a provider double. This
one goes through the whole path a user's `llm-eval run` takes - loader,
resolver, runner, evaluator, storage - and asserts that the judge provenance
SURVIVES persistence. Everything the judge recorded is worthless if the raw
outputs and the dispersion are dropped between the evaluator and the database.
"""

from pathlib import Path

import pytest
import structlog

from llm_eval_lab.models import (
    CaseQuery,
    CaseResult,
    EvaluationStatus,
    UnitOfWorkFactory,
)
from llm_eval_lab.services import RunRequest, build_run_service
from llm_eval_lab.settings import Settings

pytestmark = pytest.mark.integration

EXPECTED_RAW_SCORE = 4.4
EXPECTED_CASES = 2

JUDGE_PRICE_TABLE = """\
id: judge-pricing-test
version: "1"
currency: USD
models:
  - provider: fake
    model: fake-1
    input_per_mtok: "0"
    output_per_mtok: "0"
  - provider: fake
    model: fake-judge-1
    input_per_mtok: "10"
    output_per_mtok: "20"
"""
"""A price table that prices the judge model but not the candidate model.

The candidate is priced at zero and the judge is priced well above zero, so a
non-zero `judge_cost` alongside an unchanged (zero) `total_cost` can only come
from the judge usage having been priced separately - not folded into candidate
spend.
"""


async def _run_judge_suite(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    fixtures_dir: Path,
    logger: structlog.BoundLogger,
) -> list[CaseResult]:
    """Execute the judge fixture suite end to end and return its case results."""
    service = build_run_service(settings, uow_factory, logger)
    outcome = await service.execute(
        RunRequest(
            suite_path=fixtures_dir / "suites" / "judge_rubric.yaml",
            provider="fake",
            model="fake-1",
        )
    )
    async with uow_factory() as uow:
        page = await uow.cases.list_for_run(outcome.run.id, CaseQuery())
    return sorted(page.items, key=lambda item: item.case_id)


async def test_a_judge_backed_suite_runs_end_to_end_with_no_network(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    fixtures_dir: Path,
    logger: structlog.BoundLogger,
) -> None:
    results = await _run_judge_suite(settings, uow_factory, fixtures_dir, logger)
    assert len(results) == EXPECTED_CASES
    assert [result.case_id for result in results] == ["capital-gated", "capital-graded"]
    for result in results:
        assert result.evaluations[0].status is EvaluationStatus.PASSED


async def test_the_persisted_case_result_carries_full_judge_provenance(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    fixtures_dir: Path,
    logger: structlog.BoundLogger,
) -> None:
    results = await _run_judge_suite(settings, uow_factory, fixtures_dir, logger)
    provenance = results[0].evaluations[0].judge
    assert provenance is not None

    assert provenance.judge_models == ("fake:fake-judge-1",)
    assert len(provenance.raw_outputs) == 1
    assert "overall_score" in provenance.raw_outputs[0]
    assert provenance.dispersion == pytest.approx(0.0)
    assert provenance.agreement == pytest.approx(1.0)
    assert len(provenance.rendered_prompts) == 1
    assert len(provenance.verdicts) == 1
    assert provenance.template_id == "single_answer_grading_v2"
    assert provenance.rubric_hash.startswith("sha256:")
    assert provenance.parse_failures == 0
    assert provenance.self_preference_risk is False
    assert provenance.high_judge_variance is False


async def test_the_score_only_case_is_excluded_from_the_pass_rate(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    fixtures_dir: Path,
    logger: structlog.BoundLogger,
) -> None:
    """One case gates on a threshold, one only scores; only the first is counted."""
    results = await _run_judge_suite(settings, uow_factory, fixtures_dir, logger)
    by_id = {result.case_id: result for result in results}

    gated = by_id["capital-gated"].evaluations[0]
    assert gated.passed is True
    assert gated.raw_score == pytest.approx(EXPECTED_RAW_SCORE)

    graded = by_id["capital-graded"].evaluations[0]
    assert graded.status is EvaluationStatus.PASSED
    assert graded.passed is None
    assert graded.score is not None


async def test_judge_tokens_stay_out_of_the_candidate_cost(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    fixtures_dir: Path,
    logger: structlog.BoundLogger,
) -> None:
    """`total_cost` is candidate spend; judge usage is recorded on its own."""
    results = await _run_judge_suite(settings, uow_factory, fixtures_dir, logger)
    result = results[0]
    provenance = result.evaluations[0].judge
    assert provenance is not None
    assert result.response is not None

    judge_input = provenance.usage.input_tokens
    candidate_input = result.response.usage.input_tokens
    assert judge_input is not None
    assert candidate_input is not None
    # The judge prompt is far larger than the case prompt, so folding one into
    # the other would be obvious here rather than a rounding difference.
    assert judge_input > candidate_input


async def test_judge_usage_is_priced_separately_and_never_folded_into_total_cost(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    fixtures_dir: Path,
    logger: structlog.BoundLogger,
    tmp_path: Path,
) -> None:
    """`CostBreakdown.judge_cost` is populated from the judge's own usage and price.

    The candidate model is priced at zero and the judge model is not, so a
    positive `judge_cost` next to a zero, unchanged `total_cost` can only come
    from the runner pricing the judge's `JudgeProvenance.usage` on its own,
    against the judge's own provider and model - never adding it to the
    candidate response's usage.
    """
    prices = tmp_path / "prices.yaml"
    prices.write_text(JUDGE_PRICE_TABLE, encoding="utf-8")
    service = build_run_service(settings, uow_factory, logger)
    outcome = await service.execute(
        RunRequest(
            suite_path=fixtures_dir / "suites" / "judge_rubric.yaml",
            provider="fake",
            model="fake-1",
            price_table_path=prices,
        )
    )
    async with uow_factory() as uow:
        page = await uow.cases.list_for_run(outcome.run.id, CaseQuery())
    results = sorted(page.items, key=lambda item: item.case_id)

    for result in results:
        assert result.cost is not None
        assert result.cost.total_cost == 0, "the candidate model is priced at zero"
        assert result.cost.judge_cost is not None
        assert result.cost.judge_cost > 0, "the judge model is not, so its cost must show up"


async def test_a_self_judging_run_surfaces_the_run_level_self_preference_warning(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    fixtures_dir: Path,
    logger: structlog.BoundLogger,
) -> None:
    """When the candidate model equals the configured judge model, the RUN must say so.

    `JudgeProvenance.self_preference_risk` is already set correctly per evaluation
    (see `test_the_persisted_case_result_carries_full_judge_provenance`). This
    proves the advisory also reaches `Run.warnings`, both on the value returned by
    `execute()` and on the row as re-read from storage - the surface the API and
    dashboard actually read from.
    """
    service = build_run_service(settings, uow_factory, logger)
    outcome = await service.execute(
        RunRequest(
            suite_path=fixtures_dir / "suites" / "judge_rubric.yaml",
            provider="fake",
            # Matches the suite's own judge provider (`provider: fake, model:
            # fake-judge-1`), which is exactly what trips self-preference risk.
            model="fake-judge-1",
        )
    )
    assert "self_preference_risk" in outcome.run.warnings

    async with uow_factory() as uow:
        stored = await uow.runs.get(outcome.run.id)
    assert stored is not None
    assert "self_preference_risk" in stored.warnings


async def test_a_non_self_judging_run_reports_no_warnings(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    fixtures_dir: Path,
    logger: structlog.BoundLogger,
) -> None:
    """The negative control: distinct candidate/judge models raise no advisory."""
    results = await _run_judge_suite(settings, uow_factory, fixtures_dir, logger)
    assert results  # sanity: the suite actually ran
    service = build_run_service(settings, uow_factory, logger)
    outcome = await service.execute(
        RunRequest(
            suite_path=fixtures_dir / "suites" / "judge_rubric.yaml",
            provider="fake",
            model="fake-1",
        )
    )
    assert outcome.run.warnings == ()
