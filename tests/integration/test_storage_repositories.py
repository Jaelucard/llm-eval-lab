"""Repositories: domain models in, equal domain models out, and no float drift."""

import asyncio
import uuid
from datetime import timedelta
from decimal import Decimal

import pytest

from llm_eval_lab.models import (
    CaseQuery,
    CaseResult,
    CaseStatus,
    CostBreakdown,
    EvaluationRecord,
    EvaluationResult,
    EvaluationStatus,
    FinishReason,
    ModelResponse,
    ProviderConfig,
    Run,
    RunConfig,
    RunQuery,
    RunStatus,
    RunTotals,
    TokenUsage,
    UnitOfWorkFactory,
)
from llm_eval_lab.utils.time import utc_now

EXACT_DECIMAL = Decimal("0.0000012345")
"""A cost small enough that a float column would silently change it."""


def _config(**overrides: object) -> RunConfig:
    payload: dict[str, object] = {
        "suite_name": "storage",
        "suite_version": "1",
        "suite_hash": "sha256:suite",
        "suite_source": "tests/fixtures/suites/minimal.yaml",
        "case_selection": {"n_selected": 1},
        "provider": ProviderConfig(provider="fake", model="fake-1"),
        "params": {"max_output_tokens": 64},
        "evaluators": [{"type": "exact_match"}],
        "price_table_id": "test-prices",
        "price_table_version": "1",
        "price_table_hash": "sha256:prices",
        "library_version": "0.1.0",
        "python_version": "3.13.0",
        **overrides,
    }
    return RunConfig.model_validate(payload)


def _run(**overrides: object) -> Run:
    now = utc_now()
    payload: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "label": "a labelled run",
        "status": RunStatus.PENDING,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "config": _config(),
        "totals": RunTotals(n_cases=1),
        "updated_at": now,
        "tags": ("nightly",),
        **overrides,
    }
    return Run.model_validate(payload)


def _response() -> ModelResponse:
    now = utc_now()
    return ModelResponse(
        output_text="Paris",
        provider="fake",
        model="fake-1",
        requested_model="fake-1",
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(input_tokens=11, output_tokens=3),
        latency_ms=12.5,
        total_latency_ms=12.5,
        started_at=now,
        completed_at=now,
        attempts=2,
        provider_request_id="fake-abc",
        raw={"simulated": True, "mode": "expected"},
    )


def _evaluation() -> EvaluationResult:
    return EvaluationResult(
        evaluator_id="exact_match",
        evaluator_type="exact_match",
        status=EvaluationStatus.PASSED,
        passed=True,
        score=1.0,
        metadata={"case_sensitive": True},
        duration_ms=0.5,
    )


def _cost(total: Decimal) -> CostBreakdown:
    return CostBreakdown(
        input_cost=total,
        output_cost=Decimal(0),
        total_cost=total,
        priced=True,
        price_table_id="test-prices",
        price_table_version="1",
        input_per_mtok=Decimal("1.5"),
        output_per_mtok=Decimal("2.5"),
    )


def _case_result(run_id: str, response: ModelResponse, cost: CostBreakdown) -> CaseResult:
    now = utc_now()
    return CaseResult.model_validate(
        {
            "run_id": run_id,
            "case_id": "c1",
            "case_hash": "sha256:case",
            "status": CaseStatus.OK,
            "response": response,
            "evaluations": (_evaluation(),),
            "passed": True,
            "score": 1.0,
            "cost": cost,
            "attempts": 2,
            "started_at": now,
            "completed_at": now,
            "category": "geography",
            "tags": ("smoke",),
        }
    )


async def _write_full_run(
    uow_factory: UnitOfWorkFactory,
    *,
    cost_total: Decimal = EXACT_DECIMAL,
) -> tuple[Run, CaseResult]:
    """Persist a run, its response, its case result and its evaluation."""
    run = _run()
    response = _response()
    result = _case_result(run.id, response, _cost(cost_total))
    async with uow_factory() as uow:
        await uow.runs.create(run)
        await uow.responses.add(run.id, result.case_id, response)
        await uow.cases.add(result)
        await uow.evaluations.add_many(
            [EvaluationRecord(run_id=run.id, case_id=result.case_id, result=_evaluation())]
        )
        await uow.commit()
    return run, result


@pytest.mark.integration
async def test_run_round_trips_as_an_equal_domain_model(
    uow_factory: UnitOfWorkFactory,
) -> None:
    run, _ = await _write_full_run(uow_factory)
    async with uow_factory() as uow:
        stored = await uow.runs.get(run.id)
    assert stored == run


@pytest.mark.integration
async def test_case_result_round_trips_with_its_response_and_evaluations(
    uow_factory: UnitOfWorkFactory,
) -> None:
    run, written = await _write_full_run(uow_factory)
    async with uow_factory() as uow:
        stored = await uow.cases.get(run.id, "c1")

    assert stored is not None
    assert stored.response == written.response
    assert stored.evaluations == written.evaluations
    assert stored == written


@pytest.mark.integration
async def test_no_orm_instance_escapes_a_repository(uow_factory: UnitOfWorkFactory) -> None:
    run, _ = await _write_full_run(uow_factory)
    async with uow_factory() as uow:
        stored_run = await uow.runs.get(run.id)
        page = await uow.cases.list_for_run(run.id, CaseQuery())
        response = await uow.responses.get(run.id, "c1")

    for value in (stored_run, response, *page.items):
        assert type(value).__module__.startswith("llm_eval_lab.models"), (
            f"{type(value)!r} crossed the repository boundary"
        )


@pytest.mark.integration
async def test_decimal_survives_a_write_and_read_cycle_exactly(
    uow_factory: UnitOfWorkFactory,
) -> None:
    run, _ = await _write_full_run(uow_factory, cost_total=EXACT_DECIMAL)
    async with uow_factory() as uow:
        stored = await uow.cases.get(run.id, "c1")

    assert stored is not None
    assert stored.cost is not None
    assert stored.cost.total_cost == EXACT_DECIMAL
    assert str(stored.cost.total_cost) == str(EXACT_DECIMAL)


@pytest.mark.integration
async def test_timestamps_come_back_timezone_aware(uow_factory: UnitOfWorkFactory) -> None:
    run, _ = await _write_full_run(uow_factory)
    async with uow_factory() as uow:
        stored = await uow.runs.get(run.id)
    assert stored is not None
    assert stored.created_at.tzinfo is not None
    assert stored.updated_at.tzinfo is not None


@pytest.mark.integration
async def test_completed_case_ids_supports_resume(uow_factory: UnitOfWorkFactory) -> None:
    run, _ = await _write_full_run(uow_factory)
    async with uow_factory() as uow:
        assert await uow.cases.completed_case_ids(run.id) == frozenset({"c1"})


@pytest.mark.integration
async def test_iter_for_run_streams_every_case(uow_factory: UnitOfWorkFactory) -> None:
    run, _ = await _write_full_run(uow_factory)
    async with uow_factory() as uow:
        streamed = [item async for item in uow.cases.iter_for_run(run.id)]
    assert [item.case_id for item in streamed] == ["c1"]


@pytest.mark.integration
async def test_mark_stale_running_flips_only_the_older_run(
    uow_factory: UnitOfWorkFactory,
) -> None:
    now = utc_now()
    stale = _run(status=RunStatus.RUNNING, updated_at=now - timedelta(hours=2))
    fresh = _run(status=RunStatus.RUNNING, updated_at=now)
    async with uow_factory() as uow:
        await uow.runs.create(stale)
        await uow.runs.create(fresh)
        await uow.commit()

    async with uow_factory() as uow:
        changed = await uow.runs.mark_stale_running_as_interrupted(before=now - timedelta(hours=1))
        await uow.commit()

    async with uow_factory() as uow:
        stale_after = await uow.runs.get(stale.id)
        fresh_after = await uow.runs.get(fresh.id)

    assert changed == 1
    assert stale_after is not None
    assert fresh_after is not None
    assert stale_after.status is RunStatus.INTERRUPTED
    assert fresh_after.status is RunStatus.RUNNING


@pytest.mark.integration
async def test_snapshot_upsert_is_idempotent(uow_factory: UnitOfWorkFactory) -> None:
    from llm_eval_lab.models import BenchmarkSnapshot  # noqa: PLC0415 - reads next to its use

    snapshot = BenchmarkSnapshot(
        suite_hash="sha256:snapshot",
        name="minimal",
        version="1",
        n_cases=1,
        created_at=utc_now(),
        source_path=None,
        body={"suite": {"name": "minimal"}},
    )
    async with uow_factory() as uow:
        await uow.benchmarks.upsert_snapshot(snapshot)
        await uow.benchmarks.upsert_snapshot(snapshot)
        await uow.commit()

    async with uow_factory() as uow:
        stored = await uow.benchmarks.get_snapshot("sha256:snapshot")
        listed = await uow.benchmarks.list_snapshots()

    assert stored == snapshot
    assert len(listed) == 1


@pytest.mark.integration
async def test_snapshot_upsert_is_race_safe_across_concurrent_units_of_work(
    uow_factory: UnitOfWorkFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two SEPARATE sessions upserting the same digest at once must not fail.

    `test_snapshot_upsert_is_idempotent` above proves the same-session case;
    this is the one that actually flaked (`test_api_runs.py
    ::test_concurrent_launches_never_exceed_the_process_cap`, several
    concurrent launches of the identical suite, each opening its own unit of
    work): both can see `existing is None` before either commits, and the
    loser used to fail its insert's `UNIQUE` constraint on `suite_hash`,
    surfacing as an API request that failed outright over a snapshot that a
    same-moment recheck would have found already stored.

    Reproducing this from real concurrency alone is a coin flip - the actual
    flake needed a full HTTP round trip's worth of intervening `await`s to
    open a wide enough window, and a handful of directly-launched tasks
    mostly just queue for a connection and never truly overlap. So the
    check-then-act window is widened deliberately here, by making every
    session's own `get` pause briefly after returning: this exercises the
    REAL `upsert_snapshot` code path with the exact interleaving the bug
    needs, deterministically, rather than hoping the scheduler cooperates.
    The session's class is found at runtime (`type(uow.session)`) rather
    than imported, since only `storage/` may import SQLAlchemy directly.
    """
    from llm_eval_lab.models import BenchmarkSnapshot  # noqa: PLC0415 - reads next to its use

    async with uow_factory() as discover:
        # `.session` is not on the `UnitOfWork` Protocol - only the concrete
        # implementation exposes it, "for migrations and tests only" (its own
        # docstring). Exactly this use.
        session_cls = type(discover.session)  # type: ignore[attr-defined]
    original_get = session_cls.get

    async def _get_then_yield_the_race_window(
        self: object, *args: object, **kwargs: object
    ) -> object:
        result = await original_get(self, *args, **kwargs)
        await asyncio.sleep(0.05)
        return result

    monkeypatch.setattr(session_cls, "get", _get_then_yield_the_race_window)

    snapshot = BenchmarkSnapshot(
        suite_hash="sha256:concurrent-snapshot",
        name="minimal",
        version="1",
        n_cases=1,
        created_at=utc_now(),
        source_path=None,
        body={"suite": {"name": "minimal"}},
    )

    async def _upsert_in_its_own_unit_of_work() -> None:
        async with uow_factory() as uow:
            await uow.benchmarks.upsert_snapshot(snapshot)
            await uow.commit()

    results = await asyncio.gather(
        *(_upsert_in_its_own_unit_of_work() for _ in range(6)),
        return_exceptions=True,
    )
    errors = [item for item in results if isinstance(item, BaseException)]
    assert not errors, f"a concurrent upsert of the same digest raised: {errors}"

    async with uow_factory() as uow:
        stored = await uow.benchmarks.get_snapshot("sha256:concurrent-snapshot")
        listed = await uow.benchmarks.list_snapshots()

    assert stored == snapshot
    assert sum(1 for item in listed if item.suite_hash == snapshot.suite_hash) == 1


@pytest.mark.integration
async def test_a_non_duplicate_integrity_failure_during_upsert_is_not_swallowed(
    uow_factory: UnitOfWorkFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`upsert_snapshot`'s `IntegrityError` catch must be narrow by CONSTRUCTION,
    not merely because `suite_hash` is the only constraint this table has today.

    It re-checks by `suite_hash` after catching: present means the benign race
    (someone else's insert won), absent means the failure was about something
    else entirely - a future FK or CHECK constraint, say - and must surface
    rather than vanish behind a snapshot that was never actually persisted.

    Forced here by making the insert's flush raise a genuine `IntegrityError`
    while the digest was never written by anyone. The error is a REAL one,
    captured from an actual duplicate-key failure rather than constructed or
    imported, since only `storage/` may import SQLAlchemy directly and this
    project has no reason to know that exception's constructor shape.
    """
    from llm_eval_lab.models import BenchmarkSnapshot, StorageError  # noqa: PLC0415
    from llm_eval_lab.storage import mappers  # noqa: PLC0415 - reads next to its use

    already_stored = BenchmarkSnapshot(
        suite_hash="sha256:integrity-error-probe",
        name="minimal",
        version="1",
        n_cases=1,
        created_at=utc_now(),
        source_path=None,
        body={"suite": {"name": "minimal"}},
    )
    async with uow_factory() as uow:
        await uow.benchmarks.upsert_snapshot(already_stored)
        await uow.commit()

    # Trigger one genuine duplicate-key failure, bypassing `upsert_snapshot`'s
    # own guard, purely to capture a real `IntegrityError` instance to reuse.
    real_integrity_error: BaseException | None = None
    async with uow_factory() as uow:
        uow.session.add(mappers.snapshot_to_row(already_stored))  # type: ignore[attr-defined]
        try:
            await uow.session.flush()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - test-only: capturing the real error
            real_integrity_error = exc
    assert real_integrity_error is not None, "expected the duplicate insert to raise"

    not_a_duplicate = BenchmarkSnapshot(
        suite_hash="sha256:not-actually-a-duplicate",
        name="minimal",
        version="1",
        n_cases=1,
        created_at=utc_now(),
        source_path=None,
        body={"suite": {"name": "minimal"}},
    )
    async with uow_factory() as uow:
        session_cls = type(uow.session)  # type: ignore[attr-defined]

        async def _flush_raises_a_non_duplicate_integrity_error(_self: object) -> None:
            raise real_integrity_error

        monkeypatch.setattr(session_cls, "flush", _flush_raises_a_non_duplicate_integrity_error)

        with pytest.raises(StorageError):
            await uow.benchmarks.upsert_snapshot(not_a_duplicate)


@pytest.mark.integration
async def test_run_listing_filters_and_derives_a_pass_rate(
    uow_factory: UnitOfWorkFactory,
) -> None:
    run, _ = await _write_full_run(uow_factory)
    async with uow_factory() as uow:
        await uow.runs.update_status(
            run.id,
            RunStatus.COMPLETED,
            updated_at=utc_now(),
            completed_at=utc_now(),
            totals=RunTotals(n_cases=1, n_completed=1, n_passed=1),
        )
        await uow.commit()

    async with uow_factory() as uow:
        page = await uow.runs.list(RunQuery(provider="fake"))
        empty = await uow.runs.list(RunQuery(provider="nobody"))

    assert page.total == 1
    assert page.items[0].id == run.id
    assert page.items[0].pass_rate == 1.0
    assert empty.total == 0


@pytest.mark.integration
async def test_listing_and_show_agree_on_a_score_only_run(
    uow_factory: UnitOfWorkFactory,
) -> None:
    """The two pass-rate surfaces must not disagree, and score-only is where they did.

    The listing used to divide by ``n_completed - n_errors``, counting a case
    with a score but no boolean verdict in the denominator, while ``show``
    excluded it. For a judge-only suite the same run reported two different pass
    rates depending on which command you asked. Both now call
    ``llm_eval_lab.scoring.pass_rate``.
    """
    from llm_eval_lab.scoring import pass_rate_of_results  # noqa: PLC0415 - reads next to its use

    run = _run()
    response = _response()
    scored = _case_result(run.id, response, _cost(EXACT_DECIMAL))
    # One ordinary passing case, one score-only case: a score, but no verdict.
    score_only = scored.model_copy(update={"case_id": "c2", "passed": None, "score": 0.5})

    async with uow_factory() as uow:
        await uow.runs.create(run)
        await uow.responses.add(run.id, "c1", response)
        await uow.cases.add_many([scored, score_only])
        await uow.runs.update_status(
            run.id,
            RunStatus.COMPLETED,
            updated_at=utc_now(),
            completed_at=utc_now(),
            totals=RunTotals(n_cases=2, n_completed=2, n_passed=1),
        )
        await uow.commit()

    async with uow_factory() as uow:
        page = await uow.runs.list(RunQuery())
        stored = await uow.cases.list_for_run(run.id, CaseQuery())

    from_show = pass_rate_of_results(list(stored.items))
    assert page.items[0].pass_rate == from_show.rate
    assert from_show.n_denominator == 1, "the score-only case is not in the denominator"
    assert page.items[0].pass_rate == 1.0


@pytest.mark.integration
async def test_a_run_with_no_scorable_case_reports_no_pass_rate(
    uow_factory: UnitOfWorkFactory,
) -> None:
    """`None`, never `0.0`: 'nothing was eligible' is not 'everything failed'."""
    run = _run()
    response = _response()
    base = _case_result(run.id, response, _cost(EXACT_DECIMAL))
    only_scored = base.model_copy(update={"passed": None, "score": 0.5})

    async with uow_factory() as uow:
        await uow.runs.create(run)
        await uow.cases.add(only_scored)
        await uow.runs.update_status(
            run.id,
            RunStatus.COMPLETED,
            updated_at=utc_now(),
            totals=RunTotals(n_cases=1, n_completed=1),
        )
        await uow.commit()

    async with uow_factory() as uow:
        page = await uow.runs.list(RunQuery())

    assert page.items[0].pass_rate is None


@pytest.mark.integration
async def test_judge_cost_is_never_folded_into_total_cost_on_either_surface(
    uow_factory: UnitOfWorkFactory,
) -> None:
    """`total_cost` means CANDIDATE-MODEL cost, on the listing and in the rollup.

    The two surfaces used to disagree: the listing folded `judge_cost` in and the
    rollup did not, so one run published two different numbers under one name.
    Nothing populates `judge_cost` in v1, so the case results are written here
    directly to exercise the branch before the judge evaluator lands.
    """
    from llm_eval_lab.reporting.aggregate import metrics_for_run  # noqa: PLC0415

    candidate = Decimal("0.002")
    judge = Decimal("0.010")
    run = _run(status=RunStatus.COMPLETED, totals=RunTotals(n_cases=2, n_completed=2, n_passed=2))
    results = [
        CaseResult(
            run_id=run.id,
            case_id=f"case-{index}",
            case_hash="sha256:case",
            status=CaseStatus.OK,
            response=_response(),
            evaluations=(),
            passed=True,
            score=1.0,
            cost=CostBreakdown(
                input_cost=candidate,
                output_cost=Decimal(0),
                total_cost=candidate,
                judge_cost=judge,
                priced=True,
                price_table_id="test-prices",
                price_table_version="1",
            ),
            started_at=run.created_at,
            completed_at=run.created_at,
        )
        for index in range(2)
    ]

    async with uow_factory() as uow:
        await uow.runs.create(run)
        for result in results:
            await uow.responses.add(run.id, result.case_id, _response())
        await uow.cases.add_many(results)
        await uow.metrics.put(metrics_for_run(run, results))
        await uow.commit()

    async with uow_factory() as uow:
        rollup = await uow.metrics.get(run.id)
        page = await uow.runs.list(RunQuery(limit=10))

    assert rollup is not None
    summary = next(item for item in page.items if item.id == run.id)

    expected_candidate = candidate * 2
    assert rollup.cost.total_cost == expected_candidate
    assert summary.total_cost == expected_candidate, "the listing publishes the same quantity"
    assert rollup.cost.judge_cost == judge * 2, "judge spend is reported separately"

    # Within one rollup the figures reconcile, which they did not while
    # `cost_per_case` divided a judge-inclusive total by the case count.
    assert rollup.cost_per_case is not None
    assert rollup.cost_per_case * 2 == expected_candidate
    assert rollup.cost_per_successful_evaluation == rollup.cost_per_case
