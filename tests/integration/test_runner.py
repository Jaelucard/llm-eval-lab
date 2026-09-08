"""The run engine: retries, isolation, cancellation and progress."""

import asyncio
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import pytest
import structlog

from llm_eval_lab.models import (
    CaseResult,
    CaseStatus,
    EvaluationStatus,
    ProgressEvent,
    ProgressEventType,
    ProviderErrorKind,
    RunQuery,
    RunStatus,
    UnitOfWork,
    UnitOfWorkFactory,
)
from llm_eval_lab.services import RunRequest, build_run_service
from llm_eval_lab.settings import Settings

SMOKE_CASES = 6
THIRD_ATTEMPT = 3
HIGH_CONCURRENCY = 32
SECOND_WRITE = 2


def _request(fixtures_dir: Path, **options: Any) -> RunRequest:
    """Build a run request against the six-case smoke fixture."""
    provider_options = {"mode": "expected", **options}
    return RunRequest(
        suite_path=fixtures_dir / "suites" / "smoke.yaml",
        provider="fake",
        model="fake-1",
        provider_options=provider_options,
        concurrency=provider_options.pop("_concurrency", None) or 4,
    )


class Recorder:
    """A progress callback that records every event it is handed."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def __call__(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def of_type(self, event_type: ProgressEventType) -> list[ProgressEvent]:
        return [event for event in self.events if event.type is event_type]


@pytest.mark.integration
async def test_a_clean_run_completes_with_every_case_persisted(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    outcome = await service.execute(_request(fixtures_dir))

    assert outcome.run.status is RunStatus.COMPLETED
    assert len(outcome.results) == SMOKE_CASES
    assert all(result.passed for result in outcome.results)

    view = await service.get_run(outcome.run.id)
    assert view is not None
    assert view.n_completed == SMOKE_CASES
    assert view.pass_rate == 1.0


@pytest.mark.integration
async def test_a_retryable_failure_succeeds_on_the_third_attempt(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    request = _request(fixtures_dir, fail_first_n_attempts=2, fail_kind="server")
    outcome = await service.execute(request)

    assert outcome.run.status is RunStatus.COMPLETED
    for result in outcome.results:
        assert result.status is CaseStatus.OK
        assert result.response is not None
        assert result.response.attempts == THIRD_ATTEMPT
        assert result.attempts == THIRD_ATTEMPT


@pytest.mark.integration
async def test_a_non_retryable_failure_makes_exactly_one_attempt(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    request = _request(fixtures_dir, fail_case_ids=["capital-exact"], fail_kind="authentication")
    outcome = await service.execute(request)

    failed = [item for item in outcome.results if item.case_id == "capital-exact"]
    assert len(failed) == 1
    assert failed[0].status is CaseStatus.ERROR
    assert failed[0].attempts == 1
    assert failed[0].error is not None
    assert failed[0].error.kind is ProviderErrorKind.AUTHENTICATION
    assert failed[0].error.attempts == 1


@pytest.mark.integration
async def test_one_failing_case_does_not_abort_the_run(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    request = _request(fixtures_dir, fail_case_ids=["pi-numeric"], fail_kind="authentication")
    outcome = await service.execute(request)

    assert len(outcome.results) == SMOKE_CASES
    errored = [item for item in outcome.results if item.status is CaseStatus.ERROR]
    assert len(errored) == 1
    assert outcome.run.status is RunStatus.PARTIAL

    view = await service.get_run(outcome.run.id)
    assert view is not None
    assert view.n_completed == SMOKE_CASES
    assert view.n_errors == 1


@pytest.mark.integration
async def test_cancelling_mid_run_keeps_the_completed_cases(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    cancel = asyncio.Event()

    class CancelOnFirstStart:
        """Request cancellation as soon as the first case begins.

        The trigger is CASE_STARTED, not CASE_COMPLETED, and the difference
        matters. The runner emits the terminal event AFTER persisting the case
        and releasing its concurrency slot, so a callback that cancels there is
        racing the next case's own cancellation check. Cancelling at the start of
        the first case is deterministic at concurrency 1: that case is in flight
        and finishes, and every later case sees the flag before it begins.
        """

        def __call__(self, event: ProgressEvent) -> None:
            if event.type is ProgressEventType.CASE_STARTED:
                cancel.set()

    request = RunRequest(
        suite_path=fixtures_dir / "suites" / "smoke.yaml",
        provider="fake",
        model="fake-1",
        provider_options={"mode": "expected"},
        concurrency=1,
    )
    outcome = await service.execute(request, cancel_event=cancel, progress=CancelOnFirstStart())

    assert outcome.run.status is RunStatus.CANCELLED
    assert outcome.run.error == "cancelled"
    assert len(outcome.results) == 1, "the in-flight case finishes; nothing later starts"
    assert outcome.run.totals.n_cancelled == SMOKE_CASES - 1

    view = await service.get_run(outcome.run.id)
    assert view is not None
    assert view.n_completed == len(outcome.results)


@pytest.mark.integration
async def test_progress_events_are_well_formed(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    recorder = Recorder()
    await service.execute(_request(fixtures_dir), progress=recorder)

    assert len(recorder.of_type(ProgressEventType.RUN_STARTED)) == 1
    assert len(recorder.of_type(ProgressEventType.RUN_COMPLETED)) == 1

    terminal = recorder.of_type(ProgressEventType.CASE_COMPLETED) + recorder.of_type(
        ProgressEventType.CASE_FAILED
    )
    assert len(terminal) == SMOKE_CASES
    assert len({event.case_id for event in terminal}) == SMOKE_CASES

    completed = [event.completed for event in recorder.events]
    assert completed == sorted(completed), "completed must never go backwards"
    assert recorder.events[-1].completed == SMOKE_CASES
    assert all(event.total == SMOKE_CASES for event in recorder.events)


@pytest.mark.integration
async def test_a_retried_case_emits_a_populated_retry_event(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)
    recorder = Recorder()
    request = _request(fixtures_dir, fail_first_n_attempts=1, fail_kind="server")
    await service.execute(request, progress=recorder)

    retries = recorder.of_type(ProgressEventType.CASE_RETRY)
    assert len(retries) == SMOKE_CASES
    for event in retries:
        assert event.attempt == 1
        assert event.retry_in_s is not None
        assert event.retry_in_s >= 0.0


@pytest.mark.integration
async def test_the_same_suite_runs_identically_at_any_concurrency(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    service = build_run_service(settings, uow_factory, logger)

    async def outputs(concurrency: int) -> list[tuple[str, str | None, float]]:
        request = RunRequest(
            suite_path=fixtures_dir / "suites" / "smoke.yaml",
            provider="fake",
            model="fake-1",
            provider_options={"mode": "derived", "seed": 11},
            concurrency=concurrency,
        )
        outcome = await service.execute(request)
        return sorted(
            (
                item.case_id,
                None if item.response is None else item.response.output_text,
                0.0 if item.response is None else item.response.latency_ms,
            )
            for item in outcome.results
        )

    assert await outputs(1) == await outputs(HIGH_CONCURRENCY)


@pytest.mark.integration
async def test_a_storage_failure_aborts_the_run_and_marks_it_failed(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """StorageError is deliberately not isolated, and must surface as itself.

    A database that cannot be written to is a broken run, not a bad case. The
    caller therefore gets a single `StorageError`, not the `TaskGroup`'s
    `ExceptionGroup` - the CLI's exception guard is bound to this project's
    hierarchy and cannot be expected to unwrap a group - and the run row records
    FAILED rather than being abandoned in RUNNING forever.
    """
    from llm_eval_lab.models import StorageError  # noqa: PLC0415 - reads next to its use
    from llm_eval_lab.storage.repositories import (  # noqa: PLC0415 - reads next to its use
        CaseResultRepository,
    )

    original = CaseResultRepository.add
    calls = 0

    async def _fail_on_the_second_write(self: CaseResultRepository, result: CaseResult) -> None:
        nonlocal calls
        calls += 1
        if calls >= SECOND_WRITE:
            msg = "disk on fire"
            raise StorageError(msg)
        await original(self, result)

    monkeypatch.setattr(CaseResultRepository, "add", _fail_on_the_second_write)
    service = build_run_service(settings, uow_factory, logger)

    request = RunRequest(
        suite_path=fixtures_dir / "suites" / "smoke.yaml",
        provider="fake",
        model="fake-1",
        provider_options={"mode": "expected"},
        concurrency=1,
    )
    with pytest.raises(StorageError, match="disk on fire"):
        await service.execute(request)

    monkeypatch.undo()
    page = await service.list_runs(RunQuery(limit=5))
    assert page.items[0].status is RunStatus.FAILED

    view = await service.get_run(page.items[0].id)
    assert view is not None
    assert view.run.error is not None
    assert "disk on fire" in view.run.error


@pytest.mark.integration
async def test_an_errored_evaluation_makes_its_case_errored(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    """An evaluator that could not conclude is a run failure, not a silent pass.

    Before this, a suite whose every evaluator timed out reported `n_errors: 0`,
    finished COMPLETED and exited 0, so a CI gate keyed on the exit code passed a
    benchmark in which nothing had been scored.
    """
    service = build_run_service(settings, uow_factory, logger)
    request = RunRequest(
        suite_path=fixtures_dir / "suites" / "regex_timeout.yaml",
        provider="fake",
        model="fake-1",
        provider_options={"mode": "expected"},
        concurrency=2,
    )
    outcome = await service.execute(request)

    by_id = {item.case_id: item for item in outcome.results}
    assert by_id["catastrophic"].status is CaseStatus.ERROR
    assert by_id["well-behaved"].status is CaseStatus.OK

    timed_out = by_id["catastrophic"].evaluations[0]
    assert timed_out.status is EvaluationStatus.ERROR
    assert timed_out.metadata["error_code"] == "regex_timeout"

    assert outcome.run.totals.n_errors == 1

    # The run FINISHES rather than aborting, and its status says so: PARTIAL,
    # which maps to exit code 5 (run incomplete). COMPLETED with exit 0 would
    # leave the CI hole this change exists to close, and would make
    # `RunStatus.PARTIAL` unreachable for the one situation it names.
    assert outcome.run.status is RunStatus.PARTIAL

    # The pass rate is computed over the cases that were actually scorable.
    view = await service.get_run(outcome.run.id)
    assert view is not None
    assert view.n_errors == 1
    assert view.n_pass_denominator == 1
    assert view.pass_rate == 1.0


@pytest.mark.integration
async def test_finalization_writes_status_and_metrics_in_one_unit_of_work(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    """The terminal status write and the metrics rollup write share one transaction.

    Before this, the finalization tail opened three separate units of work:
    one to write the terminal status, one to reload the run, and a third (in
    a since-removed `_materialize_metrics`) to compute and write the rollup. A
    reader could land in the gap between the first and the third and see a run
    reporting COMPLETED with no rollup behind it yet - exactly what
    `X-Metrics-Materialized: false` on an already-finished run would mean.
    Counting how many units of work the tail opens is a direct check that the
    gap is closed, rather than a timing-dependent race test that could pass by
    luck under asyncio's cooperative scheduling either way.
    """
    opened = 0

    def counting_factory() -> AbstractAsyncContextManager[UnitOfWork]:
        nonlocal opened
        opened += 1
        return uow_factory()

    service = build_run_service(settings, counting_factory, logger)
    before = opened
    outcome = await service.execute(_request(fixtures_dir))
    after = opened

    assert outcome.run.status is RunStatus.COMPLETED
    # One unit of work to create the PENDING run row, one per persisted case,
    # one to mark the run RUNNING, and exactly ONE for the whole finalization
    # tail - not the three the old code opened for it.
    assert after - before == SMOKE_CASES + 3
