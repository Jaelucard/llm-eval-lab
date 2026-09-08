"""Repository implementations. The only code in the project that writes SQL.

Each repository implements the matching Protocol in
``llm_eval_lab.models.protocols`` and returns frozen domain models, never ORM
instances, so nothing above this layer can hold a session-bound object or
lazy-load across a transaction boundary.

**One response, one write.** A :class:`~llm_eval_lab.models.ModelResponse` is
written exactly once, by :class:`ResponseRepository`, into the
``model_responses`` row keyed by ``(run_id, case_id)``.
:class:`CaseResultRepository` deliberately does NOT write it, and hydrates
``CaseResult.response`` from that row on read. Writing it in both places would
give the same fact two homes and, eventually, two different values.

Every ``SQLAlchemyError`` is translated into
:class:`~llm_eval_lab.models.StorageError`. That is a narrowing, not a broad
catch: callers above this layer bound their handling to the project's own
exception hierarchy, and the runner deliberately does not isolate a
``StorageError`` - a database that cannot be written to aborts the run.
"""

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, NoReturn

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from llm_eval_lab.models import (
    AggregateMetrics,
    BenchmarkSnapshot,
    CaseQuery,
    CaseResult,
    CaseStatus,
    EvaluationRecord,
    EvaluationResult,
    ModelResponse,
    Page,
    Run,
    RunQuery,
    RunStatus,
    RunSummary,
    RunTotals,
    StorageError,
)
from llm_eval_lab.scoring import (
    DEFAULT_ERROR_POLICY,
    CaseVerdict,
    normalize_error_policy,
    pass_rate,
    verdict_of,
)
from llm_eval_lab.storage import mappers
from llm_eval_lab.storage.orm import (
    BenchmarkSnapshotRow,
    CaseResultRow,
    EvaluationRow,
    ModelResponseRow,
    RunMetricsRow,
    RunRow,
)
from llm_eval_lab.utils.time import ensure_utc, utc_now

ITER_BATCH_SIZE = 200
"""How many case results `iter_for_run` pulls per round trip while streaming."""


def _new_id() -> str:
    """Return a fresh canonical lowercase uuid4 string, generated in the domain layer."""
    return str(uuid.uuid4())


def _raise_storage_error(operation: str, exc: SQLAlchemyError) -> NoReturn:
    """Re-raise a SQLAlchemy failure as the project's own exception type.

    Raising here rather than returning an exception for the caller to raise
    keeps every call site a single line and gives it a ``NoReturn`` type, so
    the type checker knows control does not continue past the handler.

    Raises:
        StorageError: always.
    """
    msg = f"{operation} failed: {exc}"
    raise StorageError(msg) from exc


class BenchmarkRepository:
    """Persistence for content-addressed benchmark snapshots."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to one session."""
        self._session = session

    async def upsert_snapshot(self, snapshot: BenchmarkSnapshot) -> None:
        """Store the snapshot, or do nothing if its digest is already present.

        The insert is FLUSHED immediately. The session runs with autoflush off,
        so without this a second upsert of the same digest inside one unit of
        work would not see the first one pending, would insert a duplicate, and
        would fail on the primary key at commit time - which is exactly what
        happens when a run re-persists a suite it has already snapshotted.

        The check-then-act above is only safe WITHIN one unit of work; it does
        not close the race ACROSS two. Two concurrent launches of the same
        suite each open their own session, each can see ``existing is None``
        before either commits, and both then insert - the loser's write fails
        its ``UNIQUE`` constraint on ``suite_hash``. That failure is not a
        fault: it says the exact fact a same-moment recheck would have found,
        that this snapshot is already stored. The insert therefore runs inside
        its own ``SAVEPOINT`` (``begin_nested``), so a losing insert rolls back
        only itself - leaving the SURROUNDING unit of work (the run row this
        launch is about to write) perfectly usable.

        The resulting ``IntegrityError`` is swallowed only after confirming
        the row it named is actually present, never on the exception type
        alone. ``suite_hash`` is today the only constraint this table can
        fail - but that catch would otherwise be narrow only by the accident
        of no other constraint existing yet, and would go on silently
        swallowing a real fault the day this table gains an ``FK`` or a
        ``CHECK``. The re-check is what makes it narrow by CONSTRUCTION
        instead: present means someone else won the exact race this method
        exists to tolerate, and the failure is discarded; absent means the
        integrity error was about something else entirely, and it is
        re-raised so it surfaces loudly rather than vanishing as a snapshot
        that, by the time the caller sees the response, was never actually
        persisted at all.
        """
        try:
            existing = await self._session.get(BenchmarkSnapshotRow, snapshot.suite_hash)
            if existing is not None:
                return
            try:
                async with self._session.begin_nested():
                    self._session.add(mappers.snapshot_to_row(snapshot))
                    await self._session.flush()
            except IntegrityError:
                winner = await self._session.get(BenchmarkSnapshotRow, snapshot.suite_hash)
                if winner is None:
                    raise
        except SQLAlchemyError as exc:
            _raise_storage_error("upsert_snapshot", exc)

    async def get_snapshot(self, suite_hash: str) -> BenchmarkSnapshot | None:
        """Fetch a snapshot by digest."""
        try:
            row = await self._session.get(BenchmarkSnapshotRow, suite_hash)
        except SQLAlchemyError as exc:
            _raise_storage_error("get_snapshot", exc)
        return None if row is None else mappers.row_to_snapshot(row)

    async def list_snapshots(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[BenchmarkSnapshot]:
        """List stored snapshots, newest first."""
        statement = (
            sa.select(BenchmarkSnapshotRow)
            .order_by(BenchmarkSnapshotRow.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        try:
            rows = (await self._session.scalars(statement)).all()
        except SQLAlchemyError as exc:
            _raise_storage_error("list_snapshots", exc)
        return [mappers.row_to_snapshot(row) for row in rows]


class RunRepository:
    """Persistence for runs and their lifecycle."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to one session."""
        self._session = session

    async def create(self, run: Run) -> None:
        """Insert a new run."""
        try:
            self._session.add(mappers.run_to_row(run))
            await self._session.flush()
        except SQLAlchemyError as exc:
            _raise_storage_error("create run", exc)

    async def get(self, run_id: str) -> Run | None:
        """Fetch one run by id."""
        try:
            row = await self._session.get(RunRow, run_id)
        except SQLAlchemyError as exc:
            _raise_storage_error("get run", exc)
        return None if row is None else mappers.row_to_run(row)

    async def update_status(  # noqa: PLR0913 - one heartbeat write per status transition
        self,
        run_id: str,
        status: RunStatus,
        *,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        updated_at: datetime | None = None,
        totals: RunTotals | None = None,
        error: str | None = None,
        warnings: tuple[str, ...] | None = None,
    ) -> None:
        """Advance a run's status and heartbeat in one write.

        Every datetime reaching this method must be timezone-aware UTC;
        :func:`~llm_eval_lab.utils.time.ensure_utc` normalizes defensively so a
        caller that forgot cannot write a naive value SQLite would hand back
        indistinguishable from a UTC one.
        """
        values: dict[str, Any] = {"status": status.value}
        if started_at is not None:
            values["started_at"] = ensure_utc(started_at)
        if completed_at is not None:
            values["completed_at"] = ensure_utc(completed_at)
        # `utc_now()` rather than `sa.func.now()`: on SQLite the latter renders
        # CURRENT_TIMESTAMP, which is naive and second-granular, against a
        # `DateTime(timezone=True)` column, while on PostgreSQL it is
        # transaction-start time. One clock, one precision, both backends.
        values["updated_at"] = ensure_utc(updated_at) if updated_at is not None else utc_now()
        if totals is not None:
            values.update(
                n_cases=totals.n_cases,
                n_completed=totals.n_completed,
                n_passed=totals.n_passed,
                n_errors=totals.n_errors,
                n_cancelled=totals.n_cancelled,
            )
        if error is not None:
            values["error"] = error
        if warnings is not None:
            values["warnings"] = list(warnings)

        try:
            await self._session.execute(
                sa.update(RunRow).where(RunRow.id == run_id).values(**values)
            )
        except SQLAlchemyError as exc:
            _raise_storage_error("update_status", exc)

    def _summary_statement(self, query: RunQuery) -> sa.Select[Any]:
        """Build the run-listing select, joined to metrics for the derived figures."""
        statement = sa.select(RunRow, RunMetricsRow).outerjoin(
            RunMetricsRow, RunMetricsRow.run_id == RunRow.id
        )
        for condition in self._filters(query):
            statement = statement.where(condition)
        order = RunRow.created_at.desc() if query.order == "-created_at" else RunRow.created_at
        return statement.order_by(order, RunRow.id).limit(query.limit).offset(query.offset)

    @staticmethod
    def _filters(query: RunQuery) -> list[sa.ColumnElement[bool]]:
        """Translate a run query into SQL predicates."""
        conditions: list[sa.ColumnElement[bool]] = []
        if query.status is not None:
            conditions.append(RunRow.status == query.status.value)
        if query.provider is not None:
            conditions.append(RunRow.provider == query.provider)
        if query.model is not None:
            conditions.append(RunRow.model == query.model)
        if query.suite_name is not None:
            conditions.append(RunRow.suite_name == query.suite_name)
        if query.suite_hash is not None:
            conditions.append(RunRow.suite_hash == query.suite_hash)
        if query.since is not None:
            conditions.append(RunRow.created_at >= ensure_utc(query.since))
        return conditions

    async def _verdicts_for(self, run_ids: Sequence[str]) -> dict[str, list[CaseVerdict]]:
        """Fetch the pass/fail verdicts of every case of every listed run, in one query.

        One query for the whole page rather than one per run. The listing needs
        real verdicts, not counters, because the pass-rate rule has to exclude
        score-only cases and a counter cannot say which cases those were.
        """
        if not run_ids:
            return {}
        statement = sa.select(
            CaseResultRow.run_id, CaseResultRow.status, CaseResultRow.passed
        ).where(CaseResultRow.run_id.in_(list(run_ids)))
        try:
            rows = (await self._session.execute(statement)).all()
        except SQLAlchemyError as exc:
            _raise_storage_error("list case verdicts", exc)
        grouped: dict[str, list[CaseVerdict]] = {}
        for run_id, status, passed in rows:
            grouped.setdefault(run_id, []).append(verdict_of(CaseStatus(status), passed=passed))
        return grouped

    @staticmethod
    def _summary(
        row: RunRow,
        metrics: RunMetricsRow | None,
        verdicts: Sequence[CaseVerdict],
    ) -> RunSummary:
        """Project one run row, plus its metrics row when present, into a summary.

        ``pass_rate`` comes from the metrics row when a rollup exists. Otherwise
        it is computed by :func:`llm_eval_lab.scoring.pass_rate`, the SAME
        function the ``show`` view uses. That is deliberate: this method used to
        reimplement the arithmetic as ``n_completed - n_errors``, which counted
        score-only cases in the denominator while ``show`` excluded them, so the
        two surfaces reported different pass rates for the same run.
        """
        if metrics is not None:
            rate = metrics.pass_rate
        else:
            rate = pass_rate(verdicts, error_policy=_error_policy_of(row)).rate
        return RunSummary(
            id=row.id,
            label=row.label,
            status=RunStatus(row.status),
            created_at=ensure_utc(row.created_at),
            completed_at=None if row.completed_at is None else ensure_utc(row.completed_at),
            provider=row.provider,
            model=row.model,
            suite_name=row.suite_name,
            suite_version=row.suite_version,
            suite_hash=row.suite_hash,
            n_cases=row.n_cases,
            n_completed=row.n_completed,
            n_errors=row.n_errors,
            pass_rate=rate,
            total_cost=None if metrics is None else _stored_run_cost(metrics.cost),
            p95_ms=None if metrics is None else _json_float(metrics.latency, "p95_ms"),
        )

    async def list(self, query: RunQuery) -> Page[RunSummary]:
        """List runs matching the query."""
        count_statement = sa.select(sa.func.count()).select_from(RunRow)
        for condition in self._filters(query):
            count_statement = count_statement.where(condition)
        try:
            total = (await self._session.scalar(count_statement)) or 0
            rows = (await self._session.execute(self._summary_statement(query))).all()
        except SQLAlchemyError as exc:
            _raise_storage_error("list runs", exc)
        verdicts = await self._verdicts_for([run.id for run, _ in rows])
        return Page[RunSummary](
            items=tuple(
                self._summary(run, metrics, verdicts.get(run.id, ())) for run, metrics in rows
            ),
            total=total,
            limit=query.limit,
            offset=query.offset,
        )

    async def mark_stale_running_as_interrupted(self, *, before: datetime) -> int:
        """Mark runs still RUNNING but not heartbeated since `before` as INTERRUPTED."""
        statement = (
            sa.update(RunRow)
            .where(RunRow.status == RunStatus.RUNNING.value, RunRow.updated_at < ensure_utc(before))
            .values(status=RunStatus.INTERRUPTED.value, error="interrupted")
        )
        try:
            result = await self._session.execute(statement)
        except SQLAlchemyError as exc:
            _raise_storage_error("mark_stale_running_as_interrupted", exc)
        # `Session.execute` is typed as returning `Result`, but an UPDATE always
        # produces a `CursorResult`, which is where `rowcount` lives.
        rowcount = getattr(result, "rowcount", 0)
        return int(rowcount) if isinstance(rowcount, int) else 0


def _error_policy_of(row: RunRow) -> str:
    """Read a run's error policy out of its persisted configuration.

    Falls back to the contract default when the stored config predates the
    field or holds a value the scoring rule does not recognise, so a listing
    never fails because of one unreadable row. The narrowing happens here, at
    the boundary where untyped JSON becomes a typed value, which is what lets
    `scoring.pass_rate` refuse an unknown policy outright everywhere else.
    """
    config = row.config
    policy = config.get("error_policy") if isinstance(config, dict) else None
    if not isinstance(policy, str):
        return DEFAULT_ERROR_POLICY
    try:
        return normalize_error_policy(policy)
    except ValueError:
        return DEFAULT_ERROR_POLICY


def _json_float(payload: dict[str, Any], key: str) -> float | None:
    """Read one float out of a JSON column, tolerating a missing or null entry."""
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _json_decimal(payload: dict[str, Any], key: str) -> Decimal | None:
    """Read one money value out of a JSON column, exactly.

    Money is written into the JSON column as its exact textual form, so it is
    read back through ``Decimal(str(...))`` rather than through ``float``. A
    round trip via a binary double is the one thing the money type exists to
    prevent.
    """
    value = payload.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _stored_run_cost(payload: dict[str, Any]) -> Decimal | None:
    """Return a stored rollup's candidate-model spend.

    ``RunSummary.total_cost`` carries exactly what ``AggregateMetrics.cost``
    calls ``total_cost``: the candidate model's cost, with judge spend excluded.
    The two surfaces publish the same quantity under the same name, which they
    did not when this folded ``judge_cost`` in and the rollup did not.
    ``RunSummary`` is frozen and has no judge field, so the separate judge
    figure is reported by ``llm-eval metrics`` rather than by the listing.

    ``None`` when nothing was priced, never ``Decimal(0)``: an unpriced run and
    a free one are different facts and the listing must not merge them.
    """
    return _json_decimal(payload, "total_cost")


class ResponseRepository:
    """Persistence for raw model responses. The single write path for a response."""

    def __init__(self, session: AsyncSession, *, max_raw_bytes: int | None = None) -> None:
        """Bind the repository to one session and the payload cap it enforces."""
        self._session = session
        self._max_raw_bytes = max_raw_bytes or mappers.DEFAULT_MAX_RAW_BYTES

    async def add(self, run_id: str, case_id: str, response: ModelResponse) -> None:
        """Store one response, keyed by (run_id, case_id)."""
        try:
            self._session.add(
                mappers.response_to_row(
                    run_id, case_id, response, max_raw_bytes=self._max_raw_bytes
                )
            )
            await self._session.flush()
        except SQLAlchemyError as exc:
            _raise_storage_error("add response", exc)

    async def get(self, run_id: str, case_id: str) -> ModelResponse | None:
        """Fetch one response."""
        try:
            row = await self._session.get(ModelResponseRow, (run_id, case_id))
        except SQLAlchemyError as exc:
            _raise_storage_error("get response", exc)
        return None if row is None else mappers.row_to_response(row)


class EvaluationRepository:
    """Persistence for individual evaluation results."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to one session."""
        self._session = session

    async def add_many(self, records: Sequence[EvaluationRecord]) -> None:
        """Store several evaluation records."""
        if not records:
            return
        try:
            self._session.add_all(
                [
                    mappers.evaluation_to_row(
                        record.run_id, record.case_id, record.result, row_id=_new_id()
                    )
                    for record in records
                ]
            )
            await self._session.flush()
        except SQLAlchemyError as exc:
            _raise_storage_error("add evaluations", exc)

    async def list_for_case(self, run_id: str, case_id: str) -> Sequence[EvaluationResult]:
        """List the evaluations recorded for one case."""
        statement = (
            sa.select(EvaluationRow)
            .where(EvaluationRow.run_id == run_id, EvaluationRow.case_id == case_id)
            .order_by(EvaluationRow.id)
        )
        try:
            rows = (await self._session.scalars(statement)).all()
        except SQLAlchemyError as exc:
            _raise_storage_error("list evaluations", exc)
        return [mappers.row_to_evaluation(row) for row in rows]

    async def map_for_run(self, run_id: str) -> dict[str, tuple[EvaluationResult, ...]]:
        """Return every evaluation of a run, grouped by case id.

        One query instead of one per case: hydrating a page of results
        otherwise costs a round trip per row.
        """
        statement = (
            sa.select(EvaluationRow)
            .where(EvaluationRow.run_id == run_id)
            .order_by(EvaluationRow.id)
        )
        try:
            rows = (await self._session.scalars(statement)).all()
        except SQLAlchemyError as exc:
            _raise_storage_error("map evaluations", exc)
        grouped: dict[str, list[EvaluationResult]] = {}
        for row in rows:
            grouped.setdefault(row.case_id, []).append(mappers.row_to_evaluation(row))
        return {case_id: tuple(items) for case_id, items in grouped.items()}


class CaseResultRepository:
    """Persistence for per-case results.

    Writes only the ``case_results`` row. The response and the evaluations are
    owned by their own repositories; see this module's docstring.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        responses: ResponseRepository,
        evaluations: EvaluationRepository,
    ) -> None:
        """Bind the repository to one session and the two it hydrates from."""
        self._session = session
        self._responses = responses
        self._evaluations = evaluations

    async def add(self, result: CaseResult) -> None:
        """Store one case result."""
        try:
            self._session.add(mappers.case_result_to_row(result, row_id=_new_id()))
            await self._session.flush()
        except SQLAlchemyError as exc:
            _raise_storage_error("add case result", exc)

    async def add_many(self, results: Sequence[CaseResult]) -> None:
        """Store several case results."""
        if not results:
            return
        try:
            self._session.add_all(
                [mappers.case_result_to_row(result, row_id=_new_id()) for result in results]
            )
            await self._session.flush()
        except SQLAlchemyError as exc:
            _raise_storage_error("add case results", exc)

    async def get(self, run_id: str, case_id: str) -> CaseResult | None:
        """Fetch one case result, hydrated with its response and evaluations."""
        statement = sa.select(CaseResultRow).where(
            CaseResultRow.run_id == run_id, CaseResultRow.case_id == case_id
        )
        try:
            row = (await self._session.scalars(statement)).first()
        except SQLAlchemyError as exc:
            _raise_storage_error("get case result", exc)
        if row is None:
            return None
        response = await self._responses.get(run_id, case_id)
        evaluations = tuple(await self._evaluations.list_for_case(run_id, case_id))
        return mappers.row_to_case_result(row, response=response, evaluations=evaluations)

    @staticmethod
    def _case_filters(run_id: str, query: CaseQuery) -> list[sa.ColumnElement[bool]]:
        """Translate a case query into SQL predicates."""
        conditions: list[sa.ColumnElement[bool]] = [CaseResultRow.run_id == run_id]
        if query.status is not None:
            conditions.append(CaseResultRow.status == query.status.value)
        if query.passed is not None:
            conditions.append(CaseResultRow.passed.is_(query.passed))
        if query.category is not None:
            conditions.append(CaseResultRow.category == query.category)
        if query.q is not None:
            conditions.append(CaseResultRow.case_id.like(f"%{query.q}%"))
        return conditions

    async def list_for_run(self, run_id: str, query: CaseQuery) -> Page[CaseResult]:
        """List the case results of one run, hydrated in two extra queries, not N."""
        conditions = self._case_filters(run_id, query)
        count_statement = sa.select(sa.func.count()).select_from(CaseResultRow).where(*conditions)
        statement = (
            sa.select(CaseResultRow)
            .where(*conditions)
            .order_by(CaseResultRow.case_id)
            .limit(query.limit)
            .offset(query.offset)
        )
        try:
            total = (await self._session.scalar(count_statement)) or 0
            rows = list((await self._session.scalars(statement)).all())
        except SQLAlchemyError as exc:
            _raise_storage_error("list case results", exc)

        # `tag` is a JSON array on the row, so it is filtered in Python rather
        # than in SQL: a JSON containment predicate is not portable between
        # SQLite and PostgreSQL, and this listing is already page-bounded.
        if query.tag is not None:
            rows = [row for row in rows if query.tag in row.tags]

        evaluations = await self._evaluations.map_for_run(run_id)
        items = [
            mappers.row_to_case_result(
                row,
                response=await self._responses.get(run_id, row.case_id),
                evaluations=evaluations.get(row.case_id, ()),
            )
            for row in rows
        ]
        return Page[CaseResult](
            items=tuple(items), total=total, limit=query.limit, offset=query.offset
        )

    async def completed_case_ids(self, run_id: str) -> frozenset[str]:
        """Return the ids already completed for this run, for resume support."""
        statement = sa.select(CaseResultRow.case_id).where(CaseResultRow.run_id == run_id)
        try:
            rows = (await self._session.scalars(statement)).all()
        except SQLAlchemyError as exc:
            _raise_storage_error("completed_case_ids", exc)
        return frozenset(rows)

    async def iter_for_run(self, run_id: str) -> AsyncIterator[CaseResult]:
        """Stream every case result of one run, for export without loading it all.

        Declared on the Protocol as a plain ``def`` returning an
        ``AsyncIterator``; implemented as an async generator, which is the same
        type. The caller must drain it inside the owning unit of work, because
        the session it reads through closes when that scope exits.
        """
        evaluations = await self._evaluations.map_for_run(run_id)
        offset = 0
        while True:
            statement = (
                sa.select(CaseResultRow)
                .where(CaseResultRow.run_id == run_id)
                .order_by(CaseResultRow.case_id)
                .limit(ITER_BATCH_SIZE)
                .offset(offset)
            )
            try:
                rows = (await self._session.scalars(statement)).all()
            except SQLAlchemyError as exc:
                _raise_storage_error("iter case results", exc)
            if not rows:
                return
            for row in rows:
                yield mappers.row_to_case_result(
                    row,
                    response=await self._responses.get(run_id, row.case_id),
                    evaluations=evaluations.get(row.case_id, ()),
                )
            offset += len(rows)


class MetricsRepository:
    """Persistence for the aggregate rollup. Exactly one row per run."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to one session."""
        self._session = session

    async def put(self, metrics: AggregateMetrics) -> None:
        """Insert or replace the rollup for one run."""
        try:
            await self._session.execute(
                sa.delete(RunMetricsRow).where(RunMetricsRow.run_id == metrics.run_id)
            )
            self._session.add(mappers.metrics_to_row(metrics))
            await self._session.flush()
        except SQLAlchemyError as exc:
            _raise_storage_error("put metrics", exc)

    async def get(self, run_id: str) -> AggregateMetrics | None:
        """Fetch the rollup for one run."""
        try:
            row = await self._session.get(RunMetricsRow, run_id)
        except SQLAlchemyError as exc:
            _raise_storage_error("get metrics", exc)
        return None if row is None else mappers.row_to_metrics(row)

    async def get_many(self, run_ids: Sequence[str]) -> dict[str, AggregateMetrics]:
        """Fetch rollups for several runs, keyed by run id. Misses are omitted."""
        if not run_ids:
            return {}
        statement = sa.select(RunMetricsRow).where(RunMetricsRow.run_id.in_(list(run_ids)))
        try:
            rows = (await self._session.scalars(statement)).all()
        except SQLAlchemyError as exc:
            _raise_storage_error("get_many metrics", exc)
        return {row.run_id: mappers.row_to_metrics(row) for row in rows}
