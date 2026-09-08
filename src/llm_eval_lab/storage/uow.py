"""The unit of work: one transactional scope over every repository.

A session is never shared across concurrent tasks. The runner opens a FRESH
unit of work per case write, which is what makes incremental persistence safe
under a ``TaskGroup``: two cases finishing at the same moment write through two
independent sessions rather than racing on one.

Exiting the scope without calling :meth:`commit` rolls back. That default is
deliberate - a write that reached an exception should not be half-applied
because somebody forgot an ``else`` branch.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Self

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from llm_eval_lab.models import (
    BenchmarkRepository as BenchmarkRepositoryProtocol,
)
from llm_eval_lab.models import (
    CaseResultRepository as CaseResultRepositoryProtocol,
)
from llm_eval_lab.models import (
    EvaluationRepository as EvaluationRepositoryProtocol,
)
from llm_eval_lab.models import (
    MetricsRepository as MetricsRepositoryProtocol,
)
from llm_eval_lab.models import (
    ResponseRepository as ResponseRepositoryProtocol,
)
from llm_eval_lab.models import (
    RunRepository as RunRepositoryProtocol,
)
from llm_eval_lab.models import (
    StorageError,
    UnitOfWork,
    UnitOfWorkFactory,
)
from llm_eval_lab.redaction import redact_error_text
from llm_eval_lab.storage.engine import create_session_factory
from llm_eval_lab.storage.repositories import (
    BenchmarkRepository,
    CaseResultRepository,
    EvaluationRepository,
    MetricsRepository,
    ResponseRepository,
    RunRepository,
)


class SqlAlchemyUnitOfWork:
    """One transaction, one session, one set of repositories bound to it."""

    def __init__(self, session: AsyncSession, *, max_raw_bytes: int | None = None) -> None:
        """Bind every repository to the same session.

        The attributes are annotated with the PROTOCOL types, not the concrete
        ones. A Protocol attribute is invariant, so a concrete annotation here
        would make this class fail to satisfy ``UnitOfWork`` even though every
        method matches - and satisfying that Protocol is the whole point of the
        class.
        """
        self._session = session
        responses = ResponseRepository(session, max_raw_bytes=max_raw_bytes)
        evaluations = EvaluationRepository(session)
        self.benchmarks: BenchmarkRepositoryProtocol = BenchmarkRepository(session)
        self.runs: RunRepositoryProtocol = RunRepository(session)
        self.responses: ResponseRepositoryProtocol = responses
        self.evaluations: EvaluationRepositoryProtocol = evaluations
        self.cases: CaseResultRepositoryProtocol = CaseResultRepository(
            session, responses=responses, evaluations=evaluations
        )
        self.metrics: MetricsRepositoryProtocol = MetricsRepository(session)

    @property
    def session(self) -> AsyncSession:
        """Return the session this scope owns. For migrations and tests only."""
        return self._session

    async def __aenter__(self) -> Self:
        """Begin the transactional scope."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Roll back unless `commit` was called."""
        await self.rollback()

    async def commit(self) -> None:
        """Commit the scope."""
        try:
            await self._session.commit()
        except SQLAlchemyError as exc:
            # A driver's failure text can echo the DSN it was using, so the
            # cause is scrubbed and not chained. See `engine.py::create_engine`.
            msg = f"commit failed: {type(exc).__name__}: {redact_error_text(str(exc))}"
            raise StorageError(msg) from None

    async def rollback(self) -> None:
        """Roll the scope back."""
        try:
            await self._session.rollback()
        except SQLAlchemyError as exc:
            msg = f"rollback failed: {type(exc).__name__}: {redact_error_text(str(exc))}"
            raise StorageError(msg) from None


def make_uow_factory(
    engine: AsyncEngine,
    *,
    max_raw_bytes: int | None = None,
) -> UnitOfWorkFactory:
    """Build the factory that opens a fresh unit of work per transactional scope."""
    session_factory: async_sessionmaker[AsyncSession] = create_session_factory(engine)

    @asynccontextmanager
    async def _open() -> AsyncIterator[UnitOfWork]:
        async with session_factory() as session:
            uow = SqlAlchemyUnitOfWork(session, max_raw_bytes=max_raw_bytes)
            async with uow:
                yield uow

    return _open
