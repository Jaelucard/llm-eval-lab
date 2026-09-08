"""Structural interfaces every layer depends on, and nothing depends on.

Nothing in this module mentions SQLAlchemy, FastAPI, Typer or a vendor SDK,
and ``tests/unit/test_contract_purity.py`` proves it. That is what lets the
runner and the services layer reach persistence and providers through types
alone, and what lets a test substitute an in-memory double without importing a
database driver.
"""

import builtins
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from llm_eval_lab.models.benchmark import BenchmarkSnapshot, ResolvedCase
from llm_eval_lab.models.common import ProviderConfig, ProviderRequest
from llm_eval_lab.models.enums import RunStatus
from llm_eval_lab.models.evaluation import EvaluationResult
from llm_eval_lab.models.response import ModelResponse
from llm_eval_lab.models.run import (
    AggregateMetrics,
    CaseQuery,
    CaseResult,
    EvaluationRecord,
    Page,
    Run,
    RunQuery,
    RunSummary,
    RunTotals,
)

if TYPE_CHECKING:  # pragma: no cover - imported for typing only, never at runtime
    import structlog


class EvaluatorSettings(BaseModel):
    """The evaluator-facing slice of configuration.

    Passed in rather than read from the environment, so an evaluator never
    touches ``settings.py`` and is trivially testable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_concurrency: int = Field(default=4, ge=1, le=64)
    max_output_chars: int = Field(default=8000, ge=1)
    max_regex_match_s: float = Field(default=2.0, gt=0.0)
    max_raw_bytes: int = Field(default=65536, ge=1024)


@runtime_checkable
class Provider(Protocol):
    """One attempt per call.

    Retries, backoff and rate-limit pacing belong to the runner (decision D4).
    """

    name: ClassVar[str]

    @property
    def config(self) -> ProviderConfig:
        """Return the configuration this provider was constructed with."""
        ...

    async def generate(self, request: ProviderRequest) -> ModelResponse:
        """Perform exactly one generation attempt.

        Returns a SUCCESSFUL ModelResponse, or raises ProviderError. Must never
        return a ModelResponse carrying an error; that shape is produced by the
        runner after retries.
        """
        ...

    async def aclose(self) -> None:
        """Release any transport resources held by this provider."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Optional capability. ``semantic_similarity`` requires it (decision D-SIM)."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text, aligned by index."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores one model response against one resolved case."""

    type: ClassVar[str]
    # `builtins.type` because the `type` ClassVar above shadows the builtin here.
    params_model: ClassVar[builtins.type[BaseModel]]
    needs_expected: ClassVar[bool]
    is_model_graded: ClassVar[bool]

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: "EvaluationContext",
    ) -> EvaluationResult:
        """Score one response.

        Must NOT raise for ordinary scoring failure - return ``status=ERROR``
        with ``error`` populated. May raise only ``asyncio.CancelledError``.
        """
        ...


class CredentialResolver(Protocol):
    """Resolves credential environment variable names to their values."""

    def resolve(self, env_var: str | None) -> SecretStr | None:
        """Return the credential for env_var, or None.

        Raises MissingCredentialError when the provider requires one and it is
        absent. Names the variable, never its value.
        """
        ...


class BenchmarkRepository(Protocol):
    """Persistence for content-addressed benchmark snapshots."""

    async def upsert_snapshot(self, snapshot: BenchmarkSnapshot) -> None:
        """Store the snapshot, or do nothing if its digest is already present."""
        ...

    async def get_snapshot(self, suite_hash: str) -> BenchmarkSnapshot | None:
        """Fetch a snapshot by digest."""
        ...

    async def list_snapshots(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[BenchmarkSnapshot]:
        """List stored snapshots, newest first."""
        ...


class RunRepository(Protocol):
    """Persistence for runs and their lifecycle."""

    async def create(self, run: Run) -> None:
        """Insert a new run."""
        ...

    async def get(self, run_id: str) -> Run | None:
        """Fetch one run by id."""
        ...

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
        """Advance a run's status and heartbeat in one write."""
        ...

    async def list(self, query: RunQuery) -> Page[RunSummary]:
        """List runs matching the query."""
        ...

    async def mark_stale_running_as_interrupted(self, *, before: datetime) -> int:
        """Mark runs still RUNNING but not heartbeated since `before` as INTERRUPTED.

        Returns the number of runs changed. Uses ``updated_at`` rather than
        ``started_at``, so a long healthy run is never mistaken for a dead one.
        """
        ...


class ResponseRepository(Protocol):
    """Persistence for raw model responses."""

    async def add(self, run_id: str, case_id: str, response: ModelResponse) -> None:
        """Store one response."""
        ...

    async def get(self, run_id: str, case_id: str) -> ModelResponse | None:
        """Fetch one response."""
        ...


class CaseResultRepository(Protocol):
    """Persistence for per-case results."""

    async def add(self, result: CaseResult) -> None:
        """Store one case result."""
        ...

    async def add_many(self, results: Sequence[CaseResult]) -> None:
        """Store several case results."""
        ...

    async def get(self, run_id: str, case_id: str) -> CaseResult | None:
        """Fetch one case result."""
        ...

    async def list_for_run(self, run_id: str, query: CaseQuery) -> Page[CaseResult]:
        """List the case results of one run."""
        ...

    async def completed_case_ids(self, run_id: str) -> frozenset[str]:
        """Return the ids already completed for this run, for resume support."""
        ...

    def iter_for_run(self, run_id: str) -> AsyncIterator[CaseResult]:
        """Stream every case result of one run, for export without loading it all."""
        ...


class EvaluationRepository(Protocol):
    """Persistence for individual evaluation results."""

    async def add_many(self, records: Sequence[EvaluationRecord]) -> None:
        """Store several evaluation records."""
        ...

    async def list_for_case(self, run_id: str, case_id: str) -> Sequence[EvaluationResult]:
        """List the evaluations recorded for one case."""
        ...


class MetricsRepository(Protocol):
    """Persistence for the aggregate rollup. Exactly one row per run."""

    async def put(self, metrics: AggregateMetrics) -> None:
        """Insert or replace the rollup for one run."""
        ...

    async def get(self, run_id: str) -> AggregateMetrics | None:
        """Fetch the rollup for one run."""
        ...

    async def get_many(self, run_ids: Sequence[str]) -> dict[str, AggregateMetrics]:
        """Fetch rollups for several runs, keyed by run id. Misses are omitted."""
        ...


class UnitOfWork(Protocol):
    """One transactional scope over every repository."""

    benchmarks: BenchmarkRepository
    runs: RunRepository
    responses: ResponseRepository
    cases: CaseResultRepository
    evaluations: EvaluationRepository
    metrics: MetricsRepository

    async def __aenter__(self) -> Self:
        """Begin the transactional scope."""
        ...

    async def __aexit__(self, *exc: object) -> None:
        """Roll back unless `commit` was called."""
        ...

    async def commit(self) -> None:
        """Commit the scope."""
        ...

    async def rollback(self) -> None:
        """Roll the scope back."""
        ...


type UnitOfWorkFactory = Callable[[], AbstractAsyncContextManager[UnitOfWork]]
"""Opens a fresh unit of work. Sessions are never shared across concurrent tasks."""


@dataclass(frozen=True)
class EvaluationContext:
    """Everything an evaluator may need, injected.

    Evaluators construct nothing themselves: a judge evaluator receives a
    provider factory rather than building a vendor client of its own.
    """

    run_id: str
    suite_hash: str
    case_hash: str
    provider_factory: Callable[[ProviderConfig], AbstractAsyncContextManager[Provider]]
    deadline: float | None
    logger: "structlog.BoundLogger"
    settings: EvaluatorSettings
