"""SQLAlchemy 2.x declarative models. The only place table shapes are defined.

No ORM instance ever leaves this package: repositories map rows to the frozen
domain models in ``llm_eval_lab.models`` and back, so nothing above the storage
layer can accidentally hold a live, session-bound object.

Three schema decisions worth stating outright.

**The naming convention is mandatory, not cosmetic.** SQLite cannot ``ALTER``
most constraints, so Alembic rewrites the table instead; that only works when
every constraint has a deterministic name Alembic can reproduce. Without a
convention, an unnamed ``CHECK`` or ``UNIQUE`` silently disappears on the first
batch migration.

**`model_responses` is keyed by (run_id, case_id).** A response has no identity
of its own beyond the case it answers, and a composite primary key makes "one
response per case per run" a structural guarantee rather than an index someone
has to remember to add.

**`run_metrics` has `run_id` as its primary key.** Exactly one metrics row per
run, enforced by the key itself. The rollup is computed by a later slice; the
table is created here so that slice adds no migration and ``alembic check``
never reports drift.
"""

from datetime import datetime
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from llm_eval_lab.storage.types import JSON_TYPE, MONEY_TYPE, UTC_DATETIME

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

UUID_LENGTH = 36
CASE_ID_LENGTH = 128
HASH_LENGTH = 80
NAME_LENGTH = 128
MODEL_LENGTH = 255
SHORT_ENUM_LENGTH = 32
PATH_LENGTH = 1024


class Base(DeclarativeBase):
    """Declarative base carrying the shared naming convention."""

    metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)


class BenchmarkSnapshotRow(Base):
    """A content-addressed copy of a resolved suite, deduplicated by digest."""

    __tablename__ = "benchmark_snapshots"

    suite_hash: Mapped[str] = mapped_column(sa.String(HASH_LENGTH), primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    version: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    n_cases: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)
    source_path: Mapped[str | None] = mapped_column(sa.String(PATH_LENGTH), nullable=True)
    body: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)


class RunRow(Base):
    """One execution of a suite against one model.

    Provider, model and the suite identifiers are denormalized out of the
    persisted `config` blob because `RunQuery` filters on them; a JSON path
    filter would be unindexable on SQLite.
    """

    __tablename__ = "runs"
    __table_args__ = (
        sa.Index("ix_runs_created_at", "created_at"),
        sa.Index("ix_runs_status", "status"),
        sa.Index("ix_runs_suite_hash", "suite_hash"),
    )

    id: Mapped[str] = mapped_column(sa.String(UUID_LENGTH), primary_key=True)
    label: Mapped[str | None] = mapped_column(sa.String(MODEL_LENGTH), nullable=True)
    status: Mapped[str] = mapped_column(sa.String(SHORT_ENUM_LENGTH), nullable=False)

    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTC_DATETIME, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(UTC_DATETIME, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)

    suite_name: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    suite_version: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    suite_hash: Mapped[str] = mapped_column(sa.String(HASH_LENGTH), nullable=False)
    provider: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    model: Mapped[str] = mapped_column(sa.String(MODEL_LENGTH), nullable=False)

    config: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)

    n_cases: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    n_completed: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    n_passed: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    n_errors: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    n_cancelled: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)

    baseline_run_id: Mapped[str | None] = mapped_column(sa.String(UUID_LENGTH), nullable=True)
    error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    warnings: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False)


class ModelResponseRow(Base):
    """The normalized provider response for one case of one run."""

    __tablename__ = "model_responses"

    run_id: Mapped[str] = mapped_column(
        sa.String(UUID_LENGTH),
        sa.ForeignKey("runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    case_id: Mapped[str] = mapped_column(sa.String(CASE_ID_LENGTH), primary_key=True)

    output_text: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    provider: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    model: Mapped[str] = mapped_column(sa.String(MODEL_LENGTH), nullable=False)
    requested_model: Mapped[str] = mapped_column(sa.String(MODEL_LENGTH), nullable=False)
    finish_reason: Mapped[str] = mapped_column(sa.String(SHORT_ENUM_LENGTH), nullable=False)
    usage: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)
    latency_ms: Mapped[float] = mapped_column(sa.Float, nullable=False)
    total_latency_ms: Mapped[float] = mapped_column(sa.Float, nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)
    attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=1)
    truncated: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    error: Mapped[dict[str, object] | None] = mapped_column(JSON_TYPE, nullable=True)
    provider_request_id: Mapped[str | None] = mapped_column(sa.String(MODEL_LENGTH), nullable=True)
    raw: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)


class CaseResultRow(Base):
    """Everything recorded about one case in one run, except its response and evaluations."""

    __tablename__ = "case_results"
    __table_args__ = (
        sa.UniqueConstraint("run_id", "case_id", name="uq_case_results_run_id_case_id"),
        sa.Index("ix_case_results_run_id_status", "run_id", "status"),
    )

    id: Mapped[str] = mapped_column(sa.String(UUID_LENGTH), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        sa.String(UUID_LENGTH),
        sa.ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    case_id: Mapped[str] = mapped_column(sa.String(CASE_ID_LENGTH), nullable=False)
    case_hash: Mapped[str] = mapped_column(sa.String(HASH_LENGTH), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(SHORT_ENUM_LENGTH), nullable=False)
    passed: Mapped[bool | None] = mapped_column(sa.Boolean, nullable=True)
    score: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    cost: Mapped[dict[str, object] | None] = mapped_column(JSON_TYPE, nullable=True)
    attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=1)
    started_at: Mapped[datetime | None] = mapped_column(UTC_DATETIME, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(UTC_DATETIME, nullable=True)
    error: Mapped[dict[str, object] | None] = mapped_column(JSON_TYPE, nullable=True)
    category: Mapped[str | None] = mapped_column(sa.String(NAME_LENGTH), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False)


class EvaluationRow(Base):
    """One evaluator's verdict on one case.

    The queryable fields are real columns; the full `EvaluationResult` is kept
    in `payload` so nothing is lost on the way back out, including a judge's
    complete provenance record.
    """

    __tablename__ = "evaluations"
    __table_args__ = (
        sa.Index("ix_evaluations_run_id_case_id", "run_id", "case_id"),
        sa.Index("ix_evaluations_run_id_evaluator_type", "run_id", "evaluator_type"),
    )

    id: Mapped[str] = mapped_column(sa.String(UUID_LENGTH), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        sa.String(UUID_LENGTH),
        sa.ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    case_id: Mapped[str] = mapped_column(sa.String(CASE_ID_LENGTH), nullable=False)
    evaluator_id: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    evaluator_type: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(SHORT_ENUM_LENGTH), nullable=False)
    passed: Mapped[bool | None] = mapped_column(sa.Boolean, nullable=True)
    score: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)


class RunMetricsRow(Base):
    """The aggregate rollup for one run. Exactly one row per run, by construction.

    Created in the initial migration even though the rollup is computed in a
    later slice, so that slice ships no migration of its own and
    ``alembic check`` stays clean.
    """

    __tablename__ = "run_metrics"

    run_id: Mapped[str] = mapped_column(
        sa.String(UUID_LENGTH),
        sa.ForeignKey("runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    computed_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False)
    price_table_id: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    price_table_version: Mapped[str] = mapped_column(sa.String(NAME_LENGTH), nullable=False)
    error_policy: Mapped[str] = mapped_column(sa.String(SHORT_ENUM_LENGTH), nullable=False)

    n_cases: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    n_completed: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    n_errors: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    n_timeouts: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    n_skipped: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    n_scored: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    n_score_only: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    n_passed: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    n_pass_denominator: Mapped[int] = mapped_column(sa.Integer, nullable=False)

    error_rate: Mapped[float] = mapped_column(sa.Float, nullable=False)
    pass_rate: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    pass_rate_ci_low: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    pass_rate_ci_high: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    mean_score: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    median_score: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    token_usage_coverage: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    cost_coverage: Mapped[float | None] = mapped_column(sa.Float, nullable=True)

    cost_per_case: Mapped[Decimal | None] = mapped_column(MONEY_TYPE, nullable=True)
    cost_per_successful_evaluation: Mapped[Decimal | None] = mapped_column(
        MONEY_TYPE, nullable=True
    )

    latency: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)
    tokens: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)
    cost: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)
    error_breakdown: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)
    by_category: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)
    by_tag: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)
    by_evaluator: Mapped[dict[str, object]] = mapped_column(JSON_TYPE, nullable=False)


__all__ = [
    "NAMING_CONVENTION",
    "Base",
    "BenchmarkSnapshotRow",
    "CaseResultRow",
    "EvaluationRow",
    "ModelResponseRow",
    "RunMetricsRow",
    "RunRow",
]
