# ruff: noqa: INP001
# Alembic loads revision scripts by path rather than importing them as part of a
# package, so `migrations/versions/` deliberately has no `__init__.py`.
"""Initial schema: snapshots, runs, responses, case results, evaluations, metrics.

Everything the walking skeleton needs, plus the ``run_metrics`` table, which is
created here even though the rollup that fills it is computed in a later slice.
Pinning it now is deliberate: that slice then ships no migration of its own and
``alembic check`` never reports drift between the ORM metadata and the applied
schema.

``run_metrics`` has ``run_id`` as its PRIMARY KEY, so "exactly one metrics row
per run" is structural rather than an index somebody has to remember to add.

Index and constraint names come from the metadata naming convention on
``llm_eval_lab.storage.orm.Base``, reproduced through ``op.f()`` where Alembic
generated them. Deterministic names are what make SQLite's batch table-rebuild
strategy work at all: an unnamed constraint cannot be recreated after the
rebuild, so it would silently vanish.

Revision ID: 0001
Revises:
Create Date: 2026-09-06
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from llm_eval_lab.storage.types import MONEY_PRECISION, MONEY_SCALE, MONEY_STRING_LENGTH, Money

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID_LENGTH = 36
CASE_ID_LENGTH = 128
HASH_LENGTH = 80
NAME_LENGTH = 128
MODEL_LENGTH = 255
SHORT_ENUM_LENGTH = 32
PATH_LENGTH = 1024


def json_type() -> sa.types.TypeEngine[object]:
    """Return the portable JSON column type: `JSON` on SQLite, `JSONB` on PostgreSQL."""
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def money_type() -> sa.types.TypeEngine[object]:
    """Return the exact-decimal money type: text on SQLite, `NUMERIC` on PostgreSQL."""
    return Money(length=MONEY_STRING_LENGTH).with_variant(
        sa.Numeric(precision=MONEY_PRECISION, scale=MONEY_SCALE), "postgresql"
    )


def _create_benchmark_snapshots() -> None:
    """Create the content-addressed suite snapshot table."""
    op.create_table(
        "benchmark_snapshots",
        sa.Column("suite_hash", sa.String(length=HASH_LENGTH), nullable=False),
        sa.Column("name", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("version", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("n_cases", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_path", sa.String(length=PATH_LENGTH), nullable=True),
        sa.Column("body", json_type(), nullable=False),
        sa.PrimaryKeyConstraint("suite_hash", name=op.f("pk_benchmark_snapshots")),
    )


def _create_runs() -> None:
    """Create the runs table and the three indices run listings depend on."""
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=UUID_LENGTH), nullable=False),
        sa.Column("label", sa.String(length=MODEL_LENGTH), nullable=True),
        sa.Column("status", sa.String(length=SHORT_ENUM_LENGTH), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("suite_name", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("suite_version", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("suite_hash", sa.String(length=HASH_LENGTH), nullable=False),
        sa.Column("provider", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("model", sa.String(length=MODEL_LENGTH), nullable=False),
        sa.Column("config", json_type(), nullable=False),
        sa.Column("n_cases", sa.Integer(), nullable=False),
        sa.Column("n_completed", sa.Integer(), nullable=False),
        sa.Column("n_passed", sa.Integer(), nullable=False),
        sa.Column("n_errors", sa.Integer(), nullable=False),
        sa.Column("n_cancelled", sa.Integer(), nullable=False),
        sa.Column("baseline_run_id", sa.String(length=UUID_LENGTH), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("warnings", json_type(), nullable=False),
        sa.Column("tags", json_type(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
    )
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.create_index("ix_runs_created_at", ["created_at"], unique=False)
        batch_op.create_index("ix_runs_status", ["status"], unique=False)
        batch_op.create_index("ix_runs_suite_hash", ["suite_hash"], unique=False)


def _create_case_results() -> None:
    """Create the per-case result table, unique on (run_id, case_id)."""
    op.create_table(
        "case_results",
        sa.Column("id", sa.String(length=UUID_LENGTH), nullable=False),
        sa.Column("run_id", sa.String(length=UUID_LENGTH), nullable=False),
        sa.Column("case_id", sa.String(length=CASE_ID_LENGTH), nullable=False),
        sa.Column("case_hash", sa.String(length=HASH_LENGTH), nullable=False),
        sa.Column("status", sa.String(length=SHORT_ENUM_LENGTH), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("cost", json_type(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", json_type(), nullable=True),
        sa.Column("category", sa.String(length=NAME_LENGTH), nullable=True),
        sa.Column("tags", json_type(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f("fk_case_results_run_id_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_case_results")),
        sa.UniqueConstraint("run_id", "case_id", name="uq_case_results_run_id_case_id"),
    )
    with op.batch_alter_table("case_results", schema=None) as batch_op:
        batch_op.create_index("ix_case_results_run_id_status", ["run_id", "status"], unique=False)


def _create_evaluations() -> None:
    """Create the per-evaluation table."""
    op.create_table(
        "evaluations",
        sa.Column("id", sa.String(length=UUID_LENGTH), nullable=False),
        sa.Column("run_id", sa.String(length=UUID_LENGTH), nullable=False),
        sa.Column("case_id", sa.String(length=CASE_ID_LENGTH), nullable=False),
        sa.Column("evaluator_id", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("evaluator_type", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("status", sa.String(length=SHORT_ENUM_LENGTH), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("payload", json_type(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f("fk_evaluations_run_id_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluations")),
    )
    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.create_index("ix_evaluations_run_id_case_id", ["run_id", "case_id"], unique=False)
        batch_op.create_index(
            "ix_evaluations_run_id_evaluator_type", ["run_id", "evaluator_type"], unique=False
        )


def _create_model_responses() -> None:
    """Create the response table, keyed by (run_id, case_id)."""
    op.create_table(
        "model_responses",
        sa.Column("run_id", sa.String(length=UUID_LENGTH), nullable=False),
        sa.Column("case_id", sa.String(length=CASE_ID_LENGTH), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("model", sa.String(length=MODEL_LENGTH), nullable=False),
        sa.Column("requested_model", sa.String(length=MODEL_LENGTH), nullable=False),
        sa.Column("finish_reason", sa.String(length=SHORT_ENUM_LENGTH), nullable=False),
        sa.Column("usage", json_type(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("total_latency_ms", sa.Float(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.Column("error", json_type(), nullable=True),
        sa.Column("provider_request_id", sa.String(length=MODEL_LENGTH), nullable=True),
        sa.Column("raw", json_type(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f("fk_model_responses_run_id_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "case_id", name=op.f("pk_model_responses")),
    )


def _create_run_metrics() -> None:
    """Create the aggregate rollup table: exactly one row per run, by primary key."""
    op.create_table(
        "run_metrics",
        sa.Column("run_id", sa.String(length=UUID_LENGTH), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("price_table_id", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("price_table_version", sa.String(length=NAME_LENGTH), nullable=False),
        sa.Column("error_policy", sa.String(length=SHORT_ENUM_LENGTH), nullable=False),
        sa.Column("n_cases", sa.Integer(), nullable=False),
        sa.Column("n_completed", sa.Integer(), nullable=False),
        sa.Column("n_errors", sa.Integer(), nullable=False),
        sa.Column("n_timeouts", sa.Integer(), nullable=False),
        sa.Column("n_skipped", sa.Integer(), nullable=False),
        sa.Column("n_scored", sa.Integer(), nullable=False),
        sa.Column("n_score_only", sa.Integer(), nullable=False),
        sa.Column("n_passed", sa.Integer(), nullable=False),
        sa.Column("n_pass_denominator", sa.Integer(), nullable=False),
        sa.Column("error_rate", sa.Float(), nullable=False),
        sa.Column("pass_rate", sa.Float(), nullable=True),
        sa.Column("pass_rate_ci_low", sa.Float(), nullable=True),
        sa.Column("pass_rate_ci_high", sa.Float(), nullable=True),
        sa.Column("mean_score", sa.Float(), nullable=True),
        sa.Column("median_score", sa.Float(), nullable=True),
        sa.Column("token_usage_coverage", sa.Float(), nullable=True),
        sa.Column("cost_coverage", sa.Float(), nullable=True),
        sa.Column("cost_per_case", money_type(), nullable=True),
        sa.Column("cost_per_successful_evaluation", money_type(), nullable=True),
        sa.Column("latency", json_type(), nullable=False),
        sa.Column("tokens", json_type(), nullable=False),
        sa.Column("cost", json_type(), nullable=False),
        sa.Column("error_breakdown", json_type(), nullable=False),
        sa.Column("by_category", json_type(), nullable=False),
        sa.Column("by_tag", json_type(), nullable=False),
        sa.Column("by_evaluator", json_type(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f("fk_run_metrics_run_id_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_run_metrics")),
    )


def upgrade() -> None:
    """Create every table, parents before children so the foreign keys resolve."""
    _create_benchmark_snapshots()
    _create_runs()
    _create_case_results()
    _create_evaluations()
    _create_model_responses()
    _create_run_metrics()


def downgrade() -> None:
    """Drop every table, children before parents."""
    op.drop_table("run_metrics")
    op.drop_table("model_responses")
    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.drop_index("ix_evaluations_run_id_evaluator_type")
        batch_op.drop_index("ix_evaluations_run_id_case_id")
    op.drop_table("evaluations")
    with op.batch_alter_table("case_results", schema=None) as batch_op:
        batch_op.drop_index("ix_case_results_run_id_status")
    op.drop_table("case_results")
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_index("ix_runs_suite_hash")
        batch_op.drop_index("ix_runs_status")
        batch_op.drop_index("ix_runs_created_at")
    op.drop_table("runs")
    op.drop_table("benchmark_snapshots")
