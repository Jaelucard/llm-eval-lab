# ruff: noqa: INP001
# Alembic loads this file by path rather than importing it as part of a package,
# so it deliberately has no `__init__.py`; adding one would break `alembic`.
"""Alembic migration environment, async by default.

Serves two callers with one code path.

When ``config.attributes["connection"]`` is populated, a connection has already
been opened by the application (``llm-eval db upgrade``) and the migration runs
on it directly. That is what lets the CLI upgrade a database without opening a
second engine or a second connection pool.

Otherwise the URL is resolved from the environment or the application settings
and an async engine is created here, which is the path the ``alembic``
command-line tool takes.

``render_as_batch=True`` is mandatory rather than optional: SQLite cannot
``ALTER`` most constraints, so Alembic has to rewrite the table instead, and
that only works when every constraint carries a deterministic name from the
metadata naming convention.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from llm_eval_lab.settings import get_settings
from llm_eval_lab.storage.orm import Base

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

config = context.config
target_metadata = Base.metadata


def resolve_url() -> str:
    """Return the database URL for a command-line invocation."""
    return (
        config.get_main_option("sqlalchemy.url")
        or os.environ.get("LLM_EVAL_DATABASE_URL")
        or get_settings().resolved_database_url()
    )


def run_migrations_offline() -> None:
    """Emit migration SQL without connecting to a database."""
    context.configure(
        url=resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run the migrations against an open synchronous connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Open an async engine of our own and run the migrations through it."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = resolve_url()
    connectable = async_engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations, reusing an application-supplied connection when there is one."""
    existing = config.attributes.get("connection")
    if existing is not None:
        do_run_migrations(existing)
        return
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
