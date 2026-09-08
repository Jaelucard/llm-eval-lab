"""Async engine and session construction, plus the SQLite runtime pragmas.

The pragmas are set on every connection through an event listener, at RUNTIME,
never in a migration. A pragma written into a migration would apply once, on
one machine, and would be silently absent from every other process that ever
opens the file.

* ``foreign_keys=ON`` - SQLite enforces foreign keys only when asked, per
  connection. Without it the ``ON DELETE CASCADE`` clauses in the schema are
  decorative and deleting a run leaves orphaned case results behind.
* ``journal_mode=WAL`` - lets the CLI read a database while a run writes to it.
* ``busy_timeout=5000`` - a concurrent writer waits five seconds instead of
  failing instantly with "database is locked".
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from llm_eval_lab.models import StorageError

SQLITE_PRAGMAS: tuple[tuple[str, str], ...] = (
    ("foreign_keys", "ON"),
    ("journal_mode", "WAL"),
    ("busy_timeout", "5000"),
)
"""Applied to every SQLite connection, in this order."""


def is_sqlite_url(url: str) -> bool:
    """Report whether a SQLAlchemy URL names a SQLite database."""
    return urlparse(url).scheme.split("+")[0] == "sqlite"


def sqlite_path_of(url: str) -> Path | None:
    """Return the on-disk path a SQLite URL points at, or ``None`` for in-memory."""
    if not is_sqlite_url(url):
        return None
    _, _, remainder = url.partition("///")
    if not remainder or remainder.startswith(":memory:"):
        return None
    return Path(remainder)


def _register_sqlite_pragmas(engine: AsyncEngine) -> None:
    """Attach the connection listener that applies the SQLite pragmas."""

    @sa.event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            for pragma, value in SQLITE_PRAGMAS:
                cursor.execute(f"PRAGMA {pragma}={value}")
        finally:
            cursor.close()


def create_engine(url: str, *, echo: bool = False) -> AsyncEngine:
    """Create the async engine for `url`, creating a SQLite parent directory if needed.

    Raises:
        StorageError: when the URL is unusable or the directory cannot be made.
    """
    path = sqlite_path_of(url)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"cannot create database directory {path.parent}: {exc.strerror or exc}"
            raise StorageError(msg) from exc
    try:
        engine = create_async_engine(url, echo=echo, future=True)
    except (SQLAlchemyError, ValueError) as exc:
        msg = f"cannot open database {url}: {exc}"
        raise StorageError(msg) from exc
    if is_sqlite_url(url):
        _register_sqlite_pragmas(engine)
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build the session factory every unit of work opens a session from."""
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
        class_=AsyncSession,
    )


@asynccontextmanager
async def engine_scope(url: str, *, echo: bool = False) -> AsyncIterator[AsyncEngine]:
    """Yield an engine for `url` and dispose of it when the block exits."""
    engine = create_engine(url, echo=echo)
    try:
        yield engine
    finally:
        await engine.dispose()
