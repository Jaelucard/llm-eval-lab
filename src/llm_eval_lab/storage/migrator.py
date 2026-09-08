"""Running Alembic migrations from inside the application.

``llm-eval db upgrade`` has to work without the user standing in the repository
root, so the Alembic configuration is built programmatically here rather than
read from a file next to the working directory. This is also the only module
outside ``migrations/`` that imports Alembic, which is what keeps the ban on
Alembic elsewhere in the codebase enforceable.

Alembic's API is synchronous. It is driven through
:meth:`~sqlalchemy.ext.asyncio.AsyncConnection.run_sync`, which hands the
migration a real synchronous connection borrowed from the async engine, and
that connection is passed down through ``config.attributes`` - the documented
way to let one ``env.py`` serve both ``alembic upgrade`` on the command line
and a programmatic upgrade from inside a running process.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Connection
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from llm_eval_lab.models import StorageError
from llm_eval_lab.redaction import redact_error_text

CONNECTION_ATTRIBUTE = "connection"
"""`config.attributes` key `migrations/env.py` looks for before opening its own engine."""


def find_migrations_path() -> Path:
    """Locate the migration script directory.

    Two candidates, both anchored to THIS FILE: the copy packaged inside the
    installed distribution, then the repository-root directory a source checkout
    has. The working directory is deliberately not consulted.

    That omission is the point. Alembic does not read migration scripts, it
    IMPORTS them: ``env.py`` and every version module are executed as Python. A
    fallback to ``Path.cwd() / "migrations"`` would therefore mean
    ``llm-eval db upgrade``, run from a directory someone else prepared,
    executes that directory's code. That is the same trust boundary decision
    D-CUSTOM protects, reached through a different door, so the fallback is
    absent rather than merely discouraged.

    Raises:
        StorageError: when neither candidate exists. The message names both
            places looked, because "alembic could not find its scripts" is
            otherwise one of the least actionable errors a user can receive.
    """
    packaged = Path(__file__).resolve().parents[1] / "migrations"
    checkout = Path(__file__).resolve().parents[3] / "migrations"
    for candidate in (packaged, checkout):
        if (candidate / "env.py").is_file():
            return candidate
    msg = (
        f"cannot find the migrations directory; looked in {packaged} and {checkout}. "
        f"This build is incomplete: migrations ship inside the package, and the "
        f"working directory is deliberately never searched."
    )
    raise StorageError(msg)


def alembic_config(*, script_location: Path | None = None) -> Config:
    """Build an Alembic configuration that needs no `alembic.ini` on disk."""
    config = Config()
    config.set_main_option("script_location", str(script_location or find_migrations_path()))
    config.set_main_option("version_path_separator", "os")
    return config


def _run_upgrade(connection: Connection, *, script_location: Path | None) -> None:
    """Run every pending migration against an open synchronous connection."""
    config = alembic_config(script_location=script_location)
    config.attributes[CONNECTION_ATTRIBUTE] = connection
    command.upgrade(config, "head")


async def upgrade_to_head(engine: AsyncEngine, *, script_location: Path | None = None) -> None:
    """Bring the database at `engine` up to the latest migration.

    Raises:
        StorageError: when the migration cannot be applied.
    """
    try:
        async with engine.begin() as connection:
            await connection.run_sync(_run_upgrade, script_location=script_location)
    except SQLAlchemyError as exc:
        # The engine already opened successfully by the time a migration runs,
        # so this is not the primary leak path - but a driver-level failure
        # during the migration (a dropped connection, an auth error mid-run)
        # can still echo the DSN it was using, so the same scrub applies. See
        # `engine.py::create_engine` for why `from None` replaces `from exc`.
        cause = redact_error_text(str(exc), url=engine.url.render_as_string(hide_password=False))
        msg = f"database upgrade failed: {type(exc).__name__}: {cause}"
        raise StorageError(msg) from None


def _read_revision(connection: Connection) -> str | None:
    """Read the applied revision from an open synchronous connection."""
    return MigrationContext.configure(connection).get_current_revision()


async def current_revision(engine: AsyncEngine) -> str | None:
    """Return the migration revision the database is currently at, or ``None``.

    Raises:
        StorageError: when the revision cannot be read.
    """
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(_read_revision)
    except SQLAlchemyError as exc:
        cause = redact_error_text(str(exc), url=engine.url.render_as_string(hide_password=False))
        msg = f"cannot read the database revision: {type(exc).__name__}: {cause}"
        raise StorageError(msg) from None
