"""Database lifecycle operations, exposed to the CLI without leaking SQLAlchemy.

``cli`` may not import ``storage`` (a forbidden-import contract enforces it), so
``llm-eval db upgrade`` reaches the migrator through this service. It is a thin
seam on purpose: it owns no logic beyond opening an engine and closing it.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from llm_eval_lab.models import UnitOfWorkFactory
from llm_eval_lab.storage.engine import engine_scope
from llm_eval_lab.storage.migrator import current_revision, upgrade_to_head
from llm_eval_lab.storage.uow import make_uow_factory


async def upgrade_database(url: str) -> str | None:
    """Bring the database at `url` to the latest migration and report the revision.

    Raises:
        StorageError: when the database cannot be opened or migrated.
    """
    async with engine_scope(url) as engine:
        await upgrade_to_head(engine)
        return await current_revision(engine)


async def database_revision(url: str) -> str | None:
    """Return the migration revision the database at `url` is currently at.

    Raises:
        StorageError: when the database cannot be opened or read.
    """
    async with engine_scope(url) as engine:
        return await current_revision(engine)


@asynccontextmanager
async def open_unit_of_work_factory(
    url: str,
    *,
    max_raw_bytes: int | None = None,
) -> AsyncIterator[UnitOfWorkFactory]:
    """Yield a unit-of-work factory backed by an engine for `url`, then dispose of it.

    This is the seam the CLI and the API open persistence through. Neither may
    import ``storage`` directly, and neither should be deciding when an engine
    is created or disposed.

    Raises:
        StorageError: when the database cannot be opened.
    """
    async with engine_scope(url) as engine:
        yield make_uow_factory(engine, max_raw_bytes=max_raw_bytes)
