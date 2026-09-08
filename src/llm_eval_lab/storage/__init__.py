"""Persistence: async SQLAlchemy engine, ORM, repositories and the unit of work.

Nothing above this package imports SQLAlchemy. Everything crossing the boundary
is a frozen domain model, and the repository Protocols in
``llm_eval_lab.models.protocols`` are the only shape the runner and the
services layer ever see.
"""

from llm_eval_lab.storage.engine import (
    SQLITE_PRAGMAS,
    create_engine,
    create_session_factory,
    engine_scope,
    is_sqlite_url,
    sqlite_path_of,
)
from llm_eval_lab.storage.migrator import current_revision, upgrade_to_head
from llm_eval_lab.storage.orm import Base
from llm_eval_lab.storage.uow import SqlAlchemyUnitOfWork, make_uow_factory

__all__ = [
    "SQLITE_PRAGMAS",
    "Base",
    "SqlAlchemyUnitOfWork",
    "create_engine",
    "create_session_factory",
    "current_revision",
    "engine_scope",
    "is_sqlite_url",
    "make_uow_factory",
    "sqlite_path_of",
    "upgrade_to_head",
]
