"""Runtime storage failures must never carry a driver's raw DSN text.

`create_engine` only sees a URL that failed to PARSE. Every commit, rollback
and repository write goes through the sites exercised here instead, and a
driver's connection-level failure text can echo the DSN it was using,
credentials included. The same scrub and the same severed cause chain apply.
"""

import traceback
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError  # noqa: TID251 - the failure under test

from llm_eval_lab.models import StorageError
from llm_eval_lab.storage.repositories import _raise_storage_error
from llm_eval_lab.storage.uow import SqlAlchemyUnitOfWork

CANARY_USER = "canary-user-2b9d"
CANARY_PASSWORD = "canary-pw-7f3a9c1e"  # noqa: S105 - a planted canary, not a credential
DSN = f"postgresql+asyncpg://{CANARY_USER}:{CANARY_PASSWORD}@dbhost:5432/evals"


def _driver_failure() -> OperationalError:
    """A driver error whose own text echoes the DSN, the way real ones can."""
    return OperationalError("COMMIT", {}, Exception(f"connection to {DSN} was lost"))


def _assert_scrubbed(exc: StorageError, prefix: str) -> None:
    text = str(exc) + "".join(traceback.format_exception(exc)) + repr(exc.args)
    assert CANARY_PASSWORD not in text
    assert CANARY_USER not in text
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
    assert str(exc).startswith(prefix)
    assert "OperationalError" in str(exc)
    assert "dbhost" in str(exc)


def _uow_with_failing(method: str) -> SqlAlchemyUnitOfWork:
    session: Any = AsyncMock()
    getattr(session, method).side_effect = _driver_failure()
    return SqlAlchemyUnitOfWork(session)


async def test_a_commit_failure_never_carries_the_driver_dsn() -> None:
    with pytest.raises(StorageError) as excinfo:
        await _uow_with_failing("commit").commit()
    _assert_scrubbed(excinfo.value, "commit failed: ")


async def test_a_rollback_failure_never_carries_the_driver_dsn() -> None:
    with pytest.raises(StorageError) as excinfo:
        await _uow_with_failing("rollback").rollback()
    _assert_scrubbed(excinfo.value, "rollback failed: ")


def test_a_repository_failure_never_carries_the_driver_dsn() -> None:
    with pytest.raises(StorageError) as excinfo:
        _raise_storage_error("insert benchmark", _driver_failure())
    _assert_scrubbed(excinfo.value, "insert benchmark failed: ")
