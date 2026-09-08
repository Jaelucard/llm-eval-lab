"""Shared pytest fixtures.

Markers (`external`, `slow`, `integration`, `e2e`) are registered in
`pyproject.toml` under `[tool.pytest.ini_options]`, alongside
`--strict-markers`, so an unregistered or misspelled marker fails collection
rather than being silently accepted.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import pytest
import structlog

from llm_eval_lab.models import (
    EvaluationContext,
    EvaluatorSettings,
    Provider,
    ProviderConfig,
    UnitOfWorkFactory,
)
from llm_eval_lab.services import open_unit_of_work_factory
from llm_eval_lab.settings import Settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

_NETWORK_BLOCKED_MESSAGE = (
    "network access blocked in tests; mark the test with @pytest.mark.external to opt in"
)


def _blocked_connect(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(_NETWORK_BLOCKED_MESSAGE)


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Block real network access in every test not marked `external`.

    Patches `socket.socket.connect` and `socket.create_connection`, which
    covers both raw sockets and every higher-level client built on top of
    them (httpx, requests, vendor SDKs). Tests that genuinely need the
    network opt in explicitly with `@pytest.mark.external`, and are
    deselected by default via `addopts = "-m 'not external'"`.
    """
    if request.node.get_closest_marker("external") is not None:
        yield
        return

    original_socket_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.create_connection = _blocked_connect
    try:
        yield
    finally:
        socket.socket.connect = original_socket_connect  # type: ignore[method-assign]
        socket.create_connection = original_create_connection


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Undo any structlog configuration a test performed.

    `configure_logging` is process-global: it binds a logger factory to whatever
    `sys.stderr` was at the time. A test that redirects stderr to a buffer to
    inspect log output therefore leaves every LATER test writing into a stream
    that pytest has since closed, which surfaces as an unrelated
    "I/O operation on closed file" several files away. Resetting here keeps that
    contained to the test that caused it.
    """
    yield
    structlog.reset_defaults()


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the directory holding the shared test fixture files."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root, where `migrations/` and `examples/` live."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    """Return an async SQLite URL for a database private to one test."""
    return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def settings(tmp_path: Path, db_url: str) -> Settings:
    """Build settings isolated from the developer's environment.

    ``_env_file=None`` is not optional. `Settings` resolves `.env` relative to
    the WORKING DIRECTORY, so without this a developer's own file would leak
    into every test and the suite would pass or fail differently on different
    machines.
    """
    return Settings(
        _env_file=None,
        database_url=db_url,
        data_dir=tmp_path / "data",
        concurrency=4,
        case_timeout_s=10.0,
    )


@pytest.fixture
def logger() -> structlog.BoundLogger:
    """Return a logger for components that require one injected."""
    bound: structlog.BoundLogger = structlog.get_logger("tests")
    return bound


def apply_migrations(url: str, repo_root: Path) -> None:
    """Bring a fresh database up to head by invoking Alembic as a subprocess.

    A subprocess rather than an import because `alembic` is a storage-only
    dependency: importing it from a test would violate the banned-import rule
    that keeps migrations out of the rest of the codebase.
    """
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=repo_root,
        env={**os.environ, "LLM_EVAL_DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"alembic upgrade head failed:\n{result.stdout}\n{result.stderr}")


@pytest.fixture
def migrated_db(db_url: str, repo_root: Path) -> str:
    """Return the URL of a database that has been migrated to head."""
    apply_migrations(db_url, repo_root)
    return db_url


@pytest.fixture
async def uow_factory(migrated_db: str) -> AsyncIterator[UnitOfWorkFactory]:
    """Yield a unit-of-work factory over a migrated, test-private database."""
    async with open_unit_of_work_factory(migrated_db) as factory:
        yield factory


class RefusingProviderFactory:
    """A provider factory that fails loudly if anything tries to use it.

    Handed to deterministic evaluators in tests. They must never call a model,
    and an assertion here is a far clearer failure than a silent network call
    or a mock that quietly returns nothing.
    """

    def __call__(self, config: ProviderConfig) -> Self:
        """Return self as the context manager for `config`."""
        del config
        return self

    async def __aenter__(self) -> Provider:
        """Refuse to open a provider.

        Raises:
            AssertionError: always.
        """
        msg = "a deterministic evaluator must not construct a provider"
        raise AssertionError(msg)

    async def __aexit__(self, *exc: object) -> None:
        """Close nothing; nothing was ever opened."""
        return


@pytest.fixture
def evaluation_context(logger: structlog.BoundLogger) -> EvaluationContext:
    """Build the context an evaluator is handed, with nothing real behind it."""

    return EvaluationContext(
        run_id="test-run",
        suite_hash="sha256:test",
        case_hash="sha256:case",
        provider_factory=RefusingProviderFactory(),
        deadline=None,
        logger=logger,
        settings=EvaluatorSettings(),
    )
