"""Fixtures for driving the HTTP API in-process.

The application is exercised over ASGI rather than over a socket. The lifespan
is entered explicitly, in the test's own event loop, because that is where the
background run tasks the API launches have to live: driving the app through a
client that owns a second loop would leave a run executing in one loop while the
test's database fixture lived in another.

Everything shared is exposed as a FIXTURE rather than as an importable helper.
`tests/` is not a package, so a cross-module import from a test file works only
by accident of `sys.path`; a fixture is how pytest is meant to share this and it
keeps the test files free of import bookkeeping.

`raise_app_exceptions` defaults to ``True`` everywhere except the tests that
deliberately force an internal failure. An unexpected exception should stop a
test loudly rather than arrive as a tidy 500 nobody reads.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol

import httpx
import pytest

from llm_eval_lab.api.app import create_app
from llm_eval_lab.datasets.resolve import snapshot_of
from llm_eval_lab.models import Run, RunStatus, RunTotals
from llm_eval_lab.services import RunRequest, build_run_service
from llm_eval_lab.settings import Settings
from llm_eval_lab.utils.time import utc_now

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from datetime import datetime
    from pathlib import Path

    import structlog
    from fastapi import FastAPI

    from llm_eval_lab.models import UnitOfWorkFactory

BASE_URL = "http://api.test"
"""Any absolute base works; nothing resolves it, because nothing opens a socket."""


@asynccontextmanager
async def _running_app(
    app: FastAPI,
    *,
    raise_app_exceptions: bool = True,
) -> AsyncIterator[httpx.AsyncClient]:
    """Enter the application's lifespan and yield a client bound to it."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            yield client


class ClientFactory(Protocol):
    """Opens a lifespan-managed client for one application."""

    def __call__(
        self,
        app: FastAPI,
        *,
        raise_app_exceptions: bool = True,
    ) -> AbstractAsyncContextManager[httpx.AsyncClient]:
        """Return a context manager yielding a client for `app`."""
        ...


class SeedRuns(Protocol):
    """Writes run rows directly, without executing any of them."""

    async def __call__(
        self,
        *,
        count: int,
        status: RunStatus = RunStatus.COMPLETED,
        updated_at: datetime | None = None,
    ) -> tuple[str, ...]:
        """Write `count` rows and return their ids."""
        ...


@pytest.fixture
def api_settings(tmp_path: Path, migrated_db: str, fixtures_dir: Path) -> Settings:
    """Settings for an API bound to loopback over a migrated, test-private database.

    ``suites_root`` points at the shared suite fixtures, so every benchmark the
    API can address by name is one of them and nothing else on the machine is
    reachable through a suite name.
    """
    return Settings(
        _env_file=None,
        database_url=migrated_db,
        data_dir=tmp_path / "data",
        suites_root=fixtures_dir / "suites",
        concurrency=4,
        case_timeout_s=10.0,
    )


@pytest.fixture
def api_app(api_settings: Settings) -> FastAPI:
    """Build the application without entering its lifespan."""
    return create_app(api_settings)


@pytest.fixture
def client_for() -> ClientFactory:
    """Return the factory a test uses to drive an application it built itself."""
    return _running_app


@pytest.fixture
async def api_client(api_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Yield a client for an application whose lifespan has run."""
    async with _running_app(api_app) as client:
        yield client


@pytest.fixture
def smoke_run_request(fixtures_dir: Path) -> RunRequest:
    """A run request against the shared smoke suite and the fake provider."""
    return RunRequest(
        suite_path=fixtures_dir / "suites" / "smoke.yaml",
        provider="fake",
        model="fake-1",
        provider_options={"mode": "expected"},
    )


@pytest.fixture
def seed_runs(
    api_settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    smoke_run_request: RunRequest,
) -> SeedRuns:
    """Return a helper that writes run rows without executing them.

    The rows carry a real :class:`~llm_eval_lab.models.RunConfig`, resolved once
    from the smoke suite, so a listing over them exercises the same projection a
    listing over executed runs does. Executing twenty-five runs to test paging
    would spend twenty-five runs' worth of time proving something about a
    ``LIMIT`` clause.
    """
    prepared = build_run_service(api_settings, uow_factory, logger).prepare(smoke_run_request)

    async def _seed(
        *,
        count: int,
        status: RunStatus = RunStatus.COMPLETED,
        updated_at: datetime | None = None,
    ) -> tuple[str, ...]:
        stamp = updated_at if updated_at is not None else utc_now()
        ids: list[str] = []
        async with uow_factory() as uow:
            await uow.benchmarks.upsert_snapshot(
                snapshot_of(prepared.suite, source_path=prepared.config.suite_source)
            )
            for index in range(count):
                run_id = str(uuid.uuid4())
                ids.append(run_id)
                await uow.runs.create(
                    Run(
                        id=run_id,
                        label=f"seeded-{index}",
                        status=status,
                        created_at=stamp,
                        started_at=stamp,
                        completed_at=stamp if status is RunStatus.COMPLETED else None,
                        config=prepared.config,
                        totals=RunTotals(n_cases=len(prepared.cases)),
                        updated_at=stamp,
                    )
                )
            await uow.commit()
        return tuple(ids)

    return _seed


@pytest.fixture
def launch_body() -> dict[str, Any]:
    """The `POST /api/runs` body for the smoke suite against the fake provider."""
    return {
        "suite": "smoke",
        "provider": {"provider": "fake", "model": "fake-1", "options": {"mode": "expected"}},
    }
