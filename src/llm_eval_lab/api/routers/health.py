"""Liveness and identity: the two endpoints a supervisor and a client start with.

``/api/health`` is the only route the bearer-token gate does not cover. A
process supervisor has to be able to ask whether the process is up without
holding a credential, and the answer tells it nothing else.
"""

import platform

from fastapi import APIRouter, Response

from llm_eval_lab import __version__
from llm_eval_lab.api.deps import Resources
from llm_eval_lab.api.schemas import API_SCHEMA_VERSION, HealthResponse, VersionResponse
from llm_eval_lab.models import RunQuery, StorageError

router = APIRouter(prefix="/api", tags=["health"])
"""Liveness only. Deliberately NOT behind the bearer-token gate."""

version_router = APIRouter(prefix="/api", tags=["health"])
"""Identity. Behind the token gate with everything else: a build and interpreter
version is fingerprinting material, and only `/api/health` has a reason to be
answerable without a credential."""


@router.get("/health")
async def health(resources: Resources, response: Response) -> HealthResponse:
    """Report whether this process, and the database behind it, are usable.

    The database is checked by ISSUING A QUERY. A health check that only
    confirms a connection object exists reports that the process is running,
    which is the one thing the caller could already see.

    Answers ``503`` when the query fails, so a supervisor keys off the status
    code rather than parsing the body.
    """
    reachable = True
    try:
        async with resources.uow_factory() as uow:
            await uow.runs.list(RunQuery(limit=1))
    except StorageError:
        reachable = False
        response.status_code = 503
    return HealthResponse(
        status="ok" if reachable else "degraded",
        version=__version__,
        database="ok" if reachable else "error",
        runs_in_flight=resources.manager.in_flight(),
    )


@version_router.get("/version")
async def version() -> VersionResponse:
    """Report the library version, the interpreter, and this API's schema version."""
    return VersionResponse(
        version=__version__,
        python=platform.python_version(),
        schema_version=API_SCHEMA_VERSION,
    )


__all__ = ["router", "version_router"]
