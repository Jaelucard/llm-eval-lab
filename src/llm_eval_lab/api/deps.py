"""What a request handler is given, and the two gates it passes through first.

Everything a route needs is assembled once, at startup, and hung on
``app.state`` as :class:`AppResources`. A route asks for it with one dependency
rather than constructing a service per request: the database engine, the
provider registry and the run manager are all process-scoped, and building them
per request would open a new engine for every poll of a run's status.

The bearer-token gate lives here too. It is a dependency rather than middleware
on purpose: middleware runs before CORS preflight is answered, and a preflight
carries no ``Authorization`` header, so token checking in middleware breaks
every browser client it is meant to protect.
"""

import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from llm_eval_lab.models import UnitOfWorkFactory
from llm_eval_lab.services import (
    CatalogService,
    ComparisonService,
    MetricsService,
    RunManager,
    RunService,
)
from llm_eval_lab.settings import Settings

BEARER_PREFIX = "Bearer "
"""The one authorization scheme this API accepts."""


@dataclass(frozen=True)
class AppResources:
    """Everything built once at startup and shared by every request.

    ``process_start`` is stamped before the startup sweep runs and is what the
    sweep compares ``Run.updated_at`` against, so a run heartbeated by THIS
    process is never mistaken for one abandoned by a previous one.
    """

    settings: Settings
    uow_factory: UnitOfWorkFactory
    runs: RunService
    catalog: CatalogService
    metrics: MetricsService
    comparisons: ComparisonService
    manager: RunManager
    process_start: datetime


def get_resources(request: Request) -> AppResources:
    """Return the process-wide resources hung on the application state.

    Raises:
        RuntimeError: when the application's lifespan has not run. That is a
            programming error - an app driven without its lifespan - and it is
            reported as one rather than being papered over with a lazily built
            second set of services.
    """
    resources: AppResources | None = getattr(request.app.state, "resources", None)
    if resources is None:
        msg = "application resources are unavailable: the lifespan did not run"
        raise RuntimeError(msg)
    return resources


type Resources = Annotated[AppResources, Depends(get_resources)]
"""The one dependency every route uses to reach its services."""


def _unauthorised() -> HTTPException:
    """Build the one 401 this gate raises, however the credential was wrong.

    One message for an absent header, a malformed one and a wrong value alike: a
    gate that distinguishes them tells a prober which half of its guess was
    right.
    """
    return HTTPException(
        status_code=401,
        detail="A bearer token is required. Send Authorization: Bearer <token>.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_token(request: Request) -> None:
    """Require a valid bearer token when one is configured.

    With no ``LLM_EVAL_API_TOKEN`` set the API is loopback-only - a non-loopback
    bind without a token is refused before anything listens - and this gate
    passes everything. With a token set, every route it guards requires
    ``Authorization: Bearer <token>``.

    Three properties, each of which has been a real defect in this gate.

    **It fails CLOSED on an empty configured token.**
    :class:`~llm_eval_lab.settings.Settings` already normalises a set-but-empty
    token to ``None``, so this branch is unreachable through settings; it exists
    because an empty secret reaching here by any other route - a hand-built
    ``Settings`` copy, a future configuration source - would otherwise make
    ``compare_digest("", "")`` authorise a request carrying no header at all.
    An empty expected value refuses every request rather than accepting them.

    **It never raises on a hostile header, and it still matches a token an
    operator wrote in their own alphabet.** Starlette decodes headers as latin-1
    and :func:`secrets.compare_digest` rejects ``str`` arguments holding a code
    point above 0x7F with a ``TypeError``, so one such byte in ``Authorization``
    used to produce an unauthenticated 500 and a logged traceback, at whatever
    rate the caller liked - on the one code path that must never raise. Both
    sides are compared as ``bytes``, which ``compare_digest`` accepts
    unconditionally, and the two encodings are chosen so the comparison is of
    WIRE BYTES: latin-1 on the presented value exactly reverses Starlette's
    decode and recovers what the client sent, and UTF-8 on the configured value
    is what a client encoding a non-ASCII token would have put on the wire.

    **The comparison is constant time**, so its duration does not leak how much
    of a guessed prefix was right.

    ``/api/health`` is deliberately NOT guarded: a process supervisor has to be
    able to ask whether the process is alive without holding a credential, and
    the answer reveals only that it is.

    Raises:
        HTTPException: 401 when the header is absent, malformed or wrong.
    """
    configured = get_resources(request).settings.api_token
    if configured is None:
        return
    expected = configured.get_secret_value()
    if not expected.strip():
        raise _unauthorised()
    header = request.headers.get("authorization", "")
    presented = header[len(BEARER_PREFIX) :] if header.startswith(BEARER_PREFIX) else ""
    if not presented:
        raise _unauthorised()
    if not secrets.compare_digest(
        presented.encode("latin-1", "replace"),
        expected.encode("utf-8"),
    ):
        raise _unauthorised()


__all__ = [
    "BEARER_PREFIX",
    "AppResources",
    "Resources",
    "get_resources",
    "require_token",
]
