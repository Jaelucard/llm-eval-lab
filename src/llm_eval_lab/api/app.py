"""The application factory, its lifespan, and the three gates in front of it.

`create_app` builds a fully wired application without opening a socket, which
is what lets `scripts/export_openapi.py` produce the schema offline and lets a
test drive the whole surface in-process.

Three things happen before any handler runs.

**The bind is checked.** A non-loopback bind with no ``LLM_EVAL_API_TOKEN`` is
refused here, at construction, before anything listens. This API starts runs
that spend money against the operator's credentials; exposing it unauthenticated
on an address other machines can reach is not a configuration this project
offers, and a warning would be ignored.

**The body is capped.** A request body larger than ``max_request_bytes`` is
refused with ``413`` from its ``Content-Length`` alone, before a byte is read.
A body-bearing request that declines to declare its length is refused with
``411``: an unbounded chunked upload is precisely what the cap exists to stop,
and accepting it while pretending to have a cap would be worse than having none.

**The origin is checked.** CORS allows exactly the configured origins.
``Settings`` refuses ``*`` outright, so there is no configuration in which any
page a browser visits can start a run.

The lifespan opens one database engine for the process, builds the services on
it, sweeps runs left ``RUNNING`` by a dead process, and on the way out asks
every run this process is executing to stop and waits for it.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_eval_lab import __version__
from llm_eval_lab.api.deps import AppResources, require_token
from llm_eval_lab.api.errors import (
    CATALOG_RESPONSES,
    COMPARE_RESPONSES,
    HEALTH_RESPONSES,
    METRICS_RESPONSES,
    RUNS_RESPONSES,
    VERSION_RESPONSES,
    InsecureBindError,
    install_error_handlers,
    problem_response,
)
from llm_eval_lab.api.routers import catalog, compare, health, metrics, runs
from llm_eval_lab.api.schemas import ProblemDetail
from llm_eval_lab.models import StorageError
from llm_eval_lab.observability.logging import configure_logging, get_logger
from llm_eval_lab.services import (
    RunManager,
    RunService,
    build_catalog_service,
    build_comparison_service,
    build_metrics_service,
    build_run_service,
    open_unit_of_work_factory,
)
from llm_eval_lab.settings import Settings, get_settings, is_loopback_host
from llm_eval_lab.utils.time import utc_now

API_TITLE = "llm-eval-lab"
"""Title carried into the OpenAPI document the dashboard generates types from."""

API_DESCRIPTION = (
    "Local-first LLM evaluation: benchmarks, runs, statistics and regression gating. "
    "Errors are RFC 9457 problem details with the media type application/problem+json."
)

BODY_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH"})
"""Methods whose request body is capped."""

LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})
"""Documentation only. The actual check is `llm_eval_lab.settings.is_loopback_host`.

Kept and exported for anything that wants a quick, non-exhaustive example of
what counts as loopback; `require_loopback_or_token` never reads this set
itself, because the bind check now takes the address the SERVER was given,
which is not always the one in the settings, and because a finite set cannot
express "any address in 127.0.0.0/8" or "``[::1]`` in bracket notation" the way
`is_loopback_host` does. The empty string used to live in this set - it is
uvicorn's own spelling of "bind every interface", not a loopback address, and
is refused outright rather than treated as safe.
"""

SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}
"""Sent on every response. Cheap, and they cost a browser nothing to honour."""

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)
"""Applied to HTML responses. No inline script, no third-party origin, no framing.

The dashboard is served from the same origin as an API that returns benchmark
content and stored model output, so its own bundle is allowed and nothing else
is. Lane D builds against this rather than needing it relaxed afterwards. The
FastAPI documentation routes are exempt (`DOCS_PATHS`): they load a bundle from a
CDN by design, and they are a loopback development tool, not a shipped surface.
"""

DOCS_PATHS: frozenset[str] = frozenset({"/docs", "/redoc", "/docs/oauth2-redirect"})
"""Paths exempt from the content-security policy. See `CONTENT_SECURITY_POLICY`."""

STATIC_DIR = Path(__file__).parent / "static"
"""Where the frontend build hook drops the dashboard, when there is one."""

PLACEHOLDER_CSS_PATH = "/_placeholder.css"
"""Where the placeholder page's stylesheet is served from, same-origin."""

PLACEHOLDER_CSS = """body { font: 16px/1.5 system-ui, sans-serif; margin: 3rem auto;
       max-width: 40rem; padding: 0 1rem; }
code { background: #f2f2f2; padding: 0.1em 0.35em; border-radius: 3px; }
"""
"""The placeholder page's styles, served as a file rather than inlined.

An inline `<style>` block is exactly what `style-src 'self'` forbids, so the page
this project serves would have been the first thing its own content-security
policy blocked. Serving the same three rules from a same-origin path costs one
route and keeps the policy strict, which is the point: a policy relaxed to
accommodate the page that announces the API is a policy that will be relaxed
again for the dashboard.
"""

NO_DASHBOARD_HTML = f"""<!doctype html>
<meta charset="utf-8">
<title>llm-eval-lab</title>
<link rel="stylesheet" href="{PLACEHOLDER_CSS_PATH}">
<h1>llm-eval-lab</h1>
<p>The API is running. No dashboard build is bundled with this installation.</p>
<p>To build it, run <code>npm install &amp;&amp; npm run build</code> in <code>frontend/</code>,
then reinstall the package: the build copies <code>frontend/dist</code> into the
package's static directory.</p>
<p>The API itself is under <code>/api</code>, and its OpenAPI schema is at
<code>/openapi.json</code>.</p>
"""
"""Served at `/` when no dashboard was bundled. Says what to do, not just that it is missing.

Carries no inline `<style>` and no `<script>` at all, so it loads intact under
the strict `CONTENT_SECURITY_POLICY` this application sends with it.
"""


class BodySizeLimitMiddleware:
    """Refuse an oversized or undeclared request body before reading it.

    Pure ASGI rather than ``BaseHTTPMiddleware`` because the decision is made
    from the headers alone: there is no reason to buffer a request through a
    second task just to look at ``Content-Length``.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        """Wrap `app`, capping request bodies at `max_bytes`."""
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check the declared body size, then hand the request on."""
        if scope["type"] != "http" or scope["method"] not in BODY_METHODS:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is None:
            if "chunked" in headers.get("transfer-encoding", "").lower():
                await self._refuse(scope, send, status=411, title="Length required")
                return
            await self.app(scope, receive, send)
            return
        try:
            length = int(declared)
        except ValueError:
            await self._refuse(scope, send, status=400, title="Bad request")
            return
        if length > self.max_bytes:
            await self._refuse(scope, send, status=413, title="Request body too large")
            return
        await self.app(scope, receive, send)

    async def _refuse(self, scope: Scope, send: Send, *, status: int, title: str) -> None:
        """Send a problem detail without invoking the application."""
        detail = {
            411: (
                "A request body must declare its length. This API caps request bodies and "
                "cannot cap a body whose size is never stated."
            ),
            400: "The Content-Length header is not a number.",
            413: f"The request body exceeds the {self.max_bytes} byte cap.",
        }[status]
        response = problem_response(
            ProblemDetail(
                type="/problems/request-body",
                title=title,
                status=status,
                detail=detail,
                instance=scope.get("path"),
            )
        )
        await response(scope, _no_body, send)


async def _no_body() -> Message:
    """Receive channel for a response sent without reading the request body."""
    return {"type": "http.disconnect"}


class SecurityHeadersMiddleware:
    """Stamp the browser-facing defences onto every response.

    Pure ASGI so it can rewrite the header block as it passes without buffering
    the body. `nosniff` matters on an API that returns benchmark content and
    stored model output: without it a browser may re-interpret a JSON body as
    HTML on the strength of its bytes. The content-security policy is applied to
    HTML only, because it is the HTML surface - the placeholder page today, the
    dashboard tomorrow - that could execute anything.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap `app`."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Add the headers to the response `app` produces."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers.setdefault(name, value)
                content_type = headers.get("content-type", "")
                if content_type.startswith("text/html") and path not in DOCS_PATHS:
                    headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class SpaStaticFiles(StaticFiles):
    """Serve the bundled dashboard, falling back to `index.html` for client routes.

    Two rules. A path under ``/api`` is a 404 from here, always: the API routes
    are registered before this mount, so anything under ``/api`` that reaches it
    is a route that does not exist, and answering it with an HTML page would
    turn every client's typo into an unparseable success. Anything else that is
    not a file on disk gets ``index.html``, because a single-page dashboard owns
    its own routing and a deep link into it must survive a reload.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Return the file at `path`, or the dashboard shell for a client route."""
        if path == "api" or path.startswith("api/"):
            raise StarletteHTTPException(status_code=404, detail="Not found")
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:  # noqa: PLR2004 - the HTTP status, not a magic number
                raise
            return await super().get_response("index.html", scope)


def require_loopback_or_token(settings: Settings, host: str | None = None) -> None:
    """Refuse a non-loopback bind that has no bearer token configured.

    `host` is the address the server will ACTUALLY bind, which is not always
    ``settings.api_host``: any ASGI entry point can be handed a different one, and
    a check that only ever read the setting passed while the socket listened
    somewhere else. Callers that know the real address pass it;
    :func:`llm_eval_lab.api.server.build_server` is the one that always does.

    A blank or whitespace-only `host` is refused unconditionally, even with
    `settings.api_token` set: it is not a bind address an operator chose, it is
    uvicorn's own spelling of "bind every interface", and a caller that means
    to bind everywhere should say so with an explicit ``0.0.0.0`` or ``::``
    rather than by omission.

    Raises:
        InsecureBindError: when the bind would expose an unauthenticated API,
            or when `host` is blank.
    """
    bind = settings.api_host if host is None else host
    if not bind.strip():
        msg = (
            f"refusing to bind {bind!r}: a blank host is uvicorn's own spelling of "
            f"bind-every-interface, not a bind address an operator chose, so it is "
            f"refused whether or not LLM_EVAL_API_TOKEN is set. Bind an explicit "
            f"address, e.g. 127.0.0.1, or 0.0.0.0 if binding everywhere is intended."
        )
        raise InsecureBindError(msg)
    if is_loopback_host(bind) or settings.api_token is not None:
        return
    msg = (
        f"refusing to bind {bind}: an address other machines can reach "
        f"requires authentication, because this API starts runs that spend money. "
        f"Set LLM_EVAL_API_TOKEN, or bind 127.0.0.1."
    )
    raise InsecureBindError(msg)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the process's resources, reconcile abandoned runs, and tear down cleanly."""
    settings: Settings = app.state.settings
    logger: structlog.BoundLogger = get_logger("llm_eval_lab.api")
    process_start: datetime = utc_now()

    async with open_unit_of_work_factory(
        settings.resolved_database_url(),
        max_raw_bytes=settings.max_raw_bytes,
    ) as uow_factory:
        run_service = build_run_service(settings, uow_factory, logger)
        catalog_service = build_catalog_service(settings)
        manager = RunManager(runs=run_service, catalog=catalog_service, logger=logger)
        app.state.resources = AppResources(
            settings=settings,
            uow_factory=uow_factory,
            runs=run_service,
            catalog=catalog_service,
            metrics=build_metrics_service(uow_factory, logger),
            comparisons=build_comparison_service(uow_factory, logger),
            manager=manager,
            process_start=process_start,
        )
        await _sweep_abandoned_runs(run_service, logger, before=process_start)
        try:
            yield
        finally:
            await manager.shutdown()
            app.state.resources = None


async def _sweep_abandoned_runs(
    run_service: RunService,
    logger: structlog.BoundLogger,
    *,
    before: datetime,
) -> None:
    """Mark runs left RUNNING by a dead process as INTERRUPTED.

    A failure here is logged rather than raised. The sweep is reconciliation, not
    a precondition: if the database is unreachable the process should still come
    up and say so through ``/api/health``, which is more useful than a start-up
    crash with the reason buried in a supervisor's log.
    """
    try:
        interrupted = await run_service.sweep_stale_runs(before=before)
    except StorageError:
        logger.exception("startup_sweep_failed")
        return
    logger.info("startup_sweep", n_interrupted=interrupted, before=before.isoformat())


def create_app(settings: Settings | None = None, *, bind_host: str | None = None) -> FastAPI:
    """Build the application. Opens no socket and touches no database.

    **This factory does not, and cannot, enforce the bind.** It checks
    `bind_host` when one is given and ``settings.api_host`` otherwise, but any
    ASGI entry point may bind an address it never told this function about -
    ``uvicorn --factory llm_eval_lab.api.app:create_app --host 0.0.0.0`` is the
    obvious one. The enforcing check lives in
    :func:`llm_eval_lab.api.server.build_server`, which is handed the real
    address. A deployment path that does not go through it is responsible for
    setting ``LLM_EVAL_API_TOKEN``.

    Args:
        settings: process configuration; the environment's when omitted.
        bind_host: the address the caller intends to bind, when it knows it.

    Raises:
        InsecureBindError: when that bind would expose an unauthenticated API.
    """
    resolved = settings if settings is not None else get_settings()
    require_loopback_or_token(resolved, bind_host)

    # The "no credential reaches a log" guarantee is a property of the scrubbing
    # processor `configure_logging` installs, so it has to be installed before
    # the first request rather than by whichever caller happened to start us.
    # Only when nothing has configured structlog yet: `llm-eval serve` has
    # already applied the operator's --log-level and --log-format by this point,
    # and overriding them here would silently discard them.
    if not structlog.is_configured():
        configure_logging(
            level=resolved.log_level,
            fmt=resolved.log_format,
            log_prompts=resolved.log_prompts,
        )

    # With a token configured the schema and its documentation UI are guarded
    # like everything else (see `_install_schema_routes`), so FastAPI's own
    # unguarded routes are switched off and replaced.
    guarded_schema = resolved.api_token is not None
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        openapi_url=None if guarded_schema else "/openapi.json",
        docs_url=None if guarded_schema else "/docs",
        redoc_url=None if guarded_schema else "/redoc",
    )
    app.state.settings = resolved
    install_error_handlers(app)

    # Ordering: the body cap is added first so CORS ends up OUTSIDE it, which is
    # what lets a browser read the 413 rather than seeing an opaque CORS failure.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=resolved.max_request_bytes)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["authorization", "content-type"],
        expose_headers=["X-Metrics-Materialized", "X-Listing-Truncated"],
    )

    # Health carries no token guard: a supervisor must be able to ask whether the
    # process is alive without holding a credential, and the answer says nothing else.
    app.include_router(health.router, responses=HEALTH_RESPONSES)
    guard = [Depends(require_token)]
    # One response map per router, so the schema describes what each can actually
    # answer with. See `errors.responses_for`.
    for router, documented in (
        (health.version_router, VERSION_RESPONSES),
        (catalog.router, CATALOG_RESPONSES),
        (runs.router, RUNS_RESPONSES),
        (compare.router, COMPARE_RESPONSES),
        (metrics.router, METRICS_RESPONSES),
    ):
        app.include_router(router, dependencies=guard, responses=documented)

    if guarded_schema:
        _install_schema_routes(app)
    _install_dashboard(app)
    return app


def _install_schema_routes(app: FastAPI) -> None:
    """Re-add `/openapi.json`, `/docs` and `/redoc` behind the bearer-token gate.

    FastAPI's own schema routes sit outside the `/api` prefix, so the per-router
    token dependency never covered them. The document is not a credential, but it
    is the complete route table, request shapes and error taxonomy of a
    deployment that is only reachable off loopback BECAUSE a token was set, and
    handing that to an unauthenticated caller is free reconnaissance - with
    `/docs` throwing in an interactive client to probe with.

    A browser opening `/docs` will not be able to fetch `/openapi.json`, because
    it sends no `Authorization` header. That is the intended trade: the
    documentation UI is a loopback development convenience, and a loopback
    deployment needs no token and keeps FastAPI's own unguarded routes.
    """
    guard = [Depends(require_token)]

    @app.get("/openapi.json", include_in_schema=False, dependencies=guard)
    async def openapi_schema() -> JSONResponse:
        """Serve the OpenAPI document to an authenticated caller."""
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False, dependencies=guard)
    async def swagger_ui() -> HTMLResponse:
        """Serve the Swagger UI to an authenticated caller."""
        return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{API_TITLE} - docs")

    @app.get("/redoc", include_in_schema=False, dependencies=guard)
    async def redoc_ui() -> HTMLResponse:
        """Serve the ReDoc UI to an authenticated caller."""
        return get_redoc_html(openapi_url="/openapi.json", title=f"{API_TITLE} - docs")


def _install_dashboard(app: FastAPI) -> None:
    """Serve the bundled dashboard at `/`, or explain how to build one.

    Mounted LAST, after every API route, so the catch-all mount can only receive
    what the API did not claim.
    """
    if (STATIC_DIR / "index.html").is_file():
        app.mount("/", SpaStaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
        return

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_placeholder() -> HTMLResponse:
        """Explain that no dashboard build is bundled, and how to make one."""
        return HTMLResponse(NO_DASHBOARD_HTML)

    @app.get(PLACEHOLDER_CSS_PATH, include_in_schema=False)
    async def placeholder_stylesheet() -> Response:
        """Serve the placeholder page's styles from this origin.

        A file rather than an inline block, so `style-src 'self'` holds. See
        :data:`PLACEHOLDER_CSS`.
        """
        return Response(PLACEHOLDER_CSS, media_type="text/css; charset=utf-8")


__all__ = [
    "API_DESCRIPTION",
    "API_TITLE",
    "CONTENT_SECURITY_POLICY",
    "LOOPBACK_HOSTS",
    "NO_DASHBOARD_HTML",
    "PLACEHOLDER_CSS",
    "PLACEHOLDER_CSS_PATH",
    "SECURITY_HEADERS",
    "STATIC_DIR",
    "BodySizeLimitMiddleware",
    "SecurityHeadersMiddleware",
    "SpaStaticFiles",
    "create_app",
    "lifespan",
    "require_loopback_or_token",
]
