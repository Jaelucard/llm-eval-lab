"""One mapping from this project's exceptions to RFC 9457 problem details.

Every failure the API can produce is described by one media type,
``application/problem+json``, and one shape. A client therefore parses errors
once rather than per endpoint, and the dashboard can render a failure it has
never seen before.

**Tracebacks never cross the wire, and neither do server file paths.** Each
mapped exception contributes only the facts a client can act on: the field
locations of a malformed suite, the NAME of an unset credential variable, the
NAME of an uninstalled extra. An unmapped exception is answered with an opaque
``500`` carrying a correlation id, and the traceback is written to the server
log under that same id - so the operator can find it and the client learns
nothing about the internals.

**Validation errors are re-rendered rather than forwarded.** FastAPI's own
``RequestValidationError`` payload includes the offending ``input`` verbatim,
which is exactly how a credential a client should never have sent would come
straight back out in the ``422``. :func:`request_validation_problem` keeps the
location, the message and the error type and drops the input, the context and
the documentation URL.

The ``type`` values are relative URIs, resolved by a client against the request
URI as RFC 9457 permits. A project that does not own a documentation domain
should not invent one.
"""

import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from llm_eval_lab.api.schemas import ApiFieldError, HealthResponse, ProblemDetail
from llm_eval_lab.models import (
    BenchmarkValidationError,
    EvaluatorConfigError,
    FieldError,
    LLMEvalError,
    MissingCredentialError,
    PricingError,
    ProviderNotInstalledError,
    RegressionInputError,
    StorageError,
)
from llm_eval_lab.observability.logging import get_logger
from llm_eval_lab.services import (
    RunNotFoundError,
    SuiteNameError,
    SuiteNotFoundError,
    TooManyRunsError,
)

PROBLEM_MEDIA_TYPE = "application/problem+json"
"""The media type RFC 9457 defines for these documents."""

OPAQUE_500_DETAIL = (
    "The server failed to handle this request. The failure has been logged with the "
    "correlation id in this document; quote it when reporting the problem."
)
"""The only thing a client is told about an unmapped failure. Deliberately contentless."""

logger: structlog.BoundLogger = get_logger("llm_eval_lab.api")
"""Where an unmapped failure's traceback goes, keyed by its correlation id."""


class InsecureBindError(LLMEvalError):
    """The API was asked to bind a non-loopback address with no token configured.

    Raised at application construction, before anything listens. Exposing an API
    that can start runs which spend money, on an address other machines can
    reach, with no authentication at all, is not a configuration this project
    offers.
    """


def problem_response(problem: ProblemDetail) -> JSONResponse:
    """Render one problem detail as an ``application/problem+json`` response."""
    return JSONResponse(
        status_code=problem.status,
        content=problem.model_dump(mode="json", exclude_none=True),
        media_type=PROBLEM_MEDIA_TYPE,
    )


def _problem(  # noqa: PLR0913 - one parameter per RFC 9457 member this project sets
    request: Request,
    *,
    status: int,
    title: str,
    detail: str,
    problem_type: str,
    errors: tuple[ApiFieldError, ...] | None = None,
    env_var: str | None = None,
    extra: str | None = None,
    correlation_id: str | None = None,
) -> JSONResponse:
    """Build a problem response for `request`."""
    return problem_response(
        ProblemDetail(
            type=problem_type,
            title=title,
            status=status,
            detail=detail,
            instance=_instance_of(request),
            errors=errors,
            env_var=env_var,
            extra=extra,
            correlation_id=correlation_id,
        )
    )


MAX_INSTANCE_CHARS = 200
"""Cap on the reflected request path in `instance`."""


def _instance_of(request: Request) -> str:
    """Return the request path for RFC 9457's `instance`, bounded in length.

    Reflecting the caller's own decoded path is a deliberate choice, not a
    default: `instance` exists to identify the specific occurrence of a problem,
    and a client correlating a failure to the request that caused it needs it.
    It is JSON-encoded into an `application/problem+json` body, so it is data
    rather than markup, and it is truncated so a multi-kilobyte probe path cannot
    make the error response larger than the request that provoked it.
    """
    path = request.url.path
    return path if len(path) <= MAX_INSTANCE_CHARS else path[:MAX_INSTANCE_CHARS] + "..."


def field_errors_of(errors: Sequence[FieldError]) -> tuple[ApiFieldError, ...]:
    """Project loader field errors into the API's own shape.

    Both members that could carry something the client should not see - the
    server's path to the file, and a repr of the file's own content - are
    dropped rather than filtered. See :class:`ApiFieldError`.
    """
    return tuple(
        ApiFieldError(location=item.location, case_id=item.case_id, message=item.message)
        for item in errors
    )


def request_validation_errors(details: Sequence[Any]) -> tuple[ApiFieldError, ...]:
    """Project pydantic's request errors, keeping location and message ONLY.

    The dropped members are the point of this function. ``input`` echoes the
    offending value, which for a request body carrying a credential in the wrong
    place would return that credential to the sender inside the very response
    that refused it. ``ctx`` can carry the same value again, and ``url`` is a
    pydantic documentation link that says nothing about this request.
    """
    projected: list[ApiFieldError] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        location = ".".join(str(part) for part in detail.get("loc", ()))
        projected.append(
            ApiFieldError(
                location=location or "<body>",
                case_id=None,
                message=str(detail.get("msg", "invalid value")),
            )
        )
    return tuple(projected)


def _documented(description: str) -> dict[str, Any]:
    """One entry of an OpenAPI `responses` map, carrying the problem schema."""
    return {"model": ProblemDetail, "description": description}


_BY_STATUS: dict[int, dict[str, Any]] = {
    400: _documented("The request was malformed, or a suite name was not usable."),
    401: _documented("A bearer token is required and was absent or wrong."),
    404: _documented("No run, case or benchmark suite has that identifier."),
    409: _documented("The request conflicts with the current state of the run."),
    411: _documented("A request body was sent without declaring its length."),
    413: _documented("The request body exceeds the configured cap."),
    422: _documented("The request or the benchmark suite it names did not validate."),
    424: _documented("A provider is unavailable, or its credential variable is unset."),
    429: _documented("Too many runs are already executing in this process."),
    500: _documented("An unmapped failure, reported with a correlation id and nothing else."),
    503: _documented("The database could not be reached."),
}
"""Every error this API documents, by status. Routers take the subset they can produce."""


def responses_for(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """Build the OpenAPI `responses` map for one router.

    Per router rather than one shared map for all of them. FastAPI otherwise
    documents a `422` of its own `HTTPValidationError` shape, which this API never
    sends and a generated client would build its error handling against; but
    attaching every status to every router is the same defect pointing the other
    way, telling a client that `/api/version` can answer `409`, `413` and `424`
    when it cannot.
    """
    return {status: _BY_STATUS[status] for status in statuses}


CATALOG_RESPONSES = responses_for(400, 401, 404, 411, 413, 422, 500, 503)
"""Suite names can be unusable (400) or name nothing (404); bodies are capped."""

RUNS_RESPONSES = responses_for(400, 401, 404, 409, 411, 413, 422, 424, 429, 500, 503)
"""Everything: launching a run can fail on the suite, the provider or the database."""

COMPARE_RESPONSES = responses_for(401, 404, 409, 411, 413, 422, 500, 503)
"""Two runs may not exist (404) or not be comparable (409)."""

METRICS_RESPONSES = responses_for(401, 500, 503)
"""A read with no parameters beyond the token."""

VERSION_RESPONSES = responses_for(401, 500)
"""Constant facts about the process. It does not touch the database."""

HEALTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    500: _BY_STATUS[500],
    503: {
        "model": HealthResponse,
        "description": (
            "The database could not be reached. The body is still the health report, "
            "not a problem detail: the caller asked for the health of the process and "
            "that question was answered."
        ),
    },
}
"""The unguarded health route. Its 503 carries a `HealthResponse`, not a problem detail."""


type _Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]


async def benchmark_invalid(request: Request, exc: Exception) -> JSONResponse:
    """Map an invalid benchmark file to 422 with every field error it carried.

    The detail is COUNTED here, never forwarded. ``BenchmarkValidationError``'s
    message is assembled by the loader as ``{absolute path}: N validation
    error(s)`` followed by one line per error including a repr of the offending
    input - so ``str(exc)`` put both the server's file system and the benchmark
    file's content into a response, in the same handler whose module docstring
    promises neither ever crosses the wire. The structured ``errors`` member
    carries everything a client can act on.
    """
    found = exc.errors if isinstance(exc, BenchmarkValidationError) else ()
    plural = "" if len(found) == 1 else "s"
    return _problem(
        request,
        status=422,
        title="Benchmark suite is invalid",
        detail=(
            f"The benchmark suite has {len(found)} validation error{plural}; "
            f"see the errors member for the location of each."
        ),
        problem_type="/problems/benchmark-invalid",
        errors=field_errors_of(found),
    )


async def request_validation_problem(request: Request, exc: Exception) -> JSONResponse:
    """Map a request that failed validation to 422, echoing none of its input."""
    details = exc.errors() if isinstance(exc, RequestValidationError) else ()
    return _problem(
        request,
        status=422,
        title="Request is invalid",
        detail="The request body or its parameters did not validate.",
        problem_type="/problems/request-invalid",
        errors=request_validation_errors(details),
    )


async def credential_missing(request: Request, exc: Exception) -> JSONResponse:
    """Map an unset credential variable to 424, naming the variable only.

    The NAME is read off the exception, which the raise site sets as an
    attribute. It used to be recovered by splitting the message on whitespace
    and hoping the first two words were still "environment variable" - a
    coupling between two layers that nothing pinned, so a reworded message would
    have turned the member into ``null`` without a single test noticing.
    """
    return _problem(
        request,
        status=424,
        title="A required credential is not configured",
        detail=str(exc),
        problem_type="/problems/credential-missing",
        env_var=_attribute(exc, "env_var"),
    )


def _attribute(exc: Exception, name: str) -> str | None:
    """Read one string extension member off an exception, or ``None``."""
    value = getattr(exc, name, None)
    return value if isinstance(value, str) and value else None


async def provider_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """Map an uninstalled or unknown provider to 424, naming the extra it needs.

    ``extra`` is the name of the optional install extra, when the provider is
    one this build knows about but cannot construct. It is absent for a provider
    name nobody registered, because there is no extra that would fix a typo.
    """
    return _problem(
        request,
        status=424,
        title="The requested provider is not available",
        detail=str(exc),
        problem_type="/problems/provider-unavailable",
        extra=_attribute(exc, "extra"),
    )


async def runs_incomparable(request: Request, exc: Exception) -> JSONResponse:
    """Map two runs that cannot be compared to 409."""
    return _problem(
        request,
        status=409,
        title="The two runs cannot be compared",
        detail=str(exc),
        problem_type="/problems/runs-incomparable",
    )


async def run_not_found(request: Request, exc: Exception) -> JSONResponse:
    """Map an unknown run id to 404."""
    return _problem(
        request,
        status=404,
        title="No such run",
        detail=str(exc),
        problem_type="/problems/run-not-found",
    )


async def suite_not_found(request: Request, exc: Exception) -> JSONResponse:
    """Map a suite name that names nothing to 404."""
    return _problem(
        request,
        status=404,
        title="No such benchmark suite",
        detail=str(exc),
        problem_type="/problems/suite-not-found",
    )


async def suite_name_invalid(request: Request, exc: Exception) -> JSONResponse:
    """Map an unusable or escaping suite name to 400.

    The detail repeats the name the client sent and the rule it broke. It never
    contains the resolved absolute path, which would describe the server's file
    system to whoever probed it.
    """
    return _problem(
        request,
        status=400,
        title="Benchmark suite name is not usable",
        detail=str(exc),
        problem_type="/problems/suite-name-invalid",
    )


async def too_many_runs(request: Request, exc: Exception) -> JSONResponse:
    """Map a launch beyond this process's concurrency cap to 429."""
    return _problem(
        request,
        status=429,
        title="Too many runs are already executing",
        detail=str(exc),
        problem_type="/problems/too-many-runs",
    )


async def policy_invalid(request: Request, exc: Exception) -> JSONResponse:
    """Map an unusable threshold policy or evaluator configuration to 422."""
    return _problem(
        request,
        status=422,
        title="Configuration is not usable",
        detail=str(exc),
        problem_type="/problems/policy-invalid",
    )


async def pricing_unusable(request: Request, exc: Exception) -> JSONResponse:
    """Map a price table that cannot be loaded to 422."""
    return _problem(
        request,
        status=422,
        title="Price table is not usable",
        detail=str(exc),
        problem_type="/problems/price-table-invalid",
    )


async def storage_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """Map a persistence failure to 503, saying nothing about the database itself."""
    correlation_id = uuid.uuid4().hex
    logger.error(
        "api_storage_error",
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )
    return _problem(
        request,
        status=503,
        title="The database is unavailable",
        detail=(
            "The request could not be served because the database could not be reached. "
            "The failure has been logged with the correlation id in this document."
        ),
        problem_type="/problems/storage-unavailable",
        correlation_id=correlation_id,
    )


async def http_problem(request: Request, exc: Exception) -> JSONResponse:
    """Render Starlette's own HTTP errors - 401, 404, 405 - in the same problem shape."""
    status = exc.status_code if isinstance(exc, StarletteHTTPException) else 500
    detail = str(exc.detail) if isinstance(exc, StarletteHTTPException) else "Request failed."
    response = _problem(
        request,
        status=status,
        title=_http_title(status),
        detail=detail,
        problem_type="/problems/http-error",
    )
    # `WWW-Authenticate` on a 401 and `Allow` on a 405 are part of the answer,
    # not decoration: a client that cannot see them cannot tell what to do next.
    headers = getattr(exc, "headers", None)
    if isinstance(headers, dict):
        for name, value in headers.items():
            response.headers[str(name)] = str(value)
    return response


def _http_title(status: int) -> str:
    """Return a short title for one of Starlette's own HTTP errors."""
    titles = {
        400: "Bad request",
        401: "Authentication required",
        403: "Forbidden",
        404: "Not found",
        405: "Method not allowed",
        411: "Length required",
        413: "Request body too large",
    }
    return titles.get(status, "Request failed")


async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Answer an unmapped failure with an opaque 500 and log it under a correlation id.

    This is the one broad catch the API performs, and it is the reason no other
    layer needs one. The traceback goes to the log; the client gets a status, a
    title and an id. Nothing about the exception type, the message, the module or
    the file it came from crosses the wire, because any of those is a free
    description of the server's internals to whoever provoked the failure.
    """
    correlation_id = uuid.uuid4().hex
    logger.error(
        "unhandled_api_error",
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )
    return _problem(
        request,
        status=500,
        title="Internal server error",
        detail=OPAQUE_500_DETAIL,
        problem_type="/problems/internal-error",
        correlation_id=correlation_id,
    )


HANDLERS: tuple[tuple[type[Exception], _Handler], ...] = (
    # Most specific first is not required - Starlette walks the raised
    # exception's MRO and takes the first registered ancestor - but the
    # subclass entries below (SuiteNotFoundError before SuiteNameError,
    # MissingCredentialError before ProviderNotInstalledError) are what make
    # that walk land on the narrower answer.
    (BenchmarkValidationError, benchmark_invalid),
    (RequestValidationError, request_validation_problem),
    (MissingCredentialError, credential_missing),
    (ProviderNotInstalledError, provider_unavailable),
    (RegressionInputError, runs_incomparable),
    (TooManyRunsError, too_many_runs),
    (RunNotFoundError, run_not_found),
    (SuiteNotFoundError, suite_not_found),
    (SuiteNameError, suite_name_invalid),
    (EvaluatorConfigError, policy_invalid),
    (PricingError, pricing_unusable),
    (StorageError, storage_unavailable),
    (StarletteHTTPException, http_problem),
    (Exception, unhandled),
)
"""Every exception this API maps, paired with the handler that renders it."""


def install_error_handlers(app: FastAPI) -> None:
    """Register every problem-detail handler on `app`."""
    for exception_type, handler in HANDLERS:
        app.add_exception_handler(exception_type, handler)


__all__ = [
    "CATALOG_RESPONSES",
    "COMPARE_RESPONSES",
    "HEALTH_RESPONSES",
    "METRICS_RESPONSES",
    "OPAQUE_500_DETAIL",
    "PROBLEM_MEDIA_TYPE",
    "RUNS_RESPONSES",
    "VERSION_RESPONSES",
    "ApiFieldError",
    "InsecureBindError",
    "ProblemDetail",
    "field_errors_of",
    "install_error_handlers",
    "problem_response",
    "request_validation_errors",
    "responses_for",
]
