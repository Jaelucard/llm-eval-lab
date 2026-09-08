"""The run lifecycle: launch, poll, inspect, cancel, export.

``POST /api/runs`` answers ``202`` and a run id. It does not answer the result,
because a benchmark run takes minutes and an HTTP request that waits for one is
a request that times out. Everything decidable up front is decided before the
``202`` - the suite is loaded and validated, the provider name is resolved, the
price table is read - so a client learns about a malformed suite from the
response to its own request rather than from a poll thirty seconds later.

Cancellation is cooperative and honest about its limits. A run this process is
not executing cannot be stopped by it, and the ``409`` says so rather than
returning ``202`` for an instruction nothing will carry out.
"""

import io
import json
import re
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse

from llm_eval_lab.api.deps import AppResources, Resources
from llm_eval_lab.api.schemas import (
    API_SCHEMA_VERSION,
    CancelResponse,
    CaseResultSummary,
    CreateRunRequest,
    CreateRunResponse,
    RunDetail,
    RunLinks,
    RunStatusResponse,
)
from llm_eval_lab.models import (
    AggregateMetrics,
    CaseQuery,
    CaseResult,
    CaseStatus,
    JSONValue,
    Page,
    RunQuery,
    RunStatus,
    RunSummary,
)
from llm_eval_lab.reporting.formatters import write_case_csv
from llm_eval_lab.services import (
    CancelOutcome,
    LaunchRequest,
    RunNotFoundError,
    export_document,
)

router = APIRouter(prefix="/api", tags=["runs"])

STREAM_CHUNK_BYTES = 65536
"""How much of an export is handed to the transport at a time.

Bytes, which is what `_chunks` slices; it was named for characters and sliced
encoded bytes, which are the same thing only for ASCII.
"""

RUN_ID_PATTERN = re.compile(r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
"""The contract's run-id shape: a canonical lowercase uuid4 string.

Used to decide whether an id may be interpolated into a `Content-Disposition`
header. Ids are generated as `str(uuid.uuid4())`, so today every real one
matches; the check is here because that is a property of the generator and not
of this line, and a future import path carrying a quote or a newline into a
response header would be a header-injection bug found the hard way.
"""

MATERIALIZED_HEADER = "X-Metrics-Materialized"
"""Says whether a rollup was read from its stored row or derived for this request.

An interrupted run has no stored rollup, so its figures are computed from the
cases persisted so far. That is a weaker claim than a rollup written when a run
completed, and a client that renders the two identically is misreporting one of
them. The body is exactly `AggregateMetrics`, so the distinction lives in a
header rather than in a wrapper that would change the documented body shape.
"""


def links_for(run_id: str) -> RunLinks:
    """Build the follow-up links for one run, so no client hard-codes a path."""
    base = f"/api/runs/{run_id}"
    return RunLinks(
        self=base,
        status=f"{base}/status",
        cases=f"{base}/cases",
        metrics=f"{base}/metrics",
        cancel=f"{base}/cancel",
        export=f"{base}/export",
    )


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_run(resources: Resources, body: CreateRunRequest) -> CreateRunResponse:
    """Validate and start one run, returning immediately with its id.

    The run executes as a background task owned by this process. Poll
    ``links.status`` until it reports a terminal status.
    """
    run = await resources.manager.launch(
        LaunchRequest(
            suite=body.suite,
            provider=body.provider.provider,
            model=body.provider.model,
            api_key_env=body.provider.api_key_env,
            base_url=body.provider.base_url,
            provider_options=dict(body.provider.options),
            label=body.label,
            tags=body.tags,
            select_tags=body.select_tags,
            case_ids=body.case_ids,
            limit=body.limit,
            sample=body.sample,
            sample_seed=body.sample_seed,
            concurrency=body.concurrency,
            case_timeout_s=body.case_timeout_s,
            max_cost=body.max_cost,
            error_policy=body.error_policy,
            seed=body.seed,
            notes=body.notes,
        )
    )
    return CreateRunResponse(run_id=run.id, status=run.status, links=links_for(run.id))


@router.get("/runs")
async def list_runs(  # noqa: PLR0913, PLR0917 - one parameter per documented filter
    resources: Resources,
    run_status: Annotated[RunStatus | None, Query(alias="status")] = None,
    provider: Annotated[str | None, Query(max_length=200)] = None,
    model: Annotated[str | None, Query(max_length=200)] = None,
    suite: Annotated[str | None, Query(alias="suite", max_length=512)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    order: Annotated[Literal["created_at", "-created_at"], Query()] = "-created_at",
) -> Page[RunSummary]:
    """List runs newest first, with the total behind the page."""
    return await resources.runs.list_runs(
        RunQuery(
            status=run_status,
            provider=provider,
            model=model,
            suite_name=suite,
            limit=limit,
            offset=offset,
            order=order,
        )
    )


@router.get("/runs/{id}")
async def get_run(resources: Resources, id: str) -> RunDetail:  # noqa: A002 - the documented path parameter is {id}
    """Return one run with the pass rate computed from its persisted cases.

    ``run.config.suite_source`` is ``None``: the server's path to the suite file
    is not part of the answer to "what did this run do". The suite name, version
    and content hash beside it are.
    """
    view = await resources.runs.get_run(id)
    if view is None:
        raise RunNotFoundError(id)
    return RunDetail.of(view)


@router.get("/runs/{id}/status")
async def run_status(resources: Resources, id: str) -> RunStatusResponse:  # noqa: A002 - the documented path parameter is {id}
    """Report one run's progress. The endpoint a client polls after a ``202``."""
    progress = await resources.manager.status(id)
    if progress is None:
        raise RunNotFoundError(id)
    return RunStatusResponse(
        run_id=progress.run_id,
        status=progress.status,
        completed=progress.completed,
        total=progress.total,
        passed=progress.passed,
        errors=progress.errors,
        cancelled=progress.cancelled,
        started_at=progress.started_at,
        updated_at=progress.updated_at,
        error=progress.error,
        managed=progress.managed,
        last_case_id=progress.last_case_id,
    )


@router.get("/runs/{id}/cases")
async def list_cases(  # noqa: PLR0913, PLR0917 - one parameter per documented filter
    resources: Resources,
    id: str,  # noqa: A002 - the documented path parameter is {id}
    case_status: Annotated[CaseStatus | None, Query(alias="status")] = None,
    passed: Annotated[bool | None, Query()] = None,
    tag: Annotated[str | None, Query(max_length=200)] = None,
    category: Annotated[str | None, Query(max_length=200)] = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[CaseResultSummary]:
    """List one run's cases as rows, without their model outputs."""
    await _require_run(resources, id)
    page = await resources.runs.list_cases(
        id,
        CaseQuery(
            status=case_status,
            passed=passed,
            tag=tag,
            category=category,
            q=q,
            limit=limit,
            offset=offset,
        ),
    )
    return Page[CaseResultSummary](
        items=tuple(CaseResultSummary.of(item) for item in page.items),
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/runs/{id}/cases/{case_id}")
async def get_case(resources: Resources, id: str, case_id: str) -> CaseResult:  # noqa: A002 - the documented path parameter is {id}
    """Return one case in full: its response, its evaluations and their provenance."""
    result = await resources.runs.get_case(id, case_id)
    if result is None:
        await _require_run(resources, id)
        raise HTTPException(status_code=404, detail=f"run {id} has no case {case_id!r}")
    return result


@router.get("/runs/{id}/metrics")
async def run_metrics(resources: Resources, id: str, response: Response) -> AggregateMetrics:  # noqa: A002 - the documented path parameter is {id}
    """Return one run's aggregate rollup.

    See :data:`MATERIALIZED_HEADER` for how a stored rollup is distinguished from
    one computed for this request.
    """
    view = await resources.metrics.get(id)
    if view is None:
        raise RunNotFoundError(id)
    response.headers[MATERIALIZED_HEADER] = "true" if view.materialized else "false"
    return view.metrics


@router.post("/runs/{id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_run(resources: Resources, id: str) -> CancelResponse:  # noqa: A002 - the documented path parameter is {id}
    """Ask one run to stop after the cases already in flight finish.

    ``202``, not ``200``: the run is asked to stop, and stops when its in-flight
    cases have been written. A run that has already finished, or that this
    process is not executing, is a ``409`` - answering ``202`` for an instruction
    nothing will carry out is worse than an error.
    """
    outcome = await resources.manager.cancel(id)
    if outcome is CancelOutcome.NOT_FOUND:
        raise RunNotFoundError(id)
    if outcome is CancelOutcome.ALREADY_TERMINAL:
        raise HTTPException(status_code=409, detail=f"run {id} has already finished")
    if outcome is CancelOutcome.NOT_MANAGED:
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {id} is not being executed by this process, so it cannot be "
                f"cancelled here. It was launched by another process, which owns it."
            ),
        )
    progress = await resources.manager.status(id)
    if progress is None:
        raise RunNotFoundError(id)
    return CancelResponse(
        run_id=id,
        status=progress.status,
        outcome=outcome.value,
        detail="The run will stop once the cases already in flight have been written.",
    )


@router.get("/runs/{id}/export")
async def export_run(
    resources: Resources,
    id: str,  # noqa: A002 - the documented path parameter is {id}
    export_format: Annotated[Literal["json", "csv"], Query(alias="format")] = "json",
) -> StreamingResponse:
    """Send one run's case results as a JSON document or a CSV file.

    The CSV column order, the JSON row shape and the JSON envelope all come from
    shared code, so an export taken here is comparable with one taken through
    ``llm-eval export``.

    Chunked transfer, not incremental streaming: the case results are read in
    full and the document is rendered before the first byte goes out, so peak
    memory is the size of the whole export. Rendering lazily would need the
    repository's own async iterator to survive past its unit of work, which is a
    persistence-layer change rather than a routing one. The chunking still keeps
    the transport from holding a second copy.
    """
    view = await resources.metrics.get(id)
    if view is None:
        raise RunNotFoundError(id)
    results = await resources.metrics.cases(id)

    if export_format == "csv":
        return StreamingResponse(
            _chunks(_csv_text(results)),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": _attachment(id, "csv")},
        )
    document: dict[str, JSONValue] = {
        "schema_version": API_SCHEMA_VERSION,
        **export_document(id, results, view.metrics),
    }
    return StreamingResponse(
        _chunks(json.dumps(document, ensure_ascii=False)),
        media_type="application/json",
        headers={"Content-Disposition": _attachment(id, "json")},
    )


def _attachment(run_id: str, extension: str) -> str:
    """Build a `Content-Disposition` value, refusing to interpolate an odd id.

    An id that is not the contract's canonical uuid form is replaced by a fixed
    name rather than written into a response header. See :data:`RUN_ID_PATTERN`.
    """
    stem = run_id if RUN_ID_PATTERN.match(run_id) else "export"
    return f'attachment; filename="run-{stem}.{extension}"'


def _csv_text(results: tuple[CaseResult, ...]) -> str:
    """Render the case rows as CSV through the shared writer.

    ``write_case_csv`` rather than a second ``csv.DictWriter`` here, because that
    writer owns the column order AND the formula-injection guard that stops a
    benchmark author's category of ``=HYPERLINK(...)`` becoming a live formula in
    whoever opens the download. A separate writer would be a second, unguarded
    implementation of the same file format.
    """
    buffer = io.StringIO()
    write_case_csv(results, buffer)
    return buffer.getvalue()


async def _chunks(text: str) -> AsyncIterator[bytes]:
    """Hand an export to the transport in bounded pieces."""
    encoded = text.encode("utf-8")
    for start in range(0, len(encoded), STREAM_CHUNK_BYTES):
        yield encoded[start : start + STREAM_CHUNK_BYTES]


async def _require_run(resources: AppResources, run_id: str) -> None:
    """Raise 404 unless a run with this id exists.

    Raises:
        RunNotFoundError: when no run has that id.
    """
    if await resources.runs.fetch_run(run_id) is None:
        raise RunNotFoundError(run_id)


__all__ = ["MATERIALIZED_HEADER", "RUN_ID_PATTERN", "router"]
