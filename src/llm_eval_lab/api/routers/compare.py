"""Baseline versus candidate, over HTTP, with the payload the CLI already emits.

The response body is `RegressionReport.model_dump_json()` - the exact bytes
``llm-eval compare --json`` writes to stdout. Not an equivalent document: the
same one, produced by the same encoder. A dashboard and a CI pipeline reading
the same comparison must not be looking at two spellings of it, and the only way
to guarantee that is to have one serializer rather than two that agree today.

Threshold policies arrive INLINE, as a
:class:`~llm_eval_lab.models.RegressionThresholds` object, never as a path. The
API accepts no server-side file paths, and a policy file is a file like any
other. Omitting the policy uses the one shipped with the package, which is the
same default the CLI uses.
"""

from typing import Annotated

from fastapi import APIRouter, Query, Response

from llm_eval_lab.api.deps import AppResources, Resources
from llm_eval_lab.api.schemas import CompareRequest
from llm_eval_lab.models import RegressionReport, RegressionThresholds

router = APIRouter(prefix="/api", tags=["compare"])

REPORT_MEDIA_TYPE = "application/json"
"""The report is ordinary JSON; it carries its own `schema_version` inside."""


async def _compare(
    resources: AppResources,
    baseline_run_id: str,
    candidate_run_id: str,
    thresholds: RegressionThresholds | None,
) -> Response:
    """Run one comparison and render the report as the CLI renders it.

    Raises:
        RunNotFoundError: when either run id is unknown.
        RegressionInputError: when the two runs cannot be compared.
        EvaluatorConfigError: when the policy names an unaddressable metric.
    """
    service = resources.comparisons
    view = await service.compare(
        baseline_run_id,
        candidate_run_id,
        thresholds=thresholds if thresholds is not None else service.thresholds(),
    )
    return Response(content=view.report.model_dump_json(), media_type=REPORT_MEDIA_TYPE)


@router.post("/compare", response_model=RegressionReport)
async def compare(resources: Resources, body: CompareRequest) -> Response:
    """Compare a candidate run against a baseline under an inline or default policy."""
    return await _compare(resources, body.baseline_run_id, body.candidate_run_id, body.thresholds)


@router.get("/compare", response_model=RegressionReport)
async def compare_by_query(
    resources: Resources,
    baseline_run_id: Annotated[str, Query(min_length=1, max_length=200)],
    candidate_run_id: Annotated[str, Query(min_length=1, max_length=200)],
) -> Response:
    """Compare two runs under the shipped default policy.

    The linkable form, for a dashboard that wants a comparison in a URL. A custom
    policy needs a body, so it needs the ``POST``.
    """
    return await _compare(resources, baseline_run_id, candidate_run_id, None)


__all__ = ["router"]
