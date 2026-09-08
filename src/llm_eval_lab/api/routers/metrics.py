"""The cross-run leaderboard: one row per model, pooled over its stored runs.

A leaderboard invites a comparison, so every row says what it was computed over.
The pass rate is pooled from the numerators and denominators of the individual
runs rather than averaged across them, so a two-case run does not weigh as much
as a two-hundred-case one, and the Wilson interval beside it is over that pooled
denominator.

The latency figure is NAMED for what it is, ``median_run_p95_ms``: the median of
the per-run p95 values, because per-case latencies are not loaded here. A field
called ``p95_ms`` would be read as a pooled percentile by every dashboard author
who never opened its description, and the name is the part that reaches them.
"""

from fastapi import APIRouter

from llm_eval_lab.api.deps import Resources
from llm_eval_lab.api.schemas import ModelSummaryRow

router = APIRouter(prefix="/api", tags=["metrics"])


@router.get("/models/summary")
async def model_summary(resources: Resources) -> list[ModelSummaryRow]:
    """Pool every stored run rollup into one row per provider and model."""
    return [
        ModelSummaryRow(
            provider=row.provider,
            model=row.model,
            n_runs=row.n_runs,
            n_cases=row.n_cases,
            n_passed=row.n_passed,
            n_pass_denominator=row.n_pass_denominator,
            pass_rate=row.pass_rate,
            pass_rate_ci=row.pass_rate_ci,
            median_run_p95_ms=row.median_run_p95_ms,
            n_runs_with_latency=row.n_runs_with_latency,
            total_cost=row.total_cost,
            cost_per_case=row.cost_per_case,
            cost_coverage=row.cost_coverage,
            last_run_at=row.last_run_at,
        )
        for row in await resources.metrics.model_summaries()
    ]


__all__ = ["router"]
