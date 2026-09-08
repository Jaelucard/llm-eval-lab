"""Translation between database rows and the frozen domain models.

This module is the boundary. Every function here takes a row and returns a
domain object, or takes a domain object and returns column values; no ORM
instance is ever handed upward and no domain object is ever handed to a query.

Three obligations it discharges.

**Timezones.** SQLite returns naive datetimes no matter what was written, while
the contract's ``UtcDatetime`` refuses naive values. Every read re-attaches UTC
through :func:`~llm_eval_lab.utils.time.ensure_utc`, which is what makes the
same code produce the same values on SQLite and PostgreSQL.

**Secret scrubbing.** ``GenerationParams.extra``, ``EvaluatorSpec.params`` and a
provider's ``raw`` payload are free-form dictionaries that a user or a vendor
fills in. They are scrubbed on the way INTO the database, here, rather than by
a model validator - a validator cannot tell ``api_key_env`` (a legitimate
contract field naming a variable) from ``api_key`` (a credential), and would
have to reject both or neither.

**Payload size.** A vendor's ``raw`` payload is capped before storage. An
oversized payload is replaced by a marker recording its size, so the row stays
bounded and the truncation is visible rather than silent.
"""

from typing import Any

from llm_eval_lab.models import (
    AggregateMetrics,
    BenchmarkSnapshot,
    CaseResult,
    CaseStatus,
    CostBreakdown,
    EvaluationResult,
    ModelResponse,
    ProviderErrorInfo,
    Run,
    RunConfig,
    RunStatus,
    RunTotals,
)
from llm_eval_lab.redaction import scrub_mapping
from llm_eval_lab.storage.orm import (
    BenchmarkSnapshotRow,
    CaseResultRow,
    EvaluationRow,
    ModelResponseRow,
    RunMetricsRow,
    RunRow,
)
from llm_eval_lab.utils.json import canonical_json_bytes
from llm_eval_lab.utils.time import ensure_utc, ensure_utc_optional

DEFAULT_MAX_RAW_BYTES = 65536
"""Fallback cap on a stored vendor payload. Matches `settings.max_raw_bytes`."""

RAW_TRUNCATED_KEY = "_truncated"
"""Marker key present on a `raw` payload that was too large to store."""


def _json(value: Any) -> dict[str, Any]:
    """Return a scrubbed, JSON-safe dict for a free-form pydantic model dump."""
    return scrub_mapping(value)


def scrub_run_config(dump: dict[str, Any]) -> dict[str, Any]:
    """Scrub the free-form dictionaries inside a `RunConfig` dump, and only those.

    Precisely three places in a run configuration hold keys a USER chose rather
    than the contract: ``params.extra``, each ``evaluators[i].params``, and
    ``provider.options``. Those are scrubbed.

    The rest of the document is left alone deliberately. Scrubbing the whole
    config was the obvious first implementation and it was wrong: the contract's
    own field names include token COUNTS, and replacing an integer with a
    redaction marker makes the row fail validation the next time it is read. A
    security control that corrupts the data it protects is not a security
    control.
    """
    scrubbed = dict(dump)
    params = scrubbed.get("params")
    if isinstance(params, dict) and isinstance(params.get("extra"), dict):
        scrubbed["params"] = {**params, "extra": scrub_mapping(params["extra"])}
    provider = scrubbed.get("provider")
    if isinstance(provider, dict) and isinstance(provider.get("options"), dict):
        scrubbed["provider"] = {**provider, "options": scrub_mapping(provider["options"])}
    evaluators = scrubbed.get("evaluators")
    if isinstance(evaluators, list):
        scrubbed["evaluators"] = [
            {**spec, "params": scrub_mapping(spec["params"])}
            if isinstance(spec, dict) and isinstance(spec.get("params"), dict)
            else spec
            for spec in evaluators
        ]
    return scrubbed


def scrub_evaluation_payload(dump: dict[str, Any]) -> dict[str, Any]:
    """Scrub the free-form `metadata` of an evaluation result, and only that.

    Same reasoning as :func:`scrub_run_config`: `metadata` is the only part an
    evaluator fills with keys nobody declared, and the surrounding fields are
    contract-shaped and must survive a round trip intact.
    """
    metadata = dump.get("metadata")
    if isinstance(metadata, dict):
        return {**dump, "metadata": scrub_mapping(metadata)}
    return dump


def cap_raw_payload(raw: dict[str, Any], *, max_bytes: int) -> dict[str, Any]:
    """Return `raw` unchanged, or a marker recording that it was too large."""
    try:
        size = len(canonical_json_bytes(raw))
    except (TypeError, ValueError):
        return {RAW_TRUNCATED_KEY: True, "reason": "payload is not JSON-serializable"}
    if size <= max_bytes:
        return raw
    return {RAW_TRUNCATED_KEY: True, "reason": "payload exceeded max_raw_bytes", "bytes": size}


# ---------------------------------------------------------------------------
# benchmark snapshots
# ---------------------------------------------------------------------------


def snapshot_to_row(snapshot: BenchmarkSnapshot) -> BenchmarkSnapshotRow:
    """Build the row for a benchmark snapshot, scrubbing the stored body.

    The body is the whole resolved suite, which carries the SAME free-form
    dictionaries that :func:`scrub_run_config` scrubs on the way into
    ``runs.config`` - every case's ``params.extra`` and each
    ``evaluators[i].params``. Without this, one fact would have two write paths
    and only one of them would apply the security control, so a credential
    pasted into a benchmark file would be redacted in ``runs.config`` and
    persisted verbatim two tables away.

    The scrub is applied to the whole body rather than to named paths. The body
    is authored data end to end, its shape will grow (D-SIM puts a
    ``ProviderConfig`` with its own ``options`` inside an evaluator's params),
    and a rule that has to be updated whenever the shape changes is a rule that
    will be out of date.

    One consequence, stated because it is not obvious: a snapshot whose body was
    scrubbed can no longer be used to RECOMPUTE ``suite_hash``, because the hash
    covers the unredacted ``params``. The hash is still recorded, still
    comparable, and still what pairs runs together; the body remains what makes
    a run interpretable. Reproducing a redacted value means putting the
    credential back in the environment it should have come from.
    """
    return BenchmarkSnapshotRow(
        suite_hash=snapshot.suite_hash,
        name=snapshot.name,
        version=snapshot.version,
        n_cases=snapshot.n_cases,
        created_at=snapshot.created_at,
        source_path=snapshot.source_path,
        body=scrub_mapping(dict(snapshot.body)),
    )


def row_to_snapshot(row: BenchmarkSnapshotRow) -> BenchmarkSnapshot:
    """Rebuild a benchmark snapshot from its row."""
    return BenchmarkSnapshot(
        suite_hash=row.suite_hash,
        name=row.name,
        version=row.version,
        n_cases=row.n_cases,
        created_at=ensure_utc(row.created_at),
        source_path=row.source_path,
        body=dict(row.body),
    )


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def run_to_row(run: Run) -> RunRow:
    """Build the row for a run, denormalizing the fields queries filter on."""
    return RunRow(
        id=run.id,
        label=run.label,
        status=run.status.value,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        updated_at=run.updated_at,
        suite_name=run.config.suite_name,
        suite_version=run.config.suite_version,
        suite_hash=run.config.suite_hash,
        provider=run.config.provider.provider,
        model=run.config.provider.model,
        config=scrub_run_config(run.config.model_dump(mode="json")),
        n_cases=run.totals.n_cases,
        n_completed=run.totals.n_completed,
        n_passed=run.totals.n_passed,
        n_errors=run.totals.n_errors,
        n_cancelled=run.totals.n_cancelled,
        baseline_run_id=run.baseline_run_id,
        error=run.error,
        warnings=list(run.warnings),
        tags=list(run.tags),
    )


def row_to_run(row: RunRow) -> Run:
    """Rebuild a run from its row."""
    return Run(
        id=row.id,
        label=row.label,
        status=RunStatus(row.status),
        created_at=ensure_utc(row.created_at),
        started_at=ensure_utc_optional(row.started_at),
        completed_at=ensure_utc_optional(row.completed_at),
        config=RunConfig.model_validate(row.config),
        totals=RunTotals(
            n_cases=row.n_cases,
            n_completed=row.n_completed,
            n_passed=row.n_passed,
            n_errors=row.n_errors,
            n_cancelled=row.n_cancelled,
        ),
        updated_at=ensure_utc(row.updated_at),
        baseline_run_id=row.baseline_run_id,
        error=row.error,
        warnings=tuple(row.warnings),
        tags=tuple(row.tags),
    )


# ---------------------------------------------------------------------------
# model responses
# ---------------------------------------------------------------------------


def response_to_row(
    run_id: str,
    case_id: str,
    response: ModelResponse,
    *,
    max_raw_bytes: int = DEFAULT_MAX_RAW_BYTES,
) -> ModelResponseRow:
    """Build the row for one model response."""
    return ModelResponseRow(
        run_id=run_id,
        case_id=case_id,
        output_text=response.output_text,
        provider=response.provider,
        model=response.model,
        requested_model=response.requested_model,
        finish_reason=response.finish_reason.value,
        usage=response.usage.model_dump(mode="json"),
        latency_ms=response.latency_ms,
        total_latency_ms=response.total_latency_ms,
        started_at=response.started_at,
        completed_at=response.completed_at,
        attempts=response.attempts,
        truncated=response.truncated,
        error=None if response.error is None else response.error.model_dump(mode="json"),
        provider_request_id=response.provider_request_id,
        raw=cap_raw_payload(_json(response.raw), max_bytes=max_raw_bytes),
    )


def row_to_response(row: ModelResponseRow) -> ModelResponse:
    """Rebuild a model response from its row."""
    return ModelResponse.model_validate(
        {
            "output_text": row.output_text,
            "provider": row.provider,
            "model": row.model,
            "requested_model": row.requested_model,
            "finish_reason": row.finish_reason,
            "usage": row.usage,
            "latency_ms": row.latency_ms,
            "total_latency_ms": row.total_latency_ms,
            "started_at": ensure_utc(row.started_at),
            "completed_at": ensure_utc(row.completed_at),
            "attempts": row.attempts,
            "truncated": row.truncated,
            "error": row.error,
            "provider_request_id": row.provider_request_id,
            "raw": row.raw,
        }
    )


# ---------------------------------------------------------------------------
# case results
# ---------------------------------------------------------------------------


def case_result_to_row(result: CaseResult, *, row_id: str) -> CaseResultRow:
    """Build the row for one case result.

    The response and the evaluations are NOT written here: they own their own
    tables, and `model_responses` keyed by (run_id, case_id) is the single write
    path for a response.
    """
    return CaseResultRow(
        id=row_id,
        run_id=result.run_id,
        case_id=result.case_id,
        case_hash=result.case_hash,
        status=result.status.value,
        passed=result.passed,
        score=result.score,
        cost=None if result.cost is None else result.cost.model_dump(mode="json"),
        attempts=result.attempts,
        started_at=result.started_at,
        completed_at=result.completed_at,
        error=None if result.error is None else result.error.model_dump(mode="json"),
        category=result.category,
        tags=list(result.tags),
    )


def row_to_case_result(
    row: CaseResultRow,
    *,
    response: ModelResponse | None,
    evaluations: tuple[EvaluationResult, ...],
) -> CaseResult:
    """Rebuild a case result, hydrating its response and evaluations from their own rows."""
    return CaseResult(
        run_id=row.run_id,
        case_id=row.case_id,
        case_hash=row.case_hash,
        status=CaseStatus(row.status),
        response=response,
        evaluations=evaluations,
        passed=row.passed,
        score=row.score,
        cost=None if row.cost is None else CostBreakdown.model_validate(row.cost),
        attempts=row.attempts,
        started_at=ensure_utc_optional(row.started_at),
        completed_at=ensure_utc_optional(row.completed_at),
        error=None if row.error is None else ProviderErrorInfo.model_validate(row.error),
        category=row.category,
        tags=tuple(row.tags),
    )


# ---------------------------------------------------------------------------
# evaluations
# ---------------------------------------------------------------------------


def evaluation_to_row(
    run_id: str,
    case_id: str,
    result: EvaluationResult,
    *,
    row_id: str,
) -> EvaluationRow:
    """Build the row for one evaluation result."""
    return EvaluationRow(
        id=row_id,
        run_id=run_id,
        case_id=case_id,
        evaluator_id=result.evaluator_id,
        evaluator_type=result.evaluator_type,
        status=result.status.value,
        passed=result.passed,
        score=result.score,
        payload=scrub_evaluation_payload(result.model_dump(mode="json")),
    )


def row_to_evaluation(row: EvaluationRow) -> EvaluationResult:
    """Rebuild an evaluation result from its row."""
    return EvaluationResult.model_validate(row.payload)


# ---------------------------------------------------------------------------
# aggregate metrics
# ---------------------------------------------------------------------------


def metrics_to_row(metrics: AggregateMetrics) -> RunMetricsRow:
    """Build the single metrics row for one run."""
    ci = metrics.pass_rate_ci
    return RunMetricsRow(
        run_id=metrics.run_id,
        computed_at=metrics.computed_at,
        price_table_id=metrics.cost.price_table_id,
        price_table_version=metrics.cost.price_table_version,
        error_policy=metrics.error_policy,
        n_cases=metrics.n_cases,
        n_completed=metrics.n_completed,
        n_errors=metrics.n_errors,
        n_timeouts=metrics.n_timeouts,
        n_skipped=metrics.n_skipped,
        n_scored=metrics.n_scored,
        n_score_only=metrics.n_score_only,
        n_passed=metrics.n_passed,
        n_pass_denominator=metrics.n_pass_denominator,
        error_rate=metrics.error_rate,
        pass_rate=metrics.pass_rate,
        pass_rate_ci_low=None if ci is None else ci[0],
        pass_rate_ci_high=None if ci is None else ci[1],
        mean_score=metrics.mean_score,
        median_score=metrics.median_score,
        token_usage_coverage=metrics.token_usage_coverage,
        cost_coverage=metrics.cost_coverage,
        cost_per_case=metrics.cost_per_case,
        cost_per_successful_evaluation=metrics.cost_per_successful_evaluation,
        latency=metrics.latency.model_dump(mode="json"),
        tokens=metrics.tokens.model_dump(mode="json"),
        cost=metrics.cost.model_dump(mode="json"),
        error_breakdown=dict(metrics.error_breakdown),
        by_category={
            key: value.model_dump(mode="json") for key, value in metrics.by_category.items()
        },
        by_tag={key: value.model_dump(mode="json") for key, value in metrics.by_tag.items()},
        by_evaluator={
            key: value.model_dump(mode="json") for key, value in metrics.by_evaluator.items()
        },
    )


def row_to_metrics(row: RunMetricsRow) -> AggregateMetrics:
    """Rebuild the aggregate metrics for one run from its row."""
    ci = (
        None
        if row.pass_rate_ci_low is None or row.pass_rate_ci_high is None
        else (row.pass_rate_ci_low, row.pass_rate_ci_high)
    )
    return AggregateMetrics.model_validate(
        {
            "run_id": row.run_id,
            "computed_at": ensure_utc(row.computed_at),
            "n_cases": row.n_cases,
            "n_completed": row.n_completed,
            "n_errors": row.n_errors,
            "n_timeouts": row.n_timeouts,
            "n_skipped": row.n_skipped,
            "error_rate": row.error_rate,
            "error_breakdown": dict(row.error_breakdown),
            "error_policy": row.error_policy,
            "pass_rate": row.pass_rate,
            "pass_rate_ci": ci,
            "n_passed": row.n_passed,
            "n_pass_denominator": row.n_pass_denominator,
            "n_scored": row.n_scored,
            "mean_score": row.mean_score,
            "median_score": row.median_score,
            "n_score_only": row.n_score_only,
            "latency": row.latency,
            "tokens": row.tokens,
            "token_usage_coverage": row.token_usage_coverage,
            "cost": row.cost,
            "cost_coverage": row.cost_coverage,
            "cost_per_case": row.cost_per_case,
            "cost_per_successful_evaluation": row.cost_per_successful_evaluation,
            "by_category": row.by_category,
            "by_tag": row.by_tag,
            "by_evaluator": row.by_evaluator,
        }
    )
