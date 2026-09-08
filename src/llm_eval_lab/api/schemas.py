"""HTTP data-transfer objects: the shapes the wire carries, and nothing else.

These are deliberately separate from the frozen contract models. A domain model
answers "what is a run"; a DTO answers "what does this endpoint accept and
return", and the two drift apart the moment an endpoint needs a field the domain
does not have (``links``) or must withhold one it does (the absolute path of a
suite file on the server).

Where a contract model IS the right answer it is used directly rather than
copied - :class:`~llm_eval_lab.models.Run`,
:class:`~llm_eval_lab.models.ResolvedSuite`,
:class:`~llm_eval_lab.models.AggregateMetrics`,
:class:`~llm_eval_lab.models.RegressionReport` and
:class:`~llm_eval_lab.models.PriceTable` all cross the wire as themselves. A
hand-maintained duplicate of those would be a second definition of the same
facts, and the two would disagree within a release.

Every request model forbids extra fields and bounds every string and collection
it accepts. An unbounded ``notes`` on an endpoint that spends money is a way to
fill a disk.
"""

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from llm_eval_lab.models import (
    CaseResult,
    CaseStatus,
    ProviderConfig,
    RegressionThresholds,
    ResolvedSuite,
    Run,
    RunStatus,
)
from llm_eval_lab.services import RunView

API_SCHEMA_VERSION = 1
"""Version of the shapes in this module.

Distinct from the CLI envelope's own ``schema_version``: the two surfaces carry
the same FACTS but not the same documents, and one number covering both would
have to be bumped for changes the other never saw.
"""

MAX_LABEL_CHARS = 200
MAX_NOTES_CHARS = 4000
MAX_TAGS = 32
MAX_CASE_IDS = 1000
MAX_NAME_CHARS = 512

type ShortText = Annotated[str, Field(min_length=1, max_length=MAX_LABEL_CHARS)]
type SuiteName = Annotated[str, Field(min_length=1, max_length=MAX_NAME_CHARS)]


class ApiFieldError(BaseModel):
    """One problem found in a benchmark file, as the API reports it.

    Deliberately NOT :class:`~llm_eval_lab.models.FieldError`, and deliberately
    two fields shorter than it.

    ``file`` is dropped because it is the absolute path of the file on the
    server, and an error response is not the place to describe the server's file
    system to a client.

    ``input_repr`` is dropped because it is a repr of whatever the loader was
    looking at when validation failed, and that is benchmark-file content. When
    a required field is missing, pydantic reports the failure at the location of
    the FIELD and echoes the whole PARENT object as the input - so a case that
    omits ``id`` and carries a credential in its ``metadata`` echoed the
    credential back, in a body the API returned with a 200. Redacting by
    inspecting the location name could not see that, because the location named
    the missing field and not the secret beside it. Removing the member removes
    the class of defect, and matches what
    :func:`request_validation_errors` already does for request bodies. The
    location, the case id and the message say what is wrong; the file on the
    operator's disk says what it is wrong in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    location: str
    case_id: str | None
    message: str


class ProblemDetail(BaseModel):
    """An RFC 9457 problem detail, with this API's extension members.

    Every extension is optional and omitted when absent, so a client can read
    ``status`` and ``title`` from any error and reach for the rest only when it
    knows the ``type``.
    """

    model_config = ConfigDict(frozen=True)

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str | None = None
    errors: tuple[ApiFieldError, ...] | None = None
    """Field-level problems, for a benchmark or request validation failure."""
    env_var: str | None = None
    """The NAME of an unset credential environment variable. Never its value."""
    extra: str | None = None
    """The name of the optional install extra a provider needs."""
    correlation_id: str | None = None
    """The id an unmapped failure was logged under."""


class HealthResponse(BaseModel):
    """What ``GET /api/health`` reports.

    ``database`` is the result of an actual query, not of holding a connection
    object: a health check that never touches the database reports the process
    is alive, which is the one thing nobody was worried about.

    ``runs_in_flight`` is this process's own count: runs it is executing plus
    runs it has reserved a launch slot for, out of
    :data:`~llm_eval_lab.services.run_manager.MAX_CONCURRENT_RUNS`. A
    supervisor watching this number sees load before it sees a ``429``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok", "degraded"]
    version: str
    database: Literal["ok", "error"]
    runs_in_flight: int


class VersionResponse(BaseModel):
    """What ``GET /api/version`` reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    python: str
    schema_version: int


class ProviderEntry(BaseModel):
    """One row of ``GET /api/providers``.

    ``credential_env`` is an environment variable NAME and ``credential_resolved``
    says only whether that variable is set. There is no field on this object that
    could hold a credential value, which is what makes the endpoint safe to call
    from a browser.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    available: bool
    requires_credential: bool
    credential_env: str | None
    credential_resolved: bool | None
    extra: str | None
    summary: str
    models: tuple[str, ...]


class EvaluatorEntry(BaseModel):
    """One row of ``GET /api/evaluators``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    params_schema: dict[str, Any]
    needs_expected: bool
    is_model_graded: bool
    summary: str


class BenchmarkEntry(BaseModel):
    """One row of ``GET /api/benchmarks``.

    A suite that does not load is listed with ``valid: false`` and the number of
    problems in it, rather than omitted. A hidden broken file looks exactly like
    a missing one, and sends the operator looking for the wrong fault.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    valid: bool
    suite_name: str | None
    version: str | None
    suite_hash: str | None
    n_cases: int | None
    evaluator_types: tuple[str, ...]
    n_errors: int


class ValidateRequest(BaseModel):
    """Body of ``POST /api/benchmarks/validate``.

    A NAME, never a path. The API resolves names under the configured benchmark
    root and accepts nothing that escapes it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SuiteName


class ValidateResponse(BaseModel):
    """What ``POST /api/benchmarks/validate`` returns.

    A ``200`` with ``valid: false`` rather than a ``422``: the client ASKED
    whether the file is valid, and "no" is a successful answer to that question.
    The ``422`` problem detail is what a client gets when it tries to RUN an
    invalid suite.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    valid: bool
    errors: tuple[ApiFieldError, ...]


class BenchmarkDetail(BaseModel):
    """What ``GET /api/benchmarks/{name}`` returns: the resolved suite and its digest.

    The resolved form, because that is what a run actually executes and what
    ``suite_hash`` addresses. The absolute path of the file is deliberately
    absent: the client addressed it by name and does not need the server's
    directory layout to address it again.

    Secret-shaped values in the two free-form dictionaries a benchmark author
    controls are replaced before the suite is returned, by the same scrubber the
    storage layer uses. ``redacted`` says whether anything was replaced, so
    ``suite_hash`` - which addresses the ORIGINAL file - is never mistaken for a
    digest of the body beside it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    suite_hash: str
    redacted: bool
    suite: ResolvedSuite


class RunLinks(BaseModel):
    """Where to go next after launching a run, so a client hard-codes no paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    self: str
    status: str
    cases: str
    metrics: str
    cancel: str
    export: str


class CreateRunRequest(BaseModel):
    """Body of ``POST /api/runs``.

    ``provider`` is the frozen :class:`~llm_eval_lab.models.ProviderConfig`,
    used here rather than re-declared, because its validators are the security
    control: it carries ``api_key_env`` (a NAME) and its recursive validator
    refuses any option key that looks like a place somebody put a credential.
    Re-declaring the shape here would be re-implementing that check, badly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite: SuiteName
    provider: ProviderConfig
    label: ShortText | None = None
    tags: Annotated[tuple[ShortText, ...], Field(max_length=MAX_TAGS)] = ()
    select_tags: Annotated[tuple[ShortText, ...], Field(max_length=MAX_TAGS)] = ()
    case_ids: Annotated[tuple[ShortText, ...], Field(max_length=MAX_CASE_IDS)] = ()
    limit: int | None = Field(default=None, ge=1)
    sample: int | None = Field(default=None, ge=1)
    sample_seed: int | None = None
    concurrency: int | None = Field(default=None, ge=1, le=256)
    case_timeout_s: float | None = Field(default=None, gt=0.0, le=3600.0)
    max_cost: Decimal | None = Field(
        default=None,
        ge=0,
        description=(
            "Running spend cap for THIS run, in the price table's currency; the runner "
            "cancels the run once accumulated known cost exceeds it. This per-run "
            "ceiling is the only server-side spend limit: the deployment imposes no "
            "cap of its own, so omitting this launches a run with no budget. The "
            "process will not execute more than a fixed number of runs at once, which "
            "bounds concurrent spend but not the spend of any single run."
        ),
    )
    error_policy: Literal["exclude", "fail"] = "exclude"
    seed: int | None = None
    notes: Annotated[str, Field(max_length=MAX_NOTES_CHARS)] | None = None


class CreateRunResponse(BaseModel):
    """The ``202`` answer to ``POST /api/runs``: an id and where to poll it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    status: RunStatus
    links: RunLinks


class RunStatusResponse(BaseModel):
    """What ``GET /api/runs/{id}/status`` reports, for a polling client.

    Read from the persisted run row, so it is equally correct for a run this
    process launched, a run the CLI launched and a run a previous process left
    behind. ``managed`` says whether this process holds the task, which is what
    decides whether cancelling can work.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    status: RunStatus
    completed: int
    total: int
    passed: int
    errors: int
    cancelled: int
    started_at: AwareDatetime | None
    updated_at: AwareDatetime
    error: str | None
    managed: bool
    last_case_id: str | None = Field(
        default=None,
        description=(
            "The case named by the most recent progress event, for a run this process "
            "is executing. A display aid, not a count: with concurrency above one "
            "several cases are in flight and this names the one that most recently "
            "changed state. Null whenever managed is false."
        ),
    )


def redacted_run(run: Run) -> Run:
    """Return `run` with the server's path to its suite file removed.

    ``RunConfig.suite_source`` is the absolute path the run was launched from. It
    belongs in the database, where it makes a historical run traceable to the
    file it came from, and it belongs in the CLI, which is already standing in
    that file system. It does not belong in an HTTP response: a client addressed
    the suite by NAME, and the reply should not describe the server's directory
    layout back to it.

    Nothing is lost by removing it. ``suite_name``, ``suite_version`` and
    ``suite_hash`` all survive on the same object, and the hash is the identifier
    that actually addresses the suite's content. The stored row is untouched;
    this is a projection at the boundary, not a redaction of the record.

    Deliberately ``None`` rather than the name relative to the configured root: a
    run launched by the CLI can name a file anywhere on disk, so a "relative"
    form would be derivable for some runs and not others, and a field that is
    sometimes a name and sometimes absent is worse than one that is always
    absent.
    """
    return run.model_copy(update={"config": run.config.model_copy(update={"suite_source": None})})


class RunDetail(BaseModel):
    """What ``GET /api/runs/{id}`` returns: the run plus its derived figures.

    ``pass_rate`` is computed from the persisted case results under the run's own
    ``error_policy``, and ``n_pass_denominator`` says what it was computed over,
    so a rate is never shown without the population behind it.

    ``run.config.suite_source`` is always ``None`` on this response. It is the
    server's own path to the suite file, which the database keeps and the API
    withholds; ``suite_name``, ``suite_version`` and ``suite_hash`` beside it say
    which suite ran.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: Run
    n_cases: int
    n_completed: int
    n_errors: int
    n_passed: int
    n_pass_denominator: int
    pass_rate: float | None

    @classmethod
    def of(cls, view: RunView) -> "RunDetail":
        """Project one run view into its HTTP shape, without the server's path."""
        return cls(
            run=redacted_run(view.run),
            n_cases=view.n_cases,
            n_completed=view.n_completed,
            n_errors=view.n_errors,
            n_passed=view.n_passed,
            n_pass_denominator=view.n_pass_denominator,
            pass_rate=view.pass_rate,
        )


class CaseResultSummary(BaseModel):
    """One row of ``GET /api/runs/{id}/cases``.

    The model's OUTPUT is deliberately absent. A page of fifty cases is a table,
    and whole responses belong behind the per-case detail endpoint, which returns
    the full :class:`~llm_eval_lab.models.CaseResult` with its evaluations and
    their judge provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    case_id: str
    case_hash: str
    status: CaseStatus
    passed: bool | None
    score: float | None
    attempts: int
    category: str | None
    tags: tuple[str, ...]
    latency_ms: float | None
    total_tokens: int | None
    total_cost: Decimal | None
    priced: bool | None
    error_kind: str | None
    started_at: AwareDatetime | None
    completed_at: AwareDatetime | None

    @classmethod
    def of(cls, result: CaseResult) -> "CaseResultSummary":
        """Project one persisted case result into its listing row."""
        response = result.response
        usage = response.usage if response is not None else None
        cost = result.cost
        return cls(
            run_id=result.run_id,
            case_id=result.case_id,
            case_hash=result.case_hash,
            status=result.status,
            passed=result.passed,
            score=result.score,
            attempts=result.attempts,
            category=result.category,
            tags=result.tags,
            latency_ms=None if response is None else response.latency_ms,
            total_tokens=None if usage is None else usage.total_tokens,
            total_cost=None if cost is None else cost.total_cost,
            priced=None if cost is None else cost.priced,
            error_kind=None if result.error is None else result.error.kind.value,
            started_at=result.started_at,
            completed_at=result.completed_at,
        )


class CancelResponse(BaseModel):
    """The ``202`` answer to ``POST /api/runs/{id}/cancel``.

    ``202`` rather than ``200`` because cancellation is cooperative: the run is
    asked to stop and finishes the cases already in flight. ``outcome`` says what
    the request actually achieved, and ``status`` is the run's status at the
    moment the request was handled, not the status it will end at.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    status: RunStatus
    outcome: str
    detail: str


class CompareRequest(BaseModel):
    """Body of ``POST /api/compare``.

    ``thresholds`` is an inline policy, not a path: the API accepts no
    server-side file paths. Omitting it uses the policy shipped with the package,
    which is the same default ``llm-eval compare`` uses.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_run_id: ShortText
    candidate_run_id: ShortText
    thresholds: RegressionThresholds | None = None


class ModelSummaryRow(BaseModel):
    """One row of ``GET /api/models/summary``.

    Every field says what it was computed over, because a leaderboard that shows
    a pass rate without a denominator invites a comparison the data does not
    support.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    n_runs: int
    n_cases: int
    n_passed: int
    n_pass_denominator: int
    pass_rate: float | None = Field(
        default=None,
        description="Pooled over every stored run of this model, not an average of run rates.",
    )
    pass_rate_ci: tuple[float, float] | None = Field(
        default=None,
        description="Wilson score interval at 95% over the pooled numerator and denominator.",
    )
    median_run_p95_ms: float | None = Field(
        default=None,
        description=(
            "Median of the per-run p95 latencies. NOT a pooled percentile over cases: "
            "per-case latencies are not loaded by this endpoint, so this is a weaker "
            "statement than a p95 and is named for what it is. Read it with "
            "n_runs_with_latency."
        ),
    )
    n_runs_with_latency: int
    total_cost: Decimal | None
    cost_per_case: Decimal | None = Field(
        default=None,
        description=(
            "Summed cost divided by the summed COMPLETED cases, the same definition a "
            "single run's metrics use, so cost_per_case * n_cases reconciles against "
            "total_cost. Cases the price table could not cover contribute to the "
            "denominator and not to the numerator, so this figure is a lower bound "
            "whenever cost_coverage is below 1: read the two together."
        ),
    )
    cost_coverage: float | None = Field(
        default=None,
        description="Case-weighted fraction of cases that carried a price, across these runs.",
    )
    last_run_at: AwareDatetime | None


__all__ = [
    "API_SCHEMA_VERSION",
    "ApiFieldError",
    "BenchmarkDetail",
    "BenchmarkEntry",
    "CancelResponse",
    "CaseResultSummary",
    "CompareRequest",
    "CreateRunRequest",
    "CreateRunResponse",
    "EvaluatorEntry",
    "HealthResponse",
    "ModelSummaryRow",
    "ProblemDetail",
    "ProviderEntry",
    "RunDetail",
    "RunLinks",
    "RunStatusResponse",
    "ValidateRequest",
    "ValidateResponse",
    "VersionResponse",
    "redacted_run",
]
