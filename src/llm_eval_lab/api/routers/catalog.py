"""Read-only catalog: providers, evaluators, prices and benchmark suites.

Every route here answers a question the dashboard asks before it can offer a
choice: which providers can this build reach, which evaluator types exist and
what parameters do they take, what does a token cost, and what suites are
installed.

**A suite is addressed by NAME, never by path.** The client sends a name, the
catalog service resolves it under the configured benchmark root, and anything
that resolves outside the root - through ``..``, through an absolute path, or
through a symlink - is refused before the file system is read. Nothing on this
router accepts a path from a client.
"""

from fastapi import APIRouter, Response

from llm_eval_lab.api.deps import Resources
from llm_eval_lab.api.errors import field_errors_of
from llm_eval_lab.api.schemas import (
    BenchmarkDetail,
    BenchmarkEntry,
    EvaluatorEntry,
    ProviderEntry,
    ValidateRequest,
    ValidateResponse,
)
from llm_eval_lab.models import BenchmarkValidationError, Page, PriceTable
from llm_eval_lab.services import MAX_LISTED_SUITES

router = APIRouter(prefix="/api", tags=["catalog"])

LISTING_TRUNCATED_HEADER = "X-Listing-Truncated"
"""Says whether the benchmark root holds more suites than this listing returned.

`SuiteListing.truncated` is a fact about the LISTING, not about any one
suite, so it travels as a header rather than as a new member of the `Page`
body: the `Page[BenchmarkEntry]` contract stays exactly what every other
paged endpoint returns, and a client that only wants "is this everything"
reads one header instead of parsing the page to find out.
"""


@router.get("/providers")
async def providers(resources: Resources) -> list[ProviderEntry]:
    """List every registered provider, its required extra and its credential VARIABLE.

    ``credential_env`` is a variable name and ``credential_resolved`` is a
    boolean. No field on this response can hold a credential value, which is what
    makes the endpoint safe to render in a browser.
    """
    catalog = resources.catalog
    entries: list[ProviderEntry] = []
    for info in catalog.providers():
        resolved = catalog.credential_status(info.name)
        entries.append(
            ProviderEntry(
                name=info.name,
                # BOTH facts, not credential status alone: a provider whose
                # optional extra is not installed is not available no matter
                # what the environment holds.
                available=info.available and resolved is not False,
                requires_credential=info.requires_credential,
                credential_env=info.credential_env,
                credential_resolved=resolved,
                extra=info.extra,
                summary=info.summary,
                models=info.models,
            )
        )
    return entries


@router.get("/evaluators")
async def evaluators(resources: Resources) -> list[EvaluatorEntry]:
    """List every registered evaluator type with the JSON Schema of its parameters."""
    return [
        EvaluatorEntry(
            type=info.type,
            params_schema=info.params_schema(),
            needs_expected=info.needs_expected,
            is_model_graded=info.is_model_graded,
            summary=info.summary,
        )
        for info in resources.catalog.evaluators()
    ]


@router.get("/pricing")
async def pricing(resources: Resources) -> PriceTable:
    """Return the price table this process costs runs against.

    The ``content_hash`` travels with it deliberately: it is what lets somebody
    confirm a historical run's costs came from this exact table rather than from
    a later edit that kept the same version string.
    """
    return resources.catalog.price_table()


@router.get("/benchmarks")
async def benchmarks(resources: Resources, response: Response) -> Page[BenchmarkEntry]:
    """List the suites under the configured benchmark root.

    A page rather than a bare array, using the same
    :class:`~llm_eval_lab.models.Page` shape the runs and cases listings use. A
    root holding more suites than one listing returns is truncated
    deterministically to the lexicographically first of them, and ``total``
    greater than the number of items is how a client learns that happened - a
    silently short array would read as an inventory. The same fact is also
    sent as :data:`LISTING_TRUNCATED_HEADER`, so a caller that wants a plain
    yes/no need not compare ``total`` against ``len(items)`` itself.

    Returns an empty page when no root is configured, rather than an error: "no
    suites are addressable by name here" is a true and useful answer, and the
    attempt to address one by name reports the missing configuration precisely.
    """
    summaries, listing = resources.catalog.suite_summaries()
    response.headers[LISTING_TRUNCATED_HEADER] = "true" if listing.truncated else "false"
    return Page[BenchmarkEntry](
        items=tuple(
            BenchmarkEntry(
                name=summary.name,
                valid=summary.valid,
                suite_name=summary.suite_name,
                version=summary.version,
                suite_hash=summary.suite_hash,
                n_cases=summary.n_cases,
                evaluator_types=summary.evaluator_types,
                n_errors=summary.n_errors,
            )
            for summary in summaries
        ),
        total=listing.n_found,
        limit=MAX_LISTED_SUITES,
        offset=0,
    )


@router.post("/benchmarks/validate")
async def validate(resources: Resources, body: ValidateRequest) -> ValidateResponse:
    """Report every problem in one suite, or that it has none.

    A ``200`` carrying ``valid: false`` rather than a ``422``, because the caller
    asked a question and "no" is a successful answer to it. Trying to RUN an
    invalid suite is what produces the ``422`` problem detail.
    """
    path = resources.catalog.suite_path(body.name)
    try:
        resources.catalog.validate(path)
    except BenchmarkValidationError as exc:
        return ValidateResponse(name=body.name, valid=False, errors=field_errors_of(exc.errors))
    return ValidateResponse(name=body.name, valid=True, errors=())


@router.get("/benchmarks/{name:path}")
async def benchmark(resources: Resources, name: str) -> BenchmarkDetail:
    """Return one suite in resolved, executable form, with the digest a run records.

    Secret-shaped values are scrubbed on the way out; see
    :class:`~llm_eval_lab.api.schemas.BenchmarkDetail`.
    """
    loaded = resources.catalog.load_redacted_suite(resources.catalog.suite_path(name))
    return BenchmarkDetail(
        name=name,
        suite_hash=loaded.suite.suite_hash,
        redacted=loaded.redacted,
        suite=loaded.suite,
    )


__all__ = ["LISTING_TRUNCATED_HEADER", "router"]
