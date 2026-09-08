"""Composing a run: load, resolve, select, price, persist, execute.

This is the only orchestration layer. The CLI and the API both call into it and
neither of them knows about the runner, the registries, the loader or the
database. Everything a run needs to be reproducible is assembled here and
persisted verbatim in :class:`~llm_eval_lab.models.RunConfig` - the resolved
suite digest, the exact provider configuration (credential NAMES only, never
values), the price table identity, and the library and interpreter versions.

``--dry-run`` is handled here too, and its contract is strict: it validates the
suite, resolves the provider entry and the price table, reports the effective
configuration and a cost ESTIMATE, and returns without ever constructing a
provider client or issuing a request. A test proves that by patching the
provider factory to raise on construction.
"""

import asyncio
import platform
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pydantic
import structlog

from llm_eval_lab import __version__
from llm_eval_lab.datasets.resolve import select_cases, snapshot_of
from llm_eval_lab.evaluators import default_registry as evaluator_registry
from llm_eval_lab.evaluators.registry import EvaluatorRegistry
from llm_eval_lab.models import (
    BenchmarkValidationError,
    CaseQuery,
    CaseResult,
    CaseStatus,
    EvaluatorSpec,
    FieldError,
    GenerationParams,
    JSONValue,
    Page,
    PluginRecord,
    PriceTable,
    ProgressCallback,
    ProviderConfig,
    ResolvedCase,
    ResolvedSuite,
    Run,
    RunConfig,
    RunQuery,
    RunStatus,
    RunSummary,
    RunTotals,
    TokenUsage,
    UnitOfWorkFactory,
)
from llm_eval_lab.pricing.calculator import compute_cost
from llm_eval_lab.pricing.loader import load_price_table
from llm_eval_lab.providers import default_registry as provider_registry
from llm_eval_lab.providers.registry import ProviderRegistry
from llm_eval_lab.runner.engine import RunEngine, RunOutcome
from llm_eval_lab.scoring import pass_rate_of_results
from llm_eval_lab.services.catalog_service import CatalogService
from llm_eval_lab.settings import Settings
from llm_eval_lab.utils.time import utc_now

CHARS_PER_TOKEN = 4
"""The estimator's assumption, stated so a dry-run cost estimate can be judged."""

DEFAULT_ESTIMATED_OUTPUT_TOKENS = 256
"""Assumed output length when a case sets no `max_output_tokens`."""

TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.INTERRUPTED,
    }
)
"""Statuses a run does not leave. `fail_run` refuses to overwrite one."""


@dataclass(frozen=True)
class RunRequest:
    """Everything the caller chose about one run, before any of it is resolved."""

    suite_path: Path
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    provider_options: dict[str, JSONValue] = field(default_factory=dict)
    label: str | None = None
    tags: tuple[str, ...] = ()
    select_tags: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    limit: int | None = None
    sample: int | None = None
    sample_seed: int | None = None
    concurrency: int | None = None
    case_timeout_s: float | None = None
    max_cost: Decimal | None = None
    error_policy: Literal["exclude", "fail"] = "exclude"
    seed: int | None = None
    price_table_path: Path | None = None
    notes: str | None = None
    plugins: tuple[PluginRecord, ...] = ()
    """Evaluator plugins already loaded by the caller (decision D-CUSTOM).

    This service never loads a plugin itself - see
    `llm_eval_lab.evaluators.plugins` for why that stays a CLI-only,
    process-start action. It only records what the caller (`cli/main.py`)
    already loaded, so a run's persisted `RunConfig.plugins` says which
    third-party evaluator code, if any, it depended on.
    """


@dataclass(frozen=True)
class PreparedRun:
    """Everything a run needs, resolved once: config, suite, cases and prices."""

    config: RunConfig
    suite: ResolvedSuite
    cases: tuple[ResolvedCase, ...]
    prices: PriceTable


@dataclass(frozen=True)
class DryRunPlan:
    """What a run WOULD do, produced without constructing a provider client."""

    config: RunConfig
    n_cases: int
    case_ids: tuple[str, ...]
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: Decimal | None
    estimate_unpriced_reason: str | None
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class RunView:
    """A run plus the derived figures a caller wants next to it."""

    run: Run
    n_cases: int
    n_completed: int
    n_errors: int
    n_passed: int
    n_pass_denominator: int
    pass_rate: float | None


def compute_pass_rate(
    results: Sequence[CaseResult],
    *,
    error_policy: str = "exclude",
) -> tuple[int, int, float | None]:
    """Return (passed, denominator, pass_rate) over a set of case results.

    A thin adapter over :func:`llm_eval_lab.scoring.pass_rate_of_results`, which
    is the single definition of the rule. The runs listing in ``storage`` calls
    the same function, so the two surfaces cannot drift apart again.
    """
    computed = pass_rate_of_results(results, error_policy=error_policy)
    return computed.n_passed, computed.n_denominator, computed.rate


_PROVIDER_CONFIG_LABEL = "<provider configuration>"
"""The `FieldError.file` value for a `ProviderConfig` construction failure.

There is no benchmark file behind this error - the offending input is a CLI
flag or an API request field - so this stands in for a file name the same way
`datasets.loader._file_level_error` uses `<file>` for a file-level problem.
"""


def _provider_config_validation_error(exc: pydantic.ValidationError) -> BenchmarkValidationError:
    """Map a `ProviderConfig` construction failure to the project's validation error.

    `ProviderConfig`'s own validators (see `models.common`) already produce a
    precise, safe message naming the offending option's LOCATION - never its
    value: `pydantic.ValidationError.errors()[i]["input"]` for a secret-shaped
    option carries the rejected value verbatim, so it is never read here. Only
    `loc` (the field path) and `msg` (the validator's own text) travel into the
    resulting `FieldError`, which is what keeps a secret pasted into
    `--provider-option` out of CLI output, logs and API responses alike - the
    same guarantee the rejection itself exists to provide.

    This reuses `BenchmarkValidationError` rather than a new exception type:
    it is already the project's one "malformed input, exit 3, 422 over HTTP"
    error, wired through `cli/main.py`'s `guard()` and `api/errors.py` alike.
    """
    errors = tuple(
        FieldError(
            file=_PROVIDER_CONFIG_LABEL,
            location=".".join(str(part) for part in item["loc"]) or _PROVIDER_CONFIG_LABEL,
            case_id=None,
            message=str(item["msg"]).removeprefix("Value error, "),
            input_repr=None,
        )
        for item in exc.errors()
    )
    lines = [
        f"{_PROVIDER_CONFIG_LABEL}: {len(errors)} validation error(s)",
        *(f"  {item.location}: {item.message}" for item in errors),
    ]
    return BenchmarkValidationError("\n".join(lines), errors=errors)


class RunService:
    """Builds, persists and executes runs. The single orchestration entry point."""

    def __init__(
        self,
        *,
        settings: Settings,
        uow_factory: UnitOfWorkFactory,
        providers: ProviderRegistry,
        evaluators: EvaluatorRegistry,
        logger: structlog.BoundLogger,
    ) -> None:
        """Bind the service to its collaborators."""
        self._settings = settings
        self._uow_factory = uow_factory
        self._providers = providers
        self._evaluators = evaluators
        self._log = logger
        self._catalog = CatalogService(
            settings=settings, providers=providers, evaluators=evaluators
        )

    # -- assembling a run ------------------------------------------------

    def provider_config(self, request: RunRequest) -> ProviderConfig:
        """Build the provider configuration for a request.

        Records the NAME of the credential environment variable, never a value,
        which is what makes this object safe to persist, log and return over the
        API verbatim.

        Raises:
            BenchmarkValidationError: when `request.provider_options` fails
                `ProviderConfig`'s own validation - most notably a secret-shaped
                option key (see `models.common._reject_secretish_option_keys`).
                Mapped from the raw `pydantic.ValidationError` so the CLI's
                `guard()` reports this as a clean, single-line usage error at
                exit code 3 instead of an unhandled traceback at exit code 1;
                the API already turns the same error type into a 422.
        """
        try:
            return ProviderConfig(
                provider=request.provider,
                model=request.model,
                api_key_env=request.api_key_env,
                base_url=request.base_url,
                options=dict(request.provider_options),
            )
        except pydantic.ValidationError as exc:
            raise _provider_config_validation_error(exc) from exc

    def ensure_credential(self, config: ProviderConfig) -> None:
        """Check that a provider configuration's credential resolves, discarding it.

        The seam the API uses to fail a launch synchronously rather than in a
        background task. Deliberately NOT folded into :meth:`prepare`: the CLI
        runs a suite in the foreground and already surfaces the same error from
        the engine with its own exit code, and moving the check would change when
        a `--dry-run` fails.

        Raises:
            ProviderNotInstalledError: when the provider name is not registered.
            MissingCredentialError: when a named credential variable is unset.
        """
        self._providers.ensure_credential(config)

    def price_table(self, request: RunRequest) -> PriceTable:
        """Load the price table this run will be costed against.

        Raises:
            PricingError: when the table cannot be loaded.
        """
        return load_price_table(request.price_table_path)

    def build_config(  # noqa: PLR0913 - one parameter per independent input
        self,
        request: RunRequest,
        suite: ResolvedSuite,
        cases: Sequence[ResolvedCase],
        selection_source: object,
        prices: PriceTable,
        *,
        params: GenerationParams,
    ) -> RunConfig:
        """Assemble the effective, persisted configuration of a run.

        ``params`` is the SUITE's default generation parameters. Per-case
        overrides are already merged into the resolved cases and travel in the
        snapshot; recording the defaults here is what stops the run-level field
        being an uninformative empty object.
        """
        evaluators: list[EvaluatorSpec] = []
        seen: set[str] = set()
        for case in cases:
            for spec in case.evaluators:
                key = f"{spec.type}:{spec.id or ''}"
                if key not in seen:
                    seen.add(key)
                    evaluators.append(spec)
        return RunConfig.model_validate(
            {
                "suite_name": suite.name,
                "suite_version": suite.version,
                "suite_hash": suite.suite_hash,
                "suite_source": str(request.suite_path),
                "case_selection": selection_source,
                "provider": self.provider_config(request),
                "params": params,
                "evaluators": tuple(evaluators),
                "concurrency": request.concurrency or self._settings.concurrency,
                "case_timeout_s": request.case_timeout_s or self._settings.case_timeout_s,
                "judge_concurrency": self._settings.judge_concurrency,
                "error_policy": request.error_policy,
                "max_cost": request.max_cost,
                "price_table_id": prices.id,
                "price_table_version": prices.version,
                "price_table_hash": prices.content_hash,
                "library_version": __version__,
                "python_version": platform.python_version(),
                "seed": request.seed,
                "notes": request.notes,
                "plugins": request.plugins,
            }
        )

    def prepare(self, request: RunRequest) -> PreparedRun:
        """Load and resolve everything a run needs, touching no provider and no database.

        The price table is loaded here and CARRIED on the result. It used to be
        loaded again by each caller, which read, parsed and content-hashed a
        user-supplied ``--prices`` file twice per run for no reason.

        Raises:
            BenchmarkValidationError: when the suite file is invalid.
            PricingError: when the price table cannot be loaded.
            ProviderNotInstalledError: when the provider name is not registered.
        """
        authored, suite = self._catalog.load_suite_pair(request.suite_path)
        cases, selection = select_cases(
            suite,
            tags=request.select_tags,
            case_ids=request.case_ids,
            limit=request.limit,
            sample=request.sample,
            sample_seed=request.sample_seed,
        )
        # Resolving the provider ENTRY (not constructing a client) so a typo in
        # --provider fails before anything is written or spent.
        self._providers.info(request.provider)
        prices = self.price_table(request)
        config = self.build_config(
            request, suite, cases, selection, prices, params=authored.defaults.params
        )
        return PreparedRun(config=config, suite=suite, cases=cases, prices=prices)

    # -- dry run ---------------------------------------------------------

    def dry_run(self, request: RunRequest) -> DryRunPlan:
        """Report what a run would do, without constructing a provider or writing a row.

        Raises:
            BenchmarkValidationError: when the suite file is invalid.
            PricingError: when the price table cannot be loaded.
            ProviderNotInstalledError: when the provider name is not registered.
        """
        prepared = self.prepare(request)
        config, cases, prices = prepared.config, prepared.cases, prepared.prices

        input_tokens = 0
        output_tokens = 0
        for case in cases:
            prompt = "".join(message.content for message in case.messages) + (case.system or "")
            input_tokens += max(1, len(prompt) // CHARS_PER_TOKEN)
            output_tokens += case.params.max_output_tokens or DEFAULT_ESTIMATED_OUTPUT_TOKENS

        estimate = compute_cost(
            prices,
            provider=config.provider.provider,
            model=config.provider.model,
            usage=_usage(input_tokens, output_tokens),
        )
        return DryRunPlan(
            config=config,
            n_cases=len(cases),
            case_ids=tuple(case.id for case in cases),
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost=estimate.total_cost,
            estimate_unpriced_reason=estimate.unpriced_reason,
            assumptions=(
                f"input tokens estimated at one token per {CHARS_PER_TOKEN} prompt characters",
                (
                    f"output tokens estimated at max_output_tokens, or "
                    f"{DEFAULT_ESTIMATED_OUTPUT_TOKENS} where the case sets none"
                ),
                "no request was issued and no provider client was constructed",
            ),
        )

    # -- executing a run -------------------------------------------------

    async def create_run(
        self,
        config: RunConfig,
        suite: ResolvedSuite,
        cases: Sequence[ResolvedCase],
        *,
        label: str | None,
        tags: tuple[str, ...],
    ) -> Run:
        """Persist the suite snapshot and the PENDING run row."""
        now = utc_now()
        run = Run(
            id=str(uuid.uuid4()),
            label=label,
            status=RunStatus.PENDING,
            created_at=now,
            started_at=None,
            completed_at=None,
            config=config,
            totals=RunTotals(n_cases=len(cases)),
            updated_at=now,
            tags=tags,
        )
        async with self._uow_factory() as uow:
            await uow.benchmarks.upsert_snapshot(
                snapshot_of(suite, source_path=config.suite_source)
            )
            await uow.runs.create(run)
            await uow.commit()
        return run

    def engine(self, prices: PriceTable, logger: structlog.BoundLogger) -> RunEngine:
        """Build the run engine with everything it depends on injected."""
        return RunEngine(
            uow_factory=self._uow_factory,
            provider_factory=self._providers.open,
            evaluators=self._evaluators,
            price_table=prices,
            evaluator_settings=self._settings.evaluator_settings(),
            logger=logger,
        )

    async def execute(
        self,
        request: RunRequest,
        *,
        cancel_event: asyncio.Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> RunOutcome:
        """Run a benchmark end to end and return what it produced.

        Raises:
            BenchmarkValidationError: when the suite file is invalid.
            PricingError: when the price table cannot be loaded.
            ProviderNotInstalledError | MissingCredentialError: when the provider
                cannot be constructed.
            StorageError: when persistence fails.
        """
        prepared = self.prepare(request)
        run = await self.create_run(
            prepared.config,
            prepared.suite,
            prepared.cases,
            label=request.label,
            tags=request.tags,
        )
        logger = self._log.bind(run_id=run.id)
        engine = self.engine(prepared.prices, logger)
        return await engine.execute(
            run, prepared.cases, cancel_event=cancel_event, progress=progress
        )

    # -- reading runs ----------------------------------------------------

    async def sweep_stale_runs(self, *, before: datetime) -> int:
        """Mark runs left RUNNING by a dead process as INTERRUPTED."""
        async with self._uow_factory() as uow:
            changed = await uow.runs.mark_stale_running_as_interrupted(before=before)
            await uow.commit()
        return changed

    async def fetch_run(self, run_id: str) -> Run | None:
        """Read one run row, without loading its case results.

        The cheap read behind status polling. :meth:`get_run` loads every
        persisted case so it can report a pass rate; a client polling once a
        second does not want that, and does not need it: the engine heartbeats
        `Run.totals` alongside every case it writes.
        """
        async with self._uow_factory() as uow:
            return await uow.runs.get(run_id)

    async def fail_run(self, run_id: str, *, error: str) -> None:
        """Record a run as FAILED because its execution raised before finishing.

        The engine writes its own terminal status for every failure it can see.
        This exists for the failures it cannot: a provider that could not be
        constructed at all raises out of :meth:`execute` before the first case,
        leaving the row RUNNING with nothing on the way. Without this, that run
        stays RUNNING until the next process restart sweeps it.

        A run that already holds a terminal status is left alone. The engine
        writes its own verdict before re-raising a `StorageError`, and a blind
        write here would replace an accurate CANCELLED or PARTIAL with FAILED,
        losing the one fact the row existed to record.

        Raises:
            StorageError: when the status cannot be written.
        """
        now = utc_now()
        async with self._uow_factory() as uow:
            existing = await uow.runs.get(run_id)
            if existing is None or existing.status in TERMINAL_RUN_STATUSES:
                return
            await uow.runs.update_status(
                run_id,
                RunStatus.FAILED,
                completed_at=now,
                updated_at=now,
                error=error,
            )
            await uow.commit()

    async def list_runs(self, query: RunQuery) -> Page[RunSummary]:
        """List runs matching the query."""
        async with self._uow_factory() as uow:
            return await uow.runs.list(query)

    async def get_run(self, run_id: str) -> RunView | None:
        """Fetch one run with its pass rate computed from the persisted cases."""
        async with self._uow_factory() as uow:
            run = await uow.runs.get(run_id)
            if run is None:
                return None
            page = await uow.cases.list_for_run(run_id, CaseQuery(limit=500))
            results = list(page.items)
            offset = len(results)
            while offset < page.total:
                more = await uow.cases.list_for_run(run_id, CaseQuery(limit=500, offset=offset))
                if not more.items:
                    break
                results.extend(more.items)
                offset += len(more.items)

        n_errors = sum(1 for item in results if item.status is not CaseStatus.OK)
        passed, denominator, rate = compute_pass_rate(results, error_policy=run.config.error_policy)
        return RunView(
            run=run,
            n_cases=run.totals.n_cases,
            n_completed=len(results),
            n_errors=n_errors,
            n_passed=passed,
            n_pass_denominator=denominator,
            pass_rate=rate,
        )

    async def list_cases(self, run_id: str, query: CaseQuery) -> Page[CaseResult]:
        """List the case results of one run."""
        async with self._uow_factory() as uow:
            return await uow.cases.list_for_run(run_id, query)

    async def get_case(self, run_id: str, case_id: str) -> CaseResult | None:
        """Fetch one case result of one run, with its evaluations."""
        async with self._uow_factory() as uow:
            return await uow.cases.get(run_id, case_id)


def _usage(input_tokens: int, output_tokens: int) -> TokenUsage:
    """Build a `TokenUsage` for the dry-run estimate."""
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def run_summary_payload(outcome: RunOutcome) -> dict[str, JSONValue]:
    """Project a finished run into the flat document the CLI and the API report.

    Lives here rather than in the CLI because both surfaces report the same
    facts, and because ``cli`` is forbidden from importing ``runner`` and so
    cannot name :class:`~llm_eval_lab.runner.engine.RunOutcome` at all.
    """
    finished = outcome.run
    return {
        "ok": finished.status is RunStatus.COMPLETED,
        "run_id": finished.id,
        "status": finished.status.value,
        "error": finished.error,
        "n_cases": finished.totals.n_cases,
        "n_completed": finished.totals.n_completed,
        "n_passed": finished.totals.n_passed,
        "n_errors": finished.totals.n_errors,
        "n_cancelled": finished.totals.n_cancelled,
        "budget_spent": str(outcome.budget_spent),
        "unpriced_cases": outcome.unpriced_cases,
        "suite_hash": finished.config.suite_hash,
    }


def build_run_service(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
) -> RunService:
    """Build a run service against the process-wide registries.

    The registries are reached here rather than in the CLI or the API, both of
    which are forbidden from importing ``providers``, ``evaluators`` and
    ``runner`` directly.
    """
    return RunService(
        settings=settings,
        uow_factory=uow_factory,
        providers=provider_registry(),
        evaluators=evaluator_registry(),
        logger=logger,
    )
