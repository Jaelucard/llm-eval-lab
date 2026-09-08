"""Read-only catalog queries and benchmark validation.

The CLI and the API both need to answer "what providers exist", "what
evaluators exist" and "is this benchmark file valid" without either of them
reaching into the registries or the loader directly. Those answers live here,
once.

Validation composes two passes that cannot live in one package: the structural
pass in ``datasets.loader`` and the evaluator-spec pass in
``evaluators.registry``. ``datasets`` sits below ``evaluators`` in the layered
contract and cannot import it, so the service injects the second pass into the
first. The result is a single call that reports every error in a file.
"""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Literal

from llm_eval_lab.datasets.loader import load_suite
from llm_eval_lab.datasets.resolve import resolve_suite
from llm_eval_lab.evaluators import default_registry as evaluator_registry
from llm_eval_lab.evaluators.registry import EvaluatorInfo, EvaluatorRegistry
from llm_eval_lab.models import (
    BenchmarkSuite,
    BenchmarkValidationError,
    LLMEvalError,
    PriceTable,
    ResolvedSuite,
)
from llm_eval_lab.pricing.loader import DEFAULT_PRICE_TABLE_PATH, load_price_table
from llm_eval_lab.providers import default_registry as provider_registry
from llm_eval_lab.providers.registry import ProviderInfo, ProviderRegistry
from llm_eval_lab.redaction import scrub_value
from llm_eval_lab.settings import Settings


@dataclass(frozen=True)
class ModelInfo:
    """One row of the model catalog.

    ``source`` says where the row came from: a provider that advertises the
    model, or a price table that quotes it. A model both of them know produces
    one row with ``source="provider"`` carrying the quoted prices.

    ``credential_env`` is an environment variable NAME and ``credential_resolved``
    says only whether that variable is set. Neither this object nor anything
    built from it ever holds a credential value.
    """

    provider: str
    model: str
    source: Literal["provider", "price_table"]
    match: str | None
    input_per_mtok: Decimal | None
    output_per_mtok: Decimal | None
    available: bool
    extra: str | None
    credential_env: str | None
    credential_resolved: bool | None


@dataclass(frozen=True)
class PriceTableReport:
    """The outcome of validating a price file."""

    path: str
    table: PriceTable
    conflicts: tuple[str, ...]
    duplicates: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Report whether the file is usable as it stands."""
        return not self.conflicts and not self.duplicates


@dataclass(frozen=True)
class ValidationReport:
    """A validated suite plus the facts a caller wants to print about it."""

    path: str
    suite_name: str
    suite_version: str
    suite_hash: str
    n_cases: int
    evaluator_types: tuple[str, ...]
    resolved: ResolvedSuite


SUITE_SUFFIXES: tuple[str, ...] = (".yaml", ".yml", ".json")
"""Extensions a suite name may omit, tried in this order."""

MAX_LISTED_SUITES = 500
"""Cap on how many suite names one listing RETURNS."""

MAX_SCANNED_ENTRIES = 20_000
"""Cap on how many directory entries one listing WALKS to find them.

Two bounds, because one cannot do both jobs. Stopping the walk at
`MAX_LISTED_SUITES` bounded the work but made the result depend on the order the
file system happened to yield entries, so the same root could list different
suites on two calls and a suite could be unlistable-but-loadable. Scanning to
this larger bound, then sorting and truncating, makes the result deterministic -
the lexicographically first `MAX_LISTED_SUITES` names - while still refusing to
walk a root somebody pointed at a home directory.

A listing that hit either bound reports `truncated`, so "there are more suites
than these" is a fact the caller is told rather than one it has to infer.
"""


class SuiteNameError(LLMEvalError):
    """A suite NAME could not be turned into a readable file under the root.

    Raised for a name that is unusable as a name: absolute, containing a `..`
    segment, containing a path separator this project does not accept, or
    resolving (after symlinks) outside the configured benchmark root. Callers
    render it as a client error, never as a file-system error, and the message
    never contains the resolved absolute path.
    """


class SuiteNotFoundError(SuiteNameError):
    """The name was well formed and inside the root, but no such file exists."""


def _reject_unusable_name(name: str) -> PurePosixPath:
    """Turn an untrusted suite name into a relative POSIX path, or refuse it.

    Every refusal here happens BEFORE the file system is touched, so a hostile
    name cannot be probed for existence. The residual case a syntactic check
    cannot catch - a symlink inside the root pointing outside it - is caught by
    the containment check in :func:`resolve_suite_name` after `resolve()`.

    Raises:
        SuiteNameError: when the name is not usable as a suite name.
    """
    if not name or name != name.strip():
        msg = "suite name must be a non-empty name with no surrounding whitespace"
        raise SuiteNameError(msg)
    if "\\" in name or "\x00" in name:
        msg = "suite name must not contain a backslash or a null byte"
        raise SuiteNameError(msg)
    if any(character < " " for character in name):
        msg = "suite name must not contain control characters"
        raise SuiteNameError(msg)
    if name.startswith("/") or Path(name).is_absolute():
        msg = "suite name must be relative to the configured benchmark root"
        raise SuiteNameError(msg)
    relative = PurePosixPath(name)
    if any(part in {"..", "."} for part in relative.parts):
        msg = "suite name must not contain a '.' or '..' path segment"
        raise SuiteNameError(msg)
    return relative


def resolve_suite_name(root: Path | None, name: str) -> Path:
    """Resolve one untrusted suite NAME to a file inside `root`.

    The API never accepts a server-side path from a client; it accepts a name,
    and this function is the only place a name becomes a path. Containment is
    checked with ``Path.resolve().is_relative_to(root)`` AFTER symlinks are
    followed, which is what refuses a symlink planted inside the root that
    points outside it.

    Raises:
        SuiteNameError: when no root is configured, when the name is unusable,
            or when it resolves outside the root.
        SuiteNotFoundError: when nothing readable is at the resolved location.
    """
    # The NAME is judged before the configuration is. With the order reversed, a
    # deployment that had simply not set a root answered every traversal probe
    # with "no benchmark root is configured", which misattributes the refusal:
    # an operator reading their logs would go looking for missing configuration
    # instead of at the probe.
    relative = _reject_unusable_name(name)
    if root is None:
        msg = (
            "no benchmark root is configured, so suites cannot be addressed by name; "
            "set LLM_EVAL_SUITES_ROOT to the directory holding the benchmark files"
        )
        raise SuiteNameError(msg)
    resolved_root = Path(root).expanduser().resolve()

    bases = [resolved_root / Path(*relative.parts)]
    if relative.suffix.lower() not in SUITE_SUFFIXES:
        bases = [base.with_name(base.name + suffix) for base in bases for suffix in SUITE_SUFFIXES]

    for base in bases:
        resolved = base.resolve()
        if not resolved.is_relative_to(resolved_root):
            msg = f"suite name {name!r} resolves outside the configured benchmark root"
            raise SuiteNameError(msg)
        if resolved.is_file():
            return resolved
    msg = f"no benchmark suite named {name!r} under the configured benchmark root"
    raise SuiteNotFoundError(msg)


@dataclass(frozen=True)
class SuiteListing:
    """The suite names under a root, and whether that is all of them.

    ``truncated`` is true when either bound was reached: more matching files than
    :data:`MAX_LISTED_SUITES`, or more directory entries than
    :data:`MAX_SCANNED_ENTRIES`. A caller that sees it knows the list is a
    window, not an inventory, and can address the missing suites by name even
    though they are not shown.
    """

    names: tuple[str, ...]
    n_found: int
    truncated: bool


def list_suite_names(root: Path | None) -> SuiteListing:
    """List the suite files under `root`, by the name a client addresses them with.

    Deterministic: every matching name found within the scan bound is collected,
    sorted, and the first :data:`MAX_LISTED_SUITES` are returned. Truncating the
    WALK instead bounded the same work but returned whatever the file system
    yielded first, so two calls against one root could disagree.

    A file whose resolved path escapes the root - a symlink to elsewhere - is
    skipped rather than listed, so the listing and :func:`resolve_suite_name`
    agree about what exists.
    """
    if root is None:
        return SuiteListing(names=(), n_found=0, truncated=False)
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.is_dir():
        return SuiteListing(names=(), n_found=0, truncated=False)

    names: list[str] = []
    scanned = 0
    scan_exhausted = False
    for path in resolved_root.rglob("*"):
        scanned += 1
        if scanned > MAX_SCANNED_ENTRIES:
            scan_exhausted = True
            break
        if path.suffix.lower() not in SUITE_SUFFIXES:
            continue
        resolved = path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
            continue
        names.append(path.relative_to(resolved_root).as_posix())

    names.sort()
    return SuiteListing(
        names=tuple(names[:MAX_LISTED_SUITES]),
        n_found=len(names),
        truncated=scan_exhausted or len(names) > MAX_LISTED_SUITES,
    )


@dataclass(frozen=True)
class RedactedSuite:
    """A resolved suite with every secret-shaped value replaced, ready to return.

    ``suite_hash`` still addresses the ORIGINAL file, which is what a run records
    and what a comparison pairs on. ``redacted`` says whether the body beside it
    was altered, so a consumer never silently treats a scrubbed suite as the
    bytes that were hashed.
    """

    suite: ResolvedSuite
    redacted: bool


@dataclass(frozen=True)
class SuiteSummary:
    """One row of the benchmark listing.

    A file that does not load is LISTED, with `valid=False` and the count of
    problems in it. Hiding it would make a broken suite look like a missing one,
    and the operator would go looking for the wrong fault.
    """

    name: str
    valid: bool
    suite_name: str | None
    version: str | None
    suite_hash: str | None
    n_cases: int | None
    evaluator_types: tuple[str, ...]
    n_errors: int


class CatalogService:
    """Answers catalog and validation questions for the CLI and the API."""

    def __init__(
        self,
        *,
        settings: Settings,
        providers: ProviderRegistry,
        evaluators: EvaluatorRegistry,
    ) -> None:
        """Bind the service to the registries it reports on."""
        self._settings = settings
        self._providers = providers
        self._evaluators = evaluators

    def providers(self) -> tuple[ProviderInfo, ...]:
        """Return every registered provider."""
        return self._providers.catalog()

    def evaluators(self) -> tuple[EvaluatorInfo, ...]:
        """Return every registered evaluator type."""
        return self._evaluators.catalog()

    def credential_status(self, provider: str) -> bool | None:
        """Report whether a provider's default credential variable is set.

        ``None`` for a provider that needs none, so "not required" is never
        rendered as "required and missing". The value is never returned; only
        whether the variable resolved.

        Raises:
            ProviderNotInstalledError: when the provider is not registered.
        """
        return self._providers.credential_status(provider)

    def price_table(self, path: Path | str | None = None) -> PriceTable:
        """Load a price table from `path`, or the one shipped with the package.

        Raises:
            PricingError: when the table cannot be read or is invalid.
        """
        return load_price_table(path)

    def models(self, prices: PriceTable) -> tuple[ModelInfo, ...]:
        """List every model a provider advertises or a price table quotes.

        A provider-advertised model is looked up in the table so its unit prices
        travel with it. A price-table entry no provider advertises is still
        listed, because a table quoting a model this build cannot reach is
        exactly the mismatch an operator needs to see. An entry matched by
        prefix or regex keeps its pattern in ``model`` and says so in ``match``,
        rather than being expanded into model ids nobody has verified exist.
        """
        rows: list[ModelInfo] = []
        seen: set[tuple[str, str]] = set()
        registered = {info.name: info for info in self._providers.catalog()}

        for info in registered.values():
            resolved = self._providers.credential_status(info.name)
            for model in info.models:
                entry = prices.lookup(info.name, model)
                seen.add((info.name, model))
                rows.append(
                    ModelInfo(
                        provider=info.name,
                        model=model,
                        source="provider",
                        match=None if entry is None else entry.match,
                        input_per_mtok=None if entry is None else entry.input_per_mtok,
                        output_per_mtok=None if entry is None else entry.output_per_mtok,
                        # BOTH facts, not credential status alone: a provider
                        # whose optional extra is not installed is not
                        # available no matter what the environment holds.
                        available=info.available and resolved is not False,
                        extra=info.extra,
                        credential_env=info.credential_env,
                        credential_resolved=resolved,
                    )
                )

        for entry in prices.models:
            if (entry.provider, entry.model) in seen:
                continue
            quoted = registered.get(entry.provider)
            resolved = None if quoted is None else self._providers.credential_status(entry.provider)
            rows.append(
                ModelInfo(
                    provider=entry.provider,
                    model=entry.model,
                    source="price_table",
                    match=entry.match,
                    input_per_mtok=entry.input_per_mtok,
                    output_per_mtok=entry.output_per_mtok,
                    available=(quoted is not None and quoted.available and resolved is not False),
                    extra=None if quoted is None else quoted.extra,
                    credential_env=None if quoted is None else quoted.credential_env,
                    credential_resolved=resolved,
                )
            )
        return tuple(sorted(rows, key=lambda row: (row.provider, row.model, row.source)))

    def validate_prices(self, path: Path | str | None = None) -> PriceTableReport:
        """Load a price file and report every problem that makes it unsafe to use.

        Two problems are checked, and both of them are ways a cost figure stops
        being reproducible.

        A file that reuses the SHIPPED table's ``id`` and ``version`` with
        different content is refused. The pair is what a run records to say
        which prices it was costed with, so two different contents under one
        pair make a historical run's costs unrecoverable. Bumping the version
        is the fix.

        A file quoting the same provider and model twice under the same match
        mode is refused too: ``PriceTable.lookup`` resolves ties in declaration
        order, so a duplicate silently picks a winner the author did not choose.

        Raises:
            PricingError: when the file cannot be read or parsed at all.
        """
        table = load_price_table(path)
        resolved_path = str(path) if path is not None else str(DEFAULT_PRICE_TABLE_PATH)

        conflicts: list[str] = []
        shipped = load_price_table(None)
        same_identity = table.id == shipped.id and table.version == shipped.version
        if same_identity and table.content_hash != shipped.content_hash:
            conflicts.append(
                f"id {table.id!r} version {table.version!r} is already used by the price "
                f"table shipped with this package, with different content "
                f"({shipped.content_hash} vs {table.content_hash}). Bump the version: a run "
                f"records the id and version to say which prices it was costed with."
            )

        counts: dict[tuple[str, str, str], int] = {}
        for entry in table.models:
            key = (entry.provider, entry.model, entry.match)
            counts[key] = counts.get(key, 0) + 1
        duplicates = tuple(
            f"{provider}/{model} is declared {count} times with match={match!r}"
            for (provider, model, match), count in sorted(counts.items())
            if count > 1
        )

        return PriceTableReport(
            path=resolved_path,
            table=table,
            conflicts=tuple(conflicts),
            duplicates=duplicates,
        )

    def load_suite_pair(self, path: Path | str) -> tuple[BenchmarkSuite, ResolvedSuite]:
        """Load one benchmark file, returning the authored AND the resolved form.

        Both, because they answer different questions. The resolved form is what
        executes and what is hashed; the authored form still carries the suite's
        DEFAULTS, which the resolved form has already merged away, and a run
        records those defaults so its configuration says something.

        Raises:
            BenchmarkValidationError: carrying every problem found in the file.
        """
        suite = load_suite(
            path,
            root=self._settings.suites_root,
            max_bytes=self._settings.max_suite_bytes,
            validators=(self._evaluators.validate_specs,),
        )
        return suite, resolve_suite(suite)

    def load_resolved_suite(self, path: Path | str) -> ResolvedSuite:
        """Load, validate and resolve one benchmark file.

        Raises:
            BenchmarkValidationError: carrying every problem found in the file.
        """
        return self.load_suite_pair(path)[1]

    def load_redacted_suite(self, path: Path | str) -> RedactedSuite:
        """Load one suite with every secret-shaped value replaced.

        A benchmark file is a document an operator edits by hand, and the two
        free-form dictionaries in it - a suite's ``params.extra`` and an
        evaluator's ``params`` - are exactly where somebody eventually pastes a
        key. The storage layer already scrubs them on the way into the snapshot
        table; returning the file unscrubbed over HTTP would reopen the same hole
        one layer up. The same scrubber runs here, so the two paths cannot
        disagree about what counts as secret-shaped.

        Raises:
            BenchmarkValidationError: carrying every problem found in the file.
        """
        resolved = self.load_resolved_suite(path)
        original = resolved.model_dump(mode="json")
        scrubbed = scrub_value(original)
        if scrubbed == original:
            return RedactedSuite(suite=resolved, redacted=False)
        return RedactedSuite(suite=ResolvedSuite.model_validate(scrubbed), redacted=True)

    def validate(self, path: Path | str) -> ValidationReport:
        """Validate one benchmark file and summarize what it contains.

        Raises:
            BenchmarkValidationError: carrying every problem found in the file.
        """
        resolved = self.load_resolved_suite(path)
        types = sorted({spec.type for case in resolved.cases for spec in case.evaluators})
        return ValidationReport(
            path=str(path),
            suite_name=resolved.name,
            suite_version=resolved.version,
            suite_hash=resolved.suite_hash,
            n_cases=len(resolved.cases),
            evaluator_types=tuple(types),
            resolved=resolved,
        )

    # -- suites addressed by name ----------------------------------------

    def suite_root(self) -> Path | None:
        """Return the configured benchmark root, or ``None`` when there is none."""
        return self._settings.suites_root

    def suite_names(self) -> SuiteListing:
        """List the addressable suite names under the configured root."""
        return list_suite_names(self._settings.suites_root)

    def suite_path(self, name: str) -> Path:
        """Resolve one suite NAME to a file inside the configured root.

        Raises:
            SuiteNameError: when no root is configured or the name escapes it.
            SuiteNotFoundError: when no such file exists.
        """
        return resolve_suite_name(self._settings.suites_root, name)

    def suite_summaries(self) -> tuple[tuple[SuiteSummary, ...], SuiteListing]:
        """Summarize the addressable suites, marking the ones that do not load.

        Returns the summaries AND the listing they came from, so a caller can
        tell "these are all of them" from "these are the first few hundred".
        """
        listing = self.suite_names()
        summaries: list[SuiteSummary] = []
        for name in listing.names:
            try:
                report = self.validate(self.suite_path(name))
            except BenchmarkValidationError as exc:
                summaries.append(
                    SuiteSummary(
                        name=name,
                        valid=False,
                        suite_name=None,
                        version=None,
                        suite_hash=None,
                        n_cases=None,
                        evaluator_types=(),
                        n_errors=len(exc.errors),
                    )
                )
                continue
            summaries.append(
                SuiteSummary(
                    name=name,
                    valid=True,
                    suite_name=report.suite_name,
                    version=report.suite_version,
                    suite_hash=report.suite_hash,
                    n_cases=report.n_cases,
                    evaluator_types=report.evaluator_types,
                    n_errors=0,
                )
            )
        return tuple(summaries), listing


def build_catalog_service(settings: Settings) -> CatalogService:
    """Build a catalog service against the process-wide registries.

    The registries are reached here rather than in the CLI or the API, both of
    which are forbidden from importing ``providers`` and ``evaluators`` directly.
    """
    return CatalogService(
        settings=settings,
        providers=provider_registry(),
        evaluators=evaluator_registry(),
    )
