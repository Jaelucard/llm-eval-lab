"""Orchestration: the only layer that composes the others.

The CLI and the API call into these services and know nothing about the runner,
the registries, the loader or the database. Everything below is reached through
the frozen Protocols in ``llm_eval_lab.models``.

``BUDGET_EXCEEDED`` is the one bare re-export: it is the exact value
``RunEngine`` writes to ``Run.error`` for a budget-triggered cancellation, and
the CLI's exit-code mapping (`cli.commands.run.exit_code_for`) has to compare
against that same string to tell it apart from a `SIGINT` cancellation, which
shares the same `RunStatus.CANCELLED`. The import-linter layering contract
forbids `cli` from importing `llm_eval_lab.runner` directly, so this is the
one legitimate path the constant travels through rather than being respelled
as a second string literal in the CLI, which is exactly the "one implementation"
discipline this project's own docstrings apply everywhere else.
"""

from llm_eval_lab.runner.engine import BUDGET_EXCEEDED
from llm_eval_lab.services.catalog_service import (
    MAX_LISTED_SUITES,
    CatalogService,
    ModelInfo,
    PriceTableReport,
    RedactedSuite,
    SuiteListing,
    SuiteNameError,
    SuiteNotFoundError,
    SuiteSummary,
    ValidationReport,
    build_catalog_service,
    list_suite_names,
    resolve_suite_name,
)
from llm_eval_lab.services.comparison_service import (
    ComparisonService,
    ComparisonView,
    RunNotFoundError,
    build_comparison_service,
)
from llm_eval_lab.services.db_service import (
    database_revision,
    open_unit_of_work_factory,
    upgrade_database,
)
from llm_eval_lab.services.metrics_service import (
    MetricsService,
    MetricsView,
    ModelSummary,
    build_metrics_service,
    export_document,
)
from llm_eval_lab.services.run_manager import (
    CancelOutcome,
    LaunchRequest,
    RunManager,
    RunProgress,
    TooManyRunsError,
)
from llm_eval_lab.services.run_service import (
    DryRunPlan,
    PreparedRun,
    RunRequest,
    RunService,
    RunView,
    build_run_service,
    compute_pass_rate,
    run_summary_payload,
)

__all__ = [
    "BUDGET_EXCEEDED",
    "MAX_LISTED_SUITES",
    "CancelOutcome",
    "CatalogService",
    "ComparisonService",
    "ComparisonView",
    "DryRunPlan",
    "LaunchRequest",
    "MetricsService",
    "MetricsView",
    "ModelInfo",
    "ModelSummary",
    "PreparedRun",
    "PriceTableReport",
    "RedactedSuite",
    "RunManager",
    "RunNotFoundError",
    "RunProgress",
    "RunRequest",
    "RunService",
    "RunView",
    "SuiteListing",
    "SuiteNameError",
    "SuiteNotFoundError",
    "SuiteSummary",
    "TooManyRunsError",
    "ValidationReport",
    "build_catalog_service",
    "build_comparison_service",
    "build_metrics_service",
    "build_run_service",
    "compute_pass_rate",
    "database_revision",
    "export_document",
    "list_suite_names",
    "open_unit_of_work_factory",
    "resolve_suite_name",
    "run_summary_payload",
    "upgrade_database",
]
