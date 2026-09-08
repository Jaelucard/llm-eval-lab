"""Rich renderables for the terminal.

Three presentation rules, all of them honesty rules rather than style ones.

An unknown value renders as ``-`` and never as ``0`` or as a blank that reads
like zero. A percentile below its sample-size gate renders with a ``low-n``
marker and a one-line explanation underneath the table, so a number nobody
should trust is never shown looking exactly like one that can be.

Every data-derived cell and every table TITLE goes through :func:`markup_safe`.
Rich reads square brackets as style tags and DROPS what it cannot resolve, so an
unescaped cell is silently deleted rather than mis-styled: a category named
``[urgent]`` vanishes from the breakdown, and the redaction marker
``[redacted]`` vanishes from a printed URL, which turns "a password was removed
here" into "this database has an empty password". A benchmark file and a price
table are both untrusted data (decision D-CUSTOM), and a title interpolating a
suite or price-table name is as exposed as a cell.

:func:`markup_safe` lives here rather than in ``cli`` because both surfaces
render Rich and the layered contract lets ``cli`` import ``reporting.formatters``
but not the reverse. One implementation, so the next renderer cannot ship a
weaker one.
"""

from collections.abc import Mapping
from decimal import Decimal

from rich.markup import escape
from rich.table import Table

from llm_eval_lab.models import (
    AggregateMetrics,
    CategoryMetrics,
    EvaluatorMetrics,
    LatencyStats,
    PriceEntry,
)
from llm_eval_lab.reporting.statistics import P50_MIN_N, P90_MIN_N, P95_MIN_N, P99_MIN_N

LOW_CONFIDENCE_MARKER = "low-n"
"""Appended to a percentile whose sample count is below its gate."""

LOW_CONFIDENCE_NOTE = (
    f"low-n: insufficient samples for this percentile to be trustworthy "
    f"(p50 needs {P50_MIN_N}, p90 needs {P90_MIN_N}, p95 needs {P95_MIN_N}, "
    f"p99 needs {P99_MIN_N}). The value is still computed and shown."
)
"""The one-line explanation printed beneath any table carrying a marker."""


def markup_safe(value: object) -> str:
    """Render `value` as literal text for a Rich sink.

    Every string reaching a console or a table cell goes through this. Rich
    treats ``[...]`` as a style tag and silently discards one it cannot resolve,
    so an unescaped value is not mis-styled, it is DELETED.
    """
    return escape(str(value))


def _number(value: float | None, places: int = 4) -> str:
    """Render a float, or ``-`` when it is genuinely unknown."""
    return "-" if value is None else f"{value:.{places}f}"


def _money(value: Decimal | None) -> str:
    """Render a money value at full precision, or ``-`` when it is unknown."""
    return "-" if value is None else str(value)


def _interval(bounds: tuple[float, float] | None) -> str:
    """Render a confidence interval, or ``-`` when there was nothing to compute one over."""
    return "-" if bounds is None else f"[{bounds[0]:.4f}, {bounds[1]:.4f}]"


def _count(value: int, total: int) -> str:
    """Render a count against its denominator, so a rate is never read alone."""
    return f"{value}/{total}"


def metrics_table(metrics: AggregateMetrics) -> Table:
    """Render the headline figures for one run.

    The pass rate appears beside the counts it came from, because a rate of
    ``1.0`` over three eligible cases and over three hundred are different
    claims and only the denominator distinguishes them.
    """
    table = Table(
        title=markup_safe(f"metrics for run {metrics.run_id}"), show_header=False, box=None
    )
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("cases", _count(metrics.n_completed, metrics.n_cases))
    table.add_row("pass rate", _number(metrics.pass_rate))
    table.add_row("passed", _count(metrics.n_passed, metrics.n_pass_denominator))
    table.add_row("pass rate 95% CI", _interval(metrics.pass_rate_ci))
    table.add_row("error rate", _number(metrics.error_rate))
    table.add_row("errors", str(metrics.n_errors))
    table.add_row("timeouts", str(metrics.n_timeouts))
    table.add_row("error policy", metrics.error_policy)
    table.add_row("score-only evaluations", str(metrics.n_score_only))
    table.add_row("mean score", _number(metrics.mean_score))
    table.add_row("median score", _number(metrics.median_score))
    table.add_row("scored cases", str(metrics.n_scored))
    table.add_row("token usage coverage", _number(metrics.token_usage_coverage))
    table.add_row("cost coverage", _number(metrics.cost_coverage))
    table.add_row("total cost (candidate model)", _money(metrics.cost.total_cost))
    table.add_row("judge cost (separate)", _money(metrics.cost.judge_cost))
    table.add_row("cost per case", _money(metrics.cost_per_case))
    table.add_row("cost per passing case", _money(metrics.cost_per_successful_evaluation))
    # `markup_safe`, because the price-table id and version come out of a
    # user-supplied YAML file. Rich reads `[...]` as a style tag and silently
    # DELETES one it cannot resolve, so an id like `prices[2026-q3]` would render
    # as `prices@1` and quietly misname the table a historical cost was computed
    # against. Every other data-derived cell in this module is already escaped;
    # this one was the gap.
    table.add_row(
        "price table",
        markup_safe(f"{metrics.cost.price_table_id}@{metrics.cost.price_table_version}"),
    )
    return table


def latency_table(latency: LatencyStats) -> Table:
    """Render the latency distribution, marking every percentile below its gate.

    Each row carries its display label AND its contract field name. The marker
    is matched on the field name, because ``low_confidence`` holds field names;
    matching on the label would silently stop marking anything the first time
    somebody tidied a heading.
    """
    table = Table(title=markup_safe(f"latency over {latency.n} successful cases"), box=None)
    table.add_column("statistic", style="bold")
    table.add_column("ms")
    table.add_column("confidence")

    rows: tuple[tuple[str, str, float | None], ...] = (
        ("mean", "mean_ms", latency.mean_ms),
        ("p50", "p50_ms", latency.p50_ms),
        ("p90", "p90_ms", latency.p90_ms),
        ("p95", "p95_ms", latency.p95_ms),
        ("p99", "p99_ms", latency.p99_ms),
        ("max", "max_ms", latency.max_ms),
    )
    for label, field, value in rows:
        marker = LOW_CONFIDENCE_MARKER if field in latency.low_confidence else ""
        table.add_row(label, _number(value, places=1), marker)
    return table


def category_table(buckets: Mapping[str, CategoryMetrics], title: str) -> Table:
    """Render a category or tag breakdown."""
    table = Table(title=markup_safe(title), box=None)
    for column in ("bucket", "cases", "passed", "pass rate", "95% CI", "mean score"):
        table.add_column(column)
    for key, bucket in buckets.items():
        table.add_row(
            markup_safe(key),
            str(bucket.n),
            str(bucket.n_passed),
            _number(bucket.pass_rate),
            _interval(bucket.pass_rate_ci),
            _number(bucket.mean_score),
        )
    return table


def evaluator_table(buckets: Mapping[str, EvaluatorMetrics]) -> Table:
    """Render the per-evaluator breakdown.

    Per-evaluator means stand in their own rows and are never combined. A
    lexical similarity score and a judge's normalized rating are different
    measurements on different scales, and one blended number would be
    arithmetic on incompatible units.
    """
    table = Table(title="by evaluator", box=None)
    for column in (
        "evaluator",
        "type",
        "n",
        "passed",
        "errors",
        "score-only",
        "pass rate",
        "mean score",
        "median score",
    ):
        table.add_column(column)
    for key, bucket in buckets.items():
        table.add_row(
            markup_safe(key),
            markup_safe(bucket.evaluator_type),
            str(bucket.n),
            str(bucket.n_passed),
            str(bucket.n_errors),
            str(bucket.n_score_only),
            _number(bucket.pass_rate),
            _number(bucket.mean_score),
            _number(bucket.median_score),
        )
    return table


def price_entry_table(entries: tuple[PriceEntry, ...], title: str = "prices") -> Table:
    """Render the entries of a price table, per million tokens."""
    table = Table(title=markup_safe(title), box=None)
    for column in ("provider", "model", "match", "input /Mtok", "output /Mtok", "cached /Mtok"):
        table.add_column(column)
    for entry in entries:
        table.add_row(
            markup_safe(entry.provider),
            markup_safe(entry.model),
            entry.match,
            _money(entry.input_per_mtok),
            _money(entry.output_per_mtok),
            _money(entry.cached_input_per_mtok),
        )
    return table


def models_table(rows: tuple[Mapping[str, object], ...]) -> Table:
    """Render the model catalog.

    A model that needs an uninstalled extra or an unset credential is listed
    with the extra's name and the environment variable's NAME. The value is
    never read and never shown; naming the variable is what makes the row
    actionable without turning a catalog listing into a credential dump.
    """
    table = Table(title="models", box=None)
    for column in (
        "provider",
        "model",
        "source",
        "input /Mtok",
        "output /Mtok",
        "available",
        "extra",
        "credential env var",
    ):
        table.add_column(column)
    for row in rows:
        table.add_row(
            markup_safe(row.get("provider", "-")),
            markup_safe(row.get("model", "-")),
            str(row.get("source", "-")),
            "-" if row.get("input_per_mtok") is None else str(row["input_per_mtok"]),
            "-" if row.get("output_per_mtok") is None else str(row["output_per_mtok"]),
            "yes" if row.get("available") else "no",
            markup_safe(row.get("extra") or "-"),
            markup_safe(row.get("credential_env") or "-"),
        )
    return table


__all__ = [
    "LOW_CONFIDENCE_MARKER",
    "LOW_CONFIDENCE_NOTE",
    "category_table",
    "evaluator_table",
    "latency_table",
    "markup_safe",
    "metrics_table",
    "models_table",
    "price_entry_table",
]
