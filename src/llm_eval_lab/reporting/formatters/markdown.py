"""Markdown reports, for a run and for a comparison.

Markdown rather than HTML, deliberately. A reviewer already reads Markdown in a
pull request, a terminal pager and a chat window, and JSON already serves
machines. An HTML template would buy presentation polish at the cost of a third
rendering path to test and to secure against the untrusted model output and
benchmark text it would embed.

Two presentation rules carry over from the Rich renderers, for the same reasons.
An unknown value renders as an em dash and never as ``0``: a cost of zero and a
cost nobody could compute are different facts. And every data-derived string
goes through :func:`markdown_cell`, because a case id or a category name comes
out of a benchmark file, which is untrusted data (decision D-CUSTOM). An
unescaped pipe there does not merely look wrong, it silently splits one table
cell into two and shifts every column after it.
"""

from collections.abc import Iterable, Sequence

from llm_eval_lab.models import (
    AggregateMetrics,
    CheckOutcome,
    CheckStatus,
    LatencyStats,
    RegressionReport,
    Run,
    ThresholdDirection,
)

UNKNOWN = "&mdash;"
"""What an absent value renders as. Never a zero, which reads as a measurement."""

_STATUS_LABELS: dict[CheckStatus, str] = {
    CheckStatus.PASSED: "PASSED",
    CheckStatus.FAILED: "FAILED",
    CheckStatus.WARNING: "WARNING",
    CheckStatus.INSUFFICIENT_DATA: "INSUFFICIENT_DATA",
    CheckStatus.MISSING_METRIC: "MISSING_METRIC",
}
"""Uppercase status names, so a report is greppable by the contract's own words."""

_ARROWS: dict[ThresholdDirection, str] = {
    ThresholdDirection.HIGHER_IS_BETTER: "higher is better",
    ThresholdDirection.LOWER_IS_BETTER: "lower is better",
}
"""What ``direction`` is for: labelling, not a second directional rule."""

_MAX_LISTED_CASES = 50
"""How many case ids a list renders before it says how many more there are."""


def markdown_cell(value: object) -> str:
    """Render `value` as literal text inside a Markdown table cell.

    A pipe inside a cell ends the cell, so an unescaped case id containing one
    shifts every column after it. Newlines end the ROW, which is worse. Both are
    neutralised here rather than at each of the dozen call sites.
    """
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _number(value: float | None, places: int = 4) -> str:
    """Render a float, or the unknown marker."""
    return UNKNOWN if value is None else f"{value:.{places}f}"


def _signed(value: float | None, places: int = 4) -> str:
    """Render a delta with an explicit sign, so a decrease is visible at a glance."""
    return UNKNOWN if value is None else f"{value:+.{places}f}"


def _interval(bounds: tuple[float, float] | None, places: int = 4) -> str:
    """Render a confidence interval, or the unknown marker."""
    if bounds is None:
        return UNKNOWN
    return f"[{bounds[0]:.{places}f}, {bounds[1]:.{places}f}]"


def _case_list(case_ids: Sequence[str]) -> str:
    """Render a list of case ids as a bulleted list, or say there were none."""
    if not case_ids:
        return "_none_\n"
    shown = [f"- `{markdown_cell(case_id)}`" for case_id in case_ids[:_MAX_LISTED_CASES]]
    if len(case_ids) > _MAX_LISTED_CASES:
        shown.append(f"- _and {len(case_ids) - _MAX_LISTED_CASES} more_")
    return "\n".join(shown) + "\n"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Render one Markdown table. Cells are assumed already escaped."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _check_row(check: CheckOutcome) -> list[str]:
    """Render one threshold outcome as a table row.

    Every row carries baseline, candidate, delta and threshold even when the
    check did not resolve, because a row that dropped those columns would be a
    different shape from the rows around it and would break a reader scanning
    down one column.
    """
    return [
        markdown_cell(check.label),
        markdown_cell(check.metric),
        _number(check.baseline),
        _number(check.candidate),
        _signed(check.delta),
        _number(check.threshold_value),
        markdown_cell(check.violated_bound or UNKNOWN),
        f"{check.n_baseline} / {check.n_candidate}",
        _STATUS_LABELS[check.status],
    ]


def check_table(checks: Sequence[CheckOutcome]) -> str:
    """Render every threshold outcome, one row each."""
    if not checks:
        return "_no checks were evaluated_\n"
    headers = (
        "check",
        "metric",
        "baseline",
        "candidate",
        "delta",
        "threshold",
        "violated bound",
        "n base / cand",
        "status",
    )
    return _table(headers, [_check_row(check) for check in checks])


def _advisory_row(check: CheckOutcome) -> list[str]:
    """Render one check's advisory statistics."""
    return [
        markdown_cell(check.label),
        markdown_cell(_ARROWS[check.direction]),
        _interval(check.confidence_interval),
        _interval(check.baseline_ci),
        _interval(check.candidate_ci),
        UNKNOWN if check.intervals_overlap is None else str(check.intervals_overlap).lower(),
        UNKNOWN if check.p_value is None else f"{check.p_value:.6f}",
    ]


def advisory_table(checks: Sequence[CheckOutcome]) -> str:
    """Render the statistics that inform a reader and never move a verdict."""
    interesting = [
        check
        for check in checks
        if check.confidence_interval is not None
        or check.baseline_ci is not None
        or check.p_value is not None
    ]
    if not interesting:
        return "_no advisory statistics apply to this comparison_\n"
    headers = (
        "check",
        "direction",
        "95% CI on delta",
        "baseline 95% CI",
        "candidate 95% CI",
        "intervals overlap",
        "McNemar p",
    )
    return _table(headers, [_advisory_row(check) for check in interesting])


def _notes(checks: Sequence[CheckOutcome]) -> str:
    """Render the per-check notes, which is where every caveat is written down."""
    lines = [
        f"- **{markdown_cell(check.label)}** &mdash; {markdown_cell(check.note)}"
        for check in checks
        if check.note
    ]
    return "\n".join(lines) + "\n" if lines else "_none_\n"


def _header(report: RegressionReport) -> list[str]:
    """Render the title block: who was compared, and how the runs line up."""
    baseline = report.baseline_label or report.baseline_run_id
    candidate = report.candidate_label or report.candidate_run_id
    return [
        "# Regression report",
        "",
        f"**Verdict: {report.verdict.value.upper()}**",
        "",
        f"- Comparison mode: **{report.mode}**",
        f"- Paired cases: **{report.paired_case_count}**",
        (f"- Baseline: `{markdown_cell(report.baseline_run_id)}` ({markdown_cell(baseline)})"),
        (f"- Candidate: `{markdown_cell(report.candidate_run_id)}` ({markdown_cell(candidate)})"),
        (
            f"- Suite hash match: **{str(report.suite_hash_match).lower()}** "
            f"(`{markdown_cell(report.baseline_suite_hash)}` vs "
            f"`{markdown_cell(report.candidate_suite_hash)}`)"
        ),
        (
            f"- Threshold policy: `{markdown_cell(report.thresholds_id)}` "
            f"version `{markdown_cell(report.thresholds_version)}`"
        ),
        f"- Generated at: {report.generated_at.isoformat()}",
        "",
    ]


def _gate_note(report: RegressionReport) -> list[str]:
    """State the gate rule, once, in the report itself."""
    return [
        (
            "> Thresholds compare point estimates and resolve deterministically, so the "
            "same two runs always produce the same verdict. The intervals and the "
            "McNemar p-value below are advisory context for a human and never change a "
            f"status. This comparison ran in **{report.mode}** mode."
        ),
        "",
    ]


def comparison_markdown(report: RegressionReport) -> str:
    """Render a full comparison as Markdown.

    The document is deliberately greppable: the verdict appears as an uppercase
    word, every check is one table row carrying baseline, candidate, delta and
    threshold, and the newly-failing cases are a named section rather than a
    parenthetical.
    """
    lines = _header(report)
    if not report.comparable:
        lines.extend(
            [
                "## Not comparable",
                "",
                markdown_cell(report.incomparable_reason or "the two runs cannot be compared"),
                "",
            ]
        )
    lines.extend(_gate_note(report))
    lines.extend(["## Threshold checks", "", check_table(report.checks), ""])
    lines.extend(["## Advisory statistics", "", advisory_table(report.checks), ""])
    lines.extend(["## Notes", "", _notes(report.checks), ""])
    lines.extend(
        [
            "## Case movement",
            "",
            (
                f"Paired cases: **{report.paired_case_count}**. "
                f"Cases that flipped: **{report.summary.n_flipped}**."
            ),
            "",
            "### Newly failing",
            "",
            _case_list(report.summary.newly_failing),
            "",
            "### Newly passing",
            "",
            _case_list(report.summary.newly_passing),
            "",
            "### Still failing",
            "",
            _case_list(report.summary.still_failing),
            "",
        ]
    )
    if report.summary.largest_score_drops:
        rows = [
            [
                markdown_cell(drop.case_id),
                markdown_cell(drop.category or UNKNOWN),
                _number(drop.baseline_score),
                _number(drop.candidate_score),
                _signed(drop.delta),
            ]
            for drop in report.summary.largest_score_drops
        ]
        lines.extend(
            [
                "### Largest score drops",
                "",
                _table(
                    ("case", "category", "baseline", "candidate", "delta"),
                    rows,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Case set differences",
            "",
            "### Changed cases (same id, different hash; excluded from pairing)",
            "",
            _case_list(report.changed_cases),
            "",
            "### Only in the baseline run",
            "",
            _case_list(report.only_in_baseline),
            "",
            "### Only in the candidate run",
            "",
            _case_list(report.only_in_candidate),
            "",
        ]
    )
    return "\n".join(lines)


def _latency_rows(latency: LatencyStats) -> list[list[str]]:
    """Render the latency distribution, marking anything below its sample gate."""
    fields: tuple[tuple[str, str, float | None], ...] = (
        ("mean", "mean_ms", latency.mean_ms),
        ("p50", "p50_ms", latency.p50_ms),
        ("p90", "p90_ms", latency.p90_ms),
        ("p95", "p95_ms", latency.p95_ms),
        ("p99", "p99_ms", latency.p99_ms),
        ("max", "max_ms", latency.max_ms),
    )
    return [
        [
            label,
            _number(value, places=1),
            "insufficient samples" if field in latency.low_confidence else "",
        ]
        for label, field, value in fields
    ]


def metrics_markdown(run: Run, metrics: AggregateMetrics, *, materialized: bool) -> str:
    """Render one run's rollup as Markdown.

    Every rate appears beside the counts behind it. A pass rate of ``1.0`` over
    three eligible cases and over three hundred are different claims, and only
    the denominator distinguishes them.
    """
    lines = [
        "# Run report",
        "",
        f"- Run: `{markdown_cell(run.id)}`"
        + (f" ({markdown_cell(run.label)})" if run.label else ""),
        f"- Status: **{run.status.value}**",
        (
            f"- Suite: `{markdown_cell(run.config.suite_name)}` "
            f"version `{markdown_cell(run.config.suite_version)}` "
            f"(`{markdown_cell(run.config.suite_hash)}`)"
        ),
        (
            f"- Model: `{markdown_cell(run.config.provider.provider)}` / "
            f"`{markdown_cell(run.config.provider.model)}`"
        ),
        f"- Error policy: `{markdown_cell(metrics.error_policy)}`",
        f"- Rollup source: {'stored' if materialized else 'computed on demand'}",
        "",
        "## Headline figures",
        "",
    ]
    rows = [
        ["cases completed", f"{metrics.n_completed} / {metrics.n_cases}"],
        ["pass rate", _number(metrics.pass_rate)],
        ["passed", f"{metrics.n_passed} / {metrics.n_pass_denominator}"],
        ["pass rate 95% CI", _interval(metrics.pass_rate_ci)],
        ["error rate", _number(metrics.error_rate)],
        ["errors", str(metrics.n_errors)],
        ["timeouts", str(metrics.n_timeouts)],
        ["score-only evaluations", str(metrics.n_score_only)],
        ["mean score", _number(metrics.mean_score)],
        ["median score", _number(metrics.median_score)],
        ["token usage coverage", _number(metrics.token_usage_coverage)],
        ["cost coverage", _number(metrics.cost_coverage)],
        [
            "total cost (candidate model)",
            UNKNOWN if metrics.cost.total_cost is None else str(metrics.cost.total_cost),
        ],
        [
            "judge cost (separate)",
            UNKNOWN if metrics.cost.judge_cost is None else str(metrics.cost.judge_cost),
        ],
    ]
    lines.extend([_table(("figure", "value"), rows), ""])
    lines.extend(
        [
            "## Latency over successful model attempts",
            "",
            (
                f"Samples: **{metrics.latency.n}**. Retry backoff is excluded; it lives "
                f"in `total_latency_ms`."
            ),
            "",
            _table(("statistic", "ms", "confidence"), _latency_rows(metrics.latency)),
            "",
        ]
    )
    if metrics.by_category:
        rows = [
            [
                markdown_cell(name),
                str(bucket.n),
                str(bucket.n_passed),
                _number(bucket.pass_rate),
                _interval(bucket.pass_rate_ci),
                _number(bucket.mean_score),
            ]
            for name, bucket in metrics.by_category.items()
        ]
        lines.extend(
            [
                "## By category",
                "",
                _table(
                    ("category", "cases", "passed", "pass rate", "95% CI", "mean score"),
                    rows,
                ),
                "",
            ]
        )
    if metrics.by_evaluator:
        rows = [
            [
                markdown_cell(name),
                markdown_cell(bucket.evaluator_type),
                str(bucket.n),
                str(bucket.n_passed),
                str(bucket.n_errors),
                _number(bucket.pass_rate),
                _number(bucket.mean_score),
            ]
            for name, bucket in metrics.by_evaluator.items()
        ]
        lines.extend(
            [
                "## By evaluator",
                "",
                (
                    "Per-evaluator means are never blended: different evaluators score "
                    "on different scales."
                ),
                "",
                _table(
                    ("evaluator", "type", "n", "passed", "errors", "pass rate", "mean score"),
                    rows,
                ),
                "",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "UNKNOWN",
    "advisory_table",
    "check_table",
    "comparison_markdown",
    "markdown_cell",
    "metrics_markdown",
]
