"""The renderers: Markdown documents, and the Rich escaping that protects them.

Rich treats ``[...]`` as a style tag and DROPS what it cannot resolve, so an
unescaped cell is deleted rather than mis-styled. Markdown has the same hazard
with a different character: an unescaped pipe splits one cell into two and
shifts every column after it. Both are data-integrity failures, and both are
tested here rather than left to a reader to notice.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from rich.console import Console

from llm_eval_lab.models import (
    CheckOutcome,
    CheckStatus,
    RegressionReport,
    RegressionSummary,
    ThresholdDirection,
    Verdict,
)
from llm_eval_lab.reporting.aggregate import compute_metrics
from llm_eval_lab.reporting.formatters import (
    check_table,
    comparison_markdown,
    markdown_cell,
    metrics_table,
)

AT = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)


def _outcome(**overrides: object) -> CheckOutcome:
    """Build one check outcome with every required field filled in."""
    payload: dict[str, object] = {
        "metric": "pass_rate",
        "label": "overall pass rate",
        "direction": ThresholdDirection.HIGHER_IS_BETTER,
        "status": CheckStatus.FAILED,
        "baseline": 0.9,
        "candidate": 0.84,
        "delta": -0.06,
        "relative_delta": -0.0667,
        "violated_bound": "max_absolute_decrease",
        "threshold_value": 0.88,
        "n_baseline": 50,
        "n_candidate": 50,
        "n_paired": 50,
        "confidence_interval": (-0.1749, 0.0487),
        "p_value": 0.453125,
        "significant": False,
        "note": None,
    }
    payload.update(overrides)
    return CheckOutcome.model_validate(payload)


def _report(**overrides: object) -> RegressionReport:
    """Build one regression report with every required field filled in."""
    payload: dict[str, object] = {
        "generated_at": AT,
        "baseline_run_id": "base",
        "candidate_run_id": "cand",
        "baseline_label": None,
        "candidate_label": None,
        "verdict": Verdict.FAIL,
        "mode": "paired",
        "comparable": True,
        "incomparable_reason": None,
        "suite_hash_match": True,
        "baseline_suite_hash": "sha256:suite",
        "candidate_suite_hash": "sha256:suite",
        "paired_case_count": 50,
        "only_in_baseline": (),
        "only_in_candidate": (),
        "changed_cases": (),
        "thresholds_id": "default",
        "thresholds_version": "1",
        "checks": (_outcome(),),
        "summary": RegressionSummary(
            newly_failing=("alpha", "beta"),
            newly_passing=(),
            still_failing=(),
            n_flipped=2,
            largest_score_drops=(),
        ),
    }
    payload.update(overrides)
    return RegressionReport.model_validate(payload)


def test_the_price_table_cell_is_escaped_for_rich() -> None:
    metrics = compute_metrics(
        run_id="run-1",
        n_cases=0,
        results=[],
        error_policy="exclude",
        price_table_id="prices[2026-q3]",
        price_table_version="7",
        computed_at=AT,
    )
    console = Console(file=None, no_color=True, width=200, record=True)

    console.print(metrics_table(metrics))

    # Without escaping Rich drops `[2026-q3]` as an unresolvable style tag and
    # the row silently misnames the table the costs came from.
    assert "prices[2026-q3]@7" in console.export_text()


def test_a_markdown_cell_cannot_break_out_of_its_column() -> None:
    assert markdown_cell("a|b") == r"a\|b"
    assert markdown_cell("line\nbreak") == "line break"
    assert "\n" not in markdown_cell("a\r\nb")


def test_a_case_id_containing_a_pipe_does_not_shift_the_table() -> None:
    table = check_table((_outcome(label="odd|label", metric="pass|rate"),))

    for line in table.strip().splitlines():
        assert line.count("|") - line.count(r"\|") == 10


def test_the_comparison_report_carries_everything_a_reviewer_greps_for() -> None:
    rendered = comparison_markdown(_report())

    assert "**Verdict: FAIL**" in rendered
    assert "Paired cases: **50**" in rendered
    assert "### Newly failing" in rendered
    assert "- `alpha`" in rendered
    assert "- `beta`" in rendered
    # One row per check, carrying baseline, candidate, delta and threshold.
    assert "| overall pass rate | pass_rate | 0.9000 | 0.8400 | -0.0600 | 0.8800 " in rendered
    assert "FAILED" in rendered
    assert "0.453125" in rendered


def test_an_incomparable_report_says_why() -> None:
    rendered = comparison_markdown(
        _report(
            verdict=Verdict.INCOMPARABLE,
            comparable=False,
            incomparable_reason="the two runs used different benchmark suites",
            checks=(),
        )
    )

    assert "**Verdict: INCOMPARABLE**" in rendered
    assert "## Not comparable" in rendered
    assert "different benchmark suites" in rendered
    assert "_no checks were evaluated_" in rendered


def test_an_unknown_value_never_renders_as_a_zero() -> None:
    rendered = comparison_markdown(
        _report(
            checks=(
                _outcome(
                    status=CheckStatus.MISSING_METRIC,
                    baseline=None,
                    candidate=None,
                    delta=None,
                    threshold_value=None,
                    violated_bound=None,
                    confidence_interval=None,
                    p_value=None,
                    significant=None,
                ),
            )
        )
    )

    row = next(line for line in rendered.splitlines() if "MISSING_METRIC" in line)
    assert "0.0000" not in row
    assert row.count("&mdash;") >= 4


@pytest.mark.parametrize("value", [Decimal("0.0000012345"), Decimal(0)])
def test_markdown_cells_do_not_round_money(value: Decimal) -> None:
    assert markdown_cell(value) == str(value)
