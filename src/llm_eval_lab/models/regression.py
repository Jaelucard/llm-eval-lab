"""Regression threshold policy and comparison report.

The gate is deterministic by design (decision D-GATE): thresholds evaluate
point estimates only, so the same two runs always produce the same verdict.
Wilson intervals, the Newcombe Method-10 interval on paired pass-rate deltas
and the McNemar exact p-value are computed and displayed as advisory context
for a human, and never participate in pass/fail logic.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_eval_lab.models.common import UtcDatetime
from llm_eval_lab.models.enums import CheckStatus, ThresholdDirection, Verdict
from llm_eval_lab.models.errors import EvaluatorConfigError

_REQUIRE_SIGNIFICANT_MESSAGE = "require_significant is reserved for a future release"


class MetricCheck(BaseModel):
    """One threshold applied to one metric. Pure data; it evaluates nothing.

    Bound semantics, which the phase 4 engine in ``reporting/regression.py``
    implements and which are carried in the field descriptions so there is one
    written source: ``min_value`` is an INCLUSIVE floor, violated when the
    candidate is strictly below it. ``max_value`` is an EXCLUSIVE ceiling,
    violated when the candidate is greater than or equal to it. That asymmetry
    is what makes "error rate must remain below 1%" expressible as
    ``max_value: 0.01``, where a candidate of exactly ``0.01`` fails.

    Evaluation deliberately does not live here. Gate semantics are engine
    behaviour, not contract shape, and a frozen model carrying them would force
    every later correction through the frozen-file protocol.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    direction: ThresholdDirection = Field(
        description=(
            "Which way this metric has to move to be an improvement. Descriptive: "
            "it labels the metric for rendering and arrow direction. The bounds "
            "below are already directional, so an engine must not derive a second "
            "directional rule from this field."
        ),
    )
    label: str | None = None
    max_absolute_decrease: float | None = None
    max_relative_decrease: float | None = None
    max_absolute_increase: float | None = None
    max_relative_increase: float | None = None
    min_value: float | None = Field(
        default=None,
        description=(
            "INCLUSIVE floor. Violated when the candidate is strictly below it, "
            "so a candidate exactly equal to min_value passes."
        ),
    )
    max_value: float | None = Field(
        default=None,
        description=(
            "EXCLUSIVE ceiling. Violated when the candidate is greater than or "
            "equal to it, so a candidate exactly equal to max_value FAILS. That "
            "asymmetry is what makes 'error rate must remain below 1%' "
            "expressible as max_value: 0.01."
        ),
    )
    require_significant: bool = False
    severity: Literal["error", "warning"] = "error"
    min_samples: int = 1

    @model_validator(mode="after")
    def _reject_reserved_require_significant(self) -> "MetricCheck":
        """Reject `require_significant`, which is reserved and unimplemented in v1."""
        if self.require_significant:
            raise EvaluatorConfigError(_REQUIRE_SIGNIFICANT_MESSAGE)
        return self


class RegressionThresholds(BaseModel):
    """A named, versioned threshold policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = "default"
    version: str = "1"
    checks: tuple[MetricCheck, ...] = Field(min_length=1)
    paired: bool = True
    require_same_suite: bool = True
    min_paired_cases: int = 1
    fail_on_missing_metric: bool = True
    confidence_level: float = 0.95


class CheckOutcome(BaseModel):
    """The result of one threshold check, with its advisory statistics.

    ``p_value`` is the McNemar exact test and is populated only for paired
    pass-rate comparisons. ``confidence_interval`` is the Newcombe Method-10
    interval on the paired pass-rate delta, and is ``None`` for every other
    metric and in unpaired mode, where the two Wilson intervals in
    ``baseline_ci`` and ``candidate_ci`` take its place instead.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    label: str
    direction: ThresholdDirection
    status: CheckStatus
    baseline: float | None
    candidate: float | None
    delta: float | None
    relative_delta: float | None
    violated_bound: str | None
    threshold_value: float | None
    n_baseline: int
    n_candidate: int
    n_paired: int | None
    confidence_interval: tuple[float, float] | None
    baseline_ci: tuple[float, float] | None = None
    candidate_ci: tuple[float, float] | None = None
    intervals_overlap: bool | None = None
    p_value: float | None
    significant: bool | None
    note: str | None


class CaseDelta(BaseModel):
    """How one case moved between the baseline run and the candidate run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    baseline_score: float | None
    candidate_score: float | None
    delta: float | None
    baseline_passed: bool | None
    candidate_passed: bool | None
    category: str | None = None


class RegressionSummary(BaseModel):
    """Which cases flipped, and the largest score drops behind the headline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    newly_failing: tuple[str, ...]
    newly_passing: tuple[str, ...]
    still_failing: tuple[str, ...]
    n_flipped: int
    largest_score_drops: tuple[CaseDelta, ...]


class RegressionReport(BaseModel):
    """A complete baseline-versus-candidate comparison.

    ``mode`` is ``paired`` when the two runs share case ids and ``unpaired``
    otherwise. Thresholds evaluate point estimates in both modes, so the gate
    verdict never depends on the mode.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    generated_at: UtcDatetime
    baseline_run_id: str
    candidate_run_id: str
    baseline_label: str | None
    candidate_label: str | None
    verdict: Verdict
    mode: Literal["paired", "unpaired"]
    intervals_overlap: bool | None = None
    comparable: bool
    incomparable_reason: str | None
    suite_hash_match: bool
    baseline_suite_hash: str
    candidate_suite_hash: str
    paired_case_count: int
    only_in_baseline: tuple[str, ...]
    only_in_candidate: tuple[str, ...]
    changed_cases: tuple[str, ...]
    thresholds_id: str
    thresholds_version: str
    checks: tuple[CheckOutcome, ...]
    summary: RegressionSummary
