"""The pass-rate rule, tested directly rather than only through its two callers."""

import pytest

from llm_eval_lab.models import CaseStatus
from llm_eval_lab.scoring import CaseVerdict, pass_rate, verdict_of

TWO_OF_THREE = 2 / 3


def _verdicts(*pairs: tuple[bool, bool | None]) -> list[CaseVerdict]:
    return [CaseVerdict(errored=errored, passed=passed) for errored, passed in pairs]


def test_an_empty_set_has_no_pass_rate() -> None:
    result = pass_rate([])
    assert result.rate is None, "'nothing was eligible' must not read as 'everything failed'"
    assert result.n_denominator == 0


def test_errored_cases_are_excluded_under_the_default_policy() -> None:
    result = pass_rate(_verdicts((False, True), (False, False), (True, None)))
    assert result.n_denominator == 2
    assert result.n_passed == 1
    assert result.rate == 0.5


def test_errored_cases_count_as_failures_under_the_fail_policy() -> None:
    result = pass_rate(_verdicts((False, True), (True, None)), error_policy="fail")
    assert result.n_denominator == 2
    assert result.n_passed == 1
    assert result.rate == 0.5


def test_score_only_cases_are_in_neither_the_numerator_nor_the_denominator() -> None:
    result = pass_rate(_verdicts((False, True), (False, None), (False, None)))
    assert result.n_denominator == 1
    assert result.rate == 1.0


def test_a_run_of_only_score_only_cases_has_no_pass_rate() -> None:
    assert pass_rate(_verdicts((False, None), (False, None))).rate is None


def test_score_only_cases_are_excluded_under_the_fail_policy_too() -> None:
    """`fail` changes how ERRORS count; it says nothing about an absent verdict."""
    result = pass_rate(_verdicts((False, None), (False, True)), error_policy="fail")
    assert result.n_denominator == 1
    assert result.rate == 1.0


def test_all_passing() -> None:
    assert pass_rate(_verdicts((False, True), (False, True))).rate == 1.0


def test_mixed_fractions_are_exact_enough_to_compare() -> None:
    result = pass_rate(_verdicts((False, True), (False, True), (False, False)))
    assert result.rate == pytest.approx(TWO_OF_THREE)


@pytest.mark.parametrize(
    ("status", "expected_errored"),
    [
        (CaseStatus.OK, False),
        (CaseStatus.ERROR, True),
        (CaseStatus.TIMEOUT, True),
        (CaseStatus.SKIPPED, True),
        (CaseStatus.CANCELLED, True),
    ],
)
def test_every_non_ok_status_is_an_error_for_scoring(
    status: CaseStatus,
    expected_errored: bool,  # noqa: FBT001 - a parametrized expectation, not a flag
) -> None:
    assert verdict_of(status, passed=None).errored is expected_errored
