"""The pass-rate rule. One definition, used by every surface that reports one.

There were two implementations of this arithmetic and they disagreed. The runs
listing divided by ``n_completed - n_errors``; the ``show`` view additionally
dropped every case whose evaluators produced a score but no boolean. For a
score-only suite the two surfaces therefore reported different numbers for the
same run, and phase 3 was about to build ``AggregateMetrics`` on one of them.

So the rule lives here, once, and both surfaces call it.

The rule, stated plainly:

* An ERRORED case is excluded from the denominator under ``error_policy
  ="exclude"`` (the default) and counted as a failure under ``"fail"``. Either
  way it is reported separately as an error rate, over total attempted.
* A SCORE-ONLY case - one whose evaluators returned a score but no pass/fail
  verdict, so ``passed is None`` - is excluded from both the numerator and the
  denominator. It is not a failure; nobody said what passing would mean.
* An empty denominator yields ``None``, never ``0.0``. "Nothing was eligible to
  be scored" and "everything failed" are different facts and a reporting layer
  that conflates them is lying.

This module sits at the package root, outside the layer stack, for the same
reason ``redaction.py`` does: ``storage`` and ``services`` both need it and the
layered contract gives them no shared package above ``models`` to reach it
through.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from llm_eval_lab.models import CaseResult, CaseStatus

type ErrorPolicy = Literal["exclude", "fail"]

DEFAULT_ERROR_POLICY: ErrorPolicy = "exclude"


@dataclass(frozen=True)
class CaseVerdict:
    """The two facts about a case that decide whether it counts toward a pass rate."""

    errored: bool
    passed: bool | None


@dataclass(frozen=True)
class PassRate:
    """A pass rate together with the counts behind it.

    The counts travel with the ratio on purpose: a rate of ``1.0`` over one
    eligible case and over four hundred are very different claims, and a caller
    that only receives the ratio cannot tell them apart.
    """

    n_passed: int
    n_denominator: int
    rate: float | None


def verdict_of(status: CaseStatus, *, passed: bool | None) -> CaseVerdict:
    """Build a verdict from the two persisted columns that decide it."""
    return CaseVerdict(errored=status is not CaseStatus.OK, passed=passed)


def normalize_error_policy(value: str) -> ErrorPolicy:
    """Return `value` as a known error policy.

    Raises:
        ValueError: for anything that is not `exclude` or `fail`. Falling back
            to the default instead would make a typo in a config file change
            every pass rate in the report while looking like it had been
            honoured.
    """
    if value == "exclude":
        return "exclude"
    if value == "fail":
        return "fail"
    msg = f"error_policy must be 'exclude' or 'fail', got {value!r}"
    raise ValueError(msg)


def pass_rate(
    verdicts: Iterable[CaseVerdict],
    *,
    error_policy: str = DEFAULT_ERROR_POLICY,
) -> PassRate:
    """Apply the pass-rate rule to a set of case verdicts. See the module docstring.

    Raises:
        ValueError: when `error_policy` is neither `exclude` nor `fail`.
    """
    policy = normalize_error_policy(error_policy)
    n_passed = 0
    n_denominator = 0
    for verdict in verdicts:
        if verdict.errored:
            if policy != "fail":
                continue
            n_denominator += 1
            continue
        if verdict.passed is None:
            continue
        n_denominator += 1
        if verdict.passed:
            n_passed += 1
    return PassRate(
        n_passed=n_passed,
        n_denominator=n_denominator,
        rate=(n_passed / n_denominator if n_denominator else None),
    )


def pass_rate_of_results(
    results: Sequence[CaseResult],
    *,
    error_policy: str = DEFAULT_ERROR_POLICY,
) -> PassRate:
    """Apply the pass-rate rule to persisted case results.

    Raises:
        ValueError: when `error_policy` is neither `exclude` nor `fail`.
    """
    return pass_rate(
        (verdict_of(result.status, passed=result.passed) for result in results),
        error_policy=error_policy,
    )
