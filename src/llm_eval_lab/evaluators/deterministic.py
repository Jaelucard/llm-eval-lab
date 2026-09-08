"""The five deterministic evaluators: no model call, no network, no randomness.

``exact_match``, ``exact_match_ci``, ``contains``, ``regex`` and
``numeric_tolerance``. Given the same response and the same parameters each one
returns the same verdict on every machine, forever, which is what makes them
usable as a regression gate.

**The regex ReDoS mitigation, stated in full because it is the only genuinely
dangerous input path here.** A benchmark file supplies the pattern and the model
supplies the subject string, so catastrophic backtracking is reachable from data.
A length cap on the pattern does not help - ``(a+)+$`` is six characters.

The mitigation has two layers. At LOAD time the pattern is compiled and capped,
so an uncompilable pattern is a benchmark validation error naming the case and
field. At MATCH time the match runs in a SEPARATE PROCESS under
``asyncio.timeout``, and on expiry that process is killed; the evaluator returns
``status=ERROR`` with ``error.kind="timeout"`` and
``metadata["error_code"] = "regex_timeout"``, the case continues and the run is
unaffected.

The subprocess plumbing itself lives in :mod:`llm_eval_lab.evaluators.bounded`,
shared with the ``json_schema`` evaluator, whose JSON Schema ``pattern`` keyword
is the same hazard reached through a different door.

A separate process, specifically, and not a worker thread. CPython's ``re``
executes a match in a single C call that never returns to the interpreter loop
and never releases the GIL. A runaway match in a thread therefore does not merely
outlive its timeout - it starves every other thread in the process, including the
event loop that was supposed to time it out, so the timeout never fires and the
whole run hangs. That was measured, not assumed. Killing a process is the only
mechanism CPython offers that actually stops an ``re`` match in progress.

The cost is honest and bounded: one interpreter start (about 15 ms, measured, with
``-I -S`` and no project imports) per regex evaluation. Against a benchmark run
whose other half is a network call to a model, that is noise; if it ever stops
being noise, a persistent worker process is the obvious next step, and
:func:`bounded_match` is the seam for it.

How many can exist at once: exactly as many as the RUN's concurrency
(``RunConfig.concurrency``), because a case holds its concurrency slot across
generation and evaluation alike. There is no separate evaluator concurrency
limit in this slice. At ``--concurrency 32`` on a regex-heavy suite that is up
to 32 simultaneous interpreter starts, which is worth knowing before raising
concurrency on such a suite.
"""

import builtins
import json
import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.evaluators.bounded import WorkerError, run_worker
from llm_eval_lab.models import (
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorSpec,
    JSONValue,
    ModelResponse,
    ResolvedCase,
)
from llm_eval_lab.utils.time import monotonic_ms

MAX_PATTERN_CHARS = 1000
"""Cap on an authored regex. Not a ReDoS defence on its own - see the module docstring."""

DEFAULT_MATCH_TIMEOUT_S = 2.0
"""Default wall-clock bound on one regex match. Capped by `settings.max_regex_match_s`."""

MAX_SUBJECT_CHARS = 1_000_000
"""Cap on the text handed to a regex worker, so the pipe payload stays bounded."""

MATCH_WORKER_SOURCE = """\
import json, re, sys

request = json.loads(sys.stdin.read())
try:
    pattern = re.compile(request["pattern"], request["flags"])
    match = (
        pattern.fullmatch(request["text"])
        if request["fullmatch"]
        else pattern.search(request["text"])
    )
except (re.error, RecursionError, MemoryError) as exc:
    json.dump({"error": type(exc).__name__, "message": str(exc)}, sys.stdout)
else:
    groups = None if match is None else [match.group(0), *list(match.groups())]
    json.dump({"matched": match is not None, "groups": groups}, sys.stdout)
"""
"""The whole regex worker program.

Deliberately imports nothing from this project: the child is started with ``-I``
and ``-S`` and only needs ``re`` and ``json``, which is what keeps its startup
cost at roughly 15 ms rather than the several hundred a full package import
would take.
"""

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_WHITESPACE_RUN = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str, *, strip: bool, normalize_whitespace: bool) -> str:
    """Apply the two documented text normalizations, in a fixed order.

    ``normalize_whitespace`` collapses every run of whitespace to a single
    space (which also turns a newline into a space); ``strip`` then removes
    leading and trailing whitespace. Collapsing first means `strip` has nothing
    left to do at the ends, so the two options compose predictably instead of
    depending on which was applied first.
    """
    result = text
    if normalize_whitespace:
        result = _WHITESPACE_RUN.sub(" ", result)
    if strip:
        result = result.strip()
    return result


def expected_as_text(case: ResolvedCase, override: str | None) -> str | None:
    """Return the expected answer as text, or ``None`` when there is not one.

    A parameter override wins over the case's ``expected`` field, so one case
    can carry several evaluators comparing against different strings. A
    non-scalar ``expected`` (a list or a mapping) returns ``None``: rendering it
    as a string would compare model output against a Python repr, which is a
    comparison nobody authored.
    """
    if override is not None:
        return override
    value = case.expected
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _output_or_error(
    evaluator: BaseEvaluator[Any],
    response: ModelResponse,
    started_ms: float,
) -> tuple[str | None, EvaluationResult | None]:
    """Return the response text, or the ERROR result to emit when there is none."""
    if response.output_text is None:
        return None, evaluator.error(
            kind="runtime",
            message="response carries no output text",
            error_code="no_output",
            duration_ms=monotonic_ms() - started_ms,
        )
    return response.output_text, None


def _missing_expected(
    evaluator: BaseEvaluator[Any],
    started_ms: float,
) -> EvaluationResult:
    """Build the ERROR result for a case with no usable expected value."""
    return evaluator.error(
        kind="config",
        message=(
            f"evaluator {evaluator.type!r} requires an expected value; the case has none "
            f"and no 'expected' parameter was supplied"
        ),
        error_code="missing_expected",
        duration_ms=monotonic_ms() - started_ms,
    )


# ---------------------------------------------------------------------------
# exact_match and exact_match_ci
# ---------------------------------------------------------------------------


class ExactMatchParams(BaseModel):
    """Parameters shared by `exact_match` and `exact_match_ci`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: str | None = Field(
        default=None,
        description="Overrides the case's `expected` field for this evaluator only.",
    )
    strip: bool = Field(default=True, description="Strip leading and trailing whitespace.")
    normalize_whitespace: bool = Field(
        default=False,
        description="Collapse every run of whitespace to a single space before comparing.",
    )


class ExactMatchEvaluator(BaseEvaluator[ExactMatchParams]):
    """Case-sensitive string equality after the configured normalization."""

    type: ClassVar[str] = "exact_match"
    params_model: ClassVar[builtins.type[BaseModel]] = ExactMatchParams
    needs_expected: ClassVar[bool] = True
    case_sensitive: ClassVar[bool] = True

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Compare the response text with the expected text for equality."""
        del context
        started = monotonic_ms()
        output, failure = _output_or_error(self, response, started)
        if failure is not None:
            return failure
        expected = expected_as_text(case, self.params.expected)
        if expected is None:
            return _missing_expected(self, started)

        actual = normalize_text(
            output or "",
            strip=self.params.strip,
            normalize_whitespace=self.params.normalize_whitespace,
        )
        target = normalize_text(
            expected,
            strip=self.params.strip,
            normalize_whitespace=self.params.normalize_whitespace,
        )
        if not self.case_sensitive:
            actual = actual.casefold()
            target = target.casefold()

        return self.verdict(
            passed=actual == target,
            metadata={"case_sensitive": self.case_sensitive},
            duration_ms=monotonic_ms() - started,
        )


class ExactMatchCaseInsensitiveEvaluator(ExactMatchEvaluator):
    """String equality ignoring case, via `str.casefold`."""

    type: ClassVar[str] = "exact_match_ci"
    params_model: ClassVar[builtins.type[BaseModel]] = ExactMatchParams
    needs_expected: ClassVar[bool] = True
    case_sensitive: ClassVar[bool] = False


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------


class ContainsParams(BaseModel):
    """Parameters for the `contains` evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: tuple[str, ...] = Field(
        min_length=1,
        description="Substrings to look for in the response text.",
    )
    mode: Literal["all", "any"] = Field(
        default="all",
        description="`all` requires every value to be present; `any` requires at least one.",
    )
    case_sensitive: bool = True


class ContainsEvaluator(BaseEvaluator[ContainsParams]):
    """Substring presence, scored as the fraction of values found.

    The score is the fraction regardless of `mode`, so a suite can see that a
    response matched two of three required strings rather than only that it
    failed.
    """

    type: ClassVar[str] = "contains"
    params_model: ClassVar[builtins.type[BaseModel]] = ContainsParams
    needs_expected: ClassVar[bool] = False

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Report which of the configured values appear in the response."""
        del case, context
        started = monotonic_ms()
        output, failure = _output_or_error(self, response, started)
        if failure is not None:
            return failure

        haystack = output or ""
        if self.params.case_sensitive:
            needles = list(self.params.values)
        else:
            haystack = haystack.casefold()
            needles = [value.casefold() for value in self.params.values]

        found = [needle in haystack for needle in needles]
        n_found = sum(found)
        score = n_found / len(found)
        passed = all(found) if self.params.mode == "all" else any(found)
        missing: JSONValue = [
            value for value, present in zip(self.params.values, found, strict=True) if not present
        ]

        return self.verdict(
            passed=passed,
            score=score,
            raw_score=float(n_found),
            metadata={"mode": self.params.mode, "n_found": n_found, "missing": missing},
            duration_ms=monotonic_ms() - started,
        )


# ---------------------------------------------------------------------------
# regex
# ---------------------------------------------------------------------------


class RegexParams(BaseModel):
    """Parameters for the `regex` evaluator.

    ``pattern`` is compiled by a validator, so an uncompilable pattern is
    rejected at benchmark load time rather than at match time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: str = Field(max_length=MAX_PATTERN_CHARS)
    must_match: bool = Field(
        default=True,
        description="True requires a match; False requires the absence of one.",
    )
    group: int | None = Field(
        default=None,
        ge=0,
        description="Capture group whose text is recorded in metadata['captured'].",
    )
    ignore_case: bool = False
    multiline: bool = False
    dotall: bool = False
    fullmatch: bool = Field(
        default=False,
        description="Anchor the pattern to the whole string instead of searching within it.",
    )
    match_timeout_s: float = Field(
        default=DEFAULT_MATCH_TIMEOUT_S,
        gt=0.0,
        le=60.0,
        description="Wall-clock bound on one match, further capped by settings.max_regex_match_s.",
    )

    @field_validator("pattern")
    @classmethod
    def _pattern_compiles(cls, value: str) -> str:
        """Reject a pattern `re` cannot compile, at load time."""
        try:
            re.compile(value)
        except re.error as exc:
            msg = f"invalid regular expression: {exc}"
            raise ValueError(msg) from exc
        return value

    @model_validator(mode="after")
    def _group_exists(self) -> Self:
        """Reject a `group` index the pattern does not have."""
        if self.group is None:
            return self
        compiled = re.compile(self.pattern)
        if self.group > compiled.groups:
            msg = (
                f"group {self.group} does not exist; the pattern has "
                f"{compiled.groups} capture group(s)"
            )
            raise ValueError(msg)
        return self

    def flag_bits(self) -> int:
        """Return the `re` flag bits the configured options add up to."""
        flags = re.NOFLAG
        if self.ignore_case:
            flags |= re.IGNORECASE
        if self.multiline:
            flags |= re.MULTILINE
        if self.dotall:
            flags |= re.DOTALL
        return int(flags)

    def compiled(self) -> re.Pattern[str]:
        """Return the compiled pattern with the configured flags applied.

        Used at LOAD time to validate the pattern. The match itself happens in a
        separate process, which compiles it there.
        """
        return re.compile(self.pattern, self.flag_bits())


@dataclass(frozen=True)
class MatchOutcome:
    """The picklable part of an `re.Match`: whether it matched, and its groups.

    An `re.Match` cannot cross a process boundary, and the evaluator only ever
    needs these two facts.
    """

    matched: bool
    groups: tuple[str | None, ...]

    def group(self, index: int) -> str | None:
        """Return the text of capture group `index`, where 0 is the whole match."""
        if index < 0 or index >= len(self.groups):
            return None
        return self.groups[index]


async def bounded_match(
    pattern: str,
    text: str,
    *,
    flags: int = 0,
    timeout_s: float,
    fullmatch: bool = False,
) -> MatchOutcome | None:
    """Run one regex match in a separate process, under a wall-clock bound.

    Returns ``None`` when the pattern did not match.

    **A process, and not a worker thread, because of the GIL.** CPython executes
    a regex match inside a single C call that never returns to the interpreter
    loop and never releases the GIL. A catastrophic pattern running in a thread
    therefore does not merely outlive its timeout - it starves every other thread
    in the process, including the event loop that was supposed to time it out, so
    the timeout never fires and the whole run hangs. That was measured: a main
    thread printing every 200 ms produced no output at all for 60 seconds while
    ``(a+)+$`` ran against 41 characters in a daemon thread. Killing a process is
    the only mechanism CPython offers that actually stops a match in progress.

    Raises:
        TimeoutError: when the match did not finish within `timeout_s`. The
            worker process is killed before this propagates.
        re.error: when the worker could not compile or run the pattern.
    """
    payload = json.dumps(
        {
            "pattern": pattern,
            "flags": flags,
            "text": text[:MAX_SUBJECT_CHARS],
            "fullmatch": fullmatch,
        }
    ).encode("utf-8")

    try:
        stdout = await run_worker(
            MATCH_WORKER_SOURCE,
            payload,
            timeout_s=timeout_s,
            stdlib_only=True,
        )
    except WorkerError as exc:
        # Re-raised as `re.error` so this function's contract is unchanged: its
        # one caller already treats a worker fault and a pattern fault the same
        # way, as `regex_error`.
        raise re.error(str(exc)) from exc

    result = json.loads(stdout.decode("utf-8"))
    if "error" in result:
        msg = f"{result['error']}: {result['message']}"
        raise re.error(msg)
    if not result["matched"]:
        return None
    return MatchOutcome(matched=True, groups=tuple(result["groups"]))


class RegexEvaluator(BaseEvaluator[RegexParams]):
    """Pattern matching against the response text, under a wall-clock bound."""

    type: ClassVar[str] = "regex"
    params_model: ClassVar[builtins.type[BaseModel]] = RegexParams
    needs_expected: ClassVar[bool] = False

    def __init__(self, spec: EvaluatorSpec, params: RegexParams) -> None:
        """Resolve the pattern's flag bits once, at construction, rather than per case."""
        super().__init__(spec, params)
        self._flags = params.flag_bits()

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Match the response text, honouring `must_match` polarity."""
        del case
        started = monotonic_ms()
        output, failure = _output_or_error(self, response, started)
        if failure is not None:
            return failure

        budget = min(self.params.match_timeout_s, context.settings.max_regex_match_s)
        try:
            matched = await bounded_match(
                self.params.pattern,
                output or "",
                flags=self._flags,
                timeout_s=budget,
                fullmatch=self.params.fullmatch,
            )
        except TimeoutError:
            return self.error(
                kind="timeout",
                message=f"regex match exceeded {budget}s and was abandoned",
                error_code="regex_timeout",
                metadata={"pattern": self.params.pattern, "timeout_s": budget},
                duration_ms=monotonic_ms() - started,
            )
        except (re.error, RecursionError, MemoryError) as exc:
            return self.error(
                kind="runtime",
                message=f"regex match failed: {exc}",
                error_code="regex_error",
                exception_type=type(exc).__name__,
                duration_ms=monotonic_ms() - started,
            )

        metadata: dict[str, JSONValue] = {
            "pattern": self.params.pattern,
            "must_match": self.params.must_match,
            "matched": matched is not None,
        }
        if matched is not None and self.params.group is not None:
            metadata["captured"] = matched.group(self.params.group)

        return self.verdict(
            passed=(matched is not None) is self.params.must_match,
            metadata=metadata,
            duration_ms=monotonic_ms() - started,
        )


# ---------------------------------------------------------------------------
# numeric_tolerance
# ---------------------------------------------------------------------------


class NumericToleranceParams(BaseModel):
    """Parameters for the `numeric_tolerance` evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: float | None = Field(
        default=None,
        description="Overrides the case's `expected` field for this evaluator only.",
    )
    abs_tol: float | None = Field(default=None, ge=0.0)
    rel_tol: float | None = Field(default=None, ge=0.0)
    extract: Literal["first", "last", "only"] = Field(
        default="first",
        description=(
            "Which number in the response to compare. `only` requires the response to "
            "contain exactly one number."
        ),
    )


class NumericToleranceEvaluator(BaseEvaluator[NumericToleranceParams]):
    """Numeric comparison with an absolute and/or relative tolerance.

    Tolerances follow :func:`math.isclose`: a value passes when EITHER
    configured tolerance is satisfied, and with neither configured the
    comparison is exact.

    A response with no number in it is a FAILED evaluation, not an ERROR. The
    evaluator worked perfectly and reached a conclusion; the answer was simply
    not a number.
    """

    type: ClassVar[str] = "numeric_tolerance"
    params_model: ClassVar[builtins.type[BaseModel]] = NumericToleranceParams
    needs_expected: ClassVar[bool] = True

    def _extract(self, text: str) -> tuple[float | None, int]:
        """Pull the configured number out of the response text."""
        found = _NUMBER_PATTERN.findall(text)
        numbers: list[float] = []
        for token in found:
            try:
                numbers.append(float(token))
            except ValueError:  # pragma: no cover - the pattern only matches parseable text
                continue
        if not numbers:
            return None, 0
        if self.params.extract == "last":
            return numbers[-1], len(numbers)
        if self.params.extract == "only":
            return (numbers[0] if len(numbers) == 1 else None), len(numbers)
        return numbers[0], len(numbers)

    def _expected_value(self, case: ResolvedCase) -> float | None:
        """Return the expected number from the parameters or the case."""
        if self.params.expected is not None:
            return self.params.expected
        value = case.expected
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Compare the extracted number against the expected value."""
        del context
        started = monotonic_ms()
        output, failure = _output_or_error(self, response, started)
        if failure is not None:
            return failure

        expected = self._expected_value(case)
        if expected is None:
            return _missing_expected(self, started)

        observed, n_numbers = self._extract(output or "")
        if observed is None:
            return self.verdict(
                passed=False,
                explanation=(
                    "no number found in the response"
                    if n_numbers == 0
                    else f"extract='only' but the response contains {n_numbers} numbers"
                ),
                metadata={
                    "error_code": "no_number_extracted",
                    "n_numbers": n_numbers,
                    "expected": expected,
                },
                duration_ms=monotonic_ms() - started,
            )

        passed = math.isclose(
            observed,
            expected,
            rel_tol=self.params.rel_tol or 0.0,
            abs_tol=self.params.abs_tol or 0.0,
        )
        return self.verdict(
            passed=passed,
            raw_score=observed,
            metadata={
                "observed": observed,
                "expected": expected,
                "abs_delta": abs(observed - expected),
                "n_numbers": n_numbers,
            },
            duration_ms=monotonic_ms() - started,
        )


DETERMINISTIC_EVALUATORS: tuple[type[BaseEvaluator[Any]], ...] = (
    ExactMatchEvaluator,
    ExactMatchCaseInsensitiveEvaluator,
    ContainsEvaluator,
    RegexEvaluator,
    NumericToleranceEvaluator,
)
"""Every evaluator this module defines, in registration order."""

__all__ = [
    "DEFAULT_MATCH_TIMEOUT_S",
    "DETERMINISTIC_EVALUATORS",
    "MAX_PATTERN_CHARS",
    "ContainsEvaluator",
    "ContainsParams",
    "EvaluationStatus",
    "ExactMatchCaseInsensitiveEvaluator",
    "ExactMatchEvaluator",
    "ExactMatchParams",
    "MatchOutcome",
    "NumericToleranceEvaluator",
    "NumericToleranceParams",
    "RegexEvaluator",
    "RegexParams",
    "bounded_match",
    "expected_as_text",
    "normalize_text",
]
