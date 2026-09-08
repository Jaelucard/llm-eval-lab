"""Strict parsing of a judge's response into a validated verdict.

**Strict, and deliberately so.** A judge that returned something this module
cannot validate has not produced a low score; it has produced no measurement at
all. Every rejection here becomes ``status=ERROR`` on the evaluation, never a
zero and never the scale midpoint. Coercing an unparseable judgment to a number
is how a broken judge starts looking like a failing model, and how one bad
response silently drags a benchmark mean down.

Four things are checked, in this order, and the first failure is reported with
enough detail to be sent back to the judge as a single repair instruction:

1. the text is JSON, after at most one wrapping markdown fence is removed;
2. it matches the structural schema for this configuration;
3. every configured criterion appears exactly once, and nothing else appears;
4. every score is within the declared scale, and an integer when the scale says
   integer, and every reported number is finite.

**Out-of-range scores are rejected, not clamped.** A judge answering 9 on a 1-5
scale did not mean 5. It either misread the scale or ignored it, and in both
cases the judgment is unusable. :meth:`~llm_eval_lab.models.JudgeScale.normalize`
clamps because a value that reaches it has already been validated here; nothing
out of range ever gets that far.

**The overall score is computed, not trusted.** When the rubric declares
weighted criteria, the raw score is the weight-normalized sum of the per-criterion
scores, calculated here. The judge's own ``overall_score`` is validated, recorded
and compared, but it does not decide the number - arithmetic is not something a
language model should be relied on for when the inputs are already in hand.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

# `jsonschema` ships no type stubs and `types-jsonschema` is not in the dev
# dependency group, which lives in the frozen `pyproject.toml`. Ignored here,
# narrowly and with a reason, rather than loosening the strict mypy settings.
import jsonschema  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

from llm_eval_lab.evaluators.json_evaluators import strip_markdown_fence
from llm_eval_lab.evaluators.judge.prompts import verdict_schema
from llm_eval_lab.models import JudgeConfig, JudgeVerdict

MAX_FAILURE_MESSAGE_CHARS = 400
"""Cap on a rejection message, which is sent back to the judge verbatim."""

OVERALL_CRITERION_ID = "overall"
"""The single criterion name used when a rubric declares none of its own."""

OVERALL_MISMATCH_TOLERANCE = 0.5
"""How far the judge's reported overall may drift from the computed one before it is noted.

Half a point on the raw scale. A drift beyond this is recorded in the result
metadata as ``overall_score_mismatch``; it is information about the judge's
arithmetic, not grounds for rejecting an otherwise valid judgment.
"""

FailureKind = Literal["json", "schema", "criteria", "range"]


@dataclass(frozen=True)
class VerdictParseFailure:
    """A judge response that could not be turned into a usable verdict."""

    kind: FailureKind
    message: str

    def repair_instruction(self) -> str:
        """Return the sentence handed to the judge for its one repair attempt."""
        text = self.message
        if len(text) > MAX_FAILURE_MESSAGE_CHARS:
            text = text[: MAX_FAILURE_MESSAGE_CHARS - 3] + "..."
        return text


@dataclass(frozen=True)
class ParsedVerdict:
    """A validated judge response, with the raw score this project derives from it."""

    verdict: JudgeVerdict
    raw_score: float
    reported_overall: float
    fenced: bool

    @property
    def overall_mismatch(self) -> float:
        """Return how far the judge's own overall drifted from the computed one."""
        return abs(self.reported_overall - self.raw_score)


type ParseOutcome = ParsedVerdict | VerdictParseFailure


def _in_scale(config: JudgeConfig, value: float) -> bool:
    """Report whether a raw score lies within the declared scale."""
    return config.scale.minimum <= value <= config.scale.maximum


def _scale_text(config: JudgeConfig) -> str:
    """Render the declared scale for a rejection message."""
    return f"{config.scale.minimum:g} to {config.scale.maximum:g}"


def _check_score(
    config: JudgeConfig,
    name: str,
    value: float,
    *,
    integral: bool,
) -> VerdictParseFailure | None:
    """Validate one score against the configured scale.

    ``integral`` is true for a score the judge chose from the anchors and false
    for the overall figure. An overall score is a WEIGHTED SUM of the criteria -
    4 and 5 at weights 0.6 and 0.4 is 4.4 - so requiring it to be a whole number
    would reject every correctly weighted verdict on an integer scale.
    """
    if not math.isfinite(value):
        return VerdictParseFailure(
            kind="range",
            message=f"the score for {name!r} is not a finite number.",
        )
    if not _in_scale(config, value):
        return VerdictParseFailure(
            kind="range",
            message=(
                f"the score for {name!r} was {value:g}, which is outside the scale "
                f"{_scale_text(config)}. Scores are not clamped: use only the declared scale."
            ),
        )
    if integral and config.scale.kind == "integer" and value != int(value):
        return VerdictParseFailure(
            kind="range",
            message=(
                f"the score for {name!r} was {value:g}, but this scale takes whole numbers "
                f"from {_scale_text(config)}."
            ),
        )
    return None


def _reported_criteria(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the criteria entries the judge reported, as a list of mappings."""
    entries = payload.get("criteria")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _collect_scores(
    config: JudgeConfig,
    payload: dict[str, Any],
) -> tuple[dict[str, float], VerdictParseFailure | None]:
    """Map criterion name to score, refusing a set that does not match the rubric."""
    scores: dict[str, float] = {}
    for entry in _reported_criteria(payload):
        name = entry["name"]
        if name in scores:
            return {}, VerdictParseFailure(
                kind="criteria",
                message=(
                    f"criterion {name!r} was scored more than once; "
                    f"score each criterion exactly once."
                ),
            )
        scores[name] = float(entry["score"])

    if not config.criteria:
        return scores, None

    declared = {criterion.id for criterion in config.criteria}
    missing = sorted(declared - set(scores))
    unexpected = sorted(set(scores) - declared)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing {', '.join(repr(name) for name in missing)}")
        if unexpected:
            parts.append(f"unexpected {', '.join(repr(name) for name in unexpected)}")
        return {}, VerdictParseFailure(
            kind="criteria",
            message=(
                f"the reported criteria do not match the rubric: {'; '.join(parts)}. "
                f"Score exactly these criteria: {', '.join(sorted(declared))}."
            ),
        )
    return scores, None


def _weighted_raw_score(config: JudgeConfig, scores: dict[str, float]) -> float:
    """Return the weight-normalized overall score across the declared criteria.

    Weights are normalized by their own sum rather than assumed to sum to one,
    so a rubric written with weights 3 and 1 means the same thing as one written
    with 0.75 and 0.25. A zero total weight is refused at benchmark load time.

    The rescaling is not silent: when the authored weights do not already sum to
    1.0, ``LlmJudgeEvaluator`` logs a warning and records
    ``metadata["weights_renormalised"]`` with the original sum, because weights
    of 0.6 and 0.5 are equally likely to be a considered choice and a typo.
    """
    total_weight = sum(criterion.weight for criterion in config.criteria)
    weighted = sum(criterion.weight * scores[criterion.id] for criterion in config.criteria)
    return weighted / total_weight


def parse_verdict(  # noqa: PLR0911 - one early return per distinct rejection reason
    text: str,
    config: JudgeConfig,
) -> ParseOutcome:
    """Turn one raw judge response into a validated verdict, or say why not.

    Args:
        text: The judge's response, exactly as the provider returned it.
        config: The configuration the judgment was requested under, which
            supplies the criteria, the scale and the expected output shape.

    Returns:
        A :class:`ParsedVerdict`, or a :class:`VerdictParseFailure` whose
        message is suitable for both the stored error and the single repair
        turn sent back to the judge.
    """
    payload_text, fenced = strip_markdown_fence(text)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return VerdictParseFailure(
            kind="json",
            message=(
                f"the response was not valid JSON ({exc.msg} at line {exc.lineno}, "
                f"column {exc.colno})."
            ),
        )
    if not isinstance(payload, dict):
        return VerdictParseFailure(
            kind="json",
            message=(
                f"the response parsed as a JSON {type(payload).__name__}, but a verdict must "
                f"be a JSON object."
            ),
        )

    try:
        jsonschema.validate(payload, verdict_schema(config))
    except ValidationError as exc:
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        return VerdictParseFailure(
            kind="schema",
            message=f"the JSON did not match the required shape at {location}: {exc.message}",
        )

    scores, failure = _collect_scores(config, payload)
    if failure is not None:
        return failure

    for name, value in sorted(scores.items()):
        problem = _check_score(config, name, value, integral=True)
        if problem is not None:
            return problem

    reported_overall = float(payload["overall_score"])
    problem = _check_score(config, "overall_score", reported_overall, integral=False)
    if problem is not None:
        return problem

    # With criteria configured the score is COMPUTED from them. With none
    # configured there is nothing authorised to compute from - any criteria the
    # judge invented are not in the rubric and carry no declared weights - so the
    # judge's own `overall_score` is the score. Averaging invented criteria was
    # the previous behaviour and was a rule nobody had stated; the entries are
    # still recorded in `per_criterion` for inspection.
    raw_score = _weighted_raw_score(config, scores) if config.criteria else reported_overall

    confidence = payload.get("confidence")
    if confidence is not None and not math.isfinite(float(confidence)):
        # `json.loads` accepts the non-standard literals NaN, Infinity and
        # -Infinity, and the schema's `minimum`/`maximum` bounds do not catch
        # NaN because every IEEE comparison against it is false. Without this
        # guard the value reached `JudgeVerdict`, whose `le=1` raised out of a
        # function documented never to raise - failing the whole CASE and losing
        # the other evaluators' results for it, rather than the judgment alone.
        return VerdictParseFailure(
            kind="range",
            message=(
                "the reported confidence is not a finite number; confidence must be a "
                "number between 0 and 1."
            ),
        )
    violations = payload.get("violations") or ()
    verdict = JudgeVerdict(
        score=raw_score,
        per_criterion=dict(scores) if scores else {OVERALL_CRITERION_ID: raw_score},
        reasoning=str(payload["rationale"]),
        violations=tuple(str(item) for item in violations),
        confidence=None if confidence is None else float(confidence),
    )
    return ParsedVerdict(
        verdict=verdict,
        raw_score=raw_score,
        reported_overall=reported_overall,
        fenced=fenced,
    )


__all__ = [
    "OVERALL_CRITERION_ID",
    "OVERALL_MISMATCH_TOLERANCE",
    "ParseOutcome",
    "ParsedVerdict",
    "VerdictParseFailure",
    "parse_verdict",
]
