"""The `llm_judge` evaluator: a model-graded measurement, treated as one.

A judge score is an INSTRUMENT READING, not ground truth, and every design
choice here follows from that.

**A failed judgment is never a score.** A provider error, a timeout, or a
response that survives neither parsing nor its one repair attempt produces
``status=ERROR``. It never produces 0.0, and it never produces the scale
midpoint. Both of those would be indistinguishable from a real judgment in every
aggregate the project computes.

**`passed` is set only when `pass_threshold` is configured.** Otherwise the
judge contributes ``score`` and leaves ``passed`` as ``None``, which the
aggregate counts under ``n_score_only`` and excludes from ``pass_rate``. That is
requirement R-JUDGE-09 made structural: a suite cannot accidentally read a
judge's opinion as a pass/fail fact, because the judge did not return one.
``pass_threshold`` is expressed on the RAW scale the rubric declares - 4 on a
1-5 scale - and a value outside that scale is a benchmark validation error, so
the "is 0.8 a normalized score or a raw one" ambiguity cannot arise.

**Repeats and panels are aggregated by median.** One wild judgment moves a
median by a fraction of a point and a mean by a third of the range, and a median
needs no outlier-detection logic to explain. Spread is always reported beside
the number: ``dispersion`` is the range across every judgment collected, and
``high_judge_variance`` flags a spread wider than the configured threshold. A
panel reports min, median and max across judges rather than one blended figure.

**Bias controls.** Temperature defaults to 0. The candidate output is wrapped in
digest-derived delimiters with a system-level instruction to treat it as data
(see :mod:`~llm_eval_lab.evaluators.judge.prompts`). Candidate output length is
recorded next to the score so verbosity-correlated inflation is visible in the
data. When a judge's provider and model equal the candidate's,
``self_preference_risk`` is set: visible, never silent, never blocking.

**Judge spend is kept apart.** The tokens every judge call consumed are summed
into ``JudgeProvenance.usage`` and never added to the candidate response's
usage, which is what keeps ``CostBreakdown.total_cost`` meaning candidate-model
cost alone. ``JudgeProvenance.cost_usd`` is left ``None`` here on purpose: the
layered contract forbids ``evaluators`` importing ``pricing``, so pricing the
recorded usage is the runner's job, against the same price table the rest of the
run uses.
"""

import asyncio
import builtins
import re
import statistics
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, model_validator

from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.evaluators.deterministic import expected_as_text
from llm_eval_lab.evaluators.judge.parsing import (
    OVERALL_MISMATCH_TOLERANCE,
    ParsedVerdict,
    VerdictParseFailure,
    parse_verdict,
)
from llm_eval_lab.evaluators.judge.prompts import (
    TEMPLATE_IDS,
    RenderedPrompt,
    render_prompt,
    render_repair,
    rubric_hash,
    truncate_echo,
)
from llm_eval_lab.models import (
    ChatMessage,
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorConfigError,
    EvaluatorSpec,
    JSONValue,
    JudgeConfig,
    JudgeProvenance,
    JudgeVerdict,
    ModelResponse,
    Provider,
    ProviderConfig,
    ProviderError,
    ProviderRequest,
    ResolvedCase,
    TokenUsage,
)
from llm_eval_lab.redaction import looks_like_url, redact_url
from llm_eval_lab.utils.time import monotonic_ms

_SEMAPHORES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = (
    weakref.WeakKeyDictionary()
)
"""One judge semaphore per event loop, keyed weakly so a finished loop is collectable.

Module state rather than evaluator state because the runner constructs a fresh
evaluator per case: a per-instance semaphore would bound one case's judge calls
against themselves and nothing else, which is not a bound at all.
"""


WEIGHT_SUM_TOLERANCE = 1e-9
"""How far criterion weights may sum from 1.0 before the rescaling is announced."""

_WHITESPACE_SPLIT = re.compile(r"(\s+|\\[nrt])")
r"""Splits text into words and the whitespace between them, KEEPING the separators.

The capturing group is what makes the split reversible: `"".join(parts)` returns
the original string, so redacting one part cannot disturb the layout of a
multi-line prompt.

The second alternative covers the ESCAPED spellings of the same whitespace.
``raw_outputs`` holds a judge's response verbatim, and that response is JSON, so
a newline inside its ``rationale`` arrives as the two characters ``\n`` rather
than as a line break. Without this, a connection string on its own line of a
judge's reasoning stayed welded to ``cited\npostgresql://...``, which has no
valid scheme, so ``redact_url`` found nothing to redact and the credential was
persisted. Treating the escape as the separator it represents is the same rule,
applied to the same whitespace, in the encoding it actually arrives in.
"""


def persistable_text(text: str, *, max_chars: int) -> str:
    r"""Return one judge string in the form that is safe to store, capped and scrubbed.

    Two problems, one function, applied at the boundary where text stops being
    working memory and becomes a database row.

    **Size.** ``raw_outputs`` and ``rendered_prompts`` grow with
    ``repetitions x len(judges)``, and each rendered prompt already contains the
    whole rubric and the whole candidate answer. Unlike a provider's ``raw``
    payload, which the storage mapper caps, these rode into the database
    unbounded. They are capped at ``max_output_chars`` - the same bound the
    candidate answer is shown to the judge under - with the truncation stated in
    the text rather than left to be inferred from a suspiciously round length.

    **Credentials.** A judge prompt is assembled from benchmark text and model
    output, and a connection string or a signed URL in either would otherwise be
    persisted verbatim. Every URL-shaped token goes through the project's one
    URL redaction implementation, which strips userinfo and secret-shaped query
    parameters using the same secret-name list the logging and persistence
    scrubbers use. It is defence in depth, not the primary control: vendor error
    text is already scrubbed where it is raised, and ``JudgeProvenance`` carries
    no ``ProviderConfig``.

    **Splitting on ALL whitespace, escaped or not, with the separators kept.**
    Splitting on the space character alone missed every URL that touched a
    newline or a tab, which is most of them: a rendered judge prompt is a
    multi-line document, so a connection string on its own line arrived at
    ``redact_url`` still welded to the surrounding newlines and did not look like
    a URL. A judge's raw response has the same problem one encoding further out,
    where a line break inside its JSON ``rationale`` is the two characters
    ``\n``. :data:`_WHITESPACE_SPLIT` covers both spellings and captures its
    separator, so rejoining reproduces the original exactly - every newline, tab,
    escape and run of spaces - and text with nothing to redact comes back
    byte-identical.

    Truncation happens AFTER redaction, so a credential straddling the cut is
    still redacted rather than half-persisted.
    """
    scrubbed = "".join(
        redact_url(part) if looks_like_url(part) else part for part in _WHITESPACE_SPLIT.split(text)
    )
    if len(scrubbed) <= max_chars:
        return scrubbed
    return f"{scrubbed[:max_chars]}... [truncated: {len(scrubbed)} characters]"


def judge_semaphore(limit: int) -> asyncio.Semaphore:
    """Return the semaphore bounding concurrent judge calls on this event loop.

    Rebuilt when the configured limit changes, which happens only between runs
    in the same process. Waiters on the previous semaphore are unaffected: they
    hold their permit until they release it, and the change takes effect for
    calls that start afterwards.
    """
    loop = asyncio.get_running_loop()
    entry = _SEMAPHORES.get(loop)
    if entry is None or entry[0] != limit:
        semaphore = asyncio.Semaphore(limit)
        _SEMAPHORES[loop] = (limit, semaphore)
        return semaphore
    return entry[1]


class JudgeParams(JudgeConfig):
    """A :class:`~llm_eval_lab.models.JudgeConfig` with its load-time refusals.

    The refusals live here rather than on the contract because they are the
    EVALUATOR's requirements: the contract type is also used to describe a
    judgment that has already happened, where a missing provider is a fact
    rather than a fault. Each one turns a configuration that could only fail at
    run time - on every case, identically - into a benchmark validation error
    naming the case and the field.
    """

    @model_validator(mode="after")
    def _configuration_is_usable(self) -> Self:
        """Refuse a judge configuration that cannot produce an interpretable score."""
        if self.provider is None and not self.judges:
            msg = (
                "llm_judge requires a judge model: set 'provider' for a single judge, or "
                "'judges' for a panel. There is no default judge, because a score from an "
                "unnamed model cannot be interpreted or reproduced."
            )
            raise ValueError(msg)
        if self.template_id not in TEMPLATE_IDS:
            msg = (
                f"unknown judge template {self.template_id!r}; known templates are "
                f"{', '.join(sorted(TEMPLATE_IDS))}"
            )
            raise ValueError(msg)
        if self.criteria:
            ids = [criterion.id for criterion in self.criteria]
            duplicates = sorted({name for name in ids if ids.count(name) > 1})
            if duplicates:
                msg = f"duplicate criterion id(s): {', '.join(duplicates)}"
                raise ValueError(msg)
            if sum(criterion.weight for criterion in self.criteria) <= 0.0:
                msg = (
                    "criterion weights sum to zero, so no criterion could contribute to the "
                    "overall score. Give at least one criterion a positive weight."
                )
                raise ValueError(msg)
        if self.pass_threshold is not None and not (
            self.scale.minimum <= self.pass_threshold <= self.scale.maximum
        ):
            msg = (
                f"pass_threshold {self.pass_threshold:g} is outside the declared scale "
                f"{self.scale.minimum:g} to {self.scale.maximum:g}. It is expressed on the "
                f"RAW scale the rubric uses, not on the normalized 0..1 score."
            )
            raise ValueError(msg)
        return self


@dataclass
class JudgeCall:
    """Everything one judge call produced, successful or not."""

    judge_model: str
    repeat: int
    prompt: str
    raw_output: str
    usage: TokenUsage
    latency_ms: float
    parsed: ParsedVerdict | None = None
    failure: VerdictParseFailure | None = None
    provider_error: str | None = None
    repaired: bool = False
    terminal: Literal["parse", "provider"] | None = None
    """What actually ended this call, when it ended badly.

    Recorded explicitly rather than inferred from which fields are populated. A
    repair attempt that dies on a provider error carries BOTH a parse failure
    (the first response) and a provider error (the second), and inferring the
    cause from field presence labelled that outage as a parsing problem.
    """


@dataclass
class JudgeOutcome:
    """The collected judgments of one case, before they become a result."""

    calls: list[JudgeCall] = field(default_factory=list)
    timed_out: bool = False

    @property
    def parsed(self) -> list[JudgeCall]:
        """Return the calls that produced a usable verdict."""
        return [call for call in self.calls if call.parsed is not None]

    @property
    def parse_failures(self) -> int:
        """Return how many calls ended without a usable verdict."""
        return sum(1 for call in self.calls if call.parsed is None)


def _model_id(config: ProviderConfig) -> str:
    """Return the `provider:model` identity recorded for a judge."""
    return f"{config.provider}:{config.model}"


def _total(values: Sequence[int | None]) -> int | None:
    """Total a column of possibly-absent token counts, or report it as absent."""
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def sum_usage(usages: Sequence[TokenUsage]) -> TokenUsage:
    """Sum token usage across judge calls, keeping absent counts absent.

    A total built only from the calls that reported usage would understate the
    spend without saying so, so a single unreported half makes that half
    ``None`` for the whole judgment: "partly unknown" is reported as unknown
    rather than as a smaller number that looks exact.
    """
    return TokenUsage(
        input_tokens=_total([usage.input_tokens for usage in usages]),
        output_tokens=_total([usage.output_tokens for usage in usages]),
    )


def _aggregate(values: Sequence[float], how: str) -> float:
    """Reduce one judge's repeated raw scores to a single number."""
    ordered = sorted(values)
    if how == "mean":
        return statistics.fmean(ordered)
    if how == "min":
        return ordered[0]
    if how == "majority":
        # Ties break towards the LOWER score: a judge split between 3 and 4 is
        # not evidence for 4, and a deterministic tie-break beats whichever
        # value a counter happened to see first.
        counts = {value: ordered.count(value) for value in ordered}
        best = max(counts.values())
        return min(value for value, count in counts.items() if count == best)
    return statistics.median(ordered)


class LlmJudgeEvaluator(BaseEvaluator[JudgeParams]):
    """Model-graded scoring against a weighted rubric, with full provenance."""

    type: ClassVar[str] = "llm_judge"
    params_model: ClassVar[builtins.type[BaseModel]] = JudgeParams
    needs_expected: ClassVar[bool] = False
    is_model_graded: ClassVar[bool] = True

    # -- configuration ---------------------------------------------------

    def _judges(self) -> tuple[ProviderConfig, ...]:
        """Return the judge panel, which the params validator guarantees is non-empty."""
        if self.params.judges:
            return self.params.judges
        if self.params.provider is None:  # pragma: no cover - refused by JudgeParams
            msg = "llm_judge has no judge model configured"
            raise EvaluatorConfigError(msg)
        return (self.params.provider,)

    def __init__(self, spec: EvaluatorSpec, params: JudgeParams) -> None:
        """Bind the configuration and reset the once-per-instance warning latch."""
        super().__init__(spec, params)
        self._warned_about_weights = False

    def _weight_total(self) -> float | None:
        """Return the criterion weight sum when it is not already 1.0, else None."""
        if not self.params.criteria:
            return None
        total = sum(criterion.weight for criterion in self.params.criteria)
        return None if abs(total - 1.0) <= WEIGHT_SUM_TOLERANCE else total

    def _warn_if_weights_renormalised(self, context: EvaluationContext) -> None:
        """Say out loud that the authored weights were rescaled, once.

        The parser divides by the weight sum, so weights of 0.6 and 0.5 silently
        become 0.545 and 0.455 - which is what the author probably meant, and is
        also exactly what a mistyped weight looks like. Normalising without
        saying so leaves a typo indistinguishable from an intent.

        The warning is about the CONFIGURATION, not about any one response, so
        emitting it per case would put the same line in the log a thousand times
        for a thousand-case suite and bury everything else. It fires once per
        evaluator instance. The per-case record stays per case:
        ``metadata["weights_renormalised"]`` and ``metadata["criterion_weight_sum"]``
        are written on every result, because a log line is gone by the time
        anyone reads the run and the stored data is not.

        A caveat worth stating rather than assuming: the runner builds a fresh
        evaluator per case, so today this is one line per case in a real run.
        The flag is what makes it one line for any caller that reuses an
        instance, and what stops a future batching change from regressing.
        """
        total = self._weight_total()
        if total is None or self._warned_about_weights:
            return
        self._warned_about_weights = True
        context.logger.warning(
            "judge_criterion_weights_renormalised",
            evaluator_id=self.evaluator_id,
            weight_sum=total,
            criteria=[criterion.id for criterion in self.params.criteria],
        )

    def _reference_for(self, case: ResolvedCase) -> str | None:
        """Return the reference answer shown to the judge, if any."""
        if not self.params.use_reference:
            return None
        if self.params.reference is not None:
            return self.params.reference
        return expected_as_text(case, None)

    # -- one judge call --------------------------------------------------

    async def _generate(
        self,
        provider: Provider,
        messages: tuple[ChatMessage, ...],
        prompt: RenderedPrompt,
        request_id: str,
    ) -> tuple[ModelResponse | None, str | None]:
        """Issue one judge request, returning the response or a scrubbed error."""
        request = ProviderRequest(
            messages=messages,
            system=prompt.system,
            params=self.params.params,
            request_id=request_id,
        )
        try:
            return await provider.generate(request), None
        except ProviderError as exc:
            return None, f"{type(exc).__name__}: {exc}"

    async def _one_call(
        self,
        provider: Provider,
        *,
        judge_model: str,
        prompt: RenderedPrompt,
        repeat: int,
        request_prefix: str,
    ) -> JudgeCall:
        """Run one judgment: one call, then at most one repair attempt.

        Exactly one repair, and no more. A judge that cannot produce the
        required shape twice is not going to produce it on the third try, and a
        longer loop spends real money converging on a number that was never
        measured.
        """
        started = monotonic_ms()
        messages = (ChatMessage(role="user", content=prompt.user),)
        call = JudgeCall(
            judge_model=judge_model,
            repeat=repeat,
            prompt=prompt.full_text,
            raw_output="",
            usage=TokenUsage(),
            latency_ms=0.0,
        )

        response, error = await self._generate(
            provider, messages, prompt, f"{request_prefix}:{repeat}:1"
        )
        if response is None:
            call.provider_error = error
            call.terminal = "provider"
            call.latency_ms = monotonic_ms() - started
            return call

        call.raw_output = response.output_text or ""
        call.usage = response.usage
        outcome = parse_verdict(call.raw_output, self.params)
        if isinstance(outcome, ParsedVerdict):
            call.parsed = outcome
            call.latency_ms = monotonic_ms() - started
            return call

        repair_messages = (
            *messages,
            ChatMessage(role="assistant", content=truncate_echo(call.raw_output)),
            ChatMessage(role="user", content=render_repair(outcome.repair_instruction())),
        )
        call.repaired = True
        repaired, error = await self._generate(
            provider, repair_messages, prompt, f"{request_prefix}:{repeat}:2"
        )
        if repaired is None:
            call.failure = outcome
            call.provider_error = error
            call.terminal = "provider"
            call.latency_ms = monotonic_ms() - started
            return call

        call.raw_output = f"{call.raw_output}\n---REPAIR ATTEMPT---\n{repaired.output_text or ''}"
        call.usage = sum_usage([call.usage, repaired.usage])
        second = parse_verdict(repaired.output_text or "", self.params)
        if isinstance(second, ParsedVerdict):
            call.parsed = second
        else:
            call.failure = second
            call.terminal = "parse"
        call.latency_ms = monotonic_ms() - started
        return call

    async def _collect(
        self,
        context: EvaluationContext,
        prompt: RenderedPrompt,
        case_id: str,
        outcome: JudgeOutcome,
    ) -> None:
        """Run every judge for every repetition, under the judge concurrency bound.

        The outcome is OWNED BY THE CALLER and appended to as each call finishes,
        rather than built here and returned at the end. That is what makes a
        deadline expiry survivable: cancellation unwinds out of this coroutine
        with no return value, and a returned-at-the-end collection would take
        every completed judgment - its raw output, its rendered prompt, the
        tokens it was billed for - down with it. The caller still holds the
        evidence for everything that finished before the clock ran out.

        The semaphore is acquired BEFORE the provider is opened, so a judge call
        queued behind the concurrency bound is not holding a client open while it
        waits. One permit covers one judge's whole repeat sequence, and those
        repeats run one at a time, so the bound on simultaneous judge calls is
        still exactly `judge_concurrency`.
        """
        semaphore = judge_semaphore(context.settings.judge_concurrency)
        for index, config in enumerate(self._judges()):
            judge_model = _model_id(config)
            prefix = f"{context.run_id}:{case_id}:judge{index}"
            async with semaphore, context.provider_factory(config) as provider:
                for repeat in range(self.params.repetitions):
                    outcome.calls.append(
                        await self._one_call(
                            provider,
                            judge_model=judge_model,
                            prompt=prompt,
                            repeat=repeat,
                            request_prefix=prefix,
                        )
                    )

    # -- provenance and result -------------------------------------------

    def _provenance(  # noqa: PLR0913 - one parameter per recorded field
        self,
        outcome: JudgeOutcome,
        *,
        max_chars: int,
        dispersion: float | None,
        agreement: float | None,
        self_preference_risk: bool,
        high_variance: bool,
    ) -> JudgeProvenance:
        """Assemble everything needed to audit or reproduce this judgment.

        The raw outputs and rendered prompts are capped and scrubbed HERE rather
        than in memory, because ``_one_call`` needs the full untouched response
        to build its repair turn from. This is the boundary where the text stops
        being working memory and becomes a persisted row.
        """
        verdicts: tuple[JudgeVerdict, ...] = tuple(
            call.parsed.verdict for call in outcome.parsed if call.parsed is not None
        )
        return JudgeProvenance(
            template_id=self.params.template_id,
            rubric_hash=rubric_hash(self.params),
            scale=self.params.scale,
            aggregation=self.params.aggregation,
            verdicts=verdicts,
            judge_models=tuple(dict.fromkeys(call.judge_model for call in outcome.calls)),
            raw_outputs=tuple(
                persistable_text(call.raw_output, max_chars=max_chars) for call in outcome.calls
            ),
            rendered_prompts=tuple(
                persistable_text(prompt, max_chars=max_chars)
                for prompt in dict.fromkeys(call.prompt for call in outcome.calls)
            ),
            usage=sum_usage([call.usage for call in outcome.calls]),
            # Pricing lives above this layer; the runner prices `usage` against
            # the run's own price table and records it as judge_cost.
            cost_usd=None,
            dispersion=dispersion,
            agreement=agreement,
            parse_failures=outcome.parse_failures,
            self_preference_risk=self_preference_risk,
            high_judge_variance=high_variance,
        )

    def _call_evidence(self, outcome: JudgeOutcome) -> dict[str, JSONValue]:
        """Return the per-call facts every judgment records, scored or failed.

        ``JudgeProvenance`` is a frozen contract and cannot grow fields, so the
        per-call detail lives in ``metadata``. The three lists are index-aligned
        with ``JudgeProvenance.raw_outputs``, which is what lets a stored verdict
        in a three-judge panel with repeats be attributed to the judge and the
        repetition that produced it.
        """
        return {
            "judge_call_models": [call.judge_model for call in outcome.calls],
            "judge_call_repeats": [call.repeat for call in outcome.calls],
            "judge_call_latency_ms": [call.latency_ms for call in outcome.calls],
            # Dumped WITHOUT `exclude_none`: "seed was not set" and "seed is not
            # recorded" are different facts, and only the first is true here.
            "judge_params": self.params.params.model_dump(mode="json"),
            "weights_renormalised": self._weight_total() is not None,
            "criterion_weight_sum": self._weight_total(),
        }

    def _failure_result(
        self,
        outcome: JudgeOutcome,
        *,
        prompt: RenderedPrompt,
        max_chars: int,
        started: float,
        self_preference_risk: bool,
    ) -> EvaluationResult:
        """Build the ERROR result for a judgment that produced no usable verdict."""
        provider_errors = [call.provider_error for call in outcome.calls if call.provider_error]
        parse_errors = [call.failure.message for call in outcome.calls if call.failure is not None]
        terminal = next(
            (call.terminal for call in reversed(outcome.calls) if call.terminal is not None),
            None,
        )
        if outcome.timed_out:
            kind, message, code = (
                "timeout",
                "the judge did not answer within the case's time budget",
                "judge_timeout",
            )
        elif terminal == "provider" and provider_errors:
            # The terminal cause wins over the first one. A repair attempt that
            # died on a provider error is an outage, and labelling it a parsing
            # problem sends whoever is debugging it to the wrong place.
            kind, message, code = (
                "provider",
                f"the judge call failed: {provider_errors[-1]}",
                "judge_provider_error",
            )
        elif parse_errors:
            kind, message, code = (
                "parse",
                (
                    f"the judge's response could not be validated after one repair attempt: "
                    f"{parse_errors[-1]}"
                ),
                "judge_unparseable",
            )
        elif provider_errors:
            kind, message, code = (
                "provider",
                f"every judge call failed: {provider_errors[-1]}",
                "judge_provider_error",
            )
        else:
            kind, message, code = (
                "runtime",
                "the judge produced no judgments at all",
                "judge_no_judgments",
            )
        result = self.error(
            kind=kind,
            message=message,
            error_code=code,
            metadata={
                "template_id": self.params.template_id,
                "n_calls": len(outcome.calls),
                "n_judgments": len(outcome.parsed),
                "parse_failures": outcome.parse_failures,
                "self_preference_risk": self_preference_risk,
                "candidate_chars": prompt.candidate_chars,
                "candidate_truncated": prompt.candidate_truncated,
                **self._call_evidence(outcome),
            },
            duration_ms=monotonic_ms() - started,
        )
        provenance = self._provenance(
            outcome,
            max_chars=max_chars,
            dispersion=None,
            agreement=None,
            self_preference_risk=self_preference_risk,
            high_variance=False,
        )
        return result.model_copy(update={"judge": provenance})

    # -- entry point ------------------------------------------------------

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Grade one response against the rubric and record how it was graded."""
        started = monotonic_ms()
        if response.output_text is None:
            return self.error(
                kind="runtime",
                message="response carries no output text",
                error_code="no_output",
                duration_ms=monotonic_ms() - started,
            )

        self_preference_risk = any(
            config.provider == response.provider and config.model == response.model
            for config in self._judges()
        )
        # One cap, used for what the judge is SHOWN and for what is STORED: a
        # suite that bounded the candidate answer to N characters did not agree
        # to persist a prompt many times that size.
        max_chars = min(self.params.max_output_chars, context.settings.max_output_chars)
        prompt = render_prompt(
            self.params,
            question=tuple(
                message.content if message.role == "user" else f"[{message.role}] {message.content}"
                for message in case.messages
            ),
            candidate=response.output_text,
            reference=self._reference_for(case),
            max_candidate_chars=max_chars,
        )

        self._warn_if_weights_renormalised(context)

        outcome = JudgeOutcome()
        try:
            if context.deadline is None:
                await self._collect(context, prompt, case.id, outcome)
            else:
                async with asyncio.timeout_at(context.deadline):
                    await self._collect(context, prompt, case.id, outcome)
        except TimeoutError:
            # A timed-out judgment is an ERROR even when some calls succeeded.
            # Scoring a panel that was cut off partway through would report a
            # number over a different set of judges than the suite asked for,
            # and nothing downstream would say so. The evidence for the calls
            # that DID finish is attached either way.
            outcome.timed_out = True
            return self._failure_result(
                outcome,
                prompt=prompt,
                max_chars=max_chars,
                started=started,
                self_preference_risk=self_preference_risk,
            )

        if not outcome.parsed:
            return self._failure_result(
                outcome,
                prompt=prompt,
                max_chars=max_chars,
                started=started,
                self_preference_risk=self_preference_risk,
            )
        return self._score(
            outcome,
            prompt=prompt,
            max_chars=max_chars,
            started=started,
            self_preference_risk=self_preference_risk,
        )

    def _score(
        self,
        outcome: JudgeOutcome,
        *,
        prompt: RenderedPrompt,
        max_chars: int,
        started: float,
        self_preference_risk: bool,
    ) -> EvaluationResult:
        """Aggregate the collected judgments into one scored result."""
        by_judge: dict[str, list[float]] = {}
        for call in outcome.parsed:
            if call.parsed is not None:
                by_judge.setdefault(call.judge_model, []).append(call.parsed.raw_score)

        per_judge = {
            judge: _aggregate(scores, self.params.aggregation)
            for judge, scores in sorted(by_judge.items())
        }
        raw_all = sorted(score for scores in by_judge.values() for score in scores)
        # Across a panel the reported number is the median of the judges' own
        # aggregates, never a mean of everything: one talkative judge with three
        # repeats must not outvote two judges with one each.
        raw_score = statistics.median(sorted(per_judge.values()))
        dispersion = raw_all[-1] - raw_all[0]
        judge_range = max(per_judge.values()) - min(per_judge.values())
        span = self.params.scale.maximum - self.params.scale.minimum
        agreement = min(1.0, max(0.0, 1.0 - dispersion / span))
        # Two different questions, so two different flags. `high_judge_variance`
        # is the range across EVERY judgment collected, which mixes one judge's
        # instability across repeats with genuine disagreement between judges.
        # `judge_disagreement` is only about the judges' own aggregates, and is
        # meaningless with a single judge, so it is false there by construction.
        high_variance = dispersion > self.params.high_variance_threshold
        judge_disagreement = (
            len(per_judge) > 1 and judge_range > self.params.high_variance_threshold
        )

        normalized = self.params.scale.normalize(raw_score)
        provenance = self._provenance(
            outcome,
            max_chars=max_chars,
            dispersion=dispersion,
            agreement=agreement,
            self_preference_risk=self_preference_risk,
            high_variance=high_variance,
        )
        metadata = self._metadata(
            outcome,
            prompt=prompt,
            per_judge=per_judge,
            raw_all=raw_all,
            raw_score=raw_score,
            dispersion=dispersion,
            judge_range=judge_range,
            agreement=agreement,
            high_variance=high_variance,
            judge_disagreement=judge_disagreement,
            self_preference_risk=self_preference_risk,
        )
        explanation = self._explanation(outcome, raw_score=raw_score, dispersion=dispersion)
        duration_ms = monotonic_ms() - started

        if self.params.pass_threshold is None:
            # Score-only: `status=PASSED` with `passed=None` is the one shape
            # `n_score_only` counts, and it is what keeps a judge's opinion out
            # of the pass rate.
            result = self.result(
                status=EvaluationStatus.PASSED,
                passed=None,
                score=normalized,
                raw_score=raw_score,
                explanation=explanation,
                metadata=metadata,
                duration_ms=duration_ms,
            )
        else:
            result = self.verdict(
                passed=raw_score >= self.params.pass_threshold,
                score=normalized,
                raw_score=raw_score,
                explanation=explanation,
                metadata=metadata,
                duration_ms=duration_ms,
            )
        return result.model_copy(update={"judge": provenance})

    def _metadata(  # noqa: PLR0913 - one parameter per recorded field
        self,
        outcome: JudgeOutcome,
        *,
        prompt: RenderedPrompt,
        per_judge: dict[str, float],
        raw_all: Sequence[float],
        raw_score: float,
        dispersion: float,
        judge_range: float,
        agreement: float,
        high_variance: bool,
        judge_disagreement: bool,
        self_preference_risk: bool,
    ) -> dict[str, JSONValue]:
        """Assemble the inspectable facts that sit beside the score."""
        mismatches = [
            call.parsed.overall_mismatch
            for call in outcome.parsed
            if call.parsed is not None and call.parsed.overall_mismatch > OVERALL_MISMATCH_TOLERANCE
        ]
        return {
            "template_id": self.params.template_id,
            "rubric_hash": rubric_hash(self.params),
            "aggregation": self.params.aggregation,
            "repetitions": self.params.repetitions,
            "raw_scores": list(raw_all),
            "per_judge_score": dict(per_judge),
            "judge_min": min(per_judge.values()),
            "judge_median": raw_score,
            "judge_max": max(per_judge.values()),
            "judge_median_range": judge_range,
            "dispersion": dispersion,
            "agreement": agreement,
            "high_judge_variance": high_variance,
            "judge_disagreement": judge_disagreement,
            "n_judges": len(per_judge),
            "self_preference_risk": self_preference_risk,
            "parse_failures": outcome.parse_failures,
            "repairs_attempted": sum(1 for call in outcome.calls if call.repaired),
            # Logged so verbosity-correlated score inflation is visible in the
            # data even though nothing here corrects for it.
            "candidate_chars": prompt.candidate_chars,
            "candidate_truncated": prompt.candidate_truncated,
            "temperature": self.params.params.temperature,
            "overall_score_mismatches": len(mismatches),
            "pass_threshold": self.params.pass_threshold,
            **self._call_evidence(outcome),
        }

    def _explanation(
        self,
        outcome: JudgeOutcome,
        *,
        raw_score: float,
        dispersion: float,
    ) -> str:
        """Render the one-paragraph summary a reader sees first."""
        reasons = [
            call.parsed.verdict.reasoning for call in outcome.parsed if call.parsed is not None
        ]
        head = (
            f"Judged {raw_score:.2f} on a {self.params.scale.minimum:g}-"
            f"{self.params.scale.maximum:g} scale by "
            f"{len(outcome.parsed)} judgment(s), spread {dispersion:.2f}. "
            f"This is a model's opinion, not ground truth."
        )
        return f"{head} {reasons[0]}" if reasons else head


JUDGE_EVALUATORS: tuple[type[BaseEvaluator[Any]], ...] = (LlmJudgeEvaluator,)
"""Every evaluator this module defines, in registration order."""

__all__ = [
    "JUDGE_EVALUATORS",
    "WEIGHT_SUM_TOLERANCE",
    "JudgeCall",
    "JudgeOutcome",
    "JudgeParams",
    "LlmJudgeEvaluator",
    "judge_semaphore",
    "persistable_text",
]
