"""The deterministic fake provider: the whole test suite's model backend.

Determinism is the entire point. Every stochastic quantity - the generated
text, the simulated latency, the probabilistic failure draw - comes from one
``blake2b`` digest over the request:

``f"{seed}|{model}|{request_id}|{canonical_json(messages)}|{system}|{canonical_json(params)}"``

which seeds a LOCAL :class:`random.Random`. The global RNG is never touched, so
nothing this provider does can perturb, or be perturbed by, anything else in
the process. The output therefore depends on the inputs alone: not on
scheduling order, not on concurrency level, not on the wall clock. Running the
same suite at ``concurrency=1`` and at ``concurrency=32`` produces byte-identical
output, which is what lets the rest of the project be tested at all.

Five modes.

``derived``   Deterministic pseudo-text built from the digest.
``echo``      Returns the last user message verbatim.
``expected``  Returns the case's expected answer, giving a guaranteed all-pass
              baseline. This is what the end-to-end smoke run uses.
``mutate``    Returns the expected answer for most cases and a degraded answer
              for a deterministic subset. This is how the intentional-regression
              fixture is built in a later slice.
``scripted``  Returns a canned answer per case id.

``finish_reason`` and ``truncated`` are separate options rather than modes, so a
fixture can produce a cut-off answer in any mode.

Failure injection raises real :class:`~llm_eval_lab.models.ProviderError`
instances rather than returning error-shaped responses, so the runner's retry,
backoff and rate-limit paths are genuinely exercised rather than simulated.
"""

import asyncio
import hashlib
import math
import random
from collections.abc import Sequence
from typing import ClassVar, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_eval_lab.models import (
    FinishReason,
    JSONValue,
    ModelResponse,
    ProviderConfig,
    ProviderError,
    ProviderErrorKind,
    ProviderInvalidRequestError,
    ProviderRequest,
    TokenUsage,
)
from llm_eval_lab.providers.base import (
    ATTEMPT_METADATA_KEY,
    CASE_ID_METADATA_KEY,
    EXPECTED_METADATA_KEY,
    BaseProvider,
)
from llm_eval_lab.utils.json import canonical_json
from llm_eval_lab.utils.time import utc_now

_VOCABULARY: tuple[str, ...] = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
)

_DIGEST_SIZE = 16
_CHARS_PER_TOKEN = 4

EMBEDDING_DIMENSIONS = 16
"""Dimension of a fake-provider embedding vector. Small on purpose: nothing
about the vectors' content is meaningful, only their reproducibility."""

_BYTES_PER_DIMENSION = 2
_UNSIGNED_16_MIDPOINT = 32767.5
"""Centers a big-endian unsigned 16-bit int (0..65535) onto roughly [-1, 1]."""


class FakeProviderOptions(BaseModel):
    """Everything the fake provider can be told to do, parsed from `config.options`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["derived", "scripted", "echo", "expected", "mutate"] = "derived"
    seed: int = 0

    scripted: dict[str, str] = Field(
        default_factory=dict,
        description="Case id to canned answer, for `scripted` mode.",
    )
    fallback: str = Field(
        default="",
        description="Answer used by `scripted` mode for a case with no script entry.",
    )

    fail_case_ids: tuple[str, ...] = Field(
        default=(),
        description="Case ids that always fail, on every attempt.",
    )
    failure_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of cases that fail, drawn from the request digest. Deterministic "
            "per case: a case that fails fails on every attempt."
        ),
    )
    fail_first_n_attempts: int = Field(
        default=0,
        ge=0,
        description=(
            "Fail the first N attempts of every case, then succeed. Deterministic rather "
            "than probabilistic, because the retry path has to be exercised reproducibly: "
            "the digest deliberately excludes the attempt number, so `failure_rate` alone "
            "can only express 'always' or 'never'."
        ),
    )
    fail_kind: ProviderErrorKind = Field(
        default=ProviderErrorKind.SERVER,
        description="Error kind raised by every injected failure.",
    )
    retry_after_s: float | None = Field(
        default=None,
        ge=0.0,
        description="`Retry-After` value attached to injected failures.",
    )

    mutate_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of cases `mutate` mode degrades.",
    )
    finish_reason: Literal["stop", "length", "content_filter", "tool_use", "unknown"] = Field(
        default="stop",
        description=(
            "Finish reason to report. `length` plus `truncated` is how a fixture "
            "produces a cut-off answer for the parsing paths added in a later slice. "
            "`error` is deliberately not offered: a failed attempt raises a "
            "ProviderError, it does not return an error-shaped response."
        ),
    )
    truncated: bool = Field(
        default=False,
        description="Report the response as truncated.",
    )
    latency_min_ms: float = Field(default=5.0, ge=0.0)
    latency_max_ms: float = Field(default=50.0, ge=0.0)
    sleep: bool = Field(
        default=False,
        description=(
            "Actually await the simulated latency. Off by default so the test suite is "
            "fast; the reported latency_ms is the same either way."
        ),
    )
    n_words: int = Field(default=12, ge=1, le=500)


def _last_user_message(request: ProviderRequest) -> str:
    """Return the content of the last user turn, or the last turn of any role."""
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content
    return request.messages[-1].content


def _estimate_tokens(text: str) -> int:
    """Return a deterministic token estimate for simulated usage accounting."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


class FakeProvider(BaseProvider):
    """A provider that fabricates deterministic responses without any network.

    Also implements the ``EmbeddingProvider`` capability (structurally; see
    :meth:`embed`), so ``semantic_similarity`` can be exercised offline with
    ``embedding_provider: {provider: fake, model: fake-embed}``. The vectors
    carry no semantic content - they are a hash of the text, not a model's
    understanding of it - and exist purely to let the evaluator's plumbing
    (embedding, cosine, thresholding, the ``method`` tag) run without a
    vendor credential. A suite that wants a MEANING check needs a real
    embedding-capable provider; see the fake-provider note in
    ``docs/evaluators.md``.
    """

    name: ClassVar[str] = "fake"

    def __init__(self, config: ProviderConfig) -> None:
        """Parse the fake-specific options out of the provider configuration.

        Raises:
            ProviderInvalidRequestError: for an unusable `options` mapping. It is
                surfaced as a provider error rather than a bare pydantic error so
                the runner classifies it like any other non-retryable failure.
        """
        super().__init__(config)
        try:
            self.options = FakeProviderOptions.model_validate(config.options)
        except ValidationError as exc:
            msg = f"invalid options for the fake provider: {exc}"
            raise ProviderInvalidRequestError(msg, provider=self.name) from exc

    def digest(self, request: ProviderRequest) -> str:
        """Return the hex digest every stochastic quantity for this request derives from.

        The composition is fixed and documented: seed, model, request id,
        canonical messages, system prompt, canonical parameters. The ATTEMPT
        NUMBER is deliberately absent, so retrying a case cannot change what the
        model would have said.
        """
        material = "|".join(
            (
                str(self.options.seed),
                self.config.model,
                request.request_id,
                canonical_json([message.model_dump(mode="json") for message in request.messages]),
                request.system or "",
                canonical_json(request.params.model_dump(mode="json")),
            )
        )
        return hashlib.blake2b(material.encode("utf-8"), digest_size=_DIGEST_SIZE).hexdigest()

    def _rng(self, digest: str) -> random.Random:
        """Return a private RNG seeded from the digest. The global RNG is untouched."""
        return random.Random(int(digest, 16))  # noqa: S311 - simulation, not cryptography

    def _text_for(self, request: ProviderRequest, rng: random.Random) -> str:
        """Produce the response text for the configured mode."""
        case_id = request.metadata.get(CASE_ID_METADATA_KEY, request.request_id)
        expected = request.metadata.get(EXPECTED_METADATA_KEY, "")
        mode = self.options.mode

        if mode == "echo":
            return _last_user_message(request)
        if mode == "expected":
            return expected
        if mode == "scripted":
            return self.options.scripted.get(case_id, self.options.fallback)
        if mode == "mutate":
            if rng.random() < self.options.mutate_rate:
                return f"{expected} [mutated]" if expected else "[mutated]"
            return expected
        return " ".join(rng.choice(_VOCABULARY) for _ in range(self.options.n_words))

    def _should_fail(self, request: ProviderRequest, rng: random.Random) -> bool:
        """Decide whether this attempt fails, deterministically."""
        case_id = request.metadata.get(CASE_ID_METADATA_KEY, request.request_id)
        if case_id in self.options.fail_case_ids:
            return True
        attempt = int(request.metadata.get(ATTEMPT_METADATA_KEY, "1"))
        if attempt <= self.options.fail_first_n_attempts:
            return True
        return rng.random() < self.options.failure_rate

    def _raise_injected_failure(self, case_id: str) -> NoReturn:
        """Raise the configured injected failure.

        Raises:
            ProviderError: always. The kind comes from `fail_kind`, so the
                runner's retryable/terminal split is exercised for real.
        """
        msg = f"injected {self.options.fail_kind.value} failure for case {case_id!r}"
        raise ProviderError(
            msg,
            kind=self.options.fail_kind,
            provider=self.name,
            retry_after_s=self.options.retry_after_s,
        )

    async def generate(self, request: ProviderRequest) -> ModelResponse:
        """Return one deterministic simulated response, or raise an injected failure."""
        started_at = utc_now()
        digest = self.digest(request)
        rng = self._rng(digest)

        # The failure draw consumes from the RNG BEFORE the text draw, so a
        # given case's text is the same whether or not failure injection is
        # configured. Reversing the order would make `failure_rate` silently
        # change the content of every passing case.
        failing = self._should_fail(request, rng)
        text = self._text_for(request, rng)
        latency_ms = round(rng.uniform(self.options.latency_min_ms, self.options.latency_max_ms), 3)

        if failing:
            self._raise_injected_failure(
                request.metadata.get(CASE_ID_METADATA_KEY, request.request_id)
            )

        if self.options.sleep:
            await asyncio.sleep(latency_ms / 1000.0)

        prompt_text = "".join(message.content for message in request.messages)
        raw: dict[str, JSONValue] = {
            "simulated": True,
            "seed_digest": digest,
            "mode": self.options.mode,
        }
        return ModelResponse(
            output_text=text,
            provider=self.name,
            model=self.config.model,
            requested_model=self.config.model,
            finish_reason=FinishReason(self.options.finish_reason),
            usage=TokenUsage(
                input_tokens=_estimate_tokens(prompt_text + (request.system or "")),
                output_tokens=_estimate_tokens(text),
            ),
            latency_ms=latency_ms,
            total_latency_ms=latency_ms,
            started_at=started_at,
            completed_at=utc_now(),
            attempts=1,
            truncated=self.options.truncated,
            error=None,
            provider_request_id=f"fake-{digest[:16]}",
            raw=raw,
        )

    # -- EmbeddingProvider (structural; see llm_eval_lab.models.EmbeddingProvider) --

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one deterministic, unit-normalized vector per input text.

        Each vector is derived from a ``blake2b`` digest of the SEED (the
        provider's ``seed`` option, not the request digest above - there is
        no request here, only text) and the text itself, so the same seed and
        text always embed to the same vector, in this process or any other,
        and two providers configured with different seeds embed the same
        text differently. No network, no vendor SDK: genuinely offline.
        """
        return [self._embedding_for(text) for text in texts]

    def _embedding_for(self, text: str) -> list[float]:
        """Hash `text` into a unit-normalized vector of `EMBEDDING_DIMENSIONS` floats."""
        material = f"{self.options.seed}|{text}"
        digest = hashlib.blake2b(
            material.encode("utf-8"),
            digest_size=EMBEDDING_DIMENSIONS * _BYTES_PER_DIMENSION,
        ).digest()
        raw = [
            int.from_bytes(digest[index : index + _BYTES_PER_DIMENSION], "big")
            / _UNSIGNED_16_MIDPOINT
            - 1.0
            for index in range(0, len(digest), _BYTES_PER_DIMENSION)
        ]
        norm = math.sqrt(math.fsum(value * value for value in raw))
        if norm == 0.0:
            # Vanishingly unlikely for a 16-dimensional hash-derived vector,
            # but a defined fallback beats dividing by zero.
            return [1.0, *([0.0] * (EMBEDDING_DIMENSIONS - 1))]
        return [value / norm for value in raw]
