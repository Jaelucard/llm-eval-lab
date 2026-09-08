"""The two similarity evaluators, kept deliberately far apart.

``lexical_similarity`` measures SURFACE WORD OVERLAP. ``semantic_similarity``
measures the cosine angle between two embedding vectors. They answer different
questions, and decision D-SIM is that they must therefore have different names,
different parameters and no code path between them.

**There is no fallback, and that is the point.** A ``semantic_similarity`` case
with no ``embedding_provider`` is a benchmark VALIDATION error naming the case
and the field - the parameter is required, so the registry's load-time pass
refuses the suite before a single request is issued. It never degrades to
lexical overlap. Silently substituting word overlap for meaning is precisely the
dishonesty this project exists to avoid: the number would still be printed, the
dashboard would still render it, and nobody reading the report would know the
question had changed.

Every result carries ``metadata["method"]``: ``lexical:token_set``,
``lexical:char_ngram`` or ``embedding:<provider>:<model>``. The prefix is
generated from the evaluator's own class constant, not from a parameter, so a
lexical result cannot claim to be an embedding one.

**Threshold guidance.** Embedding cosine around 0.80-0.85 is a reasonable
"similar enough" default, but it must be calibrated per embedding model and
never carried across a model change; the ``method`` tag is what makes such a
change visible. Lexical overlap is much coarser - roughly 0.5-0.6 as "similar" -
and is a smoke check, not a measure of meaning. It will fail a correct
paraphrase that shares little vocabulary and pass a wrong answer that happens to
share words with the reference.
"""

import builtins
import math
import re
import unicodedata
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_eval_lab.evaluators.base import BaseEvaluator
from llm_eval_lab.evaluators.deterministic import expected_as_text
from llm_eval_lab.models import (
    EmbeddingProvider,
    EvaluationContext,
    EvaluationResult,
    JSONValue,
    ModelResponse,
    ProviderConfig,
    ProviderError,
    ResolvedCase,
)
from llm_eval_lab.utils.time import monotonic_ms

MIN_NGRAM = 2
MAX_NGRAM = 8

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_for_similarity(text: str, *, normalize: bool) -> str:
    """Fold away the differences that surface comparison should not see.

    With ``normalize`` on: NFKC normalization, case folding, punctuation
    replaced by spaces, whitespace runs collapsed, ends stripped. That order is
    fixed and documented because changing it changes every score this evaluator
    has ever produced. With it off, only the surrounding whitespace is stripped,
    so a suite can compare text exactly as written.
    """
    if not normalize:
        return text.strip()
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE_RUN.sub(" ", _NON_WORD.sub(" ", folded)).strip()


def token_set_similarity(left: str, right: str) -> float:
    """Return the Jaccard overlap of the two texts' token SETS.

    Sets, not multisets: repeating a word does not make an answer more similar
    to a reference that used it once. Two empty texts are identical and score
    1.0; one empty text shares nothing and scores 0.0.
    """
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def char_ngram_similarity(left: str, right: str, *, size: int) -> float:
    """Return the Jaccard overlap of the two texts' character n-gram sets.

    Tolerates inflection and small typos that token comparison treats as total
    misses. A text shorter than ``size`` contributes itself as a single gram, so
    two short identical strings still score 1.0.
    """
    left_grams = _ngrams(left, size)
    right_grams = _ngrams(right, size)
    if not left_grams and not right_grams:
        return 1.0
    union = left_grams | right_grams
    if not union:
        return 1.0
    return len(left_grams & right_grams) / len(union)


def _ngrams(text: str, size: int) -> set[str]:
    """Return the set of character n-grams of `text`."""
    if not text:
        return set()
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    """Return the cosine of the angle between two vectors, or None if undefined.

    ``None`` for a zero-magnitude vector or a length mismatch: both mean the
    embedding backend returned something unusable, and inventing 0.0 for them
    would report "completely dissimilar" for what is actually "no measurement".
    Pure Python on purpose - numpy lives behind an optional extra, and a dot
    product over a few thousand floats does not justify a dependency.
    """
    if len(left) != len(right) or not left:
        return None
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return dot / (left_norm * right_norm)


class LexicalSimilarityParams(BaseModel):
    """Parameters for the `lexical_similarity` evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: str | None = Field(
        default=None,
        description="Overrides the case's `expected` field for this evaluator only.",
    )
    threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Overlap at or above which the case passes. Coarse; calibrate per suite.",
    )
    method: Literal["token_set", "char_ngram"] = Field(
        default="token_set",
        description="`token_set` compares word sets; `char_ngram` compares character n-grams.",
    )
    ngram: int = Field(
        default=3,
        ge=MIN_NGRAM,
        le=MAX_NGRAM,
        description="Character n-gram length, used only by `method: char_ngram`.",
    )
    normalize: bool = Field(
        default=True,
        description="Case-fold, strip punctuation and collapse whitespace before comparing.",
    )


class LexicalSimilarityEvaluator(BaseEvaluator[LexicalSimilarityParams]):
    """Surface word or character overlap. Measures vocabulary, not meaning."""

    type: ClassVar[str] = "lexical_similarity"
    params_model: ClassVar[builtins.type[BaseModel]] = LexicalSimilarityParams
    needs_expected: ClassVar[bool] = True
    method_prefix: ClassVar[str] = "lexical"

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Score the overlap between the response and the expected text."""
        del context
        started = monotonic_ms()
        if response.output_text is None:
            return self.error(
                kind="runtime",
                message="response carries no output text",
                error_code="no_output",
                duration_ms=monotonic_ms() - started,
            )
        expected = expected_as_text(case, self.params.expected)
        if expected is None:
            return self.error(
                kind="config",
                message=(
                    "evaluator 'lexical_similarity' requires an expected value; the case has "
                    "none and no 'expected' parameter was supplied"
                ),
                error_code="missing_expected",
                duration_ms=monotonic_ms() - started,
            )

        actual = normalize_for_similarity(response.output_text, normalize=self.params.normalize)
        target = normalize_for_similarity(expected, normalize=self.params.normalize)
        if self.params.method == "token_set":
            score = token_set_similarity(actual, target)
        else:
            score = char_ngram_similarity(actual, target, size=self.params.ngram)

        method = f"{self.method_prefix}:{self.params.method}"
        return self.verdict(
            passed=score >= self.params.threshold,
            score=score,
            raw_score=score,
            explanation=(
                f"surface {self.params.method} overlap of {score:.3f} against a threshold of "
                f"{self.params.threshold:.3f}. This measures shared vocabulary, not meaning: a "
                f"correct paraphrase using different words scores low, and a wrong answer "
                f"reusing the reference's words scores high."
            ),
            metadata={
                "method": method,
                "threshold": self.params.threshold,
                "normalize": self.params.normalize,
            },
            duration_ms=monotonic_ms() - started,
        )


class SemanticSimilarityParams(BaseModel):
    """Parameters for the `semantic_similarity` evaluator.

    ``embedding_provider`` is REQUIRED (decision D-SIM). Omitting it is a
    benchmark validation error naming the case and the field, not a downgrade to
    lexical overlap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_provider: ProviderConfig = Field(
        description=(
            "The provider and model that produce the embeddings. Required: there is no "
            "lexical fallback, because substituting word overlap for meaning would change "
            "what the reported number means without saying so."
        ),
    )
    expected: str | None = Field(
        default=None,
        description="Overrides the case's `expected` field for this evaluator only.",
    )
    threshold: float = Field(
        default=0.82,
        ge=-1.0,
        le=1.0,
        description=(
            "Cosine at or above which the case passes. Calibrate per embedding model; a "
            "threshold tuned for one model is not portable to another."
        ),
    )
    normalize: bool = Field(
        default=True,
        description="Case-fold, strip punctuation and collapse whitespace before embedding.",
    )


class SemanticSimilarityEvaluator(BaseEvaluator[SemanticSimilarityParams]):
    """Cosine similarity between provider embeddings of the response and reference."""

    type: ClassVar[str] = "semantic_similarity"
    params_model: ClassVar[builtins.type[BaseModel]] = SemanticSimilarityParams
    needs_expected: ClassVar[bool] = True
    is_model_graded: ClassVar[bool] = True
    method_prefix: ClassVar[str] = "embedding"

    def _method(self) -> str:
        """Return the `method` tag naming the exact embedding model used."""
        config = self.params.embedding_provider
        return f"{self.method_prefix}:{config.provider}:{config.model}"

    async def _embed(
        self,
        context: EvaluationContext,
        texts: tuple[str, str],
    ) -> list[list[float]] | EvaluationResult:
        """Embed both texts through the configured provider.

        Returns the vectors, or the ERROR result to emit instead. A provider
        that cannot embed is a CONFIGURATION fault, reported as such: the suite
        named a backend that does not offer the capability it was asked for.
        """
        started = monotonic_ms()
        config = self.params.embedding_provider
        try:
            async with context.provider_factory(config) as provider:
                if not isinstance(provider, EmbeddingProvider):
                    return self.error(
                        kind="config",
                        message=(
                            f"provider {config.provider!r} does not implement the embedding "
                            f"capability that 'semantic_similarity' requires; configure a "
                            f"provider that does, or use 'lexical_similarity' and accept that "
                            f"it measures word overlap rather than meaning"
                        ),
                        error_code="embedding_unsupported",
                        metadata={"method": self._method()},
                        duration_ms=monotonic_ms() - started,
                    )
                vectors = await provider.embed(list(texts))
        except ProviderError as exc:
            return self.error(
                kind="provider",
                message=f"embedding call failed: {exc}",
                error_code="embedding_failed",
                exception_type=type(exc).__name__,
                metadata={"method": self._method()},
                duration_ms=monotonic_ms() - started,
            )

        if len(vectors) != len(texts):
            return self.error(
                kind="runtime",
                message=(
                    f"embedding provider returned {len(vectors)} vectors for {len(texts)} texts"
                ),
                error_code="embedding_count_mismatch",
                metadata={"method": self._method()},
                duration_ms=monotonic_ms() - started,
            )
        return vectors

    async def evaluate(
        self,
        case: ResolvedCase,
        response: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Embed the response and the reference, and compare their directions."""
        started = monotonic_ms()
        if response.output_text is None:
            return self.error(
                kind="runtime",
                message="response carries no output text",
                error_code="no_output",
                duration_ms=monotonic_ms() - started,
            )
        expected = expected_as_text(case, self.params.expected)
        if expected is None:
            return self.error(
                kind="config",
                message=(
                    "evaluator 'semantic_similarity' requires an expected value; the case has "
                    "none and no 'expected' parameter was supplied"
                ),
                error_code="missing_expected",
                duration_ms=monotonic_ms() - started,
            )

        actual = normalize_for_similarity(response.output_text, normalize=self.params.normalize)
        target = normalize_for_similarity(expected, normalize=self.params.normalize)
        embedded = await self._embed(context, (actual, target))
        if isinstance(embedded, EvaluationResult):
            return embedded

        cosine = cosine_similarity(embedded[0], embedded[1])
        if cosine is None:
            return self.error(
                kind="runtime",
                message=(
                    "cosine similarity is undefined for the returned embeddings: a vector was "
                    "zero-magnitude or the two had different dimensions"
                ),
                error_code="degenerate_embedding",
                metadata={"method": self._method()},
                duration_ms=monotonic_ms() - started,
            )

        metadata: dict[str, JSONValue] = {
            "method": self._method(),
            "threshold": self.params.threshold,
            "normalize": self.params.normalize,
            "dimensions": len(embedded[0]),
        }
        # `score` is contractually 0.0..1.0 while cosine spans -1..1. The true
        # value is kept in `raw_score` and the threshold is compared against it,
        # so clamping affects presentation only and never the verdict.
        return self.verdict(
            passed=cosine >= self.params.threshold,
            score=min(1.0, max(0.0, cosine)),
            raw_score=cosine,
            explanation=(
                f"embedding cosine of {cosine:.4f} against a threshold of "
                f"{self.params.threshold:.4f}, measured with {self._method()}. Thresholds are "
                f"specific to the embedding model and are not portable across a model change."
            ),
            metadata=metadata,
            duration_ms=monotonic_ms() - started,
        )


SIMILARITY_EVALUATORS: tuple[type[BaseEvaluator[Any]], ...] = (
    LexicalSimilarityEvaluator,
    SemanticSimilarityEvaluator,
)
"""Every evaluator this module defines, in registration order."""

__all__ = [
    "SIMILARITY_EVALUATORS",
    "LexicalSimilarityEvaluator",
    "LexicalSimilarityParams",
    "SemanticSimilarityEvaluator",
    "SemanticSimilarityParams",
    "char_ngram_similarity",
    "cosine_similarity",
    "normalize_for_similarity",
    "token_set_similarity",
]
