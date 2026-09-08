"""Lexical and semantic similarity, and the wall between them (decision D-SIM)."""

import hashlib
import math
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import pytest
import structlog

from llm_eval_lab.datasets.loader import load_suite
from llm_eval_lab.datasets.resolve import resolve_suite
from llm_eval_lab.evaluators import default_registry
from llm_eval_lab.evaluators.similarity import (
    char_ngram_similarity,
    cosine_similarity,
    token_set_similarity,
)
from llm_eval_lab.models import (
    BenchmarkSuite,
    BenchmarkValidationError,
    EmbeddingProvider,
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    EvaluatorSettings,
    EvaluatorSpec,
    FinishReason,
    JSONValue,
    ModelResponse,
    ProviderConfig,
    ProviderRequest,
    ResolvedCase,
)
from llm_eval_lab.utils.time import utc_now

EMBEDDING_DIMENSIONS = 32
EMBEDDING_PROVIDER: dict[str, JSONValue] = {
    "provider": "fake",
    "model": "fake-embed-1",
}
HALF = 0.5
THIRD = 1 / 3

# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class DeterministicEmbeddingProvider:
    """An embedding backend whose vectors depend only on the text.

    Hashed bag-of-characters rather than anything meaningful: the evaluator's
    job is to compute a cosine over whatever the backend returned, and a
    reproducible vector is what lets a test assert an exact number. Identical
    texts embed identically, so their cosine is exactly 1.0.
    """

    name: ClassVar[str] = "fake"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self.calls = 0

    @property
    def config(self) -> ProviderConfig:
        return self._config

    async def generate(self, request: ProviderRequest) -> ModelResponse:
        del request
        msg = "this double only embeds"
        raise AssertionError(msg)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    async def aclose(self) -> None:
        return

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=EMBEDDING_DIMENSIONS).digest()
        return [float(byte) for byte in digest]


class OrthogonalEmbeddingProvider(DeterministicEmbeddingProvider):
    """Returns two vectors that are exactly perpendicular, whatever the input."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        del texts
        return [[1.0, 0.0], [0.0, 1.0]]


class ZeroEmbeddingProvider(DeterministicEmbeddingProvider):
    """Returns a zero-magnitude vector, for which cosine is undefined."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.0, 0.0] for _ in texts]


class TextOnlyProvider:
    """A provider with no embedding capability at all."""

    name: ClassVar[str] = "fake"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    @property
    def config(self) -> ProviderConfig:
        return self._config

    async def generate(self, request: ProviderRequest) -> ModelResponse:
        del request
        msg = "not reached"
        raise AssertionError(msg)

    async def aclose(self) -> None:
        return


class ProviderFactory:
    """Hands out one prepared provider double, and records what was asked for."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.configs: list[ProviderConfig] = []

    def __call__(self, config: ProviderConfig) -> Any:
        self.configs.append(config)
        return self._open()

    @asynccontextmanager
    async def _open(self) -> Any:
        yield self.provider


def _context(factory: ProviderFactory) -> EvaluationContext:
    logger: structlog.BoundLogger = structlog.get_logger("tests")
    return EvaluationContext(
        run_id="test-run",
        suite_hash="sha256:test",
        case_hash="sha256:case",
        provider_factory=factory,
        deadline=None,
        logger=logger,
        settings=EvaluatorSettings(),
    )


def _case(expected: JSONValue, spec: dict[str, Any]) -> ResolvedCase:
    suite = BenchmarkSuite.model_validate(
        {
            "schema_version": 1,
            "name": "similarity",
            "version": "1",
            "cases": [{"id": "c1", "input": "prompt", "expected": expected, "evaluators": [spec]}],
        }
    )
    return resolve_suite(suite).cases[0]


def _response(text: str) -> ModelResponse:
    now = utc_now()
    return ModelResponse(
        output_text=text,
        provider="fake",
        model="fake-1",
        requested_model="fake-1",
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        total_latency_ms=1.0,
        started_at=now,
        completed_at=now,
    )


async def _evaluate(
    expected: JSONValue,
    spec: dict[str, Any],
    output: str,
    context: EvaluationContext,
) -> EvaluationResult:
    case = _case(expected, spec)
    evaluator = default_registry().create(EvaluatorSpec.model_validate(spec))
    return await evaluator.evaluate(case, _response(output), context)


# ---------------------------------------------------------------------------
# the pure similarity functions
# ---------------------------------------------------------------------------


def test_identical_text_scores_one_and_disjoint_text_scores_zero() -> None:
    assert token_set_similarity("alpha beta", "alpha beta") == 1.0
    assert token_set_similarity("alpha beta", "gamma delta") == 0.0
    assert char_ngram_similarity("alpha", "alpha", size=3) == 1.0
    assert char_ngram_similarity("aaa", "zzz", size=3) == 0.0


def test_token_set_similarity_is_jaccard_over_sets() -> None:
    # {a, b} vs {b, c}: one shared token out of three distinct ones.
    assert token_set_similarity("a b", "b c") == pytest.approx(THIRD)
    # Repetition does not increase similarity: sets, not multisets.
    assert token_set_similarity("a a a b", "b c") == pytest.approx(THIRD)


def test_two_empty_texts_are_identical_and_one_empty_text_shares_nothing() -> None:
    assert token_set_similarity("", "") == 1.0
    assert token_set_similarity("", "alpha") == 0.0


def test_cosine_is_undefined_for_a_zero_vector_or_a_dimension_mismatch() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) is None
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0]) is None
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# lexical_similarity
# ---------------------------------------------------------------------------


async def test_lexical_similarity_scores_and_gates_on_its_threshold(
    evaluation_context: EvaluationContext,
) -> None:
    spec = {"type": "lexical_similarity", "params": {"threshold": HALF}}
    same = await _evaluate("alpha beta", spec, "alpha beta", evaluation_context)
    assert same.status is EvaluationStatus.PASSED
    assert same.score == 1.0
    assert same.raw_score == 1.0

    other = await _evaluate("alpha beta", spec, "gamma delta", evaluation_context)
    assert other.status is EvaluationStatus.FAILED
    assert other.score == 0.0


async def test_lexical_similarity_normalizes_case_and_punctuation(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate(
        "Hello, world!",
        {"type": "lexical_similarity"},
        "  HELLO   world  ",
        evaluation_context,
    )
    assert result.score == 1.0

    literal = await _evaluate(
        "Hello, world!",
        {"type": "lexical_similarity", "params": {"normalize": False}},
        "  HELLO   world  ",
        evaluation_context,
    )
    assert literal.score == 0.0


async def test_lexical_similarity_says_plainly_what_it_measures(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate(
        "alpha beta", {"type": "lexical_similarity"}, "alpha beta", evaluation_context
    )
    assert result.explanation is not None
    assert "not meaning" in result.explanation


async def test_lexical_similarity_needs_an_expected_value(
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate(None, {"type": "lexical_similarity"}, "alpha", evaluation_context)
    assert result.status is EvaluationStatus.ERROR
    assert result.passed is None
    assert result.metadata["error_code"] == "missing_expected"


# ---------------------------------------------------------------------------
# D-SIM: the two evaluators are separate, and there is no path between them
# ---------------------------------------------------------------------------


def test_semantic_similarity_without_an_embedding_provider_fails_at_load_time(
    tmp_path: Any,
) -> None:
    """The refusal names the case and the field, and happens before any request."""
    path = tmp_path / "suite.yaml"
    path.write_text(
        "schema_version: 1\n"
        "name: similarity\n"
        'version: "1"\n'
        "cases:\n"
        "  - id: paraphrase-case\n"
        "    input: Describe the tower.\n"
        "    expected: A tower in Paris.\n"
        "    evaluators:\n"
        "      - type: semantic_similarity\n"
        "        params:\n"
        "          threshold: 0.8\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkValidationError) as caught:
        load_suite(path, validators=(default_registry().validate_specs,))

    errors = caught.value.errors
    assert len(errors) == 1
    assert errors[0].case_id == "paraphrase-case"
    assert errors[0].location.endswith(".params.embedding_provider")
    assert "required" in errors[0].message.lower()


def test_there_is_no_fallback_from_semantic_to_lexical() -> None:
    """Neither evaluator can reach the other's implementation.

    Asserted structurally rather than behaviourally: a fallback would have to be
    a call from one class to the other, and there is none. The `method` prefixes
    are class constants, so a lexical result cannot be labelled as an embedding
    one either.
    """
    from llm_eval_lab.evaluators.similarity import (  # noqa: PLC0415
        LexicalSimilarityEvaluator,
        SemanticSimilarityEvaluator,
    )

    assert LexicalSimilarityEvaluator.method_prefix == "lexical"
    assert SemanticSimilarityEvaluator.method_prefix == "embedding"
    assert not issubclass(SemanticSimilarityEvaluator, LexicalSimilarityEvaluator)
    assert not issubclass(LexicalSimilarityEvaluator, SemanticSimilarityEvaluator)


async def test_method_tags_never_cross_between_the_two_evaluators(
    evaluation_context: EvaluationContext,
) -> None:
    lexical = await _evaluate(
        "alpha beta", {"type": "lexical_similarity"}, "alpha beta", evaluation_context
    )
    method = lexical.metadata["method"]
    assert isinstance(method, str)
    assert method.startswith("lexical:")
    assert not method.startswith("embedding:")

    factory = ProviderFactory(DeterministicEmbeddingProvider(ProviderConfig(**EMBEDDING_PROVIDER)))
    semantic = await _evaluate(
        "alpha beta",
        {
            "type": "semantic_similarity",
            "params": {"embedding_provider": EMBEDDING_PROVIDER},
        },
        "alpha beta",
        _context(factory),
    )
    method = semantic.metadata["method"]
    assert isinstance(method, str)
    assert method.startswith("embedding:")
    assert not method.startswith("lexical:")
    assert method == "embedding:fake:fake-embed-1"


@pytest.mark.parametrize("method", ["token_set", "char_ngram"])
async def test_no_lexical_variant_ever_emits_an_embedding_method(
    method: str,
    evaluation_context: EvaluationContext,
) -> None:
    result = await _evaluate(
        "alpha beta",
        {"type": "lexical_similarity", "params": {"method": method}},
        "alpha gamma",
        evaluation_context,
    )
    assert result.metadata["method"] == f"lexical:{method}"


# ---------------------------------------------------------------------------
# semantic_similarity
# ---------------------------------------------------------------------------


async def test_identical_text_embeds_to_a_cosine_of_one() -> None:
    factory = ProviderFactory(DeterministicEmbeddingProvider(ProviderConfig(**EMBEDDING_PROVIDER)))
    result = await _evaluate(
        "the eiffel tower is in paris",
        {"type": "semantic_similarity", "params": {"embedding_provider": EMBEDDING_PROVIDER}},
        "the eiffel tower is in paris",
        _context(factory),
    )
    assert result.status is EvaluationStatus.PASSED
    assert result.raw_score == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)
    assert result.metadata["dimensions"] == EMBEDDING_DIMENSIONS


async def test_orthogonal_embeddings_score_zero_and_fail_the_threshold() -> None:
    factory = ProviderFactory(OrthogonalEmbeddingProvider(ProviderConfig(**EMBEDDING_PROVIDER)))
    result = await _evaluate(
        "alpha",
        {"type": "semantic_similarity", "params": {"embedding_provider": EMBEDDING_PROVIDER}},
        "beta",
        _context(factory),
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.raw_score == pytest.approx(0.0)


async def test_the_same_text_yields_the_same_cosine_on_every_call() -> None:
    """Determinism is what makes an embedding threshold reproducible at all."""
    provider = DeterministicEmbeddingProvider(ProviderConfig(**EMBEDDING_PROVIDER))
    factory = ProviderFactory(provider)
    spec = {
        "type": "semantic_similarity",
        "params": {"embedding_provider": EMBEDDING_PROVIDER, "threshold": 0.0},
    }
    scores = [
        (
            await _evaluate("a tower in paris", spec, "the tower of paris", _context(factory))
        ).raw_score
        for _ in range(3)
    ]
    assert scores[0] is not None
    assert scores == [scores[0], scores[0], scores[0]]
    assert provider.calls == 3


async def test_a_provider_without_the_embedding_capability_is_a_config_error() -> None:
    factory = ProviderFactory(TextOnlyProvider(ProviderConfig(**EMBEDDING_PROVIDER)))
    result = await _evaluate(
        "alpha",
        {"type": "semantic_similarity", "params": {"embedding_provider": EMBEDDING_PROVIDER}},
        "alpha",
        _context(factory),
    )
    assert result.status is EvaluationStatus.ERROR
    assert result.passed is None
    assert result.score is None
    assert result.metadata["error_code"] == "embedding_unsupported"
    assert result.error is not None
    assert result.error.kind == "config"


async def test_a_degenerate_embedding_is_an_error_not_a_zero() -> None:
    factory = ProviderFactory(ZeroEmbeddingProvider(ProviderConfig(**EMBEDDING_PROVIDER)))
    result = await _evaluate(
        "alpha",
        {"type": "semantic_similarity", "params": {"embedding_provider": EMBEDDING_PROVIDER}},
        "alpha",
        _context(factory),
    )
    assert result.status is EvaluationStatus.ERROR
    assert result.metadata["error_code"] == "degenerate_embedding"


async def test_the_embedding_provider_config_is_the_one_that_was_opened() -> None:
    factory = ProviderFactory(DeterministicEmbeddingProvider(ProviderConfig(**EMBEDDING_PROVIDER)))
    await _evaluate(
        "alpha",
        {"type": "semantic_similarity", "params": {"embedding_provider": EMBEDDING_PROVIDER}},
        "alpha",
        _context(factory),
    )
    assert [config.model for config in factory.configs] == ["fake-embed-1"]


def test_the_embedding_provider_double_satisfies_the_declared_capability() -> None:
    provider = DeterministicEmbeddingProvider(ProviderConfig(**EMBEDDING_PROVIDER))
    assert isinstance(provider, EmbeddingProvider)
    assert not isinstance(TextOnlyProvider(ProviderConfig(**EMBEDDING_PROVIDER)), EmbeddingProvider)


def test_the_hashed_double_produces_unit_comparable_vectors() -> None:
    """Guards the fixture itself: a constant vector would make every test pass."""
    left = DeterministicEmbeddingProvider._vector("alpha")
    right = DeterministicEmbeddingProvider._vector("beta")
    assert left != right
    cosine = cosine_similarity(left, right)
    assert cosine is not None
    assert math.isfinite(cosine)
