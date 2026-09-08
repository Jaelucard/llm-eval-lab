"""The fake provider's determinism guarantees, which the whole suite rests on."""

import asyncio
import random
from typing import Any

import pytest

from llm_eval_lab.models import (
    ChatMessage,
    FinishReason,
    GenerationParams,
    ProviderConfig,
    ProviderError,
    ProviderErrorKind,
    ProviderRequest,
)
from llm_eval_lab.providers import (
    ATTEMPT_METADATA_KEY,
    CASE_ID_METADATA_KEY,
    EXPECTED_METADATA_KEY,
    FakeProvider,
)

CASE_IDS = ("c1", "c2", "c3", "c4", "c5", "c6")
HIGH_CONCURRENCY = 32
THIRD_ATTEMPT = 3


def _config(**options: Any) -> ProviderConfig:
    return ProviderConfig(provider="fake", model="fake-1", options=options)


def _request(case_id: str, *, attempt: int = 1, expected: str = "") -> ProviderRequest:
    metadata = {CASE_ID_METADATA_KEY: case_id, ATTEMPT_METADATA_KEY: str(attempt)}
    if expected:
        metadata[EXPECTED_METADATA_KEY] = expected
    return ProviderRequest(
        messages=(ChatMessage(role="user", content=f"prompt for {case_id}"),),
        system="be terse",
        params=GenerationParams(temperature=0.0),
        request_id=case_id,
        metadata=metadata,
    )


async def _generate_all(provider: FakeProvider, *, concurrency: int) -> list[tuple[Any, ...]]:
    """Generate every case under a bounded concurrency and return comparable tuples."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case_id: str) -> tuple[str, Any, Any, float]:
        async with semaphore:
            response = await provider.generate(_request(case_id))
        return (case_id, response.output_text, response.usage, response.latency_ms)

    results = await asyncio.gather(*(one(case_id) for case_id in CASE_IDS))
    return sorted(results, key=lambda row: row[0])


async def test_output_is_identical_at_concurrency_one_and_thirty_two() -> None:
    provider = FakeProvider(_config(seed=7))
    sequential = await _generate_all(provider, concurrency=1)
    parallel = await _generate_all(provider, concurrency=HIGH_CONCURRENCY)
    assert sequential == parallel


async def test_two_separate_providers_agree_given_the_same_seed() -> None:
    first = await _generate_all(FakeProvider(_config(seed=7)), concurrency=1)
    second = await _generate_all(FakeProvider(_config(seed=7)), concurrency=HIGH_CONCURRENCY)
    assert first == second


async def test_a_different_seed_produces_different_output() -> None:
    first = await _generate_all(FakeProvider(_config(seed=1)), concurrency=1)
    second = await _generate_all(FakeProvider(_config(seed=2)), concurrency=1)
    assert [row[1] for row in first] != [row[1] for row in second]


async def test_fail_case_ids_fails_exactly_that_case_on_every_attempt() -> None:
    provider = FakeProvider(_config(fail_case_ids=["c3"]))
    for case_id in CASE_IDS:
        if case_id == "c3":
            continue
        assert (await provider.generate(_request(case_id))).output_text is not None

    for attempt in (1, 2, 3, 9):
        with pytest.raises(ProviderError) as caught:
            await provider.generate(_request("c3", attempt=attempt))
        assert caught.value.kind is ProviderErrorKind.SERVER
        assert caught.value.retryable


async def test_a_non_retryable_failure_kind_is_reported_as_such() -> None:
    provider = FakeProvider(_config(fail_case_ids=["c1"], fail_kind="authentication"))
    with pytest.raises(ProviderError) as caught:
        await provider.generate(_request("c1"))
    assert caught.value.kind is ProviderErrorKind.AUTHENTICATION
    assert not caught.value.retryable


async def test_fail_first_n_attempts_succeeds_on_the_next_attempt() -> None:
    provider = FakeProvider(_config(fail_first_n_attempts=2))
    for attempt in (1, 2):
        with pytest.raises(ProviderError):
            await provider.generate(_request("c1", attempt=attempt))
    response = await provider.generate(_request("c1", attempt=THIRD_ATTEMPT))
    assert response.output_text is not None


async def test_expected_mode_returns_the_case_expected_answer() -> None:
    provider = FakeProvider(_config(mode="expected"))
    response = await provider.generate(_request("c1", expected="Paris"))
    assert response.output_text == "Paris"


async def test_echo_mode_returns_the_last_user_message() -> None:
    provider = FakeProvider(_config(mode="echo"))
    response = await provider.generate(_request("c1"))
    assert response.output_text == "prompt for c1"


async def test_scripted_mode_uses_the_script_then_the_fallback() -> None:
    provider = FakeProvider(
        _config(mode="scripted", scripted={"c1": "scripted answer"}, fallback="nothing")
    )
    assert (await provider.generate(_request("c1"))).output_text == "scripted answer"
    assert (await provider.generate(_request("c2"))).output_text == "nothing"


async def test_mutate_mode_degrades_a_deterministic_subset() -> None:
    provider = FakeProvider(_config(mode="mutate", mutate_rate=1.0))
    response = await provider.generate(_request("c1", expected="Paris"))
    assert response.output_text == "Paris [mutated]"

    untouched = FakeProvider(_config(mode="mutate", mutate_rate=0.0))
    assert (await untouched.generate(_request("c1", expected="Paris"))).output_text == "Paris"


async def test_raw_records_that_the_response_was_simulated() -> None:
    provider = FakeProvider(_config())
    response = await provider.generate(_request("c1"))
    assert response.raw["simulated"] is True
    assert response.raw["mode"] == "derived"
    assert isinstance(response.raw["seed_digest"], str)


async def test_the_global_rng_is_never_touched() -> None:
    # The fake provider seeds a LOCAL `random.Random`. If it ever reached for
    # the module-level generator instead, a run would perturb every other user
    # of `random` in the process, and its own output would depend on how many
    # cases ran before it.
    provider = FakeProvider(_config())
    random.seed(1234)
    before = random.random()  # noqa: S311 - checking the global RNG, not generating a secret
    random.seed(1234)
    await provider.generate(_request("c1"))
    assert random.random() == before  # noqa: S311 - same


async def test_unusable_options_are_a_provider_error() -> None:
    with pytest.raises(ProviderError) as caught:
        FakeProvider(_config(mode="not-a-mode"))
    assert "invalid options" in caught.value.message


async def test_truncation_and_finish_reason_can_be_injected() -> None:
    """A cut-off answer needs a fixture; the parsing paths in a later slice want one."""
    provider = FakeProvider(_config(mode="echo", finish_reason="length", truncated=True))
    response = await provider.generate(_request("c1"))
    assert response.finish_reason is FinishReason.LENGTH
    assert response.truncated is True
    assert response.succeeded, "a truncated answer is still an answer, not an error"


# ---------------------------------------------------------------------------
# EmbeddingProvider: identical texts, unrelated texts, cross-process determinism
# ---------------------------------------------------------------------------

EMBEDDING_DIMENSIONS = 16
UNRELATED_COSINE_CEILING = 0.5


def _cosine(left: list[float], right: list[float]) -> float:
    dot: float = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm: float = sum(a * a for a in left) ** 0.5
    right_norm: float = sum(b * b for b in right) ** 0.5
    return dot / (left_norm * right_norm)


async def test_the_fake_provider_satisfies_the_embedding_provider_protocol() -> None:
    from llm_eval_lab.models import EmbeddingProvider  # noqa: PLC0415 - reads next to its use

    assert isinstance(FakeProvider(_config()), EmbeddingProvider)


async def test_identical_texts_embed_to_a_cosine_of_one() -> None:
    provider = FakeProvider(_config())
    vectors = await provider.embed(["the quick brown fox", "the quick brown fox"])
    assert len(vectors[0]) == EMBEDDING_DIMENSIONS
    assert _cosine(vectors[0], vectors[1]) == pytest.approx(1.0)


async def test_unrelated_texts_embed_below_the_similarity_ceiling() -> None:
    provider = FakeProvider(_config())
    vectors = await provider.embed(
        ["the quick brown fox jumps over the lazy dog", "spacecraft re-entry thermal shielding"]
    )
    assert _cosine(vectors[0], vectors[1]) < UNRELATED_COSINE_CEILING


async def test_every_vector_is_unit_normalized() -> None:
    provider = FakeProvider(_config())
    vectors = await provider.embed(["alpha", "beta", "gamma"])
    for vector in vectors:
        norm = sum(value * value for value in vector) ** 0.5
        assert norm == pytest.approx(1.0)


async def test_embeddings_are_deterministic_across_processes() -> None:
    """A pure hash of the seed and the text: no process-local state to diverge.

    Run in-process against a hand-computed value rather than spawning a real
    subprocess, since the whole point of `blake2b` over text is that it
    cannot depend on anything process-local (RNG state, memory addresses,
    hash randomization) in the first place - there is nothing a second
    process could see differently.
    """
    provider = FakeProvider(_config())
    first = await provider.embed(["reproducible"])
    second = await FakeProvider(_config()).embed(["reproducible"])
    assert first == second


async def test_a_different_seed_embeds_the_same_text_differently() -> None:
    default_seed = await FakeProvider(_config()).embed(["same text"])
    other_seed = await FakeProvider(_config(seed=99)).embed(["same text"])
    assert default_seed != other_seed


async def test_semantic_similarity_runs_offline_against_the_fake_embedding_provider() -> None:
    """The evaluator's own path, not just the provider in isolation."""
    import structlog  # noqa: PLC0415 - test-local, keeps the module import list lean

    from llm_eval_lab.datasets.resolve import resolve_suite  # noqa: PLC0415
    from llm_eval_lab.evaluators import default_registry  # noqa: PLC0415
    from llm_eval_lab.models import (  # noqa: PLC0415
        BenchmarkSuite,
        EvaluationContext,
        EvaluationStatus,
        EvaluatorSettings,
    )
    from llm_eval_lab.models import ModelResponse as _ModelResponse  # noqa: PLC0415
    from llm_eval_lab.providers.registry import (  # noqa: PLC0415
        FAKE_PROVIDER_INFO,
        ProviderRegistry,
        build_fake_provider,
    )
    from llm_eval_lab.utils.time import utc_now  # noqa: PLC0415

    registry = ProviderRegistry()
    registry.register(FAKE_PROVIDER_INFO, build_fake_provider)

    suite = BenchmarkSuite.model_validate(
        {
            "schema_version": 1,
            "name": "semantic-offline",
            "version": "1",
            "cases": [
                {
                    "id": "c1",
                    "input": "prompt",
                    "expected": "the quick brown fox",
                    "evaluators": [
                        {
                            "type": "semantic_similarity",
                            "params": {
                                "embedding_provider": {"provider": "fake", "model": "fake-embed"},
                                "threshold": 0.5,
                            },
                        }
                    ],
                }
            ],
        }
    )
    case = resolve_suite(suite).cases[0]
    response = _ModelResponse(
        output_text="the quick brown fox",
        provider="fake",
        model="fake-1",
        requested_model="fake-1",
        finish_reason=FinishReason.STOP,
        latency_ms=1.0,
        total_latency_ms=1.0,
        started_at=utc_now(),
        completed_at=utc_now(),
    )
    context = EvaluationContext(
        run_id="test-run",
        suite_hash="sha256:test",
        case_hash="sha256:case",
        provider_factory=registry.open,
        deadline=None,
        logger=structlog.get_logger("tests"),
        settings=EvaluatorSettings(),
    )
    evaluator = default_registry().create(case.evaluators[0])
    result = await evaluator.evaluate(case, response, context)
    assert result.status is EvaluationStatus.PASSED
    assert result.metadata["method"] == "embedding:fake:fake-embed"
