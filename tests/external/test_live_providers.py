"""One real request per provider, against the real vendor. Opt-in only.

Every test here is marked `external` and is deselected by default through
`addopts = "-m 'not external'"`, so an ordinary `uv run pytest` never touches
the network and never spends money. Run them deliberately:

    uv run pytest -m external -q

A test whose credential variable is unset SKIPS rather than failing, so running
the whole file with one key configured exercises exactly that one provider. The
Ollama test additionally skips when nothing is listening on the local server,
because "no Ollama running" is a normal state on a laptop, not a defect.

What these prove that the fixture-based unit tests cannot: that the request this
project builds is one the vendor actually accepts, and that the response shape
the adapters normalize is the shape the vendor currently sends. They are
therefore deliberately minimal - one short prompt, a tight output cap - so the
whole file costs a fraction of a cent.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx
import pytest

from llm_eval_lab.models import (
    ChatMessage,
    FinishReason,
    GenerationParams,
    ProviderConfig,
    ProviderRequest,
)
from llm_eval_lab.providers import default_registry
from llm_eval_lab.providers.ollama_provider import OLLAMA_DEFAULT_BASE_URL

if TYPE_CHECKING:
    from llm_eval_lab.models import ModelResponse

pytestmark = pytest.mark.external

MAX_OUTPUT_TOKENS = 16
PROMPT = "Reply with exactly one word: the capital city of France."

LIVE_MODELS: dict[str, str] = {
    "openai": "gpt-5-nano",
    "anthropic": "claude-haiku-4-5",
    "google": "gemini-2.5-flash-lite",
}
"""The cheapest current model per vendor. Override with LLM_EVAL_LIVE_<PROVIDER>_MODEL."""


def model_for(provider: str) -> str:
    """Return the model to exercise, allowing an override from the environment."""
    return os.environ.get(f"LLM_EVAL_LIVE_{provider.upper()}_MODEL", LIVE_MODELS[provider])


def require_credential(provider: str) -> str:
    """Skip unless this provider's credential variable is set.

    Asked through the registry's own resolver rather than by reading
    ``os.environ``, so the secret is never bound to a local just to test its
    truthiness. ``credential_status`` returns ``True``, ``False`` or ``None``
    and never the value.
    """
    info = default_registry().info(provider)
    assert info.credential_env is not None
    if default_registry().credential_status(provider) is not True:
        pytest.skip(f"{info.credential_env} is not set")
    return info.credential_env


def require_ollama(base_url: str) -> None:
    """Skip unless a real Ollama server answers, so a laptop without one is not a failure.

    An HTTP probe rather than a bare TCP connect: a proxy, a port-forward or an
    unrelated service can accept a connection on the port without being Ollama,
    and a test that then fails on the real request is reporting the wrong thing.
    """
    try:
        probe = httpx.get(base_url.removesuffix("/v1"), timeout=2.0)
    except httpx.HTTPError as exc:
        pytest.skip(f"no Ollama server reachable at {base_url}: {type(exc).__name__}")
    if probe.status_code != httpx.codes.OK or "ollama" not in probe.text.lower():
        pytest.skip(f"the server at {base_url} does not identify itself as Ollama")


def ollama_model(base_url: str, override_var: str) -> str:
    """Return a model this server has actually pulled, or skip.

    A running server with nothing pulled is as normal a laptop state as no
    server at all, and "model not found" is a 404 that says nothing about
    whether this project builds a request Ollama accepts. When the caller names
    no model, whatever the server already has is used, so the live path is
    genuinely exercised on any machine rather than skipped on most of them.
    """
    server = base_url.removesuffix("/v1")
    try:
        listing = httpx.get(f"{server}/api/tags", timeout=5.0)
        listing.raise_for_status()
        available = [
            entry["model"]
            for entry in listing.json().get("models", [])
            if isinstance(entry, dict) and isinstance(entry.get("model"), str)
        ]
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        pytest.skip(f"cannot list models at {server}: {type(exc).__name__}")

    wanted = os.environ.get(override_var)
    if wanted is not None:
        if wanted not in available:
            pytest.skip(f"{override_var}={wanted} is not pulled on {server}")
        return wanted
    if not available:
        pytest.skip(f"the Ollama server at {server} has no models pulled")
    first: str = available[0]
    return first


def request_for(prompt: str = PROMPT) -> ProviderRequest:
    """Build the one small request every live test sends."""
    return ProviderRequest(
        messages=(ChatMessage(role="user", content=prompt),),
        system="Answer with a single word and no punctuation.",
        params=GenerationParams(max_output_tokens=MAX_OUTPUT_TOKENS, temperature=0.0),
        request_id="live-1",
    )


def assert_normalized(response: ModelResponse, provider: str) -> None:
    """Assert the invariants every real response must satisfy, whoever served it."""
    assert response.provider == provider
    assert response.error is None
    assert response.output_text is not None
    assert response.finish_reason is not FinishReason.ERROR
    assert response.attempts == 1, "a provider makes exactly one attempt (decision D4)"
    assert response.latency_ms > 0.0
    assert response.completed_at >= response.started_at
    assert response.model
    assert response.raw


async def run_live(
    provider: str, *, model: str | None = None, base_url: str | None = None
) -> ModelResponse:
    """Perform one real request through the registry, closing the client afterwards."""
    info = default_registry().info(provider)
    config = ProviderConfig(
        provider=provider,
        model=model if model is not None else model_for(provider),
        api_key_env=info.credential_env,
        base_url=base_url,
        timeout_s=60.0,
    )
    async with default_registry().open(config) as built:
        return await built.generate(request_for())


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
async def test_a_real_request_normalizes_into_the_response_contract(provider: str) -> None:
    require_credential(provider)

    response = await run_live(provider)

    assert_normalized(response, provider)
    assert "paris" in (response.output_text or "").lower()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
async def test_a_real_response_reports_token_counts(provider: str) -> None:
    """Every hosted vendor reports usage; a None here means the mapping drifted."""
    require_credential(provider)

    usage = (await run_live(provider)).usage

    assert usage.input_tokens is not None
    assert usage.output_tokens is not None
    assert usage.total_tokens is not None


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
async def test_a_bad_credential_is_classified_as_an_authentication_failure(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the live error path, not just the fixture-constructed one."""
    from llm_eval_lab.models import ProviderError, ProviderErrorKind  # noqa: PLC0415

    require_credential(provider)
    monkeypatch.setenv("LLM_EVAL_LIVE_BAD_KEY", "not-a-valid-credential")
    info = default_registry().info(provider)
    config = ProviderConfig(
        provider=provider,
        model=model_for(provider),
        api_key_env="LLM_EVAL_LIVE_BAD_KEY",
        timeout_s=60.0,
    )

    async with default_registry().open(config) as built:
        with pytest.raises(ProviderError) as caught:
            await built.generate(request_for())

    assert caught.value.kind in {
        ProviderErrorKind.AUTHENTICATION,
        ProviderErrorKind.PERMISSION,
        ProviderErrorKind.INVALID_REQUEST,
    }, f"{info.name} classified a bad credential as {caught.value.kind}"
    assert caught.value.retryable is False


async def test_a_real_ollama_request_normalizes_into_the_response_contract() -> None:
    base_url = os.environ.get("LLM_EVAL_LIVE_OLLAMA_URL", OLLAMA_DEFAULT_BASE_URL)
    require_ollama(base_url)
    model = ollama_model(base_url, "LLM_EVAL_LIVE_OLLAMA_MODEL")

    response = await run_live("ollama", model=model, base_url=base_url)

    assert_normalized(response, "ollama")


async def test_a_real_openai_compatible_request_normalizes_into_the_response_contract() -> None:
    """Exercised against Ollama's own OpenAI-compatible endpoint when one is running."""
    base_url = os.environ.get("LLM_EVAL_LIVE_COMPATIBLE_URL", f"{OLLAMA_DEFAULT_BASE_URL}/v1")
    require_ollama(base_url)
    model = ollama_model(base_url, "LLM_EVAL_LIVE_COMPATIBLE_MODEL")

    response = await run_live("openai_compatible", model=model, base_url=base_url)

    assert_normalized(response, "openai_compatible")
