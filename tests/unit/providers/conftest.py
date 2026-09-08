"""Shared helpers for the provider normalization tests.

Every test in this package works from a recorded vendor payload under
`tests/fixtures/provider_payloads/`. Those files are exactly what an adapter's
`_invoke` returns - an SDK response object's `model_dump(mode="json")`, or, for
Ollama, the wire JSON - so calling `_normalize` on one exercises the real
mapping with no client, no credential and no socket. The autouse network block
in the root conftest is active throughout.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from llm_eval_lab.models import ChatMessage, GenerationParams, ProviderConfig, ProviderRequest
from llm_eval_lab.providers.remote import AttemptTiming

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

STARTED_AT = datetime(2026, 9, 7, 9, 14, 2, tzinfo=UTC)
COMPLETED_AT = datetime(2026, 9, 7, 9, 14, 3, tzinfo=UTC)
LATENCY_MS = 1234.5


@pytest.fixture
def timing() -> AttemptTiming:
    """Fixed timing, so a normalization assertion never depends on a clock."""
    return AttemptTiming(started_at=STARTED_AT, completed_at=COMPLETED_AT, latency_ms=LATENCY_MS)


@pytest.fixture
def payloads(fixtures_dir: Path) -> Callable[[str], dict[str, Any]]:
    """Return a loader for one recorded vendor payload by file name."""

    def load(name: str) -> dict[str, Any]:
        text = (fixtures_dir / "provider_payloads" / name).read_text(encoding="utf-8")
        loaded: dict[str, Any] = json.loads(text)
        return loaded

    return load


def config(provider: str, model: str, **overrides: Any) -> ProviderConfig:
    """Build a provider configuration naming a credential VARIABLE, never a value."""
    return ProviderConfig(provider=provider, model=model, **overrides)


def request_for(text: str = "What is the capital of France?", **kwargs: Any) -> ProviderRequest:
    """Build a one-turn request with a stable request id."""
    return ProviderRequest(
        messages=(ChatMessage(role="user", content=text),),
        request_id="req-1",
        params=GenerationParams(**kwargs.pop("params", {})),
        **kwargs,
    )
