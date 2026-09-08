"""Normalization for the OpenAI-compatible adapter, including its missing fields.

The point of this adapter is graceful degradation: third-party servers implement
the Chat Completions schema to varying depths, and every field below the response
text has to be allowed to go missing without the run inventing a value for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import openai
import pytest
from pydantic import SecretStr

from llm_eval_lab.models import FinishReason, ProviderError, ProviderErrorKind
from llm_eval_lab.providers.openai_compatible import (
    PLACEHOLDER_API_KEY,
    OpenAICompatibleProvider,
)

from .conftest import LATENCY_MS, config, request_for

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_eval_lab.providers.remote import AttemptTiming

MODEL = "Qwen/Qwen3-32B-Instruct"
BASE_URL = "http://localhost:8000/v1"


def provider(**overrides: Any) -> OpenAICompatibleProvider:
    overrides.setdefault("base_url", BASE_URL)
    return OpenAICompatibleProvider(config("openai_compatible", MODEL, **overrides))


# --- happy path ------------------------------------------------------------


def test_a_stop_completion_normalizes_with_full_usage(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("compatible_chat_completion.json"), timing)

    assert response.output_text == "Paris."
    assert response.provider == "openai_compatible"
    assert response.model == MODEL
    assert response.requested_model == MODEL
    assert response.finish_reason is FinishReason.STOP
    assert response.truncated is False
    assert response.attempts == 1
    assert response.provider_request_id == "chatcmpl-3f9c2a1b"
    assert response.latency_ms == LATENCY_MS

    assert response.usage.input_tokens == 31
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 36


def test_the_served_model_and_the_requested_model_are_recorded_separately(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    """Endpoints routinely echo back a differently spelled id, and both matter."""
    payload = payloads("compatible_chat_completion.json")
    payload["model"] = "qwen3-32b-awq"

    response = provider()._normalize(payload, timing)

    assert response.model == "qwen3-32b-awq"
    assert response.requested_model == MODEL


# --- truncation ------------------------------------------------------------


def test_a_length_finish_reason_is_length_and_truncated(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("compatible_length.json"), timing)

    assert response.finish_reason is FinishReason.LENGTH
    assert response.truncated is True
    assert response.output_text == "The capital of Fran"


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("tool_calls", FinishReason.TOOL_USE),
        ("function_call", FinishReason.TOOL_USE),
        (None, FinishReason.UNKNOWN),
        ("eos", FinishReason.UNKNOWN),
    ],
)
def test_the_finish_reason_set_maps_and_unknown_values_are_not_guessed(
    payloads: Callable[[str], dict[str, Any]],
    timing: AttemptTiming,
    finish_reason: str | None,
    expected: FinishReason,
) -> None:
    payload = payloads("compatible_chat_completion.json")
    payload["choices"][0]["finish_reason"] = finish_reason

    assert provider()._normalize(payload, timing).finish_reason is expected


# --- degradation -----------------------------------------------------------


def test_a_minimally_conformant_reply_degrades_to_none_rather_than_inventing_values(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("compatible_minimal.json"), timing)

    assert response.output_text == "Paris."
    assert response.finish_reason is FinishReason.UNKNOWN
    assert response.provider_request_id is None
    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.total_tokens is None
    assert response.usage.cached_input_tokens is None
    assert response.usage.reasoning_tokens is None


def test_a_tool_call_only_reply_has_empty_output_rather_than_none(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("compatible_chat_completion.json")
    payload["choices"][0]["message"]["content"] = None
    payload["choices"][0]["finish_reason"] = "tool_calls"

    response = provider()._normalize(payload, timing)

    assert response.output_text == ""
    assert response.finish_reason is FinishReason.TOOL_USE
    assert response.error is None


def test_a_content_part_array_is_joined_into_text(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("compatible_chat_completion.json")
    payload["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "Par"},
        {"type": "text", "text": "is."},
    ]

    assert provider()._normalize(payload, timing).output_text == "Paris."


def test_a_reply_with_no_choices_raises_rather_than_reporting_empty_output(
    timing: AttemptTiming,
) -> None:
    with pytest.raises(ProviderError) as caught:
        provider()._normalize({"id": "x", "model": MODEL, "choices": []}, timing)

    assert caught.value.kind is ProviderErrorKind.UNKNOWN


# --- configuration ---------------------------------------------------------


def test_a_base_url_is_required_because_there_is_no_safe_default() -> None:
    with pytest.raises(ProviderError) as caught:
        OpenAICompatibleProvider(config("openai_compatible", MODEL))

    assert caught.value.kind is ProviderErrorKind.INVALID_REQUEST
    assert "base_url" in caught.value.message


def test_a_placeholder_key_is_sent_when_the_endpoint_needs_no_credential() -> None:
    client = provider().client()

    assert client.api_key == PLACEHOLDER_API_KEY
    assert not PLACEHOLDER_API_KEY.startswith("sk")


def test_a_configured_credential_is_used_when_one_is_resolved() -> None:
    built = OpenAICompatibleProvider(
        config("openai_compatible", MODEL, base_url=BASE_URL, api_key_env="LOCAL_TOKEN_ENV"),
        SecretStr("test-key-not-a-real-credential"),
    )

    assert built.client().api_key == "test-key-not-a-real-credential"


def test_the_client_points_at_the_configured_endpoint_with_retries_disabled() -> None:
    client = provider().client()

    assert str(client.base_url).rstrip("/") == BASE_URL
    assert client.max_retries == 0


# --- request translation ---------------------------------------------------


def test_the_output_cap_uses_the_chat_completions_parameter_name() -> None:
    kwargs = provider()._build_kwargs(request_for(params={"max_output_tokens": 64}))

    assert kwargs["max_completion_tokens"] == 64


def test_stop_and_seed_are_sent_because_chat_completions_accepts_them() -> None:
    kwargs = provider()._build_kwargs(request_for(params={"stop": ("END",), "seed": 11}))

    assert kwargs["stop"] == ["END"]
    assert kwargs["seed"] == 11


def test_the_system_prompt_becomes_a_leading_system_message() -> None:
    request = request_for().model_copy(update={"system": "Be terse."})

    kwargs = provider()._build_kwargs(request)

    assert kwargs["messages"][0] == {"role": "system", "content": "Be terse."}


# --- error classification --------------------------------------------------


def test_errors_are_classified_by_the_shared_openai_mapping_under_this_providers_name() -> None:
    request = httpx.Request("POST", f"{BASE_URL}/chat/completions")
    body = {"error": {"message": "no key", "code": "invalid_api_key"}}
    response = httpx.Response(401, json=body, request=request)
    exc = openai.AuthenticationError("no key", response=response, body=body["error"])

    error = provider()._classify_error(exc)

    assert error.kind is ProviderErrorKind.AUTHENTICATION
    assert error.provider == "openai_compatible"


def test_an_unreachable_local_server_maps_to_connection() -> None:
    request = httpx.Request("POST", f"{BASE_URL}/chat/completions")
    exc = openai.APIConnectionError(message="connection refused", request=request)

    error = provider()._classify_error(exc)

    assert error.kind is ProviderErrorKind.CONNECTION
    assert error.retryable is True
