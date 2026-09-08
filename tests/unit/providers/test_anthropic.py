"""Normalization and error classification for the Anthropic Messages adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anthropic
import httpx
import pytest
from pydantic import SecretStr

from llm_eval_lab.models import FinishReason, ProviderError, ProviderErrorKind
from llm_eval_lab.providers.anthropic_provider import DEFAULT_MAX_TOKENS, AnthropicProvider

from .conftest import COMPLETED_AT, LATENCY_MS, STARTED_AT, config, request_for

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_eval_lab.providers.remote import AttemptTiming

MODEL = "claude-sonnet-5"
CREDENTIAL = SecretStr("test-key-not-a-real-credential")


def provider(**overrides: Any) -> AnthropicProvider:
    return AnthropicProvider(
        config("anthropic", MODEL, api_key_env="ANTHROPIC_API_KEY", **overrides), CREDENTIAL
    )


def status_error(
    exc_class: type[anthropic.APIStatusError],
    status: int,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> anthropic.APIStatusError:
    """Build a real SDK exception around a real httpx response. No socket is opened."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, headers=headers or {}, json=body, request=request)
    return exc_class(body["error"]["message"], response=response, body=body)


# --- happy path ------------------------------------------------------------


def test_an_end_turn_message_normalizes_to_stop_with_usage(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("anthropic_message.json"), timing)

    assert response.output_text == "Paris."
    assert response.provider == "anthropic"
    assert response.model == MODEL
    assert response.requested_model == MODEL
    assert response.finish_reason is FinishReason.STOP
    assert response.truncated is False
    assert response.attempts == 1
    assert response.provider_request_id == "msg_01XFDUDYJgAACzvnptvVoYEL"

    assert response.usage.input_tokens == 25
    assert response.usage.output_tokens == 4
    assert response.usage.cached_input_tokens == 12
    assert response.usage.total_tokens == 29, "the API reports no total, so it is derived"

    assert response.latency_ms == LATENCY_MS
    assert response.started_at == STARTED_AT
    assert response.completed_at == COMPLETED_AT


def test_only_text_blocks_contribute_to_the_output(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("anthropic_message.json")
    payload["content"] = [
        {"type": "thinking", "thinking": "internal reasoning nobody scores"},
        {"type": "text", "text": "Paris"},
        {"type": "tool_use", "id": "tu_1", "name": "lookup", "input": {}},
        {"type": "text", "text": "."},
    ]

    assert provider()._normalize(payload, timing).output_text == "Paris."


def test_the_vendor_payload_is_preserved_in_raw(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("anthropic_message.json")

    assert provider()._normalize(payload, timing).raw["id"] == payload["id"]


# --- truncation and the full stop_reason set --------------------------------


def test_a_max_tokens_stop_is_length_and_truncated(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("anthropic_max_tokens.json"), timing)

    assert response.finish_reason is FinishReason.LENGTH
    assert response.truncated is True
    assert response.output_text == "The capital of Fran"


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("max_tokens", FinishReason.LENGTH),
        ("model_context_window_exceeded", FinishReason.LENGTH),
        ("tool_use", FinishReason.TOOL_USE),
        ("refusal", FinishReason.CONTENT_FILTER),
        ("pause_turn", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
        ("something_new", FinishReason.UNKNOWN),
    ],
)
def test_every_pinned_stop_reason_maps_to_the_documented_finish_reason(
    payloads: Callable[[str], dict[str, Any]],
    timing: AttemptTiming,
    stop_reason: str | None,
    expected: FinishReason,
) -> None:
    payload = payloads("anthropic_message.json")
    payload["stop_reason"] = stop_reason

    assert provider()._normalize(payload, timing).finish_reason is expected


# --- missing usage ---------------------------------------------------------


def test_a_message_without_usage_reports_none_not_zero(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("anthropic_no_usage.json"), timing)

    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.total_tokens is None
    assert response.usage.cached_input_tokens is None


# --- error classification --------------------------------------------------


def test_a_401_maps_to_authentication_and_is_not_retryable(
    payloads: Callable[[str], dict[str, Any]],
) -> None:
    body = payloads("anthropic_error_401.json")
    exc = status_error(anthropic.AuthenticationError, 401, body)

    error = provider()._classify_error(exc)

    assert error.kind is ProviderErrorKind.AUTHENTICATION
    assert error.retryable is False
    assert error.status_code == 401
    assert error.vendor_code == "authentication_error"


def test_a_429_maps_to_rate_limit_and_honours_retry_after() -> None:
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    exc = status_error(anthropic.RateLimitError, 429, body, headers={"retry-after": "12"})

    error = provider()._classify_error(exc)

    assert error.kind is ProviderErrorKind.RATE_LIMIT
    assert error.retryable is True
    assert error.retry_after_s == 12.0


@pytest.mark.parametrize(
    ("exc_class", "status", "error_type", "expected"),
    [
        (anthropic.PermissionDeniedError, 403, "permission_error", ProviderErrorKind.PERMISSION),
        (anthropic.NotFoundError, 404, "not_found_error", ProviderErrorKind.NOT_FOUND),
        (
            anthropic.RequestTooLargeError,
            413,
            "request_too_large",
            ProviderErrorKind.CONTEXT_LENGTH,
        ),
        (anthropic.InternalServerError, 500, "api_error", ProviderErrorKind.SERVER),
        (anthropic.OverloadedError, 529, "overloaded_error", ProviderErrorKind.SERVER),
        (
            anthropic.BadRequestError,
            400,
            "invalid_request_error",
            ProviderErrorKind.INVALID_REQUEST,
        ),
        (anthropic.BadRequestError, 400, "prompt_too_long", ProviderErrorKind.CONTEXT_LENGTH),
    ],
)
def test_sdk_status_exceptions_map_to_the_documented_kinds(
    exc_class: type[anthropic.APIStatusError],
    status: int,
    error_type: str,
    expected: ProviderErrorKind,
) -> None:
    body = {"type": "error", "error": {"type": error_type, "message": "nope"}}

    error = provider()._classify_error(status_error(exc_class, status, body))

    assert error.kind is expected
    assert error.status_code == status


def test_a_timeout_maps_to_timeout_and_stays_retryable() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    error = provider()._classify_error(anthropic.APITimeoutError(request=request))

    assert error.kind is ProviderErrorKind.TIMEOUT
    assert error.retryable is True


def test_a_connection_failure_maps_to_connection() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = anthropic.APIConnectionError(message="cannot connect", request=request)

    assert provider()._classify_error(exc).kind is ProviderErrorKind.CONNECTION


def test_an_unrecognised_exception_is_unknown_and_terminal() -> None:
    error = provider()._classify_error(RuntimeError("something else"))

    assert error.kind is ProviderErrorKind.UNKNOWN
    assert error.retryable is False


# --- request translation ---------------------------------------------------


def test_max_tokens_is_always_sent_because_the_api_requires_it() -> None:
    assert provider()._build_kwargs(request_for())["max_tokens"] == DEFAULT_MAX_TOKENS


def test_the_benchmarks_output_cap_wins_over_the_default() -> None:
    kwargs = provider()._build_kwargs(request_for(params={"max_output_tokens": 64}))

    assert kwargs["max_tokens"] == 64


def test_the_default_output_cap_is_configurable_per_provider() -> None:
    """Spelled without "token": ProviderConfig refuses option keys containing it."""
    built = AnthropicProvider(
        config("anthropic", MODEL, options={"default_output_cap": 256}), CREDENTIAL
    )

    assert built._build_kwargs(request_for())["max_tokens"] == 256


def test_the_system_prompt_is_a_top_level_parameter_not_a_message() -> None:
    request = request_for().model_copy(update={"system": "Be terse."})

    kwargs = provider()._build_kwargs(request)

    assert kwargs["system"] == "Be terse."
    assert all(message["role"] != "system" for message in kwargs["messages"])


def test_temperature_is_omitted_unless_the_benchmark_set_it_explicitly() -> None:
    assert "temperature" not in provider()._build_kwargs(request_for())
    assert provider()._build_kwargs(request_for(params={"temperature": 0.3}))["temperature"] == 0.3


def test_stop_sequences_are_translated() -> None:
    kwargs = provider()._build_kwargs(request_for(params={"stop": ("END", "STOP")}))

    assert kwargs["stop_sequences"] == ["END", "STOP"]


def test_an_unknown_option_key_is_refused_rather_than_silently_ignored() -> None:
    with pytest.raises(ProviderError) as caught:
        AnthropicProvider(config("anthropic", MODEL, options={"nope": 1}), CREDENTIAL)

    assert caught.value.kind is ProviderErrorKind.INVALID_REQUEST


# --- client configuration --------------------------------------------------


def test_the_client_disables_the_sdks_own_retry_loop() -> None:
    assert provider().client().max_retries == 0


def test_building_a_client_with_no_credential_names_the_variable_to_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This SDK constructs happily with no key and only fails inside the request.

    Left alone that surfaces as a TypeError no error class covers, which the
    classifier would report as an unexplained UNKNOWN rather than as the
    configuration mistake it is.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ProviderError) as caught:
        AnthropicProvider(config("anthropic", MODEL), None).client()

    assert caught.value.kind is ProviderErrorKind.AUTHENTICATION
    assert caught.value.retryable is False
    assert "ANTHROPIC_API_KEY" in caught.value.message


def test_the_sdk_is_never_left_to_read_the_environment_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key only in the environment must not silently pay for a run."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value-not-a-real-credential")

    with pytest.raises(ProviderError) as caught:
        AnthropicProvider(config("anthropic", MODEL), None).client()

    assert caught.value.kind is ProviderErrorKind.AUTHENTICATION


def test_a_separate_connect_budget_is_honoured() -> None:
    built = AnthropicProvider(
        config("anthropic", MODEL, timeout_s=30.0, connect_timeout_s=5.0), CREDENTIAL
    ).client()

    assert isinstance(built.timeout, httpx.Timeout)
    assert built.timeout.connect == 5.0
    assert built.timeout.read == 30.0
