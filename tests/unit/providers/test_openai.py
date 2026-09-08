"""Normalization and error classification for the OpenAI Responses adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import openai
import pytest
from pydantic import SecretStr

from llm_eval_lab.models import (
    FinishReason,
    GenerationParams,
    ProviderError,
    ProviderErrorKind,
)
from llm_eval_lab.providers.openai_provider import OpenAIProvider
from llm_eval_lab.providers.remote import RAW_TRUNCATED_KEY

from .conftest import COMPLETED_AT, LATENCY_MS, STARTED_AT, config, request_for

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_eval_lab.providers.remote import AttemptTiming

MODEL = "gpt-5.1"


CREDENTIAL = SecretStr("test-key-not-a-real-credential")


def provider(**overrides: Any) -> OpenAIProvider:
    """Build a provider the way the registry does: config plus a resolved credential."""
    return OpenAIProvider(
        config("openai", MODEL, api_key_env="OPENAI_API_KEY", **overrides), CREDENTIAL
    )


def status_error(
    exc_class: type[openai.APIStatusError],
    status: int,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> openai.APIStatusError:
    """Build a real SDK exception around a real httpx response. No socket is opened."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status, headers=headers or {}, json=body, request=request)
    message = body.get("error", {}).get("message", "error")
    return exc_class(message, response=response, body=body["error"])


# --- happy path ------------------------------------------------------------


def test_a_completed_response_normalizes_to_stop_with_full_usage(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("openai_response.json"), timing)

    assert response.output_text == "Paris."
    assert response.provider == "openai"
    assert response.model == "gpt-5.1-2026-04-14"
    assert response.requested_model == MODEL
    assert response.finish_reason is FinishReason.STOP
    assert response.truncated is False
    assert response.error is None
    assert response.attempts == 1
    assert response.provider_request_id == "resp_6f2b1c9d4e5a4f8b90a1b2c3d4e5f607"

    assert response.usage.input_tokens == 42
    assert response.usage.output_tokens == 7
    assert response.usage.total_tokens == 49
    assert response.usage.cached_input_tokens == 16
    assert response.usage.reasoning_tokens == 0


def test_the_measured_latency_and_timestamps_are_carried_through(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("openai_response.json"), timing)

    assert response.latency_ms == LATENCY_MS
    assert response.total_latency_ms == LATENCY_MS
    assert response.started_at == STARTED_AT
    assert response.completed_at == COMPLETED_AT


def test_the_vendor_payload_is_preserved_in_raw(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("openai_response.json")
    response = provider()._normalize(payload, timing)

    assert response.raw["id"] == payload["id"]
    assert RAW_TRUNCATED_KEY not in response.raw


def test_an_oversized_payload_is_replaced_by_a_marker(timing: AttemptTiming) -> None:
    payload = {
        "id": "resp_big",
        "status": "completed",
        "model": MODEL,
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
        ],
        "padding": "x" * 100_000,
    }
    response = provider()._normalize(payload, timing)

    assert response.output_text == "ok"
    assert response.raw[RAW_TRUNCATED_KEY] is True
    assert "padding" not in response.raw


# --- truncation ------------------------------------------------------------


def test_an_incomplete_response_capped_by_output_tokens_is_length_and_truncated(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("openai_incomplete_length.json"), timing)

    assert response.finish_reason is FinishReason.LENGTH
    assert response.truncated is True
    assert response.output_text == "The capital of Fran"


def test_an_incomplete_response_stopped_by_a_content_filter_is_reported_as_such(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("openai_incomplete_length.json")
    payload["incomplete_details"] = {"reason": "content_filter"}

    response = provider()._normalize(payload, timing)

    assert response.finish_reason is FinishReason.CONTENT_FILTER
    assert response.truncated is False


def test_an_incomplete_reason_outside_the_pinned_set_is_unknown_not_guessed(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("openai_incomplete_length.json")
    payload["incomplete_details"] = {"reason": "steered"}

    assert provider()._normalize(payload, timing).finish_reason is FinishReason.UNKNOWN


def test_a_tool_call_output_item_is_reported_as_tool_use(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("openai_response.json")
    payload["output"].append({"type": "function_call", "name": "lookup", "arguments": "{}"})

    assert provider()._normalize(payload, timing).finish_reason is FinishReason.TOOL_USE


def test_a_refusal_block_is_reported_as_a_content_filter(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("openai_response.json")
    payload["output"] = [
        {"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}
    ]

    response = provider()._normalize(payload, timing)

    assert response.finish_reason is FinishReason.CONTENT_FILTER
    assert response.output_text == ""


# --- missing usage ---------------------------------------------------------


def test_a_response_without_usage_reports_none_not_zero(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("openai_no_usage.json"), timing)

    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.total_tokens is None
    assert response.usage.cached_input_tokens is None
    assert response.usage.reasoning_tokens is None


# --- terminal statuses -----------------------------------------------------


def test_a_failed_response_raises_rather_than_returning_empty_output(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("openai_response.json")
    payload["status"] = "failed"
    payload["error"] = {"code": "server_error", "message": "upstream exploded"}

    with pytest.raises(ProviderError) as caught:
        provider()._normalize(payload, timing)

    assert caught.value.kind is ProviderErrorKind.SERVER
    assert caught.value.retryable is True


def test_a_failed_response_flagged_as_content_policy_is_not_retryable(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("openai_response.json")
    payload["status"] = "failed"
    payload["error"] = {"code": "content_filter", "message": "refused"}

    with pytest.raises(ProviderError) as caught:
        provider()._normalize(payload, timing)

    assert caught.value.kind is ProviderErrorKind.CONTENT_FILTER
    assert caught.value.retryable is False


def test_a_cancelled_response_raises_a_cancelled_error(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("openai_response.json")
    payload["status"] = "cancelled"

    with pytest.raises(ProviderError) as caught:
        provider()._normalize(payload, timing)

    assert caught.value.kind is ProviderErrorKind.CANCELLED


# --- error classification --------------------------------------------------


def test_a_429_maps_to_rate_limit_and_honours_retry_after(
    payloads: Callable[[str], dict[str, Any]],
) -> None:
    body = payloads("openai_error_429.json")
    exc = status_error(openai.RateLimitError, 429, body, headers={"retry-after": "30"})

    error = provider()._classify_error(exc)

    assert error.kind is ProviderErrorKind.RATE_LIMIT
    assert error.retryable is True
    assert error.status_code == 429
    assert error.retry_after_s == 30.0
    assert error.vendor_code == "rate_limit_exceeded"


def test_a_rate_limit_without_a_retry_after_header_reports_none(
    payloads: Callable[[str], dict[str, Any]],
) -> None:
    exc = status_error(openai.RateLimitError, 429, payloads("openai_error_429.json"))

    assert provider()._classify_error(exc).retry_after_s is None


@pytest.mark.parametrize(
    ("exc_class", "status", "code", "expected"),
    [
        (openai.AuthenticationError, 401, "invalid_api_key", ProviderErrorKind.AUTHENTICATION),
        (openai.PermissionDeniedError, 403, "insufficient_scope", ProviderErrorKind.PERMISSION),
        (openai.NotFoundError, 404, "model_not_found", ProviderErrorKind.NOT_FOUND),
        (openai.InternalServerError, 500, "server_error", ProviderErrorKind.SERVER),
        (openai.UnprocessableEntityError, 422, "invalid", ProviderErrorKind.INVALID_REQUEST),
        (openai.BadRequestError, 400, "invalid_prompt", ProviderErrorKind.INVALID_REQUEST),
        (openai.BadRequestError, 400, "context_length_exceeded", ProviderErrorKind.CONTEXT_LENGTH),
        (openai.BadRequestError, 400, "content_policy_violation", ProviderErrorKind.CONTENT_FILTER),
        (openai.BadRequestError, 400, "insufficient_quota", ProviderErrorKind.QUOTA),
        (openai.BadRequestError, 400, "unsupported_parameter", ProviderErrorKind.UNSUPPORTED),
    ],
)
def test_sdk_status_exceptions_map_to_the_documented_kinds(
    exc_class: type[openai.APIStatusError],
    status: int,
    code: str,
    expected: ProviderErrorKind,
) -> None:
    body = {"error": {"message": "nope", "code": code, "type": "invalid_request_error"}}

    error = provider()._classify_error(status_error(exc_class, status, body))

    assert error.kind is expected
    assert error.status_code == status
    assert error.vendor_code == code


def test_a_timeout_maps_to_timeout_and_stays_retryable() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    error = provider()._classify_error(openai.APITimeoutError(request=request))

    assert error.kind is ProviderErrorKind.TIMEOUT
    assert error.retryable is True


def test_a_connection_failure_maps_to_connection() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    exc = openai.APIConnectionError(message="cannot connect", request=request)

    assert provider()._classify_error(exc).kind is ProviderErrorKind.CONNECTION


def test_a_raw_httpx_timeout_below_the_sdk_still_maps_to_timeout() -> None:
    assert provider()._classify_error(httpx.ReadTimeout("slow")).kind is ProviderErrorKind.TIMEOUT


def test_an_unrecognised_exception_is_unknown_and_terminal() -> None:
    error = provider()._classify_error(RuntimeError("something else"))

    assert error.kind is ProviderErrorKind.UNKNOWN
    assert error.retryable is False


def test_a_credential_echoed_back_in_an_error_message_is_scrubbed() -> None:
    leaked = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    body = {
        "error": {"message": f"Incorrect API key provided: {leaked}", "code": "invalid_api_key"}
    }

    error = provider()._classify_error(status_error(openai.AuthenticationError, 401, body))

    assert leaked not in error.message
    assert "[redacted]" in error.message


# --- request translation ---------------------------------------------------


def test_the_system_prompt_becomes_instructions_and_the_output_cap_keeps_its_name() -> None:
    request = request_for(params={"max_output_tokens": 128, "temperature": 0.2})
    kwargs = provider()._build_kwargs(request.model_copy(update={"system": "Be terse."}))

    assert kwargs["instructions"] == "Be terse."
    assert kwargs["max_output_tokens"] == 128
    assert kwargs["temperature"] == 0.2
    assert kwargs["model"] == MODEL
    assert kwargs["input"] == [{"role": "user", "content": "What is the capital of France?"}]


def test_json_output_is_requested_through_the_text_format_parameter() -> None:
    kwargs = provider()._build_kwargs(request_for(params={"response_format": "json_object"}))

    assert kwargs["text"] == {"format": {"type": "json_object"}}


def test_stop_and_seed_are_dropped_because_the_responses_api_has_no_such_parameters() -> None:
    request = request_for(params={"stop": ("END",), "seed": 7})
    kwargs = provider()._build_kwargs(request)

    assert "stop" not in kwargs
    assert "seed" not in kwargs


def test_responses_are_not_stored_by_the_vendor_unless_asked_for() -> None:
    assert provider()._build_kwargs(request_for())["store"] is False
    stored = OpenAIProvider(config("openai", MODEL, options={"store": True}), CREDENTIAL)
    assert stored._build_kwargs(request_for())["store"] is True


def test_an_unknown_option_key_is_refused_rather_than_silently_ignored() -> None:
    with pytest.raises(ProviderError) as caught:
        OpenAIProvider(config("openai", MODEL, options={"stroe": True}), CREDENTIAL)

    assert caught.value.kind is ProviderErrorKind.INVALID_REQUEST


def test_temperature_is_omitted_when_the_benchmark_never_set_it() -> None:
    assert "temperature" not in provider()._build_kwargs(
        request_for(params=GenerationParams().model_dump())
    )


# --- client configuration --------------------------------------------------


def test_the_client_disables_the_sdks_own_retry_loop() -> None:
    client = provider().client()

    assert client.max_retries == 0


def test_the_client_honours_the_configured_base_url_and_timeout() -> None:
    built = OpenAIProvider(
        config("openai", MODEL, base_url="https://example.test/v1", timeout_s=12.5), CREDENTIAL
    ).client()

    assert str(built.base_url).rstrip("/") == "https://example.test/v1"
    assert isinstance(built.timeout, httpx.Timeout)
    assert built.timeout.read == 12.5


def test_building_a_client_with_no_credential_names_the_variable_to_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing key is an authentication failure, not an unexplained one."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    with pytest.raises(ProviderError) as caught:
        OpenAIProvider(config("openai", MODEL), None).client()

    assert caught.value.kind is ProviderErrorKind.AUTHENTICATION
    assert caught.value.retryable is False
    assert "OPENAI_API_KEY" in caught.value.message
