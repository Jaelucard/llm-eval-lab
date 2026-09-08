"""Normalization and error classification for the Google Gemini adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest
from google.genai import errors as genai_errors
from pydantic import SecretStr

from llm_eval_lab.models import (
    ChatMessage,
    FinishReason,
    ProviderError,
    ProviderErrorKind,
    ProviderRequest,
)
from llm_eval_lab.providers.google_provider import GoogleProvider

from .conftest import COMPLETED_AT, LATENCY_MS, STARTED_AT, config, request_for

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_eval_lab.providers.remote import AttemptTiming

MODEL = "gemini-2.5-flash"
CREDENTIAL = SecretStr("test-key-not-a-real-credential")


def provider(**overrides: Any) -> GoogleProvider:
    return GoogleProvider(
        config("google", MODEL, api_key_env="GOOGLE_API_KEY", **overrides), CREDENTIAL
    )


def api_error(
    exc_class: type[genai_errors.APIError], code: int, status: str
) -> genai_errors.APIError:
    """Build a real SDK exception around a real httpx response. No socket is opened."""
    body = {"error": {"code": code, "status": status, "message": "vendor said no"}}
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models")
    response = httpx.Response(code, json=body, request=request)
    return exc_class(code, body, response)


# --- happy path ------------------------------------------------------------


def test_a_stop_candidate_normalizes_to_stop_with_usage(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("google_generate_content.json"), timing)

    assert response.output_text == "Paris."
    assert response.provider == "google"
    assert response.model == "gemini-2.5-flash"
    assert response.requested_model == MODEL
    assert response.finish_reason is FinishReason.STOP
    assert response.truncated is False
    assert response.attempts == 1
    assert response.provider_request_id == "F3l9aJvTKcSfz7IPu4-x2A0"

    assert response.usage.input_tokens == 18
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 21
    assert response.usage.cached_input_tokens == 6
    assert response.usage.reasoning_tokens == 0

    assert response.latency_ms == LATENCY_MS
    assert response.started_at == STARTED_AT
    assert response.completed_at == COMPLETED_AT


def test_thought_parts_are_excluded_from_the_output(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("google_generate_content.json")
    payload["candidates"][0]["content"]["parts"] = [
        {"text": "internal reasoning nobody scores", "thought": True},
        {"text": "Paris."},
    ]

    assert provider()._normalize(payload, timing).output_text == "Paris."


def test_the_vendor_payload_is_preserved_in_raw(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("google_generate_content.json")

    assert provider()._normalize(payload, timing).raw["response_id"] == payload["response_id"]


# --- truncation and the finish-reason set -----------------------------------


def test_a_max_tokens_candidate_is_length_and_truncated(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("google_max_tokens.json"), timing)

    assert response.finish_reason is FinishReason.LENGTH
    assert response.truncated is True
    assert response.output_text == "The capital of Fran"


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("STOP", FinishReason.STOP),
        ("MAX_TOKENS", FinishReason.LENGTH),
        ("SAFETY", FinishReason.CONTENT_FILTER),
        ("RECITATION", FinishReason.CONTENT_FILTER),
        ("BLOCKLIST", FinishReason.CONTENT_FILTER),
        ("PROHIBITED_CONTENT", FinishReason.CONTENT_FILTER),
        ("SPII", FinishReason.CONTENT_FILTER),
        ("MALFORMED_FUNCTION_CALL", FinishReason.TOOL_USE),
        ("UNEXPECTED_TOOL_CALL", FinishReason.TOOL_USE),
        ("OTHER", FinishReason.UNKNOWN),
        ("LANGUAGE", FinishReason.UNKNOWN),
        ("FINISH_REASON_UNSPECIFIED", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ],
)
def test_the_finish_reason_set_maps_as_documented(
    payloads: Callable[[str], dict[str, Any]],
    timing: AttemptTiming,
    finish_reason: str | None,
    expected: FinishReason,
) -> None:
    payload = payloads("google_generate_content.json")
    payload["candidates"][0]["finish_reason"] = finish_reason

    assert provider()._normalize(payload, timing).finish_reason is expected


def test_a_safety_stopped_candidate_is_a_response_not_an_error(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    """The model was asked and answered with nothing usable. That is reportable."""
    payload = payloads("google_generate_content.json")
    payload["candidates"][0]["finish_reason"] = "SAFETY"
    payload["candidates"][0]["content"] = {"role": "model", "parts": []}

    response = provider()._normalize(payload, timing)

    assert response.finish_reason is FinishReason.CONTENT_FILTER
    assert response.output_text == ""
    assert response.error is None


# --- missing usage ---------------------------------------------------------


def test_a_response_without_usage_metadata_reports_none_not_zero(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("google_no_usage.json"), timing)

    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.total_tokens is None
    assert response.usage.cached_input_tokens is None
    assert response.usage.reasoning_tokens is None


# --- blocked prompts -------------------------------------------------------


def test_a_blocked_prompt_raises_a_content_filter_error(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    """No candidates means nothing was generated, which is not an empty answer."""
    with pytest.raises(ProviderError) as caught:
        provider()._normalize(payloads("google_blocked.json"), timing)

    assert caught.value.kind is ProviderErrorKind.CONTENT_FILTER
    assert caught.value.retryable is False
    assert caught.value.vendor_code == "SAFETY"


def test_an_empty_reply_with_no_block_reason_is_unknown_and_terminal(
    timing: AttemptTiming,
) -> None:
    with pytest.raises(ProviderError) as caught:
        provider()._normalize({"response_id": "x", "candidates": []}, timing)

    assert caught.value.kind is ProviderErrorKind.UNKNOWN
    assert caught.value.retryable is False


# --- error classification --------------------------------------------------


@pytest.mark.parametrize(
    ("exc_class", "code", "status", "expected"),
    [
        (genai_errors.ClientError, 401, "UNAUTHENTICATED", ProviderErrorKind.AUTHENTICATION),
        (genai_errors.ClientError, 403, "PERMISSION_DENIED", ProviderErrorKind.PERMISSION),
        (genai_errors.ClientError, 404, "NOT_FOUND", ProviderErrorKind.NOT_FOUND),
        (genai_errors.ClientError, 400, "INVALID_ARGUMENT", ProviderErrorKind.INVALID_REQUEST),
        (genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED", ProviderErrorKind.RATE_LIMIT),
        (genai_errors.ServerError, 503, "UNAVAILABLE", ProviderErrorKind.SERVER),
        (genai_errors.ServerError, 500, "INTERNAL", ProviderErrorKind.SERVER),
        (genai_errors.ClientError, 499, "CANCELLED", ProviderErrorKind.CANCELLED),
    ],
)
def test_vendor_statuses_map_to_the_documented_kinds(
    exc_class: type[genai_errors.APIError],
    code: int,
    status: str,
    expected: ProviderErrorKind,
) -> None:
    error = provider()._classify_error(api_error(exc_class, code, status))

    assert error.kind is expected
    assert error.status_code == code
    assert error.vendor_code == status


def test_a_throttle_is_retryable_because_it_is_a_rate_limit_not_an_exhausted_quota() -> None:
    error = provider()._classify_error(
        api_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED")
    )

    assert error.retryable is True


def test_an_unrecognised_vendor_status_falls_back_to_the_http_status() -> None:
    error = provider()._classify_error(api_error(genai_errors.ServerError, 502, "SOMETHING_NEW"))

    assert error.kind is ProviderErrorKind.SERVER


def test_a_raw_httpx_timeout_below_the_sdk_maps_to_timeout() -> None:
    error = provider()._classify_error(httpx.ReadTimeout("slow"))

    assert error.kind is ProviderErrorKind.TIMEOUT
    assert error.retryable is True


def test_an_unrecognised_exception_is_unknown_and_terminal() -> None:
    error = provider()._classify_error(RuntimeError("something else"))

    assert error.kind is ProviderErrorKind.UNKNOWN
    assert error.retryable is False


# --- request translation ---------------------------------------------------


def test_the_system_prompt_and_output_cap_live_on_the_generation_config() -> None:
    from google.genai import types  # noqa: PLC0415 - mirrors the adapter's lazy import

    request = request_for(params={"max_output_tokens": 96, "temperature": 0.4}).model_copy(
        update={"system": "Be terse."}
    )

    built = provider()._build_config(request, types)

    assert built.system_instruction == "Be terse."
    assert built.max_output_tokens == 96
    assert built.temperature == 0.4


def test_json_output_is_requested_through_the_response_mime_type() -> None:
    from google.genai import types  # noqa: PLC0415 - mirrors the adapter's lazy import

    built = provider()._build_config(request_for(params={"response_format": "json_object"}), types)

    assert built.response_mime_type == "application/json"


def test_the_assistant_role_is_renamed_to_model_and_system_turns_are_lifted_out() -> None:
    from google.genai import types  # noqa: PLC0415 - mirrors the adapter's lazy import

    request = ProviderRequest(
        messages=(
            ChatMessage(role="system", content="Be terse."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
            ChatMessage(role="user", content="Capital of France?"),
        ),
        request_id="req-1",
    )

    contents = provider()._build_contents(request, types)

    assert [item.role for item in contents] == ["user", "model", "user"]
    assert provider()._build_config(request, types).system_instruction == "Be terse."


def test_an_unknown_option_key_is_refused_rather_than_silently_ignored() -> None:
    with pytest.raises(ProviderError) as caught:
        GoogleProvider(config("google", MODEL, options={"nope": 1}), CREDENTIAL)

    assert caught.value.kind is ProviderErrorKind.INVALID_REQUEST


# --- client configuration --------------------------------------------------


def test_the_client_disables_the_sdks_own_retry_loop() -> None:
    """This SDK has no max_retries; HttpRetryOptions(attempts=1) is the equivalent."""
    options = provider().client()._api_client._http_options

    assert options.retry_options is not None
    assert options.retry_options.attempts == 1


def test_the_client_timeout_is_expressed_in_milliseconds() -> None:
    built = GoogleProvider(config("google", MODEL, timeout_s=12.5), CREDENTIAL).client()

    assert built._api_client._http_options.timeout == 12500


def test_vendor_specific_knobs_travel_through_params_extra() -> None:
    """`extra` reaches the request through http_options.extra_body, as elsewhere."""
    from google.genai import types  # noqa: PLC0415 - mirrors the adapter's lazy import

    built = provider()._build_config(
        request_for(params={"extra": {"safetySettings": [], "labels": {"team": "evals"}}}), types
    )

    assert built.http_options is not None
    assert built.http_options.extra_body == {
        "safetySettings": [],
        "labels": {"team": "evals"},
    }


def test_no_http_options_are_attached_when_the_benchmark_sets_no_extra() -> None:
    from google.genai import types  # noqa: PLC0415 - mirrors the adapter's lazy import

    assert provider()._build_config(request_for(), types).http_options is None


def test_building_a_client_with_no_credential_names_the_variable_to_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This SDK raises a bare ValueError, which no error class covers."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ProviderError) as caught:
        GoogleProvider(config("google", MODEL), None).client()

    assert caught.value.kind is ProviderErrorKind.AUTHENTICATION
    assert caught.value.retryable is False
    assert "GOOGLE_API_KEY" in caught.value.message


def test_the_sdk_is_never_left_to_read_the_environment_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This SDK reads GOOGLE_API_KEY and GEMINI_API_KEY on its own when given None."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-value-not-a-real-credential")
    monkeypatch.setenv("GEMINI_API_KEY", "test-value-not-a-real-credential")

    with pytest.raises(ProviderError) as caught:
        GoogleProvider(config("google", MODEL), None).client()

    assert caught.value.kind is ProviderErrorKind.AUTHENTICATION


def test_a_separate_connect_budget_is_honoured() -> None:
    """HttpOptions.timeout is one millisecond value, so connect goes to the httpx client."""
    built = GoogleProvider(
        config("google", MODEL, timeout_s=30.0, connect_timeout_s=5.0), CREDENTIAL
    ).client()
    args = built._api_client._http_options.async_client_args

    assert args is not None
    assert args["timeout"].connect == 5.0
    assert args["timeout"].read == 30.0
