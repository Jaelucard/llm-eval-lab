"""Normalization and error classification for the Ollama adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from llm_eval_lab.models import FinishReason, ProviderError, ProviderErrorKind
from llm_eval_lab.providers.ollama_provider import (
    OLLAMA_DEFAULT_BASE_URL,
    OllamaProvider,
)

from .conftest import COMPLETED_AT, LATENCY_MS, STARTED_AT, config, request_for

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm_eval_lab.providers.remote import AttemptTiming

MODEL = "llama3.2"


def provider(**overrides: Any) -> OllamaProvider:
    return OllamaProvider(config("ollama", MODEL, **overrides))


def response_for(status: int, body: Any) -> httpx.Response:
    """Build a real httpx response. Constructing one opens no socket."""
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.Response(status, json=body, request=request)


# --- happy path ------------------------------------------------------------


def test_a_stop_reply_normalizes_to_stop_with_usage(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("ollama_chat.json"), timing)

    assert response.output_text == "Paris."
    assert response.provider == "ollama"
    assert response.model == MODEL
    assert response.requested_model == MODEL
    assert response.finish_reason is FinishReason.STOP
    assert response.truncated is False
    assert response.attempts == 1
    assert response.provider_request_id is None, "the endpoint returns no request id"

    assert response.usage.input_tokens == 26
    assert response.usage.output_tokens == 4
    assert response.usage.cached_input_tokens == 8
    assert response.usage.total_tokens == 30, "no total is reported, so it is derived"

    assert response.latency_ms == LATENCY_MS
    assert response.started_at == STARTED_AT
    assert response.completed_at == COMPLETED_AT


def test_the_vendor_payload_is_preserved_in_raw(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    payload = payloads("ollama_chat.json")

    assert provider()._normalize(payload, timing).raw["total_duration"] == 4883583458


# --- truncation ------------------------------------------------------------


def test_a_length_reply_is_length_and_truncated(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    response = provider()._normalize(payloads("ollama_length.json"), timing)

    assert response.finish_reason is FinishReason.LENGTH
    assert response.truncated is True
    assert response.output_text == "The capital of Fran"


@pytest.mark.parametrize(
    ("done_reason", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("load", FinishReason.UNKNOWN),
        ("unload", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
        ("something_new", FinishReason.UNKNOWN),
    ],
)
def test_the_done_reason_set_maps_as_documented(
    payloads: Callable[[str], dict[str, Any]],
    timing: AttemptTiming,
    done_reason: str | None,
    expected: FinishReason,
) -> None:
    payload = payloads("ollama_chat.json")
    payload["done_reason"] = done_reason

    assert provider()._normalize(payload, timing).finish_reason is expected


# --- missing usage: the whole reason this adapter exists in this shape -------


def test_absent_token_counters_report_none_and_never_zero(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    """A missing count is silence, not a free request."""
    response = provider()._normalize(payloads("ollama_no_usage.json"), timing)

    assert response.usage.input_tokens is None
    assert response.usage.output_tokens is None
    assert response.usage.total_tokens is None
    assert response.usage.cached_input_tokens is None
    assert response.usage.is_complete is False


def test_a_reported_zero_is_kept_as_zero(
    payloads: Callable[[str], dict[str, Any]], timing: AttemptTiming
) -> None:
    """The counterpart: a server that really did report zero is believed."""
    payload = payloads("ollama_chat.json")
    payload["eval_count"] = 0

    assert provider()._normalize(payload, timing).usage.output_tokens == 0


# --- error classification --------------------------------------------------


def test_a_404_naming_a_missing_model_is_surfaced_verbatim() -> None:
    body = {"error": 'model "llama3.2" not found, try pulling it first'}

    error = provider()._status_error(response_for(404, body))

    assert error.kind is ProviderErrorKind.NOT_FOUND
    assert error.retryable is False
    assert error.status_code == 404
    assert "try pulling it first" in error.message


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, ProviderErrorKind.INVALID_REQUEST),
        (401, ProviderErrorKind.AUTHENTICATION),
        (403, ProviderErrorKind.PERMISSION),
        (404, ProviderErrorKind.NOT_FOUND),
        (429, ProviderErrorKind.RATE_LIMIT),
        (500, ProviderErrorKind.SERVER),
        (503, ProviderErrorKind.SERVER),
    ],
)
def test_http_statuses_map_to_the_documented_kinds(
    status: int, expected: ProviderErrorKind
) -> None:
    error = provider()._status_error(response_for(status, {"error": "nope"}))

    assert error.kind is expected


def test_a_non_json_failure_body_still_produces_a_usable_error() -> None:
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    response = httpx.Response(502, text="<html>bad gateway</html>", request=request)

    error = provider()._status_error(response)

    assert error.kind is ProviderErrorKind.SERVER
    assert "502" in error.message


def test_an_unreachable_server_maps_to_connection_and_stays_retryable() -> None:
    error = provider()._classify_error(httpx.ConnectError("connection refused"))

    assert error.kind is ProviderErrorKind.CONNECTION
    assert error.retryable is True


def test_a_timeout_maps_to_timeout_and_stays_retryable() -> None:
    error = provider()._classify_error(httpx.ReadTimeout("slow"))

    assert error.kind is ProviderErrorKind.TIMEOUT
    assert error.retryable is True


def test_an_unrecognised_exception_is_unknown_and_terminal() -> None:
    error = provider()._classify_error(RuntimeError("something else"))

    assert error.kind is ProviderErrorKind.UNKNOWN
    assert error.retryable is False


# --- request translation ---------------------------------------------------


def test_the_output_cap_becomes_num_predict_under_options() -> None:
    body = provider()._build_body(request_for(params={"max_output_tokens": 64}))

    assert body["options"]["num_predict"] == 64


def test_streaming_is_always_disabled() -> None:
    assert provider()._build_body(request_for())["stream"] is False


def test_the_system_prompt_becomes_a_leading_system_message() -> None:
    request = request_for().model_copy(update={"system": "Be terse."})

    body = provider()._build_body(request)

    assert body["messages"][0] == {"role": "system", "content": "Be terse."}


def test_json_output_is_requested_through_the_format_field() -> None:
    body = provider()._build_body(request_for(params={"response_format": "json_object"}))

    assert body["format"] == "json"


def test_decoding_parameters_are_translated_into_ollamas_vocabulary() -> None:
    body = provider()._build_body(
        request_for(params={"temperature": 0.7, "top_p": 0.9, "stop": ("END",), "seed": 11})
    )

    assert body["options"] == {
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["END"],
        "seed": 11,
    }


def test_vendor_specific_sampler_knobs_travel_through_params_extra() -> None:
    body = provider()._build_body(request_for(params={"extra": {"top_k": 40}}))

    assert body["options"]["top_k"] == 40


def test_a_neutral_parameter_is_never_overridden_by_an_extra_of_the_same_name() -> None:
    """One flat options bag, so a collision must not silently beat the suite."""
    body = provider()._build_body(
        request_for(params={"max_output_tokens": 64, "extra": {"num_predict": 8}})
    )

    assert body["options"]["num_predict"] == 64


def test_an_unknown_option_key_is_refused_rather_than_silently_ignored() -> None:
    with pytest.raises(ProviderError) as caught:
        OllamaProvider(config("ollama", MODEL, options={"nope": 1}))

    assert caught.value.kind is ProviderErrorKind.INVALID_REQUEST


# --- client configuration --------------------------------------------------


def test_the_default_base_url_is_the_local_server() -> None:
    assert provider().base_url == OLLAMA_DEFAULT_BASE_URL


def test_a_configured_base_url_wins_and_a_trailing_slash_is_dropped() -> None:
    built = provider(base_url="http://gpu-box.local:11434/")

    assert built.base_url == "http://gpu-box.local:11434"


def test_no_authorization_header_is_sent_when_no_credential_is_configured() -> None:
    assert "Authorization" not in provider().client().headers


def test_a_configured_credential_becomes_a_bearer_header() -> None:
    """What a reverse-proxy deployment in front of Ollama depends on."""
    from pydantic import SecretStr  # noqa: PLC0415 - used only by this test

    built = OllamaProvider(
        config("ollama", MODEL, api_key_env="OLLAMA_PROXY_CREDENTIAL"),
        SecretStr("test-value-not-a-real-credential"),
    )

    header = built.client().headers["Authorization"]

    assert header == "Bearer test-value-not-a-real-credential"
