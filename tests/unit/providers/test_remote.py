"""The shared remote-provider machinery: one attempt, honest counts, scrubbed payloads."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from llm_eval_lab.models import (
    FinishReason,
    ModelResponse,
    ProviderError,
    ProviderErrorKind,
    ProviderRequest,
    TokenUsage,
)
from llm_eval_lab.providers.remote import (
    MAX_ERROR_MESSAGE_CHARS,
    MAX_RAW_BYTES,
    RAW_TRUNCATED_KEY,
    AttemptTiming,
    RemoteProvider,
    capped_raw,
    identifier_at,
    optional_count,
    parse_retry_after,
    scrub_message,
    status_kind,
)

from .conftest import config, request_for

if TYPE_CHECKING:
    from llm_eval_lab.models import ProviderConfig


class StubProvider(RemoteProvider[dict[str, Any]]):
    """A remote provider whose one request is whatever the test tells it to be."""

    name: ClassVar[str] = "stub"

    def __init__(self, cfg: ProviderConfig, *, raises: Exception | None = None) -> None:
        super().__init__(cfg, None)
        self.calls = 0
        self._raises = raises

    async def _invoke(self, request: ProviderRequest) -> dict[str, Any]:
        del request
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return {"text": "ok"}

    def _normalize(self, payload: dict[str, Any], timing: AttemptTiming) -> ModelResponse:
        return ModelResponse(
            output_text=str(payload["text"]),
            provider=self.name,
            model=self._config.model,
            requested_model=self._config.model,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(),
            latency_ms=timing.latency_ms,
            total_latency_ms=timing.latency_ms,
            started_at=timing.started_at,
            completed_at=timing.completed_at,
            raw=capped_raw(payload),
        )

    def _classify_error(self, exc: Exception) -> ProviderError:
        return ProviderError(scrub_message(exc), kind=ProviderErrorKind.SERVER, provider=self.name)


def stub(**kwargs: Any) -> StubProvider:
    return StubProvider(config("stub", "stub-1"), **kwargs)


# --- one attempt per call ---------------------------------------------------


async def test_generate_performs_exactly_one_attempt() -> None:
    provider = stub()

    response = await provider.generate(request_for())

    assert provider.calls == 1
    assert response.attempts == 1
    assert response.output_text == "ok"


async def test_a_failing_attempt_is_not_retried_by_the_provider() -> None:
    provider = stub(raises=RuntimeError("boom"))

    with pytest.raises(ProviderError):
        await provider.generate(request_for())

    assert provider.calls == 1, "retries belong to the runner, not to a provider"


async def test_an_already_typed_provider_error_passes_through_unchanged() -> None:
    original = ProviderError("nope", kind=ProviderErrorKind.AUTHENTICATION, provider="stub")
    provider = stub(raises=original)

    with pytest.raises(ProviderError) as caught:
        await provider.generate(request_for())

    assert caught.value is original
    assert caught.value.kind is ProviderErrorKind.AUTHENTICATION


async def test_cancellation_is_never_converted_into_a_provider_error() -> None:
    """A cancelled run must cancel, not be recorded as a vendor failure."""
    provider = stub(raises=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await provider.generate(request_for())


async def test_the_measured_latency_is_non_negative_and_the_stamps_are_ordered() -> None:
    response = await stub().generate(request_for())

    assert response.latency_ms >= 0.0
    assert response.completed_at >= response.started_at


# --- token counts -----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (7, 7),
        (7.0, 7),
        (None, None),
        (-1, None),
        (True, None),
        (False, None),
        ("7", None),
        (7.5, None),
        ([], None),
    ],
)
def test_only_a_real_non_negative_count_becomes_a_count(value: Any, expected: int | None) -> None:
    assert optional_count(value) == expected


def test_a_boolean_is_not_a_token_count() -> None:
    """`True` is an int in Python and would otherwise arrive as one token."""
    truthy: Any = True

    assert optional_count(truthy) is None


# --- retry-after ------------------------------------------------------------


def test_a_numeric_retry_after_is_read_as_seconds() -> None:
    assert parse_retry_after("30") == 30.0
    assert parse_retry_after("1.5") == 1.5


def test_an_http_date_retry_after_becomes_a_delay() -> None:
    now = datetime(2026, 9, 7, 9, 0, 0, tzinfo=UTC)
    header = "Mon, 07 Sep 2026 09:00:30 GMT"

    assert parse_retry_after(header, now=now) == 30.0


def test_a_past_dated_retry_after_is_zero_rather_than_negative() -> None:
    now = datetime(2026, 9, 7, 9, 1, 0, tzinfo=UTC)
    header = "Mon, 07 Sep 2026 09:00:30 GMT"

    assert parse_retry_after(header, now=now) == 0.0


@pytest.mark.parametrize("value", [None, "", "   ", "soon", "-5"])
def test_an_unusable_retry_after_yields_none_rather_than_a_guess(value: str | None) -> None:
    assert parse_retry_after(value) is None


def test_a_future_dated_retry_after_is_measured_against_the_current_time() -> None:
    header = (datetime.now(UTC) + timedelta(seconds=45)).strftime("%a, %d %b %Y %H:%M:%S GMT")

    parsed = parse_retry_after(header)

    assert parsed is not None
    assert 40.0 <= parsed <= 46.0


# --- raw payloads -----------------------------------------------------------


def test_a_small_payload_passes_through_intact() -> None:
    assert capped_raw({"id": "x", "n": 1}) == {"id": "x", "n": 1}


def test_a_secret_shaped_key_inside_a_vendor_payload_is_redacted() -> None:
    capped = capped_raw({"echo": {"headers": {"Authorization": "Bearer abcdefghijkl"}}})

    echo = capped["echo"]
    assert isinstance(echo, dict)
    headers = echo["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "[redacted]"


def test_token_counts_are_not_mistaken_for_credentials() -> None:
    """A substring match on "token" would redact every count in the contract."""
    capped = capped_raw({"usage": {"input_tokens": 42, "total_tokens": 49}})

    assert capped["usage"] == {"input_tokens": 42, "total_tokens": 49}


def test_an_oversized_payload_is_replaced_by_a_marker_recording_its_size() -> None:
    capped = capped_raw({"blob": "x" * (MAX_RAW_BYTES + 1000)})

    assert capped[RAW_TRUNCATED_KEY] is True
    assert isinstance(capped["bytes"], int)
    assert capped["bytes"] > MAX_RAW_BYTES
    assert "blob" not in capped


def test_a_payload_that_cannot_be_serialized_is_marked_rather_than_crashing() -> None:
    capped = capped_raw({"weird": object()})

    assert capped[RAW_TRUNCATED_KEY] is True


# --- message scrubbing ------------------------------------------------------


def test_a_key_shaped_substring_in_a_vendor_message_is_redacted() -> None:
    leaked = "sk-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3"

    scrubbed = scrub_message(f"Incorrect API key provided: {leaked}")

    assert leaked not in scrubbed
    assert "[redacted]" in scrubbed


def test_a_bearer_header_echoed_back_is_redacted() -> None:
    scrubbed = scrub_message("rejected header Bearer abcdefghijklmnop")

    assert "abcdefghijklmnop" not in scrubbed


def test_a_google_style_key_is_redacted() -> None:
    leaked = "AIz" + "a" + "B" * 34

    assert leaked not in scrub_message(f"API key not valid: {leaked}")


def test_an_enormous_vendor_message_is_truncated() -> None:
    scrubbed = scrub_message("y" * (MAX_ERROR_MESSAGE_CHARS * 3))

    assert len(scrubbed) < MAX_ERROR_MESSAGE_CHARS * 2
    assert scrubbed.endswith("[truncated]")


def test_an_ordinary_message_is_left_alone() -> None:
    assert scrub_message("model not found, try pulling it first") == (
        "model not found, try pulling it first"
    )


# --- small helpers ----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (None, ProviderErrorKind.UNKNOWN),
        (401, ProviderErrorKind.AUTHENTICATION),
        (403, ProviderErrorKind.PERMISSION),
        (404, ProviderErrorKind.NOT_FOUND),
        (408, ProviderErrorKind.TIMEOUT),
        (413, ProviderErrorKind.CONTEXT_LENGTH),
        (429, ProviderErrorKind.RATE_LIMIT),
        (418, ProviderErrorKind.INVALID_REQUEST),
        (500, ProviderErrorKind.SERVER),
        (503, ProviderErrorKind.SERVER),
        (200, ProviderErrorKind.UNKNOWN),
    ],
)
def test_http_statuses_classify_as_documented(
    status: int | None, expected: ProviderErrorKind
) -> None:
    assert status_kind(status) is expected


def test_an_empty_vendor_id_is_no_id_at_all() -> None:
    assert identifier_at({"id": ""}, "id") is None
    assert identifier_at({}, "id") is None
    assert identifier_at({"id": "resp_1"}, "id") == "resp_1"
