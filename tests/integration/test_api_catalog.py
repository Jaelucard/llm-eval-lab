"""The read-only catalog: providers, evaluators, prices and benchmark suites."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import httpx

CANARY_VALUE = "sk-canary-DO-NOT-LEAK-0123456789"
"""The same planted value `test_secret_leak.py` uses, asserted against here too."""


@pytest.mark.integration
async def test_providers_lists_fake_and_returns_a_variable_name_not_a_value(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A credential VALUE has no field to travel in, and this proves it stays true.

    A real-looking key is planted in the environment under the variable name the
    fake provider would use if it needed one, so a response that resolved a
    credential rather than reporting its name would carry the canary.
    """
    monkeypatch.setenv("LLM_EVAL_CANARY_KEY", CANARY_VALUE)

    response = await api_client.get("/api/providers")
    assert response.status_code == 200, response.text
    rows = response.json()
    by_name = {row["name"]: row for row in rows}
    assert "fake" in by_name

    fake = by_name["fake"]
    assert fake["available"] is True
    assert fake["requires_credential"] is False
    assert fake["credential_env"] is None
    assert fake["credential_resolved"] is None
    assert "fake-1" in fake["models"]

    body = response.text
    assert CANARY_VALUE not in body
    for row in rows:
        assert set(row) == {
            "name",
            "available",
            "requires_credential",
            "credential_env",
            "credential_resolved",
            "extra",
            "summary",
            "models",
        }, "no field on this row could hold a credential value"


@pytest.mark.integration
async def test_provider_availability_reflects_the_sdk_and_the_credential_separately(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`available` is false when EITHER the SDK is missing OR the credential is.

    Before this, `available` was computed from credential status alone, so a
    provider whose optional extra was never installed still reported
    `available: true` as long as nobody happened to have set its credential
    variable - the exact opposite of a useful "can this build reach it" signal.
    Every provider this project ships an extra for has that extra installed in
    this environment, so a real, un-stubbed check that the credential variable
    alone is not read as availability is exercised here rather than faked.
    """
    for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    response = await api_client.get("/api/providers")
    assert response.status_code == 200, response.text
    by_name = {row["name"]: row for row in response.json()}

    for name in ("openai", "anthropic", "google"):
        row = by_name[name]
        assert row["credential_resolved"] is False, name
        assert row["available"] is False, (
            f"{name}'s SDK is installed but its credential is not; 'available' must say so"
        )

    fake = by_name["fake"]
    assert fake["available"] is True
    assert fake["credential_resolved"] is None


@pytest.mark.integration
async def test_evaluators_carry_the_json_schema_of_their_parameters(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.get("/api/evaluators")
    assert response.status_code == 200, response.text
    rows = {row["type"]: row for row in response.json()}
    assert {"exact_match", "contains", "regex", "numeric_tolerance"} <= set(rows)

    regex = rows["regex"]
    assert regex["needs_expected"] is False
    assert regex["is_model_graded"] is False
    assert "pattern" in regex["params_schema"]["properties"]


@pytest.mark.integration
async def test_pricing_returns_the_table_identity_and_its_content_hash(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.get("/api/pricing")
    assert response.status_code == 200, response.text
    table = response.json()
    assert table["id"]
    assert table["version"]
    assert table["content_hash"].startswith("sha256:")
    assert isinstance(table["models"], list)


@pytest.mark.integration
async def test_benchmarks_lists_valid_and_invalid_suites_and_hides_neither(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.get("/api/benchmarks")
    assert response.status_code == 200, response.text
    assert response.headers["x-listing-truncated"] == "false", (
        "the fixture root fits in one listing"
    )
    page = response.json()
    rows = {row["name"]: row for row in page["items"]}
    assert page["total"] == len(page["items"]), "the fixture root fits in one listing"
    assert page["offset"] == 0

    assert "smoke.yaml" in rows
    smoke = rows["smoke.yaml"]
    assert smoke["valid"] is True
    assert smoke["suite_name"] == "smoke"
    assert smoke["n_cases"] == 6
    assert smoke["suite_hash"].startswith("sha256:")

    broken = rows["malformed_missing_id.yaml"]
    assert broken["valid"] is False
    assert broken["n_errors"] >= 1
    assert broken["suite_hash"] is None


@pytest.mark.integration
async def test_benchmarks_reports_truncation_as_a_header_not_only_via_total(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`X-Listing-Truncated` is a plain yes/no, so a caller need not compare
    `total` against `len(items)` itself to learn "there are more than these"."""
    from llm_eval_lab.services import catalog_service  # noqa: PLC0415 - reads next to its use

    monkeypatch.setattr(catalog_service, "MAX_LISTED_SUITES", 2)

    response = await api_client.get("/api/benchmarks")
    assert response.status_code == 200, response.text
    assert response.headers["x-listing-truncated"] == "true"
    page = response.json()
    assert len(page["items"]) == 2
    assert page["total"] > len(page["items"])


@pytest.mark.integration
async def test_one_benchmark_returns_the_resolved_suite_and_no_server_path(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.get("/api/benchmarks/smoke")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["name"] == "smoke"
    assert payload["suite"]["name"] == "smoke"
    assert len(payload["suite"]["cases"]) == 6
    assert payload["suite_hash"] == payload["suite"]["suite_hash"]
    assert "/tmp" not in response.text  # noqa: S108 - asserting a server path is ABSENT
    assert "suites/" not in response.text


@pytest.mark.integration
async def test_validate_answers_two_hundred_for_a_file_that_does_not_load(
    api_client: httpx.AsyncClient,
) -> None:
    """ "Is it valid" is a question, and "no" is a successful answer to it."""
    good = await api_client.post("/api/benchmarks/validate", json={"name": "smoke"})
    assert good.status_code == 200, good.text
    assert good.json() == {"name": "smoke", "valid": True, "errors": []}

    bad = await api_client.post(
        "/api/benchmarks/validate", json={"name": "malformed_bad_evaluator"}
    )
    assert bad.status_code == 200, bad.text
    payload = bad.json()
    assert payload["valid"] is False
    assert payload["errors"]
    for item in payload["errors"]:
        assert set(item) <= {"location", "case_id", "message"}
        assert "input" not in item, (
            "the offending value is benchmark-file content and does not cross the wire"
        )
        assert item["location"]


@pytest.mark.integration
async def test_a_suite_name_that_names_nothing_is_a_four_oh_four(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.get("/api/benchmarks/no-such-suite")
    assert response.status_code == 404
    assert response.json()["type"] == "/problems/suite-not-found"
