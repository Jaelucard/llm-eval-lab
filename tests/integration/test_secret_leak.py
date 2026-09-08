"""Canary test: a credential value must not reach a log, a row, or the CLI.

The setup is deliberately the worst case. A real-looking key is put in the
environment, a provider is configured to name that variable, logging is turned
up to DEBUG with ``log_prompts`` explicitly enabled, and a full run is executed.
The literal canary string must then appear in NONE of the places a leak would
show up.

The file grows with the surface. ``llm-eval metrics --json`` and ``llm-eval
config show`` are asserted over the same planted canary, and so is every route
of the HTTP API, including the body of a deliberately forced 500 - an error path
is where a value scrubbed everywhere else tends to reappear. A later slice adds
every real provider's response bodies.

The URL redactor is unit-tested here too rather than beside the other
formatters, because it is the same control: a credential value hiding inside a
connection string, where the key-name scrubber above cannot see it.
"""

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import structlog
from pydantic import SecretStr

from llm_eval_lab.models import CaseQuery, RunStatus, UnitOfWorkFactory
from llm_eval_lab.observability.logging import configure_logging
from llm_eval_lab.redaction import REDACTED, redact_setting_value, redact_url
from llm_eval_lab.reporting.formatters import case_rows, metrics_document
from llm_eval_lab.services import (
    RunRequest,
    build_metrics_service,
    build_run_service,
    compute_pass_rate,
)
from llm_eval_lab.settings import Settings

CANARY_ENV_VAR = "LLM_EVAL_CANARY_KEY"
CANARY_VALUE = "sk-canary-DO-NOT-LEAK-0123456789"


@pytest.fixture
def captured_logs(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Route the real structlog pipeline into a buffer at DEBUG with prompts on.

    `sys.stderr` is replaced BEFORE `configure_logging` runs, because the
    structlog logger factory binds the stream it was given at configuration
    time. This exercises the genuine processor chain, scrubbing included, rather
    than a capture shim that would bypass the very processor under test.
    """
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    configure_logging(level="DEBUG", fmt="json", log_prompts=True, colors=False)
    return buffer


@pytest.mark.integration
async def test_the_canary_never_leaves_the_environment(  # noqa: PLR0913, PLR0917 - one fixture per place a leak could surface
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_logs: io.StringIO,
) -> None:
    monkeypatch.setenv(CANARY_ENV_VAR, CANARY_VALUE)

    service = build_run_service(settings, uow_factory, logger)
    request = RunRequest(
        suite_path=fixtures_dir / "suites" / "smoke.yaml",
        provider="fake",
        model="fake-1",
        api_key_env=CANARY_ENV_VAR,
        provider_options={"mode": "expected"},
        concurrency=2,
    )
    outcome = await service.execute(request)
    assert outcome.run.status is RunStatus.COMPLETED

    view = await service.get_run(outcome.run.id)
    assert view is not None

    # 1. captured log output, at DEBUG, with prompt logging enabled
    assert CANARY_VALUE not in captured_logs.getvalue()

    # 2. the SQLite file, read as raw bytes
    database_bytes = (tmp_path / "test.db").read_bytes()
    assert CANARY_VALUE.encode() not in database_bytes

    # 3. what `llm-eval show --json` would print
    shown = json.dumps(
        {
            "run_id": view.run.id,
            "status": view.run.status.value,
            "config": view.run.config.model_dump(mode="json"),
            "pass_rate": view.pass_rate,
        },
        default=str,
    )
    assert CANARY_VALUE not in shown

    # 4. the persisted RunConfig itself
    assert CANARY_VALUE not in view.run.config.model_dump_json()

    # 5. every persisted case result, response and evaluation
    cases = await service.list_cases(outcome.run.id, CaseQuery(limit=100))
    for case in cases.items:
        assert CANARY_VALUE not in case.model_dump_json()

    # 6. what `llm-eval metrics --json` would print, including the export rows
    metrics_service = build_metrics_service(uow_factory, logger)
    metrics_view = await metrics_service.get(outcome.run.id)
    assert metrics_view is not None
    assert CANARY_VALUE not in json.dumps(metrics_document(metrics_view.metrics), default=str)
    exported = await metrics_service.cases(outcome.run.id)
    assert CANARY_VALUE not in json.dumps(case_rows(exported), default=str)


@pytest.mark.integration
async def test_a_canary_planted_in_a_benchmark_file_never_reaches_the_database(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    """C1: the snapshot body is a second write path for the same free-form data.

    ``runs.config`` and ``benchmark_snapshots.body`` both carry a suite's
    ``params.extra`` and each evaluator's ``params``. Scrubbing only the first
    left a credential pasted into a benchmark file sitting in plain text two
    tables away, which the raw-bytes assertion below is the only thing that
    catches.
    """
    service = build_run_service(settings, uow_factory, logger)
    request = RunRequest(
        suite_path=fixtures_dir / "suites" / "secretish.yaml",
        provider="fake",
        model="fake-1",
        provider_options={"mode": "expected"},
    )
    outcome = await service.execute(request)
    assert outcome.run.status is RunStatus.COMPLETED

    database_bytes = (tmp_path / "test.db").read_bytes()
    assert CANARY_VALUE.encode() not in database_bytes, (
        "the canary reached the database through the snapshot body or the run config"
    )

    async with uow_factory() as uow:
        snapshot = await uow.benchmarks.get_snapshot(outcome.run.config.suite_hash)
    assert snapshot is not None
    body = json.dumps(snapshot.body)
    assert CANARY_VALUE not in body
    assert REDACTED in body, "the secret-shaped keys must be present but redacted"

    # The non-secret sibling key in the same dictionary is untouched, so the
    # scrubber is redacting a value rather than discarding the structure.
    assert "max_output_tokens_hint" in body


@pytest.mark.integration
def test_the_environment_variable_name_is_still_recorded(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording the NAME is the design, not an oversight.

    A run has to be reproducible, and knowing which variable to set is exactly
    what makes it so. The scrubber must therefore keep `api_key_env` while
    dropping anything that could hold the value.
    """
    from llm_eval_lab.redaction import scrub_mapping  # noqa: PLC0415 - reads next to its use

    monkeypatch.setenv(CANARY_ENV_VAR, CANARY_VALUE)
    scrubbed = scrub_mapping(
        {"api_key_env": CANARY_ENV_VAR, "api_key": CANARY_VALUE, "max_output_tokens": 64}
    )
    assert scrubbed["api_key_env"] == CANARY_ENV_VAR
    assert scrubbed["api_key"] != CANARY_VALUE
    assert scrubbed["max_output_tokens"] == 64, (
        "a token COUNT is not a credential; redacting it corrupts the row"
    )
    del settings


@pytest.mark.integration
def test_pass_rate_is_none_rather_than_zero_when_nothing_is_eligible() -> None:
    assert compute_pass_rate([]) == (0, 0, None)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            f"postgresql+asyncpg://admin:{CANARY_VALUE}@db.example.test:5432/evals",
            "postgresql+asyncpg://admin:[redacted]@db.example.test:5432/evals",
        ),
        (
            f"postgresql://u:{CANARY_VALUE}@Host.EXAMPLE.test/db?sslmode=require",
            "postgresql://u:[redacted]@Host.EXAMPLE.test/db?sslmode=require",
        ),
        # A password containing an unencoded "@": the LAST one separates the
        # userinfo from the host, so splitting on the first would leak the tail.
        ("postgresql://u:pa@ss@host/db", "postgresql://u:[redacted]@host/db"),
    ],
)
def test_redact_url_removes_the_password_and_keeps_everything_else(url: str, expected: str) -> None:
    redacted = redact_url(url)
    assert redacted == expected
    assert CANARY_VALUE not in redacted


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # A password in the QUERY STRING, which several drivers use instead of
        # the userinfo and which userinfo-only redaction sailed straight past.
        (
            f"postgresql://host:5432/evals?sslmode=require&password={CANARY_VALUE}",
            "postgresql://host:5432/evals?sslmode=require&password=***",
        ),
        # libpq's own spelling.
        (
            f"postgresql://host/evals?sslpassword={CANARY_VALUE}",
            "postgresql://host/evals?sslpassword=***",
        ),
        # The hosted-database spellings.
        (f"postgresql://host/db?token={CANARY_VALUE}", "postgresql://host/db?token=***"),
        (f"postgresql://host/db?api_key={CANARY_VALUE}", "postgresql://host/db?api_key=***"),
        (f"postgresql://host/db?secret={CANARY_VALUE}", "postgresql://host/db?secret=***"),
        (
            f"postgresql://host/db?authorization={CANARY_VALUE}",
            "postgresql://host/db?authorization=***",
        ),
        # Key matching is case-insensitive, like the key-name scrubber above.
        (f"postgresql://host/db?PASSWORD={CANARY_VALUE}", "postgresql://host/db?PASSWORD=***"),
        # Both halves at once, and every non-secret parameter untouched.
        (
            (
                f"postgresql://admin:{CANARY_VALUE}@host:5432/db"
                f"?sslmode=require&password={CANARY_VALUE}&region=eu-west-1"
            ),
            (
                "postgresql://admin:[redacted]@host:5432/db"
                "?sslmode=require&password=***&region=eu-west-1"
            ),
        ),
    ],
)
def test_redact_url_removes_a_password_from_the_query_string(url: str, expected: str) -> None:
    redacted = redact_url(url)
    assert redacted == expected
    assert CANARY_VALUE not in redacted


def test_a_query_parameter_naming_a_credential_variable_is_left_readable() -> None:
    # `api_key_env` records a variable NAME, which is the whole point of
    # `ProviderConfig`. The `_env` suffix exemption that protects it in the
    # key-name scrubber has to protect it here too.
    url = "postgresql://host/db?api_key_env=OPENAI_API_KEY"
    assert redact_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "sqlite+aiosqlite:////var/lib/llm-eval-lab/lab.db",
        "postgresql://user@host/db",
        "postgresql://host/db?sslmode=require&connect_timeout=10",
        "not a url at all",
        "",
    ],
)
def test_redact_url_leaves_a_url_with_no_password_alone(url: str) -> None:
    # Mangling the one setting people actually read would be its own defect.
    assert redact_url(url) == url


def test_redact_url_fails_closed_on_an_unparseable_url() -> None:
    # A URL that will not parse cannot be reasoned about, and the only safe
    # direction for a function whose job is to stop a value being printed is to
    # print nothing of it.
    assert redact_url("http://[::1:80/x") == REDACTED


def test_redact_setting_value_never_unwraps_a_secret() -> None:
    assert redact_setting_value(SecretStr(CANARY_VALUE)) is None


def test_redact_setting_value_redacts_urls_inside_a_sequence() -> None:
    rendered = redact_setting_value([f"https://u:{CANARY_VALUE}@origin.test"])
    assert CANARY_VALUE not in str(rendered)


# ---------------------------------------------------------------------------
# Phase 5 extension: the canary must not reach ANY HTTP response body.
#
# The setup plants the value twice over. It is in the environment under the
# variable the run names, and it is written into the benchmark file the run
# executes, in both free-form dictionaries a benchmark author controls. Every
# route in the API surface is then called and its body checked, including the
# 500 produced by a deliberately broken service, because an error path is where
# a value that is scrubbed everywhere else tends to reappear.
# ---------------------------------------------------------------------------

CANARY_SUITE = "secretish"
"""The fixture suite carrying the canary in `params.extra` and in case metadata."""


async def _await_terminal(client: Any, run_id: str) -> None:
    """Poll one run to a terminal status, so every read-side route has data."""
    import asyncio  # noqa: PLC0415 - used only by this helper

    terminal = {"completed", "partial", "failed", "cancelled", "interrupted"}
    for _ in range(200):
        status = (await client.get(f"/api/runs/{run_id}/status")).json()
        if status["status"] in terminal:
            return
        await asyncio.sleep(0.05)
    pytest.fail(f"run {run_id} did not finish")


@pytest.mark.integration
async def test_the_canary_reaches_no_api_response_body(
    api_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CANARY_ENV_VAR, CANARY_VALUE)

    created = await api_client.post(
        "/api/runs",
        json={
            "suite": CANARY_SUITE,
            "provider": {
                "provider": "fake",
                "model": "fake-1",
                "api_key_env": CANARY_ENV_VAR,
                "options": {"mode": "expected"},
            },
            "label": "canary",
            "notes": "planted",
        },
    )
    assert created.status_code == 202, created.text
    assert CANARY_VALUE not in created.text
    run_id = created.json()["run_id"]
    await _await_terminal(api_client, run_id)

    gets = [
        "/api/health",
        "/api/version",
        "/api/providers",
        "/api/evaluators",
        "/api/pricing",
        "/api/benchmarks",
        f"/api/benchmarks/{CANARY_SUITE}",
        "/api/runs",
        f"/api/runs/{run_id}",
        f"/api/runs/{run_id}/status",
        f"/api/runs/{run_id}/cases",
        f"/api/runs/{run_id}/cases/planted",
        f"/api/runs/{run_id}/metrics",
        f"/api/runs/{run_id}/export?format=json",
        f"/api/runs/{run_id}/export?format=csv",
        f"/api/compare?baseline_run_id={run_id}&candidate_run_id={run_id}",
        "/api/models/summary",
    ]
    for path in gets:
        response = await api_client.get(path)
        assert response.status_code < 500, f"{path} -> {response.status_code}: {response.text}"
        assert CANARY_VALUE not in response.text, f"the canary leaked from {path}"

    posts: list[tuple[str, dict[str, Any]]] = [
        ("/api/benchmarks/validate", {"name": CANARY_SUITE}),
        ("/api/compare", {"baseline_run_id": run_id, "candidate_run_id": run_id}),
    ]
    for path, body in posts:
        response = await api_client.post(path, json=body)
        assert CANARY_VALUE not in response.text, f"the canary leaked from POST {path}"

    cancelled = await api_client.post(f"/api/runs/{run_id}/cancel")
    assert CANARY_VALUE not in cancelled.text

    # The suite body IS returned, scrubbed rather than withheld: the endpoint is
    # still useful, and the marker proves the scrubber ran rather than the file
    # simply lacking the value.
    suite = await api_client.get(f"/api/benchmarks/{CANARY_SUITE}")
    assert suite.json()["redacted"] is True
    assert REDACTED in suite.text
    assert "max_output_tokens_hint" in suite.text


@pytest.mark.integration
async def test_the_canary_reaches_no_forced_five_hundred_body(
    api_settings: Any,
    client_for: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error path gets the same treatment as the success path.

    The exception carries the canary in its own message, which is the worst case:
    a handler that reported `str(exc)` would hand it straight back.
    """
    from llm_eval_lab.api.app import create_app  # noqa: PLC0415 - reads next to its use
    from llm_eval_lab.services import MetricsService  # noqa: PLC0415 - reads next to its use

    monkeypatch.setenv(CANARY_ENV_VAR, CANARY_VALUE)

    message = f"upstream rejected {CANARY_VALUE}"

    async def explode(self: MetricsService, run_id: str) -> None:
        del self, run_id
        raise RuntimeError(message)

    monkeypatch.setattr(MetricsService, "get", explode)

    async with client_for(create_app(api_settings), raise_app_exceptions=False) as client:
        response = await client.get("/api/runs/anything/metrics")

    assert response.status_code == 500
    assert CANARY_VALUE not in response.text
    assert "RuntimeError" not in response.text
    assert response.json()["correlation_id"]


CANARY_INVALID_SUITE = "malformed_leaky"
"""A suite that does NOT load, carrying the canary where a parent field fails.

`secretish.yaml` covers the success path only, so the canary sweep above could
never take the error path at all. Pydantic reports a missing required field at
the location of the FIELD and offers the whole PARENT object as the input, so a
case omitting `id` with a credential in its `metadata` used to echo that
credential back - twice per response, in `detail` and in `errors[].input`, on a
200 from the validate endpoint as well as on a 422 from the launch endpoint.
"""


@pytest.mark.integration
async def test_the_canary_in_an_invalid_suite_reaches_no_response_body(
    api_client: Any,
) -> None:
    validated = await api_client.post(
        "/api/benchmarks/validate", json={"name": CANARY_INVALID_SUITE}
    )
    assert validated.status_code == 200, validated.text
    payload = validated.json()
    assert payload["valid"] is False
    assert payload["errors"], "the locations a client can act on still travel"
    assert CANARY_VALUE not in validated.text, "the canary leaked through the validate endpoint"

    launched = await api_client.post(
        "/api/runs",
        json={
            "suite": CANARY_INVALID_SUITE,
            "provider": {"provider": "fake", "model": "fake-1", "options": {"mode": "expected"}},
        },
    )
    assert launched.status_code == 422, launched.text
    assert CANARY_VALUE not in launched.text, "the canary leaked through the launch endpoint"
    assert "sk-canary" not in launched.text
    assert "metadata" not in launched.json()["detail"]

    listed = await api_client.get("/api/benchmarks")
    assert CANARY_VALUE not in listed.text
    entry = next(
        row for row in listed.json()["items"] if row["name"] == f"{CANARY_INVALID_SUITE}.yaml"
    )
    assert entry["valid"] is False
    assert entry["n_errors"] >= 1


@pytest.mark.integration
async def test_the_invalid_suite_fixture_really_does_carry_the_canary(
    fixtures_dir: Path,
) -> None:
    """Guard against the fixture being edited into one that proves nothing.

    A canary test whose canary has been removed passes for free. This asserts the
    value is in the file on disk, so the two tests above are testing what they
    say they are.
    """
    body = (fixtures_dir / "suites" / f"{CANARY_INVALID_SUITE}.yaml").read_text(encoding="utf-8")
    assert CANARY_VALUE in body
    assert "id:" not in body, "the case must still be missing its required id"


@pytest.mark.integration
async def test_no_response_body_carries_the_servers_path_to_the_suites(
    api_client: Any,
    api_settings: Any,
) -> None:
    """SM5: the API tells a client what a suite IS, never where it lives.

    `RunConfig.suite_source` is the absolute path a run was launched from. It is
    right in the database, where it makes a historical run traceable, and right
    in the CLI, which is already standing in that file system. Returned over HTTP
    it describes the server's directory layout to whoever asked, and
    `GET /api/runs/{id}` returned it verbatim inside `run.config`.

    The sweep is over the configured root string rather than over one field, so a
    second route that starts carrying a path fails this test rather than waiting
    for the next review.
    """
    root = str(api_settings.suites_root)
    assert root.startswith("/"), "the fixture configures an absolute suites root"
    # The markers of a SERVER path, as opposed to a suite name or a URL path.
    # `/api/runs` is a URL and `smoke.yaml` is the name a client addresses a
    # suite by, so neither is a leak; an absolute directory the client never
    # named is.
    forbidden = (root, str(api_settings.suites_root.parent), str(api_settings.data_dir))

    created = await api_client.post(
        "/api/runs",
        json={
            "suite": "smoke",
            "provider": {"provider": "fake", "model": "fake-1", "options": {"mode": "expected"}},
            "label": "path-sweep",
        },
    )
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]
    await _await_terminal(api_client, run_id)

    paths = [
        "/api/health",
        "/api/version",
        "/api/providers",
        "/api/evaluators",
        "/api/pricing",
        "/api/benchmarks",
        "/api/benchmarks/smoke",
        "/api/runs",
        f"/api/runs/{run_id}",
        f"/api/runs/{run_id}/status",
        f"/api/runs/{run_id}/cases",
        f"/api/runs/{run_id}/cases/capital-exact",
        f"/api/runs/{run_id}/metrics",
        f"/api/runs/{run_id}/export?format=json",
        f"/api/runs/{run_id}/export?format=csv",
        f"/api/compare?baseline_run_id={run_id}&candidate_run_id={run_id}",
        "/api/models/summary",
        "/openapi.json",
    ]
    for path in paths:
        response = await api_client.get(path)
        assert response.status_code < 500, f"{path} -> {response.status_code}"
        for marker in forbidden:
            assert marker not in response.text, f"{marker} leaked from {path}"
        assert "/Users/" not in response.text, f"an absolute path leaked from {path}"

    for path, body in (
        ("/api/benchmarks/validate", {"name": "smoke"}),
        ("/api/compare", {"baseline_run_id": run_id, "candidate_run_id": run_id}),
    ):
        response = await api_client.post(path, json=body)
        for marker in forbidden:
            assert marker not in response.text, f"{marker} leaked from POST {path}"


@pytest.mark.integration
async def test_the_run_detail_reports_the_suite_by_name_and_digest_not_by_path(
    api_client: Any,
) -> None:
    """Removing the path must not remove the ability to say which suite ran."""
    created = await api_client.post(
        "/api/runs",
        json={
            "suite": "smoke",
            "provider": {"provider": "fake", "model": "fake-1", "options": {"mode": "expected"}},
        },
    )
    run_id = created.json()["run_id"]
    await _await_terminal(api_client, run_id)

    detail = await api_client.get(f"/api/runs/{run_id}")
    assert detail.status_code == 200, detail.text
    config = detail.json()["run"]["config"]

    assert config["suite_source"] is None
    assert config["suite_name"] == "smoke"
    assert config["suite_version"] == "1"
    assert config["suite_hash"].startswith("sha256:")


@pytest.mark.integration
async def test_the_stored_run_still_records_where_it_came_from(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    logger: structlog.BoundLogger,
    fixtures_dir: Path,
) -> None:
    """The projection is at the HTTP boundary, not a redaction of the record.

    A historical run has to stay traceable to the file it was launched from, and
    the CLI is already standing in that file system. Only the API withholds it.
    """
    service = build_run_service(settings, uow_factory, logger)
    outcome = await service.execute(
        RunRequest(
            suite_path=fixtures_dir / "suites" / "smoke.yaml",
            provider="fake",
            model="fake-1",
            provider_options={"mode": "expected"},
        )
    )
    stored = await service.fetch_run(outcome.run.id)
    assert stored is not None
    assert stored.config.suite_source is not None
    assert stored.config.suite_source.endswith("smoke.yaml")
