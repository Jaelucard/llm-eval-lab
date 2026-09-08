"""The API's security boundary: bind, token, CORS, body cap, error shape.

Each test here corresponds to one control, and each asserts the ABSENCE of
something as well as the presence of the right answer. A 422 that refuses a
credential and then quotes it back is worse than no check at all, so the
assertions are on the response body, not only on the status code.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import TYPE_CHECKING, Any

import pytest
import structlog
from pydantic import SecretStr, ValidationError

from llm_eval_lab.api.app import create_app, require_loopback_or_token
from llm_eval_lab.api.errors import InsecureBindError
from llm_eval_lab.api.routers.runs import _attachment
from llm_eval_lab.api.server import build_server
from llm_eval_lab.observability.logging import configure_logging, scrub_secrets
from llm_eval_lab.providers.registry import (
    ProviderExtraRequiredError,
    ProviderInfo,
    ProviderRegistry,
)
from llm_eval_lab.services import MetricsService, run_manager
from llm_eval_lab.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path

    import httpx

    from integration.conftest import ClientFactory

PLANTED_KEY = "sk-test-DO-NOT-ECHO-987654321"
NON_LOOPBACK = "0.0.0.0"  # noqa: S104 - the address under test, never bound here
TOKEN = "correct-horse-battery-staple"  # noqa: S105 - a test token, not a credential


# -- credentials never travel, in either direction -------------------------


@pytest.mark.integration
async def test_a_credential_in_provider_options_is_refused_and_never_echoed(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    """`ProviderConfig`'s recursive validator makes the leak structurally impossible.

    The 422 is only half the control. FastAPI's own validation payload includes
    the offending ``input`` verbatim, so a handler that forwarded it would answer
    a request carrying a credential by returning that credential.
    """
    response = await api_client.post(
        "/api/runs",
        json=launch_body
        | {
            "provider": {
                "provider": "fake",
                "model": "fake-1",
                "options": {"api_key": PLANTED_KEY},
            }
        },
    )
    assert response.status_code == 422, response.text
    assert PLANTED_KEY not in response.text
    body = response.json()
    assert body["type"] == "/problems/request-invalid"
    assert any("options" in item["location"] for item in body["errors"])
    assert all(item.get("input") is None for item in body["errors"]), (
        "the offending value is dropped, not echoed back to whoever sent it"
    )


@pytest.mark.integration
async def test_a_credential_nested_deeper_in_options_is_refused_too(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    response = await api_client.post(
        "/api/runs",
        json=launch_body
        | {
            "provider": {
                "provider": "fake",
                "model": "fake-1",
                "options": {"headers": {"Authorization": f"Bearer {PLANTED_KEY}"}},
            }
        },
    )
    assert response.status_code == 422, response.text
    assert PLANTED_KEY not in response.text


# -- path traversal ---------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    [
        # A literal `../` is normalised out of the URL by httpx before the request
        # is sent, so it never reaches the resolver and would pass for the wrong
        # reason. Every form below survives to the handler as a decoded name.
        # `test_suite_paths.py` exercises the resolver directly as well.
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "/etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "%2E%2E%2Fetc%2Fpasswd",
    ],
)
async def test_a_traversing_benchmark_name_never_returns_file_content(
    api_client: httpx.AsyncClient,
    name: str,
) -> None:
    response = await api_client.get(f"/api/benchmarks/{name}")
    assert response.status_code in {400, 404}, f"{name} -> {response.status_code}"
    assert "root:" not in response.text
    assert "/bin/" not in response.text


# -- the shape of an unmapped failure --------------------------------------


@pytest.mark.integration
async def test_a_forced_internal_failure_returns_an_opaque_five_hundred(
    api_settings: Settings,
    client_for: ClientFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A traceback, a file path and an exception message all stay on the server.

    The failure is forced inside a service rather than through a route added for
    the purpose, so the response travels the same handler chain a genuine bug
    would.
    """
    internal_detail = "connection to /var/secret/socket refused for user hunter2"

    async def explode(self: MetricsService, run_id: str) -> None:
        del self, run_id
        raise RuntimeError(internal_detail)

    monkeypatch.setattr(MetricsService, "get", explode)

    async with client_for(create_app(api_settings), raise_app_exceptions=False) as client:
        response = await client.get("/api/runs/whatever/metrics")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["title"] == "Internal server error"
    assert body["correlation_id"]

    text = response.text
    assert internal_detail not in text
    assert "RuntimeError" not in text
    assert "Traceback" not in text
    assert ".py" not in text, "no source file path crosses the wire"
    assert "hunter2" not in text


# -- bind address -----------------------------------------------------------


@pytest.mark.integration
def test_binding_a_non_loopback_address_without_a_token_is_refused_at_startup() -> None:
    settings = Settings(_env_file=None, api_host=NON_LOOPBACK, api_token=None)
    with pytest.raises(InsecureBindError, match="LLM_EVAL_API_TOKEN"):
        create_app(settings)


@pytest.mark.integration
def test_binding_a_non_loopback_address_with_a_token_is_allowed() -> None:
    settings = Settings(_env_file=None, api_host=NON_LOOPBACK, api_token=SecretStr(TOKEN))
    assert create_app(settings) is not None


@pytest.mark.integration
def test_loopback_needs_no_token() -> None:
    assert create_app(Settings(_env_file=None, api_host="127.0.0.1")) is not None


# -- bearer token -----------------------------------------------------------


@pytest.fixture
def guarded_settings(api_settings: Settings) -> Settings:
    """The same API, with a bearer token configured."""
    return api_settings.model_copy(update={"api_token": SecretStr(TOKEN)})


@pytest.mark.integration
async def test_a_configured_token_guards_every_route_except_health(
    guarded_settings: Settings,
    client_for: ClientFactory,
) -> None:
    async with client_for(create_app(guarded_settings)) as client:
        assert (await client.get("/api/health")).status_code == 200

        for path in ("/api/version", "/api/providers", "/api/runs", "/api/models/summary"):
            unauthenticated = await client.get(path)
            assert unauthenticated.status_code == 401, path
            assert unauthenticated.headers["www-authenticate"] == "Bearer"
            assert TOKEN not in unauthenticated.text

        wrong = await client.get("/api/providers", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401

        malformed = await client.get("/api/providers", headers={"Authorization": TOKEN})
        assert malformed.status_code == 401

        authorised = await client.get(
            "/api/providers", headers={"Authorization": f"Bearer {TOKEN}"}
        )
        assert authorised.status_code == 200


@pytest.mark.integration
async def test_no_token_configured_means_no_gate(api_client: httpx.AsyncClient) -> None:
    """Loopback-only is the other half of the bargain; the bind check enforces it."""
    assert (await api_client.get("/api/providers")).status_code == 200


# -- CORS -------------------------------------------------------------------


@pytest.mark.integration
async def test_cors_allows_the_configured_origin_and_nothing_else(
    api_client: httpx.AsyncClient,
) -> None:
    allowed = await api_client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"

    refused = await api_client.get("/api/health", headers={"Origin": "http://evil.test"})
    assert "access-control-allow-origin" not in refused.headers

    preflight = await api_client.request(
        "OPTIONS",
        "/api/runs",
        headers={
            "Origin": "http://evil.test",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert preflight.headers.get("access-control-allow-origin") != "*"


@pytest.mark.integration
def test_settings_refuse_a_wildcard_origin_outright() -> None:
    with pytest.raises(ValidationError, match=r"must not contain"):
        Settings(_env_file=None, cors_origins=("*",))


# -- request body cap -------------------------------------------------------


@pytest.mark.integration
async def test_an_oversized_body_is_refused_before_it_is_parsed(
    api_settings: Settings,
    client_for: ClientFactory,
    launch_body: dict[str, Any],
) -> None:
    tiny = api_settings.model_copy(update={"max_request_bytes": 1024})
    async with client_for(create_app(tiny)) as client:
        response = await client.post("/api/runs", json=launch_body | {"notes": "x" * 2000})
    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["type"] == "/problems/request-body"


@pytest.mark.integration
async def test_a_body_of_undeclared_length_is_refused(
    api_settings: Settings,
    client_for: ClientFactory,
) -> None:
    """A chunked body has no length to check, and an uncapped upload is the risk."""

    async def chunks() -> Any:
        yield b'{"suite":"smoke"}'

    async with client_for(create_app(api_settings)) as client:
        response = await client.post("/api/runs", content=chunks())
    assert response.status_code == 411
    assert response.json()["type"] == "/problems/request-body"


# -- S-1: an empty token must not disable the controls it configures ---------


@pytest.mark.integration
def test_an_empty_token_does_not_permit_a_non_loopback_bind() -> None:
    """Settings normalises the empty token away, so the bind check sees no token."""
    settings = Settings(_env_file=None, api_host=NON_LOOPBACK, api_token="")
    assert settings.api_token is None
    with pytest.raises(InsecureBindError, match="LLM_EVAL_API_TOKEN"):
        create_app(settings)


@pytest.mark.integration
async def test_the_gate_fails_closed_on_an_empty_configured_token(
    api_settings: Settings,
    client_for: ClientFactory,
) -> None:
    """The second layer, independent of the first.

    `model_copy` does not re-run validators, so this injects the exact value the
    settings normalisation exists to remove. Without the explicit empty check in
    the gate, `compare_digest("", "")` would authorise a request carrying no
    `Authorization` header at all.
    """
    app = create_app(api_settings)
    app.state.settings = api_settings.model_copy(update={"api_token": SecretStr("")})

    async with client_for(app) as client:
        assert client is not None
        for headers in ({}, {"Authorization": "Bearer "}, {"Authorization": "Bearer"}):
            response = await client.get("/api/providers", headers=headers)
            assert response.status_code == 401, f"{headers} was authorised by an empty token"


# -- S-2: the bind check must see the address actually bound -----------------


@pytest.mark.integration
def test_the_server_entry_point_checks_the_host_it_is_given_not_the_setting(
    api_settings: Settings,
) -> None:
    """`create_app` can only check what it was told; `build_server` knows the truth.

    The settings say loopback. The server is handed a public address, which is
    exactly what `uvicorn --factory ... --host 0.0.0.0` does to an app built by a
    factory that never heard about the host.
    """
    assert api_settings.api_host == "127.0.0.1"
    app = create_app(api_settings)
    with pytest.raises(InsecureBindError, match=re.escape(NON_LOOPBACK)):
        build_server(app, api_settings, host=NON_LOOPBACK)


@pytest.mark.integration
def test_the_server_entry_point_permits_a_public_bind_once_a_token_is_set(
    api_settings: Settings,
) -> None:
    guarded = api_settings.model_copy(update={"api_token": SecretStr(TOKEN)})
    server = build_server(create_app(guarded), guarded, host=NON_LOOPBACK, port=8123)
    assert server.config.host == NON_LOOPBACK
    assert server.config.port == 8123
    assert server.config.workers == 1


@pytest.mark.integration
def test_the_server_entry_point_binds_the_host_it_was_given(api_settings: Settings) -> None:
    server = build_server(create_app(api_settings), api_settings, host="127.0.0.1", port=8124)
    assert server.config.host == "127.0.0.1"
    assert server.config.port == 8124


# -- S-4: a blank host is bind-all, and is refused unconditionally -----------


@pytest.mark.integration
@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_a_blank_host_is_refused_even_with_a_token_configured(
    api_settings: Settings, blank: str
) -> None:
    """An empty or whitespace host is uvicorn's own spelling of bind-all.

    It is not a bind address an operator chose, so a token being configured
    does not save it - unlike a genuine non-loopback address, this is refused
    unconditionally.
    """
    guarded = api_settings.model_copy(update={"api_token": SecretStr(TOKEN)})
    with pytest.raises(InsecureBindError):
        require_loopback_or_token(guarded, blank)


@pytest.mark.integration
def test_a_blank_host_without_a_token_is_also_refused() -> None:
    settings = Settings(_env_file=None, api_host="127.0.0.1", api_token=None)
    with pytest.raises(InsecureBindError):
        require_loopback_or_token(settings, " ")


@pytest.mark.integration
def test_a_wildcard_bind_without_a_token_is_refused(api_settings: Settings) -> None:
    with pytest.raises(InsecureBindError, match="LLM_EVAL_API_TOKEN"):
        require_loopback_or_token(api_settings, NON_LOOPBACK)


@pytest.mark.integration
def test_build_server_refuses_a_blank_host(api_settings: Settings) -> None:
    guarded = api_settings.model_copy(update={"api_token": SecretStr(TOKEN)})
    with pytest.raises(InsecureBindError):
        build_server(create_app(guarded), guarded, host="", port=8123)


@pytest.mark.integration
@pytest.mark.parametrize("bad_port", [0, -1, 70000])
def test_build_server_refuses_a_port_outside_the_valid_range(
    api_settings: Settings, bad_port: int
) -> None:
    with pytest.raises(InsecureBindError):
        build_server(create_app(api_settings), api_settings, host="127.0.0.1", port=bad_port)


# -- S-3: a hostile Authorization header must not crash the gate -------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "header",
    [
        b"Bearer \xe9token",
        b"Bearer \xff\xfe",
        b"Bearer " + b"\xe9" * 64,
        b"\xe9",
        b"Bearer \x00\x01\x02",
    ],
)
async def test_a_non_ascii_authorization_header_is_a_401_not_a_500(
    guarded_settings: Settings,
    client_for: ClientFactory,
    header: bytes,
) -> None:
    """`secrets.compare_digest` refuses str arguments above 0x7F with a TypeError.

    Starlette decodes headers as latin-1, so one such byte used to produce an
    unauthenticated 500 and a full logged traceback, at whatever rate the caller
    liked - on the one code path that must never raise.
    """
    async with client_for(create_app(guarded_settings), raise_app_exceptions=False) as client:
        response = await client.get("/api/providers", headers={b"Authorization": header})
    assert response.status_code == 401, response.text
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.integration
async def test_a_non_ascii_configured_token_still_authorises_its_own_value(
    api_settings: Settings,
    client_for: ClientFactory,
) -> None:
    """A token an operator chose in their own alphabet must work, not 500."""
    token = "clé-de-passage-très-longue"  # noqa: S105 - a test token, not a credential
    settings = api_settings.model_copy(update={"api_token": SecretStr(token)})
    # A client puts UTF-8 bytes on the wire; the comparison is of wire bytes.
    correct = b"Bearer " + token.encode("utf-8")
    wrong_value = b"Bearer " + "clé-fausse".encode()
    async with client_for(create_app(settings), raise_app_exceptions=False) as client:
        ok = await client.get("/api/providers", headers={b"Authorization": correct})
        wrong = await client.get("/api/providers", headers={b"Authorization": wrong_value})
    assert ok.status_code == 200, ok.text
    assert wrong.status_code == 401


# -- S-4: the schema and its documentation UI are guarded too ----------------


@pytest.mark.integration
async def test_the_schema_and_docs_require_the_token_when_one_is_configured(
    guarded_settings: Settings,
    client_for: ClientFactory,
) -> None:
    """The document is the whole route table of a deployment exposed off loopback."""
    authorised = {"Authorization": f"Bearer {TOKEN}"}
    async with client_for(create_app(guarded_settings)) as client:
        for path in ("/openapi.json", "/docs", "/redoc"):
            assert (await client.get(path)).status_code == 401, path
            assert (await client.get(path, headers=authorised)).status_code == 200, path

        schema = (await client.get("/openapi.json", headers=authorised)).json()
        assert "/api/runs" in schema["paths"]
        assert TOKEN not in json.dumps(schema)


@pytest.mark.integration
async def test_the_schema_stays_open_on_a_loopback_deployment(
    api_client: httpx.AsyncClient,
) -> None:
    """No token means loopback-only, and the brief's own curl check reads this path."""
    response = await api_client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/runs" in response.json()["paths"]


@pytest.mark.integration
def test_the_offline_schema_export_needs_no_server_and_no_token() -> None:
    """S-4's fix must not break the snapshot Lane D generates types from."""
    settings = Settings(_env_file=None, api_host="127.0.0.1", api_token=SecretStr(TOKEN))
    schema = create_app(settings).openapi()
    assert "/api/runs" in schema["paths"]
    assert TOKEN not in json.dumps(schema)


# -- L-1: browser-facing response headers ------------------------------------


@pytest.mark.integration
async def test_every_response_carries_nosniff(api_client: httpx.AsyncClient) -> None:
    for path in ("/api/health", "/api/providers", "/"):
        response = await api_client.get(path)
        assert response.headers["x-content-type-options"] == "nosniff", path
        assert response.headers["x-frame-options"] == "DENY", path


@pytest.mark.integration
async def test_html_carries_a_policy_that_forbids_inline_and_third_party_script(
    api_client: httpx.AsyncClient,
) -> None:
    html = await api_client.get("/")
    policy = html.headers["content-security-policy"]
    assert "default-src 'self'" in policy
    assert "script-src 'self'" in policy
    assert "'unsafe-inline'" not in policy
    assert "'unsafe-eval'" not in policy
    assert "object-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy

    # JSON is not a script host, so the policy is not spent on it.
    assert "content-security-policy" not in (await api_client.get("/api/health")).headers


# -- L-2: the run id is validated before it reaches a response header --------


@pytest.mark.integration
@pytest.mark.parametrize(
    "run_id",
    ['a"; filename="evil', "line\r\nInjected: yes", "../../etc/passwd", ""],
)
def test_an_odd_run_id_never_reaches_a_content_disposition_header(run_id: str) -> None:
    header = _attachment(run_id, "csv")
    assert header == 'attachment; filename="run-export.csv"'
    assert "\r" not in header
    assert "\n" not in header


@pytest.mark.integration
def test_a_canonical_run_id_is_used_as_the_filename() -> None:
    run_id = str(uuid.uuid4())
    assert _attachment(run_id, "json") == f'attachment; filename="run-{run_id}.json"'


# -- L-4: the scrubbing processor is installed before any request ------------


@pytest.mark.integration
def test_create_app_configures_logging_when_nothing_else_has(api_settings: Settings) -> None:
    """The "no credential in a log" guarantee is a property of a processor.

    Under `llm-eval serve` the CLI installs it. Any other entry point - a
    supervisor, a test, `uvicorn --factory` - got structlog's defaults, whose
    traceback renderer prints local variables and the ASGI scope among them,
    `Authorization` header included.
    """
    structlog.reset_defaults()
    assert not structlog.is_configured()

    create_app(api_settings)

    assert structlog.is_configured()
    processors = structlog.get_config()["processors"]
    assert scrub_secrets in processors, "the redaction processor is installed"


@pytest.mark.integration
def test_create_app_does_not_override_logging_the_cli_already_configured(
    api_settings: Settings,
) -> None:
    """`llm-eval serve` has already applied --log-level and --log-format by now."""
    configure_logging(level="DEBUG", fmt="json", log_prompts=False)
    before = structlog.get_config()["processors"]

    create_app(api_settings)

    assert structlog.get_config()["processors"] == before


# -- C1: no server file path in a validation problem detail ------------------


@pytest.mark.integration
async def test_an_invalid_suite_never_returns_the_servers_path_to_it(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
    fixtures_dir: Path,
) -> None:
    """The 422 detail is counted here, never forwarded from the exception.

    `BenchmarkValidationError`'s message is `{absolute path}: N validation
    error(s)` followed by a repr of each offending input, so forwarding it put
    both the server's file system and the file's content into the response.
    """
    response = await api_client.post(
        "/api/runs", json=launch_body | {"suite": "malformed_missing_id"}
    )
    assert response.status_code == 422, response.text
    text = response.text
    assert str(fixtures_dir) not in text
    assert str(fixtures_dir.parent) not in text
    assert "malformed_missing_id.yaml" not in text
    assert ".yaml" not in text
    assert "/Users/" not in text
    body = response.json()
    assert body["detail"].startswith("The benchmark suite has ")
    assert body["errors"], "the locations a client can act on still travel"
    assert all("input" not in item for item in body["errors"])


# -- I2: the 424 extension members name a variable and an extra --------------


@pytest.mark.integration
async def test_an_unset_credential_variable_is_a_424_naming_the_variable(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_EVAL_NOT_SET_ANYWHERE", raising=False)
    response = await api_client.post(
        "/api/runs",
        json=launch_body
        | {
            "provider": {
                "provider": "fake",
                "model": "fake-1",
                "api_key_env": "LLM_EVAL_NOT_SET_ANYWHERE",
                "options": {"mode": "expected"},
            }
        },
    )
    assert response.status_code == 424, response.text
    body = response.json()
    assert body["type"] == "/problems/credential-missing"
    assert body["env_var"] == "LLM_EVAL_NOT_SET_ANYWHERE"
    assert "not set" in body["detail"]


@pytest.mark.integration
async def test_a_provider_whose_extra_is_missing_is_a_424_naming_the_extra(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `extra` member is what makes a 424 actionable rather than merely negative."""

    def refuse(self: ProviderRegistry, name: str) -> ProviderInfo:
        del self
        msg = f"the {name!r} provider needs an optional extra that is not installed"
        raise ProviderExtraRequiredError(msg, provider=name, extra=name)

    monkeypatch.setattr(ProviderRegistry, "info", refuse)

    response = await api_client.post("/api/runs", json=launch_body)
    assert response.status_code == 424, response.text
    body = response.json()
    assert body["type"] == "/problems/provider-unavailable"
    assert body["extra"] == "fake"
    assert "not installed" in body["detail"]


@pytest.mark.integration
async def test_an_unknown_provider_is_a_424_with_no_extra_to_offer(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    """There is no extra that fixes a typo, so the member is absent rather than guessed."""
    response = await api_client.post(
        "/api/runs",
        json=launch_body | {"provider": {"provider": "no-such-vendor", "model": "x"}},
    )
    assert response.status_code == 424, response.text
    body = response.json()
    assert "extra" not in body
    assert "no-such-vendor" in body["detail"]


# -- the concurrency cap -----------------------------------------------------


@pytest.mark.integration
async def test_launching_beyond_the_process_cap_is_refused(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each launched run spends money; `POST /api/runs` in a loop must not be free."""
    monkeypatch.setattr(run_manager, "MAX_CONCURRENT_RUNS", 1)
    slow = launch_body | {
        "provider": {
            "provider": "fake",
            "model": "fake-1",
            "options": {
                "mode": "expected",
                "sleep": True,
                "latency_min_ms": 400.0,
                "latency_max_ms": 400.0,
            },
        },
        "concurrency": 1,
    }
    first = await api_client.post("/api/runs", json=slow)
    assert first.status_code == 202, first.text

    second = await api_client.post("/api/runs", json=slow)
    assert second.status_code == 429, second.text
    assert second.json()["type"] == "/problems/too-many-runs"


# -- token-protected mode does not lock the dashboard's own door out --------


@pytest.mark.integration
async def test_the_dashboard_shell_stays_reachable_without_a_token(
    guarded_settings: Settings,
    client_for: ClientFactory,
) -> None:
    """`/` is mounted outside the `/api` guard, so the page can always load.

    The bundled dashboard never holds or sends the bearer token (see
    `README.md#the-dashboard`) - it is a static file served unauthenticated -
    while the data it tries to fetch from `/api/*` still needs one. This is
    what lets the frontend detect token mode and explain it instead of
    failing to load at all.
    """
    async with client_for(create_app(guarded_settings)) as client:
        shell = await client.get("/")
        assert shell.status_code == 200

        health_response = await client.get("/api/health")
        assert health_response.status_code == 200

        unauthenticated = await client.get("/api/version")
        assert unauthenticated.status_code == 401
        assert unauthenticated.headers["www-authenticate"] == "Bearer"

        authorised = await client.get("/api/version", headers={"Authorization": f"Bearer {TOKEN}"})
        assert authorised.status_code == 200
