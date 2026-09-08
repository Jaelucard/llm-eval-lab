"""``llm-eval serve``: the CLI must not let an invalid override reach a socket.

`serve` rebuilds `Settings` from `--host`/`--port` overrides through a
validated path (`Settings.model_validate`, not `model_copy`), so a value a
field validator would refuse - a blank host, an out-of-range port - has to be
refused here too, before `create_app` or `run_server` ever sees it. Every test
that expects a refusal also asserts `run_server` was never called: a `refusing
to start` message next to a socket that opened anyway would be worse than no
message at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from llm_eval_lab.cli.main import ExitCode, app
from llm_eval_lab.settings import get_settings

runner = CliRunner()
DEFAULT_PORT = 8000
"""The documented default `api_port`."""


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, db_url: str) -> None:
    """Point every invocation at a test-private database, never the operator's."""
    monkeypatch.setenv("LLM_EVAL_DATABASE_URL", db_url)
    get_settings.cache_clear()


@pytest.fixture
def run_server_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace `run_server` with a recorder, so a test can assert it did or did not run.

    Patched on `llm_eval_lab.api.server`, the module `serve()` imports it from
    at call time, so the deferred `from llm_eval_lab.api.server import
    run_server` inside the command picks up this replacement.
    """
    calls: list[dict[str, Any]] = []

    def fake_run_server(application: object, settings: object, **kwargs: Any) -> None:
        calls.append({"application": application, "settings": settings, **kwargs})

    monkeypatch.setattr("llm_eval_lab.api.server.run_server", fake_run_server)
    return calls


def _fail_if_called(*_args: object, **_kwargs: object) -> None:
    msg = "run_server must not be called when the bind is refused"
    raise AssertionError(msg)


@pytest.fixture
def run_server_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if `serve` ever reaches `run_server` despite an invalid override."""
    monkeypatch.setattr("llm_eval_lab.api.server.run_server", _fail_if_called)


# -- an invalid override is refused before anything listens -----------------


@pytest.mark.integration
def test_serve_refuses_a_blank_host(run_server_must_not_run: None) -> None:
    del run_server_must_not_run
    result = runner.invoke(app, ["serve", "--host", ""])
    assert result.exit_code == ExitCode.USAGE, result.output
    assert "refusing to start" in result.output


@pytest.mark.integration
@pytest.mark.parametrize("bad_port", ["0", "-1", "70000"])
def test_serve_refuses_a_port_outside_the_valid_range(
    run_server_must_not_run: None, bad_port: str
) -> None:
    del run_server_must_not_run
    result = runner.invoke(app, ["serve", "--port", bad_port])
    assert result.exit_code == ExitCode.USAGE, result.output
    assert "refusing to start" in result.output


@pytest.mark.integration
def test_serve_refuses_a_whitespace_host(run_server_must_not_run: None) -> None:
    del run_server_must_not_run
    result = runner.invoke(app, ["serve", "--host", " "])
    assert result.exit_code == ExitCode.USAGE, result.output
    assert "refusing to start" in result.output


# -- a valid override still reaches run_server -------------------------------


@pytest.mark.integration
def test_serve_passes_a_valid_port_override_through_to_run_server(
    run_server_spy: list[dict[str, Any]],
) -> None:
    result = runner.invoke(app, ["serve", "--port", "8123"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    assert len(run_server_spy) == 1
    assert run_server_spy[0]["port"] == 8123
    assert run_server_spy[0]["host"] == "127.0.0.1"


@pytest.mark.integration
def test_serve_with_no_overrides_still_reaches_run_server(
    run_server_spy: list[dict[str, Any]],
) -> None:
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == ExitCode.SUCCESS, result.output
    assert len(run_server_spy) == 1
    assert run_server_spy[0]["host"] == "127.0.0.1"
    assert run_server_spy[0]["port"] == DEFAULT_PORT
