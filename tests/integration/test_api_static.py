"""What `/` serves: the bundled dashboard when there is one, an explanation when not.

The mounted case matters more than it looks. The dashboard mount is a catch-all
at `/`, so the thing worth proving is not that it serves `index.html` - it is
that it never answers for a path under `/api`. A dashboard that returned its own
HTML shell for an unknown API route would turn every client's typo into an
unparseable success.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llm_eval_lab.api import app as app_module
from llm_eval_lab.api.app import create_app

if TYPE_CHECKING:
    from pathlib import Path

    import httpx

    from integration.conftest import ClientFactory
    from llm_eval_lab.settings import Settings

SHELL = "<!doctype html><title>dashboard</title><div id=root></div>"


@pytest.fixture
def bundled_dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pretend the frontend build hook has populated the package's static directory."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(SHELL, encoding="utf-8")
    (static / "assets").mkdir()
    (static / "assets" / "app.js").write_text("console.log('hi')\n", encoding="utf-8")
    monkeypatch.setattr(app_module, "STATIC_DIR", static)
    return static


@pytest.fixture(params=["absent", "present"])
def static_dir_state(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Point `STATIC_DIR` at a private, temp directory this test controls fully.

    A test that instead reads the package's REAL `api/static/` - as
    `api_client` does, built from a module-level `STATIC_DIR` no fixture here
    overrides - depends on whether a frontend build happens to be bundled at
    the moment it runs, rather than on the code path it exists to check. This
    checkout genuinely carries a built dashboard some of the time (the hatch
    build hook copies `frontend/dist` into that exact directory), which is
    precisely what made the "no build present" assumption false and broke
    both tests using this fixture. Parametrising over a temp directory removes
    that dependency and exercises both of `_install_dashboard`'s branches from
    a controlled starting point instead.
    """
    state: str = request.param
    static = tmp_path / "controlled-static"
    if state == "present":
        static.mkdir()
        (static / "index.html").write_text(SHELL, encoding="utf-8")
    monkeypatch.setattr(app_module, "STATIC_DIR", static)
    return state


@pytest.mark.integration
async def test_without_a_build_the_root_explains_how_to_make_one(
    api_settings: Settings,
    client_for: ClientFactory,
    static_dir_state: str,
) -> None:
    """Checked in both states: absent explains itself, present serves the dashboard."""
    async with client_for(create_app(api_settings)) as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    if static_dir_state == "present":
        assert SHELL in response.text
        return
    assert "npm run build" in response.text
    assert "/openapi.json" in response.text


@pytest.mark.integration
async def test_a_bundled_dashboard_is_served_and_survives_a_deep_link(
    api_settings: Settings,
    client_for: ClientFactory,
    bundled_dashboard: Path,
) -> None:
    del bundled_dashboard
    async with client_for(create_app(api_settings)) as client:
        root = await client.get("/")
        assert root.status_code == 200
        assert SHELL in root.text

        asset = await client.get("/assets/app.js")
        assert asset.status_code == 200
        assert "console.log" in asset.text

        # A client-side route is not a file on disk; the shell has to answer it
        # or a reload on a deep link would 404.
        deep = await client.get("/runs/abc123")
        assert deep.status_code == 200
        assert SHELL in deep.text


@pytest.mark.integration
async def test_a_bundled_dashboard_never_answers_for_an_api_path(
    api_settings: Settings,
    client_for: ClientFactory,
    bundled_dashboard: Path,
) -> None:
    del bundled_dashboard
    async with client_for(create_app(api_settings)) as client:
        real = await client.get("/api/health")
        assert real.status_code == 200
        assert real.json()["database"] == "ok"

        unknown = await client.get("/api/no-such-route")
        assert unknown.status_code == 404
        assert SHELL not in unknown.text
        assert unknown.headers["content-type"].startswith("application/problem+json")


@pytest.mark.integration
async def test_the_placeholder_page_loads_intact_under_its_own_policy(
    api_settings: Settings,
    client_for: ClientFactory,
    static_dir_state: str,
) -> None:
    """The page this project serves must not be the first thing its CSP blocks.

    The placeholder carried an inline `<style>` block while the same response
    sent `style-src 'self'`, so a browser would have dropped the styles and
    logged a violation. The rules now come from a same-origin stylesheet, which
    keeps the policy strict rather than relaxing it for the one page that
    announces the API.

    Checked in both states: when a dashboard IS bundled there is no placeholder
    route at all - `/_placeholder.css` is just another unmatched path, so the
    SPA fallback answers it with the dashboard shell the same way it would any
    other deep link, rather than with the placeholder's own stylesheet.
    """
    async with client_for(create_app(api_settings)) as client:
        page = await client.get("/")
        assert page.status_code == 200
        body = page.text

        if static_dir_state == "present":
            assert SHELL in body
            not_a_real_route = await client.get("/_placeholder.css")
            assert not_a_real_route.status_code == 200
            assert SHELL in not_a_real_route.text
            return

        assert "<style" not in body.lower(), "an inline style block is blocked by style-src 'self'"
        assert "<script" not in body.lower(), "the page needs no script, so it carries none"
        assert "onload=" not in body.lower()
        assert 'href="/_placeholder.css"' in body

        policy = page.headers["content-security-policy"]
        assert "style-src 'self'" in policy
        assert "'unsafe-inline'" not in policy

        stylesheet = await client.get("/_placeholder.css")
        assert stylesheet.status_code == 200
        assert stylesheet.headers["content-type"].startswith("text/css")
        assert "system-ui" in stylesheet.text
        assert stylesheet.headers["x-content-type-options"] == "nosniff"


@pytest.mark.integration
async def test_the_placeholder_references_only_same_origin_resources(
    api_client: httpx.AsyncClient,
) -> None:
    """`default-src 'self'` means every URL the page pulls must be relative."""
    body = (await api_client.get("/")).text
    for scheme in ("http://", "https://", "//cdn", "data:"):
        assert scheme not in body, f"the page references {scheme}, which its own CSP forbids"
