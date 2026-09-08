"""Suite names are confined to a root, and the confinement is not "refuse everything".

The four escapes tested here are the four that actually get tried: a relative
``..`` walk, an absolute path, a percent-encoded walk that survives one round of
decoding, and a symlink planted inside the root that points out of it. The last
one is the reason the check happens after ``Path.resolve()`` rather than on the
string.

The positive control is not a formality. A confinement that refuses every name
passes all four negative tests and is useless, so a legitimate suite inside the
root is loaded in the same file, through the same code path, and its content is
asserted.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from llm_eval_lab.api.app import create_app
from llm_eval_lab.services import (
    SuiteNameError,
    SuiteNotFoundError,
    build_catalog_service,
    catalog_service,
    resolve_suite_name,
)
from llm_eval_lab.services.catalog_service import list_suite_names as resolve_listing

if TYPE_CHECKING:
    from pathlib import Path

    from integration.conftest import ClientFactory
    from llm_eval_lab.settings import Settings


def _one_case_suite(name: str) -> str:
    """Render the smallest suite that actually validates.

    A suite with no cases is refused by the loader, so a listing fixture built
    from empty suites would test the error path by accident.
    """
    return (
        f"schema_version: 1\nname: {name}\nversion: '1'\n"
        f"cases:\n"
        f"  - id: only\n"
        f"    input: hi\n"
        f"    expected: hi\n"
        f"    evaluators:\n"
        f"      - type: exact_match\n"
    )


MARKER = "OUTSIDE-THE-ROOT-MARKER"
"""Planted in the file outside the root. Its absence from a response is the assertion."""


@pytest.fixture
def confined_root(tmp_path: Path, fixtures_dir: Path) -> Path:
    """Build a benchmark root with one real suite and one symlink that escapes it."""
    root = tmp_path / "suites"
    root.mkdir()
    shutil.copy(fixtures_dir / "suites" / "smoke.yaml", root / "good.yaml")
    (root / "nested").mkdir()
    shutil.copy(fixtures_dir / "suites" / "smoke.yaml", root / "nested" / "deep.yaml")

    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.yaml"
    secret.write_text(
        f"schema_version: 1\nname: {MARKER}\nversion: '1'\ncases: []\n", encoding="utf-8"
    )
    (root / "escape.yaml").symlink_to(secret)
    return root


@pytest.fixture
def confined_settings(api_settings: Settings, confined_root: Path) -> Settings:
    """The API, pointed at the purpose-built root."""
    return api_settings.model_copy(update={"suites_root": confined_root})


ESCAPES: tuple[tuple[str, str], ...] = (
    ("relative traversal", "../outside/secret"),
    ("deep relative traversal", "nested/../../outside/secret"),
    ("absolute path", "/etc/passwd"),
    ("symlink out of the root", "escape"),
)
"""The escapes, named so a failure says which one got through."""


@pytest.mark.integration
@pytest.mark.parametrize(("label", "name"), ESCAPES)
def test_a_name_that_escapes_the_root_is_refused_by_the_resolver(
    confined_root: Path,
    label: str,
    name: str,
) -> None:
    with pytest.raises(SuiteNameError) as caught:
        resolve_suite_name(confined_root, name)
    assert not isinstance(caught.value, SuiteNotFoundError), (
        f"{label} must be refused as an escape, not reported as a missing file"
    )
    assert MARKER not in str(caught.value)
    assert "/etc/passwd" not in str(caught.value) or "outside the configured" in str(caught.value)


@pytest.mark.integration
@pytest.mark.parametrize(("label", "name"), ESCAPES)
async def test_a_name_that_escapes_the_root_returns_no_file_content_over_http(
    confined_settings: Settings,
    client_for: ClientFactory,
    label: str,
    name: str,
) -> None:
    async with client_for(create_app(confined_settings)) as client:
        response = await client.get(f"/api/benchmarks/{name}")
    assert response.status_code in {400, 404}, f"{label}: {response.status_code}"
    assert MARKER not in response.text, f"{label} returned the file outside the root"
    assert "root:" not in response.text


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    ["%2e%2e%2foutside%2fsecret", "..%2Foutside%2Fsecret", "%2E%2E/outside/secret"],
)
async def test_a_percent_encoded_traversal_is_refused_over_http(
    confined_settings: Settings,
    client_for: ClientFactory,
    name: str,
) -> None:
    """The router receives the DECODED name, which is where the check has to be."""
    async with client_for(create_app(confined_settings)) as client:
        response = await client.get(f"/api/benchmarks/{name}")
    assert response.status_code in {400, 404}, f"{name}: {response.status_code}"
    assert MARKER not in response.text


@pytest.mark.integration
def test_a_legitimate_suite_inside_the_root_still_loads(
    confined_settings: Settings,
    confined_root: Path,
) -> None:
    """The positive control: the confinement is not simply refusing everything."""
    resolved = resolve_suite_name(confined_root, "good")
    assert resolved == (confined_root / "good.yaml").resolve()

    catalog = build_catalog_service(confined_settings)
    suite = catalog.load_resolved_suite(catalog.suite_path("good"))
    assert suite.name == "smoke"
    assert len(suite.cases) == 6

    nested = catalog.load_resolved_suite(catalog.suite_path("nested/deep.yaml"))
    assert nested.name == "smoke", "a nested name inside the root resolves too"


@pytest.mark.integration
async def test_a_legitimate_suite_inside_the_root_still_loads_over_http(
    confined_settings: Settings,
    client_for: ClientFactory,
) -> None:
    async with client_for(create_app(confined_settings)) as client:
        response = await client.get("/api/benchmarks/good")
        assert response.status_code == 200, response.text
        assert response.json()["suite"]["name"] == "smoke"

        listing = await client.get("/api/benchmarks")
        assert listing.status_code == 200
        names = {row["name"] for row in listing.json()["items"]}
        assert {"good.yaml", "nested/deep.yaml"} <= names
        assert "escape.yaml" not in names, (
            "a symlink out of the root is not listed, so the listing and the "
            "resolver agree about what exists"
        )


@pytest.mark.integration
def test_no_configured_root_means_no_suite_is_addressable_by_name(
    api_settings: Settings,
) -> None:
    """Without a root there is no boundary, so name resolution is refused outright.

    The alternative - resolving a name against the working directory - would make
    the API's reach depend on where the process happened to be started.
    """
    with pytest.raises(SuiteNameError, match="LLM_EVAL_SUITES_ROOT"):
        resolve_suite_name(None, "smoke")
    del api_settings


@pytest.mark.integration
def test_the_listing_is_the_lexicographically_first_names_and_says_it_truncated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N3: a bounded listing must still be a deterministic one.

    Truncating the WALK bounded the work but returned whatever the file system
    yielded first, so two calls against one root could disagree about which
    suites exist and a suite could be unlistable yet loadable. Every matching
    name inside the scan bound is now collected, sorted, and cut to the cap.
    """
    cap = 5
    monkeypatch.setattr(catalog_service, "MAX_LISTED_SUITES", cap)

    root = tmp_path / "many"
    root.mkdir()
    # Created in an order unrelated to their names, so passing cannot be an
    # accident of directory ordering.
    names = [f"suite-{index:02d}" for index in range(cap + 3)]
    for name in reversed(names):
        (root / f"{name}.yaml").write_text(_one_case_suite(name), encoding="utf-8")

    listing = resolve_listing(root)
    assert listing.truncated is True
    assert listing.n_found == cap + 3
    assert list(listing.names) == [f"{name}.yaml" for name in names[:cap]]
    assert list(listing.names) == sorted(listing.names)


@pytest.mark.integration
def test_a_listing_that_fits_is_not_marked_truncated(confined_root: Path) -> None:
    """The negative control: `truncated` must mean something."""
    listing = resolve_listing(confined_root)
    assert listing.truncated is False
    assert listing.n_found == len(listing.names)
    assert "good.yaml" in listing.names


@pytest.mark.integration
async def test_a_truncated_listing_is_visible_over_http(
    api_settings: Settings,
    client_for: ClientFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`total` above the number of items is how a client learns it saw a window."""
    cap = 4
    monkeypatch.setattr(catalog_service, "MAX_LISTED_SUITES", cap)
    root = tmp_path / "lots"
    root.mkdir()
    for index in range(cap + 3):
        (root / f"s{index:02d}.yaml").write_text(_one_case_suite(f"s{index:02d}"), encoding="utf-8")

    settings = api_settings.model_copy(update={"suites_root": root})
    async with client_for(create_app(settings)) as client:
        response = await client.get("/api/benchmarks")
    assert response.status_code == 200, response.text
    page = response.json()
    assert len(page["items"]) == cap
    assert page["total"] == cap + 3
    assert page["offset"] == 0
    assert [row["name"] for row in page["items"]] == [f"s{index:02d}.yaml" for index in range(cap)]

    # A suite the listing did not show is still addressable by name.
    async with client_for(create_app(settings)) as client:
        assert (await client.get(f"/api/benchmarks/s{cap + 2:02d}")).status_code == 200
