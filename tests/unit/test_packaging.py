"""Packaging and toolchain smoke tests.

These prove the package imports, the network-blocking fixture actually blocks
the network, the environment template is present and secret-free, and that
`uv build` - run either way - bundles a pre-built dashboard into the wheel.
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

import llm_eval_lab

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
_FAKE_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9]{16,}")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_and_has_a_semver_version() -> None:
    assert _VERSION_PATTERN.match(llm_eval_lab.__version__)


def test_network_is_blocked_by_default() -> None:
    with pytest.raises(RuntimeError, match="network access blocked"):
        socket.create_connection(("example.com", 80))


def test_env_example_exists_is_nonempty_and_has_no_real_looking_secret() -> None:
    env_example = _REPO_ROOT / ".env.example"
    assert env_example.is_file(), f"{env_example} must exist"

    content = env_example.read_text(encoding="utf-8")
    assert content.strip(), f"{env_example} must not be empty"
    assert not _FAKE_SECRET_PATTERN.search(content), (
        f"{env_example} must contain no string matching {_FAKE_SECRET_PATTERN.pattern!r}"
    )


# ---------------------------------------------------------------------------
# `uv build` bundles a pre-built dashboard, both ways it can be invoked
# ---------------------------------------------------------------------------

_UV = shutil.which("uv")
_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"


def _wheel_static_entries(out_dir: Path, *extra_args: str) -> list[str]:
    """Run `uv build` into `out_dir` and return the wheel's `api/static/*` names."""
    assert _UV is not None, "callers skip when uv is not on PATH"
    subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [_UV, "build", "--out-dir", str(out_dir), *extra_args],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel in {out_dir}, found {wheels}"
    with zipfile.ZipFile(wheels[0]) as archive:
        return [name for name in archive.namelist() if "api/static" in name]


@pytest.mark.slow
@pytest.mark.skipif(_UV is None, reason="uv is not on PATH")
@pytest.mark.skipif(
    not _FRONTEND_DIST.is_dir(), reason="frontend/dist is not built; run npm run build first"
)
def test_uv_build_wheel_from_a_checkout_bundles_the_dashboard() -> None:
    """`uv build --wheel`: the simplest, checkout-direct path."""
    with tempfile.TemporaryDirectory() as tmp:
        entries = _wheel_static_entries(Path(tmp), "--wheel")
    assert "llm_eval_lab/api/static/index.html" in entries


@pytest.mark.slow
@pytest.mark.skipif(_UV is None, reason="uv is not on PATH")
@pytest.mark.skipif(
    not _FRONTEND_DIST.is_dir(), reason="frontend/dist is not built; run npm run build first"
)
def test_plain_uv_build_also_bundles_the_dashboard_via_the_sdist() -> None:
    """`uv build` (no flag): sdist first, then the wheel built FROM it.

    Needs `frontend/dist` to travel inside the sdist itself, which is what the
    sdist's `artifacts` entry for it is for - a checkout-only `frontend/dist`
    would not survive the trip through an isolated sdist extraction.
    """
    with tempfile.TemporaryDirectory() as tmp:
        entries = _wheel_static_entries(Path(tmp))
    assert "llm_eval_lab/api/static/index.html" in entries


@pytest.mark.skipif(sys.platform == "win32", reason="path handling below assumes POSIX")
def test_the_sdist_never_carries_frontend_source_only_its_built_output() -> None:
    """The sdist artifact declaration names `frontend/dist`, never `frontend/`.

    A static check rather than another `uv build` invocation: this is a claim
    about the packaging configuration, checkable without a Node toolchain or a
    pre-built dashboard, and it is the property that keeps a source
    distribution installable with no Node.js and no network access.
    """
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "/frontend/dist/**/*" in text
    assert '"/frontend"' not in text.replace("/frontend/dist", "")
