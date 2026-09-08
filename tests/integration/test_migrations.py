"""Migrations: they apply from empty, and they match the ORM metadata exactly."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

EXPECTED_TABLES = {
    "alembic_version",
    "benchmark_snapshots",
    "case_results",
    "evaluations",
    "model_responses",
    "run_metrics",
    "runs",
}


def _alembic(repo_root: Path, url: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the Alembic CLI as a subprocess.

    A subprocess rather than an import: `alembic` is a storage-only dependency
    and importing it from a test would defeat the banned-import rule that keeps
    migrations out of the rest of the codebase.
    """
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-controlled input
        [sys.executable, "-m", "alembic", *args],
        cwd=repo_root,
        env={**os.environ, "LLM_EVAL_DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.integration
def test_upgrade_head_succeeds_from_an_empty_database(
    tmp_path: Path, repo_root: Path, db_url: str
) -> None:
    assert not (tmp_path / "test.db").exists()
    result = _alembic(repo_root, db_url, "upgrade", "head")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    with sqlite3.connect(tmp_path / "test.db") as connection:
        tables = {
            row[0]
            for row in connection.execute("select name from sqlite_master where type='table'")
        }
    assert tables >= EXPECTED_TABLES


@pytest.mark.integration
def test_alembic_check_reports_no_drift_against_the_orm(repo_root: Path, migrated_db: str) -> None:
    result = _alembic(repo_root, migrated_db, "check")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "No new upgrade operations detected" in (result.stdout + result.stderr)


@pytest.mark.integration
def test_run_metrics_has_run_id_as_its_primary_key(tmp_path: Path, migrated_db: str) -> None:
    del migrated_db
    with sqlite3.connect(tmp_path / "test.db") as connection:
        columns = list(connection.execute("pragma table_info('run_metrics')"))
    primary_key = [row[1] for row in columns if row[5]]
    assert primary_key == ["run_id"], "exactly one metrics row per run must be structural"


@pytest.mark.integration
def test_foreign_keys_cascade_from_runs_downward(tmp_path: Path, migrated_db: str) -> None:
    del migrated_db
    with sqlite3.connect(tmp_path / "test.db") as connection:
        for table in ("case_results", "evaluations", "model_responses", "run_metrics"):
            keys = list(connection.execute(f"pragma foreign_key_list('{table}')"))
            assert keys, f"{table} must reference runs"
            assert all(row[6] == "CASCADE" for row in keys), f"{table} must cascade on delete"


@pytest.mark.integration
def test_the_migrations_path_never_comes_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alembic IMPORTS what it finds, so a working-directory fallback executes code.

    `llm-eval db upgrade` run from a directory someone else prepared must not
    load that directory's `migrations/env.py`. The resolved path is anchored to
    the installed module, never to `Path.cwd()`.
    """
    from llm_eval_lab.storage.migrator import (  # noqa: PLC0415 - reads next to its use
        find_migrations_path,
    )

    hostile = tmp_path / "migrations"
    hostile.mkdir()
    (hostile / "env.py").write_text("raise SystemExit('executed')", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved = find_migrations_path()
    assert resolved != hostile
    assert tmp_path not in resolved.parents
    assert (resolved / "versions" / "0001_initial.py").is_file()


@pytest.mark.integration
def test_an_upgrade_works_from_an_unrelated_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of packaging the scripts: the command works from anywhere."""
    import asyncio  # noqa: PLC0415 - reads next to its use

    from llm_eval_lab.services import upgrade_database  # noqa: PLC0415 - reads next to its use

    hostile = tmp_path / "migrations"
    hostile.mkdir()
    (hostile / "env.py").write_text("raise SystemExit('executed')", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    target = tmp_path / "elsewhere.db"
    revision = asyncio.run(upgrade_database(f"sqlite+aiosqlite:///{target}"))

    assert revision == "0001"
    assert target.is_file()


@pytest.mark.integration
@pytest.mark.slow
def test_the_built_wheel_contains_the_migration_scripts(repo_root: Path, tmp_path: Path) -> None:
    """An installed wheel with no migrations cannot upgrade a database at all."""
    import shutil  # noqa: PLC0415 - reads next to its use
    import zipfile  # noqa: PLC0415 - reads next to its use

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH, so the wheel cannot be built here")

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-controlled input
        [uv, "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    wheels = list(tmp_path.glob("*.whl"))
    assert wheels, "uv build produced no wheel"
    names = set(zipfile.ZipFile(wheels[0]).namelist())
    assert "llm_eval_lab/migrations/env.py" in names
    assert "llm_eval_lab/migrations/versions/0001_initial.py" in names
