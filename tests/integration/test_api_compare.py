"""Comparison over HTTP, and the promise that it is the CLI's own payload.

A dashboard and a CI pipeline must not be looking at two spellings of the same
comparison. The API returns ``RegressionReport.model_dump_json()`` verbatim -
the same encoder, the same bytes - and the byte comparison below is what keeps
that true rather than merely intended.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from llm_eval_lab.models import RegressionReport

if TYPE_CHECKING:
    from pathlib import Path

    import httpx

TERMINAL = {"completed", "partial", "failed", "cancelled", "interrupted"}
POLL_TIMEOUT_S = 15.0
CLI_TIMEOUT_S = 120
MUTATE_RATE = 0.2
"""The same fraction the end-to-end regression test uses; the fake degrades deterministically."""

_TIMESTAMP = re.compile(r'"generated_at":"[^"]*"')


def _blank_timestamp(text: str) -> str:
    """Blank the one field two separate comparisons cannot agree on.

    Everything else in the document is a function of the two runs and the policy,
    so blanking ``generated_at`` leaves a genuine byte comparison rather than a
    structural one.
    """
    return _TIMESTAMP.sub('"generated_at":"X"', text)


async def _launch(client: httpx.AsyncClient, options: dict[str, Any], label: str) -> str:
    """Start one run over HTTP and wait for it to finish."""
    created = await client.post(
        "/api/runs",
        json={
            "suite": "smoke",
            "provider": {"provider": "fake", "model": "fake-1", "options": options},
            "label": label,
        },
    )
    assert created.status_code == 202, created.text
    run_id: str = created.json()["run_id"]

    deadline = asyncio.get_running_loop().time() + POLL_TIMEOUT_S
    while asyncio.get_running_loop().time() < deadline:
        status = (await client.get(f"/api/runs/{run_id}/status")).json()
        if status["status"] in TERMINAL:
            return run_id
        await asyncio.sleep(0.05)
    pytest.fail(f"run {label} did not finish in {POLL_TIMEOUT_S}s")


@pytest.fixture
async def regression_pair(api_client: httpx.AsyncClient) -> tuple[str, str]:
    """A clean baseline and a deliberately degraded candidate, both over HTTP."""
    clean = await _launch(api_client, {"mode": "expected"}, "clean")
    degraded = await _launch(api_client, {"mode": "mutate", "mutate_rate": MUTATE_RATE}, "degraded")
    return clean, degraded


@pytest.mark.integration
async def test_the_intentional_regression_pair_fails_over_http(
    api_client: httpx.AsyncClient,
    regression_pair: tuple[str, str],
) -> None:
    baseline, candidate = regression_pair
    response = await api_client.post(
        "/api/compare",
        json={"baseline_run_id": baseline, "candidate_run_id": candidate},
    )
    assert response.status_code == 200, response.text

    report = RegressionReport.model_validate_json(response.content)
    assert report.verdict.value == "fail"
    assert report.mode == "paired"
    assert report.comparable is True
    assert report.suite_hash_match is True
    assert report.summary.newly_failing
    assert report.summary.newly_passing == ()

    overall = next(check for check in report.checks if check.metric == "pass_rate")
    assert overall.status.value == "failed"
    assert overall.candidate is not None
    assert overall.baseline is not None
    assert overall.candidate < overall.baseline


@pytest.mark.integration
async def test_the_http_body_is_the_payload_the_cli_writes(
    api_client: httpx.AsyncClient,
    regression_pair: tuple[str, str],
    migrated_db: str,
    repo_root: Path,
) -> None:
    baseline, candidate = regression_pair
    response = await api_client.post(
        "/api/compare",
        json={"baseline_run_id": baseline, "candidate_run_id": candidate},
    )
    assert response.status_code == 200, response.text

    # In a worker thread: the CLI is a blocking subprocess and this test's own
    # event loop is the one the API's background tasks live in.
    completed = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "llm_eval_lab.cli", "compare", baseline, candidate, "--json"],
        cwd=repo_root,
        env={**os.environ, "LLM_EVAL_DATABASE_URL": migrated_db},
        capture_output=True,
        text=True,
        check=False,
        timeout=CLI_TIMEOUT_S,
    )
    assert completed.stdout, completed.stderr

    assert _blank_timestamp(response.text) == _blank_timestamp(completed.stdout.strip()), (
        "the API and the CLI must serialise one report one way"
    )
    # And the body really is the model's own encoding rather than a re-render.
    assert response.text == RegressionReport.model_validate_json(response.content).model_dump_json()


@pytest.mark.integration
async def test_the_linkable_get_form_uses_the_shipped_policy(
    api_client: httpx.AsyncClient,
    regression_pair: tuple[str, str],
) -> None:
    baseline, candidate = regression_pair
    posted = await api_client.post(
        "/api/compare",
        json={"baseline_run_id": baseline, "candidate_run_id": candidate},
    )
    fetched = await api_client.get(
        "/api/compare",
        params={"baseline_run_id": baseline, "candidate_run_id": candidate},
    )
    assert fetched.status_code == 200, fetched.text
    assert _blank_timestamp(fetched.text) == _blank_timestamp(posted.text)


@pytest.mark.integration
async def test_an_inline_policy_is_honoured_and_a_reserved_bound_is_refused(
    api_client: httpx.AsyncClient,
    regression_pair: tuple[str, str],
) -> None:
    baseline, candidate = regression_pair
    permissive = {
        "id": "loose",
        "version": "1",
        "checks": [
            {
                "metric": "pass_rate",
                "direction": "higher_is_better",
                "max_absolute_decrease": 1.0,
                "min_samples": 1,
            }
        ],
    }
    response = await api_client.post(
        "/api/compare",
        json={
            "baseline_run_id": baseline,
            "candidate_run_id": candidate,
            "thresholds": permissive,
        },
    )
    assert response.status_code == 200, response.text
    report = RegressionReport.model_validate_json(response.content)
    assert report.verdict.value == "pass", "a policy that tolerates the drop passes it"
    assert report.thresholds_id == "loose"

    reserved = json.loads(json.dumps(permissive))
    reserved["checks"][0]["require_significant"] = True
    refused = await api_client.post(
        "/api/compare",
        json={
            "baseline_run_id": baseline,
            "candidate_run_id": candidate,
            "thresholds": reserved,
        },
    )
    assert refused.status_code == 422, refused.text
    assert "require_significant is reserved" in refused.json()["detail"]


@pytest.mark.integration
async def test_comparing_an_unknown_run_is_a_four_oh_four_problem_detail(
    api_client: httpx.AsyncClient,
    regression_pair: tuple[str, str],
) -> None:
    baseline, _ = regression_pair
    response = await api_client.post(
        "/api/compare",
        json={"baseline_run_id": baseline, "candidate_run_id": "no-such-run"},
    )
    assert response.status_code == 404
    assert response.json()["type"] == "/problems/run-not-found"
    assert "no-such-run" in response.json()["detail"]
