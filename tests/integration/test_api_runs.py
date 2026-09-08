"""The run lifecycle over HTTP: launch, poll, cancel, sweep, page."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from llm_eval_lab.api.app import create_app
from llm_eval_lab.models import (
    AggregateMetrics,
    CostBreakdown,
    LatencyStats,
    RunStatus,
    TokenTotals,
    UnitOfWorkFactory,
)
from llm_eval_lab.services import run_manager
from llm_eval_lab.utils.time import utc_now

if TYPE_CHECKING:
    import httpx

    from integration.conftest import ClientFactory, SeedRuns
    from llm_eval_lab.settings import Settings

TERMINAL = {"completed", "partial", "failed", "cancelled", "interrupted"}
POLL_TIMEOUT_S = 10.0
POLL_INTERVAL_S = 0.05


async def poll_until_terminal(client: httpx.AsyncClient, run_id: str) -> dict[str, Any]:
    """Poll a run's status until it reports a terminal state, or fail the test."""
    deadline = asyncio.get_running_loop().time() + POLL_TIMEOUT_S
    last: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}/status")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in TERMINAL:
            return last
        await asyncio.sleep(POLL_INTERVAL_S)
    pytest.fail(f"run {run_id} did not reach a terminal status in {POLL_TIMEOUT_S}s: {last}")


@pytest.mark.integration
async def test_a_launched_run_reaches_a_terminal_status_and_reports_metrics(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    created = await api_client.post("/api/runs", json=launch_body)
    assert created.status_code == 202, created.text
    payload = created.json()
    run_id = payload["run_id"]
    assert payload["status"] == RunStatus.PENDING.value
    assert payload["links"]["status"] == f"/api/runs/{run_id}/status"

    final = await poll_until_terminal(api_client, run_id)
    assert final["status"] == RunStatus.COMPLETED.value
    assert final["completed"] == final["total"] == 6

    metrics = await api_client.get(f"/api/runs/{run_id}/metrics")
    assert metrics.status_code == 200, metrics.text
    assert metrics.json()["pass_rate"] == 1.0

    # The engine writes the run's terminal status and THEN materialises its
    # rollup, so a client that polls quickly enough can see `completed` in the
    # window before the row exists. The header is honest about that - it reports
    # the figures as derived - and `MetricsService.get` backfills the row on the
    # way past, so the next read is materialised. Polling here rather than
    # asserting on the first response is what makes the test independent of how
    # fast the machine running it happens to be.
    materialized = metrics.headers["X-Metrics-Materialized"]
    for _ in range(20):
        if materialized == "true":
            break
        await asyncio.sleep(POLL_INTERVAL_S)
        materialized = (await api_client.get(f"/api/runs/{run_id}/metrics")).headers[
            "X-Metrics-Materialized"
        ]
    assert materialized == "true", "the rollup is stored once the run has finished"

    detail = await api_client.get(f"/api/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["pass_rate"] == 1.0

    cases = await api_client.get(f"/api/runs/{run_id}/cases")
    assert cases.status_code == 200
    listed = cases.json()
    assert listed["total"] == 6
    assert {item["case_id"] for item in listed["items"]} >= {"capital-exact"}

    one = await api_client.get(f"/api/runs/{run_id}/cases/capital-exact")
    assert one.status_code == 200
    assert one.json()["evaluations"], "the case detail carries its evaluations"


@pytest.mark.integration
async def test_json_timestamps_use_the_trailing_z_form(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    """The API and the CLI must spell one instant one way.

    `cli/output.py` renders a UTC offset as a trailing `Z`; a response that said
    `+00:00` would make the same timestamp compare unequal as a string between
    the two surfaces.
    """
    created = await api_client.post("/api/runs", json=launch_body)
    run_id = created.json()["run_id"]
    await poll_until_terminal(api_client, run_id)

    status = (await api_client.get(f"/api/runs/{run_id}/status")).json()
    assert status["updated_at"].endswith("Z")
    assert "+00:00" not in status["updated_at"]


@pytest.mark.integration
async def test_cancelling_a_running_run_leaves_it_cancelled_with_partial_results(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    """Cancellation is cooperative: in-flight cases finish and are persisted.

    The run is deliberately slowed - one case at a time, each sleeping a real
    quarter second - so there is a window in which it is genuinely running. A run
    that finished before the cancel arrived would make this test pass for the
    wrong reason, which is why the terminal status is asserted to be `cancelled`
    rather than merely terminal.
    """
    created = await api_client.post(
        "/api/runs",
        json=launch_body
        | {
            "provider": {
                "provider": "fake",
                "model": "fake-1",
                "options": {
                    "mode": "expected",
                    "sleep": True,
                    "latency_min_ms": 250.0,
                    "latency_max_ms": 250.0,
                },
            },
            "concurrency": 1,
        },
    )
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]

    deadline = asyncio.get_running_loop().time() + POLL_TIMEOUT_S
    while asyncio.get_running_loop().time() < deadline:
        snapshot = (await api_client.get(f"/api/runs/{run_id}/status")).json()
        if snapshot["completed"] >= 1:
            break
        await asyncio.sleep(POLL_INTERVAL_S)
    else:  # pragma: no cover - only on a machine too slow to run one fake case
        pytest.fail("no case completed before the cancellation window closed")

    cancelled = await api_client.post(f"/api/runs/{run_id}/cancel")
    assert cancelled.status_code == 202, cancelled.text
    assert cancelled.json()["outcome"] == "requested"

    final = await poll_until_terminal(api_client, run_id)
    assert final["status"] == RunStatus.CANCELLED.value
    assert 0 < final["completed"] < final["total"], "partial results, not none and not all"

    cases = await api_client.get(f"/api/runs/{run_id}/cases")
    assert cases.json()["total"] == final["completed"], "the partial results are persisted"

    again = await api_client.post(f"/api/runs/{run_id}/cancel")
    assert again.status_code == 409, "a finished run cannot be cancelled"


@pytest.mark.integration
async def test_a_run_left_running_by_a_dead_process_is_interrupted_by_the_startup_sweep(
    api_settings: Settings,
    seed_runs: SeedRuns,
    client_for: ClientFactory,
) -> None:
    stale = utc_now() - timedelta(hours=1)
    (run_id,) = await seed_runs(count=1, status=RunStatus.RUNNING, updated_at=stale)

    async with client_for(create_app(api_settings)) as client:
        status = await client.get(f"/api/runs/{run_id}/status")
        assert status.status_code == 200, status.text
        assert status.json()["status"] == RunStatus.INTERRUPTED.value
        assert status.json()["managed"] is False


@pytest.mark.integration
async def test_a_run_this_process_did_not_launch_cannot_be_cancelled_here(
    api_settings: Settings,
    seed_runs: SeedRuns,
    client_for: ClientFactory,
) -> None:
    """The single-worker assumption is reported, not hidden.

    A run heartbeated recently by another process is still RUNNING after the
    sweep, and this process holds no task for it. Answering 202 would be a
    promise nothing keeps.
    """
    (run_id,) = await seed_runs(
        count=1,
        status=RunStatus.RUNNING,
        updated_at=utc_now() + timedelta(minutes=5),
    )
    async with client_for(create_app(api_settings)) as client:
        response = await client.post(f"/api/runs/{run_id}/cancel")
    assert response.status_code == 409
    assert "not being executed by this process" in response.json()["detail"]


@pytest.mark.integration
async def test_paging_reports_the_window_and_the_total_behind_it(
    api_settings: Settings,
    seed_runs: SeedRuns,
    client_for: ClientFactory,
) -> None:
    await seed_runs(count=25)

    async with client_for(create_app(api_settings)) as client:
        page = await client.get("/api/runs", params={"limit": 10, "offset": 20})
        assert page.status_code == 200, page.text
        payload = page.json()
        assert payload["total"] == 25
        assert payload["limit"] == 10
        assert payload["offset"] == 20
        assert len(payload["items"]) == 5

        filtered = await client.get("/api/runs", params={"provider": "fake", "suite": "smoke"})
        assert filtered.json()["total"] == 25
        missing = await client.get("/api/runs", params={"provider": "nobody"})
        assert missing.json()["total"] == 0


@pytest.mark.integration
async def test_an_unknown_run_is_a_problem_detail_not_a_bare_404(
    api_client: httpx.AsyncClient,
) -> None:
    for path in ("/api/runs/nope", "/api/runs/nope/status", "/api/runs/nope/metrics"):
        response = await api_client.get(path)
        assert response.status_code == 404, path
        assert response.headers["content-type"].startswith("application/problem+json")
        body = response.json()
        assert body["status"] == 404
        assert body["type"] == "/problems/run-not-found"


@pytest.mark.integration
async def test_a_run_exports_as_json_and_as_csv(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    created = await api_client.post("/api/runs", json=launch_body)
    run_id = created.json()["run_id"]
    await poll_until_terminal(api_client, run_id)

    exported = await api_client.get(f"/api/runs/{run_id}/export", params={"format": "json"})
    assert exported.status_code == 200
    document = exported.json()
    assert document["n_cases"] == 6
    assert len(document["cases"]) == 6
    assert document["metrics"]["pass_rate"] == 1.0

    csv_export = await api_client.get(f"/api/runs/{run_id}/export", params={"format": "csv"})
    assert csv_export.status_code == 200
    assert csv_export.headers["content-type"].startswith("text/csv")
    lines = csv_export.text.strip().splitlines()
    assert lines[0].startswith("run_id,case_id,case_hash,status,passed")
    assert len(lines) == 7, "a header and one line per case"


@pytest.mark.integration
async def test_launching_a_run_against_an_invalid_suite_reports_every_field_error(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    response = await api_client.post(
        "/api/runs", json=launch_body | {"suite": "malformed_missing_id"}
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["type"] == "/problems/benchmark-invalid"
    assert body["errors"], "the field errors travel with the problem detail"
    assert all("file" not in item for item in body["errors"]), (
        "an error response does not describe the server's file system"
    )


@pytest.mark.integration
async def test_a_secret_shaped_provider_option_is_rejected_with_422(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    """The same secret-shaped-option rejection the CLI hits must be a clean 422 here too.

    `CreateRunRequest.provider` is typed as `ProviderConfig` itself, so FastAPI's
    own request-body validation constructs it directly and rejects the secret
    before any service code runs - proving the CLI fix (`run_service.py`'s
    `provider_config`) did not need a matching change on this path, only that
    the frozen `ProviderConfig` validator continues to fire here as it always
    did. The planted secret must never appear in the response body.
    """
    body = launch_body | {
        "provider": {
            "provider": "fake",
            "model": "fake-1",
            "options": {"api_key": "sk-leak-me-not"},
        }
    }
    response = await api_client.post("/api/runs", json=body)
    assert response.status_code == 422, response.text
    assert "sk-leak-me-not" not in response.text


@pytest.mark.integration
async def test_the_run_listing_and_the_leaderboard_agree_about_a_finished_run(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    created = await api_client.post("/api/runs", json=launch_body)
    run_id = created.json()["run_id"]
    await poll_until_terminal(api_client, run_id)

    leaderboard = await api_client.get("/api/models/summary")
    assert leaderboard.status_code == 200, leaderboard.text
    rows = leaderboard.json()
    assert len(rows) == 1
    row = rows[0]
    assert (row["provider"], row["model"]) == ("fake", "fake-1")
    assert row["n_runs"] == 1
    assert row["pass_rate"] == 1.0
    assert row["n_pass_denominator"] == 6
    low, high = row["pass_rate_ci"]
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert low <= high


@pytest.mark.integration
async def test_shutting_the_process_down_stops_the_runs_it_is_executing(
    api_settings: Settings,
    client_for: ClientFactory,
    launch_body: dict[str, Any],
) -> None:
    """Leaving the lifespan asks every managed run to stop, and waits for it.

    A run still writing cases when the process exits is the case the startup
    sweep exists to clean up. Stopping cooperatively first means the common
    shutdown leaves a run with a terminal status rather than one for the next
    process to reconcile.
    """
    slow = launch_body | {
        "provider": {
            "provider": "fake",
            "model": "fake-1",
            "options": {
                "mode": "expected",
                "sleep": True,
                "latency_min_ms": 200.0,
                "latency_max_ms": 200.0,
            },
        },
        "concurrency": 1,
    }
    app = create_app(api_settings)
    async with client_for(app) as client:
        created = await client.post("/api/runs", json=slow)
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]
        for _ in range(int(POLL_TIMEOUT_S / POLL_INTERVAL_S)):
            if (await client.get(f"/api/runs/{run_id}/status")).json()["completed"] >= 1:
                break
            await asyncio.sleep(POLL_INTERVAL_S)
        else:  # pragma: no cover - only on a machine too slow to run one fake case
            pytest.fail("no case completed before the shutdown window closed")

    # The lifespan has now exited. A second application over the same database
    # reports what the first one left behind.
    async with client_for(create_app(api_settings)) as client:
        final = (await client.get(f"/api/runs/{run_id}/status")).json()
    assert final["status"] in TERMINAL, "the run did not survive shutdown as a terminal row"
    assert final["managed"] is False
    assert final["last_case_id"] is None, "progress snapshots do not outlive their process"


@pytest.mark.integration
async def test_a_managed_run_reports_the_case_its_latest_event_named(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
) -> None:
    """The progress snapshot is populated while the run is executing, and only then."""
    created = await api_client.post(
        "/api/runs",
        json=launch_body
        | {
            "provider": {
                "provider": "fake",
                "model": "fake-1",
                "options": {
                    "mode": "expected",
                    "sleep": True,
                    "latency_min_ms": 200.0,
                    "latency_max_ms": 200.0,
                },
            },
            "concurrency": 1,
        },
    )
    run_id = created.json()["run_id"]

    seen: str | None = None
    for _ in range(int(POLL_TIMEOUT_S / POLL_INTERVAL_S)):
        snapshot = (await api_client.get(f"/api/runs/{run_id}/status")).json()
        if snapshot["last_case_id"] is not None:
            seen = snapshot["last_case_id"]
            assert snapshot["managed"] is True
            break
        if snapshot["status"] in TERMINAL:
            break
        await asyncio.sleep(POLL_INTERVAL_S)
    else:  # pragma: no cover - only on a machine too slow to run one fake case
        pytest.fail("the status endpoint was polled to exhaustion with no progress event")
    assert seen is not None, "no progress event was ever observed mid-run"

    await poll_until_terminal(api_client, run_id)
    listed = (await api_client.get(f"/api/runs/{run_id}/cases", params={"limit": 100})).json()
    assert seen in {item["case_id"] for item in listed["items"]}, (
        "the snapshot named a case the run actually executed"
    )

    # Once the task finishes it is untracked, and the snapshot goes with it.
    assert (await api_client.get(f"/api/runs/{run_id}/status")).json()["last_case_id"] is None


@pytest.mark.integration
async def test_an_unset_credential_variable_leaves_no_run_row_behind(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 424 is refused BEFORE the row is written, not after.

    Resolving the credential inside the background task produced a 202, then a
    run that turned FAILED with the bare string "MissingCredentialError" and the
    variable's name reaching only the server log. The listing is what proves the
    check now happens on the caller's own request.
    """
    monkeypatch.delenv("LLM_EVAL_ALSO_NOT_SET", raising=False)
    before = (await api_client.get("/api/runs")).json()["total"]

    refused = await api_client.post(
        "/api/runs",
        json=launch_body
        | {
            "provider": {
                "provider": "fake",
                "model": "fake-1",
                "api_key_env": "LLM_EVAL_ALSO_NOT_SET",
                "options": {"mode": "expected"},
            }
        },
    )
    assert refused.status_code == 424, refused.text
    assert refused.json()["env_var"] == "LLM_EVAL_ALSO_NOT_SET"

    after = (await api_client.get("/api/runs")).json()
    assert after["total"] == before, "no run row was created for a launch that was refused"


@pytest.mark.integration
async def test_a_set_credential_variable_launches_normally(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive control: the new check is not simply refusing every launch."""
    monkeypatch.setenv("LLM_EVAL_IS_SET_FOR_THIS_TEST", "not-a-real-credential")
    created = await api_client.post(
        "/api/runs",
        json=launch_body
        | {
            "provider": {
                "provider": "fake",
                "model": "fake-1",
                "api_key_env": "LLM_EVAL_IS_SET_FOR_THIS_TEST",
                "options": {"mode": "expected"},
            }
        },
    )
    assert created.status_code == 202, created.text
    final = await poll_until_terminal(api_client, created.json()["run_id"])
    assert final["status"] == RunStatus.COMPLETED.value


@pytest.mark.integration
async def test_cost_per_case_is_the_same_number_on_both_surfaces(
    api_settings: Settings,
    uow_factory: UnitOfWorkFactory,
    seed_runs: SeedRuns,
    client_for: ClientFactory,
) -> None:
    """N1: one field name must not carry two definitions.

    Phase 3 pins `cost_per_case = run_cost / n_completed` for a single run, with
    the identity `cost_per_case * n_completed == cost.total_cost`. The
    leaderboard publishes the same field name, so it must publish the same
    number - including when half the cases carried no price, which is exactly the
    case where a priced-only denominator would disagree.
    """
    (run_id,) = await seed_runs(count=1)
    stored = AggregateMetrics(
        run_id=run_id,
        computed_at=utc_now(),
        n_cases=100,
        n_completed=100,
        n_errors=0,
        n_timeouts=0,
        n_skipped=0,
        error_rate=0.0,
        error_breakdown={},
        error_policy="exclude",
        pass_rate=1.0,
        pass_rate_ci=None,
        n_passed=100,
        n_pass_denominator=100,
        n_scored=100,
        mean_score=1.0,
        median_score=1.0,
        latency=LatencyStats(
            n=100, mean_ms=10.0, p50_ms=10.0, p90_ms=10.0, p95_ms=10.0, p99_ms=10.0, max_ms=10.0
        ),
        tokens=TokenTotals(
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            mean_input_tokens=None,
            mean_output_tokens=None,
            n_with_usage=0,
            n_missing_usage=100,
        ),
        token_usage_coverage=None,
        cost=CostBreakdown(
            input_cost=None,
            output_cost=None,
            total_cost=Decimal("5.00"),
            priced=True,
            price_table_id="test",
            price_table_version="1",
        ),
        # Half the cases carried a price. The denominator is still every
        # completed case, so the figure is a lower bound and `cost_coverage`
        # beside it is what says so.
        cost_coverage=0.5,
        cost_per_case=Decimal("0.05"),
        cost_per_successful_evaluation=None,
        by_category={},
        by_tag={},
        by_evaluator={},
    )
    async with uow_factory() as uow:
        await uow.metrics.put(stored)
        await uow.commit()

    async with client_for(create_app(api_settings)) as client:
        run_metrics = await client.get(f"/api/runs/{run_id}/metrics")
        assert run_metrics.status_code == 200, run_metrics.text
        assert run_metrics.json()["cost_per_case"] == "0.05"

        leaderboard = await client.get("/api/models/summary")
        assert leaderboard.status_code == 200, leaderboard.text
        (row,) = leaderboard.json()

    assert row["cost_per_case"] == "0.05", "the leaderboard publishes the run's own definition"
    assert row["n_cases"] == 100
    assert row["total_cost"] == "5.00"
    assert Decimal(row["cost_per_case"]) * row["n_cases"] == Decimal(row["total_cost"]), (
        "cost_per_case * n_cases reconciles against total_cost"
    )
    assert row["cost_coverage"] == 0.5, "the coverage caveat travels beside the figure"


@pytest.mark.integration
async def test_concurrent_launches_never_exceed_the_process_cap(
    api_client: httpx.AsyncClient,
    launch_body: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N2: the cap was check-then-act across three awaits.

    Forty simultaneous requests all saw an empty tracked dictionary, all passed
    the check, and all launched - each one spending money. The slot is now
    reserved synchronously, in the same uninterrupted step as the check.
    """
    cap = 16
    monkeypatch.setattr(run_manager, "MAX_CONCURRENT_RUNS", cap)
    slow = launch_body | {
        "provider": {
            "provider": "fake",
            "model": "fake-1",
            "options": {
                "mode": "expected",
                "sleep": True,
                "latency_min_ms": 900.0,
                "latency_max_ms": 900.0,
            },
        },
        "concurrency": 1,
    }

    # A generous, explicit per-request timeout, not the client's 5s default.
    # httpx's default (`Timeout(timeout=5.0)`) coincides almost exactly with
    # `storage/engine.py`'s SQLite `busy_timeout=5000` pragma: forty POSTs
    # launched at once all reach the SAME sqlite file through one connection,
    # and under real machine contention (another process competing for CPU,
    # not a defect in the cap logic this test exists to prove) a write can
    # legitimately need most of that 5s busy-timeout window to get its turn.
    # A tight client-side timeout then races the very thing it is trying to
    # observe and fails on a `ReadTimeout` instead of an assertion - which is
    # exactly the transient failure this test hit under concurrent load. The
    # cap and the accept/refuse counts are unchanged; only the artificial
    # deadline this test itself imposed is relaxed.
    responses = await asyncio.gather(
        *(api_client.post("/api/runs", json=slow, timeout=30.0) for _ in range(40))
    )
    codes = [response.status_code for response in responses]
    accepted = codes.count(202)
    refused = codes.count(429)

    assert accepted + refused == 40, f"unexpected statuses: {sorted(set(codes))}"
    assert accepted <= cap, f"{accepted} runs launched against a cap of {cap}"
    assert accepted >= 1, "the cap refused everything, which is not a cap"
    assert refused >= 40 - cap

    for response in responses:
        if response.status_code == 429:
            assert response.json()["type"] == "/problems/too-many-runs"

    listed = (await api_client.get("/api/runs", timeout=30.0)).json()
    assert listed["total"] == accepted, "a refused launch writes no run row"


@pytest.mark.integration
async def test_a_self_judging_run_reports_self_preference_risk_over_http(
    api_client: httpx.AsyncClient,
) -> None:
    """`GET /api/runs/{id}` must surface `self_preference_risk` in `run.warnings`.

    The judge fixture suite (addressed by its file name, `judge_rubric`)
    configures its judge as `provider: fake, model: fake-judge-1`. Launching the
    candidate against that same provider and model is exactly the condition the
    frozen-contract addendum requires to be "visible, never silent" at the run
    level, not only inside each case's own judge provenance.
    """
    body = {
        "suite": "judge_rubric",
        "provider": {"provider": "fake", "model": "fake-judge-1"},
    }
    created = await api_client.post("/api/runs", json=body)
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]

    final = await poll_until_terminal(api_client, run_id)
    assert final["status"] == RunStatus.COMPLETED.value

    detail = await api_client.get(f"/api/runs/{run_id}")
    assert detail.status_code == 200, detail.text
    assert "self_preference_risk" in detail.json()["run"]["warnings"]
