"""`GET /api/health`: liveness, the database check, and the in-flight count."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import httpx


@pytest.mark.integration
async def test_health_reports_ok_and_zero_runs_in_flight_when_idle(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.get("/api/health")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database"] == "ok"
    assert payload["runs_in_flight"] == 0, "no run has been launched against this process"
