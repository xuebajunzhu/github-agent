"""Breaker state observability: /api/admin/breakers (DFR)."""
from __future__ import annotations

import pytest

from app.llm.base import AnalysisError, LLMAnalyzer


class Flaky(LLMAnalyzer):
    """Always fails when called — exercises the breaker path."""

    name = "flaky"

    def __init__(self):
        self.calls = 0

    def available(self) -> bool:
        return True

    def analyze(self, project_data: dict) -> AnalysisResult:
        self.calls += 1
        raise AnalysisError("401 invalid key")

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        raise AnalysisError("401 invalid key")


def test_breakers_endpoint_reports_state(client):
    flaky = Flaky()
    services = client.app.state.services
    services.analysis.analyzers.insert(0, flaky)
    services.analysis._breakers["flaky"] = services.analysis._breaker(flaky)

    # trip the breaker: threshold 3 consecutive failures
    for _ in range(3):
        with pytest.raises(AnalysisError):
            services.analysis.chat("system", "user")

    response = client.get(
        "/api/admin/breakers", headers={"X-Admin-Token": "test-admin-token"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["flaky"] == {"failures": 3, "open": True}
    # every other analyzer reports a closed circuit
    assert body.get("rule", {}).get("open") is False


def test_breakers_endpoint_requires_admin(client):
    assert client.get("/api/admin/breakers").status_code == 401
