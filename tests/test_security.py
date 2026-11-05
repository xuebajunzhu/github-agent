"""Security hardening tests: response headers + research budget."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def sec_client():
    settings = Settings(
        _env_file=None,
        database_url="sqlite://",
        scheduler_enabled=False,
        llm_provider="rule",
        llm_fallback_providers=[],
        embedder_provider="tfidf",
        embedding_dimension=64,
        vector_store_backend="memory",
        admin_token="test-admin-token",
        secret_key="test-secret",
        asr_provider="none",
        tts_provider="none",
    )
    app = create_app(settings)
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


def test_security_headers_on_every_response(sec_client):
    response = sec_client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    csp = response.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    # page + API get the same treatment
    assert sec_client.get("/").headers["x-frame-options"] == "DENY"


def test_research_budget_per_ip(sec_client):
    from tests.test_research import make_repo  # reuse the canned repo shape

    services = sec_client.app.state.services

    class FakePlanner:
        name = "fake"

        def available(self) -> bool:
            return True

        def analyze(self, project_data: dict):
            raise NotImplementedError

        def chat(self, system: str, user: str) -> str:
            import json

            return json.dumps({"keywords": ["kw"]})

    services.analysis.analyzers.insert(0, FakePlanner())
    services.research._analysis = services.analysis

    class FakeGitHub:
        def search_repositories(self, keyword, **kwargs):
            return []

        def close(self):
            pass

    services.research._github = FakeGitHub()
    # clear the hot cache so every request is a fresh run cost
    services.research._hot_cache.delete = lambda key: None  # type: ignore[method-assign]

    codes = []
    for _ in range(7):  # PER_MINUTE = 5 -> 5 ok, then 429
        response = sec_client.post("/api/research", json={"problem": "unique problem xyz"})
        codes.append(response.status_code)
    assert codes[:5] == [200] * 5
    assert codes[5] == 429 and codes[6] == 429
