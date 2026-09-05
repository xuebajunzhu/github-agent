from __future__ import annotations

from app.config import Settings
from app.main import create_app


def test_submit_and_report_feedback(client):
    from tests.test_api import seed_project_with_embedding

    services = client.app.state.services
    project_id = seed_project_with_embedding(
        services, 31, "acme/loved", "a beloved toolkit", ["testing"]
    )

    # a real recommendation first, so query_history has the query
    rec = client.post("/api/recommend", json={"query": "testing tools", "top_k": 2})
    assert rec.status_code == 200 and rec.json()["results"]

    ok = client.post(
        "/api/feedback",
        json={"query": "testing tools", "vote": "up", "project_id": project_id, "source": "web"},
    )
    assert ok.status_code == 200 and ok.json()["ok"] is True

    down = client.post(
        "/api/feedback",
        json={
            "query": "testing tools",
            "vote": "down",
            "project_id": project_id,
            "comment": "not what I wanted",
            "source": "api",
        },
    )
    assert down.status_code == 200

    report = client.get("/api/admin/feedback", headers={"X-Admin-Token": "test-admin-token"}).json()
    assert report["total"] == 2
    assert report["up"] == 1 and report["down"] == 1
    assert report["downvoted_projects"] == [{"project": "acme/loved", "votes": 1}]
    assert report["recent"][0]["comment"] == "not what I wanted"
    assert any("testing tools" in item["query"] for item in report["top_queries"])


def test_feedback_validates_vote(client):
    bad = client.post("/api/feedback", json={"query": "x", "vote": "meh"})
    assert bad.status_code == 422


def test_feedback_report_requires_admin(client):
    assert client.get("/api/admin/feedback").status_code == 401


def test_landing_page_and_info(client):
    root = client.get("/")
    assert root.status_code == 200
    assert "GitHub" in root.text and "text/html" in root.headers["content-type"]
    info = client.get("/info").json()
    assert info["docs"] == "/docs"
    assert "/api/recommend" in info["endpoints"]


def test_rate_limit_kicks_in(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'rl.db').as_posix()}",
        scheduler_enabled=False,
        llm_provider="rule",
        llm_fallback_providers=[],
        embedder_provider="tfidf",
        embedding_dimension=32,
        vector_store_backend="memory",
        asr_provider="none",
        tts_provider="none",
        rate_limit_per_minute=3,
    )
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings)) as client:
        codes = [client.get("/health").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429 and codes[4] == 429
