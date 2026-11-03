"""Tests for the weekly digest endpoint (feedback optimization loop input)."""
from __future__ import annotations

from app.models.user import FeedbackRecord, QueryHistory


def test_digest_aggregates_window(client):
    services = client.app.state.services
    with services.session_factory() as session:
        session.add(QueryHistory(query="terminal file manager", top_k=5, result_count=3))
        session.add(QueryHistory(query="terminal file manager", top_k=5, result_count=2))
        session.add(QueryHistory(query="离线语音识别", top_k=5, result_count=0))
        session.add(
            FeedbackRecord(
                query="terminal file manager",
                project_full_name="acme/nnn",
                vote="up",
                source="web",
            )
        )
        session.add(
            FeedbackRecord(
                query="terminal file manager",
                project_full_name="acme/bad",
                vote="down",
                source="web",
            )
        )
        session.commit()

    response = client.get("/api/admin/digest", headers={"X-Admin-Token": "test-admin-token"})
    assert response.status_code == 200
    body = response.json()

    assert body["totals"]["queries"] == 3
    assert body["totals"]["feedback"] == 2
    # sorted by usage; the repeated query outranks the single zero-result one
    assert body["top_queries"][0] == {"query": "terminal file manager", "count": 2}
    assert body["zero_result_queries"] == [{"query": "离线语音识别", "count": 1}]
    assert body["top_upvoted_projects"] == [{"project": "acme/nnn", "votes": 1}]
    assert body["top_downvoted_projects"] == [{"project": "acme/bad", "votes": 1}]
    assert body["period_days"] == 7


def test_digest_requires_admin(client):
    assert client.get("/api/admin/digest").status_code == 401


def test_digest_days_bounds(client):
    response = client.get(
        "/api/admin/digest", params={"days": 999}, headers={"X-Admin-Token": "test-admin-token"}
    )
    assert response.status_code == 200
    assert response.json()["period_days"] == 90
