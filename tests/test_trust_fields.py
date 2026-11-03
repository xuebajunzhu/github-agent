"""Tests for the trust signals surfaced by the project API (market-driven)."""
from __future__ import annotations

from datetime import timedelta

from app.models.project import Project
from app.utils.timeutil import utcnow


def _seed(services, github_id: int, full_name: str, license_id: str | None, pushed_days_ago: int | None):
    with services.session_factory() as session:
        project = Project(
            github_id=github_id,
            full_name=full_name,
            url=f"https://github.com/{full_name}",
            description="a toolkit",
            stars=500,
            forks=20,
            language="Python",
            license=license_id,
            topics=["testing"],
            analysis_status="done",
            pushed_at=(utcnow() - timedelta(days=pushed_days_ago)) if pushed_days_ago is not None else None,
        )
        session.add(project)
        session.commit()
        return project.id


def test_project_out_exposes_trust_fields(client):
    services = client.app.state.services
    project_id = _seed(services, 21, "acme/withlic", "MIT", 10)

    body = client.get(f"/api/projects/{project_id}").json()
    assert body["license"] == "MIT"
    assert body["pushed_at"] is not None
    assert body["maintenance"] == "active"


def test_maintenance_reflects_push_age(client):
    services = client.app.state.services
    stale_id = _seed(services, 22, "acme/stale", None, 900)
    unknown_id = _seed(services, 23, "acme/unknown", None, None)

    assert client.get(f"/api/projects/{stale_id}").json()["maintenance"] == "stale"
    assert client.get(f"/api/projects/{unknown_id}").json()["maintenance"] == "unknown"
