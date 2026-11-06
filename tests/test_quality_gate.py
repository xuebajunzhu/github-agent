"""Tests for the ingest quality gate and the StageTimers admin surface."""
from __future__ import annotations

import pytest


def make_repo(github_id, full_name, topics=None):
    return {
        "github_id": github_id,
        "full_name": full_name,
        "url": f"https://github.com/{full_name}",
        "description": "A real project",
        "stars": 100,
        "forks": 10,
        "language": "Python",
        "topics": topics or [],
        "pushed_at": None,
    }


class RecordingGitHub:
    def __init__(self, repos):
        self._repos = repos
        self.searched: list[str] = []

    def search_repositories(self, keyword, **kwargs):
        self.searched.append(keyword)
        return [dict(repo) for repo in self._repos]

    def get_repository(self, full_name):
        raise AssertionError("not used")

    def get_readme_excerpt(self, full_name):
        return None

    def close(self):
        pass


@pytest.fixture()
def gated_client(client):
    """client fixture with a RecordingGitHub swapped into ingest."""
    services = client.app.state.services
    return client, services


def test_awesome_list_filtered(client):
    services = client.app.state.services
    github = RecordingGitHub(
        [
            make_repo(101, "sindresorhus/awesome", topics=["awesome-list"]),
            make_repo(102, "acme/awesome-cli"),
            make_repo(103, "acme/real-tool"),
        ]
    )
    original = services.ingest._github
    services.ingest._github = github
    try:
        stats = services.ingest.run_scheduled_fetch(["kw"])
    finally:
        services.ingest._github = original

    assert stats["filtered_quality"] == 2, "topic awesome-list + name starting awesome"
    assert stats["filtered_duplicate"] == 0
    assert stats["fetched"] == 1, "only acme/real-tool survives"


def test_same_name_duplicate_skips_newcomer(client):
    services = client.app.state.services
    with services.session_factory() as session:
        from app.models.project import Project

        session.add(
            Project(
                github_id=200,
                full_name="pytorch/pytorch",
                url="https://github.com/pytorch/pytorch",
                description="the original",
                stars=80000,
                forks=10000,
                language="Python",
                topics=[],
                analysis_status="done",
            )
        )
        session.commit()

    github = RecordingGitHub(
        [
            make_repo(201, "some-fork/pytorch"),
            make_repo(202, "pytorch/pytorch"),  # the original itself: not a dup of itself
        ]
    )
    original = services.ingest._github
    services.ingest._github = github
    try:
        stats = services.ingest.run_scheduled_fetch(["kw"])
    finally:
        services.ingest._github = original

    assert stats["filtered_duplicate"] == 1, "the fork is skipped, the original passes"
    assert stats["fetched"] == 1


def test_quality_gate_can_be_disabled(client):
    services = client.app.state.services
    services.settings.ingest_quality_filter = False
    github = RecordingGitHub([make_repo(101, "sindresorhus/awesome", topics=["awesome-list"])])
    original = services.ingest._github
    services.ingest._github = github
    try:
        stats = services.ingest.run_scheduled_fetch(["kw"])
    finally:
        services.ingest._github = original
        services.settings.ingest_quality_filter = True

    assert stats["filtered_quality"] == 0
    assert stats["fetched"] == 1


def test_timers_snapshot_after_recommend(client):
    from tests.test_api import seed_project_with_embedding

    services = client.app.state.services
    seed_project_with_embedding(services, 81, "acme/timed", "a timed testing toolkit", ["testing"])
    client.post("/api/recommend", json={"query": "testing toolkit", "top_k": 2})

    response = client.get(
        "/api/admin/timers", headers={"X-Admin-Token": "test-admin-token"}
    )
    assert response.status_code == 200
    stages = response.json()["stages"]
    for stage in ("encode", "total"):
        assert stage in stages, f"stage {stage} recorded"
        assert stages[stage]["count"] >= 1
        assert stages[stage]["p95_ms"] >= stages[stage]["p50_ms"] >= 0


def test_timers_requires_admin(client):
    assert client.get("/api/admin/timers").status_code == 401


def test_stage_timers_percentile_math():
    from app.services.perf_counters import StageTimers

    timers = StageTimers()
    for value in range(1, 101):
        timers.record("stage", float(value))
    snapshot = timers.snapshot()["stage"]
    assert snapshot["count"] == 100
    # nearest-index percentile: round(0.50 * 99) -> index 50 -> value 51
    assert snapshot["p50_ms"] == 51.0
    assert snapshot["p95_ms"] == 95.0
    assert snapshot["avg_ms"] == 50.5
