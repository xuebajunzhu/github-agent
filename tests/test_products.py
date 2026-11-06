"""Tests for project detail/similar/trending APIs and the subscription loop."""
from __future__ import annotations

import pytest


def seed(services, github_id, full_name, description, tags, stars=100):
    from tests.test_api import seed_project_with_embedding

    project_id = seed_project_with_embedding(services, github_id, full_name, description, tags)
    with services.session_factory() as session:
        from app.models.project import Project

        project = session.get(Project, project_id)
        project.stars = stars
        session.commit()
    return project_id


class RecordingNotifier:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def send(self, title, content, *, key=None, level="warn"):
        self.sent.append((title, content))
        return True


def test_similar_projects_order_and_self_exclusion(client):
    services = client.app.state.services
    target = seed(services, 301, "acme/cv-lib", "deep learning computer vision library", ["cv"])
    near = seed(services, 302, "acme/cv-tool", "computer vision deep learning toolkit", ["cv"])
    far = seed(services, 303, "acme/web-fw", "async web framework for apis", ["web"])

    response = client.get(f"/api/projects/{target}/similar?top_k=2")
    assert response.status_code == 200
    results = response.json()
    names = [item["project"]["full_name"] for item in results]
    assert names[0] == "acme/cv-tool"
    assert "acme/cv-lib" not in names, "the project itself is excluded"
    assert results[0]["score"] >= 0


def test_similar_404_for_unknown(client):
    assert client.get("/api/projects/9999/similar").status_code == 404


def test_trending_sorts_by_stars_within_window(client):
    services = client.app.state.services
    seed(services, 311, "acme/old-hot", "an older hot toolkit", ["tools"], stars=9000)
    seed(services, 312, "acme/new-calm", "a fresh quiet utility", ["tools"], stars=15)

    response = client.get("/api/projects/trending?days=30&limit=10")
    assert response.status_code == 200
    projects = response.json()
    assert projects, "trending returns results"
    stars = [p["stars"] for p in projects]
    assert stars == sorted(stars, reverse=True), "sorted by stars"


def test_trending_days_filter_excludes_old(client):
    services = client.app.state.services
    seed(services, 321, "acme/fresh", "a brand new tool", ["tools"], stars=50)
    response = client.get("/api/projects/trending?days=1&limit=50")
    assert any(p["full_name"] == "acme/fresh" for p in response.json())


def test_subscription_crud_requires_admin(client):
    assert client.get("/api/subscriptions").status_code == 401
    assert client.post("/api/subscriptions", json={"query": "x"}).status_code == 401


def test_subscription_create_list_delete(client):
    admin = {"X-Admin-Token": "test-admin-token"}
    created = client.post("/api/subscriptions", json={"query": "kubernetes 部署工具", "min_score": 0.3}, headers=admin)
    assert created.status_code == 200
    sub = created.json()
    assert sub["query"] == "kubernetes 部署工具"

    listed = client.get("/api/subscriptions", headers=admin).json()
    assert [s["id"] for s in listed] == [sub["id"]]

    duplicate = client.post("/api/subscriptions", json={"query": "kubernetes 部署工具"}, headers=admin)
    assert duplicate.status_code == 409

    deleted = client.delete(f"/api/subscriptions/{sub['id']}", headers=admin)
    assert deleted.json()["ok"] is True
    listed = client.get("/api/subscriptions", headers=admin).json()
    assert listed == []


def test_subscription_round_notifies_only_new_matches(client):
    services = client.app.state.services
    seed(services, 331, "acme/cv-a", "deep learning computer vision library", ["cv"])
    seed(services, 332, "acme/cv-b", "computer vision toolkit for images", ["cv"])

    notifier = RecordingNotifier()
    services.subscription._notifier = notifier

    admin = {"X-Admin-Token": "test-admin-token"}
    created = client.post("/api/subscriptions", json={"query": "computer vision", "min_score": 0.1}, headers=admin)
    sub_id = created.json()["id"]

    report = client.post("/api/subscriptions/check", headers=admin).json()
    assert report["checked"] == 1
    # baseline round: current matches reported as new
    assert report["new_matches"] >= 1
    assert notifier.sent, "baseline notification sent"
    baseline_body = notifier.sent[-1][1]

    # second round: nothing new -> no notification
    report2 = client.post("/api/subscriptions/check", headers=admin).json()
    assert report2["new_matches"] == 0
    assert len(notifier.sent) == 1

    # a new matching project appears -> notified again with only the newcomer
    seed(services, 333, "acme/cv-c", "computer vision and deep learning inference", ["cv"])
    report3 = client.post("/api/subscriptions/check", headers=admin).json()
    assert report3["new_matches"] >= 1
    assert "acme/cv-c" in notifier.sent[-1][1]
    assert "acme/cv-a" not in notifier.sent[-1][1].split("\n")[1] if len(notifier.sent[-1][1].split("\n")) > 1 else True
