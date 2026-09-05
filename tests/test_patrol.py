from __future__ import annotations

from app.config import Settings
from app.models.project import Project
from app.services.ingest_service import IngestService


def seed_raw_project(services, github_id, full_name, description, status):
    with services.session_factory() as session:
        session.add(
            Project(
                github_id=github_id,
                full_name=full_name,
                url=f"https://github.com/{full_name}",
                description=description,
                stars=10,
                forks=1,
                language="Python",
                topics=[],
                analysis_status=status,
            )
        )
        session.commit()


def test_patrol_processes_backlog_end_to_end(client):
    services = client.app.state.services
    seed_raw_project(services, 11, "acme/vision", "deep learning computer vision library", "pending")
    seed_raw_project(services, 12, "acme/broken", "a tool that failed before", "failed")

    report = services.patrol.run_patrol()

    assert report["requeued_failed"] == 1  # failed -> pending
    assert report["analyzed"] == 2  # both projects analyzed by the rule chain
    assert report["analysis_failed"] == 0
    assert report["embedded"] == 2  # both embedded into the vector store
    assert report["reconciled_vectors"] == 0
    assert services.vectors.count() == 2

    status = services.patrol.last_report
    assert status == report


def test_patrol_self_heals_vector_drift(client):
    services = client.app.state.services
    seed_raw_project(services, 21, "acme/tool", "a cli tool for files", "pending")
    services.patrol.run_patrol()
    assert services.vectors.count() == 1

    # simulate an in-memory store restart: all vectors are gone
    embedding_ids = []
    with services.session_factory() as session:
        for project in session.query(Project).all():
            embedding_ids.append(project.embedding_id)
    services.vectors.delete([embedding_id for embedding_id in embedding_ids if embedding_id])
    assert services.vectors.count() == 0

    report = services.patrol.run_patrol()

    assert report["reconciled_vectors"] == 1  # drift repaired
    assert services.vectors.count() == 1
    # the DB was not re-analyzed (embedding_id unchanged) - no duplicate work
    assert report["analyzed"] == 0


def test_keyword_rotation_sweeps_the_list(settings):
    settings.keywords_per_fetch = 2
    settings.github_search_keywords = ["a", "b", "c", "d"]
    calls: list[str] = []

    class RecordingGitHub:
        def search_repositories(self, keyword, **kwargs):
            calls.append(keyword)
            return []

        def get_readme_excerpt(self, full_name):
            return None

        def close(self):
            pass

    service = IngestService(settings, None, RecordingGitHub(), None, None, None)
    for _ in range(3):
        service.run_scheduled_fetch()

    assert calls == ["a", "b", "c", "d", "a", "b"]


def test_explicit_keywords_bypass_rotation(client):
    services = client.app.state.services
    calls: list[str] = []

    class RecordingGitHub:
        def search_repositories(self, keyword, **kwargs):
            calls.append(keyword)
            return []

        def get_readme_excerpt(self, full_name):
            return None

        def close(self):
            pass

    original = services.github
    services.github = RecordingGitHub()
    services.ingest._github = services.github
    try:
        services.ingest.run_scheduled_fetch(["explicit"])
    finally:
        services.ingest._github = original
        services.github = original
    assert calls == ["explicit"]
