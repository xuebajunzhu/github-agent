from __future__ import annotations

from app.models.project import Project
from app.services.embedding_service import project_embedding_id


def make_repo(github_id=1001, full_name="acme/ml-lib") -> dict:
    return {
        "github_id": github_id,
        "full_name": full_name,
        "url": f"https://github.com/{full_name}",
        "description": "A deep learning library for computer vision",
        "stars": 1500,
        "forks": 120,
        "language": "Python",
        "topics": ["deep-learning"],
        "pushed_at": None,
    }


class FakeGitHub:
    def __init__(self, repos: list[dict]):
        self._repos = repos

    def search_repositories(self, keyword, **kwargs):
        return [dict(repo) for repo in self._repos]

    def get_repository(self, full_name):
        return dict(self._repos[0])

    def get_readme_excerpt(self, full_name):
        return "An awesome deep learning library."

    def close(self):
        pass


def seed_project_with_embedding(services, github_id, full_name, description, tags):
    with services.session_factory() as session:
        project = Project(
            github_id=github_id,
            full_name=full_name,
            url=f"https://github.com/{full_name}",
            description=description,
            stars=100,
            forks=10,
            language="Python",
            topics=[],
            ai_summary=description,
            ai_tags=tags,
            analysis_status="done",
        )
        session.add(project)
        session.commit()
        project_id = project.id
    text = f"{full_name}. {description}. {' '.join(tags)}"
    vector = services.embedding.embed(text)
    services.vectors.upsert(
        [project_embedding_id(project_id)],
        [vector],
        [text],
        [{"github_id": github_id, "full_name": full_name}],
    )
    with services.session_factory() as session:
        stored = session.get(Project, project_id)
        stored.embedding_id = project_embedding_id(project_id)
        session.commit()
    return project_id


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    components = body["components"]
    assert components["database"] == "ok"
    assert components["llm_provider"] == "rule"
    assert components["embedder_provider"] == "tfidf"
    assert components["vector_store"] == "MemoryVectorStore"


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = client.get("/info").json()
    assert body["name"] == "github-agent"
    assert body["docs"] == "/docs"


def test_auth_login_returns_503_when_not_configured(client):
    response = client.get("/auth/github/login", follow_redirects=False)
    assert response.status_code == 503


def test_get_project_found_and_404(client):
    services = client.app.state.services
    with services.session_factory() as session:
        session.add(
            Project(
                github_id=42,
                full_name="acme/tool",
                url="https://github.com/acme/tool",
                description="A CLI tool",
                stars=10,
                forks=2,
                language="Go",
                topics=[],
            )
        )
        session.commit()

    response = client.get("/api/projects/1")
    assert response.status_code == 200
    assert response.json()["full_name"] == "acme/tool"
    assert client.get("/api/projects/999").status_code == 404


def test_recommend_ranks_by_similarity(client):
    services = client.app.state.services
    seed_project_with_embedding(
        services, 1, "acme/ml-lib", "deep learning computer vision library", ["deep-learning"]
    )
    seed_project_with_embedding(
        services, 2, "acme/web-fw", "async web framework for building apis", ["web", "api"]
    )

    response = client.post(
        "/api/recommend", json={"query": "computer vision deep learning", "top_k": 2}
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["project"]["full_name"] == "acme/ml-lib"
    assert results[0]["score"] >= results[1]["score"]


def test_recommend_requires_query(client):
    assert client.post("/api/recommend", json={"query": "", "top_k": 5}).status_code == 422


def test_admin_requires_token(client):
    assert client.post("/api/admin/trigger-fetch", json={}).status_code == 401
    assert client.get("/api/admin/status").status_code == 401


def test_admin_trigger_fetch_end_to_end(client):
    services = client.app.state.services
    # IngestService keeps its own reference to the GitHub client, patch both.
    fake_github = FakeGitHub([make_repo()])
    services.github = fake_github
    services.ingest._github = fake_github

    response = client.post(
        "/api/admin/trigger-fetch",
        headers={"X-Admin-Token": "test-admin-token"},
        json={"keywords": ["deep learning"]},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "triggered"

    status = client.get(
        "/api/admin/status", headers={"X-Admin-Token": "test-admin-token"}
    ).json()
    assert status["projects"] == 1
    assert status["analyzed"] == 1
    assert status["vector_count"] == 1

    # the full degraded pipeline must make the repo searchable
    results = client.post(
        "/api/recommend", json={"query": "deep learning computer vision", "top_k": 5}
    ).json()["results"]
    assert results
    top = results[0]["project"]
    assert top["full_name"] == "acme/ml-lib"
    assert top["analysis_status"] == "done"
    assert top["ai_summary"]
    assert "pip" in results[0]["dependencies"]


def test_recommend_records_query_history(client):
    services = client.app.state.services
    seed_project_with_embedding(services, 7, "acme/x", "a testing toolkit", ["testing"])
    client.post("/api/recommend", json={"query": "testing tools", "top_k": 1})
    with services.session_factory() as session:
        from app.models.user import QueryHistory

        rows = session.query(QueryHistory).all()
    assert len(rows) == 1
    assert rows[0].query == "testing tools"
