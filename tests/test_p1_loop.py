"""P1 loop tests: feedback -> ranking, feedback -> re-analysis, gap scout,
ingest dynamic keywords, notifier formats, nightly backup."""
from __future__ import annotations

import json

import httpx
import pytest

from app.utils.notifier import Notifier

QUERY = "testing toolkit"


def _feedback(services, project_id, vote):
    with services.session_factory() as session:
        session.add(
            __import__("app.models.user", fromlist=["FeedbackRecord"]).FeedbackRecord(
                query=QUERY, project_id=project_id, vote=vote
            )
        )
        session.commit()


def _seed_pair(services):
    """Two projects with identical text -> identical base scores."""
    from tests.test_api import seed_project_with_embedding

    first = seed_project_with_embedding(services, 61, "acme/first", QUERY, ["testing"])
    second = seed_project_with_embedding(services, 62, "acme/second", QUERY, ["testing"])
    return first, second


def test_downvote_sinks_a_project(client):
    services = client.app.state.services
    first, second = _seed_pair(services)

    before = client.post("/api/recommend", json={"query": QUERY, "top_k": 2}).json()["results"]
    names = [r["project"]["full_name"] for r in before]
    leader, laggard = names[0], names[1]
    leader_id = before[0]["project"]["id"]

    # two downvotes on the leader -> -25% multiplier -> drops below the twin
    _feedback(services, leader_id, "down")
    _feedback(services, leader_id, "down")
    after = client.post("/api/recommend", json={"query": QUERY, "top_k": 2}).json()["results"]
    assert after[0]["project"]["full_name"] == laggard
    assert after[1]["project"]["full_name"] == leader


def test_upvote_lifts_a_project(client):
    services = client.app.state.services
    first, second = _seed_pair(services)

    before = client.post("/api/recommend", json={"query": QUERY, "top_k": 2}).json()["results"]
    laggard_id = before[1]["project"]["id"]
    laggard_name = before[1]["project"]["full_name"]
    leader_name = before[0]["project"]["full_name"]

    # upvotes on the laggard (near-tie texts) -> up to +10% -> rises above
    for _ in range(3):
        _feedback(services, laggard_id, "up")
    after = client.post("/api/recommend", json={"query": QUERY, "top_k": 2}).json()["results"]
    assert after[0]["project"]["full_name"] == laggard_name
    assert after[1]["project"]["full_name"] == leader_name


def test_feedback_disabled_restores_order(client):
    services = client.app.state.services
    first, second = _seed_pair(services)

    before = client.post("/api/recommend", json={"query": QUERY, "top_k": 2}).json()["results"]
    leader_id = before[0]["project"]["id"]
    base_order = [r["project"]["full_name"] for r in before]

    _feedback(services, leader_id, "down")
    _feedback(services, leader_id, "down")
    services.settings.feedback_in_ranking = False
    results = client.post("/api/recommend", json={"query": QUERY, "top_k": 2}).json()["results"]
    assert [r["project"]["full_name"] for r in results] == base_order


def test_patrol_requeues_downvoted_projects(client):
    services = client.app.state.services
    project_id = seed_done_project(services)
    _feedback(services, project_id, "down")
    _feedback(services, project_id, "down")
    _feedback(services, project_id, "down")

    report = services.patrol.run_patrol()

    assert report["requeued_downvoted"] == 1
    with services.session_factory() as session:
        project = session.get(type(_project(services, project_id)), project_id)
        assert project.analysis_status == "done"  # re-analyzed within the same patrol
        assert project.embedding_id  # re-embedded too


def _project(services, project_id):
    from app.models.project import Project

    with services.session_factory() as session:
        return session.get(Project, project_id)


def seed_done_project(services, github_id=71, full_name="acme/requeued"):
    from app.models.project import Project
    from app.services.embedding_service import build_project_text, project_embedding_id

    with services.session_factory() as session:
        project = Project(
            github_id=github_id,
            full_name=full_name,
            url=f"https://github.com/{full_name}",
            description="a downvoted testing toolkit",
            stars=5,
            forks=1,
            language="Python",
            topics=[],
            ai_summary="a downvoted testing toolkit",
            ai_tags=["testing"],
            analysis_status="done",
        )
        session.add(project)
        session.commit()
        project_id = project.id
        text = build_project_text(project)
        project.embedding_id = project_embedding_id(project.id)
        services.vectors.upsert(
            [project.embedding_id],
            [services.embedding.embed(text)],
            [text],
            [{"github_id": github_id}],
        )
        session.commit()
    return project_id


def test_gap_scout_expands_zero_result_queries(client, tmp_path, monkeypatch):
    services = client.app.state.services
    services.settings.dynamic_keywords_file = str(tmp_path / "dynamic_keywords.json")

    with services.session_factory() as session:
        from app.models.user import QueryHistory

        session.add(QueryHistory(query="no-such-thing query", top_k=5, result_count=0))
        session.commit()

    class ChattingAnalyzer:
        name = "fake-llm"

        def available(self):
            return True

        def chat(self, system, user):
            return '{"keywords": ["no-such-thing"]}'

    services.gap_scout._analysis.analyzers = [ChattingAnalyzer()]
    report = services.gap_scout.run_scout()

    assert report["expanded"] == 1
    assert report["keywords"] == ["no-such-thing"]
    cache = json.loads((tmp_path / "dynamic_keywords.json").read_text(encoding="utf-8"))
    assert cache["no-such-thing query"] == ["no-such-thing"]
    # second run: cached, no new expansion, but keywords still served
    again = services.gap_scout.run_scout()
    assert again["expanded"] == 0
    assert services.ingest._dynamic_keywords() == ["no-such-thing"]


def test_ingest_joins_dynamic_keywords_into_fetch(client, tmp_path):
    services = client.app.state.services
    services.settings.dynamic_keywords_file = str(tmp_path / "dynamic_keywords.json")
    (tmp_path / "dynamic_keywords.json").write_text(
        json.dumps({"q": ["gap-keyword"]}), encoding="utf-8"
    )

    calls: list[str] = []

    class RecordingGitHub:
        def search_repositories(self, keyword, **kwargs):
            calls.append(keyword)
            return []

        def get_repository(self, full_name):
            raise AssertionError("not used")

        def get_readme_excerpt(self, full_name):
            return None

        def close(self):
            pass

    original = services.ingest._github
    services.ingest._github = RecordingGitHub()
    try:
        services.ingest.run_scheduled_fetch()
    finally:
        services.ingest._github = original
    assert "gap-keyword" in calls


def test_notifier_formats_and_cooldown():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    settings = type(
        "S",
        (),
        {
            "notify_webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=x",
            "notify_telegram_chat_id": None,
            "notify_cooldown_minutes": 30,
        },
    )()
    notifier = Notifier(settings, transport=httpx.MockTransport(handler))
    assert notifier.enabled
    assert notifier.send("title", "content", key="k") is True
    assert notifier.send("title2", "content2", key="k") is False  # cooldown
    request = seen[0]
    assert "qyapi.weixin.qq.com" in str(request.url)
    assert b'"msgtype":"text"' in request.content.replace(b" ", b"")


def test_notifier_generic_and_telegram_payloads():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    generic = Notifier(
        type("S", (), {"notify_webhook_url": "https://ops.example.com/hook",
                       "notify_telegram_chat_id": None, "notify_cooldown_minutes": 0})(),
        transport=httpx.MockTransport(handler),
    )
    assert generic.send("t", "c", key="g") is True
    assert json.loads(seen[-1].content)["title"] == "t"

    telegram = Notifier(
        type("S", (), {"notify_webhook_url": "https://api.telegram.org/botT/sendMessage",
                       "notify_telegram_chat_id": "42", "notify_cooldown_minutes": 0})(),
        transport=httpx.MockTransport(handler),
    )
    assert telegram.send("t", "c", key="tg") is True
    assert json.loads(seen[-1].content)["chat_id"] == "42"


def test_notifier_disabled_without_url():
    notifier = Notifier(
        type("S", (), {"notify_webhook_url": None, "notify_telegram_chat_id": None,
                       "notify_cooldown_minutes": 30})()
    )
    assert notifier.enabled is False
    assert notifier.send("t", "c") is False


def test_backup_service_round_and_retention(tmp_path):
    from app.services.backup_service import BackupService

    db_path = tmp_path / "agent.db"
    import sqlite3

    connection = sqlite3.connect(str(db_path))
    connection.execute("CREATE TABLE t (x INTEGER)")
    connection.commit()
    connection.close()
    chroma = tmp_path / "chroma_data"
    chroma.mkdir()
    (chroma / "sqlite.db").write_bytes(b"fake")

    settings = type(
        "S",
        (),
        {
            "database_url": f"sqlite:///{db_path}",
            "chroma_persist_dir": str(chroma),
            "backup_dir": str(tmp_path / "backups"),
            "backup_retention": 2,
        },
    )()
    service = BackupService(settings)

    first = service.run_backup()
    assert len(first["files"]) == 2 and not first["errors"]

    # retention: 3 extra fake db backups -> only the 2 newest survive the run
    for index in range(3):
        (tmp_path / "backups" / f"db-fake{index}.db").write_bytes(b"x")
    second = service.run_backup()
    assert not second["errors"]
    db_backups = sorted((tmp_path / "backups").glob("db-*.db"))
    assert len(db_backups) == settings.backup_retention
