"""Tests for problem research (plan -> gather -> classify -> cache), fully faked."""
from __future__ import annotations

import json
from datetime import timedelta

from app.config import Settings
from app.database import create_db_engine, create_sessionmaker, init_db
from app.llm.base import LLMAnalyzer
from app.models.research import ResearchCache
from app.services.analysis_service import AnalysisService
from app.services.github_service import GitHubRateLimitError
from app.services.research_service import ResearchService
from app.utils.timeutil import utcnow


class FakeChatAnalyzer(LLMAnalyzer):
    """Returns canned plan/classify/summary JSON; counts chat calls."""

    name = "fake-llm"

    def __init__(self):
        self.calls = 0

    def available(self) -> bool:
        return True

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        if "research planner" in system:
            return json.dumps(
                {
                    "domain": "terminal file management",
                    "search_keywords": ["terminal file manager", "tui file browser"],
                    "alternative_angles": ["gui file managers"],
                }
            )
        if "classify GitHub repositories" in system:
            # label index 0 direct, 1 alternative, 2 irrelevant (one batch)
            return json.dumps(
                {
                    "results": [
                        {"index": 0, "category": "direct", "reason": "就是终端文件管理器"},
                        {"index": 1, "category": "alternative", "reason": "双面板替代形态"},
                        {"index": 2, "category": "irrelevant", "reason": "JS运行时"},
                    ]
                }
            )
        return "这是一个研究摘要"

    def analyze(self, project_data: dict):
        raise NotImplementedError


def make_repo(github_id: int, full_name: str, stars: int) -> dict:
    return {
        "github_id": github_id,
        "full_name": full_name,
        "url": f"https://github.com/{full_name}",
        "description": f"{full_name} description",
        "stars": stars,
        "forks": 5,
        "language": "Go",
        "license": "MIT",
        "topics": ["cli"],
        "pushed_at": utcnow(),
    }


class FakeGitHub:
    def __init__(self, repos: list[dict], rate_limited: bool = False):
        self._repos = repos
        self._rate_limited = rate_limited
        self.search_calls = 0

    def search_repositories(self, keyword, **kwargs):
        self.search_calls += 1
        if self._rate_limited:
            raise GitHubRateLimitError("rate limited")
        return [dict(repo) for repo in self._repos]

    def close(self):
        pass


class FakeEmbedding:
    def embed(self, text: str) -> list[float]:
        return [0.0] * 64


class FakeVectors:
    def query(self, embedding, top_k: int = 5):
        return []


def make_service(tmp_path, github: FakeGitHub, llm: FakeChatAnalyzer | None = None):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'research.db').as_posix()}",
        embedder_provider="tfidf",
        embedding_dimension=64,
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    session_factory = create_sessionmaker(engine)
    analysis = AnalysisService([llm or FakeChatAnalyzer()])
    return ResearchService(
        settings, session_factory, github, analysis, FakeEmbedding(), FakeVectors()
    )


def test_research_splits_direct_and_alternatives(tmp_path):
    llm = FakeChatAnalyzer()
    github = FakeGitHub(
        [
            make_repo(1, "acme/lf", 9000),
            make_repo(2, "acme/mc", 4000),
            make_repo(3, "acme/deno", 100000),
        ]
    )
    service = make_service(tmp_path, github, llm)

    result = service.research("我需要一个终端文件管理器")

    assert [item["full_name"] for item in result["direct"]] == ["acme/lf"]
    assert result["direct"][0]["reason"] == "就是终端文件管理器"
    assert [item["full_name"] for item in result["alternatives"]] == ["acme/mc"]
    assert result["cached"] is False
    assert result["keywords"] == ["terminal file manager", "tui file browser"]
    assert result["summary"] == "这是一个研究摘要"
    assert result["direct"][0]["maintenance"] in {"active", "quiet", "stale"}


def test_research_uses_cache_on_second_call(tmp_path):
    llm = FakeChatAnalyzer()
    service = make_service(tmp_path, FakeGitHub([make_repo(1, "acme/lf", 100)]), llm)

    first = service.research("cache hit terminal manager")
    assert first["cached"] is False
    calls_after_first = llm.calls

    second = service.research("Cache Hit Terminal Manager ")  # same after normalize
    assert second["cached"] is True
    assert llm.calls == calls_after_first, "cache hit must not spend LLM calls"
    assert second["direct"][0]["full_name"] == "acme/lf"


def test_research_cache_expires(tmp_path):
    llm = FakeChatAnalyzer()
    service = make_service(tmp_path, FakeGitHub([make_repo(1, "acme/lf", 100)]), llm)

    problem = "cache expiry terminal manager"
    service.research(problem)
    calls_after_first = llm.calls

    with service._session_factory() as session:
        row = session.query(ResearchCache).one()
        row.created_at = utcnow() - timedelta(hours=25)
        session.commit()
    # the hot layer shares the DB TTL; clear it to prove the DB layer expires
    service._hot_cache.delete(f"research:{service._hash(problem)}")

    result = service.research(problem)
    assert result["cached"] is False
    assert llm.calls > calls_after_first


def test_research_survives_rate_limit(tmp_path):
    service = make_service(tmp_path, FakeGitHub([], rate_limited=True))
    result = service.research("rate limit survives monitoring problem")
    assert result["direct"] == [] and result["alternatives"] == []


def test_unclassified_batch_defaults_to_alternative(tmp_path):
    class BadClassifier(FakeChatAnalyzer):
        def chat(self, system: str, user: str) -> str:
            if "classify" in system:
                return "not json at all"  # extract_json raises -> batch lost
            return super().chat(system, user)

    service = make_service(tmp_path, FakeGitHub([make_repo(1, "acme/lf", 100)]), BadClassifier())
    result = service.research("bad classifier unclassified problem")
    # classified batch was unparseable: items survive as alternatives, nothing vanishes
    assert result["direct"] == []
    assert len(result["alternatives"]) == 1
    assert result["alternatives"][0]["reason"] == "分类校验失败，默认归入替代项"
