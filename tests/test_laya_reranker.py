"""Tests for the Laya reranker and the rerank-provider dispatch (mocked model)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from app.config import Settings
from app.llm.laya_reranker import LayaReranker
from app.models.project import Project
from app.services.recommend_service import RecommendHit, RecommendService


class FakeRouter:
    """Canned yes-probabilities keyed by the candidate's full_name substring."""

    def __init__(self, probabilities: dict[str, float], fail: bool = False):
        self._probabilities = probabilities
        self._fail = fail
        self.calls = 0

    def predict(self, state: str, questions: dict):
        self.calls += 1
        if self._fail:
            raise RuntimeError("router exploded")
        for key, value in self._probabilities.items():
            if key in state:
                return {
                    "answers": {
                        "fit": {"type": "choice", "probabilities": {"yes": value, "no": 1 - value}}
                    }
                }
        return {"answers": {"fit": {"type": "choice", "probabilities": {"yes": 0.5, "no": 0.5}}}}


def make_reranker(probabilities: dict[str, float], fail: bool = False) -> LayaReranker:
    reranker = LayaReranker(Settings(_env_file=None))
    reranker._router = FakeRouter(probabilities, fail=fail)
    return reranker


def make_hit(full_name: str, summary: str, score: float) -> RecommendHit:
    project = Project(full_name=full_name, description=summary, stars=100, forks=1, topics=[])
    return RecommendHit(project=project, score=score, dependencies=[])


def test_rank_candidates_orders_by_yes_probability():
    reranker = make_reranker({"lf": 0.9, "deno": 0.4, "firecrawl": 0.7})
    candidates = [
        {"full_name": "acme/lf", "summary": "file manager", "tags": ["cli"]},
        {"full_name": "acme/deno", "summary": "js runtime", "tags": []},
        {"full_name": "acme/firecrawl", "summary": "web scraper", "tags": []},
    ]
    outcome = reranker.rank_candidates("终端文件管理器", candidates)
    assert outcome is not None
    ranking, reason = outcome
    assert ranking == [0, 2, 1]  # highest yes-probability first
    assert reason == ""  # decision head, not a generator


def test_rank_candidates_returns_none_when_router_fails():
    reranker = make_reranker({}, fail=True)
    assert reranker.rank_candidates("q", [{"full_name": "a", "summary": "", "tags": []}]) is None


def test_rank_candidates_unavailable_without_laya_package(monkeypatch):
    reranker = LayaReranker(Settings(_env_file=None))
    monkeypatch.setattr(LayaReranker, "available", lambda self: False)
    assert reranker.rank_candidates("q", [{"full_name": "a", "summary": "", "tags": []}]) is None


def _service_with(reranker: LayaReranker | None) -> RecommendService:
    settings = Settings(
        _env_file=None,
        enable_llm_rerank=True,
        rerank_provider="laya" if reranker is not None else "llm",
    )
    service = RecommendService(settings, None, None, None, None)
    service._laya = reranker
    return service


def test_rerank_laya_reorders_hits():
    service = _service_with(make_reranker({"lf": 0.9, "deno": 0.4}))
    hits = [make_hit("acme/deno", "js runtime", 0.9), make_hit("acme/lf", "file manager", 0.8)]
    reordered = service._rerank_laya("终端文件管理器", hits)
    assert [hit.project.full_name for hit in reordered] == ["acme/lf", "acme/deno"]


def test_rerank_laya_falls_back_to_llm_on_failure():
    service = _service_with(make_reranker({}, fail=True))

    class FallbackAnalyzer:
        name = "deepseek"

        @staticmethod
        def available():
            return True

        @staticmethod
        def rank_candidates(query, candidates):
            return [1, 0], "llm knows best"

    service._analysis = type("A", (), {"analyzers": [FallbackAnalyzer()]})()
    hits = [make_hit("acme/deno", "js runtime", 0.9), make_hit("acme/lf", "file manager", 0.8)]
    reordered = service._rerank_laya("终端文件管理器", hits)
    assert [hit.project.full_name for hit in reordered] == ["acme/lf", "acme/deno"]
    assert reordered[0].reason == "llm knows best"


def test_apply_ranking_ignores_garbage_indices():
    hits = [make_hit("a", "", 1.0), make_hit("b", "", 0.9)]
    reordered = RecommendService._apply_ranking([99, 1, -1, 1, 0], "why", hits)
    assert [hit.project.full_name for hit in reordered] == ["b", "a"]
    assert reordered[0].reason == "why"
    assert RecommendService._apply_ranking([], "why", hits) is None


def test_rerank_provider_llm_does_not_build_laya():
    settings = Settings(_env_file=None, rerank_provider="llm", enable_llm_rerank=True)
    service = RecommendService(settings, None, None, None, None)
    assert service._laya is None


def test_rerank_provider_laya_builds_reranker():
    settings = Settings(_env_file=None, rerank_provider="laya", enable_llm_rerank=True)
    service = RecommendService(settings, None, None, None, None)
    assert service._laya is not None
