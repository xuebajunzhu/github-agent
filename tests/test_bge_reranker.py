"""Tests for the BGE cross-encoder reranker (mocked model + dispatch wiring)."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.llm.bge_reranker import BgeReranker
from app.models.project import Project
from app.services.recommend_service import RecommendHit, RecommendService


class FakeBgeModel:
    """Returns descending logits keyed by a substring of the passage."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        import torch

        batch = kwargs["input_ids"].shape[0]
        logits = []
        # the tokenizer input is opaque; score by call order via a fixed table
        for i in range(batch):
            logits.append([self._scores.get(str(i), 0.0)])
        return type("Out", (), {"logits": torch.tensor(logits)})()


def make_reranker(scores: dict[str, float], fail: bool = False) -> BgeReranker:
    reranker = BgeReranker(Settings(_env_file=None))
    if fail:
        reranker._failed = True
    else:
        reranker._tokenizer = _FakeTokenizer()
        reranker._model = FakeBgeModel(scores)
        reranker._torch = __import__("torch")
    return reranker


class _FakeTokenizer:
    def __call__(self, pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        import torch

        return {
            "input_ids": torch.ones((len(pairs), 4), dtype=torch.long),
            "attention_mask": torch.ones((len(pairs), 4), dtype=torch.long),
        }


def make_hit(full_name: str, summary: str, score: float) -> RecommendHit:
    project = Project(full_name=full_name, description=summary, stars=100, forks=1, topics=[])
    return RecommendHit(project=project, score=score, dependencies=[])


def test_rank_candidates_orders_by_cross_encoder_logits():
    reranker = make_reranker({"0": 1.2, "1": -0.5, "2": 3.4})
    candidates = [
        {"full_name": "acme/a", "summary": "s1", "tags": []},
        {"full_name": "acme/b", "summary": "s2", "tags": []},
        {"full_name": "acme/c", "summary": "s3", "tags": []},
    ]
    outcome = reranker.rank_candidates("query", candidates)
    assert outcome is not None
    ranking, reason = outcome
    assert ranking == [2, 0, 1]  # highest logit first
    assert reason == ""


def test_rank_candidates_returns_none_on_model_failure():
    reranker = make_reranker({}, fail=True)
    assert reranker.rank_candidates("q", [{"full_name": "a", "summary": "", "tags": []}]) is None


def test_unavailable_without_transformers(monkeypatch):
    reranker = BgeReranker(Settings(_env_file=None))
    monkeypatch.setattr(BgeReranker, "available", lambda self: False)
    assert reranker.rank_candidates("q", [{"full_name": "a", "summary": "", "tags": []}]) is None


def _service(provider: str, reranker) -> RecommendService:
    settings = Settings(_env_file=None, enable_llm_rerank=True, rerank_provider=provider)
    service = RecommendService(settings, None, None, None, None)
    if provider == "bge":
        service._bge_reranker = reranker
    elif provider == "laya":
        service._laya = reranker
    return service


def test_provider_bge_dispatches_to_bge_reranker():
    service = _service("bge", make_reranker({"0": 0.1, "1": 5.0}))
    hits = [make_hit("acme/a", "s1", 0.9), make_hit("acme/b", "s2", 0.8)]
    reordered = service._rerank_local(service._bge_reranker, "query", hits)
    assert [hit.project.full_name for hit in reordered] == ["acme/b", "acme/a"]


def test_provider_bge_falls_back_to_llm_on_failure():
    service = _service("bge", make_reranker({}, fail=True))

    class FallbackAnalyzer:
        name = "deepseek"

        @staticmethod
        def available():
            return True

        @staticmethod
        def rank_candidates(query, candidates):
            return [1, 0], "llm knows best"

    service._analysis = type("A", (), {"analyzers": [FallbackAnalyzer()]})()
    hits = [make_hit("acme/a", "s1", 0.9), make_hit("acme/b", "s2", 0.8)]
    reordered = service._rerank_local(service._bge_reranker, "query", hits)
    assert [hit.project.full_name for hit in reordered] == ["acme/b", "acme/a"]
    assert reordered[0].reason == "llm knows best"


def test_provider_llm_does_not_build_local_rerankers():
    settings = Settings(_env_file=None, rerank_provider="llm", enable_llm_rerank=True)
    service = RecommendService(settings, None, None, None, None)
    assert service._bge_reranker is None and service._laya is None
