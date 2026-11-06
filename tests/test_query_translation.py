"""Tests for the cross-lingual query-translation retrieval channel."""
from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.services.recommend_service import RecommendService, RecommendHit


class FakeVectors:
    """Records searches; returns deterministic per-language results."""

    def __init__(self):
        self.searched: list[list[float]] = []
        self.by_index: dict[int, list[tuple[str, float]]] = {
            0: [("project-1", 0.9), ("project-2", 0.8)],  # zh channel
            1: [("project-3", 0.95), ("project-1", 0.7)],  # en channel
        }

    def query(self, embedding, top_k=5):
        self.searched.append(list(embedding))
        # channels alternate: zh search, en search, zh, en, ...
        return self.by_index[(len(self.searched) - 1) % 2][:top_k]


class FakeEmbedder:
    def embed(self, text):
        return [float(len(text)), 0.0]


class FakeProject:
    def __init__(self, project_id, full_name):
        self.id = project_id
        self.full_name = full_name
        self.ai_dependencies = []
        self.pushed_at = None
        self.description = ""
        self.ai_summary = ""
        self.ai_tags = []


class FakeSessionFactory:
    """Returns a context manager whose query() finds our fake projects."""

    def __call__(self):
        return FakeSession()


class FakeSession:
    projects = {
        1: FakeProject(1, "acme/zh-doc"),
        2: FakeProject(2, "acme/other"),
        3: FakeProject(3, "acme/en-doc"),
    }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def query(self, *_args, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return list(self.projects.values())


class ChattingAnalyzer:
    name = "fake-llm"

    def __init__(self, fail=False):
        self.calls = 0
        self._fail = fail

    def available(self):
        return True

    def chat(self, system, user):
        self.calls += 1
        if self._fail:
            raise RuntimeError("llm down")
        return '{"translation": "self hosted deployment platform"}'


def make_service(settings, vectors, analysis) -> RecommendService:
    return RecommendService(
        settings, FakeSessionFactory(), FakeEmbedder(), vectors, analysis
    )


def make_settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        hybrid_retrieval=False,
        feedback_in_ranking=False,
        query_translations_file=str(tmp_path / "query_translations.json"),
        query_translation=True,
        # keep the real offline model out of unit tests unless a test opts in
        offline_translator_model=str(tmp_path / "no-offline-model"),
    )
    defaults.update(overrides)
    return Settings(**defaults)


class FakeAnalysis:
    def __init__(self, analyzers):
        self.analyzers = analyzers


class FakeOfflineTranslator:
    def __init__(self, text=None, fail=False):
        self.text = text
        self._fail = fail

    def available(self):
        return True

    def translate(self, text):
        if self._fail:
            return None
        return self.text


def test_translation_runs_both_channels_and_merges(tmp_path):
    vectors = FakeVectors()
    analyzer = ChattingAnalyzer()
    service = make_service(make_settings(tmp_path), vectors, FakeAnalysis([analyzer]))

    hits = service.recommend("部署平台", top_k=3)

    assert len(vectors.searched) == 2, "zh + en channels both searched"
    assert analyzer.calls == 1
    names = [hit.project.full_name for hit in hits]
    assert names[0] == "acme/en-doc", "en channel's 0.95 beats zh's 0.9"
    assert {"acme/zh-doc", "acme/other"} & set(names)
    cache = json.loads((tmp_path / "query_translations.json").read_text(encoding="utf-8"))
    assert cache["部署平台"] == "self hosted deployment platform"


def test_translation_cached_no_second_llm_call(tmp_path):
    vectors = FakeVectors()
    analyzer = ChattingAnalyzer()
    service = make_service(make_settings(tmp_path), vectors, FakeAnalysis([analyzer]))
    service.recommend("部署平台", top_k=2)
    service.recommend("部署平台", top_k=2)
    assert analyzer.calls == 1, "second query served from the disk cache"
    assert len(vectors.searched) == 4, "but both search channels still run"


def test_no_cjk_short_circuits_translation(tmp_path):
    vectors = FakeVectors()
    analyzer = ChattingAnalyzer()
    service = make_service(make_settings(tmp_path), vectors, FakeAnalysis([analyzer]))
    service.recommend("terminal file manager", top_k=2)
    assert analyzer.calls == 0
    assert len(vectors.searched) == 1, "single-channel search for non-CJK queries"


def test_llm_failure_degrades_to_single_channel(tmp_path):
    vectors = FakeVectors()
    service = make_service(
        make_settings(tmp_path), vectors, FakeAnalysis([ChattingAnalyzer(fail=True)])
    )
    hits = service.recommend("部署平台", top_k=2)
    assert len(vectors.searched) == 1, "no LLM + no offline model -> single channel"
    assert hits, "search still returns results"
    assert not (tmp_path / "query_translations.json").exists(), "failures are not cached"


def test_offline_translator_is_the_second_leg(tmp_path):
    vectors = FakeVectors()
    service = make_service(
        make_settings(tmp_path), vectors, FakeAnalysis([ChattingAnalyzer(fail=True)])
    )
    service._offline_translator = FakeOfflineTranslator("password management tool")
    hits = service.recommend("密码管理工具", top_k=3)

    assert len(vectors.searched) == 2, "offline leg provides the second channel"
    assert (tmp_path / "query_translations.json").exists(), "offline result cached too"
    names = [hit.project.full_name for hit in hits]
    assert names[0] == "acme/en-doc", "en channel result wins on merged score"


def test_offline_leg_failure_falls_back_to_single_channel(tmp_path):
    vectors = FakeVectors()
    service = make_service(
        make_settings(tmp_path), vectors, FakeAnalysis([ChattingAnalyzer(fail=True)])
    )
    service._offline_translator = FakeOfflineTranslator(fail=True)
    hits = service.recommend("密码管理工具", top_k=2)
    assert len(vectors.searched) == 1
    assert hits


def test_translation_disabled(tmp_path):
    vectors = FakeVectors()
    service = make_service(
        make_settings(tmp_path, query_translation=False), vectors, FakeAnalysis([ChattingAnalyzer()])
    )
    service.recommend("部署平台", top_k=2)
    assert len(vectors.searched) == 1
    assert not (tmp_path / "query_translations.json").exists()


def test_bm25_channel_receives_translation(tmp_path, monkeypatch):
    searched_texts: list[str] = []

    class FakeBM25:
        def search(self, text, top_k=5):
            searched_texts.append(text)
            return []

    service = make_service(make_settings(tmp_path, hybrid_retrieval=True), FakeVectors(), FakeAnalysis([ChattingAnalyzer()]))
    monkeypatch.setattr(service, "_get_bm25_index", lambda: FakeBM25())
    service.recommend("部署平台", top_k=2)
    assert "部署平台" in searched_texts
    assert "self hosted deployment platform" in searched_texts
