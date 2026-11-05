"""Tests for hybrid retrieval: BM25 correctness, RRF fusion, config switch."""
from __future__ import annotations

from app.config import Settings
from app.services.bm25 import BM25Index
from app.services.recommend_service import RecommendService


def test_bm25_ranks_exact_keyword_first():
    index = BM25Index(
        [1, 2, 3],
        [
            "gokcehan/lf. Terminal file manager written in Go. tags: cli, file-manager",
            "denoland/deno. A modern runtime for JavaScript and TypeScript. tags: js, runtime",
            "firecrawl/firecrawl. API to scrape and interact with the web. tags: scraper",
        ],
    )
    hits = index.search("terminal file manager", top_k=3)
    # zero-overlap docs score 0 and are filtered out entirely
    assert len(hits) == 1
    assert hits[0][0] == 1  # the exact keyword hit


def test_bm25_chinese_bigram_match():
    index = BM25Index(
        [1, 2],
        ["netdata/netdata. 实时基础设施监控与可视化面板. tags: monitoring",
         "denoland/deno. JavaScript 运行时. tags: js"],
    )
    hits = index.search("服务器监控面板", top_k=2)
    assert hits[0][0] == 1  # 监控/面板 bigrams hit doc 1, not doc 2


def test_bm25_empty_index_and_no_hits():
    assert BM25Index([], []).search("anything") == []
    index = BM25Index([1], ["completely unrelated corpus text"])
    assert index.search("zzz qqq") == []


class _FakeEmbedding:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 64


class _DegenerateVectors:
    """All documents look equally similar to the dense channel."""

    def query(self, embedding, top_k: int = 5):
        return [(f"project-{pid}", 0.5) for pid in [1, 2, 3]][:top_k]


class _KeywordCorpus:
    """BM25 channel: doc 1 matches the query keywords, docs 2-3 do not."""

    def __init__(self):
        from app.services.bm25 import BM25Index

        self._index = BM25Index(
            [1, 2, 3],
            [
                "acme/jq. Command-line JSON processor. tags: json, cli",
                "acme/web. Web framework. tags: web",
                "acme/ml. Deep learning library. tags: ml",
            ],
        )

    def search(self, query, top_k: int = 10):
        return self._index.search(query, top_k=top_k)


def test_fusion_pulls_exact_keyword_hit_into_dense_noise(tmp_path, monkeypatch):
    import json
    import tempfile

    from app.models.project import Project
    from app.database import create_db_engine, create_sessionmaker, init_db

    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'hybrid.db').as_posix()}",
        embedder_provider="tfidf",
        embedding_dimension=64,
        hybrid_retrieval=True,
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    session_factory = create_sessionmaker(engine)
    with session_factory() as session:
        # dense scores are degenerate (all 0.5): without BM25 the order is
        # arbitrary and acme/jq may not surface
        for pid, name, desc in [(1, "acme/jq", "Command-line JSON processor"),
                                (2, "acme/web", "Web framework"),
                                (3, "acme/ml", "Deep learning library")]:
            session.add(
                Project(
                    github_id=pid, full_name=name, url=f"https://github.com/{name}",
                    description=desc, stars=100, forks=1, language="Go",
                    topics=["json"] if pid == 1 else ["other"],
                    analysis_status="done", ai_summary=desc,
                )
            )
        session.commit()

    service = RecommendService(
        settings, session_factory, _FakeEmbedding(), _DegenerateVectors(), None
    )
    monkeypatch.setattr(service, "_get_bm25_index", lambda: _KeywordCorpus())

    hits = service.recommend("command line json processor", top_k=3)
    assert [hit.project.full_name for hit in hits][0] == "acme/jq", \
        "the lexical channel must decide when the dense channel is degenerate"


def test_hybrid_off_keeps_dense_only_order(tmp_path, monkeypatch):
    import tempfile

    from app.models.project import Project
    from app.database import create_db_engine, create_sessionmaker, init_db

    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'dense.db').as_posix()}",
        embedder_provider="tfidf",
        embedding_dimension=64,
        hybrid_retrieval=False,
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    session_factory = create_sessionmaker(engine)
    with session_factory() as session:
        for pid, name, desc in [(1, "acme/jq", "Command-line JSON processor"),
                                (2, "acme/web", "Web framework"),
                                (3, "acme/ml", "Deep learning library")]:
            session.add(
                Project(
                    github_id=pid, full_name=name, url=f"https://github.com/{name}",
                    description=desc, stars=100, forks=1, language="Go", topics=[],
                    analysis_status="done", ai_summary=desc,
                )
            )
        session.commit()

    dense_order = [("project-2", 1.0), ("project-3", 0.9), ("project-1", 0.8)]  # jq LAST

    class FixedVectors:
        def query(self, embedding, top_k: int = 5):
            return dense_order[:top_k]

    service = RecommendService(
        settings, session_factory, _FakeEmbedding(), FixedVectors(), None
    )
    monkeypatch.setattr(service, "_get_bm25_index", lambda: _KeywordCorpus())

    hits = service.recommend("command line json processor", top_k=3)
    # hybrid off: BM25 must not influence the order even though jq is the keyword match
    assert [hit.project.full_name for hit in hits] == ["acme/web", "acme/ml", "acme/jq"]
