"""Tests for CJK bigram tokenization in the hashing TF-IDF fallback.

Before the fix a whole Chinese sentence collapsed into a single hash bucket,
making the fallback path near-random for Chinese queries.
"""
from __future__ import annotations

from app.embedders.tfidf_embedder import HashingTfidfEmbedder, _tokenize
from app.utils.timeutil import maintenance_decay, maintenance_status


def test_cjk_runs_tokenize_into_bigrams():
    tokens = _tokenize("终端文件管理器")
    assert tokens == ["终端", "端文", "文件", "件管", "管理", "理器"]
    # latin behavior is unchanged
    assert _tokenize("terminal file Manager 2") == ["terminal", "file", "manager", "2"]
    # single CJK char kept as a unigram
    assert _tokenize("云") == ["云"]


def test_chinese_similarity_is_meaningful():
    embedder = HashingTfidfEmbedder(dimension=64)
    query = embedder.embed_text("我需要一个终端文件管理器")
    assert len(query) == 64  # dimension unchanged by the tokenizer

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    related = embedder.embed_text("一个在终端里运行的文件管理工具")
    unrelated = embedder.embed_text("分布式数据库连接池中间件")
    # shared bigrams (终端/文件/管理) must beat a disjoint sentence; with a
    # small hash table some bucket collisions are expected, so assert the
    # relative ordering rather than exact zero
    assert cosine(query, related) > 0
    assert cosine(query, related) > cosine(query, unrelated)


def test_maintenance_tiers_and_decay():
    from datetime import timedelta

    from app.utils.timeutil import utcnow

    assert maintenance_status(None) == "unknown"
    assert maintenance_status(utcnow() - timedelta(days=10)) == "active"
    assert maintenance_status(utcnow() - timedelta(days=400)) == "quiet"
    assert maintenance_status(utcnow() - timedelta(days=900)) == "stale"

    assert maintenance_decay(None) == 1.0  # unknown is never punished
    assert maintenance_decay(utcnow() - timedelta(days=10)) == 1.0
    assert maintenance_decay(utcnow() - timedelta(days=400)) == 0.93
    assert maintenance_decay(utcnow() - timedelta(days=900)) == 0.85
