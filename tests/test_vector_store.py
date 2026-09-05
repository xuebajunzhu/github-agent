from __future__ import annotations

import importlib.util

import pytest

from app.config import Settings
from app.vectorstore import create_vector_store
from app.vectorstore.base import VectorDimensionError
from app.vectorstore.memory_store import MemoryVectorStore


def test_query_returns_most_similar_first():
    store = MemoryVectorStore()
    store.upsert(
        ["a", "b", "c"],
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        documents=["doc a", "doc b", "doc c"],
    )
    results = store.query([1.0, 0.0], top_k=2)
    assert results[0][0] == "a"
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][0] == "c"
    assert len(results) == 2


def test_query_empty_store():
    store = MemoryVectorStore()
    assert store.query([1.0, 0.0]) == []
    assert store.count() == 0


def test_upsert_overwrites_same_id():
    store = MemoryVectorStore()
    store.upsert(["a"], [[1.0, 0.0]])
    store.upsert(["a"], [[0.0, 1.0]])
    assert store.count() == 1
    assert store.query([0.0, 1.0])[0][0] == "a"


def test_dimension_mismatch_raises():
    store = MemoryVectorStore()
    store.upsert(["a"], [[1.0, 0.0]])
    with pytest.raises(VectorDimensionError):
        store.upsert(["b"], [[1.0, 0.0, 0.0]])
    with pytest.raises(VectorDimensionError):
        store.query([1.0, 0.0, 0.0])


def test_delete():
    store = MemoryVectorStore()
    store.upsert(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    store.delete(["a"])
    assert store.count() == 1
    assert [vector_id for vector_id, _ in store.query([1.0, 0.0])] == ["b"]


def test_existing_ids_reports_drift():
    store = MemoryVectorStore()
    store.upsert(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    assert store.existing_ids(["a", "b", "c"]) == {"a", "b"}
    assert store.existing_ids(["x"]) == set()


def test_chroma_backend_falls_back_to_memory_without_chromadb(settings):
    if importlib.util.find_spec("chromadb") is not None:
        pytest.skip("chromadb is installed; the fallback path is not exercised")
    settings.vector_store_backend = "chroma"
    store = create_vector_store(settings)
    assert isinstance(store, MemoryVectorStore)
