"""Tests for the BGE-M3 embedder (wiring +, when the model is cached, numerics)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.embedders.bge_m3_embedder import BgeM3Embedder


def test_space_id_names_model_and_provider():
    # bypass __init__ (no torch/model needed): space_id is pure plumbing
    embedder = BgeM3Embedder.__new__(BgeM3Embedder)
    embedder._model_name = "BAAI/bge-m3"
    embedder.dimension = 1024
    assert embedder.space_id == "bge-m3:BAAI/bge-m3"
    assert embedder.name == "bge-m3"


def test_factory_builds_bge_m3(monkeypatch):
    import app.embedders.bge_m3_embedder as bge_module

    created = {}

    class FakeBgeM3(BgeM3Embedder):
        def __init__(self, settings):
            created["settings"] = settings
            self._model_name = "BAAI/bge-m3"
            self.dimension = 1024

    monkeypatch.setattr(bge_module, "BgeM3Embedder", FakeBgeM3)
    from app.embedders import create_embedder

    settings = Settings(_env_file=None)
    embedder = create_embedder("bge-m3", settings)
    assert embedder.space_id == "bge-m3:BAAI/bge-m3"
    assert created["settings"] is settings


def _model_cached() -> bool:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained("BAAI/bge-m3", local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _model_cached(), reason="BAAI/bge-m3 not downloaded")
def test_bge_m3_numerics_and_batching():
    embedder = BgeM3Embedder(Settings(_env_file=None, bge_m3_max_len=512))
    vectors = embedder.embed(
        ["终端文件管理器", "一个在终端里运行的管理文件的工具", "分布式数据库连接池"]
    )
    assert len(vectors) == 3
    assert len(vectors[0]) == 1024
    norms = [sum(v * v for v in vec) ** 0.5 for vec in vectors]
    assert all(abs(norm - 1.0) < 1e-3 for norm in norms), "CLS embeddings must be L2-normalized"

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot  # already normalized

    related = cosine(vectors[0], vectors[1])
    unrelated = cosine(vectors[0], vectors[2])
    assert related > unrelated > 0, "multilingual model must separate related Chinese texts"
