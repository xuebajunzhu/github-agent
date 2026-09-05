from __future__ import annotations

import math

from app.config import Settings
from app.embedders.tfidf_embedder import HashingTfidfEmbedder
from app.services.embedding_service import EmbeddingService


def test_deterministic_and_unit_norm():
    embedder = HashingTfidfEmbedder(64)
    vector_a = embedder.embed_text("python web framework")
    vector_b = embedder.embed_text("python web framework")
    assert vector_a == vector_b
    assert len(vector_a) == 64
    norm = math.sqrt(sum(value * value for value in vector_a))
    assert abs(norm - 1.0) < 1e-6


def test_related_texts_more_similar_than_unrelated():
    embedder = HashingTfidfEmbedder(128)

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    related_a = embedder.embed_text("a python framework for building web apis")
    related_b = embedder.embed_text("a python framework for building web applications")
    unrelated = embedder.embed_text("retro arcade game emulator written in rust")
    assert dot(related_a, related_b) > dot(related_a, unrelated)


def test_service_falls_back_when_provider_unavailable():
    settings = Settings(
        _env_file=None,
        embedder_provider="openai",
        openai_api_key=None,
        openai_embedding_api_key=None,
        embedding_dimension=32,
    )
    service = EmbeddingService(settings)
    # default fallback chain is ["trained", "tfidf"]; a trained model exists in
    # this repo, so it wins over the hashing fallback when present
    assert service.provider_name in ("trained", "tfidf")
    assert service.is_fallback
    vector = service.embed("hello world")
    expected_dim = 384 if service.provider_name == "trained" else 32
    assert len(vector) == expected_dim


def test_service_uses_configured_provider_directly():
    settings = Settings(_env_file=None, embedder_provider="tfidf", embedding_dimension=48)
    service = EmbeddingService(settings)
    assert service.provider_name == "tfidf"
    assert not service.is_fallback


def test_service_falls_back_to_tfidf_when_trained_model_missing():
    settings = Settings(
        _env_file=None,
        embedder_provider="trained",
        embedder_fallback_providers=["tfidf"],
        trained_model_path="./models/does-not-exist",
        embedding_dimension=24,
    )
    service = EmbeddingService(settings)
    assert service.provider_name == "tfidf"
    assert service.is_fallback
