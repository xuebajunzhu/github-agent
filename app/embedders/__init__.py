"""Pluggable embedder package with a provider factory."""
from __future__ import annotations

from app.config import Settings
from app.embedders.base import Embedder
from app.embedders.openai_embedder import OpenAIEmbedder
from app.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.embedders.tfidf_embedder import HashingTfidfEmbedder

__all__ = [
    "Embedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "HashingTfidfEmbedder",
    "TrainedEmbedder",
    "create_embedder",
]


def create_embedder(name: str, settings: Settings) -> Embedder:
    """Instantiate an embedder by provider name; raises on invalid name/config."""
    if name == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model_name)
    if name == "openai":
        return OpenAIEmbedder(settings)
    if name == "tfidf":
        return HashingTfidfEmbedder(settings.embedding_dimension)
    if name == "trained":
        from app.embedders.trained_embedder import TrainedEmbedder

        return TrainedEmbedder(settings.trained_model_path)
    raise ValueError(f"Unknown embedder provider: {name}")
