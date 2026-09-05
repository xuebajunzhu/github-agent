"""Local sentence-transformers embedder (default, offline capable)."""
from __future__ import annotations

from app.embedders.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    name = "sentence_transformers"

    def __init__(self, model_name: str, dimension: int | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = dimension or self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(value) for value in vector] for vector in vectors]
