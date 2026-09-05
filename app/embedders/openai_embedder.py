"""OpenAI embeddings API client."""
from __future__ import annotations

from app.config import Settings
from app.embedders.base import Embedder


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, settings: Settings):
        api_key = settings.openai_embedding_api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI embedder requires OPENAI_EMBEDDING_API_KEY or OPENAI_API_KEY")
        self._api_key = api_key
        self._model = settings.openai_embedding_model
        self.dimension = settings.embedding_dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, timeout=60.0)
        response = client.embeddings.create(model=self._model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if vectors:
            self.dimension = len(vectors[0])
        return vectors
