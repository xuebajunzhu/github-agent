"""Embedder abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    name: str = "base"
    dimension: int | None = None

    @property
    def space_id(self) -> str:
        """Identity of the embedding space this embedder produces.

        Persisted vectors from a different space_id are incompatible for
        similarity search even when dimensions happen to match, so the vector
        store records it and rebuilds the index when it changes.
        """
        return self.name

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into fixed-size vectors."""

    def embed_text(self, text: str) -> list[float]:
        return self.embed([text])[0]
