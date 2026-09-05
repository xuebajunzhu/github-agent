"""Embedder abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    name: str = "base"
    dimension: int | None = None

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into fixed-size vectors."""

    def embed_text(self, text: str) -> list[float]:
        return self.embed([text])[0]
