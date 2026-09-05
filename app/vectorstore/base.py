"""Vector store abstraction shared by the Chroma and in-memory backends."""
from __future__ import annotations

from abc import ABC, abstractmethod


class VectorStoreError(Exception):
    pass


class VectorDimensionError(VectorStoreError):
    pass


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update vectors by id."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """Return (id, similarity) pairs, most similar first."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove vectors by id."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored vectors."""

    @abstractmethod
    def existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of the given ids that currently exist in the store.

        Used by the 24/7 patrol to self-heal index drift (e.g. after a restart
        of an in-memory store).
        """
