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

    def ensure_space(self, fingerprint: str) -> bool:
        """Ensure stored vectors belong to the given embedding space.

        Returns True when the store was wiped and every project must be
        re-embedded. The default backend is ephemeral/empty and never wipes;
        persistent backends should override this to detect a mismatch between
        the stored vectors and the embedder selected at startup (mixing two
        embedding spaces silently corrupts similarity search).
        """
        return False


def upsert_batch_safe(
    store: VectorStore,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str] | None = None,
    metadatas: list[dict] | None = None,
) -> list[bool]:
    """Upsert a batch in one call, degrading to per-item on failure so one bad
    vector cannot void its siblings. Returns per-item success flags."""
    try:
        store.upsert(ids, embeddings, documents, metadatas)
        return [True] * len(ids)
    except Exception:  # noqa: BLE001
        results: list[bool] = []
        for position in range(len(ids)):
            try:
                store.upsert(
                    [ids[position]],
                    [embeddings[position]],
                    [documents[position]] if documents else None,
                    [metadatas[position]] if metadatas else None,
                )
                results.append(True)
            except Exception:  # noqa: BLE001
                results.append(False)
        return results
