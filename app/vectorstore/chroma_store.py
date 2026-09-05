"""Chroma-backed vector store (persistent, cosine space)."""
from __future__ import annotations

from loguru import logger

from app.vectorstore.base import VectorStore, VectorStoreError


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str, collection_name: str):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Chroma vector store ready: path={persist_dir}, collection={collection_name}"
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Chroma upsert failed: {exc}") from exc

    def query(self, embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        total = self._collection.count()
        if total == 0:
            return []
        try:
            response = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, total),
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        # cosine space: distance = 1 - similarity
        return [
            (vector_id, max(0.0, 1.0 - float(distance)))
            for vector_id, distance in zip(ids, distances)
        ]

    def delete(self, ids: list[str]) -> None:
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Chroma delete failed: {exc}") from exc

    def count(self) -> int:
        return self._collection.count()

    def existing_ids(self, ids: list[str]) -> set[str]:
        found: set[str] = set()
        # Chroma caps get() batch sizes; keep chunks modest.
        for start in range(0, len(ids), 256):
            chunk = ids[start : start + 256]
            result = self._collection.get(ids=chunk)
            found.update(result.get("ids") or [])
        return found
