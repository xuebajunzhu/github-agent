"""In-memory vector store with cosine similarity: zero-dependency fallback."""
from __future__ import annotations

import math
import threading

from app.vectorstore.base import VectorDimensionError, VectorStore


class MemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._documents: dict[str, str] = {}
        self._metadatas: dict[str, dict] = {}
        self._dimension: int | None = None
        # patrol writes run on scheduler threads while API threads query:
        # dict mutation during iteration raises RuntimeError without a lock
        self._lock = threading.Lock()

    def _check_dimension(self, vector: list[float]) -> None:
        if self._dimension is None:
            self._dimension = len(vector)
        elif len(vector) != self._dimension:
            raise VectorDimensionError(
                f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}. "
                "The embedding model likely changed -- reindex the projects."
            )

    def ensure_space(self, fingerprint: str) -> bool:
        return False  # ephemeral: always starts empty, nothing to wipe

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        with self._lock:
            for position, vector_id in enumerate(ids):
                vector = [float(value) for value in embeddings[position]]
                self._check_dimension(vector)
                self._vectors[vector_id] = vector
                self._documents[vector_id] = documents[position] if documents else ""
                self._metadatas[vector_id] = metadatas[position] if metadatas else {}

    def query(self, embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        with self._lock:
            if not self._vectors:
                return []
            self._check_dimension([float(value) for value in embedding])
            items = list(self._vectors.items())
        query_norm = math.sqrt(sum(value * value for value in embedding))
        scored: list[tuple[str, float]] = []
        for vector_id, vector in items:
            dot = sum(a * b for a, b in zip(embedding, vector))
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if query_norm == 0 or vector_norm == 0:
                similarity = 0.0
            else:
                similarity = dot / (query_norm * vector_norm)
            scored.append((vector_id, similarity))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def delete(self, ids: list[str]) -> None:
        with self._lock:
            for vector_id in ids:
                self._vectors.pop(vector_id, None)
                self._documents.pop(vector_id, None)
                self._metadatas.pop(vector_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._vectors)

    def existing_ids(self, ids: list[str]) -> set[str]:
        with self._lock:
            return {vector_id for vector_id in ids if vector_id in self._vectors}
