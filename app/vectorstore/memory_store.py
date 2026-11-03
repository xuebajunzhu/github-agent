"""In-memory vector store with cosine similarity: zero-dependency fallback.

The working set is kept in a single float32 matrix with cached row norms, so
a top-k query is one vectorized matrix product instead of a per-vector Python
loop (the pure-Python version measured ~1.3s per query at 20k x 384; this one
is two orders of magnitude faster). Deletions rebuild the matrix -- they are
rare (drift repair), so the O(n) rebuild is acceptable.
"""
from __future__ import annotations

import threading

import numpy as np

from app.vectorstore.base import VectorDimensionError, VectorStore


class MemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._row: dict[str, int] = {}
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._norms = np.zeros((0,), dtype=np.float32)
        self._documents: dict[str, str] = {}
        self._metadatas: dict[str, dict] = {}
        self._dimension: int | None = None
        # patrol writes run on scheduler threads while API threads query:
        # serialize all access (queries hold it only for microseconds-range
        # matmuls, so contention is negligible).
        self._lock = threading.Lock()

    def _check_dimension(self, length: int) -> None:
        if self._dimension is None:
            self._dimension = length
        elif length != self._dimension:
            raise VectorDimensionError(
                f"Vector dimension mismatch: expected {self._dimension}, got {length}. "
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
        if not ids:
            return
        with self._lock:
            rows = np.asarray(embeddings, dtype=np.float32)
            if rows.ndim == 1:
                rows = rows.reshape(1, -1)
            self._check_dimension(rows.shape[1])
            norms = np.linalg.norm(rows, axis=1)
            # unseen ids are appended as new rows in one concatenate
            new_positions = [p for p, vector_id in enumerate(ids) if vector_id not in self._row]
            if new_positions:
                new_rows = rows[new_positions]
                new_norms = np.linalg.norm(new_rows, axis=1)
                base = len(self._ids)
                self._matrix = (
                    new_rows
                    if self._matrix.size == 0
                    else np.concatenate([self._matrix, new_rows], axis=0)
                )
                self._norms = (
                    new_norms
                    if self._norms.size == 0
                    else np.concatenate([self._norms, new_norms])
                )
                self._ids.extend(ids[p] for p in new_positions)
                for offset, p in enumerate(new_positions):
                    self._row[ids[p]] = base + offset
            # every id (new or overwriting) gets its row/docs/metadata set
            for p, vector_id in enumerate(ids):
                row = self._row[vector_id]
                self._matrix[row] = rows[p]
                self._norms[row] = norms[p]
                self._documents[vector_id] = documents[p] if documents else ""
                self._metadatas[vector_id] = metadatas[p] if metadatas else {}

    def query(self, embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        with self._lock:
            if not self._ids:
                return []
            self._check_dimension(len(embedding))
            query_vec = np.asarray(embedding, dtype=np.float32)
            query_norm = float(np.linalg.norm(query_vec))
            dots = self._matrix @ query_vec
            # zero-norm rows (or query) produce zero similarity, never NaN
            denom = self._norms * query_norm
            similarities = np.divide(
                dots, denom, out=np.zeros_like(dots, dtype=np.float32), where=denom > 0
            )
            order = (
                np.argsort(similarities)[::-1][:top_k]
                if len(self._ids) > top_k
                else np.argsort(similarities)[::-1]
            )
            return [
                (self._ids[index], float(similarities[index])) for index in order
            ]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._lock:
            doomed = {vector_id for vector_id in ids if vector_id in self._row}
            if not doomed:
                return
            keep_rows = [row for row, vector_id in enumerate(self._ids) if vector_id not in doomed]
            if keep_rows:
                kept_matrix = self._matrix[keep_rows]
                kept_norms = self._norms[keep_rows]
                kept_ids = [self._ids[row] for row in keep_rows]
            else:
                kept_matrix = np.zeros((0, self._matrix.shape[1]), dtype=np.float32)
                kept_norms = np.zeros((0,), dtype=np.float32)
                kept_ids = []
            self._matrix, self._norms, self._ids = kept_matrix, kept_norms, kept_ids
            self._row = {vector_id: row for row, vector_id in enumerate(self._ids)}
            for vector_id in doomed:
                self._documents.pop(vector_id, None)
                self._metadatas.pop(vector_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def existing_ids(self, ids: list[str]) -> set[str]:
        with self._lock:
            return {vector_id for vector_id in ids if vector_id in self._row}
