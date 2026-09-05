"""Vector store package with a backend factory (Chroma, memory fallback)."""
from __future__ import annotations

from loguru import logger

from app.config import Settings
from app.vectorstore.base import VectorStore, VectorDimensionError, VectorStoreError
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.memory_store import MemoryVectorStore

__all__ = [
    "VectorStore",
    "VectorStoreError",
    "VectorDimensionError",
    "ChromaVectorStore",
    "MemoryVectorStore",
    "create_vector_store",
]


def create_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store_backend == "memory":
        return MemoryVectorStore()
    if settings.vector_store_backend == "chroma":
        try:
            return ChromaVectorStore(settings.chroma_persist_dir, settings.chroma_collection_name)
        except ImportError:
            logger.warning(
                "chromadb is not installed; degrading to the in-memory vector store. "
                "Install requirements-ml.txt for persistence."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Chroma init failed ({exc}); degrading to the in-memory vector store.")
        return MemoryVectorStore()
    raise ValueError(f"Unknown vector store backend: {settings.vector_store_backend}")
