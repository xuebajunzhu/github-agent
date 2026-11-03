"""Embedding service: resolves the embedder provider with a guaranteed fallback."""
from __future__ import annotations

from loguru import logger

from app.config import Settings
from app.embedders import HashingTfidfEmbedder, create_embedder
from app.embedders.base import Embedder
from app.models.project import Project


class EmbeddingService:
    """Resolves embedders at startup (never per-call): switching embedding models
    mid-index would silently corrupt similarity search, so availability is decided
    once and the always-available hashing TF-IDF embedder is the last resort."""

    def __init__(self, settings: Settings):
        self._settings = settings
        candidates: list[str] = list(
            dict.fromkeys([settings.embedder_provider, *settings.embedder_fallback_providers])
        )
        resolved: Embedder | None = None
        for name in candidates:
            try:
                resolved = create_embedder(name, settings)
                break  # first available provider wins; later ones are pure fallbacks
            except Exception as exc:  # noqa: BLE001  # import/config/model-load failures
                logger.warning(f"Embedder '{name}' unavailable: {exc}")
        if resolved is None:  # tfidf never fails in practice, but stay safe
            resolved = HashingTfidfEmbedder(settings.embedding_dimension)
        self.embedder = resolved
        self.is_fallback = resolved.name != settings.embedder_provider
        if self.is_fallback:
            logger.warning(
                f"Embedding degraded: using '{resolved.name}' instead of "
                f"'{settings.embedder_provider}'."
            )
        else:
            logger.info(f"Embedding provider: {resolved.name} (dimension={resolved.dimension})")

    @property
    def provider_name(self) -> str:
        return self.embedder.name

    @property
    def space_id(self) -> str:
        """Fingerprint of the embedding space actually in use; the vector
        store persists it and refuses to mix spaces across restarts."""
        return self.embedder.space_id

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed_text(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)

    def embed_batch_safe(self, texts: list[str]) -> list[list[float] | None]:
        """Batch-embed with per-item isolation: the batched forward pass is
        ~3x faster on CPU than text-by-text calls, but one malformed text
        must not void the whole batch, so failures degrade to per-item and
        produce None slots the caller can skip."""
        try:
            return list(self.embed_batch(texts))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Batch embedding failed ({exc}); falling back to per-item.")
            results: list[list[float] | None] = []
            for text in texts:
                try:
                    results.append(self.embed(text))
                except Exception as item_exc:  # noqa: BLE001
                    logger.error(f"Embedding failed for a text ({item_exc}).")
                    results.append(None)
            return results


def project_embedding_id(project_id: int) -> str:
    return f"project-{project_id}"


def parse_embedding_id(embedding_id: str) -> int | None:
    if embedding_id.startswith("project-"):
        try:
            return int(embedding_id.split("-", 1)[1])
        except ValueError:
            return None
    return None


def build_project_text(project: Project) -> str:
    """Compose the text that gets embedded for a project."""
    parts = [
        project.full_name,
        project.description or "",
        project.ai_summary or "",
        " ".join(project.ai_tags or []),
        " ".join(project.topics or []),
        project.language or "",
    ]
    return ". ".join(part.strip() for part in parts if part and part.strip())
