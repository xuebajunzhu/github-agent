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

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed_text(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)


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
