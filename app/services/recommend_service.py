"""Recommendation service: semantic search over embedded projects."""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models.project import Project
from app.models.user import QueryHistory
from app.services.analysis_service import AnalysisService
from app.services.embedding_service import EmbeddingService, parse_embedding_id
from app.vectorstore.base import VectorStore, VectorStoreError


class RecommendError(Exception):
    pass


@dataclass
class RecommendHit:
    project: Project
    score: float
    dependencies: list[str]
    reason: str | None = None


class RecommendService:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker,
        embedding: EmbeddingService,
        vectors: VectorStore,
        analysis: AnalysisService,
    ):
        self._settings = settings
        self._session_factory = session_factory
        self._embedding = embedding
        self._vectors = vectors
        self._analysis = analysis

    def recommend(self, query: str, top_k: int = 5) -> list[RecommendHit]:
        query_vector = self._embedding.embed(query)
        try:
            hits = self._vectors.query(query_vector, top_k=top_k)
        except VectorStoreError as exc:
            raise RecommendError(f"Vector search failed: {exc}") from exc
        if not hits:
            return []

        project_ids = [
            project_id
            for project_id in (parse_embedding_id(embedding_id) for embedding_id, _ in hits)
            if project_id is not None
        ]
        with self._session_factory() as session:
            projects = {
                project.id: project
                for project in session.query(Project).filter(Project.id.in_(project_ids)).all()
            }

        results: list[RecommendHit] = []
        for embedding_id, similarity in hits:
            project_id = parse_embedding_id(embedding_id)
            project = projects.get(project_id) if project_id else None
            if project is None:
                continue
            results.append(
                RecommendHit(
                    project=project,
                    score=float(similarity),
                    dependencies=list(project.ai_dependencies or []),
                )
            )
        results = results[:top_k]

        if self._settings.enable_llm_rerank and results:
            results = self._rerank(query, results)

        self._record_history(query, top_k, len(results))
        return results

    def _rerank(self, query: str, results: list[RecommendHit]) -> list[RecommendHit]:
        candidates = [
            {
                "full_name": hit.project.full_name,
                "summary": hit.project.ai_summary or hit.project.description or "",
                "tags": hit.project.ai_tags or [],
            }
            for hit in results
        ]
        for analyzer in self._analysis.analyzers:
            if not analyzer.available():
                continue
            try:
                outcome = analyzer.rank_candidates(query, candidates)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"LLM rerank failed with '{analyzer.name}': {exc}")
                return results
            if not outcome:
                return results
            ranking, reason = outcome
            reordered: list[RecommendHit] = []
            for index in ranking:
                if isinstance(index, int) and 0 <= index < len(results) and results[index] not in reordered:
                    reordered.append(results[index])
            for hit in results:
                if hit not in reordered:
                    reordered.append(hit)
            reordered[0].reason = reason
            logger.info(f"Reranked {len(results)} candidates with '{analyzer.name}'.")
            return reordered
        return results

    def _record_history(self, query: str, top_k: int, result_count: int) -> None:
        try:
            with self._session_factory() as session:
                session.add(QueryHistory(query=query, top_k=top_k, result_count=result_count))
                session.commit()
        except Exception as exc:  # noqa: BLE001  # history is optional, never break recommends
            logger.warning(f"Failed to record query history: {exc}")
