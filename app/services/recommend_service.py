"""Recommendation service: semantic search over embedded projects."""
from __future__ import annotations

import threading
from dataclasses import dataclass

from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.llm.bge_reranker import BgeReranker
from app.llm.laya_reranker import LayaReranker
from app.models.project import Project
from app.models.user import QueryHistory
from app.services.analysis_service import AnalysisService
from app.services.bm25 import BM25Index
from app.services.embedding_service import EmbeddingService, build_project_text, parse_embedding_id
from app.utils.timeutil import maintenance_decay
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
        self._laya: LayaReranker | None = None
        self._bge_reranker: BgeReranker | None = None
        if settings.rerank_provider == "laya":
            self._laya = LayaReranker(settings)
        elif settings.rerank_provider == "bge":
            self._bge_reranker = BgeReranker(settings)
        # hybrid retrieval: BM25 over the done+embedded corpus, lazily built
        # and rebuilt only when that population changes
        self._bm25: BM25Index | None = None
        self._bm25_signature: tuple[int, str] | None = None
        self._bm25_lock = threading.Lock()

    def _get_bm25_index(self) -> BM25Index | None:
        with self._session_factory() as session:
            count, last_updated = (
                session.query(func.count(Project.id), func.max(Project.updated_at))
                .filter(Project.analysis_status == "done", Project.embedding_id.isnot(None))
                .one()
            )
        signature = (int(count or 0), str(last_updated))
        with self._bm25_lock:
            if self._bm25 is None or self._bm25_signature != signature:
                with self._session_factory() as session:
                    rows = (
                        session.query(Project)
                        .filter(Project.analysis_status == "done", Project.embedding_id.isnot(None))
                        .order_by(Project.id)
                        .all()
                    )
                if not rows:
                    return None
                self._bm25 = BM25Index(
                    [row.id for row in rows],
                    [build_project_text(row) for row in rows],
                )
                self._bm25_signature = signature
                logger.info(f"BM25 index rebuilt: {len(rows)} documents.")
        return self._bm25

    @staticmethod
    def _rrf_fuse(
        dense_ranked: list[int], bm25_ranked: list[int], k: int = 60
    ) -> tuple[list[int], dict[int, float]]:
        """Reciprocal Rank Fusion of the dense and lexical rankings."""
        scores: dict[int, float] = {}
        for rank, project_id in enumerate(dense_ranked):
            scores[project_id] = scores.get(project_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, project_id in enumerate(bm25_ranked):
            scores[project_id] = scores.get(project_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda pid: -scores[pid]), scores

    def preload_reranker(self) -> None:
        """Warm the configured local reranker in the background (called by the
        container, never in unit tests -- it loads real model weights)."""
        if self._laya is not None:
            self._laya.preload_async()
        elif self._bge_reranker is not None:
            self._bge_reranker.preload_async()

    def recommend(self, query: str, top_k: int = 5) -> list[RecommendHit]:
        query_vector = self._embedding.embed(query)
        try:
            fetch_k = top_k * 2 if self._settings.hybrid_retrieval else top_k
            hits = self._vectors.query(query_vector, top_k=fetch_k)
        except VectorStoreError as exc:
            raise RecommendError(f"Vector search failed: {exc}") from exc

        dense_ranked: list[int] = []
        similarity_by_id: dict[int, float] = {}
        for embedding_id, similarity in hits:
            project_id = parse_embedding_id(embedding_id)
            if project_id is not None:
                dense_ranked.append(project_id)
                similarity_by_id[project_id] = float(similarity)

        # lexical channel: exact keyword hits that embeddings blur; fused with
        # dense via Reciprocal Rank Fusion when enabled
        bm25_ranked: list[int] = []
        if self._settings.hybrid_retrieval:
            try:
                index = self._get_bm25_index()
                if index is not None:
                    bm25_ranked = [pid for pid, _ in index.search(query, top_k=fetch_k)]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"BM25 channel failed (dense-only this request): {exc}")

        if bm25_ranked:
            project_ids, fused_scores = self._rrf_fuse(dense_ranked, bm25_ranked)
            # normalize fused RRF scores to a 0-1 display scale
            max_fused = max(fused_scores.values()) or 1.0
            score_by_id = {pid: fused_scores[pid] / max_fused for pid in project_ids}
        else:
            project_ids = dense_ranked
            score_by_id = dict(similarity_by_id)

        if not project_ids:
            return []

        with self._session_factory() as session:
            projects = {
                project.id: project
                for project in session.query(Project).filter(Project.id.in_(project_ids)).all()
            }

        results: list[RecommendHit] = []
        for project_id in project_ids:
            project = projects.get(project_id)
            if project is None:
                continue
            results.append(
                RecommendHit(
                    project=project,
                    score=round(score_by_id.get(project_id, 0.0), 6),
                    dependencies=list(project.ai_dependencies or []),
                )
            )
        # trust signal: long-unmaintained repos decay slightly so actively
        # maintained alternatives win ties; unknown pushed_at is not punished
        for hit in results:
            hit.score = round(hit.score * maintenance_decay(hit.project.pushed_at), 6)
        results.sort(key=lambda hit: hit.score, reverse=True)
        results = results[:top_k]

        if self._settings.enable_llm_rerank and results:
            if self._bge_reranker is not None:
                results = self._rerank_local(self._bge_reranker, query, results)
            elif self._laya is not None:
                results = self._rerank_local(self._laya, query, results)
            else:
                results = self._rerank(query, results)

        self._record_history(query, top_k, len(results))
        return results

    def _rerank_local(
        self, reranker, query: str, results: list[RecommendHit]
    ) -> list[RecommendHit]:
        """Shared path for local decision-head rerankers (bge / laya); falls
        back to the LLM reranker when the local one is unusable."""
        candidates = self._rerank_candidates(results)
        try:
            outcome = reranker.rank_candidates(query, candidates)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Local rerank '{reranker.name}' failed: {exc}")
            outcome = None
        if outcome is not None:
            reordered = self._apply_ranking(outcome[0], outcome[1], results)
            if reordered is not None:
                return reordered
        logger.info(f"Local rerank '{reranker.name}' unusable; falling back to the LLM reranker.")
        return self._rerank(query, results)

    def _rerank_laya(self, query: str, results: list[RecommendHit]) -> list[RecommendHit]:
        assert self._laya is not None
        return self._rerank_local(self._laya, query, results)

    @staticmethod
    def _apply_ranking(
        ranking: list[int], reason: str, results: list[RecommendHit]
    ) -> list[RecommendHit] | None:
        """Reorder hits by a reranker's index list; None if the ranking is unusable."""
        if not ranking:
            return None
        reordered: list[RecommendHit] = []
        for index in ranking:
            if isinstance(index, int) and 0 <= index < len(results) and results[index] not in reordered:
                reordered.append(results[index])
        for hit in results:
            if hit not in reordered:
                reordered.append(hit)
        if reason and reordered:
            reordered[0].reason = reason
        return reordered

    @staticmethod
    def _rerank_candidates(results: list[RecommendHit]) -> list[dict]:
        return [
            {
                "full_name": hit.project.full_name,
                "summary": hit.project.ai_summary or hit.project.description or "",
                "tags": hit.project.ai_tags or [],
            }
            for hit in results
        ]

    def _rerank(self, query: str, results: list[RecommendHit]) -> list[RecommendHit]:
        candidates = self._rerank_candidates(results)
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
            reordered = self._apply_ranking(ranking, reason, results)
            if reordered is not None:
                logger.info(f"Reranked {len(results)} candidates with '{analyzer.name}'.")
                return reordered
            return results
        return results

    def _record_history(self, query: str, top_k: int, result_count: int) -> None:
        try:
            with self._session_factory() as session:
                session.add(QueryHistory(query=query, top_k=top_k, result_count=result_count))
                session.commit()
        except Exception as exc:  # noqa: BLE001  # history is optional, never break recommends
            logger.warning(f"Failed to record query history: {exc}")
