"""Recommendation service: semantic search over embedded projects."""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.llm.bge_reranker import BgeReranker
from app.llm.laya_reranker import LayaReranker
from app.models.project import Project
from app.models.user import FeedbackRecord, QueryHistory
from app.services.analysis_service import AnalysisService
from app.services.bm25 import BM25Index
from app.services.embedding_service import EmbeddingService, build_project_text, parse_embedding_id
from app.services.perf_counters import TIMERS, record_ms
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
        # feedback weights: per-project up/down vote ratios, refreshed only
        # when the feedback table changes
        self._feedback_weights: dict[int, float] = {}
        self._feedback_signature: tuple[int, str] | None = None
        self._feedback_lock = threading.Lock()
        # offline zh->en translation leg (opus-mt), constructed lazily because
        # it only materializes when the model files exist
        self._offline_translator = None
        if settings.query_translation:
            from app.utils.translator import OfflineTranslator

            candidate = OfflineTranslator(settings.offline_translator_model)
            if candidate.available():
                self._offline_translator = candidate

    def _get_feedback_weights(self) -> dict[int, float]:
        """project_id -> score multiplier in [1-bonus^-1, 1+bonus]: downvotes
        cut the score, upvotes lift it. Recomputed only when the feedback
        table's row count / newest timestamp changes."""
        with self._session_factory() as session:
            count, last_time = (
                session.query(func.count(FeedbackRecord.id), func.max(FeedbackRecord.created_at))
                .one()
            )
        signature = (int(count or 0), str(last_time))
        with self._feedback_lock:
            if self._feedback_signature == signature:
                return self._feedback_weights
            rows = (
                session.query(
                    FeedbackRecord.project_id,
                    FeedbackRecord.vote,
                    func.count(FeedbackRecord.id),
                )
                .filter(FeedbackRecord.project_id.isnot(None))
                .group_by(FeedbackRecord.project_id, FeedbackRecord.vote)
                .all()
            )
            up: dict[int, int] = {}
            down: dict[int, int] = {}
            for project_id, vote, votes in rows:
                bucket = up if vote == "up" else down
                bucket[project_id] = bucket.get(project_id, 0) + int(votes)
            weights: dict[int, float] = {}
            for project_id in set(up) | set(down):
                ups, downs = up.get(project_id, 0), down.get(project_id, 0)
                total = ups + downs
                if total == 0:
                    continue
                multiplier = 1.0
                if downs:
                    multiplier -= self._settings.feedback_penalty_weight * (downs / total)
                if ups:
                    multiplier += self._settings.feedback_upvote_bonus * (ups / total)
                weights[project_id] = max(0.5, min(1.5, multiplier))
            self._feedback_weights = weights
            self._feedback_signature = signature
            if weights:
                logger.info(f"Feedback weights refreshed: {len(weights)} projects scored.")
        return self._feedback_weights

    def _apply_feedback(self, results: list[RecommendHit]) -> None:
        if not results:
            return
        try:
            weights = self._get_feedback_weights()
        except Exception as exc:  # noqa: BLE001  # feedback must never break search
            logger.warning(f"Feedback weights unavailable (skipped): {exc}")
            return
        adjusted = False
        for hit in results:
            multiplier = weights.get(hit.project.id)
            if multiplier is not None and multiplier != 1.0:
                hit.score = round(hit.score * multiplier, 6)
                adjusted = True
        if adjusted:
            results.sort(key=lambda hit: hit.score, reverse=True)

    _CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")

    def _english_translation(self, query: str) -> str | None:
        """English rendering of a CJK query, for the dense+BM25 second channel.

        Chain: primary LLM (one call per distinct query, cached on disk) ->
        offline opus-mt model (zero cost, no API) -> None. Returns None when
        the query has no CJK, the feature is off, or every leg is
        unavailable -- the search then degrades to single-language behavior.
        """
        if not self._settings.query_translation or not self._CJK_RE.search(query):
            return None
        cache_path = Path(self._settings.query_translations_file)
        cache: dict[str, str] = {}
        try:
            if cache_path.exists():
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"Translation cache unreadable: {exc}")
        if query in cache:
            return cache[query] or None

        def store(translation: str) -> str:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache[query] = translation
                cache_path.write_text(
                    json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError as exc:
                logger.warning(f"Could not persist translation cache: {exc}")
            logger.debug(f"Translated {query!r} -> {translation!r}")
            return translation

        # leg 1: chat-capable analyzers (e.g. DeepSeek)
        for analyzer in self._analysis.analyzers:
            chat = getattr(analyzer, "chat", None)
            if chat is None or not analyzer.available():
                continue
            try:
                content = chat(
                    "Translate the user's request into one concise English sentence "
                    "describing the kind of open-source project they want. Respond with "
                    'strict JSON only: {"translation": "..."}.',
                    query,
                )
                from app.llm.base import extract_json

                translation = str(extract_json(content).get("translation") or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Query translation failed for {query!r}: {exc}")
                translation = ""
            if translation:
                return store(translation)
            break  # the primary LLM failed; don't loop through weaker analyzers

        # leg 2: offline neural translation (zero cost, no API key needed)
        translator = self._offline_translator
        if translator is not None and translator.available():
            offline = translator.translate(query)
            if offline:
                return store(offline)
        return None

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
        started = time.perf_counter()
        try:
            return self._recommend(query, top_k)
        finally:
            TIMERS.record("total", (time.perf_counter() - started) * 1000)

    def _recommend(self, query: str, top_k: int = 5) -> list[RecommendHit]:
        translation = None
        with record_ms(TIMERS, "translate"):
            translation = self._english_translation(query)
        with record_ms(TIMERS, "encode"):
            query_vectors = [self._embedding.embed(query)]
            if translation:
                query_vectors.append(self._embedding.embed(translation))
        try:
            fetch_k = top_k * 2 if self._settings.hybrid_retrieval else top_k
            # multi-language dense search: best similarity wins per document
            with record_ms(TIMERS, "dense"):
                merged: dict[str, float] = {}
                for vector in query_vectors:
                    for embedding_id, similarity in self._vectors.query(vector, top_k=fetch_k):
                        if embedding_id not in merged or similarity > merged[embedding_id]:
                            merged[embedding_id] = similarity
                hits = sorted(merged.items(), key=lambda pair: -pair[1])[:fetch_k]
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
        # dense via Reciprocal Rank Fusion when enabled. The English
        # translation joins this channel too (a Chinese query matches English
        # docs lexically only through it).
        bm25_ranked: list[int] = []
        if self._settings.hybrid_retrieval:
            with record_ms(TIMERS, "bm25"):
                try:
                    index = self._get_bm25_index()
                    if index is not None:
                        bm25_texts = [query] + ([translation] if translation else [])
                        best: dict[int, float] = {}
                        for text in bm25_texts:
                            for pid, score in index.search(text, top_k=fetch_k):
                                best[pid] = max(best.get(pid, 0.0), score)
                        bm25_ranked = [
                            pid
                            for pid, _ in sorted(best.items(), key=lambda pair: -pair[1])[:fetch_k]
                        ]
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"BM25 channel failed (dense-only this request): {exc}")

        if bm25_ranked:
            with record_ms(TIMERS, "fuse"):
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

        # external feedback: downvoted projects sink, upvoted ones rise
        if self._settings.feedback_in_ranking:
            with record_ms(TIMERS, "feedback"):
                self._apply_feedback(results)

        results = results[:top_k]

        if self._settings.enable_llm_rerank and results:
            with record_ms(TIMERS, "rerank"):
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
        for analyzer in getattr(self._analysis, "analyzers", None) or []:
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
