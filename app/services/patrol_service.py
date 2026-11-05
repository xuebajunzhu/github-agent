"""Patrol service: the 24/7 self-healing loop.

Each patrol run:
  1. requeues failed analyses for retry (optional),
  2. requeues heavily downvoted projects for re-analysis with the current,
     improved pipeline (feedback -> quality loop),
  3. processes the analysis/embedding backlog,
  4. reconciles vector-store drift (e.g. after an in-memory store restart),
so the pipeline never silently stalls, no matter what happened before.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy import func

from app.config import Settings
from app.models.project import Project
from app.models.user import FeedbackRecord
from app.services.embedding_service import (
    EmbeddingService,
    build_project_text,
    project_embedding_id,
)
from app.services.ingest_service import IngestService
from app.utils.timeutil import utcnow
from app.vectorstore.base import VectorStore


@dataclass
class PatrolReport:
    started_at: str
    finished_at: str = ""
    requeued_failed: int = 0
    requeued_downvoted: int = 0
    analyzed: int = 0
    analysis_failed: int = 0
    embedded: int = 0
    reconciled_vectors: int = 0
    pruned_queries: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "requeued_failed": self.requeued_failed,
            "requeued_downvoted": self.requeued_downvoted,
            "analyzed": self.analyzed,
            "analysis_failed": self.analysis_failed,
            "embedded": self.embedded,
            "reconciled_vectors": self.reconciled_vectors,
            "pruned_queries": self.pruned_queries,
            "errors": self.errors,
        }


class PatrolService:
    def __init__(
        self,
        settings: Settings,
        session_factory,
        ingest: IngestService,
        embedding: EmbeddingService,
        vectors: VectorStore,
        notifier=None,
    ):
        self._settings = settings
        self._session_factory = session_factory
        self._ingest = ingest
        self._embedding = embedding
        self._vectors = vectors
        self._notifier = notifier
        self.last_report: dict | None = None
        self.run_count = 0

    def run_patrol(self) -> dict:
        report = PatrolReport(started_at=utcnow().isoformat())
        self.run_count += 1

        if self._settings.patrol_requeue_failed:
            try:
                report.requeued_failed = self._ingest.requeue_failed()
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"requeue_failed: {exc}")
                logger.error(f"Patrol requeue failed: {exc}")

        if getattr(self._settings, "patrol_requeue_downvote_threshold", 0) > 0:
            try:
                report.requeued_downvoted = self._requeue_downvoted()
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"requeue_downvoted: {exc}")
                logger.error(f"Patrol downvote requeue failed: {exc}")

        try:
            backlog = self._ingest.process_backlog()
            report.analyzed = backlog["analyzed"]
            report.analysis_failed = backlog["analysis_failed"]
            report.embedded = backlog["embedded"]
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"process_backlog: {exc}")
            logger.error(f"Patrol backlog processing failed: {exc}")

        if self._settings.patrol_reconcile_vectors:
            try:
                report.reconciled_vectors = self.reconcile_vectors()
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"reconcile_vectors: {exc}")
                logger.error(f"Patrol vector reconciliation failed: {exc}")

        try:
            report.pruned_queries = self._prune_query_history()
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"prune_query_history: {exc}")
            logger.error(f"Patrol query-history pruning failed: {exc}")

        report.finished_at = utcnow().isoformat()
        self.last_report = report.as_dict()
        if any(
            [
                report.requeued_failed,
                report.requeued_downvoted,
                report.analyzed,
                report.embedded,
                report.reconciled_vectors,
                report.pruned_queries,
            ]
        ):
            logger.info(f"Patrol #{self.run_count}: {self.last_report}")
        else:
            logger.debug(f"Patrol #{self.run_count}: nothing to do.")
        if self._notifier is not None and report.errors:
            self._notifier.send(
                f"[github-agent] Patrol reported {len(report.errors)} error(s)",
                "\n".join(report.errors),
                key="patrol-errors",
            )
        return self.last_report

    def _requeue_downvoted(self) -> int:
        """Feedback -> quality loop: projects rejected by several voters go
        back into the analysis queue so the current (improved) pipeline —
        e.g. with README content now enabled — takes another shot at them."""
        threshold = self._settings.patrol_requeue_downvote_threshold
        limit = self._settings.patrol_requeue_downvote_limit
        with self._session_factory() as session:
            downvote_counts = (
                session.query(FeedbackRecord.project_id, func.count(FeedbackRecord.id))
                .filter(FeedbackRecord.vote == "down", FeedbackRecord.project_id.isnot(None))
                .group_by(FeedbackRecord.project_id)
                .having(func.count(FeedbackRecord.id) >= threshold)
                .limit(limit)
                .all()
            )
            if not downvote_counts:
                return 0
            project_ids = [project_id for project_id, _ in downvote_counts]
            requeued = (
                session.query(Project)
                .filter(
                    Project.id.in_(project_ids),
                    Project.analysis_status == "done",
                )
                .update(
                    {"analysis_status": "pending", "embedding_id": None},
                    synchronize_session=False,
                )
            )
            session.commit()
        if requeued:
            logger.warning(
                f"Requeued {requeued} downvoted projects for re-analysis "
                f"(>= {threshold} downvotes)."
            )
        return int(requeued)

    def _prune_query_history(self) -> int:
        """query_history grows on every recommend; without a retention pass the
        admin aggregations (and the DB itself) degrade over months of 24/7 running."""
        from datetime import timedelta

        from app.models.research import ResearchCache
        from app.models.user import QueryHistory

        pruned = 0
        retention = self._settings.query_history_retention_days
        if retention > 0:
            cutoff = utcnow() - timedelta(days=retention)
            with self._session_factory() as session:
                pruned += session.query(QueryHistory).filter(QueryHistory.created_at < cutoff).delete()
                session.commit()
            if pruned:
                logger.info(f"Pruned {pruned} query-history rows older than {retention} days.")
        research_days = getattr(self._settings, "research_cache_retention_days", 7)
        if research_days > 0:
            cutoff = utcnow() - timedelta(days=research_days)
            with self._session_factory() as session:
                research_pruned = (
                    session.query(ResearchCache).filter(ResearchCache.created_at < cutoff).delete()
                )
                session.commit()
            if research_pruned:
                logger.info(f"Pruned {research_pruned} expired research-cache rows.")
            pruned += research_pruned
        return pruned

    def reconcile_vectors(self, limit: int | None = None) -> int:
        """Re-embed projects whose vectors vanished from the store (drift)."""
        limit = limit or self._settings.patrol_reconcile_limit
        with self._session_factory() as session:
            rows = (
                session.query(Project)
                .filter(
                    Project.analysis_status == "done",
                    Project.embedding_id.isnot(None),
                )
                .order_by(Project.id)
                .limit(limit)
                .all()
            )
            ids = [row.embedding_id for row in rows]
            projects = rows
        if not ids:
            return 0
        present_ids = self._vectors.existing_ids(ids)
        missing = [project for project in projects if project.embedding_id not in present_ids]
        if not missing:
            return 0
        # batched rebuild of the drifted slice (embed once, upsert once)
        texts = [build_project_text(project) for project in missing]
        vectors = self._embedding.embed_batch_safe(texts)
        usable = [
            (project, text, vector)
            for project, text, vector in zip(missing, texts, vectors)
            if vector is not None
        ]
        repaired = 0
        if usable:
            from app.vectorstore.base import upsert_batch_safe

            ok = upsert_batch_safe(
                self._vectors,
                [project.embedding_id for project, _, _ in usable],
                [vector for _, _, vector in usable],
                [text for _, text, _ in usable],
                [
                    {"github_id": project.github_id, "full_name": project.full_name}
                    for project, _, _ in usable
                ],
            )
            repaired = sum(ok)
        if repaired:
            logger.warning(
                f"Vector drift repaired: re-upserted {repaired}/{len(projects)} embeddings."
            )
        return repaired
