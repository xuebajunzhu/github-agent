"""Patrol service: the 24/7 self-healing loop.

Each patrol run:
  1. requeues failed analyses for retry (optional),
  2. processes the analysis/embedding backlog,
  3. reconciles vector-store drift (e.g. after an in-memory store restart),
so the pipeline never silently stalls, no matter what happened before.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from app.config import Settings
from app.models.project import Project
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
    analyzed: int = 0
    analysis_failed: int = 0
    embedded: int = 0
    reconciled_vectors: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "requeued_failed": self.requeued_failed,
            "analyzed": self.analyzed,
            "analysis_failed": self.analysis_failed,
            "embedded": self.embedded,
            "reconciled_vectors": self.reconciled_vectors,
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
    ):
        self._settings = settings
        self._session_factory = session_factory
        self._ingest = ingest
        self._embedding = embedding
        self._vectors = vectors
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

        report.finished_at = utcnow().isoformat()
        self.last_report = report.as_dict()
        if any([report.requeued_failed, report.analyzed, report.embedded, report.reconciled_vectors]):
            logger.info(f"Patrol #{self.run_count}: {self.last_report}")
        else:
            logger.debug(f"Patrol #{self.run_count}: nothing to do.")
        return self.last_report

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
        missing_ids = self._vectors.existing_ids(ids)
        repaired = 0
        for project in projects:
            if project.embedding_id in missing_ids:
                continue
            try:
                text = build_project_text(project)
                vector = self._embedding.embed(text)
                self._vectors.upsert(
                    [project.embedding_id],
                    [vector],
                    [text],
                    [{"github_id": project.github_id, "full_name": project.full_name}],
                )
                repaired += 1
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Vector reconciliation failed for {project.full_name}: {exc}")
        if repaired:
            logger.warning(
                f"Vector drift repaired: re-upserted {repaired}/{len(projects)} embeddings."
            )
        return repaired
