"""Ingest pipeline: fetch -> upsert -> analyze -> embed, per the design doc data flow."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from loguru import logger
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models.project import Project
from app.services.analysis_service import AnalysisService
from app.services.embedding_service import (
    EmbeddingService,
    build_project_text,
    project_embedding_id,
)
from app.services.github_service import (
    GitHubAPIError,
    GitHubRateLimitError,
    GitHubService,
)
from app.utils.timeutil import utcnow
from app.vectorstore.base import VectorStore


@dataclass
class IngestStats:
    fetched: int = 0
    created: int = 0
    updated: int = 0
    analyzed: int = 0
    analysis_failed: int = 0
    embedded: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class IngestService:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker,
        github: GitHubService,
        analysis: AnalysisService,
        embedding: EmbeddingService,
        vectors: VectorStore,
    ):
        self._settings = settings
        self._session_factory = session_factory
        self._github = github
        self._analysis = analysis
        self._embedding = embedding
        self._vectors = vectors
        # Rotating cursor so 24/7 scheduled fetches sweep the whole keyword
        # list instead of always hitting the same slice.
        self._keyword_cursor = 0
        self._cursor_lock = threading.Lock()
        self._analysis_pool = (
            ThreadPoolExecutor(
                max_workers=max(1, settings.analysis_workers),
                thread_name_prefix="analyze",
            )
            if settings.analysis_workers > 1
            else None
        )

    def run_scheduled_fetch(self, keywords: list[str] | None = None) -> dict:
        """Search GitHub for the configured keywords and ingest everything found.

        With keywords_per_fetch > 0 each call consumes the next slice of the
        keyword list (rotating); pass keywords explicitly to override.
        """
        if keywords is None:
            keywords = self._next_keyword_slice()
        stats = IngestStats()
        seen: set[int] = set()
        repos: list[dict] = []
        for keyword in keywords:
            try:
                found = self._github.search_repositories(keyword)
            except GitHubRateLimitError as exc:
                logger.warning(f"Stopping keyword fetch due to rate limit: {exc}")
                break
            except GitHubAPIError as exc:
                logger.error(f"GitHub search failed for keyword '{keyword}': {exc}")
                continue
            for repo in found:
                if repo["github_id"] not in seen:
                    seen.add(repo["github_id"])
                    repos.append(repo)
        stats.fetched = len(repos)
        if repos:
            result = self.ingest_repositories(repos)
            stats.created = result["created"]
            stats.updated = result["updated"]
            stats.analyzed = result["analyzed"]
            stats.analysis_failed = result["analysis_failed"]
            stats.embedded = result["embedded"]
        logger.info(f"Scheduled fetch finished: {stats.as_dict()}")
        return stats.as_dict()

    def _next_keyword_slice(self) -> list[str]:
        all_keywords = list(self._settings.github_search_keywords)
        per_fetch = self._settings.keywords_per_fetch
        if per_fetch <= 0 or per_fetch >= len(all_keywords):
            return all_keywords
        with self._cursor_lock:
            start = self._keyword_cursor % len(all_keywords)
            self._keyword_cursor = (start + per_fetch) % len(all_keywords)
        rotated = all_keywords[start:] + all_keywords[:start]
        return rotated[:per_fetch]

    def process_backlog(self) -> dict:
        """Patrol sweep: analyze pending projects, embed analyzed-but-unembedded ones."""
        stats = IngestStats()
        with self._session_factory() as session:
            self._analyze_pending(session, stats)
            self._embed_pending(session, stats)
        return stats.as_dict()

    def requeue_failed(self) -> int:
        """Put failed analyses back into the pending queue for another attempt."""
        with self._session_factory() as session:
            count = (
                session.query(Project)
                .filter(Project.analysis_status == "failed")
                .update({"analysis_status": "pending"})
            )
            session.commit()
        if count:
            logger.info(f"Requeued {count} failed analyses for retry.")
        return int(count)

    def ingest_repositories(self, repos: list[dict]) -> dict:
        """Upsert normalized repositories, analyze pending ones, embed analyzed ones."""
        stats = IngestStats(fetched=len(repos))
        with self._session_factory() as session:
            for data in repos:
                project = (
                    session.query(Project).filter_by(github_id=data["github_id"]).one_or_none()
                )
                if project is None:
                    project = Project(
                        github_id=data["github_id"],
                        full_name=data["full_name"],
                        url=data["url"],
                        description=data["description"],
                        stars=data["stars"],
                        forks=data["forks"],
                        language=data["language"],
                        topics=data["topics"],
                        pushed_at=data["pushed_at"],
                        analysis_status="pending",
                    )
                    session.add(project)
                    stats.created += 1
                    continue
                project.stars = data["stars"]
                project.forks = data["forks"]
                project.pushed_at = data["pushed_at"]
                project.topics = data["topics"]
                project.language = data["language"] or project.language
                if (data["description"] or "") != (project.description or ""):
                    project.description = data["description"]
                    # description drives the analysis/embedding -> redo both
                    project.analysis_status = "pending"
                    project.embedding_id = None
                stats.updated += 1
            session.commit()
            self._analyze_pending(session, stats)
            self._embed_pending(session, stats)
        return stats.as_dict()

    def refresh_existing(self, limit: int | None = None) -> dict:
        """Update stored projects (stars/forks/pushed_at/description) from GitHub."""
        limit = limit or self._settings.refresh_batch_size
        stats = IngestStats()
        with self._session_factory() as session:
            projects = (
                session.query(Project).order_by(Project.updated_at.asc()).limit(limit).all()
            )
            for project in projects:
                try:
                    data = self._github.get_repository(project.full_name)
                except (GitHubRateLimitError, GitHubAPIError) as exc:
                    logger.warning(f"Refresh skipped for {project.full_name}: {exc}")
                    continue
                project.stars = data["stars"]
                project.forks = data["forks"]
                project.pushed_at = data["pushed_at"]
                if (data["description"] or "") != (project.description or ""):
                    project.description = data["description"]
                    project.analysis_status = "pending"
                    project.embedding_id = None
                    stats.updated += 1
                session.commit()
        with self._session_factory() as session:
            self._analyze_pending(session, stats)
            self._embed_pending(session, stats)
        logger.info(f"Refresh finished: {stats.as_dict()}")
        return stats.as_dict()

    def _analyze_pending(self, session, stats: IngestStats) -> None:
        pending = (
            session.query(Project)
            .filter(Project.analysis_status == "pending")
            .order_by(Project.id)
            .limit(self._settings.github_max_analyze_per_run)
            .all()
        )
        if not pending:
            return

        # README fetches are GitHub-API-bound: keep them sequential in this
        # thread so rate limits are respected.
        for project in pending:
            if self._settings.github_fetch_readme and not project.readme_excerpt:
                try:
                    project.readme_excerpt = self._github.get_readme_excerpt(project.full_name)
                except (GitHubRateLimitError, GitHubAPIError) as exc:
                    logger.warning(f"README fetch skipped for {project.full_name}: {exc}")
            session.commit()

        # Analysis is CPU-bound per project and independent -> fan out.
        inputs = [project.to_analysis_dict() for project in pending]
        if self._analysis_pool is not None and len(inputs) > 1:
            outcomes = list(self._analysis_pool.map(self._analyze_one, inputs))
        else:
            outcomes = [self._analyze_one(item) for item in inputs]

        for project, (result, error) in zip(pending, outcomes):
            if error is not None:
                project.analysis_status = "failed"
                stats.analysis_failed += 1
                logger.error(f"Analysis failed for {project.full_name}: {error}")
                continue
            project.ai_summary = result.summary
            project.ai_use_cases = result.use_cases
            project.ai_tags = result.tags
            project.ai_dependencies = result.dependencies
            project.analysis_status = "done"
            project.last_analyzed_at = utcnow()
            stats.analyzed += 1
            logger.info(f"Analyzed {project.full_name} via '{result.provider}'.")
        session.commit()

    def _analyze_one(self, analysis_input: dict):
        """Runs inside worker threads; returns (result, error) so the pipeline
        never dies on a single bad project."""
        try:
            return self._analysis.analyze(analysis_input), None
        except Exception as exc:  # noqa: BLE001
            return None, exc

    def _embed_pending(self, session, stats: IngestStats) -> None:
        to_embed = (
            session.query(Project)
            .filter(Project.analysis_status == "done", Project.embedding_id.is_(None))
            .order_by(Project.id)
            .limit(self._settings.github_max_analyze_per_run)
            .all()
        )
        for project in to_embed:
            try:
                text = build_project_text(project)
                vector = self._embedding.embed(text)
                embedding_id = project_embedding_id(project.id)
                self._vectors.upsert(
                    [embedding_id],
                    [vector],
                    [text],
                    [{"github_id": project.github_id, "full_name": project.full_name}],
                )
                project.embedding_id = embedding_id
                stats.embedded += 1
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Embedding failed for {project.full_name}: {exc}")
            session.commit()
