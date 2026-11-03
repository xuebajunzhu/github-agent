"""Problem research: from a user problem to (a) directly usable projects and
(b) projects with similar/substitutable functionality.

Pipeline (each stage is a plain function; no agent framework):
  1. PLAN       DeepSeek turns the problem (any language) into English search
                keywords + alternative angles. GitHub's index is mostly
                English, so cross-lingual expansion is the recall lever.
  2. GATHER     local BGE-M3 index hits (verified, with summaries) + live
                GitHub searches per keyword, merged by github_id.
  3. CLASSIFY   DeepSeek labels every candidate `direct` / `alternative` /
                `irrelevant` in batches, with a short Chinese reason.
  4. SYNTHESIZE one short Chinese summary contrasting the two groups.
  5. CACHE      keyed by problem hash with a TTL (a run costs several LLM
                calls plus GitHub quota).
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any

from loguru import logger
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.infra import get_cache
from app.llm.base import AnalysisError, extract_json
from app.models.project import Project
from app.models.research import ResearchCache
from app.schemas.research import ResearchItem, ResearchResponse
from app.services.analysis_service import AnalysisService
from app.services.embedding_service import EmbeddingService, parse_embedding_id
from app.services.github_service import GitHubRateLimitError, GitHubService
from app.utils.timeutil import maintenance_status, utcnow
from app.vectorstore.base import VectorStore

PLAN_SYSTEM = (
    "You are a GitHub research planner. The user states a problem in any language. "
    "Produce a strict JSON object: {\"domain\": \"<short English domain label>\", "
    "\"search_keywords\": [<4-6 English GitHub search queries, each 2-5 words>], "
    "\"alternative_angles\": [<0-3 short English phrases describing adjacent/alternative "
    "approaches that solve a similar need differently>]}. "
    "Search keywords must be English (GitHub's index is mostly English) and cover "
    "synonyms of the core capability. JSON only."
)

CLASSIFY_SYSTEM = (
    "You classify GitHub repositories against a user problem. Categories: "
    "\"direct\" = solves the stated problem as its core purpose; "
    "\"alternative\" = has similar/overlapping functionality usable as a substitute "
    "(different approach, different form factor, or solves the core need as a side "
    "capability); \"irrelevant\" = unrelated. Respond with strict JSON only: "
    "{\"results\": [{\"index\": <int>, \"category\": \"direct|alternative|irrelevant\", "
    "\"reason\": \"<short Chinese justification, <=30 chars>\"}]}. "
    "Every input index must appear exactly once."
)

SUMMARY_SYSTEM = (
    "You summarize research results for a Chinese-speaking developer. Given a problem "
    "and two groups of GitHub repositories (direct solutions and alternatives), write "
    "2-3 short Chinese sentences: what distinguishes the strongest direct picks, and "
    "when one would prefer the alternatives instead. Plain text, no lists."
)


class ResearchError(Exception):
    pass


class ResearchService:
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
        # L1 cache: Redis when REDIS_URL is set (cross-process, restart-safe),
        # in-memory otherwise; the research_cache table stays the durable layer
        self._hot_cache = get_cache(settings)

    # ------------------------------------------------------------------ main

    def research(self, problem: str, include_alternatives: bool = True) -> dict:
        problem = problem.strip()[:500]
        cached = self._cache_get(problem)
        if cached is not None:
            return cached

        plan = self._plan(problem)
        candidates = self._gather(problem, plan["search_keywords"])

        direct, alternatives = self._classify(problem, candidates, include_alternatives)
        direct.sort(key=lambda item: item.stars, reverse=True)
        alternatives.sort(key=lambda item: item.stars, reverse=True)

        summary = self._summarize(problem, direct, alternatives)

        response = {
            "problem": problem,
            "domain": plan.get("domain", ""),
            "keywords": plan.get("search_keywords", []),
            "direct": [item.model_dump(mode="json") for item in direct],
            "alternatives": [item.model_dump(mode="json") for item in alternatives],
            "summary": summary,
            "cached": False,
        }
        self._cache_put(problem, response)
        return response

    # ----------------------------------------------------------------- stages

    def _plan(self, problem: str) -> dict[str, Any]:
        try:
            content = self._analysis.chat(PLAN_SYSTEM, f"Problem: {problem}")
            plan = extract_json(content)
        except AnalysisError as exc:
            logger.warning(f"Research planning failed ({exc}); using the raw problem.")
            return {"domain": "", "search_keywords": [problem[:60]], "alternative_angles": []}
        keywords = [
            str(k).strip()[:60]
            for k in (plan.get("search_keywords") or [])
            if str(k).strip()
        ][: max(1, self._settings.research_keywords)]
        if not keywords:
            keywords = [problem[:60]]
        return {
            "domain": str(plan.get("domain") or "")[:80],
            "search_keywords": keywords,
            "alternative_angles": [str(a)[:80] for a in (plan.get("alternative_angles") or [])[:3]],
        }

    def _gather(self, problem: str, keywords: list[str]) -> list[dict]:
        """Local index hits first (verified, summarized), then live GitHub search,
        merged by github_id and capped."""
        merged: dict[int, dict] = {}

        try:
            query_vector = self._embedding.embed(problem)
            hits = self._vectors.query(query_vector, top_k=15)
        except Exception as exc:  # noqa: BLE001  # local search must not kill research
            logger.warning(f"Local index search failed during research: {exc}")
            hits = []
        local_ids = [
            pid for pid in (parse_embedding_id(embedding_id) for embedding_id, _ in hits) if pid
        ]
        if local_ids:
            with self._session_factory() as session:
                rows = session.query(Project).filter(Project.id.in_(local_ids)).all()
            by_project_id = {row.id: row for row in rows}
            for embedding_id, similarity in hits:
                pid = parse_embedding_id(embedding_id)
                row = by_project_id.get(pid) if pid else None
                if row is None:
                    continue
                merged[row.github_id] = self._from_project(row, similarity)
        local_count = len(merged)

        for keyword in keywords:
            try:
                found = self._github.search_repositories(
                    keyword, max_repos=10, min_stars=self._settings.research_min_stars
                )
            except GitHubRateLimitError as exc:
                logger.warning(f"Research GitHub search rate-limited at '{keyword}': {exc}")
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Research search failed for '{keyword}': {exc}")
                continue
            for repo in found:
                if repo["github_id"] not in merged:
                    merged[repo["github_id"]] = self._from_repo_dict(repo)

        cap = self._settings.research_max_candidates
        candidates = list(merged.values())[:cap]
        logger.info(
            f"Research gather: {local_count} from index + "
            f"{len(candidates) - local_count} from GitHub = {len(candidates)} candidates."
        )
        return candidates

    def _classify(
        self, problem: str, candidates: list[dict], include_alternatives: bool
    ) -> tuple[list[ResearchItem], list[ResearchItem]]:
        direct: list[ResearchItem] = []
        alternatives: list[ResearchItem] = []
        batch_size = 10
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            labels = self._classify_batch(problem, batch, start)
            for offset, candidate in enumerate(batch):
                category, reason = labels.get(
                    start + offset, ("alternative", "分类校验失败，默认归入替代项")
                )
                if category == "irrelevant":
                    continue
                if category == "direct":
                    direct.append(self._to_item(candidate, "direct", reason))
                elif include_alternatives:
                    alternatives.append(self._to_item(candidate, "alternative", reason))
        return direct, alternatives

    def _classify_batch(
        self, problem: str, batch: list[dict], start: int
    ) -> dict[int, tuple[str, str]]:
        lines = [
            f"{start + offset}. {c['full_name']}: {(c['description'] or '')[:200]} "
            f"(topics: {', '.join(c['topics'] or []) or 'none'}, stars: {c['stars']})"
            for offset, c in enumerate(batch)
        ]
        user = f"Problem: {problem}\n\nCandidates:\n" + "\n".join(lines)
        try:
            content = self._analysis.chat(CLASSIFY_SYSTEM, user)
            data = extract_json(content)
        except AnalysisError as exc:
            logger.warning(f"Classification batch failed: {exc}")
            return {}
        labels: dict[int, tuple[str, str]] = {}
        valid = {"direct", "alternative", "irrelevant"}
        for entry in data.get("results") or []:
            try:
                index = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            category = str(entry.get("category") or "").strip().lower()
            if category not in valid or not (0 <= index < len(batch)):
                continue
            labels[index] = (category, str(entry.get("reason") or "")[:60])
        return labels

    def _summarize(self, problem: str, direct: list[ResearchItem], alternatives: list[ResearchItem]) -> str | None:
        if not direct and not alternatives:
            return None
        def brief(items: list[ResearchItem]) -> str:
            return "; ".join(f"{i.full_name}({i.stars}★)" for i in items[:5]) or "无"

        try:
            user = (
                f"Problem: {problem}\n"
                f"直接可用: {brief(direct)}\n"
                f"类似可替代: {brief(alternatives)}\n"
                f"直接项理由: {' | '.join(i.reason for i in direct[:3]) or '无'}\n"
                f"替代项理由: {' | '.join(i.reason for i in alternatives[:3]) or '无'}"
            )
            return self._analysis.chat(SUMMARY_SYSTEM, user).strip()[:600]
        except AnalysisError as exc:
            logger.warning(f"Research summary failed: {exc}")
            return None

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _from_project(project: Project, similarity: float) -> dict:
        return {
            "github_id": project.github_id,
            "full_name": project.full_name,
            "url": project.url,
            "description": project.description,
            "stars": project.stars,
            "forks": project.forks,
            "language": project.language,
            "license": project.license,
            "topics": list(project.topics or []),
            "pushed_at": project.pushed_at,
            "summary": project.ai_summary or "",
            "source": "index",
            "similarity": similarity,
        }

    @staticmethod
    def _from_repo_dict(repo: dict) -> dict:
        return {
            "github_id": repo["github_id"],
            "full_name": repo["full_name"],
            "url": repo["url"],
            "description": repo["description"],
            "stars": repo["stars"],
            "forks": repo["forks"],
            "language": repo["language"],
            "license": repo.get("license"),
            "topics": list(repo.get("topics") or []),
            "pushed_at": repo.get("pushed_at"),
            "summary": "",
            "source": "github",
        }

    @staticmethod
    def _to_item(candidate: dict, category: str, reason: str) -> ResearchItem:
        return ResearchItem(
            full_name=candidate["full_name"],
            url=candidate["url"],
            description=candidate["description"],
            stars=candidate["stars"],
            forks=candidate["forks"],
            language=candidate["language"],
            license=candidate.get("license"),
            topics=candidate["topics"],
            maintenance=maintenance_status(candidate.get("pushed_at")),
            pushed_at=candidate.get("pushed_at"),
            category=category,
            reason=reason,
            source=candidate["source"],
        )

    # ------------------------------------------------------------------- cache

    @staticmethod
    def _hash(problem: str) -> str:
        return hashlib.sha256(problem.strip().lower().encode("utf-8")).hexdigest()

    def _cache_get(self, problem: str) -> dict | None:
        key = f"research:{self._hash(problem)}"
        hot = self._hot_cache.get(key)
        if hot is not None:
            hot["cached"] = True
            return hot
        ttl = self._settings.research_cache_hours
        try:
            with self._session_factory() as session:
                row = (
                    session.query(ResearchCache)
                    .filter(ResearchCache.query_hash == self._hash(problem))
                    .one_or_none()
                )
            if row is None:
                return None
            age = utcnow() - row.created_at
            if age > timedelta(hours=max(1, ttl)):
                return None
            payload = json.loads(row.result_json)
            payload["cached"] = True
            return payload
        except Exception as exc:  # noqa: BLE001  # cache is optional
            logger.warning(f"Research cache read failed: {exc}")
            return None

    def _cache_put(self, problem: str, response: dict) -> None:
        self._hot_cache.put(
            f"research:{self._hash(problem)}",
            response,
            max(1, self._settings.research_cache_hours) * 3600,
        )
        try:
            with self._session_factory() as session:
                row = (
                    session.query(ResearchCache)
                    .filter(ResearchCache.query_hash == self._hash(problem))
                    .one_or_none()
                )
                payload = dict(response)
                payload.pop("cached", None)
                if row is None:
                    row = ResearchCache(
                        query_hash=self._hash(problem),
                        query=problem,
                        result_json=json.dumps(payload, ensure_ascii=False),
                    )
                    session.add(row)
                else:
                    row.result_json = json.dumps(payload, ensure_ascii=False)
                    row.created_at = utcnow()
                session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Research cache write failed: {exc}")
