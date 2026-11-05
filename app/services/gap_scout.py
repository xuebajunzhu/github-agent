"""Gap Scout: turn zero-result user queries into new index coverage.

Users type natural-language scenarios ("我需要一个xxx工具"); when semantic
search returns nothing, that query is recorded in query_history with
result_count=0 — a direct signal for what the index is missing.

This job (1) reads those zero-result queries, (2) asks the primary LLM to
extract 1-3 concrete GitHub search keywords from each, (3) appends them to
`dynamic_keywords.json`, which the ingest pipeline merges into its rotating
keyword list. Each query costs at most one LLM call, ever (results are
cached; empty expansions are also cached so dead queries don't retry).

This closes the outer feedback loop: what users search for and don't find
becomes next week's crawl targets.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.utils.timeutil import utcnow


@dataclass
class GapScoutReport:
    started_at: str
    finished_at: str = ""
    zero_result_queries: int = 0
    expanded: int = 0
    keywords: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class GapScoutService:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker,
        analysis,
        notifier=None,
    ):
        self._settings = settings
        self._session_factory = session_factory
        self._analysis = analysis
        self._notifier = notifier
        self.last_report: dict | None = None
        self.run_count = 0

    def run_scout(self) -> dict:
        report = GapScoutReport(started_at=utcnow().isoformat())
        self.run_count += 1
        try:
            self._scout(report)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"{type(exc).__name__}: {exc}")
            logger.exception("Gap scout run failed")
        report.finished_at = utcnow().isoformat()
        self.last_report = report.as_dict()
        if report.expanded:
            logger.info(f"Gap scout: {report.as_dict()}")
        else:
            logger.debug(f"Gap scout #{self.run_count}: nothing to expand.")
        if self._notifier is not None and report.errors:
            self._notifier.send(
                "[github-agent] Gap scout errors",
                "\n".join(report.errors),
                key="gap-scout-errors",
            )
        return self.last_report

    def _scout(self, report: GapScoutReport) -> None:
        limit = self._settings.gap_scout_max_queries
        if limit <= 0:
            return
        queries = self._zero_result_queries(limit)
        report.zero_result_queries = len(queries)
        if not queries:
            return

        cache_path = Path(self._settings.dynamic_keywords_file)
        cache: dict[str, list[str]] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning(f"Dynamic keywords cache unreadable, rebuilding: {exc}")
                cache = {}

        dirty = False
        for query in queries:
            if query in cache:
                continue
            cache[query] = self._extract_keywords(query)
            if cache[query]:
                report.expanded += 1
                report.keywords.extend(cache[query])
            dirty = True

        if dirty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def _zero_result_queries(self, limit: int) -> list[str]:
        from app.models.user import QueryHistory

        with self._session_factory() as session:
            rows = (
                session.query(QueryHistory.query)
                .filter(QueryHistory.result_count == 0)
                .order_by(QueryHistory.created_at.desc())
                .limit(50)
                .all()
            )
        seen: list[str] = []
        for (query,) in rows:
            if query and query not in seen:
                seen.append(query)
            if len(seen) >= limit:
                break
        return seen

    def _extract_keywords(self, query: str) -> list[str]:
        """1-3 GitHub search keywords via the primary LLM; [] when no LLM
        with a chat interface is available or the call fails (cached, so no
        retry storm)."""
        for analyzer in self._analysis.analyzers:
            chat = getattr(analyzer, "chat", None)
            if chat is None or not analyzer.available():
                continue
            try:
                content = chat(
                    "Convert the user's request into 1-3 concise GitHub repository "
                    "search keywords (English preferred, lowercase, space-separated "
                    'words). Respond with strict JSON only: {"keywords": ["...", ...]}.',
                    query,
                )
                from app.llm.base import extract_json

                data = extract_json(content)
                keywords = [
                    str(keyword).strip().lower()
                    for keyword in (data.get("keywords") or [])
                    if str(keyword).strip()
                ]
                return keywords[:3]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Keyword extraction failed for {query!r}: {exc}")
                return []
        logger.debug("No LLM with a chat interface available; query left unexpanded.")
        return []
