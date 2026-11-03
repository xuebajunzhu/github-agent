"""Weekly digest: turn the agent's own telemetry into actionable content.

The feedback optimization loop described in the README (zero-result queries ->
new keywords; downvoted projects -> quality checks; frequent queries ->
coverage) needs its input aggregated somewhere convenient. This service
produces that aggregate over a rolling window so it can be reviewed via API,
cron, or a future newsletter channel.
"""
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from app.models.project import Project
from app.models.user import FeedbackRecord, QueryHistory
from app.utils.timeutil import utcnow


class DigestService:
    def __init__(self, settings, session_factory: sessionmaker):
        self._settings = settings
        self._session_factory = session_factory

    def build(self, days: int = 7) -> dict:
        days = max(1, min(days, 90))
        since = utcnow().replace(microsecond=0)
        from datetime import timedelta

        window_start = since - timedelta(days=days)
        with self._session_factory() as session:
            new_projects = (
                session.query(Project)
                .filter(Project.created_at >= window_start)
                .order_by(Project.stars.desc())
                .limit(10)
                .all()
            )
            top_queries = (
                session.query(QueryHistory.query, func.count(QueryHistory.id).label("uses"))
                .filter(QueryHistory.created_at >= window_start)
                .group_by(QueryHistory.query)
                .order_by(func.count(QueryHistory.id).desc())
                .limit(10)
                .all()
            )
            zero_result = (
                session.query(QueryHistory.query, func.count(QueryHistory.id).label("hits"))
                .filter(QueryHistory.created_at >= window_start, QueryHistory.result_count == 0)
                .group_by(QueryHistory.query)
                .order_by(func.count(QueryHistory.id).desc())
                .limit(10)
                .all()
            )
            upvoted = (
                session.query(
                    FeedbackRecord.project_full_name,
                    func.count(FeedbackRecord.id).label("votes"),
                )
                .filter(
                    FeedbackRecord.created_at >= window_start,
                    FeedbackRecord.vote == "up",
                    FeedbackRecord.project_full_name.isnot(None),
                )
                .group_by(FeedbackRecord.project_full_name)
                .order_by(func.count(FeedbackRecord.id).desc())
                .limit(10)
                .all()
            )
            downvoted = (
                session.query(
                    FeedbackRecord.project_full_name,
                    func.count(FeedbackRecord.id).label("votes"),
                )
                .filter(
                    FeedbackRecord.created_at >= window_start,
                    FeedbackRecord.vote == "down",
                    FeedbackRecord.project_full_name.isnot(None),
                )
                .group_by(FeedbackRecord.project_full_name)
                .order_by(func.count(FeedbackRecord.id).desc())
                .limit(10)
                .all()
            )
            feedback_total = (
                session.query(FeedbackRecord)
                .filter(FeedbackRecord.created_at >= window_start)
                .count()
            )
            query_total = (
                session.query(QueryHistory)
                .filter(QueryHistory.created_at >= window_start)
                .count()
            )
            new_project_count = (
                session.query(Project)
                .filter(Project.created_at >= window_start)
                .count()
            )
        return {
            "generated_at": since.isoformat(),
            "period_days": days,
            "window_start": window_start.isoformat(),
            "totals": {
                "queries": query_total,
                "new_projects": new_project_count,
                "feedback": feedback_total,
            },
            # zero-result queries are the keyword-expansion input: users asked
            # for something the index does not cover yet
            "zero_result_queries": [{"query": q, "count": c} for q, c in zero_result],
            "top_queries": [{"query": q, "count": c} for q, c in top_queries],
            "top_upvoted_projects": [{"project": name, "votes": v} for name, v in upvoted],
            "top_downvoted_projects": [{"project": name, "votes": v} for name, v in downvoted],
            "new_projects": [
                {
                    "full_name": p.full_name,
                    "url": p.url,
                    "stars": p.stars,
                    "language": p.language,
                    "description": p.description,
                }
                for p in new_projects
            ],
        }
