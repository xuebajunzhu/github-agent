"""ORM model for cached problem-research results (query -> direct/alternatives)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.utils.timeutil import utcnow


class ResearchCache(Base):
    """Cached output of a problem-research run, keyed by the exact problem text.

    A research run costs several LLM calls plus GitHub API quota, so results
    are persisted and served from cache until they expire (TTL enforced by
    the caller). The JSON payload stores the full ResearchResponse shape.
    """

    __tablename__ = "research_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # sha256
    query: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
