"""ORM model for GitHub repositories and their AI analysis results."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.utils.timeutil import utcnow


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    github_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255), index=True)
    url: Mapped[str] = mapped_column(String(512))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stars: Mapped[int] = mapped_column(Integer, default=0)
    forks: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    topics: Mapped[list] = mapped_column(JSON, default=list)
    readme_excerpt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    pushed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    ai_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_use_cases: Mapped[list] = mapped_column(JSON, default=list)
    ai_tags: Mapped[list] = mapped_column(JSON, default=list)
    ai_dependencies: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    embedding_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    analysis_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    last_analyzed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_analysis_dict(self) -> dict:
        return {
            "full_name": self.full_name,
            "description": self.description,
            "language": self.language,
            "topics": list(self.topics or []),
            "readme_excerpt": self.readme_excerpt,
        }
