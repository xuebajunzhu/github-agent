from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    github_id: int
    full_name: str
    url: str
    description: Optional[str] = None
    stars: int = 0
    forks: int = 0
    language: Optional[str] = None
    topics: list[str] = []
    ai_summary: Optional[str] = None
    ai_use_cases: list[str] = []
    ai_tags: list[str] = []
    ai_dependencies: Optional[list[str]] = None
    analysis_status: str
    last_analyzed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
