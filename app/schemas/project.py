from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, computed_field

from app.utils.timeutil import maintenance_status


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
    license: Optional[str] = None
    topics: list[str] = []
    ai_summary: Optional[str] = None
    ai_use_cases: list[str] = []
    ai_tags: list[str] = []
    ai_dependencies: Optional[list[str]] = None
    analysis_status: str
    last_analyzed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    pushed_at: Optional[datetime] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def maintenance(self) -> str:
        """Maintenance tier derived from the last push time (active/quiet/stale/unknown)."""
        return maintenance_status(self.pushed_at)
