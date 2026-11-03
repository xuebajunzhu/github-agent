from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.project import ProjectOut


class RecommendRequest(BaseModel):
    # bounded: the query gets embedded and stored in query_history forever
    query: str = Field(
        min_length=1, max_length=300, description="Natural language scenario description"
    )
    top_k: int = Field(default=5, ge=1, le=50)


class RecommendationItem(BaseModel):
    project: ProjectOut
    score: float
    dependencies: list[str] = []
    reason: Optional[str] = None


class RecommendResponse(BaseModel):
    query: str
    results: list[RecommendationItem] = []
