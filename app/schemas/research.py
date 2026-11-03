"""Request/response models for problem research (direct solutions + alternatives)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    problem: str = Field(
        min_length=2, max_length=500, description="The problem/need, any language"
    )
    include_alternatives: bool = Field(
        default=True, description="Also return projects with similar/substitutable functionality"
    )


class ResearchItem(BaseModel):
    full_name: str
    url: str
    description: Optional[str] = None
    stars: int = 0
    forks: int = 0
    language: Optional[str] = None
    license: Optional[str] = None
    topics: list[str] = []
    maintenance: str = "unknown"
    pushed_at: Optional[datetime] = None
    category: str  # direct | alternative
    reason: str = ""  # short LLM justification (Chinese)
    source: str = "github"  # github | index


class ResearchResponse(BaseModel):
    problem: str
    domain: str = ""
    keywords: list[str] = []  # the generated search plan (transparency)
    direct: list[ResearchItem] = []
    alternatives: list[ResearchItem] = []
    summary: Optional[str] = None  # short Chinese synthesis
    cached: bool = False
