from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TriggerFetchRequest(BaseModel):
    keywords: Optional[list[str]] = None


class TriggerFetchResponse(BaseModel):
    status: str
    keywords: Optional[list[str]] = None


class AdminStatusResponse(BaseModel):
    projects: int
    analyzed: int
    pending: int
    failed: int
    vector_count: int
    llm_provider: str
    embedder_provider: str
    vector_store: str
    scheduler_running: bool
    patrol_run_count: int = 0
    patrol_last_report: Optional[dict] = None
