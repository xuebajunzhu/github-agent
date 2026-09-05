"""Public feedback endpoints: the data source for the optimization loop."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.models.project import Project
from app.models.user import FeedbackRecord

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    vote: str = Field(pattern="^(up|down)$")
    project_id: int | None = None
    project_full_name: str | None = Field(default=None, max_length=255)
    comment: str | None = Field(default=None, max_length=2000)
    source: str | None = Field(default=None, pattern="^(web|voice|api)$")


@router.post("")
def submit_feedback(payload: FeedbackIn, request: Request) -> dict:
    services = request.app.state.services
    with services.session_factory() as session:
        if payload.project_id is not None and payload.project_full_name is None:
            project = session.get(Project, payload.project_id)
            if project is not None:
                payload.project_full_name = project.full_name
        session.add(
            FeedbackRecord(
                query=payload.query,
                project_id=payload.project_id,
                project_full_name=payload.project_full_name,
                vote=payload.vote,
                comment=payload.comment,
                source=payload.source,
            )
        )
        session.commit()
    return {"ok": True}
