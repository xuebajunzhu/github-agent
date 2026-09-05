"""Project read endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_services
from app.models.project import Project
from app.schemas.project import ProjectOut
from app.services.container import ServiceContainer

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, services: ServiceContainer = Depends(get_services)) -> Project:
    with services.session_factory() as session:
        project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(
    limit: int = 20,
    offset: int = 0,
    language: str | None = None,
    services: ServiceContainer = Depends(get_services),
) -> list[Project]:
    with services.session_factory() as session:
        query = session.query(Project).order_by(Project.stars.desc())
        if language:
            query = query.filter(Project.language == language)
        return query.offset(max(0, offset)).limit(min(max(1, limit), 100)).all()
