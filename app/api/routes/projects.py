"""Project read endpoints."""
from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_services
from app.models.project import Project
from app.schemas.project import ProjectOut
from app.services.container import ServiceContainer
from app.utils.timeutil import utcnow

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("/trending", response_model=list[ProjectOut])
def trending(
    days: int = 7,
    limit: int = 20,
    services: ServiceContainer = Depends(get_services),
) -> list[Project]:
    """Projects ingested within the last `days` days, hottest first.

    Honesty note: this is "recently added to the index, by stars" -- star
    velocity would need two snapshots per project, which the pipeline does
    not keep (yet).
    """
    days = min(max(1, days), 90)
    cutoff = utcnow() - timedelta(days=days)
    with services.session_factory() as session:
        return (
            session.query(Project)
            .filter(Project.created_at >= cutoff)
            .order_by(Project.stars.desc())
            .limit(min(max(1, limit), 100))
            .all()
        )


@router.get("/{project_id}/similar", response_model=list[dict])
def similar_projects(
    project_id: int,
    top_k: int = 5,
    services: ServiceContainer = Depends(get_services),
) -> list[dict]:
    """Semantic neighbours of a project, computed from its own indexed text."""
    from app.services.embedding_service import build_project_text

    with services.session_factory() as session:
        project = session.get(Project, project_id)
        if project is None or project.embedding_id is None:
            raise HTTPException(status_code=404, detail="Project not found or not indexed")
        text = build_project_text(project)
    try:
        vector = services.embedding.embed(text)
        hits = services.vectors.query(vector, top_k=top_k + 1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Similarity search failed: {exc}") from exc

    ids = []
    scores: dict[int, float] = {}
    for embedding_id, similarity in hits:
        from app.services.embedding_service import parse_embedding_id

        pid = parse_embedding_id(embedding_id)
        if pid is not None and pid != project_id:
            ids.append(pid)
            scores[pid] = float(similarity)
    if not ids:
        return []
    with services.session_factory() as session:
        projects = session.query(Project).filter(Project.id.in_(ids)).all()
    return [
        {"project": ProjectOut.model_validate(project).model_dump(), "score": round(scores[project.id], 4)}
        for project in sorted(projects, key=lambda p: -scores[p.id])
    ][:top_k]


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
