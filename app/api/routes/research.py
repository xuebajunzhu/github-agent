"""Problem research endpoint: problem -> direct solutions + alternatives."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.api.dependencies import get_services
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.analysis_service import AnalysisError
from app.services.container import ServiceContainer
from app.services.github_service import GitHubRateLimitError
from app.services.research_service import ResearchService

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("", response_model=ResearchResponse)
def research_problem(
    payload: ResearchRequest, services: ServiceContainer = Depends(get_services)
) -> ResearchResponse:
    try:
        result = services.research.research(payload.problem, payload.include_alternatives)
    except AnalysisError as exc:
        raise HTTPException(status_code=503, detail=f"LLM unavailable for research: {exc}") from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"GitHub rate limited: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research failed")
        raise HTTPException(status_code=500, detail="Research failed") from exc
    return ResearchResponse(**result)
