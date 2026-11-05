"""Problem research endpoint: problem -> direct solutions + alternatives."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from app.api.dependencies import get_services
from app.infra import get_cache
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.analysis_service import AnalysisError
from app.services.container import ServiceContainer
from app.services.github_service import GitHubRateLimitError
from app.services.research_service import ResearchService

router = APIRouter(prefix="/api/research", tags=["research"])

# cost-control limits: one research run costs several LLM calls + GitHub
# quota, and unique problems bypass the result cache — so the per-IP budget
# is much tighter than the global rate limiter. Approximate sliding window
# via the cache backend (slight concurrency overage is acceptable; Redis
# makes it cross-process).
PER_MINUTE = 5
PER_HOUR = 30


def _research_rate_limited(request: Request, services: ServiceContainer) -> str | None:
    client = request.client
    ip = client.host if client else "unknown"
    now = time.time()
    cache = get_cache(services.settings)
    for name, limit, ttl in (("min", PER_MINUTE, 60), ("hour", PER_HOUR, 3600)):
        key = f"research-rl:{name}:{ip}"
        hits = cache.get(key) or []
        fresh = [t for t in hits if now - t < ttl]
        if len(fresh) >= limit:
            return name
        fresh.append(now)
        cache.put(key, fresh, ttl)
    return None


@router.post("", response_model=ResearchResponse)
def research_problem(
    payload: ResearchRequest,
    request: Request,
    services: ServiceContainer = Depends(get_services),
) -> ResearchResponse:
    limited = _research_rate_limited(request, services)
    if limited:
        raise HTTPException(
            status_code=429,
            detail=f"Research budget exceeded ({limited} window). Try again later.",
        )
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
