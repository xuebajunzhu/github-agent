"""Scenario recommendation endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.recommend import RecommendRequest, RecommendResponse, RecommendationItem
from app.services.recommend_service import RecommendError

router = APIRouter(prefix="/api/recommend", tags=["recommend"])


@router.post("", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, request: Request) -> RecommendResponse:
    services = request.app.state.services
    try:
        hits = services.recommend.recommend(payload.query, payload.top_k)
    except RecommendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RecommendResponse(
        query=payload.query,
        results=[
            RecommendationItem(
                project=hit.project,
                score=round(hit.score, 4),
                dependencies=hit.dependencies,
                reason=hit.reason,
            )
            for hit in hits
        ],
    )
