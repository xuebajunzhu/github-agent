"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.utils.timeutil import utcnow

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    services = request.app.state.services
    database_ok = True
    try:
        with services.session_factory() as session:
            session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        database_ok = False
    scheduler = request.app.state.scheduler
    return {
        "status": "ok" if database_ok else "degraded",
        "time": utcnow().isoformat(),
        "components": {
            "database": "ok" if database_ok else "error",
            "llm_provider": services.analysis.provider_name,
            "embedder_provider": services.embedding.provider_name,
            "embedder_is_fallback": services.embedding.is_fallback,
            "vector_store": type(services.vectors).__name__,
            "scheduler_running": bool(scheduler and scheduler.running),
            "asr_available": services.stt is not None,
            "tts_available": services.tts is not None,
        },
    }
