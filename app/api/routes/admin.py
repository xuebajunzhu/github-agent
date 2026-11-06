"""Admin endpoints: manual fetch/patrol triggers and pipeline status."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from loguru import logger
from sqlalchemy import text

from app.api.dependencies import get_services, require_admin
from app.llm import prompt_registry
from app.schemas.admin import AdminStatusResponse, TriggerFetchRequest, TriggerFetchResponse
from app.services.container import ServiceContainer

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _run_fetch_safely(services: ServiceContainer, keywords: list[str] | None) -> None:
    try:
        services.ingest.run_scheduled_fetch(keywords)
    except Exception:  # noqa: BLE001
        logger.exception("Manual fetch trigger failed")


def _run_patrol_safely(services: ServiceContainer) -> None:
    try:
        services.patrol.run_patrol()
    except Exception:  # noqa: BLE001
        logger.exception("Manual patrol trigger failed")


def _run_evolution_safely(services: ServiceContainer) -> None:
    try:
        services.evolution.run_cycle()
    except Exception:  # noqa: BLE001
        logger.exception("Manual evolution trigger failed")


def _run_prompt_evolution_safely(services: ServiceContainer) -> None:
    try:
        services.prompt_evolution.run_cycle()
    except Exception:  # noqa: BLE001
        logger.exception("Manual prompt evolution trigger failed")


@router.post("/trigger-fetch", response_model=TriggerFetchResponse)
def trigger_fetch(
    background_tasks: BackgroundTasks,
    request: Request,
    payload: TriggerFetchRequest | None = None,
    services: ServiceContainer = Depends(get_services),
) -> TriggerFetchResponse:
    keywords = payload.keywords if payload else None
    background_tasks.add_task(_run_fetch_safely, services, keywords)
    return TriggerFetchResponse(status="triggered", keywords=keywords)


@router.post("/run-patrol", response_model=TriggerFetchResponse)
def run_patrol(
    background_tasks: BackgroundTasks,
    services: ServiceContainer = Depends(get_services),
) -> TriggerFetchResponse:
    """Manually trigger one patrol sweep (requeue failures + backlog + self-heal)."""
    background_tasks.add_task(_run_patrol_safely, services)
    return TriggerFetchResponse(status="patrol_triggered", keywords=None)


@router.post("/run-evolution", response_model=TriggerFetchResponse)
def run_evolution(
    background_tasks: BackgroundTasks,
    services: ServiceContainer = Depends(get_services),
) -> TriggerFetchResponse:
    """Manually trigger one self-evolution cycle (harvest/train/gate/promote)."""
    background_tasks.add_task(_run_evolution_safely, services)
    return TriggerFetchResponse(status="evolution_triggered", keywords=None)


@router.post("/run-prompt-evolution", response_model=TriggerFetchResponse)
def run_prompt_evolution(
    background_tasks: BackgroundTasks,
    services: ServiceContainer = Depends(get_services),
) -> TriggerFetchResponse:
    """Trigger one prompt-evolution cycle (harvest/mutate/gate/promote the
    analysis system prompt). Costs a few hundred flash-tier LLM calls."""
    background_tasks.add_task(_run_prompt_evolution_safely, services)
    return TriggerFetchResponse(status="prompt_evolution_triggered", keywords=None)


@router.get("/breakers")
def breaker_status(services: ServiceContainer = Depends(get_services)):
    """Per-analyzer circuit breaker state (DFR observability): consecutive
    failure counts and whether the circuit is currently open."""
    return services.analysis.breaker_states()


@router.post("/backup-drill", response_model=None)
def backup_drill(services: ServiceContainer = Depends(get_services)) -> dict:
    """Restore drill (DFR): restores the newest backups into staging, verifies
    SQLite integrity and row counts against the live database. Read-only for
    live files; staging is cleaned up afterwards."""
    return services.backup.restore_drill()


@router.get("/evolution")
def evolution_status(services: ServiceContainer = Depends(get_services)):
    return {
        "enabled": services.settings.evolution_enabled,
        "interval_minutes": services.settings.evolution_interval_minutes,
        "run_count": services.evolution.run_count,
        "current_version": services.evolution.current_version(),
        "last_report": services.evolution.last_report,
        "history": services.evolution.history[-10:],
        "prompt_evolution": {
            "current_prompt_version": prompt_registry.current_version(),
            "last_report": services.prompt_evolution.last_report,
            "run_count": services.prompt_evolution.run_count,
        },
    }


@router.get("/status", response_model=AdminStatusResponse)
def admin_status(
    request: Request, services: ServiceContainer = Depends(get_services)
) -> AdminStatusResponse:
    with services.session_factory() as session:
        total = _count(session, "SELECT COUNT(*) FROM projects")
        analyzed = _count(session, "SELECT COUNT(*) FROM projects WHERE analysis_status = 'done'")
        pending = _count(session, "SELECT COUNT(*) FROM projects WHERE analysis_status = 'pending'")
        failed = _count(session, "SELECT COUNT(*) FROM projects WHERE analysis_status = 'failed'")
    scheduler = request.app.state.scheduler
    return AdminStatusResponse(
        projects=total,
        analyzed=analyzed,
        pending=pending,
        failed=failed,
        vector_count=services.vectors.count(),
        llm_provider=services.analysis.provider_name,
        embedder_provider=services.embedding.provider_name,
        vector_store=type(services.vectors).__name__,
        scheduler_running=bool(scheduler and scheduler.running),
        patrol_last_report=services.patrol.last_report,
        patrol_run_count=services.patrol.run_count,
    )


@router.get("/digest", response_model=None)
def weekly_digest(
    request: Request, days: int = 7, _svc: ServiceContainer = Depends(get_services)
):
    """Rolling-window digest of the feedback optimization loop: top/zero-result
    queries, voted projects and newly indexed repos. This is the input for
    keyword expansion and the raw material for a weekly content channel."""
    services = request.app.state.services
    return services.digest.build(days=days)


@router.get("/feedback", response_model=None)
def feedback_report(
    request: Request, limit: int = 50, days: int = 30, _svc: ServiceContainer = Depends(get_services)
):
    """Aggregated external-user feedback: the input for the optimization loop.

    The GROUP BY aggregations are windowed (default 30 days) so they stay flat
    as query_history grows instead of full-scanning an ever-larger table."""
    services = request.app.state.services
    from datetime import timedelta

    from sqlalchemy import func

    from app.models.user import FeedbackRecord, QueryHistory
    from app.utils.timeutil import utcnow

    days = max(1, min(days, 365))
    window_start = utcnow() - timedelta(days=days)
    with services.session_factory() as session:
        total = session.query(FeedbackRecord).count()
        up = session.query(FeedbackRecord).filter(FeedbackRecord.vote == "up").count()
        down = session.query(FeedbackRecord).filter(FeedbackRecord.vote == "down").count()
        downvoted = (
            session.query(
                FeedbackRecord.project_full_name,
                func.count(FeedbackRecord.id).label("votes"),
            )
            .filter(
                FeedbackRecord.created_at >= window_start,
                FeedbackRecord.vote == "down",
                FeedbackRecord.project_full_name.isnot(None),
            )
            .group_by(FeedbackRecord.project_full_name)
            .order_by(func.count(FeedbackRecord.id).desc())
            .limit(20)
            .all()
        )
        top_queries = (
            session.query(QueryHistory.query, func.count(QueryHistory.id).label("uses"))
            .filter(QueryHistory.created_at >= window_start)
            .group_by(QueryHistory.query)
            .order_by(func.count(QueryHistory.id).desc())
            .limit(20)
            .all()
        )
        zero_result = (
            session.query(QueryHistory.query)
            .filter(QueryHistory.created_at >= window_start, QueryHistory.result_count == 0)
            .order_by(QueryHistory.created_at.desc())
            .limit(limit)
            .all()
        )
        recent = (
            session.query(FeedbackRecord)
            .order_by(FeedbackRecord.created_at.desc())
            .limit(limit)
            .all()
        )
    return {
        "total": total,
        "up": up,
        "down": down,
        "downvoted_projects": [{"project": name, "votes": votes} for name, votes in downvoted],
        "top_queries": [{"query": q, "uses": uses} for q, uses in top_queries],
        "zero_result_queries": [q for (q,) in zero_result],
        "recent": [
            {
                "query": record.query,
                "vote": record.vote,
                "project": record.project_full_name,
                "comment": record.comment,
                "source": record.source,
                "created_at": record.created_at.isoformat() if record.created_at else None,
            }
            for record in recent
        ],
    }


def _count(session, sql: str) -> int:
    return int(session.execute(text(sql)).scalar() or 0)
