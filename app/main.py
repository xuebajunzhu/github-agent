"""FastAPI application entry point: app factory + lifespan wiring."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware

from app.api.middleware import RateLimitMiddleware
from app.api.routes import all_routers
from app.config import Settings
from app.database import create_db_engine, init_db
from app.services.container import build_services
from app.services.scheduler import SchedulerService
from app.utils.logging import configure_logging, logger

VERSION = "0.1.0"
LANDING_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    services = build_services(settings, engine)
    app.state.services = services
    scheduler: SchedulerService | None = None
    if settings.scheduler_enabled:
        scheduler = SchedulerService(services)
        scheduler.start()
    app.state.scheduler = scheduler
    logger.info(
        f"{settings.app_name} v{VERSION} started. "
        f"llm='{services.analysis.provider_name}', "
        f"embedder='{services.embedding.provider_name}', "
        f"vector_store='{type(services.vectors).__name__}'."
    )
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown()
        services.github.close()
        engine.dispose()
        logger.info("Application shutdown complete.")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.debug, settings.log_file)

    app = FastAPI(
        title="GitHub Agent",
        description="Retrieves GitHub open-source projects on a schedule, analyzes them "
        "with pluggable LLMs, embeds them for semantic scenario recommendations.",
        version=VERSION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.scheduler = None

    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, same_site="lax")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if settings.rate_limit_per_minute > 0:
        app.add_middleware(RateLimitMiddleware, max_per_minute=settings.rate_limit_per_minute)

    for router in all_routers:
        app.include_router(router)

    @app.get("/", include_in_schema=False, response_model=None)
    def root():
        if LANDING_PAGE.exists():
            return FileResponse(LANDING_PAGE)
        return _info_json(app)

    @app.get("/info", include_in_schema=False)
    def info() -> dict:
        return _info_json(app)

    return app


def _info_json(app: FastAPI) -> dict:
    settings: Settings = app.state.settings
    return {
        "name": settings.app_name,
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
        "voice_page": "/api/voice/page",
        "endpoints": ["/api/recommend", "/api/projects", "/api/feedback"],
    }


app = create_app()
