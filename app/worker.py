"""Standalone 24/7 worker: runs the autonomous loop without the HTTP API.

Usage:
    .venv\\Scripts\\python.exe -m app.worker

Keep this process alive (console window, Windows Task Scheduler autostart,
NSSM service, ...) and the agent will continuously fetch, analyze, embed and
self-heal around the clock. Run alongside `uvicorn app.main:app` when the
HTTP API is also needed; both share the same database.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from app.config import Settings
from app.database import create_db_engine, init_db
from app.services.container import ServiceContainer, build_services
from app.services.scheduler import SchedulerService
from app.utils.logging import configure_logging, logger


@asynccontextmanager
async def _worker_lifecycle(settings: Settings):
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    services = build_services(settings, engine)
    scheduler = SchedulerService(services)
    scheduler.start()
    logger.info(
        "24/7 worker is live: fetch / refresh / patrol jobs are scheduled. "
        "Press Ctrl+C to stop."
    )
    try:
        yield services
    finally:
        scheduler.shutdown()
        # give in-flight to_thread jobs a moment to finish before disposing
        await asyncio.sleep(2)
        services.github.close()
        engine.dispose()
        logger.info("24/7 worker stopped cleanly.")


async def _run_forever() -> None:
    settings = Settings()
    configure_logging(settings.debug)
    logger.info(f"Starting 24/7 worker (db={settings.database_url.split('://')[0]}).")
    async with _worker_lifecycle(settings) as services:
        logger.info(
            f"Autonomous loop active: fetch every {settings.fetch_interval_minutes} min, "
            f"patrol every {settings.patrol_interval_minutes} min (patrol_enabled={settings.patrol_enabled})."
        )
        # Sleep until cancelled (Ctrl+C); the scheduler drives all work.
        await asyncio.Event().wait()


def main() -> None:
    try:
        asyncio.run(_run_forever())
    except KeyboardInterrupt:
        logger.info("Stop requested (Ctrl+C).")


if __name__ == "__main__":
    main()
