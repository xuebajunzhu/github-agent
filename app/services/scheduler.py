"""APScheduler setup for the 24/7 autonomous loop (fetch / refresh / patrol)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from app.services.container import ServiceContainer


class SchedulerService:
    def __init__(self, services: ServiceContainer):
        self._services = services
        self._settings = services.settings
        self._scheduler = AsyncIOScheduler(timezone="UTC")

    def start(self) -> None:
        settings = self._settings
        self._scheduler.add_job(
            self._run_fetch,
            "interval",
            minutes=settings.fetch_interval_minutes,
            id="fetch_github_trending",
            name="Fetch GitHub repositories by keywords",
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.add_job(
            self._run_refresh,
            "interval",
            minutes=settings.refresh_interval_minutes,
            id="update_existing_projects",
            name="Update stored project metadata",
            max_instances=1,
            coalesce=True,
        )
        if settings.patrol_enabled:
            self._scheduler.add_job(
                self._run_patrol,
                "interval",
                minutes=settings.patrol_interval_minutes,
                id="patrol_analysis",
                name="Patrol: backlog + retries + vector self-healing",
                max_instances=1,
                coalesce=True,
                # run once immediately: self-heal whatever happened before restart
                # (aware UTC so the scheduler's UTC timezone interprets it correctly)
                next_run_time=datetime.now(timezone.utc),
            )
        if settings.evolution_enabled:
            self._scheduler.add_job(
                self._run_evolution,
                "interval",
                minutes=settings.evolution_interval_minutes,
                id="model_evolution",
                name="Self-evolution: harvest, retrain, gate, promote/rollback",
                max_instances=1,
                coalesce=True,
            )
        self._scheduler.start()
        extras = []
        if settings.patrol_enabled:
            extras.append(f"patrol every {settings.patrol_interval_minutes} min")
        if settings.evolution_enabled:
            extras.append(f"evolution every {settings.evolution_interval_minutes} min")
        extra_note = (", " + ", ".join(extras)) if extras else ""
        logger.info(
            f"Scheduler started: fetch every {settings.fetch_interval_minutes} min, "
            f"refresh every {settings.refresh_interval_minutes} min{extra_note}."
        )

    async def _run_fetch(self) -> None:
        try:
            await asyncio.to_thread(self._services.ingest.run_scheduled_fetch)
        except Exception:  # noqa: BLE001
            logger.exception("Scheduled fetch failed")

    async def _run_refresh(self) -> None:
        try:
            await asyncio.to_thread(self._services.ingest.refresh_existing)
        except Exception:  # noqa: BLE001
            logger.exception("Scheduled refresh failed")

    async def _run_patrol(self) -> None:
        try:
            await asyncio.to_thread(self._services.patrol.run_patrol)
        except Exception:  # noqa: BLE001
            logger.exception("Patrol failed")

    async def _run_evolution(self) -> None:
        try:
            await asyncio.to_thread(self._services.evolution.run_cycle)
        except Exception:  # noqa: BLE001
            logger.exception("Model evolution cycle failed")

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    @property
    def running(self) -> bool:
        return self._scheduler.running
