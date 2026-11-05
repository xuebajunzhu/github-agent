"""Nightly backup: SQLite via the backup API + chroma dir tarball.

Keeps `backup_retention` newest files per prefix. Failures are logged (and
notified via the Notifier when configured) but never crash the scheduler.
"""
from __future__ import annotations

import sqlite3
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


class BackupService:
    def __init__(self, settings):
        self._settings = settings
        self._dir = Path(settings.backup_dir)
        self.last_report: dict | None = None

    def run_backup(self) -> dict:
        started = datetime.now(timezone.utc).isoformat()
        files: list[str] = []
        errors: list[str] = []
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._dir.mkdir(parents=True, exist_ok=True)

        db_url = self._settings.database_url
        if db_url.startswith("sqlite:///"):
            db_path = Path(db_url.removeprefix("sqlite:///"))
            if db_path.exists():
                target = self._dir / f"db-{stamp}.db"
                try:
                    source = sqlite3.connect(str(db_path))
                    destination = sqlite3.connect(str(target))
                    with destination:
                        source.backup(destination)
                    destination.close()
                    source.close()
                    files.append(str(target))
                except sqlite3.Error as exc:
                    errors.append(f"sqlite backup: {exc}")
                    logger.error(f"SQLite backup failed: {exc}")
        else:
            errors.append(f"unsupported database_url for backup: {db_url.split('://')[0]}")

        chroma_dir = Path(self._settings.chroma_persist_dir)
        if chroma_dir.exists():
            target = self._dir / f"chroma-{stamp}.tar.gz"
            try:
                with tarfile.open(target, "w:gz") as tar:
                    tar.add(chroma_dir, arcname=chroma_dir.name)
                files.append(str(target))
            except (OSError, tarfile.TarError) as exc:
                errors.append(f"chroma backup: {exc}")
                logger.error(f"Chroma backup failed: {exc}")

        self._rotate("db-", self._settings.backup_retention)
        self._rotate("chroma-", self._settings.backup_retention)

        self.last_report = {
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "errors": errors,
        }
        if errors:
            logger.warning(f"Backup finished with errors: {self.last_report}")
        else:
            logger.info(f"Backup ok: {files}")
        return self.last_report

    def _rotate(self, prefix: str, retention: int) -> tuple[int, int]:
        matching = sorted(self._dir.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        for stale in matching[retention:]:
            try:
                stale.unlink()
                removed += 1
            except OSError as exc:
                logger.warning(f"Could not remove stale backup {stale.name}: {exc}")
        return len(matching), removed
