"""Nightly backup: SQLite via the backup API + chroma dir tarball.

Keeps `backup_retention` newest files per prefix. Failures are logged (and
notified via the Notifier when configured) but never crash the scheduler.

restore_drill() closes the DFR loop: a backup that has never been restored is
a hypothesis, not a backup. The drill restores the newest artifacts into a
staging area (never touching the live files), runs SQLite integrity checks,
and compares row counts against the live database.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

_EXPECTED_TABLES = ("projects", "feedback", "query_history", "users")


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

    # ------------------------------------------------------------- restore

    def _newest(self, prefix: str) -> Path | None:
        candidates = sorted(self._dir.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    @staticmethod
    def _sqlite_counts(db_path: Path, tables: tuple[str, ...]) -> dict[str, int]:
        counts: dict[str, int] = {}
        conn = sqlite3.connect(str(db_path))
        try:
            existing = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            for table in tables:
                counts[table] = (
                    conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    if table in existing
                    else -1  # table absent in this backup
                )
        finally:
            conn.close()
        return counts

    def restore_drill(
        self,
        db_backup: Path | None = None,
        chroma_backup: Path | None = None,
        keep_staging: bool = False,
    ) -> dict:
        """Prove the newest backups are actually restorable, without touching
        the live files: copy into a staging dir, run SQLite integrity checks,
        compare row counts against the live database, then clean up.

        The chroma tarball is extracted with filter="data" (path traversal,
        absolute paths and device files are rejected by the stdlib).
        """
        started = datetime.now(timezone.utc).isoformat()
        report: dict = {"started_at": started, "errors": [], "db": {}, "chroma": {}}

        db_backup = db_backup or self._newest("db-")
        chroma_backup = chroma_backup or self._newest("chroma-")
        if db_backup is None and chroma_backup is None:
            report["errors"].append("no backups found to drill against")
            return report

        staging = Path(tempfile.mkdtemp(prefix="restore-drill-"))
        try:
            if db_backup is not None and db_backup.exists():
                restored = staging / f"restored-{db_backup.name}"
                shutil.copy2(db_backup, restored)
                conn = sqlite3.connect(str(restored))
                try:
                    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                finally:
                    conn.close()
                backup_counts = self._sqlite_counts(
                    restored, _EXPECTED_TABLES + ("research_cache",)
                )
                live_path = Path(self._settings.database_url.removeprefix("sqlite:///"))
                live_counts = (
                    self._sqlite_counts(live_path, _EXPECTED_TABLES) if live_path.exists() else {}
                )
                restorable = integrity == "ok" and backup_counts.get("projects", -1) >= 0
                report["db"] = {
                    "backup": str(db_backup),
                    "staging": str(restored) if keep_staging else "(cleaned)",
                    "integrity": integrity,
                    "backup_rows": backup_counts,
                    "live_rows_today": live_counts.get("projects"),
                    "restorable": restorable,
                }
                if not restorable:
                    report["errors"].append("database restore drill failed integrity check")

            if chroma_backup is not None and chroma_backup.exists():
                chroma_staging = staging / "chroma"
                chroma_staging.mkdir(parents=True, exist_ok=True)
                with tarfile.open(chroma_backup, "r:gz") as tar:
                    # filter="data": rejects absolute paths, '..' traversal and
                    # device nodes -- the archive is data, not instructions
                    tar.extractall(chroma_staging, filter="data")
                inner = next(chroma_staging.rglob("chroma.sqlite3"), None)
                chroma_integrity = "missing"
                if inner is not None and inner.exists():
                    conn = sqlite3.connect(str(inner))
                    try:
                        chroma_integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                    finally:
                        conn.close()
                report["chroma"] = {
                    "backup": str(chroma_backup),
                    "chroma_sqlite3": chroma_integrity,
                    "extracted_files": sum(1 for _ in chroma_staging.rglob("*")),
                }
                if chroma_integrity != "ok":
                    report["errors"].append("chroma restore drill failed integrity check")
        except (OSError, sqlite3.Error, tarfile.TarError) as exc:
            report["errors"].append(f"restore drill: {exc}")
            logger.error(f"Restore drill failed: {exc}")
        finally:
            if not keep_staging:
                shutil.rmtree(staging, ignore_errors=True)

        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["restorable"] = not report["errors"]
        if report["errors"]:
            logger.warning(f"Restore drill finished with errors: {report}")
        else:
            logger.info(f"Restore drill ok: {json.dumps(report['db'], default=str)[:200]}")
        return report
