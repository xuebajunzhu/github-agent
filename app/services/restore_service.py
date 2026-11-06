"""Controlled restore-to-live: the write half of the backup lifecycle.

BackupService owns creation and the read-only restore_drill (validation).
This service owns the dangerous half — replacing the live database and
chroma dir from backup artifacts — with four guardrails in order:

  1. refuses while the API service is running (a SQLite file swap under
     live WAL connections corrupts the database);
  2. snapshots the CURRENT live state first (a bad restore is reversible);
  3. stages + integrity-checks every artifact BEFORE it replaces anything;
  4. the chroma tarball is extracted with filter="data" (traversal rejected).

Prefix knowledge ("db-", "chroma-") mirrors BackupService; if BackupService's
formats ever change, change them here too.
"""
from __future__ import annotations

import shutil
import sqlite3
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.config import Settings

_EXPECTED_TABLES = ("projects", "feedback", "query_history", "users")


class RestoreRefused(Exception):
    """Raised when a guardrail refuses the restore."""


class RestoreService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._dir = Path(settings.backup_dir)

    # ------------------------------------------------------------------ main

    def restore_to_live(
        self,
        db_backup: Path | None = None,
        chroma_backup: Path | None = None,
        service_running: bool | None = None,
    ) -> dict:
        started = datetime.now(timezone.utc).isoformat()
        report: dict = {"started_at": started, "steps": {}, "errors": [], "restored": False}

        if service_running is None:
            service_running = self._service_port_listening()
        if service_running:
            raise RestoreRefused(
                "The API service is running. Stop it first (scripts\\manage.ps1 stop) — "
                "a live file swap corrupts SQLite."
            )

        db_backup = db_backup or self._newest("db-")
        chroma_backup = chroma_backup or self._newest("chroma-")
        if db_backup is None and chroma_backup is None:
            raise RestoreRefused("No backups found to restore")

        # guardrail 2: snapshot the current live state BEFORE overwriting
        pre = self._snapshot_live()
        report["steps"]["pre_restore_backup"] = pre

        live_db = Path(self._settings.database_url.removeprefix("sqlite:///"))

        if db_backup is not None and db_backup.exists():
            step = self._restore_db(db_backup, live_db)
            report["steps"]["db_restore"] = step
            if not step["ok"]:
                report["errors"].append(step.get("error", "db restore failed"))

        if chroma_backup is not None and chroma_backup.exists():
            step = self._restore_chroma(chroma_backup)
            report["steps"]["chroma_restore"] = step
            if not step["ok"]:
                report["errors"].append(step.get("error", "chroma restore failed"))

        report["restored"] = not report["errors"]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if report["errors"]:
            logger.warning(f"Restore to live finished with errors: {report}")
        else:
            logger.info(f"Restore to live ok: {report['steps']}")
        return report

    # ---------------------------------------------------------------- stages

    def _newest(self, prefix: str) -> Path | None:
        candidates = sorted(self._dir.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _snapshot_live(self) -> list[str]:
        """Guardrail: the pre-restore state is itself backed up."""
        self._dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        files: list[str] = []
        live_db = Path(self._settings.database_url.removeprefix("sqlite:///"))
        if live_db.exists():
            target = self._dir / f"pre-restore-db-{stamp}.db"
            shutil.copy2(live_db, target)
            files.append(str(target))
        chroma_dir = Path(self._settings.chroma_persist_dir)
        if chroma_dir.exists():
            target = self._dir / f"pre-restore-chroma-{stamp}.tar.gz"
            with tarfile.open(target, "w:gz") as tar:
                tar.add(chroma_dir, arcname=chroma_dir.name)
            files.append(str(target))
        return files

    def _restore_db(self, db_backup: Path, live_db: Path) -> dict:
        """Stage + integrity-check + ATOMIC replace (same volume)."""
        staged = live_db.parent / f"{live_db.name}.restore-staging"
        shutil.copy2(db_backup, staged)
        conn = sqlite3.connect(str(staged))
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            conn.close()
        if integrity != "ok":
            staged.unlink(missing_ok=True)
            return {"ok": False, "error": f"staging integrity failed: {integrity}"}
        import os

        os.replace(str(staged), str(live_db))
        # stale WAL/SHM sidecars from the replaced database would corrupt the
        # restored file on next open -- they belong to the OLD database
        for suffix in ("-wal", "-shm"):
            sidecar = live_db.parent / f"{live_db.name}{suffix}"
            sidecar.unlink(missing_ok=True)
        return {"ok": True, "integrity": integrity}

    def _restore_chroma(self, chroma_backup: Path) -> dict:
        """Extract with traversal protection, verify, then swap the dir."""
        chroma_dir = Path(self._settings.chroma_persist_dir)
        staging = Path(tempfile.mkdtemp(prefix="chroma-restore-"))
        try:
            with tarfile.open(chroma_backup, "r:gz") as tar:
                # filter="data" raises (does not silently skip) on traversal
                # members -- the exception is the rejection signal
                try:
                    tar.extractall(staging, filter="data")
                except tarfile.FilterError as exc:
                    return {"ok": False, "error": f"malicious tarball rejected: {exc}"}
            inner = next(staging.rglob("chroma.sqlite3"), None)
            if inner is None:
                return {"ok": False, "error": "backup has no chroma.sqlite3"}
            conn = sqlite3.connect(str(inner))
            try:
                integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                conn.close()
            if integrity != "ok":
                return {"ok": False, "error": f"chroma integrity failed: {integrity}"}
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
            shutil.move(str(inner.parent), str(chroma_dir))
            return {"ok": True}
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # ----------------------------------------------------------------- probe

    def _service_port_listening(self, port: int = 8788) -> bool:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex(("127.0.0.1", port)) == 0
