"""Restore-drill tests (DFR): prove backups are restorable, safely."""
from __future__ import annotations

import sqlite3
import tarfile
from datetime import timedelta

import pytest

from app.config import Settings
from app.models.project import Project
from app.services.backup_service import BackupService
from app.utils.timeutil import utcnow


def _drill_service(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'live.db').as_posix()}",
        chroma_persist_dir=str(tmp_path / "chroma_data"),
        backup_dir=str(tmp_path / "backups"),
        backup_retention=5,
    )
    engine = __import__(
        "app.database", fromlist=["create_db_engine"]
    ).create_db_engine(settings.database_url)
    __import__("app.database", fromlist=["init_db"]).init_db(engine)
    session_factory = __import__(
        "app.database", fromlist=["create_sessionmaker"]
    ).create_sessionmaker(engine)
    return BackupService(settings), session_factory


def _seed_live(session_factory, projects: int, start_id: int = 0):
    """Seed through the ORM so every NOT NULL column gets its default."""
    with session_factory() as session:
        for i in range(projects):
            session.add(
                Project(
                    github_id=start_id + i,
                    full_name=f"p{start_id + i}",
                    url=f"https://github.com/p{start_id + i}",
                    description="drill test project",
                    stars=10,
                    forks=1,
                    topics=["drill"],
                )
            )
        session.commit()


def _live_project_count(db_path) -> int:
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    conn.close()
    return count


def test_restore_drill_proves_backup_restorable(tmp_path):
    service, session_factory = _drill_service(tmp_path)
    _seed_live(session_factory, projects=5)

    backup = service.run_backup()
    assert backup["errors"] == []

    # simulate data loss on the live database (raw DELETE: no NOT NULL issues)
    conn = sqlite3.connect(str(tmp_path / "live.db"))
    conn.execute("DELETE FROM projects")
    conn.commit()
    conn.close()

    report = service.restore_drill()

    assert report["errors"] == [], report
    assert report["restorable"] is True
    assert report["db"]["integrity"] == "ok"
    assert report["db"]["backup_rows"]["projects"] == 5
    # the drill is read-only for live files: the loss is still there,
    # proving the drill never touched the live database
    assert _live_project_count(tmp_path / "live.db") == 0


def test_drill_picks_the_newest_backup(tmp_path):
    service, session_factory = _drill_service(tmp_path)
    _seed_live(session_factory, projects=1)
    service.run_backup()
    _seed_live(session_factory, projects=3, start_id=100)  # grow, backup again
    service.run_backup()

    report = service.restore_drill()
    # the second backup holds all 4 rows (1 seeded + 3 grown); the FIRST
    # backup would have had only 1 -- drilling the newest is the point
    assert report["db"]["backup_rows"]["projects"] == 4, "must drill the newest"


def test_chroma_drill_checks_internal_sqlite(tmp_path):
    service, session_factory = _drill_service(tmp_path)
    chroma_dir = tmp_path / "chroma_data"
    chroma_dir.mkdir()
    conn = sqlite3.connect(str(chroma_dir / "chroma.sqlite3"))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.commit()
    conn.close()

    service.run_backup()
    report = service.restore_drill()
    assert report["chroma"]["chroma_sqlite3"] == "ok"


def test_drill_rejects_malicious_tarball(tmp_path):
    """A backup tarball is untrusted data: traversal members must be rejected
    without writing outside the staging dir."""
    import io

    service, _ = _drill_service(tmp_path)
    evil_tar = tmp_path / "chroma-evil.tar.gz"
    with tarfile.open(evil_tar, "w:gz") as tar:
        payload = b"pwned"
        info = tarfile.TarInfo(name="../evil.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    report = service.restore_drill(chroma_backup=evil_tar)

    assert report["errors"], "traversal attempt must be reported"
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "backups" / "evil.txt").exists()


def test_drill_reports_missing_backups(tmp_path):
    service, _ = _drill_service(tmp_path)
    report = service.restore_drill()
    assert report["errors"] == ["no backups found to drill against"]
