"""Restore-to-live tests: guardrails, atomic replace, sidecar cleanup."""
from __future__ import annotations

import sqlite3
import tarfile
from datetime import timedelta
from pathlib import Path

import pytest

from app.config import Settings
from app.services.restore_service import RestoreRefused, RestoreService
from app.utils.timeutil import utcnow


def _service(tmp_path, port_listening: bool | None = None):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'live.db').as_posix()}",
        chroma_persist_dir=str(tmp_path / "chroma_data"),
        backup_dir=str(tmp_path / "backups"),
    )
    engine = __import__(
        "app.database", fromlist=["create_db_engine"]
    ).create_db_engine(settings.database_url)
    __import__("app.database", fromlist=["init_db"]).init_db(engine)
    session_factory = __import__(
        "app.database", fromlist=["create_sessionmaker"]
    ).create_sessionmaker(engine)
    service = RestoreService(settings)
    if port_listening is not None:
        service._service_port_listening = lambda port=8788: port_listening
    return service, session_factory, engine


def _dispose(engine) -> None:
    """Release pooled sqlite handles: Windows cannot os.replace a file with
    an open handle, and the restore's atomic swap hits exactly that."""
    engine.dispose()


def _seed(session_factory, projects: int):
    from app.models.project import Project

    with session_factory() as session:
        for i in range(projects):
            session.add(
                Project(
                    github_id=i, full_name=f"p{i}", url=f"https://github.com/p{i}",
                    description="d", stars=10, forks=1, topics=["t"],
                )
            )
        session.commit()


def _live_projects(db_path) -> int:
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    conn.close()
    return count


def _make_chroma_tarball(tmp_path, projects_dir_name="chroma_data") -> Path:
    chroma_dir = tmp_path / projects_dir_name
    chroma_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(chroma_dir / "chroma.sqlite3"))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.commit()
    conn.close()
    tarball = tmp_path / "chroma-restore-src.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(chroma_dir, arcname=projects_dir_name)
    return tarball


def test_restore_replaces_live_db_from_backup(tmp_path):
    service, session_factory, engine = _service(tmp_path)
    _seed(session_factory, 5)
    service._dir.mkdir(parents=True, exist_ok=True)
    db_backup = tmp_path / "backups" / "db-x.db"
    conn = sqlite3.connect(str(db_backup))
    conn.execute("CREATE TABLE projects (id INTEGER PRIMARY KEY, full_name TEXT)")
    for i in range(8):
        conn.execute("INSERT INTO projects (id, full_name) VALUES (?, ?)", (i, f"row{i}"))
    conn.commit()
    conn.close()

    # live db also needs the projects table for the post-restore count check
    conn = sqlite3.connect(str(tmp_path / "live.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, full_name TEXT)")
    conn.commit()
    conn.close()
    _dispose(engine)  # release pooled handles so the swap can happen

    report = service.restore_to_live(db_backup=db_backup, service_running=False)

    assert report["restored"] is True
    assert report["errors"] == []
    assert report["steps"]["db_restore"]["integrity"] == "ok"
    # restored content is live
    assert _live_projects(tmp_path / "live.db") == 8
    # stale WAL/SHM sidecars were removed with the old file
    assert not (tmp_path / "live.db-wal").exists()


def test_restore_refuses_while_service_running(tmp_path):
    service, session_factory, engine = _service(tmp_path)
    _seed(session_factory, 3)
    service._dir.mkdir(parents=True, exist_ok=True)
    db_backup = tmp_path / "backups" / "db-x.db"
    conn = sqlite3.connect(str(db_backup))
    conn.execute("CREATE TABLE projects (id INTEGER PRIMARY KEY, full_name TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(RestoreRefused):
        # simulate the API listening on 8788
        import socket

        listener = socket.socket()
        listener.bind(("127.0.0.1", 8788))
        listener.listen(1)
        try:
            service.restore_to_live(db_backup=db_backup, service_running=None)
        finally:
            listener.close()


def test_restore_snapshots_current_state_first(tmp_path):
    service, session_factory, engine = _service(tmp_path)
    _seed(session_factory, 3)
    service._dir.mkdir(parents=True, exist_ok=True)
    db_backup = tmp_path / "backups" / "db-x.db"
    conn = sqlite3.connect(str(db_backup))
    conn.execute("CREATE TABLE projects (id INTEGER PRIMARY KEY, full_name TEXT)")
    for i in range(8):
        conn.execute("INSERT INTO projects (id, full_name) VALUES (?, ?)", (i, f"row{i}"))
    conn.commit()
    conn.close()
    _dispose(engine)

    service.restore_to_live(db_backup=db_backup, service_running=False)

    pre_restore = list(service._dir.glob("pre-restore-db-*.db"))
    assert pre_restore, "the pre-restore snapshot must exist"
    conn = sqlite3.connect(str(pre_restore[0]))
    count = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    conn.close()
    assert count == 3, "the snapshot preserves the PRE-restore live state"


def test_restore_chroma_replaces_dir(tmp_path):
    service, session_factory, engine = _service(tmp_path)
    _seed(session_factory, 2)
    tarball = _make_chroma_tarball(tmp_path)

    report = service.restore_to_live(chroma_backup=tarball, service_running=False)

    assert report["restored"] is True
    assert report["steps"]["chroma_restore"]["ok"] is True
    restored_sqlite = tmp_path / "chroma_data" / "chroma.sqlite3"
    assert restored_sqlite.exists()


def test_restore_chroma_rejects_traversal_tarball(tmp_path):
    service, session_factory, engine = _service(tmp_path)
    _seed(session_factory, 2)
    evil_tar = tmp_path / "chroma-evil.tar.gz"
    import io

    with tarfile.open(evil_tar, "w:gz") as tar:
        info = tarfile.TarInfo(name="../evil.txt")
        payload = b"pwned"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    report = service.restore_to_live(chroma_backup=evil_tar, service_running=False)

    assert report["restored"] is False
    assert report["errors"]
    assert not (tmp_path / "evil.txt").exists()
