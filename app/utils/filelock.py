"""Cross-process job locks: file-based default, Redis when REDIS_URL is set.

File locks (msvcrt/flock) are the zero-dependency default and work across
processes AND threads on one box. When REDIS_URL is configured, a Redis
SET NX PX lock takes over instead -- that is what makes the job mutex work
across multiple machines, and it survives neither-needed-file-systems
semantics on containerized deployments.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
LOCK_DIR = ROOT / "logs"


class ProcessFileLock:
    """Non-blocking exclusive lock on a lock file (msvcrt on Windows, flock elsewhere)."""

    def __init__(self, path: Path):
        self._path = path
        self._fd: int | None = None

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR)
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(self._fd)
            self._fd = None


class RedisLock:
    """SET NX PX based mutual exclusion (multi-box safe); auto-expires so a
    crashed holder cannot deadlock the pipeline forever. The TTL must exceed
    the longest job (model evolution trains for up to
    EVOLUTION_TRAIN_TIMEOUT_SECONDS = 2h), otherwise a live holder expires
    mid-job and a second job starts on top of it."""

    def __init__(self, redis_url: str, name: str, ttl_seconds: int = 4 * 3600):
        import redis

        self._client = redis.Redis.from_url(redis_url)
        self._key = f"github-agent:lock:{name}"
        self._ttl = ttl_seconds
        self._token = f"{os.getpid()}-{id(self)}"

    def acquire(self) -> bool:
        try:
            return bool(self._client.set(self._key, self._token, nx=True, px=self._ttl * 1000))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Redis lock unavailable ({exc}); falling back to file lock.")
            self._client = None
            return ProcessFileLock(LOCK_DIR / "redis-fallback.lock").acquire()

    def release(self) -> None:
        if getattr(self, "_client", None) is None:
            return
        try:
            self._client.delete(self._key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Redis lock release failed: {exc}")


def _redis_url() -> str | None:
    try:
        from app.config import get_settings

        return get_settings().redis_url
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def job_lock(name: str = "pipeline") -> Iterator[bool]:
    """Serialize a scheduled/manual job.

    Yields True when the lock was acquired (run the job), False when another
    pipeline job is already running (skip; the scheduler will come back).
    Uses Redis when REDIS_URL is configured, else an OS file lock.
    """
    url = _redis_url()
    lock: ProcessFileLock | RedisLock
    if url:
        lock = RedisLock(url, name)
    else:
        lock = ProcessFileLock(LOCK_DIR / f"{name}.lock")
    acquired = lock.acquire()
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()
