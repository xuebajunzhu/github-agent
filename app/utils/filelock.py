"""Cross-process advisory file lock for the 24/7 job loops.

SQLite (WAL) serializes the data itself, but the fetch / refresh / patrol /
evolution jobs must not run concurrently across the API process and the worker
process: they would double-analyze the same pending queue and race the model
swap directory. An OS-level lock on a lock file makes every job exclusive
across processes *and* threads (each acquire opens its own file descriptor).
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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


@contextmanager
def job_lock(name: str = "pipeline") -> Iterator[bool]:
    """Serialize a scheduled/manual job.

    Yields True when the lock was acquired (run the job), False when another
    pipeline job is already running (skip; the scheduler will come back).
    """
    lock = ProcessFileLock(LOCK_DIR / f"{name}.lock")
    acquired = lock.acquire()
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()
