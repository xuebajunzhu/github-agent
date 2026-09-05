"""Loguru-based logging setup."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_logging(debug: bool = False, log_file: str | None = None) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "INFO",
        format=_FORMAT,
        backtrace=False,
        diagnose=False,
    )
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG" if debug else "INFO",
            format=_FORMAT,
            rotation="10 MB",
            retention=5,
            encoding="utf-8",
            backtrace=False,
            diagnose=False,
        )


__all__ = ["configure_logging", "logger"]
