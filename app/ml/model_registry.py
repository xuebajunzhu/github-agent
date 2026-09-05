"""Model registry helpers: version stamping, atomic-ish swap, hot reload lock.

The live model lives in settings.trained_model_path. The self-evolution loop
trains a challenger into a staging directory and, after gates pass, swaps it
in. Consumers (analyzer / embedder) detect the swap via a version stamp and
hot-reload on their next use; MODEL_SWAP_LOCK keeps the swap + vector rebuild
serialized against in-flight inference.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

MODEL_SWAP_LOCK = threading.Lock()

VERSION_FILE = "version.json"
STAMP_FILE = "analyzer_config.json"


def current_version(models_dir: str | Path) -> str | None:
    version_file = Path(models_dir) / VERSION_FILE
    if version_file.exists():
        try:
            return json.loads(version_file.read_text(encoding="utf-8")).get("version")
        except (ValueError, OSError):
            return None
    return None


def stamp(models_dir: str | Path) -> tuple | None:
    """Cheap change-detection stamp for the deployed model."""
    stamp_file = Path(models_dir) / STAMP_FILE
    if not stamp_file.exists():
        return None
    stat = stamp_file.stat()
    version = current_version(models_dir)
    return (stat.st_mtime_ns, stat.st_size, version)


def read_version_info(models_dir: str | Path) -> dict:
    version_file = Path(models_dir) / VERSION_FILE
    if not version_file.exists():
        return {}
    try:
        return json.loads(version_file.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
