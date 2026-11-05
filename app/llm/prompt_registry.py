"""Versioned prompt registry: prompts live on disk, code holds the fallbacks.

The analysis/rerank system prompts are the "weights" of the LLM path — the
self-evolution loop (PromptEvolutionService) mutates them, gates them against
the golden set and promotes them here, exactly like model weights. When no
promoted file exists (fresh clone), the built-in defaults from app.llm.base
apply. Files are hot-reloaded on mtime change; `promote()` swaps atomically
and archives the previous version for rollback.
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

from loguru import logger

from app.llm.base import RERANK_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.utils.timeutil import utcnow

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"
CURRENT_DIR = PROMPTS_DIR / "current"
ARCHIVE_DIR = PROMPTS_DIR / "archive"
HISTORY_FILE = PROMPTS_DIR / "evolution_history.json"

_lock = threading.Lock()
_cache: dict[str, tuple[float, str]] = {}


def load_prompt(name: str, fallback: str) -> str:
    """Current promoted prompt for `name` ("analysis" | "rerank"), hot-reloaded
    on file change; falls back to the built-in default when absent/empty."""
    path = CURRENT_DIR / f"{name}.txt"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return fallback
    cached = _cache.get(name)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning(f"Prompt file {path} unreadable ({exc}); using built-in default.")
        return fallback
    if not text:
        return fallback
    _cache[name] = (mtime, text)
    return text


def analysis_prompt() -> str:
    return load_prompt("analysis", SYSTEM_PROMPT)


def rerank_prompt() -> str:
    return load_prompt("rerank", RERANK_SYSTEM_PROMPT)


def current_version() -> dict:
    try:
        return json.loads((CURRENT_DIR / "version.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": "builtin"}


def promote(
    analysis_text: str, rerank_text: str, version: str, metrics: dict
) -> Path:
    """Atomically install a new prompt version; archives the previous one."""
    with _lock:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        # CURRENT_DIR must NOT pre-exist here: shutil.move renames staging into
        # place, and an existing target directory would nest it one level down
        if (CURRENT_DIR / "version.json").exists():
            archive = ARCHIVE_DIR / current_version().get("version", "unknown").replace("/", "-")
            shutil.move(str(CURRENT_DIR), str(archive))
        staging = PROMPTS_DIR / f"staging-{version}"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "analysis.txt").write_text(analysis_text, encoding="utf-8")
        (staging / "rerank.txt").write_text(rerank_text, encoding="utf-8")
        (staging / "version.json").write_text(
            json.dumps(
                {"version": version, "promoted_at": utcnow().isoformat(), "metrics": metrics},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        shutil.move(str(staging), str(CURRENT_DIR))
        _cache.clear()
    logger.info(f"Prompt version promoted: {version}")
    return CURRENT_DIR


def append_history(report: dict) -> None:
    history: list[dict] = []
    try:
        if HISTORY_FILE.exists():
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        history = []
    history.append(report)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(history[-20:], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(HISTORY_FILE)
