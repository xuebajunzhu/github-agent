"""Laya reranker: a 421M multilingual "System 1" decision model (HF
convaiinnovations/laya) used as a fast, zero-cost local reranker.

Honest positioning, from measurements on this project's domain (see
scripts/bench notes): Laya answers yes/no relevance in ~150-220ms per
candidate on CPU, but its probability separation is weak (correct candidate
led by <0.01 over distractors, and the score head ranked a distractor first).
It is therefore offered as an experimental `RERANK_PROVIDER=laya` option --
zero API cost, fully offline -- while `RERANK_PROVIDER=llm` (DeepSeek) remains
the default reranker and produces the human-readable reasons.
"""
from __future__ import annotations

import threading

from loguru import logger

from app.config import Settings


class LayaReranker:
    name = "laya"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._router = None
        self._load_lock = threading.Lock()
        self._failed = False

    def available(self) -> bool:
        if self._failed:
            return False
        try:
            import laya  # noqa: F401
        except ImportError:
            return False
        return True

    def preload_async(self) -> None:
        """Warm the router in the background so the first recommend request
        does not pay the checkpoint load (~30s+)."""
        if not self.available():

            def _warm() -> None:
                try:
                    self._ensure_router()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Laya preload failed: {exc}")

            threading.Thread(target=_warm, name="laya-preload", daemon=True).start()

    def _ensure_router(self):
        with self._load_lock:
            if self._router is None and not self._failed:
                try:
                    from laya import Router

                    self._router = Router()
                except Exception as exc:  # noqa: BLE001
                    self._failed = True
                    logger.warning(f"Laya router init failed (reranker disabled): {exc}")
            return self._router

    def _relevance_probability(self, state: str) -> float | None:
        router = self._ensure_router()
        if router is None:
            return None
        result = router.predict(
            state,
            {
                "fit": {
                    "type": "choice",
                    "instructions": "该候选项目与用户描述的场景匹配吗?",
                    "criteria": {"yes": "匹配", "no": "不匹配"},
                }
            },
        )
        probabilities = result["answers"]["fit"].get("probabilities") or {}
        value = probabilities.get("yes")
        return float(value) if value is not None else None

    def rank_candidates(self, query: str, candidates: list[dict]) -> tuple[list[int], str] | None:
        """Score every candidate with Laya's yes-probability and rank by it.

        Compatible with the AnalysisService rerank protocol: returns
        (ranking indices, reason). No natural-language reason -- Laya is a
        decision head, not a generator."""
        if not self.available() or not candidates:
            return None
        scored: list[tuple[float, int]] = []
        for index, candidate in enumerate(candidates):
            state = (
                f"用户场景: {query}\n"
                f"候选项目: {candidate.get('full_name', '')}\n"
                f"描述: {candidate.get('summary', '')}\n"
                f"标签: {', '.join(candidate.get('tags') or [])}"
            )
            try:
                probability = self._relevance_probability(state)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Laya scoring failed: {exc}")
                return None
            if probability is None:
                return None
            scored.append((probability, index))
        # stable: equal probabilities keep the semantic-search order
        ranking = [index for _, index in sorted(scored, key=lambda pair: -pair[0])]
        logger.info(f"Reranked {len(candidates)} candidates with 'laya' (local).")
        return ranking, ""
