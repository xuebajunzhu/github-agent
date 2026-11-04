"""BGE reranker (BAAI/bge-reranker-v2-m3): local cross-encoder relevance head.

A 568M cross-encoder that scores (query, passage) pairs in one forward pass --
the official companion to BGE-M3 embeddings and the quality counterpart to the
Laya decision model (whose probability separation measured <0.01 on this
domain). Latency on CPU is ~150-400ms per batch of candidates, zero API cost,
fully offline. Selected via RERANK_PROVIDER=bge; the LLM reranker remains the
"generates human-readable reasons" option.
"""
from __future__ import annotations

import threading

from loguru import logger

from app.config import Settings


class BgeReranker:
    name = "bge-reranker"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._tokenizer = None
        self._failed = False
        self._load_lock = threading.Lock()

    def available(self) -> bool:
        if self._failed:
            return False
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def preload_async(self) -> None:
        def _warm() -> None:
            try:
                self._ensure_model()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"BGE reranker preload failed: {exc}")

        threading.Thread(target=_warm, name="bge-reranker-preload", daemon=True).start()

    def _ensure_model(self):
        with self._load_lock:
            if self._model is None and not self._failed:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                name = self._settings.reranker_model_name
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        name, local_files_only=True
                    )
                except Exception:  # noqa: BLE001  # first download or partial snapshot
                    self._tokenizer = AutoTokenizer.from_pretrained(name)
                    self._model = AutoModelForSequenceClassification.from_pretrained(name)
                self._model.eval()
                self._torch = torch
                logger.info(f"BGE reranker loaded: {name}")
            return self._model

    def rank_candidates(
        self, query: str, candidates: list[dict]
    ) -> tuple[list[int], str] | None:
        """Score every (query, candidate) pair with the cross-encoder and rank
        by relevance. No natural-language reason: decision head, not generator."""
        if not candidates or not self.available():
            return None
        model = self._ensure_model()
        if model is None:
            return None
        try:
            pairs = [
                [
                    query,
                    f"{c.get('full_name', '')}: {(c.get('summary') or '')[:400]} "
                    f"[tags: {', '.join(c.get('tags') or [])}]",
                ]
                for c in candidates
            ]
            with self._torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                logits = model(**inputs).logits.view(-1).float()
            # logit is monotonic with relevance; stable sort keeps ties in
            # the semantic-search order
            ranking = sorted(range(len(candidates)), key=lambda i: -float(logits[i]))
            logger.info(f"Reranked {len(candidates)} candidates with '{self.name}' (local).")
            return ranking, ""
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"BGE rerank failed: {exc}")
            return None
