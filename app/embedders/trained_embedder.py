"""Embedder backed by the trained RepoAnalyzerModel's multilingual encoder.

The classification fine-tune keeps the encoder a usable sentence embedder for
our own index (projects and queries are embedded by the same model, so only
self-consistency matters). Reusing it means one local model powers both
analysis and semantic search -- fully offline, Chinese queries included.

Hot reload: when the self-evolution loop swaps the deployed model, the stamp
check reloads here too; the evolution service rebuilds stored vectors in the
same swap window so project/query spaces stay consistent.
"""
from __future__ import annotations

import torch

from app.embedders.base import Embedder
from app.ml.model_registry import MODEL_SWAP_LOCK, stamp
from app.ml.repo_analyzer import load_repo_analyzer


class TrainedEmbedder(Embedder):
    name = "trained"

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._stamp = None
        self._model, self._tokenizer, config, self._device = load_repo_analyzer(model_path)
        self.dimension = self._model.encoder.config.hidden_size
        self._max_len = config["max_len"]

    def _reload_if_swapped(self) -> None:
        current = stamp(self._model_path)
        if current != self._stamp:
            self._model, self._tokenizer, config, self._device = load_repo_analyzer(self._model_path)
            self.dimension = self._model.encoder.config.hidden_size
            self._max_len = config["max_len"]
            self._stamp = current

    @torch.no_grad()
    def embed(self, texts: list[str]) -> list[list[float]]:
        # serialize against evolution swaps so spaces never mix mid-rebuild
        with MODEL_SWAP_LOCK:
            self._reload_if_swapped()
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_len,
                return_tensors="pt",
            ).to(self._device)
            outputs = self._model.encoder(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
            )
            mask = encoded["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return [[float(value) for value in vector] for vector in pooled.cpu()]
