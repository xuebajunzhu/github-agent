"""BGE-M3 embedder (BAAI/bge-m3): 1024-dim multilingual dense embeddings.

Dense pooling follows the official FlagEmbedding recipe -- CLS token + L2
normalize -- so cosine similarity matches the model's training objective.
Implemented directly on transformers (already an ml dependency) instead of
adding FlagEmbedding; the sparse/ColBERT outputs are unused by our cosine
vector stores. 100+ languages, with Chinese as a first-class citizen (the
project's primary query language).
"""
from __future__ import annotations

from app.config import Settings
from app.embedders.base import Embedder


class BgeM3Embedder(Embedder):
    name = "bge-m3"

    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._model_name = settings.bge_m3_model_name
        self._max_len = settings.bge_m3_max_len
        # cached-first: never let a hub HEAD request stall startup when the
        # weights are already local; on a cache miss download once (honors
        # HF_ENDPOINT, e.g. the hf-mirror.com mirror this project uses)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, local_files_only=True)
            self._model = AutoModel.from_pretrained(self._model_name, local_files_only=True)
        except Exception:  # noqa: BLE001  # first download or partial snapshot
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()
        self._torch = torch
        self.dimension = self._model.config.hidden_size  # 1024

    @property
    def space_id(self) -> str:
        return f"{self.name}:{self._model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._torch.no_grad():
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self._max_len,
                return_tensors="pt",
            )
            outputs = self._model(**encoded)
            cls = outputs.last_hidden_state[:, 0]
            cls = self._torch.nn.functional.normalize(cls, p=2, dim=1)
            return [[float(value) for value in row] for row in cls.cpu()]
