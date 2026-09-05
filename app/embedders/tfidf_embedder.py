"""Pure-Python hashing TF-IDF embedder: the offline, dependency-free fallback.

Tokens are hashed into a fixed number of buckets (stable across runs via
SHA-256, unlike Python's salted hash()) with sublinear term frequency and L2
normalization. Deterministic and instant, but quality is below real embedding
models -- enable it explicitly via EMBEDDER_PROVIDER=tfidf when no better
option is available.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from app.embedders.base import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingTfidfEmbedder(Embedder):
    name = "tfidf"

    def __init__(self, dimension: int = 384):
        self.dimension = max(16, dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts = Counter(_TOKEN_RE.findall(text.lower()))
        for token, count in counts.items():
            bucket = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
            index = bucket % self.dimension
            vector[index] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector
