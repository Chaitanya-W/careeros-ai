"""Local deterministic embedder — no paid API, no model download, CPU-only.

Pipeline stage: CHUNKING -> EMBEDDING.

This embedder uses **word + character n-gram hashing** into a fixed-size
vector with L2 normalization. It is:
- deterministic (stable across processes — uses ``hashlib``, not Python's
  salted ``hash()``),
- dependency-free (pure Python),
- CPU-only (no GPU, no model weights, no network),
- fast enough for capstone-scale document collections.

This is NOT a semantic embedding model. It is a lightweight local
embedding that captures lexical overlap (word and sub-word n-grams),
sufficient for a free-tier, offline-capable RAG demo. The retrieval
threshold (see ``retrieval.py``) accounts for its lexical nature.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Final

DEFAULT_DIMS: Final[int] = 256
_WORD_RE = re.compile(r"[a-z0-9]+")


class LocalEmbedder:
    """Deterministic local embedding via n-gram hashing + L2 normalization."""

    def __init__(self, dims: int = DEFAULT_DIMS) -> None:
        if dims <= 0:
            raise ValueError("dims must be positive")
        self._dims = dims

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        """Embed a single text into a fixed-size L2-normalized vector."""
        if not text or not text.strip():
            return [0.0] * self._dims
        vec = [0.0] * self._dims
        tokens = _tokenize(text)
        features = _features(tokens)
        for feat in features:
            idx = self._hash_index(feat)
            vec[idx] += 1.0
        _l2_normalize(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def _hash_index(self, feature: str) -> int:
        """Stable hash -> vector index (deterministic across processes)."""
        h = hashlib.sha256(feature.encode("utf-8", errors="replace"))
        # Take the first 8 bytes as an unsigned int, mod dims.
        digest = h.digest()
        n = int.from_bytes(digest[:8], "big", signed=False)
        return n % self._dims


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two L2-normalized vectors = dot product.

    Falls back to a safe dot-product-with-normalization if inputs aren't
    pre-normalized.
    """
    if not a or not b:
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t]


def _features(tokens: list[str]) -> list[str]:
    """Word unigrams + bigrams + character 3-grams of each word.

    The character n-grams give sub-word overlap (helps with morphological
    variants), while word n-grams give phrase overlap.
    """
    feats: list[str] = []
    # Word unigrams.
    feats.extend(f"w:{t}" for t in tokens)
    # Word bigrams.
    feats.extend(f"b:{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    # Character 3-grams of each token (skip very short tokens).
    for t in tokens:
        if len(t) < 4:
            continue
        padded = f"#{t}#"
        for i in range(len(padded) - 2):
            feats.append(f"c:{padded[i : i + 3]}")
    return feats


def _l2_normalize(vec: list[float]) -> None:
    """L2-normalize ``vec`` in place. A zero vector stays zero."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return
    for i in range(len(vec)):
        vec[i] /= norm
