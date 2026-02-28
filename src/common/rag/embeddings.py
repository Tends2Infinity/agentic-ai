"""
embeddings.py

Deterministic "embedding-like" vectors using feature hashing.

Why this approach?
- Works offline (no API keys)
- Deterministic across machines
- Fast enough for small corpora
- CI-friendly
- Lets us implement hybrid retrieval now and swap to true embeddings later

How it works:
- Tokenize text
- Hash each token into [0..dim-1]
- Accumulate counts (or weighted counts)
- L2 normalize to unit vector
"""

from __future__ import annotations

import hashlib
import math

from src.common.rag.text import tokenize


def _hash_token(token: str, dim: int) -> int:
    """
    Stable hash (not Python's built-in hash, which can vary by process).
    """
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % dim


def embed_text(text: str, *, dim: int = 512) -> list[float]:
    """
    Convert text into a dense vector of length dim.
    """
    if dim <= 0:
        raise ValueError("dim must be > 0")

    vec = [0.0] * dim
    for tok in tokenize(text):
        idx = _hash_token(tok, dim)
        vec[idx] += 1.0

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def cosine_similarity_dense(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity for dense unit vectors.

    Since we normalize vectors at creation, cosine is just dot product.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))