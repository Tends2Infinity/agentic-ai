"""
tfidf.py

Pure-python TF-IDF + cosine similarity for sparse vectors.

This is our deterministic retrieval baseline:
- Works offline
- Works in CI
- No model keys required

Later we can swap retrieval backend to embeddings by:
- Keeping RagIndex / SearchResult interfaces stable
- Replacing vectorization/scoring internals
"""

from __future__ import annotations

import math
from collections import Counter

from src.common.rag.text import tokenize


def build_idf(doc_texts: list[str]) -> dict[str, float]:
    """
    Compute IDF for a set of documents/chunks.

    idf(term) = log((N + 1) / (df + 1)) + 1

    We add +1 smoothing to avoid division by zero.
    """
    n = len(doc_texts)
    df: Counter[str] = Counter()

    for text in doc_texts:
        terms = set(tokenize(text))
        df.update(terms)

    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((n + 1) / (freq + 1)) + 1.0

    return idf


def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    """
    Convert text -> sparse TF-IDF vector: term -> weight.

    tf(term) is raw term count in this version (simple).
    """
    terms = tokenize(text)
    tf = Counter(terms)

    vec: dict[str, float] = {}
    for term, count in tf.items():
        if term in idf:
            vec[term] = float(count) * idf[term]

    return vec


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """
    Compute cosine similarity between sparse vectors a and b.

    cosine = dot(a,b) / (||a|| * ||b||)
    """
    if not a or not b:
        return 0.0

    # dot product
    dot = 0.0
    # Iterate smaller dict for speed
    if len(a) > len(b):
        a, b = b, a
    for term, w in a.items():
        dot += w * b.get(term, 0.0)

    # norms
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)