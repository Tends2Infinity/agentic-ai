"""
retriever.py

RAG retrieval:
- vectorize query using index.idf
- cosine similarity against chunk vectors
- rank top-k
- apply injection hygiene:
  - warn mode: keep chunks but return warnings
  - trusted_only: filter suspicious chunks out

This module is intentionally "LLM-agnostic".
"""

from __future__ import annotations

from src.common.rag.injection import detect_injection_patterns
from src.common.rag.tfidf import cosine_similarity, vectorize
from src.common.rag.types import RagIndex, RetrievedChunk, SearchResult


def search(
    *,
    index: RagIndex,
    query: str,
    top_k: int = 5,
    mode: str = "warn",
) -> SearchResult:
    """
    Search index for relevant chunks.

    mode:
      - "warn": return suspicious chunks with warnings
      - "trusted_only": drop suspicious chunks

    Returns:
      SearchResult(query, results[])
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if mode not in ("warn", "trusted_only"):
        raise ValueError("mode must be 'warn' or 'trusted_only'")

    q_vec = vectorize(query, index.idf)

    scored: list[RetrievedChunk] = []
    for c in index.chunks:
        score = cosine_similarity(q_vec, c.vector)
        if score <= 0.0:
            continue

        warnings = detect_injection_patterns(c.text)
        if mode == "trusted_only" and warnings:
            continue

        scored.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                source=c.source,
                text=c.text,
                score=score,
                warnings=warnings,
            )
        )

    scored.sort(key=lambda r: r.score, reverse=True)
    return SearchResult(query=query, results=scored[:top_k])