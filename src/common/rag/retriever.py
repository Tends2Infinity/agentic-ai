"""
retriever.py

RAG retrieval with multiple backends:
- tfidf: sparse TF-IDF cosine
- embeddings: dense hashed embeddings cosine
- hybrid: combine both scores

Injection hygiene:
- warn: include suspicious chunks but attach warnings
- trusted_only: drop suspicious chunks

Interface stability:
- Still returns SearchResult with RetrievedChunk objects.
"""

from __future__ import annotations

from src.common.rag.embeddings import cosine_similarity_dense, embed_text
from src.common.rag.injection import detect_injection_patterns
from src.common.rag.tfidf import cosine_similarity, vectorize
from src.common.rag.types import RagIndex, RetrievedChunk, SearchResult


def _normalize_scores(pairs: list[tuple[RetrievedChunk, float]]) -> list[tuple[RetrievedChunk, float]]:
    """
    Normalize scores to [0,1] using max scaling, to combine backends fairly.
    """
    if not pairs:
        return pairs
    max_score = max(s for _, s in pairs)
    if max_score <= 0.0:
        return [(r, 0.0) for r, _ in pairs]
    return [(r, s / max_score) for r, s in pairs]


def search(
    *,
    index: RagIndex,
    query: str,
    top_k: int = 5,
    mode: str = "warn",
    backend: str = "tfidf",
) -> SearchResult:
    """
    Search index for relevant chunks.

    mode:
      - warn
      - trusted_only

    backend:
      - tfidf
      - embeddings
      - hybrid

    Returns:
      SearchResult(query, results[])
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if mode not in ("warn", "trusted_only"):
        raise ValueError("mode must be 'warn' or 'trusted_only'")
    if backend not in ("tfidf", "embeddings", "hybrid"):
        raise ValueError("backend must be 'tfidf', 'embeddings', or 'hybrid'")

    # Prepare query representations
    q_tfidf = vectorize(query, index.idf)
    q_emb = embed_text(query, dim=512)

    # Collect candidates with hygiene applied
    candidates: list[RetrievedChunk] = []
    for c in index.chunks:
        warnings = detect_injection_patterns(c.text)
        if mode == "trusted_only" and warnings:
            continue

        candidates.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                source=c.source,
                text=c.text,
                score=0.0,  # placeholder; we fill later
                warnings=warnings,
            )
        )

    # Score candidates
    tfidf_scored: list[tuple[RetrievedChunk, float]] = []
    emb_scored: list[tuple[RetrievedChunk, float]] = []

    for r in candidates:
        # Find the matching chunk in index (by chunk_id)
        # For small corpora this linear lookup is fine; later we can map chunk_id -> chunk object.
        chunk = next(c for c in index.chunks if c.chunk_id == r.chunk_id)

        if backend in ("tfidf", "hybrid"):
            s1 = cosine_similarity(q_tfidf, chunk.tfidf_vector)
            tfidf_scored.append((r, s1))

        if backend in ("embeddings", "hybrid"):
            # embedding_vector might be missing for older indexes
            if chunk.embedding_vector is None:
                s2 = 0.0
            else:
                s2 = cosine_similarity_dense(q_emb, chunk.embedding_vector)
            emb_scored.append((r, s2))

    # Combine according to backend
    final: list[RetrievedChunk] = []
    if backend == "tfidf":
        for r, s in tfidf_scored:
            if s > 0.0:
                final.append(RetrievedChunk(**{**r.__dict__, "score": s}))
    elif backend == "embeddings":
        for r, s in emb_scored:
            if s > 0.0:
                final.append(RetrievedChunk(**{**r.__dict__, "score": s}))
    else:
        # Hybrid: normalize each score list, then weighted sum
        tfidf_norm = _normalize_scores(tfidf_scored)
        emb_norm = _normalize_scores(emb_scored)

        # Build maps chunk_id -> normalized score
        tfidf_map = {r.chunk_id: s for r, s in tfidf_norm}
        emb_map = {r.chunk_id: s for r, s in emb_norm}

        for r in candidates:
            s = 0.6 * tfidf_map.get(r.chunk_id, 0.0) + 0.4 * emb_map.get(r.chunk_id, 0.0)
            if s > 0.0:
                final.append(RetrievedChunk(**{**r.__dict__, "score": s}))

    final.sort(key=lambda x: x.score, reverse=True)
    return SearchResult(query=query, results=final[:top_k])