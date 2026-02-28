"""
types.py

RAG data structures.

We keep these plain and explicit because:
- They are stable interfaces for the rest of the platform.
- We can later swap TF-IDF retrieval for embeddings without changing callers.

Key idea:
- Ingestion produces an Index (documents -> chunks -> vectors).
- Retrieval returns RetrievedChunk results with citations + optional warnings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """
    A chunk is a small piece of source text.

    chunk_id:
      Unique identifier within the index.
    source:
      Path or filename of the original document.
    text:
      Actual chunk content.
    """

    chunk_id: str
    source: str
    text: str


@dataclass(frozen=True)
class IndexedChunk(Chunk):
    """
    Chunk plus a sparse TF-IDF vector.

    vector is a sparse dict: term -> tf-idf weight
    We use sparse representation because most terms are not present in a chunk.
    """

    vector: dict[str, float]


@dataclass(frozen=True)
class RagIndex:
    """
    A RAG index is:

    - idf: term -> inverse document frequency weight
    - chunks: list of indexed chunks

    Why store idf separately?
    - Query vector uses the SAME idf weights as the chunks.
    - Ensures consistent scoring.
    """

    idf: dict[str, float]
    chunks: list[IndexedChunk]


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Retrieval output for one chunk.

    score:
      Similarity score between query and chunk.
    warnings:
      Prompt-injection warnings derived from content checks.
    """

    chunk_id: str
    source: str
    text: str
    score: float
    warnings: list[str]


@dataclass(frozen=True)
class SearchResult:
    """
    Full search response.

    query:
      Original user query
    results:
      Ranked list of retrieved chunks
    """

    query: str
    results: list[RetrievedChunk]