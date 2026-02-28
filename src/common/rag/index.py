"""
index.py

RAG ingestion and index persistence.

Responsibilities:
- read documents from a folder
- chunk them
- build IDF across all chunks
- create TF-IDF vectors for each chunk
- save/load index to/from JSON

Why save index?
- Repeatable retrieval without re-ingesting
- Enables "ingest once, query many times"
"""

from __future__ import annotations

import json
import os

from src.common.rag.text import chunk_text
from src.common.rag.tfidf import build_idf, vectorize
from src.common.rag.types import IndexedChunk, RagIndex


def read_text_file(path: str) -> str:
    """
    Read a text/markdown file with UTF-8 fallback.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def ingest_folder(
    *,
    docs_dir: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> RagIndex:
    """
    Ingest all files under docs_dir (non-recursive in v1).

    Returns:
      RagIndex containing idf map + indexed chunks.

    Note:
    - We keep v1 simple: ingest all files in folder.
    - Later: recursive walk, file type filters, metadata, etc.
    """
    if not os.path.isdir(docs_dir):
        raise ValueError(f"docs_dir not found: {docs_dir}")

    files = [
        os.path.join(docs_dir, f)
        for f in os.listdir(docs_dir)
        if os.path.isfile(os.path.join(docs_dir, f))
    ]
    if not files:
        raise ValueError(f"No files found in docs_dir: {docs_dir}")

    # 1) Chunk documents
    raw_chunks: list[tuple[str, str]] = []  # (source, chunk_text)
    for path in sorted(files):
        text = read_text_file(path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for c in chunks:
            raw_chunks.append((os.path.basename(path), c))

    # 2) Build IDF across all chunks
    idf = build_idf([c for _, c in raw_chunks])

    # 3) Build vectors for each chunk
    indexed_chunks: list[IndexedChunk] = []
    for i, (source, ctext) in enumerate(raw_chunks):
        chunk_id = f"{source}::chunk_{i:04d}"
        vec = vectorize(ctext, idf)
        indexed_chunks.append(IndexedChunk(chunk_id=chunk_id, source=source, text=ctext, vector=vec))

    return RagIndex(idf=idf, chunks=indexed_chunks)


def save_index(index: RagIndex, path: str) -> None:
    """
    Save index as JSON.

    This is not optimized for huge corpora, but it's perfect for:
    - demos
    - small knowledge bases
    - CI-friendly examples
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "idf": index.idf,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "text": c.text,
                "vector": c.vector,
            }
            for c in index.chunks
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_index(path: str) -> RagIndex:
    """
    Load index from JSON.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    chunks = [
        IndexedChunk(
            chunk_id=c["chunk_id"],
            source=c["source"],
            text=c["text"],
            vector=c["vector"],
        )
        for c in payload["chunks"]
    ]
    return RagIndex(idf=payload["idf"], chunks=chunks)