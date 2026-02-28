"""
text.py

Text utilities for deterministic RAG:

- normalize: lowercasing
- tokenize: basic tokenization (letters+digits)
- chunk_text: break long docs into overlapping chunks

We keep this minimal and dependency-free.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    """
    Tokenize into alphanumeric word tokens.

    Note:
    - This is intentionally simple.
    - Later: add stopword removal, stemming, better tokenization if needed.
    """
    return _WORD_RE.findall(normalize(text))


def chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """
    Chunk the text by character length with overlap.

    Why character-based chunking now?
    - Deterministic
    - No extra libs
    - Works for plain text/markdown

    chunk_size:
      Approx size of each chunk (chars).
    overlap:
      Overlap chars between chunks to reduce boundary losses.

    Returns:
      list of chunk strings
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    t = text.strip()
    if not t:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        chunks.append(t[start:end])
        if end == len(t):
            break
        start = end - overlap

    return chunks