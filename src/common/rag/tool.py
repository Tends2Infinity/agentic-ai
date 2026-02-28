"""
tool.py

Expose RAG retrieval as a governed ToolRegistry tool.

Tool name:
- retrieve_context

Why this is important:
- AgentRunner executes only registered tools.
- This makes RAG callable by an agent safely and consistently.
- Schema validation ensures stable inputs/outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.common.rag.index import load_index
from src.common.rag.retriever import search


class RetrieveContextIn(BaseModel):
    """
    Input schema for the RAG tool.

    index_path:
      Path to JSON index built by ingest.
    query:
      Query string from the agent/user.
    top_k:
      How many chunks to retrieve.
    mode:
      warn | trusted_only (injection hygiene)
    backend:
      tfidf | embeddings | hybrid
    """

    index_path: str = Field(..., description="Path to RAG index JSON file.")
    query: str = Field(..., description="Search query.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to return.")
    mode: str = Field("warn", description="warn | trusted_only")
    backend: str = Field("tfidf", description="tfidf | embeddings | hybrid")


class RetrievedItem(BaseModel):
    chunk_id: str
    source: str
    score: float
    warnings: list[str]
    text: str


class RetrieveContextOut(BaseModel):
    query: str
    results: list[RetrievedItem]


def retrieve_context_tool(inp: RetrieveContextIn) -> dict:
    """
    Tool implementation.

    Loads index from disk and runs retrieval.
    Returns a dict matching RetrieveContextOut schema.
    """
    idx = load_index(inp.index_path)
    res = search(index=idx, query=inp.query, top_k=inp.top_k, mode=inp.mode, backend=inp.backend)
    return {
        "query": res.query,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "source": r.source,
                "score": r.score,
                "warnings": r.warnings,
                "text": r.text,
            }
            for r in res.results
        ],
    }