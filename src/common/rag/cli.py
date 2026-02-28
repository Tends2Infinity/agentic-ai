"""
cli.py

RAG CLI that makes the module demoable:

1) Ingest docs into an index:
   python -m src.common.rag.cli ingest --docs-dir data/sample_docs --out data/indexes/sample.json

2) Search:
   python -m src.common.rag.cli search --index data/indexes/sample.json --query "onboarding SOP" --mode trusted_only
"""

from __future__ import annotations

import typer

from src.common.rag.index import ingest_folder, load_index, save_index
from src.common.rag.retriever import search

app = typer.Typer(add_completion=False)


@app.command()
def ingest(
    docs_dir: str = typer.Option(..., help="Folder containing text/markdown documents."),
    out: str = typer.Option(..., help="Path to write the index JSON file."),
    chunk_size: int = typer.Option(500, help="Chunk size in characters."),
    overlap: int = typer.Option(80, help="Overlap between chunks in characters."),
) -> None:
    """
    Build an index from documents.
    """
    idx = ingest_folder(docs_dir=docs_dir, chunk_size=chunk_size, overlap=overlap)
    save_index(idx, out)
    print(f"✅ Index written: {out}")
    print(f"Chunks: {len(idx.chunks)}  |  Unique terms in IDF: {len(idx.idf)}")


@app.command("search")
def search_cmd(
    index: str = typer.Option(..., help="Path to index JSON file."),
    query: str = typer.Option(..., help="Search query string."),
    top_k: int = typer.Option(5, help="Number of chunks to return."),
    mode: str = typer.Option("warn", help="warn | trusted_only"),
) -> None:
    """
    Search an existing index.
    """
    idx = load_index(index)
    result = search(index=idx, query=query, top_k=top_k, mode=mode)

    print(f"\nQuery: {result.query}")
    print(f"Mode: {mode} | TopK: {top_k}")
    print("=" * 80)

    if not result.results:
        print("No results.")
        return

    for i, r in enumerate(result.results, start=1):
        warn = f"  ⚠️ warnings={r.warnings}" if r.warnings else ""
        print(f"\n[{i}] score={r.score:.4f}  source={r.source}  chunk_id={r.chunk_id}{warn}")
        print("-" * 80)
        print(r.text.strip())


if __name__ == "__main__":
    app()