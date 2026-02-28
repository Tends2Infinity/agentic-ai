from src.common.rag.index import ingest_folder
from src.common.rag.retriever import search


def test_rag_backends_work(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "a.txt").write_text("Onboarding SOP validate email and domain.", encoding="utf-8")
    (docs_dir / "b.txt").write_text("Revenue KPI sum invoice amounts by month.", encoding="utf-8")

    idx = ingest_folder(docs_dir=str(docs_dir), chunk_size=200, overlap=20)

    res_tfidf = search(index=idx, query="onboarding", backend="tfidf")
    res_emb = search(index=idx, query="onboarding", backend="embeddings")
    res_hybrid = search(index=idx, query="onboarding", backend="hybrid")

    assert len(res_tfidf.results) >= 1
    assert len(res_emb.results) >= 1
    assert len(res_hybrid.results) >= 1