from src.common.rag.index import ingest_folder, save_index
from src.common.rag.tool import RetrieveContextIn, RetrieveContextOut, retrieve_context_tool
from src.common.tools.registry import ToolRegistry, ToolSpec


def test_rag_tool_executes_via_registry(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("Onboarding SOP validate email and domain.", encoding="utf-8")

    idx = ingest_folder(docs_dir=str(docs_dir), chunk_size=200, overlap=20)
    index_path = tmp_path / "index.json"
    save_index(idx, str(index_path))

    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="retrieve_context",
            description="RAG tool",
            input_model=RetrieveContextIn,
            output_model=RetrieveContextOut,
            fn=retrieve_context_tool,
        )
    )

    out = reg.call(
        "retrieve_context",
        {
            "index_path": str(index_path),
            "query": "onboarding",
            "top_k": 3,
            "mode": "warn",
            "backend": "hybrid",
        },
    )

    assert out["query"] == "onboarding"
    assert len(out["results"]) >= 1