from __future__ import annotations

import typer

from src.common.agents.runner import AgentRunner
from src.common.agents.simple_planner_verifier import SimpleVerifier
from src.common.agents.types import Plan, PlanStep
from src.common.rag.tool import RetrieveContextIn, RetrieveContextOut, retrieve_context_tool
from src.common.tools.registry import ToolRegistry, ToolSpec
from src.common.utils.config import load_config
from src.common.utils.logging import new_run_context

app = typer.Typer(add_completion=False)


class RagPlanner:
    def __init__(self, index_path: str, backend: str, mode: str, top_k: int) -> None:
        self.index_path = index_path
        self.backend = backend
        self.mode = mode
        self.top_k = top_k

    def create_plan(self, user_input: str) -> Plan:
        return Plan(
            objective="Retrieve relevant context for the query",
            steps=[
                PlanStep(
                    step_id="step_1",
                    tool_name="retrieve_context",
                    args={
                        "index_path": self.index_path,
                        "query": user_input,
                        "top_k": self.top_k,
                        "mode": self.mode,
                        "backend": self.backend,
                    },
                    rationale="Use RAG to fetch relevant chunks with injection hygiene.",
                )
            ],
        )


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="retrieve_context",
            description="Retrieve relevant chunks from a RAG index with injection hygiene.",
            input_model=RetrieveContextIn,
            output_model=RetrieveContextOut,
            fn=retrieve_context_tool,
        )
    )
    return reg


@app.command("run")
def run_cmd(
    query: str = typer.Argument(..., help="Query string, e.g. 'onboarding SOP'"),
    index_path: str = typer.Option("data/indexes/sample.json", help="Path to the RAG index JSON."),
    backend: str = typer.Option("hybrid", help="tfidf | embeddings | hybrid"),
    mode: str = typer.Option("warn", help="warn | trusted_only"),
    top_k: int = typer.Option(5, help="Top K results"),
) -> None:
    cfg = load_config()
    ctx = new_run_context()

    registry = build_registry()
    planner = RagPlanner(index_path=index_path, backend=backend, mode=mode, top_k=top_k)
    verifier = SimpleVerifier()

    runner = AgentRunner(
        planner=planner,
        registry=registry,
        verifier=verifier,
        runs_dir=cfg.runs_dir,
        run_id=ctx.run_id,
        max_steps=3,
    )

    result = runner.run(query)

    print("\n=== SUMMARY ===")
    print(f"run_id: {ctx.run_id}")
    print(f"verified: {result.verified}")

    # Print returned citations in a readable way
    for sr in result.step_results:
        print(f"\n- {sr.step_id} tool={sr.tool_name} status={sr.status}")
        if sr.output and "results" in sr.output:
            for i, item in enumerate(sr.output["results"], start=1):
                warn = f" warnings={item['warnings']}" if item.get("warnings") else ""
                print(f"  [{i}] score={item['score']:.4f} source={item['source']} chunk={item['chunk_id']}{warn}")


if __name__ == "__main__":
    app()