"""
demo_cli.py

Day 2 demo entrypoint.

Goal:
- Provide a clean CLI UX:
    python -m src.common.agents.demo_cli "add 2 3"
    python -m src.common.agents.demo_cli "hello world"

This proves:
- Planner → Executor → Verifier loop
- Governed ToolRegistry calls
- Structured JSONL logs under runs/<run_id>.jsonl
"""

from __future__ import annotations

import typer
from pydantic import BaseModel

from src.common.agents.runner import AgentRunner
from src.common.agents.simple_planner_verifier import SimplePlanner, SimpleVerifier
from src.common.tools.registry import ToolRegistry, ToolSpec
from src.common.utils.config import load_config
from src.common.utils.logging import new_run_context

app = typer.Typer(add_completion=False)


# -----------------------------
# Tool schemas (Pydantic models)
# -----------------------------

class EchoIn(BaseModel):
    text: str


class EchoOut(BaseModel):
    echoed: str


class AddIn(BaseModel):
    a: int
    b: int


class AddOut(BaseModel):
    sum: int


# -----------------------------
# Tool implementations
# -----------------------------

def echo_tool(inp: EchoIn) -> dict:
    """Echo tool: returns the same text under the 'echoed' key."""
    return {"echoed": inp.text}


def add_numbers_tool(inp: AddIn) -> dict:
    """Add tool: returns a+b under the 'sum' key."""
    return {"sum": inp.a + inp.b}


def build_registry() -> ToolRegistry:
    """
    Register demo tools.

    Later:
    - Ops Copilot will register: ticket_store, task_store, sop_store, file_search...
    - Analytics Agent will register: schema_introspect, sql_execute, sql_validate...
    """
    reg = ToolRegistry()

    reg.register(
        ToolSpec(
            name="echo",
            description="Echo input text.",
            input_model=EchoIn,
            output_model=EchoOut,
            fn=echo_tool,
        )
    )

    reg.register(
        ToolSpec(
            name="add_numbers",
            description="Add two integers.",
            input_model=AddIn,
            output_model=AddOut,
            fn=add_numbers_tool,
        )
    )

    return reg


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    user_input: str = typer.Argument(..., help='Any string. Examples: "add 2 3", "hello world".'),
) -> None:
    """
    Default command: run the demo agent for the provided USER_INPUT.

    Typer pattern:
    - We use @app.callback(invoke_without_command=True) so the first argument is user_input
      and we don't need a subcommand like "run".
    """
    # If a subcommand existed and was invoked, ctx.invoked_subcommand would be set.
    # We intentionally keep this demo to a single command for simplicity.
    _ = ctx

    cfg = load_config()
    run_ctx = new_run_context()

    registry = build_registry()
    planner = SimplePlanner()
    verifier = SimpleVerifier()

    runner = AgentRunner(
        planner=planner,
        registry=registry,
        verifier=verifier,
        runs_dir=cfg.runs_dir,
        run_id=run_ctx.run_id,
        max_steps=5,
    )

    result = runner.run(user_input)

    # Print a human-friendly summary (the JSON logs are already written by log_event).
    print("\n=== SUMMARY ===")
    print(f"run_id: {run_ctx.run_id}")
    print(f"objective: {result.plan.objective}")
    print(f"verified: {result.verified}")
    if result.verification_errors:
        print("verification_errors:")
        for e in result.verification_errors:
            print(f"  - {e}")

    print("\nstep results:")
    for sr in result.step_results:
        print(f"- {sr.step_id} tool={sr.tool_name} status={sr.status} output={sr.output} error={sr.error}")


if __name__ == "__main__":
    app()