from pydantic import BaseModel

from src.common.agents.runner import AgentRunner
from src.common.agents.simple_planner_verifier import SimplePlanner, SimpleVerifier
from src.common.tools.registry import ToolRegistry, ToolSpec
from src.common.utils.logging import new_run_context


class EchoIn(BaseModel):
    text: str


class EchoOut(BaseModel):
    echoed: str


def echo_tool(inp: EchoIn) -> dict:
    return {"echoed": inp.text}


def test_agent_runner_end_to_end_echo():
    reg = ToolRegistry()
    reg.register(ToolSpec("echo", "Echo tool", EchoIn, EchoOut, echo_tool))

    ctx = new_run_context()
    runner = AgentRunner(
        planner=SimplePlanner(),
        registry=reg,
        verifier=SimpleVerifier(),
        runs_dir="runs_test",
        run_id=ctx.run_id,
        max_steps=3,
    )

    result = runner.run("hello world")
    assert result.verified is True
    assert len(result.step_results) == 1
    assert result.step_results[0].status == "success"
    assert result.step_results[0].output["echoed"] == "hello world"