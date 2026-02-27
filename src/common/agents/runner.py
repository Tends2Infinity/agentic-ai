"""
runner.py

Goal:
- Implement Planner → Executor → Verifier architecture as a deterministic runtime.

Why this architecture matters:
- Planner: decides what to do (later powered by LLM)
- Executor: does it (tool calls)
- Verifier: checks it (guardrails, correctness constraints)

This separation is a credibility multiplier because it mirrors real agent systems:
- You can swap planner (LLM vs rules)
- Keep executor stable and safe
- Improve verifier for reliability over time
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.common.agents.types import Plan, RunResult, StepResult
from src.common.tools.registry import ToolError, ToolRegistry
from src.common.utils.logging import log_event


class Planner(Protocol):
    """
    Planner interface.

    In Day 2 we implement a deterministic planner (rule-based).
    Later we implement an LLM planner that outputs the same Plan structure.
    """

    def create_plan(self, user_input: str) -> Plan: ...


class Verifier(Protocol):
    """
    Verifier interface.

    Verifier checks:
    - plan validity (tool exists, args shape, step constraints)
    - step outputs (optional checks)
    """

    def verify_plan(self, plan: Plan, registry: ToolRegistry) -> list[str]: ...

    def verify_results(self, plan: Plan, results: list[StepResult]) -> list[str]: ...


@dataclass(frozen=True)
class AgentRunner:
    """
    AgentRunner glues everything together.

    Inputs:
    - planner: creates plan
    - registry: executes tools
    - verifier: validates plan and results
    - runs_dir/run_id: structured logging configuration
    """

    planner: Planner
    registry: ToolRegistry
    verifier: Verifier
    runs_dir: str
    run_id: str
    max_steps: int = 8

    def run(self, user_input: str) -> RunResult:
        # 1) PLANNING
        plan = self.planner.create_plan(user_input)
        log_event(
            runs_dir=self.runs_dir,
            run_id=self.run_id,
            event_type="plan_created",
            payload={
                "objective": plan.objective,
                "num_steps": len(plan.steps),
                "steps": [
                    {
                        "step_id": s.step_id,
                        "tool_name": s.tool_name,
                        "args": s.args,
                        "rationale": s.rationale,
                    }
                    for s in plan.steps
                ],
            },
        )

        # 2) PLAN VERIFICATION (pre-flight)
        plan_errors = self.verifier.verify_plan(plan, self.registry)
        if plan_errors:
            log_event(
                runs_dir=self.runs_dir,
                run_id=self.run_id,
                event_type="plan_verification_failed",
                payload={"errors": plan_errors},
            )
            return RunResult(plan=plan, step_results=[], verified=False, verification_errors=plan_errors)

        # 3) EXECUTION
        step_results: list[StepResult] = []
        for idx, step in enumerate(plan.steps):
            if idx >= self.max_steps:
                step_results.append(
                    StepResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        status="failure",
                        error=f"Max steps exceeded ({self.max_steps}).",
                    )
                )
                break

            log_event(
                runs_dir=self.runs_dir,
                run_id=self.run_id,
                event_type="tool_call_started",
                payload={"step_id": step.step_id, "tool_name": step.tool_name, "args": step.args},
            )

            try:
                out = self.registry.call(step.tool_name, step.args)
                step_results.append(
                    StepResult(step_id=step.step_id, tool_name=step.tool_name, status="success", output=out)
                )
                log_event(
                    runs_dir=self.runs_dir,
                    run_id=self.run_id,
                    event_type="tool_call_succeeded",
                    payload={"step_id": step.step_id, "tool_name": step.tool_name, "output": out},
                )
            except ToolError as e:
                step_results.append(
                    StepResult(step_id=step.step_id, tool_name=step.tool_name, status="failure", error=str(e))
                )
                log_event(
                    runs_dir=self.runs_dir,
                    run_id=self.run_id,
                    event_type="tool_call_failed",
                    payload={"step_id": step.step_id, "tool_name": step.tool_name, "error": str(e)},
                )

        # 4) RESULT VERIFICATION (post-flight)
        result_errors = self.verifier.verify_results(plan, step_results)
        verified = len(result_errors) == 0

        log_event(
            runs_dir=self.runs_dir,
            run_id=self.run_id,
            event_type="run_completed",
            payload={
                "verified": verified,
                "verification_errors": result_errors,
                "num_steps_executed": len(step_results),
            },
        )

        return RunResult(
            plan=plan,
            step_results=step_results,
            verified=verified,
            verification_errors=result_errors,
        )