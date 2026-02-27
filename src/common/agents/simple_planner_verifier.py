"""
simple_planner_verifier.py

Deterministic Day-2 implementations of:
- Planner: generates a plan without any LLM (rule-based)
- Verifier: checks plan + results for basic safety and correctness

Why deterministic now?
- Makes testing easy.
- Makes debugging easy.
- Keeps CI stable.
- Later we replace Planner with an LLM planner without changing Runner/Registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.agents.types import Plan, PlanStep, StepResult
from src.common.tools.registry import ToolRegistry


@dataclass(frozen=True)
class SimplePlanner:
    """
    Tiny rule-based planner for Day 2.

    Rules:
    - If user_input contains "add", plan calls add_numbers.
      Expected format: "add 2 3" (last two tokens are ints)
    - Otherwise, plan calls echo.
    """

    def create_plan(self, user_input: str) -> Plan:
        text = user_input.strip()

        if "add" in text.lower():
            parts = text.split()
            a = int(parts[-2])
            b = int(parts[-1])

            return Plan(
                objective=f"Add {a} and {b}",
                steps=[
                    PlanStep(
                        step_id="step_1",
                        tool_name="add_numbers",
                        args={"a": a, "b": b},
                        rationale="User requested an addition; use the add_numbers tool.",
                    )
                ],
            )

        return Plan(
            objective="Echo the user input",
            steps=[
                PlanStep(
                    step_id="step_1",
                    tool_name="echo",
                    args={"text": text},
                    rationale="No specific operation detected; echo the input for demonstration.",
                )
            ],
        )


@dataclass(frozen=True)
class SimpleVerifier:
    """
    Verifier for Day 2.

    Pre-flight checks:
    - Plan not empty
    - Step IDs unique
    - Tools referenced exist in registry

    Post-flight checks:
    - All executed steps succeeded
    """

    def verify_plan(self, plan: Plan, registry: ToolRegistry) -> list[str]:
        errors: list[str] = []

        if not plan.steps:
            errors.append("Plan has no steps.")

        step_ids = [s.step_id for s in plan.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step_id found in plan.")

        for step in plan.steps:
            if not registry.has_tool(step.tool_name):
                errors.append(f"Unknown tool referenced in plan: {step.tool_name}")

        return errors

    def verify_results(self, plan: Plan, results: list[StepResult]) -> list[str]:
        errors: list[str] = []

        for r in results:
            if r.status != "success":
                errors.append(f"Step failed: {r.step_id} ({r.tool_name}) error={r.error}")

        return errors