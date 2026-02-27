"""
types.py

Typed data structures for Planner → Executor → Verifier.

Keeping types explicit is a credibility signal:
- Planner produces Plan
- Executor produces StepResults
- Verifier checks both plan and results

Python 3.11 typing style:
- Prefer built-in generics: list[...] and dict[...]
- Prefer X | None over Optional[X]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PlanStep:
    """
    One unit of work in a Plan.

    tool_name must match ToolRegistry tool names.
    args will be validated against tool input schema (Pydantic) at runtime.
    rationale is human-readable reasoning for debugging/eval.
    """

    step_id: str
    tool_name: str
    args: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class Plan:
    """
    A plan is an ordered list of steps.
    """

    objective: str
    steps: list[PlanStep]


@dataclass(frozen=True)
class StepResult:
    """
    Result of executing one plan step.
    """

    step_id: str
    tool_name: str
    status: Literal["success", "failure"]
    output: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    """
    Final output of a run: plan + results + verification status.
    """

    plan: Plan
    step_results: list[StepResult]
    verified: bool
    verification_errors: list[str] = field(default_factory=list)