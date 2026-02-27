"""
registry.py

Governed registry of tools the agent is allowed to call.

Enforces:
- tool must be registered
- tool input validated via Pydantic schema
- tool output validated via Pydantic schema

This prevents:
- hallucinated tool calls
- invalid tool inputs
- inconsistent tool outputs

Later upgrades:
- permission tiers
- timeouts/retries
- audit policies
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """Raised when tool invocation or validation fails."""


@dataclass(frozen=True)
class ToolSpec:
    """
    Tool specification stored in the registry.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: Callable[[BaseModel], Any]


class ToolRegistry:
    """
    Stores ToolSpec objects by name and provides a strict call() API.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ToolError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        return self._tools[name]

    def call(self, name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
        """
        Strict tool invocation:
        1) Validate inputs
        2) Execute tool
        3) Validate outputs
        """
        spec = self.get(name)

        try:
            validated_in = spec.input_model.model_validate(raw_args)
        except ValidationError as e:
            raise ToolError(f"Invalid inputs for tool '{name}': {e}") from e

        result = spec.fn(validated_in)

        try:
            validated_out = spec.output_model.model_validate(result)
        except ValidationError as e:
            raise ToolError(f"Invalid output from tool '{name}': {e}") from e

        return validated_out.model_dump()