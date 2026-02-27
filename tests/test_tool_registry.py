import pytest
from pydantic import BaseModel

from src.common.tools.registry import ToolError, ToolRegistry, ToolSpec


class InModel(BaseModel):
    x: int


class OutModel(BaseModel):
    y: int


def tool_fn(inp: InModel) -> dict:
    return {"y": inp.x + 1}


def test_registry_register_and_call_success():
    reg = ToolRegistry()
    reg.register(ToolSpec("t1", "test tool", InModel, OutModel, tool_fn))

    out = reg.call("t1", {"x": 10})
    assert out["y"] == 11


def test_registry_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        reg.call("missing", {"x": 1})


def test_registry_input_validation_fails():
    reg = ToolRegistry()
    reg.register(ToolSpec("t1", "test tool", InModel, OutModel, tool_fn))

    with pytest.raises(ToolError):
        reg.call("t1", {"x": "not_an_int"})