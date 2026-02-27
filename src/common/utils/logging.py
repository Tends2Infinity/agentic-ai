"""
logging.py

Structured JSON logs with run_id for traceability.
Writes JSONL: one event per line.

Enables later:
- trace viewer in Streamlit
- replay runs
- debugging and auditability
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunContext:
    run_id: str
    started_at_ms: int


def new_run_context() -> RunContext:
    return RunContext(run_id=str(uuid.uuid4()), started_at_ms=int(time.time() * 1000))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log_event(
    *,
    runs_dir: str,
    run_id: str,
    event_type: str,
    payload: dict[str, Any],
    also_print: bool = True,
) -> None:
    """
    Append a structured event to runs/<run_id>.jsonl and optionally print it.
    """
    ensure_dir(runs_dir)

    event = {
        "ts_ms": int(time.time() * 1000),
        "run_id": run_id,
        "event_type": event_type,
        "payload": payload,
    }

    filepath = os.path.join(runs_dir, f"{run_id}.jsonl")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    if also_print:
        print(json.dumps(event, ensure_ascii=False))