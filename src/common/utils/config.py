"""
config.py

Goal:
- Provide a single place to read environment/config settings.
- Keep this minimal now, but future-proof for Day 3+ when we add LLM providers and RAG.

Design choices:
- Use Pydantic BaseModel for typed config (strong signals of engineering discipline).
- Allow loading from environment variables (and later .env via python-dotenv).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """
    Minimal typed config for Day 2.

    Why dataclass?
    - Simple, lightweight, no runtime dependency.
    - Easy to extend later.

    Fields:
    - provider: which model runtime we eventually use (local/openai/bedrock)
    - log_level: INFO/DEBUG, etc.
    - runs_dir: where we store run artifacts (structured logs)
    """

    provider: str
    log_level: str
    runs_dir: str


def load_config() -> AppConfig:
    """
    Load configuration from environment variables.

    We intentionally DO NOT fail hard if .env doesn't exist, because:
    - Day 2 demo doesn't need secrets or model keys.
    - CI should run without any special environment.

    Later (Day 3+), we will:
    - load from .env using python-dotenv
    - validate provider-specific settings (OpenAI key, Ollama URL, etc.)
    """
    provider = os.getenv("AGENTIC_AI_PROVIDER", "local").strip()
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    # Where to store run logs. This makes the system "replayable" later.
    runs_dir = os.getenv("AGENTIC_AI_RUNS_DIR", "runs").strip()

    return AppConfig(provider=provider, log_level=log_level, runs_dir=runs_dir)