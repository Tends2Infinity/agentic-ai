"""
injection.py

Prompt injection hygiene.

RAG is dangerous if you treat retrieved text as "instructions".
Attackers can insert content like:
- "Ignore previous instructions"
- "Reveal system prompt"
- "Exfiltrate secrets"

We can't prevent all attacks with regex, but we can:
- Detect common patterns
- Warn or filter suspicious chunks
- Encourage a robust "untrusted context" policy

Design:
- detect_injection_patterns(text) -> list of warnings
- is_suspicious(text) convenience wrapper
"""
from __future__ import annotations

import re

# Common, high-signal prompt injection phrases.
# These patterns are intentionally simple and explainable.
# Later we can add a classifier / scoring model.
_PATTERNS: list[tuple[str, str]] = [
    (
        "ignore_previous",
        r"\bignore\b.*\b(previous|prior)\b.*\b(instruction|instructions|message|messages|context)\b",
    ),
    ("reveal_system_prompt", r"\b(system\s+prompt|developer\s+message)\b"),
    ("exfiltrate_secrets", r"\b(api\s+key|secret|password|token|credential)s?\b"),
    (
        "do_not_disclose",
        r"\b(do\s+not\s+mention|don['’]t\s+mention)\b.*\b(safety|policy|rule)s?\b",
    ),
    (
        "override_policy",
        r"\b(disable|bypass|override)\b.*\b(safety|policy|guardrail)s?\b",
    ),
]


def detect_injection_patterns(text: str) -> list[str]:
    """
    Return a list of warning codes if suspicious patterns are found.

    Important:
    - This is a heuristic (not perfect).
    - We aim for high recall for common attacks in early version.
    """
    t = text.lower()
    warnings: list[str] = []

    for code, pattern in _PATTERNS:
        if re.search(pattern, t, flags=re.IGNORECASE | re.DOTALL):
            warnings.append(code)

    return warnings


def is_suspicious(text: str) -> bool:
    """
    Convenience wrapper: returns True if any warnings are detected.
    """
    return len(detect_injection_patterns(text)) > 0