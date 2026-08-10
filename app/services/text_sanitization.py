from __future__ import annotations

import re
from pathlib import Path


SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str | re.Callable[[re.Match[str]], str]]] = [
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"), "[REDACTED_GOOGLE_KEY]"),
    (
        re.compile(r"(?im)\b(password|passwd|api[_-]?key|secret|token)\b\s*[:=]\s*([^\s,;]+)"),
        lambda match: f"{match.group(1)}=[REDACTED]",
    ),
    (
        re.compile(r"(?im)(authorization\s*:\s*bearer\s+)([A-Za-z0-9._-]+)"),
        lambda match: f"{match.group(1)}[REDACTED]",
    ),
]


def sanitize_text_content(value: str) -> str:
    sanitized = value.replace("\x00", "")
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def read_sanitized_utf8_text(path: Path) -> str:
    return sanitize_text_content(path.read_text(encoding="utf-8", errors="ignore"))
