from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


QUESTION_PREFIX_RE = re.compile(
    r"^(?:"
    r"informacion faltante|"
    r"informacion requerida|"
    r"completar informacion|"
    r"pregunta abierta|"
    r"gap en [a-z0-9_ -]+"
    r")\s*[:\-]\s*",
    re.IGNORECASE,
)


def normalize_question_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").strip().lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = " ".join(text.split())
    for _ in range(3):
        stripped = QUESTION_PREFIX_RE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def question_dedupe_signature(question_text: Any, *, fallback_key: Any = "") -> str:
    normalized = normalize_question_text(question_text)
    if normalized:
        return f"q:{normalized}"
    fallback = normalize_question_text(fallback_key)
    return f"k:{fallback or 'item'}"


def merge_unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged
