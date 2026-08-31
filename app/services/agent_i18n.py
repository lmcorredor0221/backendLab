"""
Localization utilities for agent-facing prompts and responses.
Keep user-facing output aligned with the preferred language while
preserving technical identifiers and JSON contracts.
"""

from __future__ import annotations

import re
from typing import Any

SUPPORTED_AGENT_LANGUAGES = {"es", "en", "pt"}

LANGUAGE_DIRECTIVES: dict[str, str] = {
    "es": (
        "\n\n[INSTRUCCION DE IDIOMA Y PRESENTACION]\n"
        "Debes comunicarte y generar todas las explicaciones, analisis, resumenes y respuestas al usuario en ESPANOL. "
        "Manten inalterados los identificadores tecnicos, nombres de herramientas, estados y esquemas JSON requeridos por la plataforma."
    ),
    "en": (
        "\n\n[LANGUAGE AND PRESENTATION DIRECTIVE]\n"
        "You must communicate and generate all user-facing explanations, analysis, summaries, and responses in ENGLISH. "
        "Keep internal technical identifiers, tool names, state labels, and required JSON schemas strictly unchanged."
    ),
    "pt": (
        "\n\n[DIRETIVA DE IDIOMA E APRESENTACAO]\n"
        "Voce deve se comunicar e gerar todas as explicacoes, analises, resumos e respostas para o usuario em PORTUGUES. "
        "Mantenha inalterados os identificadores tecnicos internos, nomes de ferramentas, estados e esquemas JSON exigidos pela plataforma."
    ),
}


def get_effective_language(language: str | None) -> str:
    lang_key = (language or "es").strip().lower()
    return lang_key if lang_key in SUPPORTED_AGENT_LANGUAGES else "es"


def apply_agent_language_directive(system_prompt: str, language: str = "es") -> str:
    directive = LANGUAGE_DIRECTIVES[get_effective_language(language)]
    return f"{system_prompt.rstrip()}{directive}"


_TECHNICAL_FIELD_HINTS = (
    "id",
    "key",
    "version",
    "schema",
    "status",
    "stage",
    "source",
    "ref",
    "refs",
    "contract",
    "enum",
)

_VISIBLE_FIELD_HINTS = (
    "answer",
    "description",
    "finding",
    "fit",
    "hypothesis",
    "impact",
    "information",
    "label",
    "narrative",
    "note",
    "policy",
    "question",
    "rationale",
    "reason",
    "risk",
    "strategy",
    "summary",
    "title",
    "tradeoff",
    "warning",
    "why",
)

_LANGUAGE_MARKERS: dict[str, set[str]] = {
    "en": {
        "about",
        "and",
        "before",
        "because",
        "data",
        "decision",
        "generate",
        "generated",
        "needs",
        "pending",
        "question",
        "requires",
        "should",
        "system",
        "the",
        "this",
        "user",
        "with",
        "without",
    },
    "es": {
        "antes",
        "con",
        "datos",
        "debe",
        "decision",
        "el",
        "esta",
        "generado",
        "generar",
        "la",
        "necesita",
        "pendiente",
        "porque",
        "pregunta",
        "que",
        "requiere",
        "sistema",
        "sin",
        "usuario",
    },
    "pt": {
        "antes",
        "com",
        "dados",
        "deve",
        "decisao",
        "esta",
        "gerado",
        "gerar",
        "necessita",
        "pendente",
        "porque",
        "pergunta",
        "que",
        "requer",
        "sem",
        "sistema",
        "usuario",
    },
}


def _is_visible_field(field_name: str) -> bool:
    name = field_name.lower()
    if any(hint in name for hint in _VISIBLE_FIELD_HINTS):
        return True
    return not any(hint in name for hint in _TECHNICAL_FIELD_HINTS)


def _iter_visible_texts(value: Any, *, field_name: str = "", depth: int = 0) -> list[str]:
    if value is None or depth > 5:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text and _is_visible_field(field_name) else []
    if isinstance(value, (int, float, bool)):
        return []
    if isinstance(value, (list, tuple, set)):
        texts: list[str] = []
        for item in value:
            texts.extend(_iter_visible_texts(item, field_name=field_name, depth=depth + 1))
        return texts
    if hasattr(value, "model_dump"):
        try:
            return _iter_visible_texts(value.model_dump(mode="json"), field_name=field_name, depth=depth + 1)
        except Exception:  # noqa: BLE001
            return []
    if isinstance(value, dict):
        texts: list[str] = []
        for key, item in value.items():
            texts.extend(_iter_visible_texts(item, field_name=str(key), depth=depth + 1))
        return texts
    if hasattr(value, "__dict__"):
        return _iter_visible_texts(vars(value), field_name=field_name, depth=depth + 1)
    return []


def detect_user_visible_language_status(value: Any, language: str = "es") -> str:
    """Return ok/mismatch/not_checked for user-visible text only.

    This is intentionally conservative. It avoids technical identifiers and only
    flags a mismatch when a substantial amount of visible prose points elsewhere.
    """

    effective_language = get_effective_language(language)
    texts = _iter_visible_texts(value)
    body = " ".join(texts).lower()
    words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑçãõâêôÇÃÕÂÊÔ]+", body)
    if len(words) < 20:
        return "not_checked"

    desired_count = sum(1 for word in words if word in _LANGUAGE_MARKERS[effective_language])
    other_counts = {
        lang: sum(1 for word in words if word in markers)
        for lang, markers in _LANGUAGE_MARKERS.items()
        if lang != effective_language
    }
    strongest_other = max(other_counts.values() or [0])
    if strongest_other >= 8 and strongest_other > max(3, desired_count) * 1.5:
        return "mismatch"
    return "ok"
