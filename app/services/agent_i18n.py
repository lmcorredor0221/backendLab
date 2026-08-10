"""
Localization utilities for agent-facing prompts and responses.
Keep user-facing output aligned with the preferred language while
preserving technical identifiers and JSON contracts.
"""

from __future__ import annotations

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
