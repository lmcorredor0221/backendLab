from __future__ import annotations

import re
import unicodedata


SUPPORTED_TITLE_LANGUAGES = {"es", "en", "pt"}
DEFAULT_MAX_PROJECT_TITLE_LENGTH = 30

_FALLBACK_TITLES = {
    "es": "Proyecto IA",
    "en": "AI Project",
    "pt": "Projeto IA",
}

_STOPWORDS = {
    "a",
    "al",
    "and",
    "as",
    "com",
    "como",
    "con",
    "contra",
    "da",
    "das",
    "de",
    "del",
    "do",
    "dos",
    "e",
    "el",
    "em",
    "en",
    "every",
    "for",
    "from",
    "la",
    "las",
    "le",
    "los",
    "me",
    "mi",
    "mis",
    "my",
    "need",
    "necesito",
    "no",
    "nos",
    "o",
    "of",
    "para",
    "por",
    "que",
    "quiero",
    "sem",
    "sin",
    "su",
    "sus",
    "the",
    "to",
    "tu",
    "um",
    "uma",
    "un",
    "una",
    "y",
}

_FILLER_PATTERNS = (
    r"\bquiero\s+(automatizar|crear|hacer|validar)\b",
    r"\bqueremos\s+(automatizar|crear|hacer|validar|reducir)\b",
    r"\bnecesito\s+(automatizar|crear|hacer|validar)\b",
    r"\bnecesitamos\s+(automatizar|crear|hacer|validar)\b",
    r"\btengo\s+\d*\s*(personas?|equipos?|usuarios?)?\b",
    r"\btenemos\s+\d*\s*(personas?|equipos?|usuarios?)?\b",
    r"\bi\s+(want|need|have)\s+to\b",
    r"\bwe\s+(want|need|have)\s+to\b",
    r"\bquero\s+(automatizar|criar|fazer|validar)\b",
    r"\bpreciso\s+(automatizar|criar|fazer|validar)\b",
)

_ACRONYMS = {"AI", "API", "B2B", "B2C", "CRM", "ERP", "HITL", "IA", "PDF", "RAG", "SAP", "TI"}

_TITLE_RULES: tuple[tuple[tuple[tuple[str, ...], ...], dict[str, str]], ...] = (
    ((("factura", "facturas", "invoice", "invoices", "fatura", "faturas"),), {"es": "Facturas IA", "en": "AI Invoices", "pt": "Faturas IA"}),
    (
        (
            ("correo", "correos", "email", "emails", "e-mail", "e-mails"),
            ("excel", "spreadsheet", "planilha"),
        ),
        {"es": "Correos a Excel", "en": "Email to Excel", "pt": "E-mails Excel"},
    ),
    ((("contrato", "contratos", "contract", "contracts", "politica", "politicas"),), {"es": "Contratos IA", "en": "AI Contracts", "pt": "Contratos IA"}),
    (
        (
            ("cliente", "clientes", "customer", "customers"),
            ("duda", "dudas", "question", "questions", "suporte", "soporte", "support"),
        ),
        {"es": "Soporte Clientes", "en": "Customer Support", "pt": "Suporte Clientes"},
    ),
    ((("venta", "ventas", "sales", "venda", "vendas", "pipeline"),), {"es": "Ventas IA", "en": "AI Sales", "pt": "Vendas IA"}),
    ((("demanda", "demand", "forecast", "prevision", "previsao"),), {"es": "Demanda IA", "en": "AI Demand", "pt": "Demanda IA"}),
    ((("ticket", "tickets", "incidente", "incidentes", "it", "ti"),), {"es": "Soporte TI", "en": "IT Support", "pt": "Suporte TI"}),
)


def _normalize_language(language: str | None) -> str:
    normalized = (language or "es").strip().lower()
    return normalized if normalized in SUPPORTED_TITLE_LANGUAGES else "es"


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _compact_spaces(value: str) -> str:
    return " ".join(value.strip().split())


def _truncate_at_word(value: str, max_length: int) -> str:
    value = _compact_spaces(value)
    if len(value) <= max_length:
        return value

    candidate = value[:max_length].rstrip()
    if " " in candidate:
        candidate = candidate.rsplit(" ", 1)[0].rstrip()
    return candidate[:max_length].rstrip(" -–—,.;:")


def _title_token(token: str) -> str:
    folded = _ascii_fold(token).upper()
    if folded in _ACRONYMS:
        return folded
    if len(token) <= 3 and token.isupper():
        return token
    return token[:1].upper() + token[1:].lower()


def generate_commercial_project_title(
    problem_statement: str | None,
    *,
    language: str | None = "es",
    max_length: int = DEFAULT_MAX_PROJECT_TITLE_LENGTH,
) -> str:
    """Return a short, commercial project title for generated LAB projects.

    The function is deterministic by design: it avoids LLM calls, keeps manual
    title behavior separate, and guarantees a non-empty title no longer than
    ``max_length``.
    """

    lang = _normalize_language(language)
    safe_max = max(8, int(max_length or DEFAULT_MAX_PROJECT_TITLE_LENGTH))
    fallback = _FALLBACK_TITLES[lang]
    text = _compact_spaces(str(problem_statement or ""))
    if not text:
        return _truncate_at_word(fallback, safe_max)

    folded = _ascii_fold(text)
    folded_tokens = set(re.findall(r"[a-z0-9]+", folded))
    for keyword_groups, titles in _TITLE_RULES:
        if all(
            any((keyword in folded if "-" in keyword or " " in keyword else keyword in folded_tokens) for keyword in keywords)
            for keywords in keyword_groups
        ):
            return _truncate_at_word(titles[lang], safe_max)

    cleaned = text
    for pattern in _FILLER_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    raw_tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñÀ-ÿ0-9]+", cleaned)
    meaningful: list[str] = []
    for token in raw_tokens:
        folded_token = _ascii_fold(token)
        if folded_token in _STOPWORDS or folded_token.isdigit() or len(folded_token) <= 2:
            continue
        meaningful.append(_title_token(token))
        if len(meaningful) >= 3:
            break

    if not meaningful:
        return _truncate_at_word(fallback, safe_max)

    candidate = _compact_spaces(" ".join(meaningful))
    suffix = "AI" if lang == "en" else "IA"
    if suffix not in candidate.split() and len(candidate) + len(suffix) + 1 <= safe_max:
        candidate = f"{candidate} {suffix}"

    return _truncate_at_word(candidate, safe_max) or _truncate_at_word(fallback, safe_max)
