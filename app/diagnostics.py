from __future__ import annotations

import unicodedata
from typing import Final


AUTONOMY_LOW: Final = "low"
AUTONOMY_MEDIUM: Final = "medium"
AUTONOMY_HIGH: Final = "high"

CASE_TYPE_INFORMATION: Final = "informacion"
CASE_TYPE_AUTOMATION: Final = "automatizacion"
CASE_TYPE_COPILOT: Final = "copiloto"
CASE_TYPE_AUTONOMOUS_OPERATOR: Final = "operador_autonomo"
CASE_TYPE_MULTIAGENT_SYSTEM: Final = "sistema_multiagente"

CANONICAL_AUTONOMY_LEVELS: Final[tuple[str, ...]] = (
    AUTONOMY_LOW,
    AUTONOMY_MEDIUM,
    AUTONOMY_HIGH,
)

CANONICAL_CASE_TYPES: Final[tuple[str, ...]] = (
    CASE_TYPE_INFORMATION,
    CASE_TYPE_AUTOMATION,
    CASE_TYPE_COPILOT,
    CASE_TYPE_AUTONOMOUS_OPERATOR,
    CASE_TYPE_MULTIAGENT_SYSTEM,
)

LEGACY_AUTONOMY_LEVEL_MAP: Final[dict[str, str]] = {
    AUTONOMY_LOW: AUTONOMY_LOW,
    AUTONOMY_MEDIUM: AUTONOMY_MEDIUM,
    AUTONOMY_HIGH: AUTONOMY_HIGH,
    "assist": AUTONOMY_LOW,
    "assistant": AUTONOMY_LOW,
    "assisted": AUTONOMY_LOW,
    "copilot": AUTONOMY_MEDIUM,
    "supervised": AUTONOMY_MEDIUM,
    "autonomous": AUTONOMY_HIGH,
}

LEGACY_CASE_TYPE_MAP: Final[dict[str, str]] = {
    CASE_TYPE_INFORMATION: CASE_TYPE_INFORMATION,
    CASE_TYPE_AUTOMATION: CASE_TYPE_AUTOMATION,
    CASE_TYPE_COPILOT: CASE_TYPE_COPILOT,
    CASE_TYPE_AUTONOMOUS_OPERATOR: CASE_TYPE_AUTONOMOUS_OPERATOR,
    CASE_TYPE_MULTIAGENT_SYSTEM: CASE_TYPE_MULTIAGENT_SYSTEM,
    "workflow_automation": CASE_TYPE_AUTOMATION,
    "informational_assistant": CASE_TYPE_INFORMATION,
    "copilot": CASE_TYPE_COPILOT,
    "single_task_builder": CASE_TYPE_COPILOT,
    "autonomous_operator": CASE_TYPE_AUTONOMOUS_OPERATOR,
    "multi_agent_system": CASE_TYPE_MULTIAGENT_SYSTEM,
    "multiagent_system": CASE_TYPE_MULTIAGENT_SYSTEM,
}


def _normalize_identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    collapsed = "_".join(ascii_only.strip().lower().replace("-", " ").split())
    return collapsed


def normalize_autonomy_level(value: str, *, default: str = AUTONOMY_MEDIUM) -> str:
    normalized = _normalize_identifier(value)
    if not normalized:
        return default
    return LEGACY_AUTONOMY_LEVEL_MAP.get(normalized, default)


def normalize_case_type(value: str, *, default: str = "") -> str:
    normalized = _normalize_identifier(value)
    if not normalized:
        return default
    return LEGACY_CASE_TYPE_MAP.get(normalized, default)


def is_workflow_case(value: str) -> bool:
    normalized = normalize_case_type(value)
    return normalized in {CASE_TYPE_AUTOMATION, CASE_TYPE_AUTONOMOUS_OPERATOR}


def is_multiagent_case(value: str) -> bool:
    return normalize_case_type(value) == CASE_TYPE_MULTIAGENT_SYSTEM


def is_information_case(value: str) -> bool:
    return normalize_case_type(value) == CASE_TYPE_INFORMATION

