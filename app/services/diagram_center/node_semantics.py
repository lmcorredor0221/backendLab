from __future__ import annotations

import re
from typing import Any
import unicodedata


def normalize_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", text).strip()


def normalize_kind(value: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", normalize_label(value)).strip("_")


def _metadata_value(node: Any, key: str) -> object:
    metadata = getattr(node, "metadata", {}) if node is not None else {}
    if isinstance(metadata, dict):
        return metadata.get(key) or ""
    if hasattr(metadata, "model_dump"):
        value = metadata.model_dump(mode="json")
        if isinstance(value, dict):
            return value.get(key) or ""
    return ""


def node_memory_kind(node: Any) -> str:
    return normalize_kind(getattr(node, "memory_kind", "") or _metadata_value(node, "memory_kind"))


def node_tool_kind(node: Any) -> str:
    return normalize_kind(getattr(node, "tool_kind", "") or _metadata_value(node, "tool_kind"))


def is_storage_node(
    node: Any,
    *,
    kind: str | None = None,
    label: str | None = None,
    memory_kind: str | None = None,
) -> bool:
    normalized_kind = kind if kind is not None else normalize_kind(getattr(node, "kind", ""))
    normalized_label = label if label is not None else normalize_label(getattr(node, "label", ""))
    normalized_memory_kind = memory_kind if memory_kind is not None else node_memory_kind(node)

    if normalized_memory_kind in {"vector_store", "short_term_buffer", "working_memory", "shared_state"}:
        return True
    if any(token in normalized_kind for token in ("store", "database", "data_store", "db", "log", "audit", "ledger", "memory", "rag", "vector")):
        return True
    return any(
        phrase in normalized_label
        for phrase in (
            "decision log",
            "checkpoint log",
            "approval audit log",
            "audit log",
            "observation log",
            "log de",
            "registro de",
            "bitacora",
            "memoria",
            "memory",
            "rag",
            "knowledge base",
            "base de conocimiento",
            "vector",
        )
    )


def is_decision_gate_node(
    node: Any,
    *,
    kind: str | None = None,
    label: str | None = None,
    tool_kind: str | None = None,
    memory_kind: str | None = None,
) -> bool:
    normalized_kind = kind if kind is not None else normalize_kind(getattr(node, "kind", ""))
    normalized_label = label if label is not None else normalize_label(getattr(node, "label", ""))
    normalized_tool_kind = tool_kind if tool_kind is not None else node_tool_kind(node)
    normalized_memory_kind = memory_kind if memory_kind is not None else node_memory_kind(node)

    if is_storage_node(node, kind=normalized_kind, label=normalized_label, memory_kind=normalized_memory_kind):
        return False
    if normalized_tool_kind in {"approval_gate", "guardrail_gate", "human_gate"}:
        return True
    if any(
        token in normalized_kind
        for token in ("decision", "gateway", "gate", "approval", "hitl", "question", "condition", "choice", "branch", "checkpoint")
    ):
        return True
    return any(
        phrase in normalized_label
        for phrase in (
            "approval gate",
            "gate de aprobacion",
            "aprobacion humana",
            "aprobador humano",
            "requiere aprobacion",
            "guardrail",
            "hitl",
            "human in the loop",
            "human-in-the-loop",
            "pregunta",
            "question",
            "checkpoint",
            "validar si",
            "condicion",
            "decision",
            "decision gate",
        )
    ) or "?" in normalized_label
