from __future__ import annotations

from typing import Any


def evaluate_tools_memory_compatibility(tools: Any, memory: Any) -> tuple[list[str], bool, str]:
    """Check the explicit RAG/tool contract before Memory can be promoted."""
    if tools is None or memory is None:
        return ["Tools y Memory deben existir antes de evaluar compatibilidad."], True, "Falta una propuesta transversal."

    approved_digest = getattr(tools, "approved_tools_digest", None)
    approved_keys = set(getattr(approved_digest, "approved_tool_keys", []) or [])
    knowledge_keys = set(getattr(approved_digest, "knowledge_tool_keys", []) or [])
    knowledge = getattr(memory, "knowledge_design", None)
    rag_required = bool(getattr(knowledge, "rag_required", False)) or str(getattr(knowledge, "mode", "")).lower() in {"rag", "hybrid"}
    required = {"document_ingestion", "knowledge_retrieval"} if rag_required else set()
    available = approved_keys | knowledge_keys
    missing = sorted(required - available)
    if missing:
        return (
            [
                "Memory propone RAG, pero Tools no tiene aprobadas las capacidades "
                + ", ".join(missing)
                + ". Regenera Tools o simplifica Memory antes de aprobar.",
            ],
            True,
            "La estrategia RAG no tiene un contrato de ingestion y retrieval completo.",
        )
    if rag_required:
        return [], False, "RAG tiene ingestion, retrieval y lineage representados en Tools."
    return [], False, "Memory no requiere una dependencia RAG adicional."
