from __future__ import annotations

from dataclasses import dataclass

from app.models import ApprovedToolsDigest, KnowledgeProfile, MemoryProfile, MemoryToolDependency

SCHEDULE_KEYWORDS = (
    "hour",
    "hourly",
    "day",
    "daily",
    "week",
    "weekly",
    "month",
    "monthly",
    "cron",
    "diar",
    "semanal",
    "mensual",
)
APPROVAL_KEYWORDS = ("approval", "aprob", "human")
ORDERED_KNOWLEDGE_TOOL_KEYS = (
    "knowledge_retrieval",
    "document_ingestion",
    "scheduler",
    "approval_gate",
)


@dataclass(frozen=True)
class KnowledgeToolPolicyDecision:
    tool_key: str
    required: bool
    reason: str
    capabilities: tuple[str, ...]
    optional_status: str = "optional"


def _contains_keywords(*parts: str | None, keywords: tuple[str, ...]) -> bool:
    haystack = " ".join(str(part or "") for part in parts).lower()
    return any(keyword in haystack for keyword in keywords)


def build_knowledge_tool_policy(
    *,
    knowledge_profile: KnowledgeProfile,
    memory_profile: MemoryProfile | None = None,
) -> dict[str, KnowledgeToolPolicyDecision]:
    memory_profile = memory_profile or MemoryProfile()
    knowledge_mode = knowledge_profile.mode.strip().lower()
    has_sources = bool(knowledge_profile.sources)
    rag_enabled = knowledge_mode == "rag"
    scheduler_required = rag_enabled and _contains_keywords(
        knowledge_profile.refresh_policy.frequency,
        " ".join(knowledge_profile.refresh_policy.triggers),
        keywords=SCHEDULE_KEYWORDS,
    )
    approval_required = _contains_keywords(
        memory_profile.write_policy,
        memory_profile.retention_policy,
        " ".join(memory_profile.sensitivity_rules),
        " ".join(knowledge_profile.sensitivity_rules),
        keywords=APPROVAL_KEYWORDS,
    )

    return {
        "knowledge_retrieval": KnowledgeToolPolicyDecision(
            tool_key="knowledge_retrieval",
            required=rag_enabled or has_sources,
            reason=(
                "RAG requiere retrieval gobernado y citas trazables."
                if rag_enabled or has_sources
                else "No se requiere retrieval documental para la estrategia actual."
            ),
            capabilities=("grounded_answers", "citations", "source_filters"),
            optional_status="not_needed",
        ),
        "document_ingestion": KnowledgeToolPolicyDecision(
            tool_key="document_ingestion",
            required=rag_enabled or has_sources,
            reason=(
                "Las fuentes documentales aprobadas necesitan pipeline de ingesta y refresh."
                if rag_enabled and has_sources
                else (
                    "RAG requiere una capacidad de ingesta aunque el corpus exacto quede como decision diferida."
                    if rag_enabled
                    else "Existen fuentes aprobadas; se necesita ingesta para mantener lineage y refresh."
                    if has_sources
                    else "No hace falta pipeline de ingesta dedicado para esta estrategia."
                )
            ),
            capabilities=("chunking", "refresh", "lineage"),
        ),
        "scheduler": KnowledgeToolPolicyDecision(
            tool_key="scheduler",
            required=scheduler_required,
            reason=(
                "El refresh de conocimiento requiere triggers programados para mantener frescura y lineage."
                if scheduler_required
                else "Permite refresh programado del conocimiento cuando la politica lo exija."
            ),
            capabilities=("refresh_trigger", "scheduled_rebuild"),
        ),
        "approval_gate": KnowledgeToolPolicyDecision(
            tool_key="approval_gate",
            required=approval_required,
            reason=(
                "La estrategia de memoria exige aprobacion explicita para retencion, sensibilidad o excepciones."
                if approval_required
                else "No se detectan politicas de memoria que exijan aprobacion explicita adicional."
            ),
            capabilities=("human_review", "retention_exception", "sensitive_release"),
        ),
    }


def build_memory_tool_dependencies(
    *,
    approved_tools_digest: ApprovedToolsDigest | None,
    knowledge_profile: KnowledgeProfile,
    memory_profile: MemoryProfile,
) -> list[MemoryToolDependency]:
    approved_keys = (
        {item.strip().lower() for item in approved_tools_digest.approved_tool_keys}
        if approved_tools_digest is not None
        else set()
    )
    policy = build_knowledge_tool_policy(
        knowledge_profile=knowledge_profile,
        memory_profile=memory_profile,
    )
    dependencies: list[MemoryToolDependency] = []
    for tool_key in ORDERED_KNOWLEDGE_TOOL_KEYS:
        decision = policy[tool_key]
        status = (
            "approved"
            if tool_key in approved_keys
            else "missing"
            if decision.required
            else decision.optional_status
        )
        dependencies.append(
            MemoryToolDependency(
                tool_key=tool_key,
                required=decision.required,
                status=status,
                reason=decision.reason,
                capabilities=list(decision.capabilities),
            )
        )
    return dependencies
