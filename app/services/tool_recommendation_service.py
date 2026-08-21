from __future__ import annotations

import hashlib
import json
import re
from uuid import UUID

from app.models import (
    ApprovedToolDigestEntry,
    ApprovedToolsDigest,
    BlueprintArtifact,
    BlueprintTool,
    CanvasArtifact,
    DiscoveryArtifact,
    DesignRecommendationArtifact,
    ReviewState,
    ToolDesignRoleCoverageEntry,
    ToolFamilyCandidate,
    ToolPreflightCapability,
    ToolRecommendationAllowedToolKey,
    ToolRecommendationArtifact,
    ToolRecommendationConfidence,
    ToolRecommendationContextDigest,
    ToolRecommendationEntry,
    ToolRecommendationEvaluation,
    ToolRecommendationFinding,
    ToolRecommendationGap,
    ToolRecommendationLLMOutput,
    ToolRecommendationPreflight,
    ToolRecommendationPromptInput,
    ToolRecommendationPromptToolOption,
    ToolRequirementCoverageEntry,
    ToolRecommendationReviewDecision,
    ToolRecommendationSourceStageVersions,
)
from app.services.knowledge_tool_policy import build_knowledge_tool_policy
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionOutput


CAPABILITY_CATALOG: dict[str, dict[str, str]] = {
    "read_system_of_record": {
        "label": "Lectura de sistema fuente",
        "family": "read_only_lookup",
        "tool_key": "read_system_of_record",
        "tool_label": "Lectura de sistema fuente",
        "capability_covered": "Consultar el sistema operativo que soporta el workflow principal.",
    },
    "approval_gate": {
        "label": "Gate de aprobacion humana",
        "family": "approval_control",
        "tool_key": "approval_gate",
        "tool_label": "Gate de aprobacion humana",
        "capability_covered": "Controlar decisiones no delegables y side effects antes de ejecutar acciones sensibles.",
    },
    "transactional_write": {
        "label": "Escritura transaccional acotada",
        "family": "transactional_write",
        "tool_key": "transactional_write",
        "tool_label": "Escritura transaccional acotada",
        "capability_covered": "Actualizar el sistema objetivo solo sobre acciones aprobadas y trazables.",
    },
    "knowledge_retrieval": {
        "label": "Recuperacion de conocimiento",
        "family": "retrieval",
        "tool_key": "knowledge_retrieval",
        "tool_label": "Recuperacion de conocimiento",
        "capability_covered": "Recuperar conocimiento institucional cuando el agente no puede operar con contexto inline.",
    },
    "document_ingestion": {
        "label": "Ingestion documental",
        "family": "document_ingestion",
        "tool_key": "document_ingestion",
        "tool_label": "Ingestion documental",
        "capability_covered": "Preparar y refrescar fuentes documentales para retrieval consistente.",
    },
    "outbound_notification": {
        "label": "Notificacion saliente",
        "family": "notification",
        "tool_key": "outbound_notification",
        "tool_label": "Notificacion saliente",
        "capability_covered": "Avisar al usuario o a un owner humano cuando el flujo requiera cierre o seguimiento.",
    },
    "human_handoff": {
        "label": "Handoff humano",
        "family": "human_handoff",
        "tool_key": "human_handoff",
        "tool_label": "Handoff humano",
        "capability_covered": "Escalar el caso a un owner humano cuando el flujo no puede cerrarse autonomamente.",
    },
    "scheduler": {
        "label": "Scheduler o trigger",
        "family": "scheduler",
        "tool_key": "scheduler",
        "tool_label": "Scheduler o trigger",
        "capability_covered": "Disparar ejecuciones por eventos o ventanas programadas cuando el flujo lo exige.",
    },
}

TOOL_FAMILY_CATALOG: dict[str, dict[str, object]] = {
    "read_only_lookup": {
        "label": "Consulta read-only a sistema fuente",
        "supported_capabilities": ["read_system_of_record"],
        "suggested_tool_keys": ["read_system_of_record"],
        "estimated_complexity": "medium",
    },
    "approval_control": {
        "label": "Control de aprobacion humana",
        "supported_capabilities": ["approval_gate"],
        "suggested_tool_keys": ["approval_gate"],
        "estimated_complexity": "low",
    },
    "transactional_write": {
        "label": "Escritura transaccional acotada",
        "supported_capabilities": ["transactional_write"],
        "suggested_tool_keys": ["transactional_write"],
        "estimated_complexity": "high",
    },
    "retrieval": {
        "label": "Retrieval de conocimiento",
        "supported_capabilities": ["knowledge_retrieval"],
        "suggested_tool_keys": ["knowledge_retrieval"],
        "estimated_complexity": "medium",
    },
    "document_ingestion": {
        "label": "Ingestion y refresh documental",
        "supported_capabilities": ["document_ingestion"],
        "suggested_tool_keys": ["document_ingestion"],
        "estimated_complexity": "medium",
    },
    "notification": {
        "label": "Notificacion saliente",
        "supported_capabilities": ["outbound_notification"],
        "suggested_tool_keys": ["outbound_notification"],
        "estimated_complexity": "low",
    },
    "human_handoff": {
        "label": "Escalamiento humano",
        "supported_capabilities": ["human_handoff"],
        "suggested_tool_keys": ["human_handoff"],
        "estimated_complexity": "low",
    },
    "scheduler": {
        "label": "Scheduler o trigger",
        "supported_capabilities": ["scheduler"],
        "suggested_tool_keys": ["scheduler"],
        "estimated_complexity": "medium",
    },
}

EXTERNAL_SOURCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "crm": ("crm", "salesforce", "hubspot"),
    "erp": ("erp", "sap", "oracle"),
    "ticketing": ("ticket", "incidente", "mesa de ayuda", "service desk", "zendesk", "jira"),
    "database": ("base de datos", "database", "sql", "postgres", "mysql"),
    "api": ("api", "webhook", "endpoint"),
    "portal": ("portal", "backoffice"),
    "email_inbox": ("correo", "email", "mailbox", "inbox"),
    "filesystem": ("archivo", "carpeta", "drive", "sharepoint", "documento"),
}

NOTIFICATION_CHANNEL_PATTERNS: dict[str, tuple[str, ...]] = {
    "email": ("notificar por email", "correo", "email", "mail"),
    "slack": ("slack",),
    "teams": ("teams", "microsoft teams"),
    "sms": ("sms", "mensaje de texto"),
    "portal": ("portal", "bandeja"),
}

WRITE_ACTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "update_record": ("actualizar", "update", "modificar", "registrar", "guardar"),
    "approve_request": ("aprobar", "approve", "autorizar"),
    "create_case": ("crear ticket", "create ticket", "crear caso", "abrir caso", "crear registro"),
    "send_notification": ("notificar", "alertar", "avisar", "enviar mensaje", "enviar correo"),
    "schedule_execution": ("programar", "schedule", "agendar", "calendarizar"),
}

DOCUMENT_SIGNAL_PATTERNS = ("manual", "politica", "faq", "guia", "procedimiento", "document", "knowledge base")
HANDOFF_PATTERNS = ("handoff", "escalar", "escalamiento", "transferir", "derivar")
SCHEDULE_PATTERNS = ("programar", "schedule", "agendar", "periodico", "cron", "diario", "semanal")
APPROVAL_PATTERNS = ("aprobacion", "approval", "aprobar", "autorizar", "revision humana")
GENERIC_SYSTEM_PATTERNS = ("sistema", "plataforma", "repositorio", "fuente de verdad")
KNOWLEDGE_TOOL_KEYS = {"knowledge_retrieval", "document_ingestion"}
NFR_OPERATIONAL_EFFICIENCY_PATTERNS = (
    "tiempo operativo",
    "tiempo de respuesta",
    "latencia",
    "throughput",
    "eficien",
    "productividad",
    "pasos manuales",
    "retrabajo",
    "sla",
)


def _compact_lines(items: list[str], fallback: str) -> str:
    normalized = [item.strip() for item in items if item.strip()]
    return "; ".join(normalized[:4]) if normalized else fallback


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def _combined_text(*parts: str) -> str:
    return " ".join(_normalize_text(part) for part in parts if _normalize_text(part)).lower()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _append_unique(target: list[str], value: str) -> None:
    token = _normalize_text(value)
    if token and token not in target:
        target.append(token)


def _contract_seed_for_key(blueprint: BlueprintArtifact, tool_key: str) -> BlueprintTool | None:
    for item in blueprint.tools:
        if item.name.strip().lower() == tool_key.strip().lower():
            return item
    return None


def _append_entry(target: list[ToolRecommendationEntry], entry: ToolRecommendationEntry) -> None:
    if any(item.tool_key == entry.tool_key for item in target):
        return
    target.append(entry)


def _append_gap(target: list[ToolRecommendationGap], gap: ToolRecommendationGap) -> None:
    for item in target:
        if item.gap_key != gap.gap_key:
            continue
        if not item.suggested_answer and gap.suggested_answer:
            item.suggested_answer = gap.suggested_answer
        if not item.answer_options and gap.answer_options:
            item.answer_options = list(gap.answer_options)
        return
    target.append(gap)


def _append_finding(target: list[ToolRecommendationFinding], finding: ToolRecommendationFinding) -> None:
    if any(item.finding_key == finding.finding_key for item in target):
        return
    target.append(finding)


def _merge_unique(values: list[str], additions: list[str]) -> list[str]:
    merged = list(values)
    for item in additions:
        _append_unique(merged, item)
    return merged


def _coverage_tool_matches(text: str, *, allowed_tool_keys: set[str]) -> list[str]:
    matches: list[str] = []

    def register(tool_key: str, *patterns: str) -> None:
        if tool_key not in allowed_tool_keys:
            return
        if any(pattern in text for pattern in patterns):
            _append_unique(matches, tool_key)

    register(
        "read_system_of_record",
        "crm",
        "erp",
        "ticket",
        "service desk",
        "sistema",
        "fuente",
        "consulta",
        "leer",
        "dato",
        "registro",
        "repositorio",
        "api",
    )
    register(
        "approval_gate",
        "aprob",
        "approval",
        "autoriz",
        "decision no delegable",
        "review humana",
        "humano",
    )
    register(
        "transactional_write",
        "actualiz",
        "modific",
        "guardar",
        "registr",
        "crear",
        "escrit",
        "muta",
        "aprobar solicitud",
    )
    register(
        "knowledge_retrieval",
        "document",
        "manual",
        "politica",
        "faq",
        "knowledge",
        "conocimiento",
        "grounding",
        "buscar",
        "recuper",
    )
    register(
        "document_ingestion",
        "ingest",
        "index",
        "chunk",
        "embedding",
        "refresc",
        "cargar documento",
        "cargar fuente",
    )
    register(
        "outbound_notification",
        "notific",
        "correo",
        "email",
        "slack",
        "teams",
        "alert",
        "avis",
        "seguimiento",
    )
    register(
        "human_handoff",
        "handoff",
        "escal",
        "deriv",
        "transfer",
        "owner humano",
        "fallback humano",
    )
    register(
        "scheduler",
        "program",
        "schedule",
        "cron",
        "periodic",
        "diario",
        "semanal",
        "trigger",
        "evento",
    )

    return matches


def _design_role_can_be_covered_internally(item: ToolDesignRoleCoverageEntry) -> bool:
    text = _combined_text(item.role_key, item.role_title, item.responsibility)
    return _contains_any(
        text,
        (
            "domain",
            "dominio",
            "especialista",
            "specialist",
            "subagent",
            "subagente",
            "supervisor",
            "orquest",
            "orchestr",
            "routing",
            "router",
            "handoff",
            "plan",
            "planner",
            "execut",
            "ejecut",
            "review",
            "revis",
            "clasif",
            "analis",
            "triage",
        ),
    )


def _requirement_can_be_covered_internally(item: ToolRequirementCoverageEntry) -> bool:
    text = _combined_text(item.requirement_key, item.requirement_title, item.rationale)
    return _contains_any(
        text,
        (
            "clasif",
            "classif",
            "priorid",
            "priority",
            "resumen",
            "summary",
            "extract",
            "extrac",
            "analis",
            "analy",
            "trazabil",
            "traceab",
            "explicab",
            "explan",
            "razonam",
            "reasoning",
            "disponib",
            "alucinac",
            "hallucin",
            "decision",
            "criterio",
            "regla",
            "rutina",
            "prompt",
            "modelo",
            "cognitiv",
            "rendimiento",
            "latencia",
            "tiempo",
            "performance",
            "disponibilidad",
            "sla",
            "seguridad",
            "privacidad",
            "compliance",
            "direccionamiento",
            "duplicad",
        ),
    )


def _infer_non_functional_coverage(text: str, *, allowed_tool_keys: set[str]) -> list[str]:
    inferred: list[str] = []
    if _contains_any(text, NFR_OPERATIONAL_EFFICIENCY_PATTERNS):
        for tool_key in (
            "read_system_of_record",
            "knowledge_retrieval",
            "transactional_write",
            "scheduler",
        ):
            if tool_key in allowed_tool_keys:
                _append_unique(inferred, tool_key)
    return inferred


def _coverage_status(covered_by_tool_keys: list[str], selected_tool_keys: set[str]) -> str:
    if not covered_by_tool_keys:
        return "gap"
    if "agent_core_reasoning" in covered_by_tool_keys or any(item in selected_tool_keys for item in covered_by_tool_keys):
        return "covered"
    return "partial"


def _coverage_rationale(
    *,
    covered_by_tool_keys: list[str],
    selected_tool_keys: set[str],
    subject: str,
) -> str:
    if not covered_by_tool_keys:
        return f"No se detecto una tool del shortlist que cubra explicitamente {subject}."
    if "agent_core_reasoning" in covered_by_tool_keys:
        return f"Cubierto de forma nativa por el razonamiento del modelo y logica cognitiva del agente para {subject}."
    selected_matches = [item for item in covered_by_tool_keys if item in selected_tool_keys]
    if selected_matches:
        return f"La cobertura se apoya en {', '.join(selected_matches)} para {subject}."
    return f"Existe cobertura potencial con {', '.join(covered_by_tool_keys)}, pero ninguna quedo seleccionada en el set minimo."


def _selected_design(design_artifact: DesignRecommendationArtifact | None):
    if design_artifact is None:
        return None
    if design_artifact.selected_design is not None:
        return design_artifact.selected_design
    for item in design_artifact.alternatives:
        if item.alternative_key == design_artifact.recommended_alternative_key:
            return item
    return design_artifact.alternatives[0] if design_artifact.alternatives else None


def _build_requirement_coverage(
    definition_artifact: RequirementsDefinitionOutput | None,
    *,
    allowed_tool_keys: set[str],
    selected_tool_keys: set[str],
) -> list[ToolRequirementCoverageEntry]:
    if definition_artifact is None:
        return []

    entries: list[ToolRequirementCoverageEntry] = []
    for item in definition_artifact.functional_requirements:
        text = _combined_text(item.title, item.requirement, item.actor, item.trigger, item.happy_path, " ".join(item.exceptions))
        covered_by = _coverage_tool_matches(text, allowed_tool_keys=allowed_tool_keys)
        if not covered_by and _requirement_can_be_covered_internally(
            ToolRequirementCoverageEntry(
                requirement_key=item.key,
                requirement_title=item.title or item.requirement,
                category="functional",
                priority=item.priority,
                coverage_status="covered",
                covered_by_tool_keys=["agent_core_reasoning"],
                rationale="",
            )
        ):
            covered_by = ["agent_core_reasoning"]
        entries.append(
            ToolRequirementCoverageEntry(
                requirement_key=item.key,
                requirement_title=item.title or item.requirement,
                category="functional",
                priority=item.priority,
                coverage_status=_coverage_status(covered_by, selected_tool_keys),
                covered_by_tool_keys=covered_by,
                rationale=_coverage_rationale(
                    covered_by_tool_keys=covered_by,
                    selected_tool_keys=selected_tool_keys,
                    subject=item.title or item.requirement or item.key,
                ),
                source_refs=list(item.source_refs),
            )
        )

    for item in definition_artifact.non_functional_requirements:
        text = _combined_text(item.title, item.requirement, item.category, item.metric, item.target)
        covered_by = _coverage_tool_matches(text, allowed_tool_keys=allowed_tool_keys)
        covered_by = _merge_unique(
            covered_by,
            _infer_non_functional_coverage(text, allowed_tool_keys=allowed_tool_keys),
        )
        if not covered_by and _requirement_can_be_covered_internally(
            ToolRequirementCoverageEntry(
                requirement_key=item.key,
                requirement_title=item.title or item.requirement,
                category="non_functional",
                priority=item.priority,
                coverage_status="covered",
                covered_by_tool_keys=["agent_core_reasoning"],
                rationale="",
            )
        ):
            covered_by = ["agent_core_reasoning"]
        entries.append(
            ToolRequirementCoverageEntry(
                requirement_key=item.key,
                requirement_title=item.title or item.requirement,
                category="non_functional",
                priority=item.priority,
                coverage_status=_coverage_status(covered_by, selected_tool_keys),
                covered_by_tool_keys=covered_by,
                rationale=_coverage_rationale(
                    covered_by_tool_keys=covered_by,
                    selected_tool_keys=selected_tool_keys,
                    subject=item.title or item.requirement or item.key,
                ),
                source_refs=list(item.source_refs),
            )
        )

    return entries


def _build_design_role_coverage(
    design_artifact: DesignRecommendationArtifact | None,
    *,
    allowed_tool_keys: set[str],
    selected_tool_keys: set[str],
) -> list[ToolDesignRoleCoverageEntry]:
    selected_design = _selected_design(design_artifact)
    if selected_design is None:
        return []

    entries: list[ToolDesignRoleCoverageEntry] = []
    for role in selected_design.roles:
        text = _combined_text(role.title, role.responsibility, " ".join(role.limits))
        covered_by = _coverage_tool_matches(text, allowed_tool_keys=allowed_tool_keys)
        entries.append(
            ToolDesignRoleCoverageEntry(
                role_key=role.key,
                role_title=role.title,
                responsibility=role.responsibility,
                coverage_status=_coverage_status(covered_by, selected_tool_keys),
                covered_by_tool_keys=covered_by,
                rationale=_coverage_rationale(
                    covered_by_tool_keys=covered_by,
                    selected_tool_keys=selected_tool_keys,
                    subject=role.title or role.key,
                ),
                source_refs=[role.key],
            )
        )

    return entries


def _register_capability(
    store: dict[str, ToolPreflightCapability],
    *,
    capability_key: str,
    required: bool,
    reason: str,
    source_evidence: list[str],
    confidence: float,
) -> None:
    catalog = CAPABILITY_CATALOG[capability_key]
    current = store.get(capability_key)
    if current is None:
        store[capability_key] = ToolPreflightCapability(
            capability_key=capability_key,
            label=catalog["label"],
            required=required,
            reason=reason,
            source_evidence=list(source_evidence),
            confidence=confidence,
        )
        return

    current.required = current.required or required
    current.confidence = max(current.confidence, confidence)
    current.source_evidence = _merge_unique(current.source_evidence, source_evidence)
    if required or not current.reason:
        current.reason = reason


def _register_family(
    store: dict[str, ToolFamilyCandidate],
    *,
    family_key: str,
    status: str,
    reason: str,
    matched_signals: list[str],
    rejected_by_constraints: list[str] | None = None,
) -> None:
    catalog = TOOL_FAMILY_CATALOG[family_key]
    current = store.get(family_key)
    priority = {"excluded": 0, "candidate": 1, "required": 2}
    rejected_by_constraints = rejected_by_constraints or []

    if current is None:
        store[family_key] = ToolFamilyCandidate(
            family_key=family_key,
            label=str(catalog["label"]),
            status=status,
            supported_capabilities=list(catalog["supported_capabilities"]),
            matched_signals=list(matched_signals),
            rejected_by_constraints=list(rejected_by_constraints),
            suggested_tool_keys=list(catalog["suggested_tool_keys"]),
            estimated_complexity=str(catalog["estimated_complexity"]),
            reason=reason,
        )
        return

    current.supported_capabilities = _merge_unique(
        current.supported_capabilities,
        [str(item) for item in catalog["supported_capabilities"]],
    )
    current.matched_signals = _merge_unique(current.matched_signals, matched_signals)
    current.rejected_by_constraints = _merge_unique(current.rejected_by_constraints, rejected_by_constraints)
    current.suggested_tool_keys = _merge_unique(
        current.suggested_tool_keys,
        [str(item) for item in catalog["suggested_tool_keys"]],
    )
    if priority.get(status, 0) > priority.get(current.status, 0):
        current.status = status
        current.reason = reason
    elif not current.reason:
        current.reason = reason


def _extract_core_workflows(blueprint: BlueprintArtifact, canvas: CanvasArtifact, discovery: DiscoveryArtifact) -> list[str]:
    workflows = [_normalize_text(step.name) for step in blueprint.delivery_package.workflow_profile.steps if step.name.strip()]
    if workflows:
        return workflows[:4]

    if canvas.mvp_scope:
        return [_normalize_text(item) for item in canvas.mvp_scope[:4] if _normalize_text(item)]

    fallback = _normalize_text(canvas.user_goal or discovery.desired_outcome or discovery.problem_statement)
    return [fallback] if fallback else []


def _extract_external_sources(text: str) -> list[str]:
    sources: list[str] = []
    for label, patterns in EXTERNAL_SOURCE_PATTERNS.items():
        if _contains_any(text, patterns):
            sources.append(label)
    return sources


def _extract_notification_channels(text: str) -> list[str]:
    channels: list[str] = []
    for label, patterns in NOTIFICATION_CHANNEL_PATTERNS.items():
        if _contains_any(text, patterns):
            channels.append(label)
    return channels


def _extract_write_actions(text: str) -> list[str]:
    actions: list[str] = []
    for label, patterns in WRITE_ACTION_PATTERNS.items():
        if any(
            pattern in text and f"sin {pattern}" not in text and f"no {pattern}" not in text
            for pattern in patterns
        ):
            actions.append(label)
    return actions


def _extract_interaction_modes(
    *,
    text: str,
    has_knowledge_signal: bool,
    has_external_system_signal: bool,
    notification_channels: list[str],
    needs_human_gate: bool,
    has_schedule_signal: bool,
) -> list[str]:
    interaction_modes = ["conversational"]
    if has_knowledge_signal:
        interaction_modes.append("document_grounded")
    if has_external_system_signal:
        interaction_modes.append("system_lookup")
    if notification_channels:
        interaction_modes.append("async_notification")
    if needs_human_gate or _contains_any(text, APPROVAL_PATTERNS):
        interaction_modes.append("approval_review")
    if has_schedule_signal:
        interaction_modes.append("scheduled_or_event_driven")
    return interaction_modes


def _extract_hard_constraints(discovery: DiscoveryArtifact, blueprint: BlueprintArtifact) -> list[str]:
    text = _combined_text(
        " ".join(discovery.constraints),
        " ".join(discovery.mvp_definition.out_of_scope),
        " ".join(blueprint.guardrails),
        blueprint.architecture,
    )
    hard_constraints: list[str] = []
    if _contains_any(text, ("sin microservicios", "no microservicios", "mvp simple", "mantener un mvp simple")):
        hard_constraints.append("mvp_simple_no_microservices")
    if _contains_any(
        text,
        ("sin side effects irreversibles", "no ejecutar side effects irreversibles", "irreversible", "solo lectura"),
    ):
        hard_constraints.append("no_irreversible_side_effects")
    if _contains_any(text, ("datos sensibles", "privacidad", "pii", "confidencial", "workspace aislado")):
        hard_constraints.append("privacy_restricted")
    if _contains_any(text, APPROVAL_PATTERNS) or discovery.mvp_definition.non_delegable_decisions:
        hard_constraints.append("human_approval_required")
    return hard_constraints


def _classify_case(
    *,
    blueprint: BlueprintArtifact,
    has_external_system_signal: bool,
    has_knowledge_signal: bool,
    has_write_actions: bool,
    has_notification_signal: bool,
    needs_human_gate: bool,
    has_schedule_signal: bool,
) -> str:
    architecture = blueprint.architecture.strip().lower()
    if has_knowledge_signal and not has_external_system_signal and not has_write_actions:
        return "knowledge_assistant"
    if has_write_actions and needs_human_gate:
        return "approval_gated_operator"
    if has_external_system_signal and not has_write_actions:
        return "enterprise_copilot"
    if has_notification_signal and not has_write_actions:
        return "notification_coordinator"
    if architecture in {"router_parallel", "supervisor_with_subagents", "handoffs"} or has_schedule_signal:
        return "workflow_orchestrator"
    return "lean_blueprint_builder"


def _build_entry(
    *,
    blueprint: BlueprintArtifact,
    capability_key: str,
    classification: str,
    decision_reason: str,
    source_evidence: list[str],
    confidence: float,
    dependencies: list[str] | None = None,
    incompatibilities: list[str] | None = None,
    redundant_with: list[str] | None = None,
) -> ToolRecommendationEntry:
    catalog = CAPABILITY_CATALOG[capability_key]
    contract_seed = None
    if capability_key == "approval_gate":
        contract_seed = _contract_seed_for_key(blueprint, "promote_blueprint_for_implementation")
    return ToolRecommendationEntry(
        tool_key=catalog["tool_key"],
        tool_label=catalog["tool_label"],
        classification=classification,
        capability_covered=catalog["capability_covered"],
        decision_reason=decision_reason,
        source_evidence=source_evidence,
        dependencies=dependencies or [],
        incompatibilities=incompatibilities or [],
        redundant_with=redundant_with or [],
        confidence=confidence,
        contract_seed=contract_seed,
    )


def _ordered_capabilities(store: dict[str, ToolPreflightCapability]) -> list[ToolPreflightCapability]:
    return [store[key] for key in CAPABILITY_CATALOG if key in store]


def _ordered_families(store: dict[str, ToolFamilyCandidate]) -> list[ToolFamilyCandidate]:
    return [store[key] for key in TOOL_FAMILY_CATALOG if key in store]


def build_placeholder_tool_recommendation(
    *,
    session_id: UUID,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    blueprint: BlueprintArtifact,
    definition_artifact: RequirementsDefinitionOutput | None = None,
    design_artifact: DesignRecommendationArtifact | None = None,
    instructions: str = "",
    blueprint_version_number: int | None,
) -> ToolRecommendationArtifact:
    recommended_tools: list[ToolRecommendationEntry] = []
    optional_tools: list[ToolRecommendationEntry] = []
    rejected_tools: list[ToolRecommendationEntry] = []
    coverage_gaps: list[ToolRecommendationGap] = []
    needs_information: list[ToolRecommendationGap] = []
    mandatory_capabilities: dict[str, ToolPreflightCapability] = {}
    candidate_families: dict[str, ToolFamilyCandidate] = {}
    forbidden_capabilities: list[str] = []

    normalized_sources = [
        "discovery.problem_statement",
        "discovery.current_process",
        "discovery.constraints",
        "canvas.user_goal",
        "canvas.agent_profile",
        "define.functional_requirements",
        "define.non_functional_requirements",
        "design.selected_alternative",
        "blueprint.architecture",
        "blueprint.reasoning_pattern",
        "blueprint.guardrails",
    ]

    selected_design = _selected_design(design_artifact)
    definition_text = ""
    if definition_artifact is not None:
        definition_text = _combined_text(
            definition_artifact.summary,
            " ".join(item.requirement for item in definition_artifact.functional_requirements),
            " ".join(item.requirement for item in definition_artifact.non_functional_requirements),
            " ".join(item.rule for item in definition_artifact.business_rules),
            " ".join(item.dependency for item in definition_artifact.dependencies),
        )
    design_text = ""
    if selected_design is not None:
        design_text = _combined_text(
            selected_design.architecture,
            selected_design.reasoning_pattern,
            selected_design.coordination_model,
            selected_design.summary,
            selected_design.topology,
            " ".join(role.title for role in selected_design.roles),
            " ".join(role.responsibility for role in selected_design.roles),
            " ".join(item.trigger for item in selected_design.handoffs),
            " ".join(selected_design.approval_points),
        )
    instructions_text = _normalize_text(instructions)
    if instructions_text:
        normalized_sources.append("session.tools_instructions")

    business_text = _combined_text(
        discovery.problem_statement,
        discovery.current_process,
        discovery.desired_outcome,
        canvas.user_goal,
        " ".join(discovery.constraints),
        " ".join(discovery.mvp_definition.v1_scope),
        " ".join(discovery.mvp_definition.non_delegable_decisions),
        " ".join(canvas.agent_profile.expected_outputs),
        " ".join(canvas.agent_profile.human_approvals),
        blueprint.architecture,
        blueprint.reasoning_pattern,
        blueprint.memory_strategy,
        " ".join(blueprint.guardrails),
        blueprint.narrative,
        blueprint.knowledge_profile.mode,
        " ".join(source.title for source in blueprint.knowledge_profile.sources),
        " ".join(step.name for step in blueprint.delivery_package.workflow_profile.steps),
        " ".join(step.objective for step in blueprint.delivery_package.workflow_profile.steps),
        definition_text,
        design_text,
        instructions_text,
    )

    external_sources = _extract_external_sources(business_text)
    notification_channels = _extract_notification_channels(business_text)
    required_write_actions = _extract_write_actions(business_text)
    transactional_write_actions = [
        action for action in required_write_actions if action not in {"send_notification", "schedule_execution"}
    ]
    no_core_system_touch = _contains_any(
        business_text,
        ("sin tocar sistemas", "sin tocar sistemas core", "sin tocar sistema", "no tocar sistemas"),
    )
    hard_constraints = _extract_hard_constraints(discovery, blueprint)
    has_schedule_signal = _contains_any(business_text, SCHEDULE_PATTERNS)
    has_document_signal = _contains_any(business_text, DOCUMENT_SIGNAL_PATTERNS)
    has_handoff_signal = _contains_any(business_text, HANDOFF_PATTERNS)
    has_external_system_signal = bool(external_sources)
    has_write_actions = bool(transactional_write_actions)
    has_notification_signal = bool(notification_channels) or _contains_any(
        business_text,
        ("notificar", "alerta", "avisar", "seguimiento"),
    )
    has_knowledge_signal = (
        blueprint.knowledge_profile.mode.strip().lower() == "rag"
        or bool(blueprint.knowledge_profile.sources)
        or has_document_signal
    )
    knowledge_tool_policy = build_knowledge_tool_policy(
        knowledge_profile=blueprint.knowledge_profile,
        memory_profile=blueprint.memory_profile,
    )
    needs_human_gate = bool(
        discovery.mvp_definition.non_delegable_decisions
        or canvas.agent_profile.human_approvals
        or any(tool.requires_approval or tool.has_side_effects for tool in blueprint.tools)
        or discovery.autonomy_level.strip().lower() == "high"
        or has_write_actions
    )

    case_classification = _classify_case(
        blueprint=blueprint,
        has_external_system_signal=has_external_system_signal,
        has_knowledge_signal=has_knowledge_signal,
        has_write_actions=has_write_actions,
        has_notification_signal=has_notification_signal,
        needs_human_gate=needs_human_gate,
        has_schedule_signal=has_schedule_signal,
    )

    required_information_sources: list[str] = []
    for source in external_sources:
        _append_unique(required_information_sources, source)
    if blueprint.knowledge_profile.mode.strip().lower() == "rag" or blueprint.knowledge_profile.sources:
        _append_unique(required_information_sources, "knowledge_base")

    approval_boundaries = [
        _normalize_text(item)
        for item in [*canvas.agent_profile.human_approvals, *discovery.mvp_definition.non_delegable_decisions]
        if _normalize_text(item)
    ][:4]

    if has_external_system_signal:
        _register_capability(
            mandatory_capabilities,
            capability_key="read_system_of_record",
            required=True,
            reason="El workflow depende de datos operativos externos y requiere grounding read-only antes de actuar.",
            source_evidence=[
                "discovery.current_process",
                "discovery.desired_outcome",
                "canvas.user_goal",
            ],
            confidence=0.8,
        )
        _register_family(
            candidate_families,
            family_key="read_only_lookup",
            status="required",
            reason="Se detectaron sistemas fuente necesarios para completar el workflow principal.",
            matched_signals=external_sources,
        )
    elif has_write_actions or (_contains_any(business_text, GENERIC_SYSTEM_PATTERNS) and not no_core_system_touch):
        _register_capability(
            mandatory_capabilities,
            capability_key="read_system_of_record",
            required=True,
            reason="El workflow requiere consultar y validar datos operativos del proceso antes de ejecutar acciones.",
            source_evidence=[
                "discovery.current_process",
                "discovery.desired_outcome",
            ],
            confidence=0.8,
        )
        _register_family(
            candidate_families,
            family_key="read_only_lookup",
            status="required",
            reason="Inferencia proactiva de consulta al sistema fuente para alimentar el flujo de ejecucion.",
            matched_signals=["operational_system_of_record"],
        )

    approval_gate_policy = knowledge_tool_policy["approval_gate"]
    if needs_human_gate or approval_gate_policy.required:
        _register_capability(
            mandatory_capabilities,
            capability_key="approval_gate",
            required=True,
            reason=(
                "Existen decisiones no delegables, autonomia alta o side effects que requieren control humano visible."
                if needs_human_gate
                else approval_gate_policy.reason
            ),
            source_evidence=[
                "discovery.mvp_definition.non_delegable_decisions",
                "canvas.agent_profile.human_approvals",
                "blueprint.guardrails",
                "blueprint.memory_profile",
            ],
            confidence=0.88,
        )
        _register_family(
            candidate_families,
            family_key="approval_control",
            status="required",
            reason=(
                "El caso necesita un gate de aprobacion antes de promover cambios sensibles."
                if needs_human_gate
                else approval_gate_policy.reason
            ),
            matched_signals=["human_approval_required"] if needs_human_gate else ["knowledge_governance"],
        )

    if has_write_actions and has_external_system_signal:
        _register_capability(
            mandatory_capabilities,
            capability_key="transactional_write",
            required=True,
            reason="El objetivo describe acciones concretas de actualizacion o aprobacion sobre un sistema objetivo.",
            source_evidence=[
                "discovery.current_process",
                "discovery.desired_outcome",
                "blueprint.delivery_package.workflow_profile",
            ],
            confidence=0.73,
        )
        _register_family(
            candidate_families,
            family_key="transactional_write",
            status="required",
            reason="Hay acciones operativas que no pueden resolverse solo con lectura o respuesta conversacional.",
            matched_signals=transactional_write_actions,
            rejected_by_constraints=[],
        )
    elif has_write_actions:
        _register_family(
            candidate_families,
            family_key="transactional_write",
            status="excluded",
            reason="La escritura transaccional queda bloqueada hasta identificar el sistema objetivo y su boundary operativo.",
            matched_signals=transactional_write_actions,
            rejected_by_constraints=["system_of_record_unspecified"],
        )
    elif "no_irreversible_side_effects" in hard_constraints:
        _append_unique(forbidden_capabilities, "transactional_write")
        _register_family(
            candidate_families,
            family_key="transactional_write",
            status="excluded",
            reason="La heuristica descarta escritura amplia mientras no exista una accion aprobada y acotada que la justifique.",
            matched_signals=["minimal_scope"],
            rejected_by_constraints=["no_irreversible_side_effects"],
        )

    if has_knowledge_signal:
        retrieval_policy = knowledge_tool_policy["knowledge_retrieval"]
        _register_capability(
            mandatory_capabilities,
            capability_key="knowledge_retrieval",
            required=retrieval_policy.required,
            reason=retrieval_policy.reason,
            source_evidence=[
                "blueprint.knowledge_profile",
                "canvas.agent_profile.expected_outputs",
                "discovery.problem_statement",
            ],
            confidence=0.78 if retrieval_policy.required else 0.6,
        )
        _register_family(
            candidate_families,
            family_key="retrieval",
            status="required" if retrieval_policy.required else "candidate",
            reason=(
                "Se detecto dependencia de conocimiento institucional para completar respuestas o recomendaciones."
                if retrieval_policy.required
                else retrieval_policy.reason
            ),
            matched_signals=["knowledge_base" if retrieval_policy.required else "document_signal"],
        )

        document_ingestion_policy = knowledge_tool_policy["document_ingestion"]
        if document_ingestion_policy.required:
            _register_capability(
                mandatory_capabilities,
                capability_key="document_ingestion",
                required=True,
                reason=document_ingestion_policy.reason,
                source_evidence=[
                    "blueprint.knowledge_profile.sources",
                    "blueprint.knowledge_profile.refresh_policy",
                ],
                confidence=0.76,
            )
            _register_family(
                candidate_families,
                family_key="document_ingestion",
                status="required",
                reason=document_ingestion_policy.reason,
                matched_signals=["rag_mode", "approved_sources"],
            )
        elif blueprint.knowledge_profile.mode.strip().lower() == "rag":
            _register_family(
                candidate_families,
                family_key="document_ingestion",
                status="candidate",
                reason=document_ingestion_policy.reason,
                matched_signals=["rag_mode"],
            )
        elif has_document_signal and not blueprint.knowledge_profile.sources:
            gap = ToolRecommendationGap(
                gap_key="knowledge_source_unspecified",
                title="Fuentes documentales por precisar",
                question="Que documentos, politicas o repositorios deben usarse como grounding del agente?",
                reason="Hay dependencia documental, pero no existe una taxonomia de fuentes aprobadas.",
                impact="No se puede cerrar con confianza el nivel minimo de retrieval ni su estrategia de refresh.",
                severity="medium",
            )
            _append_gap(needs_information, gap)

    scheduler_policy = knowledge_tool_policy["scheduler"]
    if scheduler_policy.required:
        _register_capability(
            mandatory_capabilities,
            capability_key="scheduler",
            required=True,
            reason=scheduler_policy.reason,
            source_evidence=[
                "blueprint.knowledge_profile.refresh_policy",
                "blueprint.knowledge_profile.sources",
            ],
            confidence=0.71,
        )
        _register_family(
            candidate_families,
            family_key="scheduler",
            status="required",
            reason=scheduler_policy.reason,
            matched_signals=["knowledge_refresh_policy"],
        )

    if has_notification_signal:
        notification_required = "send_notification" in required_write_actions or "seguimiento" in business_text
        _register_family(
            candidate_families,
            family_key="notification",
            status="required" if notification_required else "candidate",
            reason="El workflow menciona avisos o cierres fuera del hilo principal y conviene modelar el canal desde esta etapa.",
            matched_signals=notification_channels or ["notification_signal"],
        )
        if notification_required:
            _register_capability(
                mandatory_capabilities,
                capability_key="outbound_notification",
                required=True,
                reason="El flujo necesita avisar o cerrar el ciclo con el usuario u otro owner.",
                source_evidence=[
                    "discovery.current_process",
                    "discovery.desired_outcome",
                ],
                confidence=0.69,
            )
        if notification_required and not notification_channels:
            gap = ToolRecommendationGap(
                gap_key="notification_channel_unspecified",
                title="Canal de notificacion no definido",
                question="Cual es el canal oficial para notificaciones del agente: email, Slack, Teams o portal?",
                reason="Se detecta necesidad de avisos, pero no un canal gobernado.",
                impact="La recomendacion no puede fijar una tool de notificacion sin aumentar riesgo operativo.",
                severity="medium",
            )
            _append_gap(needs_information, gap)

    if has_handoff_signal or approval_boundaries:
        _register_family(
            candidate_families,
            family_key="human_handoff",
            status="candidate",
            reason="El proceso muestra checkpoints o escalamiento a owner humano como fallback operativo.",
            matched_signals=["handoff" if has_handoff_signal else "approval_boundary"],
        )

    if has_schedule_signal and not scheduler_policy.required:
        _register_family(
            candidate_families,
            family_key="scheduler",
            status="candidate",
            reason="El workflow incluye disparos por agenda o eventos y conviene reservar el slot de trigger.",
            matched_signals=["schedule_signal"],
        )

    if has_write_actions and not approval_boundaries:
        gap = ToolRecommendationGap(
            gap_key="approval_boundary_unspecified",
            title="Boundary de aprobacion incompleto",
            question="Que accion de escritura requiere aprobacion humana y quien la aprueba?",
            reason="El caso pide side effects, pero no deja un boundary explicito de control.",
            impact="La seleccion minima no puede promover escritura sin un gate de gobernanza claro.",
            severity="high",
        )
        _append_gap(needs_information, gap)

    if not mandatory_capabilities and not candidate_families:
        if _normalize_text(discovery.problem_statement) and _normalize_text(canvas.user_goal):
            _append_unique(forbidden_capabilities, "transactional_write")
        else:
            gap = ToolRecommendationGap(
                gap_key="workflow_scope_underdefined",
                title="Workflow insuficientemente definido",
                question="Cual es el flujo feliz minimo que el agente debe completar en el MVP?",
                reason="El contexto no deja claro si el agente consulta, decide, escribe o solo responde con contexto inline.",
                impact="Sin workflow minimo aprobado no se puede defender el shortlist para la etapa Herramientas.",
                severity="high",
            )
            _append_gap(needs_information, gap)

    coverage_gaps = [gap for gap in needs_information if gap.severity == "high"]

    mandatory_capability_list = _ordered_capabilities(mandatory_capabilities)
    candidate_family_list = _ordered_families(candidate_families)
    allowed_tool_keys = {
        *[item.capability_key for item in mandatory_capability_list if item.capability_key in CAPABILITY_CATALOG],
        *[
            tool_key
            for family in candidate_family_list
            if family.status != "excluded"
            for tool_key in family.suggested_tool_keys
            if tool_key in CAPABILITY_CATALOG
        ],
    }

    for capability in mandatory_capability_list:
        decision_reason = capability.reason
        dependencies: list[str] = []
        incompatibilities: list[str] = []
        if capability.capability_key == "transactional_write":
            dependencies = ["approval_gate"]
            if has_external_system_signal:
                dependencies.append("read_system_of_record")
            incompatibilities = ["broad_write_backoffice"] if "no_irreversible_side_effects" in hard_constraints else []
        _append_entry(
            recommended_tools,
            _build_entry(
                blueprint=blueprint,
                capability_key=capability.capability_key,
                classification="mandatory",
                decision_reason=decision_reason,
                source_evidence=capability.source_evidence,
                confidence=capability.confidence,
                dependencies=dependencies,
                incompatibilities=incompatibilities,
            ),
        )

    for family in candidate_family_list:
        if family.status != "candidate":
            continue
        for capability_key in family.supported_capabilities:
            if capability_key in mandatory_capabilities:
                continue
            if capability_key in forbidden_capabilities:
                continue
            confidence = 0.58
            if family.family_key == "notification":
                confidence = 0.56
            elif family.family_key == "human_handoff":
                confidence = 0.62
            elif family.family_key == "document_ingestion":
                confidence = 0.6
            elif family.family_key == "scheduler":
                confidence = 0.55
            _append_entry(
                optional_tools,
                _build_entry(
                    blueprint=blueprint,
                    capability_key=capability_key,
                    classification="optional",
                    decision_reason=family.reason,
                    source_evidence=["preflight.candidate_tool_families"],
                    confidence=confidence,
                ),
            )

    if "transactional_write" in forbidden_capabilities or not has_write_actions:
        _append_entry(
            rejected_tools,
            ToolRecommendationEntry(
                tool_key="broad_write_backoffice",
                tool_label="Escritura amplia en sistemas core",
                classification="unnecessary",
                capability_covered="Mutacion masiva o irreversible sobre sistemas de negocio.",
                decision_reason=(
                    "El contexto aprobado no justifica una tool amplia de escritura. Mantenerla fuera reduce complejidad, costo y riesgo."
                ),
                source_evidence=[
                    "discovery.mvp_definition.v1_scope",
                    "discovery.constraints",
                    "blueprint.guardrails",
                ],
                incompatibilities=["MVP simple", "side effects no justificados"],
                confidence=0.84,
            ),
        )

    confidence_value = 0.46
    if has_external_system_signal:
        confidence_value += 0.12
    if has_knowledge_signal and (blueprint.knowledge_profile.mode.strip().lower() == "rag" or blueprint.knowledge_profile.sources):
        confidence_value += 0.1
    if has_write_actions:
        confidence_value += 0.08
    if notification_channels:
        confidence_value += 0.04
    if coverage_gaps:
        confidence_value -= 0.16
    elif needs_information:
        confidence_value -= 0.08
    confidence_value = max(0.28, min(confidence_value, 0.84))
    confidence_band = "low" if confidence_value < 0.5 else "medium"

    workflow_summary = _compact_lines(
        [step.name for step in blueprint.delivery_package.workflow_profile.steps],
        fallback=canvas.user_goal or discovery.desired_outcome,
    )

    context_digest = ToolRecommendationContextDigest(
        workflow_summary=workflow_summary,
        constraints_summary=_compact_lines(
            discovery.constraints,
            fallback="Sin restricciones explicitas adicionales.",
        ),
        source_refs=normalized_sources,
    )

    preflight = ToolRecommendationPreflight(
        case_classification=case_classification,
        agent_goal=_normalize_text(canvas.user_goal or discovery.desired_outcome or discovery.problem_statement),
        primary_user=_normalize_text(canvas.agent_profile.primary_user or discovery.current_user),
        core_workflows=_extract_core_workflows(blueprint, canvas, discovery),
        interaction_modes=_extract_interaction_modes(
            text=business_text,
            has_knowledge_signal=has_knowledge_signal,
            has_external_system_signal=has_external_system_signal,
            notification_channels=notification_channels,
            needs_human_gate=needs_human_gate,
            has_schedule_signal=has_schedule_signal,
        ),
        required_information_sources=required_information_sources,
        required_write_actions=required_write_actions,
        approval_boundaries=approval_boundaries,
        hard_constraints=hard_constraints,
        mandatory_capabilities=mandatory_capability_list,
        forbidden_capabilities=forbidden_capabilities,
        candidate_tool_families=candidate_family_list,
        missing_information=needs_information,
    )

    selected_tool_keys = {item.tool_key for item in [*recommended_tools, *optional_tools]}
    requirements_coverage = _build_requirement_coverage(
        definition_artifact,
        allowed_tool_keys=allowed_tool_keys,
        selected_tool_keys=selected_tool_keys,
    )
    design_role_coverage = _build_design_role_coverage(
        design_artifact,
        allowed_tool_keys=allowed_tool_keys,
        selected_tool_keys=selected_tool_keys,
    )

    summary = (
        f"Preflight heuristico HT2 listo para {case_classification}: "
        f"{len(recommended_tools)} herramientas obligatorias, "
        f"{len(optional_tools)} candidatas opcionales y "
        f"{len(needs_information)} gaps por resolver antes de la poda LLM."
    )
    if not recommended_tools and not optional_tools and not needs_information:
        summary = (
            "Preflight heuristico HT2 listo para un caso inline-first: no se detectaron herramientas minimas obligatorias "
            "mas alla de los controles ya cubiertos por el producto."
        )

    artifact = ToolRecommendationArtifact(
        source_session_id=session_id,
        source_blueprint_version=blueprint_version_number,
        generation_instructions=instructions_text,
        source_stage_versions=ToolRecommendationSourceStageVersions(
            discover=1 if discovery.problem_statement.strip() else None,
            define=1 if canvas.user_goal.strip() else None,
            design=blueprint_version_number,
        ),
        context_digest=context_digest,
        preflight=preflight,
        recommended_tools=recommended_tools,
        optional_tools=optional_tools,
        rejected_tools=rejected_tools,
        requirements_coverage=requirements_coverage,
        design_role_coverage=design_role_coverage,
        coverage_gaps=coverage_gaps,
        needs_information=needs_information,
        confidence=ToolRecommendationConfidence(
            overall=confidence_value,
            band=confidence_band,
            rationale=(
                "La confianza refleja solo el preflight heuristico de HT2. El shortlist ya aplica reglas duras y poda inicial, "
                "pero la seleccion minima por LLM y la autoevaluacion estructurada llegaran en HT3 y HT4."
            ),
        ),
        review_state=ReviewState.partial,
        summary=summary,
    )
    artifact = artifact.model_copy(
        update={
            "context_digest": artifact.context_digest.model_copy(
                update={"digest_sha256": build_tool_recommendation_context_fingerprint(artifact)}
            )
        }
    )
    return _attach_recommendation_contract_seeds(artifact)


def build_tool_recommendation_preflight(
    *,
    session_id: UUID,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    blueprint: BlueprintArtifact,
    definition_artifact: RequirementsDefinitionOutput | None = None,
    design_artifact: DesignRecommendationArtifact | None = None,
    instructions: str = "",
    blueprint_version_number: int | None,
) -> ToolRecommendationArtifact:
    return build_placeholder_tool_recommendation(
        session_id=session_id,
        discovery=discovery,
        canvas=canvas,
        blueprint=blueprint,
        definition_artifact=definition_artifact,
        design_artifact=design_artifact,
        instructions=instructions,
        blueprint_version_number=blueprint_version_number,
    )


def ensure_document_ingestion_for_knowledge_retrieval(
    *,
    artifact: ToolRecommendationArtifact,
    blueprint: BlueprintArtifact,
) -> tuple[ToolRecommendationArtifact, bool]:
    """Close the Tools->Memory RAG dependency before Memory discovers it late.

    knowledge_retrieval and document_ingestion are a pair for governed RAG: the
    first reads from the corpus, the second keeps that corpus ingestible,
    refreshable and traceable. The exact corpus can stay as a deferred decision,
    but the capability must be represented in Tools so Memory can compile.
    """

    selected_keys = {
        item.tool_key
        for item in [*artifact.recommended_tools, *artifact.optional_tools, *artifact.rejected_tools]
        if item.tool_key
    }
    approved_keys = (
        {item.strip().lower() for item in artifact.approved_tools_digest.approved_tool_keys}
        if artifact.approved_tools_digest is not None
        else set()
    )
    knowledge_signalled = (
        "knowledge_retrieval" in selected_keys
        or "knowledge_retrieval" in approved_keys
        or "knowledge_retrieval" in {
            item.capability_key for item in artifact.preflight.mandatory_capabilities
        }
        or "document_grounded" in artifact.preflight.interaction_modes
        or "knowledge.approved_sources" in (
            artifact.approved_tools_digest.retrieval_scopes
            if artifact.approved_tools_digest is not None
            else []
        )
    )
    if not knowledge_signalled or "document_ingestion" in selected_keys or "document_ingestion" in approved_keys:
        return artifact, False

    remediation_reason = (
        "Remediacion automatica: knowledge_retrieval implica una capacidad minima de ingesta, refresh y lineage "
        "para que Memoria pueda declarar RAG sin depender de una herramienta inexistente."
    )
    source_evidence = [
        "tools.approved_tools_digest.knowledge_tool_keys",
        "tools.preflight.interaction_modes",
        "memory.rag_dependency_preflight",
    ]

    patched = artifact.model_copy(deep=True)
    if not any(item.capability_key == "document_ingestion" for item in patched.preflight.mandatory_capabilities):
        patched.preflight.mandatory_capabilities.append(
            ToolPreflightCapability(
                capability_key="document_ingestion",
                label=str(CAPABILITY_CATALOG["document_ingestion"]["label"]),
                required=True,
                reason=remediation_reason,
                source_evidence=source_evidence,
                confidence=0.74,
            )
        )
    for family in patched.preflight.candidate_tool_families:
        if family.family_key != "document_ingestion":
            continue
        family.status = "required"
        family.reason = remediation_reason
        family.matched_signals = _merge_unique(family.matched_signals, ["knowledge_retrieval_dependency"])
        break
    else:
        catalog = TOOL_FAMILY_CATALOG["document_ingestion"]
        patched.preflight.candidate_tool_families.append(
            ToolFamilyCandidate(
                family_key="document_ingestion",
                label=str(catalog["label"]),
                status="required",
                supported_capabilities=list(catalog["supported_capabilities"]),
                matched_signals=["knowledge_retrieval_dependency"],
                suggested_tool_keys=list(catalog["suggested_tool_keys"]),
                estimated_complexity=str(catalog["estimated_complexity"]),
                reason=remediation_reason,
            )
        )

    existing_entry = next(
        (
            item
            for item in [*patched.optional_tools, *patched.rejected_tools]
            if item.tool_key == "document_ingestion"
        ),
        None,
    )
    if existing_entry is None:
        existing_entry = _build_entry(
            blueprint=blueprint,
            capability_key="document_ingestion",
            classification="mandatory",
            decision_reason=remediation_reason,
            source_evidence=source_evidence,
            confidence=0.74,
        )
    else:
        existing_entry = existing_entry.model_copy(
            update={
                "classification": "mandatory",
                "decision_reason": remediation_reason,
                "source_evidence": _merge_unique(existing_entry.source_evidence, source_evidence),
                "confidence": max(existing_entry.confidence, 0.74),
            }
        )

    patched.optional_tools = [item for item in patched.optional_tools if item.tool_key != "document_ingestion"]
    patched.rejected_tools = [item for item in patched.rejected_tools if item.tool_key != "document_ingestion"]
    _append_entry(patched.recommended_tools, existing_entry)
    patched.summary = (
        f"{patched.summary} Se agrego document_ingestion automaticamente para cerrar la dependencia RAG "
        "antes de generar Memoria."
    ).strip()
    return evaluate_tool_recommendation_artifact(patched), True


def _tool_key_to_enum(tool_key: str) -> ToolRecommendationAllowedToolKey | None:
    try:
        return ToolRecommendationAllowedToolKey(tool_key)
    except ValueError:
        return None


def _mandatory_capability_map(artifact: ToolRecommendationArtifact) -> dict[str, ToolPreflightCapability]:
    return {
        item.capability_key: item
        for item in artifact.preflight.mandatory_capabilities
        if item.capability_key in CAPABILITY_CATALOG
    }


def _allowed_tool_keys(artifact: ToolRecommendationArtifact) -> set[str]:
    allowed: set[str] = set()
    for capability in artifact.preflight.mandatory_capabilities:
        if capability.capability_key in CAPABILITY_CATALOG:
            allowed.add(capability.capability_key)
    for family in artifact.preflight.candidate_tool_families:
        if family.status == "excluded":
            continue
        for tool_key in family.suggested_tool_keys:
            if tool_key in CAPABILITY_CATALOG:
                allowed.add(tool_key)
    return allowed


def _confidence_band(value: float) -> str:
    if value >= 0.8:
        return "high"
    if value >= 0.55:
        return "medium"
    return "low"


def _status_for_categories(
    findings: list[ToolRecommendationFinding],
    *categories: str,
) -> ReviewState:
    relevant = [item for item in findings if item.category in categories]
    if any(item.severity == "blocking" for item in relevant):
        return ReviewState.blocked
    if relevant:
        return ReviewState.partial
    return ReviewState.complete


def _evaluation_penalty_for_findings(findings: list[ToolRecommendationFinding]) -> float:
    blocking_count = sum(1 for item in findings if item.severity == "blocking")
    warning_count = sum(1 for item in findings if item.severity == "warning")
    return min(0.48, (blocking_count * 0.14) + (warning_count * 0.05))


def _historical_evaluation_penalty_from_rationale(rationale: str) -> float:
    """Recover prior HT4 penalties so re-evaluation does not compound stale scores."""

    penalty = 0.0
    for match in re.finditer(r"HT4 encontro (\d+) bloqueos y (\d+) alertas", rationale):
        blocking_count = int(match.group(1))
        warning_count = int(match.group(2))
        penalty += min(0.48, (blocking_count * 0.14) + (warning_count * 0.05))
    return min(0.74, penalty)


def _selected_tool_keys(artifact: ToolRecommendationArtifact) -> set[str]:
    return {item.tool_key for item in [*artifact.recommended_tools, *artifact.optional_tools]}


def _stable_payload_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def build_tool_recommendation_context_fingerprint(artifact: ToolRecommendationArtifact) -> str:
    prompt_payload = build_tool_recommendation_prompt_input(artifact).model_dump(mode="json")
    prompt_payload.pop("source_session_id", None)
    prompt_payload.pop("source_blueprint_version", None)
    return _stable_payload_hash(prompt_payload)


def _memory_implications_for_tool(tool: BlueprintTool) -> list[str]:
    implications: list[str] = []
    if tool.has_side_effects or tool.requires_approval:
        implications.append("checkpoint_before_execution")
        implications.append("persist_tool_execution_audit")
    if tool.name in KNOWLEDGE_TOOL_KEYS:
        implications.append("knowledge_grounding")
    if tool.name == "scheduler":
        implications.append("scheduled_state_tracking")
    if tool.name == "human_handoff":
        implications.append("handoff_state_tracking")
    return implications


def build_approved_tools_digest_from_blueprint_tools(
    tools: list[BlueprintTool],
    *,
    source_session_id: UUID | None = None,
    source_blueprint_version: int | None = None,
    promoted_blueprint_version: int | None = None,
    mandatory_tool_keys: list[str] | None = None,
    optional_tool_keys: list[str] | None = None,
) -> ApprovedToolsDigest:
    normalized_tools = [item for item in tools if _normalize_text(item.name)]
    approved_tool_keys = [_normalize_text(item.name) for item in normalized_tools]
    mandatory_tool_keys = [_normalize_text(item) for item in (mandatory_tool_keys or []) if _normalize_text(item)]
    optional_tool_keys = [_normalize_text(item) for item in (optional_tool_keys or []) if _normalize_text(item)]
    side_effect_tool_keys = [_normalize_text(item.name) for item in normalized_tools if item.has_side_effects]
    approval_required_tool_keys = [_normalize_text(item.name) for item in normalized_tools if item.requires_approval]
    knowledge_tool_keys = [_normalize_text(item.name) for item in normalized_tools if item.name in KNOWLEDGE_TOOL_KEYS]
    retrieval_scopes = ["approved_tools_digest"]
    memory_hints = ["approved_tools_only"]
    if knowledge_tool_keys:
        retrieval_scopes.append("knowledge.approved_sources")
        memory_hints.append("knowledge_grounding_required")
    if side_effect_tool_keys or approval_required_tool_keys:
        memory_hints.append("checkpoint_before_side_effects")
        memory_hints.append("approval_lineage_required")
    if any(item.name == "scheduler" for item in normalized_tools):
        memory_hints.append("scheduled_resume_support")
    if any(item.name == "human_handoff" for item in normalized_tools):
        memory_hints.append("handoff_history_required")

    entry_payload = [
        ApprovedToolDigestEntry(
            tool_key=_normalize_text(item.name),
            tool_label=item.purpose or item.name.replace("_", " ").title(),
            blueprint_tool_name=item.name,
            classification=(
                "mandatory"
                if _normalize_text(item.name) in set(mandatory_tool_keys)
                else "optional"
                if _normalize_text(item.name) in set(optional_tool_keys)
                else "approved"
            ),
            integration_kind=item.integration_kind,
            owner=item.owner,
            requires_approval=item.requires_approval,
            has_side_effects=item.has_side_effects,
            memory_implications=_memory_implications_for_tool(item),
        ).model_dump(mode="json")
        for item in normalized_tools
    ]
    digest_sha256 = _stable_payload_hash(
        {
            "approved_tool_keys": approved_tool_keys,
            "entry_payload": entry_payload,
            "memory_hints": memory_hints,
            "retrieval_scopes": retrieval_scopes,
        }
    )

    recommended_memory_strategy = (
        "persistent_memory"
        if knowledge_tool_keys or side_effect_tool_keys or approval_required_tool_keys
        else "session_memory"
    )
    summary = (
        f"Digest compacto de {len(approved_tool_keys)} tools aprobadas: "
        f"{', '.join(approved_tool_keys) if approved_tool_keys else 'sin tools promovidas'}."
    )

    return ApprovedToolsDigest(
        digest_sha256=digest_sha256,
        source_session_id=source_session_id,
        source_blueprint_version=source_blueprint_version,
        promoted_blueprint_version=promoted_blueprint_version,
        tool_count=len(approved_tool_keys),
        approved_tool_keys=approved_tool_keys,
        mandatory_tool_keys=mandatory_tool_keys,
        optional_tool_keys=optional_tool_keys,
        side_effect_tool_keys=side_effect_tool_keys,
        approval_required_tool_keys=approval_required_tool_keys,
        knowledge_tool_keys=knowledge_tool_keys,
        selected_blueprint_tool_names=[item.name for item in normalized_tools],
        retrieval_scopes=retrieval_scopes,
        memory_hints=memory_hints,
        recommended_memory_strategy=recommended_memory_strategy,
        summary=summary,
    )


def _build_blueprint_tool_from_recommendation(
    *,
    artifact: ToolRecommendationArtifact,
    entry: ToolRecommendationEntry,
) -> BlueprintTool:
    seed = entry.contract_seed or BlueprintTool()
    read_inputs = artifact.preflight.required_information_sources or ["request_context"]
    write_inputs = artifact.preflight.required_write_actions or ["approved_action"]
    approval_inputs = artifact.preflight.approval_boundaries or ["approval_decision"]

    if entry.tool_key == "read_system_of_record":
        return BlueprintTool(
            name="read_system_of_record",
            purpose=entry.capability_covered or "Consultar la fuente operativa aprobada (CRM, ERP, DB) antes de responder o decidir.",
            owner=seed.owner or "system_owner_pending",
            archetype="read_only_lookup",
            tool_type="external",
            execution_stage="execution",
            when_to_use="Utilizar en la fase de ejecucion cuando el agente requiera consultar datos actualizados de clientes, transacciones o inventario desde un sistema externo antes de formular una decision.",
            integration_kind=seed.integration_kind or "api",
            endpoint_reference=seed.endpoint_reference or "integration://system-of-record/read",
            auth_reference=seed.auth_reference or "workspace_managed_secret",
            risk_level=seed.risk_level or "medium",
            requires_approval=False,
            inputs=read_inputs,
            outputs=["normalized_system_record"],
            request_schema={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "Identificador unico del registro en el sistema fuente"},
                    "entity_type": {"type": "string", "description": "Tipo de entidad (customer, invoice, ticket)"},
                    "fields": {"type": "array", "items": {"type": "string"}, "description": "Campos especificos a recuperar"}
                },
                "required": ["record_id", "entity_type"]
            },
            response_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "error"]},
                    "data": {"type": "object", "description": "Objeto con los atributos recuperados del sistema fuente"},
                    "fetched_at": {"type": "string", "format": "date-time"}
                },
                "required": ["status", "data"]
            },
            usage_examples=[
                {
                    "title": "Consulta de cliente por ID",
                    "request": {"record_id": "CUST-9842", "entity_type": "customer", "fields": ["name", "email", "status"]},
                    "response": {"status": "success", "data": {"id": "CUST-9842", "name": "Acme Corp", "status": "active"}, "fetched_at": "2026-07-29T10:00:00Z"}
                }
            ],
            security_config={"auth_type": "bearer", "secret_ref": "WORKSPACE_SOR_API_KEY", "encrypt_transit": True},
            validations=["tenant_scope_validation", "request_schema_validation", "response_schema_validation"],
            typed_errors=["RECORD_NOT_FOUND", "AUTH_EXPIRED", "RATE_LIMITED", "UPSTREAM_TIMEOUT"],
            permissions=["read_system_of_record"],
            scopes=["workspace", "read_only"],
            sensitive_data=["business_record"],
            audit_rules=["Registrar request_id, source_ref y latencia de cada lectura aprobada."],
            has_side_effects=False,
            execution_mode=seed.execution_mode or "sync",
            approval_policy="No requiere aprobacion adicional; solo lectura auditada.",
            retry_strategy="Retry exponencial corto (max 3 intentos) para fallas transitorias 5xx.",
            idempotency_strategy="Lectura idempotente por naturaleza.",
            compensation_strategy="No aplica por ser read-only.",
            approval_reason="",
            failure_mode="Declarar needs_review cuando la fuente operativa no responda o devuelva 404.",
            rate_limit_policy="Aplicar limite de 100 req/min por workspace.",
            timeout_policy="Timeout de 5000ms con fallback controlado.",
            contract_review_state="needs-review",
        )

    if entry.tool_key == "approval_gate":
        return BlueprintTool(
            name="approval_gate",
            purpose=entry.capability_covered or "Capturar la decision humana requerida antes de ejecutar acciones sensibles.",
            owner=seed.owner or "business_owner_pending",
            archetype="governance_gate",
            tool_type="internal",
            execution_stage="execution",
            when_to_use="Interviene inmediatamente antes de cualquier ejecucion de side effects o escritura transaccional para solicitar confirmacion o firma digital al usuario autorizador.",
            integration_kind=seed.integration_kind or "governed_handoff",
            endpoint_reference=seed.endpoint_reference or "workflow://approval/gate",
            auth_reference=seed.auth_reference or "workspace_member_session",
            risk_level=seed.risk_level or "medium",
            requires_approval=True,
            inputs=["proposed_action", "evidence_bundle", *approval_inputs],
            outputs=["approval_decision", "approval_audit_entry"],
            request_schema={
                "type": "object",
                "properties": {
                    "gate_key": {"type": "string", "description": "Identificador del gate de gobierno"},
                    "proposed_action": {"type": "string", "description": "Resumen de la accion que requiere autorizacion"},
                    "payload_summary": {"type": "object", "description": "Detalle de los parametros de la accion"}
                },
                "required": ["gate_key", "proposed_action"]
            },
            response_schema={
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean", "description": "Resultado de la aprobacion (true/false)"},
                    "approver_id": {"type": "string", "description": "ID del usuario que firmo la decision"},
                    "resolution_note": {"type": "string", "description": "Nota u observacion de la aprobacion"}
                },
                "required": ["approved", "approver_id"]
            },
            usage_examples=[
                {
                    "title": "Solicitud de aprobacion para cambio de plan",
                    "request": {"gate_key": "gate_tier_change", "proposed_action": "Actualizar suscripcion a Enterprise", "payload_summary": {"new_tier": "enterprise", "monthly_cost": 499}},
                    "response": {"approved": True, "approver_id": "user_admin_01", "resolution_note": "Aprobado en comite de compras"}
                }
            ],
            security_config={"auth_type": "session_token", "rbac_role": "approver", "encrypt_transit": True},
            validations=["approval_payload_validation", "actor_role_validation", "policy_guard_validation"],
            typed_errors=["APPROVAL_PENDING", "APPROVAL_REJECTED", "GATE_TIMEOUT", "POLICY_VIOLATION"],
            permissions=["request_approval"],
            scopes=["workspace", "human_in_the_loop"],
            sensitive_data=["approval_context"],
            audit_rules=["Registrar approver, decision, rationale y timestamp de la aprobacion."],
            has_side_effects=False,
            execution_mode=seed.execution_mode or "async",
            approval_policy="Requiere decision humana explicita antes de continuar.",
            retry_strategy="Reintentar solo reenvio de notificaciones de espera, no la decision humana.",
            idempotency_strategy="Usar request_id unico por solicitud de aprobacion.",
            compensation_strategy="Cancelar la accion propuesta si la aprobacion expira o es rechazada.",
            approval_reason="Tool obligatoria cuando el caso tiene decisiones no delegables o side effects.",
            failure_mode="Bloquear la accion y escalar si no hay aprobacion vigente.",
            rate_limit_policy="Limitar solicitudes de gate a 10 activas por sesion.",
            timeout_policy="Esperar ventana de aprobacion de 24 horas y luego expirar.",
            contract_review_state="needs-review",
        )

    if entry.tool_key == "transactional_write":
        return BlueprintTool(
            name="transactional_write",
            purpose=entry.capability_covered or "Ejecutar la accion operativa aprobada sobre el sistema objetivo.",
            owner=seed.owner or "system_owner_pending",
            archetype="transactional_write",
            tool_type="external",
            execution_stage="execution",
            when_to_use="Utilizar unicamente cuando el agente tenga una aprobacion valida en el approval_gate para aplicar mutaciones, crear registros o modificar estados en el sistema externo.",
            integration_kind=seed.integration_kind or "api",
            endpoint_reference=seed.endpoint_reference or "integration://system-of-record/write",
            auth_reference=seed.auth_reference or "workspace_managed_secret",
            risk_level=seed.risk_level or "high",
            requires_approval=True,
            inputs=["approved_action", "approval_token", *write_inputs],
            outputs=["write_receipt", "updated_record_ref"],
            request_schema={
                "type": "object",
                "properties": {
                    "action_name": {"type": "string", "description": "Nombre de la accion de escritura (create_ticket, update_customer)"},
                    "approval_token": {"type": "string", "description": "Token de autorizacion emitido por el approval_gate"},
                    "payload": {"type": "object", "description": "Campos a escribir o mutar"}
                },
                "required": ["action_name", "approval_token", "payload"]
            },
            response_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "failed"]},
                    "transaction_id": {"type": "string", "description": "Identificador unico de la transaccion"},
                    "receipt": {"type": "object", "description": "Estado resultante del objeto en la BD externa"}
                },
                "required": ["status", "transaction_id"]
            },
            usage_examples=[
                {
                    "title": "Escritura transaccional de ticket",
                    "request": {"action_name": "create_ticket", "approval_token": "TOK-APP-9921", "payload": {"subject": "Falla en integracion", "priority": "high"}},
                    "response": {"status": "success", "transaction_id": "TX-88321", "receipt": {"ticket_id": "TCK-5512", "status": "open"}}
                }
            ],
            security_config={"auth_type": "bearer", "secret_ref": "WORKSPACE_WRITE_API_KEY", "idempotency_header": "X-Idempotency-Key"},
            validations=["approval_token_validation", "payload_schema_validation", "idempotency_key_validation"],
            typed_errors=["INVALID_APPROVAL_TOKEN", "WRITE_CONFLICT", "SCHEMA_VALIDATION_ERROR", "COMPENSATION_REQUIRED"],
            permissions=write_inputs,
            scopes=["workspace", "mutating_operation"],
            sensitive_data=["business_record"],
            audit_rules=["Persistir request_id, actor, approval_id, payload_hash y resultado de escritura."],
            has_side_effects=True,
            execution_mode=seed.execution_mode or "sync",
            approval_policy="Solo puede ejecutarse con approval gate resuelto y payload validado.",
            retry_strategy="Retry selectivo solo para errores transitorios 503 sin mutacion efectuada.",
            idempotency_strategy="Exigir idempotency_key estable por accion aprobada.",
            compensation_strategy="Aplicar rollback funcional o remediation guiada cuando exista fallo parcial.",
            approval_reason="Tool obligatoria cuando el workflow requiere side effects aprobados.",
            failure_mode="Bloquear, registrar incidente y escalar si el write no confirma consistencia.",
            rate_limit_policy="Limitar a 20 mutaciones por minuto por workspace.",
            timeout_policy="Timeout de 10000ms con confirmacion explicita del estado final.",
            contract_review_state="needs-review",
        )

    if entry.tool_key == "knowledge_retrieval":
        return BlueprintTool(
            name="knowledge_retrieval",
            purpose=entry.capability_covered or "Recuperar conocimiento aprobado para grounding y respuestas trazables.",
            owner=seed.owner or "knowledge_owner_pending",
            archetype="knowledge_retrieval",
            tool_type="internal",
            execution_stage="discovery",
            when_to_use="Invocada durante el razonamiento o respuesta conversacional cuando el agente requiere evidencia fáctica de manuales, normativas o documentos aprobados.",
            integration_kind=seed.integration_kind or "retrieval",
            endpoint_reference=seed.endpoint_reference or "knowledge://approved-retrieval/query",
            auth_reference=seed.auth_reference or "workspace_managed_secret",
            risk_level=seed.risk_level or "medium",
            requires_approval=False,
            inputs=["question", "approved_source_filters"],
            outputs=["grounded_answer_context", "citations_bundle"],
            request_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta o pregunta en lenguaje natural"},
                    "top_k": {"type": "integer", "default": 5, "description": "Cantidad maxima de pasajes a recuperar"},
                    "filters": {"type": "object", "description": "Filtros por dominio o fecha"}
                },
                "required": ["query"]
            },
            response_schema={
                "type": "object",
                "properties": {
                    "passages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "source_document": {"type": "string"},
                                "score": {"type": "number"}
                            }
                        }
                    }
                },
                "required": ["passages"]
            },
            usage_examples=[
                {
                    "title": "Búsqueda en política de devoluciones",
                    "request": {"query": "Cual es el plazo limite para devoluciones de hardware?", "top_k": 3},
                    "response": {"passages": [{"content": "El plazo maximo de devolucion es de 30 dias calendario desde la compra.", "source_document": "politica_garantias_v2.pdf", "score": 0.92}]}
                }
            ],
            security_config={"auth_type": "internal_token", "read_only": True},
            validations=["query_schema_validation", "approved_source_validation"],
            typed_errors=["NO_EVIDENCE_FOUND", "INDEX_UNAVAILABLE", "RETRIEVAL_TIMEOUT"],
            permissions=["read_approved_knowledge"],
            scopes=["workspace", "knowledge"],
            sensitive_data=["knowledge_reference"],
            audit_rules=["Registrar corpus version, source ids y score de retrieval."],
            has_side_effects=False,
            execution_mode=seed.execution_mode or "sync",
            approval_policy="Usar solo fuentes aprobadas y trazables del workspace.",
            retry_strategy="Retry corto solo para fallas transitorias del retrieval.",
            idempotency_strategy="Consultas deterministicas sobre filtros aprobados.",
            compensation_strategy="No aplica; responder falta de evidencia cuando no exista grounding.",
            approval_reason="",
            failure_mode="Declarar needs_review si no hay evidencia suficiente o las fuentes fallan.",
            rate_limit_policy="Sin limite directo dentro del runtime interno.",
            timeout_policy="Timeout de 3000ms con fallback a falta de evidencia.",
            contract_review_state="needs-review",
        )

    if entry.tool_key == "document_ingestion":
        return BlueprintTool(
            name="document_ingestion",
            purpose=entry.capability_covered or "Preparar y refrescar fuentes documentales aprobadas para retrieval consistente.",
            owner=seed.owner or "knowledge_owner_pending",
            archetype="document_ingestion",
            tool_type="external",
            execution_stage="tools",
            when_to_use="Utilizada en la fase de administracion del conocimiento cuando se cargan nuevos archivos o se sincroniza una carpeta de SharePoint / Google Drive.",
            integration_kind=seed.integration_kind or "pipeline",
            endpoint_reference=seed.endpoint_reference or "knowledge://approved-ingestion/refresh",
            auth_reference=seed.auth_reference or "workspace_managed_secret",
            risk_level=seed.risk_level or "medium",
            requires_approval=False,
            inputs=["approved_documents", "ingestion_policy"],
            outputs=["ingestion_report", "indexed_source_refs"],
            request_schema={
                "type": "object",
                "properties": {
                    "source_uri": {"type": "string", "description": "URI del repositorio documental o archivo"},
                    "refresh_mode": {"type": "string", "enum": ["full", "incremental"]}
                },
                "required": ["source_uri"]
            },
            response_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["completed", "indexing", "failed"]},
                    "processed_documents": {"type": "integer"},
                    "chunks_created": {"type": "integer"}
                },
                "required": ["status", "processed_documents"]
            },
            usage_examples=[
                {
                    "title": "Ingesta incremental de manuales",
                    "request": {"source_uri": "s3://company-docs/manuals/2026/", "refresh_mode": "incremental"},
                    "response": {"status": "completed", "processed_documents": 12, "chunks_created": 480}
                }
            ],
            security_config={"auth_type": "iam_role", "secret_ref": "KNOWLEDGE_INGESTION_CREDENTIALS"},
            validations=["document_schema_validation", "approved_source_validation"],
            typed_errors=["PARSER_FAILURE", "STORAGE_UNREACHABLE", "FILE_SIZE_EXCEEDED"],
            permissions=["ingest_approved_documents"],
            scopes=["workspace", "knowledge"],
            sensitive_data=["document_metadata"],
            audit_rules=["Registrar source_version, parser, chunking_policy y resultado del refresh."],
            has_side_effects=True,
            execution_mode=seed.execution_mode or "async",
            approval_policy="Solo ingestar documentos previamente aprobados para el agente.",
            retry_strategy="Retry asincrono con backoff y deduplicacion por source_version.",
            idempotency_strategy="Evitar doble indexacion del mismo documento y version.",
            compensation_strategy="Revertir o aislar lotes corruptos del indice aprobado.",
            approval_reason="",
            failure_mode="Congelar el refresh y escalar si el indice queda inconsistente.",
            rate_limit_policy="Maximo 5 ejecuciones de ingesta simultaneas.",
            timeout_policy="Timeout de 60000ms con monitoreo de progreso.",
            contract_review_state="needs-review",
        )

    if entry.tool_key == "outbound_notification":
        return BlueprintTool(
            name="outbound_notification",
            purpose=entry.capability_covered or "Cerrar el loop operativo enviando notificaciones a canales externos (Slack, Email, Teams).",
            owner=seed.owner or "ops_owner_pending",
            archetype="notification",
            tool_type="external",
            execution_stage="execution",
            when_to_use="Se utiliza al finalizar un flujo de trabajo o tras una aprobacion para enviar reportes, alertas o confirmaciones al usuario final o a un equipo en Slack/Email.",
            integration_kind=seed.integration_kind or "api",
            endpoint_reference=seed.endpoint_reference or "notification://approved-channel/send",
            auth_reference=seed.auth_reference or "workspace_managed_secret",
            risk_level=seed.risk_level or "medium",
            requires_approval=False,
            inputs=["recipient_ref", "approved_message_template", "delivery_channel"],
            outputs=["delivery_receipt"],
            request_schema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": ["email", "slack", "teams", "webhook"]},
                    "recipient": {"type": "string", "description": "Email, ID de canal o URL objetivo"},
                    "subject": {"type": "string"},
                    "body_markdown": {"type": "string"}
                },
                "required": ["channel", "recipient", "body_markdown"]
            },
            response_schema={
                "type": "object",
                "properties": {
                    "delivered": {"type": "boolean"},
                    "message_id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"}
                },
                "required": ["delivered", "message_id"]
            },
            usage_examples=[
                {
                    "title": "Envío de alerta por Slack",
                    "request": {"channel": "slack", "recipient": "#ops-alerts", "subject": "Alerta de Incidencia", "body_markdown": "Se aprobo la accion en el ticket TCK-5512"},
                    "response": {"delivered": True, "message_id": "MSG-99214", "timestamp": "2026-07-29T10:15:00Z"}
                }
            ],
            security_config={"auth_type": "api_key", "secret_ref": "SLACK_WEBHOOK_URL"},
            validations=["recipient_validation", "template_policy_validation"],
            typed_errors=["DELIVERY_FAILED", "INVALID_RECIPIENT", "CHANNEL_UNAVAILABLE"],
            permissions=["send_notification"],
            scopes=["workspace", "notification"],
            sensitive_data=["recipient_ref"],
            audit_rules=["Registrar recipient_ref, channel, template_id y delivery_receipt."],
            has_side_effects=True,
            execution_mode=seed.execution_mode or "async",
            approval_policy="Usar solo templates y canales aprobados para el workspace.",
            retry_strategy="Retry asincrono con circuit breaker por canal.",
            idempotency_strategy="Deduplicar mensajes por workflow_step y recipient_ref.",
            compensation_strategy="Evitar reenvios duplicados y escalar cuando el canal falle.",
            approval_reason="",
            failure_mode="Escalar a owner si la notificacion no se entrega en la ventana esperada.",
            rate_limit_policy="Aplicar limites de 60 msgs/min por canal.",
            timeout_policy="Timeout de 5000ms con confirmacion de entrega.",
            contract_review_state="needs-review",
        )

    if entry.tool_key == "human_handoff":
        return BlueprintTool(
            name="human_handoff",
            purpose=entry.capability_covered or "Escalar el caso a un owner humano con contexto, evidencia y trazabilidad.",
            owner=seed.owner or "ops_owner_pending",
            archetype="human_handoff",
            tool_type="internal",
            execution_stage="validation",
            when_to_use="Invocada cuando el agente detecta condiciones fuera del contrato, falta de confianza en la respuesta o solicitudes directas de asistencia humana.",
            integration_kind=seed.integration_kind or "governed_handoff",
            endpoint_reference=seed.endpoint_reference or "workflow://handoff/human_owner",
            auth_reference=seed.auth_reference or "workspace_member_session",
            risk_level=seed.risk_level or "medium",
            requires_approval=False,
            inputs=["case_context", "evidence_bundle", "handoff_reason"],
            outputs=["handoff_record", "owner_assignment"],
            request_schema={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Causa del escalamiento"},
                    "context_snapshot": {"type": "object", "description": "Estado del agente y variables actuales"},
                    "assigned_team": {"type": "string", "description": "Equipo objetivo para recibir el caso"}
                },
                "required": ["reason", "context_snapshot"]
            },
            response_schema={
                "type": "object",
                "properties": {
                    "handoff_id": {"type": "string"},
                    "assigned_user": {"type": "string"},
                    "ticket_ref": {"type": "string"}
                },
                "required": ["handoff_id"]
            },
            usage_examples=[
                {
                    "title": "Escalamiento por baja confianza",
                    "request": {"reason": "Confianza menor a 50% en clasificacion legal", "context_snapshot": {"case_id": "CASE-112"}, "assigned_team": "legal_ops"},
                    "response": {"handoff_id": "HND-7712", "assigned_user": "ana.garcia@company.com", "ticket_ref": "LEG-330"}
                }
            ],
            security_config={"auth_type": "internal_token", "rbac_role": "agent_runtime"},
            validations=["handoff_payload_validation", "owner_route_validation"],
            typed_errors=["OWNER_UNAVAILABLE", "INVALID_HANDOFF_ROUTE"],
            permissions=["create_handoff"],
            scopes=["workspace", "handoff"],
            sensitive_data=["case_context"],
            audit_rules=["Registrar owner, handoff_reason y evidencia entregada al humano."],
            has_side_effects=False,
            execution_mode=seed.execution_mode or "async",
            approval_policy="Escalar cuando el workflow no pueda cerrarse dentro del contrato aprobado.",
            retry_strategy="Reintentar asignacion solo cuando falle el canal de handoff.",
            idempotency_strategy="Usar handoff_key unico por caso y etapa.",
            compensation_strategy="No aplica; dejar evidencia y espera controlada.",
            approval_reason="",
            failure_mode="Escalar a cola de incidentes si no existe owner disponible.",
            rate_limit_policy="Limitar handoffs duplicados por caso activo.",
            timeout_policy="Timeout medio con ack de recepcion humana.",
            contract_review_state="needs-review",
        )

    if entry.tool_key == "scheduler":
        return BlueprintTool(
            name="scheduler",
            purpose=entry.capability_covered or "Disparar ejecuciones programadas o por evento dentro de ventanas controladas.",
            owner=seed.owner or "ops_owner_pending",
            archetype="scheduler",
            tool_type="internal",
            execution_stage="define",
            when_to_use="Utilizada para agendar tareas recurrentes, temporizadores o monitoreo periódico de condiciones del negocio.",
            integration_kind=seed.integration_kind or "event_trigger",
            endpoint_reference=seed.endpoint_reference or "workflow://scheduler/trigger",
            auth_reference=seed.auth_reference or "workspace_member_session",
            risk_level=seed.risk_level or "low",
            requires_approval=False,
            inputs=["schedule_expression", "trigger_payload"],
            outputs=["scheduled_run_ref"],
            request_schema={
                "type": "object",
                "properties": {
                    "cron_expression": {"type": "string", "description": "Expresion cron o sintaxis ISO-8601"},
                    "task_name": {"type": "string", "description": "Nombre de la tarea a agendar"},
                    "payload": {"type": "object"}
                },
                "required": ["cron_expression", "task_name"]
            },
            response_schema={
                "type": "object",
                "properties": {
                    "schedule_id": {"type": "string"},
                    "next_run_at": {"type": "string", "format": "date-time"}
                },
                "required": ["schedule_id", "next_run_at"]
            },
            usage_examples=[
                {
                    "title": "Programación de reporte diario",
                    "request": {"cron_expression": "0 8 * * 1-5", "task_name": "daily_summary_report", "payload": {"format": "pdf"}},
                    "response": {"schedule_id": "SCH-4410", "next_run_at": "2026-07-30T08:00:00Z"}
                }
            ],
            security_config={"auth_type": "internal_token"},
            validations=["schedule_validation", "payload_schema_validation"],
            typed_errors=["INVALID_CRON_SYNTAX", "SCHEDULER_UNAVAILABLE"],
            permissions=["schedule_execution"],
            scopes=["workspace", "scheduling"],
            sensitive_data=[],
            audit_rules=["Registrar schedule_expression, actor y run_ref generado."],
            has_side_effects=False,
            execution_mode=seed.execution_mode or "async",
            approval_policy="Usar solo ventanas y triggers aprobados para el workflow.",
            retry_strategy="Reintentar registro del trigger cuando falle el scheduler.",
            idempotency_strategy="Deduplicar alta de triggers por workflow y ventana.",
            compensation_strategy="Eliminar triggers huerfanos o duplicados si la configuracion cambia.",
            approval_reason="",
            failure_mode="Escalar si el trigger no puede registrarse o queda inconsistente.",
            rate_limit_policy="Limitar frecuencia de altas y disparos por workspace.",
            timeout_policy="Timeout corto para el alta; ejecucion real fuera de banda.",
            contract_review_state="needs-review",
        )

    raise ValueError(f"Unsupported recommendation tool key for promotion: {entry.tool_key}")


def _entry_with_contract_seed(
    artifact: ToolRecommendationArtifact,
    entry: ToolRecommendationEntry,
) -> ToolRecommendationEntry:
    if entry.contract_seed is not None or entry.tool_key not in CAPABILITY_CATALOG:
        return entry
    try:
        contract_seed = _build_blueprint_tool_from_recommendation(artifact=artifact, entry=entry)
    except ValueError:
        return entry
    return entry.model_copy(update={"contract_seed": contract_seed})


def _attach_recommendation_contract_seeds(
    artifact: ToolRecommendationArtifact,
) -> ToolRecommendationArtifact:
    return artifact.model_copy(
        update={
            "recommended_tools": [
                _entry_with_contract_seed(artifact, item) for item in artifact.recommended_tools
            ],
            "optional_tools": [
                _entry_with_contract_seed(artifact, item) for item in artifact.optional_tools
            ],
            "rejected_tools": [
                _entry_with_contract_seed(artifact, item) for item in artifact.rejected_tools
            ],
        }
    )


def promote_tool_recommendation_to_blueprint_tools(
    artifact: ToolRecommendationArtifact,
    *,
    include_optional_tool_keys: list[str] | None = None,
) -> tuple[list[BlueprintTool], list[ToolRecommendationReviewDecision], ApprovedToolsDigest]:
    if artifact.evaluation.promotion_blocked:
        re_evaluated = evaluate_tool_recommendation_artifact(artifact)
        if not re_evaluated.evaluation.promotion_blocked:
            artifact = re_evaluated
        else:
            blocking_findings = [f for f in re_evaluated.evaluation.findings if f.severity == "blocking"]
            only_soft_blockers = all(f.category in {"coverage", "confidence", "minimality"} for f in blocking_findings)
            if only_soft_blockers:
                artifact = re_evaluated.model_copy(
                    update={
                        "evaluation": re_evaluated.evaluation.model_copy(
                            update={
                                "promotion_blocked": False,
                                "overall_status": ReviewState.partial,
                            }
                        )
                    }
                )
            else:
                raise ValueError("La recomendacion de tools sigue bloqueada por HT4 y no puede promoverse a blueprint.tools.")

    optional_allowlist = {_normalize_text(item) for item in (include_optional_tool_keys or []) if _normalize_text(item)}
    known_optional_keys = {_normalize_text(item.tool_key) for item in artifact.optional_tools}
    unknown_optional_keys = sorted(optional_allowlist - known_optional_keys)
    if unknown_optional_keys:
        raise ValueError(f"Optional tools desconocidas para promover: {', '.join(unknown_optional_keys)}")

    approved_entries = list(artifact.recommended_tools)
    approved_entries.extend(item for item in artifact.optional_tools if _normalize_text(item.tool_key) in optional_allowlist)

    review_decisions = [
        ToolRecommendationReviewDecision(
            tool_key=item.tool_key,
            classification=item.classification,
            decision="approved",
            decision_reason="Tool mandatory promovida como parte del set minimo aprobado.",
        )
        for item in artifact.recommended_tools
    ]
    review_decisions.extend(
        ToolRecommendationReviewDecision(
            tool_key=item.tool_key,
            classification=item.classification,
            decision="approved" if _normalize_text(item.tool_key) in optional_allowlist else "rejected",
            decision_reason=(
                "Tool optional incluida por decision del usuario."
                if _normalize_text(item.tool_key) in optional_allowlist
                else "Tool optional descartada para mantener el set minimo."
            ),
        )
        for item in artifact.optional_tools
    )
    review_decisions.extend(
        ToolRecommendationReviewDecision(
            tool_key=item.tool_key,
            classification=item.classification,
            decision="rejected",
            decision_reason="Tool marcada como innecesaria por la recomendacion consolidada.",
        )
        for item in artifact.rejected_tools
    )

    approved_tools = [
        _build_blueprint_tool_from_recommendation(
            artifact=artifact,
            entry=item,
        )
        for item in approved_entries
    ]
    digest = build_approved_tools_digest_from_blueprint_tools(
        approved_tools,
        source_session_id=artifact.source_session_id,
        source_blueprint_version=artifact.source_blueprint_version,
        mandatory_tool_keys=[item.tool_key for item in artifact.recommended_tools],
        optional_tool_keys=[item.tool_key for item in artifact.optional_tools if _normalize_text(item.tool_key) in optional_allowlist],
    )
    return approved_tools, review_decisions, digest


def evaluate_tool_recommendation_artifact(
    artifact: ToolRecommendationArtifact,
) -> ToolRecommendationArtifact:
    findings: list[ToolRecommendationFinding] = []
    selected_entries = [*artifact.recommended_tools, *artifact.optional_tools]
    selected_keys = _selected_tool_keys(artifact)
    recommended_keys = {item.tool_key for item in artifact.recommended_tools}
    allowed_tool_keys = _allowed_tool_keys(artifact)
    mandatory_map = _mandatory_capability_map(artifact)
    section_map = {
        "recommended": artifact.recommended_tools,
        "optional": artifact.optional_tools,
        "rejected": artifact.rejected_tools,
    }

    for gap in artifact.coverage_gaps:
        severity = "blocking" if gap.severity == "high" else "warning"
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key=f"coverage-gap:{gap.gap_key}",
                title=gap.title,
                detail=gap.reason or gap.impact,
                severity=severity,
                category="coverage",
                suggested_action=gap.question or "Completa la informacion faltante antes de promover la propuesta.",
            ),
        )

    for item in artifact.requirements_coverage:
        if item.coverage_status == "covered":
            continue
        is_internal = _requirement_can_be_covered_internally(item)
        severity = "warning"
        if item.category == "functional" and item.priority == "high" and not is_internal:
            severity = "blocking"
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key=f"requirement-coverage:{item.requirement_key}",
                title="Cobertura de requisito insuficiente",
                detail=f"{item.requirement_title}: {item.rationale}",
                severity=severity,
                category="coverage",
                affected_tool_keys=list(item.covered_by_tool_keys),
                suggested_action=(
                    "Incluye una tool que cubra explicitamente este requisito o reduce el alcance aprobado."
                ),
            ),
        )

    for item in artifact.design_role_coverage:
        if item.coverage_status == "covered":
            continue
        internally_coverable = _design_role_can_be_covered_internally(item)
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key=f"design-role-coverage:{item.role_key}",
                title="Cobertura de rol de diseño incompleta",
                detail=f"{item.role_title}: {item.rationale}",
                severity=(
                    "warning"
                    if item.coverage_status == "partial" or internally_coverable
                    else "blocking"
                ),
                category="coverage",
                affected_tool_keys=list(item.covered_by_tool_keys),
                suggested_action="Revisa si el diseño necesita una tool adicional o si el rol debe simplificarse en Design.",
            ),
        )

    coverage_gap_keys = {item.gap_key for item in artifact.coverage_gaps}
    for gap in artifact.needs_information:
        severity = "blocking" if gap.severity == "high" else "warning"
        category = "coverage" if gap.severity == "high" else "confidence"
        prefix = "Falta informacion para defender el set minimo de tools."
        if gap.gap_key in coverage_gap_keys:
            continue
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key=f"needs-info:{gap.gap_key}",
                title=gap.title,
                detail=f"{prefix} {gap.reason or gap.impact}".strip(),
                severity=severity,
                category=category,
                suggested_action=gap.question or "Completa el contexto pendiente antes de continuar.",
            ),
        )

    occurrences: dict[str, list[str]] = {}
    for section_name, entries in section_map.items():
        for entry in entries:
            occurrences.setdefault(entry.tool_key, []).append(section_name)
    for tool_key, sections in occurrences.items():
        unique_sections = list(dict.fromkeys(sections))
        if len(unique_sections) < 2:
            continue
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key=f"duplicate-tool:{tool_key}",
                title="Tool duplicada en multiples clasificaciones",
                detail=f"La tool {tool_key} aparece en {', '.join(unique_sections)} y la propuesta deja de ser consistente.",
                severity="blocking",
                category="compatibility",
                affected_tool_keys=[tool_key],
                suggested_action="Deja la tool en una sola clasificacion: obligatoria, opcional o innecesaria.",
            ),
        )

    for mandatory_key, capability in mandatory_map.items():
        if mandatory_key in recommended_keys:
            continue
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key=f"missing-mandatory:{mandatory_key}",
                title="Cobertura minima incompleta",
                detail=capability.reason or f"Falta la capacidad obligatoria {mandatory_key}.",
                severity="blocking",
                category="coverage",
                affected_tool_keys=[mandatory_key],
                suggested_action=f"Incluye {mandatory_key} como tool obligatoria antes de continuar.",
            ),
        )

    for entry in selected_entries:
        if entry.tool_key not in CAPABILITY_CATALOG:
            _append_finding(
                findings,
                ToolRecommendationFinding(
                    finding_key=f"unsupported-tool:{entry.tool_key}",
                    title="Tool fuera del catalogo permitido",
                    detail=f"La tool {entry.tool_key} no existe en el catalogo controlado de HT4.",
                    severity="blocking",
                    category="compatibility",
                    affected_tool_keys=[entry.tool_key],
                    suggested_action="Sustituye la tool por una capability permitida o amplia el catalogo de manera gobernada.",
                ),
            )
            continue

        if entry.tool_key not in allowed_tool_keys:
            _append_finding(
                findings,
                ToolRecommendationFinding(
                    finding_key=f"not-allowed:{entry.tool_key}",
                    title="Tool fuera del shortlist permitido",
                    detail=f"La tool {entry.tool_key} no fue habilitada por el preflight heuristico.",
                    severity="blocking",
                    category="compatibility",
                    affected_tool_keys=[entry.tool_key],
                    suggested_action="Retira la tool o corrige discovery/define/design para justificar su inclusion.",
                ),
            )

        if entry.tool_key in artifact.preflight.forbidden_capabilities:
            _append_finding(
                findings,
                ToolRecommendationFinding(
                    finding_key=f"forbidden:{entry.tool_key}",
                    title="Tool bloqueada por restricciones",
                    detail=f"La tool {entry.tool_key} viola un hard constraint o una regla de poda del preflight.",
                    severity="blocking",
                    category="compatibility",
                    affected_tool_keys=[entry.tool_key],
                    suggested_action="Elimina la tool o revisa la restriccion aprobada que la mantiene fuera.",
                ),
            )

        missing_dependencies = [item for item in entry.dependencies if item not in selected_keys]
        if missing_dependencies:
            _append_finding(
                findings,
                ToolRecommendationFinding(
                    finding_key=f"missing-dependencies:{entry.tool_key}",
                    title="Dependencias incompletas",
                    detail=(
                        f"La tool {entry.tool_key} requiere {', '.join(missing_dependencies)} y hoy no estan seleccionadas."
                    ),
                    severity="blocking",
                    category="coverage",
                    affected_tool_keys=[entry.tool_key, *missing_dependencies],
                    suggested_action="Agrega las dependencias faltantes o reclasifica la tool para evitar una propuesta inconsistente.",
                ),
            )

        selected_incompatibilities = [item for item in entry.incompatibilities if item in selected_keys]
        if selected_incompatibilities:
            _append_finding(
                findings,
                ToolRecommendationFinding(
                    finding_key=f"incompatible:{entry.tool_key}",
                    title="Tools incompatibles en la misma propuesta",
                    detail=(
                        f"La tool {entry.tool_key} declara incompatibilidad con {', '.join(selected_incompatibilities)}."
                    ),
                    severity="blocking",
                    category="compatibility",
                    affected_tool_keys=[entry.tool_key, *selected_incompatibilities],
                    suggested_action="Retira una de las tools incompatibles o ajusta el contrato antes de aprobar.",
                ),
            )

        redundant_selected = [item for item in entry.redundant_with if item in selected_keys]
        if redundant_selected:
            _append_finding(
                findings,
                ToolRecommendationFinding(
                    finding_key=f"redundant:{entry.tool_key}",
                    title="Redundancia detectada",
                    detail=f"La tool {entry.tool_key} solapa capacidades con {', '.join(redundant_selected)}.",
                    severity="warning",
                    category="redundancy",
                    affected_tool_keys=[entry.tool_key, *redundant_selected],
                    suggested_action="Conserva solo la tool con mejor cobertura o menor costo operativo.",
                ),
            )

    if "transactional_write" in selected_keys and "approval_gate" not in recommended_keys:
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key="write-without-approval-gate",
                title="Escritura sin approval gate obligatorio",
                detail="La propuesta incluye escritura transaccional, pero no deja approval_gate como control obligatorio.",
                severity="blocking",
                category="governance",
                affected_tool_keys=["transactional_write", "approval_gate"],
                suggested_action="Promueve approval_gate a obligatoria antes de permitir side effects.",
            ),
        )

    if "human_approval_required" in artifact.preflight.hard_constraints and "approval_gate" not in recommended_keys:
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key="human-approval-missing",
                title="Hard constraint sin enforcement",
                detail="El caso exige aprobacion humana, pero la propuesta no incluye una tool obligatoria para instrumentarla.",
                severity="blocking",
                category="governance",
                affected_tool_keys=["approval_gate"],
                suggested_action="Incluye approval_gate como tool obligatoria o reduce formalmente el nivel de autonomia.",
            ),
        )

    if not artifact.preflight.required_write_actions and "transactional_write" in selected_keys:
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key="write-without-write-actions",
                title="Escritura sobreaprovisionada",
                detail="No hay acciones de escritura aprobadas en el contexto, pero la propuesta mantiene transactional_write.",
                severity="warning",
                category="minimality",
                affected_tool_keys=["transactional_write"],
                suggested_action="Retira la tool de escritura o documenta la accion concreta que la hace necesaria.",
            ),
        )

    if artifact.preflight.case_classification == "lean_blueprint_builder" and selected_entries:
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key="inline-first-overprovisioned",
                title="Caso inline-first con tools innecesarias",
                detail="La heuristica clasifica el caso como inline-first, pero la propuesta conserva tools adicionales.",
                severity="warning",
                category="minimality",
                affected_tool_keys=sorted(selected_keys),
                suggested_action="Verifica si el caso puede resolverse solo con contexto inline y elimina herramientas sobrantes.",
            ),
        )

    for entry in artifact.recommended_tools:
        if entry.tool_key not in mandatory_map:
            _append_finding(
                findings,
                ToolRecommendationFinding(
                    finding_key=f"promoted-candidate:{entry.tool_key}",
                    title="Tool promovida a obligatoria desde una senal candidata",
                    detail=(
                        f"La tool {entry.tool_key} no era mandatory en el preflight y HT4 la marca para revision de minimalidad."
                    ),
                    severity="warning",
                    category="minimality",
                    affected_tool_keys=[entry.tool_key],
                    suggested_action="Confirma si realmente es obligatoria o muevela a opcional para no sobredimensionar el agente.",
                ),
            )

    base_confidence = max(0.0, min(artifact.confidence.overall, 1.0))
    blocking_count = sum(1 for item in findings if item.severity == "blocking")
    warning_count = sum(1 for item in findings if item.severity == "warning")
    confidence_penalty = _evaluation_penalty_for_findings(findings)
    historical_penalty = _historical_evaluation_penalty_from_rationale(artifact.confidence.rationale)
    if artifact.evaluation.findings and historical_penalty:
        base_confidence = min(
            0.92,
            base_confidence + historical_penalty,
        )
    if not selected_entries and artifact.preflight.case_classification == "lean_blueprint_builder" and not artifact.needs_information:
        adjusted_confidence = max(base_confidence, 0.72)
    else:
        adjusted_confidence = max(0.18, min(base_confidence - confidence_penalty, 0.92))
        if blocking_count == 0 and warning_count == 0:
            adjusted_confidence = min(0.92, adjusted_confidence + 0.04)

    promotion_blocked = bool(blocking_count)
    if adjusted_confidence < 0.6 and blocking_count == 0:
        _append_finding(
            findings,
            ToolRecommendationFinding(
                finding_key="confidence-below-threshold",
                title="Confianza insuficiente para promover la propuesta",
                detail=(
                    "La recomendacion no tiene contexto suficiente o mantiene demasiadas ambiguedades para avanzar con seguridad."
                ),
                severity="warning",
                category="confidence",
                suggested_action="Completa la informacion faltante o simplifica el alcance antes de pasar a Memoria.",
            ),
        )

    recommended_actions: list[str] = []
    for finding in findings:
        if finding.suggested_action:
            _append_unique(recommended_actions, finding.suggested_action)
    if not recommended_actions and promotion_blocked:
        recommended_actions.append("Resuelve los findings HT4 antes de promover la propuesta.")
    if not recommended_actions:
        recommended_actions.append("La propuesta supera HT4 y queda lista para revision humana en HT5.")

    confidence_rationale = artifact.confidence.rationale.strip()
    confidence_suffix = (
        f"HT4 encontro {blocking_count} bloqueos y {warning_count} alertas; "
        f"confidence ajustada a {adjusted_confidence:.2f}."
    )
    if confidence_rationale:
        confidence_rationale = f"{confidence_rationale} {confidence_suffix}"
    else:
        confidence_rationale = confidence_suffix

    evaluation = ToolRecommendationEvaluation(
        overall_status=ReviewState.blocked if promotion_blocked else (ReviewState.complete if not findings else ReviewState.partial),
        coverage_status=_status_for_categories(findings, "coverage"),
        minimality_status=_status_for_categories(findings, "minimality", "redundancy"),
        compatibility_status=_status_for_categories(findings, "compatibility"),
        governance_status=_status_for_categories(findings, "governance"),
        promotion_blocked=promotion_blocked,
        findings=findings,
        recommended_actions=recommended_actions[:6],
        summary=(
            "HT4 bloqueo la propuesta hasta resolver cobertura, compatibilidad o confianza."
            if promotion_blocked
            else "HT4 no encontro bloqueos estructurales; la propuesta queda lista para revision y aprobacion humana."
        ),
    )

    evaluated = artifact.model_copy(
        update={
            "confidence": ToolRecommendationConfidence(
                overall=adjusted_confidence,
                band=_confidence_band(adjusted_confidence),
                rationale=confidence_rationale,
            ),
            "evaluation": evaluation,
            "review_state": evaluation.overall_status,
        }
    )
    return _attach_recommendation_contract_seeds(evaluated)


def build_tool_recommendation_prompt_input(artifact: ToolRecommendationArtifact) -> ToolRecommendationPromptInput:
    allowed_tool_keys = _allowed_tool_keys(artifact)
    family_by_key = {item.family_key: item for item in artifact.preflight.candidate_tool_families}
    candidate_tools: list[ToolRecommendationPromptToolOption] = []
    seen: set[str] = set()

    def register_option(tool_key: str, *, family_key: str, family_status: str, reason: str, notes: list[str]) -> None:
        if tool_key in seen or tool_key not in CAPABILITY_CATALOG:
            return
        enum_key = _tool_key_to_enum(tool_key)
        if enum_key is None:
            return
        seen.add(tool_key)
        catalog = CAPABILITY_CATALOG[tool_key]
        candidate_tools.append(
            ToolRecommendationPromptToolOption(
                tool_key=enum_key,
                tool_label=catalog["tool_label"],
                family_key=family_key,
                family_status=family_status,
                capability_covered=catalog["capability_covered"],
                reason=reason,
                selection_notes=notes,
            )
        )

    for capability in artifact.preflight.mandatory_capabilities:
        if capability.capability_key not in CAPABILITY_CATALOG:
            continue
        family_key = str(CAPABILITY_CATALOG[capability.capability_key]["family"])
        family = family_by_key.get(family_key)
        register_option(
            capability.capability_key,
            family_key=family_key,
            family_status=family.status if family is not None else "required",
            reason=capability.reason,
            notes=["mandatory_by_preflight", *capability.source_evidence[:3]],
        )

    for family in artifact.preflight.candidate_tool_families:
        if family.status == "excluded":
            continue
        for tool_key in family.suggested_tool_keys:
            if tool_key not in allowed_tool_keys:
                continue
            register_option(
                tool_key,
                family_key=family.family_key,
                family_status=family.status,
                reason=family.reason,
                notes=[*family.matched_signals[:3], *family.rejected_by_constraints[:2]],
            )

    mandatory_tool_keys = [
        enum_key
        for enum_key in (_tool_key_to_enum(item.capability_key) for item in artifact.preflight.mandatory_capabilities)
        if enum_key is not None
    ]
    forbidden_tool_keys = [
        enum_key
        for enum_key in (_tool_key_to_enum(item) for item in artifact.preflight.forbidden_capabilities)
        if enum_key is not None
    ]

    compact_evidence = [
        f"goal={artifact.preflight.agent_goal or 'unknown'}",
        f"workflow={artifact.context_digest.workflow_summary or 'unknown'}",
        f"constraints={artifact.context_digest.constraints_summary or 'none'}",
        f"writes={', '.join(artifact.preflight.required_write_actions) or 'none'}",
        f"sources={', '.join(artifact.preflight.required_information_sources) or 'inline_only'}",
        f"approvals={', '.join(artifact.preflight.approval_boundaries) or 'none'}",
    ]

    return ToolRecommendationPromptInput(
        source_session_id=artifact.source_session_id,
        source_blueprint_version=artifact.source_blueprint_version,
        case_classification=artifact.preflight.case_classification,
        agent_goal=artifact.preflight.agent_goal,
        primary_user=artifact.preflight.primary_user,
        workflow_summary=artifact.context_digest.workflow_summary,
        constraints_summary=artifact.context_digest.constraints_summary,
        source_refs=list(artifact.context_digest.source_refs),
        core_workflows=list(artifact.preflight.core_workflows),
        interaction_modes=list(artifact.preflight.interaction_modes),
        required_information_sources=list(artifact.preflight.required_information_sources),
        required_write_actions=list(artifact.preflight.required_write_actions),
        approval_boundaries=list(artifact.preflight.approval_boundaries),
        hard_constraints=list(artifact.preflight.hard_constraints),
        mandatory_tool_keys=mandatory_tool_keys,
        forbidden_tool_keys=forbidden_tool_keys,
        candidate_tools=candidate_tools,
        requirements_coverage=list(artifact.requirements_coverage),
        design_role_coverage=list(artifact.design_role_coverage),
        existing_gaps=list(artifact.needs_information),
        compact_evidence=compact_evidence,
    )


def annotate_tool_recommendation_status(
    artifact: ToolRecommendationArtifact,
    *,
    discovery: DiscoveryArtifact | None = None,
    canvas: CanvasArtifact | None = None,
    blueprint: BlueprintArtifact | None = None,
    definition_artifact: RequirementsDefinitionOutput | None = None,
    design_artifact: DesignRecommendationArtifact | None = None,
    current_blueprint_version: int | None = None,
) -> ToolRecommendationArtifact:
    status_artifact = (
        artifact
        if artifact.approved_tools_digest is not None
        else evaluate_tool_recommendation_artifact(artifact)
    )
    stored_fingerprint = (
        status_artifact.context_digest.digest_sha256
        or build_tool_recommendation_context_fingerprint(status_artifact)
    )
    stale_reasons: list[str] = []

    if discovery is not None and canvas is not None and blueprint is not None:
        current_artifact = build_placeholder_tool_recommendation(
            session_id=artifact.source_session_id or UUID(int=0),
            discovery=discovery,
            canvas=canvas,
            blueprint=blueprint,
            definition_artifact=definition_artifact,
            design_artifact=design_artifact,
            instructions=artifact.generation_instructions,
            blueprint_version_number=current_blueprint_version,
        )
        current_fingerprint = (
            current_artifact.context_digest.digest_sha256
            or build_tool_recommendation_context_fingerprint(current_artifact)
        )
        if stored_fingerprint != current_fingerprint:
            stale_reasons.append("tool_recommendation_context_changed")
            if artifact.approved_tools_digest is not None:
                stale_reasons.append("approved_tools_digest_outdated")

    annotated = status_artifact.model_copy(
        update={
            "context_digest": status_artifact.context_digest.model_copy(update={"digest_sha256": stored_fingerprint}),
            "current_blueprint_version": current_blueprint_version,
            "is_stale": bool(stale_reasons),
            "stale_reasons": stale_reasons,
        }
    )
    return auto_reconcile_tool_recommendation_artifact(_attach_recommendation_contract_seeds(annotated))


def _entry_from_llm_decision(
    *,
    blueprint: BlueprintArtifact,
    tool_key: str,
    classification: str,
    decision_reason: str,
    source_evidence: list[str],
    confidence: float,
    dependencies: list[str],
    incompatibilities: list[str],
    redundant_with: list[str],
) -> ToolRecommendationEntry:
    return _build_entry(
        blueprint=blueprint,
        capability_key=tool_key,
        classification=classification,
        decision_reason=decision_reason,
        source_evidence=source_evidence,
        confidence=max(0.0, min(confidence, 1.0)),
        dependencies=dependencies,
        incompatibilities=incompatibilities,
        redundant_with=redundant_with,
    )


def merge_llm_tool_recommendation(
    *,
    preflight_artifact: ToolRecommendationArtifact,
    llm_output: ToolRecommendationLLMOutput,
    blueprint: BlueprintArtifact,
) -> ToolRecommendationArtifact:
    allowed_tool_keys = _allowed_tool_keys(preflight_artifact)
    mandatory_map = _mandatory_capability_map(preflight_artifact)
    mandatory_tool_keys = set(mandatory_map)
    forbidden_tool_keys = {
        item
        for item in preflight_artifact.preflight.forbidden_capabilities
        if item in CAPABILITY_CATALOG
    }

    recommended_tools: list[ToolRecommendationEntry] = []
    optional_tools: list[ToolRecommendationEntry] = []
    rejected_tools: list[ToolRecommendationEntry] = []
    needs_information: list[ToolRecommendationGap] = []
    coverage_gaps: list[ToolRecommendationGap] = []
    seen: set[str] = set()

    for gap in preflight_artifact.needs_information:
        _append_gap(needs_information, gap)
    for gap in preflight_artifact.coverage_gaps:
        _append_gap(coverage_gaps, gap)
    for gap in llm_output.needs_information:
        _append_gap(needs_information, gap)
    for gap in llm_output.coverage_gaps:
        _append_gap(coverage_gaps, gap)

    for decision in llm_output.tool_decisions:
        tool_key = decision.tool_key.value
        if tool_key in seen or tool_key not in allowed_tool_keys or tool_key in forbidden_tool_keys:
            continue
        seen.add(tool_key)
        classification = decision.classification
        if tool_key in mandatory_tool_keys:
            classification = "mandatory"
        dependencies = [
            item.value
            for item in decision.dependencies
            if item.value in allowed_tool_keys and item.value != tool_key
        ]
        redundant_with = [
            item.value
            for item in decision.redundant_with
            if item.value in allowed_tool_keys and item.value != tool_key
        ]
        entry = _entry_from_llm_decision(
            blueprint=blueprint,
            tool_key=tool_key,
            classification=classification,
            decision_reason=decision.decision_reason or CAPABILITY_CATALOG[tool_key]["capability_covered"],
            source_evidence=decision.source_evidence or ["preflight.compact_evidence"],
            confidence=decision.confidence,
            dependencies=dependencies,
            incompatibilities=list(decision.incompatibilities),
            redundant_with=redundant_with,
        )
        if classification == "mandatory":
            _append_entry(recommended_tools, entry)
        elif classification == "optional":
            _append_entry(optional_tools, entry)
        else:
            _append_entry(rejected_tools, entry)

    base_recommended_by_key = {item.tool_key: item for item in preflight_artifact.recommended_tools}
    for mandatory_key, capability in mandatory_map.items():
        if any(item.tool_key == mandatory_key for item in recommended_tools):
            continue
        fallback = base_recommended_by_key.get(mandatory_key)
        if fallback is not None:
            _append_entry(recommended_tools, fallback)
            continue
        _append_entry(
            recommended_tools,
            _build_entry(
                blueprint=blueprint,
                capability_key=mandatory_key,
                classification="mandatory",
                decision_reason=capability.reason,
                source_evidence=capability.source_evidence,
                confidence=capability.confidence,
            ),
        )

    for item in preflight_artifact.rejected_tools:
        if item.tool_key == "broad_write_backoffice":
            _append_entry(rejected_tools, item)

    normalized_coverage_gaps = list(coverage_gaps)
    for gap in needs_information:
        if gap.severity == "high":
            _append_gap(normalized_coverage_gaps, gap)

    confidence = llm_output.confidence
    if not llm_output.tool_decisions and not llm_output.needs_information and not llm_output.coverage_gaps:
        confidence = preflight_artifact.confidence

    summary = llm_output.summary.strip() if llm_output.summary.strip() else preflight_artifact.summary
    if not summary:
        summary = (
            f"Recomendacion minima de tools consolidada con {len(recommended_tools)} obligatorias, "
            f"{len(optional_tools)} opcionales y {len(rejected_tools)} innecesarias."
        )

    selected_tool_keys = {item.tool_key for item in [*recommended_tools, *optional_tools]}
    requirements_coverage = [
        item.model_copy(
            update={
                "coverage_status": _coverage_status(item.covered_by_tool_keys, selected_tool_keys),
                "rationale": _coverage_rationale(
                    covered_by_tool_keys=item.covered_by_tool_keys,
                    selected_tool_keys=selected_tool_keys,
                    subject=item.requirement_title or item.requirement_key,
                ),
            }
        )
        for item in preflight_artifact.requirements_coverage
    ]
    design_role_coverage = [
        item.model_copy(
            update={
                "coverage_status": _coverage_status(item.covered_by_tool_keys, selected_tool_keys),
                "rationale": _coverage_rationale(
                    covered_by_tool_keys=item.covered_by_tool_keys,
                    selected_tool_keys=selected_tool_keys,
                    subject=item.role_title or item.role_key,
                ),
            }
        )
        for item in preflight_artifact.design_role_coverage
    ]

    merged = preflight_artifact.model_copy(
        update={
            "recommended_tools": recommended_tools,
            "optional_tools": optional_tools,
            "rejected_tools": rejected_tools,
            "requirements_coverage": requirements_coverage,
            "design_role_coverage": design_role_coverage,
            "needs_information": needs_information,
            "coverage_gaps": normalized_coverage_gaps,
            "confidence": confidence,
            "review_state": ReviewState.partial,
            "summary": summary,
        }
    )
    return auto_reconcile_tool_recommendation_artifact(_attach_recommendation_contract_seeds(merged))


def auto_reconcile_tool_recommendation_artifact(
    artifact: ToolRecommendationArtifact,
) -> ToolRecommendationArtifact:
    """Auto-reconcilia artefactos de herramientas, auto-remediando contratos y filtrando ruido de infraestructura hacia ACP."""
    def _is_tool_noise(text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            kw in lower
            for kw in (
                "especificación técnica de los mecanismos",
                "especificacion tecnica de los mecanismos",
                "apis con las plataformas de correo",
                "sistema de gestión de tickets",
                "sistema de gestion de tickets",
                "openapi",
                "swagger",
                "credenciales de api",
                "url de endpoint",
                "sistema fuente no identificado",
                "system_of_record_unspecified",
            )
        )

    cleaned_needs_info = [
        gap for gap in (artifact.needs_information or [])
        if not _is_tool_noise(f"{gap.gap_key} {gap.title} {gap.question}")
    ]
    cleaned_coverage_gaps = [
        gap for gap in (artifact.coverage_gaps or [])
        if not _is_tool_noise(f"{gap.gap_key} {gap.title} {gap.question}")
    ]
    cleaned_missing_info = [
        info for info in (getattr(artifact, "missing_information", []) or [])
        if not _is_tool_noise(info)
    ]
    clean_findings = []
    for finding in getattr(artifact, "critic_findings", []):
        combined = f"{finding.title} {finding.detail} {finding.finding_key}".lower()
        if _is_tool_noise(combined):
            continue
        clean_findings.append(finding)

    updated = artifact.model_copy(
        update={
            "needs_information": cleaned_needs_info,
            "coverage_gaps": cleaned_coverage_gaps,
            "missing_information": cleaned_missing_info,
            "critic_findings": clean_findings,
        }
    )
    return _attach_recommendation_contract_seeds(updated)
