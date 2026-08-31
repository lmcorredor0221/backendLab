from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any

from app.services.llm_runtime.builder_contracts import (
    DeferredResolutionItem,
    DiscoveryAnalysisOutput,
    PrioritizedQuestion,
    StructuredInsight,
)


DISCOVERY_REQUIRED_FIELD_PATHS = {
    "problem_statement",
    "current_user",
    "current_process",
    "desired_outcome",
    "autonomy_level",
    "operational_baseline.current_time_spent",
    "operational_baseline.current_cost",
    "operational_baseline.frequent_errors",
    "operational_baseline.automation_opportunities",
    "mvp_definition.north_star_metric",
    "mvp_definition.v1_scope",
    "mvp_definition.out_of_scope",
    "mvp_definition.non_delegable_decisions",
}


DISCOVERY_ALLOWED_BLOCKING_STAGES = {"discover", "define"}


DISCOVERY_DEFERRED_TERMS = (
    "api",
    "apis",
    "arquitectura",
    "aws",
    "azure",
    "base de datos",
    "bandeja",
    "canal",
    "caso complejo",
    "cloud",
    "configuracion",
    "configuraciones",
    "consentimiento",
    "contrato",
    "contratos",
    "costo por contacto",
    "credencial",
    "credenciales",
    "crm",
    "database",
    "deployment",
    "despliegue",
    "documental",
    "documentales",
    "documento",
    "documentos",
    "embedding",
    "endpoint",
    "erp",
    "escalamiento",
    "esquema",
    "framework",
    "fuente",
    "fuentes",
    "gcp",
    "handoff",
    "herramienta",
    "herramientas",
    "horario",
    "idioma",
    "idiomas",
    "implementacion",
    "infraestructura",
    "integracion",
    "integraciones",
    "knowledge",
    "lenguaje",
    "marca",
    "memoria",
    "metodo de integracion",
    "politica de retencion",
    "privacidad",
    "proteccion de datos",
    "postgres",
    "pii",
    "rag",
    "regla de escalamiento",
    "retencion",
    "retrieval",
    "runtime",
    "schema",
    "secret",
    "secrets",
    "sla",
    "sqlite",
    "sql",
    "tasa de escalamiento",
    "tecnologia",
    "tecnologias",
    "ticketing",
    "tono",
    "volumen",
    "webhook",
)

DISCOVERY_BUSINESS_CLARIFICATION_TERMS = (
    "alcance",
    "alcance exacto",
    "alcance inicial",
    "alcance limitado",
    "aceptacion",
    "criterio de aceptacion",
    "criterios de aceptacion",
    "grupo de usuarios",
    "mvp",
    "primera version",
    "primera versión",
    "segmento prioritario",
    "usuario prioritario",
    "usuarios prioritarios",
    "metrica",
    "metricas",
    "métrica",
    "métricas",
    "tiempo actual",
    "costo actual",
    "frecuencia",
    "volumen",
    "casos por semana",
)

DISCOVERY_TOOLS_TARGET_TERMS = (
    "api",
    "apis",
    "canal",
    "crm",
    "erp",
    "herramienta",
    "herramientas",
    "integracion",
    "integraciones",
    "ticketing",
    "webhook",
)

DISCOVERY_MEMORY_TARGET_TERMS = (
    "chunking",
    "fuente",
    "fuentes",
    "knowledge",
    "memoria",
    "rag",
    "retencion",
    "retrieval",
)

DISCOVERY_ESTIMATE_TARGET_TERMS = (
    "costo",
    "costos",
    "metrica",
    "metricas",
    "métrica",
    "métricas",
    "roi",
    "tiempo",
    "volumen",
)

DISCOVERY_IMPLEMENTATION_TARGET_TERMS = (
    "aws",
    "azure",
    "base de datos",
    "cloud",
    "credencial",
    "credenciales",
    "deployment",
    "despliegue",
    "endpoint",
    "framework",
    "gcp",
    "infraestructura",
    "postgres",
    "runtime",
    "schema",
    "secret",
    "secrets",
    "sqlite",
    "stack",
    "tecnologia",
    "tecnologias",
)

TECHNICAL_IMPLEMENTATION_TERMS = (
    "aws",
    "azure",
    "base de datos",
    "ci/cd",
    "cloud",
    "configuracion",
    "configuraciones",
    "credencial",
    "credenciales",
    "database",
    "deployment",
    "despliegue",
    "endpoint",
    "framework",
    "gcp",
    "hosting",
    "infraestructura",
    "lenguaje",
    "modelo de datos",
    "postgres",
    "runtime",
    "schema",
    "secret",
    "secrets",
    "sqlite",
    "stack",
    "tecnologia",
    "tecnologias",
)

TOOLS_STAGE_ALLOWED_TERMS = (
    "api",
    "apis",
    "canal",
    "contrato funcional",
    "crm",
    "erp",
    "herramienta",
    "herramientas",
    "integracion",
    "integraciones",
    "ticketing",
    "tool",
    "tools",
    "webhook",
)

MEMORY_STAGE_ALLOWED_TERMS = (
    "chunking",
    "fuente",
    "fuentes",
    "knowledge",
    "memoria",
    "rag",
    "retencion",
    "retrieval",
)

LEAN_STAGE_KEYS = {"discover", "define", "design", "tools", "memory", "estimate"}
PRODUCT_STAGE_KEYS = {"blueprint", "blueprint_free", "blueprint_pro", "validate", "package"}
ACP_STAGE_KEYS = {"acp", "package", "implementation", "implementation_questions", "construction"}
CANONICAL_DELEGATION_TARGET_STAGES = LEAN_STAGE_KEYS | PRODUCT_STAGE_KEYS | ACP_STAGE_KEYS
DELEGATION_TARGET_STAGE_ALIASES = {
    "descubrir": "discover",
    "discovery": "discover",
    "definir": "define",
    "disenar": "design",
    "diseñar": "design",
    "herramientas": "tools",
    "memoria": "memory",
    "estimar": "estimate",
    "estimacion": "estimate",
    "validar": "validate",
    "validacion": "validate",
    "blueprint_basic": "blueprint",
    "blueprint_basico": "blueprint",
    "blueprint_free": "blueprint",
    "blueprint_pro": "blueprint_pro",
    "bp": "blueprint",
    "bp_pro": "blueprint_pro",
    "empaquetado": "package",
    "implementacion": "implementation",
}


@dataclass(frozen=True)
class StageQuestionPolicy:
    stage: str
    deferred_terms: tuple[str, ...]
    allowed_terms: tuple[str, ...] = ()
    max_blocking: int = 4
    max_nonblocking: int = 6


@dataclass(frozen=True)
class StageQuestionDecision:
    status: str
    reason: str = ""
    deferral_target_stage: str = ""


STAGE_POLICIES: dict[str, StageQuestionPolicy] = {
    "discover": StageQuestionPolicy(
        stage="discover",
        deferred_terms=DISCOVERY_DEFERRED_TERMS,
        max_blocking=3,
        max_nonblocking=4,
    ),
    "define": StageQuestionPolicy(
        stage="define",
        deferred_terms=TECHNICAL_IMPLEMENTATION_TERMS,
        max_blocking=4,
        max_nonblocking=5,
    ),
    "design": StageQuestionPolicy(
        stage="design",
        deferred_terms=TECHNICAL_IMPLEMENTATION_TERMS,
        max_blocking=4,
        max_nonblocking=5,
    ),
    "tools": StageQuestionPolicy(
        stage="tools",
        deferred_terms=tuple(term for term in TECHNICAL_IMPLEMENTATION_TERMS if term not in TOOLS_STAGE_ALLOWED_TERMS),
        allowed_terms=TOOLS_STAGE_ALLOWED_TERMS,
        max_blocking=4,
        max_nonblocking=5,
    ),
    "memory": StageQuestionPolicy(
        stage="memory",
        deferred_terms=tuple(term for term in TECHNICAL_IMPLEMENTATION_TERMS if term not in MEMORY_STAGE_ALLOWED_TERMS),
        allowed_terms=MEMORY_STAGE_ALLOWED_TERMS,
        max_blocking=4,
        max_nonblocking=5,
    ),
    "validate": StageQuestionPolicy(
        stage="validate",
        deferred_terms=TECHNICAL_IMPLEMENTATION_TERMS,
        max_blocking=4,
        max_nonblocking=5,
    ),
    "estimate": StageQuestionPolicy(
        stage="estimate",
        deferred_terms=TECHNICAL_IMPLEMENTATION_TERMS,
        max_blocking=3,
        max_nonblocking=5,
    ),
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.split())


def _normalize_stage_target(value: Any) -> str:
    normalized = _normalize_text(value).replace("-", "_").replace(" ", "_")
    return DELEGATION_TARGET_STAGE_ALIASES.get(normalized, normalized)


def _is_allowed_delegation_target(stage: str) -> bool:
    return _normalize_stage_target(stage) in CANONICAL_DELEGATION_TARGET_STAGES


def _question_text(question: PrioritizedQuestion | Mapping[str, Any] | str) -> str:
    if isinstance(question, PrioritizedQuestion):
        return " ".join(
            [
                question.key,
                question.question,
                question.rationale,
                question.suggested_answer,
                " ".join(question.blocking_stages),
            ]
        )
    if isinstance(question, Mapping):
        return " ".join(
            str(question.get(key) or "")
            for key in (
                "key",
                "title",
                "question",
                "question_text",
                "rationale",
                "reason",
                "suggested_answer",
                "stage_scope",
                "deferral_target_stage",
            )
        )
    return str(question or "")


def _blocking_stages(question: PrioritizedQuestion | Mapping[str, Any] | str) -> set[str]:
    if isinstance(question, PrioritizedQuestion):
        return {_normalize_stage_target(stage) for stage in question.blocking_stages if _normalize_text(stage)}
    if isinstance(question, Mapping):
        raw = question.get("blocking_stages") or []
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, Iterable):
            return {_normalize_stage_target(stage) for stage in raw if _normalize_text(stage)}
    return set()


def _field_value(question: Any, key: str) -> str:
    if isinstance(question, Mapping):
        return _normalize_text(question.get(key))
    return _normalize_text(getattr(question, key, ""))


def _priority(question: Any) -> str:
    return _field_value(question, "priority") or "medium"


def _is_blocking_question(question: Any) -> bool:
    if isinstance(question, Mapping):
        if bool(question.get("blocking", False)):
            return True
    elif bool(getattr(question, "blocking", False)):
        return True
    return _priority(question) == "high" or bool(_blocking_stages(question))


def _is_acp_stage(stage: str) -> bool:
    return _normalize_stage_target(stage) in ACP_STAGE_KEYS


def _is_required_field_reference(value: Any) -> bool:
    normalized = _normalize_text(value)
    for prefix in ("falta informacion:", "missing:", "question:"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix).strip()
    normalized = normalized.replace(" > ", ".").replace(" ", "_")
    return normalized in DISCOVERY_REQUIRED_FIELD_PATHS


def _is_discovery_current_stage_required_gap(value: Any) -> bool:
    return _is_required_field_reference(value)


def _is_discovery_delegable_business_clarification(value: Any) -> bool:
    normalized = _normalize_text(value)
    return bool(normalized) and not _is_discovery_current_stage_required_gap(normalized) and any(
        term in normalized for term in DISCOVERY_BUSINESS_CLARIFICATION_TERMS
    )


def _discovery_deferral_target(value: Any) -> str:
    normalized = _normalize_text(value)
    if any(term in normalized for term in DISCOVERY_IMPLEMENTATION_TARGET_TERMS):
        return "acp"
    if any(term in normalized for term in DISCOVERY_MEMORY_TARGET_TERMS):
        return "memory"
    if any(term in normalized for term in DISCOVERY_TOOLS_TARGET_TERMS):
        return "tools"
    if any(term in normalized for term in DISCOVERY_ESTIMATE_TARGET_TERMS):
        return "estimate"
    return "define"


def _deferred_resolution_kind(value: Any) -> str:
    normalized = _normalize_text(value)
    if _is_discovery_delegable_business_clarification(normalized):
        return "business_clarification"
    if any(term in normalized for term in DISCOVERY_MEMORY_TARGET_TERMS):
        return "memory_dependency"
    if any(term in normalized for term in DISCOVERY_TOOLS_TARGET_TERMS):
        return "tool_dependency"
    if any(term in normalized for term in DISCOVERY_IMPLEMENTATION_TARGET_TERMS):
        return "implementation_decision"
    if any(term in normalized for term in DISCOVERY_ESTIMATE_TARGET_TERMS):
        return "estimation_input"
    return "later_stage_question"


def _default_discovery_recommendation(target_stage: str) -> str:
    if target_stage == "tools":
        return "Usar la inferencia actual en Free y confirmar la capacidad requerida al promover Herramientas."
    if target_stage == "memory":
        return "Continuar con la inferencia actual y confirmar la estrategia de memoria cuando llegue la etapa Memoria."
    if target_stage == "estimate":
        return "Mantener la inferencia actual y confirmar metricas o volumen al llegar a Estimar."
    if target_stage == "acp":
        return "Documentar la decision y resolverla en ACP o implementacion cuando exista contexto tecnico suficiente."
    return "Usar la inferencia actual en Free y confirmar la aclaracion en la siguiente etapa."


def _build_deferred_resolution_item(
    *,
    question_text: str,
    source_stage: str,
    target_stage: str,
    reason: str,
    recommendation: str,
    source_refs: list[str],
) -> DeferredResolutionItem:
    return DeferredResolutionItem(
        source_stage=source_stage,
        target_stage=target_stage,
        kind=_deferred_resolution_kind(question_text),
        question=question_text,
        recommendation=recommendation or _default_discovery_recommendation(target_stage),
        reason=reason,
        source_refs=source_refs,
    )


def is_discovery_later_stage_question(question: PrioritizedQuestion | Mapping[str, Any] | str) -> bool:
    if isinstance(question, PrioritizedQuestion) and question.key.startswith("question:"):
        return False
    text = _normalize_text(_question_text(question))
    if _is_discovery_current_stage_required_gap(text):
        return False
    if any(term in text for term in DISCOVERY_DEFERRED_TERMS):
        return True
    if _is_discovery_delegable_business_clarification(text):
        return True
    stages = _blocking_stages(question)
    return bool(stages) and stages.isdisjoint(DISCOVERY_ALLOWED_BLOCKING_STAGES)


def classify_stage_question(stage: str, question: Any) -> StageQuestionDecision:
    normalized_stage = _normalize_stage_target(stage) or "discover"
    if _is_acp_stage(normalized_stage):
        return StageQuestionDecision(status="allowed_now")

    text = _normalize_text(_question_text(question))
    question_key = _field_value(question, "key")
    if not text:
        return StageQuestionDecision(status="reject_as_noise", reason="Pregunta vacia.")
    if _is_discovery_current_stage_required_gap(text) or _is_discovery_current_stage_required_gap(question_key):
        return StageQuestionDecision(status="allowed_now", reason="Campo obligatorio de la etapa.")

    explicit_scope = _normalize_stage_target(_field_value(question, "stage_scope"))
    explicit_deferral = _normalize_stage_target(_field_value(question, "deferral_target_stage"))
    if explicit_deferral and explicit_deferral not in {normalized_stage, "none"}:
        if not _is_allowed_delegation_target(explicit_deferral):
            return StageQuestionDecision(
                status="reject_as_noise",
                reason=f"Destino de diferimiento no gobernado: {explicit_deferral}.",
            )
        return StageQuestionDecision(
            status="defer_to_acp" if explicit_deferral in ACP_STAGE_KEYS else "defer_to_next_stage",
            reason=f"La pregunta declara diferimiento a {explicit_deferral}.",
            deferral_target_stage=explicit_deferral,
        )
    if explicit_scope and explicit_scope not in {normalized_stage, "cross_stage", "transversal"}:
        if not _is_allowed_delegation_target(explicit_scope):
            return StageQuestionDecision(
                status="reject_as_noise",
                reason=f"Alcance de etapa no gobernado: {explicit_scope}.",
            )
        return StageQuestionDecision(
            status="defer_to_next_stage",
            reason=f"La pregunta pertenece a {explicit_scope}, no a {normalized_stage}.",
            deferral_target_stage=explicit_scope,
        )

    if normalized_stage == "discover" and is_discovery_later_stage_question(question):
        target_stage = _discovery_deferral_target(question)
        return StageQuestionDecision(
            status="defer_to_acp" if target_stage == "acp" else "defer_to_next_stage",
            reason="Discover solo acepta dudas de problema, usuario, proceso actual y resultado esperado.",
            deferral_target_stage=target_stage,
        )

    policy = STAGE_POLICIES.get(normalized_stage)
    if policy is None:
        return StageQuestionDecision(status="allowed_now")
    if policy.allowed_terms and any(term in text for term in policy.allowed_terms):
        return StageQuestionDecision(status="allowed_now")
    if any(term in text for term in policy.deferred_terms):
        return StageQuestionDecision(
            status="defer_to_acp",
            reason="La pregunta depende de decisiones tecnicas de implementacion.",
            deferral_target_stage="acp",
        )

    stages = _blocking_stages(question)
    unmanaged_targets = sorted(
        stage for stage in stages if stage != "transversal" and not _is_allowed_delegation_target(stage)
    )
    if unmanaged_targets:
        return StageQuestionDecision(
            status="reject_as_noise",
            reason=f"Etapa bloqueante no gobernada: {unmanaged_targets[0]}.",
        )
    if normalized_stage == "discover" and stages and not stages.isdisjoint(DISCOVERY_ALLOWED_BLOCKING_STAGES):
        return StageQuestionDecision(status="allowed_now")
    if stages and normalized_stage not in stages and "transversal" not in stages:
        target = sorted(stages)[0]
        return StageQuestionDecision(
            status="defer_to_acp" if target in ACP_STAGE_KEYS else "defer_to_next_stage",
            reason=f"La pregunta bloquea {target}, no {normalized_stage}.",
            deferral_target_stage=target,
        )
    return StageQuestionDecision(status="allowed_now")


def should_surface_stage_question(stage: str, question: Any) -> bool:
    return classify_stage_question(stage, question).status == "allowed_now"


def filter_stage_question_texts(stage: str, questions: Iterable[Any]) -> list[Any]:
    normalized_stage = _normalize_stage_target(stage) or "discover"
    policy = STAGE_POLICIES.get(normalized_stage)
    allowed = [question for question in questions if should_surface_stage_question(normalized_stage, question)]
    if policy is None:
        return allowed

    blocking: list[Any] = []
    nonblocking: list[Any] = []
    for question in allowed:
        if _is_blocking_question(question):
            blocking.append(question)
        else:
            nonblocking.append(question)
    return [*blocking[: policy.max_blocking], *nonblocking[: policy.max_nonblocking]]


def deferred_stage_questions(stage: str, questions: Iterable[Any]) -> list[dict[str, str]]:
    deferred: list[dict[str, str]] = []
    for question in questions:
        decision = classify_stage_question(stage, question)
        if decision.status not in {"defer_to_next_stage", "defer_to_acp"}:
            continue
        deferred.append(
            {
                "question": _question_text(question).strip(),
                "source_stage": _normalize_stage_target(stage) or "discover",
                "target_stage": decision.deferral_target_stage or ("acp" if decision.status == "defer_to_acp" else ""),
                "reason": decision.reason,
                "status": decision.status,
            }
        )
    return deferred


def sanitize_discovery_analysis_output(analysis: DiscoveryAnalysisOutput) -> DiscoveryAnalysisOutput:
    deferred_open_questions = []
    open_questions = []
    for question in analysis.open_questions:
        decision = classify_stage_question("discover", question)
        if decision.status in {"defer_to_next_stage", "defer_to_acp"}:
            deferred_open_questions.append((question, decision))
        else:
            open_questions.append(question)
    deferred_missing_information = [
        item
        for item in analysis.missing_information
        if not _is_discovery_current_stage_required_gap(item)
    ]
    missing_information = [
        item
        for item in analysis.missing_information
        if _is_discovery_current_stage_required_gap(item)
    ]
    risk_signals = list(analysis.risk_signals)
    deferred_resolution_items = list(analysis.deferred_resolution_items)
    for index, pair in enumerate(deferred_open_questions, start=1):
        question, decision = pair
        key = _field_value(question, "key") or f"deferred_question_{index}"
        question_text = _question_text(question).strip()
        risk_signals.append(
            StructuredInsight(
                key=f"deferred:{key}",
                statement=(
                    f"Pregunta diferida a {decision.deferral_target_stage or 'etapa posterior'}: "
                    f"{question_text}"
                ),
                source_refs=["discovery.open_questions"],
                confidence=0.72,
            )
        )
        deferred_resolution_items.append(
            _build_deferred_resolution_item(
                question_text=question_text,
                source_stage="discover",
                target_stage=decision.deferral_target_stage or "define",
                reason=decision.reason,
                recommendation=_field_value(question, "suggested_answer") or _default_discovery_recommendation(decision.deferral_target_stage or "define"),
                source_refs=["discovery.open_questions"],
            )
        )
    for index, item in enumerate(deferred_missing_information, start=1):
        decision = classify_stage_question("discover", item)
        if decision.status not in {"defer_to_next_stage", "defer_to_acp"}:
            continue
        question_text = str(item).strip()
        risk_signals.append(
            StructuredInsight(
                key=f"deferred:missing_information_{index}",
                statement=f"Informacion diferida a {decision.deferral_target_stage or 'etapa posterior'}: {question_text}",
                source_refs=["discovery.missing_information"],
                confidence=0.66,
            )
        )
        deferred_resolution_items.append(
            _build_deferred_resolution_item(
                question_text=question_text,
                source_stage="discover",
                target_stage=decision.deferral_target_stage or "define",
                reason=decision.reason,
                recommendation=_default_discovery_recommendation(decision.deferral_target_stage or "define"),
                source_refs=["discovery.missing_information"],
            )
        )
    return analysis.model_copy(
        update={
            "open_questions": open_questions,
            "missing_information": missing_information,
            "deferred_resolution_items": deferred_resolution_items,
            "risk_signals": risk_signals,
        }
    )
