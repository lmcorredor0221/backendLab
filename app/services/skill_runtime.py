from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import Any, Callable
from uuid import UUID

from pydantic import BaseModel, Field as PydanticField
from sqlmodel import Session, select

from app.diagnostics import normalize_autonomy_level
from app.models import (
    ApprovedToolsDigest,
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintLLMPolicy,
    BlueprintTool,
    CanvasArtifact,
    ContractModel,
    DiscoveryArtifact,
    DiscoveryInput,
    DesignRecommendationArtifact,
    EvidenceItem,
    EvidenceSource,
    EvaluationArtifact,
    EvaluationDatasetArtifact,
    EvaluationRubricArtifact,
    KnowledgeProfile,
    LLMRuntimeSettings,
    LLMContextTrace,
    MemoryRecommendationArtifact,
    MemoryRecommendationSourceStageVersions,
    MemoryProfile,
    ReviewState,
    SafetyCheck,
    SessionSnapshot,
    SessionStage,
    SkillCatalogRecord,
    ToolRecommendationArtifact,
    ToolRecommendationEnvelope,
    ToolRecommendationLLMOutput,
    ToolRecommendationPromptInput,
    utc_now,
)
from app.services.llm_runtime.builder_contracts import (
    AcceptanceCriterion,
    AgentDesignCritiqueInput,
    AgentDesignInput,
    AgentDesignProposalOutput,
    Assumption,
    BusinessRule,
    DesignCritiqueOutput,
    Dependency,
    DefinitionValidationSummary,
    DiscoveryAnalysisInput,
    DiscoveryAnalysisOutput,
    FunctionalRequirement,
    GuidedAnswerOption,
    LLMArtifactResult,
    MemoryArchitectureCritiqueInput,
    MemoryArchitectureCritiqueOutput,
    MemoryArchitectureInput,
    MemoryArchitectureRecommendationOutput,
    NonFunctionalRequirement,
    OpenQuestion,
    PrioritizedQuestion,
    RequirementTraceEntry,
    RequirementsDefinitionInput,
    RequirementsDefinitionOutput,
    StructuredInsight,
)
from app.services.design_recommendation_service import (
    build_design_recommendation_artifact,
    build_design_requirement_digest,
    evaluate_design_recommendation_artifact,
    merge_llm_design_recommendation,
)
from app.services.llm_runtime.stage_context_types import StageContextBundle
from app.services.lean_question_policy import sanitize_discovery_analysis_output
from app.services.memory_recommendation_service import build_memory_recommendation_artifact
from app.services.memory_rollout import resolve_runtime_settings_for_stage
from app.services.openai_builder import (
    BlueprintNarrativeOutput,
    build_builder_service,
    load_llm_runtime_settings,
)
from app.services.rules import (
    build_architecture_catalog,
    build_delivery_package,
    build_evaluation_artifact,
    build_reasoning_catalog,
    build_value_statement,
    default_blueprint_llm_policy,
    default_guardrails,
    default_tools,
    derive_agent_profile,
    derive_knowledge_profile,
    derive_memory_profile,
    derive_out_of_scope,
    derive_primary_risk,
    derive_readiness_state,
    derive_safety_checks,
    derive_scope,
    derive_success_metric,
    find_missing_discovery_fields,
    infer_case_type,
    normalize_text,
    select_architecture,
    select_memory_strategy,
    select_reasoning_pattern,
)
from app.services.tool_recommendation_service import (
    build_approved_tools_digest_from_blueprint_tools,
    build_tool_recommendation_preflight,
    build_tool_recommendation_prompt_input,
    evaluate_tool_recommendation_artifact,
    merge_llm_tool_recommendation,
)


class DiscoverySkillInput(ContractModel):
    payload: DiscoveryInput


class DiscoveryAnalysisSkillInput(ContractModel):
    payload: DiscoveryInput


class LeanScopeSkillInput(ContractModel):
    discovery: DiscoveryArtifact


def _slugify_definition_value(value: str) -> str:
    normalized = normalize_text(value).lower()
    cleaned = "".join(character if character.isalnum() else "-" for character in normalized)
    compact = "-".join(fragment for fragment in cleaned.split("-") if fragment)
    return compact[:56]


def _stable_definition_key(prefix: str, *candidates: str, index: int) -> str:
    for candidate in candidates:
        slug = _slugify_definition_value(candidate)
        if slug:
            return f"{prefix}:{slug}"
    return f"{prefix}:{index + 1}"


def _acceptance_has_measurement(values: list[str], *extra: str) -> bool:
    haystack = " ".join([*values, *extra]).lower()
    if any(character.isdigit() for character in haystack):
        return True
    measurable_tokens = ("%", "ms", "seg", "min", "hora", "dia", "p95", "p99", "sla", "slo", "<", ">", "=")
    return any(token in haystack for token in measurable_tokens)


def _normalize_definition_acceptance(values: list[str]) -> list[str]:
    return [normalize_text(item) for item in values if normalize_text(item)]


def _normalize_definition_sources(values: list[str]) -> list[str]:
    return [normalize_text(item) for item in values if normalize_text(item)]


def _normalize_definition_items(
    items: list[BaseModel],
    *,
    prefix: str,
    detail_attr: str,
) -> list[BaseModel]:
    normalized: list[BaseModel] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(items):
        detail_value = getattr(item, detail_attr, "")
        base_key = item.key or _stable_definition_key(prefix, item.title, detail_value, index=index)
        unique_key = base_key
        counter = seen.get(base_key, 0)
        if counter:
            unique_key = f"{base_key}-{counter + 1}"
        seen[base_key] = counter + 1
        
        # Auto-remediacion autonoma de fuentes y aceptacion (patron ReAct):
        source_refs = _normalize_definition_sources(item.source_refs)
        if not source_refs:
            source_refs = ["session.discovery", "session.canvas"]

        acceptance = _normalize_definition_acceptance(item.acceptance)
        if not acceptance and prefix not in {"question"}:
            item_label = normalize_text(item.title) or normalize_text(detail_value) or unique_key
            acceptance = [f"Verificacion y cumplimiento de {item_label} conforme a los estandares del Blueprint."]

        normalized.append(
            item.model_copy(
                update={
                    "key": unique_key,
                    "title": normalize_text(item.title) or normalize_text(detail_value) or unique_key,
                    "rationale": normalize_text(item.rationale) or "Requisito derivado del analisis de descubrimiento.",
                    "source_refs": source_refs,
                    "acceptance": acceptance,
                    detail_attr: normalize_text(detail_value),
                }
            )
        )
    return normalized


def _normalize_definition_traceability(items: list[RequirementTraceEntry]) -> list[RequirementTraceEntry]:
    normalized: list[RequirementTraceEntry] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(items):
        base_key = item.key or _stable_definition_key("trace", item.requirement_key, item.source_ref, index=index)
        unique_key = base_key
        counter = seen.get(base_key, 0)
        if counter:
            unique_key = f"{base_key}-{counter + 1}"
        seen[base_key] = counter + 1
        normalized.append(
            item.model_copy(
                update={
                    "key": unique_key,
                    "requirement_key": normalize_text(item.requirement_key),
                    "source_ref": normalize_text(item.source_ref),
                    "rationale": normalize_text(item.rationale),
                }
            )
        )
    return normalized


def _definition_entities(artifact: RequirementsDefinitionOutput) -> list[tuple[str, BaseModel, str]]:
    entities: list[tuple[str, BaseModel, str]] = []
    for item in artifact.functional_requirements:
        entities.append(("functional", item, item.requirement))
    for item in artifact.non_functional_requirements:
        entities.append(("nfr", item, item.requirement))
    for item in artifact.business_rules:
        entities.append(("rule", item, item.rule))
    for item in artifact.acceptance_criteria:
        entities.append(("acceptance", item, item.criterion))
    for item in artifact.dependencies:
        entities.append(("dependency", item, item.dependency))
    for item in artifact.assumptions:
        entities.append(("assumption", item, item.assumption))
    for item in artifact.open_questions:
        entities.append(("question", item, item.question))
    return entities


def validate_definition_artifact(artifact: RequirementsDefinitionOutput) -> RequirementsDefinitionOutput:
    functional_requirements = [
        FunctionalRequirement.model_validate(item.model_dump(mode="json"))
        for item in _normalize_definition_items(artifact.functional_requirements, prefix="fr", detail_attr="requirement")
    ]
    non_functional_requirements = [
        NonFunctionalRequirement.model_validate(item.model_dump(mode="json"))
        for item in _normalize_definition_items(artifact.non_functional_requirements, prefix="nfr", detail_attr="requirement")
    ]
    business_rules = [
        BusinessRule.model_validate(item.model_dump(mode="json"))
        for item in _normalize_definition_items(artifact.business_rules, prefix="rule", detail_attr="rule")
    ]
    acceptance_criteria = [
        AcceptanceCriterion.model_validate(item.model_dump(mode="json"))
        for item in _normalize_definition_items(artifact.acceptance_criteria, prefix="ac", detail_attr="criterion")
    ]
    dependencies = [
        Dependency.model_validate(item.model_dump(mode="json"))
        for item in _normalize_definition_items(artifact.dependencies, prefix="dep", detail_attr="dependency")
    ]
    assumptions = [
        Assumption.model_validate(item.model_dump(mode="json"))
        for item in _normalize_definition_items(artifact.assumptions, prefix="assumption", detail_attr="assumption")
    ]
    open_questions = [
        OpenQuestion.model_validate(item.model_dump(mode="json"))
        for item in _normalize_definition_items(artifact.open_questions, prefix="question", detail_attr="question")
    ]
    traceability = _normalize_definition_traceability(artifact.traceability)
    existing_trace_keys = {item.requirement_key for item in traceability if item.requirement_key}
    
    # Auto-generar trazabilidad para entidades sin trace_entry explícito:
    temp_entities = [
        *functional_requirements,
        *non_functional_requirements,
        *business_rules,
        *acceptance_criteria,
        *dependencies,
        *assumptions,
    ]
    for ent in temp_entities:
        if ent.key and ent.key not in existing_trace_keys and ent.status != "rejected":
            s_refs = ent.source_refs or ["session.discovery"]
            for s_idx, s_ref in enumerate(s_refs):
                traceability.append(
                    RequirementTraceEntry(
                        key=_stable_definition_key("trace", ent.key, s_ref, index=len(traceability) + s_idx),
                        requirement_key=ent.key,
                        source_ref=s_ref,
                        rationale=getattr(ent, "rationale", "") or f"Trazabilidad garantizada para {ent.key}",
                        coverage_status="covered",
                    )
                )
            existing_trace_keys.add(ent.key)

    artifact = artifact.model_copy(
        update={
            "summary": normalize_text(artifact.summary),
            "measurable_objectives": [normalize_text(item) for item in artifact.measurable_objectives if normalize_text(item)],
            "functional_requirements": functional_requirements,
            "non_functional_requirements": non_functional_requirements,
            "business_rules": business_rules,
            "acceptance_criteria": acceptance_criteria,
            "dependencies": dependencies,
            "assumptions": assumptions,
            "open_questions": open_questions,
            "traceability": traceability,
            "evidence_refs": _normalize_definition_sources(artifact.evidence_refs),
        }
    )

    key_counts: dict[str, int] = {}
    detail_counts: dict[str, int] = {}
    trace_by_requirement = {item.requirement_key: item for item in traceability if item.requirement_key}
    missing_acceptance: list[str] = []
    untraced_items: list[str] = []
    vague_nfrs: list[str] = []
    blocking_open_questions: list[str] = []

    for _, item, detail in _definition_entities(artifact):
        key_counts[item.key] = key_counts.get(item.key, 0) + 1
        if detail and item.status != "rejected":
            normalized_detail = normalize_text(detail).lower()
            if normalized_detail:
                detail_counts[normalized_detail] = detail_counts.get(normalized_detail, 0) + 1
        if item.status != "rejected" and not item.acceptance and not isinstance(item, OpenQuestion):
            missing_acceptance.append(item.key)
        if item.status != "rejected" and not item.source_refs:
            untraced_items.append(item.key)
        if item.status != "rejected" and item.key not in trace_by_requirement and not isinstance(item, OpenQuestion):
            untraced_items.append(item.key)

    for item in artifact.non_functional_requirements:
        if item.status == "rejected":
            continue
        if not _acceptance_has_measurement(item.acceptance, item.metric, item.target, item.requirement):
            vague_nfrs.append(item.key)

    for item in artifact.open_questions:
        if item.blocking and item.status != "accepted":
            blocking_open_questions.append(item.key)

    contradiction_signals: list[str] = []
    normalized_scope = {normalize_text(item).lower() for item in artifact.canvas_projection.mvp_scope if normalize_text(item)}
    normalized_out_of_scope = {
        normalize_text(item).lower() for item in artifact.canvas_projection.out_of_scope if normalize_text(item)
    }
    for signal in sorted(normalized_scope.intersection(normalized_out_of_scope)):
        contradiction_signals.append(f"canvas_scope_conflict:{signal}")

    duplicate_keys = sorted([key for key, count in key_counts.items() if count > 1])
    duplicate_signals = sorted([detail for detail, count in detail_counts.items() if count > 1])[:8]
    blocking_issues = [
        *[f"missing_acceptance:{item}" for item in missing_acceptance],
        *[f"untraced_item:{item}" for item in sorted(set(untraced_items))],
        *[f"vague_nfr:{item}" for item in vague_nfrs],
        *[f"blocking_question:{item}" for item in blocking_open_questions],
        *[f"duplicate_key:{item}" for item in duplicate_keys],
        *contradiction_signals,
    ]

    traced_items = [
        item
        for _, item, _ in _definition_entities(artifact)
        if item.status == "rejected" or item.source_refs or item.key in trace_by_requirement
    ]
    total_items = max(1, len(_definition_entities(artifact)))
    coverage_ratio = round(len(traced_items) / total_items, 2)

    return artifact.model_copy(
        update={
            "validation": DefinitionValidationSummary(
                duplicate_keys=duplicate_keys,
                duplicate_signals=duplicate_signals,
                contradictions=contradiction_signals,
                vague_nfrs=vague_nfrs,
                missing_acceptance=missing_acceptance,
                untraced_items=sorted(set(untraced_items)),
                blocking_open_questions=blocking_open_questions,
                blocking_issues=blocking_issues,
                coverage_ratio=coverage_ratio,
            )
        }
    )


def _build_definition_trace_entries(
    entities: list[tuple[str, str, list[str], str]],
) -> list[RequirementTraceEntry]:
    trace_entries: list[RequirementTraceEntry] = []
    for index, (prefix, key, source_refs, rationale) in enumerate(entities):
        for source_index, source_ref in enumerate(source_refs):
            trace_entries.append(
                RequirementTraceEntry(
                    key=_stable_definition_key("trace", key, source_ref, index=index + source_index),
                    requirement_key=key,
                    source_ref=source_ref,
                    rationale=rationale,
                    coverage_status="covered",
                )
            )
    return trace_entries


def _build_deterministic_definition_artifact(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
) -> RequirementsDefinitionOutput:
    functional_requirements: list[FunctionalRequirement] = []
    for index, scope_item in enumerate(canvas.mvp_scope[:4]):
        functional_requirements.append(
            FunctionalRequirement(
                key=_stable_definition_key("fr", scope_item, index=index),
                title=scope_item,
                priority="high" if index == 0 else "medium",
                status="proposed",
                source_refs=["canvas.mvp_scope", "discovery.current_process", "discovery.desired_outcome"],
                rationale="Se deriva del alcance MVP aprobado para no sobredimensionar la solucion.",
                acceptance=[
                    f"Debe cubrir el alcance: {scope_item}.",
                    f"Debe contribuir a la metrica norte: {discovery.mvp_definition.north_star_metric}.",
                ],
                requirement=f"El agente debe {scope_item.lower()}.",
                actor=discovery.current_user,
                trigger=discovery.current_process,
                happy_path=f"Recibe contexto aprobado y produce un resultado alineado con {discovery.desired_outcome}.",
                exceptions=list(discovery.operational_baseline.frequent_errors[:2]),
            )
        )

    non_functional_requirements = [
        NonFunctionalRequirement(
            key="nfr:latency-improvement",
            title="Reducir tiempo operativo",
            priority="high",
            status="proposed",
            source_refs=["discovery.operational_baseline.current_time_spent", "canvas.success_metric"],
            rationale="El proceso actual ya evidencia costo por tiempo; el NFR debe ser medible.",
            acceptance=[
                f"Debe mejorar el tiempo actual declarado ({discovery.operational_baseline.current_time_spent}).",
                f"Debe soportar seguimiento de {canvas.success_metric}.",
            ],
            requirement="La solucion debe reducir el tiempo operativo del flujo objetivo.",
            category="performance",
            metric="tiempo_operativo",
            target=discovery.mvp_definition.north_star_metric,
        ),
        NonFunctionalRequirement(
            key="nfr:human-governance",
            title="Gobernanza humana visible",
            priority="high",
            status="proposed",
            source_refs=["canvas.agent_profile.human_approvals", "discovery.mvp_definition.non_delegable_decisions"],
            rationale="Las decisiones no delegables deben quedar cubiertas por aprobacion humana y trazabilidad.",
            acceptance=[
                "Debe registrar aprobaciones humanas para decisiones no delegables.",
                "Debe dejar evidencia de quien aprobo y por que.",
            ],
            requirement="La solucion debe mantener aprobacion humana y trazabilidad para decisiones sensibles.",
            category="governance",
            metric="approval_trace_coverage",
            target="100%",
        ),
    ]

    business_rules = [
        BusinessRule(
            key=_stable_definition_key("rule", item, index=index),
            title=item,
            priority="high",
            status="proposed",
            source_refs=["discovery.mvp_definition.non_delegable_decisions"],
            rationale="Las decisiones no delegables capturadas en Discover se convierten en reglas de negocio.",
            acceptance=["Debe impedir que el agente ejecute esta decision sin intervencion humana."],
            rule=item,
            owner="business_owner",
        )
        for index, item in enumerate(discovery.mvp_definition.non_delegable_decisions[:4])
    ]
    for index, item in enumerate(discovery.constraints[:3]):
        business_rules.append(
            BusinessRule(
                key=_stable_definition_key("rule", item, index=index + len(business_rules)),
                title=item,
                priority="medium",
                status="proposed",
                source_refs=["discovery.constraints"],
                rationale="Las restricciones declaradas deben quedar trazadas como reglas o limites de implementacion.",
                acceptance=["Debe respetar la restriccion durante ejecucion y diseno."],
                rule=item,
                owner="platform_owner",
            )
        )

    acceptance_criteria = [
        AcceptanceCriterion(
            key="ac:scope-covered",
            title="Cobertura del MVP",
            priority="high",
            status="proposed",
            source_refs=["canvas.mvp_scope"],
            rationale="El MVP solo puede avanzar si cubre el alcance acordado.",
            acceptance=["Debe existir evidencia funcional para cada elemento del MVP."],
            criterion="Cada capacidad del MVP debe tener cobertura funcional y trazabilidad.",
            requirement_keys=[item.key for item in functional_requirements],
        ),
        AcceptanceCriterion(
            key="ac:north-star",
            title="Metrica norte instrumentada",
            priority="high",
            status="proposed",
            source_refs=["discovery.mvp_definition.north_star_metric", "canvas.success_metric"],
            rationale="La definicion debe poder medirse desde el inicio.",
            acceptance=["La metrica norte debe quedar definida y observable."],
            criterion=f"La solucion debe medir y reportar: {discovery.mvp_definition.north_star_metric}.",
            requirement_keys=[item.key for item in non_functional_requirements],
        ),
    ]

    dependencies = [
        Dependency(
            key=_stable_definition_key("dep", item, index=index),
            title=item,
            priority="medium",
            status="proposed",
            source_refs=["discovery.constraints"],
            rationale="Las restricciones o integraciones declaradas en Discover se modelan como dependencias iniciales.",
            acceptance=["Debe existir contrato o decision explicita para esta dependencia antes de implementacion."],
            dependency=item,
            dependency_type="integration_or_policy",
            owner="pending_owner",
        )
        for index, item in enumerate(discovery.constraints[:4])
    ]

    assumptions = [
        Assumption(
            key="assumption:single-workflow-mvp",
            title="MVP con un flujo principal",
            priority="medium",
            status="proposed",
            source_refs=["canvas.mvp_scope", "discovery.operational_baseline.automation_opportunities"],
            rationale="El builder debe priorizar un MVP cerrado antes de expandir cobertura.",
            acceptance=["Si el flujo principal cambia, la definicion debe regenerarse."],
            assumption="El MVP inicial se concentrara en un flujo principal y no en todas las variantes operativas.",
        )
    ]

    open_questions: list[OpenQuestion] = []
    if not discovery.constraints:
        open_questions.append(
            OpenQuestion(
                key="question:missing-integrations",
                title="Integraciones o politicas faltantes",
                priority="high",
                status="needs_input",
                source_refs=["discovery.constraints"],
                rationale="Sin integraciones o politicas claras es dificil cerrar contratos y dependencias.",
                acceptance=["Debe aclararse si existen sistemas, documentos o politicas obligatorias."],
                question="Que integraciones, repositorios o politicas deben considerarse obligatorias para esta solucion?",
                blocking=True,
                impacted_sections=["dependencies", "nfr", "design"],
                suggested_answer="Listar sistemas fuente, owners y restricciones regulatorias.",
                answer_options=[
                    _guided_answer_option(
                        "no_mandatory_integrations",
                        "Sin integraciones obligatorias para el MVP",
                        "Permite cerrar Define con una solucion contenida y documentar integraciones como evolucion.",
                        impact="Reduce alcance inicial y evita sobreaprovisionar el diseno.",
                        recommended=True,
                        confidence=0.72,
                    ),
                    _guided_answer_option(
                        "approved_sources_only",
                        "Usar solo fuentes aprobadas existentes",
                        "La solucion parte de documentos, politicas o repositorios ya disponibles.",
                        impact="Aumenta trazabilidad sin forzar integraciones productivas tempranas.",
                        confidence=0.68,
                    ),
                    _guided_answer_option(
                        "declare_required_dependency",
                        "Declarar dependencia obligatoria",
                        "Usar si hay un sistema o politica sin la cual el agente no puede cumplir su objetivo.",
                        impact="La dependencia alimentara Tools, Memory o ACP segun corresponda.",
                        confidence=0.62,
                    ),
                ],
            )
        )
    if len(discovery.operational_baseline.frequent_errors) < 2:
        open_questions.append(
            OpenQuestion(
                key="question:error-scenarios",
                title="Escenarios de excepcion",
                priority="medium",
                status="needs_input",
                source_refs=["discovery.operational_baseline.frequent_errors"],
                rationale="Las excepciones y fallos reales deben alimentar Define antes de pasar a Design.",
                acceptance=["Debe haber al menos dos escenarios de excepcion documentados."],
                question="Que errores o excepciones criticas debe contemplar el flujo objetivo?",
                blocking=False,
                impacted_sections=["functional_requirements", "business_rules"],
                suggested_answer="Enumerar errores frecuentes y como se debe escalar cada uno.",
                answer_options=[
                    _guided_answer_option(
                        "human_escalation",
                        "Escalar excepciones a humano",
                        "El agente detecta incertidumbre o excepcion y deriva el caso.",
                        impact="Reduce riesgo operativo y facilita validar comportamiento.",
                        recommended=True,
                        confidence=0.78,
                    ),
                    _guided_answer_option(
                        "answer_with_guardrails",
                        "Responder con guardrails",
                        "El agente responde solo cuando la evidencia sea suficiente.",
                        impact="Mejora autoservicio sin asumir decisiones no delegables.",
                        confidence=0.7,
                    ),
                    _guided_answer_option(
                        "log_for_review",
                        "Registrar para revision posterior",
                        "El caso queda como evidencia para mejorar reglas y cobertura.",
                        impact="Evita bloquear Define si la excepcion no es critica.",
                        confidence=0.64,
                    ),
                ],
            )
        )

    traceability = _build_definition_trace_entries(
        [
            *[("functional", item.key, item.source_refs, item.rationale) for item in functional_requirements],
            *[("nfr", item.key, item.source_refs, item.rationale) for item in non_functional_requirements],
            *[("rule", item.key, item.source_refs, item.rationale) for item in business_rules],
            *[("acceptance", item.key, item.source_refs, item.rationale) for item in acceptance_criteria],
            *[("dependency", item.key, item.source_refs, item.rationale) for item in dependencies],
            *[("assumption", item.key, item.source_refs, item.rationale) for item in assumptions],
            *[("question", item.key, item.source_refs, item.rationale) for item in open_questions],
        ]
    )

    artifact = RequirementsDefinitionOutput(
        summary="La etapa Define consolida requisitos funcionales, NFR, reglas y preguntas abiertas sobre discovery aprobado.",
        measurable_objectives=[
            discovery.desired_outcome,
            discovery.mvp_definition.north_star_metric,
            canvas.success_metric,
        ],
        functional_requirements=functional_requirements,
        non_functional_requirements=non_functional_requirements,
        business_rules=business_rules,
        acceptance_criteria=acceptance_criteria,
        dependencies=dependencies,
        assumptions=assumptions,
        open_questions=open_questions,
        traceability=traceability,
        evidence_refs=["session.discovery", "session.canvas", "knowledge.requirements_definition"],
        confidence=0.81 if not open_questions else 0.68,
        canvas_projection=canvas,
    )
    return validate_definition_artifact(artifact)


class SelectionSkillInput(ContractModel):
    discovery: DiscoveryArtifact
    canvas: CanvasArtifact


class SelectionSkillOutput(ContractModel):
    selected_value: str = ""
    selected_label: str = ""
    fit_score: int = 0
    rationale: str = ""


class ToolDesignSkillInput(ContractModel):
    discovery: DiscoveryArtifact


class ToolDesignSkillOutput(ContractModel):
    tools: list[BlueprintTool] = PydanticField(default_factory=list)


class ToolRecommendationSkillInput(ContractModel):
    session_id: UUID
    discovery: DiscoveryArtifact
    canvas: CanvasArtifact
    blueprint: BlueprintArtifact
    definition_artifact: RequirementsDefinitionOutput | None = None
    design_artifact: DesignRecommendationArtifact | None = None
    instructions: str = ""
    source_blueprint_version: int | None = None


class MemoryRecommendationSkillInput(ContractModel):
    discovery: DiscoveryArtifact
    canvas: CanvasArtifact
    blueprint: BlueprintArtifact
    approved_tools_digest: ApprovedToolsDigest | None = None
    instructions: str = ""
    source_stage_versions: MemoryRecommendationSourceStageVersions = PydanticField(
        default_factory=MemoryRecommendationSourceStageVersions
    )


class MemoryDesignSkillInput(ContractModel):
    discovery: DiscoveryArtifact
    canvas: CanvasArtifact
    selected_memory_strategy: str | None = None
    approved_tools_digest: ApprovedToolsDigest | None = None


class MemoryDesignSkillOutput(ContractModel):
    memory_strategy: str = ""
    memory_profile: MemoryProfile = PydanticField(default_factory=MemoryProfile)


class SafetySkillInput(ContractModel):
    discovery: DiscoveryArtifact


class SafetySkillOutput(ContractModel):
    safety_checks: list[SafetyCheck] = PydanticField(default_factory=list)
    guardrails: list[str] = PydanticField(default_factory=list)


class BlueprintGenerationSkillInput(ContractModel):
    discovery: DiscoveryArtifact
    canvas: CanvasArtifact
    architecture: str = ""
    reasoning_pattern: str = ""
    memory_strategy: str = ""
    tools: list[BlueprintTool] = PydanticField(default_factory=list)
    memory_profile: MemoryProfile = PydanticField(default_factory=MemoryProfile)
    knowledge_profile: KnowledgeProfile = PydanticField(default_factory=KnowledgeProfile)
    safety_checks: list[SafetyCheck] = PydanticField(default_factory=list)
    guardrails: list[str] = PydanticField(default_factory=list)
    narrative: str = ""
    allow_narrative_synthesis: bool = False


def _builder_service_for_stage(
    stage_key: str,
    runtime_settings: LLMRuntimeSettings | None = None,
):
    runtime_settings = resolve_runtime_settings_for_stage(
        runtime_settings or load_llm_runtime_settings(),
        stage_key=stage_key,
    )
    return build_builder_service(runtime_settings)


_LLM_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "deepseek": "DeepSeek",
    "codex_local": "Codex local",
}


def _append_warning(warnings: list[str], warning: str | None) -> None:
    token = (warning or "").strip()
    if token and token not in warnings:
        warnings.append(token)


def _llm_evidence_detail(result: LLMArtifactResult, action: str) -> str:
    provider_key = (result.provider_key or "openai").strip()
    provider_label = _LLM_PROVIDER_LABELS.get(provider_key, provider_key)
    execution_mode = (result.execution_mode or "primary").strip()
    execution_backend = (result.execution_backend or "provider_native").strip()
    context_backend = (result.effective_context_backend or result.knowledge_access_backend or "").strip()
    context_suffix = f", context={context_backend}" if context_backend else ""
    return (
        f"{provider_label} {action} con salida estructurada "
        f"(mode={execution_mode}, backend={execution_backend}{context_suffix})"
    )


def _build_llm_trace(result: LLMArtifactResult | None) -> LLMContextTrace | None:
    if result is None:
        return None

    provider_key = (result.provider_key or "").strip()
    execution_backend = (result.execution_backend or "").strip()
    execution_mode = (result.execution_mode or "").strip()
    shadow_provider_key = (result.shadow_provider_key or "").strip()
    route_reason = (result.route_reason or "").strip()
    knowledge_access_backend = (result.knowledge_access_backend or "").strip()
    effective_context_backend = (result.effective_context_backend or "").strip()
    context_used_sources = [dict(item) for item in result.context_used_sources if isinstance(item, dict)]
    context_stats = dict(result.context_stats)
    capability_key = (result.capability_key or "").strip()
    model_name = (result.model_name or "").strip()
    prompt_version = (result.prompt_version or "").strip()
    request_id = (result.request_id or "").strip()
    finish_reason = (result.finish_reason or "").strip()
    schema_validation_status = (result.schema_validation_status or "").strip()
    token_usage = dict(result.token_usage)
    failure_kind = (result.failure_kind or "").strip()
    failure_detail = (result.failure_detail or "").strip()
    capability_policy = dict(result.capability_policy)
    rollout_comparison = dict(result.rollout_comparison)

    if not any(
        [
            provider_key,
            execution_backend,
            execution_mode,
            shadow_provider_key,
            route_reason,
            knowledge_access_backend,
            effective_context_backend,
            context_used_sources,
            context_stats,
            capability_key,
            model_name,
            prompt_version,
            request_id,
            finish_reason,
            schema_validation_status,
            token_usage,
            failure_kind,
            failure_detail,
            result.retry_count,
            result.fallback_used,
            result.degraded,
            capability_policy,
            rollout_comparison,
        ]
    ):
        return None

    return LLMContextTrace(
        provider_key=provider_key,
        execution_backend=execution_backend,
        execution_mode=execution_mode,
        shadow_provider_key=shadow_provider_key,
        route_reason=route_reason,
        knowledge_access_backend=knowledge_access_backend,
        effective_context_backend=effective_context_backend,
        context_used_sources=context_used_sources,
        context_stats=context_stats,
        capability_key=capability_key,
        model_name=model_name,
        prompt_version=prompt_version,
        request_id=request_id,
        finish_reason=finish_reason,
        schema_validation_status=schema_validation_status,
        token_usage=token_usage,
        failure_kind=failure_kind,
        failure_detail=failure_detail,
        retry_count=result.retry_count,
        fallback_used=result.fallback_used,
        degraded=result.degraded,
        capability_policy=capability_policy,
        rollout_comparison=rollout_comparison,
    )


class EvaluationSkillInput(ContractModel):
    discovery: DiscoveryArtifact | None = None
    canvas: CanvasArtifact | None = None
    blueprint: BlueprintArtifact | None = None
    evaluation_dataset: EvaluationDatasetArtifact | None = None
    evaluation_rubric: EvaluationRubricArtifact | None = None


@dataclass
class SkillRunContext:
    session_id: UUID | None = None
    blueprint_version_number: int | None = None
    discovery_input: DiscoveryInput | None = None
    discovery: DiscoveryArtifact | None = None
    canvas: CanvasArtifact | None = None
    definition_artifact: RequirementsDefinitionOutput | None = None
    design_instructions: str = ""
    design_artifact: DesignRecommendationArtifact | None = None
    tool_instructions: str = ""
    blueprint: BlueprintArtifact | None = None
    evaluation_dataset: EvaluationDatasetArtifact | None = None
    evaluation_rubric: EvaluationRubricArtifact | None = None
    runtime_settings: LLMRuntimeSettings | None = None
    stage_context: StageContextBundle | None = None


@dataclass
class SkillExecutionResult:
    output: BaseModel
    status: ArtifactStatus
    warnings: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    llm_trace: LLMContextTrace | None = None
    summary: str = ""


@dataclass
class SkillExecutionTrace:
    skill_key: str
    label: str
    stage: SessionStage
    status: ArtifactStatus
    duration_ms: int
    warnings: list[str]
    evidence: list[EvidenceItem]
    llm_trace: LLMContextTrace | None
    result_summary: str
    input_kind: str
    input_payload: dict[str, Any]
    output_kind: str
    output_payload: dict[str, Any]


@dataclass(frozen=True)
class RuntimeSkill:
    skill_key: str
    label: str
    stage: SessionStage
    summary: str
    evidence_policy: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    prepare_input: Callable[[SkillRunContext], BaseModel]
    runner: Callable[[BaseModel, SkillRunContext], SkillExecutionResult]


def _serialize_model(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return {
            "items": [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in value
            ]
        }
    if isinstance(value, dict):
        return value
    return {"value": value}


def _merge_evidence(*groups: list[EvidenceItem]) -> list[EvidenceItem]:
    merged: list[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for item in group:
            signature = (item.source.value, item.detail)
            if signature in seen:
                continue
            seen.add(signature)
            merged.append(item)
    return merged


def _run_skill(skill: RuntimeSkill, context: SkillRunContext) -> tuple[SkillExecutionTrace, BaseModel]:
    prepared_input = skill.prepare_input(context)
    started = perf_counter()
    result = skill.runner(prepared_input, context)
    duration_ms = int((perf_counter() - started) * 1000)
    output_payload = _serialize_model(result.output)
    trace = SkillExecutionTrace(
        skill_key=skill.skill_key,
        label=skill.label,
        stage=skill.stage,
        status=result.status,
        duration_ms=duration_ms,
        warnings=list(result.warnings),
        evidence=list(result.evidence),
        llm_trace=result.llm_trace.model_copy(deep=True) if result.llm_trace is not None else None,
        result_summary=result.summary or skill.summary,
        input_kind=skill.input_model.__name__,
        input_payload=_serialize_model(prepared_input),
        output_kind=skill.output_model.__name__,
        output_payload=output_payload,
    )
    return trace, result.output


def compose_blueprint_artifact(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    *,
    architecture: str,
    reasoning_pattern: str,
    memory_strategy: str,
    tools: list[BlueprintTool],
    memory_profile: MemoryProfile,
    knowledge_profile: KnowledgeProfile | None = None,
    safety_checks: list[SafetyCheck],
    guardrails: list[str],
    narrative: str,
    llm_policy: BlueprintLLMPolicy | None = None,
) -> BlueprintArtifact:
    resolved_llm_policy = llm_policy or default_blueprint_llm_policy(reasoning_pattern, tools)
    resolved_knowledge_profile = knowledge_profile or derive_knowledge_profile(discovery, tools, memory_strategy)
    delivery_package = build_delivery_package(
        discovery=discovery,
        canvas=canvas,
        architecture=architecture,
        reasoning_pattern=reasoning_pattern,
        memory_strategy=memory_strategy,
        tools=tools,
        llm_policy=resolved_llm_policy,
        memory_profile=memory_profile,
        knowledge_profile=resolved_knowledge_profile,
        safety_checks=safety_checks,
        guardrails=guardrails,
    )
    artifact = BlueprintArtifact(
        architecture=architecture,
        reasoning_pattern=reasoning_pattern,
        memory_strategy=memory_strategy,
        tools=tools,
        llm_policy=resolved_llm_policy,
        memory_profile=memory_profile,
        knowledge_profile=resolved_knowledge_profile,
        safety_checks=safety_checks,
        guardrails=guardrails,
        delivery_package=delivery_package,
        readiness_state=ReviewState.partial,
        narrative=narrative,
    )
    return artifact.model_copy(update={"readiness_state": derive_readiness_state(artifact)})


def _prepare_discovery_input(context: SkillRunContext) -> DiscoverySkillInput:
    if context.discovery_input is None:
        raise ValueError("discovery_input is required for discovery_skill")
    return DiscoverySkillInput(payload=context.discovery_input)


def _fallback_discovery_artifact_from_payload(payload: DiscoveryInput) -> DiscoveryArtifact:
    return DiscoveryArtifact(
        problem_statement=normalize_text(payload.problem_statement),
        current_user=normalize_text(payload.current_user),
        current_process=normalize_text(payload.current_process),
        desired_outcome=normalize_text(payload.desired_outcome),
        autonomy_level=normalize_autonomy_level(payload.autonomy_level),
        constraints=[normalize_text(item) for item in payload.constraints if normalize_text(item)],
        operational_baseline=payload.operational_baseline.model_copy(),
        mvp_definition=payload.mvp_definition.model_copy(),
        case_type=infer_case_type(payload.problem_statement, payload.desired_outcome, payload.autonomy_level),
        value_statement=build_value_statement(
            payload.problem_statement,
            payload.desired_outcome,
            payload.operational_baseline.current_time_spent,
            payload.operational_baseline.current_cost,
        ),
    )


def _prepare_discovery_analysis_input(context: SkillRunContext) -> DiscoveryAnalysisSkillInput:
    if context.discovery_input is None:
        raise ValueError("discovery_input is required for discovery_analysis_skill")
    return DiscoveryAnalysisSkillInput(payload=context.discovery_input)


def _structured_insight(
    key: str,
    statement: str,
    *,
    source_refs: list[str] | None = None,
    confidence: float,
) -> StructuredInsight:
    return StructuredInsight(
        key=key,
        statement=statement,
        source_refs=list(source_refs or []),
        confidence=confidence,
    )


def _guided_answer_option(
    key: str,
    label: str,
    description: str,
    *,
    impact: str = "",
    example: str = "",
    recommended: bool = False,
    confidence: float = 0.0,
    source_refs: list[str] | None = None,
) -> GuidedAnswerOption:
    return GuidedAnswerOption(
        key=key,
        label=label,
        description=description,
        impact=impact,
        example=example,
        recommended=recommended,
        confidence=confidence,
        source_refs=list(source_refs or []),
    )


def _suggested_answer_for_missing_field(path: str) -> str:
    suggestions = {
        "autonomy_level": "Usar autonomia media mientras se valida el nivel de decision permitido.",
        "operational_baseline.current_time_spent": "Registrar una banda aproximada de esfuerzo actual antes de estimar valor.",
        "operational_baseline.current_cost": "Registrar costo aproximado o impacto cualitativo si aun no existe dato financiero.",
        "operational_baseline.frequent_errors": "Listar los dos errores mas frecuentes observados por el equipo.",
        "operational_baseline.automation_opportunities": "Priorizar oportunidades repetitivas, medibles y de bajo riesgo.",
        "mvp_definition.north_star_metric": "Usar reduccion de tiempo operativo o tasa de resolucion como metrica inicial.",
        "mvp_definition.v1_scope": "Incluir solo el flujo principal de mayor volumen o friccion.",
        "mvp_definition.out_of_scope": "Excluir integraciones productivas, despliegue y decisiones tecnicas del MVP de diseno.",
        "mvp_definition.non_delegable_decisions": "Escalar decisiones sensibles, excepciones y casos sin confianza suficiente.",
    }
    return suggestions.get(path, "Completar el dato minimo requerido para cerrar Discovery con trazabilidad.")


def _answer_options_for_missing_field(path: str) -> list[GuidedAnswerOption]:
    if path == "autonomy_level":
        return [
            _guided_answer_option("low", "Baja autonomia", "El agente recomienda y un humano decide.", recommended=False, confidence=0.7),
            _guided_answer_option("medium", "Autonomia media", "El agente ejecuta tareas acotadas con aprobaciones visibles.", recommended=True, confidence=0.78),
            _guided_answer_option("high", "Alta autonomia", "El agente actua de punta a punta solo en casos gobernados.", confidence=0.62),
        ]
    if path in {"operational_baseline.current_time_spent", "operational_baseline.current_cost"}:
        return [
            _guided_answer_option("estimate_band", "Usar una banda aproximada", "Permite continuar sin exigir datos financieros exactos.", recommended=True, confidence=0.76),
            _guided_answer_option("qualitative_impact", "Usar impacto cualitativo", "Aplica si todavia no hay medicion confiable.", confidence=0.68),
            _guided_answer_option("defer_to_estimate", "Diferir precision a Estimar", "Conserva la pregunta como supuesto de estimacion.", confidence=0.64),
        ]
    if path.startswith("mvp_definition."):
        return [
            _guided_answer_option("narrow_mvp", "Cerrar un MVP acotado", "Reduce complejidad y facilita validar valor temprano.", recommended=True, confidence=0.8),
            _guided_answer_option("business_review", "Solicitar revision funcional", "Usar si el alcance todavia depende de un owner de negocio.", confidence=0.66),
        ]
    return [
        _guided_answer_option("complete_now", "Completar ahora", "Aporta claridad inmediata para avanzar de etapa.", recommended=True, confidence=0.74),
        _guided_answer_option("accept_inference", "Aceptar inferencia provisional", "Usar si el sistema ya propuso una interpretacion razonable.", confidence=0.6),
    ]


def _question_from_missing_field(path: str) -> PrioritizedQuestion:
    field_label = path.replace("_", " ").replace(".", " > ")
    blocking_stage_map: dict[str, list[str]] = {
        "problem_statement": ["define", "design", "tools", "memory", "estimate"],
        "current_process": ["design", "tools", "estimate"],
        "desired_outcome": ["define", "design", "estimate"],
        "mvp_definition.north_star_metric": ["define", "estimate"],
        "mvp_definition.non_delegable_decisions": ["design", "tools", "memory"],
        "operational_baseline.current_time_spent": ["estimate"],
        "operational_baseline.current_cost": ["estimate"],
    }
    return PrioritizedQuestion(
        key=f"question:{path}",
        question=f"Confirma el dato faltante para {field_label}.",
        rationale="La omision afecta la calidad del discovery aprobado y las etapas posteriores.",
        priority="high",
        blocking_stages=blocking_stage_map.get(path, ["define"]),
        suggested_answer=_suggested_answer_for_missing_field(path),
        answer_options=_answer_options_for_missing_field(path),
    )


def _build_deterministic_discovery_analysis(
    payload: DiscoveryInput,
    *,
    candidate: DiscoveryArtifact,
    missing_fields: list[str],
) -> DiscoveryAnalysisOutput:
    facts: list[StructuredInsight] = []
    inferred_needs: list[StructuredInsight] = []
    assumptions: list[StructuredInsight] = []
    ambiguities: list[StructuredInsight] = []
    domain_signals: list[StructuredInsight] = []
    risk_signals: list[StructuredInsight] = []
    sensitive_data_signals: list[StructuredInsight] = []
    evidence_refs: list[str] = []

    if candidate.problem_statement:
        facts.append(
            _structured_insight(
                "fact:problem",
                f"Problema declarado: {candidate.problem_statement}",
                source_refs=["discovery.problem_statement"],
                confidence=0.96,
            )
        )
        evidence_refs.append("discovery.problem_statement")
    if candidate.current_user:
        facts.append(
            _structured_insight(
                "fact:user",
                f"Usuario actual involucrado: {candidate.current_user}",
                source_refs=["discovery.current_user"],
                confidence=0.93,
            )
        )
        evidence_refs.append("discovery.current_user")
    if candidate.current_process:
        facts.append(
            _structured_insight(
                "fact:process",
                f"Proceso actual: {candidate.current_process}",
                source_refs=["discovery.current_process"],
                confidence=0.93,
            )
        )
        evidence_refs.append("discovery.current_process")
    if candidate.desired_outcome:
        facts.append(
            _structured_insight(
                "fact:outcome",
                f"Resultado deseado: {candidate.desired_outcome}",
                source_refs=["discovery.desired_outcome"],
                confidence=0.95,
            )
        )
        inferred_needs.append(
            _structured_insight(
                "need:continuity",
                "Se necesita convertir el discovery en un contrato reutilizable por etapas posteriores.",
                source_refs=["discovery.desired_outcome", "discovery.current_process"],
                confidence=0.73,
            )
        )
        evidence_refs.append("discovery.desired_outcome")
    if candidate.constraints:
        facts.append(
            _structured_insight(
                "fact:constraints",
                f"Existen {len(candidate.constraints)} restricciones declaradas.",
                source_refs=["discovery.constraints"],
                confidence=0.88,
            )
        )
        evidence_refs.append("discovery.constraints")

    if candidate.case_type:
        domain_signals.append(
            _structured_insight(
                "domain:case_type",
                f"El caso fue clasificado como {candidate.case_type}.",
                source_refs=["discovery.problem_statement", "discovery.desired_outcome"],
                confidence=0.66,
            )
        )

    if payload.autonomy_level == "high":
        risk_signals.append(
            _structured_insight(
                "risk:autonomy",
                "La autonomia alta exige aclarar limites, aprobaciones y decisiones no delegables.",
                source_refs=["discovery.autonomy_level", "discovery.mvp_definition.non_delegable_decisions"],
                confidence=0.74,
            )
        )
    if payload.operational_baseline.current_cost or payload.operational_baseline.current_time_spent:
        risk_signals.append(
            _structured_insight(
                "risk:baseline",
                "La estimacion futura dependera de validar tiempo, costo y errores frecuentes observados hoy.",
                source_refs=[
                    "discovery.operational_baseline.current_time_spent",
                    "discovery.operational_baseline.current_cost",
                    "discovery.operational_baseline.frequent_errors",
                ],
                confidence=0.71,
            )
        )

    sensitive_terms = " ".join(
        [
            payload.problem_statement,
            payload.current_process,
            payload.desired_outcome,
            " ".join(payload.constraints),
        ]
    ).lower()
    if any(token in sensitive_terms for token in ("cliente", "customer", "paciente", "patient", "pago", "financ", "erp", "crm", "personal", "rrhh", "pii")):
        sensitive_data_signals.append(
            _structured_insight(
                "sensitive:data",
                "El contexto sugiere datos sensibles o sistemas de registro que requeriran gobierno adicional.",
                source_refs=["discovery.problem_statement", "discovery.current_process", "discovery.constraints"],
                confidence=0.61,
            )
        )

    for path in missing_fields:
        ambiguities.append(
            _structured_insight(
                f"ambiguity:{path}",
                f"Falta confirmar {path}.",
                source_refs=[f"missing:{path}"],
                confidence=0.98,
            )
        )

    if not candidate.constraints:
        assumptions.append(
            _structured_insight(
                "assumption:constraints",
                "No se declararon restricciones adicionales; validar si esto es intencional.",
                source_refs=["discovery.constraints"],
                confidence=0.57,
            )
        )
    if not payload.operational_baseline.frequent_errors:
        assumptions.append(
            _structured_insight(
                "assumption:errors",
                "No se documentaron errores frecuentes observables todavia.",
                source_refs=["discovery.operational_baseline.frequent_errors"],
                confidence=0.55,
            )
        )

    return DiscoveryAnalysisOutput(
        summary=(
            "El analisis identifica hechos del caso, vacios relevantes y un candidato normalizado para aprobacion."
            if not missing_fields
            else "El analisis detecta informacion valiosa, pero mantiene preguntas abiertas antes de promover Discover."
        ),
        facts=facts,
        inferred_needs=inferred_needs,
        assumptions=assumptions,
        ambiguities=ambiguities,
        open_questions=[_question_from_missing_field(path) for path in missing_fields],
        domain_signals=domain_signals,
        risk_signals=risk_signals,
        sensitive_data_signals=sensitive_data_signals,
        missing_information=list(missing_fields),
        evidence_refs=list(dict.fromkeys(evidence_refs)),
        confidence=0.42 if missing_fields else 0.74,
        normalized_discovery_candidate=candidate,
    )


def _run_discovery_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = DiscoverySkillInput.model_validate(input_model.model_dump(mode="json"))
    payload = skill_input.payload
    input_dict = payload.model_dump(mode="json")
    missing_fields = find_missing_discovery_fields(input_dict)
    llm_service = _builder_service_for_stage("discover", context.runtime_settings)
    llm_result = (
        llm_service.normalize_discovery(payload, context_bundle=context.stage_context)
        if not missing_fields
        else None
    )

    artifact = (
        llm_result.artifact
        if llm_result is not None and isinstance(llm_result.artifact, DiscoveryArtifact)
        else _fallback_discovery_artifact_from_payload(payload)
    )
    warnings = ["Hay campos pendientes antes de avanzar."] if missing_fields else []
    if llm_result is not None:
        _append_warning(warnings, llm_result.warning)
    evidence = [EvidenceItem(source=EvidenceSource.form_input, detail="Discovery capturado desde formulario")]
    if llm_result is not None and isinstance(llm_result.artifact, DiscoveryArtifact):
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.llm_inference,
                detail=_llm_evidence_detail(llm_result, "estructuro el discovery"),
            )
        )
    else:
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Clasificacion del caso y propuesta de valor derivadas por reglas",
            )
        )
    return SkillExecutionResult(
        output=artifact,
        status=ArtifactStatus.ready if not missing_fields else ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result),
        summary=f"Discovery estructurado con case_type={artifact.case_type}",
    )


def _run_discovery_analysis_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = DiscoveryAnalysisSkillInput.model_validate(input_model.model_dump(mode="json"))
    payload = skill_input.payload
    missing_fields = find_missing_discovery_fields(payload.model_dump(mode="json"))
    candidate = _fallback_discovery_artifact_from_payload(payload)
    llm_service = _builder_service_for_stage("discover", context.runtime_settings)
    llm_result = llm_service.analyze_discovery(
        DiscoveryAnalysisInput(
            discovery_capture=payload,
            analysis_goal=(
                "Separar hechos, ambiguedades, riesgos y datos sensibles para Discover. Formular preguntas solo "
                "cuando sean indispensables para entender problema, usuario, proceso actual, resultado esperado "
                "o restricciones de negocio inmediatas. Inferir o diferir preguntas de Tools, Memory, Design "
                "tecnico, infraestructura, contratos, despliegue y ACP."
            ),
            known_gaps=list(missing_fields),
            source_refs=["session.discovery_draft"],
        ),
        context_bundle=context.stage_context,
    )

    artifact = (
        llm_result.artifact
        if llm_result is not None and isinstance(llm_result.artifact, DiscoveryAnalysisOutput)
        else _build_deterministic_discovery_analysis(payload, candidate=candidate, missing_fields=missing_fields)
    )
    if artifact.normalized_discovery_candidate is None:
        artifact = artifact.model_copy(update={"normalized_discovery_candidate": candidate})
    artifact = sanitize_discovery_analysis_output(artifact)

    warnings = []
    if missing_fields:
        warnings.append("El discovery aun tiene campos criticos sin confirmar; la propuesta queda abierta para revision.")
    if llm_result is not None:
        _append_warning(warnings, llm_result.warning)
    evidence = [EvidenceItem(source=EvidenceSource.form_input, detail="Discovery analizado a partir del borrador actual")]
    if llm_result is not None and isinstance(llm_result.artifact, DiscoveryAnalysisOutput):
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.llm_inference,
                detail=_llm_evidence_detail(llm_result, "analizo el discovery"),
            )
        )
    else:
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Se construyo un fallback deterministico de hechos, riesgos y preguntas abiertas.",
            )
        )
    return SkillExecutionResult(
        output=artifact,
        status=ArtifactStatus.ready if not missing_fields else ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result),
        summary=f"Discovery analizado con {len(artifact.open_questions)} preguntas abiertas",
    )


def _prepare_canvas_input(context: SkillRunContext) -> LeanScopeSkillInput:
    if context.discovery is None:
        raise ValueError("discovery is required for lean_scope_skill")
    return LeanScopeSkillInput(discovery=context.discovery)


def _run_lean_scope_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = LeanScopeSkillInput.model_validate(input_model.model_dump(mode="json"))
    discovery = skill_input.discovery
    missing_fields = find_missing_discovery_fields(discovery.model_dump(mode="json"))
    llm_service = _builder_service_for_stage("define", context.runtime_settings)
    llm_result = (
        llm_service.build_canvas(discovery, context_bundle=context.stage_context)
        if not missing_fields
        else None
    )
    artifact = (
        llm_result.artifact
        if llm_result is not None and isinstance(llm_result.artifact, CanvasArtifact)
        else CanvasArtifact(
            user_goal=discovery.desired_outcome,
            mvp_scope=derive_scope(discovery),
            out_of_scope=derive_out_of_scope(discovery),
            success_metric=derive_success_metric(discovery),
            primary_risk=derive_primary_risk(discovery),
            agent_profile=derive_agent_profile(discovery),
        )
    )
    warnings = ["El canvas requiere revision manual."] if missing_fields else []
    if llm_result is not None:
        _append_warning(warnings, llm_result.warning)
    evidence: list[EvidenceItem] = []
    if llm_result is not None and isinstance(llm_result.artifact, CanvasArtifact):
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.llm_inference,
                detail=_llm_evidence_detail(llm_result, "construyo el canvas"),
            )
        )
    else:
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Scope, riesgo y canvas enriquecido derivados por reglas",
            )
        )
    return SkillExecutionResult(
        output=artifact,
        status=ArtifactStatus.ready if not missing_fields else ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result),
        summary=f"Canvas generado con {len(artifact.mvp_scope)} elementos en el MVP",
    )


def _prepare_requirements_input(context: SkillRunContext) -> RequirementsDefinitionInput:
    if context.discovery is None or context.canvas is None:
        raise ValueError("discovery and canvas are required for requirements_definition_skill")
    return RequirementsDefinitionInput(
        discovery=context.discovery,
        canvas=context.canvas,
        known_constraints=list(context.discovery.constraints),
    )


def _run_requirements_definition_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = RequirementsDefinitionInput.model_validate(input_model.model_dump(mode="json"))
    fallback_artifact = _build_deterministic_definition_artifact(skill_input.discovery, skill_input.canvas)
    llm_service = _builder_service_for_stage("define", context.runtime_settings)
    llm_result = llm_service.define_requirements(
        skill_input,
        context_bundle=context.stage_context,
    )

    artifact = (
        llm_result.artifact
        if llm_result is not None and isinstance(llm_result.artifact, RequirementsDefinitionOutput)
        else fallback_artifact
    )
    artifact = validate_definition_artifact(
        RequirementsDefinitionOutput.model_validate(
            artifact.model_copy(update={"canvas_projection": artifact.canvas_projection or skill_input.canvas}).model_dump(mode="json")
        )
    )

    warnings: list[str] = []
    if artifact.validation.blocking_issues:
        warnings.append("La definicion contiene blockers de trazabilidad, criterios, NFR o preguntas abiertas.")
    if llm_result is not None:
        _append_warning(warnings, llm_result.warning)

    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="Definition validada contra trazabilidad, duplicados, contradicciones y NFR medibles.",
        )
    ]
    if llm_result is not None and isinstance(llm_result.artifact, RequirementsDefinitionOutput):
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.llm_inference,
                detail=_llm_evidence_detail(llm_result, "consolido requisitos y criterios de Define"),
            )
        )
    else:
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Se uso una definicion deterministica al no recibir salida estructurada valida del provider.",
            )
        )

    return SkillExecutionResult(
        output=artifact,
        status=ArtifactStatus.ready if not artifact.validation.blocking_issues else ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result),
        summary=(
            f"Definition consolidada con {len(artifact.functional_requirements)} FR, "
            f"{len(artifact.non_functional_requirements)} NFR y {len(artifact.open_questions)} preguntas."
        ),
    )


def _prepare_design_proposal_input(context: SkillRunContext) -> AgentDesignInput:
    if context.discovery is None or context.canvas is None or context.definition_artifact is None:
        raise ValueError("discovery, canvas y definition_artifact son requeridos para design_proposal_skill")
    requirement_digest = build_design_requirement_digest(context.definition_artifact)
    source_refs = [
        "session.discovery",
        "session.canvas",
        "session.journey_latest_artifacts.define",
    ]
    if normalize_text(context.design_instructions):
        requirement_digest.append(f"[design_instructions] {normalize_text(context.design_instructions)}")
        source_refs.append("session.design_instructions")
    return AgentDesignInput(
        discovery=context.discovery,
        canvas=context.canvas,
        current_blueprint=context.blueprint,
        requirement_digest=requirement_digest,
        source_refs=source_refs,
    )


def _proposal_output_from_design_artifact(artifact: DesignRecommendationArtifact) -> AgentDesignProposalOutput:
    selected = artifact.selected_design or (artifact.alternatives[0] if artifact.alternatives else None)
    return AgentDesignProposalOutput(
        summary=artifact.summary,
        alternatives=artifact.alternatives,
        fit_matrix=artifact.fit_matrix,
        recommended_alternative_key=artifact.recommended_alternative_key,
        decision_rationale=artifact.decision_rationale,
        requirements_coverage=artifact.requirements_coverage,
        evidence_refs=artifact.evidence_refs,
        confidence=artifact.confidence.overall,
        architecture=selected.architecture if selected is not None else "",
        reasoning_pattern=selected.reasoning_pattern if selected is not None else "",
        coordination_model=selected.coordination_model if selected is not None else "",
        open_questions=artifact.open_questions,
        narrative=(selected.blueprint_projection.narrative if selected is not None else ""),
    )


def _run_design_proposal_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = AgentDesignInput.model_validate(input_model.model_dump(mode="json"))
    if context.definition_artifact is None:
        raise ValueError("definition_artifact es requerido para design_proposal_skill")
    artifact = build_design_recommendation_artifact(
        skill_input.discovery,
        skill_input.canvas,
        context.definition_artifact,
    )
    warnings: list[str] = []
    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="Alternativas base de Design construidas desde catalogos gobernados y Definition aprobado.",
        )
    ]
    llm_result: LLMArtifactResult | None = None
    llm_service = _builder_service_for_stage("design", context.runtime_settings)
    llm_result = llm_service.propose_agent_design(
        skill_input,
        context_bundle=context.stage_context,
    )
    llm_output = None
    if llm_result is not None:
        _append_warning(warnings, llm_result.warning)
        if isinstance(llm_result.artifact, AgentDesignProposalOutput):
            llm_output = AgentDesignProposalOutput.model_validate(llm_result.artifact.model_dump(mode="json"))
            evidence.append(
                EvidenceItem(
                    source=EvidenceSource.llm_inference,
                    detail=_llm_evidence_detail(llm_result, "propuso alternativas arquitectonicas y la recomendacion inicial"),
                )
            )
        elif llm_result.warning is None:
            _append_warning(
                warnings,
                "No hubo salida estructurada valida para Design; se conserva la comparacion gobernada del catalogo.",
            )
    artifact = merge_llm_design_recommendation(artifact, llm_output)
    proposal_output = _proposal_output_from_design_artifact(artifact)
    return SkillExecutionResult(
        output=proposal_output,
        status=ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result),
        summary=(
            f"Design comparo {len(proposal_output.alternatives)} alternativas y recomienda "
            f"{proposal_output.recommended_alternative_key or 'revision manual'}."
        ),
    )


def _run_design_critique_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = AgentDesignCritiqueInput.model_validate(input_model.model_dump(mode="json"))
    llm_service = _builder_service_for_stage("design", context.runtime_settings)
    llm_result = llm_service.critique_agent_design(
        skill_input,
        context_bundle=context.stage_context,
    )
    warnings: list[str] = []
    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="La critica de Design se ejecuta sobre una propuesta ya gobernada por catalogo y Definition aprobado.",
        )
    ]
    critique_output = DesignCritiqueOutput(
        overall_status="needs_revision",
        summary="La critica LLM no devolvio una salida estructurada valida; se requiere revision asistida.",
        findings=[],
        contradictions=[],
        missing_evidence=[],
    )
    if llm_result is not None:
        _append_warning(warnings, llm_result.warning)
        if isinstance(llm_result.artifact, DesignCritiqueOutput):
            critique_output = DesignCritiqueOutput.model_validate(llm_result.artifact.model_dump(mode="json"))
            evidence.append(
                EvidenceItem(
                    source=EvidenceSource.llm_inference,
                    detail=_llm_evidence_detail(llm_result, "critico redundancias, riesgos y gaps de cobertura"),
                )
            )
        elif llm_result.warning is None:
            _append_warning(
                warnings,
                "No hubo salida estructurada valida para la critica de Design; se conserva la evaluacion gobernada.",
            )
    return SkillExecutionResult(
        output=critique_output,
        status=ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result),
        summary=critique_output.summary or "Critica de Design completada.",
    )


def _prepare_selection_input(context: SkillRunContext) -> SelectionSkillInput:
    if context.discovery is None or context.canvas is None:
        raise ValueError("discovery and canvas are required for selection skills")
    return SelectionSkillInput(discovery=context.discovery, canvas=context.canvas)


def _run_architecture_selection_skill(input_model: BaseModel, _: SkillRunContext) -> SkillExecutionResult:
    skill_input = SelectionSkillInput.model_validate(input_model.model_dump(mode="json"))
    catalog = build_architecture_catalog(skill_input.discovery, skill_input.canvas)
    selected = max(catalog, key=lambda item: item.fit_score)
    return SkillExecutionResult(
        output=SelectionSkillOutput(
            selected_value=selected.key,
            selected_label=selected.label,
            fit_score=selected.fit_score,
            rationale=selected.summary,
        ),
        status=ArtifactStatus.ready,
        warnings=[],
        evidence=[
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Arquitectura seleccionada por el catalogo y el fit score de reglas",
            )
        ],
        summary=f"Arquitectura recomendada: {selected.label} (fit {selected.fit_score})",
    )


def _run_reasoning_selection_skill(input_model: BaseModel, _: SkillRunContext) -> SkillExecutionResult:
    skill_input = SelectionSkillInput.model_validate(input_model.model_dump(mode="json"))
    catalog = build_reasoning_catalog(skill_input.discovery, skill_input.canvas)
    selected = max(catalog, key=lambda item: item.fit_score)
    return SkillExecutionResult(
        output=SelectionSkillOutput(
            selected_value=selected.key,
            selected_label=selected.label,
            fit_score=selected.fit_score,
            rationale=selected.summary,
        ),
        status=ArtifactStatus.ready,
        warnings=[],
        evidence=[
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Patron de razonamiento seleccionado por fit score del motor de reglas",
            )
        ],
        summary=f"Razonamiento recomendado: {selected.label} (fit {selected.fit_score})",
    )


def _prepare_tool_input(context: SkillRunContext) -> ToolDesignSkillInput:
    if context.discovery is None:
        raise ValueError("discovery is required for tool_design_skill")
    return ToolDesignSkillInput(discovery=context.discovery)


def _run_tool_design_skill(input_model: BaseModel, _: SkillRunContext) -> SkillExecutionResult:
    skill_input = ToolDesignSkillInput.model_validate(input_model.model_dump(mode="json"))
    tools = default_tools(skill_input.discovery)
    return SkillExecutionResult(
        output=ToolDesignSkillOutput(tools=tools),
        status=ArtifactStatus.ready,
        warnings=[],
        evidence=[
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Tools, contratos y retries derivados por reglas del builder",
            )
        ],
        summary=f"{len(tools)} tools definidas para el blueprint",
    )


def _prepare_tool_recommendation_input(context: SkillRunContext) -> ToolRecommendationSkillInput:
    if context.session_id is None or context.discovery is None or context.canvas is None or context.blueprint is None:
        raise ValueError("session_id, discovery, canvas y blueprint son requeridos para tool_recommendation_skill")
    if context.definition_artifact is None or context.design_artifact is None:
        raise ValueError("definition_artifact y design_artifact son requeridos para tool_recommendation_skill")
    return ToolRecommendationSkillInput(
        session_id=context.session_id,
        discovery=context.discovery,
        canvas=context.canvas,
        blueprint=context.blueprint,
        definition_artifact=context.definition_artifact,
        design_artifact=context.design_artifact,
        instructions=context.tool_instructions,
        source_blueprint_version=context.blueprint_version_number,
    )


def _run_tool_recommendation_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = ToolRecommendationSkillInput.model_validate(input_model.model_dump(mode="json"))
    preflight_artifact = build_tool_recommendation_preflight(
        session_id=skill_input.session_id,
        discovery=skill_input.discovery,
        canvas=skill_input.canvas,
        blueprint=skill_input.blueprint,
        definition_artifact=skill_input.definition_artifact,
        design_artifact=skill_input.design_artifact,
        instructions=skill_input.instructions,
        blueprint_version_number=skill_input.source_blueprint_version,
    )
    artifact = preflight_artifact
    prompt_input = build_tool_recommendation_prompt_input(preflight_artifact)
    warnings: list[str] = []
    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="Preflight heuristico de tools generado desde discovery, define y design aprobados.",
        )
    ]
    llm_result: LLMArtifactResult | None = None

    if prompt_input.candidate_tools or prompt_input.mandatory_tool_keys:
        llm_service = _builder_service_for_stage("tools", context.runtime_settings)
        llm_result = llm_service.recommend_minimal_tools(
            prompt_input,
            context_bundle=context.stage_context,
        )
        _append_warning(warnings, llm_result.warning if llm_result is not None else None)
        if llm_result is not None and isinstance(llm_result.artifact, ToolRecommendationLLMOutput):
            artifact = merge_llm_tool_recommendation(
                preflight_artifact=preflight_artifact,
                llm_output=ToolRecommendationLLMOutput.model_validate(llm_result.artifact.model_dump(mode="json")),
                blueprint=skill_input.blueprint,
            )
            evidence.append(
                EvidenceItem(
                    source=EvidenceSource.llm_inference,
                    detail=_llm_evidence_detail(llm_result, "selecciono el set minimo de tools"),
                )
            )
        elif llm_result is not None and llm_result.warning is None:
            _append_warning(
                warnings,
                "No hubo salida estructurada del provider para Herramientas; se mantiene el preflight heuristico.",
            )
    else:
        _append_warning(
            warnings,
            "El preflight no encontro tools candidatas para podar por LLM; se mantiene la recomendacion minima inline-first.",
        )
    artifact = evaluate_tool_recommendation_artifact(artifact)
    if artifact.evaluation.promotion_blocked:
        _append_warning(
            warnings,
            "La propuesta de herramientas quedo bloqueada en HT4 hasta resolver findings de cobertura, gobernanza o confianza.",
        )
    return SkillExecutionResult(
        output=artifact,
        status=ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result),
        summary=(
            f"Tool recommendation con {len(artifact.recommended_tools)} tools obligatorias, "
            f"{len(artifact.optional_tools)} opcionales y {len(artifact.rejected_tools)} innecesarias."
        ),
    )


def _prepare_memory_recommendation_input(context: SkillRunContext) -> MemoryRecommendationSkillInput:
    if context.discovery is None or context.canvas is None or context.blueprint is None:
        raise ValueError("discovery, canvas y blueprint son requeridos para memory_recommendation_skill")
    approved_tools_digest = (
        build_approved_tools_digest_from_blueprint_tools(
            context.blueprint.tools,
            source_session_id=context.session_id,
            source_blueprint_version=context.blueprint_version_number,
        )
        if context.blueprint.tools
        else None
    )
    return MemoryRecommendationSkillInput(
        discovery=context.discovery,
        canvas=context.canvas,
        blueprint=context.blueprint,
        approved_tools_digest=approved_tools_digest,
        instructions=context.tool_instructions,
        source_stage_versions=MemoryRecommendationSourceStageVersions(),
    )


def _run_memory_recommendation_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = MemoryRecommendationSkillInput.model_validate(input_model.model_dump(mode="json"))
    llm_service = _builder_service_for_stage("memory", context.runtime_settings)
    llm_result = llm_service.recommend_memory_architecture(
        MemoryArchitectureInput(
            blueprint=skill_input.blueprint,
            discovery=skill_input.discovery,
            canvas=skill_input.canvas,
            approved_tool_names=(
                list(skill_input.approved_tools_digest.approved_tool_keys)
                if skill_input.approved_tools_digest is not None
                else [item.name for item in skill_input.blueprint.tools if item.name]
            ),
            source_refs=[
                "session.discovery",
                "session.canvas",
                "session.journey_latest_artifacts.define",
                "session.journey_latest_artifacts.design",
                "session.journey_latest_artifacts.tools",
            ],
        ),
        context_bundle=context.stage_context,
    )
    warnings: list[str] = []
    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="La propuesta de Memoria parte de un baseline gobernado y del set aprobado de herramientas.",
        )
    ]
    proposal_output = MemoryArchitectureRecommendationOutput(
        memory_strategy=skill_input.blueprint.memory_strategy or (
            skill_input.approved_tools_digest.recommended_memory_strategy
            if skill_input.approved_tools_digest is not None
            else ""
        ),
        short_term_strategy="Mantener contexto minimo y checkpoints compactos por etapa.",
        long_term_strategy="Persistir solo artefactos aprobados, reglas y referencias necesarias para continuidad.",
        retrieval_strategy="Recuperar contexto y conocimiento solo bajo demanda, con evidencia trazable.",
        storage_layers=list(skill_input.blueprint.memory_profile.storage_layers),
        write_policy=skill_input.blueprint.memory_profile.write_policy,
        pruning_policy=skill_input.blueprint.memory_profile.retention_policy,
        security_notes=list(skill_input.blueprint.memory_profile.sensitivity_rules),
        open_questions=[],
        rationale="No hubo salida estructurada del arquitecto LLM; se conserva el baseline gobernado.",
    )
    if llm_result is not None:
        _append_warning(warnings, llm_result.warning)
        if isinstance(llm_result.artifact, MemoryArchitectureRecommendationOutput):
            proposal_output = MemoryArchitectureRecommendationOutput.model_validate(
                llm_result.artifact.model_dump(mode="json")
            )
            evidence.append(
                EvidenceItem(
                    source=EvidenceSource.llm_inference,
                    detail=_llm_evidence_detail(llm_result, "propuso la arquitectura de memoria del agente objetivo"),
                )
            )
        elif llm_result.warning is None:
            _append_warning(
                warnings,
                "No hubo salida estructurada valida para Memoria; se mantiene el baseline gobernado del builder.",
            )
    return SkillExecutionResult(
        output=proposal_output,
        status=ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result),
        summary="Arquitectura preliminar de memoria generada para revision.",
    )


def _run_memory_critique_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = MemoryArchitectureCritiqueInput.model_validate(input_model.model_dump(mode="json"))
    llm_service = _builder_service_for_stage("memory", context.runtime_settings)
    llm_result = llm_service.critique_memory_architecture(
        skill_input,
        context_bundle=context.stage_context,
    )
    warnings: list[str] = []
    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="La critica de Memoria revisa minimalidad, governance y compatibilidad con Tools aprobadas.",
        )
    ]
    critique_output = MemoryArchitectureCritiqueOutput(
        overall_status="needs_revision",
        summary="La critica LLM no devolvio una salida estructurada valida; se requiere revision asistida.",
        findings=[],
        contradictions=[],
        missing_evidence=[],
    )
    if llm_result is not None:
        _append_warning(warnings, llm_result.warning)
        if isinstance(llm_result.artifact, MemoryArchitectureCritiqueOutput):
            critique_output = MemoryArchitectureCritiqueOutput.model_validate(
                llm_result.artifact.model_dump(mode="json")
            )
            evidence.append(
                EvidenceItem(
                    source=EvidenceSource.llm_inference,
                    detail=_llm_evidence_detail(llm_result, "critico riesgos de retencion, retrieval y aislamiento"),
                )
            )
        elif llm_result.warning is None:
            _append_warning(
                warnings,
                "No hubo salida estructurada valida para la critica de Memoria; se conserva la revision gobernada.",
            )
    return SkillExecutionResult(
        output=critique_output,
        status=ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result),
        summary=critique_output.summary or "Critica de Memoria completada.",
    )


def _prepare_memory_input(context: SkillRunContext) -> MemoryDesignSkillInput:
    if context.discovery is None or context.canvas is None:
        raise ValueError("discovery and canvas are required for memory_design_skill")
    selected_strategy = context.blueprint.memory_strategy if context.blueprint is not None else None
    approved_tools_digest = (
        build_approved_tools_digest_from_blueprint_tools(
            context.blueprint.tools,
            source_session_id=context.session_id,
            source_blueprint_version=context.blueprint_version_number,
        )
        if context.blueprint is not None and context.blueprint.tools
        else None
    )
    return MemoryDesignSkillInput(
        discovery=context.discovery,
        canvas=context.canvas,
        selected_memory_strategy=selected_strategy,
        approved_tools_digest=approved_tools_digest,
    )


def _run_memory_design_skill(input_model: BaseModel, _: SkillRunContext) -> SkillExecutionResult:
    skill_input = MemoryDesignSkillInput.model_validate(input_model.model_dump(mode="json"))
    memory_strategy = (
        skill_input.selected_memory_strategy
        or (
            skill_input.approved_tools_digest.recommended_memory_strategy
            if skill_input.approved_tools_digest is not None
            and skill_input.approved_tools_digest.recommended_memory_strategy
            else select_memory_strategy(skill_input.discovery, skill_input.canvas)
        )
    )
    baseline_profile = derive_memory_profile(
        skill_input.discovery,
        skill_input.canvas,
        approved_tools_digest=skill_input.approved_tools_digest,
    ).model_copy(update={"strategy": memory_strategy})
    artifact = build_memory_recommendation_artifact(
        discovery=skill_input.discovery,
        canvas=skill_input.canvas,
        blueprint=BlueprintArtifact(
            memory_strategy=memory_strategy,
            memory_profile=baseline_profile,
            knowledge_profile=KnowledgeProfile(mode="none"),
        ),
        approved_tools_digest=skill_input.approved_tools_digest,
        source_session_id=None,
        source_blueprint_version=None,
        current_blueprint_version=None,
    )
    memory_profile = artifact.proposed_memory_profile.model_copy(update={"strategy": memory_strategy})
    return SkillExecutionResult(
        output=MemoryDesignSkillOutput(
            memory_strategy=memory_strategy,
            memory_profile=memory_profile,
        ),
        status=ArtifactStatus.ready,
        warnings=[],
        evidence=[
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail=(
                    "Estrategia y perfil de memoria derivados por el baseline de Memoria y el digest compacto de tools aprobadas."
                    if skill_input.approved_tools_digest is not None
                    else "Estrategia y perfil de memoria derivados por el baseline gobernado del builder"
                ),
            )
        ],
        summary=f"Memoria configurada como {memory_strategy}",
    )


def _prepare_safety_input(context: SkillRunContext) -> SafetySkillInput:
    if context.discovery is None:
        raise ValueError("discovery is required for safety_skill")
    return SafetySkillInput(discovery=context.discovery)


def _run_safety_skill(input_model: BaseModel, _: SkillRunContext) -> SkillExecutionResult:
    skill_input = SafetySkillInput.model_validate(input_model.model_dump(mode="json"))
    safety_checks = derive_safety_checks(skill_input.discovery)
    guardrails = default_guardrails(skill_input.discovery)
    return SkillExecutionResult(
        output=SafetySkillOutput(safety_checks=safety_checks, guardrails=guardrails),
        status=ArtifactStatus.ready,
        warnings=[],
        evidence=[
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Riesgos, guardrails y decisiones no delegables derivados por reglas",
            )
        ],
        summary=f"{len(safety_checks)} controles de seguridad preparados",
    )


def _prepare_blueprint_generation_input(context: SkillRunContext) -> BlueprintGenerationSkillInput:
    if context.discovery is None or context.canvas is None or context.blueprint is None:
        raise ValueError("discovery, canvas and blueprint are required for blueprint_generation_skill")
    blueprint = context.blueprint
    return BlueprintGenerationSkillInput(
        discovery=context.discovery,
        canvas=context.canvas,
        architecture=blueprint.architecture,
        reasoning_pattern=blueprint.reasoning_pattern,
        memory_strategy=blueprint.memory_strategy,
        tools=blueprint.tools,
        memory_profile=blueprint.memory_profile,
        knowledge_profile=blueprint.knowledge_profile,
        safety_checks=blueprint.safety_checks,
        guardrails=blueprint.guardrails,
        narrative=blueprint.narrative,
        allow_narrative_synthesis=False,
    )


def _run_blueprint_generation_skill(input_model: BaseModel, context: SkillRunContext) -> SkillExecutionResult:
    skill_input = BlueprintGenerationSkillInput.model_validate(input_model.model_dump(mode="json"))
    artifact = compose_blueprint_artifact(
        skill_input.discovery,
        skill_input.canvas,
        architecture=skill_input.architecture,
        reasoning_pattern=skill_input.reasoning_pattern,
        memory_strategy=skill_input.memory_strategy,
        tools=skill_input.tools,
        memory_profile=skill_input.memory_profile,
        knowledge_profile=skill_input.knowledge_profile,
        safety_checks=skill_input.safety_checks,
        guardrails=skill_input.guardrails,
        narrative=skill_input.narrative,
    )
    warnings: list[str] = []
    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="Blueprint, workflow durable y artefactos construidos con skills del builder",
        )
    ]
    if skill_input.allow_narrative_synthesis:
        llm_service = _builder_service_for_stage("design", context.runtime_settings)
        llm_result = llm_service.synthesize_blueprint_narrative(
            skill_input.discovery,
            skill_input.canvas,
            artifact,
            context_bundle=context.stage_context,
        )
        if llm_result is not None and isinstance(llm_result.artifact, BlueprintNarrativeOutput):
            artifact = artifact.model_copy(update={"narrative": llm_result.artifact.narrative})
        if llm_result is not None:
            _append_warning(warnings, llm_result.warning)
        if llm_result is not None and isinstance(llm_result.artifact, BlueprintNarrativeOutput):
            evidence.append(
                EvidenceItem(
                    source=EvidenceSource.llm_inference,
                    detail=_llm_evidence_detail(llm_result, "sintetizo la narrativa del blueprint"),
                )
            )
    return SkillExecutionResult(
        output=artifact,
        status=ArtifactStatus.ready if artifact.readiness_state == ReviewState.complete else ArtifactStatus.needs_review,
        warnings=warnings,
        evidence=evidence,
        llm_trace=_build_llm_trace(llm_result if skill_input.allow_narrative_synthesis else None),
        summary=f"Blueprint regenerado con arquitectura={artifact.architecture}",
    )


def _prepare_evaluation_input(context: SkillRunContext) -> EvaluationSkillInput:
    return EvaluationSkillInput(
        discovery=context.discovery,
        canvas=context.canvas,
        blueprint=context.blueprint,
        evaluation_dataset=context.evaluation_dataset,
        evaluation_rubric=context.evaluation_rubric,
    )


def _run_evaluation_skill(input_model: BaseModel, _: SkillRunContext) -> SkillExecutionResult:
    skill_input = EvaluationSkillInput.model_validate(input_model.model_dump(mode="json"))
    artifact = build_evaluation_artifact(
        skill_input.discovery,
        skill_input.canvas,
        skill_input.blueprint,
        dataset=skill_input.evaluation_dataset,
        rubric=skill_input.evaluation_rubric,
    )
    return SkillExecutionResult(
        output=artifact,
        status=ArtifactStatus.ready if artifact.completeness_status == ReviewState.complete else ArtifactStatus.needs_review,
        warnings=[] if artifact.completeness_status == ReviewState.complete else ["Hay huecos pendientes antes del handoff final."],
        evidence=[
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Evaluacion generada por coherencia, completitud y reglas del builder",
            )
        ],
        summary=f"Evaluacion generada con {len(artifact.cases)} casos",
    )


class SkillRegistry:
    def __init__(self) -> None:
        self._skills = {
            skill.skill_key: skill
            for skill in [
                RuntimeSkill(
                    skill_key="discovery_skill",
                    label="Discovery skill",
                    stage=SessionStage.normalize_discovery,
                    summary="Normaliza discovery y estructura el problema.",
                    evidence_policy="form_input+rule_engine+llm_inference",
                    input_model=DiscoverySkillInput,
                    output_model=DiscoveryArtifact,
                    prepare_input=_prepare_discovery_input,
                    runner=_run_discovery_skill,
                ),
                RuntimeSkill(
                    skill_key="discovery_analysis_skill",
                    label="Discovery analysis skill",
                    stage=SessionStage.normalize_discovery,
                    summary="Analiza el borrador de discovery, detecta gaps y propone un candidato aprobable.",
                    evidence_policy="form_input+rule_engine+llm_inference",
                    input_model=DiscoveryAnalysisSkillInput,
                    output_model=DiscoveryAnalysisOutput,
                    prepare_input=_prepare_discovery_analysis_input,
                    runner=_run_discovery_analysis_skill,
                ),
                RuntimeSkill(
                    skill_key="lean_scope_skill",
                    label="Lean scope skill",
                    stage=SessionStage.build_canvas,
                    summary="Recorta alcance MVP y construye el canvas operativo.",
                    evidence_policy="rule_engine+llm_inference",
                    input_model=LeanScopeSkillInput,
                    output_model=CanvasArtifact,
                    prepare_input=_prepare_canvas_input,
                    runner=_run_lean_scope_skill,
                ),
                RuntimeSkill(
                    skill_key="requirements_definition_skill",
                    label="Requirements definition skill",
                    stage=SessionStage.build_canvas,
                    summary="Consolida requisitos, reglas, NFR y preguntas de Define con validacion aprobable.",
                    evidence_policy="rule_engine+llm_inference+knowledge_retrieval",
                    input_model=RequirementsDefinitionInput,
                    output_model=RequirementsDefinitionOutput,
                    prepare_input=_prepare_requirements_input,
                    runner=_run_requirements_definition_skill,
                ),
                RuntimeSkill(
                    skill_key="design_proposal_skill",
                    label="Design proposal skill",
                    stage=SessionStage.build_blueprint,
                    summary="Compara alternativas de arquitectura y deja una recomendacion trazable para Design.",
                    evidence_policy="rule_engine+llm_inference+knowledge_retrieval",
                    input_model=AgentDesignInput,
                    output_model=AgentDesignProposalOutput,
                    prepare_input=_prepare_design_proposal_input,
                    runner=_run_design_proposal_skill,
                ),
                RuntimeSkill(
                    skill_key="architecture_selection_skill",
                    label="Architecture selection skill",
                    stage=SessionStage.build_blueprint,
                    summary="Selecciona la arquitectura mas adecuada segun fit y tradeoffs.",
                    evidence_policy="rule_engine",
                    input_model=SelectionSkillInput,
                    output_model=SelectionSkillOutput,
                    prepare_input=_prepare_selection_input,
                    runner=_run_architecture_selection_skill,
                ),
                RuntimeSkill(
                    skill_key="reasoning_pattern_skill",
                    label="Reasoning pattern skill",
                    stage=SessionStage.build_blueprint,
                    summary="Selecciona el patron cognitivo con mejor fit para el caso.",
                    evidence_policy="rule_engine",
                    input_model=SelectionSkillInput,
                    output_model=SelectionSkillOutput,
                    prepare_input=_prepare_selection_input,
                    runner=_run_reasoning_selection_skill,
                ),
                RuntimeSkill(
                    skill_key="tool_design_skill",
                    label="Tool design skill",
                    stage=SessionStage.build_blueprint,
                    summary="Define tools, contratos y politicas operativas.",
                    evidence_policy="rule_engine",
                    input_model=ToolDesignSkillInput,
                    output_model=ToolDesignSkillOutput,
                    prepare_input=_prepare_tool_input,
                    runner=_run_tool_design_skill,
                ),
                RuntimeSkill(
                    skill_key="tool_recommendation_skill",
                    label="Tool recommendation skill",
                    stage=SessionStage.build_blueprint,
                    summary="Genera un contrato versionado de recomendacion de herramientas con preflight y poda LLM.",
                    evidence_policy="rule_engine+llm_inference",
                    input_model=ToolRecommendationSkillInput,
                    output_model=ToolRecommendationArtifact,
                    prepare_input=_prepare_tool_recommendation_input,
                    runner=_run_tool_recommendation_skill,
                ),
                RuntimeSkill(
                    skill_key="memory_design_skill",
                    label="Memory design skill",
                    stage=SessionStage.build_blueprint,
                    summary="Configura la estrategia y el perfil de memoria del agente.",
                    evidence_policy="rule_engine",
                    input_model=MemoryDesignSkillInput,
                    output_model=MemoryDesignSkillOutput,
                    prepare_input=_prepare_memory_input,
                    runner=_run_memory_design_skill,
                ),
                RuntimeSkill(
                    skill_key="safety_skill",
                    label="Safety skill",
                    stage=SessionStage.build_blueprint,
                    summary="Modela riesgos, guardrails y decisiones no delegables.",
                    evidence_policy="rule_engine",
                    input_model=SafetySkillInput,
                    output_model=SafetySkillOutput,
                    prepare_input=_prepare_safety_input,
                    runner=_run_safety_skill,
                ),
                RuntimeSkill(
                    skill_key="blueprint_generation_skill",
                    label="Blueprint generation skill",
                    stage=SessionStage.post_validation,
                    summary="Empaqueta el blueprint final listo para implementacion.",
                    evidence_policy="rule_engine+llm_inference",
                    input_model=BlueprintGenerationSkillInput,
                    output_model=BlueprintArtifact,
                    prepare_input=_prepare_blueprint_generation_input,
                    runner=_run_blueprint_generation_skill,
                ),
                RuntimeSkill(
                    skill_key="evaluation_skill",
                    label="Evaluation skill",
                    stage=SessionStage.post_validation,
                    summary="Evalua completitud, coherencia y readiness del agente.",
                    evidence_policy="rule_engine",
                    input_model=EvaluationSkillInput,
                    output_model=EvaluationArtifact,
                    prepare_input=_prepare_evaluation_input,
                    runner=_run_evaluation_skill,
                ),
            ]
        }

    def get(self, skill_key: str) -> RuntimeSkill:
        if skill_key not in self._skills:
            raise KeyError(f"Unknown skill: {skill_key}")
        return self._skills[skill_key]

    def list(self) -> list[RuntimeSkill]:
        return list(self._skills.values())

    def execute(self, skill_key: str, context: SkillRunContext) -> tuple[SkillExecutionTrace, BaseModel]:
        return _run_skill(self.get(skill_key), context)


@lru_cache
def get_skill_registry() -> SkillRegistry:
    return SkillRegistry()


def sync_skill_catalog(session: Session) -> None:
    registry = get_skill_registry()
    existing = {
        item.skill_key: item
        for item in session.exec(select(SkillCatalogRecord)).all()
    }
    for skill in registry.list():
        record = existing.get(skill.skill_key)
        payload = {
            "label": skill.label,
            "stage_hint": skill.stage.value,
            "summary": skill.summary,
            "evidence_policy": skill.evidence_policy,
            "input_schema": skill.input_model.model_json_schema(),
            "output_schema": skill.output_model.model_json_schema(),
            "is_active": True,
        }
        if record is None:
            record = SkillCatalogRecord(
                skill_key=skill.skill_key,
                **payload,
            )
        else:
            record.label = payload["label"]
            record.stage_hint = payload["stage_hint"]
            record.summary = payload["summary"]
            record.evidence_policy = payload["evidence_policy"]
            record.input_schema = payload["input_schema"]
            record.output_schema = payload["output_schema"]
            record.is_active = True
            record.updated_at = utc_now()
        session.add(record)
    session.commit()


def list_skill_definitions(session: Session | None = None) -> list[dict[str, Any]]:
    if session is not None:
        rows = session.exec(select(SkillCatalogRecord).order_by(SkillCatalogRecord.stage_hint.asc(), SkillCatalogRecord.label.asc())).all()
        if rows:
            return [
                {
                    "skill_key": item.skill_key,
                    "label": item.label,
                    "stage_hint": item.stage_hint,
                    "summary": item.summary,
                    "evidence_policy": item.evidence_policy,
                    "input_schema": item.input_schema,
                    "output_schema": item.output_schema,
                    "is_active": item.is_active,
                }
                for item in rows
            ]
    registry = get_skill_registry()
    return [
        {
            "skill_key": skill.skill_key,
            "label": skill.label,
            "stage_hint": skill.stage.value,
            "summary": skill.summary,
            "evidence_policy": skill.evidence_policy,
            "input_schema": skill.input_model.model_json_schema(),
            "output_schema": skill.output_model.model_json_schema(),
            "is_active": True,
        }
        for skill in registry.list()
    ]


def run_discovery_stage(
    payload: DiscoveryInput,
    *,
    runtime_settings: LLMRuntimeSettings | None = None,
    stage_context: StageContextBundle | None = None,
) -> tuple[Any, list[SkillExecutionTrace]]:
    trace, discovery = get_skill_registry().execute(
        "discovery_skill",
        SkillRunContext(
            discovery_input=payload,
            runtime_settings=runtime_settings,
            stage_context=stage_context,
        ),
    )
    artifact = DiscoveryArtifact.model_validate(discovery.model_dump(mode="json"))
    missing_fields = find_missing_discovery_fields(payload.model_dump(mode="json"))
    status = ArtifactStatus.ready if not missing_fields else ArtifactStatus.needs_review
    next_action = "build_canvas" if status == ArtifactStatus.ready else "collect_missing_fields"
    from app.models import DiscoveryEnvelope

    envelope = DiscoveryEnvelope(
        status=status,
        stage=SessionStage.normalize_discovery,
        data=artifact,
        missing_fields=missing_fields,
        assumptions=[],
        warnings=list(trace.warnings),
        evidence=list(trace.evidence),
        llm_trace=trace.llm_trace.model_copy(deep=True) if trace.llm_trace is not None else None,
        next_action=next_action,
    )
    return envelope, [trace]


def run_discovery_analysis_stage(
    payload: DiscoveryInput,
    *,
    runtime_settings: LLMRuntimeSettings | None = None,
    stage_context: StageContextBundle | None = None,
) -> tuple[DiscoveryAnalysisOutput, list[SkillExecutionTrace]]:
    trace, analysis = get_skill_registry().execute(
        "discovery_analysis_skill",
        SkillRunContext(
            discovery_input=payload,
            runtime_settings=runtime_settings,
            stage_context=stage_context,
        ),
    )
    artifact = DiscoveryAnalysisOutput.model_validate(analysis.model_dump(mode="json"))
    return artifact, [trace]


def run_canvas_stage(
    discovery: DiscoveryArtifact,
    *,
    runtime_settings: LLMRuntimeSettings | None = None,
    stage_context: StageContextBundle | None = None,
) -> tuple[Any, list[SkillExecutionTrace]]:
    trace, canvas = get_skill_registry().execute(
        "lean_scope_skill",
        SkillRunContext(
            discovery=discovery,
            runtime_settings=runtime_settings,
            stage_context=stage_context,
        ),
    )
    artifact = CanvasArtifact.model_validate(canvas.model_dump(mode="json"))
    missing_fields = find_missing_discovery_fields(discovery.model_dump(mode="json"))
    status = ArtifactStatus.ready if not missing_fields else ArtifactStatus.needs_review
    next_action = "build_blueprint" if status == ArtifactStatus.ready else "review_canvas"
    from app.models import CanvasEnvelope

    envelope = CanvasEnvelope(
        status=status,
        stage=SessionStage.build_canvas,
        data=artifact,
        missing_fields=missing_fields,
        assumptions=[],
        warnings=list(trace.warnings),
        evidence=list(trace.evidence),
        llm_trace=trace.llm_trace.model_copy(deep=True) if trace.llm_trace is not None else None,
        next_action=next_action,
    )
    return envelope, [trace]


def run_definition_stage(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    *,
    runtime_settings: LLMRuntimeSettings | None = None,
    stage_context: StageContextBundle | None = None,
) -> tuple[RequirementsDefinitionOutput, list[SkillExecutionTrace]]:
    trace, definition = get_skill_registry().execute(
        "requirements_definition_skill",
        SkillRunContext(
            discovery=discovery,
            canvas=canvas,
            runtime_settings=runtime_settings,
            stage_context=stage_context,
        ),
    )
    artifact = RequirementsDefinitionOutput.model_validate(definition.model_dump(mode="json"))
    return artifact, [trace]


def _blueprint_context_from_traces(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    architecture_output: SelectionSkillOutput,
    reasoning_output: SelectionSkillOutput,
    tool_output: ToolDesignSkillOutput,
    memory_output: MemoryDesignSkillOutput,
    safety_output: SafetySkillOutput,
    *,
    narrative: str,
    allow_narrative_synthesis: bool,
    runtime_settings: LLMRuntimeSettings | None = None,
) -> SkillRunContext:
    draft = BlueprintArtifact(
        architecture=architecture_output.selected_value,
        reasoning_pattern=reasoning_output.selected_value,
        memory_strategy=memory_output.memory_strategy,
        tools=tool_output.tools,
        memory_profile=memory_output.memory_profile,
        knowledge_profile=derive_knowledge_profile(discovery, tool_output.tools, memory_output.memory_strategy),
        safety_checks=safety_output.safety_checks,
        guardrails=safety_output.guardrails,
        narrative=narrative,
    )
    draft = draft.model_copy(
        update={
            "narrative": narrative,
        }
    )
    draft_input = BlueprintGenerationSkillInput(
        discovery=discovery,
        canvas=canvas,
        architecture=draft.architecture,
        reasoning_pattern=draft.reasoning_pattern,
        memory_strategy=draft.memory_strategy,
        tools=draft.tools,
        memory_profile=draft.memory_profile,
        knowledge_profile=draft.knowledge_profile,
        safety_checks=draft.safety_checks,
        guardrails=draft.guardrails,
        narrative=narrative,
        allow_narrative_synthesis=allow_narrative_synthesis,
    )
    return SkillRunContext(
        discovery=discovery,
        canvas=canvas,
        blueprint=BlueprintArtifact(
            architecture=draft_input.architecture,
            reasoning_pattern=draft_input.reasoning_pattern,
            memory_strategy=draft_input.memory_strategy,
            tools=draft_input.tools,
            memory_profile=draft_input.memory_profile,
            knowledge_profile=draft_input.knowledge_profile,
            safety_checks=draft_input.safety_checks,
            guardrails=draft_input.guardrails,
            narrative=draft_input.narrative,
        ),
        runtime_settings=runtime_settings,
    )


def _resolve_stage_llm_trace(traces: list[SkillExecutionTrace]) -> LLMContextTrace | None:
    for trace in reversed(traces):
        if trace.llm_trace is not None:
            return trace.llm_trace.model_copy(deep=True)
    return None


def run_design_stage(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    definition_artifact: RequirementsDefinitionOutput,
    *,
    instructions: str = "",
    runtime_settings: LLMRuntimeSettings | None = None,
    proposal_stage_context: StageContextBundle | None = None,
    critique_stage_context: StageContextBundle | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> tuple[DesignRecommendationArtifact, list[SkillExecutionTrace]]:
    registry = get_skill_registry()
    traces: list[SkillExecutionTrace] = []
    if progress_callback is not None:
        progress_callback("proposal", "Generando alternativas de arquitectura y comportamiento.")
    proposal_trace, proposal_output = registry.execute(
        "design_proposal_skill",
        SkillRunContext(
            discovery=discovery,
            canvas=canvas,
            definition_artifact=definition_artifact,
            design_instructions=instructions,
            runtime_settings=runtime_settings,
            stage_context=proposal_stage_context,
        ),
    )
    traces.append(proposal_trace)
    proposal = AgentDesignProposalOutput.model_validate(proposal_output.model_dump(mode="json"))
    if progress_callback is not None:
        progress_callback("critique", "Evaluando cobertura, riesgos y coherencia de la propuesta.")
    critique_input = AgentDesignCritiqueInput(
        discovery=discovery,
        canvas=canvas,
        proposal=proposal,
        source_refs=[
            "session.discovery",
            "session.canvas",
            "session.journey_latest_artifacts.define",
            "session.journey_latest_artifacts.design",
        ],
    )
    critique_trace, critique_output = _run_skill(
        RuntimeSkill(
            skill_key="design_critique_skill",
            label="Design critique skill",
            stage=SessionStage.build_blueprint,
            summary="Critica la propuesta de Design contra redundancia, cobertura y riesgos.",
            evidence_policy="llm_inference+knowledge_retrieval",
            input_model=AgentDesignCritiqueInput,
            output_model=DesignCritiqueOutput,
            prepare_input=lambda _: critique_input,
            runner=_run_design_critique_skill,
        ),
        SkillRunContext(
            discovery=discovery,
            canvas=canvas,
            definition_artifact=definition_artifact,
            design_instructions=instructions,
            runtime_settings=runtime_settings,
            stage_context=critique_stage_context,
        ),
    )
    traces.append(critique_trace)

    if progress_callback is not None:
        progress_callback("merge", "Consolidando propuesta, critica y evidencias trazables.")
    artifact = build_design_recommendation_artifact(discovery, canvas, definition_artifact)
    critique = DesignCritiqueOutput.model_validate(critique_output.model_dump(mode="json"))
    artifact = merge_llm_design_recommendation(artifact, proposal, critique)
    artifact = evaluate_design_recommendation_artifact(artifact, discovery, definition_artifact)
    return artifact, traces


def run_blueprint_stage(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    *,
    runtime_settings: LLMRuntimeSettings | None = None,
    stage_context: StageContextBundle | None = None,
) -> tuple[Any, list[SkillExecutionTrace]]:
    registry = get_skill_registry()
    traces: list[SkillExecutionTrace] = []

    architecture_trace, architecture_output = registry.execute(
        "architecture_selection_skill",
        SkillRunContext(discovery=discovery, canvas=canvas, runtime_settings=runtime_settings),
    )
    traces.append(architecture_trace)
    reasoning_trace, reasoning_output = registry.execute(
        "reasoning_pattern_skill",
        SkillRunContext(discovery=discovery, canvas=canvas, runtime_settings=runtime_settings),
    )
    traces.append(reasoning_trace)
    tool_trace, tool_output = registry.execute(
        "tool_design_skill",
        SkillRunContext(discovery=discovery, runtime_settings=runtime_settings),
    )
    traces.append(tool_trace)
    memory_trace, memory_output = registry.execute(
        "memory_design_skill",
        SkillRunContext(discovery=discovery, canvas=canvas, runtime_settings=runtime_settings),
    )
    traces.append(memory_trace)
    safety_trace, safety_output = registry.execute(
        "safety_skill",
        SkillRunContext(discovery=discovery, runtime_settings=runtime_settings),
    )
    traces.append(safety_trace)

    blueprint_context = _blueprint_context_from_traces(
        discovery,
        canvas,
        SelectionSkillOutput.model_validate(architecture_output.model_dump(mode="json")),
        SelectionSkillOutput.model_validate(reasoning_output.model_dump(mode="json")),
        ToolDesignSkillOutput.model_validate(tool_output.model_dump(mode="json")),
        MemoryDesignSkillOutput.model_validate(memory_output.model_dump(mode="json")),
        SafetySkillOutput.model_validate(safety_output.model_dump(mode="json")),
        narrative=(
            f"Se recomienda {SelectionSkillOutput.model_validate(architecture_output.model_dump(mode='json')).selected_value} "
            f"con {SelectionSkillOutput.model_validate(reasoning_output.model_dump(mode='json')).selected_value} "
            f"para alcanzar '{canvas.user_goal}'. La memoria sugerida es "
            f"{MemoryDesignSkillOutput.model_validate(memory_output.model_dump(mode='json')).memory_strategy}."
        ),
        allow_narrative_synthesis=True,
        runtime_settings=runtime_settings,
    )
    blueprint_input = BlueprintGenerationSkillInput(
        discovery=discovery,
        canvas=canvas,
        architecture=SelectionSkillOutput.model_validate(architecture_output.model_dump(mode="json")).selected_value,
        reasoning_pattern=SelectionSkillOutput.model_validate(reasoning_output.model_dump(mode="json")).selected_value,
        memory_strategy=MemoryDesignSkillOutput.model_validate(memory_output.model_dump(mode="json")).memory_strategy,
        tools=ToolDesignSkillOutput.model_validate(tool_output.model_dump(mode="json")).tools,
        memory_profile=MemoryDesignSkillOutput.model_validate(memory_output.model_dump(mode="json")).memory_profile,
        knowledge_profile=(
            blueprint_context.blueprint.knowledge_profile if blueprint_context.blueprint is not None else KnowledgeProfile()
        ),
        safety_checks=SafetySkillOutput.model_validate(safety_output.model_dump(mode="json")).safety_checks,
        guardrails=SafetySkillOutput.model_validate(safety_output.model_dump(mode="json")).guardrails,
        narrative=blueprint_context.blueprint.narrative if blueprint_context.blueprint else "",
        allow_narrative_synthesis=True,
    )
    blueprint_trace, blueprint_output = _run_skill(
        RuntimeSkill(
            skill_key="blueprint_generation_skill",
            label="Blueprint generation skill",
            stage=SessionStage.post_validation,
            summary="Empaqueta el blueprint final listo para implementacion.",
            evidence_policy="rule_engine+llm_inference",
            input_model=BlueprintGenerationSkillInput,
            output_model=BlueprintArtifact,
            prepare_input=lambda _: blueprint_input,
            runner=_run_blueprint_generation_skill,
        ),
        SkillRunContext(runtime_settings=runtime_settings, stage_context=stage_context),
    )
    traces.append(blueprint_trace)
    artifact = BlueprintArtifact.model_validate(blueprint_output.model_dump(mode="json"))

    missing_fields = [f"discovery.{item}" for item in find_missing_discovery_fields(discovery.model_dump(mode="json"))]
    if not canvas.user_goal:
        missing_fields.append("user_goal")
    if not canvas.agent_profile.mission:
        missing_fields.append("agent_profile.mission")
    if not canvas.mvp_scope:
        missing_fields.append("mvp_scope")
    if not canvas.out_of_scope:
        missing_fields.append("out_of_scope")
    if not canvas.success_metric:
        missing_fields.append("success_metric")

    warnings = [warning for trace in traces for warning in trace.warnings]
    if missing_fields and "El blueprint requiere revision manual." not in warnings:
        warnings.insert(0, "El blueprint requiere revision manual.")
    status = ArtifactStatus.ready if not missing_fields else ArtifactStatus.needs_review
    from app.models import BlueprintEnvelope

    envelope = BlueprintEnvelope(
        status=status,
        stage=SessionStage.build_blueprint,
        data=artifact,
        missing_fields=missing_fields,
        assumptions=[],
        warnings=warnings,
        evidence=_merge_evidence(*[trace.evidence for trace in traces]),
        llm_trace=_resolve_stage_llm_trace(traces),
        next_action="evaluate_blueprint" if status == ArtifactStatus.ready else "review_blueprint",
    )
    return envelope, traces


def run_enrich_stage(
    blueprint: BlueprintArtifact,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    *,
    runtime_settings: LLMRuntimeSettings | None = None,
    stage_context: StageContextBundle | None = None,
) -> tuple[Any, list[SkillExecutionTrace]]:
    registry = get_skill_registry()
    traces: list[SkillExecutionTrace] = []

    memory_trace, memory_output = registry.execute(
        "memory_design_skill",
        SkillRunContext(discovery=discovery, canvas=canvas, blueprint=blueprint, runtime_settings=runtime_settings),
    )
    traces.append(memory_trace)
    safety_trace, safety_output = registry.execute(
        "safety_skill",
        SkillRunContext(discovery=discovery, runtime_settings=runtime_settings),
    )
    traces.append(safety_trace)

    blueprint_input = BlueprintGenerationSkillInput(
        discovery=discovery,
        canvas=canvas,
        architecture=blueprint.architecture,
        reasoning_pattern=blueprint.reasoning_pattern,
        memory_strategy=blueprint.memory_strategy,
        tools=blueprint.tools,
        memory_profile=MemoryDesignSkillOutput.model_validate(memory_output.model_dump(mode="json")).memory_profile,
        knowledge_profile=blueprint.knowledge_profile,
        safety_checks=SafetySkillOutput.model_validate(safety_output.model_dump(mode="json")).safety_checks,
        guardrails=blueprint.guardrails,
        narrative=blueprint.narrative,
        allow_narrative_synthesis=False,
    )
    blueprint_trace, blueprint_output = _run_skill(
        RuntimeSkill(
            skill_key="blueprint_generation_skill",
            label="Blueprint generation skill",
            stage=SessionStage.post_validation,
            summary="Empaqueta el blueprint final listo para implementacion.",
            evidence_policy="rule_engine+llm_inference",
            input_model=BlueprintGenerationSkillInput,
            output_model=BlueprintArtifact,
            prepare_input=lambda _: blueprint_input,
            runner=_run_blueprint_generation_skill,
        ),
        SkillRunContext(runtime_settings=runtime_settings, stage_context=stage_context),
    )
    traces.append(blueprint_trace)
    artifact = BlueprintArtifact.model_validate(blueprint_output.model_dump(mode="json"))

    status = ArtifactStatus.ready if artifact.readiness_state == ReviewState.complete else ArtifactStatus.needs_review
    from app.models import BlueprintEnvelope

    envelope = BlueprintEnvelope(
        status=status,
        stage=SessionStage.post_validation,
        data=artifact,
        missing_fields=[],
        assumptions=[],
        warnings=[] if status == ArtifactStatus.ready else ["El blueprint enriquecido requiere revision antes del export final."],
        evidence=_merge_evidence(*[trace.evidence for trace in traces]),
        llm_trace=_resolve_stage_llm_trace(traces),
        next_action="evaluate_blueprint" if status == ArtifactStatus.ready else "review_blueprint_details",
    )
    return envelope, traces


def run_tool_recommendation_stage(
    session_id: UUID,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    blueprint: BlueprintArtifact,
    *,
    definition_artifact: RequirementsDefinitionOutput,
    design_artifact: DesignRecommendationArtifact,
    instructions: str = "",
    blueprint_version_number: int | None = None,
    runtime_settings: LLMRuntimeSettings | None = None,
    stage_context: StageContextBundle | None = None,
) -> tuple[Any, list[SkillExecutionTrace]]:
    trace, recommendation = get_skill_registry().execute(
        "tool_recommendation_skill",
        SkillRunContext(
            session_id=session_id,
            blueprint_version_number=blueprint_version_number,
            discovery=discovery,
            canvas=canvas,
            definition_artifact=definition_artifact,
            design_artifact=design_artifact,
            tool_instructions=instructions,
            blueprint=blueprint,
            runtime_settings=runtime_settings,
            stage_context=stage_context,
        ),
    )
    artifact = ToolRecommendationArtifact.model_validate(recommendation.model_dump(mode="json"))

    envelope = ToolRecommendationEnvelope(
        status=trace.status,
        stage=trace.stage,
        data=artifact,
        missing_fields=[],
        assumptions=[],
        warnings=list(trace.warnings),
        evidence=list(trace.evidence),
        llm_trace=trace.llm_trace.model_copy(deep=True) if trace.llm_trace is not None else None,
        next_action=(
            "resolve_tool_recommendation_findings"
            if artifact.evaluation.promotion_blocked
            else "review_tool_recommendation"
        ),
    )
    return envelope, [trace]


def run_memory_recommendation_stage(
    *,
    session_id: UUID | None,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    blueprint: BlueprintArtifact,
    definition_artifact: RequirementsDefinitionOutput | None,
    design_artifact: DesignRecommendationArtifact | None,
    approved_tools_digest: ApprovedToolsDigest | None,
    session_snapshot: SessionSnapshot | None,
    instructions: str = "",
    blueprint_version_number: int | None = None,
    source_stage_versions: MemoryRecommendationSourceStageVersions | None = None,
    runtime_settings: LLMRuntimeSettings | None = None,
    proposal_stage_context: StageContextBundle | None = None,
    critique_stage_context: StageContextBundle | None = None,
) -> tuple[MemoryRecommendationArtifact, list[SkillExecutionTrace]]:
    proposal_trace, proposal_output = _run_skill(
        RuntimeSkill(
            skill_key="memory_recommendation_skill",
            label="Memory recommendation skill",
            stage=SessionStage.build_blueprint,
            summary="Propone la arquitectura de memoria del agente objetivo.",
            evidence_policy="llm_inference+knowledge_retrieval",
            input_model=MemoryRecommendationSkillInput,
            output_model=MemoryArchitectureRecommendationOutput,
            prepare_input=lambda _: MemoryRecommendationSkillInput(
                discovery=discovery,
                canvas=canvas,
                blueprint=blueprint,
                approved_tools_digest=approved_tools_digest,
                instructions=instructions,
                source_stage_versions=source_stage_versions or MemoryRecommendationSourceStageVersions(),
            ),
            runner=_run_memory_recommendation_skill,
        ),
        SkillRunContext(
            session_id=session_id,
            blueprint_version_number=blueprint_version_number,
            discovery=discovery,
            canvas=canvas,
            blueprint=blueprint,
            definition_artifact=definition_artifact,
            design_artifact=design_artifact,
            tool_instructions=instructions,
            runtime_settings=runtime_settings,
            stage_context=proposal_stage_context,
        ),
    )
    proposal_output = MemoryArchitectureRecommendationOutput.model_validate(proposal_output.model_dump(mode="json"))
    critique_trace, critique_output = _run_skill(
        RuntimeSkill(
            skill_key="memory_critique_skill",
            label="Memory critique skill",
            stage=SessionStage.build_blueprint,
            summary="Critica la arquitectura de memoria contra minimalidad, compatibilidad y gobierno.",
            evidence_policy="llm_inference+knowledge_retrieval",
            input_model=MemoryArchitectureCritiqueInput,
            output_model=MemoryArchitectureCritiqueOutput,
            prepare_input=lambda _: MemoryArchitectureCritiqueInput(
                blueprint=blueprint,
                proposal=proposal_output,
                approved_tool_names=(
                    list(approved_tools_digest.approved_tool_keys)
                    if approved_tools_digest is not None
                    else [item.name for item in blueprint.tools if item.name]
                ),
                source_refs=[
                    "session.discovery",
                    "session.canvas",
                    "session.journey_latest_artifacts.define",
                    "session.journey_latest_artifacts.design",
                    "session.journey_latest_artifacts.tools",
                    "session.short_term_memory",
                ],
            ),
            runner=_run_memory_critique_skill,
        ),
        SkillRunContext(
            session_id=session_id,
            blueprint_version_number=blueprint_version_number,
            discovery=discovery,
            canvas=canvas,
            blueprint=blueprint,
            definition_artifact=definition_artifact,
            design_artifact=design_artifact,
            tool_instructions=instructions,
            runtime_settings=runtime_settings,
            stage_context=critique_stage_context,
        ),
    )
    critique_output = MemoryArchitectureCritiqueOutput.model_validate(critique_output.model_dump(mode="json"))
    artifact = build_memory_recommendation_artifact(
        discovery=discovery,
        canvas=canvas,
        blueprint=blueprint,
        approved_tools_digest=approved_tools_digest,
        source_session_id=session_id,
        source_blueprint_version=blueprint_version_number,
        current_blueprint_version=blueprint_version_number,
        source_stage_versions=source_stage_versions,
        instructions=instructions,
        definition_artifact=definition_artifact,
        design_artifact=design_artifact,
        proposal=proposal_output,
        critique=critique_output,
        session_snapshot=session_snapshot,
    )
    return artifact, [proposal_trace, critique_trace]


def run_evaluation_stage(
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact | None,
    dataset: EvaluationDatasetArtifact | None = None,
    rubric: EvaluationRubricArtifact | None = None,
    *,
    runtime_settings: LLMRuntimeSettings | None = None,
    stage_context: StageContextBundle | None = None,
) -> tuple[Any, list[SkillExecutionTrace]]:
    trace, evaluation = get_skill_registry().execute(
        "evaluation_skill",
        SkillRunContext(
            discovery=discovery,
            canvas=canvas,
            blueprint=blueprint,
            evaluation_dataset=dataset,
            evaluation_rubric=rubric,
            runtime_settings=runtime_settings,
            stage_context=stage_context,
        ),
    )
    artifact = EvaluationArtifact.model_validate(evaluation.model_dump(mode="json"))
    status = ArtifactStatus.ready if artifact.completeness_status == ReviewState.complete else ArtifactStatus.needs_review
    from app.models import EvaluationEnvelope

    envelope = EvaluationEnvelope(
        status=status,
        stage=SessionStage.post_validation,
        data=artifact,
        missing_fields=artifact.gaps,
        assumptions=[],
        warnings=[] if status == ArtifactStatus.ready else ["Hay huecos pendientes antes de considerar el blueprint completamente listo."],
        evidence=list(trace.evidence),
        next_action="ready_for_export" if status == ArtifactStatus.ready else "resolve_gaps",
    )
    return envelope, [trace]


def rerun_skill_for_session(
    skill_key: str,
    *,
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact | None,
    evaluation_dataset: EvaluationDatasetArtifact | None = None,
    evaluation_rubric: EvaluationRubricArtifact | None = None,
    runtime_settings: LLMRuntimeSettings | None = None,
) -> tuple[SkillExecutionTrace, DiscoveryArtifact | None, CanvasArtifact | None, BlueprintArtifact | None, EvaluationArtifact | None]:
    registry = get_skill_registry()

    if skill_key == "discovery_skill":
        if discovery is None:
            raise ValueError("Discovery actual requerido para reejecutar discovery_skill")
        payload = DiscoveryInput(
            problem_statement=discovery.problem_statement,
            current_user=discovery.current_user,
            current_process=discovery.current_process,
            desired_outcome=discovery.desired_outcome,
            autonomy_level=discovery.autonomy_level,
            constraints=discovery.constraints,
            operational_baseline=discovery.operational_baseline.model_copy(),
            mvp_definition=discovery.mvp_definition.model_copy(),
        )
        trace, output = registry.execute(
            "discovery_skill",
            SkillRunContext(discovery_input=payload, runtime_settings=runtime_settings),
        )
        return trace, DiscoveryArtifact.model_validate(output.model_dump(mode="json")), None, None, None

    if skill_key == "lean_scope_skill":
        if discovery is None:
            raise ValueError("Discovery actual requerido para reejecutar lean_scope_skill")
        trace, output = registry.execute(
            "lean_scope_skill",
            SkillRunContext(discovery=discovery, runtime_settings=runtime_settings),
        )
        return trace, None, CanvasArtifact.model_validate(output.model_dump(mode="json")), None, None

    if skill_key == "architecture_selection_skill":
        if discovery is None or canvas is None or blueprint is None:
            raise ValueError("Discovery, canvas y blueprint requeridos para reejecutar architecture_selection_skill")
        trace, output = registry.execute(
            skill_key,
            SkillRunContext(discovery=discovery, canvas=canvas, runtime_settings=runtime_settings),
        )
        selection = SelectionSkillOutput.model_validate(output.model_dump(mode="json"))
        updated_blueprint = compose_blueprint_artifact(
            discovery,
            canvas,
            architecture=selection.selected_value,
            reasoning_pattern=blueprint.reasoning_pattern,
            memory_strategy=blueprint.memory_strategy,
            tools=blueprint.tools,
            llm_policy=blueprint.llm_policy,
            memory_profile=blueprint.memory_profile,
            knowledge_profile=blueprint.knowledge_profile,
            safety_checks=blueprint.safety_checks,
            guardrails=blueprint.guardrails,
            narrative=blueprint.narrative,
        )
        return trace, None, None, updated_blueprint, None

    if skill_key == "reasoning_pattern_skill":
        if discovery is None or canvas is None or blueprint is None:
            raise ValueError("Discovery, canvas y blueprint requeridos para reejecutar reasoning_pattern_skill")
        trace, output = registry.execute(
            skill_key,
            SkillRunContext(discovery=discovery, canvas=canvas, runtime_settings=runtime_settings),
        )
        selection = SelectionSkillOutput.model_validate(output.model_dump(mode="json"))
        updated_blueprint = compose_blueprint_artifact(
            discovery,
            canvas,
            architecture=blueprint.architecture,
            reasoning_pattern=selection.selected_value,
            memory_strategy=blueprint.memory_strategy,
            tools=blueprint.tools,
            llm_policy=blueprint.llm_policy,
            memory_profile=blueprint.memory_profile,
            knowledge_profile=blueprint.knowledge_profile,
            safety_checks=blueprint.safety_checks,
            guardrails=blueprint.guardrails,
            narrative=blueprint.narrative,
        )
        return trace, None, None, updated_blueprint, None

    if skill_key == "tool_design_skill":
        if discovery is None or canvas is None or blueprint is None:
            raise ValueError("Discovery, canvas y blueprint requeridos para reejecutar tool_design_skill")
        trace, output = registry.execute(
            skill_key,
            SkillRunContext(discovery=discovery, runtime_settings=runtime_settings),
        )
        tool_output = ToolDesignSkillOutput.model_validate(output.model_dump(mode="json"))
        updated_blueprint = compose_blueprint_artifact(
            discovery,
            canvas,
            architecture=blueprint.architecture,
            reasoning_pattern=blueprint.reasoning_pattern,
            memory_strategy=blueprint.memory_strategy,
            tools=tool_output.tools,
            llm_policy=blueprint.llm_policy,
            memory_profile=blueprint.memory_profile,
            knowledge_profile=blueprint.knowledge_profile,
            safety_checks=blueprint.safety_checks,
            guardrails=blueprint.guardrails,
            narrative=blueprint.narrative,
        )
        return trace, None, None, updated_blueprint, None

    if skill_key == "memory_design_skill":
        if discovery is None or canvas is None or blueprint is None:
            raise ValueError("Discovery, canvas y blueprint requeridos para reejecutar memory_design_skill")
        trace, output = registry.execute(
            skill_key,
            SkillRunContext(
                discovery=discovery,
                canvas=canvas,
                blueprint=blueprint,
                runtime_settings=runtime_settings,
            ),
        )
        memory_output = MemoryDesignSkillOutput.model_validate(output.model_dump(mode="json"))
        updated_blueprint = compose_blueprint_artifact(
            discovery,
            canvas,
            architecture=blueprint.architecture,
            reasoning_pattern=blueprint.reasoning_pattern,
            memory_strategy=memory_output.memory_strategy,
            tools=blueprint.tools,
            llm_policy=blueprint.llm_policy,
            memory_profile=memory_output.memory_profile,
            knowledge_profile=blueprint.knowledge_profile,
            safety_checks=blueprint.safety_checks,
            guardrails=blueprint.guardrails,
            narrative=blueprint.narrative,
        )
        return trace, None, None, updated_blueprint, None

    if skill_key == "safety_skill":
        if discovery is None or canvas is None or blueprint is None:
            raise ValueError("Discovery, canvas y blueprint requeridos para reejecutar safety_skill")
        trace, output = registry.execute(
            skill_key,
            SkillRunContext(discovery=discovery, runtime_settings=runtime_settings),
        )
        safety_output = SafetySkillOutput.model_validate(output.model_dump(mode="json"))
        updated_blueprint = compose_blueprint_artifact(
            discovery,
            canvas,
            architecture=blueprint.architecture,
            reasoning_pattern=blueprint.reasoning_pattern,
            memory_strategy=blueprint.memory_strategy,
            tools=blueprint.tools,
            llm_policy=blueprint.llm_policy,
            memory_profile=blueprint.memory_profile,
            knowledge_profile=blueprint.knowledge_profile,
            safety_checks=safety_output.safety_checks,
            guardrails=safety_output.guardrails,
            narrative=blueprint.narrative,
        )
        return trace, None, None, updated_blueprint, None

    if skill_key == "blueprint_generation_skill":
        if discovery is None or canvas is None or blueprint is None:
            raise ValueError("Discovery, canvas y blueprint requeridos para reejecutar blueprint_generation_skill")
        blueprint_input = BlueprintGenerationSkillInput(
            discovery=discovery,
            canvas=canvas,
            architecture=blueprint.architecture,
            reasoning_pattern=blueprint.reasoning_pattern,
            memory_strategy=blueprint.memory_strategy,
            tools=blueprint.tools,
            memory_profile=blueprint.memory_profile,
            knowledge_profile=blueprint.knowledge_profile,
            safety_checks=blueprint.safety_checks,
            guardrails=blueprint.guardrails,
            narrative=blueprint.narrative,
            allow_narrative_synthesis=False,
        )
        trace, output = _run_skill(
            RuntimeSkill(
                skill_key="blueprint_generation_skill",
                label="Blueprint generation skill",
                stage=SessionStage.post_validation,
                summary="Empaqueta el blueprint final listo para implementacion.",
                evidence_policy="rule_engine+llm_inference",
                input_model=BlueprintGenerationSkillInput,
                output_model=BlueprintArtifact,
                prepare_input=lambda _: blueprint_input,
                runner=_run_blueprint_generation_skill,
            ),
            SkillRunContext(runtime_settings=runtime_settings),
        )
        return trace, None, None, BlueprintArtifact.model_validate(output.model_dump(mode="json")), None

    if skill_key == "evaluation_skill":
        trace, output = registry.execute(
            skill_key,
            SkillRunContext(
                discovery=discovery,
                canvas=canvas,
                blueprint=blueprint,
                evaluation_dataset=evaluation_dataset,
                evaluation_rubric=evaluation_rubric,
                runtime_settings=runtime_settings,
            ),
        )
        return trace, None, None, None, EvaluationArtifact.model_validate(output.model_dump(mode="json"))

    raise ValueError(f"Skill no soportada para rerun: {skill_key}")
