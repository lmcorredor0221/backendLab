from __future__ import annotations

from typing import Any
from uuid import UUID

from app.models import (
    ACPFileEntry,
    ACPPreview,
    CommercialTier,
    DiagramAccessPolicy,
    DiagramCatalogEntry,
    DiagramCatalogResponse,
    DiagramContentResponse,
    DiagramUpsellMessage,
    SessionSnapshot,
)
from app.services.commercial_access import CommercialEntitlementContext, resolve_capability_access, tier_rank
from app.services.diagram_access_policy_service import get_diagram_access_policy, list_diagram_access_policies


LEAN_STAGE_ORDER = ("discover", "define", "design", "tools", "memory", "estimate", "validate", "package")
SESSION_STAGE_TO_LEAN_STAGE = {
    "draft_capture": "discover",
    "input_validation": "discover",
    "normalize_discovery": "discover",
    "build_canvas": "define",
    "build_blueprint": "design",
    "post_validation": "validate",
    "ready_for_export": "package",
}


DIAGRAM_CONTENT_PATHS: dict[str, dict[str, list[str]]] = {
    "architecture_overview": {
        "svg": ["ACP/svg/Architecture.svg", "ACP/svg/ExecutiveCanvas.svg"],
        "mermaid": ["ACP/mermaid/Architecture.mmd", "ACP/mermaid/ExecutiveCanvas.mmd"],
        "markdown": ["ACP/diagrams/Architecture.md", "ACP/diagrams/ExecutiveCanvas.md"],
    },
    "c4_context": {
        "mermaid": ["ACP/mermaid/Architecture.mmd"],
        "svg": ["ACP/svg/Architecture.svg"],
        "markdown": ["ACP/architecture/c4-context.md", "ACP/diagrams/Architecture.md"],
    },
    "c4_container": {
        "mermaid": ["ACP/mermaid/DataModel.mmd"],
        "svg": ["ACP/svg/DataModel.svg"],
        "markdown": ["ACP/diagrams/DataModel.md"],
    },
    "agent_orchestration": {
        "svg": ["ACP/svg/AgentLoop.svg", "ACP/svg/Workflow.svg"],
        "mermaid": ["ACP/mermaid/AgentLoop.mmd", "ACP/mermaid/Workflow.mmd"],
        "yaml": ["ACP/workflows/durable-workflow.yaml"],
    },
    "runtime_workflow": {
        "mermaid": ["ACP/mermaid/Workflow.mmd", "ACP/mermaid/StateMachine.mmd"],
        "svg": ["ACP/svg/Workflow.svg", "ACP/svg/StateMachine.svg"],
        "yaml": ["ACP/workflows/state-machine.yaml", "ACP/workflows/durable-workflow.yaml"],
    },
    "decision_model": {
        "mermaid": ["ACP/mermaid/StateMachine.mmd"],
        "svg": ["ACP/svg/StateMachine.svg"],
        "markdown": ["ACP/diagrams/StateMachine.md"],
        "yaml": ["ACP/cognition/guardrails.yaml"],
    },
    "tool_capability_map": {
        "svg": ["ACP/svg/ToolMap.svg", "ACP/svg/CapabilityMap.svg"],
        "mermaid": ["ACP/mermaid/ToolMap.mmd", "ACP/mermaid/CapabilityMap.mmd"],
        "yaml": ["ACP/tools/permissions.yaml"],
    },
    "tool_contract_sequence": {
        "mermaid": ["ACP/mermaid/ToolMap.mmd"],
        "svg": ["ACP/svg/ToolMap.svg"],
        "yaml": ["ACP/tools/permissions.yaml"],
    },
    "memory_rag_architecture": {
        "svg": ["ACP/svg/Memory.svg"],
        "mermaid": ["ACP/mermaid/Memory.mmd"],
        "json": ["ACP/blueprint.graph.json"],
    },
    "knowledge_graph": {
        "json": ["ACP/blueprint.graph.json"],
        "svg": ["ACP/svg/KnowledgeGraph.svg"],
        "graphml": ["ACP/blueprint.graphml"],
        "cypher": ["ACP/blueprint.cypher"],
    },
    "data_contracts": {
        "svg": ["ACP/svg/DataModel.svg"],
        "json_schema": ["ACP/evaluation/golden-dataset.json"],
        "yaml": ["ACP/tools/permissions.yaml", "ACP/workflows/state-machine.yaml"],
    },
    "security_guardrails": {
        "svg": ["ACP/svg/DependencyGraph.svg", "ACP/svg/Traceability.svg"],
        "mermaid": ["ACP/mermaid/DependencyGraph.mmd", "ACP/mermaid/Traceability.mmd"],
        "markdown": ["ACP/diagrams/DependencyGraph.md"],
        "yaml": ["ACP/cognition/guardrails.yaml"],
    },
    "integration_boundaries": {
        "mermaid": ["ACP/mermaid/Integrations.mmd"],
        "svg": ["ACP/svg/Integrations.svg"],
        "markdown": ["ACP/diagrams/Integrations.md"],
        "yaml": ["ACP/runtime/providers.yaml"],
    },
    "deployment_decision_matrix": {
        "yaml": ["ACP/deployment/env.template", "ACP/deployment/docker-compose.yaml"],
        "markdown": ["ACP/deployment/cicd/README.md", "ACP/deployment/kubernetes/README.md"],
        "svg": ["ACP/svg/Infrastructure.svg"],
    },
    "evaluation_test_suite": {
        "yaml": ["ACP/evaluation/benchmarks.yaml", "ACP/evaluation/rubrics.yaml"],
        "svg": ["ACP/svg/ConstructionFlow.svg"],
        "mermaid": ["ACP/mermaid/ConstructionFlow.mmd"],
    },
    "producer_lineage": {
        "json": ["ACP/blueprint.manifest.json"],
        "graphml": ["ACP/blueprint.graphml"],
        "cypher": ["ACP/blueprint.cypher"],
    },
    "prompt_reasoning_playbook": {
        "mermaid": ["ACP/diagrams/implementation/prompt-reasoning-playbook.mmd"],
        "markdown": ["ACP/prompts/README.md"],
        "yaml": ["ACP/prompts/prompt-pack.yaml"],
    },
    "human_intervention_flow": {
        "mermaid": ["Blueprint/diagrams/human-intervention-flow.mmd"],
        "svg": ["Blueprint/diagrams/human-intervention-flow.svg"],
        "yaml": ["ACP/workflows/human-intervention.yaml"],
    },
    "rag_ingestion_pipeline": {
        "mermaid": ["ACP/diagrams/implementation/rag-ingestion-pipeline.mmd"],
        "yaml": ["ACP/knowledge/rag-ingestion-pipeline.yaml"],
        "svg": ["ACP/diagrams/implementation/rag-ingestion-pipeline.svg"],
    },
    "short_term_context_budget": {
        "mermaid": ["Blueprint/diagrams/context-budget.mmd"],
        "json": ["ACP/memory/context-budget.json"],
        "markdown": ["ACP/memory/context-budget.md"],
    },
    "observability_event_model": {
        "yaml": ["ACP/observability/event-model.yaml"],
        "mermaid": ["ACP/diagrams/implementation/observability-event-model.mmd"],
        "json": ["ACP/observability/event-model.json"],
    },
    "api_integration_sequence": {
        "mermaid": ["ACP/diagrams/implementation/api-integration-sequence.mmd"],
        "yaml": ["ACP/integrations/api-contracts.yaml"],
        "svg": ["ACP/diagrams/implementation/api-integration-sequence.svg"],
    },
    "data_lineage_map": {
        "json": ["Blueprint/diagrams/data-lineage-map.json", "ACP/knowledge/data-lineage-map.json"],
        "graphml": ["ACP/knowledge/data-lineage-map.graphml"],
        "mermaid": ["Blueprint/diagrams/data-lineage-map.mmd"],
    },
    "commercial_value_flow": {
        "svg": ["Blueprint/diagrams/commercial-value-flow.svg"],
        "mermaid": ["Blueprint/diagrams/commercial-value-flow.mmd"],
        "markdown": ["Blueprint/diagrams/commercial-value-flow.md"],
    },
}


def _stage_index(stage: str) -> int:
    try:
        return LEAN_STAGE_ORDER.index(stage)
    except ValueError:
        return -1


def _state_value(value: Any) -> str:
    return getattr(value, "value", str(value or ""))


def _session_stage_value(snapshot: SessionSnapshot) -> str:
    return getattr(snapshot.session.current_stage, "value", str(snapshot.session.current_stage or ""))


def resolve_current_diagram_stage(snapshot: SessionSnapshot) -> str:
    stage = SESSION_STAGE_TO_LEAN_STAGE.get(_session_stage_value(snapshot), "discover")
    for stage_key, artifact in snapshot.journey_latest_artifacts.items():
        if stage_key in LEAN_STAGE_ORDER and _state_value(artifact.state) in {
            "generated",
            "reviewed",
            "approved",
            "approved_legacy",
        }:
            if _stage_index(stage_key) > _stage_index(stage):
                stage = stage_key
    if snapshot.estimation_report is not None and _stage_index("estimate") > _stage_index(stage):
        stage = "estimate"
    if snapshot.simulation_runs and _stage_index("validate") > _stage_index(stage):
        stage = "validate"
    return stage


def _has_reached_stage(current_stage: str, required_stage: str) -> bool:
    return _stage_index(current_stage) >= _stage_index(required_stage)


def _file_by_path(preview: ACPPreview) -> dict[str, ACPFileEntry]:
    return {item.path: item for item in preview.files}


def _candidate_paths(policy: DiagramAccessPolicy) -> dict[str, list[str]]:
    configured = DIAGRAM_CONTENT_PATHS.get(policy.diagram_key, {})
    return {format_key: list(configured.get(format_key, [])) for format_key in policy.available_formats}


def _existing_content_paths(policy: DiagramAccessPolicy, preview: ACPPreview) -> dict[str, str]:
    file_map = _file_by_path(preview)
    resolved: dict[str, str] = {}
    for format_key, paths in _candidate_paths(policy).items():
        for path in paths:
            if path in file_map:
                resolved[format_key] = path
                break
    return resolved


def _generation_state(policy: DiagramAccessPolicy, preview: ACPPreview) -> str:
    if _existing_content_paths(policy, preview):
        return "generated"
    return policy.default_generation_state if policy.default_generation_state != "generated" else "pending_generation"


def _capability_for_policy(policy: DiagramAccessPolicy) -> str:
    if policy.required_tier == CommercialTier.acp:
        return "diagram.view.acp"
    if policy.required_tier == CommercialTier.blueprint_pro:
        return "diagram.view.blueprint"
    return "diagram.view.sample"


def _locked_state(policy: DiagramAccessPolicy) -> str:
    return "locked_acp" if policy.required_tier == CommercialTier.acp else "locked_blueprint"


def _upsell_for_state(policy: DiagramAccessPolicy, access_state: str) -> DiagramUpsellMessage | None:
    if access_state == "unlocked":
        return None
    if access_state == "sample":
        return DiagramUpsellMessage(
            title="Muestra del valor generado",
            message="Este diagrama es una muestra protegida. Adquiere Blueprint Profesional para ver todo el catalogo.",
            cta_label="Adquirir Blueprint Profesional",
            target_tier=CommercialTier.blueprint_pro,
            product="blueprint",
        )
    if access_state == "stage_locked":
        return DiagramUpsellMessage(
            title="Diagrama disponible en una etapa posterior",
            message=policy.upsell.get(
                "stage_locked_message",
                f"Este diagrama se habilita al llegar a la etapa {policy.enabled_from_stage}.",
            ),
            cta_label="Continuar el flujo",
            target_tier=policy.required_tier,
            product=policy.product_scope[0] if policy.product_scope else "blueprint",
        )
    if access_state == "not_generated":
        return DiagramUpsellMessage(
            title="Diagrama pendiente de generacion",
            message="La plataforma ya conoce este activo, pero aun falta generar la evidencia necesaria para renderizarlo.",
            cta_label="Generar artefactos previos",
            target_tier=policy.required_tier,
            product=policy.product_scope[0] if policy.product_scope else "blueprint",
        )
    if access_state == "locked_acp":
        return DiagramUpsellMessage(
            title="Contenido premium ACP",
            message=policy.upsell.get(
                "locked_acp_message",
                "Este diagrama hace parte del Agent Construction Package (ACP). Adquiere el ACP para acceder a los artefactos de implementacion y construccion.",
            ),
            cta_label="Adquirir ACP",
            target_tier=CommercialTier.acp,
            product="acp",
        )
    return DiagramUpsellMessage(
        title="Contenido Blueprint Profesional",
        message=policy.upsell.get(
            "locked_blueprint_message",
            "Este diagrama forma parte del Blueprint Profesional. Adquierelo para acceder a la documentacion completa y al diseno integral.",
        ),
        cta_label="Adquirir Blueprint Profesional",
        target_tier=CommercialTier.blueprint_pro,
        product="blueprint",
    )


def _access_state(
    policy: DiagramAccessPolicy,
    *,
    context: CommercialEntitlementContext,
    current_stage: str,
    generation_state: str,
) -> str:
    if not _has_reached_stage(current_stage, policy.enabled_from_stage):
        return "stage_locked"
    if generation_state != "generated":
        return "not_generated"
    if policy.sample_enabled and policy.access_level == "sample" and context.tier == CommercialTier.blueprint:
        return "sample"
    decision = resolve_capability_access(context, _capability_for_policy(policy))
    if decision.allowed:
        return "unlocked"
    if policy.sample_enabled and tier_rank(context.tier) >= tier_rank(policy.sample_tier):
        return "sample"
    return _locked_state(policy)


def _locked_reason(access_state: str, policy: DiagramAccessPolicy) -> str:
    if access_state == "unlocked":
        return ""
    if access_state == "sample":
        return "Muestra protegida disponible en el nivel actual."
    if access_state == "stage_locked":
        return f"Se habilita al llegar a la etapa {policy.enabled_from_stage}."
    if access_state == "not_generated":
        return "El diagrama esta catalogado, pero todavia no existe contenido generado para esta sesion."
    if access_state == "locked_acp":
        return "Requiere adquirir ACP Premium."
    return "Requiere adquirir Blueprint Profesional."


def build_diagram_catalog(
    *,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    context: CommercialEntitlementContext,
    workspace_id: UUID,
) -> DiagramCatalogResponse:
    current_stage = resolve_current_diagram_stage(snapshot)
    entries: list[DiagramCatalogEntry] = []
    for policy in list_diagram_access_policies():
        existing_paths = _existing_content_paths(policy, preview)
        generation_state = _generation_state(policy, preview)
        access_state = _access_state(
            policy,
            context=context,
            current_stage=current_stage,
            generation_state=generation_state,
        )
        exposed_source_paths = [existing_paths[key] for key in sorted(existing_paths)] if access_state == "unlocked" else []
        entries.append(
            DiagramCatalogEntry(
                diagram_key=policy.diagram_key,
                title=policy.title,
                category=policy.category,
                summary=policy.description,
                diagram_surface=policy.diagram_surface,
                product_scope=policy.product_scope,
                required_tier=policy.required_tier,
                enabled_from_stage=policy.enabled_from_stage,
                generation_state=generation_state,
                access_state=access_state,
                locked_reason=_locked_reason(access_state, policy),
                upgrade_cta_label=(_upsell_for_state(policy, access_state).cta_label if _upsell_for_state(policy, access_state) else ""),
                preferred_format=policy.preferred_format,
                available_formats=policy.available_formats,
                available_content_formats=sorted(existing_paths.keys()),
                source_artifact_count=len(policy.source_artifact_keys),
                source_paths=exposed_source_paths,
                protection=policy.content_protection,
                upsell=_upsell_for_state(policy, access_state),
            )
        )
    return DiagramCatalogResponse(
        session_id=snapshot.session.id,
        workspace_id=workspace_id,
        current_stage=current_stage,
        tier=context.tier,
        total_count=len(entries),
        unlocked_count=sum(1 for item in entries if item.access_state == "unlocked"),
        locked_count=sum(1 for item in entries if item.access_state in {"locked_blueprint", "locked_acp", "stage_locked"}),
        sample_count=sum(1 for item in entries if item.access_state == "sample"),
        pending_count=sum(1 for item in entries if item.access_state == "not_generated"),
        entries=entries,
    )


def _resolve_content_path(policy: DiagramAccessPolicy, preview: ACPPreview, requested_format: str | None) -> tuple[str, str] | None:
    existing_paths = _existing_content_paths(policy, preview)
    if not existing_paths:
        return None
    candidates = []
    if requested_format:
        candidates.append(requested_format.strip())
    candidates.extend([policy.preferred_format, *policy.available_formats, *sorted(existing_paths)])
    for candidate in candidates:
        if candidate in existing_paths:
            return candidate, existing_paths[candidate]
    return None


def build_diagram_content(
    *,
    diagram_key: str,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    context: CommercialEntitlementContext,
    workspace_id: UUID,
    requested_format: str | None = None,
) -> DiagramContentResponse | None:
    policy = get_diagram_access_policy(diagram_key)
    if policy is None:
        return None
    catalog = build_diagram_catalog(snapshot=snapshot, preview=preview, context=context, workspace_id=workspace_id)
    entry = next(item for item in catalog.entries if item.diagram_key == policy.diagram_key)
    resolved = _resolve_content_path(policy, preview, requested_format)
    content_format = requested_format.strip() if requested_format else policy.preferred_format
    content = None
    source_path = ""
    content_hash = ""
    if entry.access_state in {"unlocked", "sample"} and resolved is not None:
        content_format, source_path = resolved
        file_entry = _file_by_path(preview)[source_path]
        content = file_entry.content_text
        content_hash = file_entry.content_hash
    return DiagramContentResponse(
        diagram_key=policy.diagram_key,
        access_state=entry.access_state,
        generation_state=entry.generation_state,
        format=content_format,
        content=content,
        asset_url=None,
        protection=entry.protection,
        upsell=entry.upsell,
        metadata={
            "workspace_id": str(workspace_id),
            "session_id": str(snapshot.session.id),
            "current_stage": catalog.current_stage,
            "title": policy.title,
            "category": policy.category,
            "diagram_surface": policy.diagram_surface,
            "product_scope": policy.product_scope,
            "required_tier": policy.required_tier,
            "source_path": source_path if entry.access_state == "unlocked" else "",
            "content_hash": content_hash,
            "source_artifact_keys": policy.source_artifact_keys,
            "available_formats": entry.available_formats,
            "available_content_formats": entry.available_content_formats,
        },
    )
