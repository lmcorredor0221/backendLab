from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from app.contracts.canonical_v1 import (
    AcpV2AgentRuntime,
    AcpV2BuildPlan,
    AcpV2BuildStep,
    AcpV2CheckpointSpec,
    AcpV2CapabilityContract,
    AcpV2CompatibilityRule,
    AcpV2ConformanceRule,
    AcpV2DecisionOption,
    AcpV2DecisionRegistryEntry,
    AcpV2ImplementationDecision,
    AcpV2ContextWindowPolicy,
    AcpV2KnowledgeArtifactRef,
    AcpV2KnowledgeSource,
    AcpV2ManifestContractEntry,
    AcpV2MemoryKnowledgePlan,
    AcpV2MemoryNamespace,
    AcpV2MemoryStrategy,
    AcpV2MigrationInfo,
    AcpV2PortableManifest,
    AcpV2ProducerMetadata,
    AcpV2PromptRef,
    AcpV2RagCapabilityDependency,
    AcpV2RagPipelineSpec,
    AcpV2RuntimeAgent,
    AcpV2RuntimeTarget,
    AcpV2RuntimeTargetPolicy,
    AcpV2DeploymentGuide,
    AcpV2DeploymentGuideStep,
    AcpV2TechnologyDecision,
    AcpV2TechnologyOption,
    AcpV2TestAsset,
    AcpV2ToolAnalysis,
    AcpV2ToolContractRef,
    AcpV2ToolBinding,
    AcpV2ToolIncompatibility,
    AcpV2ToolRedundancy,
    AcpV2WorkflowNode,
    AcpV2WorkflowSpec,
    AcpV2WorkflowTransition,
    AgentConstructionPackageV2,
    ApprovalGateSummary,
    BehaviorSpecV1,
    BehaviorState,
    BlueprintCoreV1,
    BlueprintIdentity,
    BlueprintPurpose,
    BlueprintScope,
    CanonicalDependency,
    CanonicalOpenQuestion,
    CanonicalProvenanceEntry,
    ConstructionComponent,
    ConstructionFileManifestEntry,
    ConstructionPackV1,
    ConstructionReadinessV1,
    ContractReference,
    EstimationPackV1,
    EstimationSensitivityDriver,
    EvaluationCaseV1,
    EvaluationPackV1,
    HeuristicDecisionFact,
    HeuristicDecisionV1,
    KnowledgeManifestSourceV1,
    KnowledgeManifestV1,
    KnowledgeEmbeddingPolicyV1,
    KnowledgeContractV1,
    KnowledgeIngestionPolicyV1,
    KnowledgeRefreshPolicyV1,
    KnowledgeRetrievalPolicyV1,
    KnowledgeSourceRef,
    LLMFunctionPolicy,
    LLMPolicyV1,
    MemoryContextBudgetV1,
    MemoryPolicyV1,
    PromptArtifactV1,
    PromptPackOrigin,
    PromptPackV1,
    PromptVariable,
    ReadinessGapEntry,
    RiskEntry,
    StableIssueCatalogEntryV1,
    SuccessCriterion,
    TestPackAcceptanceJourneyV1,
    TestPackCommandV1,
    TestPackExternalConsumerV1,
    TestPackFixtureRef,
    TestPackMutationCaseV1,
    TestPackPromptEvaluationCaseV1,
    TestPackRecoveryCaseV1,
    TestPackV1,
    ToolContractV1,
    ToolSchemaField,
    ToolSchemaShape,
    ShortTermMemoryCompactionV1,
    ShortTermMemoryNamespaceV1,
    ShortTermMemoryRefV1,
    ShortTermMemoryV1,
)
from app.models import ApprovalStatus, ArtifactStatus, ReviewState, SessionSnapshot, utc_now
from app.services.acp_construction_readiness import CONSTRUCTION_GAP_CATALOG
from app.services.acp_validation import VALIDATION_ISSUE_CATALOG
from app.services.blueprint_consistency_service import ensure_blueprint_consistency_report
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionOutput
from app.services.stage4_compiler import compile_stage4_artifacts

KNOWLEDGE_KEYWORDS = (
    "knowledge",
    "rag",
    "retrieval",
    "runbook",
    "faq",
    "base_conocimiento",
    "documentacion",
    "wiki",
    "confluence",
)

MEMORY_TAXONOMY_PATH = Path(__file__).resolve().parents[3] / "Docs" / "system-analysis" / "29-memory-m0-taxonomy-manifest.json"

MEMORY_CONTEXT_BUDGET_DEFAULTS = (
    ("planner", 2400, 8, 12000),
    ("executor", 1800, 10, 9000),
    ("evaluator", 2200, 8, 11000),
    ("tool_use", 1200, 6, 6000),
    ("memory", 1600, 12, 8000),
    ("retrieval", 1400, 6, 7000),
    ("recovery", 1200, 6, 6000),
)


def _base_metadata(
    snapshot: SessionSnapshot,
    generated_at,
    *,
    source_blueprint_version: int | None = None,
) -> dict[str, Any]:
    blueprint_version = (
        source_blueprint_version
        if source_blueprint_version is not None
        else latest_blueprint_version(snapshot)
    )
    return {
        "source_session_id": snapshot.session.id,
        "generated_at": generated_at,
        "source_blueprint_version": blueprint_version,
    }


def _provenance(*entries: tuple[str, list[str], str]) -> list[CanonicalProvenanceEntry]:
    return [
        CanonicalProvenanceEntry(target_path=target_path, source_paths=source_paths, note=note)
        for target_path, source_paths, note in entries
    ]


def _normalized_items(items: list[str] | None) -> list[str]:
    return [item.strip() for item in items or [] if isinstance(item, str) and item.strip()]


def _tool_schema(fields: list[str]) -> ToolSchemaShape:
    properties = {
        field_name: ToolSchemaField(
            type="string",
            description=f"Campo requerido por la herramienta: {field_name}.",
        )
        for field_name in _normalized_items(fields)
    }
    return ToolSchemaShape(properties=properties, required=list(properties.keys()))


def _coalesce(*values: str | None, fallback: str = "") -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _contains_keywords(*values: str | None) -> bool:
    haystack = " ".join(value.strip().lower() for value in values if isinstance(value, str) and value.strip())
    return any(keyword in haystack for keyword in KNOWLEDGE_KEYWORDS)


def _tool_is_idempotent(idempotency_strategy: str | None, has_side_effects: bool) -> bool:
    if not has_side_effects:
        return True
    normalized = _coalesce(idempotency_strategy, fallback="").lower()
    if not normalized:
        return False
    return not any(flag in normalized for flag in ("no idempot", "non idempot", "manual only"))


def latest_blueprint_version(snapshot: SessionSnapshot) -> int | None:
    versions = snapshot.blueprint_versions or []
    if not versions:
        return None
    return max(item.version_number for item in versions)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _serialize_datetime(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value) if value is not None else ""


def _session_stage(snapshot: SessionSnapshot) -> str:
    return snapshot.session.current_stage.value if hasattr(snapshot.session.current_stage, "value") else str(snapshot.session.current_stage)


def _stage_affinity(snapshot: SessionSnapshot) -> str:
    stage = _session_stage(snapshot)
    if stage in {"draft_capture", "input_validation", "normalize_discovery"}:
        return "discover"
    if stage == "build_canvas":
        return "define"
    if stage == "build_blueprint":
        return "design"
    if stage == "post_validation":
        return "evaluate"
    if stage == "ready_for_export":
        return "build"
    return stage


def _memory_context_budgets() -> list[MemoryContextBudgetV1]:
    return [
        MemoryContextBudgetV1(
            role=role,
            max_tokens=max_tokens,
            max_items=max_items,
            max_chars=max_chars,
            compaction_trigger="Resumir cuando se supere cualquiera de los limites antes de reenviar contexto.",
            overflow_policy="compact_by_checkpoint_then_retrieve_by_reference",
        )
        for role, max_tokens, max_items, max_chars in MEMORY_CONTEXT_BUDGET_DEFAULTS
    ]


def _memory_retrieval_scopes(snapshot: SessionSnapshot, *, knowledge_enabled: bool) -> list[str]:
    scopes = [
        "session.short_term.summary_cache",
        "session.checkpoints.stage",
        "session.blueprint.current",
    ]
    if snapshot.blueprint_versions:
        scopes.append("session.blueprint.versions")
    if snapshot.artifact_records:
        scopes.append("session.artifact_registry")
    if snapshot.skill_runs:
        scopes.append("session.skill_runs")
    if snapshot.approvals:
        scopes.append("session.approvals")
    if snapshot.handoff_records or snapshot.subagent_runs:
        scopes.append("session.branch_board")
    scopes.append("knowledge.required_sources")
    scopes.append("knowledge.candidate_sources")
    if knowledge_enabled:
        scopes.append("knowledge.approved_sources")
    return _dedupe_preserve_order(scopes)


def _load_memory_taxonomy_rules() -> tuple[str, list[dict[str, Any]]]:
    if not MEMORY_TAXONOMY_PATH.exists():
        return "", []
    try:
        payload = json.loads(MEMORY_TAXONOMY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", []
    generated_at = str(payload.get("generated_at", ""))
    rules = payload.get("rules", [])
    return generated_at, rules if isinstance(rules, list) else []


def _rule_uri(rule: dict[str, Any]) -> str:
    include_prefixes = rule.get("include_prefixes")
    if isinstance(include_prefixes, list) and include_prefixes:
        return f"repo://{include_prefixes[0]}"
    include_filenames = rule.get("include_filenames")
    if isinstance(include_filenames, list) and include_filenames:
        return f"repo://Docs/{include_filenames[0]}"
    include_suffixes = rule.get("include_suffixes")
    if isinstance(include_suffixes, list) and include_suffixes:
        suffix = str(include_suffixes[0]).lstrip("*")
        return f"repo://Docs/**/*{suffix}"
    return f"repo://{rule.get('rule_key', 'memory-rule')}"


def _knowledge_manifest_source_from_rule(
    rule: dict[str, Any],
    *,
    source_version: str,
    required: bool,
) -> KnowledgeManifestSourceV1:
    return KnowledgeManifestSourceV1(
        key=str(rule.get("rule_key", "memory-rule")),
        title=str(rule.get("rule_key", "memory-rule")).replace("_", " "),
        uri=_rule_uri(rule),
        source_type="repo_rule",
        authority_level=str(rule.get("authority_level", "")),
        memory_usage=str(rule.get("memory_usage", "")),
        stage_affinity=[str(item) for item in rule.get("stage_affinity", []) if str(item).strip()],
        agent_affinity=[str(item) for item in rule.get("agent_affinity", []) if str(item).strip()],
        owner="repo_memory_manifest",
        source_version=source_version,
        required=required,
        summary=str(rule.get("notes", "")),
    )


def _deliverable_map(snapshot: SessionSnapshot) -> dict[str, str]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return {}
    return {
        item.key: item.content_markdown.strip()
        for item in blueprint.delivery_package.deliverables
        if item.key and item.content_markdown.strip()
    }


def _evaluation_case_key(case: Any, index: int) -> str:
    raw = getattr(case, "case_key", None) or getattr(case, "name", None) or getattr(case, "title", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return f"case_{index + 1}"


def _evaluation_cases(snapshot: SessionSnapshot) -> list[EvaluationCaseV1]:
    if snapshot.evaluation_dataset is not None and snapshot.evaluation_dataset.cases:
        return [
            EvaluationCaseV1(
                key=_evaluation_case_key(case, index),
                title=case.title,
                category=case.category,
                scenario=case.scenario,
                expected_result=case.expected_result,
            )
            for index, case in enumerate(snapshot.evaluation_dataset.cases)
        ]

    if snapshot.evaluation is not None and snapshot.evaluation.cases:
        return [
            EvaluationCaseV1(
                key=_evaluation_case_key(case, index),
                title=case.name,
                category=case.category,
                scenario=case.scenario,
                expected_result=case.expected_result,
            )
            for index, case in enumerate(snapshot.evaluation.cases)
        ]

    return []


def build_behavior_spec(snapshot: SessionSnapshot, *, generated_at=None) -> BehaviorSpecV1:
    generated_at = generated_at or utc_now()
    tool_contracts = build_tool_contracts(snapshot, generated_at=generated_at)
    memory_policy = build_memory_policy(snapshot, generated_at=generated_at)
    knowledge_contract = build_knowledge_contract(snapshot, generated_at=generated_at)
    stage4 = compile_stage4_artifacts(
        snapshot,
        generated_at=generated_at,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=_success_criteria(snapshot),
    )
    return stage4.behavior_spec


def build_tool_contracts(snapshot: SessionSnapshot, *, generated_at=None) -> list[ToolContractV1]:
    generated_at = generated_at or utc_now()
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []

    contracts: list[ToolContractV1] = []
    for tool in blueprint.tools:
        permissions = _normalized_items(tool.permissions) or (
            ["approved_human_review"] if tool.requires_approval else ["implicit_read_access"]
        )
        scopes = _normalized_items(tool.scopes) or (["write"] if tool.has_side_effects else ["read"])
        sensitive_data = _normalized_items(tool.sensitive_data)
        if tool.has_side_effects:
            sensitive_data.append("system_state")
        if _contains_keywords(tool.name, tool.purpose):
            sensitive_data.append("knowledge_reference")

        contracts.append(
            ToolContractV1(
                **_base_metadata(snapshot, generated_at),
                name=tool.name or "unnamed_tool",
                purpose=_coalesce(tool.purpose, fallback="Herramienta especializada del blueprint."),
                owner=_coalesce(tool.owner, fallback="owner_pending"),
                archetype=_coalesce(
                    tool.archetype,
                    fallback="side_effect" if tool.has_side_effects else "read_only",
                ),
                integration_kind=_coalesce(tool.integration_kind, fallback="integration_pending"),
                endpoint_reference=_coalesce(tool.endpoint_reference, fallback="endpoint://pending"),
                auth_reference=_coalesce(tool.auth_reference, fallback="auth://pending-reference"),
                risk_level=_coalesce(tool.risk_level, fallback="medium"),
                execution_mode=_coalesce(tool.execution_mode, fallback="sync"),
                requires_approval=tool.requires_approval,
                approval_reason=tool.approval_reason,
                approval_policy=_coalesce(
                    tool.approval_policy,
                    fallback="local_admin_mandatory" if tool.requires_approval else "not_required",
                ),
                side_effects=tool.has_side_effects,
                idempotent=_tool_is_idempotent(tool.idempotency_strategy, tool.has_side_effects),
                idempotency_strategy=_coalesce(
                    tool.idempotency_strategy,
                    fallback="Idempotencia pendiente de documentar.",
                ),
                input_schema=_tool_schema(tool.inputs),
                output_schema=_tool_schema(tool.outputs),
                validations=_normalized_items(tool.validations),
                typed_errors=_normalized_items(tool.typed_errors) or ["unknown_tool_error"],
                retry_strategy=_coalesce(tool.retry_strategy, fallback="Reintentar una vez con trazabilidad."),
                compensation_strategy=_coalesce(
                    tool.compensation_strategy,
                    fallback="No aplica o requiere remediation manual controlada.",
                ),
                failure_mode=_coalesce(tool.failure_mode, fallback="Falla operacional sin side effects confirmados."),
                permissions=permissions,
                scopes=scopes,
                sensitive_data=sensitive_data,
                audit_rules=_normalized_items(tool.audit_rules) or ["Registrar request_id y resultado auditado."],
                rate_limit_policy=_coalesce(
                    tool.rate_limit_policy,
                    fallback="Rate limit pendiente de documentar.",
                ),
                timeout_policy=_coalesce(
                    tool.timeout_policy,
                    fallback="Timeout pendiente de documentar.",
                ),
                contract_review_state=_coalesce(
                    tool.contract_review_state,
                    fallback="needs-review",
                ),
                provenance=_provenance(
                    (
                        "name",
                        ["blueprint.tools"],
                        "El nombre y la semantica provienen del tool editor actual del blueprint.",
                    ),
                    (
                        "input_schema",
                        ["blueprint.tools.inputs"],
                        "Los inputs se proyectan a un schema minimo sin inferir tipos avanzados.",
                    ),
                    (
                        "output_schema",
                        ["blueprint.tools.outputs"],
                        "Los outputs se mantienen como schema string-first mientras el builder no modele tipos avanzados.",
                    ),
                    (
                        "endpoint_reference",
                        ["blueprint.tools.endpoint_reference"],
                        "La referencia tecnica del endpoint sale del contrato guiado y nunca expone secretos.",
                    ),
                    (
                        "auth_reference",
                        ["blueprint.tools.auth_reference"],
                        "La autenticacion viaja como referencia abstracta y no como secreto exportado.",
                    ),
                ),
            )
        )
    return contracts


def build_memory_policy(snapshot: SessionSnapshot, *, generated_at=None) -> MemoryPolicyV1:
    generated_at = generated_at or utc_now()
    blueprint = snapshot.blueprint
    memory_profile = blueprint.memory_profile if blueprint is not None else None
    workflow_profile = blueprint.delivery_package.workflow_profile if blueprint is not None else None
    observability_plan = blueprint.delivery_package.observability_plan if blueprint is not None else None
    knowledge_profile = blueprint.knowledge_profile if blueprint is not None else None
    strategy = _coalesce(
        blueprint.memory_strategy if blueprint is not None else None,
        memory_profile.strategy if memory_profile is not None else None,
        fallback="session_memory",
    )
    storage_layers = _normalized_items(memory_profile.storage_layers if memory_profile is not None else [])
    if not storage_layers:
        storage_layers = ["session_state"]
    knowledge_enabled = (
        knowledge_profile.mode.strip().lower() == "rag"
        if knowledge_profile is not None and isinstance(knowledge_profile.mode, str)
        else bool(knowledge_profile.sources if knowledge_profile is not None else False)
    )

    return MemoryPolicyV1(
        **_base_metadata(snapshot, generated_at),
        strategy=strategy,
        storage_layers=storage_layers,
        context_budgets=_memory_context_budgets(),
        write_policy=_coalesce(
            memory_profile.write_policy if memory_profile is not None else None,
            fallback="Persistir al cerrar cada checkpoint aprobado.",
        ),
        retrieval_policy=_coalesce(
            memory_profile.retrieval_policy if memory_profile is not None else None,
            fallback="Recuperar contexto por prioridad, evidencia y cercania al objetivo.",
        ),
        retrieval_scopes=_memory_retrieval_scopes(snapshot, knowledge_enabled=knowledge_enabled),
        summary_policy=_coalesce(
            observability_plan.plan_summary_policy if observability_plan is not None else None,
            fallback="Resumir por checkpoint aprobado, consolidar decisiones vigentes y reemplazar contexto redundante por referencias a artefactos.",
        ),
        invalidation_policy=_coalesce(
            fallback=(
                "Invalidar summaries, cache de retrieval y planes compactados cuando cambie la version del blueprint, se rechace un approval gate, "
                "cambien tools o cambie la politica LLM/memoria."
            ),
        ),
        review_trigger=_coalesce(
            memory_profile.review_trigger if memory_profile is not None else None,
            fallback="Revisar cuando cambie el objetivo, el riesgo o la aprobacion requerida.",
        ),
        goal_drift_guard=_coalesce(
            memory_profile.goal_drift_guard if memory_profile is not None else None,
            fallback="Revalidar el objetivo antes de cada accion relevante.",
        ),
        retention_policy=_coalesce(
            memory_profile.retention_policy if memory_profile is not None else None,
            fallback="Mantener memoria operativa solo el tiempo necesario para el workflow y sus checkpoints.",
        ),
        ttl_policy=_coalesce(
            memory_profile.ttl_policy if memory_profile is not None else None,
            fallback="TTL corto para sesion y checkpoints; extender solo bajo necesidad aprobada.",
        ),
        workspace_scope=_coalesce(
            memory_profile.workspace_scope if memory_profile is not None else None,
            fallback="Usar memoria del workspace solo para continuidad del blueprint y evidencias aprobadas.",
        ),
        agent_scope=_coalesce(
            memory_profile.agent_scope if memory_profile is not None else None,
            fallback="Exportar al agente final solo resumentes, checkpoints y preferencias aprobadas.",
        ),
        grounding_policy={
            "citations_policy": _coalesce(
                memory_profile.grounding_policy.citations_policy if memory_profile is not None else None,
                fallback="La memoria debe citar el artefacto o checkpoint del que proviene.",
            ),
            "confidence_policy": _coalesce(
                memory_profile.grounding_policy.confidence_policy if memory_profile is not None else None,
                fallback="Usar memoria previa solo cuando mantenga trazabilidad y no contradiga evidencia vigente.",
            ),
            "no_evidence_behavior": _coalesce(
                memory_profile.grounding_policy.no_evidence_behavior if memory_profile is not None else None,
                fallback="Si la memoria no aporta evidencia confiable, seguir solo el estado explicito del workflow.",
            ),
            "contradictory_evidence_behavior": _coalesce(
                memory_profile.grounding_policy.contradictory_evidence_behavior if memory_profile is not None else None,
                fallback="Escalar a revision humana si memoria y evidencia activa se contradicen.",
            ),
        },
        sensitivity_rules=(
            _normalized_items(memory_profile.sensitivity_rules if memory_profile is not None else [])
            or [
                "No exportar secretos ni configuraciones privadas del runtime.",
                "Toda memoria persistente requiere owner y criterio de borrado en etapas posteriores.",
            ]
        ),
        checkpoints_required=(
            "checkpoint" in strategy.lower()
            or "persistent" in strategy.lower()
            or bool(_coalesce(workflow_profile.checkpoint_policy if workflow_profile is not None else None, fallback=""))
        ),
        provenance=_provenance(
            (
                "strategy",
                ["blueprint.memory_strategy", "blueprint.memory_profile.strategy"],
                "La estrategia primaria sale del blueprint actual y no de estado operativo interno.",
            ),
            (
                "storage_layers",
                ["blueprint.memory_profile.storage_layers"],
                "Las capas de memoria respetan el editor guiado disponible hoy.",
            ),
            (
                "context_budgets",
                ["blueprint.delivery_package.observability_plan", "Docs/system-analysis/28-plan-memoria-agentica-hibrida.md"],
                "Los budgets de contexto gobiernan la memoria operativa por rol para evitar reenviar contexto redundante.",
            ),
            (
                "retrieval_scopes",
                ["session.snapshot_projection", "Docs/system-analysis/30-memory-m0-agents-and-store-matrix.md"],
                "Los retrieval scopes apuntan solo a stores y namespaces ya gobernados por el sistema.",
            ),
        ),
    )


def build_knowledge_contract(snapshot: SessionSnapshot, *, generated_at=None) -> KnowledgeContractV1:
    generated_at = generated_at or utc_now()
    blueprint = snapshot.blueprint
    discovery = snapshot.discovery
    tool_entries = blueprint.tools if blueprint is not None else []
    knowledge_profile = blueprint.knowledge_profile if blueprint is not None else None
    inferred_enabled = any(
        _contains_keywords(tool.name, tool.purpose, " ".join(tool.inputs), " ".join(tool.outputs))
        for tool in tool_entries
    ) or _contains_keywords(
        discovery.problem_statement if discovery is not None else None,
        discovery.desired_outcome if discovery is not None else None,
        blueprint.narrative if blueprint is not None else None,
    )
    profile_mode = _coalesce(knowledge_profile.mode if knowledge_profile is not None else None, fallback="")
    explicit_mode = bool(profile_mode)
    knowledge_enabled = profile_mode == "rag" if explicit_mode else inferred_enabled
    mode = profile_mode or ("rag" if knowledge_enabled else "none")
    sources: list[KnowledgeSourceRef] = []
    if knowledge_profile is not None and knowledge_profile.sources:
        for source in knowledge_profile.sources:
            source_version = _coalesce(source.source_version, fallback="pending")
            sources.append(
                KnowledgeSourceRef(
                    key=source.key or source.title or "knowledge-source",
                    title=source.title or source.key or "Fuente sin titulo",
                    source_type=source.source_type or "document_repository",
                    uri=source.uri,
                    owner=source.owner,
                    sensitivity=source.sensitivity,
                    license=source.license,
                    description=source.description,
                    source_version=source_version,
                    lineage_key=f"{(source.key or source.title or 'knowledge-source').strip()}::{source_version}",
                )
            )
    elif knowledge_enabled and not explicit_mode:
        for tool in tool_entries:
            if _contains_keywords(tool.name, tool.purpose):
                sources.append(
                    KnowledgeSourceRef(
                        key=tool.name,
                        title=tool.name.replace("_", " ").title(),
                        source_type="tool_reference",
                        uri=f"tool://{tool.name}",
                        owner="knowledge_owner_pending",
                        sensitivity="internal",
                        license="pending_review",
                        description=tool.purpose,
                        source_version="pending",
                        lineage_key=f"{tool.name}::pending",
                    )
                )

    open_questions = []
    if knowledge_enabled and not sources:
        open_questions.append("Definir al menos una fuente aprobada y su owner antes de construir retrieval real.")
    if mode == "rag":
        if knowledge_profile is None or not _coalesce(knowledge_profile.ingestion_policy.chunking_policy, fallback=""):
            open_questions.append("Definir parser, chunking y metadata de ingestion antes de automatizar RAG.")
        if knowledge_profile is None or knowledge_profile.embedding_policy.dimensions <= 0:
            open_questions.append("Definir provider, modelo y dimensions de embeddings para retrieval real.")
        if knowledge_profile is None or knowledge_profile.retrieval_policy.top_k <= 0:
            open_questions.append("Definir top-k, filtros y fallback de retrieval antes de exportar el contrato final.")
        if knowledge_profile is None or not _coalesce(knowledge_profile.refresh_policy.frequency, fallback=""):
            open_questions.append("Definir refresh policy y estrategia de borrado para mantener freshness y lineage.")

    source_lineage = [item.lineage_key for item in sources if _coalesce(item.lineage_key, fallback="")]
    ingestion_policy = None
    embedding_policy = None
    retrieval_policy = None
    refresh_policy = None
    default_grounding_policy = {
        "citations_policy": "Responder solo con evidencia citada o declarar falta de soporte documental.",
        "confidence_policy": "Priorizar fuentes aprobadas y vigentes antes de responder.",
        "no_evidence_behavior": "Devolver needs-resolution cuando no exista evidencia suficiente.",
        "contradictory_evidence_behavior": "Escalar cuando dos fuentes aprobadas se contradigan.",
    }
    grounding_policy = dict(default_grounding_policy)
    if knowledge_profile is not None:
        grounding_policy = {
            "citations_policy": _coalesce(
                knowledge_profile.grounding_policy.citations_policy,
                fallback=default_grounding_policy["citations_policy"],
            ),
            "confidence_policy": _coalesce(
                knowledge_profile.grounding_policy.confidence_policy,
                fallback=default_grounding_policy["confidence_policy"],
            ),
            "no_evidence_behavior": _coalesce(
                knowledge_profile.grounding_policy.no_evidence_behavior,
                fallback=default_grounding_policy["no_evidence_behavior"],
            ),
            "contradictory_evidence_behavior": _coalesce(
                knowledge_profile.grounding_policy.contradictory_evidence_behavior,
                fallback=default_grounding_policy["contradictory_evidence_behavior"],
            ),
        }
    if knowledge_enabled and knowledge_profile is not None and mode == "rag":
        ingestion_policy = KnowledgeIngestionPolicyV1(
            parser=_coalesce(knowledge_profile.ingestion_policy.parser, fallback="pending_review"),
            chunking_policy=_coalesce(knowledge_profile.ingestion_policy.chunking_policy, fallback="pending_review"),
            metadata_fields=_normalized_items(knowledge_profile.ingestion_policy.metadata_fields),
            include_filters=_normalized_items(knowledge_profile.ingestion_policy.include_filters),
            exclude_filters=_normalized_items(knowledge_profile.ingestion_policy.exclude_filters),
        )
        embedding_policy = KnowledgeEmbeddingPolicyV1(
            provider=_coalesce(knowledge_profile.embedding_policy.provider, fallback="pending_review"),
            model=_coalesce(knowledge_profile.embedding_policy.model, fallback="pending_review"),
            dimensions=knowledge_profile.embedding_policy.dimensions,
            version=_coalesce(knowledge_profile.embedding_policy.version, fallback="pending"),
        )
        retrieval_policy = KnowledgeRetrievalPolicyV1(
            top_k=knowledge_profile.retrieval_policy.top_k,
            filters=_normalized_items(knowledge_profile.retrieval_policy.filters),
            search_mode=_coalesce(knowledge_profile.retrieval_policy.search_mode, fallback="hybrid"),
            reranking_policy=_coalesce(knowledge_profile.retrieval_policy.reranking_policy, fallback="pending_review"),
            fallback_behavior=_coalesce(
                knowledge_profile.retrieval_policy.fallback_behavior,
                fallback="Declarar falta de evidencia y escalar a remediation guiada.",
            ),
        )
        refresh_policy = KnowledgeRefreshPolicyV1(
            frequency=_coalesce(knowledge_profile.refresh_policy.frequency, fallback="pending_review"),
            triggers=_normalized_items(knowledge_profile.refresh_policy.triggers),
            expiration_policy=_coalesce(
                knowledge_profile.refresh_policy.expiration_policy,
                fallback="Definir expiracion y freshness antes de produccion.",
            ),
            deletion_policy=_coalesce(
                knowledge_profile.refresh_policy.deletion_policy,
                fallback="Definir borrado y retencion de referencias antes de produccion.",
            ),
        )
    sensitivity_rules = [
        "Nunca exportar documentos privados ni credenciales.",
        "Las referencias quedan abstractas hasta cerrar el contrato de knowledge real.",
    ]
    if knowledge_profile is not None and knowledge_profile.sensitivity_rules:
        sensitivity_rules = _normalized_items(knowledge_profile.sensitivity_rules)
    elif not knowledge_enabled:
        sensitivity_rules = ["Knowledge deshabilitado para este caso; no se empaquetan fuentes privadas."]

    return KnowledgeContractV1(
        **_base_metadata(snapshot, generated_at),
        enabled=knowledge_enabled,
        mode=mode,
        sources=sources,
        source_lineage=source_lineage,
        ingestion_policy=ingestion_policy,
        embedding_policy=embedding_policy,
        retrieval_policy=retrieval_policy,
        refresh_policy=refresh_policy,
        grounding_policy=grounding_policy,
        sensitivity_rules=sensitivity_rules,
        open_questions=open_questions,
        provenance=_provenance(
            (
                "enabled",
                ["blueprint.knowledge_profile", "blueprint.tools", "discovery.problem_statement", "blueprint.narrative"],
                "Cuando existe knowledge_profile explicito, ese contrato manda; si no, se permite inferencia conservadora.",
            ),
            (
                "sources",
                ["blueprint.knowledge_profile", "blueprint.tools"],
                "Las fuentes explicitas del blueprint tienen prioridad; solo se derivan desde tools cuando no existe contrato persistido.",
            ),
        ),
    )


def build_short_term_memory(snapshot: SessionSnapshot, *, generated_at=None) -> ShortTermMemoryV1:
    generated_at = generated_at or utc_now()
    blueprint = snapshot.blueprint
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    evaluation = snapshot.evaluation
    memory_policy = build_memory_policy(snapshot, generated_at=generated_at)
    knowledge_contract = build_knowledge_contract(snapshot, generated_at=generated_at)

    session_status = snapshot.session.status.value if hasattr(snapshot.session.status, "value") else str(snapshot.session.status)
    active_stage = _stage_affinity(snapshot)
    active_goal = _coalesce(
        canvas.user_goal if canvas is not None else None,
        discovery.desired_outcome if discovery is not None else None,
        blueprint.narrative if blueprint is not None else None,
        fallback="Consolidar el blueprint vigente con contexto minimo y trazable.",
    )
    current_focus = _coalesce(
        blueprint.delivery_package.decision_summary if blueprint is not None else None,
        canvas.success_metric if canvas is not None else None,
        discovery.problem_statement if discovery is not None else None,
        fallback="Mantener continuidad entre etapas y checkpoints aprobados.",
    )

    checkpoint_refs: list[ShortTermMemoryRefV1] = []
    if discovery is not None:
        checkpoint_refs.append(
            ShortTermMemoryRefV1(
                key="stage_checkpoint:discover",
                kind="stage_checkpoint",
                stage="discover",
                source="session.discovery",
                summary=_coalesce(discovery.desired_outcome, discovery.problem_statement, fallback="Discovery capturado."),
                status=session_status,
            )
        )
    latest_define_artifact = (snapshot.journey_latest_artifacts or {}).get("define")
    approved_definition = None
    if latest_define_artifact is not None:
        payload = latest_define_artifact.proposal_payload
        if latest_define_artifact.state in {"approved", "approved_legacy"} and (
            latest_define_artifact.schema_version == "definition-artifact.v1" or "functional_requirements" in payload
        ):
            approved_definition = RequirementsDefinitionOutput.model_validate(payload)
    if canvas is not None:
        checkpoint_refs.append(
            ShortTermMemoryRefV1(
                key="stage_checkpoint:define",
                kind="stage_checkpoint",
                stage="define",
                source="session.canvas" if approved_definition is None else "session.definition",
                summary=(
                    _coalesce(
                        approved_definition.summary,
                        f"{len(approved_definition.functional_requirements)} requisitos funcionales y "
                        f"{len(approved_definition.open_questions)} preguntas abiertas.",
                        fallback="Definition consolidada."
                    )
                    if approved_definition is not None
                    else _coalesce(canvas.user_goal, canvas.success_metric, fallback="Canvas consolidado.")
                ),
                status=session_status,
            )
        )
    if blueprint is not None:
        checkpoint_refs.append(
            ShortTermMemoryRefV1(
                key="stage_checkpoint:design",
                kind="stage_checkpoint",
                stage="design",
                source="session.blueprint",
                summary=_coalesce(
                    blueprint.delivery_package.decision_summary,
                    blueprint.narrative,
                    fallback="Blueprint vigente listo para orquestacion posterior.",
                ),
                status=str(blueprint.readiness_state.value if hasattr(blueprint.readiness_state, "value") else blueprint.readiness_state),
                blueprint_version_number=latest_blueprint_version(snapshot),
            )
        )
    if evaluation is not None:
        checkpoint_refs.append(
            ShortTermMemoryRefV1(
                key="stage_checkpoint:evaluate",
                kind="stage_checkpoint",
                stage="evaluate",
                source="session.evaluation",
                summary=_coalesce(
                    " ".join(evaluation.recommendations[:2]) if evaluation.recommendations else None,
                    " ".join(evaluation.gaps[:2]) if evaluation.gaps else None,
                    fallback="Evaluacion disponible para cerrar gaps de continuidad.",
                ),
                status=str(evaluation.completeness_status.value if hasattr(evaluation.completeness_status, "value") else evaluation.completeness_status),
            )
        )

    version_refs = [
        ShortTermMemoryRefV1(
            key=f"blueprint_version:{item.version_number}",
            kind="blueprint_version",
            stage="design",
            source=item.source_action or "blueprint_version",
            summary=(
                f"Blueprint v{item.version_number} con arquitectura {item.architecture or 'pending'} "
                f"y razonamiento {item.reasoning_pattern or 'pending'}."
            ),
            status=str(item.readiness_state.value if hasattr(item.readiness_state, "value") else item.readiness_state),
            created_at=_serialize_datetime(item.created_at),
            blueprint_version_number=item.version_number,
        )
        for item in sorted(snapshot.blueprint_versions, key=lambda entry: entry.version_number, reverse=True)[:3]
    ]
    checkpoint_refs.extend(version_refs)

    artifact_refs = [
        ShortTermMemoryRefV1(
            key=f"artifact:{item.artifact_key or item.id}",
            kind=item.artifact_kind or "artifact",
            stage=item.stage.value if hasattr(item.stage, "value") else str(item.stage),
            source=item.source_action or "artifact_registry",
            summary=_coalesce(
                item.artifact_title,
                str(item.artifact_metadata.get("summary", "")) if item.artifact_metadata else None,
                item.artifact_key,
                fallback="Artifacto persistido en el registro de sesion.",
            ),
            status=item.export_format or "stored",
            created_at=_serialize_datetime(item.created_at),
            blueprint_version_number=item.blueprint_version_number,
            evidence_paths=[item.artifact_key] if item.artifact_key else [],
        )
        for item in sorted(snapshot.artifact_records, key=lambda entry: entry.created_at, reverse=True)[:6]
    ]

    skill_run_refs = [
        ShortTermMemoryRefV1(
            key=f"skill_run:{item.id}",
            kind="skill_run",
            stage=item.stage.value if hasattr(item.stage, "value") else str(item.stage),
            source=item.source_action or item.skill_key or "skill_run",
            summary=_coalesce(item.result_summary, item.label, item.skill_key, fallback="Skill run persistido."),
            status=item.status.value if hasattr(item.status, "value") else str(item.status),
            created_at=_serialize_datetime(item.created_at),
            blueprint_version_number=item.blueprint_version_number,
            evidence_paths=[artifact.artifact_role for artifact in item.artifacts if artifact.artifact_role],
        )
        for item in sorted(snapshot.skill_runs, key=lambda entry: entry.created_at, reverse=True)[:6]
    ]

    branch_refs: list[ShortTermMemoryRefV1] = [
        ShortTermMemoryRefV1(
            key=f"handoff:{item.handoff_key or item.id}",
            kind="handoff",
            stage=(
                f"{item.from_stage.value if hasattr(item.from_stage, 'value') else item.from_stage}"
                f"->{item.to_stage.value if hasattr(item.to_stage, 'value') else item.to_stage}"
            ),
            source=item.triggered_by or "handoff",
            summary=_coalesce(item.summary, item.title, item.handoff_key, fallback="Handoff activo en sesion."),
            status=item.status,
            created_at=_serialize_datetime(item.created_at),
            blueprint_version_number=item.blueprint_version_number,
            evidence_paths=[item.handoff_key] if item.handoff_key else [],
        )
        for item in sorted(snapshot.handoff_records, key=lambda entry: entry.created_at, reverse=True)[:3]
    ]
    branch_refs.extend(
        [
            ShortTermMemoryRefV1(
                key=f"subagent_run:{item.id}",
                kind="subagent_run",
                stage="branch_execution",
                source=item.run_kind or "subagent",
                summary=_coalesce(item.summary, item.title, item.run_kind, fallback="Subagente ejecutado."),
                status=item.status.value if hasattr(item.status, "value") else str(item.status),
                created_at=_serialize_datetime(item.created_at),
                blueprint_version_number=item.blueprint_version_number,
                evidence_paths=[item.feature_flag_key] if item.feature_flag_key else [],
            )
            for item in sorted(snapshot.subagent_runs, key=lambda entry: entry.created_at, reverse=True)[:3]
        ]
    )

    pending_approvals = sorted(
        {
            item.gate_key
            for item in snapshot.approvals
            if item.status == ApprovalStatus.pending
        }
    )
    open_handoffs = sorted({item.handoff_key for item in snapshot.handoff_records if item.status == "pending" and item.handoff_key})

    recent_decisions = []
    if blueprint is not None:
        for item in blueprint.delivery_package.decision_trace[:5]:
            selected_value = item.selected_label or item.selected_value
            if item.dimension and selected_value:
                recent_decisions.append(f"{item.dimension}: {selected_value}")
        if not recent_decisions:
            recent_decisions = _dedupe_preserve_order(
                [
                    f"architecture: {blueprint.architecture}" if blueprint.architecture else "",
                    f"reasoning_pattern: {blueprint.reasoning_pattern}" if blueprint.reasoning_pattern else "",
                    f"knowledge_mode: {knowledge_contract.mode}" if knowledge_contract.mode else "",
                ]
            )

    checkpoint_keys = [item.key for item in checkpoint_refs]
    artifact_keys = [item.key for item in artifact_refs]
    skill_keys = [item.key for item in skill_run_refs]
    branch_keys = [item.key for item in branch_refs]

    namespaces: list[ShortTermMemoryNamespaceV1] = []
    if discovery is not None:
        namespaces.append(
            ShortTermMemoryNamespaceV1(
                namespace="session.short_term.discovery",
                summary="Contexto descubierto y aprobado por el usuario.",
                ref_keys=[key for key in checkpoint_keys if key == "stage_checkpoint:discover"],
                freshness="refresh_on_discovery_change",
                read_roles=["planner", "memory", "retrieval"],
                write_roles=["discovery_skill", "memory"],
            )
        )
    if canvas is not None:
        namespaces.append(
            ShortTermMemoryNamespaceV1(
                namespace="session.short_term.canvas",
                summary="Alcance y objetivo inmediato del agente en construccion.",
                ref_keys=[key for key in checkpoint_keys if key == "stage_checkpoint:define"],
                freshness="refresh_on_canvas_change",
                read_roles=["planner", "executor", "memory"],
                write_roles=["lean_scope_skill", "memory"],
            )
        )
    if blueprint is not None:
        namespaces.append(
            ShortTermMemoryNamespaceV1(
                namespace="session.short_term.plan",
                summary="Decision summary, blueprint actual y checkpoints de diseño.",
                ref_keys=[key for key in checkpoint_keys if key.startswith("stage_checkpoint:design") or key.startswith("blueprint_version:")],
                freshness="refresh_on_blueprint_promotion",
                read_roles=["planner", "executor", "evaluator", "memory"],
                write_roles=["blueprint_generation_skill", "memory"],
            )
        )
    if skill_keys:
        namespaces.append(
            ShortTermMemoryNamespaceV1(
                namespace="session.short_term.execution",
                summary="Ultimas ejecuciones y observaciones operativas de la sesion.",
                ref_keys=skill_keys,
                freshness="rolling_window_last_runs",
                read_roles=["executor", "recovery", "memory"],
                write_roles=["executor", "tool_use", "memory"],
            )
        )
    namespaces.append(
        ShortTermMemoryNamespaceV1(
            namespace="session.short_term.summary_cache",
            summary="Resumen compacto por checkpoint con referencias a artefactos y versiones.",
            ref_keys=_dedupe_preserve_order(checkpoint_keys[:4] + artifact_keys[:2]),
            freshness="compact_on_budget_threshold",
            read_roles=["planner", "executor", "memory", "retrieval"],
            write_roles=["memory"],
        )
    )
    if knowledge_contract.enabled or knowledge_contract.sources:
        namespaces.append(
            ShortTermMemoryNamespaceV1(
                namespace="session.short_term.retrieval",
                summary="Cache corto de findings citadas y ausencia de evidencia documentada.",
                ref_keys=artifact_keys[:2],
                freshness="expire_with_retrieval_ttl",
                read_roles=["retrieval", "executor", "memory"],
                write_roles=["retrieval", "memory"],
            )
        )
    if branch_keys:
        namespaces.append(
            ShortTermMemoryNamespaceV1(
                namespace="session.branch_board",
                summary="Board operativo de ramas, handoffs y ejecuciones paralelas del runtime.",
                ref_keys=branch_keys,
                freshness="refresh_on_branch_activity",
                read_roles=["planner", "executor", "recovery", "memory"],
                write_roles=["supervisor", "subagent", "memory"],
            )
        )
    if pending_approvals or open_handoffs or snapshot.validations:
        namespaces.append(
            ShortTermMemoryNamespaceV1(
                namespace="session.recovery.current",
                summary="Bloqueos activos, approvals pendientes y handoffs abiertos.",
                ref_keys=_dedupe_preserve_order(branch_keys + checkpoint_keys[-1:]),
                freshness="refresh_on_blocker_change",
                read_roles=["recovery", "supervisor", "memory"],
                write_roles=["recovery", "memory"],
            )
        )

    return ShortTermMemoryV1(
        **_base_metadata(snapshot, generated_at),
        active_stage=active_stage,
        active_goal=active_goal,
        current_focus=current_focus,
        pending_approvals=pending_approvals,
        open_handoffs=open_handoffs,
        recent_decisions=recent_decisions,
        namespaces=namespaces,
        checkpoint_refs=checkpoint_refs,
        artifact_refs=artifact_refs,
        skill_run_refs=skill_run_refs,
        branch_refs=branch_refs[:6],
        compaction=ShortTermMemoryCompactionV1(
            summary_policy=memory_policy.summary_policy,
            invalidation_policy=memory_policy.invalidation_policy,
            eviction_policy=_coalesce(
                memory_policy.ttl_policy,
                fallback="Evict oldest summaries first and keep only checkpoint references when budgets are exceeded.",
            ),
            last_compacted_at="",
        ),
        provenance=_provenance(
            (
                "checkpoint_refs",
                ["session.discovery", "session.canvas", "session.blueprint", "session.blueprint_versions", "session.evaluation"],
                "La memoria corta se ancla a checkpoints de etapa y versiones persistidas, no a prompts transitorios.",
            ),
            (
                "artifact_refs",
                ["session.artifact_registry", "session.skill_runs", "session.handoff_records"],
                "Los pointers activos se reducen a referencias trazables para evitar reenviar payloads completos.",
            ),
        ),
    )


def build_knowledge_manifest(snapshot: SessionSnapshot, *, generated_at=None) -> KnowledgeManifestV1:
    generated_at = generated_at or utc_now()
    knowledge_contract = build_knowledge_contract(snapshot, generated_at=generated_at)
    taxonomy_version, taxonomy_rules = _load_memory_taxonomy_rules()

    required_sources: list[KnowledgeManifestSourceV1] = []
    for source in knowledge_contract.sources:
        key = source.key or source.title or "knowledge-source"
        required_sources.append(
            KnowledgeManifestSourceV1(
                key=key,
                title=source.title or key,
                uri=source.uri or f"knowledge://{key}",
                source_type=source.source_type or "document_repository",
                authority_level="approved_blueprint_source",
                memory_usage="required_retrieval" if knowledge_contract.enabled else "candidate_retrieval",
                stage_affinity=["design", "tools", "memory", "runtime"],
                agent_affinity=["builder", "retrieval", "memory"],
                owner=source.owner or "knowledge_owner_pending",
                source_version=source.source_version or "pending",
                required=True,
                summary=source.description or "Fuente aprobada y versionada desde el blueprint actual.",
            )
        )

    candidate_sources: list[KnowledgeManifestSourceV1] = []
    for rule in taxonomy_rules:
        memory_usage = str(rule.get("memory_usage", ""))
        if memory_usage == "visual_only":
            continue
        manifest_source = _knowledge_manifest_source_from_rule(
            rule,
            source_version=taxonomy_version or "pending",
            required=memory_usage == "required_retrieval",
        )
        if manifest_source.required:
            required_sources.append(manifest_source)
        else:
            candidate_sources.append(manifest_source)

    seen_required: set[str] = set()
    deduped_required: list[KnowledgeManifestSourceV1] = []
    for item in required_sources:
        if item.key in seen_required:
            continue
        seen_required.add(item.key)
        deduped_required.append(item)

    active_stage = _stage_affinity(snapshot)
    candidate_sources = [
        item
        for item in candidate_sources
        if item.key not in seen_required
    ]
    candidate_sources.sort(
        key=lambda item: (
            active_stage not in item.stage_affinity,
            item.authority_level not in {"canonical", "operational"},
            item.key,
        )
    )

    knowledge_backend_mode = "hybrid_docs_session_and_rag" if knowledge_contract.enabled else "hybrid_docs_and_session"

    return KnowledgeManifestV1(
        **_base_metadata(snapshot, generated_at),
        knowledge_backend_mode=knowledge_backend_mode,
        operating_summary=(
            f"Manifest hibrido para la etapa {active_stage}: required primero, candidate por afinidad y retrieval solo por referencia trazable."
        ),
        retrieval_scopes=_memory_retrieval_scopes(snapshot, knowledge_enabled=knowledge_contract.enabled),
        required_sources=deduped_required,
        candidate_sources=candidate_sources,
        selection_policy=(
            "Leer primero required_sources; usar candidate_sources solo cuando falte evidencia o para ampliar contexto de la etapa activa "
            "sin reenviar artefactos completos."
        ),
        fallback_policy=(
            "Si required_sources no alcanza, escalar por authority_level y stage_affinity; si aun falta evidencia, devolver needs-resolution "
            "y registrar la ausencia en memoria corta."
        ),
        provenance=_provenance(
            (
                "required_sources",
                ["blueprint.knowledge_profile", "Docs/system-analysis/29-memory-m0-taxonomy-manifest.json"],
                "El manifiesto mezcla fuentes aprobadas del blueprint con la base canonica del repositorio.",
            ),
            (
                "candidate_sources",
                ["Docs/system-analysis/29-memory-m0-taxonomy-manifest.json"],
                "Las fuentes candidatas salen de la taxonomia M0 y conservan authority_level, afinidad y uso esperado.",
            ),
        ),
    )


def build_heuristic_decision(snapshot: SessionSnapshot, *, generated_at=None) -> HeuristicDecisionV1:
    generated_at = generated_at or utc_now()
    tool_contracts = build_tool_contracts(snapshot, generated_at=generated_at)
    memory_policy = build_memory_policy(snapshot, generated_at=generated_at)
    knowledge_contract = build_knowledge_contract(snapshot, generated_at=generated_at)
    stage4 = compile_stage4_artifacts(
        snapshot,
        generated_at=generated_at,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=_success_criteria(snapshot),
    )
    return stage4.heuristic_decision


def build_llm_policy(
    snapshot: SessionSnapshot,
    *,
    behavior_spec: BehaviorSpecV1,
    tool_contracts: list[ToolContractV1],
    memory_policy: MemoryPolicyV1,
    knowledge_contract: KnowledgeContractV1,
    generated_at=None,
) -> LLMPolicyV1:
    generated_at = generated_at or utc_now()
    stage4 = compile_stage4_artifacts(
        snapshot,
        generated_at=generated_at,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=_success_criteria(snapshot),
    )
    return stage4.llm_policy


def build_evaluation_pack(snapshot: SessionSnapshot, *, generated_at=None) -> EvaluationPackV1:
    generated_at = generated_at or utc_now()
    evaluation = snapshot.evaluation
    cases = _evaluation_cases(snapshot)
    dataset_version = snapshot.evaluation_dataset.version_number if snapshot.evaluation_dataset is not None else None
    rubric_version = snapshot.evaluation_rubric.version_number if snapshot.evaluation_rubric is not None else None
    latest_run_status = snapshot.evaluation_runs[0].status if snapshot.evaluation_runs else None

    blocking_issues = []
    recommendations = []
    scores: dict[str, int] = {}
    readiness_state = ReviewState.partial
    if evaluation is not None:
        blocking_issues = _normalized_items(evaluation.gaps)
        recommendations = _normalized_items(evaluation.recommendations)
        scores = {key: int(value) for key, value in evaluation.scores.items()}
        readiness_state = evaluation.completeness_status if evaluation.gaps else evaluation.coherence_status

    return EvaluationPackV1(
        **_base_metadata(snapshot, generated_at),
        readiness_state=readiness_state,
        scores=scores,
        blocking_issues=blocking_issues,
        recommendations=recommendations,
        cases=cases,
        dataset_version_number=dataset_version,
        rubric_version_number=rubric_version,
        latest_run_status=latest_run_status,
        acceptance_cases=cases,
        provenance=_provenance(
            (
                "cases",
                ["evaluation_dataset.cases", "evaluation.cases"],
                "Los acceptance cases salen del dataset canonico si existe; si no, caen al resumen de evaluacion.",
            ),
            (
                "scores",
                ["evaluation.scores", "evaluation_runs"],
                "Los scores mantienen la lectura actual del workbench de evaluacion.",
            ),
        ),
    )


def _prompt_output_schema(role: str) -> dict[str, Any]:
    if role == "planner":
        return {
            "type": "object",
            "properties": {
                "plan": {"type": "array", "items": {"type": "string"}},
                "risks": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["plan"],
        }
    if role == "executor":
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "artifacts": {"type": "array", "items": {"type": "string"}},
                "needs_approval": {"type": "boolean"},
            },
            "required": ["action", "needs_approval"],
        }
    if role == "evaluator":
        return {
            "type": "object",
            "properties": {
                "readiness": {"type": "string"},
                "blocking_issues": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["readiness"],
        }
    return {
        "type": "object",
        "properties": {
            "response": {"type": "string"},
        },
        "required": ["response"],
    }


def _prompt_variables(tool_contracts: list[ToolContractV1], success_criteria: list[SuccessCriterion]) -> list[PromptVariable]:
    variables = [
        PromptVariable(
            name="goal",
            description="Objetivo principal del blueprint en la ejecucion actual.",
            source_paths=["discovery.desired_outcome", "canvas.user_goal"],
        ),
        PromptVariable(
            name="constraints",
            description="Restricciones obligatorias del proyecto.",
            source_paths=["discovery.constraints", "blueprint.guardrails"],
        ),
    ]
    if tool_contracts:
        variables.append(
            PromptVariable(
                name="approved_tools",
                description="Herramientas disponibles para la sesion actual.",
                source_paths=["blueprint.tools"],
            )
        )
    if success_criteria:
        variables.append(
            PromptVariable(
                name="success_criteria",
                description="Criterios medibles de exito.",
                source_paths=["canvas.success_metric", "evaluation.scores"],
            )
        )
    return variables


def _prompt_stop_conditions(snapshot: SessionSnapshot) -> list[str]:
    conditions = [
        "Detenerse si falta evidencia estructural para responder o decidir.",
        "Detenerse si una accion contradice constraints, approvals o guardrails aprobados.",
    ]
    if any(item.status == "pending" for item in snapshot.approvals):
        conditions.append("Detenerse cuando una accion con side effects requiera approval pendiente.")
    return conditions


def _stable_hash_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable_hash_payload(item)
            for key, item in value.items()
            if key not in {"source_session_id", "generated_at"}
        }
    if isinstance(value, list):
        return [_stable_hash_payload(item) for item in value]
    return value


def _render_system_prompt(snapshot: SessionSnapshot, success_criteria: list[SuccessCriterion]) -> str:
    deliverables = _deliverable_map(snapshot)
    if deliverables.get("system_prompt"):
        return deliverables["system_prompt"]

    criteria = "\n".join(f"- {item.description}" for item in success_criteria) or "- Mantener trazabilidad del blueprint."
    return "\n".join(
        [
            "Actua como constructor Lean de blueprints agenticos.",
            "Trabaja solo con contratos versionados, evidencia aprobada y restricciones trazables.",
            "No inventes tools, memoria, prompts ni politicas fuera del contrato.",
            "Criterios de exito:",
            criteria,
        ]
    )


def build_prompt_pack(
    snapshot: SessionSnapshot,
    *,
    behavior_spec: BehaviorSpecV1,
    heuristic_decision: HeuristicDecisionV1,
    llm_policy: LLMPolicyV1,
    memory_policy: MemoryPolicyV1,
    knowledge_contract: KnowledgeContractV1,
    tool_contracts: list[ToolContractV1],
    success_criteria: list[SuccessCriterion],
    generated_at=None,
) -> PromptPackV1:
    generated_at = generated_at or utc_now()
    stage4 = compile_stage4_artifacts(
        snapshot,
        generated_at=generated_at,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=success_criteria,
    )
    return stage4.prompt_pack


def _success_criteria(snapshot: SessionSnapshot) -> list[SuccessCriterion]:
    criteria: list[SuccessCriterion] = []
    if snapshot.canvas is not None and snapshot.canvas.success_metric.strip():
        criteria.append(
            SuccessCriterion(
                key="success_metric",
                description=snapshot.canvas.success_metric.strip(),
                source="canvas.success_metric",
            )
        )
    if snapshot.discovery is not None and snapshot.discovery.mvp_definition.north_star_metric.strip():
        criteria.append(
            SuccessCriterion(
                key="north_star_metric",
                description=snapshot.discovery.mvp_definition.north_star_metric.strip(),
                source="discovery.mvp_definition.north_star_metric",
            )
        )
    if snapshot.canvas is not None:
        for index, metric in enumerate(_normalized_items(snapshot.canvas.agent_profile.success_metrics), start=1):
            criteria.append(
                SuccessCriterion(
                    key=f"agent_profile_metric_{index}",
                    description=metric,
                    source="canvas.agent_profile.success_metrics",
                )
            )
    return criteria or [
        SuccessCriterion(
            key="readiness",
            description="Mantener un blueprint coherente, trazable y listo para handoff controlado.",
            source="derived",
        )
    ]


def _completion_criteria(snapshot: SessionSnapshot) -> list[str]:
    criteria = [
        "El blueprint conserva arquitectura, tools, memoria y guardrails en un contrato versionado.",
        "No quedan approvals pendientes para cerrar el handoff principal.",
    ]
    if snapshot.evaluation is not None and snapshot.evaluation.gaps:
        criteria.append("Resolver o aceptar con evidencia todos los gaps de evaluacion.")
    return criteria


def _dependencies(snapshot: SessionSnapshot, knowledge_contract: KnowledgeContractV1) -> list[CanonicalDependency]:
    dependencies = [
        CanonicalDependency(
            key="session_snapshot",
            kind="internal_projection",
            description="La sesion sigue siendo source of truth para regenerar contratos canonicos.",
        ),
        CanonicalDependency(
            key="human_review",
            kind="approval_gate",
            description="El journey requiere revision humana antes de cualquier promotion con side effects.",
        ),
    ]
    if knowledge_contract.enabled:
        dependencies.append(
            CanonicalDependency(
                key="knowledge_owner",
                kind="knowledge_governance",
                description="Las fuentes de knowledge necesitan owner explicito antes de build real.",
            )
        )
    return dependencies


def _open_questions(snapshot: SessionSnapshot, knowledge_contract: KnowledgeContractV1) -> list[CanonicalOpenQuestion]:
    questions = [
        CanonicalOpenQuestion(
            key=f"approval_{approval.id}",
            question=approval.title,
            owner="human_reviewer",
        )
        for approval in snapshot.approvals
        if approval.status == "pending"
    ]
    for index, item in enumerate(knowledge_contract.open_questions, start=1):
        questions.append(
            CanonicalOpenQuestion(
                key=f"knowledge_{index}",
                question=item,
                owner="knowledge_owner",
            )
        )
    if snapshot.evaluation is not None:
        for index, gap in enumerate(_normalized_items(snapshot.evaluation.gaps), start=1):
            questions.append(
                CanonicalOpenQuestion(
                    key=f"evaluation_gap_{index}",
                    question=gap,
                    owner="project_owner",
                )
            )
    return questions


def _approvals(snapshot: SessionSnapshot) -> list[ApprovalGateSummary]:
    return [
        ApprovalGateSummary(
            gate_key=item.gate_key,
            title=item.title,
            status=item.status,
            requested_in_stage=str(item.requested_in_stage),
            rationale=item.rationale,
        )
        for item in snapshot.approvals
    ]


def _risks(snapshot: SessionSnapshot, tool_contracts: list[ToolContractV1]) -> list[RiskEntry]:
    risks: list[RiskEntry] = []
    if snapshot.blueprint is not None:
        for check in snapshot.blueprint.safety_checks:
            risks.append(
                RiskEntry(
                    category=check.category or "safety",
                    severity=check.severity or "medium",
                    summary=check.risk or "Riesgo operativo sin descripcion adicional.",
                    mitigation=check.mitigation,
                    status=check.status,
                )
            )
    if snapshot.canvas is not None and snapshot.canvas.primary_risk.strip():
        risks.append(
            RiskEntry(
                category="product",
                severity="medium",
                summary=snapshot.canvas.primary_risk.strip(),
                mitigation="Revisar el canvas y la evaluacion antes de empaquetar.",
            )
        )
    if any(tool.side_effects for tool in tool_contracts):
        risks.append(
            RiskEntry(
                category="tools",
                severity="high",
                summary="Existen tools con side effects que requieren approvals y rollback claros.",
                mitigation="No marcar readiness completa sin aprobar sus contratos.",
            )
        )
    return risks


def build_blueprint_core(snapshot: SessionSnapshot, *, generated_at=None) -> BlueprintCoreV1:
    generated_at = generated_at or utc_now()
    tool_contracts = build_tool_contracts(snapshot, generated_at=generated_at)
    memory_policy = build_memory_policy(snapshot, generated_at=generated_at)
    knowledge_contract = build_knowledge_contract(snapshot, generated_at=generated_at)
    consistency_report = ensure_blueprint_consistency_report(snapshot)
    stage4 = compile_stage4_artifacts(
        snapshot,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=_success_criteria(snapshot),
        generated_at=generated_at,
    )
    behavior_spec = stage4.behavior_spec
    heuristic_decision = stage4.heuristic_decision
    llm_policy = stage4.llm_policy

    discovery = snapshot.discovery
    canvas = snapshot.canvas
    blueprint = snapshot.blueprint
    success_criteria = _success_criteria(snapshot)
    assumptions = _normalized_items(snapshot.estimation_report.assumptions if snapshot.estimation_report is not None else [])

    return BlueprintCoreV1(
        **_base_metadata(snapshot, generated_at),
        identity=BlueprintIdentity(
            title=snapshot.session.title,
            case_type=_coalesce(discovery.case_type if discovery is not None else None, fallback="unknown"),
            current_stage=str(snapshot.session.current_stage),
            blueprint_version_number=latest_blueprint_version(snapshot),
        ),
        purpose=BlueprintPurpose(
            problem_statement=_coalesce(discovery.problem_statement if discovery is not None else None, fallback="Problema no documentado."),
            primary_user=_coalesce(
                canvas.agent_profile.primary_user if canvas is not None else None,
                discovery.current_user if discovery is not None else None,
                fallback="Usuario primario no documentado.",
            ),
            desired_outcome=_coalesce(discovery.desired_outcome if discovery is not None else None, fallback="Resultado esperado no documentado."),
            value_statement=_coalesce(discovery.value_statement if discovery is not None else None, fallback="Valor esperado por confirmar."),
        ),
        scope=BlueprintScope(
            in_scope=_normalized_items(canvas.mvp_scope if canvas is not None else []),
            out_of_scope=_normalized_items(canvas.out_of_scope if canvas is not None else []),
            constraints=_normalized_items(discovery.constraints if discovery is not None else []),
            non_delegable_decisions=_normalized_items(
                discovery.mvp_definition.non_delegable_decisions if discovery is not None else []
            ),
        ),
        behavior_spec=behavior_spec,
        heuristic_decision=heuristic_decision,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        llm_policy=llm_policy,
        guardrails=_normalized_items(blueprint.guardrails if blueprint is not None else []),
        approvals=_approvals(snapshot),
        risks=_risks(snapshot, tool_contracts),
        success_criteria=success_criteria,
        completion_criteria=_completion_criteria(snapshot),
        dependencies=_dependencies(snapshot, knowledge_contract),
        assumptions=assumptions,
        open_questions=_open_questions(snapshot, knowledge_contract),
        provenance=_provenance(
            (
                "purpose",
                ["discovery", "canvas.agent_profile"],
                "El nucleo de negocio sigue saliendo de discovery y canvas, no del workspace operativo.",
            ),
            (
                "behavior_spec",
                ["blueprint.delivery_package.workflow_profile", "blueprint.reasoning_pattern"],
                "El comportamiento exportable se separa del SessionSnapshot sin perder trazabilidad.",
            ),
            (
                "llm_policy",
                ["estimation_report", "blueprint.tools", "blueprint.guardrails"],
                "La politica LLM inicial se deriva de costo, tools aprobadas y restricciones persistidas.",
            ),
            (
                "approvals",
                [
                    f"journey.{item.stage_key}.v{item.version_number}"
                    for item in consistency_report.approved_stage_lineage
                    if item.version_number is not None
                ],
                "El export canonico se ancla a la cadena aprobada de etapas y no a decisiones legacy implícitas.",
            ),
        ),
    )


def _readiness_gap_entries(snapshot: SessionSnapshot, knowledge_contract: KnowledgeContractV1) -> list[ReadinessGapEntry]:
    gaps = []
    if snapshot.evaluation is not None:
        for gap in _normalized_items(snapshot.evaluation.gaps):
            gaps.append(
                ReadinessGapEntry(
                    code="evaluation_gap",
                    severity="blocking",
                    summary=gap,
                    remediation="Resolver el gap o documentar aceptacion explicita con evidencia.",
                )
            )
    for approval in snapshot.approvals:
        if approval.status == "pending":
            gaps.append(
                ReadinessGapEntry(
                    code="approval_pending",
                    severity="warning",
                    summary=approval.title,
                    remediation="Resolver la aprobacion antes de empaquetar para construccion.",
                )
            )
    for question in knowledge_contract.open_questions:
        gaps.append(
            ReadinessGapEntry(
                code="knowledge_question",
                severity="warning",
                summary=question,
                remediation="Definir owner y source canonica para knowledge antes del build real.",
            )
        )
    consistency_report = ensure_blueprint_consistency_report(snapshot)
    for item in consistency_report.blocking_issues:
        gaps.append(
            ReadinessGapEntry(
                code="consistency_gap",
                severity="blocking",
                summary=item,
                remediation="Alinear el blueprint vivo con la ultima cadena aprobada antes de exportar o construir.",
            )
        )
    for item in consistency_report.warnings:
        gaps.append(
            ReadinessGapEntry(
                code="consistency_warning",
                severity="warning",
                summary=item,
                remediation="Revisar cobertura, lineage o vigencia antes de distribuir el package como referencia final.",
            )
        )
    return gaps


def _construction_components(snapshot: SessionSnapshot, tool_contracts: list[ToolContractV1], knowledge_contract: KnowledgeContractV1) -> list[ConstructionComponent]:
    blueprint = snapshot.blueprint
    evaluation_state = ReviewState.partial
    if snapshot.evaluation is not None and not snapshot.evaluation.gaps:
        evaluation_state = ReviewState.complete
    tool_state = ReviewState.complete if tool_contracts else ReviewState.partial
    if any(tool.requires_approval and not tool.approval_reason for tool in tool_contracts):
        tool_state = ReviewState.blocked
    knowledge_state = ReviewState.complete if not knowledge_contract.open_questions else ReviewState.partial
    return [
        ConstructionComponent(
            key="architecture",
            label="Architecture",
            role="topology",
            status=blueprint.readiness_state if blueprint is not None else ReviewState.partial,
            summary=_coalesce(blueprint.architecture if blueprint is not None else None, fallback="Arquitectura pendiente."),
        ),
        ConstructionComponent(
            key="tools",
            label="Tools",
            role="integration",
            status=tool_state,
            summary=f"{len(tool_contracts)} tool contracts versionados para build.",
        ),
        ConstructionComponent(
            key="memory",
            label="Memory",
            role="state",
            status=ReviewState.complete if blueprint is not None and blueprint.memory_strategy else ReviewState.partial,
            summary=_coalesce(blueprint.memory_strategy if blueprint is not None else None, fallback="Estrategia de memoria pendiente."),
        ),
        ConstructionComponent(
            key="knowledge",
            label="Knowledge",
            role="retrieval",
            status=knowledge_state if knowledge_contract.enabled else ReviewState.complete,
            summary="Knowledge habilitado." if knowledge_contract.enabled else "Caso sin knowledge o RAG dedicado.",
        ),
        ConstructionComponent(
            key="evaluation",
            label="Evaluation",
            role="quality_gate",
            status=evaluation_state,
            summary="La evaluacion actual alimenta readiness y acceptance cases.",
        ),
    ]


def _file_manifest(
    snapshot: SessionSnapshot,
    prompt_pack: PromptPackV1,
    tool_contracts: list[ToolContractV1],
    evaluation_pack: EvaluationPackV1,
) -> list[ConstructionFileManifestEntry]:
    files = [
        ConstructionFileManifestEntry(
            path="contracts/blueprint-core.v1.json",
            kind="json",
            summary="Contrato canonico principal para una plataforma constructora.",
            source_contract="blueprint-core.v1",
            generated_from=["discovery", "canvas", "blueprint"],
        ),
        ConstructionFileManifestEntry(
            path="contracts/construction-pack.v1.json",
            kind="json",
            summary="Paquete de construccion con prompts, policies y readiness.",
            source_contract="construction-pack.v1",
            generated_from=["blueprint", "evaluation", "estimation"],
        ),
        ConstructionFileManifestEntry(
            path="contracts/prompt-pack.v1.json",
            kind="json",
            summary="Prompt pack trazable y versionado por rol.",
            source_contract="prompt-pack.v1",
            generated_from=["behavior-spec.v1", "llm-policy.v1", "heuristic-decision.v1"],
        ),
        ConstructionFileManifestEntry(
            path="contracts/evaluation-pack.v1.json",
            kind="json",
            summary="Acceptance cases y readouts de evaluacion.",
            source_contract="evaluation-pack.v1",
            generated_from=["evaluation", "evaluation_dataset", "evaluation_rubric"],
        ),
        ConstructionFileManifestEntry(
            path="contracts/test-pack.v1.json",
            kind="json",
            summary="Pack ejecutable de validacion, mutacion, prompts y consumidor externo.",
            source_contract="test-pack.v1",
            generated_from=["construction-pack.v1", "prompt-pack.v1", "evaluation-pack.v1"],
        ),
    ]
    for tool in tool_contracts:
        files.append(
            ConstructionFileManifestEntry(
                path=f"tools/{tool.name}.contract.json",
                kind="json",
                summary=tool.purpose,
                source_contract=tool.schema_version,
                generated_from=["blueprint.tools"],
            )
        )
    files.extend(
        [
            ConstructionFileManifestEntry(
                path="prompts/system.md",
                kind="markdown",
                summary=prompt_pack.system_prompt.title,
                source_contract="prompt-pack.v1",
                generated_from=["blueprint.delivery_package.deliverables", "blueprint.guardrails"],
            ),
            ConstructionFileManifestEntry(
                path="prompts/planner.md",
                kind="markdown",
                summary=prompt_pack.planner_prompt.title,
                source_contract="prompt-pack.v1",
                generated_from=["heuristic-decision.v1", "behavior-spec.v1"],
            ),
            ConstructionFileManifestEntry(
                path="prompts/executor.md",
                kind="markdown",
                summary=prompt_pack.executor_prompt.title,
                source_contract="prompt-pack.v1",
                generated_from=["behavior-spec.v1", "memory-policy.v1"],
            ),
            ConstructionFileManifestEntry(
                path="prompts/evaluator.md",
                kind="markdown",
                summary=prompt_pack.evaluator_prompt.title,
                source_contract="prompt-pack.v1",
                generated_from=["evaluation-pack.v1", "behavior-spec.v1"],
            ),
            ConstructionFileManifestEntry(
                path="tests/acceptance-cases.json",
                kind="json",
                summary=f"{len(evaluation_pack.acceptance_cases)} acceptance cases derivados de evaluacion.",
                source_contract=evaluation_pack.schema_version,
                generated_from=["evaluation_dataset", "evaluation"],
            ),
            ConstructionFileManifestEntry(
                path="tests/mutation-cases.json",
                kind="json",
                summary="Casos de mutacion sobre campos requeridos de contratos canonicos.",
                source_contract="test-pack.v1",
                generated_from=["shared_specs/schemas", "contracts/*.json"],
            ),
            ConstructionFileManifestEntry(
                path="tests/prompt-evaluation.json",
                kind="json",
                summary="Casos positivos y negativos para prompts criticos.",
                source_contract="test-pack.v1",
                generated_from=["prompt-pack.v1", "evaluation-pack.v1"],
            ),
            ConstructionFileManifestEntry(
                path="consumers/python/reference_consumer.py",
                kind="python",
                summary="Consumidor externo de referencia que valida el pack sin importar el builder.",
                source_contract="test-pack.v1",
                generated_from=["test-pack.v1"],
            ),
        ]
    )
    for artifact in prompt_pack.agent_role_prompts:
        files.append(
            ConstructionFileManifestEntry(
                path=f"prompts/agents/{artifact.prompt_key}.md",
                kind="markdown",
                summary=artifact.title,
                source_contract="prompt-pack.v1",
                generated_from=["behavior-spec.v1", "llm-policy.v1"],
            )
        )
    for artifact in prompt_pack.handoff_prompts:
        files.append(
            ConstructionFileManifestEntry(
                path=f"prompts/handoffs/{artifact.prompt_key}.md",
                kind="markdown",
                summary=artifact.title,
                source_contract="prompt-pack.v1",
                generated_from=["behavior-spec.v1", "evaluation-pack.v1"],
            )
        )
    return files


def build_construction_pack(snapshot: SessionSnapshot, *, generated_at=None) -> ConstructionPackV1:
    generated_at = generated_at or utc_now()
    tool_contracts = build_tool_contracts(snapshot, generated_at=generated_at)
    memory_policy = build_memory_policy(snapshot, generated_at=generated_at)
    knowledge_contract = build_knowledge_contract(snapshot, generated_at=generated_at)
    consistency_report = ensure_blueprint_consistency_report(snapshot)
    success_criteria = _success_criteria(snapshot)
    stage4 = compile_stage4_artifacts(
        snapshot,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=success_criteria,
        generated_at=generated_at,
    )
    behavior_spec = stage4.behavior_spec
    heuristic_decision = stage4.heuristic_decision
    llm_policy = stage4.llm_policy
    prompt_pack = stage4.prompt_pack
    evaluation_pack = build_evaluation_pack(snapshot, generated_at=generated_at)
    gaps = _readiness_gap_entries(snapshot, knowledge_contract)
    blocking_issues = [gap.summary for gap in gaps if gap.severity == "blocking"]
    warnings = [gap.summary for gap in gaps if gap.severity != "blocking"]
    remediation_notes = [gap.remediation for gap in gaps if gap.remediation]
    readiness_status = ReviewState.complete if not blocking_issues else ReviewState.blocked

    blueprint = snapshot.blueprint
    multi_agent_topology = behavior_spec.multi_agent_topology
    topology = {
        "architecture": blueprint.architecture if blueprint is not None else "",
        "reasoning_pattern": blueprint.reasoning_pattern if blueprint is not None else "",
        "workflow_template": snapshot.selected_workflow_template_key or "",
        "components": [component.model_dump(mode="json") for component in _construction_components(snapshot, tool_contracts, knowledge_contract)],
    }
    if multi_agent_topology is not None:
        topology.update(
            {
                "runtime_pattern": multi_agent_topology.runtime_pattern,
                "support_state": multi_agent_topology.support_state,
                "agent_count": len(multi_agent_topology.agent_contracts),
                "handoff_count": len(multi_agent_topology.handoff_contracts),
                "shared_state_contracts": len(multi_agent_topology.shared_state_contracts),
            }
        )
    topology["approved_stage_lineage"] = [
        item.model_dump(mode="json") for item in consistency_report.approved_stage_lineage
    ]
    topology["consistency_summary"] = {
        "overall_status": consistency_report.overall_status,
        "blocking_issues": consistency_report.blocking_issues,
        "warnings": consistency_report.warnings,
        "exportable_lineage": consistency_report.exportable_lineage,
        "restricted_lineage": consistency_report.restricted_lineage,
    }

    return ConstructionPackV1(
        **_base_metadata(snapshot, generated_at),
        blueprint_ref=ContractReference(
            contract_kind="blueprint-core",
            schema_version="blueprint-core.v1",
            source_blueprint_version=latest_blueprint_version(snapshot),
        ),
        components=_construction_components(snapshot, tool_contracts, knowledge_contract),
        topology=topology,
        multi_agent_benchmark=multi_agent_topology.benchmark if multi_agent_topology is not None else None,
        behavior_spec=behavior_spec,
        heuristic_decision=heuristic_decision,
        prompt_pack=prompt_pack,
        llm_policy=llm_policy,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        evaluation_pack=evaluation_pack,
        acceptance_cases=evaluation_pack.acceptance_cases,
        file_manifest=_file_manifest(snapshot, prompt_pack, tool_contracts, evaluation_pack),
        readiness=ConstructionReadinessV1(
            status=readiness_status,
            can_build=not blocking_issues,
            blocking_issues=blocking_issues,
            warnings=warnings,
            remediation_notes=remediation_notes,
        ),
        gaps=gaps,
        remediation_notes=remediation_notes,
        provenance=_provenance(
            (
                "prompt_pack",
                ["behavior-spec.v1", "heuristic-decision.v1", "llm-policy.v1"],
                "El construction pack materializa la separacion entre comportamiento, politica y prompts.",
            ),
            (
                "readiness",
                ["evaluation", "approvals"],
                "La readiness actual se deriva de gaps persistidos y approvals pendientes.",
            ),
        ),
    )


def _canonical_payload_checksum(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _contract_entry(contract_key: str, relative_path: str, payload: Any, *, required: bool = True) -> AcpV2ManifestContractEntry:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return AcpV2ManifestContractEntry(
        contract_key=contract_key,
        schema_version=contract_key,
        relative_path=relative_path,
        checksum_sha256=_canonical_payload_checksum(payload),
        required=required,
    )


def _portable_sequence_key(prefix: str, index: int) -> str:
    return f"{prefix}_{index:02d}"


def _acp_v2_build_plan() -> AcpV2BuildPlan:
    return AcpV2BuildPlan(
        entrypoint="README.md#agent-construction-package-v2",
        completion_criteria=[
            "Todas las decisiones de implementacion marcadas como required estan resueltas o aceptadas por el owner.",
            "Los contratos de herramientas, memoria, prompts y pruebas validan contra el schema agent-construction-package.v2.",
            "El runtime objetivo puede interpretar la especificacion sin depender de servicios internos de Lean Agent Builder.",
            "La suite de pruebas declarada se ejecuta y documenta resultados antes de promover a construccion.",
        ],
        steps=[
            AcpV2BuildStep(
                step_key="review-blueprint",
                title="Revisar diseno aprobado",
                objective="Confirmar objetivo, alcance, arquitectura, comportamiento y restricciones del Blueprint.",
                depends_on=[],
                inputs=["blueprint-core.v1", "construction-pack.v1"],
                outputs=["implementation-decisions.initial"],
                actions=[
                    "Leer la especificacion del sistema agentico.",
                    "Confirmar que los supuestos y preguntas abiertas estan clasificados como diseno o implementacion.",
                ],
                validation=["No existen gaps de diseno bloqueantes sin decision documentada."],
            ),
            AcpV2BuildStep(
                step_key="resolve-implementation-decisions",
                title="Cerrar decisiones de implementacion",
                objective="Solicitar solo las respuestas que dependen del entorno real de construccion.",
                depends_on=["review-blueprint"],
                inputs=["implementation_decisions"],
                outputs=["implementation-decisions.resolved"],
                actions=[
                    "Presentar cada pregunta con contexto, impacto, opciones y default recomendado.",
                    "Registrar la respuesta antes de generar configuraciones concretas.",
                ],
                validation=["Cada decision required tiene respuesta, waiver o responsable asignado."],
            ),
            AcpV2BuildStep(
                step_key="select-runtime",
                title="Seleccionar runtime agentico",
                objective="Mapear la especificacion declarativa al framework o herramienta agentica elegida.",
                depends_on=["resolve-implementation-decisions"],
                inputs=["agent_runtime", "compatibility"],
                outputs=["runtime.adapter-plan"],
                actions=[
                    "Elegir framework o IDE agentico compatible.",
                    "Mapear agentes, handoffs, estado compartido y presupuestos de ejecucion al runtime elegido.",
                ],
                validation=["No se requiere ningun endpoint interno de Lean Agent Builder para ejecutar el runtime objetivo."],
            ),
            AcpV2BuildStep(
                step_key="implement-agents",
                title="Implementar agentes y prompts",
                objective="Construir roles, instrucciones, handoffs y guardrails del sistema agentico.",
                depends_on=["select-runtime"],
                inputs=["agent_runtime", "prompts"],
                outputs=["agent-runtime.implementation"],
                actions=[
                    "Crear agentes segun role contracts.",
                    "Instalar prompts versionados y sus esquemas de salida.",
                    "Configurar guardrails y stop conditions.",
                ],
                validation=["Cada agente tiene rol, objetivo, entradas, salidas y senales de exito verificables."],
            ),
            AcpV2BuildStep(
                step_key="wire-tools",
                title="Conectar herramientas minimas",
                objective="Implementar solo las herramientas requeridas por capacidades y contratos aprobados.",
                depends_on=["implement-agents"],
                inputs=["tool_contracts"],
                outputs=["tools.implementation"],
                actions=[
                    "Crear adaptadores contra APIs o servicios del cliente.",
                    "Aplicar autenticacion, idempotencia, validaciones, retries y compensacion.",
                    "Bloquear side effects sin approval gate.",
                ],
                validation=["No hay herramientas redundantes ni side effects sin gobierno declarado."],
            ),
            AcpV2BuildStep(
                step_key="configure-memory",
                title="Configurar memoria y conocimiento",
                objective="Materializar memoria corta, memoria larga, RAG y politicas de recuperacion.",
                depends_on=["wire-tools"],
                inputs=["memory_strategy", "knowledge_sources"],
                outputs=["memory.implementation"],
                actions=[
                    "Configurar presupuestos de contexto y compactacion.",
                    "Preparar fuentes, ingestion, embeddings, retrieval, refresh y lineage cuando aplique.",
                ],
                validation=["La recuperacion usa solo contexto necesario y preserva trazabilidad de fuentes."],
            ),
            AcpV2BuildStep(
                step_key="run-tests",
                title="Ejecutar suite de pruebas",
                objective="Validar comportamiento, prompts, herramientas, memoria y recuperacion antes del go-live.",
                depends_on=["configure-memory"],
                inputs=["tests", "conformance"],
                outputs=["test-results"],
                actions=[
                    "Ejecutar pruebas funcionales y de conformance.",
                    "Registrar fallos, remediaciones y decisiones humanas pendientes.",
                ],
                validation=["No quedan reglas blocking de conformance fallando."],
            ),
            AcpV2BuildStep(
                step_key="release-handoff",
                title="Preparar handoff de construccion",
                objective="Entregar el paquete implementado o listo para ejecucion por la herramienta agentica elegida.",
                depends_on=["run-tests"],
                inputs=["test-results", "implementation-decisions.resolved"],
                outputs=["release-handoff"],
                actions=[
                    "Documentar runtime elegido, configuraciones finales y riesgos residuales.",
                    "Preparar instrucciones de ejecucion del entorno objetivo.",
                ],
                validation=["El handoff referencia artefactos portables y no identificadores operativos internos."],
            ),
        ],
    )


def _acp_v2_runtime_state_machine(construction_pack: ConstructionPackV1) -> list[dict[str, Any]]:
    behavior = construction_pack.behavior_spec
    states: list[dict[str, Any]] = []
    for index, state in enumerate(behavior.states, start=1):
        states.append(
            {
                "state_key": _portable_sequence_key("runtime_state", index),
                "title": f"Runtime state {index}",
                "actor": state.actor,
                "objective": state.objective,
                "outputs": state.outputs,
                "fallback": state.fallback,
                "requires_approval": state.requires_approval,
                "source_contract_ref": "behavior-spec.v1:states",
            }
        )
    return states


def _acp_v2_runtime(construction_pack: ConstructionPackV1) -> AcpV2AgentRuntime:
    behavior = construction_pack.behavior_spec
    topology = behavior.multi_agent_topology
    state_machine = _acp_v2_runtime_state_machine(construction_pack)

    if topology is not None:
        agents = [
            AcpV2RuntimeAgent(
                agent_key=agent.agent_key,
                role=agent.role,
                goal=agent.purpose,
                runtime_mode=agent.runtime_mode,
                inputs=agent.input_contracts,
                outputs=agent.output_contracts,
                tools=agent.permissions.allowed_tools,
                memory_refs=list(construction_pack.memory_policy.retrieval_scopes),
                handoff_targets=[
                    handoff.to_agent
                    for handoff in topology.handoff_contracts
                    if handoff.from_agent == agent.agent_key
                ],
                success_signals=agent.success_signals,
                failure_mode=agent.failure_mode,
            )
            for agent in topology.agent_contracts
        ]
        routing_rules = [
            {
                "handoff_key": handoff.handoff_key,
                "from_agent": handoff.from_agent,
                "to_agent": handoff.to_agent,
                "trigger": handoff.trigger,
                "ownership_transfer": handoff.ownership_transfer,
                "required_artifacts": handoff.required_artifacts,
                "success_criteria": handoff.success_criteria,
                "failure_behavior": handoff.failure_behavior,
            }
            for handoff in topology.handoff_contracts
        ]
        routing_rules.extend(
            {
                "message_key": message.message_key,
                "from_agent": message.from_agent,
                "to_agent": message.to_agent,
                "purpose": message.purpose,
                "required_fields": message.required_fields,
                "idempotency_strategy": message.idempotency_strategy,
                "failure_behavior": message.failure_behavior,
            }
            for message in topology.message_contracts
        )
        return AcpV2AgentRuntime(
            runtime_model=topology.runtime_pattern,
            orchestration_pattern=topology.declared_pattern,
            state_machine=state_machine,
            routing_rules=routing_rules,
            agents=agents,
            failure_modes=list(topology.failure_isolation_rules),
            execution_budget=topology.execution_budget.model_dump(mode="json") if topology.execution_budget is not None else {},
        )

    return AcpV2AgentRuntime(
        runtime_model=behavior.execution_pattern,
        orchestration_pattern=behavior.reasoning_pattern,
        state_machine=state_machine,
        routing_rules=[
            {
                "policy": "sequential_state_progression",
                "checkpoint_policy": behavior.checkpoint_policy,
                "retry_strategy": behavior.retry_strategy,
                "compensation_strategy": behavior.compensation_strategy,
                "approval_pause": behavior.approval_pause,
            }
        ],
        agents=[
            AcpV2RuntimeAgent(
                agent_key="primary_agent",
                role="Agente principal",
                goal=behavior.execution_pattern,
                runtime_mode="single_agent",
                inputs=["blueprint-core.v1", "prompt-pack.v1", "tool-contract.v1", "memory-policy.v1"],
                outputs=list(behavior.outputs),
                tools=[tool.name for tool in construction_pack.tool_contracts],
                memory_refs=list(construction_pack.memory_policy.retrieval_scopes),
                handoff_targets=[],
                success_signals=list(behavior.termination_criteria),
                failure_mode=behavior.timeout_policy,
            )
        ],
        failure_modes=[behavior.retry_strategy, behavior.compensation_strategy, behavior.timeout_policy],
        execution_budget={},
    )


def _acp_v2_implementation_decisions(
    blueprint_core: BlueprintCoreV1,
    construction_pack: ConstructionPackV1,
) -> list[AcpV2ImplementationDecision]:
    decisions: list[AcpV2ImplementationDecision] = []
    seen: set[str] = set()
    for gap in construction_pack.gaps:
        key = f"gap:{gap.code}"
        seen.add(key)
        required = gap.severity == "blocking"
        decisions.append(
            AcpV2ImplementationDecision(
                decision_key=key,
                decision_type="implementation_gap" if required else "implementation_follow_up",
                question=gap.summary,
                owner="implementation_owner",
                timing="before_runtime_build" if required else "during_runtime_hardening",
                required=required,
                options=[
                    gap.remediation or "Resolver y documentar la decision antes de continuar.",
                    "Asignar owner y convertir en waiver temporal con fecha de cierre.",
                ],
                impact=gap.remediation or gap.summary,
                default_option=gap.remediation,
                source_ref="construction-pack.v1:gaps",
            )
        )

    for question in blueprint_core.open_questions:
        key = f"blueprint-question:{question.key}"
        if key in seen:
            continue
        decisions.append(
            AcpV2ImplementationDecision(
                decision_key=key,
                decision_type="blueprint_open_question",
                question=question.question,
                owner=question.owner or "business_owner",
                timing="before_runtime_build",
                required=True,
                options=[
                    "Responder con decision aprobada por el owner indicado.",
                    "Documentar que la decision depende del entorno de implementacion y moverla al backlog del ACP.",
                ],
                impact="Puede modificar alcance, configuracion o criterios de aceptacion durante la implementacion.",
                default_option="Responder con decision aprobada por el owner indicado.",
                source_ref="blueprint-core.v1:open_questions",
            )
        )
    return decisions


def _acp_v2_decision_registry(
    implementation_decisions: list[AcpV2ImplementationDecision],
    technology_decisions: list[AcpV2TechnologyDecision] | None = None,
) -> list[AcpV2DecisionRegistryEntry]:
    registry: list[AcpV2DecisionRegistryEntry] = []
    for decision in implementation_decisions:
        if not decision.required:
            classification: Literal["mandatory", "optional", "deferable", "environment_dependent"] = "deferable"
            blocking_scope: Literal["package", "implementation", "none"] = "none"
        elif decision.decision_type == "implementation_gap":
            classification = "environment_dependent"
            blocking_scope = "implementation"
        elif decision.decision_type == "implementation_follow_up":
            classification = "optional"
            blocking_scope = "none"
        else:
            classification = "mandatory"
            blocking_scope = "implementation"

        options = [
            AcpV2DecisionOption(
                option_key=_portable_sequence_key("option", index),
                label=option[:72],
                description=option,
                tradeoffs=[
                    "Puede modificar configuracion, alcance tecnico o secuencia de implementacion.",
                    "Debe registrarse con owner y fecha si se convierte en waiver temporal.",
                ],
                recommended=(index == 1 and bool(decision.default_option)),
            )
            for index, option in enumerate(decision.options, start=1)
        ]
        if not options:
            options.append(
                AcpV2DecisionOption(
                    option_key="option_01",
                    label="Resolver con owner",
                    description="Responder la decision con evidencia y owner antes de implementar el componente afectado.",
                    tradeoffs=["Bloquea la implementacion del componente afectado si no se responde."],
                    recommended=True,
                )
            )

        registry.append(
            AcpV2DecisionRegistryEntry(
                decision_key=decision.decision_key,
                classification=classification,
                blocking_scope=blocking_scope,
                question=decision.question,
                context=f"Decision derivada de {decision.source_ref}. Timing sugerido: {decision.timing}.",
                owner=decision.owner,
                recommended_moment=decision.timing,
                impact=decision.impact,
                options=options,
                examples=[
                    "Ejemplo de cierre: seleccionar la opcion recomendada y documentar evidencia o restriccion del entorno.",
                    "Ejemplo de waiver: diferir la decision con owner, fecha objetivo e impacto aceptado.",
                ],
                source_ref=decision.source_ref,
            )
        )
    for technology in technology_decisions or []:
        classification: Literal["mandatory", "optional", "deferable", "environment_dependent"] = (
            "environment_dependent"
            if technology.required_for_implementation
            else "optional"
        )
        blocking_scope: Literal["package", "implementation", "none"] = (
            "implementation"
            if technology.required_for_implementation
            else "none"
        )
        registry.append(
            AcpV2DecisionRegistryEntry(
                decision_key=technology.decision_key,
                classification=classification,
                blocking_scope=blocking_scope,
                question=technology.question,
                context=(
                    f"Decision tecnologica de categoria {technology.category}. "
                    "Se resuelve durante la implementacion del ACP, no durante la generacion del paquete."
                ),
                owner="implementation_owner",
                recommended_moment="during_acp_execution_before_generating_stack_specific_code",
                impact=technology.default_guidance,
                options=[
                    AcpV2DecisionOption(
                        option_key=option.option_key,
                        label=option.label,
                        description=option.rationale,
                        tradeoffs=list(option.tradeoffs),
                        recommended=option.recommendation_level == "recommended",
                    )
                    for option in technology.options
                ],
                examples=[
                    example
                    for option in technology.options
                    for example in option.examples
                ],
                source_ref=technology.source_ref,
            )
        )
    return registry


def _acp_v2_checkpoints(
    build_plan: AcpV2BuildPlan,
    agent_runtime: AcpV2AgentRuntime,
    decision_registry: list[AcpV2DecisionRegistryEntry],
) -> list[AcpV2CheckpointSpec]:
    checkpoints: list[AcpV2CheckpointSpec] = []
    for index, step in enumerate(build_plan.steps, start=1):
        checkpoints.append(
            AcpV2CheckpointSpec(
                checkpoint_key=_portable_sequence_key("checkpoint_construction", index),
                title=f"Construction checkpoint {index}",
                scope="construction",
                trigger=f"after:{_portable_sequence_key('construction_step', index)}",
                required_artifacts=list(step.outputs),
                validation=list(step.validation),
                resume_strategy="Reanudar desde el ultimo output validado; no repetir acciones con side effects sin idempotency key.",
                storage_hint="Persistir como archivo/registro portable del runtime elegido.",
                portable_ref=f"build_plan.steps[{index - 1}]",
            )
        )

    approval_required = any(bool(state.get("requires_approval")) for state in agent_runtime.state_machine)
    checkpoints.append(
        AcpV2CheckpointSpec(
            checkpoint_key="checkpoint_runtime_resume",
            title="Runtime resume checkpoint",
            scope="runtime",
            trigger="before_runtime_resume_or_retry",
            required_artifacts=["current_state", "last_valid_output", "pending_tool_calls"],
            validation=[
                "El estado actual existe en workflows.runtime_operational.",
                "No se reintentan operaciones no idempotentes sin confirmacion.",
            ],
            resume_strategy="Reconstruir solo el contexto minimo desde memory_strategy y el ultimo output valido.",
            storage_hint="Persistir fuera de Lean Agent Builder en el storage del runtime objetivo.",
            portable_ref="agent_runtime.state_machine",
        )
    )
    checkpoints.append(
        AcpV2CheckpointSpec(
            checkpoint_key="checkpoint_runtime_context_compaction",
            title="Runtime context compaction checkpoint",
            scope="runtime",
            trigger="when_context_budget_threshold_is_reached",
            required_artifacts=["short_term_summary", "retrieval_refs", "open_decisions"],
            validation=[
                "El resumen conserva objetivo, restricciones, decisiones y referencias necesarias.",
                "No se envia contexto redundante al LLM despues de compactar.",
            ],
            resume_strategy="Usar resumen compacto y recuperar artefactos por referencia.",
            storage_hint="Persistir en la memoria corta/larga declarada por memory_strategy.",
            portable_ref="memory_strategy.context_budget",
        )
    )
    if approval_required:
        checkpoints.append(
            AcpV2CheckpointSpec(
                checkpoint_key="checkpoint_runtime_approval_gate",
                title="Runtime approval checkpoint",
                scope="runtime",
                trigger="before_approval_required_action",
                required_artifacts=["approval_request", "rationale", "impact", "rollback_or_compensation_plan"],
                validation=[
                    "La aprobacion humana incluye contexto, riesgo e impacto.",
                    "La accion no continua hasta recibir decision explicita.",
                ],
                resume_strategy="Continuar, cancelar o replanificar segun decision humana registrada.",
                storage_hint="Persistir como decision auditable del runtime objetivo.",
                portable_ref="agent_runtime.state_machine.requires_approval",
            )
        )

    required_decisions = [entry.decision_key for entry in decision_registry if entry.blocking_scope == "implementation"]
    deferred_decisions = [entry.decision_key for entry in decision_registry if entry.classification == "deferable"]
    checkpoints.append(
        AcpV2CheckpointSpec(
            checkpoint_key="checkpoint_decisions_required_closed",
            title="Required implementation decisions checkpoint",
            scope="human_decision",
            trigger="before_implementing_affected_component",
            required_artifacts=required_decisions,
            validation=[
                "Cada decision obligatoria o dependiente del entorno tiene respuesta, waiver o owner asignado.",
                "Ninguna decision pendiente bloquea la existencia del ACP; solo la implementacion del componente afectado.",
            ],
            resume_strategy="Reanudar el workflow de implementacion cuando las decisiones requeridas tengan cierre controlado.",
            storage_hint="Persistir en decision registry portable, no como estado interno de Lean.",
            portable_ref="decision_registry",
        )
    )
    if deferred_decisions:
        checkpoints.append(
            AcpV2CheckpointSpec(
                checkpoint_key="checkpoint_decisions_deferred_registered",
                title="Deferred decisions checkpoint",
                scope="human_decision",
                trigger="before_release_handoff",
                required_artifacts=deferred_decisions,
                validation=[
                    "Cada decision diferible incluye owner, momento recomendado, impacto y opciones.",
                    "La decision diferible no bloquea el paquete, pero queda visible para implementacion.",
                ],
                resume_strategy="Retomar la decision en el momento recomendado por decision_registry.",
                storage_hint="Persistir como backlog portable de decisiones humanas.",
                portable_ref="decision_registry[classification=deferable]",
            )
        )
    return checkpoints


def _checkpoint_for(checkpoints: list[AcpV2CheckpointSpec], key: str) -> str:
    return key if any(checkpoint.checkpoint_key == key for checkpoint in checkpoints) else ""


def _acp_v2_workflows(
    build_plan: AcpV2BuildPlan,
    agent_runtime: AcpV2AgentRuntime,
    decision_registry: list[AcpV2DecisionRegistryEntry],
    checkpoints: list[AcpV2CheckpointSpec],
) -> list[AcpV2WorkflowSpec]:
    construction_node_by_step: dict[str, str] = {}
    construction_nodes: list[AcpV2WorkflowNode] = []
    for index, step in enumerate(build_plan.steps, start=1):
        node_key = _portable_sequence_key("construction_step", index)
        construction_node_by_step[step.step_key] = node_key
        construction_nodes.append(
            AcpV2WorkflowNode(
                node_key=node_key,
                title=step.title,
                workflow_role="construction",
                objective=step.objective,
                actor="implementation_agent_or_developer",
                inputs=list(step.inputs),
                outputs=list(step.outputs),
                portable_state="pending|running|completed|blocked",
                checkpoint_ref=_checkpoint_for(checkpoints, _portable_sequence_key("checkpoint_construction", index)),
                decision_refs=[entry.decision_key for entry in decision_registry] if step.step_key == "resolve-implementation-decisions" else [],
                timeout_policy="Definir SLA por herramienta agentica antes de ejecutar la implementacion.",
                retry_policy="Reintentar solo operaciones idempotentes o con compensacion declarada.",
                context_refs=["system_specification", "build_plan", "decision_registry"],
            )
        )

    construction_transitions: list[AcpV2WorkflowTransition] = []
    for index, step in enumerate(build_plan.steps, start=1):
        target = construction_node_by_step[step.step_key]
        dependencies = [construction_node_by_step[item] for item in step.depends_on if item in construction_node_by_step]
        if not dependencies and index > 1:
            dependencies = [_portable_sequence_key("construction_step", index - 1)]
        for dep_index, dependency in enumerate(dependencies, start=1):
            construction_transitions.append(
                AcpV2WorkflowTransition(
                    transition_key=f"{dependency}_to_{target}_{dep_index}",
                    from_node=dependency,
                    to_node=target,
                    condition="dependency_completed_and_checkpoint_valid",
                    routing_rule="dependency_order",
                    checkpoint_ref=_checkpoint_for(checkpoints, _portable_sequence_key("checkpoint_construction", index)),
                    failure_behavior="pause_and_request_remediation",
                )
            )

    runtime_nodes = [
        AcpV2WorkflowNode(
            node_key=str(state.get("state_key")),
            title=str(state.get("title")),
            workflow_role="runtime",
            objective=str(state.get("objective") or ""),
            actor=str(state.get("actor") or "agent_runtime"),
            inputs=["runtime_input", "memory_context", "tool_results"],
            outputs=[str(item) for item in state.get("outputs", [])],
            portable_state="idle|executing|waiting_for_tool|waiting_for_approval|completed|failed",
            checkpoint_ref=(
                _checkpoint_for(checkpoints, "checkpoint_runtime_approval_gate")
                if state.get("requires_approval")
                else _checkpoint_for(checkpoints, "checkpoint_runtime_resume")
            ),
            decision_refs=[],
            timeout_policy="Aplicar timeout por nodo y circuit breaker ante fallos repetidos.",
            retry_policy="Usar retry con backoff solo si la accion es idempotente.",
            context_refs=["agent_runtime", "memory_strategy", "tool_contracts"],
        )
        for state in agent_runtime.state_machine
    ]
    runtime_transitions = [
        AcpV2WorkflowTransition(
            transition_key=f"{runtime_nodes[index].node_key}_to_{runtime_nodes[index + 1].node_key}",
            from_node=runtime_nodes[index].node_key,
            to_node=runtime_nodes[index + 1].node_key,
            condition="state_output_valid",
            routing_rule="sequential_or_runtime_adapter",
            checkpoint_ref=_checkpoint_for(checkpoints, "checkpoint_runtime_resume"),
            failure_behavior="retry_or_resume_from_last_checkpoint",
        )
        for index in range(max(len(runtime_nodes) - 1, 0))
    ]

    decision_refs = [entry.decision_key for entry in decision_registry]
    human_nodes = [
        AcpV2WorkflowNode(
            node_key="human_decision_step_01",
            title="Decision intake",
            workflow_role="human_decision",
            objective="Presentar decisiones pendientes con contexto, owner, impacto y momento recomendado.",
            actor="implementation_owner",
            inputs=["decision_registry"],
            outputs=["decision_queue"],
            portable_state="pending|answered|waived|deferred",
            checkpoint_ref=_checkpoint_for(checkpoints, "checkpoint_decisions_required_closed"),
            decision_refs=decision_refs,
            timeout_policy="Definir SLA de respuesta por criticidad de decision.",
            retry_policy="Recordatorio escalable; no inventar respuestas ausentes.",
            context_refs=["decision_registry", "system_specification"],
        ),
        AcpV2WorkflowNode(
            node_key="human_decision_step_02",
            title="Resolve required decisions",
            workflow_role="human_decision",
            objective="Cerrar decisiones obligatorias o dependientes del entorno antes de implementar componentes afectados.",
            actor="business_or_technical_owner",
            inputs=["decision_queue"],
            outputs=["resolved_required_decisions"],
            portable_state="pending|answered|waived",
            checkpoint_ref=_checkpoint_for(checkpoints, "checkpoint_decisions_required_closed"),
            decision_refs=[entry.decision_key for entry in decision_registry if entry.blocking_scope == "implementation"],
            timeout_policy="Bloquear solo el componente afectado si no hay respuesta.",
            retry_policy="Permitir waiver documentado con owner e impacto.",
            context_refs=["decision_registry"],
        ),
        AcpV2WorkflowNode(
            node_key="human_decision_step_03",
            title="Register deferred decisions",
            workflow_role="human_decision",
            objective="Registrar decisiones diferibles sin bloquear la existencia del ACP.",
            actor="implementation_owner",
            inputs=["decision_queue"],
            outputs=["deferred_decision_backlog"],
            portable_state="registered|deferred",
            checkpoint_ref=_checkpoint_for(checkpoints, "checkpoint_decisions_deferred_registered"),
            decision_refs=[entry.decision_key for entry in decision_registry if entry.classification == "deferable"],
            timeout_policy="Sin bloqueo de paquete; revisar en el momento recomendado.",
            retry_policy="Reabrir cuando el runtime/stack seleccionado requiera la decision.",
            context_refs=["decision_registry"],
        ),
        AcpV2WorkflowNode(
            node_key="human_decision_step_04",
            title="Implementation unblocked",
            workflow_role="human_decision",
            objective="Permitir continuar implementacion con decisiones requeridas cerradas y diferibles registradas.",
            actor="implementation_agent_or_developer",
            inputs=["resolved_required_decisions", "deferred_decision_backlog"],
            outputs=["implementation_decision_state"],
            portable_state="ready_for_implementation",
            checkpoint_ref=_checkpoint_for(checkpoints, "checkpoint_decisions_required_closed"),
            decision_refs=decision_refs,
            timeout_policy="No aplica.",
            retry_policy="No aplica.",
            context_refs=["decision_registry", "build_plan"],
        ),
    ]
    human_transitions = [
        AcpV2WorkflowTransition(
            transition_key="human_decision_step_01_to_human_decision_step_02",
            from_node="human_decision_step_01",
            to_node="human_decision_step_02",
            condition="required_or_environment_dependent_decisions_exist",
            routing_rule="human_owner_routing",
            requires_decision=True,
            checkpoint_ref=_checkpoint_for(checkpoints, "checkpoint_decisions_required_closed"),
            failure_behavior="pause_affected_component_only",
        ),
        AcpV2WorkflowTransition(
            transition_key="human_decision_step_02_to_human_decision_step_03",
            from_node="human_decision_step_02",
            to_node="human_decision_step_03",
            condition="required_decisions_answered_or_waived",
            routing_rule="deferred_decision_registration",
            checkpoint_ref=_checkpoint_for(checkpoints, "checkpoint_decisions_deferred_registered"),
            failure_behavior="continue_package_with_visible_backlog",
        ),
        AcpV2WorkflowTransition(
            transition_key="human_decision_step_03_to_human_decision_step_04",
            from_node="human_decision_step_03",
            to_node="human_decision_step_04",
            condition="deferred_decisions_registered",
            routing_rule="implementation_unblock",
            checkpoint_ref=_checkpoint_for(checkpoints, "checkpoint_decisions_required_closed"),
            failure_behavior="continue_only_for_unaffected_components",
        ),
    ]

    workflows = [
        AcpV2WorkflowSpec(
            workflow_key="construction_workflow",
            workflow_type="construction",
            topology="sequential",
            entry_node=construction_nodes[0].node_key if construction_nodes else "",
            terminal_nodes=[construction_nodes[-1].node_key] if construction_nodes else [],
            nodes=construction_nodes,
            transitions=construction_transitions,
            portable_state_policy="Las claves de nodos son genericas y no dependen de etapas internas del productor.",
            handoff_contract_refs=[],
        ),
        AcpV2WorkflowSpec(
            workflow_key="runtime_operational_workflow",
            workflow_type="runtime_operational",
            topology="hierarchical" if len(agent_runtime.agents) > 1 else "sequential",
            entry_node=runtime_nodes[0].node_key if runtime_nodes else "",
            terminal_nodes=[runtime_nodes[-1].node_key] if runtime_nodes else [],
            nodes=runtime_nodes,
            transitions=runtime_transitions,
            portable_state_policy="El runtime objetivo interpreta estados genericos y mapea rutas segun su framework.",
            handoff_contract_refs=[
                str(rule.get("handoff_key"))
                for rule in agent_runtime.routing_rules
                if isinstance(rule, dict) and rule.get("handoff_key")
            ],
        ),
        AcpV2WorkflowSpec(
            workflow_key="human_decision_resolution_workflow",
            workflow_type="human_decision_resolution",
            topology="event_driven",
            entry_node=human_nodes[0].node_key,
            terminal_nodes=[human_nodes[-1].node_key],
            nodes=human_nodes,
            transitions=human_transitions,
            portable_state_policy="Las decisiones humanas se gestionan por decision_registry y no por estados internos de Lean.",
            handoff_contract_refs=[],
        ),
    ]
    return workflows


def _acp_v2_tool_contracts(construction_pack: ConstructionPackV1) -> list[AcpV2ToolContractRef]:
    return [
        AcpV2ToolContractRef(
            tool_key=tool.name,
            display_name=tool.name,
            purpose=tool.purpose,
            requirement_level=_acp_v2_tool_requirement_level(tool),
            capability=tool.archetype,
            integration_kind=tool.integration_kind,
            auth_requirements=[item for item in [tool.auth_reference, *tool.permissions, *tool.scopes] if item],
            side_effects=tool.side_effects,
            idempotent=tool.idempotent,
            input_schema=tool.input_schema.model_dump(mode="json"),
            output_schema=tool.output_schema.model_dump(mode="json"),
            validations=list(tool.validations),
            retry_strategy=tool.retry_strategy,
            compensation_strategy=tool.compensation_strategy,
            source_ref=f"tool-contract.v1:{tool.name}",
        )
        for tool in construction_pack.tool_contracts
    ]


def _portable_key(value: str, fallback: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    normalized = "_".join(item for item in normalized.split("_") if item)
    return normalized or fallback


def _capability_key_for_tool(tool: ToolContractV1) -> str:
    base = tool.archetype or tool.integration_kind or tool.name
    return f"capability_{_portable_key(base, 'tool')}"


def _acp_v2_tool_requirement_level(
    tool: ToolContractV1,
) -> Literal["required", "optional", "conditional", "replaceable", "not_recommended"]:
    review_state = (tool.contract_review_state or "").lower()
    if review_state in {"rejected", "not_recommended"}:
        return "not_recommended"
    if tool.side_effects and not tool.requires_approval:
        return "not_recommended"
    if review_state in {"approved", "complete"}:
        return "required"
    if tool.risk_level.lower() in {"critical", "high"}:
        return "replaceable"
    if tool.integration_kind.lower() in {"internal", "lean_internal", "platform_internal"}:
        return "replaceable"
    return "optional" if review_state in {"optional", "candidate"} else "replaceable"


def _acp_v2_binding_type(tool: ToolContractV1) -> Literal["producer_internal_tool", "external_api", "abstract_contract", "runtime_adapter"]:
    integration = tool.integration_kind.lower()
    endpoint = tool.endpoint_reference.lower()
    if "lean" in integration or "internal" in integration or "lean" in endpoint:
        return "producer_internal_tool"
    if "api" in integration or "http" in endpoint or endpoint.startswith("openapi"):
        return "external_api"
    if integration in {"runtime", "framework", "sdk"}:
        return "runtime_adapter"
    return "abstract_contract"


def _acp_v2_provider_boundary(
    binding_type: Literal["producer_internal_tool", "external_api", "abstract_contract", "runtime_adapter"],
) -> Literal["producer_internal", "customer_external", "framework_runtime", "abstract"]:
    if binding_type == "producer_internal_tool":
        return "producer_internal"
    if binding_type == "external_api":
        return "customer_external"
    if binding_type == "runtime_adapter":
        return "framework_runtime"
    return "abstract"


def _acp_v2_tool_binding(tool: ToolContractV1) -> AcpV2ToolBinding:
    capability_key = _capability_key_for_tool(tool)
    requirement_level = _acp_v2_tool_requirement_level(tool)
    binding_type = _acp_v2_binding_type(tool)
    provider_boundary = _acp_v2_provider_boundary(binding_type)
    replaceable = requirement_level in {"replaceable", "optional", "not_recommended"} or provider_boundary == "producer_internal"
    permissions = [item for item in [*tool.permissions, *tool.scopes] if item]
    credentials_policy = (
        "Definir credenciales en el entorno del cliente; no incluir secretos en el ACP."
        if tool.auth_reference
        else "No se declara autenticacion; confirmar durante implementacion."
    )
    cost_profile = "variable_by_provider_or_usage" if binding_type == "external_api" else "implementation_effort_only"
    risk_profile = (
        f"risk={tool.risk_level}; side_effects={'yes' if tool.side_effects else 'no'}; "
        f"sensitive_data={', '.join(tool.sensitive_data) if tool.sensitive_data else 'none'}"
    )
    fallback_strategy = tool.failure_mode or tool.compensation_strategy or "Degradar con mensaje claro y registrar incidente."
    return AcpV2ToolBinding(
        binding_key=f"binding_{_portable_key(tool.name, 'tool')}",
        capability_key=capability_key,
        tool_key=tool.name,
        binding_type=binding_type,
        provider_boundary=provider_boundary,
        requirement_level="replaceable" if requirement_level == "conditional" else requirement_level,
        replaceable=replaceable,
        replacement_strategy=(
            "Reemplazar por cualquier adapter que cumpla abstract_inputs, abstract_outputs, permisos y politicas de side effects."
            if replaceable
            else "Mantener salvo decision explicita del owner de implementacion."
        ),
        external_contract_hint=tool.endpoint_reference or f"abstract://{capability_key}",
        credentials_policy=credentials_policy,
        permissions=permissions,
        side_effects=tool.side_effects,
        idempotent=tool.idempotent,
        cost_profile=cost_profile,
        risk_profile=risk_profile,
        fallback_strategy=fallback_strategy,
        source_ref=f"tool-contract.v1:{tool.name}",
    )


def _acp_v2_tool_bindings(construction_pack: ConstructionPackV1) -> list[AcpV2ToolBinding]:
    return [_acp_v2_tool_binding(tool) for tool in construction_pack.tool_contracts]


def _merge_schema_shapes(tools: list[ToolContractV1], attr: str) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: set[str] = set()
    for tool in tools:
        schema = getattr(tool, attr)
        payload = schema.model_dump(mode="json")
        properties.update(payload.get("properties", {}))
        required.update(str(item) for item in payload.get("required", []))
    return {"type": "object", "properties": properties, "required": sorted(required)}


def _portable_location_hint(value: str, fallback_key: str) -> str:
    normalized = _coalesce(value, fallback="")
    lower_value = normalized.lower().replace("\\", "/")
    portable_prefixes = (
        "artifact://",
        "http://",
        "https://",
        "s3://",
        "gs://",
        "azure://",
        "tool://",
        "repo://",
    )
    if lower_value.startswith(portable_prefixes):
        return normalized
    return f"artifact://knowledge/{_portable_key(fallback_key, 'knowledge_source')}"


def _acp_v2_rag_capability_dependencies(*, enabled: bool) -> list[AcpV2RagCapabilityDependency]:
    return [
        AcpV2RagCapabilityDependency(
            capability_key="capability_document_ingestion",
            required=enabled,
            reason="Normalizar, parsear y versionar fuentes documentales aprobadas antes de construir indices reutilizables.",
            fallback="Registrar fuente como artifact:// y pedir ingestion manual durante implementacion.",
        ),
        AcpV2RagCapabilityDependency(
            capability_key="capability_embedding",
            required=enabled,
            reason="Convertir chunks aprobados en representaciones vectoriales sin fijar proveedor ni modelo desde el ACP.",
            fallback="Usar busqueda lexical o retrieval humano asistido hasta seleccionar proveedor de embeddings.",
        ),
        AcpV2RagCapabilityDependency(
            capability_key="capability_vector_search",
            required=enabled,
            reason="Consultar conocimiento indexado por similitud, filtros y freshness sin imponer un vector store unico.",
            fallback="Usar busqueda documental filtrada o repository search mientras se decide el vector store.",
        ),
        AcpV2RagCapabilityDependency(
            capability_key="capability_knowledge_retrieval",
            required=enabled,
            reason="Recuperar evidencia citada y trazable para evitar saturar la ventana de contexto.",
            fallback="Responder con falta de evidencia y abrir decision humana cuando no exista retrieval confiable.",
        ),
    ]


def _acp_v2_rag_capability_contracts(
    construction_pack: ConstructionPackV1,
    knowledge_tool_refs: list[str],
) -> list[AcpV2CapabilityContract]:
    consumers = [agent.agent_key for agent in _acp_v2_runtime(construction_pack).agents]
    memory_refs = ["documentary_knowledge_sources", "rag_vector_index"]
    return [
        AcpV2CapabilityContract(
            capability_key="capability_document_ingestion",
            title="document_ingestion",
            description="Ingerir fuentes documentales aprobadas como artefactos versionados y trazables.",
            requirement_level="required",
            rationale="RAG portable necesita una capacidad abstracta de ingestion sin depender de servicios internos de Lean.",
            consumers=consumers,
            abstract_inputs={
                "type": "object",
                "properties": {
                    "artifact_ref": {"type": "string"},
                    "source_version": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["artifact_ref"],
            },
            abstract_outputs={
                "type": "object",
                "properties": {
                    "chunks": {"type": "array"},
                    "lineage": {"type": "array"},
                },
                "required": ["chunks", "lineage"],
            },
            required_permissions=["read_approved_knowledge_sources", "write_ingestion_manifest"],
            side_effect_profile="writes_index_or_manifest_requires_lineage",
            memory_refs=memory_refs,
            tool_refs=knowledge_tool_refs,
            replacement_options=["custom_document_pipeline", "managed_document_ai", "enterprise_content_connector"],
            source_refs=["knowledge-contract.v1", "memory-policy.v1"],
        ),
        AcpV2CapabilityContract(
            capability_key="capability_embedding",
            title="embedding",
            description="Generar embeddings para chunks aprobados respetando sensibilidad, version y proveedor seleccionado.",
            requirement_level="required",
            rationale="La decision de proveedor/modelo se deja abierta, pero la capacidad es necesaria para retrieval vectorial.",
            consumers=consumers,
            abstract_inputs={
                "type": "object",
                "properties": {"chunks": {"type": "array"}, "embedding_model": {"type": "string"}},
                "required": ["chunks"],
            },
            abstract_outputs={
                "type": "object",
                "properties": {"vectors": {"type": "array"}, "embedding_version": {"type": "string"}},
                "required": ["vectors"],
            },
            required_permissions=["process_approved_knowledge_chunks"],
            side_effect_profile="compute_only",
            memory_refs=memory_refs,
            tool_refs=[],
            replacement_options=["openai_embeddings", "azure_openai_embeddings", "local_embedding_model"],
            source_refs=["knowledge-contract.v1", "technology_decisions:tech_vector_store"],
        ),
        AcpV2CapabilityContract(
            capability_key="capability_vector_search",
            title="vector_search",
            description="Buscar chunks por similitud, filtros y freshness sobre el vector store que el implementador seleccione.",
            requirement_level="required",
            rationale="El ACP debe poder implementarse con distintos vector stores sin reescribir la estrategia de memoria.",
            consumers=consumers,
            abstract_inputs={
                "type": "object",
                "properties": {"query_vector": {"type": "array"}, "filters": {"type": "object"}, "top_k": {"type": "integer"}},
                "required": ["query_vector"],
            },
            abstract_outputs={
                "type": "object",
                "properties": {"matches": {"type": "array"}, "scores": {"type": "array"}},
                "required": ["matches"],
            },
            required_permissions=["read_vector_index"],
            side_effect_profile="read_only",
            memory_refs=memory_refs,
            tool_refs=[],
            replacement_options=["pgvector", "pinecone", "weaviate", "opensearch_vector"],
            source_refs=["knowledge-contract.v1", "technology_decisions:tech_vector_store"],
        ),
        AcpV2CapabilityContract(
            capability_key="capability_knowledge_retrieval",
            title="knowledge_retrieval",
            description="Recuperar conocimiento aprobado con trazabilidad, filtros, freshness, citas y grounding.",
            requirement_level="required",
            rationale="La estrategia de memoria/RAG requiere retrieval aunque la herramienta concreta pueda cambiar.",
            consumers=consumers,
            abstract_inputs={
                "type": "object",
                "properties": {"query": {"type": "string"}, "filters": {"type": "object"}, "budget": {"type": "object"}},
                "required": ["query"],
            },
            abstract_outputs={
                "type": "object",
                "properties": {"chunks": {"type": "array"}, "citations": {"type": "array"}, "confidence": {"type": "number"}},
                "required": ["chunks", "citations"],
            },
            required_permissions=["read_approved_knowledge_sources"],
            side_effect_profile="read_only",
            memory_refs=memory_refs,
            tool_refs=knowledge_tool_refs,
            replacement_options=["pgvector_retriever", "managed_vector_db_retriever", "enterprise_search_adapter"],
            source_refs=["knowledge-contract.v1", "memory-policy.v1"],
        ),
    ]


def _acp_v2_capability_catalog(
    construction_pack: ConstructionPackV1,
    tool_bindings: list[AcpV2ToolBinding],
) -> list[AcpV2CapabilityContract]:
    tools_by_capability: dict[str, list[ToolContractV1]] = {}
    bindings_by_capability: dict[str, list[AcpV2ToolBinding]] = {}
    for tool in construction_pack.tool_contracts:
        tools_by_capability.setdefault(_capability_key_for_tool(tool), []).append(tool)
    for binding in tool_bindings:
        bindings_by_capability.setdefault(binding.capability_key, []).append(binding)

    catalog: list[AcpV2CapabilityContract] = []
    for capability_key, tools in sorted(tools_by_capability.items()):
        bindings = bindings_by_capability.get(capability_key, [])
        requirement_levels = {binding.requirement_level for binding in bindings}
        if "required" in requirement_levels:
            requirement_level: Literal["required", "optional", "replaceable", "not_recommended"] = "required"
        elif "not_recommended" in requirement_levels:
            requirement_level = "not_recommended"
        elif "replaceable" in requirement_levels:
            requirement_level = "replaceable"
        else:
            requirement_level = "optional"
        side_effects = any(tool.side_effects for tool in tools)
        permissions = sorted({permission for tool in tools for permission in [*tool.permissions, *tool.scopes]})
        catalog.append(
            AcpV2CapabilityContract(
                capability_key=capability_key,
                title=tools[0].archetype or tools[0].name,
                description=" / ".join(sorted({tool.purpose for tool in tools if tool.purpose})),
                requirement_level=requirement_level,
                rationale=(
                    "Capacidad derivada de herramientas aprobadas y expresada como contrato abstracto reemplazable."
                    if requirement_level != "not_recommended"
                    else "Capacidad marcada como no recomendada por riesgo o ausencia de gobierno suficiente."
                ),
                consumers=[
                    agent.agent_key
                    for agent in _acp_v2_runtime(construction_pack).agents
                    if any(tool.name in agent.tools for tool in tools)
                ],
                abstract_inputs=_merge_schema_shapes(tools, "input_schema"),
                abstract_outputs=_merge_schema_shapes(tools, "output_schema"),
                required_permissions=permissions,
                side_effect_profile="side_effecting_requires_approval" if side_effects else "read_or_compute_only",
                memory_refs=list(construction_pack.memory_policy.retrieval_scopes),
                tool_refs=[tool.name for tool in tools],
                replacement_options=[
                    "Implementar API externa equivalente con el mismo contrato abstracto.",
                    "Usar adapter del runtime elegido si cumple inputs, outputs, permisos y fallbacks.",
                    "Sustituir por proceso humano si la capacidad no es automatizable de forma segura.",
                ],
                source_refs=[f"tool-contract.v1:{tool.name}" for tool in tools],
            )
        )

    if construction_pack.knowledge_contract.enabled:
        knowledge_tool_refs = [
            binding.tool_key
            for binding in tool_bindings
            if "document" in binding.capability_key or "retrieval" in binding.capability_key
        ]
        known_keys = {item.capability_key for item in catalog}
        for capability in _acp_v2_rag_capability_contracts(construction_pack, knowledge_tool_refs):
            if capability.capability_key in known_keys:
                continue
            catalog.append(capability)
            known_keys.add(capability.capability_key)
    return catalog


def _acp_v2_tool_analysis(
    construction_pack: ConstructionPackV1,
    capability_catalog: list[AcpV2CapabilityContract],
    tool_bindings: list[AcpV2ToolBinding],
) -> AcpV2ToolAnalysis:
    bindings_by_capability: dict[str, list[AcpV2ToolBinding]] = {}
    for binding in tool_bindings:
        bindings_by_capability.setdefault(binding.capability_key, []).append(binding)

    redundancies: list[AcpV2ToolRedundancy] = []
    for capability_key, bindings in sorted(bindings_by_capability.items()):
        if len(bindings) <= 1:
            continue
        redundancies.append(
            AcpV2ToolRedundancy(
                redundancy_key=f"redundancy_{_portable_key(capability_key, 'capability')}",
                capability_key=capability_key,
                tool_keys=[binding.tool_key for binding in bindings],
                severity="warning",
                rationale="Varias herramientas cubren la misma capacidad abstracta.",
                recommendation="Seleccionar una herramienta primaria y dejar las demas como reemplazo/fallback documentado.",
            )
        )

    incompatibilities: list[AcpV2ToolIncompatibility] = []
    for binding in tool_bindings:
        if binding.side_effects and not binding.idempotent:
            incompatibilities.append(
                AcpV2ToolIncompatibility(
                    incompatibility_key=f"incompatibility_non_idempotent_{_portable_key(binding.tool_key, 'tool')}",
                    tool_keys=[binding.tool_key],
                    severity="warning",
                    reason="La herramienta declara side effects y no es idempotente.",
                    mitigation="Exigir approval gate, idempotency key, compensacion o ejecucion humana asistida.",
                )
            )
        if binding.provider_boundary == "producer_internal":
            incompatibilities.append(
                AcpV2ToolIncompatibility(
                    incompatibility_key=f"incompatibility_producer_internal_{_portable_key(binding.tool_key, 'tool')}",
                    tool_keys=[binding.tool_key],
                    severity="blocking",
                    reason="La herramienta apunta a una capacidad interna del productor y no debe ser requisito portable.",
                    mitigation="Reemplazar por API externa, adapter del runtime o contrato abstracto durante implementacion.",
                )
            )

    not_recommended = [
        binding.tool_key
        for binding in tool_bindings
        if binding.requirement_level == "not_recommended"
    ]
    return AcpV2ToolAnalysis(
        summary=(
            f"{len(capability_catalog)} capacidades abstractas y {len(tool_bindings)} bindings de herramienta normalizados. "
            "El ACP minimiza sobreaprovisionamiento al exigir capacidades, no proveedores concretos."
        ),
        overprovisioning_policy=(
            "No agregar herramientas nuevas si una capacidad requerida ya esta cubierta por un binding reemplazable y validado."
        ),
        minimal_tooling_policy=(
            "Implementar solo capacidades requirement_level=required; las opcionales o reemplazables se activan cuando el flujo, riesgo o owner lo justifique."
        ),
        redundancy_findings=redundancies,
        incompatibility_findings=incompatibilities,
        not_recommended_tools=not_recommended,
    )


def _acp_v2_memory_strategy(construction_pack: ConstructionPackV1) -> AcpV2MemoryStrategy:
    memory = construction_pack.memory_policy
    knowledge = construction_pack.knowledge_contract
    return AcpV2MemoryStrategy(
        short_term={
            "strategy": memory.strategy,
            "summary_policy": memory.summary_policy,
            "invalidation_policy": memory.invalidation_policy,
            "review_trigger": memory.review_trigger,
            "goal_drift_guard": memory.goal_drift_guard,
        },
        long_term={
            "storage_layers": memory.storage_layers,
            "retention_policy": memory.retention_policy,
            "ttl_policy": memory.ttl_policy,
            "workspace_scope": memory.workspace_scope,
            "agent_scope": memory.agent_scope,
            "sensitivity_rules": memory.sensitivity_rules,
        },
        retrieval={
            "enabled": knowledge.enabled,
            "mode": knowledge.mode,
            "policy": knowledge.retrieval_policy.model_dump(mode="json") if knowledge.retrieval_policy is not None else {},
            "grounding_policy": knowledge.grounding_policy,
            "retrieval_scopes": memory.retrieval_scopes,
        },
        context_budget=[budget.model_dump(mode="json") for budget in memory.context_budgets],
        persistence={
            "write_policy": memory.write_policy,
            "checkpoints_required": memory.checkpoints_required,
            "source_lineage": knowledge.source_lineage,
            "refresh_policy": knowledge.refresh_policy.model_dump(mode="json") if knowledge.refresh_policy is not None else {},
        },
        source_refs=["memory-policy.v1", "knowledge-contract.v1"],
    )


def _acp_v2_memory_namespaces(construction_pack: ConstructionPackV1) -> list[AcpV2MemoryNamespace]:
    memory = construction_pack.memory_policy
    knowledge = construction_pack.knowledge_contract
    retention = _coalesce(memory.retention_policy, fallback="Retener segun politica del workspace y regulacion aplicable.")
    freshness = (
        knowledge.refresh_policy.frequency
        if knowledge.refresh_policy is not None and knowledge.refresh_policy.frequency
        else "manual_review_before_production"
    )
    privacy = "; ".join(memory.sensitivity_rules or knowledge.sensitivity_rules) or "No persistir secretos ni datos privados fuera del entorno del cliente."
    return [
        AcpV2MemoryNamespace(
            namespace_key="short_term_working_context",
            memory_type="short_term",
            purpose="Mantener el objetivo activo, ultimas decisiones y resumen minimo necesario para la siguiente accion.",
            scope="session_portable",
            read_roles=["planner", "executor", "evaluator"],
            write_roles=["runtime_adapter", "memory_manager"],
            retention_policy=_coalesce(memory.ttl_policy, fallback="Expirar al cerrar la ejecucion o al crear checkpoint consolidado."),
            compaction_policy=_coalesce(memory.summary_policy, fallback="Compactar por resumen antes de reinyectar contexto."),
            privacy_policy=privacy,
            freshness_policy="Invalidar al aprobar un nuevo artefacto o cambiar una decision humana.",
            portable_ref="memory://short_term_working_context",
        ),
        AcpV2MemoryNamespace(
            namespace_key="long_term_agent_profile",
            memory_type="long_term",
            purpose="Persistir reglas, decisiones aprobadas, perfiles de agentes y criterios de exito reutilizables.",
            scope="agent",
            read_roles=["planner", "executor", "evaluator"],
            write_roles=["memory_manager"],
            retention_policy=retention,
            compaction_policy="Persistir artefactos aprobados por version; no duplicar contenido que pueda referenciarse por artifact_ref.",
            privacy_policy=privacy,
            freshness_policy="Actualizar solo mediante aprobacion explicita o migration controlada.",
            portable_ref="memory://long_term_agent_profile",
        ),
        AcpV2MemoryNamespace(
            namespace_key="documentary_knowledge_sources",
            memory_type="documentary_knowledge",
            purpose="Catalogar fuentes documentales aprobadas con owner, sensibilidad, version y permisos.",
            scope="workspace",
            read_roles=["retrieval", "evaluator", "memory_manager"],
            write_roles=["knowledge_owner", "memory_manager"],
            retention_policy=retention,
            compaction_policy="Referenciar documentos por artifact_ref y almacenar solo metadata/resumen verificable.",
            privacy_policy=privacy,
            freshness_policy=freshness,
            portable_ref="memory://documentary_knowledge_sources",
        ),
        AcpV2MemoryNamespace(
            namespace_key="rag_vector_index",
            memory_type="rag_index",
            purpose=(
                "Indice vectorial reemplazable para chunks aprobados."
                if knowledge.enabled
                else "Namespace reservado para RAG si el implementador decide habilitarlo."
            ),
            scope="workspace",
            read_roles=["retrieval"],
            write_roles=["document_ingestion", "embedding"],
            retention_policy=retention if knowledge.enabled else "No materializar hasta habilitar knowledge_contract.",
            compaction_policy="Reindexar por version de fuente; eliminar chunks obsoletos segun deletion_policy.",
            privacy_policy=privacy,
            freshness_policy=freshness,
            portable_ref="memory://rag_vector_index",
        ),
        AcpV2MemoryNamespace(
            namespace_key="audit_decision_trace",
            memory_type="audit",
            purpose="Trazar decisiones humanas, preguntas de implementacion, evidencia usada y checkpoints.",
            scope="tenant",
            read_roles=["auditor", "evaluator", "implementation_owner"],
            write_roles=["runtime_adapter", "decision_owner"],
            retention_policy=retention,
            compaction_policy="No resumir eventos regulatorios; compactar solo vistas derivadas.",
            privacy_policy=privacy,
            freshness_policy="Append-only con correlacion a decision_registry y checkpoints.",
            portable_ref="memory://audit_decision_trace",
        ),
    ]


def _acp_v2_knowledge_artifacts(construction_pack: ConstructionPackV1) -> list[AcpV2KnowledgeArtifactRef]:
    knowledge = construction_pack.knowledge_contract
    refresh_triggers = (
        _normalized_items(knowledge.refresh_policy.triggers)
        if knowledge.refresh_policy is not None
        else []
    )
    expiration_policy = (
        knowledge.refresh_policy.expiration_policy
        if knowledge.refresh_policy is not None and knowledge.refresh_policy.expiration_policy
        else "Definir expiracion/freshness durante implementacion si la fuente cambia."
    )
    artifacts: list[AcpV2KnowledgeArtifactRef] = []
    for source in knowledge.sources:
        source_key = source.key or source.title or "knowledge_source"
        artifact_key = f"knowledge_artifact_{_portable_key(source_key, 'knowledge_source')}"
        owner = _coalesce(source.owner, fallback="knowledge_owner_pending")
        sensitivity = _coalesce(source.sensitivity, fallback="internal")
        artifacts.append(
            AcpV2KnowledgeArtifactRef(
                artifact_key=artifact_key,
                title=_coalesce(source.title, fallback=source_key),
                source_type=_coalesce(source.source_type, fallback="document_repository"),
                location_hint=_portable_location_hint(source.uri, source_key),
                owner=owner,
                sensitivity=sensitivity,
                license=_coalesce(source.license, fallback="pending_review"),
                source_version=_coalesce(source.source_version, fallback="pending"),
                indexing_required=knowledge.enabled,
                reason_to_index=(
                    f"Fuente declarada en knowledge_contract modo {knowledge.mode}; aporta grounding, citas y recuperacion selectiva."
                    if knowledge.enabled
                    else "Fuente catalogada como referencia; no requiere indice hasta habilitar RAG."
                ),
                ingestion_capability_ref="capability_document_ingestion",
                retrieval_capability_ref="capability_knowledge_retrieval",
                permissions=[f"read:{owner}", f"sensitivity:{sensitivity}"],
                refresh_triggers=refresh_triggers or ["manual_refresh_before_production"],
                expiration_policy=expiration_policy,
                source_ref=f"knowledge-contract.v1:{source_key}",
            )
        )
    return artifacts


def _acp_v2_rag_pipeline(construction_pack: ConstructionPackV1) -> AcpV2RagPipelineSpec:
    knowledge = construction_pack.knowledge_contract
    retrieval_policy = knowledge.retrieval_policy.model_dump(mode="json") if knowledge.retrieval_policy is not None else {}
    refresh_policy = knowledge.refresh_policy.model_dump(mode="json") if knowledge.refresh_policy is not None else {}
    return AcpV2RagPipelineSpec(
        enabled=knowledge.enabled,
        mode=knowledge.mode,
        capability_dependencies=_acp_v2_rag_capability_dependencies(enabled=knowledge.enabled),
        vector_store_decision_ref="tech_vector_store",
        ingestion_policy=knowledge.ingestion_policy.model_dump(mode="json") if knowledge.ingestion_policy is not None else {},
        embedding_policy=knowledge.embedding_policy.model_dump(mode="json") if knowledge.embedding_policy is not None else {},
        retrieval_policy=retrieval_policy,
        refresh_policy=refresh_policy,
        grounding_policy=knowledge.grounding_policy,
        citation_policy=_coalesce(
            knowledge.grounding_policy.get("citations_policy"),
            fallback="Citar artefactos recuperados o declarar falta de evidencia.",
        ),
        deletion_policy=_coalesce(
            str(refresh_policy.get("deletion_policy") or ""),
            fallback="Borrar o invalidar chunks cuando cambie la fuente o expire su permiso.",
        ),
        fallback_policy=_coalesce(
            str(retrieval_policy.get("fallback_behavior") or ""),
            knowledge.grounding_policy.get("no_evidence_behavior"),
            fallback="No inventar evidencia; abrir decision humana o continuar con contexto aprobado no documental.",
        ),
        source_refs=[f"knowledge-contract.v1:{source.key}" for source in knowledge.sources] or ["knowledge-contract.v1"],
    )


def _acp_v2_context_window_policy(construction_pack: ConstructionPackV1) -> AcpV2ContextWindowPolicy:
    memory = construction_pack.memory_policy
    return AcpV2ContextWindowPolicy(
        max_context_utilization_percent=85,
        short_term_budget_refs=[f"memory_budget_{_portable_key(budget.role, 'role')}" for budget in memory.context_budgets],
        compaction_trigger=_coalesce(
            *(budget.compaction_trigger for budget in memory.context_budgets),
            fallback="Compactar cuando el contexto supere 85% del presupuesto o antes de invocar retrieval extenso.",
        ),
        anti_redundancy_rules=[
            "Enviar solo resumen corto, decisiones abiertas y referencias a artefactos aprobados.",
            "No reenviar documentos completos si existe artifact_ref o knowledge_artifact_ref recuperable.",
            "Paginar resultados RAG y limitar chunks por top_k, filtros y presupuesto de rol.",
            "Recalcular contexto desde checkpoints portables, no desde historiales completos de Lean Agent Builder.",
        ],
        retrieval_context_policy=(
            "Recuperar unicamente fuentes necesarias para la tarea activa usando filtros, freshness, sensibilidad y owner."
        ),
        pagination_policy="Entregar evidencia en paginas/chunks con citas; pedir mas contexto solo si la confianza es insuficiente.",
        artifact_reference_policy="Referenciar artefactos por artifact://, memory:// o contract_ref en lugar de copiar contenido redundante.",
    )


def _acp_v2_memory_knowledge_plan(construction_pack: ConstructionPackV1) -> AcpV2MemoryKnowledgePlan:
    rag_pipeline = _acp_v2_rag_pipeline(construction_pack)
    capability_dependencies = [
        dependency.capability_key
        for dependency in rag_pipeline.capability_dependencies
        if dependency.required
    ]
    return AcpV2MemoryKnowledgePlan(
        namespaces=_acp_v2_memory_namespaces(construction_pack),
        knowledge_artifacts=_acp_v2_knowledge_artifacts(construction_pack),
        rag_pipeline=rag_pipeline,
        context_window_policy=_acp_v2_context_window_policy(construction_pack),
        capability_dependencies=capability_dependencies,
        source_refs=["memory-policy.v1", "knowledge-contract.v1", "capability_catalog", "technology_decisions:tech_vector_store"],
    )


def _acp_v2_knowledge_sources(construction_pack: ConstructionPackV1) -> list[AcpV2KnowledgeSource]:
    knowledge = construction_pack.knowledge_contract
    return [
        AcpV2KnowledgeSource(
            source_key=source.key,
            title=source.title,
            kind=source.source_type,
            location_hint=source.uri or "implementation-owner-provided",
            ingestion_required=knowledge.enabled,
            freshness=source.source_version or "pending",
            owner=source.owner or "knowledge_owner",
            source_ref=f"knowledge-contract.v1:{source.key}",
        )
        for source in knowledge.sources
    ]


def _acp_v2_prompts(construction_pack: ConstructionPackV1) -> list[AcpV2PromptRef]:
    prompts: list[AcpV2PromptRef] = []
    for prompt in _iter_prompt_artifacts(construction_pack.prompt_pack):
        prompts.append(
            AcpV2PromptRef(
                prompt_key=prompt.prompt_key,
                role=prompt.role,
                title=prompt.title,
                content=prompt.content,
                required=True,
                usage=_first_prompt_signal(prompt.content),
                context_sources=list(prompt.context_sources),
                input_contracts=list(prompt.input_contracts),
                output_schema=prompt.output_schema,
                guardrails=list(prompt.guardrails),
                source_ref=f"prompt-pack.v1:{prompt.prompt_key}",
            )
        )
    return prompts


def _acp_v2_tests(construction_pack: ConstructionPackV1) -> list[AcpV2TestAsset]:
    tests: list[AcpV2TestAsset] = []
    for case in construction_pack.evaluation_pack.acceptance_cases or construction_pack.evaluation_pack.cases:
        tests.append(
            AcpV2TestAsset(
                test_key=case.key,
                kind=case.category or "acceptance",
                title=case.title,
                scenario=case.scenario,
                expected_result=case.expected_result,
                required=True,
                acceptance_criteria=[case.expected_result],
                source_ref=f"evaluation-pack.v1:{case.key}",
            )
        )
    return tests


def _acp_v2_runtime_targets(agent_runtime: AcpV2AgentRuntime) -> list[AcpV2RuntimeTarget]:
    is_multi_agent = len(agent_runtime.agents) > 1 or agent_runtime.orchestration_pattern in {
        "supervisor_with_subagents",
        "hierarchical",
    }
    return [
        AcpV2RuntimeTarget(
            target_key="codex-cli",
            label="Codex CLI",
            category="agentic_ide",
            recommendation_level="recommended",
            required=False,
            rationale="Buen punto de partida para ejecutar el ACP como paquete local, iterar artefactos y mantener trazabilidad en workspace.",
            selection_criteria=[
                "El equipo quiere usar un asistente agentico local sobre el repositorio.",
                "Se requiere ejecutar el build_plan por pasos y revisar cambios antes de aplicarlos.",
                "Se prioriza portabilidad del paquete sobre acoplamiento a un framework especifico.",
            ],
            prerequisites=[
                "Codex CLI instalado o herramienta compatible con lectura de archivos locales.",
                "Repositorio o workspace donde materializar los artefactos del ACP.",
            ],
            tradeoffs=[
                "Acelera la implementacion guiada, pero no sustituye decisiones de entorno, credenciales o infraestructura.",
                "Puede requerir adaptadores manuales para runtimes productivos especificos.",
            ],
            adapter_notes=[
                "Usar build_plan como secuencia de ejecucion.",
                "Resolver technology_decisions antes de crear codigo dependiente del stack.",
                "Usar deployment_guide como checklist, no como script obligatorio.",
            ],
            source_ref="compatibility:codex-cli",
        ),
        AcpV2RuntimeTarget(
            target_key="openai-agents-sdk",
            label="OpenAI Agents SDK",
            category="agent_framework",
            recommendation_level="recommended" if not is_multi_agent else "compatible",
            required=False,
            rationale="Adecuado cuando se quiere implementar agentes, tools y handoffs con SDK programatico y evaluable.",
            selection_criteria=[
                "El equipo busca un runtime de agentes con tools y contratos explicitos.",
                "Se acepta implementar adaptadores desde agent_runtime y tool_contracts.",
            ],
            prerequisites=[
                "Seleccionar lenguaje soportado por el equipo.",
                "Configurar proveedor LLM y gestion segura de secrets en el entorno elegido.",
            ],
            tradeoffs=[
                "Ofrece estructura de runtime, pero exige decisiones de framework, despliegue y observabilidad.",
                "La seleccion de modelos y costos sigue siendo decision del implementador.",
            ],
            adapter_notes=[
                "Mapear agent_runtime.agents a agentes del SDK.",
                "Mapear tool_contracts a tools con validacion y approval gates.",
            ],
            source_ref="compatibility:openai-agents-sdk",
        ),
        AcpV2RuntimeTarget(
            target_key="langgraph",
            label="LangGraph",
            category="orchestration_runtime",
            recommendation_level="recommended" if is_multi_agent else "compatible",
            required=False,
            rationale="Conveniente para workflows con grafo, checkpoints y enrutamiento explicito.",
            selection_criteria=[
                "El agente requiere estados, transiciones, resume/checkpoints o subagentes coordinados.",
                "El equipo prefiere modelar el runtime como grafo verificable.",
            ],
            prerequisites=[
                "Equipo con experiencia Python/TypeScript segun implementacion seleccionada.",
                "Definir almacenamiento externo para checkpoints y memoria.",
            ],
            tradeoffs=[
                "Mayor control del grafo a cambio de mayor complejidad de implementacion.",
                "No debe asumirse como obligatorio si un runtime mas simple cumple el caso.",
            ],
            adapter_notes=[
                "Mapear workflows.runtime_operational_workflow a nodos y edges.",
                "Usar checkpoints como estrategia de resume fuera de Lean Agent Builder.",
            ],
            source_ref="compatibility:langgraph",
        ),
        AcpV2RuntimeTarget(
            target_key="custom-runtime",
            label="Runtime propio del cliente",
            category="custom_runtime",
            recommendation_level="compatible",
            required=False,
            rationale="Valido cuando existen restricciones corporativas, plataformas internas o integraciones no cubiertas por frameworks comunes.",
            selection_criteria=[
                "El cliente debe cumplir estandares internos de arquitectura, seguridad o despliegue.",
                "Se requiere integrar el agente con una plataforma corporativa existente.",
            ],
            prerequisites=[
                "Definir lenguaje, framework, autenticacion, persistencia, observabilidad y estrategia de pruebas.",
                "Implementar validacion de conformance contra el ACP v2.",
            ],
            tradeoffs=[
                "Maxima flexibilidad, pero mayor esfuerzo y responsabilidad de mantenimiento.",
                "Requiere construir adaptadores para prompts, herramientas, memoria y checkpoints.",
            ],
            adapter_notes=[
                "Usar el ACP como especificacion declarativa.",
                "No importar estructuras internas del productor; implementar adapters propios.",
            ],
            source_ref="compatibility:custom-runtime",
        ),
    ]


def _acp_v2_runtime_target_policy(runtime_targets: list[AcpV2RuntimeTarget]) -> AcpV2RuntimeTargetPolicy:
    return AcpV2RuntimeTargetPolicy(
        recommended_runtime=[
            target.target_key
            for target in runtime_targets
            if target.recommendation_level == "recommended"
        ],
        required_runtime=[],
        selection_policy=(
            "El ACP recomienda runtimes segun complejidad, workflows y capacidades, pero no impone uno. "
            "El implementador debe seleccionar el runtime durante la ejecucion del ACP."
        ),
        override_policy=(
            "Cualquier runtime compatible puede reemplazar la recomendacion si implementa workflows, tool_contracts, "
            "memory_strategy, checkpoints, decision_registry y conformance."
        ),
    )


def _technology_option(
    option_key: str,
    label: str,
    recommendation_level: Literal["recommended", "compatible", "optional", "not_recommended"],
    rationale: str,
    prerequisites: list[str],
    tradeoffs: list[str],
    examples: list[str],
) -> AcpV2TechnologyOption:
    return AcpV2TechnologyOption(
        option_key=option_key,
        label=label,
        recommendation_level=recommendation_level,
        rationale=rationale,
        prerequisites=prerequisites,
        tradeoffs=tradeoffs,
        examples=examples,
    )


def _acp_v2_technology_decisions(construction_pack: ConstructionPackV1) -> list[AcpV2TechnologyDecision]:
    needs_rag = construction_pack.knowledge_contract.enabled
    has_tools = bool(construction_pack.tool_contracts)
    return [
        AcpV2TechnologyDecision(
            decision_key="tech_language",
            category="language",
            question="Que lenguaje usara el equipo para implementar el agente?",
            selection_criteria=[
                "Experiencia del equipo implementador.",
                "Compatibilidad con el runtime agentico elegido.",
                "Ecosistema disponible para integraciones, memoria, testing y observabilidad.",
            ],
            options=[
                _technology_option(
                    "python",
                    "Python",
                    "recommended",
                    "Amplio ecosistema para agentes, RAG, evaluaciones y automatizacion backend.",
                    ["Equipo con capacidad Python.", "Runtime compatible o adaptador propio."],
                    ["Rapido para prototipar y fuerte en IA; puede requerir disciplina adicional para grandes codebases."],
                    ["FastAPI + OpenAI Agents SDK", "LangGraph + PostgreSQL + vector store"],
                ),
                _technology_option(
                    "typescript",
                    "TypeScript",
                    "compatible",
                    "Adecuado si el equipo prioriza full-stack web, tipado y despliegues Node/Edge.",
                    ["Equipo con capacidad TypeScript.", "Framework agentico o adapters disponibles."],
                    ["Excelente para producto web; algunas capacidades RAG pueden requerir librerias adicionales."],
                    ["Next.js API + runtime agentico", "Node.js workers + PostgreSQL"],
                ),
            ],
            default_guidance="Si no existe restriccion corporativa, iniciar con Python para backend agentico y documentar adapters web si aplica.",
            source_ref="runtime_targets",
        ),
        AcpV2TechnologyDecision(
            decision_key="tech_framework",
            category="framework",
            question="Que framework o runtime agentico materializara workflows, tools y memoria?",
            selection_criteria=[
                "Nivel de orquestacion requerido.",
                "Necesidad de checkpoints/resume.",
                "Soporte para tools, handoffs, evaluaciones y observabilidad.",
            ],
            options=[
                _technology_option(
                    "codex_cli_workspace",
                    "Codex CLI sobre workspace",
                    "recommended",
                    "Muy util para iniciar construccion guiada y resolver decisiones sin acoplarse a un framework final.",
                    ["Codex CLI o herramienta equivalente instalada.", "Repositorio inicial disponible."],
                    ["Excelente para construccion; el runtime productivo debe seleccionarse si el agente se despliega."],
                    ["Ejecutar build_plan con Codex y generar adaptadores del runtime elegido."],
                ),
                _technology_option(
                    "openai_agents_sdk",
                    "OpenAI Agents SDK",
                    "compatible",
                    "Permite implementar agentes, tools y handoffs con contratos programaticos.",
                    ["Proveedor/modelos configurados.", "Gestion segura de secrets."],
                    ["Requiere disenar persistencia y despliegue del entorno productivo."],
                    ["Agentes por rol + tools con schemas + evaluaciones."],
                ),
                _technology_option(
                    "langgraph",
                    "LangGraph",
                    "compatible",
                    "Buena opcion si los workflows portables deben mapearse a grafo con checkpoints.",
                    ["Equipo familiarizado con grafos de estado.", "Persistencia para checkpoints."],
                    ["Mas control y robustez a cambio de mas complejidad."],
                    ["Mapear runtime_operational_workflow a nodos y transiciones."],
                ),
            ],
            default_guidance="Seleccionar el framework al ejecutar el ACP; no es requisito para validar ni descargar el paquete.",
            source_ref="workflows",
        ),
        AcpV2TechnologyDecision(
            decision_key="tech_database",
            category="database",
            question="Que base de datos persistira estado, auditoria, decisiones y configuracion?",
            selection_criteria=[
                "Politicas corporativas de datos.",
                "Volumen esperado de sesiones, checkpoints y eventos.",
                "Necesidad de transacciones, auditoria y multi-tenant.",
            ],
            options=[
                _technology_option(
                    "postgresql",
                    "PostgreSQL",
                    "recommended",
                    "Solida para estado transaccional, auditoria, multi-tenant y extensiones vectoriales si se requieren.",
                    ["Instancia administrada o self-hosted.", "Modelo de seguridad y backups definidos."],
                    ["Muy flexible; requiere operacion y tuning segun carga."],
                    ["PostgreSQL + pgvector", "PostgreSQL administrado + storage de objetos"],
                ),
                _technology_option(
                    "managed_document_db",
                    "Base documental administrada",
                    "compatible",
                    "Puede ser adecuada si el estado es principalmente documental y el equipo ya la opera.",
                    ["Definir consistencia, indices y estrategia de auditoria."],
                    ["Menos rigida que relacional; cuidado con reporting y relaciones complejas."],
                    ["MongoDB Atlas", "Cosmos DB"],
                ),
            ],
            default_guidance="Usar PostgreSQL salvo restriccion corporativa o runtime que provea persistencia equivalente.",
            source_ref="checkpoints",
        ),
        AcpV2TechnologyDecision(
            decision_key="tech_vector_store",
            category="vector_store",
            question="Que vector store usara la estrategia RAG si knowledge_contract esta habilitado?",
            required_for_implementation=needs_rag,
            selection_criteria=[
                "Volumen documental y frecuencia de refresh.",
                "Filtros por tenant, permisos y sensibilidad.",
                "Costo operativo, latencia y capacidades de metadata filtering.",
            ],
            options=[
                _technology_option(
                    "pgvector",
                    "pgvector",
                    "recommended" if needs_rag else "optional",
                    "Reduce componentes si PostgreSQL ya es la base transaccional y el volumen es moderado.",
                    ["PostgreSQL compatible.", "Politica de embeddings y dimensiones definida."],
                    ["Menos piezas de infraestructura; puede no ser ideal para volumen/vector search avanzado."],
                    ["PostgreSQL + pgvector por workspace/tenant"],
                ),
                _technology_option(
                    "managed_vector_db",
                    "Vector DB administrada",
                    "compatible",
                    "Conveniente cuando se requieren capacidades avanzadas de busqueda vectorial o escala dedicada.",
                    ["Proveedor aprobado.", "Modelo de seguridad, costos y lifecycle definido."],
                    ["Mas capacidades, pero mayor costo y componente adicional."],
                    ["Pinecone", "Weaviate", "Qdrant Cloud", "Azure AI Search"],
                ),
                _technology_option(
                    "none",
                    "Sin vector store",
                    "compatible" if not needs_rag else "not_recommended",
                    "Valido cuando el agente no requiere RAG ni busqueda semantica documental.",
                    ["Confirmar que memory_strategy no depende de retrieval vectorial."],
                    ["Reduce costo, pero limita grounding documental."],
                    ["Solo memoria transaccional y prompts"],
                ),
            ],
            default_guidance=(
                "Si RAG esta habilitado, seleccionar pgvector o vector DB administrada antes de implementar ingestion/retrieval. "
                "Si RAG no aplica, registrar explicitamente que no se requiere vector store."
            ),
            source_ref="knowledge_contract.v1",
        ),
        AcpV2TechnologyDecision(
            decision_key="tech_hosting",
            category="hosting",
            question="Donde se desplegaran API, workers, memoria y runtime agentico?",
            selection_criteria=[
                "Requisitos de seguridad, residencia de datos y compliance.",
                "Necesidad de workers asincronos, colas y escalabilidad.",
                "Capacidad operativa del equipo.",
            ],
            options=[
                _technology_option(
                    "managed_cloud",
                    "Cloud administrada",
                    "recommended",
                    "Reduce carga operativa y acelera despliegue con servicios administrados.",
                    ["Cuenta cloud aprobada.", "Networking, secrets y monitoreo definidos."],
                    ["Menor operacion directa, pero dependencia del proveedor cloud."],
                    ["Azure Container Apps", "AWS ECS/Fargate", "GCP Cloud Run"],
                ),
                _technology_option(
                    "customer_platform",
                    "Plataforma corporativa del cliente",
                    "compatible",
                    "Alineada a estandares internos cuando existen restricciones corporativas.",
                    ["Pipeline y plataforma disponibles.", "Owners de infraestructura asignados."],
                    ["Mayor control/compliance, pero mas dependencias organizacionales."],
                    ["Kubernetes corporativo", "OpenShift", "Plataforma interna"],
                ),
            ],
            default_guidance="Seleccionar hosting durante implementacion segun restricciones del cliente; el ACP solo entrega guia.",
            source_ref="deployment_guide",
        ),
        AcpV2TechnologyDecision(
            decision_key="tech_ci_cd",
            category="ci_cd",
            question="Que pipeline CI/CD validara, empacara y promovera el agente?",
            selection_criteria=[
                "Herramienta corporativa existente.",
                "Necesidad de gates de seguridad, pruebas y aprobaciones.",
                "Compatibilidad con entorno de despliegue seleccionado.",
            ],
            options=[
                _technology_option(
                    "github_actions",
                    "GitHub Actions",
                    "compatible",
                    "Adecuado si el repositorio vive en GitHub y se quieren gates automatizados.",
                    ["Repositorio GitHub.", "Secrets y environments configurados."],
                    ["Rapido de adoptar; depende de politicas del repositorio."],
                    ["Lint + tests + package + deploy"],
                ),
                _technology_option(
                    "azure_devops",
                    "Azure DevOps",
                    "compatible",
                    "Comun en organizaciones Microsoft y entornos enterprise.",
                    ["Proyecto Azure DevOps.", "Service connections aprobadas."],
                    ["Fuerte en enterprise; puede requerir configuracion adicional."],
                    ["Build pipeline + release gates"],
                ),
                _technology_option(
                    "customer_ci_cd",
                    "Pipeline corporativo existente",
                    "recommended",
                    "Minimiza friccion organizacional si ya hay una herramienta estandar.",
                    ["Owner DevOps asignado.", "Plantillas y politicas corporativas disponibles."],
                    ["Alineado con compliance; puede tardar mas en habilitarse."],
                    ["Jenkins", "GitLab CI", "Bitbucket Pipelines"],
                ),
            ],
            default_guidance="Reusar CI/CD corporativo cuando exista; si no, elegir pipeline simple con pruebas y gates de seguridad.",
            source_ref="deployment_guide",
        ),
        AcpV2TechnologyDecision(
            decision_key="tech_observability",
            category="observability",
            question="Como se observaran trazas, costo, latencia, calidad, errores y decisiones del agente?",
            selection_criteria=[
                "Necesidad de auditoria y trazabilidad de decisiones.",
                "Metricas de costo/latencia por etapa y herramienta.",
                "Integracion con monitoreo corporativo.",
            ],
            options=[
                _technology_option(
                    "opentelemetry",
                    "OpenTelemetry",
                    "recommended",
                    "Estandar portable para trazas y metricas sin atarse a un vendor.",
                    ["Collector/exporter definido.", "Campos sensibles redaccionados."],
                    ["Portable y extensible; requiere backend de observabilidad."],
                    ["OTel + Grafana/Tempo", "OTel + Azure Monitor"],
                ),
                _technology_option(
                    "vendor_apm",
                    "APM corporativo",
                    "compatible",
                    "Aprovecha plataforma existente de monitoreo y alertamiento.",
                    ["Licenciamiento y agentes aprobados.", "Dashboards y alertas definidos."],
                    ["Rapido si ya existe; menor portabilidad entre vendors."],
                    ["Datadog", "New Relic", "Dynatrace", "Azure Monitor"],
                ),
            ],
            default_guidance="Instrumentar al menos trazas, costos, latencia, errores, tool calls y decisiones humanas.",
            source_ref="conformance",
        ),
    ]


def _acp_v2_deployment_guide(technology_decisions: list[AcpV2TechnologyDecision]) -> AcpV2DeploymentGuide:
    deployment_refs = [
        decision.decision_key
        for decision in technology_decisions
        if decision.category in {"hosting", "ci_cd", "observability", "database", "vector_store"}
    ]
    return AcpV2DeploymentGuide(
        guide_key="deployment_guidance_v1",
        required_script=False,
        deployment_decision_refs=deployment_refs,
        environment_prerequisites=[
            "Runtime agentico seleccionado y documentado.",
            "Lenguaje/framework/base de datos/vector store/hosting/CI-CD/observabilidad resueltos o con waiver.",
            "Secrets, permisos y owners definidos fuera del ACP antes de conectar servicios reales.",
            "Estrategia de rollback, backup y auditoria aprobada por el equipo implementador.",
        ],
        steps=[
            AcpV2DeploymentGuideStep(
                step_key="deploy_guide_01",
                title="Resolver decisiones de entorno",
                objective="Cerrar las decisiones tecnologicas que condicionan implementacion y despliegue.",
                prerequisites=["technology_decisions disponible", "decision_registry disponible"],
                actions=[
                    "Seleccionar runtime, lenguaje, framework, base de datos, vector store, hosting, CI/CD y observabilidad.",
                    "Registrar justificacion y owners para cada decision.",
                ],
                validation=["No quedan decisiones requeridas para implementacion sin respuesta o waiver controlado."],
            ),
            AcpV2DeploymentGuideStep(
                step_key="deploy_guide_02",
                title="Preparar configuracion segura",
                objective="Configurar variables, secrets, permisos, redes y politicas sin incluir credenciales en el ACP.",
                prerequisites=["Hosting y CI/CD seleccionados", "Politica de secrets aprobada"],
                actions=[
                    "Crear secrets en el vault/plataforma elegida.",
                    "Definir permisos minimos para APIs, herramientas y fuentes de conocimiento.",
                    "Configurar redaccion de logs y proteccion de datos sensibles.",
                ],
                validation=["No hay secretos hardcodeados ni permisos excesivos en configuracion."],
            ),
            AcpV2DeploymentGuideStep(
                step_key="deploy_guide_03",
                title="Provisionar persistencia y memoria",
                objective="Materializar base de datos, almacenamiento, vector store y politicas RAG cuando apliquen.",
                prerequisites=["Database y vector store decididos", "Knowledge sources aprobadas"],
                actions=[
                    "Crear esquemas, indices, tenants y backups segun arquitectura seleccionada.",
                    "Configurar ingestion, embeddings, retrieval y refresh si RAG esta habilitado.",
                ],
                validation=["Checkpoints, memoria y fuentes RAG validan con datos de prueba no sensibles."],
            ),
            AcpV2DeploymentGuideStep(
                step_key="deploy_guide_04",
                title="Ejecutar pipeline de validacion",
                objective="Ejecutar pruebas, conformance y gates antes de promover.",
                prerequisites=["CI/CD seleccionado", "Tests y conformance disponibles"],
                actions=[
                    "Ejecutar lint, unit tests, integration tests, prompt evals y mutation/conformance gates.",
                    "Bloquear promocion si fallan reglas blocking.",
                ],
                validation=["La suite reporta resultados trazables y sin bloqueos criticos."],
            ),
            AcpV2DeploymentGuideStep(
                step_key="deploy_guide_05",
                title="Promover despliegue controlado",
                objective="Publicar el agente con monitoreo, rollback y aprobaciones segun politica del cliente.",
                prerequisites=["Observability configurada", "Rollback definido", "Owners disponibles"],
                actions=[
                    "Desplegar en ambiente controlado.",
                    "Monitorear latencia, costo, errores, tool calls, approval gates y calidad de respuestas.",
                    "Promover gradualmente segun resultados.",
                ],
                validation=["Dashboards y alertas estan operativos antes de trafico real."],
            ),
        ],
        rollback_guidance=[
            "Mantener version anterior del runtime o feature flag de desactivacion.",
            "Revertir solo cambios idempotentes automaticamente; side effects requieren plan de compensacion.",
            "Conservar snapshots/checkpoints suficientes para diagnostico y recuperacion.",
        ],
        security_considerations=[
            "No almacenar credenciales en el ACP.",
            "Aplicar minimo privilegio en herramientas y fuentes de conocimiento.",
            "Redactar PII/secrets en logs, prompts y trazas.",
            "Validar aislamiento tenant/workspace en persistencia y retrieval.",
        ],
        observability_considerations=[
            "Trazar cada llamada LLM con input hash, output hash, tokens, latencia, costo y decision.",
            "Registrar tool calls, retries, fallos, approvals y compensaciones.",
            "Monitorear drift de calidad, freshness de conocimiento y uso de ventana de contexto.",
        ],
    )


def _acp_v2_conformance_rules(construction_pack: ConstructionPackV1) -> list[AcpV2ConformanceRule]:
    rules = [
        AcpV2ConformanceRule(
            rule_key="schema-valid-agent-construction-package-v2",
            severity="blocking",
            requirement="El paquete debe validar contra agent-construction-package.v2.schema.json.",
            validation_method="json_schema_validation",
        ),
        AcpV2ConformanceRule(
            rule_key="no-required-lean-runtime",
            severity="blocking",
            requirement="La especificacion, el plan de construccion y el runtime no deben requerir endpoints, estados o servicios internos de Lean Agent Builder.",
            validation_method="static_scan_for_internal_runtime_dependencies",
        ),
        AcpV2ConformanceRule(
            rule_key="checksums-present",
            severity="blocking",
            requirement="Cada contrato fuente incluido en el manifest debe declarar checksum SHA-256.",
            validation_method="manifest_checksum_presence",
        ),
        AcpV2ConformanceRule(
            rule_key="human-decisions-structured",
            severity="blocking",
            requirement="Toda decision pendiente debe incluir pregunta, owner, momento recomendado, opciones, ejemplos e impacto.",
            validation_method="decision_registry_required_fields",
        ),
        AcpV2ConformanceRule(
            rule_key="portable-workflows-declared",
            severity="blocking",
            requirement="El ACP debe separar workflow de construccion, runtime operativo y resolucion de decisiones humanas.",
            validation_method="workflow_contract_presence",
        ),
        AcpV2ConformanceRule(
            rule_key="portable-checkpoints-no-internal-ids",
            severity="blocking",
            requirement="Los checkpoints deben ser puntos de control portables y no IDs de ejecucion internos del productor.",
            validation_method="checkpoint_key_static_scan",
        ),
        AcpV2ConformanceRule(
            rule_key="deferred-decisions-do-not-block-package",
            severity="blocking",
            requirement="Las decisiones diferibles no bloquean la existencia del paquete, solo su resolucion posterior en implementacion.",
            validation_method="decision_registry_blocking_scope_scan",
        ),
        AcpV2ConformanceRule(
            rule_key="runtime-recommendations-not-required",
            severity="blocking",
            requirement="El ACP debe separar recommended_runtime de required_runtime y no imponer runtime unico por defecto.",
            validation_method="runtime_target_policy_scan",
        ),
        AcpV2ConformanceRule(
            rule_key="technology-decisions-guided",
            severity="blocking",
            requirement="Lenguaje, framework, database, vector store, hosting, CI/CD y observabilidad deben ser decisiones guiadas con criterios, tradeoffs, prerequisitos y ejemplos.",
            validation_method="technology_decisions_category_coverage",
        ),
        AcpV2ConformanceRule(
            rule_key="deployment-guidance-only",
            severity="blocking",
            requirement="La guia de deployment debe ser guidance_only y no un script obligatorio.",
            validation_method="deployment_guide_mode_scan",
        ),
        AcpV2ConformanceRule(
            rule_key="abstract-capabilities-declared",
            severity="blocking",
            requirement="Cada herramienta aprobada debe mapearse a una capacidad abstracta reemplazable.",
            validation_method="capability_catalog_coverage",
        ),
        AcpV2ConformanceRule(
            rule_key="tools-boundary-classified",
            severity="blocking",
            requirement="Cada binding de herramienta debe clasificar si es interno del productor, API externa, adapter runtime o contrato abstracto.",
            validation_method="tool_bindings_provider_boundary_scan",
        ),
        AcpV2ConformanceRule(
            rule_key="tool-risks-costs-fallbacks-declared",
            severity="blocking",
            requirement="Cada tool binding debe declarar permisos, credenciales, side effects, costo, riesgo y fallback.",
            validation_method="tool_binding_operational_fields",
        ),
        AcpV2ConformanceRule(
            rule_key="no-producer-internal-required-tools",
            severity="blocking",
            requirement="Una herramienta interna del productor no puede ser requisito portable del ACP.",
            validation_method="producer_internal_tool_required_scan",
        ),
        AcpV2ConformanceRule(
            rule_key="side-effects-governed",
            severity="blocking" if any(tool.side_effects for tool in construction_pack.tool_contracts) else "info",
            requirement="Las herramientas con side effects deben declarar aprobacion, idempotencia, retry o compensacion.",
            validation_method="tool_contract_side_effect_policy",
        ),
        AcpV2ConformanceRule(
            rule_key="memory-retrieval-declared",
            severity="blocking" if construction_pack.knowledge_contract.enabled and not construction_pack.knowledge_contract.sources else "info",
            requirement="Si RAG o memoria larga estan habilitados, las fuentes y politicas de recuperacion deben estar declaradas.",
            validation_method="memory_knowledge_contract_scan",
        ),
        AcpV2ConformanceRule(
            rule_key="portable-memory-namespaces",
            severity="blocking",
            requirement="La memoria debe declararse con namespaces portables, sin sessionId, projectId ni servicios internos de Lean.",
            validation_method="memory_namespace_portability_scan",
        ),
        AcpV2ConformanceRule(
            rule_key="rag-capability-dependencies",
            severity="blocking" if construction_pack.knowledge_contract.enabled else "info",
            requirement="Cuando RAG esta habilitado, document_ingestion, embedding, vector_search y knowledge_retrieval deben existir como capacidades abstractas.",
            validation_method="rag_capability_dependency_coverage",
        ),
        AcpV2ConformanceRule(
            rule_key="knowledge-artifacts-not-internal-paths",
            severity="blocking",
            requirement="Las fuentes de conocimiento deben referenciarse por artefacto portable y no por rutas internas locales del productor.",
            validation_method="knowledge_artifact_location_scan",
        ),
        AcpV2ConformanceRule(
            rule_key="context-window-anti-redundancy",
            severity="blocking",
            requirement="La politica de contexto corto debe impedir reenviar informacion redundante cuando existe resumen, pagina o artifact_ref.",
            validation_method="context_window_policy_scan",
        ),
    ]
    return rules


def _acp_v2_compatibility_rules() -> list[AcpV2CompatibilityRule]:
    return [
        AcpV2CompatibilityRule(
            target="codex-cli",
            support_level="supported_with_adapter",
            adapter_notes=[
                "Usar build_plan como checklist de ejecucion.",
                "Cargar prompts y tool_contracts como archivos locales o instrucciones del workspace.",
            ],
            unsupported_features=[],
        ),
        AcpV2CompatibilityRule(
            target="cursor",
            support_level="supported_with_manual_mapping",
            adapter_notes=[
                "Mapear prompts a reglas/profiles del proyecto.",
                "Usar tests y conformance como checklist antes de aplicar cambios.",
            ],
            unsupported_features=[],
        ),
        AcpV2CompatibilityRule(
            target="claude-code",
            support_level="supported_with_manual_mapping",
            adapter_notes=[
                "Cargar ACP como contexto local y ejecutar build_plan por etapas.",
                "Responder implementation_decisions antes de generar codigo dependiente del entorno.",
            ],
            unsupported_features=[],
        ),
        AcpV2CompatibilityRule(
            target="github-copilot",
            support_level="supported_as_reference_package",
            adapter_notes=[
                "Usar los artefactos como especificacion y prompts de implementacion.",
            ],
            unsupported_features=["No interpreta automaticamente state_machine ni conformance sin tooling adicional."],
        ),
        AcpV2CompatibilityRule(
            target="openai-agents-sdk",
            support_level="supported_with_runtime_adapter",
            adapter_notes=[
                "Mapear agent_runtime.agents a agentes SDK.",
                "Mapear tool_contracts a tools con schemas y approval gates.",
            ],
            unsupported_features=[],
        ),
        AcpV2CompatibilityRule(
            target="langgraph",
            support_level="supported_with_runtime_adapter",
            adapter_notes=[
                "Mapear state_machine y routing_rules a nodos, edges y condiciones.",
                "Persistir memory_strategy segun storage elegido por el usuario.",
            ],
            unsupported_features=[],
        ),
    ]


def build_agent_construction_package_v2(
    snapshot: SessionSnapshot,
    *,
    blueprint_core: BlueprintCoreV1 | None = None,
    construction_pack: ConstructionPackV1 | None = None,
    generated_at=None,
) -> AgentConstructionPackageV2:
    generated_at = generated_at or utc_now()
    construction_pack = construction_pack or build_construction_pack(snapshot, generated_at=generated_at)
    blueprint_core = blueprint_core or build_blueprint_core(snapshot, generated_at=generated_at)

    source_contracts = {
        "blueprint-core.v1": blueprint_core,
        "construction-pack.v1": construction_pack,
        "prompt-pack.v1": construction_pack.prompt_pack,
        "evaluation-pack.v1": construction_pack.evaluation_pack,
        "memory-policy.v1": construction_pack.memory_policy,
        "knowledge-contract.v1": construction_pack.knowledge_contract,
        "tool-contract.v1": [tool.model_dump(mode="json") for tool in construction_pack.tool_contracts],
    }
    manifest_entries = [
        _contract_entry("blueprint-core.v1", "contracts/blueprint-core.v1.json", source_contracts["blueprint-core.v1"]),
        _contract_entry("construction-pack.v1", "contracts/construction-pack.v1.json", source_contracts["construction-pack.v1"]),
        _contract_entry("prompt-pack.v1", "contracts/prompt-pack.v1.json", source_contracts["prompt-pack.v1"]),
        _contract_entry("evaluation-pack.v1", "contracts/evaluation-pack.v1.json", source_contracts["evaluation-pack.v1"]),
        _contract_entry("memory-policy.v1", "contracts/memory-policy.v1.json", source_contracts["memory-policy.v1"]),
        _contract_entry("knowledge-contract.v1", "contracts/knowledge-contract.v1.json", source_contracts["knowledge-contract.v1"]),
        _contract_entry("tool-contract.v1", "contracts/tool-contracts.v1.json", source_contracts["tool-contract.v1"], required=False),
    ]
    compatibility = _acp_v2_compatibility_rules()
    compatibility_targets = [item.target for item in compatibility]
    construction_payload = construction_pack.model_dump(mode="json")
    build_plan = _acp_v2_build_plan()
    agent_runtime = _acp_v2_runtime(construction_pack)
    runtime_targets = _acp_v2_runtime_targets(agent_runtime)
    runtime_target_policy = _acp_v2_runtime_target_policy(runtime_targets)
    technology_decisions = _acp_v2_technology_decisions(construction_pack)
    deployment_guide = _acp_v2_deployment_guide(technology_decisions)
    implementation_decisions = _acp_v2_implementation_decisions(blueprint_core, construction_pack)
    decision_registry = _acp_v2_decision_registry(implementation_decisions, technology_decisions)
    checkpoints = _acp_v2_checkpoints(build_plan, agent_runtime, decision_registry)
    workflows = _acp_v2_workflows(build_plan, agent_runtime, decision_registry, checkpoints)
    tool_contract_refs = _acp_v2_tool_contracts(construction_pack)
    tool_bindings = _acp_v2_tool_bindings(construction_pack)
    capability_catalog = _acp_v2_capability_catalog(construction_pack, tool_bindings)
    tool_analysis = _acp_v2_tool_analysis(construction_pack, capability_catalog, tool_bindings)
    memory_knowledge_plan = _acp_v2_memory_knowledge_plan(construction_pack)

    return AgentConstructionPackageV2(
        **_base_metadata(snapshot, generated_at),
        producer_metadata=AcpV2ProducerMetadata(
            producer_name="Lean Agent Builder",
            generated_from_contracts=list(source_contracts.keys()),
            notes=[
                "La metadata del productor es solo trazabilidad; no debe ser requisito para ejecutar el ACP.",
                "La especificacion del sistema agentico vive en system_specification, build_plan y agent_runtime.",
            ],
        ),
        portable_manifest=AcpV2PortableManifest(
            package_id=f"acp-v2-{snapshot.session.id}",
            created_at=generated_at,
            compatibility_targets=compatibility_targets,
            contracts=manifest_entries,
        ),
        migration=AcpV2MigrationInfo(
            from_schema_version="construction-pack.v1",
            source_checksum_sha256=_canonical_payload_checksum(construction_payload),
            migration_strategy=(
                "Proyeccion declarativa desde construction-pack.v1; conserva diseno, prompts, tools, memoria, "
                "tests y decisiones, pero elimina dependencias obligatorias del motor productor."
            ),
            breaking_changes=[],
            compatibility_notes=[
                "Los consumidores externos deben usar agent_runtime y build_plan como contrato principal.",
                "source_session_id se conserva solo como referencia de auditoria del productor.",
            ],
        ),
        system_specification={
            "identity": blueprint_core.identity.model_dump(mode="json"),
            "purpose": blueprint_core.purpose.model_dump(mode="json"),
            "scope": blueprint_core.scope.model_dump(mode="json"),
            "architecture": construction_pack.topology.get("architecture", ""),
            "reasoning_pattern": construction_pack.topology.get("reasoning_pattern", ""),
            "workflow_template": construction_pack.topology.get("workflow_template", ""),
            "guardrails": blueprint_core.guardrails,
            "approvals": [approval.model_dump(mode="json") for approval in blueprint_core.approvals],
            "success_criteria": [criterion.model_dump(mode="json") for criterion in blueprint_core.success_criteria],
            "risks": [risk.model_dump(mode="json") for risk in blueprint_core.risks],
            "assumptions": blueprint_core.assumptions,
        },
        build_plan=build_plan,
        agent_runtime=agent_runtime,
        implementation_decisions=implementation_decisions,
        workflows=workflows,
        checkpoints=checkpoints,
        decision_registry=decision_registry,
        runtime_target_policy=runtime_target_policy,
        runtime_targets=runtime_targets,
        technology_decisions=technology_decisions,
        deployment_guide=deployment_guide,
        capability_catalog=capability_catalog,
        tool_contracts=tool_contract_refs,
        tool_bindings=tool_bindings,
        tool_analysis=tool_analysis,
        memory_strategy=_acp_v2_memory_strategy(construction_pack),
        memory_knowledge_plan=memory_knowledge_plan,
        knowledge_sources=_acp_v2_knowledge_sources(construction_pack),
        prompts=_acp_v2_prompts(construction_pack),
        tests=_acp_v2_tests(construction_pack),
        conformance=_acp_v2_conformance_rules(construction_pack),
        compatibility=compatibility,
        provenance=_provenance(
            (
                "system_specification",
                ["blueprint-core.v1", "construction-pack.v1"],
                "El ACP v2 separa la especificacion agentica del metadata del productor.",
            ),
            (
                "agent_runtime",
                ["behavior-spec.v1"],
                "La maquina de estados y reglas de enrutamiento se expresan de forma declarativa y portable.",
            ),
            (
                "workflows",
                ["build_plan", "agent_runtime", "decision_registry"],
                "Los workflows separan construccion, runtime operativo y resolucion HITL sin depender de etapas Lean.",
            ),
            (
                "checkpoints",
                ["build_plan", "agent_runtime", "decision_registry"],
                "Los checkpoints se expresan como puntos de control portables y no como IDs de ejecucion internos.",
            ),
            (
                "implementation_decisions",
                ["blueprint-core.v1:open_questions", "construction-pack.v1:gaps"],
                "Las preguntas pendientes quedan estructuradas para resolverse durante implementacion.",
            ),
            (
                "decision_registry",
                ["implementation_decisions"],
                "Las decisiones humanas quedan clasificadas con impacto, opciones, ejemplos y momento recomendado.",
            ),
            (
                "runtime_target_policy",
                ["compatibility", "agent_runtime"],
                "El ACP separa runtimes recomendados de runtimes requeridos para no imponer stack unico.",
            ),
            (
                "technology_decisions",
                ["agent_runtime", "memory_strategy", "knowledge-contract.v1", "tool-contract.v1"],
                "Lenguaje, framework, database, vector store, hosting, CI/CD y observabilidad se resuelven como decisiones guiadas.",
            ),
            (
                "deployment_guide",
                ["technology_decisions", "conformance"],
                "El despliegue queda documentado como guia portable y no como script obligatorio.",
            ),
            (
                "capability_catalog",
                ["tool-contract.v1", "memory-policy.v1", "knowledge-contract.v1"],
                "Las herramientas concretas se normalizan como capacidades abstractas reemplazables.",
            ),
            (
                "tool_bindings",
                ["tool-contract.v1"],
                "Cada herramienta se clasifica por boundary, reemplazabilidad, credenciales, costo, riesgo y fallback.",
            ),
            (
                "tool_analysis",
                ["capability_catalog", "tool_bindings"],
                "El ACP detecta redundancias, incompatibilidades y herramientas no recomendadas antes de implementar.",
            ),
            (
                "memory_knowledge_plan",
                ["memory-policy.v1", "knowledge-contract.v1", "capability_catalog", "technology_decisions"],
                "Memoria corta, memoria larga, conocimiento documental y RAG se publican como contrato portable por namespaces y artefactos.",
            ),
        ),
    )


def build_estimation_pack(snapshot: SessionSnapshot, *, generated_at=None) -> EstimationPackV1:
    generated_at = generated_at or utc_now()
    estimation = snapshot.estimation_report
    if estimation is None:
        raise ValueError("Session snapshot does not contain an estimation_report artifact")
    bound_blueprint_version = estimation.blueprint_version_number or latest_blueprint_version(snapshot)

    saved_cost = estimation.traditional.estimated_cost - estimation.agentic.estimated_cost
    base_cost = estimation.traditional.estimated_cost or 1
    roi_percent = round((saved_cost / base_cost) * 100, 2)
    sensitivity_drivers = [
        EstimationSensitivityDriver(
            key=f"positive_{index + 1}",
            summary=signal,
            impact="positive",
        )
        for index, signal in enumerate(estimation.confidence.positive_signals)
    ]
    sensitivity_drivers.extend(
        EstimationSensitivityDriver(
            key=f"negative_{index + 1}",
            summary=signal,
            impact="negative",
        )
        for index, signal in enumerate(estimation.confidence.negative_signals)
    )

    return EstimationPackV1(
        **_base_metadata(snapshot, generated_at, source_blueprint_version=bound_blueprint_version),
        blueprint_ref=ContractReference(
            contract_kind="blueprint-core",
            schema_version="blueprint-core.v1",
            source_blueprint_version=bound_blueprint_version,
        ),
        maturity_stage=str(estimation.maturity_stage),
        traditional=estimation.traditional,
        agentic=estimation.agentic,
        confidence=estimation.confidence,
        base_confidence=estimation.base_confidence,
        analysis=estimation.analysis,
        deterministic_inputs=estimation.deterministic_inputs,
        assumptions=_normalized_items(estimation.assumptions),
        risk_drivers=_normalized_items(estimation.risk_drivers),
        sensitivity_drivers=sensitivity_drivers,
        roi_summary=(
            f"Ahorro estimado de {saved_cost:.2f} con un ROI relativo de {roi_percent:.2f}% frente al escenario tradicional."
        ),
        estimation_runs=list(snapshot.estimation_runs),
        actuals_count=len(snapshot.project_actuals),
        provenance=_provenance(
            (
                "traditional",
                ["estimation_report.traditional"],
                "El escenario tradicional se conserva como fuente de verdad economica del snapshot.",
            ),
            (
                "agentic",
                ["estimation_report.agentic"],
                "El escenario agentic preserva proveedor, cobertura y costos estimados ya calculados.",
            ),
            (
                "confidence",
                ["estimation_report.confidence", "estimation_runs"],
                "La confianza y sus señales siguen saliendo del motor de estimacion actual.",
            ),
        ),
    )


def _iter_prompt_artifacts(prompt_pack: PromptPackV1) -> list[PromptArtifactV1]:
    return [
        item
        for item in [
            prompt_pack.system_prompt,
            prompt_pack.planner_prompt,
            prompt_pack.executor_prompt,
            prompt_pack.evaluator_prompt,
            prompt_pack.tool_use_prompt,
            prompt_pack.memory_prompt,
            prompt_pack.retrieval_prompt,
            prompt_pack.recovery_prompt,
            *prompt_pack.agent_role_prompts,
            *prompt_pack.handoff_prompts,
        ]
        if item is not None
    ]


def _first_prompt_signal(content: str) -> str:
    for line in content.splitlines():
        normalized = line.strip()
        if normalized:
            return normalized[:120]
    return content.strip()[:120]


def _required_fields_for(schema_version: str) -> list[str]:
    from app.contracts import CANONICAL_CONTRACT_MODELS

    model = CANONICAL_CONTRACT_MODELS[schema_version]
    return sorted(str(item) for item in model.model_json_schema().get("required", []))


def _build_test_pack_mutation_cases() -> list[TestPackMutationCaseV1]:
    cases: list[TestPackMutationCaseV1] = []
    for contract_key in (
        "blueprint-core.v1",
        "construction-pack.v1",
        "prompt-pack.v1",
        "estimation-pack.v1",
        "evaluation-pack.v1",
        "test-pack.v1",
    ):
        for field in _required_fields_for(contract_key):
            normalized_field = field.replace(".", "_").replace("-", "_")
            cases.append(
                TestPackMutationCaseV1(
                    key=f"{contract_key}:{normalized_field}:missing",
                    contract_key=contract_key,
                    path=field,
                    mutation="remove_field",
                    expected_issue_code="missing",
                    expected_issue_path=field,
                    blocks_readiness=True,
                )
            )
    return cases


def _build_prompt_evaluation_cases(prompt_pack: PromptPackV1) -> list[TestPackPromptEvaluationCaseV1]:
    cases: list[TestPackPromptEvaluationCaseV1] = []
    forbidden_by_prompt = {
        "system": ["ignore previous instructions", "bypass approval"],
        "planner": ["skip plan", "ignore constraints"],
        "executor": ["bypass approval", "ignore validations"],
        "evaluator": ["ignore blocking issues", "force ready"],
        "tool_use": ["execute destructive action without approval", "skip typed errors"],
        "memory": ["ignore goal drift", "persist secrets"],
        "retrieval": ["answer without evidence", "drop citations"],
        "recovery": ["pretend issue resolved", "ignore failure context"],
    }
    for artifact in _iter_prompt_artifacts(prompt_pack):
        prompt_signal = _first_prompt_signal(artifact.content)
        cases.append(
            TestPackPromptEvaluationCaseV1(
                key=f"{artifact.prompt_key}_positive",
                prompt_key=artifact.prompt_key,
                mode="positive",
                expected_substrings=[prompt_signal] if prompt_signal else [],
                forbidden_substrings=[],
                measurable_criterion="El prompt conserva su instruccion principal y permanece trazable en el pack.",
                blocking=True,
            )
        )
        cases.append(
            TestPackPromptEvaluationCaseV1(
                key=f"{artifact.prompt_key}_failure",
                prompt_key=artifact.prompt_key,
                mode="failure",
                failure_mode=(
                    "injection"
                    if artifact.prompt_key in {"system", "recovery"}
                    else "contradiction"
                    if artifact.prompt_key in {"tool_use", "memory", "retrieval"}
                    else "omission"
                ),
                expected_substrings=[],
                forbidden_substrings=forbidden_by_prompt.get(artifact.prompt_key, ["ignore previous instructions"]),
                measurable_criterion="El prompt no contiene bypasses obvios ni omite sus controles criticos.",
                blocking=artifact.prompt_key in {"system", "planner", "executor", "evaluator", "recovery"},
            )
        )
    return cases


def _build_recovery_cases(
    prompt_pack: PromptPackV1,
    knowledge_contract: KnowledgeContractV1,
) -> list[TestPackRecoveryCaseV1]:
    cases = [
        TestPackRecoveryCaseV1(
            key="tool_failure_recovery",
            trigger="tool_failure",
            expected_prompt_key="recovery" if prompt_pack.recovery_prompt is not None else "executor",
            expected_behavior="El pack debe preservar fallback o prompt de recovery para fallos de tools.",
            measurable_criterion="Existe `recovery_prompt` o el executor define fallback no vacio.",
        ),
        TestPackRecoveryCaseV1(
            key="llm_timeout_recovery",
            trigger="llm_timeout",
            expected_prompt_key="recovery",
            expected_behavior="El pack debe ofrecer un camino de escalamiento ante timeout o patron no soportado.",
            measurable_criterion="Existe `recovery_prompt` y `llm_policy.fallback_model` no esta vacio.",
        ),
    ]
    if prompt_pack.retrieval_prompt is not None or knowledge_contract.mode == "rag":
        cases.append(
            TestPackRecoveryCaseV1(
                key="retrieval_no_evidence_recovery",
                trigger="retrieval_no_evidence",
                expected_prompt_key="retrieval",
                expected_behavior="El retrieval debe declarar falta de evidencia y escalar segun fallback aprobado.",
                measurable_criterion="Existe `retrieval_prompt` y `knowledge_contract.grounding_policy.no_evidence_behavior` no esta vacio.",
            )
        )
    return cases


def _build_acceptance_journeys(evaluation_pack: EvaluationPackV1) -> list[TestPackAcceptanceJourneyV1]:
    return [
        TestPackAcceptanceJourneyV1(
            key=item.key,
            title=item.title,
            input_reference=item.scenario,
            expected_behavior=item.expected_result,
            measurable_criterion=f"Debe existir acceptance case activo con key={item.key}.",
        )
        for item in evaluation_pack.acceptance_cases
    ]


def _stable_issue_catalog_entries() -> list[StableIssueCatalogEntryV1]:
    entries = [
        StableIssueCatalogEntryV1(
            code=code,
            kind="validation_issue",
            severity=metadata.get("severity", "warning"),
            remediation=metadata.get("remediation", ""),
        )
        for code, metadata in sorted(VALIDATION_ISSUE_CATALOG.items())
    ]
    entries.extend(
        StableIssueCatalogEntryV1(
            code=code,
            kind="construction_gap",
            severity=metadata.get("severity", "warning"),
            remediation=metadata.get("remediation", ""),
        )
        for code, metadata in sorted(CONSTRUCTION_GAP_CATALOG.items())
    )
    return entries


def build_test_pack(
    snapshot: SessionSnapshot,
    *,
    construction_pack: ConstructionPackV1,
    estimation_pack: EstimationPackV1,
    generated_at=None,
) -> TestPackV1:
    generated_at = generated_at or utc_now()
    fixtures = [
        TestPackFixtureRef(
            key="blueprint_core",
            contract_key="blueprint-core.v1",
            relative_path="contracts/blueprint-core.v1.json",
            valid=True,
            summary="Contrato base para consumidores externos.",
        ),
        TestPackFixtureRef(
            key="construction_pack",
            contract_key="construction-pack.v1",
            relative_path="contracts/construction-pack.v1.json",
            valid=True,
            summary="Paquete de construccion listo para validaciones de consumer.",
        ),
        TestPackFixtureRef(
            key="prompt_pack",
            contract_key="prompt-pack.v1",
            relative_path="contracts/prompt-pack.v1.json",
            valid=True,
            summary="Prompt pack por rol para evaluacion de cobertura critica.",
        ),
        TestPackFixtureRef(
            key="evaluation_pack",
            contract_key="evaluation-pack.v1",
            relative_path="contracts/evaluation-pack.v1.json",
            valid=True,
            summary="Acceptance cases y readouts de evaluacion para journeys ejecutables.",
        ),
        TestPackFixtureRef(
            key="estimation_pack",
            contract_key="estimation-pack.v1",
            relative_path="contracts/estimation-pack.v1.json",
            valid=True,
            summary="Pack de estimacion ligado al blueprint exacto.",
        ),
        TestPackFixtureRef(
            key="test_pack",
            contract_key="test-pack.v1",
            relative_path="contracts/test-pack.v1.json",
            valid=True,
            summary="Contrato de pruebas ejecutables y consumidor externo.",
        ),
    ]
    mutation_cases = _build_test_pack_mutation_cases()
    invalid_fixtures = [
        TestPackFixtureRef(
            key=f"invalid_{case.key}",
            contract_key=case.contract_key,
            relative_path=f"contracts/invalid/{case.contract_key}.{case.path.replace('.', '_').replace('-', '_')}.missing.json",
            valid=False,
            summary=f"Fixture invalido generado removiendo `{case.path}`.",
        )
        for case in mutation_cases
    ]
    prompt_evaluation_cases = _build_prompt_evaluation_cases(construction_pack.prompt_pack)
    recovery_cases = _build_recovery_cases(construction_pack.prompt_pack, construction_pack.knowledge_contract)
    acceptance_journeys = _build_acceptance_journeys(construction_pack.evaluation_pack)

    return TestPackV1(
        **_base_metadata(snapshot, generated_at),
        blueprint_ref=ContractReference(
            contract_kind="blueprint-core",
            schema_version="blueprint-core.v1",
            source_blueprint_version=latest_blueprint_version(snapshot),
        ),
        framework_target="python-stdlib-external-consumer",
        fixtures=fixtures,
        invalid_fixtures=invalid_fixtures,
        commands=[
            TestPackCommandV1(
                key="schema_contract_validation",
                title="Schema and required-field validation",
                kind="schema_validation",
                command="python consumers/python/reference_consumer.py --pack contracts/test-pack.v1.json --contracts contracts --mode schema",
            ),
            TestPackCommandV1(
                key="mutation_gate",
                title="Mutation gate for required fields",
                kind="mutation",
                command="python consumers/python/reference_consumer.py --pack contracts/test-pack.v1.json --contracts contracts --mode mutations",
            ),
            TestPackCommandV1(
                key="prompt_and_recovery_coverage",
                title="Prompt, recovery and acceptance coverage",
                kind="prompt_recovery",
                command="python consumers/python/reference_consumer.py --pack contracts/test-pack.v1.json --contracts contracts --mode prompts",
            ),
            TestPackCommandV1(
                key="external_consumer_full",
                title="Full external consumer run",
                kind="external_consumer",
                command="python consumers/python/reference_consumer.py --pack contracts/test-pack.v1.json --contracts contracts --mode full",
            ),
        ],
        mutation_cases=mutation_cases,
        prompt_evaluation_cases=prompt_evaluation_cases,
        recovery_cases=recovery_cases,
        acceptance_journeys=acceptance_journeys,
        stable_issue_catalog=_stable_issue_catalog_entries(),
        external_consumer=TestPackExternalConsumerV1(
            relative_path="consumers/python/reference_consumer.py",
            entry_command="python consumers/python/reference_consumer.py --pack contracts/test-pack.v1.json --contracts contracts --mode full",
            constraints=[
                "No debe importar modulos internos del builder.",
                "Debe ejecutarse desde un directorio limpio con solo JSON y un script Python estandar.",
                "Debe bloquear cuando falle una mutacion blocking o falte cobertura critica de prompts.",
            ],
        ),
        provenance=_provenance(
            (
                "mutation_cases",
                ["shared_specs/schemas", "contracts/*.json"],
                "Los casos de mutacion salen de campos requeridos del contrato canonico actual.",
            ),
            (
                "prompt_evaluation_cases",
                ["prompt-pack.v1", "evaluation-pack.v1"],
                "La cobertura de prompts se deriva del prompt pack activo y sus acceptance cases.",
            ),
            (
                "acceptance_journeys",
                ["evaluation-pack.v1"],
                "Los journeys ejecutables heredan los acceptance cases activos del evaluation pack.",
            ),
        ),
    )


def build_contract_bundle(snapshot: SessionSnapshot, *, generated_at=None) -> dict[str, Any]:
    generated_at = generated_at or utc_now()
    blueprint_core = build_blueprint_core(snapshot, generated_at=generated_at)
    construction_pack = build_construction_pack(snapshot, generated_at=generated_at)
    agent_construction_package_v2 = build_agent_construction_package_v2(
        snapshot,
        blueprint_core=blueprint_core,
        construction_pack=construction_pack,
        generated_at=generated_at,
    )
    estimation_pack = build_estimation_pack(snapshot, generated_at=generated_at)
    short_term_memory = build_short_term_memory(snapshot, generated_at=generated_at)
    knowledge_manifest = build_knowledge_manifest(snapshot, generated_at=generated_at)
    test_pack = build_test_pack(
        snapshot,
        construction_pack=construction_pack,
        estimation_pack=estimation_pack,
        generated_at=generated_at,
    )

    specialized = {
        "behavior-spec.v1": construction_pack.behavior_spec,
        "tool-contract.v1": construction_pack.tool_contracts,
        "heuristic-decision.v1": construction_pack.heuristic_decision,
        "llm-policy.v1": construction_pack.llm_policy,
        "memory-policy.v1": construction_pack.memory_policy,
        "short-term-memory.v1": short_term_memory,
        "knowledge-contract.v1": construction_pack.knowledge_contract,
        "knowledge-manifest.v1": knowledge_manifest,
        "evaluation-pack.v1": construction_pack.evaluation_pack,
        "test-pack.v1": test_pack,
    }

    return {
        "blueprint-core.v1": blueprint_core,
        "construction-pack.v1": construction_pack,
        "agent-construction-package.v2": agent_construction_package_v2,
        "prompt-pack.v1": construction_pack.prompt_pack,
        "estimation-pack.v1": estimation_pack,
        **specialized,
    }
