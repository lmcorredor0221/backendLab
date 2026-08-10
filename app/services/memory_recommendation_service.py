from __future__ import annotations

from typing import Any
from uuid import UUID

from app.models import (
    ApprovedToolsDigest,
    BlueprintArtifact,
    CanvasArtifact,
    DesignRecommendationArtifact,
    DiscoveryArtifact,
    GuidedAnswerOptionEntry,
    GuidedQuestionEntry,
    KnowledgeProfile,
    KnowledgeSource,
    MemoryContextBudgetEntry,
    MemoryDryCompileStatus,
    MemoryKnowledgeDesign,
    MemoryLayerDesign,
    MemoryNeedDecision,
    MemoryProfile,
    MemoryRecommendationArtifact,
    MemoryRecommendationConfidence,
    MemoryRecommendationFinding,
    MemoryRecommendationSourceStageVersions,
    MemoryRetentionDeletionRule,
    MemorySensitivityIsolationRule,
    MemoryToolDependency,
    MemoryWriteReadRule,
    ReviewState,
    SessionSnapshot,
    utc_now,
)
from app.services.canonical_exports import (
    build_knowledge_contract,
    build_memory_policy,
    build_short_term_memory,
    build_tool_contracts,
)
from app.services.knowledge_tool_policy import build_memory_tool_dependencies
from app.services.llm_runtime.builder_contracts import (
    MemoryArchitectureCritiqueOutput,
    MemoryArchitectureRecommendationOutput,
    PrioritizedQuestion,
    RequirementsDefinitionOutput,
)
from app.services.rules import derive_knowledge_profile, derive_memory_profile, select_memory_strategy
from app.services.stage4_compiler import compile_stage4_artifacts


def _guided_question_from_prioritized(question: PrioritizedQuestion, *, stage_scope: str) -> GuidedQuestionEntry:
    return GuidedQuestionEntry(
        key=question.key,
        question=question.question,
        rationale=question.rationale,
        priority=question.priority,
        blocking_stages=list(question.blocking_stages),
        suggested_answer=question.suggested_answer,
        answer_options=[
            GuidedAnswerOptionEntry(
                key=option.key,
                label=option.label,
                description=option.description,
                impact=option.impact,
                example=option.example,
                recommended=option.recommended,
                confidence=option.confidence,
                source_refs=list(option.source_refs),
            )
            for option in question.answer_options
        ],
        stage_scope=stage_scope,
        confidence=max([option.confidence for option in question.answer_options] or [0.0]),
    )


def _coalesce(*values: str | None, fallback: str = "") -> str:
    for value in values:
        token = str(value or "").strip()
        if token:
            return token
    return fallback


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for raw in items:
        token = str(raw or "").strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        values.append(token)
    return values


def _contains_keywords(*parts: str | None, keywords: tuple[str, ...]) -> bool:
    haystack = " ".join(str(part or "") for part in parts).lower()
    return any(keyword in haystack for keyword in keywords)


def _default_ttl(strategy: str, *, sensitive: bool) -> str:
    if strategy == "no_memory":
        return "Sin TTL durable; el contexto se descarta al terminar la sesion."
    if sensitive:
        return "TTL corto con eliminacion programada y revision humana antes de prolongar retencion."
    if strategy == "persistent_memory":
        return "TTL por tipo de artefacto con limpieza de checkpoints obsoletos y snapshots reemplazados."
    return "TTL alineado al ciclo de la sesion y checkpoints recientes."


def _band_for_score(score: float) -> str:
    if score >= 0.84:
        return "high"
    if score >= 0.68:
        return "medium"
    return "low"


def _is_sensitive_case(discovery: DiscoveryArtifact, blueprint: BlueprintArtifact) -> bool:
    return _contains_keywords(
        discovery.problem_statement,
        discovery.current_process,
        discovery.desired_outcome,
        " ".join(discovery.constraints),
        " ".join(blueprint.guardrails),
        keywords=("pii", "dato sensible", "sensible", "financ", "salud", "credential", "privad", "compliance"),
    )


def _effective_knowledge_profile(
    discovery: DiscoveryArtifact,
    blueprint: BlueprintArtifact,
    memory_strategy: str,
) -> KnowledgeProfile:
    existing = blueprint.knowledge_profile or KnowledgeProfile()
    if existing.mode.strip().lower() == "rag" or existing.sources:
        return existing.model_copy(deep=True)
    return derive_knowledge_profile(discovery, blueprint.tools, memory_strategy)


def _memory_need_decision(
    *,
    strategy: str,
    knowledge_mode: str,
    rationale: str,
    source_refs: list[str],
) -> MemoryNeedDecision:
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy == "no_memory":
        return MemoryNeedDecision(
            mode="stateless",
            required=False,
            summary="El agente puede operar sin memoria durable y con contexto solo de la interaccion activa.",
            rationale=rationale,
            source_refs=source_refs,
        )
    if knowledge_mode == "rag":
        return MemoryNeedDecision(
            mode="knowledge_rag",
            required=True,
            summary="El agente necesita memoria operativa y retrieval gobernado sobre fuentes aprobadas.",
            rationale=rationale,
            source_refs=source_refs,
        )
    if normalized_strategy == "persistent_memory":
        return MemoryNeedDecision(
            mode="durable_memory",
            required=True,
            summary="El agente necesita checkpoints y memoria durable para continuidad entre corridas.",
            rationale=rationale,
            source_refs=source_refs,
        )
    if normalized_strategy == "session_memory_with_checkpoints":
        return MemoryNeedDecision(
            mode="working_memory",
            required=True,
            summary="El agente necesita continuidad intra-sesion y checkpoints compactos, sin sobrepersistir ruido.",
            rationale=rationale,
            source_refs=source_refs,
        )
    return MemoryNeedDecision(
        mode="session_only",
        required=True,
        summary="El agente necesita memoria acotada a la sesion para mantener continuidad y grounding operativo.",
        rationale=rationale,
        source_refs=source_refs,
    )


def _build_layer_designs(
    *,
    memory_profile: MemoryProfile,
    knowledge_profile: KnowledgeProfile,
    proposal: MemoryArchitectureRecommendationOutput | None,
    sensitive_case: bool,
) -> tuple[MemoryLayerDesign, MemoryLayerDesign, MemoryLayerDesign]:
    strategy = memory_profile.strategy.strip().lower()
    short_term = MemoryLayerDesign(
        layer_key="short_term",
        label="Contexto corto de runtime",
        owner="runtime_builder",
        summary=_coalesce(
            proposal.short_term_strategy if proposal is not None else "",
            "Mantener solo el contexto minimo necesario para la tarea activa, con compaction progresiva.",
        ),
        stores=["runtime_context", "approved_refs"],
        write_triggers=["nuevo turno", "salida de tool relevante", "cambio de etapa"],
        read_paths=["inicio de tarea", "antes de decidir", "antes de invocar tools"],
        compaction_policy="Resumir y paginar cuando el presupuesto de contexto se acerque al limite operativo.",
        retention_policy="Descartar al cerrar la sesion o al crear un checkpoint mas reciente.",
    )
    working = MemoryLayerDesign(
        layer_key="working_memory",
        label="Memoria de trabajo",
        owner="agent_runtime",
        summary=(
            "No requiere memoria de trabajo persistente fuera del turno activo."
            if strategy == "no_memory"
            else _coalesce(
                proposal.write_policy if proposal is not None else "",
                "Persistir solo checkpoints, decisiones aprobadas y estado reanudable; nunca prompts crudos ni ruido transitorio.",
            )
        ),
        stores=(
            ["session_checkpoint", "decision_digest"]
            if strategy in {"session_memory", "session_memory_with_checkpoints", "persistent_memory"}
            else ["ephemeral_turn_state"]
        ),
        write_triggers=["checkpoint consistente", "approval relevante", "handoff", "error recuperable"],
        read_paths=["reanudar flujo", "resolver handoff", "auditoria operativa"],
        compaction_policy="Compactar por hitos, no por cada mensaje; evitar loops de escritura de observaciones redundantes.",
        retention_policy=(
            "Mantener solo checkpoints recientes y artefactos con owner claro."
            if not sensitive_case
            else "Retener checkpoints minimos con TTL corto y limpieza reforzada por sensibilidad."
        ),
    )
    long_term = MemoryLayerDesign(
        layer_key="long_term",
        label="Memoria durable y conocimiento",
        owner="knowledge_runtime",
        summary=(
            "No se requiere memoria durable; solo referencias derivadas de la sesion."
            if strategy == "no_memory" and knowledge_profile.mode != "rag"
            else _coalesce(
                proposal.long_term_strategy if proposal is not None else "",
                "Persistir solo artefactos explicables: decisiones, reglas, hechos aprobados y knowledge source refs.",
            )
        ),
        stores=(
            ["knowledge_contract", "durable_decision_log", "artifact_refs"]
            if strategy == "persistent_memory" or knowledge_profile.mode == "rag"
            else ["artifact_refs"]
        ),
        write_triggers=["aprobacion de artefacto", "ingesta aprobada", "actualizacion de fuente", "policy change"],
        read_paths=["inicio de fase", "recovery", "grounding documental", "exportes"],
        compaction_policy="Referenciar artefactos y manifests por id en vez de reenviar cuerpos completos al LLM.",
        retention_policy=(
            "Aplicar TTL, lineage y borrado por politica para evitar sobre-retencion."
            if strategy != "no_memory" or knowledge_profile.mode == "rag"
            else "Sin escritura durable salvo manifests temporales del builder."
        ),
    )
    return short_term, working, long_term


def _build_context_budget_plan(
    *,
    knowledge_mode: str,
    definition_artifact: RequirementsDefinitionOutput | None,
    design_artifact: DesignRecommendationArtifact | None,
) -> list[MemoryContextBudgetEntry]:
    role_count = 0
    if design_artifact is not None and design_artifact.selected_design is not None:
        role_count = len(design_artifact.selected_design.roles)
    has_open_questions = bool(definition_artifact and definition_artifact.open_questions)
    budget_entries = [
        MemoryContextBudgetEntry(
            role="planner",
            task_kind="planning_runtime",
            max_context_tokens=3200 if has_open_questions else 2800,
            max_short_term_items=6,
            max_retrieved_sources=3 if knowledge_mode == "rag" else 1,
            strategy="Priorizar resumen ejecutivo, decisiones aprobadas y gaps vivos antes de expandir evidencia.",
            source_refs=["session.journey_latest_artifacts.define", "session.journey_latest_artifacts.design"],
        ),
        MemoryContextBudgetEntry(
            role="executor",
            task_kind="execution_runtime",
            max_context_tokens=2200,
            max_short_term_items=5,
            max_retrieved_sources=2 if knowledge_mode == "rag" else 1,
            strategy="Usar solo el contexto necesario para el paso actual y leer evidencias bajo demanda.",
            source_refs=["session.blueprint", "session.short_term_memory"],
        ),
    ]
    if knowledge_mode == "rag":
        budget_entries.append(
            MemoryContextBudgetEntry(
                role="retrieval",
                task_kind="knowledge_runtime",
                max_context_tokens=1800,
                max_short_term_items=3,
                max_retrieved_sources=4,
                strategy="Traer pocas fuentes aprobadas, rerankear y citar; no mezclar corpus completo en cada llamada.",
                source_refs=["session.blueprint.knowledge_profile", "workspace.knowledge_manifest"],
            )
        )
    if role_count > 1:
        budget_entries.append(
            MemoryContextBudgetEntry(
                role="coordinator",
                task_kind="handoff_runtime",
                max_context_tokens=2000,
                max_short_term_items=4,
                max_retrieved_sources=2,
                strategy="Pasar handoffs por digest, artefactos versionados y referencias, nunca por transcripcion completa.",
                source_refs=["session.journey_latest_artifacts.design"],
            )
        )
    return budget_entries


def _build_write_read_matrix(knowledge_mode: str) -> list[MemoryWriteReadRule]:
    matrix = [
        MemoryWriteReadRule(
            scope="runtime_context",
            owner="runtime_builder",
            write_when="Capturar solo hechos del turno, salidas relevantes y estado minimo reanudable.",
            do_not_write_when="No escribir prompts crudos, pensamientos intermedios redundantes ni respuestas sin evidencia.",
            read_when="Al iniciar un paso, antes de decidir y antes de invocar una tool aprobada.",
            compact_when="Cuando el presupuesto de contexto se acerque al umbral o cambie la fase del workflow.",
        ),
        MemoryWriteReadRule(
            scope="durable_memory",
            owner="agent_runtime",
            write_when="Guardar decisiones aprobadas, checkpoints consistentes, reglas y justificaciones con owner claro.",
            do_not_write_when="No persistir ruido de conversacion, intentos fallidos transitorios ni duplicados del mismo estado.",
            read_when="En reanudacion, auditoria, handoff o cuando una decision previa deba mantenerse coherente.",
            compact_when="Al cerrar hitos, approvals o cambios mayores de blueprint y herramientas.",
        ),
    ]
    if knowledge_mode == "rag":
        matrix.append(
            MemoryWriteReadRule(
                scope="knowledge_runtime",
                owner="knowledge_runtime",
                write_when="Ingestar solo fuentes aprobadas, versionadas y con politica de refresh definida.",
                do_not_write_when="No convertir automaticamente Docs/ del builder en fuente del agente objetivo.",
                read_when="Solo cuando la tarea requiera grounding documental o evidencia externa trazable.",
                compact_when="Al refrescar el corpus, invalidar citas obsoletas o detectar contradicciones.",
            )
        )
    return matrix


def _build_retention_rules(memory_profile: MemoryProfile, knowledge_profile: KnowledgeProfile) -> list[MemoryRetentionDeletionRule]:
    rules = [
        MemoryRetentionDeletionRule(
            scope="short_term",
            retention_policy="Mantener solo el resumen vivo y los checkpoints recientes de la sesion.",
            ttl_policy=_coalesce(memory_profile.ttl_policy, fallback="TTL ligado a la sesion activa."),
            deletion_policy="Eliminar al cerrar sesion o al compactar checkpoints sustituidos.",
            residency="workspace_scoped_runtime",
            source_refs=["blueprint.memory_profile", "session.short_term_memory"],
        ),
        MemoryRetentionDeletionRule(
            scope="durable_memory",
            retention_policy=_coalesce(
                memory_profile.retention_policy,
                fallback="Retener solo artefactos explicables y aprobados; borrar duplicados y ruido operacional.",
            ),
            ttl_policy=_coalesce(memory_profile.ttl_policy, fallback="TTL por tipo de artefacto y criticidad."),
            deletion_policy="Borrado por caducidad, reemplazo de version o retiro del owner.",
            residency="workspace_agent_namespace",
            source_refs=["blueprint.memory_profile"],
        ),
    ]
    if knowledge_profile.mode == "rag":
        rules.append(
            MemoryRetentionDeletionRule(
                scope="knowledge_sources",
                retention_policy="Retener solo fuentes aprobadas, con lineage, owner y licencia compatibles.",
                ttl_policy=_coalesce(
                    knowledge_profile.refresh_policy.expiration_policy,
                    fallback="TTL y freshness definidos por source contract.",
                ),
                deletion_policy=_coalesce(
                    knowledge_profile.refresh_policy.deletion_policy,
                    fallback="Eliminar chunks y embeddings de fuentes revocadas u obsoletas.",
                ),
                residency="workspace_knowledge_namespace",
                source_refs=["blueprint.knowledge_profile"],
            )
        )
    return rules


def _build_sensitivity_rules(
    *,
    sensitive_case: bool,
    memory_profile: MemoryProfile,
    knowledge_profile: KnowledgeProfile,
) -> list[MemorySensitivityIsolationRule]:
    restrictions = _dedupe(
        [
            *list(memory_profile.sensitivity_rules or []),
            *list(knowledge_profile.sensitivity_rules or []),
        ]
    )
    rules = [
        MemorySensitivityIsolationRule(
            scope="workspace",
            isolation_mode=_coalesce(memory_profile.workspace_scope, fallback="strict_workspace_isolation"),
            data_classes=["session_state", "approved_artifacts"],
            restrictions=restrictions
            or ["Aislar datos y manifests por workspace y no compartir memoria entre tenants."],
            source_refs=["blueprint.memory_profile", "workspace.contract"],
        ),
        MemorySensitivityIsolationRule(
            scope="agent",
            isolation_mode=_coalesce(memory_profile.agent_scope, fallback="agent_namespace_only"),
            data_classes=["decision_log", "checkpoint", "knowledge_refs"],
            restrictions=[
                "Separar memoria por agente, rol y corrida para evitar contaminacion cruzada."
            ],
            source_refs=["blueprint.memory_profile"],
        ),
    ]
    if sensitive_case:
        rules.append(
            MemorySensitivityIsolationRule(
                scope="sensitive_data",
                isolation_mode="restricted_namespace_with_short_ttl",
                data_classes=["sensitive_business_data"],
                restrictions=[
                    "Aplicar TTL corto, borrado visible, residencia controlada y aprobacion para ampliar retencion.",
                    "No almacenar secretos, credenciales ni datos personales completos en memoria durable.",
                ],
                source_refs=["discovery.constraints", "blueprint.guardrails"],
            )
        )
    return rules


def _build_tool_dependencies(
    *,
    approved_tools_digest: ApprovedToolsDigest | None,
    knowledge_profile: KnowledgeProfile,
    memory_profile: MemoryProfile,
) -> list[MemoryToolDependency]:
    return build_memory_tool_dependencies(
        approved_tools_digest=approved_tools_digest,
        knowledge_profile=knowledge_profile,
        memory_profile=memory_profile,
    )


def _map_critic_findings(critique: MemoryArchitectureCritiqueOutput | None) -> list[MemoryRecommendationFinding]:
    if critique is None:
        return []
    return [
        MemoryRecommendationFinding(
            finding_key=item.finding_key or f"critic-{index + 1}",
            title=item.title or "Finding de memoria",
            detail=item.detail,
            severity=item.severity,
            category="critic",
            suggested_action=item.suggested_action or "Revisar la recomendacion antes de aprobar.",
            source_refs=list(item.source_refs),
        )
        for index, item in enumerate(critique.findings)
    ]


def _append_finding(findings: list[MemoryRecommendationFinding], finding: MemoryRecommendationFinding) -> None:
    signature = (finding.finding_key.strip().lower(), finding.title.strip().lower(), finding.detail.strip().lower())
    if signature in {
        (item.finding_key.strip().lower(), item.title.strip().lower(), item.detail.strip().lower())
        for item in findings
    }:
        return
    findings.append(finding)


def _evaluate_memory_findings(
    *,
    artifact: MemoryRecommendationArtifact,
    critique: MemoryArchitectureCritiqueOutput | None,
) -> list[MemoryRecommendationFinding]:
    findings = _map_critic_findings(critique)
    durable_layers = " ".join(artifact.proposed_memory_profile.storage_layers).lower()
    knowledge_mode = artifact.proposed_knowledge_profile.mode.strip().lower()
    if artifact.memory_need_decision.mode == "stateless" and _contains_keywords(
        durable_layers,
        artifact.proposed_memory_profile.retention_policy,
        keywords=("vector", "persistent", "database", "durable", "checkpoint"),
    ):
        _append_finding(
            findings,
            MemoryRecommendationFinding(
                finding_key="memory-overprovisioned-stateless",
                title="Memoria durable innecesaria para un caso stateless",
                detail="La propuesta conserva capas durables aunque la decision de memoria indica operacion stateless.",
                severity="blocking",
                category="minimality",
                suggested_action="Eliminar storage durable o cambiar la decision de necesidad de memoria con evidencia suficiente.",
                source_refs=["memory_need_decision", "proposed_memory_profile.storage_layers"],
            ),
        )
    if knowledge_mode == "rag" and not artifact.proposed_knowledge_profile.sources:
        _append_finding(
            findings,
            MemoryRecommendationFinding(
                finding_key="rag-without-sources",
                title="RAG sin fuentes aprobadas",
                detail="La propuesta activa retrieval documental pero no declara fuentes aprobadas ni trazables.",
                severity="blocking",
                category="knowledge",
                suggested_action="Declara sources reales o desactiva RAG antes de aprobar.",
                source_refs=["proposed_knowledge_profile.mode", "proposed_knowledge_profile.sources"],
            ),
        )
    for dependency in artifact.tool_dependencies:
        if dependency.required and dependency.status == "missing":
            _append_finding(
                findings,
                MemoryRecommendationFinding(
                    finding_key=f"missing-tool:{dependency.tool_key}",
                    title="Dependencia de tool no aprobada",
                    detail=f"La estrategia de memoria requiere {dependency.tool_key} y no esta aprobada en Herramientas.",
                    severity="blocking",
                    category="compatibility",
                    suggested_action="Ajusta Tools o simplifica la estrategia de memoria para evitar esta dependencia.",
                    source_refs=["tool_dependencies", "approved_tools_digest"],
                ),
            )
    if not artifact.retention_and_deletion:
        _append_finding(
            findings,
            MemoryRecommendationFinding(
                finding_key="missing-retention-policy",
                title="Politica de retencion incompleta",
                detail="La propuesta no expone reglas claras de TTL, borrado y residencia.",
                severity="warning",
                category="governance",
                suggested_action="Define TTL, borrado y residencia por capa antes de promover la memoria.",
                source_refs=["retention_and_deletion"],
            ),
        )
    if not artifact.sensitivity_and_isolation:
        _append_finding(
            findings,
            MemoryRecommendationFinding(
                finding_key="missing-isolation-policy",
                title="Politica de aislamiento incompleta",
                detail="La propuesta no documenta aislamiento por workspace o agente.",
                severity="warning",
                category="security",
                suggested_action="Declara namespaces, owners y restricciones por tipo de memoria.",
                source_refs=["sensitivity_and_isolation"],
            ),
        )
    if critique is not None:
        for contradiction in critique.contradictions:
            _append_finding(
                findings,
                MemoryRecommendationFinding(
                    finding_key=f"contradiction:{len(findings) + 1}",
                    title="Contradiccion detectada por la critica",
                    detail=contradiction,
                    severity="warning",
                    category="critic",
                    suggested_action="Ajusta la recomendacion o documenta la excepcion antes de aprobar.",
                    source_refs=["critic_output.contradictions"],
                ),
            )
        for missing_evidence in critique.missing_evidence:
            _append_finding(
                findings,
                MemoryRecommendationFinding(
                    finding_key=f"missing-evidence:{len(findings) + 1}",
                    title="Evidencia faltante para justificar la memoria",
                    detail=missing_evidence,
                    severity="warning",
                    category="confidence",
                    suggested_action="Agregar evidencia aprobada o simplificar la estrategia propuesta.",
                    source_refs=["critic_output.missing_evidence"],
                ),
            )
    return findings


def _build_dry_compile_status(
    session_snapshot: SessionSnapshot | None,
    artifact: MemoryRecommendationArtifact,
) -> MemoryDryCompileStatus:
    if session_snapshot is None or session_snapshot.blueprint is None:
        return MemoryDryCompileStatus(
            status="pending",
            summary="No se ejecuto compilacion seca porque el snapshot canonico aun no esta disponible.",
        )
    projected_blueprint = session_snapshot.blueprint.model_copy(
        update={
            "memory_strategy": artifact.proposed_memory_profile.strategy,
            "memory_profile": artifact.proposed_memory_profile,
            "knowledge_profile": artifact.proposed_knowledge_profile,
        },
        deep=True,
    )
    candidate_snapshot = session_snapshot.model_copy(update={"blueprint": projected_blueprint}, deep=True)
    generated_at = utc_now()
    try:
        memory_policy = build_memory_policy(candidate_snapshot, generated_at=generated_at)
        build_short_term_memory(candidate_snapshot, generated_at=generated_at)
        knowledge_contract = build_knowledge_contract(candidate_snapshot, generated_at=generated_at)
        tool_contracts = build_tool_contracts(candidate_snapshot, generated_at=generated_at)
        compile_stage4_artifacts(
            candidate_snapshot,
            generated_at=generated_at,
            tool_contracts=tool_contracts,
            memory_policy=memory_policy,
            knowledge_contract=knowledge_contract,
        )
        return MemoryDryCompileStatus(
            status="ready",
            summary="Stage4 consumio correctamente memory-policy.v1, short-term-memory.v1 y knowledge-contract.v1.",
            generated_contracts=["memory-policy.v1", "short-term-memory.v1", "knowledge-contract.v1"],
            blocking_issues=[],
        )
    except Exception as exc:  # noqa: BLE001
        return MemoryDryCompileStatus(
            status="blocked",
            summary="La compilacion seca fallo y la memoria no debe promoverse hasta corregir el contrato.",
            generated_contracts=["memory-policy.v1", "short-term-memory.v1", "knowledge-contract.v1"],
            blocking_issues=[str(exc)],
        )


def _apply_proposal_to_profiles(
    *,
    discovery: DiscoveryArtifact,
    blueprint: BlueprintArtifact,
    baseline_memory_profile: MemoryProfile,
    baseline_knowledge_profile: KnowledgeProfile,
    approved_tools_digest: ApprovedToolsDigest | None,
    proposal: MemoryArchitectureRecommendationOutput | None,
    sensitive_case: bool,
) -> tuple[MemoryProfile, KnowledgeProfile]:
    if proposal is None:
        strategy = baseline_memory_profile.strategy
        return (
            baseline_memory_profile.model_copy(
                update={
                    "ttl_policy": _coalesce(
                        baseline_memory_profile.ttl_policy,
                        fallback=_default_ttl(strategy, sensitive=sensitive_case),
                    ),
                    "workspace_scope": _coalesce(
                        baseline_memory_profile.workspace_scope,
                        fallback="strict_workspace_isolation",
                    ),
                    "agent_scope": _coalesce(
                        baseline_memory_profile.agent_scope,
                        fallback="agent_namespace_only",
                    ),
                }
            ),
            baseline_knowledge_profile.model_copy(deep=True),
        )

    strategy = _coalesce(proposal.memory_strategy, baseline_memory_profile.strategy, fallback="session_memory")
    retrieval_policy = _coalesce(
        proposal.retrieval_strategy,
        baseline_memory_profile.retrieval_policy,
        fallback="Recuperar solo el contexto y las referencias estrictamente necesarias para la tarea activa.",
    )
    next_memory_profile = baseline_memory_profile.model_copy(
        update={
            "strategy": strategy,
            "storage_layers": _dedupe(list(proposal.storage_layers or baseline_memory_profile.storage_layers)),
            "write_policy": _coalesce(
                proposal.write_policy,
                baseline_memory_profile.write_policy,
                fallback="Persistir solo checkpoints consistentes, decisiones aprobadas y artefactos referenciables.",
            ),
            "retrieval_policy": retrieval_policy,
            "retention_policy": _coalesce(
                proposal.pruning_policy,
                baseline_memory_profile.retention_policy,
                fallback="Retener solo informacion con owner, proposito y politica de borrado visibles.",
            ),
            "ttl_policy": _coalesce(
                baseline_memory_profile.ttl_policy,
                fallback=_default_ttl(strategy, sensitive=sensitive_case),
            ),
            "workspace_scope": _coalesce(
                baseline_memory_profile.workspace_scope,
                fallback="strict_workspace_isolation",
            ),
            "agent_scope": _coalesce(
                baseline_memory_profile.agent_scope,
                fallback="agent_namespace_only",
            ),
            "sensitivity_rules": _dedupe(
                [
                    *list(baseline_memory_profile.sensitivity_rules or []),
                    *list(proposal.security_notes or []),
                ]
            ),
        }
    )

    current_knowledge = baseline_knowledge_profile.model_copy(deep=True)
    knowledge_mode = current_knowledge.mode.strip().lower()
    approved_knowledge_tools = (
        {item.strip().lower() for item in approved_tools_digest.knowledge_tool_keys}
        if approved_tools_digest is not None
        else set()
    )
    rag_signals = knowledge_mode == "rag" or "knowledge_retrieval" in approved_knowledge_tools
    rag_signals = bool(
        rag_signals
        or _contains_keywords(
            proposal.retrieval_strategy,
            proposal.long_term_strategy,
            proposal.rationale,
            keywords=("rag", "retrieval", "document", "citation", "fuente", "knowledge"),
        )
    )
    if rag_signals:
        if current_knowledge.mode.strip().lower() != "rag":
            current_knowledge = derive_knowledge_profile(discovery, blueprint.tools, strategy)
        current_knowledge = current_knowledge.model_copy(
            update={
                "mode": "rag",
                "notes": _coalesce(
                    current_knowledge.notes,
                    proposal.rationale,
                    fallback="RAG habilitado solo sobre fuentes aprobadas y con grounding obligatorio.",
                ),
            }
        )
    else:
        current_knowledge = KnowledgeProfile(mode="none")
    return next_memory_profile, current_knowledge


def _build_confidence(
    *,
    proposal: MemoryArchitectureRecommendationOutput | None,
    critique: MemoryArchitectureCritiqueOutput | None,
    findings: list[MemoryRecommendationFinding],
    dry_compile_status: MemoryDryCompileStatus,
) -> MemoryRecommendationConfidence:
    score = 0.52
    rationale = ["Baseline deterministico del builder disponible."]
    if proposal is not None:
        score += 0.16
        rationale.append("El arquitecto LLM propuso una estrategia estructurada de memoria.")
    if critique is not None:
        score += 0.12
        rationale.append("La critica LLM reviso minimalidad, compatibilidad y gobierno.")
    if dry_compile_status.status == "ready":
        score += 0.12
        rationale.append("La compilacion seca confirma compatibilidad con Stage4.")
    blocking = sum(1 for item in findings if item.severity == "blocking")
    warnings = sum(1 for item in findings if item.severity == "warning")
    score -= min(0.18, blocking * 0.08)
    score -= min(0.08, warnings * 0.02)
    if blocking:
        rationale.append("Persisten findings bloqueantes que reducen la confianza operativa.")
    elif warnings:
        rationale.append("Persisten advertencias menores que requieren revision humana.")
    return MemoryRecommendationConfidence(
        overall=max(0.0, min(0.96, round(score, 2))),
        band=_band_for_score(max(0.0, min(0.96, score))),
        rationale=" ".join(rationale),
    )


def _review_state_for_artifact(
    *,
    findings: list[MemoryRecommendationFinding],
    dry_compile_status: MemoryDryCompileStatus,
    critique: MemoryArchitectureCritiqueOutput | None,
) -> ReviewState:
    if dry_compile_status.status == "blocked" or any(item.severity == "blocking" for item in findings):
        return ReviewState.blocked
    if critique is not None and critique.overall_status == "accepted" and not findings:
        return ReviewState.complete
    if not findings and dry_compile_status.status == "ready":
        return ReviewState.complete
    return ReviewState.partial


def build_memory_recommendation_artifact(
    *,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    blueprint: BlueprintArtifact,
    approved_tools_digest: ApprovedToolsDigest | None,
    source_session_id: UUID | None,
    source_blueprint_version: int | None,
    current_blueprint_version: int | None,
    source_stage_versions: MemoryRecommendationSourceStageVersions | None = None,
    instructions: str = "",
    definition_artifact: RequirementsDefinitionOutput | None = None,
    design_artifact: DesignRecommendationArtifact | None = None,
    proposal: MemoryArchitectureRecommendationOutput | None = None,
    critique: MemoryArchitectureCritiqueOutput | None = None,
    session_snapshot: SessionSnapshot | None = None,
    proposed_memory_profile: MemoryProfile | None = None,
    proposed_knowledge_profile: KnowledgeProfile | None = None,
) -> MemoryRecommendationArtifact:
    baseline_strategy = (
        approved_tools_digest.recommended_memory_strategy
        if approved_tools_digest is not None and approved_tools_digest.recommended_memory_strategy
        else select_memory_strategy(discovery, canvas)
    )
    baseline_memory_profile = derive_memory_profile(
        discovery,
        canvas,
        approved_tools_digest=approved_tools_digest,
    ).model_copy(update={"strategy": baseline_strategy})
    baseline_knowledge_profile = _effective_knowledge_profile(discovery, blueprint, baseline_strategy)
    sensitive_case = _is_sensitive_case(discovery, blueprint)
    if proposed_memory_profile is not None and proposed_knowledge_profile is not None:
        memory_profile = proposed_memory_profile.model_copy(
            update={
                "ttl_policy": _coalesce(
                    proposed_memory_profile.ttl_policy,
                    fallback=_default_ttl(proposed_memory_profile.strategy, sensitive=sensitive_case),
                ),
                "workspace_scope": _coalesce(
                    proposed_memory_profile.workspace_scope,
                    fallback="strict_workspace_isolation",
                ),
                "agent_scope": _coalesce(
                    proposed_memory_profile.agent_scope,
                    fallback="agent_namespace_only",
                ),
            },
            deep=True,
        )
        knowledge_profile = proposed_knowledge_profile.model_copy(deep=True)
    else:
        memory_profile, knowledge_profile = _apply_proposal_to_profiles(
            discovery=discovery,
            blueprint=blueprint,
            baseline_memory_profile=baseline_memory_profile,
            baseline_knowledge_profile=baseline_knowledge_profile,
            approved_tools_digest=approved_tools_digest,
            proposal=proposal,
            sensitive_case=sensitive_case,
        )
    decision_rationale = _coalesce(
        proposal.rationale if proposal is not None else "",
        f"La estrategia base deriva de discovery aprobado y del digest vigente de tools ({approved_tools_digest.digest_sha256})."
        if approved_tools_digest is not None
        else "La estrategia base deriva de discovery aprobado y del blueprint vigente.",
    )
    memory_need_decision = _memory_need_decision(
        strategy=memory_profile.strategy,
        knowledge_mode=knowledge_profile.mode.strip().lower(),
        rationale=decision_rationale,
        source_refs=[
            "session.discovery",
            "session.canvas",
            "session.blueprint",
            "session.latest_tool_recommendation",
        ],
    )
    short_term_design, working_memory_design, long_term_design = _build_layer_designs(
        memory_profile=memory_profile,
        knowledge_profile=knowledge_profile,
        proposal=proposal,
        sensitive_case=sensitive_case,
    )
    knowledge_design = MemoryKnowledgeDesign(
        mode=knowledge_profile.mode,
        rag_required=knowledge_profile.mode.strip().lower() == "rag",
        summary=(
            "No se requiere knowledge retrieval durable; el agente debe operar con contexto aprobado y tools transaccionales."
            if knowledge_profile.mode.strip().lower() != "rag"
            else "El retrieval debe ser trazable, con fuentes aprobadas, versionadas y citadas en cada respuesta grounded."
        ),
        source_scope=(
            "Solo fuentes explicitamente aprobadas para este agente; Docs/ del builder no viaja automaticamente."
            if knowledge_profile.mode.strip().lower() == "rag"
            else "Sin corpus durable dedicado para el agente objetivo."
        ),
        approved_sources=[KnowledgeSource.model_validate(item.model_dump(mode="json")) for item in knowledge_profile.sources],
        ingestion_policy=knowledge_profile.ingestion_policy.model_copy(deep=True),
        embedding_policy=knowledge_profile.embedding_policy.model_copy(deep=True),
        retrieval_policy=knowledge_profile.retrieval_policy.model_copy(deep=True),
        refresh_policy=knowledge_profile.refresh_policy.model_copy(deep=True),
        grounding_policy=knowledge_profile.grounding_policy.model_copy(deep=True),
        notes=_dedupe(
            [
                knowledge_profile.notes,
                "El corpus Docs/ del builder solo informa mejores practicas y no se convierte en fuente del agente salvo aprobacion explicita.",
            ]
        ),
    )
    artifact = MemoryRecommendationArtifact(
        source_session_id=source_session_id,
        source_blueprint_version=source_blueprint_version,
        current_blueprint_version=current_blueprint_version if current_blueprint_version is not None else source_blueprint_version,
        generation_instructions=instructions,
        source_stage_versions=source_stage_versions or MemoryRecommendationSourceStageVersions(),
        memory_need_decision=memory_need_decision,
        short_term_design=short_term_design,
        working_memory_design=working_memory_design,
        long_term_design=long_term_design,
        knowledge_design=knowledge_design,
        context_budget_plan=_build_context_budget_plan(
            knowledge_mode=knowledge_profile.mode.strip().lower(),
            definition_artifact=definition_artifact,
            design_artifact=design_artifact,
        ),
        write_read_matrix=_build_write_read_matrix(knowledge_profile.mode.strip().lower()),
        retention_and_deletion=_build_retention_rules(memory_profile, knowledge_profile),
        sensitivity_and_isolation=_build_sensitivity_rules(
            sensitive_case=sensitive_case,
            memory_profile=memory_profile,
            knowledge_profile=knowledge_profile,
        ),
        tool_dependencies=_build_tool_dependencies(
            approved_tools_digest=approved_tools_digest,
            knowledge_profile=knowledge_profile,
            memory_profile=memory_profile,
        ),
        evidence_refs=_dedupe(
            [
                "session.discovery",
                "session.canvas",
                "session.blueprint",
                "session.journey_latest_artifacts.define",
                "session.journey_latest_artifacts.design",
                "session.journey_latest_artifacts.tools",
            ]
        ),
        open_questions=list(proposal.open_questions) if proposal is not None else [],
        guided_questions=[
            _guided_question_from_prioritized(question, stage_scope="memory")
            for question in (proposal.guided_questions if proposal is not None else [])
        ],
        missing_information=list(critique.missing_evidence) if critique is not None else [],
        proposed_memory_profile=memory_profile,
        proposed_knowledge_profile=knowledge_profile,
    )
    dry_compile_status = _build_dry_compile_status(session_snapshot, artifact)
    findings = _evaluate_memory_findings(artifact=artifact, critique=critique)
    if dry_compile_status.status == "blocked":
        _append_finding(
            findings,
            MemoryRecommendationFinding(
                finding_key="dry-compile-blocked",
                title="La compilacion seca de Stage4 fallo",
                detail="La politica de memoria o el contrato de conocimiento no compilan con el runtime posterior.",
                severity="blocking",
                category="compilation",
                suggested_action="Corrige la politica de memoria, knowledge contract o dependencias de tools antes de aprobar.",
                source_refs=["dry_compile_status", "stage4_compiler"],
            ),
        )
    confidence = _build_confidence(
        proposal=proposal,
        critique=critique,
        findings=findings,
        dry_compile_status=dry_compile_status,
    )
    review_state = _review_state_for_artifact(
        findings=findings,
        dry_compile_status=dry_compile_status,
        critique=critique,
    )
    return artifact.model_copy(
        update={
            "critic_findings": findings,
            "dry_compile_status": dry_compile_status,
            "confidence": confidence,
            "review_state": review_state,
            "summary": _coalesce(
                critique.summary if critique is not None else "",
                proposal.rationale if proposal is not None else "",
                fallback=memory_need_decision.summary,
            ),
        }
    )
