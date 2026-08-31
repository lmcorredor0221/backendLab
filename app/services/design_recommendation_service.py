from __future__ import annotations

from collections.abc import Iterable

from app.models import (
    CanvasArtifact,
    DesignAlternative,
    DesignBlueprintProjection,
    DesignCritiqueFinding,
    DesignFailureMode,
    DesignFitAlternativeScore,
    DesignFitMatrixEntry,
    DesignHandoff,
    GuidedAnswerOptionEntry,
    GuidedQuestionEntry,
    DesignRecommendationArtifact,
    DesignRecommendationConfidence,
    DesignRequirementCoverageEntry,
    DesignRole,
    DiscoveryArtifact,
    PatternCatalogEntry,
    ReviewState,
)
from app.services.llm_runtime.builder_contracts import (
    AgentDesignProposalOutput,
    DesignCritiqueOutput,
    PrioritizedQuestion,
    RequirementsDefinitionOutput,
)
from app.services.rules import (
    build_agent_archetype_catalog,
    build_architecture_catalog,
    build_pattern_family_catalog,
    build_reasoning_catalog,
    default_guardrails,
    derive_safety_checks,
    normalize_text,
)


DESIGN_INTELLIGENCE_EVIDENCE_REFS = [
    "knowledge.agentic_knowledge_base.patterns_playbook_2026",
    "knowledge.agentic_knowledge_base.tools_governance_catalog_playbook_2026",
    "knowledge.agentic_knowledge_base.memory_architecture_playbook_2026",
]


DESIGN_ARCHITECTURE_INTELLIGENCE: dict[str, dict[str, str | list[str]]] = {
    "single_agent": {
        "agent_archetype": "Agente operativo compacto",
        "pattern_family": "Single-agent / ReAct gobernado",
        "business_fit": "Adecuado cuando el valor principal esta en resolver un flujo acotado con baja coordinacion.",
        "value_hypothesis": "Reduce friccion y retrabajo sin introducir costos de orquestacion innecesarios.",
        "operational_model": "Un agente conserva contexto, decide el siguiente paso y escala solo ante riesgo o falta de evidencia.",
        "why_recommended": "Maximiza simplicidad, trazabilidad y velocidad de implementacion para MVPs de alcance claro.",
        "why_not_simpler": "No conviene reducirlo a automatizacion estatica si el usuario necesita razonamiento contextual.",
        "why_not_more_complex": "No conviene dividir en agentes si no hay dominios independientes ni paralelismo material.",
        "tool_implications": [
            "approval_gate: mantener compuertas visibles cuando existan decisiones no delegables.",
        ],
        "memory_implications": [
            "session_memory_with_checkpoints: conservar resumen vivo y decisiones aprobadas por etapa.",
            "context_budgeting: compactar evidencia para evitar reenviar todo el historial.",
        ],
        "risk_tradeoffs": [
            "Menor coordinacion, pero tambien menor especializacion ante casos heterogeneos.",
            "El principal riesgo es sobrecargar al agente con demasiadas responsabilidades.",
        ],
        "memory_strategy": "session_memory_with_checkpoints",
    },
    "single_agent_with_skills": {
        "agent_archetype": "Orquestador unico con skills",
        "pattern_family": "ReAct + tool-use modular",
        "business_fit": "Adecuado cuando el negocio requiere una experiencia simple, pero con capacidades especializadas bajo demanda.",
        "value_hypothesis": "Mantiene una sola conversacion de negocio y activa capacidades solo cuando aportan cobertura real.",
        "operational_model": "El orquestador decide, llama skills/tools acotadas y consolida la respuesta con evidencia.",
        "why_recommended": "Equilibra calidad, costo y extensibilidad sin pasar prematuramente a una arquitectura multiagente.",
        "why_not_simpler": "Un agente sin skills se quedaria corto si debe consultar fuentes, validar reglas o generar artefactos.",
        "why_not_more_complex": "Un equipo multiagente agregaria handoffs y latencia si el flujo sigue siendo controlable por un solo owner.",
        "tool_implications": [
            "read_system_of_record: consultar fuentes operativas cuando el flujo dependa de estado externo.",
            "knowledge_retrieval: recuperar conocimiento gobernado cuando no baste contexto inline.",
            "approval_gate: pausar side effects y decisiones no delegables.",
        ],
        "memory_implications": [
            "session_memory_with_checkpoints: preservar objetivo, decisiones y estado de cada skill invocada.",
            "tool_observation_log: registrar observaciones usadas para justificar la siguiente accion.",
            "react_dependency_gap_tracking: convertir dependencias faltantes en gaps gobernados y reanudables.",
        ],
        "risk_tradeoffs": [
            "Buen balance para MVP, pero requiere contratos claros por skill/tool.",
            "Si las skills crecen sin catalogo, puede aparecer deuda de capacidades duplicadas.",
        ],
        "memory_strategy": "session_memory_with_checkpoints",
    },
    "handoffs": {
        "agent_archetype": "Cadena planner-executor-reviewer",
        "pattern_family": "Plan-and-Execute + handoffs gobernados",
        "business_fit": "Adecuado cuando el negocio necesita etapas claras, ownership por paso y revision antes del cierre.",
        "value_hypothesis": "Aumenta confianza y auditabilidad en flujos donde cada transicion debe dejar evidencia.",
        "operational_model": "Planner define el plan, Executor produce entregables y Reviewer valida cobertura antes de promover.",
        "why_recommended": "Hace visible quien hace que, que evidencia se transfiere y cuando se debe detener el flujo.",
        "why_not_simpler": "Una sola rutina puede ocultar responsabilidades si hay aprobaciones, revision o recuperacion por paso.",
        "why_not_more_complex": "No requiere un supervisor con subagentes si las responsabilidades son secuenciales y no paralelas.",
        "tool_implications": [
            "approval_gate: revisar checkpoints de promocion y decisiones sensibles.",
            "human_handoff: escalar cuando el reviewer detecte ambiguedad o riesgo no delegable.",
            "scheduler: reanudar pasos pendientes cuando exista trabajo asincrono.",
        ],
        "memory_implications": [
            "handoff_state_tracking: guardar payload, owner, criterio de exito y estado por handoff.",
            "checkpoint_resume: retomar desde el ultimo paso estable sin repetir todo el flujo.",
            "decision_log: conservar rationale y aprobaciones entre roles.",
        ],
        "risk_tradeoffs": [
            "Mas trazabilidad, pero tambien mas superficie de estados intermedios.",
            "Si no hay limites de reintento, los handoffs pueden crear bucles costosos.",
        ],
        "memory_strategy": "workflow_memory_with_handoffs",
    },
    "supervisor_with_subagents": {
        "agent_archetype": "Supervisor con especialistas",
        "pattern_family": "Supervisor / multi-agent orchestration",
        "business_fit": "Adecuado cuando existen dominios diferenciados que requieren especialistas y consolidacion central.",
        "value_hypothesis": "Mejora profundidad y cobertura en problemas complejos sin perder un punto unico de control.",
        "operational_model": "El supervisor divide subobjetivos, asigna especialistas, compara evidencia y decide consolidacion.",
        "why_recommended": "Justifica mayor complejidad cuando hay dominios independientes, alto riesgo o entregables heterogeneos.",
        "why_not_simpler": "Un solo agente puede perder precision si debe cubrir dominios tecnicos, negocio y validacion a la vez.",
        "why_not_more_complex": "No conviene fan-out amplio si no hay paralelismo real ni metricas que justifiquen costo.",
        "tool_implications": [
            "human_handoff: escalar decisiones que el supervisor no pueda resolver con evidencia.",
            "approval_gate: controlar consolidaciones con impacto operacional o comercial.",
            "knowledge_retrieval: compartir grounding comun entre especialistas.",
        ],
        "memory_implications": [
            "shared_blackboard: mantener estado comun entre especialistas sin duplicar contexto.",
            "specialist_summaries: compactar salida por subagente antes de consolidar.",
            "loop_guardrails: limitar iteraciones supervisor-especialista y registrar causa de reintento.",
        ],
        "risk_tradeoffs": [
            "Mejor especializacion, pero mayor latencia y costo LLM.",
            "Riesgo de inconsistencias si cada especialista opera con fuentes distintas.",
        ],
        "memory_strategy": "shared_blackboard_with_checkpoints",
    },
    "router_parallel": {
        "agent_archetype": "Router con workers paralelos",
        "pattern_family": "Routing / fan-out-fan-in controlado",
        "business_fit": "Adecuado cuando el caso requiere clasificar solicitudes o consultar rutas independientes en paralelo.",
        "value_hypothesis": "Reduce tiempo de ciclo cuando hay trabajos independientes y una agregacion final verificable.",
        "operational_model": "El router clasifica, dispara workers acotados, agrega resultados y aplica criterios de consistencia.",
        "why_recommended": "Aporta velocidad cuando el paralelismo es real y cada worker tiene contrato claro de entrada/salida.",
        "why_not_simpler": "Una ruta secuencial puede ser lenta o insuficiente si debe comparar fuentes o alternativas independientes.",
        "why_not_more_complex": "Un supervisor pesado no se justifica si solo se necesita enrutar y agregar resultados.",
        "tool_implications": [
            "scheduler: controlar ejecuciones asincronas, timeouts y fan-in.",
            "read_system_of_record: consultar fuentes independientes sin side effects.",
            "approval_gate: revisar agregaciones con riesgo antes de ejecutar acciones.",
        ],
        "memory_implications": [
            "parallel_branch_state: aislar contexto por rama y conservar resultados compactos.",
            "fan_in_summary: agregar evidencia sin mezclar trazas incompatibles.",
            "retry_budget: limitar reintentos por worker para evitar loops.",
        ],
        "risk_tradeoffs": [
            "Mejor throughput, pero mayor riesgo de respuestas divergentes entre ramas.",
            "Necesita timeouts e idempotencia para que un worker fallido no bloquee todo.",
        ],
        "memory_strategy": "parallel_branch_memory_with_fan_in",
    },
}


def _normalized_list(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = normalize_text(value)
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(item)
    return normalized


def _normalize_design_fit_score(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0 < score <= 1:
        score *= 100
    return max(0.0, min(100.0, score))


def _definition_question_blocks_design(question) -> bool:  # noqa: ANN001 - supports legacy and LLM question contracts.
    explicit_blocking = getattr(question, "blocking", None)
    if explicit_blocking is not None:
        return bool(explicit_blocking)
    blocking_stages = getattr(question, "blocking_stages", None) or []
    return any(str(stage or "").strip().lower() in {"design", "build_blueprint", "blueprint"} for stage in blocking_stages)


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


def _merge_guided_questions(
    current: Iterable[GuidedQuestionEntry],
    incoming: Iterable[PrioritizedQuestion],
    *,
    stage_scope: str,
) -> list[GuidedQuestionEntry]:
    merged: list[GuidedQuestionEntry] = []
    seen: set[str] = set()
    for question in [
        *current,
        *[_guided_question_from_prioritized(item, stage_scope=stage_scope) for item in incoming],
    ]:
        key = question.key or question.question
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        merged.append(question)
    return merged


def _definition_requirements(definition: RequirementsDefinitionOutput) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for item in definition.functional_requirements:
        entries.append(
            {
                "key": item.key,
                "title": item.title or item.requirement,
                "detail": item.requirement,
                "priority": item.priority,
                "category": "functional",
            }
        )
    for item in definition.non_functional_requirements:
        entries.append(
            {
                "key": item.key,
                "title": item.title or item.requirement,
                "detail": item.requirement,
                "priority": item.priority,
                "category": "non_functional",
            }
        )
    for item in definition.business_rules:
        entries.append(
            {
                "key": item.key,
                "title": item.title or item.rule,
                "detail": item.rule,
                "priority": item.priority,
                "category": "business_rule",
            }
        )
    return entries


def build_design_requirement_digest(definition: RequirementsDefinitionOutput) -> list[str]:
    digest: list[str] = []
    for item in _definition_requirements(definition)[:18]:
        digest.append(f"[{item['category']}/{item['priority']}] {item['detail']}")
    return digest


def _reasoning_for_architecture(
    architecture: PatternCatalogEntry,
    reasoning_catalog: list[PatternCatalogEntry],
) -> PatternCatalogEntry:
    preferred_by_architecture = {
        "single_agent": ("ReAct", "Plan-and-Execute"),
        "single_agent_with_skills": ("ReAct", "Plan-and-Execute"),
        "handoffs": ("Plan-and-Execute", "ReAct"),
        "supervisor_with_subagents": ("Plan-and-Execute", "ToT"),
        "router_parallel": ("ToT", "ReAct"),
    }
    for candidate_key in preferred_by_architecture.get(architecture.key, ("ReAct", "Plan-and-Execute", "ToT")):
        for candidate in reasoning_catalog:
            if candidate.key == candidate_key:
                return candidate
    return max(reasoning_catalog, key=lambda item: item.fit_score)


def _preferred_catalog_entry(
    catalog: list[PatternCatalogEntry],
    preferred_keys: tuple[str, ...],
) -> PatternCatalogEntry:
    if not catalog:
        return PatternCatalogEntry()
    preferred_rank = {key: index for index, key in enumerate(preferred_keys)}
    preferred = [item for item in catalog if item.key in preferred_rank]
    if preferred:
        return max(
            preferred,
            key=lambda item: (
                item.fit_score,
                -preferred_rank[item.key],
            ),
        )
    return max(catalog, key=lambda item: item.fit_score)


def _catalog_intelligence_for_architecture(
    architecture_key: str,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
) -> tuple[PatternCatalogEntry, PatternCatalogEntry]:
    canonical = _canonical_architecture_key(architecture_key)
    archetype_preferences = {
        "single_agent": ("copilot_assistant", "workflow_operator", "research_synthesizer"),
        "single_agent_with_skills": (
            "workflow_operator",
            "rag_knowledge_assistant",
            "knowledge_steward",
            "human_approval_agent",
        ),
        "handoffs": ("workflow_operator", "human_approval_agent", "planning_scheduler_agent"),
        "supervisor_with_subagents": ("supervisor_multiagent", "research_synthesizer", "human_approval_agent"),
        "router_parallel": ("triage_router", "research_synthesizer", "monitoring_reactive_agent"),
    }
    pattern_preferences = {
        "single_agent": ("react_loop", "critic_evaluator_loop", "checkpoint_resume"),
        "single_agent_with_skills": ("react_loop", "rag_grounded", "human_in_the_loop", "checkpoint_resume"),
        "handoffs": ("plan_execute", "checkpoint_resume", "human_in_the_loop"),
        "supervisor_with_subagents": ("supervisor_subagents", "critic_evaluator_loop", "checkpoint_resume"),
        "router_parallel": ("router_worker", "parallel_fanout_fanin", "event_driven_reactor"),
    }
    archetype = _preferred_catalog_entry(
        build_agent_archetype_catalog(discovery, canvas),
        archetype_preferences.get(canonical, ("workflow_operator", "copilot_assistant")),
    )
    pattern_family = _preferred_catalog_entry(
        build_pattern_family_catalog(discovery, canvas),
        pattern_preferences.get(canonical, ("react_loop", "plan_execute")),
    )
    return archetype, pattern_family


def _business_technical_fit_score(
    architecture: PatternCatalogEntry,
    reasoning: PatternCatalogEntry,
    archetype: PatternCatalogEntry,
    pattern_family: PatternCatalogEntry,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
) -> float:
    base_score = (
        architecture.fit_score * 0.32
        + reasoning.fit_score * 0.22
        + archetype.fit_score * 0.28
        + pattern_family.fit_score * 0.18
    )
    case_text = _case_text(discovery, canvas)
    scope_count = len(canvas.mvp_scope)
    automation_count = len(discovery.operational_baseline.automation_opportunities)
    approval_count = len(canvas.agent_profile.human_approvals) + len(discovery.mvp_definition.non_delegable_decisions)
    complexity, relative_cost, _maintainability = _operating_profile(architecture.key)
    score = base_score

    if complexity == "high" and scope_count <= 2 and automation_count <= 1:
        score -= 10
    if architecture.key in {"single_agent", "single_agent_with_skills"} and (
        "paralel" in case_text or "multiagente" in case_text or "especialist" in case_text
    ):
        score -= 8
    if approval_count and pattern_family.key in {"human_in_the_loop", "checkpoint_resume", "plan_execute"}:
        score += 5
    if any(token in case_text for token in ("document", "manual", "politica", "política", "knowledge", "conocimiento")):
        if archetype.key in {"rag_knowledge_assistant", "knowledge_steward"} or pattern_family.key == "rag_grounded":
            score += 5
    if any(token in case_text for token in ("crm", "erp", "ticket", "api", "actualizar", "registrar")):
        if archetype.key == "transactional_agent" or architecture.key == "single_agent_with_skills":
            score += 4
    if relative_cost == "high" and "mvp" in case_text:
        score -= 4

    return round(max(0.0, min(100.0, score)), 2)


def _topology_for_architecture(architecture_key: str) -> tuple[str, list[DesignRole], list[DesignHandoff]]:
    if architecture_key == "single_agent":
        roles = [
            DesignRole(
                key="agent_core",
                title="Agente principal",
                responsibility="Entender la solicitud, recuperar contexto y ejecutar el flujo end-to-end.",
                limits=["No promueve cambios irreversibles sin aprobacion", "No divide el trabajo en subagentes"],
            )
        ]
        handoffs: list[DesignHandoff] = []
        return (
            "Un solo agente conserva el contexto operativo y ejecuta el flujo completo con checkpoints humanos visibles.",
            roles,
            handoffs,
        )
    if architecture_key == "single_agent_with_skills":
        roles = [
            DesignRole(
                key="agent_core",
                title="Orquestador unico",
                responsibility="Mantener el objetivo principal y decidir que capacidad especializada invocar.",
                limits=["No delega autonomia de negocio fuera del agente", "Mantiene el estado centralizado"],
            ),
            DesignRole(
                key="skill_modules",
                title="Capacidades especializadas",
                responsibility="Resolver tareas acotadas como analisis, recuperacion y validacion.",
                limits=["No toman decisiones finales de negocio", "Operan bajo contratos predefinidos"],
            ),
        ]
        handoffs = [
            DesignHandoff(
                from_role="Orquestador unico",
                to_role="Capacidades especializadas",
                trigger="Se requiere una capacidad puntual o una consulta gobernada.",
                payload="Contexto compacto y objetivo de la sub-tarea.",
                approval_required=False,
            )
        ]
        return (
            "Un orquestador unico conserva la conversacion y activa skills especializadas solo cuando agregan valor.",
            roles,
            handoffs,
        )
    if architecture_key == "handoffs":
        roles = [
            DesignRole(
                key="planner",
                title="Planner",
                responsibility="Construir el plan del caso y definir checkpoints antes de ejecutar.",
                limits=["No ejecuta side effects", "Debe dejar handoffs explicitos"],
            ),
            DesignRole(
                key="executor",
                title="Executor",
                responsibility="Ejecutar pasos del workflow con evidencia y trazabilidad.",
                limits=["No redefine el objetivo", "Escala ante ambiguedad o riesgo"],
            ),
            DesignRole(
                key="reviewer",
                title="Reviewer",
                responsibility="Validar cobertura, consistencia y cumplimiento antes del cierre.",
                limits=["No reemplaza aprobaciones humanas obligatorias"],
            ),
        ]
        handoffs = [
            DesignHandoff(
                from_role="Planner",
                to_role="Executor",
                trigger="Plan aprobado o suficiente para avanzar.",
                payload="Plan de trabajo, restricciones y criterios de exito.",
                approval_required=False,
            ),
            DesignHandoff(
                from_role="Executor",
                to_role="Reviewer",
                trigger="Resultado preliminar disponible o side effect completado.",
                payload="Evidencia, decisiones y riesgos residuales.",
                approval_required=True,
            ),
        ]
        return (
            "El caso se resuelve en handoffs secuenciales con ownership claro, checkpoints visibles y revision final.",
            roles,
            handoffs,
        )
    if architecture_key == "supervisor_with_subagents":
        roles = [
            DesignRole(
                key="supervisor",
                title="Supervisor",
                responsibility="Partir el problema, asignar especialistas y consolidar el resultado.",
                limits=["No permite loops indefinidos", "Escala decisiones no delegables"],
            ),
            DesignRole(
                key="domain_specialists",
                title="Especialistas",
                responsibility="Resolver dominios separados como analisis, integraciones o validacion.",
                limits=["Trabajan con objetivos acotados", "No redefinen reglas de negocio"],
            ),
        ]
        handoffs = [
            DesignHandoff(
                from_role="Supervisor",
                to_role="Especialistas",
                trigger="El caso exige dominios diferenciados o sub-problemas independientes.",
                payload="Subobjetivo, contexto parcial y condiciones de retorno.",
                approval_required=False,
            ),
            DesignHandoff(
                from_role="Especialistas",
                to_role="Supervisor",
                trigger="Subtarea completada o bloqueada.",
                payload="Resultado, evidencia y riesgos abiertos.",
                approval_required=False,
            ),
        ]
        return (
            "Un supervisor coordina especialistas solo cuando el problema realmente supera a un agente unico.",
            roles,
            handoffs,
        )
    roles = [
        DesignRole(
            key="router",
            title="Router",
            responsibility="Clasificar la solicitud y decidir ejecucion secuencial o paralela.",
            limits=["No ejecuta side effects directamente", "Debe preservar el contexto canonico"],
        ),
        DesignRole(
            key="workers",
            title="Workers especializados",
            responsibility="Consultar fuentes o ejecutar tareas paralelas acotadas.",
            limits=["No cierran el caso por si solos", "No toman decisiones no delegables"],
        ),
    ]
    handoffs = [
        DesignHandoff(
            from_role="Router",
            to_role="Workers especializados",
            trigger="Se requieren consultas paralelas o rutas alternativas controladas.",
            payload="Solicitud clasificada, filtros y criterio de agregacion.",
            approval_required=False,
        )
    ]
    return (
        "Un router central distribuye trabajo a workers especializados cuando el caso exige clasificacion y paralelismo real.",
        roles,
        handoffs,
    )


def _operating_profile(architecture_key: str) -> tuple[str, str, str]:
    if architecture_key in {"single_agent", "single_agent_with_skills"}:
        return ("low", "low", "high")
    if architecture_key == "handoffs":
        return ("medium", "medium", "medium")
    if architecture_key == "supervisor_with_subagents":
        return ("high", "high", "medium")
    return ("high", "high", "low")


def _approval_points(discovery: DiscoveryArtifact, canvas: CanvasArtifact) -> list[str]:
    return _normalized_list(
        [
            *discovery.mvp_definition.non_delegable_decisions,
            *canvas.agent_profile.human_approvals,
            "Escalar cuando la evidencia no cubra un requisito prioritario o exista riesgo operacional alto.",
        ]
    )


def _escalation_conditions(discovery: DiscoveryArtifact) -> list[str]:
    constraints = _normalized_list(discovery.constraints)
    return _normalized_list(
        [
            *constraints[:3],
            "Escalar cuando se detecte informacion contradictoria o insuficiente.",
            "Escalar ante side effects con impacto sobre clientes, finanzas o cumplimiento.",
        ]
    )


def _failure_modes(architecture_key: str) -> list[DesignFailureMode]:
    base = [
        DesignFailureMode(
            scenario="El LLM devuelve salida parcial o inconsistente.",
            retry_strategy="Reintentar una vez con contexto resumido y validacion estructurada.",
            compensation_strategy="Detener la promocion y pedir revision humana con evidencia visible.",
            idempotency_notes="No se ejecutan side effects mientras no exista salida validada.",
        ),
        DesignFailureMode(
            scenario="El contexto recuperado es insuficiente para cubrir un requisito prioritario.",
            retry_strategy="Recuperar solo fuentes adicionales de alta autoridad o pedir aclaracion.",
            compensation_strategy="Mantener el ultimo estado aprobado y marcar el artefacto en revision.",
            idempotency_notes="La misma entrada debe producir el mismo checkpoint de revision.",
        ),
    ]
    if architecture_key in {"handoffs", "supervisor_with_subagents", "router_parallel"}:
        base.append(
            DesignFailureMode(
                scenario="Un handoff llega ambiguo o el subagente deriva fuera del objetivo.",
                retry_strategy="Reenviar el subobjetivo con contrato mas estricto una sola vez.",
                compensation_strategy="Volver el control al coordinador y bloquear nuevas delegaciones.",
                idempotency_notes="Cada handoff usa un identificador estable para evitar ejecuciones duplicadas.",
            )
        )
    return base


def _canonical_architecture_key(architecture_key: str) -> str:
    token = normalize_text(architecture_key).lower()
    if token in DESIGN_ARCHITECTURE_INTELLIGENCE:
        return token
    if "router" in token or "parallel" in token or "paralel" in token:
        return "router_parallel"
    if "supervisor" in token or "especialist" in token or "multiagente" in token or "multi-agent" in token:
        return "supervisor_with_subagents"
    if "handoff" in token or "planner" in token or "executor" in token or "reviewer" in token:
        return "handoffs"
    if "skill" in token or "tool" in token or "rag" in token or "retrieval" in token:
        return "single_agent_with_skills"
    if "single" in token or "unico" in token or "único" in token:
        return "single_agent"
    return "single_agent_with_skills"


def _profile_for_architecture(architecture_key: str) -> dict[str, str | list[str]]:
    return DESIGN_ARCHITECTURE_INTELLIGENCE.get(
        _canonical_architecture_key(architecture_key),
        DESIGN_ARCHITECTURE_INTELLIGENCE["single_agent_with_skills"],
    )


def _profile_text(profile: dict[str, str | list[str]], key: str) -> str:
    value = profile.get(key, "")
    return value if isinstance(value, str) else ""


def _profile_list(profile: dict[str, str | list[str]], key: str) -> list[str]:
    value = profile.get(key, [])
    return list(value) if isinstance(value, list) else []


def _case_focus(discovery: DiscoveryArtifact | None, canvas: CanvasArtifact | None) -> str:
    if canvas is not None and normalize_text(canvas.user_goal):
        return normalize_text(canvas.user_goal)
    if discovery is not None and normalize_text(discovery.desired_outcome):
        return normalize_text(discovery.desired_outcome)
    if discovery is not None and normalize_text(discovery.problem_statement):
        return normalize_text(discovery.problem_statement)
    return "el flujo aprobado del MVP"


def _case_text(discovery: DiscoveryArtifact | None, canvas: CanvasArtifact | None) -> str:
    parts: list[str] = []
    if discovery is not None:
        parts.extend(
            [
                discovery.problem_statement,
                discovery.current_process,
                discovery.desired_outcome,
                " ".join(discovery.constraints),
                " ".join(discovery.mvp_definition.v1_scope),
                " ".join(discovery.mvp_definition.non_delegable_decisions),
            ]
        )
    if canvas is not None:
        parts.extend(
            [
                canvas.user_goal,
                canvas.primary_risk,
                canvas.success_metric,
                " ".join(canvas.agent_profile.expected_outputs),
                " ".join(canvas.agent_profile.human_approvals),
            ]
        )
    return " ".join(normalize_text(part) for part in parts if normalize_text(part)).lower()


def _contains_case_signal(case_text: str, signals: tuple[str, ...]) -> bool:
    return any(signal in case_text for signal in signals)


def _merge_implication_lines(*groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            item = normalize_text(value)
            if not item:
                continue
            signature = item.split(":", 1)[0].strip().lower() if ":" in item else item.lower()
            if signature in seen:
                continue
            seen.add(signature)
            merged.append(item)
    return merged


def _finding_text(finding: DesignCritiqueFinding) -> str:
    return " ".join(
        normalize_text(value).lower()
        for value in (
            finding.finding_key,
            finding.title,
            finding.detail,
            finding.suggested_action,
            " ".join(finding.source_refs),
        )
        if normalize_text(value)
    )


def _matches_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _design_repair_scope(text: str) -> str:
    if _matches_any(
        text,
        (
            "handoff_ambiguity",
            "handoff ambiguity",
            "handoffs ambiguos",
            "payload",
            "criterio de retorno",
            "contrato de handoff",
        ),
    ):
        return "handoff_contract"
    if _matches_any(
        text,
        (
            "loop_risk",
            "riesgo de loop",
            "terminacion",
            "terminación",
            "prevencion de loops",
            "prevención de loops",
            "reintento",
            "ciclo",
        ),
    ):
        return "loop_guard"
    if _matches_any(
        text,
        (
            "missing_evaluation_plan",
            "plan de evaluacion",
            "plan de evaluación",
            "validate",
            "metricas de rendimiento",
            "métricas de rendimiento",
            "rubrica",
            "rúbrica",
        ),
    ):
        return "validate_deferral"
    if _matches_any(
        text,
        (
            "premature_permission_assumption",
            "permisos asumidos",
            "credenciales",
            "permisos exactos",
            "autorizacion exacta",
            "autorización exacta",
        ),
    ):
        return "tools_deferral"
    if _matches_any(
        text,
        (
            "arquitectura de componentes",
            "componentes e interacciones",
            "component interactions",
        ),
    ):
        return "component_contract"
    return ""


def _append_design_sentence(current: str, sentence: str) -> str:
    current = normalize_text(current)
    sentence = normalize_text(sentence)
    if not sentence:
        return current
    if sentence.lower() in current.lower():
        return current
    return f"{current} {sentence}".strip()


def _repair_handoff_contracts(selected_design: DesignAlternative) -> DesignAlternative:
    if not selected_design.handoffs:
        return selected_design
    role_titles = [role.title or role.key for role in selected_design.roles if role.title or role.key]
    default_from = role_titles[0] if role_titles else "Rol origen"
    default_to = role_titles[1] if len(role_titles) > 1 else "Rol destino"
    repaired_handoffs: list[DesignHandoff] = []
    for handoff in selected_design.handoffs:
        repaired_handoffs.append(
            handoff.model_copy(
                update={
                    "from_role": handoff.from_role or default_from,
                    "to_role": handoff.to_role or default_to,
                    "trigger": handoff.trigger
                    or "El paso previo entrega evidencia suficiente para continuar.",
                    "payload": handoff.payload
                    or "Objetivo del paso, entradas aprobadas, evidencia usada, criterio de exito, owner y riesgos abiertos.",
                }
            )
        )
    return selected_design.model_copy(update={"handoffs": repaired_handoffs})


def _repair_selected_design_from_react_findings(
    selected_design: DesignAlternative,
    repair_scopes: set[str],
) -> DesignAlternative:
    if not repair_scopes:
        return selected_design

    selected_design = _repair_handoff_contracts(selected_design)
    projection = selected_design.blueprint_projection
    projection_update: dict[str, object] = {}
    guardrails = list(projection.guardrails)
    tool_implications = list(projection.tool_implications)
    memory_implications = list(projection.memory_implications)
    cost_complexity_implications = list(projection.cost_complexity_implications)
    business_metrics = list(selected_design.business_metrics)
    risk_tradeoffs = list(selected_design.risk_tradeoffs)
    escalation_conditions = list(selected_design.escalation_conditions)
    failure_modes = list(selected_design.failure_modes)
    decision_policy = selected_design.decision_policy

    if "handoff_contract" in repair_scopes or "component_contract" in repair_scopes:
        decision_policy = _append_design_sentence(
            decision_policy,
            "Cada handoff debe declarar owner, payload, criterio de exito, evidencia de entrada y condicion de retorno.",
        )
        escalation_conditions = _normalized_list(
            [
                *escalation_conditions,
                "Escalar a revision humana si un handoff queda sin owner, payload, evidencia o criterio de exito despues del reintento.",
            ]
        )
        memory_implications = _merge_implication_lines(
            memory_implications,
            [
                "checkpoint_resume: persistir owner, payload, criterio de exito y estado por handoff.",
                "handoff_state_tracking: conservar ultima salida estable y evidencia transferida.",
            ],
        )

    if "loop_guard" in repair_scopes:
        decision_policy = _append_design_sentence(
            decision_policy,
            "Todo ciclo de razonamiento, revision u handoff permite un intento inicial y un unico reintento gobernado antes de escalar.",
        )
        guardrails = _normalized_list(
            [
                *guardrails,
                "Cada handoff o ciclo de revision permite un intento inicial y un unico reintento gobernado antes de escalar.",
            ]
        )
        if not any("loop" in item.scenario.lower() or "ciclo" in item.scenario.lower() for item in failure_modes):
            failure_modes.append(
                DesignFailureMode(
                    scenario="Un handoff, revision o ciclo ReAct no converge dentro del presupuesto definido.",
                    retry_strategy="Ejecutar un unico reintento con contexto compacto, causa del fallo y salida esperada.",
                    compensation_strategy="Detener nuevas iteraciones, conservar el ultimo estado aprobado y crear una decision de atencion.",
                    idempotency_notes="Usar identificador estable por handoff y no repetir side effects durante el reintento.",
                )
            )
        risk_tradeoffs = _normalized_list(
            [
                *risk_tradeoffs,
                "La autonomia se conserva, pero los loops quedan acotados por presupuesto de reintentos y checkpoint.",
            ]
        )

    if "validate_deferral" in repair_scopes:
        guardrails = _normalized_list(
            [
                *guardrails,
                "Validate debe comprobar cobertura de requisitos prioritarios, riesgos operativos y coherencia de handoffs antes de Package.",
            ]
        )
        business_metrics = _normalized_list(
            [
                *business_metrics,
                "Cobertura de requisitos prioritarios validada en Validate.",
                "Porcentaje de handoffs con owner, payload y criterio de exito verificable.",
            ]
        )

    if "tools_deferral" in repair_scopes:
        tool_implications = _merge_implication_lines(
            tool_implications,
            [
                "approval_gate: validar permisos, side effects e idempotencia exacta en Tools antes de ejecutar acciones.",
            ],
        )
        if not selected_design.approval_points:
            selected_design = selected_design.model_copy(
                update={
                    "approval_points": [
                        "Compuerta de aprobacion humana para decisiones no delegables y acciones con efectos secundarios."
                    ]
                }
            )
        risk_tradeoffs = _normalized_list(
            [
                *risk_tradeoffs,
                "Design declara la frontera de permisos; Tools cierra contratos, credenciales y side effects concretos.",
            ]
        )

    if repair_scopes & {"loop_guard", "validate_deferral", "handoff_contract"}:
        cost_complexity_implications = _normalized_list(
            [
                *cost_complexity_implications,
                "Estimate debe validar costo, latencia y consumo LLM de los handoffs y reintentos gobernados.",
            ]
        )

    projection_update.update(
        {
            "guardrails": guardrails,
            "tool_implications": tool_implications,
            "memory_implications": memory_implications,
            "cost_complexity_implications": cost_complexity_implications,
        }
    )
    return selected_design.model_copy(
        update={
            "decision_policy": decision_policy,
            "escalation_conditions": escalation_conditions,
            "failure_modes": failure_modes,
            "risk_tradeoffs": risk_tradeoffs,
            "business_metrics": business_metrics,
            "blueprint_projection": projection.model_copy(update=projection_update),
        }
    )


def _case_tool_implications(
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
) -> list[str]:
    text = _case_text(discovery, canvas)
    implications: list[str] = []
    has_non_delegable = bool(
        discovery is not None and discovery.mvp_definition.non_delegable_decisions
    ) or bool(canvas is not None and canvas.agent_profile.human_approvals)
    if has_non_delegable or _contains_case_signal(text, ("aprob", "approval", "autorizar", "humano")):
        implications.append("approval_gate: gobernar decisiones no delegables y acciones sensibles.")
    if _contains_case_signal(text, ("crm", "erp", "ticket", "jira", "zendesk", "api", "base de datos", "postgres")):
        implications.append("read_system_of_record: consultar fuente operacional antes de decidir.")
    if _contains_case_signal(text, ("manual", "politica", "política", "faq", "procedimiento", "document", "knowledge")):
        implications.append("knowledge_retrieval: recuperar conocimiento aprobado con citas y filtros.")
        implications.append("document_ingestion: mantener corpus, lineage y refresh cuando existan fuentes.")
    if _contains_case_signal(text, ("notificar", "alerta", "avisar", "email", "slack", "teams")):
        implications.append("outbound_notification: cerrar el ciclo con el owner o usuario por canal gobernado.")
    if _contains_case_signal(text, ("actualizar", "registrar", "guardar", "crear caso", "create ticket")):
        implications.append("transactional_write: ejecutar escrituras solo con boundary, idempotencia y aprobacion.")
    if _contains_case_signal(text, ("programar", "schedule", "agendar", "cron", "diario", "semanal")):
        implications.append("scheduler: reanudar trabajos programados o asincronos con limites de reintento.")
    return implications


def _case_memory_implications(
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
) -> list[str]:
    text = _case_text(discovery, canvas)
    implications = [
        "decision_traceability: conservar decision, evidencia y owner para cada promocion.",
    ]
    if _contains_case_signal(text, ("document", "manual", "politica", "política", "faq", "knowledge")):
        implications.append("source_ref_grounding: guardar referencias y version de fuente, no cuerpos completos.")
    if _contains_case_signal(text, ("actualizar", "registrar", "aprobar", "autorizar", "side effect")):
        implications.append("approval_checkpoint: persistir estado previo/posterior a acciones sensibles.")
    if _contains_case_signal(text, ("handoff", "escalar", "derivar", "humano")):
        implications.append("handoff_resume_context: retomar desde el owner y payload correcto.")
    return implications


def _business_metrics(discovery: DiscoveryArtifact | None, canvas: CanvasArtifact | None) -> list[str]:
    metrics: list[str] = []
    if discovery is not None:
        metrics.extend(
            [
                discovery.mvp_definition.north_star_metric,
                discovery.operational_baseline.current_time_spent,
                discovery.operational_baseline.current_cost,
            ]
        )
    if canvas is not None:
        metrics.extend([canvas.success_metric, *canvas.agent_profile.success_metrics])
    metrics.extend(
        [
            "Tiempo de ciclo del caso.",
            "Porcentaje de decisiones con evidencia trazable.",
            "Costo LLM por caso cerrado.",
        ]
    )
    return _normalized_list(metrics)


def _enrich_design_alternative(
    alternative: DesignAlternative,
    discovery: DiscoveryArtifact | None = None,
    canvas: CanvasArtifact | None = None,
    *,
    recommendation_role: str | None = None,
) -> DesignAlternative:
    profile = _profile_for_architecture(alternative.architecture)
    canonical_architecture = _canonical_architecture_key(alternative.architecture)
    focus = _case_focus(discovery, canvas)
    archetype_entry = PatternCatalogEntry(label=_profile_text(profile, "agent_archetype"))
    pattern_entry = PatternCatalogEntry(label=_profile_text(profile, "pattern_family"))
    if discovery is not None and canvas is not None:
        archetype_entry, pattern_entry = _catalog_intelligence_for_architecture(
            canonical_architecture,
            discovery,
            canvas,
        )
    tool_implications = _merge_implication_lines(
        _profile_list(profile, "tool_implications"),
        _case_tool_implications(discovery, canvas),
        alternative.tool_implications,
    )
    memory_implications = _merge_implication_lines(
        _profile_list(profile, "memory_implications"),
        _case_memory_implications(discovery, canvas),
        alternative.memory_implications,
    )
    risk_tradeoffs = _normalized_list(
        [
            *_profile_list(profile, "risk_tradeoffs"),
            *alternative.risk_tradeoffs,
            *alternative.tradeoffs[:2],
        ]
    )
    business_metrics = _normalized_list([*_business_metrics(discovery, canvas), *alternative.business_metrics])
    projection = alternative.blueprint_projection
    projection_update = {
        "architecture": projection.architecture or alternative.architecture,
        "reasoning_pattern": projection.reasoning_pattern or alternative.reasoning_pattern,
        "tool_implications": _merge_implication_lines(projection.tool_implications, tool_implications),
        "memory_strategy": projection.memory_strategy or _profile_text(profile, "memory_strategy"),
        "memory_implications": _merge_implication_lines(projection.memory_implications, memory_implications),
        "cost_complexity_implications": _normalized_list(
            [
                f"Costo relativo: {alternative.relative_cost or 'medium'}",
                f"Complejidad operacional: {alternative.operational_complexity or 'medium'}",
                f"Mantenibilidad: {alternative.maintainability or 'medium'}",
            ]
        ),
    }
    return alternative.model_copy(
        update={
            "recommendation_role": alternative.recommendation_role or recommendation_role or "alternative",
            "agent_archetype": alternative.agent_archetype or archetype_entry.label or _profile_text(profile, "agent_archetype"),
            "pattern_family": alternative.pattern_family or pattern_entry.label or _profile_text(profile, "pattern_family"),
            "business_fit": alternative.business_fit
            or f"{_profile_text(profile, 'business_fit')} Objetivo conectado: {focus}.",
            "value_hypothesis": alternative.value_hypothesis or _profile_text(profile, "value_hypothesis"),
            "operational_model": alternative.operational_model or _profile_text(profile, "operational_model"),
            "why_recommended": alternative.why_recommended or _profile_text(profile, "why_recommended"),
            "why_not_simpler": alternative.why_not_simpler or _profile_text(profile, "why_not_simpler"),
            "why_not_more_complex": alternative.why_not_more_complex or _profile_text(profile, "why_not_more_complex"),
            "fit_score": _normalize_design_fit_score(alternative.fit_score),
            "tool_implications": tool_implications,
            "memory_implications": memory_implications,
            "risk_tradeoffs": risk_tradeoffs,
            "business_metrics": business_metrics,
            "fit_rationale": _normalized_list(
                [
                    *alternative.fit_rationale,
                    archetype_entry.summary,
                    pattern_entry.summary,
                ]
            ),
            "evidence_refs": _normalized_list(
                [
                    *alternative.evidence_refs,
                    *DESIGN_INTELLIGENCE_EVIDENCE_REFS,
                    f"catalog.agent_archetype.{archetype_entry.key}" if archetype_entry.key else "",
                    f"catalog.pattern_family.{pattern_entry.key}" if pattern_entry.key else "",
                ]
            ),
            "coordination_model": alternative.coordination_model or canonical_architecture,
            "blueprint_projection": projection.model_copy(update=projection_update),
        }
    )


def _assign_recommendation_roles(
    alternatives: list[DesignAlternative],
    recommended_key: str | None = None,
) -> list[DesignAlternative]:
    if not alternatives:
        return []
    complexity_rank = {"low": 0, "medium": 1, "high": 2}
    recommended_key = recommended_key or max(alternatives, key=lambda item: item.fit_score).alternative_key
    simplest_key = min(
        alternatives,
        key=lambda item: (
            complexity_rank.get((item.operational_complexity or "medium").lower(), 1),
            complexity_rank.get((item.relative_cost or "medium").lower(), 1),
            -item.fit_score,
        ),
    ).alternative_key
    most_powerful_key = max(
        alternatives,
        key=lambda item: (
            complexity_rank.get((item.operational_complexity or "medium").lower(), 1),
            item.fit_score,
        ),
    ).alternative_key
    updated: list[DesignAlternative] = []
    for item in alternatives:
        role = "alternative"
        if item.alternative_key == recommended_key:
            role = "recommended"
        elif item.alternative_key == simplest_key:
            role = "simpler_baseline"
        elif item.alternative_key == most_powerful_key:
            role = "powerful_option"
        updated.append(item.model_copy(update={"recommendation_role": role}))
    return updated


def _significant_words(value: str) -> set[str]:
    stopwords = {
        "agente",
        "agents",
        "agent",
        "para",
        "como",
        "cuando",
        "porque",
        "desde",
        "sobre",
        "entre",
        "esta",
        "este",
        "estos",
        "estas",
        "with",
        "that",
        "this",
        "the",
    }
    tokens = [
        token.strip(".,;:()[]{}'\"").lower()
        for token in normalize_text(value).replace("/", " ").replace("-", " ").split()
    ]
    return {token for token in tokens if len(token) >= 5 and token not in stopwords}


def _design_explanation_quality_gaps(
    selected_design: DesignAlternative,
    discovery: DiscoveryArtifact,
    definition: RequirementsDefinitionOutput,
) -> list[str]:
    gaps: list[str] = []
    business_fit = normalize_text(selected_design.business_fit)
    why_recommended = normalize_text(selected_design.why_recommended)
    combined_explanation = f"{business_fit} {why_recommended}"
    if len(business_fit) < 48:
        gaps.append("business_fit demasiado corto para explicar relacion con el negocio.")
    if len(why_recommended) < 42:
        gaps.append("why_recommended no justifica suficientemente la seleccion.")
    project_signal_text = " ".join(
        [
            discovery.problem_statement,
            discovery.current_process,
            discovery.desired_outcome,
            " ".join(discovery.constraints),
            " ".join(discovery.mvp_definition.v1_scope),
            " ".join(item.requirement for item in definition.functional_requirements[:8]),
            " ".join(item.requirement for item in definition.non_functional_requirements[:8]),
        ]
    )
    signal_words = _significant_words(project_signal_text)
    explanation_words = _significant_words(combined_explanation)
    if signal_words and not (signal_words & explanation_words):
        gaps.append("la explicacion no referencia senales concretas del proyecto o requisitos.")
    if not selected_design.tool_implications and not selected_design.blueprint_projection.tool_implications:
        gaps.append("la alternativa no declara impacto hacia Tools.")
    if not selected_design.memory_implications and not selected_design.blueprint_projection.memory_implications:
        gaps.append("la alternativa no declara impacto hacia Memory.")
    generic_phrases = (
        "cubre el alcance",
        "mejor alternativa",
        "opcion adecuada",
        "opción adecuada",
        "cumple los requisitos",
    )
    generic_hits = sum(1 for phrase in generic_phrases if phrase in combined_explanation.lower())
    if generic_hits >= 2 and len(explanation_words) < 14:
        gaps.append("la explicacion parece generica y no una decision de producto.")
    return gaps


def _score_requirement_for_alternative(
    requirement: dict[str, str],
    alternative: DesignAlternative,
) -> DesignFitAlternativeScore:
    detail = f"{requirement['title']} {requirement['detail']}".lower()
    score = int(round(alternative.fit_score))
    rationale = ["Parte de la seleccion base por fit del catalogo."]
    if "aprob" in detail or "humano" in detail:
        if alternative.approval_points:
            score += 8
            rationale.append("Incluye approval points explicitos.")
        else:
            score -= 20
            rationale.append("No deja checkpoints humanos visibles.")
    if any(token in detail for token in ("paralel", "simultan", "multi", "especialist")):
        if alternative.architecture in {"supervisor_with_subagents", "router_parallel"}:
            score += 10
            rationale.append("La topologia soporta especializacion o paralelismo cuando hace falta.")
        else:
            score -= 12
            rationale.append("Puede quedarse corta si el caso exige especializacion real.")
    if any(token in detail for token in ("costo", "latencia", "mantenimiento", "simple")):
        if alternative.operational_complexity == "low":
            score += 6
            rationale.append("La alternativa minimiza complejidad operacional.")
        elif alternative.operational_complexity == "high":
            score -= 10
            rationale.append("La alternativa incrementa costo y complejidad.")
    if any(token in detail for token in ("seguridad", "riesgo", "trazabilidad", "auditoria", "cumpl")):
        if alternative.security_notes:
            score += 6
            rationale.append("Declara limites de seguridad y trazabilidad.")
    if requirement["priority"] == "high" and alternative.operational_complexity == "high":
        score -= 4
        rationale.append("La complejidad extra debe justificarse mejor para un requisito prioritario.")
    bounded = max(0, min(100, score))
    coverage_status = "covered" if bounded >= 70 else "partial" if bounded >= 50 else "gap"
    return DesignFitAlternativeScore(
        alternative_key=alternative.alternative_key,
        score=bounded,
        coverage_status=coverage_status,
        rationale=" ".join(rationale),
    )


def _selected_requirements_coverage(
    fit_matrix: list[DesignFitMatrixEntry],
    selected_alternative_key: str,
) -> list[DesignRequirementCoverageEntry]:
    coverage: list[DesignRequirementCoverageEntry] = []
    for row in fit_matrix:
        selected_score = next((item for item in row.scores if item.alternative_key == selected_alternative_key), None)
        if selected_score is None:
            continue
        coverage.append(
            DesignRequirementCoverageEntry(
                requirement_key=row.requirement_key,
                requirement_title=row.requirement_title,
                category=row.category,
                priority=row.priority,
                coverage_status=selected_score.coverage_status,
                rationale=selected_score.rationale,
                source_refs=[row.requirement_key],
            )
        )
    return coverage


def _find_selected_alternative(
    alternatives: list[DesignAlternative],
    selected_key: str,
) -> DesignAlternative | None:
    for item in alternatives:
        if item.alternative_key == selected_key:
            return item
    return alternatives[0] if alternatives else None


def _build_fallback_alternatives(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
) -> list[DesignAlternative]:
    architecture_catalog = sorted(
        build_architecture_catalog(discovery, canvas),
        key=lambda item: item.fit_score,
        reverse=True,
    )
    reasoning_catalog = sorted(
        build_reasoning_catalog(discovery, canvas),
        key=lambda item: item.fit_score,
        reverse=True,
    )
    alternatives: list[DesignAlternative] = []
    selected_architectures: set[str] = set()
    safety_checks = derive_safety_checks(discovery)
    guardrails = default_guardrails(discovery)
    approvals = _approval_points(discovery, canvas)
    escalations = _escalation_conditions(discovery)

    for architecture in architecture_catalog:
        if architecture.key in selected_architectures:
            continue
        reasoning = _reasoning_for_architecture(architecture, reasoning_catalog)
        archetype, pattern_family = _catalog_intelligence_for_architecture(architecture.key, discovery, canvas)
        topology, roles, handoffs = _topology_for_architecture(architecture.key)
        complexity, cost, maintainability = _operating_profile(architecture.key)
        selected_architectures.add(architecture.key)
        fit_score = _business_technical_fit_score(
            architecture,
            reasoning,
            archetype,
            pattern_family,
            discovery,
            canvas,
        )
        alternative = DesignAlternative(
                alternative_key=architecture.key,
                label=architecture.label,
                architecture=architecture.key,
                reasoning_pattern=reasoning.key,
                coordination_model=architecture.key,
                summary=f"{architecture.summary} Patron cognitivo sugerido: {reasoning.summary}",
                topology=topology,
                roles=roles,
                handoffs=handoffs,
                approval_points=approvals,
                decision_policy=(
                    "Priorizar contexto aprobado, evitar delegaciones innecesarias y escalar decisiones no delegables."
                ),
                escalation_conditions=escalations,
                concurrency_strategy=(
                    "Secuencial por defecto; solo habilitar concurrencia si reduce tiempo sin perder trazabilidad."
                    if architecture.key not in {"router_parallel"}
                    else "Paralelismo controlado con agregacion final centralizada y limites por tarea."
                ),
                failure_modes=_failure_modes(architecture.key),
                security_notes=_normalized_list(
                    [
                        "Las decisiones no delegables permanecen bajo aprobacion humana.",
                        "El agente solo consume contexto aprobado y evidencias trazables.",
                        "Toda accion sensible debe registrar rationale y source refs.",
                    ]
                ),
                operational_complexity=complexity,
                relative_cost=cost,
                maintainability=maintainability,
                tradeoffs=_normalized_list([*architecture.tradeoffs, *reasoning.tradeoffs[:2]]),
                assumptions=_normalized_list(
                    [
                        "Discover y Define ya fueron aprobados.",
                        "La seleccion de tools y memoria ocurrira en etapas posteriores.",
                    ]
                ),
                fit_score=fit_score,
                fit_rationale=_normalized_list(
                    [
                        *architecture.use_when[:3],
                        *reasoning.use_when[:2],
                        f"Scoring negocio-tecnico: arquitectura {architecture.fit_score}, razonamiento {reasoning.fit_score}, arquetipo {archetype.fit_score}, patron {pattern_family.fit_score}.",
                    ]
                ),
                evidence_refs=[
                    f"catalog.architecture.{architecture.key}",
                    f"catalog.reasoning.{reasoning.key}",
                    f"catalog.agent_archetype.{archetype.key}",
                    f"catalog.pattern_family.{pattern_family.key}",
                ],
                blueprint_projection=DesignBlueprintProjection(
                    architecture=architecture.key,
                    reasoning_pattern=reasoning.key,
                    safety_checks=safety_checks,
                    guardrails=guardrails,
                    narrative=(
                        f"Se recomienda {architecture.label} con {reasoning.label} para mantener cobertura "
                        f"contra el alcance aprobado sin mezclar Tools ni Memory antes de tiempo."
                    ),
                ),
            )
        alternatives.append(_enrich_design_alternative(alternative, discovery, canvas))
        if len(alternatives) == 3:
            break
    return _assign_recommendation_roles(alternatives)


def build_design_recommendation_artifact(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    definition: RequirementsDefinitionOutput,
) -> DesignRecommendationArtifact:
    alternatives = _build_fallback_alternatives(discovery, canvas)
    fit_matrix: list[DesignFitMatrixEntry] = []
    for requirement in _definition_requirements(definition)[:18]:
        fit_matrix.append(
            DesignFitMatrixEntry(
                requirement_key=requirement["key"],
                requirement_title=requirement["title"],
                category=requirement["category"],
                priority=requirement["priority"],
                scores=[_score_requirement_for_alternative(requirement, alternative) for alternative in alternatives],
            )
        )
    recommended = max(alternatives, key=lambda item: item.fit_score, default=None)
    recommended_key = recommended.alternative_key if recommended is not None else ""
    requirements_coverage = _selected_requirements_coverage(fit_matrix, recommended_key)
    return DesignRecommendationArtifact(
        alternatives=alternatives,
        fit_matrix=fit_matrix,
        recommended_alternative_key=recommended_key,
        selected_design=recommended,
        decision_rationale=(
            "La recomendacion inicial nace del catalogo gobernado, balanceando cobertura, costo y complejidad."
        ),
        requirements_coverage=requirements_coverage,
        evidence_refs=_normalized_list(
            [ref for alternative in alternatives for ref in alternative.evidence_refs] + list(definition.evidence_refs)
        ),
        confidence=DesignRecommendationConfidence(
            overall=0.62 if alternatives else 0.0,
            band="medium" if alternatives else "low",
            rationale="La base viene del catalogo gobernado y debe enriquecerse con el pase arquitecto/critico.",
        ),
        open_questions=_normalized_list([item.question for item in definition.open_questions[:5]]),
        missing_information=_normalized_list(
            [item.question for item in definition.open_questions if _definition_question_blocks_design(item)]
        ),
        summary="Comparador inicial de alternativas construido desde catalogos gobernados y Definition aprobado.",
    )


def _auto_reconcile_design_artifact(
    artifact: DesignRecommendationArtifact,
    discovery: DiscoveryArtifact | None = None,
) -> DesignRecommendationArtifact:
    alternatives = [
        _enrich_design_alternative(item, discovery)
        for item in artifact.alternatives
    ]
    if not alternatives:
        return artifact
    alternatives = _assign_recommendation_roles(
        alternatives,
        artifact.recommended_alternative_key or alternatives[0].alternative_key,
    )

    alternatives_by_key = {item.alternative_key: item for item in alternatives}
    alternatives_by_arch = {item.architecture: item for item in alternatives}

    recommended_key = artifact.recommended_alternative_key or alternatives[0].alternative_key
    selected_design = _find_selected_alternative(alternatives, recommended_key) or alternatives[0]

    rationale = artifact.decision_rationale or ""
    summary = artifact.summary or ""
    rationale_lower = f"{rationale} {summary}".lower()

    arch_keywords = [
        ("supervisor_with_subagents", ["router-worker", "supervisor", "jerárquica", "jerarquica"]),
        ("handoffs", ["handoff", "handoffs secuenciales"]),
        ("single_agent", ["single agent", "agente único", "agente unico"]),
        ("single_agent_with_skills", ["single agent with skills", "agente con skills"]),
        ("plan_and_execute", ["plan-and-execute", "plan and execute"]),
    ]

    justified_arch = None
    for arch_key, keywords in arch_keywords:
        if any(kw in rationale_lower for kw in keywords):
            justified_arch = arch_key
            break

    if justified_arch and justified_arch in alternatives_by_arch:
        justified_alt = alternatives_by_arch[justified_arch]
        if selected_design.architecture != justified_arch:
            if justified_alt.fit_score >= selected_design.fit_score - 8:
                recommended_key = justified_alt.alternative_key
                selected_design = justified_alt
            else:
                rationale = (
                    f"Se selecciona {selected_design.label} ({selected_design.architecture}) "
                    f"como la opción óptima para el alcance del MVP, balanceando cobertura funcional, "
                    f"costo operativo y gobernanza."
                )

    if discovery is not None and discovery.mvp_definition and discovery.mvp_definition.non_delegable_decisions:
        if not selected_design.approval_points:
            selected_design = selected_design.model_copy(
                update={
                    "approval_points": [
                        "Compuerta de aprobación humana para decisiones no delegables y acciones con efectos secundarios."
                    ]
                }
            )

    arch_label = selected_design.label or selected_design.architecture
    reasoning_label = selected_design.blueprint_projection.reasoning_pattern or selected_design.reasoning_pattern
    narrative = (
        f"Se recomienda {arch_label} con {reasoning_label} para balancear cobertura, "
        f"seguridad y costo sin sobredimensionar la solución."
    )

    # Auto-remediate empty design decisions, tooling principles, and memory strategy
    projection = selected_design.blueprint_projection
    proj_update = {
        "architecture": selected_design.architecture,
        "reasoning_pattern": selected_design.reasoning_pattern,
        "narrative": projection.narrative or narrative,
    }
    if not getattr(projection, "memory_strategy", None):
        proj_update["memory_strategy"] = "session_memory_with_checkpoints"

    selected_design = selected_design.model_copy(
        update={"blueprint_projection": projection.model_copy(update=proj_update)}
    )

    # Auto-remediate routine handoff approvals in selected_design
    if selected_design.handoffs:
        remediated_handoffs = []
        for h in selected_design.handoffs:
            # Routine handoffs between automated roles should not require human approval unless explicitly an escalation
            target = (h.to_role or "").lower()
            trigger = (h.trigger or "").lower()
            is_escalation = "human" in target or "supervisor" in target or "escal" in trigger or "ambig" in trigger
            if not is_escalation and getattr(h, "approval_required", False):
                h = h.model_copy(update={"approval_required": False})
            remediated_handoffs.append(h)
        selected_design = selected_design.model_copy(update={"handoffs": remediated_handoffs})

    repair_scopes: set[str] = set()
    clean_findings: list[DesignCritiqueFinding] = []
    for finding in artifact.critic_findings:
        title_lower = (finding.title or "").lower()
        detail_lower = (finding.detail or "").lower()
        key_lower = (finding.finding_key or "").lower()
        refs_lower = " ".join(finding.source_refs or []).lower()
        combined = f"{title_lower} {detail_lower} {key_lower} {refs_lower}"
        repair_scope = _design_repair_scope(_finding_text(finding))
        if repair_scope:
            repair_scopes.add(repair_scope)
            continue

        is_approved_definition_debt = (
            finding.severity == "blocking"
            and (
                "approved_definition" in refs_lower
                or "pregunta bloqueante aprobada" in combined
                or "aprobada con deuda" in combined
                or "aprobada sin resolver" in combined
            )
        )
        if is_approved_definition_debt:
            clean_findings.append(
                finding.model_copy(
                    update={
                        "severity": "warning",
                        "detail": _append_design_sentence(
                            finding.detail,
                            "Tratado como deuda aprobada heredada de Definition: debe permanecer visible, pero no bloquear Design nuevamente.",
                        ),
                    }
                )
            )
            continue

        is_contradiction = (
            "inconsistencia" in combined
            or "contradiction" in combined
            or ("router-worker" in combined and "handoffs" in combined)
            or ("supervisor" in combined and "handoffs" in combined)
        )
        is_routine_handoff_block = (
            "handoff con aprobaci" in combined
            or "bloquea la resoluci" in combined
            or "approval_required" in combined
        )
        is_resolved_approvals = (
            "missing-approvals" in key_lower
            or "approval points" in title_lower
        ) and bool(selected_design.approval_points)
        is_id_discrepancy = (
            "discrepancia de identificadores" in combined
            or "identificadores y categor" in combined
        )
        is_infra_or_benchmark = any(
            kw in combined
            for kw in (
                "calibración matemática",
                "calibracion matematica",
                "datos históricos",
                "datos historicos",
                "mecanismos de integración",
                "mecanismos de integracion",
                "design_decisions",
                "tooling_principles",
                "memory_strategy",
                "benchmark de latencia",
                "filtro de sanitización",
                "filtro de sanitizacion",
                "volumen cuantitativo",
                "taxonomía completa",
                "taxonomia completa",
            )
        )

        if is_contradiction or is_routine_handoff_block or is_resolved_approvals or is_id_discrepancy or is_infra_or_benchmark:
            # Auto-remediated by self-healing / deferred to ACP
            continue
        clean_findings.append(finding)

    # Harmonize requirements_coverage keys with fit_matrix if needed
    requirements_coverage = list(artifact.requirements_coverage)
    if artifact.fit_matrix and requirements_coverage:
        fit_keys = [entry.requirement_key for entry in artifact.fit_matrix]
        # If requirements_coverage uses generic REQ-xx, map them to fit_matrix keys
        if len(requirements_coverage) <= len(fit_keys):
            updated_coverage = []
            for i, cov in enumerate(requirements_coverage):
                if cov.requirement_key not in fit_keys and i < len(fit_keys):
                    cov = cov.model_copy(update={"requirement_key": fit_keys[i]})
                updated_coverage.append(cov)
            requirements_coverage = updated_coverage

    def _is_design_noise(text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            kw in lower
            for kw in (
                "calibración matemática",
                "calibracion matematica",
                "datos históricos",
                "datos historicos",
                "mecanismos de integración",
                "mecanismos de integracion",
                "design_decisions",
                "tooling_principles",
                "memory_strategy",
                "benchmark de latencia",
                "filtro de sanitización",
                "filtro de sanitizacion",
                "volumen cuantitativo",
                "taxonomía completa",
                "taxonomia completa",
            )
        )

    cleaned_missing_info: list[str] = []
    for item in artifact.missing_information or []:
        repair_scope = _design_repair_scope(normalize_text(item).lower())
        if repair_scope:
            repair_scopes.add(repair_scope)
            continue
        if _is_design_noise(item):
            continue
        cleaned_missing_info.append(item)

    selected_design = _repair_selected_design_from_react_findings(selected_design, repair_scopes)
    updated_alternatives = [
        selected_design if item.alternative_key == selected_design.alternative_key else item
        for item in alternatives
    ]
    remediation_summary = artifact.remediation_summary
    if repair_scopes:
        remediation_summary = _append_design_sentence(
            remediation_summary,
            "Auto-reconciliacion Design ReAct: se repararon observaciones de "
            + ", ".join(sorted(repair_scopes))
            + " usando contratos, guardrails y deferrals hacia Tools, Memory, Validate o Estimate.",
        )

    return artifact.model_copy(
        update={
            "alternatives": updated_alternatives,
            "recommended_alternative_key": recommended_key,
            "selected_design": selected_design,
            "decision_rationale": rationale or artifact.decision_rationale,
            "critic_findings": clean_findings,
            "missing_information": cleaned_missing_info,
            "remediation_summary": remediation_summary,
        }
    )


def downgrade_design_recommendation_to_legacy(
    artifact: DesignRecommendationArtifact,
) -> DesignRecommendationArtifact:
    """Return a legacy-compatible Design projection for controlled rollout rollback."""

    alternatives: list[DesignAlternative] = []
    for alternative in artifact.alternatives:
        projection = alternative.blueprint_projection.model_copy(
            update={
                "tool_implications": [],
                "memory_strategy": "",
                "memory_implications": [],
                "cost_complexity_implications": [],
            }
        )
        alternatives.append(
            alternative.model_copy(
                update={
                    "recommendation_role": "",
                    "agent_archetype": "",
                    "pattern_family": "",
                    "business_fit": "",
                    "value_hypothesis": "",
                    "operational_model": "",
                    "why_recommended": "",
                    "why_not_simpler": "",
                    "why_not_more_complex": "",
                    "tool_implications": [],
                    "memory_implications": [],
                    "risk_tradeoffs": [],
                    "business_metrics": [],
                    "blueprint_projection": projection,
                }
            )
        )
    selected_design = _find_selected_alternative(alternatives, artifact.recommended_alternative_key)
    if selected_design is None and alternatives:
        selected_design = alternatives[0]
    summary_suffix = "Design Intelligence v2 desactivado por feature flag; se emitio proyeccion legacy-compatible."
    summary = artifact.summary
    if summary_suffix not in summary:
        summary = f"{summary} {summary_suffix}".strip()
    return artifact.model_copy(
        update={
            "alternatives": alternatives,
            "selected_design": selected_design,
            "summary": summary,
        }
    )


_DESIGN_INTELLIGENCE_V2_FIELDS = (
    "recommendation_role",
    "agent_archetype",
    "pattern_family",
    "business_fit",
    "value_hypothesis",
    "operational_model",
    "why_recommended",
    "why_not_simpler",
    "why_not_more_complex",
    "tool_implications",
    "memory_implications",
    "risk_tradeoffs",
    "business_metrics",
)


def _has_shadow_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def build_design_intelligence_shadow_report(
    artifact: DesignRecommendationArtifact,
) -> dict[str, object]:
    """Summarize the v2 fields that would be hidden by a legacy-compatible rollback."""

    legacy_projection = downgrade_design_recommendation_to_legacy(artifact)
    field_presence = {field: 0 for field in _DESIGN_INTELLIGENCE_V2_FIELDS}
    alternative_reports: list[dict[str, object]] = []
    changed_alternative_count = 0

    for index, alternative in enumerate(artifact.alternatives):
        legacy_alternative = (
            legacy_projection.alternatives[index]
            if index < len(legacy_projection.alternatives)
            else None
        )
        populated_fields = [
            field
            for field in _DESIGN_INTELLIGENCE_V2_FIELDS
            if _has_shadow_value(getattr(alternative, field, None))
        ]
        for field in populated_fields:
            field_presence[field] += 1

        projection_fields = [
            field
            for field in (
                "blueprint_projection.tool_implications",
                "blueprint_projection.memory_strategy",
                "blueprint_projection.memory_implications",
                "blueprint_projection.cost_complexity_implications",
            )
            if _has_shadow_value(
                getattr(
                    alternative.blueprint_projection,
                    field.removeprefix("blueprint_projection."),
                    None,
                )
            )
        ]
        if populated_fields or projection_fields:
            changed_alternative_count += 1
        alternative_reports.append(
            {
                "alternative_key": alternative.alternative_key,
                "legacy_alternative_key": legacy_alternative.alternative_key if legacy_alternative is not None else "",
                "architecture": alternative.architecture,
                "reasoning_pattern": alternative.reasoning_pattern,
                "v2_fields_hidden_by_legacy": populated_fields,
                "projection_fields_hidden_by_legacy": projection_fields,
            }
        )

    selected = artifact.selected_design
    return {
        "contract_version": "design-intelligence-shadow.v1",
        "mode": "v2_active_legacy_projection_shadow",
        "changed_alternative_count": changed_alternative_count,
        "alternative_count": len(artifact.alternatives),
        "field_presence": field_presence,
        "selected_alternative_key": selected.alternative_key if selected is not None else "",
        "selected_architecture": selected.architecture if selected is not None else "",
        "selected_reasoning_pattern": selected.reasoning_pattern if selected is not None else "",
        "selected_agent_archetype": selected.agent_archetype if selected is not None else "",
        "selected_pattern_family": selected.pattern_family if selected is not None else "",
        "legacy_summary": legacy_projection.summary,
        "alternatives": alternative_reports,
    }


def merge_llm_design_recommendation(
    artifact: DesignRecommendationArtifact,
    llm_output: AgentDesignProposalOutput | None,
    critique_output: DesignCritiqueOutput | None = None,
) -> DesignRecommendationArtifact:
    if llm_output is None:
        if critique_output is not None:
            merged = merge_design_critique(artifact, critique_output)
            return _auto_reconcile_design_artifact(merged)
        return _auto_reconcile_design_artifact(artifact)

    alternatives_by_key = {item.alternative_key: item for item in artifact.alternatives}
    updated_alternatives: list[DesignAlternative] = []
    for base_alternative in artifact.alternatives:
        candidate = next(
            (
                item
                for item in llm_output.alternatives
                if item.alternative_key == base_alternative.alternative_key or item.architecture == base_alternative.architecture
            ),
            None,
        )
        if candidate is None:
            updated_alternatives.append(base_alternative)
            continue
        updated_alternatives.append(
            base_alternative.model_copy(
                update={
                    "label": candidate.label or base_alternative.label,
                    "recommendation_role": candidate.recommendation_role or base_alternative.recommendation_role,
                    "agent_archetype": candidate.agent_archetype or base_alternative.agent_archetype,
                    "pattern_family": candidate.pattern_family or base_alternative.pattern_family,
                    "summary": candidate.summary or base_alternative.summary,
                    "business_fit": candidate.business_fit or base_alternative.business_fit,
                    "value_hypothesis": candidate.value_hypothesis or base_alternative.value_hypothesis,
                    "operational_model": candidate.operational_model or base_alternative.operational_model,
                    "why_recommended": candidate.why_recommended or base_alternative.why_recommended,
                    "why_not_simpler": candidate.why_not_simpler or base_alternative.why_not_simpler,
                    "why_not_more_complex": candidate.why_not_more_complex or base_alternative.why_not_more_complex,
                    "topology": candidate.topology or base_alternative.topology,
                    "roles": candidate.roles or base_alternative.roles,
                    "handoffs": candidate.handoffs or base_alternative.handoffs,
                    "approval_points": candidate.approval_points or base_alternative.approval_points,
                    "decision_policy": candidate.decision_policy or base_alternative.decision_policy,
                    "escalation_conditions": candidate.escalation_conditions or base_alternative.escalation_conditions,
                    "concurrency_strategy": candidate.concurrency_strategy or base_alternative.concurrency_strategy,
                    "failure_modes": candidate.failure_modes or base_alternative.failure_modes,
                    "security_notes": candidate.security_notes or base_alternative.security_notes,
                    "operational_complexity": candidate.operational_complexity or base_alternative.operational_complexity,
                    "relative_cost": candidate.relative_cost or base_alternative.relative_cost,
                    "maintainability": candidate.maintainability or base_alternative.maintainability,
                    "tradeoffs": candidate.tradeoffs or base_alternative.tradeoffs,
                    "assumptions": candidate.assumptions or base_alternative.assumptions,
                    "fit_score": candidate.fit_score or base_alternative.fit_score,
                    "fit_rationale": candidate.fit_rationale or base_alternative.fit_rationale,
                    "tool_implications": _merge_implication_lines(
                        base_alternative.tool_implications,
                        candidate.tool_implications,
                    ),
                    "memory_implications": _merge_implication_lines(
                        base_alternative.memory_implications,
                        candidate.memory_implications,
                    ),
                    "risk_tradeoffs": _normalized_list(
                        [*base_alternative.risk_tradeoffs, *candidate.risk_tradeoffs]
                    ),
                    "business_metrics": _normalized_list(
                        [*base_alternative.business_metrics, *candidate.business_metrics]
                    ),
                    "evidence_refs": _normalized_list([*base_alternative.evidence_refs, *candidate.evidence_refs]),
                    "blueprint_projection": base_alternative.blueprint_projection.model_copy(
                        update={
                            "narrative": candidate.blueprint_projection.narrative
                            or base_alternative.blueprint_projection.narrative,
                            "tool_implications": _merge_implication_lines(
                                base_alternative.blueprint_projection.tool_implications,
                                candidate.blueprint_projection.tool_implications,
                            ),
                            "memory_strategy": candidate.blueprint_projection.memory_strategy
                            or base_alternative.blueprint_projection.memory_strategy,
                            "memory_implications": _merge_implication_lines(
                                base_alternative.blueprint_projection.memory_implications,
                                candidate.blueprint_projection.memory_implications,
                            ),
                            "cost_complexity_implications": _normalized_list(
                                [
                                    *base_alternative.blueprint_projection.cost_complexity_implications,
                                    *candidate.blueprint_projection.cost_complexity_implications,
                                ]
                            ),
                        }
                    ),
                }
            )
        )
    recommended_key = llm_output.recommended_alternative_key or artifact.recommended_alternative_key
    if recommended_key not in alternatives_by_key and recommended_key:
        recommended_key = artifact.recommended_alternative_key
    merged = artifact.model_copy(
        update={
            "alternatives": updated_alternatives,
            "recommended_alternative_key": recommended_key,
            "selected_design": _find_selected_alternative(updated_alternatives, recommended_key),
            "decision_rationale": llm_output.decision_rationale or artifact.decision_rationale,
            "requirements_coverage": llm_output.requirements_coverage or artifact.requirements_coverage,
            "evidence_refs": _normalized_list([*artifact.evidence_refs, *llm_output.evidence_refs]),
            "confidence": DesignRecommendationConfidence(
                overall=llm_output.confidence or artifact.confidence.overall,
                band=artifact.confidence.band,
                rationale=artifact.confidence.rationale,
            ),
            "open_questions": _normalized_list([*artifact.open_questions, *llm_output.open_questions]),
            "guided_questions": _merge_guided_questions(
                artifact.guided_questions,
                llm_output.guided_questions,
                stage_scope="design",
            ),
            "summary": llm_output.summary or artifact.summary,
        }
    )
    if critique_output is not None:
        merged = merge_design_critique(merged, critique_output)
    return _auto_reconcile_design_artifact(merged)


def merge_design_critique(
    artifact: DesignRecommendationArtifact,
    critique_output: DesignCritiqueOutput | None,
) -> DesignRecommendationArtifact:
    if critique_output is None:
        return artifact
    findings = [
        DesignCritiqueFinding(
            finding_key=item.finding_key,
            title=item.title,
            severity=item.severity,
            detail=item.detail,
            suggested_action=item.suggested_action,
            source_refs=item.source_refs,
        )
        for item in critique_output.findings
    ]
    merged = artifact.model_copy(
        update={
            "critic_findings": findings,
            "remediation_summary": critique_output.summary or artifact.remediation_summary,
            "missing_information": _normalized_list([*artifact.missing_information, *critique_output.missing_evidence]),
        }
    )
    return _auto_reconcile_design_artifact(merged)


def evaluate_design_recommendation_artifact(
    artifact: DesignRecommendationArtifact,
    discovery: DiscoveryArtifact,
    definition: RequirementsDefinitionOutput,
) -> DesignRecommendationArtifact:
    reconciled = _auto_reconcile_design_artifact(artifact, discovery=discovery)
    alternatives = reconciled.alternatives[:3]
    recommended_key = reconciled.recommended_alternative_key or (alternatives[0].alternative_key if alternatives else "")
    selected_design = _find_selected_alternative(alternatives, recommended_key)
    findings = list(reconciled.critic_findings)

    if not alternatives:
        findings.append(
            DesignCritiqueFinding(
                finding_key="design-no-alternatives",
                title="No hay alternativas comparables",
                severity="blocking",
                detail="Design requiere al menos una alternativa valida antes de aprobar.",
                suggested_action="Regenerar la propuesta con contexto aprobado o revisar los catalogos.",
                source_refs=["design.alternatives"],
            )
        )

    if selected_design is not None:
        explanation_gaps = _design_explanation_quality_gaps(selected_design, discovery, definition)
        if explanation_gaps:
            findings.append(
                DesignCritiqueFinding(
                    finding_key="design-weak-business-explanation",
                    title="La recomendacion necesita mejor explicacion de negocio",
                    severity="warning",
                    detail=" ".join(explanation_gaps),
                    suggested_action=(
                        "Regenerar o enriquecer Design conectando la alternativa con objetivos, requisitos, "
                        "metricas e impacto hacia Tools y Memory."
                    ),
                    source_refs=["design.selected_design", "definition.requirements"],
                )
            )
        high_priority_gaps = [
            item
            for item in _selected_requirements_coverage(reconciled.fit_matrix, selected_design.alternative_key)
            if item.priority == "high" and item.coverage_status == "gap"
        ]
        if high_priority_gaps:
            findings.append(
                DesignCritiqueFinding(
                    finding_key="design-high-priority-gap",
                    title="La alternativa recomendada no cubre requisitos prioritarios",
                    severity="warning",
                    detail=(
                        "Persisten gaps sobre: "
                        + ", ".join(item.requirement_title for item in high_priority_gaps[:3])
                    ),
                    suggested_action="Seleccionar otra alternativa o regenerar Design con instrucciones mas precisas.",
                    source_refs=[item.requirement_key for item in high_priority_gaps[:3]],
                )
            )
        if selected_design.architecture in {"supervisor_with_subagents", "router_parallel"}:
            simplest = next((item for item in alternatives if item.architecture in {"single_agent", "single_agent_with_skills"}), None)
            if simplest is not None and selected_design.fit_score <= simplest.fit_score + 6:
                findings.append(
                    DesignCritiqueFinding(
                        finding_key="design-overarchitecture",
                        title="La alternativa recomendada puede estar sobredimensionada",
                        severity="warning",
                        detail="La ganancia frente a una opcion mas simple no justifica claramente la complejidad adicional.",
                        suggested_action="Comparar explicitamente el costo operativo contra una alternativa mas simple.",
                        source_refs=["design.alternatives"],
                    )
                )
        approval_required = bool(discovery.mvp_definition.non_delegable_decisions)
        if approval_required and not selected_design.approval_points:
            selected_design = selected_design.model_copy(
                update={
                    "approval_points": [
                        "Compuerta de aprobación humana para decisiones no delegables y acciones con efectos secundarios."
                    ]
                }
            )
            alternatives = [
                selected_design if item.alternative_key == selected_design.alternative_key else item
                for item in alternatives
            ]

    requirements_coverage = _selected_requirements_coverage(
        reconciled.fit_matrix,
        selected_design.alternative_key if selected_design else "",
    )
    blocking_count = sum(1 for item in findings if item.severity == "blocking")
    warning_count = sum(1 for item in findings if item.severity == "warning")
    missing_information = [
        item
        for item in _normalized_list(
            [
                *reconciled.missing_information,
                *[item.question for item in definition.open_questions if _definition_question_blocks_design(item)],
            ]
        )
        if not any(
            kw in item.lower()
            for kw in (
                "calibración matemática",
                "calibracion matematica",
                "datos históricos",
                "datos historicos",
                "mecanismos de integración",
                "mecanismos de integracion",
                "design_decisions",
                "tooling_principles",
                "memory_strategy",
                "benchmark de latencia",
                "filtro de sanitización",
                "filtro de sanitizacion",
                "volumen cuantitativo",
                "taxonomía completa",
                "taxonomia completa",
            )
        )
    ]
    review_state = (
        ReviewState.blocked
        if blocking_count > 0
        else ReviewState.partial
        if warning_count > 0 or missing_information
        else ReviewState.complete
    )
    average_fit = (
        round(sum(item.fit_score for item in alternatives) / len(alternatives), 2)
        if alternatives
        else 0.0
    )
    selected_fit = selected_design.fit_score if selected_design is not None else average_fit
    confidence_overall = max(0.0, min(1.0, (selected_fit / 100) - (blocking_count * 0.18) - (warning_count * 0.05)))
    confidence_band = "high" if confidence_overall >= 0.8 else "medium" if confidence_overall >= 0.6 else "low"
    return reconciled.model_copy(
        update={
            "alternatives": alternatives,
            "recommended_alternative_key": recommended_key,
            "selected_design": selected_design,
            "critic_findings": findings,
            "requirements_coverage": requirements_coverage,
            "missing_information": missing_information,
            "review_state": review_state,
            "confidence": DesignRecommendationConfidence(
                overall=confidence_overall,
                band=confidence_band,
                rationale=(
                    "La confianza combina fit de la alternativa seleccionada, critic findings sin reconciliar y preguntas abiertas heredadas de Definition."
                ),
            ),
            "summary": reconciled.summary or "Design comparo alternativas, las critico y dejo una recomendacion trazable.",
        }
    )
