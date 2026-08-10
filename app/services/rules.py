from app.models import (
    ApprovedToolsDigest,
    AgentCanvasProfile,
    BlueprintArtifact,
    BlueprintCoverageSummary,
    BlueprintLLMFunctionPolicy,
    BlueprintLLMPolicy,
    BlueprintSectionCoverageEntry,
    BlueprintTool,
    CanvasArtifact,
    ComponentCheckItem,
    ComponentReadinessEntry,
    DecisionTraceEntry,
    DeliveryPackage,
    DiscoveryArtifact,
    EmbeddingPolicy,
    EvaluationDatasetArtifact,
    EvaluationRubricArtifact,
    EvaluationArtifact,
    EvaluationCase,
    GeneratedDeliverable,
    GroundingPolicy,
    IngestionPolicy,
    KnowledgeProfile,
    KnowledgeSource,
    MemoryProfile,
    ObservabilityPlan,
    PatternCatalogEntry,
    RefreshPolicy,
    RetrievalPolicyProfile,
    RoadmapEvolution,
    RoadmapMilestone,
    RiskSummary,
    ReviewState,
    SafetyCheck,
    WorkflowProfile,
    WorkflowStep,
)
from app.diagnostics import (
    AUTONOMY_HIGH,
    AUTONOMY_LOW,
    AUTONOMY_MEDIUM,
    CASE_TYPE_AUTOMATION,
    CASE_TYPE_AUTONOMOUS_OPERATOR,
    CASE_TYPE_COPILOT,
    CASE_TYPE_INFORMATION,
    CASE_TYPE_MULTIAGENT_SYSTEM,
    is_multiagent_case,
    is_workflow_case,
    normalize_autonomy_level,
    normalize_case_type,
)
from app.services.evaluation_workbench import (
    build_default_evaluation_dataset,
    build_default_evaluation_rubric,
    build_evaluation_artifact_from_run,
    score_evaluation_workbench,
)


REQUIRED_DISCOVERY_TEXT_FIELDS = [
    "problem_statement",
    "current_user",
    "current_process",
    "desired_outcome",
    "autonomy_level",
    "operational_baseline.current_time_spent",
    "operational_baseline.current_cost",
    "mvp_definition.north_star_metric",
]

REQUIRED_DISCOVERY_LIST_FIELDS = [
    "operational_baseline.frequent_errors",
    "operational_baseline.automation_opportunities",
    "mvp_definition.v1_scope",
    "mvp_definition.out_of_scope",
    "mvp_definition.non_delegable_decisions",
]


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_list(items: list[str]) -> list[str]:
    return [normalize_text(item) for item in items if normalize_text(item)]


def _read_nested_value(payload: dict[str, object], path: str) -> object | None:
    current: object = payload
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def find_missing_discovery_fields(payload: dict[str, object]) -> list[str]:
    missing: list[str] = []
    for field_name in REQUIRED_DISCOVERY_TEXT_FIELDS:
        value = _read_nested_value(payload, field_name)
        if not isinstance(value, str) or not normalize_text(value):
            missing.append(field_name)
    for field_name in REQUIRED_DISCOVERY_LIST_FIELDS:
        value = _read_nested_value(payload, field_name)
        if not isinstance(value, list) or not normalize_list(
            [item for item in value if isinstance(item, str)]
        ):
            missing.append(field_name)
    return missing


def infer_case_type(problem_statement: str, desired_outcome: str, autonomy_level: str) -> str:
    text = normalize_text(f"{problem_statement} {desired_outcome}").lower()
    autonomy = normalize_autonomy_level(autonomy_level)

    multiagent_keywords = [
        "multiagente",
        "multi agente",
        "subagente",
        "sub agente",
        "varios agentes",
        "agentes especialistas",
        "orquestador",
        "supervisor",
        "coordinar agentes",
    ]
    information_keywords = [
        "faq",
        "preguntas frecuentes",
        "consulta",
        "consultas",
        "consultar",
        "informacion",
        "documentacion",
        "base de conocimiento",
        "knowledge base",
        "responder dudas",
    ]
    autonomous_keywords = [
        "sin intervencion humana",
        "sin intervención humana",
        "sin aprobacion humana",
        "sin aprobación humana",
        "end to end",
        "cerrar el caso",
        "resolver completo",
        "operar por si mismo",
        "operar por sí mismo",
        "ejecutar acciones",
    ]
    automation_keywords = [
        "automat",
        "workflow",
        "orquest",
        "ticket",
        "aprobacion",
        "approval",
        "pipeline",
        "provision",
        "desplieg",
        "backoffice",
        "integracion",
        "integración",
    ]
    copilot_keywords = [
        "anal",
        "invest",
        "diagnost",
        "recomend",
        "propuest",
        "blueprint",
        "compar",
        "explor",
        "definir",
        "dise",
    ]

    if any(keyword in text for keyword in multiagent_keywords):
        return CASE_TYPE_MULTIAGENT_SYSTEM

    if autonomy == AUTONOMY_HIGH and any(keyword in text for keyword in autonomous_keywords):
        return CASE_TYPE_AUTONOMOUS_OPERATOR

    if any(keyword in text for keyword in information_keywords) and autonomy != AUTONOMY_HIGH:
        if not any(keyword in text for keyword in automation_keywords):
            return CASE_TYPE_INFORMATION

    if any(keyword in text for keyword in automation_keywords) or autonomy == AUTONOMY_HIGH:
        return CASE_TYPE_AUTOMATION

    if any(keyword in text for keyword in information_keywords):
        return CASE_TYPE_INFORMATION

    if any(keyword in text for keyword in copilot_keywords):
        return CASE_TYPE_COPILOT

    return CASE_TYPE_COPILOT


def build_value_statement(
    problem_statement: str,
    desired_outcome: str,
    current_time_spent: str = "",
    current_cost: str = "",
) -> str:
    if normalize_text(current_time_spent) or normalize_text(current_cost):
        details = ", ".join(
            item
            for item in [
                f"tiempo actual {normalize_text(current_time_spent)}" if normalize_text(current_time_spent) else "",
                f"costo actual {normalize_text(current_cost)}" if normalize_text(current_cost) else "",
            ]
            if item
        )
        return (
            f"Reducir friccion en '{normalize_text(problem_statement)}', mejorar '{normalize_text(desired_outcome)}' "
            f"y bajar el esfuerzo operativo ({details})."
        )
    return f"Reducir friccion en '{normalize_text(problem_statement)}' y acercar el resultado a '{normalize_text(desired_outcome)}'."


def derive_scope(discovery: DiscoveryArtifact) -> list[str]:
    if discovery.mvp_definition.v1_scope:
        return normalize_list(discovery.mvp_definition.v1_scope)
    return [
        "Capturar el problema y contexto del usuario",
        "Generar un canvas estructurado del agente",
        "Producir un blueprint tecnico base con workflow y guardrails",
        "Entregar artefactos listos para implementacion",
    ]


def derive_out_of_scope(discovery: DiscoveryArtifact) -> list[str]:
    if discovery.mvp_definition.out_of_scope:
        return normalize_list(discovery.mvp_definition.out_of_scope)
    return [
        "Multiagente operativo en el MVP",
        "Automatizacion de despliegue o provisioning",
        "Monitoreo distribuido a nivel de produccion",
    ]


def derive_success_metric(discovery: DiscoveryArtifact) -> str:
    if normalize_text(discovery.mvp_definition.north_star_metric):
        return normalize_text(discovery.mvp_definition.north_star_metric)
    if is_workflow_case(discovery.case_type):
        return "El usuario obtiene un paquete de implementacion util en una sola sesion estructurada."
    return "El usuario sale con canvas, blueprint y backlog inicial sin retrabajo manual pesado."


def derive_primary_risk(discovery: DiscoveryArtifact) -> str:
    if discovery.operational_baseline.frequent_errors:
        return (
            "El MVP debe reducir primero el error mas repetido del proceso actual: "
            f"{normalize_text(discovery.operational_baseline.frequent_errors[0])}."
        )
    if normalize_autonomy_level(discovery.autonomy_level) == AUTONOMY_HIGH:
        return "Sobrealcance funcional antes de validar el MVP."
    return "Ambiguedad en entradas y requisitos del agente."


def derive_agent_profile(discovery: DiscoveryArtifact) -> AgentCanvasProfile:
    prohibited = normalize_list(
        [
            *discovery.constraints,
            *discovery.mvp_definition.non_delegable_decisions,
        ]
    )
    if not prohibited:
        prohibited = ["Tomar decisiones irreversibles sin aprobacion humana."]

    allowed = [
        "Sintetizar discovery en artefactos estructurados.",
        "Recomendar arquitectura minima viable.",
        "Proponer herramientas, memoria y controles antes de implementar.",
    ]
    if normalize_autonomy_level(discovery.autonomy_level) == AUTONOMY_HIGH:
        allowed.append("Preparar handoff tecnico para un workflow durable con checkpoints.")

    return AgentCanvasProfile(
        mission=f"Transformar '{normalize_text(discovery.problem_statement)}' en un agente implementable y medible.",
        primary_user=normalize_text(discovery.current_user),
        agent_task=normalize_text(discovery.problem_statement or discovery.desired_outcome),
        allowed_decisions=allowed,
        prohibited_decisions=prohibited,
        key_inputs=normalize_list(
            [
                discovery.problem_statement,
                discovery.current_process,
                discovery.desired_outcome,
                discovery.operational_baseline.current_time_spent,
                discovery.operational_baseline.current_cost,
                *discovery.operational_baseline.frequent_errors,
                *discovery.operational_baseline.automation_opportunities,
                *discovery.constraints,
            ]
        ),
        expected_outputs=[
            "Discovery normalizado",
            "Canvas Lean del agente",
            "Blueprint tecnico con herramientas y memoria",
            "Paquete de implementacion inicial",
        ],
        human_approvals=[
            "Aprobar el blueprint antes de implementarlo.",
            "Revisar cualquier paso con side effects o promocion de cambios.",
        ],
        success_metrics=[
            derive_success_metric(discovery),
            "Reducir ambiguedad del caso antes de pasar a implementacion.",
        ],
    )


ARCHITECTURE_LABELS = {
    "single_agent": "Agente unico",
    "single_agent_with_skills": "Agente con skills",
    "handoffs": "Handoffs secuenciales",
    "supervisor_with_subagents": "Supervisor con subagentes",
    "router_parallel": "Router paralelo",
}

REASONING_LABELS = {
    "ReAct": "ReAct",
    "Plan-and-Execute": "Plan & Execute",
    "Reflexion": "Reflexion",
    "ToT": "Tree of Thoughts",
    "HTN": "HTN",
}

MEMORY_LABELS = {
    "no_memory": "Sin memoria",
    "session_memory": "Memoria de sesion",
    "session_memory_with_checkpoints": "Memoria con checkpoints",
    "persistent_memory": "Memoria persistente",
}


def _label_for(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value)


def _clamp_score(value: int) -> int:
    return max(0, min(100, value))


def _contains_any(text: str, keywords: list[str]) -> bool:
    normalized = normalize_text(text).lower()
    return any(keyword in normalized for keyword in keywords)


def _context_metrics(discovery: DiscoveryArtifact, canvas: CanvasArtifact) -> dict[str, int | bool]:
    combined_text = " ".join(
        [
            discovery.problem_statement,
            discovery.current_process,
            discovery.desired_outcome,
            " ".join(discovery.mvp_definition.non_delegable_decisions),
        ]
    )
    sequential_flow = _contains_any(
        combined_text,
        ["etapa", "paso", "flujo", "aprob", "revision", "escalar", "handoff", "transfer", "validar"],
    )
    parallel_need = _contains_any(
        combined_text,
        ["paralel", "multiples fuentes", "varias fuentes", "simultane", "router", "enrutar", "clasificar"],
    )
    return {
        "scope_count": len(canvas.mvp_scope),
        "automation_count": len(discovery.operational_baseline.automation_opportunities),
        "error_count": len(discovery.operational_baseline.frequent_errors),
        "approval_count": len(canvas.agent_profile.human_approvals),
        "non_delegable_count": len(discovery.mvp_definition.non_delegable_decisions),
        "sequential_flow": sequential_flow,
        "parallel_need": parallel_need,
    }


def _fallback_canvas_for_rules(discovery: DiscoveryArtifact) -> CanvasArtifact:
    lightweight_scope = derive_scope(discovery) if discovery.mvp_definition.v1_scope else [normalize_text(discovery.problem_statement or discovery.desired_outcome)]
    return CanvasArtifact(
        user_goal=normalize_text(discovery.desired_outcome or discovery.problem_statement),
        mvp_scope=[item for item in lightweight_scope if item][:2],
        out_of_scope=derive_out_of_scope(discovery) if discovery.mvp_definition.out_of_scope else [],
        success_metric=derive_success_metric(discovery),
        primary_risk=derive_primary_risk(discovery),
        agent_profile=AgentCanvasProfile(
            mission=normalize_text(discovery.problem_statement or discovery.desired_outcome),
            primary_user=normalize_text(discovery.current_user),
            agent_task=normalize_text(discovery.problem_statement or discovery.desired_outcome),
            human_approvals=normalize_list(discovery.mvp_definition.non_delegable_decisions),
        ),
    )


def build_architecture_catalog(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact | None,
) -> list[PatternCatalogEntry]:
    canvas = canvas or _fallback_canvas_for_rules(discovery)
    metrics = _context_metrics(discovery, canvas)
    scope_count = int(metrics["scope_count"])
    automation_count = int(metrics["automation_count"])
    approval_count = int(metrics["approval_count"])
    sequential_flow = bool(metrics["sequential_flow"])
    parallel_need = bool(metrics["parallel_need"])
    autonomy_high = normalize_autonomy_level(discovery.autonomy_level) == AUTONOMY_HIGH
    workflow_case = is_workflow_case(discovery.case_type)
    multiagent_case = is_multiagent_case(discovery.case_type)

    return [
        PatternCatalogEntry(
            family="architecture",
            key="single_agent",
            label=_label_for("single_agent", ARCHITECTURE_LABELS),
            summary="Una sola unidad de decision cuando el dominio cabe en una misma conversacion y el MVP debe salir rapido.",
            use_when=["Problema acotado", "Pocas herramientas", "Baja coordinacion entre etapas"],
            tradeoffs=["Menor modularidad", "Escala peor si aparecen dominios muy distintos"],
            fit_score=_clamp_score(
                82
                - max(scope_count - 2, 0) * 12
                - (18 if sequential_flow else 0)
                - (22 if parallel_need else 0)
                - (12 if autonomy_high else 0)
                - (24 if multiagent_case else 0)
            ),
        ),
        PatternCatalogEntry(
            family="architecture",
            key="single_agent_with_skills",
            label=_label_for("single_agent_with_skills", ARCHITECTURE_LABELS),
            summary="Una sola interfaz con especialidades delimitadas, util para mantener simplicidad sin perder cobertura funcional.",
            use_when=["Un mismo agente necesita varios bloques especializados", "El flujo tiene varias capacidades pero un solo owner"],
            tradeoffs=["Requiere contratos claros por skill", "Puede crecer de mas si no se controla el scope"],
            fit_score=_clamp_score(
                48
                + (18 if workflow_case else 0)
                + min(automation_count, 3) * 8
                + (12 if scope_count >= 3 else 0)
                + (8 if approval_count >= 1 else 0)
                - (8 if parallel_need else 0)
                - (10 if multiagent_case else 0)
            ),
        ),
        PatternCatalogEntry(
            family="architecture",
            key="handoffs",
            label=_label_for("handoffs", ARCHITECTURE_LABELS),
            summary="Cadena secuencial de etapas cuando el flujo necesita cambios de contexto claros y checkpoints visibles.",
            use_when=["Proceso por etapas", "Necesidad de aprobaciones o validaciones intermedias", "Cambio explicito de responsabilidad"],
            tradeoffs=["Mas coordinacion", "Mas puntos de orquestacion a controlar"],
            fit_score=_clamp_score(
                18
                + (36 if sequential_flow else 0)
                + (22 if workflow_case else 0)
                + (12 if approval_count >= 2 else 0)
                + (10 if scope_count >= 3 else 0)
                + (8 if autonomy_high else 0)
            ),
        ),
        PatternCatalogEntry(
            family="architecture",
            key="supervisor_with_subagents",
            label=_label_for("supervisor_with_subagents", ARCHITECTURE_LABELS),
            summary="Control central con especialistas separados cuando el dominio ya es demasiado amplio para un solo agente.",
            use_when=["Dominios separados", "Alta autonomia con varias decisiones", "Necesidad real de delegacion"],
            tradeoffs=["Mas costo operativo", "Mayor complejidad de pruebas y coordinacion"],
            fit_score=_clamp_score(
                14
                + (28 if autonomy_high else 0)
                + (34 if multiagent_case else 0)
                + (18 if scope_count >= 4 else 0)
                + (12 if automation_count >= 3 else 0)
                + (10 if approval_count >= 2 else 0)
            ),
        ),
        PatternCatalogEntry(
            family="architecture",
            key="router_parallel",
            label=_label_for("router_parallel", ARCHITECTURE_LABELS),
            summary="Ruteo paralelo entre fuentes o capacidades cuando el caso exige clasificacion y consultas simultaneas.",
            use_when=["Multiples fuentes simultaneas", "Clasificacion previa al procesamiento", "Consultas paralelas controladas"],
            tradeoffs=["Mas latencia de coordinacion", "Mayor complejidad para MVP temprano"],
            fit_score=_clamp_score(
                10
                + (38 if parallel_need else 0)
                + (18 if multiagent_case else 0)
                + (10 if automation_count >= 3 else 0)
                + (8 if scope_count >= 3 else 0)
                - (10 if approval_count > 0 else 0)
            ),
        ),
    ]


def build_reasoning_catalog(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact | None,
) -> list[PatternCatalogEntry]:
    canvas = canvas or _fallback_canvas_for_rules(discovery)
    metrics = _context_metrics(discovery, canvas)
    scope_count = int(metrics["scope_count"])
    error_count = int(metrics["error_count"])
    sequential_flow = bool(metrics["sequential_flow"])
    parallel_need = bool(metrics["parallel_need"])
    workflow_case = is_workflow_case(discovery.case_type)
    multiagent_case = is_multiagent_case(discovery.case_type)
    autonomy_high = normalize_autonomy_level(discovery.autonomy_level) == AUTONOMY_HIGH

    return [
        PatternCatalogEntry(
            family="reasoning",
            key="ReAct",
            label=_label_for("ReAct", REASONING_LABELS),
            summary="Buen default para tareas dinamicas con herramientas y decisiones locales de bajo costo.",
            use_when=["Interaccion con tools", "Flujo corto", "Necesidad de iterar sobre observacion y accion"],
            tradeoffs=["Menor control sobre planes largos", "Puede quedarse corto en workflows extensos"],
            fit_score=_clamp_score(80 - max(scope_count - 2, 0) * 10 - (12 if sequential_flow else 0)),
        ),
        PatternCatalogEntry(
            family="reasoning",
            key="Plan-and-Execute",
            label=_label_for("Plan-and-Execute", REASONING_LABELS),
            summary="Separa plan y ejecucion para workflows largos o con varias etapas visibles.",
            use_when=["Flujos secuenciales", "Alcance MVP con varias actividades", "Necesidad de checkpoints"],
            tradeoffs=["Mas estructura upfront", "Puede sobredisenar tareas simples"],
            fit_score=_clamp_score(
                42
                + (24 if workflow_case else 0)
                + (12 if multiagent_case else 0)
                + (18 if sequential_flow else 0)
                + (12 if scope_count >= 3 else 0)
            ),
        ),
        PatternCatalogEntry(
            family="reasoning",
            key="Reflexion",
            label=_label_for("Reflexion", REASONING_LABELS),
            summary="Introduce autoevaluacion cuando el agente debe corregirse ante errores repetidos o incertidumbre alta.",
            use_when=["Errores frecuentes", "Necesidad de mejora iterativa", "Casos ambiguos"],
            tradeoffs=["Mas costo por iteracion", "No siempre aporta valor en MVP corto"],
            fit_score=_clamp_score(18 + min(error_count, 4) * 10 + (10 if autonomy_high else 0)),
        ),
        PatternCatalogEntry(
            family="reasoning",
            key="ToT",
            label=_label_for("ToT", REASONING_LABELS),
            summary="Explora varias rutas antes de decidir cuando el problema es ambiguo, amplio o requiere comparar alternativas.",
            use_when=["Exploracion compleja", "Varias rutas posibles", "Necesidad de comparar alternativas antes del cierre"],
            tradeoffs=["Mayor costo de razonamiento", "No conviene para MVP simples o flujos cortos"],
            fit_score=_clamp_score(
                12
                + (18 if scope_count >= 4 else 0)
                + (12 if error_count >= 2 else 0)
                + (10 if autonomy_high else 0)
                + (8 if parallel_need else 0)
                - (10 if sequential_flow else 0)
            ),
        ),
        PatternCatalogEntry(
            family="reasoning",
            key="HTN",
            label=_label_for("HTN", REASONING_LABELS),
            summary="Conviene cuando el proceso ya esta muy jerarquizado y requiere descomposicion formal por etapas.",
            use_when=["Proceso formal", "Dependencias entre pasos", "Reglas de negocio secuenciales"],
            tradeoffs=["Mas rigidez", "Sobrepeso para discovery o MVP temprano"],
            fit_score=_clamp_score(16 + (20 if sequential_flow else 0) + (16 if scope_count >= 4 else 0)),
        ),
    ]


def build_memory_catalog(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact | None,
) -> list[PatternCatalogEntry]:
    canvas = canvas or _fallback_canvas_for_rules(discovery)
    metrics = _context_metrics(discovery, canvas)
    scope_count = int(metrics["scope_count"])
    approval_count = int(metrics["approval_count"])
    non_delegable_count = int(metrics["non_delegable_count"])
    sequential_flow = bool(metrics["sequential_flow"])
    normalized_autonomy = normalize_autonomy_level(discovery.autonomy_level)
    autonomy_low = normalized_autonomy == AUTONOMY_LOW
    autonomy_high = normalized_autonomy == AUTONOMY_HIGH

    return [
        PatternCatalogEntry(
            family="memory",
            key="no_memory",
            label=_label_for("no_memory", MEMORY_LABELS),
            summary="Solo para tareas muy puntuales donde no hace falta retener contexto entre pasos.",
            use_when=["Caso puntual", "Baja autonomia", "Sin necesidad de retomar estado"],
            tradeoffs=["No retiene aprendizaje", "Se rompe facil si el flujo crece"],
            fit_score=_clamp_score(20 + (18 if autonomy_low else 0) - (18 if sequential_flow else 0) - scope_count * 6),
        ),
        PatternCatalogEntry(
            family="memory",
            key="session_memory",
            label=_label_for("session_memory", MEMORY_LABELS),
            summary="Default economico para conservar contexto de una sesion sin complicar la persistencia.",
            use_when=["MVP conversacional", "Una sola sesion de trabajo", "Contexto acotado"],
            tradeoffs=["No aprende a largo plazo", "Puede quedarse corto en workflows largos"],
            fit_score=_clamp_score(74 + (8 if normalized_autonomy == AUTONOMY_MEDIUM else 0) - (10 if scope_count >= 4 else 0)),
        ),
        PatternCatalogEntry(
            family="memory",
            key="session_memory_with_checkpoints",
            label=_label_for("session_memory_with_checkpoints", MEMORY_LABELS),
            summary="Agrega checkpoints cuando el flujo necesita pausas, aprobaciones o continuidad controlada.",
            use_when=["Flujo secuencial", "Aprobaciones humanas", "Necesidad de resumir y retomar"],
            tradeoffs=["Mas estado a gestionar", "Mayor disciplina de escritura"],
            fit_score=_clamp_score(
                28 + (24 if autonomy_high else 0) + (18 if sequential_flow else 0) + (12 if approval_count >= 1 else 0)
            ),
        ),
        PatternCatalogEntry(
            family="memory",
            key="persistent_memory",
            label=_label_for("persistent_memory", MEMORY_LABELS),
            summary="Util cuando el agente debe recordar decisiones estables o patrones entre sesiones.",
            use_when=["Conocimiento estable", "Muchas decisiones no delegables", "Uso recurrente del mismo agente"],
            tradeoffs=["Mayor gobierno de datos", "No conviene activarla temprano sin necesidad real"],
            fit_score=_clamp_score(14 + non_delegable_count * 10 + (10 if scope_count >= 4 else 0)),
        ),
    ]


def _select_best_pattern(catalog: list[PatternCatalogEntry]) -> str:
    return max(catalog, key=lambda item: item.fit_score).key


def select_architecture(discovery: DiscoveryArtifact, canvas: CanvasArtifact | None = None) -> str:
    return _select_best_pattern(build_architecture_catalog(discovery, canvas))


def select_reasoning_pattern(discovery: DiscoveryArtifact, canvas: CanvasArtifact | None = None) -> str:
    return _select_best_pattern(build_reasoning_catalog(discovery, canvas))


def select_memory_strategy(discovery: DiscoveryArtifact, canvas: CanvasArtifact | None = None) -> str:
    return _select_best_pattern(build_memory_catalog(discovery, canvas))


def _decision_evidence(
    dimension: str,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
) -> list[str]:
    metrics = _context_metrics(discovery, canvas)
    if dimension == "architecture":
        return [
            f"scope_v1={metrics['scope_count']}",
            f"automation_opportunities={metrics['automation_count']}",
            f"sequential_flow={'yes' if metrics['sequential_flow'] else 'no'}",
            f"parallel_need={'yes' if metrics['parallel_need'] else 'no'}",
        ]
    if dimension == "reasoning":
        return [
            f"case_type={normalize_case_type(discovery.case_type) or discovery.case_type}",
            f"scope_v1={metrics['scope_count']}",
            f"frequent_errors={metrics['error_count']}",
            f"sequential_flow={'yes' if metrics['sequential_flow'] else 'no'}",
        ]
    return [
        f"autonomy_level={normalize_autonomy_level(discovery.autonomy_level)}",
        f"human_approvals={metrics['approval_count']}",
        f"non_delegable={metrics['non_delegable_count']}",
        f"scope_v1={metrics['scope_count']}",
    ]


def build_decision_report(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    architecture: str,
    reasoning_pattern: str,
    memory_strategy: str,
) -> tuple[str, list[DecisionTraceEntry], list[PatternCatalogEntry]]:
    families = [
        (
            "architecture",
            build_architecture_catalog(discovery, canvas),
            architecture,
            ARCHITECTURE_LABELS,
        ),
        (
            "reasoning",
            build_reasoning_catalog(discovery, canvas),
            reasoning_pattern,
            REASONING_LABELS,
        ),
        (
            "memory",
            build_memory_catalog(discovery, canvas),
            memory_strategy,
            MEMORY_LABELS,
        ),
    ]
    decision_trace: list[DecisionTraceEntry] = []
    pattern_catalog: list[PatternCatalogEntry] = []
    summary_lines: list[str] = []

    for dimension, catalog, selected_value, labels in families:
        recommended = max(catalog, key=lambda item: item.fit_score)
        selected_entry = next((item for item in catalog if item.key == selected_value), recommended)
        dimension_label = {
            "architecture": "arquitectura",
            "reasoning": "razonamiento",
            "memory": "memoria",
        }.get(dimension, dimension)
        pattern_catalog.extend([item.model_copy(update={"selected": item.key == selected_value}) for item in catalog])
        is_override = selected_value != recommended.key
        decision_trace.append(
            DecisionTraceEntry(
                dimension=dimension,
                selected_value=selected_value,
                selected_label=_label_for(selected_value, labels),
                recommended_value=recommended.key,
                recommended_label=recommended.label,
                decision_source="manual_override" if is_override else "rule_engine",
                rationale=selected_entry.summary,
                evidence=_decision_evidence(dimension, discovery, canvas),
                review_note=(
                    f"Se aplico override manual sobre la recomendacion base ({recommended.label})."
                    if is_override
                    else "La seleccion queda alineada con la recomendacion del motor de reglas."
                ),
            )
        )
        summary_lines.append(
            f"{dimension_label}: {_label_for(selected_value, labels)}"
            + (f" (override sobre {recommended.label})" if is_override else f" recomendado por reglas con fit {recommended.fit_score}")
        )

    summary = (
        "Decision rule-first del blueprint: "
        + "; ".join(summary_lines)
        + ". El LLM solo sintetiza narrativa y no modifica la seleccion estructural."
    )
    return summary, decision_trace, pattern_catalog


def default_tools(discovery: DiscoveryArtifact) -> list[BlueprintTool]:
    return [
        BlueprintTool(
            name="normalize_discovery",
            purpose="Transformar entradas del usuario en una estructura valida y trazable.",
            owner="builder",
            archetype="read_only",
            integration_kind="local_runtime",
            endpoint_reference="internal://skill_runtime/normalize_discovery",
            auth_reference="none",
            risk_level="low",
            requires_approval=False,
            inputs=["problem_statement", "current_user", "current_process", "desired_outcome", "autonomy_level"],
            outputs=["normalized_discovery"],
            validations=["Campos obligatorios presentes", "Sin valores inventados", "Constraints limpias"],
            typed_errors=["validation_error", "missing_required_field"],
            permissions=["read_discovery_input"],
            scopes=["read"],
            sensitive_data=["user_problem_statement"],
            audit_rules=["Registrar input hash y resultado estructurado.", "No persistir secretos en logs."],
            has_side_effects=False,
            execution_mode="in_process_validation",
            approval_policy="not_required",
            retry_strategy="Reintentar tras corregir campos faltantes",
            idempotency_strategy="Repetible sobre el mismo input estructurado sin side effects.",
            compensation_strategy="No aplica porque no hay side effects",
            failure_mode="Campos obligatorios ausentes o ambiguos",
            rate_limit_policy="Sin limite externo; una ejecucion por accion del usuario.",
            timeout_policy="Timeout local corto de 10 segundos.",
            contract_review_state="needs-review",
        ),
        BlueprintTool(
            name="build_canvas",
            purpose="Convertir discovery en un canvas Lean con alcance, riesgo y metricas.",
            owner="builder",
            archetype="read_only",
            integration_kind="local_runtime",
            endpoint_reference="internal://skill_runtime/build_canvas",
            auth_reference="none",
            risk_level="low",
            requires_approval=False,
            inputs=["normalized_discovery"],
            outputs=["canvas"],
            validations=["Canvas con objetivo", "Alcance v1 definido", "Riesgo principal declarado"],
            typed_errors=["derivation_error", "incomplete_canvas"],
            permissions=["read_discovery_state"],
            scopes=["read"],
            sensitive_data=["canvas_scope"],
            audit_rules=["Versionar el canvas derivado.", "Registrar rationale resumido del alcance."],
            has_side_effects=False,
            execution_mode="in_process_derivation",
            approval_policy="not_required",
            retry_strategy="Repetir la derivacion cuando cambie discovery",
            idempotency_strategy="Repetible mientras el snapshot de discovery no cambie.",
            compensation_strategy="No aplica porque solo genera artefactos",
            failure_mode="Canvas parcial por discovery incompleto",
            rate_limit_policy="Sin limite externo; una derivacion por cambio relevante.",
            timeout_policy="Timeout local corto de 15 segundos.",
            contract_review_state="needs-review",
        ),
        BlueprintTool(
            name="build_blueprint",
            purpose="Generar el blueprint tecnico con herramientas, memoria, workflow y artefactos base.",
            owner="builder",
            archetype="async_job",
            integration_kind="local_runtime",
            endpoint_reference="internal://skill_runtime/build_blueprint",
            auth_reference="none",
            risk_level="medium",
            requires_approval=False,
            inputs=["normalized_discovery", "canvas"],
            outputs=["blueprint_package"],
            validations=["Arquitectura justificada", "Memoria definida", "Workflow durable descrito"],
            typed_errors=["generation_timeout", "contract_gap_detected"],
            permissions=["read_blueprint_inputs", "write_blueprint_artifacts"],
            scopes=["read", "write"],
            sensitive_data=["blueprint_draft", "delivery_package"],
            audit_rules=["Versionar el blueprint antes de promotion.", "Registrar warnings y gaps abiertos."],
            has_side_effects=False,
            execution_mode="async_job",
            approval_policy="not_required",
            retry_strategy="Regenerar tras ajustar canvas o restricciones",
            idempotency_strategy="Idempotente por snapshot; mismo input produce una nueva version trazable.",
            compensation_strategy="No aplica porque no muta sistemas externos",
            failure_mode="Blueprint inconsistente, incompleto o sin entregables minimos",
            rate_limit_policy="Una corrida concurrente por sesion.",
            timeout_policy="Timeout de 60 segundos con cancelacion visible.",
            contract_review_state="needs-review",
        ),
        BlueprintTool(
            name="promote_blueprint_for_implementation",
            purpose="Solicitar el paso a implementacion solo despues de validar riesgos y artefactos finales.",
            owner="local_admin",
            archetype="side_effect",
            integration_kind="governed_handoff",
            endpoint_reference="workflow://handoff/promote_blueprint_for_implementation",
            auth_reference="local_admin_session",
            risk_level="high",
            requires_approval=True,
            inputs=["approved_blueprint_package"],
            outputs=["implementation_handoff"],
            validations=["Existe backlog MVP", "Hay plan de seguridad", "Se resolvieron approvals pendientes"],
            typed_errors=["approval_missing", "handoff_rejected", "open_risk_detected"],
            permissions=["promote_blueprint_package", "open_implementation_handoff"],
            scopes=["write", "handoff"],
            sensitive_data=["implementation_commitment", "approval_context"],
            audit_rules=["Registrar approver, motivo y version promovida.", "Bloquear promotion si cambian riesgos o approvals."],
            has_side_effects=True,
            execution_mode="collect_intent_then_workflow_execute",
            approval_policy="local_admin_mandatory",
            retry_strategy="Reenfile manual del handoff si el gate es rechazado o expira",
            idempotency_strategy="Usar session_id + blueprint_version_number como clave de promotion.",
            compensation_strategy="Revertir la promocion y volver a estado needs_review si la aprobacion se revoca",
            approval_reason="Promover un blueprint puede disparar trabajo tecnico o compromisos de implementacion.",
            failure_mode="Se intenta ejecutar sin aprobacion humana o con riesgos abiertos",
            rate_limit_policy="Una promotion activa por version de blueprint.",
            timeout_policy="Timeout de 30 segundos y rollback si no confirma el gate.",
            contract_review_state="needs-review",
        ),
    ]


def _default_llm_role(
    role: str,
    *,
    provider: str,
    model: str,
    reasoning_effort: str,
    max_tokens: int,
    fallback_model: str,
    tool_availability: list[str] | None = None,
) -> BlueprintLLMFunctionPolicy:
    return BlueprintLLMFunctionPolicy(
        role=role,
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        fallback_model=fallback_model,
        tool_availability=tool_availability or [],
    )


def default_blueprint_llm_policy(reasoning_pattern: str, tools: list[BlueprintTool]) -> BlueprintLLMPolicy:
    provider = "openai"
    fast_model = "gpt-5-mini"
    reasoning_model = "gpt-5.5"
    fallback_model = "manual_review_gate"
    tool_names = [tool.name for tool in tools if normalize_text(tool.name)]
    reasoning_effort = "medium" if reasoning_pattern == "ReAct" else "high"
    functions = [
        _default_llm_role(
            "planner",
            provider=provider,
            model=reasoning_model,
            reasoning_effort="high",
            max_tokens=2200,
            fallback_model=fallback_model,
        ),
        _default_llm_role(
            "executor",
            provider=provider,
            model=fast_model,
            reasoning_effort=reasoning_effort,
            max_tokens=1600,
            fallback_model=fallback_model,
            tool_availability=tool_names,
        ),
        _default_llm_role(
            "evaluator",
            provider=provider,
            model=reasoning_model,
            reasoning_effort="high",
            max_tokens=1800,
            fallback_model=fallback_model,
        ),
    ]
    if tool_names:
        functions.append(
            _default_llm_role(
                "tool_use",
                provider=provider,
                model=fast_model,
                reasoning_effort="medium",
                max_tokens=1200,
                fallback_model=fallback_model,
                tool_availability=tool_names,
            )
        )

    return BlueprintLLMPolicy(
        provider=provider,
        fast_model=fast_model,
        reasoning_model=reasoning_model,
        fallback_model=fallback_model,
        context_policy="Trabajar solo con contratos aprobados, evidencia trazable y el snapshot vigente de la sesion.",
        sampling_policy="Planner y evaluator con temperatura baja; executor y tool_use sin improvisacion fuera del contrato.",
        fallback_policy="Si el proveedor no responde o el contrato no alcanza, usar el fallback declarado o bloquear con revision humana explicita.",
        circuit_breaker_policy="Abrir circuit breaker tras 3 fallos consecutivos del mismo proveedor o tool path y escalar a review.",
        budget_policy="Reservar razonamiento largo para planner y evaluator; limitar executor y tool_use a interacciones acotadas.",
        output_validation_policy="Validar cada salida estructurada contra schemas versionados antes de promover estado o artefactos.",
        log_redaction_policy="Redactar secretos, credenciales y datos sensibles; conservar ids, hashes y trazas auditables.",
        review_state="needs-review",
        functions=functions,
    )


def default_guardrails(discovery: DiscoveryArtifact) -> list[str]:
    return [
        "No inventar campos obligatorios ausentes.",
        "Usar reglas antes que inferencia libre cuando falte evidencia.",
        "No ejecutar side effects fuera de un gate de aprobacion.",
        "Persistir checkpoints y decisiones visibles por etapa.",
        "No usar memoria libre como fuente de verdad del proyecto.",
    ]


def derive_memory_profile(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact | None = None,
    approved_tools_digest: ApprovedToolsDigest | None = None,
) -> MemoryProfile:
    storage_layers = ["session_state", "session_snapshot"]
    if normalize_autonomy_level(discovery.autonomy_level) == AUTONOMY_HIGH or (
        approved_tools_digest is not None
        and (
            approved_tools_digest.side_effect_tool_keys
            or approved_tools_digest.approval_required_tool_keys
        )
    ):
        storage_layers.append("checkpoint_summary")
    if approved_tools_digest is not None and approved_tools_digest.knowledge_tool_keys:
        storage_layers.append("approved_knowledge_index")

    write_policy = "Persistir solo estado validado por etapa y snapshots versionados."
    retrieval_policy = "Recuperar por session_id, artefacto actual y checkpoints, nunca por historial libre."
    workspace_scope = "Usar memoria del workspace solo para continuidad del blueprint y evidencia aprobada."
    agent_scope = "Exportar al agente final solo resumentes, checkpoints y preferencias aprobadas."
    review_trigger = "Marcar needs_review cuando falten campos criticos o existan contradicciones."

    if approved_tools_digest is not None and approved_tools_digest.tool_count > 0:
        approved_tools_line = ", ".join(approved_tools_digest.approved_tool_keys)
        write_policy = (
            "Persistir solo estado validado por etapa, snapshots versionados y evidencia producida por tools aprobadas: "
            f"{approved_tools_line}."
        )
        retrieval_policy = (
            "Recuperar solo por session_id, blueprint activo, checkpoints aprobados y digest compacto de tools; "
            f"scopes activos: {', '.join(approved_tools_digest.retrieval_scopes)}."
        )
        workspace_scope = (
            "Usar memoria del workspace solo para continuidad del blueprint, knowledge aprobada y tools promovidas."
        )
        agent_scope = (
            "Exportar al agente final solo resumentes, checkpoints, knowledge aprobada y el set minimo de tools promovidas."
        )
        review_trigger = (
            "Marcar needs_review cuando cambie el digest de tools aprobadas, falten campos criticos o existan contradicciones."
        )

    return MemoryProfile(
        strategy=select_memory_strategy(discovery, canvas),
        storage_layers=storage_layers,
        write_policy=write_policy,
        retrieval_policy=retrieval_policy,
        review_trigger=review_trigger,
        goal_drift_guard="Comparar siempre el objetivo del canvas con el desired_outcome antes de avanzar.",
        retention_policy="Conservar memoria operativa solo mientras soporte el workflow activo y sus checkpoints aprobados.",
        ttl_policy="TTL corto para sesion y checkpoints; extender solo bajo necesidad explicita del caso.",
        workspace_scope=workspace_scope,
        agent_scope=agent_scope,
        grounding_policy=GroundingPolicy(
            citations_policy="La memoria debe citar el artefacto o checkpoint del que proviene.",
            confidence_policy="Usar memoria previa solo cuando mantenga trazabilidad y no contradiga evidencia vigente.",
            no_evidence_behavior="Si la memoria no aporta evidencia confiable, seguir solo el estado explicito del workflow.",
            contradictory_evidence_behavior="Escalar a revision humana si memoria y evidencia activa se contradicen.",
        ),
        sensitivity_rules=[
            "No persistir secretos ni credenciales en memoria exportable.",
            "Toda memoria persistente requiere owner, TTL y criterio de borrado visibles.",
        ],
    )


def _contains_knowledge_signals(*values: str | None) -> bool:
    haystack = " ".join(normalize_text(value) for value in values if value)
    normalized = haystack.lower()
    tokens = (
        "knowledge",
        "retrieval",
        "rag",
        "wiki",
        "faq",
        "runbook",
        "confluence",
        "document",
        "manual",
        "base de conocimiento",
        "evidencia",
    )
    return any(token in normalized for token in tokens)


def derive_knowledge_profile(
    discovery: DiscoveryArtifact,
    tools: list[BlueprintTool],
    memory_strategy: str,
) -> KnowledgeProfile:
    knowledge_enabled = _contains_knowledge_signals(
        discovery.problem_statement,
        discovery.current_process,
        discovery.desired_outcome,
        *[tool.name for tool in tools],
        *[tool.purpose for tool in tools],
    )

    if not knowledge_enabled:
        return KnowledgeProfile(
            mode="none",
            grounding_policy=GroundingPolicy(
                citations_policy="No aplica porque el caso no depende de retrieval documental.",
                confidence_policy="No aplica.",
                no_evidence_behavior="Continuar con estado de workflow sin retrieval adicional.",
                contradictory_evidence_behavior="No aplica.",
            ),
            notes="Caso sin knowledge o RAG dedicado.",
        )

    source_candidates = [
        tool
        for tool in tools
        if _contains_knowledge_signals(tool.name, tool.purpose, " ".join(tool.inputs), " ".join(tool.outputs))
    ]
    sources = [
        KnowledgeSource(
            key=tool.name,
            title=tool.name.replace("_", " ").title(),
            source_type="tool_reference",
            uri=f"tool://{tool.name}",
            owner=tool.owner or "knowledge_owner_pending",
            sensitivity="internal",
            license="pending_review",
            description=tool.purpose,
            source_version="pending",
        )
        for tool in source_candidates
    ]
    if not sources:
        sources = [
            KnowledgeSource(
                key="knowledge_source_1",
                title="Fuente principal pendiente",
                source_type="document_repository",
                uri="abstract://knowledge/source-1",
                owner="knowledge_owner_pending",
                sensitivity="internal",
                license="pending_review",
                description="Definir la fuente primaria aprobada para retrieval.",
                source_version="pending",
            )
        ]

    return KnowledgeProfile(
        mode="rag",
        sources=sources,
        ingestion_policy=IngestionPolicy(
            parser="structured_documents",
            chunking_policy="pending_review",
            metadata_fields=["owner", "source_version", "updated_at"],
            include_filters=["approved_only"],
            exclude_filters=["private_credentials"],
        ),
        embedding_policy=EmbeddingPolicy(
            provider="pending_review",
            model="pending_review",
            dimensions=0,
            version="pending",
        ),
        retrieval_policy=RetrievalPolicyProfile(
            top_k=5,
            filters=["approved_only"],
            search_mode="hybrid",
            reranking_policy="pending_review",
            fallback_behavior="Declarar falta de evidencia y escalar a remediation guiada.",
        ),
        refresh_policy=RefreshPolicy(
            frequency="pending_review",
            triggers=["source_change", "manual_review"],
            expiration_policy=(
                "TTL alineado con la memoria persistente."
                if memory_strategy == "persistent_memory"
                else "Revisar freshness por sesion o evento relevante."
            ),
            deletion_policy="Borrar referencias obsoletas y mantener lineage de cambios aprobados.",
        ),
        grounding_policy=GroundingPolicy(
            citations_policy="Toda respuesta recuperada debe citar fuente, version y owner.",
            confidence_policy="Usar solo evidencia con confianza suficiente y owner aprobado.",
            no_evidence_behavior="Responder needs-resolution cuando el retrieval no aporte soporte suficiente.",
            contradictory_evidence_behavior="Escalar cuando dos fuentes aprobadas se contradigan.",
        ),
        sensitivity_rules=[
            "No exportar documentos privados ni credenciales; solo referencias abstractas.",
            "Las fuentes sensibles requieren owner, refresh policy y borrado visibles.",
        ],
        notes="Knowledge detectado por evidencia del caso; cerrar ownership y politica semantica antes de construir retrieval real.",
    )


def derive_safety_checks(discovery: DiscoveryArtifact) -> list[SafetyCheck]:
    return [
        SafetyCheck(
            category="input_quality",
            risk="Entradas ambiguas o incompletas",
            severity="medium",
            mitigation="Bloquear avance y exponer missing_fields.",
            status="required",
        ),
        SafetyCheck(
            category="hallucination_control",
            risk="Campos inventados por inferencia libre",
            severity="high",
            mitigation="Usar unknown, validacion posterior y evidencia rule-first.",
            status="required",
        ),
        SafetyCheck(
            category="change_control",
            risk="Desvio del stack o la arquitectura planificada",
            severity="medium",
            mitigation="Aplicar gates por etapa y versionado de blueprint.",
            status="required",
        ),
        SafetyCheck(
            category="approval_governance",
            risk="Promocion prematura a implementacion sin control humano",
            severity="high",
            mitigation="Crear un approval gate explicito antes de cualquier side effect.",
            status="required",
        ),
    ]


def _build_check_item(
    *,
    key: str,
    title: str,
    ok: bool,
    pass_detail: str,
    fail_detail: str,
    fail_status: ReviewState = ReviewState.partial,
) -> ComponentCheckItem:
    return ComponentCheckItem(
        key=key,
        title=title,
        status=ReviewState.complete if ok else fail_status,
        detail=pass_detail if ok else fail_detail,
    )


def _summarize_component(
    component: str,
    label: str,
    checks: list[ComponentCheckItem],
) -> ComponentReadinessEntry:
    total = len(checks)
    completed = sum(1 for item in checks if item.status == ReviewState.complete)
    if any(item.status == ReviewState.blocked for item in checks):
        status = ReviewState.blocked
    elif any(item.status == ReviewState.partial for item in checks):
        status = ReviewState.partial
    else:
        status = ReviewState.complete
    return ComponentReadinessEntry(
        component=component,
        label=label,
        status=status,
        score=round((completed / total) * 100) if total else 0,
        completed_checks=completed,
        total_checks=total,
        blocking_issues=[item.detail for item in checks if item.status != ReviewState.complete],
        checks=checks,
    )


def build_tools_readiness(tools: list[BlueprintTool]) -> ComponentReadinessEntry:
    checks = [
        _build_check_item(
            key="tool_inventory",
            title="Inventario minimo de tools",
            ok=bool(tools),
            pass_detail=f"El blueprint declara {len(tools)} tools.",
            fail_detail="No hay tools declaradas para ejecutar el flujo base.",
            fail_status=ReviewState.blocked,
        ),
        _build_check_item(
            key="tool_contracts",
            title="Contratos basicos",
            ok=bool(tools)
            and all(
                normalize_text(tool.name)
                and normalize_text(tool.purpose)
                and tool.inputs
                and tool.outputs
                for tool in tools
            ),
            pass_detail="Todas las tools tienen nombre, proposito, inputs y outputs.",
            fail_detail="Hay tools con contrato incompleto en nombre, proposito, inputs o outputs.",
        ),
        _build_check_item(
            key="tool_validations",
            title="Validaciones y modo de ejecucion",
            ok=bool(tools)
            and all(
                tool.validations
                and tool.typed_errors
                and normalize_text(tool.execution_mode)
                and normalize_text(tool.timeout_policy)
                and normalize_text(tool.rate_limit_policy)
                for tool in tools
            ),
            pass_detail="Cada tool declara validaciones y execution mode.",
            fail_detail="Faltan validaciones, typed errors, timeout, rate limit o execution mode en una o mas tools.",
        ),
        _build_check_item(
            key="tool_reliability",
            title="Retry y compensacion",
            ok=bool(tools)
            and all(
                normalize_text(tool.retry_strategy)
                and normalize_text(tool.failure_mode)
                and normalize_text(tool.idempotency_strategy)
                and (not tool.has_side_effects or normalize_text(tool.compensation_strategy))
                for tool in tools
            ),
            pass_detail="Las tools declaran retry, idempotencia, failure mode y compensacion cuando aplica.",
            fail_detail="Persisten huecos de retry, idempotencia, failure mode o compensacion en tools con side effects.",
        ),
        _build_check_item(
            key="tool_governance",
            title="Gobernanza de side effects",
            ok=all(
                not tool.has_side_effects
                or (
                    tool.requires_approval
                    and normalize_text(tool.approval_reason)
                    and normalize_text(tool.approval_policy)
                )
                for tool in tools
            ),
            pass_detail="Toda tool con side effects exige approval gate visible.",
            fail_detail="Hay side effects sin approval gate, approval policy o razon de aprobacion.",
        ),
        _build_check_item(
            key="tool_integration_contract",
            title="Owner e integracion real",
            ok=bool(tools)
            and all(
                normalize_text(tool.owner)
                and normalize_text(tool.archetype)
                and normalize_text(tool.integration_kind)
                and normalize_text(tool.endpoint_reference)
                and normalize_text(tool.auth_reference)
                and tool.permissions
                and tool.scopes
                and tool.audit_rules
                for tool in tools
            ),
            pass_detail="Cada tool declara owner, integracion, endpoint, auth por referencia y auditoria.",
            fail_detail="Faltan owner, integracion, endpoint, auth por referencia, permisos, scopes o auditoria en una o mas tools.",
        ),
    ]
    return _summarize_component("tools", "Tools", checks)


def build_llm_policy_readiness(llm_policy: BlueprintLLMPolicy, tools: list[BlueprintTool]) -> ComponentReadinessEntry:
    tool_names = {tool.name for tool in tools if normalize_text(tool.name)}
    roles_by_key = {
        normalize_text(item.role).lower(): item
        for item in llm_policy.functions
        if normalize_text(item.role)
    }
    required_roles = {"planner", "executor", "evaluator"}
    if tool_names:
        required_roles.add("tool_use")

    checks = [
        _build_check_item(
            key="llm_provider_defaults",
            title="Provider y modelos base",
            ok=bool(
                normalize_text(llm_policy.provider)
                and normalize_text(llm_policy.fast_model)
                and normalize_text(llm_policy.reasoning_model)
                and normalize_text(llm_policy.fallback_model)
            ),
            pass_detail="La policy declara provider, fast model, reasoning model y fallback model.",
            fail_detail="Falta provider o uno de los modelos base de la policy LLM.",
            fail_status=ReviewState.blocked,
        ),
        _build_check_item(
            key="llm_global_policies",
            title="Politicas globales",
            ok=bool(
                normalize_text(llm_policy.context_policy)
                and normalize_text(llm_policy.sampling_policy)
                and normalize_text(llm_policy.fallback_policy)
                and normalize_text(llm_policy.circuit_breaker_policy)
                and normalize_text(llm_policy.budget_policy)
                and normalize_text(llm_policy.output_validation_policy)
                and normalize_text(llm_policy.log_redaction_policy)
            ),
            pass_detail="Contexto, sampling, fallback, circuit breaker, budget y redaccion estan declarados.",
            fail_detail="Falta una o mas politicas globales obligatorias para el runtime LLM.",
        ),
        _build_check_item(
            key="llm_required_roles",
            title="Roles obligatorios",
            ok=all(role in roles_by_key for role in required_roles),
            pass_detail="Planner, executor, evaluator y tool_use cuando aplica ya tienen politica declarada.",
            fail_detail="Falta alguna politica por rol obligatoria para el blueprint actual.",
        ),
        _build_check_item(
            key="llm_role_contracts",
            title="Contratos por rol",
            ok=all(
                normalize_text(item.provider)
                and normalize_text(item.model)
                and normalize_text(item.reasoning_effort)
                and item.max_tokens > 0
                and normalize_text(item.fallback_model)
                for item in roles_by_key.values()
            ),
            pass_detail="Cada rol declara provider, model, reasoning effort, max tokens y fallback model.",
            fail_detail="Persisten huecos en provider, model, effort, budget o fallback por rol.",
        ),
        _build_check_item(
            key="llm_tool_bindings",
            title="Binding de tools aprobadas",
            ok=(
                not tool_names
                or (
                    "tool_use" in roles_by_key
                    and set(roles_by_key["tool_use"].tool_availability).issuperset(tool_names)
                )
            ),
            pass_detail="La policy expone exactamente las tools aprobadas al rol que las consume.",
            fail_detail="La policy LLM no cubre todas las tools aprobadas del blueprint.",
        ),
    ]
    return _summarize_component("llm_policy", "LLM Policy", checks)


def build_memory_readiness(memory_profile: MemoryProfile) -> ComponentReadinessEntry:
    needs_checkpoints = memory_profile.strategy == "session_memory_with_checkpoints"
    has_sensitivity_rules = bool(normalize_list(memory_profile.sensitivity_rules))
    checks = [
        _build_check_item(
            key="memory_strategy",
            title="Estrategia seleccionada",
            ok=bool(normalize_text(memory_profile.strategy)),
            pass_detail="Existe una estrategia de memoria explicita.",
            fail_detail="No se selecciono estrategia de memoria.",
            fail_status=ReviewState.blocked,
        ),
        _build_check_item(
            key="memory_storage",
            title="Capas de almacenamiento",
            ok=bool(memory_profile.storage_layers),
            pass_detail="La memoria declara storage layers suficientes para el MVP.",
            fail_detail="No hay storage layers declaradas para la memoria.",
            fail_status=ReviewState.blocked,
        ),
        _build_check_item(
            key="memory_write_retrieval",
            title="Politicas de escritura y recuperacion",
            ok=bool(normalize_text(memory_profile.write_policy)) and bool(normalize_text(memory_profile.retrieval_policy)),
            pass_detail="Write policy y retrieval policy estan definidas.",
            fail_detail="Faltan politicas de escritura o recuperacion.",
        ),
        _build_check_item(
            key="memory_review_guard",
            title="Review trigger y goal drift guard",
            ok=bool(normalize_text(memory_profile.review_trigger)) and bool(normalize_text(memory_profile.goal_drift_guard)),
            pass_detail="Existen reglas para revisar y evitar goal drift.",
            fail_detail="Falta review trigger o goal drift guard.",
        ),
        _build_check_item(
            key="memory_retention_ttl",
            title="Retention, TTL y scopes",
            ok=bool(
                normalize_text(memory_profile.retention_policy)
                and normalize_text(memory_profile.ttl_policy)
                and normalize_text(memory_profile.workspace_scope)
                and normalize_text(memory_profile.agent_scope)
            ),
            pass_detail="La memoria separa alcance del workspace, alcance del agente, retention y TTL.",
            fail_detail="Falta definir retention, TTL o separar memoria del workspace frente a la memoria del agente.",
        ),
        _build_check_item(
            key="memory_grounding",
            title="Grounding y evidencia",
            ok=bool(
                normalize_text(memory_profile.grounding_policy.citations_policy)
                and normalize_text(memory_profile.grounding_policy.confidence_policy)
                and normalize_text(memory_profile.grounding_policy.no_evidence_behavior)
            ),
            pass_detail="La memoria declara citas, confianza y fallback sin evidencia.",
            fail_detail="Falta la politica de grounding para memoria o su fallback sin evidencia.",
        ),
        _build_check_item(
            key="memory_sensitivity",
            title="Sensibilidad y retencion alineadas",
            ok=(not has_sensitivity_rules) or bool(normalize_text(memory_profile.ttl_policy) and normalize_text(memory_profile.retention_policy)),
            pass_detail="Las reglas sensibles tienen TTL y retention coherentes.",
            fail_detail="Las reglas de sensibilidad exigen TTL y retention explicitas.",
        ),
        _build_check_item(
            key="memory_checkpoints",
            title="Checkpoints cuando aplica",
            ok=(not needs_checkpoints) or ("checkpoint_summary" in memory_profile.storage_layers),
            pass_detail="La memoria con checkpoints incluye una capa de resumen persistible.",
            fail_detail="La estrategia con checkpoints no declara checkpoint_summary en storage layers.",
        ),
    ]
    return _summarize_component("memory", "Memoria", checks)


def build_knowledge_readiness(knowledge_profile: KnowledgeProfile) -> ComponentReadinessEntry:
    knowledge_mode = normalize_text(knowledge_profile.mode).lower()
    rag_enabled = knowledge_mode == "rag"
    checks = [
        _build_check_item(
            key="knowledge_mode",
            title="Modo de knowledge",
            ok=bool(knowledge_mode),
            pass_detail=f"El blueprint declara mode={knowledge_mode}.",
            fail_detail="No se definio el modo de knowledge.",
            fail_status=ReviewState.blocked,
        ),
        _build_check_item(
            key="knowledge_sources",
            title="Fuentes con ownership y version",
            ok=(not rag_enabled)
            or bool(knowledge_profile.sources)
            and all(
                normalize_text(item.key)
                and normalize_text(item.title)
                and normalize_text(item.source_type)
                and normalize_text(item.uri)
                and normalize_text(item.owner)
                and normalize_text(item.sensitivity)
                and normalize_text(item.source_version)
                for item in knowledge_profile.sources
            ),
            pass_detail="Las fuentes de knowledge tienen owner, sensibilidad, URI abstracta y version.",
            fail_detail="El modo RAG exige fuentes con owner, sensibilidad, URI abstracta y version visibles.",
            fail_status=ReviewState.blocked if rag_enabled else ReviewState.partial,
        ),
        _build_check_item(
            key="knowledge_ingestion",
            title="Ingestion policy",
            ok=(not rag_enabled)
            or bool(
                normalize_text(knowledge_profile.ingestion_policy.parser)
                and normalize_text(knowledge_profile.ingestion_policy.chunking_policy)
                and knowledge_profile.ingestion_policy.metadata_fields
            ),
            pass_detail="Parser, chunking y metadata de ingestion quedaron definidos.",
            fail_detail="RAG requiere parser, chunking y metadata minima de ingestion.",
        ),
        _build_check_item(
            key="knowledge_embeddings_retrieval",
            title="Embeddings y retrieval",
            ok=(not rag_enabled)
            or bool(
                normalize_text(knowledge_profile.embedding_policy.provider)
                and normalize_text(knowledge_profile.embedding_policy.model)
                and knowledge_profile.embedding_policy.dimensions > 0
                and normalize_text(knowledge_profile.embedding_policy.version)
                and knowledge_profile.retrieval_policy.top_k > 0
                and normalize_text(knowledge_profile.retrieval_policy.search_mode)
                and normalize_text(knowledge_profile.retrieval_policy.fallback_behavior)
            ),
            pass_detail="Embeddings, top-k, modo de busqueda y fallback de retrieval quedaron definidos.",
            fail_detail="RAG exige provider/model/dimensions/version de embeddings y politica de retrieval con top-k y fallback.",
        ),
        _build_check_item(
            key="knowledge_refresh_lineage",
            title="Refresh y lineage",
            ok=(not rag_enabled)
            or bool(
                normalize_text(knowledge_profile.refresh_policy.frequency)
                and knowledge_profile.refresh_policy.triggers
                and normalize_text(knowledge_profile.refresh_policy.expiration_policy)
                and normalize_text(knowledge_profile.refresh_policy.deletion_policy)
                and all(normalize_text(item.source_version) for item in knowledge_profile.sources)
            ),
            pass_detail="Frecuencia, eventos, expiracion, borrado y versionado de fuentes quedaron visibles.",
            fail_detail="RAG requiere refresh policy y versionado de fuentes para mantener lineage trazable.",
        ),
        _build_check_item(
            key="knowledge_grounding",
            title="Grounding sensible",
            ok=(not rag_enabled)
            or bool(
                normalize_text(knowledge_profile.grounding_policy.citations_policy)
                and normalize_text(knowledge_profile.grounding_policy.confidence_policy)
                and normalize_text(knowledge_profile.grounding_policy.no_evidence_behavior)
                and normalize_text(knowledge_profile.grounding_policy.contradictory_evidence_behavior)
                and knowledge_profile.sensitivity_rules
            ),
            pass_detail="Knowledge declara citas, confianza, fallback sin evidencia y reglas de datos sensibles.",
            fail_detail="Faltan grounding policy o sensibilidad explicita para knowledge/RAG.",
        ),
    ]
    return _summarize_component("knowledge", "Knowledge", checks)


def build_security_readiness(
    tools: list[BlueprintTool],
    safety_checks: list[SafetyCheck],
    guardrails: list[str],
) -> tuple[ComponentReadinessEntry, RiskSummary]:
    high_risks = sum(1 for item in safety_checks if item.severity == "high")
    medium_risks = sum(1 for item in safety_checks if item.severity == "medium")
    low_risks = sum(1 for item in safety_checks if item.severity == "low")
    side_effect_tools = sum(1 for item in tools if item.has_side_effects)
    approval_gates_required = sum(1 for item in tools if item.requires_approval)
    approval_governed = all(
        not tool.has_side_effects or (tool.requires_approval and normalize_text(tool.approval_reason))
        for tool in tools
    )
    checks = [
        _build_check_item(
            key="safety_inventory",
            title="Matriz minima de riesgos",
            ok=bool(safety_checks),
            pass_detail=f"El blueprint declara {len(safety_checks)} safety checks.",
            fail_detail="No hay safety checks declarados.",
            fail_status=ReviewState.blocked,
        ),
        _build_check_item(
            key="safety_mitigations",
            title="Mitigaciones explicitas",
            ok=bool(safety_checks)
            and all(normalize_text(item.mitigation) and normalize_text(item.status) for item in safety_checks),
            pass_detail="Cada riesgo tiene mitigacion y estado.",
            fail_detail="Hay riesgos sin mitigacion o estado visible.",
        ),
        _build_check_item(
            key="guardrails_inventory",
            title="Guardrails visibles",
            ok=bool(guardrails),
            pass_detail=f"Se declararon {len(guardrails)} guardrails para el agente.",
            fail_detail="No hay guardrails visibles para gobernar el agente.",
            fail_status=ReviewState.blocked,
        ),
        _build_check_item(
            key="approval_governance",
            title="Approval gates para side effects",
            ok=approval_governed,
            pass_detail="Los side effects quedan protegidos por approval gates con razon explicita.",
            fail_detail="Persisten side effects sin approval gate o sin rationale.",
        ),
        _build_check_item(
            key="high_risk_visibility",
            title="Riesgos altos explicados",
            ok=high_risks == 0 or all(item.severity != "high" or normalize_text(item.mitigation) for item in safety_checks),
            pass_detail="Los riesgos altos cuentan con mitigacion visible.",
            fail_detail="Hay riesgos altos sin mitigacion suficientemente visible.",
        ),
    ]
    readiness = _summarize_component("security", "Seguridad", checks)
    if readiness.status == ReviewState.blocked:
        overall_status = ReviewState.blocked
    elif high_risks > 0 or not approval_governed:
        overall_status = ReviewState.partial
    else:
        overall_status = ReviewState.complete
    risk_summary = RiskSummary(
        overall_status=overall_status,
        total_checks=len(safety_checks),
        high_risks=high_risks,
        medium_risks=medium_risks,
        low_risks=low_risks,
        approval_gates_required=approval_gates_required,
        side_effect_tools=side_effect_tools,
        summary=(
            f"{high_risks} riesgos altos, {medium_risks} medios y {low_risks} bajos; "
            f"{approval_gates_required} gates de aprobacion requeridos para {side_effect_tools} tools con side effects."
        ),
    )
    return readiness, risk_summary


def derive_workflow_profile(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    architecture: str,
    tools: list[BlueprintTool],
) -> WorkflowProfile:
    approval_steps = [tool for tool in tools if tool.requires_approval or tool.has_side_effects]
    return WorkflowProfile(
        execution_pattern="durable_linear_workflow" if is_workflow_case(discovery.case_type) else "guided_builder_flow",
        inbox_strategy="Capturar intencion y validar campos antes de mutar el estado.",
        outbox_strategy="Separar la intencion del efecto final usando versionado y gates de aprobacion.",
        checkpoint_policy="Persistir snapshot al cerrar cada etapa y versionar cada patch del blueprint.",
        retry_strategy="Reintentar solo pasos idempotentes; reenfile manual para acciones con side effects.",
        compensation_strategy="Revertir la promocion del blueprint y volver a needs_review si un gate se rechaza.",
        approval_pause="Pausar antes de promover el blueprint o ejecutar herramientas con side effects."
        if approval_steps
        else "Sin pausa extra mientras no existan side effects.",
        timeout_policy="Escalar a revision humana si un gate pendiente excede el SLA del proyecto.",
        steps=[
            WorkflowStep(
                name="discover",
                objective="Normalizar el problema y clasificar el caso.",
                actor="builder",
                outputs=["normalized_discovery"],
                fallback="Solicitar datos faltantes antes de seguir.",
                requires_approval=False,
            ),
            WorkflowStep(
                name="define_canvas",
                objective="Recortar el MVP y documentar riesgo y metricas.",
                actor="builder",
                outputs=["canvas"],
                fallback="Revisar discovery si el objetivo no es claro.",
                requires_approval=False,
            ),
            WorkflowStep(
                name="design_blueprint",
                objective="Generar arquitectura, memoria, herramientas y artefactos iniciales.",
                actor="builder",
                outputs=["blueprint_package"],
                fallback="Reabrir el canvas o hacer patch manual.",
                requires_approval=False,
            ),
            WorkflowStep(
                name="approval_gate",
                objective="Solicitar aprobacion humana antes de la promocion a implementacion.",
                actor="human_reviewer",
                outputs=["approved_blueprint_package" if approval_steps else "review_acknowledged"],
                fallback="Mantener la sesion en needs_review hasta resolver observaciones.",
                requires_approval=bool(approval_steps),
            ),
            WorkflowStep(
                name="implementation_handoff",
                objective=f"Entregar el paquete final para ejecutar el MVP de '{canvas.user_goal}'.",
                actor="delivery_workflow",
                outputs=["implementation_handoff"],
                fallback="Volver al builder y actualizar backlog o riesgos.",
                requires_approval=bool(approval_steps),
            ),
        ],
    )


def derive_observability_plan(discovery: DiscoveryArtifact) -> ObservabilityPlan:
    return ObservabilityPlan(
        captured_signals=[
            "input",
            "plan_summary",
            "tools_called",
            "tool_responses",
            "state",
            "errors",
            "decisions",
            "cost",
            "duration",
            "final_result",
        ],
        plan_summary_policy="Registrar un resumen corto por etapa en activity y export final.",
        tool_response_logging="Guardar payload estructurado de cada accion del builder en execution_logs.",
        decision_logging="Capturar arquitectura, memoria, risk gates y readiness en cada snapshot.",
        cost_tracking="Registrar costo estimado por llamada LLM cuando el proveedor este configurado.",
        duration_tracking="Medir la duracion por etapa y marcar hotspots operativos en monitoreo.",
        alert_triggers=[
            "Campos criticos faltantes",
            "Gates de aprobacion pendientes",
            "Blueprint parcial o bloqueado",
            "Intentos de side effects sin aprobacion",
        ],
        result_tracking="Versionar el blueprint, el export Markdown y la evaluacion asociada a la sesion.",
    )


def _tool_schema_markdown(tools: list[BlueprintTool]) -> str:
    sections: list[str] = ["# Esquema de Tools", ""]
    for tool in tools:
        sections.extend(
            [
                f"## {tool.name}",
                f"- proposito: {tool.purpose}",
                f"- inputs: {', '.join(tool.inputs)}",
                f"- outputs: {', '.join(tool.outputs)}",
                f"- validaciones: {', '.join(tool.validations)}",
                f"- requires_approval: {tool.requires_approval}",
                f"- has_side_effects: {tool.has_side_effects}",
                f"- execution_mode: {tool.execution_mode}",
                f"- retry_strategy: {tool.retry_strategy}",
                f"- compensation_strategy: {tool.compensation_strategy}",
                f"- failure_mode: {tool.failure_mode}",
                "",
            ]
        )
    return "\n".join(sections).strip()


def _state_flow_markdown(workflow_profile: WorkflowProfile) -> str:
    sections = [
        "# Flujo de Estados",
        "",
        f"- execution_pattern: {workflow_profile.execution_pattern}",
        f"- checkpoint_policy: {workflow_profile.checkpoint_policy}",
        f"- approval_pause: {workflow_profile.approval_pause}",
        "",
        "## Steps",
    ]
    for step in workflow_profile.steps:
        sections.extend(
            [
                f"- {step.name}",
                f"  - actor: {step.actor}",
                f"  - objective: {step.objective}",
                f"  - outputs: {', '.join(step.outputs)}",
                f"  - fallback: {step.fallback}",
                f"  - requires_approval: {step.requires_approval}",
            ]
        )
    return "\n".join(sections).strip()


def _decision_trace_markdown(decision_summary: str, decision_trace: list[DecisionTraceEntry]) -> str:
    sections = ["# Reporte de decision", "", decision_summary, "", "## Trazabilidad por capa"]
    for entry in decision_trace:
        sections.extend(
            [
                f"- {entry.dimension}",
                f"  - seleccionado: {entry.selected_label} ({entry.selected_value})",
                f"  - recomendado: {entry.recommended_label} ({entry.recommended_value})",
                f"  - source: {entry.decision_source}",
                f"  - rationale: {entry.rationale}",
                f"  - review_note: {entry.review_note}",
                f"  - evidence: {', '.join(entry.evidence)}",
            ]
        )
    return "\n".join(sections).strip()


def _component_readiness_markdown(
    component_readiness: list[ComponentReadinessEntry],
    risk_summary: RiskSummary,
) -> str:
    sections = [
        "# Checklist de completitud",
        "",
        f"- seguridad_global: {risk_summary.overall_status}",
        f"- resumen_riesgo: {risk_summary.summary}",
        "",
    ]
    for item in component_readiness:
        sections.extend(
            [
                f"## {item.label}",
                f"- status: {item.status}",
                f"- score: {item.score}",
                f"- completed_checks: {item.completed_checks}/{item.total_checks}",
                "- checks:",
            ]
        )
        for check in item.checks:
            sections.append(f"  - {check.title} [{check.status}] -> {check.detail}")
        if item.blocking_issues:
            sections.extend(["- blocking_issues:", *[f"  - {issue}" for issue in item.blocking_issues]])
        sections.append("")
    return "\n".join(sections).strip()


def build_roadmap_evolution(
    architecture: str,
    reasoning_pattern: str,
    memory_strategy: str,
) -> RoadmapEvolution:
    release_map = {
        "MVP 1": RoadmapMilestone(
            release="MVP 1",
            title="Blueprint Lean gobernado",
            objective="Cerrar discovery, canvas, blueprint, exportes y gates humanos con una sola sesion operable.",
            when_to_unlock="Base actual del builder y primer handoff tecnico consistente.",
            capabilities=[
                f"Arquitectura inicial: {architecture}",
                f"Patron de razonamiento: {reasoning_pattern}",
                f"Memoria base: {memory_strategy}",
                "Tools con contratos minimos y approval gates visibles",
                "Exportes JSON y Markdown gobernados",
            ],
        ),
        "MVP 2": RoadmapMilestone(
            release="MVP 2",
            title="Evaluacion con evidencia",
            objective="Agregar datasets, rubricas, pruebas de contexto y fallos de tools sin rearmar el runtime base.",
            when_to_unlock="Cuando el blueprint ya se use en sesiones repetibles y se necesite evidencia de calidad.",
            capabilities=[
                "Datasets persistidos por blueprint",
                "Rubricas y scoring por categoria",
                "Pruebas de contexto y recuperacion",
                "Matriz de riesgos mas profunda",
            ],
        ),
        "MVP 3": RoadmapMilestone(
            release="MVP 3",
            title="Operacion real y escalado controlado",
            objective="Activar handoffs, monitoreo real, gobierno y capacidades especializadas solo cuando exista evidencia de necesidad.",
            when_to_unlock="Cuando la validacion del agente deje de ser local y se necesite operar con trazabilidad continua.",
            capabilities=[
                "Plantillas de workflow durable",
                "Monitoreo y alertas activas",
                "Handoffs y gobierno por politica",
                "Subagentes solo bajo control explicito",
            ],
        ),
    }
    return RoadmapEvolution(
        current_release="MVP 1",
        current_focus="Cerrar el agente correcto antes de complejizar el sistema agentico.",
        milestones=[release_map["MVP 1"], release_map["MVP 2"], release_map["MVP 3"]],
    )


def _roadmap_markdown(roadmap: RoadmapEvolution) -> str:
    sections = [
        "# Roadmap de evolucion",
        "",
        f"- current_release: {roadmap.current_release}",
        f"- current_focus: {roadmap.current_focus}",
        "",
    ]
    for item in roadmap.milestones:
        sections.extend(
            [
                f"## {item.release} - {item.title}",
                f"- objective: {item.objective}",
                f"- when_to_unlock: {item.when_to_unlock}",
                "- capabilities:",
                *[f"  - {capability}" for capability in item.capabilities],
                "",
            ]
        )
    return "\n".join(sections).strip()


def build_blueprint_coverage(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    architecture: str,
    reasoning_pattern: str,
    tools: list[BlueprintTool],
    llm_policy: BlueprintLLMPolicy,
    memory_profile: MemoryProfile,
    knowledge_profile: KnowledgeProfile,
    workflow_profile: WorkflowProfile,
    safety_checks: list[SafetyCheck],
    guardrails: list[str],
    observability_plan: ObservabilityPlan,
    roadmap: RoadmapEvolution,
) -> BlueprintCoverageSummary:
    knowledge_mode = normalize_text(knowledge_profile.mode).lower()
    knowledge_complete = knowledge_mode != "rag" or (
        bool(knowledge_profile.sources)
        and normalize_text(knowledge_profile.ingestion_policy.chunking_policy)
        and normalize_text(knowledge_profile.embedding_policy.provider)
        and knowledge_profile.embedding_policy.dimensions > 0
        and knowledge_profile.retrieval_policy.top_k > 0
        and normalize_text(knowledge_profile.refresh_policy.frequency)
    )
    sections = [
        BlueprintSectionCoverageEntry(
            key="problem",
            title="Problema que resuelve",
            status=ReviewState.complete if normalize_text(discovery.problem_statement) else ReviewState.blocked,
            source="discovery.problem_statement",
            note=discovery.problem_statement or "Falta definir el problema.",
        ),
        BlueprintSectionCoverageEntry(
            key="target_user",
            title="Usuario objetivo",
            status=ReviewState.complete if normalize_text(canvas.agent_profile.primary_user) else ReviewState.blocked,
            source="canvas.agent_profile.primary_user",
            note=canvas.agent_profile.primary_user or "Falta identificar el usuario principal.",
        ),
        BlueprintSectionCoverageEntry(
            key="expected_result",
            title="Resultado esperado",
            status=ReviewState.complete if normalize_text(discovery.desired_outcome) else ReviewState.blocked,
            source="discovery.desired_outcome",
            note=discovery.desired_outcome or "Falta definir el resultado esperado.",
        ),
        BlueprintSectionCoverageEntry(
            key="mvp_scope",
            title="Alcance del MVP",
            status=ReviewState.complete if canvas.mvp_scope and canvas.out_of_scope else ReviewState.partial,
            source="canvas.mvp_scope + canvas.out_of_scope",
            note="El alcance y el fuera de alcance ya quedaron estructurados." if canvas.mvp_scope and canvas.out_of_scope else "Falta cerrar alcance y exclusiones del MVP.",
        ),
        BlueprintSectionCoverageEntry(
            key="architecture",
            title="Arquitectura elegida",
            status=ReviewState.complete if normalize_text(architecture) else ReviewState.blocked,
            source="blueprint.architecture",
            note=architecture or "Falta arquitectura seleccionada.",
        ),
        BlueprintSectionCoverageEntry(
            key="reasoning",
            title="Patron de razonamiento",
            status=ReviewState.complete if normalize_text(reasoning_pattern) else ReviewState.blocked,
            source="blueprint.reasoning_pattern",
            note=reasoning_pattern or "Falta patron de razonamiento.",
        ),
        BlueprintSectionCoverageEntry(
            key="tools",
            title="Herramientas y contratos",
            status=ReviewState.complete if tools else ReviewState.blocked,
            source="blueprint.tools",
            note="Las tools incluyen contratos, validaciones y modo de falla." if tools else "No hay tools definidas.",
        ),
        BlueprintSectionCoverageEntry(
            key="llm_policy",
            title="Politica LLM",
            status=(
                ReviewState.complete
                if normalize_text(llm_policy.provider)
                and normalize_text(llm_policy.fast_model)
                and normalize_text(llm_policy.reasoning_model)
                and llm_policy.functions
                else ReviewState.partial
            ),
            source="blueprint.llm_policy",
            note=(
                "La policy LLM ya define provider, modelos base y funciones por rol."
                if normalize_text(llm_policy.provider)
                and normalize_text(llm_policy.fast_model)
                and normalize_text(llm_policy.reasoning_model)
                and llm_policy.functions
                else "Falta cerrar provider, modelos o bindings por rol de la policy LLM."
            ),
        ),
        BlueprintSectionCoverageEntry(
            key="memory",
            title="Memoria y contexto",
            status=(
                ReviewState.complete
                if normalize_text(memory_profile.strategy) and memory_profile.storage_layers and knowledge_complete
                else ReviewState.partial
            ),
            source="blueprint.memory_profile",
            note=(
                "La memoria y el knowledge/RAG quedaron definidos con trazabilidad."
                if normalize_text(memory_profile.strategy) and memory_profile.storage_layers and knowledge_complete
                else "Faltan estrategia/capas de memoria o definiciones de knowledge/RAG."
            ),
        ),
        BlueprintSectionCoverageEntry(
            key="execution_flow",
            title="Flujo de ejecucion",
            status=ReviewState.complete if workflow_profile.steps else ReviewState.blocked,
            source="delivery_package.workflow_profile",
            note="El workflow durable ya describe pasos, fallback y approvals." if workflow_profile.steps else "Falta modelar el flujo operativo.",
        ),
        BlueprintSectionCoverageEntry(
            key="human_in_loop",
            title="Human-in-the-loop",
            status=ReviewState.complete if canvas.agent_profile.human_approvals else ReviewState.partial,
            source="canvas.agent_profile.human_approvals",
            note="Las aprobaciones humanas quedaron visibles." if canvas.agent_profile.human_approvals else "Falta declarar aprobaciones humanas.",
        ),
        BlueprintSectionCoverageEntry(
            key="risks_guardrails",
            title="Riesgos y guardrails",
            status=ReviewState.complete if safety_checks and guardrails else ReviewState.partial,
            source="blueprint.safety_checks + blueprint.guardrails",
            note="Ya existe matriz minima de riesgos y guardrails." if safety_checks and guardrails else "Faltan riesgos o guardrails visibles.",
        ),
        BlueprintSectionCoverageEntry(
            key="evaluation_initial",
            title="Evaluacion inicial",
            status=ReviewState.complete,
            source="delivery_package.test_cases",
            note="El paquete ya incluye casos de prueba iniciales para validar antes de escalar.",
        ),
        BlueprintSectionCoverageEntry(
            key="monitoring",
            title="Monitoreo",
            status=ReviewState.complete if observability_plan.captured_signals else ReviewState.partial,
            source="delivery_package.observability_plan",
            note="El plan de observabilidad ya define senales, logging y alertas." if observability_plan.captured_signals else "Falta plan de observabilidad.",
        ),
        BlueprintSectionCoverageEntry(
            key="roadmap",
            title="Roadmap de evolucion",
            status=ReviewState.complete if roadmap.milestones else ReviewState.partial,
            source="delivery_package.roadmap_evolution",
            note="El roadmap explicita como pasar de MVP 1 a MVP 3." if roadmap.milestones else "Falta roadmap de evolucion.",
        ),
    ]
    covered_sections = sum(1 for item in sections if item.status == ReviewState.complete)
    missing_sections = [item.title for item in sections if item.status != ReviewState.complete]
    overall_status = ReviewState.complete if covered_sections == len(sections) else ReviewState.partial
    return BlueprintCoverageSummary(
        overall_status=overall_status,
        covered_sections=covered_sections,
        total_sections=len(sections),
        missing_sections=missing_sections,
        sections=sections,
    )


def _blueprint_coverage_markdown(coverage: BlueprintCoverageSummary) -> str:
    sections = [
        "# Cobertura del blueprint final",
        "",
        f"- overall_status: {coverage.overall_status}",
        f"- covered_sections: {coverage.covered_sections}/{coverage.total_sections}",
        "",
    ]
    for item in coverage.sections:
        sections.extend(
            [
                f"- {item.title}",
                f"  - key: {item.key}",
                f"  - status: {item.status}",
                f"  - source: {item.source}",
                f"  - note: {item.note}",
            ]
        )
    if coverage.missing_sections:
        sections.extend(["", "- missing_sections:", *[f"  - {item}" for item in coverage.missing_sections]])
    return "\n".join(sections).strip()


def build_delivery_package(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    architecture: str,
    reasoning_pattern: str,
    memory_strategy: str,
    tools: list[BlueprintTool],
    llm_policy: BlueprintLLMPolicy,
    memory_profile: MemoryProfile,
    knowledge_profile: KnowledgeProfile,
    safety_checks: list[SafetyCheck],
    guardrails: list[str],
) -> DeliveryPackage:
    workflow_profile = derive_workflow_profile(discovery, canvas, architecture, tools)
    observability_plan = derive_observability_plan(discovery)
    decision_summary, decision_trace, pattern_catalog = build_decision_report(
        discovery,
        canvas,
        architecture,
        reasoning_pattern,
        memory_strategy,
    )
    tools_readiness = build_tools_readiness(tools)
    llm_readiness = build_llm_policy_readiness(llm_policy, tools)
    memory_readiness = build_memory_readiness(memory_profile)
    knowledge_readiness = build_knowledge_readiness(knowledge_profile)
    security_readiness, risk_summary = build_security_readiness(tools, safety_checks, guardrails)
    component_readiness = [tools_readiness, llm_readiness, memory_readiness, knowledge_readiness, security_readiness]
    roadmap_evolution = build_roadmap_evolution(architecture, reasoning_pattern, memory_strategy)
    blueprint_coverage = build_blueprint_coverage(
        discovery,
        canvas,
        architecture,
        reasoning_pattern,
        tools,
        llm_policy,
        memory_profile,
        knowledge_profile,
        workflow_profile,
        safety_checks,
        guardrails,
        observability_plan,
        roadmap_evolution,
    )

    prd_content = "\n".join(
        [
            "# PRD del agente",
            "",
            f"- problema: {discovery.problem_statement}",
            f"- usuario objetivo: {canvas.agent_profile.primary_user}",
            f"- resultado esperado: {discovery.desired_outcome}",
            f"- objetivo del agente: {canvas.user_goal}",
            f"- metrica norte: {discovery.mvp_definition.north_star_metric or canvas.success_metric}",
            f"- tiempo actual: {discovery.operational_baseline.current_time_spent}",
            f"- costo actual: {discovery.operational_baseline.current_cost}",
            "",
            "## Baseline operativo",
            *[f"- error frecuente: {item}" for item in discovery.operational_baseline.frequent_errors],
            *[f"- oportunidad de automatizacion: {item}" for item in discovery.operational_baseline.automation_opportunities],
            "",
            "## Alcance MVP",
            *[f"- {item}" for item in canvas.mvp_scope],
            "",
            "## Fuera de alcance",
            *[f"- {item}" for item in canvas.out_of_scope],
            "",
            "## Decisiones no delegables",
            *[f"- {item}" for item in discovery.mvp_definition.non_delegable_decisions],
            "",
            "## Riesgo principal",
            f"- {canvas.primary_risk}",
        ]
    ).strip()

    technical_spec_content = "\n".join(
        [
            "# Especificacion tecnica inicial",
            "",
            f"- arquitectura: {architecture}",
            f"- razonamiento: {reasoning_pattern}",
            f"- memoria: {memory_strategy}",
            f"- llm provider: {llm_policy.provider or 'pending'}",
            f"- llm models: fast={llm_policy.fast_model or 'pending'} | reasoning={llm_policy.reasoning_model or 'pending'}",
            f"- estrategia de observabilidad: {observability_plan.plan_summary_policy}",
            f"- decision_summary: {decision_summary}",
            f"- blueprint_coverage: {blueprint_coverage.covered_sections}/{blueprint_coverage.total_sections}",
            "",
            "## Modulos minimos",
            "- intake y normalizacion de discovery",
            "- canvas Lean y policy gates",
            "- blueprint tecnico con versionado",
            "- evaluacion inicial y export final",
        ]
    ).strip()

    system_prompt_content = "\n".join(
        [
            "# System Prompt Base",
            "",
            "Eres un constructor Lean de agentes.",
            "Primero reduces ambiguedad, luego recortas alcance y solo despues recomiendas arquitectura.",
            "Nunca inventas campos faltantes.",
            "No ejecutas side effects sin approval gate.",
            f"Tu objetivo principal es: {canvas.user_goal}.",
        ]
    ).strip()

    skill_spec_content = "\n".join(
        [
            "# Skill Specs Base",
            "",
            "- discovery_skill: captura problema, usuario, restricciones y valor.",
            "- lean_scope_skill: convierte el caso en alcance MVP y fuera de alcance.",
            f"- architecture_selection_skill: recomienda {architecture} y explica tradeoffs.",
            f"- reasoning_pattern_skill: aplica {reasoning_pattern} segun el caso.",
            "- tool_design_skill: define contratos, validaciones y approval gates.",
            "- memory_design_skill: protege contexto, checkpoints y goal drift.",
            "- safety_skill: revisa riesgos y decisiones no delegables antes del handoff.",
            "- blueprint_generation_skill: empaqueta la salida final para implementacion.",
        ]
    ).strip()

    risk_matrix_content = "\n".join(
        [
            "# Matriz de riesgos",
            "",
            f"- resumen: {risk_summary.summary}",
            f"- overall_status: {risk_summary.overall_status}",
            "",
            *[
                f"- {item.category} | severidad={item.severity} | riesgo={item.risk} | mitigacion={item.mitigation}"
                for item in safety_checks
            ],
        ]
    ).strip()

    test_cases_content = "\n".join(
        [
            "# Casos de prueba iniciales",
            "",
            "- Happy path completo desde discovery hasta export.",
            "- Discovery con campos faltantes y bloqueo por validation gate.",
            "- Patch manual del blueprint y recreacion del snapshot.",
            "- Approval gate pendiente antes del handoff a implementacion.",
            "- Rechazo del gate y retorno a needs_review.",
        ]
    ).strip()

    backlog_content = "\n".join(
        [
            "# Backlog MVP",
            "",
            "- Implementar canvas enriquecido y artefactos persistidos.",
            "- Exponer approval gates y resolucion humana.",
            "- Versionar export Markdown y paquete tecnico.",
            "- Expandir evaluacion con contexto, riesgo y fallos de tools.",
        ]
    ).strip()

    deliverables = [
        GeneratedDeliverable(
            key="prd",
            title="PRD del agente",
            summary="Documento base con problema, objetivo, alcance MVP y riesgo principal.",
            content_markdown=prd_content,
        ),
        GeneratedDeliverable(
            key="technical_spec",
            title="Especificacion tecnica inicial",
            summary="Arquitectura, modulos minimos y estrategia operativa del builder.",
            content_markdown=technical_spec_content,
        ),
        GeneratedDeliverable(
            key="system_prompt",
            title="System prompt base",
            summary="Prompt inicial para conservar el comportamiento Lean del agente.",
            content_markdown=system_prompt_content,
        ),
        GeneratedDeliverable(
            key="skill_spec",
            title="Skill specs base",
            summary="Listado inicial de skills y responsabilidad de cada una.",
            content_markdown=skill_spec_content,
        ),
        GeneratedDeliverable(
            key="tool_schema",
            title="Esquema de tools",
            summary="Contratos minimos con validaciones, retries y compensaciones.",
            content_markdown=_tool_schema_markdown(tools),
        ),
        GeneratedDeliverable(
            key="state_flow",
            title="Flujo de estados",
            summary="Workflow durable sugerido para operar el agente con checkpoints.",
            content_markdown=_state_flow_markdown(workflow_profile),
        ),
        GeneratedDeliverable(
            key="decision_trace",
            title="Reporte de decision",
            summary="Explica por que el motor de reglas eligio arquitectura, razonamiento y memoria.",
            content_markdown=_decision_trace_markdown(decision_summary, decision_trace),
        ),
        GeneratedDeliverable(
            key="component_checklist",
            title="Checklist de completitud",
            summary="Semaforo por componente para tools, memoria y seguridad antes del handoff.",
            content_markdown=_component_readiness_markdown(component_readiness, risk_summary),
        ),
        GeneratedDeliverable(
            key="test_cases",
            title="Casos de prueba iniciales",
            summary="Cobertura minima para validar el agente antes de escalar.",
            content_markdown=test_cases_content,
        ),
        GeneratedDeliverable(
            key="risk_matrix",
            title="Matriz de riesgos",
            summary="Mapa inicial de riesgos y mitigaciones requeridas.",
            content_markdown=risk_matrix_content,
        ),
        GeneratedDeliverable(
            key="mvp_backlog",
            title="Backlog MVP",
            summary="Lista corta de trabajo para ejecutar el primer release del agente.",
            content_markdown=backlog_content,
        ),
        GeneratedDeliverable(
            key="evolution_roadmap",
            title="Roadmap de evolucion",
            summary="Secuencia propuesta para pasar de MVP 1 a MVP 3 sin sobredisenar.",
            content_markdown=_roadmap_markdown(roadmap_evolution),
        ),
    ]

    return DeliveryPackage(
        workflow_profile=workflow_profile,
        observability_plan=observability_plan,
        deliverables=deliverables,
        decision_summary=decision_summary,
        decision_trace=decision_trace,
        pattern_catalog=pattern_catalog,
        component_readiness=component_readiness,
        risk_summary=risk_summary,
        roadmap_evolution=roadmap_evolution,
        blueprint_coverage=blueprint_coverage,
    )


def derive_readiness_state(blueprint: BlueprintArtifact) -> ReviewState:
    if (
        not blueprint.tools
        or not blueprint.guardrails
        or not blueprint.memory_profile.storage_layers
        or not normalize_text(blueprint.llm_policy.provider)
    ):
        return ReviewState.blocked
    if not blueprint.delivery_package.deliverables:
        return ReviewState.blocked
    component_states = [item.status for item in blueprint.delivery_package.component_readiness]
    if component_states and any(item == ReviewState.blocked for item in component_states):
        return ReviewState.blocked
    if any(tool.has_side_effects and not tool.requires_approval for tool in blueprint.tools):
        return ReviewState.partial
    if any(tool.has_side_effects and not tool.compensation_strategy for tool in blueprint.tools):
        return ReviewState.partial
    if any(tool.has_side_effects and not tool.retry_strategy for tool in blueprint.tools):
        return ReviewState.partial
    if any(tool.requires_approval and not tool.approval_reason for tool in blueprint.tools):
        return ReviewState.partial
    if any(
        not normalize_text(tool.owner)
        or not normalize_text(tool.archetype)
        or not normalize_text(tool.integration_kind)
        or not normalize_text(tool.endpoint_reference)
        or not normalize_text(tool.auth_reference)
        or not normalize_text(tool.approval_policy)
        or not normalize_text(tool.rate_limit_policy)
        or not normalize_text(tool.timeout_policy)
        or not normalize_text(tool.idempotency_strategy)
        or not tool.typed_errors
        or not tool.permissions
        or not tool.scopes
        or not tool.audit_rules
        for tool in blueprint.tools
    ):
        return ReviewState.partial
    if not blueprint.llm_policy.functions:
        return ReviewState.partial
    if any(
        not normalize_text(blueprint.llm_policy.fast_model)
        or not normalize_text(blueprint.llm_policy.reasoning_model)
        or not normalize_text(blueprint.llm_policy.fallback_model)
        or not normalize_text(blueprint.llm_policy.context_policy)
        or not normalize_text(blueprint.llm_policy.sampling_policy)
        or not normalize_text(blueprint.llm_policy.fallback_policy)
        or not normalize_text(blueprint.llm_policy.circuit_breaker_policy)
        or not normalize_text(blueprint.llm_policy.budget_policy)
        or not normalize_text(blueprint.llm_policy.output_validation_policy)
        or not normalize_text(blueprint.llm_policy.log_redaction_policy)
        for _ in [0]
    ):
        return ReviewState.partial
    if component_states and any(item == ReviewState.partial for item in component_states):
        return ReviewState.partial
    return ReviewState.complete


def build_evaluation_cases(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    blueprint: BlueprintArtifact,
) -> list[EvaluationCase]:
    return [
        EvaluationCase(
            name="happy_path",
            category="functional",
            scenario="El usuario completa discovery, genera canvas, blueprint, approval gates y export final.",
            expected_result="Se genera un paquete consistente con entregables y trazabilidad por etapa.",
        ),
        EvaluationCase(
            name="missing_required_fields",
            category="validation",
            scenario="Faltan campos obligatorios en discovery.",
            expected_result="El sistema devuelve missing_fields y no avanza a canvas.",
        ),
        EvaluationCase(
            name="canvas_contract",
            category="consistency",
            scenario="Se revisa que el canvas incluya mision, decisiones, inputs, outputs y aprobaciones.",
            expected_result="El agent_profile del canvas queda completo y alineado al problema.",
        ),
        EvaluationCase(
            name="approval_gate_before_handoff",
            category="safety",
            scenario="Existe una tool con side effects y se intenta promover el blueprint sin gate humano.",
            expected_result="La promocion exige aprobacion previa y deja trazabilidad del gate.",
        ),
        EvaluationCase(
            name="artifact_package_export",
            category="delivery",
            scenario="El usuario exporta el blueprint final con artefactos tecnicos incluidos.",
            expected_result="La salida Markdown contiene PRD, spec tecnica, tool schema, state flow y backlog.",
        ),
    ]


def build_evaluation_artifact(
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact | None,
    *,
    dataset: EvaluationDatasetArtifact | None = None,
    rubric: EvaluationRubricArtifact | None = None,
) -> EvaluationArtifact:
    current_dataset = dataset or build_default_evaluation_dataset(discovery, canvas, blueprint)
    current_rubric = rubric or build_default_evaluation_rubric()
    run_summary = score_evaluation_workbench(
        current_dataset,
        current_rubric,
        discovery,
        canvas,
        blueprint,
        source_action="evaluation_skill",
    )
    return build_evaluation_artifact_from_run(current_dataset, run_summary)
