from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from app.models import CanvasArtifact, DiscoveryArtifact, ToolRecommendationLLMOutput
from app.services.diagram_center.contracts import StructuredDiagramModel
from app.services.llm_runtime.builder_contracts import (
    AgentDesignProposalOutput,
    BlueprintNarrativeOutput,
    DesignCritiqueOutput,
    DiscoveryAnalysisOutput,
    EstimationRiskAnalysisOutput,
    MemoryArchitectureCritiqueOutput,
    MemoryArchitectureRecommendationOutput,
    RequirementsDefinitionOutput,
    ValidationRunJudgmentOutput,
    ValidationScenarioGenerationOutput,
    ValidationSimulationOutput,
)
from app.services.llm_runtime.prompt_templates import (
    build_tool_recommendation_registry_task_instruction,
    build_tool_recommendation_system_instruction,
)


class BuilderCapability(StrEnum):
    normalize_discovery = "normalize_discovery"
    analyze_discovery = "analyze_discovery"
    build_canvas = "build_canvas"
    define_requirements = "define_requirements"
    synthesize_blueprint_narrative = "synthesize_blueprint_narrative"
    propose_agent_design = "propose_agent_design"
    critique_agent_design = "critique_agent_design"
    recommend_minimal_tools = "recommend_minimal_tools"
    recommend_memory_architecture = "recommend_memory_architecture"
    critique_memory_architecture = "critique_memory_architecture"
    generate_validation_scenarios = "generate_validation_scenarios"
    simulate_validation_scenario = "simulate_validation_scenario"
    judge_validation_run = "judge_validation_run"
    analyze_estimation_risks = "analyze_estimation_risks"
    generate_diagram_model = "generate_diagram_model"


@dataclass(frozen=True)
class BuilderCapabilitySpec:
    capability: BuilderCapability
    task_kind: str
    prompt_version: str
    source_key: str
    source_title: str
    source_summary: str
    system_instruction: str
    task_instruction: str
    output_model: type[BaseModel]
    preferred_model: str
    llm_required: bool
    critic_required: bool
    timeout_ms: int
    max_retries: int
    fallback_policy: str


CAPABILITY_ALIASES: dict[BuilderCapability, set[str]] = {
    BuilderCapability.normalize_discovery: {
        "normalize_discovery",
        "discovery",
        "discover",
        "draft_capture",
        "input_validation",
        "discovery_skill",
        "discovery_normalization",
    },
    BuilderCapability.analyze_discovery: {
        "analyze_discovery",
        "discovery_analysis",
        "discover_analysis",
        "discovery_gaps",
    },
    BuilderCapability.build_canvas: {
        "build_canvas",
        "canvas",
        "define",
        "lean_scope_skill",
        "canvas_generation",
    },
    BuilderCapability.define_requirements: {
        "define_requirements",
        "requirements",
        "requirements_definition",
        "define_specs",
    },
    BuilderCapability.synthesize_blueprint_narrative: {
        "synthesize_blueprint_narrative",
        "blueprint_narrative",
        "build_blueprint",
        "blueprint",
        "design",
        "blueprint_generation_skill",
        "prd_narrative",
    },
    BuilderCapability.propose_agent_design: {
        "propose_agent_design",
        "agent_design",
        "design_proposal",
        "architecture_proposal",
    },
    BuilderCapability.critique_agent_design: {
        "critique_agent_design",
        "design_critique",
        "design_review",
        "architecture_critique",
    },
    BuilderCapability.recommend_minimal_tools: {
        "recommend_minimal_tools",
        "tool_recommendation",
        "tool_recommendation_skill",
        "tools",
        "recommend_tools",
    },
    BuilderCapability.recommend_memory_architecture: {
        "recommend_memory_architecture",
        "memory_architecture",
        "memory_design",
        "memory_recommendation",
    },
    BuilderCapability.critique_memory_architecture: {
        "critique_memory_architecture",
        "memory_critique",
        "memory_review",
    },
    BuilderCapability.generate_validation_scenarios: {
        "generate_validation_scenarios",
        "validation_scenarios",
        "validate_scenarios",
    },
    BuilderCapability.simulate_validation_scenario: {
        "simulate_validation_scenario",
        "validation_simulation",
        "validate_simulation",
    },
    BuilderCapability.judge_validation_run: {
        "judge_validation_run",
        "validation_judgment",
        "validate_judgment",
    },
    BuilderCapability.analyze_estimation_risks: {
        "analyze_estimation_risks",
        "estimation_risks",
        "estimate_risks",
        "estimation_analysis",
    },
    BuilderCapability.generate_diagram_model: {
        "generate_diagram_model",
        "diagram_generation",
        "diagrams",
        "diagram_center",
        "architecture_diagram",
    },
}


GUIDED_QUESTION_INSTRUCTION = (
    "Si generas preguntas, primero intenta inferir la respuesta con el contexto y las fuentes aprobadas. "
    "Formula solo preguntas indispensables para el objetivo funcional de la etapa actual y no adelantes "
    "decisiones de etapas posteriores. Antes de preguntar, documenta implicitamente que intentaste inferir "
    "la respuesta; si la confianza es razonable, usa la inferencia como supuesto trazable en vez de preguntar. "
    "Cada pregunta nueva debe incluir `suggested_answer`; si existen varias rutas validas, agrega "
    "`answer_options` con 2 a 4 opciones, maximo una marcada como recommended, e incluye description, impact, "
    "example, confidence y source_refs cuando haya evidencia. Las opciones deben ser comprensibles para el "
    "owner funcional, no para un administrador tecnico. Si la pregunta es tecnica de implementacion temprana "
    "(framework, base de datos, credenciales, despliegue, endpoints finales, configuracion de entorno o "
    "contratos fisicos), no la hagas al usuario durante Blueprint: registrala como supuesto, riesgo o "
    "decision diferida hacia ACP con impacto y momento de cierre."
)

DISCOVERY_SCOPE_INSTRUCTION = (
    "Alcance Discover: problema, usuario afectado, proceso actual, resultado esperado y restricciones de negocio inmediatas. "
    "Difiere herramientas, integraciones, memoria, RAG, stack, infraestructura, despliegue, credenciales y contratos a etapas posteriores."
)

DEFINE_SCOPE_INSTRUCTION = (
    "Alcance Define: objetivos, alcance, requisitos, reglas de negocio, NFR, criterios de aceptacion y owners funcionales. "
    "Difiere framework, base de datos, credenciales, despliegue, endpoints finales y stack al ACP salvo que sean restricciones ya declaradas."
)

DESIGN_SCOPE_INSTRUCTION = (
    "Alcance Design: patrones agentivos, roles, autonomia, handoffs, guardrails, razonamiento y comportamiento. "
    "No conviertas decisiones de stack, base de datos o despliegue en preguntas de diseno; difierelas al ACP."
)

TOOLS_SCOPE_INSTRUCTION = (
    "Alcance Tools: capacidades externas minimas, categorias de herramienta, contratos funcionales, redundancia e incompatibilidades. "
    "Puedes preguntar solo por decisiones necesarias para seleccionar o descartar herramientas; credenciales, endpoints finales y configuraciones quedan para ACP."
)

MEMORY_SCOPE_INSTRUCTION = (
    "Alcance Memory: memoria corta, memoria larga, RAG, fuentes de conocimiento, recuperacion, retencion conceptual y gobierno. "
    "Difiere motores concretos, secrets, dimensiones de embeddings y decisiones de infraestructura al ACP si no estan aprobadas."
)

VALIDATE_SCOPE_INSTRUCTION = (
    "Alcance Validate: escenarios, criterios, simulacion, gaps de comportamiento y remediaciones. "
    "No pidas configuracion de despliegue o stack; esos datos pertenecen al ACP."
)

ESTIMATE_SCOPE_INSTRUCTION = (
    "Alcance Estimate: supuestos de costo, bandas, riesgo residual, esfuerzo y ROI. "
    "Las preguntas deben cerrar supuestos de estimacion; valores privados de contrato o entorno se documentan como diferidos al ACP."
)


CAPABILITY_SPECS: dict[BuilderCapability, BuilderCapabilitySpec] = {
    BuilderCapability.normalize_discovery: BuilderCapabilitySpec(
        capability=BuilderCapability.normalize_discovery,
        task_kind="discovery_normalization",
        prompt_version="normalize_discovery.v1",
        source_key="discovery_capture",
        source_title="Discovery capture",
        source_summary="Captura cruda de discovery para normalizacion estructurada.",
        system_instruction=(
            "Convierte la captura en un discovery estructurado para un builder Lean de agentes. "
            "Usa solo hechos presentes en la entrada y declara unknown cuando falte evidencia."
        ),
        task_instruction=(
            "Normaliza `discovery_capture` en un discovery estructurado. "
            "Usa case_type permitido y autonomy_level low, medium o high."
        ),
        output_model=DiscoveryArtifact,
        preferred_model="fast",
        llm_required=True,
        critic_required=False,
        timeout_ms=60000,
        max_retries=1,
        fallback_policy="deterministic_fallback_visible",
    ),
    BuilderCapability.analyze_discovery: BuilderCapabilitySpec(
        capability=BuilderCapability.analyze_discovery,
        task_kind="discovery_analysis",
        prompt_version="analyze_discovery.v1",
        source_key="discovery_analysis_input",
        source_title="Discovery analysis input",
        source_summary="Discovery parcial o completo para analisis profundo, ambiguedades y preguntas.",
        system_instruction=(
            "Analiza discovery con rigor Lean y progresion por etapas. Separa hechos, inferencias, supuestos "
            "y riesgos. Antes de preguntar, intenta inferir desde el contexto disponible y no inventes datos "
            "de negocio faltantes."
        ),
        task_instruction=(
            "Analiza `discovery_analysis_input` solo para la etapa Discover. Devuelve preguntas abiertas "
            "unicamente si son indispensables para entender el problema, usuario, proceso actual, resultado "
            "esperado o restricciones de negocio inmediatas, y no pueden inferirse con confianza razonable. "
            "No preguntes por herramientas, integraciones, ERP, ticketing, APIs, RAG, memoria, frameworks, "
            "infraestructura, despliegue, credenciales, contratos ni configuraciones; esos temas se infieren "
            "como senales o se difieren a Tools, Memory o ACP segun corresponda. "
            f"{DISCOVERY_SCOPE_INSTRUCTION} {GUIDED_QUESTION_INSTRUCTION}"
        ),
        output_model=DiscoveryAnalysisOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=90000,
        max_retries=0,
        fallback_policy="needs_review_on_provider_or_schema_failure",
    ),
    BuilderCapability.build_canvas: BuilderCapabilitySpec(
        capability=BuilderCapability.build_canvas,
        task_kind="canvas_generation",
        prompt_version="build_canvas.v1",
        source_key="normalized_discovery",
        source_title="Normalized discovery",
        source_summary="Discovery estructurado aprobado para construir el canvas Lean.",
        system_instruction=(
            "Genera un canvas Lean corto y concreto para un agente usando solo discovery aprobado."
        ),
        task_instruction=(
            "Construye un canvas usando `normalized_discovery` sin inventar alcance ni aprobaciones."
        ),
        output_model=CanvasArtifact,
        preferred_model="fast",
        llm_required=True,
        critic_required=False,
        timeout_ms=60000,
        max_retries=1,
        fallback_policy="deterministic_fallback_visible",
    ),
    BuilderCapability.define_requirements: BuilderCapabilitySpec(
        capability=BuilderCapability.define_requirements,
        task_kind="requirements_definition",
        prompt_version="define_requirements.v1",
        source_key="requirements_definition_input",
        source_title="Requirements definition input",
        source_summary="Discovery y canvas aprobados para consolidar requisitos funcionales y no funcionales.",
        system_instruction=(
            "Consolida requerimientos funcionales, no funcionales, reglas y restricciones usando solo contexto aprobado."
        ),
        task_instruction=(
            "A partir de `requirements_definition_input`, produce un set estructurado de requisitos, criterios de "
            "aceptacion y dependencias sin duplicar ni inventar evidencia. "
            f"{DEFINE_SCOPE_INSTRUCTION} {GUIDED_QUESTION_INSTRUCTION}"
        ),
        output_model=RequirementsDefinitionOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=240000,
        max_retries=0,
        fallback_policy="needs_review_on_provider_or_schema_failure",
    ),
    BuilderCapability.synthesize_blueprint_narrative: BuilderCapabilitySpec(
        capability=BuilderCapability.synthesize_blueprint_narrative,
        task_kind="blueprint_narrative",
        prompt_version="synthesize_blueprint_narrative.v1",
        source_key="narrative_blueprint_bundle",
        source_title="Blueprint narrative bundle",
        source_summary="Bundle compacto aprobado para sintetizar narrativa tecnica del blueprint.",
        system_instruction=(
            "Redacta la narrativa tecnica del blueprint sin alterar arquitectura, memoria, tools ni guardrails."
        ),
        task_instruction=(
            "Sintetiza la narrativa usando solo el bundle compacto aprobado y resalta tradeoffs reales."
        ),
        output_model=BlueprintNarrativeOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=90000,
        max_retries=0,
        fallback_policy="preserve_base_narrative_on_failure",
    ),
    BuilderCapability.propose_agent_design: BuilderCapabilitySpec(
        capability=BuilderCapability.propose_agent_design,
        task_kind="agent_design_proposal",
        prompt_version="propose_agent_design.v1",
        source_key="agent_design_input",
        source_title="Agent design input",
        source_summary="Discovery, canvas y blueprint base para proponer arquitectura y comportamiento del agente.",
        system_instruction=(
            "Propone hasta tres alternativas reales de diseno del agente usando solo contexto aprobado, "
            "sin definir Tools ni Memory de forma canonica, pero declarando sus implicaciones arquitectonicas."
        ),
        task_instruction=(
            "A partir de `agent_design_input`, compara alternativas realmente distintas, justifica la recomendada, "
            "declara arquetipo de agente, familia de patron, ajuste con el negocio, hipotesis de valor, modelo "
            "operativo, por que no basta una opcion mas simple, por que no conviene una mas compleja, roles, "
            "handoffs, riesgos, approval points, tradeoffs, metricas de negocio, implicaciones para Tools, "
            "implicaciones para Memory y cobertura contra requisitos. "
            f"{DESIGN_SCOPE_INSTRUCTION} {GUIDED_QUESTION_INSTRUCTION}"
        ),
        output_model=AgentDesignProposalOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=True,
        timeout_ms=300000,
        max_retries=0,
        fallback_policy="blocked_until_critique_or_user_review",
    ),
    BuilderCapability.critique_agent_design: BuilderCapabilitySpec(
        capability=BuilderCapability.critique_agent_design,
        task_kind="agent_design_critique",
        prompt_version="critique_agent_design.v1",
        source_key="agent_design_critique_input",
        source_title="Agent design critique input",
        source_summary="Propuesta de diseno del agente y contexto aprobado para revision critica.",
        system_instruction=(
            "Critica una propuesta de diseno buscando sobrearquitectura, gaps, riesgos y evidencia faltante."
        ),
        task_instruction=(
            "Evalua `agent_design_critique_input` y devuelve findings priorizados sobre redundancia, cobertura, "
            "handoffs ambiguos, riesgos de loops, costo desproporcionado y supuestos prematuros."
        ),
        output_model=DesignCritiqueOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=240000,
        max_retries=0,
        fallback_policy="needs_review_if_critique_missing",
    ),
    BuilderCapability.recommend_minimal_tools: BuilderCapabilitySpec(
        capability=BuilderCapability.recommend_minimal_tools,
        task_kind="tool_recommendation_minimal",
        prompt_version="recommend_minimal_tools.v1",
        source_key="tool_recommendation_case",
        source_title="Tool recommendation case",
        source_summary="Digest aprobado desde discovery, define y design para seleccionar el set minimo de tools.",
        system_instruction=(
            build_tool_recommendation_system_instruction()
        ),
        task_instruction=(
            build_tool_recommendation_registry_task_instruction(
                tools_scope_instruction=TOOLS_SCOPE_INSTRUCTION,
                guided_question_instruction=GUIDED_QUESTION_INSTRUCTION,
            )
        ),
        output_model=ToolRecommendationLLMOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=120000,
        max_retries=0,
        fallback_policy="preflight_only_if_provider_fails",
    ),
    BuilderCapability.recommend_memory_architecture: BuilderCapabilitySpec(
        capability=BuilderCapability.recommend_memory_architecture,
        task_kind="memory_architecture_recommendation",
        prompt_version="recommend_memory_architecture.v1",
        source_key="memory_architecture_input",
        source_title="Memory architecture input",
        source_summary="Blueprint y tools aprobadas para proponer la estrategia de memoria del agente objetivo.",
        system_instruction=(
            "Disena memoria corta, larga y retrieval del agente objetivo sin confundirla con la memoria del builder."
        ),
        task_instruction=(
            "A partir de `memory_architecture_input`, recomienda estrategias de memoria, storage layers, write policy, "
            "retrieval strategy, pruning y preguntas abiertas. Si la arquitectura de memoria requiere una capacidad "
            "de herramienta faltante, declarala en `tool_dependency_requests` usando solo keys canonicas del catalogo "
            "cuando apliquen: knowledge_retrieval, document_ingestion, scheduler, approval_gate, human_handoff u "
            "outbound_notification. No inventes tool keys ni nombres de integraciones; si requiere credenciales, "
            "side effects sensibles o decision de negocio, expresa la necesidad como pregunta/gap y no como tool ejecutable. "
            f"{MEMORY_SCOPE_INSTRUCTION} {GUIDED_QUESTION_INSTRUCTION}"
        ),
        output_model=MemoryArchitectureRecommendationOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=True,
        timeout_ms=120000,
        max_retries=0,
        fallback_policy="blocked_until_review_if_missing",
    ),
    BuilderCapability.critique_memory_architecture: BuilderCapabilitySpec(
        capability=BuilderCapability.critique_memory_architecture,
        task_kind="memory_architecture_critique",
        prompt_version="critique_memory_architecture.v1",
        source_key="memory_architecture_critique_input",
        source_title="Memory architecture critique input",
        source_summary="Propuesta de memoria del agente objetivo para revision critica y cierre de contradicciones.",
        system_instruction=(
            "Critica arquitectura de memoria buscando saturacion de contexto, gaps de retrieval y riesgos de gobierno."
        ),
        task_instruction=(
            "Evalua `memory_architecture_critique_input` y devuelve findings, contradicciones y evidencia faltante."
        ),
        output_model=MemoryArchitectureCritiqueOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=90000,
        max_retries=0,
        fallback_policy="needs_review_if_critique_missing",
    ),
    BuilderCapability.generate_validation_scenarios: BuilderCapabilitySpec(
        capability=BuilderCapability.generate_validation_scenarios,
        task_kind="validation_scenario_generation",
        prompt_version="generate_validation_scenarios.v1",
        source_key="validation_scenario_generation_input",
        source_title="Validation scenario generation input",
        source_summary="Blueprint aprobado y foco de validacion para crear escenarios representativos.",
        system_instruction=(
            "Genera escenarios de validacion representativos, trazables y utiles para revisar el comportamiento esperado."
        ),
        task_instruction=(
            "Usa `validation_scenario_generation_input` para generar escenarios, cobertura y gaps aun no validados. "
            f"{VALIDATE_SCOPE_INSTRUCTION} {GUIDED_QUESTION_INSTRUCTION}"
        ),
        output_model=ValidationScenarioGenerationOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=90000,
        max_retries=0,
        fallback_policy="needs_review_if_scenarios_missing",
    ),
    BuilderCapability.simulate_validation_scenario: BuilderCapabilitySpec(
        capability=BuilderCapability.simulate_validation_scenario,
        task_kind="validation_scenario_simulation",
        prompt_version="simulate_validation_scenario.v1",
        source_key="validation_simulation_input",
        source_title="Validation simulation input",
        source_summary="Blueprint y escenario aprobado para simular transcript, decisiones e issues.",
        system_instruction=(
            "Simula de forma explicable la ejecucion esperada de un escenario de validacion."
        ),
        task_instruction=(
            "A partir de `validation_simulation_input`, genera transcript, decisiones, interacciones y issues observados."
        ),
        output_model=ValidationSimulationOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=120000,
        max_retries=0,
        fallback_policy="needs_review_if_simulation_missing",
    ),
    BuilderCapability.judge_validation_run: BuilderCapabilitySpec(
        capability=BuilderCapability.judge_validation_run,
        task_kind="validation_run_judgment",
        prompt_version="judge_validation_run.v1",
        source_key="validation_run_judgment_input",
        source_title="Validation run judgment input",
        source_summary="Simulacion o corrida de validacion para emitir juicio estructurado.",
        system_instruction=(
            "Juzga una corrida de validacion con criterios claros, findings accionables y score visible."
        ),
        task_instruction=(
            "Evalua `validation_run_judgment_input` y devuelve juicio, summary, findings y score."
        ),
        output_model=ValidationRunJudgmentOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=90000,
        max_retries=0,
        fallback_policy="needs_review_if_judgment_missing",
    ),
    BuilderCapability.analyze_estimation_risks: BuilderCapabilitySpec(
        capability=BuilderCapability.analyze_estimation_risks,
        task_kind="estimation_risk_analysis",
        prompt_version="analyze_estimation_risks.v1",
        source_key="estimation_risk_analysis_input",
        source_title="Estimation risk analysis input",
        source_summary="Blueprint, reporte deterministico y referencias calibradas para analizar riesgos, benchmarks y sensibilidad sin alterar los montos base.",
        system_instruction=(
            "Analiza una estimacion usando solo artefactos aprobados, memoria recuperada y benchmarks autorizados. "
            "No inventes tarifas, costos finales ni pricing."
        ),
        task_instruction=(
            "A partir de `estimation_risk_analysis_input`, devuelve drivers de complejidad, risk register, benchmark refs, "
            "factores de incertidumbre, escenarios optimistic/base/conservative como multiplicadores acotados, "
            "oportunidades de ahorro, preguntas y una propuesta acotada de ajuste de confianza. "
            f"{ESTIMATE_SCOPE_INSTRUCTION} {GUIDED_QUESTION_INSTRUCTION}"
        ),
        output_model=EstimationRiskAnalysisOutput,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=90000,
        max_retries=0,
        fallback_policy="keep_deterministic_estimate_and_mark_needs_review",
    ),
    BuilderCapability.generate_diagram_model: BuilderCapabilitySpec(
        capability=BuilderCapability.generate_diagram_model,
        task_kind="diagram_model_generation",
        prompt_version="diagram-prompts.v1.0.0",
        source_key="diagram_generation_input",
        source_title="Diagram generation input",
        source_summary="PromptSpec gobernado y contexto aprobado para producir un DiagramModel canónico.",
        system_instruction=(
            "Eres el motor de modelado de diagramas de una plataforma enterprise. Devuelve exclusivamente un "
            "DiagramModel v1 válido y trazable. Usa solo el contexto aprobado; no inventes componentes, actores, "
            "tecnologías, cardinalidades ni decisiones. El campo notation debe respetar la notación solicitada. "
            "Los ids deben iniciar con letra y usar solo letras, números, punto, dos puntos, guion o guion bajo. "
            "Toda arista debe referenciar nodos existentes. Si la notación es BPMN, infiere pools y lanes desde "
            "participantes, áreas, roles o sistemas aprobados, decláralos en `pools` y asigna cada nodo con "
            "`metadata.pool_id` y `metadata.lane_id`. Nunca incluyas secretos o datos personales."
        ),
        task_instruction=(
            "Construye el modelo solicitado en `diagram_generation_input`. Aplica objetivo, reglas semánticas y "
            "exclusiones del PromptSpec. Mantén el nivel de detalle solicitado, agrega source_refs y registra como "
            "assumptions solo inferencias indispensables claramente identificadas. Si `context_brief` está presente, "
            "úsalo como digest ejecutivo de la evidencia aprobada. Si `resolved_inputs` está presente, trátalo como "
            "el mapeo canónico entre `required_inputs` y la evidencia aprobada. No concluyas que falta contexto si "
            "`resolved_inputs` ya contiene evidencia suficiente para modelar una vista mínima y trazable. No "
            "devuelvas Mermaid, SVG ni texto explicativo fuera del contrato JSON. Para BPMN usa sequence_flow "
            "dentro del mismo pool y message_flow entre pools; no sustituyas BPMN por un grafo dirigido genérico."
        ),
        output_model=StructuredDiagramModel,
        preferred_model="reasoning",
        llm_required=True,
        critic_required=False,
        timeout_ms=120000,
        max_retries=1,
        fallback_policy="fail_visible_without_synthetic_diagram",
    ),
}


def get_builder_capability_spec(capability: BuilderCapability) -> BuilderCapabilitySpec:
    return CAPABILITY_SPECS[capability]
