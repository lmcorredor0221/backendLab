from __future__ import annotations

import hashlib
import json
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

import app.services.skill_runtime as skill_runtime
from app.models import (
    ArtifactStatus,
    BlueprintArtifact,
    CanvasArtifact,
    DiscoveryArtifact,
    EvidenceItem,
    EvidenceSource,
    ReviewState,
    SessionSnapshot,
    SessionStage,
    SimulationEdge,
    SimulationEvent,
    SimulationJudgement,
    SimulationJudgementFinding,
    SimulationNode,
    SimulationRunRecord,
    SimulationScenario,
    SimulationSpecificationArtifact,
    ValidationSimulationRunRequest,
)
from app.services.llm_runtime.builder_contracts import (
    CritiqueFinding,
    RequirementsDefinitionOutput,
    ValidationRunJudgmentInput,
    ValidationRunJudgmentOutput,
    ValidationScenarioGenerationInput,
    ValidationScenarioGenerationOutput,
    ValidationScenarioItem,
    ValidationScenarioSimulationInput,
    ValidationSimulationOutput,
)
from app.services.llm_runtime.stage_context_types import StageContextBundle


def _stable_signature(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _clamp_priority(value: str) -> str:
    return value if value in {"high", "medium", "low"} else "medium"


def _scenario_item_from_scenario(scenario: SimulationScenario) -> ValidationScenarioItem:
    steps = scenario.state_transitions or scenario.decision_criteria or scenario.preconditions
    expected_outcomes = scenario.success_criteria or ([scenario.expected_outcome] if scenario.expected_outcome else [])
    return ValidationScenarioItem(
        scenario_key=scenario.scenario_key,
        title=scenario.title,
        objective=scenario.objective,
        steps=list(steps),
        expected_outcomes=list(expected_outcomes),
        failure_signals=list(scenario.blocking_failures),
        priority=_clamp_priority(scenario.priority),
    )


def _build_trace(
    *,
    skill_key: str,
    label: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    status: ArtifactStatus,
    result_summary: str,
    warnings: list[str],
    evidence: list[EvidenceItem],
    llm_result,
    duration_ms: int,
):
    return skill_runtime.SkillExecutionTrace(
        skill_key=skill_key,
        label=label,
        stage=SessionStage.post_validation,
        status=status,
        duration_ms=duration_ms,
        warnings=list(warnings),
        evidence=list(evidence),
        llm_trace=skill_runtime._build_llm_trace(llm_result),
        result_summary=result_summary,
        input_kind="ValidationSimulationInput",
        input_payload=input_payload,
        output_kind="ValidationSimulationOutput",
        output_payload=output_payload,
    )


def _common_nodes(
    *,
    tool_label: str,
    decision_label: str,
    human_label: str,
    end_label: str,
) -> tuple[list[SimulationNode], list[SimulationEdge]]:
    nodes = [
        SimulationNode(node_key="human_start", label="Usuario", node_type="human", description="Dispara el escenario.", x=40, y=140),
        SimulationNode(node_key="agent_plan", label="Agente", node_type="agent", description="Orquesta el flujo y decide siguiente paso.", x=240, y=140),
        SimulationNode(node_key="memory_context", label="Memoria", node_type="memory", description="Recupera contexto aprobado y artefactos.", x=450, y=60),
        SimulationNode(node_key="tool_action", label=tool_label, node_type="tool", description="Invoca la capacidad externa requerida.", x=450, y=220),
        SimulationNode(node_key="decision_policy", label=decision_label, node_type="decision", description="Evalua evidencia, riesgos y gates.", x=690, y=140),
        SimulationNode(node_key="human_gate", label=human_label, node_type="human", description="Intervencion o aprobacion requerida.", x=930, y=60),
        SimulationNode(node_key="end_result", label=end_label, node_type="end", description="Cierre esperado del escenario.", x=930, y=220),
    ]
    edges = [
        SimulationEdge(edge_key="edge-start-plan", from_node_key="human_start", to_node_key="agent_plan", label="input"),
        SimulationEdge(edge_key="edge-plan-memory", from_node_key="agent_plan", to_node_key="memory_context", label="recupera contexto"),
        SimulationEdge(edge_key="edge-plan-tool", from_node_key="agent_plan", to_node_key="tool_action", label="invoca capability"),
        SimulationEdge(edge_key="edge-memory-decision", from_node_key="memory_context", to_node_key="decision_policy", label="evidencia"),
        SimulationEdge(edge_key="edge-tool-decision", from_node_key="tool_action", to_node_key="decision_policy", label="respuesta externa"),
        SimulationEdge(edge_key="edge-decision-human", from_node_key="decision_policy", to_node_key="human_gate", label="approval/escalation"),
        SimulationEdge(edge_key="edge-decision-end", from_node_key="decision_policy", to_node_key="end_result", label="outcome"),
        SimulationEdge(edge_key="edge-human-end", from_node_key="human_gate", to_node_key="end_result", label="cierre aprobado"),
    ]
    return nodes, edges


def _build_baseline_scenarios(
    *,
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact,
    definition_artifact: RequirementsDefinitionOutput | None,
) -> list[SimulationScenario]:
    primary_user = (
        (canvas.agent_profile.primary_user if canvas is not None else "")
        or (discovery.current_user if discovery is not None else "")
        or "Usuario operador"
    )
    primary_goal = (
        (canvas.user_goal if canvas is not None else "")
        or (discovery.desired_outcome if discovery is not None else "")
        or "Completar el objetivo del agente con evidencia"
    )
    success_metric = canvas.success_metric if canvas is not None else "Cumplir el flujo esperado sin romper governance"
    first_tool = next((item.name for item in blueprint.tools if item.name), "Tool principal")
    second_tool = next((item.name for item in blueprint.tools[1:] if item.name), first_tool)
    approval_text = (
        (definition_artifact.acceptance_criteria[0].criterion if definition_artifact and definition_artifact.acceptance_criteria else "")
        or "Solicitar aprobacion humana antes de cualquier decision sensible o irreversible."
    )
    memory_layers = blueprint.memory_profile.storage_layers or ["session_memory"]
    memory_label = ", ".join(memory_layers[:2])

    happy_nodes, happy_edges = _common_nodes(
        tool_label=first_tool,
        decision_label="Decision guiada por evidencia",
        human_label="Approval gate",
        end_label="Resultado aprobado",
    )
    tool_nodes, tool_edges = _common_nodes(
        tool_label=second_tool,
        decision_label="Compensacion y recovery",
        human_label="Escalacion operativa",
        end_label="Continuidad controlada",
    )
    no_evidence_nodes, no_evidence_edges = _common_nodes(
        tool_label="Knowledge retrieval",
        decision_label="No-evidence policy",
        human_label="Escalacion humana",
        end_label="Salida segura",
    )

    return [
        SimulationScenario(
            scenario_key="happy_path_high_value",
            title="Happy path de mayor valor",
            actor=primary_user,
            objective=f"Completar el flujo principal del agente para {primary_goal.lower()}",
            priority="high",
            initial_input="El usuario solicita una ejecucion completa del caso principal con datos suficientes.",
            preconditions=[
                "Discovery, Define, Design, Tools y Memory se encuentran aprobados.",
                f"El blueprint vigente declara {len(blueprint.tools)} tools aprobadas y memoria {memory_label}.",
            ],
            state_transitions=[
                "El agente recupera memoria de trabajo y reglas aprobadas.",
                f"El agente invoca {first_tool} y valida la respuesta.",
                "El agente decide si puede cerrar o si requiere gate humano.",
            ],
            decision_criteria=[
                success_metric,
                approval_text,
                "No continuar con side effects si la evidencia es incompleta.",
            ],
            tools_invoked=[first_tool],
            simulated_tool_responses=[f"{first_tool} responde sin errores y con payload suficiente para continuar."],
            memory_reads=["Resumen aprobado del problema", "Reglas de negocio y constraints vigentes"],
            memory_writes=["Decision trace final", "Resumen episodico del escenario exitoso"],
            approval_gates=["Confirmacion de salida o side effect sensible"],
            expected_outcome="El agente completa la tarea principal con evidencia, trazabilidad y salida aprobable.",
            success_criteria=[
                "Se visualizan lecturas y escrituras de memoria.",
                "La decision principal deja visible su criterio de aprobacion.",
                "No se dispara ninguna accion sensible sin gate humano.",
            ],
            blocking_failures=[
                "Cerrar el caso sin evidencia suficiente.",
                "Omitir el approval gate cuando existe side effect o ambiguedad relevante.",
            ],
            suggested_injections=["tool_timeout", "no_evidence"],
            source_refs=["session.discovery", "session.canvas", "session.blueprint", "session.journey_latest_artifacts.memory"],
            nodes=happy_nodes,
            edges=happy_edges,
        ),
        SimulationScenario(
            scenario_key="tool_timeout_compensation",
            title="Timeout de herramienta y compensacion",
            actor=primary_user,
            objective="Verificar que el agente recupere continuidad operativa ante un fallo temporal de herramienta.",
            priority="high",
            initial_input="Durante la ejecucion, una tool critica responde fuera del SLA acordado.",
            preconditions=[
                f"La herramienta {second_tool} esta disponible en el blueprint aprobado.",
                "El agente cuenta con retries, compensacion o escalation documentados.",
            ],
            state_transitions=[
                f"El agente llama {second_tool}.",
                "La llamada expira y se activa la ruta de compensacion.",
                "El agente registra el incidente, evita efectos colaterales y escala si corresponde.",
            ],
            decision_criteria=[
                "No perder trazabilidad del error ni del fallback ejecutado.",
                "No repetir side effects sin estrategia de idempotencia.",
                "Escalar al humano cuando la herramienta comprometida bloquea el resultado final.",
            ],
            tools_invoked=[second_tool],
            simulated_tool_responses=[
                f"{second_tool} responde timeout.",
                "El fallback local provee estado parcial pero no confirmacion final.",
            ],
            memory_reads=["Politica de retries", "Contrato de typed errors de la herramienta"],
            memory_writes=["Registro del timeout", "Decision de compensacion aplicada"],
            approval_gates=["Aprobacion para continuar con salida parcial o escalar"],
            expected_outcome="El agente maneja el timeout sin perder control, evidencia ni seguridad.",
            success_criteria=[
                "Se registra el timeout como evento visible.",
                "Existe una decision explicita de compensacion o escalation.",
                "El flujo termina en estado controlado aunque no haya exito funcional completo.",
            ],
            blocking_failures=[
                "Reintentar sin control hasta degradar el flujo.",
                "Ocultar el timeout al usuario o a la bitacora de validacion.",
            ],
            suggested_injections=["tool_timeout", "approval_rejected"],
            source_refs=["session.blueprint.tools", "session.journey_latest_artifacts.design", "session.journey_latest_artifacts.tools"],
            nodes=tool_nodes,
            edges=tool_edges,
        ),
        SimulationScenario(
            scenario_key="no_evidence_human_escalation",
            title="Ambiguedad, no-evidence y escalacion humana",
            actor=primary_user,
            objective="Comprobar que el agente no invente respuestas y active la ruta humana cuando la evidencia es insuficiente.",
            priority="high",
            initial_input="El usuario pide una respuesta sensible pero la evidencia recuperada es ambigua o inexistente.",
            preconditions=[
                "La politica grounding/no-evidence fue aprobada en Memory.",
                "Existen reglas de negocio que exigen justificacion y trazabilidad.",
            ],
            state_transitions=[
                "El agente consulta memoria y retrieval de conocimiento.",
                "No encuentra evidencia concluyente.",
                "El agente explica el gap, solicita aclaracion o escala a humano responsable.",
            ],
            decision_criteria=[
                "No afirmar informacion no soportada por evidencia recuperable.",
                "Escalar cuando la ambiguedad afecte decisiones de negocio o seguridad.",
                "Persistir la razon del no-evidence para futuras iteraciones.",
            ],
            tools_invoked=["knowledge_retrieval"],
            simulated_tool_responses=["La recuperacion devuelve resultados ambiguos o insuficientes."],
            memory_reads=["Grounding policy aprobada", "Artefactos aprobados de etapas previas"],
            memory_writes=["Registro del no-evidence", "Necesidad de aclaracion para siguiente iteracion"],
            approval_gates=["Escalacion humana obligatoria antes de responder en falso"],
            expected_outcome="El agente se detiene de forma segura, comunica el gap y deriva al humano adecuado.",
            success_criteria=[
                "Se hace visible el evento no-evidence.",
                "La salida no contiene afirmaciones no respaldadas.",
                "Se muestra la escalacion humana como parte del flujo.",
            ],
            blocking_failures=[
                "Responder con informacion inventada.",
                "Omitir la politica de grounding o la escalacion requerida.",
            ],
            suggested_injections=["no_evidence"],
            source_refs=["session.blueprint.memory_profile", "session.blueprint.knowledge_profile", "knowledge.long_term_memory"],
            nodes=no_evidence_nodes,
            edges=no_evidence_edges,
        ),
    ]


def _merge_llm_scenarios(
    *,
    baseline: list[SimulationScenario],
    llm_output: ValidationScenarioGenerationOutput | None,
) -> list[SimulationScenario]:
    if llm_output is None or not llm_output.scenarios:
        return baseline

    by_key = {item.scenario_key: item for item in llm_output.scenarios if item.scenario_key}
    merged: list[SimulationScenario] = []
    for index, scenario in enumerate(baseline):
        llm_item = by_key.get(scenario.scenario_key) or (llm_output.scenarios[index] if index < len(llm_output.scenarios) else None)
        if llm_item is None:
            merged.append(scenario)
            continue
        merged.append(
            scenario.model_copy(
                update={
                    "title": llm_item.title or scenario.title,
                    "objective": llm_item.objective or scenario.objective,
                    "priority": _clamp_priority(llm_item.priority),
                    "state_transitions": list(llm_item.steps) or scenario.state_transitions,
                    "success_criteria": list(llm_item.expected_outcomes) or scenario.success_criteria,
                    "blocking_failures": list(llm_item.failure_signals) or scenario.blocking_failures,
                }
            )
        )
    return merged


def build_validation_simulation_specification(
    *,
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact,
    definition_artifact: RequirementsDefinitionOutput | None,
    session_snapshot: SessionSnapshot | None,
    blueprint_version_number: int | None,
    source_stage_versions: dict[str, int | None],
    instructions: str = "",
    runtime_settings=None,
    stage_context: StageContextBundle | None = None,
) -> tuple[SimulationSpecificationArtifact, list]:
    started = perf_counter()
    baseline = _build_baseline_scenarios(
        discovery=discovery,
        canvas=canvas,
        blueprint=blueprint,
        definition_artifact=definition_artifact,
    )
    warnings: list[str] = []
    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="Validate compilo escenarios base desde artefactos aprobados, contratos de herramientas y estrategia de memoria.",
        )
    ]
    llm_result = None
    llm_output: ValidationScenarioGenerationOutput | None = None

    llm_service = skill_runtime._builder_service_for_stage("validate", runtime_settings)
    llm_input = ValidationScenarioGenerationInput(
        blueprint=blueprint,
        discovery=discovery,
        canvas=canvas,
        focus_areas=list(dict.fromkeys(["happy_path", "tool_failure", "security_or_no_evidence", *([instructions] if instructions else [])])),
        source_refs=[
            "session.discovery",
            "session.canvas",
            "session.blueprint",
            "session.journey_latest_artifacts.define",
            "session.journey_latest_artifacts.design",
            "session.journey_latest_artifacts.tools",
            "session.journey_latest_artifacts.memory",
            "session.short_term_memory",
        ],
    )
    llm_result = llm_service.generate_validation_scenarios(llm_input, context_bundle=stage_context)
    skill_runtime._append_warning(warnings, llm_result.warning if llm_result is not None else None)
    if llm_result is not None and isinstance(llm_result.artifact, ValidationScenarioGenerationOutput):
        llm_output = ValidationScenarioGenerationOutput.model_validate(llm_result.artifact.model_dump(mode="json"))
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.llm_inference,
                detail=skill_runtime._llm_evidence_detail(llm_result, "propuso escenarios representativos de validacion"),
            )
        )
    elif llm_result is not None and llm_result.warning is None:
        skill_runtime._append_warning(
            warnings,
            "No hubo salida estructurada valida para los escenarios de Validate; se mantiene el baseline deterministico.",
        )

    scenarios = _merge_llm_scenarios(baseline=baseline, llm_output=llm_output)
    missing_information: list[str] = []
    if definition_artifact is None:
        missing_information.append("Define aprobado no pudo resolverse completamente para reforzar criterios de simulacion.")
    if session_snapshot is None:
        missing_information.append("No se incluyo snapshot de sesion para enriquecer episodios previos y governance.")

    artifact = SimulationSpecificationArtifact(
        summary=(
            llm_output.summary
            if llm_output is not None and llm_output.summary
            else "Escenarios de validacion compilados para happy path, timeout de herramientas y no-evidence."
        ),
        review_state=ReviewState.complete if not missing_information else ReviewState.partial,
        confidence=0.82 if llm_output is not None else 0.68,
        source_blueprint_version=blueprint_version_number,
        source_stage_versions=dict(source_stage_versions),
        coverage_gaps=list(llm_output.coverage_gaps) if llm_output is not None else [],
        missing_information=missing_information,
        warnings=warnings,
        scenarios=scenarios,
    )
    duration_ms = int((perf_counter() - started) * 1000)
    trace = _build_trace(
        skill_key="validation_scenario_generation_skill",
        label="Validation scenario generation skill",
        input_payload=llm_input.model_dump(mode="json"),
        output_payload=artifact.model_dump(mode="json"),
        status=ArtifactStatus.needs_review,
        result_summary=artifact.summary,
        warnings=warnings,
        evidence=evidence,
        llm_result=llm_result,
        duration_ms=duration_ms,
    )
    return artifact, [trace]


def _build_fault_events(injected_conditions: list[str]) -> list[SimulationEvent]:
    events: list[SimulationEvent] = []
    for index, condition in enumerate(injected_conditions, start=1):
        title = "Fallo inyectado"
        detail = {
            "tool_timeout": "Se fuerza timeout de herramienta para probar recovery y compensacion.",
            "no_evidence": "Se fuerza ausencia de evidencia para evaluar grounding y escalacion.",
            "approval_rejected": "Se simula rechazo humano para validar cierre seguro.",
        }.get(condition, f"Se inyecta la condicion {condition}.")
        events.append(
            SimulationEvent(
                event_key=f"fault-{index}",
                event_index=0,
                event_type="fault_injected",
                tone="warning",
                title=title,
                detail=detail,
                actor="simulator",
                node_key="agent_plan",
                payload={"condition": condition},
            )
        )
    return events


def _build_deterministic_simulation_events(
    scenario: SimulationScenario,
    *,
    initial_input: str,
    injected_conditions: list[str],
    llm_output: ValidationSimulationOutput | None,
) -> list[SimulationEvent]:
    events: list[SimulationEvent] = [
        SimulationEvent(
            event_key="event-1",
            event_index=1,
            event_type="start",
            tone="info",
            title="Inicio del escenario",
            detail=scenario.title,
            actor=scenario.actor,
            node_key="human_start",
        ),
        SimulationEvent(
            event_key="event-2",
            event_index=2,
            event_type="input",
            tone="info",
            title="Input inicial",
            detail=initial_input or scenario.initial_input,
            actor=scenario.actor,
            node_key="human_start",
        ),
    ]

    fault_events = _build_fault_events(injected_conditions)
    events.extend(fault_events)
    events.append(
        SimulationEvent(
            event_key="event-3",
            event_index=3 + len(fault_events),
            event_type="memory_read",
            tone="info",
            title="Lectura de memoria",
            detail=", ".join(scenario.memory_reads) or "Se recupera contexto aprobado.",
            actor="agent",
            node_key="memory_context",
        )
    )
    events.append(
        SimulationEvent(
            event_key="event-4",
            event_index=4 + len(fault_events),
            event_type="agent_response",
            tone="info",
            title="Plan del agente",
            detail=(
                llm_output.simulated_transcript[0]
                if llm_output is not None and llm_output.simulated_transcript
                else "El agente resume el objetivo, recupera contexto y prepara la siguiente accion."
            ),
            actor="agent",
            node_key="agent_plan",
        )
    )
    if scenario.tools_invoked:
        tool_name = scenario.tools_invoked[0]
        events.append(
            SimulationEvent(
                event_key="event-5",
                event_index=5 + len(fault_events),
                event_type="tool_call",
                tone="info",
                title="Invocacion de herramienta",
                detail=f"El agente invoca {tool_name}.",
                actor="agent",
                node_key="tool_action",
                payload={"tool_name": tool_name},
            )
        )
        tool_timeout = "tool_timeout" in injected_conditions or "tool_timeout" in scenario.scenario_key
        no_evidence = "no_evidence" in injected_conditions or "no_evidence" in scenario.scenario_key
        tone = "warning" if tool_timeout or no_evidence else "success"
        detail = (
            f"{tool_name} responde timeout y activa compensacion."
            if tool_timeout
            else (
                f"{tool_name} no aporta evidencia concluyente."
                if no_evidence
                else (
                    llm_output.tool_interactions[0]
                    if llm_output is not None and llm_output.tool_interactions
                    else scenario.simulated_tool_responses[0] if scenario.simulated_tool_responses else f"{tool_name} responde correctamente."
                )
            )
        )
        events.append(
            SimulationEvent(
                event_key="event-6",
                event_index=6 + len(fault_events),
                event_type="tool_result",
                tone=tone,
                title="Respuesta de herramienta",
                detail=detail,
                actor=tool_name,
                node_key="tool_action",
                payload={"tool_name": tool_name, "condition": injected_conditions[0] if injected_conditions else ""},
            )
        )

    decision_detail = "El agente confirma que puede cerrar el caso con evidencia y gates visibles."
    decision_tone: str = "success"
    if "tool_timeout" in injected_conditions or "tool_timeout" in scenario.scenario_key:
        decision_detail = "El agente ejecuta compensacion, evita side effects adicionales y escala para continuidad controlada."
        decision_tone = "warning"
    elif "no_evidence" in injected_conditions or "no_evidence" in scenario.scenario_key:
        decision_detail = "El agente activa la politica no-evidence, evita inventar datos y decide escalar al humano."
        decision_tone = "warning"
    elif llm_output is not None and llm_output.observed_decisions:
        decision_detail = llm_output.observed_decisions[0]
    events.append(
        SimulationEvent(
            event_key="event-7",
            event_index=7 + len(fault_events),
            event_type="decision",
            tone=decision_tone,  # type: ignore[arg-type]
            title="Decision del agente",
            detail=decision_detail,
            actor="agent",
            node_key="decision_policy",
        )
    )

    if scenario.approval_gates:
        approval_rejected = "approval_rejected" in injected_conditions
        events.append(
            SimulationEvent(
                event_key="event-8",
                event_index=8 + len(fault_events),
                event_type="approval_gate",
                tone="blocked" if approval_rejected else "info",
                title="Approval gate",
                detail=(
                    "El humano rechaza continuar y el flujo queda detenido de forma segura."
                    if approval_rejected
                    else scenario.approval_gates[0]
                ),
                actor="human",
                node_key="human_gate",
            )
        )

    if scenario.memory_writes:
        events.append(
            SimulationEvent(
                event_key="event-9",
                event_index=9 + len(fault_events),
                event_type="memory_write",
                tone="info",
                title="Escritura de memoria",
                detail=", ".join(scenario.memory_writes),
                actor="agent",
                node_key="memory_context",
            )
        )

    final_detail = scenario.expected_outcome
    final_tone = "success"
    if "approval_rejected" in injected_conditions:
        final_detail = "El flujo se cierra sin promover la accion, preservando trazabilidad y seguridad."
        final_tone = "warning"
    events.append(
        SimulationEvent(
            event_key="event-10",
            event_index=10 + len(fault_events),
            event_type="end",
            tone=final_tone,  # type: ignore[arg-type]
            title="Resultado del escenario",
            detail=final_detail,
            actor="system",
            node_key="end_result",
        )
    )

    return [
        item.model_copy(update={"event_index": index})
        for index, item in enumerate(events, start=1)
    ]


def _hard_gate_evaluation(
    scenario: SimulationScenario,
    *,
    events: list[SimulationEvent],
    injected_conditions: list[str],
) -> tuple[str, list[str], str]:
    findings: list[str] = []
    event_types = {item.event_type for item in events}
    decision_details = " ".join(item.detail.lower() for item in events if item.event_type == "decision")

    if scenario.memory_reads and "memory_read" not in event_types:
        findings.append("No se registraron lecturas de memoria requeridas.")
    if scenario.memory_writes and "memory_write" not in event_types:
        findings.append("No se registraron escrituras de memoria esperadas.")
    if scenario.approval_gates and "approval_gate" not in event_types:
        findings.append("No se visualizo el approval gate esperado.")
    if "tool_timeout" in injected_conditions or "tool_timeout" in scenario.scenario_key:
        if "compens" not in decision_details and "escal" not in decision_details:
            findings.append("El timeout no activo una compensacion o escalation visible.")
    if "no_evidence" in injected_conditions or "no_evidence" in scenario.scenario_key:
        if "no-evidence" not in decision_details and "escal" not in decision_details and "evidence" not in decision_details:
            findings.append("La politica no-evidence no quedo visible en la decision.")
    if "approval_rejected" in injected_conditions and "approval_gate" not in event_types:
        findings.append("Se simulo rechazo humano pero no quedo reflejado en el gate.")

    status = "pass" if not findings else "fail"
    summary = (
        "La corrida cumple los hard gates deterministas del escenario."
        if status == "pass"
        else "La corrida no cumple todos los hard gates deterministas del escenario."
    )
    return status, findings, summary


def _provisional_judgement(
    scenario_key: str,
    hard_gate_status: str,
    findings: list[str],
    summary: str,
) -> SimulationJudgement:
    return SimulationJudgement(
        scenario_key=scenario_key,
        hard_gate_status=hard_gate_status,
        llm_judgment=hard_gate_status,
        final_status=hard_gate_status,
        score=100 if hard_gate_status == "pass" else 55,
        summary=summary,
        hard_gate_findings=list(findings),
        findings=[
            SimulationJudgementFinding(
                finding_key=f"hard-gate-{index + 1}",
                title="Hallazgo determinista",
                severity="blocking",
                detail=detail,
                suggested_action="Corregir el flujo o reforzar la simulacion antes de aprobar Validate.",
            )
            for index, detail in enumerate(findings)
        ],
    )


def execute_validation_simulation(
    *,
    blueprint: BlueprintArtifact,
    scenario: SimulationScenario,
    request: ValidationSimulationRunRequest,
    blueprint_version_number: int | None,
    specification_artifact_id: UUID | None,
    scenario_version_number: int,
    runtime_settings=None,
    stage_context: StageContextBundle | None = None,
    source_action: str = "run_validation_simulation",
) -> tuple[SimulationRunRecord, list]:
    started = perf_counter()
    warnings: list[str] = []
    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="La simulacion ejecuta transiciones deterministas para tools, memoria, aprobaciones y errores tipados.",
        )
    ]
    llm_result = None
    llm_output: ValidationSimulationOutput | None = None
    llm_service = skill_runtime._builder_service_for_stage("validate", runtime_settings)
    llm_input = ValidationScenarioSimulationInput(
        blueprint=blueprint,
        scenario=_scenario_item_from_scenario(scenario),
        source_refs=scenario.source_refs,
    )
    llm_result = llm_service.simulate_validation_scenario(llm_input, context_bundle=stage_context)
    skill_runtime._append_warning(warnings, llm_result.warning if llm_result is not None else None)
    if llm_result is not None and isinstance(llm_result.artifact, ValidationSimulationOutput):
        llm_output = ValidationSimulationOutput.model_validate(llm_result.artifact.model_dump(mode="json"))
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.llm_inference,
                detail=skill_runtime._llm_evidence_detail(llm_result, "simulo respuestas conversacionales y observaciones semanticas"),
            )
        )
    elif llm_result is not None and llm_result.warning is None:
        skill_runtime._append_warning(
            warnings,
            "No hubo salida estructurada valida para la simulacion LLM; se conserva el replay determinista.",
        )

    events = _build_deterministic_simulation_events(
        scenario,
        initial_input=request.initial_input_override,
        injected_conditions=request.injected_conditions,
        llm_output=llm_output,
    )
    hard_gate_status, hard_gate_findings, summary = _hard_gate_evaluation(
        scenario,
        events=events,
        injected_conditions=request.injected_conditions,
    )
    judgement = _provisional_judgement(
        scenario.scenario_key,
        hard_gate_status,
        hard_gate_findings,
        summary,
    )
    now = skill_runtime.utc_now()
    record = SimulationRunRecord(
        id=uuid4(),
        specification_artifact_id=specification_artifact_id,
        blueprint_version_number=blueprint_version_number,
        scenario_key=scenario.scenario_key,
        scenario_title=scenario.title,
        scenario_version_number=scenario_version_number,
        source_action=source_action,
        status=ArtifactStatus.ready if hard_gate_status == "pass" else ArtifactStatus.needs_review,
        execution_state="completed",
        hard_gate_status=hard_gate_status,
        final_status=hard_gate_status,
        active_node_key=events[-1].node_key if events else "",
        summary=summary,
        injected_conditions=list(request.injected_conditions),
        deterministic_signature=_stable_signature(
            {
                "scenario_key": scenario.scenario_key,
                "initial_input": request.initial_input_override or scenario.initial_input,
                "injected_conditions": request.injected_conditions,
                "events": [item.model_dump(mode="json") for item in events],
            }
        ),
        events=events,
        judgement=judgement,
        created_at=now,
        updated_at=now,
    )
    duration_ms = int((perf_counter() - started) * 1000)
    trace = _build_trace(
        skill_key="validation_scenario_simulation_skill",
        label="Validation scenario simulation skill",
        input_payload={
            "scenario": scenario.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
        },
        output_payload=record.model_dump(mode="json"),
        status=record.status,
        result_summary=record.summary,
        warnings=warnings,
        evidence=evidence,
        llm_result=llm_result,
        duration_ms=duration_ms,
    )
    return record, [trace]


def _map_llm_findings(findings: list[CritiqueFinding]) -> list[SimulationJudgementFinding]:
    return [
        SimulationJudgementFinding(
            finding_key=item.finding_key,
            title=item.title,
            severity=item.severity,
            detail=item.detail,
            suggested_action=item.suggested_action,
            source_refs=list(item.source_refs),
        )
        for item in findings
    ]


def judge_validation_simulation_run(
    *,
    run_record: SimulationRunRecord,
    blueprint: BlueprintArtifact,
    scenario: SimulationScenario,
    runtime_settings=None,
    stage_context: StageContextBundle | None = None,
) -> tuple[SimulationJudgement, list]:
    started = perf_counter()
    warnings: list[str] = []
    evidence = [
        EvidenceItem(
            source=EvidenceSource.rule_engine,
            detail="El hard gate determinista conserva autoridad sobre bloqueos antes del juicio cualitativo LLM.",
        )
    ]
    llm_result = None
    llm_output: ValidationRunJudgmentOutput | None = None
    llm_service = skill_runtime._builder_service_for_stage("validate", runtime_settings)
    llm_input = ValidationRunJudgmentInput(
        simulation=ValidationSimulationOutput(
            scenario_key=run_record.scenario_key,
            result_status=run_record.hard_gate_status,
            simulated_transcript=[item.detail for item in run_record.events if item.event_type in {"input", "agent_response", "end"}],
            observed_decisions=[item.detail for item in run_record.events if item.event_type == "decision"],
            tool_interactions=[item.detail for item in run_record.events if item.event_type in {"tool_call", "tool_result"}],
            issues=list(run_record.judgement.hard_gate_findings if run_record.judgement is not None else []),
        ),
        blueprint=blueprint,
        source_refs=scenario.source_refs,
    )
    llm_result = llm_service.judge_validation_run(llm_input, context_bundle=stage_context)
    skill_runtime._append_warning(warnings, llm_result.warning if llm_result is not None else None)
    if llm_result is not None and isinstance(llm_result.artifact, ValidationRunJudgmentOutput):
        llm_output = ValidationRunJudgmentOutput.model_validate(llm_result.artifact.model_dump(mode="json"))
        evidence.append(
            EvidenceItem(
                source=EvidenceSource.llm_inference,
                detail=skill_runtime._llm_evidence_detail(llm_result, "emitio un juicio cualitativo sobre transcript y criterios"),
            )
        )
    elif llm_result is not None and llm_result.warning is None:
        skill_runtime._append_warning(
            warnings,
            "No hubo salida estructurada valida para el judge LLM; se conserva el veredicto determinista.",
        )

    llm_judgment = llm_output.judgment if llm_output is not None else run_record.hard_gate_status
    severity_order = {"pass": 0, "needs_revision": 1, "fail": 2}
    final_status = (
        run_record.hard_gate_status
        if severity_order.get(run_record.hard_gate_status, 1) >= severity_order.get(llm_judgment, 1)
        else llm_judgment
    )
    findings = _map_llm_findings(llm_output.findings) if llm_output is not None else []
    judgement = SimulationJudgement(
        scenario_key=run_record.scenario_key,
        hard_gate_status=run_record.hard_gate_status,
        llm_judgment=llm_judgment,
        final_status=final_status,
        score=llm_output.score if llm_output is not None else (100 if run_record.hard_gate_status == "pass" else 55),
        summary=(
            llm_output.summary
            if llm_output is not None and llm_output.summary
            else (run_record.judgement.summary if run_record.judgement is not None else run_record.summary)
        ),
        hard_gate_findings=list(run_record.judgement.hard_gate_findings if run_record.judgement is not None else []),
        findings=(
            findings
            if findings
            else list(run_record.judgement.findings if run_record.judgement is not None else [])
        ),
    )
    duration_ms = int((perf_counter() - started) * 1000)
    trace = _build_trace(
        skill_key="validation_run_judgement_skill",
        label="Validation run judgement skill",
        input_payload=llm_input.model_dump(mode="json"),
        output_payload=judgement.model_dump(mode="json"),
        status=ArtifactStatus.ready if judgement.final_status == "pass" else ArtifactStatus.needs_review,
        result_summary=judgement.summary,
        warnings=warnings,
        evidence=evidence,
        llm_result=llm_result,
        duration_ms=duration_ms,
    )
    return judgement, [trace]
