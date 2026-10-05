from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.diagram_center.contracts import (
    AgentNodeKind,
    DiagramEdge,
    DiagramGenerationInput,
    DiagramModel,
    DiagramNode,
    DiagramNotation,
    ToolNodeKind,
)
from app.services.deliverable_catalog.project_generation_context import ProjectGenerationContext


DETERMINISTIC_DIAGRAM_KEYS = frozenset(
    {
        "target_capabilities_map",
        "agent_orchestration",
        "decision_model",
        "tool_capability_map",
        "security_guardrails",
        "human_intervention_flow",
        "tool_contract_sequence",
        "deployment_decision_matrix",
        "prompt_reasoning_playbook",
    }
)


@dataclass(frozen=True)
class DeterministicDiagramContextError(ValueError):
    code: str
    message: str
    missing_inputs: tuple[str, ...] = ()


def supports_deterministic_diagram(diagram_key: str) -> bool:
    return str(diagram_key or "").strip().removeprefix("diagram.") in DETERMINISTIC_DIAGRAM_KEYS


def _brief(input_payload: DiagramGenerationInput, fallback: str) -> str:
    brief = " ".join(str(input_payload.context_brief or "").split()).strip()
    if brief:
        return brief[:700]
    for item in input_payload.resolved_inputs:
        value = " ".join(str(item.get("brief") or "").split()).strip()
        if value:
            return value[:700]
    return fallback


def _source_refs(input_payload: DiagramGenerationInput) -> list[str]:
    refs = [str(ref or "").strip() for ref in input_payload.source_refs if str(ref or "").strip()]
    for item in input_payload.resolved_inputs:
        for ref in item.get("artifact_refs", []) or []:
            normalized = str(ref or "").strip()
            if normalized:
                refs.append(normalized)
    return list(dict.fromkeys(refs))[:12]


def _require_context(input_payload: DiagramGenerationInput) -> list[str]:
    refs = _source_refs(input_payload)
    missing = tuple(str(key) for key in input_payload.missing_required_inputs if str(key).strip())
    if missing:
        raise DeterministicDiagramContextError(
            code="approved_context_missing",
            message="Faltan entradas aprobadas para construir el diagrama deterministico.",
            missing_inputs=missing,
        )
    if not refs:
        raise DeterministicDiagramContextError(
            code="source_refs_missing",
            message="No hay referencias aprobadas para respaldar el diagrama.",
            missing_inputs=("source_refs",),
        )
    return refs


def _stage_for_input_key(input_key: str) -> str:
    normalized = str(input_key or "").strip()
    if normalized.startswith(("discovery.", "session.discovery")):
        return "discover"
    if normalized.startswith(("definition.", "session.canvas")):
        return "define"
    if normalized.startswith(("design.", "blueprint.")):
        return "design"
    if normalized.startswith(("tools.",)):
        return "tools"
    if normalized.startswith(("memory.", "knowledge.")):
        return "memory"
    if normalized.startswith(("estimate.",)):
        return "estimate"
    if normalized.startswith(("validation.",)):
        return "validate"
    return "design"


def _context_payload(input_payload: DiagramGenerationInput) -> dict[str, object]:
    stages: dict[str, object] = {}
    for item in input_payload.resolved_inputs:
        input_key = str(item.get("input_key") or "")
        stage_key = _stage_for_input_key(input_key)
        evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        stage_values: list[object] = []
        for evidence_item in evidence:
            if isinstance(evidence_item, dict) and "content" in evidence_item:
                stage_values.append(evidence_item["content"])
        if not stage_values and item.get("brief"):
            stage_values.append({"summary": item.get("brief")})
        if not stage_values:
            continue
        if stage_key not in stages:
            stages[stage_key] = stage_values[0] if len(stage_values) == 1 else {"evidence": stage_values}
        else:
            previous = stages[stage_key]
            stages[stage_key] = {"evidence": [previous, *stage_values]}
    source_context = input_payload.source_context or {}
    project = source_context.get("project") if isinstance(source_context.get("project"), dict) else {}
    return {
        "project_title": str(project.get("title") or input_payload.title),
        "approved_context_refs": _source_refs(input_payload),
        "approved_context": {
            "stages": stages,
            "artifacts": {},
        },
    }


def _generation_context(input_payload: DiagramGenerationInput) -> ProjectGenerationContext:
    return ProjectGenerationContext.from_approved_payload(
        _context_payload(input_payload),
        deliverable_key=f"diagram.{input_payload.diagram_key}",
        policy=None,
    )


def _label(value: object, fallback: str, *, limit: int = 72) -> str:
    text = " ".join(str(value or "").split()).strip()
    return (text or fallback)[:limit]


def _first(values: list[str], fallback: str) -> str:
    return _label(values[0] if values else "", fallback)


def _tool_label(context: ProjectGenerationContext) -> str:
    if not context.tools:
        return "Pendiente: tools aprobadas"
    names = [tool.name for tool in context.tools if tool.name]
    return _label(", ".join(names[:3]), "Pendiente: tools aprobadas")


def _context_metadata(context: ProjectGenerationContext) -> dict[str, Any]:
    return {
        "context_version": context.context_version,
        "input_fingerprint": context.input_fingerprint,
        "specificity_anchors": [anchor.value for anchor in context.specificity_anchors],
        "missing_fields": [field.field for field in context.missing_fields],
        "estimated_input_tokens": context.estimated_input_tokens,
    }


def _node(node_id: str, label: str, kind: str, refs: list[str], **metadata: Any) -> DiagramNode:
    return DiagramNode(
        id=node_id,
        label=label,
        kind=kind,
        description=str(metadata.pop("description", "") or ""),
        agent_kind=metadata.pop("agent_kind", None),
        memory_kind=metadata.pop("memory_kind", None),
        tool_kind=metadata.pop("tool_kind", None),
        metadata={key: value for key, value in metadata.items() if value not in (None, "", [], {})},
        source_refs=refs[:4],
    )


def _edge(
    edge_id: str,
    source: str,
    target: str,
    label: str,
    refs: list[str] | None = None,
    kind: str = "relationship",
) -> DiagramEdge:
    return DiagramEdge(id=edge_id, source=source, target=target, label=label, kind=kind, source_refs=list(refs or [])[:4])


def _model(
    input_payload: DiagramGenerationInput,
    *,
    nodes: list[DiagramNode],
    edges: list[DiagramEdge],
    description: str,
    context: ProjectGenerationContext | None = None,
    notation: DiagramNotation | None = None,
    direction: str = "LR",
) -> DiagramModel:
    refs = _source_refs(input_payload)
    return DiagramModel(
        diagram_key=input_payload.diagram_key,
        title=input_payload.title,
        description=description,
        notation=notation or input_payload.notation,
        direction=direction,  # type: ignore[arg-type]
        nodes=nodes,
        edges=edges,
        source_refs=refs,
        assumptions=[
            "Modelo deterministico construido solo con contexto aprobado.",
            "Las integraciones sin binding operativo se representan como contratos o decisiones pendientes.",
        ],
        metadata={
            "generated_by": "deterministic_python",
            "prompt_spec_version": input_payload.prompt_spec_version,
            "source_contract": input_payload.source_contract,
            "renderer_key": input_payload.renderer_key,
            **(_context_metadata(context) if context is not None else {}),
        },
    )


def _capabilities(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    context = _generation_context(input_payload)
    objective = context.desired_outcome or context.problem_statement or input_payload.objective
    capability = _first(context.mvp_scope or context.objectives, "Pendiente: capacidad objetivo")
    limit = _first(context.out_of_scope or context.constraints, "Pendiente: limites aprobados")
    nodes = [
        _node("objective", _label(objective, "Objetivo aprobado"), "goal", refs),
        _node("inputs", _label(context.current_process, "Entradas del proceso"), "input", refs),
        _node("capabilities", capability, "capability", refs),
        _node("limits", limit, "guardrail", refs),
        _node("outputs", _label(context.desired_outcome, "Resultado esperado pendiente"), "output", refs),
    ]
    edges = [
        _edge("e1", "objective", "capabilities", "define"),
        _edge("e2", "inputs", "capabilities", "alimenta"),
        _edge("e3", "capabilities", "outputs", "produce"),
        _edge("e4", "limits", "capabilities", "acota"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Mapa de capacidades objetivo."), context=context)


def _agent_orchestration(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    context = _generation_context(input_payload)
    role_names = [role.value for role in context.roles]
    planner = _label(role_names[0] if role_names else context.reasoning_pattern, "Pendiente: rol planificador")
    executor = _label(role_names[1] if len(role_names) > 1 else context.architecture, "Pendiente: rol ejecutor")
    nodes = [
        _node("user_request", _label(context.current_user or context.problem_statement, "Solicitud aprobada"), "input", refs),
        _node("supervisor", _label(context.coordination_model or context.architecture, "Orquestador pendiente"), "orchestrator", refs, agent_kind=AgentNodeKind.orchestrator),
        _node("planner_agent", planner, "worker_agent", refs, agent_kind=AgentNodeKind.worker),
        _node("executor_agent", executor, "worker_agent", refs, agent_kind=AgentNodeKind.worker),
        _node("tool_layer", _tool_label(context), "tool_layer", refs, tool_kind=ToolNodeKind.internal_tool),
        _node("memory_context", _label(context.memory_strategy, "Pendiente: memoria aprobada"), "memory", refs),
        _node("guardrails", _first(context.guardrails, "Pendiente: guardrails aprobados"), "guardrail_gate", refs, tool_kind=ToolNodeKind.guardrail_gate),
        _node("hitl_gate", _first(context.nondelegable_decisions, "Pendiente: aprobacion humana"), "human_gate", refs, agent_kind=AgentNodeKind.human_gate),
        _node("output", _label(context.desired_outcome, "Resultado verificable"), "output", refs),
        _node("fallback", _first(context.open_questions and [item.question for item in context.open_questions] or [], "Escalamiento por dato faltante"), "fallback", refs),
    ]
    edges = [
        _edge("e1", "user_request", "supervisor", "inicia"),
        _edge("e2", "supervisor", "planner_agent", "handoff planifica"),
        _edge("e3", "planner_agent", "executor_agent", "handoff delega"),
        _edge("e4", "executor_agent", "tool_layer", "tool call controlado"),
        _edge("e5", "supervisor", "memory_context", "lee checkpoints"),
        _edge("e6", "supervisor", "guardrails", "verifica policy"),
        _edge("e7", "guardrails", "hitl_gate", "requiere aprobacion"),
        _edge("e8", "hitl_gate", "output", "aprueba resultado"),
        _edge("e9", "guardrails", "fallback", "fallback ante error"),
        _edge("e10", "fallback", "supervisor", "retry escalado"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Orquestacion agentiva."), context=context)


def _decision_model(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    context = _generation_context(input_payload)
    nodes = [
        _node("event", _label(context.current_process or context.problem_statement, "Evento aprobado"), "event", refs),
        _node("rule", _first(context.constraints or context.guardrails, "Pendiente: regla aprobada"), "rule", refs),
        _node("decision", _label(context.reasoning_pattern, "Decision del agente pendiente"), "decision", refs),
        _node("human_review", _first(context.nondelegable_decisions, "Escalamiento humano pendiente"), "human_gate", refs),
        _node("result", _label(context.desired_outcome, "Resultado trazable"), "output", refs),
    ]
    edges = [
        _edge("e1", "event", "rule", "evalua"),
        _edge("e2", "rule", "decision", "habilita"),
        _edge("e3", "decision", "human_review", "si no delegable"),
        _edge("e4", "decision", "result", "si permitido"),
        _edge("e5", "human_review", "result", "aprueba o rechaza"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Modelo de decisiones."), context=context)


def _tool_capability(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    context = _generation_context(input_payload)
    nodes = [
        _node("capability", _first(context.mvp_scope or context.objectives, "Capacidad pendiente"), "capability", refs),
        _node("tool_contract", _tool_label(context), "tool_contract", refs),
        _node("permission", _first(context.nondelegable_decisions or context.constraints, "Permisos pendientes"), "permission", refs),
        _node("side_effect_gate", _first(context.guardrails, "Gate side effect pendiente"), "guardrail_gate", refs),
        _node("audit", "Auditoria", "audit", refs),
    ]
    edges = [
        _edge("e1", "capability", "tool_contract", "requiere"),
        _edge("e2", "tool_contract", "permission", "declara"),
        _edge("e3", "permission", "side_effect_gate", "controla"),
        _edge("e4", "side_effect_gate", "audit", "registra"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Mapa tool-capability."), context=context)


def _security_guardrails(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    context = _generation_context(input_payload)
    nodes = [
        _node("data_action", _label(context.current_process or context.problem_statement, "Dato o accion sensible"), "risk_source", refs),
        _node("risk", _first(context.risks, "Riesgo pendiente"), "risk", refs),
        _node("control", _first(context.guardrails, "Control pendiente"), "guardrail", refs),
        _node("approval", _first(context.nondelegable_decisions, "Aprobacion pendiente"), "human_gate", refs),
        _node("audit", "Registro auditable", "audit", refs),
    ]
    edges = [
        _edge("e1", "data_action", "risk", "expone"),
        _edge("e2", "risk", "control", "mitiga"),
        _edge("e3", "control", "approval", "solicita"),
        _edge("e4", "approval", "audit", "evidencia"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Guardrails de seguridad."), context=context)


def _human_intervention(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    context = _generation_context(input_payload)
    nodes = [
        _node("trigger", _first(context.nondelegable_decisions or context.risks, "Disparador HITL pendiente"), "trigger", refs),
        _node("pause", "Pausa controlada", "pause", refs),
        _node("approver", _label(context.current_user, "Aprobador humano pendiente"), "human_gate", refs),
        _node("decision", _label(context.reasoning_pattern, "Decision pendiente"), "decision", refs),
        _node("resume", "Reanudacion y auditoria", "audit", refs),
    ]
    edges = [
        _edge("e1", "trigger", "pause", "detiene"),
        _edge("e2", "pause", "approver", "notifica"),
        _edge("e3", "approver", "decision", "resuelve"),
        _edge("e4", "decision", "resume", "reanuda"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Flujo de intervencion humana."), context=context)


def _tool_sequence(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    context = _generation_context(input_payload)
    nodes = [
        _node("requester", _label(context.current_user, "Solicitante pendiente"), "actor", refs),
        _node("agent", _label(context.architecture or context.reasoning_pattern, "Agente pendiente"), "agent", refs),
        _node("gate", _first(context.nondelegable_decisions or context.guardrails, "Gate de permiso pendiente"), "guardrail_gate", refs),
        _node("contract", _tool_label(context), "tool_contract", refs),
        _node("fallback", "Fallback manual", "fallback", refs),
        _node("result", _label(context.desired_outcome, "Resultado pendiente"), "output", refs),
    ]
    edges = [
        _edge("e1", "requester", "agent", "solicita", refs, kind="message"),
        _edge("e2", "agent", "gate", "valida", refs, kind="message"),
        _edge("e3", "gate", "contract", "autoriza contrato", refs, kind="message"),
        _edge("e4", "contract", "result", "devuelve", refs, kind="message"),
        _edge("e5", "gate", "fallback", "si falta binding", refs, kind="message"),
    ]
    return _model(
        input_payload,
        nodes=nodes,
        edges=edges,
        description=_brief(input_payload, "Secuencia de contrato de herramientas."),
        context=context,
        notation=DiagramNotation.sequence,
        direction="LR",
    )


def _deployment_decision(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    context = _generation_context(input_payload)
    nodes = [
        _node("options", "Opciones de despliegue", "option", refs),
        _node("criteria", _first(context.acceptance_criteria or context.objectives, "Criterios pendientes"), "criteria", refs),
        _node("constraints", _first(context.constraints, "Restricciones pendientes"), "constraint", refs),
        _node("pending_decision", _first(context.open_questions and [item.question for item in context.open_questions] or [], "Decision pendiente"), "decision", refs),
        _node("evidence", _label(context.project_title or context.problem_statement, "Evidencia requerida"), "evidence", refs),
    ]
    edges = [
        _edge("e1", "options", "criteria", "se compara por"),
        _edge("e2", "criteria", "constraints", "filtra"),
        _edge("e3", "constraints", "pending_decision", "requiere decision"),
        _edge("e4", "evidence", "pending_decision", "respalda"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Matriz de decision de despliegue."), context=context)


def _prompt_playbook(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    context = _generation_context(input_payload)
    nodes = [
        _node("context", _label(context.problem_statement, "Contexto aprobado"), "context", refs),
        _node("instructions", _label(context.reasoning_pattern, "Instrucciones pendientes"), "prompt", refs),
        _node("tool_use", _tool_label(context), "tool_call", refs),
        _node("verification", _first(context.acceptance_criteria or context.guardrails, "Verificacion pendiente"), "evaluator", refs),
        _node("hitl", _first(context.nondelegable_decisions, "HITL pendiente"), "human_gate", refs),
        _node("response", _label(context.desired_outcome, "Respuesta pendiente"), "output", refs),
    ]
    edges = [
        _edge("e1", "context", "instructions", "fundamenta"),
        _edge("e2", "instructions", "tool_use", "orienta"),
        _edge("e3", "tool_use", "verification", "evidencia"),
        _edge("e4", "verification", "hitl", "si requiere aprobacion"),
        _edge("e5", "verification", "response", "si cumple"),
        _edge("e6", "hitl", "response", "autoriza"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Playbook de prompt y razonamiento."), context=context)


def build_deterministic_diagram(input_payload: DiagramGenerationInput) -> DiagramModel:
    key = str(input_payload.diagram_key or "").strip()
    if key == "target_capabilities_map":
        return _capabilities(input_payload)
    if key == "agent_orchestration":
        return _agent_orchestration(input_payload)
    if key == "decision_model":
        return _decision_model(input_payload)
    if key == "tool_capability_map":
        return _tool_capability(input_payload)
    if key == "security_guardrails":
        return _security_guardrails(input_payload)
    if key == "human_intervention_flow":
        return _human_intervention(input_payload)
    if key == "tool_contract_sequence":
        return _tool_sequence(input_payload)
    if key == "deployment_decision_matrix":
        return _deployment_decision(input_payload)
    if key == "prompt_reasoning_playbook":
        return _prompt_playbook(input_payload)
    raise DeterministicDiagramContextError(
        code="deterministic_diagram_not_supported",
        message=f"No deterministic diagram builder is registered for {key}.",
    )
