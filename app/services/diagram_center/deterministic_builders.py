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
        },
    )


def _capabilities(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    nodes = [
        _node("objective", "Objetivo aprobado", "goal", refs),
        _node("inputs", "Entradas del proceso", "input", refs),
        _node("capabilities", "Capacidades objetivo", "capability", refs),
        _node("limits", "Limites y exclusiones", "guardrail", refs),
        _node("outputs", "Resultados esperados", "output", refs),
    ]
    edges = [
        _edge("e1", "objective", "capabilities", "define"),
        _edge("e2", "inputs", "capabilities", "alimenta"),
        _edge("e3", "capabilities", "outputs", "produce"),
        _edge("e4", "limits", "capabilities", "acota"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Mapa de capacidades objetivo."))


def _agent_orchestration(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    nodes = [
        _node("user_request", "Solicitud del usuario", "input", refs),
        _node("supervisor", "Supervisor / Orquestador", "orchestrator", refs, agent_kind=AgentNodeKind.orchestrator),
        _node("planner_agent", "Agente planificador", "worker_agent", refs, agent_kind=AgentNodeKind.worker),
        _node("executor_agent", "Agente ejecutor", "worker_agent", refs, agent_kind=AgentNodeKind.worker),
        _node("tool_layer", "Herramientas y contratos", "tool_layer", refs, tool_kind=ToolNodeKind.internal_tool),
        _node("memory_context", "Memoria y contexto", "memory", refs),
        _node("guardrails", "Guardrails y politicas", "guardrail_gate", refs, tool_kind=ToolNodeKind.guardrail_gate),
        _node("hitl_gate", "Aprobacion humana HITL", "human_gate", refs, agent_kind=AgentNodeKind.human_gate),
        _node("output", "Resultado / entregable", "output", refs),
        _node("fallback", "Fallback y escalamiento", "fallback", refs),
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
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Orquestacion agentiva."))


def _decision_model(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    nodes = [
        _node("event", "Evento o solicitud", "event", refs),
        _node("rule", "Regla aprobada", "rule", refs),
        _node("decision", "Decision del agente", "decision", refs),
        _node("human_review", "Escalamiento humano", "human_gate", refs),
        _node("result", "Resultado trazable", "output", refs),
    ]
    edges = [
        _edge("e1", "event", "rule", "evalua"),
        _edge("e2", "rule", "decision", "habilita"),
        _edge("e3", "decision", "human_review", "si no delegable"),
        _edge("e4", "decision", "result", "si permitido"),
        _edge("e5", "human_review", "result", "aprueba o rechaza"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Modelo de decisiones."))


def _tool_capability(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    nodes = [
        _node("capability", "Capacidad", "capability", refs),
        _node("tool_contract", "Tool propuesta / contrato", "tool_contract", refs),
        _node("permission", "Permisos", "permission", refs),
        _node("side_effect_gate", "Gate side effect", "guardrail_gate", refs),
        _node("audit", "Auditoria", "audit", refs),
    ]
    edges = [
        _edge("e1", "capability", "tool_contract", "requiere"),
        _edge("e2", "tool_contract", "permission", "declara"),
        _edge("e3", "permission", "side_effect_gate", "controla"),
        _edge("e4", "side_effect_gate", "audit", "registra"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Mapa tool-capability."))


def _security_guardrails(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    nodes = [
        _node("data_action", "Dato o accion sensible", "risk_source", refs),
        _node("risk", "Riesgo", "risk", refs),
        _node("control", "Control / guardrail", "guardrail", refs),
        _node("approval", "Aprobacion", "human_gate", refs),
        _node("audit", "Registro auditable", "audit", refs),
    ]
    edges = [
        _edge("e1", "data_action", "risk", "expone"),
        _edge("e2", "risk", "control", "mitiga"),
        _edge("e3", "control", "approval", "solicita"),
        _edge("e4", "approval", "audit", "evidencia"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Guardrails de seguridad."))


def _human_intervention(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    nodes = [
        _node("trigger", "Disparador HITL", "trigger", refs),
        _node("pause", "Pausa controlada", "pause", refs),
        _node("approver", "Aprobador humano", "human_gate", refs),
        _node("decision", "Decision", "decision", refs),
        _node("resume", "Reanudacion y auditoria", "audit", refs),
    ]
    edges = [
        _edge("e1", "trigger", "pause", "detiene"),
        _edge("e2", "pause", "approver", "notifica"),
        _edge("e3", "approver", "decision", "resuelve"),
        _edge("e4", "decision", "resume", "reanuda"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Flujo de intervencion humana."))


def _tool_sequence(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    nodes = [
        _node("requester", "Solicitante", "actor", refs),
        _node("agent", "Agente", "agent", refs),
        _node("gate", "Gate de permiso", "guardrail_gate", refs),
        _node("contract", "Contrato de tool", "tool_contract", refs),
        _node("fallback", "Fallback manual", "fallback", refs),
        _node("result", "Resultado", "output", refs),
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
        notation=DiagramNotation.sequence,
        direction="LR",
    )


def _deployment_decision(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    nodes = [
        _node("options", "Opciones de despliegue", "option", refs),
        _node("criteria", "Criterios", "criteria", refs),
        _node("constraints", "Restricciones", "constraint", refs),
        _node("pending_decision", "Decision pendiente", "decision", refs),
        _node("evidence", "Evidencia requerida", "evidence", refs),
    ]
    edges = [
        _edge("e1", "options", "criteria", "se compara por"),
        _edge("e2", "criteria", "constraints", "filtra"),
        _edge("e3", "constraints", "pending_decision", "requiere decision"),
        _edge("e4", "evidence", "pending_decision", "respalda"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Matriz de decision de despliegue."))


def _prompt_playbook(input_payload: DiagramGenerationInput) -> DiagramModel:
    refs = _require_context(input_payload)
    nodes = [
        _node("context", "Contexto aprobado", "context", refs),
        _node("instructions", "Instrucciones", "prompt", refs),
        _node("tool_use", "Uso de herramienta", "tool_call", refs),
        _node("verification", "Verificacion", "evaluator", refs),
        _node("hitl", "HITL", "human_gate", refs),
        _node("response", "Respuesta", "output", refs),
    ]
    edges = [
        _edge("e1", "context", "instructions", "fundamenta"),
        _edge("e2", "instructions", "tool_use", "orienta"),
        _edge("e3", "tool_use", "verification", "evidencia"),
        _edge("e4", "verification", "hitl", "si requiere aprobacion"),
        _edge("e5", "verification", "response", "si cumple"),
        _edge("e6", "hitl", "response", "autoriza"),
    ]
    return _model(input_payload, nodes=nodes, edges=edges, description=_brief(input_payload, "Playbook de prompt y razonamiento."))


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
