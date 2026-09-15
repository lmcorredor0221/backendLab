from __future__ import annotations

from collections import Counter, deque
from typing import Any

from app.services.diagram_center.contracts import DiagramModel, DiagramNotation, DiagramQualityReport
from app.services.diagram_center.layout_analysis import analyze_diagram_complexity
from app.services.diagram_center.layout_contracts import DiagramLayoutRisk
from app.services.diagram_center.layout_split import recommend_diagram_split


FORBIDDEN_CONTENT = ("api_key", "password", "secret=", "sk-proj-", "BEGIN PRIVATE KEY")


def _kind(value: str) -> str:
    return str(value or "").lower().replace("-", "_").replace(" ", "_")


def _has_kind(model: DiagramModel, *tokens: str) -> bool:
    return any(any(token in _kind(node.kind) for token in tokens) for node in model.nodes)


def _metadata_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_metadata_text(child)}" for key, child in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_metadata_text(child) for child in value)
    return str(value or "")


def _node_text(node) -> str:
    return _kind(
        " ".join(
            [
                node.kind,
                node.label,
                node.description,
                str(node.agent_kind or ""),
                str(node.memory_kind or ""),
                str(node.tool_kind or ""),
                _metadata_text(node.metadata),
            ]
        )
    )


def _edge_text(edge) -> str:
    return _kind(" ".join([edge.kind, edge.label, _metadata_text(edge.metadata)]))


def _contains_any(values: list[str], *tokens: str) -> bool:
    normalized_tokens = [_kind(token) for token in tokens]
    return any(token in value for value in values for token in normalized_tokens)


def _agent_orchestration_checks(
    model: DiagramModel,
    checks: dict[str, bool],
    warnings: list[str],
    errors: list[str],
) -> None:
    if _kind(model.diagram_key) != "agent_orchestration":
        return

    node_texts = [_node_text(node) for node in model.nodes]
    edge_texts = [_edge_text(edge) for edge in model.edges]
    all_texts = [*node_texts, *edge_texts, _kind(model.description), _kind(_metadata_text(model.metadata))]
    worker_texts = [
        text
        for text in node_texts
        if not _contains_any([text], "orchestrator", "orquestador", "supervisor")
        and _contains_any(
            [text],
            "agent",
            "agente",
            "worker",
            "subagent",
            "subagente",
            "evaluator",
            "evaluador",
            "capability",
            "capacidad",
            "planner",
            "analyst",
            "designer",
        )
    ]

    required_checks = {
        "agent_orchestration_has_orchestrator": (
            _contains_any(node_texts, "orchestrator", "orquestador", "supervisor", "coordinator", "coordinador"),
            "orquestador/supervisor",
        ),
        "agent_orchestration_has_multiple_agents": (len(worker_texts) >= 2, "al menos dos agentes o capacidades"),
        "agent_orchestration_has_handoffs": (
            _contains_any(edge_texts, "handoff", "handover", "delegate", "delegacion", "delega", "route", "enruta"),
            "handoffs explicitos",
        ),
        "agent_orchestration_has_tools": (
            _contains_any(all_texts, "tool", "tools", "tool_call", "herramienta", "mcp", "api", "connector", "conector"),
            "herramientas o conectores",
        ),
        "agent_orchestration_has_memory": (
            _contains_any(all_texts, "memory", "memoria", "rag", "vector", "checkpoint", "context", "contexto", "shared_state"),
            "memoria, contexto o checkpoints",
        ),
        "agent_orchestration_has_guardrails": (
            _contains_any(all_texts, "guardrail", "guardrails", "policy", "security", "seguridad", "boundary", "limite", "control"),
            "guardrails o limites de seguridad",
        ),
        "agent_orchestration_has_hitl": (
            _contains_any(all_texts, "hitl", "human", "humano", "approval", "aprobacion", "gate", "human_gate"),
            "punto HITL/aprobacion humana",
        ),
        "agent_orchestration_has_output": (
            _contains_any(all_texts, "output", "resultado", "deliverable", "entregable", "dossier", "document", "documento"),
            "salida o entregable",
        ),
        "agent_orchestration_has_fallback": (
            _contains_any(all_texts, "fallback", "retry", "reintento", "escalation", "escalamiento", "error"),
            "fallback, reintento o escalamiento",
        ),
        "agent_orchestration_has_source_refs": (
            bool(model.source_refs or any(node.source_refs for node in model.nodes) or any(edge.source_refs for edge in model.edges)),
            "referencias de origen",
        ),
    }

    missing: list[str] = []
    for check_key, (passed, label) in required_checks.items():
        checks[check_key] = passed
        if not passed:
            missing.append(label)

    if missing:
        errors.append("Orquestacion agentiva incompleta: falta " + ", ".join(missing) + ".")
    elif len(model.nodes) < 6:
        warnings.append("La orquestacion agentiva esta completa, pero podria ganar claridad con mas detalle visual.")


def _semantic_checks(model: DiagramModel, checks: dict[str, bool], warnings: list[str]) -> None:
    if model.notation == DiagramNotation.uml_use_case:
        checks["uml_use_case_has_actor"] = _has_kind(model, "actor", "person")
        checks["uml_use_case_has_use_case"] = _has_kind(model, "use_case") or len(model.nodes) >= 2
        if not checks["uml_use_case_has_actor"]:
            warnings.append("UML Use Case deberia incluir al menos un actor externo.")
        if not checks["uml_use_case_has_use_case"]:
            warnings.append("UML Use Case deberia incluir casos de uso dentro del boundary del sistema.")

    if model.notation == DiagramNotation.uml_activity:
        checks["uml_activity_has_start"] = _has_kind(model, "start", "initial")
        checks["uml_activity_has_final"] = _has_kind(model, "final", "end")
        checks["uml_activity_has_control_flow"] = bool(model.edges)
        if not checks["uml_activity_has_start"]:
            warnings.append("UML Activity deberia declarar un nodo inicial.")
        if not checks["uml_activity_has_final"]:
            warnings.append("UML Activity deberia declarar un nodo final.")
        if not checks["uml_activity_has_control_flow"]:
            warnings.append("UML Activity requiere flujos de control entre acciones.")

    if model.notation == DiagramNotation.bpmn:
        checks["bpmn_has_start_event"] = _has_kind(model, "start_event", "start")
        checks["bpmn_has_end_event"] = _has_kind(model, "end_event", "end")
        checks["bpmn_has_task_or_subprocess"] = _has_kind(model, "task", "subprocess", "process")
        lane_ids = {lane.id for pool in model.pools for lane in pool.lanes}
        pool_ids = {pool.id for pool in model.pools}
        checks["bpmn_has_dynamic_pool"] = bool(pool_ids)
        checks["bpmn_has_dynamic_lane"] = bool(lane_ids)
        checks["bpmn_nodes_have_lane_assignment"] = not model.nodes or all(
            str(node.metadata.get("lane_id") or "") in lane_ids for node in model.nodes
        )
        checks["bpmn_cross_pool_uses_message_flow"] = True
        if pool_ids and lane_ids:
            assignments = {
                node.id: (
                    str(node.metadata.get("pool_id") or ""),
                    str(node.metadata.get("lane_id") or ""),
                )
                for node in model.nodes
            }
            for edge in model.edges:
                source_pool = assignments.get(edge.source, ("", ""))[0]
                target_pool = assignments.get(edge.target, ("", ""))[0]
                if source_pool and target_pool and source_pool != target_pool and _kind(edge.kind) != "message_flow":
                    checks["bpmn_cross_pool_uses_message_flow"] = False
                    break
        if not checks["bpmn_has_start_event"]:
            warnings.append("BPMN 2.0 deberia incluir un start event.")
        if not checks["bpmn_has_end_event"]:
            warnings.append("BPMN 2.0 deberia incluir un end event.")
        if not checks["bpmn_has_task_or_subprocess"]:
            warnings.append("BPMN 2.0 deberia incluir tareas o subprocesos, no solo cajas genericas.")
        if not checks["bpmn_has_dynamic_pool"]:
            warnings.append("BPMN 2.0 deberia declarar pools inferidos desde participantes, areas o sistemas.")
        if not checks["bpmn_has_dynamic_lane"]:
            warnings.append("BPMN 2.0 deberia declarar lanes inferidas desde roles o responsabilidades.")
        if not checks["bpmn_nodes_have_lane_assignment"]:
            warnings.append("Cada nodo BPMN deberia declarar metadata.pool_id y metadata.lane_id.")
        if not checks["bpmn_cross_pool_uses_message_flow"]:
            warnings.append("Los flujos BPMN entre pools deberian ser message_flow.")

    if model.notation == DiagramNotation.uml_component:
        checks["uml_component_has_component"] = _has_kind(model, "component")
        checks["uml_component_has_dependency"] = bool(model.edges)
        if not checks["uml_component_has_component"]:
            warnings.append("UML Component deberia usar elementos de tipo component/interface.")
        if not checks["uml_component_has_dependency"]:
            warnings.append("UML Component deberia mostrar dependencias o interfaces relevantes.")

    if model.notation == DiagramNotation.c4:
        kinds = {_kind(node.kind) for node in model.nodes}
        checks["c4_has_system_or_person"] = any("system" in kind or "person" in kind or "actor" in kind for kind in kinds)
        checks["c4_no_class_mix"] = not any("class" in kind or "attribute" in kind for kind in kinds)
        if not checks["c4_has_system_or_person"]:
            warnings.append("C4 deberia modelar personas, sistemas, contenedores o boundaries segun el nivel.")
        if not checks["c4_no_class_mix"]:
            warnings.append("C4 no debe mezclar clases/atributos UML de bajo nivel.")

    if any(token in (model.diagram_key or "").lower() for token in ("agent", "orchestration", "memory", "rag", "tool")):
        checks["agentic_has_taxonomy_diversity"] = _has_kind(model, "orchestrator", "worker", "evaluator", "vector", "memory", "mcp", "gate", "guardrail", "tool")
        if not checks["agentic_has_taxonomy_diversity"]:
            warnings.append("El diagrama agentico deberia diferenciar visualmente orquestadores, memorias RAG y herramientas MCP.")


def evaluate_diagram_quality(model: DiagramModel) -> DiagramQualityReport:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    checks["has_nodes"] = bool(model.nodes)
    if not model.nodes:
        errors.append("El diagrama no contiene nodos.")

    checks["has_source_refs"] = bool(model.source_refs or any(node.source_refs for node in model.nodes))
    if not checks["has_source_refs"]:
        warnings.append("El diagrama no declara referencias de origen.")

    normalized_labels = [node.label.strip().lower() for node in model.nodes]
    duplicate_labels = [label for label, count in Counter(normalized_labels).items() if count > 1]
    checks["labels_are_distinct"] = not duplicate_labels
    if duplicate_labels:
        warnings.append("Hay etiquetas de nodo duplicadas que reducen la legibilidad.")

    serialized = model.model_dump_json().lower()
    leaked_tokens = [token for token in FORBIDDEN_CONTENT if token.lower() in serialized]
    checks["contains_no_secrets"] = not leaked_tokens
    if leaked_tokens:
        errors.append("El contenido contiene patrones que podrian corresponder a secretos.")

    node_ids = {node.id for node in model.nodes}
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in model.edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    visited: set[str] = set()
    if node_ids:
        queue: deque[str] = deque([next(iter(node_ids))])
        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            queue.extend(adjacency[node_id] - visited)
    checks["graph_is_connected"] = not node_ids or visited == node_ids
    if not checks["graph_is_connected"]:
        warnings.append("Existen nodos desconectados; valide si son intencionales.")

    checks["reasonable_density"] = len(model.nodes) <= 40 and len(model.edges) <= 80
    if not checks["reasonable_density"]:
        warnings.append("El diagrama supera la densidad recomendada y deberia dividirse por nivel.")

    layout_metrics = analyze_diagram_complexity(model)
    checks["layout_risk_acceptable"] = layout_metrics.estimated_crossing_risk not in {
        DiagramLayoutRisk.high,
        DiagramLayoutRisk.critical,
    }
    checks["layout_split_not_required"] = not layout_metrics.split_recommended
    checks["layout_has_adaptive_lanes"] = model.notation != DiagramNotation.bpmn or layout_metrics.lane_count > 0
    if not checks["layout_risk_acceptable"]:
        warnings.append(
            "El diagrama tiene alta densidad visual; requiere layout adaptativo, routing o split para lectura humana."
        )
    if not checks["layout_split_not_required"]:
        split = recommend_diagram_split(model)
        suffix = f" Sugerencias: {', '.join(split.suggested_chunks)}." if split.suggested_chunks else ""
        warnings.append(f"El diagrama deberia dividirse en vistas por nivel o responsabilidad.{suffix}")

    _semantic_checks(model, checks, warnings)
    _agent_orchestration_checks(model, checks, warnings, errors)

    score = 100 - (30 * len(errors)) - (8 * len(warnings))
    if _kind(model.diagram_key) == "agent_orchestration" and not errors and score < 90:
        errors.append("Orquestacion agentiva no alcanzo el score minimo 90 para visualizacion principal.")
        score = 100 - (30 * len(errors)) - (8 * len(warnings))
    return DiagramQualityReport(valid=not errors, score=max(0, score), errors=errors, warnings=warnings, checks=checks)
