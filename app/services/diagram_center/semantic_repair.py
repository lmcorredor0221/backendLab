from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.services.diagram_center.contracts import DiagramNotation, StructuredDiagramModel


def _kind(value: str) -> str:
    return str(value or "").lower().replace("-", "_").replace(" ", "_")


def _has_kind(nodes: list[dict[str, Any]], *tokens: str) -> bool:
    return any(any(token in _kind(str(node.get("kind") or "")) for token in tokens) for node in nodes)


def _is_terminal_event(node: dict[str, Any]) -> bool:
    kind = _kind(str(node.get("kind") or ""))
    return "start" in kind or "end" in kind


def _normalized_label(node: dict[str, Any]) -> str:
    return " ".join(str(node.get("label") or "").lower().split()).strip()


def _promote_existing_terminal_candidate(
    nodes: list[dict[str, Any]],
    *,
    kind: str,
    label_tokens: set[str],
    reverse: bool = False,
) -> dict[str, Any] | None:
    iterable = reversed(nodes) if reverse else nodes
    for node in iterable:
        if _is_terminal_event(node):
            continue
        if _normalized_label(node) not in label_tokens:
            continue
        node["kind"] = kind
        return node
    return None


def _unique_identifier(prefix: str, existing_ids: set[str]) -> str:
    candidate = prefix
    suffix = 1
    while candidate in existing_ids:
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    existing_ids.add(candidate)
    return candidate


def _node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    return {
        "pool_id": str(metadata.get("pool_id") or ""),
        "lane_id": str(metadata.get("lane_id") or ""),
        "attributes": list(metadata.get("attributes") or []),
    }


def _source_refs(node: dict[str, Any], diagram_source_refs: list[str]) -> list[str]:
    node_refs = node.get("source_refs")
    if isinstance(node_refs, list) and node_refs:
        return [str(item) for item in node_refs if str(item).strip()]
    return list(diagram_source_refs)


def _select_entry_anchor(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [node for node in nodes if not _is_terminal_event(node)]
    if not candidates:
        return nodes[0] if nodes else None
    incoming: dict[str, int] = {str(node.get("id") or ""): 0 for node in candidates}
    for edge in edges:
        target = str(edge.get("target") or "")
        if target in incoming:
            incoming[target] += 1
    for node in candidates:
        if incoming.get(str(node.get("id") or ""), 0) == 0:
            return node
    return candidates[0]


def _select_exit_anchor(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [node for node in nodes if not _is_terminal_event(node)]
    if not candidates:
        return nodes[-1] if nodes else None
    outgoing: dict[str, int] = {str(node.get("id") or ""): 0 for node in candidates}
    for edge in edges:
        source = str(edge.get("source") or "")
        if source in outgoing:
            outgoing[source] += 1
    for node in reversed(candidates):
        if outgoing.get(str(node.get("id") or ""), 0) == 0:
            return node
    return candidates[-1]


def repair_structured_diagram_model(model: StructuredDiagramModel) -> tuple[StructuredDiagramModel, list[str]]:
    if model.notation != DiagramNotation.bpmn or not model.nodes:
        return model, []

    payload = model.model_dump(mode="json")
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not nodes:
        return model, []

    node_ids = {str(node.get("id") or "") for node in nodes if isinstance(node, dict)}
    edge_ids = {str(edge.get("id") or "") for edge in edges if isinstance(edge, dict)}
    repairs: list[str] = []
    diagram_source_refs = [str(item) for item in payload.get("source_refs", []) if str(item).strip()]

    if not _has_kind(nodes, "start_event", "start"):
        promoted_start = _promote_existing_terminal_candidate(
            nodes,
            kind="start_event",
            label_tokens={"inicio", "start", "inicio del proceso"},
        )
        if promoted_start is not None:
            repairs.append("Se promovio un nodo existente a start_event BPMN para completar la semantica del diagrama.")
        else:
            entry_anchor = _select_entry_anchor(nodes, edges)
            if entry_anchor is None:
                entry_anchor = nodes[0] if nodes else None
        if promoted_start is None and entry_anchor is not None:
            start_id = _unique_identifier("bpmn_start_event", node_ids)
            start_edge_id = _unique_identifier("bpmn_flow_start", edge_ids)
            start_refs = _source_refs(entry_anchor, diagram_source_refs)
            nodes.insert(
                0,
                {
                    "id": start_id,
                    "label": "Inicio",
                    "kind": "start_event",
                    "metadata": _node_metadata(entry_anchor),
                    "source_refs": start_refs,
                },
            )
            edges.insert(
                0,
                {
                    "id": start_edge_id,
                    "source": start_id,
                    "target": str(entry_anchor.get("id") or ""),
                    "kind": "sequence_flow",
                    "label": "",
                    "source_refs": start_refs,
                },
            )
            repairs.append("Se agrego un start event BPMN minimo para completar la semantica del diagrama.")

    if not _has_kind(nodes, "end_event", "end"):
        promoted_end = _promote_existing_terminal_candidate(
            nodes,
            kind="end_event",
            label_tokens={"fin", "end", "fin del proceso"},
            reverse=True,
        )
        if promoted_end is not None:
            repairs.append("Se promovio un nodo existente a end_event BPMN para completar la semantica del diagrama.")
        else:
            exit_anchor = _select_exit_anchor(nodes, edges)
            if exit_anchor is None:
                exit_anchor = nodes[-1] if nodes else None
        if promoted_end is None and exit_anchor is not None:
            end_id = _unique_identifier("bpmn_end_event", node_ids)
            end_edge_id = _unique_identifier("bpmn_flow_end", edge_ids)
            end_refs = _source_refs(exit_anchor, diagram_source_refs)
            nodes.append(
                {
                    "id": end_id,
                    "label": "Fin",
                    "kind": "end_event",
                    "metadata": _node_metadata(exit_anchor),
                    "source_refs": end_refs,
                }
            )
            edges.append(
                {
                    "id": end_edge_id,
                    "source": str(exit_anchor.get("id") or ""),
                    "target": end_id,
                    "kind": "sequence_flow",
                    "label": "",
                    "source_refs": end_refs,
                }
            )
            repairs.append("Se agrego un end event BPMN minimo para completar la semantica del diagrama.")

    if not repairs:
        return model, []

    assumptions = payload.get("assumptions")
    if not isinstance(assumptions, list):
        assumptions = []
    for repair_note in repairs:
        if repair_note not in assumptions:
            assumptions.append(repair_note)
    payload["assumptions"] = assumptions
    return StructuredDiagramModel.model_validate(payload), repairs


def finalize_structured_diagram_artifact(
    artifact: BaseModel,
    *,
    schema_status: str,
) -> tuple[BaseModel, str]:
    if not isinstance(artifact, StructuredDiagramModel):
        return artifact, schema_status
    repaired, repairs = repair_structured_diagram_model(artifact)
    if not repairs:
        return artifact, schema_status
    base_status = schema_status.strip() or "valid"
    if base_status == "valid":
        return repaired, "repaired_bpmn_terminals"
    if "bpmn_terminals" in base_status:
        return repaired, base_status
    return repaired, f"{base_status}_bpmn_terminals"
