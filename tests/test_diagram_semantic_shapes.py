from __future__ import annotations

from app.services.diagram_center.contracts import DiagramEdge, DiagramModel, DiagramNode
from app.services.diagram_center.renderer_service import render_mermaid, render_svg


def test_agentic_workflow_renders_gates_as_diamonds_and_logs_as_storage() -> None:
    model = DiagramModel(
        diagram_key="memory_rag_architecture",
        title="Arquitectura de memoria y RAG",
        notation="flowchart",
        direction="LR",
        nodes=[
            DiagramNode(id="decision_promotion", label="Promocion/aprobacion de decision", kind="process"),
            DiagramNode(id="approval_gate", label="Approval Gate", kind="process"),
            DiagramNode(id="checkpoint", label="Checkpoint de salida", kind="process"),
            DiagramNode(id="decision_log", label="Decision Log", kind="store"),
            DiagramNode(id="approval_audit_log", label="Approval Audit Log", kind="store"),
            DiagramNode(id="checkpoint_log", label="Checkpoint Log", kind="store"),
        ],
        edges=[
            DiagramEdge(id="e1", source="decision_promotion", target="approval_gate", label="solicitar aprobacion"),
            DiagramEdge(id="e2", source="approval_gate", target="approval_audit_log", label="audit"),
            DiagramEdge(id="e3", source="decision_promotion", target="decision_log", label="persistir decision"),
            DiagramEdge(id="e4", source="checkpoint", target="checkpoint_log", label="checkpoint"),
        ],
        source_refs=["test:semantic-shapes"],
    )

    svg = render_svg(model)
    mermaid = render_mermaid(model)

    assert 'data-node-id="decision_promotion" data-node-kind="process" data-node-shape="decision"' in svg
    assert 'data-node-id="approval_gate" data-node-kind="process" data-node-shape="decision"' in svg
    assert 'data-node-id="checkpoint" data-node-kind="process" data-node-shape="decision"' in svg
    assert 'data-node-id="decision_log" data-node-kind="store" data-node-shape="storage"' in svg
    assert 'data-node-id="approval_audit_log" data-node-kind="store" data-node-shape="storage"' in svg
    assert 'data-node-id="checkpoint_log" data-node-kind="store" data-node-shape="storage"' in svg
    assert svg.count('data-node-shape="decision"') == 3
    assert svg.count('data-node-shape="storage"') == 3
    assert "decision_promotion{Promocion/aprobacion de decision}" in mermaid
    assert "approval_gate{Approval Gate}" in mermaid
    assert 'decision_log[("Decision Log")]' in mermaid
    assert 'approval_audit_log[("Approval Audit Log")]' in mermaid
