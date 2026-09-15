from __future__ import annotations

from pathlib import Path

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.quality_service import evaluate_diagram_quality


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_dlg8_visual_quality_gate_warns_about_dense_layout() -> None:
    model = _load_model("dense_generic_agentic_graph.json")

    report = evaluate_diagram_quality(model)

    assert report.valid is True
    assert report.checks["layout_risk_acceptable"] is False
    assert any("alta densidad visual" in warning for warning in report.warnings)


def test_dlg8_visual_quality_gate_accepts_small_readable_diagram() -> None:
    model = DiagramModel(
        diagram_key="small_readable",
        title="Small readable",
        notation="flowchart",
        nodes=[
            {"id": "start", "label": "Inicio", "kind": "start"},
            {"id": "done", "label": "Fin", "kind": "end"},
        ],
        edges=[{"id": "e1", "source": "start", "target": "done"}],
        source_refs=["test"],
    )

    report = evaluate_diagram_quality(model)

    assert report.valid is True
    assert report.checks["layout_risk_acceptable"] is True
    assert report.checks["layout_split_not_required"] is True


def test_agent_orchestration_quality_gate_rejects_incomplete_models() -> None:
    model = DiagramModel(
        diagram_key="agent_orchestration",
        title="Orquestacion agentiva",
        notation="flowchart",
        nodes=[
            {"id": "orchestrator", "label": "Orquestador principal", "kind": "orchestrator", "source_refs": ["design:1"]},
            {"id": "agent_1", "label": "Agente de analisis", "kind": "worker", "source_refs": ["design:1"]},
        ],
        edges=[
            {"id": "handoff_1", "source": "orchestrator", "target": "agent_1", "kind": "handoff", "source_refs": ["design:1"]},
        ],
        source_refs=["design:1"],
    )

    report = evaluate_diagram_quality(model)

    assert report.valid is False
    assert report.checks["agent_orchestration_has_orchestrator"] is True
    assert report.checks["agent_orchestration_has_multiple_agents"] is False
    assert report.checks["agent_orchestration_has_memory"] is False
    assert any("Orquestacion agentiva incompleta" in error for error in report.errors)


def test_agent_orchestration_quality_gate_accepts_complete_models() -> None:
    model = DiagramModel(
        diagram_key="agent_orchestration",
        title="Orquestacion agentiva",
        notation="flowchart",
        nodes=[
            {
                "id": "orchestrator",
                "label": "Orquestador principal",
                "kind": "orchestrator",
                "agent_kind": "orchestrator",
                "description": "Planifica, enruta y coordina handoffs.",
                "source_refs": ["design:1"],
            },
            {
                "id": "analysis_agent",
                "label": "Agente de analisis",
                "kind": "agent",
                "agent_kind": "worker",
                "source_refs": ["design:1"],
            },
            {
                "id": "design_agent",
                "label": "Agente de diseno",
                "kind": "agent",
                "agent_kind": "worker",
                "source_refs": ["design:1"],
            },
            {
                "id": "memory",
                "label": "Memoria RAG y checkpoints",
                "kind": "memory",
                "memory_kind": "vector_store",
                "source_refs": ["memory:1"],
            },
            {
                "id": "tools",
                "label": "Herramientas MCP y APIs",
                "kind": "tool",
                "tool_kind": "mcp_server",
                "source_refs": ["tools:1"],
            },
            {
                "id": "guardrail",
                "label": "Guardrails y aprobacion HITL",
                "kind": "guardrail_gate",
                "tool_kind": "guardrail_gate",
                "source_refs": ["design:2"],
            },
            {
                "id": "fallback",
                "label": "Fallback y escalamiento",
                "kind": "fallback",
                "source_refs": ["design:2"],
            },
            {
                "id": "output",
                "label": "Resultado entregable",
                "kind": "output",
                "source_refs": ["estimate:1"],
            },
        ],
        edges=[
            {"id": "e1", "source": "orchestrator", "target": "analysis_agent", "kind": "handoff", "label": "delega analisis", "source_refs": ["design:1"]},
            {"id": "e2", "source": "analysis_agent", "target": "design_agent", "kind": "handoff", "label": "transfiere contexto", "source_refs": ["design:1"]},
            {"id": "e3", "source": "design_agent", "target": "memory", "kind": "checkpoint_resume", "label": "actualiza memoria", "source_refs": ["memory:1"]},
            {"id": "e4", "source": "design_agent", "target": "tools", "kind": "tool_call", "label": "usa herramientas", "source_refs": ["tools:1"]},
            {"id": "e5", "source": "tools", "target": "guardrail", "kind": "decision", "label": "control HITL", "source_refs": ["design:2"]},
            {"id": "e6", "source": "guardrail", "target": "fallback", "kind": "escalation", "label": "escala error", "source_refs": ["design:2"]},
            {"id": "e7", "source": "fallback", "target": "output", "kind": "retry", "label": "reintento controlado", "source_refs": ["estimate:1"]},
        ],
        source_refs=["design:1", "memory:1", "tools:1", "estimate:1"],
    )

    report = evaluate_diagram_quality(model)

    assert report.valid is True
    assert report.score >= 90
    assert report.checks["agent_orchestration_has_orchestrator"] is True
    assert report.checks["agent_orchestration_has_multiple_agents"] is True
    assert report.checks["agent_orchestration_has_handoffs"] is True
    assert report.checks["agent_orchestration_has_tools"] is True
    assert report.checks["agent_orchestration_has_memory"] is True
    assert report.checks["agent_orchestration_has_guardrails"] is True
    assert report.checks["agent_orchestration_has_hitl"] is True
    assert report.checks["agent_orchestration_has_output"] is True
    assert report.checks["agent_orchestration_has_fallback"] is True
