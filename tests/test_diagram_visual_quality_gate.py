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
