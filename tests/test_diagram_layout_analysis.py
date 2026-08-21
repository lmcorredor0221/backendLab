from __future__ import annotations

from pathlib import Path

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.layout_analysis import analyze_diagram_complexity
from app.services.diagram_center.layout_contracts import DiagramLayoutRisk


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_dlg2_complexity_analyzer_scores_dense_generic_diagram() -> None:
    model = _load_model("dense_generic_agentic_graph.json")

    metrics = analyze_diagram_complexity(model)

    assert metrics.node_count == 16
    assert metrics.edge_count == 20
    assert metrics.edge_density == 1.25
    assert metrics.max_degree >= 3
    assert metrics.estimated_crossing_risk in {DiagramLayoutRisk.high, DiagramLayoutRisk.critical}


def test_dlg2_complexity_analyzer_detects_bpmn_lanes_and_risk() -> None:
    model = _load_model("dense_bpmn_process.json")

    metrics = analyze_diagram_complexity(model)

    assert metrics.pool_count == 1
    assert metrics.lane_count == 3
    assert metrics.node_count == 15
    assert metrics.estimated_crossing_risk in {DiagramLayoutRisk.high, DiagramLayoutRisk.critical}


def test_dlg2_complexity_analyzer_keeps_small_diagrams_low_risk() -> None:
    model = DiagramModel(
        diagram_key="small",
        title="Small",
        notation="flowchart",
        nodes=[
            {"id": "a", "label": "A", "kind": "start"},
            {"id": "b", "label": "B", "kind": "process"},
            {"id": "c", "label": "C", "kind": "end"},
        ],
        edges=[
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "b", "target": "c"},
        ],
    )

    metrics = analyze_diagram_complexity(model)

    assert metrics.estimated_crossing_risk == DiagramLayoutRisk.low
    assert metrics.split_recommended is False
