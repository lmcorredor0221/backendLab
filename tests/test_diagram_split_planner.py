from __future__ import annotations

from pathlib import Path

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.layout_split import recommend_diagram_split


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_dlg9_split_planner_recommends_chunks_for_dense_bpmn() -> None:
    model = _load_model("dense_bpmn_process.json")

    recommendation = recommend_diagram_split(model)

    assert recommendation.recommended is True
    assert "Riesgo visual" in recommendation.reason
    assert any(chunk.startswith("lane:") for chunk in recommendation.suggested_chunks)


def test_dlg9_split_planner_skips_small_diagrams() -> None:
    model = DiagramModel(
        diagram_key="small",
        title="Small",
        notation="flowchart",
        nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        edges=[{"id": "e1", "source": "a", "target": "b"}],
    )

    recommendation = recommend_diagram_split(model)

    assert recommendation.recommended is False
    assert recommendation.suggested_chunks == []
