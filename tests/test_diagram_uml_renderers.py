from __future__ import annotations

from pathlib import Path

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.renderer_service import render_svg


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_dlg7_activity_renderer_uses_uml_activity_semantics() -> None:
    model = _load_model("dense_uml_activity.json")

    svg = render_svg(model)

    assert "UML ACTIVITY" in svg
    assert 'data-diagram-notation="uml_activity"' in svg
    assert "<polygon" in svg
    assert 'data-edge-kind="relationship"' in svg
    assert "uml-activity-arrow" in svg


def test_dlg7_use_case_renderer_keeps_actor_and_use_case_semantics() -> None:
    model = _load_model("dense_uml_use_case.json")

    svg = render_svg(model)

    assert 'data-node-kind="actor"' in svg
    assert 'data-node-kind="use_case"' in svg
    assert "UML USE CASE" in svg
