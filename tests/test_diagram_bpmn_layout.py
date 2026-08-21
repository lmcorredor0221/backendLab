from __future__ import annotations

from pathlib import Path
import re

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.renderer_service import render_svg


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_dlg6_bpmn_renderer_uses_adaptive_width_and_lanes() -> None:
    model = _load_model("dense_bpmn_process.json")

    svg = render_svg(model)

    width = int(re.search(r'width="(\d+)"', svg).group(1))  # type: ignore[union-attr]
    assert width > 1180
    assert svg.count('data-bpmn-kind="lane"') == 3
    assert 'data-node-kind="exclusive_gateway"' in svg
    assert 'data-edge-kind="sequence_flow"' in svg
