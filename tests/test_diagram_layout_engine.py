from __future__ import annotations

from pathlib import Path

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.layout_engine import compute_layered_layout
from app.services.diagram_center.layout_sizing import measure_generic_node


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_dlg4_layered_layout_uses_more_than_fixed_three_columns_for_dense_graph() -> None:
    model = _load_model("dense_generic_agentic_graph.json")
    sizes = {node.id: measure_generic_node(node, model.notation) for node in model.nodes}

    layout = compute_layered_layout(model, sizes)

    assert len(layout.positions) == len(model.nodes)
    assert max(layout.layers.values()) >= 5
    assert layout.width > 1120
    assert layout.height >= 420


def test_dlg4_layered_layout_keeps_all_nodes_inside_canvas() -> None:
    model = _load_model("dense_uml_activity.json")
    sizes = {node.id: measure_generic_node(node, model.notation) for node in model.nodes}

    layout = compute_layered_layout(model, sizes)

    for node_id, (x, y) in layout.positions.items():
        size = sizes[node_id]
        assert x >= 0
        assert y >= 0
        assert x + size.width <= layout.width
        assert y + size.height <= layout.height
