from __future__ import annotations

from pathlib import Path

from app.services.diagram_center.contracts import DiagramEdge, DiagramModel, DiagramNode
from app.services.diagram_center.layout_engine import compute_layered_layout
from app.services.diagram_center.layout_sizing import measure_generic_node


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _linear_activity_model(*, direction: str) -> DiagramModel:
    nodes = [
        DiagramNode(id="start", label="Inicio", kind="start"),
        DiagramNode(id="receive", label="Recibir solicitud", kind="activity"),
        DiagramNode(id="classify", label="Clasificar intencion", kind="activity"),
        DiagramNode(id="decide", label="Requiere aprobacion", kind="decision"),
        DiagramNode(id="execute", label="Ejecutar accion", kind="activity"),
        DiagramNode(id="end", label="Fin", kind="end"),
    ]
    edges = [
        DiagramEdge(id="e1", source="start", target="receive"),
        DiagramEdge(id="e2", source="receive", target="classify"),
        DiagramEdge(id="e3", source="classify", target="decide"),
        DiagramEdge(id="e4", source="decide", target="execute"),
        DiagramEdge(id="e5", source="execute", target="end"),
    ]
    return DiagramModel(
        diagram_key="linear_activity",
        title="Actividad vertical",
        notation="uml_activity",
        direction=direction,
        nodes=nodes,
        edges=edges,
        source_refs=["test:linear-activity"],
    )


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


def test_dlg4_layered_layout_honors_top_bottom_direction() -> None:
    model = _linear_activity_model(direction="TB")
    sizes = {node.id: measure_generic_node(node, model.notation) for node in model.nodes}

    layout = compute_layered_layout(model, sizes, min_width=360)

    assert layout.height > layout.width
    assert layout.positions["start"][1] < layout.positions["receive"][1]
    assert layout.positions["receive"][1] < layout.positions["classify"][1]
    assert layout.positions["classify"][1] < layout.positions["decide"][1]
