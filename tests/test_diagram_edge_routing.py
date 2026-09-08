from __future__ import annotations

from pathlib import Path

from app.services.diagram_center.contracts import DiagramEdge, DiagramModel, DiagramNode
from app.services.diagram_center.layout_engine import compute_layered_layout, route_layered_edges
from app.services.diagram_center.layout_sizing import measure_generic_node


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _linear_activity_model(*, direction: str) -> DiagramModel:
    nodes = [
        DiagramNode(id="start", label="Inicio", kind="start"),
        DiagramNode(id="receive", label="Recibir solicitud", kind="activity"),
        DiagramNode(id="end", label="Fin", kind="end"),
    ]
    edges = [
        DiagramEdge(id="e1", source="start", target="receive"),
        DiagramEdge(id="e2", source="receive", target="end"),
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


def test_dlg5_routes_use_orthogonal_control_points() -> None:
    model = _load_model("dense_generic_agentic_graph.json")
    sizes = {node.id: measure_generic_node(node, model.notation) for node in model.nodes}
    layout = compute_layered_layout(model, sizes)

    routes = route_layered_edges(model, layout.positions, sizes)

    assert len(routes) == len(model.edges)
    assert all(len(route.points) == 4 for route in routes.values())
    assert all(route.label_position for route in routes.values())


def test_dlg5_routes_stay_connected_to_node_boundaries() -> None:
    model = _load_model("dense_uml_activity.json")
    sizes = {node.id: measure_generic_node(node, model.notation) for node in model.nodes}
    layout = compute_layered_layout(model, sizes)
    routes = route_layered_edges(model, layout.positions, sizes)

    first_edge = model.edges[0]
    route = routes[first_edge.id]
    source_x, source_y = layout.positions[first_edge.source]
    source_size = sizes[first_edge.source]

    assert route.points[0][0] in {source_x, source_x + source_size.width}
    assert route.points[0][1] == source_y + source_size.height / 2


def test_dlg5_routes_top_bottom_edges_from_vertical_node_boundaries() -> None:
    model = _linear_activity_model(direction="TB")
    sizes = {node.id: measure_generic_node(node, model.notation) for node in model.nodes}
    layout = compute_layered_layout(model, sizes, min_width=360)
    routes = route_layered_edges(model, layout.positions, sizes)

    route = routes["e1"]
    source_x, source_y = layout.positions["start"]
    source_size = sizes["start"]
    target_x, target_y = layout.positions["receive"]
    target_size = sizes["receive"]

    assert route.points[0] == (source_x + source_size.width / 2, source_y + source_size.height)
    assert route.points[-1] == (target_x + target_size.width / 2, target_y)
