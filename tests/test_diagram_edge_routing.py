from __future__ import annotations

from pathlib import Path

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.layout_engine import compute_layered_layout, route_layered_edges
from app.services.diagram_center.layout_sizing import measure_generic_node


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


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
