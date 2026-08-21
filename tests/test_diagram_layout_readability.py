from __future__ import annotations

from pathlib import Path
import re

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.renderer_service import render_svg


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _svg_size(svg: str) -> tuple[int, int]:
    width = re.search(r'width="(\d+)"', svg)
    height = re.search(r'height="(\d+)"', svg)
    assert width and height
    return int(width.group(1)), int(height.group(1))


def _baseline_metrics(model: DiagramModel) -> dict[str, float | int | str]:
    svg = render_svg(model)
    width, height = _svg_size(svg)
    direct_line_edges = len(re.findall(r'<path[^>]+data-edge-id="[^"]+"', svg))
    rendered_paths = svg.count("<path")
    return {
        "diagram_key": model.diagram_key,
        "notation": model.notation.value,
        "node_count": len(model.nodes),
        "edge_count": len(model.edges),
        "edge_density": round(len(model.edges) / max(len(model.nodes), 1), 2),
        "svg_width": width,
        "svg_height": height,
        "rendered_paths": rendered_paths,
        "direct_line_edges": direct_line_edges,
    }


def test_dlg0_dense_fixtures_render_and_reproduce_readability_risk() -> None:
    fixture_names = [
        "dense_generic_agentic_graph.json",
        "dense_bpmn_process.json",
        "dense_uml_use_case.json",
        "dense_uml_activity.json",
    ]

    metrics = [_baseline_metrics(_load_model(name)) for name in fixture_names]

    assert {item["diagram_key"] for item in metrics} == {
        "dense_agentic_architecture",
        "dense_bpmn_process",
        "dense_use_case",
        "dense_activity",
    }
    assert all(item["node_count"] >= 12 for item in metrics)
    assert all(item["edge_density"] >= 1.0 for item in metrics)
    assert any(item["direct_line_edges"] >= 15 for item in metrics)
    assert any(item["svg_width"] == 1120 for item in metrics)


def test_dlg0_baseline_identifies_fixed_canvas_and_dense_edges() -> None:
    generic_model = _load_model("dense_generic_agentic_graph.json")
    activity_model = _load_model("dense_uml_activity.json")

    generic_metrics = _baseline_metrics(generic_model)
    activity_metrics = _baseline_metrics(activity_model)

    assert generic_metrics["svg_width"] >= 1120
    assert activity_metrics["svg_width"] >= 1120
    assert generic_metrics["direct_line_edges"] == len(generic_model.edges)
    assert activity_metrics["direct_line_edges"] == len(activity_model.edges)
    assert generic_metrics["edge_density"] > 1.2
    assert activity_metrics["edge_density"] > 1.1
