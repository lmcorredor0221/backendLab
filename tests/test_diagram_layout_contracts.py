from __future__ import annotations

import json

from app.services.diagram_center.layout_contracts import (
    DiagramLayoutEdgeRoute,
    DiagramLayoutMetrics,
    DiagramLayoutNodeBox,
    DiagramLayoutPlan,
    DiagramLayoutPoint,
    DiagramLayoutStrategy,
    DiagramLayoutViewport,
)
from app.services.shared_specs import resolve_shared_spec_path


SCHEMA_PATH = resolve_shared_spec_path("schemas", "diagram-layout-plan.v1.schema.json")


def test_dlg1_layout_plan_contract_round_trips() -> None:
    plan = DiagramLayoutPlan(
        diagram_key="current_process_map",
        notation="bpmn",
        strategy=DiagramLayoutStrategy.bpmn_swimlane,
        metrics=DiagramLayoutMetrics(
            node_count=2,
            edge_count=1,
            edge_density=0.5,
            max_degree=1,
            label_avg_chars=14.5,
            lane_count=2,
            pool_count=1,
        ),
        viewport=DiagramLayoutViewport(width=1280, height=720),
        nodes=[
            DiagramLayoutNodeBox(node_id="start", x=80, y=120, width=72, height=72, kind="start_event", layer=0),
            DiagramLayoutNodeBox(node_id="task", x=240, y=120, width=280, height=92, kind="task", layer=1),
        ],
        edges=[
            DiagramLayoutEdgeRoute(
                edge_id="e1",
                source="start",
                target="task",
                points=[DiagramLayoutPoint(x=152, y=156), DiagramLayoutPoint(x=240, y=166)],
                label="continua",
                label_position=DiagramLayoutPoint(x=196, y=138),
            )
        ],
        renderer_revision="diagram-renderer.vNext",
    )

    payload = plan.model_dump(mode="json")
    assert payload["schema_version"] == "diagram-layout-plan.v1"
    assert DiagramLayoutPlan.model_validate(payload).diagram_key == "current_process_map"


def test_dlg1_shared_schema_exists_and_matches_contract_name() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"]["const"] == "diagram-layout-plan.v1"
    assert "bpmn_swimlane" in schema["properties"]["strategy"]["enum"]
    assert "nodes" in schema["required"]
    assert "edges" in schema["required"]
