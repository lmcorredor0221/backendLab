from __future__ import annotations

from app.services.diagram_center.contracts import DiagramNotation
from app.services.diagram_center.layout_policy import layout_policy_for_notation, merge_layout_policy
from app.services.diagram_center.registry_service import build_prompt_spec, get_registry_entry


def test_layout_policy_profiles_are_notation_specific() -> None:
    bpmn = layout_policy_for_notation(DiagramNotation.bpmn)
    activity = layout_policy_for_notation(DiagramNotation.uml_activity)
    flowchart = layout_policy_for_notation(DiagramNotation.flowchart)

    assert bpmn["preferred_strategy"] == "bpmn_swimlane"
    assert activity["preferred_strategy"] == "uml_activity"
    assert flowchart["preferred_strategy"] == "layered"
    assert bpmn["max_nodes_per_view"] < flowchart["max_nodes_per_view"]
    assert bpmn["visual_quality_min_score"] >= flowchart["visual_quality_min_score"]


def test_layout_policy_override_is_sanitized_and_merged_into_prompt_spec() -> None:
    entry = get_registry_entry("current_process_map")
    assert entry is not None

    prompt = build_prompt_spec(
        entry,
        override={
            "notation": "bpmn",
            "layout_guidance": {
                "preferred_strategy": "bpmn_swimlane",
                "max_nodes_per_view": 999,
                "max_edges_per_view": 999,
                "visual_quality_min_score": 101,
            },
        },
    )

    assert prompt["layout_guidance"]["preferred_strategy"] == "bpmn_swimlane"
    assert prompt["layout_guidance"]["max_nodes_per_view"] == 80
    assert prompt["layout_guidance"]["max_edges_per_view"] == 160
    assert prompt["layout_guidance"]["visual_quality_min_score"] == 100
    assert "Estrategia de layout esperada: bpmn_swimlane." in prompt["quality_gates"]


def test_merge_layout_policy_ignores_unknown_override_keys() -> None:
    merged = merge_layout_policy(
        layout_policy_for_notation("flowchart"),
        {"max_nodes_per_view": 6, "unsafe": "ignored"},
    )

    assert merged["max_nodes_per_view"] == 6
    assert "unsafe" not in merged
