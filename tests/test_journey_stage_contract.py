from __future__ import annotations

import json
from pathlib import Path
import re

from app.services.journey_stage_contract import (
    CANONICAL_ARTIFACT_LIFECYCLE_STATES,
    get_journey_stage_boundary,
    journey_stage_for_source_action,
    list_journey_stage_boundaries,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
ROUTE_FILES = (
    BACKEND_ROOT / "app/api/routes/sessions.py",
    BACKEND_ROOT / "app/api/routes/session_operations.py",
    BACKEND_ROOT / "app/api/routes/session_acp.py",
)
FIXTURE_ROOT = BACKEND_ROOT / "tests/fixtures/lean_journey"
FIXTURE_MANIFEST = FIXTURE_ROOT / "manifest.json"
SOURCE_ACTION_RE = re.compile(r'source_action\s*=\s*f?"([^"]+)"')


def _extract_route_source_actions() -> tuple[str, ...]:
    discovered: set[str] = set()
    for path in ROUTE_FILES:
        discovered.update(SOURCE_ACTION_RE.findall(path.read_text(encoding="utf-8")))
    return tuple(sorted(discovered))


def test_journey_stage_contract_exposes_estimate_and_canonical_ownership_boundaries() -> None:
    boundaries = {item.stage_key: item for item in list_journey_stage_boundaries()}

    assert CANONICAL_ARTIFACT_LIFECYCLE_STATES == ("generated", "reviewed", "approved", "stale")
    assert tuple(boundaries) == (
        "discover",
        "define",
        "design",
        "tools",
        "memory",
        "validate",
        "estimate",
        "build",
    )
    assert boundaries["design"].owns_blueprint_sections == (
        "architecture",
        "reasoning_pattern",
        "safety_checks",
        "guardrails",
        "narrative",
    )
    assert boundaries["tools"].required_predecessors == ("discover", "define", "design")
    assert boundaries["tools"].owns_blueprint_sections == ("tools",)
    assert boundaries["memory"].required_predecessors == ("discover", "define", "design", "tools")
    assert boundaries["memory"].owns_blueprint_sections == (
        "memory_strategy",
        "memory_profile",
        "knowledge_profile",
    )
    assert boundaries["estimate"].required_predecessors == (
        "discover",
        "define",
        "design",
        "tools",
        "memory",
        "validate",
    )
    assert boundaries["build"].required_predecessors == (
        "discover",
        "define",
        "design",
        "tools",
        "memory",
        "validate",
        "estimate",
    )
    assert "Design no es propietario canonico de Tools ni Memory." in boundaries["design"].transition_contract


def test_journey_stage_contract_maps_current_future_and_dynamic_source_actions() -> None:
    assert journey_stage_for_source_action("build_blueprint") == ("design", "Design")
    assert journey_stage_for_source_action("enrich_blueprint") == ("design", "Design")
    assert journey_stage_for_source_action("manual_patch") == ("design", "Design")
    assert journey_stage_for_source_action("recommend_tools") == ("tools", "Tools")
    assert journey_stage_for_source_action("approve_tools_selection") == ("tools", "Tools")
    assert journey_stage_for_source_action("load_short_term_memory") == ("memory", "Memory")
    assert journey_stage_for_source_action("recommend_memory_architecture") == ("memory", "Memory")
    assert journey_stage_for_source_action("evaluate_blueprint") == ("validate", "Validate")
    assert journey_stage_for_source_action("generate_validation_scenarios") == ("validate", "Validate")
    assert journey_stage_for_source_action("run_validation_simulation") == ("validate", "Validate")
    assert journey_stage_for_source_action("judge_validation_run") == ("validate", "Validate")
    assert journey_stage_for_source_action("run_subagent:supervisor_orchestrator") == ("validate", "Validate")
    assert journey_stage_for_source_action("rerun:build_blueprint") == ("design", "Design")
    assert journey_stage_for_source_action("generate_estimation_report") == ("estimate", "Estimate")
    assert journey_stage_for_source_action("upsert_estimation_actuals") == ("estimate", "Estimate")
    assert journey_stage_for_source_action("generate_acp_preview") == ("build", "Build")
    assert journey_stage_for_source_action("export_blueprint_core") == ("build", "Build")


def test_journey_stage_contract_maps_all_declared_route_source_actions() -> None:
    unresolved = [
        source_action
        for source_action in _extract_route_source_actions()
        if journey_stage_for_source_action(source_action) is None
    ]
    assert unresolved == []


def test_ci0_lean_journey_fixtures_cover_success_missing_input_fallback_and_stale() -> None:
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    case_files = [FIXTURE_ROOT / relative_path for relative_path in manifest["cases"]]

    assert manifest["schema_version"] == "lean-journey-fixture-manifest.v1"
    assert manifest["required_scenarios"] == ["success", "missing_input", "fallback", "stale"]
    assert all(path.exists() for path in case_files)

    loaded_cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_files]
    assert sorted(item["scenario"] for item in loaded_cases) == sorted(manifest["required_scenarios"])

    for payload in loaded_cases:
        assert payload["schema_version"] == "lean-journey-fixture.v1"
        assert payload["journey_stage"] == get_journey_stage_boundary(payload["journey_stage"]).stage_key
        assert payload["session_status"] in {"ready", "needs_review", "failed"}
        assert isinstance(payload["artifact_versions"], dict) and payload["artifact_versions"]
        assert isinstance(payload["source_actions"], list) and payload["source_actions"]
        assert isinstance(payload["trace_summary"]["llm_run_count"], int)
        assert isinstance(payload["trace_summary"]["warnings_count"], int)
        unresolved = [
            source_action
            for source_action in payload["source_actions"]
            if journey_stage_for_source_action(source_action) is None
        ]
        assert unresolved == []

    by_scenario = {item["scenario"]: item for item in loaded_cases}
    assert by_scenario["missing_input"]["warnings"]
    assert by_scenario["fallback"]["warnings"]
    assert by_scenario["stale"]["stale_reasons"]


def test_journey_stage_contract_rejects_unknown_stage_key() -> None:
    try:
        get_journey_stage_boundary("unknown")
    except KeyError as exc:
        assert "Unknown journey stage boundary" in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown journey stage boundary")
