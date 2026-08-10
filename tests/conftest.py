from __future__ import annotations

from pathlib import Path

import pytest


SMOKE_TESTS = {
    "test_backend_smoke_flow_covers_login_discovery_blueprint_evaluation_and_acp_export",
}

SLOW_API_TESTS = {
    "test_acp_routes_generate_preview_validate_file_and_zip",
    "test_acp_questions_can_be_answered_and_reinjected_into_regeneration",
    "test_session_flow_exposes_enriched_artifacts_and_pending_approvals",
    "test_resolving_approval_unlocks_export_markdown",
    "test_operational_snapshot_and_monitoring_workspace_expose_stage4_state",
    "test_artifact_browser_library_filters_and_exports_persist_records",
    "test_integrations_routes_refresh_health_and_append_operational_trace",
    "test_stage5_workflow_templates_and_governance_are_exposed_and_exported",
    "test_stage5_handoffs_and_feature_flags_support_controlled_return_and_toggle",
    "test_stage5_subagents_require_flag_and_leave_specialized_trace",
    "test_rerun_skill_persists_trace_and_diff",
    "test_evaluation_workbench_bootstrap_updates_and_persists_runs",
}

SLOW_UNIT_FILES = {
    "test_evaluation_workbench.py",
    "test_skill_runtime.py",
}

SLOW_UNIT_TESTS = {
    "test_build_blueprint_exposes_rule_first_decision_report",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        file_name = Path(str(item.fspath)).name
        if file_name == "test_sessions_api.py":
            item.add_marker(pytest.mark.api)
            if item.name in SMOKE_TESTS:
                item.add_marker(pytest.mark.smoke)
            if item.name in SLOW_API_TESTS:
                item.add_marker(pytest.mark.slow)
            continue

        item.add_marker(pytest.mark.unit)
        if file_name in SLOW_UNIT_FILES or item.name in SLOW_UNIT_TESTS:
            item.add_marker(pytest.mark.slow)
