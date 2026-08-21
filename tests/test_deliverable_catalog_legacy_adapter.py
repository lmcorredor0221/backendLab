from app.services.artifact_diagram_taxonomy import get_artifact_taxonomy_entries, get_diagram_taxonomy_entries
from app.services.deliverable_catalog import get_registry_entry, load_deliverable_registry
from app.services.diagram_center.registry_service import load_diagram_registry


def test_unified_deliverable_registry_includes_legacy_artifacts_and_diagrams() -> None:
    registry = load_deliverable_registry()
    keys = {entry.deliverable_key for entry in registry.entries}
    artifact_keys = {entry["artifact_key"] for entry in get_artifact_taxonomy_entries()}
    diagram_keys = {f"diagram.{entry['diagram_key']}" for entry in get_diagram_taxonomy_entries()}

    assert artifact_keys <= keys
    assert diagram_keys <= keys
    assert "discovery.analysis" in keys
    assert "diagram.architecture_overview" in keys
    assert "discovery.problem_context_brief" in keys
    assert len(keys) == len(registry.entries)


def test_adapted_diagrams_are_governed_llm_supported_deliverables() -> None:
    entry = get_registry_entry("diagram.architecture_overview")

    assert entry is not None
    assert entry.deliverable_type == "diagram"
    assert entry.quality_policy.schema_contract == "diagram-model.v1"
    assert entry.prompt_policy.prompt_template_key == "deliverables.diagram.architecture_overview.v1"
    assert entry.prompt_policy.validator_key == "diagram.graph_integrity.v1"


def test_existing_diagram_center_registry_remains_unchanged() -> None:
    diagram_registry = load_diagram_registry()

    assert diagram_registry.schema_version == "diagram-registry.v1"
    assert len(diagram_registry.entries) == 33
