from __future__ import annotations

from app.services.artifact_diagram_taxonomy import (
    ALLOWED_ARTIFACT_CATEGORIES,
    ALLOWED_DIAGRAM_CATEGORIES,
    get_artifact_taxonomy_by_key,
    get_artifact_taxonomy_entries,
    get_diagram_taxonomy_by_key,
    get_diagram_taxonomy_entries,
    load_artifact_diagram_taxonomy,
    validate_artifact_diagram_taxonomy,
)


def test_taxonomy_manifest_loads_and_declares_required_coverage() -> None:
    manifest = load_artifact_diagram_taxonomy()

    assert manifest["schema_version"] == "artifact-diagram-taxonomy.v1"
    assert set(manifest["artifact_categories"]) == ALLOWED_ARTIFACT_CATEGORIES
    assert set(manifest["diagram_categories"]) == ALLOWED_DIAGRAM_CATEGORIES
    assert get_artifact_taxonomy_entries()
    assert len(get_diagram_taxonomy_entries()) == 24
    assert any(item["default_generation_state"] == "not_generated" for item in get_diagram_taxonomy_entries())


def test_every_diagram_has_access_policy_and_existing_sources() -> None:
    artifact_by_key = get_artifact_taxonomy_by_key()

    for diagram in get_diagram_taxonomy_entries():
      assert diagram["required_tier"] in {"blueprint", "blueprint_pro", "acp"}
      assert diagram["access_level"] in {"sample", "view_only", "downloadable", "premium", "restricted"}
      assert diagram["enabled_from_stage"] in {
          "discover",
          "define",
          "design",
          "tools",
          "memory",
          "estimate",
          "validate",
          "package",
      }
      assert diagram["source_artifact_keys"]
      for source_key in diagram["source_artifact_keys"]:
          assert source_key in artifact_by_key


def test_agent_diagrams_do_not_expose_internal_lean_model() -> None:
    for diagram in get_diagram_taxonomy_entries():
        if diagram["diagram_surface"] in {"agent_design", "agent_runtime", "implementation_guide"}:
            assert diagram["contains_internal_lean_model"] is False

        if diagram["contains_internal_lean_model"]:
            assert diagram["diagram_surface"] == "builder_provenance"
            assert diagram["category"] == "lineage"
            assert diagram["access_level"] == "restricted"


def test_product_download_boundaries_are_consistent() -> None:
    for artifact in get_artifact_taxonomy_entries():
        if artifact["product_owner"] == "acp":
            assert artifact["blueprint_download"] is False
            assert artifact["acp_download"] is True

        if artifact["internal_lean_refs_allowed"]:
            assert artifact["portable_scope"] == "producer_lineage"


def test_validator_reports_duplicate_and_unknown_references() -> None:
    manifest = load_artifact_diagram_taxonomy()
    mutated = {
        **manifest,
        "artifact_entries": [
            *manifest["artifact_entries"],
            manifest["artifact_entries"][0],
        ],
        "diagram_entries": [
            {
                **manifest["diagram_entries"][0],
                "diagram_key": "broken_reference",
                "source_artifact_keys": ["missing.artifact"],
            }
        ],
    }

    errors = validate_artifact_diagram_taxonomy(mutated)

    assert any("duplicate artifact_key" in error for error in errors)
    assert any("unknown source_artifact_key missing.artifact" in error for error in errors)
    assert not get_diagram_taxonomy_by_key()["architecture_overview"]["contains_internal_lean_model"]
