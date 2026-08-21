from copy import deepcopy

from app.services.deliverable_catalog import load_seed_deliverable_catalog, validate_deliverable_catalog


def test_seed_deliverable_catalog_loads_and_contains_artifact_and_diagram_examples() -> None:
    catalog = load_seed_deliverable_catalog()
    keys = {entry.deliverable_key for entry in catalog.entries}

    assert catalog.schema_version == "deliverable-catalog.v1"
    assert "discovery.problem_context_brief" in keys
    assert "diagram.problem_context_map" in keys
    assert "diagram.stakeholder_map" in keys
    assert "diagram.current_process_map" in keys
    assert "diagram.traceability_matrix" in keys
    assert "definition.requirements_brief" in keys
    assert any(entry.deliverable_type == "artifact" for entry in catalog.entries)
    assert any(entry.deliverable_type == "diagram" for entry in catalog.entries)


def test_deliverable_catalog_rejects_duplicate_keys() -> None:
    payload = load_seed_deliverable_catalog().model_dump(mode="json")
    payload["entries"].append(deepcopy(payload["entries"][0]))

    errors = validate_deliverable_catalog(payload)

    assert any("duplicate deliverable keys" in error for error in errors)


def test_llm_deliverable_requires_prompt_schema_validator_and_fallback() -> None:
    payload = load_seed_deliverable_catalog().model_dump(mode="json")
    payload["entries"][0]["prompt_policy"]["prompt_template_key"] = ""

    errors = validate_deliverable_catalog(payload)

    assert any("LLM generation requires prompt policy fields" in error for error in errors)


def test_exportable_deliverable_requires_canonical_or_portable_paths() -> None:
    payload = load_seed_deliverable_catalog().model_dump(mode="json")
    payload["entries"][0]["canonical_paths"] = []
    payload["entries"][0]["portable_paths"] = []

    errors = validate_deliverable_catalog(payload)

    assert any("exportable deliverables require canonical_paths or portable_paths" in error for error in errors)
