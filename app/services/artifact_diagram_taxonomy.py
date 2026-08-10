from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
TAXONOMY_MANIFEST_PATH = REPO_ROOT / "shared_specs" / "artifact-diagram-taxonomy.v1.json"

ALLOWED_PRODUCTS = {"blueprint", "acp", "shared"}
ALLOWED_DIAGRAM_PRODUCTS = {"blueprint", "acp"}
ALLOWED_ACCESS_LEVELS = {"sample", "view_only", "downloadable", "premium", "restricted"}
ALLOWED_STAGES = {"discover", "define", "design", "tools", "memory", "estimate", "validate", "package"}
ALLOWED_TIERS = {"blueprint", "blueprint_pro", "acp"}
ALLOWED_ARTIFACT_CATEGORIES = {
    "functional",
    "technical",
    "commercial",
    "diagram",
    "prompt",
    "tool",
    "memory",
    "contract",
    "estimation",
    "package",
    "lineage",
}
ALLOWED_DIAGRAM_CATEGORIES = {
    "architecture",
    "orchestration",
    "tools",
    "memory",
    "flow",
    "knowledge",
    "data",
    "security",
    "deployment",
    "decisions",
    "integrations",
    "evaluation",
    "lineage",
}
ALLOWED_DIAGRAM_SURFACES = {"agent_design", "agent_runtime", "implementation_guide", "builder_provenance"}
ALLOWED_SOURCE_KINDS = {
    "approved_journey_artifact",
    "generated_blueprint_export",
    "generated_acp_file",
    "generated_diagram",
    "commercial_summary",
    "producer_provenance",
}
ALLOWED_PORTABLE_SCOPES = {
    "agent_specification",
    "construction_package",
    "commercial_value",
    "producer_lineage",
}
ALLOWED_GENERATION_STATES = {"generated", "planned", "pending_generation", "not_generated"}


class TaxonomyValidationError(ValueError):
    pass


def _as_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
            continue
        seen.add(value)
    return sorted(duplicates)


def _validate_top_level(payload: dict[str, Any], errors: list[str]) -> None:
    if payload.get("schema_version") != "artifact-diagram-taxonomy.v1":
        errors.append("schema_version must be artifact-diagram-taxonomy.v1")

    products = set(_as_list(payload, "products"))
    if products != ALLOWED_PRODUCTS:
        errors.append("products must declare blueprint, acp and shared")

    access_levels = set(_as_list(payload, "access_levels"))
    if access_levels != ALLOWED_ACCESS_LEVELS:
        errors.append("access_levels must declare sample, view_only, downloadable, premium and restricted")

    artifact_categories = set(_as_list(payload, "artifact_categories"))
    missing_artifact_categories = ALLOWED_ARTIFACT_CATEGORIES - artifact_categories
    if missing_artifact_categories:
        errors.append(f"artifact_categories missing: {sorted(missing_artifact_categories)}")

    diagram_categories = set(_as_list(payload, "diagram_categories"))
    missing_diagram_categories = ALLOWED_DIAGRAM_CATEGORIES - diagram_categories
    if missing_diagram_categories:
        errors.append(f"diagram_categories missing: {sorted(missing_diagram_categories)}")


def _validate_artifacts(artifacts: list[dict[str, Any]], errors: list[str]) -> set[str]:
    artifact_keys = [str(item.get("artifact_key", "")) for item in artifacts]
    for duplicate in _find_duplicates(artifact_keys):
        errors.append(f"duplicate artifact_key: {duplicate}")

    for item in artifacts:
        artifact_key = str(item.get("artifact_key", ""))
        product_owner = item.get("product_owner")
        category = item.get("category")
        access_level = item.get("access_level")
        stage_owner = item.get("stage_owner")
        source_kind = item.get("source_kind")
        portable_scope = item.get("portable_scope")

        if not artifact_key:
            errors.append("artifact entry without artifact_key")
        if product_owner not in ALLOWED_PRODUCTS:
            errors.append(f"{artifact_key}: invalid product_owner {product_owner}")
        if category not in ALLOWED_ARTIFACT_CATEGORIES:
            errors.append(f"{artifact_key}: invalid category {category}")
        if access_level not in ALLOWED_ACCESS_LEVELS:
            errors.append(f"{artifact_key}: invalid access_level {access_level}")
        if stage_owner not in ALLOWED_STAGES:
            errors.append(f"{artifact_key}: invalid stage_owner {stage_owner}")
        if source_kind not in ALLOWED_SOURCE_KINDS:
            errors.append(f"{artifact_key}: invalid source_kind {source_kind}")
        if portable_scope not in ALLOWED_PORTABLE_SCOPES:
            errors.append(f"{artifact_key}: invalid portable_scope {portable_scope}")
        if not item.get("formats"):
            errors.append(f"{artifact_key}: formats cannot be empty")

        internal_refs_allowed = bool(item.get("internal_lean_refs_allowed"))
        if internal_refs_allowed and portable_scope != "producer_lineage":
            errors.append(f"{artifact_key}: internal Lean refs only allowed under producer_lineage")
        if product_owner == "acp" and bool(item.get("blueprint_download")):
            errors.append(f"{artifact_key}: acp-only artifact cannot be in blueprint_download")

    return set(artifact_keys)


def _validate_diagrams(
    diagrams: list[dict[str, Any]],
    artifact_keys: set[str],
    errors: list[str],
) -> None:
    diagram_keys = [str(item.get("diagram_key", "")) for item in diagrams]
    for duplicate in _find_duplicates(diagram_keys):
        errors.append(f"duplicate diagram_key: {duplicate}")

    represented_categories = {str(item.get("category", "")) for item in diagrams if item.get("is_active", True)}
    missing_diagram_categories = ALLOWED_DIAGRAM_CATEGORIES - represented_categories
    if missing_diagram_categories:
        errors.append(f"active diagram catalog missing categories: {sorted(missing_diagram_categories)}")

    for item in diagrams:
        diagram_key = str(item.get("diagram_key", ""))
        category = item.get("category")
        surface = item.get("diagram_surface")
        product_scope = item.get("product_scope") if isinstance(item.get("product_scope"), list) else []
        required_tier = item.get("required_tier")
        access_level = item.get("access_level")
        enabled_from_stage = item.get("enabled_from_stage")
        generation_state = item.get("default_generation_state")
        formats = item.get("formats") if isinstance(item.get("formats"), dict) else {}
        source_artifact_keys = item.get("source_artifact_keys") if isinstance(item.get("source_artifact_keys"), list) else []
        contains_internal = bool(item.get("contains_internal_lean_model"))

        if not diagram_key:
            errors.append("diagram entry without diagram_key")
        if category not in ALLOWED_DIAGRAM_CATEGORIES:
            errors.append(f"{diagram_key}: invalid category {category}")
        if surface not in ALLOWED_DIAGRAM_SURFACES:
            errors.append(f"{diagram_key}: invalid diagram_surface {surface}")
        if not product_scope or any(product not in ALLOWED_DIAGRAM_PRODUCTS for product in product_scope):
            errors.append(f"{diagram_key}: invalid product_scope {product_scope}")
        if required_tier not in ALLOWED_TIERS:
            errors.append(f"{diagram_key}: invalid required_tier {required_tier}")
        if access_level not in ALLOWED_ACCESS_LEVELS:
            errors.append(f"{diagram_key}: invalid access_level {access_level}")
        if enabled_from_stage not in ALLOWED_STAGES:
            errors.append(f"{diagram_key}: invalid enabled_from_stage {enabled_from_stage}")
        if generation_state not in ALLOWED_GENERATION_STATES:
            errors.append(f"{diagram_key}: invalid default_generation_state {generation_state}")
        if formats.get("preferred") not in set(formats.get("available", [])):
            errors.append(f"{diagram_key}: preferred format must be included in available formats")

        for source_key in source_artifact_keys:
            if source_key not in artifact_keys:
                errors.append(f"{diagram_key}: unknown source_artifact_key {source_key}")

        if "blueprint" in product_scope and contains_internal:
            errors.append(f"{diagram_key}: blueprint diagram cannot contain internal Lean model")
        if contains_internal and surface != "builder_provenance":
            errors.append(f"{diagram_key}: internal Lean model must be isolated under builder_provenance")
        if category == "lineage" and surface != "builder_provenance":
            errors.append(f"{diagram_key}: lineage diagrams must use builder_provenance surface")


def validate_artifact_diagram_taxonomy(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _validate_top_level(payload, errors)

    artifacts = [item for item in _as_list(payload, "artifact_entries") if isinstance(item, dict)]
    diagrams = [item for item in _as_list(payload, "diagram_entries") if isinstance(item, dict)]
    artifact_keys = _validate_artifacts(artifacts, errors)
    _validate_diagrams(diagrams, artifact_keys, errors)

    return errors


@lru_cache(maxsize=1)
def load_artifact_diagram_taxonomy() -> dict[str, Any]:
    payload = json.loads(TAXONOMY_MANIFEST_PATH.read_text(encoding="utf-8"))
    errors = validate_artifact_diagram_taxonomy(payload)
    if errors:
        raise TaxonomyValidationError("; ".join(errors))
    return payload


def get_artifact_taxonomy_entries() -> list[dict[str, Any]]:
    return list(load_artifact_diagram_taxonomy()["artifact_entries"])


def get_diagram_taxonomy_entries() -> list[dict[str, Any]]:
    return list(load_artifact_diagram_taxonomy()["diagram_entries"])


def get_artifact_taxonomy_by_key() -> dict[str, dict[str, Any]]:
    return {item["artifact_key"]: item for item in get_artifact_taxonomy_entries()}


def get_diagram_taxonomy_by_key() -> dict[str, dict[str, Any]]:
    return {item["diagram_key"]: item for item in get_diagram_taxonomy_entries()}
