from __future__ import annotations

import json
from typing import Any

from app.models import CommercialTier
from app.services.artifact_diagram_taxonomy import (
    get_artifact_taxonomy_entries,
    get_diagram_taxonomy_entries,
)
from app.services.deliverable_catalog.contracts import (
    DeliverableAccessPolicy,
    DeliverableContentProtection,
    DeliverableContextPolicy,
    DeliverableDependencyPolicy,
    DeliverableFormats,
    DeliverableGenerationMode,
    DeliverablePromptPolicy,
    DeliverableQualityPolicy,
    DeliverableRegistryEntry,
    DeliverableType,
)
from app.services.shared_specs import resolve_shared_spec_path


_TYPE_BY_ARTIFACT_CATEGORY = {
    "contract": DeliverableType.contract,
    "prompt": DeliverableType.prompt,
    "package": DeliverableType.package,
    "lineage": DeliverableType.lineage,
}


def _required_tier(product_owner: str, access_level: str, required_tier: str = "") -> CommercialTier:
    if required_tier:
        return CommercialTier(required_tier)
    if product_owner == "acp" or access_level == "restricted":
        return CommercialTier.acp
    if access_level in {"downloadable", "premium"}:
        return CommercialTier.blueprint_pro
    return CommercialTier.blueprint


def _product_scope(product_owner: str, *, blueprint_download: bool = False, acp_download: bool = False) -> list[str]:
    if product_owner == "shared":
        return ["blueprint", "blueprint_pro", "acp"]
    scope: list[str] = []
    if product_owner == "blueprint" or blueprint_download:
        scope.extend(["blueprint", "blueprint_pro"])
    if product_owner == "acp" or acp_download:
        scope.append("acp")
    return list(dict.fromkeys(scope or ["blueprint"]))


def _diagram_product_scope(products: list[str], required_tier: str) -> list[str]:
    scope: list[str] = []
    if "blueprint" in products:
        scope.append("blueprint")
        scope.append("blueprint_pro")
    if "acp" in products:
        scope.append("acp")
    if required_tier == "blueprint_pro" and "blueprint_pro" not in scope:
        scope.append("blueprint_pro")
    return list(dict.fromkeys(scope or ["blueprint"]))


def _formats_from_artifact(item: dict[str, Any]) -> DeliverableFormats:
    formats = [str(value) for value in item.get("formats", []) if str(value).strip()]
    canonical_paths = item.get("canonical_paths") or []
    if canonical_paths and str(canonical_paths[0]).endswith(".md"):
        preferred = "markdown" if "markdown" in formats else ("md" if "md" in formats else (formats[0] if formats else "markdown"))
    elif canonical_paths and str(canonical_paths[0]).endswith(".json"):
        preferred = "json"
    else:
        preferred = formats[0] if formats else "json"
    return DeliverableFormats(preferred=preferred, available=formats or [preferred])


def _formats_from_diagram(item: dict[str, Any]) -> DeliverableFormats:
    formats = item.get("formats") if isinstance(item.get("formats"), dict) else {}
    available = [str(value) for value in formats.get("available", []) if str(value).strip()]
    preferred = str(formats.get("preferred") or (available[0] if available else "svg"))
    return DeliverableFormats(preferred=preferred, available=available or [preferred])


def _formats_from_diagram_center_entry() -> DeliverableFormats:
    return DeliverableFormats(preferred="svg", available=["svg", "mermaid", "json"])


def _content_protection(access_level: str) -> DeliverableContentProtection:
    protected = access_level in {"sample", "view_only", "premium", "restricted"}
    return DeliverableContentProtection(
        disable_copy=protected,
        disable_download=protected,
        disable_context_menu=protected,
    )


def adapt_artifact_entry(item: dict[str, Any], *, sort_offset: int = 1000) -> DeliverableRegistryEntry:
    artifact_key = str(item["artifact_key"])
    access_level = str(item.get("access_level", "view_only"))
    product_owner = str(item.get("product_owner", "blueprint"))
    blueprint_download = bool(item.get("blueprint_download", False))
    acp_download = bool(item.get("acp_download", False))
    exportable = blueprint_download or acp_download or bool(item.get("canonical_paths"))
    return DeliverableRegistryEntry(
        deliverable_key=artifact_key,
        title=str(item.get("title") or artifact_key),
        description=str(item.get("description") or "Entregable legacy migrado desde artifact-diagram taxonomy."),
        deliverable_type=_TYPE_BY_ARTIFACT_CATEGORY.get(str(item.get("category", "")), DeliverableType.artifact),
        category=str(item.get("category") or "functional"),
        stage=str(item.get("stage_owner") or "discover"),
        enabled_from_stage=str(item.get("stage_owner") or "discover"),
        product_scope=_product_scope(product_owner, blueprint_download=blueprint_download, acp_download=acp_download),
        required_tier=_required_tier(product_owner, access_level),
        access_level=access_level,
        formats=_formats_from_artifact(item),
        generation_mode=DeliverableGenerationMode.deterministic,
        prompt_policy=DeliverablePromptPolicy(
            schema_contract="deliverable-artifact.v1",
            validator_key="artifact.generic.v1",
            fallback_policy="deterministic_only",
        ),
        context_policy=DeliverableContextPolicy(
            short_term_refs=[f"stage.{item.get('stage_owner', 'discover')}"],
            long_term_collections=[],
            max_context_tokens=3000,
            retrieval_strategy="approved_stage_snapshot_only",
        ),
        quality_policy=DeliverableQualityPolicy(
            schema_contract="deliverable-artifact.v1",
            validator_key="artifact.generic.v1",
            minimum_score=75,
            checks=["has_title", "has_description", "has_canonical_source"],
        ),
        dependency_policy=DeliverableDependencyPolicy(
            depends_on=[str(item.get("source_kind") or "approved_journey_artifact")],
            invalidates_on_change=[f"stage.{item.get('stage_owner', 'discover')}"],
        ),
        access_policy=DeliverableAccessPolicy(
            preview_mode="full" if access_level in {"sample", "view_only"} else "limited",
            sample_enabled=access_level == "sample",
            content_protection=_content_protection(access_level),
        ),
        canonical_paths=list(item.get("canonical_paths", []) or []),
        portable_paths=list(item.get("portable_paths", []) or []),
        exportable=exportable,
        blueprint_download=blueprint_download,
        acp_download=acp_download,
        sort_order=sort_offset,
        active=True,
    )


def adapt_diagram_entry(item: dict[str, Any], *, sort_offset: int = 2000) -> DeliverableRegistryEntry:
    diagram_key = str(item["diagram_key"])
    access_level = str(item.get("access_level", "view_only"))
    required_tier = str(item.get("required_tier", "blueprint"))
    products = [str(value) for value in item.get("product_scope", [])]
    prompt_key = f"deliverables.diagram.{diagram_key}.v1"
    fallback = (
        "deterministic_existing_content"
        if item.get("default_generation_state") == "generated"
        else "fail_visible_without_synthetic_diagram"
    )
    return DeliverableRegistryEntry(
        deliverable_key=f"diagram.{diagram_key}",
        title=str(item.get("title") or diagram_key),
        description=str(item.get("description") or "Diagrama legacy migrado desde artifact-diagram taxonomy."),
        deliverable_type=DeliverableType.diagram,
        category=str(item.get("category") or "architecture"),
        stage=str(item.get("enabled_from_stage") or "design"),
        enabled_from_stage=str(item.get("enabled_from_stage") or "design"),
        product_scope=_diagram_product_scope(products, required_tier),
        required_tier=CommercialTier(required_tier),
        access_level=access_level,
        formats=_formats_from_diagram(item),
        generation_mode=DeliverableGenerationMode.llm_supported,
        prompt_policy=DeliverablePromptPolicy(
            prompt_template_key=prompt_key,
            prompt_status="active",
            prompt_version="1.0.0",
            schema_contract="diagram-model.v1",
            validator_key="diagram.graph_integrity.v1",
            fallback_policy=fallback,
            max_iterations=3,
        ),
        context_policy=DeliverableContextPolicy(
            short_term_refs=list(item.get("source_artifact_keys", []) or []),
            long_term_collections=["repo_docs.agent_patterns"],
            max_context_tokens=6000,
            retrieval_strategy="stage_artifacts_plus_relevant_long_term_memory",
        ),
        quality_policy=DeliverableQualityPolicy(
            schema_contract="diagram-model.v1",
            validator_key="diagram.graph_integrity.v1",
            minimum_score=80,
            checks=["unique_node_ids", "valid_edges", "source_refs_present"],
        ),
        dependency_policy=DeliverableDependencyPolicy(
            depends_on=list(item.get("source_artifact_keys", []) or []),
            invalidates_on_change=list(item.get("source_artifact_keys", []) or []),
        ),
        access_policy=DeliverableAccessPolicy(
            preview_mode="limited" if access_level in {"sample", "view_only"} else "none",
            sample_enabled=access_level == "sample",
            content_protection=_content_protection(access_level),
        ),
        canonical_paths=list(item.get("portable_paths", []) or []),
        portable_paths=list(item.get("portable_paths", []) or []),
        exportable=bool(item.get("portable_paths")),
        blueprint_download=CommercialTier(required_tier) != CommercialTier.acp and "blueprint" in products,
        acp_download="acp" in products,
        sort_order=int(item.get("sort_order") or sort_offset),
        active=bool(item.get("is_active", True)),
    )


def _diagram_center_access_level(required_tier: str) -> str:
    if required_tier == CommercialTier.acp.value:
        return "restricted"
    if required_tier == CommercialTier.blueprint_pro.value:
        return "premium"
    return "view_only"


def _diagram_center_portable_paths(diagram_key: str, item: dict[str, Any]) -> list[str]:
    paths = [str(value) for value in item.get("portable_paths", []) if str(value).strip()]
    return paths or [f"Deliverables/diagram.{diagram_key}.svg"]


def _diagram_center_entries() -> list[dict[str, Any]]:
    payload = json.loads(resolve_shared_spec_path("diagram-registry.v1.json").read_text(encoding="utf-8"))
    return [entry for entry in payload.get("entries", []) if isinstance(entry, dict)]


def adapt_diagram_center_registry_entry(item: dict[str, Any], *, sort_offset: int = 3000) -> DeliverableRegistryEntry:
    diagram_key = str(item["key"])
    required_tier = str(item.get("required_tier") or CommercialTier.blueprint.value)
    tier = CommercialTier(required_tier)
    products = [str(value) for value in item.get("products", []) if str(value).strip()]
    required_inputs = [str(value) for value in item.get("required_inputs", []) if str(value).strip()]
    access_level = _diagram_center_access_level(required_tier)
    portable_paths = _diagram_center_portable_paths(diagram_key, item)
    return DeliverableRegistryEntry(
        deliverable_key=f"diagram.{diagram_key}",
        title=str(item.get("title") or diagram_key),
        description=str(item.get("description") or "Diagrama migrado desde diagram-registry."),
        deliverable_type=DeliverableType.diagram,
        category=str(item.get("category") or "architecture"),
        stage=str(item.get("stage") or "design"),
        enabled_from_stage=str(item.get("stage") or "design"),
        product_scope=_diagram_product_scope(products, required_tier),
        required_tier=tier,
        access_level=access_level,
        formats=_formats_from_diagram_center_entry(),
        generation_mode=DeliverableGenerationMode.llm_supported,
        prompt_policy=DeliverablePromptPolicy(
            prompt_template_key=f"deliverables.diagram.{diagram_key}.v1",
            prompt_status="active",
            prompt_version="1.0.0",
            schema_contract="diagram-model.v1",
            validator_key="diagram.graph_integrity.v1",
            fallback_policy="fail_visible_without_synthetic_diagram",
            max_iterations=3,
        ),
        context_policy=DeliverableContextPolicy(
            short_term_refs=required_inputs,
            long_term_collections=["repo_docs.agent_patterns"],
            max_context_tokens=6000,
            retrieval_strategy="stage_artifacts_plus_relevant_long_term_memory",
        ),
        quality_policy=DeliverableQualityPolicy(
            schema_contract="diagram-model.v1",
            validator_key="diagram.graph_integrity.v1",
            minimum_score=80,
            checks=["unique_node_ids", "valid_edges", "source_refs_present"],
        ),
        dependency_policy=DeliverableDependencyPolicy(
            depends_on=required_inputs,
            invalidates_on_change=required_inputs,
        ),
        access_policy=DeliverableAccessPolicy(
            preview_mode=str(item.get("preview_mode") or ("none" if tier == CommercialTier.acp else "limited")),
            sample_enabled=False,
            content_protection=_content_protection(access_level),
        ),
        canonical_paths=portable_paths,
        portable_paths=portable_paths,
        exportable=True,
        blueprint_download=tier != CommercialTier.acp and "blueprint" in products,
        acp_download="acp" in products,
        sort_order=sort_offset + int(item.get("sort_order") or 0),
        active=bool(item.get("active", True)),
    )


def adapt_diagram_center_registry_entries(*, existing_keys: set[str] | None = None) -> list[DeliverableRegistryEntry]:
    seen = {key.lower() for key in existing_keys or set()}
    entries: list[DeliverableRegistryEntry] = []
    for item in _diagram_center_entries():
        deliverable_key = f"diagram.{item.get('key')}"
        if deliverable_key.lower() in seen:
            continue
        entry = adapt_diagram_center_registry_entry(item)
        entries.append(entry)
        seen.add(entry.deliverable_key.lower())
    return entries


def adapt_legacy_taxonomy_entries() -> list[DeliverableRegistryEntry]:
    entries: list[DeliverableRegistryEntry] = []
    for index, item in enumerate(get_artifact_taxonomy_entries(), start=1):
        entries.append(adapt_artifact_entry(item, sort_offset=1000 + index))
    for index, item in enumerate(get_diagram_taxonomy_entries(), start=1):
        entries.append(adapt_diagram_entry(item, sort_offset=2000 + index))
    entries.extend(adapt_diagram_center_registry_entries(existing_keys={entry.deliverable_key for entry in entries}))
    return entries
