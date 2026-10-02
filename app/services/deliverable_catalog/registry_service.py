from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass

from app.models import utc_now
from app.services.deliverable_catalog.contracts import (
    DeliverableCatalog,
    DeliverableGenerationMode,
    DeliverableRegistryEntry,
    DeliverableType,
    LEAN_STAGE_ORDER,
    ProductDeliveryProfile,
)
from app.services.deliverable_catalog.legacy_adapter import adapt_legacy_taxonomy_entries
from app.services.deliverable_catalog.manifest_service import load_seed_deliverable_catalog


class DeliverableRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedProductDeliveryPlan:
    product_key: str
    profile_key: str
    profile_revision: str
    generated_keys: tuple[str, ...]
    inherited_keys: tuple[str, ...]
    conditional_keys: tuple[dict[str, object], ...]
    unknown_dependency_keys: tuple[str, ...] = ()
    selection_source: str = "catalog_profile"

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "profile_key": self.profile_key,
            "profile_revision": self.profile_revision,
            "selection_source": self.selection_source,
            "generated_deliverable_keys": list(self.generated_keys),
            "inherited_deliverable_keys": list(self.inherited_keys),
            "conditional_deliverable_keys": list(self.conditional_keys),
            "unknown_dependency_keys": list(self.unknown_dependency_keys),
            "created_at": utc_now().isoformat(),
        }


@lru_cache(maxsize=1)
def load_deliverable_registry(*, include_seed: bool = True, include_legacy: bool = True) -> DeliverableCatalog:
    seed = load_seed_deliverable_catalog()
    entries = []
    if include_seed:
        entries.extend(seed.entries)
    if include_legacy:
        entries.extend(adapt_legacy_taxonomy_entries())

    by_key = {}
    duplicates = []
    for entry in entries:
        if entry.deliverable_key in by_key:
            duplicates.append(entry.deliverable_key)
            continue
        by_key[entry.deliverable_key] = entry
    if duplicates:
        raise DeliverableRegistryError(f"Duplicate deliverable keys: {sorted(set(duplicates))}")

    registry = DeliverableCatalog(
        generated_at=utc_now().date().isoformat(),
        lean_stage_order=list(LEAN_STAGE_ORDER),
        products=["blueprint", "blueprint_pro", "acp"],
        deliverable_types=[item for item in DeliverableType],
        generation_modes=[item for item in DeliverableGenerationMode],
        entries=sorted(by_key.values(), key=lambda entry: (entry.sort_order, entry.deliverable_key)),
        product_delivery_profiles=list(seed.product_delivery_profiles),
        validation_rules=seed.validation_rules,
    )
    validate_product_delivery_profiles(seed, registry)
    return registry


def list_registry_entries(*, include_inactive: bool = False) -> list:
    registry = load_deliverable_registry()
    if include_inactive:
        return list(registry.entries)
    return [entry for entry in registry.entries if entry.active]


def get_registry_entry(deliverable_key: str):
    normalized = str(deliverable_key or "").strip()
    return next((entry for entry in list_registry_entries(include_inactive=True) if entry.deliverable_key == normalized), None)


def load_product_delivery_profile(product_key: str) -> ProductDeliveryProfile:
    normalized = str(product_key or "").strip()
    registry = load_deliverable_registry()
    profile = next((item for item in registry.product_delivery_profiles if item.product_key == normalized), None)
    if profile is None:
        raise DeliverableRegistryError(f"Product delivery profile not found for product_key={normalized!r}")
    return profile


def validate_product_delivery_profiles(catalog: DeliverableCatalog, effective_registry: DeliverableCatalog) -> None:
    entries_by_key = {entry.deliverable_key: entry for entry in effective_registry.entries}
    errors: list[str] = []
    for profile in catalog.product_delivery_profiles:
        referenced_keys = [
            *profile.default_deliverable_keys,
            *profile.inherited_deliverable_keys,
            *[
                key
                for rule in profile.conditional_rules
                for key in rule.deliverable_keys
            ],
        ]
        for key in referenced_keys:
            entry = entries_by_key.get(key)
            if entry is None:
                errors.append(f"{profile.profile_key}: unknown deliverable key {key}")
                continue
            if not entry.active:
                errors.append(f"{profile.profile_key}: inactive deliverable key {key}")
        for rule in profile.conditional_rules:
            if not rule.deliverable_keys:
                errors.append(f"{profile.profile_key}: conditional rule {rule.condition_key} has no deliverables")
            if not rule.required_evidence:
                errors.append(f"{profile.profile_key}: conditional rule {rule.condition_key} has no required evidence")
    if errors:
        raise DeliverableRegistryError("; ".join(errors))


def resolve_product_delivery_plan(
    product_key: str,
    registry_entries: list[DeliverableRegistryEntry] | None = None,
    confirmed_signals: list[str] | tuple[str, ...] | set[str] | None = None,
) -> ResolvedProductDeliveryPlan:
    normalized_product_key = str(product_key or "").strip()
    profile = load_product_delivery_profile(normalized_product_key)
    entries = registry_entries if registry_entries is not None else list_registry_entries(include_inactive=True)
    active_keys = {entry.deliverable_key for entry in entries if entry.active}
    known_keys = {entry.deliverable_key for entry in entries}
    generated_keys = tuple(key for key in profile.default_deliverable_keys if key in active_keys)
    inherited_keys = tuple(key for key in profile.inherited_deliverable_keys if key in active_keys)
    unknown_dependency_keys = tuple(
        key
        for key in [*profile.default_deliverable_keys, *profile.inherited_deliverable_keys]
        if key not in known_keys or key not in active_keys
    )
    signals = {str(value or "").strip() for value in confirmed_signals or [] if str(value or "").strip()}
    conditional_payload: list[dict[str, object]] = []
    for rule in profile.conditional_rules:
        included = bool(signals.intersection(rule.required_evidence))
        conditional_payload.append(
            {
                "condition_key": rule.condition_key,
                "deliverable_keys": [key for key in rule.deliverable_keys if key in active_keys],
                "required_evidence": list(rule.required_evidence),
                "activation_mode": rule.activation_mode,
                "included": included,
                "reason": "confirmed_signal" if included else "excluded_until_condition_is_confirmed",
            }
        )
    if unknown_dependency_keys:
        raise DeliverableRegistryError(
            f"Product delivery profile {profile.profile_key} references unknown or inactive keys: "
            f"{sorted(set(unknown_dependency_keys))}"
        )
    return ResolvedProductDeliveryPlan(
        product_key=normalized_product_key,
        profile_key=profile.profile_key,
        profile_revision=profile.revision,
        generated_keys=generated_keys,
        inherited_keys=inherited_keys,
        conditional_keys=tuple(conditional_payload),
    )
