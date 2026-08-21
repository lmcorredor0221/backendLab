from __future__ import annotations

from functools import lru_cache

from app.models import utc_now
from app.services.deliverable_catalog.contracts import (
    DeliverableCatalog,
    DeliverableGenerationMode,
    DeliverableType,
    LEAN_STAGE_ORDER,
)
from app.services.deliverable_catalog.legacy_adapter import adapt_legacy_taxonomy_entries
from app.services.deliverable_catalog.manifest_service import load_seed_deliverable_catalog


class DeliverableRegistryError(ValueError):
    pass


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

    return DeliverableCatalog(
        generated_at=utc_now().date().isoformat(),
        lean_stage_order=list(LEAN_STAGE_ORDER),
        products=["blueprint", "blueprint_pro", "acp"],
        deliverable_types=[item for item in DeliverableType],
        generation_modes=[item for item in DeliverableGenerationMode],
        entries=sorted(by_key.values(), key=lambda entry: (entry.sort_order, entry.deliverable_key)),
        validation_rules=seed.validation_rules,
    )


def list_registry_entries(*, include_inactive: bool = False) -> list:
    registry = load_deliverable_registry()
    if include_inactive:
        return list(registry.entries)
    return [entry for entry in registry.entries if entry.active]


def get_registry_entry(deliverable_key: str):
    normalized = str(deliverable_key or "").strip()
    return next((entry for entry in list_registry_entries(include_inactive=True) if entry.deliverable_key == normalized), None)
