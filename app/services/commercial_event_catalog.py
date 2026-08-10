from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMERCIAL_EVENT_CATALOG_PATH = REPO_ROOT / "shared_specs" / "commercial-event-catalog.v1.json"
DEFAULT_EVENT_SCHEMA_VERSION = "commercial-event.v1"


@lru_cache
def load_commercial_event_catalog() -> dict[str, Any]:
    return json.loads(COMMERCIAL_EVENT_CATALOG_PATH.read_text(encoding="utf-8"))


def resolve_commercial_event_catalog_entry(event_key: str) -> dict[str, Any]:
    catalog = load_commercial_event_catalog()
    normalized_key = event_key.strip()
    for entry in catalog.get("events", []):
        if entry.get("event_key") == normalized_key:
            return {**entry, "catalog_state": "registered"}
    for pattern_entry in catalog.get("event_key_patterns", []):
        if re.match(str(pattern_entry.get("pattern", "")), normalized_key):
            return {
                **pattern_entry,
                "event_key": normalized_key,
                "catalog_state": "pattern_registered",
            }
    return {
        "event_key": normalized_key or "unknown",
        "schema_version": DEFAULT_EVENT_SCHEMA_VERSION,
        "category": "custom",
        "product": "commercial",
        "source": "custom",
        "revenue_semantics": "unknown",
        "catalog_state": "unregistered",
        "description": "Evento no registrado todavia en el catalogo formal.",
    }


def enrich_commercial_event_metadata(event_key: str, metadata: dict | None = None) -> dict[str, Any]:
    entry = resolve_commercial_event_catalog_entry(event_key)
    enriched = dict(metadata or {})
    enriched.setdefault("event_schema_version", entry.get("schema_version") or DEFAULT_EVENT_SCHEMA_VERSION)
    enriched.setdefault("event_category", entry.get("category") or "custom")
    enriched.setdefault("event_catalog_state", entry.get("catalog_state") or "unregistered")
    enriched.setdefault("event_revenue_semantics", entry.get("revenue_semantics") or "unknown")
    return enriched
