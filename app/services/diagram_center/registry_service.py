from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.diagram_center.contracts import DiagramRegistry, DiagramRegistryEntry


REGISTRY_PATH = Path(__file__).resolve().parents[4] / "shared_specs" / "diagram-registry.v1.json"


@lru_cache(maxsize=1)
def load_diagram_registry() -> DiagramRegistry:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return DiagramRegistry.model_validate(payload)


def list_registry_entries(*, include_inactive: bool = False) -> list[DiagramRegistryEntry]:
    entries = load_diagram_registry().entries
    if not include_inactive:
        entries = [entry for entry in entries if entry.active]
    return sorted(entries, key=lambda entry: (entry.sort_order, entry.title.lower()))


def get_registry_entry(diagram_key: str) -> DiagramRegistryEntry | None:
    normalized = diagram_key.strip().lower()
    return next((entry for entry in load_diagram_registry().entries if entry.key.lower() == normalized), None)


def build_prompt_spec(entry: DiagramRegistryEntry, *, override: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = load_diagram_registry()
    prompt_spec: dict[str, Any] = {
        "version": registry.prompt_spec_version,
        "diagram_key": entry.key,
        "objective": entry.objective,
        "notation": entry.notation.value,
        "required_inputs": list(entry.required_inputs),
        "semantic_rules": list(entry.semantic_rules),
        "exclusions": list(entry.exclusions),
        "output_contract": "diagram-model.v1",
        "quality_gates": [
            "Todos los nodos y relaciones deben estar respaldados por el contexto aprobado.",
            "Los identificadores deben ser estables, únicos y seguros.",
            "Toda relación debe apuntar a nodos existentes.",
            "No deben aparecer secretos, datos personales ni instrucciones internas.",
            "El diagrama debe ser legible con el mínimo número de elementos necesario.",
        ],
    }
    if override:
        for key in ("objective", "semantic_rules", "exclusions", "quality_gates"):
            if key in override:
                prompt_spec[key] = override[key]
    return prompt_spec

