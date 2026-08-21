from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.diagram_center.contracts import DiagramNotation


SPEC_ROOT = Path(__file__).resolve().parents[4] / "shared_specs"
LAYOUT_POLICY_PATH = SPEC_ROOT / "diagram-layout-policy.v1.json"

DEFAULT_LAYOUT_POLICY: dict[str, Any] = {
    "schema_version": "diagram-layout-guidance.v1",
    "preferred_strategy": "layered",
    "preferred_direction": "LR",
    "max_nodes_per_view": 16,
    "max_edges_per_view": 22,
    "max_edge_density": 0.16,
    "must_split_when_dense": True,
    "enable_adaptive_sizing": True,
    "enable_edge_routing": True,
    "visual_quality_min_score": 82,
    "label_policy": "Usar etiquetas cortas, de negocio, sin frases largas; mover detalles a description/source_refs.",
    "layout_policy": "Priorizar capas, lanes o boundaries para evitar cruces y solapamiento de textos.",
}


@lru_cache(maxsize=1)
def load_layout_policy_profiles() -> dict[str, dict[str, Any]]:
    payload = json.loads(LAYOUT_POLICY_PATH.read_text(encoding="utf-8"))
    return {
        str(profile["notation"]): profile
        for profile in payload.get("profiles", [])
        if isinstance(profile, dict) and profile.get("notation")
    }


def layout_policy_for_notation(notation: DiagramNotation | str) -> dict[str, Any]:
    notation_value = notation.value if isinstance(notation, DiagramNotation) else str(notation)
    profile = load_layout_policy_profiles().get(notation_value) or load_layout_policy_profiles().get("flowchart") or {}
    policy = dict(DEFAULT_LAYOUT_POLICY)
    policy.update(
        {
            "preferred_strategy": profile.get("layout_strategy") or policy["preferred_strategy"],
            "preferred_direction": profile.get("preferred_direction") or policy["preferred_direction"],
            "max_nodes_per_view": int(profile.get("max_nodes_before_split") or policy["max_nodes_per_view"]),
            "max_edges_per_view": int(profile.get("max_edges_before_split") or policy["max_edges_per_view"]),
            "max_edge_density": float(profile.get("max_edge_density") or policy["max_edge_density"]),
            "enable_adaptive_sizing": bool(profile.get("enable_adaptive_sizing", policy["enable_adaptive_sizing"])),
            "enable_edge_routing": bool(profile.get("enable_edge_routing", policy["enable_edge_routing"])),
            "visual_quality_min_score": int(profile.get("visual_quality_min_score") or policy["visual_quality_min_score"]),
        }
    )
    return policy


def merge_layout_policy(base: dict[str, Any], override: Any) -> dict[str, Any]:
    if not isinstance(override, dict):
        return dict(base)
    allowed = set(DEFAULT_LAYOUT_POLICY)
    merged = dict(base)
    for key, value in override.items():
        if key in allowed:
            merged[key] = value
    merged["schema_version"] = "diagram-layout-guidance.v1"
    return normalize_layout_policy(merged)


def normalize_layout_policy(policy: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(DEFAULT_LAYOUT_POLICY)
    normalized.update(policy)
    normalized["max_nodes_per_view"] = max(4, min(80, int(normalized["max_nodes_per_view"])))
    normalized["max_edges_per_view"] = max(4, min(160, int(normalized["max_edges_per_view"])))
    normalized["max_edge_density"] = max(0.01, min(1.0, float(normalized["max_edge_density"])))
    normalized["visual_quality_min_score"] = max(50, min(100, int(normalized["visual_quality_min_score"])))
    normalized["must_split_when_dense"] = bool(normalized["must_split_when_dense"])
    normalized["enable_adaptive_sizing"] = bool(normalized["enable_adaptive_sizing"])
    normalized["enable_edge_routing"] = bool(normalized["enable_edge_routing"])
    normalized["preferred_strategy"] = str(normalized["preferred_strategy"] or "layered")
    normalized["preferred_direction"] = str(normalized["preferred_direction"] or "LR")
    normalized["schema_version"] = "diagram-layout-guidance.v1"
    return normalized
