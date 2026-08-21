from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services.deliverable_catalog.contracts import DeliverableRegistryEntry, DeliverableType
from app.services.deliverable_catalog.registry_service import list_registry_entries as list_deliverable_registry_entries
from app.services.diagram_center.contracts import DiagramNotation, DiagramRegistry, DiagramRegistryEntry
from app.services.diagram_center.layout_policy import layout_policy_for_notation, merge_layout_policy


SPEC_ROOT = Path(__file__).resolve().parents[4] / "shared_specs"
REGISTRY_PATH = SPEC_ROOT / "diagram-registry.v1.json"
STANDARD_PROFILES_PATH = SPEC_ROOT / "deliverable-standard-profiles.v1.json"

_STANDARD_BY_NOTATION: dict[DiagramNotation, str] = {
    DiagramNotation.flowchart: "Mermaid flowchart",
    DiagramNotation.sequence: "UML Sequence Diagram",
    DiagramNotation.class_diagram: "UML Class Diagram",
    DiagramNotation.entity_relationship: "Entity Relationship Diagram",
    DiagramNotation.state: "UML State Machine Diagram",
    DiagramNotation.journey: "User Journey Map",
    DiagramNotation.c4: "C4 Model",
    DiagramNotation.bpmn: "BPMN 2.0",
    DiagramNotation.uml_use_case: "UML Use Case Diagram",
    DiagramNotation.uml_activity: "UML Activity Diagram",
    DiagramNotation.uml_component: "UML Component Diagram",
    DiagramNotation.deployment: "UML Deployment Diagram",
    DiagramNotation.package: "UML Package Diagram",
    DiagramNotation.capability: "Capability Map",
}


def _empty_profile() -> dict[str, Any]:
    return {
        "notation": "flowchart",
        "standard": "Generic directed graph",
        "source_contract": "diagram-model.v1",
        "presentation_contract": "diagram-presentation.v1",
        "renderer_key": "renderer.svg.generic.v1",
        "validator_key": "diagram.graph_integrity.v1",
        "allowed_elements": ["node", "group", "edge"],
        "allowed_relationships": ["relationship"],
        "forbidden_mixes": ["secret", "credential", "platform_internal_id"],
    }


@lru_cache(maxsize=1)
def load_diagram_registry() -> DiagramRegistry:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return DiagramRegistry.model_validate(payload)


@lru_cache(maxsize=1)
def load_standard_profiles() -> dict[str, dict[str, Any]]:
    payload = json.loads(STANDARD_PROFILES_PATH.read_text(encoding="utf-8"))
    return {
        str(profile["profile_key"]): profile
        for profile in payload.get("profiles", [])
        if isinstance(profile, dict) and profile.get("profile_key")
    }


def _profile_key_for_entry(*, key: str, category: str = "", family: str = "", type_: str = "", notation: str = "") -> str:
    lookup = " ".join([key, category, family, type_, notation]).lower()
    if "use_case" in lookup or "caso" in lookup:
        return "uml.use_case.v1"
    if "activity" in lookup or "actividad" in lookup:
        return "uml.activity.v1"
    if (
        "bpmn" in lookup
        or "current_process" in lookup
        or "current process" in lookup
        or "process_map" in lookup
        or "proceso actual" in lookup
        or ("process" in lookup and "business" in lookup)
    ):
        return "bpmn.2_0.v1"
    if "sequence" in lookup or "tool_contract" in lookup:
        return "uml.sequence.v1"
    if "component" in lookup:
        return "uml.component.v1"
    if "class" in lookup or "code" in lookup:
        return "uml.class.v1"
    if "state" in lookup:
        return "uml.state.v1"
    if "c4_context" in lookup or "solution_architecture" in lookup:
        return "c4.context.v1"
    if "c4_container" in lookup or "logical_architecture" in lookup:
        return "c4.container.v1"
    if "entity_relationship" in lookup or "data_model" in lookup or "erd" in lookup:
        return "data.erd.v1"
    if "agent" in lookup or "orchestration" in lookup or "memory_rag" in lookup:
        return "agentic.workflow.v1"
    return ""


def _profile_for_entry(*, key: str, category: str = "", family: str = "", type_: str = "", notation: str = "") -> dict[str, Any]:
    profile_key = _profile_key_for_entry(key=key, category=category, family=family, type_=type_, notation=notation)
    if profile_key:
        return dict(load_standard_profiles().get(profile_key, _empty_profile()))
    for profile in load_standard_profiles().values():
        if profile.get("notation") == notation and notation not in {"flowchart", ""}:
            return dict(profile)
    return _empty_profile()


def _profile_for_notation(notation: str) -> dict[str, Any]:
    for profile in load_standard_profiles().values():
        if profile.get("notation") == notation:
            return dict(profile)
    return _empty_profile()


def _with_standard_profile(entry: DiagramRegistryEntry) -> DiagramRegistryEntry:
    profile = _profile_for_entry(
        key=entry.key,
        category=entry.category,
        family=entry.family,
        type_=entry.type,
        notation=entry.notation.value,
    )
    def profile_value(field: str, generic_default: str = "") -> str:
        current = str(getattr(entry, field, "") or "")
        candidate = str(profile.get(field, "") or "")
        if candidate and current in {"", generic_default}:
            return candidate
        return current

    update: dict[str, Any] = {
        "standard": entry.standard or profile.get("standard", ""),
        "source_contract": profile_value("source_contract", "diagram-model.v1"),
        "presentation_contract": profile_value("presentation_contract", "diagram-presentation.v1"),
        "renderer_key": profile_value("renderer_key", "renderer.svg.generic.v1"),
        "validator_key": profile_value("validator_key", "diagram.graph_integrity.v1"),
        "allowed_elements": entry.allowed_elements or list(profile.get("allowed_elements", [])),
        "allowed_relationships": entry.allowed_relationships or list(profile.get("allowed_relationships", [])),
        "forbidden_mixes": entry.forbidden_mixes or list(profile.get("forbidden_mixes", [])),
    }
    profile_notation = profile.get("notation")
    if profile_notation and entry.notation.value == "flowchart":
        update["notation"] = profile_notation
    payload = entry.model_dump(mode="json")
    payload.update(update)
    return DiagramRegistryEntry.model_validate(payload)


def diagram_entry_from_deliverable(entry: DeliverableRegistryEntry) -> DiagramRegistryEntry:
    key = entry.deliverable_key.removeprefix("diagram.")
    base_notation = "capability" if entry.category == "capability" else "flowchart"
    profile = _profile_for_entry(
        key=key,
        category=entry.category,
        family=entry.deliverable_type.value,
        type_=entry.category,
        notation=base_notation,
    )
    return _with_standard_profile(
        DiagramRegistryEntry(
            key=key,
            title=entry.title,
            description=entry.description,
            benefit=entry.description,
            category=entry.category,
            type=entry.category,
            family="deliverable_catalog",
            notation=str(profile.get("notation") or base_notation),
            standard=str(profile.get("standard") or ""),
            source_contract=str(profile.get("source_contract") or "diagram-model.v1"),
            presentation_contract=str(profile.get("presentation_contract") or "diagram-presentation.v1"),
            renderer_key=str(profile.get("renderer_key") or "renderer.svg.generic.v1"),
            validator_key=str(profile.get("validator_key") or "diagram.graph_integrity.v1"),
            allowed_elements=list(profile.get("allowed_elements", [])),
            allowed_relationships=list(profile.get("allowed_relationships", [])),
            forbidden_mixes=list(profile.get("forbidden_mixes", [])),
            complexity="basic" if entry.required_tier.value == "blueprint" else "intermediate",
            stage=entry.stage,
            required_tier=entry.required_tier.value,
            preview_mode=entry.access_policy.preview_mode,
            products=list(entry.product_scope),
            required_inputs=list(entry.context_policy.short_term_refs or entry.dependency_policy.depends_on),
            actions=["view", "generate"],
            objective=entry.description,
            semantic_rules=[
                "Usar solo contexto aprobado y trazable.",
                "Mantener el menor numero de nodos necesario para explicar la decision.",
                "No exponer informacion interna de la plataforma.",
            ],
            exclusions=[
                "No incluir secretos, credenciales ni identificadores internos.",
            ],
            sort_order=entry.sort_order,
            active=entry.active,
        )
    )


def list_registry_entries(*, include_inactive: bool = False) -> list[DiagramRegistryEntry]:
    entries = [_with_standard_profile(entry) for entry in load_diagram_registry().entries]
    existing_keys = {entry.key.lower() for entry in entries}
    for deliverable in list_deliverable_registry_entries(include_inactive=include_inactive):
        if deliverable.deliverable_type != DeliverableType.diagram:
            continue
        derived = diagram_entry_from_deliverable(deliverable)
        if derived.key.lower() not in existing_keys:
            entries.append(derived)
            existing_keys.add(derived.key.lower())
    if not include_inactive:
        entries = [entry for entry in entries if entry.active]
    return sorted(entries, key=lambda entry: (entry.sort_order, entry.title.lower()))


def get_registry_entry(diagram_key: str) -> DiagramRegistryEntry | None:
    normalized = diagram_key.strip().lower()
    return next((entry for entry in list_registry_entries(include_inactive=True) if entry.key.lower() == normalized), None)


def _safe_notation(value: Any, fallback: DiagramNotation) -> DiagramNotation:
    if not value:
        return fallback
    try:
        return DiagramNotation(str(value))
    except ValueError:
        return fallback


def build_prompt_spec(entry: DiagramRegistryEntry, *, override: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = load_diagram_registry()
    override = override or {}
    effective_notation = _safe_notation(override.get("notation"), entry.notation)
    notation_overridden = bool(override.get("notation"))
    notation_profile = _profile_for_notation(effective_notation.value) if notation_overridden else {}
    effective_standard = str(
        override.get("standard")
        or notation_profile.get("standard")
        or (_STANDARD_BY_NOTATION.get(effective_notation) if notation_overridden else entry.standard)
        or _STANDARD_BY_NOTATION.get(effective_notation)
        or effective_notation.value
    )
    source_contract = str(override.get("source_contract") or notation_profile.get("source_contract") or entry.source_contract)
    presentation_contract = str(
        override.get("presentation_contract") or notation_profile.get("presentation_contract") or entry.presentation_contract
    )
    renderer_key = str(override.get("renderer_key") or notation_profile.get("renderer_key") or entry.renderer_key)
    validator_key = str(override.get("validator_key") or notation_profile.get("validator_key") or entry.validator_key)
    allowed_elements = list(override.get("allowed_elements") or notation_profile.get("allowed_elements") or entry.allowed_elements)
    allowed_relationships = list(
        override.get("allowed_relationships") or notation_profile.get("allowed_relationships") or entry.allowed_relationships
    )
    forbidden_mixes = list(override.get("forbidden_mixes") or notation_profile.get("forbidden_mixes") or entry.forbidden_mixes)
    layout_guidance = merge_layout_policy(
        layout_policy_for_notation(effective_notation),
        override.get("layout_guidance"),
    )
    semantic_rules = [
        f"Modelar exclusivamente con la notacion {effective_standard}.",
        "Usar solo los elementos permitidos para este estandar.",
        "Usar solo relaciones validas para esta notacion.",
        "Mantener trazabilidad a fuentes aprobadas en todos los nodos, relaciones y supuestos.",
        *list(entry.semantic_rules),
    ]
    if effective_notation == DiagramNotation.bpmn:
        semantic_rules.extend(
            [
                "Inferir dinamicamente pools como participantes, organizaciones, sistemas externos o areas responsables desde el contexto aprobado.",
                "Inferir dinamicamente lanes como roles, equipos o responsabilidades dentro de cada pool.",
                "Declarar pools y lanes en el campo `pools`; no usar un pool o lane generico salvo que no exista evidencia suficiente.",
                "Asignar cada nodo BPMN a `metadata.pool_id` y `metadata.lane_id` usando ids existentes en `pools`.",
                "Usar `sequence_flow` dentro del mismo pool y `message_flow` cuando la relacion cruza participantes o sistemas.",
                "No representar BPMN como grafo generico; usar eventos, tareas, gateways, subprocesos, pools, lanes y flujos BPMN segun aplique.",
            ]
        )
    exclusions = [
        *list(entry.exclusions),
        *[f"No mezclar con {item}." for item in forbidden_mixes],
        "No inventar tecnologia, proveedor, dato, costo, endpoint, credencial ni owner no aprobado.",
    ]
    prompt_spec: dict[str, Any] = {
        "version": registry.prompt_spec_version,
        "diagram_key": entry.key,
        "deliverable_key": entry.key,
        "deliverable_family": entry.family,
        "objective": str(override.get("objective") or entry.objective),
        "notation": effective_notation.value,
        "standard": effective_standard,
        "required_inputs": list(entry.required_inputs),
        "semantic_rules": semantic_rules,
        "exclusions": exclusions,
        "output_contract": source_contract,
        "legacy_output_contract": "diagram-model.v1",
        "source_contract": source_contract,
        "presentation_contract": presentation_contract,
        "renderer_key": renderer_key,
        "validator_key": validator_key,
        "allowed_elements": allowed_elements,
        "allowed_relationships": allowed_relationships,
        "forbidden_mixes": forbidden_mixes,
        "inherits_from": list(entry.inherits_from or entry.required_inputs),
        "transform_rules": list(entry.transform_rules),
        "layout_guidance": layout_guidance,
        "generation_permissions": dict(
            entry.generation_permissions
            or {
                "may_infer": True,
                "must_register_assumptions": True,
                "must_defer_implementation_decisions": True,
                "must_not_generate_restricted_values": True,
            }
        ),
        "quality_gates": [
            f"El resultado debe respetar {effective_standard}.",
            f"El renderer objetivo es {renderer_key}.",
            f"El validador objetivo es {validator_key}.",
            "Todos los nodos y relaciones deben estar respaldados por el contexto aprobado.",
            "Los identificadores deben ser estables, unicos y seguros.",
            "Toda relacion debe apuntar a nodos existentes.",
            "No deben aparecer secretos, datos personales ni instrucciones internas.",
            "El diagrama debe ser legible con el minimo numero de elementos necesario.",
            "Si la vista supera la densidad recomendada, generar vista resumen y proponer vistas de detalle en lugar de amontonar todos los nodos.",
            f"Estrategia de layout esperada: {layout_guidance['preferred_strategy']}.",
        ],
    }
    if effective_notation == DiagramNotation.bpmn:
        prompt_spec["quality_gates"].extend(
            [
                "BPMN debe incluir pools/lanes inferidos cuando existan participantes, roles o sistemas diferenciables.",
                "Cada nodo debe quedar asignado a una lane trazable mediante metadata.pool_id y metadata.lane_id.",
                "Los flujos entre pools deben modelarse como message_flow, no como sequence_flow.",
            ]
        )
    if override:
        for key in (
            "semantic_rules",
            "exclusions",
            "quality_gates",
            "inherits_from",
            "transform_rules",
            "generation_permissions",
        ):
            if key in override:
                prompt_spec[key] = override[key]
        prompt_spec["output_contract"] = str(prompt_spec.get("source_contract") or prompt_spec["output_contract"])
    return prompt_spec
