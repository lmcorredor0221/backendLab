from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from app.models import AttentionItemV2, AttentionOptionV2, CommercialAccessSnapshotV2, CommercialTier, SessionRecord
from app.services.attention.contract import create_attention_item_v2
from app.services.product_processing.contracts import ProductProcessingMode
from app.services.product_processing.policy import resolve_product_processing_mode


_BASIC_SURFACED_TYPES = {"access_request", "runtime_error", "stale"}
_VALIDATION_SUMMARY_THRESHOLD = 4
_GENERIC_VALIDATION_WARNING_MARKERS = (
    "contiene blockers",
    "preguntas abiertas",
    "trazabilidad, criterios",
)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _mode_for_session(record: SessionRecord, access: CommercialAccessSnapshotV2) -> ProductProcessingMode:
    tier = getattr(access, "tier", None) or getattr(record, "commercial_tier", None) or CommercialTier.blueprint
    return resolve_product_processing_mode(tier)


def _is_basic_blueprint_noise(item: AttentionItemV2) -> bool:
    if item.product != "blueprint":
        return False
    if item.type in _BASIC_SURFACED_TYPES:
        return False
    if item.type == "approval":
        entity_id = getattr(item.source_ref, "entity_id", "") or ""
        field_path = getattr(item.source_ref, "field_path", "") or ""
        return entity_id.startswith("tool:") or "tool:" in field_path
    return item.type in {
        "question",
        "validation",
        "inconsistency",
        "gap",
        "decision",
        "confirmation",
        "hitl",
    }


def _is_generic_validation_warning(item: AttentionItemV2) -> bool:
    if item.type != "inconsistency":
        return False
    normalized = f"{item.title} {item.reason}".lower()
    return any(marker in normalized for marker in _GENERIC_VALIDATION_WARNING_MARKERS)


def _stage_label(stage: str) -> str:
    labels = {
        "discover": "Descubrir",
        "define": "Definir",
        "design": "Disenar",
        "tools": "Herramientas",
        "memory": "Memoria",
        "validate": "Validar",
        "estimate": "Estimar",
        "package": "Package",
    }
    return labels.get(stage, stage.title() if stage else "la etapa")


def _validation_group_key(item: AttentionItemV2) -> tuple[str, str, str, str, str]:
    ref = item.source_ref
    return (
        item.product,
        item.stage,
        item.source,
        _value(ref.artifact_id),
        _value(ref.artifact_version),
    )


def _summarize_validation_group(group: list[AttentionItemV2]) -> AttentionItemV2:
    first = group[0]
    severity = "blocking" if any(item.severity == "blocking" for item in group) else "warning"
    refs = sorted({ref for item in group for ref in item.affected_artifact_refs})
    issue_labels = ", ".join(sorted({item.title for item in group})[:3])
    stage_label = _stage_label(first.stage)
    artifact_id = _value(first.source_ref.artifact_id)
    artifact_version = _value(first.source_ref.artifact_version)
    return create_attention_item_v2(
        key=f"attention.v2:validation-summary:{first.product}:{first.stage}:{first.source}:{artifact_id}:{artifact_version}",
        item_type="validation",
        severity=severity,
        product=first.product,
        stage=first.stage,
        source=first.source,
        source_ref={
            "artifact_id": artifact_id,
            "artifact_version": first.source_ref.artifact_version,
            "entity_id": "validation_summary",
            "field_path": "validation_issues",
        },
        title=f"{stage_label} tiene {len(group)} hallazgo(s) de calidad por revisar",
        reason=(
            "Se agruparon hallazgos internos de trazabilidad, criterios o consistencia para evitar "
            "convertirlos en una lista larga de preguntas al usuario."
        ),
        impact=(
            "Reduce carga cognitiva y mantiene la revision enfocada en una accion de calidad, no en "
            "multiples decisiones tecnicas aisladas."
        ),
        consequence_if_unresolved=(
            "Los hallazgos quedan como deuda de enriquecimiento y pueden afectar la confianza del Blueprint "
            "o la preparacion del ACP."
        ),
        action_kind="confirm",
        action_label="Revisar calidad",
        href=first.action.href,
        return_href=first.action.return_href,
        owner_role=first.owner_role or "business_owner",
        options=[
            AttentionOptionV2(
                key="register_as_enrichment_work",
                label="Registrar para enriquecimiento",
                description="Mantener los hallazgos como trabajo de mejora sin bloquear el flujo actual.",
                impact="Alinea Blueprint Basico/Premium con una revision posterior controlada.",
                recommended=True,
                confidence=0.76,
                source_refs=refs,
            ),
            AttentionOptionV2(
                key="review_in_stage",
                label=f"Revisar en {stage_label}",
                description="Abrir la etapa para ajustar criterios, trazabilidad o fuentes.",
                impact="Mejora la calidad del artefacto antes de aprobar o exportar.",
                recommended=False,
                confidence=0.7,
                source_refs=refs,
            ),
        ],
        suggested_answer=(
            f"Registrar los {len(group)} hallazgo(s) como enriquecimiento guiado y revisar {stage_label} "
            "solo si afectan una decision de negocio."
        ),
        affected_artifact_refs=refs,
        can_resolve_inline=True,
    )


def _group_validation_noise(items: list[AttentionItemV2]) -> list[AttentionItemV2]:
    validation_groups: dict[tuple[str, str, str, str, str], list[AttentionItemV2]] = defaultdict(list)
    passthrough: list[AttentionItemV2] = []

    for item in items:
        if item.type == "validation" and item.source.startswith("journey."):
            validation_groups[_validation_group_key(item)].append(item)
        else:
            passthrough.append(item)

    grouped_items: list[AttentionItemV2] = []
    grouped_stages: set[tuple[str, str]] = set()
    for key, group in validation_groups.items():
        if len(group) >= _VALIDATION_SUMMARY_THRESHOLD:
            grouped_items.append(_summarize_validation_group(group))
            grouped_stages.add((key[0], key[1]))
        else:
            grouped_items.extend(group)

    if grouped_stages:
        passthrough = [
            item
            for item in passthrough
            if not ((item.product, item.stage) in grouped_stages and _is_generic_validation_warning(item))
        ]

    return [*passthrough, *grouped_items]


def _normalized_refs(item: AttentionItemV2) -> set[str]:
    refs: set[str] = set()
    for raw_ref in item.affected_artifact_refs:
        ref = str(raw_ref or "").strip().lower()
        if ref:
            refs.add(ref)
    return refs


def _is_derived_promotion_blocker(item: AttentionItemV2) -> bool:
    if item.source != "governance_policy" or item.type != "validation" or item.severity != "blocking":
        return False
    refs = _normalized_refs(item)
    return "policy_key=promotion_blockers" in refs and "blueprint_readiness=blocked" in refs


def _is_actionable_blueprint_blocker(item: AttentionItemV2) -> bool:
    return item.product == "blueprint" and item.severity == "blocking" and item.source != "governance_policy"


def _suppress_derived_promotion_blockers(items: list[AttentionItemV2]) -> list[AttentionItemV2]:
    if not any(_is_derived_promotion_blocker(item) for item in items):
        return items
    if not any(_is_actionable_blueprint_blocker(item) for item in items):
        return items
    return [item for item in items if not _is_derived_promotion_blocker(item)]


def govern_attention_items(
    items: Iterable[AttentionItemV2],
    *,
    record: SessionRecord,
    access: CommercialAccessSnapshotV2,
    current_stage: str = "",
) -> list[AttentionItemV2]:
    """Apply SaaS/product policy before Attention becomes user-facing.

    Attention V3 is currently an internal decision contract. This governor is the
    mandatory publication boundary that prevents deferred Basic uncertainty from
    leaking into the active user inbox while preserving runtime recovery items.
    """

    del current_stage  # Reserved for future stage-aware prioritization.
    mode = _mode_for_session(record, access)
    visible = list(items)

    if mode == ProductProcessingMode.basic_free:
        visible = [item for item in visible if not _is_basic_blueprint_noise(item)]
        return visible

    visible = _group_validation_noise(visible)
    return _suppress_derived_promotion_blockers(visible)
