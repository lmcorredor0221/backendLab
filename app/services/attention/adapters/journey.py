from __future__ import annotations

from app.models import AttentionItemV2
from app.services.attention.contract import create_attention_item_v2


STAGE_KIND_LABELS: dict[str, str] = {
    "estimation_report_artifact": "El reporte de estimación",
    "discovery_artifact": "El artefacto de descubrimiento",
    "definition_artifact": "La definición de requerimientos",
    "design_recommendation_artifact": "El diseño de arquitectura",
    "tool_recommendation_artifact": "La recomendación de herramientas",
    "memory_recommendation_artifact": "La política de memoria",
    "evaluation_artifact": "La evaluación del blueprint",
}


def _humanize_stale_reason(reason: str, state: str) -> str:
    if not reason:
        return f"El artefacto está en estado {state} y requiere revisión."
    if "upstream_" in reason:
        return "Se generaron o aprobaron nuevas versiones en etapas previas. Es necesario regenerar esta etapa para reflejar los últimos cambios y mantener la consistencia."
    return reason


def items_from_stage_artifact_state(
    *,
    stage: str,
    artifact_id: str,
    artifact_version: int | None,
    artifact_kind: str,
    state: str,
    reason: str,
    href: str,
    return_href: str,
) -> list[AttentionItemV2]:
    normalized_state = state.strip().lower()
    if normalized_state not in {"stale", "blocked", "requires_review", "needs_review_legacy", "rejected"}:
        return []
    item_type = "stale" if normalized_state == "stale" else "validation"
    severity = "blocking" if normalized_state in {"blocked", "rejected"} else "warning"
    label = STAGE_KIND_LABELS.get(artifact_kind, f"El artefacto {artifact_kind}")
    title = f"{label} está desactualizado" if normalized_state == "stale" else f"{label} requiere revisión"
    human_reason = _humanize_stale_reason(reason, normalized_state)

    return [
        create_attention_item_v2(
            item_type=item_type,
            severity=severity,
            product="blueprint",
            stage=stage,
            source="journey_artifact",
            source_ref={
                "artifact_id": artifact_id,
                "artifact_version": artifact_version,
                "field_path": f"{artifact_kind}.state",
            },
            title=title,
            reason=human_reason,
            impact="Puede afectar la trazabilidad y la consistencia entre etapas del Blueprint.",
            consequence_if_unresolved="El Blueprint continuará utilizando datos de una versión previa desactualizada.",
            action_kind="regenerate" if state == "stale" else "navigate",
            href=href,
            return_href=return_href,
            affected_artifact_refs=[f"{artifact_kind}:{artifact_id}:v{artifact_version or 0}"],
        )
    ]
