from __future__ import annotations

from app.models import AttentionItemV2
from app.services.attention.contract import create_attention_item_v2


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
            title=f"{artifact_kind} requiere revision",
            reason=reason or "El artefacto cambio de estado y requiere atencion del usuario.",
            impact="Puede afectar la trazabilidad y la consistencia entre etapas LEAN.",
            consequence_if_unresolved="La etapa puede continuar usando una version no vigente o no aprobada.",
            action_kind="regenerate" if state == "stale" else "navigate",
            href=href,
            return_href=return_href,
            affected_artifact_refs=[f"{artifact_kind}:{artifact_id}:v{artifact_version or 0}"],
        )
    ]
