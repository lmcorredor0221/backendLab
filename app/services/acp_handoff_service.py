from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlmodel import Session, select

from app.models import (
    CommercialEventRecord,
    JourneyArtifactState,
    JourneyDecisionType,
    JourneyStageArtifactRecord,
    JourneyStageDecisionRecord,
    SessionRecord,
    utc_now,
)

BLUEPRINT_ACP_HANDOFF_EVENT_KEY = "blueprint_acp_handoff_finalized"
BLUEPRINT_ACP_HANDOFF_SOURCE = "blueprint_acp_handoff"
BLUEPRINT_PROCESS_DEBT_STAGE_KEYS = {"tools", "memory", "validate", "estimate"}


def has_blueprint_acp_handoff_finalized(db: Session, *, session_id: UUID) -> bool:
    return (
        db.exec(
            select(CommercialEventRecord.id).where(
                CommercialEventRecord.session_id == session_id,
                CommercialEventRecord.product_key == "acp",
                CommercialEventRecord.event_key == BLUEPRINT_ACP_HANDOFF_EVENT_KEY,
            )
        ).first()
        is not None
    )


def finalize_blueprint_for_acp_handoff(
    db: Session,
    *,
    session_record: SessionRecord,
    actor_user_id: UUID | None,
    source: str,
    correlation_id: str = "",
) -> dict[str, Any]:
    """Close Blueprint-only operational debt when the user advances to ACP.

    This is deliberately conservative: it does not approve real business gates or
    erase history. It clears stale process markers from Blueprint construction
    artifacts and writes a durable audit event so Attention/ACP can start clean.
    """

    if session_record.workspace_id is None:
        return {
            "status": "skipped",
            "reason": "workspace_required",
            "closed_process_items": [],
            "preserved_real_debt": [],
        }

    if has_blueprint_acp_handoff_finalized(db, session_id=session_record.id):
        return {
            "status": "already_finalized",
            "closed_process_items": [],
            "preserved_real_debt": [],
        }

    now = utc_now()
    closed_process_items: list[dict[str, Any]] = []
    artifacts = db.exec(
        select(JourneyStageArtifactRecord).where(
            JourneyStageArtifactRecord.workspace_id == session_record.workspace_id,
            JourneyStageArtifactRecord.session_id == session_record.id,
            JourneyStageArtifactRecord.stage_key.in_(tuple(BLUEPRINT_PROCESS_DEBT_STAGE_KEYS)),
        )
    ).all()
    for artifact in artifacts:
        if artifact.state != JourneyArtifactState.stale and not artifact.stale_reasons:
            continue
        previous_state = artifact.state
        previous_reasons = list(artifact.stale_reasons or [])
        artifact.state = JourneyArtifactState.approved if artifact.state == JourneyArtifactState.stale else artifact.state
        artifact.stale_reasons = []
        artifact.stale_at = None
        artifact.updated_at = now
        db.add(artifact)
        db.add(
            JourneyStageDecisionRecord(
                workspace_id=session_record.workspace_id,
                session_id=session_record.id,
                artifact_id=artifact.id,
                stage_key=artifact.stage_key,
                decision_type=JourneyDecisionType.approve,
                previous_state=previous_state,
                next_state=artifact.state,
                actor_user_id=actor_user_id,
                note="Cierre operativo Blueprint -> ACP: el usuario tomo el Blueprint actual como verdad para ACP.",
                payload={
                    "source": source,
                    "policy": "blueprint_acp_handoff_closes_process_debt",
                    "closed_stale_reasons": previous_reasons,
                    "debt_kind": "process_debt",
                },
            )
        )
        closed_process_items.append(
            {
                "artifact_id": str(artifact.id),
                "stage_key": artifact.stage_key,
                "artifact_kind": artifact.artifact_kind,
                "previous_state": str(previous_state),
                "closed_stale_reasons": previous_reasons,
            }
        )

    event = CommercialEventRecord(
        workspace_id=session_record.workspace_id,
        session_id=session_record.id,
        user_id=actor_user_id,
        event_key=BLUEPRINT_ACP_HANDOFF_EVENT_KEY,
        product_key="acp",
        source=BLUEPRINT_ACP_HANDOFF_SOURCE,
        correlation_id=(correlation_id or str(uuid4()))[:128],
        metadata_payload={
            "source": source,
            "closed_process_items": closed_process_items,
            "preserved_real_debt": ["acp_questions", "delegated_decisions", "construction_readiness_gaps"],
            "preserved_policy": "real_questions_gaps_delegations_remain_in_acp_readiness",
            "critical_integrity_policy": "critical_issues_are_not_silently_closed",
            "idempotency": "one_finalization_event_per_session",
        },
    )
    db.add(event)
    session_record.updated_at = now
    db.add(session_record)
    db.flush()
    return {
        "status": "finalized",
        "closed_process_items": closed_process_items,
        "preserved_real_debt": ["acp_questions", "delegated_decisions", "construction_readiness_gaps"],
    }
