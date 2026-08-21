from __future__ import annotations

from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.models import utc_now
from app.services.deliverable_catalog.persistence import DeliverableGovernanceRecord
from app.services.deliverable_catalog.policy_service import scope_key_for_workspace
from app.services.deliverable_catalog.registry_service import list_registry_entries


def _seed_deliverable_governance_defaults_once(session: Session, *, workspace_id: UUID | None = None) -> int:
    scope_key = scope_key_for_workspace(workspace_id)
    existing = {
        item.deliverable_key: item
        for item in session.exec(
            select(DeliverableGovernanceRecord).where(DeliverableGovernanceRecord.scope_key == scope_key)
        ).all()
    }
    created = 0
    for entry in list_registry_entries(include_inactive=True):
        if entry.deliverable_key in existing:
            continue
        session.add(
            DeliverableGovernanceRecord(
                scope_key=scope_key,
                workspace_id=workspace_id,
                deliverable_key=entry.deliverable_key,
                enabled=entry.active,
                generation_enabled=True,
                required_tier_override="",
                preview_mode_override="",
                prompt_status=entry.prompt_policy.prompt_status,
                prompt_override={},
                notes="BDG17 default heredado desde deliverable-catalog.v1.",
                updated_at=utc_now(),
            )
        )
        created += 1
    session.commit()
    return created


def seed_deliverable_governance_defaults(session: Session, *, workspace_id: UUID | None = None) -> int:
    for attempt in range(3):
        try:
            return _seed_deliverable_governance_defaults_once(session, workspace_id=workspace_id)
        except IntegrityError:
            session.rollback()
            if attempt == 2:
                raise
    return 0
