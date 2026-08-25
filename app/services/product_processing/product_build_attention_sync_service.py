from __future__ import annotations

from app.services.product_processing.contracts import ProductBuildStatus
from app.services.product_processing.product_build_progress_sync_service import sync_product_builds_for_stage_progress
from app.models import SessionRecord, SessionSnapshot, UserRecord
from sqlmodel import Session


def sync_product_builds_after_attention_action(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot | None = None,
    current_stage: str = "",
    current_user: UserRecord | None = None,
) -> list[ProductBuildStatus]:
    """Refresh product build runs after an Attention action changes state.

    Attention can resolve a question, approval, retry, or implementation dependency
    that was previously blocking a product build. This keeps the product surfaces
    from showing stale blockers after the user has already acted.
    """

    return sync_product_builds_for_stage_progress(
        db,
        record=record,
        snapshot=snapshot,
        current_stage=current_stage or getattr(record.current_stage, "value", str(record.current_stage or "discover")),
        current_user=current_user,
        source="attention_action",
    )
