from __future__ import annotations

from app.models import CommercialTier, SessionRecord, SessionSnapshot, UserRecord
from app.services.commerce_service import tier_rank
from app.services.product_processing.acp_product_orchestration_service import ensure_acp_product_orchestration
from app.services.product_processing.contracts import ProductBuildStatus
from app.services.product_processing.premium_enrichment_service import sync_premium_enrichment_product_run
from sqlmodel import Session


def sync_product_builds_after_attention_action(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot | None = None,
    current_user: UserRecord | None = None,
) -> list[ProductBuildStatus]:
    """Refresh product build runs after an Attention action changes state.

    Attention can resolve a question, approval, retry, or implementation dependency
    that was previously blocking a product build. This keeps the product surfaces
    from showing stale blockers after the user has already acted.
    """

    statuses: list[ProductBuildStatus] = []
    current_tier = record.commercial_tier
    if tier_rank(current_tier) >= tier_rank(CommercialTier.blueprint_pro):
        sync_premium_enrichment_product_run(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            current_tier=current_tier,
            current_user=current_user,
            source="attention_action",
        )
    if tier_rank(current_tier) >= tier_rank(CommercialTier.acp):
        statuses.append(
            ensure_acp_product_orchestration(
                db,
                record=record,
                snapshot=snapshot,
                current_user=current_user,
                activation_payload={"source": "attention_action"},
            )
        )
    return statuses
