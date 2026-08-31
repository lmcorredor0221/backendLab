from __future__ import annotations

from app.models import CommercialTier, SessionRecord, SessionSnapshot, UserRecord
from app.services.commerce_service import tier_rank
from app.services.product_processing.acp_product_orchestration_service import ensure_acp_product_orchestration
from app.services.product_processing.contracts import ProductBuildProductKey, ProductBuildStatus
from app.services.product_processing.journey_state_machine_service import transition_for_stage_approval
from app.services.product_processing.premium_enrichment_service import sync_premium_enrichment_product_run
from app.services.product_processing.product_build_orchestrator import (
    ProductBuildOrchestrationOptions,
    ensure_product_build_orchestration,
)
from sqlmodel import Session


STAGE_FLOW_ORDER = (
    "discover",
    "define",
    "design",
    "tools",
    "memory",
    "estimate",
    "validate",
    "package",
)


def sync_product_builds_after_stage_approval(
    db: Session,
    *,
    record: SessionRecord,
    stage_key: str,
    snapshot: SessionSnapshot | None = None,
    current_user: UserRecord | None = None,
) -> list[ProductBuildStatus]:
    normalized_stage = _normalize_stage_key(stage_key)
    transition_for_stage_approval(
        db,
        record=record,
        approved_stage_key=normalized_stage,
        actor_user_id=current_user.id if current_user is not None else None,
        # The proposal service/touch_session updates this timestamp for every actual approval,
        # while retried work in the same transaction remains idempotent.
        correlation_id=f"stage-approval:{record.id}:{normalized_stage}:{record.updated_at.isoformat()}",
    )
    return sync_product_builds_for_stage_progress(
        db,
        record=record,
        snapshot=snapshot,
        current_stage=normalized_stage,
        current_user=current_user,
        source=f"stage_approval:{normalized_stage}",
    )


def sync_product_builds_for_stage_progress(
    db: Session,
    *,
    record: SessionRecord,
    current_stage: str,
    snapshot: SessionSnapshot | None = None,
    current_user: UserRecord | None = None,
    source: str,
) -> list[ProductBuildStatus]:
    statuses: list[ProductBuildStatus] = []
    normalized_stage = _normalize_stage_key(current_stage)
    current_tier = record.commercial_tier

    if tier_rank(current_tier) >= tier_rank(CommercialTier.blueprint_pro):
        premium_status = sync_premium_enrichment_product_run(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            current_tier=current_tier,
            current_user=current_user,
            source=source,
            current_stage=normalized_stage,
            auto_execute_when_ready=True,
            allow_llm=True,
        )
        if premium_status is not None:
            statuses.append(premium_status)

    if tier_rank(current_tier) < tier_rank(CommercialTier.acp):
        return statuses

    if normalized_stage == "package":
        statuses.append(
            ensure_acp_product_orchestration(
                db,
                record=record,
                snapshot=snapshot,
                current_user=current_user,
                execute_jobs=True,
                allow_llm=True,
                activation_payload={
                    "source": source,
                    "stage_key": normalized_stage,
                    "auto_execute": True,
                },
            )
        )
        return statuses

    acp_status = _sync_stage_scoped_product_build(
        db,
        record=record,
        product_key=ProductBuildProductKey.acp,
        current_stage=normalized_stage,
        current_user=current_user,
        source=source,
        allow_llm=True,
    )
    if acp_status is not None:
        statuses.append(acp_status)
    return statuses


def _sync_stage_scoped_product_build(
    db: Session,
    *,
    record: SessionRecord,
    product_key: ProductBuildProductKey,
    current_stage: str,
    current_user: UserRecord | None,
    source: str,
    allow_llm: bool,
) -> ProductBuildStatus:
    normalized_stage = _normalize_stage_key(current_stage)
    status = ensure_product_build_orchestration(
        db,
        record=record,
        product_key=product_key,
        current_user=current_user,
        options=ProductBuildOrchestrationOptions(
            current_stage=normalized_stage,
            activation_payload={
                "source": source,
                "stage_key": normalized_stage,
            },
        ),
        catalog_stage_override=normalized_stage,
    )
    if not _should_auto_execute_stage_build(status, normalized_stage):
        return status
    return ensure_product_build_orchestration(
        db,
        record=record,
        product_key=product_key,
        current_user=current_user,
        options=ProductBuildOrchestrationOptions(
            current_stage=normalized_stage,
            execute_jobs=True,
            allow_llm=allow_llm,
            activation_payload={
                "source": source,
                "stage_key": normalized_stage,
                "auto_execute": True,
            },
        ),
        catalog_stage_override=normalized_stage,
    )


def _normalize_stage_key(value: str | None) -> str:
    stage = str(value or "").strip().lower()
    if stage in STAGE_FLOW_ORDER:
        return stage
    legacy_map = {
        "draft_capture": "discover",
        "input_validation": "discover",
        "normalize_discovery": "discover",
        "build_canvas": "define",
        "build_blueprint": "design",
        "post_validation": "validate",
        "ready_for_export": "package",
    }
    return legacy_map.get(stage, "discover")


def _stage_index(stage_key: str | None) -> int:
    normalized = _normalize_stage_key(stage_key)
    try:
        return STAGE_FLOW_ORDER.index(normalized)
    except ValueError:
        return 0


def _should_auto_execute_stage_build(status: ProductBuildStatus | None, current_stage: str) -> bool:
    if status is None or status.entitlement.access_state != "allowed":
        return False
    current_stage_idx = _stage_index(current_stage)
    if any(
        item.blocking and _stage_index(item.stage_key or current_stage) <= current_stage_idx
        for item in status.attention.items
    ):
        return False
    relevant_deliverables = [
        item for item in status.deliverables if _stage_index(item.stage_key) <= current_stage_idx
    ]
    if not relevant_deliverables:
        relevant_deliverables = list(status.deliverables)
    if any(getattr(item.state, "value", str(item.state)) in {"queued", "generating"} for item in relevant_deliverables):
        return False
    return any(getattr(item.state, "value", str(item.state)) in {"pending", "stale"} for item in relevant_deliverables)
