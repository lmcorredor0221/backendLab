from __future__ import annotations

from uuid import UUID

from sqlmodel import Session

from app.models import CommercialTier, SessionRecord, UserRecord
from app.services.commerce_service import tier_rank
from app.services.product_processing.contracts import ProductBuildProductKey, ProductBuildStatus
from app.services.product_processing.product_build_orchestrator import (
    ProductBuildOrchestrationOptions,
    ensure_product_build_orchestration,
)


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
    try:
        return STAGE_FLOW_ORDER.index(_normalize_stage_key(stage_key))
    except ValueError:
        return 0


def _should_auto_execute_blueprint_pro_build(
    status: ProductBuildStatus | None,
    current_stage: str | None = None,
) -> bool:
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


def sync_blueprint_pro_product_run(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    current_tier: CommercialTier,
    current_user: UserRecord | None = None,
    source: str = "blueprint_pro_build",
    current_stage: str | None = None,
    auto_execute_when_ready: bool = False,
    allow_llm: bool = False,
) -> ProductBuildStatus | None:
    """Synchronize Blueprint Pro from real deliverable work only.

    Business uncertainty belongs to ACP. It must never be materialized as a
    Blueprint Pro build step or change the lifecycle of the professional build.
    """
    if tier_rank(current_tier) < tier_rank(CommercialTier.blueprint_pro):
        return None
    record = db.get(SessionRecord, session_id)
    if record is None or record.workspace_id != workspace_id:
        return None

    effective_stage = str(current_stage or getattr(record.current_stage, "value", str(record.current_stage or "discover")))
    status = ensure_product_build_orchestration(
        db,
        record=record,
        product_key=ProductBuildProductKey.blueprint_pro,
        current_user=current_user,
        options=ProductBuildOrchestrationOptions(
            current_stage=effective_stage,
            activation_payload={
                "source": source,
                "workspace_id": str(workspace_id),
                "session_id": str(session_id),
            },
        ),
    )
    if auto_execute_when_ready and _should_auto_execute_blueprint_pro_build(status, effective_stage):
        status = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=current_user,
            options=ProductBuildOrchestrationOptions(
                current_stage=effective_stage,
                execute_jobs=True,
                allow_llm=allow_llm,
                activation_payload={
                    "source": source,
                    "workspace_id": str(workspace_id),
                    "session_id": str(session_id),
                    "auto_execute": True,
                },
            ),
            catalog_stage_override=effective_stage,
        )
    return status
