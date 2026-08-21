from __future__ import annotations

from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    CommercialTier,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.product_processing import (
    ProductBuildProductKey,
    ProductJourneyOverview,
    ProductProcessingMode,
    build_product_journey_overview,
)
from app.services.product_processing.persistence import UncertaintyBacklogRecord


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _seed_session(db: Session, *, tier: CommercialTier = CommercialTier.blueprint) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=f"journey-{uuid4()}@leanbuilder.local",
        full_name="Journey Tester",
        password_hash=hash_password("Secret123!"),
    )
    db.add(user)
    db.flush()
    workspace = WorkspaceRecord(name="Journey Workspace", slug=f"journey-{str(user.id)[:8]}", created_by_user_id=user.id)
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="Journey Project",
        current_stage=SessionStage.ready_for_export,
        commercial_tier=tier,
    )
    db.add(record)
    db.commit()
    db.refresh(user)
    db.refresh(record)
    return user, record


def test_product_journey_overview_serializes_contract_v2() -> None:
    workspace_id = uuid4()
    session_id = uuid4()
    payload = ProductJourneyOverview(
        workspace_id=workspace_id,
        session_id=session_id,
        project_title="Canonical Journey",
    ).model_dump(mode="json")

    assert payload["contract_version"] == "product-journey-overview.v2"
    assert payload["workspace_id"] == str(workspace_id)
    assert payload["session_id"] == str(session_id)
    assert payload["recommended_next_action"] is None


def test_product_journey_overview_keeps_purchase_separate_from_completion() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        overview = build_product_journey_overview(db, record=record, current_user=user)

    pro = next(product for product in overview.products if product.product_key == ProductBuildProductKey.blueprint_pro)

    assert pro.is_purchased is True
    assert pro.purchase_required is False
    assert pro.progress_percent < 100
    assert pro.lifecycle.value != "completed"
    assert overview.recommended_next_action is not None


def test_product_journey_overview_exposes_single_recommended_next_action() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint)
        overview = build_product_journey_overview(db, record=record, current_user=user)

    assert overview.recommended_next_action is not None
    assert overview.recommended_next_action.primary is True
    assert overview.recommended_next_action.product_key == ProductBuildProductKey.blueprint_basic


def test_product_journey_overview_prioritizes_blocking_attention_over_upsell() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        db.add(
            UncertaintyBacklogRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                uncertainty_key=f"blocking-{uuid4()}",
                product_mode=ProductProcessingMode.basic_free.value,
                source_stage="define",
                target_stage="define",
                disposition="block",
                status="open",
                title="Falta decidir el proceso principal",
                reason="Sin esta decision no se puede cerrar el Blueprint.",
            )
        )
        db.commit()

        overview = build_product_journey_overview(db, record=record, current_user=user)

    assert overview.blocking_attention_count == 1
    assert overview.recommended_next_action is not None
    assert overview.recommended_next_action.action_key == "open_attention"
    assert overview.recommended_next_action.product_key == ProductBuildProductKey.blueprint_basic
