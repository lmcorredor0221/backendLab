from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    AccessRequestCreateRequest,
    AccessRequestResolveRequest,
    CommercialAccessRequestRecord,
    CommercialEventRecord,
    CommercialAccessRequestStatus,
    CommercialEntitlementRecord,
    CommercialEntitlementSource,
    CommercialEntitlementStatus,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPackageCatalogUpsertRequest,
    CommercialQuotaSourceKind,
    CommercialTier,
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    PlatformRole,
    PlatformRoleAssignmentRecord,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commerce_service import request_access, resolve_access_request
from app.services.commercial_access import build_commercial_access_snapshot_v2
from app.services.commercial_catalog_service import recommend_package_for_product, upsert_package_catalog_entry
from app.services.commercial_debt_service import list_commercial_debts
from app.services.commercial_quota_service import grant_balance_units
from app.services.product_processing import (
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductProcessingMode,
    ensure_product_build_run,
    list_product_build_runs,
)
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord  # noqa: F401


@pytest.fixture()
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_package_recommendation_prefers_minimum_sufficient_offer(db_session: Session) -> None:
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bp-pack-1",
            display_name="Blueprint Pack 1",
            product_key="blueprint_pro",
            granted_units=1,
            recommendation_priority=20,
        ),
    )
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bundle-monthly",
            display_name="Bundle mensual",
            product_key="bundle",
            package_type="bundle_subscription",
            granted_units_blueprint_pro=2,
            granted_units_acp=1,
            recommendation_priority=10,
        ),
    )
    upsert_package_catalog_entry(
        db_session,
        payload=CommercialPackageCatalogUpsertRequest(
            package_code="bp-pack-3",
            display_name="Blueprint Pack 3",
            product_key="blueprint_pro",
            granted_units=3,
            recommendation_priority=5,
        ),
    )
    db_session.commit()

    recommendation = recommend_package_for_product(
        db_session,
        product_key="blueprint_pro",
        required_units=2,
    )

    assert recommendation.package_code == "bundle-monthly"
    assert recommendation.granted_units_for_product == 2


def test_debt_pending_resolution_opens_debt_and_blocks_next_auto_approval(db_session: Session) -> None:
    user = UserRecord(
        email="commercial-debt@leanbuilder.local",
        full_name="Commercial Debt Tester",
        password_hash=hash_password("Secret123!"),
    )
    db_session.add(user)
    db_session.flush()
    workspace = WorkspaceRecord(
        name="Commercial Debt Workspace",
        slug=f"commercial-debt-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    db_session.add(workspace)
    db_session.flush()
    db_session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    db_session.add(
        PlatformRoleAssignmentRecord(
            user_id=user.id,
            role=PlatformRole.platform_admin,
            is_active=True,
        )
    )
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Debt Project")
    db_session.add(record)
    db_session.commit()
    db_session.refresh(user)
    db_session.refresh(workspace)
    db_session.refresh(record)

    first_request = request_access(
        db_session,
        payload=AccessRequestCreateRequest(session_id=record.id, capability="blueprint.build", reason="Initial request"),
        record=record,
        current_user=user,
        product_key="blueprint_pro",
        target_tier=CommercialTier.blueprint_pro,
    )
    db_session.commit()
    stored_request = db_session.exec(select(CommercialAccessRequestRecord).where(CommercialAccessRequestRecord.id == first_request.id)).one()

    resolve_access_request(
        db_session,
        access_request=stored_request,
        payload=AccessRequestResolveRequest(
            decision="approved",
            resolution_note="Aprobacion con deuda.",
            approval_mode="debt_pending",
            debt_reason_code="manual_debt",
            debt_reason_label="Deuda manual",
        ),
        current_user=user,
    )
    db_session.commit()

    debts = list_commercial_debts(db_session, workspace_id=workspace.id, status="open")
    assert len(debts) == 1
    assert debts[0].reason_code == "manual_debt"

    grant_balance_units(
        db_session,
        workspace_id=workspace.id,
        product_key="blueprint_pro",
        bucket_key="one-time:bp:1",
        source_kind=CommercialQuotaSourceKind.one_time,
        units=1,
        source_ref="test-balance",
        actor_user_id=user.id,
    )
    second_request = request_access(
        db_session,
        payload=AccessRequestCreateRequest(session_id=record.id, capability="blueprint.build", reason="Blocked by debt"),
        record=record,
        current_user=user,
        product_key="blueprint_pro",
        target_tier=CommercialTier.blueprint_pro,
    )
    db_session.commit()

    assert second_request.status == CommercialAccessRequestStatus.pending


def test_manual_acp_approval_finalizes_blueprint_handoff(db_session: Session) -> None:
    user = UserRecord(
        email="manual-acp@leanbuilder.local",
        full_name="Manual ACP Admin",
        password_hash=hash_password("Secret123!"),
    )
    db_session.add(user)
    db_session.flush()
    workspace = WorkspaceRecord(
        name="Manual ACP Workspace",
        slug=f"manual-acp-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    db_session.add(workspace)
    db_session.flush()
    db_session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    db_session.add(
        PlatformRoleAssignmentRecord(
            user_id=user.id,
            role=PlatformRole.platform_admin,
            is_active=True,
        )
    )
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Manual ACP Project")
    db_session.add(record)
    db_session.flush()
    tools_artifact = JourneyStageArtifactRecord(
        workspace_id=workspace.id,
        session_id=record.id,
        stage_key="tools",
        artifact_kind="tool_recommendation",
        version_number=1,
        state=JourneyArtifactState.stale,
        stale_reasons=["memory_reprocessed"],
    )
    db_session.add(tools_artifact)
    db_session.commit()

    access_response = request_access(
        db_session,
        payload=AccessRequestCreateRequest(session_id=record.id, capability="acp.build", reason="Manual ACP"),
        record=record,
        current_user=user,
        product_key="acp",
        target_tier=CommercialTier.acp,
    )
    stored_request = db_session.exec(
        select(CommercialAccessRequestRecord).where(CommercialAccessRequestRecord.id == access_response.id)
    ).one()
    assert stored_request.status == CommercialAccessRequestStatus.pending

    resolve_access_request(
        db_session,
        access_request=stored_request,
        payload=AccessRequestResolveRequest(
            decision="approved",
            resolution_note="Aprobacion manual ACP.",
            approval_mode="manual_standard",
        ),
        current_user=user,
    )
    db_session.commit()
    db_session.refresh(tools_artifact)

    handoff_events = db_session.exec(
        select(CommercialEventRecord).where(
            CommercialEventRecord.session_id == record.id,
            CommercialEventRecord.event_key == "blueprint_acp_handoff_finalized",
        )
    ).all()
    assert tools_artifact.state == JourneyArtifactState.approved
    assert tools_artifact.stale_reasons == []
    assert len(handoff_events) == 1


def test_workspace_owner_without_platform_admin_cannot_approve_access_request(db_session: Session) -> None:
    owner = UserRecord(
        email="workspace-owner-access@leanbuilder.local",
        full_name="Workspace Owner Access",
        password_hash=hash_password("Secret123!"),
    )
    db_session.add(owner)
    db_session.flush()
    workspace = WorkspaceRecord(
        name="Workspace Owner Access Workspace",
        slug=f"workspace-owner-access-{str(owner.id)[:8]}",
        created_by_user_id=owner.id,
    )
    db_session.add(workspace)
    db_session.flush()
    db_session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=owner.id, role=WorkspaceRole.owner))
    record = SessionRecord(user_id=owner.id, workspace_id=workspace.id, title="Owner Approval Project")
    db_session.add(record)
    db_session.commit()

    access_response = request_access(
        db_session,
        payload=AccessRequestCreateRequest(
            session_id=record.id,
            capability="blueprint.export",
            reason="Owner tries to approve",
        ),
        record=record,
        current_user=owner,
        product_key="blueprint_pro",
        target_tier=CommercialTier.blueprint_pro,
    )
    db_session.commit()
    stored_request = db_session.exec(
        select(CommercialAccessRequestRecord).where(CommercialAccessRequestRecord.id == access_response.id)
    ).one()

    with pytest.raises(PermissionError, match="platform admin puede aprobar"):
        resolve_access_request(
            db_session,
            access_request=stored_request,
            payload=AccessRequestResolveRequest(
                decision="approved",
                resolution_note="Aprobacion manual Blueprint Pro.",
                approval_mode="manual_standard",
            ),
            current_user=owner,
        )
    db_session.rollback()

    refreshed_request = db_session.get(CommercialAccessRequestRecord, stored_request.id)
    assert refreshed_request is not None
    assert refreshed_request.status == CommercialAccessRequestStatus.pending


def test_manual_blueprint_pro_approval_refreshes_locked_product_build(db_session: Session) -> None:
    user = UserRecord(
        email="manual-blueprint-pro@leanbuilder.local",
        full_name="Manual Blueprint Admin",
        password_hash=hash_password("Secret123!"),
    )
    db_session.add(user)
    db_session.flush()
    workspace = WorkspaceRecord(
        name="Manual Blueprint Workspace",
        slug=f"manual-blueprint-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    db_session.add(workspace)
    db_session.flush()
    db_session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    db_session.add(
        PlatformRoleAssignmentRecord(
            user_id=user.id,
            role=PlatformRole.platform_admin,
            is_active=True,
        )
    )
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Manual Blueprint Project")
    db_session.add(record)
    db_session.flush()
    stale_run = ensure_product_build_run(
        db_session,
        workspace_id=workspace.id,
        session_id=record.id,
        product_key=ProductBuildProductKey.blueprint_pro,
        product_mode=ProductProcessingMode.premium_enrichment,
        entitlement_tier=CommercialTier.blueprint,
        access_state="locked",
        lifecycle=ProductBuildLifecycle.requires_attention,
        idempotency_key=f"product-build:{record.id}:blueprint_pro",
    )
    db_session.commit()

    access_response = request_access(
        db_session,
        payload=AccessRequestCreateRequest(
            session_id=record.id,
            capability="blueprint.export",
            reason="Manual Blueprint Pro",
        ),
        record=record,
        current_user=user,
        product_key="blueprint_pro",
        target_tier=CommercialTier.blueprint_pro,
    )
    stored_request = db_session.exec(
        select(CommercialAccessRequestRecord).where(CommercialAccessRequestRecord.id == access_response.id)
    ).one()
    assert stored_request.status == CommercialAccessRequestStatus.pending

    resolve_access_request(
        db_session,
        access_request=stored_request,
        payload=AccessRequestResolveRequest(
            decision="approved",
            resolution_note="Aprobacion manual Blueprint Pro.",
            approval_mode="manual_standard",
        ),
        current_user=user,
    )
    db_session.commit()

    runs = list_product_build_runs(
        db_session,
        workspace_id=workspace.id,
        session_id=record.id,
        product_key=ProductBuildProductKey.blueprint_pro,
    )
    refreshed_run = db_session.get(ProductBuildRunRecord, stale_run.id)
    assert len(runs) == 1
    assert refreshed_run is not None
    assert refreshed_run.access_state == "allowed"
    assert refreshed_run.entitlement_tier == "blueprint_pro"


def test_active_entitlement_keeps_access_allowed_despite_stale_pending_checkout(db_session: Session) -> None:
    user = UserRecord(
        email="stale-checkout@leanbuilder.local",
        full_name="Stale Checkout Owner",
        password_hash=hash_password("Secret123!"),
    )
    db_session.add(user)
    db_session.flush()
    workspace = WorkspaceRecord(
        name="Stale Checkout Workspace",
        slug=f"stale-checkout-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    db_session.add(workspace)
    db_session.flush()
    db_session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Stale Checkout Project")
    db_session.add(record)
    db_session.flush()
    order = CommercialOrderRecord(
        workspace_id=workspace.id,
        session_id=record.id,
        buyer_user_id=user.id,
        status=CommercialOrderStatus.pending,
        currency="USD",
        subtotal_cents=4900,
        total_cents=4900,
        provider="hotmart",
        checkout_ref=f"hotmart_{record.id}",
        idempotency_key=f"stale-checkout:{record.id}",
    )
    db_session.add(order)
    db_session.flush()
    db_session.add(
        CommercialOrderLineRecord(
            order_id=order.id,
            product_key="blueprint_pro",
            price_code="blueprint-pro-usd-v1",
            quantity=1,
            unit_amount_cents=4900,
            total_amount_cents=4900,
        )
    )
    db_session.add(
        CommercialEntitlementRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            product_key="blueprint_pro",
            tier=CommercialTier.blueprint_pro,
            status=CommercialEntitlementStatus.active,
            source=CommercialEntitlementSource.admin_grant,
            granted_by_user_id=user.id,
        )
    )
    db_session.add(
        CommercialEntitlementRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            product_key="blueprint_pro",
            tier=CommercialTier.blueprint_pro,
            status=CommercialEntitlementStatus.revoked,
            source=CommercialEntitlementSource.admin_grant,
            revoked_by_user_id=user.id,
        )
    )
    db_session.commit()

    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)

    assert access.tier == CommercialTier.blueprint_pro
    assert access.reason_code == "allowed"
    assert access.checkout_state == "pending"
