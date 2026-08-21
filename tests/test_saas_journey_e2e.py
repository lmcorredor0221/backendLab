from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.api.routes.sessions import build_snapshot, resolve_acp_preview
from app.models import (
    CommercialEntitlementRecord,
    CommercialEntitlementSource,
    CommercialEntitlementStatus,
    CommercialEventRecord,
    CommercialTier,
    ExportJobCreateRequest,
    ExportJobStatus,
    ProjectTitleSource,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commercial_access import build_commercial_access_snapshot_v2
from app.services.commercial_observability_service import build_commercial_audit_report
from app.services.export_delivery_service import create_export_job
from app.services.product_processing import (
    ProductBuildOrchestrationOptions,
    ProductBuildProductKey,
    build_product_journey_overview,
    ensure_product_build_orchestration,
)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_complete_saas_journey_e2e(db_session: Session) -> None:
    """End-to-End test of the complete SaaS Journey:

    Register -> Workspace -> Project -> LEAN -> Basic -> Pro -> ACP -> Export -> Funnel.
    """
    # 1. Register & Workspace Setup
    user = UserRecord(
        email=f"founder-{uuid4().hex[:6]}@leanbuilder.local",
        full_name="E2E Founder",
        password_hash=hash_password("ValidPassword123!"),
    )
    db_session.add(user)
    db_session.flush()

    workspace = WorkspaceRecord(
        name="E2E Startup Workspace",
        slug=f"e2e-ws-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    db_session.add(workspace)
    db_session.flush()

    db_session.add(
        WorkspaceMembershipRecord(
            workspace_id=workspace.id,
            user_id=user.id,
            role=WorkspaceRole.owner,
        )
    )

    # 2. Create Project Session
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="Automated Assistant SaaS",
        title_source=ProjectTitleSource.manual,
        current_stage=SessionStage.draft_capture,
        commercial_tier=CommercialTier.blueprint,
    )
    db_session.add(record)
    db_session.flush()

    # 3. Complete LEAN stages through ready_for_export
    record.current_stage = SessionStage.ready_for_export
    db_session.add(record)
    db_session.commit()

    # 4. Generate Blueprint Basic
    build_response = ensure_product_build_orchestration(
        db_session,
        record=record,
        product_key=ProductBuildProductKey.blueprint_basic,
        current_user=user,
        options=ProductBuildOrchestrationOptions(idempotency_key="e2e-blueprint-basic"),
    )
    db_session.commit()
    assert build_response.product_key == ProductBuildProductKey.blueprint_basic

    # 5. Query Product Journey Overview v2
    overview = build_product_journey_overview(db_session, record=record, current_user=user)
    assert overview.contract_version == "product-journey-overview.v2"
    assert overview.session_id == record.id
    assert len(overview.products) >= 1

    # 6. Commercial Upgrade to Blueprint Pro
    record.commercial_tier = CommercialTier.blueprint_pro
    db_session.add(record)
    db_session.add(
        CommercialEntitlementRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            user_id=user.id,
            product_key="blueprint_pro",
            tier=CommercialTier.blueprint_pro,
            status=CommercialEntitlementStatus.active,
            source=CommercialEntitlementSource.checkout,
        )
    )
    db_session.add(
        CommercialEventRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            user_id=user.id,
            event_key="payment_confirmed",
            source="commerce_webhook",
            tier=CommercialTier.blueprint_pro,
            event_metadata={"product_key": "blueprint_pro", "amount_cents": 4900},
        )
    )
    db_session.commit()

    # 7. Commercial Upgrade to ACP
    record.commercial_tier = CommercialTier.acp
    record.current_stage = SessionStage.ready_for_export
    db_session.add(record)
    db_session.add(
        CommercialEntitlementRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            user_id=user.id,
            product_key="acp",
            tier=CommercialTier.acp,
            status=CommercialEntitlementStatus.active,
            source=CommercialEntitlementSource.checkout,
        )
    )
    db_session.add(
        CommercialEventRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            user_id=user.id,
            event_key="acp_phase_completed",
            source="acp_workflow",
            tier=CommercialTier.acp,
            event_metadata={"phase_key": "package_build"},
        )
    )
    db_session.commit()

    # 8. Create Export Job for Blueprint Professional & ACP Portable ZIP
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)

    job_response = create_export_job(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        payload=ExportJobCreateRequest(
            artifact_kind="blueprint_professional",
            profile="professional",
        ),
    )
    db_session.commit()
    assert job_response.session_id == record.id
    assert job_response.status == ExportJobStatus.ready
    assert job_response.checksum_sha256 != ""

    # 9. Verify Commercial Audit Funnel
    audit_report = build_commercial_audit_report(db_session, record=record, current_user=user)
    assert audit_report.session_id == record.id
    assert audit_report.current_tier == CommercialTier.acp
    assert len(audit_report.funnel) > 0
    assert len(audit_report.recent_events) >= 2

