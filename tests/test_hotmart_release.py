from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialEventRecord,
    HotmartCredentialUpsertRequest,
    HotmartIntegrationConfigRecord,
    HotmartPaymentLinkRecord,
    HotmartProductMappingRecord,
    HotmartSyncRunRecord,
    HotmartWebhookEventRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.auth_service import hash_password
from app.services.hotmart.release import build_hotmart_release_readiness, list_hotmart_operational_alerts
from app.services.hotmart.secrets import upsert_hotmart_credentials


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


def test_hotmart_release_readiness_is_ready_with_complete_operational_evidence(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    _configure_connected_hotmart(db_session, workspace)
    db_session.add(
        HotmartProductMappingRecord(
            workspace_id=workspace.id,
            environment="sandbox",
            internal_product_key="blueprint_pro",
            hotmart_product_id="1234567",
            is_active=True,
        )
    )
    db_session.add(
        HotmartPaymentLinkRecord(
            workspace_id=workspace.id,
            created_by_user_id=user.id,
            environment="sandbox",
            internal_product_key="blueprint_pro",
            hotmart_payment_link_id="payment-link-1",
            checkout_url="https://pay.hotmart.test/payment-link-1",
            activation_status="active",
            provider_ref="payment-link-1",
            gross_amount_cents=10000,
            net_amount_cents=10000,
            internal_unit_amount_usd_cents=10000,
        )
    )
    db_session.add(
        HotmartWebhookEventRecord(
            event_id="event-approved-1",
            event_type="PURCHASE_APPROVED",
            transaction="HP123",
            workspace_id=workspace.id,
            hottok_validated=True,
            processing_status="processed",
            payload_hash="hash-approved",
            payload_redacted={"event": "PURCHASE_APPROVED"},
            processed_at=utc_now(),
        )
    )
    for event_key in ("hotmart_payment_approved", "hotmart_payment_refunded"):
        db_session.add(
            CommercialEventRecord(
                workspace_id=workspace.id,
                user_id=user.id,
                event_key=event_key,
                product_key="blueprint_pro",
                source="hotmart_webhook",
                correlation_id=event_key,
            )
        )
    for resource in ("club", "coupons", "payment_links", "products", "sales"):
        db_session.add(
            HotmartSyncRunRecord(
                workspace_id=workspace.id,
                environment="sandbox",
                resource=resource,
                status="succeeded",
                records_read=1,
                finished_at=utc_now(),
            )
        )
    db_session.commit()

    readiness = build_hotmart_release_readiness(db_session, workspace_id=workspace.id, environment="sandbox")

    assert readiness.overall_status == "ready"
    assert readiness.release_candidate is True
    assert readiness.alerts == []
    assert readiness.metrics["active_mappings"] == 1
    assert readiness.metrics["successful_sync_resources"] == 5
    assert {item.status for item in readiness.checklist} == {"passed"}
    assert any(section.key == "rollback" for section in readiness.runbook)


def test_hotmart_release_readiness_blocks_incomplete_production_configuration(db_session: Session) -> None:
    _, workspace = _seed_workspace(db_session)

    readiness = build_hotmart_release_readiness(db_session, workspace_id=workspace.id, environment="production")
    alerts = list_hotmart_operational_alerts(db_session, workspace_id=workspace.id, environment="production")

    assert readiness.overall_status == "blocked"
    assert readiness.release_candidate is False
    assert any(item.key == "oauth_credentials_configured" and item.status == "failed" for item in readiness.checklist)
    assert any(alert.key == "hotmart_oauth_credentials_incomplete" for alert in alerts)
    assert any(alert.severity == "critical" for alert in alerts)


def _seed_workspace(session: Session) -> tuple[UserRecord, WorkspaceRecord]:
    user = UserRecord(
        email="hotmart-release@leanbuilder.local",
        full_name="Hotmart Release Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(
        name="Hotmart Release Workspace",
        slug=f"hotmart-release-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    return user, workspace


def _configure_connected_hotmart(session: Session, workspace: WorkspaceRecord) -> None:
    upsert_hotmart_credentials(
        session,
        workspace_id=workspace.id,
        payload=HotmartCredentialUpsertRequest(
            environment="sandbox",
            enabled=True,
            client_id="client-id-value",
            client_secret="client-secret-value",
            basic_token="basic-token-value",
            hottok="hottok-value",
            webhook_public_url="https://example.com/api/v1/webhooks/hotmart",
        ),
    )
    config = session.exec(
        select(HotmartIntegrationConfigRecord).where(
            HotmartIntegrationConfigRecord.workspace_id == workspace.id,
            HotmartIntegrationConfigRecord.environment == "sandbox",
        )
    ).one()
    config.status = "connected"
    config.last_health_status = "connected"
    config.last_health_message = "OAuth token exchange succeeded."
    config.last_health_check_at = utc_now()
    session.add(config)
    session.commit()
