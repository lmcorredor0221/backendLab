from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from sqlalchemy import event
from sqlalchemy.pool import QueuePool
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialEventRecord,
    HotmartCredentialUpsertRequest,
    HotmartProductMappingUpsertRequest,
    HotmartReconciliationIssueRecord,
    HotmartReconciliationResolveRequest,
    HotmartSyncCursorRecord,
    HotmartSyncRequest,
    HotmartSyncRunRecord,
    HotmartWebhookEventRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.hotmart.payment_links import upsert_hotmart_product_mapping
from app.services.hotmart.secrets import upsert_hotmart_credentials
from app.services.hotmart.sync import (
    replay_hotmart_webhook_event,
    resolve_hotmart_reconciliation_issue,
    run_hotmart_manual_sync,
)


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


def test_hotmart_product_sync_uses_cursor_and_does_not_duplicate_issues(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    _configure_credentials(db_session, workspace)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}?{request.url.query.decode()}")
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        assert request.url.path == "/products/api/v1/products"
        return httpx.Response(
            200,
            json={
                "items": [{"id": 7654321, "name": "Unmapped Hotmart Product", "status": "ACTIVE"}],
                "page_info": {"next_page_token": "cursor-2"},
            },
        )

    first = run_hotmart_manual_sync(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartSyncRequest(environment="sandbox", resource="products"),
        actor_user_id=user.id,
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()
    second = run_hotmart_manual_sync(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartSyncRequest(environment="sandbox", resource="products"),
        actor_user_id=user.id,
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    issues = db_session.exec(select(HotmartReconciliationIssueRecord)).all()
    runs = db_session.exec(select(HotmartSyncRunRecord)).all()
    cursor = db_session.exec(select(HotmartSyncCursorRecord)).one()
    assert first.records_created == 1
    assert second.records_updated == 1
    assert len(issues) == 1
    assert issues[0].issue_type == "hotmart_product_without_mapping"
    assert len(runs) == 2
    assert cursor.page_token == "cursor-2"
    assert any("page_token=cursor-2" in call for call in calls)


def test_hotmart_sales_sync_opens_and_resolves_reconciliation_issue(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    _configure_credentials(db_session, workspace)
    _configure_mapping(db_session, workspace, product_id="1234567")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        assert request.url.path == "/payments/api/v1/sales/history"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "product": {"id": 1234567, "name": "Blueprint Pro"},
                        "transaction": "HP17715690036014",
                        "transaction_status": "APPROVED",
                    }
                ],
                "page_info": {},
            },
        )

    run = run_hotmart_manual_sync(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartSyncRequest(environment="sandbox", resource="sales"),
        actor_user_id=user.id,
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    issue = db_session.exec(select(HotmartReconciliationIssueRecord)).one()
    assert run.records_created == 1
    assert issue.issue_type == "hotmart_payment_without_internal_order"
    assert issue.provider_ref == "HP17715690036014"

    resolved = resolve_hotmart_reconciliation_issue(
        db_session,
        workspace_id=workspace.id,
        issue_id=issue.id,
        payload=HotmartReconciliationResolveRequest(
            resolution_action="linked_manually",
            resolution_note="Linked after validating buyer email.",
        ),
        actor_user_id=user.id,
    )
    db_session.commit()

    assert resolved.status == "resolved"
    assert resolved.resolution_action == "linked_manually"
    event = db_session.exec(
        select(CommercialEventRecord).where(CommercialEventRecord.event_key == "hotmart_reconciliation_resolved")
    ).one()
    assert event.product_key == "hotmart_payment_without_internal_order"


def test_hotmart_manual_sync_releases_db_connection_before_provider_calls(tmp_path) -> None:
    database_path = tmp_path / "hotmart-sync-pool.db"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
    )
    SQLModel.metadata.create_all(engine)
    lifecycle: list[str] = []

    @event.listens_for(engine, "checkout")
    def _record_checkout(*_args) -> None:
        lifecycle.append("checkout")

    @event.listens_for(engine, "checkin")
    def _record_checkin(*_args) -> None:
        lifecycle.append("checkin")

    with Session(engine) as session:
        user, workspace = _seed_workspace(session)
        _configure_credentials(session, workspace)
        lifecycle.clear()

        def handler(request: httpx.Request) -> httpx.Response:
            lifecycle.append(f"http:{request.url.path}")
            if request.url.path == "/security/oauth/token":
                return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
            assert request.url.path == "/products/api/v1/products"
            return httpx.Response(200, json={"items": [], "page_info": {}})

        run = run_hotmart_manual_sync(
            session,
            workspace_id=workspace.id,
            payload=HotmartSyncRequest(environment="sandbox", resource="products"),
            actor_user_id=user.id,
            transport=httpx.MockTransport(handler),
        )

    assert run.status == "succeeded"
    first_http_index = lifecycle.index("http:/security/oauth/token")
    assert "checkin" in lifecycle[:first_http_index]
    assert "checkout" in lifecycle[first_http_index + 1 :]


def test_hotmart_webhook_replay_opens_review_issue(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    webhook = HotmartWebhookEventRecord(
        event_id="webhook-event-1",
        event_type="PURCHASE_APPROVED",
        transaction="HP17715690036014",
        workspace_id=workspace.id,
        hottok_validated=True,
        processing_status="unresolved",
        payload_hash="hash",
        payload_redacted={"event": "PURCHASE_APPROVED"},
    )
    db_session.add(webhook)
    db_session.commit()

    response = replay_hotmart_webhook_event(
        db_session,
        workspace_id=workspace.id,
        event_ref="webhook-event-1",
        environment="sandbox",
        actor_user_id=user.id,
    )
    db_session.commit()

    stored = db_session.get(HotmartWebhookEventRecord, webhook.id)
    issue = db_session.exec(select(HotmartReconciliationIssueRecord)).one()
    assert stored is not None
    assert stored.retries == 1
    assert stored.processing_status == "replay_requested"
    assert response.issue_id == issue.id
    assert issue.issue_type == "webhook_replay_requested"


def _seed_workspace(session: Session) -> tuple[UserRecord, WorkspaceRecord]:
    user = UserRecord(
        email="hotmart-sync@leanbuilder.local",
        full_name="Hotmart Sync Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(
        name="Hotmart Sync Workspace",
        slug=f"hotmart-sync-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    return user, workspace


def _configure_credentials(session: Session, workspace: WorkspaceRecord) -> None:
    upsert_hotmart_credentials(
        session,
        workspace_id=workspace.id,
        payload=HotmartCredentialUpsertRequest(
            environment="sandbox",
            enabled=True,
            client_id="client-id-value",
            client_secret="client-secret-value",
            basic_token="basic-token-value",
        ),
    )
    session.commit()


def _configure_mapping(session: Session, workspace: WorkspaceRecord, *, product_id: str) -> None:
    upsert_hotmart_product_mapping(
        session,
        workspace_id=workspace.id,
        payload=HotmartProductMappingUpsertRequest(
            environment="sandbox",
            internal_product_key="blueprint_pro",
            hotmart_product_id=product_id,
            billing_mode="one_time",
            currency="USD",
        ),
    )
    session.commit()
