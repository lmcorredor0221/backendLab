from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialEntitlementRecord,
    CommercialEntitlementStatus,
    CommercialOrderRecord,
    CommercialOrderStatus,
    HotmartClubSyncRequest,
    HotmartCredentialUpsertRequest,
    HotmartReconciliationIssueRecord,
    HotmartSyncCursorRecord,
    HotmartSyncRunRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.hotmart.club import (
    get_hotmart_club_overview,
    list_hotmart_club_modules,
    list_hotmart_club_pages,
    list_hotmart_club_progress,
    list_hotmart_club_students,
    sync_hotmart_club,
)
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


def test_hotmart_club_sync_stores_snapshot_and_opens_student_access_issue(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    _configure_credentials(db_session, workspace)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        if request.url.path == "/club/api/v1/modules":
            assert request.url.params["subdomain"] == "leanclub"
            return httpx.Response(200, json={"items": [{"id": "module-1", "name": "Primeros pasos", "total_pages": 1}]})
        if request.url.path == "/club/api/v1/modules/module-1/pages":
            assert request.url.params["subdomain"] == "leanclub"
            return httpx.Response(200, json={"items": [{"id": "page-1", "name": "Bienvenida", "page_order": 1, "type": "video"}]})
        if request.url.path == "/club/api/v1/users":
            assert request.url.params["subdomain"] == "leanclub"
            return httpx.Response(
                200,
                json={"items": [{"id": "student-1", "name": "Student One", "email": "student@example.com", "status": "ACTIVE"}]},
            )
        if request.url.path == "/club/api/v1/users/student-1/lessons":
            assert request.url.params["subdomain"] == "leanclub"
            return httpx.Response(
                200,
                json={"items": [{"id": "lesson-1", "name": "Bienvenida", "completed": True, "completed_at": "2026-08-14T10:00:00Z"}]},
            )
        return httpx.Response(404, json={"error": f"Unexpected {request.url.path}"})

    run = sync_hotmart_club(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartClubSyncRequest(environment="sandbox", subdomain="leanclub", sync_progress=True),
        actor_user_id=user.id,
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    overview = get_hotmart_club_overview(db_session, workspace_id=workspace.id, environment="sandbox")
    issue = db_session.exec(select(HotmartReconciliationIssueRecord)).one()
    cursor = db_session.exec(select(HotmartSyncCursorRecord)).one()
    runs = db_session.exec(select(HotmartSyncRunRecord)).all()

    assert run.status == "succeeded"
    assert run.records_read == 4
    assert run.records_created == 1
    assert overview.subdomain == "leanclub"
    assert overview.modules_count == 1
    assert overview.pages_count == 1
    assert overview.students_count == 1
    assert overview.progress_count == 1
    assert overview.open_issue_count == 1
    assert issue.issue_type == "club_student_without_internal_access"
    assert issue.provider_ref == "student-1"
    assert cursor.resource == "club"
    assert len(runs) == 1
    assert list_hotmart_club_modules(db_session, workspace_id=workspace.id)[0].name == "Primeros pasos"
    assert list_hotmart_club_pages(db_session, workspace_id=workspace.id)[0].module_id == "module-1"
    assert list_hotmart_club_students(db_session, workspace_id=workspace.id)[0].email == "student@example.com"
    assert list_hotmart_club_progress(db_session, workspace_id=workspace.id)[0].completed is True


def test_hotmart_club_student_with_active_internal_entitlement_does_not_open_issue(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    _configure_credentials(db_session, workspace)
    buyer = _seed_buyer_with_entitlement(db_session, workspace, email="student@example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        assert request.url.path == "/club/api/v1/users"
        return httpx.Response(200, json={"items": [{"id": "student-1", "email": buyer.email, "status": "ACTIVE"}]})

    run = sync_hotmart_club(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartClubSyncRequest(
            environment="sandbox",
            subdomain="leanclub",
            sync_modules=False,
            sync_pages=False,
            sync_students=True,
        ),
        actor_user_id=user.id,
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    assert run.records_created == 0
    assert db_session.exec(select(HotmartReconciliationIssueRecord)).all() == []


def test_hotmart_club_sync_opens_issue_for_internal_access_missing_in_club(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    _configure_credentials(db_session, workspace)
    buyer = _seed_buyer_with_entitlement(db_session, workspace, email="buyer@example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        assert request.url.path == "/club/api/v1/users"
        return httpx.Response(200, json={"items": []})

    run = sync_hotmart_club(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartClubSyncRequest(
            environment="sandbox",
            subdomain="leanclub",
            sync_modules=False,
            sync_pages=False,
            sync_students=True,
        ),
        actor_user_id=user.id,
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    issue = db_session.exec(select(HotmartReconciliationIssueRecord)).one()
    assert run.records_created == 1
    assert issue.issue_type == "internal_access_without_club_student"
    assert issue.provider_ref == buyer.email


def _seed_workspace(session: Session) -> tuple[UserRecord, WorkspaceRecord]:
    user = UserRecord(
        email="hotmart-club@leanbuilder.local",
        full_name="Hotmart Club Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(
        name="Hotmart Club Workspace",
        slug=f"hotmart-club-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    return user, workspace


def _seed_buyer_with_entitlement(session: Session, workspace: WorkspaceRecord, *, email: str) -> UserRecord:
    buyer = UserRecord(
        email=email,
        full_name="Hotmart Buyer",
        password_hash=hash_password("Secret123!"),
    )
    session.add(buyer)
    session.flush()
    order = CommercialOrderRecord(
        workspace_id=workspace.id,
        buyer_user_id=buyer.id,
        status=CommercialOrderStatus.paid,
        currency="USD",
        provider="hotmart",
        checkout_ref=f"club-order-{str(buyer.id)[:8]}",
        idempotency_key=f"club-order-{str(buyer.id)}",
        total_cents=10000,
    )
    session.add(order)
    session.flush()
    session.add(
        CommercialEntitlementRecord(
            workspace_id=workspace.id,
            product_key="blueprint_pro",
            status=CommercialEntitlementStatus.active,
            order_id=order.id,
        )
    )
    session.commit()
    session.refresh(buyer)
    return buyer


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
