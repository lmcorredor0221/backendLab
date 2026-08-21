from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialCheckoutSessionRequest,
    CommercialOrderRecord,
    CommercialOrderStatus,
    HotmartPaymentLinkRecord,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commerce_provider_router import (
    get_commerce_payment_provider,
    normalize_commerce_payment_provider,
)
from app.services.commerce_service import create_checkout_session


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


def _seed_checkout_context(session: Session) -> tuple[UserRecord, WorkspaceRecord, SessionRecord]:
    user = UserRecord(
        email="commerce-provider@leanbuilder.local",
        full_name="Commerce Provider Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(
        name="Commerce Provider Workspace",
        slug=f"commerce-provider-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Commerce Provider Project")
    session.add(record)
    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    session.refresh(record)
    return user, workspace, record


def test_commerce_provider_router_normalizes_supported_providers() -> None:
    assert normalize_commerce_payment_provider(None) == "sandbox"
    assert normalize_commerce_payment_provider("default") == "sandbox"
    assert normalize_commerce_payment_provider("HOTMART") == "hotmart"
    assert get_commerce_payment_provider("sandbox").provider_key == "sandbox"
    assert get_commerce_payment_provider("hotmart").provider_key == "hotmart"

    with pytest.raises(ValueError, match="Unsupported commerce checkout provider"):
        normalize_commerce_payment_provider("stripe")


def test_sandbox_checkout_provider_preserves_existing_order_flow(db_session: Session) -> None:
    user, _, record = _seed_checkout_context(db_session)

    response = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            idempotency_key=f"{record.id}:sandbox-provider",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    assert response.provider == "sandbox"
    assert response.checkout_ref.startswith("sandbox_")
    assert response.checkout_url.endswith(f"/checkout/sandbox/{response.checkout_ref}")
    assert response.next_action == "open_checkout"
    assert order.provider == "sandbox"
    assert order.status == CommercialOrderStatus.pending
    assert order.metadata_payload["provider_stage"] == "sandbox_checkout"


def test_hotmart_checkout_provider_creates_pending_order_without_payment_link(db_session: Session) -> None:
    user, _, record = _seed_checkout_context(db_session)

    response = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="hotmart",
            idempotency_key=f"{record.id}:hotmart-provider",
            success_url="https://example.test/success",
            cancel_url="https://example.test/cancel",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    order = db_session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == response.order_id)).one()
    payment_links = db_session.exec(select(HotmartPaymentLinkRecord)).all()
    assert response.provider == "hotmart"
    assert response.checkout_ref.startswith("hotmart_")
    assert response.checkout_url == ""
    assert response.status == CommercialOrderStatus.pending
    assert response.next_action == "await_payment_link"
    assert order.provider == "hotmart"
    assert order.status == CommercialOrderStatus.pending
    assert order.total_cents > 0
    assert order.metadata_payload["provider_stage"] == "hotmart_order_pending_payment_link"
    assert order.metadata_payload["requires_payment_link"] is True
    assert order.metadata_payload["payment_link_stage"] == "stage_3"
    assert payment_links == []


def test_checkout_rejects_open_redirect_urls(db_session: Session) -> None:
    user, _, record = _seed_checkout_context(db_session)

    with pytest.raises(ValueError, match="Redirect URL host is not permitted"):
        create_checkout_session(
            db_session,
            payload=CommercialCheckoutSessionRequest(
                session_id=record.id,
                product_key="blueprint_pro",
                success_url="https://phishing-site.example.org/steal",
            ),
            record=record,
            current_user=user,
            base_url="http://localhost:3200",
        )

    with pytest.raises(ValueError, match="Invalid redirect URL format"):
        create_checkout_session(
            db_session,
            payload=CommercialCheckoutSessionRequest(
                session_id=record.id,
                product_key="blueprint_pro",
                success_url="javascript:alert(1)",
            ),
            record=record,
            current_user=user,
            base_url="http://localhost:3200",
        )


def test_checkout_enforces_idempotency(db_session: Session) -> None:
    user, _, record = _seed_checkout_context(db_session)
    idempotency_key = f"{record.id}:idempotent-checkout-test"

    response1 = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            idempotency_key=idempotency_key,
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    db_session.commit()

    response2 = create_checkout_session(
        db_session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            idempotency_key=idempotency_key,
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )

    assert response1.order_id == response2.order_id
    assert response1.checkout_ref == response2.checkout_ref
    orders = db_session.exec(
        select(CommercialOrderRecord).where(CommercialOrderRecord.idempotency_key == idempotency_key)
    ).all()
    assert len(orders) == 1


def test_checkout_enforces_buyer_permission(db_session: Session) -> None:
    _, workspace, record = _seed_checkout_context(db_session)
    viewer = UserRecord(
        email="viewer@leanbuilder.local",
        full_name="Viewer User",
        password_hash=hash_password("Secret123!"),
    )
    db_session.add(viewer)
    db_session.flush()
    db_session.add(
        WorkspaceMembershipRecord(
            workspace_id=workspace.id,
            user_id=viewer.id,
            role=WorkspaceRole.viewer,
        )
    )
    db_session.commit()

    with pytest.raises(PermissionError, match="Only workspace owners or admins can start checkout"):
        create_checkout_session(
            db_session,
            payload=CommercialCheckoutSessionRequest(
                session_id=record.id,
                product_key="blueprint_pro",
            ),
            record=record,
            current_user=viewer,
            base_url="http://localhost:3200",
        )


