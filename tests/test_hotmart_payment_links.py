from __future__ import annotations

from collections.abc import Iterator
import json

import httpx
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialCheckoutSessionRequest,
    CommercialEventRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    HotmartCredentialUpsertRequest,
    HotmartPaymentLinkCreateRequest,
    HotmartPaymentLinkRecord,
    HotmartProductMappingUpsertRequest,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commerce_service import create_checkout_session
from app.services.hotmart.payment_links import (
    HotmartPaymentLinkError,
    create_hotmart_payment_link_for_order,
    refresh_hotmart_payment_link,
    upsert_hotmart_product_mapping,
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


def _seed_checkout_context(session: Session) -> tuple[UserRecord, WorkspaceRecord, SessionRecord]:
    user = UserRecord(
        email="hotmart-links@leanbuilder.local",
        full_name="Hotmart Links Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(
        name="Hotmart Links Workspace",
        slug=f"hotmart-links-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Hotmart Links Project")
    session.add(record)
    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    session.refresh(record)
    return user, workspace, record


def _configure_hotmart(session: Session, workspace: WorkspaceRecord) -> None:
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
    upsert_hotmart_product_mapping(
        session,
        workspace_id=workspace.id,
        payload=HotmartProductMappingUpsertRequest(
            environment="sandbox",
            internal_product_key="blueprint_pro",
            hotmart_product_ucode="hotmart-product-ucode",
            currency="USD",
        ),
    )
    session.commit()


def _create_hotmart_order(session: Session, record: SessionRecord, user: UserRecord) -> CommercialOrderRecord:
    checkout = create_checkout_session(
        session,
        payload=CommercialCheckoutSessionRequest(
            session_id=record.id,
            product_key="blueprint_pro",
            provider="hotmart",
            idempotency_key=f"{record.id}:hotmart-link-order",
        ),
        record=record,
        current_user=user,
        base_url="http://localhost:3200",
    )
    session.commit()
    return session.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.id == checkout.order_id)).one()


def test_create_hotmart_payment_link_from_mapped_pending_order(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hotmart(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        assert request.url.path == "/payments/api/v1/payment-links"
        assert request.headers["Authorization"] == "Bearer access-token-value"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["name"] == f"blueprint_pro-{order.checkout_ref}"
        assert payload["value"] == order.total_cents / 100
        assert payload["currency"] == "USD"
        assert payload["link_configuration"]["link_callback_url"] == "https://example.test/hotmart/webhook"
        assert payload["product"] == {"ucode": "hotmart-product-ucode"}
        return httpx.Response(
            201,
            json={
                "ucode": "pl-ucode-123",
                "url": "https://pay.hotmart.test/pl-ucode-123",
            },
        )

    response = create_hotmart_payment_link_for_order(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartPaymentLinkCreateRequest(
            order_id=order.id,
            environment="sandbox",
            callback_url="https://example.test/hotmart/webhook",
        ),
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    refreshed_order = db_session.get(CommercialOrderRecord, order.id)
    assert refreshed_order is not None
    assert response.hotmart_payment_link_id == "pl-ucode-123"
    assert response.checkout_url == "https://pay.hotmart.test/pl-ucode-123"
    assert response.activation_status == "pending_activation"
    assert response.gross_amount_cents == order.subtotal_cents
    assert response.net_amount_cents == order.total_cents
    assert response.discount_amount_cents == 0
    assert refreshed_order.status == CommercialOrderStatus.pending
    assert refreshed_order.checkout_url == "https://pay.hotmart.test/pl-ucode-123"
    assert refreshed_order.metadata_payload["hotmart_payment_link_activation_status"] == "pending_activation"
    assert calls == ["/security/oauth/token", "/payments/api/v1/payment-links"]

    stored = db_session.exec(select(HotmartPaymentLinkRecord)).one()
    serialized = str(stored.request_payload_redacted) + str(stored.response_payload_redacted)
    assert "access-token-value" not in serialized
    assert "client-secret-value" not in serialized
    trace = stored.request_payload_redacted["_lab_payment_link_trace"]
    trace_steps = [entry["step"] for entry in trace]
    assert "create_hotmart_payment_link_for_order.started" in trace_steps
    assert "order.resolved" in trace_steps
    assert "mapping.resolved" in trace_steps
    assert "oauth.succeeded" in trace_steps
    assert "provider_payment_link.succeeded" in trace_steps
    assert "payment_link_record.persisted" in trace_steps
    assert "commercial_event.recorded" in trace_steps
    assert "_lab_payment_link_http_diagnostics" in stored.response_payload_redacted
    event = db_session.exec(
        select(CommercialEventRecord).where(CommercialEventRecord.event_key == "hotmart_payment_link_pending_activation")
    ).one()
    assert event.correlation_id == order.checkout_ref


def test_create_hotmart_payment_link_uses_platform_integration_scope(db_session: Session) -> None:
    user, customer_workspace, record = _seed_checkout_context(db_session)
    platform_workspace = WorkspaceRecord(name="Platform Hotmart Scope", slug="platform-hotmart-scope")
    db_session.add(platform_workspace)
    db_session.commit()
    db_session.refresh(platform_workspace)
    _configure_hotmart(db_session, platform_workspace)
    order = _create_hotmart_order(db_session, record, user)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "platform-access-token", "expires_in": 3600})
        return httpx.Response(201, json={"ucode": "platform-link-1", "url": "https://pay.hotmart.test/platform-link-1"})

    response = create_hotmart_payment_link_for_order(
        db_session,
        workspace_id=customer_workspace.id,
        integration_workspace_id=platform_workspace.id,
        payload=HotmartPaymentLinkCreateRequest(
            order_id=order.id,
            environment="sandbox",
            callback_url="https://example.test/hotmart/webhook",
        ),
        transport=httpx.MockTransport(handler),
    )

    assert response.workspace_id == customer_workspace.id
    assert response.checkout_url == "https://pay.hotmart.test/platform-link-1"


def test_create_hotmart_payment_link_includes_product_offer_and_environment(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    upsert_hotmart_credentials(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartCredentialUpsertRequest(
            environment="production",
            enabled=True,
            client_id="client-id-value",
            client_secret="client-secret-value",
            basic_token="basic-token-value",
        ),
    )
    upsert_hotmart_product_mapping(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartProductMappingUpsertRequest(
            environment="production",
            internal_product_key="blueprint_pro",
            hotmart_product_id="8322629",
            offer_code="nb6j5s6j",
            currency="USD",
        ),
    )
    db_session.commit()
    order = _create_hotmart_order(db_session, record, user)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["product"] == {"id": "8322629"}
        assert payload["offer"] == {"code": "nb6j5s6j"}
        assert "plan" not in payload
        return httpx.Response(201, json={"ucode": "pl-prod-123", "url": "https://pay.hotmart.test/pl-prod-123"})

    response = create_hotmart_payment_link_for_order(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartPaymentLinkCreateRequest(
            order_id=order.id,
            environment="production",
            callback_url="https://example.test/hotmart/webhook",
        ),
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    stored = db_session.exec(select(HotmartPaymentLinkRecord)).one()
    assert response.checkout_url == "https://pay.hotmart.test/pl-prod-123"
    assert stored.environment == "production"


def test_hotmart_payment_link_error_keeps_order_pending_and_records_failed_attempt(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hotmart(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        return httpx.Response(400, json={"error": "invalid_parameter", "client_secret": "must-redact"})

    with pytest.raises(HotmartPaymentLinkError):
        create_hotmart_payment_link_for_order(
            db_session,
            workspace_id=workspace.id,
            payload=HotmartPaymentLinkCreateRequest(
                order_id=order.id,
                environment="sandbox",
                callback_url="https://example.test/hotmart/webhook",
            ),
            transport=httpx.MockTransport(handler),
        )
    db_session.commit()

    refreshed_order = db_session.get(CommercialOrderRecord, order.id)
    failed = db_session.exec(select(HotmartPaymentLinkRecord)).one()
    assert refreshed_order is not None
    assert refreshed_order.status == CommercialOrderStatus.pending
    assert refreshed_order.checkout_url == ""
    assert failed.environment == "sandbox"
    assert failed.activation_status == "failed"
    assert failed.response_payload_redacted["client_secret"] == "[redacted]"
    assert failed.response_payload_redacted["_lab_payment_link_error"]["error_code"] == "payment_link_rejected"
    trace_steps = [entry["step"] for entry in failed.request_payload_redacted["_lab_payment_link_trace"]]
    assert "provider_payment_link.failed" in trace_steps
    assert "failed_attempt.persisted" in trace_steps
    assert "access-token-value" not in str(failed.request_payload_redacted)
    assert "must-redact" not in str(failed.response_payload_redacted)


def test_hotmart_oauth_error_keeps_order_pending_and_records_failed_attempt(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hotmart(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/security/oauth/token"
        return httpx.Response(
            401,
            json={
                "error": "unauthorized_client",
                "error_description": "The user does not have permission to continue with this request.",
                "client_secret": "must-redact",
            },
        )

    with pytest.raises(HotmartPaymentLinkError) as exc_info:
        create_hotmart_payment_link_for_order(
            db_session,
            workspace_id=workspace.id,
            payload=HotmartPaymentLinkCreateRequest(
                order_id=order.id,
                environment="sandbox",
                callback_url="https://example.test/hotmart/webhook",
            ),
            transport=httpx.MockTransport(handler),
        )
    db_session.commit()

    refreshed_order = db_session.get(CommercialOrderRecord, order.id)
    failed = db_session.exec(select(HotmartPaymentLinkRecord)).one()
    assert exc_info.value.code == "oauth_rejected"
    assert exc_info.value.http_status == 401
    assert refreshed_order is not None
    assert refreshed_order.status == CommercialOrderStatus.pending
    assert failed.activation_status == "failed"
    assert failed.response_payload_redacted["error"] == "unauthorized_client"
    assert failed.response_payload_redacted["client_secret"] == "[redacted]"
    assert failed.response_payload_redacted["_lab_payment_link_error"]["error_code"] == "oauth_rejected"
    trace_steps = [entry["step"] for entry in failed.request_payload_redacted["_lab_payment_link_trace"]]
    assert "credentials.loaded" in trace_steps
    assert "provider_payment_link.failed" in trace_steps
    assert "failed_attempt.persisted" in trace_steps


def test_refresh_hotmart_payment_link_marks_available_link_active(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    _configure_hotmart(db_session, workspace)
    order = _create_hotmart_order(db_session, record, user)

    def create_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        return httpx.Response(201, json={"ucode": "pl-ucode-123", "url": "https://pay.hotmart.test/pl-ucode-123"})

    response = create_hotmart_payment_link_for_order(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartPaymentLinkCreateRequest(
            order_id=order.id,
            environment="sandbox",
            callback_url="https://example.test/hotmart/webhook",
        ),
        transport=httpx.MockTransport(create_handler),
    )
    db_session.commit()

    def refresh_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={"items": [{"ucode": "pl-ucode-123", "url": "https://pay.hotmart.test/pl-ucode-123"}]},
        )

    refreshed = refresh_hotmart_payment_link(
        db_session,
        workspace_id=workspace.id,
        payment_link_id=response.id,
        environment="sandbox",
        transport=httpx.MockTransport(refresh_handler),
    )
    db_session.commit()

    assert refreshed.activation_status == "active"
    order_after = db_session.get(CommercialOrderRecord, order.id)
    assert order_after is not None
    assert order_after.metadata_payload["hotmart_payment_link_activation_status"] == "active"


def test_hotmart_payment_link_uses_frozen_snapshot_for_cop_checkout(db_session: Session) -> None:
    user, workspace, record = _seed_checkout_context(db_session)
    upsert_hotmart_credentials(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartCredentialUpsertRequest(
            environment="sandbox",
            enabled=True,
            client_id="client-id-value",
            client_secret="client-secret-value",
            basic_token="basic-token-value",
        ),
    )
    upsert_hotmart_product_mapping(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartProductMappingUpsertRequest(
            environment="sandbox",
            internal_product_key="blueprint_pro",
            hotmart_product_ucode="hotmart-product-ucode",
            currency="COP",
        ),
    )
    db_session.commit()
    order = _create_hotmart_order(db_session, record, user)
    snapshot = dict(order.metadata_payload["commercial_snapshot"])
    snapshot["trm_cop_frozen"] = 4000.0
    order.metadata_payload = {**order.metadata_payload, "commercial_snapshot": snapshot}
    db_session.add(order)
    db_session.commit()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["currency"] == "COP"
        assert payload["value"] == 196000.0
        return httpx.Response(201, json={"ucode": "pl-cop-123", "url": "https://pay.hotmart.test/pl-cop-123"})

    response = create_hotmart_payment_link_for_order(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartPaymentLinkCreateRequest(
            order_id=order.id,
            environment="sandbox",
            callback_url="https://example.test/hotmart/webhook",
        ),
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    assert response.currency == "COP"
    assert response.net_amount_cents == 19600000
