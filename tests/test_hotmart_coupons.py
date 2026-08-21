from __future__ import annotations

from collections.abc import Iterator
import json

import httpx
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialEventRecord,
    HotmartCredentialUpsertRequest,
    HotmartProductMappingUpsertRequest,
    HotmartPromotionCreateRequest,
    HotmartPromotionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.hotmart.coupons import (
    build_hotmart_promotion_metrics,
    create_hotmart_coupon_promotion,
    delete_hotmart_coupon_promotion,
)
from app.services.hotmart.payment_links import upsert_hotmart_product_mapping
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


def _seed_workspace(session: Session) -> tuple[UserRecord, WorkspaceRecord]:
    user = UserRecord(
        email="hotmart-coupons@leanbuilder.local",
        full_name="Hotmart Coupons Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(
        name="Hotmart Coupons Workspace",
        slug=f"hotmart-coupons-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()
    session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    return user, workspace


def _configure_hotmart(
    session: Session,
    workspace: WorkspaceRecord,
    *,
    billing_mode: str = "one_time",
) -> None:
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
            hotmart_product_id="1234567",
            offer_code="111222",
            billing_mode=billing_mode,
            currency="USD",
        ),
    )
    session.commit()


def test_create_hotmart_coupon_promotion_publishes_and_syncs_coupon(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    _configure_hotmart(db_session, workspace)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        if request.method == "POST":
            assert request.url.path == "/products/api/v1/product/1234567/coupon"
            assert request.headers["Authorization"] == "Bearer access-token-value"
            payload = json.loads(request.content.decode("utf-8"))
            assert payload["code"] == "BLACK10"
            assert payload["discount"] == 0.1
            assert payload["offer_ids"] == ["111222"]
            return httpx.Response(201, json={"id": "98765", "coupon_code": "BLACK10", "active": True, "status": "VALID"})
        assert request.method == "GET"
        assert request.url.path == "/products/api/v1/coupon/product/1234567"
        assert request.url.params["code"] == "BLACK10"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "98765",
                        "coupon_code": "BLACK10",
                        "active": True,
                        "discount": 0.1,
                        "status": "VALID",
                    }
                ]
            },
        )

    response = create_hotmart_coupon_promotion(
        db_session,
        workspace_id=workspace.id,
        payload=HotmartPromotionCreateRequest(
            environment="sandbox",
            internal_campaign_key="black-friday",
            internal_product_key="blueprint_pro",
            coupon_code="black10",
            discount_percent=10,
        ),
        actor_user_id=user.id,
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    assert response.coupon_code == "BLACK10"
    assert response.coupon_id == "98765"
    assert response.status == "active"
    assert calls == [
        "POST /security/oauth/token",
        "POST /products/api/v1/product/1234567/coupon",
        "GET /products/api/v1/coupon/product/1234567",
    ]
    stored = db_session.exec(select(HotmartPromotionRecord)).one()
    assert stored.internal_campaign_key == "black-friday"
    assert stored.discount_origin == "provider_coupon"
    assert stored.metadata_payload["hotmart_request_redacted"]["discount"] == 0.1
    event = db_session.exec(select(CommercialEventRecord).where(CommercialEventRecord.event_key == "hotmart_coupon_created")).one()
    assert event.correlation_id == "BLACK10"


def test_subscription_mapping_blocks_coupon_before_hotmart_call(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    _configure_hotmart(db_session, workspace, billing_mode="subscription")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500, json={"unexpected": True})

    with pytest.raises(ValueError, match="subscription products"):
        create_hotmart_coupon_promotion(
            db_session,
            workspace_id=workspace.id,
            payload=HotmartPromotionCreateRequest(
                environment="sandbox",
                internal_product_key="blueprint_pro",
                coupon_code="SUB10",
                discount_percent=10,
            ),
            actor_user_id=user.id,
            transport=httpx.MockTransport(handler),
        )

    assert calls == []
    assert db_session.exec(select(HotmartPromotionRecord)).all() == []


def test_delete_hotmart_coupon_promotion_deletes_remote_and_marks_local_record(db_session: Session) -> None:
    user, workspace = _seed_workspace(db_session)
    _configure_hotmart(db_session, workspace)
    record = HotmartPromotionRecord(
        workspace_id=workspace.id,
        environment="sandbox",
        internal_campaign_key="black-friday",
        internal_product_key="blueprint_pro",
        hotmart_product_id="1234567",
        coupon_id="98765",
        coupon_code="BLACK10",
        discount_percent=10,
        status="active",
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/security/oauth/token":
            return httpx.Response(200, json={"access_token": "access-token-value", "expires_in": 3600})
        assert request.method == "DELETE"
        assert request.url.path == "/products/api/v1/coupon/98765"
        return httpx.Response(204)

    response = delete_hotmart_coupon_promotion(
        db_session,
        workspace_id=workspace.id,
        coupon_ref=str(record.id),
        environment="sandbox",
        actor_user_id=user.id,
        transport=httpx.MockTransport(handler),
    )
    db_session.commit()

    assert response.deleted_remote is True
    assert response.status == "deleted"
    assert calls == ["POST /security/oauth/token", "DELETE /products/api/v1/coupon/98765"]
    stored = db_session.get(HotmartPromotionRecord, record.id)
    assert stored is not None
    assert stored.status == "deleted"
    event = db_session.exec(select(CommercialEventRecord).where(CommercialEventRecord.event_key == "hotmart_coupon_deleted")).one()
    assert event.correlation_id == "BLACK10"


def test_hotmart_promotion_metrics_count_statuses_and_discount_origins(db_session: Session) -> None:
    _, workspace = _seed_workspace(db_session)
    records = [
        HotmartPromotionRecord(
            workspace_id=workspace.id,
            environment="sandbox",
            internal_product_key="blueprint_pro",
            coupon_code="ACTIVE10",
            discount_percent=10,
            status="active",
            discount_origin="provider_coupon",
        ),
        HotmartPromotionRecord(
            workspace_id=workspace.id,
            environment="sandbox",
            internal_product_key="blueprint_pro",
            coupon_code="SYNCERR",
            discount_percent=10,
            status="sync_error",
            discount_origin="provider_coupon",
        ),
        HotmartPromotionRecord(
            workspace_id=workspace.id,
            environment="sandbox",
            internal_product_key="acp",
            coupon_code="UPGRADE-CREDIT",
            discount_percent=0,
            status="active",
            discount_origin="internal_upgrade_credit",
        ),
    ]
    db_session.add_all(records)
    db_session.commit()

    metrics = build_hotmart_promotion_metrics(db_session, workspace_id=workspace.id, environment="sandbox")

    assert metrics.total == 3
    assert metrics.active == 2
    assert metrics.sync_error == 1
    assert metrics.provider_coupon_count == 2
    assert metrics.internal_upgrade_credit_count == 1
