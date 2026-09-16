from __future__ import annotations

from uuid import uuid4

from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    CommercialEventRecord,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    MarketingAnalyticsOutboxRecord,
    SessionRecord,
    UserRecord,
    WorkspaceRecord,
)
from app.services.marketing_analytics_service import enqueue_from_commercial_event, sanitize_marketing_context


def _session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _seed_paid_order(db: Session, *, provider: str = "mercadopago") -> tuple[SessionRecord, CommercialOrderRecord]:
    user = UserRecord(email=f"user-{uuid4()}@example.com", full_name="LAB User", password_hash="hash")
    workspace = WorkspaceRecord(name="LAB", slug=f"lab-{uuid4().hex[:8]}", created_by_user_id=user.id)
    db.add(user)
    db.add(workspace)
    db.flush()
    context = sanitize_marketing_context(
        {
            "consent": {
                "analytics_storage": "granted",
                "ad_storage": "granted",
                "ad_user_data": "granted",
                "ad_personalization": "granted",
                "version": 1,
            },
            "ga_client_id": "12345.67890",
            "ga_session_id": "999",
            "attribution": {
                "captured_at": "2026-09-16T10:00:00Z",
                "expires_at": "2026-10-16T10:00:00Z",
                "last_touch": {"utm_source": "google", "utm_campaign": "lab"},
            },
        }
    )
    project = SessionRecord(user_id=user.id, workspace_id=workspace.id, marketing_context=context)
    db.add(project)
    db.flush()
    order = CommercialOrderRecord(
        workspace_id=workspace.id,
        session_id=project.id,
        buyer_user_id=user.id,
        provider=provider,
        checkout_ref=f"chk-{uuid4().hex[:8]}",
        currency="COP",
        total_cents=120000,
        marketing_context=context,
        metadata_payload={"provider_checkout_id": "mp-order-1", "product_key": "blueprint_pro"},
    )
    db.add(order)
    db.flush()
    db.add(
        CommercialOrderLineRecord(
            order_id=order.id,
            product_key="blueprint_pro",
            price_code="bp-pro-cop",
            quantity=1,
            total_amount_cents=120000,
        )
    )
    db.flush()
    return project, order


def test_purchase_event_creates_single_ga4_outbox_record():
    with _session() as db:
        project, order = _seed_paid_order(db)
        event = CommercialEventRecord(
            workspace_id=order.workspace_id,
            session_id=project.id,
            user_id=project.user_id,
            event_key="mercadopago_order_processed",
            product_key="blueprint_pro",
            source="mercadopago_webhook",
            correlation_id="mp-event-1",
            revenue_cents=120000,
            currency="COP",
            metadata_payload={"order_id": str(order.id), "email": "must-not-leak@example.com"},
        )
        db.add(event)
        db.flush()

        enqueue_from_commercial_event(db, event)
        enqueue_from_commercial_event(db, event)

        records = db.exec(select(MarketingAnalyticsOutboxRecord)).all()
        assert len(records) == 1
        payload = records[0].payload
        assert payload["client_id"] == "12345.67890"
        assert payload["events"][0]["name"] == "purchase"
        assert payload["events"][0]["params"]["transaction_id"] == "mp-order-1"
        assert "must-not-leak" not in str(payload)


def test_sandbox_provider_is_not_enqueued_as_revenue():
    with _session() as db:
        project, order = _seed_paid_order(db, provider="sandbox")
        event = CommercialEventRecord(
            workspace_id=order.workspace_id,
            session_id=project.id,
            user_id=project.user_id,
            event_key="mercadopago_order_processed",
            product_key="blueprint_pro",
            source="sandbox",
            correlation_id="sandbox-event",
            revenue_cents=120000,
            currency="COP",
            metadata_payload={"order_id": str(order.id)},
        )
        db.add(event)
        db.flush()

        assert enqueue_from_commercial_event(db, event) is None
        assert db.exec(select(MarketingAnalyticsOutboxRecord)).all() == []
