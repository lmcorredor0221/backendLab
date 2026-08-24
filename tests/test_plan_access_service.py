from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    AccessRequestCreateRequest,
    CommercialOrderLineRecord,
    CommercialOrderRecord,
    CommercialOrderStatus,
    CommercialPackageCatalogUpsertRequest,
    CommercialPackageType,
    CommercialQuotaSourceKind,
    CommercialTier,
    SessionRecord,
    UserRecord,
)
from app.services.auth_service import hash_password
from app.services.commerce_service import request_access
from app.services.commercial_catalog_service import upsert_package_catalog_entry
from app.services.commercial_debt_service import create_commercial_debt
from app.services.commercial_quota_service import grant_balance_units
from app.services.plan_access_service import build_workspace_commercial_summary
from app.services.workspace_access import ensure_personal_workspace


def _db_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _seed_project_context(session: Session, *, email: str) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=email,
        full_name=email.split("@")[0],
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    workspace = ensure_personal_workspace(session, user).workspace
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="Workspace Commercial Summary",
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return user, record


def test_build_workspace_commercial_summary_exposes_balance_offers_orders_debts_and_requests() -> None:
    with _db_session() as session:
        user, record = _seed_project_context(session, email="plan-access-summary@leanbuilder.local")
        grant_balance_units(
            session,
            workspace_id=record.workspace_id,
            product_key="blueprint_pro",
            source_kind=CommercialQuotaSourceKind.one_time,
            units=2,
            bucket_key="bp-pack",
            source_ref="order:bp-pack",
            actor_user_id=user.id,
        )
        request_access(
            session,
            payload=AccessRequestCreateRequest(
                session_id=record.id,
                capability="acp.build",
                reason="Continuar ACP",
            ),
            record=record,
            current_user=user,
            product_key="acp",
            target_tier=CommercialTier.acp,
        )
        create_commercial_debt(
            session,
            workspace_id=record.workspace_id,
            product_key="acp",
            access_request_id=None,
            amount_cents=15000,
            currency="USD",
            actor_user_id=user.id,
            reason_code="debt_pending",
            reason_label="Deuda pendiente",
            summary="Aprobacion comercial con deuda pendiente.",
        )
        upsert_package_catalog_entry(
            session,
            payload=CommercialPackageCatalogUpsertRequest(
                package_code="acp_monthly",
                display_name="ACP Mensual",
                product_key="acp",
                package_type=CommercialPackageType.subscription,
                enabled=True,
                granted_units=0,
                granted_units_blueprint_pro=0,
                granted_units_acp=3,
                recommendation_priority=10,
                hotmart_environment="sandbox",
                hotmart_product_id="hm-acp-monthly",
                hotmart_product_ucode="ucode-acp-monthly",
                offer_code="offer-acp-monthly",
                plan_code="plan-acp-monthly",
            ),
        )
        order = CommercialOrderRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            buyer_user_id=user.id,
            status=CommercialOrderStatus.pending,
            currency="USD",
            subtotal_cents=30000,
            total_cents=30000,
            provider="hotmart",
            checkout_ref="checkout-acp-1",
            checkout_url="https://checkout.example.com/acp-1",
            idempotency_key="order-acp-1",
            metadata_payload={"product_key": "acp", "price_code": "acp-premium-usd-v1"},
        )
        session.add(order)
        session.flush()
        session.add(
            CommercialOrderLineRecord(
                order_id=order.id,
                product_key="acp",
                price_code="acp-premium-usd-v1",
                quantity=1,
                unit_amount_cents=30000,
                total_amount_cents=30000,
            )
        )
        session.commit()

        summary = build_workspace_commercial_summary(
            session,
            workspace_id=record.workspace_id,
            session_id=record.id,
        )

        blueprint_summary = next(item for item in summary.products if item.product_key == "blueprint_pro")
        acp_summary = next(item for item in summary.products if item.product_key == "acp")

        assert blueprint_summary.available_units == 2
        assert blueprint_summary.one_time_units == 2
        assert acp_summary.available_units == 0
        assert acp_summary.pending_request_count == 1
        assert acp_summary.open_debt_count == 1
        assert acp_summary.recommendation is not None
        assert acp_summary.recommendation.package_code == "acp_monthly"
        assert summary.open_debts[0].reason_label == "Deuda pendiente"
        assert summary.recent_orders[0].checkout_url == "https://checkout.example.com/acp-1"
        assert summary.request_history[0].capability == "acp.build"
        assert summary.request_history[0].status == "pending"
