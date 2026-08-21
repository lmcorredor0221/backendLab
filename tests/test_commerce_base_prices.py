from collections.abc import Iterator

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import CommercialPriceStatus, ProductPriceRecord
from app.services import commerce_service


@pytest.fixture()
def db_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    monkeypatch.setattr(
        commerce_service,
        "get_today_trm_data",
        lambda force_refresh=False: {"rate": 3171.93, "date": "2026-08-15"},
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_base_prices_summary_uses_seeded_platform_prices(db_session: Session) -> None:
    response = commerce_service.get_base_prices_summary(db_session)

    assert response.blueprint_pro_usd == 49.0
    assert response.acp_premium_usd == 149.0


def test_base_prices_summary_migrates_legacy_defaults(db_session: Session) -> None:
    db_session.add(
        ProductPriceRecord(
            product_key="blueprint_pro",
            price_code="blueprint-pro-legacy-test",
            currency="USD",
            unit_amount_cents=6000,
            unit_amount_usd_cents=6000,
            billing_period="one_time",
            status=CommercialPriceStatus.active,
        )
    )
    db_session.add(
        ProductPriceRecord(
            product_key="acp",
            price_code="acp-legacy-test",
            currency="USD",
            unit_amount_cents=22000,
            unit_amount_usd_cents=22000,
            billing_period="one_time",
            status=CommercialPriceStatus.active,
        )
    )
    db_session.commit()

    response = commerce_service.get_base_prices_summary(db_session)

    assert response.blueprint_pro_usd == 49.0
    assert response.acp_premium_usd == 149.0
