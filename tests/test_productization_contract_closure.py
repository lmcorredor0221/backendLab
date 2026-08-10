from __future__ import annotations

from sqlalchemy import create_engine, text

import pytest

from app.services.alembic_runtime_guard import assert_alembic_head_applied, resolve_expected_alembic_heads
from app.services.commercial_event_catalog import (
    load_commercial_event_catalog,
    resolve_commercial_event_catalog_entry,
)


def test_alembic_runtime_guard_accepts_current_head() -> None:
    expected_head = resolve_expected_alembic_heads()[0]
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        connection.execute(text("INSERT INTO alembic_version (version_num) VALUES (:head)"), {"head": expected_head})

    assert_alembic_head_applied(engine)


def test_alembic_runtime_guard_blocks_missing_or_outdated_head() -> None:
    engine = create_engine("sqlite://")
    with pytest.raises(RuntimeError, match="initialized database"):
        assert_alembic_head_applied(engine)

    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        connection.execute(text("INSERT INTO alembic_version (version_num) VALUES ('outdated')"))

    with pytest.raises(RuntimeError, match="out-of-date database"):
        assert_alembic_head_applied(engine)


def test_commercial_event_catalog_is_versioned_and_covers_runtime_events() -> None:
    catalog = load_commercial_event_catalog()
    event_keys = {item["event_key"] for item in catalog["events"]}

    assert catalog["contract_version"] == "commercial-event-catalog.v1"
    assert {"checkout_started", "payment_confirmed", "export_job_ready", "acp_launcher_report_received"} <= event_keys

    resolved = resolve_commercial_event_catalog_entry("access_request_approved")
    assert resolved["catalog_state"] == "pattern_registered"
    assert resolved["schema_version"] == "commercial-event.v1"
