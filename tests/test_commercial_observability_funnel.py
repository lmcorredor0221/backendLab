from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    CommercialEventRecord,
    CommercialTier,
    ProjectTitleSource,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commercial_observability_service import (
    _sanitize_value,
    build_commercial_audit_report,
)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_sensitive_data_redaction() -> None:
    sensitive_payload = {
        "api_key": "sk-live-123456789",
        "password": "SuperSecretPassword123!",
        "auth_token": "bearer-token-val",
        "public_data": "visible_metadata",
        "nested": {
            "prompt": "Instructions for agent",
            "count": 42,
        },
    }

    sanitized = _sanitize_value(sensitive_payload)

    assert sanitized["api_key"] == "[redacted]"
    assert sanitized["password"] == "[redacted]"
    assert sanitized["auth_token"] == "[redacted]"
    assert sanitized["public_data"] == "visible_metadata"
    assert sanitized["nested"]["prompt"] == "[redacted]"
    assert sanitized["nested"]["count"] == 42


def test_commercial_funnel_reconstruction(db_session: Session) -> None:
    user = UserRecord(
        email=f"funnel-{uuid4().hex[:6]}@leanbuilder.local",
        full_name="Funnel Tester",
        password_hash=hash_password("Secret123!"),
    )
    db_session.add(user)
    db_session.flush()

    workspace = WorkspaceRecord(
        name="Funnel Workspace",
        slug=f"funnel-ws-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    db_session.add(workspace)
    db_session.flush()

    db_session.add(
        WorkspaceMembershipRecord(
            workspace_id=workspace.id,
            user_id=user.id,
            role=WorkspaceRole.owner,
        )
    )

    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="Funnel Project",
        title_source=ProjectTitleSource.manual,
        current_stage=SessionStage.ready_for_export,
        commercial_tier=CommercialTier.acp,
    )
    db_session.add(record)
    db_session.flush()

    # Record sequence of events
    events = [
        ("registration_completed", "auth", {}),
        ("project_created", "workspace", {}),
        ("blueprint_basic_completed", "product_build", {}),
        ("checkout_created", "commerce_checkout", {"product_key": "blueprint_pro"}),
        ("payment_confirmed", "commerce_webhook", {"product_key": "blueprint_pro"}),
        ("pro_build_completed", "product_build", {}),
        ("acp_phase_completed", "acp_workflow", {"phase_key": "blueprint_validation"}),
        ("export_downloaded", "export_delivery", {"artifact_kind": "acp_portable_zip"}),
    ]

    for event_key, source, metadata in events:
        db_session.add(
            CommercialEventRecord(
                workspace_id=workspace.id,
                session_id=record.id,
                user_id=user.id,
                event_key=event_key,
                source=source,
                tier=CommercialTier.acp,
                event_metadata=metadata,
            )
        )
    db_session.commit()

    report = build_commercial_audit_report(db_session, record=record, current_user=user)

    assert report.session_id == record.id
    assert report.workspace_id == workspace.id
    assert len(report.funnel) > 0
    assert len(report.recent_events) >= len(events)
