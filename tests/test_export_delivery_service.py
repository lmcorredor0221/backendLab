from __future__ import annotations

import hashlib
from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    AgenticEstimate,
    CommercialAccessSnapshotV2,
    CommercialCapabilityDecisionEntry,
    CommercialEntitlementRecord,
    CommercialEntitlementSource,
    CommercialEntitlementStatus,
    CommercialTier,
    ConfidenceBreakdown,
    EstimationConfidenceLabel,
    EstimationConstructionScenario,
    EstimationReportArtifact,
    ExportJobCreateRequest,
    ExportJobStatus,
    ProjectTitleSource,
    SessionRecord,
    SessionStage,
    TraditionalEstimate,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commercial_access import build_commercial_access_snapshot_v2
from app.services import export_delivery_service
from app.services.export_delivery_service import (
    create_export_job,
    get_export_job_response,
    read_export_job_bytes,
    retry_export_job_response,
)
from app.api.routes.sessions import build_snapshot, resolve_acp_preview


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


def _seed_user_and_session(
    session: Session,
    *,
    tier: CommercialTier = CommercialTier.blueprint_pro,
    role: WorkspaceRole = WorkspaceRole.owner,
) -> tuple[UserRecord, WorkspaceRecord, SessionRecord]:
    user = UserRecord(
        email=f"export-tester-{uuid4().hex[:6]}@leanbuilder.local",
        full_name="Export Tester",
        password_hash=hash_password("Secret123!"),
    )
    session.add(user)
    session.flush()

    workspace = WorkspaceRecord(
        name="Export Workspace",
        slug=f"export-ws-{str(user.id)[:8]}",
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.flush()

    session.add(
        WorkspaceMembershipRecord(
            workspace_id=workspace.id,
            user_id=user.id,
            role=role,
        )
    )

    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="Agentic Pro Project",
        title_source=ProjectTitleSource.manual,
        current_stage=SessionStage.ready_for_export,
        commercial_tier=tier,
    )
    session.add(record)
    session.flush()

    if tier in {CommercialTier.blueprint_pro, CommercialTier.acp}:
        entitlement = CommercialEntitlementRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            user_id=user.id,
            product_key="blueprint_pro" if tier == CommercialTier.blueprint_pro else "acp",
            tier=tier,
            status=CommercialEntitlementStatus.active,
            source=CommercialEntitlementSource.checkout,
        )
        session.add(entitlement)

    session.commit()
    session.refresh(user)
    session.refresh(workspace)
    session.refresh(record)
    return user, workspace, record


def test_create_export_job_blueprint_professional(db_session: Session) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint_pro)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)

    job_response = create_export_job(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        payload=ExportJobCreateRequest(
            artifact_kind="blueprint_professional",
            profile="professional",
        ),
    )
    db_session.commit()

    assert job_response.product_key == "blueprint_pro"
    assert job_response.artifact_kind == "blueprint_professional"
    assert job_response.status == ExportJobStatus.ready
    assert job_response.file_name.endswith("blueprint_professional.md")
    assert job_response.checksum_sha256 != ""
    assert job_response.size_bytes > 0
    assert job_response.download_url.startswith(f"/api/v1/sessions/{record.id}/exports/jobs/")

    # Read bytes and verify checksum and content
    job_record, raw_bytes = read_export_job_bytes(db_session, record=record, job_id=job_response.id)
    assert hashlib.sha256(raw_bytes).hexdigest() == job_response.checksum_sha256

    content_str = raw_bytes.decode("utf-8")
    assert "Agentic Pro Project" in content_str
    assert "Blueprint Profesional" in content_str
    # Verify it does NOT contain ACP internal zip / runtime files
    assert "ACP/runtime/" not in content_str
    assert "ACP/manifest.json" not in content_str


def test_create_export_job_blueprint_professional_with_estimation_report(db_session: Session) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint_pro)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)
    snapshot.estimation_report = EstimationReportArtifact(
        traditional=TraditionalEstimate(
            estimated_hours_total=320,
            estimated_duration_weeks=10,
            estimated_cost=50_000_000,
        ),
        agentic=AgenticEstimate(
            estimated_hours_total=145,
            estimated_duration_weeks=5,
            estimated_cost=20_500_000,
            automation_coverage_percent=59,
            human_supervision_hours=18,
            net_savings_vs_traditional=29_500_000,
        ),
        confidence=ConfidenceBreakdown(
            score=96,
            label=EstimationConfidenceLabel.high,
            uncertainty_band_percent=12,
        ),
        construction_scenarios=[
            EstimationConstructionScenario(
                scenario_key="acp_agentic",
                label="ACP + herramientas agenticas",
                estimated_hours_total=128.7,
                estimated_duration_weeks=4.5,
                estimated_cost=20_511_057,
                effort_reduction_vs_traditional_percent=59,
                cost_savings_vs_traditional=29_405_156,
            )
        ],
        assumptions=["Las integraciones externas quedan acotadas al alcance aprobado."],
        risk_drivers=["Las decisiones de entorno pueden cambiar la banda final de costos."],
    )

    job_response = create_export_job(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        payload=ExportJobCreateRequest(
            artifact_kind="blueprint_professional",
            profile="professional",
            idempotency_key=f"{record.id}:blueprint-professional-estimation",
        ),
    )
    db_session.commit()

    assert job_response.status == ExportJobStatus.ready
    _, raw_bytes = read_export_job_bytes(db_session, record=record, job_id=job_response.id)
    content_str = raw_bytes.decode("utf-8")
    assert "ACP + herramientas agenticas" in content_str
    assert "59.0%" in content_str
    assert "96%" in content_str


def test_create_export_job_reruns_failed_existing_job_with_same_idempotency_key(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint_pro)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)
    original_run = export_delivery_service._run_export_generation
    calls = {"count": 0}

    def flaky_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient export failure")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(export_delivery_service, "_run_export_generation", flaky_run)

    payload = ExportJobCreateRequest(
        artifact_kind="blueprint_professional",
        profile="professional",
        idempotency_key=f"{record.id}:rerun-blueprint-export",
    )

    first = create_export_job(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        payload=payload,
    )
    db_session.commit()
    assert first.status == ExportJobStatus.failed

    second = create_export_job(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        payload=payload,
    )
    db_session.commit()

    assert second.id == first.id
    assert second.status == ExportJobStatus.ready
    assert calls["count"] == 2


def test_create_export_job_forbidden_for_free_tier(db_session: Session) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)

    with pytest.raises(PermissionError, match="blueprint.download"):
        create_export_job(
            db_session,
            record=record,
            current_user=user,
            access=access,
            snapshot=snapshot,
            preview=preview,
            payload=ExportJobCreateRequest(
                artifact_kind="blueprint_professional",
                profile="professional",
            ),
        )


def test_get_and_retry_export_job(db_session: Session) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint_pro)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)

    job_response = create_export_job(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        payload=ExportJobCreateRequest(
            artifact_kind="blueprint_professional",
            profile="professional",
        ),
    )
    db_session.commit()

    fetched = get_export_job_response(db_session, record=record, job_id=job_response.id)
    assert fetched.id == job_response.id
    assert fetched.status == ExportJobStatus.ready

    retried = retry_export_job_response(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        job_id=job_response.id,
    )
    assert retried.id == job_response.id
    assert retried.status == ExportJobStatus.ready


def test_create_export_job_acp_conformance_failure(db_session: Session) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.acp)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)

    job_response = create_export_job(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        payload=ExportJobCreateRequest(
            artifact_kind="acp_portable_zip",
            profile="acp-portable",
        ),
    )
    db_session.commit()

    assert job_response.product_key == "acp"
    assert job_response.artifact_kind == "acp_portable_zip"
    assert job_response.status == ExportJobStatus.failed
    assert "ACP conformance is not ready" in (job_response.error_message or "")


def test_create_export_job_acp_conformance_success(db_session: Session) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.acp)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)

    # Enable conformance
    preview.validation.can_export_zip = True
    preview.construction_readiness.blocking_gaps = 0
    preview.construction_readiness.open_questions = 0

    job_response = create_export_job(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        payload=ExportJobCreateRequest(
            artifact_kind="acp_portable_zip",
            profile="acp-portable",
        ),
    )
    db_session.commit()

    assert job_response.product_key == "acp"
    assert job_response.artifact_kind == "acp_portable_zip"
    assert job_response.status == ExportJobStatus.ready
    assert job_response.file_name.endswith(".zip")
    assert job_response.checksum_sha256 != ""
    assert job_response.size_bytes > 0
    assert job_response.download_url.startswith(f"/api/v1/sessions/{record.id}/exports/jobs/")

    job_record, raw_bytes = read_export_job_bytes(db_session, record=record, job_id=job_response.id)
    assert hashlib.sha256(raw_bytes).hexdigest() == job_response.checksum_sha256


def test_create_export_job_acp_forbidden_without_entitlement(db_session: Session) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint_pro)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)

    with pytest.raises(PermissionError, match="export_acp_zip"):
        create_export_job(
            db_session,
            record=record,
            current_user=user,
            access=access,
            snapshot=snapshot,
            preview=preview,
            payload=ExportJobCreateRequest(
                artifact_kind="acp_portable_zip",
                profile="acp-portable",
            ),
        )

