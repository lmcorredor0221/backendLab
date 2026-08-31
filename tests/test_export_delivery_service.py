from __future__ import annotations

import hashlib
from datetime import timedelta
from io import BytesIO
from collections.abc import Iterator
from uuid import uuid4
from zipfile import ZipFile

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    AgenticEstimate,
    ArtifactRegistryRecord,
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
    ExportJobRecord,
    ExportJobStatus,
    JourneyStateRecord,
    ProjectTitleSource,
    SessionRecord,
    SessionStage,
    TraditionalEstimate,
    UserRecord,
    utc_now,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.commercial_access import build_commercial_access_snapshot_v2
from app.services import export_delivery_service
from app.services.product_processing.contracts import JourneyStateKey
from app.services.diagram_center.persistence import DiagramVersionRecord
from app.services.export_delivery_service import (
    create_export_job,
    get_export_job_response,
    read_export_job_bytes,
    retry_export_job_response,
)
from app.services.product_processing import ProductProcessingMode, classify_uncertainty_for_profile, upsert_uncertainty_backlog
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


def _zip_members(raw_bytes: bytes) -> dict[str, bytes]:
    with ZipFile(BytesIO(raw_bytes)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


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
    assert job_response.content_type == "application/zip"
    assert job_response.file_name.endswith("blueprint_professional.zip")
    assert job_response.checksum_sha256 != ""
    assert job_response.size_bytes > 0
    assert job_response.download_url.startswith(f"/api/v1/sessions/{record.id}/exports/jobs/")
    journey_state = db_session.exec(
        select(JourneyStateRecord).where(JourneyStateRecord.session_id == record.id)
    ).one()
    assert journey_state.state_key == JourneyStateKey.blueprint_pro_active.value
    assert journey_state.substate == "completed"

    # Read bytes and verify checksum and content
    job_record, raw_bytes = read_export_job_bytes(db_session, record=record, job_id=job_response.id)
    assert hashlib.sha256(raw_bytes).hexdigest() == job_response.checksum_sha256

    members = _zip_members(raw_bytes)
    assert "Blueprint/README.md" in members
    assert "Blueprint/contracts/blueprint-core.v1.json" in members
    assert "Blueprint/governance/decisiones-delegadas-y-supuestos.md" in members
    assert "Blueprint/manifest.json" in members
    assert b"Agentic Pro Project" in members["Blueprint/README.md"]
    assert b"Blueprint Profesional" in members["Blueprint/README.md"]
    governance_doc = members["Blueprint/governance/decisiones-delegadas-y-supuestos.md"].decode("utf-8")
    assert "Decisiones Delegadas y Supuestos de Implementacion" in governance_doc
    assert "No hay decisiones delegadas" in governance_doc
    assert "ACP/runtime/" not in "\n".join(members)
    assert "ACP/manifest.json" not in "\n".join(members)


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
    members = _zip_members(raw_bytes)
    content_str = members["Blueprint/README.md"].decode("utf-8")
    assert "ACP + herramientas agenticas" in content_str
    assert "59.0%" in content_str
    assert "96%" in content_str


def test_create_export_job_blueprint_professional_includes_persisted_artifacts_and_diagrams(
    db_session: Session,
) -> None:
    user, workspace, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint_pro)
    db_session.add(
        ArtifactRegistryRecord(
            session_id=record.id,
            blueprint_version_number=7,
            artifact_key="Blueprint/architecture/architecture.md",
            artifact_title="Especificacion de arquitectura",
            artifact_kind="artifact",
            stage=SessionStage.ready_for_export,
            source_action="test_seed",
            export_format="markdown",
            content_text="# Arquitectura\n\nContenido versionado.",
            content_hash="hash-architecture",
        )
    )
    db_session.add(
        ArtifactRegistryRecord(
            session_id=record.id,
            blueprint_version_number=7,
            artifact_key="ACP/estimation/estimation-report.json",
            artifact_title="Estimation report",
            artifact_kind="estimation_report",
            stage=SessionStage.ready_for_export,
            source_action="test_seed",
            export_format="json",
            content_text=EstimationReportArtifact(
                traditional=TraditionalEstimate(
                    estimated_hours_total=320,
                    estimated_duration_weeks=10,
                    estimated_cost=50_000_000,
                ),
                agentic=AgenticEstimate(
                    estimated_hours_total=145,
                    estimated_duration_weeks=5,
                    estimated_cost=20_511_057,
                    automation_coverage_percent=59,
                    human_supervision_hours=18,
                    net_savings_vs_traditional=29_405_156,
                ),
                confidence=ConfidenceBreakdown(
                    score=96,
                    label=EstimationConfidenceLabel.high,
                    uncertainty_band_percent=12,
                ),
                assumptions=["Fixture de estimacion valida para export ZIP."],
                risk_drivers=["Sincronizar costos con el alcance aprobado."],
            ).model_dump_json(),
            content_hash="hash-estimation",
        )
    )
    db_session.add(
        DiagramVersionRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            diagram_key="architecture_context",
            version_number=3,
            state="available",
            diagram_model={"diagram_key": "architecture_context", "title": "Arquitectura"},
            renderings={
                "mermaid": "flowchart LR\n  A-->B\n",
                "svg": "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>",
                "presentation": "{\"rendering_format\":\"svg\"}",
            },
            quality_report={"state": "passed"},
            source_fingerprint="fp-1",
        )
    )
    db_session.commit()

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
            idempotency_key=f"{record.id}:blueprint-with-artifacts",
        ),
    )
    db_session.commit()

    assert job_response.status == ExportJobStatus.ready
    _, raw_bytes = read_export_job_bytes(db_session, record=record, job_id=job_response.id)
    members = _zip_members(raw_bytes)

    assert "Blueprint/architecture/architecture.md" in members
    assert "Blueprint/estimation/estimation-report.json" in members
    assert "Blueprint/diagrams/architecture_context/diagram-model.v1.json" in members
    assert "Blueprint/diagrams/architecture_context/architecture_context.mmd" in members
    assert "Blueprint/diagrams/architecture_context/architecture_context.svg" in members
    assert "Blueprint/diagrams/architecture_context/diagram-presentation.v1.json" in members
    assert members["Blueprint/architecture/architecture.md"].decode("utf-8").startswith("# Arquitectura")
    assert "\"estimated_cost\":20511057" in members["Blueprint/estimation/estimation-report.json"].decode("utf-8")


def test_blueprint_professional_zip_includes_delegated_decisions_document_from_backlog(
    db_session: Session,
) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint_pro)
    classification = classify_uncertainty_for_profile(
        "design",
        {
            "key": "document_provider_decision",
            "question": "Confirmar proveedor documental y credenciales durante la implementacion.",
            "description": "Seleccionar proveedor documental final para alimentar memoria y RAG.",
            "impact": "Afecta Tools, Memory, RAG, permisos y observabilidad.",
            "confidence": 0.42,
            "priority": "high",
            "deferral_target_stage": "ACP",
            "suggested_answer": "Usar repositorio documental centralizado con permisos por rol.",
            "assumed_answer": "Documentos aprobados disponibles por API o export controlado.",
            "affected_deliverable_keys": ["diagram.c4_context", "memory.strategy", "tools.minimum_set"],
            "source_refs": ["journey.design.open_questions"],
        },
        ProductProcessingMode.basic_free,
    )
    upsert_uncertainty_backlog(
        db_session,
        workspace_id=record.workspace_id,
        session_id=record.id,
        classification=classification,
        dependency_keys=["design.selected_alternative", "memory.knowledge_sources"],
        created_from="test",
    )
    db_session.commit()

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
            idempotency_key=f"{record.id}:blueprint-delegated-decisions",
        ),
    )
    db_session.commit()

    assert job_response.status == ExportJobStatus.ready
    _, raw_bytes = read_export_job_bytes(db_session, record=record, job_id=job_response.id)
    members = _zip_members(raw_bytes)
    content = members["Blueprint/governance/decisiones-delegadas-y-supuestos.md"].decode("utf-8")

    assert "document_provider_decision" in content
    assert "Estado: `delegado`" in content
    assert "Momento recomendado: `ACP o implementacion`" in content
    assert "Usar repositorio documental centralizado" in content
    assert "Documentos aprobados disponibles por API" in content
    assert "Afecta Tools, Memory, RAG" in content
    assert "diagram.c4_context, memory.strategy, tools.minimum_set" in content
    assert "design.selected_alternative, memory.knowledge_sources" in content


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


def test_get_export_job_marks_ready_job_as_failed_when_payload_is_missing(db_session: Session) -> None:
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

    job_record = db_session.get(ExportJobRecord, job_response.id)
    assert job_record is not None
    payload_path = export_delivery_service._storage_path(job_record)
    payload_path.unlink()

    fetched = get_export_job_response(db_session, record=record, job_id=job_response.id)

    assert fetched.id == job_response.id
    assert fetched.status == ExportJobStatus.failed
    assert "payload not found" in fetched.error_message.lower()


def test_create_export_job_regenerates_ready_job_when_payload_is_missing_locally(db_session: Session) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint_pro)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)
    payload = ExportJobCreateRequest(
        artifact_kind="blueprint_professional",
        profile="professional",
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

    job_record = db_session.get(ExportJobRecord, first.id)
    assert job_record is not None
    payload_path = export_delivery_service._storage_path(job_record)
    original_bytes = payload_path.read_bytes()
    payload_path.unlink()

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

    refreshed_record = db_session.get(ExportJobRecord, first.id)
    assert refreshed_record is not None
    assert second.id == first.id
    assert second.status == ExportJobStatus.ready
    assert payload_path.exists()
    assert payload_path.read_bytes() == original_bytes
    assert int(refreshed_record.metadata_payload.get("auto_regeneration_count", 0) or 0) >= 1


def test_create_export_job_regenerates_ready_job_when_contract_is_legacy_or_stale(db_session: Session) -> None:
    user, _, record = _seed_user_and_session(db_session, tier=CommercialTier.blueprint_pro)
    snapshot = build_snapshot(db_session, record, current_user=user)
    preview = resolve_acp_preview(db_session, record)
    access = build_commercial_access_snapshot_v2(db_session, record, current_user=user)
    idempotency_key = f"{record.id}:legacy-blueprint-professional"

    legacy_job = ExportJobRecord(
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=user.id,
        product_key="blueprint_pro",
        profile="professional",
        artifact_kind="blueprint_professional",
        status=ExportJobStatus.ready,
        idempotency_key=idempotency_key,
        content_type="text/markdown",
        file_name="legacy-blueprint-professional.md",
        storage_key=f"{record.workspace_id}/{record.id}/{idempotency_key}/legacy-blueprint-professional.md",
        checksum_sha256="legacy-checksum",
        size_bytes=27,
        expires_at=utc_now() + timedelta(hours=24),
        metadata_payload={
            "contract_version": "export-job.v1",
            "required_capability": "blueprint.download",
            "blueprint_version_number": (preview.blueprint_version_number or 1) - 1,
        },
        completed_at=utc_now(),
    )
    db_session.add(legacy_job)
    db_session.commit()

    legacy_path = export_delivery_service._storage_path(legacy_job)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("# Legacy markdown payload\n", encoding="utf-8")

    response = create_export_job(
        db_session,
        record=record,
        current_user=user,
        access=access,
        snapshot=snapshot,
        preview=preview,
        payload=ExportJobCreateRequest(
            artifact_kind="blueprint_professional",
            profile="professional",
            idempotency_key=idempotency_key,
        ),
    )
    db_session.commit()

    refreshed_record = db_session.get(ExportJobRecord, legacy_job.id)
    assert refreshed_record is not None
    assert response.id == legacy_job.id
    assert response.status == ExportJobStatus.ready
    assert response.content_type == "application/zip"
    assert response.file_name.endswith(".zip")
    assert not legacy_path.exists()

    reasons = set(refreshed_record.metadata_payload.get("last_regeneration_reasons", []))
    assert {"content_type", "file_name", "storage_key", "blueprint_version_number"} <= reasons

    _, raw_bytes = read_export_job_bytes(db_session, record=record, job_id=response.id)
    members = _zip_members(raw_bytes)
    assert "Blueprint/README.md" in members
    assert raw_bytes[:2] == b"PK"


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
