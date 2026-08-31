from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    CommercialEventRecord,
    CommercialTier,
    ExportJobRecord,
    ExportJobStatus,
    SessionRecord,
    SessionStage,
    StageOperationRecord,
    StageOperationStatus,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.auth_service import hash_password
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.product_processing import (
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductProcessingMode,
    build_product_build_telemetry_report,
    ensure_product_build_run,
    update_product_build_run_state,
    upsert_product_build_step,
)


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _seed_session(db: Session) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=f"eov15-{uuid4()}@leanbuilder.local",
        full_name="EOV15 Tester",
        password_hash=hash_password("Secret123!"),
    )
    db.add(user)
    db.flush()
    workspace = WorkspaceRecord(name="EOV15 Workspace", slug=f"eov15-{str(user.id)[:8]}", created_by_user_id=user.id)
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="EOV15 Project",
        current_stage=SessionStage.ready_for_export,
        commercial_tier=CommercialTier.acp,
    )
    db.add(record)
    db.commit()
    db.refresh(user)
    db.refresh(record)
    return user, record


def test_product_build_telemetry_groups_costs_duration_and_redacts_sensitive_metadata() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db)
        now = utc_now()
        run = ensure_product_build_run(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_basic,
            product_mode=ProductProcessingMode.basic_free,
            entitlement_tier=CommercialTier.blueprint,
            access_state="allowed",
            lifecycle=ProductBuildLifecycle.running,
            idempotency_key=f"blueprint-basic:{record.id}",
            created_by_user_id=user.id,
            checkpoint_payload={"surface": "blueprint", "prompt": "raw prompt should not leak"},
        )
        run.started_at = now - timedelta(seconds=95)
        job = DeliverableGenerationJobRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            deliverable_key="commercial_blueprint_summary",
            status="failed",
            product_mode="basic_free",
            idempotency_key=f"job:{uuid4()}",
            tokens_input=12,
            tokens_output=18,
            estimated_cost_usd=0.0123456,
            request_metadata={"surface": "blueprint", "api_key": "hidden", "raw_prompt": "hidden"},
            started_at=now - timedelta(seconds=70),
            completed_at=now - timedelta(seconds=10),
        )
        db.add(job)
        db.flush()
        upsert_product_build_step(
            db,
            run=run,
            step_key="commercial_blueprint_summary",
            status="requires_attention",
            stage_key="estimate",
            deliverable_key="commercial_blueprint_summary",
            job_id=job.id,
            sequence=1,
            error_payload={"code": "renderer_failed", "token": "hidden"},
        )
        update_product_build_run_state(
            db,
            run=run,
            lifecycle=ProductBuildLifecycle.completed,
            completed_units=1,
            total_units=1,
        )
        db.add(
            CommercialEventRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                user_id=user.id,
                event_key="blueprint_results_viewed",
                product_key="blueprint",
                source="product_page",
                metadata_payload={"surface": "blueprint", "prompt": "hidden"},
            )
        )
        db.add(
            StageOperationRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                user_id=user.id,
                stage_key="estimate",
                action="retry_product_build",
                status=StageOperationStatus.completed,
                request_payload={"product_key": "blueprint_basic", "reason": "manual retry", "secret": "hidden"},
            )
        )
        db.commit()

        report = build_product_build_telemetry_report(db, record=record, current_user=user)

    assert report.contract_version == "product-build-telemetry.v1"
    assert report.workspace_id == record.workspace_id
    assert report.session_id == record.id
    basic = next(item for item in report.products if item.product_key == ProductBuildProductKey.blueprint_basic)
    assert basic.run_count == 1
    assert basic.step_count == 1
    assert basic.deliverable_count == 1
    assert basic.cta_count == 1
    assert basic.retry_count == 1
    assert basic.requires_attention_count == 1
    assert basic.run_completed_count == 1
    assert basic.tokens_total == 30
    assert basic.estimated_cost_usd == 0.012346
    assert basic.run_duration_seconds >= 90
    assert basic.deliverable_duration_seconds == 60
    assert report.totals.estimated_cost_usd == 0.012346
    assert all(event.workspace_id == record.workspace_id for event in report.events)
    assert all(event.session_id == record.id for event in report.events)
    assert any(event.run_id == str(run.id) for event in report.events)
    serialized = report.model_dump_json()
    assert "raw prompt should not leak" not in serialized
    assert "api_key" not in serialized
    assert "raw_prompt" not in serialized
    assert "hidden" not in serialized


def test_product_build_telemetry_includes_export_jobs_and_operation_ids() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db)
        now = utc_now()
        run = ensure_product_build_run(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro,
            product_mode=ProductProcessingMode.premium_enrichment,
            entitlement_tier=CommercialTier.blueprint_pro,
            access_state="allowed",
            lifecycle=ProductBuildLifecycle.completed,
            idempotency_key=f"blueprint-pro:{record.id}",
            created_by_user_id=user.id,
        )
        run.started_at = now - timedelta(seconds=40)
        run.completed_at = now - timedelta(seconds=5)
        operation = StageOperationRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=user.id,
            stage_key="package",
            action="resume_product_build",
            status=StageOperationStatus.completed,
            request_payload={"product_key": "blueprint_pro", "resume": True},
        )
        export_job = ExportJobRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=user.id,
            product_key="blueprint_pro",
            profile="professional",
            artifact_kind="blueprint_professional",
            status=ExportJobStatus.ready,
            idempotency_key=f"export:{record.id}:blueprint-pro",
            content_type="application/zip",
            file_name="eov15-project-blueprint_professional.zip",
            storage_key=f"{record.workspace_id}/{record.id}/export/eov15-project-blueprint_professional.zip",
            checksum_sha256="sha-telemetry",
            size_bytes=4096,
            expires_at=now + timedelta(hours=24),
            metadata_payload={
                "retry_count": 1,
                "auto_regeneration_count": 1,
                "last_regeneration_reasons": ["blueprint_version_number"],
                "blueprint_version_number": 4,
            },
            completed_at=now - timedelta(seconds=2),
        )
        db.add(operation)
        db.add(export_job)
        db.commit()

        report = build_product_build_telemetry_report(db, record=record, current_user=user)

    export_event = next(item for item in report.events if item.export_job_id == str(export_job.id))
    assert export_event.event_key == "export_job_regenerated"
    assert export_event.event_type == "export"
    assert export_event.product_key == ProductBuildProductKey.blueprint_pro

    operation_event = next(item for item in report.events if item.operation_id == str(operation.id))
    assert operation_event.event_key == "resume_product_build"
    assert operation_event.source == "stage_operation"
    assert operation_event.product_key == ProductBuildProductKey.blueprint_pro

    assert any("export jobs regenerados automaticamente" in warning.lower() for warning in report.warnings)
