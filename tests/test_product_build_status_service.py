from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    ArtifactRegistryRecord,
    CommercialTier,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.auth_service import hash_password
from app.services.deliverable_catalog import build_deliverable_catalog_response
from app.services.deliverable_catalog.registry_service import get_registry_entry
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord, UncertaintyBacklogRecord
from app.services.product_processing import (
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductProcessingMode,
    UncertaintyBacklogStatus,
    UncertaintyDisposition,
    build_product_build_status,
)


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _seed_session(
    db: Session,
    *,
    tier: CommercialTier = CommercialTier.blueprint,
    stage: SessionStage = SessionStage.ready_for_export,
) -> tuple[UserRecord, SessionRecord]:
    user = UserRecord(
        email=f"eov3-{uuid4()}@leanbuilder.local",
        full_name="EOV3 Tester",
        password_hash=hash_password("Secret123!"),
    )
    db.add(user)
    db.flush()
    workspace = WorkspaceRecord(name="EOV3 Workspace", slug=f"eov3-{str(user.id)[:8]}", created_by_user_id=user.id)
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(
        user_id=user.id,
        workspace_id=workspace.id,
        title="EOV3 Project",
        current_stage=stage,
        commercial_tier=tier,
    )
    db.add(record)
    db.commit()
    db.refresh(user)
    db.refresh(record)
    return user, record


def test_product_build_status_locks_unpurchased_product() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint)
        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
        )

    assert status.lifecycle == ProductBuildLifecycle.not_purchased
    assert status.entitlement.purchase_required is True
    assert status.actions[0].action_key == "buy_product"
    assert status.actions[0].primary is True


def test_product_build_status_hides_legacy_run_activity_when_product_is_locked() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        run = ProductBuildRunRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key="acp",
            product_mode="acp_implementation",
            entitlement_tier="blueprint_pro",
            access_state="locked",
            lifecycle="requires_attention",
            progress_percent=55,
            completed_units=12,
            total_units=16,
            blocked_units=2,
            idempotency_key=f"eov3-acp-legacy-{uuid4()}",
            checkpoint_payload={
                "processing_queue": {
                    "queue_id": "queue-acp-legacy",
                    "mode": "process_pending",
                    "status": "completed_with_errors",
                    "selected_deliverable_keys": ["diagram.human_intervention_flow"],
                    "summary": "Cola legacy ACP generada antes del gate comercial.",
                }
            },
            error_payload={
                "code": "legacy_locked_acp",
                "message": "No deberia mostrarse mientras ACP siga bloqueado.",
            },
            created_by_user_id=user.id,
        )
        db.add(run)
        db.flush()
        db.add(
            ProductBuildStepRecord(
                run_id=run.id,
                workspace_id=record.workspace_id,
                session_id=record.id,
                step_key="deliverable:diagram.human_intervention_flow",
                stage_key="validate",
                deliverable_key="diagram.human_intervention_flow",
                status="requires_attention",
                progress_percent=0,
                checkpoint_payload={"attempt_count": 1, "title": "Intervencion humana y aprobaciones"},
                error_payload={"message": "Legacy ACP queue noise."},
            )
        )
        db.commit()

        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.acp,
            current_user=user,
        )

    assert status.lifecycle == ProductBuildLifecycle.not_purchased
    assert status.entitlement.purchase_required is True
    assert status.current_activity is None


def test_product_build_status_counts_only_running_items_as_processing() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        run = ProductBuildRunRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro.value,
            product_mode=ProductProcessingMode.premium_enrichment.value,
            entitlement_tier=CommercialTier.blueprint_pro.value,
            access_state="allowed",
            lifecycle=ProductBuildLifecycle.running.value,
            progress_percent=25,
            completed_units=1,
            total_units=4,
            idempotency_key=f"eov3-blueprint-pro-queue-{uuid4()}",
            checkpoint_payload={
                "processing_queue": {
                    "queue_id": "queue-blueprint-pro",
                    "mode": "process_pending",
                    "status": "running",
                    "selected_deliverable_keys": ["architecture.spec", "diagram.knowledge_graph"],
                    "summary": "Procesando fixture.",
                }
            },
            created_by_user_id=user.id,
        )
        db.add(run)
        db.flush()
        db.add(
            ProductBuildStepRecord(
                run_id=run.id,
                workspace_id=record.workspace_id,
                session_id=record.id,
                step_key="deliverable:architecture.spec",
                stage_key="design",
                deliverable_key="architecture.spec",
                status="queued",
                sequence=1,
                progress_percent=10,
                checkpoint_payload={"title": "Arquitectura"},
            )
        )
        db.add(
            ProductBuildStepRecord(
                run_id=run.id,
                workspace_id=record.workspace_id,
                session_id=record.id,
                step_key="deliverable:diagram.knowledge_graph",
                stage_key="design",
                deliverable_key="diagram.knowledge_graph",
                status="generating",
                sequence=2,
                progress_percent=55,
                checkpoint_payload={"title": "Knowledge graph"},
            )
        )
        db.commit()

        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
        )

    assert status.processing_queue is not None
    assert status.processing_queue.processing_count == 1
    assert status.processing_queue.pending_count == 1


def test_product_build_status_projects_stale_active_queue_as_recoverable_without_mutating() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        stale_at = utc_now() - timedelta(minutes=20)
        run = ProductBuildRunRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro.value,
            product_mode=ProductProcessingMode.premium_enrichment.value,
            entitlement_tier=CommercialTier.blueprint_pro.value,
            access_state="allowed",
            lifecycle=ProductBuildLifecycle.running.value,
            progress_percent=84,
            completed_units=3,
            total_units=4,
            idempotency_key=f"eov3-blueprint-pro-stale-{uuid4()}",
            checkpoint_payload={
                "processing_queue": {
                    "queue_id": "queue-blueprint-pro-stale",
                    "mode": "process_pending",
                    "status": "running",
                    "selected_deliverable_keys": ["diagram.knowledge_graph"],
                    "summary": "Procesando fixture stale.",
                }
            },
            created_by_user_id=user.id,
        )
        db.add(run)
        db.flush()
        step = ProductBuildStepRecord(
            run_id=run.id,
            workspace_id=record.workspace_id,
            session_id=record.id,
            step_key="deliverable:diagram.knowledge_graph",
            stage_key="design",
            deliverable_key="diagram.knowledge_graph",
            status="generating",
            sequence=1,
            progress_percent=55,
            checkpoint_payload={"title": "Knowledge graph"},
            updated_at=stale_at,
        )
        db.add(step)
        db.commit()

        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
        )
        db.expire_all()
        persisted_run = db.get(ProductBuildRunRecord, run.id)
        persisted_step = db.get(ProductBuildStepRecord, step.id)

    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert status.last_error is not None
    assert status.last_error.code == "processing_queue_possibly_interrupted"
    assert status.last_error.retry_action_key == "retry_failed"
    assert persisted_run is not None
    assert persisted_run.lifecycle == ProductBuildLifecycle.running.value
    assert persisted_step is not None
    assert persisted_step.status == "generating"


def test_product_build_status_does_not_mark_queue_interrupted_when_in_flight_step_is_active() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    stale_at = utc_now() - timedelta(minutes=20)
    recent_at = utc_now() - timedelta(seconds=30)
    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        run = ProductBuildRunRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro.value,
            product_mode=ProductProcessingMode.premium_enrichment.value,
            lifecycle=ProductBuildLifecycle.running.value,
            idempotency_key=f"eov3-blueprint-pro-active-{uuid4()}",
            checkpoint_payload={
                "processing_queue": {
                    "queue_id": "queue-blueprint-pro-active",
                    "mode": "process_pending",
                    "status": "running",
                    "selected_deliverable_keys": ["diagram.knowledge_graph", "diagram.architecture_c4"],
                    "summary": "Procesando fixture activo.",
                }
            },
            created_by_user_id=user.id,
        )
        db.add(run)
        db.flush()
        # Step 1 is in-flight and recent (30s ago)
        step_active = ProductBuildStepRecord(
            run_id=run.id,
            workspace_id=record.workspace_id,
            session_id=record.id,
            step_key="deliverable:diagram.knowledge_graph",
            stage_key="design",
            deliverable_key="diagram.knowledge_graph",
            status="generating",
            sequence=1,
            progress_percent=55,
            checkpoint_payload={"title": "Knowledge graph"},
            updated_at=recent_at,
        )
        # Step 2 is queued from 10 minutes ago
        step_queued = ProductBuildStepRecord(
            run_id=run.id,
            workspace_id=record.workspace_id,
            session_id=record.id,
            step_key="deliverable:diagram.architecture_c4",
            stage_key="design",
            deliverable_key="diagram.architecture_c4",
            status="queued",
            sequence=2,
            progress_percent=0,
            checkpoint_payload={"title": "Architecture C4"},
            updated_at=stale_at,
        )
        db.add(step_active)
        db.add(step_queued)
        db.commit()

        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
        )

    # Should remain running, not requires_attention!
    assert status.lifecycle == ProductBuildLifecycle.running
    assert status.last_error is None


def test_product_build_status_projects_recoverable_run_error_as_retry_action() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        run = ProductBuildRunRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            product_key=ProductBuildProductKey.blueprint_pro.value,
            product_mode=ProductProcessingMode.premium_enrichment.value,
            entitlement_tier=CommercialTier.blueprint_pro.value,
            access_state="allowed",
            lifecycle=ProductBuildLifecycle.running.value,
            progress_percent=32,
            completed_units=3,
            total_units=4,
            idempotency_key=f"eov3-blueprint-pro-recoverable-{uuid4()}",
            checkpoint_payload={
                "processing_queue": {
                    "queue_id": "queue-blueprint-pro-recoverable",
                    "mode": "retry_failed",
                    "status": "running",
                    "selected_deliverable_keys": ["diagram.knowledge_graph"],
                }
            },
            error_payload={
                "code": "processing_queue_orphaned",
                "title": "El procesamiento fue interrumpido",
                "message": "Algunos entregables requieren un reintento controlado.",
                "retry_action_key": "retry_failed",
                "trace_refs": ["diagram.knowledge_graph"],
            },
            created_by_user_id=user.id,
        )
        db.add(run)
        db.flush()
        db.add(
            ProductBuildStepRecord(
                run_id=run.id,
                workspace_id=record.workspace_id,
                session_id=record.id,
                step_key="deliverable:diagram.knowledge_graph",
                stage_key="design",
                deliverable_key="diagram.knowledge_graph",
                status="generating",
                sequence=1,
                progress_percent=55,
                checkpoint_payload={"title": "Knowledge graph"},
            )
        )
        db.commit()

        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
        )

    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert status.last_error is not None
    assert status.last_error.code == "processing_queue_orphaned"
    assert status.actions
    assert status.actions[0].action_key == "retry_failed"
    assert status.actions[0].state.value == "recommended"


def test_blueprint_pro_treats_deferred_basic_uncertainty_as_non_blocking() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint_pro)
        db.add(
            UncertaintyBacklogRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                uncertainty_key="design:deferred-basic-question",
                product_mode=ProductProcessingMode.basic_free.value,
                source_stage="design",
                target_stage="",
                kind="question",
                disposition=UncertaintyDisposition.defer.value,
                status=UncertaintyBacklogStatus.deferred.value,
                title="Decision diferida de diseno",
                reason="Basico delega dudas no indispensables sin detener el flujo.",
            )
        )
        db.commit()

        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
        )

    assert status.attention.total == 1
    assert status.attention.blocking_count == 0
    assert status.attention.items[0].blocking is False
    assert status.processing_queue is None
    assert status.last_error is None
    assert status.progress.percent == 0


def test_product_build_status_uses_governed_catalog_count() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.acp)
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.acp,
            current_stage="package",
        )
        expected = [item for item in catalog.entries if set(item.product_scope).intersection({"blueprint", "blueprint_pro", "acp"})]
        status = build_product_build_status(db, record=record, product_key=ProductBuildProductKey.acp, current_user=user)

    assert len(status.deliverables) == len(expected)
    assert status.progress.total_units == float(len(expected))
    assert "deliverable-catalog-response.v1" in status.source_contracts


def test_product_build_status_surfaces_failed_deliverable_as_attention() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint)
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint,
            current_stage="package",
        )
        deliverable_key = next(item.key for item in catalog.entries if "blueprint" in item.product_scope)
        db.add(
            DeliverableGenerationJobRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                deliverable_key=deliverable_key,
                status="failed",
                product_mode="basic_free",
                idempotency_key=f"eov3-job-{uuid4()}",
                error_code="renderer_failed",
                error_message="Renderer failed while producing the governed deliverable.",
            )
        )
        db.commit()

        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_basic,
            current_user=user,
        )
        failed = next(item for item in status.deliverables if item.deliverable_key == deliverable_key)

    assert failed.state.value == "error"
    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert status.attention.blocking_count == 1
    assert status.last_error is not None
    assert status.last_error.code == "deliverable_generation_error"


def test_product_build_status_surfaces_failed_diagram_job_as_attention() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint)
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint,
            current_stage="package",
        )
        diagram_item = next(
            item
            for item in catalog.entries
            if item.deliverable_type.value == "diagram" and "blueprint" in item.product_scope
        )
        db.add(
            DiagramGenerationJobRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                diagram_key=diagram_item.key.removeprefix("diagram."),
                status="failed",
                idempotency_key=f"eov3-diagram-job-{uuid4()}",
                error_code="diagram_renderer_failed",
                error_message="Diagram renderer failed while producing the governed diagram.",
            )
        )
        db.commit()

        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_basic,
            current_user=user,
        )
        failed = next(item for item in status.deliverables if item.deliverable_key == diagram_item.key)

    assert failed.state.value == "error"
    assert status.lifecycle == ProductBuildLifecycle.requires_attention
    assert status.attention.blocking_count == 1
    assert status.last_error is not None
    assert status.last_error.trace_refs == [diagram_item.key]


def test_product_build_status_uses_active_diagram_job_as_current_activity() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(db, tier=CommercialTier.blueprint)
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=record.workspace_id,
            session_id=record.id,
            role=WorkspaceRole.owner,
            tier=CommercialTier.blueprint,
            current_stage="package",
        )
        diagram_item = next(
            item
            for item in catalog.entries
            if item.deliverable_type.value == "diagram" and "blueprint" in item.product_scope
        )
        db.add(
            DiagramGenerationJobRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                diagram_key=diagram_item.key.removeprefix("diagram."),
                status="generating",
                idempotency_key=f"eov3-diagram-job-{uuid4()}",
            )
        )
        db.commit()

        status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_basic,
            current_user=user,
        )

    assert status.lifecycle == ProductBuildLifecycle.running
    assert status.current_activity is not None
    assert status.current_activity.label == "Generando diagrama"
    assert status.current_activity.detail == diagram_item.key


def test_product_build_status_allows_cumulative_product_surface_override() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        user, record = _seed_session(
            db,
            tier=CommercialTier.blueprint_pro,
            stage=SessionStage.build_canvas,
        )
        entry = get_registry_entry("diagram.c4_context")
        assert entry is not None
        assert entry.canonical_paths

        db.add(
            ArtifactRegistryRecord(
                session_id=record.id,
                artifact_key=entry.canonical_paths[0],
                artifact_title=entry.title,
                artifact_kind=entry.deliverable_type.value,
                stage=SessionStage.ready_for_export,
                source_action="deliverable_generation_agent",
                export_format=entry.formats.preferred,
                content_text="graph TD\n    Client --> Agent",
                content_hash="c4-context-available",
                artifact_metadata={
                    "deliverable_key": entry.deliverable_key,
                    "product_scope": list(entry.product_scope),
                },
            )
        )
        db.commit()

        default_status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
        )
        product_surface_status = build_product_build_status(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=user,
            catalog_stage_override="package",
        )

        default_item = next(item for item in default_status.deliverables if item.deliverable_key == entry.deliverable_key)
        product_surface_item = next(
            item for item in product_surface_status.deliverables if item.deliverable_key == entry.deliverable_key
        )

    assert default_item.state.value == "locked"
    assert product_surface_item.state.value == "available"
