from __future__ import annotations

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
)
from app.services.auth_service import hash_password
from app.services.deliverable_catalog import build_deliverable_catalog_response
from app.services.deliverable_catalog.registry_service import get_registry_entry
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.product_processing import ProductBuildLifecycle, ProductBuildProductKey, build_product_build_status


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
