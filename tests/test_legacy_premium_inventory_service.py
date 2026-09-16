from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import CommercialTier, SessionRecord, UserRecord, WorkspaceRecord
from app.services.auth_service import hash_password
from app.services.product_processing.legacy_premium_inventory_service import (
    build_legacy_premium_inventory_report,
    record_legacy_premium_endpoint_invocation,
)
from app.services.product_processing.legacy_premium_migration_service import (
    build_legacy_premium_migration_dry_run,
    execute_legacy_premium_migration_batch,
)
from app.services.product_processing.persistence import (
    ProductBuildRunRecord,
    ProductBuildStepRecord,
    UncertaintyBacklogRecord,
)


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def test_legacy_premium_inventory_classifies_records_collisions_and_usage() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        user = UserRecord(
            email="inventory@example.com",
            full_name="Inventory Admin",
            password_hash=hash_password("Secret123!"),
        )
        db.add(user)
        db.flush()
        workspace = WorkspaceRecord(name="Inventory", slug="inventory", created_by_user_id=user.id)
        db.add(workspace)
        db.flush()
        project = SessionRecord(
            user_id=user.id,
            workspace_id=workspace.id,
            title="Legacy Premium inventory",
            commercial_tier=CommercialTier.blueprint_pro,
        )
        db.add(project)
        db.flush()

        db.add_all(
            [
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="business_gap",
                    product_mode="premium_enrichment",
                    source_stage="design",
                    kind="gap",
                    status="open",
                    disposition="resolve_now",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="business_gap",
                    product_mode="basic_free",
                    source_stage="design",
                    kind="gap",
                    status="deferred",
                    disposition="defer",
                    target_stage="acp",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="deferred_to_acp",
                    product_mode="premium_enrichment",
                    source_stage="define",
                    target_stage="acp",
                    kind="question",
                    status="deferred",
                    disposition="defer",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="technical_error",
                    product_mode="premium_enrichment",
                    source_stage="package",
                    kind="runtime_error",
                    status="open",
                    disposition="resolve_now",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="historical_answer",
                    product_mode="premium_enrichment",
                    source_stage="define",
                    kind="decision",
                    status="resolved",
                    disposition="resolve_now",
                    assumed_answer="Mantener aprobacion humana.",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="unknown_legacy_shape",
                    product_mode="premium_enrichment",
                    source_stage="design",
                    kind="legacy_unknown",
                    status="open",
                    disposition="resolve_now",
                ),
            ]
        )
        db.flush()
        business_run = ProductBuildRunRecord(
            workspace_id=workspace.id,
            session_id=project.id,
            product_key="blueprint_pro",
            product_mode="premium_enrichment",
            entitlement_tier="blueprint_pro",
            access_state="allowed",
            lifecycle="requires_attention",
            idempotency_key=f"legacy-business:{project.id}",
        )
        technical_run = ProductBuildRunRecord(
            workspace_id=workspace.id,
            session_id=project.id,
            product_key="blueprint_pro",
            product_mode="premium_enrichment",
            entitlement_tier="blueprint_pro",
            access_state="allowed",
            lifecycle="requires_attention",
            idempotency_key=f"legacy-technical:{project.id}",
            error_payload={"code": "provider_timeout"},
        )
        db.add(business_run)
        db.add(technical_run)
        db.flush()
        db.add(
            ProductBuildStepRecord(
                run_id=business_run.id,
                workspace_id=workspace.id,
                session_id=project.id,
                step_key="premium_backlog:legacy",
                status="requires_attention",
            )
        )
        record_legacy_premium_endpoint_invocation(
            db,
            workspace_id=workspace.id,
            session_id=project.id,
            user_id=user.id,
            operation="resolve",
        )
        record_legacy_premium_endpoint_invocation(
            db,
            workspace_id=workspace.id,
            session_id=project.id,
            user_id=user.id,
            operation="resolve",
        )
        db.commit()

        before_count = len(db.exec(select(UncertaintyBacklogRecord)).all())
        report = build_legacy_premium_inventory_report(db, workspace_id=workspace.id)
        after_count = len(db.exec(select(UncertaintyBacklogRecord)).all())

    assert before_count == after_count
    assert report.total_records == 5
    assert report.business_pre_acp_count == 1
    assert report.acp_managed_count == 1
    assert report.technical_pro_count == 1
    assert report.closed_count == 1
    assert report.ambiguous_count == 1
    assert report.collision_count == 1
    assert report.build_health.business_backlog_attention_count == 1
    assert report.build_health.technical_attention_count == 1
    assert report.build_health.unattributed_attention_count == 0
    assert report.endpoint_usage[0].operation == "resolve"
    assert report.endpoint_usage[0].invocation_count == 2
    assert report.migration_ready is False
    assert any("colisiones" in warning.lower() for warning in report.warnings)
    assert any("endpoints premium legacy" in warning.lower() for warning in report.warnings)


def test_legacy_premium_migration_dry_run_is_read_only_and_classifies_actions() -> None:
    engine = _engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        user = UserRecord(
            email="dry-run@example.com",
            full_name="Dry Run Admin",
            password_hash=hash_password("Secret123!"),
        )
        db.add(user)
        db.flush()
        workspace = WorkspaceRecord(name="Dry Run", slug="dry-run", created_by_user_id=user.id)
        db.add(workspace)
        db.flush()
        project = SessionRecord(
            user_id=user.id,
            workspace_id=workspace.id,
            title="Legacy Premium dry run",
            commercial_tier=CommercialTier.blueprint_pro,
        )
        db.add(project)
        db.flush()
        db.add_all(
            [
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="merge_me",
                    product_mode="premium_enrichment",
                    source_stage="design",
                    kind="gap",
                    status="deferred",
                    disposition="infer",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="merge_me",
                    product_mode="basic_free",
                    source_stage="design",
                    target_stage="acp",
                    kind="gap",
                    status="deferred",
                    disposition="defer",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="create_me",
                    product_mode="premium_enrichment",
                    source_stage="define",
                    kind="question",
                    status="deferred",
                    disposition="infer",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="historic_answer",
                    product_mode="premium_enrichment",
                    source_stage="define",
                    kind="decision",
                    status="resolved",
                    disposition="resolve_now",
                    assumed_answer="Conservar aprobacion humana.",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="technical_error",
                    product_mode="premium_enrichment",
                    source_stage="package",
                    kind="runtime_error",
                    status="open",
                    disposition="resolve_now",
                ),
                UncertaintyBacklogRecord(
                    workspace_id=workspace.id,
                    session_id=project.id,
                    uncertainty_key="unknown_shape",
                    product_mode="premium_enrichment",
                    source_stage="design",
                    kind="legacy_unknown",
                    status="open",
                    disposition="resolve_now",
                ),
            ]
        )
        business_run = ProductBuildRunRecord(
            workspace_id=workspace.id,
            session_id=project.id,
            product_key="blueprint_pro",
            product_mode="premium_enrichment",
            entitlement_tier="blueprint_pro",
            access_state="allowed",
            lifecycle="requires_attention",
            idempotency_key=f"legacy-business:{project.id}",
        )
        db.add(business_run)
        db.flush()
        db.add(
            ProductBuildStepRecord(
                run_id=business_run.id,
                workspace_id=workspace.id,
                session_id=project.id,
                step_key="premium_backlog:legacy",
                status="requires_attention",
            )
        )
        db.commit()

        before_count = len(db.exec(select(UncertaintyBacklogRecord)).all())
        report = build_legacy_premium_migration_dry_run(db, workspace_id=workspace.id)
        after_count = len(db.exec(select(UncertaintyBacklogRecord)).all())

        with pytest.raises(ValueError, match="ambiguos"):
            execute_legacy_premium_migration_batch(db, workspace_id=workspace.id)
        unknown = db.exec(
            select(UncertaintyBacklogRecord).where(
                UncertaintyBacklogRecord.workspace_id == workspace.id,
                UncertaintyBacklogRecord.uncertainty_key == "unknown_shape",
                UncertaintyBacklogRecord.product_mode == "premium_enrichment",
            )
        ).one()
        unknown.status = "dismissed"
        db.add(unknown)
        db.flush()
        result = execute_legacy_premium_migration_batch(db, workspace_id=workspace.id)
        db.commit()
        next_dry_run = build_legacy_premium_migration_dry_run(db, workspace_id=workspace.id)
        migrated_source = db.exec(
            select(UncertaintyBacklogRecord).where(
                UncertaintyBacklogRecord.workspace_id == workspace.id,
                UncertaintyBacklogRecord.uncertainty_key == "merge_me",
                UncertaintyBacklogRecord.product_mode == "premium_enrichment",
            )
        ).one()
        merged_target = db.exec(
            select(UncertaintyBacklogRecord).where(
                UncertaintyBacklogRecord.workspace_id == workspace.id,
                UncertaintyBacklogRecord.uncertainty_key == "merge_me",
                UncertaintyBacklogRecord.product_mode == "basic_free",
            )
        ).one()
        second_result = execute_legacy_premium_migration_batch(db, workspace_id=workspace.id)
        db.commit()
        normalized_run = db.get(ProductBuildRunRecord, business_run.id)
        normalized_step = db.exec(
            select(ProductBuildStepRecord).where(ProductBuildStepRecord.run_id == business_run.id)
        ).one()
        migrated_source_status = migrated_source.status
        merged_target_stage = merged_target.target_stage
        merged_target_disposition = merged_target.disposition
        merged_target_sources = list(merged_target.payload["migration_v1"]["source_records"])
        normalized_run_lifecycle = normalized_run.lifecycle if normalized_run is not None else ""
        normalized_step_status = normalized_step.status

    actions = {action.uncertainty_key: action for action in report.actions}
    assert before_count == after_count
    assert report.total_candidates == 5
    assert report.proposed_create_count == 1
    assert report.proposed_merge_count == 1
    assert report.preserved_history_count == 1
    assert report.retained_technical_count == 1
    assert report.manual_review_count == 1
    assert actions["merge_me"].proposed_action == "merge_into_basic"
    assert actions["merge_me"].target_record_id is not None
    assert actions["create_me"].proposed_action == "create_basic_defer_to_acp"
    assert actions["historic_answer"].proposed_action == "preserve_historical_response"
    assert actions["technical_error"].proposed_action == "retain_technical_pro"
    assert actions["unknown_shape"].proposed_action == "manual_review"
    assert result.migrated_count == 3
    assert result.created_basic_count == 2
    assert result.merged_basic_count == 1
    assert result.preserved_historical_response_count == 1
    assert result.normalized_business_attention_run_count == 1
    assert next_dry_run.total_candidates == 1
    assert next_dry_run.retained_technical_count == 1
    assert next_dry_run.actions[0].uncertainty_key == "technical_error"
    assert next_dry_run.actions[0].source_record_id not in result.source_record_ids
    assert migrated_source_status == "superseded"
    assert merged_target_stage == "acp"
    assert merged_target_disposition == "defer"
    assert merged_target_sources
    assert normalized_run_lifecycle == "ready_to_start"
    assert normalized_step_status == "skipped"
    assert second_result.migrated_count == 0
