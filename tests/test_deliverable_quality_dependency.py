from __future__ import annotations

from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.services.deliverable_catalog import (
    compute_deliverable_staleness,
    evaluate_deliverable_quality,
    get_registry_entry,
    invalidate_deliverables_for_change,
    record_deliverable_quality_snapshot,
    resolve_regeneration_scope,
)
from app.services.deliverable_catalog.persistence import DeliverableQualitySnapshotRecord
from app.services.product_processing.contracts import UncertaintyBacklogStatus
from app.services.product_processing.persistence import UncertaintyBacklogRecord


def _session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_quality_evaluation_validates_artifact_and_records_snapshot() -> None:
    workspace_id = uuid4()
    session_id = uuid4()
    with _session() as db:
        entry = get_registry_entry("discovery.problem_context_brief")
        assert entry is not None

        passed = evaluate_deliverable_quality(entry, {"title": "Problema", "content": "Resumen trazable."})
        failed = evaluate_deliverable_quality(entry, {"title": "Problema"})
        snapshot = record_deliverable_quality_snapshot(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            entry=entry,
            version_ref="v1",
            payload={"title": "Problema", "content": "Resumen trazable."},
        )
        snapshot_state = snapshot.state
        snapshot_score = snapshot.score
        db.commit()

    assert passed.state == "passed"
    assert failed.state == "failed"
    assert "artifact_content_missing" in failed.errors
    assert snapshot_state == "passed"
    assert snapshot_score == 100


def test_dependency_service_computes_selective_staleness_and_regeneration_order() -> None:
    report = compute_deliverable_staleness(["definition.requirements"])
    scope = resolve_regeneration_scope(changed_dependency_keys=["definition.requirements"])

    assert "diagram.c4_context" in report.stale_deliverable_keys
    assert "diagram.security_guardrails" in report.stale_deliverable_keys
    assert "diagram.deployment_decision_matrix" in report.unchanged_deliverable_keys
    assert scope.ordered_regeneration_keys
    assert scope.ordered_regeneration_keys.index("diagram.c4_context") < scope.ordered_regeneration_keys.index(
        "diagram.security_guardrails"
    )


def test_invalidation_records_stale_snapshots_and_supersedes_related_uncertainties() -> None:
    workspace_id = uuid4()
    session_id = uuid4()
    with _session() as db:
        db.add(
            UncertaintyBacklogRecord(
                workspace_id=workspace_id,
                session_id=session_id,
                uncertainty_key="question-requirements-context",
                product_mode="premium_enrichment",
                source_stage="define",
                target_stage="design",
                kind="question",
                disposition="resolve_now",
                status=UncertaintyBacklogStatus.open.value,
                title="Actualizar requisitos",
                reason="La definicion cambio.",
                affected_deliverable_keys=["diagram.c4_context"],
            )
        )
        db.flush()

        report = invalidate_deliverables_for_change(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            changed_dependency_keys=["definition.requirements"],
            source_deliverable_key="definition.requirements",
        )
        db.commit()

        snapshots = db.exec(
            select(DeliverableQualitySnapshotRecord).where(
                DeliverableQualitySnapshotRecord.session_id == session_id,
                DeliverableQualitySnapshotRecord.state == "stale",
            )
        ).all()
        backlog = db.exec(select(UncertaintyBacklogRecord)).one()

    assert report.superseded_uncertainty_count == 1
    assert any(snapshot.deliverable_key == "diagram.c4_context" for snapshot in snapshots)
    assert backlog.status == UncertaintyBacklogStatus.superseded.value
