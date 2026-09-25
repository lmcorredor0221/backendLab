from uuid import uuid4
from types import SimpleNamespace
import threading
import time

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import CommercialTier, SessionRecord
from app.services.deliverable_catalog.contracts import (
    DeliverableGenerationResult,
    DeliverableRegenerationScope,
)
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.product_processing.persistence import UncertaintyBacklogRecord
from app.services.product_processing.uncertainty_reconciliation_service import (
    UncertaintyReconciliationPlan,
    execute_uncertainty_reconciliation,
    reconcile_confirmed_acp_uncertainties,
)


def _engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    return engine


def _file_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'uncertainty-reconciliation.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def test_acp_reconciliation_ignores_unconfirmed_inherited_uncertainties() -> None:
    engine = _engine()
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()
    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="ACP sin confirmacion",
                commercial_tier=CommercialTier.acp,
            )
        )
        record = UncertaintyBacklogRecord(
            workspace_id=workspace_id,
            session_id=session_id,
            uncertainty_key="runtime_owner",
            product_mode="basic_free",
            source_stage="define",
            target_stage="acp",
            disposition="infer",
            status="deferred",
            title="Confirmar responsable runtime",
            dependency_keys=["definition.requirements"],
        )
        db.add(record)
        db.commit()

        summary = reconcile_confirmed_acp_uncertainties(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            actor_user_id=actor_id,
        )
        db.refresh(record)

    assert summary.reviewed_backlog_ids == []
    assert record.payload.get("acp_reconciliation") is None
    assert record.status == "deferred"


def test_acp_reconciliation_runs_only_after_answer_and_records_auditable_checkpoint() -> None:
    engine = _engine()
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()
    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="ACP con respuesta confirmada",
                commercial_tier=CommercialTier.acp,
            )
        )
        record = UncertaintyBacklogRecord(
            workspace_id=workspace_id,
            session_id=session_id,
            uncertainty_key="orchestration_owner",
            product_mode="basic_free",
            source_stage="define",
            target_stage="acp",
            disposition="infer",
            status="resolved",
            title="Confirmar orquestacion",
            assumed_answer="Usar un orquestador con handoffs trazables.",
            dependency_keys=["definition.requirements"],
            payload={"acp_resolution": {"decision": "answer", "answered_by_display": "Usuario ACP"}},
        )
        db.add(record)
        db.commit()

        summary = reconcile_confirmed_acp_uncertainties(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            actor_user_id=actor_id,
            max_deliverables=1,
        )
        db.refresh(record)

    assert summary.reviewed_backlog_ids == [str(record.id)]
    assert summary.reconciled_backlog_ids == [str(record.id)]
    assert summary.queue_total == 1
    checkpoint = record.payload["acp_reconciliation"]
    assert checkpoint["reconciliation_status"] == "completed"
    assert checkpoint["queue_completed"] == 1
    assert checkpoint["generation_job_ids"]


def test_uncertainty_reconciliation_runs_selected_deliverables_in_batches_of_three(
    monkeypatch,
    tmp_path,
) -> None:
    engine = _file_engine(tmp_path)
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()
    queue = [
        "discovery.problem_context_brief",
        "discovery.stakeholder_inventory",
        "definition.acceptance_trace",
        "design.architecture_options",
        "tools.minimum_set",
    ]
    monkeypatch.setattr(
        "app.services.product_processing.uncertainty_reconciliation_service.get_settings",
        lambda: SimpleNamespace(product_build_batch_size=3),
    )
    monkeypatch.setattr(
        "app.services.product_processing.uncertainty_reconciliation_service.invalidate_deliverables_for_change",
        lambda *args, **kwargs: SimpleNamespace(stale_deliverable_keys=[], superseded_uncertainty_count=0),
    )
    monkeypatch.setattr(
        "app.services.product_processing.uncertainty_reconciliation_service.build_uncertainty_reconciliation_plan",
        lambda record: UncertaintyReconciliationPlan(
            changed_dependency_keys=["definition.requirements"],
            scope=DeliverableRegenerationScope(
                source_deliverable_key=queue[0],
                changed_dependency_keys=["definition.requirements"],
                affected_deliverable_keys=list(queue),
                ordered_regeneration_keys=list(queue),
            ),
            reconciliation_decision="structural_reconciliation",
            material_impact=True,
            recommended_action="review_and_apply_structural_reconciliation",
            impact_summary="Fixture batch reconciliation.",
        ),
    )
    concurrency_lock = threading.Lock()
    active_workers = 0
    max_active_workers = 0

    def fake_runner(db: Session, task):
        nonlocal active_workers, max_active_workers
        with concurrency_lock:
            active_workers += 1
            max_active_workers = max(max_active_workers, active_workers)
        try:
            time.sleep(0.05)
            job = DeliverableGenerationJobRecord(
                workspace_id=task.workspace_id,
                session_id=task.session_id,
                deliverable_key=task.deliverable_key,
                status="available",
                product_mode=task.product_mode,
                idempotency_key=task.idempotency_key,
            )
            db.add(job)
            db.flush()
            return job, DeliverableGenerationResult(deliverable_key=task.deliverable_key, status="available")
        finally:
            with concurrency_lock:
                active_workers -= 1

    monkeypatch.setattr(
        "app.services.product_processing.uncertainty_reconciliation_service.run_deliverable_generation_task",
        fake_runner,
    )

    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="ACP batch reconciliation",
                commercial_tier=CommercialTier.acp,
            )
        )
        record = UncertaintyBacklogRecord(
            workspace_id=workspace_id,
            session_id=session_id,
            uncertainty_key="batch_reconciliation",
            product_mode="basic_free",
            source_stage="define",
            target_stage="acp",
            disposition="infer",
            status="resolved",
            title="Confirmar alcance batch",
            assumed_answer="Usar batch seguro.",
            dependency_keys=["definition.requirements"],
        )
        db.add(record)
        db.commit()

        result = execute_uncertainty_reconciliation(
            db,
            record=record,
            actor_user_id=actor_id,
            answer="Usar batch seguro.",
            resolution_key="acp_reconciliation",
            product_mode="acp_implementation",
            tier=CommercialTier.acp,
            idempotency_prefix=f"acp-reconciliation:{session_id}",
            max_deliverables=5,
        )

    assert result.queue_total == 5
    assert result.queue_completed == 5
    assert max_active_workers > 1
    assert max_active_workers <= 3
