from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import CommercialTier, SessionRecord
from app.services.product_processing.persistence import UncertaintyBacklogRecord
from app.services.product_processing.uncertainty_reconciliation_service import (
    reconcile_confirmed_acp_uncertainties,
)


def _engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
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
