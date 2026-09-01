from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models import ArtifactRegistryRecord, CommercialTier, WorkspaceRole
from app.services.deliverable_catalog import (
    DeliverableGenerationTask,
    DeliverableGovernanceUpdate,
    build_deliverable_catalog_response,
    get_registry_entry,
    run_deliverable_generation_task,
    upsert_deliverable_governance,
)
from app.services.deliverable_catalog.persistence import (
    DeliverableGenerationJobRecord,
    DeliverableQualitySnapshotRecord,
)
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


def _task(*, context: dict[str, object] | None = None) -> DeliverableGenerationTask:
    return DeliverableGenerationTask(
        workspace_id=uuid4(),
        session_id=uuid4(),
        deliverable_key="discovery.problem_context_brief",
        product_mode="basic_free",
        current_stage="discover",
        tier=CommercialTier.blueprint,
        idempotency_key=f"job-{uuid4()}",
        context_payload=context or {},
        approved_context_refs=["session.discovery"],
        allow_llm=False,
    )


def test_generation_service_runs_react_fallback_and_records_quality_snapshot() -> None:
    with _session() as db:
        task = _task(context={"summary": "El usuario necesita un agente para clasificar solicitudes internas."})

        job, result = run_deliverable_generation_task(db, task)
        job_status = job.status
        db.commit()
        snapshots = db.exec(select(DeliverableQualitySnapshotRecord)).all()
        jobs = db.exec(select(DeliverableGenerationJobRecord)).all()
        artifact = next(
            record
            for record in db.exec(
                select(ArtifactRegistryRecord).where(ArtifactRegistryRecord.session_id == task.session_id)
            ).all()
            if record.artifact_metadata.get("deliverable_key") == "discovery.problem_context_brief"
        )
        catalog = build_deliverable_catalog_response(
            db,
            workspace_id=task.workspace_id,
            session_id=task.session_id,
            role=WorkspaceRole.admin,
            tier=CommercialTier.blueprint,
            current_stage="discover",
        )

    assert result is not None
    assert result.status == "available"
    assert result.used_fallback is True
    assert job_status == "available"
    assert jobs[0].output_version_id == snapshots[0].id
    assert snapshots[0].state == "passed"
    assert artifact.source_action == "deliverable_generation_agent"
    assert artifact.artifact_metadata["deliverable_key"] == "discovery.problem_context_brief"
    assert "El usuario necesita un agente" in artifact.content_text
    assert "session.discovery" in artifact.content_text
    catalog_item = next(item for item in catalog.entries if item.key == "discovery.problem_context_brief")
    assert catalog_item.access.access_state == "available"
    assert catalog_item.access.can_view is True
    assert [step.step for step in result.public_trace] == ["reason", "act", "observe", "evaluate", "finish"]


def test_generation_service_creates_attention_when_context_is_missing() -> None:
    with _session() as db:
        task = DeliverableGenerationTask(
            workspace_id=uuid4(),
            session_id=uuid4(),
            deliverable_key="discovery.problem_context_brief",
            current_stage="discover",
            tier=CommercialTier.blueprint,
            idempotency_key=f"job-{uuid4()}",
        )

        job, result = run_deliverable_generation_task(db, task)
        job_status = job.status
        db.commit()
        backlog = db.exec(select(UncertaintyBacklogRecord)).one()

    assert result is not None
    assert result.status == "requires_attention"
    assert result.error_code == "context_missing"
    assert job_status == "requires_attention"
    assert backlog.status == UncertaintyBacklogStatus.open.value
    assert backlog.affected_deliverable_keys == ["discovery.problem_context_brief"]


def test_generation_service_retries_retryable_terminal_job_with_same_intention() -> None:
    workspace_id = uuid4()
    session_id = uuid4()
    idempotency_key = f"job-{uuid4()}"
    with _session() as db:
        missing_context_task = DeliverableGenerationTask(
            workspace_id=workspace_id,
            session_id=session_id,
            deliverable_key="discovery.problem_context_brief",
            current_stage="discover",
            tier=CommercialTier.blueprint,
            idempotency_key=idempotency_key,
        )
        first_job, first_result = run_deliverable_generation_task(db, missing_context_task)
        db.commit()

        retry_task = DeliverableGenerationTask(
            workspace_id=workspace_id,
            session_id=session_id,
            deliverable_key="discovery.problem_context_brief",
            current_stage="discover",
            tier=CommercialTier.blueprint,
            idempotency_key=idempotency_key,
            context_payload={"summary": "Ahora existe contexto aprobado suficiente para publicar el entregable."},
            approved_context_refs=["session.discovery"],
            allow_llm=False,
        )
        retry_job, retry_result = run_deliverable_generation_task(db, retry_task)
        db.commit()
        jobs = db.exec(
            select(DeliverableGenerationJobRecord)
            .where(DeliverableGenerationJobRecord.session_id == session_id)
            .order_by(DeliverableGenerationJobRecord.requested_at)
        ).all()
        backlog = db.exec(
            select(UncertaintyBacklogRecord).where(UncertaintyBacklogRecord.session_id == session_id)
        ).one()
        artifact = next(
            record
            for record in db.exec(select(ArtifactRegistryRecord).where(ArtifactRegistryRecord.session_id == session_id)).all()
            if record.artifact_metadata.get("deliverable_key") == "discovery.problem_context_brief"
        )

    assert first_result is not None
    assert first_result.status == "requires_attention"
    assert first_job.status == "requires_attention"
    assert retry_result is not None
    assert retry_result.status == "available"
    assert retry_job.status == "available"
    assert retry_job.idempotency_key.startswith(f"{idempotency_key}:retry:")
    assert [job.status for job in jobs] == ["requires_attention", "available"]
    assert backlog.status == UncertaintyBacklogStatus.superseded.value
    assert backlog.superseded_at is not None
    assert backlog.payload["superseded_reason"] == "deliverable_available"
    assert artifact.source_action == "deliverable_generation_agent"


def test_generation_service_respects_paused_prompt_policy() -> None:
    with _session() as db:
        entry = get_registry_entry("discovery.problem_context_brief")
        assert entry is not None
        task = _task(context={"summary": "Contexto suficiente."})
        upsert_deliverable_governance(
            db,
            entry,
            DeliverableGovernanceUpdate(prompt_status="paused"),
            workspace_id=task.workspace_id,
        )
        db.commit()

        with pytest.raises(PermissionError):
            run_deliverable_generation_task(db, task)
