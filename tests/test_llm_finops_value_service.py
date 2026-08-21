from __future__ import annotations

from uuid import uuid4

from sqlmodel import SQLModel, Session, create_engine, select

from app.models import LLMUsageLedgerRecord, LLMValueAnnotationRecord
from app.services.llm_finops import (
    LLMCallContext,
    LLMUsageCostBreakdown,
    LLMUsageRecordInput,
    NormalizedLLMUsage,
)
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_finops.value_service import LLMValueAnnotationInput, LLMValueService


def build_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def record_usage(db: Session, *, workspace_id=None, request_id: str = "req-value-1", cost_total: float = 0.12):
    workspace_id = workspace_id or uuid4()
    result = LLMUsageLedgerService().record_call(
        db,
        LLMUsageRecordInput(
            context=LLMCallContext(
                workspace_id=workspace_id,
                stage="define",
                agent_key="requirements_builder",
                capability_key="define_requirements",
                action_key="define_requirements",
            ),
            provider_key="openai",
            model_name="gpt-test",
            requested_model="gpt-test",
            execution_backend="provider_native",
            request_id=request_id,
            usage=NormalizedLLMUsage(input_tokens=100, output_tokens=50),
            cost=LLMUsageCostBreakdown(cost_total=cost_total, currency="USD"),
            value_signal="requirements_ready",
        ),
    )
    record = db.get(LLMUsageLedgerRecord, result.usage_record_id)
    assert record is not None
    return record


def test_annotate_usage_value_links_consumption_to_artifact_and_stage_outcome() -> None:
    db = build_session()
    workspace_id = uuid4()
    usage_record = record_usage(db, workspace_id=workspace_id)
    service = LLMValueService()

    annotation = service.annotate_usage_value(
        db,
        LLMValueAnnotationInput(
            usage_record_id=usage_record.id,
            artifact_type="requirements",
            artifact_id="req-pack-1",
            result_type="stage",
            result_id="define",
            artifact_created=True,
            stage_completed=True,
            evaluation_passed=True,
            metadata={"review": "automatic"},
        ),
    )
    annotations = service.list_annotations_for_usage(db, usage_record.id)

    assert annotation is not None
    assert annotation.workspace_id == workspace_id
    assert annotation.stage == "define"
    assert annotation.value_signal == "requirements_ready"
    assert annotation.metadata_payload == {"review": "automatic"}
    assert len(annotations) == 1


def test_value_service_summarizes_cost_by_artifact_and_result() -> None:
    db = build_session()
    workspace_id = uuid4()
    first_usage = record_usage(db, workspace_id=workspace_id, request_id="req-value-1", cost_total=0.15)
    second_usage = record_usage(db, workspace_id=workspace_id, request_id="req-value-2", cost_total=0.35)
    service = LLMValueService()

    service.annotate_usage_value(
        db,
        LLMValueAnnotationInput(
            usage_record_id=first_usage.id,
            artifact_type="blueprint",
            artifact_id="bp-1",
            result_type="evaluation",
            result_id="eval-1",
            artifact_created=True,
            evaluation_passed=True,
        ),
    )
    service.annotate_usage_value(
        db,
        LLMValueAnnotationInput(
            usage_record_id=second_usage.id,
            artifact_type="blueprint",
            artifact_id="bp-1",
            result_type="evaluation",
            result_id="eval-1",
            human_review_needed=True,
        ),
    )

    artifact_summary = service.summarize_cost_by_artifact(
        db,
        workspace_id=workspace_id,
        artifact_type="blueprint",
        artifact_id="bp-1",
    )
    result_summary = service.summarize_cost_by_result(db, workspace_id=workspace_id)

    assert artifact_summary == [
        {
            "dimension": "artifact",
            "artifact_type": "blueprint",
            "artifact_id": "bp-1",
            "annotation_count": 2,
            "call_count": 2,
            "cost_total": 0.5,
            "total_tokens": 300,
            "artifact_created_count": 1,
            "stage_completed_count": 0,
            "evaluation_passed_count": 1,
            "human_review_needed_count": 1,
        }
    ]
    assert result_summary[0]["result_type"] == "evaluation"
    assert result_summary[0]["result_id"] == "eval-1"
    assert result_summary[0]["cost_total"] == 0.5


def test_value_annotation_is_optional_and_missing_usage_does_not_block() -> None:
    db = build_session()
    service = LLMValueService()

    missing = service.annotate_usage_value(
        db,
        LLMValueAnnotationInput(
            usage_record_id=uuid4(),
            artifact_type="decision",
            artifact_id="decision-1",
        ),
    )
    summary = service.summarize_cost_by_artifact(db)
    records = db.exec(select(LLMValueAnnotationRecord)).all()

    assert missing is None
    assert summary == []
    assert records == []
