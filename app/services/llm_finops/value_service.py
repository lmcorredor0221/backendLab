from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlmodel import Session, select

from app.models import LLMUsageLedgerRecord, LLMValueAnnotationRecord


class LLMValueAnnotationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usage_record_id: UUID | None = None
    workspace_id: UUID | None = None
    artifact_type: str = ""
    artifact_id: str = ""
    result_type: str = ""
    result_id: str = ""
    stage: str = ""
    decision_key: str = ""
    value_signal: str = ""
    artifact_created: bool = False
    stage_completed: bool = False
    evaluation_passed: bool = False
    human_review_needed: bool = False
    created_by_user_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "artifact_type",
        "artifact_id",
        "result_type",
        "result_id",
        "stage",
        "decision_key",
        "value_signal",
        mode="before",
    )
    @classmethod
    def normalize_string_fields(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


class LLMValueService:
    def annotate_usage_value(
        self,
        session: Session,
        payload: LLMValueAnnotationInput,
    ) -> LLMValueAnnotationRecord | None:
        if payload.usage_record_id is None:
            return None
        usage_record = session.get(LLMUsageLedgerRecord, payload.usage_record_id)
        if usage_record is None:
            return None

        record = LLMValueAnnotationRecord(
            workspace_id=payload.workspace_id or usage_record.workspace_id,
            usage_record_id=usage_record.id,
            artifact_type=payload.artifact_type,
            artifact_id=payload.artifact_id,
            result_type=payload.result_type,
            result_id=payload.result_id,
            stage=payload.stage or usage_record.stage,
            decision_key=payload.decision_key,
            value_signal=payload.value_signal or usage_record.value_signal,
            artifact_created=payload.artifact_created,
            stage_completed=payload.stage_completed,
            evaluation_passed=payload.evaluation_passed,
            human_review_needed=payload.human_review_needed,
            created_by_user_id=payload.created_by_user_id,
            metadata_payload=dict(payload.metadata),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record

    def list_annotations_for_usage(
        self,
        session: Session,
        usage_record_id: UUID,
    ) -> list[LLMValueAnnotationRecord]:
        return list(
            session.exec(
                select(LLMValueAnnotationRecord)
                .where(LLMValueAnnotationRecord.usage_record_id == usage_record_id)
                .order_by(LLMValueAnnotationRecord.created_at.asc())
            ).all()
        )

    def summarize_cost_by_artifact(
        self,
        session: Session,
        *,
        workspace_id: UUID | None = None,
        artifact_type: str = "",
        artifact_id: str = "",
    ) -> list[dict[str, Any]]:
        return self._summarize_cost(
            session,
            dimension="artifact",
            workspace_id=workspace_id,
            type_filter=artifact_type,
            id_filter=artifact_id,
        )

    def summarize_cost_by_result(
        self,
        session: Session,
        *,
        workspace_id: UUID | None = None,
        result_type: str = "",
        result_id: str = "",
    ) -> list[dict[str, Any]]:
        return self._summarize_cost(
            session,
            dimension="result",
            workspace_id=workspace_id,
            type_filter=result_type,
            id_filter=result_id,
        )

    def _summarize_cost(
        self,
        session: Session,
        *,
        dimension: Literal["artifact", "result"],
        workspace_id: UUID | None,
        type_filter: str,
        id_filter: str,
    ) -> list[dict[str, Any]]:
        statement = select(LLMValueAnnotationRecord)
        if workspace_id is not None:
            statement = statement.where(LLMValueAnnotationRecord.workspace_id == workspace_id)
        type_field = "artifact_type" if dimension == "artifact" else "result_type"
        id_field = "artifact_id" if dimension == "artifact" else "result_id"
        if type_filter:
            statement = statement.where(getattr(LLMValueAnnotationRecord, type_field) == type_filter)
        if id_filter:
            statement = statement.where(getattr(LLMValueAnnotationRecord, id_field) == id_filter)

        annotations = list(session.exec(statement).all())
        usage_ids = [item.usage_record_id for item in annotations if item.usage_record_id is not None]
        usage_records = {
            item.id: item
            for item in session.exec(select(LLMUsageLedgerRecord).where(LLMUsageLedgerRecord.id.in_(usage_ids))).all()
        } if usage_ids else {}

        grouped: dict[str, dict[str, Any]] = {}
        for annotation in annotations:
            usage = usage_records.get(annotation.usage_record_id)
            if usage is None:
                continue
            group_type = getattr(annotation, type_field) or "unassigned"
            group_id = getattr(annotation, id_field) or "unassigned"
            key = f"{group_type}:{group_id}"
            bucket = grouped.setdefault(
                key,
                {
                    "dimension": dimension,
                    f"{dimension}_type": group_type,
                    f"{dimension}_id": group_id,
                    "annotation_count": 0,
                    "call_count": 0,
                    "cost_total": 0.0,
                    "total_tokens": 0,
                    "artifact_created_count": 0,
                    "stage_completed_count": 0,
                    "evaluation_passed_count": 0,
                    "human_review_needed_count": 0,
                    "_usage_ids": set(),
                },
            )
            bucket["annotation_count"] += 1
            if usage.id not in bucket["_usage_ids"]:
                bucket["_usage_ids"].add(usage.id)
                bucket["call_count"] += 1
                bucket["cost_total"] = round(bucket["cost_total"] + usage.cost_total, 8)
                bucket["total_tokens"] += usage.total_tokens
            if annotation.artifact_created:
                bucket["artifact_created_count"] += 1
            if annotation.stage_completed:
                bucket["stage_completed_count"] += 1
            if annotation.evaluation_passed:
                bucket["evaluation_passed_count"] += 1
            if annotation.human_review_needed:
                bucket["human_review_needed_count"] += 1

        summaries = []
        for bucket in grouped.values():
            bucket.pop("_usage_ids", None)
            summaries.append(bucket)
        return sorted(
            summaries,
            key=lambda item: (item["cost_total"], item["annotation_count"], item["call_count"]),
            reverse=True,
        )
