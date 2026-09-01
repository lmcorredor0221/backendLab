from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from sqlmodel import Session, select

from app.models import ArtifactRegistryRecord, SessionStage, WorkspaceRole, utc_now
from app.services.deliverable_catalog.contracts import (
    DeliverableGenerationResult,
    DeliverableGenerationTask,
    DeliverablePolicyContext,
)
from app.services.deliverable_catalog.deliverable_generation_agent import DeliverableGenerationAgent, LLMExecutor
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.deliverable_catalog.policy_service import resolve_deliverable_policy
from app.services.deliverable_catalog.prompt_service import get_deliverable_prompt
from app.services.deliverable_catalog.quality_service import record_deliverable_quality_snapshot
from app.services.deliverable_catalog.registry_service import get_registry_entry
from app.services.product_processing.contracts import UncertaintyBacklogStatus
from app.services.product_processing.persistence import UncertaintyBacklogRecord


TERMINAL_RETRYABLE_JOB_STATUSES = {"error", "failed", "requires_attention"}
SOURCE_ACTION = "deliverable_generation_agent"


def _role_for_generation(task: DeliverableGenerationTask) -> WorkspaceRole:
    return WorkspaceRole.admin if task.requested_by_user_id is None else WorkspaceRole.editor


def _create_generation_attention(
    db: Session,
    *,
    task: DeliverableGenerationTask,
    result: DeliverableGenerationResult,
) -> None:
    uncertainty_key = f"deliverable_generation:{task.deliverable_key}:{result.error_code or result.status}"
    existing = db.exec(
        select(UncertaintyBacklogRecord).where(
            UncertaintyBacklogRecord.workspace_id == task.workspace_id,
            UncertaintyBacklogRecord.session_id == task.session_id,
            UncertaintyBacklogRecord.uncertainty_key == uncertainty_key,
            UncertaintyBacklogRecord.product_mode == task.product_mode,
        )
    ).first()
    record = existing or UncertaintyBacklogRecord(
        workspace_id=task.workspace_id,
        session_id=task.session_id,
        uncertainty_key=uncertainty_key,
        product_mode=task.product_mode,
    )
    record.source_stage = task.current_stage
    record.target_stage = task.current_stage
    record.kind = "gap"
    record.disposition = "resolve_now"
    record.status = UncertaintyBacklogStatus.open.value
    record.title = f"Revisar generacion de {task.deliverable_key}"
    record.reason = result.error_message or result.error_code or "La generacion requiere intervencion humana."
    record.impact = "Puede impedir que el entregable quede listo para el producto seleccionado."
    record.confidence = 0.4
    record.suggested_answer = "Revisar contexto aprobado, prompt, proveedor LLM o fallback antes de regenerar."
    record.affected_deliverable_keys = [task.deliverable_key]
    record.dependency_keys = task.approved_context_refs
    record.created_from = "deliverable_generation_agent"
    record.payload = result.model_dump(mode="json")
    record.updated_at = utc_now()
    db.add(record)
    db.flush()


def _supersede_generation_attention(
    db: Session,
    *,
    task: DeliverableGenerationTask,
) -> None:
    now = utc_now()
    records = db.exec(
        select(UncertaintyBacklogRecord).where(
            UncertaintyBacklogRecord.workspace_id == task.workspace_id,
            UncertaintyBacklogRecord.session_id == task.session_id,
            UncertaintyBacklogRecord.product_mode == task.product_mode,
            UncertaintyBacklogRecord.created_from == SOURCE_ACTION,
            UncertaintyBacklogRecord.status.notin_(
                [
                    UncertaintyBacklogStatus.resolved.value,
                    UncertaintyBacklogStatus.dismissed.value,
                    UncertaintyBacklogStatus.superseded.value,
                ]
            ),
        )
    ).all()
    for record in records:
        affected_keys = [str(value) for value in record.affected_deliverable_keys or []]
        if task.deliverable_key not in affected_keys:
            continue
        record.status = UncertaintyBacklogStatus.superseded.value
        record.superseded_at = now
        record.updated_at = now
        record.payload = {
            **(record.payload or {}),
            "superseded_reason": "deliverable_available",
            "superseded_by_deliverable_key": task.deliverable_key,
        }
        db.add(record)
    db.flush()


def _artifact_key_for_entry(entry) -> str:
    if entry.canonical_paths:
        return entry.canonical_paths[0]
    if entry.portable_paths:
        return entry.portable_paths[0]
    return f"Deliverables/{entry.deliverable_key}.{entry.formats.preferred}"


def _content_hash(content_text: str) -> str:
    return hashlib.sha256(content_text.encode("utf-8")).hexdigest()


def _render_artifact_content(payload: dict[str, object], *, preferred_format: str) -> str:
    if preferred_format.lower() in {"json", "application/json"}:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    title = str(payload.get("title") or "").strip()
    content = str(payload.get("content") or "").strip()
    sections = payload.get("sections")
    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    if content:
        lines.extend([content, ""])
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            section_title = str(section.get("title") or "").strip()
            section_content = str(section.get("content") or "").strip()
            if section_title:
                lines.extend([f"## {section_title}", ""])
            if section_content:
                lines.extend([section_content, ""])
    if lines:
        return "\n".join(lines).strip()
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _blueprint_version_from_task(task: DeliverableGenerationTask) -> int | None:
    value = task.context_payload.get("blueprint_version_number")
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _upsert_generated_artifact_record(
    db: Session,
    *,
    task: DeliverableGenerationTask,
    job: DeliverableGenerationJobRecord,
    result: DeliverableGenerationResult,
    entry,
) -> ArtifactRegistryRecord:
    artifact_key = _artifact_key_for_entry(entry)
    export_format = entry.formats.preferred
    content_text = _render_artifact_content(result.output_payload, preferred_format=export_format)
    existing = db.exec(
        select(ArtifactRegistryRecord).where(
            ArtifactRegistryRecord.session_id == task.session_id,
            ArtifactRegistryRecord.artifact_key == artifact_key,
            ArtifactRegistryRecord.source_action == SOURCE_ACTION,
        )
    ).first()
    record = existing or ArtifactRegistryRecord(
        session_id=task.session_id,
        artifact_key=artifact_key,
        source_action=SOURCE_ACTION,
    )
    metadata = {
        "artifact_key": entry.deliverable_key,
        "deliverable_key": entry.deliverable_key,
        "product": task.tier.value if hasattr(task.tier, "value") else str(task.tier),
        "product_scope": list(entry.product_scope),
        "required_tier": entry.required_tier.value if hasattr(entry.required_tier, "value") else str(entry.required_tier),
        "surface": "governed",
        "category": entry.category,
        "stage_key": entry.stage,
        "enabled_from_stage": entry.enabled_from_stage,
        "generation_mode": entry.generation_mode.value,
        "source_refs": list(task.approved_context_refs),
        "schema_version": str(result.output_payload.get("schema_version") or ""),
        "quality_state": result.quality.state if result.quality is not None else "unknown",
        "quality_score": result.quality.score if result.quality is not None else 0,
        "generation_job_id": str(job.id),
        "output_version_id": str(job.output_version_id) if job.output_version_id is not None else "",
        "prompt_version": result.prompt_version,
        "used_fallback": result.used_fallback,
        "content_length": len(content_text),
    }
    record.blueprint_version_number = _blueprint_version_from_task(task)
    record.artifact_title = entry.title
    record.artifact_kind = entry.deliverable_type.value
    record.stage = SessionStage.ready_for_export
    record.export_format = export_format
    record.content_text = content_text
    record.content_hash = _content_hash(content_text)
    record.artifact_metadata = {
        **metadata,
        "content_hash": record.content_hash,
    }
    db.add(record)
    db.flush()
    return record


def run_deliverable_generation_task(
    db: Session,
    task: DeliverableGenerationTask,
    *,
    llm_executor: LLMExecutor | None = None,
) -> tuple[DeliverableGenerationJobRecord, DeliverableGenerationResult | None]:
    entry = get_registry_entry(task.deliverable_key)
    if entry is None:
        raise LookupError("Deliverable not found")

    original_idempotency_key = task.idempotency_key
    existing_job = db.exec(
        select(DeliverableGenerationJobRecord).where(
            DeliverableGenerationJobRecord.workspace_id == task.workspace_id,
            DeliverableGenerationJobRecord.idempotency_key == original_idempotency_key,
        )
    ).first()
    if existing_job is not None and existing_job.status not in {"queued", "generating", "updating"}:
        if existing_job.status not in TERMINAL_RETRYABLE_JOB_STATUSES:
            return existing_job, None
        task = task.model_copy(update={"idempotency_key": f"{original_idempotency_key}:retry:{uuid4()}"})
        existing_job = None

    prompt = get_deliverable_prompt(db, entry, workspace_id=task.workspace_id)
    access = resolve_deliverable_policy(
        db,
        entry,
        DeliverablePolicyContext(
            workspace_id=task.workspace_id,
            user_id=task.requested_by_user_id,
            role=_role_for_generation(task),
            tier=task.tier,
            current_stage=task.current_stage,
        ),
    )
    if not access.can_generate:
        raise PermissionError(access.reason_code or "deliverable_generation_not_allowed")

    job = existing_job or DeliverableGenerationJobRecord(
        workspace_id=task.workspace_id,
        session_id=task.session_id,
        deliverable_key=task.deliverable_key,
        requested_by_user_id=task.requested_by_user_id,
        product_mode=task.product_mode,
        generation_mode=entry.generation_mode.value,
        idempotency_key=task.idempotency_key,
        prompt_version_id=prompt.versions[0].id if prompt.versions else None,
        request_metadata={"task": task.model_dump(mode="json")},
    )
    job.status = "generating"
    job.started_at = job.started_at or utc_now()
    job.updated_at = utc_now()
    db.add(job)
    db.flush()

    result = DeliverableGenerationAgent(llm_executor=llm_executor).run(entry=entry, prompt=prompt, task=task)
    job.provider_key = result.provider_key
    job.model_name = result.model_name
    job.tokens_input = result.tokens_input
    job.tokens_output = result.tokens_output
    job.estimated_cost_usd = result.estimated_cost_usd
    job.error_code = result.error_code
    job.error_message = result.error_message
    job.request_metadata = {
        **(job.request_metadata or {}),
        "result": result.model_dump(mode="json"),
        "public_trace": [step.model_dump(mode="json") for step in result.public_trace],
        "internal_trace_hash": result.internal_trace_hash,
    }
    if result.status == "available":
        snapshot = record_deliverable_quality_snapshot(
            db,
            workspace_id=task.workspace_id,
            session_id=task.session_id,
            entry=entry,
            version_ref=f"job::{job.id}",
            payload=result.output_payload,
        )
        job.output_version_id = snapshot.id
        job.status = "available"
        _upsert_generated_artifact_record(db, task=task, job=job, result=result, entry=entry)
        _supersede_generation_attention(db, task=task)
    elif result.status == "requires_attention":
        _create_generation_attention(db, task=task, result=result)
        job.status = "requires_attention"
    else:
        job.status = "error"
    job.completed_at = utc_now()
    job.updated_at = utc_now()
    db.add(job)
    db.flush()
    return job, result
