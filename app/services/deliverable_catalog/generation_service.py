from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID, uuid4

from sqlmodel import Session, select

from app.db import commit_without_expiring
from app.models import ArtifactRegistryRecord, SessionStage, WorkspaceRole, utc_now
from app.services.deliverable_catalog.contracts import (
    DeliverableGenerationResult,
    DeliverableGenerationTask,
    DeliverablePolicyContext,
)
from app.services.deliverable_catalog.deliverable_generation_agent import DeliverableGenerationAgent, LLMExecutor
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord, DeliverableQualitySnapshotRecord
from app.services.deliverable_catalog.policy_service import resolve_deliverable_policy
from app.services.deliverable_catalog.project_generation_context import BUILDER_VERSION, ProjectGenerationContext
from app.services.deliverable_catalog.prompt_service import get_deliverable_prompt
from app.services.deliverable_catalog.quality_service import record_deliverable_quality_snapshot
from app.services.deliverable_catalog.registry_service import get_registry_entry
from app.services.product_processing.contracts import UncertaintyBacklogStatus
from app.services.product_processing.persistence import UncertaintyBacklogRecord


TERMINAL_RETRYABLE_JOB_STATUSES = {"error", "failed", "requires_attention"}
SOURCE_ACTION = "deliverable_generation_agent"
GENERATION_PROFILE_VERSION = "deliverable-generation-profile.v1"
CACHE_OBSERVATION_SCHEMA_VERSION = "deliverable-generation-cache-observation.v1"


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


def _output_metadata(result: DeliverableGenerationResult) -> dict[str, Any]:
    metadata = result.output_payload.get("metadata") if isinstance(result.output_payload, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


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


def _generation_identity_for_task(task: DeliverableGenerationTask, entry) -> dict[str, Any]:
    if not task.context_payload and not task.approved_context_refs:
        return {
            "input_fingerprint": None,
            "builder_version": BUILDER_VERSION,
            "generation_profile_version": GENERATION_PROFILE_VERSION,
            "source_versions": [],
            "source_refs": [],
            "skip_reason": "context_missing",
        }

    try:
        generation_context = ProjectGenerationContext.from_approved_payload(
            task.context_payload,
            deliverable_key=entry.deliverable_key,
            policy=entry.context_policy,
        )
    except Exception as exc:
        return {
            "input_fingerprint": None,
            "builder_version": BUILDER_VERSION,
            "generation_profile_version": GENERATION_PROFILE_VERSION,
            "source_versions": [],
            "source_refs": [],
            "skip_reason": f"identity_error:{type(exc).__name__}",
        }
    return {
        "input_fingerprint": generation_context.input_fingerprint or None,
        "builder_version": BUILDER_VERSION,
        "generation_profile_version": GENERATION_PROFILE_VERSION,
        "source_versions": [source.model_dump(mode="json") for source in generation_context.source_versions],
        "source_refs": [source.ref for source in generation_context.source_refs],
        "skip_reason": "",
    }


def _source_version_signature(identity: dict[str, Any]) -> list[dict[str, str]]:
    source_versions = identity.get("source_versions")
    if not isinstance(source_versions, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in source_versions:
        if not isinstance(item, dict):
            continue
        source_ref = str(item.get("source_ref") or "").strip()
        if not source_ref:
            continue
        normalized.append({"source_ref": source_ref, "version": str(item.get("version") or "").strip()})
    return sorted(normalized, key=lambda item: (item["source_ref"], item["version"]))


def _cache_observation_for_generation(
    db: Session,
    *,
    task: DeliverableGenerationTask,
    entry,
    current_job: DeliverableGenerationJobRecord,
    identity: dict[str, Any],
) -> dict[str, Any]:
    input_fingerprint = str(identity.get("input_fingerprint") or "").strip()
    builder_version = str(identity.get("builder_version") or "").strip()
    if not input_fingerprint or not builder_version:
        return {
            "schema_version": CACHE_OBSERVATION_SCHEMA_VERSION,
            "mode": "observation",
            "decision": "bypass",
            "reason": identity.get("skip_reason") or "identity_incomplete",
            "candidate_count": 0,
        }

    candidates = db.exec(
        select(DeliverableGenerationJobRecord)
        .where(
            DeliverableGenerationJobRecord.workspace_id == task.workspace_id,
            DeliverableGenerationJobRecord.session_id == task.session_id,
            DeliverableGenerationJobRecord.deliverable_key == task.deliverable_key,
            DeliverableGenerationJobRecord.input_fingerprint == input_fingerprint,
            DeliverableGenerationJobRecord.builder_version == builder_version,
            DeliverableGenerationJobRecord.status == "available",
            DeliverableGenerationJobRecord.id != current_job.id,
        )
        .order_by(DeliverableGenerationJobRecord.completed_at.desc())
    ).all()
    artifact_key = _artifact_key_for_entry(entry)
    artifacts = db.exec(
        select(ArtifactRegistryRecord).where(
            ArtifactRegistryRecord.session_id == task.session_id,
            ArtifactRegistryRecord.artifact_key == artifact_key,
            ArtifactRegistryRecord.source_action == SOURCE_ACTION,
        )
    ).all()
    artifacts_by_job = {
        str(record.artifact_metadata.get("generation_job_id") or ""): record
        for record in artifacts
        if isinstance(record.artifact_metadata, dict)
    }

    current_source_versions = _source_version_signature(identity)
    rejected: dict[str, int] = {}
    valid: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_identity = (
            candidate.request_metadata.get("generation_identity")
            if isinstance(candidate.request_metadata, dict) and isinstance(candidate.request_metadata.get("generation_identity"), dict)
            else {}
        )
        candidate_source_versions = _source_version_signature(candidate_identity)
        if not candidate_source_versions:
            rejected["source_versions_missing"] = rejected.get("source_versions_missing", 0) + 1
            continue
        if candidate_source_versions != current_source_versions:
            rejected["source_versions_changed"] = rejected.get("source_versions_changed", 0) + 1
            continue
        if candidate.output_version_id is None:
            rejected["quality_snapshot_missing"] = rejected.get("quality_snapshot_missing", 0) + 1
            continue
        snapshot = db.get(DeliverableQualitySnapshotRecord, candidate.output_version_id)
        if snapshot is None:
            rejected["quality_snapshot_missing"] = rejected.get("quality_snapshot_missing", 0) + 1
            continue
        if snapshot.state != "passed":
            rejected["quality_not_passed"] = rejected.get("quality_not_passed", 0) + 1
            continue
        artifact = artifacts_by_job.get(str(candidate.id))
        if artifact is None:
            rejected["artifact_missing"] = rejected.get("artifact_missing", 0) + 1
            continue
        valid.append(
            {
                "job_id": str(candidate.id),
                "artifact_id": str(artifact.id),
                "output_version_id": str(candidate.output_version_id),
                "completed_at": candidate.completed_at.isoformat() if candidate.completed_at is not None else "",
                "quality_state": snapshot.state,
                "quality_score": snapshot.score,
                "source_version_count": len(candidate_source_versions),
            }
        )

    observation: dict[str, Any] = {
        "schema_version": CACHE_OBSERVATION_SCHEMA_VERSION,
        "mode": "observation",
        "input_fingerprint_prefix": input_fingerprint[:12],
        "builder_version": builder_version,
        "generation_profile_version": identity.get("generation_profile_version") or GENERATION_PROFILE_VERSION,
        "source_ref_count": len(identity.get("source_refs") if isinstance(identity.get("source_refs"), list) else []),
        "source_version_count": len(current_source_versions),
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "rejected_counts": rejected,
    }
    if len(valid) == 1:
        observation.update({"decision": "would_reuse", "candidate": valid[0]})
    elif len(valid) > 1:
        observation.update({"decision": "bypass", "reason": "ambiguous_candidates"})
    else:
        observation.update({"decision": "bypass", "reason": "no_valid_candidate"})
    return observation


def _cache_observation_with_comparison(
    db: Session,
    *,
    observation: dict[str, Any],
    current_job: DeliverableGenerationJobRecord,
    current_snapshot: DeliverableQualitySnapshotRecord | None,
) -> dict[str, Any]:
    if observation.get("decision") != "would_reuse":
        return {
            **observation,
            "comparison": {
                "performed": False,
                "reason": observation.get("reason") or "no_single_valid_candidate",
            },
        }
    if current_snapshot is None or current_job.output_version_id is None:
        return {
            **observation,
            "comparison": {
                "performed": False,
                "reason": "current_generation_unavailable",
                "current_status": current_job.status,
            },
        }

    candidate = observation.get("candidate") if isinstance(observation.get("candidate"), dict) else {}
    candidate_output_version_id = str(candidate.get("output_version_id") or "").strip()
    try:
        parsed_candidate_snapshot_id = UUID(candidate_output_version_id)
    except (TypeError, ValueError):
        return {
            **observation,
            "comparison": {
                "performed": False,
                "reason": "candidate_snapshot_reference_invalid",
            },
        }
    candidate_snapshot = db.get(DeliverableQualitySnapshotRecord, parsed_candidate_snapshot_id)
    if candidate_snapshot is None:
        return {
            **observation,
            "comparison": {
                "performed": False,
                "reason": "candidate_snapshot_missing_after_generation",
            },
        }

    source_fingerprint_match = candidate_snapshot.source_fingerprint == current_snapshot.source_fingerprint
    quality_state_match = candidate_snapshot.state == current_snapshot.state
    quality_score_delta = current_snapshot.score - candidate_snapshot.score
    return {
        **observation,
        "comparison": {
            "performed": True,
            "current_job_id": str(current_job.id),
            "current_output_version_id": str(current_job.output_version_id),
            "candidate_output_version_id": str(candidate_snapshot.id),
            "source_fingerprint_match": source_fingerprint_match,
            "candidate_source_fingerprint_prefix": candidate_snapshot.source_fingerprint[:12],
            "current_source_fingerprint_prefix": current_snapshot.source_fingerprint[:12],
            "quality_state_match": quality_state_match,
            "candidate_quality_state": candidate_snapshot.state,
            "current_quality_state": current_snapshot.state,
            "candidate_quality_score": candidate_snapshot.score,
            "current_quality_score": current_snapshot.score,
            "quality_score_delta": quality_score_delta,
            "divergence": "none" if source_fingerprint_match and quality_state_match and quality_score_delta == 0 else "observed",
        },
    }


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
    output_metadata = _output_metadata(result)
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
        "context_version": str(output_metadata.get("context_version") or ""),
        "input_fingerprint": str(output_metadata.get("input_fingerprint") or ""),
        "builder_version": str(output_metadata.get("builder_version") or ""),
        "generation_profile_version": str(output_metadata.get("generation_profile_version") or GENERATION_PROFILE_VERSION),
        "estimated_input_tokens": int(output_metadata.get("estimated_input_tokens") or 0),
        "specificity_anchors": list(output_metadata.get("specificity_anchors") or []),
        "missing_fields": list(output_metadata.get("missing_fields") or []),
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

    generation_identity = _generation_identity_for_task(task, entry)
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
    job.input_fingerprint = str(generation_identity.get("input_fingerprint") or "") or None
    job.builder_version = str(generation_identity.get("builder_version") or BUILDER_VERSION) or None
    job.generation_profile_version = str(generation_identity.get("generation_profile_version") or GENERATION_PROFILE_VERSION) or None
    job.status = "generating"
    job.started_at = job.started_at or utc_now()
    job.updated_at = utc_now()
    db.add(job)
    db.flush()
    cache_observation = _cache_observation_for_generation(
        db,
        task=task,
        entry=entry,
        current_job=job,
        identity=generation_identity,
    )
    job.request_metadata = {
        **(job.request_metadata or {}),
        "generation_identity": {
            "input_fingerprint": job.input_fingerprint,
            "builder_version": job.builder_version,
            "generation_profile_version": job.generation_profile_version,
            "source_versions": generation_identity.get("source_versions") if isinstance(generation_identity.get("source_versions"), list) else [],
            "source_refs": generation_identity.get("source_refs") if isinstance(generation_identity.get("source_refs"), list) else [],
        },
        "cache_observation": cache_observation,
    }
    db.add(job)
    db.flush()
    job_id = job.id
    commit_without_expiring(db)

    try:
        result = DeliverableGenerationAgent(llm_executor=llm_executor).run(entry=entry, prompt=prompt, task=task)
    except Exception as exc:
        failed_job = db.get(DeliverableGenerationJobRecord, job_id)
        if failed_job is not None:
            failed_job.status = "error"
            failed_job.error_code = type(exc).__name__
            failed_job.error_message = str(exc)[:700] or "La generacion fallo antes de producir resultado."
            failed_job.completed_at = utc_now()
            failed_job.updated_at = failed_job.completed_at
            db.add(failed_job)
            db.flush()
        raise

    job = db.get(DeliverableGenerationJobRecord, job_id)
    if job is None:
        raise LookupError("Deliverable generation job disappeared before result persistence")
    job.provider_key = result.provider_key
    job.model_name = result.model_name
    job.tokens_input = result.tokens_input
    job.tokens_output = result.tokens_output
    job.estimated_cost_usd = result.estimated_cost_usd
    job.error_code = result.error_code
    job.error_message = result.error_message
    output_metadata = _output_metadata(result)
    job.input_fingerprint = str(output_metadata.get("input_fingerprint") or "") or None
    job.builder_version = str(output_metadata.get("builder_version") or BUILDER_VERSION) or None
    job.generation_profile_version = str(output_metadata.get("generation_profile_version") or GENERATION_PROFILE_VERSION) or None
    job.request_metadata = {
        **(job.request_metadata or {}),
        "result": result.model_dump(mode="json"),
        "generation_identity": {
            "input_fingerprint": job.input_fingerprint,
            "builder_version": job.builder_version,
            "generation_profile_version": job.generation_profile_version,
            "source_versions": generation_identity.get("source_versions") if isinstance(generation_identity.get("source_versions"), list) else [],
            "source_refs": generation_identity.get("source_refs") if isinstance(generation_identity.get("source_refs"), list) else [],
        },
        "public_trace": [step.model_dump(mode="json") for step in result.public_trace],
        "internal_trace_hash": result.internal_trace_hash,
    }
    current_snapshot: DeliverableQualitySnapshotRecord | None = None
    if result.status == "available":
        snapshot = record_deliverable_quality_snapshot(
            db,
            workspace_id=task.workspace_id,
            session_id=task.session_id,
            entry=entry,
            version_ref=f"job::{job.id}",
            payload=result.output_payload,
        )
        current_snapshot = snapshot
        job.output_version_id = snapshot.id
        job.status = "available"
        _upsert_generated_artifact_record(db, task=task, job=job, result=result, entry=entry)
        _supersede_generation_attention(db, task=task)
    elif result.status == "requires_attention":
        _create_generation_attention(db, task=task, result=result)
        job.status = "requires_attention"
    else:
        job.status = "error"
    job.request_metadata = {
        **(job.request_metadata or {}),
        "cache_observation": _cache_observation_with_comparison(
            db,
            observation=cache_observation,
            current_job=job,
            current_snapshot=current_snapshot,
        ),
    }
    job.completed_at = utc_now()
    job.updated_at = utc_now()
    db.add(job)
    db.flush()
    return job, result
