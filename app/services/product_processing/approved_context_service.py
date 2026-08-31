from __future__ import annotations

import json

from sqlmodel import Session, select

from app.models import ArtifactRegistryRecord, JourneyArtifactState, JourneyStageArtifactRecord, SessionRecord
from app.services.deliverable_catalog.registry_service import get_registry_entry


APPROVED_STATES = (JourneyArtifactState.approved, JourneyArtifactState.approved_legacy)
MAX_CONTEXT_CHARS = 24_000


def build_approved_deliverable_context(
    db: Session,
    *,
    record: SessionRecord,
    deliverable_key: str,
) -> tuple[dict[str, object], list[str]]:
    """Build a bounded, traceable context from the sources allowed by the catalog."""
    entry = get_registry_entry(deliverable_key)
    if entry is None:
        return {}, []

    context_policy = entry.context_policy
    requested_refs = list(dict.fromkeys([*context_policy.short_term_refs, *entry.dependency_policy.depends_on]))
    stage_keys = {ref.removeprefix("session.") for ref in requested_refs if ref.startswith("session.")}
    artifact_keys = {ref for ref in requested_refs if not ref.startswith("session.")}

    approved_records = db.exec(
        select(JourneyStageArtifactRecord)
        .where(
            JourneyStageArtifactRecord.session_id == record.id,
            JourneyStageArtifactRecord.state.in_(APPROVED_STATES),
        )
        .order_by(JourneyStageArtifactRecord.stage_key.asc(), JourneyStageArtifactRecord.version_number.desc())
    ).all()
    latest_by_stage: dict[str, JourneyStageArtifactRecord] = {}
    for artifact in approved_records:
        if artifact.stage_key in stage_keys and artifact.stage_key not in latest_by_stage:
            latest_by_stage[artifact.stage_key] = artifact

    registry_records = []
    if artifact_keys:
        registry_records = db.exec(
            select(ArtifactRegistryRecord)
            .where(
                ArtifactRegistryRecord.session_id == record.id,
                ArtifactRegistryRecord.artifact_key.in_(tuple(artifact_keys)),
            )
            .order_by(ArtifactRegistryRecord.artifact_key.asc(), ArtifactRegistryRecord.created_at.desc())
        ).all()

    refs: list[str] = []
    stages: dict[str, object] = {}
    artifacts: dict[str, object] = {}
    budget = min(MAX_CONTEXT_CHARS, max(1_000, int(context_policy.max_context_tokens or 5_000) * 4))
    used = 0

    for stage_key in sorted(latest_by_stage):
        artifact = latest_by_stage[stage_key]
        value, size = _bounded_value(artifact.proposal_payload, budget - used)
        if size <= 0:
            continue
        stages[stage_key] = value
        used += size
        refs.append(f"journey:{artifact.id}:v{artifact.version_number}")

    seen_artifact_keys: set[str] = set()
    for artifact in registry_records:
        if artifact.artifact_key in seen_artifact_keys:
            continue
        value, size = _bounded_value(
            {"content": artifact.content_text, "metadata": artifact.artifact_metadata},
            budget - used,
        )
        if size <= 0:
            continue
        seen_artifact_keys.add(artifact.artifact_key)
        artifacts[artifact.artifact_key] = value
        used += size
        refs.append(f"artifact:{artifact.id}")

    if not refs:
        return {}, []

    return (
        {
            "summary": f"Contexto aprobado y acotado para {entry.title}.",
            "project_title": record.title,
            "deliverable_key": deliverable_key,
            "context_policy": {
                "retrieval_strategy": context_policy.retrieval_strategy,
                "requested_refs": requested_refs,
                "max_context_tokens": context_policy.max_context_tokens,
            },
            "approved_context": {"stages": stages, "artifacts": artifacts},
        },
        refs,
    )


def _bounded_value(value: object, remaining_chars: int) -> tuple[object, int]:
    if remaining_chars <= 0:
        return {}, 0
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    if len(serialized) <= remaining_chars:
        return value, len(serialized)
    # Preserve a valid and explicit partial representation rather than silently dropping evidence.
    return {"truncated": True, "content": serialized[:remaining_chars]}, remaining_chars
