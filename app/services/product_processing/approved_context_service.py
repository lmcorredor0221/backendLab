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
        return _build_snapshot_fallback_context(
            db,
            record=record,
            deliverable_key=deliverable_key,
            requested_refs=requested_refs,
            max_context_tokens=context_policy.max_context_tokens,
        )

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


def _build_snapshot_fallback_context(
    db: Session,
    *,
    record: SessionRecord,
    deliverable_key: str,
    requested_refs: list[str],
    max_context_tokens: int,
) -> tuple[dict[str, object], list[str]]:
    """Fallback for migrated sessions without formal approved stage artifacts."""

    from app.api.routes.sessions import build_snapshot

    snapshot = build_snapshot(db, record, current_user=None)
    context = _snapshot_context_payload(snapshot)
    value, size = _bounded_value(context, min(MAX_CONTEXT_CHARS, max(1_000, int(max_context_tokens or 5_000) * 4)))
    refs = _snapshot_refs(snapshot)
    if size <= 0 or not refs:
        return {}, []
    return (
        {
            "summary": f"Contexto consolidado de la sesion para {deliverable_key}.",
            "project_title": record.title,
            "deliverable_key": deliverable_key,
            "context_policy": {
                "retrieval_strategy": "approved_snapshot_fallback",
                "requested_refs": requested_refs,
                "max_context_tokens": max_context_tokens,
            },
            "approved_context": {"snapshot": value},
        },
        refs,
    )


def _snapshot_refs(snapshot: object) -> list[str]:
    refs: list[str] = []
    for key in ("discovery", "canvas", "blueprint", "latest_tool_recommendation", "estimation_report"):
        if getattr(snapshot, key, None) is not None:
            refs.append(f"session.{key}")
    return refs


def _snapshot_context_payload(snapshot: object) -> dict[str, object]:
    discovery = getattr(snapshot, "discovery", None)
    canvas = getattr(snapshot, "canvas", None)
    blueprint = getattr(snapshot, "blueprint", None)
    estimation = getattr(snapshot, "estimation_report", None)
    session = getattr(snapshot, "session", None)
    tools = list(getattr(blueprint, "tools", []) or []) if blueprint is not None else []
    return {
        "session_id": str(getattr(session, "id", "")),
        "workspace_id": str(getattr(session, "workspace_id", "")),
        "session_title": getattr(session, "title", "") or "Agente Inteligente",
        "problem_statement": getattr(discovery, "problem_statement", "") if discovery else "",
        "current_process": getattr(discovery, "current_process", "") if discovery else "",
        "current_user": getattr(discovery, "current_user", "") if discovery else "",
        "desired_outcome": getattr(discovery, "desired_outcome", "") if discovery else "",
        "value_statement": getattr(discovery, "value_statement", "") if discovery else "",
        "autonomy_level": getattr(discovery, "autonomy_level", "") if discovery else "",
        "constraints": list(getattr(discovery, "constraints", []) or []) if discovery else [],
        "user_goal": getattr(canvas, "user_goal", "") if canvas else "",
        "mvp_scope": list(getattr(canvas, "mvp_scope", []) or []) if canvas else [],
        "out_of_scope": list(getattr(canvas, "out_of_scope", []) or []) if canvas else [],
        "primary_risk": getattr(canvas, "primary_risk", "") if canvas else "",
        "success_metric": getattr(canvas, "success_metric", "") if canvas else "",
        "architecture": getattr(blueprint, "architecture", "") if blueprint else "",
        "reasoning_pattern": getattr(blueprint, "reasoning_pattern", "") if blueprint else "",
        "memory_strategy": getattr(blueprint, "memory_strategy", "") if blueprint else "",
        "guardrails": list(getattr(blueprint, "guardrails", []) or []) if blueprint else [],
        "narrative": getattr(blueprint, "narrative", "") if blueprint else "",
        "tools": [
            {
                "name": getattr(tool, "name", ""),
                "purpose": getattr(tool, "purpose", ""),
                "requires_approval": bool(getattr(tool, "requires_approval", False)),
                "has_side_effects": bool(getattr(tool, "has_side_effects", False)),
                "inputs": list(getattr(tool, "inputs", []) or []),
                "outputs": list(getattr(tool, "outputs", []) or []),
            }
            for tool in tools
        ],
        "tool_count": len(tools),
        "estimation_report": estimation.model_dump(mode="json") if hasattr(estimation, "model_dump") else {},
    }
