from __future__ import annotations

import json

from sqlmodel import Session, select

from app.models import (
    ArtifactRegistryRecord,
    ConstructionQuestionResponseRecord,
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    SessionRecord,
)
from app.services.deliverable_catalog.contracts import LEAN_STAGE_ORDER
from app.services.deliverable_catalog.registry_service import get_registry_entry


APPROVED_STATES = (JourneyArtifactState.approved, JourneyArtifactState.approved_legacy)
MAX_CONTEXT_CHARS = 24_000
_STAGE_KEYS = set(LEAN_STAGE_ORDER)
_ACP_CONTEXT_STAGE_KEYS = {"validate", "package"}
_ACP_CONTEXT_REFS = {"generated_acp_file", "generated_blueprint_export"}
_SESSION_STAGE_ALIASES = {
    "discovery": "discover",
    "canvas": "define",
    "blueprint": "design",
    "latest_tool_recommendation": "tools",
    "estimation_report": "estimate",
}


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
    stage_keys = {
        stage_key
        for ref in requested_refs
        if (stage_key := _stage_key_from_context_ref(ref)) is not None
    }
    artifact_keys = {ref for ref in requested_refs if _stage_key_from_context_ref(ref) is None}

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

    registry_records = _load_registry_records_for_refs(db, record=record, artifact_keys=artifact_keys)

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
        artifact_context_key = _artifact_context_key(artifact)
        if artifact_context_key in seen_artifact_keys:
            continue
        value, size = _bounded_value(
            {"content": artifact.content_text, "metadata": artifact.artifact_metadata},
            budget - used,
        )
        if size <= 0:
            continue
        seen_artifact_keys.add(artifact_context_key)
        artifacts[artifact_context_key] = value
        used += size
        refs.append(f"artifact:{artifact.id}")

    if not refs:
        snapshot_context, snapshot_refs = _build_snapshot_fallback_context(
            db,
            record=record,
            deliverable_key=deliverable_key,
            requested_refs=requested_refs,
            max_context_tokens=context_policy.max_context_tokens,
        )
        if snapshot_refs:
            return snapshot_context, snapshot_refs
        if _needs_acp_package_context(deliverable_key=deliverable_key, requested_refs=requested_refs, stage_keys=stage_keys):
            return _build_acp_package_fallback_context(
                db,
                record=record,
                deliverable_key=deliverable_key,
                requested_refs=requested_refs,
                max_context_tokens=context_policy.max_context_tokens,
            )
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


def _stage_key_from_context_ref(ref: str) -> str | None:
    normalized = str(ref or "").strip()
    if normalized.startswith("stage."):
        candidate = normalized.removeprefix("stage.")
    elif normalized.startswith("session."):
        candidate = normalized.removeprefix("session.")
    else:
        return None
    candidate = _SESSION_STAGE_ALIASES.get(candidate, candidate)
    return candidate if candidate in _STAGE_KEYS else None


def _load_registry_records_for_refs(
    db: Session,
    *,
    record: SessionRecord,
    artifact_keys: set[str],
) -> list[ArtifactRegistryRecord]:
    if not artifact_keys:
        return []

    candidates = db.exec(
        select(ArtifactRegistryRecord)
        .where(ArtifactRegistryRecord.session_id == record.id)
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).all()
    selected: list[ArtifactRegistryRecord] = []
    seen: set[str] = set()
    for artifact in candidates:
        aliases = _artifact_aliases(artifact)
        if not aliases.intersection(artifact_keys):
            continue
        context_key = _artifact_context_key(artifact)
        if context_key in seen:
            continue
        seen.add(context_key)
        selected.append(artifact)
    return selected


def _artifact_aliases(artifact: ArtifactRegistryRecord) -> set[str]:
    aliases = {str(artifact.artifact_key or "").strip()}
    metadata = artifact.artifact_metadata or {}
    for key in ("deliverable_key", "artifact_key", "catalog_key"):
        value = metadata.get(key)
        if value is not None:
            aliases.add(str(value).strip())
    return {alias for alias in aliases if alias}


def _artifact_context_key(artifact: ArtifactRegistryRecord) -> str:
    metadata = artifact.artifact_metadata or {}
    for key in ("deliverable_key", "artifact_key", "catalog_key"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return str(artifact.artifact_key or "").strip()


def _needs_acp_package_context(
    *,
    deliverable_key: str,
    requested_refs: list[str],
    stage_keys: set[str],
) -> bool:
    if stage_keys.intersection(_ACP_CONTEXT_STAGE_KEYS):
        return True
    if set(requested_refs).intersection(_ACP_CONTEXT_REFS):
        return True
    return deliverable_key.startswith(("acp.", "validation.", "evaluation."))


def _build_acp_package_fallback_context(
    db: Session,
    *,
    record: SessionRecord,
    deliverable_key: str,
    requested_refs: list[str],
    max_context_tokens: int,
) -> tuple[dict[str, object], list[str]]:
    """Build ACP construction context from generated product artifacts and resolved questions."""

    budget = min(MAX_CONTEXT_CHARS, max(1_000, int(max_context_tokens or 5_000) * 4))
    used = 0
    refs: list[str] = []
    artifacts: dict[str, object] = {}
    questions: list[dict[str, object]] = []

    artifact_records = db.exec(
        select(ArtifactRegistryRecord)
        .where(ArtifactRegistryRecord.session_id == record.id)
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).all()
    seen_artifact_keys: set[str] = set()
    for artifact in artifact_records:
        context_key = _artifact_context_key(artifact)
        if context_key in seen_artifact_keys:
            continue
        value, size = _bounded_value(
            {"content": artifact.content_text, "metadata": artifact.artifact_metadata},
            budget - used,
        )
        if size <= 0:
            break
        seen_artifact_keys.add(context_key)
        artifacts[context_key] = value
        used += size
        refs.append(f"artifact:{artifact.id}")

    question_records = db.exec(
        select(ConstructionQuestionResponseRecord)
        .where(ConstructionQuestionResponseRecord.session_id == record.id)
        .order_by(ConstructionQuestionResponseRecord.updated_at.desc())
    ).all()
    for question in question_records:
        value, size = _bounded_value(
            {
                "question_key": question.question_key,
                "gap_key": question.gap_key,
                "gap_title": question.gap_title,
                "domain": question.domain,
                "question_text": question.question_text,
                "rationale": question.rationale,
                "expected_answer_format": question.expected_answer_format,
                "target_owner": question.target_owner,
                "blocking": question.blocking,
                "status": question.status,
                "answer_text": question.answer_text,
                "owner_role": question.owner_role,
                "impacted_artifacts": question.impacted_artifacts,
                "decision_context": question.decision_context,
            },
            budget - used,
        )
        if size <= 0:
            break
        questions.append(value if isinstance(value, dict) else {"value": value})
        used += size
        refs.append(f"acp-question:{question.id}")

    if not refs:
        return {}, []

    return (
        {
            "summary": f"Contexto ACP consolidado para {deliverable_key}.",
            "project_title": record.title,
            "deliverable_key": deliverable_key,
            "context_policy": {
                "retrieval_strategy": "acp_package_context_from_product_artifacts_and_readiness_questions",
                "requested_refs": requested_refs,
                "max_context_tokens": max_context_tokens,
            },
            "approved_context": {
                "artifacts": artifacts,
                "construction_questions": questions,
            },
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
