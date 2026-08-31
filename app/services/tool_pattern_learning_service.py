from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CandidateToolPattern,
    ToolPatternLearningCandidate,
    ToolPatternLearningCandidateRecord,
    ToolPatternLearningPersistenceSummary,
    ToolRecommendationArtifact,
    utc_now,
)


def _candidate_contract_payload(
    candidate: ToolPatternLearningCandidate,
    *,
    patterns_by_id: dict[str, CandidateToolPattern],
) -> dict[str, object]:
    pattern = patterns_by_id.get(candidate.candidate_pattern_id)
    if pattern is None or pattern.contract_seed is None:
        return {}
    return pattern.contract_seed.model_dump(mode="json")


def _candidate_metadata_payload(candidate: ToolPatternLearningCandidate) -> dict[str, object]:
    return {
        "global_write_allowed": candidate.global_promotion_allowed,
        "replacement_global_pattern_id": candidate.replacement_global_pattern_id,
        "risk_flags": list(candidate.risk_flags),
        "reason": candidate.reason,
    }


def _apply_candidate_to_record(
    record: ToolPatternLearningCandidateRecord,
    candidate: ToolPatternLearningCandidate,
    *,
    source_artifact_id: UUID | None,
    source_blueprint_version: int | None,
    contract_seed_payload: dict[str, object],
    now,
) -> None:
    record.source_artifact_id = source_artifact_id
    record.source_blueprint_version = source_blueprint_version
    record.candidate_pattern_id = candidate.candidate_pattern_id
    record.capability_key = candidate.capability_key
    record.family_key = candidate.family_key
    record.label = candidate.label
    record.source_level = candidate.source_level
    record.promotion_status = candidate.promotion_status
    record.global_promotion_allowed = candidate.global_promotion_allowed
    record.replacement_global_pattern_id = candidate.replacement_global_pattern_id
    record.contract_quality = candidate.contract_quality
    record.risk_flags = list(candidate.risk_flags)
    record.source_refs = list(candidate.source_refs)
    record.evidence_refs = list(candidate.evidence_refs)
    record.contract_seed_payload = contract_seed_payload
    record.metadata_payload = _candidate_metadata_payload(candidate)
    record.last_seen_at = now
    record.updated_at = now


def persist_tool_pattern_learning_candidates(
    session: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    recommendation: ToolRecommendationArtifact,
    source_artifact_id: UUID | None = None,
) -> ToolPatternLearningPersistenceSummary:
    """Persist the governed learning queue without promoting anything globally."""

    report = recommendation.learning_report
    candidates = [item for item in report.candidates if item.dedupe_signature.strip()]
    skipped_count = report.candidate_count - len(candidates)
    if not candidates:
        return ToolPatternLearningPersistenceSummary(
            skipped_count=max(skipped_count, 0),
            summary="No tool pattern learning candidates to persist.",
        )

    signatures = [item.dedupe_signature for item in candidates]
    existing_rows = session.exec(
        select(ToolPatternLearningCandidateRecord).where(
            ToolPatternLearningCandidateRecord.workspace_id == workspace_id,
            ToolPatternLearningCandidateRecord.session_id == session_id,
            ToolPatternLearningCandidateRecord.dedupe_signature.in_(signatures),  # type: ignore[attr-defined]
        )
    ).all()
    existing_by_signature = {item.dedupe_signature: item for item in existing_rows}
    patterns_by_id = {item.candidate_pattern_id: item for item in recommendation.candidate_tool_patterns}
    inserted_count = 0
    updated_count = 0
    persisted_candidate_ids: list[UUID] = []
    now = utc_now()

    for candidate in candidates:
        contract_seed_payload = _candidate_contract_payload(candidate, patterns_by_id=patterns_by_id)
        record = existing_by_signature.get(candidate.dedupe_signature)
        if record is None:
            record = ToolPatternLearningCandidateRecord(
                workspace_id=workspace_id,
                session_id=session_id,
                dedupe_signature=candidate.dedupe_signature,
                observation_count=1,
                first_seen_at=now,
                created_at=now,
            )
            inserted_count += 1
        else:
            record.observation_count += 1
            updated_count += 1

        _apply_candidate_to_record(
            record,
            candidate,
            source_artifact_id=source_artifact_id,
            source_blueprint_version=report.source_blueprint_version,
            contract_seed_payload=contract_seed_payload,
            now=now,
        )
        session.add(record)
        session.flush()
        persisted_candidate_ids.append(record.id)

    return ToolPatternLearningPersistenceSummary(
        inserted_count=inserted_count,
        updated_count=updated_count,
        skipped_count=max(skipped_count, 0),
        persisted_candidate_ids=persisted_candidate_ids,
        summary=(
            f"Persisted {inserted_count} new and updated {updated_count} tool pattern learning candidates; "
            f"skipped {max(skipped_count, 0)} without dedupe signature."
        ),
    )
