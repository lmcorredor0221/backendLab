from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    ACPPreview,
    ConstructionGapEntry,
    ConstructionQuestionEntry,
    ConstructionQuestionResponseRecord,
    ConstructionQuestionViewEntry,
    ConstructionReadinessReport,
    utc_now,
)
from app.services.acp_paths import slugify_acp_token


CURRENT_QUESTION_STATUS_ORDER = {
    "open": 0,
    "answered": 1,
    "resolved": 2,
}


def load_construction_question_response_records(
    session: Session,
    session_id: UUID,
) -> list[ConstructionQuestionResponseRecord]:
    return session.exec(
        select(ConstructionQuestionResponseRecord)
        .where(ConstructionQuestionResponseRecord.session_id == session_id)
        .order_by(ConstructionQuestionResponseRecord.updated_at.desc())
    ).all()


def index_construction_question_responses(
    records: list[ConstructionQuestionResponseRecord],
) -> dict[str, ConstructionQuestionResponseRecord]:
    indexed: dict[str, ConstructionQuestionResponseRecord] = {}
    for record in reversed(records):
        indexed[record.question_key] = record
    return indexed


def build_continuity_answer_map(
    records: list[ConstructionQuestionResponseRecord],
) -> dict[str, str]:
    return {
        record.question_key: record.answer_text.strip()
        for record in records
        if record.answer_text.strip()
    }


def build_construction_question_views(
    preview: ACPPreview,
    records: list[ConstructionQuestionResponseRecord],
) -> list[ConstructionQuestionViewEntry]:
    indexed = index_construction_question_responses(records)
    current_keys: set[str] = set()
    items: list[ConstructionQuestionViewEntry] = []

    for gap in preview.construction_readiness.gaps:
        for question in gap.questions:
            current_keys.add(question.question_key)
            items.append(_build_question_view(gap, question, indexed.get(question.question_key)))

    for record in records:
        if record.question_key in current_keys:
            continue
        items.append(_build_question_view_from_record(record, status_override="resolved"))

    return sorted(
        items,
        key=lambda item: (
            CURRENT_QUESTION_STATUS_ORDER.get(item.status, 99),
            item.domain,
            item.gap_key,
            item.question_key,
        ),
    )


def build_construction_gap_entries(
    preview: ACPPreview,
    records: list[ConstructionQuestionResponseRecord],
) -> list[ConstructionGapEntry]:
    indexed = index_construction_question_responses(records)
    gaps: list[ConstructionGapEntry] = []
    for gap in preview.construction_readiness.gaps:
        if not gap.questions:
            gaps.append(gap)
            continue

        question_statuses = [
            _current_question_status(question.question_key, indexed.get(question.question_key))
            for question in gap.questions
        ]
        gap_status = "answered" if question_statuses and all(status != "open" for status in question_statuses) else "open"
        gaps.append(gap.model_copy(update={"status": gap_status}))
    return gaps


def overlay_construction_readiness(
    preview: ACPPreview,
    records: list[ConstructionQuestionResponseRecord],
) -> ConstructionReadinessReport:
    base = preview.construction_readiness
    if not base.gaps:
        return base

    indexed = index_construction_question_responses(records)
    gaps = build_construction_gap_entries(preview, records)
    open_questions = sum(
        1
        for gap in preview.construction_readiness.gaps
        for question in gap.questions
        if _current_question_status(question.question_key, indexed.get(question.question_key)) == "open"
    )
    blocking_gaps = sum(1 for gap in gaps if gap.severity == "blocking" and gap.status not in {"answered", "resolved"})
    has_answered_gap = any(gap.status == "answered" for gap in gaps)

    overall_status = base.overall_status
    next_recommended_action = base.next_recommended_action
    can_start_build = blocking_gaps == 0 and open_questions == 0

    if not gaps:
        overall_status = "ready_to_build"
        next_recommended_action = "start_agentic_build"
    elif blocking_gaps > 0:
        overall_status = "blocked"
        if open_questions == 0 and has_answered_gap:
            next_recommended_action = "regenerate_acp_with_answers"
        elif open_questions > 0:
            next_recommended_action = "resolve_blocking_construction_gaps"
    else:
        overall_status = "needs_questions"
        if open_questions == 0 and has_answered_gap:
            next_recommended_action = "regenerate_acp_with_answers"
        else:
            next_recommended_action = "answer_open_questions"

    return ConstructionReadinessReport(
        overall_status=overall_status,
        can_start_build=can_start_build,
        blocking_gaps=blocking_gaps,
        open_questions=open_questions,
        assumptions_count=base.assumptions_count,
        gaps=gaps,
        next_recommended_action=next_recommended_action,
    )


def sync_construction_question_response_records(
    session: Session,
    preview: ACPPreview,
    records: list[ConstructionQuestionResponseRecord],
) -> bool:
    current_keys = {
        question.question_key
        for gap in preview.construction_readiness.gaps
        for question in gap.questions
    }
    now = utc_now()
    changed = False

    for record in records:
        if not record.answer_text.strip():
            continue
        if record.question_key in current_keys:
            if record.status == "resolved":
                record.status = "answered"
                record.resolved_at = None
                record.updated_at = now
                session.add(record)
                changed = True
            continue
        if record.status != "resolved":
            record.status = "resolved"
            record.resolved_at = now
            record.updated_at = now
            session.add(record)
            changed = True

    return changed


def is_no_applicable_answer(value: str) -> bool:
    normalized = _normalize_whitespace(value).lower()
    return normalized in {
        "n/a",
        "na",
        "no aplica",
        "no_aplica",
        "no aplica por ahora",
        "no requiere",
        "sin vector store",
        "sin embeddings",
    } or normalized.startswith("no aplica")


def parse_answer_pairs(value: str, aliases: dict[str, str] | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    alias_map = aliases or {}
    segments = re.split(r"[\n;]+", value)
    for raw_segment in segments:
        segment = raw_segment.strip().lstrip("-").strip()
        if not segment:
            continue
        separator = ":" if ":" in segment else "=" if "=" in segment else None
        if separator is None:
            continue
        key, parsed_value = segment.split(separator, 1)
        normalized_key = slugify_acp_token(key.replace("_", "-"), default="field").replace("-", "_")
        normalized_key = alias_map.get(normalized_key, normalized_key)
        normalized_value = _normalize_whitespace(parsed_value)
        if normalized_key and normalized_value:
            result[normalized_key] = normalized_value
    return result


def parse_answer_list(value: str) -> list[str]:
    if not value.strip():
        return []
    items: list[str] = []
    seen: set[str] = set()
    for raw_line in re.split(r"[\n;]+", value):
        normalized = _normalize_whitespace(raw_line.lstrip("-").strip())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        items.append(normalized)
    return items


def parse_contract_answer_entries(value: str) -> list[dict[str, str]]:
    aliases = {
        "tool": "tool",
        "tool_name": "tool",
        "tool_key": "tool",
        "system": "system",
        "system_name": "system",
        "endpoint": "endpoint",
        "action": "action",
        "auth": "auth",
        "authentication": "auth",
        "request": "request",
        "request_schema": "request",
        "response": "response",
        "response_schema": "response",
        "errors": "errors",
        "error_schema": "errors",
        "notes": "notes",
    }
    entries: list[dict[str, str]] = []
    for line in value.splitlines():
        pairs = parse_answer_pairs(line, aliases=aliases)
        if not pairs:
            continue
        entries.append(pairs)
    return entries


def _current_question_status(
    question_key: str,
    record: ConstructionQuestionResponseRecord | None,
) -> str:
    if record is None or not record.answer_text.strip():
        return "open"
    if record.status == "resolved":
        return "answered"
    return "answered"


def _build_question_view(
    gap: ConstructionGapEntry,
    question: ConstructionQuestionEntry,
    record: ConstructionQuestionResponseRecord | None,
) -> ConstructionQuestionViewEntry:
    if record is None:
        return ConstructionQuestionViewEntry(
            question_key=question.question_key,
            gap_key=gap.gap_key,
            gap_title=gap.title,
            domain=gap.domain,
            question_text=question.question_text,
            rationale=question.rationale,
            purpose=question.purpose,
            expected_answer_format=question.expected_answer_format,
            target_owner=question.target_owner,
            blocking=question.blocking,
            status="open",
            impacted_artifacts=gap.evidence_paths,
            options=question.options,
        )
    return ConstructionQuestionViewEntry(
        question_key=question.question_key,
        gap_key=gap.gap_key,
        gap_title=gap.title,
        domain=gap.domain,
        question_text=question.question_text,
        rationale=question.rationale,
        purpose=question.purpose,
        expected_answer_format=question.expected_answer_format,
        target_owner=question.target_owner,
        blocking=question.blocking,
        status=_current_question_status(question.question_key, record),
        answer_text=record.answer_text,
        owner_role=record.owner_role,
        answered_by_display=record.answered_by_display,
        answered_at=record.answered_at,
        resolved_at=record.resolved_at,
        impacted_artifacts=record.impacted_artifacts or gap.evidence_paths,
        options=question.options,
    )


def _build_question_view_from_record(
    record: ConstructionQuestionResponseRecord,
    *,
    status_override: str | None = None,
) -> ConstructionQuestionViewEntry:
    return ConstructionQuestionViewEntry(
        question_key=record.question_key,
        gap_key=record.gap_key,
        gap_title=record.gap_title,
        domain=record.domain,
        question_text=record.question_text,
        rationale=record.rationale,
        expected_answer_format=record.expected_answer_format,
        target_owner=record.target_owner,
        blocking=record.blocking,
        status=status_override or record.status,
        answer_text=record.answer_text,
        owner_role=record.owner_role,
        answered_by_display=record.answered_by_display,
        answered_at=record.answered_at,
        resolved_at=record.resolved_at,
        impacted_artifacts=record.impacted_artifacts,
    )


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
