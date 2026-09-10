from __future__ import annotations

import re
from typing import Any, Iterable
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    ACPPreview,
    ConstructionGapEntry,
    ConstructionQuestionEntry,
    ConstructionQuestionAnswerRequest,
    ConstructionQuestionImpactAnalysis,
    ConstructionQuestionOption,
    ConstructionQuestionResponseRecord,
    ConstructionQuestionViewEntry,
    ConstructionReadinessReport,
    UserRecord,
    utc_now,
)
from app.services.acp_paths import slugify_acp_token
from app.services.product_processing.persistence import UncertaintyBacklogRecord
from app.services.question_identity import merge_unique_strings, question_dedupe_signature


CURRENT_QUESTION_STATUS_ORDER = {
    "open": 0,
    "answered": 1,
    "deferred": 2,
    "resolved": 3,
    "dismissed": 4,
}

QUESTION_RESOLUTION_PRIORITY = {
    "open": 0,
    "deferred": 1,
    "dismissed": 2,
    "answered": 3,
    "resolved": 3,
}

BACKLOG_RESOLUTION_PRIORITY = {
    "open": 0,
    "in_progress": 0,
    "deferred": 1,
    "dismissed": 2,
    "superseded": 2,
    "resolved": 3,
}

DOMAIN_PHASE_HINTS: dict[str, tuple[str, ...]] = {
    "deployment": ("acp_questions_resolution", "acp_artifact_reconciliation", "acp_package_build", "acp_download_ready"),
    "runtime": ("acp_questions_resolution", "acp_artifact_reconciliation", "acp_package_build", "acp_download_ready"),
    "knowledge": ("acp_questions_resolution", "acp_artifact_reconciliation"),
    "memory": ("acp_questions_resolution", "acp_artifact_reconciliation"),
    "contracts": ("acp_questions_resolution", "acp_artifact_reconciliation", "acp_package_build", "acp_download_ready"),
    "integrations": ("acp_questions_resolution", "acp_artifact_reconciliation"),
    "integration": ("acp_questions_resolution", "acp_artifact_reconciliation"),
    "security": ("acp_input_readiness", "acp_quality_gates", "acp_artifact_reconciliation", "acp_download_ready"),
    "observability": ("acp_test_suite", "acp_graphic_simulation", "acp_quality_gates", "acp_download_ready"),
}

ARTIFACT_PREFIX_PHASE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ACP/deployment/", ("acp_artifact_reconciliation", "acp_package_build", "acp_download_ready")),
    ("ACP/runtime/", ("acp_artifact_reconciliation", "acp_package_build", "acp_download_ready")),
    ("ACP/observability/", ("acp_graphic_simulation", "acp_quality_gates", "acp_download_ready")),
    ("ACP/evaluation/", ("acp_test_suite", "acp_quality_gates")),
    ("ACP/knowledge/", ("acp_questions_resolution", "acp_artifact_reconciliation")),
    ("ACP/memory/", ("acp_questions_resolution", "acp_artifact_reconciliation")),
    ("ACP/tools/", ("acp_questions_resolution", "acp_artifact_reconciliation")),
    ("ACP/prompts/", ("acp_questions_resolution", "acp_artifact_reconciliation")),
    ("ACP/conformance/", ("acp_download_ready",)),
    ("ACP/manifest", ("acp_artifact_reconciliation", "acp_package_build", "acp_download_ready")),
)

PHASE_TO_STAGE_HINTS: dict[str, tuple[str, ...]] = {
    "acp_input_readiness": ("acp", "validate"),
    "acp_questions_resolution": ("acp",),
    "acp_test_suite": ("acp", "validate"),
    "acp_graphic_simulation": ("acp", "validate"),
    "acp_quality_gates": ("acp", "validate"),
    "acp_artifact_reconciliation": ("acp", "package"),
    "acp_package_build": ("acp", "package"),
    "acp_download_ready": ("acp", "package"),
}

UNCERTAINTY_BACKLOG_QUESTION_PREFIX = "uncertainty_backlog:"
UNCERTAINTY_BACKLOG_CLOSED_STATUSES = {"dismissed", "superseded"}
UNCERTAINTY_BACKLOG_IMPLEMENTATION_TARGETS = {
    "acp",
    "acp_questions_resolution",
    "acp_artifact_reconciliation",
    "acp_package_build",
    "acp_download_ready",
    "package",
    "implementation",
    "implementation_questions",
    "construction",
}


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    return merge_unique_strings(values)


def _strongest_question_status(statuses: Iterable[str]) -> str:
    strongest = "open"
    for status in statuses:
        normalized = str(status or "open").strip().lower()
        if QUESTION_RESOLUTION_PRIORITY.get(normalized, 0) > QUESTION_RESOLUTION_PRIORITY.get(strongest, 0):
            strongest = normalized
    return strongest


def _record_question_signature(record: ConstructionQuestionResponseRecord) -> str:
    return question_dedupe_signature(record.question_text, fallback_key=record.question_key)


def _question_entry_signature(gap: ConstructionGapEntry, question: ConstructionQuestionEntry) -> str:
    return question_dedupe_signature(question.question_text, fallback_key=f"{gap.gap_key}:{question.question_key}")


def _backlog_question_signature(record: UncertaintyBacklogRecord) -> str:
    return question_dedupe_signature(
        record.description or record.title,
        fallback_key=record.uncertainty_key,
    )


def _backlog_record_rank(record: UncertaintyBacklogRecord) -> tuple[int, int, int, float, object]:
    status = str(record.status or "open").strip().lower()
    answer_text = record.assumed_answer or record.suggested_answer
    impacted_count = len(record.affected_deliverable_keys or []) + len(record.dependency_keys or [])
    return (
        BACKLOG_RESOLUTION_PRIORITY.get(status, 0),
        1 if str(answer_text or "").strip() else 0,
        impacted_count,
        float(record.confidence or 0),
        record.updated_at or record.created_at,
    )


def _merge_backlog_answer_options(records: list[UncertaintyBacklogRecord]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for record in records:
        for option in record.answer_options or []:
            if not isinstance(option, dict):
                continue
            key = question_dedupe_signature(option.get("label"), fallback_key=option.get("key"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(option))
    return merged


def _merge_backlog_record_group(records: list[UncertaintyBacklogRecord]) -> UncertaintyBacklogRecord:
    if len(records) == 1:
        return records[0]
    winner = max(records, key=_backlog_record_rank)
    signature = _backlog_question_signature(winner)
    payload = dict(winner.payload or {})
    payload["dedupe_signature"] = signature
    payload["merged_uncertainty_ids"] = [
        str(record.id)
        for record in records
        if record.id != winner.id
    ]
    return winner.model_copy(
        update={
            "source_refs": merge_unique_strings(
                source_ref
                for record in records
                for source_ref in record.source_refs or []
            ),
            "affected_deliverable_keys": merge_unique_strings(
                deliverable_key
                for record in records
                for deliverable_key in record.affected_deliverable_keys or []
            ),
            "dependency_keys": merge_unique_strings(
                dependency_key
                for record in records
                for dependency_key in record.dependency_keys or []
            ),
            "answer_options": _merge_backlog_answer_options(records),
            "payload": payload,
        }
    )


def dedupe_uncertainty_backlog_records(
    records: list[UncertaintyBacklogRecord],
) -> list[UncertaintyBacklogRecord]:
    grouped: dict[str, list[UncertaintyBacklogRecord]] = {}
    for record in records:
        grouped.setdefault(_backlog_question_signature(record), []).append(record)
    return [_merge_backlog_record_group(group) for group in grouped.values()]


def _merge_question_options(items: list[ConstructionQuestionViewEntry]) -> list[ConstructionQuestionOption]:
    seen: set[str] = set()
    merged: list[ConstructionQuestionOption] = []
    for item in items:
        for option in item.options or []:
            key = question_dedupe_signature(option.label, fallback_key=option.key)
            if key in seen:
                continue
            seen.add(key)
            merged.append(option)
    return merged


def _question_view_signature(item: ConstructionQuestionViewEntry) -> str:
    return question_dedupe_signature(item.question_text, fallback_key=item.question_key)


def _question_view_rank(item: ConstructionQuestionViewEntry) -> tuple[int, int, int, int]:
    return (
        QUESTION_RESOLUTION_PRIORITY.get(item.status, 0),
        1 if item.answer_text.strip() else 0,
        len(item.impacted_artifacts or []),
        len(item.options or []),
    )


def _first_non_empty(values: Iterable[str]) -> str:
    for value in values:
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return ""


def _first_non_none(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _merge_question_view_group(items: list[ConstructionQuestionViewEntry]) -> ConstructionQuestionViewEntry:
    if len(items) == 1:
        return items[0]
    ranked = sorted(items, key=_question_view_rank, reverse=True)
    primary = ranked[0]
    status = _strongest_question_status(item.status for item in items)
    return primary.model_copy(
        update={
            "status": status,
            "blocking": any(item.blocking for item in items) if status == "open" else False,
            "answer_text": primary.answer_text.strip() or _first_non_empty(item.answer_text for item in ranked),
            "owner_role": primary.owner_role.strip() or _first_non_empty(item.owner_role for item in ranked),
            "answered_by_display": primary.answered_by_display.strip()
            or _first_non_empty(item.answered_by_display for item in ranked),
            "answered_at": primary.answered_at or _first_non_none(item.answered_at for item in ranked),
            "resolved_at": primary.resolved_at or _first_non_none(item.resolved_at for item in ranked),
            "impacted_artifacts": merge_unique_strings(
                artifact
                for item in items
                for artifact in item.impacted_artifacts or []
            ),
            "options": _merge_question_options(items),
            "impact_analysis": primary.impact_analysis or _first_non_none(item.impact_analysis for item in ranked),
        }
    )


def dedupe_construction_question_views(
    items: list[ConstructionQuestionViewEntry],
) -> list[ConstructionQuestionViewEntry]:
    grouped: dict[str, list[ConstructionQuestionViewEntry]] = {}
    for item in items:
        grouped.setdefault(_question_view_signature(item), []).append(item)
    return [_merge_question_view_group(group) for group in grouped.values()]


def _question_status_by_signature(
    preview: ACPPreview,
    records: list[ConstructionQuestionResponseRecord],
) -> dict[str, str]:
    indexed = index_construction_question_responses(records)
    statuses: dict[str, str] = {}
    for record in records:
        signature = _record_question_signature(record)
        status = _current_question_status(record.question_key, record)
        statuses[signature] = _strongest_question_status([statuses.get(signature, "open"), status])
    for gap in preview.construction_readiness.gaps:
        for question in gap.questions:
            signature = _question_entry_signature(gap, question)
            status = _current_question_status(question.question_key, indexed.get(question.question_key))
            if status == "open" and signature in statuses:
                status = statuses[signature]
            statuses[signature] = _strongest_question_status([statuses.get(signature, "open"), status])
    return statuses


def load_construction_question_response_records(
    session: Session,
    session_id: UUID,
) -> list[ConstructionQuestionResponseRecord]:
    return session.exec(
        select(ConstructionQuestionResponseRecord)
        .where(ConstructionQuestionResponseRecord.session_id == session_id)
        .order_by(ConstructionQuestionResponseRecord.updated_at.desc())
    ).all()


def load_uncertainty_backlog_records(
    session: Session,
    session_id: UUID,
) -> list[UncertaintyBacklogRecord]:
    rows = session.exec(
        select(UncertaintyBacklogRecord)
        .where(UncertaintyBacklogRecord.session_id == session_id)
        .order_by(UncertaintyBacklogRecord.updated_at.desc())
    ).all()
    return [
        row
        for row in rows
        if str(row.status or "").strip().lower() not in UNCERTAINTY_BACKLOG_CLOSED_STATUSES
    ]


def uncertainty_backlog_question_key(record: UncertaintyBacklogRecord) -> str:
    return f"{UNCERTAINTY_BACKLOG_QUESTION_PREFIX}{record.id}"


def uncertainty_backlog_id_from_question_key(question_key: str) -> UUID | None:
    normalized = str(question_key or "").strip()
    if not normalized.startswith(UNCERTAINTY_BACKLOG_QUESTION_PREFIX):
        return None
    try:
        return UUID(normalized.removeprefix(UNCERTAINTY_BACKLOG_QUESTION_PREFIX))
    except ValueError:
        return None


def _backlog_domain(record: UncertaintyBacklogRecord) -> str:
    text = " ".join(
        [
            record.source_stage,
            record.target_stage,
            record.kind,
            record.title,
            record.description,
            record.reason,
            record.impact,
            " ".join(record.affected_deliverable_keys or []),
            " ".join(record.dependency_keys or []),
        ]
    ).lower()
    if any(term in text for term in ("deployment", "despliegue", "hosting", "docker", "kubernetes")):
        return "deployment"
    if any(term in text for term in ("runtime", "llm", "modelo", "vector", "secret", "credencial")):
        return "runtime"
    if any(term in text for term in ("knowledge", "memoria", "memory", "rag", "retrieval", "document")):
        return "knowledge"
    if any(term in text for term in ("api", "integration", "integracion", "tool", "herramienta", "webhook")):
        return "integrations"
    if any(term in text for term in ("security", "seguridad", "approval", "aprobacion", "pii", "privacidad")):
        return "security"
    return "implementation"


def _backlog_gap_status(record: UncertaintyBacklogRecord) -> str:
    status = str(record.status or "").strip().lower()
    disposition = str(record.disposition or "").strip().lower()
    if status == "resolved" or disposition in {"defer", "infer"} or status == "deferred":
        return "answered"
    return "open"


def _backlog_is_blocking(record: UncertaintyBacklogRecord) -> bool:
    return str(record.disposition or "").strip().lower() == "block"


def _backlog_owner(record: UncertaintyBacklogRecord) -> str:
    payload = record.payload if isinstance(record.payload, dict) else {}
    owner = str(payload.get("target_owner") or payload.get("owner") or "").strip()
    if owner:
        return owner
    if _backlog_domain(record) in {"deployment", "runtime", "integrations"}:
        return "implementation_owner"
    return "product_owner"


def _backlog_answer_text(record: UncertaintyBacklogRecord) -> str:
    status = str(record.status or "").strip().lower()
    disposition = str(record.disposition or "").strip().lower()
    if status == "resolved":
        return record.assumed_answer or record.suggested_answer or "Decision resuelta desde backlog LAB."
    if status == "deferred" or disposition == "defer":
        return record.assumed_answer or (
            "Delegado a implementacion. Resolver durante la construccion con trazabilidad ACP."
        )
    if disposition == "infer":
        return record.assumed_answer or record.suggested_answer or "Supuesto inferido por LAB para mantener continuidad."
    return ""


def _backlog_response_status(record: UncertaintyBacklogRecord) -> str:
    status = str(record.status or "").strip().lower()
    disposition = str(record.disposition or "").strip().lower()
    if status == "resolved" or disposition == "infer":
        return "answered"
    if status == "deferred" or disposition == "defer":
        return "deferred"
    return "open"


def build_construction_gaps_from_uncertainty_backlog(
    records: list[UncertaintyBacklogRecord],
) -> list[ConstructionGapEntry]:
    gaps: list[ConstructionGapEntry] = []
    for record in dedupe_uncertainty_backlog_records(records):
        status = _backlog_gap_status(record)
        blocking = _backlog_is_blocking(record)
        gap_key = f"uncertainty_backlog:{record.product_mode}:{slugify_acp_token(record.uncertainty_key, default='item')}"
        impacted = _dedupe_strings([*(record.affected_deliverable_keys or []), *(record.dependency_keys or [])])
        question = ConstructionQuestionEntry(
            question_key=uncertainty_backlog_question_key(record),
            question_text=record.description or record.title or record.uncertainty_key,
            rationale=record.reason or record.impact or "Decision heredada del backlog de incertidumbres LAB.",
            purpose="Resolver, confirmar o delegar una decision trasladada desde Blueprint al ACP.",
            expected_answer_format="decision=<respuesta>; owner=<responsable>; notes=<detalle>",
            target_owner=_backlog_owner(record),
            blocking=blocking,
        )
        gaps.append(
            ConstructionGapEntry(
                gap_key=gap_key,
                title=record.title or record.uncertainty_key,
                domain=_backlog_domain(record),
                severity="blocking" if blocking else "warning",
                status=status,
                blocking_stage=record.target_stage or "acp_questions_resolution",
                summary=record.reason or record.description or record.title or "Decision heredada del backlog LAB.",
                remediation=record.suggested_answer or "Resolver, confirmar o delegar con trazabilidad antes de implementar.",
                evidence_paths=impacted,
                source_sections=_dedupe_strings(
                    [
                        f"uncertainty_backlog.{record.product_mode}",
                        f"journey.{record.source_stage}" if record.source_stage else "",
                    ]
                ),
                current_assumptions=_dedupe_strings([record.assumed_answer]),
                closure_criteria=[
                    "Registrar respuesta, owner o delegacion explicita.",
                    "Conservar trazabilidad hacia el backlog LAB original.",
                ],
                questions=[question] if status == "open" else [],
            )
        )
    return gaps


def build_construction_question_response_records_from_uncertainty_backlog(
    records: list[UncertaintyBacklogRecord],
    *,
    existing_records: list[ConstructionQuestionResponseRecord] | None = None,
) -> list[ConstructionQuestionResponseRecord]:
    existing_keys = {record.question_key for record in existing_records or []}
    existing_signatures = {_record_question_signature(record) for record in existing_records or []}
    synthetic: list[ConstructionQuestionResponseRecord] = []
    for record in dedupe_uncertainty_backlog_records(records):
        question_key = uncertainty_backlog_question_key(record)
        signature = _backlog_question_signature(record)
        if question_key in existing_keys or signature in existing_signatures:
            continue
        status = _backlog_response_status(record)
        if status == "open":
            continue
        synthetic.append(
            ConstructionQuestionResponseRecord(
                session_id=record.session_id,
                question_key=question_key,
                gap_key=f"uncertainty_backlog:{record.product_mode}:{slugify_acp_token(record.uncertainty_key, default='item')}",
                gap_title=record.title or record.uncertainty_key,
                domain=_backlog_domain(record),
                question_text=record.description or record.title or record.uncertainty_key,
                rationale=record.reason or record.impact or "Decision heredada del backlog de incertidumbres LAB.",
                expected_answer_format="decision=<respuesta>; owner=<responsable>; notes=<detalle>",
                target_owner=_backlog_owner(record),
                blocking=_backlog_is_blocking(record),
                status=status,
                answer_text=_backlog_answer_text(record),
                owner_role=_backlog_owner(record),
                impacted_artifacts=_dedupe_strings(
                    [*(record.affected_deliverable_keys or []), *(record.dependency_keys or [])]
                ),
                answered_by_display="Lean Agent Builder backlog",
                answered_at=record.resolved_at or record.updated_at,
                resolved_at=record.resolved_at if status == "answered" else None,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
        )
    return synthetic


def merge_construction_question_records_with_uncertainty_backlog(
    records: list[ConstructionQuestionResponseRecord],
    backlog_records: list[UncertaintyBacklogRecord],
) -> list[ConstructionQuestionResponseRecord]:
    return [
        *records,
        *build_construction_question_response_records_from_uncertainty_backlog(
            backlog_records,
            existing_records=records,
        ),
    ]


def load_construction_question_response_records_for_preview(
    session: Session,
    session_id: UUID,
) -> list[ConstructionQuestionResponseRecord]:
    records = load_construction_question_response_records(session, session_id)
    backlog_records = load_uncertainty_backlog_records(session, session_id)
    return merge_construction_question_records_with_uncertainty_backlog(records, backlog_records)


def append_construction_readiness_gaps(
    preview: ACPPreview,
    extra_gaps: list[ConstructionGapEntry] | None,
) -> ACPPreview:
    if not extra_gaps:
        return preview
    existing_keys = {gap.gap_key for gap in preview.construction_readiness.gaps}
    gaps = [
        *preview.construction_readiness.gaps,
        *[gap for gap in extra_gaps if gap.gap_key not in existing_keys],
    ]
    blocking_gaps = sum(1 for gap in gaps if gap.severity == "blocking" and gap.status not in {"answered", "resolved"})
    open_questions = sum(len(gap.questions) for gap in gaps if gap.status == "open")
    assumptions = _dedupe_strings(
        assumption
        for gap in gaps
        for assumption in gap.current_assumptions
    )
    can_start_build = preview.validation.can_export_zip and blocking_gaps == 0 and open_questions == 0
    if can_start_build:
        overall_status = "ready_to_build"
        next_action = "start_agentic_build"
    elif blocking_gaps > 0:
        overall_status = "blocked"
        next_action = "resolve_blocking_construction_gaps"
    else:
        overall_status = "needs_questions"
        next_action = "answer_open_questions"
    readiness = preview.construction_readiness.model_copy(
        update={
            "overall_status": overall_status,
            "can_start_build": can_start_build,
            "blocking_gaps": blocking_gaps,
            "open_questions": open_questions,
            "assumptions_count": len(assumptions),
            "gaps": gaps,
            "next_recommended_action": next_action,
        }
    )
    return preview.model_copy(update={"construction_readiness": readiness})


def apply_uncertainty_backlog_acp_answer(
    session: Session,
    *,
    session_id: UUID,
    backlog_id: UUID,
    payload: ConstructionQuestionAnswerRequest,
    current_user: UserRecord,
) -> UncertaintyBacklogRecord:
    record = session.get(UncertaintyBacklogRecord, backlog_id)
    if record is None or record.session_id != session_id:
        raise LookupError("Uncertainty backlog item not found")
    now = utc_now()
    current_payload = dict(record.payload or {})
    current_payload["acp_resolution"] = {
        "decision": payload.decision,
        "owner_role": payload.owner_role.strip() or _backlog_owner(record),
        "answered_by_user_id": str(current_user.id),
        "answered_by_display": current_user.full_name or current_user.email,
        "answered_at": now.isoformat(),
    }
    if payload.decision == "delegate":
        record.disposition = "defer"
        record.target_stage = "implementation"
        record.status = "deferred"
        record.assumed_answer = payload.answer_text.strip() or (
            "Delegado a implementacion. Resolver durante la construccion con trazabilidad ACP."
        )
    elif payload.decision == "dismiss":
        record.disposition = "dismiss"
        record.target_stage = "closed"
        record.status = "dismissed"
        record.assumed_answer = payload.answer_text.strip() or (
            "Descartado por el usuario. No aplicable para este alcance."
        )
        record.resolved_at = now
    else:
        record.status = "resolved"
        record.assumed_answer = payload.answer_text.strip() or payload.selected_option_key.strip()
        record.resolved_at = now
    if payload.impacted_artifacts:
        record.affected_deliverable_keys = _dedupe_strings(payload.impacted_artifacts)
    record.payload = current_payload
    record.updated_at = now
    session.add(record)
    session.flush()
    return record


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
        if record.answer_text.strip() and record.status != "deferred"
    }


def build_construction_decision_log(
    preview: ACPPreview,
    records: list[ConstructionQuestionResponseRecord],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in build_construction_question_views(preview, records):
        if item.status == "open":
            continue
        analysis = item.impact_analysis
        entries.append(
            {
                "question_key": item.question_key,
                "gap_key": item.gap_key,
                "gap_title": item.gap_title,
                "domain": item.domain,
                "status": item.status,
                "blocking": item.blocking,
                "question_text": item.question_text,
                "rationale": item.rationale,
                "expected_answer_format": item.expected_answer_format,
                "target_owner": item.target_owner,
                "owner_role": item.owner_role,
                "answer_text": item.answer_text,
                "answered_by_display": item.answered_by_display,
                "answered_at": item.answered_at.isoformat() if item.answered_at else None,
                "resolved_at": item.resolved_at.isoformat() if item.resolved_at else None,
                "impacted_artifacts": list(item.impacted_artifacts or []),
                "impact_analysis": analysis.model_dump(mode="json") if analysis else None,
            }
        )
    return entries


def build_deferred_construction_decision_backlog(
    preview: ACPPreview,
    records: list[ConstructionQuestionResponseRecord],
) -> list[dict[str, Any]]:
    return [
        item
        for item in build_construction_decision_log(preview, records)
        if item["status"] == "deferred"
    ]


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
        status_override = None if record.status in {"deferred", "open"} else "resolved"
        items.append(_build_question_view_from_record(record, status_override=status_override))

    return sorted(
        dedupe_construction_question_views(items),
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
    status_by_signature = _question_status_by_signature(preview, records)
    gaps: list[ConstructionGapEntry] = []
    for gap in preview.construction_readiness.gaps:
        if not gap.questions:
            gaps.append(gap)
            continue

        question_statuses = [
            status_by_signature.get(
                _question_entry_signature(gap, question),
                _current_question_status(question.question_key, indexed.get(question.question_key)),
            )
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

    gaps = build_construction_gap_entries(preview, records)
    question_views = build_construction_question_views(preview, records)
    open_questions = sum(1 for item in question_views if item.status == "open")
    blocking_gaps = sum(1 for gap in gaps if gap.severity == "blocking" and gap.status not in {"answered", "resolved"})
    has_answered_gap = any(gap.status == "answered" for gap in gaps)
    validation_allows_build = bool(preview.validation.can_export_zip)

    overall_status = base.overall_status
    next_recommended_action = base.next_recommended_action
    can_start_build = validation_allows_build and blocking_gaps == 0 and open_questions == 0

    if not validation_allows_build:
        overall_status = "blocked"
        next_recommended_action = "resolve_blocking_construction_gaps"
    elif not gaps:
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
        if record.question_key.startswith(UNCERTAINTY_BACKLOG_QUESTION_PREFIX):
            continue
        if not record.answer_text.strip():
            if record.status == "deferred":
                continue
            continue
        if record.question_key in current_keys:
            if record.status == "resolved":
                record.status = "answered"
                record.resolved_at = None
                record.updated_at = now
                session.add(record)
                changed = True
            continue
        if record.status not in {"resolved", "deferred"}:
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
    if record is None:
        return "open"
    if record.status == "deferred":
        return "deferred"
    if record.status == "dismissed":
        return "dismissed"
    if not record.answer_text.strip():
        return "open"
    if record.status == "resolved":
        return "answered"
    return "answered"


def _build_question_view(
    gap: ConstructionGapEntry,
    question: ConstructionQuestionEntry,
    record: ConstructionQuestionResponseRecord | None,
) -> ConstructionQuestionViewEntry:
    impact_analysis = _build_question_impact_analysis(gap, question, record)
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
            impact_analysis=impact_analysis,
        )
    resolved_status = _current_question_status(question.question_key, record)
    is_actively_blocking = question.blocking if resolved_status == "open" else False
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
        blocking=is_actively_blocking,
        status=resolved_status,
        answer_text=record.answer_text,
        owner_role=record.owner_role,
        answered_by_display=record.answered_by_display,
        answered_at=record.answered_at,
        resolved_at=record.resolved_at,
        impacted_artifacts=record.impacted_artifacts or gap.evidence_paths,
        options=question.options,
        impact_analysis=impact_analysis,
    )


def _build_question_view_from_record(
    record: ConstructionQuestionResponseRecord,
    *,
    status_override: str | None = None,
) -> ConstructionQuestionViewEntry:
    impact_analysis = _build_record_impact_analysis(record, impacted_artifacts=record.impacted_artifacts)
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
        impact_analysis=impact_analysis,
    )


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _build_question_impact_analysis(
    gap: ConstructionGapEntry,
    question: ConstructionQuestionEntry,
    record: ConstructionQuestionResponseRecord | None,
) -> ConstructionQuestionImpactAnalysis | None:
    if record is None:
        return None
    impacted_artifacts = _dedupe_strings(record.impacted_artifacts or gap.evidence_paths)
    return _build_record_impact_analysis(
        record,
        impacted_artifacts=impacted_artifacts,
        blocking=question.blocking,
        domain=gap.domain,
    )


def _build_record_impact_analysis(
    record: ConstructionQuestionResponseRecord,
    *,
    impacted_artifacts: list[str],
    blocking: bool | None = None,
    domain: str | None = None,
) -> ConstructionQuestionImpactAnalysis | None:
    normalized_answer = record.answer_text.strip()
    if not normalized_answer and record.status != "deferred":
        return None

    resolved_blocking = record.blocking if blocking is None else blocking
    resolved_domain = (domain or record.domain or "").strip().lower()
    affected_phase_keys = _resolve_affected_phase_keys(
        resolved_domain,
        impacted_artifacts,
        blocking=resolved_blocking,
    )
    affected_stage_keys = _resolve_affected_stage_keys(affected_phase_keys)

    if record.status == "deferred":
        return ConstructionQuestionImpactAnalysis(
            impact_kind="delegated_to_implementation",
            material_impact=False,
            reprocess_decision="delegated_to_implementation",
            impact_summary=(
                "La decision se difiere a implementacion con trazabilidad ACP y no altera automaticamente los entregables actuales."
            ),
            recommended_action=(
                "Conserva esta decision en el paquete ACP y solicitala solo cuando la implementacion necesite cerrar el punto."
            ),
            affected_phase_keys=affected_phase_keys,
            affected_stage_keys=affected_stage_keys,
        )

    if not impacted_artifacts and not resolved_blocking and len(affected_phase_keys) <= 1:
        return ConstructionQuestionImpactAnalysis(
            impact_kind="no_material_impact",
            material_impact=False,
            reprocess_decision="document_only",
            impact_summary=(
                "La respuesta se conserva como contexto adicional y no justifica reconciliar entregables ACP por si sola."
            ),
            recommended_action="Documenta la aclaracion y continua con el ACP sin reconciliar entregables.",
            affected_phase_keys=affected_phase_keys,
            affected_stage_keys=affected_stage_keys,
        )

    if (
        len(affected_phase_keys) >= 4
        or len(impacted_artifacts) >= 4
        or (resolved_blocking and len(impacted_artifacts) >= 3)
    ):
        return ConstructionQuestionImpactAnalysis(
            impact_kind="structural_impact",
            material_impact=True,
            reprocess_decision="structural_reconciliation",
            impact_summary=(
                f"La respuesta cambia una dependencia estructural y afecta {len(impacted_artifacts)} artefacto(s) y {len(affected_phase_keys)} fase(s) ACP."
            ),
            recommended_action=(
                "Antes de exportar, revisa el plan y reconcilia los entregables ACP afectados sin reabrir fases anteriores."
            ),
            affected_phase_keys=affected_phase_keys,
            affected_stage_keys=affected_stage_keys,
        )

    return ConstructionQuestionImpactAnalysis(
        impact_kind="localized_impact",
        material_impact=True,
        reprocess_decision="localized_reconciliation",
        impact_summary=(
            f"La respuesta impacta de forma localizada {len(impacted_artifacts)} artefacto(s) y puede reconciliarse solo en los entregables ACP afectados."
        ),
        recommended_action=(
            "Mantiene la respuesta acumulada y reconcilia solo los entregables ACP impactados cuando vayas a actualizarlos."
        ),
        affected_phase_keys=affected_phase_keys,
        affected_stage_keys=affected_stage_keys,
    )


def _resolve_affected_phase_keys(
    domain: str,
    impacted_artifacts: list[str],
    *,
    blocking: bool,
) -> list[str]:
    phase_keys: list[str] = []
    phase_keys.extend(DOMAIN_PHASE_HINTS.get(domain, ("acp_questions_resolution",)))
    for artifact in impacted_artifacts:
        normalized = artifact.strip()
        if not normalized:
            continue
        for prefix, hinted_phases in ARTIFACT_PREFIX_PHASE_HINTS:
            if normalized.startswith(prefix):
                phase_keys.extend(hinted_phases)
    if blocking:
        phase_keys.extend(("acp_questions_resolution", "acp_artifact_reconciliation"))
    return _dedupe_strings(phase_keys)


def _resolve_affected_stage_keys(phase_keys: list[str]) -> list[str]:
    stage_keys: list[str] = []
    for phase_key in phase_keys:
        stage_keys.extend(PHASE_TO_STAGE_HINTS.get(phase_key, ("acp",)))
    return _dedupe_strings(stage_keys)
