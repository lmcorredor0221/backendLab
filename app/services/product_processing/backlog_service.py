from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import utc_now
from app.services.product_processing.contracts import (
    ProductProcessingMode,
    UncertaintyBacklogEntry,
    UncertaintyBacklogStatus,
    UncertaintyClassification,
    UncertaintyDisposition,
    UncertaintyKind,
    UncertaintyOption,
)
from app.services.product_processing.persistence import UncertaintyBacklogRecord


def _status_for_classification(classification: UncertaintyClassification) -> str:
    if classification.disposition in {UncertaintyDisposition.block, UncertaintyDisposition.resolve_now}:
        return UncertaintyBacklogStatus.open.value
    return UncertaintyBacklogStatus.deferred.value


def _option_payload(options: list[UncertaintyOption]) -> list[dict[str, object]]:
    return [option.model_dump(mode="json") for option in options]


def backlog_entry_from_record(record: UncertaintyBacklogRecord) -> UncertaintyBacklogEntry:
    return UncertaintyBacklogEntry(
        id=str(record.id),
        workspace_id=str(record.workspace_id),
        session_id=str(record.session_id),
        uncertainty_key=record.uncertainty_key,
        product_mode=ProductProcessingMode(record.product_mode),
        source_stage=record.source_stage,
        target_stage=record.target_stage,
        kind=UncertaintyKind(record.kind),
        disposition=UncertaintyDisposition(record.disposition),
        status=UncertaintyBacklogStatus(record.status),
        title=record.title,
        reason=record.reason,
        impact=record.impact,
        confidence=record.confidence,
        cost_to_resolve_units=record.cost_to_resolve_units,
        assumed_answer=record.assumed_answer,
        suggested_answer=record.suggested_answer,
        answer_options=[UncertaintyOption.model_validate(option) for option in record.answer_options],
        source_refs=record.source_refs,
        affected_deliverable_keys=record.affected_deliverable_keys,
        dependency_keys=record.dependency_keys,
        created_from=record.created_from,
    )


def upsert_uncertainty_backlog(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    classification: UncertaintyClassification,
    dependency_keys: list[str] | None = None,
    created_from: str = "runtime",
) -> UncertaintyBacklogEntry:
    uncertainty = classification.uncertainty
    existing = db.exec(
        select(UncertaintyBacklogRecord).where(
            UncertaintyBacklogRecord.session_id == session_id,
            UncertaintyBacklogRecord.uncertainty_key == uncertainty.key,
            UncertaintyBacklogRecord.product_mode == classification.profile_mode.value,
        )
    ).first()
    record = existing or UncertaintyBacklogRecord(
        workspace_id=workspace_id,
        session_id=session_id,
        uncertainty_key=uncertainty.key,
        product_mode=classification.profile_mode.value,
    )
    if existing and existing.status in {
        UncertaintyBacklogStatus.resolved.value,
        UncertaintyBacklogStatus.deferred.value,
        UncertaintyBacklogStatus.dismissed.value,
        UncertaintyBacklogStatus.superseded.value,
    }:
        record.updated_at = utc_now()
        db.add(record)
        db.flush()
        return backlog_entry_from_record(record)

    record.source_stage = uncertainty.stage
    record.target_stage = classification.target_stage or uncertainty.deferral_target_stage
    record.kind = uncertainty.kind.value
    record.disposition = classification.disposition.value
    record.status = _status_for_classification(classification)
    record.title = uncertainty.title
    record.description = uncertainty.description
    record.reason = classification.reason or uncertainty.reason
    record.impact = uncertainty.impact
    record.confidence = uncertainty.confidence
    record.assumed_answer = uncertainty.assumed_answer
    record.suggested_answer = uncertainty.suggested_answer
    record.answer_options = _option_payload(uncertainty.answer_options)
    record.source_refs = list(uncertainty.source_refs)
    record.affected_deliverable_keys = list(uncertainty.affected_deliverable_keys)
    record.dependency_keys = list(dependency_keys or [])
    record.payload = classification.model_dump(mode="json")
    record.created_from = created_from
    record.updated_at = utc_now()
    record.resolved_at = None
    record.superseded_at = None
    db.add(record)
    db.flush()
    return backlog_entry_from_record(record)


def list_uncertainty_backlog(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    product_mode: ProductProcessingMode | str | None = None,
    include_closed: bool = False,
) -> list[UncertaintyBacklogEntry]:
    statement = select(UncertaintyBacklogRecord).where(
        UncertaintyBacklogRecord.workspace_id == workspace_id,
        UncertaintyBacklogRecord.session_id == session_id,
    )
    if product_mode is not None:
        normalized_mode = product_mode if isinstance(product_mode, ProductProcessingMode) else ProductProcessingMode(str(product_mode))
        statement = statement.where(UncertaintyBacklogRecord.product_mode == normalized_mode.value)
    if not include_closed:
        statement = statement.where(
            UncertaintyBacklogRecord.status.notin_(
                [
                    UncertaintyBacklogStatus.resolved.value,
                    UncertaintyBacklogStatus.dismissed.value,
                    UncertaintyBacklogStatus.superseded.value,
                ]
            )
        )
    rows = db.exec(
        statement.order_by(
            UncertaintyBacklogRecord.source_stage.asc(),
            UncertaintyBacklogRecord.updated_at.desc(),
        )
    ).all()
    return [backlog_entry_from_record(row) for row in rows]


def prioritize_uncertainty_backlog(entries: list[UncertaintyBacklogEntry]) -> list[UncertaintyBacklogEntry]:
    disposition_weight = {
        UncertaintyDisposition.block: 0,
        UncertaintyDisposition.resolve_now: 1,
        UncertaintyDisposition.defer: 2,
        UncertaintyDisposition.infer: 3,
    }
    status_weight = {
        UncertaintyBacklogStatus.open: 0,
        UncertaintyBacklogStatus.in_progress: 1,
        UncertaintyBacklogStatus.deferred: 2,
        UncertaintyBacklogStatus.resolved: 3,
        UncertaintyBacklogStatus.dismissed: 4,
        UncertaintyBacklogStatus.superseded: 5,
    }
    return sorted(
        entries,
        key=lambda entry: (
            status_weight.get(entry.status, 9),
            disposition_weight.get(entry.disposition, 9),
            -entry.confidence,
            -len(entry.affected_deliverable_keys),
            entry.cost_to_resolve_units,
            entry.source_stage,
            entry.uncertainty_key,
        ),
    )


def resolve_uncertainty(
    db: Session,
    *,
    backlog_id: UUID,
    resolved_answer: str,
) -> UncertaintyBacklogEntry:
    record = db.get(UncertaintyBacklogRecord, backlog_id)
    if record is None:
        raise LookupError("Uncertainty backlog item not found")
    record.status = UncertaintyBacklogStatus.resolved.value
    record.assumed_answer = resolved_answer
    record.resolved_at = utc_now()
    record.updated_at = utc_now()
    db.add(record)
    db.flush()
    return backlog_entry_from_record(record)


def supersede_unresolved_uncertainties(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    source_stage: str | None = None,
    product_mode: ProductProcessingMode | str | None = None,
) -> int:
    entries = list_uncertainty_backlog(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        product_mode=product_mode,
        include_closed=False,
    )
    ids = [
        entry.id
        for entry in entries
        if entry.status != UncertaintyBacklogStatus.resolved
        and (source_stage is None or entry.source_stage == source_stage)
    ]
    if not ids:
        return 0
    rows = db.exec(
        select(UncertaintyBacklogRecord).where(
            UncertaintyBacklogRecord.id.in_([UUID(value) for value in ids])
        )
    ).all()
    now = utc_now()
    for record in rows:
        record.status = UncertaintyBacklogStatus.superseded.value
        record.superseded_at = now
        record.updated_at = now
        db.add(record)
    db.flush()
    return len(rows)
