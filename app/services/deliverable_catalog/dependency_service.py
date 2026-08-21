from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import utc_now
from app.services.deliverable_catalog.contracts import (
    DeliverableRegenerationScope,
    DeliverableStalenessReport,
    LEAN_STAGE_ORDER,
)
from app.services.deliverable_catalog.persistence import DeliverableQualitySnapshotRecord
from app.services.deliverable_catalog.quality_service import fingerprint_payload
from app.services.deliverable_catalog.registry_service import list_registry_entries
from app.services.product_processing.contracts import UncertaintyBacklogStatus
from app.services.product_processing.persistence import UncertaintyBacklogRecord


CLOSED_UNCERTAINTY_STATUSES = {
    UncertaintyBacklogStatus.resolved.value,
    UncertaintyBacklogStatus.dismissed.value,
    UncertaintyBacklogStatus.superseded.value,
}


def _stage_order(stage: str) -> int:
    try:
        return LEAN_STAGE_ORDER.index(stage)
    except ValueError:
        return 999


def _entry_dependency_keys(entry) -> set[str]:
    return set(entry.dependency_policy.depends_on) | set(entry.dependency_policy.invalidates_on_change)


def compute_deliverable_staleness(
    changed_dependency_keys: list[str],
) -> DeliverableStalenessReport:
    changed = {key for key in changed_dependency_keys if str(key or "").strip()}
    if not changed:
        return DeliverableStalenessReport()

    entries = list_registry_entries()
    stale_keys: set[str] = set()
    reasons: dict[str, list[str]] = {}
    frontier = set(changed)
    while frontier:
        next_frontier: set[str] = set()
        for entry in entries:
            if entry.deliverable_key in stale_keys:
                continue
            matches = sorted(_entry_dependency_keys(entry).intersection(frontier))
            if not matches:
                continue
            stale_keys.add(entry.deliverable_key)
            next_frontier.add(entry.deliverable_key)
            reasons[entry.deliverable_key] = [f"dependency_changed:{item}" for item in matches]
        frontier = next_frontier

    all_keys = {entry.deliverable_key for entry in entries}
    return DeliverableStalenessReport(
        changed_dependency_keys=sorted(changed),
        stale_deliverable_keys=sorted(
            stale_keys,
            key=lambda key: (
                _stage_order(next((entry.stage for entry in entries if entry.deliverable_key == key), "")),
                key,
            ),
        ),
        unchanged_deliverable_keys=sorted(all_keys - stale_keys),
        reasons_by_deliverable=reasons,
    )


def resolve_regeneration_scope(
    *,
    changed_dependency_keys: list[str],
    source_deliverable_key: str = "",
) -> DeliverableRegenerationScope:
    report = compute_deliverable_staleness(changed_dependency_keys)
    entries_by_key = {entry.deliverable_key: entry for entry in list_registry_entries()}
    ordered = sorted(
        report.stale_deliverable_keys,
        key=lambda key: (_stage_order(entries_by_key[key].stage), entries_by_key[key].sort_order, key),
    )
    if source_deliverable_key and source_deliverable_key in ordered:
        ordered = [key for key in ordered if key != source_deliverable_key]
    return DeliverableRegenerationScope(
        source_deliverable_key=source_deliverable_key,
        changed_dependency_keys=report.changed_dependency_keys,
        affected_deliverable_keys=report.stale_deliverable_keys,
        ordered_regeneration_keys=ordered,
        unaffected_deliverable_keys=report.unchanged_deliverable_keys,
    )


def supersede_uncertainties_for_stale_deliverables(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    stale_deliverable_keys: list[str],
) -> int:
    stale = set(stale_deliverable_keys)
    if not stale:
        return 0
    rows = db.exec(
        select(UncertaintyBacklogRecord).where(
            UncertaintyBacklogRecord.workspace_id == workspace_id,
            UncertaintyBacklogRecord.session_id == session_id,
            UncertaintyBacklogRecord.status.notin_(list(CLOSED_UNCERTAINTY_STATUSES)),
        )
    ).all()
    now = utc_now()
    changed = 0
    for row in rows:
        if stale.intersection(set(row.affected_deliverable_keys or [])):
            row.status = UncertaintyBacklogStatus.superseded.value
            row.superseded_at = now
            row.updated_at = now
            db.add(row)
            changed += 1
    db.flush()
    return changed


def invalidate_deliverables_for_change(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    changed_dependency_keys: list[str],
    source_deliverable_key: str = "",
) -> DeliverableStalenessReport:
    report = compute_deliverable_staleness(changed_dependency_keys)
    source_fingerprint = fingerprint_payload(
        {
            "changed_dependency_keys": report.changed_dependency_keys,
            "source_deliverable_key": source_deliverable_key,
        }
    )
    for key in report.stale_deliverable_keys:
        db.add(
            DeliverableQualitySnapshotRecord(
                workspace_id=workspace_id,
                session_id=session_id,
                deliverable_key=key,
                version_ref=f"stale::{source_deliverable_key or 'dependency_change'}",
                state="stale",
                score=0,
                warnings=report.reasons_by_deliverable.get(key, []),
                checks={"stale": True},
                source_fingerprint=source_fingerprint,
            )
        )
    superseded = supersede_uncertainties_for_stale_deliverables(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        stale_deliverable_keys=report.stale_deliverable_keys,
    )
    db.flush()
    return report.model_copy(update={"superseded_uncertainty_count": superseded})
