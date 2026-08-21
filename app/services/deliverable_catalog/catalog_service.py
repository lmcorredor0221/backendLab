from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import ArtifactRegistryRecord, CommercialTier, WorkspaceRole
from app.services.deliverable_catalog.contracts import (
    DeliverableCatalogItem,
    DeliverableCatalogResponse,
    DeliverableDetailResponse,
    DeliverableGovernanceOverview,
    DeliverableQualitySnapshotSummaryEntry,
    DeliverableQualitySummary,
    DeliverablePolicyContext,
    DeliverableRegistryEntry,
    DeliverableType,
)
from app.services.deliverable_catalog.persistence import (
    DeliverableGenerationJobRecord,
    DeliverableGovernanceAuditRecord,
    DeliverableGovernanceRecord,
    DeliverableQualitySnapshotRecord,
)
from app.services.deliverable_catalog.policy_service import (
    deliverable_governance_entry,
    governance_audit_entry,
    normalize_deliverable_stage,
    resolve_deliverable_policy,
)
from app.services.deliverable_catalog.registry_service import get_registry_entry, list_registry_entries


ACTIVE_GENERATION_STATES = {"queued", "generating", "updating", "available"}


def _artifact_keys_for_entry(entry: DeliverableRegistryEntry) -> list[str]:
    keys = [entry.deliverable_key, *entry.canonical_paths, *entry.portable_paths]
    fallback_key = f"Deliverables/{entry.deliverable_key}.{entry.formats.preferred}"
    keys.append(fallback_key)
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        normalized = str(key or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _runtime_state_for_entry(
    db: Session,
    entry: DeliverableRegistryEntry,
    *,
    session_id: UUID | None,
) -> tuple[bool, str, str]:
    if session_id is None:
        return False, "pending", "unknown"

    artifact_keys = _artifact_keys_for_entry(entry)
    artifact = db.exec(
        select(ArtifactRegistryRecord)
        .where(
            ArtifactRegistryRecord.session_id == session_id,
            ArtifactRegistryRecord.artifact_key.in_(artifact_keys),
        )
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).first()
    job = db.exec(
        select(DeliverableGenerationJobRecord)
        .where(
            DeliverableGenerationJobRecord.session_id == session_id,
            DeliverableGenerationJobRecord.deliverable_key == entry.deliverable_key,
        )
        .order_by(DeliverableGenerationJobRecord.updated_at.desc())
    ).first()
    quality = db.exec(
        select(DeliverableQualitySnapshotRecord)
        .where(
            DeliverableQualitySnapshotRecord.session_id == session_id,
            DeliverableQualitySnapshotRecord.deliverable_key == entry.deliverable_key,
        )
        .order_by(DeliverableQualitySnapshotRecord.created_at.desc())
    ).first()
    if artifact is not None:
        generation_state = "available"
    elif job is not None and job.status in ACTIVE_GENERATION_STATES:
        generation_state = job.status
    elif job is not None and job.status in {"error", "failed", "requires_attention"}:
        generation_state = "error"
    else:
        generation_state = "pending"
    quality_state = quality.state if quality is not None else "unknown"
    return artifact is not None, generation_state, quality_state


def _catalog_item(
    db: Session,
    entry: DeliverableRegistryEntry,
    *,
    workspace_id: UUID,
    session_id: UUID | None,
    role: WorkspaceRole,
    tier: CommercialTier,
    current_stage: str,
) -> DeliverableCatalogItem:
    has_current_version, generation_state, quality_state = _runtime_state_for_entry(db, entry, session_id=session_id)
    access = resolve_deliverable_policy(
        db,
        entry,
        DeliverablePolicyContext(
            workspace_id=workspace_id,
            has_current_version=has_current_version,
            generation_state=generation_state,  # type: ignore[arg-type]
            quality_state=quality_state,  # type: ignore[arg-type]
            role=role,
            tier=tier,
            current_stage=current_stage,
        ),
    )
    return DeliverableCatalogItem(
        key=entry.deliverable_key,
        title=entry.title,
        description=entry.description,
        deliverable_type=entry.deliverable_type,
        category=entry.category,
        stage=entry.stage,
        enabled_from_stage=entry.enabled_from_stage,
        product_scope=list(entry.product_scope),
        required_tier=access.required_tier,
        access_level=entry.access_level,
        generation_mode=entry.generation_mode,
        formats=entry.formats,
        exportable=entry.exportable,
        blueprint_download=entry.blueprint_download,
        acp_download=entry.acp_download,
        sort_order=entry.sort_order,
        access=access,
    )


def build_deliverable_catalog_response(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID | None = None,
    role: WorkspaceRole,
    tier: CommercialTier,
    current_stage: str,
    stage_filter: str | None = None,
    include_inactive: bool = False,
) -> DeliverableCatalogResponse:
    normalized_stage = normalize_deliverable_stage(current_stage)
    entries = list_registry_entries(include_inactive=include_inactive)
    if stage_filter:
        entries = [entry for entry in entries if entry.stage == stage_filter]
    return DeliverableCatalogResponse(
        current_stage=normalized_stage,
        tier=tier,
        entries=[
            _catalog_item(
                db,
                entry,
                workspace_id=workspace_id,
                session_id=session_id,
                role=role,
                tier=tier,
                current_stage=normalized_stage,
            )
            for entry in entries
        ],
    )


def build_deliverable_detail_response(
    db: Session,
    *,
    deliverable_key: str,
    workspace_id: UUID,
    session_id: UUID | None = None,
    role: WorkspaceRole,
    tier: CommercialTier,
    current_stage: str,
) -> DeliverableDetailResponse | None:
    normalized_stage = normalize_deliverable_stage(current_stage)
    entry = get_registry_entry(deliverable_key)
    if entry is None:
        return None
    has_current_version, generation_state, quality_state = _runtime_state_for_entry(db, entry, session_id=session_id)
    access = resolve_deliverable_policy(
        db,
        entry,
        DeliverablePolicyContext(
            workspace_id=workspace_id,
            has_current_version=has_current_version,
            generation_state=generation_state,  # type: ignore[arg-type]
            quality_state=quality_state,  # type: ignore[arg-type]
            role=role,
            tier=tier,
            current_stage=normalized_stage,
        ),
    )
    return DeliverableDetailResponse(
        entry=entry,
        access=access,
        governance=deliverable_governance_entry(db, entry, workspace_id=workspace_id),
    )


def build_deliverable_governance_overview(
    db: Session,
    *,
        workspace_id: UUID | None = None,
    tier: CommercialTier = CommercialTier.blueprint,
    current_stage: str = "discover",
    role: WorkspaceRole = WorkspaceRole.admin,
    product: str | None = None,
    stage: str | None = None,
    deliverable_type: DeliverableType | None = None,
    quality_state: str | None = None,
) -> DeliverableGovernanceOverview:
    normalized_stage = normalize_deliverable_stage(current_stage)
    entries = list_registry_entries(include_inactive=True)
    if product:
        entries = [entry for entry in entries if product in entry.product_scope]
    if stage:
        entries = [entry for entry in entries if entry.stage == stage]
    if deliverable_type:
        entries = [entry for entry in entries if entry.deliverable_type == deliverable_type]
    active_entries = [entry for entry in entries if entry.active]
    entry_by_key = {entry.deliverable_key: entry for entry in entries}
    entry_keys = set(entry_by_key)
    scope_keys = ["platform"]
    if workspace_id is not None:
        scope_keys.append(str(workspace_id))
    governed_count = []
    if entry_keys:
        governed_count = db.exec(
            select(DeliverableGovernanceRecord).where(
                DeliverableGovernanceRecord.scope_key.in_(scope_keys),
                DeliverableGovernanceRecord.deliverable_key.in_(entry_keys),
            )
        ).all()
    by_type: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    by_access_state: dict[str, int] = {}
    by_prompt_status: dict[str, int] = {}
    for entry in entries:
        by_type[entry.deliverable_type.value] = by_type.get(entry.deliverable_type.value, 0) + 1
        by_stage[entry.stage] = by_stage.get(entry.stage, 0) + 1
        access = resolve_deliverable_policy(
            db,
            entry,
            DeliverablePolicyContext(
                workspace_id=workspace_id,
                role=role,
                tier=tier,
                current_stage=normalized_stage,
            ),
        )
        by_access_state[access.access_state] = by_access_state.get(access.access_state, 0) + 1
        by_prompt_status[access.effective_prompt_status] = by_prompt_status.get(access.effective_prompt_status, 0) + 1
    audit_query = select(DeliverableGovernanceAuditRecord).order_by(DeliverableGovernanceAuditRecord.created_at.desc())
    if entry_keys:
        audit_query = audit_query.where(DeliverableGovernanceAuditRecord.deliverable_key.in_(entry_keys))
    audits = db.exec(audit_query.limit(20)).all()
    quality_summary = _deliverable_quality_summary(
        db,
        entry_by_key=entry_by_key,
        workspace_id=workspace_id,
        quality_state=quality_state,
    )
    return DeliverableGovernanceOverview(
        total_entries=len(entries),
        active_entries=len(active_entries),
        governed_entries=len(governed_count),
        by_type=by_type,
        by_stage=by_stage,
        by_access_state=by_access_state,
        by_prompt_status=by_prompt_status,
        quality_summary=quality_summary,
        recent_audit=[governance_audit_entry(record) for record in audits],
    )


def _deliverable_quality_summary(
    db: Session,
    *,
    entry_by_key: dict[str, DeliverableRegistryEntry],
    workspace_id: UUID | None,
    quality_state: str | None,
) -> DeliverableQualitySummary:
    if not entry_by_key:
        return DeliverableQualitySummary()

    query = select(DeliverableQualitySnapshotRecord).where(
        DeliverableQualitySnapshotRecord.deliverable_key.in_(set(entry_by_key)),
    )
    if workspace_id is not None:
        query = query.where(DeliverableQualitySnapshotRecord.workspace_id == workspace_id)
    if quality_state:
        query = query.where(DeliverableQualitySnapshotRecord.state == quality_state)

    snapshots = db.exec(query.order_by(DeliverableQualitySnapshotRecord.created_at.desc())).all()
    by_state: dict[str, int] = {}
    score_sum = 0
    recent: list[DeliverableQualitySnapshotSummaryEntry] = []
    for snapshot in snapshots:
        state = snapshot.state or "unknown"
        by_state[state] = by_state.get(state, 0) + 1
        score_sum += int(snapshot.score or 0)
        if len(recent) < 20:
            entry = entry_by_key.get(snapshot.deliverable_key)
            recent.append(
                DeliverableQualitySnapshotSummaryEntry(
                    id=snapshot.id,
                    workspace_id=snapshot.workspace_id,
                    session_id=snapshot.session_id,
                    deliverable_key=snapshot.deliverable_key,
                    title=entry.title if entry else "",
                    deliverable_type=entry.deliverable_type.value if entry else "",
                    stage=entry.stage if entry else "",
                    product_scope=list(entry.product_scope) if entry else [],
                    version_ref=snapshot.version_ref,
                    state=state,
                    score=snapshot.score,
                    errors_count=len(snapshot.errors or []),
                    warnings_count=len(snapshot.warnings or []),
                    created_at=snapshot.created_at,
                )
            )
    average_score = round(score_sum / len(snapshots), 2) if snapshots else 0.0
    return DeliverableQualitySummary(
        total_snapshots=len(snapshots),
        average_score=average_score,
        by_state=by_state,
        recent_snapshots=recent,
    )
