from __future__ import annotations

from hashlib import sha256
import json
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, status
from sqlmodel import Session, select

from app.api.routes.sessions import get_or_404
from app.db import get_session
from app.models import UserRecord
from app.services.auth_service import get_current_user
from app.services.artifact_diagram_taxonomy import get_diagram_taxonomy_by_key
from app.services.diagram_center.catalog_service import build_catalog_v3, build_diagram_detail_v3
from app.services.diagram_center.contracts import (
    DiagramCatalogV3Response,
    DiagramDetailV3Response,
    DiagramGenerationJobResponse,
    DiagramGenerationRequest,
    DiagramGovernanceEntry,
    DiagramGovernanceAuditEntry,
    DiagramGovernanceOverview,
    DiagramGovernanceResponse,
    DiagramGovernanceUpdate,
)
from app.services.diagram_center.generation_service import create_generation_job, job_response, run_generation_job
from app.services.diagram_center.persistence import (
    DiagramGenerationJobRecord,
    DiagramGovernanceAuditRecord,
    DiagramGovernanceRecord,
    DiagramVersionRecord,
)
from app.services.diagram_center.policy_service import effective_registry_policy
from app.services.diagram_center.registry_service import (
    build_prompt_spec,
    get_registry_entry,
    list_registry_entries,
    load_diagram_registry,
)
from app.services.llm_runtime.runtime_settings_service import load_effective_runtime_settings
from app.services.openai_builder import build_builder_service
from app.services.runtime_access_control import ensure_platform_admin
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context
from app.models import utc_now


router = APIRouter(prefix="/v3", tags=["diagram-center-v3"])


def _project_context(
    db: Session,
    *,
    project_id: UUID,
    current_user: UserRecord,
    workspace_context: WorkspaceAccessContext,
):
    record = get_or_404(db, project_id, current_user.id)
    if record.workspace_id != workspace_context.workspace.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return record


@router.get("/projects/{project_id}/diagrams", response_model=DiagramCatalogV3Response)
def get_diagram_catalog_v3(
    project_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DiagramCatalogV3Response:
    record = _project_context(
        db,
        project_id=project_id,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    return build_catalog_v3(db, record=record, role=workspace_context.membership.role)


@router.get("/projects/{project_id}/diagrams/{diagram_key}", response_model=DiagramDetailV3Response)
def get_diagram_detail_v3(
    project_id: UUID,
    diagram_key: str,
    version_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DiagramDetailV3Response:
    record = _project_context(
        db,
        project_id=project_id,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    detail = build_diagram_detail_v3(
        db,
        record=record,
        role=workspace_context.membership.role,
        diagram_key=diagram_key,
        version_id=version_id,
    )
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagram not found")
    return detail


@router.post(
    "/projects/{project_id}/diagrams/{diagram_key}/generate",
    response_model=DiagramGenerationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def generate_diagram_v3(
    project_id: UUID,
    diagram_key: str,
    payload: DiagramGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DiagramGenerationJobResponse:
    record = _project_context(
        db,
        project_id=project_id,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    catalog = build_catalog_v3(db, record=record, role=workspace_context.membership.role)
    item = next((entry for entry in catalog.entries if entry.key == diagram_key), None)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagram not found")
    if not item.access.can_generate and not (item.current_version and item.access.can_regenerate):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": item.access.reason_code, "message": item.access.reason, "cta_label": item.access.cta_label},
        )
    try:
        job = create_generation_job(
            db,
            record=record,
            diagram_key=diagram_key,
            user_id=current_user.id,
            detail_level=payload.detail_level,
            reason=payload.reason,
            idempotency_key=payload.idempotency_key,
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if job.status in {"queued", "updating"}:
        background_tasks.add_task(run_generation_job, job.id, db.get_bind())
    return job_response(job)


@router.get("/projects/{project_id}/diagram-jobs/{job_id}", response_model=DiagramGenerationJobResponse)
def get_diagram_job_v3(
    project_id: UUID,
    job_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DiagramGenerationJobResponse:
    record = _project_context(
        db,
        project_id=project_id,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    job = db.get(DiagramGenerationJobRecord, job_id)
    if job is None or job.session_id != record.id or job.workspace_id != record.workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagram job not found")
    return job_response(job)


@router.get("/projects/{project_id}/diagrams/{diagram_key}/download")
def download_diagram_v3(
    project_id: UUID,
    diagram_key: str,
    format: str = Query(default="svg", pattern="^(svg|mermaid|json)$"),
    version_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> Response:
    record = _project_context(
        db,
        project_id=project_id,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    detail = build_diagram_detail_v3(
        db,
        record=record,
        role=workspace_context.membership.role,
        diagram_key=diagram_key,
        version_id=version_id,
    )
    if detail is None or detail.model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generated diagram not found")
    if not detail.item.access.can_download:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="The current plan cannot download this diagram")
    content = detail.renderings.get(format)
    if content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Format not available")
    media_types = {"svg": "image/svg+xml", "mermaid": "text/plain", "json": "application/json"}
    extensions = {"svg": "svg", "mermaid": "mmd", "json": "json"}
    return Response(
        content=content,
        media_type=media_types[format],
        headers={"Content-Disposition": f'attachment; filename="{diagram_key}.{extensions[format]}"'},
    )


@router.get("/projects/{project_id}/diagrams/{diagram_key}/compare")
def compare_diagram_versions_v3(
    project_id: UUID,
    diagram_key: str,
    base_version_id: UUID,
    target_version_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    record = _project_context(
        db,
        project_id=project_id,
        current_user=current_user,
        workspace_context=workspace_context,
    )
    catalog = build_catalog_v3(db, record=record, role=workspace_context.membership.role)
    item = next((entry for entry in catalog.entries if entry.key == diagram_key), None)
    if item is None or not item.access.can_compare:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Version comparison is not available")
    rows = db.exec(
        select(DiagramVersionRecord).where(
            DiagramVersionRecord.session_id == record.id,
            DiagramVersionRecord.diagram_key == diagram_key,
            DiagramVersionRecord.id.in_([base_version_id, target_version_id]),
        )
    ).all()
    by_id = {row.id: row for row in rows}
    if base_version_id not in by_id or target_version_id not in by_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagram version not found")
    base_nodes = {node["id"]: node for node in by_id[base_version_id].diagram_model.get("nodes", [])}
    target_nodes = {node["id"]: node for node in by_id[target_version_id].diagram_model.get("nodes", [])}
    return {
        "contract_version": "diagram-version-comparison.v1",
        "diagram_key": diagram_key,
        "base_version_id": base_version_id,
        "target_version_id": target_version_id,
        "added_nodes": [target_nodes[key] for key in sorted(set(target_nodes) - set(base_nodes))],
        "removed_nodes": [base_nodes[key] for key in sorted(set(base_nodes) - set(target_nodes))],
        "changed_nodes": [
            {"before": base_nodes[key], "after": target_nodes[key]}
            for key in sorted(set(base_nodes).intersection(target_nodes))
            if base_nodes[key] != target_nodes[key]
        ],
    }


def _governance_entry(db: Session, diagram_key: str) -> DiagramGovernanceEntry:
    entry = get_registry_entry(diagram_key)
    if entry is None:
        raise LookupError("Diagram type not found")
    enabled, generation_enabled, required_tier, preview_mode, governance = effective_registry_policy(db, entry)
    prompt_spec = build_prompt_spec(entry, override=governance.prompt_override if governance else None)
    taxonomy = get_diagram_taxonomy_by_key().get(entry.key, {})
    return DiagramGovernanceEntry(
        diagram_key=entry.key,
        title=entry.title,
        description=str(taxonomy.get("description") or entry.description),
        category=str(taxonomy.get("category") or entry.category),
        diagram_surface=str(taxonomy.get("diagram_surface") or entry.type),
        notation=str(prompt_spec.get("notation") or entry.notation.value),
        product_scope=[str(value) for value in (taxonomy.get("product_scope") or entry.products)],
        enabled_from_stage=str(taxonomy.get("enabled_from_stage") or entry.stage),
        access_level=str(taxonomy.get("access_level") or ""),
        default_generation_state=str(taxonomy.get("default_generation_state") or ""),
        formats=dict(taxonomy.get("formats") or {}),
        source_artifact_keys=[str(value) for value in taxonomy.get("source_artifact_keys", [])],
        portable_paths=[str(value) for value in taxonomy.get("portable_paths", [])],
        active=bool(taxonomy.get("is_active", entry.active)),
        enabled=enabled,
        generation_enabled=generation_enabled,
        required_tier=required_tier,
        preview_mode=preview_mode,
        prompt_spec_version=prompt_spec["version"],
        prompt_status=governance.prompt_status if governance else "active",
        prompt_override=governance.prompt_override if governance else {},
        prompt_spec=prompt_spec,
        notes=governance.notes if governance else "",
        updated_at=governance.updated_at if governance else None,
    )


def _audit_entry(record: DiagramGovernanceAuditRecord) -> DiagramGovernanceAuditEntry:
    return DiagramGovernanceAuditEntry(
        id=record.id,
        diagram_key=record.diagram_key,
        action=record.action,
        changed_fields=record.changed_fields,
        actor_user_id=record.actor_user_id,
        reason=record.reason,
        created_at=record.created_at,
    )


def _audit_safe_payload(entry: DiagramGovernanceEntry) -> dict[str, object]:
    prompt_override = entry.prompt_override
    prompt_hash = sha256(
        json.dumps(prompt_override, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {
        "enabled": entry.enabled,
        "generation_enabled": entry.generation_enabled,
        "required_tier": entry.required_tier,
        "preview_mode": entry.preview_mode,
        "prompt_status": entry.prompt_status,
        "prompt_override_hash": prompt_hash,
        "prompt_spec_version": entry.prompt_spec_version,
        "notation": str(entry.prompt_spec.get("notation") or ""),
        "standard": str(entry.prompt_spec.get("standard") or ""),
        "renderer_key": str(entry.prompt_spec.get("renderer_key") or ""),
        "validator_key": str(entry.prompt_spec.get("validator_key") or ""),
    }


@router.get("/admin/diagram-governance", response_model=DiagramGovernanceResponse)
def list_diagram_governance_v3(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> DiagramGovernanceResponse:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return DiagramGovernanceResponse(entries=[_governance_entry(db, entry.key) for entry in list_registry_entries(include_inactive=True)])


@router.get("/admin/diagram-governance/overview", response_model=DiagramGovernanceOverview)
def get_diagram_governance_overview_v3(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> DiagramGovernanceOverview:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    runtime_settings = load_effective_runtime_settings(db, workspace_context.workspace.id)
    provider_summary = build_builder_service(runtime_settings).provider_summary()
    jobs = db.exec(
        select(DiagramGenerationJobRecord)
        .where(DiagramGenerationJobRecord.workspace_id == workspace_context.workspace.id)
        .order_by(DiagramGenerationJobRecord.requested_at.desc())
        .limit(50)
    ).all()
    versions = db.exec(
        select(DiagramVersionRecord)
        .where(DiagramVersionRecord.workspace_id == workspace_context.workspace.id)
        .order_by(DiagramVersionRecord.created_at.desc())
        .limit(500)
    ).all()
    audits = db.exec(
        select(DiagramGovernanceAuditRecord)
        .order_by(DiagramGovernanceAuditRecord.created_at.desc())
        .limit(20)
    ).all()
    job_counts = {state: 0 for state in ("queued", "generating", "updating", "available", "error")}
    for job in jobs:
        job_counts[job.status] = job_counts.get(job.status, 0) + 1
    scores = [int(version.quality_report.get("score", 0)) for version in versions if version.quality_report]
    registry = load_diagram_registry()
    return DiagramGovernanceOverview(
        active_provider=str(provider_summary.get("provider", runtime_settings.active_provider.value)),
        provider_mode=str(provider_summary.get("mode", "")),
        model_name=str(
            provider_summary.get("reasoning_model")
            or provider_summary.get("fast_model")
            or provider_summary.get("model", "")
        ),
        provider_configured=bool(provider_summary.get("configured", False)),
        registry_version=registry.schema_version,
        prompt_spec_version=registry.prompt_spec_version,
        job_counts=job_counts,
        total_versions=len(versions),
        average_quality_score=round(sum(scores) / len(scores)) if scores else 0,
        recent_jobs=[job_response(job) for job in jobs[:10]],
        recent_audit=[_audit_entry(record) for record in audits],
    )


@router.patch("/admin/diagram-governance/{diagram_key}", response_model=DiagramGovernanceEntry)
def update_diagram_governance_v3(
    diagram_key: str,
    payload: DiagramGovernanceUpdate,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> DiagramGovernanceEntry:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    entry = get_registry_entry(diagram_key)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagram type not found")
    before_entry = _governance_entry(db, entry.key)
    governance = db.exec(
        select(DiagramGovernanceRecord).where(DiagramGovernanceRecord.diagram_key == entry.key)
    ).first()
    if governance is None:
        governance = DiagramGovernanceRecord(diagram_key=entry.key)
    governance.enabled = payload.enabled
    governance.generation_enabled = payload.generation_enabled
    governance.required_tier_override = payload.required_tier_override
    governance.preview_mode_override = payload.preview_mode_override
    governance.prompt_status = payload.prompt_status
    governance.prompt_override = payload.prompt_override
    governance.notes = payload.notes
    governance.updated_by_user_id = current_user.id
    governance.updated_at = utc_now()
    db.add(governance)
    db.flush()
    after_entry = _governance_entry(db, entry.key)
    before_payload = _audit_safe_payload(before_entry)
    after_payload = _audit_safe_payload(after_entry)
    changed_fields = [key for key in after_payload if before_payload.get(key) != after_payload.get(key)]
    db.add(
        DiagramGovernanceAuditRecord(
            diagram_key=entry.key,
            changed_fields=changed_fields,
            before_payload=before_payload,
            after_payload=after_payload,
            actor_user_id=current_user.id,
            reason=payload.notes[:500],
        )
    )
    db.commit()
    return _governance_entry(db, entry.key)
