from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from sqlmodel import Session, select

from app.models import SessionRecord, WorkspaceRole
from app.services.diagram_center.contracts import (
    DiagramCatalogItemV3,
    DiagramCatalogV3Response,
    DiagramDetailV3Response,
    DiagramModel,
    DiagramQualityReport,
    DiagramVersionSummary,
)
from app.services.diagram_center.persistence import DiagramGenerationJobRecord, DiagramVersionRecord
from app.services.diagram_center.policy_service import effective_registry_policy, resolve_diagram_policy, resolve_project_stage
from app.services.diagram_center.registry_service import get_registry_entry, list_registry_entries
from app.services.llm_runtime.runtime_settings_service import load_effective_runtime_settings


def _version_summary(record: DiagramVersionRecord) -> DiagramVersionSummary:
    quality_score = int(record.quality_report.get("score", 0)) if isinstance(record.quality_report, dict) else 0
    return DiagramVersionSummary(
        id=record.id,
        version_number=record.version_number,
        state=record.state,
        provider_key=record.provider_key,
        model_name=record.model_name,
        prompt_spec_version=record.prompt_spec_version,
        quality_score=quality_score,
        created_at=record.created_at,
    )


def _generation_state(
    latest_version: DiagramVersionRecord | None,
    latest_job: DiagramGenerationJobRecord | None,
) -> str:
    if latest_job is not None and latest_job.status in {"queued", "generating", "updating"}:
        return latest_job.status
    if latest_job is not None and latest_job.status == "error" and latest_version is None:
        return "error"
    if latest_version is not None:
        return "available"
    return "pending"


def _available_actions(item: DiagramCatalogItemV3, version_count: int) -> list[str]:
    actions: list[str] = []
    if item.access.can_view and item.current_version is not None:
        actions.append("view")
    if item.access.can_generate and item.current_version is None:
        actions.append("generate")
    if item.access.can_regenerate and item.current_version is not None:
        actions.append("regenerate")
    if item.access.can_download and item.current_version is not None:
        actions.append("download")
    if item.access.can_compare and version_count > 1:
        actions.append("compare")
    return actions


def build_catalog_v3(
    db: Session,
    *,
    record: SessionRecord,
    role: WorkspaceRole | None,
) -> DiagramCatalogV3Response:
    versions = db.exec(
        select(DiagramVersionRecord)
        .where(DiagramVersionRecord.session_id == record.id)
        .order_by(DiagramVersionRecord.diagram_key, DiagramVersionRecord.version_number.desc())
    ).all()
    jobs = db.exec(
        select(DiagramGenerationJobRecord)
        .where(DiagramGenerationJobRecord.session_id == record.id)
        .order_by(DiagramGenerationJobRecord.diagram_key, DiagramGenerationJobRecord.requested_at.desc())
    ).all()
    versions_by_key: dict[str, list[DiagramVersionRecord]] = defaultdict(list)
    jobs_by_key: dict[str, list[DiagramGenerationJobRecord]] = defaultdict(list)
    for version in versions:
        versions_by_key[version.diagram_key].append(version)
    for job in jobs:
        jobs_by_key[job.diagram_key].append(job)

    project_stage = resolve_project_stage(db, record)
    tier = record.commercial_tier.value
    items: list[DiagramCatalogItemV3] = []
    for entry in list_registry_entries():
        enabled, generation_enabled, required_tier, preview_mode, _ = effective_registry_policy(db, entry)
        access = resolve_diagram_policy(
            entry=entry,
            project_stage=project_stage,
            current_tier=tier,
            role=role,
            enabled=enabled,
            generation_enabled=generation_enabled,
            required_tier=required_tier,
            preview_mode=preview_mode,
        )
        latest_version = versions_by_key[entry.key][0] if versions_by_key[entry.key] else None
        latest_job = jobs_by_key[entry.key][0] if jobs_by_key[entry.key] else None
        item = DiagramCatalogItemV3(
            key=entry.key,
            title=entry.title,
            description=entry.description,
            benefit=entry.benefit,
            category=entry.category,
            type=entry.type,
            family=entry.family,
            notation=entry.notation,
            complexity=entry.complexity,
            stage=entry.stage,
            required_tier=required_tier,
            products=entry.products,
            generation_state=_generation_state(latest_version, latest_job),
            access=access,
            updated_at=latest_version.created_at if latest_version is not None else None,
            current_version=_version_summary(latest_version) if latest_version is not None else None,
        )
        item.available_actions = _available_actions(item, len(versions_by_key[entry.key]))
        items.append(item)

    runtime = load_effective_runtime_settings(db, record.workspace_id)
    return DiagramCatalogV3Response(
        project_id=record.id,
        workspace_id=record.workspace_id,
        current_stage=project_stage,
        tier=tier,
        provider_key=runtime.active_provider.value,
        total_count=len(items),
        available_count=sum(item.access.access_state == "available" for item in items),
        preview_count=sum(item.access.access_state == "preview" for item in items),
        locked_count=sum(item.access.access_state in {"locked", "stage_locked", "disabled"} for item in items),
        entries=items,
    )


def _limited_preview(model: DiagramModel) -> DiagramModel:
    nodes = model.nodes[:4]
    node_ids = {node.id for node in nodes}
    return model.model_copy(
        update={
            "description": "Vista previa limitada. Desbloquea el plan requerido para consultar el diagrama completo.",
            "nodes": nodes,
            "edges": [edge for edge in model.edges if edge.source in node_ids and edge.target in node_ids][:4],
            "groups": [],
            "metadata": {**model.metadata, "preview_limited": True},
        }
    )


def build_diagram_detail_v3(
    db: Session,
    *,
    record: SessionRecord,
    role: WorkspaceRole | None,
    diagram_key: str,
    version_id: UUID | None = None,
) -> DiagramDetailV3Response | None:
    entry = get_registry_entry(diagram_key)
    if entry is None:
        return None
    catalog = build_catalog_v3(db, record=record, role=role)
    item = next((candidate for candidate in catalog.entries if candidate.key == entry.key), None)
    if item is None:
        return None
    version_query = select(DiagramVersionRecord).where(
        DiagramVersionRecord.session_id == record.id,
        DiagramVersionRecord.diagram_key == entry.key,
    )
    versions = db.exec(version_query.order_by(DiagramVersionRecord.version_number.desc())).all()
    selected = next((version for version in versions if version.id == version_id), None) if version_id else (versions[0] if versions else None)
    if selected is None or not item.access.can_view:
        return DiagramDetailV3Response(project_id=record.id, item=item, versions=[_version_summary(value) for value in versions])
    model = DiagramModel.model_validate(selected.diagram_model)
    renderings = dict(selected.renderings)
    if item.access.access_state == "preview":
        model = _limited_preview(model)
        from app.services.diagram_center.renderer_service import render_diagram

        renderings = render_diagram(model)
    return DiagramDetailV3Response(
        project_id=record.id,
        item=item,
        model=model,
        renderings=renderings,
        quality=DiagramQualityReport.model_validate(selected.quality_report),
        versions=[_version_summary(value) for value in versions],
    )

