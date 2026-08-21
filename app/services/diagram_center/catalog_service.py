from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from sqlmodel import Session, select

from app.models import SessionRecord, WorkspaceRole
from app.services.diagram_center.contracts import (
    DiagramCatalogItemV3,
    DiagramCatalogV3Response,
    DiagramDetailV3Response,
    DiagramNotation,
    DiagramModel,
    DiagramQualityReport,
    DiagramVersionSummary,
)
from app.services.diagram_center.persistence import DiagramGenerationJobRecord, DiagramVersionRecord
from app.services.diagram_center.policy_service import effective_registry_policy, resolve_diagram_policy, resolve_project_stage
from app.services.diagram_center.quality_service import evaluate_diagram_quality
from app.services.diagram_center.registry_service import build_prompt_spec, get_registry_entry, list_registry_entries
from app.services.diagram_center.renderer_service import RENDERER_REVISION, render_diagram
from app.services.llm_runtime.runtime_settings_service import load_effective_runtime_settings


_PLANTUML_NOTATIONS = {
    "uml_use_case",
    "uml_activity",
    "uml_component",
    "sequence",
    "class",
    "state",
    "deployment",
    "package",
    "c4",
}

_SPECIALIZED_SVG_NOTATIONS = {
    "uml_use_case",
    "uml_activity",
    "uml_component",
    "bpmn",
    "deployment",
    "package",
}


def _svg_renderer_revision(svg: str) -> str:
    marker = 'data-renderer-revision="'
    start = svg.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = svg.find('"', start)
    return svg[start:end] if end >= start else ""


def _notation_value(value: DiagramNotation | str | object) -> str:
    if isinstance(value, DiagramNotation):
        return value.value
    return str(value or "")


def _current_policy_metadata(item: DiagramCatalogItemV3) -> dict[str, str]:
    return {
        "standard": item.standard,
        "source_contract": item.source_contract,
        "presentation_contract": item.presentation_contract,
        "renderer_key": item.renderer_key,
        "validator_key": item.validator_key,
    }


def _metadata_conflicts_with_policy(metadata: dict[str, object], item: DiagramCatalogItemV3) -> bool:
    for key, expected in _current_policy_metadata(item).items():
        stored = str(metadata.get(key) or "")
        if stored and stored != expected:
            return True
    return False


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
    if item.needs_layout_upgrade and item.access.can_generate:
        actions.append("layout_upgrade")
    return actions


def _layout_upgrade_reason(
    latest_version: DiagramVersionRecord | None,
    item: DiagramCatalogItemV3,
) -> str:
    if latest_version is None:
        return ""
    renderings = latest_version.renderings if isinstance(latest_version.renderings, dict) else {}
    svg = str(renderings.get("svg") or "")
    current_revision = _svg_renderer_revision(svg)
    if not current_revision:
        return "La version actual no declara revision de renderer; conviene regenerar para aplicar sizing, routing y split legible."
    if current_revision != RENDERER_REVISION:
        return (
            f"La version actual usa {current_revision}; la politica vigente usa {RENDERER_REVISION} "
            "con mejoras de legibilidad, espaciado y ruteo."
        )
    return ""


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
        enabled, generation_enabled, required_tier, preview_mode, governance = effective_registry_policy(db, entry)
        prompt_spec = build_prompt_spec(entry, override=governance.prompt_override if governance else None)
        effective_notation = DiagramNotation(str(prompt_spec.get("notation") or entry.notation.value))
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
            notation=effective_notation,
            standard=str(prompt_spec.get("standard") or entry.standard),
            source_contract=str(prompt_spec.get("source_contract") or entry.source_contract),
            presentation_contract=str(prompt_spec.get("presentation_contract") or entry.presentation_contract),
            renderer_key=str(prompt_spec.get("renderer_key") or entry.renderer_key),
            validator_key=str(prompt_spec.get("validator_key") or entry.validator_key),
            complexity=entry.complexity,
            stage=entry.stage,
            required_tier=required_tier,
            products=entry.products,
            generation_state=_generation_state(latest_version, latest_job),
            access=access,
            updated_at=latest_version.created_at if latest_version is not None else None,
            current_version=_version_summary(latest_version) if latest_version is not None else None,
        )
        item.layout_upgrade_reason = _layout_upgrade_reason(latest_version, item)
        item.needs_layout_upgrade = bool(item.layout_upgrade_reason)
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


def _hydrate_model_for_current_registry(model: DiagramModel, item: DiagramCatalogItemV3) -> DiagramModel:
    metadata = dict(model.metadata or {})
    stored_notation = _notation_value(model.notation)
    current_notation = _notation_value(item.notation)
    policy_changed = stored_notation != current_notation or _metadata_conflicts_with_policy(metadata, item)
    refreshed_metadata = dict(metadata)
    for key, expected in _current_policy_metadata(item).items():
        previous = refreshed_metadata.get(key)
        if policy_changed or not previous:
            if previous and str(previous) != expected and f"legacy_{key}" not in refreshed_metadata:
                refreshed_metadata[f"legacy_{key}"] = previous
            refreshed_metadata[key] = expected
    refreshed_metadata["compatibility_hydrated"] = True
    refreshed_metadata["effective_notation"] = current_notation
    if policy_changed:
        refreshed_metadata["policy_rehydrated"] = True
        refreshed_metadata["policy_rehydrated_reason"] = "registry_or_governance_changed"
        if stored_notation and stored_notation != current_notation and "legacy_notation" not in refreshed_metadata:
            refreshed_metadata["legacy_notation"] = stored_notation
    return model.model_copy(
        update={
            "diagram_key": item.key,
            "title": model.title or item.title,
            "notation": item.notation,
            "metadata": refreshed_metadata,
        }
    )


def _renderings_need_refresh(renderings: dict[str, str], model: DiagramModel, item: DiagramCatalogItemV3) -> bool:
    svg = renderings.get("svg") or ""
    if not svg:
        return True
    if not renderings.get("presentation"):
        return True
    if model.metadata.get("renderer_key") != item.renderer_key:
        return True
    if _svg_renderer_revision(svg) != RENDERER_REVISION:
        return True
    notation = item.notation.value
    if notation in _SPECIALIZED_SVG_NOTATIONS and f'data-diagram-notation="{notation}"' not in svg:
        return True
    if notation == "bpmn":
        expected_source_keys = {"bpmn_xml"}
    elif notation in _PLANTUML_NOTATIONS:
        expected_source_keys = {"plantuml"}
    else:
        expected_source_keys = {"mermaid"}
    return not bool(expected_source_keys.intersection(renderings))


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
    model = _hydrate_model_for_current_registry(DiagramModel.model_validate(selected.diagram_model), item)
    renderings = dict(selected.renderings)
    quality = DiagramQualityReport.model_validate(selected.quality_report)
    should_persist_refresh = False
    if selected.diagram_model != model.model_dump(mode="json"):
        should_persist_refresh = True
    if item.access.access_state == "preview":
        model = _limited_preview(model)

        renderings = render_diagram(model)
    elif _renderings_need_refresh(renderings, model, item) and not item.needs_layout_upgrade:
        metadata = dict(model.metadata or {})
        previous_revision = _svg_renderer_revision(renderings.get("svg") or "")
        if previous_revision and previous_revision != RENDERER_REVISION:
            metadata["legacy_renderer_revision"] = previous_revision
            metadata["layout_upgrade_reason"] = "layout_upgrade"
        metadata["renderer_revision"] = RENDERER_REVISION
        model = model.model_copy(update={"metadata": metadata})
        renderings = render_diagram(model)
        should_persist_refresh = True
    if item.access.access_state != "preview" and should_persist_refresh:
        quality = evaluate_diagram_quality(model)
        selected.diagram_model = model.model_dump(mode="json")
        selected.renderings = renderings
        selected.quality_report = quality.model_dump(mode="json")
        db.add(selected)
        db.commit()
        db.refresh(selected)
    return DiagramDetailV3Response(
        project_id=record.id,
        item=item,
        model=model,
        renderings=renderings,
        quality=quality,
        versions=[_version_summary(value) for value in versions],
    )
