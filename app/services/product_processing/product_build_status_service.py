from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sqlmodel import Session, select

from app.models import CommercialTier, SessionRecord, UserRecord, WorkspaceRole, utc_now
from app.services.commerce_service import role_for_user, tier_rank
from app.services.commercial_access import build_commercial_access_snapshot_v2
from app.services.deliverable_catalog.catalog_service import build_deliverable_catalog_response
from app.services.deliverable_catalog.contracts import DeliverableCatalogItem
from app.services.deliverable_catalog.persistence import DeliverableGenerationJobRecord
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.product_processing.contracts import (
    ProductBuildAction,
    ProductBuildActionState,
    ProductBuildAttentionItem,
    ProductBuildAttentionSeverity,
    ProductBuildAttentionSummary,
    ProductBuildCurrentActivity,
    ProductBuildDeliverableState,
    ProductBuildDeliverableStatus,
    ProductBuildEntitlement,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductBuildProgress,
    ProductBuildRecoverableError,
    ProductBuildStageStatus,
    ProductBuildStatus,
    ProductProcessingMode,
    UncertaintyBacklogStatus,
    calculate_product_build_percent,
)
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord, UncertaintyBacklogRecord
from app.services.product_processing.product_build_run_service import list_product_build_runs, list_product_build_steps


@dataclass(frozen=True)
class ProductBuildMeta:
    product_key: ProductBuildProductKey
    product_mode: ProductProcessingMode
    required_tier: CommercialTier
    label: str
    included_scopes: tuple[str, ...]
    checkout_href: str


PRODUCT_BUILD_META: dict[ProductBuildProductKey, ProductBuildMeta] = {
    ProductBuildProductKey.blueprint_basic: ProductBuildMeta(
        product_key=ProductBuildProductKey.blueprint_basic,
        product_mode=ProductProcessingMode.basic_free,
        required_tier=CommercialTier.blueprint,
        label="Blueprint Basico",
        included_scopes=("blueprint",),
        checkout_href="",
    ),
    ProductBuildProductKey.blueprint_pro: ProductBuildMeta(
        product_key=ProductBuildProductKey.blueprint_pro,
        product_mode=ProductProcessingMode.premium_enrichment,
        required_tier=CommercialTier.blueprint_pro,
        label="Blueprint Pro",
        included_scopes=("blueprint", "blueprint_pro"),
        checkout_href="/pricing/blueprint-pro",
    ),
    ProductBuildProductKey.acp: ProductBuildMeta(
        product_key=ProductBuildProductKey.acp,
        product_mode=ProductProcessingMode.acp_implementation,
        required_tier=CommercialTier.acp,
        label="Agent Construction Package",
        included_scopes=("blueprint", "blueprint_pro", "acp"),
        checkout_href="/pricing/acp",
    ),
}

ACTIVE_JOB_STATES = {"queued", "generating", "updating", "running"}
ERROR_JOB_STATES = {"error", "failed"}
CLOSED_UNCERTAINTY_STATUSES = {
    UncertaintyBacklogStatus.resolved.value,
    UncertaintyBacklogStatus.dismissed.value,
    UncertaintyBacklogStatus.superseded.value,
}
STAGE_LABELS: dict[str, str] = {
    "discover": "Descubrir",
    "define": "Definir",
    "design": "Disenar",
    "tools": "Herramientas",
    "memory": "Memoria",
    "estimate": "Estimar",
    "validate": "Validar",
    "package": "Package",
}


def build_product_build_status(
    db: Session,
    *,
    record: SessionRecord,
    product_key: ProductBuildProductKey | str,
    current_user: UserRecord | None = None,
    catalog_stage_override: str | None = None,
) -> ProductBuildStatus:
    meta = _product_meta(product_key)
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    entitlement = _build_entitlement(access.tier, meta, checkout_state=access.checkout_state)
    role = _resolve_role(db, record=record, current_user=current_user)
    stage_val = getattr(record.current_stage, "value", str(record.current_stage or "discover"))
    current_stage = _normalize_catalog_stage(catalog_stage_override or stage_val)
    catalog = build_deliverable_catalog_response(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        role=role,
        tier=access.tier,
        current_stage=current_stage,
    )
    product_items = [item for item in catalog.entries if _is_expected_for_product(item, meta)]
    product_items_by_key = {item.key: item for item in product_items}
    jobs_by_key = _latest_jobs_by_key(db, session_id=record.id)
    diagram_jobs_by_key = _latest_diagram_jobs_by_key(db, session_id=record.id)
    product_jobs_by_key = {item.key: jobs_by_key[item.key] for item in product_items if item.key in jobs_by_key}
    diagram_jobs_by_deliverable_key = {
        item.key: diagram_job
        for item in product_items
        if (diagram_job := _diagram_job_for_item(item, diagram_jobs_by_key)) is not None
    }
    deliverables = [
        _build_deliverable_status(
            item,
            meta,
            job=jobs_by_key.get(item.key),
            diagram_job=diagram_jobs_by_deliverable_key.get(item.key),
            session_id=str(record.id),
        )
        for item in product_items
    ]
    run = _latest_run(db, record=record, product_key=meta.product_key)
    attention_items = _build_attention_items(
        db,
        record=record,
        meta=meta,
        product_items_by_key=product_items_by_key,
        jobs_by_key=product_jobs_by_key,
        diagram_jobs_by_key=diagram_jobs_by_deliverable_key,
        run=run,
    )
    progress = _build_progress(run, deliverables)
    lifecycle = _derive_lifecycle(run, entitlement, deliverables, attention_items)
    current_activity = _build_current_activity(
        db,
        run,
        [*product_jobs_by_key.values(), *diagram_jobs_by_deliverable_key.values()],
        lifecycle,
    )
    actions = _build_actions(meta, lifecycle, entitlement, record_id=str(record.id))
    last_error = _build_last_error(run, deliverables)
    stages = _build_stage_statuses(
        deliverables,
        attention_items,
        record=record,
        run=run,
        overall_progress=progress,
    )

    return ProductBuildStatus(
        workspace_id=record.workspace_id,
        session_id=record.id,
        product_key=meta.product_key,
        product_mode=meta.product_mode,
        product_label=meta.label,
        lifecycle=lifecycle,
        entitlement=entitlement,
        progress=progress,
        current_activity=current_activity,
        stages=stages,
        deliverables=deliverables,
        attention=_summarize_attention(attention_items),
        actions=actions,
        last_error=last_error,
        generated_at=utc_now().isoformat(),
        source_contracts=[
            "commercial-access.v2",
            "deliverable-catalog-response.v1",
            "product-build-runs.v1",
            "uncertainty-backlog.v1",
        ],
    )


def build_all_product_build_statuses(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord | None = None,
) -> list[ProductBuildStatus]:
    return [
        build_product_build_status(db, record=record, product_key=product_key, current_user=current_user)
        for product_key in (
            ProductBuildProductKey.blueprint_basic,
            ProductBuildProductKey.blueprint_pro,
            ProductBuildProductKey.acp,
        )
    ]


def _product_meta(product_key: ProductBuildProductKey | str) -> ProductBuildMeta:
    normalized = product_key if isinstance(product_key, ProductBuildProductKey) else ProductBuildProductKey(str(product_key))
    return PRODUCT_BUILD_META[normalized]


def _is_expected_for_product(item: DeliverableCatalogItem, meta: ProductBuildMeta) -> bool:
    return bool(set(item.product_scope).intersection(meta.included_scopes)) and tier_rank(item.required_tier) <= tier_rank(meta.required_tier)


def _resolve_role(db: Session, *, record: SessionRecord, current_user: UserRecord | None) -> WorkspaceRole:
    if current_user is None or record.workspace_id is None:
        return WorkspaceRole.admin
    return role_for_user(db, workspace_id=record.workspace_id, user_id=current_user.id) or WorkspaceRole.viewer


def _normalize_catalog_stage(value: str) -> str:
    stage = str(value or "").strip().lower()
    if stage in STAGE_LABELS:
        return stage
    legacy_map = {
        "draft_capture": "discover",
        "input_validation": "discover",
        "normalize_discovery": "discover",
        "build_canvas": "define",
        "build_blueprint": "design",
        "post_validation": "validate",
        "ready_for_export": "package",
    }
    return legacy_map.get(stage, "discover")


def _build_entitlement(tier: CommercialTier, meta: ProductBuildMeta, *, checkout_state: str) -> ProductBuildEntitlement:
    allowed = tier_rank(tier) >= tier_rank(meta.required_tier)
    pending_payment = (not allowed) and checkout_state == "pending"
    return ProductBuildEntitlement(
        tier=tier,
        access_state="allowed" if allowed else "payment_pending" if pending_payment else "locked",
        is_purchased=allowed,
        purchase_required=not allowed,
        checkout_href=meta.checkout_href,
        upgrade_label="" if allowed else f"Adquirir {meta.label}",
    )


def _latest_jobs_by_key(db: Session, *, session_id) -> dict[str, DeliverableGenerationJobRecord]:
    jobs = db.exec(
        select(DeliverableGenerationJobRecord)
        .where(DeliverableGenerationJobRecord.session_id == session_id)
        .order_by(DeliverableGenerationJobRecord.updated_at.desc())
    ).all()
    by_key: dict[str, DeliverableGenerationJobRecord] = {}
    for job in jobs:
        by_key.setdefault(job.deliverable_key, job)
    return by_key


def _latest_diagram_jobs_by_key(db: Session, *, session_id) -> dict[str, DiagramGenerationJobRecord]:
    jobs = db.exec(
        select(DiagramGenerationJobRecord)
        .where(DiagramGenerationJobRecord.session_id == session_id)
        .order_by(DiagramGenerationJobRecord.updated_at.desc())
    ).all()
    by_key: dict[str, DiagramGenerationJobRecord] = {}
    for job in jobs:
        by_key.setdefault(job.diagram_key, job)
    return by_key


def _build_deliverable_status(
    item: DeliverableCatalogItem,
    meta: ProductBuildMeta,
    *,
    job: DeliverableGenerationJobRecord | None,
    diagram_job: DiagramGenerationJobRecord | None,
    session_id: str,
) -> ProductBuildDeliverableStatus:
    effective_job = job or diagram_job
    state = _deliverable_state(item, job, diagram_job=diagram_job)
    return ProductBuildDeliverableStatus(
        deliverable_key=item.key,
        title=item.title,
        deliverable_type=item.deliverable_type.value,
        state=state,
        product_surface=meta.product_key,
        stage_key=item.stage,
        required=True,
        job_id=str(effective_job.id) if effective_job is not None else "",
        updated_at=(effective_job.updated_at.isoformat() if effective_job is not None else ""),
        href=f"/projects/{session_id}/blueprint?deliverable={item.key}",
    )


def _deliverable_state(
    item: DeliverableCatalogItem,
    job: DeliverableGenerationJobRecord | None,
    *,
    diagram_job: DiagramGenerationJobRecord | None = None,
) -> ProductBuildDeliverableState:
    effective_job = job or diagram_job
    access_state = str(item.access.access_state or "")
    job_status = str(effective_job.status or "") if effective_job is not None else ""
    if access_state in {"locked", "stage_locked", "disabled"}:
        return ProductBuildDeliverableState.locked
    if access_state == "quality_failed" or job_status in ERROR_JOB_STATES:
        return ProductBuildDeliverableState.error
    if access_state == "available" or job_status == "available":
        return ProductBuildDeliverableState.available
    if access_state == "stale":
        return ProductBuildDeliverableState.stale
    if job_status == "requires_attention":
        return ProductBuildDeliverableState.requires_attention
    if job_status in ACTIVE_JOB_STATES:
        return ProductBuildDeliverableState.generating if job_status in {"generating", "updating", "running"} else ProductBuildDeliverableState.queued
    if access_state in {"preview", "not_generated"}:
        return ProductBuildDeliverableState.pending
    return ProductBuildDeliverableState.pending


def _build_attention_items(
    db: Session,
    *,
    record: SessionRecord,
    meta: ProductBuildMeta,
    product_items_by_key: dict[str, DeliverableCatalogItem],
    jobs_by_key: dict[str, DeliverableGenerationJobRecord],
    diagram_jobs_by_key: dict[str, DiagramGenerationJobRecord],
    run: ProductBuildRunRecord | None,
) -> list[ProductBuildAttentionItem]:
    items: list[ProductBuildAttentionItem] = []
    steps_by_key = _steps_by_key(db, run)
    backlog = db.exec(
        select(UncertaintyBacklogRecord).where(
            UncertaintyBacklogRecord.workspace_id == record.workspace_id,
            UncertaintyBacklogRecord.session_id == record.id,
            UncertaintyBacklogRecord.product_mode.in_(_attention_product_modes(meta)),
            UncertaintyBacklogRecord.status.notin_(list(CLOSED_UNCERTAINTY_STATUSES)),
        )
    ).all()
    for row in backlog:
        blocking = _uncertainty_blocks_product(row, meta)
        linked_step = _step_for_uncertainty(row, steps_by_key)
        items.append(
            ProductBuildAttentionItem(
                key=f"uncertainty:{row.uncertainty_key}",
                title=row.title or row.uncertainty_key,
                severity=ProductBuildAttentionSeverity.blocking if blocking else ProductBuildAttentionSeverity.warning,
                product_key=meta.product_key.value,
                run_id=str(run.id) if run is not None else "",
                step_id=str(linked_step.id) if linked_step is not None else "",
                source="uncertainty_backlog",
                stage_key=row.source_stage,
                deliverable_key=(row.affected_deliverable_keys or [""])[0],
                href=f"/projects/{record.id}/work/{row.source_stage}?attention={row.uncertainty_key}",
                reason=row.reason or row.description,
                blocking=blocking,
            )
        )
    for deliverable_key, job in jobs_by_key.items():
        if job.product_mode != meta.product_mode.value or job.status not in {"error", "failed", "requires_attention"}:
            continue
        linked_step = steps_by_key.get(f"deliverable:{deliverable_key}")
        items.append(
            ProductBuildAttentionItem(
                key=f"deliverable-job:{deliverable_key}",
                title=f"No se pudo generar {deliverable_key}",
                severity=ProductBuildAttentionSeverity.technical_error
                if job.status in ERROR_JOB_STATES
                else ProductBuildAttentionSeverity.blocking,
                product_key=meta.product_key.value,
                run_id=str(run.id) if run is not None else "",
                step_id=str(linked_step.id) if linked_step is not None else "",
                source="deliverable_job",
                deliverable_key=deliverable_key,
                href=f"/projects/{record.id}/blueprint?deliverable={deliverable_key}",
                reason=job.error_message or job.error_code or "El job de generacion requiere revision.",
                blocking=True,
            )
        )
    for deliverable_key, job in diagram_jobs_by_key.items():
        if job.status not in {"error", "failed", "requires_attention"}:
            continue
        item = product_items_by_key.get(deliverable_key)
        linked_step = steps_by_key.get(f"deliverable:{deliverable_key}")
        items.append(
            ProductBuildAttentionItem(
                key=f"diagram-job:{deliverable_key}",
                title=f"No se pudo generar {item.title if item is not None else deliverable_key}",
                severity=ProductBuildAttentionSeverity.technical_error
                if job.status in ERROR_JOB_STATES
                else ProductBuildAttentionSeverity.blocking,
                product_key=meta.product_key.value,
                run_id=str(run.id) if run is not None else "",
                step_id=str(linked_step.id) if linked_step is not None else "",
                source="diagram_job",
                deliverable_key=deliverable_key,
                href=f"/projects/{record.id}/blueprint?deliverable={deliverable_key}",
                reason=job.error_message or job.error_code or "El job de generacion del diagrama requiere revision.",
                blocking=True,
            )
        )
    return items


def _diagram_job_for_item(
    item: DeliverableCatalogItem,
    diagram_jobs_by_key: dict[str, DiagramGenerationJobRecord],
) -> DiagramGenerationJobRecord | None:
    if item.deliverable_type.value != "diagram":
        return None
    return diagram_jobs_by_key.get(item.key.removeprefix("diagram."))


def _steps_by_key(db: Session, run: ProductBuildRunRecord | None) -> dict[str, ProductBuildStepRecord]:
    if run is None:
        return {}
    return {step.step_key: step for step in list_product_build_steps(db, run_id=run.id)}


def _step_for_uncertainty(
    row: UncertaintyBacklogRecord,
    steps_by_key: dict[str, ProductBuildStepRecord],
) -> ProductBuildStepRecord | None:
    premium_key = f"premium_backlog:{row.id}"
    if premium_key in steps_by_key:
        return steps_by_key[premium_key]
    stage = str(row.source_stage or row.target_stage or "").strip().lower()
    if stage and f"acp_dependency:{stage}" in steps_by_key:
        return steps_by_key[f"acp_dependency:{stage}"]
    for deliverable_key in row.affected_deliverable_keys or []:
        step = steps_by_key.get(f"deliverable:{deliverable_key}")
        if step is not None:
            return step
    return None


def _attention_product_modes(meta: ProductBuildMeta) -> list[str]:
    if meta.product_key == ProductBuildProductKey.blueprint_pro:
        return [
            ProductProcessingMode.basic_free.value,
            ProductProcessingMode.premium_enrichment.value,
        ]
    if meta.product_key == ProductBuildProductKey.acp:
        return [
            ProductProcessingMode.basic_free.value,
            ProductProcessingMode.premium_enrichment.value,
            ProductProcessingMode.acp_implementation.value,
        ]
    return [meta.product_mode.value]


def _uncertainty_blocks_product(row: UncertaintyBacklogRecord, meta: ProductBuildMeta) -> bool:
    if row.status not in {
        UncertaintyBacklogStatus.open.value,
        UncertaintyBacklogStatus.in_progress.value,
        UncertaintyBacklogStatus.deferred.value,
    }:
        return False
    if meta.product_key == ProductBuildProductKey.blueprint_basic:
        return row.disposition == "block"
    if meta.product_key == ProductBuildProductKey.blueprint_pro:
        if _is_deferred_to_acp(row):
            return False
        return row.disposition in {"block", "resolve_now", "defer"}
    return row.disposition in {"block", "resolve_now"} or (
        row.product_mode == ProductProcessingMode.acp_implementation.value
        and row.status in {UncertaintyBacklogStatus.open.value, UncertaintyBacklogStatus.in_progress.value}
    )


def _is_deferred_to_acp(row: UncertaintyBacklogRecord) -> bool:
    target = str(row.target_stage or "").strip().lower()
    return row.disposition == "defer" and target in {"acp", "package", "implementation", "implementacion", "deployment", "despliegue"}


def _latest_run(
    db: Session,
    *,
    record: SessionRecord,
    product_key: ProductBuildProductKey,
) -> ProductBuildRunRecord | None:
    runs = list_product_build_runs(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        product_key=product_key,
    )
    return runs[0] if runs else None


def _build_progress(
    run: ProductBuildRunRecord | None,
    deliverables: list[ProductBuildDeliverableStatus],
) -> ProductBuildProgress:
    if run is not None and run.total_units > 0:
        return ProductBuildProgress(
            percent=run.progress_percent,
            completed_units=run.completed_units,
            total_units=run.total_units,
            blocked_units=run.blocked_units,
            calculation="manual",
            label="Progreso persistido del build.",
        )
    completed = sum(1 for item in deliverables if item.state == ProductBuildDeliverableState.available)
    blocked = sum(1 for item in deliverables if item.state in {ProductBuildDeliverableState.error, ProductBuildDeliverableState.requires_attention})
    total = len(deliverables)
    return ProductBuildProgress(
        percent=calculate_product_build_percent(completed, total),
        completed_units=float(completed),
        total_units=float(total),
        blocked_units=float(blocked),
        calculation="weighted_units",
        label="Calculado desde catalogo y versiones disponibles.",
    )


def _derive_lifecycle(
    run: ProductBuildRunRecord | None,
    entitlement: ProductBuildEntitlement,
    deliverables: list[ProductBuildDeliverableStatus],
    attention_items: list[ProductBuildAttentionItem],
) -> ProductBuildLifecycle:
    if entitlement.access_state == "payment_pending":
        return ProductBuildLifecycle.payment_pending
    if entitlement.access_state == "locked":
        return ProductBuildLifecycle.not_purchased
    if run is not None and run.lifecycle in {
        ProductBuildLifecycle.queued.value,
        ProductBuildLifecycle.preparing.value,
        ProductBuildLifecycle.running.value,
        ProductBuildLifecycle.requires_attention.value,
        ProductBuildLifecycle.error.value,
        ProductBuildLifecycle.completed.value,
    }:
        return ProductBuildLifecycle(run.lifecycle)
    if any(item.blocking for item in attention_items):
        return ProductBuildLifecycle.requires_attention
    if any(item.state in {ProductBuildDeliverableState.error, ProductBuildDeliverableState.requires_attention} for item in deliverables):
        return ProductBuildLifecycle.requires_attention
    if any(item.state in {ProductBuildDeliverableState.queued, ProductBuildDeliverableState.generating} for item in deliverables):
        return ProductBuildLifecycle.running
    if deliverables and all(item.state == ProductBuildDeliverableState.available for item in deliverables):
        return ProductBuildLifecycle.completed
    if any(item.state == ProductBuildDeliverableState.available for item in deliverables):
        return ProductBuildLifecycle.partial
    return ProductBuildLifecycle.ready_to_start


def _build_current_activity(
    db: Session,
    run: ProductBuildRunRecord | None,
    jobs: Iterable[DeliverableGenerationJobRecord | DiagramGenerationJobRecord],
    lifecycle: ProductBuildLifecycle,
) -> ProductBuildCurrentActivity | None:
    active_job = next((job for job in jobs if job.status in ACTIVE_JOB_STATES), None)
    if active_job is not None:
        if isinstance(active_job, DiagramGenerationJobRecord):
            reference_key = f"diagram.{active_job.diagram_key}"
            label = "Generando diagrama"
        else:
            reference_key = active_job.deliverable_key
            label = "Generando entregable"
        return ProductBuildCurrentActivity(
            activity_key=f"deliverable:{reference_key}",
            label=label,
            detail=reference_key,
            step_key=reference_key,
            status="running" if active_job.status in {"generating", "updating", "running"} else "queued",
            started_at=active_job.started_at.isoformat() if active_job.started_at else "",
            updated_at=active_job.updated_at.isoformat(),
        )
    if run is None:
        return None
    steps = list_product_build_steps(db, run_id=run.id)
    active_step = next((step for step in steps if step.status in {"queued", "running", "generating"}), None)
    if active_step is None:
        return None
    return ProductBuildCurrentActivity(
        activity_key=active_step.step_key,
        label="Procesando producto",
        detail=active_step.deliverable_key or active_step.dependency_key,
        step_key=active_step.step_key,
        status="running" if lifecycle == ProductBuildLifecycle.running else "queued",
        started_at=active_step.started_at.isoformat() if active_step.started_at else "",
        updated_at=active_step.updated_at.isoformat(),
    )


def _build_actions(
    meta: ProductBuildMeta,
    lifecycle: ProductBuildLifecycle,
    entitlement: ProductBuildEntitlement,
    *,
    record_id: str,
) -> list[ProductBuildAction]:
    if entitlement.purchase_required:
        return [
            ProductBuildAction(
                action_key="buy_product",
                label=entitlement.upgrade_label,
                state=ProductBuildActionState.recommended,
                href=entitlement.checkout_href,
                reason="Producto no adquirido.",
                primary=True,
            )
        ]
    if lifecycle == ProductBuildLifecycle.requires_attention:
        return [
            ProductBuildAction(
                action_key="open_attention",
                label="Abrir Atencion",
                state=ProductBuildActionState.recommended,
                href=f"/projects/{record_id}/attention",
                reason="Hay items que requieren revision antes de cerrar el producto.",
                primary=True,
            )
        ]
    if lifecycle in {ProductBuildLifecycle.running, ProductBuildLifecycle.queued, ProductBuildLifecycle.preparing}:
        return [
            ProductBuildAction(
                action_key="view_progress",
                label="Ver progreso",
                state=ProductBuildActionState.running,
                href=f"/projects/{record_id}/activity",
                reason="El producto se esta construyendo.",
                primary=True,
            )
        ]
    return [
        ProductBuildAction(
            action_key="open_product",
            label=f"Ver {meta.label}",
            state=ProductBuildActionState.available if lifecycle != ProductBuildLifecycle.ready_to_start else ProductBuildActionState.recommended,
            href=_product_href(record_id, meta.product_key),
            reason="Abrir experiencia del producto.",
            primary=True,
        )
    ]


def _product_href(record_id: str, product_key: ProductBuildProductKey) -> str:
    if product_key == ProductBuildProductKey.blueprint_pro:
        return f"/projects/{record_id}/blueprint/pro"
    if product_key == ProductBuildProductKey.acp:
        return f"/projects/{record_id}/acp"
    return f"/projects/{record_id}/blueprint"


def _build_last_error(
    run: ProductBuildRunRecord | None,
    deliverables: list[ProductBuildDeliverableStatus],
) -> ProductBuildRecoverableError | None:
    if run is not None and run.error_payload:
        return ProductBuildRecoverableError(
            code=str(run.error_payload.get("code") or "product_build_error"),
            title=str(run.error_payload.get("title") or "El build requiere recuperacion"),
            message=str(run.error_payload.get("message") or "Revisa el ultimo error antes de reintentar."),
            technical_message=str(run.error_payload.get("technical_message") or ""),
            retry_action_key=str(run.error_payload.get("retry_action_key") or "retry_product_build"),
            trace_refs=[str(ref) for ref in run.error_payload.get("trace_refs", [])],
        )
    failed = next((item for item in deliverables if item.state == ProductBuildDeliverableState.error), None)
    if failed is None:
        return None
    return ProductBuildRecoverableError(
        code="deliverable_generation_error",
        title=f"No se pudo generar {failed.title}",
        message="Un entregable fallo o no paso validacion. Revisa Atencion antes de cerrar el producto.",
        retry_action_key="retry_deliverable_generation",
        trace_refs=[failed.deliverable_key],
    )


STAGE_FLOW_ORDER = (
    "discover",
    "define",
    "design",
    "tools",
    "memory",
    "estimate",
    "validate",
    "package",
)


def _build_stage_statuses(
    deliverables: list[ProductBuildDeliverableStatus],
    attention_items: list[ProductBuildAttentionItem],
    *,
    record: SessionRecord | None = None,
    run: ProductBuildRunRecord | None = None,
    overall_progress: ProductBuildProgress | None = None,
) -> list[ProductBuildStageStatus]:
    stage_keys = list(STAGE_LABELS)
    stage_val = getattr(record.current_stage, "value", str(record.current_stage or "package")) if record else "package"
    current_stage_normalized = _normalize_catalog_stage(stage_val)
    current_stage_idx = (
        STAGE_FLOW_ORDER.index(current_stage_normalized)
        if current_stage_normalized in STAGE_FLOW_ORDER
        else len(STAGE_FLOW_ORDER) - 1
    )
    is_product_completed = (
        (run is not None and run.lifecycle == "completed")
        or (overall_progress is not None and overall_progress.percent >= 100)
    )

    statuses: list[ProductBuildStageStatus] = []
    for idx, stage_key in enumerate(stage_keys):
        stage_deliverables = [item for item in deliverables if item.stage_key == stage_key]
        stage_attention = [item for item in attention_items if item.stage_key == stage_key]
        completed = sum(1 for item in stage_deliverables if item.state == ProductBuildDeliverableState.available)
        total = len(stage_deliverables)
        has_blocking_attention = any(item.blocking for item in stage_attention)

        is_preceding_or_current = idx <= current_stage_idx or is_product_completed
        if has_blocking_attention:
            lifecycle = ProductBuildLifecycle.requires_attention
            progress_percent = calculate_product_build_percent(completed, total) if total else 50.0
            label = "La etapa requiere atención antes de cerrar el producto."
        elif is_product_completed or (total > 0 and completed == total) or (is_preceding_or_current and (completed > 0 or total == 0)):
            lifecycle = ProductBuildLifecycle.completed
            progress_percent = 100.0
            label = "Etapa validada y completada en el viaje Lean."
        elif completed > 0 or is_preceding_or_current:
            lifecycle = ProductBuildLifecycle.partial
            progress_percent = max(calculate_product_build_percent(completed, total), 75.0 if is_preceding_or_current else 50.0)
            label = "Etapa avanzada con entregables en curso."
        else:
            lifecycle = ProductBuildLifecycle.ready_to_start
            progress_percent = 0.0
            label = "Etapa lista para iniciar."

        statuses.append(
            ProductBuildStageStatus(
                stage_key=stage_key,
                label=STAGE_LABELS[stage_key],
                lifecycle=lifecycle,
                progress=ProductBuildProgress(
                    percent=progress_percent,
                    completed_units=float(completed if total > 0 else (1 if lifecycle == ProductBuildLifecycle.completed else 0)),
                    total_units=float(total if total > 0 else 1),
                    blocked_units=float(sum(1 for item in stage_deliverables if item.state in {ProductBuildDeliverableState.error, ProductBuildDeliverableState.requires_attention})),
                    label=label,
                ),
                blocker_count=sum(1 for item in stage_attention if item.blocking),
                deliverable_count=total if total > 0 else 1,
            )
        )
    return statuses


def _summarize_attention(items: list[ProductBuildAttentionItem]) -> ProductBuildAttentionSummary:
    return ProductBuildAttentionSummary(
        total=len(items),
        blocking_count=sum(1 for item in items if item.blocking),
        warning_count=sum(1 for item in items if item.severity == ProductBuildAttentionSeverity.warning),
        technical_error_count=sum(1 for item in items if item.severity == ProductBuildAttentionSeverity.technical_error),
        items=items[:10],
    )
