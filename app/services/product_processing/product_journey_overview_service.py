from __future__ import annotations

from collections import Counter
from typing import Iterable

from sqlmodel import Session

from app.models import SessionRecord, UserRecord, utc_now
from app.services.product_processing.contracts import (
    ProductBuildAction,
    ProductBuildActionState,
    ProductBuildCurrentActivity,
    ProductBuildDeliverableState,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductBuildStageStatus,
    ProductBuildStatus,
    ProductJourneyCurrentStage,
    ProductJourneyDeliverableSummary,
    ProductJourneyOutcome,
    ProductJourneyOverview,
    ProductJourneyProductSummary,
    ProductJourneyRecommendedAction,
)
from app.services.product_processing.product_build_status_service import build_all_product_build_statuses


PRODUCT_ORDER: tuple[ProductBuildProductKey, ...] = (
    ProductBuildProductKey.blueprint_basic,
    ProductBuildProductKey.blueprint_pro,
    ProductBuildProductKey.acp,
)

ACTIVE_LIFECYCLES = {
    ProductBuildLifecycle.queued,
    ProductBuildLifecycle.preparing,
    ProductBuildLifecycle.running,
}

OPEN_LIFECYCLES = {
    ProductBuildLifecycle.ready_to_start,
    ProductBuildLifecycle.partial,
    ProductBuildLifecycle.error,
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


def build_product_journey_overview(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord | None = None,
) -> ProductJourneyOverview:
    statuses = build_all_product_build_statuses(db, record=record, current_user=current_user)
    products = [_summarize_product(status) for status in statuses]
    recommended_next_action = _select_recommended_next_action(statuses)
    attention_items = _unique_attention_items(statuses)

    return ProductJourneyOverview(
        workspace_id=record.workspace_id,
        session_id=record.id,
        project_title=record.title,
        current_stage=_select_current_stage(statuses, record),
        achieved_outcomes=_build_achieved_outcomes(statuses),
        active_operation=_select_active_operation(statuses),
        blocking_attention_count=sum(1 for item in attention_items if item.blocking),
        warning_attention_count=sum(1 for item in attention_items if item.severity.value == "warning"),
        technical_error_count=sum(1 for item in attention_items if item.severity.value == "technical_error"),
        recommended_next_action=recommended_next_action,
        products=products,
        deliverable_summary=_summarize_deliverables(statuses),
        generated_at=utc_now().isoformat(),
        source_contracts=_source_contracts(statuses),
    )


def _summarize_product(status: ProductBuildStatus) -> ProductJourneyProductSummary:
    available = sum(1 for item in status.deliverables if item.state == ProductBuildDeliverableState.available)
    return ProductJourneyProductSummary(
        product_key=status.product_key,
        product_label=status.product_label,
        lifecycle=status.lifecycle,
        access_state=status.entitlement.access_state,
        is_purchased=status.entitlement.is_purchased,
        purchase_required=status.entitlement.purchase_required,
        progress_percent=status.progress.percent,
        available_deliverable_count=available,
        total_deliverable_count=len(status.deliverables),
        blocking_attention_count=status.attention.blocking_count,
        warning_attention_count=status.attention.warning_count,
        technical_error_count=status.attention.technical_error_count,
        active_operation=status.current_activity,
        primary_action=_primary_action_for(status),
    )


def _unique_attention_items(statuses: list[ProductBuildStatus]):
    items_by_key = {}
    for status in _ordered(statuses):
        for item in status.attention.items:
            items_by_key.setdefault(item.key, item)
    return list(items_by_key.values())


def _primary_action_for(status: ProductBuildStatus) -> ProductJourneyRecommendedAction | None:
    primary = next((action for action in status.actions if action.primary), None)
    if primary is None:
        return None
    return _journey_action(status, primary)


def _journey_action(status: ProductBuildStatus, action: ProductBuildAction) -> ProductJourneyRecommendedAction:
    return ProductJourneyRecommendedAction(
        action_key=action.action_key,
        label=action.label,
        state=action.state,
        href=action.href,
        reason=action.reason,
        product_key=status.product_key,
        primary=action.primary,
    )


def _select_recommended_next_action(statuses: list[ProductBuildStatus]) -> ProductJourneyRecommendedAction | None:
    by_key = {status.product_key: status for status in statuses}

    for status in _ordered(statuses):
        if status.attention.blocking_count > 0:
            action = next((item for item in status.actions if item.action_key == "open_attention"), None)
            if action is not None:
                return _journey_action(status, action)
            return ProductJourneyRecommendedAction(
                action_key="open_attention",
                label="Abrir Atencion",
                state=ProductBuildActionState.recommended,
                href=f"/projects/{status.session_id}/attention",
                reason="Hay bloqueos que requieren intervencion del usuario.",
                product_key=status.product_key,
            )

    for status in _ordered(statuses):
        if status.lifecycle in ACTIVE_LIFECYCLES:
            primary = _primary_action_for(status)
            if primary is not None:
                return primary

    blueprint = by_key.get(ProductBuildProductKey.blueprint_basic)
    if blueprint is not None and not blueprint.entitlement.purchase_required and blueprint.lifecycle != ProductBuildLifecycle.completed:
        primary = _primary_action_for(blueprint)
        if primary is not None:
            return primary

    pro = by_key.get(ProductBuildProductKey.blueprint_pro)
    if pro is not None:
        if pro.entitlement.purchase_required and _is_complete_or_useful(blueprint):
            primary = _primary_action_for(pro)
            if primary is not None:
                return primary
        if not pro.entitlement.purchase_required and pro.lifecycle != ProductBuildLifecycle.completed:
            primary = _primary_action_for(pro)
            if primary is not None:
                return primary

    acp = by_key.get(ProductBuildProductKey.acp)
    if acp is not None:
        if acp.entitlement.purchase_required and _is_complete_or_useful(pro):
            primary = _primary_action_for(acp)
            if primary is not None:
                return primary
        if not acp.entitlement.purchase_required and acp.lifecycle != ProductBuildLifecycle.completed:
            primary = _primary_action_for(acp)
            if primary is not None:
                return primary

    for status in _ordered(statuses):
        primary = _primary_action_for(status)
        if primary is not None:
            return primary
    return None


def _is_complete_or_useful(status: ProductBuildStatus | None) -> bool:
    if status is None:
        return False
    return status.lifecycle in {ProductBuildLifecycle.completed, ProductBuildLifecycle.partial} or status.progress.percent > 0


def _select_current_stage(statuses: list[ProductBuildStatus], record: SessionRecord) -> ProductJourneyCurrentStage:
    for status in _ordered(statuses):
        blocked_stage = next((stage for stage in status.stages if stage.lifecycle == ProductBuildLifecycle.requires_attention), None)
        if blocked_stage is not None:
            return _stage_to_current(status, blocked_stage)

    for status in _ordered(statuses):
        if status.lifecycle in ACTIVE_LIFECYCLES:
            open_stage = _first_open_stage(status)
            if open_stage is not None:
                return _stage_to_current(status, open_stage)

    for status in _ordered(statuses):
        if not status.entitlement.purchase_required and status.lifecycle != ProductBuildLifecycle.completed:
            open_stage = _first_open_stage(status)
            if open_stage is not None:
                return _stage_to_current(status, open_stage)

    normalized = _normalize_stage_key(getattr(record.current_stage, "value", str(record.current_stage)))
    return ProductJourneyCurrentStage(
        stage_key=normalized,
        label=STAGE_LABELS.get(normalized, normalized.title()),
        lifecycle=ProductBuildLifecycle.completed,
        progress_percent=100 if statuses and all(status.lifecycle == ProductBuildLifecycle.completed for status in statuses if not status.entitlement.purchase_required) else 0,
        product_key=ProductBuildProductKey.blueprint_basic,
    )


def _first_open_stage(status: ProductBuildStatus) -> ProductBuildStageStatus | None:
    return next(
        (
            stage
            for stage in status.stages
            if stage.lifecycle in OPEN_LIFECYCLES
            or stage.lifecycle in ACTIVE_LIFECYCLES
            or stage.lifecycle == ProductBuildLifecycle.requires_attention
        ),
        None,
    )


def _stage_to_current(status: ProductBuildStatus, stage: ProductBuildStageStatus) -> ProductJourneyCurrentStage:
    return ProductJourneyCurrentStage(
        stage_key=stage.stage_key,
        label=stage.label,
        lifecycle=stage.lifecycle,
        progress_percent=stage.progress.percent,
        product_key=status.product_key,
    )


def _normalize_stage_key(value: str) -> str:
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


def _select_active_operation(statuses: list[ProductBuildStatus]) -> ProductBuildCurrentActivity | None:
    for status in _ordered(statuses):
        if status.current_activity is not None and status.lifecycle in ACTIVE_LIFECYCLES:
            return status.current_activity
    return None


def _build_achieved_outcomes(statuses: list[ProductBuildStatus]) -> list[ProductJourneyOutcome]:
    outcomes: list[ProductJourneyOutcome] = []
    for status in _ordered(statuses):
        if status.lifecycle == ProductBuildLifecycle.completed:
            outcomes.append(
                ProductJourneyOutcome(
                    key=f"product:{status.product_key.value}:completed",
                    title=f"{status.product_label} listo",
                    detail="El producto alcanzo estado completado segun runs/entregables.",
                    product_key=status.product_key,
                    href=_product_href(str(status.session_id), status.product_key),
                )
            )
        for deliverable in status.deliverables:
            if deliverable.state != ProductBuildDeliverableState.available:
                continue
            outcomes.append(
                ProductJourneyOutcome(
                    key=f"deliverable:{deliverable.deliverable_key}",
                    title=deliverable.title,
                    detail="Entregable disponible.",
                    product_key=status.product_key,
                    stage_key=deliverable.stage_key,
                    href=deliverable.href,
                )
            )
    return outcomes[:12]


def _summarize_deliverables(statuses: Iterable[ProductBuildStatus]) -> ProductJourneyDeliverableSummary:
    counter: Counter[str] = Counter()
    for status in statuses:
        for item in status.deliverables:
            counter[item.state.value] += 1
    return ProductJourneyDeliverableSummary(
        total_count=sum(counter.values()),
        available_count=counter[ProductBuildDeliverableState.available.value],
        pending_count=counter[ProductBuildDeliverableState.pending.value],
        running_count=counter[ProductBuildDeliverableState.queued.value]
        + counter[ProductBuildDeliverableState.generating.value],
        locked_count=counter[ProductBuildDeliverableState.locked.value],
        stale_count=counter[ProductBuildDeliverableState.stale.value],
        attention_count=counter[ProductBuildDeliverableState.requires_attention.value],
        error_count=counter[ProductBuildDeliverableState.error.value],
    )


def _source_contracts(statuses: Iterable[ProductBuildStatus]) -> list[str]:
    contracts = {"product-journey-overview.v2", "product-build-status.v1"}
    for status in statuses:
        contracts.update(status.source_contracts)
    return sorted(contracts)


def _ordered(statuses: list[ProductBuildStatus]) -> list[ProductBuildStatus]:
    order = {key: index for index, key in enumerate(PRODUCT_ORDER)}
    return sorted(statuses, key=lambda status: order.get(status.product_key, len(order)))


def _product_href(record_id: str, product_key: ProductBuildProductKey) -> str:
    if product_key == ProductBuildProductKey.blueprint_pro:
        return f"/projects/{record_id}/blueprint/pro"
    if product_key == ProductBuildProductKey.acp:
        return f"/projects/{record_id}/acp"
    return f"/projects/{record_id}/blueprint"
