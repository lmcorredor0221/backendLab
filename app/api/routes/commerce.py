from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.api.routes.sessions import build_snapshot, sync_short_term_memory_checkpoint
from app.db import get_session
from app.models import (
    AccessRequestCreateRequest,
    AccessRequestResolveRequest,
    AccessRequestResponse,
    AcpInvitationResponse,
    BasePricesResponse,
    BasePricesUpdateRequest,
    BlueprintResultResponse,
    CommercialAccessSnapshotV2,
    CommercialAccessRequestRecord,
    CommercialCheckoutCompletionRequest,
    CommercialCheckoutSessionRequest,
    CommercialCheckoutSessionResponse,
    CommercialOrderRecord,
    CommercialOrderResponse,
    CommercialTier,
    HotmartPendingActivationBootstrapRequest,
    HotmartPendingActivationBootstrapResponse,
    HotmartPendingActivationClaimRequest,
    HotmartPendingActivationPublicResponse,
    HotmartPendingActivationResponse,
    ProductCatalogResponse,
    ProductOfferResponse,
    ProductOverviewItem,
    ProductOverviewResponse,
    ProductAttentionItem,
    SessionRecord,
    TRMResponse,
    UserRecord,
    WorkspaceRole,
)
from app.services.acp_generator import generate_acp_preview
from app.services.auth_service import get_current_user
from app.services.commerce_service import (
    TRM_SOURCE_LABEL,
    build_access_request_response,
    build_order_response,
    build_product_response,
    complete_sandbox_checkout,
    create_checkout_session,
    get_access_requests_count,
    get_active_product,
    get_base_prices_summary,
    get_today_trm_data,
    list_active_products,
    list_all_access_requests,
    request_access,
    resolve_access_request,
    resolve_effective_entitlement_state,
    role_for_user,
    tier_rank,
    update_base_prices_usd,
)
from app.services.commercial_access import build_commercial_access_snapshot_v2, build_entitlement_context
from app.services.diagram_catalog_service import build_diagram_catalog
from app.services.hotmart.pending_activations import (
    claim_hotmart_pending_activation,
    get_hotmart_pending_activation_public,
    get_hotmart_pending_activation_record,
    list_user_pending_hotmart_activations,
)
from app.services.operations_service import capture_operational_state
from app.services.product_processing import (
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductJourneyOverview,
    ProductJourneyProductSummary,
    build_product_journey_overview,
)
from app.services.runtime_access_control import is_platform_admin
from app.services.workspace_membership_service import get_effective_workspace_membership
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context
from app.services.workspace_bootstrap import apply_workspace_bootstrap
from app.core.config import get_settings


router = APIRouter(tags=["commerce"])


def _get_record_or_404(db: Session, session_id: UUID, user_id: UUID) -> SessionRecord:
    record = db.get(SessionRecord, session_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    membership = get_effective_workspace_membership(
        db,
        workspace_id=record.workspace_id,
        user_id=user_id,
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return record


def _route_item(key: str, label: str, href: str, *, detail: str = "", access_state: str = "allowed") -> ProductOverviewItem:
    return ProductOverviewItem(
        key=key,
        label=label,
        href=href,
        detail=detail,
        access_state=access_state,
        cta_label="Abrir",
        status="available" if access_state == "allowed" else "locked",
    )


def _blocked_attention(key: str, title: str, reason: str, href: str) -> ProductAttentionItem:
    return ProductAttentionItem(key=key, title=title, severity="blocking", reason=reason, href=href)


def _create_hotmart_activation_session(
    db: Session,
    *,
    current_user: UserRecord,
    workspace_context: WorkspaceAccessContext,
) -> SessionRecord:
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    record = SessionRecord(user_id=current_user.id, workspace_id=workspace_context.workspace.id)
    db.add(record)
    db.flush()
    capture_operational_state(db, session_id=record.id, source_action="create_session")
    sync_short_term_memory_checkpoint(db, record=record, source_action="create_session")
    return record


def _session_work_redirect_path(record: SessionRecord) -> str:
    return f"/projects/{record.id}/discover"


def _session_product_redirect_path(record: SessionRecord) -> str:
    if record.commercial_tier == CommercialTier.acp:
        return f"/projects/{record.id}/acp"
    if record.commercial_tier == CommercialTier.blueprint_pro:
        return f"/projects/{record.id}/blueprint/pro"
    return f"/projects/{record.id}/blueprint"


def _legacy_product_key(product_key: ProductBuildProductKey) -> str:
    if product_key == ProductBuildProductKey.blueprint_basic:
        return "blueprint"
    return product_key.value


def _legacy_product_status(product: ProductJourneyProductSummary) -> str:
    if product.purchase_required:
        return "locked"
    if product.lifecycle == ProductBuildLifecycle.completed:
        return "generated"
    if product.lifecycle in {
        ProductBuildLifecycle.queued,
        ProductBuildLifecycle.preparing,
        ProductBuildLifecycle.running,
    }:
        return "running"
    if product.lifecycle == ProductBuildLifecycle.requires_attention:
        return "requires_attention"
    if product.lifecycle == ProductBuildLifecycle.partial:
        return "partial"
    if product.lifecycle == ProductBuildLifecycle.error:
        return "error"
    return "available"


def _legacy_product_access_state(product: ProductJourneyProductSummary) -> str:
    if product.access_state == "allowed":
        return "allowed"
    if product.access_state == "payment_pending":
        return "payment_pending"
    return "requires_purchase" if product.purchase_required else product.access_state


def _legacy_product_cta(product: ProductJourneyProductSummary) -> str:
    if product.primary_action is not None and product.primary_action.label:
        return product.primary_action.label
    if product.purchase_required:
        return f"Adquirir {product.product_label}"
    return f"Ver {product.product_label}"


def _legacy_product_href(product: ProductJourneyProductSummary, base: str) -> str:
    if product.primary_action is not None and product.primary_action.href:
        return product.primary_action.href
    if product.product_key == ProductBuildProductKey.blueprint_pro:
        return f"{base}/blueprint/pro"
    if product.product_key == ProductBuildProductKey.acp:
        return f"{base}/acp"
    return f"{base}/blueprint"


def _legacy_products(overview: ProductJourneyOverview, base: str) -> list[ProductOverviewItem]:
    items: list[ProductOverviewItem] = []
    for product in overview.products:
        items.append(
            ProductOverviewItem(
                key=_legacy_product_key(product.product_key),
                label=product.product_label,
                href=_legacy_product_href(product, base),
                status=_legacy_product_status(product),
                access_state=_legacy_product_access_state(product),
                cta_label=_legacy_product_cta(product),
                progress_percent=product.progress_percent,
                detail=(
                    f"{product.available_deliverable_count}/{product.total_deliverable_count} entregables disponibles. "
                    f"Estado: {product.lifecycle.value}."
                ),
            )
        )
    return items


def _legacy_attention(overview: ProductJourneyOverview) -> list[ProductAttentionItem]:
    attention: list[ProductAttentionItem] = []
    if overview.blocking_attention_count and overview.recommended_next_action is not None:
        attention.append(
            _blocked_attention(
                "canonical_blocking_attention",
                "Hay decisiones pendientes",
                overview.recommended_next_action.reason or "El journey requiere intervencion antes de avanzar.",
                overview.recommended_next_action.href,
            )
        )
    if overview.technical_error_count and not attention:
        attention.append(
            _blocked_attention(
                "canonical_technical_error",
                "Hay un error recuperable",
                "Revisa la actividad o atencion antes de continuar.",
                overview.recommended_next_action.href if overview.recommended_next_action is not None else "",
            )
        )
    return attention


def _product_overview(db: Session, record: SessionRecord, current_user: UserRecord) -> ProductOverviewResponse:
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    overview = build_product_journey_overview(db, record=record, current_user=current_user)
    base = f"/projects/{record.id}"
    attention = _legacy_attention(overview)
    if access.checkout_state == "pending":
        attention.append(
            _blocked_attention(
                "checkout_pending",
                "Checkout pendiente",
                "Hay una orden pendiente antes de activar el producto.",
                f"{base}/blueprint/pro",
            )
        )
    if access.tier != CommercialTier.acp and not overview.blocking_attention_count:
        attention.append(
            ProductAttentionItem(
                key="acp_upsell",
                title="ACP disponible como siguiente nivel",
                severity="info",
                reason="El Blueprint ya puede convertirse en paquete tecnico portable al adquirir ACP.",
                href=f"{base}/acp",
            )
        )
    return ProductOverviewResponse(
        session_id=record.id,
        workspace_id=record.workspace_id,
        project_title=record.title,
        active_stage=overview.current_stage.stage_key,
        lean_progress_percent=overview.current_stage.progress_percent,
        access=access,
        products=_legacy_products(overview, base),
        attention=attention,
        exports=[
            _route_item("artifacts", "Artefactos", f"{base}/artifacts", detail="Centro de exportaciones autorizado."),
            _route_item("diagrams", "Diagramas", f"{base}/diagrams", detail="Catalogo completo con upsell por policy."),
        ],
        navigation=[
            _route_item("work", "Trabajo", f"{base}/work/discover", detail="Etapas LEAN y accion principal."),
            _route_item("blueprint", "Blueprint", f"{base}/blueprint"),
            _route_item("acp", "ACP", f"{base}/acp", access_state="allowed" if access.tier == CommercialTier.acp else "requires_purchase"),
            _route_item("attention", "Atencion", f"{base}/attention"),
            _route_item("activity", "Actividad", f"{base}/activity"),
        ],
        canonical_overview_contract=overview.contract_version,
        recommended_next_action=overview.recommended_next_action.model_dump(mode="json")
        if overview.recommended_next_action is not None
        else {},
        source_contracts=overview.source_contracts,
    )


@router.get("/commerce/products", response_model=list[ProductCatalogResponse])
@router.get("/commerce/catalog", response_model=list[ProductCatalogResponse])
def list_products_route(db: Session = Depends(get_session)) -> list[ProductCatalogResponse]:
    return list_active_products(db)


@router.post("/commerce/checkout-sessions", response_model=CommercialCheckoutSessionResponse)
def create_checkout_session_route(
    payload: CommercialCheckoutSessionRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> CommercialCheckoutSessionResponse:
    record = _get_record_or_404(db, payload.session_id, current_user.id)
    try:
        response = create_checkout_session(
            db,
            payload=payload,
            record=record,
            current_user=current_user,
            base_url=get_settings().commerce_public_base_url,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.post("/commerce/checkout-sessions/{checkout_ref}/sandbox-complete", response_model=CommercialOrderResponse)
def complete_sandbox_checkout_route(
    checkout_ref: str,
    payload: CommercialCheckoutCompletionRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> CommercialOrderResponse:
    try:
        response = complete_sandbox_checkout(db, checkout_ref=checkout_ref, payload=payload, current_user=current_user)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/commerce/orders/{order_id}", response_model=CommercialOrderResponse)
def get_order_route(
    order_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> CommercialOrderResponse:
    order = db.get(CommercialOrderRecord, order_id)
    if order is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    role = role_for_user(db, workspace_id=order.workspace_id, user_id=current_user.id)
    if role is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    return build_order_response(db, order)


@router.get("/commerce/hotmart/pending-activations", response_model=list[HotmartPendingActivationResponse])
def list_hotmart_pending_activations_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[HotmartPendingActivationResponse]:
    return list_user_pending_hotmart_activations(db, current_user=current_user)


@router.get(
    "/commerce/hotmart/pending-activations/{activation_token}/public",
    response_model=HotmartPendingActivationPublicResponse,
)
def get_hotmart_pending_activation_public_route(
    activation_token: str,
    db: Session = Depends(get_session),
) -> HotmartPendingActivationPublicResponse:
    try:
        return get_hotmart_pending_activation_public(db, activation_token=activation_token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/commerce/hotmart/pending-activations/{activation_token}/claim",
    response_model=HotmartPendingActivationResponse,
)
def claim_hotmart_pending_activation_route(
    activation_token: str,
    payload: HotmartPendingActivationClaimRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> HotmartPendingActivationResponse:
    try:
        response = claim_hotmart_pending_activation(
            db,
            activation_token=activation_token,
            payload=payload,
            current_user=current_user,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.post(
    "/commerce/hotmart/pending-activations/{activation_token}/bootstrap",
    response_model=HotmartPendingActivationBootstrapResponse,
)
def bootstrap_hotmart_pending_activation_route(
    activation_token: str,
    payload: HotmartPendingActivationBootstrapRequest | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartPendingActivationBootstrapResponse:
    try:
        pending_record = get_hotmart_pending_activation_record(db, activation_token=activation_token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    platform_admin = is_platform_admin(db, current_user)
    buyer_email = pending_record.buyer_email.strip().lower()
    if buyer_email and buyer_email != current_user.email.strip().lower() and not platform_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This Hotmart purchase belongs to another buyer email.",
        )

    created_session = False
    if pending_record.claimed_session_id is not None:
        if pending_record.claimed_by_user_id not in {None, current_user.id} and not platform_admin:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This Hotmart purchase was already activated by another user.",
            )
        target_session = _get_record_or_404(db, pending_record.claimed_session_id, current_user.id)
    elif payload is not None and payload.session_id is not None:
        target_session = _get_record_or_404(db, payload.session_id, current_user.id)
    else:
        target_session = _create_hotmart_activation_session(
            db,
            current_user=current_user,
            workspace_context=workspace_context,
        )
        created_session = True

    try:
        response = claim_hotmart_pending_activation(
            db,
            activation_token=activation_token,
            payload=HotmartPendingActivationClaimRequest(session_id=target_session.id),
            current_user=current_user,
        )
    except PermissionError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    db.commit()
    db.refresh(target_session)
    work_redirect_path = _session_work_redirect_path(target_session)
    product_redirect_path = _session_product_redirect_path(target_session)
    return HotmartPendingActivationBootstrapResponse(
        created_session=created_session,
        pending_activation=response,
        project_title=target_session.title,
        redirect_path=work_redirect_path if created_session else product_redirect_path,
        product_redirect_path=product_redirect_path,
        session_id=target_session.id,
        work_redirect_path=work_redirect_path,
        workspace_id=target_session.workspace_id,
    )


@router.get("/sessions/{session_id}/product-overview", response_model=ProductOverviewResponse)
def get_product_overview_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ProductOverviewResponse:
    record = _get_record_or_404(db, session_id, current_user.id)
    return _product_overview(db, record, current_user)


@router.get("/sessions/{session_id}/commercial-access", response_model=CommercialAccessSnapshotV2)
def get_commercial_access_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> CommercialAccessSnapshotV2:
    record = _get_record_or_404(db, session_id, current_user.id)
    return build_commercial_access_snapshot_v2(db, record, current_user=current_user)


@router.get("/sessions/{session_id}/blueprint/result", response_model=BlueprintResultResponse)
def get_blueprint_result_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> BlueprintResultResponse:
    record = _get_record_or_404(db, session_id, current_user.id)
    snapshot = build_snapshot(db, record, include_short_term=False, current_user=current_user)
    preview = generate_acp_preview(snapshot)
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    context_state = resolve_effective_entitlement_state(db, record)
    diagram_context = build_entitlement_context(
        tier=context_state.tier,
        workspace_id=record.workspace_id,
        user_id=current_user.id,
        role=role_for_user(db, workspace_id=record.workspace_id, user_id=current_user.id),
        purchase_refs=context_state.purchase_refs,
    )
    diagram_catalog = build_diagram_catalog(
        snapshot=snapshot,
        preview=preview,
        context=diagram_context,
        workspace_id=record.workspace_id,
    )
    blueprint = snapshot.blueprint
    sections = []
    if blueprint is not None:
        sections = [
            {"key": "architecture", "title": "Arquitectura", "status": "ready", "summary": blueprint.architecture},
            {"key": "reasoning", "title": "Patron de razonamiento", "status": "ready", "summary": blueprint.reasoning_pattern},
            {"key": "memory", "title": "Memoria", "status": "ready", "summary": blueprint.memory_strategy},
            {"key": "tools", "title": "Herramientas", "status": "ready", "summary": f"{len(blueprint.tools)} herramienta(s) definidas."},
        ]
    return BlueprintResultResponse(
        session_id=record.id,
        workspace_id=record.workspace_id,
        title=record.title,
        version_number=snapshot.blueprint_versions[0].version_number if snapshot.blueprint_versions else None,
        state="generated" if blueprint is not None else "not_generated",
        stale=bool(
            getattr(snapshot.blueprint_consistency, "is_stale", False)
            or getattr(snapshot.blueprint_consistency, "stale_stage_keys", [])
        ),
        access=access,
        summary=blueprint.narrative if blueprint is not None else "Todavia no existe Blueprint generado para esta sesion.",
        architecture_sample=blueprint.architecture if blueprint is not None else "",
        sections=sections,
        diagrams=diagram_catalog.entries[:8],
        estimation=snapshot.estimation_report.model_dump(mode="json") if snapshot.estimation_report is not None else {},
    )


@router.get("/sessions/{session_id}/blueprint/offer", response_model=ProductOfferResponse)
def get_blueprint_offer_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> ProductOfferResponse:
    record = _get_record_or_404(db, session_id, current_user.id)
    product = get_active_product(db, "blueprint_pro")
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    role = role_for_user(db, workspace_id=record.workspace_id, user_id=current_user.id)
    return ProductOfferResponse(
        session_id=record.id,
        workspace_id=record.workspace_id,
        product=build_product_response(db, product),
        access=access,
        can_checkout=role in {WorkspaceRole.owner, WorkspaceRole.admin}
        and tier_rank(access.tier) < tier_rank(CommercialTier.blueprint_pro),
        checkout_disabled_reason="" if tier_rank(access.tier) < tier_rank(CommercialTier.blueprint_pro) else "El Blueprint Profesional ya esta activo.",
        comparison={
            "traditional_vs_blueprint": "El Blueprint reduce redescubrimiento, retrabajo de arquitectura y alineacion inicial.",
            "download_value": "Permite sacar el diseno de la plataforma como documento profesional.",
        },
    )


@router.get("/sessions/{session_id}/acp/invitation", response_model=AcpInvitationResponse)
def get_acp_invitation_route(
    session_id: UUID,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> AcpInvitationResponse:
    record = _get_record_or_404(db, session_id, current_user.id)
    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    product = get_active_product(db, "acp")
    state = "active" if access.tier == CommercialTier.acp else access.checkout_state
    if state == "not_started":
        state = "purchase_required"
    return AcpInvitationResponse(
        session_id=record.id,
        workspace_id=record.workspace_id,
        access=access,
        state=state,
        metrics={
            "estimated_additional_effort_reduction_percent": 28,
            "estimated_risk_reduction_percent": 35,
            "automation_readiness_lift_percent": 22,
        },
        comparison={
            "blueprint_basic": "Producto de entrada: inferir, registrar supuestos y mostrar valor con bajo costo.",
            "blueprint_premium": "Blueprint enriquecido con preguntas resueltas y reprocesamiento selectivo.",
            "acp": "Paquete tecnico portable para iniciar construccion con menos friccion.",
            "acp_agentic": "Prompts, contratos y pruebas preparados para herramientas agenticas.",
        },
        benefits=product.benefits,
        next_action="open_workspace" if access.tier == CommercialTier.acp else "checkout",
    )


@router.get("/trm", response_model=TRMResponse)
@router.get("/commerce/trm", response_model=TRMResponse)
def get_trm_route() -> TRMResponse:
    data = get_today_trm_data()
    return TRMResponse(
        unit_usd=1.0,
        trm_cop=data["rate"],
        date=data["date"],
        source=TRM_SOURCE_LABEL,
    )


@router.get("/base-prices", response_model=BasePricesResponse)
@router.get("/commerce/base-prices", response_model=BasePricesResponse)
def get_base_prices_route(db: Session = Depends(get_session)) -> BasePricesResponse:
    return get_base_prices_summary(db)


@router.patch("/base-prices", response_model=BasePricesResponse)
@router.patch("/commerce/base-prices", response_model=BasePricesResponse)
def update_base_prices_route(
    payload: BasePricesUpdateRequest,
    db: Session = Depends(get_session),
) -> BasePricesResponse:
    return update_base_prices_usd(db, payload.blueprint_pro_usd, payload.acp_premium_usd)


@router.post("/sessions/{session_id}/access-requests", response_model=AccessRequestResponse)
def create_access_request_route(
    session_id: UUID,
    payload: AccessRequestCreateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> AccessRequestResponse:
    record = _get_record_or_404(db, session_id, current_user.id)
    target_tier = CommercialTier.acp if payload.capability.startswith("acp") else CommercialTier.blueprint_pro
    product_key = "acp" if target_tier == CommercialTier.acp else "blueprint_pro"
    response = request_access(
        db,
        payload=payload,
        record=record,
        current_user=current_user,
        product_key=product_key,
        target_tier=target_tier,
    )
    db.commit()
    return response


@router.post("/access-requests/{request_id}/resolve", response_model=AccessRequestResponse)
def resolve_access_request_route(
    request_id: UUID,
    payload: AccessRequestResolveRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> AccessRequestResponse:
    access_request = db.get(CommercialAccessRequestRecord, request_id)
    if access_request is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Access request not found")
    try:
        response = resolve_access_request(db, access_request=access_request, payload=payload, current_user=current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/commerce/access-requests", response_model=list[AccessRequestResponse])
def list_access_requests_route(
    status: str | None = None,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[AccessRequestResponse]:
    try:
        return list_all_access_requests(db, status_filter=status, current_user=current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@router.get("/commerce/access-requests/count")
def get_access_requests_count_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> dict[str, int]:
    return get_access_requests_count(db, current_user=current_user)
