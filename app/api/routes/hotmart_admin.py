from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    CommercialAdminBootstrapResponse,
    CommercialBalanceLedgerResponse,
    CommercialBalanceSnapshotResponse,
    CommercialDebtResponse,
    CommercialDebtSettlementRequest,
    CommercialLegacyPackageResolutionResolveRequest,
    CommercialLegacyPackageResolutionResponse,
    CommercialPackageCatalogResponse,
    CommercialPackageCatalogUpsertRequest,
    CommercialPackageRecommendationResponse,
    CommercialQuotaBucketStatus,
    CommercialQuotaEffectiveConfigResponse,
    CommercialQuotaLedgerMovementType,
    CommercialQuotaProductConfigResponse,
    CommercialQuotaProductConfigUpsertRequest,
    CommercialQuotaSourceKind,
    CommercialQuotaWorkspaceOverrideRecord,
    CommercialQuotaWorkspaceOverrideResponse,
    CommercialQuotaWorkspaceOverrideUpsertRequest,
    HotmartClubModuleResponse,
    HotmartClubOverviewResponse,
    HotmartClubPageResponse,
    HotmartClubProgressResponse,
    HotmartClubStudentResponse,
    HotmartClubSyncRequest,
    HotmartCredentialUpsertRequest,
    HotmartIntegrationStatusResponse,
    HotmartPaymentLinkCreateRequest,
    HotmartPaymentLinkResponse,
    HotmartOperationalAlertResponse,
    HotmartPendingActivationResponse,
    HotmartPromotionCreateRequest,
    HotmartPromotionDeleteResponse,
    HotmartPromotionMetricsResponse,
    HotmartPromotionResponse,
    HotmartProductMappingResponse,
    HotmartProductMappingUpsertRequest,
    HotmartReconciliationIssueResponse,
    HotmartReconciliationResolveRequest,
    HotmartReleaseReadinessResponse,
    HotmartRunbookSectionResponse,
    HotmartSyncCursorResponse,
    HotmartSyncRequest,
    HotmartSyncRunResponse,
    HotmartTestConnectionResponse,
    HotmartWebhookReplayResponse,
    UserRecord,
)
from app.services.auth_service import get_current_user
from app.services.commercial_catalog_service import (
    list_package_catalog,
    recommend_package_for_product,
    upsert_package_catalog_entry,
)
from app.services.commercial_debt_service import count_commercial_debts, list_commercial_debts, settle_commercial_debt
from app.services.commerce_service import record_commercial_event
from app.services.commercial_package_fulfillment_service import (
    list_legacy_package_resolutions,
    resolve_legacy_package_resolution,
)
from app.services.commercial_quota_service import (
    get_balance_snapshot,
    list_balance_ledger,
    list_quota_product_configs,
    resolve_effective_quota_config,
    upsert_quota_product_config,
    upsert_workspace_quota_override,
)
from app.services.hotmart.payment_links import (
    HotmartPaymentLinkError,
    create_hotmart_payment_link_for_order,
    list_hotmart_payment_links,
    list_hotmart_product_mappings,
    refresh_hotmart_payment_link,
    upsert_hotmart_product_mapping,
)
from app.services.hotmart.pending_activations import list_pending_hotmart_activations
from app.services.hotmart.coupons import (
    HotmartCouponError,
    build_hotmart_promotion_metrics,
    create_hotmart_coupon_promotion,
    delete_hotmart_coupon_promotion,
    list_hotmart_promotions,
)
from app.services.hotmart.club import (
    HotmartClubError,
    get_hotmart_club_overview,
    list_hotmart_club_modules,
    list_hotmart_club_pages,
    list_hotmart_club_progress,
    list_hotmart_club_students,
    sync_hotmart_club,
)
from app.services.hotmart.secrets import (
    build_hotmart_status,
    test_hotmart_connection,
    upsert_hotmart_credentials,
)
from app.services.hotmart.release import (
    build_hotmart_release_readiness,
    list_hotmart_operational_alerts,
    list_hotmart_runbook_sections,
)
from app.services.hotmart.sync import (
    HotmartSyncError,
    list_hotmart_reconciliation_issues,
    list_hotmart_sync_cursors,
    list_hotmart_sync_runs,
    replay_hotmart_webhook_event,
    resolve_hotmart_reconciliation_issue,
    run_hotmart_manual_sync,
)
from app.services.runtime_access_control import ensure_platform_admin
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context
from app.services.workspace_bootstrap import apply_workspace_bootstrap, resolve_platform_admin_template_workspace_id


router = APIRouter(prefix="/admin/integrations/hotmart", tags=["hotmart-admin"])


@router.get("/status", response_model=HotmartIntegrationStatusResponse)
def get_hotmart_status_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartIntegrationStatusResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return build_hotmart_status(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
    )


@router.post("/credentials", response_model=HotmartIntegrationStatusResponse)
def upsert_hotmart_credentials_route(
    payload: HotmartCredentialUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartIntegrationStatusResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, platform_workspace_id)
    try:
        response = upsert_hotmart_credentials(
            db,
            workspace_id=platform_workspace_id,
            payload=payload,
            actor_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/mappings", response_model=list[HotmartProductMappingResponse])
def list_hotmart_mappings_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartProductMappingResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_product_mappings(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
    )


@router.post("/mappings", response_model=HotmartProductMappingResponse)
def upsert_hotmart_mapping_route(
    payload: HotmartProductMappingUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartProductMappingResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, platform_workspace_id)
    try:
        response = upsert_hotmart_product_mapping(
            db,
            workspace_id=platform_workspace_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/payment-links", response_model=list[HotmartPaymentLinkResponse])
def list_hotmart_payment_links_route(
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartPaymentLinkResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    return list_hotmart_payment_links(db, workspace_id=workspace_context.workspace.id, limit=limit)


@router.post("/payment-links", response_model=HotmartPaymentLinkResponse)
def create_hotmart_payment_link_route(
    payload: HotmartPaymentLinkCreateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartPaymentLinkResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = create_hotmart_payment_link_for_order(
            db,
            workspace_id=workspace_context.workspace.id,
            payload=payload,
            integration_workspace_id=platform_workspace_id,
        )
    except HotmartPaymentLinkError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.post("/payment-links/{payment_link_id}/refresh", response_model=HotmartPaymentLinkResponse)
def refresh_hotmart_payment_link_route(
    payment_link_id: UUID,
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartPaymentLinkResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = refresh_hotmart_payment_link(
            db,
            workspace_id=workspace_context.workspace.id,
            payment_link_id=payment_link_id,
            environment=environment,
            integration_workspace_id=platform_workspace_id,
        )
    except HotmartPaymentLinkError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/coupons", response_model=list[HotmartPromotionResponse])
def list_hotmart_coupon_promotions_route(
    environment: str = Query(default="sandbox"),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartPromotionResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_promotions(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
        limit=limit,
    )


@router.get("/coupons/metrics", response_model=HotmartPromotionMetricsResponse)
def get_hotmart_coupon_metrics_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartPromotionMetricsResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return build_hotmart_promotion_metrics(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
    )


@router.post("/coupons", response_model=HotmartPromotionResponse)
def create_hotmart_coupon_promotion_route(
    payload: HotmartPromotionCreateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartPromotionResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, platform_workspace_id)
    try:
        response = create_hotmart_coupon_promotion(
            db,
            workspace_id=platform_workspace_id,
            payload=payload,
            actor_user_id=current_user.id,
        )
    except HotmartCouponError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.delete("/coupons/{coupon_ref}", response_model=HotmartPromotionDeleteResponse)
def delete_hotmart_coupon_promotion_route(
    coupon_ref: str,
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartPromotionDeleteResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, platform_workspace_id)
    try:
        response = delete_hotmart_coupon_promotion(
            db,
            workspace_id=platform_workspace_id,
            coupon_ref=coupon_ref,
            environment=environment,
            actor_user_id=current_user.id,
        )
    except HotmartCouponError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.post("/sync", response_model=HotmartSyncRunResponse)
def run_hotmart_manual_sync_route(
    payload: HotmartSyncRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartSyncRunResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, platform_workspace_id)
    try:
        response = run_hotmart_manual_sync(
            db,
            workspace_id=platform_workspace_id,
            payload=payload,
            actor_user_id=current_user.id,
        )
    except HotmartSyncError as exc:
        db.commit()
        http_status = status.HTTP_429_TOO_MANY_REQUESTS if exc.code == "rate_limited" else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=http_status, detail=str(exc)) from exc
    except ValueError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/sync-runs", response_model=list[HotmartSyncRunResponse])
def list_hotmart_sync_runs_route(
    environment: str = Query(default="sandbox"),
    resource: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartSyncRunResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_sync_runs(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
        resource=resource,
        limit=limit,
    )


@router.get("/sync-cursors", response_model=list[HotmartSyncCursorResponse])
def list_hotmart_sync_cursors_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartSyncCursorResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_sync_cursors(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
    )


@router.get("/reconciliation", response_model=list[HotmartReconciliationIssueResponse])
def list_hotmart_reconciliation_issues_route(
    environment: str = Query(default="sandbox"),
    status_filter: str = Query(default="open", alias="status"),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartReconciliationIssueResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_reconciliation_issues(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
        status=status_filter,
        limit=limit,
    )


@router.get("/pending-activations", response_model=list[HotmartPendingActivationResponse])
def list_hotmart_pending_activations_route(
    workspace_id: UUID | None = Query(default=None),
    status_filter: str = Query(default="pending_activation", alias="status"),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[HotmartPendingActivationResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    return list_pending_hotmart_activations(
        db,
        source_workspace_id=workspace_id,
        status_filter=status_filter,
        limit=limit,
    )


@router.post("/reconciliation/{issue_id}/resolve", response_model=HotmartReconciliationIssueResponse)
def resolve_hotmart_reconciliation_issue_route(
    issue_id: UUID,
    payload: HotmartReconciliationResolveRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartReconciliationIssueResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, platform_workspace_id)
    try:
        response = resolve_hotmart_reconciliation_issue(
            db,
            workspace_id=platform_workspace_id,
            issue_id=issue_id,
            payload=payload,
            actor_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.post("/webhooks/{event_ref}/replay", response_model=HotmartWebhookReplayResponse)
def replay_hotmart_webhook_event_route(
    event_ref: str,
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartWebhookReplayResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, platform_workspace_id)
    try:
        response = replay_hotmart_webhook_event(
            db,
            workspace_id=platform_workspace_id,
            event_ref=event_ref,
            environment=environment,
            actor_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/club/overview", response_model=HotmartClubOverviewResponse)
def get_hotmart_club_overview_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartClubOverviewResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return get_hotmart_club_overview(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
    )


@router.post("/club/sync", response_model=HotmartSyncRunResponse)
def sync_hotmart_club_route(
    payload: HotmartClubSyncRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartSyncRunResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, platform_workspace_id)
    try:
        response = sync_hotmart_club(
            db,
            workspace_id=platform_workspace_id,
            payload=payload,
            actor_user_id=current_user.id,
        )
    except HotmartClubError as exc:
        db.commit()
        http_status = status.HTTP_429_TOO_MANY_REQUESTS if exc.code == "rate_limited" else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=http_status, detail=str(exc)) from exc
    except ValueError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/club/modules", response_model=list[HotmartClubModuleResponse])
def list_hotmart_club_modules_route(
    environment: str = Query(default="sandbox"),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartClubModuleResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_club_modules(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
        limit=limit,
    )


@router.get("/club/pages", response_model=list[HotmartClubPageResponse])
def list_hotmart_club_pages_route(
    environment: str = Query(default="sandbox"),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartClubPageResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_club_pages(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
        limit=limit,
    )


@router.get("/club/students", response_model=list[HotmartClubStudentResponse])
def list_hotmart_club_students_route(
    environment: str = Query(default="sandbox"),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartClubStudentResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_club_students(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
        limit=limit,
    )


@router.get("/club/progress", response_model=list[HotmartClubProgressResponse])
def list_hotmart_club_progress_route(
    environment: str = Query(default="sandbox"),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartClubProgressResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_club_progress(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
        limit=limit,
    )


@router.get("/release-readiness", response_model=HotmartReleaseReadinessResponse)
def get_hotmart_release_readiness_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartReleaseReadinessResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return build_hotmart_release_readiness(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
    )


@router.get("/alerts", response_model=list[HotmartOperationalAlertResponse])
def list_hotmart_operational_alerts_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartOperationalAlertResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    return list_hotmart_operational_alerts(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
    )


@router.get("/runbook", response_model=list[HotmartRunbookSectionResponse])
def list_hotmart_runbook_sections_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartRunbookSectionResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    return list_hotmart_runbook_sections()


@router.post("/test-connection", response_model=HotmartTestConnectionResponse)
def test_hotmart_connection_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartTestConnectionResponse:
    _ensure_platform_admin_or_403(db, current_user)
    platform_workspace_id = _hotmart_platform_workspace_id(db, current_user, workspace_context)
    apply_workspace_bootstrap(db, platform_workspace_id)
    response = test_hotmart_connection(
        db,
        workspace_id=platform_workspace_id,
        environment=environment,
    )
    db.commit()
    return response


def _ensure_platform_admin_or_403(db: Session, current_user: UserRecord) -> None:
    try:
        ensure_platform_admin(db, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def _hotmart_platform_workspace_id(
    db: Session,
    current_user: UserRecord,
    workspace_context: WorkspaceAccessContext,
) -> UUID:
    """Resolve the single platform scope used to store Hotmart configuration."""
    platform_workspace_id = resolve_platform_admin_template_workspace_id(db)
    if platform_workspace_id is not None:
        return platform_workspace_id
    if current_user.default_workspace_id is not None:
        return current_user.default_workspace_id
    return workspace_context.workspace.id


def _serialize_quota_product_config(record) -> CommercialQuotaProductConfigResponse:
    return CommercialQuotaProductConfigResponse(
        id=record.id,
        product_key=record.product_key,
        display_name=record.display_name,
        enabled=record.enabled,
        initial_free_units=record.initial_free_units,
        consumption_priority=list(record.consumption_priority),
        checkout_required_on_zero_balance=record.checkout_required_on_zero_balance,
        fifo_auto_approval_enabled=record.fifo_auto_approval_enabled,
        default_blocked_request_ttl_hours=record.default_blocked_request_ttl_hours,
        default_checkout_ttl_minutes=record.default_checkout_ttl_minutes,
        debt_enabled=record.debt_enabled,
        allow_manual_override_without_charge=record.allow_manual_override_without_charge,
        allow_courtesy=record.allow_courtesy,
        allow_debt_pending=record.allow_debt_pending,
        catalog_priority_strategy=record.catalog_priority_strategy,
        sync_retry_limit=record.sync_retry_limit,
        duplicate_conflict_visibility=record.duplicate_conflict_visibility,
        updated_at=record.updated_at,
    )


def _serialize_workspace_override(record: CommercialQuotaWorkspaceOverrideRecord) -> CommercialQuotaWorkspaceOverrideResponse:
    return CommercialQuotaWorkspaceOverrideResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        product_key=record.product_key,
        is_active=record.is_active,
        enabled_override=record.enabled_override,
        free_units_override=record.free_units_override,
        consumption_priority_override=list(record.consumption_priority_override),
        checkout_required_on_zero_balance_override=record.checkout_required_on_zero_balance_override,
        fifo_auto_approval_enabled_override=record.fifo_auto_approval_enabled_override,
        default_blocked_request_ttl_hours_override=record.default_blocked_request_ttl_hours_override,
        default_checkout_ttl_minutes_override=record.default_checkout_ttl_minutes_override,
        debt_enabled_override=record.debt_enabled_override,
        effective_from=record.effective_from,
        effective_to=record.effective_to,
        notes=record.notes,
        updated_by_user_id=record.updated_by_user_id,
        updated_at=record.updated_at,
    )


def _serialize_effective_quota(
    *,
    workspace_id: UUID,
    config,
) -> CommercialQuotaEffectiveConfigResponse:
    return CommercialQuotaEffectiveConfigResponse(
        workspace_id=workspace_id,
        product_key=config.product_key,
        display_name=config.display_name,
        enabled=config.enabled,
        initial_free_units=config.initial_free_units,
        consumption_priority=[item.value for item in config.consumption_priority],
        checkout_required_on_zero_balance=config.checkout_required_on_zero_balance,
        fifo_auto_approval_enabled=config.fifo_auto_approval_enabled,
        default_blocked_request_ttl_hours=config.default_blocked_request_ttl_hours,
        default_checkout_ttl_minutes=config.default_checkout_ttl_minutes,
        debt_enabled=config.debt_enabled,
        allow_manual_override_without_charge=config.allow_manual_override_without_charge,
        allow_courtesy=config.allow_courtesy,
        allow_debt_pending=config.allow_debt_pending,
        catalog_priority_strategy=config.catalog_priority_strategy,
        sync_retry_limit=config.sync_retry_limit,
        duplicate_conflict_visibility=config.duplicate_conflict_visibility,
        override_id=config.override_id,
    )


def _serialize_balance_snapshot(snapshot) -> CommercialBalanceSnapshotResponse:
    return CommercialBalanceSnapshotResponse(
        workspace_id=snapshot.workspace_id,
        product_key=snapshot.product_key,
        total_available_units=snapshot.total_available_units,
        by_source_kind={key.value: value for key, value in snapshot.by_source_kind.items()},
        buckets=[
            {
                "bucket_id": bucket.bucket_id,
                "bucket_key": bucket.bucket_key,
                "source_kind": bucket.source_kind,
                "status": bucket.status,
                "units_granted": bucket.units_granted,
                "units_consumed": bucket.units_consumed,
                "available_units": bucket.available_units,
                "starts_at": bucket.starts_at,
                "ends_at": bucket.ends_at,
                "source_ref": bucket.source_ref,
            }
            for bucket in snapshot.buckets
        ],
    )


def _serialize_balance_ledger_entry(record) -> CommercialBalanceLedgerResponse:
    return CommercialBalanceLedgerResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        product_key=record.product_key,
        bucket_id=record.bucket_id,
        movement_type=record.movement_type,
        source_kind=record.source_kind,
        delta_units=record.delta_units,
        balance_before_units=record.balance_before_units,
        balance_after_units=record.balance_after_units,
        bucket_balance_before_units=record.bucket_balance_before_units,
        bucket_balance_after_units=record.bucket_balance_after_units,
        source_ref=record.source_ref,
        actor_user_id=record.actor_user_id,
        order_id=record.order_id,
        payment_id=record.payment_id,
        access_request_id=record.access_request_id,
        metadata=record.metadata_payload,
        created_at=record.created_at,
    )


@router.get("/commercial/quota-products", response_model=list[CommercialQuotaProductConfigResponse])
def list_commercial_quota_products_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[CommercialQuotaProductConfigResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    return [_serialize_quota_product_config(item) for item in list_quota_product_configs(db)]


@router.get("/commercial/bootstrap", response_model=CommercialAdminBootstrapResponse)
def get_commercial_bootstrap_route(
    product_key: str,
    workspace_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommercialAdminBootstrapResponse:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    quota_configs = [_serialize_quota_product_config(item) for item in list_quota_product_configs(db)]
    workspace_overrides = [
        _serialize_workspace_override(row)
        for row in db.exec(
            select(CommercialQuotaWorkspaceOverrideRecord)
            .where(CommercialQuotaWorkspaceOverrideRecord.workspace_id == target_workspace_id)
            .order_by(CommercialQuotaWorkspaceOverrideRecord.product_key.asc())
        ).all()
    ]
    effective_config = _serialize_effective_quota(
        workspace_id=target_workspace_id,
        config=resolve_effective_quota_config(
            db,
            workspace_id=target_workspace_id,
            product_key=product_key,
        ),
    )
    balance_snapshot = _serialize_balance_snapshot(
        get_balance_snapshot(
            db,
            workspace_id=target_workspace_id,
            product_key=product_key,
        )
    )
    recommendation = recommend_package_for_product(
        db,
        product_key=product_key,
        required_units=1,
        workspace_id=target_workspace_id,
    )
    return CommercialAdminBootstrapResponse(
        workspace_id=target_workspace_id,
        product_key=product_key,
        balance_snapshot=balance_snapshot,
        effective_config=effective_config,
        open_debt_count=count_commercial_debts(
            db,
            workspace_id=target_workspace_id,
            status="open",
            product_key=product_key,
        ),
        quota_configs=quota_configs,
        recommendation=recommendation,
        workspace_overrides=workspace_overrides,
    )


@router.post("/commercial/quota-products", response_model=CommercialQuotaProductConfigResponse)
def upsert_commercial_quota_product_route(
    payload: CommercialQuotaProductConfigUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> CommercialQuotaProductConfigResponse:
    _ensure_platform_admin_or_403(db, current_user)
    record = upsert_quota_product_config(
        db,
        product_key=payload.product_key,
        display_name=payload.display_name,
        enabled=payload.enabled,
        initial_free_units=payload.initial_free_units,
        consumption_priority=payload.consumption_priority,
        checkout_required_on_zero_balance=payload.checkout_required_on_zero_balance,
        fifo_auto_approval_enabled=payload.fifo_auto_approval_enabled,
        default_blocked_request_ttl_hours=payload.default_blocked_request_ttl_hours,
        default_checkout_ttl_minutes=payload.default_checkout_ttl_minutes,
        debt_enabled=payload.debt_enabled,
        allow_manual_override_without_charge=payload.allow_manual_override_without_charge,
        allow_courtesy=payload.allow_courtesy,
        allow_debt_pending=payload.allow_debt_pending,
        catalog_priority_strategy=payload.catalog_priority_strategy,
        sync_retry_limit=payload.sync_retry_limit,
        duplicate_conflict_visibility=payload.duplicate_conflict_visibility,
        metadata=payload.metadata,
    )
    db.commit()
    return _serialize_quota_product_config(record)


@router.get("/commercial/workspace-overrides", response_model=list[CommercialQuotaWorkspaceOverrideResponse])
def list_commercial_workspace_overrides_route(
    workspace_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[CommercialQuotaWorkspaceOverrideResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    rows = db.exec(
        select(CommercialQuotaWorkspaceOverrideRecord)
        .where(CommercialQuotaWorkspaceOverrideRecord.workspace_id == target_workspace_id)
        .order_by(CommercialQuotaWorkspaceOverrideRecord.product_key.asc())
    ).all()
    return [_serialize_workspace_override(row) for row in rows]


@router.post("/commercial/workspace-overrides", response_model=CommercialQuotaWorkspaceOverrideResponse)
def upsert_commercial_workspace_override_route(
    payload: CommercialQuotaWorkspaceOverrideUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> CommercialQuotaWorkspaceOverrideResponse:
    _ensure_platform_admin_or_403(db, current_user)
    record = upsert_workspace_quota_override(
        db,
        workspace_id=payload.workspace_id,
        product_key=payload.product_key,
        is_active=payload.is_active,
        enabled_override=payload.enabled_override,
        free_units_override=payload.free_units_override,
        consumption_priority_override=payload.consumption_priority_override,
        checkout_required_on_zero_balance_override=payload.checkout_required_on_zero_balance_override,
        fifo_auto_approval_enabled_override=payload.fifo_auto_approval_enabled_override,
        default_blocked_request_ttl_hours_override=payload.default_blocked_request_ttl_hours_override,
        default_checkout_ttl_minutes_override=payload.default_checkout_ttl_minutes_override,
        debt_enabled_override=payload.debt_enabled_override,
        effective_from=payload.effective_from,
        effective_to=payload.effective_to,
        notes=payload.notes,
        updated_by_user_id=current_user.id,
        metadata=payload.metadata,
    )
    db.commit()
    return _serialize_workspace_override(record)


@router.get("/commercial/effective-config", response_model=CommercialQuotaEffectiveConfigResponse)
def get_commercial_effective_config_route(
    product_key: str,
    workspace_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommercialQuotaEffectiveConfigResponse:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    resolved = resolve_effective_quota_config(
        db,
        workspace_id=target_workspace_id,
        product_key=product_key,
    )
    return _serialize_effective_quota(workspace_id=target_workspace_id, config=resolved)


@router.get("/commercial/balance-snapshot", response_model=CommercialBalanceSnapshotResponse)
def get_commercial_balance_snapshot_route(
    product_key: str,
    workspace_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommercialBalanceSnapshotResponse:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    snapshot = get_balance_snapshot(
        db,
        workspace_id=target_workspace_id,
        product_key=product_key,
    )
    return _serialize_balance_snapshot(snapshot)


@router.get("/commercial/balance-ledger", response_model=list[CommercialBalanceLedgerResponse])
def list_commercial_balance_ledger_route(
    product_key: str,
    workspace_id: UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[CommercialBalanceLedgerResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    return [
        _serialize_balance_ledger_entry(item)
        for item in list_balance_ledger(db, workspace_id=target_workspace_id, product_key=product_key, limit=limit)
    ]


@router.get("/commercial/package-catalog", response_model=list[CommercialPackageCatalogResponse])
def list_commercial_package_catalog_route(
    product_key: str = Query(default=""),
    include_disabled: bool = Query(default=True),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> list[CommercialPackageCatalogResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    return list_package_catalog(db, product_key=product_key, include_disabled=include_disabled)


@router.post("/commercial/package-catalog", response_model=CommercialPackageCatalogResponse)
def upsert_commercial_package_catalog_route(
    payload: CommercialPackageCatalogUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> CommercialPackageCatalogResponse:
    _ensure_platform_admin_or_403(db, current_user)
    response = upsert_package_catalog_entry(db, payload=payload)
    db.commit()
    return response


@router.get("/commercial/package-recommendation", response_model=CommercialPackageRecommendationResponse)
def get_commercial_package_recommendation_route(
    product_key: str,
    required_units: int = Query(default=1),
    workspace_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommercialPackageRecommendationResponse:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    return recommend_package_for_product(
        db,
        product_key=product_key,
        required_units=required_units,
        workspace_id=target_workspace_id,
    )


@router.get("/commercial/debts", response_model=list[CommercialDebtResponse])
def list_commercial_debts_route(
    status: str = Query(default="open"),
    product_key: str = Query(default=""),
    workspace_id: UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[CommercialDebtResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    return list_commercial_debts(
        db,
        workspace_id=target_workspace_id,
        status=status,
        product_key=product_key,
        limit=limit,
    )


@router.post("/commercial/debts/{debt_id}/settle", response_model=CommercialDebtResponse)
def settle_commercial_debt_route(
    debt_id: UUID,
    payload: CommercialDebtSettlementRequest,
    workspace_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommercialDebtResponse:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    response = settle_commercial_debt(
        db,
        workspace_id=target_workspace_id,
        debt_id=debt_id,
        payload=payload,
        actor_user_id=current_user.id,
    )
    db.commit()
    return response


@router.get("/commercial/legacy-package-resolutions", response_model=list[CommercialLegacyPackageResolutionResponse])
def list_commercial_legacy_package_resolutions_route(
    status_filter: str = Query(default="pending", alias="status"),
    product_key: str = Query(default=""),
    workspace_id: UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[CommercialLegacyPackageResolutionResponse]:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    return list_legacy_package_resolutions(
        db,
        workspace_id=target_workspace_id,
        status_filter=status_filter,
        product_key=product_key,
        limit=limit,
    )


@router.post(
    "/commercial/legacy-package-resolutions/{order_id}/resolve",
    response_model=CommercialLegacyPackageResolutionResponse,
)
def resolve_commercial_legacy_package_resolution_route(
    order_id: UUID,
    payload: CommercialLegacyPackageResolutionResolveRequest,
    workspace_id: UUID | None = Query(default=None),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> CommercialLegacyPackageResolutionResponse:
    _ensure_platform_admin_or_403(db, current_user)
    target_workspace_id = workspace_id or workspace_context.workspace.id
    try:
        response = resolve_legacy_package_resolution(
            db,
            workspace_id=target_workspace_id,
            order_id=order_id,
            package_code=payload.package_code,
            resolution_note=payload.resolution_note,
            actor_user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    record_commercial_event(
        db,
        workspace_id=response.workspace_id,
        session_id=response.session_id,
        user_id=current_user.id,
        event_key="legacy_package_resolution_resolved",
        product_key=response.product_key,
        source="hotmart_admin",
        metadata={
            "order_id": str(response.order_id),
            "package_code": response.selected_package_code,
        },
        correlation_id=response.checkout_ref,
    )
    db.commit()
    return response
