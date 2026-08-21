from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db import get_session
from app.models import (
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
from app.services.hotmart.payment_links import (
    HotmartPaymentLinkError,
    create_hotmart_payment_link_for_order,
    list_hotmart_payment_links,
    list_hotmart_product_mappings,
    refresh_hotmart_payment_link,
    upsert_hotmart_product_mapping,
)
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
from app.services.runtime_access_control import ensure_workspace_runtime_admin
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context
from app.services.workspace_bootstrap import apply_workspace_bootstrap


router = APIRouter(prefix="/admin/integrations/hotmart", tags=["hotmart-admin"])


@router.get("/status", response_model=HotmartIntegrationStatusResponse)
def get_hotmart_status_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartIntegrationStatusResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return build_hotmart_status(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.post("/credentials", response_model=HotmartIntegrationStatusResponse)
def upsert_hotmart_credentials_route(
    payload: HotmartCredentialUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartIntegrationStatusResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = upsert_hotmart_credentials(
            db,
            workspace_id=workspace_context.workspace.id,
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
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_product_mappings(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.post("/mappings", response_model=HotmartProductMappingResponse)
def upsert_hotmart_mapping_route(
    payload: HotmartProductMappingUpsertRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartProductMappingResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = upsert_hotmart_product_mapping(
            db,
            workspace_id=workspace_context.workspace.id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


@router.get("/payment-links", response_model=list[HotmartPaymentLinkResponse])
def list_hotmart_payment_links_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartPaymentLinkResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_payment_links(db, workspace_id=workspace_context.workspace.id)


@router.post("/payment-links", response_model=HotmartPaymentLinkResponse)
def create_hotmart_payment_link_route(
    payload: HotmartPaymentLinkCreateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartPaymentLinkResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = create_hotmart_payment_link_for_order(
            db,
            workspace_id=workspace_context.workspace.id,
            payload=payload,
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
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = refresh_hotmart_payment_link(
            db,
            workspace_id=workspace_context.workspace.id,
            payment_link_id=payment_link_id,
            environment=environment,
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
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartPromotionResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_promotions(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.get("/coupons/metrics", response_model=HotmartPromotionMetricsResponse)
def get_hotmart_coupon_metrics_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartPromotionMetricsResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return build_hotmart_promotion_metrics(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.post("/coupons", response_model=HotmartPromotionResponse)
def create_hotmart_coupon_promotion_route(
    payload: HotmartPromotionCreateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartPromotionResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = create_hotmart_coupon_promotion(
            db,
            workspace_id=workspace_context.workspace.id,
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
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = delete_hotmart_coupon_promotion(
            db,
            workspace_id=workspace_context.workspace.id,
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
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = run_hotmart_manual_sync(
            db,
            workspace_id=workspace_context.workspace.id,
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
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartSyncRunResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_sync_runs(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
        resource=resource,
    )


@router.get("/sync-cursors", response_model=list[HotmartSyncCursorResponse])
def list_hotmart_sync_cursors_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartSyncCursorResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_sync_cursors(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.get("/reconciliation", response_model=list[HotmartReconciliationIssueResponse])
def list_hotmart_reconciliation_issues_route(
    environment: str = Query(default="sandbox"),
    status_filter: str = Query(default="open", alias="status"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartReconciliationIssueResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_reconciliation_issues(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
        status=status_filter,
    )


@router.post("/reconciliation/{issue_id}/resolve", response_model=HotmartReconciliationIssueResponse)
def resolve_hotmart_reconciliation_issue_route(
    issue_id: UUID,
    payload: HotmartReconciliationResolveRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartReconciliationIssueResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = resolve_hotmart_reconciliation_issue(
            db,
            workspace_id=workspace_context.workspace.id,
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
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = replay_hotmart_webhook_event(
            db,
            workspace_id=workspace_context.workspace.id,
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
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return get_hotmart_club_overview(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.post("/club/sync", response_model=HotmartSyncRunResponse)
def sync_hotmart_club_route(
    payload: HotmartClubSyncRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartSyncRunResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    try:
        response = sync_hotmart_club(
            db,
            workspace_id=workspace_context.workspace.id,
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
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartClubModuleResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_club_modules(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.get("/club/pages", response_model=list[HotmartClubPageResponse])
def list_hotmart_club_pages_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartClubPageResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_club_pages(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.get("/club/students", response_model=list[HotmartClubStudentResponse])
def list_hotmart_club_students_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartClubStudentResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_club_students(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.get("/club/progress", response_model=list[HotmartClubProgressResponse])
def list_hotmart_club_progress_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartClubProgressResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_club_progress(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.get("/release-readiness", response_model=HotmartReleaseReadinessResponse)
def get_hotmart_release_readiness_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartReleaseReadinessResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return build_hotmart_release_readiness(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.get("/alerts", response_model=list[HotmartOperationalAlertResponse])
def list_hotmart_operational_alerts_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartOperationalAlertResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_operational_alerts(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )


@router.get("/runbook", response_model=list[HotmartRunbookSectionResponse])
def list_hotmart_runbook_sections_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> list[HotmartRunbookSectionResponse]:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    return list_hotmart_runbook_sections()


@router.post("/test-connection", response_model=HotmartTestConnectionResponse)
def test_hotmart_connection_route(
    environment: str = Query(default="sandbox"),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> HotmartTestConnectionResponse:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    response = test_hotmart_connection(
        db,
        workspace_id=workspace_context.workspace.id,
        environment=environment,
    )
    db.commit()
    return response
