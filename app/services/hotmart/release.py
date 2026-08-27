from __future__ import annotations

from typing import Literal
from uuid import UUID

from sqlalchemy import func
from sqlmodel import Session, select

from app.models import (
    CommercialEventRecord,
    HotmartIntegrationStatusResponse,
    HotmartOperationalAlertResponse,
    HotmartPaymentLinkRecord,
    HotmartProductMappingRecord,
    HotmartPromotionRecord,
    HotmartReconciliationIssueRecord,
    HotmartReleaseChecklistItemResponse,
    HotmartReleaseReadinessResponse,
    HotmartRunbookSectionResponse,
    HotmartSyncRunRecord,
    HotmartWebhookEventRecord,
    utc_now,
)
from app.services.hotmart.auth import normalize_hotmart_environment
from app.services.hotmart.secrets import build_hotmart_status


AlertSeverity = Literal["critical", "high", "medium", "low"]
ChecklistStatus = Literal["failed", "manual", "passed", "warning"]

FAILED_WEBHOOK_STATUSES = {"rejected", "unresolved", "replay_requested"}
FAILED_SYNC_STATUSES = {"failed", "rate_limited"}
REVOCATION_EVENT_KEYS = {"hotmart_chargeback_received", "hotmart_payment_canceled", "hotmart_payment_refunded"}
REQUIRED_SYNC_RESOURCES = {"club", "coupons", "payment_links", "products", "sales"}


def _count_rows(session: Session, statement) -> int:  # noqa: ANN001 - SQLModel statement type is verbose and not useful here.
    return int(session.exec(select(func.count()).select_from(statement.subquery())).one())


def _check(
    *,
    key: str,
    label: str,
    passed: bool,
    detail: str,
    evidence: list[str] | None = None,
    severity: AlertSeverity = "medium",
    required: bool = True,
    warning: bool = False,
) -> HotmartReleaseChecklistItemResponse:
    status: ChecklistStatus = "passed" if passed else ("warning" if warning else "failed")
    return HotmartReleaseChecklistItemResponse(
        key=key,
        label=label,
        status=status,
        severity=severity,
        required=required,
        detail=detail,
        evidence=evidence or [],
    )


def _alert(
    *,
    key: str,
    severity: AlertSeverity,
    title: str,
    message: str,
    evidence: list[str] | None = None,
) -> HotmartOperationalAlertResponse:
    return HotmartOperationalAlertResponse(
        key=key,
        severity=severity,
        status="active",
        title=title,
        message=message,
        evidence=evidence or [],
        created_at=utc_now(),
    )


def _commercial_event_count(session: Session, *, workspace_id: UUID, event_keys: set[str]) -> int:
    return _count_rows(
        session,
        select(CommercialEventRecord).where(
            CommercialEventRecord.workspace_id == workspace_id,
            CommercialEventRecord.event_key.in_(event_keys),
        ),
    )


def _sync_success_count(session: Session, *, workspace_id: UUID, environment: str) -> int:
    return int(
        session.exec(
            select(func.count(func.distinct(HotmartSyncRunRecord.resource))).where(
                HotmartSyncRunRecord.workspace_id == workspace_id,
                HotmartSyncRunRecord.environment == environment,
                HotmartSyncRunRecord.status == "succeeded",
            )
        ).one()
    )


def _release_runbook() -> list[HotmartRunbookSectionResponse]:
    return [
        HotmartRunbookSectionResponse(
            key="credentials",
            title="Configurar credenciales",
            steps=[
                "Crear credenciales OAuth Hotmart en el panel de desarrolladores.",
                "Guardar Client ID, Client Secret, Basic Token y HOTTOK desde la consola admin.",
                "No pegar secretos en tickets, logs ni documentos compartidos.",
                "Ejecutar Probar conexion y verificar status connected.",
            ],
            links=["https://developers.hotmart.com/docs/en/start/authentication/"],
        ),
        HotmartRunbookSectionResponse(
            key="sandbox",
            title="Validar sandbox",
            steps=[
                "Seleccionar ambiente sandbox en la consola.",
                "Configurar webhook publico de pruebas.",
                "Mapear al menos un producto interno contra producto/oferta Hotmart sandbox.",
                "Generar un payment link y ejecutar webhook PURCHASE_APPROVED con payload de prueba.",
            ],
            links=["https://developers.hotmart.com/docs/en/start/sandbox/"],
        ),
        HotmartRunbookSectionResponse(
            key="production",
            title="Preparar produccion",
            steps=[
                "Rotar credenciales de produccion con permisos minimos necesarios.",
                "Confirmar visualmente badge production antes de guardar.",
                "Validar webhook publico HTTPS y HOTTOK productivo.",
                "Ejecutar sync productos, ventas, cupones y Club antes de abrir trafico.",
            ],
            links=[],
        ),
        HotmartRunbookSectionResponse(
            key="webhooks",
            title="Operar webhooks",
            steps=[
                "Revisar eventos rejected/unresolved desde alertas y reconciliacion.",
                "Si el HOTTOK falla, pausar Hotmart, rotar secreto y repetir prueba.",
                "Usar replay administrativo solo para abrir revision; no duplicar efectos manualmente sin evidencia.",
                "Ante refund o chargeback, verificar payment, order y entitlement.",
            ],
            links=["https://developers.hotmart.com/docs/en/v1/webhook/"],
        ),
        HotmartRunbookSectionResponse(
            key="reconciliation",
            title="Resolver diferencias",
            steps=[
                "Ejecutar sync por recurso y revisar issues abiertos.",
                "Resolver solo con accion y nota trazable.",
                "No cerrar issues criticos sin confirmar orden, pago y entitlement.",
                "Si Hotmart limita la API, reintentar desde el cursor guardado.",
            ],
            links=[],
        ),
        HotmartRunbookSectionResponse(
            key="rollback",
            title="Pausar y volver a fallback",
            steps=[
                "Deshabilitar Hotmart en Credenciales si el proveedor queda degradado.",
                "Cambiar provider comercial a sandbox/fallback segun politica del release.",
                "Mantener webhooks recibidos en estado auditable.",
                "Registrar motivo operativo y reconciliar pagos pendientes antes de reactivar.",
            ],
            links=[],
        ),
    ]


def list_hotmart_runbook_sections() -> list[HotmartRunbookSectionResponse]:
    return _release_runbook()


def _build_alerts(
    *,
    status: HotmartIntegrationStatusResponse,
    metrics: dict[str, int],
) -> list[HotmartOperationalAlertResponse]:
    alerts: list[HotmartOperationalAlertResponse] = []
    if not status.enabled:
        alerts.append(
            _alert(
                key="hotmart_disabled",
                severity="high",
                title="Hotmart esta deshabilitado",
                message="El modulo no puede operar como proveedor activo mientras enabled=false.",
                evidence=[f"environment={status.environment}"],
            )
        )
    if not (status.client_id_configured and status.client_secret_configured and status.basic_token_configured):
        alerts.append(
            _alert(
                key="hotmart_oauth_credentials_incomplete",
                severity="critical",
                title="Credenciales OAuth incompletas",
                message="Client ID, Client Secret y Basic Token son obligatorios para operar.",
                evidence=[
                    f"client_id={status.client_id_configured}",
                    f"client_secret={status.client_secret_configured}",
                    f"basic_token={status.basic_token_configured}",
                ],
            )
        )
    if not status.hottok_configured:
        alerts.append(
            _alert(
                key="hotmart_hottok_missing",
                severity="critical",
                title="HOTTOK no configurado",
                message="Los webhooks no deben habilitarse sin validacion HOTTOK.",
                evidence=["hottok_configured=false"],
            )
        )
    if not status.webhook_public_url.strip():
        alerts.append(
            _alert(
                key="hotmart_webhook_public_url_missing",
                severity="high",
                title="Webhook publico faltante",
                message="Hotmart no podra entregar eventos si la URL publica no esta configurada.",
                evidence=["webhook_public_url=empty"],
            )
        )
    if status.last_health_status not in {"connected", "healthy"}:
        alerts.append(
            _alert(
                key="hotmart_health_not_connected",
                severity="high",
                title="Conexion Hotmart no validada",
                message="Ejecuta Probar conexion antes de liberar.",
                evidence=[f"last_health_status={status.last_health_status or 'empty'}"],
            )
        )
    if metrics["active_mappings"] == 0:
        alerts.append(
            _alert(
                key="hotmart_no_active_mappings",
                severity="high",
                title="Sin mappings activos",
                message="No hay productos internos conectados a productos Hotmart.",
                evidence=["active_mappings=0"],
            )
        )
    if metrics["failed_webhooks"] > 0:
        alerts.append(
            _alert(
                key="hotmart_webhook_failures",
                severity="critical",
                title="Webhooks fallidos o pendientes",
                message="Existen webhooks rechazados, unresolved o pendientes de replay.",
                evidence=[f"failed_webhooks={metrics['failed_webhooks']}"],
            )
        )
    if metrics["failed_sync_runs"] > 0:
        alerts.append(
            _alert(
                key="hotmart_sync_failures",
                severity="high",
                title="Sync Hotmart fallido",
                message="Hay sync runs failed/rate_limited que deben revisarse.",
                evidence=[f"failed_sync_runs={metrics['failed_sync_runs']}"],
            )
        )
    if metrics["open_reconciliation_issues"] > 0:
        alerts.append(
            _alert(
                key="hotmart_reconciliation_open",
                severity="high" if metrics["critical_reconciliation_issues"] > 0 else "medium",
                title="Diferencias de reconciliacion abiertas",
                message="La cola de reconciliacion debe quedar revisada antes de produccion.",
                evidence=[
                    f"open_reconciliation_issues={metrics['open_reconciliation_issues']}",
                    f"critical_reconciliation_issues={metrics['critical_reconciliation_issues']}",
                ],
            )
        )
    if metrics["promotion_sync_errors"] > 0:
        alerts.append(
            _alert(
                key="hotmart_coupon_sync_errors",
                severity="medium",
                title="Promociones con sync_error",
                message="Existen cupones/promociones que no quedaron sincronizadas con Hotmart.",
                evidence=[f"promotion_sync_errors={metrics['promotion_sync_errors']}"],
            )
        )
    return alerts


def list_hotmart_operational_alerts(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> list[HotmartOperationalAlertResponse]:
    return build_hotmart_release_readiness(
        session,
        workspace_id=workspace_id,
        environment=environment,
    ).alerts


def build_hotmart_release_readiness(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> HotmartReleaseReadinessResponse:
    env = normalize_hotmart_environment(environment)
    status = build_hotmart_status(session, workspace_id=workspace_id, environment=env)
    metrics = {
        "active_mappings": _count_rows(
            session,
            select(HotmartProductMappingRecord).where(
                HotmartProductMappingRecord.workspace_id == workspace_id,
                HotmartProductMappingRecord.environment == env,
                HotmartProductMappingRecord.is_active == True,  # noqa: E712
            ),
        ),
        "payment_links": _count_rows(
            session,
            select(HotmartPaymentLinkRecord).where(
                HotmartPaymentLinkRecord.workspace_id == workspace_id,
                HotmartPaymentLinkRecord.environment == env,
            ),
        ),
        "active_payment_links": _count_rows(
            session,
            select(HotmartPaymentLinkRecord).where(
                HotmartPaymentLinkRecord.workspace_id == workspace_id,
                HotmartPaymentLinkRecord.environment == env,
                HotmartPaymentLinkRecord.activation_status == "active",
            ),
        ),
        "failed_webhooks": _count_rows(
            session,
            select(HotmartWebhookEventRecord).where(
                HotmartWebhookEventRecord.workspace_id == workspace_id,
                HotmartWebhookEventRecord.processing_status.in_(FAILED_WEBHOOK_STATUSES),
            ),
        ),
        "processed_webhooks": _count_rows(
            session,
            select(HotmartWebhookEventRecord).where(
                HotmartWebhookEventRecord.workspace_id == workspace_id,
                HotmartWebhookEventRecord.processing_status == "processed",
            ),
        ),
        "failed_sync_runs": _count_rows(
            session,
            select(HotmartSyncRunRecord).where(
                HotmartSyncRunRecord.workspace_id == workspace_id,
                HotmartSyncRunRecord.environment == env,
                HotmartSyncRunRecord.status.in_(FAILED_SYNC_STATUSES),
            ),
        ),
        "successful_sync_resources": _sync_success_count(session, workspace_id=workspace_id, environment=env),
        "open_reconciliation_issues": _count_rows(
            session,
            select(HotmartReconciliationIssueRecord).where(
                HotmartReconciliationIssueRecord.workspace_id == workspace_id,
                HotmartReconciliationIssueRecord.environment == env,
                HotmartReconciliationIssueRecord.status == "open",
            ),
        ),
        "critical_reconciliation_issues": _count_rows(
            session,
            select(HotmartReconciliationIssueRecord).where(
                HotmartReconciliationIssueRecord.workspace_id == workspace_id,
                HotmartReconciliationIssueRecord.environment == env,
                HotmartReconciliationIssueRecord.status == "open",
                HotmartReconciliationIssueRecord.severity.in_({"critical", "high"}),
            ),
        ),
        "promotion_sync_errors": _count_rows(
            session,
            select(HotmartPromotionRecord).where(
                HotmartPromotionRecord.workspace_id == workspace_id,
                HotmartPromotionRecord.environment == env,
                HotmartPromotionRecord.status == "sync_error",
            ),
        ),
        "approved_purchase_events": _commercial_event_count(
            session,
            workspace_id=workspace_id,
            event_keys={"hotmart_payment_approved"},
        ),
        "revocation_events": _commercial_event_count(
            session,
            workspace_id=workspace_id,
            event_keys=REVOCATION_EVENT_KEYS,
        ),
    }
    metrics["required_sync_resource_target"] = len(REQUIRED_SYNC_RESOURCES)

    checklist = [
        _check(
            key="integration_enabled",
            label="Integracion habilitada",
            passed=status.enabled,
            severity="high",
            detail="Hotmart debe estar enabled=true para operar.",
            evidence=[f"enabled={status.enabled}"],
        ),
        _check(
            key="oauth_credentials_configured",
            label="Credenciales OAuth configuradas",
            passed=status.client_id_configured and status.client_secret_configured and status.basic_token_configured,
            severity="critical",
            detail="Client ID, Client Secret y Basic Token deben estar guardados en storage seguro.",
            evidence=[
                f"client_id={status.client_id_configured}",
                f"client_secret={status.client_secret_configured}",
                f"basic_token={status.basic_token_configured}",
            ],
        ),
        _check(
            key="hottok_configured",
            label="HOTTOK configurado",
            passed=status.hottok_configured,
            severity="critical",
            detail="La validacion HOTTOK es obligatoria para webhooks productivos.",
            evidence=[f"hottok={status.hottok_configured}"],
        ),
        _check(
            key="webhook_public_url",
            label="Webhook publico configurado",
            passed=bool(status.webhook_public_url.strip()),
            severity="high",
            detail="La URL publica debe estar registrada en Hotmart.",
            evidence=[status.webhook_public_url or "empty"],
        ),
        _check(
            key="connection_health",
            label="Conexion validada",
            passed=status.last_health_status in {"connected", "healthy"},
            severity="high",
            detail="El ultimo health check debe ser connected/healthy.",
            evidence=[f"last_health_status={status.last_health_status or 'empty'}"],
        ),
        _check(
            key="active_product_mappings",
            label="Mappings activos",
            passed=metrics["active_mappings"] > 0,
            severity="high",
            detail="Debe existir al menos un mapping activo producto interno -> Hotmart.",
            evidence=[f"active_mappings={metrics['active_mappings']}"],
        ),
        _check(
            key="payment_link_generated",
            label="Payment link probado",
            passed=metrics["payment_links"] > 0,
            severity="medium",
            detail="Debe existir evidencia local de al menos un payment link Hotmart.",
            evidence=[f"payment_links={metrics['payment_links']}"],
            warning=metrics["payment_links"] == 0,
        ),
        _check(
            key="purchase_to_entitlement",
            label="Compra aprobada concede acceso",
            passed=metrics["approved_purchase_events"] > 0,
            severity="high",
            detail="Debe existir evidencia de webhook aprobado procesado.",
            evidence=[f"approved_purchase_events={metrics['approved_purchase_events']}"],
            warning=metrics["approved_purchase_events"] == 0,
        ),
        _check(
            key="refund_or_chargeback",
            label="Refund/chargeback afecta acceso",
            passed=metrics["revocation_events"] > 0,
            severity="high",
            detail="Debe existir evidencia de refund, cancelacion o chargeback procesado.",
            evidence=[f"revocation_events={metrics['revocation_events']}"],
            warning=metrics["revocation_events"] == 0,
        ),
        _check(
            key="sync_coverage",
            label="Sync de recursos ejecutado",
            passed=metrics["successful_sync_resources"] >= len(REQUIRED_SYNC_RESOURCES),
            severity="medium",
            detail="Se espera sync exitoso de productos, ventas, payment links, cupones y Club.",
            evidence=[
                f"successful_sync_resources={metrics['successful_sync_resources']}",
                f"target={len(REQUIRED_SYNC_RESOURCES)}",
            ],
            warning=metrics["successful_sync_resources"] < len(REQUIRED_SYNC_RESOURCES),
        ),
        _check(
            key="no_failed_webhooks",
            label="Sin webhooks fallidos activos",
            passed=metrics["failed_webhooks"] == 0,
            severity="critical",
            detail="No deben quedar webhooks rejected/unresolved/replay_requested antes del release.",
            evidence=[f"failed_webhooks={metrics['failed_webhooks']}"],
        ),
        _check(
            key="no_failed_sync_runs",
            label="Sin sync fallido activo",
            passed=metrics["failed_sync_runs"] == 0,
            severity="high",
            detail="Los sync failed/rate_limited deben revisarse antes del release.",
            evidence=[f"failed_sync_runs={metrics['failed_sync_runs']}"],
        ),
        _check(
            key="reconciliation_reviewed",
            label="Reconciliacion revisada",
            passed=metrics["open_reconciliation_issues"] == 0,
            severity="high",
            detail="La cola de diferencias debe quedar vacia o justificada antes de produccion.",
            evidence=[
                f"open_reconciliation_issues={metrics['open_reconciliation_issues']}",
                f"critical_reconciliation_issues={metrics['critical_reconciliation_issues']}",
            ],
            warning=metrics["open_reconciliation_issues"] > 0 and metrics["critical_reconciliation_issues"] == 0,
        ),
    ]
    alerts = _build_alerts(status=status, metrics=metrics)
    blocking_failures = [item for item in checklist if item.required and item.status == "failed"]
    critical_alerts = [alert for alert in alerts if alert.status == "active" and alert.severity == "critical"]
    warnings = [item for item in checklist if item.status in {"manual", "warning"}]
    if blocking_failures or critical_alerts:
        overall_status: Literal["blocked", "needs_attention", "ready"] = "blocked"
    elif warnings or alerts:
        overall_status = "needs_attention"
    else:
        overall_status = "ready"

    return HotmartReleaseReadinessResponse(
        workspace_id=workspace_id,
        environment=env,  # type: ignore[arg-type]
        generated_at=utc_now(),
        overall_status=overall_status,
        release_candidate=overall_status == "ready",
        metrics=metrics,
        checklist=checklist,
        alerts=alerts,
        runbook=_release_runbook(),
    )
