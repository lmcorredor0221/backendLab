from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CommerceProviderDefinitionResponse,
    CommerceProviderProductMappingRecord,
    CommerceProviderReadinessCheckResponse,
    CommerceProviderReadinessResponse,
    utc_now,
)
from app.services.commerce_provider_registry import list_commerce_provider_definitions
from app.services.commerce_provider_secrets import build_commerce_provider_status
from app.services.commerce_provider_utils import (
    normalize_commerce_provider_environment,
    normalize_commerce_provider_key,
)


def list_commerce_providers() -> list[CommerceProviderDefinitionResponse]:
    return [
        CommerceProviderDefinitionResponse(
            provider_key=definition.provider_key,
            display_name=definition.display_name,
            capabilities=list(definition.capabilities),
            default_environment=definition.default_environment,  # type: ignore[arg-type]
        )
        for definition in list_commerce_provider_definitions()
    ]


def build_commerce_provider_readiness(
    session: Session,
    *,
    workspace_id: UUID,
    provider_key: str,
    environment: str = "sandbox",
) -> CommerceProviderReadinessResponse:
    provider = normalize_commerce_provider_key(provider_key)
    env = normalize_commerce_provider_environment(environment)
    status = build_commerce_provider_status(
        session,
        workspace_id=workspace_id,
        provider_key=provider,
        environment=env,
    )
    checks: list[CommerceProviderReadinessCheckResponse] = []
    checks.append(
        CommerceProviderReadinessCheckResponse(
            key="provider_enabled",
            label="Provider habilitado",
            status="ok" if status.enabled else "blocking",
            detail="El provider puede recibir checkouts." if status.enabled else "Activa el provider antes de dirigir compras.",
        )
    )
    required_secret_kinds = {"rebill": {"secret_key", "webhook_signing_secret"}}.get(provider, set())
    configured_secret_kinds = {item.secret_kind for item in status.secret_statuses if item.configured}
    for secret_kind in sorted(required_secret_kinds):
        checks.append(
            CommerceProviderReadinessCheckResponse(
                key=f"secret:{secret_kind}",
                label=f"Secreto {secret_kind}",
                status="ok" if secret_kind in configured_secret_kinds else "blocking",
                detail="Configurado." if secret_kind in configured_secret_kinds else "Falta configurar este secreto.",
            )
        )
    if "webhooks" in status.capabilities:
        checks.append(
            CommerceProviderReadinessCheckResponse(
                key="webhook_public_url",
                label="Webhook publico",
                status="ok" if status.webhook_public_url else "blocking",
                detail=status.webhook_public_url or "Configura la URL publica del webhook.",
            )
        )
    mapping_count = session.exec(
        select(CommerceProviderProductMappingRecord).where(
            CommerceProviderProductMappingRecord.workspace_id == workspace_id,
            CommerceProviderProductMappingRecord.provider_key == provider,
            CommerceProviderProductMappingRecord.environment == env,
            CommerceProviderProductMappingRecord.is_active == True,  # noqa: E712
        )
    ).all()
    checks.append(
        CommerceProviderReadinessCheckResponse(
            key="active_mappings",
            label="Mappings activos",
            status="ok" if mapping_count else ("warning" if provider == "sandbox" else "blocking"),
            detail=f"{len(mapping_count)} mapping(s) activos." if mapping_count else "Crea al menos un mapping por producto/paquete LAB.",
        )
    )
    ready = all(check.status != "blocking" for check in checks)
    return CommerceProviderReadinessResponse(
        workspace_id=workspace_id,
        provider_key=provider,
        environment=env,  # type: ignore[arg-type]
        ready=ready,
        status="ready" if ready else "blocked",
        checks=checks,
        checked_at=utc_now(),
    )
