from __future__ import annotations

from sqlmodel import Session, select

from app.models import (
    LLMProviderKey,
    PlatformRuntimeProviderRecord,
    WorkspaceRuntimeHealthCheckEntry,
    WorkspaceRuntimeHealthResponse,
    utc_now,
)
from app.services.llm_runtime.runtime_secrets_service import annotate_runtime_settings_with_workspace_secrets
from app.services.llm_runtime.runtime_settings_service import load_effective_runtime_settings
from app.services.runtime_governance_bootstrap import backfill_platform_runtime_governance


def _selected_provider_state(runtime_settings):
    if runtime_settings.active_provider == LLMProviderKey.openai:
        return {
            "label": "OpenAI",
            "configured": runtime_settings.openai.api_key_configured,
            "reachable": runtime_settings.openai.available,
            "secret_source": runtime_settings.openai.secret_source,
            "health_status": runtime_settings.openai.health_status,
            "status_note": runtime_settings.openai.status_note,
        }
    if runtime_settings.active_provider == LLMProviderKey.deepseek:
        return {
            "label": "DeepSeek",
            "configured": runtime_settings.deepseek.api_key_configured,
            "reachable": runtime_settings.deepseek.available,
            "secret_source": runtime_settings.deepseek.secret_source,
            "health_status": runtime_settings.deepseek.health_status,
            "status_note": runtime_settings.deepseek.status_note,
        }
    if runtime_settings.active_provider == LLMProviderKey.antigravity_cli:
        return {
            "label": "Antigravity CLI",
            "configured": bool(runtime_settings.antigravity.executable and runtime_settings.antigravity.model),
            "reachable": runtime_settings.antigravity.available,
            "secret_source": runtime_settings.antigravity.secret_source,
            "health_status": runtime_settings.antigravity.health_status,
            "status_note": runtime_settings.antigravity.status_note,
        }
    return {
        "label": "Codex local",
        "configured": bool(runtime_settings.codex_local.command and runtime_settings.codex_local.model),
        "reachable": runtime_settings.codex_local.available,
        "secret_source": runtime_settings.codex_local.secret_source,
        "health_status": runtime_settings.codex_local.health_status,
        "status_note": runtime_settings.codex_local.status_note,
    }


def build_workspace_runtime_health(
    session: Session,
    workspace_id,
    *,
    mode: str = "health",
) -> WorkspaceRuntimeHealthResponse:
    backfill_platform_runtime_governance(session)
    runtime_settings = annotate_runtime_settings_with_workspace_secrets(
        session,
        workspace_id,
        load_effective_runtime_settings(session, workspace_id),
    )
    provider_state = _selected_provider_state(runtime_settings)
    provider_row = session.exec(
        select(PlatformRuntimeProviderRecord).where(
            PlatformRuntimeProviderRecord.provider_key == runtime_settings.active_provider
        )
    ).first()
    provider_enabled = provider_row.is_enabled if provider_row is not None else False

    checks = [
        WorkspaceRuntimeHealthCheckEntry(
            check_key="provider_enabled",
            label="Provider habilitado por plataforma",
            status="pass" if provider_enabled else "fail",
            detail=runtime_settings.active_provider.value,
        ),
        WorkspaceRuntimeHealthCheckEntry(
            check_key="credential_resolution",
            label="Resolucion de credenciales",
            status="pass" if provider_state["configured"] else "fail",
            detail=f"source={provider_state['secret_source']} health={provider_state['health_status']}",
        ),
        WorkspaceRuntimeHealthCheckEntry(
            check_key="provider_reachability",
            label="Disponibilidad del provider",
            status="pass" if provider_state["reachable"] else "fail",
            detail=provider_state["status_note"],
        ),
        WorkspaceRuntimeHealthCheckEntry(
            check_key="execution_backend",
            label="Backend de ejecucion",
            status="pass",
            detail=runtime_settings.agent_execution_backend.value,
        ),
        WorkspaceRuntimeHealthCheckEntry(
            check_key="knowledge_backend",
            label="Backend de conocimiento",
            status="pass",
            detail=runtime_settings.knowledge_access_backend.value,
        ),
    ]
    overall_status = "healthy" if all(item.status == "pass" for item in checks[:3]) else "degraded"
    if mode == "test" and overall_status == "healthy":
        checks.append(
            WorkspaceRuntimeHealthCheckEntry(
                check_key="dry_run",
                label="Prueba deterministica",
                status="pass",
                detail="El runtime supera el chequeo local de configuracion, secrets y reachability.",
            )
        )
    elif mode == "test":
        checks.append(
            WorkspaceRuntimeHealthCheckEntry(
                check_key="dry_run",
                label="Prueba deterministica",
                status="fail",
                detail="El runtime no puede promoverse a smoke remoto porque fallo el readiness local.",
            )
        )

    return WorkspaceRuntimeHealthResponse(
        workspace_id=workspace_id,
        mode=mode,
        overall_status=overall_status,
        provider_key=runtime_settings.active_provider,
        provider_label=provider_state["label"],
        secret_source=provider_state["secret_source"],
        health_status=provider_state["health_status"],
        uses_platform_credentials=runtime_settings.uses_platform_credentials,
        agent_execution_backend=runtime_settings.agent_execution_backend,
        knowledge_access_backend=runtime_settings.knowledge_access_backend,
        checked_at=utc_now(),
        checks=checks,
    )
