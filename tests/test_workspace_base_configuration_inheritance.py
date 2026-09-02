from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.core.config import get_settings
from app.models import (
    GovernancePolicyRecord,
    LLMRuntimeSettingsUpdateRequest,
    PlatformRole,
    PlatformRoleAssignmentRecord,
    CommercialQuotaWorkspaceOverrideRecord,
    RuntimeFeatureFlagRecord,
    RuntimeGovernanceScopeType,
    RuntimeSettingsAuditRecord,
    UserRecord,
    WorkspaceRecord,
    WorkspaceRuntimeSettingsRecord,
    WorkflowTemplateRecord,
)
from app.services.auth_service import hash_password
from app.services.llm_runtime.runtime_settings_service import persist_workspace_runtime_settings
from app.services.workspace_access import ensure_personal_workspace
from app.services.workspace_bootstrap import (
    FEATURE_FLAG_REACT_RUNTIME,
    WORKSPACE_BASE_INHERITANCE_EVENT,
    apply_workspace_bootstrap,
)


def _build_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _seed_user(session: Session, *, email: str, full_name: str = "User") -> UserRecord:
    user = UserRecord(
        email=email,
        full_name=full_name,
        password_hash=hash_password("LeanBuilder123!"),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def test_new_workspace_inherits_platform_admin_workspace_configuration() -> None:
    settings = get_settings()
    original_admin_email = settings.local_admin_email
    settings.local_admin_email = "platform-admin@leanbuilder.local"
    try:
        engine = _build_engine()
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            admin = _seed_user(session, email=settings.local_admin_email, full_name="Platform Admin")
            admin_workspace = WorkspaceRecord(
                name="Platform Admin Workspace",
                slug="platform-admin-workspace",
                created_by_user_id=admin.id,
            )
            session.add(admin_workspace)
            session.flush()
            admin.default_workspace_id = admin_workspace.id
            session.add(admin)
            session.add(PlatformRoleAssignmentRecord(user_id=admin.id, role=PlatformRole.platform_admin))
            session.commit()

            apply_workspace_bootstrap(session, admin_workspace.id)
            react_flag = session.exec(
                select(RuntimeFeatureFlagRecord).where(
                    RuntimeFeatureFlagRecord.workspace_id == admin_workspace.id,
                    RuntimeFeatureFlagRecord.flag_key == FEATURE_FLAG_REACT_RUNTIME,
                )
            ).one()
            react_flag.enabled = True
            session.add(react_flag)

            template = session.exec(
                select(WorkflowTemplateRecord).where(
                    WorkflowTemplateRecord.workspace_id == admin_workspace.id,
                    WorkflowTemplateRecord.template_key == "single_agent_linear",
                )
            ).one()
            template.label = "Plantilla maestra lineal"
            template.workflow_profile = {"mode": "admin-template"}
            session.add(template)

            policy = session.exec(
                select(GovernancePolicyRecord).where(
                    GovernancePolicyRecord.workspace_id == admin_workspace.id,
                    GovernancePolicyRecord.policy_key == "mandatory_gate_for_side_effects",
                )
            ).one()
            policy.policy_payload = {"requires_platform_approval": True}
            session.add(policy)
            session.add(
                CommercialQuotaWorkspaceOverrideRecord(
                    workspace_id=admin_workspace.id,
                    product_key="acp",
                    free_units_override=3,
                    checkout_required_on_zero_balance_override=False,
                    fifo_auto_approval_enabled_override=True,
                    notes="Plantilla comercial maestra",
                )
            )

            persist_workspace_runtime_settings(
                session,
                admin_workspace.id,
                payload=LLMRuntimeSettingsUpdateRequest.model_validate(
                    {
                        "active_provider": "deepseek",
                        "agent_execution_backend": "provider_native",
                        "knowledge_access_backend": "workspace_staged",
                        "uses_platform_credentials": False,
                        "openai": {
                            "fast_model": "gpt-5.4-mini",
                            "reasoning_model": "gpt-5.5",
                            "reasoning_effort": "low",
                        },
                        "deepseek": {
                            "base_url": "https://api.deepseek.com",
                            "fast_model": "deepseek-chat",
                            "reasoning_model": "deepseek-reasoner",
                            "reasoning_effort": "high",
                        },
                        "codex_local": {
                            "command": "codex",
                            "model": "gpt-5.5",
                            "profile": "admin-template",
                            "cost_policy": "hybrid",
                            "timeout_ms": 180000,
                            "max_concurrency": 1,
                            "runner_id": "platform",
                            "auth_mode": "auto",
                            "fallback_models": [],
                            "primary_agents": [],
                            "shadow_agents": [],
                            "staged_agents": [],
                        },
                    }
                ),
                actor_user_id=admin.id,
                mirror_legacy_runtime=False,
            )

            user = _seed_user(session, email="new-user@leanbuilder.local", full_name="New User")
            new_context = ensure_personal_workspace(session, user, default_name="Cliente Nuevo")

            inherited_flag = session.exec(
                select(RuntimeFeatureFlagRecord).where(
                    RuntimeFeatureFlagRecord.workspace_id == new_context.workspace.id,
                    RuntimeFeatureFlagRecord.flag_key == FEATURE_FLAG_REACT_RUNTIME,
                )
            ).one()
            inherited_template = session.exec(
                select(WorkflowTemplateRecord).where(
                    WorkflowTemplateRecord.workspace_id == new_context.workspace.id,
                    WorkflowTemplateRecord.template_key == "single_agent_linear",
                )
            ).one()
            inherited_policy = session.exec(
                select(GovernancePolicyRecord).where(
                    GovernancePolicyRecord.workspace_id == new_context.workspace.id,
                    GovernancePolicyRecord.policy_key == "mandatory_gate_for_side_effects",
                )
            ).one()
            inherited_runtime = session.exec(
                select(WorkspaceRuntimeSettingsRecord).where(
                    WorkspaceRuntimeSettingsRecord.workspace_id == new_context.workspace.id,
                    WorkspaceRuntimeSettingsRecord.is_active == True,  # noqa: E712
                )
            ).one()
            inherited_quota_override = session.exec(
                select(CommercialQuotaWorkspaceOverrideRecord).where(
                    CommercialQuotaWorkspaceOverrideRecord.workspace_id == new_context.workspace.id,
                    CommercialQuotaWorkspaceOverrideRecord.product_key == "acp",
                )
            ).one()
            audit_row = session.exec(
                select(RuntimeSettingsAuditRecord).where(
                    RuntimeSettingsAuditRecord.scope_type == RuntimeGovernanceScopeType.workspace,
                    RuntimeSettingsAuditRecord.scope_id == str(new_context.workspace.id),
                    RuntimeSettingsAuditRecord.change_type == WORKSPACE_BASE_INHERITANCE_EVENT,
                )
            ).one()
    finally:
        settings.local_admin_email = original_admin_email

    assert inherited_flag.enabled is True
    assert inherited_template.label == "Plantilla maestra lineal"
    assert inherited_template.workflow_profile == {"mode": "admin-template"}
    assert inherited_policy.policy_payload == {"requires_platform_approval": True}
    assert inherited_quota_override.free_units_override == 3
    assert inherited_quota_override.checkout_required_on_zero_balance_override is False
    assert inherited_quota_override.notes == "Plantilla comercial maestra"
    assert inherited_runtime.active_provider.value == "deepseek"
    assert inherited_runtime.provider_overrides["deepseek"]["reasoning_model"] == "deepseek-reasoner"
    assert inherited_runtime.uses_platform_credentials is True
    assert (
        audit_row.after_payload_redacted["secret_policy"]
        == "provider secrets are not copied between workspaces; inherited runtime is forced to platform-managed credentials"
    )
    assert audit_row.after_payload_redacted["runtime_policy"] == "LLM runtime parameters are copied from the Platform Admin template workspace when present"


def test_new_workspace_falls_back_to_default_bootstrap_without_platform_admin_template() -> None:
    engine = _build_engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        user = _seed_user(session, email="standalone@leanbuilder.local", full_name="Standalone")
        context = ensure_personal_workspace(session, user, default_name="Standalone Workspace")
        flags = session.exec(
            select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.workspace_id == context.workspace.id)
        ).all()
        templates = session.exec(
            select(WorkflowTemplateRecord).where(WorkflowTemplateRecord.workspace_id == context.workspace.id)
        ).all()
        policies = session.exec(
            select(GovernancePolicyRecord).where(GovernancePolicyRecord.workspace_id == context.workspace.id)
        ).all()

    assert any(flag.flag_key == FEATURE_FLAG_REACT_RUNTIME for flag in flags)
    assert templates
    assert policies
