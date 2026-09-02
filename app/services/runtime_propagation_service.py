from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    RuntimePropagationItemRecord,
    RuntimePropagationItemResponse,
    RuntimePropagationMode,
    RuntimePropagationRequest,
    RuntimePropagationRunRecord,
    RuntimePropagationRunResponse,
    RuntimePropagationStatus,
    WorkspaceRecord,
    WorkspaceRuntimeSettingsRecord,
    utc_now,
)
from app.services.llm_runtime.runtime_settings_service import (
    persist_platform_runtime_defaults,
    persist_workspace_runtime_settings,
    reset_workspace_runtime_settings,
)


def _active_workspace_runtime_record(
    session: Session,
    *,
    workspace_id: UUID,
) -> WorkspaceRuntimeSettingsRecord | None:
    return session.exec(
        select(WorkspaceRuntimeSettingsRecord).where(
            WorkspaceRuntimeSettingsRecord.workspace_id == workspace_id,
            WorkspaceRuntimeSettingsRecord.is_active == True,  # noqa: E712
        )
    ).first()


def _target_workspaces(
    session: Session,
    *,
    mode: RuntimePropagationMode,
    workspace_ids: list[UUID],
) -> list[WorkspaceRecord]:
    if mode in {RuntimePropagationMode.force_selected, RuntimePropagationMode.reset_to_platform}:
        if not workspace_ids:
            raise ValueError("workspace_ids is required for selected/reset propagation modes.")
        return session.exec(
            select(WorkspaceRecord)
            .where(WorkspaceRecord.id.in_(workspace_ids))
            .order_by(WorkspaceRecord.created_at.asc(), WorkspaceRecord.name.asc())
        ).all()
    return session.exec(select(WorkspaceRecord).order_by(WorkspaceRecord.created_at.asc(), WorkspaceRecord.name.asc())).all()


def _item_response(
    *,
    workspace: WorkspaceRecord,
    record: RuntimePropagationItemRecord,
    had_override: bool,
) -> RuntimePropagationItemResponse:
    return RuntimePropagationItemResponse(
        workspace_id=workspace.id,
        workspace_name=workspace.name,
        previous_workspace_runtime_version=record.previous_workspace_runtime_version,
        next_workspace_runtime_version=record.next_workspace_runtime_version,
        had_override=had_override,
        action=record.action,
        status=record.status,
        error_message=record.error_message,
    )


def _serialize_run(
    run: RuntimePropagationRunRecord,
    *,
    items: list[RuntimePropagationItemResponse],
) -> RuntimePropagationRunResponse:
    return RuntimePropagationRunResponse(
        id=run.id,
        mode=run.mode,
        provider_key=run.provider_key,
        target_scope=run.target_scope,
        dry_run=run.dry_run,
        status=run.status,
        payload_redacted=dict(run.payload_redacted),
        created_at=run.created_at,
        completed_at=run.completed_at,
        items=items,
    )


def propagate_platform_runtime_settings(
    session: Session,
    *,
    payload: RuntimePropagationRequest,
    actor_user_id: UUID | None = None,
) -> RuntimePropagationRunResponse:
    targets = _target_workspaces(session, mode=payload.mode, workspace_ids=payload.workspace_ids)
    missing_targets = set(payload.workspace_ids) - {workspace.id for workspace in targets}
    if missing_targets:
        raise ValueError(f"Unknown workspace ids: {', '.join(sorted(str(item) for item in missing_targets))}")

    run = RuntimePropagationRunRecord(
        actor_user_id=actor_user_id,
        mode=payload.mode,
        provider_key=payload.payload.active_provider,
        target_scope="selected" if payload.workspace_ids else "all",
        payload_redacted=payload.payload.model_dump(mode="json"),
        dry_run=payload.dry_run,
        status=RuntimePropagationStatus.planned,
    )
    session.add(run)
    session.flush()

    if not payload.dry_run:
        persist_platform_runtime_defaults(session, payload.payload, actor_user_id=actor_user_id)
        run = session.get(RuntimePropagationRunRecord, run.id) or run

    responses: list[RuntimePropagationItemResponse] = []
    overall_status = RuntimePropagationStatus.applied if not payload.dry_run else RuntimePropagationStatus.planned

    for workspace in targets:
        active_record = _active_workspace_runtime_record(session, workspace_id=workspace.id)
        previous_version = active_record.version if active_record is not None else None
        had_override = active_record is not None
        action = "inherits_platform_default"
        item_status = RuntimePropagationStatus.planned if payload.dry_run else RuntimePropagationStatus.applied
        next_version = previous_version
        error_message = ""

        try:
            if payload.mode == RuntimePropagationMode.fallback_only:
                if had_override:
                    action = "preserve_workspace_override"
                    item_status = RuntimePropagationStatus.skipped
                else:
                    action = "inherits_platform_default"
            elif payload.mode == RuntimePropagationMode.reset_to_platform:
                action = "reset_workspace_override_to_platform"
                if not had_override:
                    item_status = RuntimePropagationStatus.skipped
                elif not payload.dry_run:
                    reset_workspace_runtime_settings(session, workspace.id, actor_user_id=actor_user_id)
                    next_version = None
            elif payload.mode in {RuntimePropagationMode.force_selected, RuntimePropagationMode.force_all}:
                action = "write_workspace_runtime_settings"
                if not payload.dry_run:
                    persist_workspace_runtime_settings(session, workspace.id, payload.payload, actor_user_id=actor_user_id)
                    refreshed_record = _active_workspace_runtime_record(session, workspace_id=workspace.id)
                    next_version = refreshed_record.version if refreshed_record is not None else None
        except Exception as exc:  # noqa: BLE001 - every item must be auditable.
            item_status = RuntimePropagationStatus.failed
            overall_status = RuntimePropagationStatus.failed
            error_message = str(exc)

        record = RuntimePropagationItemRecord(
            run_id=run.id,
            workspace_id=workspace.id,
            previous_workspace_runtime_version=previous_version,
            next_workspace_runtime_version=next_version,
            action=action,
            status=item_status,
            error_message=error_message,
        )
        session.add(record)
        session.flush()
        responses.append(_item_response(workspace=workspace, record=record, had_override=had_override))

    run.status = overall_status
    run.completed_at = utc_now()
    session.add(run)
    session.commit()
    session.refresh(run)
    return _serialize_run(run, items=responses)
