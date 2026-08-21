from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field as PydanticField
from sqlmodel import Session

from app.db import get_session
from app.models import (
    LLMBudgetPeriodType,
    LLMBudgetPolicyRecord,
    LLMBudgetScopeType,
    LLMFinOpsAlertRecord,
    LLMUsageLedgerRecord,
    UserRecord,
)
from app.services.auth_service import get_current_user
from app.services.llm_finops.analytics_service import LLMUsageAnalyticsFilters, LLMUsageAnalyticsService
from app.services.llm_finops.alert_service import LLMFinOpsAlertService
from app.services.llm_finops.budget_service import LLMBudgetService
from app.services.runtime_access_control import ensure_workspace_runtime_admin
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context


router = APIRouter(prefix="/finops/llm", tags=["llm-finops"])


class LLMBudgetPolicyCreateRequest(BaseModel):
    policy_key: str = ""
    name: str = ""
    description: str = ""
    scope_type: LLMBudgetScopeType = LLMBudgetScopeType.workspace
    scope_value: str = ""
    user_id: UUID | None = None
    project_id: UUID | None = None
    initiative_id: UUID | None = None
    stage: str = ""
    provider_key: str = ""
    model_name: str = ""
    period_type: LLMBudgetPeriodType = LLMBudgetPeriodType.monthly
    custom_period_start: datetime | None = None
    custom_period_end: datetime | None = None
    limit_amount: float = PydanticField(default=0, ge=0)
    currency: str = "USD"
    threshold_percentages: list[float] = PydanticField(default_factory=lambda: [50.0, 80.0, 95.0, 100.0])
    hard_limit_percent: float = PydanticField(default=100, ge=0)
    metadata: dict[str, object] = PydanticField(default_factory=dict)


class LLMBudgetPolicyPatchRequest(BaseModel):
    policy_key: str | None = None
    name: str | None = None
    description: str | None = None
    scope_type: LLMBudgetScopeType | None = None
    scope_value: str | None = None
    user_id: UUID | None = None
    project_id: UUID | None = None
    initiative_id: UUID | None = None
    stage: str | None = None
    provider_key: str | None = None
    model_name: str | None = None
    period_type: LLMBudgetPeriodType | None = None
    custom_period_start: datetime | None = None
    custom_period_end: datetime | None = None
    limit_amount: float | None = PydanticField(default=None, ge=0)
    currency: str | None = None
    threshold_percentages: list[float] | None = None
    hard_limit_percent: float | None = PydanticField(default=None, ge=0)
    is_active: bool | None = None
    metadata: dict[str, object] | None = None


@router.get("/budgets")
def list_llm_budgets_route(
    include_inactive: bool = Query(default=False),
    include_evaluations: bool = Query(default=True),
    as_of: datetime | None = None,
    db: Session = Depends(get_session),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    service = LLMBudgetService()
    policies = service.list_policies(
        db,
        workspace_id=workspace_context.workspace.id,
        include_inactive=include_inactive,
    )
    items = [
        _budget_payload(
            policy,
            evaluation=service.evaluate_policy(db, policy, now=as_of) if include_evaluations else None,
        )
        for policy in policies
    ]
    return {"items": items, "count": len(items)}


@router.post("/budgets", status_code=status.HTTP_201_CREATED)
def create_llm_budget_route(
    payload: LLMBudgetPolicyCreateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    _ensure_finops_admin(db, current_user=current_user, workspace_context=workspace_context)
    service = LLMBudgetService()
    policy = service.create_policy(
        db,
        _budget_policy_from_payload(
            payload,
            workspace_id=workspace_context.workspace.id,
            actor_user_id=current_user.id,
        ),
    )
    return _budget_payload(policy, evaluation=service.evaluate_policy(db, policy))


@router.patch("/budgets/{budget_id}")
def patch_llm_budget_route(
    budget_id: UUID,
    payload: LLMBudgetPolicyPatchRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    _ensure_finops_admin(db, current_user=current_user, workspace_context=workspace_context)
    service = LLMBudgetService()
    policy = service.get_policy(db, workspace_id=workspace_context.workspace.id, policy_id=budget_id)
    if policy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No existe el presupuesto solicitado.")
    updates = payload.model_dump(exclude_unset=True)
    if "metadata" in updates:
        updates["metadata_payload"] = updates.pop("metadata")
    updates["updated_by_user_id"] = current_user.id
    policy = service.update_policy(db, policy, updates)
    return _budget_payload(policy, evaluation=service.evaluate_policy(db, policy))


@router.get("/alerts")
def list_llm_finops_alerts_route(
    status_filter: str = Query(default="active", alias="status"),
    sync: bool = Query(default=True),
    as_of: datetime | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_session),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    service = LLMFinOpsAlertService()
    if sync:
        service.sync_alerts(db, workspace_id=workspace_context.workspace.id, now=as_of)
    rows = service.list_alerts(
        db,
        workspace_id=workspace_context.workspace.id,
        status=status_filter,
        limit=limit,
    )
    return {"items": [_alert_payload(item) for item in rows], "count": len(rows)}


@router.get("/usage")
def list_llm_usage_route(
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    user_id: UUID | None = None,
    session_id: UUID | None = None,
    project_id: UUID | None = None,
    initiative_id: UUID | None = None,
    stage: str = "",
    agent_key: str = "",
    capability_key: str = "",
    provider_key: str = "",
    model_name: str = "",
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_session),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    filters = _filters(
        workspace_context=workspace_context,
        started_from=started_from,
        started_to=started_to,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        initiative_id=initiative_id,
        stage=stage,
        agent_key=agent_key,
        capability_key=capability_key,
        provider_key=provider_key,
        model_name=model_name,
    )
    rows = LLMUsageAnalyticsService().list_usage(db, filters, limit=limit, offset=offset)
    return {"items": [_usage_payload(item) for item in rows], "count": len(rows), "limit": limit, "offset": offset}


@router.get("/summary")
def get_llm_usage_summary_route(
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    user_id: UUID | None = None,
    session_id: UUID | None = None,
    project_id: UUID | None = None,
    initiative_id: UUID | None = None,
    stage: str = "",
    agent_key: str = "",
    capability_key: str = "",
    provider_key: str = "",
    model_name: str = "",
    db: Session = Depends(get_session),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    return LLMUsageAnalyticsService().summarize(
        db,
        _filters(
            workspace_context=workspace_context,
            started_from=started_from,
            started_to=started_to,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            initiative_id=initiative_id,
            stage=stage,
            agent_key=agent_key,
            capability_key=capability_key,
            provider_key=provider_key,
            model_name=model_name,
        ),
    )


@router.get("/top-consumers")
def get_llm_top_consumers_route(
    dimension: str = Query(default="user_id"),
    limit: int = Query(default=10, ge=1, le=50),
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    user_id: UUID | None = None,
    session_id: UUID | None = None,
    project_id: UUID | None = None,
    initiative_id: UUID | None = None,
    stage: str = "",
    agent_key: str = "",
    capability_key: str = "",
    provider_key: str = "",
    model_name: str = "",
    db: Session = Depends(get_session),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    rows = LLMUsageAnalyticsService().top_consumers(
        db,
        _filters(
            workspace_context=workspace_context,
            started_from=started_from,
            started_to=started_to,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            initiative_id=initiative_id,
            stage=stage,
            agent_key=agent_key,
            capability_key=capability_key,
            provider_key=provider_key,
            model_name=model_name,
        ),
        dimension=dimension,
        limit=limit,
    )
    return {"items": rows, "dimension": dimension, "count": len(rows)}


@router.get("/provider-breakdown")
def get_llm_provider_breakdown_route(
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    user_id: UUID | None = None,
    session_id: UUID | None = None,
    project_id: UUID | None = None,
    initiative_id: UUID | None = None,
    stage: str = "",
    agent_key: str = "",
    capability_key: str = "",
    provider_key: str = "",
    model_name: str = "",
    db: Session = Depends(get_session),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    rows = LLMUsageAnalyticsService().provider_breakdown(
        db,
        _filters(
            workspace_context=workspace_context,
            started_from=started_from,
            started_to=started_to,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            initiative_id=initiative_id,
            stage=stage,
            agent_key=agent_key,
            capability_key=capability_key,
            provider_key=provider_key,
            model_name=model_name,
        ),
    )
    return {"items": rows, "count": len(rows)}


@router.get("/timeseries")
def get_llm_timeseries_route(
    granularity: str = Query(default="day", pattern="^(day|week|month)$"),
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    user_id: UUID | None = None,
    session_id: UUID | None = None,
    project_id: UUID | None = None,
    initiative_id: UUID | None = None,
    stage: str = "",
    agent_key: str = "",
    capability_key: str = "",
    provider_key: str = "",
    model_name: str = "",
    db: Session = Depends(get_session),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> dict[str, object]:
    rows = LLMUsageAnalyticsService().timeseries(
        db,
        _filters(
            workspace_context=workspace_context,
            started_from=started_from,
            started_to=started_to,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            initiative_id=initiative_id,
            stage=stage,
            agent_key=agent_key,
            capability_key=capability_key,
            provider_key=provider_key,
            model_name=model_name,
        ),
        granularity=granularity,
    )
    return {
        "items": rows,
        "count": len(rows),
        "granularity": granularity,
        "availability": {
            "status": "available" if rows else "empty",
            "source": "llm_usage_ledger.started_at",
            "reason": "Buckets temporales calculados desde el ledger LLM filtrado por workspace.",
        },
    }


def _ensure_finops_admin(
    db: Session,
    *,
    current_user: UserRecord,
    workspace_context: WorkspaceAccessContext,
) -> None:
    try:
        ensure_workspace_runtime_admin(db, current_user, workspace_context)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def _budget_policy_from_payload(
    payload: LLMBudgetPolicyCreateRequest,
    *,
    workspace_id: UUID,
    actor_user_id: UUID,
) -> LLMBudgetPolicyRecord:
    data = payload.model_dump()
    metadata = data.pop("metadata", {})
    return LLMBudgetPolicyRecord(
        workspace_id=workspace_id,
        created_by_user_id=actor_user_id,
        updated_by_user_id=actor_user_id,
        metadata_payload=metadata,
        **data,
    )


def _budget_payload(policy: LLMBudgetPolicyRecord, *, evaluation: object | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(policy.id),
        "workspace_id": str(policy.workspace_id),
        "policy_key": policy.policy_key,
        "name": policy.name,
        "description": policy.description,
        "scope_type": _enum_value(policy.scope_type),
        "scope_value": policy.scope_value,
        "user_id": str(policy.user_id) if policy.user_id else None,
        "project_id": str(policy.project_id) if policy.project_id else None,
        "initiative_id": str(policy.initiative_id) if policy.initiative_id else None,
        "stage": policy.stage,
        "provider_key": policy.provider_key,
        "model_name": policy.model_name,
        "period_type": _enum_value(policy.period_type),
        "custom_period_start": _iso_or_none(policy.custom_period_start),
        "custom_period_end": _iso_or_none(policy.custom_period_end),
        "limit_amount": policy.limit_amount,
        "currency": policy.currency,
        "threshold_percentages": list(policy.threshold_percentages),
        "hard_limit_percent": policy.hard_limit_percent,
        "is_active": policy.is_active,
        "metadata": policy.metadata_payload,
        "created_at": policy.created_at.isoformat(),
        "updated_at": policy.updated_at.isoformat(),
    }
    if evaluation is not None and hasattr(evaluation, "as_dict"):
        payload["evaluation"] = evaluation.as_dict()
    return payload


def _alert_payload(record: LLMFinOpsAlertRecord) -> dict[str, object]:
    return {
        "id": str(record.id),
        "workspace_id": str(record.workspace_id),
        "budget_policy_id": str(record.budget_policy_id) if record.budget_policy_id else None,
        "usage_record_id": str(record.usage_record_id) if record.usage_record_id else None,
        "alert_key": record.alert_key,
        "alert_type": record.alert_type,
        "severity": record.severity,
        "title": record.title,
        "message": record.message,
        "status": record.status,
        "scope_type": record.scope_type,
        "scope_value": record.scope_value,
        "provider_key": record.provider_key,
        "model_name": record.model_name,
        "stage": record.stage,
        "threshold_percent": record.threshold_percent,
        "period_start": record.period_start.isoformat(),
        "period_end": record.period_end.isoformat(),
        "consumed_amount": record.consumed_amount,
        "limit_amount": record.limit_amount,
        "currency": record.currency,
        "evidence": record.evidence,
        "metadata": record.metadata_payload,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "resolved_at": _iso_or_none(record.resolved_at),
    }


def _enum_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "")


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _filters(
    *,
    workspace_context: WorkspaceAccessContext,
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    user_id: UUID | None = None,
    session_id: UUID | None = None,
    project_id: UUID | None = None,
    initiative_id: UUID | None = None,
    stage: str = "",
    agent_key: str = "",
    capability_key: str = "",
    provider_key: str = "",
    model_name: str = "",
) -> LLMUsageAnalyticsFilters:
    return LLMUsageAnalyticsFilters(
        workspace_id=workspace_context.workspace.id,
        started_from=started_from,
        started_to=started_to,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        initiative_id=initiative_id,
        stage=stage.strip(),
        agent_key=agent_key.strip(),
        capability_key=capability_key.strip(),
        provider_key=provider_key.strip(),
        model_name=model_name.strip(),
    )


def _usage_payload(record: LLMUsageLedgerRecord) -> dict[str, object]:
    return {
        "id": str(record.id),
        "workspace_id": str(record.workspace_id) if record.workspace_id else None,
        "user_id": str(record.user_id) if record.user_id else None,
        "session_id": str(record.session_id) if record.session_id else None,
        "project_id": str(record.project_id) if record.project_id else None,
        "initiative_id": str(record.initiative_id) if record.initiative_id else None,
        "started_at": record.started_at.isoformat(),
        "stage": record.stage,
        "substage": record.substage,
        "agent_key": record.agent_key,
        "capability_key": record.capability_key,
        "provider_key": record.provider_key,
        "model_name": record.model_name,
        "execution_backend": record.execution_backend,
        "execution_mode": record.execution_mode,
        "request_id": record.request_id,
        "status": record.status,
        "duration_ms": record.duration_ms,
        "queue_wait_ms": record.queue_wait_ms,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "total_tokens": record.total_tokens,
        "cached_input_tokens": record.cached_input_tokens,
        "reasoning_tokens": record.reasoning_tokens,
        "cost_input": record.cost_input,
        "cost_output": record.cost_output,
        "cost_other": record.cost_other,
        "cost_total": record.cost_total,
        "currency": record.currency,
        "pricing_profile_key": record.pricing_profile_key,
        "fallback_used": record.fallback_used,
        "retry_count": record.retry_count,
    }
