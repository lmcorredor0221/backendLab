from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    LLMBudgetPeriodType,
    LLMBudgetPolicyRecord,
    LLMBudgetScopeType,
    LLMFinOpsAlertRecord,
    LLMUsageLedgerRecord,
    utc_now,
)
from app.services.llm_finops.budget_service import LLMBudgetService, resolve_budget_period


class LLMFinOpsAlertService:
    def __init__(
        self,
        *,
        budget_service: LLMBudgetService | None = None,
        fallback_count_threshold: int = 3,
        high_cost_call_threshold: float = 10.0,
    ) -> None:
        self._budget_service = budget_service or LLMBudgetService()
        self._fallback_count_threshold = max(1, int(fallback_count_threshold or 3))
        self._high_cost_call_threshold = max(0.0, float(high_cost_call_threshold or 0.0))

    def sync_budget_threshold_alerts(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        now: datetime | None = None,
    ) -> list[LLMFinOpsAlertRecord]:
        alerts: list[LLMFinOpsAlertRecord] = []
        for policy in self._budget_service.list_policies(session, workspace_id=workspace_id):
            evaluation = self._budget_service.evaluate_policy(session, policy, now=now)
            for threshold in evaluation.thresholds_reached:
                alerts.append(
                    self._get_or_create_alert(
                        session,
                        workspace_id=workspace_id,
                        alert_key=f"budget_threshold:{policy.id}:{threshold:g}",
                        alert_type="budget_threshold",
                        severity=_severity_for_threshold(threshold),
                        title=f"LLM budget threshold {threshold:g}% reached",
                        message=(
                            f"Budget {policy.policy_key} consumed {evaluation.percent_consumed:g}% "
                            f"of {evaluation.limit_amount:g} {evaluation.currency}."
                        ),
                        period_start=evaluation.period_start,
                        period_end=evaluation.period_end,
                        budget_policy_id=policy.id,
                        scope_type=evaluation.scope_type,
                        scope_value=evaluation.scope_value,
                        provider_key=policy.provider_key,
                        model_name=policy.model_name,
                        stage=policy.stage,
                        threshold_percent=threshold,
                        consumed_amount=evaluation.consumed_amount,
                        limit_amount=evaluation.limit_amount,
                        currency=evaluation.currency,
                        evidence={
                            "percent_consumed": evaluation.percent_consumed,
                            "thresholds_reached": evaluation.thresholds_reached,
                            "call_count": evaluation.call_count,
                            "total_tokens": evaluation.total_tokens,
                        },
                    )
                )
        return alerts

    def sync_operational_alerts(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        now: datetime | None = None,
        period_type: LLMBudgetPeriodType = LLMBudgetPeriodType.monthly,
        allowed_models: set[str] | None = None,
        high_cost_call_threshold: float | None = None,
    ) -> list[LLMFinOpsAlertRecord]:
        period_start, period_end = _operational_period(workspace_id=workspace_id, period_type=period_type, now=now)
        records = _usage_records(session, workspace_id=workspace_id, period_start=period_start, period_end=period_end)
        alerts: list[LLMFinOpsAlertRecord] = []
        alerts.extend(
            self._sync_repeated_fallback_alerts(
                session,
                workspace_id=workspace_id,
                period_start=period_start,
                period_end=period_end,
                records=records,
            )
        )
        alerts.extend(
            self._sync_model_policy_alerts(
                session,
                workspace_id=workspace_id,
                period_start=period_start,
                period_end=period_end,
                records=records,
                allowed_models=allowed_models,
            )
        )
        alerts.extend(
            self._sync_high_cost_call_alerts(
                session,
                workspace_id=workspace_id,
                period_start=period_start,
                period_end=period_end,
                records=records,
                threshold=(
                    self._high_cost_call_threshold
                    if high_cost_call_threshold is None
                    else max(0.0, float(high_cost_call_threshold))
                ),
            )
        )
        return alerts

    def sync_alerts(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        now: datetime | None = None,
        allowed_models: set[str] | None = None,
    ) -> list[LLMFinOpsAlertRecord]:
        return [
            *self.sync_budget_threshold_alerts(session, workspace_id=workspace_id, now=now),
            *self.sync_operational_alerts(
                session,
                workspace_id=workspace_id,
                now=now,
                allowed_models=allowed_models,
            ),
        ]

    def list_alerts(
        self,
        session: Session,
        *,
        workspace_id: UUID | None,
        status: str = "active",
        limit: int = 100,
    ) -> list[LLMFinOpsAlertRecord]:
        statement = select(LLMFinOpsAlertRecord)
        if workspace_id is not None:
            statement = statement.where(LLMFinOpsAlertRecord.workspace_id == workspace_id)
        normalized_status = str(status or "").strip()
        if normalized_status:
            statement = statement.where(LLMFinOpsAlertRecord.status == normalized_status)
        return list(
            session.exec(
                statement.order_by(LLMFinOpsAlertRecord.created_at.desc()).limit(max(1, min(500, limit)))
            ).all()
        )

    def _sync_repeated_fallback_alerts(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        period_start: datetime,
        period_end: datetime,
        records: list[LLMUsageLedgerRecord],
    ) -> list[LLMFinOpsAlertRecord]:
        grouped: dict[tuple[str, str, str], list[LLMUsageLedgerRecord]] = defaultdict(list)
        for record in records:
            if record.fallback_used:
                grouped[(record.provider_key, record.model_name, record.stage)].append(record)
        alerts: list[LLMFinOpsAlertRecord] = []
        for (provider_key, model_name, stage), items in grouped.items():
            if len(items) < self._fallback_count_threshold:
                continue
            alerts.append(
                self._get_or_create_alert(
                    session,
                    workspace_id=workspace_id,
                    alert_key=f"fallback_repeated:{provider_key}:{model_name}:{stage}",
                    alert_type="fallback_repeated",
                    severity="high",
                    title="Repeated LLM fallback detected",
                    message=f"{len(items)} calls used fallback for {provider_key}/{model_name} in {stage}.",
                    period_start=period_start,
                    period_end=period_end,
                    scope_type=LLMBudgetScopeType.provider.value,
                    scope_value=provider_key,
                    provider_key=provider_key,
                    model_name=model_name,
                    stage=stage,
                    consumed_amount=float(len(items)),
                    limit_amount=float(self._fallback_count_threshold),
                    evidence={
                        "fallback_count": len(items),
                        "request_ids": [item.request_id for item in items[:10] if item.request_id],
                    },
                )
            )
        return alerts

    def _sync_model_policy_alerts(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        period_start: datetime,
        period_end: datetime,
        records: list[LLMUsageLedgerRecord],
        allowed_models: set[str] | None,
    ) -> list[LLMFinOpsAlertRecord]:
        allowed = _normalize_allowed_models(allowed_models) or _allowed_models_from_policies(session, workspace_id)
        if not allowed:
            return []
        grouped: dict[tuple[str, str], list[LLMUsageLedgerRecord]] = defaultdict(list)
        for record in records:
            if not _model_is_allowed(record, allowed):
                grouped[(record.provider_key, record.model_name)].append(record)
        alerts: list[LLMFinOpsAlertRecord] = []
        for (provider_key, model_name), items in grouped.items():
            alerts.append(
                self._get_or_create_alert(
                    session,
                    workspace_id=workspace_id,
                    alert_key=f"model_outside_policy:{provider_key}:{model_name}",
                    alert_type="model_outside_policy",
                    severity="high",
                    title="LLM model outside policy",
                    message=f"{len(items)} calls used {provider_key}/{model_name}, which is outside the model policy.",
                    period_start=period_start,
                    period_end=period_end,
                    scope_type=LLMBudgetScopeType.model.value,
                    scope_value=model_name,
                    provider_key=provider_key,
                    model_name=model_name,
                    consumed_amount=float(len(items)),
                    limit_amount=0.0,
                    evidence={
                        "allowed_models": sorted(allowed),
                        "call_count": len(items),
                        "request_ids": [item.request_id for item in items[:10] if item.request_id],
                    },
                )
            )
        return alerts

    def _sync_high_cost_call_alerts(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        period_start: datetime,
        period_end: datetime,
        records: list[LLMUsageLedgerRecord],
        threshold: float,
    ) -> list[LLMFinOpsAlertRecord]:
        if threshold <= 0:
            return []
        alerts: list[LLMFinOpsAlertRecord] = []
        for record in records:
            if record.cost_total < threshold:
                continue
            alerts.append(
                self._get_or_create_alert(
                    session,
                    workspace_id=workspace_id,
                    alert_key=f"high_cost_call:{record.id}",
                    alert_type="high_cost_call",
                    severity="medium" if record.cost_total < threshold * 2 else "high",
                    title="High-cost LLM call detected",
                    message=f"Call {record.request_id or record.id} cost {record.cost_total:g} {record.currency}.",
                    period_start=period_start,
                    period_end=period_end,
                    usage_record_id=record.id,
                    scope_type=LLMBudgetScopeType.model.value,
                    scope_value=record.model_name,
                    provider_key=record.provider_key,
                    model_name=record.model_name,
                    stage=record.stage,
                    consumed_amount=record.cost_total,
                    limit_amount=threshold,
                    currency=record.currency,
                    evidence={
                        "request_id": record.request_id,
                        "total_tokens": record.total_tokens,
                        "duration_ms": record.duration_ms,
                    },
                )
            )
        return alerts

    def _get_or_create_alert(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        alert_key: str,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        period_start: datetime,
        period_end: datetime,
        scope_type: str,
        scope_value: str,
        provider_key: str = "",
        model_name: str = "",
        stage: str = "",
        threshold_percent: float = 0.0,
        consumed_amount: float = 0.0,
        limit_amount: float = 0.0,
        currency: str = "USD",
        budget_policy_id: UUID | None = None,
        usage_record_id: UUID | None = None,
        evidence: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMFinOpsAlertRecord:
        existing = session.exec(
            select(LLMFinOpsAlertRecord).where(
                LLMFinOpsAlertRecord.workspace_id == workspace_id,
                LLMFinOpsAlertRecord.alert_key == alert_key,
                LLMFinOpsAlertRecord.period_start == period_start,
                LLMFinOpsAlertRecord.period_end == period_end,
            )
        ).first()
        if existing is not None:
            existing.severity = severity
            existing.title = title
            existing.message = message
            existing.status = "active"
            existing.consumed_amount = round(max(0.0, float(consumed_amount or 0.0)), 8)
            existing.limit_amount = round(max(0.0, float(limit_amount or 0.0)), 8)
            existing.evidence = evidence or {}
            existing.updated_at = utc_now()
            session.add(existing)
            session.commit()
            session.refresh(existing)
            return existing

        record = LLMFinOpsAlertRecord(
            workspace_id=workspace_id,
            budget_policy_id=budget_policy_id,
            usage_record_id=usage_record_id,
            alert_key=alert_key,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            status="active",
            scope_type=scope_type,
            scope_value=str(scope_value or ""),
            provider_key=str(provider_key or ""),
            model_name=str(model_name or ""),
            stage=str(stage or ""),
            threshold_percent=round(max(0.0, float(threshold_percent or 0.0)), 4),
            period_start=period_start,
            period_end=period_end,
            consumed_amount=round(max(0.0, float(consumed_amount or 0.0)), 8),
            limit_amount=round(max(0.0, float(limit_amount or 0.0)), 8),
            currency=(str(currency or "USD").strip() or "USD").upper(),
            evidence=evidence or {},
            metadata_payload=metadata or {},
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record


def _operational_period(
    *,
    workspace_id: UUID,
    period_type: LLMBudgetPeriodType,
    now: datetime | None,
) -> tuple[datetime, datetime]:
    policy = LLMBudgetPolicyRecord(
        workspace_id=workspace_id,
        scope_type=LLMBudgetScopeType.workspace,
        period_type=period_type,
        limit_amount=1,
    )
    return resolve_budget_period(policy, now=now)


def _usage_records(
    session: Session,
    *,
    workspace_id: UUID,
    period_start: datetime,
    period_end: datetime,
) -> list[LLMUsageLedgerRecord]:
    return list(
        session.exec(
            select(LLMUsageLedgerRecord).where(
                LLMUsageLedgerRecord.workspace_id == workspace_id,
                LLMUsageLedgerRecord.started_at >= _as_naive_utc(period_start),
                LLMUsageLedgerRecord.started_at < _as_naive_utc(period_end),
            )
        ).all()
    )


def _allowed_models_from_policies(session: Session, workspace_id: UUID) -> set[str]:
    allowed: set[str] = set()
    policies = session.exec(
        select(LLMBudgetPolicyRecord).where(
            LLMBudgetPolicyRecord.workspace_id == workspace_id,
            LLMBudgetPolicyRecord.is_active == True,  # noqa: E712
        )
    ).all()
    for policy in policies:
        metadata = policy.metadata_payload or {}
        allowed.update(_normalize_allowed_models(set(metadata.get("allowed_models", []))))
        allowed.update(_normalize_allowed_models(set(metadata.get("allowed_provider_models", []))))
        if _enum_value(policy.scope_type) == LLMBudgetScopeType.model.value:
            allowed.update(_normalize_allowed_models({policy.model_name, policy.scope_value}))
    return allowed


def _normalize_allowed_models(values: set[str] | None) -> set[str]:
    return {str(value).strip().lower() for value in values or set() if str(value or "").strip()}


def _model_is_allowed(record: LLMUsageLedgerRecord, allowed: set[str]) -> bool:
    model_name = str(record.model_name or "").strip().lower()
    provider_model = f"{str(record.provider_key or '').strip().lower()}:{model_name}"
    return model_name in allowed or provider_model in allowed


def _severity_for_threshold(threshold: float) -> str:
    if threshold >= 100:
        return "critical"
    if threshold >= 95:
        return "high"
    if threshold >= 80:
        return "medium"
    return "low"


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "").strip()


def _as_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
