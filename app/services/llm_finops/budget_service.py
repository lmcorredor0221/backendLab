from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import LLMBudgetPeriodType, LLMBudgetPolicyRecord, LLMBudgetScopeType, LLMUsageLedgerRecord, utc_now


@dataclass(frozen=True)
class LLMBudgetEvaluation:
    policy_id: UUID
    policy_key: str
    workspace_id: UUID
    scope_type: str
    scope_value: str
    period_type: str
    period_start: datetime
    period_end: datetime
    limit_amount: float
    currency: str
    consumed_amount: float
    remaining_amount: float
    percent_consumed: float
    status: str
    thresholds_reached: list[float]
    next_threshold_percent: float | None
    call_count: int
    total_tokens: int

    @property
    def is_warning(self) -> bool:
        return self.status == "warning"

    @property
    def is_hard_limit(self) -> bool:
        return self.status == "hard_limit"

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": str(self.policy_id),
            "policy_key": self.policy_key,
            "workspace_id": str(self.workspace_id),
            "scope_type": self.scope_type,
            "scope_value": self.scope_value,
            "period_type": self.period_type,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "limit_amount": self.limit_amount,
            "currency": self.currency,
            "consumed_amount": self.consumed_amount,
            "remaining_amount": self.remaining_amount,
            "percent_consumed": self.percent_consumed,
            "status": self.status,
            "thresholds_reached": list(self.thresholds_reached),
            "next_threshold_percent": self.next_threshold_percent,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "is_warning": self.is_warning,
            "is_hard_limit": self.is_hard_limit,
        }


class LLMBudgetService:
    def create_policy(self, session: Session, policy: LLMBudgetPolicyRecord) -> LLMBudgetPolicyRecord:
        _normalize_policy(policy)
        session.add(policy)
        session.commit()
        session.refresh(policy)
        return policy

    def get_policy(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        policy_id: UUID,
    ) -> LLMBudgetPolicyRecord | None:
        return session.exec(
            select(LLMBudgetPolicyRecord).where(
                LLMBudgetPolicyRecord.id == policy_id,
                LLMBudgetPolicyRecord.workspace_id == workspace_id,
            )
        ).first()

    def update_policy(
        self,
        session: Session,
        policy: LLMBudgetPolicyRecord,
        updates: dict[str, Any],
    ) -> LLMBudgetPolicyRecord:
        for field_name, value in updates.items():
            if hasattr(policy, field_name):
                setattr(policy, field_name, value)
        _normalize_policy(policy)
        session.add(policy)
        session.commit()
        session.refresh(policy)
        return policy

    def list_policies(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        include_inactive: bool = False,
    ) -> list[LLMBudgetPolicyRecord]:
        statement = select(LLMBudgetPolicyRecord).where(LLMBudgetPolicyRecord.workspace_id == workspace_id)
        if not include_inactive:
            statement = statement.where(LLMBudgetPolicyRecord.is_active == True)  # noqa: E712
        return list(session.exec(statement.order_by(LLMBudgetPolicyRecord.created_at.desc())).all())

    def evaluate_policy(
        self,
        session: Session,
        policy: LLMBudgetPolicyRecord,
        *,
        now: datetime | None = None,
    ) -> LLMBudgetEvaluation:
        _normalize_policy(policy)
        period_start, period_end = resolve_budget_period(policy, now=now)
        records = _matching_usage_records(session, policy, period_start=period_start, period_end=period_end)
        consumed_amount = round(sum(record.cost_total for record in records), 8)
        total_tokens = sum(record.total_tokens for record in records)
        limit_amount = round(max(0.0, float(policy.limit_amount or 0.0)), 8)
        if not policy.is_active:
            status = "inactive"
            percent_consumed = 0.0
            thresholds_reached: list[float] = []
            next_threshold_percent = None
        elif limit_amount <= 0:
            status = "unconfigured"
            percent_consumed = 0.0
            thresholds_reached = []
            next_threshold_percent = None
        else:
            percent_consumed = round((consumed_amount / limit_amount) * 100, 4)
            thresholds = _normalize_thresholds(policy.threshold_percentages)
            thresholds_reached = [threshold for threshold in thresholds if percent_consumed >= threshold]
            next_threshold_percent = next((threshold for threshold in thresholds if percent_consumed < threshold), None)
            if percent_consumed >= max(0.0, float(policy.hard_limit_percent or 100.0)):
                status = "hard_limit"
            elif thresholds_reached:
                status = "warning"
            else:
                status = "ok"
        return LLMBudgetEvaluation(
            policy_id=policy.id,
            policy_key=policy.policy_key,
            workspace_id=policy.workspace_id,
            scope_type=_enum_value(policy.scope_type),
            scope_value=policy.scope_value,
            period_type=_enum_value(policy.period_type),
            period_start=period_start,
            period_end=period_end,
            limit_amount=limit_amount,
            currency=_normalize_currency(policy.currency),
            consumed_amount=consumed_amount,
            remaining_amount=round(max(0.0, limit_amount - consumed_amount), 8),
            percent_consumed=percent_consumed,
            status=status,
            thresholds_reached=thresholds_reached,
            next_threshold_percent=next_threshold_percent,
            call_count=len(records),
            total_tokens=total_tokens,
        )

    def evaluate_active_policies(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        now: datetime | None = None,
    ) -> list[LLMBudgetEvaluation]:
        return [
            self.evaluate_policy(session, policy, now=now)
            for policy in self.list_policies(session, workspace_id=workspace_id)
        ]


def resolve_budget_period(
    policy: LLMBudgetPolicyRecord,
    *,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    reference = _as_naive_utc(now or utc_now())
    period_type = _enum_value(policy.period_type)
    if period_type == LLMBudgetPeriodType.custom.value:
        start = _as_naive_utc(policy.custom_period_start or reference)
        end = _as_naive_utc(policy.custom_period_end or reference)
        if end <= start:
            end = start + timedelta(days=1)
        return start, end
    if period_type == LLMBudgetPeriodType.daily.value:
        start = reference.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)
    if period_type == LLMBudgetPeriodType.weekly.value:
        start = (reference - timedelta(days=reference.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=7)
    start = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        return start, start.replace(year=start.year + 1, month=1)
    return start, start.replace(month=start.month + 1)


def _matching_usage_records(
    session: Session,
    policy: LLMBudgetPolicyRecord,
    *,
    period_start: datetime,
    period_end: datetime,
) -> list[LLMUsageLedgerRecord]:
    statement = (
        select(LLMUsageLedgerRecord)
        .where(LLMUsageLedgerRecord.workspace_id == policy.workspace_id)
        .where(LLMUsageLedgerRecord.started_at >= period_start)
        .where(LLMUsageLedgerRecord.started_at < period_end)
    )
    currency = _normalize_currency(policy.currency)
    if currency:
        statement = statement.where(LLMUsageLedgerRecord.currency == currency)
    statement = _apply_scope_filter(statement, policy)
    statement = _apply_optional_dimension_filters(statement, policy)
    return list(session.exec(statement).all())


def _apply_scope_filter(statement: Any, policy: LLMBudgetPolicyRecord) -> Any:
    scope_type = _enum_value(policy.scope_type)
    if scope_type == LLMBudgetScopeType.workspace.value:
        return statement
    if scope_type == LLMBudgetScopeType.user.value:
        return _where_uuid_or_empty(statement, LLMUsageLedgerRecord.user_id, policy.user_id, policy.scope_value)
    if scope_type == LLMBudgetScopeType.project.value:
        return _where_uuid_or_empty(statement, LLMUsageLedgerRecord.project_id, policy.project_id, policy.scope_value)
    if scope_type == LLMBudgetScopeType.initiative.value:
        return _where_uuid_or_empty(
            statement,
            LLMUsageLedgerRecord.initiative_id,
            policy.initiative_id,
            policy.scope_value,
        )
    if scope_type == LLMBudgetScopeType.stage.value:
        return _where_string_or_empty(statement, LLMUsageLedgerRecord.stage, policy.stage or policy.scope_value)
    if scope_type == LLMBudgetScopeType.provider.value:
        return _where_string_or_empty(statement, LLMUsageLedgerRecord.provider_key, policy.provider_key or policy.scope_value)
    if scope_type == LLMBudgetScopeType.model.value:
        return _where_string_or_empty(statement, LLMUsageLedgerRecord.model_name, policy.model_name or policy.scope_value)
    return statement.where(False)


def _apply_optional_dimension_filters(statement: Any, policy: LLMBudgetPolicyRecord) -> Any:
    for column, value in (
        (LLMUsageLedgerRecord.user_id, policy.user_id),
        (LLMUsageLedgerRecord.project_id, policy.project_id),
        (LLMUsageLedgerRecord.initiative_id, policy.initiative_id),
    ):
        if value is not None:
            statement = statement.where(column == value)
    for column, value in (
        (LLMUsageLedgerRecord.stage, policy.stage),
        (LLMUsageLedgerRecord.provider_key, policy.provider_key),
        (LLMUsageLedgerRecord.model_name, policy.model_name),
    ):
        value = str(value or "").strip()
        if value:
            statement = statement.where(column == value)
    return statement


def _normalize_policy(policy: LLMBudgetPolicyRecord) -> None:
    policy.policy_key = str(policy.policy_key or "").strip() or _default_policy_key(policy)
    policy.name = str(policy.name or "").strip() or policy.policy_key
    policy.description = str(policy.description or "").strip()
    policy.scope_value = _resolve_scope_value(policy)
    policy.stage = str(policy.stage or "").strip()
    policy.provider_key = str(policy.provider_key or "").strip()
    policy.model_name = str(policy.model_name or "").strip()
    policy.currency = _normalize_currency(policy.currency)
    policy.limit_amount = round(max(0.0, float(policy.limit_amount or 0.0)), 8)
    policy.threshold_percentages = _normalize_thresholds(policy.threshold_percentages)
    policy.hard_limit_percent = max(0.0, float(policy.hard_limit_percent or 100.0))
    policy.updated_at = utc_now()


def _default_policy_key(policy: LLMBudgetPolicyRecord) -> str:
    scope_value = _resolve_scope_value(policy) or "workspace"
    return f"{_enum_value(policy.scope_type)}-{scope_value}-{_enum_value(policy.period_type)}".lower()


def _resolve_scope_value(policy: LLMBudgetPolicyRecord) -> str:
    existing = str(policy.scope_value or "").strip()
    if existing:
        return existing
    scope_type = _enum_value(policy.scope_type)
    if scope_type == LLMBudgetScopeType.workspace.value:
        return str(policy.workspace_id)
    if scope_type == LLMBudgetScopeType.user.value and policy.user_id:
        return str(policy.user_id)
    if scope_type == LLMBudgetScopeType.project.value and policy.project_id:
        return str(policy.project_id)
    if scope_type == LLMBudgetScopeType.initiative.value and policy.initiative_id:
        return str(policy.initiative_id)
    if scope_type == LLMBudgetScopeType.stage.value:
        return str(policy.stage or "").strip()
    if scope_type == LLMBudgetScopeType.provider.value:
        return str(policy.provider_key or "").strip()
    if scope_type == LLMBudgetScopeType.model.value:
        return str(policy.model_name or "").strip()
    return ""


def _where_uuid_or_empty(statement: Any, column: Any, typed_value: UUID | None, raw_value: str) -> Any:
    value = typed_value or _parse_uuid(raw_value)
    if value is None:
        return statement.where(False)
    return statement.where(column == value)


def _where_string_or_empty(statement: Any, column: Any, value: str) -> Any:
    normalized = str(value or "").strip()
    if not normalized:
        return statement.where(False)
    return statement.where(column == normalized)


def _parse_uuid(value: str) -> UUID | None:
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _normalize_thresholds(values: list[float] | None) -> list[float]:
    thresholds: set[float] = set()
    for value in values or [50.0, 80.0, 95.0, 100.0]:
        try:
            parsed = round(float(value), 4)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            thresholds.add(parsed)
    return sorted(thresholds)


def _normalize_currency(value: str) -> str:
    return (str(value or "USD").strip() or "USD").upper()


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "").strip()


def _as_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
