from __future__ import annotations

import re
from typing import Any

from sqlmodel import Session, select

from app.models import CodexLocalCostPolicy, LLMPricingProfile, LLMProviderKey, LLMUsageLedgerRecord, utc_now
from app.services.llm_finops.contracts import LLMUsageCostBreakdown, LLMUsageRecordInput, LLMUsageRecordResult
from app.services.llm_finops.pricing_resolver import PricingResolver


_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)=([^,\s]+)"),
)


class LLMUsageLedgerService:
    def __init__(
        self,
        *,
        pricing_profiles: dict[LLMProviderKey | str, list[LLMPricingProfile]] | None = None,
        pricing_resolver: PricingResolver | None = None,
        local_cost_policy: CodexLocalCostPolicy | None = None,
    ) -> None:
        self._pricing_profiles = (
            _normalize_pricing_profiles(pricing_profiles) if pricing_profiles is not None else None
        )
        self._pricing_resolver = pricing_resolver or PricingResolver()
        self._local_cost_policy = local_cost_policy

    def record_call(self, session: Session, record_input: LLMUsageRecordInput) -> LLMUsageRecordResult:
        existing = self._find_existing_record(session, record_input)
        if existing is not None:
            return LLMUsageRecordResult(
                usage_record_id=existing.id,
                created=False,
                duplicate=True,
                cost=record_input.cost,
                warnings=["Duplicate LLM usage record ignored."],
            )

        context = record_input.context
        usage = record_input.usage
        started_at = record_input.started_at or utc_now()
        cost = self._resolve_cost(record_input, occurred_at=started_at)
        record = LLMUsageLedgerRecord(
            workspace_id=context.workspace_id,
            user_id=context.user_id,
            session_id=context.session_id,
            project_id=context.project_id,
            initiative_id=context.initiative_id,
            stage=context.stage,
            substage=context.substage,
            agent_key=context.agent_key,
            capability_key=record_input.context.capability_key,
            action_key=context.action_key,
            operation_id=context.operation_id,
            parent_run_id=context.parent_run_id,
            correlation_id=context.correlation_id,
            provider_key=record_input.provider_key,
            model_name=record_input.model_name,
            requested_model=record_input.requested_model,
            execution_backend=record_input.execution_backend,
            execution_mode=record_input.execution_mode or context.execution_mode,
            request_id=record_input.request_id or record_input.provider_request_id,
            provider_request_id=record_input.provider_request_id,
            attempt_number=record_input.attempt_number,
            retry_count=record_input.retry_count,
            fallback_used=record_input.fallback_used,
            shadow_provider_key=record_input.shadow_provider_key,
            status=record_input.status.value,
            failure_kind=record_input.failure_kind,
            failure_detail_redacted=_redact_text(record_input.failure_detail)[:400],
            started_at=started_at,
            finished_at=record_input.finished_at,
            duration_ms=record_input.duration_ms,
            queue_wait_ms=record_input.queue_wait_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            other_token_metrics={
                "accepted_prediction_tokens": usage.accepted_prediction_tokens,
                "rejected_prediction_tokens": usage.rejected_prediction_tokens,
                "tool_call_count": usage.tool_call_count,
                "usage_is_estimated": usage.usage_is_estimated,
                "normalization_version": usage.normalization_version,
            },
            provider_metrics=_redact_payload(usage.provider_metrics),
            cost_input=cost.cost_input,
            cost_output=cost.cost_output,
            cost_other=cost.cost_other,
            cost_total=cost.cost_total,
            currency=cost.currency,
            fx_rate=cost.fx_rate,
            pricing_profile_key=cost.pricing_profile_key,
            pricing_snapshot=_redact_payload(cost.pricing_snapshot),
            usage_raw_redacted=_redact_payload(usage.raw_usage),
            prompt_hash=record_input.prompt_hash,
            response_hash=record_input.response_hash,
            schema_validation_status=record_input.schema_validation_status,
            finish_reason=record_input.finish_reason,
            value_signal=record_input.value_signal,
            metadata_payload=_redact_payload(
                {
                    **context.metadata,
                    **record_input.metadata,
                    "cost_warnings": list(cost.warnings),
                }
            ),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return LLMUsageRecordResult(
            usage_record_id=record.id,
            created=True,
            duplicate=False,
            cost=cost,
            warnings=list(cost.warnings),
        )

    def _resolve_cost(
        self,
        record_input: LLMUsageRecordInput,
        *,
        occurred_at: Any,
    ) -> LLMUsageCostBreakdown:
        if self._pricing_profiles is None:
            return record_input.cost
        return self._pricing_resolver.resolve_call_cost(
            provider_key=record_input.provider_key,
            model_name=record_input.model_name or record_input.requested_model,
            usage=record_input.usage,
            pricing_profiles=self._pricing_profiles,
            occurred_at=occurred_at,
            local_cost_policy=self._local_cost_policy,
        )

    def _find_existing_record(
        self,
        session: Session,
        record_input: LLMUsageRecordInput,
    ) -> LLMUsageLedgerRecord | None:
        request_id = record_input.request_id or record_input.provider_request_id
        if not request_id:
            return None
        context = record_input.context
        return session.exec(
            select(LLMUsageLedgerRecord).where(
                LLMUsageLedgerRecord.workspace_id == context.workspace_id,
                LLMUsageLedgerRecord.provider_key == record_input.provider_key,
                LLMUsageLedgerRecord.request_id == request_id,
                LLMUsageLedgerRecord.attempt_number == record_input.attempt_number,
            )
        ).first()


def _redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(value: str) -> str:
    redacted = str(value or "")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]" if match.lastindex else "[REDACTED]", redacted)
    return redacted


def _normalize_pricing_profiles(
    pricing_profiles: dict[LLMProviderKey | str, list[LLMPricingProfile]],
) -> dict[LLMProviderKey, list[LLMPricingProfile]]:
    normalized: dict[LLMProviderKey, list[LLMPricingProfile]] = {}
    for provider_key, profiles in pricing_profiles.items():
        try:
            provider = provider_key if isinstance(provider_key, LLMProviderKey) else LLMProviderKey(str(provider_key))
        except ValueError:
            continue
        normalized.setdefault(provider, []).extend(profiles)
    return normalized
