from uuid import uuid4

from sqlmodel import SQLModel, Session, create_engine, select

from app.models import LLMPricingProfile, LLMPricingRateEntry, LLMProviderKey, LLMUsageLedgerRecord
from app.services.llm_finops import (
    LLMCallContext,
    LLMCallStatus,
    LLMUsageCostBreakdown,
    LLMUsageRecordInput,
    NormalizedLLMUsage,
)
from app.services.llm_finops.ledger_service import LLMUsageLedgerService


def build_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def build_record_input(*, request_id: str = "req-1", attempt_number: int = 1) -> LLMUsageRecordInput:
    return LLMUsageRecordInput(
        context=LLMCallContext(
            workspace_id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            stage="discover",
            capability_key="analyze_discovery",
            action_key="discovery_analysis",
        ),
        provider_key="openai",
        model_name="gpt-test",
        requested_model="gpt-test",
        execution_backend="provider_native",
        execution_mode="primary",
        request_id=request_id,
        attempt_number=attempt_number,
        usage=NormalizedLLMUsage(input_tokens=120, output_tokens=48, cached_input_tokens=12),
        cost=LLMUsageCostBreakdown(cost_input=0.01, cost_output=0.03, currency="usd"),
    )


def build_pricing_profiles() -> dict[LLMProviderKey, list[LLMPricingProfile]]:
    return {
        LLMProviderKey.openai: [
            LLMPricingProfile(
                profile_key="openai_structured_output",
                label="OpenAI structured output",
                provider=LLMProviderKey.openai,
                model="fast=gpt-test | reasoning=gpt-reasoning",
                mode="standard",
                effective_from="2026-07-08",
                cop_per_usd=4000,
                rates=[
                    LLMPricingRateEntry(metric_key="fast_input_tokens_m", amount_usd=0.375),
                    LLMPricingRateEntry(metric_key="fast_cached_input_tokens_m", amount_usd=0.0375),
                    LLMPricingRateEntry(metric_key="fast_output_tokens_m", amount_usd=2.25),
                    LLMPricingRateEntry(metric_key="tool_calls_k", amount_usd=0),
                    LLMPricingRateEntry(metric_key="provider_session", amount_usd=0),
                ],
            )
        ],
    }


def test_record_call_persists_usage_event() -> None:
    db = build_session()
    payload = build_record_input()

    result = LLMUsageLedgerService().record_call(db, payload)
    record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert result.created is True
    assert result.duplicate is False
    assert record is not None
    assert record.provider_key == "openai"
    assert record.input_tokens == 120
    assert record.output_tokens == 48
    assert record.total_tokens == 168
    assert record.cached_input_tokens == 12
    assert record.cost_total == 0.04


def test_record_call_calculates_and_persists_cost_when_pricing_profiles_are_configured() -> None:
    db = build_session()
    payload = build_record_input(request_id="req-priced").model_copy(
        update={
            "usage": NormalizedLLMUsage(
                input_tokens=2_000_000,
                output_tokens=1_000_000,
                cached_input_tokens=1_000_000,
            ),
            "cost": LLMUsageCostBreakdown(),
        }
    )

    result = LLMUsageLedgerService(pricing_profiles=build_pricing_profiles()).record_call(db, payload)
    record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert result.cost.cost_total == 2.6625
    assert result.cost.pricing_profile_key == "openai_structured_output"
    assert record is not None
    assert record.cost_input == 0.4125
    assert record.cost_output == 2.25
    assert record.cost_total == 2.6625
    assert record.currency == "USD"
    assert record.fx_rate == 4000
    assert record.pricing_profile_key == "openai_structured_output"
    assert record.pricing_snapshot["model_rate_prefix"] == "fast"


def test_record_call_persists_zero_cost_warning_when_pricing_is_missing() -> None:
    db = build_session()
    payload = build_record_input(request_id="req-unpriced").model_copy(
        update={"cost": LLMUsageCostBreakdown(cost_input=9, cost_output=1)}
    )

    result = LLMUsageLedgerService(pricing_profiles={}).record_call(db, payload)
    record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert result.cost.cost_total == 0
    assert result.warnings
    assert record is not None
    assert record.cost_total == 0
    assert record.metadata_payload["cost_warnings"]


def test_record_call_is_idempotent_for_same_request_and_attempt() -> None:
    db = build_session()
    service = LLMUsageLedgerService()
    payload = build_record_input(request_id="req-duplicate", attempt_number=1)

    first = service.record_call(db, payload)
    second = service.record_call(db, payload)
    records = db.exec(select(LLMUsageLedgerRecord)).all()

    assert first.created is True
    assert second.duplicate is True
    assert first.usage_record_id == second.usage_record_id
    assert len(records) == 1


def test_record_call_allows_distinct_attempts_for_same_request() -> None:
    db = build_session()
    service = LLMUsageLedgerService()
    first_payload = build_record_input(request_id="req-retry", attempt_number=1)
    second_payload = first_payload.model_copy(update={"attempt_number": 2})

    service.record_call(db, first_payload)
    service.record_call(db, second_payload)
    records = db.exec(select(LLMUsageLedgerRecord).order_by(LLMUsageLedgerRecord.attempt_number)).all()

    assert [item.attempt_number for item in records] == [1, 2]


def test_record_call_persists_failed_call_without_usage_and_redacts_error() -> None:
    db = build_session()
    payload = build_record_input(request_id="req-failed").model_copy(
        update={
            "status": LLMCallStatus.failed,
            "failure_kind": "provider_error",
            "failure_detail": "provider failed with api_key=sk-super-secret-value",
            "usage": NormalizedLLMUsage(),
        }
    )

    result = LLMUsageLedgerService().record_call(db, payload)
    record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert record is not None
    assert record.status == "failed"
    assert record.total_tokens == 0
    assert "sk-super-secret-value" not in record.failure_detail_redacted
    assert "[REDACTED]" in record.failure_detail_redacted
