from uuid import uuid4

from app.services.llm_finops import (
    LLMCallContext,
    LLMCallStatus,
    LLMUsageCostBreakdown,
    LLMUsageRecordInput,
    NormalizedLLMUsage,
)


def test_llm_call_context_defaults_are_provider_agnostic() -> None:
    context = LLMCallContext()

    assert context.workspace_id is None
    assert context.execution_mode == "primary"
    assert context.source == "builder_runtime"
    assert context.metadata == {}


def test_llm_call_context_accepts_business_traceability_fields() -> None:
    workspace_id = uuid4()
    user_id = uuid4()
    session_id = uuid4()

    context = LLMCallContext(
        workspace_id=workspace_id,
        user_id=user_id,
        session_id=session_id,
        stage=" discover ",
        capability_key="analyze_discovery",
        action_key="discovery_analysis",
    )

    assert context.workspace_id == workspace_id
    assert context.user_id == user_id
    assert context.session_id == session_id
    assert context.stage == "discover"
    assert context.capability_key == "analyze_discovery"


def test_normalized_usage_fills_total_and_compatibility_payload() -> None:
    usage = NormalizedLLMUsage(input_tokens=120, output_tokens=45)

    assert usage.total_tokens == 165
    assert usage.compatibility_token_usage() == {
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "input_tokens": 120,
        "output_tokens": 45,
        "total_tokens": 165,
    }


def test_cost_breakdown_fills_total_and_normalizes_currency() -> None:
    cost = LLMUsageCostBreakdown(
        cost_input=0.10,
        cost_output=0.25,
        cost_other=0.05,
        currency="cop",
    )

    assert cost.cost_total == 0.40
    assert cost.currency == "COP"


def test_record_input_supports_failure_status_and_usage() -> None:
    record = LLMUsageRecordInput(
        provider_key="openai",
        model_name="gpt-test",
        status=LLMCallStatus.provider_unavailable,
        attempt_number=0,
        usage=NormalizedLLMUsage(input_tokens=10, output_tokens=5),
    )

    assert record.provider_key == "openai"
    assert record.status == LLMCallStatus.provider_unavailable
    assert record.attempt_number == 1
    assert record.usage.total_tokens == 15
