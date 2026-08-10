from dataclasses import dataclass

from app.models import (
    CodexLocalCostPolicy,
    CodexLocalProviderConfig,
    DeepSeekProviderConfig,
    EstimationComplexityLevel,
    EstimationMaturityStage,
    LLMPricingProfile,
    LLMPricingRateEntry,
    LLMProviderKey,
    LLMRuntimeSettings,
    OpenAIProviderConfig,
)
from app.services.llm_cost_engine import estimate_agentic_costs


@dataclass(frozen=True)
class FakeSignals:
    maturity_stage: EstimationMaturityStage
    complexity: EstimationComplexityLevel
    active_provider: LLMProviderKey
    scope_points: int
    tool_count: int
    side_effect_tools: int
    workflow_steps: int
    evaluation_cases: int
    blocking_gaps: int
    open_questions: int


def build_runtime_settings(active_provider: LLMProviderKey, *, codex_policy: CodexLocalCostPolicy = CodexLocalCostPolicy.hybrid) -> LLMRuntimeSettings:
    return LLMRuntimeSettings(
        active_provider=active_provider,
        openai=OpenAIProviderConfig(fast_model="gpt-5.4-mini", reasoning_model="gpt-5.5", reasoning_effort="low"),
        deepseek=DeepSeekProviderConfig(
            base_url="https://api.deepseek.com",
            fast_model="deepseek-v4-flash",
            reasoning_model="deepseek-v4-pro",
            reasoning_effort="max",
        ),
        codex_local=CodexLocalProviderConfig(
            command="codex",
            model="gpt-5.5",
            profile="deep-review",
            cost_policy=codex_policy,
            executable_found=True,
            available=True,
        ),
    )


def build_profiles() -> dict[LLMProviderKey, list[LLMPricingProfile]]:
    return {
        LLMProviderKey.openai: [
            LLMPricingProfile(
                profile_key="openai_structured_output",
                label="OpenAI structured output",
                provider=LLMProviderKey.openai,
                model="fast=gpt-5.4-mini | reasoning=gpt-5.5",
                mode="standard",
                effective_from="2026-07-08",
                cop_per_usd=4000,
                rates=[
                    LLMPricingRateEntry(metric_key="fast_input_tokens_m", amount_usd=0.375),
                    LLMPricingRateEntry(metric_key="fast_cached_input_tokens_m", amount_usd=0.0375),
                    LLMPricingRateEntry(metric_key="fast_output_tokens_m", amount_usd=2.25),
                    LLMPricingRateEntry(metric_key="reasoning_input_tokens_m", amount_usd=5.0),
                    LLMPricingRateEntry(metric_key="reasoning_cached_input_tokens_m", amount_usd=0.5),
                    LLMPricingRateEntry(metric_key="reasoning_output_tokens_m", amount_usd=22.5),
                    LLMPricingRateEntry(metric_key="tool_calls_k", amount_usd=0),
                    LLMPricingRateEntry(metric_key="provider_session", amount_usd=0),
                ],
            )
        ],
        LLMProviderKey.deepseek: [
            LLMPricingProfile(
                profile_key="deepseek_api_profile",
                label="DeepSeek API",
                provider=LLMProviderKey.deepseek,
                model="fast=deepseek-v4-flash | reasoning=deepseek-v4-pro",
                mode="api",
                effective_from="2026-07-08",
                cop_per_usd=4000,
                rates=[
                    LLMPricingRateEntry(metric_key="fast_input_cache_hit_tokens_m", amount_usd=0.0028),
                    LLMPricingRateEntry(metric_key="fast_input_cache_miss_tokens_m", amount_usd=0.14),
                    LLMPricingRateEntry(metric_key="fast_output_tokens_m", amount_usd=0.28),
                    LLMPricingRateEntry(metric_key="reasoning_input_cache_hit_tokens_m", amount_usd=0.003625),
                    LLMPricingRateEntry(metric_key="reasoning_input_cache_miss_tokens_m", amount_usd=0.435),
                    LLMPricingRateEntry(metric_key="reasoning_output_tokens_m", amount_usd=0.87),
                    LLMPricingRateEntry(metric_key="tool_calls_k", amount_usd=0),
                    LLMPricingRateEntry(metric_key="provider_session", amount_usd=0),
                ],
            )
        ],
        LLMProviderKey.codex_local: [
            LLMPricingProfile(
                profile_key="codex_local_hybrid",
                label="Codex local",
                provider=LLMProviderKey.codex_local,
                model="command=codex | tier=local",
                mode="local",
                is_local_inference=True,
                local_cost_policy=CodexLocalCostPolicy.hybrid,
                effective_from="2026-07-08",
                cop_per_usd=4000,
                rates=[
                    LLMPricingRateEntry(metric_key="compute_hour_core", amount_usd=0.85),
                    LLMPricingRateEntry(metric_key="tool_calls_k", amount_usd=0.03),
                    LLMPricingRateEntry(metric_key="local_session", amount_usd=0.35),
                    LLMPricingRateEntry(metric_key="workstation_hour_hybrid", amount_usd=1.25),
                    LLMPricingRateEntry(metric_key="workstation_hour_fully_loaded", amount_usd=3.25),
                ],
            )
        ],
    }


def test_llm_cost_engine_changes_cost_by_provider() -> None:
    profiles = build_profiles()
    signals = FakeSignals(
        maturity_stage=EstimationMaturityStage.blueprint,
        complexity=EstimationComplexityLevel.complex,
        active_provider=LLMProviderKey.openai,
        scope_points=70,
        tool_count=3,
        side_effect_tools=1,
        workflow_steps=4,
        evaluation_cases=3,
        blocking_gaps=0,
        open_questions=1,
    )

    openai_cost = estimate_agentic_costs(
        signals=signals,
        runtime_settings=build_runtime_settings(LLMProviderKey.openai),
        pricing_profiles=profiles,
        human_supervision_hours=24,
        human_supervision_rate_cop=150000,
    )
    deepseek_cost = estimate_agentic_costs(
        signals=signals.__class__(**{**signals.__dict__, "active_provider": LLMProviderKey.deepseek}),
        runtime_settings=build_runtime_settings(LLMProviderKey.deepseek),
        pricing_profiles=profiles,
        human_supervision_hours=24,
        human_supervision_rate_cop=150000,
    )

    assert openai_cost.pricing_snapshot is not None
    assert deepseek_cost.pricing_snapshot is not None
    assert openai_cost.pricing_snapshot.provider == "openai"
    assert deepseek_cost.pricing_snapshot.provider == "deepseek"
    assert openai_cost.provider_runtime_cost_total_usd != deepseek_cost.provider_runtime_cost_total_usd


def test_llm_cost_engine_respects_codex_local_cost_policy() -> None:
    profiles = build_profiles()
    signals = FakeSignals(
        maturity_stage=EstimationMaturityStage.ready_to_build,
        complexity=EstimationComplexityLevel.complex,
        active_provider=LLMProviderKey.codex_local,
        scope_points=82,
        tool_count=4,
        side_effect_tools=1,
        workflow_steps=5,
        evaluation_cases=4,
        blocking_gaps=0,
        open_questions=0,
    )

    hybrid_cost = estimate_agentic_costs(
        signals=signals,
        runtime_settings=build_runtime_settings(LLMProviderKey.codex_local, codex_policy=CodexLocalCostPolicy.hybrid),
        pricing_profiles=profiles,
        human_supervision_hours=28,
        human_supervision_rate_cop=150000,
    )
    full_cost = estimate_agentic_costs(
        signals=signals,
        runtime_settings=build_runtime_settings(LLMProviderKey.codex_local, codex_policy=CodexLocalCostPolicy.fully_loaded),
        pricing_profiles=profiles,
        human_supervision_hours=28,
        human_supervision_rate_cop=150000,
    )

    assert hybrid_cost.economic_model == "hybrid"
    assert full_cost.economic_model == "fully_loaded"
    assert full_cost.platform_overhead_cost_usd > hybrid_cost.platform_overhead_cost_usd
    assert full_cost.provider_runtime_cost_total_usd > hybrid_cost.provider_runtime_cost_total_usd


def test_llm_cost_engine_falls_back_when_pricing_is_missing() -> None:
    signals = FakeSignals(
        maturity_stage=EstimationMaturityStage.canvas,
        complexity=EstimationComplexityLevel.moderate,
        active_provider=LLMProviderKey.openai,
        scope_points=40,
        tool_count=1,
        side_effect_tools=0,
        workflow_steps=2,
        evaluation_cases=0,
        blocking_gaps=0,
        open_questions=0,
    )

    breakdown = estimate_agentic_costs(
        signals=signals,
        runtime_settings=build_runtime_settings(LLMProviderKey.openai),
        pricing_profiles={},
        human_supervision_hours=10,
        human_supervision_rate_cop=150000,
    )

    assert breakdown.pricing_snapshot is None
    assert breakdown.provider_runtime_cost_total_usd == 0
    assert breakdown.warnings


def test_llm_cost_engine_falls_back_to_latest_complete_profile() -> None:
    signals = FakeSignals(
        maturity_stage=EstimationMaturityStage.blueprint,
        complexity=EstimationComplexityLevel.moderate,
        active_provider=LLMProviderKey.openai,
        scope_points=54,
        tool_count=2,
        side_effect_tools=0,
        workflow_steps=3,
        evaluation_cases=2,
        blocking_gaps=0,
        open_questions=0,
    )
    incomplete_latest = LLMPricingProfile(
        profile_key="openai_incomplete_latest",
        label="OpenAI incomplete latest",
        provider=LLMProviderKey.openai,
        model="fast=gpt-5.4-mini | reasoning=gpt-5.5",
        mode="standard",
        effective_from="2026-07-10",
        cop_per_usd=4000,
        rates=[
            LLMPricingRateEntry(metric_key="fast_input_tokens_m", amount_usd=0.375),
            LLMPricingRateEntry(metric_key="fast_output_tokens_m", amount_usd=2.25),
        ],
    )
    complete_fallback = build_profiles()[LLMProviderKey.openai][0]

    breakdown = estimate_agentic_costs(
        signals=signals,
        runtime_settings=build_runtime_settings(LLMProviderKey.openai),
        pricing_profiles={LLMProviderKey.openai: [incomplete_latest, complete_fallback]},
        human_supervision_hours=12,
        human_supervision_rate_cop=150000,
    )

    assert breakdown.pricing_snapshot is not None
    assert breakdown.pricing_snapshot.profile_key == "openai_structured_output"
    assert any("fallback aplicado" in item.lower() for item in breakdown.warnings)
