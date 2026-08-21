from app.models import (
    CodexLocalCostPolicy,
    LLMPricingProfile,
    LLMPricingRateEntry,
    LLMProviderKey,
)
from app.services.llm_finops import NormalizedLLMUsage
from app.services.llm_finops.pricing_resolver import PricingResolver


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
                ],
            )
        ],
    }


def test_pricing_resolver_calculates_openai_fast_cost_from_real_usage() -> None:
    cost = PricingResolver().resolve_call_cost(
        provider_key=LLMProviderKey.openai,
        model_name="gpt-5.4-mini",
        usage=NormalizedLLMUsage(
            input_tokens=2_000_000,
            output_tokens=1_000_000,
            cached_input_tokens=1_000_000,
        ),
        pricing_profiles=build_profiles(),
    )

    assert cost.cost_input == 0.4125
    assert cost.cost_output == 2.25
    assert cost.cost_total == 2.6625
    assert cost.currency == "USD"
    assert cost.fx_rate == 4000
    assert cost.pricing_profile_key == "openai_structured_output"
    assert cost.pricing_snapshot["model_rate_prefix"] == "fast"


def test_pricing_resolver_calculates_deepseek_cache_cost() -> None:
    cost = PricingResolver().resolve_call_cost(
        provider_key=LLMProviderKey.deepseek,
        model_name="deepseek-v4-flash",
        usage=NormalizedLLMUsage(
            input_tokens=2_000_000,
            output_tokens=1_000_000,
            cached_input_tokens=1_000_000,
        ),
        pricing_profiles=build_profiles(),
    )

    assert cost.cost_input == 0.1428
    assert cost.cost_output == 0.28
    assert cost.cost_total == 0.4228
    assert cost.pricing_snapshot["model_rate_prefix"] == "fast"


def test_pricing_resolver_calculates_local_runtime_cost_from_duration() -> None:
    cost = PricingResolver().resolve_call_cost(
        provider_key=LLMProviderKey.codex_local,
        model_name="gpt-5.5",
        usage=NormalizedLLMUsage(
            provider_metrics={"duration_ms": 3_600_000},
            usage_is_estimated=True,
        ),
        pricing_profiles=build_profiles(),
        local_cost_policy=CodexLocalCostPolicy.hybrid,
    )

    assert cost.cost_input == 0
    assert cost.cost_output == 0
    assert cost.cost_other == 2.45
    assert cost.cost_total == 2.45
    assert cost.pricing_snapshot["pricing_kind"] == "local_runtime"


def test_pricing_resolver_returns_zero_with_warning_when_profile_missing() -> None:
    cost = PricingResolver().resolve_call_cost(
        provider_key=LLMProviderKey.openai,
        model_name="gpt-5.4-mini",
        usage=NormalizedLLMUsage(input_tokens=10, output_tokens=2),
        pricing_profiles={},
    )

    assert cost.cost_total == 0
    assert cost.warnings
