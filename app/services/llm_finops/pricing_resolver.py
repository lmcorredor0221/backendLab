from __future__ import annotations

from datetime import datetime
from typing import Any

from app.models import CodexLocalCostPolicy, LLMPricingProfile, LLMProviderKey
from app.services.llm_finops.contracts import LLMUsageCostBreakdown, NormalizedLLMUsage


class PricingResolver:
    def zero_cost(self, *, warning: str = "") -> LLMUsageCostBreakdown:
        warnings = [warning] if warning else []
        return LLMUsageCostBreakdown(warnings=warnings)

    def resolve_call_cost(
        self,
        *,
        provider_key: LLMProviderKey | str,
        model_name: str,
        usage: NormalizedLLMUsage,
        pricing_profiles: dict[LLMProviderKey, list[LLMPricingProfile]],
        occurred_at: datetime | None = None,
        local_cost_policy: CodexLocalCostPolicy | None = None,
    ) -> LLMUsageCostBreakdown:
        provider = _coerce_provider(provider_key)
        if provider is None:
            return self.zero_cost(warning=f"Unknown LLM provider for pricing: {provider_key}")

        profile = _select_profile(
            pricing_profiles.get(provider, []),
            model_name=model_name,
            occurred_at=occurred_at,
            local_cost_policy=local_cost_policy,
        )
        if profile is None:
            return self.zero_cost(warning=f"No pricing profile configured for provider {provider.value}.")

        if provider in {LLMProviderKey.codex_local, LLMProviderKey.antigravity_cli}:
            return _resolve_local_cost(
                profile=profile,
                usage=usage,
                model_name=model_name,
                local_cost_policy=local_cost_policy or profile.local_cost_policy,
            )
        if provider == LLMProviderKey.deepseek:
            return _resolve_token_cost(
                profile=profile,
                usage=usage,
                model_name=model_name,
                provider=provider,
                prefix=_resolve_model_prefix(profile, model_name, fast_markers=("flash", "mini", "fast")),
                cache_hit_metric="input_cache_hit",
                cache_miss_metric="input_cache_miss",
            )
        return _resolve_token_cost(
            profile=profile,
            usage=usage,
            model_name=model_name,
            provider=provider,
            prefix=_resolve_model_prefix(profile, model_name, fast_markers=("mini", "flash", "fast")),
            cache_hit_metric="cached_input",
            cache_miss_metric="input",
        )


def _coerce_provider(provider_key: LLMProviderKey | str) -> LLMProviderKey | None:
    if isinstance(provider_key, LLMProviderKey):
        return provider_key
    try:
        return LLMProviderKey(str(provider_key))
    except ValueError:
        return None


def _select_profile(
    candidates: list[LLMPricingProfile],
    *,
    model_name: str,
    occurred_at: datetime | None,
    local_cost_policy: CodexLocalCostPolicy | None,
) -> LLMPricingProfile | None:
    if not candidates:
        return None
    effective_date = (occurred_at or datetime.max).date()
    eligible: list[LLMPricingProfile] = []
    for profile in candidates:
        if local_cost_policy is not None and profile.local_cost_policy not in {None, local_cost_policy}:
            continue
        if profile.effective_from:
            try:
                profile_date = datetime.fromisoformat(profile.effective_from).date()
            except ValueError:
                profile_date = datetime.min.date()
            if profile_date > effective_date:
                continue
        eligible.append(profile)
    if not eligible:
        eligible = list(candidates)

    model_filtered = [profile for profile in eligible if _profile_matches_model(profile, model_name)]
    ordered = sorted(
        model_filtered or eligible,
        key=lambda item: (item.effective_from or "", item.profile_key),
        reverse=True,
    )
    return ordered[0] if ordered else None


def _profile_matches_model(profile: LLMPricingProfile, model_name: str) -> bool:
    normalized_model = model_name.strip().lower()
    if not normalized_model:
        return True
    profile_model = profile.model.strip().lower()
    return not profile_model or normalized_model in profile_model or profile_model in normalized_model


def _resolve_model_prefix(
    profile: LLMPricingProfile,
    model_name: str,
    *,
    fast_markers: tuple[str, ...],
) -> str:
    normalized_model = model_name.strip().lower()
    profile_model = profile.model.strip().lower()
    if "reasoning=" in profile_model and normalized_model:
        reasoning_segment = profile_model.split("reasoning=", 1)[1].split("|", 1)[0].strip()
        if normalized_model and normalized_model in reasoning_segment:
            return "reasoning"
    if "fast=" in profile_model and normalized_model:
        fast_segment = profile_model.split("fast=", 1)[1].split("|", 1)[0].strip()
        if normalized_model and normalized_model in fast_segment:
            return "fast"
    if any(marker in normalized_model for marker in fast_markers):
        return "fast"
    return "reasoning"


def _resolve_token_cost(
    *,
    profile: LLMPricingProfile,
    usage: NormalizedLLMUsage,
    model_name: str,
    provider: LLMProviderKey,
    prefix: str,
    cache_hit_metric: str,
    cache_miss_metric: str,
) -> LLMUsageCostBreakdown:
    rate_map = {item.metric_key: item.amount_usd for item in profile.rates}
    cached_input_m = usage.cached_input_tokens / 1_000_000
    uncached_input_m = max(usage.input_tokens - usage.cached_input_tokens, 0) / 1_000_000
    output_m = usage.output_tokens / 1_000_000
    reasoning_m = usage.reasoning_tokens / 1_000_000
    tool_calls_k = usage.tool_call_count / 1_000

    if provider == LLMProviderKey.deepseek:
        input_rate_key = f"{prefix}_{cache_miss_metric}_tokens_m"
        cached_rate_key = f"{prefix}_{cache_hit_metric}_tokens_m"
    else:
        input_rate_key = f"{prefix}_{cache_miss_metric}_tokens_m"
        cached_rate_key = f"{prefix}_{cache_hit_metric}_tokens_m"
    output_rate_key = f"{prefix}_output_tokens_m"
    reasoning_rate_key = f"{prefix}_reasoning_tokens_m"

    cost_input = (
        uncached_input_m * rate_map.get(input_rate_key, 0)
        + cached_input_m * rate_map.get(cached_rate_key, 0)
    )
    cost_output = output_m * rate_map.get(output_rate_key, 0)
    cost_other = (
        reasoning_m * rate_map.get(reasoning_rate_key, 0)
        + tool_calls_k * rate_map.get("tool_calls_k", 0)
        + rate_map.get("provider_session", 0)
    )
    applied_rate_keys = [
        input_rate_key,
        cached_rate_key,
        output_rate_key,
        reasoning_rate_key,
        "tool_calls_k",
        "provider_session",
    ]
    warnings = _missing_rate_warnings(rate_map, applied_rate_keys)
    return _build_breakdown(
        profile=profile,
        model_name=model_name,
        cost_input=cost_input,
        cost_output=cost_output,
        cost_other=cost_other,
        applied_rate_keys=applied_rate_keys,
        warnings=warnings,
        extra_snapshot={"pricing_kind": "token", "model_rate_prefix": prefix},
    )


def _resolve_local_cost(
    *,
    profile: LLMPricingProfile,
    usage: NormalizedLLMUsage,
    model_name: str,
    local_cost_policy: CodexLocalCostPolicy | None,
) -> LLMUsageCostBreakdown:
    rate_map = {item.metric_key: item.amount_usd for item in profile.rates}
    policy = local_cost_policy or CodexLocalCostPolicy.hybrid
    duration_ms = _metric_number(usage.provider_metrics, "duration_ms")
    compute_hours = duration_ms / 3_600_000 if duration_ms else _metric_number(usage.provider_metrics, "compute_hours")
    tool_calls_k = usage.tool_call_count / 1_000
    cost_input = 0.0
    cost_output = 0.0
    cost_other = (
        compute_hours * rate_map.get("compute_hour_core", 0)
        + tool_calls_k * rate_map.get("tool_calls_k", 0)
        + rate_map.get("local_session", 0)
    )
    applied_rate_keys = ["compute_hour_core", "tool_calls_k", "local_session"]
    if policy == CodexLocalCostPolicy.hybrid:
        cost_other += compute_hours * rate_map.get("workstation_hour_hybrid", 0)
        applied_rate_keys.append("workstation_hour_hybrid")
    elif policy == CodexLocalCostPolicy.fully_loaded:
        cost_other += compute_hours * rate_map.get("workstation_hour_fully_loaded", 0)
        applied_rate_keys.append("workstation_hour_fully_loaded")
    warnings = _missing_rate_warnings(rate_map, applied_rate_keys)
    return _build_breakdown(
        profile=profile,
        model_name=model_name,
        cost_input=cost_input,
        cost_output=cost_output,
        cost_other=cost_other,
        applied_rate_keys=applied_rate_keys,
        warnings=warnings,
        extra_snapshot={
            "pricing_kind": "local_runtime",
            "local_cost_policy": policy.value,
            "compute_hours": compute_hours,
        },
    )


def _metric_number(metrics: dict[str, Any], key: str) -> float:
    try:
        return max(0.0, float(metrics.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def _missing_rate_warnings(rate_map: dict[str, float], rate_keys: list[str]) -> list[str]:
    missing = sorted(key for key in set(rate_keys) if key not in rate_map and key not in {"provider_session", "tool_calls_k"})
    if not missing:
        return []
    return ["Missing pricing rates: " + ", ".join(missing)]


def _build_breakdown(
    *,
    profile: LLMPricingProfile,
    model_name: str,
    cost_input: float,
    cost_output: float,
    cost_other: float,
    applied_rate_keys: list[str],
    warnings: list[str],
    extra_snapshot: dict[str, Any],
) -> LLMUsageCostBreakdown:
    snapshot = {
        "provider": profile.provider.value if isinstance(profile.provider, LLMProviderKey) else str(profile.provider),
        "profile_key": profile.profile_key,
        "label": profile.label,
        "model": model_name or profile.model,
        "pricing_mode": profile.mode,
        "effective_from": profile.effective_from,
        "is_local_inference": profile.is_local_inference,
        "local_cost_policy": profile.local_cost_policy.value if profile.local_cost_policy else None,
        "cop_per_usd": profile.cop_per_usd,
        "applied_rate_keys": applied_rate_keys,
        "rates": [
            item.model_dump(mode="json")
            for item in profile.rates
            if item.metric_key in set(applied_rate_keys)
        ],
        **extra_snapshot,
    }
    return LLMUsageCostBreakdown(
        cost_input=round(cost_input, 8),
        cost_output=round(cost_output, 8),
        cost_other=round(cost_other, 8),
        cost_total=round(cost_input + cost_output + cost_other, 8),
        currency="USD",
        fx_rate=profile.cop_per_usd,
        pricing_profile_key=profile.profile_key,
        pricing_snapshot=snapshot,
        warnings=warnings,
    )
