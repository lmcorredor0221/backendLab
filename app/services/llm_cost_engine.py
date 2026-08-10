from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models import (
    CodexLocalCostPolicy,
    EstimationComplexityLevel,
    EstimationMaturityStage,
    EstimationPricingSnapshot,
    LLMPricingProfile,
    LLMPricingRateEntry,
    LLMProviderKey,
    LLMRuntimeSettings,
)


class EstimationSignalsLike(Protocol):
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


@dataclass(frozen=True)
class AgenticCostBreakdown:
    provider_model: str
    economic_model: str
    human_supervision_cost: float
    llm_runtime_cost_usd: float
    tool_runtime_cost_usd: float
    platform_overhead_cost_usd: float
    provider_runtime_cost_total_usd: float
    provider_runtime_cost_total_cop: float
    pricing_assumptions: list[str]
    warnings: list[str]
    pricing_snapshot: EstimationPricingSnapshot | None


@dataclass(frozen=True)
class ResolvedPricingProfile:
    profile: LLMPricingProfile | None
    warnings: list[str]


def estimate_agentic_costs(
    *,
    signals: EstimationSignalsLike,
    runtime_settings: LLMRuntimeSettings,
    pricing_profiles: dict[LLMProviderKey, list[LLMPricingProfile]],
    human_supervision_hours: float,
    human_supervision_rate_cop: float,
) -> AgenticCostBreakdown:
    human_supervision_cost = round(human_supervision_hours * max(human_supervision_rate_cop, 0), 2)
    local_cost_policy = (
        runtime_settings.codex_local.cost_policy
        if signals.active_provider == LLMProviderKey.codex_local
        else None
    )
    resolved_profile = _resolve_active_profile(
        pricing_profiles,
        signals.active_provider,
        local_cost_policy=local_cost_policy,
    )
    profile = resolved_profile.profile

    if profile is None or not profile.rates:
        assumptions = [
            "No hay pricing profile vigente para el proveedor activo; el costo variable queda en cero hasta cargar tarifas.",
        ]
        warnings = [
            f"El proveedor {signals.active_provider.value} no tiene pricing vigente cargado en el workspace.",
        ] + resolved_profile.warnings
        return AgenticCostBreakdown(
            provider_model=_resolve_provider_model(runtime_settings),
            economic_model=_resolve_economic_model(runtime_settings, None),
            human_supervision_cost=human_supervision_cost,
            llm_runtime_cost_usd=0.0,
            tool_runtime_cost_usd=0.0,
            platform_overhead_cost_usd=0.0,
            provider_runtime_cost_total_usd=0.0,
            provider_runtime_cost_total_cop=0.0,
            pricing_assumptions=assumptions,
            warnings=warnings,
            pricing_snapshot=None,
        )

    if signals.active_provider == LLMProviderKey.deepseek:
        return _estimate_deepseek_costs(
            signals=signals,
            runtime_settings=runtime_settings,
            profile=profile,
            human_supervision_cost=human_supervision_cost,
            precomputed_warnings=resolved_profile.warnings,
        )
    if signals.active_provider == LLMProviderKey.codex_local:
        return _estimate_codex_local_costs(
            signals=signals,
            runtime_settings=runtime_settings,
            profile=profile,
            human_supervision_cost=human_supervision_cost,
            precomputed_warnings=resolved_profile.warnings,
        )
    return _estimate_openai_costs(
        signals=signals,
        runtime_settings=runtime_settings,
        profile=profile,
        human_supervision_cost=human_supervision_cost,
        precomputed_warnings=resolved_profile.warnings,
    )


def _estimate_openai_costs(
    *,
    signals: EstimationSignalsLike,
    runtime_settings: LLMRuntimeSettings,
    profile: LLMPricingProfile,
    human_supervision_cost: float,
    precomputed_warnings: list[str],
) -> AgenticCostBreakdown:
    total_input_m = _base_input_tokens_m(signals)
    fast_share, reasoning_share = _provider_mix(signals.maturity_stage)
    fast_input_m = total_input_m * fast_share
    reasoning_input_m = total_input_m * reasoning_share
    cache_ratio = {EstimationMaturityStage.canvas: 0.38, EstimationMaturityStage.blueprint: 0.46, EstimationMaturityStage.ready_to_build: 0.54}[signals.maturity_stage]
    fast_cached_m = fast_input_m * cache_ratio
    fast_uncached_m = max(fast_input_m - fast_cached_m, 0.0)
    reasoning_cached_m = reasoning_input_m * min(cache_ratio + 0.08, 0.72)
    reasoning_uncached_m = max(reasoning_input_m - reasoning_cached_m, 0.0)
    fast_output_m = fast_input_m * 0.34
    reasoning_output_m = reasoning_input_m * 0.62
    tool_calls_k = _tool_call_units_k(signals)

    rate_map = {item.metric_key: item.amount_usd for item in profile.rates}
    used_keys = [
        "fast_input_tokens_m",
        "fast_cached_input_tokens_m",
        "fast_output_tokens_m",
        "reasoning_input_tokens_m",
        "reasoning_cached_input_tokens_m",
        "reasoning_output_tokens_m",
        "tool_calls_k",
        "provider_session",
    ]
    llm_runtime_cost = (
        fast_uncached_m * rate_map.get("fast_input_tokens_m", 0)
        + fast_cached_m * rate_map.get("fast_cached_input_tokens_m", 0)
        + fast_output_m * rate_map.get("fast_output_tokens_m", 0)
        + reasoning_uncached_m * rate_map.get("reasoning_input_tokens_m", 0)
        + reasoning_cached_m * rate_map.get("reasoning_cached_input_tokens_m", 0)
        + reasoning_output_m * rate_map.get("reasoning_output_tokens_m", 0)
    )
    tool_runtime_cost = tool_calls_k * rate_map.get("tool_calls_k", 0)
    platform_overhead_cost = rate_map.get("provider_session", 0)

    assumptions = [
        "OpenAI usa mezcla de modelos fast/reasoning segun la etapa de madurez del proyecto.",
        f"Se uso pricing profile vigente desde {profile.effective_from or 'sin fecha declarada'}.",
        "Tool calls propios del builder no agregan fee del proveedor salvo que el perfil lo declare.",
    ]
    return _finalize_breakdown(
        profile=profile,
        provider_model=f"fast={runtime_settings.openai.fast_model} | reasoning={runtime_settings.openai.reasoning_model}",
        economic_model=profile.mode or "standard",
        human_supervision_cost=human_supervision_cost,
        llm_runtime_cost_usd=llm_runtime_cost,
        tool_runtime_cost_usd=tool_runtime_cost,
        platform_overhead_cost_usd=platform_overhead_cost,
        pricing_assumptions=assumptions,
        used_rate_keys=used_keys,
        precomputed_warnings=precomputed_warnings,
    )


def _estimate_deepseek_costs(
    *,
    signals: EstimationSignalsLike,
    runtime_settings: LLMRuntimeSettings,
    profile: LLMPricingProfile,
    human_supervision_cost: float,
    precomputed_warnings: list[str],
) -> AgenticCostBreakdown:
    total_input_m = _base_input_tokens_m(signals) * 0.92
    fast_share, reasoning_share = _provider_mix(signals.maturity_stage)
    fast_input_m = total_input_m * max(fast_share - 0.04, 0.2)
    reasoning_input_m = total_input_m * min(reasoning_share + 0.04, 0.8)
    cache_hit_ratio = {EstimationMaturityStage.canvas: 0.48, EstimationMaturityStage.blueprint: 0.56, EstimationMaturityStage.ready_to_build: 0.62}[signals.maturity_stage]
    fast_cache_hit_m = fast_input_m * cache_hit_ratio
    fast_cache_miss_m = max(fast_input_m - fast_cache_hit_m, 0.0)
    reasoning_cache_hit_m = reasoning_input_m * min(cache_hit_ratio + 0.06, 0.76)
    reasoning_cache_miss_m = max(reasoning_input_m - reasoning_cache_hit_m, 0.0)
    fast_output_m = fast_input_m * 0.31
    reasoning_output_m = reasoning_input_m * 0.58
    tool_calls_k = _tool_call_units_k(signals)

    rate_map = {item.metric_key: item.amount_usd for item in profile.rates}
    used_keys = [
        "fast_input_cache_hit_tokens_m",
        "fast_input_cache_miss_tokens_m",
        "fast_output_tokens_m",
        "reasoning_input_cache_hit_tokens_m",
        "reasoning_input_cache_miss_tokens_m",
        "reasoning_output_tokens_m",
        "tool_calls_k",
        "provider_session",
    ]
    llm_runtime_cost = (
        fast_cache_hit_m * rate_map.get("fast_input_cache_hit_tokens_m", 0)
        + fast_cache_miss_m * rate_map.get("fast_input_cache_miss_tokens_m", 0)
        + fast_output_m * rate_map.get("fast_output_tokens_m", 0)
        + reasoning_cache_hit_m * rate_map.get("reasoning_input_cache_hit_tokens_m", 0)
        + reasoning_cache_miss_m * rate_map.get("reasoning_input_cache_miss_tokens_m", 0)
        + reasoning_output_m * rate_map.get("reasoning_output_tokens_m", 0)
    )
    tool_runtime_cost = tool_calls_k * rate_map.get("tool_calls_k", 0)
    platform_overhead_cost = rate_map.get("provider_session", 0)

    assumptions = [
        "DeepSeek usa mezcla de modelos flash/pro con cache hit y cache miss diferenciados.",
        f"Base URL activa considerada: {runtime_settings.deepseek.base_url}.",
        f"Se uso pricing profile vigente desde {profile.effective_from or 'sin fecha declarada'}.",
    ]
    return _finalize_breakdown(
        profile=profile,
        provider_model=f"fast={runtime_settings.deepseek.fast_model} | reasoning={runtime_settings.deepseek.reasoning_model}",
        economic_model=profile.mode or "api",
        human_supervision_cost=human_supervision_cost,
        llm_runtime_cost_usd=llm_runtime_cost,
        tool_runtime_cost_usd=tool_runtime_cost,
        platform_overhead_cost_usd=platform_overhead_cost,
        pricing_assumptions=assumptions,
        used_rate_keys=used_keys,
        precomputed_warnings=precomputed_warnings,
    )


def _estimate_codex_local_costs(
    *,
    signals: EstimationSignalsLike,
    runtime_settings: LLMRuntimeSettings,
    profile: LLMPricingProfile,
    human_supervision_cost: float,
    precomputed_warnings: list[str],
) -> AgenticCostBreakdown:
    rate_map = {item.metric_key: item.amount_usd for item in profile.rates}
    policy = runtime_settings.codex_local.cost_policy or profile.local_cost_policy or CodexLocalCostPolicy.hybrid
    compute_hours = _local_compute_hours(signals)
    tool_calls_k = _tool_call_units_k(signals)

    llm_runtime_cost = compute_hours * rate_map.get("compute_hour_core", 0)
    tool_runtime_cost = tool_calls_k * rate_map.get("tool_calls_k", 0)
    platform_overhead_cost = rate_map.get("local_session", 0)
    if policy == CodexLocalCostPolicy.hybrid:
        platform_overhead_cost += compute_hours * rate_map.get("workstation_hour_hybrid", 0)
    elif policy == CodexLocalCostPolicy.fully_loaded:
        platform_overhead_cost += compute_hours * rate_map.get("workstation_hour_fully_loaded", 0)

    assumptions = [
        f"Codex local se estima con politica {policy.value}.",
        "El costo local no tokeniza consumo; se aproxima por horas de compute y overhead del workstation.",
        f"Se uso pricing profile vigente desde {profile.effective_from or 'sin fecha declarada'}.",
    ]
    used_rate_keys = ["compute_hour_core", "tool_calls_k", "local_session"]
    if policy == CodexLocalCostPolicy.hybrid:
        used_rate_keys.append("workstation_hour_hybrid")
    elif policy == CodexLocalCostPolicy.fully_loaded:
        used_rate_keys.append("workstation_hour_fully_loaded")
    return _finalize_breakdown(
        profile=profile,
        provider_model=f"command={runtime_settings.codex_local.command} | model={runtime_settings.codex_local.model}",
        economic_model=policy.value,
        human_supervision_cost=human_supervision_cost,
        llm_runtime_cost_usd=llm_runtime_cost,
        tool_runtime_cost_usd=tool_runtime_cost,
        platform_overhead_cost_usd=platform_overhead_cost,
        pricing_assumptions=assumptions,
        used_rate_keys=used_rate_keys,
        local_cost_policy=policy,
        precomputed_warnings=precomputed_warnings,
    )


def _finalize_breakdown(
    *,
    profile: LLMPricingProfile,
    provider_model: str,
    economic_model: str,
    human_supervision_cost: float,
    llm_runtime_cost_usd: float,
    tool_runtime_cost_usd: float,
    platform_overhead_cost_usd: float,
    pricing_assumptions: list[str],
    used_rate_keys: list[str],
    local_cost_policy: CodexLocalCostPolicy | None = None,
    precomputed_warnings: list[str] | None = None,
) -> AgenticCostBreakdown:
    total_usd = round(llm_runtime_cost_usd + tool_runtime_cost_usd + platform_overhead_cost_usd, 4)
    total_cop = round(total_usd * profile.cop_per_usd, 2)
    selected_keys = set(used_rate_keys)
    present_keys = {item.metric_key for item in profile.rates}
    warnings = list(precomputed_warnings or [])
    missing_keys = sorted(selected_keys - present_keys)
    if missing_keys:
        warnings.append(
            "Faltan metricas de pricing en el perfil activo: " + ", ".join(missing_keys)
        )
    if profile.cop_per_usd <= 0:
        warnings.append("El pricing profile activo no define una tasa COP/USD positiva; el costo variable convertido a COP puede quedar subestimado.")
    snapshot = EstimationPricingSnapshot(
        provider=profile.provider,
        profile_key=profile.profile_key,
        label=profile.label,
        model=provider_model,
        pricing_mode=economic_model,
        effective_from=profile.effective_from,
        is_local_inference=profile.is_local_inference,
        local_cost_policy=local_cost_policy or profile.local_cost_policy,
        cop_per_usd=profile.cop_per_usd,
        assumptions=pricing_assumptions,
        rates=[item for item in profile.rates if item.metric_key in set(used_rate_keys)],
    )
    return AgenticCostBreakdown(
        provider_model=provider_model,
        economic_model=economic_model,
        human_supervision_cost=round(human_supervision_cost, 2),
        llm_runtime_cost_usd=round(llm_runtime_cost_usd, 4),
        tool_runtime_cost_usd=round(tool_runtime_cost_usd, 4),
        platform_overhead_cost_usd=round(platform_overhead_cost_usd, 4),
        provider_runtime_cost_total_usd=total_usd,
        provider_runtime_cost_total_cop=total_cop,
        pricing_assumptions=pricing_assumptions,
        warnings=warnings,
        pricing_snapshot=snapshot,
    )


def _resolve_active_profile(
    pricing_profiles: dict[LLMProviderKey, list[LLMPricingProfile]],
    provider: LLMProviderKey,
    *,
    local_cost_policy: CodexLocalCostPolicy | None,
) -> ResolvedPricingProfile:
    candidates = pricing_profiles.get(provider, [])
    if not candidates:
        return ResolvedPricingProfile(profile=None, warnings=[])
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (item.effective_from or "", item.profile_key),
        reverse=True,
    )
    primary = ordered_candidates[0]
    required_keys = _required_rate_keys(
        provider,
        local_cost_policy=(
            local_cost_policy
            if provider == LLMProviderKey.codex_local
            else None
        ),
    )
    primary_missing_keys = _missing_required_rate_keys(primary, required_keys)
    if not primary_missing_keys:
        return ResolvedPricingProfile(profile=primary, warnings=[])

    fallback = next(
        (item for item in ordered_candidates[1:] if not _missing_required_rate_keys(item, required_keys)),
        None,
    )
    if fallback is not None:
        return ResolvedPricingProfile(
            profile=fallback,
            warnings=[
                "El profile mas reciente del proveedor activo esta incompleto; se reutilizo el ultimo profile con cobertura total de metricas.",
                f"Fallback aplicado: {fallback.profile_key} ({fallback.effective_from or 'sin fecha declarada'}).",
            ],
        )

    return ResolvedPricingProfile(
        profile=primary,
        warnings=[
            "El pricing profile vigente esta incompleto; las metricas faltantes se estiman en cero hasta completar el catalogo.",
            "Metricas faltantes: " + ", ".join(primary_missing_keys),
        ],
    )


def _required_rate_keys(
    provider: LLMProviderKey,
    *,
    local_cost_policy: CodexLocalCostPolicy | None,
) -> set[str]:
    if provider == LLMProviderKey.deepseek:
        return {
            "fast_input_cache_hit_tokens_m",
            "fast_input_cache_miss_tokens_m",
            "fast_output_tokens_m",
            "reasoning_input_cache_hit_tokens_m",
            "reasoning_input_cache_miss_tokens_m",
            "reasoning_output_tokens_m",
            "tool_calls_k",
            "provider_session",
        }
    if provider == LLMProviderKey.codex_local:
        required = {
            "compute_hour_core",
            "tool_calls_k",
            "local_session",
        }
        if local_cost_policy == CodexLocalCostPolicy.hybrid:
            required.add("workstation_hour_hybrid")
        elif local_cost_policy == CodexLocalCostPolicy.fully_loaded:
            required.add("workstation_hour_fully_loaded")
        return required
    return {
        "fast_input_tokens_m",
        "fast_cached_input_tokens_m",
        "fast_output_tokens_m",
        "reasoning_input_tokens_m",
        "reasoning_cached_input_tokens_m",
        "reasoning_output_tokens_m",
        "tool_calls_k",
        "provider_session",
    }


def _missing_required_rate_keys(
    profile: LLMPricingProfile,
    required_keys: set[str],
) -> list[str]:
    present_keys = {item.metric_key for item in profile.rates}
    return sorted(required_keys - present_keys)


def _resolve_provider_model(runtime_settings: LLMRuntimeSettings) -> str:
    if runtime_settings.active_provider == LLMProviderKey.deepseek:
        return f"fast={runtime_settings.deepseek.fast_model} | reasoning={runtime_settings.deepseek.reasoning_model}"
    if runtime_settings.active_provider == LLMProviderKey.codex_local:
        return f"command={runtime_settings.codex_local.command} | model={runtime_settings.codex_local.model}"
    return f"fast={runtime_settings.openai.fast_model} | reasoning={runtime_settings.openai.reasoning_model}"


def _resolve_economic_model(
    runtime_settings: LLMRuntimeSettings,
    profile: LLMPricingProfile | None,
) -> str:
    if runtime_settings.active_provider == LLMProviderKey.codex_local:
        return runtime_settings.codex_local.cost_policy.value
    if profile is not None and profile.mode:
        return profile.mode
    return "standard"


def _base_input_tokens_m(signals: EstimationSignalsLike) -> float:
    base = {
        EstimationMaturityStage.canvas: 0.16,
        EstimationMaturityStage.blueprint: 0.34,
        EstimationMaturityStage.ready_to_build: 0.58,
    }[signals.maturity_stage]
    complexity_factor = {
        EstimationComplexityLevel.simple: 0.82,
        EstimationComplexityLevel.moderate: 1.0,
        EstimationComplexityLevel.complex: 1.32,
        EstimationComplexityLevel.critical: 1.68,
    }[signals.complexity]
    scope_factor = _clamp(signals.scope_points / 52, 0.78, 1.42)
    risk_factor = 1 + min(0.18, signals.side_effect_tools * 0.03 + signals.blocking_gaps * 0.02)
    return base * complexity_factor * scope_factor * risk_factor


def _provider_mix(maturity_stage: EstimationMaturityStage) -> tuple[float, float]:
    if maturity_stage == EstimationMaturityStage.canvas:
        return 0.78, 0.22
    if maturity_stage == EstimationMaturityStage.ready_to_build:
        return 0.42, 0.58
    return 0.56, 0.44


def _local_compute_hours(signals: EstimationSignalsLike) -> float:
    base = {
        EstimationMaturityStage.canvas: 0.7,
        EstimationMaturityStage.blueprint: 1.35,
        EstimationMaturityStage.ready_to_build: 2.25,
    }[signals.maturity_stage]
    complexity_factor = {
        EstimationComplexityLevel.simple: 0.85,
        EstimationComplexityLevel.moderate: 1.0,
        EstimationComplexityLevel.complex: 1.28,
        EstimationComplexityLevel.critical: 1.62,
    }[signals.complexity]
    activity_factor = 1 + min(0.45, signals.tool_count * 0.04 + signals.workflow_steps * 0.03 + signals.evaluation_cases * 0.02)
    return round(base * complexity_factor * activity_factor, 3)


def _tool_call_units_k(signals: EstimationSignalsLike) -> float:
    return round(max(1.0, float(signals.tool_count + max(signals.evaluation_cases, 1) + max(signals.workflow_steps / 2, 1))), 3)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
