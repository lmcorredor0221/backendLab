from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models import (
    AutomationAdjustmentRule,
    AutomationFamilyAssessment,
    AutomationMatrixProfile,
    EstimationComplexityLevel,
    EstimationMaturityStage,
)


WORKSTREAM_ORDER = ("backend", "frontend", "integrations", "data", "qa", "devops")
DEFAULT_AUTOMATION_BY_WORKSTREAM: dict[EstimationMaturityStage, dict[str, int]] = {
    EstimationMaturityStage.canvas: {
        "backend": 44,
        "frontend": 38,
        "integrations": 22,
        "data": 30,
        "qa": 32,
        "devops": 20,
    },
    EstimationMaturityStage.blueprint: {
        "backend": 52,
        "frontend": 46,
        "integrations": 28,
        "data": 38,
        "qa": 42,
        "devops": 26,
    },
    EstimationMaturityStage.ready_to_build: {
        "backend": 58,
        "frontend": 50,
        "integrations": 33,
        "data": 44,
        "qa": 48,
        "devops": 30,
    },
}
COMPLEXITY_AUTOMATION_ADJUSTMENT: dict[EstimationComplexityLevel, int] = {
    EstimationComplexityLevel.simple: 4,
    EstimationComplexityLevel.moderate: 0,
    EstimationComplexityLevel.complex: -6,
    EstimationComplexityLevel.critical: -12,
}
WORKSTREAM_FAMILY_GROUPS: dict[str, list[str]] = {
    "backend": ["architecture_spec", "tool_schemas", "implementation_code"],
    "frontend": ["prd_narrative", "architecture_spec", "acp_packaging"],
    "integrations": ["external_api_contracts", "tool_schemas", "runtime_config"],
    "data": ["knowledge_retrieval", "runtime_config"],
    "qa": ["evaluation_assets", "observability"],
    "devops": ["runtime_config", "deployment_infra", "observability"],
}


class AutomationSignalsLike(Protocol):
    maturity_stage: EstimationMaturityStage
    complexity: EstimationComplexityLevel
    project_archetype: str
    blocking_gaps: int
    open_questions: int
    design_gap_count: int
    implementation_gap_count: int
    design_open_questions: int
    implementation_open_questions: int
    tool_count: int
    side_effect_tools: int
    implementation_side_effect_tools: int
    external_tool_count: int
    safety_checks: int
    memory_layers: int
    observability_signals: int
    evaluation_cases: int
    schema_validated_tools: int
    acp_ready_to_build: bool
    evaluation_complete: bool


@dataclass(frozen=True)
class AutomationMatrixResult:
    automation_by_workstream: dict[str, int]
    automation_by_artifact_family: dict[str, int]
    family_assessments: list[AutomationFamilyAssessment]


def build_automation_matrix(
    signals: AutomationSignalsLike,
    automation_profiles: dict[str, AutomationMatrixProfile],
) -> AutomationMatrixResult:
    assessments: list[AutomationFamilyAssessment] = []
    for family_key in _resolve_relevant_family_keys(signals):
        profile = automation_profiles.get(family_key)
        if profile is None:
            continue
        assessments.append(_build_family_assessment(signals, profile))

    automation_by_family = {item.family_key: item.coverage_percent for item in assessments}
    automation_by_workstream = _build_automation_by_workstream(signals, automation_by_family)
    return AutomationMatrixResult(
        automation_by_workstream=automation_by_workstream,
        automation_by_artifact_family=automation_by_family,
        family_assessments=assessments,
    )


def _resolve_relevant_family_keys(signals: AutomationSignalsLike) -> list[str]:
    family_keys = ["discovery_canvas", "prd_narrative"]
    if signals.maturity_stage != EstimationMaturityStage.canvas:
        family_keys.extend(
            [
                "architecture_spec",
                "prompts_playbooks",
                "tool_schemas",
                "runtime_config",
                "implementation_code",
            ]
        )
    if signals.evaluation_cases > 0:
        family_keys.append("evaluation_assets")
    if signals.memory_layers > 0 or signals.project_archetype == "agentic_platform":
        family_keys.append("knowledge_retrieval")
    if signals.observability_signals > 0:
        family_keys.append("observability")
    if signals.external_tool_count > 0 or signals.project_archetype == "enterprise_integrations":
        family_keys.append("external_api_contracts")
    if signals.maturity_stage == EstimationMaturityStage.ready_to_build:
        family_keys.extend(["deployment_infra", "acp_packaging"])
    return _dedupe(family_keys)


def _build_family_assessment(
    signals: AutomationSignalsLike,
    profile: AutomationMatrixProfile,
) -> AutomationFamilyAssessment:
    band = _get_band(profile, signals.complexity)
    if band is None:
        return AutomationFamilyAssessment(
            family_key=profile.family_key,
            label=profile.label or profile.family_key,
            complexity=signals.complexity,
            coverage_percent=0,
            risk_tier="unknown",
            notes=profile.notes,
        )

    penalty_rules = _resolve_triggered_rules(profile.penalty_rules, signals, profile.family_key)
    bonus_rules = _resolve_triggered_rules(profile.bonus_rules, signals, profile.family_key)
    blocking_conditions = [key for key in profile.blocking_conditions if _rule_is_triggered(key, signals, profile.family_key)]

    penalty_delta = sum(rule.delta_percent for rule in penalty_rules)
    bonus_delta = sum(rule.delta_percent for rule in bonus_rules)
    coverage = int(round(_clamp(band.base_automation_percent - penalty_delta + bonus_delta, 10, band.automation_ceiling_percent)))

    non_automatable_reasons = _build_non_automatable_reasons(
        family_key=profile.family_key,
        coverage=coverage,
        blocking_conditions=blocking_conditions,
        mandatory_human_review=band.mandatory_human_review,
        signals=signals,
    )

    return AutomationFamilyAssessment(
        family_key=profile.family_key,
        label=profile.label or profile.family_key,
        complexity=signals.complexity,
        coverage_percent=coverage,
        risk_tier=band.risk_tier,
        mandatory_human_review=band.mandatory_human_review,
        blocking_conditions=blocking_conditions,
        penalties_applied=[rule.label for rule in penalty_rules],
        bonuses_applied=[rule.label for rule in bonus_rules],
        non_automatable_reasons=non_automatable_reasons,
        notes=profile.notes,
    )


def _build_non_automatable_reasons(
    *,
    family_key: str,
    coverage: int,
    blocking_conditions: list[str],
    mandatory_human_review: bool,
    signals: AutomationSignalsLike,
) -> list[str]:
    reasons: list[str] = []
    if "api_contract_missing" in blocking_conditions or "sandbox_unknown" in blocking_conditions:
        reasons.append("Faltan contratos o sandbox verificables de terceros.")
    if "target_environment_unknown" in blocking_conditions or "secret_owner_missing" in blocking_conditions:
        reasons.append("Deployment, secretos u ownership operativo siguen abiertos.")
    if "knowledge_owner_missing" in blocking_conditions or "refresh_policy_missing" in blocking_conditions:
        reasons.append("Knowledge, owner o refresh policy todavia requieren definicion humana.")
    if "tool_side_effects_not_governed" in blocking_conditions or "production_side_effects" in blocking_conditions:
        reasons.append("Hay side effects productivos que no deben automatizarse sin gate humano.")
    if "provider_or_secret_source_unknown" in blocking_conditions:
        reasons.append("Runtime, modelo o fuente de secretos aun no estan completamente cerrados.")
    if "regulated_domain_requirements" in blocking_conditions:
        reasons.append("El dominio exige control humano por riesgo regulatorio u operacional.")
    if coverage <= 32:
        reasons.append("Cobertura automatizable demasiado baja para delegar construccion end-to-end.")
    if family_key == "implementation_code" and _implementation_side_effect_tools(signals) > 0:
        reasons.append("El codigo ejecutable con integraciones activas requiere hardening y aprobacion humana.")
    if not reasons and mandatory_human_review and coverage <= 45:
        reasons.append("Aunque es parcialmente automatizable, exige revision humana antes de pasar a build.")
    return _dedupe(reasons)


def _resolve_triggered_rules(
    rules: list[AutomationAdjustmentRule],
    signals: AutomationSignalsLike,
    family_key: str,
) -> list[AutomationAdjustmentRule]:
    return [rule for rule in rules if _rule_is_triggered(rule.rule_key, signals, family_key)]


def _rule_is_triggered(rule_key: str, signals: AutomationSignalsLike, family_key: str) -> bool:
    if rule_key == "api_contract_missing":
        return (signals.project_archetype == "enterprise_integrations" or signals.external_tool_count > 0) and signals.maturity_stage != EstimationMaturityStage.ready_to_build
    if rule_key == "sandbox_unknown":
        return (signals.project_archetype == "enterprise_integrations" or signals.external_tool_count > 0) and (signals.blocking_gaps > 0 or signals.open_questions > 0)
    if rule_key == "tool_side_effects_not_governed":
        return _implementation_side_effect_tools(signals) > 0
    if rule_key == "provider_or_secret_source_unknown":
        return signals.maturity_stage != EstimationMaturityStage.ready_to_build
    if rule_key == "target_environment_unknown":
        return signals.maturity_stage != EstimationMaturityStage.ready_to_build
    if rule_key == "secret_owner_missing":
        return signals.maturity_stage != EstimationMaturityStage.ready_to_build or signals.open_questions > 0
    if rule_key == "knowledge_owner_missing":
        return family_key == "knowledge_retrieval" and signals.maturity_stage != EstimationMaturityStage.ready_to_build
    if rule_key == "refresh_policy_missing":
        return family_key == "knowledge_retrieval" and (signals.memory_layers == 0 or signals.open_questions > 0)
    if rule_key == "production_side_effects":
        return _implementation_side_effect_tools(signals) > 0
    if rule_key == "regulated_domain_requirements":
        return _implementation_side_effect_tools(signals) > 0 and signals.safety_checks >= 4
    if rule_key == "schema_complete":
        return signals.tool_count > 0 and signals.schema_validated_tools == signals.tool_count
    if rule_key == "acp_ready_to_build":
        return signals.acp_ready_to_build
    if rule_key == "evaluation_complete":
        return signals.evaluation_complete
    return False


def _build_automation_by_workstream(
    signals: AutomationSignalsLike,
    automation_by_family: dict[str, int],
) -> dict[str, int]:
    default_values = DEFAULT_AUTOMATION_BY_WORKSTREAM[signals.maturity_stage]
    coverage: dict[str, int] = {}
    
    # Progressive boost: Blueprint/ACP uncertainty is capped separately from execution-time uncertainty.
    # This avoids treating environment-only questions as product-design gaps.
    acp_boost = max(24, 48 - _residual_uncertainty_penalty(signals))

    for workstream_key in WORKSTREAM_ORDER:
        family_scores = [automation_by_family[key] for key in WORKSTREAM_FAMILY_GROUPS[workstream_key] if key in automation_by_family]
        if family_scores:
            value = max(default_values[workstream_key], round(sum(family_scores) / len(family_scores)))
        else:
            value = default_values[workstream_key]

        value += acp_boost
        value += COMPLEXITY_AUTOMATION_ADJUSTMENT[signals.complexity]
        if workstream_key == "integrations" and _implementation_side_effect_tools(signals) > 0:
            value -= 8
        if workstream_key == "backend" and _implementation_side_effect_tools(signals) > 0:
            value -= 4
        if workstream_key == "qa" and signals.evaluation_cases == 0 and signals.maturity_stage != EstimationMaturityStage.canvas:
            value -= 5
        if workstream_key == "devops" and signals.maturity_stage != EstimationMaturityStage.ready_to_build:
            value -= 4
        
        ceiling = 92 if (signals.blocking_gaps == 0 and signals.open_questions == 0) else 88
        coverage[workstream_key] = int(_clamp(value, 12, ceiling))
    return coverage


def _residual_uncertainty_penalty(signals: AutomationSignalsLike) -> int:
    design_gap_count = _signal_int(signals, "design_gap_count", signals.blocking_gaps)
    implementation_gap_count = _signal_int(signals, "implementation_gap_count", 0)
    design_open_questions = _signal_int(signals, "design_open_questions", signals.open_questions)
    implementation_open_questions = _signal_int(signals, "implementation_open_questions", 0)
    question_penalty = min(6, design_open_questions * 2 + implementation_open_questions)
    return design_gap_count * 4 + implementation_gap_count * 2 + question_penalty


def _get_band(profile: AutomationMatrixProfile, complexity: EstimationComplexityLevel):
    for band in profile.bands:
        if band.complexity == complexity:
            return band
    return None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _implementation_side_effect_tools(signals: AutomationSignalsLike) -> int:
    return getattr(signals, "implementation_side_effect_tools", signals.side_effect_tools)


def _signal_int(signals: AutomationSignalsLike, attr_name: str, fallback: int) -> int:
    value = getattr(signals, attr_name, fallback)
    return int(value or 0)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
