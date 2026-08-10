from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from app.diagnostics import normalize_autonomy_level, normalize_case_type
from app.models import (
    BlueprintArtifact,
    CanvasArtifact,
    DesignAlternative,
    DesignFitMatrixEntry,
    DesignRequirementCoverageEntry,
    DiscoveryArtifact,
    DiscoveryInput,
    EstimationAnalysisArtifact,
    EstimationReportArtifact,
)
from app.services.rules import infer_case_type, normalize_text


class BlueprintNarrativeOutput(BaseModel):
    narrative: str


class StructuredInsight(BaseModel):
    key: str = ""
    statement: str = ""
    source_refs: list[str] = []
    confidence: float = 0.0


class GuidedAnswerOption(BaseModel):
    key: str = ""
    label: str = ""
    description: str = ""
    impact: str = ""
    example: str = ""
    recommended: bool = False
    confidence: float = 0.0
    source_refs: list[str] = []


class PrioritizedQuestion(BaseModel):
    key: str = ""
    question: str = ""
    rationale: str = ""
    priority: Literal["high", "medium", "low"] = "medium"
    blocking_stages: list[str] = []
    suggested_answer: str = ""
    answer_options: list[GuidedAnswerOption] = []


class DiscoveryAnalysisOutput(BaseModel):
    summary: str = ""
    facts: list[StructuredInsight] = []
    inferred_needs: list[StructuredInsight] = []
    assumptions: list[StructuredInsight] = []
    ambiguities: list[StructuredInsight] = []
    open_questions: list[PrioritizedQuestion] = []
    domain_signals: list[StructuredInsight] = []
    risk_signals: list[StructuredInsight] = []
    sensitive_data_signals: list[StructuredInsight] = []
    missing_information: list[str] = []
    evidence_refs: list[str] = []
    confidence: float = 0.0
    normalized_discovery_candidate: DiscoveryArtifact = DiscoveryArtifact()


DefinitionPriority = Literal["high", "medium", "low"]
DefinitionItemStatus = Literal["proposed", "accepted", "rejected", "needs_input"]
TraceCoverageStatus = Literal["covered", "partial", "gap"]


class DefinitionEntityBase(BaseModel):
    key: str = ""
    title: str = ""
    priority: DefinitionPriority = "medium"
    status: DefinitionItemStatus = "proposed"
    source_refs: list[str] = []
    rationale: str = ""
    acceptance: list[str] = []


class FunctionalRequirement(DefinitionEntityBase):
    requirement: str = ""
    actor: str = ""
    trigger: str = ""
    happy_path: str = ""
    exceptions: list[str] = []


class NonFunctionalRequirement(DefinitionEntityBase):
    requirement: str = ""
    category: str = ""
    metric: str = ""
    target: str = ""


class BusinessRule(DefinitionEntityBase):
    rule: str = ""
    owner: str = ""


class AcceptanceCriterion(DefinitionEntityBase):
    criterion: str = ""
    requirement_keys: list[str] = []


class Dependency(DefinitionEntityBase):
    dependency: str = ""
    dependency_type: str = ""
    owner: str = ""


class Assumption(DefinitionEntityBase):
    assumption: str = ""


class OpenQuestion(DefinitionEntityBase):
    question: str = ""
    blocking: bool = False
    impacted_sections: list[str] = []
    suggested_answer: str = ""
    answer_options: list[GuidedAnswerOption] = []


class RequirementTraceEntry(BaseModel):
    key: str = ""
    requirement_key: str = ""
    source_ref: str = ""
    rationale: str = ""
    coverage_status: TraceCoverageStatus = "covered"


class RequirementEntry(BaseModel):
    key: str = ""
    category: Literal["functional", "non_functional", "business_rule", "constraint"] = "functional"
    requirement: str = ""
    rationale: str = ""
    source_refs: list[str] = []
    priority: Literal["high", "medium", "low"] = "medium"


class DefinitionValidationSummary(BaseModel):
    duplicate_keys: list[str] = []
    duplicate_signals: list[str] = []
    contradictions: list[str] = []
    vague_nfrs: list[str] = []
    missing_acceptance: list[str] = []
    untraced_items: list[str] = []
    blocking_open_questions: list[str] = []
    blocking_issues: list[str] = []
    coverage_ratio: float = 0.0


class RequirementsDefinitionOutput(BaseModel):
    summary: str = ""
    measurable_objectives: list[str] = []
    functional_requirements: list[FunctionalRequirement] = []
    non_functional_requirements: list[NonFunctionalRequirement] = []
    business_rules: list[BusinessRule] = []
    acceptance_criteria: list[AcceptanceCriterion] = []
    dependencies: list[Dependency] = []
    assumptions: list[Assumption] = []
    open_questions: list[OpenQuestion] = []
    traceability: list[RequirementTraceEntry] = []
    evidence_refs: list[str] = []
    confidence: float = 0.0
    validation: DefinitionValidationSummary = DefinitionValidationSummary()
    canvas_projection: CanvasArtifact = CanvasArtifact()


class DesignDecision(BaseModel):
    dimension: str = ""
    selected_option: str = ""
    rationale: str = ""
    tradeoffs: list[str] = []
    source_refs: list[str] = []


class AgentDesignProposalOutput(BaseModel):
    summary: str = ""
    alternatives: list[DesignAlternative] = []
    fit_matrix: list[DesignFitMatrixEntry] = []
    recommended_alternative_key: str = ""
    decision_rationale: str = ""
    requirements_coverage: list[DesignRequirementCoverageEntry] = []
    evidence_refs: list[str] = []
    confidence: float = 0.0
    architecture: str = ""
    reasoning_pattern: str = ""
    memory_strategy: str = ""
    coordination_model: str = ""
    tooling_principles: list[str] = []
    design_decisions: list[DesignDecision] = []
    open_questions: list[str] = []
    guided_questions: list[PrioritizedQuestion] = []
    narrative: str = ""


class CritiqueFinding(BaseModel):
    finding_key: str = ""
    title: str = ""
    severity: Literal["info", "warning", "blocking"] = "warning"
    detail: str = ""
    suggested_action: str = ""
    source_refs: list[str] = []


class DesignCritiqueOutput(BaseModel):
    overall_status: Literal["accepted", "needs_revision", "blocked"] = "needs_revision"
    summary: str = ""
    findings: list[CritiqueFinding] = []
    contradictions: list[str] = []
    missing_evidence: list[str] = []


class MemoryArchitectureRecommendationOutput(BaseModel):
    memory_strategy: str = ""
    short_term_strategy: str = ""
    long_term_strategy: str = ""
    retrieval_strategy: str = ""
    storage_layers: list[str] = []
    write_policy: str = ""
    pruning_policy: str = ""
    security_notes: list[str] = []
    open_questions: list[str] = []
    guided_questions: list[PrioritizedQuestion] = []
    rationale: str = ""


class MemoryArchitectureCritiqueOutput(BaseModel):
    overall_status: Literal["accepted", "needs_revision", "blocked"] = "needs_revision"
    summary: str = ""
    findings: list[CritiqueFinding] = []
    contradictions: list[str] = []
    missing_evidence: list[str] = []


class ValidationScenarioItem(BaseModel):
    scenario_key: str = ""
    title: str = ""
    objective: str = ""
    steps: list[str] = []
    expected_outcomes: list[str] = []
    failure_signals: list[str] = []
    priority: Literal["high", "medium", "low"] = "medium"


class ValidationScenarioGenerationOutput(BaseModel):
    summary: str = ""
    scenarios: list[ValidationScenarioItem] = []
    coverage_gaps: list[str] = []


class ValidationSimulationOutput(BaseModel):
    scenario_key: str = ""
    result_status: Literal["pass", "needs_revision", "fail"] = "needs_revision"
    simulated_transcript: list[str] = []
    observed_decisions: list[str] = []
    tool_interactions: list[str] = []
    issues: list[str] = []


class ValidationRunJudgmentOutput(BaseModel):
    scenario_key: str = ""
    judgment: Literal["pass", "needs_revision", "fail"] = "needs_revision"
    summary: str = ""
    findings: list[CritiqueFinding] = []
    score: int = 0


class EstimationRiskAnalysisOutput(EstimationAnalysisArtifact):
    pass


class DiscoveryAnalysisInput(BaseModel):
    discovery_capture: DiscoveryInput
    analysis_goal: str = ""
    known_gaps: list[str] = []
    source_refs: list[str] = []


class RequirementsDefinitionInput(BaseModel):
    discovery: DiscoveryArtifact
    canvas: CanvasArtifact | None = None
    known_constraints: list[str] = []
    source_refs: list[str] = []


class AgentDesignInput(BaseModel):
    discovery: DiscoveryArtifact
    canvas: CanvasArtifact
    current_blueprint: BlueprintArtifact | None = None
    requirement_digest: list[str] = []
    source_refs: list[str] = []


class AgentDesignCritiqueInput(BaseModel):
    discovery: DiscoveryArtifact
    canvas: CanvasArtifact
    proposal: AgentDesignProposalOutput
    source_refs: list[str] = []


class MemoryArchitectureInput(BaseModel):
    blueprint: BlueprintArtifact
    discovery: DiscoveryArtifact | None = None
    canvas: CanvasArtifact | None = None
    approved_tool_names: list[str] = []
    source_refs: list[str] = []


class MemoryArchitectureCritiqueInput(BaseModel):
    blueprint: BlueprintArtifact
    proposal: MemoryArchitectureRecommendationOutput
    approved_tool_names: list[str] = []
    source_refs: list[str] = []


class ValidationScenarioGenerationInput(BaseModel):
    blueprint: BlueprintArtifact
    discovery: DiscoveryArtifact | None = None
    canvas: CanvasArtifact | None = None
    focus_areas: list[str] = []
    source_refs: list[str] = []


class ValidationScenarioSimulationInput(BaseModel):
    blueprint: BlueprintArtifact
    scenario: ValidationScenarioItem
    source_refs: list[str] = []


class ValidationRunJudgmentInput(BaseModel):
    simulation: ValidationSimulationOutput
    blueprint: BlueprintArtifact | None = None
    source_refs: list[str] = []


class EstimationRiskAnalysisInput(BaseModel):
    blueprint: BlueprintArtifact | None = None
    estimation_report: EstimationReportArtifact
    pricing_summary: list[str] = []
    validation_summary: list[str] = []
    workspace_calibration_summary: list[str] = []
    benchmark_hints: list[str] = []
    source_refs: list[str] = []


@dataclass
class LLMArtifactResult:
    artifact: BaseModel | None
    warning: str | None = None
    provider_key: str | None = None
    execution_backend: str | None = None
    execution_mode: str | None = None
    shadow_provider_key: str | None = None
    route_reason: str | None = None
    knowledge_access_backend: str | None = None
    effective_context_backend: str | None = None
    context_used_sources: list[dict[str, object]] = field(default_factory=list)
    context_stats: dict[str, object] = field(default_factory=dict)
    capability_key: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    schema_validation_status: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    failure_kind: str | None = None
    failure_detail: str | None = None
    retry_count: int = 0
    fallback_used: bool = False
    degraded: bool = False
    capability_policy: dict[str, Any] = field(default_factory=dict)
    rollout_comparison: dict[str, Any] = field(default_factory=dict)


def validate_or_repair_structured_payload(
    payload: Any,
    output_model: type[BaseModel],
) -> tuple[BaseModel, str]:
    if isinstance(payload, dict):
        model_fields = getattr(output_model, "model_fields", {})
        for wrapper_key in ("result", "data", "output", "artifact", "payload"):
            nested = payload.get(wrapper_key)
            if wrapper_key not in model_fields and isinstance(nested, dict):
                return output_model.model_validate(nested), "repaired_wrapper_unwrap"
        if len(payload) == 1:
            only_key, nested = next(iter(payload.items()))
            if only_key not in model_fields and isinstance(nested, dict):
                return output_model.model_validate(nested), "repaired_single_key_unwrap"

    try:
        return output_model.model_validate(payload), "valid"
    except Exception:
        pass

    return output_model.model_validate(payload), "invalid"


def merge_warnings(*warnings: str | None) -> str | None:
    normalized: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        token = (warning or "").strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(token)
    return " ".join(normalized) if normalized else None


def sanitize_discovery(artifact: DiscoveryArtifact) -> DiscoveryArtifact:
    normalized_autonomy = normalize_autonomy_level(artifact.autonomy_level)
    normalized_case_type = normalize_case_type(artifact.case_type) or infer_case_type(
        artifact.problem_statement,
        artifact.desired_outcome,
        normalized_autonomy,
    )
    return artifact.model_copy(
        update={
            "problem_statement": normalize_text(artifact.problem_statement),
            "current_user": normalize_text(artifact.current_user),
            "current_process": normalize_text(artifact.current_process),
            "desired_outcome": normalize_text(artifact.desired_outcome),
            "autonomy_level": normalized_autonomy,
            "constraints": [normalize_text(item) for item in artifact.constraints if normalize_text(item)],
            "operational_baseline": artifact.operational_baseline.model_copy(
                update={
                    "current_time_spent": normalize_text(artifact.operational_baseline.current_time_spent),
                    "current_cost": normalize_text(artifact.operational_baseline.current_cost),
                    "frequent_errors": [
                        normalize_text(item)
                        for item in artifact.operational_baseline.frequent_errors
                        if normalize_text(item)
                    ],
                    "automation_opportunities": [
                        normalize_text(item)
                        for item in artifact.operational_baseline.automation_opportunities
                        if normalize_text(item)
                    ],
                }
            ),
            "mvp_definition": artifact.mvp_definition.model_copy(
                update={
                    "v1_scope": [
                        normalize_text(item) for item in artifact.mvp_definition.v1_scope if normalize_text(item)
                    ],
                    "out_of_scope": [
                        normalize_text(item)
                        for item in artifact.mvp_definition.out_of_scope
                        if normalize_text(item)
                    ],
                    "north_star_metric": normalize_text(artifact.mvp_definition.north_star_metric),
                    "non_delegable_decisions": [
                        normalize_text(item)
                        for item in artifact.mvp_definition.non_delegable_decisions
                        if normalize_text(item)
                    ],
                }
            ),
            "case_type": normalized_case_type,
            "value_statement": normalize_text(artifact.value_statement),
        }
    )


def sanitize_canvas(artifact: CanvasArtifact) -> CanvasArtifact:
    return artifact.model_copy(
        update={
            "user_goal": normalize_text(artifact.user_goal),
            "mvp_scope": [normalize_text(item) for item in artifact.mvp_scope if normalize_text(item)],
            "out_of_scope": [normalize_text(item) for item in artifact.out_of_scope if normalize_text(item)],
            "success_metric": normalize_text(artifact.success_metric),
            "primary_risk": normalize_text(artifact.primary_risk),
            "agent_profile": artifact.agent_profile.model_copy(
                update={
                    "mission": normalize_text(artifact.agent_profile.mission),
                    "primary_user": normalize_text(artifact.agent_profile.primary_user),
                    "agent_task": normalize_text(artifact.agent_profile.agent_task),
                    "allowed_decisions": [
                        normalize_text(item)
                        for item in artifact.agent_profile.allowed_decisions
                        if normalize_text(item)
                    ],
                    "prohibited_decisions": [
                        normalize_text(item)
                        for item in artifact.agent_profile.prohibited_decisions
                        if normalize_text(item)
                    ],
                    "key_inputs": [
                        normalize_text(item) for item in artifact.agent_profile.key_inputs if normalize_text(item)
                    ],
                    "expected_outputs": [
                        normalize_text(item)
                        for item in artifact.agent_profile.expected_outputs
                        if normalize_text(item)
                    ],
                    "human_approvals": [
                        normalize_text(item)
                        for item in artifact.agent_profile.human_approvals
                        if normalize_text(item)
                    ],
                    "success_metrics": [
                        normalize_text(item)
                        for item in artifact.agent_profile.success_metrics
                        if normalize_text(item)
                    ],
                }
            ),
        }
    )
