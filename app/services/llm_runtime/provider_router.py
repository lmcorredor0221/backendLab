from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID, uuid4

from app.models import (
    AgentExecutionBackend,
    BlueprintArtifact,
    CanvasArtifact,
    DiscoveryArtifact,
    DiscoveryInput,
    LLMProviderKey,
    LLMRuntimeSettings,
    ToolRecommendationPromptInput,
)
from app.services.diagram_center.contracts import DiagramGenerationInput
from app.services.llm_runtime.builder_contracts import (
    AgentDesignCritiqueInput,
    AgentDesignInput,
    BlueprintNarrativeOutput,
    DiscoveryAnalysisInput,
    EstimationRiskAnalysisInput,
    LLMArtifactResult,
    MemoryArchitectureCritiqueInput,
    MemoryArchitectureInput,
    RequirementsDefinitionInput,
    ValidationRunJudgmentInput,
    ValidationScenarioGenerationInput,
    ValidationScenarioSimulationInput,
    merge_warnings,
)
from app.services.llm_runtime.capability_registry import (
    CAPABILITY_ALIASES,
    BuilderCapability,
    get_builder_capability_spec,
)
from app.services.llm_runtime.stage_context_types import StageContextBundle, build_llm_call_context


class BuilderExecutionMode(StrEnum):
    primary = "primary"
    shadow = "shadow"
    staged = "staged"


@dataclass(frozen=True)
class BuilderRouteDecision:
    capability: BuilderCapability
    selected_provider: LLMProviderKey
    execution_mode: BuilderExecutionMode
    execution_backend: AgentExecutionBackend
    fallback_provider: LLMProviderKey | None
    shadow_provider: LLMProviderKey | None
    reason: str


class BuilderProviderService(Protocol):
    def can_attempt(self) -> bool: ...

    def is_available(self) -> bool: ...

    def provider_summary(self) -> dict[str, str | bool]: ...

    def normalize_discovery(
        self,
        payload: DiscoveryInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def analyze_discovery(
        self,
        payload: DiscoveryAnalysisInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def build_canvas(
        self,
        discovery: DiscoveryArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def define_requirements(
        self,
        payload: RequirementsDefinitionInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def synthesize_blueprint_narrative(
        self,
        discovery: DiscoveryArtifact,
        canvas: CanvasArtifact,
        blueprint: BlueprintArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def propose_agent_design(
        self,
        payload: AgentDesignInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def critique_agent_design(
        self,
        payload: AgentDesignCritiqueInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def recommend_minimal_tools(
        self,
        prompt_input: ToolRecommendationPromptInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def recommend_memory_architecture(
        self,
        payload: MemoryArchitectureInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def critique_memory_architecture(
        self,
        payload: MemoryArchitectureCritiqueInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def generate_validation_scenarios(
        self,
        payload: ValidationScenarioGenerationInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def simulate_validation_scenario(
        self,
        payload: ValidationScenarioSimulationInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def judge_validation_run(
        self,
        payload: ValidationRunJudgmentInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def analyze_estimation_risks(
        self,
        payload: EstimationRiskAnalysisInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...

    def generate_diagram_model(
        self,
        payload: DiagramGenerationInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult: ...


_PROVIDER_LABELS = {
    LLMProviderKey.openai: "OpenAI",
    LLMProviderKey.deepseek: "DeepSeek",
    LLMProviderKey.codex_local: "Codex local",
    LLMProviderKey.antigravity_cli: "Antigravity CLI",
}


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _matches_rollout(values: list[str], capability: BuilderCapability) -> bool:
    normalized_values = {_normalize_token(item) for item in values if item.strip()}
    return bool(normalized_values.intersection(CAPABILITY_ALIASES[capability]))


def _provider_label(provider_key: LLMProviderKey) -> str:
    return _PROVIDER_LABELS.get(provider_key, provider_key.value)


def _token_usage_total(result: LLMArtifactResult) -> int:
    if not result.token_usage:
        return 0
    return int(result.token_usage.get("total_tokens", 0) or 0)


def _flatten_scalar_map(value: Any, prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_scalar_map(item, next_prefix))
        return flattened
    if isinstance(value, list):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            flattened.update(_flatten_scalar_map(item, next_prefix))
        return flattened
    if value in ("", None, [], {}):
        return {}
    flattened[prefix or "value"] = str(value)
    return flattened


def _collect_source_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source_refs", "evidence_refs", "citations"} and isinstance(item, list):
                refs.extend(str(entry).strip() for entry in item if str(entry).strip())
            else:
                refs.extend(_collect_source_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_collect_source_refs(item))
    return list(dict.fromkeys(refs))


def _compare_artifact_outputs(
    capability: BuilderCapability,
    primary_result: LLMArtifactResult,
    shadow_result: LLMArtifactResult,
) -> dict[str, Any]:
    primary_artifact = primary_result.artifact
    shadow_artifact = shadow_result.artifact
    if primary_artifact is None or shadow_artifact is None:
        return {
            "capability": capability.value,
            "comparison_status": "insufficient_data",
            "schema_match": False,
            "semantic_divergence": False,
            "coverage_overlap_pct": 0.0,
            "contradictions": [],
            "primary_source_ref_count": 0,
            "shadow_source_ref_count": 0,
            "token_cost_delta": _token_usage_total(shadow_result) - _token_usage_total(primary_result),
        }

    primary_payload = primary_artifact.model_dump(mode="json")
    shadow_payload = shadow_artifact.model_dump(mode="json")
    primary_flat = _flatten_scalar_map(primary_payload)
    shadow_flat = _flatten_scalar_map(shadow_payload)
    shared_keys = set(primary_flat).intersection(shadow_flat)
    all_keys = set(primary_flat).union(shadow_flat)
    contradictions = sorted(key for key in shared_keys if primary_flat[key] != shadow_flat[key])[:12]
    coverage_overlap_pct = round((len(shared_keys) / max(len(all_keys), 1)) * 100, 2)
    primary_refs = _collect_source_refs(primary_payload)
    shadow_refs = _collect_source_refs(shadow_payload)
    schema_match = primary_artifact.__class__ is shadow_artifact.__class__
    semantic_divergence = bool(contradictions)
    return {
        "capability": capability.value,
        "comparison_status": "compared",
        "schema_match": schema_match,
        "semantic_divergence": semantic_divergence,
        "coverage_overlap_pct": coverage_overlap_pct,
        "contradictions": contradictions,
        "primary_source_ref_count": len(primary_refs),
        "shadow_source_ref_count": len(shadow_refs),
        "token_cost_delta": _token_usage_total(shadow_result) - _token_usage_total(primary_result),
        "action": "keep_primary" if not semantic_divergence else "keep_primary_needs_review",
    }


class BuilderProviderRouter:
    def __init__(self, runtime_settings: LLMRuntimeSettings) -> None:
        self.runtime_settings = runtime_settings

    def resolve(self, capability: BuilderCapability) -> BuilderRouteDecision:
        if self.runtime_settings.active_provider == LLMProviderKey.antigravity_cli:
            return BuilderRouteDecision(
                capability=capability,
                selected_provider=LLMProviderKey.antigravity_cli,
                execution_mode=BuilderExecutionMode.primary,
                execution_backend=AgentExecutionBackend.antigravity_cli,
                fallback_provider=None,
                shadow_provider=None,
                reason="active_provider=antigravity_cli usa el runtime agy como path primario.",
            )

        if self.runtime_settings.active_provider == LLMProviderKey.codex_local:
            return BuilderRouteDecision(
                capability=capability,
                selected_provider=LLMProviderKey.codex_local,
                execution_mode=BuilderExecutionMode.primary,
                execution_backend=AgentExecutionBackend.codex_cli,
                fallback_provider=None,
                shadow_provider=None,
                reason="active_provider=codex_local usa el runtime dedicado como path primario.",
            )

        native_provider = self.runtime_settings.active_provider
        backend = self.runtime_settings.agent_execution_backend
        codex_rollout = self.runtime_settings.codex_local
        agy_rollout = self.runtime_settings.antigravity

        if backend == AgentExecutionBackend.provider_native:
            return BuilderRouteDecision(
                capability=capability,
                selected_provider=native_provider,
                execution_mode=BuilderExecutionMode.primary,
                execution_backend=AgentExecutionBackend.provider_native,
                fallback_provider=None,
                shadow_provider=None,
                reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            )

        if _matches_rollout(codex_rollout.primary_agents, capability):
            return BuilderRouteDecision(
                capability=capability,
                selected_provider=LLMProviderKey.codex_local,
                execution_mode=BuilderExecutionMode.primary,
                execution_backend=AgentExecutionBackend.codex_cli,
                fallback_provider=native_provider,
                shadow_provider=None,
                reason="La capacidad esta promovida en codex_local.primary_agents.",
            )

        if _matches_rollout(agy_rollout.primary_agents, capability):
            return BuilderRouteDecision(
                capability=capability,
                selected_provider=LLMProviderKey.antigravity_cli,
                execution_mode=BuilderExecutionMode.primary,
                execution_backend=AgentExecutionBackend.antigravity_cli,
                fallback_provider=native_provider,
                shadow_provider=None,
                reason="La capacidad esta promovida en antigravity.primary_agents.",
            )

        if _matches_rollout(codex_rollout.shadow_agents, capability):
            return BuilderRouteDecision(
                capability=capability,
                selected_provider=native_provider,
                execution_mode=BuilderExecutionMode.shadow,
                execution_backend=backend,
                fallback_provider=None,
                shadow_provider=LLMProviderKey.codex_local,
                reason="La capacidad esta marcada en codex_local.shadow_agents para corrida paralela.",
            )

        if _matches_rollout(codex_rollout.staged_agents, capability):
            return BuilderRouteDecision(
                capability=capability,
                selected_provider=native_provider,
                execution_mode=BuilderExecutionMode.staged,
                execution_backend=backend,
                fallback_provider=None,
                shadow_provider=None,
                reason="La capacidad esta marcada en codex_local.staged_agents y se mantiene en rollout controlado.",
            )

        if backend == AgentExecutionBackend.shadow_codex_cli:
            return BuilderRouteDecision(
                capability=capability,
                selected_provider=native_provider,
                execution_mode=BuilderExecutionMode.shadow,
                execution_backend=AgentExecutionBackend.shadow_codex_cli,
                fallback_provider=None,
                shadow_provider=LLMProviderKey.codex_local,
                reason="agent_execution_backend=shadow_codex_cli aplica shadow por defecto para capacidades no promovidas.",
            )

        return BuilderRouteDecision(
            capability=capability,
            selected_provider=native_provider,
            execution_mode=BuilderExecutionMode.primary,
            execution_backend=backend,
            fallback_provider=None,
            shadow_provider=None,
            reason="La capacidad no esta opt-in para Codex y sigue en el provider activo.",
        )

    def build_provider_summary(self, base_summary: dict[str, str | bool]) -> dict[str, str | bool]:
        summary = dict(base_summary)
        for capability in BuilderCapability:
            decision = self.resolve(capability)
            summary[f"{capability.value}_route"] = f"{decision.execution_mode.value}:{decision.selected_provider.value}"
        summary["execution_backend"] = (
            AgentExecutionBackend.codex_cli.value
            if self.runtime_settings.active_provider == LLMProviderKey.codex_local
            else self.runtime_settings.agent_execution_backend.value
        )
        summary["codex_shadow_enabled"] = any(
            self.resolve(capability).execution_mode == BuilderExecutionMode.shadow for capability in BuilderCapability
        )
        return summary


class BuilderProviderFacade:
    def __init__(
        self,
        runtime_settings: LLMRuntimeSettings,
        *,
        openai_service: BuilderProviderService,
        deepseek_service: BuilderProviderService,
        codex_service: BuilderProviderService,
        antigravity_service: BuilderProviderService | None = None,
    ) -> None:
        self.runtime_settings = runtime_settings
        self.router = BuilderProviderRouter(runtime_settings)
        self._openai_service = openai_service
        self._deepseek_service = deepseek_service
        self._codex_service = codex_service
        self._antigravity_service = antigravity_service

    def can_attempt(self) -> bool:
        if self.runtime_settings.active_provider == LLMProviderKey.antigravity_cli and self._antigravity_service:
            return self._antigravity_service.can_attempt()
        if self.runtime_settings.active_provider == LLMProviderKey.codex_local:
            return self._codex_service.can_attempt()
        active_service = self._native_service()
        return active_service.can_attempt() or self._codex_service.can_attempt()

    def is_available(self) -> bool:
        if self.runtime_settings.active_provider == LLMProviderKey.antigravity_cli and self._antigravity_service:
            return self._antigravity_service.is_available()
        if self.runtime_settings.active_provider == LLMProviderKey.codex_local:
            return self._codex_service.is_available()
        active_service = self._native_service()
        return active_service.is_available() or self._codex_service.is_available()

    def provider_summary(self) -> dict[str, str | bool]:
        active_service = self._service_for_provider(self.runtime_settings.active_provider)
        return self.router.build_provider_summary(active_service.provider_summary())

    def normalize_discovery(
        self,
        payload: DiscoveryInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.normalize_discovery,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.normalize_discovery(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.normalize_discovery(payload, context_bundle=route_context_bundle),
        )

    def analyze_discovery(
        self,
        payload: DiscoveryAnalysisInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.analyze_discovery,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.analyze_discovery(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.analyze_discovery(payload, context_bundle=route_context_bundle),
        )

    def build_canvas(
        self,
        discovery: DiscoveryArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.build_canvas,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.build_canvas(discovery, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.build_canvas(discovery, context_bundle=route_context_bundle),
        )

    def define_requirements(
        self,
        payload: RequirementsDefinitionInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.define_requirements,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.define_requirements(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.define_requirements(payload, context_bundle=route_context_bundle),
        )

    def synthesize_blueprint_narrative(
        self,
        discovery: DiscoveryArtifact,
        canvas: CanvasArtifact,
        blueprint: BlueprintArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.synthesize_blueprint_narrative,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.synthesize_blueprint_narrative(
                discovery,
                canvas,
                blueprint,
                context_bundle=route_context_bundle,
            ),
            codex_call=lambda service, route_context_bundle: service.synthesize_blueprint_narrative(
                discovery,
                canvas,
                blueprint,
                context_bundle=route_context_bundle,
            ),
        )

    def propose_agent_design(
        self,
        payload: AgentDesignInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.propose_agent_design,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.propose_agent_design(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.propose_agent_design(payload, context_bundle=route_context_bundle),
        )

    def critique_agent_design(
        self,
        payload: AgentDesignCritiqueInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.critique_agent_design,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.critique_agent_design(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.critique_agent_design(payload, context_bundle=route_context_bundle),
        )

    def recommend_minimal_tools(
        self,
        prompt_input: ToolRecommendationPromptInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.recommend_minimal_tools,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.recommend_minimal_tools(prompt_input, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.recommend_minimal_tools(prompt_input, context_bundle=route_context_bundle),
        )

    def recommend_memory_architecture(
        self,
        payload: MemoryArchitectureInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.recommend_memory_architecture,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.recommend_memory_architecture(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.recommend_memory_architecture(payload, context_bundle=route_context_bundle),
        )

    def critique_memory_architecture(
        self,
        payload: MemoryArchitectureCritiqueInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.critique_memory_architecture,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.critique_memory_architecture(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.critique_memory_architecture(payload, context_bundle=route_context_bundle),
        )

    def generate_validation_scenarios(
        self,
        payload: ValidationScenarioGenerationInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.generate_validation_scenarios,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.generate_validation_scenarios(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.generate_validation_scenarios(payload, context_bundle=route_context_bundle),
        )

    def simulate_validation_scenario(
        self,
        payload: ValidationScenarioSimulationInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.simulate_validation_scenario,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.simulate_validation_scenario(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.simulate_validation_scenario(payload, context_bundle=route_context_bundle),
        )

    def judge_validation_run(
        self,
        payload: ValidationRunJudgmentInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.judge_validation_run,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.judge_validation_run(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.judge_validation_run(payload, context_bundle=route_context_bundle),
        )

    def analyze_estimation_risks(
        self,
        payload: EstimationRiskAnalysisInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.analyze_estimation_risks,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.analyze_estimation_risks(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.analyze_estimation_risks(payload, context_bundle=route_context_bundle),
        )

    def generate_diagram_model(
        self,
        payload: DiagramGenerationInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_capability(
            capability=BuilderCapability.generate_diagram_model,
            context_bundle=context_bundle,
            native_call=lambda service, route_context_bundle: service.generate_diagram_model(payload, context_bundle=route_context_bundle),
            codex_call=lambda service, route_context_bundle: service.generate_diagram_model(payload, context_bundle=route_context_bundle),
        )

    def _native_service(self) -> BuilderProviderService:
        if self.runtime_settings.active_provider == LLMProviderKey.antigravity_cli and self._antigravity_service:
            return self._antigravity_service
        if self.runtime_settings.active_provider == LLMProviderKey.deepseek:
            return self._deepseek_service
        return self._openai_service

    def _service_for_provider(self, provider_key: LLMProviderKey) -> BuilderProviderService:
        if provider_key == LLMProviderKey.antigravity_cli and self._antigravity_service:
            return self._antigravity_service
        if provider_key == LLMProviderKey.codex_local:
            return self._codex_service
        if provider_key == LLMProviderKey.deepseek:
            return self._deepseek_service
        return self._openai_service

    def _annotate_result(
        self,
        result: LLMArtifactResult,
        *,
        provider_key: LLMProviderKey,
        decision: BuilderRouteDecision,
        context_bundle: StageContextBundle | None = None,
        route_reason: str | None = None,
        shadow_provider_key: LLMProviderKey | None = None,
    ) -> LLMArtifactResult:
        spec = get_builder_capability_spec(decision.capability)
        policy = {
            "llm_required": spec.llm_required,
            "critic_required": spec.critic_required,
            "timeout_ms": spec.timeout_ms,
            "max_retries": spec.max_retries,
            "fallback_policy": spec.fallback_policy,
        }
        merged_policy = dict(result.capability_policy)
        merged_policy.update(policy)
        existing_finops_context = result.finops_context
        route_metadata = {
            **(existing_finops_context.metadata if existing_finops_context is not None else {}),
            "selected_provider": decision.selected_provider.value,
            "provider_key": provider_key.value,
            "execution_backend": decision.execution_backend.value,
            "execution_mode": decision.execution_mode.value,
            "provider_route": f"{decision.execution_mode.value}:{provider_key.value}",
            "route_reason": route_reason or decision.reason,
            "fallback_provider": decision.fallback_provider.value if decision.fallback_provider is not None else "",
            "shadow_provider_key": shadow_provider_key.value if shadow_provider_key is not None else "",
            "fallback_used": result.fallback_used,
            "degraded": result.degraded,
        }
        finops_context = build_llm_call_context(
            context_bundle,
            capability=decision.capability.value,
            provider_key=provider_key.value,
            execution_backend=decision.execution_backend.value,
            execution_mode=decision.execution_mode.value,
            action_key=result.capability_key or decision.capability.value,
            operation_id=existing_finops_context.operation_id if existing_finops_context is not None else None,
            parent_run_id=existing_finops_context.parent_run_id if existing_finops_context is not None else "",
            correlation_id=existing_finops_context.correlation_id if existing_finops_context is not None else "",
            metadata=route_metadata,
        )
        return replace(
            result,
            provider_key=provider_key.value,
            execution_backend=decision.execution_backend.value,
            execution_mode=decision.execution_mode.value,
            shadow_provider_key=shadow_provider_key.value if shadow_provider_key is not None else None,
            route_reason=route_reason or decision.reason,
            capability_key=result.capability_key or decision.capability.value,
            prompt_version=result.prompt_version or spec.prompt_version,
            finops_context=finops_context,
            capability_policy=merged_policy,
        )

    def _build_route_context_bundle(
        self,
        context_bundle: StageContextBundle | None,
        *,
        decision: BuilderRouteDecision,
        provider_key: LLMProviderKey,
        operation_id: UUID,
        route_leg: str,
        fallback_used: bool = False,
        degraded: bool = False,
        shadow_provider_key: LLMProviderKey | None = None,
    ) -> StageContextBundle | None:
        if context_bundle is None:
            return None
        metadata = {
            **context_bundle.finops_metadata,
            "selected_provider": decision.selected_provider.value,
            "provider_key": provider_key.value,
            "execution_backend": decision.execution_backend.value,
            "execution_mode": decision.execution_mode.value,
            "provider_route": f"{decision.execution_mode.value}:{provider_key.value}",
            "route_leg": route_leg,
            "route_reason": decision.reason,
            "fallback_provider": decision.fallback_provider.value if decision.fallback_provider is not None else "",
            "shadow_provider_key": shadow_provider_key.value if shadow_provider_key is not None else "",
            "fallback_used": fallback_used,
            "degraded": degraded,
        }
        correlation_id = (
            context_bundle.finops_correlation_id
            or context_bundle.context_fingerprint
            or f"{context_bundle.session_id}:{decision.capability.value}:{operation_id}"
        )
        return replace(
            context_bundle,
            finops_operation_id=operation_id,
            finops_correlation_id=str(correlation_id),
            finops_execution_mode=decision.execution_mode.value,
            finops_metadata=metadata,
        )

    def _execute_capability(
        self,
        *,
        capability: BuilderCapability,
        native_call,
        codex_call,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        decision = self.router.resolve(capability)
        operation_id = uuid4()
        native_provider = self.runtime_settings.active_provider
        native_service = self._native_service()
        codex_service = self._codex_service

        if decision.selected_provider == LLMProviderKey.antigravity_cli and self._antigravity_service:
            agy_service = self._antigravity_service
            agy_context_bundle = self._build_route_context_bundle(
                context_bundle,
                decision=decision,
                provider_key=LLMProviderKey.antigravity_cli,
                operation_id=operation_id,
                route_leg="primary",
            )
            agy_result = self._annotate_result(
                codex_call(agy_service, agy_context_bundle),
                provider_key=LLMProviderKey.antigravity_cli,
                decision=decision,
                context_bundle=agy_context_bundle,
            )
            if agy_result.artifact is not None or decision.fallback_provider is None:
                return agy_result
            fallback_context_bundle = self._build_route_context_bundle(
                context_bundle,
                decision=decision,
                provider_key=decision.fallback_provider,
                operation_id=operation_id,
                route_leg="fallback",
                fallback_used=True,
                degraded=True,
            )
            fallback_result = native_call(native_service, fallback_context_bundle)
            fallback_warning = merge_warnings(
                agy_result.warning,
                (
                    f"Antigravity CLI no estuvo disponible para {capability.value}; "
                    f"se uso {_provider_label(decision.fallback_provider)} como fallback lateral."
                ),
                fallback_result.warning,
            )
            return self._annotate_result(
                replace(
                    fallback_result,
                    warning=fallback_warning,
                    fallback_used=True,
                    degraded=True,
                ),
                provider_key=decision.fallback_provider,
                decision=decision,
                context_bundle=fallback_context_bundle,
                route_reason=(
                    f"{decision.reason} Fallback lateral activado hacia "
                    f"{_provider_label(decision.fallback_provider)}."
                ),
            )

        if decision.selected_provider == LLMProviderKey.codex_local:
            codex_context_bundle = self._build_route_context_bundle(
                context_bundle,
                decision=decision,
                provider_key=LLMProviderKey.codex_local,
                operation_id=operation_id,
                route_leg="primary",
            )
            codex_result = self._annotate_result(
                codex_call(codex_service, codex_context_bundle),
                provider_key=LLMProviderKey.codex_local,
                decision=decision,
                context_bundle=codex_context_bundle,
            )
            if codex_result.artifact is not None or decision.fallback_provider is None:
                return codex_result
            fallback_context_bundle = self._build_route_context_bundle(
                context_bundle,
                decision=decision,
                provider_key=decision.fallback_provider,
                operation_id=operation_id,
                route_leg="fallback",
                fallback_used=True,
                degraded=True,
            )
            fallback_result = native_call(native_service, fallback_context_bundle)
            fallback_warning = merge_warnings(
                codex_result.warning,
                (
                    f"Codex local no estuvo disponible para {capability.value}; "
                    f"se uso {_provider_label(decision.fallback_provider)} como fallback lateral."
                ),
                fallback_result.warning,
            )
            return self._annotate_result(
                replace(
                    fallback_result,
                    warning=fallback_warning,
                    fallback_used=True,
                    degraded=True,
                ),
                provider_key=decision.fallback_provider,
                decision=decision,
                context_bundle=fallback_context_bundle,
                route_reason=(
                    f"{decision.reason} Fallback lateral activado hacia "
                    f"{_provider_label(decision.fallback_provider)}."
                ),
            )

        primary_context_bundle = self._build_route_context_bundle(
            context_bundle,
            decision=decision,
            provider_key=native_provider,
            operation_id=operation_id,
            route_leg="primary",
            shadow_provider_key=decision.shadow_provider,
        )
        primary_result = self._annotate_result(
            native_call(native_service, primary_context_bundle),
            provider_key=native_provider,
            decision=decision,
            context_bundle=primary_context_bundle,
            shadow_provider_key=decision.shadow_provider,
        )

        if decision.execution_mode != BuilderExecutionMode.shadow or decision.shadow_provider is None:
            return primary_result

        shadow_context_bundle = self._build_route_context_bundle(
            context_bundle,
            decision=decision,
            provider_key=LLMProviderKey.codex_local,
            operation_id=operation_id,
            route_leg="shadow",
            shadow_provider_key=decision.shadow_provider,
        )
        shadow_result = self._annotate_result(
            codex_call(codex_service, shadow_context_bundle),
            provider_key=LLMProviderKey.codex_local,
            decision=decision,
            context_bundle=shadow_context_bundle,
            shadow_provider_key=decision.shadow_provider,
        )
        if primary_result.artifact is not None:
            if shadow_result.artifact is not None:
                comparison = _compare_artifact_outputs(capability, primary_result, shadow_result)
                primary_result = replace(primary_result, rollout_comparison=comparison)
                if comparison["semantic_divergence"]:
                    return replace(
                        primary_result,
                        warning=merge_warnings(
                            primary_result.warning,
                            f"Shadow comparison detecto divergencias en {capability.value}; se conserva el resultado primario.",
                        ),
                    )
            elif shadow_result.warning:
                return replace(primary_result, warning=merge_warnings(primary_result.warning, shadow_result.warning))
            return primary_result

        if shadow_result.artifact is not None:
            comparison = _compare_artifact_outputs(capability, primary_result, shadow_result)
            comparison.update({"action": "shadow_promoted"})
            shadow_warning = merge_warnings(
                primary_result.warning,
                (
                    f"{_provider_label(native_provider)} no estuvo disponible para {capability.value}; "
                    "se promovio Codex local desde shadow."
                ),
            )
            return self._annotate_result(
                replace(
                    shadow_result,
                    warning=shadow_warning,
                    fallback_used=True,
                    degraded=True,
                    rollout_comparison=comparison,
                ),
                provider_key=LLMProviderKey.codex_local,
                decision=decision,
                context_bundle=shadow_context_bundle,
                route_reason=f"{decision.reason} Shadow promovido por indisponibilidad del provider activo.",
            )
        if shadow_result.warning:
            return replace(primary_result, warning=merge_warnings(primary_result.warning, shadow_result.warning))
        return primary_result
