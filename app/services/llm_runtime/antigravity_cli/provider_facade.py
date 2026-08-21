from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from app.models import (
    AgentExecutionBackend,
    BlueprintArtifact,
    CanvasArtifact,
    DiscoveryArtifact,
    DiscoveryInput,
    LLMProviderKey,
    LLMRuntimeSettings,
    ToolRecommendationLLMOutput,
    ToolRecommendationPromptInput,
)
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_finops.provider_instrumentation import FinOpsSessionFactory, record_provider_result
from app.services.llm_finops.usage_normalization import normalize_cli_usage
from app.services.agent_i18n import apply_agent_language_directive, get_effective_language
from app.services.diagram_center.contracts import DiagramGenerationInput
from app.services.llm_runtime.antigravity_cli.execution_service import AgyExecutionService
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
    sanitize_canvas,
    sanitize_discovery,
)
from app.services.llm_runtime.capability_registry import (
    BuilderCapability,
    BuilderCapabilitySpec,
    get_builder_capability_spec,
)
from app.services.llm_runtime.stage_context_types import StageContextBundle, build_llm_call_context
from app.services.rules import normalize_text


def _capability_policy_payload(spec: BuilderCapabilitySpec) -> dict[str, object]:
    return {
        "llm_required": spec.llm_required,
        "critic_required": spec.critic_required,
        "timeout_ms": spec.timeout_ms,
        "max_retries": spec.max_retries,
        "fallback_policy": spec.fallback_policy,
    }


def _localized_prompt(prompt: str, context_bundle: StageContextBundle | None) -> str:
    language = get_effective_language(context_bundle.effective_language if context_bundle is not None else None)
    return apply_agent_language_directive(prompt, language)


class AntigravityLocalBuilderService:
    """
    Fachada del proveedor Antigravity CLI para el sistema de builder de agentes.

    Implementa el mismo Protocol BuilderProviderService que CodexLocalBuilderService:
      - can_attempt() / is_available()
      - provider_summary()
      - Una implementacion para cada BuilderCapability registrado

    La ejecucion real se delega a AgyExecutionService, que maneja el spawn
    del proceso agy, el loop de fallback de modelos y la auditoria.

    Diferencia justificada respecto a CodexLocalBuilderService:
    - No usa CodexContextAssembler / CodexContextRequest porque agy recibe el prompt
      directamente por stdin y resuelve su propio contexto de workspace.
    - El prompt se construye de forma inline (igual al modo no-staged de Codex).
    """

    def __init__(
        self,
        runtime_settings: LLMRuntimeSettings,
        *,
        finops_session_factory: FinOpsSessionFactory | None = None,
        finops_ledger_service: LLMUsageLedgerService | None = None,
    ) -> None:
        self.runtime_settings = runtime_settings
        self.execution_service = AgyExecutionService(runtime_settings)
        self._finops_session_factory = finops_session_factory
        self._finops_ledger_service = finops_ledger_service

    # ------------------------------------------------------------------
    # BuilderProviderService Protocol
    # ------------------------------------------------------------------

    def can_attempt(self) -> bool:
        return self.runtime_settings.active_provider == LLMProviderKey.antigravity_cli

    def is_available(self) -> bool:
        return (
            self.can_attempt()
            and self.runtime_settings.antigravity.executable_found is not False
            and bool(self.execution_service.resolve_executable())
        )

    def provider_summary(self) -> dict[str, str | bool]:
        cfg = self.runtime_settings.antigravity
        executable_found = bool(self.execution_service.resolve_executable())
        return {
            "provider": self.runtime_settings.active_provider.value,
            "mode": "local_exec",
            "configured": bool(cfg.executable and cfg.model),
            "sdk_ready": executable_found,
            "fast_model": cfg.model,
            "reasoning_model": cfg.model,
            "executable": cfg.executable,
            "effort": cfg.effort,
            "status_note": cfg.status_note or ("Binario agy detectado." if executable_found else "No se encontro el binario agy."),
        }

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _base_result(self, capability: BuilderCapability, spec: BuilderCapabilitySpec) -> LLMArtifactResult:
        return LLMArtifactResult(
            artifact=None,
            provider_key=LLMProviderKey.antigravity_cli.value,
            capability_key=capability.value,
            model_name=self.runtime_settings.antigravity.model,
            prompt_version=spec.prompt_version,
            schema_validation_status="not_attempted",
            capability_policy=_capability_policy_payload(spec),
        )

    def _unavailable_result(
        self, capability: BuilderCapability, spec: BuilderCapabilitySpec
    ) -> LLMArtifactResult:
        return replace(
            self._base_result(capability, spec),
            warning=f"Antigravity CLI no esta disponible para {capability.value}; policy={spec.fallback_policy}.",
            finish_reason="provider_unavailable",
            failure_kind="provider_unavailable",
            degraded=True,
        )

    def _attach_finops_record(
        self,
        result: LLMArtifactResult,
        *,
        capability: BuilderCapability,
        context_bundle: StageContextBundle | None,
        audit: dict[str, Any],
        prompt_text: str = "",
        output_text: str = "",
    ) -> LLMArtifactResult:
        cfg = self.runtime_settings.antigravity
        model_name = str(audit.get("selected_model", cfg.model) or cfg.model)
        requested_model = str(audit.get("requested_model", cfg.model) or cfg.model)
        call_context = build_llm_call_context(
            context_bundle,
            capability=capability.value,
            provider_key=LLMProviderKey.antigravity_cli.value,
            execution_backend=AgentExecutionBackend.antigravity_cli.value,
            metadata={
                "runner_id": cfg.runner_id,
                "effort": cfg.effort,
                "runtime": "antigravity_cli",
            },
        )
        usage = normalize_cli_usage(audit, prompt_text=prompt_text, output_text=output_text)
        metrics = audit.get("metrics", {}) if isinstance(audit.get("metrics"), dict) else {}
        duration_ms = int(metrics.get("duration_ms", 0) or 0)
        queue_wait_ms = int(metrics.get("queue_wait_ms", 0) or 0)
        attempts = audit.get("attempts", [])
        retry_count = max(0, len(attempts) - 1) if isinstance(attempts, list) else 0
        enriched = replace(
            result,
            provider_key=LLMProviderKey.antigravity_cli.value,
            execution_backend=AgentExecutionBackend.antigravity_cli.value,
            execution_mode=call_context.execution_mode,
            capability_key=result.capability_key or capability.value,
            request_id=str(audit.get("run_id", "") or result.request_id or ""),
            finish_reason=str(audit.get("status", result.finish_reason or "") or ""),
            model_name=model_name,
            retry_count=retry_count,
            fallback_used=bool(audit.get("fallback_used", result.fallback_used)),
            duration_ms=duration_ms,
            queue_wait_ms=queue_wait_ms,
            token_usage=usage.compatibility_token_usage(),
            normalized_usage=usage,
            finops_context=call_context,
        )
        return record_provider_result(
            enriched,
            call_context=call_context,
            provider_key=LLMProviderKey.antigravity_cli,
            model_name=model_name,
            requested_model=requested_model,
            execution_backend=AgentExecutionBackend.antigravity_cli.value,
            execution_mode=call_context.execution_mode,
            started_at=audit.get("started_at"),
            finished_at=audit.get("finished_at"),
            duration_ms=duration_ms,
            ledger_service=self._finops_ledger_service,
            session_factory=self._finops_session_factory,
            metadata={
                "attempted_models": audit.get("attempted_models", []),
                "fallback_used": bool(audit.get("fallback_used", False)),
                "runtime": "antigravity_cli",
            },
        )

    def _execute_structured_capability(
        self,
        *,
        capability: BuilderCapability,
        payload,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        """
        Ejecuta una capacidad estructurada contra agy.

        Construye el prompt de forma inline (sin CodexContextAssembler) porque agy
        resuelve su propio contexto de workspace desde el directorio --dir.
        """
        spec = get_builder_capability_spec(capability)
        base_result = self._base_result(capability, spec)

        if not self.is_available():
            return self._attach_finops_record(
                self._unavailable_result(capability, spec),
                capability=capability,
                context_bundle=context_bundle,
                audit={},
            )

        schema_json = json.dumps(spec.output_model.model_json_schema(), ensure_ascii=True)
        prompt = _localized_prompt(
            "Devuelve exclusivamente un JSON valido que cumpla con el siguiente schema JSON:\n"
            f"{schema_json}\n\n"
            f"{spec.task_instruction}\n\n"
            f"INPUT:\n{json.dumps(payload.model_dump(mode='json'), ensure_ascii=True)}",
            context_bundle,
        )

        try:
            parsed = self.execution_service.execute_structured_prompt(
                task_kind=spec.task_kind,
                prompt=prompt,
                output_model=spec.output_model,
                timeout_ms=spec.timeout_ms,
            )
            audit = self.execution_service.read_last_known_result() or {}
            result = replace(
                base_result,
                artifact=spec.output_model.model_validate(parsed.model_dump(mode="json")),
                request_id=str(audit.get("run_id", "") or ""),
                finish_reason=str(audit.get("status", "succeeded") or "succeeded"),
                model_name=str(
                    audit.get("selected_model", self.runtime_settings.antigravity.model)
                    or self.runtime_settings.antigravity.model
                ),
                schema_validation_status="valid",
            )
            return self._attach_finops_record(
                result,
                capability=capability,
                context_bundle=context_bundle,
                audit=audit,
                prompt_text=prompt,
                output_text=json.dumps(parsed.model_dump(mode="json"), ensure_ascii=True),
            )
        except Exception as exc:
            audit = self.execution_service.read_last_known_result() or {}
            result = replace(
                base_result,
                warning=f"Antigravity CLI no pudo ejecutar {capability.value}; policy={spec.fallback_policy}.",
                request_id=str(audit.get("run_id", "") or ""),
                finish_reason=str(audit.get("status", "failed") or "failed"),
                model_name=str(
                    audit.get("selected_model", self.runtime_settings.antigravity.model)
                    or self.runtime_settings.antigravity.model
                ),
                schema_validation_status="invalid",
                failure_kind="provider_error",
                failure_detail=str(exc)[:400],
                degraded=True,
            )
            return self._attach_finops_record(
                result,
                capability=capability,
                context_bundle=context_bundle,
                audit=audit,
                prompt_text=prompt,
            )

    # ------------------------------------------------------------------
    # Implementaciones de cada capacidad del builder
    # ------------------------------------------------------------------

    def normalize_discovery(
        self,
        payload: DiscoveryInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        if not self.is_available():
            return LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.antigravity_cli.value)

        schema_json = json.dumps(DiscoveryArtifact.model_json_schema(), ensure_ascii=True)
        prompt = _localized_prompt(
            "Devuelve exclusivamente un JSON valido que cumpla con el siguiente schema JSON:\n"
            f"{schema_json}\n\n"
            "Normaliza esta captura a un discovery estructurado para un builder Lean de agentes. "
            "Usa solo hechos presentes en la entrada. Si un dato no esta claro, usa 'unknown'. "
            "Para case_type usa solo: informacion, automatizacion, copiloto, operador_autonomo, sistema_multiagente. "
            "Para autonomy_level usa solo: low, medium, high.\n\n"
            f"INPUT:\n{json.dumps(payload.model_dump(mode='json'), ensure_ascii=True)}",
            context_bundle,
        )
        try:
            parsed = self.execution_service.execute_structured_prompt(
                task_kind="discovery_normalization",
                prompt=prompt,
                output_model=DiscoveryArtifact,
            )
            normalized = DiscoveryArtifact.model_validate(parsed.model_dump(mode="json"))
            return LLMArtifactResult(
                artifact=sanitize_discovery(normalized),
                provider_key=LLMProviderKey.antigravity_cli.value,
            )
        except Exception:
            return LLMArtifactResult(
                artifact=None,
                warning="Antigravity CLI no pudo normalizar discovery; se uso fallback deterministico.",
                provider_key=LLMProviderKey.antigravity_cli.value,
            )

    def build_canvas(
        self,
        discovery: DiscoveryArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        if not self.is_available():
            return LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.antigravity_cli.value)

        schema_json = json.dumps(CanvasArtifact.model_json_schema(), ensure_ascii=True)
        prompt = _localized_prompt(
            "Devuelve exclusivamente un JSON valido que cumpla con el siguiente schema JSON:\n"
            f"{schema_json}\n\n"
            "Genera un canvas Lean para un agente usando solo el discovery recibido. "
            "Manten el alcance corto, concreto y util para un MVP. Si algo no esta claro, usa 'unknown'.\n\n"
            f"DISCOVERY:\n{json.dumps(discovery.model_dump(mode='json'), ensure_ascii=True)}",
            context_bundle,
        )
        try:
            parsed = self.execution_service.execute_structured_prompt(
                task_kind="canvas_generation",
                prompt=prompt,
                output_model=CanvasArtifact,
            )
            normalized = CanvasArtifact.model_validate(parsed.model_dump(mode="json"))
            return LLMArtifactResult(
                artifact=sanitize_canvas(normalized),
                provider_key=LLMProviderKey.antigravity_cli.value,
            )
        except Exception:
            return LLMArtifactResult(
                artifact=None,
                warning="Antigravity CLI no pudo construir el canvas; se uso fallback deterministico.",
                provider_key=LLMProviderKey.antigravity_cli.value,
            )

    def synthesize_blueprint_narrative(
        self,
        discovery: DiscoveryArtifact,
        canvas: CanvasArtifact,
        blueprint: BlueprintArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        if not self.is_available():
            return LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.antigravity_cli.value)

        schema_json = json.dumps(BlueprintNarrativeOutput.model_json_schema(), ensure_ascii=True)
        prompt = _localized_prompt(
            "Devuelve exclusivamente un JSON valido que cumpla con el siguiente schema JSON:\n"
            f"{schema_json}\n\n"
            "Redacta la narrativa tecnica de un blueprint Lean para un agente. "
            "No cambies la arquitectura, memoria, tools ni guardrails ya definidos. "
            "Explica por que la recomendacion encaja con el discovery y el canvas, "
            "y resalta tradeoffs relevantes sin inventar nuevos componentes.\n\n"
            f"DISCOVERY:\n{json.dumps(discovery.model_dump(mode='json'), ensure_ascii=True)}\n\n"
            f"CANVAS:\n{json.dumps(canvas.model_dump(mode='json'), ensure_ascii=True)}\n\n"
            f"BLUEPRINT:\n{json.dumps(blueprint.model_dump(mode='json'), ensure_ascii=True)}",
            context_bundle,
        )
        try:
            parsed = self.execution_service.execute_structured_prompt(
                task_kind="blueprint_narrative",
                prompt=prompt,
                output_model=BlueprintNarrativeOutput,
            )
            normalized = BlueprintNarrativeOutput.model_validate(parsed.model_dump(mode="json"))
            return LLMArtifactResult(
                artifact=BlueprintNarrativeOutput(narrative=normalize_text(normalized.narrative)),
                provider_key=LLMProviderKey.antigravity_cli.value,
            )
        except Exception:
            return LLMArtifactResult(
                artifact=None,
                warning="Antigravity CLI no pudo sintetizar la narrativa; se mantuvo la narrativa base.",
                provider_key=LLMProviderKey.antigravity_cli.value,
            )

    def recommend_minimal_tools(
        self,
        prompt_input: ToolRecommendationPromptInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        if not self.is_available():
            return LLMArtifactResult(
                artifact=None,
                warning="Antigravity CLI no esta disponible para recomendar tools minimas; se mantiene el preflight heuristico.",
                provider_key=LLMProviderKey.antigravity_cli.value,
            )

        case_payload = prompt_input.model_dump(
            mode="json",
            exclude={"candidate_tools", "mandatory_tool_keys", "forbidden_tool_keys"},
        )
        catalog_payload = {
            "mandatory_tool_keys": [item.value for item in prompt_input.mandatory_tool_keys],
            "forbidden_tool_keys": [item.value for item in prompt_input.forbidden_tool_keys],
            "candidate_tools": [item.model_dump(mode="json") for item in prompt_input.candidate_tools],
        }
        schema_json = json.dumps(ToolRecommendationLLMOutput.model_json_schema(), ensure_ascii=True)
        prompt = _localized_prompt(
            "Devuelve exclusivamente un JSON valido que cumpla con el siguiente schema JSON:\n"
            f"{schema_json}\n\n"
            "Selecciona el conjunto minimo de herramientas para un agente Lean usando solo el contexto aprobado "
            "y el catalogo permitido. Nunca inventes tool keys fuera del catalogo. "
            "Manten toda tool mandatory si la evidencia la sostiene. "
            "Marca como unnecessary cualquier tool candidata que no aporte capacidad unica. "
            "Si falta informacion, devuelve gaps estructurados en lugar de inventar tools.\n\n"
            f"CASE:\n{json.dumps(case_payload, ensure_ascii=True)}\n\n"
            f"CATALOG:\n{json.dumps(catalog_payload, ensure_ascii=True)}",
            context_bundle,
        )
        try:
            parsed = self.execution_service.execute_structured_prompt(
                task_kind="tool_recommendation_minimal",
                prompt=prompt,
                output_model=ToolRecommendationLLMOutput,
            )
            normalized = ToolRecommendationLLMOutput.model_validate(parsed.model_dump(mode="json"))
            return LLMArtifactResult(
                artifact=normalized,
                provider_key=LLMProviderKey.antigravity_cli.value,
            )
        except Exception:
            return LLMArtifactResult(
                artifact=None,
                warning="Antigravity CLI no pudo recomendar tools minimas; se mantuvo el preflight heuristico.",
                provider_key=LLMProviderKey.antigravity_cli.value,
            )

    def analyze_discovery(
        self,
        payload: DiscoveryAnalysisInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.analyze_discovery,
            payload=payload,
            context_bundle=context_bundle,
        )

    def define_requirements(
        self,
        payload: RequirementsDefinitionInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.define_requirements,
            payload=payload,
            context_bundle=context_bundle,
        )

    def propose_agent_design(
        self,
        payload: AgentDesignInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.propose_agent_design,
            payload=payload,
            context_bundle=context_bundle,
        )

    def critique_agent_design(
        self,
        payload: AgentDesignCritiqueInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.critique_agent_design,
            payload=payload,
            context_bundle=context_bundle,
        )

    def recommend_memory_architecture(
        self,
        payload: MemoryArchitectureInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.recommend_memory_architecture,
            payload=payload,
            context_bundle=context_bundle,
        )

    def critique_memory_architecture(
        self,
        payload: MemoryArchitectureCritiqueInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.critique_memory_architecture,
            payload=payload,
            context_bundle=context_bundle,
        )

    def generate_validation_scenarios(
        self,
        payload: ValidationScenarioGenerationInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.generate_validation_scenarios,
            payload=payload,
            context_bundle=context_bundle,
        )

    def simulate_validation_scenario(
        self,
        payload: ValidationScenarioSimulationInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.simulate_validation_scenario,
            payload=payload,
            context_bundle=context_bundle,
        )

    def judge_validation_run(
        self,
        payload: ValidationRunJudgmentInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.judge_validation_run,
            payload=payload,
            context_bundle=context_bundle,
        )

    def analyze_estimation_risks(
        self,
        payload: EstimationRiskAnalysisInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.analyze_estimation_risks,
            payload=payload,
            context_bundle=context_bundle,
        )

    def generate_diagram_model(
        self,
        payload: DiagramGenerationInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        return self._execute_structured_capability(
            capability=BuilderCapability.generate_diagram_model,
            payload=payload,
            context_bundle=context_bundle,
        )
