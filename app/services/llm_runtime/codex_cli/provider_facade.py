from __future__ import annotations

from dataclasses import dataclass, replace
import json
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
from app.services.diagram_center.semantic_repair import finalize_structured_diagram_artifact
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
from app.services.llm_runtime.capability_registry import BuilderCapability, BuilderCapabilitySpec, get_builder_capability_spec
from app.services.llm_runtime.codex_cli.context_assembler import (
    CodexContextAssembler,
    CodexContextInlineSource,
    CodexContextRequest,
)
from app.services.llm_runtime.codex_cli.execution_service import CodexExecutionService
from app.services.llm_runtime.prompt_templates import (
    build_tool_recommendation_inline_prompt,
    build_tool_recommendation_staged_prompt,
)
from app.services.llm_runtime.stage_context_types import StageContextBundle, build_llm_call_context
from app.services.rules import normalize_text


@dataclass(frozen=True)
class _CodexContextEnvelope:
    context_request: CodexContextRequest | None
    knowledge_access_backend: str
    effective_context_backend: str
    used_sources: list[dict[str, object]]
    context_stats: dict[str, object]


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


class CodexLocalBuilderService:
    def __init__(
        self,
        runtime_settings: LLMRuntimeSettings,
        *,
        finops_session_factory: FinOpsSessionFactory | None = None,
        finops_ledger_service: LLMUsageLedgerService | None = None,
    ) -> None:
        self.runtime_settings = runtime_settings
        self.execution_service = CodexExecutionService(runtime_settings)
        self._context_assembler = CodexContextAssembler()
        self._finops_session_factory = finops_session_factory
        self._finops_ledger_service = finops_ledger_service

    def can_attempt(self) -> bool:
        return self.runtime_settings.active_provider == LLMProviderKey.codex_local

    def is_available(self) -> bool:
        return self.can_attempt() and self.runtime_settings.codex_local.executable_found

    def provider_summary(self) -> dict[str, str | bool]:
        return {
            "provider": self.runtime_settings.active_provider.value,
            "mode": "local_exec",
            "configured": bool(self.runtime_settings.codex_local.command and self.runtime_settings.codex_local.model),
            "sdk_ready": self.runtime_settings.codex_local.executable_found,
            "fast_model": self.runtime_settings.codex_local.model,
            "reasoning_model": self.runtime_settings.codex_local.model,
            "command": self.runtime_settings.codex_local.command,
            "profile": self.runtime_settings.codex_local.profile,
            "status_note": self.runtime_settings.codex_local.status_note,
        }

    def _uses_staged_context(self) -> bool:
        return self.runtime_settings.knowledge_access_backend.value in {"workspace_staged", "hybrid"}

    def _context_request(
        self,
        *,
        role: str,
        sources: list[CodexContextInlineSource],
        context_bundle: StageContextBundle | None = None,
    ) -> CodexContextRequest:
        return CodexContextRequest(
            role=role,
            knowledge_access_backend=self.runtime_settings.knowledge_access_backend.value,
            inline_sources=sources,
            workspace_id=context_bundle.workspace_id if context_bundle is not None else None,
            session_id=context_bundle.session_id if context_bundle is not None else None,
            session_snapshot=context_bundle.session_snapshot if context_bundle is not None else None,
            knowledge_manifest=context_bundle.knowledge_manifest if context_bundle is not None else None,
            memory_policy=context_bundle.memory_policy if context_bundle is not None else None,
            short_term_memory=context_bundle.short_term_memory if context_bundle is not None else None,
            approved_refs=list(context_bundle.approved_refs) if context_bundle is not None else [],
            retrieved_hits=list(context_bundle.retrieved_hits) if context_bundle is not None else [],
            strict_budget=context_bundle.strict_budget if context_bundle is not None else None,
            stage_hint=context_bundle.stage if context_bundle is not None else "",
            context_fingerprint=context_bundle.context_fingerprint if context_bundle is not None else "",
            corpus_hash=context_bundle.corpus_hash if context_bundle is not None else "",
            retrieval_pages=context_bundle.retrieval_pages if context_bundle is not None else 0,
            absence_reason=context_bundle.absence_reason if context_bundle is not None else "",
        )

    def _effective_context_backend(self, knowledge_access_backend: str) -> str:
        if knowledge_access_backend == "workspace_staged":
            return "workspace_staged_filesystem"
        if knowledge_access_backend == "hybrid":
            return "hybrid_workspace_staged"
        return "inline_context_raw"

    def _build_context_envelope(
        self,
        *,
        task_kind: str,
        role: str,
        sources: list[CodexContextInlineSource],
        context_bundle: StageContextBundle | None = None,
    ) -> _CodexContextEnvelope:
        knowledge_access_backend = self.runtime_settings.knowledge_access_backend.value
        effective_context_backend = self._effective_context_backend(knowledge_access_backend)
        if knowledge_access_backend == "inline_context" and context_bundle is not None:
            effective_context_backend = "inline_context_compact"
        if self._uses_staged_context() or context_bundle is not None:
            context_request = self._context_request(role=role, sources=sources, context_bundle=context_bundle)
            assembly = self._context_assembler.assemble(task_kind=task_kind, request=context_request)
            metadata = assembly.metadata_payload()
            return _CodexContextEnvelope(
                context_request=context_request,
                knowledge_access_backend=knowledge_access_backend,
                effective_context_backend=effective_context_backend,
                used_sources=list(metadata.get("used_sources", [])),
                context_stats=dict(metadata.get("context_stats", {})),
            )

        used_sources: list[dict[str, object]] = []
        baseline_estimated_tokens = 0
        for item in sources:
            char_count = len(item.content)
            token_estimate = max(1, (char_count + 3) // 4) if char_count else 0
            baseline_estimated_tokens += token_estimate
            used_sources.append(
                {
                    "key": item.key,
                    "title": item.title,
                    "source_type": item.source_type,
                    "uri": item.uri,
                    "authority_level": item.authority_level,
                    "required": item.required,
                    "summary": item.summary,
                    "relative_path": "",
                    "baseline_chars": char_count,
                    "assembled_chars": char_count,
                    "token_estimate": token_estimate,
                    "truncated": False,
                    "source_refs": [],
                    "stage_affinity": list(item.stage_affinity),
                    "agent_affinity": list(item.agent_affinity),
                    "delivery_mode": "inline_raw",
                }
            )

        context_stats: dict[str, object] = {
            "role": role,
            "budget_tokens": baseline_estimated_tokens,
            "budget_chars": sum(len(item.content) for item in sources),
            "max_items": len(sources),
            "baseline_estimated_tokens": baseline_estimated_tokens,
            "assembled_estimated_tokens": baseline_estimated_tokens,
            "reduction_estimated_tokens": 0,
            "used_full_documents": False,
            "truncated_source_count": 0,
            "required_source_count": sum(1 for item in sources if item.required),
            "candidate_source_count": sum(1 for item in sources if not item.required),
            "staged_workspace_used": False,
        }
        return _CodexContextEnvelope(
            context_request=None,
            knowledge_access_backend=knowledge_access_backend,
            effective_context_backend=effective_context_backend,
            used_sources=used_sources,
            context_stats=context_stats,
        )

    def _attach_context_metadata(
        self,
        result: LLMArtifactResult,
        *,
        context_envelope: _CodexContextEnvelope,
    ) -> LLMArtifactResult:
        result.knowledge_access_backend = context_envelope.knowledge_access_backend
        result.effective_context_backend = context_envelope.effective_context_backend
        result.context_used_sources = list(context_envelope.used_sources)
        result.context_stats = dict(context_envelope.context_stats)
        return result

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
        model_name = str(
            audit.get("selected_model", self.runtime_settings.codex_local.model)
            or self.runtime_settings.codex_local.model
        )
        requested_model = str(
            audit.get("requested_model", self.runtime_settings.codex_local.model)
            or self.runtime_settings.codex_local.model
        )
        call_context = build_llm_call_context(
            context_bundle,
            capability=capability.value,
            provider_key=LLMProviderKey.codex_local.value,
            execution_backend=AgentExecutionBackend.codex_cli.value,
            metadata={
                "runner_id": self.runtime_settings.codex_local.runner_id,
                "profile": self.runtime_settings.codex_local.profile,
                "cost_policy": self.runtime_settings.codex_local.cost_policy.value,
                "runtime": "codex_cli",
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
            provider_key=LLMProviderKey.codex_local.value,
            execution_backend=AgentExecutionBackend.codex_cli.value,
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
            provider_key=LLMProviderKey.codex_local,
            model_name=model_name,
            requested_model=requested_model,
            execution_backend=AgentExecutionBackend.codex_cli.value,
            execution_mode=call_context.execution_mode,
            started_at=audit.get("started_at"),
            finished_at=audit.get("finished_at"),
            duration_ms=duration_ms,
            ledger_service=self._finops_ledger_service,
            session_factory=self._finops_session_factory,
            metadata={
                "attempted_models": audit.get("attempted_models", []),
                "fallback_used": bool(audit.get("fallback_used", False)),
                "runtime": "codex_cli",
            },
        )

    def _capability_source(self, spec: BuilderCapabilitySpec, payload) -> CodexContextInlineSource:
        raw_json = json.dumps(payload.model_dump(mode="json"), ensure_ascii=True, default=str)
        return CodexContextInlineSource(
            key=spec.source_key,
            title=spec.source_title,
            content=json.dumps(payload.model_dump(mode="json"), ensure_ascii=True, indent=2, default=str),
            required=True,
            summary=spec.source_summary,
            metadata={
                "context_quality_version": "context-quality.v1",
                "input_payload_chars": len(raw_json),
                "compact_payload_chars": len(raw_json),
                "compact_payload_tokens_est": max(1, (len(raw_json) + 3) // 4),
                "compact_retention_pct": 100.0,
                "payload_model": payload.__class__.__name__,
                "source_key": spec.source_key,
                "api_compaction_applied": False,
            },
        )

    def _execute_structured_capability(
        self,
        *,
        capability: BuilderCapability,
        payload,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        spec = get_builder_capability_spec(capability)
        context_envelope = self._build_context_envelope(
            task_kind=spec.task_kind,
            role="builder",
            sources=[self._capability_source(spec, payload)],
            context_bundle=context_bundle,
        )
        base_result = LLMArtifactResult(
            artifact=None,
            provider_key=LLMProviderKey.codex_local.value,
            capability_key=capability.value,
            model_name=self.runtime_settings.codex_local.model,
            prompt_version=spec.prompt_version,
            schema_validation_status="not_attempted",
            capability_policy=_capability_policy_payload(spec),
        )
        if not self.is_available():
            result = self._attach_context_metadata(
                replace(
                    base_result,
                    warning=f"Codex local no esta disponible para {capability.value}; policy={spec.fallback_policy}.",
                    finish_reason="provider_unavailable",
                    failure_kind="provider_unavailable",
                    degraded=True,
                ),
                context_envelope=context_envelope,
            )
            return self._attach_finops_record(
                result,
                capability=capability,
                context_bundle=context_bundle,
                audit={},
            )
        prompt = _localized_prompt(
            "Devuelve exclusivamente JSON valido segun el schema provisto. "
            f"{spec.task_instruction}",
            context_bundle,
        )
        try:
            parsed = self.execution_service.execute_structured_prompt(
                task_kind=spec.task_kind,
                prompt=prompt,
                output_model=spec.output_model,
                timeout_ms=spec.timeout_ms,
                context_request=context_envelope.context_request,
            )
            audit = self.execution_service.read_last_known_result() or {}
            normalized = spec.output_model.model_validate(parsed.model_dump(mode="json"))
            schema_status = "valid"
            if capability == BuilderCapability.generate_diagram_model:
                normalized, schema_status = finalize_structured_diagram_artifact(
                    normalized,
                    schema_status=schema_status,
                )
            result = self._attach_context_metadata(
                replace(
                    base_result,
                    artifact=normalized,
                    request_id=str(audit.get("run_id", "") or ""),
                    finish_reason=str(audit.get("status", "succeeded") or "succeeded"),
                    model_name=str(audit.get("selected_model", self.runtime_settings.codex_local.model) or self.runtime_settings.codex_local.model),
                    schema_validation_status=schema_status,
                ),
                context_envelope=context_envelope,
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
            result = self._attach_context_metadata(
                replace(
                    base_result,
                    warning=f"Codex local no pudo ejecutar {capability.value}; policy={spec.fallback_policy}.",
                    request_id=str(audit.get("run_id", "") or ""),
                    finish_reason=str(audit.get("status", "failed") or "failed"),
                    model_name=str(audit.get("selected_model", self.runtime_settings.codex_local.model) or self.runtime_settings.codex_local.model),
                    schema_validation_status="invalid",
                    failure_kind="provider_error",
                    failure_detail=str(exc)[:400],
                    degraded=True,
                ),
                context_envelope=context_envelope,
            )
            return self._attach_finops_record(
                result,
                capability=capability,
                context_bundle=context_bundle,
                audit=audit,
                prompt_text=prompt,
            )

    def normalize_discovery(
        self,
        payload: DiscoveryInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        if not self.is_available():
            return LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.codex_local.value)

        context_sources = [
            CodexContextInlineSource(
                key="discovery_capture",
                title="Discovery capture",
                content=json.dumps(payload.model_dump(mode="json"), ensure_ascii=True, indent=2),
                required=True,
                summary="Captura cruda de discovery para normalizacion estructurada.",
            )
        ]
        context_envelope = self._build_context_envelope(
            task_kind="discovery_normalization",
            role="builder",
            sources=context_sources,
            context_bundle=context_bundle,
        )
        if self._uses_staged_context():
            prompt = _localized_prompt(
                "Devuelve exclusivamente JSON valido segun el schema provisto. "
                "Normaliza la captura staged `discovery_capture` a un discovery estructurado para un builder Lean de agentes. "
                "Usa solo hechos presentes en la entrada staged. Si un dato no esta claro, usa 'unknown'. "
                "Para case_type usa solo: informacion, automatizacion, copiloto, operador_autonomo, sistema_multiagente. "
                "Para autonomy_level usa solo: low, medium, high.",
                context_bundle,
            )
        else:
            prompt = _localized_prompt(
                "Devuelve exclusivamente JSON valido segun el schema provisto. "
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
                context_request=context_envelope.context_request,
            )
            normalized = DiscoveryArtifact.model_validate(parsed.model_dump(mode="json"))
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=sanitize_discovery(normalized),
                    provider_key=LLMProviderKey.codex_local.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="Codex local no pudo normalizar discovery; se uso fallback deterministico.",
                    provider_key=LLMProviderKey.codex_local.value,
                ),
                context_envelope=context_envelope,
            )

    def build_canvas(
        self,
        discovery: DiscoveryArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        if not self.is_available():
            return LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.codex_local.value)

        context_sources = [
            CodexContextInlineSource(
                key="normalized_discovery",
                title="Normalized discovery",
                content=json.dumps(discovery.model_dump(mode="json"), ensure_ascii=True, indent=2),
                required=True,
                summary="Discovery estructurado aprobado para construir el canvas.",
            )
        ]
        context_envelope = self._build_context_envelope(
            task_kind="canvas_generation",
            role="builder",
            sources=context_sources,
            context_bundle=context_bundle,
        )
        if self._uses_staged_context():
            prompt = _localized_prompt(
                "Devuelve exclusivamente JSON valido segun el schema provisto. "
                "Genera un canvas Lean para un agente usando solo el discovery staged `normalized_discovery`. "
                "Manten el alcance corto, concreto y util para un MVP. Si algo no esta claro, usa 'unknown'.",
                context_bundle,
            )
        else:
            prompt = _localized_prompt(
                "Devuelve exclusivamente JSON valido segun el schema provisto. "
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
                context_request=context_envelope.context_request,
            )
            normalized = CanvasArtifact.model_validate(parsed.model_dump(mode="json"))
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=sanitize_canvas(normalized),
                    provider_key=LLMProviderKey.codex_local.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="Codex local no pudo construir el canvas; se uso fallback deterministico.",
                    provider_key=LLMProviderKey.codex_local.value,
                ),
                context_envelope=context_envelope,
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
            return LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.codex_local.value)

        context_sources = [
            CodexContextInlineSource(
                key="narrative_discovery",
                title="Discovery for blueprint narrative",
                content=json.dumps(discovery.model_dump(mode="json"), ensure_ascii=True, indent=2),
                required=True,
                summary="Discovery estructurado ya aprobado para la narrativa.",
            ),
            CodexContextInlineSource(
                key="narrative_canvas",
                title="Canvas for blueprint narrative",
                content=json.dumps(canvas.model_dump(mode="json"), ensure_ascii=True, indent=2),
                required=True,
                summary="Canvas Lean aprobado que define alcance, meta y riesgo principal.",
            ),
            CodexContextInlineSource(
                key="narrative_blueprint",
                title="Blueprint for narrative synthesis",
                content=json.dumps(blueprint.model_dump(mode="json"), ensure_ascii=True, indent=2),
                required=True,
                summary="Blueprint estructural base cuya narrativa debe sintetizarse sin cambiar contratos.",
            ),
        ]
        context_envelope = self._build_context_envelope(
            task_kind="blueprint_narrative",
            role="builder",
            sources=context_sources,
            context_bundle=context_bundle,
        )
        if self._uses_staged_context():
            prompt = _localized_prompt(
                "Devuelve exclusivamente JSON valido segun el schema provisto. "
                "Redacta la narrativa tecnica de un blueprint Lean para un agente usando solo las fuentes staged "
                "`narrative_discovery`, `narrative_canvas` y `narrative_blueprint`. "
                "No cambies la arquitectura, memoria, tools ni guardrails ya definidos. "
                "Explica por que la recomendacion encaja con el discovery y el canvas, "
                "y resalta tradeoffs relevantes sin inventar nuevos componentes.",
                context_bundle,
            )
        else:
            prompt = _localized_prompt(
                "Devuelve exclusivamente JSON valido segun el schema provisto. "
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
                context_request=context_envelope.context_request,
            )
            normalized = BlueprintNarrativeOutput.model_validate(parsed.model_dump(mode="json"))
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=BlueprintNarrativeOutput(narrative=normalize_text(normalized.narrative)),
                    provider_key=LLMProviderKey.codex_local.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="Codex local no pudo sintetizar la narrativa; se mantuvo la narrativa base.",
                    provider_key=LLMProviderKey.codex_local.value,
                ),
                context_envelope=context_envelope,
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
                warning="Codex local no esta disponible para recomendar tools minimas; se mantiene el preflight heuristico.",
                provider_key=LLMProviderKey.codex_local.value,
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
        context_sources = [
            CodexContextInlineSource(
                key="tool_recommendation_case",
                title="Compact tool recommendation case",
                content=json.dumps(case_payload, ensure_ascii=True, indent=2),
                required=True,
                summary="Digest compacto aprobado desde discovery, define y design para seleccionar tools minimas.",
            ),
            CodexContextInlineSource(
                key="tool_recommendation_catalog",
                title="Allowed tool recommendation catalog",
                content=json.dumps(catalog_payload, ensure_ascii=True, indent=2),
                required=True,
                summary="Catalogo permitido y shortlist heuristico prefiltrado que limita las tools elegibles.",
            ),
        ]
        context_envelope = self._build_context_envelope(
            task_kind="tool_recommendation_minimal",
            role="builder",
            sources=context_sources,
            context_bundle=context_bundle,
        )
        if self._uses_staged_context():
            prompt = _localized_prompt(
                build_tool_recommendation_staged_prompt(),
                context_bundle,
            )
        else:
            prompt = _localized_prompt(
                build_tool_recommendation_inline_prompt(
                    case_json=json.dumps(case_payload, ensure_ascii=True),
                    catalog_json=json.dumps(catalog_payload, ensure_ascii=True),
                ),
                context_bundle,
            )
        try:
            parsed = self.execution_service.execute_structured_prompt(
                task_kind="tool_recommendation_minimal",
                prompt=prompt,
                output_model=ToolRecommendationLLMOutput,
                context_request=context_envelope.context_request,
            )
            normalized = ToolRecommendationLLMOutput.model_validate(parsed.model_dump(mode="json"))
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=normalized,
                    provider_key=LLMProviderKey.codex_local.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="Codex local no pudo recomendar tools minimas; se mantuvo el preflight heuristico.",
                    provider_key=LLMProviderKey.codex_local.value,
                ),
                context_envelope=context_envelope,
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
