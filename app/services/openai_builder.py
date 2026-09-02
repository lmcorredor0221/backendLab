from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.core.config import get_settings, runtime_legacy_file_fallback_enabled
from app.models import (
    AgentExecutionBackend,
    BlueprintArtifact,
    CodexAuthMode,
    CanvasArtifact,
    CodexLocalProviderConfig,
    CodexLocalProviderConfigUpdate,
    AntigravityProviderConfig,
    AntigravityProviderConfigUpdate,
    DeepSeekProviderConfig,
    DeepSeekProviderConfigUpdate,
    DiscoveryArtifact,
    DiscoveryInput,
    EstimationReportArtifact,
    KnowledgeAccessBackend,
    LLMProviderKey,
    LLMProviderOption,
    LLMRuntimeSettings,
    LLMRuntimeSettingsUpdateRequest,
    OpenAIProviderConfig,
    OpenAIProviderConfigUpdate,
    ToolRecommendationLLMOutput,
    ToolRecommendationPromptInput,
    utc_now,
)
from app.services.agent_i18n import apply_agent_language_directive, get_effective_language
from app.services.llm_runtime.builder_contracts import (
    AgentDesignCritiqueInput,
    AgentDesignProposalOutput,
    AgentDesignInput,
    BlueprintNarrativeOutput,
    DesignCritiqueOutput,
    DiscoveryAnalysisInput,
    DiscoveryAnalysisOutput,
    EstimationRiskAnalysisInput,
    EstimationRiskAnalysisOutput,
    LLMArtifactResult,
    MemoryArchitectureCritiqueInput,
    MemoryArchitectureCritiqueOutput,
    MemoryArchitectureInput,
    MemoryArchitectureRecommendationOutput,
    RequirementsDefinitionInput,
    RequirementsDefinitionOutput,
    ValidationRunJudgmentInput,
    ValidationRunJudgmentOutput,
    ValidationScenarioGenerationInput,
    ValidationScenarioGenerationOutput,
    ValidationScenarioSimulationInput,
    ValidationSimulationOutput,
    sanitize_canvas,
    sanitize_discovery,
    validate_or_repair_structured_payload,
)
from app.services.diagram_center.contracts import DiagramGenerationInput, DiagramNotation
from app.services.diagram_center.semantic_repair import finalize_structured_diagram_artifact
from app.services.llm_runtime.api_context_adapter import APIProviderContextAdapter, APIProviderContextEnvelope
from app.services.llm_runtime.capability_registry import BuilderCapability, BuilderCapabilitySpec, get_builder_capability_spec
from app.services.llm_runtime.codex_cli.context_assembler import CodexContextInlineSource
from app.services.llm_runtime.codex_cli.execution_service import resolve_codex_executable_path
from app.services.llm_runtime.codex_cli.provider_facade import CodexLocalBuilderService
from app.services.llm_runtime.antigravity_cli.execution_service import resolve_agy_executable
from app.services.llm_runtime.antigravity_cli.provider_facade import AntigravityLocalBuilderService
from app.services.llm_runtime.prompt_templates import (
    build_tool_recommendation_context_task_instruction,
    build_tool_recommendation_system_instruction,
)
from app.services.llm_runtime.provider_router import BuilderProviderFacade
from app.services.llm_runtime.stage_context_types import StageContextBundle, build_llm_call_context
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_finops.provider_instrumentation import (
    FinOpsSessionFactory,
    default_finops_session_factory,
    record_provider_call,
)
from app.services.llm_finops.usage_normalization import normalize_deepseek_usage, normalize_openai_usage
from app.services.rules import normalize_text

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled through runtime fallback
    OpenAI = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


def _localized_instruction(instruction: str, context_bundle: StageContextBundle | None) -> str:
    language = get_effective_language(context_bundle.effective_language if context_bundle is not None else None)
    return apply_agent_language_directive(instruction, language)


class _APIContextAwareBuilderMixin:
    runtime_settings: LLMRuntimeSettings
    _context_adapter: APIProviderContextAdapter

    def _build_context_envelope(
        self,
        *,
        role: str,
        task_kind: str,
        task_instruction: str,
        inline_sources: list[CodexContextInlineSource],
        context_bundle: StageContextBundle | None = None,
    ) -> APIProviderContextEnvelope:
        return self._context_adapter.build(
            role=role,
            task_kind=task_kind,
            knowledge_access_backend=self.runtime_settings.knowledge_access_backend.value,
            task_instruction=task_instruction,
            inline_sources=inline_sources,
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

    def _attach_context_metadata(
        self,
        result: LLMArtifactResult,
        *,
        context_envelope: APIProviderContextEnvelope,
    ) -> LLMArtifactResult:
        result.knowledge_access_backend = context_envelope.knowledge_access_backend
        result.effective_context_backend = context_envelope.effective_context_backend
        result.context_used_sources = list(context_envelope.used_sources)
        result.context_stats = dict(context_envelope.context_stats)
        return result


def _capability_policy_payload(spec: BuilderCapabilitySpec) -> dict[str, object]:
    return {
        "llm_required": spec.llm_required,
        "critic_required": spec.critic_required,
        "timeout_ms": spec.timeout_ms,
        "max_retries": spec.max_retries,
        "fallback_policy": spec.fallback_policy,
    }


def _format_provider_bootstrap_error(detail: str | None, *, fallback: str) -> str:
    normalized = " ".join((detail or "").split()).strip()
    if not normalized:
        return fallback
    return normalized[:400]


def _format_provider_unavailable_warning(
    *,
    provider_label: str,
    capability: BuilderCapability,
    spec: BuilderCapabilitySpec,
    detail: str | None = None,
) -> str:
    base = f"{provider_label} no esta disponible para {capability.value}; policy={spec.fallback_policy}."
    normalized_detail = " ".join((detail or "").split()).strip()
    if not normalized_detail:
        return base
    return f"{base} Detalle: {normalized_detail[:240]}"


def _runtime_config_path() -> Path:
    path = Path(get_settings().llm_config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _clean_text(value: str, fallback: str) -> str:
    normalized = value.strip()
    return normalized or fallback


def _clean_url(value: str, fallback: str) -> str:
    normalized = value.strip().rstrip("/")
    return normalized or fallback.rstrip("/")


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, list):
        candidates = value
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        token = str(item).strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(token)
    return normalized


def _normalize_positive_int(value: Any, fallback: int, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(fallback)
    return max(minimum, parsed)


def _normalize_codex_auth_mode(value: str, fallback: str) -> str:
    normalized = _clean_text(value, fallback).strip().lower().replace("-", "_")
    aliases = {
        "chatgpt": "chatgpt_session",
        "session": "chatgpt_session",
        "token": "access_token",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"auto", "api_key", "access_token", "chatgpt_session", "profile"}:
        return normalized
    return "auto"


def _normalize_agent_execution_backend(value: str, fallback: str) -> str:
    normalized = _clean_text(value, fallback).strip().lower().replace("-", "_")
    if normalized in {"provider_native", "codex_cli", "shadow_codex_cli"}:
        return normalized
    return "provider_native"


def _normalize_knowledge_access_backend(value: str, fallback: str) -> str:
    normalized = _clean_text(value, fallback).strip().lower().replace("-", "_")
    if normalized in {"inline_context", "workspace_staged", "hybrid"}:
        return normalized
    return "inline_context"


def _normalize_deepseek_reasoning_effort(value: str, fallback: str) -> str:
    normalized = value.strip().lower() or fallback.strip().lower() or "high"
    if normalized in {"max", "xhigh"}:
        return "max"
    return "high"


def _is_bpmn_diagram_payload(payload: object) -> bool:
    return isinstance(payload, DiagramGenerationInput) and payload.notation == DiagramNotation.bpmn


def _effective_deepseek_reasoning_effort(
    capability: BuilderCapability,
    configured_effort: str,
    *,
    payload: object | None = None,
) -> str:
    normalized = _normalize_deepseek_reasoning_effort(configured_effort, configured_effort)
    if capability == BuilderCapability.generate_diagram_model and _is_bpmn_diagram_payload(payload):
        return "high"
    return normalized


def _structured_capability_max_tokens(capability: BuilderCapability, *, payload: object | None = None) -> int:
    if capability == BuilderCapability.generate_diagram_model and _is_bpmn_diagram_payload(payload):
        return 4096
    if capability in {
        BuilderCapability.propose_agent_design,
        BuilderCapability.critique_agent_design,
        BuilderCapability.recommend_memory_architecture,
        BuilderCapability.critique_memory_architecture,
        BuilderCapability.analyze_estimation_risks,
    }:
        return 6144
    return 4096


def _preserve_deepseek_reasoning_on_retry(capability: BuilderCapability, *, payload: object | None = None) -> bool:
    return capability == BuilderCapability.generate_diagram_model and not _is_bpmn_diagram_payload(payload)


def _expand_deepseek_retry_budget(capability: BuilderCapability, *, payload: object | None = None) -> bool:
    if capability == BuilderCapability.generate_diagram_model and _is_bpmn_diagram_payload(payload):
        return False
    return True


def _deepseek_retry_instruction(capability: BuilderCapability, *, payload: object | None = None) -> str:
    if capability == BuilderCapability.generate_diagram_model and _is_bpmn_diagram_payload(payload):
        return (
            "La respuesta anterior fue truncada por longitud. Regenera desde cero un unico objeto JSON valido, "
            "completo y mas compacto. Usa el minimo numero de nodos, edges, pools y lanes necesario para una vista "
            "BPMN trazable de nivel standard. Elimina detalle redundante, descripciones extensas y supuestos no "
            "esenciales. No incluyas markdown ni explicaciones."
        )
    if capability == BuilderCapability.generate_diagram_model:
        return (
            "La respuesta anterior fue truncada por longitud. Regenera desde cero un unico objeto JSON valido, "
            "completo y compacto. Conserva nodos, edges, pools y lanes indispensables para mantener trazabilidad. "
            "Si notation=bpmn, incluye start_event, end_event y usa sequence_flow dentro del mismo pool y "
            "message_flow entre pools. No incluyas markdown ni explicaciones."
        )
    return (
        "La respuesta anterior fue truncada por longitud. Regenera desde cero un unico objeto JSON valido, "
        "completo y compacto. No incluyas markdown ni explicaciones."
    )


def _resolve_schema_node(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in schema:
        ref = str(schema["$ref"])
        if ref.startswith("#/$defs/"):
            key = ref.split("/")[-1]
            referenced = root.get("$defs", {}).get(key, {})
            if isinstance(referenced, dict):
                return _resolve_schema_node(referenced, root)
    for composite_key in ("anyOf", "oneOf"):
        variants = schema.get(composite_key)
        if isinstance(variants, list) and variants:
            preferred = next(
                (
                    item
                    for item in variants
                    if isinstance(item, dict) and item.get("type") not in {None, "null"}
                ),
                variants[0],
            )
            if isinstance(preferred, dict):
                return _resolve_schema_node(preferred, root)
    if "allOf" in schema and isinstance(schema["allOf"], list):
        merged: dict[str, Any] = {}
        for item in schema["allOf"]:
            if isinstance(item, dict):
                merged = _merge_dicts(merged, _resolve_schema_node(item, root))
        if merged:
            return merged
    return schema


def _schema_example_payload(schema: dict[str, Any], root: dict[str, Any]) -> Any:
    resolved = _resolve_schema_node(schema, root)
    if "default" in resolved:
        return resolved["default"]
    if isinstance(resolved.get("enum"), list) and resolved["enum"]:
        return resolved["enum"][0]
    schema_type = resolved.get("type")
    if schema_type == "object":
        properties = resolved.get("properties", {})
        if not isinstance(properties, dict):
            return {}
        return {
            key: _schema_example_payload(value, root)
            for key, value in properties.items()
            if isinstance(value, dict)
        }
    if schema_type == "array":
        items = resolved.get("items")
        if isinstance(items, dict):
            return [_schema_example_payload(items, root)]
        return []
    if schema_type == "boolean":
        return False
    if schema_type in {"integer", "number"}:
        return 0
    if schema_type == "string":
        if resolved.get("format") == "date-time":
            return "2026-01-01T00:00:00Z"
        title = str(resolved.get("title", "value")).strip().lower().replace(" ", "_")
        return title or "value"
    return ""


def _build_deepseek_json_guidance(output_model: type[BaseModel]) -> str:
    schema = output_model.model_json_schema()
    example_payload = _schema_example_payload(schema, schema)
    return (
        "Responde solo con un objeto json valido, sin markdown ni comentarios.\n"
        f"JSON schema:\n{json.dumps(schema, ensure_ascii=True, indent=2)}\n\n"
        f"Ejemplo de json valido:\n{json.dumps(example_payload, ensure_ascii=True, indent=2)}"
    )


def _extract_json_payload(raw_content: str) -> dict[str, Any]:
    normalized = raw_content.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if len(lines) >= 3:
            normalized = "\n".join(lines[1:-1]).strip()
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start != -1 and end != -1 and end >= start:
        normalized = normalized[start : end + 1]
    payload = json.loads(normalized)
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek no devolvio un objeto JSON.")
    return payload


def _is_deepseek_length_finish_reason(finish_reason: str) -> bool:
    return finish_reason.strip().lower() in {"length", "max_tokens"}


def _deepseek_retry_max_tokens(max_tokens: int, *, expand_budget: bool) -> int:
    if not expand_budget:
        return max_tokens
    return min(max_tokens * 2, 8192)


def _compact_text(value: object, *, limit: int = 220, fallback: str = "") -> str:
    normalized = " ".join(str(value or "").split()).strip()
    if not normalized:
        return fallback
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _compact_string_list(values: list[object], *, limit: int = 5, item_limit: int = 180) -> list[str]:
    items: list[str] = []
    for value in values[:limit]:
        compact = _compact_text(value, limit=item_limit)
        if compact:
            items.append(compact)
    return items


def _compact_source_refs(values: list[str], *, limit: int = 8) -> list[str]:
    return [item.strip() for item in values[:limit] if item and item.strip()]


def _compact_discovery_input(discovery: DiscoveryInput | DiscoveryArtifact) -> dict[str, Any]:
    return {
        "problem_statement": _compact_text(discovery.problem_statement, limit=520),
        "current_user": _compact_text(discovery.current_user, limit=180),
        "current_process": _compact_text(discovery.current_process, limit=640),
        "desired_outcome": _compact_text(discovery.desired_outcome, limit=520),
        "autonomy_level": _compact_text(discovery.autonomy_level, limit=24),
        "constraints": _compact_string_list(list(discovery.constraints), limit=10, item_limit=220),
        "operational_baseline": {
            "current_time_spent": _compact_text(discovery.operational_baseline.current_time_spent, limit=140),
            "current_cost": _compact_text(discovery.operational_baseline.current_cost, limit=220),
            "frequent_errors": _compact_string_list(
                list(discovery.operational_baseline.frequent_errors),
                limit=10,
                item_limit=220,
            ),
            "automation_opportunities": _compact_string_list(
                list(discovery.operational_baseline.automation_opportunities),
                limit=10,
                item_limit=220,
            ),
        },
        "mvp_definition": {
            "v1_scope": _compact_string_list(list(discovery.mvp_definition.v1_scope), limit=10, item_limit=180),
            "out_of_scope": _compact_string_list(list(discovery.mvp_definition.out_of_scope), limit=10, item_limit=180),
            "north_star_metric": _compact_text(discovery.mvp_definition.north_star_metric, limit=260),
            "non_delegable_decisions": _compact_string_list(
                list(discovery.mvp_definition.non_delegable_decisions),
                limit=10,
                item_limit=180,
            ),
        },
    }


def _compact_discovery_artifact(discovery: DiscoveryArtifact) -> dict[str, Any]:
    payload = _compact_discovery_input(discovery)
    payload.update(
        {
            "case_type": _compact_text(discovery.case_type, limit=60),
            "value_statement": _compact_text(discovery.value_statement, limit=420),
        }
    )
    return payload


def _compact_canvas_artifact(canvas: CanvasArtifact) -> dict[str, Any]:
    return {
        "user_goal": _compact_text(canvas.user_goal, limit=520),
        "success_metric": _compact_text(canvas.success_metric, limit=260),
        "primary_risk": _compact_text(canvas.primary_risk, limit=320),
        "mvp_scope": _compact_string_list(list(canvas.mvp_scope), limit=10, item_limit=180),
        "out_of_scope": _compact_string_list(list(canvas.out_of_scope), limit=10, item_limit=180),
        "agent_profile": {
            "mission": _compact_text(canvas.agent_profile.mission, limit=420),
            "primary_user": _compact_text(canvas.agent_profile.primary_user, limit=180),
            "agent_task": _compact_text(canvas.agent_profile.agent_task, limit=420),
            "allowed_decisions": _compact_string_list(
                list(canvas.agent_profile.allowed_decisions),
                limit=8,
                item_limit=180,
            ),
            "prohibited_decisions": _compact_string_list(
                list(canvas.agent_profile.prohibited_decisions),
                limit=8,
                item_limit=180,
            ),
            "key_inputs": _compact_string_list(list(canvas.agent_profile.key_inputs), limit=10, item_limit=160),
            "expected_outputs": _compact_string_list(
                list(canvas.agent_profile.expected_outputs),
                limit=10,
                item_limit=160,
            ),
            "human_approvals": _compact_string_list(
                list(canvas.agent_profile.human_approvals),
                limit=8,
                item_limit=180,
            ),
            "success_metrics": _compact_string_list(
                list(canvas.agent_profile.success_metrics),
                limit=8,
                item_limit=180,
            ),
        },
    }


def _compact_safety_checks(checks: list[Any], *, limit: int = 6) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in checks[:limit]:
        compacted.append(
            {
                "category": _compact_text(getattr(item, "category", ""), limit=80),
                "risk": _compact_text(getattr(item, "risk", ""), limit=140),
                "severity": _compact_text(getattr(item, "severity", ""), limit=40),
                "mitigation": _compact_text(getattr(item, "mitigation", ""), limit=180),
                "status": _compact_text(getattr(item, "status", ""), limit=40),
            }
        )
    return compacted


def _compact_blueprint_tool(tool: Any) -> dict[str, Any]:
    return {
        "name": _compact_text(getattr(tool, "name", ""), limit=120),
        "purpose": _compact_text(getattr(tool, "purpose", ""), limit=260),
        "owner": _compact_text(getattr(tool, "owner", ""), limit=80),
        "archetype": _compact_text(getattr(tool, "archetype", ""), limit=60),
        "integration_kind": _compact_text(getattr(tool, "integration_kind", ""), limit=60),
        "tool_type": _compact_text(getattr(tool, "tool_type", ""), limit=40),
        "execution_stage": _compact_text(getattr(tool, "execution_stage", ""), limit=40),
        "when_to_use": _compact_text(getattr(tool, "when_to_use", ""), limit=260),
        "risk_level": _compact_text(getattr(tool, "risk_level", ""), limit=40),
        "requires_approval": bool(getattr(tool, "requires_approval", False)),
        "has_side_effects": bool(getattr(tool, "has_side_effects", False)),
        "inputs": _compact_string_list(list(getattr(tool, "inputs", [])), limit=8, item_limit=140),
        "outputs": _compact_string_list(list(getattr(tool, "outputs", [])), limit=8, item_limit=140),
        "permissions": _compact_string_list(list(getattr(tool, "permissions", [])), limit=8, item_limit=140),
        "scopes": _compact_string_list(list(getattr(tool, "scopes", [])), limit=8, item_limit=140),
        "validations": _compact_string_list(list(getattr(tool, "validations", [])), limit=8, item_limit=160),
        "failure_mode": _compact_text(getattr(tool, "failure_mode", ""), limit=200),
        "timeout_policy": _compact_text(getattr(tool, "timeout_policy", ""), limit=140),
    }


def _compact_blueprint_artifact(blueprint: BlueprintArtifact) -> dict[str, Any]:
    compact_tools = [_compact_blueprint_tool(item) for item in blueprint.tools[:16]]
    return {
        "architecture": _compact_text(blueprint.architecture, limit=220),
        "reasoning_pattern": _compact_text(blueprint.reasoning_pattern, limit=220),
        "memory_strategy": _compact_text(blueprint.memory_strategy, limit=260),
        "readiness_state": getattr(blueprint.readiness_state, "value", blueprint.readiness_state),
        "guardrails": _compact_string_list(list(blueprint.guardrails), limit=12, item_limit=220),
        "tools": compact_tools,
        "tool_count": len(blueprint.tools),
        "omitted_tool_count": max(0, len(blueprint.tools) - len(compact_tools)),
        "llm_policy": {
            "provider": _compact_text(blueprint.llm_policy.provider, limit=40),
            "fast_model": _compact_text(blueprint.llm_policy.fast_model, limit=80),
            "reasoning_model": _compact_text(blueprint.llm_policy.reasoning_model, limit=80),
            "context_policy": _compact_text(blueprint.llm_policy.context_policy, limit=220),
            "fallback_policy": _compact_text(blueprint.llm_policy.fallback_policy, limit=220),
            "budget_policy": _compact_text(blueprint.llm_policy.budget_policy, limit=220),
            "output_validation_policy": _compact_text(
                blueprint.llm_policy.output_validation_policy,
                limit=220,
            ),
            "functions": [
                {
                    "role": _compact_text(item.role, limit=60),
                    "provider": _compact_text(item.provider, limit=40),
                    "model": _compact_text(item.model, limit=80),
                    "reasoning_effort": _compact_text(item.reasoning_effort, limit=40),
                    "max_tokens": int(item.max_tokens or 0),
                }
                for item in blueprint.llm_policy.functions[:6]
            ],
        },
        "memory_profile": {
            "strategy": _compact_text(blueprint.memory_profile.strategy, limit=100),
            "storage_layers": _compact_string_list(
                list(blueprint.memory_profile.storage_layers),
                limit=6,
                item_limit=80,
            ),
            "write_policy": _compact_text(blueprint.memory_profile.write_policy, limit=240),
            "retrieval_policy": _compact_text(blueprint.memory_profile.retrieval_policy, limit=240),
            "review_trigger": _compact_text(blueprint.memory_profile.review_trigger, limit=200),
            "goal_drift_guard": _compact_text(blueprint.memory_profile.goal_drift_guard, limit=220),
            "retention_policy": _compact_text(blueprint.memory_profile.retention_policy, limit=220),
            "sensitivity_rules": _compact_string_list(
                list(blueprint.memory_profile.sensitivity_rules),
                limit=6,
                item_limit=120,
            ),
        },
        "knowledge_profile": {
            "mode": _compact_text(blueprint.knowledge_profile.mode, limit=40),
            "sources": [
                {
                    "key": _compact_text(item.key, limit=80),
                    "title": _compact_text(item.title, limit=120),
                    "source_type": _compact_text(item.source_type, limit=60),
                    "owner": _compact_text(item.owner, limit=80),
                    "description": _compact_text(item.description, limit=160),
                }
                for item in blueprint.knowledge_profile.sources[:6]
            ],
            "ingestion_policy": {
                "parser": _compact_text(blueprint.knowledge_profile.ingestion_policy.parser, limit=60),
                "chunking_policy": _compact_text(
                    blueprint.knowledge_profile.ingestion_policy.chunking_policy,
                    limit=120,
                ),
                "include_filters": _compact_string_list(
                    list(blueprint.knowledge_profile.ingestion_policy.include_filters),
                    limit=5,
                    item_limit=100,
                ),
                "exclude_filters": _compact_string_list(
                    list(blueprint.knowledge_profile.ingestion_policy.exclude_filters),
                    limit=5,
                    item_limit=100,
                ),
            },
            "retrieval_policy": {
                "top_k": int(blueprint.knowledge_profile.retrieval_policy.top_k or 0),
                "search_mode": _compact_text(
                    blueprint.knowledge_profile.retrieval_policy.search_mode,
                    limit=60,
                ),
                "reranking_policy": _compact_text(
                    blueprint.knowledge_profile.retrieval_policy.reranking_policy,
                    limit=120,
                ),
                "fallback_behavior": _compact_text(
                    blueprint.knowledge_profile.retrieval_policy.fallback_behavior,
                    limit=120,
                ),
            },
            "grounding_policy": {
                "citations_policy": _compact_text(
                    blueprint.knowledge_profile.grounding_policy.citations_policy,
                    limit=120,
                ),
                "confidence_policy": _compact_text(
                    blueprint.knowledge_profile.grounding_policy.confidence_policy,
                    limit=120,
                ),
                "no_evidence_behavior": _compact_text(
                    blueprint.knowledge_profile.grounding_policy.no_evidence_behavior,
                    limit=120,
                ),
                "contradictory_evidence_behavior": _compact_text(
                    blueprint.knowledge_profile.grounding_policy.contradictory_evidence_behavior,
                    limit=120,
                ),
            },
            "sensitivity_rules": _compact_string_list(
                list(blueprint.knowledge_profile.sensitivity_rules),
                limit=6,
                item_limit=120,
            ),
            "notes": _compact_text(blueprint.knowledge_profile.notes, limit=260),
        },
        "safety_checks": _compact_safety_checks(blueprint.safety_checks),
        "delivery_package": {
            "decision_summary": _compact_text(blueprint.delivery_package.decision_summary, limit=360),
            "deliverables": _compact_string_list(
                [item.key for item in blueprint.delivery_package.deliverables],
                limit=8,
                item_limit=80,
            ),
            "pattern_catalog": _compact_string_list(
                [
                    f"{item.family}:{getattr(item, 'pattern_key', getattr(item, 'key', ''))}:{item.label}"
                    for item in blueprint.delivery_package.pattern_catalog
                ],
                limit=8,
                item_limit=120,
            ),
            "component_readiness": _compact_string_list(
                [
                    (
                        f"{item.component}:"
                        f"{getattr(getattr(item, 'status', ''), 'value', getattr(item, 'status', ''))}:"
                        f"{'; '.join(list(getattr(item, 'blocking_issues', []))[:2]) or (str(getattr(item, 'completed_checks', 0)) + '/' + str(getattr(item, 'total_checks', 0)) + ' checks')}"
                    )
                    for item in blueprint.delivery_package.component_readiness
                ],
                limit=6,
                item_limit=160,
            ),
            "risk_summary": _compact_text(
                getattr(
                    blueprint.delivery_package.risk_summary,
                    "summary",
                    getattr(blueprint.delivery_package.risk_summary, "overall_summary", ""),
                ),
                limit=220,
            ),
        },
        "narrative": _compact_text(blueprint.narrative, limit=1200),
    }


def _compact_definition_validation(validation: RequirementsDefinitionOutput) -> dict[str, Any]:
    return {
        "coverage_ratio": float(validation.validation.coverage_ratio or 0),
        "blocking_issues": _compact_string_list(list(validation.validation.blocking_issues), limit=10, item_limit=120),
        "contradictions": _compact_string_list(list(validation.validation.contradictions), limit=8, item_limit=120),
        "vague_nfrs": _compact_string_list(list(validation.validation.vague_nfrs), limit=8, item_limit=120),
        "missing_acceptance": _compact_string_list(list(validation.validation.missing_acceptance), limit=8, item_limit=80),
        "untraced_items": _compact_string_list(list(validation.validation.untraced_items), limit=8, item_limit=80),
        "blocking_open_questions": _compact_string_list(
            list(validation.validation.blocking_open_questions),
            limit=8,
            item_limit=80,
        ),
    }


def _compact_definition_artifact(definition: RequirementsDefinitionOutput) -> dict[str, Any]:
    return {
        "summary": _compact_text(definition.summary, limit=260),
        "measurable_objectives": _compact_string_list(list(definition.measurable_objectives), limit=8, item_limit=140),
        "functional_requirements": [
            {
                "key": _compact_text(item.key, limit=80),
                "title": _compact_text(item.title, limit=140),
                "priority": _compact_text(item.priority, limit=24),
                "requirement": _compact_text(item.requirement, limit=200),
                "acceptance": _compact_string_list(list(item.acceptance), limit=3, item_limit=120),
            }
            for item in definition.functional_requirements[:8]
        ],
        "non_functional_requirements": [
            {
                "key": _compact_text(item.key, limit=80),
                "title": _compact_text(item.title, limit=140),
                "category": _compact_text(item.category, limit=40),
                "metric": _compact_text(item.metric, limit=60),
                "target": _compact_text(item.target, limit=80),
                "requirement": _compact_text(item.requirement, limit=180),
            }
            for item in definition.non_functional_requirements[:8]
        ],
        "business_rules": _compact_string_list(
            [f"{item.key}: {item.rule}" for item in definition.business_rules],
            limit=6,
            item_limit=180,
        ),
        "dependencies": _compact_string_list(
            [f"{item.key}: {item.dependency}" for item in definition.dependencies],
            limit=6,
            item_limit=180,
        ),
        "assumptions": _compact_string_list(
            [item.assumption for item in definition.assumptions],
            limit=6,
            item_limit=160,
        ),
        "open_questions": _compact_string_list(
            [item.question for item in definition.open_questions],
            limit=8,
            item_limit=160,
        ),
        "validation": _compact_definition_validation(definition),
        "canvas_projection": _compact_canvas_artifact(definition.canvas_projection),
        "evidence_refs": _compact_source_refs(list(definition.evidence_refs)),
        "confidence": float(definition.confidence or 0),
    }


def _compact_design_alternative(alternative: Any) -> dict[str, Any]:
    return {
        "alternative_key": _compact_text(getattr(alternative, "alternative_key", ""), limit=80),
        "label": _compact_text(getattr(alternative, "label", ""), limit=120),
        "recommendation_role": _compact_text(getattr(alternative, "recommendation_role", ""), limit=60),
        "agent_archetype": _compact_text(getattr(alternative, "agent_archetype", ""), limit=100),
        "pattern_family": _compact_text(getattr(alternative, "pattern_family", ""), limit=120),
        "architecture": _compact_text(getattr(alternative, "architecture", ""), limit=80),
        "reasoning_pattern": _compact_text(getattr(alternative, "reasoning_pattern", ""), limit=80),
        "coordination_model": _compact_text(getattr(alternative, "coordination_model", ""), limit=80),
        "summary": _compact_text(getattr(alternative, "summary", ""), limit=220),
        "business_fit": _compact_text(getattr(alternative, "business_fit", ""), limit=220),
        "value_hypothesis": _compact_text(getattr(alternative, "value_hypothesis", ""), limit=180),
        "operational_model": _compact_text(getattr(alternative, "operational_model", ""), limit=180),
        "why_recommended": _compact_text(getattr(alternative, "why_recommended", ""), limit=180),
        "why_not_simpler": _compact_text(getattr(alternative, "why_not_simpler", ""), limit=180),
        "why_not_more_complex": _compact_text(getattr(alternative, "why_not_more_complex", ""), limit=180),
        "topology": _compact_text(getattr(alternative, "topology", ""), limit=140),
        "approval_points": _compact_string_list(list(getattr(alternative, "approval_points", [])), limit=5, item_limit=120),
        "tradeoffs": _compact_string_list(list(getattr(alternative, "tradeoffs", [])), limit=5, item_limit=120),
        "tool_implications": _compact_string_list(
            list(getattr(alternative, "tool_implications", [])),
            limit=6,
            item_limit=160,
        ),
        "memory_implications": _compact_string_list(
            list(getattr(alternative, "memory_implications", [])),
            limit=6,
            item_limit=160,
        ),
        "risk_tradeoffs": _compact_string_list(
            list(getattr(alternative, "risk_tradeoffs", [])),
            limit=5,
            item_limit=140,
        ),
        "business_metrics": _compact_string_list(
            list(getattr(alternative, "business_metrics", [])),
            limit=5,
            item_limit=120,
        ),
        "assumptions": _compact_string_list(list(getattr(alternative, "assumptions", [])), limit=5, item_limit=120),
        "fit_rationale": _compact_string_list(list(getattr(alternative, "fit_rationale", [])), limit=5, item_limit=120),
        "fit_score": float(getattr(alternative, "fit_score", 0) or 0),
        "roles": _compact_string_list(
            [
                f"{item.key}:{item.title}:{item.responsibility}"
                for item in list(getattr(alternative, "roles", []))
            ],
            limit=6,
            item_limit=160,
        ),
        "handoffs": _compact_string_list(
            [
                f"{item.from_role}->{item.to_role}:{item.trigger}"
                for item in list(getattr(alternative, "handoffs", []))
            ],
            limit=6,
            item_limit=160,
        ),
        "failure_modes": _compact_string_list(
            [
                f"{item.scenario}:{item.retry_strategy}:{item.compensation_strategy}"
                for item in list(getattr(alternative, "failure_modes", []))
            ],
            limit=5,
            item_limit=180,
        ),
        "blueprint_projection": {
            "architecture": _compact_text(getattr(getattr(alternative, "blueprint_projection", None), "architecture", ""), limit=80),
            "reasoning_pattern": _compact_text(
                getattr(getattr(alternative, "blueprint_projection", None), "reasoning_pattern", ""),
                limit=80,
            ),
            "guardrails": _compact_string_list(
                list(getattr(getattr(alternative, "blueprint_projection", None), "guardrails", [])),
                limit=6,
                item_limit=120,
            ),
            "narrative": _compact_text(
                getattr(getattr(alternative, "blueprint_projection", None), "narrative", ""),
                limit=240,
            ),
            "tool_implications": _compact_string_list(
                list(getattr(getattr(alternative, "blueprint_projection", None), "tool_implications", [])),
                limit=6,
                item_limit=160,
            ),
            "memory_strategy": _compact_text(
                getattr(getattr(alternative, "blueprint_projection", None), "memory_strategy", ""),
                limit=80,
            ),
            "memory_implications": _compact_string_list(
                list(getattr(getattr(alternative, "blueprint_projection", None), "memory_implications", [])),
                limit=6,
                item_limit=160,
            ),
            "cost_complexity_implications": _compact_string_list(
                list(getattr(getattr(alternative, "blueprint_projection", None), "cost_complexity_implications", [])),
                limit=4,
                item_limit=120,
            ),
        },
    }


def _compact_design_coverage_entry(entry: Any) -> dict[str, Any]:
    return {
        "requirement_key": _compact_text(getattr(entry, "requirement_key", ""), limit=80),
        "requirement_title": _compact_text(getattr(entry, "requirement_title", ""), limit=140),
        "category": _compact_text(getattr(entry, "category", ""), limit=40),
        "priority": _compact_text(getattr(entry, "priority", ""), limit=24),
        "coverage_status": _compact_text(getattr(entry, "coverage_status", ""), limit=40),
        "rationale": _compact_text(getattr(entry, "rationale", ""), limit=180),
        "source_refs": _compact_source_refs(list(getattr(entry, "source_refs", []))),
    }


def _compact_design_proposal_output(proposal: AgentDesignProposalOutput) -> dict[str, Any]:
    return {
        "summary": _compact_text(proposal.summary, limit=260),
        "recommended_alternative_key": _compact_text(proposal.recommended_alternative_key, limit=80),
        "architecture": _compact_text(proposal.architecture, limit=80),
        "reasoning_pattern": _compact_text(proposal.reasoning_pattern, limit=80),
        "memory_strategy": _compact_text(proposal.memory_strategy, limit=80),
        "coordination_model": _compact_text(proposal.coordination_model, limit=80),
        "decision_rationale": _compact_text(proposal.decision_rationale, limit=240),
        "alternatives": [_compact_design_alternative(item) for item in proposal.alternatives[:3]],
        "requirements_coverage": [
            _compact_design_coverage_entry(item) for item in proposal.requirements_coverage[:10]
        ],
        "tooling_principles": _compact_string_list(list(proposal.tooling_principles), limit=6, item_limit=140),
        "design_decisions": _compact_string_list(
            [
                f"{item.dimension}:{item.selected_option}:{item.rationale}"
                for item in proposal.design_decisions
            ],
            limit=8,
            item_limit=180,
        ),
        "open_questions": _compact_string_list(list(proposal.open_questions), limit=8, item_limit=160),
        "guided_questions": _compact_string_list(
            [item.question for item in proposal.guided_questions],
            limit=5,
            item_limit=160,
        ),
        "narrative": _compact_text(proposal.narrative, limit=320),
        "evidence_refs": _compact_source_refs(list(proposal.evidence_refs)),
        "confidence": float(proposal.confidence or 0),
    }


def _compact_memory_architecture_recommendation_output(
    proposal: MemoryArchitectureRecommendationOutput,
) -> dict[str, Any]:
    return {
        "memory_strategy": _compact_text(proposal.memory_strategy, limit=220),
        "short_term_strategy": _compact_text(proposal.short_term_strategy, limit=360),
        "long_term_strategy": _compact_text(proposal.long_term_strategy, limit=360),
        "retrieval_strategy": _compact_text(proposal.retrieval_strategy, limit=360),
        "storage_layers": _compact_string_list(list(proposal.storage_layers), limit=10, item_limit=140),
        "write_policy": _compact_text(proposal.write_policy, limit=360),
        "pruning_policy": _compact_text(proposal.pruning_policy, limit=300),
        "security_notes": _compact_string_list(list(proposal.security_notes), limit=10, item_limit=180),
        "tool_dependency_requests": _compact_string_list(
            list(proposal.tool_dependency_requests),
            limit=12,
            item_limit=140,
        ),
        "open_questions": _compact_string_list(list(proposal.open_questions), limit=10, item_limit=220),
        "guided_questions": _compact_string_list(
            [item.question for item in proposal.guided_questions],
            limit=8,
            item_limit=220,
        ),
        "rationale": _compact_text(proposal.rationale, limit=520),
    }


def _compact_validation_scenario_item(scenario: Any) -> dict[str, Any]:
    return {
        "scenario_key": _compact_text(getattr(scenario, "scenario_key", ""), limit=80),
        "title": _compact_text(getattr(scenario, "title", ""), limit=140),
        "objective": _compact_text(getattr(scenario, "objective", ""), limit=200),
        "steps": _compact_string_list(list(getattr(scenario, "steps", [])), limit=8, item_limit=140),
        "expected_outcomes": _compact_string_list(
            list(getattr(scenario, "expected_outcomes", [])),
            limit=6,
            item_limit=140,
        ),
        "failure_signals": _compact_string_list(
            list(getattr(scenario, "failure_signals", [])),
            limit=6,
            item_limit=140,
        ),
        "priority": _compact_text(getattr(scenario, "priority", ""), limit=24),
    }


def _compact_validation_simulation_output(simulation: ValidationSimulationOutput) -> dict[str, Any]:
    return {
        "scenario_key": _compact_text(simulation.scenario_key, limit=80),
        "result_status": _compact_text(simulation.result_status, limit=40),
        "simulated_transcript": _compact_string_list(list(simulation.simulated_transcript), limit=10, item_limit=180),
        "observed_decisions": _compact_string_list(list(simulation.observed_decisions), limit=8, item_limit=160),
        "tool_interactions": _compact_string_list(list(simulation.tool_interactions), limit=8, item_limit=160),
        "issues": _compact_string_list(list(simulation.issues), limit=8, item_limit=160),
    }


def _compact_estimation_report_artifact(report: EstimationReportArtifact) -> dict[str, Any]:
    return {
        "maturity_stage": getattr(report.maturity_stage, "value", report.maturity_stage),
        "blueprint_version_number": report.blueprint_version_number,
        "current_blueprint_version_number": report.current_blueprint_version_number,
        "source_artifacts": _compact_string_list(list(report.source_artifacts), limit=8, item_limit=100),
        "assumptions": _compact_string_list(list(report.assumptions), limit=8, item_limit=160),
        "risk_drivers": _compact_string_list(list(report.risk_drivers), limit=8, item_limit=160),
        "traditional": {
            "hours_total": float(getattr(report.traditional, "hours_total", 0) or 0),
            "duration_weeks": float(getattr(report.traditional, "duration_weeks", 0) or 0),
            "cost_total": float(getattr(report.traditional, "cost_total", 0) or 0),
        },
        "agentic": {
            "active_provider": getattr(getattr(report.agentic, "active_provider", ""), "value", getattr(report.agentic, "active_provider", "")),
            "pricing_policy": _compact_text(getattr(report.agentic, "pricing_policy", ""), limit=80),
            "provider_model": _compact_text(getattr(report.agentic, "provider_model", ""), limit=100),
            "hours_total": float(getattr(report.agentic, "hours_total", 0) or 0),
            "duration_weeks": float(getattr(report.agentic, "duration_weeks", 0) or 0),
            "cost_total": float(getattr(report.agentic, "cost_total", 0) or 0),
            "automation_coverage_percent": int(getattr(report.agentic, "automation_coverage_percent", 0) or 0),
            "blueprint_design_coverage_percent": int(getattr(report.agentic, "blueprint_design_coverage_percent", 0) or 0),
            "acp_package_readiness_percent": int(getattr(report.agentic, "acp_package_readiness_percent", 0) or 0),
            "implementation_scope_coverage_percent": int(
                getattr(report.agentic, "implementation_scope_coverage_percent", 0) or 0
            ),
            "pricing_assumptions": _compact_string_list(
                list(getattr(report.agentic, "pricing_assumptions", [])),
                limit=6,
                item_limit=140,
            ),
        },
        "confidence": {
            "score": int(report.confidence.score or 0),
            "label": getattr(report.confidence.label, "value", report.confidence.label),
            "uncertainty_band_percent": int(report.confidence.uncertainty_band_percent or 0),
            "blocking_gaps": int(report.confidence.blocking_gaps or 0),
            "open_questions": int(report.confidence.open_questions or 0),
            "design_gap_count": int(report.confidence.design_gap_count or 0),
            "implementation_gap_count": int(report.confidence.implementation_gap_count or 0),
            "positive_signals": _compact_string_list(
                list(report.confidence.positive_signals),
                limit=6,
                item_limit=140,
            ),
            "negative_signals": _compact_string_list(
                list(report.confidence.negative_signals),
                limit=6,
                item_limit=140,
            ),
            "recommended_next_actions": _compact_string_list(
                list(report.confidence.recommended_next_actions),
                limit=6,
                item_limit=140,
            ),
        },
        "deterministic_inputs": {
            "pricing_catalog_signature": _compact_text(
                report.deterministic_inputs.pricing_catalog_signature,
                limit=120,
            ),
            "validation_fingerprint": _compact_text(
                report.deterministic_inputs.validation_fingerprint,
                limit=120,
            ),
            "benchmark_corpus_hash": _compact_text(
                report.deterministic_inputs.benchmark_corpus_hash,
                limit=120,
            ),
            "catalogs_used": _compact_string_list(
                list(report.deterministic_inputs.catalogs_used),
                limit=6,
                item_limit=100,
            ),
            "benchmark_ids": _compact_string_list(
                list(report.deterministic_inputs.benchmark_ids),
                limit=8,
                item_limit=100,
            ),
            "formula_notes": _compact_string_list(
                list(report.deterministic_inputs.formula_notes),
                limit=6,
                item_limit=160,
            ),
            "calibration_sample_size": int(report.deterministic_inputs.calibration_sample_size or 0),
        },
        "notes": _compact_string_list(list(report.notes), limit=6, item_limit=140),
    }


def _compact_tool_recommendation_case_payload(prompt_input: ToolRecommendationPromptInput) -> dict[str, Any]:
    return {
        "prompt_version": _compact_text(prompt_input.prompt_version, limit=60),
        "source_session_id": str(prompt_input.source_session_id or ""),
        "source_blueprint_version": prompt_input.source_blueprint_version,
        "case_classification": _compact_text(prompt_input.case_classification, limit=80),
        "agent_goal": _compact_text(prompt_input.agent_goal, limit=240),
        "primary_user": _compact_text(prompt_input.primary_user, limit=120),
        "workflow_summary": _compact_text(prompt_input.workflow_summary, limit=260),
        "constraints_summary": _compact_text(prompt_input.constraints_summary, limit=220),
        "source_refs": _compact_source_refs(list(prompt_input.source_refs)),
        "core_workflows": _compact_string_list(list(prompt_input.core_workflows), limit=8, item_limit=120),
        "interaction_modes": _compact_string_list(list(prompt_input.interaction_modes), limit=6, item_limit=100),
        "required_information_sources": _compact_string_list(
            list(prompt_input.required_information_sources),
            limit=8,
            item_limit=120,
        ),
        "required_write_actions": _compact_string_list(
            list(prompt_input.required_write_actions),
            limit=8,
            item_limit=120,
        ),
        "approval_boundaries": _compact_string_list(
            list(prompt_input.approval_boundaries),
            limit=8,
            item_limit=120,
        ),
        "hard_constraints": _compact_string_list(list(prompt_input.hard_constraints), limit=8, item_limit=120),
        "requirements_coverage": [
            {
                "requirement_key": _compact_text(item.requirement_key, limit=80),
                "requirement_title": _compact_text(item.requirement_title, limit=140),
                "category": _compact_text(item.category, limit=40),
                "priority": _compact_text(item.priority, limit=24),
                "coverage_status": _compact_text(item.coverage_status, limit=40),
                "covered_by_tool_keys": _compact_string_list(
                    list(item.covered_by_tool_keys),
                    limit=5,
                    item_limit=80,
                ),
                "rationale": _compact_text(item.rationale, limit=180),
            }
            for item in prompt_input.requirements_coverage[:10]
        ],
        "design_role_coverage": [
            {
                "role_key": _compact_text(item.role_key, limit=80),
                "role_title": _compact_text(item.role_title, limit=120),
                "responsibility": _compact_text(item.responsibility, limit=160),
                "coverage_status": _compact_text(item.coverage_status, limit=40),
                "covered_by_tool_keys": _compact_string_list(
                    list(item.covered_by_tool_keys),
                    limit=5,
                    item_limit=80,
                ),
                "rationale": _compact_text(item.rationale, limit=180),
            }
            for item in prompt_input.design_role_coverage[:8]
        ],
        "design_tool_implications": _compact_string_list(
            list(prompt_input.design_tool_implications),
            limit=8,
            item_limit=180,
        ),
        "design_memory_implications": _compact_string_list(
            list(prompt_input.design_memory_implications),
            limit=8,
            item_limit=180,
        ),
        "existing_gaps": [
            {
                "gap_key": _compact_text(item.gap_key, limit=80),
                "title": _compact_text(item.title, limit=120),
                "question": _compact_text(item.question, limit=180),
                "reason": _compact_text(item.reason, limit=180),
                "impact": _compact_text(item.impact, limit=160),
                "severity": _compact_text(item.severity, limit=40),
            }
            for item in prompt_input.existing_gaps[:6]
        ],
        "compact_evidence": _compact_string_list(list(prompt_input.compact_evidence), limit=8, item_limit=160),
    }


def _compact_tool_recommendation_catalog_payload(prompt_input: ToolRecommendationPromptInput) -> dict[str, Any]:
    return {
        "mandatory_tool_keys": [item.value for item in prompt_input.mandatory_tool_keys[:8]],
        "forbidden_tool_keys": [item.value for item in prompt_input.forbidden_tool_keys[:8]],
        "candidate_tools": [
            {
                "tool_key": item.tool_key.value,
                "tool_label": _compact_text(item.tool_label, limit=120),
                "family_key": _compact_text(item.family_key, limit=80),
                "family_status": _compact_text(item.family_status, limit=40),
                "capability_covered": _compact_text(item.capability_covered, limit=160),
                "reason": _compact_text(item.reason, limit=180),
                "selection_notes": _compact_string_list(list(item.selection_notes), limit=4, item_limit=120),
            }
            for item in prompt_input.candidate_tools[:12]
        ],
    }


def _compact_diagram_resolved_input(value: dict[str, Any]) -> dict[str, Any]:
    evidence_items = value.get("evidence", [])
    compact_evidence: list[dict[str, Any]] = []
    if isinstance(evidence_items, list):
        for item in evidence_items[:3]:
            if not isinstance(item, dict):
                continue
            compact_evidence.append(
                {
                    "artifact_key": _compact_text(item.get("artifact_key", ""), limit=80),
                    "ref": _compact_text(item.get("ref", ""), limit=100),
                    "summary": _compact_text(
                        (item.get("content") or {}).get("summary", "") if isinstance(item.get("content"), dict) else "",
                        limit=260,
                    ),
                }
            )
    return {
        "input_key": _compact_text(value.get("input_key", ""), limit=120),
        "status": _compact_text(value.get("status", ""), limit=40),
        "matched_artifact_keys": _compact_string_list(value.get("matched_artifact_keys", []), limit=6, item_limit=80)
        if isinstance(value.get("matched_artifact_keys"), list)
        else [],
        "artifact_refs": _compact_string_list(value.get("artifact_refs", []), limit=6, item_limit=100)
        if isinstance(value.get("artifact_refs"), list)
        else [],
        "evidence": compact_evidence,
    }


def _compact_diagram_approved_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"summary": _compact_text(value, limit=220)}
    content = value.get("content") if isinstance(value.get("content"), dict) else {}
    summary = (
        value.get("summary")
        or value.get("title")
        or content.get("summary")
        or content.get("narrative")
        or content.get("description")
        or content.get("decision_summary")
        or ""
    )
    return {
        "key": _compact_text(value.get("key") or value.get("artifact_key") or "", limit=100),
        "kind": _compact_text(value.get("kind") or value.get("artifact_kind") or "", limit=80),
        "title": _compact_text(value.get("title") or content.get("title") or "", limit=140),
        "summary": _compact_text(summary, limit=420),
        "state": _compact_text(value.get("state") or value.get("status") or "", limit=60),
        "source_refs": _compact_source_refs(list(value.get("source_refs", [])), limit=6)
        if isinstance(value.get("source_refs"), list)
        else [],
    }


def _capability_source_for_api(spec: BuilderCapabilitySpec, payload: BaseModel) -> CodexContextInlineSource:
    raw_payload = payload.model_dump(mode="json")
    compact_payload = _serialize_capability_payload_for_api(payload)
    raw_json = json.dumps(raw_payload, ensure_ascii=True, default=str)
    compact_json = json.dumps(compact_payload, ensure_ascii=True, indent=2, default=str)
    compact_chars = len(compact_json)
    raw_chars = len(raw_json)
    return CodexContextInlineSource(
        key=spec.source_key,
        title=spec.source_title,
        content=compact_json,
        required=True,
        summary=spec.source_summary,
        metadata={
            "context_quality_version": "context-quality.v1",
            "input_payload_chars": raw_chars,
            "compact_payload_chars": compact_chars,
            "compact_payload_tokens_est": max(1, (compact_chars + 3) // 4),
            "compact_retention_pct": round((compact_chars / max(1, raw_chars)) * 100, 2),
            "payload_model": payload.__class__.__name__,
            "source_key": spec.source_key,
        },
    )


def _serialize_capability_payload_for_api(payload: BaseModel) -> dict[str, Any]:
    if isinstance(payload, DiscoveryAnalysisInput):
        return {
            "analysis_goal": _compact_text(payload.analysis_goal, limit=180),
            "known_gaps": _compact_string_list(list(payload.known_gaps), limit=8, item_limit=120),
            "source_refs": _compact_source_refs(list(payload.source_refs)),
            "discovery_capture": _compact_discovery_input(payload.discovery_capture),
        }
    if isinstance(payload, RequirementsDefinitionInput):
        return {
            "discovery": _compact_discovery_artifact(payload.discovery),
            "canvas": _compact_canvas_artifact(payload.canvas) if payload.canvas is not None else None,
            "known_constraints": _compact_string_list(list(payload.known_constraints), limit=8, item_limit=120),
            "source_refs": _compact_source_refs(list(payload.source_refs)),
        }
    if isinstance(payload, AgentDesignInput):
        return {
            "discovery": _compact_discovery_artifact(payload.discovery),
            "canvas": _compact_canvas_artifact(payload.canvas),
            "current_blueprint": _compact_blueprint_artifact(payload.current_blueprint)
            if payload.current_blueprint is not None
            else None,
            "requirement_digest": _compact_string_list(list(payload.requirement_digest), limit=24, item_limit=260),
            "source_refs": _compact_source_refs(list(payload.source_refs)),
        }
    if isinstance(payload, AgentDesignCritiqueInput):
        return {
            "discovery": _compact_discovery_artifact(payload.discovery),
            "canvas": _compact_canvas_artifact(payload.canvas),
            "proposal": _compact_design_proposal_output(payload.proposal),
            "source_refs": _compact_source_refs(list(payload.source_refs)),
        }
    if isinstance(payload, MemoryArchitectureInput):
        return {
            "blueprint": _compact_blueprint_artifact(payload.blueprint),
            "discovery": _compact_discovery_artifact(payload.discovery) if payload.discovery is not None else None,
            "canvas": _compact_canvas_artifact(payload.canvas) if payload.canvas is not None else None,
            "approved_tool_names": _compact_string_list(list(payload.approved_tool_names), limit=24, item_limit=120),
            "design_memory_implications": _compact_string_list(
                list(payload.design_memory_implications),
                limit=16,
                item_limit=220,
            ),
            "tools_capability_resolutions": _compact_string_list(
                list(payload.tools_capability_resolutions),
                limit=24,
                item_limit=220,
            ),
            "source_refs": _compact_source_refs(list(payload.source_refs)),
        }
    if isinstance(payload, MemoryArchitectureCritiqueInput):
        return {
            "blueprint": _compact_blueprint_artifact(payload.blueprint),
            "proposal": _compact_memory_architecture_recommendation_output(payload.proposal),
            "approved_tool_names": _compact_string_list(list(payload.approved_tool_names), limit=24, item_limit=120),
            "design_memory_implications": _compact_string_list(
                list(payload.design_memory_implications),
                limit=16,
                item_limit=220,
            ),
            "tools_capability_resolutions": _compact_string_list(
                list(payload.tools_capability_resolutions),
                limit=24,
                item_limit=220,
            ),
            "source_refs": _compact_source_refs(list(payload.source_refs)),
        }
    if isinstance(payload, ValidationScenarioGenerationInput):
        return {
            "blueprint": _compact_blueprint_artifact(payload.blueprint),
            "discovery": _compact_discovery_artifact(payload.discovery) if payload.discovery is not None else None,
            "canvas": _compact_canvas_artifact(payload.canvas) if payload.canvas is not None else None,
            "focus_areas": _compact_string_list(list(payload.focus_areas), limit=8, item_limit=140),
            "source_refs": _compact_source_refs(list(payload.source_refs)),
        }
    if isinstance(payload, ValidationScenarioSimulationInput):
        return {
            "blueprint": _compact_blueprint_artifact(payload.blueprint),
            "scenario": _compact_validation_scenario_item(payload.scenario),
            "source_refs": _compact_source_refs(list(payload.source_refs)),
        }
    if isinstance(payload, ValidationRunJudgmentInput):
        return {
            "simulation": _compact_validation_simulation_output(payload.simulation),
            "blueprint": _compact_blueprint_artifact(payload.blueprint) if payload.blueprint is not None else None,
            "source_refs": _compact_source_refs(list(payload.source_refs)),
        }
    if isinstance(payload, EstimationRiskAnalysisInput):
        return {
            "blueprint": _compact_blueprint_artifact(payload.blueprint) if payload.blueprint is not None else None,
            "estimation_report": _compact_estimation_report_artifact(payload.estimation_report),
            "pricing_summary": _compact_string_list(list(payload.pricing_summary), limit=8, item_limit=160),
            "validation_summary": _compact_string_list(list(payload.validation_summary), limit=8, item_limit=160),
            "workspace_calibration_summary": _compact_string_list(
                list(payload.workspace_calibration_summary),
                limit=8,
                item_limit=160,
            ),
            "benchmark_hints": _compact_string_list(list(payload.benchmark_hints), limit=8, item_limit=160),
            "source_refs": _compact_source_refs(list(payload.source_refs)),
        }
    if isinstance(payload, DiagramGenerationInput):
        serialized = payload.model_dump(mode="json")
        source_context = serialized.get("source_context")
        compact_context: dict[str, Any] = {}
        if isinstance(source_context, dict):
            for field_name in (
                "project",
                "coverage_summary",
                "resolved_inputs",
                "missing_required_inputs",
                "approved_artifact_keys",
            ):
                value = source_context.get(field_name)
                if value not in (None, "", [], {}):
                    compact_context[field_name] = value
            approved_artifacts = source_context.get("approved_artifacts")
            if isinstance(approved_artifacts, list):
                compact_context["approved_artifact_count"] = len(approved_artifacts)
                compact_context["approved_artifact_summaries"] = [
                    _compact_diagram_approved_artifact(item)
                    for item in approved_artifacts[:6]
                ]
                compact_context["omitted_approved_artifact_count"] = max(0, len(approved_artifacts) - 6)
        if compact_context:
            serialized["source_context"] = compact_context
        serialized["context_brief"] = _compact_text(payload.context_brief, limit=520)
        serialized["source_refs"] = _compact_source_refs(list(payload.source_refs), limit=16)
        serialized["required_inputs"] = _compact_string_list(list(payload.required_inputs), limit=12, item_limit=120)
        serialized["allowed_elements"] = _compact_string_list(list(payload.allowed_elements), limit=12, item_limit=80)
        serialized["allowed_relationships"] = _compact_string_list(
            list(payload.allowed_relationships),
            limit=12,
            item_limit=100,
        )
        serialized["forbidden_mixes"] = _compact_string_list(list(payload.forbidden_mixes), limit=8, item_limit=120)
        serialized["inherits_from"] = _compact_string_list(list(payload.inherits_from), limit=8, item_limit=120)
        serialized["transform_rules"] = _compact_string_list(list(payload.transform_rules), limit=8, item_limit=160)
        serialized["semantic_rules"] = _compact_string_list(list(payload.semantic_rules), limit=10, item_limit=160)
        serialized["exclusions"] = _compact_string_list(list(payload.exclusions), limit=8, item_limit=140)
        serialized["missing_required_inputs"] = _compact_string_list(
            list(payload.missing_required_inputs),
            limit=10,
            item_limit=120,
        )
        serialized["resolved_inputs"] = [
            _compact_diagram_resolved_input(item)
            for item in payload.resolved_inputs[:10]
            if isinstance(item, dict)
        ]
        serialized["omitted_resolved_input_count"] = max(0, len(payload.resolved_inputs) - len(serialized["resolved_inputs"]))
        return serialized
    return payload.model_dump(mode="json")


def _default_runtime_settings() -> dict[str, Any]:
    settings = get_settings()
    return {
        "active_provider": (
            settings.llm_provider if settings.llm_provider in {"openai", "deepseek", "codex_local"} else "openai"
        ),
        "agent_execution_backend": _normalize_agent_execution_backend(
            settings.agent_execution_backend,
            "provider_native",
        ),
        "knowledge_access_backend": _normalize_knowledge_access_backend(
            settings.knowledge_access_backend,
            "inline_context",
        ),
        "uses_platform_credentials": True,
        "openai": {
            "fast_model": settings.openai_model_fast,
            "reasoning_model": settings.openai_model_reasoning,
            "reasoning_effort": settings.openai_reasoning_effort,
        },
        "deepseek": {
            "base_url": settings.deepseek_base_url,
            "fast_model": settings.deepseek_model_fast,
            "reasoning_model": settings.deepseek_model_reasoning,
            "reasoning_effort": settings.deepseek_reasoning_effort,
        },
        "codex_local": {
            "command": _clean_text(settings.codex_executable, settings.codex_cli_command),
            "model": _clean_text(settings.codex_exec_model, settings.codex_model),
            "profile": settings.codex_profile,
            "cost_policy": "hybrid",
            "timeout_ms": _normalize_positive_int(settings.codex_exec_timeout_ms, 150000, minimum=1000),
            "max_concurrency": _normalize_positive_int(settings.codex_exec_max_concurrency, 1, minimum=1),
            "runner_id": _clean_text(settings.codex_runner_id, "local"),
            "auth_mode": _normalize_codex_auth_mode(settings.codex_auth_mode, "auto"),
            "fallback_models": _normalize_string_list(settings.codex_exec_fallback_models),
            "primary_agents": _normalize_string_list(settings.codex_primary_agents),
            "shadow_agents": _normalize_string_list(settings.codex_shadow_agents),
            "staged_agents": _normalize_string_list(settings.codex_staged_agents),
        },
        "antigravity": {
            "executable": "agy",
            "model": "gemini-3.6-flash",
            "effort": "high",
            "timeout_ms": 1200000,
            "max_concurrency": 1,
            "runner_id": "local-antigravity-cli",
            "auth_mode": "auto",
            "fallback_models": [],
            "primary_agents": [],
            "shadow_agents": [],
            "staged_agents": [],
        },
        "compatibility_mode": "backward_compatible",
        "updated_at": utc_now().isoformat(),
    }


def _merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_provider_options(payload: LLMRuntimeSettings) -> list[LLMProviderOption]:
    openai_option = LLMProviderOption(
        key=LLMProviderKey.openai.value,
        label="OpenAI",
        description="Responses API con structured outputs y compatibilidad total con el flujo actual.",
        configured=payload.openai.api_key_configured,
        reachable=payload.openai.available,
        selected=payload.active_provider == LLMProviderKey.openai,
        supports_structured_output=True,
        metadata={
            "fast_model": payload.openai.fast_model,
            "reasoning_model": payload.openai.reasoning_model,
            "reasoning_effort": payload.openai.reasoning_effort,
            "secret_source": payload.openai.secret_source,
            "last_rotated_at": payload.openai.last_rotated_at.isoformat() if payload.openai.last_rotated_at else "",
            "health_status": payload.openai.health_status,
            "status_note": payload.openai.status_note,
        },
    )
    deepseek_option = LLMProviderOption(
        key=LLMProviderKey.deepseek.value,
        label="DeepSeek",
        description="API compatible con OpenAI via chat.completions y JSON Output para structured artifacts.",
        configured=payload.deepseek.api_key_configured,
        reachable=payload.deepseek.available,
        selected=payload.active_provider == LLMProviderKey.deepseek,
        supports_structured_output=True,
        metadata={
            "base_url": payload.deepseek.base_url,
            "fast_model": payload.deepseek.fast_model,
            "reasoning_model": payload.deepseek.reasoning_model,
            "reasoning_effort": payload.deepseek.reasoning_effort,
            "secret_source": payload.deepseek.secret_source,
            "last_rotated_at": payload.deepseek.last_rotated_at.isoformat() if payload.deepseek.last_rotated_at else "",
            "health_status": payload.deepseek.health_status,
            "status_note": payload.deepseek.status_note,
        },
    )
    codex_option = LLMProviderOption(
        key=LLMProviderKey.codex_local.value,
        label="Codex local",
        description="Ejecucion local mediante codex exec reutilizando autenticacion/suscripcion configurada en la maquina.",
        configured=bool(payload.codex_local.command and payload.codex_local.model),
        reachable=payload.codex_local.available,
        selected=payload.active_provider == LLMProviderKey.codex_local,
        supports_structured_output=True,
        metadata={
            "command": payload.codex_local.command,
            "model": payload.codex_local.model,
            "profile": payload.codex_local.profile,
            "cost_policy": payload.codex_local.cost_policy,
            "timeout_ms": payload.codex_local.timeout_ms,
            "max_concurrency": payload.codex_local.max_concurrency,
            "runner_id": payload.codex_local.runner_id,
            "auth_mode": payload.codex_local.auth_mode.value,
            "fallback_models": payload.codex_local.fallback_models,
            "primary_agents": payload.codex_local.primary_agents,
            "shadow_agents": payload.codex_local.shadow_agents,
            "staged_agents": payload.codex_local.staged_agents,
            "secret_source": payload.codex_local.secret_source,
            "last_rotated_at": payload.codex_local.last_rotated_at.isoformat() if payload.codex_local.last_rotated_at else "",
            "health_status": payload.codex_local.health_status,
            "agent_execution_backend": payload.agent_execution_backend.value,
            "knowledge_access_backend": payload.knowledge_access_backend.value,
            "uses_platform_credentials": payload.uses_platform_credentials,
            "status_note": payload.codex_local.status_note,
        },
    )
    agy_config = getattr(payload, "antigravity_cli", getattr(payload, "antigravity", None))
    antigravity_option = LLMProviderOption(
        key=LLMProviderKey.antigravity_cli.value,
        label="Antigravity CLI",
        description="Ejecucion local mediante agy CLI reutilizando autenticacion configurada en la maquina.",
        configured=bool(agy_config.executable and agy_config.model) if agy_config else False,
        reachable=agy_config.available if agy_config else False,
        selected=payload.active_provider == LLMProviderKey.antigravity_cli,
        supports_structured_output=True,
        metadata={
            "executable": agy_config.executable if agy_config else "agy",
            "model": agy_config.model if agy_config else "gemini-3.6-flash",
            "effort": agy_config.effort if agy_config else "high",
            "timeout_ms": agy_config.timeout_ms if agy_config else 1200000,
            "max_concurrency": agy_config.max_concurrency if agy_config else 1,
            "runner_id": agy_config.runner_id if agy_config else "local-antigravity-cli",
            "auth_mode": agy_config.auth_mode if agy_config else "auto",
            "fallback_models": agy_config.fallback_models if agy_config else [],
            "primary_agents": agy_config.primary_agents if agy_config else [],
            "shadow_agents": agy_config.shadow_agents if agy_config else [],
            "staged_agents": agy_config.staged_agents if agy_config else [],
            "secret_source": agy_config.secret_source if agy_config else "local_runtime",
            "last_rotated_at": (agy_config.last_rotated_at.isoformat() if agy_config.last_rotated_at else "") if agy_config else "",
            "health_status": agy_config.health_status if agy_config else "local_runtime_missing",
            "agent_execution_backend": payload.agent_execution_backend.value,
            "knowledge_access_backend": payload.knowledge_access_backend.value,
            "uses_platform_credentials": payload.uses_platform_credentials,
            "status_note": agy_config.status_note if agy_config else "",
        },
    )
    return [openai_option, deepseek_option, codex_option, antigravity_option]


def resolve_runtime_settings_payload(payload: dict[str, Any]) -> LLMRuntimeSettings:
    settings = get_settings()
    resolved = LLMRuntimeSettings.model_validate(payload)
    openai_available = bool(OpenAI is not None and settings.openai_api_key)
    openai_note = (
        "API key y SDK disponibles."
        if openai_available
        else "Falta OPENAI_API_KEY o el SDK no esta instalado en el backend."
    )
    deepseek_available = bool(OpenAI is not None and settings.deepseek_api_key)
    deepseek_note = (
        "API key y SDK disponibles via chat.completions."
        if deepseek_available
        else "Falta DEEPSEEK_API_KEY o el SDK no esta instalado en el backend."
    )
    deepseek_base_url = _clean_url(resolved.deepseek.base_url, settings.deepseek_base_url)
    deepseek_fast_model = _clean_text(resolved.deepseek.fast_model, settings.deepseek_model_fast)
    deepseek_reasoning_model = _clean_text(
        resolved.deepseek.reasoning_model,
        settings.deepseek_model_reasoning,
    )
    deepseek_reasoning_effort = _normalize_deepseek_reasoning_effort(
        resolved.deepseek.reasoning_effort,
        settings.deepseek_reasoning_effort,
    )
    codex_command = _clean_text(resolved.codex_local.command, settings.codex_cli_command)
    codex_model = _clean_text(
        resolved.codex_local.model,
        resolved.openai.reasoning_model
        or settings.codex_exec_model
        or settings.codex_model
        or settings.openai_model_reasoning,
    )
    codex_profile = resolved.codex_local.profile.strip() or settings.codex_profile.strip()
    codex_timeout_ms = _normalize_positive_int(
        resolved.codex_local.timeout_ms,
        settings.codex_exec_timeout_ms,
        minimum=1000,
    )
    codex_max_concurrency = _normalize_positive_int(
        resolved.codex_local.max_concurrency,
        settings.codex_exec_max_concurrency,
        minimum=1,
    )
    codex_runner_id = _clean_text(resolved.codex_local.runner_id, settings.codex_runner_id or "local")
    codex_auth_mode = _normalize_codex_auth_mode(
        resolved.codex_local.auth_mode.value,
        settings.codex_auth_mode,
    )
    codex_binary = resolve_codex_executable_path(codex_command)
    codex_available = codex_binary is not None
    codex_note = (
        "Binario Codex detectado. La autenticacion se reutiliza desde la instalacion local."
        if codex_available
        else "No se encontro el binario de Codex en la ruta configurada."
    )
    agent_execution_backend = _normalize_agent_execution_backend(
        resolved.agent_execution_backend.value,
        settings.agent_execution_backend,
    )
    knowledge_access_backend = _normalize_knowledge_access_backend(
        resolved.knowledge_access_backend.value,
        settings.knowledge_access_backend,
    )

    resolved = resolved.model_copy(
        update={
            "agent_execution_backend": AgentExecutionBackend(agent_execution_backend),
            "knowledge_access_backend": KnowledgeAccessBackend(knowledge_access_backend),
            "openai": OpenAIProviderConfig(
                fast_model=_clean_text(resolved.openai.fast_model, settings.openai_model_fast),
                reasoning_model=_clean_text(resolved.openai.reasoning_model, settings.openai_model_reasoning),
                reasoning_effort=_clean_text(resolved.openai.reasoning_effort, settings.openai_reasoning_effort),
                api_key_configured=bool(settings.openai_api_key),
                available=openai_available,
                secret_source=resolved.openai.secret_source,
                last_rotated_at=resolved.openai.last_rotated_at,
                health_status=resolved.openai.health_status,
                status_note=openai_note,
            ),
            "deepseek": DeepSeekProviderConfig(
                base_url=deepseek_base_url,
                fast_model=deepseek_fast_model,
                reasoning_model=deepseek_reasoning_model,
                reasoning_effort=deepseek_reasoning_effort,
                api_key_configured=bool(settings.deepseek_api_key),
                available=deepseek_available,
                secret_source=resolved.deepseek.secret_source,
                last_rotated_at=resolved.deepseek.last_rotated_at,
                health_status=resolved.deepseek.health_status,
                status_note=deepseek_note,
            ),
            "codex_local": CodexLocalProviderConfig(
                command=codex_command,
                model=codex_model,
                profile=codex_profile,
                cost_policy=resolved.codex_local.cost_policy,
                timeout_ms=codex_timeout_ms,
                max_concurrency=codex_max_concurrency,
                runner_id=codex_runner_id,
                auth_mode=CodexAuthMode(codex_auth_mode),
                fallback_models=_normalize_string_list(resolved.codex_local.fallback_models),
                primary_agents=_normalize_string_list(resolved.codex_local.primary_agents),
                shadow_agents=_normalize_string_list(resolved.codex_local.shadow_agents),
                staged_agents=_normalize_string_list(resolved.codex_local.staged_agents),
                available=codex_available,
                executable_found=codex_available,
                secret_source=resolved.codex_local.secret_source,
                last_rotated_at=resolved.codex_local.last_rotated_at,
                health_status=resolved.codex_local.health_status,
                status_note=codex_note,
            ),
            "antigravity_cli": AntigravityProviderConfig(
                executable=(getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).executable or "agy",
                model=(getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).model or "gemini-3.6-flash",
                effort=(getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).effort or "high",
                timeout_ms=(getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).timeout_ms or 1200000,
                max_concurrency=(getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).max_concurrency or 1,
                runner_id=(getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).runner_id or "local-antigravity-cli",
                auth_mode=(getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).auth_mode or "auto",
                fallback_models=_normalize_string_list((getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).fallback_models),
                primary_agents=_normalize_string_list((getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).primary_agents),
                shadow_agents=_normalize_string_list((getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).shadow_agents),
                staged_agents=_normalize_string_list((getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).staged_agents),
                available=resolve_agy_executable((getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).executable) is not None,
                executable_found=resolve_agy_executable((getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).executable) is not None,
                secret_source=(getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).secret_source,
                last_rotated_at=(getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).last_rotated_at,
                health_status="healthy" if resolve_agy_executable((getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).executable) is not None else "local_runtime_missing",
                status_note="Binario agy detectado." if resolve_agy_executable((getattr(resolved, "antigravity_cli", getattr(resolved, "antigravity", None)) or AntigravityProviderConfig()).executable) is not None else "No se encontro el binario agy en la ruta configurada o PATH.",
            ),
        }
    )
    return resolved.model_copy(update={"provider_options": build_provider_options(resolved)})


def load_llm_runtime_settings() -> LLMRuntimeSettings:
    defaults = _default_runtime_settings()
    path = _runtime_config_path()
    payload: dict[str, Any] = defaults
    if runtime_legacy_file_fallback_enabled() and path.exists():
        try:
            persisted = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(persisted, dict):
                payload = _merge_dicts(defaults, persisted)
        except json.JSONDecodeError:
            payload = defaults
    return resolve_runtime_settings_payload(payload)


def load_default_builder_runtime_settings() -> LLMRuntimeSettings:
    return load_llm_runtime_settings()


def persist_llm_runtime_settings(payload: LLMRuntimeSettingsUpdateRequest) -> LLMRuntimeSettings:
    defaults = _default_runtime_settings()
    settings = get_settings()
    persisted_payload = {
        "active_provider": payload.active_provider.value,
        "agent_execution_backend": _normalize_agent_execution_backend(
            payload.agent_execution_backend.value,
            settings.agent_execution_backend,
        ),
        "knowledge_access_backend": _normalize_knowledge_access_backend(
            payload.knowledge_access_backend.value,
            settings.knowledge_access_backend,
        ),
        "uses_platform_credentials": True if payload.uses_platform_credentials is None else payload.uses_platform_credentials,
        "openai": OpenAIProviderConfigUpdate(
            fast_model=_clean_text(payload.openai.fast_model, settings.openai_model_fast),
            reasoning_model=_clean_text(payload.openai.reasoning_model, settings.openai_model_reasoning),
            reasoning_effort=_clean_text(payload.openai.reasoning_effort, settings.openai_reasoning_effort),
        ).model_dump(mode="json"),
        "deepseek": DeepSeekProviderConfigUpdate(
            base_url=_clean_url(payload.deepseek.base_url, settings.deepseek_base_url),
            fast_model=_clean_text(payload.deepseek.fast_model, settings.deepseek_model_fast),
            reasoning_model=_clean_text(payload.deepseek.reasoning_model, settings.deepseek_model_reasoning),
            reasoning_effort=_normalize_deepseek_reasoning_effort(
                payload.deepseek.reasoning_effort,
                settings.deepseek_reasoning_effort,
            ),
        ).model_dump(mode="json"),
        "codex_local": CodexLocalProviderConfigUpdate(
            command=_clean_text(
                payload.codex_local.command,
                settings.codex_executable or settings.codex_cli_command,
            ),
            model=_clean_text(
                payload.codex_local.model,
                settings.codex_exec_model or settings.codex_model or settings.openai_model_reasoning,
            ),
            profile=payload.codex_local.profile.strip(),
            cost_policy=payload.codex_local.cost_policy,
            timeout_ms=_normalize_positive_int(payload.codex_local.timeout_ms, settings.codex_exec_timeout_ms, minimum=1000),
            max_concurrency=_normalize_positive_int(
                payload.codex_local.max_concurrency,
                settings.codex_exec_max_concurrency,
                minimum=1,
            ),
            runner_id=_clean_text(payload.codex_local.runner_id, settings.codex_runner_id or "local"),
            auth_mode=_normalize_codex_auth_mode(payload.codex_local.auth_mode.value, settings.codex_auth_mode),
            fallback_models=_normalize_string_list(payload.codex_local.fallback_models),
            primary_agents=_normalize_string_list(payload.codex_local.primary_agents),
            shadow_agents=_normalize_string_list(payload.codex_local.shadow_agents),
            staged_agents=_normalize_string_list(payload.codex_local.staged_agents),
        ).model_dump(mode="json"),
        "compatibility_mode": defaults["compatibility_mode"],
        "updated_at": utc_now().isoformat(),
    }
    _runtime_config_path().write_text(json.dumps(persisted_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return load_llm_runtime_settings()


def build_builder_service(runtime_settings: LLMRuntimeSettings | None = None) -> BuilderProviderFacade:
    resolved_runtime_settings = runtime_settings or load_llm_runtime_settings()
    finops_session_factory = default_finops_session_factory
    finops_ledger_service = LLMUsageLedgerService()
    return BuilderProviderFacade(
        resolved_runtime_settings,
        openai_service=OpenAIBuilderService(
            resolved_runtime_settings,
            finops_session_factory=finops_session_factory,
            finops_ledger_service=finops_ledger_service,
        ),
        deepseek_service=DeepSeekBuilderService(
            resolved_runtime_settings,
            finops_session_factory=finops_session_factory,
            finops_ledger_service=finops_ledger_service,
        ),
        codex_service=CodexLocalBuilderService(
            resolved_runtime_settings,
            finops_session_factory=finops_session_factory,
            finops_ledger_service=finops_ledger_service,
        ),
        antigravity_service=AntigravityLocalBuilderService(
            resolved_runtime_settings,
            finops_session_factory=finops_session_factory,
            finops_ledger_service=finops_ledger_service,
        ),
    )


def _resolve_effective_provider_api_key(session, *, workspace_id, provider_key: LLMProviderKey) -> str | None:
    from app.services.llm_runtime.runtime_secrets_service import resolve_effective_provider_secret_value

    return resolve_effective_provider_secret_value(
        session,
        workspace_id,
        provider_key,
    )


class OpenAIBuilderService(_APIContextAwareBuilderMixin):
    def __init__(
        self,
        runtime_settings: LLMRuntimeSettings | None = None,
        *,
        finops_session_factory: FinOpsSessionFactory | None = None,
        finops_ledger_service: LLMUsageLedgerService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.runtime_settings = runtime_settings or load_llm_runtime_settings()
        self._context_adapter = APIProviderContextAdapter()
        self._finops_session_factory = finops_session_factory
        self._finops_ledger_service = finops_ledger_service
        self._workspace_client_error = ""
        self._client = None
        if OpenAI is not None and self.settings.openai_api_key:
            self._client = OpenAI(api_key=self.settings.openai_api_key)

    def _provider_api_key_configured(self) -> bool:
        return bool(self.settings.openai_api_key) or bool(self.runtime_settings.openai.api_key_configured)

    def _ensure_workspace_client(self, workspace_id) -> None:
        if self._client is not None or OpenAI is None or workspace_id is None:
            return
        if self._finops_session_factory is None:
            return
        self._workspace_client_error = ""
        try:
            with self._finops_session_factory() as session:
                api_key = _resolve_effective_provider_api_key(
                    session,
                    workspace_id=workspace_id,
                    provider_key=LLMProviderKey.openai,
                )
        except Exception as exc:
            self._workspace_client_error = _format_provider_bootstrap_error(
                str(exc),
                fallback="No se pudo resolver la credencial OpenAI del workspace.",
            )
            LOGGER.warning(
                "OpenAI workspace client bootstrap failed for workspace %s: %s",
                workspace_id,
                self._workspace_client_error,
            )
            return
        if api_key:
            self._client = OpenAI(api_key=api_key)
            self._workspace_client_error = ""
            return
        self._workspace_client_error = "No se resolvio una API key de OpenAI para el runtime efectivo."

    def _provider_unavailable_detail(self) -> str | None:
        detail = self._workspace_client_error.strip()
        return detail or None

    def can_attempt(self) -> bool:
        return (
            self.runtime_settings.active_provider == LLMProviderKey.openai
            and self.settings.llm_mode in {"openai", "hybrid"}
        )

    def is_available(self) -> bool:
        return self.can_attempt() and self._client is not None

    def provider_summary(self) -> dict[str, str | bool]:
        configured = self._provider_api_key_configured()
        return {
            "provider": self.runtime_settings.active_provider.value,
            "mode": "responses",
            "configured": configured,
            "sdk_ready": self._client is not None or (OpenAI is not None and configured),
            "fast_model": self.runtime_settings.openai.fast_model,
            "reasoning_model": self.runtime_settings.openai.reasoning_model,
            "status_note": self.runtime_settings.openai.status_note,
        }

    def _capability_source(self, spec: BuilderCapabilitySpec, payload: BaseModel) -> CodexContextInlineSource:
        return _capability_source_for_api(spec, payload)

    def _execute_structured_capability(
        self,
        *,
        capability: BuilderCapability,
        payload: BaseModel,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        spec = get_builder_capability_spec(capability)
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind=f"openai_{spec.task_kind}",
            task_instruction=spec.task_instruction,
            inline_sources=[self._capability_source(spec, payload)],
            context_bundle=context_bundle,
        )
        model_name = (
            self.runtime_settings.openai.reasoning_model
            if spec.preferred_model == "reasoning"
            else self.runtime_settings.openai.fast_model
        )
        call_context = build_llm_call_context(
            context_bundle,
            capability=capability.value,
            provider_key=LLMProviderKey.openai.value,
            execution_backend=AgentExecutionBackend.provider_native.value,
            metadata={
                "prompt_version": spec.prompt_version,
                "preferred_model": spec.preferred_model,
                "task_kind": spec.task_kind,
            },
        )
        base_result = LLMArtifactResult(
            artifact=None,
            provider_key=LLMProviderKey.openai.value,
            execution_backend=AgentExecutionBackend.provider_native.value,
            execution_mode=call_context.execution_mode,
            capability_key=capability.value,
            model_name=model_name,
            prompt_version=spec.prompt_version,
            schema_validation_status="not_attempted",
            finops_context=call_context,
            capability_policy=_capability_policy_payload(spec),
        )

        def _call_openai() -> LLMArtifactResult:
            self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
            if not self.is_available():
                detail = self._provider_unavailable_detail()
                return self._attach_context_metadata(
                    replace(
                        base_result,
                        warning=_format_provider_unavailable_warning(
                            provider_label="OpenAI",
                            capability=capability,
                            spec=spec,
                            detail=detail,
                        ),
                        finish_reason="provider_unavailable",
                        failure_kind="provider_unavailable",
                        failure_detail=detail,
                        degraded=True,
                    ),
                    context_envelope=context_envelope,
                )

            request_kwargs: dict[str, Any] = {
                "model": model_name,
                "input": [
                    {"role": "system", "content": _localized_instruction(spec.system_instruction, context_bundle)},
                    {"role": "user", "content": context_envelope.user_payload},
                ],
                "max_output_tokens": _structured_capability_max_tokens(capability, payload=payload),
                "text_format": spec.output_model,
                "timeout": max(1.0, spec.timeout_ms / 1000),
            }
            if spec.preferred_model == "reasoning":
                request_kwargs["reasoning"] = {"effort": self.runtime_settings.openai.reasoning_effort}

            try:
                response = self._client.responses.parse(**request_kwargs)
                parsed = response.output_parsed
                usage = normalize_openai_usage(getattr(response, "usage", None))
                if parsed is None:
                    return self._attach_context_metadata(
                        replace(
                            base_result,
                            warning=f"OpenAI no devolvio salida estructurada para {capability.value}; policy={spec.fallback_policy}.",
                            request_id=str(getattr(response, "id", "") or ""),
                            finish_reason=str(getattr(response, "status", "missing_output") or "missing_output"),
                            schema_validation_status="missing_output",
                            failure_kind="schema_missing_output",
                            degraded=True,
                            token_usage=usage.compatibility_token_usage(),
                            normalized_usage=usage,
                        ),
                        context_envelope=context_envelope,
                    )
                normalized = spec.output_model.model_validate(parsed.model_dump(mode="json"))
                schema_status = "valid"
                if capability == BuilderCapability.generate_diagram_model:
                    normalized, schema_status = finalize_structured_diagram_artifact(
                        normalized,
                        schema_status=schema_status,
                    )
                return self._attach_context_metadata(
                    replace(
                        base_result,
                        artifact=normalized,
                        request_id=str(getattr(response, "id", "") or ""),
                        finish_reason=str(getattr(response, "status", "completed") or "completed"),
                        schema_validation_status=schema_status,
                        token_usage=usage.compatibility_token_usage(),
                        normalized_usage=usage,
                    ),
                    context_envelope=context_envelope,
                )
            except Exception as exc:
                return self._attach_context_metadata(
                    replace(
                        base_result,
                        warning=f"OpenAI no pudo ejecutar {capability.value}; policy={spec.fallback_policy}.",
                        finish_reason="exception",
                        schema_validation_status="invalid",
                        failure_kind="provider_error",
                        failure_detail=str(exc)[:400],
                        degraded=True,
                    ),
                    context_envelope=context_envelope,
                )

        return record_provider_call(
            call=_call_openai,
            call_context=call_context,
            provider_key=LLMProviderKey.openai,
            model_name=model_name,
            requested_model=model_name,
            execution_backend=AgentExecutionBackend.provider_native.value,
            execution_mode=call_context.execution_mode,
            ledger_service=self._finops_ledger_service,
            session_factory=self._finops_session_factory,
            metadata={
                "capability": capability.value,
                "prompt_version": spec.prompt_version,
                "preferred_model": spec.preferred_model,
            },
        )

    def normalize_discovery(
        self,
        payload: DiscoveryInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="openai_discovery_normalization",
            task_instruction=(
                "Normaliza la captura compactada `discovery_capture` a un discovery estructurado para un builder Lean "
                "de agentes. Usa solo hechos presentes en las fuentes compactadas. Si un dato no esta claro, usa "
                "'unknown'. Para case_type usa solo: informacion, automatizacion, copiloto, operador_autonomo, "
                "sistema_multiagente. Para autonomy_level usa solo low, medium o high."
            ),
            inline_sources=[
                CodexContextInlineSource(
                    key="discovery_capture",
                    title="Discovery capture",
                    content=json.dumps(_compact_discovery_input(payload), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Captura cruda de discovery para normalizacion estructurada por provider API.",
                )
            ],
            context_bundle=context_bundle,
        )
        self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
        if not self.is_available():
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="OpenAI no esta disponible para recomendar tools minimas; se mantiene el preflight heuristico.",
                    provider_key=LLMProviderKey.openai.value,
                ),
                context_envelope=context_envelope,
            )
        try:
            response = self._client.responses.parse(
                model=self.runtime_settings.openai.fast_model,
                input=[
                    {
                        "role": "system",
                        "content": _localized_instruction(
                            (
                                "Convierte la captura en un discovery estructurado para un builder Lean de agentes. "
                                "Usa solo hechos presentes en la entrada. Si un dato no esta claro, usa 'unknown'. "
                                "Para case_type usa solo uno de estos valores: informacion, automatizacion, "
                                "copiloto, operador_autonomo, sistema_multiagente. "
                                "Para autonomy_level usa solo low, medium o high."
                            ),
                            context_bundle,
                        ),
                    },
                    {
                        "role": "user",
                        "content": context_envelope.user_payload,
                    },
                ],
                text_format=DiscoveryArtifact,
            )
            parsed = response.output_parsed
            if parsed is None:
                return self._attach_context_metadata(
                    LLMArtifactResult(
                        artifact=None,
                        warning="OpenAI no devolvio un discovery estructurado; se uso fallback deterministico.",
                        provider_key=LLMProviderKey.openai.value,
                    ),
                    context_envelope=context_envelope,
                )
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=sanitize_discovery(parsed),
                    provider_key=LLMProviderKey.openai.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="OpenAI no pudo normalizar discovery; se uso fallback deterministico.",
                    provider_key=LLMProviderKey.openai.value,
                ),
                context_envelope=context_envelope,
            )

    def build_canvas(
        self,
        discovery: DiscoveryArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="openai_canvas_generation",
            task_instruction=(
                "Genera un canvas Lean para un agente usando solo la fuente compactada `normalized_discovery`. "
                "Manten el alcance corto, concreto y util para un MVP. Si algo no esta claro, usa 'unknown'."
            ),
            inline_sources=[
                CodexContextInlineSource(
                    key="normalized_discovery",
                    title="Normalized discovery",
                    content=json.dumps(_compact_discovery_artifact(discovery), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Discovery estructurado aprobado para construir el canvas via provider API.",
                )
            ],
            context_bundle=context_bundle,
        )
        self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
        if not self.is_available():
            return self._attach_context_metadata(
                LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.openai.value),
                context_envelope=context_envelope,
            )
        try:
            response = self._client.responses.parse(
                model=self.runtime_settings.openai.fast_model,
                input=[
                    {
                        "role": "system",
                        "content": _localized_instruction(
                            (
                                "Genera un canvas Lean para un agente. Usa solo el discovery recibido. "
                                "Manten el alcance corto, concreto y util para un MVP. Si algo no esta claro, usa 'unknown'."
                            ),
                            context_bundle,
                        ),
                    },
                    {
                        "role": "user",
                        "content": context_envelope.user_payload,
                    },
                ],
                text_format=CanvasArtifact,
            )
            parsed = response.output_parsed
            if parsed is None:
                return self._attach_context_metadata(
                    LLMArtifactResult(
                        artifact=None,
                        warning="OpenAI no devolvio un canvas estructurado; se uso fallback deterministico.",
                        provider_key=LLMProviderKey.openai.value,
                    ),
                    context_envelope=context_envelope,
                )
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=sanitize_canvas(parsed),
                    provider_key=LLMProviderKey.openai.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="OpenAI no pudo construir el canvas; se uso fallback deterministico.",
                    provider_key=LLMProviderKey.openai.value,
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
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="openai_blueprint_narrative",
            task_instruction=(
                "Redacta la narrativa tecnica de un blueprint Lean para un agente usando solo las fuentes compactadas "
                "`narrative_discovery`, `narrative_canvas` y `narrative_blueprint`. No cambies la arquitectura, "
                "memoria, tools ni guardrails ya definidos. Explica por que la recomendacion encaja con el discovery "
                "y el canvas, y resalta tradeoffs relevantes sin inventar nuevos componentes."
            ),
            inline_sources=[
                CodexContextInlineSource(
                    key="narrative_discovery",
                    title="Discovery for blueprint narrative",
                    content=json.dumps(_compact_discovery_artifact(discovery), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Discovery estructurado aprobado para la narrativa tecnica.",
                ),
                CodexContextInlineSource(
                    key="narrative_canvas",
                    title="Canvas for blueprint narrative",
                    content=json.dumps(_compact_canvas_artifact(canvas), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Canvas Lean aprobado que define alcance y meta del blueprint.",
                ),
                CodexContextInlineSource(
                    key="narrative_blueprint",
                    title="Blueprint for narrative synthesis",
                    content=json.dumps(_compact_blueprint_artifact(blueprint), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Blueprint base cuya narrativa debe sintetizarse sin alterar contratos.",
                ),
            ],
            context_bundle=context_bundle,
        )
        self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
        if not self.is_available():
            return self._attach_context_metadata(
                LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.openai.value),
                context_envelope=context_envelope,
            )
        try:
            response = self._client.responses.parse(
                model=self.runtime_settings.openai.reasoning_model,
                input=[
                    {
                        "role": "system",
                        "content": _localized_instruction(
                            (
                                "Redacta la narrativa tecnica de un blueprint Lean para un agente. "
                                "No cambies la arquitectura, memoria, tools ni guardrails ya definidos. "
                                "Explica por que la recomendacion encaja con el discovery y el canvas, "
                                "y resalta tradeoffs relevantes sin inventar nuevos componentes."
                            ),
                            context_bundle,
                        ),
                    },
                    {
                        "role": "user",
                        "content": context_envelope.user_payload,
                    },
                ],
                reasoning={"effort": self.runtime_settings.openai.reasoning_effort},
                text_format=BlueprintNarrativeOutput,
            )
            parsed = response.output_parsed
            if parsed is None:
                return self._attach_context_metadata(
                    LLMArtifactResult(
                        artifact=None,
                        warning="OpenAI no devolvio narrativa estructurada; se mantuvo la narrativa base.",
                        provider_key=LLMProviderKey.openai.value,
                    ),
                    context_envelope=context_envelope,
                )
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=parsed.model_copy(update={"narrative": normalize_text(parsed.narrative)}),
                    provider_key=LLMProviderKey.openai.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="OpenAI no pudo sintetizar la narrativa; se mantuvo la narrativa base.",
                    provider_key=LLMProviderKey.openai.value,
                ),
                context_envelope=context_envelope,
            )

    def recommend_minimal_tools(
        self,
        prompt_input: ToolRecommendationPromptInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        case_payload = _compact_tool_recommendation_case_payload(prompt_input)
        catalog_payload = _compact_tool_recommendation_catalog_payload(prompt_input)
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="openai_tool_recommendation",
            task_instruction=build_tool_recommendation_context_task_instruction(),
            inline_sources=[
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
            ],
            context_bundle=context_bundle,
        )
        self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
        if not self.is_available():
            return self._attach_context_metadata(
                LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.openai.value),
                context_envelope=context_envelope,
            )
        try:
            response = self._client.responses.parse(
                model=self.runtime_settings.openai.reasoning_model,
                input=[
                    {
                        "role": "system",
                        "content": _localized_instruction(
                            build_tool_recommendation_system_instruction(),
                            context_bundle,
                        ),
                    },
                    {
                        "role": "user",
                        "content": context_envelope.user_payload,
                    },
                ],
                reasoning={"effort": self.runtime_settings.openai.reasoning_effort},
                text_format=ToolRecommendationLLMOutput,
            )
            parsed = response.output_parsed
            if parsed is None:
                return self._attach_context_metadata(
                    LLMArtifactResult(
                        artifact=None,
                        warning="OpenAI no devolvio una recomendacion de tools estructurada; se mantuvo el preflight heuristico.",
                        provider_key=LLMProviderKey.openai.value,
                    ),
                    context_envelope=context_envelope,
                )
            normalized = ToolRecommendationLLMOutput.model_validate(parsed.model_dump(mode="json"))
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=normalized,
                    provider_key=LLMProviderKey.openai.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="OpenAI no pudo recomendar tools minimas; se mantuvo el preflight heuristico.",
                    provider_key=LLMProviderKey.openai.value,
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


class DeepSeekBuilderService(_APIContextAwareBuilderMixin):
    def __init__(
        self,
        runtime_settings: LLMRuntimeSettings | None = None,
        *,
        finops_session_factory: FinOpsSessionFactory | None = None,
        finops_ledger_service: LLMUsageLedgerService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.runtime_settings = runtime_settings or load_llm_runtime_settings()
        self._context_adapter = APIProviderContextAdapter()
        self._finops_session_factory = finops_session_factory
        self._finops_ledger_service = finops_ledger_service
        self._workspace_client_error = ""
        self._client = None
        self._last_completion_metadata: dict[str, Any] = {}
        if OpenAI is not None and self.settings.deepseek_api_key:
            self._client = OpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.runtime_settings.deepseek.base_url,
            )

    def _provider_api_key_configured(self) -> bool:
        return bool(self.settings.deepseek_api_key) or bool(self.runtime_settings.deepseek.api_key_configured)

    def _ensure_workspace_client(self, workspace_id) -> None:
        if self._client is not None or OpenAI is None or workspace_id is None:
            return
        if self._finops_session_factory is None:
            return
        self._workspace_client_error = ""
        try:
            with self._finops_session_factory() as session:
                api_key = _resolve_effective_provider_api_key(
                    session,
                    workspace_id=workspace_id,
                    provider_key=LLMProviderKey.deepseek,
                )
        except Exception as exc:
            self._workspace_client_error = _format_provider_bootstrap_error(
                str(exc),
                fallback="No se pudo resolver la credencial DeepSeek del workspace.",
            )
            LOGGER.warning(
                "DeepSeek workspace client bootstrap failed for workspace %s: %s",
                workspace_id,
                self._workspace_client_error,
            )
            return
        if api_key:
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.runtime_settings.deepseek.base_url,
            )
            self._workspace_client_error = ""
            return
        self._workspace_client_error = "No se resolvio una API key de DeepSeek para el runtime efectivo."

    def _provider_unavailable_detail(self) -> str | None:
        detail = self._workspace_client_error.strip()
        return detail or None

    def can_attempt(self) -> bool:
        return (
            self.runtime_settings.active_provider == LLMProviderKey.deepseek
            and self.settings.llm_mode in {"openai", "hybrid"}
        )

    def is_available(self) -> bool:
        return self.can_attempt() and self._client is not None

    def provider_summary(self) -> dict[str, str | bool]:
        configured = self._provider_api_key_configured()
        return {
            "provider": self.runtime_settings.active_provider.value,
            "mode": "chat_completions",
            "configured": configured,
            "sdk_ready": self._client is not None or (OpenAI is not None and configured),
            "fast_model": self.runtime_settings.deepseek.fast_model,
            "reasoning_model": self.runtime_settings.deepseek.reasoning_model,
            "base_url": self.runtime_settings.deepseek.base_url,
            "status_note": self.runtime_settings.deepseek.status_note,
        }

    def _capability_source(self, spec: BuilderCapabilitySpec, payload: BaseModel) -> CodexContextInlineSource:
        return _capability_source_for_api(spec, payload)

    def _request_structured_completion_payload(
        self,
        *,
        model: str,
        system_instruction: str,
        user_payload: str,
        thinking_mode: str,
        reasoning_effort: str | None,
        max_tokens: int,
        output_model: type[BaseModel],
        preserve_reasoning_on_retry: bool = False,
        expand_retry_budget: bool = True,
        retry_instruction: str | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("DeepSeek client no disponible.")
        base_messages = [
            {
                "role": "system",
                "content": (
                    f"{system_instruction}\n\n"
                    f"{_build_deepseek_json_guidance(output_model)}"
                ),
            },
            {
                "role": "user",
                "content": user_payload,
            },
        ]
        base_request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": base_messages,
            "response_format": {"type": "json_object"},
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "extra_body": {"thinking": {"type": thinking_mode}},
        }
        if timeout_seconds is not None:
            base_request_kwargs["timeout"] = max(1.0, timeout_seconds)
        if reasoning_effort:
            base_request_kwargs["reasoning_effort"] = reasoning_effort
        self._last_completion_metadata = {}
        retry_limit = 1
        for retry_count in range(retry_limit + 1):
            request_kwargs = dict(base_request_kwargs)
            if retry_count > 0:
                request_kwargs["messages"] = [
                    *base_messages,
                    {
                        "role": "user",
                        "content": retry_instruction
                        or (
                            "La respuesta anterior fue truncada por longitud. "
                            "Regenera desde cero un unico objeto JSON valido, completo y compacto. "
                            "No incluyas markdown ni explicaciones."
                        ),
                    },
                ]
                request_kwargs["max_tokens"] = _deepseek_retry_max_tokens(
                    max_tokens,
                    expand_budget=expand_retry_budget,
                )
                request_kwargs["temperature"] = 0
                if not preserve_reasoning_on_retry:
                    request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                    request_kwargs.pop("reasoning_effort", None)

            response = self._client.chat.completions.create(**request_kwargs)
            message = response.choices[0].message if response.choices else None
            finish_reason = response.choices[0].finish_reason if response.choices else "unknown"
            content = message.content if message is not None and isinstance(message.content, str) else ""
            usage = normalize_deepseek_usage(getattr(response, "usage", None))
            metadata = {
                "request_id": str(getattr(response, "id", "") or ""),
                "finish_reason": str(finish_reason or "completed"),
                "token_usage": usage.compatibility_token_usage(),
                "normalized_usage": usage,
                "model_name": model,
                "output_model": output_model.__name__,
                "raw_content_length": len(content),
                "retry_count": retry_count,
            }
            self._last_completion_metadata = dict(metadata)
            try:
                if not content.strip():
                    raise ValueError("DeepSeek devolvio contenido vacio.")
                payload = _extract_json_payload(content)
                return payload, metadata
            except Exception:
                if retry_count < retry_limit and _is_deepseek_length_finish_reason(str(finish_reason or "")):
                    continue
                raise
        raise RuntimeError("DeepSeek no devolvio un payload usable.")

    def _create_structured_completion(
        self,
        *,
        model: str,
        system_instruction: str,
        user_payload: str,
        output_model: type[BaseModel],
        thinking_mode: str,
        reasoning_effort: str | None,
        max_tokens: int,
    ) -> BaseModel:
        payload, metadata = self._request_structured_completion_payload(
            model=model,
            system_instruction=system_instruction,
            user_payload=user_payload,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            output_model=output_model,
        )
        normalized, schema_status = validate_or_repair_structured_payload(payload, output_model)
        self._last_completion_metadata = {**metadata, "schema_validation_status": schema_status}
        return normalized

    def _execute_structured_capability(
        self,
        *,
        capability: BuilderCapability,
        payload: BaseModel,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        spec = get_builder_capability_spec(capability)
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind=f"deepseek_{spec.task_kind}",
            task_instruction=spec.task_instruction,
            inline_sources=[self._capability_source(spec, payload)],
            context_bundle=context_bundle,
        )
        model_name = (
            self.runtime_settings.deepseek.reasoning_model
            if spec.preferred_model == "reasoning"
            else self.runtime_settings.deepseek.fast_model
        )
        call_context = build_llm_call_context(
            context_bundle,
            capability=capability.value,
            provider_key=LLMProviderKey.deepseek.value,
            execution_backend=AgentExecutionBackend.provider_native.value,
            metadata={
                "base_url": self.runtime_settings.deepseek.base_url,
                "prompt_version": spec.prompt_version,
                "preferred_model": spec.preferred_model,
                "task_kind": spec.task_kind,
            },
        )
        base_result = LLMArtifactResult(
            artifact=None,
            provider_key=LLMProviderKey.deepseek.value,
            execution_backend=AgentExecutionBackend.provider_native.value,
            execution_mode=call_context.execution_mode,
            capability_key=capability.value,
            model_name=model_name,
            prompt_version=spec.prompt_version,
            schema_validation_status="not_attempted",
            finops_context=call_context,
            capability_policy=_capability_policy_payload(spec),
        )

        def _call_deepseek() -> LLMArtifactResult:
            self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
            if not self.is_available():
                detail = self._provider_unavailable_detail()
                return self._attach_context_metadata(
                    replace(
                        base_result,
                        warning=_format_provider_unavailable_warning(
                            provider_label="DeepSeek",
                            capability=capability,
                            spec=spec,
                            detail=detail,
                        ),
                        finish_reason="provider_unavailable",
                        failure_kind="provider_unavailable",
                        failure_detail=detail,
                        degraded=True,
                    ),
                    context_envelope=context_envelope,
                )

            try:
                reasoning_effort = (
                    _effective_deepseek_reasoning_effort(
                        capability,
                        self.runtime_settings.deepseek.reasoning_effort,
                        payload=payload,
                    )
                    if spec.preferred_model == "reasoning"
                    else None
                )
                raw_payload, metadata = self._request_structured_completion_payload(
                    model=model_name,
                    system_instruction=_localized_instruction(spec.system_instruction, context_bundle),
                    user_payload=context_envelope.user_payload,
                    thinking_mode="enabled" if spec.preferred_model == "reasoning" else "disabled",
                    reasoning_effort=reasoning_effort,
                    max_tokens=_structured_capability_max_tokens(capability, payload=payload),
                    output_model=spec.output_model,
                    preserve_reasoning_on_retry=_preserve_deepseek_reasoning_on_retry(capability, payload=payload),
                    expand_retry_budget=_expand_deepseek_retry_budget(capability, payload=payload),
                    retry_instruction=_deepseek_retry_instruction(capability, payload=payload),
                    timeout_seconds=max(1.0, spec.timeout_ms / 1000),
                )
                normalized, schema_status = validate_or_repair_structured_payload(raw_payload, spec.output_model)
                if capability == BuilderCapability.generate_diagram_model:
                    normalized, schema_status = finalize_structured_diagram_artifact(
                        normalized,
                        schema_status=schema_status,
                    )
                usage = metadata.get("normalized_usage")
                return self._attach_context_metadata(
                    replace(
                        base_result,
                        artifact=normalized,
                        request_id=str(metadata.get("request_id", "")),
                        finish_reason=str(metadata.get("finish_reason", "completed")),
                        schema_validation_status=schema_status,
                        token_usage=dict(metadata.get("token_usage", {})),
                        normalized_usage=usage if isinstance(usage, type(base_result.normalized_usage)) else usage,
                        retry_count=int(metadata.get("retry_count", 0) or 0),
                    ),
                    context_envelope=context_envelope,
                )
            except Exception as exc:
                failure_kind = "schema_invalid" if exc.__class__.__name__ == "ValidationError" else "provider_error"
                usage = self._last_completion_metadata.get("normalized_usage")
                return self._attach_context_metadata(
                    replace(
                        base_result,
                        warning=f"DeepSeek no pudo ejecutar {capability.value}; policy={spec.fallback_policy}.",
                        request_id=str(self._last_completion_metadata.get("request_id", "")),
                        finish_reason=str(self._last_completion_metadata.get("finish_reason", "exception")),
                        schema_validation_status=str(self._last_completion_metadata.get("schema_validation_status", "invalid")),
                        token_usage=dict(self._last_completion_metadata.get("token_usage", {})),
                        normalized_usage=usage if isinstance(usage, type(base_result.normalized_usage)) else usage,
                        failure_kind=failure_kind,
                        failure_detail=str(exc)[:400],
                        retry_count=int(self._last_completion_metadata.get("retry_count", 0) or 0),
                        degraded=True,
                    ),
                    context_envelope=context_envelope,
                )

        return record_provider_call(
            call=_call_deepseek,
            call_context=call_context,
            provider_key=LLMProviderKey.deepseek,
            model_name=model_name,
            requested_model=model_name,
            execution_backend=AgentExecutionBackend.provider_native.value,
            execution_mode=call_context.execution_mode,
            ledger_service=self._finops_ledger_service,
            session_factory=self._finops_session_factory,
            metadata={
                "capability": capability.value,
                "prompt_version": spec.prompt_version,
                "preferred_model": spec.preferred_model,
                "base_url": self.runtime_settings.deepseek.base_url,
            },
        )

    def normalize_discovery(
        self,
        payload: DiscoveryInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="deepseek_discovery_normalization",
            task_instruction=(
                "Normaliza la captura compactada `discovery_capture` a un discovery estructurado para un builder Lean "
                "de agentes. Usa solo hechos presentes en las fuentes compactadas. Si un dato no esta claro, usa "
                "'unknown'. Para case_type usa solo: informacion, automatizacion, copiloto, operador_autonomo, "
                "sistema_multiagente. Para autonomy_level usa solo: low, medium, high."
            ),
            inline_sources=[
                CodexContextInlineSource(
                    key="discovery_capture",
                    title="Discovery capture",
                    content=json.dumps(_compact_discovery_input(payload), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Captura cruda de discovery para normalizacion estructurada por provider API.",
                )
            ],
            context_bundle=context_bundle,
        )
        self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
        if not self.is_available():
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="DeepSeek no esta disponible para recomendar tools minimas; se mantiene el preflight heuristico.",
                    provider_key=LLMProviderKey.deepseek.value,
                ),
                context_envelope=context_envelope,
            )
        try:
            parsed = self._create_structured_completion(
                model=self.runtime_settings.deepseek.fast_model,
                system_instruction=_localized_instruction(
                    (
                        "Normaliza la captura a un discovery estructurado para un builder Lean de agentes. "
                        "Usa solo hechos presentes en la entrada. Si un dato no esta claro, usa 'unknown'. "
                        "Para case_type usa solo: informacion, automatizacion, copiloto, operador_autonomo, sistema_multiagente. "
                        "Para autonomy_level usa solo: low, medium, high."
                    ),
                    context_bundle,
                ),
                user_payload=context_envelope.user_payload,
                output_model=DiscoveryArtifact,
                thinking_mode="disabled",
                reasoning_effort=None,
                max_tokens=4096,
            )
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=sanitize_discovery(DiscoveryArtifact.model_validate(parsed.model_dump(mode="json"))),
                    provider_key=LLMProviderKey.deepseek.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="DeepSeek no pudo normalizar discovery; se uso fallback deterministico.",
                    provider_key=LLMProviderKey.deepseek.value,
                ),
                context_envelope=context_envelope,
            )

    def build_canvas(
        self,
        discovery: DiscoveryArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="deepseek_canvas_generation",
            task_instruction=(
                "Genera un canvas Lean para un agente usando solo la fuente compactada `normalized_discovery`. "
                "Manten el alcance corto, concreto y util para un MVP. Si algo no esta claro, usa 'unknown'."
            ),
            inline_sources=[
                CodexContextInlineSource(
                    key="normalized_discovery",
                    title="Normalized discovery",
                    content=json.dumps(_compact_discovery_artifact(discovery), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Discovery estructurado aprobado para construir el canvas via provider API.",
                )
            ],
            context_bundle=context_bundle,
        )
        self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
        if not self.is_available():
            return self._attach_context_metadata(
                LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.deepseek.value),
                context_envelope=context_envelope,
            )
        try:
            parsed = self._create_structured_completion(
                model=self.runtime_settings.deepseek.fast_model,
                system_instruction=_localized_instruction(
                    (
                        "Genera un canvas Lean para un agente usando solo el discovery recibido. "
                        "Manten el alcance corto, concreto y util para un MVP. Si algo no esta claro, usa 'unknown'."
                    ),
                    context_bundle,
                ),
                user_payload=context_envelope.user_payload,
                output_model=CanvasArtifact,
                thinking_mode="disabled",
                reasoning_effort=None,
                max_tokens=4096,
            )
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=sanitize_canvas(CanvasArtifact.model_validate(parsed.model_dump(mode="json"))),
                    provider_key=LLMProviderKey.deepseek.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="DeepSeek no pudo construir el canvas; se uso fallback deterministico.",
                    provider_key=LLMProviderKey.deepseek.value,
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
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="deepseek_blueprint_narrative",
            task_instruction=(
                "Redacta la narrativa tecnica de un blueprint Lean para un agente usando solo las fuentes compactadas "
                "`narrative_discovery`, `narrative_canvas` y `narrative_blueprint`. No cambies la arquitectura, "
                "memoria, tools ni guardrails ya definidos. Explica por que la recomendacion encaja con el discovery "
                "y el canvas, y resalta tradeoffs relevantes sin inventar nuevos componentes."
            ),
            inline_sources=[
                CodexContextInlineSource(
                    key="narrative_discovery",
                    title="Discovery for blueprint narrative",
                    content=json.dumps(_compact_discovery_artifact(discovery), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Discovery estructurado aprobado para la narrativa tecnica.",
                ),
                CodexContextInlineSource(
                    key="narrative_canvas",
                    title="Canvas for blueprint narrative",
                    content=json.dumps(_compact_canvas_artifact(canvas), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Canvas Lean aprobado que define alcance y meta del blueprint.",
                ),
                CodexContextInlineSource(
                    key="narrative_blueprint",
                    title="Blueprint for narrative synthesis",
                    content=json.dumps(_compact_blueprint_artifact(blueprint), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Blueprint base cuya narrativa debe sintetizarse sin alterar contratos.",
                ),
            ],
            context_bundle=context_bundle,
        )
        self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
        if not self.is_available():
            return self._attach_context_metadata(
                LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.deepseek.value),
                context_envelope=context_envelope,
            )
        try:
            parsed = self._create_structured_completion(
                model=self.runtime_settings.deepseek.reasoning_model,
                system_instruction=_localized_instruction(
                    (
                        "Redacta la narrativa tecnica de un blueprint Lean para un agente. "
                        "No cambies la arquitectura, memoria, tools ni guardrails ya definidos. "
                        "Explica por que la recomendacion encaja con el discovery y el canvas, "
                        "y resalta tradeoffs relevantes sin inventar nuevos componentes."
                    ),
                    context_bundle,
                ),
                user_payload=context_envelope.user_payload,
                output_model=BlueprintNarrativeOutput,
                thinking_mode="enabled",
                reasoning_effort=self.runtime_settings.deepseek.reasoning_effort,
                max_tokens=3072,
            )
            normalized = BlueprintNarrativeOutput.model_validate(parsed.model_dump(mode="json"))
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=BlueprintNarrativeOutput(narrative=normalize_text(normalized.narrative)),
                    provider_key=LLMProviderKey.deepseek.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="DeepSeek no pudo sintetizar la narrativa; se mantuvo la narrativa base.",
                    provider_key=LLMProviderKey.deepseek.value,
                ),
                context_envelope=context_envelope,
            )

    def recommend_minimal_tools(
        self,
        prompt_input: ToolRecommendationPromptInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        case_payload = _compact_tool_recommendation_case_payload(prompt_input)
        catalog_payload = _compact_tool_recommendation_catalog_payload(prompt_input)
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="deepseek_tool_recommendation",
            task_instruction=build_tool_recommendation_context_task_instruction(),
            inline_sources=[
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
            ],
            context_bundle=context_bundle,
        )
        self._ensure_workspace_client(context_bundle.workspace_id if context_bundle is not None else None)
        if not self.is_available():
            return self._attach_context_metadata(
                LLMArtifactResult(artifact=None, provider_key=LLMProviderKey.deepseek.value),
                context_envelope=context_envelope,
            )
        try:
            parsed = self._create_structured_completion(
                model=self.runtime_settings.deepseek.reasoning_model,
                system_instruction=_localized_instruction(
                    build_tool_recommendation_system_instruction(),
                    context_bundle,
                ),
                user_payload=context_envelope.user_payload,
                output_model=ToolRecommendationLLMOutput,
                thinking_mode="enabled",
                reasoning_effort=self.runtime_settings.deepseek.reasoning_effort,
                max_tokens=4096,
            )
            normalized = ToolRecommendationLLMOutput.model_validate(parsed.model_dump(mode="json"))
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=normalized,
                    provider_key=LLMProviderKey.deepseek.value,
                ),
                context_envelope=context_envelope,
            )
        except Exception:
            return self._attach_context_metadata(
                LLMArtifactResult(
                    artifact=None,
                    warning="DeepSeek no pudo recomendar tools minimas; se mantuvo el preflight heuristico.",
                    provider_key=LLMProviderKey.deepseek.value,
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


def get_openai_builder_service() -> BuilderProviderFacade:
    runtime_settings = load_llm_runtime_settings()
    return build_builder_service(runtime_settings)
