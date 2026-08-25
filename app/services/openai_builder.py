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
    return 4096


def _preserve_deepseek_reasoning_on_retry(capability: BuilderCapability, *, payload: object | None = None) -> bool:
    return capability == BuilderCapability.generate_diagram_model and not _is_bpmn_diagram_payload(payload)


def _expand_deepseek_retry_budget(capability: BuilderCapability, *, payload: object | None = None) -> bool:
    return capability == BuilderCapability.generate_diagram_model and not _is_bpmn_diagram_payload(payload)


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


def _serialize_capability_payload_for_api(payload: BaseModel) -> dict[str, Any]:
    serialized = payload.model_dump(mode="json")
    if not isinstance(payload, DiagramGenerationInput):
        return serialized

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
    if compact_context:
        serialized["source_context"] = compact_context

    if isinstance(serialized.get("source_refs"), list):
        serialized["source_refs"] = serialized["source_refs"][:12]
    if isinstance(serialized.get("resolved_inputs"), list):
        serialized["resolved_inputs"] = serialized["resolved_inputs"][:4]
    if isinstance(serialized.get("missing_required_inputs"), list):
        serialized["missing_required_inputs"] = serialized["missing_required_inputs"][:8]
    return serialized


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


def _resolve_workspace_provider_api_key(session, *, workspace_id, provider_key: LLMProviderKey) -> str | None:
    from app.services.llm_runtime.runtime_secrets_service import resolve_workspace_provider_secret_value

    return resolve_workspace_provider_secret_value(
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
        if self.runtime_settings.uses_platform_credentials or self._finops_session_factory is None:
            return
        self._workspace_client_error = ""
        try:
            with self._finops_session_factory() as session:
                api_key = _resolve_workspace_provider_api_key(
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
        self._workspace_client_error = "No se resolvio una API key de OpenAI para el workspace."

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
        return CodexContextInlineSource(
            key=spec.source_key,
            title=spec.source_title,
            content=json.dumps(_serialize_capability_payload_for_api(payload), ensure_ascii=True, indent=2),
            required=True,
            summary=spec.source_summary,
        )

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
                "text_format": spec.output_model,
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
                    content=json.dumps(payload.model_dump(mode="json"), ensure_ascii=True, indent=2),
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
                    content=json.dumps(discovery.model_dump(mode="json"), ensure_ascii=True, indent=2),
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
                    content=json.dumps(discovery.model_dump(mode="json"), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Discovery estructurado aprobado para la narrativa tecnica.",
                ),
                CodexContextInlineSource(
                    key="narrative_canvas",
                    title="Canvas for blueprint narrative",
                    content=json.dumps(canvas.model_dump(mode="json"), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Canvas Lean aprobado que define alcance y meta del blueprint.",
                ),
                CodexContextInlineSource(
                    key="narrative_blueprint",
                    title="Blueprint for narrative synthesis",
                    content=json.dumps(blueprint.model_dump(mode="json"), ensure_ascii=True, indent=2),
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
        case_payload = prompt_input.model_dump(
            mode="json",
            exclude={"candidate_tools", "mandatory_tool_keys", "forbidden_tool_keys"},
        )
        catalog_payload = {
            "mandatory_tool_keys": [item.value for item in prompt_input.mandatory_tool_keys],
            "forbidden_tool_keys": [item.value for item in prompt_input.forbidden_tool_keys],
            "candidate_tools": [item.model_dump(mode="json") for item in prompt_input.candidate_tools],
        }
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="openai_tool_recommendation",
            task_instruction=(
                "Selecciona el conjunto minimo de herramientas para la etapa Herramientas usando solo las fuentes "
                "compactadas `tool_recommendation_case` y `tool_recommendation_catalog`. Usa exclusivamente tools "
                "del catalogo permitido. Manten toda tool marcada como mandatory si sigue siendo necesaria segun el "
                "contexto. Usa `requirements_coverage` y `design_role_coverage` para defender cobertura real por "
                "requisito y rol de diseno. Clasifica el resto como optional o unnecessary solo si aportan "
                "capacidad unica. Si la evidencia es insuficiente, agrega gaps estructurados en lugar de inventar tools."
            ),
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
                            (
                                "Selecciona el conjunto minimo de herramientas para un agente Lean. "
                                "Usa solo el contexto aprobado y el catalogo permitido. "
                                "Nunca inventes tool keys fuera del catalogo. "
                                "Manten toda tool mandatory si la evidencia la sostiene. "
                                "Usa requirements_coverage y design_role_coverage para justificar cobertura real. "
                                "Marca como unnecessary cualquier tool candidata que no aporte capacidad unica. "
                                "Si falta informacion, devuelve gaps estructurados y conserva la propuesta minima."
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
        if self.runtime_settings.uses_platform_credentials or self._finops_session_factory is None:
            return
        self._workspace_client_error = ""
        try:
            with self._finops_session_factory() as session:
                api_key = _resolve_workspace_provider_api_key(
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
        self._workspace_client_error = "No se resolvio una API key de DeepSeek para el workspace."

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
        return CodexContextInlineSource(
            key=spec.source_key,
            title=spec.source_title,
            content=json.dumps(_serialize_capability_payload_for_api(payload), ensure_ascii=True, indent=2),
            required=True,
            summary=spec.source_summary,
        )

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
                    content=json.dumps(payload.model_dump(mode="json"), ensure_ascii=True, indent=2),
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
                    content=json.dumps(discovery.model_dump(mode="json"), ensure_ascii=True, indent=2),
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
                    content=json.dumps(discovery.model_dump(mode="json"), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Discovery estructurado aprobado para la narrativa tecnica.",
                ),
                CodexContextInlineSource(
                    key="narrative_canvas",
                    title="Canvas for blueprint narrative",
                    content=json.dumps(canvas.model_dump(mode="json"), ensure_ascii=True, indent=2),
                    required=True,
                    summary="Canvas Lean aprobado que define alcance y meta del blueprint.",
                ),
                CodexContextInlineSource(
                    key="narrative_blueprint",
                    title="Blueprint for narrative synthesis",
                    content=json.dumps(blueprint.model_dump(mode="json"), ensure_ascii=True, indent=2),
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
        case_payload = prompt_input.model_dump(
            mode="json",
            exclude={"candidate_tools", "mandatory_tool_keys", "forbidden_tool_keys"},
        )
        catalog_payload = {
            "mandatory_tool_keys": [item.value for item in prompt_input.mandatory_tool_keys],
            "forbidden_tool_keys": [item.value for item in prompt_input.forbidden_tool_keys],
            "candidate_tools": [item.model_dump(mode="json") for item in prompt_input.candidate_tools],
        }
        context_envelope = self._build_context_envelope(
            role="builder",
            task_kind="deepseek_tool_recommendation",
            task_instruction=(
                "Selecciona el conjunto minimo de herramientas usando solo `tool_recommendation_case` y "
                "`tool_recommendation_catalog`. Usa exclusivamente tools del catalogo permitido. Conserva las "
                "mandatory cuando sigan justificadas. Usa `requirements_coverage` y `design_role_coverage` para "
                "defender cobertura por requisito y rol. Marca el resto como optional o unnecessary segun cobertura "
                "real. Si falta informacion, devuelve gaps estructurados en lugar de inventar tools."
            ),
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
                    (
                        "Selecciona el conjunto minimo de herramientas para un agente Lean. "
                        "Usa solo el contexto aprobado y el catalogo permitido. "
                        "Nunca inventes tool keys fuera del catalogo. "
                        "Manten toda tool mandatory si la evidencia la sostiene. "
                        "Usa requirements_coverage y design_role_coverage para justificar cobertura real. "
                        "Marca como unnecessary cualquier tool candidata que no aporte capacidad unica. "
                        "Si falta informacion, devuelve gaps estructurados y conserva la propuesta minima."
                    ),
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
