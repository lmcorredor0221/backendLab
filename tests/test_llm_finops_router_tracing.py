from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from app.models import (
    AgentExecutionBackend,
    CanvasArtifact,
    CodexLocalProviderConfig,
    DeepSeekProviderConfig,
    DiscoveryArtifact,
    DiscoveryInput,
    LLMProviderKey,
    LLMRuntimeSettings,
    OpenAIProviderConfig,
)
from app.services.llm_runtime.builder_contracts import LLMArtifactResult
from app.services.llm_runtime.provider_router import BuilderProviderFacade
from app.services.llm_runtime.stage_context_types import StageContextBundle


def sample_discovery_input() -> DiscoveryInput:
    return DiscoveryInput(problem_statement="Normalizar discovery.")


def sample_discovery_artifact() -> DiscoveryArtifact:
    return DiscoveryArtifact(problem_statement="Normalizar discovery.")


def sample_canvas_artifact() -> CanvasArtifact:
    return CanvasArtifact(user_goal="Generar canvas trazable.")


def build_stage_context(*, capability: str, stage: str) -> StageContextBundle:
    return StageContextBundle(
        capability=capability,
        role="builder",
        stage=stage,
        workspace_id=uuid4(),
        session_id=uuid4(),
        session_snapshot=None,
        effective_language="es",
        knowledge_manifest=None,
        memory_policy=None,
        short_term_memory=None,
        context_fingerprint=f"ctx-{capability}",
    )


def build_runtime_settings(
    *,
    active_provider: LLMProviderKey,
    backend: AgentExecutionBackend,
    primary_agents: list[str] | None = None,
) -> LLMRuntimeSettings:
    return LLMRuntimeSettings(
        active_provider=active_provider,
        agent_execution_backend=backend,
        openai=OpenAIProviderConfig(
            fast_model="gpt-5.4-mini",
            reasoning_model="gpt-5.5",
            api_key_configured=True,
            available=True,
        ),
        deepseek=DeepSeekProviderConfig(
            base_url="https://api.deepseek.test",
            fast_model="deepseek-v4-flash",
            reasoning_model="deepseek-v4-pro",
            api_key_configured=True,
            available=True,
        ),
        codex_local=CodexLocalProviderConfig(
            command="codex",
            model="gpt-5.5",
            profile="finops-router",
            executable_found=True,
            available=True,
            primary_agents=primary_agents or [],
        ),
    )


@dataclass
class CapturingProviderService:
    provider_key: str
    result: LLMArtifactResult
    calls: list[StageContextBundle | None] = field(default_factory=list)

    def can_attempt(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def provider_summary(self) -> dict[str, str | bool]:
        return {"provider": self.provider_key, "configured": True, "sdk_ready": True}

    def normalize_discovery(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del payload
        self.calls.append(context_bundle)
        return self.result

    def build_canvas(self, discovery, *, context_bundle=None) -> LLMArtifactResult:
        del discovery
        self.calls.append(context_bundle)
        return self.result


def test_router_correlates_primary_and_shadow_contexts_with_same_operation_id() -> None:
    settings = build_runtime_settings(
        active_provider=LLMProviderKey.openai,
        backend=AgentExecutionBackend.shadow_codex_cli,
    )
    canvas_result = LLMArtifactResult(artifact=sample_canvas_artifact())
    openai_service = CapturingProviderService("openai", canvas_result)
    codex_service = CapturingProviderService("codex_local", canvas_result)
    facade = BuilderProviderFacade(
        settings,
        openai_service=openai_service,
        deepseek_service=CapturingProviderService("deepseek", canvas_result),
        codex_service=codex_service,
    )

    result = facade.build_canvas(
        sample_discovery_artifact(),
        context_bundle=build_stage_context(capability="build_canvas", stage="define"),
    )

    primary_context = openai_service.calls[0]
    shadow_context = codex_service.calls[0]
    assert primary_context is not None
    assert shadow_context is not None
    assert primary_context.finops_operation_id == shadow_context.finops_operation_id
    assert primary_context.finops_execution_mode == "shadow"
    assert shadow_context.finops_execution_mode == "shadow"
    assert primary_context.finops_metadata["route_leg"] == "primary"
    assert shadow_context.finops_metadata["route_leg"] == "shadow"
    assert shadow_context.finops_metadata["provider_key"] == "codex_local"
    assert result.finops_context is not None
    assert result.finops_context.operation_id == primary_context.finops_operation_id
    assert result.finops_context.metadata["shadow_provider_key"] == "codex_local"


def test_router_correlates_fallback_with_original_attempt() -> None:
    settings = build_runtime_settings(
        active_provider=LLMProviderKey.deepseek,
        backend=AgentExecutionBackend.codex_cli,
        primary_agents=["normalize_discovery"],
    )
    codex_service = CapturingProviderService(
        "codex_local",
        LLMArtifactResult(artifact=None, warning="Codex local no disponible."),
    )
    fallback_result = LLMArtifactResult(artifact=sample_discovery_artifact())
    deepseek_service = CapturingProviderService("deepseek", fallback_result)
    facade = BuilderProviderFacade(
        settings,
        openai_service=CapturingProviderService("openai", fallback_result),
        deepseek_service=deepseek_service,
        codex_service=codex_service,
    )

    result = facade.normalize_discovery(
        sample_discovery_input(),
        context_bundle=build_stage_context(capability="normalize_discovery", stage="discover"),
    )

    original_context = codex_service.calls[0]
    fallback_context = deepseek_service.calls[0]
    assert original_context is not None
    assert fallback_context is not None
    assert original_context.finops_operation_id == fallback_context.finops_operation_id
    assert original_context.finops_metadata["route_leg"] == "primary"
    assert fallback_context.finops_metadata["route_leg"] == "fallback"
    assert fallback_context.finops_metadata["provider_key"] == "deepseek"
    assert fallback_context.finops_metadata["fallback_used"] is True
    assert result.provider_key == "deepseek"
    assert result.fallback_used is True
    assert result.finops_context is not None
    assert result.finops_context.operation_id == original_context.finops_operation_id
    assert result.finops_context.metadata["fallback_used"] is True
