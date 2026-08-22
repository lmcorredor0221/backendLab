from __future__ import annotations

from dataclasses import dataclass, field

from app.models import (
    AgentExecutionBackend,
    BlueprintArtifact,
    CanvasArtifact,
    CodexLocalCostPolicy,
    CodexLocalProviderConfig,
    DeepSeekProviderConfig,
    DiscoveryArtifact,
    DiscoveryInput,
    KnowledgeAccessBackend,
    LLMProviderKey,
    LLMRuntimeSettings,
    MemoryProfile,
    OpenAIProviderConfig,
    ReviewState,
)
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_runtime.builder_contracts import BlueprintNarrativeOutput, LLMArtifactResult
from app.services.llm_runtime.codex_cli.provider_facade import CodexLocalBuilderService
from app.services.llm_runtime.provider_router import (
    BuilderCapability,
    BuilderExecutionMode,
    BuilderProviderFacade,
    BuilderProviderRouter,
)
from app.services.openai_builder import build_builder_service
from app.services.llm_runtime.stage_context_types import StageContextBundle


def sample_discovery_input() -> DiscoveryInput:
    return DiscoveryInput(
        problem_statement="Estandarizar discovery para un builder Lean.",
        current_user="Arquitecto de soluciones",
        current_process="Consolida inputs dispersos y luego redacta artefactos manualmente.",
        desired_outcome="Generar discovery consistente con menos retrabajo.",
        autonomy_level="high",
        constraints=["Sin side effects irreversibles"],
        operational_baseline={
            "current_time_spent": "4 horas",
            "current_cost": "Retrabajo del equipo",
            "frequent_errors": ["Drift del objetivo"],
            "automation_opportunities": ["Normalizar discovery"],
        },
        mvp_definition={
            "v1_scope": ["Discovery", "Canvas"],
            "out_of_scope": ["Provisioning"],
            "north_star_metric": "Discovery util en una sesion",
            "non_delegable_decisions": ["Aprobar promocion"],
        },
    )


def sample_discovery_artifact() -> DiscoveryArtifact:
    return DiscoveryArtifact(
        problem_statement="Estandarizar discovery para un builder Lean.",
        current_user="Arquitecto de soluciones",
        current_process="Consolida inputs dispersos y luego redacta artefactos manualmente.",
        desired_outcome="Generar discovery consistente con menos retrabajo.",
        autonomy_level="high",
        constraints=["Sin side effects irreversibles"],
        operational_baseline={
            "current_time_spent": "4 horas",
            "current_cost": "Retrabajo del equipo",
            "frequent_errors": ["Drift del objetivo"],
            "automation_opportunities": ["Normalizar discovery"],
        },
        mvp_definition={
            "v1_scope": ["Discovery", "Canvas"],
            "out_of_scope": ["Provisioning"],
            "north_star_metric": "Discovery util en una sesion",
            "non_delegable_decisions": ["Aprobar promocion"],
        },
        case_type="automatizacion",
        value_statement="Reducir retrabajo del equipo.",
    )


def sample_canvas_artifact() -> CanvasArtifact:
    return CanvasArtifact(
        user_goal="Generar discovery consistente con menos retrabajo.",
        mvp_scope=["Discovery", "Canvas"],
        out_of_scope=["Provisioning"],
        success_metric="Discovery util en una sesion",
        primary_risk="Drift del objetivo",
        agent_profile={
            "mission": "Normalizar discovery sin inventar datos.",
            "primary_user": "Arquitecto de soluciones",
            "agent_task": "Estructurar discovery y canvas base.",
            "allowed_decisions": ["Proponer estructura"],
            "prohibited_decisions": ["Ejecutar side effects"],
            "key_inputs": ["Problema", "Proceso actual"],
            "expected_outputs": ["Discovery", "Canvas"],
            "human_approvals": ["Promocion a implementacion"],
            "success_metrics": ["Discovery util en una sesion"],
        },
    )


def sample_blueprint_artifact() -> BlueprintArtifact:
    return BlueprintArtifact(
        architecture="single_agent_with_skills",
        reasoning_pattern="Plan-and-Execute",
        memory_strategy="session_memory_with_checkpoints",
        tools=[],
        memory_profile=MemoryProfile(
            strategy="session_memory_with_checkpoints",
            storage_layers=["session_state"],
            write_policy="Persistir estado validado",
            retrieval_policy="Recuperar por session_id",
            review_trigger="Campos faltantes",
            goal_drift_guard="Comparar contra desired_outcome",
        ),
        safety_checks=[],
        guardrails=["No inventar datos"],
        readiness_state=ReviewState.partial,
        narrative="Blueprint base.",
    )


def build_runtime_settings(
    *,
    active_provider: LLMProviderKey,
    backend: AgentExecutionBackend,
    primary_agents: list[str] | None = None,
    shadow_agents: list[str] | None = None,
    staged_agents: list[str] | None = None,
) -> LLMRuntimeSettings:
    return LLMRuntimeSettings(
        active_provider=active_provider,
        agent_execution_backend=backend,
        knowledge_access_backend=KnowledgeAccessBackend.inline_context,
        openai=OpenAIProviderConfig(
            fast_model="gpt-5.4-mini",
            reasoning_model="gpt-5.5",
            reasoning_effort="low",
            api_key_configured=True,
            available=True,
            status_note="ready",
        ),
        deepseek=DeepSeekProviderConfig(
            base_url="https://api.deepseek.com",
            fast_model="deepseek-v4-flash",
            reasoning_model="deepseek-v4-pro",
            reasoning_effort="high",
            api_key_configured=True,
            available=True,
            status_note="ready",
        ),
        codex_local=CodexLocalProviderConfig(
            command="codex",
            model="gpt-5.5",
            profile="bridge-test",
            cost_policy=CodexLocalCostPolicy.hybrid,
            timeout_ms=150000,
            max_concurrency=1,
            runner_id="bridge-test",
            auth_mode="auto",
            fallback_models=["gpt-5.5-mini"],
            primary_agents=primary_agents or [],
            shadow_agents=shadow_agents or [],
            staged_agents=staged_agents or [],
            available=True,
            executable_found=True,
            status_note="ready",
        ),
    )


@dataclass
class FakeBuilderService:
    provider_key: str
    discovery_result: LLMArtifactResult
    canvas_result: LLMArtifactResult
    narrative_result: LLMArtifactResult
    available: bool = True
    calls: dict[str, int] = field(default_factory=lambda: {"normalize_discovery": 0, "build_canvas": 0, "narrative": 0})

    def can_attempt(self) -> bool:
        return True

    def is_available(self) -> bool:
        return self.available

    def provider_summary(self) -> dict[str, str | bool]:
        return {
            "provider": self.provider_key,
            "mode": "fake",
            "configured": True,
            "sdk_ready": self.available,
            "fast_model": f"{self.provider_key}-fast",
            "reasoning_model": f"{self.provider_key}-reasoning",
        }

    def normalize_discovery(
        self,
        payload: DiscoveryInput,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        del payload, context_bundle
        self.calls["normalize_discovery"] += 1
        return self.discovery_result

    def build_canvas(
        self,
        discovery: DiscoveryArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        del discovery, context_bundle
        self.calls["build_canvas"] += 1
        return self.canvas_result

    def synthesize_blueprint_narrative(
        self,
        discovery: DiscoveryArtifact,
        canvas: CanvasArtifact,
        blueprint: BlueprintArtifact,
        *,
        context_bundle: StageContextBundle | None = None,
    ) -> LLMArtifactResult:
        del discovery, canvas, blueprint, context_bundle
        self.calls["narrative"] += 1
        return self.narrative_result


def test_router_resolves_primary_shadow_and_staged_rollouts() -> None:
    router = BuilderProviderRouter(
        build_runtime_settings(
            active_provider=LLMProviderKey.deepseek,
            backend=AgentExecutionBackend.codex_cli,
            primary_agents=["normalize_discovery"],
            shadow_agents=["build_canvas"],
            staged_agents=["build_blueprint"],
        )
    )

    discovery_route = router.resolve(BuilderCapability.normalize_discovery)
    canvas_route = router.resolve(BuilderCapability.build_canvas)
    narrative_route = router.resolve(BuilderCapability.synthesize_blueprint_narrative)

    assert discovery_route.selected_provider == LLMProviderKey.codex_local
    assert discovery_route.execution_mode == BuilderExecutionMode.primary
    assert discovery_route.fallback_provider == LLMProviderKey.deepseek

    assert canvas_route.selected_provider == LLMProviderKey.deepseek
    assert canvas_route.execution_mode == BuilderExecutionMode.shadow
    assert canvas_route.shadow_provider == LLMProviderKey.codex_local

    assert narrative_route.selected_provider == LLMProviderKey.deepseek
    assert narrative_route.execution_mode == BuilderExecutionMode.staged


def test_router_shadow_backend_defaults_to_shadow_for_unlisted_capabilities() -> None:
    router = BuilderProviderRouter(
        build_runtime_settings(
            active_provider=LLMProviderKey.openai,
            backend=AgentExecutionBackend.shadow_codex_cli,
        )
    )

    route = router.resolve(BuilderCapability.build_canvas)

    assert route.selected_provider == LLMProviderKey.openai
    assert route.execution_mode == BuilderExecutionMode.shadow
    assert route.shadow_provider == LLMProviderKey.codex_local


def test_router_uses_codex_as_primary_when_it_is_the_active_provider() -> None:
    router = BuilderProviderRouter(
        build_runtime_settings(
            active_provider=LLMProviderKey.codex_local,
            backend=AgentExecutionBackend.provider_native,
        )
    )

    route = router.resolve(BuilderCapability.synthesize_blueprint_narrative)

    assert route.selected_provider == LLMProviderKey.codex_local
    assert route.execution_mode == BuilderExecutionMode.primary
    assert route.execution_backend == AgentExecutionBackend.codex_cli


def test_facade_falls_back_to_native_provider_when_codex_primary_fails() -> None:
    runtime_settings = build_runtime_settings(
        active_provider=LLMProviderKey.deepseek,
        backend=AgentExecutionBackend.codex_cli,
        primary_agents=["normalize_discovery"],
    )
    expected_discovery = sample_discovery_artifact()
    openai_service = FakeBuilderService(
        provider_key="openai",
        discovery_result=LLMArtifactResult(artifact=None, warning="OpenAI no disponible."),
        canvas_result=LLMArtifactResult(artifact=sample_canvas_artifact()),
        narrative_result=LLMArtifactResult(artifact=BlueprintNarrativeOutput(narrative="OpenAI")),
    )
    deepseek_service = FakeBuilderService(
        provider_key="deepseek",
        discovery_result=LLMArtifactResult(artifact=expected_discovery),
        canvas_result=LLMArtifactResult(artifact=sample_canvas_artifact()),
        narrative_result=LLMArtifactResult(artifact=BlueprintNarrativeOutput(narrative="DeepSeek")),
    )
    codex_service = FakeBuilderService(
        provider_key="codex_local",
        discovery_result=LLMArtifactResult(artifact=None, warning="Codex local no pudo normalizar discovery."),
        canvas_result=LLMArtifactResult(artifact=sample_canvas_artifact()),
        narrative_result=LLMArtifactResult(artifact=BlueprintNarrativeOutput(narrative="Codex")),
    )
    facade = BuilderProviderFacade(
        runtime_settings,
        openai_service=openai_service,
        deepseek_service=deepseek_service,
        codex_service=codex_service,
    )

    result = facade.normalize_discovery(sample_discovery_input())

    assert isinstance(result.artifact, DiscoveryArtifact)
    assert result.artifact.problem_statement == expected_discovery.problem_statement
    assert result.provider_key == "deepseek"
    assert result.execution_backend == "codex_cli"
    assert result.execution_mode == "primary"
    assert "fallback lateral" in (result.warning or "").lower()
    assert deepseek_service.calls["normalize_discovery"] == 1
    assert codex_service.calls["normalize_discovery"] == 1


def test_facade_promotes_codex_from_shadow_when_primary_provider_fails() -> None:
    runtime_settings = build_runtime_settings(
        active_provider=LLMProviderKey.openai,
        backend=AgentExecutionBackend.shadow_codex_cli,
    )
    expected_canvas = sample_canvas_artifact()
    openai_service = FakeBuilderService(
        provider_key="openai",
        discovery_result=LLMArtifactResult(artifact=sample_discovery_artifact()),
        canvas_result=LLMArtifactResult(artifact=None, warning="OpenAI no pudo construir el canvas."),
        narrative_result=LLMArtifactResult(artifact=BlueprintNarrativeOutput(narrative="OpenAI")),
    )
    deepseek_service = FakeBuilderService(
        provider_key="deepseek",
        discovery_result=LLMArtifactResult(artifact=sample_discovery_artifact()),
        canvas_result=LLMArtifactResult(artifact=expected_canvas),
        narrative_result=LLMArtifactResult(artifact=BlueprintNarrativeOutput(narrative="DeepSeek")),
    )
    codex_service = FakeBuilderService(
        provider_key="codex_local",
        discovery_result=LLMArtifactResult(artifact=sample_discovery_artifact()),
        canvas_result=LLMArtifactResult(artifact=expected_canvas),
        narrative_result=LLMArtifactResult(artifact=BlueprintNarrativeOutput(narrative="Codex")),
    )
    facade = BuilderProviderFacade(
        runtime_settings,
        openai_service=openai_service,
        deepseek_service=deepseek_service,
        codex_service=codex_service,
    )

    result = facade.build_canvas(sample_discovery_artifact())

    assert isinstance(result.artifact, CanvasArtifact)
    assert result.artifact.user_goal == expected_canvas.user_goal
    assert result.provider_key == "codex_local"
    assert result.execution_mode == "shadow"
    assert "shadow" in (result.route_reason or "").lower()
    assert "promovio codex local" in (result.warning or "").lower()
    assert openai_service.calls["build_canvas"] == 1
    assert codex_service.calls["build_canvas"] == 1


def test_facade_provider_summary_exposes_route_metadata() -> None:
    runtime_settings = build_runtime_settings(
        active_provider=LLMProviderKey.deepseek,
        backend=AgentExecutionBackend.codex_cli,
        primary_agents=["normalize_discovery"],
        shadow_agents=["build_canvas"],
    )
    discovery = LLMArtifactResult(artifact=sample_discovery_artifact())
    canvas = LLMArtifactResult(artifact=sample_canvas_artifact())
    narrative = LLMArtifactResult(artifact=BlueprintNarrativeOutput(narrative="DeepSeek"))
    facade = BuilderProviderFacade(
        runtime_settings,
        openai_service=FakeBuilderService("openai", discovery, canvas, narrative),
        deepseek_service=FakeBuilderService("deepseek", discovery, canvas, narrative),
        codex_service=FakeBuilderService("codex_local", discovery, canvas, narrative),
    )

    summary = facade.provider_summary()

    assert summary["provider"] == "deepseek"
    assert summary["execution_backend"] == "codex_cli"
    assert summary["normalize_discovery_route"] == "primary:codex_local"
    assert summary["build_canvas_route"] == "shadow:deepseek"
    assert summary["synthesize_blueprint_narrative_route"] == "primary:deepseek"


def test_build_builder_service_wires_finops_dependencies_for_all_providers() -> None:
    facade = build_builder_service(
        build_runtime_settings(
            active_provider=LLMProviderKey.deepseek,
            backend=AgentExecutionBackend.provider_native,
        )
    )

    services = [
        facade._openai_service,
        facade._deepseek_service,
        facade._codex_service,
        facade._antigravity_service,
    ]

    for service in services:
        assert service is not None
        assert service._finops_session_factory is not None
        assert isinstance(service._finops_ledger_service, LLMUsageLedgerService)


def test_codex_local_builder_uses_staged_context_for_narrative_when_backend_is_hybrid() -> None:
    runtime_settings = build_runtime_settings(
        active_provider=LLMProviderKey.codex_local,
        backend=AgentExecutionBackend.codex_cli,
    )
    runtime_settings.knowledge_access_backend = KnowledgeAccessBackend.hybrid
    service = CodexLocalBuilderService(runtime_settings)
    captured: dict[str, object] = {}

    def fake_execute_structured_prompt(**kwargs):
        captured.update(kwargs)
        return BlueprintNarrativeOutput(narrative="Narrativa staged")

    service.execution_service.execute_structured_prompt = fake_execute_structured_prompt  # type: ignore[method-assign]

    result = service.synthesize_blueprint_narrative(
        sample_discovery_artifact(),
        sample_canvas_artifact(),
        sample_blueprint_artifact(),
    )

    assert isinstance(result.artifact, BlueprintNarrativeOutput)
    assert "`narrative_discovery`" in str(captured["prompt"])
    assert "DISCOVERY:" not in str(captured["prompt"])
    context_request = captured["context_request"]
    assert context_request.role == "builder"
    assert context_request.knowledge_access_backend == "hybrid"
    assert [item.key for item in context_request.inline_sources] == [
        "narrative_discovery",
        "narrative_canvas",
        "narrative_blueprint",
    ]
    assert result.knowledge_access_backend == "hybrid"
    assert result.effective_context_backend == "hybrid_workspace_staged"
    assert [item["key"] for item in result.context_used_sources] == [
        "narrative_discovery",
        "narrative_canvas",
        "narrative_blueprint",
    ]
    assert result.context_stats["reduction_estimated_tokens"] > 0


def test_codex_local_builder_reports_raw_inline_context_when_staging_is_disabled() -> None:
    runtime_settings = build_runtime_settings(
        active_provider=LLMProviderKey.codex_local,
        backend=AgentExecutionBackend.codex_cli,
    )
    service = CodexLocalBuilderService(runtime_settings)

    def fake_execute_structured_prompt(**kwargs):
        assert kwargs["context_request"] is None
        return sample_discovery_artifact()

    service.execution_service.execute_structured_prompt = fake_execute_structured_prompt  # type: ignore[method-assign]

    result = service.normalize_discovery(sample_discovery_input())

    assert isinstance(result.artifact, DiscoveryArtifact)
    assert result.knowledge_access_backend == "inline_context"
    assert result.effective_context_backend == "inline_context_raw"
    assert result.context_used_sources[0]["key"] == "discovery_capture"
    assert result.context_used_sources[0]["delivery_mode"] == "inline_raw"
    assert result.context_stats["reduction_estimated_tokens"] == 0


def test_facade_preserves_context_metadata_from_native_provider_results() -> None:
    runtime_settings = build_runtime_settings(
        active_provider=LLMProviderKey.openai,
        backend=AgentExecutionBackend.provider_native,
    )
    discovery_result = LLMArtifactResult(
        artifact=sample_discovery_artifact(),
        knowledge_access_backend="hybrid",
        effective_context_backend="hybrid_inline_compact",
        context_used_sources=[{"key": "discovery_capture"}],
        context_stats={"reduction_estimated_tokens": 84},
    )
    canvas = LLMArtifactResult(artifact=sample_canvas_artifact())
    narrative = LLMArtifactResult(artifact=BlueprintNarrativeOutput(narrative="OpenAI"))
    facade = BuilderProviderFacade(
        runtime_settings,
        openai_service=FakeBuilderService("openai", discovery_result, canvas, narrative),
        deepseek_service=FakeBuilderService("deepseek", discovery_result, canvas, narrative),
        codex_service=FakeBuilderService("codex_local", discovery_result, canvas, narrative),
    )

    result = facade.normalize_discovery(sample_discovery_input())

    assert result.provider_key == "openai"
    assert result.execution_backend == "provider_native"
    assert result.knowledge_access_backend == "hybrid"
    assert result.effective_context_backend == "hybrid_inline_compact"
    assert result.context_used_sources[0]["key"] == "discovery_capture"
    assert result.context_stats["reduction_estimated_tokens"] == 84


def test_facade_preserves_context_metadata_when_shadow_promotes_codex() -> None:
    runtime_settings = build_runtime_settings(
        active_provider=LLMProviderKey.openai,
        backend=AgentExecutionBackend.shadow_codex_cli,
    )
    openai_discovery = LLMArtifactResult(
        artifact=None,
        warning="OpenAI no disponible.",
        provider_key="openai",
        execution_backend="shadow_codex_cli",
        execution_mode="shadow",
    )
    codex_discovery = LLMArtifactResult(
        artifact=sample_discovery_artifact(),
        knowledge_access_backend="workspace_staged",
        effective_context_backend="workspace_staged_filesystem",
        context_used_sources=[{"key": "discovery_capture"}],
        context_stats={"reduction_estimated_tokens": 55},
    )
    canvas = LLMArtifactResult(artifact=sample_canvas_artifact())
    narrative = LLMArtifactResult(artifact=BlueprintNarrativeOutput(narrative="Codex"))
    facade = BuilderProviderFacade(
        runtime_settings,
        openai_service=FakeBuilderService("openai", openai_discovery, canvas, narrative),
        deepseek_service=FakeBuilderService("deepseek", codex_discovery, canvas, narrative),
        codex_service=FakeBuilderService("codex_local", codex_discovery, canvas, narrative),
    )

    result = facade.normalize_discovery(sample_discovery_input())

    assert isinstance(result.artifact, DiscoveryArtifact)
    assert result.provider_key == "codex_local"
    assert result.execution_mode == "shadow"
    assert "Shadow promovido" in (result.route_reason or "")
    assert result.knowledge_access_backend == "workspace_staged"
    assert result.effective_context_backend == "workspace_staged_filesystem"
    assert result.context_used_sources[0]["key"] == "discovery_capture"
    assert result.context_stats["reduction_estimated_tokens"] == 55
