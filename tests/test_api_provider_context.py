from __future__ import annotations

import json
from types import SimpleNamespace

from app.models import (
    AgentExecutionBackend,
    BlueprintArtifact,
    CanvasArtifact,
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
from app.services.llm_runtime.builder_contracts import BlueprintNarrativeOutput
from app.services.openai_builder import DeepSeekBuilderService, OpenAIBuilderService


def build_runtime_settings(active_provider: LLMProviderKey, *, backend: KnowledgeAccessBackend) -> LLMRuntimeSettings:
    return LLMRuntimeSettings(
        active_provider=active_provider,
        agent_execution_backend=AgentExecutionBackend.provider_native,
        knowledge_access_backend=backend,
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
            profile="api-context-test",
            executable_found=True,
            available=True,
        ),
    )


def sample_discovery_input(*, suffix: str = "") -> DiscoveryInput:
    return DiscoveryInput(
        problem_statement="Estandarizar discovery para un builder Lean.",
        current_user="Arquitecto de soluciones",
        current_process=f"Consolida inputs dispersos {suffix}",
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


def sample_discovery_artifact(*, suffix: str = "") -> DiscoveryArtifact:
    payload = sample_discovery_input(suffix=suffix).model_dump(mode="json")
    payload.update({"case_type": "automatizacion", "value_statement": "Reducir retrabajo del equipo."})
    return DiscoveryArtifact.model_validate(payload)


def sample_canvas_artifact(*, suffix: str = "") -> CanvasArtifact:
    return CanvasArtifact(
        user_goal="Generar discovery consistente con menos retrabajo.",
        mvp_scope=["Discovery", "Canvas"],
        out_of_scope=["Provisioning"],
        success_metric="Discovery util en una sesion",
        primary_risk=f"Drift del objetivo {suffix}",
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


def sample_blueprint_artifact(*, suffix: str = "") -> BlueprintArtifact:
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
        narrative=f"Blueprint base. {suffix}",
    )


def test_openai_builder_uses_compact_context_and_reports_used_sources_for_discovery() -> None:
    runtime_settings = build_runtime_settings(
        LLMProviderKey.openai,
        backend=KnowledgeAccessBackend.hybrid,
    )
    service = OpenAIBuilderService(runtime_settings)
    payload = sample_discovery_input(suffix="X" * 12000)
    raw_payload = json.dumps(payload.model_dump(mode="json"), ensure_ascii=True)
    parsed_artifact = sample_discovery_artifact()
    captured: dict[str, object] = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=parsed_artifact)

    service._client = SimpleNamespace(responses=FakeResponses())

    result = service.normalize_discovery(payload)

    user_payload = str(captured["input"][1]["content"])

    assert isinstance(result.artifact, DiscoveryArtifact)
    assert result.knowledge_access_backend == "hybrid"
    assert result.effective_context_backend == "hybrid_inline_compact"
    assert result.context_used_sources[0]["key"] == "discovery_capture"
    assert result.context_stats["reduction_estimated_tokens"] > 0
    assert "Context sources:" in user_payload
    assert len(user_payload) < len(raw_payload)


def test_deepseek_builder_uses_compact_context_for_narrative_and_reduces_inline_payload() -> None:
    runtime_settings = build_runtime_settings(
        LLMProviderKey.deepseek,
        backend=KnowledgeAccessBackend.inline_context,
    )
    service = DeepSeekBuilderService(runtime_settings)
    service._client = object()
    discovery = sample_discovery_artifact(suffix="Y" * 6000)
    canvas = sample_canvas_artifact(suffix="Z" * 5000)
    blueprint = sample_blueprint_artifact(suffix="W" * 4000)
    captured: dict[str, object] = {}

    baseline_user_payload = (
        "Genera la narrativa en json estructurado con base en estos artefactos:\n"
        f"DISCOVERY={json.dumps(discovery.model_dump(mode='json'), ensure_ascii=True)}\n"
        f"CANVAS={json.dumps(canvas.model_dump(mode='json'), ensure_ascii=True)}\n"
        f"BLUEPRINT={json.dumps(blueprint.model_dump(mode='json'), ensure_ascii=True)}"
    )

    def fake_create_structured_completion(**kwargs):
        captured.update(kwargs)
        return BlueprintNarrativeOutput(narrative="Narrativa compacta")

    service._create_structured_completion = fake_create_structured_completion  # type: ignore[method-assign]

    result = service.synthesize_blueprint_narrative(discovery, canvas, blueprint)

    user_payload = str(captured["user_payload"])

    assert isinstance(result.artifact, BlueprintNarrativeOutput)
    assert result.knowledge_access_backend == "inline_context"
    assert result.effective_context_backend == "inline_context_compact"
    assert [item["key"] for item in result.context_used_sources] == [
        "narrative_discovery",
        "narrative_canvas",
        "narrative_blueprint",
    ]
    assert result.context_stats["reduction_estimated_tokens"] > 0
    assert "DISCOVERY=" not in user_payload
    assert "[source] narrative_discovery" in user_payload
    assert len(user_payload) < len(baseline_user_payload)
