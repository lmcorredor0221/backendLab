from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from app.models import (
    AgentExecutionBackend,
    CanvasArtifact,
    DiscoveryArtifact,
    DiscoveryInput,
    LLMProviderKey,
    LLMRuntimeSettings,
)
from app.services.llm_runtime.builder_contracts import LLMArtifactResult
from app.services.llm_runtime.provider_router import BuilderProviderFacade
from app.services.llm_runtime.stage_context_types import StageContextBundle, build_llm_call_context


def sample_discovery_input() -> DiscoveryInput:
    return DiscoveryInput(
        problem_statement="Estandarizar discovery para un builder Lean.",
        current_user="Arquitecto de soluciones",
        current_process="Consolida inputs dispersos.",
        desired_outcome="Generar discovery consistente.",
        autonomy_level="high",
        constraints=["Sin side effects irreversibles"],
        operational_baseline={
            "current_time_spent": "4 horas",
            "current_cost": "Retrabajo",
            "frequent_errors": ["Drift del objetivo"],
            "automation_opportunities": ["Normalizar discovery"],
        },
        mvp_definition={
            "v1_scope": ["Discovery", "Canvas"],
            "out_of_scope": ["Provisioning"],
            "north_star_metric": "Discovery util",
            "non_delegable_decisions": ["Aprobar promocion"],
        },
    )


def sample_discovery_artifact() -> DiscoveryArtifact:
    return DiscoveryArtifact(
        problem_statement="Estandarizar discovery para un builder Lean.",
        current_user="Arquitecto de soluciones",
        current_process="Consolida inputs dispersos.",
        desired_outcome="Generar discovery consistente.",
        autonomy_level="high",
        constraints=["Sin side effects irreversibles"],
        operational_baseline={
            "current_time_spent": "4 horas",
            "current_cost": "Retrabajo",
            "frequent_errors": ["Drift del objetivo"],
            "automation_opportunities": ["Normalizar discovery"],
        },
        mvp_definition={
            "v1_scope": ["Discovery", "Canvas"],
            "out_of_scope": ["Provisioning"],
            "north_star_metric": "Discovery util",
            "non_delegable_decisions": ["Aprobar promocion"],
        },
        case_type="automatizacion",
        value_statement="Reducir retrabajo del equipo.",
    )


def sample_canvas_artifact() -> CanvasArtifact:
    return CanvasArtifact(
        user_goal="Generar discovery consistente.",
        mvp_scope=["Discovery", "Canvas"],
        out_of_scope=["Provisioning"],
        success_metric="Discovery util",
        primary_risk="Drift del objetivo",
        agent_profile={
            "mission": "Normalizar discovery.",
            "primary_user": "Arquitecto de soluciones",
            "agent_task": "Estructurar discovery y canvas base.",
            "allowed_decisions": ["Proponer estructura"],
            "prohibited_decisions": ["Ejecutar side effects"],
            "key_inputs": ["Problema", "Proceso actual"],
            "expected_outputs": ["Discovery", "Canvas"],
            "human_approvals": ["Promocion a implementacion"],
            "success_metrics": ["Discovery util"],
        },
    )


def build_stage_context(
    *,
    capability: str = "normalize_discovery",
    stage: str = "discover",
) -> StageContextBundle:
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
        context_fingerprint="ctx-finops-001",
        corpus_hash="corp-finops-001",
        retrieval_pages=2,
    )


@dataclass
class FakeProviderService:
    provider_key: str
    result: LLMArtifactResult

    def can_attempt(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def provider_summary(self) -> dict[str, str | bool]:
        return {
            "provider": self.provider_key,
            "configured": True,
            "sdk_ready": True,
        }

    def normalize_discovery(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return self.result

    def build_canvas(self, discovery, *, context_bundle=None) -> LLMArtifactResult:
        del discovery, context_bundle
        return self.result


def test_build_llm_call_context_maps_stage_bundle_without_required_business_ids() -> None:
    bundle = build_stage_context()

    context = build_llm_call_context(
        bundle,
        capability="normalize_discovery",
        provider_key="openai",
        execution_backend="provider_native",
        execution_mode="primary",
    )

    assert context.workspace_id == bundle.workspace_id
    assert context.session_id == bundle.session_id
    assert context.user_id is None
    assert context.project_id is None
    assert context.initiative_id is None
    assert context.stage == "discover"
    assert context.agent_key == "builder"
    assert context.capability_key == "normalize_discovery"
    assert context.action_key == "normalize_discovery"
    assert context.execution_mode == "primary"
    assert context.correlation_id == "ctx-finops-001"
    assert context.metadata["provider_route"] == "primary:openai"
    assert context.metadata["corpus_hash"] == "corp-finops-001"


def test_provider_facade_attaches_finops_context_to_primary_result() -> None:
    bundle = build_stage_context()
    settings = LLMRuntimeSettings(
        active_provider=LLMProviderKey.openai,
        agent_execution_backend=AgentExecutionBackend.provider_native,
    )
    result = LLMArtifactResult(artifact=sample_discovery_artifact())
    facade = BuilderProviderFacade(
        settings,
        openai_service=FakeProviderService("openai", result),
        deepseek_service=FakeProviderService("deepseek", result),
        codex_service=FakeProviderService("codex_local", result),
    )

    llm_result = facade.normalize_discovery(sample_discovery_input(), context_bundle=bundle)

    assert llm_result.finops_context is not None
    assert llm_result.finops_context.workspace_id == bundle.workspace_id
    assert llm_result.finops_context.session_id == bundle.session_id
    assert llm_result.finops_context.capability_key == "normalize_discovery"
    assert llm_result.finops_context.stage == "discover"
    assert llm_result.finops_context.metadata["provider_key"] == "openai"
    assert llm_result.finops_context.metadata["execution_backend"] == "provider_native"
    assert llm_result.finops_context.metadata["provider_route"] == "primary:openai"


def test_provider_facade_attaches_finops_context_to_shadow_route() -> None:
    bundle = build_stage_context(capability="build_canvas", stage="define")
    settings = LLMRuntimeSettings(
        active_provider=LLMProviderKey.openai,
        agent_execution_backend=AgentExecutionBackend.shadow_codex_cli,
    )
    canvas_result = LLMArtifactResult(artifact=sample_canvas_artifact())
    facade = BuilderProviderFacade(
        settings,
        openai_service=FakeProviderService("openai", canvas_result),
        deepseek_service=FakeProviderService("deepseek", canvas_result),
        codex_service=FakeProviderService("codex_local", canvas_result),
    )

    llm_result = facade.build_canvas(sample_discovery_artifact(), context_bundle=bundle)

    assert llm_result.provider_key == "openai"
    assert llm_result.execution_mode == "shadow"
    assert llm_result.shadow_provider_key == "codex_local"
    assert llm_result.finops_context is not None
    assert llm_result.finops_context.execution_mode == "shadow"
    assert llm_result.finops_context.capability_key == "build_canvas"
    assert llm_result.finops_context.metadata["provider_route"] == "shadow:openai"
    assert llm_result.finops_context.metadata["shadow_provider_key"] == "codex_local"
