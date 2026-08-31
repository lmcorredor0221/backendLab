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
from app.services.diagram_center.contracts import DiagramGenerationInput, DiagramNotation, StructuredDiagramModel
from app.services.llm_runtime.builder_contracts import BlueprintNarrativeOutput, RequirementsDefinitionInput
from app.services.llm_runtime.capability_registry import BuilderCapability
from app.services.openai_builder import (
    DeepSeekBuilderService,
    OpenAIBuilderService,
    _serialize_capability_payload_for_api,
    _structured_capability_max_tokens,
)


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
    assert result.context_stats["assembled_estimated_tokens"] > 0
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
    assert result.context_stats["assembled_estimated_tokens"] > 0
    assert "DISCOVERY=" not in user_payload
    assert "[source] narrative_discovery" in user_payload
    assert len(user_payload) < len(baseline_user_payload)


def test_deepseek_api_context_does_not_instruct_filesystem_reads_when_workspace_staged() -> None:
    runtime_settings = build_runtime_settings(
        LLMProviderKey.deepseek,
        backend=KnowledgeAccessBackend.workspace_staged,
    )
    service = DeepSeekBuilderService(runtime_settings)
    service._client = object()
    discovery = sample_discovery_artifact(suffix="Y" * 6000)
    canvas = sample_canvas_artifact(suffix="Z" * 5000)
    blueprint = sample_blueprint_artifact(suffix="W" * 4000)
    captured: dict[str, object] = {}

    def fake_create_structured_completion(**kwargs):
        captured.update(kwargs)
        return BlueprintNarrativeOutput(narrative="Narrativa compacta")

    service._create_structured_completion = fake_create_structured_completion  # type: ignore[method-assign]

    result = service.synthesize_blueprint_narrative(discovery, canvas, blueprint)

    user_payload = str(captured["user_payload"])

    assert result.effective_context_backend == "workspace_staged_unavailable_inline_compact"
    assert result.context_stats["api_context_contract"] == "provider_api_inline.v1"
    assert "knowledge/required" not in user_payload
    assert "no intentes leer archivos locales" in user_payload
    assert "usa exclusivamente las fuentes inline" in user_payload


def test_complex_capabilities_get_larger_structured_output_budget_without_expanding_bpmn() -> None:
    assert _structured_capability_max_tokens(BuilderCapability.propose_agent_design) == 6144
    assert _structured_capability_max_tokens(BuilderCapability.critique_agent_design) == 6144
    assert _structured_capability_max_tokens(BuilderCapability.recommend_memory_architecture) == 6144
    assert _structured_capability_max_tokens(BuilderCapability.critique_memory_architecture) == 6144

    bpmn_payload = DiagramGenerationInput(
        diagram_key="current_process_map",
        title="Proceso actual",
        objective="Representar flujo actual",
        notation=DiagramNotation.bpmn,
    )

    assert _structured_capability_max_tokens(
        BuilderCapability.generate_diagram_model,
        payload=bpmn_payload,
    ) == 4096


def test_capability_payload_serializer_compacts_requirements_definition_for_api() -> None:
    payload = RequirementsDefinitionInput(
        discovery=sample_discovery_artifact(suffix="D" * 8000),
        canvas=sample_canvas_artifact(suffix="C" * 8000),
        known_constraints=[
            "Mantener trazabilidad end-to-end",
            "Evitar drift entre discovery, define y design",
            "No perder evidencia aprobada del workspace",
            "Operar con contexto gobernado por etapa",
            "Respetar guardrails operativos del cliente",
            "Reducir retrabajo arquitectonico",
            "No inventar decisiones sin evidencia",
        ],
        source_refs=["session.discovery", "session.canvas", "session.definition_seed", "workspace.knowledge"],
    )

    raw_payload = json.dumps(payload.model_dump(mode="json"), ensure_ascii=True)
    compact_payload = _serialize_capability_payload_for_api(payload)
    compact_payload_json = json.dumps(compact_payload, ensure_ascii=True)

    assert len(compact_payload_json) < len(raw_payload)
    assert compact_payload["discovery"]["current_process"].endswith("...")
    assert len(compact_payload["known_constraints"]) == 7
    assert compact_payload["source_refs"] == [
        "session.discovery",
        "session.canvas",
        "session.definition_seed",
        "workspace.knowledge",
    ]


def test_openai_builder_compacts_diagram_payload_to_resolved_inputs() -> None:
    runtime_settings = build_runtime_settings(
        LLMProviderKey.openai,
        backend=KnowledgeAccessBackend.inline_context,
    )
    service = OpenAIBuilderService(runtime_settings)
    payload = DiagramGenerationInput(
        diagram_key="architecture_overview",
        title="Arquitectura propuesta",
        objective="Mostrar la arquitectura aprobada.",
        notation=DiagramNotation.flowchart,
        required_inputs=["blueprint.architecture_spec", "blueprint.patterns"],
        resolved_inputs=[
            {
                "input_key": "blueprint.architecture_spec",
                "status": "resolved",
                "matched_artifact_keys": ["design_recommendation_artifact"],
                "artifact_refs": ["journey:design:v1"],
                "evidence": [
                    {
                        "artifact_key": "design_recommendation_artifact",
                        "ref": "journey:design:v1",
                        "content": {
                            "summary": "Arquitectura aprobada con supervision y handoffs.",
                            "selected_design": {"architecture_pattern": "supervisor_with_specialists"},
                        },
                    }
                ],
            }
        ],
        source_context={
            "project": {"id": "session-1", "title": "Architecture project"},
            "coverage_summary": {"required_input_count": 2, "resolved_input_count": 1, "missing_input_count": 1},
            "resolved_inputs": [
                {
                    "input_key": "blueprint.architecture_spec",
                    "status": "resolved",
                }
            ],
            "missing_required_inputs": ["blueprint.patterns"],
            "approved_artifact_keys": ["design_recommendation_artifact"],
            "approved_artifacts": [
                {
                    "key": "design_recommendation_artifact",
                    "content": {"marker": "SHOULD_NOT_BE_IN_API_PAYLOAD", "blob": "X" * 12000},
                }
            ],
        },
        source_refs=["journey:design:v1"],
    )
    captured: dict[str, object] = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp-diagram-openai-1",
                status="completed",
                output_parsed=StructuredDiagramModel(
                    diagram_key="architecture_overview",
                    title="Arquitectura propuesta",
                    notation="flowchart",
                    nodes=[],
                    edges=[],
                    source_refs=["journey:design:v1"],
                ),
                usage=None,
            )

    service._client = SimpleNamespace(responses=FakeResponses())

    result = service.generate_diagram_model(payload)

    user_payload = str(captured["input"][1]["content"])
    baseline_payload = json.dumps(payload.model_dump(mode="json"), ensure_ascii=True)

    assert isinstance(result.artifact, StructuredDiagramModel)
    assert "resolved_inputs" in user_payload
    assert "blueprint.architecture_spec" in user_payload
    assert "SHOULD_NOT_BE_IN_API_PAYLOAD" not in user_payload
    assert len(user_payload) < len(baseline_payload)


def test_deepseek_bpmn_retry_switches_to_compact_mode_and_repairs_bpmn_terminals() -> None:
    runtime_settings = build_runtime_settings(
        LLMProviderKey.deepseek,
        backend=KnowledgeAccessBackend.inline_context,
    )
    service = DeepSeekBuilderService(runtime_settings)
    payload = DiagramGenerationInput(
        diagram_key="current_process_map",
        title="Proceso actual",
        objective="Representar el flujo actual.",
        notation=DiagramNotation.bpmn,
        source_refs=["journey:discover:v1"],
    )

    class FakeSequentialCompletionsAPI:
        def __init__(self, responses: list[object]) -> None:
            self.responses = list(responses)
            self.kwargs_history: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.kwargs_history.append(dict(kwargs))
            if not self.responses:
                raise AssertionError("No quedan respuestas fake para DeepSeek.")
            return self.responses.pop(0)

    truncated = SimpleNamespace(
        id="chatcmpl-deepseek-diagram-1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"diagram_key":"current_process_map"'),
                finish_reason="length",
            )
        ],
        usage=None,
    )
    valid_payload = {
        "diagram_key": "current_process_map",
        "title": "Proceso actual",
        "notation": "bpmn",
        "nodes": [
            {
                "id": "identify_context",
                "label": "Identificar contexto",
                "kind": "task",
                "metadata": {"pool_id": "pool_ops", "lane_id": "lane_analyst", "attributes": []},
                "source_refs": ["journey:discover:v1"],
            },
            {
                "id": "review_code",
                "label": "Revisar codigo fuente",
                "kind": "task",
                "metadata": {"pool_id": "pool_ops", "lane_id": "lane_analyst", "attributes": []},
                "source_refs": ["journey:discover:v1"],
            },
        ],
        "edges": [
            {
                "id": "flow_1",
                "source": "identify_context",
                "target": "review_code",
                "kind": "sequence_flow",
                "source_refs": ["journey:discover:v1"],
            }
        ],
        "pools": [
            {
                "id": "pool_ops",
                "label": "Operacion",
                "lanes": [
                    {
                        "id": "lane_analyst",
                        "label": "Analista",
                        "source_refs": ["journey:discover:v1"],
                    }
                ],
                "source_refs": ["journey:discover:v1"],
            }
        ],
        "source_refs": ["journey:discover:v1"],
    }
    completed = SimpleNamespace(
        id="chatcmpl-deepseek-diagram-2",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(valid_payload, ensure_ascii=True)),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    completions = FakeSequentialCompletionsAPI([truncated, completed])
    service._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = service.generate_diagram_model(payload)

    assert isinstance(result.artifact, StructuredDiagramModel)
    assert result.retry_count == 1
    assert result.schema_validation_status == "repaired_bpmn_terminals"
    assert any(node.kind == "start_event" for node in result.artifact.nodes)
    assert any(node.kind == "end_event" for node in result.artifact.nodes)
    assert completions.kwargs_history[0]["reasoning_effort"] == "high"
    assert completions.kwargs_history[0]["max_tokens"] == 4096
    assert completions.kwargs_history[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "reasoning_effort" not in completions.kwargs_history[1]
    assert completions.kwargs_history[1]["max_tokens"] == 4096
    assert completions.kwargs_history[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "minimo numero de nodos" in str(completions.kwargs_history[1]["messages"][-1]["content"])
