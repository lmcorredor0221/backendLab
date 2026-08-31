from __future__ import annotations

from uuid import uuid4
from types import SimpleNamespace

import pytest

from app.services.agentic_runtime.action_registry import BuilderActionRejectedError, BuilderActionRegistry
from app.services.agentic_runtime import controller as react_controller_module
from app.services.agentic_runtime.contracts import BuilderActionRequest, BuilderActionResult, BuilderAgentRunRequest, BuilderAgentState
from app.services.agentic_runtime.controller import BuilderReActController
from app.services.agentic_runtime.guards import BuilderLoopGuardConfig, BuilderLoopGuardState, BuilderLoopGuards, BuilderLoopGuardViolation
from app.services.agentic_runtime.stage_policy import get_stage_agent_policy
from app.models import CanvasArtifact, DiscoveryArtifact, LLMRuntimeSettings
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionOutput
from app.services.llm_runtime.stage_context_types import StageContextBundle
from app.services.agentic_runtime.stages import define as define_stage
from app.services.agentic_runtime.cross_stage_evaluator import evaluate_tools_memory_compatibility
from app.services.agentic_runtime.stages.pipeline import ReactCapabilityOutput, run_react_stage
from app.services.agentic_runtime.tracing import build_react_metrics


def test_react_controller_completes_a_safe_define_dry_run() -> None:
    executed: list[str] = []

    def executor(action: BuilderActionRequest, _state: BuilderAgentState) -> BuilderActionResult:
        executed.append(action.key)
        if action.key == "run_validator":
            return BuilderActionResult(key=action.key, output={"issues": [], "blocking": False}, summary="valid")
        return BuilderActionResult(key=action.key, summary="ok")

    result = BuilderReActController().run(
        BuilderAgentRunRequest(stage="define", capability="define_requirements", mode="dry_run"),
        executor,
    )

    assert result.status == "completed"
    assert executed == ["retrieve_context", "invoke_capability", "run_validator"]
    assert [item.action.key for item in result.traces] == [
        "retrieve_context",
        "invoke_capability",
        "run_validator",
        "persist_stage_artifact",
        "finish_stage",
    ]
    assert all("chain" not in item.reason_summary.lower() for item in result.traces)


def test_react_controller_uses_stage_specific_total_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int] = {}
    original_guards = react_controller_module.BuilderLoopGuards

    class CapturingGuards(original_guards):
        def __init__(self, config: BuilderLoopGuardConfig | None = None) -> None:
            assert config is not None
            captured["max_total_ms"] = config.max_total_ms
            super().__init__(config)

    monkeypatch.setattr(react_controller_module, "BuilderLoopGuards", CapturingGuards)

    result = react_controller_module.BuilderReActController().run(
        BuilderAgentRunRequest(stage="memory", capability="recommend_memory_architecture", mode="dry_run"),
        lambda action, _state: BuilderActionResult(key=action.key, output={"issues": [], "blocking": False}),
    )

    assert result.status == "completed"
    assert captured["max_total_ms"] == get_stage_agent_policy("memory").max_total_ms
    assert captured["max_total_ms"] > 120_000


def test_react_controller_pauses_with_checkpoint_when_validation_blocks() -> None:
    def executor(action: BuilderActionRequest, _state: BuilderAgentState) -> BuilderActionResult:
        if action.key == "run_validator":
            return BuilderActionResult(
                key=action.key,
                output={"issues": ["Falta owner de decision"], "blocking": True},
                summary="blocked",
            )
        if action.key == "create_attention_decision":
            return BuilderActionResult(
                key=action.key,
                output={"issues": ["Falta owner de decision"], "blocking": True},
                summary="attention",
            )
        return BuilderActionResult(key=action.key, summary="ok")

    result = BuilderReActController().run(
        BuilderAgentRunRequest(
            session_id=uuid4(),
            workspace_id=uuid4(),
            stage="define",
            capability="define_requirements",
        ),
        executor,
    )

    assert result.status == "waiting_human"
    assert result.checkpoint_id.startswith("react:")
    assert [item.action.key for item in result.traces][-2:] == ["create_attention_decision", "checkpoint"]


def test_react_controller_retries_recoverable_capability_before_attention() -> None:
    attempts = 0
    executed: list[str] = []

    def executor(action: BuilderActionRequest, state: BuilderAgentState) -> BuilderActionResult:
        nonlocal attempts
        executed.append(action.key)
        if action.key == "invoke_capability":
            attempts += 1
            if attempts == 1:
                return BuilderActionResult(
                    key=action.key,
                    status="retryable",
                    output={"issues": ["provider timeout"], "blocking": True, "can_auto_retry": state.llm_calls < 1},
                    summary="provider failed",
                    error_kind="provider_or_schema_failure",
                )
            return BuilderActionResult(key=action.key, output={"artifact": {"ok": True}}, summary="retry ok")
        if action.key == "run_validator":
            return BuilderActionResult(key=action.key, output={"issues": [], "blocking": False}, summary="valid")
        return BuilderActionResult(key=action.key, summary="ok")

    result = BuilderReActController().run(
        BuilderAgentRunRequest(stage="define", capability="define_requirements"),
        executor,
    )

    assert result.status == "completed"
    assert attempts == 2
    assert executed[:3] == ["retrieve_context", "invoke_capability", "invoke_capability"]
    assert "create_attention_decision" not in executed


def test_react_controller_escalates_recoverable_capability_after_retry_budget() -> None:
    attempts = 0
    executed: list[str] = []

    def executor(action: BuilderActionRequest, state: BuilderAgentState) -> BuilderActionResult:
        nonlocal attempts
        executed.append(action.key)
        if action.key == "invoke_capability":
            attempts += 1
            return BuilderActionResult(
                key=action.key,
                status="retryable",
                output={
                    "issues": ["provider schema failure"],
                    "blocking": True,
                    "can_auto_retry": state.llm_calls < 1,
                },
                summary="provider failed",
                error_kind="provider_or_schema_failure",
            )
        if action.key == "create_attention_decision":
            return BuilderActionResult(
                key=action.key,
                output={"issues": ["provider schema failure"], "blocking": True},
                summary="attention",
            )
        return BuilderActionResult(key=action.key, summary="ok")

    result = BuilderReActController().run(
        BuilderAgentRunRequest(
            session_id=uuid4(),
            workspace_id=uuid4(),
            stage="define",
            capability="define_requirements",
        ),
        executor,
    )

    assert result.status == "waiting_human"
    assert attempts == 2
    assert executed[:4] == ["retrieve_context", "invoke_capability", "invoke_capability", "create_attention_decision"]


def test_action_registry_rejects_cross_stage_capability_and_unkeyed_side_effect() -> None:
    registry = BuilderActionRegistry()
    registry.assert_allowed(
        BuilderActionRequest(
            key="invoke_capability",
            stage="discover",
            capability="analyze_discovery",
        )
    )
    with pytest.raises(BuilderActionRejectedError, match="no pertenece"):
        registry.assert_allowed(
            BuilderActionRequest(
                key="invoke_capability",
                stage="define",
                capability="propose_agent_design",
            )
        )
    with pytest.raises(BuilderActionRejectedError, match="idempotency_key"):
        registry.assert_allowed(
            BuilderActionRequest(key="persist_stage_artifact", stage="define")
        )


def test_loop_guards_stop_repeated_actions() -> None:
    guards = BuilderLoopGuards(BuilderLoopGuardConfig(repeated_action_limit=2))
    runtime = BuilderLoopGuardState()
    state = BuilderAgentState(run_id=uuid4(), stage="define", capability="define_requirements")
    action = BuilderActionRequest(key="run_validator", stage="define")
    guards.record_action(runtime, action)
    guards.record_action(runtime, action)
    with pytest.raises(BuilderLoopGuardViolation, match="repitio"):
        guards.before_action(state=state, action=action, runtime=runtime)


def test_define_pilot_reuses_existing_skill_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    definition = RequirementsDefinitionOutput(summary="Propuesta Define de prueba", confidence=0.9)
    trace = SimpleNamespace(warnings=[], llm_trace=None)

    monkeypatch.setattr(
        define_stage,
        "run_definition_stage",
        lambda *_args, **_kwargs: (definition, [trace]),
    )
    monkeypatch.setattr(define_stage, "validate_definition_artifact", lambda value: value)

    context = StageContextBundle(
        capability="define_requirements",
        role="builder",
        stage="define",
        workspace_id=uuid4(),
        session_id=uuid4(),
        session_snapshot=None,
        effective_language="es",
        knowledge_manifest=None,
        memory_policy=None,
        short_term_memory=None,
    )
    result = define_stage.run_define_react(
        discovery=DiscoveryArtifact(problem_statement="Test"),
        canvas=CanvasArtifact(user_goal="Test"),
        runtime_settings=LLMRuntimeSettings(),
        stage_context=context,
        session_id=uuid4(),
        workspace_id=uuid4(),
    )

    assert result.react_run is not None
    assert result.react_run.status == "completed"
    assert result.skill_traces == [trace]


def test_extended_pipeline_records_proposal_and_critique_before_persisting() -> None:
    result = run_react_stage(
        stage="design",
        capability="propose_agent_design",
        secondary_capability="critique_agent_design",
        session_id=uuid4(),
        workspace_id=uuid4(),
        context_refs=["session.define", "knowledge.design"],
        primary_runner=lambda: ReactCapabilityOutput(value={"recommended": "react"}, summary="proposal"),
        secondary_runner=lambda value: ReactCapabilityOutput(value={**value, "critic": "accepted"}, summary="critique"),
        validator=lambda value: ([], False, "valid"),
    )

    assert result.react_run is not None
    assert result.react_run.status == "completed"
    assert [item.action.key for item in result.react_run.traces] == [
        "retrieve_context",
        "invoke_capability",
        "invoke_critique",
        "run_validator",
        "persist_stage_artifact",
        "finish_stage",
    ]
    assert result.value["critic"] == "accepted"


def test_quality_gate_requests_bounded_repair_for_low_quality_output() -> None:
    attempts = 0

    def run_stage() -> ReactCapabilityOutput:
        nonlocal attempts
        attempts += 1
        confidence = 0.52 if attempts == 1 else 0.91
        return ReactCapabilityOutput(
            value=SimpleNamespace(confidence=SimpleNamespace(overall=confidence)),
            summary=f"attempt {attempts}",
        )

    result = run_react_stage(
        stage="estimate",
        capability="analyze_estimation_risks",
        session_id=uuid4(),
        workspace_id=uuid4(),
        context_refs=["session.memory", "knowledge.estimation"],
        primary_runner=run_stage,
        validator=lambda _value: ([], False, "valid"),
    )

    assert result.react_run is not None
    assert result.react_run.status == "completed"
    assert attempts == 3
    assert result.react_run.state.quality_repair_cycles == 2
    assert result.react_run.output["quality_gate"]["quality_confidence"] >= 0.85
    assert "create_attention_decision" not in [item.action.key for item in result.react_run.traces]


def test_quality_gate_delegates_free_questions_without_quality_penalty() -> None:
    result = run_react_stage(
        stage="design",
        capability="propose_agent_design",
        session_id=uuid4(),
        workspace_id=uuid4(),
        context_refs=["session.define", "knowledge.design"],
        primary_runner=lambda: ReactCapabilityOutput(
            value=SimpleNamespace(
                confidence=SimpleNamespace(overall=0.58),
                open_questions=["Confirmar proveedor de autenticacion en Blueprint Pro."],
            ),
            summary="proposal with delegated question",
        ),
        validator=lambda _value: ([], False, "valid"),
    )

    assert result.react_run is not None
    assert result.react_run.status == "completed"
    gate = result.react_run.output["quality_gate"]
    assert gate["repair_policy"] == "document_and_delegate"
    assert gate["quality_confidence"] >= 0.85
    assert gate["evidence_confidence"] < gate["quality_confidence"]
    assert result.react_run.state.quality_repair_cycles == 0


def test_quality_gate_repairs_language_mismatch_for_user_visible_output() -> None:
    attempts = 0

    def run_memory() -> ReactCapabilityOutput:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ReactCapabilityOutput(
                value={
                    "summary": (
                        "The memory strategy should keep short term context, long term facts, "
                        "and pending user decisions before the system continues with estimation."
                    ),
                    "confidence": {"overall": 0.91},
                },
                summary="english memory",
            )
        return ReactCapabilityOutput(
            value={
                "summary": (
                    "La estrategia de memoria debe conservar contexto corto, hechos de largo plazo "
                    "y decisiones pendientes antes de continuar con estimacion."
                ),
                "confidence": {"overall": 0.91},
            },
            summary="spanish memory",
        )

    result = run_react_stage(
        stage="memory",
        capability="recommend_memory_architecture",
        session_id=uuid4(),
        workspace_id=uuid4(),
        context_refs=["session.tools", "knowledge.memory"],
        primary_runner=run_memory,
        validator=lambda _value: ([], False, "valid"),
        effective_language="es",
    )

    assert result.react_run is not None
    assert result.react_run.status == "completed"
    assert attempts == 3
    assert result.react_run.output["quality_gate"]["language_status"] == "ok"
    assert result.react_run.state.quality_repair_cycles == 2


def test_quality_gate_repairs_required_context_truncation_once() -> None:
    attempts = 0

    def run_design() -> ReactCapabilityOutput:
        nonlocal attempts
        attempts += 1
        warnings = ["required_truncated:session.discovery"] if attempts == 1 else []
        return ReactCapabilityOutput(
            value={"summary": "Diseno con contexto suficiente.", "confidence": {"overall": 0.9}},
            summary=f"context attempt {attempts}",
            warnings=warnings,
        )

    result = run_react_stage(
        stage="design",
        capability="propose_agent_design",
        session_id=uuid4(),
        workspace_id=uuid4(),
        context_refs=["session.discovery", "knowledge.design"],
        primary_runner=run_design,
        validator=lambda _value: ([], False, "valid"),
        effective_language="es",
    )

    assert result.react_run is not None
    assert result.react_run.status == "completed"
    assert attempts == 3
    assert result.react_run.output["quality_gate"]["quality_confidence"] == 0.9
    assert result.react_run.state.quality_repair_cycles == 2


def test_quality_gate_pauses_after_repair_budget_when_quality_remains_low() -> None:
    attempts = 0

    def run_stage() -> ReactCapabilityOutput:
        nonlocal attempts
        attempts += 1
        return ReactCapabilityOutput(
            value=SimpleNamespace(confidence=SimpleNamespace(overall=0.52)),
            summary=f"attempt {attempts}",
        )

    result = run_react_stage(
        stage="estimate",
        capability="analyze_estimation_risks",
        session_id=uuid4(),
        workspace_id=uuid4(),
        context_refs=["session.memory", "knowledge.estimation"],
        primary_runner=run_stage,
        validator=lambda _value: ([], False, "valid"),
    )

    assert result.react_run is not None
    assert result.react_run.status == "waiting_human"
    assert attempts == 3
    assert result.react_run.state.quality_repair_cycles == 2
    assert "create_attention_decision" in [item.action.key for item in result.react_run.traces]
    assert result.react_run.output["quality_gate"]["repair_policy"] == "attention_required"


def test_tools_memory_evaluator_blocks_rag_without_ingestion_and_retrieval() -> None:
    tools = SimpleNamespace(
        approved_tools_digest=SimpleNamespace(
            approved_tool_keys=["read_system_of_record"],
            knowledge_tool_keys=[],
        )
    )
    memory = SimpleNamespace(knowledge_design=SimpleNamespace(rag_required=True, mode="hybrid"))

    issues, blocking, _summary = evaluate_tools_memory_compatibility(tools, memory)

    assert blocking is True
    assert "document_ingestion" in issues[0]
    assert "knowledge_retrieval" in issues[0]


def test_tools_memory_remediation_uses_a_cross_stage_action() -> None:
    result = run_react_stage(
        stage="memory",
        capability="recommend_memory_architecture",
        session_id=uuid4(),
        workspace_id=uuid4(),
        context_refs=["session.tools", "knowledge.memory"],
        primary_runner=lambda: ReactCapabilityOutput(value={"rag": True}, summary="memory"),
        validator=lambda _value: (["Tools debe aprobar document_ingestion"], True, "cross-stage"),
        remediation_action="raise_cross_stage_remediation",
    )

    assert result.react_run is not None
    assert result.react_run.status == "waiting_human"
    assert "raise_cross_stage_remediation" in [item.action.key for item in result.react_run.traces]


def test_react_metrics_are_operational_and_do_not_expose_hidden_reasoning() -> None:
    result = BuilderReActController().run(
        BuilderAgentRunRequest(stage="estimate", capability="analyze_estimation_risks", mode="dry_run"),
        lambda action, _state: BuilderActionResult(key=action.key, output={"issues": [], "blocking": False}),
    )

    metrics = build_react_metrics(result)
    serialized = str(metrics).lower()
    assert metrics["status"] == "completed"
    assert metrics["iterations"] > 0
    assert "chain-of-thought" not in serialized
    assert "reasoning" not in serialized
