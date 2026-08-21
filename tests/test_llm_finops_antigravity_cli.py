from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    AgentExecutionBackend,
    AntigravityProviderConfig,
    CanvasArtifact,
    DiscoveryArtifact,
    LLMProviderKey,
    LLMRuntimeSettings,
    LLMUsageLedgerRecord,
)
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_runtime.antigravity_cli.provider_facade import AntigravityLocalBuilderService
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionInput, RequirementsDefinitionOutput
from app.services.llm_runtime.stage_context_types import StageContextBundle


def build_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    @contextmanager
    def session_scope():
        with Session(engine) as session:
            yield session

    return engine, session_scope


def build_runtime_settings() -> LLMRuntimeSettings:
    return LLMRuntimeSettings(
        active_provider=LLMProviderKey.antigravity_cli,
        agent_execution_backend=AgentExecutionBackend.antigravity_cli,
        antigravity=AntigravityProviderConfig(
            executable="agy",
            model="gemini-3.6-flash",
            effort="high",
            runner_id="agy-finops",
            executable_found=True,
            available=True,
            fallback_models=["gemini-3.6-pro"],
        ),
    )


def build_stage_context() -> StageContextBundle:
    return StageContextBundle(
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
        context_fingerprint="ctx-agy-finops",
    )


def build_requirements_input() -> RequirementsDefinitionInput:
    return RequirementsDefinitionInput(
        discovery=DiscoveryArtifact(problem_statement="Automatizar discovery."),
        canvas=CanvasArtifact(user_goal="Generar requisitos trazables."),
    )


def build_audit_payload(**overrides):
    payload = {
        "run_id": "agy-run-1",
        "status": "succeeded",
        "requested_model": "gemini-3.6-flash",
        "selected_model": "gemini-3.6-pro",
        "attempted_models": ["gemini-3.6-flash", "gemini-3.6-pro"],
        "fallback_used": True,
        "started_at": "2026-08-13T10:00:00+00:00",
        "finished_at": "2026-08-13T10:00:05+00:00",
        "returncode": 0,
        "attempts": [{"attempt_number": 1}, {"attempt_number": 2}],
        "metrics": {
            "duration_ms": 5000,
            "queue_wait_ms": 45,
            "output_size_bytes": 196,
            "stdout_bytes": 320,
            "stderr_bytes": 0,
            "exit_code": 0,
        },
    }
    payload.update(overrides)
    return payload


class FakeAgyExecutionService:
    def __init__(self, *, audit: dict, error: Exception | None = None) -> None:
        self.audit = audit
        self.error = error

    def execute_structured_prompt(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return RequirementsDefinitionOutput(summary="Requisitos desde Antigravity.")

    def read_last_known_result(self):
        return self.audit


def test_antigravity_cli_records_successful_cli_audit_as_estimated_usage() -> None:
    engine, session_factory = build_session_factory()
    service = AntigravityLocalBuilderService(
        build_runtime_settings(),
        finops_session_factory=session_factory,
        finops_ledger_service=LLMUsageLedgerService(),
    )
    service.execution_service = FakeAgyExecutionService(audit=build_audit_payload())

    result = service.define_requirements(build_requirements_input(), context_bundle=build_stage_context())

    with Session(engine) as db:
        record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert isinstance(result.artifact, RequirementsDefinitionOutput)
    assert result.usage_record_id is not None
    assert result.normalized_usage is not None
    assert result.normalized_usage.usage_is_estimated is True
    assert result.fallback_used is True
    assert result.retry_count == 1
    assert record is not None
    assert record.status == "succeeded"
    assert record.provider_key == "antigravity_cli"
    assert record.model_name == "gemini-3.6-pro"
    assert record.request_id == "agy-run-1"
    assert record.duration_ms == 5000
    assert record.queue_wait_ms == 45
    assert record.provider_metrics["selected_model"] == "gemini-3.6-pro"
    assert record.provider_metrics["fallback_used"] is True
    assert record.total_tokens > 0


def test_antigravity_cli_records_failed_cli_audit() -> None:
    engine, session_factory = build_session_factory()
    service = AntigravityLocalBuilderService(
        build_runtime_settings(),
        finops_session_factory=session_factory,
        finops_ledger_service=LLMUsageLedgerService(),
    )
    service.execution_service = FakeAgyExecutionService(
        audit=build_audit_payload(
            run_id="agy-run-failed",
            status="failed",
            selected_model="gemini-3.6-flash",
            attempted_models=["gemini-3.6-flash"],
            fallback_used=False,
            attempts=[{"attempt_number": 1}],
            metrics={
                "duration_ms": 1500,
                "queue_wait_ms": 15,
                "output_size_bytes": 0,
                "stdout_bytes": 0,
                "stderr_bytes": 80,
                "exit_code": 1,
            },
        ),
        error=RuntimeError("agy failed"),
    )

    result = service.define_requirements(build_requirements_input(), context_bundle=build_stage_context())

    with Session(engine) as db:
        record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert result.artifact is None
    assert result.failure_kind == "provider_error"
    assert result.usage_record_id is not None
    assert record is not None
    assert record.status == "failed"
    assert record.provider_key == "antigravity_cli"
    assert record.request_id == "agy-run-failed"
    assert record.failure_kind == "provider_error"
    assert "agy failed" in record.failure_detail_redacted
    assert record.provider_metrics["status"] == "failed"
    assert record.other_token_metrics["usage_is_estimated"] is True
