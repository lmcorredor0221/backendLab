from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    AgentExecutionBackend,
    CanvasArtifact,
    DiscoveryArtifact,
    LLMProviderKey,
    LLMRuntimeSettings,
    LLMUsageLedgerRecord,
    OpenAIProviderConfig,
)
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_runtime.capability_registry import BuilderCapability, get_builder_capability_spec
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionInput, RequirementsDefinitionOutput
from app.services.llm_runtime.stage_context_types import StageContextBundle
from app.services.openai_builder import OpenAIBuilderService


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
        active_provider=LLMProviderKey.openai,
        agent_execution_backend=AgentExecutionBackend.provider_native,
        openai=OpenAIProviderConfig(
            fast_model="gpt-5.4-mini",
            reasoning_model="gpt-5.5",
            reasoning_effort="low",
            api_key_configured=True,
            available=True,
            status_note="ready",
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
        context_fingerprint="ctx-openai-finops",
    )


def build_requirements_input() -> RequirementsDefinitionInput:
    return RequirementsDefinitionInput(
        discovery=DiscoveryArtifact(problem_statement="Automatizar discovery."),
        canvas=CanvasArtifact(user_goal="Generar requisitos trazables."),
    )


class FakeResponsesAPI:
    def __init__(self, *, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeOpenAIClient:
    def __init__(self, responses_api: FakeResponsesAPI) -> None:
        self.responses = responses_api


def test_openai_structured_call_records_successful_usage_in_ledger() -> None:
    engine, session_factory = build_session_factory()
    response = SimpleNamespace(
        id="resp-openai-1",
        status="completed",
        output_parsed=RequirementsDefinitionOutput(summary="Requisitos consolidados."),
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=48,
            total_tokens=168,
            input_tokens_details=SimpleNamespace(cached_tokens=12),
        ),
    )
    service = OpenAIBuilderService(
        build_runtime_settings(),
        finops_session_factory=session_factory,
        finops_ledger_service=LLMUsageLedgerService(),
    )
    service.can_attempt = lambda: True  # type: ignore[method-assign]
    service._client = FakeOpenAIClient(FakeResponsesAPI(response=response))

    result = service.define_requirements(build_requirements_input(), context_bundle=build_stage_context())

    with Session(engine) as db:
        record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert isinstance(result.artifact, RequirementsDefinitionOutput)
    assert result.usage_record_id is not None
    assert result.cost_total == 0
    assert result.currency == "USD"
    assert result.token_usage["total_tokens"] == 168
    assert result.normalized_usage is not None
    assert service._client.responses.kwargs["timeout"] == (
        get_builder_capability_spec(BuilderCapability.define_requirements).timeout_ms / 1000
    )
    assert result.normalized_usage.cached_input_tokens == 12
    assert record is not None
    assert record.status == "succeeded"
    assert record.provider_key == "openai"
    assert record.model_name == "gpt-5.5"
    assert record.request_id == "resp-openai-1"
    assert record.input_tokens == 120
    assert record.output_tokens == 48
    assert record.cached_input_tokens == 12
    assert record.stage == "define"
    assert record.capability_key == "define_requirements"
    assert record.duration_ms >= 0


def test_openai_structured_call_records_provider_exception() -> None:
    engine, session_factory = build_session_factory()
    service = OpenAIBuilderService(
        build_runtime_settings(),
        finops_session_factory=session_factory,
        finops_ledger_service=LLMUsageLedgerService(),
    )
    service.can_attempt = lambda: True  # type: ignore[method-assign]
    service._client = FakeOpenAIClient(FakeResponsesAPI(error=RuntimeError("provider exploded")))

    result = service.define_requirements(build_requirements_input(), context_bundle=build_stage_context())

    with Session(engine) as db:
        record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert result.artifact is None
    assert result.failure_kind == "provider_error"
    assert result.usage_record_id is not None
    assert record is not None
    assert record.status == "failed"
    assert record.failure_kind == "provider_error"
    assert "provider exploded" in record.failure_detail_redacted
    assert record.total_tokens == 0
