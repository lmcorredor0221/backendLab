from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    AgentExecutionBackend,
    CanvasArtifact,
    DeepSeekProviderConfig,
    DiscoveryArtifact,
    LLMProviderKey,
    LLMRuntimeSettings,
    LLMUsageLedgerRecord,
)
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionInput, RequirementsDefinitionOutput
from app.services.llm_runtime.stage_context_types import StageContextBundle
from app.services.openai_builder import DeepSeekBuilderService


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
        active_provider=LLMProviderKey.deepseek,
        agent_execution_backend=AgentExecutionBackend.provider_native,
        deepseek=DeepSeekProviderConfig(
            base_url="https://api.deepseek.test",
            fast_model="deepseek-v4-flash",
            reasoning_model="deepseek-v4-pro",
            reasoning_effort="high",
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
        context_fingerprint="ctx-deepseek-finops",
    )


def build_requirements_input() -> RequirementsDefinitionInput:
    return RequirementsDefinitionInput(
        discovery=DiscoveryArtifact(problem_statement="Automatizar discovery."),
        canvas=CanvasArtifact(user_goal="Generar requisitos trazables."),
    )


class FakeCompletionsAPI:
    def __init__(self, *, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeDeepSeekClient:
    def __init__(self, completions_api: FakeCompletionsAPI) -> None:
        self.chat = SimpleNamespace(completions=completions_api)


def test_deepseek_structured_call_records_successful_usage_in_ledger() -> None:
    engine, session_factory = build_session_factory()
    content = json.dumps(RequirementsDefinitionOutput(summary="Requisitos consolidados.").model_dump(mode="json"))
    response = SimpleNamespace(
        id="chatcmpl-deepseek-1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=140,
            completion_tokens=60,
            total_tokens=200,
            prompt_cache_hit_tokens=30,
            prompt_cache_miss_tokens=110,
        ),
    )
    service = DeepSeekBuilderService(
        build_runtime_settings(),
        finops_session_factory=session_factory,
        finops_ledger_service=LLMUsageLedgerService(),
    )
    service.can_attempt = lambda: True  # type: ignore[method-assign]
    service._client = FakeDeepSeekClient(FakeCompletionsAPI(response=response))

    result = service.define_requirements(build_requirements_input(), context_bundle=build_stage_context())

    with Session(engine) as db:
        record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert isinstance(result.artifact, RequirementsDefinitionOutput)
    assert result.usage_record_id is not None
    assert result.token_usage["total_tokens"] == 200
    assert result.normalized_usage is not None
    assert result.normalized_usage.cached_input_tokens == 30
    assert record is not None
    assert record.status == "succeeded"
    assert record.provider_key == "deepseek"
    assert record.model_name == "deepseek-v4-pro"
    assert record.request_id == "chatcmpl-deepseek-1"
    assert record.input_tokens == 140
    assert record.output_tokens == 60
    assert record.cached_input_tokens == 30
    assert record.provider_metrics["cache_hit_tokens"] == 30
    assert record.stage == "define"
    assert record.capability_key == "define_requirements"


def test_deepseek_structured_call_records_provider_exception() -> None:
    engine, session_factory = build_session_factory()
    service = DeepSeekBuilderService(
        build_runtime_settings(),
        finops_session_factory=session_factory,
        finops_ledger_service=LLMUsageLedgerService(),
    )
    service.can_attempt = lambda: True  # type: ignore[method-assign]
    service._client = FakeDeepSeekClient(FakeCompletionsAPI(error=RuntimeError("deepseek down")))

    result = service.define_requirements(build_requirements_input(), context_bundle=build_stage_context())

    with Session(engine) as db:
        record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert result.artifact is None
    assert result.failure_kind == "provider_error"
    assert result.usage_record_id is not None
    assert record is not None
    assert record.status == "failed"
    assert record.provider_key == "deepseek"
    assert record.failure_kind == "provider_error"
    assert "deepseek down" in record.failure_detail_redacted
