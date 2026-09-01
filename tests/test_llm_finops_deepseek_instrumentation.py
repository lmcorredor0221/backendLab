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
from app.services.llm_runtime.capability_registry import BuilderCapability, get_builder_capability_spec
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


class FakeSequentialCompletionsAPI:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.kwargs_history: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.kwargs_history.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("No quedan respuestas fake para DeepSeek.")
        return self.responses.pop(0)


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
    assert service._client.chat.completions.kwargs["timeout"] == (
        get_builder_capability_spec(BuilderCapability.define_requirements).timeout_ms / 1000
    )
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
    assert record.metadata_payload["effective_context_backend"] == "workspace_staged_unavailable_inline_compact"
    assert record.metadata_payload["context_stats"]["api_context_contract"] == "provider_api_inline.v1"
    assert record.prompt_hash
    assert record.response_hash
    assert record.prompt_hash == record.metadata_payload["context_stats"]["context_user_payload_sha256"]
    assert record.metadata_payload["context_used_sources"][0]["key"] == "requirements_definition_input"
    assert record.metadata_payload["context_used_sources"][0]["metadata"]["context_quality_version"] == "context-quality.v1"
    assert record.metadata_payload["context_used_sources"][0]["metadata"]["input_payload_chars"] > 0
    assert "context_prompt_truncated_keys" in record.metadata_payload


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


def test_deepseek_parse_failure_still_records_request_and_usage() -> None:
    engine, session_factory = build_session_factory()
    response = SimpleNamespace(
        id="chatcmpl-deepseek-bad-json",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"summary":"ok"\n"missing_comma":true}'),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=140,
            completion_tokens=60,
            total_tokens=200,
            prompt_cache_hit_tokens=15,
            prompt_cache_miss_tokens=125,
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

    assert result.artifact is None
    assert result.failure_kind == "provider_error"
    assert result.request_id == "chatcmpl-deepseek-bad-json"
    assert result.usage_record_id is not None
    assert result.token_usage["total_tokens"] == 200
    assert result.normalized_usage is not None
    assert result.normalized_usage.cached_input_tokens == 15
    assert record is not None
    assert record.status == "failed"
    assert record.request_id == "chatcmpl-deepseek-bad-json"
    assert record.total_tokens == 200
    assert record.cached_input_tokens == 15
    assert record.failure_kind == "provider_error"


def test_deepseek_retries_once_when_length_truncates_json() -> None:
    engine, session_factory = build_session_factory()
    truncated_response = SimpleNamespace(
        id="chatcmpl-deepseek-length-1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"summary":"truncado"'),
                finish_reason="length",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=180,
            completion_tokens=4096,
            total_tokens=4276,
            prompt_cache_hit_tokens=0,
            prompt_cache_miss_tokens=180,
            reasoning_tokens=1200,
        ),
    )
    valid_content = json.dumps(RequirementsDefinitionOutput(summary="Requisitos recuperados tras retry.").model_dump(mode="json"))
    recovered_response = SimpleNamespace(
        id="chatcmpl-deepseek-length-2",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=valid_content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=80,
            total_tokens=280,
            prompt_cache_hit_tokens=0,
            prompt_cache_miss_tokens=200,
            reasoning_tokens=0,
        ),
    )
    completions_api = FakeSequentialCompletionsAPI([truncated_response, recovered_response])
    service = DeepSeekBuilderService(
        build_runtime_settings(),
        finops_session_factory=session_factory,
        finops_ledger_service=LLMUsageLedgerService(),
    )
    service.can_attempt = lambda: True  # type: ignore[method-assign]
    service._client = FakeDeepSeekClient(completions_api)

    result = service.define_requirements(build_requirements_input(), context_bundle=build_stage_context())

    with Session(engine) as db:
        record = db.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert isinstance(result.artifact, RequirementsDefinitionOutput)
    assert result.artifact.summary == "Requisitos recuperados tras retry."
    assert result.request_id == "chatcmpl-deepseek-length-2"
    assert result.retry_count == 1
    assert result.usage_record_id is not None
    assert result.token_usage["total_tokens"] == 280
    assert record is not None
    assert record.status == "succeeded"
    assert record.request_id == "chatcmpl-deepseek-length-2"
    assert record.retry_count == 1
    assert len(completions_api.kwargs_history) == 2
    assert int(completions_api.kwargs_history[1]["max_tokens"]) > int(completions_api.kwargs_history[0]["max_tokens"])
    assert completions_api.kwargs_history[1]["extra_body"] == {"thinking": {"type": "disabled"}}
