from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import UUID, uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.core.config import get_settings
from app.models import (
    CanvasArtifact,
    DiscoveryArtifact,
    LLMProviderKey,
    LLMUsageLedgerRecord,
    UserRecord,
    WorkspaceProviderSecretUpsertRequest,
    WorkspaceRecord,
)
from app.services.auth_service import hash_password
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionInput, RequirementsDefinitionOutput
from app.services.llm_runtime.runtime_secrets_service import (
    upsert_workspace_provider_secret,
)
from app.services.llm_runtime.runtime_settings_service import load_effective_runtime_settings
from app.services.llm_runtime.stage_context_types import StageContextBundle
from app.services.openai_builder import DeepSeekBuilderService, OpenAIBuilderService


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


def seed_user_and_workspace(
    session: Session,
    *,
    email: str,
    workspace_name: str,
) -> tuple[UserRecord, WorkspaceRecord]:
    user = UserRecord(
        email=email,
        full_name=workspace_name,
        password_hash=hash_password("LeanBuilder123!"),
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    workspace = WorkspaceRecord(
        name=workspace_name,
        slug=workspace_name.lower().replace(" ", "-"),
        created_by_user_id=user.id,
    )
    session.add(workspace)
    session.commit()
    session.refresh(workspace)
    return user, workspace


def build_stage_context(workspace_id: UUID) -> StageContextBundle:
    return StageContextBundle(
        capability="define_requirements",
        role="builder",
        stage="define",
        workspace_id=workspace_id,
        session_id=uuid4(),
        session_snapshot=None,
        effective_language="es",
        knowledge_manifest=None,
        memory_policy=None,
        short_term_memory=None,
        context_fingerprint="ctx-workspace-provider-secret",
    )


def build_requirements_input() -> RequirementsDefinitionInput:
    return RequirementsDefinitionInput(
        discovery=DiscoveryArtifact(problem_statement="Automatizar discovery."),
        canvas=CanvasArtifact(user_goal="Generar requisitos trazables."),
    )


class FakeResponsesAPI:
    def __init__(self, *, response=None) -> None:
        self.response = response

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeOpenAIClient:
    def __init__(self, responses_api: FakeResponsesAPI) -> None:
        self.responses = responses_api


class FakeCompletionsAPI:
    def __init__(self, *, response=None) -> None:
        self.response = response

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeDeepSeekClient:
    def __init__(self, completions_api: FakeCompletionsAPI) -> None:
        self.chat = SimpleNamespace(completions=completions_api)


def test_openai_service_bootstraps_client_from_workspace_secret(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "llm_mode", "openai")

    with TemporaryDirectory(prefix="lean-builder-openai-workspace-secret-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(
            json.dumps(
                {
                    "active_provider": "openai",
                    "agent_execution_backend": "provider_native",
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(settings, "llm_config_path", runtime_path)

        engine, session_factory = build_session_factory()
        with Session(engine) as session:
            actor, workspace = seed_user_and_workspace(
                session,
                email="workspace-openai@leanbuilder.local",
                workspace_name="Workspace OpenAI Secret",
            )
            upsert_workspace_provider_secret(
                session,
                workspace.id,
                LLMProviderKey.openai,
                WorkspaceProviderSecretUpsertRequest(
                    secret_value="sk-workspace-openai",
                    activate_for_runtime=True,
                ),
                actor_user_id=actor.id,
            )
            runtime_settings = load_effective_runtime_settings(session, workspace.id)

        assert runtime_settings.uses_platform_credentials is False
        assert runtime_settings.active_provider == LLMProviderKey.openai
        assert runtime_settings.openai.api_key_configured is False

        captured: dict[str, str] = {}
        response = SimpleNamespace(
            id="resp-openai-workspace-1",
            status="completed",
            output_parsed=RequirementsDefinitionOutput(summary="Requisitos consolidados desde workspace."),
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=48,
                total_tokens=168,
                input_tokens_details=SimpleNamespace(cached_tokens=8),
            ),
        )

        def fake_openai_factory(*, api_key: str, base_url: str | None = None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url or ""
            return FakeOpenAIClient(FakeResponsesAPI(response=response))

        monkeypatch.setattr("app.services.openai_builder.OpenAI", fake_openai_factory)

        service = OpenAIBuilderService(
            runtime_settings,
            finops_session_factory=session_factory,
            finops_ledger_service=LLMUsageLedgerService(),
        )
        result = service.define_requirements(
            build_requirements_input(),
            context_bundle=build_stage_context(workspace.id),
        )

        with Session(engine) as session:
            record = session.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert captured["api_key"] == "sk-workspace-openai"
    assert captured["base_url"] == ""
    assert result.artifact is not None
    assert result.artifact.summary == "Requisitos consolidados desde workspace."
    assert result.usage_record_id is not None
    assert record is not None
    assert record.provider_key == "openai"
    assert record.status == "succeeded"


def test_deepseek_service_bootstraps_client_from_workspace_secret(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    monkeypatch.setattr(settings, "llm_mode", "openai")

    with TemporaryDirectory(prefix="lean-builder-deepseek-workspace-secret-") as runtime_dir:
        runtime_path = Path(runtime_dir) / "llm_settings.json"
        runtime_path.write_text(
            json.dumps(
                {
                    "active_provider": "deepseek",
                    "agent_execution_backend": "provider_native",
                    "deepseek": {"base_url": "https://api.deepseek.test"},
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(settings, "llm_config_path", runtime_path)

        engine, session_factory = build_session_factory()
        with Session(engine) as session:
            actor, workspace = seed_user_and_workspace(
                session,
                email="workspace-deepseek@leanbuilder.local",
                workspace_name="Workspace DeepSeek Secret",
            )
            upsert_workspace_provider_secret(
                session,
                workspace.id,
                LLMProviderKey.deepseek,
                WorkspaceProviderSecretUpsertRequest(
                    secret_value="sk-workspace-deepseek",
                    activate_for_runtime=True,
                ),
                actor_user_id=actor.id,
            )
            runtime_settings = load_effective_runtime_settings(session, workspace.id)

        assert runtime_settings.uses_platform_credentials is False
        assert runtime_settings.active_provider == LLMProviderKey.deepseek
        assert runtime_settings.deepseek.api_key_configured is False

        captured: dict[str, str] = {}
        content = json.dumps(
            RequirementsDefinitionOutput(summary="Requisitos DeepSeek desde workspace.").model_dump(mode="json")
        )
        response = SimpleNamespace(
            id="chatcmpl-deepseek-workspace-1",
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
                prompt_cache_hit_tokens=12,
                prompt_cache_miss_tokens=128,
            ),
        )

        def fake_openai_factory(*, api_key: str, base_url: str | None = None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url or ""
            return FakeDeepSeekClient(FakeCompletionsAPI(response=response))

        monkeypatch.setattr("app.services.openai_builder.OpenAI", fake_openai_factory)

        service = DeepSeekBuilderService(
            runtime_settings,
            finops_session_factory=session_factory,
            finops_ledger_service=LLMUsageLedgerService(),
        )
        result = service.define_requirements(
            build_requirements_input(),
            context_bundle=build_stage_context(workspace.id),
        )

        with Session(engine) as session:
            record = session.get(LLMUsageLedgerRecord, result.usage_record_id)

    assert captured["api_key"] == "sk-workspace-deepseek"
    assert captured["base_url"] == "https://api.deepseek.test"
    assert result.artifact is not None
    assert result.artifact.summary == "Requisitos DeepSeek desde workspace."
    assert result.usage_record_id is not None
    assert record is not None
    assert record.provider_key == "deepseek"
    assert record.status == "succeeded"
