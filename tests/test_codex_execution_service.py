from __future__ import annotations

from contextlib import contextmanager
import json
import os
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from app.models import (
    CodexLocalProviderConfig,
    DeepSeekProviderConfig,
    KnowledgeAccessBackend,
    LLMProviderKey,
    LLMRuntimeSettings,
    OpenAIProviderConfig,
)
from app.services.llm_runtime.codex_cli.context_assembler import (
    CodexContextInlineSource,
    CodexContextRequest,
)
from app.services.llm_runtime.codex_cli.execution_service import (
    CodexExecutionService,
    resolve_codex_executable_path,
)
from app.services.llm_runtime.codex_cli.runtime_types import (
    CodexExecutionError,
    CodexProcessResult,
    CodexRuntimeErrorCode,
)
from app.services.llm_runtime.codex_cli.workspace_builder import CodexPromptWorkspaceBuilder


def build_runtime_settings() -> LLMRuntimeSettings:
    return LLMRuntimeSettings(
        active_provider=LLMProviderKey.codex_local,
        openai=OpenAIProviderConfig(
            fast_model="gpt-5.4-mini",
            reasoning_model="gpt-5.5",
            reasoning_effort="low",
        ),
        deepseek=DeepSeekProviderConfig(
            base_url="https://api.deepseek.com",
            fast_model="deepseek-v4-flash",
            reasoning_model="deepseek-v4-pro",
            reasoning_effort="max",
        ),
        codex_local=CodexLocalProviderConfig(
            command="codex",
            model="gpt-5.5",
            profile="deep-review",
            timeout_ms=120000,
            executable_found=True,
            available=True,
        ),
    )


class DummyOutput(BaseModel):
    value: str


class NestedDetails(BaseModel):
    note: str


class NestedOutput(BaseModel):
    status: str
    details: NestedDetails


class DefaultedNestedDetails(BaseModel):
    note: str = ""


class DefaultedOutput(BaseModel):
    summary: str = ""
    details: DefaultedNestedDetails = Field(default_factory=DefaultedNestedDetails)
    tags: list[str] = Field(default_factory=list)


class DomainFieldsOutput(BaseModel):
    title: str
    metadata: dict[str, object] = Field(default_factory=dict)


@contextmanager
def build_workspace(tmp_path: Path):
    runtime_root = tmp_path / "codex-workspaces"
    builder = CodexPromptWorkspaceBuilder(runtime_root=runtime_root)
    with builder.build(output_model=DummyOutput, task_kind="dummy_task") as workspace:
        yield builder, workspace


def test_resolve_codex_executable_path_accepts_existing_path(tmp_path: Path) -> None:
    executable = tmp_path / "codex.cmd"
    executable.write_text("@echo off\r\n", encoding="utf-8")

    assert resolve_codex_executable_path(str(executable)) == str(executable)


def test_resolve_codex_executable_path_finds_vscode_extension_without_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extension_bin = (
        tmp_path
        / ".vscode"
        / "extensions"
        / "openai.chatgpt-26.5730.61639-win32-x64"
        / "bin"
        / "windows-x86_64"
    )
    extension_bin.mkdir(parents=True)
    executable = extension_bin / "codex.exe"
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _: None)

    assert resolve_codex_executable_path("codex") == str(executable)


def test_build_execution_args_includes_profile_and_schema(tmp_path: Path) -> None:
    with build_workspace(tmp_path) as (builder, workspace):
        service = CodexExecutionService(
            build_runtime_settings(),
            repo_root=tmp_path,
            workspace_builder=builder,
        )
        args = service.build_execution_args(workspace=workspace)

    assert Path(args[0]).name.lower() in {"codex", "codex.exe", "codex.cmd"}
    assert args[1:10] == [
        "exec",
        "-C",
        str(workspace.root_dir),
        "--sandbox",
        "workspace-write",
        "--color",
        "never",
        "--skip-git-repo-check",
        "--model",
    ]
    assert args[10:14] == [
        "gpt-5.5",
        "--output-schema",
        str(workspace.schema_path),
        "--output-last-message",
    ]
    assert "--output-last-message" in args
    assert str(workspace.last_message_path) in args
    assert "-c" in args
    assert 'model_reasoning_effort="low"' in args
    assert "--profile" in args
    assert "deep-review" in args
    assert args[-1] == "-"


def test_workspace_builder_creates_staged_cr3_structure(tmp_path: Path) -> None:
    with build_workspace(tmp_path) as (_, workspace):
        manifest = json.loads(workspace.knowledge_manifest_path.read_text(encoding="utf-8"))
        schema = json.loads(workspace.schema_path.read_text(encoding="utf-8"))

    assert workspace.root_dir.parent == tmp_path / "codex-workspaces"
    assert workspace.agents_path.exists()
    assert workspace.prompt_path.exists()
    assert workspace.read_order_path.exists()
    assert workspace.knowledge_manifest_path.exists()
    assert workspace.required_knowledge_dir.is_dir()
    assert workspace.candidate_knowledge_dir.is_dir()
    assert workspace.last_message_path.exists()
    assert workspace.structured_output_path.exists()
    assert workspace.stdout_path.exists()
    assert workspace.stderr_path.exists()
    assert workspace.invocation_path.exists()
    assert manifest["task_kind"] == "dummy_task"
    assert manifest["knowledge_backend_mode"] == "filesystem_staged"
    assert "input/prompt.txt" in manifest["staged_inputs"]
    assert "output/structured_output.json" in manifest["staged_outputs"]
    assert schema["additionalProperties"] is False


def test_workspace_builder_writes_strict_nested_schema_for_codex(tmp_path: Path) -> None:
    runtime_root = tmp_path / "codex-workspaces"
    builder = CodexPromptWorkspaceBuilder(runtime_root=runtime_root)

    with builder.build(output_model=NestedOutput, task_kind="nested_schema") as workspace:
        schema = json.loads(workspace.schema_path.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["NestedDetails"]["additionalProperties"] is False


def test_workspace_builder_marks_defaulted_properties_required_for_codex_response_format(tmp_path: Path) -> None:
    runtime_root = tmp_path / "codex-workspaces"
    builder = CodexPromptWorkspaceBuilder(runtime_root=runtime_root)

    with builder.build(output_model=DefaultedOutput, task_kind="defaulted_schema") as workspace:
        schema = json.loads(workspace.schema_path.read_text(encoding="utf-8"))

    assert schema["required"] == ["summary", "details", "tags"]
    assert schema["$defs"]["DefaultedNestedDetails"]["required"] == ["note"]
    assert "default" not in json.dumps(schema)
    assert "title" not in json.dumps(schema)


def test_workspace_builder_preserves_domain_title_and_makes_free_form_objects_strict(tmp_path: Path) -> None:
    runtime_root = tmp_path / "codex-workspaces"
    builder = CodexPromptWorkspaceBuilder(runtime_root=runtime_root)

    with builder.build(output_model=DomainFieldsOutput, task_kind="domain_fields") as workspace:
        schema = json.loads(workspace.schema_path.read_text(encoding="utf-8"))

    assert list(schema["properties"]) == ["title", "metadata"]
    assert schema["required"] == ["title", "metadata"]
    assert schema["properties"]["metadata"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_resolve_timeout_ms_prefers_task_override_env() -> None:
    service = CodexExecutionService(build_runtime_settings())
    original = os.environ.get("BLUEPRINT_NARRATIVE_RUN_TIMEOUT_MS")
    os.environ["BLUEPRINT_NARRATIVE_RUN_TIMEOUT_MS"] = "45000"
    try:
        assert service.resolve_timeout_ms(task_kind="blueprint_narrative") == 45000
    finally:
        if original is None:
            os.environ.pop("BLUEPRINT_NARRATIVE_RUN_TIMEOUT_MS", None)
        else:
            os.environ["BLUEPRINT_NARRATIVE_RUN_TIMEOUT_MS"] = original


def test_execute_structured_prompt_persists_workspace_artifacts_and_uses_workspace_cwd(tmp_path: Path) -> None:
    runtime_root = tmp_path / "codex-workspaces"
    builder = CodexPromptWorkspaceBuilder(runtime_root=runtime_root)
    service = CodexExecutionService(
        build_runtime_settings(),
        repo_root=tmp_path,
        workspace_builder=builder,
    )
    captured: dict[str, object] = {}

    def fake_run_process(
        *,
        command: list[str],
        workdir: Path,
        stdin_text: str,
        timeout_seconds: float,
    ) -> CodexProcessResult:
        captured["command"] = command
        captured["workdir"] = workdir
        captured["stdin_text"] = stdin_text
        captured["timeout_seconds"] = timeout_seconds
        return CodexProcessResult(
            command=command,
            workdir=workdir,
            stdout='{"value":"ok"}',
            stderr="runtime warning",
            returncode=0,
        )

    service.run_process = fake_run_process  # type: ignore[method-assign]

    result = service.execute_structured_prompt(
        task_kind="dummy_task",
        prompt="haz algo",
        output_model=DummyOutput,
    )

    workspace_dir = next(runtime_root.iterdir())
    invocation = json.loads((workspace_dir / "invocation.json").read_text(encoding="utf-8"))
    structured_output = json.loads((workspace_dir / "output" / "structured_output.json").read_text(encoding="utf-8"))

    assert result.value == "ok"
    assert captured["workdir"] == workspace_dir
    assert captured["stdin_text"] == "haz algo"
    assert captured["timeout_seconds"] == 120.0
    assert (workspace_dir / "input" / "prompt.txt").read_text(encoding="utf-8") == "haz algo"
    assert '=== attempt 1 | model gpt-5.5 | returncode 0 | stdout ===' in (
        workspace_dir / "stdout.log"
    ).read_text(encoding="utf-8")
    assert '{"value":"ok"}' in (workspace_dir / "stdout.log").read_text(encoding="utf-8")
    assert '=== attempt 1 | model gpt-5.5 | returncode 0 | stderr ===' in (
        workspace_dir / "stderr.log"
    ).read_text(encoding="utf-8")
    assert "runtime warning" in (workspace_dir / "stderr.log").read_text(encoding="utf-8")
    assert (workspace_dir / "output" / "last_message.md").read_text(encoding="utf-8") == '{"value":"ok"}'
    assert structured_output == {"value": "ok"}
    assert invocation["status"] == "succeeded"
    assert invocation["requested_model"] == "gpt-5.5"
    assert invocation["selected_model"] == "gpt-5.5"
    assert invocation["attempted_models"] == ["gpt-5.5"]
    assert invocation["fallback_used"] is False
    assert invocation["workdir"] == str(workspace_dir)
    assert invocation["payload_source"] == "stdout"
    assert invocation["command"][-1] == "-"
    assert invocation["metrics"]["exit_code"] == 0
    assert invocation["attempts"][0]["status"] == "succeeded"
    assert invocation["attempts"][0]["model"] == "gpt-5.5"


def test_execute_structured_prompt_stages_context_sources_and_declares_audit_metadata(tmp_path: Path) -> None:
    runtime_root = tmp_path / "codex-workspaces"
    builder = CodexPromptWorkspaceBuilder(runtime_root=runtime_root)
    settings = build_runtime_settings()
    settings.knowledge_access_backend = KnowledgeAccessBackend.workspace_staged
    service = CodexExecutionService(
        settings,
        repo_root=tmp_path,
        workspace_builder=builder,
    )
    captured: dict[str, object] = {}

    def fake_run_process(
        *,
        command: list[str],
        workdir: Path,
        stdin_text: str,
        timeout_seconds: float,
    ) -> CodexProcessResult:
        captured["stdin_text"] = stdin_text
        return CodexProcessResult(
            command=command,
            workdir=workdir,
            stdout='{"value":"ok"}',
            stderr="",
            returncode=0,
        )

    service.run_process = fake_run_process  # type: ignore[method-assign]

    result = service.execute_structured_prompt(
        task_kind="dummy_task",
        prompt="Normaliza la captura staged.",
        output_model=DummyOutput,
        context_request=CodexContextRequest(
            role="builder",
            knowledge_access_backend="workspace_staged",
            inline_sources=[
                CodexContextInlineSource(
                    key="discovery_capture",
                    title="Discovery capture",
                    content=json.dumps({"problem": "drift", "notes": ["a" * 8000]}, ensure_ascii=True),
                    required=True,
                    summary="Captura cruda de discovery para la corrida actual.",
                )
            ],
        ),
    )

    workspace_dir = next(runtime_root.iterdir())
    manifest = json.loads((workspace_dir / "input" / "knowledge_manifest.json").read_text(encoding="utf-8"))
    invocation = json.loads((workspace_dir / "invocation.json").read_text(encoding="utf-8"))
    staged_file = workspace_dir / manifest["required_sources"][0]["relative_path"]

    assert result.value == "ok"
    assert manifest["knowledge_access_backend"] == "workspace_staged"
    assert manifest["required_sources"][0]["key"] == "discovery_capture"
    assert manifest["used_sources"][0]["relative_path"].startswith("knowledge/required/")
    assert manifest["context_stats"]["assembled_estimated_tokens"] > 0
    assert manifest["context_stats"]["reduction_estimated_tokens"] > 0
    assert manifest["context_stats"]["used_full_documents"] is False
    assert staged_file.exists()
    assert "Discovery capture" in staged_file.read_text(encoding="utf-8")
    assert "input/knowledge_manifest.json" in str(captured["stdin_text"])
    assert "discovery_capture" in str(captured["stdin_text"])
    assert invocation["metadata"]["context"]["used_sources"][0]["key"] == "discovery_capture"
    assert invocation["metadata"]["context"]["context_stats"]["used_full_documents"] is False


def test_execute_structured_prompt_uses_fallback_model_when_capacity_error(tmp_path: Path) -> None:
    runtime_root = tmp_path / "codex-workspaces"
    settings = build_runtime_settings()
    settings.codex_local.fallback_models = ["gpt-5.4-mini"]
    builder = CodexPromptWorkspaceBuilder(runtime_root=runtime_root)
    service = CodexExecutionService(
        settings,
        repo_root=tmp_path,
        workspace_builder=builder,
    )
    seen_models: list[str] = []

    def fake_run_process(
        *,
        command: list[str],
        workdir: Path,
        stdin_text: str,
        timeout_seconds: float,
    ) -> CodexProcessResult:
        model = command[command.index("--model") + 1]
        seen_models.append(model)
        if model == "gpt-5.5":
            return CodexProcessResult(
                command=command,
                workdir=workdir,
                stdout="",
                stderr="selected model is at capacity",
                returncode=1,
            )
        return CodexProcessResult(
            command=command,
            workdir=workdir,
            stdout='{"value":"fallback-ok"}',
            stderr="",
            returncode=0,
        )

    service.run_process = fake_run_process  # type: ignore[method-assign]

    result = service.execute_structured_prompt(
        task_kind="dummy_task",
        prompt="haz algo",
        output_model=DummyOutput,
    )

    workspace_dir = next(runtime_root.iterdir())
    invocation = json.loads((workspace_dir / "invocation.json").read_text(encoding="utf-8"))

    assert result.value == "fallback-ok"
    assert seen_models == ["gpt-5.5", "gpt-5.4-mini"]
    assert invocation["status"] == "succeeded"
    assert invocation["fallback_used"] is True
    assert invocation["selected_model"] == "gpt-5.4-mini"
    assert invocation["attempted_models"] == ["gpt-5.5", "gpt-5.4-mini"]
    assert invocation["attempts"][0]["error_code"] == CodexRuntimeErrorCode.model_capacity.value
    assert invocation["attempts"][0]["retryable"] is True
    assert invocation["attempts"][1]["status"] == "succeeded"


def test_execute_structured_prompt_raises_fallback_exhausted_when_all_models_fail(tmp_path: Path) -> None:
    runtime_root = tmp_path / "codex-workspaces"
    settings = build_runtime_settings()
    settings.codex_local.fallback_models = ["gpt-5.4-mini"]
    builder = CodexPromptWorkspaceBuilder(runtime_root=runtime_root)
    service = CodexExecutionService(
        settings,
        repo_root=tmp_path,
        workspace_builder=builder,
    )

    def fake_run_process(
        *,
        command: list[str],
        workdir: Path,
        stdin_text: str,
        timeout_seconds: float,
    ) -> CodexProcessResult:
        return CodexProcessResult(
            command=command,
            workdir=workdir,
            stdout="",
            stderr="selected model is at capacity",
            returncode=1,
        )

    service.run_process = fake_run_process  # type: ignore[method-assign]

    try:
        service.execute_structured_prompt(
            task_kind="dummy_task",
            prompt="haz algo",
            output_model=DummyOutput,
        )
        raise AssertionError("Se esperaba CodexExecutionError por fallback agotado.")
    except CodexExecutionError as exc:
        assert exc.code == CodexRuntimeErrorCode.fallback_exhausted
        assert exc.attempted_models == ["gpt-5.5", "gpt-5.4-mini"]
