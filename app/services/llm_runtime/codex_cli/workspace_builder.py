from __future__ import annotations

import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from pydantic import BaseModel

from app.services.llm_runtime.codex_cli.context_assembler import CodexContextAssembly
from app.services.llm_runtime.codex_cli.runtime_types import CodexPromptWorkspace


class CodexPromptWorkspaceBuilder:
    def __init__(
        self,
        *,
        runtime_root: Path | None = None,
        prefix: str = "lean-builder-codex",
    ) -> None:
        self.runtime_root = runtime_root or (
            Path(__file__).resolve().parents[5] / "backend" / "runtime" / "codex-workspaces"
        )
        self.prefix = prefix

    def _build_run_id(self, task_kind: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        normalized_task = re.sub(r"[^a-z0-9]+", "-", task_kind.lower()).strip("-") or "run"
        return f"{self.prefix}-{stamp}-{normalized_task}-{uuid4().hex[:8]}"

    def _write_agents_file(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "# Local Staged Workspace Instructions",
                    "",
                    "This staged workspace is self-contained and used for a single Codex run.",
                    "",
                    "Required rules:",
                    "- work only with files inside this workspace;",
                    "- do not read repo-level AGENTS or bootstrap files outside this workspace;",
                    "- start with `input/read_order.md`;",
                    "- read `knowledge/required/*` before candidate context;",
                    "- use `knowledge/candidate/*` only when more context is needed;",
                    "- staged knowledge files may be full required payloads or compact metadata cards; trust manifest delivery_mode;",
                    "- when prompt_truncated=true and staged_file_truncated=false, inspect the relative_path file before judging evidence;",
                    "- write new output only inside `output/`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _write_read_order(self, path: Path, task_kind: str, *, knowledge_access_backend: str) -> None:
        path.write_text(
            "\n".join(
                [
                    "# Read Order",
                    "",
                    f"Task kind: {task_kind}",
                    f"Knowledge backend: {knowledge_access_backend}",
                    "Workspace mode: filesystem_staged",
                    "",
                    "Read required sources first and use candidate context only if needed.",
                    "",
                    "1. [required] Local workspace rules -> AGENTS.md",
                    "2. [required] Prompt staged for this run -> input/prompt.txt",
                    "3. [required] Knowledge manifest -> input/knowledge_manifest.json",
                    "4. [required] Structured schema -> schema.json",
                    "5. [required] Required knowledge files -> knowledge/required/",
                    "6. [candidate] Optional knowledge files -> knowledge/candidate/",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _write_knowledge_manifest(
        self,
        *,
        path: Path,
        root_dir: Path,
        task_kind: str,
        output_model: type[BaseModel],
        knowledge_access_backend: str,
        context_assembly: CodexContextAssembly | None = None,
    ) -> None:
        metadata = context_assembly.metadata_payload() if context_assembly is not None else {}
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "task_kind": task_kind,
            "knowledge_backend_mode": "filesystem_staged",
            "knowledge_access_backend": knowledge_access_backend,
            "context_role": context_assembly.role if context_assembly is not None else "",
            "output_model": output_model.__name__,
            "workspace_root": str(root_dir),
            "operating_summary": "Structured Codex CLI run staged by backend runtime.",
            "staged_inputs": [
                "AGENTS.md",
                "input/read_order.md",
                "input/prompt.txt",
                "input/knowledge_manifest.json",
                "schema.json",
            ],
            "required_sources": metadata.get("required_sources", []),
            "candidate_sources": metadata.get("candidate_sources", []),
            "used_sources": metadata.get("used_sources", []),
            "context_stats": metadata.get("context_stats", {}),
            "staged_outputs": [
                "output/last_message.md",
                "output/structured_output.json",
                "stdout.log",
                "stderr.log",
                "invocation.json",
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _write_context_sources(
        self,
        *,
        root_dir: Path,
        context_assembly: CodexContextAssembly | None,
    ) -> None:
        if context_assembly is None:
            return
        for item in context_assembly.used_sources:
            target_path = root_dir / item.relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(item.workspace_content, encoding="utf-8")

    def _build_output_schema(self, output_model: type[BaseModel]) -> dict[str, object]:
        schema = output_model.model_json_schema()
        return self._enforce_strict_objects(schema)

    def _enforce_strict_objects(
        self,
        value: object,
        *,
        preserve_property_names: bool = False,
    ) -> object:
        if isinstance(value, list):
            return [self._enforce_strict_objects(item) for item in value]
        if not isinstance(value, dict):
            return value
        normalized = {
            key: self._enforce_strict_objects(
                item,
                preserve_property_names=key == "properties",
            )
            for key, item in value.items()
            if preserve_property_names or key not in {"default", "title"}
        }
        properties = normalized.get("properties")
        if normalized.get("type") == "object" or isinstance(properties, dict):
            normalized["additionalProperties"] = False
            if not isinstance(properties, dict):
                properties = {}
                normalized["properties"] = properties
            normalized["required"] = list(properties.keys())
        return normalized

    @contextmanager
    def build(
        self,
        *,
        output_model: type[BaseModel],
        task_kind: str,
        knowledge_access_backend: str = "filesystem_staged",
        context_assembly: CodexContextAssembly | None = None,
    ) -> Iterator[CodexPromptWorkspace]:
        run_id = self._build_run_id(task_kind)
        root_dir = self.runtime_root / run_id
        input_dir = root_dir / "input"
        knowledge_dir = root_dir / "knowledge"
        required_knowledge_dir = knowledge_dir / "required"
        candidate_knowledge_dir = knowledge_dir / "candidate"
        output_dir = root_dir / "output"

        self.runtime_root.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        required_knowledge_dir.mkdir(parents=True, exist_ok=True)
        candidate_knowledge_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        agents_path = root_dir / "AGENTS.md"
        schema_path = root_dir / "schema.json"
        prompt_path = input_dir / "prompt.txt"
        read_order_path = input_dir / "read_order.md"
        knowledge_manifest_path = input_dir / "knowledge_manifest.json"
        last_message_path = output_dir / "last_message.md"
        structured_output_path = output_dir / "structured_output.json"
        stdout_path = root_dir / "stdout.log"
        stderr_path = root_dir / "stderr.log"
        invocation_path = root_dir / "invocation.json"

        self._write_agents_file(agents_path)
        self._write_read_order(
            read_order_path,
            task_kind,
            knowledge_access_backend=knowledge_access_backend,
        )
        self._write_knowledge_manifest(
            path=knowledge_manifest_path,
            root_dir=root_dir,
            task_kind=task_kind,
            output_model=output_model,
            knowledge_access_backend=knowledge_access_backend,
            context_assembly=context_assembly,
        )
        self._write_context_sources(root_dir=root_dir, context_assembly=context_assembly)
        schema_path.write_text(
            json.dumps(self._build_output_schema(output_model), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        prompt_path.write_text("", encoding="utf-8")
        last_message_path.write_text("", encoding="utf-8")
        structured_output_path.write_text("", encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        invocation_path.write_text("{}", encoding="utf-8")

        yield CodexPromptWorkspace(
            run_id=run_id,
            task_kind=task_kind,
            root_dir=root_dir,
            agents_path=agents_path,
            schema_path=schema_path,
            prompt_path=prompt_path,
            read_order_path=read_order_path,
            knowledge_manifest_path=knowledge_manifest_path,
            required_knowledge_dir=required_knowledge_dir,
            candidate_knowledge_dir=candidate_knowledge_dir,
            output_dir=output_dir,
            last_message_path=last_message_path,
            structured_output_path=structured_output_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            invocation_path=invocation_path,
        )
