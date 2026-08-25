from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from app.contracts.canonical_v1 import KnowledgeManifestV1, MemoryContextBudgetV1, MemoryPolicyV1, ShortTermMemoryV1
from app.models import KnowledgeAccessBackend
from app.services.llm_runtime.codex_cli.context_assembler import (
    CodexContextAssembler,
    CodexContextAssembly,
    CodexContextInlineSource,
    CodexContextRequest,
)
from app.services.llm_runtime.stage_context_types import ApprovedArtifactReference, RetrievedKnowledgeEvidence


@dataclass(frozen=True)
class APIProviderContextEnvelope:
    user_payload: str
    knowledge_access_backend: str
    effective_context_backend: str
    used_sources: list[dict[str, Any]]
    context_stats: dict[str, Any]


class APIProviderContextAdapter:
    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.assembler = CodexContextAssembler(repo_root=repo_root)

    def build(
        self,
        *,
        role: str,
        task_kind: str,
        knowledge_access_backend: str,
        task_instruction: str,
        inline_sources: list[CodexContextInlineSource],
        workspace_id: UUID | None = None,
        session_id: UUID | None = None,
        session_snapshot=None,
        knowledge_manifest: KnowledgeManifestV1 | None = None,
        memory_policy: MemoryPolicyV1 | None = None,
        short_term_memory: ShortTermMemoryV1 | None = None,
        approved_refs: list[ApprovedArtifactReference] | None = None,
        retrieved_hits: list[RetrievedKnowledgeEvidence] | None = None,
        strict_budget: MemoryContextBudgetV1 | None = None,
        stage_hint: str = "",
        context_fingerprint: str = "",
        corpus_hash: str = "",
        retrieval_pages: int = 0,
        absence_reason: str = "",
    ) -> APIProviderContextEnvelope:
        assembly = self.assembler.assemble(
            task_kind=task_kind,
            request=CodexContextRequest(
                role=role,
                knowledge_access_backend=knowledge_access_backend,
                inline_sources=inline_sources,
                workspace_id=workspace_id,
                session_id=session_id,
                session_snapshot=session_snapshot,
                knowledge_manifest=knowledge_manifest,
                memory_policy=memory_policy,
                short_term_memory=short_term_memory,
                approved_refs=list(approved_refs or []),
                retrieved_hits=list(retrieved_hits or []),
                strict_budget=strict_budget,
                stage_hint=stage_hint,
                context_fingerprint=context_fingerprint,
                corpus_hash=corpus_hash,
                retrieval_pages=retrieval_pages,
                absence_reason=absence_reason,
            ),
        )
        effective_backend = self._effective_context_backend(knowledge_access_backend)
        user_payload = self._build_user_payload(assembly=assembly, task_instruction=task_instruction)
        metadata = assembly.metadata_payload()
        return APIProviderContextEnvelope(
            user_payload=user_payload,
            knowledge_access_backend=knowledge_access_backend,
            effective_context_backend=effective_backend,
            used_sources=list(metadata.get("used_sources", [])),
            context_stats=dict(metadata.get("context_stats", {})),
        )

    def _effective_context_backend(self, knowledge_access_backend: str) -> str:
        if knowledge_access_backend == KnowledgeAccessBackend.hybrid.value:
            return "hybrid_inline_compact"
        if knowledge_access_backend == KnowledgeAccessBackend.workspace_staged.value:
            return "workspace_staged_unavailable_inline_compact"
        return "inline_context_compact"

    def _build_user_payload(
        self,
        *,
        assembly: CodexContextAssembly,
        task_instruction: str,
    ) -> str:
        blocks = [self._render_inline_source(item) for item in assembly.used_sources]
        return "\n\n".join(
            [
                assembly.prompt_preamble,
                "Context sources:",
                "\n\n".join(blocks) if blocks else "- Sin fuentes compactadas.",
                "Task:",
                task_instruction.strip(),
            ]
        ).strip()

    def _render_inline_source(self, source) -> str:
        excerpt = self._extract_excerpt(source.content)
        lines = [
            f"[source] {source.key}",
            f"title={source.title}",
            f"type={source.source_type}",
            f"authority={source.authority_level}",
            f"required={'true' if source.required else 'false'}",
            f"summary={source.summary}",
        ]
        if getattr(source, "prompt_truncated", False) or getattr(source, "truncated", False):
            lines.append("truncated=true")
        if source.source_refs:
            lines.append("refs=" + ", ".join(source.source_refs[:3]))
        lines.extend(
            [
                "excerpt:",
                excerpt,
            ]
        )
        return "\n".join(lines).strip()

    def _extract_excerpt(self, content: str) -> str:
        normalized = content.strip()
        if not normalized:
            return ""
        return normalized
