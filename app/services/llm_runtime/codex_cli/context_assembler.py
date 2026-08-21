from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from app.contracts.canonical_v1 import (
    KnowledgeManifestSourceV1,
    KnowledgeManifestV1,
    MemoryContextBudgetV1,
    MemoryPolicyV1,
    ShortTermMemoryV1,
)
from app.models import SessionSnapshot
from app.services.agent_memory_policy import AgentMemoryPolicyService
from app.services.canonical_exports import (
    build_knowledge_manifest,
    build_memory_policy,
    build_short_term_memory,
)
from app.services.memory_traceability import (
    build_repo_document_lineage,
    build_source_version,
    build_virtual_source_lineage,
)
from app.services.llm_runtime.stage_context_types import ApprovedArtifactReference, RetrievedKnowledgeEvidence
from app.services.text_sanitization import read_sanitized_utf8_text

_TEXT_FILE_SUFFIXES = {".json", ".md", ".mmd", ".puml", ".txt", ".yaml", ".yml"}


def _estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, math.ceil(len(text) / 4))


def _slugify(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or fallback


def _collapse_text(value: str) -> str:
    lines = [line.rstrip() for line in value.replace("\r\n", "\n").splitlines()]
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        compacted.append(line)
        previous_blank = is_blank
    return "\n".join(compacted).strip()


@dataclass(frozen=True)
class CodexContextInlineSource:
    key: str
    title: str
    content: str
    required: bool = True
    source_type: str = "inline_artifact"
    uri: str = ""
    authority_level: str = "runtime_input"
    summary: str = ""
    stage_affinity: list[str] = field(default_factory=list)
    agent_affinity: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CodexContextRequest:
    role: str
    knowledge_access_backend: str
    inline_sources: list[CodexContextInlineSource] = field(default_factory=list)
    workspace_id: UUID | None = None
    session_id: UUID | None = None
    session_snapshot: SessionSnapshot | None = None
    knowledge_manifest: KnowledgeManifestV1 | None = None
    memory_policy: MemoryPolicyV1 | None = None
    short_term_memory: ShortTermMemoryV1 | None = None
    approved_refs: list[ApprovedArtifactReference] = field(default_factory=list)
    retrieved_hits: list[RetrievedKnowledgeEvidence] = field(default_factory=list)
    strict_budget: MemoryContextBudgetV1 | None = None
    context_fingerprint: str = ""
    corpus_hash: str = ""
    retrieval_pages: int = 0
    absence_reason: str = ""
    stage_hint: str = ""


@dataclass(frozen=True)
class CodexContextSource:
    key: str
    title: str
    source_type: str
    uri: str
    authority_level: str
    required: bool
    summary: str
    relative_path: str
    content: str
    baseline_chars: int
    assembled_chars: int
    token_estimate: int
    truncated: bool
    source_refs: list[str] = field(default_factory=list)
    source_lineage: list[str] = field(default_factory=list)
    source_version: str = ""
    stage_affinity: list[str] = field(default_factory=list)
    agent_affinity: list[str] = field(default_factory=list)
    staged_file_content: str = ""
    prompt_truncated: bool = False

    @property
    def workspace_content(self) -> str:
        return self.staged_file_content or self.content

    def to_payload(self) -> dict[str, Any]:
        staged_file_chars = len(self.workspace_content)
        staged_file_truncated = staged_file_chars < self.baseline_chars
        delivery_mode = "filesystem_compact"
        if self.required and self.staged_file_content:
            delivery_mode = "filesystem_required_excerpt" if staged_file_truncated else "filesystem_full_required"
        return {
            "key": self.key,
            "title": self.title,
            "source_type": self.source_type,
            "uri": self.uri,
            "authority_level": self.authority_level,
            "required": self.required,
            "summary": self.summary,
            "relative_path": self.relative_path,
            "baseline_chars": self.baseline_chars,
            "assembled_chars": self.assembled_chars,
            "prompt_chars": self.assembled_chars,
            "staged_file_chars": staged_file_chars,
            "token_estimate": self.token_estimate,
            "truncated": self.truncated,
            "prompt_truncated": self.prompt_truncated,
            "staged_file_truncated": staged_file_truncated,
            "delivery_mode": delivery_mode,
            "source_refs": list(self.source_refs),
            "source_lineage": list(self.source_lineage),
            "source_version": self.source_version,
            "stage_affinity": list(self.stage_affinity),
            "agent_affinity": list(self.agent_affinity),
        }


@dataclass(frozen=True)
class CodexContextStats:
    role: str
    budget_tokens: int
    budget_chars: int
    max_items: int
    baseline_estimated_tokens: int
    assembled_estimated_tokens: int
    reduction_estimated_tokens: int
    used_full_documents: bool
    truncated_source_count: int
    prompt_truncated_source_count: int
    required_source_count: int
    candidate_source_count: int
    discarded_candidate_count: int
    context_fingerprint: str = ""
    corpus_hash: str = ""
    budget_utilization_pct: float = 0
    compaction_ratio: float = 0
    retrieval_page_count: int = 0
    retrieval_hit_count: int = 0
    absence_reason: str = ""
    stage_hint: str = ""
    workspace_id: str = ""
    session_id: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "budget_tokens": self.budget_tokens,
            "budget_chars": self.budget_chars,
            "max_items": self.max_items,
            "baseline_estimated_tokens": self.baseline_estimated_tokens,
            "assembled_estimated_tokens": self.assembled_estimated_tokens,
            "reduction_estimated_tokens": self.reduction_estimated_tokens,
            "used_full_documents": self.used_full_documents,
            "truncated_source_count": self.truncated_source_count,
            "prompt_truncated_source_count": self.prompt_truncated_source_count,
            "required_source_count": self.required_source_count,
            "candidate_source_count": self.candidate_source_count,
            "discarded_candidate_count": self.discarded_candidate_count,
            "context_fingerprint": self.context_fingerprint,
            "corpus_hash": self.corpus_hash,
            "budget_utilization_pct": self.budget_utilization_pct,
            "compaction_ratio": self.compaction_ratio,
            "retrieval_page_count": self.retrieval_page_count,
            "retrieval_hit_count": self.retrieval_hit_count,
            "absence_reason": self.absence_reason,
            "stage_hint": self.stage_hint,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class CodexContextAssembly:
    role: str
    knowledge_access_backend: str
    prompt_preamble: str
    required_sources: list[CodexContextSource]
    candidate_sources: list[CodexContextSource]
    stats: CodexContextStats

    @property
    def used_sources(self) -> list[CodexContextSource]:
        return [*self.required_sources, *self.candidate_sources]

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "knowledge_access_backend": self.knowledge_access_backend,
            "used_sources": [item.to_payload() for item in self.used_sources],
            "required_sources": [item.to_payload() for item in self.required_sources],
            "candidate_sources": [item.to_payload() for item in self.candidate_sources],
            "context_stats": self.stats.to_payload(),
        }


@dataclass(frozen=True)
class _RawContextCandidate:
    key: str
    title: str
    source_type: str
    uri: str
    authority_level: str
    required: bool
    summary: str
    content: str
    baseline_chars: int
    source_refs: list[str]
    source_lineage: list[str]
    source_version: str
    stage_affinity: list[str]
    agent_affinity: list[str]
    score: tuple[int, int, int, int, str]


class CodexContextAssembler:
    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[5]
        self.policy_service = AgentMemoryPolicyService()
        self.taxonomy_manifest_path = self.repo_root / "Docs" / "system-analysis" / "29-memory-m0-taxonomy-manifest.json"
        self._taxonomy_payload = self._load_taxonomy_payload()
        self._taxonomy_rules = {
            str(item.get("rule_key", "")).strip(): item
            for item in self._taxonomy_payload.get("rules", [])
            if isinstance(item, dict) and str(item.get("rule_key", "")).strip()
        }
        self._docs_root = self.repo_root / Path(str(self._taxonomy_payload.get("scope_root", "Docs/")))

    def assemble(self, *, task_kind: str, request: CodexContextRequest) -> CodexContextAssembly:
        knowledge_manifest = request.knowledge_manifest
        memory_policy = request.memory_policy
        short_term_memory = request.short_term_memory
        snapshot = request.session_snapshot
        if snapshot is not None:
            if knowledge_manifest is None:
                knowledge_manifest = build_knowledge_manifest(snapshot)
            if memory_policy is None:
                memory_policy = build_memory_policy(snapshot)
            if short_term_memory is None:
                short_term_memory = build_short_term_memory(snapshot)

        budget = self._resolve_budget(request=request, memory_policy=memory_policy)
        budget_chars = max(1_200, budget.max_chars or (budget.max_tokens * 4) or 4_800)
        max_items = max(1, budget.max_items or 4)
        stage_hint = self._resolve_stage_hint(request=request, snapshot=snapshot, short_term_memory=short_term_memory)

        required_candidates: list[_RawContextCandidate] = []
        candidate_candidates: list[_RawContextCandidate] = []

        for index, source in enumerate(request.inline_sources):
            candidate = self._inline_source_candidate(source, index=index, role=request.role)
            if candidate.required:
                required_candidates.append(candidate)
            else:
                candidate_candidates.append(candidate)

        for index, source in enumerate(request.approved_refs):
            required_candidates.append(self._approved_ref_candidate(source, index=index))

        if short_term_memory is not None:
            required_candidates.append(self._short_term_memory_candidate(short_term_memory, role=request.role))

        if snapshot is not None:
            required_candidates.append(self._snapshot_focus_candidate(snapshot))

        if request.retrieved_hits:
            for item in request.retrieved_hits:
                candidate = self._retrieved_hit_candidate(item, role=request.role, stage_hint=stage_hint)
                if item.required:
                    required_candidates.append(candidate)
                else:
                    candidate_candidates.append(candidate)
        elif knowledge_manifest is not None:
            for source in knowledge_manifest.required_sources:
                if not self.policy_service.can_use_knowledge_source(role=request.role, source=source):
                    continue
                required_candidates.append(
                    self._knowledge_manifest_candidate(
                        source,
                        role=request.role,
                        stage_hint=stage_hint,
                        task_kind=task_kind,
                    )
                )
            for source in knowledge_manifest.candidate_sources:
                if not self.policy_service.can_use_knowledge_source(role=request.role, source=source):
                    continue
                candidate_candidates.append(
                    self._knowledge_manifest_candidate(
                        source,
                        role=request.role,
                        stage_hint=stage_hint,
                        task_kind=task_kind,
                    )
                )

        required_sources, remaining_chars = self._materialize_sources(
            required_candidates,
            bucket="required",
            budget_chars=budget_chars,
            preserve_all=True,
        )
        candidate_budget = max(0, budget_chars - sum(item.assembled_chars for item in required_sources))
        remaining_slots = max(0, max_items - len(required_sources))
        selected_candidates = (
            sorted(candidate_candidates, key=lambda item: item.score)[:remaining_slots]
            if remaining_slots > 0
            else []
        )
        candidate_sources, _ = self._materialize_sources(
            selected_candidates,
            bucket="candidate",
            budget_chars=max(0, max(candidate_budget, remaining_chars)),
            preserve_all=False,
        )

        all_sources = [*required_sources, *candidate_sources]
        baseline_tokens = sum(max(1, math.ceil(item.baseline_chars / 4)) for item in required_candidates + selected_candidates)
        assembled_tokens = sum(item.token_estimate for item in all_sources)
        discarded_candidate_count = max(0, len(candidate_candidates) - len(candidate_sources))
        budget_utilization_pct = round((assembled_tokens / max(1, budget.max_tokens or stats_budget_tokens(budget_chars))) * 100, 2) if all_sources else 0
        compaction_ratio = round(
            (assembled_tokens / max(1, baseline_tokens)) if baseline_tokens > 0 else 1.0,
            4,
        )
        stats = CodexContextStats(
            role=request.role,
            budget_tokens=max(300, budget.max_tokens or stats_budget_tokens(budget_chars)),
            budget_chars=budget_chars,
            max_items=max_items,
            baseline_estimated_tokens=baseline_tokens,
            assembled_estimated_tokens=assembled_tokens,
            reduction_estimated_tokens=max(0, baseline_tokens - assembled_tokens),
            used_full_documents=False,
            truncated_source_count=sum(1 for item in all_sources if item.truncated),
            prompt_truncated_source_count=sum(1 for item in all_sources if item.prompt_truncated),
            required_source_count=len(required_sources),
            candidate_source_count=len(candidate_sources),
            discarded_candidate_count=discarded_candidate_count,
            context_fingerprint=request.context_fingerprint,
            corpus_hash=request.corpus_hash,
            budget_utilization_pct=budget_utilization_pct,
            compaction_ratio=compaction_ratio,
            retrieval_page_count=request.retrieval_pages,
            retrieval_hit_count=len(request.retrieved_hits),
            absence_reason=request.absence_reason,
            stage_hint=stage_hint,
            workspace_id=str(request.workspace_id or ""),
            session_id=str(request.session_id or ""),
        )

        preamble = self._build_prompt_preamble(
            request=request,
            stats=stats,
            required_sources=required_sources,
            candidate_sources=candidate_sources,
        )
        return CodexContextAssembly(
            role=request.role,
            knowledge_access_backend=request.knowledge_access_backend,
            prompt_preamble=preamble,
            required_sources=required_sources,
            candidate_sources=candidate_sources,
            stats=stats,
        )

    def _load_taxonomy_payload(self) -> dict[str, Any]:
        if not self.taxonomy_manifest_path.exists():
            return {"scope_root": "Docs/", "rules": []}
        return json.loads(self.taxonomy_manifest_path.read_text(encoding="utf-8"))

    def _resolve_budget(self, *, request: CodexContextRequest, memory_policy: MemoryPolicyV1 | None) -> MemoryContextBudgetV1:
        if request.strict_budget is not None:
            return request.strict_budget
        role = request.role
        if memory_policy is not None:
            for item in memory_policy.context_budgets:
                if item.role.strip().lower() == role.strip().lower():
                    return item
        return MemoryContextBudgetV1(
            role=role,
            max_tokens=1_200,
            max_items=4,
            max_chars=4_800,
            compaction_trigger="default_runtime_budget",
            overflow_policy="truncate_to_summary",
        )

    def _resolve_stage_hint(
        self,
        *,
        request: CodexContextRequest,
        snapshot: SessionSnapshot | None,
        short_term_memory: ShortTermMemoryV1 | None,
    ) -> str:
        if request.stage_hint.strip():
            return request.stage_hint.strip().lower()
        if short_term_memory is not None and short_term_memory.active_stage.strip():
            return short_term_memory.active_stage.strip().lower()
        if snapshot is not None:
            current_stage = getattr(snapshot.session, "current_stage", "")
            current_value = getattr(current_stage, "value", current_stage)
            if str(current_value).strip():
                return str(current_value).strip().lower()
        return "runtime"

    def _inline_source_candidate(self, source: CodexContextInlineSource, *, index: int, role: str) -> _RawContextCandidate:
        summary = source.summary.strip() or f"Entrada staged `{source.key}` para la corrida actual."
        uri = source.uri or f"inline://{source.key}"
        content = "\n".join(
            [
                f"# {source.title}",
                "",
                f"- key: {source.key}",
                f"- source_type: {source.source_type}",
                f"- authority_level: {source.authority_level}",
                f"- uri: {uri}",
                "",
                "Summary:",
                summary,
                "",
                "Payload:",
                _collapse_text(source.content),
            ]
        )
        source_lineage = [build_virtual_source_lineage(uri, content, kind="inline")]
        return _RawContextCandidate(
            key=source.key,
            title=source.title,
            source_type=source.source_type,
            uri=uri,
            authority_level=source.authority_level,
            required=source.required,
            summary=summary,
            content=content,
            baseline_chars=len(source.content),
            source_refs=[],
            source_lineage=source_lineage,
            source_version=build_source_version(source_lineage),
            stage_affinity=list(source.stage_affinity),
            agent_affinity=list(source.agent_affinity) or [role],
            score=(0 if source.required else 1, 0, 0, index, source.key),
        )

    def _approved_ref_candidate(self, source: ApprovedArtifactReference, *, index: int) -> _RawContextCandidate:
        summary = source.summary.strip() or f"Referencia aprobada `{source.key}`."
        content = "\n".join(
            [
                f"# {source.title}",
                "",
                f"- key: {source.key}",
                f"- artifact_kind: {source.artifact_kind}",
                f"- uri: {source.uri}",
                f"- source_version: {source.source_version or 'current'}",
                "",
                "Summary:",
                summary,
                "",
                "Approved refs:",
                *([f"- {item}" for item in source.source_refs] if source.source_refs else ["- sin refs adicionales"]),
            ]
        )
        source_lineage = [build_virtual_source_lineage(source.uri, content, kind="approved_ref")]
        return _RawContextCandidate(
            key=source.key,
            title=source.title,
            source_type="approved_artifact_ref",
            uri=source.uri,
            authority_level="approved_artifact",
            required=source.required,
            summary=summary,
            content=content,
            baseline_chars=len(content),
            source_refs=list(source.source_refs),
            source_lineage=source_lineage,
            source_version=source.source_version or build_source_version(source_lineage),
            stage_affinity=list(source.stage_affinity),
            agent_affinity=list(source.agent_affinity),
            score=(0, 0, 0, index, source.key),
        )

    def _short_term_memory_candidate(self, short_term_memory: ShortTermMemoryV1, *, role: str) -> _RawContextCandidate:
        view = self.policy_service.short_term_memory_view(short_term_memory, role=role)
        recent_decisions = short_term_memory.recent_decisions[:4]
        namespace_refs = [item.namespace for item in view.visible_namespaces[:4]]
        content = "\n".join(
            [
                "# Short-term memory",
                "",
                f"- role: {role}",
                f"- active_stage: {short_term_memory.active_stage}",
                f"- active_goal: {short_term_memory.active_goal}",
                f"- current_focus: {short_term_memory.current_focus or 'sin foco explicito'}",
                f"- pending_approvals: {', '.join(short_term_memory.pending_approvals) or 'ninguno'}",
                f"- open_handoffs: {', '.join(short_term_memory.open_handoffs) or 'ninguno'}",
                f"- namespaces: {', '.join(namespace_refs) or 'sin namespaces visibles'}",
                f"- hidden_namespace_count: {len(view.hidden_namespace_keys)}",
                "",
                "Recent decisions:",
            ]
            + ([f"- {item}" for item in recent_decisions] if recent_decisions else ["- Sin decisiones recientes."])
        )
        source_lineage = [build_virtual_source_lineage("session://short-term-memory", content, kind="state")]
        return _RawContextCandidate(
            key="short_term_memory",
            title="Short-term memory",
            source_type="session_memory",
            uri="session://short-term-memory",
            authority_level="session_state",
            required=True,
            summary="Resumen operativo de etapa, foco, approvals y decisiones recientes.",
            content=content,
            baseline_chars=len(content),
            source_refs=namespace_refs,
            source_lineage=source_lineage,
            source_version=build_source_version(source_lineage),
            stage_affinity=[short_term_memory.active_stage],
            agent_affinity=["planner", "executor", "retrieval", "memory", "recovery"],
            score=(0, 0, 0, 0, "short_term_memory"),
        )

    def _snapshot_focus_candidate(self, snapshot: SessionSnapshot) -> _RawContextCandidate:
        discovery = snapshot.discovery
        canvas = snapshot.canvas
        blueprint = snapshot.blueprint
        content_lines = [
            "# Session snapshot focus",
            "",
            f"- session_title: {snapshot.session.title or 'Nueva sesion'}",
            f"- active_stage: {getattr(getattr(snapshot.session, 'current_stage', ''), 'value', snapshot.session.current_stage)}",
        ]
        if discovery is not None:
            content_lines.extend(
                [
                    f"- problem_statement: {discovery.problem_statement}",
                    f"- desired_outcome: {discovery.desired_outcome}",
                    f"- constraints: {', '.join(discovery.constraints[:4]) or 'ninguna'}",
                ]
            )
        if canvas is not None:
            content_lines.extend(
                [
                    f"- user_goal: {canvas.user_goal}",
                    f"- success_metric: {canvas.success_metric}",
                ]
            )
        if blueprint is not None:
            content_lines.extend(
                [
                    f"- architecture: {blueprint.architecture}",
                    f"- reasoning_pattern: {blueprint.reasoning_pattern}",
                    f"- memory_strategy: {blueprint.memory_strategy}",
                    f"- tools: {', '.join(item.name for item in blueprint.tools[:5]) or 'sin tools'}",
                ]
            )
        content = "\n".join(content_lines)
        source_lineage = [build_virtual_source_lineage("session://snapshot-focus", content, kind="snapshot")]
        return _RawContextCandidate(
            key="session_snapshot_focus",
            title="Session snapshot focus",
            source_type="session_snapshot",
            uri="session://snapshot-focus",
            authority_level="session_snapshot",
            required=True,
            summary="Resumen corto del estado aprobado de discovery, canvas y blueprint.",
            content=content,
            baseline_chars=len(content),
            source_refs=["session.discovery", "session.canvas", "session.blueprint"],
            source_lineage=source_lineage,
            source_version=build_source_version(source_lineage),
            stage_affinity=[],
            agent_affinity=["planner", "executor", "retrieval", "memory", "recovery"],
            score=(0, 0, 0, 1, "session_snapshot_focus"),
        )

    def _retrieved_hit_candidate(
        self,
        item: RetrievedKnowledgeEvidence,
        *,
        role: str,
        stage_hint: str,
    ) -> _RawContextCandidate:
        summary = item.summary.strip() or item.title
        content = "\n".join(
            [
                f"# {item.title}",
                "",
                f"- key: {item.key}",
                f"- uri: {item.uri}",
                f"- relative_path: {item.relative_path}",
                f"- section_key: {item.section_key}",
                f"- authority_level: {item.authority_level}",
                f"- memory_usage: {item.memory_usage}",
                f"- source_version: {item.source_version or 'current'}",
                "",
                "Summary:",
                summary,
                "",
                "Excerpted evidence:",
                item.excerpt.strip() or summary,
            ]
        )
        return _RawContextCandidate(
            key=item.key,
            title=item.title,
            source_type="retrieved_knowledge_hit",
            uri=item.uri,
            authority_level=item.authority_level,
            required=item.required,
            summary=summary,
            content=content,
            baseline_chars=max(len(item.excerpt), len(content)),
            source_refs=list(item.source_refs),
            source_lineage=list(item.source_lineage),
            source_version=item.source_version,
            stage_affinity=list(item.stage_affinity) or [stage_hint],
            agent_affinity=list(item.agent_affinity) or [role],
            score=(
                0 if item.required else 1,
                0 if stage_hint in {token.lower() for token in item.stage_affinity} else 1,
                0 if item.authority_level in {"canonical", "operational", "approved_artifact"} else 1,
                0,
                item.key,
            ),
        )

    def _knowledge_manifest_candidate(
        self,
        source: KnowledgeManifestSourceV1,
        *,
        role: str,
        stage_hint: str,
        task_kind: str,
    ) -> _RawContextCandidate:
        matched_paths = self._resolve_taxonomy_paths(source.key)
        selected_refs = self._rank_source_paths(
            matched_paths,
            source=source,
            role=role,
            stage_hint=stage_hint,
            task_kind=task_kind,
        )[:2]
        excerpt_blocks: list[str] = []
        for path in selected_refs:
            excerpt = self._read_source_excerpt(path)
            if not excerpt:
                continue
            excerpt_blocks.extend(
                [
                    f"## {path}",
                    excerpt,
                    "",
                ]
            )
        if excerpt_blocks:
            body = "\n".join(excerpt_blocks).strip()
            summary = source.summary or "Fuente documental staged por referencia trazable."
        else:
            body = "No existe contenido local resoluble para esta fuente; usar solo metadatos, URI y lineage autorizados."
            summary = source.summary or "Fuente sin payload local directo; solo metadata autorizada."
        content = "\n".join(
            [
                f"# {source.title}",
                "",
                f"- key: {source.key}",
                f"- source_type: {source.source_type}",
                f"- authority_level: {source.authority_level}",
                f"- uri: {source.uri}",
                f"- required: {'true' if source.required else 'false'}",
                f"- stage_affinity: {', '.join(source.stage_affinity) or 'n/a'}",
                f"- agent_affinity: {', '.join(source.agent_affinity) or 'n/a'}",
                "",
                "Summary:",
                summary,
                "",
                "Excerpted evidence:",
                body,
            ]
        )
        baseline_chars = len(body)
        if selected_refs:
            baseline_chars = sum(len(read_sanitized_utf8_text(self.repo_root / ref)) for ref in selected_refs)
        source_lineage = [
            lineage
            for lineage in (build_repo_document_lineage(self.repo_root, ref) for ref in selected_refs)
            if lineage
        ]
        return _RawContextCandidate(
            key=source.key,
            title=source.title,
            source_type=source.source_type,
            uri=source.uri,
            authority_level=source.authority_level,
            required=source.required,
            summary=summary,
            content=content,
            baseline_chars=max(len(content), baseline_chars),
            source_refs=selected_refs,
            source_lineage=source_lineage,
            source_version=build_source_version(source_lineage, fallback=source.source_version),
            stage_affinity=list(source.stage_affinity),
            agent_affinity=list(source.agent_affinity),
            score=self._source_score(
                source=source,
                role=role,
                stage_hint=stage_hint,
                task_kind=task_kind,
            ),
        )

    def _resolve_taxonomy_paths(self, source_key: str) -> list[str]:
        rule = self._taxonomy_rules.get(source_key)
        if rule is None:
            return []
        if not self._docs_root.exists():
            return []
        all_files = [path for path in self._docs_root.rglob("*") if path.is_file() and path.suffix.lower() in _TEXT_FILE_SUFFIXES]
        relative_files = [path.relative_to(self.repo_root).as_posix() for path in all_files]
        include_prefixes = [str(item).strip() for item in rule.get("include_prefixes", []) if str(item).strip()]
        include_suffixes = [str(item).strip() for item in rule.get("include_suffixes", []) if str(item).strip()]
        exclude_prefixes = [str(item).strip() for item in rule.get("exclude_prefixes", []) if str(item).strip()]
        exclude_filenames = {str(item).strip() for item in rule.get("exclude_filenames", []) if str(item).strip()}

        def included(path: str) -> bool:
            if include_prefixes and not any(path.startswith(prefix) for prefix in include_prefixes):
                return False
            if include_suffixes and not any(path.endswith(suffix) for suffix in include_suffixes):
                return False
            if any(path.startswith(prefix) for prefix in exclude_prefixes):
                return False
            if Path(path).name in exclude_filenames:
                return False
            return True

        return [path for path in relative_files if included(path)]

    def _rank_source_paths(
        self,
        paths: list[str],
        *,
        source: KnowledgeManifestSourceV1,
        role: str,
        stage_hint: str,
        task_kind: str,
    ) -> list[str]:
        stage_token = stage_hint.replace("_", "-")
        role_token = role.replace("_", "-")
        task_token = task_kind.replace("_", "-")

        def score(path: str) -> tuple[int, int, str]:
            lowered = path.lower()
            priority = 0
            if stage_token and stage_token in lowered:
                priority -= 3
            if role_token and role_token in lowered:
                priority -= 2
            if task_token and task_token in lowered:
                priority -= 1
            if source.key.lower() in lowered:
                priority -= 2
            if lowered.endswith(".md"):
                priority -= 1
            return (priority, lowered.count("/"), lowered)

        return sorted(paths, key=score)

    def _source_score(
        self,
        *,
        source: KnowledgeManifestSourceV1,
        role: str,
        stage_hint: str,
        task_kind: str,
    ) -> tuple[int, int, int, int, str]:
        task_tokens = {
            token.strip().lower()
            for token in re.split(r"[_\-]+", task_kind)
            if token.strip()
        }
        stage_penalty = 0 if stage_hint in {item.lower() for item in source.stage_affinity} else 1
        role_penalty = 0 if role in {item.lower() for item in source.agent_affinity} else 1
        task_penalty = 0 if (
            task_kind.replace("_", "-") in source.key.lower()
            or bool(task_tokens & {item.lower() for item in source.stage_affinity})
        ) else 1
        operational_intent = bool(task_tokens & {"implementation", "release", "runtime"})
        authority_penalty = 1 if operational_intent and source.authority_level == "golden_fixture" else 0
        return (stage_penalty, task_penalty, authority_penalty, role_penalty, source.key)

    def _read_source_excerpt(self, relative_path: str) -> str:
        path = self.repo_root / relative_path
        if not path.exists():
            return ""
        return _collapse_text(read_sanitized_utf8_text(path))[:900].strip()

    def _materialize_sources(
        self,
        candidates: list[_RawContextCandidate],
        *,
        bucket: str,
        budget_chars: int,
        preserve_all: bool,
    ) -> tuple[list[CodexContextSource], int]:
        if not candidates:
            return [], budget_chars
        sources: list[CodexContextSource] = []
        remaining_chars = max(600, budget_chars) if preserve_all else max(0, budget_chars)
        for index, candidate in enumerate(candidates, start=1):
            remaining_count = max(1, len(candidates) - len(sources))
            if not preserve_all and remaining_chars < 220:
                break
            target_chars = max(220, remaining_chars // remaining_count) if preserve_all else max(220, remaining_chars // remaining_count)
            if not preserve_all:
                target_chars = min(target_chars, 1_600)
            content = candidate.content.strip()
            prompt_truncated = len(content) > target_chars
            rendered = content[: max(0, target_chars - 32)].rstrip() + "\n\n[truncated]" if prompt_truncated else content
            rendered = rendered.strip()
            assembled_chars = len(rendered)
            remaining_chars = max(0, remaining_chars - assembled_chars)
            slug = _slugify(candidate.key or candidate.title, fallback=f"source-{index}")
            relative_path = f"knowledge/{bucket}/{index:02d}-{slug}.md"
            staged_file_content = content if preserve_all and prompt_truncated else ""
            baseline_chars = max(candidate.baseline_chars, len(candidate.content))
            workspace_content = staged_file_content or rendered
            staged_file_truncated = len(workspace_content) < baseline_chars
            sources.append(
                CodexContextSource(
                    key=candidate.key,
                    title=candidate.title,
                    source_type=candidate.source_type,
                    uri=candidate.uri,
                    authority_level=candidate.authority_level,
                    required=candidate.required,
                    summary=candidate.summary,
                    relative_path=relative_path,
                    content=rendered,
                    baseline_chars=baseline_chars,
                    assembled_chars=assembled_chars,
                    token_estimate=_estimate_tokens(rendered),
                    truncated=staged_file_truncated,
                    source_refs=list(candidate.source_refs),
                    source_lineage=list(candidate.source_lineage),
                    source_version=candidate.source_version,
                    stage_affinity=list(candidate.stage_affinity),
                    agent_affinity=list(candidate.agent_affinity),
                    staged_file_content=staged_file_content,
                    prompt_truncated=prompt_truncated,
                )
            )
        return sources, remaining_chars

    def _build_prompt_preamble(
        self,
        *,
        request: CodexContextRequest,
        stats: CodexContextStats,
        required_sources: list[CodexContextSource],
        candidate_sources: list[CodexContextSource],
    ) -> str:
        lines = [
            "Staged context policy:",
            f"- knowledge_access_backend: {request.knowledge_access_backend}",
            f"- role: {request.role}",
            f"- stage_hint: {request.stage_hint or stats.stage_hint or 'runtime'}",
            f"- budget: {stats.budget_tokens} tokens / {stats.budget_chars} chars / {stats.max_items} preferred items",
            f"- context_fingerprint: {request.context_fingerprint or 'pending'}",
            f"- corpus_hash: {request.corpus_hash or 'n/a'}",
            "- read `input/knowledge_manifest.json` and `knowledge/required/*` before responder;",
            "- usa `knowledge/candidate/*` solo si la evidencia requerida no alcanza;",
            "- cita `key` o `relative_path` de la fuente usada cuando una conclusion dependa del contexto staged;",
            "- if a required source has `prompt_truncated=true` and `staged_file_truncated=false`, treat the file at `relative_path` as the complete evidence;",
            "- no assumes unstaged knowledge and do not reconstruct full documents from excerpts.",
        ]
        if request.retrieval_pages:
            lines.append(f"- retrieval_pages: {request.retrieval_pages}")
        if request.absence_reason:
            lines.append(f"- absence_reason: {request.absence_reason}")
        if required_sources:
            lines.append(
                "- required source keys: " + ", ".join(item.key for item in required_sources)
            )
        if candidate_sources:
            lines.append(
                "- candidate source keys: " + ", ".join(item.key for item in candidate_sources)
            )
        if request.knowledge_access_backend == "hybrid":
            lines.extend(
                [
                    "",
                    "Compact digest:",
                    *(
                        f"- {item.key}: {item.summary}"
                        for item in [*required_sources, *candidate_sources][:4]
                    ),
                ]
            )
        return "\n".join(lines).strip()


def stats_budget_tokens(budget_chars: int) -> int:
    return max(1, budget_chars // 4)
