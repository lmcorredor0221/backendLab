from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.contracts.canonical_v1 import KnowledgeManifestV1, MemoryContextBudgetV1, MemoryPolicyV1, ShortTermMemoryV1
from app.models import KnowledgeSearchResponse, SessionSnapshot
from app.services.llm_finops import LLMCallContext


@dataclass(frozen=True)
class ApprovedArtifactReference:
    key: str
    title: str
    artifact_kind: str
    uri: str
    summary: str
    source_version: str = ""
    required: bool = True
    source_refs: list[str] = field(default_factory=list)
    stage_affinity: list[str] = field(default_factory=list)
    agent_affinity: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedKnowledgeEvidence:
    key: str
    title: str
    uri: str
    relative_path: str
    section_key: str
    authority_level: str
    memory_usage: str
    summary: str
    excerpt: str
    source_version: str = ""
    required: bool = False
    source_refs: list[str] = field(default_factory=list)
    source_lineage: list[str] = field(default_factory=list)
    stage_affinity: list[str] = field(default_factory=list)
    agent_affinity: list[str] = field(default_factory=list)
    score: float = 0


@dataclass(frozen=True)
class KnowledgeEnrichmentItem:
    key: str
    source_ref: str
    authority: str = "global_knowledge"
    stage_scope: list[str] = field(default_factory=list)
    recommendation: str = ""
    rationale: str = ""
    risk_if_ignored: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class StageKnowledgeEnrichment:
    stage: str
    kb_version: str = ""
    corpus_hash: str = ""
    matched_patterns: list[KnowledgeEnrichmentItem] = field(default_factory=list)
    inferred_recommendations: list[KnowledgeEnrichmentItem] = field(default_factory=list)
    risk_controls: list[KnowledgeEnrichmentItem] = field(default_factory=list)
    delegated_decision_candidates: list[KnowledgeEnrichmentItem] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StageContextBundle:
    capability: str
    role: str
    stage: str
    workspace_id: UUID | None
    session_id: UUID | None
    session_snapshot: SessionSnapshot | None
    effective_language: str
    knowledge_manifest: KnowledgeManifestV1 | None
    memory_policy: MemoryPolicyV1 | None
    short_term_memory: ShortTermMemoryV1 | None
    knowledge_enrichment: StageKnowledgeEnrichment | None = None
    approved_refs: list[ApprovedArtifactReference] = field(default_factory=list)
    retrieved_hits: list[RetrievedKnowledgeEvidence] = field(default_factory=list)
    retrieval_response: KnowledgeSearchResponse | None = None
    strict_budget: MemoryContextBudgetV1 | None = None
    context_fingerprint: str = ""
    corpus_hash: str = ""
    retrieval_pages: int = 0
    absence_reason: str = ""
    finops_operation_id: UUID | None = None
    finops_correlation_id: str = ""
    finops_execution_mode: str = ""
    finops_parent_run_id: str = ""
    finops_metadata: dict[str, Any] = field(default_factory=dict)


def build_llm_call_context(
    context_bundle: StageContextBundle | None,
    *,
    capability: str = "",
    provider_key: str = "",
    execution_backend: str = "",
    execution_mode: str = "",
    action_key: str = "",
    operation_id: UUID | None = None,
    parent_run_id: str = "",
    correlation_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> LLMCallContext:
    bundle_metadata = _stage_context_metadata(
        context_bundle,
        provider_key=provider_key,
        execution_backend=execution_backend,
        execution_mode=_resolved_execution_mode(context_bundle, execution_mode),
    )
    bundle_metadata.update(metadata or {})
    resolved_capability = capability or (context_bundle.capability if context_bundle is not None else "")
    resolved_operation_id = operation_id or (
        context_bundle.finops_operation_id if context_bundle is not None else None
    )
    resolved_execution_mode = _resolved_execution_mode(context_bundle, execution_mode)
    resolved_correlation = (
        correlation_id
        or (context_bundle.finops_correlation_id if context_bundle is not None else "")
        or _default_correlation_id(context_bundle, resolved_capability)
    )
    return LLMCallContext(
        workspace_id=context_bundle.workspace_id if context_bundle is not None else None,
        user_id=_snapshot_owner_user_id(context_bundle.session_snapshot if context_bundle is not None else None),
        session_id=context_bundle.session_id if context_bundle is not None else None,
        project_id=_snapshot_related_id(context_bundle, "project_id"),
        initiative_id=_snapshot_related_id(context_bundle, "initiative_id"),
        stage=context_bundle.stage if context_bundle is not None else "",
        agent_key=context_bundle.role if context_bundle is not None else "",
        capability_key=resolved_capability,
        action_key=action_key or resolved_capability,
        operation_id=resolved_operation_id,
        parent_run_id=parent_run_id or (context_bundle.finops_parent_run_id if context_bundle is not None else ""),
        execution_mode=resolved_execution_mode,
        correlation_id=resolved_correlation,
        source="stage_context_bundle",
        metadata=bundle_metadata,
    )


def _stage_context_metadata(
    context_bundle: StageContextBundle | None,
    *,
    provider_key: str,
    execution_backend: str,
    execution_mode: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider_key": provider_key,
        "execution_backend": execution_backend,
        "execution_mode": execution_mode,
    }
    if provider_key:
        metadata["provider_route"] = f"{execution_mode}:{provider_key}"
    if context_bundle is None:
        return metadata
    metadata.update(context_bundle.finops_metadata)
    metadata.update(
        {
            "role": context_bundle.role,
            "source_capability": context_bundle.capability,
            "context_fingerprint": context_bundle.context_fingerprint,
            "corpus_hash": context_bundle.corpus_hash,
            "retrieval_pages": context_bundle.retrieval_pages,
            "absence_reason": context_bundle.absence_reason,
        }
    )
    return metadata


def _snapshot_owner_user_id(snapshot: SessionSnapshot | None) -> UUID | None:
    if snapshot is None:
        return None
    owner = getattr(getattr(snapshot, "session", None), "owner", None)
    owner_id = getattr(owner, "id", None)
    return owner_id if isinstance(owner_id, UUID) else None


def _snapshot_related_id(context_bundle: StageContextBundle | None, field_name: str) -> UUID | None:
    if context_bundle is None or context_bundle.session_snapshot is None:
        return None
    session = getattr(context_bundle.session_snapshot, "session", None)
    value = getattr(session, field_name, None)
    return value if isinstance(value, UUID) else None


def _default_correlation_id(context_bundle: StageContextBundle | None, capability: str) -> str:
    if context_bundle is None:
        return capability
    if context_bundle.context_fingerprint:
        return context_bundle.context_fingerprint
    if context_bundle.session_id:
        return f"{context_bundle.session_id}:{capability or context_bundle.capability}"
    return capability or context_bundle.capability


def _resolved_execution_mode(context_bundle: StageContextBundle | None, execution_mode: str) -> str:
    if execution_mode:
        return execution_mode
    if context_bundle is not None and context_bundle.finops_execution_mode:
        return context_bundle.finops_execution_mode
    return "primary"
