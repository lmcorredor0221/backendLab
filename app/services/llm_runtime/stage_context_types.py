from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from app.contracts.canonical_v1 import KnowledgeManifestV1, MemoryContextBudgetV1, MemoryPolicyV1, ShortTermMemoryV1
from app.models import KnowledgeSearchResponse, SessionSnapshot


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
    approved_refs: list[ApprovedArtifactReference] = field(default_factory=list)
    retrieved_hits: list[RetrievedKnowledgeEvidence] = field(default_factory=list)
    retrieval_response: KnowledgeSearchResponse | None = None
    strict_budget: MemoryContextBudgetV1 | None = None
    context_fingerprint: str = ""
    corpus_hash: str = ""
    retrieval_pages: int = 0
    absence_reason: str = ""
