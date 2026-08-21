from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import or_
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    KnowledgeCorpusStatus,
    KnowledgeDocumentEntry,
    KnowledgeDocumentGovernancePatchRequest,
    KnowledgeDocumentRecord,
    KnowledgeDocumentStatus,
    KnowledgeIngestionReport,
    KnowledgeIngestionRunRecord,
    KnowledgeManagedDocumentUpsertRequest,
    KnowledgeScope,
    KnowledgeSearchHit,
    KnowledgeSearchResponse,
    KnowledgeSectionRecord,
    KnowledgeVisibility,
    utc_now,
)
from app.services.agent_memory_policy import AgentMemoryPolicyService
from app.services.text_sanitization import read_sanitized_utf8_text, sanitize_text_content


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
SUPPORTED_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}
DEFAULT_VECTOR_DIMENSIONS = 96
TAXONOMY_PATH = Path(__file__).resolve().parents[3] / "Docs" / "system-analysis" / "29-memory-m0-taxonomy-manifest.json"
MIN_GROUNDED_VECTOR_SCORE = 0.18
AUTHORITY_SCORE_WEIGHTS = {
    "canonical": 1.25,
    "operational": 1.1,
    "approved_artifact": 1.04,
    "research_reference": 1.0,
    "golden_fixture": 0.72,
    "visual_reference": 0.25,
}
MEMORY_USAGE_SCORE_WEIGHTS = {
    "required_retrieval": 1.2,
    "candidate_retrieval": 1.0,
    "validation_only": 0.42,
    "visual_only": 0.1,
}


@dataclass(frozen=True)
class ScopeConfig:
    scope: KnowledgeScope
    source_root: str
    docs_root: Path
    runtime_dir: Path
    overrides_root: Path
    workspace_id: UUID | None = None
    session_id: UUID | None = None


@dataclass(frozen=True)
class ParsedSection:
    section_key: str
    title: str
    heading_path: list[str]
    heading_level: int
    sort_order: int
    start_line: int
    end_line: int
    content_text: str
    content_hash: str
    source_lineage: str
    lexical_terms: list[str]
    vector_payload: list[float]


@dataclass(frozen=True)
class ParsedDocument:
    source_root: str
    scope: KnowledgeScope
    workspace_id: UUID | None
    session_id: UUID | None
    relative_path: str
    title: str
    format: str
    visibility: KnowledgeVisibility
    status: KnowledgeDocumentStatus
    authority_level: str
    memory_usage: str
    stage_affinity: list[str]
    agent_affinity: list[str]
    effective_from: datetime | None
    expires_at: datetime | None
    approved_by_user_id: UUID | None
    approved_at: datetime | None
    supersedes_document_id: UUID | None
    content_text: str
    content_hash: str
    file_size_bytes: int
    word_count: int
    source_lineage: str
    sections: list[ParsedSection]


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_newlines(value: str) -> str:
    return sanitize_text_content(value).replace("\r\n", "\n").replace("\r", "\n")


def _slugify(value: str, *, fallback: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    return slug or fallback


def _tokenize(value: str) -> list[str]:
    normalized = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return TOKEN_RE.findall(normalized)


def _build_vector(tokens: list[str], *, dimensions: int) -> list[float]:
    if not tokens:
        return [0.0] * dimensions
    counts = Counter(tokens)
    vector = [0.0] * dimensions
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + math.log(count)
        vector[bucket] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [0.0] * dimensions
    return [round(value / norm, 8) for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False))


def _governance_weight(authority_level: str, memory_usage: str) -> float:
    authority_weight = AUTHORITY_SCORE_WEIGHTS.get(authority_level, 1.0)
    usage_weight = MEMORY_USAGE_SCORE_WEIGHTS.get(memory_usage, 1.0)
    return authority_weight * usage_weight


def _normalize_affinity(items: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in items or []:
        candidate = str(raw).strip().lower()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def _affinity_marker(items: list[str] | None) -> str:
    normalized = _normalize_affinity(items)
    return "|" + "|".join(normalized) + "|" if normalized else ""


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def _encode_cursor(offset: int, query: str) -> str:
    payload = json.dumps({"offset": offset, "query": _stable_hash(query)[:16]}, ensure_ascii=True)
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str | None, query: str) -> int:
    if not cursor:
        return 0
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    if payload.get("query") != _stable_hash(query)[:16]:
        return 0
    offset = payload.get("offset", 0)
    return offset if isinstance(offset, int) and offset >= 0 else 0


class KnowledgeMemoryService:
    def __init__(
        self,
        *,
        docs_root: Path | None = None,
        runtime_root: Path | None = None,
        vector_dimensions: int = DEFAULT_VECTOR_DIMENSIONS,
    ) -> None:
        settings = get_settings()
        self.docs_root = (docs_root or settings.knowledge_docs_root).resolve()
        self.runtime_root = (runtime_root or settings.llm_config_path.parent / "knowledge-memory").resolve()
        self.vector_dimensions = vector_dimensions
        self.policy_service = AgentMemoryPolicyService()
        self.manifest_path = self.runtime_root / "knowledge-corpus-manifest.json"
        self.lexical_index_path = self.runtime_root / "lexical-index.json"
        self.vector_index_path = self.runtime_root / "vector-index.json"
        self.search_audit_path = self.runtime_root / "search-audit.jsonl"
        self.scopes_root = self.runtime_root / "scopes"
        self.overrides_root = self.runtime_root / "overrides"
        self.workspace_docs_root = self.runtime_root / "managed-workspaces"

    def ensure_repo_docs_ingested(self, session: Session) -> KnowledgeIngestionReport:
        return self.sync_docs_corpus(session, scope=KnowledgeScope.platform, force=False)

    def sync_docs_corpus(
        self,
        session: Session,
        *,
        scope: KnowledgeScope | str = KnowledgeScope.platform,
        workspace_id: UUID | None = None,
        session_id: UUID | None = None,
        force: bool = False,
    ) -> KnowledgeIngestionReport:
        scope_config = self._resolve_scope_config(scope, workspace_id=workspace_id, session_id=session_id)
        parsed_documents = [
            self._parse_document(
                path,
                scope=scope_config.scope,
                workspace_id=scope_config.workspace_id,
                session_id=scope_config.session_id,
            )
            for path in self._discover_document_paths(scope_config.docs_root)
        ]
        corpus_hash = self._build_corpus_hash(parsed_documents)
        latest_run = self._latest_run(session, scope_config)

        manifest_path = self._manifest_path_for_scope(scope_config)
        lexical_index_path = self._lexical_index_path_for_scope(scope_config)
        vector_index_path = self._vector_index_path_for_scope(scope_config)

        if (
            not force
            and latest_run is not None
            and latest_run.corpus_hash == corpus_hash
            and manifest_path.exists()
            and lexical_index_path.exists()
            and vector_index_path.exists()
        ):
            return self._build_report_from_run(
                session,
                latest_run,
                documents=self._latest_document_entries(session, scope_config=scope_config),
                changed_paths=[],
            )

        scope_config.runtime_dir.mkdir(parents=True, exist_ok=True)
        scope_config.docs_root.mkdir(parents=True, exist_ok=True)
        scope_config.overrides_root.mkdir(parents=True, exist_ok=True)

        existing_documents = {
            item.relative_path: item
            for item in session.exec(self._document_query(scope_config)).all()
        }
        current_paths = {item.relative_path for item in parsed_documents}
        changed_paths: list[str] = []
        document_entries: list[KnowledgeDocumentEntry] = []

        for stale_path, stale_record in existing_documents.items():
            if stale_path in current_paths:
                continue
            changed_paths.append(f"deleted::{stale_path}")
            with session.no_autoflush:
                stale_sections = session.exec(
                    select(KnowledgeSectionRecord).where(KnowledgeSectionRecord.document_id == stale_record.id)
                ).all()
            for section in stale_sections:
                session.delete(section)
            session.flush()
            session.delete(stale_record)
            session.flush()
        now = utc_now()

        for parsed_document in parsed_documents:
            record = existing_documents.get(parsed_document.relative_path)
            metadata_exists = self._metadata_override_exists_for_document(parsed_document)
            governance_source = parsed_document if metadata_exists or record is None else None

            resolved_authority = (
                parsed_document.authority_level
                if governance_source is not None
                else (record.authority_level or parsed_document.authority_level)
            )
            resolved_memory_usage = (
                parsed_document.memory_usage
                if governance_source is not None
                else (record.memory_usage or parsed_document.memory_usage)
            )
            resolved_stage_affinity = (
                parsed_document.stage_affinity
                if governance_source is not None
                else (record.stage_affinity or parsed_document.stage_affinity)
            )
            resolved_agent_affinity = (
                parsed_document.agent_affinity
                if governance_source is not None
                else (record.agent_affinity or parsed_document.agent_affinity)
            )
            resolved_visibility = (
                parsed_document.visibility
                if governance_source is not None
                else (record.visibility if record is not None else parsed_document.visibility)
            )
            resolved_status = (
                parsed_document.status
                if governance_source is not None
                else (record.status if record is not None else parsed_document.status)
            )
            resolved_effective_from = (
                parsed_document.effective_from
                if governance_source is not None
                else (record.effective_from if record is not None else parsed_document.effective_from)
            )
            resolved_expires_at = (
                parsed_document.expires_at
                if governance_source is not None
                else (record.expires_at if record is not None else parsed_document.expires_at)
            )
            resolved_approved_by = (
                parsed_document.approved_by_user_id
                if governance_source is not None
                else (record.approved_by_user_id if record is not None else parsed_document.approved_by_user_id)
            )
            resolved_approved_at = (
                parsed_document.approved_at
                if governance_source is not None
                else (record.approved_at if record is not None else parsed_document.approved_at)
            )
            resolved_supersedes = (
                parsed_document.supersedes_document_id
                if governance_source is not None
                else (record.supersedes_document_id if record is not None else parsed_document.supersedes_document_id)
            )

            if resolved_expires_at is not None and resolved_expires_at <= now:
                resolved_status = KnowledgeDocumentStatus.expired

            content_changed = record is None or record.content_hash != parsed_document.content_hash
            governance_changed = record is None or any(
                [
                    record.scope != parsed_document.scope,
                    record.workspace_id != parsed_document.workspace_id,
                    record.session_id != parsed_document.session_id,
                    record.visibility != resolved_visibility,
                    record.status != resolved_status,
                    record.authority_level != resolved_authority,
                    record.memory_usage != resolved_memory_usage,
                    record.stage_affinity != resolved_stage_affinity,
                    record.agent_affinity != resolved_agent_affinity,
                    record.effective_from != resolved_effective_from,
                    record.expires_at != resolved_expires_at,
                    record.approved_by_user_id != resolved_approved_by,
                    record.approved_at != resolved_approved_at,
                    record.supersedes_document_id != resolved_supersedes,
                ]
            )
            changed = content_changed or governance_changed
            if changed:
                changed_paths.append(parsed_document.relative_path if content_changed else f"metadata::{parsed_document.relative_path}")
            version_number = 1 if record is None else record.version_number + (1 if changed else 0)

            if record is None:
                record = KnowledgeDocumentRecord(
                    source_root=parsed_document.source_root,
                    scope=parsed_document.scope,
                    workspace_id=parsed_document.workspace_id,
                    session_id=parsed_document.session_id,
                    relative_path=parsed_document.relative_path,
                    title=parsed_document.title,
                    format=parsed_document.format,
                    visibility=resolved_visibility,
                    status=resolved_status,
                    authority_level=resolved_authority,
                    memory_usage=resolved_memory_usage,
                    stage_affinity=resolved_stage_affinity,
                    stage_affinity_text=_affinity_marker(resolved_stage_affinity),
                    agent_affinity=resolved_agent_affinity,
                    agent_affinity_text=_affinity_marker(resolved_agent_affinity),
                    content_hash=parsed_document.content_hash,
                    version_number=version_number,
                    file_size_bytes=parsed_document.file_size_bytes,
                    word_count=len(_tokenize(parsed_document.content_text)),
                    section_count=len(parsed_document.sections),
                    source_lineage=parsed_document.source_lineage,
                    approved_by_user_id=resolved_approved_by,
                    approved_at=resolved_approved_at,
                    effective_from=resolved_effective_from,
                    expires_at=resolved_expires_at,
                    supersedes_document_id=resolved_supersedes,
                )
                session.add(record)
            else:
                record.source_root = parsed_document.source_root
                record.scope = parsed_document.scope
                record.workspace_id = parsed_document.workspace_id
                record.session_id = parsed_document.session_id
                record.title = parsed_document.title
                record.format = parsed_document.format
                record.visibility = resolved_visibility
                record.status = resolved_status
                record.authority_level = resolved_authority
                record.memory_usage = resolved_memory_usage
                record.stage_affinity = resolved_stage_affinity
                record.stage_affinity_text = _affinity_marker(resolved_stage_affinity)
                record.agent_affinity = resolved_agent_affinity
                record.agent_affinity_text = _affinity_marker(resolved_agent_affinity)
                record.content_hash = parsed_document.content_hash
                record.version_number = version_number
                record.file_size_bytes = parsed_document.file_size_bytes
                record.word_count = len(_tokenize(parsed_document.content_text))
                record.section_count = len(parsed_document.sections)
                record.source_lineage = parsed_document.source_lineage
                record.approved_by_user_id = resolved_approved_by
                record.approved_at = resolved_approved_at
                record.effective_from = resolved_effective_from
                record.expires_at = resolved_expires_at
                record.supersedes_document_id = resolved_supersedes
                record.updated_at = now
                session.add(record)

            session.flush()
            existing_sections = session.exec(
                select(KnowledgeSectionRecord).where(KnowledgeSectionRecord.document_id == record.id)
            ).all()
            if changed or not existing_sections:
                for section in existing_sections:
                    session.delete(section)
                session.flush()
                for parsed_section in parsed_document.sections:
                    session.add(
                        KnowledgeSectionRecord(
                            document_id=record.id,
                            source_root=record.source_root,
                            scope=record.scope,
                            workspace_id=record.workspace_id,
                            session_id=record.session_id,
                            relative_path=record.relative_path,
                            section_key=parsed_section.section_key,
                            title=parsed_section.title,
                            heading_path=parsed_section.heading_path,
                            heading_level=parsed_section.heading_level,
                            sort_order=parsed_section.sort_order,
                            start_line=parsed_section.start_line,
                            end_line=parsed_section.end_line,
                            visibility=record.visibility,
                            status=record.status,
                            authority_level=record.authority_level,
                            memory_usage=record.memory_usage,
                            stage_affinity=list(record.stage_affinity),
                            stage_affinity_text=record.stage_affinity_text,
                            agent_affinity=list(record.agent_affinity),
                            agent_affinity_text=record.agent_affinity_text,
                            document_version_number=version_number,
                            content_hash=parsed_section.content_hash,
                            source_lineage=parsed_section.source_lineage,
                            content_text=parsed_section.content_text,
                            token_count=len(parsed_section.lexical_terms),
                            lexical_terms=parsed_section.lexical_terms,
                            vector_payload=parsed_section.vector_payload,
                            metadata_payload={
                                "authority_level": record.authority_level,
                                "memory_usage": record.memory_usage,
                                "scope": record.scope.value,
                            },
                            approved_by_user_id=record.approved_by_user_id,
                            approved_at=record.approved_at,
                            effective_from=record.effective_from,
                            expires_at=record.expires_at,
                        )
                    )

            document_entries.append(self._document_entry_from_record(record))

        lexical_index = self._build_lexical_index(parsed_documents)
        vector_index = self._build_vector_index(parsed_documents)
        self._write_runtime_artifacts(
            scope_config=scope_config,
            parsed_documents=parsed_documents,
            corpus_hash=corpus_hash,
            lexical_index=lexical_index,
            vector_index=vector_index,
        )

        run = KnowledgeIngestionRunRecord(
            source_root=scope_config.source_root,
            scope=scope_config.scope,
            workspace_id=scope_config.workspace_id,
            session_id=scope_config.session_id,
            status="ready",
            corpus_hash=corpus_hash,
            document_count=len(parsed_documents),
            changed_document_count=len(changed_paths),
            unchanged_document_count=max(len(parsed_documents) - len(changed_paths), 0),
            section_count=sum(len(item.sections) for item in parsed_documents),
            lexical_term_count=len(lexical_index["postings"]),
            vector_dimensions=self.vector_dimensions,
            filesystem_manifest_path=str(manifest_path),
            lexical_index_path=str(lexical_index_path),
            vector_index_path=str(vector_index_path),
            changed_paths=changed_paths,
            metadata_payload={
                "docs_root": str(scope_config.docs_root),
                "runtime_dir": str(scope_config.runtime_dir),
            },
        )
        session.add(run)
        session.flush()

        for record in session.exec(self._document_query(scope_config)).all():
            record.last_ingestion_run_id = run.id
            record.updated_at = now
            session.add(record)

        session.commit()
        return self._build_report_from_run(session, run, documents=document_entries, changed_paths=changed_paths)

    def build_corpus_status(
        self,
        session: Session,
        *,
        scope: KnowledgeScope | str = KnowledgeScope.platform,
        workspace_id: UUID | None = None,
        session_id: UUID | None = None,
        ensure_ingested: bool = True,
    ) -> KnowledgeCorpusStatus:
        scope_config = self._resolve_scope_config(scope, workspace_id=workspace_id, session_id=session_id)
        latest_run = self._latest_run(session, scope_config)
        if latest_run is None and ensure_ingested:
            report = self.sync_docs_corpus(
                session,
                scope=scope_config.scope,
                workspace_id=scope_config.workspace_id,
                session_id=scope_config.session_id,
                force=False,
            )
            latest_run = session.get(KnowledgeIngestionRunRecord, report.run_id) if report.run_id is not None else None

        if latest_run is None:
            return KnowledgeCorpusStatus(
                source_root=scope_config.source_root,
                scope=scope_config.scope,
                workspace_id=scope_config.workspace_id,
                session_id=scope_config.session_id,
                status="empty",
            )

        return KnowledgeCorpusStatus(
            source_root=latest_run.source_root,
            scope=latest_run.scope,
            workspace_id=latest_run.workspace_id,
            session_id=latest_run.session_id,
            status=latest_run.status,
            corpus_hash=latest_run.corpus_hash,
            document_count=latest_run.document_count,
            section_count=latest_run.section_count,
            lexical_term_count=latest_run.lexical_term_count,
            vector_dimensions=latest_run.vector_dimensions,
            filesystem_manifest_path=latest_run.filesystem_manifest_path,
            lexical_index_path=latest_run.lexical_index_path,
            vector_index_path=latest_run.vector_index_path,
            last_ingested_at=latest_run.created_at,
            latest_documents=self._latest_document_entries(session, scope_config=scope_config),
        )

    def search(
        self,
        session: Session,
        *,
        query: str,
        limit: int = 10,
        ensure_ingested: bool = True,
        workspace_id: UUID | None = None,
        session_id: UUID | None = None,
        stage: str | None = None,
        authority_allowlist: list[str] | None = None,
        cursor: str | None = None,
    ) -> KnowledgeSearchResponse:
        return self._search_with_filters(
            session,
            query=query,
            role="",
            limit=limit,
            ensure_ingested=ensure_ingested,
            workspace_id=workspace_id,
            session_id=session_id,
            stage=stage,
            authority_allowlist=authority_allowlist or [],
            cursor=cursor,
            governed=False,
        )

    def search_governed(
        self,
        session: Session,
        *,
        query: str,
        role: str,
        workspace_id: UUID | None = None,
        session_id: UUID | None = None,
        stage: str | None = None,
        authority_allowlist: list[str] | None = None,
        corpus_hash: str | None = None,
        limit: int = 10,
        ensure_ingested: bool = True,
        cursor: str | None = None,
    ) -> KnowledgeSearchResponse:
        response = self._search_with_filters(
            session,
            query=query,
            role=role,
            limit=limit,
            ensure_ingested=ensure_ingested,
            workspace_id=workspace_id,
            session_id=session_id,
            stage=stage,
            authority_allowlist=authority_allowlist or [],
            cursor=cursor,
            governed=True,
        )
        if corpus_hash and response.corpus_hash and response.corpus_hash != corpus_hash:
            return KnowledgeSearchResponse(
                query=query,
                role=role,
                total_hits=0,
                grounded_hits=0,
                corpus_hash=response.corpus_hash,
                evidence_status="no_evidence",
                absence_reason="requested_corpus_hash_not_available",
                applied_filters=response.applied_filters,
                authorized_scopes=response.authorized_scopes,
                items=[],
            )
        return response

    def upsert_managed_document(
        self,
        session: Session,
        *,
        payload: KnowledgeManagedDocumentUpsertRequest,
        workspace_id: UUID | None,
        actor_user_id: UUID | None,
    ) -> KnowledgeDocumentEntry:
        scope_config = self._resolve_scope_config(
            payload.scope,
            workspace_id=workspace_id,
            session_id=payload.session_id,
        )
        relative_path = self._normalize_relative_path(payload.relative_path)
        document_path = scope_config.docs_root / relative_path
        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(sanitize_text_content(payload.content_text).strip() + "\n", encoding="utf-8")
        metadata = {
            "authority_level": payload.authority_level.strip() or "operational",
            "memory_usage": payload.memory_usage.strip() or "candidate_retrieval",
            "stage_affinity": _normalize_affinity(payload.stage_affinity),
            "agent_affinity": _normalize_affinity(payload.agent_affinity),
            "visibility": (payload.visibility or self._default_visibility_for_scope(scope_config.scope)).value,
            "status": payload.status.value,
            "effective_from": payload.effective_from.isoformat() if payload.effective_from is not None else None,
            "expires_at": payload.expires_at.isoformat() if payload.expires_at is not None else None,
            "approved_by_user_id": str(actor_user_id) if actor_user_id is not None else "",
            "approved_at": utc_now().isoformat() if payload.status == KnowledgeDocumentStatus.approved else None,
        }
        self._write_metadata_override(scope_config, relative_path, metadata)
        self.sync_docs_corpus(
            session,
            scope=scope_config.scope,
            workspace_id=scope_config.workspace_id,
            session_id=scope_config.session_id,
            force=True,
        )
        record = session.exec(
            self._document_query(scope_config).where(
                KnowledgeDocumentRecord.relative_path == self._logical_path(scope_config, relative_path)
            )
        ).first()
        if record is None:
            raise LookupError("No se pudo materializar el documento gestionado.")
        return self._document_entry_from_record(record)

    def update_document_governance(
        self,
        session: Session,
        *,
        document_id: UUID,
        payload: KnowledgeDocumentGovernancePatchRequest,
        actor_user_id: UUID | None,
    ) -> KnowledgeDocumentEntry:
        record = session.get(KnowledgeDocumentRecord, document_id)
        if record is None:
            raise LookupError("No existe el documento solicitado.")

        scope_config = self._resolve_scope_config(record.scope, workspace_id=record.workspace_id, session_id=record.session_id)
        relative_path = self._document_relative_path(scope_config, record.relative_path)
        existing_metadata = self._read_metadata_override(scope_config, relative_path)
        metadata = {
            **existing_metadata,
            "authority_level": payload.authority_level.strip() if isinstance(payload.authority_level, str) else record.authority_level,
            "memory_usage": payload.memory_usage.strip() if isinstance(payload.memory_usage, str) else record.memory_usage,
            "stage_affinity": _normalize_affinity(payload.stage_affinity if payload.stage_affinity is not None else record.stage_affinity),
            "agent_affinity": _normalize_affinity(payload.agent_affinity if payload.agent_affinity is not None else record.agent_affinity),
            "visibility": (payload.visibility or record.visibility).value,
            "status": (payload.status or record.status).value,
            "effective_from": (
                payload.effective_from.isoformat()
                if payload.effective_from is not None
                else (record.effective_from.isoformat() if record.effective_from is not None else None)
            ),
            "expires_at": (
                payload.expires_at.isoformat()
                if payload.expires_at is not None
                else (record.expires_at.isoformat() if record.expires_at is not None else None)
            ),
        }
        if payload.status == KnowledgeDocumentStatus.approved or record.status == KnowledgeDocumentStatus.approved:
            metadata["approved_by_user_id"] = str(actor_user_id) if actor_user_id is not None else str(record.approved_by_user_id or "")
            metadata["approved_at"] = utc_now().isoformat()

        self._write_metadata_override(scope_config, relative_path, metadata)
        self.sync_docs_corpus(
            session,
            scope=scope_config.scope,
            workspace_id=scope_config.workspace_id,
            session_id=scope_config.session_id,
            force=True,
        )
        refreshed = session.get(KnowledgeDocumentRecord, document_id)
        if refreshed is None:
            raise LookupError("No se pudo refrescar el documento tras la actualizacion.")
        return self._document_entry_from_record(refreshed)

    def _search_with_filters(
        self,
        session: Session,
        *,
        query: str,
        role: str,
        limit: int,
        ensure_ingested: bool,
        workspace_id: UUID | None,
        session_id: UUID | None,
        stage: str | None,
        authority_allowlist: list[str],
        cursor: str | None,
        governed: bool,
    ) -> KnowledgeSearchResponse:
        if ensure_ingested:
            self.ensure_repo_docs_ingested(session)
            if workspace_id is not None:
                self.sync_docs_corpus(session, scope=KnowledgeScope.workspace, workspace_id=workspace_id, force=False)
            if workspace_id is not None and session_id is not None:
                self.sync_docs_corpus(
                    session,
                    scope=KnowledgeScope.session,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    force=False,
                )

        query_text = query.strip()
        normalized_role = role.strip()
        authorized_scopes = [KnowledgeScope.platform.value]
        if workspace_id is not None:
            authorized_scopes.append(KnowledgeScope.workspace.value)
        if workspace_id is not None and session_id is not None:
            authorized_scopes.append(KnowledgeScope.session.value)

        allowed_usages = set(self.policy_service.allowed_knowledge_memory_usages(normalized_role)) if governed else set()
        applied_filters = [
            f"grounding=lexical>0_or_vector>={MIN_GROUNDED_VECTOR_SCORE:.2f}",
            "authorized_scopes=" + ",".join(authorized_scopes),
        ]
        if normalized_role:
            applied_filters.append(f"role={normalized_role}")
        if stage and stage.strip():
            applied_filters.append(f"stage={stage.strip().lower()}")
        if authority_allowlist:
            applied_filters.append("authority=" + ",".join(sorted({item.strip().lower() for item in authority_allowlist if item.strip()})))

        if governed and not allowed_usages:
            response = KnowledgeSearchResponse(
                query=query_text,
                role=normalized_role,
                total_hits=0,
                grounded_hits=0,
                corpus_hash=self._combined_corpus_hash(session, workspace_id=workspace_id, session_id=session_id),
                evidence_status="no_evidence",
                absence_reason="role_has_no_long_term_retrieval_scope",
                applied_filters=[*applied_filters, "memory_usage=none"],
                authorized_scopes=authorized_scopes,
                items=[],
            )
            self._append_search_audit(response=response, workspace_id=workspace_id, session_id=session_id, stage=stage)
            return response

        statement = select(KnowledgeSectionRecord).where(
            KnowledgeSectionRecord.status == KnowledgeDocumentStatus.approved,
            or_(KnowledgeSectionRecord.effective_from == None, KnowledgeSectionRecord.effective_from <= utc_now()),  # noqa: E711
            or_(KnowledgeSectionRecord.expires_at == None, KnowledgeSectionRecord.expires_at > utc_now()),  # noqa: E711
            self._authorized_scope_clause(workspace_id=workspace_id, session_id=session_id),
        )

        if governed:
            statement = statement.where(KnowledgeSectionRecord.memory_usage.in_(sorted(allowed_usages)))
            applied_filters.append("memory_usage=" + ",".join(sorted(allowed_usages)))
        if authority_allowlist:
            normalized_authorities = sorted({item.strip().lower() for item in authority_allowlist if item.strip()})
            if normalized_authorities:
                statement = statement.where(KnowledgeSectionRecord.authority_level.in_(normalized_authorities))
        if stage and stage.strip():
            marker = f"|{stage.strip().lower()}|"
            statement = statement.where(
                or_(
                    KnowledgeSectionRecord.stage_affinity_text == "",
                    KnowledgeSectionRecord.stage_affinity_text.like(f"%{marker}%"),
                )
            )

        sections = session.exec(statement).all()
        combined_corpus_hash = self._combined_corpus_hash(session, workspace_id=workspace_id, session_id=session_id)
        if not sections:
            response = KnowledgeSearchResponse(
                query=query_text,
                role=normalized_role,
                total_hits=0,
                grounded_hits=0,
                corpus_hash=combined_corpus_hash,
                evidence_status="no_evidence",
                absence_reason="no_authorized_sections_for_scope",
                applied_filters=applied_filters,
                authorized_scopes=authorized_scopes,
                items=[],
            )
            self._append_search_audit(response=response, workspace_id=workspace_id, session_id=session_id, stage=stage)
            return response

        query_tokens = _tokenize(query_text)
        query_vector = _build_vector(query_tokens, dimensions=self.vector_dimensions)
        token_document_frequency: Counter[str] = Counter()
        per_section_token_counts: dict[str, Counter[str]] = {}
        for section in sections:
            counts = Counter(section.lexical_terms)
            per_section_token_counts[section.section_key] = counts
            for token in counts:
                token_document_frequency[token] += 1

        scored_hits: list[KnowledgeSearchHit] = []
        total_sections = max(len(sections), 1)
        for section in sections:
            lexical_score = 0.0
            section_counts = per_section_token_counts.get(section.section_key, Counter())
            for token in query_tokens:
                tf = section_counts.get(token, 0)
                if tf <= 0:
                    continue
                df = int(token_document_frequency.get(token, 1))
                idf = math.log((1 + total_sections) / (1 + df)) + 1.0
                lexical_score += (1.0 + math.log(tf)) * idf
            vector_score = max(_cosine_similarity(query_vector, list(section.vector_payload)), 0.0)
            if lexical_score <= 0 and vector_score < MIN_GROUNDED_VECTOR_SCORE:
                continue
            precedence_rank = self._precedence_rank(section)
            score = round(
                ((lexical_score * 0.72) + (vector_score * 0.28))
                * _governance_weight(section.authority_level, section.memory_usage)
                * (1.0 + (precedence_rank * 0.05)),
                6,
            )
            preview = re.sub(r"\s+", " ", section.content_text).strip()[:280]
            scored_hits.append(
                KnowledgeSearchHit(
                    document_id=section.document_id,
                    scope=section.scope,
                    workspace_id=section.workspace_id,
                    session_id=section.session_id,
                    relative_path=section.relative_path,
                    section_key=section.section_key,
                    title=section.title,
                    heading_path=list(section.heading_path),
                    visibility=section.visibility,
                    status=section.status,
                    authority_level=section.authority_level,
                    memory_usage=section.memory_usage,
                    stage_affinity=list(section.stage_affinity),
                    agent_affinity=list(section.agent_affinity),
                    source_lineage=section.source_lineage,
                    preview=preview,
                    score=score,
                    lexical_score=round(lexical_score, 6),
                    vector_score=round(vector_score, 6),
                    version_number=section.document_version_number,
                    approved_at=section.approved_at,
                    effective_from=section.effective_from,
                    expires_at=section.expires_at,
                )
            )

        if not scored_hits:
            response = KnowledgeSearchResponse(
                query=query_text,
                role=normalized_role,
                total_hits=0,
                grounded_hits=0,
                corpus_hash=combined_corpus_hash,
                evidence_status="no_evidence",
                absence_reason="no_grounded_evidence_after_filters",
                applied_filters=applied_filters,
                authorized_scopes=authorized_scopes,
                items=[],
            )
            self._append_search_audit(response=response, workspace_id=workspace_id, session_id=session_id, stage=stage)
            return response

        scored_hits.sort(
            key=lambda item: (
                -self._precedence_rank(item),
                -item.score,
                item.relative_path,
                item.section_key,
            )
        )
        diversified_hits = self._diversify_hits(scored_hits)
        offset = _decode_cursor(cursor, query_text)
        page_items = diversified_hits[offset : offset + max(1, min(limit, 25))]
        next_cursor = ""
        if offset + len(page_items) < len(diversified_hits):
            next_cursor = _encode_cursor(offset + len(page_items), query_text)

        response = KnowledgeSearchResponse(
            query=query_text,
            role=normalized_role,
            total_hits=len(diversified_hits),
            grounded_hits=len(page_items),
            corpus_hash=combined_corpus_hash,
            evidence_status="grounded" if page_items else "no_evidence",
            absence_reason="" if page_items else "no_hits_for_cursor_window",
            applied_filters=applied_filters,
            authorized_scopes=authorized_scopes,
            citations=[item.source_lineage for item in page_items[:5] if item.source_lineage],
            next_cursor=next_cursor,
            discarded_hits=max(len(sections) - len(diversified_hits), 0),
            items=page_items,
        )
        self._append_search_audit(response=response, workspace_id=workspace_id, session_id=session_id, stage=stage)
        return response

    def _authorized_scope_clause(self, *, workspace_id: UUID | None, session_id: UUID | None):
        clauses = [KnowledgeSectionRecord.scope == KnowledgeScope.platform]
        if workspace_id is not None:
            clauses.append(
                (
                    (KnowledgeSectionRecord.scope == KnowledgeScope.workspace)
                    & (KnowledgeSectionRecord.workspace_id == workspace_id)
                )
            )
        if workspace_id is not None and session_id is not None:
            clauses.append(
                (
                    (KnowledgeSectionRecord.scope == KnowledgeScope.session)
                    & (KnowledgeSectionRecord.workspace_id == workspace_id)
                    & (KnowledgeSectionRecord.session_id == session_id)
                )
            )
        return or_(*clauses)

    def _precedence_rank(self, item: KnowledgeSectionRecord | KnowledgeSearchHit) -> int:
        if item.scope == KnowledgeScope.platform:
            return 4
        if item.scope == KnowledgeScope.workspace and item.status == KnowledgeDocumentStatus.approved:
            return 3
        if item.scope == KnowledgeScope.session and item.status == KnowledgeDocumentStatus.approved:
            return 2
        return 1

    def _diversify_hits(self, items: list[KnowledgeSearchHit]) -> list[KnowledgeSearchHit]:
        primary: list[KnowledgeSearchHit] = []
        secondary: list[KnowledgeSearchHit] = []
        seen_documents: set[str] = set()
        for item in items:
            if item.relative_path not in seen_documents:
                seen_documents.add(item.relative_path)
                primary.append(item)
            else:
                secondary.append(item)
        return [*primary, *secondary]

    def _combined_corpus_hash(
        self,
        session: Session,
        *,
        workspace_id: UUID | None,
        session_id: UUID | None,
    ) -> str:
        scope_configs = [self._resolve_scope_config(KnowledgeScope.platform)]
        if workspace_id is not None:
            scope_configs.append(self._resolve_scope_config(KnowledgeScope.workspace, workspace_id=workspace_id))
        if workspace_id is not None and session_id is not None:
            scope_configs.append(
                self._resolve_scope_config(KnowledgeScope.session, workspace_id=workspace_id, session_id=session_id)
            )
        fragments: list[str] = []
        for scope_config in scope_configs:
            run = self._latest_run(session, scope_config)
            if run is not None and run.corpus_hash:
                fragments.append(f"{scope_config.scope.value}:{run.corpus_hash}")
        return _stable_hash("\n".join(sorted(fragments))) if fragments else ""

    def _latest_run(self, session: Session, scope_config: ScopeConfig) -> KnowledgeIngestionRunRecord | None:
        statement = (
            select(KnowledgeIngestionRunRecord)
            .where(KnowledgeIngestionRunRecord.scope == scope_config.scope)
            .order_by(KnowledgeIngestionRunRecord.created_at.desc())
        )
        if scope_config.workspace_id is not None:
            statement = statement.where(KnowledgeIngestionRunRecord.workspace_id == scope_config.workspace_id)
        else:
            statement = statement.where(KnowledgeIngestionRunRecord.workspace_id == None)  # noqa: E711
        if scope_config.session_id is not None:
            statement = statement.where(KnowledgeIngestionRunRecord.session_id == scope_config.session_id)
        else:
            statement = statement.where(KnowledgeIngestionRunRecord.session_id == None)  # noqa: E711
        return session.exec(statement).first()

    def _document_query(self, scope_config: ScopeConfig):
        statement = select(KnowledgeDocumentRecord).where(KnowledgeDocumentRecord.scope == scope_config.scope)
        if scope_config.workspace_id is not None:
            statement = statement.where(KnowledgeDocumentRecord.workspace_id == scope_config.workspace_id)
        else:
            statement = statement.where(KnowledgeDocumentRecord.workspace_id == None)  # noqa: E711
        if scope_config.session_id is not None:
            statement = statement.where(KnowledgeDocumentRecord.session_id == scope_config.session_id)
        else:
            statement = statement.where(KnowledgeDocumentRecord.session_id == None)  # noqa: E711
        return statement

    def _resolve_scope_config(
        self,
        scope: KnowledgeScope | str,
        *,
        workspace_id: UUID | None = None,
        session_id: UUID | None = None,
    ) -> ScopeConfig:
        resolved_scope = KnowledgeScope(scope)
        if resolved_scope == KnowledgeScope.platform:
            return ScopeConfig(
                scope=resolved_scope,
                source_root="Docs",
                docs_root=self.docs_root,
                runtime_dir=self.runtime_root,
                overrides_root=self.overrides_root / "platform",
            )
        if workspace_id is None:
            raise ValueError("workspace_id es obligatorio para knowledge scope workspace/session.")
        if resolved_scope == KnowledgeScope.workspace:
            workspace_root = self.workspace_docs_root / str(workspace_id)
            return ScopeConfig(
                scope=resolved_scope,
                source_root=f"Workspace/{workspace_id}",
                docs_root=workspace_root / "docs",
                runtime_dir=self.scopes_root / "workspace" / str(workspace_id),
                overrides_root=self.overrides_root / "workspace" / str(workspace_id),
                workspace_id=workspace_id,
            )
        if session_id is None:
            raise ValueError("session_id es obligatorio para knowledge scope session.")
        session_root = self.workspace_docs_root / str(workspace_id) / "sessions" / str(session_id)
        return ScopeConfig(
            scope=resolved_scope,
            source_root=f"Session/{workspace_id}/{session_id}",
            docs_root=session_root / "docs",
            runtime_dir=self.scopes_root / "session" / str(workspace_id) / str(session_id),
            overrides_root=self.overrides_root / "session" / str(workspace_id) / str(session_id),
            workspace_id=workspace_id,
            session_id=session_id,
        )

    def _default_visibility_for_scope(self, scope: KnowledgeScope) -> KnowledgeVisibility:
        if scope == KnowledgeScope.workspace:
            return KnowledgeVisibility.workspace
        if scope == KnowledgeScope.session:
            return KnowledgeVisibility.session
        return KnowledgeVisibility.platform

    def _discover_document_paths(self, docs_root: Path) -> list[Path]:
        if not docs_root.exists():
            return []
        paths = [
            path
            for path in docs_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and not path.name.endswith(".meta.json")
        ]
        return sorted(paths, key=lambda path: path.relative_to(docs_root).as_posix().lower())

    def _parse_document(
        self,
        path: Path,
        *,
        scope: KnowledgeScope | str = KnowledgeScope.platform,
        workspace_id: UUID | None = None,
        session_id: UUID | None = None,
    ) -> ParsedDocument:
        scope_config = self._resolve_scope_config(scope, workspace_id=workspace_id, session_id=session_id)
        raw_text = self._read_document_text(path)
        relative_under_root = path.relative_to(scope_config.docs_root)
        logical_path = self._logical_path(scope_config, relative_under_root)
        classification = self._classify_document(logical_path)
        overrides = self._read_metadata_override(scope_config, relative_under_root)
        normalized_text = _normalize_newlines(raw_text).strip()
        content_hash = _stable_hash(normalized_text)
        source_lineage = f"{logical_path}::doc::{content_hash[:16]}"
        tokens = _tokenize(normalized_text)
        sections = self._split_sections(
            logical_path=logical_path,
            content_text=normalized_text,
            document_hash=content_hash,
        )
        title = sections[0].title if sections else path.stem.replace("_", " ").replace("-", " ").title()
        visibility = KnowledgeVisibility(
            overrides.get("visibility", self._default_visibility_for_scope(scope_config.scope).value)
        )
        status = KnowledgeDocumentStatus(overrides.get("status", KnowledgeDocumentStatus.approved.value))
        effective_from = _coerce_datetime(overrides.get("effective_from"))
        expires_at = _coerce_datetime(overrides.get("expires_at"))
        approved_by_user_id = _coerce_uuid(overrides.get("approved_by_user_id"))
        approved_at = _coerce_datetime(overrides.get("approved_at"))
        supersedes_document_id = _coerce_uuid(overrides.get("supersedes_document_id"))
        authority_level = str(overrides.get("authority_level", classification["authority_level"])).strip().lower()
        memory_usage = str(overrides.get("memory_usage", classification["memory_usage"])).strip().lower()
        stage_affinity = _normalize_affinity(
            overrides.get("stage_affinity") if "stage_affinity" in overrides else classification["stage_affinity"]
        )
        agent_affinity = _normalize_affinity(
            overrides.get("agent_affinity") if "agent_affinity" in overrides else classification["agent_affinity"]
        )
        return ParsedDocument(
            source_root=scope_config.source_root,
            scope=scope_config.scope,
            workspace_id=scope_config.workspace_id,
            session_id=scope_config.session_id,
            relative_path=logical_path,
            title=title,
            format=path.suffix.lower().lstrip("."),
            visibility=visibility,
            status=status,
            authority_level=authority_level,
            memory_usage=memory_usage,
            stage_affinity=stage_affinity,
            agent_affinity=agent_affinity,
            effective_from=effective_from,
            expires_at=expires_at,
            approved_by_user_id=approved_by_user_id,
            approved_at=approved_at,
            supersedes_document_id=supersedes_document_id,
            content_text=normalized_text,
            content_hash=content_hash,
            file_size_bytes=path.stat().st_size,
            word_count=len(tokens),
            source_lineage=source_lineage,
            sections=sections,
        )

    def _read_document_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        raw_text = read_sanitized_utf8_text(path)
        if suffix == ".json":
            payload = json.loads(raw_text)
            return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        return raw_text

    def _split_sections(self, *, logical_path: str, content_text: str, document_hash: str) -> list[ParsedSection]:
        if logical_path.endswith(".md"):
            return self._split_markdown_sections(logical_path=logical_path, content_text=content_text, document_hash=document_hash)
        return self._build_single_section(logical_path=logical_path, content_text=content_text, document_hash=document_hash)

    def _split_markdown_sections(self, *, logical_path: str, content_text: str, document_hash: str) -> list[ParsedSection]:
        lines = content_text.split("\n")
        sections: list[ParsedSection] = []
        heading_stack: list[str] = []
        in_code_block = False
        current_title = Path(logical_path).stem.replace("_", " ").replace("-", " ").title()
        current_heading_level = 0
        current_start_line = 1
        current_lines: list[str] = []

        def finalize(end_line: int) -> None:
            nonlocal sections, current_lines, current_start_line, current_title, current_heading_level, heading_stack
            body = "\n".join(current_lines).strip()
            if not body and sections:
                current_lines = []
                return
            section_order = len(sections) + 1
            heading_path = heading_stack[:] if heading_stack else [current_title]
            slug = _slugify(" ".join(heading_path), fallback=f"section-{section_order}")
            section_hash = _stable_hash(body or current_title)
            section_key = f"{logical_path}#s{section_order:03d}:{slug}"
            sections.append(
                ParsedSection(
                    section_key=section_key,
                    title=current_title,
                    heading_path=heading_path,
                    heading_level=current_heading_level,
                    sort_order=section_order,
                    start_line=current_start_line,
                    end_line=max(current_start_line, end_line),
                    content_text=body or current_title,
                    content_hash=section_hash,
                    source_lineage=f"{logical_path}::section::{section_order}::{document_hash[:8]}::{section_hash[:8]}",
                    lexical_terms=_tokenize(body or current_title),
                    vector_payload=_build_vector(_tokenize(body or current_title), dimensions=self.vector_dimensions),
                )
            )
            current_lines = []

        for line_number, line in enumerate(lines, start=1):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
            heading_match = None if in_code_block else HEADING_RE.match(line)
            if heading_match is None:
                current_lines.append(line)
                continue

            if current_lines or not sections:
                finalize(line_number - 1)

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_title = title
            current_heading_level = level
            current_start_line = line_number
            current_lines = [line]

        finalize(len(lines) or 1)
        return sections or self._build_single_section(logical_path=logical_path, content_text=content_text, document_hash=document_hash)

    def _build_single_section(self, *, logical_path: str, content_text: str, document_hash: str) -> list[ParsedSection]:
        title = Path(logical_path).stem.replace("_", " ").replace("-", " ").title()
        body = content_text.strip() or title
        section_hash = _stable_hash(body)
        return [
            ParsedSection(
                section_key=f"{logical_path}#s001:document-root",
                title=title,
                heading_path=[title],
                heading_level=0,
                sort_order=1,
                start_line=1,
                end_line=max(1, len(content_text.split("\n"))),
                content_text=body,
                content_hash=section_hash,
                source_lineage=f"{logical_path}::section::1::{document_hash[:8]}::{section_hash[:8]}",
                lexical_terms=_tokenize(body),
                vector_payload=_build_vector(_tokenize(body), dimensions=self.vector_dimensions),
            )
        ]

    def _classify_document(self, logical_path: str) -> dict[str, Any]:
        if TAXONOMY_PATH.exists():
            payload = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
            for rule in payload.get("rules", []):
                if self._matches_rule(logical_path, rule):
                    return {
                        "authority_level": str(rule.get("authority_level", "operational")),
                        "memory_usage": str(rule.get("memory_usage", "candidate_retrieval")),
                        "stage_affinity": [str(item) for item in rule.get("stage_affinity", []) if str(item).strip()],
                        "agent_affinity": [str(item) for item in rule.get("agent_affinity", []) if str(item).strip()],
                    }
        return {
            "authority_level": "operational",
            "memory_usage": "candidate_retrieval",
            "stage_affinity": ["runtime"],
            "agent_affinity": ["retrieval", "memory"],
        }

    def _matches_rule(self, logical_path: str, rule: dict[str, Any]) -> bool:
        exclude_prefixes = [str(item) for item in rule.get("exclude_prefixes", [])]
        if any(logical_path.startswith(prefix) for prefix in exclude_prefixes):
            return False
        exclude_filenames = {str(item) for item in rule.get("exclude_filenames", [])}
        if Path(logical_path).name in exclude_filenames:
            return False

        include_prefixes = [str(item) for item in rule.get("include_prefixes", [])]
        include_filenames = {str(item) for item in rule.get("include_filenames", [])}
        include_suffixes = [str(item) for item in rule.get("include_suffixes", [])]

        if include_prefixes and any(logical_path.startswith(prefix) for prefix in include_prefixes):
            return True
        if include_filenames and Path(logical_path).name in include_filenames:
            return True
        if include_suffixes and any(logical_path.endswith(suffix) for suffix in include_suffixes):
            return True
        return False

    def _build_corpus_hash(self, parsed_documents: list[ParsedDocument]) -> str:
        digest_input = "\n".join(
            "|".join(
                [
                    item.scope.value,
                    str(item.workspace_id or ""),
                    str(item.session_id or ""),
                    item.relative_path,
                    item.content_hash,
                    item.authority_level,
                    item.memory_usage,
                    item.visibility.value,
                    item.status.value,
                    item.effective_from.isoformat() if item.effective_from is not None else "",
                    item.expires_at.isoformat() if item.expires_at is not None else "",
                ]
            )
            for item in parsed_documents
        )
        return _stable_hash(digest_input)

    def _build_lexical_index(self, parsed_documents: list[ParsedDocument]) -> dict[str, Any]:
        postings: dict[str, list[list[Any]]] = defaultdict(list)
        document_frequency: dict[str, int] = defaultdict(int)
        section_count = 0
        for document in parsed_documents:
            for section in document.sections:
                section_count += 1
                counts = Counter(section.lexical_terms)
                for token, tf in counts.items():
                    postings[token].append([section.section_key, tf])
                for token in counts:
                    document_frequency[token] += 1
        return {
            "generated_at": utc_now().isoformat(),
            "section_count": section_count,
            "document_frequency": dict(sorted(document_frequency.items())),
            "postings": dict(sorted(postings.items())),
        }

    def _build_vector_index(self, parsed_documents: list[ParsedDocument]) -> dict[str, Any]:
        return {
            "generated_at": utc_now().isoformat(),
            "dimensions": self.vector_dimensions,
            "sections": {
                section.section_key: section.vector_payload
                for document in parsed_documents
                for section in document.sections
            },
        }

    def _write_runtime_artifacts(
        self,
        *,
        scope_config: ScopeConfig,
        parsed_documents: list[ParsedDocument],
        corpus_hash: str,
        lexical_index: dict[str, Any],
        vector_index: dict[str, Any],
    ) -> None:
        manifest_path = self._manifest_path_for_scope(scope_config)
        lexical_index_path = self._lexical_index_path_for_scope(scope_config)
        vector_index_path = self._vector_index_path_for_scope(scope_config)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        lexical_index_path.parent.mkdir(parents=True, exist_ok=True)
        vector_index_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_payload = {
            "generated_at": utc_now().isoformat(),
            "source_root": scope_config.source_root,
            "scope": scope_config.scope.value,
            "workspace_id": str(scope_config.workspace_id) if scope_config.workspace_id is not None else None,
            "session_id": str(scope_config.session_id) if scope_config.session_id is not None else None,
            "corpus_hash": corpus_hash,
            "document_count": len(parsed_documents),
            "section_count": sum(len(item.sections) for item in parsed_documents),
            "documents": [
                {
                    "relative_path": item.relative_path,
                    "title": item.title,
                    "format": item.format,
                    "visibility": item.visibility.value,
                    "status": item.status.value,
                    "authority_level": item.authority_level,
                    "memory_usage": item.memory_usage,
                    "stage_affinity": item.stage_affinity,
                    "agent_affinity": item.agent_affinity,
                    "content_hash": item.content_hash,
                    "source_lineage": item.source_lineage,
                    "sections": [
                        {
                            "section_key": section.section_key,
                            "title": section.title,
                            "source_lineage": section.source_lineage,
                            "content_hash": section.content_hash,
                            "start_line": section.start_line,
                            "end_line": section.end_line,
                        }
                        for section in item.sections
                    ],
                }
                for item in parsed_documents
            ],
        }
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        lexical_index_path.write_text(json.dumps(lexical_index, ensure_ascii=False, indent=2), encoding="utf-8")
        vector_index_path.write_text(json.dumps(vector_index, ensure_ascii=False, indent=2), encoding="utf-8")

        if scope_config.scope == KnowledgeScope.platform:
            self.manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.lexical_index_path.write_text(json.dumps(lexical_index, ensure_ascii=False, indent=2), encoding="utf-8")
            self.vector_index_path.write_text(json.dumps(vector_index, ensure_ascii=False, indent=2), encoding="utf-8")

    def _latest_document_entries(
        self,
        session: Session,
        *,
        scope_config: ScopeConfig,
        limit: int = 12,
    ) -> list[KnowledgeDocumentEntry]:
        rows = session.exec(
            self._document_query(scope_config).order_by(KnowledgeDocumentRecord.updated_at.desc())
        ).all()
        return [self._document_entry_from_record(row) for row in rows[:limit]]

    def _build_report_from_run(
        self,
        session: Session,
        run: KnowledgeIngestionRunRecord,
        *,
        documents: list[KnowledgeDocumentEntry],
        changed_paths: list[str],
    ) -> KnowledgeIngestionReport:
        return KnowledgeIngestionReport(
            run_id=run.id,
            source_root=run.source_root,
            scope=run.scope,
            workspace_id=run.workspace_id,
            session_id=run.session_id,
            status=run.status,
            corpus_hash=run.corpus_hash,
            document_count=run.document_count,
            changed_document_count=len(changed_paths),
            unchanged_document_count=max(run.document_count - len(changed_paths), 0),
            section_count=run.section_count,
            lexical_term_count=run.lexical_term_count,
            vector_dimensions=run.vector_dimensions,
            filesystem_manifest_path=run.filesystem_manifest_path,
            lexical_index_path=run.lexical_index_path,
            vector_index_path=run.vector_index_path,
            changed_paths=changed_paths,
            documents=documents
            or self._latest_document_entries(
                session,
                scope_config=self._resolve_scope_config(run.scope, workspace_id=run.workspace_id, session_id=run.session_id),
            ),
            created_at=run.created_at,
        )

    def _document_entry_from_record(self, record: KnowledgeDocumentRecord) -> KnowledgeDocumentEntry:
        return KnowledgeDocumentEntry(
            id=record.id,
            scope=record.scope,
            workspace_id=record.workspace_id,
            session_id=record.session_id,
            relative_path=record.relative_path,
            title=record.title,
            format=record.format,
            visibility=record.visibility,
            status=record.status,
            authority_level=record.authority_level,
            memory_usage=record.memory_usage,
            stage_affinity=list(record.stage_affinity),
            agent_affinity=list(record.agent_affinity),
            version_number=record.version_number,
            section_count=record.section_count,
            content_hash=record.content_hash,
            source_lineage=record.source_lineage,
            approved_at=record.approved_at,
            effective_from=record.effective_from,
            expires_at=record.expires_at,
            updated_at=record.updated_at,
        )

    def _manifest_path_for_scope(self, scope_config: ScopeConfig) -> Path:
        return scope_config.runtime_dir / "knowledge-corpus-manifest.json"

    def _lexical_index_path_for_scope(self, scope_config: ScopeConfig) -> Path:
        return scope_config.runtime_dir / "lexical-index.json"

    def _vector_index_path_for_scope(self, scope_config: ScopeConfig) -> Path:
        return scope_config.runtime_dir / "vector-index.json"

    def _metadata_override_exists_for_document(self, parsed_document: ParsedDocument) -> bool:
        scope_config = self._resolve_scope_config(
            parsed_document.scope,
            workspace_id=parsed_document.workspace_id,
            session_id=parsed_document.session_id,
        )
        relative_path = self._document_relative_path(scope_config, parsed_document.relative_path)
        return self._metadata_override_path(scope_config, relative_path).exists()

    def _metadata_override_exists_for_record(self, record: KnowledgeDocumentRecord) -> bool:
        scope_config = self._resolve_scope_config(record.scope, workspace_id=record.workspace_id, session_id=record.session_id)
        relative_path = self._document_relative_path(scope_config, record.relative_path)
        return self._metadata_override_path(scope_config, relative_path).exists()

    def _metadata_override_path(self, scope_config: ScopeConfig, relative_path: Path) -> Path:
        return scope_config.overrides_root / Path(f"{relative_path.as_posix()}.meta.json")

    def _read_metadata_override(self, scope_config: ScopeConfig, relative_path: Path) -> dict[str, Any]:
        metadata_path = self._metadata_override_path(scope_config, relative_path)
        if not metadata_path.exists():
            return {}
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_metadata_override(self, scope_config: ScopeConfig, relative_path: Path, payload: dict[str, Any]) -> None:
        metadata_path = self._metadata_override_path(scope_config, relative_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _logical_path(self, scope_config: ScopeConfig, relative_path: Path) -> str:
        return f"{scope_config.source_root}/{relative_path.as_posix()}".replace("//", "/")

    def _document_relative_path(self, scope_config: ScopeConfig, logical_path: str) -> Path:
        prefix = f"{scope_config.source_root}/"
        relative = logical_path[len(prefix) :] if logical_path.startswith(prefix) else logical_path
        return Path(relative)

    def _normalize_relative_path(self, value: str) -> Path:
        candidate = Path(value.replace("\\", "/").strip("/"))
        if not candidate.as_posix() or ".." in candidate.parts:
            raise ValueError("relative_path invalido para knowledge.")
        return candidate

    def _append_search_audit(
        self,
        *,
        response: KnowledgeSearchResponse,
        workspace_id: UUID | None,
        session_id: UUID | None,
        stage: str | None,
    ) -> None:
        self.search_audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "recorded_at": utc_now().isoformat(),
            "query_hash": _stable_hash(response.query)[:24],
            "role": response.role,
            "workspace_id": str(workspace_id) if workspace_id is not None else None,
            "session_id": str(session_id) if session_id is not None else None,
            "stage": stage.strip().lower() if isinstance(stage, str) and stage.strip() else "",
            "corpus_hash": response.corpus_hash,
            "authorized_scopes": list(response.authorized_scopes),
            "applied_filters": list(response.applied_filters),
            "evidence_status": response.evidence_status,
            "absence_reason": response.absence_reason,
            "grounded_hits": response.grounded_hits,
            "total_hits": response.total_hits,
            "discarded_hits": response.discarded_hits,
            "returned_lineages": [item.source_lineage for item in response.items],
            "discarded_section_keys": [],
        }
        with self.search_audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
