from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any
from uuid import UUID

from sqlmodel import Session

from app.contracts.canonical_v1 import MemoryContextBudgetV1
from app.models import KnowledgeSearchHit, KnowledgeSearchResponse, SessionSnapshot
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionOutput
from app.services.canonical_exports import build_knowledge_manifest, build_memory_policy, build_short_term_memory
from app.services.knowledge_memory import KnowledgeMemoryService
from app.services.llm_runtime.stage_context_types import (
    ApprovedArtifactReference,
    RetrievedKnowledgeEvidence,
    StageContextBundle,
)


CAPABILITY_BUDGET_OVERRIDES: dict[str, tuple[int, int, int]] = {
    "normalize_discovery": (1200, 4, 5200),
    "build_canvas": (1400, 5, 6000),
    "define_requirements": (2100, 7, 8600),
    "propose_agent_design": (4500, 11, 19600),
    "critique_agent_design": (5200, 12, 22800),
    "synthesize_blueprint_narrative": (2600, 7, 11200),
    "recommend_minimal_tools": (2800, 9, 12400),
    "recommend_memory_architecture": (5200, 12, 22800),
    "critique_memory_architecture": (4800, 12, 20800),
    "generate_validation_scenarios": (3000, 10, 13200),
    "simulate_validation_scenario": (2200, 7, 9200),
    "judge_validation_run": (2400, 8, 10400),
    "analyze_estimation_risks": (3000, 9, 13200),
    "generate_diagram_model": (2600, 8, 11200),
}

CAPABILITY_STAGE_DEFAULTS = {
    "normalize_discovery": "discover",
    "build_canvas": "define",
    "define_requirements": "define",
    "propose_agent_design": "design",
    "critique_agent_design": "design",
    "synthesize_blueprint_narrative": "design",
    "recommend_minimal_tools": "tools",
    "recommend_memory_architecture": "memory",
    "critique_memory_architecture": "memory",
    "generate_validation_scenarios": "validate",
    "simulate_validation_scenario": "validate",
    "judge_validation_run": "validate",
    "analyze_estimation_risks": "estimate",
    "generate_diagram_model": "blueprint",
}

RETRIEVAL_ENABLED_STAGES = {"define", "design", "tools", "memory", "validate", "estimate", "package"}
_MAX_RETRIEVED_KNOWLEDGE_EXCERPT_CHARS = 1_600


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _compact_text(value: str, *, fallback: str = "unknown") -> str:
    normalized = " ".join(str(value or "").split()).strip()
    return normalized or fallback


def _first_text(value: dict[str, Any], *keys: str) -> str:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def _list_texts(value: Any, *, limit: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    terms: list[str] = []
    for item in value:
        if len(terms) >= limit:
            break
        if isinstance(item, str) and item.strip():
            terms.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = _first_text(item, "summary", "description", "detail", "title", "name", "reason", "business_fit")
        if text:
            terms.append(text)
    return terms


def _clip_text(value: str, *, limit: int = 900, fallback: str = "") -> str:
    normalized = _compact_text(value, fallback=fallback)
    if not normalized or normalized == fallback:
        return normalized
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _join_compact(items: list[str], *, limit: int = 4, item_limit: int = 180) -> str:
    compacted = [_clip_text(item, limit=item_limit, fallback="") for item in items if str(item or "").strip()]
    return "; ".join(compacted[:limit])


def _state_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _selected_design_payload(snapshot: SessionSnapshot) -> dict[str, Any] | None:
    artifact = (snapshot.journey_latest_artifacts or {}).get("design")
    if artifact is None or _state_value(getattr(artifact, "state", "")) not in {"approved", "approved_legacy"}:
        return None
    payload = artifact.proposal_payload if isinstance(artifact.proposal_payload, dict) else {}
    selected = payload.get("selected_design")
    if isinstance(selected, dict) and selected:
        return selected
    recommended_key = str(payload.get("recommended_alternative_key") or "").strip()
    alternatives = payload.get("alternatives")
    if not isinstance(alternatives, list):
        return None
    for item in alternatives:
        if not isinstance(item, dict):
            continue
        if recommended_key and str(item.get("alternative_key") or item.get("key") or "").strip() == recommended_key:
            return item
    return next((item for item in alternatives if isinstance(item, dict)), None)


def _design_memory_signal_summary(snapshot: SessionSnapshot) -> str:
    selected = _selected_design_payload(snapshot)
    if not selected:
        return ""
    projection = selected.get("blueprint_projection") if isinstance(selected.get("blueprint_projection"), dict) else {}
    memory_strategy = _first_text(projection, "memory_strategy")
    memory_implications = _list_texts(selected.get("memory_implications"), limit=4)
    memory_implications.extend(_list_texts(projection.get("memory_implications"), limit=4))
    tool_implications = _list_texts(selected.get("tool_implications"), limit=3)
    tool_implications.extend(_list_texts(projection.get("tool_implications"), limit=3))
    parts: list[str] = []
    if memory_strategy:
        parts.append(f"estrategia sugerida por Design: {memory_strategy}")
    if memory_implications:
        parts.append(f"implicaciones de memoria: {_join_compact(memory_implications, limit=5)}")
    if tool_implications:
        parts.append(f"implicaciones de tools: {_join_compact(tool_implications, limit=4)}")
    return _clip_text("; ".join(parts), limit=900, fallback="")


def _definition_reference_summary(definition: RequirementsDefinitionOutput) -> str:
    functional = _join_compact(
        [f"{item.key}: {item.title or item.requirement}" for item in definition.functional_requirements],
        limit=6,
    )
    non_functional = _join_compact(
        [f"{item.key}: {item.title or item.requirement}" for item in definition.non_functional_requirements],
        limit=5,
    )
    rules = _join_compact(
        [f"{item.key}: {item.title or item.rule}" for item in definition.business_rules],
        limit=4,
    )
    blocking_questions = _join_compact(
        [item.question for item in definition.open_questions if item.blocking],
        limit=3,
    )
    parts = [
        (
            f"Definition con {len(definition.functional_requirements)} FR, "
            f"{len(definition.non_functional_requirements)} NFR y "
            f"{len(definition.open_questions)} preguntas."
        )
    ]
    if definition.summary:
        parts.append(f"Resumen: {_clip_text(definition.summary, limit=260)}")
    if functional:
        parts.append(f"FR clave: {functional}")
    if non_functional:
        parts.append(f"NFR clave: {non_functional}")
    if rules:
        parts.append(f"Reglas: {rules}")
    if blocking_questions:
        parts.append(f"Preguntas bloqueantes aprobadas con deuda: {blocking_questions}")
    return _clip_text(" ".join(parts), limit=1_500)


def _definition_query_terms(snapshot: SessionSnapshot) -> list[str]:
    artifact = (snapshot.journey_latest_artifacts or {}).get("define")
    payload = artifact.proposal_payload if artifact is not None else {}
    if not isinstance(payload, dict):
        return []
    terms: list[str] = []
    terms.extend(_list_texts(payload.get("functional_requirements"), limit=5))
    terms.extend(_list_texts(payload.get("non_functional_requirements"), limit=4))
    terms.extend(_list_texts(payload.get("open_questions"), limit=3))
    return terms


def _design_query_terms(snapshot: SessionSnapshot) -> list[str]:
    artifact = (snapshot.journey_latest_artifacts or {}).get("design")
    payload = artifact.proposal_payload if artifact is not None else {}
    if not isinstance(payload, dict):
        return []
    terms: list[str] = []
    selected_key = str(payload.get("selected_alternative_key") or "")
    alternatives = payload.get("alternatives")
    selected_alternative: dict[str, Any] | None = None
    if isinstance(alternatives, list):
        for item in alternatives:
            if not isinstance(item, dict):
                continue
            if selected_key and str(item.get("key") or item.get("id") or "") == selected_key:
                selected_alternative = item
                break
        if selected_alternative is None:
            selected_alternative = next((item for item in alternatives if isinstance(item, dict)), None)
    if selected_alternative:
        terms.extend(
            [
                _first_text(selected_alternative, "agent_archetype", "architecture_pattern", "architecture"),
                _first_text(selected_alternative, "pattern_family", "reasoning_pattern"),
                _first_text(selected_alternative, "business_fit", "why_recommended", "value_hypothesis"),
                _first_text(selected_alternative, "operational_model"),
            ]
        )
        terms.extend(_list_texts(selected_alternative.get("tool_implications"), limit=5))
        terms.extend(_list_texts(selected_alternative.get("memory_implications"), limit=5))
        terms.extend(_list_texts(selected_alternative.get("risk_tradeoffs"), limit=4))
    projection = payload.get("blueprint_projection")
    if isinstance(projection, dict):
        terms.extend(_list_texts(projection.get("tool_implications"), limit=4))
        terms.extend(_list_texts(projection.get("memory_implications"), limit=4))
    return terms


def _tools_query_terms(snapshot: SessionSnapshot) -> list[str]:
    artifact = snapshot.latest_tool_recommendation
    if artifact is None:
        return []
    terms: list[str] = []
    for collection in (artifact.recommended_tools, artifact.optional_tools):
        for tool in collection[:5]:
            terms.append(
                " ".join(
                    str(part)
                    for part in (
                        getattr(tool, "name", ""),
                        getattr(tool, "description", ""),
                        getattr(tool, "purpose", ""),
                    )
                    if str(part).strip()
                )
            )
    for resolution in getattr(artifact, "capability_resolutions", [])[:6]:
        terms.append(
            " ".join(
                str(part)
                for part in (
                    getattr(resolution, "capability_key", ""),
                    getattr(resolution, "capability_covered", ""),
                    getattr(resolution, "reason", ""),
                    getattr(resolution, "promotion_policy", ""),
                )
                if str(part).strip()
            )
        )
    return terms


class ApprovedArtifactResolver:
    def resolve(
        self,
        snapshot: SessionSnapshot,
        *,
        stage: str,
        exclude_keys: set[str] | None = None,
    ) -> list[ApprovedArtifactReference]:
        excluded = {item.strip().lower() for item in exclude_keys or set() if str(item).strip()}
        refs: list[ApprovedArtifactReference] = []

        if snapshot.discovery is not None and "normalized_discovery" not in excluded:
            refs.append(
                ApprovedArtifactReference(
                    key="approved_discovery",
                    title="Approved discovery",
                    artifact_kind="discovery",
                    uri=f"session://{snapshot.session.id}/approved/discovery",
                    summary=_compact_text(
                        f"Problema: {snapshot.discovery.problem_statement}. "
                        f"Resultado esperado: {snapshot.discovery.desired_outcome}. "
                        f"Restricciones: {', '.join(snapshot.discovery.constraints[:3]) or 'ninguna'}."
                    ),
                    source_version="current",
                    source_refs=["session.discovery"],
                    stage_affinity=["define", "design", "tools", "memory", "validate", "estimate", "package"],
                    agent_affinity=["builder", "planner", "retrieval", "memory"],
                )
            )

        if snapshot.canvas is not None and "normalized_canvas" not in excluded:
            refs.append(
                ApprovedArtifactReference(
                    key="approved_canvas",
                    title="Approved canvas",
                    artifact_kind="canvas",
                    uri=f"session://{snapshot.session.id}/approved/canvas",
                    summary=_compact_text(
                        f"Objetivo usuario: {snapshot.canvas.user_goal}. "
                        f"Metrica: {snapshot.canvas.success_metric}. "
                        f"MVP: {', '.join(snapshot.canvas.mvp_scope[:3]) or 'sin alcance'}."
                    ),
                    source_version="current",
                    source_refs=["session.canvas"],
                    stage_affinity=["design", "tools", "memory", "validate", "estimate", "package"],
                    agent_affinity=["builder", "planner", "retrieval", "memory"],
                )
            )

        latest_define_artifact = (snapshot.journey_latest_artifacts or {}).get("define")
        if (
            latest_define_artifact is not None
            and latest_define_artifact.state in {"approved", "approved_legacy"}
            and "requirements_definition_input" not in excluded
        ):
            payload = latest_define_artifact.proposal_payload
            if latest_define_artifact.schema_version == "definition-artifact.v1" or "functional_requirements" in payload:
                definition = RequirementsDefinitionOutput.model_validate(payload)
                refs.append(
                    ApprovedArtifactReference(
                        key="approved_definition",
                        title="Approved definition",
                        artifact_kind="definition",
                        uri=f"session://{snapshot.session.id}/approved/definition",
                        summary=_definition_reference_summary(definition),
                        source_version=f"v{latest_define_artifact.version_number}",
                        source_refs=["session.journey_latest_artifacts.define"],
                        stage_affinity=["design", "tools", "memory", "validate", "estimate", "package"],
                        agent_affinity=["builder", "planner", "retrieval", "memory"],
                    )
                )

        if snapshot.blueprint is not None and "narrative_blueprint" not in excluded:
            design_memory_signal = _design_memory_signal_summary(snapshot)
            canonical_memory = _clip_text(snapshot.blueprint.memory_strategy, limit=260, fallback="")
            memory_line = (
                f"Memoria canonica: {canonical_memory}."
                if canonical_memory
                else "Memoria canonica: pendiente de etapa Memory."
            )
            design_signal_line = f" Señal Design->Memory: {design_memory_signal}." if design_memory_signal else ""
            refs.append(
                ApprovedArtifactReference(
                    key="approved_blueprint",
                    title="Approved blueprint",
                    artifact_kind="blueprint",
                    uri=f"session://{snapshot.session.id}/approved/blueprint",
                    summary=_clip_text(
                        f"Arquitectura: {snapshot.blueprint.architecture}. "
                        f"Razonamiento: {snapshot.blueprint.reasoning_pattern}. "
                        f"{memory_line}"
                        f"{design_signal_line} "
                        f"Tools actuales: {', '.join(item.name for item in snapshot.blueprint.tools[:4]) or 'sin tools aprobadas'}."
                    ),
                    source_version=str(snapshot.blueprint_versions[0].version_number) if snapshot.blueprint_versions else "current",
                    source_refs=["session.blueprint", "session.blueprint_versions", "session.journey_latest_artifacts.design"],
                    stage_affinity=["design", "tools", "memory", "validate", "estimate", "package"],
                    agent_affinity=["builder", "planner", "retrieval", "memory"],
                )
            )

        if stage in {"memory", "validate", "estimate", "package"} and snapshot.latest_tool_recommendation is not None:
            refs.append(
                ApprovedArtifactReference(
                    key="approved_tools_selection",
                    title="Approved tools selection",
                    artifact_kind="tool_selection",
                    uri=f"session://{snapshot.session.id}/approved/tools",
                    summary=_compact_text(
                        f"Tools recomendadas: {len(snapshot.latest_tool_recommendation.recommended_tools)} obligatorias, "
                        f"{len(snapshot.latest_tool_recommendation.optional_tools)} opcionales y "
                        f"{len(snapshot.latest_tool_recommendation.rejected_tools)} descartadas."
                    ),
                    source_version="current",
                    source_refs=["session.latest_tool_recommendation"],
                    stage_affinity=["memory", "validate", "estimate", "package"],
                    agent_affinity=["builder", "planner", "retrieval", "memory"],
                )
            )

        return refs


class StageKnowledgePlanner:
    def __init__(self, *, knowledge_service: KnowledgeMemoryService | None = None) -> None:
        self.knowledge_service = knowledge_service or KnowledgeMemoryService()

    def plan(
        self,
        session: Session,
        *,
        snapshot: SessionSnapshot,
        workspace_id: UUID | None,
        session_id: UUID | None,
        stage: str,
        role: str,
        approved_refs: list[ApprovedArtifactReference],
        allow_second_page: bool = False,
        page_size: int = 3,
    ) -> tuple[KnowledgeSearchResponse | None, list[RetrievedKnowledgeEvidence], int]:
        normalized_stage = stage.strip().lower()
        if normalized_stage not in RETRIEVAL_ENABLED_STAGES:
            return None, [], 0

        query = self._build_query(snapshot=snapshot, stage=normalized_stage, approved_refs=approved_refs)
        if not query:
            return None, [], 0

        response = self.knowledge_service.search_governed(
            session,
            query=query,
            role="planner",
            workspace_id=workspace_id,
            session_id=session_id,
            stage=normalized_stage,
            limit=max(1, page_size),
            ensure_ingested=True,
        )
        pages = 1 if response.total_hits or response.absence_reason else 0
        combined_items = list(response.items)

        if allow_second_page and response.next_cursor and self._needs_more_evidence(response):
            second_page = self.knowledge_service.search_governed(
                session,
                query=query,
                role="planner",
                workspace_id=workspace_id,
                session_id=session_id,
                stage=normalized_stage,
                limit=max(1, page_size),
                ensure_ingested=True,
                cursor=response.next_cursor,
            )
            pages += 1
            combined_items.extend(second_page.items)
            response = response.model_copy(
                update={
                    "grounded_hits": len(combined_items),
                    "items": combined_items,
                    "citations": list(dict.fromkeys([*response.citations, *second_page.citations])),
                    "next_cursor": second_page.next_cursor,
                    "discarded_hits": response.discarded_hits + second_page.discarded_hits,
                    "absence_reason": second_page.absence_reason if not combined_items else "",
                }
            )

        evidence = [self._to_retrieved_evidence(item, role=role) for item in combined_items]
        return response, evidence, pages

    def _needs_more_evidence(self, response: KnowledgeSearchResponse) -> bool:
        return response.evidence_status != "grounded" or response.grounded_hits < 2

    def _build_query(
        self,
        *,
        snapshot: SessionSnapshot,
        stage: str,
        approved_refs: list[ApprovedArtifactReference],
    ) -> str:
        parts = [
            f"etapa {stage}",
            snapshot.discovery.problem_statement if snapshot.discovery is not None else "",
            snapshot.discovery.desired_outcome if snapshot.discovery is not None else "",
            snapshot.canvas.user_goal if snapshot.canvas is not None else "",
            snapshot.canvas.success_metric if snapshot.canvas is not None else "",
            snapshot.blueprint.architecture if snapshot.blueprint is not None else "",
            snapshot.blueprint.reasoning_pattern if snapshot.blueprint is not None else "",
            snapshot.blueprint.memory_strategy if snapshot.blueprint is not None else "",
            " ".join(item.name for item in snapshot.blueprint.tools[:4]) if snapshot.blueprint is not None else "",
        ]
        parts.extend(_definition_query_terms(snapshot))
        parts.extend(_design_query_terms(snapshot))
        parts.extend(_tools_query_terms(snapshot))
        parts.extend(item.summary for item in approved_refs[:4])
        compact_parts = [_compact_text(item, fallback="") for item in parts]
        compact_parts = [item for item in compact_parts if item]
        return " ".join(compact_parts)[:1800]

    def _to_retrieved_evidence(self, item: KnowledgeSearchHit, *, role: str) -> RetrievedKnowledgeEvidence:
        source_version = f"v{item.version_number}" if item.version_number else "current"
        title = item.title or item.section_key
        return RetrievedKnowledgeEvidence(
            key=item.section_key,
            title=title,
            uri=f"knowledge://{item.relative_path}#{item.section_key}",
            relative_path=item.relative_path,
            section_key=item.section_key,
            authority_level=item.authority_level,
            memory_usage=item.memory_usage,
            summary=_compact_text(item.preview, fallback=title)[:280],
            excerpt=_compact_text(item.preview, fallback=title)[:_MAX_RETRIEVED_KNOWLEDGE_EXCERPT_CHARS],
            source_version=source_version,
            required=item.memory_usage == "required_retrieval",
            source_refs=[item.relative_path, item.section_key],
            source_lineage=[item.source_lineage] if item.source_lineage else [],
            stage_affinity=list(item.stage_affinity),
            agent_affinity=list(item.agent_affinity) or [role],
            score=item.score,
        )


class StageContextService:
    def __init__(
        self,
        *,
        approved_artifact_resolver: ApprovedArtifactResolver | None = None,
        knowledge_planner: StageKnowledgePlanner | None = None,
    ) -> None:
        self.approved_artifact_resolver = approved_artifact_resolver or ApprovedArtifactResolver()
        self.knowledge_planner = knowledge_planner or StageKnowledgePlanner()

    def build(
        self,
        session: Session,
        *,
        workspace_id: UUID | None,
        session_id: UUID | None,
        session_snapshot: SessionSnapshot,
        capability: str,
        effective_language: str = "es",
        role: str,
        stage: str | None = None,
        task_source_keys: list[str] | None = None,
        allow_second_page: bool = False,
    ) -> StageContextBundle:
        normalized_stage = (stage or CAPABILITY_STAGE_DEFAULTS.get(capability, "runtime")).strip().lower()
        memory_policy = build_memory_policy(session_snapshot)
        short_term_memory = build_short_term_memory(session_snapshot)
        strict_budget = self._resolve_budget(memory_policy=memory_policy, capability=capability, role=role)
        approved_refs = self.approved_artifact_resolver.resolve(
            session_snapshot,
            stage=normalized_stage,
            exclude_keys={item.strip().lower() for item in task_source_keys or [] if item.strip()},
        )

        retrieval_response = None
        retrieved_hits: list[RetrievedKnowledgeEvidence] = []
        retrieval_pages = 0
        knowledge_manifest = None
        if normalized_stage in RETRIEVAL_ENABLED_STAGES:
            knowledge_manifest = build_knowledge_manifest(session_snapshot)
            retrieval_response, retrieved_hits, retrieval_pages = self.knowledge_planner.plan(
                session,
                snapshot=session_snapshot,
                workspace_id=workspace_id,
                session_id=session_id,
                stage=normalized_stage,
                role=role,
                approved_refs=approved_refs,
                allow_second_page=allow_second_page,
                page_size=max(1, min(strict_budget.max_items, 4)),
            )

        context_fingerprint = self._build_context_fingerprint(
            capability=capability,
            role=role,
            stage=normalized_stage,
            strict_budget=strict_budget,
            approved_refs=approved_refs,
            retrieved_hits=retrieved_hits,
            session_snapshot=session_snapshot,
            corpus_hash=retrieval_response.corpus_hash if retrieval_response is not None else "",
        )
        return StageContextBundle(
            capability=capability,
            effective_language=effective_language,
            role=role,
            stage=normalized_stage,
            workspace_id=workspace_id,
            session_id=session_id,
            session_snapshot=session_snapshot,
            knowledge_manifest=knowledge_manifest,
            memory_policy=memory_policy,
            short_term_memory=short_term_memory,
            approved_refs=approved_refs,
            retrieved_hits=retrieved_hits,
            retrieval_response=retrieval_response,
            strict_budget=strict_budget,
            context_fingerprint=context_fingerprint,
            corpus_hash=retrieval_response.corpus_hash if retrieval_response is not None else "",
            retrieval_pages=retrieval_pages,
            absence_reason=retrieval_response.absence_reason if retrieval_response is not None else "",
        )

    def _resolve_budget(
        self,
        *,
        memory_policy,
        capability: str,
        role: str,
    ) -> MemoryContextBudgetV1:
        base_budget = next(
            (item for item in memory_policy.context_budgets if item.role.strip().lower() == role.strip().lower()),
            None,
        )
        override = CAPABILITY_BUDGET_OVERRIDES.get(capability)
        if override is None:
            return base_budget or MemoryContextBudgetV1(
                role=role,
                max_tokens=1400,
                max_items=5,
                max_chars=5600,
                compaction_trigger=f"{capability}_default",
                overflow_policy="truncate_to_summary",
            )
        max_tokens, max_items, max_chars = override
        if base_budget is not None:
            return base_budget.model_copy(
                update={
                    "max_tokens": min(base_budget.max_tokens or max_tokens, max_tokens),
                    "max_items": min(base_budget.max_items or max_items, max_items),
                    "max_chars": min(base_budget.max_chars or max_chars, max_chars),
                    "compaction_trigger": f"capability_budget:{capability}",
                    "overflow_policy": "compact_by_priority_then_trim_candidates",
                }
            )
        return MemoryContextBudgetV1(
            role=role,
            max_tokens=max_tokens,
            max_items=max_items,
            max_chars=max_chars,
            compaction_trigger=f"capability_budget:{capability}",
            overflow_policy="compact_by_priority_then_trim_candidates",
        )

    def _build_context_fingerprint(
        self,
        *,
        capability: str,
        role: str,
        stage: str,
        strict_budget: MemoryContextBudgetV1,
        approved_refs: list[ApprovedArtifactReference],
        retrieved_hits: list[RetrievedKnowledgeEvidence],
        session_snapshot: SessionSnapshot,
        corpus_hash: str,
    ) -> str:
        return _stable_hash(
            {
                "capability": capability,
                "role": role,
                "stage": stage,
                "session_id": str(session_snapshot.session.id),
                "workspace_id": str(session_snapshot.session.workspace_id),
                "blueprint_versions": [item.version_number for item in session_snapshot.blueprint_versions[:3]],
                "budget": strict_budget.model_dump(mode="json"),
                "approved_refs": [asdict(item) for item in approved_refs],
                "retrieved_hits": [asdict(item) for item in retrieved_hits],
                "corpus_hash": corpus_hash,
            }
        )
