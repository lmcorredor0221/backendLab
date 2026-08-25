from __future__ import annotations

from hashlib import sha256
import json
from typing import Any
from uuid import UUID, uuid4

from sqlmodel import Session, func, select
from sqlalchemy.engine import Engine

from app.db import engine
from app.models import ArtifactRegistryRecord, JourneyArtifactState, JourneyStageArtifactRecord, SessionRecord, utc_now
from app.services.diagram_center.contracts import DiagramGenerationInput, DiagramGenerationJobResponse, DiagramModel, DiagramNotation
from app.services.diagram_center.persistence import (
    DiagramGenerationJobRecord,
    DiagramGovernanceRecord,
    DiagramVersionRecord,
)
from app.services.diagram_center.quality_service import evaluate_diagram_quality
from app.services.diagram_center.registry_service import build_prompt_spec, get_registry_entry
from app.services.diagram_center.renderer_service import RENDERER_REVISION, render_diagram
from app.services.llm_runtime.capability_registry import BuilderCapability
from app.services.llm_runtime.runtime_settings_service import load_effective_runtime_settings
from app.services.llm_runtime.stage_context_types import StageContextBundle
from app.services.openai_builder import build_builder_service


MAX_CONTEXT_ITEMS = 18
MAX_CONTEXT_CHARS_PER_ITEM = 6000
RESOLVED_INPUT_EVIDENCE_LIMIT = 1400

_REQUIRED_INPUT_MATCHERS: dict[str, dict[str, set[str]]] = {
    "session.discovery": {"artifact_keys": {"discovery_artifact", "discovery_analysis_artifact"}, "stages": {"discover"}},
    "discovery.analysis": {"artifact_keys": {"discovery_analysis_artifact"}, "stages": {"discover"}},
    "discovery.problem_context_brief": {"artifact_keys": {"discovery_analysis_artifact", "discovery_artifact"}, "stages": {"discover"}},
    "session.canvas": {"artifact_keys": {"canvas_artifact"}, "stages": {"define"}},
    "definition.requirements": {"artifact_keys": {"definition_artifact"}, "stages": {"define"}},
    "design.architecture": {"artifact_keys": {"design_recommendation_artifact"}, "stages": {"design"}},
    "blueprint.architecture_spec": {"artifact_keys": {"design_recommendation_artifact"}, "stages": {"design"}},
    "blueprint.patterns": {"artifact_keys": {"design_recommendation_artifact"}, "stages": {"design"}},
    "tools.minimum_set": {"artifact_keys": {"tool_recommendation_artifact"}, "stages": {"tools"}},
    "memory.strategy": {"artifact_keys": {"memory_recommendation_artifact"}, "stages": {"memory"}},
    "estimate.analysis": {"artifact_keys": {"estimation_report_artifact"}, "stages": {"estimate"}},
    "validation.scenarios": {"artifact_keys": {"evaluation_artifact"}, "stages": {"validate"}},
}

_REQUIRED_INPUT_FIELD_HINTS: dict[str, list[str]] = {
    "session.discovery": [
        "summary",
        "facts",
        "problem_statement",
        "current_user",
        "current_process",
        "desired_outcome",
        "constraints",
        "inferred_needs",
        "open_questions",
    ],
    "discovery.analysis": ["summary", "facts", "inferred_needs", "domain_signals", "risk_signals", "open_questions"],
    "discovery.problem_context_brief": ["summary", "facts", "problem_statement", "current_process", "desired_outcome"],
    "session.canvas": ["user_goal", "mvp_scope", "success_metric", "agent_profile", "primary_risk", "summary"],
    "definition.requirements": [
        "summary",
        "functional_requirements",
        "non_functional_requirements",
        "business_rules",
        "acceptance_criteria",
        "dependencies",
        "open_questions",
    ],
    "design.architecture": [
        "summary",
        "selected_design",
        "decision_rationale",
        "requirements_coverage",
        "recommended_alternative_key",
        "open_questions",
    ],
    "blueprint.architecture_spec": [
        "summary",
        "selected_design",
        "decision_rationale",
        "requirements_coverage",
        "recommended_alternative_key",
        "open_questions",
    ],
    "blueprint.patterns": [
        "summary",
        "selected_design",
        "alternatives",
        "fit_matrix",
        "decision_rationale",
        "recommended_alternative_key",
    ],
    "tools.minimum_set": [
        "summary",
        "recommended_tools",
        "optional_tools",
        "approved_tools_digest",
        "design_role_coverage",
        "coverage_gaps",
        "needs_information",
    ],
    "memory.strategy": [
        "summary",
        "proposed_memory_profile",
        "knowledge_design",
        "working_memory_design",
        "short_term_design",
        "long_term_design",
        "context_budget_plan",
    ],
    "estimate.analysis": ["summary", "notes", "agentic", "recommendations", "risks", "assumptions"],
    "validation.scenarios": ["summary", "findings", "scenarios", "test_cases", "gaps"],
}


def _compact(value: object, *, limit: int = MAX_CONTEXT_CHARS_PER_ITEM) -> object:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return [_compact(item, limit=max(500, limit // max(len(value), 1))) for item in value[:30]]
    if isinstance(value, dict):
        return {str(key): _compact(item, limit=max(500, limit // max(len(value), 1))) for key, item in list(value.items())[:40]}
    return value


def _compact_text(value: object, *, limit: int = 240) -> str:
    normalized = " ".join(str(value or "").split()).strip()
    return normalized[:limit]


def _summarize_list(values: list[object], *, limit: int = 4, item_limit: int = 100) -> list[str]:
    return [_compact_text(item, limit=item_limit) for item in values[:limit] if _compact_text(item, limit=item_limit)]


def _design_architecture_brief(content: dict[str, Any]) -> tuple[object, str]:
    selected_design = content.get("selected_design") if isinstance(content.get("selected_design"), dict) else {}
    blueprint_projection = (
        selected_design.get("blueprint_projection")
        if isinstance(selected_design.get("blueprint_projection"), dict)
        else {}
    )
    roles = selected_design.get("roles") if isinstance(selected_design.get("roles"), list) else []
    handoffs = selected_design.get("handoffs") if isinstance(selected_design.get("handoffs"), list) else []
    concise_payload = {
        "summary": _compact_text(content.get("summary")),
        "architecture_pattern": _compact_text(
            selected_design.get("architecture")
            or selected_design.get("architecture_pattern")
            or selected_design.get("alternative_key")
        ),
        "coordination_model": _compact_text(selected_design.get("coordination_model")),
        "reasoning_pattern": _compact_text(selected_design.get("reasoning_pattern")),
        "topology": _compact_text(selected_design.get("topology"), limit=320),
        "roles": [
            {
                "key": _compact_text(item.get("key")),
                "title": _compact_text(item.get("title")),
                "responsibility": _compact_text(item.get("responsibility"), limit=180),
            }
            for item in roles[:4]
            if isinstance(item, dict)
        ],
        "handoffs": [
            {
                "from_role": _compact_text(item.get("from_role")),
                "to_role": _compact_text(item.get("to_role")),
                "trigger": _compact_text(item.get("trigger"), limit=160),
            }
            for item in handoffs[:4]
            if isinstance(item, dict)
        ],
        "approval_points": _summarize_list(
            selected_design.get("approval_points") if isinstance(selected_design.get("approval_points"), list) else [],
            limit=4,
            item_limit=120,
        ),
        "guardrails": _summarize_list(
            blueprint_projection.get("guardrails") if isinstance(blueprint_projection.get("guardrails"), list) else [],
            limit=4,
            item_limit=120,
        ),
    }
    role_titles = [item.get("title") or item.get("key") for item in concise_payload["roles"] if isinstance(item, dict)]
    handoff_pairs = [
        f"{item.get('from_role')}->{item.get('to_role')}"
        for item in concise_payload["handoffs"]
        if isinstance(item, dict) and item.get("from_role") and item.get("to_role")
    ]
    brief = (
        "Approved architecture evidence: "
        f"pattern={concise_payload['architecture_pattern'] or 'unknown'}; "
        f"reasoning={concise_payload['reasoning_pattern'] or 'unknown'}; "
        f"roles={', '.join(str(item) for item in role_titles[:4]) or 'none'}; "
        f"handoffs={', '.join(handoff_pairs[:4]) or 'none'}; "
        f"topology={concise_payload['topology'] or 'unknown'}."
    )
    return concise_payload, brief


def _discovery_brief(content: dict[str, Any]) -> tuple[object, str]:
    current_process = _compact_text(content.get("current_process"), limit=420)
    desired_outcome = _compact_text(content.get("desired_outcome"), limit=260)
    concise_payload = {
        "problem_statement": _compact_text(content.get("problem_statement"), limit=320),
        "current_user": _compact_text(content.get("current_user"), limit=220),
        "current_process": current_process,
        "desired_outcome": desired_outcome,
        "constraints": _summarize_list(
            content.get("constraints") if isinstance(content.get("constraints"), list) else [],
            limit=5,
            item_limit=140,
        ),
        "mvp_scope": _summarize_list(
            (content.get("mvp_definition") or {}).get("v1_scope") if isinstance(content.get("mvp_definition"), dict) else [],
            limit=5,
            item_limit=140,
        ),
    }
    brief = (
        "Approved discovery evidence: "
        f"problem={concise_payload['problem_statement'] or 'unknown'}; "
        f"current_process={current_process or 'unknown'}; "
        f"desired_outcome={desired_outcome or 'unknown'}."
    )
    return concise_payload, brief


def _discovery_analysis_brief(content: dict[str, Any]) -> tuple[object, str]:
    facts = content.get("facts") if isinstance(content.get("facts"), list) else []
    concise_payload = {
        "summary": _compact_text(content.get("summary"), limit=280),
        "facts": [
            {
                "key": _compact_text(item.get("key")),
                "statement": _compact_text(item.get("statement"), limit=220),
                "source_refs": item.get("source_refs", [])[:3] if isinstance(item, dict) else [],
            }
            for item in facts[:5]
            if isinstance(item, dict)
        ],
        "inferred_needs": _summarize_list(content.get("inferred_needs") if isinstance(content.get("inferred_needs"), list) else [], limit=4, item_limit=140),
        "open_questions": _summarize_list(content.get("open_questions") if isinstance(content.get("open_questions"), list) else [], limit=4, item_limit=140),
    }
    fact_digest = [item.get("statement") for item in concise_payload["facts"][:3] if isinstance(item, dict) and item.get("statement")]
    brief = (
        "Approved discovery analysis: "
        f"summary={concise_payload['summary'] or 'unknown'}; "
        f"facts={'; '.join(str(item) for item in fact_digest)[:420] or 'none'}."
    )
    return concise_payload, brief


def _resolved_input_evidence(required_input: str, artifact: dict[str, Any]) -> tuple[object, str]:
    content = artifact.get("content")
    if not isinstance(content, dict):
        compact = _compact(content, limit=RESOLVED_INPUT_EVIDENCE_LIMIT)
        return compact, _compact_text(compact, limit=360)
    if required_input in {"design.architecture", "blueprint.architecture_spec", "blueprint.patterns"}:
        return _design_architecture_brief(content)
    if required_input == "discovery.analysis":
        return _discovery_analysis_brief(content)
    if required_input in {"session.discovery", "discovery.problem_context_brief"}:
        if "facts" in content:
            return _discovery_analysis_brief(content)
        return _discovery_brief(content)
    extracted = _resolved_input_excerpt(required_input, artifact)
    return extracted, _compact_text(extracted, limit=360)


def _normalized_lookup_tokens(*values: object) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        normalized = str(value or "").strip().lower()
        if normalized:
            tokens.add(normalized)
    return tokens


def _matches_required_input(required_input: str, artifact: dict[str, Any]) -> bool:
    matcher = _REQUIRED_INPUT_MATCHERS.get(required_input, {})
    expected_keys = {item.strip().lower() for item in matcher.get("artifact_keys", set()) if str(item).strip()}
    expected_stages = {item.strip().lower() for item in matcher.get("stages", set()) if str(item).strip()}
    artifact_key = str(artifact.get("key") or artifact.get("kind") or "").strip().lower()
    artifact_stage = str(artifact.get("stage") or "").strip().lower()
    artifact_tokens = _normalized_lookup_tokens(
        artifact.get("key"),
        artifact.get("kind"),
        artifact.get("stage"),
    )
    if expected_keys & artifact_tokens:
        return True
    return bool(artifact_stage in expected_stages and artifact_key in expected_stages)


def _resolved_input_excerpt(required_input: str, artifact: dict[str, Any]) -> object:
    content = artifact.get("content")
    if isinstance(content, dict):
        hints = _REQUIRED_INPUT_FIELD_HINTS.get(required_input, [])
        extracted: dict[str, object] = {}
        for field_name in hints:
            if field_name not in content:
                continue
            value = content.get(field_name)
            if value in (None, "", [], {}):
                continue
            extracted[field_name] = _compact(value, limit=RESOLVED_INPUT_EVIDENCE_LIMIT)
            if len(json.dumps(extracted, ensure_ascii=True, default=str)) >= RESOLVED_INPUT_EVIDENCE_LIMIT:
                break
        if extracted:
            return extracted
    return _compact(content, limit=RESOLVED_INPUT_EVIDENCE_LIMIT)


def _build_resolved_inputs(
    required_inputs: list[str],
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, object]], list[str]]:
    resolved_inputs: list[dict[str, object]] = []
    missing_required_inputs: list[str] = []
    for required_input in required_inputs:
        matches = [artifact for artifact in sources if _matches_required_input(required_input, artifact)]
        if not matches:
            missing_required_inputs.append(required_input)
            continue
        evidence_items: list[dict[str, object]] = []
        brief_parts: list[str] = []
        for item in matches[:2]:
            evidence_payload, evidence_brief = _resolved_input_evidence(required_input, item)
            if evidence_brief:
                brief_parts.append(evidence_brief)
            evidence_items.append(
                {
                    "artifact_key": str(item.get("key") or ""),
                    "ref": str(item.get("ref") or ""),
                    "content": evidence_payload,
                }
            )
        resolved_inputs.append(
            {
                "input_key": required_input,
                "status": "resolved",
                "matched_artifact_keys": [str(item.get("key") or "") for item in matches[:3]],
                "artifact_refs": [str(item.get("ref") or "") for item in matches[:3]],
                "stage_keys": [str(item.get("stage") or "") for item in matches[:3] if str(item.get("stage") or "").strip()],
                "brief": " ".join(part for part in brief_parts if part).strip(),
                "evidence": evidence_items,
            }
        )
    return resolved_inputs, missing_required_inputs


def _build_context_brief(
    resolved_inputs: list[dict[str, object]],
    missing_required_inputs: list[str],
) -> str:
    parts = [str(item.get("brief") or "").strip() for item in resolved_inputs if str(item.get("brief") or "").strip()]
    if missing_required_inputs:
        parts.append("Missing required inputs: " + ", ".join(missing_required_inputs[:6]))
    return " ".join(parts)[:1600].strip()


def _source_context(
    db: Session,
    record: SessionRecord,
    *,
    required_inputs: list[str] | None = None,
) -> tuple[dict[str, object], list[str]]:
    journey_records = db.exec(
        select(JourneyStageArtifactRecord)
        .where(
            JourneyStageArtifactRecord.session_id == record.id,
            JourneyStageArtifactRecord.state.in_(
                [JourneyArtifactState.approved, JourneyArtifactState.approved_legacy]
            ),
        )
        .order_by(JourneyStageArtifactRecord.created_at.desc())
        .limit(MAX_CONTEXT_ITEMS)
    ).all()
    registry_records = db.exec(
        select(ArtifactRegistryRecord)
        .where(ArtifactRegistryRecord.session_id == record.id)
        .order_by(ArtifactRegistryRecord.created_at.desc())
        .limit(MAX_CONTEXT_ITEMS)
    ).all()

    sources: list[dict[str, object]] = []
    source_refs: list[str] = []
    seen: set[str] = set()
    for artifact in journey_records:
        key = artifact.artifact_kind or artifact.stage_key
        ref = f"journey:{artifact.id}:v{artifact.version_number}"
        dedupe_key = f"journey:{key}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        source_refs.append(ref)
        sources.append(
            {
                "key": key,
                "stage": artifact.stage_key,
                "state": artifact.state.value,
                "version": artifact.version_number,
                "ref": ref,
                "content": _compact(artifact.proposal_payload),
            }
        )
    for artifact in registry_records:
        dedupe_key = f"registry:{artifact.artifact_key}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        ref = f"artifact:{artifact.id}"
        source_refs.append(ref)
        sources.append(
            {
                "key": artifact.artifact_key,
                "kind": artifact.artifact_kind,
                "ref": ref,
                "content": _compact(artifact.content_text),
                "metadata": _compact(artifact.artifact_metadata),
            }
        )
    if not sources:
        ref = f"session:{record.id}"
        source_refs.append(ref)
        sources.append(
            {
                "key": "session.baseline",
                "stage": getattr(record.current_stage, "value", str(record.current_stage or "discover")),
                "ref": ref,
                "content": f"Project baseline for {record.title}",
            }
        )
    resolved_inputs, missing_required_inputs = _build_resolved_inputs(required_inputs or [], sources)
    context_brief = _build_context_brief(resolved_inputs, missing_required_inputs)
    return (
        {
            "project": {
                "id": str(record.id),
                "title": record.title,
                "current_stage": getattr(record.current_stage, "value", str(record.current_stage or "discover")),
                "commercial_tier": getattr(record.commercial_tier, "value", str(record.commercial_tier or "blueprint")),
            },
            "context_brief": context_brief,
            "coverage_summary": {
                "required_input_count": len(required_inputs or []),
                "resolved_input_count": len(resolved_inputs),
                "missing_input_count": len(missing_required_inputs),
            },
            "resolved_inputs": resolved_inputs,
            "missing_required_inputs": missing_required_inputs,
            "approved_artifact_keys": [str(item.get("key") or "") for item in sources[:MAX_CONTEXT_ITEMS]],
            "approved_artifacts": sources[:MAX_CONTEXT_ITEMS],
        },
        source_refs[:MAX_CONTEXT_ITEMS],
    )


def _fingerprint(payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _build_generation_context_bundle(
    *,
    record: SessionRecord,
    job: DiagramGenerationJobRecord,
    stage: str,
) -> StageContextBundle:
    return StageContextBundle(
        capability=BuilderCapability.generate_diagram_model.value,
        role="builder",
        stage=stage.strip().lower() or "design",
        workspace_id=record.workspace_id,
        session_id=record.id,
        session_snapshot=None,
        effective_language="es",
        knowledge_manifest=None,
        memory_policy=None,
        short_term_memory=None,
        context_fingerprint=f"diagram-job:{job.id}",
        finops_metadata={
            "diagram_job_id": str(job.id),
            "diagram_key": job.diagram_key,
            "generation_reason": job.reason,
            "detail_level": job.detail_level,
        },
    )


def job_response(record: DiagramGenerationJobRecord) -> DiagramGenerationJobResponse:
    return DiagramGenerationJobResponse(
        id=record.id,
        project_id=record.session_id,
        diagram_key=record.diagram_key,
        status=record.status,
        provider_key=record.provider_key,
        model_name=record.model_name,
        version_id=record.version_id,
        error_code=record.error_code,
        error_message=record.error_message,
        requested_at=record.requested_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
    )


def create_generation_job(
    db: Session,
    *,
    record: SessionRecord,
    diagram_key: str,
    user_id: UUID,
    detail_level: str,
    reason: str,
    idempotency_key: str,
) -> DiagramGenerationJobRecord:
    entry = get_registry_entry(diagram_key)
    if entry is None or not entry.active:
        raise LookupError("Diagram type not found")
    resolved_idempotency_key = idempotency_key.strip() or f"diagram:{record.id}:{diagram_key}:{uuid4()}"
    existing = db.exec(
        select(DiagramGenerationJobRecord).where(
            DiagramGenerationJobRecord.workspace_id == record.workspace_id,
            DiagramGenerationJobRecord.idempotency_key == resolved_idempotency_key,
        )
    ).first()
    if existing is not None:
        if existing.status != "error":
            return existing
        resolved_idempotency_key = f"{resolved_idempotency_key}:retry:{uuid4()}"
    has_existing_version = db.exec(
        select(DiagramVersionRecord.id).where(
            DiagramVersionRecord.session_id == record.id,
            DiagramVersionRecord.diagram_key == entry.key,
        )
    ).first()
    job = DiagramGenerationJobRecord(
        workspace_id=record.workspace_id,
        session_id=record.id,
        diagram_key=entry.key,
        requested_by_user_id=user_id,
        detail_level=detail_level,
        reason=reason,
        idempotency_key=resolved_idempotency_key,
        status="updating" if reason in {"regenerate", "layout_upgrade"} and has_existing_version is not None else "queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _fail_job(db: Session, job: DiagramGenerationJobRecord, code: str, message: str) -> None:
    job.status = "error"
    job.error_code = code
    job.error_message = message[:700]
    job.completed_at = utc_now()
    job.updated_at = utc_now()
    db.add(job)
    db.commit()


def run_generation_job(job_id: UUID, database_engine: Engine | None = None) -> None:
    with Session(database_engine or engine) as db:
        job = db.get(DiagramGenerationJobRecord, job_id)
        if job is None or job.status not in {"queued", "updating"}:
            return
        record = db.get(SessionRecord, job.session_id)
        entry = get_registry_entry(job.diagram_key)
        if record is None or entry is None:
            _fail_job(db, job, "source_not_found", "El proyecto o el tipo de diagrama ya no está disponible.")
            return
        governance = db.exec(
            select(DiagramGovernanceRecord).where(DiagramGovernanceRecord.diagram_key == entry.key)
        ).first()
        if governance is not None and (not governance.enabled or not governance.generation_enabled):
            _fail_job(db, job, "generation_disabled", "La generación fue deshabilitada por el administrador.")
            return

        job.status = "generating"
        job.started_at = utc_now()
        job.updated_at = utc_now()
        db.add(job)
        db.commit()

        prompt_spec = build_prompt_spec(entry, override=governance.prompt_override if governance else None)
        source_context, source_refs = _source_context(
            db,
            record,
            required_inputs=list(prompt_spec["required_inputs"]),
        )
        approved_items = [
            s for s in source_context.get("approved_artifacts", []) if s.get("key") != "session.baseline"
        ]
        if not approved_items:
            _fail_job(
                db,
                job,
                "approved_context_missing",
                "No existen artefactos aprobados suficientes para generar un diagrama trazable.",
            )
            return

        effective_notation = DiagramNotation(str(prompt_spec.get("notation") or entry.notation.value))

        generation_input = DiagramGenerationInput(
            diagram_key=entry.key,
            title=entry.title,
            objective=str(prompt_spec["objective"]),
            notation=effective_notation,
            standard=str(prompt_spec.get("standard") or entry.standard),
            detail_level=job.detail_level,
            required_inputs=list(prompt_spec["required_inputs"]),
            context_brief=str(source_context.get("context_brief") or ""),
            resolved_inputs=list(source_context.get("resolved_inputs", [])),
            missing_required_inputs=list(source_context.get("missing_required_inputs", [])),
            source_context=source_context,
            source_refs=source_refs,
            source_contract=str(prompt_spec.get("source_contract") or entry.source_contract),
            presentation_contract=str(prompt_spec.get("presentation_contract") or entry.presentation_contract),
            renderer_key=str(prompt_spec.get("renderer_key") or entry.renderer_key),
            validator_key=str(prompt_spec.get("validator_key") or entry.validator_key),
            allowed_elements=list(prompt_spec.get("allowed_elements") or entry.allowed_elements),
            allowed_relationships=list(prompt_spec.get("allowed_relationships") or entry.allowed_relationships),
            forbidden_mixes=list(prompt_spec.get("forbidden_mixes") or entry.forbidden_mixes),
            inherits_from=list(prompt_spec.get("inherits_from") or entry.inherits_from),
            transform_rules=list(prompt_spec.get("transform_rules") or entry.transform_rules),
            generation_permissions=dict(prompt_spec.get("generation_permissions") or entry.generation_permissions),
            semantic_rules=list(prompt_spec["semantic_rules"]),
            exclusions=list(prompt_spec["exclusions"]),
            prompt_spec_version=str(prompt_spec["version"]),
        )
        runtime_settings = load_effective_runtime_settings(db, record.workspace_id)
        provider = build_builder_service(runtime_settings)
        stage_context = _build_generation_context_bundle(record=record, job=job, stage=entry.stage)
        result = provider.generate_diagram_model(generation_input, context_bundle=stage_context)
        if result.artifact is None and result.failure_kind in {
            "provider_error",
            "schema_invalid",
            "schema_missing_output",
        }:
            result = provider.generate_diagram_model(generation_input, context_bundle=stage_context)
            job.request_metadata = {**job.request_metadata, "retry_count": 1}
        job.provider_key = result.provider_key or runtime_settings.active_provider.value
        job.model_name = result.model_name or ""
        job.prompt_spec_version = result.prompt_version or str(prompt_spec["version"])
        if result.artifact is None:
            _fail_job(
                db,
                job,
                result.failure_kind or "provider_failure",
                result.warning or result.failure_detail or "El proveedor no produjo un DiagramModel válido.",
            )
            return

        try:
            raw_model = result.artifact.model_dump(mode="json")
            existing_metadata = raw_model.get("metadata", {})
            if not isinstance(existing_metadata, dict):
                existing_metadata = {}
            previous_version_number = db.exec(
                select(func.max(DiagramVersionRecord.version_number)).where(
                    DiagramVersionRecord.session_id == record.id,
                    DiagramVersionRecord.diagram_key == entry.key,
                )
            ).one()
            metadata = {
                **existing_metadata,
                "standard": generation_input.standard,
                "source_contract": generation_input.source_contract,
                "presentation_contract": generation_input.presentation_contract,
                "renderer_key": generation_input.renderer_key,
                "validator_key": generation_input.validator_key,
                "renderer_revision": RENDERER_REVISION,
                "generation_reason": job.reason,
                "allowed_elements": list(generation_input.allowed_elements),
                "allowed_relationships": list(generation_input.allowed_relationships),
                "forbidden_mixes": list(generation_input.forbidden_mixes),
                "inherits_from": list(generation_input.inherits_from),
                "transform_rules": list(generation_input.transform_rules),
                "generation_permissions": dict(generation_input.generation_permissions),
                "prompt_spec_version": str(prompt_spec["version"]),
                "source_fingerprint": _fingerprint(source_context),
            }
            if job.reason == "layout_upgrade":
                metadata["layout_upgrade_reason"] = "layout_upgrade"
                metadata["previous_version_number"] = int(previous_version_number or 0)
            raw_model.update(
                {
                    "diagram_key": entry.key,
                    "title": entry.title,
                    "notation": effective_notation.value,
                    "source_refs": list(dict.fromkeys([*raw_model.get("source_refs", []), *source_refs])),
                    "metadata": metadata,
                }
            )
            model = DiagramModel.model_validate(raw_model)
        except Exception as exc:
            _fail_job(db, job, "diagram_model_invalid", f"El modelo canónico no superó validación: {exc}")
            return

        quality = evaluate_diagram_quality(model)
        if not quality.valid:
            _fail_job(db, job, "quality_gate_failed", " ".join(quality.errors))
            return
        renderings = render_diagram(model)
        current_max = db.exec(
            select(func.max(DiagramVersionRecord.version_number)).where(
                DiagramVersionRecord.session_id == record.id,
                DiagramVersionRecord.diagram_key == entry.key,
            )
        ).one()
        version_number = int(current_max or 0) + 1
        version = DiagramVersionRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            diagram_key=entry.key,
            job_id=job.id,
            version_number=version_number,
            diagram_model=model.model_dump(mode="json"),
            renderings=renderings,
            quality_report=quality.model_dump(mode="json"),
            source_fingerprint=_fingerprint(source_context),
            source_refs=source_refs,
            provider_key=result.provider_key or runtime_settings.active_provider.value,
            model_name=result.model_name or "",
            prompt_spec_version=result.prompt_version or str(prompt_spec["version"]),
            request_id=result.request_id or "",
            created_by_user_id=job.requested_by_user_id,
        )
        db.add(version)
        db.flush()
        job.status = "available"
        job.version_id = version.id
        job.completed_at = utc_now()
        job.updated_at = utc_now()
        db.add(job)
        db.commit()
        if record is not None:
            try:
                from app.services.product_processing.product_build_orchestrator import reconcile_product_build_run
                from app.services.product_processing.contracts import ProductBuildProductKey

                reconcile_product_build_run(db, record=record, product_key=ProductBuildProductKey.blueprint_basic)
                reconcile_product_build_run(db, record=record, product_key=ProductBuildProductKey.blueprint_pro)
                db.commit()
            except Exception:
                pass
