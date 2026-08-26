from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    AccessRequestResolveRequest,
    ApprovalGateRecord,
    AttentionItemResponse,
    AttentionActionRequestV2,
    AttentionItemV2,
    AttentionResponse,
    AttentionResponseV2,
    CommercialEventRecord,
    CommercialAccessRequestRecord,
    CommercialAccessRequestStatus,
    CommercialAccessSnapshotV2,
    CommercialTier,
    ConstructionQuestionResponseRecord,
    ConstructionReadinessReport,
    ApprovalStatus,
    BlueprintRecord,
    GovernancePolicyRecord,
    HandoffRecord,
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    ReviewState,
    SessionRecord,
    SessionSnapshot,
    StageOperationRecord,
    StageOperationStatus,
    UserRecord,
    utc_now,
)
from app.services.attention.adapters import (
    items_from_approval_gates,
    items_from_commercial_access,
    items_from_construction_readiness,
    items_from_governance_policies,
    items_from_handoffs,
    items_from_runtime_operation,
    items_from_stage_artifact_state,
    items_from_stage_payload,
)
from app.services.attention.contract import (
    build_attention_response_v2 as build_attention_contract_response_v2,
    create_attention_item_v2,
    dedupe_attention_items_v2,
    sort_attention_items_v2,
)
from app.services.attention.governor import govern_attention_items
from app.services.attention.validation_issue_normalizer import (
    items_from_validation_issues,
    split_validation_issue_codes,
)
from app.services.commerce_service import resolve_access_request
from app.services.lean_question_policy import filter_stage_question_texts
from app.services.product_processing.contracts import ProductBuildProductKey
from app.services.product_processing.persistence import (
    ProductBuildRunRecord,
    ProductBuildStepRecord,
    UncertaintyBacklogRecord,
)
from app.services.product_processing.product_build_run_service import update_product_build_run_state


PRODUCT_BUILD_ATTENTION_STEP_STATES = {"requires_attention", "locked", "error", "failed"}
ATTENTION_SUPPRESSING_STAGE_OPERATION_STATUSES = (
    StageOperationStatus.queued,
    StageOperationStatus.running,
    StageOperationStatus.waiting_for_user,
)
STAGE_OPERATION_RUNTIME_CAPABILITIES: dict[str, frozenset[str]] = {
    "analyze_discovery": frozenset({"analyze_discovery", "normalize_discovery"}),
    "define_requirements": frozenset({"build_canvas", "define_requirements", "requirements_definition_skill"}),
    "propose_design": frozenset({"critique_agent_design", "propose_agent_design", "synthesize_blueprint_narrative"}),
    "recommend_tools": frozenset({"recommend_minimal_tools"}),
    "recommend_memory": frozenset({"critique_memory_architecture", "recommend_memory_architecture"}),
    "generate_validation_scenarios": frozenset({"generate_validation_scenarios"}),
    "generate_estimation_report": frozenset({"analyze_estimation_risks"}),
}


def _state_value(value) -> str:
    return getattr(value, "value", str(value or ""))


def _count_by(items: list[Any], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(getattr(item, field_name, "") or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _attention_item(
    *,
    key: str,
    title: str,
    item_type: str,
    severity: str,
    stage: str,
    source: str,
    reason: str,
    href: str,
    impact: str = "",
    status: str = "open",
    owner_role: str = "",
    metadata: dict | None = None,
) -> AttentionItemResponse:
    return AttentionItemResponse(
        key=key,
        title=title,
        type=item_type,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        stage=stage,
        source=source,
        reason=reason,
        impact=impact,
        status=status,
        owner_role=owner_role,
        action_label="Resolver" if severity == "blocking" else "Revisar",
        href=href,
        metadata=metadata or {},
    )


def build_attention_response(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
    readiness: ConstructionReadinessReport,
    access: CommercialAccessSnapshotV2,
) -> AttentionResponse:
    base = f"/projects/{record.id}"
    items: list[AttentionItemResponse] = []

    if access.checkout_state == "pending":
        items.append(
            _attention_item(
                key="checkout_pending",
                title="Checkout pendiente",
                item_type="checkout",
                severity="warning",
                stage="commercial",
                source="commerce",
                reason="Existe una orden pendiente antes de activar el producto.",
                impact="El acceso premium se habilitara al confirmar la orden; el modo basico sigue disponible.",
                href=f"{base}/blueprint/pro",
            )
        )

    for approval in getattr(snapshot, "approvals", []) or []:
        if _state_value(getattr(approval, "status", "")) != "pending":
            continue
        stage = _state_value(getattr(approval, "requested_in_stage", "")) or "work"
        gate_key = str(getattr(approval, "gate_key", "") or "")
        is_tool_gate = gate_key.startswith("tool:") or stage in {"tools", "post_validation"}
        severity = "warning" if is_tool_gate else "blocking"
        items.append(
            _attention_item(
                key=f"approval:{getattr(approval, 'id', getattr(approval, 'gate_key', 'pending'))}",
                title=getattr(approval, "title", "Aprobacion pendiente"),
                item_type="approval",
                severity=severity,
                stage=stage,
                source="approval_gate",
                reason=getattr(approval, "rationale", "") or "La etapa requiere aprobacion antes de continuar.",
                impact=getattr(approval, "instructions", ""),
                href=f"{base}/work/{stage}",
                metadata={"gate_key": gate_key},
            )
        )

    for gap in readiness.gaps:
        gap_status = getattr(gap, "status", "open")
        if gap_status not in {"open", ""}:
            continue
        severity = "blocking" if gap.severity == "blocking" else "warning"
        items.append(
            _attention_item(
                key=f"gap:{gap.gap_key}",
                title=gap.title,
                item_type="gap",
                severity=severity,
                stage=gap.blocking_stage or "package",
                source="acp_readiness",
                reason=gap.summary,
                impact=gap.remediation,
                href=f"{base}/acp",
                metadata={
                    "gap_key": gap.gap_key,
                    "domain": gap.domain,
                    "evidence_paths": gap.evidence_paths,
                    "closure_criteria": gap.closure_criteria,
                },
            )
        )
        for question in gap.questions:
            question_key = getattr(question, "question_key", "")
            if not question_key:
                continue
            items.append(
                _attention_item(
                    key=f"question:{question_key}",
                    title=getattr(question, "question_text", "Pregunta de implementacion"),
                    item_type="question",
                    severity="blocking" if getattr(question, "blocking", False) else "warning",
                    stage=gap.blocking_stage or "package",
                    source="acp_questions",
                    reason=getattr(question, "rationale", "") or gap.summary,
                    impact=getattr(question, "purpose", "") or "Cerrar decision humana documentada para la implementacion.",
                    href=f"{base}/acp",
                    owner_role=getattr(question, "target_owner", ""),
                    metadata={"question_key": question_key, "gap_key": gap.gap_key},
                )
            )

    pending_requests = db.exec(
        select(CommercialAccessRequestRecord).where(
            CommercialAccessRequestRecord.workspace_id == record.workspace_id,
            CommercialAccessRequestRecord.session_id == record.id,
            CommercialAccessRequestRecord.status == CommercialAccessRequestStatus.pending,
        )
    ).all()
    for request in pending_requests:
        items.append(
            _attention_item(
                key=f"access-request:{request.id}",
                title=f"Solicitud de acceso a {request.product_key}",
                item_type="entitlement",
                severity="warning",
                stage="commercial",
                source="access_request",
                reason=request.reason or "Un usuario solicito habilitar una capacidad premium.",
                impact="Un owner o admin debe aprobar o rechazar la solicitud.",
                href=f"{base}/attention",
                owner_role="owner/admin",
                metadata={"request_id": str(request.id), "capability": request.capability},
            )
        )

    if not items and access.tier != "acp":
        items.append(
            _attention_item(
                key="acp_next_level",
                title="ACP disponible como siguiente nivel",
                item_type="info",
                severity="info",
                stage="commercial",
                source="product_overview",
                reason="El Blueprint puede evolucionar a paquete portable de construccion.",
                href=f"{base}/acp",
            )
        )

    deduped: dict[str, AttentionItemResponse] = {}
    for item in items:
        deduped[item.key] = item
    ordered = sorted(
        deduped.values(),
        key=lambda item: ({"blocking": 0, "warning": 1, "info": 2}.get(item.severity, 3), item.stage, item.key),
    )
    return AttentionResponse(
        session_id=record.id,
        workspace_id=record.workspace_id,
        total_count=len(ordered),
        blocking_count=sum(1 for item in ordered if item.severity == "blocking"),
        warning_count=sum(1 for item in ordered if item.severity == "warning"),
        info_count=sum(1 for item in ordered if item.severity == "info"),
        items=ordered,
    )


@dataclass(frozen=True)
class AttentionActionApplyResult:
    status: Literal["applied", "duplicate", "unsupported", "not_found", "conflict", "forbidden"]
    message: str
    resume_eligible: bool = False


def _pending_access_requests(db: Session, record: SessionRecord) -> list[CommercialAccessRequestRecord]:
    return db.exec(
        select(CommercialAccessRequestRecord).where(
            CommercialAccessRequestRecord.workspace_id == record.workspace_id,
            CommercialAccessRequestRecord.session_id == record.id,
            CommercialAccessRequestRecord.status == CommercialAccessRequestStatus.pending,
        )
    ).all()


def _approved_access_requests(db: Session, record: SessionRecord) -> list[CommercialAccessRequestRecord]:
    return db.exec(
        select(CommercialAccessRequestRecord).where(
            CommercialAccessRequestRecord.workspace_id == record.workspace_id,
            CommercialAccessRequestRecord.session_id == record.id,
            CommercialAccessRequestRecord.status == CommercialAccessRequestStatus.approved,
        )
    ).all()


def _answered_construction_question_keys(db: Session, record: SessionRecord) -> set[str]:
    rows = db.exec(
        select(ConstructionQuestionResponseRecord).where(
            ConstructionQuestionResponseRecord.session_id == record.id,
            ConstructionQuestionResponseRecord.status.in_(["answered", "resolved"]),
        )
    ).all()
    return {item.question_key for item in rows if item.question_key}


def _answered_attention_item_keys(db: Session, record: SessionRecord) -> set[str]:
    rows = db.exec(
        select(CommercialEventRecord).where(
            CommercialEventRecord.workspace_id == record.workspace_id,
            CommercialEventRecord.session_id == record.id,
            CommercialEventRecord.event_key == "attention_action_v2",
        )
    ).all()
    answered: set[str] = set()
    for event in rows:
        metadata = event.metadata_payload or {}
        if metadata.get("action_kind") not in {"answer", "confirm", "defer", "dismiss"} or metadata.get("result_status") != "applied":
            continue
        item_key = str(metadata.get("item_key") or "").strip()
        if item_key:
            answered.add(item_key)
    return answered


def _build_suppression_matcher(db: Session, record: SessionRecord):
    from app.services.attention.contract import _slug

    answered_keys = _answered_attention_item_keys(db, record)
    active_operations = db.exec(
        select(StageOperationRecord).where(
            StageOperationRecord.workspace_id == record.workspace_id,
            StageOperationRecord.session_id == record.id,
            StageOperationRecord.status.in_(list(ATTENTION_SUPPRESSING_STAGE_OPERATION_STATUSES)),
        )
    ).all()
    backlog_records = db.exec(
        select(UncertaintyBacklogRecord).where(
            UncertaintyBacklogRecord.workspace_id == record.workspace_id,
            UncertaintyBacklogRecord.session_id == record.id,
            UncertaintyBacklogRecord.status.in_(["resolved", "deferred", "dismissed", "superseded"]),
        )
    ).all()

    suppressed_exact_keys: set[str] = set(answered_keys)
    suppressed_entities: set[str] = set()
    suppressed_titles: set[str] = set()
    active_stage_keys = {
        _normalize_stage_key(getattr(operation, "stage_key", ""))
        for operation in active_operations
        if _normalize_stage_key(getattr(operation, "stage_key", ""))
    }
    active_runtime_capabilities = {
        capability
        for operation in active_operations
        for capability in STAGE_OPERATION_RUNTIME_CAPABILITIES.get(str(getattr(operation, "action", "") or "").strip().lower(), ())
    }

    def _add_token(token: str):
        if not token:
            return
        clean = token.strip()
        if not clean:
            return
        suppressed_exact_keys.add(clean)
        suppressed_entities.add(clean)
        suppressed_entities.add(clean.lower())
        slugged = _slug(clean)
        if slugged:
            suppressed_exact_keys.add(slugged)
            suppressed_entities.add(slugged)

    for b in backlog_records:
        if b.uncertainty_key:
            _add_token(b.uncertainty_key)
            if ":" in b.uncertainty_key:
                for part in b.uncertainty_key.split(":"):
                    if len(part.strip()) > 2 and not part.strip().isdigit():
                        _add_token(part.strip())
        if b.title:
            suppressed_titles.add(b.title.strip().lower())
            _add_token(b.title)
        if b.source_refs:
            for s_ref in b.source_refs:
                if isinstance(s_ref, str) and s_ref.strip():
                    _add_token(s_ref)
        if b.payload and isinstance(b.payload, dict):
            for k in ("key", "gap_key", "finding_key", "entity_id", "question", "title"):
                val = str(b.payload.get(k) or "").strip()
                if val:
                    _add_token(val)
                    if k in {"question", "title"}:
                        suppressed_titles.add(val.lower())

    def is_suppressed(item: AttentionItemV2) -> bool:
        if item.key in suppressed_exact_keys:
            return True

        entity_id = str(getattr(item.source_ref, "entity_id", "") or "").strip()
        if entity_id:
            if entity_id in suppressed_exact_keys or entity_id in suppressed_entities or entity_id.lower() in suppressed_entities:
                return True
            if _slug(entity_id) in suppressed_entities:
                return True
            if ":" in entity_id:
                for part in entity_id.split(":"):
                    p = part.strip()
                    if p in suppressed_entities or p.lower() in suppressed_entities or _slug(p) in suppressed_entities:
                        return True

        item_title = item.title.strip().lower()
        if item_title and item_title in suppressed_titles:
            return True

        for s_key in suppressed_exact_keys:
            if len(s_key) >= 4 and s_key in item.key:
                return True

        item_stage = _normalize_stage_key(item.stage)
        if item.type == "stale" and item_stage in active_stage_keys:
            return True
        if item.type == "runtime_error":
            if item_stage in active_stage_keys:
                return True
            diagnostics = item.diagnostics
            capability = _slug(getattr(diagnostics, "capability", ""))
            if capability and capability in active_runtime_capabilities:
                return True
            trace_refs = [str(ref or "").strip().lower() for ref in getattr(diagnostics, "trace_refs", []) or []]
            for active_capability in active_runtime_capabilities:
                if any(f"runtime.capability:{active_capability}" in ref for ref in trace_refs):
                    return True

        return False

    return is_suppressed


def _normalize_stage_key(value) -> str:
    return _state_value(value).strip() or "discover"


def _stage_href(base: str, stage: str) -> str:
    normalized = stage.strip() or "discover"
    if normalized in {"commercial", "acp"}:
        return f"{base}/{normalized}"
    return f"{base}/{normalized}"


def _product_for_stage(stage: str) -> str:
    return "acp" if stage in {"validate", "estimate", "package", "acp", "runtime_configuration"} else "blueprint"


def _text_from_entry(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("title", "question", "question_text", "summary", "detail", "reason", "message", "gap_key", "finding_key"):
            raw = value.get(key)
            if raw:
                return str(raw).strip()
    for key in ("title", "question", "question_text", "summary", "detail", "reason", "message", "gap_key", "finding_key"):
        raw = getattr(value, key, None)
        if raw:
            return str(raw).strip()
    return ""


def _list_from_payload(payload: dict, *keys: str) -> list[str]:
    items: list[str] = []
    for key in keys:
        raw = payload.get(key)
        if raw is None:
            continue
        raw_items = raw if isinstance(raw, list) else [raw]
        for raw_item in raw_items:
            text = _text_from_entry(raw_item)
            if text:
                items.append(text)
    return items


def _entries_from_payload(payload: dict, *keys: str) -> list[Any]:
    items: list[Any] = []
    for key in keys:
        raw = payload.get(key)
        if raw is None:
            continue
        raw_items = raw if isinstance(raw, list) else [raw]
        for raw_item in raw_items:
            if isinstance(raw_item, dict):
                items.append(raw_item)
                continue
            text = _text_from_entry(raw_item)
            if text:
                items.append(text)
    return items


def _normalize_attention_severity(value) -> str:
    normalized = _state_value(value).strip().lower()
    if normalized in {"blocking", "critical", "high", "fail", "failed"}:
        return "blocking"
    if normalized in {"warning", "medium", "needs_review", "partial"}:
        return "warning"
    return "info"


def _fallback_tool_question_options(question: Any) -> list[dict[str, Any]]:
    question_text = str(getattr(question, "question", "") or getattr(question, "title", "") or "").strip()
    if not question_text:
        return []
    return [
        {
            "key": "accept_minimal_recommendation",
            "label": "Aceptar recomendacion minima",
            "description": "Usar la clasificacion propuesta y conservar la herramienta solo si cubre una capacidad requerida.",
            "impact": "Reduce sobreaprovisionamiento y mantiene avance de Tools.",
            "example": "Mantener document_ingestion como obligatoria si Memory requiere RAG.",
            "recommended": True,
            "confidence": 0.72,
            "source_refs": ["tool_recommendation.needs_information"],
        },
        {
            "key": "defer_to_acp",
            "label": "Diferir detalle al ACP",
            "description": "Cerrar la etapa con una decision funcional y dejar endpoint, credenciales o stack para implementacion.",
            "impact": "Evita bloquear el Blueprint con decisiones tecnicas prematuras.",
            "example": "Definir la API final durante el paquete de construccion.",
            "recommended": False,
            "confidence": 0.64,
            "source_refs": ["tool_recommendation.needs_information"],
        },
    ]


def _decision_entries_from_payload(payload: dict, *keys: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for key in keys:
        raw = payload.get(key)
        if raw is None:
            continue
        raw_items = raw if isinstance(raw, list) else [raw]
        for index, raw_item in enumerate(raw_items, start=1):
            if isinstance(raw_item, dict):
                title = _text_from_entry(raw_item)
                if not title:
                    continue
                entries.append(
                    {
                        "key": str(raw_item.get("finding_key") or raw_item.get("gap_key") or raw_item.get("key") or f"{key}_{index}"),
                        "title": title,
                        "reason": str(raw_item.get("detail") or raw_item.get("reason") or raw_item.get("summary") or title),
                        "impact": str(raw_item.get("impact") or raw_item.get("suggested_action") or ""),
                        "severity": _normalize_attention_severity(raw_item.get("severity")),
                        "owner_role": str(raw_item.get("owner_role") or "business_owner"),
                    }
                )
                continue
            title = _text_from_entry(raw_item)
            if title:
                entries.append(
                    {
                        "key": f"{key}_{index}",
                        "title": title,
                        "reason": title,
                        "impact": "",
                        "severity": "warning",
                        "owner_role": "business_owner",
                    }
                )
    return entries


def _items_from_journey_artifacts(snapshot: SessionSnapshot, *, base: str) -> list[AttentionItemV2]:
    items: list[AttentionItemV2] = []
    artifacts = list((snapshot.journey_latest_artifacts or {}).values())
    seen_ids: set[UUID] = set()
    for artifact in artifacts:
        if artifact.id in seen_ids:
            continue
        seen_ids.add(artifact.id)
        stage = artifact.stage_key or "discover"
        href = _stage_href(base, stage)
        product = _product_for_stage(stage)
        state = _state_value(artifact.state)
        items.extend(
            items_from_stage_artifact_state(
                stage=stage,
                artifact_id=str(artifact.id),
                artifact_version=artifact.version_number,
                artifact_kind=artifact.artifact_kind or "stage_artifact",
                state=state,
                reason=", ".join(artifact.stale_reasons) or f"El artefacto esta en estado {state}.",
                href=href,
                return_href=href,
            )
        )
        if artifact.state == JourneyArtifactState.stale or artifact.stale_reasons:
            continue
        payload = artifact.proposal_payload or {}
        validation_issues, missing_information = split_validation_issue_codes(artifact.missing_information)
        open_questions = _entries_from_payload(payload, "guided_questions", "open_questions", "needs_information")
        open_questions.extend(f"Falta informacion: {item}" for item in missing_information)
        open_questions = filter_stage_question_texts(stage, open_questions)
        warnings = list(artifact.warnings)
        warnings.extend(_list_from_payload(payload, "warnings"))
        gaps = _list_from_payload(payload, "gaps", "coverage_gaps")
        decisions = _decision_entries_from_payload(payload, "critic_findings", "findings")
        items.extend(
            items_from_validation_issues(
                validation_issues,
                product=product,
                stage=stage,
                source=f"journey.{artifact.artifact_kind or stage}",
                artifact_id=str(artifact.id),
                artifact_version=artifact.version_number,
                href=href,
                return_href=href,
            )
        )
        items.extend(
            items_from_stage_payload(
                product=product,
                stage=stage,
                source=f"journey.{artifact.artifact_kind or stage}",
                artifact_id=str(artifact.id),
                artifact_version=artifact.version_number,
                href=href,
                return_href=href,
                open_questions=open_questions,
                gaps=gaps,
                decisions=decisions,
                warnings=warnings,
            )
        )
    return items


def _items_from_tool_recommendation(snapshot: SessionSnapshot, *, base: str) -> list[AttentionItemV2]:
    recommendation = snapshot.latest_tool_recommendation
    if recommendation is None:
        return []
    version = recommendation.current_blueprint_version or recommendation.source_blueprint_version
    href = _stage_href(base, "tools")
    items: list[AttentionItemV2] = []
    if recommendation.is_stale:
        items.append(
            create_attention_item_v2(
                item_type="stale",
                severity="warning",
                product="blueprint",
                stage="tools",
                source="tool_recommendation",
                source_ref={"artifact_id": "tool_recommendation", "artifact_version": version, "field_path": "is_stale"},
                title="La recomendacion de herramientas esta desactualizada",
                reason=", ".join(recommendation.stale_reasons) or "La etapa Tools debe regenerarse con informacion mas reciente.",
                impact="Memory y Validate pueden quedar referenciando herramientas anteriores.",
                consequence_if_unresolved="El blueprint podria conservar contratos de herramientas no vigentes.",
                action_kind="regenerate",
                href=href,
                return_href=href,
            )
        )
    for gap in recommendation.coverage_gaps:
        items.append(
            create_attention_item_v2(
                item_type="gap",
                severity=_normalize_attention_severity(gap.severity),
                product="blueprint",
                stage="tools",
                source="tool_recommendation.coverage",
                source_ref={"artifact_id": "tool_recommendation", "artifact_version": version, "entity_id": gap.gap_key, "field_path": "coverage_gaps"},
                title=gap.title or gap.gap_key or "GAP de cobertura de herramientas",
                reason=gap.reason or gap.question or "La cobertura de herramientas requiere revision.",
                impact=gap.impact,
                consequence_if_unresolved="El agente podria quedar sin una capacidad necesaria o con una herramienta sobredimensionada.",
                action_kind="navigate",
                href=href,
                return_href=href,
            )
        )
    for question in recommendation.needs_information:
        items.append(
            create_attention_item_v2(
                item_type="question",
                severity=_normalize_attention_severity(question.severity),
                product="blueprint",
                stage="tools",
                source="tool_recommendation.needs_information",
                source_ref={"artifact_id": "tool_recommendation", "artifact_version": version, "entity_id": question.gap_key, "field_path": "needs_information"},
                title=question.question or question.title or "Tools requiere informacion adicional",
                reason=question.reason or "El LLM necesita una decision humana para seleccionar el set minimo de herramientas.",
                impact=question.impact,
                consequence_if_unresolved="La propuesta de herramientas quedara con incertidumbre explicita.",
                action_kind="answer",
                href=href,
                return_href=href,
                options=[
                    option.model_dump(mode="json")
                    for option in getattr(question, "answer_options", []) or []
                ]
                or _fallback_tool_question_options(question),
                suggested_answer=getattr(question, "suggested_answer", "") or "Aceptar recomendacion minima",
                can_resolve_inline=True,
            )
        )
    for finding in recommendation.evaluation.findings:
        items.append(
            create_attention_item_v2(
                item_type="inconsistency",
                severity=finding.severity,
                product="blueprint",
                stage="tools",
                source="tool_recommendation.evaluation",
                source_ref={"artifact_id": "tool_recommendation", "artifact_version": version, "entity_id": finding.finding_key, "field_path": "evaluation.findings"},
                title=finding.title or "Finding en herramientas",
                reason=finding.detail or finding.suggested_action or "La evaluacion de herramientas detecto un finding.",
                impact=finding.suggested_action,
                consequence_if_unresolved="La calidad o minimalidad del set de herramientas podria quedar comprometida.",
                action_kind="navigate",
                href=href,
                return_href=href,
                affected_artifact_refs=finding.affected_tool_keys,
            )
        )
    return items


def _items_from_short_term_runtime(snapshot: SessionSnapshot, *, base: str) -> list[AttentionItemV2]:
    runtime = snapshot.short_term_memory
    if runtime is None:
        return []
    items: list[AttentionItemV2] = []
    memory = runtime.memory
    for approval in memory.pending_approvals:
        items.extend(
            items_from_runtime_operation(
                {
                    "id": approval,
                    "state": "waiting_for_user",
                    "stage": memory.active_stage or "memory",
                    "product": "blueprint",
                    "title": f"Aprobacion runtime pendiente: {approval}",
                    "message": "La memoria de corto plazo conserva una aprobacion pendiente.",
                    "owner_role": "operator",
                },
                href=f"{base}/attention",
                return_href=f"{base}/{memory.active_stage or 'memory'}",
            )
        )
    for handoff in memory.open_handoffs:
        items.extend(
            items_from_runtime_operation(
                {
                    "id": handoff,
                    "state": "waiting_for_user",
                    "stage": memory.active_stage or "memory",
                    "product": "blueprint",
                    "title": f"Handoff runtime pendiente: {handoff}",
                    "message": "La memoria de corto plazo conserva un handoff abierto.",
                    "owner_role": "operator",
                },
                href=f"{base}/attention",
                return_href=f"{base}/{memory.active_stage or 'memory'}",
            )
        )
    return items


def _items_from_failed_runs(snapshot: SessionSnapshot, *, base: str) -> list[AttentionItemV2]:
    items: list[AttentionItemV2] = []
    for run in [*snapshot.skill_runs, *snapshot.subagent_runs]:
        status = _state_value(getattr(run, "status", ""))
        if status not in {"failed", "needs_review"}:
            continue
        stage = _normalize_stage_key(getattr(run, "stage", getattr(run, "run_kind", "runtime")))
        run_id = str(getattr(run, "id", "run"))
        items.extend(
            items_from_runtime_operation(
                {
                    "id": run_id,
                    "state": "error" if status == "failed" else "blocked",
                    "stage": stage,
                    "product": _product_for_stage(stage),
                    "title": getattr(run, "title", "") or getattr(run, "skill_key", "") or "Ejecucion requiere revision",
                    "message": getattr(run, "summary", "") or getattr(run, "result_summary", "") or "Una ejecucion quedo en estado no exitoso.",
                    "owner_role": "operator",
                },
                href=f"{base}/attention",
                return_href=_stage_href(base, stage),
            )
        )
    return items


def _latest_product_build_runs(db: Session, record: SessionRecord) -> list[ProductBuildRunRecord]:
    runs = db.exec(
        select(ProductBuildRunRecord)
        .where(
            ProductBuildRunRecord.workspace_id == record.workspace_id,
            ProductBuildRunRecord.session_id == record.id,
        )
        .order_by(ProductBuildRunRecord.updated_at.desc())
    ).all()
    latest_by_product: dict[str, ProductBuildRunRecord] = {}
    for run in runs:
        latest_by_product.setdefault(run.product_key, run)
    return list(latest_by_product.values())


def _items_from_product_build_steps(db: Session, *, record: SessionRecord, base: str, return_href: str) -> list[AttentionItemV2]:
    latest_runs = _latest_product_build_runs(db, record)
    if not latest_runs:
        return []
    latest_run_ids = [run.id for run in latest_runs]
    run_by_id = {run.id: run for run in latest_runs}
    steps = db.exec(
        select(ProductBuildStepRecord)
        .where(
            ProductBuildStepRecord.workspace_id == record.workspace_id,
            ProductBuildStepRecord.session_id == record.id,
            ProductBuildStepRecord.run_id.in_(latest_run_ids),
            ProductBuildStepRecord.status.in_(list(PRODUCT_BUILD_ATTENTION_STEP_STATES)),
        )
        .order_by(ProductBuildStepRecord.updated_at.desc())
    ).all()
    items: list[AttentionItemV2] = []
    for step in steps:
        run = run_by_id.get(step.run_id)
        if run is None:
            continue
        items.append(_product_build_step_attention_item(record=record, run=run, step=step, base=base, return_href=return_href))
    return items


def _product_build_step_attention_item(
    *,
    record: SessionRecord,
    run: ProductBuildRunRecord,
    step: ProductBuildStepRecord,
    base: str,
    return_href: str,
) -> AttentionItemV2:
    checkpoint = step.checkpoint_payload or {}
    error = step.error_payload or {}
    stage = _normalize_stage_key(step.stage_key or checkpoint.get("stage_key") or checkpoint.get("stage") or "package")
    item_type = _product_build_step_type(step, checkpoint)
    title = _product_build_step_title(step, checkpoint, error)
    reason = _product_build_step_reason(step, checkpoint, error)
    action_kind = "retry" if step.status in {"error", "failed"} else "navigate"
    return create_attention_item_v2(
        item_type=item_type,
        severity="blocking",
        product=_product_for_product_build(run.product_key),
        stage=stage,
        source="product_build_step",
        title=title,
        reason=reason,
        impact=f"El producto {run.product_key} no puede cerrarse hasta resolver este paso.",
        consequence_if_unresolved="El build quedara detenido o parcialmente actualizado y no se podra promover con trazabilidad completa.",
        action_kind=action_kind,
        action_label="Reintentar" if action_kind == "retry" else "Abrir",
        href=_product_build_step_href(record.id, run=run, step=step, stage=stage),
        return_href=return_href,
        source_ref={
            "artifact_id": str(run.id),
            "entity_id": str(step.id),
            "field_path": step.step_key,
        },
        affected_artifact_refs=[ref for ref in (step.deliverable_key, step.dependency_key) if ref],
        diagnostics={
            "summary": reason,
            "technical_message": _product_build_step_technical_message(step, error),
            "error_kind": "runtime" if step.status in {"error", "failed"} else "dependency",
            "capability": step.deliverable_key or step.dependency_key or step.step_key,
            "capability_label": title,
            "operation_id": str(run.id),
            "retry_policy": "Reintento seguro desde checkpoint del ProductBuild." if action_kind == "retry" else "",
            "repair_hint": "Revisa el origen indicado, resuelve la dependencia o reintenta el step desde el producto.",
            "trace_refs": [
                f"product_build.run:{run.id}",
                f"product_build.step:{step.id}",
                f"product_build.step_key:{step.step_key}",
            ],
        },
    )


def _product_build_step_type(step: ProductBuildStepRecord, checkpoint: dict[str, Any]) -> str:
    if step.status in {"error", "failed"}:
        return "runtime_error"
    kind = str(checkpoint.get("kind") or checkpoint.get("uncertainty_kind") or "").strip().lower()
    if kind in {"question", "decision", "approval", "hitl", "validation", "inconsistency", "stale"}:
        return kind
    return "gap"


def _product_build_step_title(step: ProductBuildStepRecord, checkpoint: dict[str, Any], error: dict[str, Any]) -> str:
    title = str(error.get("title") or checkpoint.get("title") or checkpoint.get("label") or "").strip()
    if title:
        return title
    label = step.deliverable_key or step.dependency_key or step.step_key
    if step.status in {"error", "failed"}:
        return f"No se pudo completar {label}"
    return f"Resolver dependencia de producto: {label}"


def _product_build_step_reason(step: ProductBuildStepRecord, checkpoint: dict[str, Any], error: dict[str, Any]) -> str:
    reason = str(
        error.get("message")
        or error.get("technical_message")
        or checkpoint.get("next_action")
        or checkpoint.get("summary")
        or checkpoint.get("reason")
        or ""
    ).strip()
    return reason or "Este paso del producto requiere atencion para continuar sin perder trazabilidad."


def _product_build_step_technical_message(step: ProductBuildStepRecord, error: dict[str, Any]) -> str:
    message = str(error.get("technical_message") or error.get("message") or "").strip()
    if message:
        return message
    return f"ProductBuild step {step.step_key} quedo en estado {step.status}."


def _product_for_product_build(product_key: str) -> str:
    return "acp" if product_key == ProductBuildProductKey.acp.value else "blueprint"


def _product_build_step_href(record_id: UUID, *, run: ProductBuildRunRecord, step: ProductBuildStepRecord, stage: str) -> str:
    if step.deliverable_key:
        return f"/projects/{record_id}/blueprint?deliverable={step.deliverable_key}"
    if run.product_key == ProductBuildProductKey.acp.value:
        return f"/projects/{record_id}/acp"
    return f"/projects/{record_id}/work/{stage}"


def _collect_attention_v2_items(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
    readiness: ConstructionReadinessReport,
    access: CommercialAccessSnapshotV2,
) -> list[AttentionItemV2]:
    base = f"/projects/{record.id}"
    return_href = f"{base}/attention"
    pending_requests = _pending_access_requests(db, record)
    approved_requests = _approved_access_requests(db, record)
    answered_question_keys = _answered_construction_question_keys(db, record)
    items: list[AttentionItemV2] = []
    items.extend(
        items_from_commercial_access(
            access,
            pending_requests,
            approved_requests,
            base_href=base,
            return_href=return_href,
        )
    )
    items.extend(items_from_approval_gates(snapshot.approvals, base_href=base, return_href=return_href))
    if _can_surface_acp_attention(access):
        items.extend(
            items_from_construction_readiness(
                readiness,
                base_href=f"{base}/acp",
                return_href=return_href,
                answered_question_keys=answered_question_keys,
            )
        )
    items.extend(_items_from_journey_artifacts(snapshot, base=base))
    items.extend(_items_from_tool_recommendation(snapshot, base=base))
    items.extend(items_from_handoffs(snapshot.handoff_records, base_href=base, return_href=return_href))
    items.extend(items_from_governance_policies(snapshot.governance_policies, base_href=base, return_href=return_href))
    items.extend(_items_from_short_term_runtime(snapshot, base=base))
    items.extend(_items_from_product_build_steps(db, record=record, base=base, return_href=return_href))
    items.extend(_items_from_failed_runs(snapshot, base=base))
    return items


def _collect_governed_attention_v2_items(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
    readiness: ConstructionReadinessReport,
    access: CommercialAccessSnapshotV2,
    current_stage: str = "",
) -> list[AttentionItemV2]:
    return govern_attention_items(
        _collect_attention_v2_items(db, record=record, snapshot=snapshot, readiness=readiness, access=access),
        record=record,
        access=access,
        current_stage=current_stage,
    )


def _can_surface_acp_attention(access: Any) -> bool:
    if getattr(access, "tier", None) == CommercialTier.acp or str(getattr(access, "tier", "")) == "acp":
        return True
    capabilities = getattr(access, "capabilities", []) or []
    return any(getattr(item, "capability", "") == "acp.build" and getattr(item, "allowed", False) for item in capabilities)


def _matches_filter(value: str, expected: str | None) -> bool:
    return expected is None or expected == "" or value == expected


def _parse_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        return max(0, int(cursor))
    except ValueError:
        return 0


def build_attention_response_v2(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
    readiness: ConstructionReadinessReport,
    access: CommercialAccessSnapshotV2,
    current_stage: str = "",
    stage: str | None = None,
    product: str | None = None,
    severity: str | None = None,
    item_type: str | None = None,
    item_status: str | None = None,
    cursor: str | None = None,
    limit: int = 50,
) -> AttentionResponseV2:
    limit = min(max(limit, 1), 100)
    is_suppressed = _build_suppression_matcher(db, record)
    items = [
        item
        for item in dedupe_attention_items_v2(
            _collect_governed_attention_v2_items(
                db,
                record=record,
                snapshot=snapshot,
                readiness=readiness,
                access=access,
                current_stage=current_stage,
            ),
            current_stage=current_stage,
        )
        if not is_suppressed(item)
    ]
    filtered = [
        item
        for item in items
        if _matches_filter(item.stage, stage)
        and _matches_filter(item.product, product)
        and _matches_filter(item.severity, severity)
        and _matches_filter(item.type, item_type)
        and _matches_filter(item.status, item_status)
    ]
    ordered = sort_attention_items_v2(filtered, current_stage=current_stage)
    offset = _parse_cursor(cursor)
    page = ordered[offset : offset + limit]
    next_offset = offset + limit
    next_cursor = str(next_offset) if next_offset < len(ordered) else ""
    return build_attention_contract_response_v2(
        session_id=record.id,
        workspace_id=record.workspace_id,
        items=page,
        count_items=ordered,
        current_stage=current_stage,
        cursor=next_cursor,
    )


def build_attention_metrics_v2(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
    readiness: ConstructionReadinessReport,
    access: CommercialAccessSnapshotV2,
    current_stage: str = "",
) -> dict[str, Any]:
    is_suppressed = _build_suppression_matcher(db, record)
    all_items = dedupe_attention_items_v2(
        _collect_governed_attention_v2_items(
            db,
            record=record,
            snapshot=snapshot,
            readiness=readiness,
            access=access,
            current_stage=current_stage,
        ),
        current_stage=current_stage,
    )
    visible_items = [item for item in all_items if not is_suppressed(item)]
    question_items = [item for item in visible_items if item.type == "question"]
    events = db.exec(
        select(CommercialEventRecord).where(
            CommercialEventRecord.workspace_id == record.workspace_id,
            CommercialEventRecord.session_id == record.id,
            CommercialEventRecord.event_key == "attention_action_v2",
        )
    ).all()
    answer_events = [
        event.metadata_payload or {}
        for event in events
        if (event.metadata_payload or {}).get("action_kind") == "answer"
    ]
    return {
        "contract_version": "attention-metrics.v2",
        "session_id": str(record.id),
        "workspace_id": str(record.workspace_id),
        "current_stage": current_stage,
        "visible_questions": len(question_items),
        "visible_questions_with_suggested_answer": sum(1 for item in question_items if item.suggested_answer),
        "visible_questions_with_options": sum(1 for item in question_items if item.options),
        "visible_answer_options": sum(len(item.options) for item in question_items),
        "answered_questions": sum(1 for event in answer_events if event.get("result_status") == "applied"),
        "suggested_answer_acceptances": sum(
            1
            for event in answer_events
            if event.get("result_status") == "applied" and event.get("was_suggested_answer_used") is True
        ),
        "manual_answers": sum(
            1
            for event in answer_events
            if event.get("result_status") == "applied" and event.get("was_suggested_answer_used") is not True
        ),
        "selected_option_answers": sum(
            1
            for event in answer_events
            if event.get("result_status") == "applied" and bool(event.get("selected_option_key"))
        ),
        "questions_by_stage": _count_by(question_items, "stage"),
        "questions_by_product": _count_by(question_items, "product"),
        "generated_at": utc_now().isoformat(),
    }


def _find_attention_item_for_action(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
    readiness: ConstructionReadinessReport,
    access: CommercialAccessSnapshotV2,
    item_key: str,
) -> AttentionItemV2 | None:
    items = dedupe_attention_items_v2(
        _collect_governed_attention_v2_items(
            db,
            record=record,
            snapshot=snapshot,
            readiness=readiness,
            access=access,
        )
    )
    return next((item for item in items if item.key == item_key), None)


def _find_idempotency_event(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    idempotency_key: str,
) -> CommercialEventRecord | None:
    if not idempotency_key:
        return None
    events = db.exec(
        select(CommercialEventRecord).where(
            CommercialEventRecord.workspace_id == record.workspace_id,
            CommercialEventRecord.session_id == record.id,
            CommercialEventRecord.user_id == current_user.id,
            CommercialEventRecord.event_key == "attention_action_v2",
        )
    ).all()
    return next((event for event in events if event.metadata_payload.get("idempotency_key") == idempotency_key), None)


def _record_attention_action_event(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    item: AttentionItemV2,
    payload: AttentionActionRequestV2,
    result_status: str,
    message: str,
) -> None:
    db.add(
        CommercialEventRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=current_user.id,
            event_key="attention_action_v2",
            product_key=item.product,
            source="attention_v2",
            correlation_id=payload.idempotency_key,
            metadata_payload={
                "idempotency_key": payload.idempotency_key,
                "item_key": item.key,
                "action_kind": payload.action_kind,
                "answer_text": payload.answer_text,
                "selected_option_key": payload.selected_option_key,
                "was_suggested_answer_used": payload.was_suggested_answer_used,
                "resolution_note": payload.resolution_note,
                "source_artifact_version": payload.source_artifact_version,
                "payload": payload.payload,
                "result_status": result_status,
                "message": message or f"Attention action {payload.action_kind} {result_status}",
                "source": item.source,
                "source_ref": item.source_ref.model_dump(mode="json"),
                "stage_key": item.stage,
                "owner_role": item.owner_role,
                "affected_artifact_refs": item.affected_artifact_refs,
            },
        )
    )


def _apply_acp_question_answer(
    db: Session,
    *,
    record: SessionRecord,
    item: AttentionItemV2,
    payload: AttentionActionRequestV2,
    current_user: UserRecord,
) -> AttentionActionApplyResult:
    question_key = item.source_ref.entity_id or ""
    if not question_key:
        return AttentionActionApplyResult(status="conflict", message="La pregunta no tiene referencia de origen.")
    answer = payload.answer_text.strip()
    if not answer:
        return AttentionActionApplyResult(status="conflict", message="La respuesta no puede estar vacia.")
    now = utc_now()
    response = db.exec(
        select(ConstructionQuestionResponseRecord).where(
            ConstructionQuestionResponseRecord.session_id == record.id,
            ConstructionQuestionResponseRecord.question_key == question_key,
        )
    ).first()
    if response is None:
        response = ConstructionQuestionResponseRecord(
            session_id=record.id,
            question_key=question_key,
            created_at=now,
        )
    response.gap_key = str(item.source_ref.field_path or "").split(".")[1] if item.source_ref.field_path and "." in item.source_ref.field_path else ""
    response.question_text = item.title
    response.rationale = item.reason
    response.expected_answer_format = str(payload.payload.get("expected_answer_format", ""))
    response.target_owner = item.owner_role
    response.blocking = item.blocking
    response.status = "answered"
    response.answer_text = answer
    response.owner_role = payload.payload.get("owner_role", "") or item.owner_role
    response.impacted_artifacts = item.affected_artifact_refs
    response.answered_by_user_id = current_user.id
    response.answered_by_display = current_user.full_name or current_user.email
    response.answered_at = now
    response.resolved_at = None
    response.updated_at = now
    db.add(response)
    db.flush()
    return AttentionActionApplyResult(status="applied", message="Pregunta ACP respondida.")


def _apply_approval_action(
    db: Session,
    *,
    record: SessionRecord,
    item: AttentionItemV2,
    payload: AttentionActionRequestV2,
) -> AttentionActionApplyResult:
    approval_id = item.source_ref.entity_id
    if not approval_id:
        return AttentionActionApplyResult(status="conflict", message="La aprobacion no tiene referencia de origen.")
    approval = db.exec(
        select(ApprovalGateRecord).where(
            ApprovalGateRecord.session_id == record.id,
            ApprovalGateRecord.id == UUID(approval_id),
        )
    ).first()
    if approval is None:
        return AttentionActionApplyResult(status="not_found", message="Approval gate no encontrado.")
    if approval.status != ApprovalStatus.pending:
        return AttentionActionApplyResult(status="applied", message="Approval gate ya estaba resuelto.")
    decision = ApprovalStatus.approved if payload.action_kind == "approve" else ApprovalStatus.rejected
    approval.status = decision
    approval.resolution_note = payload.resolution_note
    approval.resolved_at = utc_now()
    db.add(approval)
    db.flush()
    return AttentionActionApplyResult(status="applied", message=f"Approval gate {decision.value}.")


def _apply_access_request_action(
    db: Session,
    *,
    record: SessionRecord,
    item: AttentionItemV2,
    payload: AttentionActionRequestV2,
    current_user: UserRecord,
) -> AttentionActionApplyResult:
    request_id = item.source_ref.entity_id
    if not request_id:
        return AttentionActionApplyResult(status="conflict", message="La solicitud no tiene referencia de origen.")
    access_request = db.get(CommercialAccessRequestRecord, UUID(request_id))
    if access_request is None or access_request.workspace_id != record.workspace_id or access_request.session_id != record.id:
        return AttentionActionApplyResult(status="not_found", message="Solicitud de acceso no encontrada.")
    decision = "approved" if payload.action_kind in {"approve", "confirm"} else "rejected"
    try:
        resolve_access_request(
            db,
            access_request=access_request,
            payload=AccessRequestResolveRequest(decision=decision, resolution_note=payload.resolution_note),
            current_user=current_user,
        )
    except PermissionError:
        return AttentionActionApplyResult(status="forbidden", message="El usuario no puede resolver solicitudes de acceso.")
    return AttentionActionApplyResult(status="applied", message=f"Solicitud de acceso {decision}.")


def _selected_option_label(item: AttentionItemV2, selected_option_key: str) -> str:
    key = selected_option_key.strip()
    if not key:
        return ""
    option = next((candidate for candidate in item.options if candidate.key == key), None)
    return option.label if option is not None else key


def _resolution_text(item: AttentionItemV2, payload: AttentionActionRequestV2) -> str:
    if payload.answer_text.strip():
        return payload.answer_text.strip()
    if payload.selected_option_key.strip():
        return _selected_option_label(item, payload.selected_option_key)
    if payload.was_suggested_answer_used and item.suggested_answer.strip():
        return item.suggested_answer.strip()
    if payload.action_kind == "defer":
        return payload.resolution_note.strip() or "Decision diferida con justificacion registrada."
    if payload.action_kind == "confirm":
        return payload.resolution_note.strip() or item.suggested_answer.strip() or "Decision confirmada."
    return ""


def _apply_journey_artifact_attention_resolution(
    db: Session,
    *,
    record: SessionRecord,
    item: AttentionItemV2,
    payload: AttentionActionRequestV2,
    current_user: UserRecord,
    resolution_text: str,
) -> bool:
    if not item.source.startswith("journey."):
        return False
    artifact_id = item.source_ref.artifact_id
    if not artifact_id:
        return False
    try:
        artifact_uuid = UUID(str(artifact_id))
    except ValueError:
        return False
    artifact = db.get(JourneyStageArtifactRecord, artifact_uuid)
    if artifact is None or artifact.session_id != record.id or artifact.workspace_id != record.workspace_id:
        return False
    raw_entity = item.source_ref.entity_id or ""
    if raw_entity and raw_entity in artifact.missing_information:
        artifact.missing_information = [entry for entry in artifact.missing_information if entry != raw_entity]
    patch = dict(artifact.user_patch or {})
    resolutions = dict(patch.get("attention_resolutions") or {})
    resolutions[item.key] = {
        "action_kind": payload.action_kind,
        "answer_text": resolution_text,
        "selected_option_key": payload.selected_option_key,
        "was_suggested_answer_used": payload.was_suggested_answer_used,
        "resolution_note": payload.resolution_note,
        "resolved_by_user_id": str(current_user.id),
        "resolved_at": utc_now().isoformat(),
        "source_ref": item.source_ref.model_dump(mode="json"),
    }
    patch["attention_resolutions"] = resolutions
    artifact.user_patch = patch
    artifact.updated_at = utc_now()
    db.add(artifact)

    # Reconciliar con UncertaintyBacklogRecord
    backlogs = db.exec(
        select(UncertaintyBacklogRecord).where(
            UncertaintyBacklogRecord.workspace_id == record.workspace_id,
            UncertaintyBacklogRecord.session_id == record.id,
        )
    ).all()
    for b in backlogs:
        if (
            b.uncertainty_key == item.key
            or b.uncertainty_key == raw_entity
            or (raw_entity and (raw_entity in b.uncertainty_key or b.uncertainty_key in raw_entity))
            or (b.title and b.title.strip().lower() == item.title.strip().lower())
        ):
            if payload.action_kind == "defer":
                b.status = "deferred"
                b.disposition = "defer"
            else:
                b.status = "resolved"
            b.assumed_answer = resolution_text
            b.resolved_at = utc_now()
            b.updated_at = utc_now()
            db.add(b)

    db.flush()
    return True


def _apply_inline_attention_action(
    db: Session,
    *,
    record: SessionRecord,
    item: AttentionItemV2,
    payload: AttentionActionRequestV2,
    current_user: UserRecord,
) -> AttentionActionApplyResult:
    resolution_text = _resolution_text(item, payload)
    if payload.action_kind == "answer" and not resolution_text:
        return AttentionActionApplyResult(status="conflict", message="La respuesta no puede estar vacia.")
    if payload.action_kind == "confirm" and not resolution_text and item.options:
        return AttentionActionApplyResult(status="conflict", message="Selecciona una opcion o agrega una nota de confirmacion.")
    patched = _apply_journey_artifact_attention_resolution(
        db,
        record=record,
        item=item,
        payload=payload,
        current_user=current_user,
        resolution_text=resolution_text,
    )
    if payload.action_kind == "defer":
        message = "Decision diferida y trazada en Attention."
    elif patched:
        message = "Decision aplicada y reconciliada con el artefacto de la etapa."
    else:
        message = "Decision registrada y trazada en Attention."
    return AttentionActionApplyResult(status="applied", message=message)


def _apply_handoff_action(
    db: Session,
    *,
    record: SessionRecord,
    item: AttentionItemV2,
    payload: AttentionActionRequestV2,
) -> AttentionActionApplyResult:
    handoff_id = item.source_ref.entity_id if item.source_ref else None
    handoff: HandoffRecord | None = None
    if handoff_id:
        try:
            uuid_val = UUID(str(handoff_id))
            handoff = db.exec(
                select(HandoffRecord).where(
                    HandoffRecord.session_id == record.id,
                    HandoffRecord.id == uuid_val,
                )
            ).first()
        except (ValueError, TypeError):
            handoff = None
    if handoff is None:
        handoff = db.exec(
            select(HandoffRecord).where(
                HandoffRecord.session_id == record.id,
                HandoffRecord.handoff_key == "governance_review",
            )
        ).first()

    if handoff is None:
        return AttentionActionApplyResult(status="not_found", message="Handoff de gobierno no encontrado.")

    decision = "completed" if payload.action_kind in {"confirm", "approve"} else "returned"
    handoff.status = decision
    handoff.resolution_note = payload.resolution_note or (payload.user_note or "Revisión de gobierno confirmada en Attention.")
    handoff.resolved_at = utc_now()
    handoff.updated_at = utc_now()
    db.add(handoff)

    # Reconciliar compuertas de aprobación si las hubiere
    approvals = db.exec(
        select(ApprovalGateRecord).where(
            ApprovalGateRecord.session_id == record.id,
            ApprovalGateRecord.status == ApprovalStatus.pending,
        )
    ).all()
    for app_gate in approvals:
        app_gate.status = ApprovalStatus.approved
        app_gate.resolved_at = utc_now()
        app_gate.resolution_note = "Aprobado junto con la revisión de gobierno."
        db.add(app_gate)

    # Actualizar readiness_state del Blueprint a complete
    bp_record = db.exec(
        select(BlueprintRecord).where(
            BlueprintRecord.session_id == record.id,
        ).order_by(BlueprintRecord.updated_at.desc())
    ).first()
    if bp_record is not None:
        bp_record.readiness_state = ReviewState.complete
        bp_record.updated_at = utc_now()
        db.add(bp_record)

    db.commit()
    return AttentionActionApplyResult(status="applied", message="Revisión de gobierno confirmada. Bloqueadores de promoción resueltos.")


def apply_attention_action_v2(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
    readiness: ConstructionReadinessReport,
    access: CommercialAccessSnapshotV2,
    current_user: UserRecord,
    item_key: str,
    payload: AttentionActionRequestV2,
) -> AttentionActionApplyResult:
    existing = _find_idempotency_event(db, record=record, current_user=current_user, idempotency_key=payload.idempotency_key)
    if existing is not None:
        return AttentionActionApplyResult(status="duplicate", message="Accion idempotente ya procesada.")
    item = _find_attention_item_for_action(
        db,
        record=record,
        snapshot=snapshot,
        readiness=readiness,
        access=access,
        item_key=item_key,
    )
    if item is None:
        return AttentionActionApplyResult(status="not_found", message="Attention item no encontrado.")
    if payload.action_kind == "answer" and item.key in _answered_attention_item_keys(db, record):
        return AttentionActionApplyResult(status="duplicate", message="Pregunta ya respondida en Attention.")
    if payload.source_artifact_version is not None and item.source_ref.artifact_version != payload.source_artifact_version:
        return AttentionActionApplyResult(status="conflict", message="La version del artefacto cambio; refresca antes de resolver.")

    if item.source == "acp_questions" and payload.action_kind == "answer":
        result = _apply_acp_question_answer(db, record=record, item=item, payload=payload, current_user=current_user)
    elif item.source == "approval_gate" and payload.action_kind in {"approve", "reject"}:
        result = _apply_approval_action(db, record=record, item=item, payload=payload)
    elif item.source == "governance_handoff" and payload.action_kind in {"confirm", "approve", "reject"}:
        result = _apply_handoff_action(db, record=record, item=item, payload=payload)
    elif item.source == "access_request" and payload.action_kind in {"approve", "reject", "confirm"}:
        result = _apply_access_request_action(db, record=record, item=item, payload=payload, current_user=current_user)
    elif payload.action_kind in {"answer", "confirm", "defer"} and item.action.can_resolve_inline:
        result = _apply_inline_attention_action(
            db,
            record=record,
            item=item,
            payload=payload,
            current_user=current_user,
        )
    elif payload.action_kind == "retry" and item.source == "runtime_operation":
        result = AttentionActionApplyResult(
            status="applied",
            message="Solicitud de reintento registrada para recuperar la operacion tecnica.",
            resume_eligible=True,
        )
    elif payload.action_kind == "retry" and item.source == "product_build_step":
        result = _apply_product_build_step_retry(db, record=record, item=item)
    elif payload.action_kind == "regenerate" and item.type == "stale":
        result = AttentionActionApplyResult(
            status="applied",
            message="Solicitud de regeneracion registrada. La vista de etapa debe iniciar el reproceso seguro.",
            )
    elif payload.action_kind in {"navigate", "retry", "regenerate"}:
        result = AttentionActionApplyResult(
            status="unsupported",
            message="Esta accion ya esta modelada para navegacion, pero todavia no tiene mutacion de dominio segura.",
        )
    else:
        result = AttentionActionApplyResult(status="unsupported", message="Accion no soportada para este item.")

    if (
        result.status == "applied"
        and item.source.startswith("journey.")
        and payload.action_kind in {"answer", "confirm"}
    ):
        result = replace(result, resume_eligible=True)

    if result.status in {"applied", "unsupported", "conflict", "forbidden"}:
        _record_attention_action_event(
            db,
            record=record,
            current_user=current_user,
            item=item,
            payload=payload,
            result_status=result.status,
            message=result.message,
        )
    return result


def _apply_product_build_step_retry(
    db: Session,
    *,
    record: SessionRecord,
    item: AttentionItemV2,
) -> AttentionActionApplyResult:
    try:
        step_id = UUID(str(item.source_ref.entity_id))
    except (TypeError, ValueError):
        return AttentionActionApplyResult(
            status="conflict",
            message="No se pudo identificar el step de producto asociado al reintento.",
        )
    step = db.get(ProductBuildStepRecord, step_id)
    if step is None or step.workspace_id != record.workspace_id or step.session_id != record.id:
        return AttentionActionApplyResult(status="not_found", message="Step de producto no encontrado.")
    run = db.get(ProductBuildRunRecord, step.run_id)
    if run is None:
        return AttentionActionApplyResult(status="not_found", message="Run de producto no encontrado.")
    step.status = "queued"
    step.progress_percent = 0
    step.error_payload = {}
    step.updated_at = utc_now()
    db.add(step)
    update_product_build_run_state(
        db,
        run=run,
        lifecycle="queued",
        error_payload={},
        checkpoint_payload={
            **(run.checkpoint_payload or {}),
            "resume_requested_from": "attention_v2",
            "resume_step_key": step.step_key,
            "resume_requested_at": step.updated_at.isoformat(),
        },
    )
    return AttentionActionApplyResult(
        status="applied",
        message="Reintento registrado. El producto puede reanudar este step desde el checkpoint disponible.",
        resume_eligible=True,
    )
