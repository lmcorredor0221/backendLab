from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CommercialTier,
    JourneyStageArtifactRecord,
    SessionRecord,
    UserRecord,
    utc_now,
)
from app.services.attention.validation_issue_normalizer import (
    _issue_copy,
    _split_issue,
    split_validation_issue_codes,
)
from app.services.commerce_service import record_commercial_event
from app.services.deliverable_catalog.contracts import DeliverableGenerationTask
from app.services.deliverable_catalog.dependency_service import (
    invalidate_deliverables_for_change,
    resolve_regeneration_scope,
)
from app.services.deliverable_catalog.generation_service import run_deliverable_generation_task
from app.services.product_processing.backlog_service import (
    backlog_entry_from_record,
    prioritize_uncertainty_backlog,
    upsert_uncertainty_backlog,
)
from app.services.product_processing.policy import (
    classify_uncertainty_for_profile,
    resolve_product_processing_mode,
)
from app.services.product_processing.contracts import (
    PremiumEnrichmentItem,
    PremiumEnrichmentWorkspace,
    PremiumSelectiveReprocessResult,
    PremiumUncertaintyResolutionRequest,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductProcessingMode,
    UncertaintyBacklogStatus,
    UncertaintyDisposition,
)
from app.services.product_processing.persistence import UncertaintyBacklogRecord
from app.services.product_processing.product_build_orchestrator import (
    ProductBuildOrchestrationOptions,
    ensure_product_build_orchestration,
)
from app.services.product_processing.product_build_run_service import (
    list_product_build_runs,
    list_product_build_steps,
    update_product_build_run_state,
    upsert_product_build_step,
)
from app.services.product_processing.product_build_status_service import build_product_build_status


STAGE_DEPENDENCY_KEY = {
    "discover": "session.discovery",
    "define": "definition.requirements",
    "design": "design.architecture",
    "tools": "tools.minimum_set",
    "memory": "memory.strategy",
    "estimate": "estimate.analysis",
    "validate": "validation.scenarios",
    "package": "package.manifest",
}
STAGE_FLOW_ORDER = tuple(STAGE_DEPENDENCY_KEY.keys())
COMPLETED_STEP_STATES = {"available", "completed", "skipped"}
ACTIVE_STEP_STATES = {"queued", "generating", "running"}
PREMIUM_BACKLOG_PRODUCT_MODES = {
    ProductProcessingMode.basic_free.value,
    ProductProcessingMode.premium_enrichment.value,
}
ACP_DEFER_TARGETS = {"acp", "package", "implementation", "implementacion", "deployment", "despliegue"}
TIER_RANK = {
    CommercialTier.blueprint: 0,
    CommercialTier.blueprint_pro: 1,
    CommercialTier.acp: 2,
}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _dependency_keys_for_entry(record: UncertaintyBacklogRecord) -> list[str]:
    return _dedupe(
        [
            *(record.dependency_keys or []),
            *(record.affected_deliverable_keys or []),
            STAGE_DEPENDENCY_KEY.get(record.source_stage, record.source_stage),
        ]
    )


def _priority_score(entry) -> float:
    status_weight = {
        UncertaintyBacklogStatus.open: 24,
        UncertaintyBacklogStatus.in_progress: 18,
        UncertaintyBacklogStatus.deferred: 10,
        UncertaintyBacklogStatus.resolved: 0,
        UncertaintyBacklogStatus.dismissed: 0,
        UncertaintyBacklogStatus.superseded: 0,
    }.get(entry.status, 6)
    disposition_weight = {
        UncertaintyDisposition.resolve_now: 22,
        UncertaintyDisposition.block: 24,
        UncertaintyDisposition.defer: 8,
        UncertaintyDisposition.infer: 4,
    }.get(entry.disposition, 4)
    kind_weight = 10 if entry.kind.value in {"gap", "decision", "hitl"} else 6
    affected_weight = min(len(entry.affected_deliverable_keys) * 4, 16)
    confidence_weight = round(entry.confidence * 24, 2)
    return min(100, round(status_weight + disposition_weight + kind_weight + affected_weight + confidence_weight, 2))


def _priority_reason(item: PremiumEnrichmentItem) -> str:
    entry = item.entry
    if entry.status == UncertaintyBacklogStatus.deferred:
        return "Oportunidad detectada en Basico; Premium puede resolverla sin frenar el flujo."
    if item.ordered_regeneration_keys:
        return "Tiene impacto trazable sobre entregables versionables y permite reproceso selectivo."
    return "Aporta claridad al Blueprint Premium sin exigir reprocesar todo el proyecto."


def _resolve_reprocess_decision(
    ordered_regeneration_keys: list[str],
) -> tuple[str, bool, str, str]:
    total = len(ordered_regeneration_keys)
    if total == 0:
        return (
            "document_only",
            False,
            "document_only",
            "La respuesta se registra como decision trazable sin reprocesar entregables porque no se detecto impacto material.",
        )
    if total <= 3:
        return (
            "localized_reprocess",
            True,
            "review_and_apply_localized_reprocess",
            f"La respuesta impacta {total} entregable(s) versionado(s) y se puede actualizar de forma localizada.",
        )
    return (
        "structural_reprocess",
        True,
        "review_and_apply_structural_reprocess",
        f"La respuesta impacta {total} entregable(s) y conviene reprocesar la cadena dependiente del Blueprint Pro.",
    )


def _should_execute_reprocess(
    payload: PremiumUncertaintyResolutionRequest,
    *,
    material_impact: bool,
) -> bool:
    if not material_impact:
        return False
    if payload.execution_mode == "apply_reprocess":
        return True
    return bool(payload.regenerate)


def _item_from_record(record: UncertaintyBacklogRecord) -> PremiumEnrichmentItem:
    entry = backlog_entry_from_record(record)
    changed = _dependency_keys_for_entry(record)
    scope = resolve_regeneration_scope(changed_dependency_keys=changed)
    affected = _dedupe([*entry.affected_deliverable_keys, *scope.affected_deliverable_keys])
    item = PremiumEnrichmentItem(
        entry=entry,
        priority_score=_priority_score(entry),
        changed_dependency_keys=changed,
        affected_deliverable_keys=affected,
        ordered_regeneration_keys=scope.ordered_regeneration_keys,
        unaffected_deliverable_count=len(scope.unaffected_deliverable_keys),
    )
    return item.model_copy(update={"priority_reason": _priority_reason(item)})


def sync_stage_artifacts_into_uncertainty_backlog(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    current_tier: CommercialTier,
) -> None:
    """Sincroniza automaticamente cualquier GAP, pregunta abierta o issue de validacion de los artefactos de la sesion."""
    mode = resolve_product_processing_mode(current_tier, premium_enrichment=True)
    artifacts = db.exec(
        select(JourneyStageArtifactRecord).where(
            JourneyStageArtifactRecord.workspace_id == workspace_id,
            JourneyStageArtifactRecord.session_id == session_id,
        )
    ).all()

    for artifact in artifacts:
        stage = artifact.stage_key or "discover"
        payload = artifact.proposal_payload or {}
        raw_items: list[dict[str, Any]] = []

        # 1. Validation issues and plain missing info
        validation_issues, missing_info = split_validation_issue_codes(artifact.missing_information or [])
        for issue in validation_issues:
            prefix, detail = _split_issue(issue)
            copy = _issue_copy(prefix, detail)
            raw_items.append(
                {
                    "key": issue,
                    "kind": copy.get("type", "validation"),
                    "title": copy.get("title", issue),
                    "question": copy.get("title", issue),
                    "reason": copy.get("reason", ""),
                    "impact": copy.get("impact", ""),
                    "suggested_answer": copy.get("suggested", ""),
                    "severity": copy.get("severity", "warning"),
                    "source_refs": [f"journey.{artifact.artifact_kind or stage}.validation"],
                    "affected_deliverable_keys": [STAGE_DEPENDENCY_KEY.get(stage, stage)],
                }
            )

        for plain in missing_info:
            if not plain:
                continue
            raw_items.append(
                {
                    "key": f"{stage}:missing_info:{plain[:40]}",
                    "kind": "gap",
                    "title": f"Completar informacion: {plain[:80]}",
                    "question": f"Informacion faltante: {plain}",
                    "reason": f"Falta informacion requerida en la etapa {stage}.",
                    "impact": "Mejora la precision del Blueprint.",
                    "suggested_answer": "Completar la informacion o diferir a ACP.",
                    "severity": "warning",
                    "source_refs": [f"journey.{stage}.missing_information"],
                    "affected_deliverable_keys": [STAGE_DEPENDENCY_KEY.get(stage, stage)],
                }
            )

        # 2. Open questions from proposal_payload
        for q_key in ("open_questions", "guided_questions", "needs_information"):
            questions = payload.get(q_key) or []
            if isinstance(questions, list):
                for index, q in enumerate(questions, start=1):
                    if isinstance(q, dict):
                        q_text = str(q.get("question") or q.get("question_text") or q.get("title") or "").strip()
                        q_k = str(q.get("key") or q.get("question_key") or f"{stage}_{q_key}_{index}")
                        q_sug = str(q.get("suggested_answer") or "").strip()
                        q_reas = str(q.get("reason") or q.get("rationale") or "").strip()
                    else:
                        q_text = str(q or "").strip()
                        q_k = f"{stage}_{q_key}_{index}"
                        q_sug = ""
                        q_reas = ""
                    if not q_text:
                        continue
                    raw_items.append(
                        {
                            "key": q_k,
                            "kind": "question",
                            "title": q_text,
                            "question": q_text,
                            "reason": q_reas or f"Pregunta abierta de la etapa {stage}.",
                            "impact": "Aporta definicion funcional.",
                            "suggested_answer": q_sug,
                            "severity": "warning",
                            "source_refs": [f"journey.{stage}.{q_key}"],
                            "affected_deliverable_keys": [STAGE_DEPENDENCY_KEY.get(stage, stage)],
                        }
                    )

        # 3. Gaps from proposal_payload
        for g_key in ("gaps", "coverage_gaps"):
            gaps = payload.get(g_key) or []
            if isinstance(gaps, list):
                for index, g in enumerate(gaps, start=1):
                    if isinstance(g, dict):
                        g_text = str(g.get("title") or g.get("summary") or g.get("detail") or "").strip()
                        g_k = str(g.get("key") or g.get("gap_key") or f"{stage}_{g_key}_{index}")
                    else:
                        g_text = str(g or "").strip()
                        g_k = f"{stage}_{g_key}_{index}"
                    if not g_text:
                        continue
                    raw_items.append(
                        {
                            "key": g_k,
                            "kind": "gap",
                            "title": f"GAP en {stage}: {g_text[:80]}",
                            "question": g_text,
                            "reason": f"Brecha identificada en la etapa {stage}.",
                            "impact": "Mejora la cobertura del Blueprint.",
                            "suggested_answer": "Resolver brecha o diferir a ACP.",
                            "severity": "warning",
                            "source_refs": [f"journey.{stage}.{g_key}"],
                            "affected_deliverable_keys": [STAGE_DEPENDENCY_KEY.get(stage, stage)],
                        }
                    )

        for item in raw_items:
            # Filtrar ruido tecnico de bajo nivel (infraestructura/embeddings/llm_policy/APIs) que debe ser inferido o ir directo a ACP
            item_text = f"{item.get('key', '')} {item.get('title', '')} {item.get('question', '')}".lower()
            if any(
                kw in item_text
                for kw in (
                    "embedding_policy",
                    "llm_policy",
                    "dimensiones y versión",
                    "dimensiones y version",
                    "knowledge_owner_pending",
                    "human_handoff",
                    "proveedor, modelo, dimensiones",
                    "modelos base y proveedor",
                    "propietario de fuentes",
                    "calibración matemática",
                    "calibracion matematica",
                    "datos históricos",
                    "datos historicos",
                    "mecanismos de integración",
                    "mecanismos de integracion",
                    "design_decisions",
                    "tooling_principles",
                    "memory_strategy",
                    "benchmark de latencia",
                    "filtro de sanitización",
                    "filtro de sanitizacion",
                    "volumen cuantitativo",
                    "taxonomía completa",
                    "taxonomia completa",
                    "system_of_record_unspecified",
                    "sistema fuente no identificado",
                    "openapi",
                    "swagger",
                )
            ):
                continue

            try:
                classification = classify_uncertainty_for_profile(stage, item, mode)
                upsert_uncertainty_backlog(
                    db,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    classification=classification,
                    dependency_keys=[STAGE_DEPENDENCY_KEY.get(stage, stage)],
                    created_from="artifact_sync",
                )
            except Exception:
                continue


def build_premium_enrichment_workspace(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    current_tier: CommercialTier,
    selectable_limit: int = 6,
    current_user: UserRecord | None = None,
) -> PremiumEnrichmentWorkspace:
    sync_stage_artifacts_into_uncertainty_backlog(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        current_tier=current_tier,
    )
    statement = select(UncertaintyBacklogRecord).where(
        UncertaintyBacklogRecord.workspace_id == workspace_id,
        UncertaintyBacklogRecord.session_id == session_id,
    )
    records = db.exec(statement).all()
    open_entries = [
        backlog_entry_from_record(record)
        for record in records
        if record.status not in {
            UncertaintyBacklogStatus.resolved.value,
            UncertaintyBacklogStatus.deferred.value,
            UncertaintyBacklogStatus.dismissed.value,
            UncertaintyBacklogStatus.superseded.value,
        }
        and getattr(record, "disposition", "") != "defer"
    ]
    prioritized_ids = {
        entry.id
        for entry in prioritize_uncertainty_backlog(open_entries)
    }
    open_items = [_item_from_record(record) for record in records if str(record.id) in prioritized_ids]
    open_items.sort(key=lambda item: (-item.priority_score, item.entry.source_stage, item.entry.uncertainty_key))
    limited_open_items = open_items[:selectable_limit]

    deferred_records = [
        record
        for record in records
        if (record.status == UncertaintyBacklogStatus.deferred.value or getattr(record, "disposition", "") == "defer")
        and record.status
        not in {
            UncertaintyBacklogStatus.resolved.value,
            UncertaintyBacklogStatus.dismissed.value,
            UncertaintyBacklogStatus.superseded.value,
        }
    ]
    deferred_items = [_item_from_record(record) for record in deferred_records]

    resolved_records = [
        record
        for record in records
        if record.status == UncertaintyBacklogStatus.resolved.value
        and record.status != UncertaintyBacklogStatus.dismissed.value
    ]
    resolved_items = [_item_from_record(record) for record in resolved_records]

    all_items = [*limited_open_items, *deferred_items, *resolved_items]

    workspace = PremiumEnrichmentWorkspace(
        workspace_id=workspace_id,
        session_id=session_id,
        current_tier=current_tier,
        selectable_limit=selectable_limit,
        total_uncertainties=len(records),
        prioritized_count=len(open_items),
        deferred_count=len(deferred_items),
        resolved_count=len(resolved_items),
        items=all_items,
        value_summary=(
            "Premium usa las incertidumbres del Blueprint Basico como backlog de enriquecimiento, "
            "prioriza por impacto y evita convertir el producto en un cuestionario largo."
        ),
        processing_guidance=(
            "Resolver una pregunta recalcula dependencias, marca entregables obsoletos y regenera "
            "solo el subconjunto afectado."
        ),
    )
    sync_premium_enrichment_product_run(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        current_tier=current_tier,
        current_user=current_user,
        source="premium_workspace",
    )
    return workspace


def sync_premium_enrichment_product_run(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    current_tier: CommercialTier,
    current_user: UserRecord | None = None,
    source: str = "premium_enrichment",
    current_stage: str | None = None,
    auto_execute_when_ready: bool = False,
    allow_llm: bool = False,
) -> ProductBuildStatus | None:
    if _tier_rank(current_tier) < _tier_rank(CommercialTier.blueprint_pro):
        return None
    record = db.get(SessionRecord, session_id)
    if record is None or record.workspace_id != workspace_id:
        return None

    effective_stage = str(current_stage or getattr(record.current_stage, "value", str(record.current_stage or "discover")))

    status = ensure_product_build_orchestration(
        db,
        record=record,
        product_key=ProductBuildProductKey.blueprint_pro,
        current_user=current_user,
        options=ProductBuildOrchestrationOptions(
            current_stage=effective_stage,
            activation_payload={
                "source": source,
                "workspace_id": str(workspace_id),
                "session_id": str(session_id),
            },
        ),
    )
    if auto_execute_when_ready and _should_auto_execute_premium_build(status, effective_stage):
        status = ensure_product_build_orchestration(
            db,
            record=record,
            product_key=ProductBuildProductKey.blueprint_pro,
            current_user=current_user,
            options=ProductBuildOrchestrationOptions(
                current_stage=effective_stage,
                execute_jobs=True,
                allow_llm=allow_llm,
                activation_payload={
                    "source": source,
                    "workspace_id": str(workspace_id),
                    "session_id": str(session_id),
                    "auto_execute": True,
                },
            ),
            catalog_stage_override=effective_stage,
        )
    runs = list_product_build_runs(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        product_key=ProductBuildProductKey.blueprint_pro,
    )
    if not runs:
        return
    run = runs[0]
    records = _premium_backlog_records(db, workspace_id=workspace_id, session_id=session_id)
    for index, backlog_record in enumerate(records, start=10_000):
        step_status = _premium_backlog_step_status(backlog_record)
        upsert_product_build_step(
            db,
            run=run,
            step_key=f"premium_backlog:{backlog_record.id}",
            status=step_status,
            stage_key=backlog_record.source_stage,
            dependency_key=",".join(_dependency_keys_for_entry(backlog_record)),
            sequence=index,
            progress_percent=_premium_backlog_progress(step_status),
            checkpoint_payload={
                "title": backlog_record.title,
                "uncertainty_key": backlog_record.uncertainty_key,
                "product_mode": backlog_record.product_mode,
                "disposition": backlog_record.disposition,
                "status": backlog_record.status,
                "target_stage": backlog_record.target_stage,
                "affected_deliverable_keys": list(backlog_record.affected_deliverable_keys or []),
                "source": source,
            },
            error_payload=_premium_backlog_error_payload(backlog_record, step_status),
    )
    _finalize_premium_run_from_all_steps(db, run=run, backlog_records=records)
    return build_product_build_status(
        db,
        record=record,
        product_key=ProductBuildProductKey.blueprint_pro,
        current_user=current_user,
        catalog_stage_override=effective_stage,
    )


def _normalize_stage_key(value: str | None) -> str:
    stage = str(value or "").strip().lower()
    if stage in STAGE_FLOW_ORDER:
        return stage
    legacy_map = {
        "draft_capture": "discover",
        "input_validation": "discover",
        "normalize_discovery": "discover",
        "build_canvas": "define",
        "build_blueprint": "design",
        "post_validation": "validate",
        "ready_for_export": "package",
    }
    return legacy_map.get(stage, "discover")


def _stage_index(stage_key: str | None) -> int:
    normalized = _normalize_stage_key(stage_key)
    try:
        return STAGE_FLOW_ORDER.index(normalized)
    except ValueError:
        return 0


def _should_auto_execute_premium_build(
    status: ProductBuildStatus | None,
    current_stage: str | None = None,
) -> bool:
    if status is None or status.entitlement.access_state != "allowed":
        return False
    current_stage_idx = _stage_index(current_stage)
    if any(
        item.blocking and _stage_index(item.stage_key or current_stage) <= current_stage_idx
        for item in status.attention.items
    ):
        return False
    relevant_deliverables = [
        item for item in status.deliverables if _stage_index(item.stage_key) <= current_stage_idx
    ]
    if not relevant_deliverables:
        relevant_deliverables = list(status.deliverables)
    if any(getattr(item.state, "value", str(item.state)) in {"queued", "generating"} for item in relevant_deliverables):
        return False
    return any(getattr(item.state, "value", str(item.state)) in {"pending", "stale"} for item in relevant_deliverables)


def _resolved_answer(record: UncertaintyBacklogRecord, payload: PremiumUncertaintyResolutionRequest) -> str:
    explicit = str(payload.answer or "").strip()
    if explicit:
        return explicit
    selected = str(payload.selected_option_key or "").strip()
    if not selected:
        return str(record.suggested_answer or "").strip()
    for option in record.answer_options or []:
        if str(option.get("key") or "") == selected:
            label = str(option.get("label") or "").strip()
            description = str(option.get("description") or "").strip()
            return " - ".join([value for value in [label, description] if value])
    return selected


def resolve_premium_uncertainty(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    backlog_id: UUID,
    actor_user_id: UUID,
    payload: PremiumUncertaintyResolutionRequest,
) -> PremiumSelectiveReprocessResult:
    record = db.get(UncertaintyBacklogRecord, backlog_id)
    if record is None or record.workspace_id != workspace_id or record.session_id != session_id:
        raise LookupError("Premium enrichment item not found")

    changed_dependency_keys = _dependency_keys_for_entry(record)
    source_key = (record.affected_deliverable_keys or [""])[0]
    stale_report = invalidate_deliverables_for_change(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        changed_dependency_keys=changed_dependency_keys,
        source_deliverable_key=source_key,
    )
    scope = resolve_regeneration_scope(
        changed_dependency_keys=changed_dependency_keys,
        source_deliverable_key=source_key,
    )
    reprocess_decision, material_impact, recommended_action, impact_summary = _resolve_reprocess_decision(
        scope.ordered_regeneration_keys
    )
    answer = _resolved_answer(record, payload)
    record.status = UncertaintyBacklogStatus.resolved.value
    record.assumed_answer = answer
    record.resolved_at = utc_now()
    record.updated_at = utc_now()
    record.payload = {
        **(record.payload or {}),
        "premium_resolution": {
            "answer": answer,
            "selected_option_key": payload.selected_option_key,
            "actor_user_id": str(actor_user_id),
            "changed_dependency_keys": changed_dependency_keys,
            "ordered_regeneration_keys": scope.ordered_regeneration_keys,
            "reprocess_decision": reprocess_decision,
            "recommended_action": recommended_action,
            "impact_summary": impact_summary,
        },
    }
    db.add(record)
    db.flush()

    regenerated: list[str] = []
    job_ids: list[str] = []
    status_by_key_map: dict[str, str] = {}
    fifo_queue = scope.ordered_regeneration_keys[: payload.max_deliverables]
    execute_reprocess = _should_execute_reprocess(payload, material_impact=material_impact)
    queue_total = len(fifo_queue) if execute_reprocess else 0
    queue_completed = 0

    if execute_reprocess:
        # Procesamiento secuencial garantizado en Cola FIFO (First In, First Out)
        for deliverable_key in fifo_queue:
            try:
                job, _ = run_deliverable_generation_task(
                    db,
                    DeliverableGenerationTask(
                        workspace_id=workspace_id,
                        session_id=session_id,
                        deliverable_key=deliverable_key,
                        product_mode=ProductProcessingMode.premium_enrichment.value,
                        current_stage=record.source_stage,
                        tier=CommercialTier.blueprint_pro,
                        idempotency_key=f"premium:{session_id}:{backlog_id}:{deliverable_key}",
                        requested_by_user_id=actor_user_id,
                        context_payload={
                            "summary": record.title,
                            "resolved_answer": answer,
                            "reason": record.reason,
                            "impact": record.impact,
                        },
                        approved_context_refs=changed_dependency_keys,
                        allow_llm=True,
                        max_iterations=5,
                    ),
                )
                regenerated.append(deliverable_key)
                job_ids.append(str(job.id))
                status_by_key_map[deliverable_key] = job.status
                queue_completed += 1
            except (LookupError, PermissionError, ValueError) as exc:
                status_by_key_map[deliverable_key] = f"skipped:{exc}"

    resolved_entry = backlog_entry_from_record(record)
    sync_premium_enrichment_product_run(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        current_tier=CommercialTier.blueprint_pro,
        source="premium_resolution",
    )
    _sync_attention_resolution(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        record=record,
        actor_user_id=actor_user_id,
        action_kind="answer",
        answer_text=answer,
    )
    return PremiumSelectiveReprocessResult(
        resolved_entry=resolved_entry,
        changed_dependency_keys=changed_dependency_keys,
        stale_deliverable_keys=stale_report.stale_deliverable_keys,
        ordered_regeneration_keys=scope.ordered_regeneration_keys,
        regenerated_deliverable_keys=regenerated,
        preserved_deliverable_keys=scope.unaffected_deliverable_keys,
        material_impact=material_impact,
        reprocess_decision=reprocess_decision,
        recommended_action=recommended_action,
        impact_summary=impact_summary,
        generation_job_ids=job_ids,
        generation_status_by_deliverable=status_by_key_map,
        superseded_uncertainty_count=stale_report.superseded_uncertainty_count,
        comparison_summary=(
            f"Se resolvio '{resolved_entry.title}'. "
            + (
                f"{len(regenerated)} entregable(s) fueron reprocesados en cola FIFO y "
                f"{len(scope.unaffected_deliverable_keys)} conservaron su version."
                if regenerated
                else impact_summary
            )
        ),
        queue_total=queue_total,
        queue_completed=queue_completed,
        queue_status=(
            "completed"
            if execute_reprocess and queue_completed == queue_total
            else "processing"
            if execute_reprocess
            else "not_requested"
        ),
        queue_processed_keys=regenerated,
    )


def _sync_attention_resolution(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    record: UncertaintyBacklogRecord,
    actor_user_id: UUID,
    action_kind: str,
    answer_text: str = "",
) -> None:
    """Sincroniza la resolucion, diferimiento o descarte para que el item salga inmediatamente del panel de atencion."""
    record_commercial_event(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        user_id=actor_user_id,
        event_key="attention_action_v2",
        product_key=record.product_mode or "blueprint_pro",
        source="premium_enrichment_resolution",
        correlation_id=f"enrichment:{record.id}",
        metadata={
            "action_kind": action_kind,
            "item_key": record.uncertainty_key,
            "result_status": "applied",
            "resolved_backlog_id": str(record.id),
            "title": record.title,
            "answer_text": answer_text,
            "source_stage": record.source_stage,
            "resolved_at": utc_now().isoformat(),
        },
    )

    if record.source_stage:
        artifacts = db.exec(
            select(JourneyStageArtifactRecord).where(
                JourneyStageArtifactRecord.workspace_id == workspace_id,
                JourneyStageArtifactRecord.session_id == session_id,
                JourneyStageArtifactRecord.stage_key == record.source_stage,
            )
        ).all()
        for artifact in artifacts:
            targets_to_remove = {
                record.uncertainty_key,
                record.title,
                *(record.source_refs or []),
            }
            if ":" in record.uncertainty_key:
                targets_to_remove.add(record.uncertainty_key.split(":")[-1])
            
            artifact.missing_information = [
                entry for entry in artifact.missing_information if entry not in targets_to_remove
            ]
            patch = dict(artifact.user_patch or {})
            resolutions = dict(patch.get("attention_resolutions") or {})
            resolutions[record.uncertainty_key] = {
                "action_kind": action_kind,
                "answer_text": answer_text or record.assumed_answer,
                "resolved_by_user_id": str(actor_user_id),
                "resolved_at": utc_now().isoformat(),
                "backlog_id": str(record.id),
            }
            patch["attention_resolutions"] = resolutions
            artifact.user_patch = patch
            artifact.updated_at = utc_now()
            db.add(artifact)
        db.flush()


def defer_premium_uncertainty_to_acp(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    backlog_id: UUID,
    actor_user_id: UUID,
) -> UncertaintyBacklogRecord:
    """Difiere una incertidumbre, duda o GAP directamente al Agent Construction Package (ACP)."""
    record = db.get(UncertaintyBacklogRecord, backlog_id)
    if record is None:
        record = db.exec(
            select(UncertaintyBacklogRecord).where(
                UncertaintyBacklogRecord.workspace_id == workspace_id,
                UncertaintyBacklogRecord.session_id == session_id,
                (UncertaintyBacklogRecord.id == backlog_id) | (UncertaintyBacklogRecord.uncertainty_key == str(backlog_id)),
            )
        ).first()

    if record is None or record.workspace_id != workspace_id or record.session_id != session_id:
        raise LookupError("Premium enrichment item not found")

    record.disposition = UncertaintyDisposition.defer.value
    record.target_stage = "acp"
    record.status = UncertaintyBacklogStatus.deferred.value
    record.updated_at = utc_now()
    current_payload = dict(record.payload or {})
    current_payload["deferred_to_acp"] = True
    current_payload["deferred_by_user_id"] = str(actor_user_id)
    current_payload["deferred_at"] = utc_now().isoformat()
    record.payload = current_payload
    db.add(record)
    db.flush()

    sync_premium_enrichment_product_run(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        current_tier=CommercialTier.blueprint_pro,
        source="defer_to_acp",
    )
    _sync_attention_resolution(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        record=record,
        actor_user_id=actor_user_id,
        action_kind="defer",
        answer_text="Diferido a ACP",
    )
    return record


def dismiss_premium_uncertainty(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    backlog_id: UUID,
    actor_user_id: UUID,
) -> UncertaintyBacklogRecord:
    """Descarta definitivamente una incertidumbre del sistema para que no vuelva a aparecer."""
    record = db.get(UncertaintyBacklogRecord, backlog_id)
    if record is None:
        record = db.exec(
            select(UncertaintyBacklogRecord).where(
                UncertaintyBacklogRecord.workspace_id == workspace_id,
                UncertaintyBacklogRecord.session_id == session_id,
                (UncertaintyBacklogRecord.id == backlog_id) | (UncertaintyBacklogRecord.uncertainty_key == str(backlog_id)),
            )
        ).first()

    if record is None or record.workspace_id != workspace_id or record.session_id != session_id:
        raise LookupError("Premium enrichment item not found")

    record.status = UncertaintyBacklogStatus.dismissed.value
    record.updated_at = utc_now()
    current_payload = dict(record.payload or {})
    current_payload["dismissed"] = True
    current_payload["dismissed_by_user_id"] = str(actor_user_id)
    current_payload["dismissed_at"] = utc_now().isoformat()
    record.payload = current_payload
    db.add(record)
    db.flush()

    sync_premium_enrichment_product_run(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        current_tier=CommercialTier.blueprint_pro,
        source="dismiss_item",
    )
    _sync_attention_resolution(
        db,
        workspace_id=workspace_id,
        session_id=session_id,
        record=record,
        actor_user_id=actor_user_id,
        action_kind="dismiss",
        answer_text="Descartado de enriquecimiento",
    )
    return record


def _tier_rank(tier: CommercialTier | str) -> int:
    normalized = tier if isinstance(tier, CommercialTier) else CommercialTier(str(tier))
    return TIER_RANK.get(normalized, 0)


def _premium_backlog_records(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
) -> list[UncertaintyBacklogRecord]:
    return list(
        db.exec(
            select(UncertaintyBacklogRecord)
            .where(
                UncertaintyBacklogRecord.workspace_id == workspace_id,
                UncertaintyBacklogRecord.session_id == session_id,
                UncertaintyBacklogRecord.product_mode.in_(list(PREMIUM_BACKLOG_PRODUCT_MODES)),
            )
            .order_by(UncertaintyBacklogRecord.source_stage.asc(), UncertaintyBacklogRecord.updated_at.desc())
        ).all()
    )


def _premium_backlog_step_status(record: UncertaintyBacklogRecord) -> str:
    if record.status == UncertaintyBacklogStatus.resolved.value:
        return "completed"
    if record.status in {UncertaintyBacklogStatus.dismissed.value, UncertaintyBacklogStatus.superseded.value}:
        return "skipped"
    if _is_deferred_to_acp(record) or record.disposition == UncertaintyDisposition.infer.value:
        return "skipped"
    if record.status in {UncertaintyBacklogStatus.open.value, UncertaintyBacklogStatus.in_progress.value, UncertaintyBacklogStatus.deferred.value}:
        return "requires_attention"
    return "pending"


def _is_deferred_to_acp(record: UncertaintyBacklogRecord) -> bool:
    target = str(record.target_stage or "").strip().lower()
    return record.disposition == UncertaintyDisposition.defer.value and target in ACP_DEFER_TARGETS


def _premium_backlog_progress(status: str) -> int:
    if status in COMPLETED_STEP_STATES:
        return 100
    if status in ACTIVE_STEP_STATES:
        return 25
    return 0


def _premium_backlog_error_payload(record: UncertaintyBacklogRecord, status: str) -> dict:
    if status != "requires_attention":
        return {}
    return {
        "code": "premium_uncertainty_requires_attention",
        "title": record.title or record.uncertainty_key,
        "message": record.reason or record.description or "Resolver esta incertidumbre permite enriquecer el Blueprint Pro.",
        "uncertainty_id": str(record.id),
        "uncertainty_key": record.uncertainty_key,
    }


def _finalize_premium_run_from_all_steps(
    db: Session,
    *,
    run,
    backlog_records: list[UncertaintyBacklogRecord],
) -> None:
    steps = list_product_build_steps(db, run_id=run.id)
    total_units = float(len(steps))
    completed_units = float(sum(1 for step in steps if step.status in COMPLETED_STEP_STATES))
    blocked_units = float(sum(1 for step in steps if step.status in {"error", "requires_attention", "locked"}))
    active_units = sum(1 for step in steps if step.status in ACTIVE_STEP_STATES)
    if blocked_units > 0:
        lifecycle = ProductBuildLifecycle.requires_attention
    elif total_units > 0 and completed_units >= total_units:
        lifecycle = ProductBuildLifecycle.completed
    elif active_units > 0:
        lifecycle = ProductBuildLifecycle.running
    elif completed_units > 0:
        lifecycle = ProductBuildLifecycle.partial
    else:
        lifecycle = ProductBuildLifecycle.ready_to_start
    update_product_build_run_state(
        db,
        run=run,
        lifecycle=lifecycle,
        completed_units=completed_units,
        total_units=total_units,
        blocked_units=blocked_units,
        checkpoint_payload={
            **(run.checkpoint_payload or {}),
            "premium_enrichment": {
                "total_backlog": len(backlog_records),
                "resolved_backlog": sum(1 for item in backlog_records if item.status == UncertaintyBacklogStatus.resolved.value),
                "deferred_to_acp": [
                    str(item.id)
                    for item in backlog_records
                    if _is_deferred_to_acp(item)
                ],
                "requires_attention": [
                    str(item.id)
                    for item in backlog_records
                    if _premium_backlog_step_status(item) == "requires_attention"
                ],
            },
        },
    )
