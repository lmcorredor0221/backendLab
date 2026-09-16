from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any
from uuid import UUID, uuid4

from sqlmodel import Session, select

from app.services.product_processing.contracts import (
    LegacyPremiumMigrationBatchResult,
    LegacyPremiumMigrationDryRunAction,
    LegacyPremiumMigrationDryRunReport,
    ProductBuildLifecycle,
    ProductProcessingMode,
    UncertaintyBacklogStatus,
)
from app.services.product_processing.legacy_premium_inventory_service import build_legacy_premium_inventory_report
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord, UncertaintyBacklogRecord
from app.services.question_identity import merge_unique_strings
from app.models import utc_now


MIGRATION_VERSION = "legacy_premium_to_basic_acp.v1"
ACTIVE_PREMIUM_STATUSES = {
    UncertaintyBacklogStatus.open.value,
    UncertaintyBacklogStatus.in_progress.value,
    UncertaintyBacklogStatus.deferred.value,
    UncertaintyBacklogStatus.resolved.value,
}


def build_legacy_premium_migration_dry_run(
    db: Session,
    *,
    workspace_id: UUID | None = None,
    batch_size: int = 200,
) -> LegacyPremiumMigrationDryRunReport:
    """Describe one reversible Premium migration batch without writing any data."""
    normalized_batch_size = max(1, min(batch_size, 1_000))
    inventory = build_legacy_premium_inventory_report(
        db,
        workspace_id=workspace_id,
        sample_limit=normalized_batch_size,
    )
    basic_records = _basic_records_by_key(db, workspace_id=workspace_id)
    sources = _eligible_sources(db, workspace_id=workspace_id, limit=normalized_batch_size)
    actions = [
        _dry_run_action(source, basic_records=basic_records)
        for source in sources
    ]
    action_counts = Counter(action.proposed_action for action in actions)

    return LegacyPremiumMigrationDryRunReport(
        generated_at=inventory.generated_at,
        workspace_id=workspace_id,
        batch_size=normalized_batch_size,
        total_candidates=len(actions),
        proposed_create_count=action_counts["create_basic_defer_to_acp"],
        proposed_merge_count=action_counts["merge_into_basic"],
        preserved_history_count=(
            action_counts["preserve_historical_response"] + action_counts["retain_audit_only"]
        ),
        retained_technical_count=action_counts["retain_technical_pro"],
        manual_review_count=action_counts["manual_review"],
        migration_ready=inventory.migration_ready,
        blocking_reasons=list(inventory.warnings),
        actions=actions,
    )


def execute_legacy_premium_migration_batch(
    db: Session,
    *,
    workspace_id: UUID,
    batch_size: int = 200,
) -> LegacyPremiumMigrationBatchResult:
    """Migrate one Premium batch atomically while keeping source rows as audit history.

    The caller owns the transaction. This service never commits, so a failed batch can
    be rolled back as one unit.
    """
    normalized_batch_size = max(1, min(batch_size, 1_000))
    inventory = build_legacy_premium_inventory_report(
        db,
        workspace_id=workspace_id,
        sample_limit=normalized_batch_size,
    )
    if inventory.ambiguous_count:
        raise ValueError("La migracion requiere revisar registros Premium ambiguos antes de escribir un lote.")

    basic_records = _basic_records_by_key(db, workspace_id=workspace_id)
    sources = _eligible_sources(db, workspace_id=workspace_id, limit=normalized_batch_size)
    migration_id = uuid4()
    created_basic_count = 0
    merged_basic_count = 0
    preserved_historical_response_count = 0
    retained_technical_count = 0
    retained_audit_count = 0
    migrated_ids: list[UUID] = []

    for source in sources:
        classification, migration_action = _classify_source(source)
        if classification == "technical_pro":
            retained_technical_count += 1
            continue
        if classification == "ambiguous":
            raise ValueError("La migracion encontro un registro Premium ambiguo durante el lote.")
        if classification == "closed" and migration_action != "preserve_historical_response":
            retained_audit_count += 1
            continue

        target_key = (source.session_id, source.uncertainty_key)
        target = basic_records.get(target_key)
        if target is None:
            target = _create_basic_target(source)
            db.add(target)
            db.flush()
            basic_records[target_key] = target
            created_basic_count += 1
        else:
            _merge_source_into_basic_target(target, source)
            merged_basic_count += 1

        _record_migration_context(
            target,
            source=source,
            migration_id=migration_id,
            migration_action=migration_action,
        )
        _mark_source_as_migrated(
            source,
            target=target,
            migration_id=migration_id,
            migration_action=migration_action,
        )
        db.add(target)
        db.add(source)
        migrated_ids.append(source.id)
        if migration_action == "preserve_historical_response":
            preserved_historical_response_count += 1

    db.flush()
    normalized_run_count = _normalize_business_backlog_attention_runs(
        db,
        workspace_id=workspace_id,
        migration_id=migration_id,
    )
    db.flush()

    return LegacyPremiumMigrationBatchResult(
        migration_id=migration_id,
        workspace_id=workspace_id,
        batch_size=normalized_batch_size,
        scanned_count=len(sources),
        migrated_count=len(migrated_ids),
        created_basic_count=created_basic_count,
        merged_basic_count=merged_basic_count,
        preserved_historical_response_count=preserved_historical_response_count,
        retained_technical_count=retained_technical_count,
        retained_audit_count=retained_audit_count,
        normalized_business_attention_run_count=normalized_run_count,
        skipped_already_migrated_count=0,
        source_record_ids=migrated_ids,
    )


def _basic_records_by_key(
    db: Session,
    *,
    workspace_id: UUID | None,
) -> dict[tuple[UUID, str], UncertaintyBacklogRecord]:
    statement = select(UncertaintyBacklogRecord).where(
        UncertaintyBacklogRecord.product_mode == ProductProcessingMode.basic_free.value
    )
    if workspace_id is not None:
        statement = statement.where(UncertaintyBacklogRecord.workspace_id == workspace_id)
    return {
        (record.session_id, record.uncertainty_key): record
        for record in db.exec(statement).all()
    }


def _eligible_sources(
    db: Session,
    *,
    workspace_id: UUID,
    limit: int,
) -> list[UncertaintyBacklogRecord]:
    return list(
        db.exec(
            select(UncertaintyBacklogRecord)
            .where(
                UncertaintyBacklogRecord.workspace_id == workspace_id,
                UncertaintyBacklogRecord.product_mode == ProductProcessingMode.premium_enrichment.value,
                UncertaintyBacklogRecord.status.in_(ACTIVE_PREMIUM_STATUSES),
            )
            .order_by(UncertaintyBacklogRecord.updated_at.asc(), UncertaintyBacklogRecord.id.asc())
            .limit(limit)
        ).all()
    )


def _dry_run_action(
    source: UncertaintyBacklogRecord,
    *,
    basic_records: dict[tuple[UUID, str], UncertaintyBacklogRecord],
) -> LegacyPremiumMigrationDryRunAction:
    classification, migration_action = _classify_source(source)
    target = basic_records.get((source.session_id, source.uncertainty_key))
    proposed_action, reason = _proposed_action(classification, migration_action, target is not None)
    return LegacyPremiumMigrationDryRunAction(
        source_record_id=source.id,
        session_id=source.session_id,
        uncertainty_key=source.uncertainty_key,
        classification=classification,
        proposed_action=proposed_action,
        target_record_id=target.id if target is not None else None,
        reason=reason,
    )


def _proposed_action(
    classification: str,
    migration_action: str,
    has_basic_collision: bool,
) -> tuple[str, str]:
    if classification in {"business_pre_acp", "acp_managed"}:
        if has_basic_collision:
            return "merge_into_basic", "Fusionaria referencias y contexto en el pendiente Basic existente para ACP."
        return "create_basic_defer_to_acp", "Crearia un pendiente Basic diferido a ACP y conservaria el origen Premium."
    if classification == "technical_pro":
        return "retain_technical_pro", "El fallo tecnico permanece visible y recuperable en Blueprint Pro."
    if classification == "ambiguous":
        return "manual_review", "No hay una regla segura para transformar este registro automaticamente."
    if migration_action == "preserve_historical_response":
        return "preserve_historical_response", "La respuesta historica se conservaria como contexto ratificable en ACP."
    return "retain_audit_only", "El registro cerrado se conservaria solo para auditoria."


def _classify_source(source: UncertaintyBacklogRecord) -> tuple[str, str]:
    status = str(source.status or "").strip().lower()
    kind = str(source.kind or "").strip().lower()
    disposition = str(source.disposition or "").strip().lower()
    target_stage = str(source.target_stage or "").strip().lower()
    payload = source.payload if isinstance(source.payload, dict) else {}

    if status in {UncertaintyBacklogStatus.dismissed.value, UncertaintyBacklogStatus.superseded.value}:
        return "closed", "retain_audit_only"
    if status == UncertaintyBacklogStatus.resolved.value:
        if source.assumed_answer or source.suggested_answer:
            return "closed", "preserve_historical_response"
        return "closed", "retain_audit_only"
    if isinstance(payload.get("acp_resolution"), dict) or (target_stage == "acp" and disposition == "defer"):
        return "acp_managed", "merge_acp_metadata"
    if kind in {"runtime_error", "stale_dependency"}:
        return "technical_pro", "retain_technical_pro"
    if kind in {"question", "gap", "assumption", "decision", "hitl"}:
        return "business_pre_acp", "defer_to_acp"
    return "ambiguous", "manual_review"


def _create_basic_target(source: UncertaintyBacklogRecord) -> UncertaintyBacklogRecord:
    return UncertaintyBacklogRecord(
        workspace_id=source.workspace_id,
        session_id=source.session_id,
        uncertainty_key=source.uncertainty_key,
        product_mode=ProductProcessingMode.basic_free.value,
        source_stage=source.source_stage,
        target_stage="acp",
        kind=source.kind,
        disposition="defer",
        status=UncertaintyBacklogStatus.deferred.value,
        title=source.title,
        description=source.description,
        reason=source.reason,
        impact=source.impact,
        confidence=source.confidence,
        cost_to_resolve_units=source.cost_to_resolve_units,
        assumed_answer=source.assumed_answer,
        suggested_answer=source.suggested_answer,
        answer_options=deepcopy(source.answer_options or []),
        source_refs=list(source.source_refs or []),
        affected_deliverable_keys=list(source.affected_deliverable_keys or []),
        dependency_keys=list(source.dependency_keys or []),
        payload={},
        created_from="legacy_premium_migration",
    )


def _merge_source_into_basic_target(target: UncertaintyBacklogRecord, source: UncertaintyBacklogRecord) -> None:
    target.target_stage = "acp"
    target.disposition = "defer"
    target.status = UncertaintyBacklogStatus.deferred.value
    target.source_stage = target.source_stage or source.source_stage
    target.title = target.title or source.title
    target.description = target.description or source.description
    target.reason = target.reason or source.reason
    target.impact = target.impact or source.impact
    target.confidence = max(float(target.confidence or 0), float(source.confidence or 0))
    target.cost_to_resolve_units = max(int(target.cost_to_resolve_units or 1), int(source.cost_to_resolve_units or 1))
    target.assumed_answer = target.assumed_answer or source.assumed_answer
    target.suggested_answer = target.suggested_answer or source.suggested_answer
    target.answer_options = _merge_json_values(target.answer_options or [], source.answer_options or [])
    target.source_refs = merge_unique_strings([*(target.source_refs or []), *(source.source_refs or [])])
    target.affected_deliverable_keys = merge_unique_strings(
        [*(target.affected_deliverable_keys or []), *(source.affected_deliverable_keys or [])]
    )
    target.dependency_keys = merge_unique_strings([*(target.dependency_keys or []), *(source.dependency_keys or [])])


def _merge_json_values(current: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in [*current, *incoming]:
        normalized = dict(value) if isinstance(value, dict) else {"value": str(value)}
        signature = json.dumps(normalized, ensure_ascii=True, sort_keys=True, default=str)
        if signature not in seen:
            seen.add(signature)
            merged.append(normalized)
    return merged


def _record_migration_context(
    target: UncertaintyBacklogRecord,
    *,
    source: UncertaintyBacklogRecord,
    migration_id: UUID,
    migration_action: str,
) -> None:
    now = utc_now()
    payload = deepcopy(target.payload) if isinstance(target.payload, dict) else {}
    migration = dict(payload.get("migration_v1") or {})
    sources = list(migration.get("source_records") or [])
    source_id = str(source.id)
    if not any(str(item.get("id") or "") == source_id for item in sources if isinstance(item, dict)):
        sources.append(
            {
                "id": source_id,
                "product_mode": source.product_mode,
                "migration_action": migration_action,
                "payload": deepcopy(source.payload) if isinstance(source.payload, dict) else {},
                "assumed_answer": source.assumed_answer,
                "suggested_answer": source.suggested_answer,
            }
        )
    migration.update(
        {
            "version": MIGRATION_VERSION,
            "migration_id": str(migration_id),
            "migrated_at": now.isoformat(),
            "source_records": sources,
        }
    )
    payload["migration_v1"] = migration
    payload["legacy_premium"] = True
    payload["source_tier"] = "blueprint_pro"
    target.payload = payload
    target.updated_at = now


def _mark_source_as_migrated(
    source: UncertaintyBacklogRecord,
    *,
    target: UncertaintyBacklogRecord,
    migration_id: UUID,
    migration_action: str,
) -> None:
    now = utc_now()
    payload = deepcopy(source.payload) if isinstance(source.payload, dict) else {}
    payload["migration_v1"] = {
        "version": MIGRATION_VERSION,
        "migration_id": str(migration_id),
        "migrated_at": now.isoformat(),
        "target_record_id": str(target.id),
        "migration_action": migration_action,
    }
    source.payload = payload
    source.status = UncertaintyBacklogStatus.superseded.value
    source.superseded_at = now
    source.updated_at = now


def _normalize_business_backlog_attention_runs(
    db: Session,
    *,
    workspace_id: UUID,
    migration_id: UUID,
) -> int:
    runs = db.exec(
        select(ProductBuildRunRecord).where(
            ProductBuildRunRecord.workspace_id == workspace_id,
            ProductBuildRunRecord.lifecycle == ProductBuildLifecycle.requires_attention.value,
        )
    ).all()
    normalized_count = 0
    for run in runs:
        steps = db.exec(select(ProductBuildStepRecord).where(ProductBuildStepRecord.run_id == run.id)).all()
        legacy_steps = [step for step in steps if step.step_key.startswith("premium_backlog:")]
        if not legacy_steps or _has_technical_attention(run, steps, legacy_steps):
            continue
        for step in legacy_steps:
            checkpoint = dict(step.checkpoint_payload or {})
            checkpoint["legacy_premium_migration_v1"] = {
                "migration_id": str(migration_id),
                "normalized_at": utc_now().isoformat(),
                "reason": "business_backlog_moved_to_acp",
            }
            step.status = "skipped"
            step.progress_percent = 100
            step.error_payload = {}
            step.checkpoint_payload = checkpoint
            step.completed_at = step.completed_at or utc_now()
            step.updated_at = utc_now()
            db.add(step)

        active_steps = [step for step in steps if step not in legacy_steps]
        run.lifecycle = _normalized_lifecycle(active_steps)
        run.blocked_units = 0
        run.requires_attention_at = None
        checkpoint = dict(run.checkpoint_payload or {})
        checkpoint["legacy_premium_migration_v1"] = {
            "migration_id": str(migration_id),
            "normalized_at": utc_now().isoformat(),
            "reason": "business_backlog_moved_to_acp",
        }
        run.checkpoint_payload = checkpoint
        run.updated_at = utc_now()
        db.add(run)
        normalized_count += 1
    return normalized_count


def _has_technical_attention(
    run: ProductBuildRunRecord,
    steps: list[ProductBuildStepRecord],
    legacy_steps: list[ProductBuildStepRecord],
) -> bool:
    legacy_ids = {step.id for step in legacy_steps}
    return bool(run.error_payload) or any(
        step.id not in legacy_ids and step.status in {"error", "failed", "requires_attention"}
        for step in steps
    )


def _normalized_lifecycle(steps: list[ProductBuildStepRecord]) -> str:
    if not steps:
        return ProductBuildLifecycle.ready_to_start.value
    if all(step.status in {"completed", "available", "skipped"} for step in steps):
        return ProductBuildLifecycle.completed.value
    return ProductBuildLifecycle.partial.value
