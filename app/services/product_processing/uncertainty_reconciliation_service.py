from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import CommercialTier, utc_now
from app.services.deliverable_catalog.contracts import DeliverableGenerationTask, DeliverableRegenerationScope
from app.services.deliverable_catalog.dependency_service import (
    invalidate_deliverables_for_change,
    resolve_regeneration_scope,
)
from app.services.deliverable_catalog.generation_service import run_deliverable_generation_task
from app.services.deliverable_catalog.registry_service import get_registry_entry
from app.services.product_processing.contracts import ProductProcessingMode
from app.services.product_processing.persistence import UncertaintyBacklogRecord


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
MAX_RECONCILIATION_BATCH_SIZE = 3


@dataclass(frozen=True)
class UncertaintyReconciliationPlan:
    changed_dependency_keys: list[str]
    scope: DeliverableRegenerationScope
    reconciliation_decision: str
    material_impact: bool
    recommended_action: str
    impact_summary: str


@dataclass(frozen=True)
class UncertaintyReconciliationResult:
    plan: UncertaintyReconciliationPlan
    reconciliation_status: str
    queue_total: int
    queue_completed: int
    reconciled_deliverable_keys: list[str]
    stale_deliverable_keys: list[str]
    generation_job_ids: list[str]
    generation_status_by_deliverable: dict[str, str]
    superseded_uncertainty_count: int


@dataclass(frozen=True)
class AcpUncertaintyReconciliationSummary:
    reviewed_backlog_ids: list[str]
    reconciled_backlog_ids: list[str]
    skipped_backlog_ids: list[str]
    failed_backlog_ids: list[str]
    queue_total: int
    queue_completed: int
    reconciled_deliverable_keys: list[str]

    def as_checkpoint(self) -> dict[str, object]:
        return {
            "reviewed_backlog_ids": self.reviewed_backlog_ids,
            "reconciled_backlog_ids": self.reconciled_backlog_ids,
            "skipped_backlog_ids": self.skipped_backlog_ids,
            "failed_backlog_ids": self.failed_backlog_ids,
            "queue_total": self.queue_total,
            "queue_completed": self.queue_completed,
            "reconciled_deliverable_keys": self.reconciled_deliverable_keys,
        }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def dependency_keys_for_uncertainty(record: UncertaintyBacklogRecord) -> list[str]:
    return _dedupe(
        [
            *(record.dependency_keys or []),
            *(record.affected_deliverable_keys or []),
            STAGE_DEPENDENCY_KEY.get(record.source_stage, record.source_stage),
        ]
    )


def resolve_reconciliation_decision(ordered_regeneration_keys: list[str]) -> tuple[str, bool, str, str]:
    total = len(ordered_regeneration_keys)
    if total == 0:
        return (
            "document_only",
            False,
            "document_only",
            "La respuesta se registra como decision trazable sin reconciliar entregables porque no se detecto impacto material.",
        )
    if total <= 3:
        return (
            "localized_reconciliation",
            True,
            "review_and_apply_localized_reconciliation",
            f"La respuesta impacta {total} entregable(s) versionado(s) y se puede reconciliar de forma localizada.",
        )
    return (
        "structural_reconciliation",
        True,
        "review_and_apply_structural_reconciliation",
        f"La respuesta impacta {total} entregable(s) y conviene reconciliar los entregables dependientes sin reabrir fases.",
    )


def build_uncertainty_reconciliation_plan(record: UncertaintyBacklogRecord) -> UncertaintyReconciliationPlan:
    changed_dependency_keys = dependency_keys_for_uncertainty(record)
    source_key = (record.affected_deliverable_keys or [""])[0]
    scope = resolve_regeneration_scope(
        changed_dependency_keys=changed_dependency_keys,
        source_deliverable_key=source_key,
    )
    decision, material_impact, action, summary = resolve_reconciliation_decision(scope.ordered_regeneration_keys)
    return UncertaintyReconciliationPlan(
        changed_dependency_keys=changed_dependency_keys,
        scope=scope,
        reconciliation_decision=decision,
        material_impact=material_impact,
        recommended_action=action,
        impact_summary=summary,
    )


def _reconciliation_batch_size() -> int:
    try:
        configured = int(getattr(get_settings(), "product_build_batch_size", 1) or 1)
    except (TypeError, ValueError):
        configured = 1
    return max(1, min(MAX_RECONCILIATION_BATCH_SIZE, configured))


def _dependencies_ready_for_reconciliation(
    deliverable_key: str,
    *,
    queue_keys: set[str],
    completed_keys: set[str],
) -> bool:
    entry = get_registry_entry(deliverable_key)
    if entry is None:
        return True
    for dependency_key in entry.dependency_policy.depends_on:
        normalized = str(dependency_key or "").strip()
        if normalized in queue_keys and normalized not in completed_keys:
            return False
    return True


def _next_reconciliation_batch(
    queue: list[str],
    *,
    processed_keys: set[str],
    completed_keys: set[str],
    batch_size: int,
) -> list[str]:
    queue_keys = set(queue)
    remaining = [key for key in queue if key not in processed_keys]
    ready = [
        key
        for key in remaining
        if _dependencies_ready_for_reconciliation(key, queue_keys=queue_keys, completed_keys=completed_keys)
    ]
    if ready:
        return ready[:batch_size]
    return remaining[:1]


def _run_reconciliation_generation(
    *,
    database_engine: Engine,
    workspace_id: UUID,
    session_id: UUID,
    source_stage: str,
    title: str,
    reason: str,
    impact: str,
    deliverable_key: str,
    product_mode: str,
    tier: CommercialTier,
    idempotency_key: str,
    actor_user_id: UUID,
    answer: str,
    changed_dependency_keys: list[str],
    allow_llm: bool,
) -> tuple[str, str, str]:
    with Session(database_engine) as worker_db:
        job, _ = run_deliverable_generation_task(
            worker_db,
            DeliverableGenerationTask(
                workspace_id=workspace_id,
                session_id=session_id,
                deliverable_key=deliverable_key,
                product_mode=product_mode,
                current_stage=source_stage or "package",
                tier=tier,
                idempotency_key=idempotency_key,
                requested_by_user_id=actor_user_id,
                context_payload={
                    "summary": title,
                    "resolved_answer": answer,
                    "reason": reason,
                    "impact": impact,
                },
                approved_context_refs=changed_dependency_keys,
                allow_llm=allow_llm,
                max_iterations=5,
            ),
        )
        worker_db.commit()
        return deliverable_key, str(job.id), str(job.status or "")


def execute_uncertainty_reconciliation(
    db: Session,
    *,
    record: UncertaintyBacklogRecord,
    actor_user_id: UUID,
    answer: str,
    resolution_key: str,
    product_mode: str,
    tier: CommercialTier,
    idempotency_prefix: str,
    max_deliverables: int = 5,
    execute: bool = True,
    allow_llm: bool = True,
) -> UncertaintyReconciliationResult:
    """Reconcile only the catalog entries affected by one already-recorded decision."""
    plan = build_uncertainty_reconciliation_plan(record)
    queue = plan.scope.ordered_regeneration_keys[:max(0, max_deliverables)]
    queue_total = len(queue) if (execute or plan.material_impact) else 0
    status = "queued" if execute and queue else "pending_user_confirmation" if plan.material_impact else "not_required"
    payload = dict(record.payload or {})
    payload[resolution_key] = {
        **(payload.get(resolution_key) if isinstance(payload.get(resolution_key), dict) else {}),
        "answer": answer,
        "actor_user_id": str(actor_user_id),
        "changed_dependency_keys": plan.changed_dependency_keys,
        "affected_deliverable_keys": plan.scope.affected_deliverable_keys,
        "ordered_regeneration_keys": plan.scope.ordered_regeneration_keys,
        "reconciliation_decision": plan.reconciliation_decision,
        "reconciliation_status": status,
        "material_impact": plan.material_impact,
        "queue_total": queue_total,
        "queue_completed": 0,
        "queue_pending_keys": queue,
        "recommended_action": plan.recommended_action,
        "impact_summary": plan.impact_summary,
        "reconciled_at": utc_now().isoformat() if execute else "",
    }
    record.payload = payload
    record.updated_at = utc_now()
    db.add(record)
    db.flush()
    db.commit()

    regenerated: list[str] = []
    job_ids: list[str] = []
    status_by_key: dict[str, str] = {}
    stale_keys: list[str] = []
    superseded_count = 0
    if execute and queue:
        stale_report = invalidate_deliverables_for_change(
            db,
            workspace_id=record.workspace_id,
            session_id=record.session_id,
            changed_dependency_keys=plan.changed_dependency_keys,
            source_deliverable_key=plan.scope.source_deliverable_key,
        )
        stale_keys = stale_report.stale_deliverable_keys
        superseded_count = stale_report.superseded_uncertainty_count
        batch_size = _reconciliation_batch_size()
        processed_keys: set[str] = set()
        completed_keys: set[str] = set()
        database_engine = db.get_bind()
        while len(processed_keys) < len(queue):
            batch = _next_reconciliation_batch(
                queue,
                processed_keys=processed_keys,
                completed_keys=completed_keys,
                batch_size=batch_size,
            )
            if batch_size <= 1:
                deliverable_key = batch[0]
                try:
                    _, job_id, job_status = _run_reconciliation_generation(
                        database_engine=database_engine,
                        workspace_id=record.workspace_id,
                        session_id=record.session_id,
                        source_stage=record.source_stage,
                        title=record.title,
                        reason=record.reason,
                        impact=record.impact,
                        deliverable_key=deliverable_key,
                        product_mode=product_mode,
                        tier=tier,
                        idempotency_key=f"{idempotency_prefix}:{record.id}:{deliverable_key}",
                        actor_user_id=actor_user_id,
                        answer=answer,
                        changed_dependency_keys=plan.changed_dependency_keys,
                        allow_llm=allow_llm,
                    )
                    job_ids.append(job_id)
                    status_by_key[deliverable_key] = job_status
                    if job_status == "available":
                        regenerated.append(deliverable_key)
                        completed_keys.add(deliverable_key)
                except (LookupError, PermissionError, ValueError) as exc:
                    status_by_key[deliverable_key] = f"skipped:{exc}"
                processed_keys.add(deliverable_key)
                continue

            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(
                        _run_reconciliation_generation,
                        database_engine=database_engine,
                        workspace_id=record.workspace_id,
                        session_id=record.session_id,
                        source_stage=record.source_stage,
                        title=record.title,
                        reason=record.reason,
                        impact=record.impact,
                        deliverable_key=deliverable_key,
                        product_mode=product_mode,
                        tier=tier,
                        idempotency_key=f"{idempotency_prefix}:{record.id}:{deliverable_key}",
                        actor_user_id=actor_user_id,
                        answer=answer,
                        changed_dependency_keys=plan.changed_dependency_keys,
                        allow_llm=allow_llm,
                    ): deliverable_key
                    for deliverable_key in batch
                }
                for future in as_completed(futures):
                    deliverable_key = futures[future]
                    try:
                        _, job_id, job_status = future.result()
                        job_ids.append(job_id)
                        status_by_key[deliverable_key] = job_status
                        if job_status == "available":
                            regenerated.append(deliverable_key)
                            completed_keys.add(deliverable_key)
                    except (LookupError, PermissionError, ValueError) as exc:
                        status_by_key[deliverable_key] = f"skipped:{exc}"
                    finally:
                        processed_keys.add(deliverable_key)
        status = "completed" if len(regenerated) == queue_total else "completed_with_errors"

    payload = dict(record.payload or {})
    payload[resolution_key] = {
        **(payload.get(resolution_key) if isinstance(payload.get(resolution_key), dict) else {}),
        "reconciliation_status": status,
        "reconciled_deliverable_keys": regenerated,
        "generation_job_ids": job_ids,
        "generation_status_by_deliverable": status_by_key,
        "queue_completed": len(regenerated),
        "queue_pending_keys": [key for key in queue if key not in set(regenerated)],
        "superseded_uncertainty_count": superseded_count,
    }
    record.payload = payload
    record.updated_at = utc_now()
    db.add(record)
    db.flush()
    return UncertaintyReconciliationResult(
        plan=plan,
        reconciliation_status=status,
        queue_total=queue_total,
        queue_completed=len(regenerated),
        reconciled_deliverable_keys=regenerated,
        stale_deliverable_keys=stale_keys,
        generation_job_ids=job_ids,
        generation_status_by_deliverable=status_by_key,
        superseded_uncertainty_count=superseded_count,
    )


def reconcile_confirmed_acp_uncertainties(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    actor_user_id: UUID,
    max_deliverables: int = 5,
) -> AcpUncertaintyReconciliationSummary:
    """Run only after ACP has explicitly confirmed an inherited uncertainty."""
    rows = db.exec(
        select(UncertaintyBacklogRecord)
        .where(
            UncertaintyBacklogRecord.workspace_id == workspace_id,
            UncertaintyBacklogRecord.session_id == session_id,
        )
        .order_by(UncertaintyBacklogRecord.created_at.asc())
    ).all()
    reviewed: list[str] = []
    reconciled: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    deliverables: list[str] = []
    queue_total = 0
    queue_completed = 0
    for record in rows:
        payload = record.payload if isinstance(record.payload, dict) else {}
        acp_resolution = payload.get("acp_resolution")
        if not isinstance(acp_resolution, dict) or str(record.status or "").lower() != "resolved":
            continue
        decision = str(acp_resolution.get("decision") or "")
        if decision not in {"answer", "choose_option"}:
            continue
        reviewed.append(str(record.id))
        previous = payload.get("acp_reconciliation")
        if isinstance(previous, dict) and previous.get("reconciliation_status") == "completed":
            skipped.append(str(record.id))
            continue
        answer = str(record.assumed_answer or acp_resolution.get("answer") or "").strip()
        try:
            result = execute_uncertainty_reconciliation(
                db,
                record=record,
                actor_user_id=actor_user_id,
                answer=answer,
                resolution_key="acp_reconciliation",
                product_mode=ProductProcessingMode.acp_implementation.value,
                tier=CommercialTier.acp,
                idempotency_prefix=f"acp-reconciliation:{session_id}",
                max_deliverables=max_deliverables,
            )
        except (LookupError, PermissionError, ValueError):
            failed.append(str(record.id))
            continue
        reconciled.append(str(record.id))
        queue_total += result.queue_total
        queue_completed += result.queue_completed
        deliverables.extend(result.reconciled_deliverable_keys)
    return AcpUncertaintyReconciliationSummary(
        reviewed_backlog_ids=reviewed,
        reconciled_backlog_ids=reconciled,
        skipped_backlog_ids=skipped,
        failed_backlog_ids=failed,
        queue_total=queue_total,
        queue_completed=queue_completed,
        reconciled_deliverable_keys=_dedupe(deliverables),
    )
