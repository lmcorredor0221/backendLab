from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from fastapi import BackgroundTasks
from sqlmodel import Session

from app.models import (
    CommercialTier,
    SessionRecord,
    SessionSnapshot,
    UserRecord,
)
from app.api.routes.sessions import ensure_commercial_capability
from app.services.deliverable_catalog.contracts import (
    DeliverableGenerationMode,
    DeliverableGenerationTask,
    DeliverableType,
)
from app.services.deliverable_catalog.generation_service import run_deliverable_generation_task
from app.services.deliverable_catalog.registry_service import get_registry_entry, list_registry_entries
from app.services.diagram_center.catalog_service import build_catalog_v3
from app.services.diagram_center.generation_service import create_generation_job, run_generation_job
from app.services.product_processing.contracts import (
    ProductBuildCommandRequest,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductBuildStatus,
    ProductProcessingMode,
)
from app.services.product_processing.product_build_orchestrator import (
    ProductBuildOrchestrationOptions,
    ensure_product_build_orchestration,
    reconcile_product_build_run,
)
from app.api.routes.sessions import (
    build_snapshot,
    capture_operational_state,
    write_log,
)
from app.services.blueprint_commercial_result_service import (
    record_blueprint_commercial_result_artifacts,
)


BLUEPRINT_COMMERCIAL_RESULT_ACTION = "prepare_blueprint_commercial_result"


def _is_blueprint_basic_auto_deliverable(entry) -> bool:
    return (
        entry.deliverable_type != DeliverableType.diagram
        and "blueprint" in entry.product_scope
        and entry.required_tier == CommercialTier.blueprint
        and entry.generation_mode
        not in {
            DeliverableGenerationMode.llm_required,
            DeliverableGenerationMode.manual_review_required,
        }
    )


def _blueprint_commercial_deliverable_context(snapshot: SessionSnapshot) -> dict[str, Any]:
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    blueprint = snapshot.blueprint
    estimation = snapshot.estimation_report

    return {
        "session_id": str(snapshot.session.id),
        "workspace_id": str(snapshot.session.workspace_id) if snapshot.session.workspace_id else "",
        "session_title": snapshot.session.title or "Agente Inteligente",
        "problem_statement": discovery.problem_statement if discovery else "",
        "current_process": discovery.current_process if discovery else "",
        "current_user": discovery.current_user if discovery else (canvas.agent_profile.primary_user if canvas and canvas.agent_profile else ""),
        "desired_outcome": discovery.desired_outcome if discovery else "",
        "value_statement": discovery.value_statement if discovery else "",
        "autonomy_level": discovery.autonomy_level if discovery else "medium",
        "current_time_spent": discovery.operational_baseline.current_time_spent if discovery and discovery.operational_baseline else "",
        "current_cost": discovery.operational_baseline.current_cost if discovery and discovery.operational_baseline else "",
        "frequent_errors": discovery.operational_baseline.frequent_errors if discovery and discovery.operational_baseline else [],
        "automation_opportunities": discovery.operational_baseline.automation_opportunities if discovery and discovery.operational_baseline else [],
        "north_star_metric": (
            discovery.mvp_definition.north_star_metric
            if discovery and discovery.mvp_definition
            else (canvas.success_metric if canvas else "")
        ),
        "non_delegable_decisions": (
            discovery.mvp_definition.non_delegable_decisions
            if discovery and discovery.mvp_definition
            else []
        ),
        "constraints": discovery.constraints if discovery else [],
        "user_goal": canvas.user_goal if canvas else "",
        "agent_mission": canvas.agent_profile.mission if canvas and canvas.agent_profile else "",
        "mvp_scope": canvas.mvp_scope if canvas else [],
        "out_of_scope": canvas.out_of_scope if canvas else [],
        "primary_risk": canvas.primary_risk if canvas else "",
        "success_metric": canvas.success_metric if canvas else "",
        "human_approvals": canvas.agent_profile.human_approvals if canvas and canvas.agent_profile else [],
        "architecture": blueprint.architecture if blueprint else "supervisor_with_subagents",
        "reasoning_pattern": blueprint.reasoning_pattern if blueprint else "Plan-and-Execute",
        "memory_strategy": blueprint.memory_strategy if blueprint else "session_and_checkpoints",
        "guardrails": blueprint.guardrails if blueprint else [],
        "narrative": blueprint.narrative if blueprint else "",
        "tools": [
            {
                "name": t.name,
                "purpose": t.purpose,
                "requires_approval": t.requires_approval,
                "has_side_effects": t.has_side_effects,
                "inputs": t.inputs,
                "outputs": t.outputs,
            }
            for t in (blueprint.tools if blueprint and blueprint.tools else [])
        ],
        "tool_count": len(blueprint.tools) if blueprint and blueprint.tools else 0,
        "discovery_summary": (
            discovery.problem_statement or discovery.value_statement if discovery else ""
        ),
        "canvas_summary": (
            canvas.user_goal or (canvas.agent_profile.mission if canvas.agent_profile else "") if canvas else ""
        ),
        "blueprint_summary": (
            blueprint.narrative or blueprint.architecture if blueprint else ""
        ),
        "tools_summary": (
            f"{len(blueprint.tools)} tools configuradas" if blueprint and blueprint.tools else ""
        ),
        "estimation_summary": (
            f"Horas agénticas: {estimation.agentic.estimated_hours_total} vs tradicionales: {estimation.traditional.estimated_hours_total}"
            if estimation
            else ""
        ),
        "estimation_report": (
            {
                "agentic_hours": estimation.agentic.estimated_hours_total,
                "agentic_cost": estimation.agentic.estimated_cost,
                "traditional_hours": estimation.traditional.estimated_hours_total if estimation.traditional else 0,
                "traditional_cost": estimation.traditional.estimated_cost if estimation.traditional else 0,
                "net_savings": estimation.agentic.net_savings_vs_traditional,
                "effort_reduction_percent": getattr(estimation.agentic, "effort_reduction_vs_traditional_percent", 59),
                "automation_coverage": getattr(estimation.agentic, "automation_coverage_percent", 77),
                "human_supervision_hours": getattr(estimation.agentic, "human_supervision_hours", 0),
                "confidence_score": estimation.confidence.score if estimation.confidence else 82,
                "confidence_label": getattr(estimation.confidence, "label", "Media-Alta") if estimation.confidence else "Media-Alta",
                "uncertainty_band": estimation.confidence.uncertainty_band_percent if estimation.confidence else 38,
                "workstreams": [
                    ws.model_dump(mode="json") if hasattr(ws, "model_dump") else ws
                    for ws in (getattr(estimation.agentic, "workstreams", []) or [])
                ],
                "construction_scenarios": [
                    sc.model_dump(mode="json") if hasattr(sc, "model_dump") else sc
                    for sc in (getattr(estimation, "construction_scenarios", []) or [])
                ],
            }
            if estimation
            else {}
        ),
    }


def _generate_blueprint_basic_deliverables(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
    current_user: UserRecord,
    allow_llm: bool = False,
) -> tuple[list[str], list[dict[str, str]]]:
    if record.workspace_id is None:
        return [], [{"deliverable_key": "*", "reason": "session_without_workspace"}]
    context_payload = _blueprint_commercial_deliverable_context(snapshot)
    generated_keys: list[str] = []
    skipped: list[dict[str, str]] = []
    for entry in list_registry_entries():
        if not _is_blueprint_basic_auto_deliverable(entry):
            continue
        version_num = max([item.version_number for item in (snapshot.blueprint_versions or [])], default=1)
        est_hash = hashlib.sha256(json.dumps(context_payload.get("estimation_report", {}), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
        task = DeliverableGenerationTask(
            workspace_id=record.workspace_id,
            session_id=record.id,
            deliverable_key=entry.deliverable_key,
            product_mode=ProductProcessingMode.basic_free.value,
            current_stage="estimate",
            tier=CommercialTier.blueprint,
            idempotency_key=f"blueprint-commercial-result:{record.id}:v{version_num}:e{est_hash}:deliverable:{entry.deliverable_key}",
            requested_by_user_id=current_user.id,
            context_payload=context_payload,
            approved_context_refs=[
                "session.discovery",
                "session.canvas",
                "session.blueprint",
                "session.tools",
                "session.estimation_report",
            ],
            allow_llm=allow_llm,
            max_iterations=entry.prompt_policy.max_iterations or 1,
        )
        try:
            job, _ = run_deliverable_generation_task(db, task)
        except (LookupError, PermissionError, ValueError) as exc:
            skipped.append({"deliverable_key": entry.deliverable_key, "reason": str(exc)})
            continue
        generated_keys.append(f"{entry.deliverable_key}:{job.status}")
    return generated_keys, skipped


def prepare_blueprint_basic_commercial_result(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    background_tasks: BackgroundTasks | None = None,
    allow_llm: bool = False,
) -> tuple[SessionSnapshot, ProductBuildStatus]:
    ensure_commercial_capability(record, "blueprint.view", db=db, current_user=current_user)
    ensure_product_build_orchestration(
        db,
        record=record,
        product_key=ProductBuildProductKey.blueprint_basic,
        current_user=current_user,
        options=ProductBuildOrchestrationOptions(current_stage="estimate"),
    )
    snapshot = build_snapshot(db, record, current_user=current_user)
    artifacts = record_blueprint_commercial_result_artifacts(db, record=record, snapshot=snapshot)
    deliverable_keys, skipped_deliverables = _generate_blueprint_basic_deliverables(
        db,
        record=record,
        snapshot=snapshot,
        current_user=current_user,
        allow_llm=allow_llm,
    )
    diagram_jobs = []
    catalog = build_catalog_v3(db, record=record, role=None)
    for item in catalog.entries:
        if "blueprint" not in item.products or item.required_tier != "blueprint":
            continue
        if item.current_version is not None:
            continue
        job = create_generation_job(
            db,
            record=record,
            diagram_key=item.key,
            user_id=current_user.id,
            detail_level="standard",
            reason=BLUEPRINT_COMMERCIAL_RESULT_ACTION,
            idempotency_key=f"blueprint-commercial-result:{record.id}:{item.key}",
        )
        if job.status in {"queued", "updating"}:
            diagram_jobs.append(job)
            if background_tasks is not None:
                background_tasks.add_task(run_generation_job, job.id, db.get_bind())
            else:
                run_generation_job(job.id, db_session=db)
    db.commit()
    write_log(
        db,
        session_id=record.id,
        stage=record.current_stage,
        status_value=record.status,
        message="Resultado comercial del Blueprint preparado.",
        payload={
            "artifact_count": len(artifacts),
            "artifact_keys": [artifact.artifact_key for artifact in artifacts],
            "deliverable_job_count": len(deliverable_keys),
            "deliverable_keys": deliverable_keys,
            "skipped_deliverables": skipped_deliverables,
            "diagram_job_count": len(diagram_jobs),
            "diagram_keys": [job.diagram_key for job in diagram_jobs],
            "source_action": BLUEPRINT_COMMERCIAL_RESULT_ACTION,
        },
    )
    capture_operational_state(
        db,
        session_id=record.id,
        source_action=BLUEPRINT_COMMERCIAL_RESULT_ACTION,
    )
    status = ensure_product_build_orchestration(
        db,
        record=record,
        product_key=ProductBuildProductKey.blueprint_basic,
        current_user=current_user,
        options=ProductBuildOrchestrationOptions(current_stage="estimate"),
    )
    db.commit()
    refreshed_snapshot = build_snapshot(db, record, current_user=current_user)
    return refreshed_snapshot, status


def execute_blueprint_basic_action(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    payload: ProductBuildCommandRequest,
    background_tasks: BackgroundTasks | None = None,
) -> ProductBuildStatus:
    if payload.action in {"start", "resume", "retry"}:
        _, status = prepare_blueprint_basic_commercial_result(
            db,
            record=record,
            current_user=current_user,
            background_tasks=background_tasks,
            allow_llm=payload.allow_llm,
        )
        return status
    return reconcile_product_build_run(
        db,
        record=record,
        product_key=ProductBuildProductKey.blueprint_basic,
        current_user=current_user,
    )
