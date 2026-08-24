from __future__ import annotations

import json
from datetime import timedelta
from hashlib import sha256
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    ACPPreview,
    AlertEventEntry,
    AlertEventRecord,
    ApprovalGateRecord,
    ApprovalStatus,
    ArtifactRecordEntry,
    ArtifactRegistryRecord,
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintRecord,
    BlueprintVersionRecord,
    CanvasArtifact,
    CanvasRecord,
    DiscoveryArtifact,
    DesignRecommendationArtifact,
    EstimationReportArtifact,
    EvaluationRunRecord,
    ExecutionLogEntry,
    ExecutionLogRecord,
    IntegrationStatusEntry,
    IntegrationStatusRecord,
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    LLMRuntimeSettings,
    MemoryObservabilityReport,
    MetricSnapshotEntry,
    MetricSnapshotRecord,
    MonitoringReleaseObservability,
    MonitoringWorkspace,
    OpportunityRecord,
    SessionRecord,
    SessionStage,
    SkillRunRecord,
    ToolRecommendationArtifact,
    UserRecord,
    ValidationReportRecord,
    utc_now,
)
from app.services.acp_serialization import serialize_json_document
from app.services.blueprint_hydration import hydrate_blueprint_record
from app.services.llm_runtime.codex_cli.execution_service import CodexExecutionService
from app.services.llm_runtime.runtime_settings_service import (
    load_effective_runtime_settings,
    load_effective_runtime_settings_for_session,
    load_platform_runtime_defaults,
)
from app.services.openai_builder import build_builder_service
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionOutput
from app.services.stage5_service import FEATURE_FLAG_TOOL_RECOMMENDATION, is_feature_flag_enabled
from app.services.tool_recommendation_service import annotate_tool_recommendation_status


ACTIVE_ALERT_STATUS = "active"
RESOLVED_ALERT_STATUS = "resolved"


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _hydrate_blueprint_record(record: BlueprintRecord) -> BlueprintArtifact:
    return hydrate_blueprint_record(record)


def _hydrate_canvas_record(record: CanvasRecord) -> CanvasArtifact:
    return CanvasArtifact.model_validate(record.model_dump(exclude={"id", "session_id", "updated_at"}))


def _hydrate_discovery_record(record: OpportunityRecord) -> DiscoveryArtifact:
    return DiscoveryArtifact.model_validate(record.model_dump(exclude={"id", "session_id", "updated_at"}))


def _latest_blueprint_version_number(session: Session, session_id: UUID) -> int | None:
    latest = session.exec(
        select(BlueprintVersionRecord)
        .where(BlueprintVersionRecord.session_id == session_id)
        .order_by(BlueprintVersionRecord.version_number.desc())
    ).first()
    return latest.version_number if latest is not None else None


def _load_latest_tool_recommendation_status(session: Session, session_id: UUID) -> ToolRecommendationArtifact | None:
    record = session.exec(
        select(ArtifactRegistryRecord)
        .where(
            ArtifactRegistryRecord.session_id == session_id,
            ArtifactRegistryRecord.artifact_kind == "tool_recommendation",
        )
        .order_by(ArtifactRegistryRecord.created_at.desc())
    ).first()
    if record is None or not record.content_text.strip():
        return None

    artifact = ToolRecommendationArtifact.model_validate(json.loads(record.content_text))
    discovery_record = session.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
    canvas_record = session.exec(select(CanvasRecord).where(CanvasRecord.session_id == session_id)).first()
    blueprint_record = session.exec(select(BlueprintRecord).where(BlueprintRecord.session_id == session_id)).first()
    if discovery_record is None or canvas_record is None or blueprint_record is None:
        return artifact

    approved_define_record = session.exec(
        select(JourneyStageArtifactRecord)
        .where(
            JourneyStageArtifactRecord.session_id == session_id,
            JourneyStageArtifactRecord.stage_key == "define",
            JourneyStageArtifactRecord.state.in_((JourneyArtifactState.approved, JourneyArtifactState.approved_legacy)),
        )
        .order_by(JourneyStageArtifactRecord.version_number.desc(), JourneyStageArtifactRecord.created_at.desc())
    ).first()
    approved_design_record = session.exec(
        select(JourneyStageArtifactRecord)
        .where(
            JourneyStageArtifactRecord.session_id == session_id,
            JourneyStageArtifactRecord.stage_key == "design",
            JourneyStageArtifactRecord.state.in_((JourneyArtifactState.approved, JourneyArtifactState.approved_legacy)),
        )
        .order_by(JourneyStageArtifactRecord.version_number.desc(), JourneyStageArtifactRecord.created_at.desc())
    ).first()

    return annotate_tool_recommendation_status(
        artifact,
        discovery=_hydrate_discovery_record(discovery_record),
        canvas=_hydrate_canvas_record(canvas_record),
        blueprint=_hydrate_blueprint_record(blueprint_record),
        definition_artifact=(
            RequirementsDefinitionOutput.model_validate(approved_define_record.proposal_payload)
            if approved_define_record is not None
            else None
        ),
        design_artifact=(
            DesignRecommendationArtifact.model_validate(approved_design_record.proposal_payload)
            if approved_design_record is not None
            else None
        ),
        current_blueprint_version=_latest_blueprint_version_number(session, session_id),
    )


def build_artifact_record_entry(record: ArtifactRegistryRecord) -> ArtifactRecordEntry:
    return ArtifactRecordEntry(
        id=record.id,
        blueprint_version_number=record.blueprint_version_number,
        artifact_key=record.artifact_key,
        artifact_title=record.artifact_title,
        artifact_kind=record.artifact_kind,
        stage=record.stage,
        source_action=record.source_action,
        export_format=record.export_format,
        content_text=record.content_text,
        content_hash=record.content_hash,
        artifact_metadata=record.artifact_metadata,
        created_at=record.created_at,
    )


def build_metric_snapshot_entry(record: MetricSnapshotRecord) -> MetricSnapshotEntry:
    return MetricSnapshotEntry(
        id=record.id,
        source_action=record.source_action,
        cost_estimate_usd=record.cost_estimate_usd,
        total_duration_ms=record.total_duration_ms,
        error_count=record.error_count,
        warning_count=record.warning_count,
        approvals_pending=record.approvals_pending,
        approvals_resolved=record.approvals_resolved,
        regenerations_count=record.regenerations_count,
        needs_review_count=record.needs_review_count,
        latest_evaluation_score=record.latest_evaluation_score,
        latest_evaluation_status=record.latest_evaluation_status,
        export_count=record.export_count,
        artifact_count=record.artifact_count,
        created_at=record.created_at,
    )


def build_alert_event_entry(record: AlertEventRecord) -> AlertEventEntry:
    return AlertEventEntry(
        id=record.id,
        alert_key=record.alert_key,
        severity=record.severity,
        title=record.title,
        message=record.message,
        status=record.status,
        evidence=record.evidence,
        created_at=record.created_at,
        updated_at=record.updated_at,
        resolved_at=record.resolved_at,
    )


def build_integration_status_entry(record: IntegrationStatusRecord) -> IntegrationStatusEntry:
    return IntegrationStatusEntry(
        id=record.id,
        integration_key=record.integration_key,
        label=record.label,
        status=record.status,
        configured=record.configured,
        reachable=record.reachable,
        detail=record.detail,
        checked_at=record.checked_at,
    )


def _estimate_skill_run_cost(run: SkillRunRecord) -> float:
    has_llm_evidence = any(item.get("source") == "llm_inference" for item in run.evidence)
    if not has_llm_evidence:
        return 0.0
    if run.skill_key == "blueprint_generation_skill":
        return 0.03
    if run.skill_key in {"discovery_skill", "lean_scope_skill"}:
        return 0.01
    return 0.008


def build_metric_snapshot(
    session: Session,
    *,
    session_id: UUID,
    source_action: str,
) -> MetricSnapshotRecord:
    skill_runs = session.exec(select(SkillRunRecord).where(SkillRunRecord.session_id == session_id)).all()
    validations = session.exec(
        select(ValidationReportRecord).where(ValidationReportRecord.session_id == session_id)
    ).all()
    approvals = session.exec(select(ApprovalGateRecord).where(ApprovalGateRecord.session_id == session_id)).all()
    activity = session.exec(select(ExecutionLogRecord).where(ExecutionLogRecord.session_id == session_id)).all()
    evaluation_runs = session.exec(
        select(EvaluationRunRecord)
        .where(EvaluationRunRecord.session_id == session_id)
        .order_by(EvaluationRunRecord.created_at.desc())
    ).all()
    artifacts = session.exec(select(ArtifactRegistryRecord).where(ArtifactRegistryRecord.session_id == session_id)).all()

    latest_evaluation = evaluation_runs[0] if evaluation_runs else None
    return MetricSnapshotRecord(
        session_id=session_id,
        source_action=source_action,
        cost_estimate_usd=round(sum(_estimate_skill_run_cost(item) for item in skill_runs), 4),
        total_duration_ms=sum(item.duration_ms for item in skill_runs),
        error_count=sum(1 for item in activity if item.status == ArtifactStatus.failed)
        + sum(1 for item in validations if item.status == ArtifactStatus.failed)
        + sum(1 for item in skill_runs if item.status == ArtifactStatus.failed),
        warning_count=sum(len(item.warnings) for item in validations) + sum(len(item.warnings) for item in skill_runs),
        approvals_pending=sum(1 for item in approvals if item.status == ApprovalStatus.pending),
        approvals_resolved=sum(1 for item in approvals if item.status in {ApprovalStatus.approved, ApprovalStatus.rejected}),
        regenerations_count=sum(1 for item in skill_runs if item.source_action.startswith("rerun:")),
        needs_review_count=sum(1 for item in activity if item.status == ArtifactStatus.needs_review)
        + sum(1 for item in validations if item.status == ArtifactStatus.needs_review)
        + sum(1 for item in skill_runs if item.status == ArtifactStatus.needs_review)
        + sum(1 for item in evaluation_runs if item.status == ArtifactStatus.needs_review),
        latest_evaluation_score=latest_evaluation.overall_score if latest_evaluation is not None else None,
        latest_evaluation_status=latest_evaluation.status if latest_evaluation is not None else "",
        export_count=sum(1 for item in artifacts if item.artifact_kind == "export"),
        artifact_count=len(artifacts),
    )


def persist_metric_snapshot(
    session: Session,
    *,
    session_id: UUID,
    source_action: str,
) -> MetricSnapshotRecord:
    snapshot = build_metric_snapshot(session, session_id=session_id, source_action=source_action)
    session.add(snapshot)
    session.flush()
    return snapshot


def record_delivery_artifacts(
    session: Session,
    *,
    session_id: UUID,
    blueprint_version_number: int | None,
    source_action: str,
    stage: SessionStage,
    blueprint: BlueprintArtifact,
) -> None:
    for deliverable in blueprint.delivery_package.deliverables:
        content_text = deliverable.content_markdown or ""
        session.add(
            ArtifactRegistryRecord(
                session_id=session_id,
                blueprint_version_number=blueprint_version_number,
                artifact_key=deliverable.key,
                artifact_title=deliverable.title,
                artifact_kind="delivery_package",
                stage=stage,
                source_action=source_action,
                export_format="markdown",
                content_text=content_text,
                content_hash=_hash_text(content_text),
                artifact_metadata={
                    "summary": deliverable.summary,
                    "blueprint_version_number": blueprint_version_number,
                },
            )
        )
    session.flush()


def record_export_artifact(
    session: Session,
    *,
    session_id: UUID,
    blueprint_version_number: int | None,
    artifact_key: str,
    artifact_title: str,
    export_format: str,
    content_text: str,
    source_action: str,
    artifact_metadata_extra: dict[str, Any] | None = None,
) -> ArtifactRegistryRecord:
    metadata = {
        "export_format": export_format,
        "content_length": len(content_text),
    }
    if artifact_metadata_extra:
        metadata.update(artifact_metadata_extra)
    record = ArtifactRegistryRecord(
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        artifact_key=artifact_key,
        artifact_title=artifact_title,
        artifact_kind="export",
        stage=SessionStage.ready_for_export,
        source_action=source_action,
        export_format=export_format,
        content_text=content_text,
        content_hash=_hash_text(content_text),
        artifact_metadata=metadata,
    )
    session.add(record)
    session.flush()
    return record


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _related_gap_keys_for_acp_path(preview: ACPPreview, path: str) -> list[str]:
    gaps = preview.construction_readiness.gaps
    related = [gap.gap_key for gap in gaps if path in gap.evidence_paths]

    if path in {
        "ACP/construction-readiness/overview.yaml",
        "ACP/construction-readiness/resolution-workflow.yaml",
        "ACP/prompts/builder-handoff.md",
        "ACP/prompts/gap-closure.md",
    }:
        related.extend(gap.gap_key for gap in gaps)
    elif path == "ACP/construction-readiness/blocking-gaps.yaml":
        related.extend(gap.gap_key for gap in gaps if gap.severity == "blocking")
    elif path == "ACP/construction-readiness/open-questions.yaml":
        related.extend(gap.gap_key for gap in gaps if gap.questions)
    elif path == "ACP/construction-readiness/assumptions.yaml":
        related.extend(gap.gap_key for gap in gaps if gap.current_assumptions)
    elif path == "ACP/construction-readiness/external-dependencies.yaml":
        related.extend(gap.gap_key for gap in gaps if gap.domain in {"integrations", "deployment", "runtime", "knowledge", "package"})
    elif path == "ACP/construction-readiness/required-api-contracts.yaml":
        related.extend(gap.gap_key for gap in gaps if gap.domain == "integrations")
    elif path == "ACP/construction-readiness/deployment-decisions-needed.yaml":
        related.extend(gap.gap_key for gap in gaps if gap.domain in {"deployment", "runtime"})
    elif path in {
        "ACP/estimation/estimation-report.json",
        "ACP/estimation/estimation-report.md",
        "ACP/estimation/sensitivity-drivers.yaml",
    }:
        related.extend(gap.gap_key for gap in gaps)
    elif path == "ACP/estimation/assumptions.yaml":
        related.extend(gap.gap_key for gap in gaps if gap.current_assumptions)

    return _dedupe_preserve_order(related)


def record_acp_preview_artifacts(
    session: Session,
    *,
    session_id: UUID,
    preview: ACPPreview,
    source_action: str,
) -> list[ArtifactRegistryRecord]:
    records: list[ArtifactRegistryRecord] = []
    blueprint_version_number = preview.blueprint_version_number
    stage = SessionStage.ready_for_export

    for file in preview.files:
        artifact_kind = "acp_manifest" if file.path == preview.manifest_path else "acp_file"
        is_construction_readiness_artifact = file.path.startswith("ACP/construction-readiness/")
        is_prompt_artifact = file.path.startswith("ACP/prompts/")
        is_estimation_artifact = file.path.startswith("ACP/estimation/")
        record = ArtifactRegistryRecord(
            session_id=session_id,
            blueprint_version_number=blueprint_version_number,
            artifact_key=file.path,
            artifact_title=file.title or file.path.rsplit("/", 1)[-1],
            artifact_kind=artifact_kind,
            stage=stage,
            source_action=source_action,
            export_format=file.format,
            content_text=file.content_text,
            content_hash=file.content_hash or _hash_text(file.content_text),
            artifact_metadata={
                "acp_path": file.path,
                "acp_domain": file.domain,
                "acp_status": file.status,
                "source_sections": file.source_sections,
                "missing_fields": file.missing_fields,
                "warnings": file.warnings,
                "bundle_version": preview.package_version,
                "lineage_scope": (
                    "construction_readiness"
                    if is_construction_readiness_artifact
                    else "prompt"
                    if is_prompt_artifact
                    else "estimation"
                    if is_estimation_artifact
                    else "core"
                ),
                "is_construction_readiness_artifact": is_construction_readiness_artifact,
                "is_prompt_artifact": is_prompt_artifact,
                "is_estimation_artifact": is_estimation_artifact,
                "related_gap_keys": _related_gap_keys_for_acp_path(preview, file.path),
            },
        )
        session.add(record)
        session.flush()
        records.append(record)

    preview_payload = serialize_json_document(preview.model_dump(mode="json"))
    preview_record = ArtifactRegistryRecord(
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        artifact_key="acp_preview",
        artifact_title="ACP preview",
        artifact_kind="acp_preview",
        stage=stage,
        source_action=source_action,
        export_format="json",
        content_text=preview_payload,
        content_hash=_hash_text(preview_payload),
        artifact_metadata={
            "bundle_version": preview.package_version,
            "manifest_path": preview.manifest_path,
            "file_count": len(preview.files),
            "completeness_percent": preview.validation.completeness_percent,
            "overall_status": preview.validation.overall_status,
            "can_export_zip": preview.validation.can_export_zip,
            "construction_readiness_status": preview.construction_readiness.overall_status,
            "can_start_build": preview.construction_readiness.can_start_build,
            "blocking_gaps": preview.construction_readiness.blocking_gaps,
            "open_questions": preview.construction_readiness.open_questions,
            "assumptions_count": preview.construction_readiness.assumptions_count,
        },
    )
    session.add(preview_record)
    session.flush()
    records.append(preview_record)
    return records


def record_estimation_artifact(
    session: Session,
    *,
    session_id: UUID,
    blueprint_version_number: int | None,
    stage: SessionStage,
    source_action: str,
    estimation_report: EstimationReportArtifact,
) -> ArtifactRegistryRecord:
    payload = serialize_json_document(estimation_report.model_dump(mode="json"))
    record = ArtifactRegistryRecord(
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        artifact_key="ACP/estimation/estimation-report.json",
        artifact_title="Estimation report",
        artifact_kind="estimation_report",
        stage=stage,
        source_action=source_action,
        export_format="json",
        content_text=payload,
        content_hash=_hash_text(payload),
        artifact_metadata={
            "maturity_stage": estimation_report.maturity_stage,
            "confidence_score": estimation_report.confidence.score,
            "confidence_label": estimation_report.confidence.label,
            "active_provider": estimation_report.agentic.active_provider,
            "pricing_policy": estimation_report.agentic.pricing_policy,
            "provider_model": estimation_report.agentic.provider_model,
            "provider_runtime_cost_total_usd": estimation_report.agentic.provider_runtime_cost_total_usd,
            "source_artifacts": estimation_report.source_artifacts,
            "estimated_hours_total": estimation_report.traditional.estimated_hours_total,
            "estimated_cost": estimation_report.traditional.estimated_cost,
        },
    )
    session.add(record)
    session.flush()
    return record


def _resolve_operational_runtime_settings(
    session: Session,
    *,
    session_id: UUID | None = None,
    workspace_id: UUID | None = None,
) -> LLMRuntimeSettings:
    if workspace_id is not None:
        return load_effective_runtime_settings(session, workspace_id)
    if session_id is not None:
        return load_effective_runtime_settings_for_session(session, session_id)
    return load_platform_runtime_defaults(session)


def _build_openai_status(runtime_settings: LLMRuntimeSettings) -> dict[str, Any]:
    configured = runtime_settings.openai.api_key_configured
    reachable = runtime_settings.openai.available
    status = "healthy" if configured and reachable else "degraded"
    detail = (
        f"provider=openai mode=responses "
        f"fast={runtime_settings.openai.fast_model} reasoning={runtime_settings.openai.reasoning_model}"
    )
    return {
        "integration_key": "openai",
        "label": "OpenAI",
        "status": status,
        "configured": configured,
        "reachable": reachable,
        "detail": detail,
    }


def _build_deepseek_status(runtime_settings: LLMRuntimeSettings) -> dict[str, Any]:
    configured = runtime_settings.deepseek.api_key_configured
    reachable = runtime_settings.deepseek.available
    status = "healthy" if configured and reachable else "degraded"
    detail = (
        f"provider=deepseek mode=chat_completions "
        f"base_url={runtime_settings.deepseek.base_url} "
        f"fast={runtime_settings.deepseek.fast_model} "
        f"reasoning={runtime_settings.deepseek.reasoning_model} "
        f"effort={runtime_settings.deepseek.reasoning_effort}"
    )
    return {
        "integration_key": "deepseek",
        "label": "DeepSeek",
        "status": status,
        "configured": configured,
        "reachable": reachable,
        "detail": detail,
    }


def _build_codex_local_status(runtime_settings: LLMRuntimeSettings) -> dict[str, Any]:
    configured = bool(runtime_settings.codex_local.command and runtime_settings.codex_local.model)
    reachable = runtime_settings.codex_local.available
    status = "healthy" if configured and reachable else "degraded"
    profile_token = runtime_settings.codex_local.profile or "default"
    detail = (
        f"provider=codex_local mode=local_exec "
        f"model={runtime_settings.codex_local.model} command={runtime_settings.codex_local.command} profile={profile_token}"
    )
    return {
        "integration_key": "codex_local",
        "label": "Codex local",
        "status": status,
        "configured": configured,
        "reachable": reachable,
        "detail": detail,
    }


def _build_active_llm_runtime_status(runtime_settings: LLMRuntimeSettings) -> dict[str, Any]:
    service = build_builder_service(runtime_settings)
    summary = service.provider_summary()
    configured = bool(summary.get("configured"))
    reachable = bool(summary.get("sdk_ready"))
    status = "healthy" if configured and reachable else "degraded"
    detail = (
        f"provider={summary.get('provider')} mode={summary.get('mode')} "
        f"backend={summary.get('execution_backend')} "
        f"knowledge={runtime_settings.knowledge_access_backend.value} "
        f"fast={summary.get('fast_model')} reasoning={summary.get('reasoning_model')}"
    )
    return {
        "integration_key": "llm_runtime",
        "label": f"LLM activo ({runtime_settings.active_provider.value})",
        "status": status,
        "configured": configured,
        "reachable": reachable,
        "detail": detail,
    }


def build_minimal_health_payload(session: Session) -> dict[str, Any]:
    runtime_settings = _resolve_operational_runtime_settings(session)
    summary = build_builder_service(runtime_settings).provider_summary()
    return {
        "status": "ok",
        "checked_at": utc_now(),
        "llm": {
            "provider": summary.get("provider"),
            "mode": summary.get("mode"),
            "configured": bool(summary.get("configured")),
            "sdk_ready": bool(summary.get("sdk_ready")),
        },
        "runtime": {
            "scope": "platform_default",
            "scope_detail": (
                "Pulso publico sin contexto de workspace o sesion; los overrides efectivos solo se resuelven "
                "en rutas autenticadas por workspace."
            ),
            "active_provider": runtime_settings.active_provider.value,
            "agent_execution_backend": runtime_settings.agent_execution_backend.value,
            "knowledge_access_backend": runtime_settings.knowledge_access_backend.value,
        },
    }


def get_codex_runtime_status(
    session: Session,
    *,
    workspace_id: UUID | None = None,
) -> dict[str, Any]:
    runtime_settings = _resolve_operational_runtime_settings(session, workspace_id=workspace_id)
    service = CodexExecutionService(runtime_settings)
    return service.get_runtime_status()


def _build_database_status(session: Session) -> dict[str, Any]:
    settings = get_settings()
    bind = session.get_bind()
    dialect_name = bind.dialect.name if bind is not None else "unknown"
    configured = bool(settings.database_url)
    reachable = True
    detail = f"dialect={dialect_name}"
    try:
        session.connection().exec_driver_sql("SELECT 1")
    except Exception as exc:  # pragma: no cover - exercised by integration route failures
        reachable = False
        detail = f"dialect={dialect_name} error={exc}"
    return {
        "integration_key": "postgresql",
        "label": "PostgreSQL",
        "status": "healthy" if configured and reachable else "degraded",
        "configured": configured,
        "reachable": reachable,
        "detail": detail,
    }


def _build_auth_status(session: Session) -> dict[str, Any]:
    settings = get_settings()
    active_users = session.exec(select(UserRecord).where(UserRecord.is_active == True)).all()  # noqa: E712
    configured = bool(settings.local_admin_email and settings.local_admin_password)
    reachable = len(active_users) > 0
    detail = f"active_users={len(active_users)} seed={settings.local_admin_email}"
    return {
        "integration_key": "local_auth",
        "label": "Auth local",
        "status": "healthy" if configured and reachable else "degraded",
        "configured": configured,
        "reachable": reachable,
        "detail": detail,
    }


def sync_integration_statuses(session: Session, *, session_id: UUID) -> list[IntegrationStatusRecord]:
    runtime_settings = _resolve_operational_runtime_settings(session, session_id=session_id)
    payloads = [
        _build_active_llm_runtime_status(runtime_settings),
        _build_openai_status(runtime_settings),
        _build_deepseek_status(runtime_settings),
        _build_codex_local_status(runtime_settings),
        _build_database_status(session),
        _build_auth_status(session),
    ]
    records: list[IntegrationStatusRecord] = []
    for payload in payloads:
        record = session.exec(
            select(IntegrationStatusRecord).where(
                IntegrationStatusRecord.session_id == session_id,
                IntegrationStatusRecord.integration_key == payload["integration_key"],
            )
        ).first()
        if record is None:
            record = IntegrationStatusRecord(session_id=session_id, **payload)
        else:
            record.label = payload["label"]
            record.status = payload["status"]
            record.configured = payload["configured"]
            record.reachable = payload["reachable"]
            record.detail = payload["detail"]
            record.checked_at = utc_now()
        session.add(record)
        session.flush()
        records.append(record)
    return records


def _derive_alert_payloads(
    session: Session,
    *,
    session_id: UUID,
    metrics: MetricSnapshotRecord,
) -> list[dict[str, Any]]:
    activity = session.exec(
        select(ExecutionLogRecord)
        .where(ExecutionLogRecord.session_id == session_id)
        .order_by(ExecutionLogRecord.created_at.desc())
    ).all()
    approvals = session.exec(select(ApprovalGateRecord).where(ApprovalGateRecord.session_id == session_id)).all()

    alerts: list[dict[str, Any]] = []
    recent_failed_logs = [item for item in activity[:8] if item.status == ArtifactStatus.failed]
    if len(recent_failed_logs) >= 2:
        alerts.append(
            {
                "alert_key": "repeated_failures",
                "severity": "high",
                "title": "Fallos repetidos",
                "message": "La sesion acumula varios fallos recientes y necesita revision antes de seguir escalando.",
                "evidence": [item.message for item in recent_failed_logs[:3]],
            }
        )

    if metrics.needs_review_count >= 3:
        alerts.append(
            {
                "alert_key": "needs_review_pressure",
                "severity": "medium",
                "title": "Presion de revision",
                "message": "Se acumulan estados needs_review en la sesion y conviene cerrar deuda antes de nuevas corridas.",
                "evidence": [f"needs_review_count={metrics.needs_review_count}"],
            }
        )

    if metrics.latest_evaluation_score is not None and (
        metrics.latest_evaluation_score < 70 or metrics.latest_evaluation_status == ArtifactStatus.failed
    ):
        alerts.append(
            {
                "alert_key": "evaluation_degraded",
                "severity": "high",
                "title": "Evaluacion degradada",
                "message": "La ultima corrida de evaluacion quedo por debajo del umbral recomendado para promotion.",
                "evidence": [f"latest_evaluation_score={metrics.latest_evaluation_score}"],
            }
        )

    pending_approvals = [item for item in approvals if item.status == ApprovalStatus.pending]
    if pending_approvals:
        oldest_pending = min(item.created_at for item in pending_approvals)
        expired = oldest_pending <= metrics.created_at - timedelta(hours=24)
        alerts.append(
            {
                "alert_key": "approvals_unresolved",
                "severity": "high" if expired else "medium",
                "title": "Approvals sin resolver",
                "message": "Hay approval gates pendientes que bloquean promotion del blueprint.",
                "evidence": [f"pending_approvals={len(pending_approvals)}"],
            }
        )

    session_record = session.get(SessionRecord, session_id)
    if session_record is not None and is_feature_flag_enabled(
        session,
        FEATURE_FLAG_TOOL_RECOMMENDATION,
        workspace_id=session_record.workspace_id,
    ):
        latest_recommendation = _load_latest_tool_recommendation_status(session, session_id)
        if latest_recommendation is not None:
            if latest_recommendation.is_stale:
                alerts.append(
                    {
                        "alert_key": "tool_recommendation_stale",
                        "severity": "high",
                        "title": "Herramientas desactualizadas",
                        "message": "La propuesta aprobada de Herramientas quedo obsoleta frente al contexto actual del blueprint.",
                        "evidence": list(latest_recommendation.stale_reasons) or ["tool_recommendation_context_changed"],
                    }
                )
            elif latest_recommendation.approved_tools_digest is None:
                alerts.append(
                    {
                        "alert_key": (
                            "tool_recommendation_blocked"
                            if latest_recommendation.evaluation.promotion_blocked
                            else "tool_recommendation_pending_promotion"
                        ),
                        "severity": "high" if latest_recommendation.evaluation.promotion_blocked else "medium",
                        "title": (
                            "Herramientas bloqueadas"
                            if latest_recommendation.evaluation.promotion_blocked
                            else "Herramientas pendientes de promocion"
                        ),
                        "message": (
                            "La propuesta de Herramientas sigue bloqueada por findings y no debe pasar a Memoria."
                            if latest_recommendation.evaluation.promotion_blocked
                            else "Existe una propuesta de Herramientas sin promocionar a blueprint.tools."
                        ),
                        "evidence": (
                            [item.finding_key for item in latest_recommendation.evaluation.findings if item.severity == "blocking"]
                            or [latest_recommendation.evaluation.summary]
                        ),
                    }
                )

    return alerts


def sync_alert_events(
    session: Session,
    *,
    session_id: UUID,
    metrics: MetricSnapshotRecord,
) -> list[AlertEventRecord]:
    desired_alerts = _derive_alert_payloads(session, session_id=session_id, metrics=metrics)
    desired_by_key = {item["alert_key"]: item for item in desired_alerts}
    existing = session.exec(select(AlertEventRecord).where(AlertEventRecord.session_id == session_id)).all()
    existing_by_key = {item.alert_key: item for item in existing}

    for alert_key, payload in desired_by_key.items():
        record = existing_by_key.get(alert_key)
        if record is None:
            record = AlertEventRecord(session_id=session_id, **payload)
        else:
            record.severity = payload["severity"]
            record.title = payload["title"]
            record.message = payload["message"]
            record.status = ACTIVE_ALERT_STATUS
            record.evidence = payload["evidence"]
            record.updated_at = metrics.created_at
            record.resolved_at = None
        session.add(record)

    for alert_key, record in existing_by_key.items():
        if alert_key in desired_by_key or record.status != ACTIVE_ALERT_STATUS:
            continue
        record.status = RESOLVED_ALERT_STATUS
        record.updated_at = metrics.created_at
        record.resolved_at = metrics.created_at
        session.add(record)

    session.flush()
    return session.exec(
        select(AlertEventRecord)
        .where(AlertEventRecord.session_id == session_id)
        .order_by(AlertEventRecord.updated_at.desc(), AlertEventRecord.created_at.desc())
    ).all()


def capture_operational_state(
    session: Session,
    *,
    session_id: UUID,
    source_action: str,
) -> MetricSnapshotRecord:
    sync_integration_statuses(session, session_id=session_id)
    metrics = persist_metric_snapshot(session, session_id=session_id, source_action=source_action)
    sync_alert_events(session, session_id=session_id, metrics=metrics)
    session.flush()
    return metrics


def build_monitoring_workspace(
    *,
    metric_records: list[MetricSnapshotRecord],
    alert_records: list[AlertEventRecord],
    recent_error_records: list[ExecutionLogRecord],
    integration_records: list[IntegrationStatusRecord],
    memory_observability: MemoryObservabilityReport | None = None,
    release_observability: MonitoringReleaseObservability | None = None,
) -> MonitoringWorkspace:
    return MonitoringWorkspace(
        current_metrics=build_metric_snapshot_entry(metric_records[0]) if metric_records else None,
        history=[build_metric_snapshot_entry(item) for item in metric_records],
        alerts=[build_alert_event_entry(item) for item in alert_records],
        recent_errors=[
            ExecutionLogEntry(
                stage=item.stage,
                status=item.status,
                message=item.message,
                payload=item.payload,
                created_at=item.created_at,
            )
            for item in recent_error_records
        ],
        integrations=[build_integration_status_entry(item) for item in integration_records],
        memory_observability=memory_observability,
        release_observability=release_observability,
    )


def filter_artifact_records(
    records: list[ArtifactRegistryRecord],
    *,
    query: str = "",
    artifact_kind: str = "",
    stage: SessionStage | None = None,
    blueprint_version_number: int | None = None,
    date_from: str = "",
    date_to: str = "",
) -> list[ArtifactRegistryRecord]:
    query_text = query.strip().lower()
    date_from_text = date_from.strip()
    date_to_text = date_to.strip()

    filtered: list[ArtifactRegistryRecord] = []
    for record in records:
        if artifact_kind and record.artifact_kind != artifact_kind:
            continue
        if stage is not None and record.stage != stage:
            continue
        if blueprint_version_number is not None and record.blueprint_version_number != blueprint_version_number:
            continue
        if date_from_text and record.created_at.date().isoformat() < date_from_text:
            continue
        if date_to_text and record.created_at.date().isoformat() > date_to_text:
            continue
        if query_text:
            haystack = " ".join(
                [
                    record.artifact_key,
                    record.artifact_title,
                    record.artifact_kind,
                    record.source_action,
                    record.content_text,
                ]
            ).lower()
            if query_text not in haystack:
                continue
        filtered.append(record)
    return filtered
