from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlmodel import Session, select

from app.models import (
    ACPBuildRunRecord,
    ACPBuildRunResponse,
    ACPPhaseCommandRequest,
    ACPPhaseDefinitionResponse,
    ACPPhaseRunRecord,
    ACPPhaseRunResponse,
    ACPPreview,
    ACPValidationReport,
    ACPWorkflowRunStatus,
    ACPWorkspaceResponse,
    CommercialAccessSnapshotV2,
    ConstructionReadinessReport,
    SessionRecord,
    SessionSnapshot,
    UserRecord,
    utc_now,
)
from app.services.product_processing.journey_state_machine_service import load_persisted_journey_state_machine


@dataclass(frozen=True)
class ACPPhaseDefinition:
    key: str
    label: str
    objective: str
    order: int


ACP_PHASES: tuple[ACPPhaseDefinition, ...] = (
    ACPPhaseDefinition(
        key="blueprint_validation",
        label="Validacion del Blueprint",
        objective="Verificar que el Blueprint base sea consistente antes de construir el paquete tecnico.",
        order=1,
    ),
    ACPPhaseDefinition(
        key="test_suite",
        label="Diseno del Test Suite",
        objective="Preparar escenarios, rubricas y criterios verificables para validar el agente.",
        order=2,
    ),
    ACPPhaseDefinition(
        key="gap_classification",
        label="Clasificacion de GAPs",
        objective="Separar brechas de diseno, implementacion y dependencias externas sin bloquear indebidamente.",
        order=3,
    ),
    ACPPhaseDefinition(
        key="implementation_questions",
        label="Preguntas de implementacion",
        objective="Documentar decisiones humanas inevitables con opciones, impacto y responsables.",
        order=4,
    ),
    ACPPhaseDefinition(
        key="package_build",
        label="Construccion del paquete",
        objective="Materializar prompts, contratos, herramientas, memoria, workflows y artefactos portables.",
        order=5,
    ),
    ACPPhaseDefinition(
        key="conformance_export",
        label="Conformance y exportacion",
        objective="Comprobar readiness, conformance y preparacion del ACP para descarga controlada.",
        order=6,
    ),
)

COMPLETED_STATUSES = {ACPWorkflowRunStatus.completed, ACPWorkflowRunStatus.completed_with_observations}


def phase_definition_responses() -> list[ACPPhaseDefinitionResponse]:
    return [
        ACPPhaseDefinitionResponse(key=item.key, label=item.label, objective=item.objective, order=item.order)
        for item in ACP_PHASES
    ]


def _definition_by_key(phase_key: str) -> ACPPhaseDefinition:
    for definition in ACP_PHASES:
        if definition.key == phase_key:
            return definition
    raise ValueError(f"Unknown ACP phase: {phase_key}")


def _blueprint_version_number(snapshot: SessionSnapshot) -> int | None:
    versions = getattr(snapshot, "blueprint_versions", []) or []
    if not versions:
        return None
    return max(int(getattr(item, "version_number", 0) or 0) for item in versions) or None


def _run_idempotency_key(record: SessionRecord, blueprint_version_number: int | None) -> str:
    return f"{record.id}:{blueprint_version_number or 0}:acp-workspace"


def _phase_order_keys() -> list[str]:
    return [item.key for item in ACP_PHASES]


def ensure_acp_run(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    snapshot: SessionSnapshot,
) -> ACPBuildRunRecord:
    blueprint_version_number = _blueprint_version_number(snapshot)
    idempotency_key = _run_idempotency_key(record, blueprint_version_number)

    stale_runs = db.exec(
        select(ACPBuildRunRecord).where(
            ACPBuildRunRecord.workspace_id == record.workspace_id,
            ACPBuildRunRecord.session_id == record.id,
            ACPBuildRunRecord.blueprint_version_number != blueprint_version_number,
            ACPBuildRunRecord.status.notin_((ACPWorkflowRunStatus.completed, ACPWorkflowRunStatus.canceled)),
        )
    ).all()
    for stale_run in stale_runs:
        stale_run.status = ACPWorkflowRunStatus.stale
        stale_run.updated_at = utc_now()
        db.add(stale_run)

    run = db.exec(
        select(ACPBuildRunRecord).where(
            ACPBuildRunRecord.workspace_id == record.workspace_id,
            ACPBuildRunRecord.session_id == record.id,
            ACPBuildRunRecord.idempotency_key == idempotency_key,
        )
    ).first()
    if run is None:
        now = utc_now()
        run = ACPBuildRunRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            created_by_user_id=current_user.id,
            blueprint_version_number=blueprint_version_number,
            status=ACPWorkflowRunStatus.not_started,
            current_phase_key=ACP_PHASES[0].key,
            phase_order=_phase_order_keys(),
            progress_percent=0,
            idempotency_key=idempotency_key,
            started_at=now,
        )
        db.add(run)
        db.flush()
    _ensure_phase_rows(db, run)
    return run


def _ensure_phase_rows(db: Session, run: ACPBuildRunRecord) -> None:
    existing = {
        item.phase_key: item
        for item in db.exec(select(ACPPhaseRunRecord).where(ACPPhaseRunRecord.run_id == run.id)).all()
    }
    for definition in ACP_PHASES:
        if definition.key in existing:
            continue
        db.add(
            ACPPhaseRunRecord(
                run_id=run.id,
                workspace_id=run.workspace_id,
                session_id=run.session_id,
                phase_key=definition.key,
                phase_label=definition.label,
                phase_order=definition.order,
                status=ACPWorkflowRunStatus.not_started,
                idempotency_key=f"{run.id}:{definition.key}",
            )
        )
    db.flush()


def _phase_rows(db: Session, run: ACPBuildRunRecord) -> list[ACPPhaseRunRecord]:
    _ensure_phase_rows(db, run)
    return db.exec(
        select(ACPPhaseRunRecord).where(ACPPhaseRunRecord.run_id == run.id).order_by(ACPPhaseRunRecord.phase_order)
    ).all()


def _preview_file_refs(preview: ACPPreview, *, domain: str | None = None, prefix: str | None = None) -> list[dict[str, Any]]:
    files = []
    for item in preview.files:
        if domain and item.domain != domain:
            continue
        if prefix and not item.path.startswith(prefix):
            continue
        files.append({"path": item.path, "format": item.format, "status": item.status, "hash": item.content_hash})
    return files


def _input_refs(phase_key: str, preview: ACPPreview) -> list[dict[str, Any]]:
    refs = [
        {"kind": "blueprint_version", "value": preview.blueprint_version_number},
        {"kind": "manifest", "path": preview.manifest_path},
    ]
    if phase_key in {"gap_classification", "implementation_questions", "conformance_export"}:
        refs.append({"kind": "construction_readiness", "status": preview.construction_readiness.overall_status})
    if phase_key in {"package_build", "conformance_export"}:
        refs.append({"kind": "acp_files", "count": len(preview.files)})
    return refs


def _output_refs(phase_key: str, preview: ACPPreview, readiness: ConstructionReadinessReport) -> list[dict[str, Any]]:
    if phase_key == "blueprint_validation":
        return [
            {
                "kind": "validation_report",
                "status": preview.validation.overall_status,
                "issue_count": len(preview.validation.issues),
                "completeness_percent": preview.validation.completeness_percent,
            }
        ]
    if phase_key == "test_suite":
        refs = _preview_file_refs(preview, domain="evaluation")
        return [{"kind": "test_suite_files", "count": len(refs), "files": refs[:12]}]
    if phase_key == "gap_classification":
        return [
            {
                "kind": "classified_gaps",
                "blocking_gaps": readiness.blocking_gaps,
                "open_questions": readiness.open_questions,
                "gaps": [item.model_dump(mode="json", exclude={"questions"}) for item in readiness.gaps[:12]],
            }
        ]
    if phase_key == "implementation_questions":
        questions = [
            question.model_dump(mode="json")
            for gap in readiness.gaps
            for question in gap.questions
            if getattr(question, "question_key", "")
        ]
        return [{"kind": "implementation_questions", "count": len(questions), "questions": questions[:20]}]
    if phase_key == "package_build":
        return [
            {
                "kind": "portable_package",
                "manifest_path": preview.manifest_path,
                "file_count": len(preview.files),
                "launcher_files": _preview_file_refs(preview, prefix="ACP/launcher/"),
            }
        ]
    if phase_key == "conformance_export":
        return [
            {
                "kind": "conformance",
                "can_export_zip": preview.validation.can_export_zip,
                "can_start_build": readiness.can_start_build,
                "readiness": readiness.overall_status,
            }
        ]
    return []


def _warnings_for_phase(phase_key: str, preview: ACPPreview, readiness: ConstructionReadinessReport) -> list[str]:
    warnings: list[str] = []
    if phase_key == "blueprint_validation":
        warnings.extend(item.message for item in preview.validation.issues if item.severity in {"warning", "info"})
    if phase_key == "test_suite" and not _preview_file_refs(preview, domain="evaluation"):
        warnings.append("No se encontraron archivos de evaluacion en el ACP generado.")
    if phase_key in {"gap_classification", "implementation_questions"} and readiness.open_questions:
        warnings.append(f"Hay {readiness.open_questions} pregunta(s) de implementacion abiertas.")
    if phase_key == "package_build":
        incomplete_count = sum(1 for item in preview.files if item.status != "complete")
        if incomplete_count:
            warnings.append(f"{incomplete_count} archivo(s) del ACP requieren revision.")
    return warnings[:20]


def _blockers_for_phase(phase_key: str, preview: ACPPreview, readiness: ConstructionReadinessReport) -> list[dict[str, Any]]:
    if phase_key == "blueprint_validation":
        return [
            {"code": item.code, "message": item.message, "path": item.path, "remediation": item.remediation}
            for item in preview.validation.issues
            if item.blocking or item.severity == "error"
        ]
    if phase_key == "gap_classification":
        return [
            {
                "code": item.gap_key,
                "message": item.summary,
                "stage": item.blocking_stage,
                "remediation": item.remediation,
            }
            for item in readiness.gaps
            if item.severity == "blocking" and item.status == "open"
        ]
    if phase_key == "conformance_export" and not preview.validation.can_export_zip:
        return [{"code": "acp_zip_not_ready", "message": "El ACP aun no puede exportarse como ZIP.", "remediation": "Resolver issues de validacion."}]
    return []


def _phase_status(
    phase_key: str,
    preview: ACPPreview,
    readiness: ConstructionReadinessReport,
    blockers: list[dict[str, Any]],
    warnings: list[str],
) -> ACPWorkflowRunStatus:
    if blockers:
        return ACPWorkflowRunStatus.blocked
    if phase_key == "implementation_questions" and readiness.open_questions:
        return ACPWorkflowRunStatus.waiting_user
    if phase_key == "conformance_export" and not readiness.can_start_build:
        return ACPWorkflowRunStatus.waiting_user
    if warnings:
        return ACPWorkflowRunStatus.completed_with_observations
    return ACPWorkflowRunStatus.completed


class ACPPhaseSequenceError(ValueError):
    def __init__(
        self,
        *,
        phase_key: str,
        blocking_phase_key: str,
        blocking_phase_status: str,
        next_action: str,
    ):
        self.phase_key = phase_key
        self.blocking_phase_key = blocking_phase_key
        self.blocking_phase_status = blocking_phase_status
        self.next_action = next_action
        super().__init__(
            f"Cannot execute ACP phase '{phase_key}'. Previous phase '{blocking_phase_key}' must be completed first (current status: {blocking_phase_status}). Next action: {next_action}"
        )


def run_acp_phase(
    db: Session,
    *,
    run: ACPBuildRunRecord,
    phase_key: str,
    payload: ACPPhaseCommandRequest,
    preview: ACPPreview,
    readiness: ConstructionReadinessReport,
) -> ACPPhaseRunRecord:
    definition = _definition_by_key(phase_key)
    phase = db.exec(
        select(ACPPhaseRunRecord).where(ACPPhaseRunRecord.run_id == run.id, ACPPhaseRunRecord.phase_key == phase_key)
    ).first()
    if phase is None:
        _ensure_phase_rows(db, run)
        phase = db.exec(
            select(ACPPhaseRunRecord).where(ACPPhaseRunRecord.run_id == run.id, ACPPhaseRunRecord.phase_key == phase_key)
        ).one()

    # FIFO sequence enforcement: phase N requires phase N-1 to be completed
    if definition.order > 1 and not payload.force:
        previous_phase_def = next(d for d in ACP_PHASES if d.order == definition.order - 1)
        prev_phase = db.exec(
            select(ACPPhaseRunRecord).where(
                ACPPhaseRunRecord.run_id == run.id,
                ACPPhaseRunRecord.phase_key == previous_phase_def.key,
            )
        ).first()
        if prev_phase is None or prev_phase.status not in COMPLETED_STATUSES:
            prev_status = prev_phase.status.value if prev_phase is not None else "not_started"
            raise ACPPhaseSequenceError(
                phase_key=phase_key,
                blocking_phase_key=previous_phase_def.key,
                blocking_phase_status=prev_status,
                next_action=f"Run and complete phase '{previous_phase_def.key}' before attempting '{phase_key}'.",
            )

    if phase.status in COMPLETED_STATUSES and not payload.force:
        return phase

    now = utc_now()
    blockers = _blockers_for_phase(phase_key, preview, readiness)
    warnings = _warnings_for_phase(phase_key, preview, readiness)
    phase.phase_label = definition.label
    phase.phase_order = definition.order
    phase.status = _phase_status(phase_key, preview, readiness, blockers, warnings)
    phase.attempt_count += 1
    phase.idempotency_key = payload.idempotency_key.strip() or phase.idempotency_key or f"{run.id}:{phase_key}"
    phase.input_refs = _input_refs(phase_key, preview)
    phase.output_refs = _output_refs(phase_key, preview, readiness)
    phase.blockers = blockers
    phase.warnings = warnings
    phase.checkpoints = {
        "source_blueprint_version": preview.blueprint_version_number,
        "preview_file_count": len(preview.files),
        "validation_status": preview.validation.overall_status,
        "readiness_status": readiness.overall_status,
        "generated_at": now.isoformat(),
    }
    phase.started_at = phase.started_at or now
    phase.completed_at = now if phase.status in {*COMPLETED_STATUSES, ACPWorkflowRunStatus.blocked, ACPWorkflowRunStatus.waiting_user} else None
    phase.updated_at = now
    db.add(phase)
    update_run_from_phases(db, run)
    return phase


def update_run_from_phases(db: Session, run: ACPBuildRunRecord) -> ACPBuildRunRecord:
    phases = _phase_rows(db, run)
    completed_count = sum(1 for item in phases if item.status in COMPLETED_STATUSES)
    blocking_phases = [item for item in phases if item.status == ACPWorkflowRunStatus.blocked]
    waiting_phases = [item for item in phases if item.status == ACPWorkflowRunStatus.waiting_user]
    run.progress_percent = round(completed_count / max(1, len(ACP_PHASES)) * 100)
    run.blockers = [blocker for phase in phases for blocker in phase.blockers]
    run.warnings = [warning for phase in phases for warning in phase.warnings][:30]
    run.artifacts = {
        "phase_outputs": {
            phase.phase_key: {"status": phase.status.value, "output_refs": phase.output_refs}
            for phase in phases
            if phase.output_refs
        }
    }
    run.checkpoints = {
        "phase_statuses": {phase.phase_key: phase.status.value for phase in phases},
        "completed_phase_count": completed_count,
        "total_phase_count": len(ACP_PHASES),
    }
    first_active = next((item for item in phases if item.status not in COMPLETED_STATUSES), None)
    run.current_phase_key = first_active.phase_key if first_active else ACP_PHASES[-1].key
    if blocking_phases:
        run.status = ACPWorkflowRunStatus.blocked
    elif waiting_phases:
        run.status = ACPWorkflowRunStatus.waiting_user
    elif completed_count == len(ACP_PHASES):
        run.status = ACPWorkflowRunStatus.completed_with_observations if run.warnings else ACPWorkflowRunStatus.completed
        run.completed_at = run.completed_at or utc_now()
    elif completed_count > 0:
        run.status = ACPWorkflowRunStatus.running
    else:
        run.status = ACPWorkflowRunStatus.not_started
    run.updated_at = utc_now()
    db.add(run)
    return run


def serialize_phase_run(record: ACPPhaseRunRecord | None, definition: ACPPhaseDefinition) -> ACPPhaseRunResponse:
    if record is None:
        return ACPPhaseRunResponse(
            phase_key=definition.key,
            phase_label=definition.label,
            phase_order=definition.order,
        )
    return ACPPhaseRunResponse(
        id=record.id,
        phase_key=record.phase_key,
        phase_label=record.phase_label,
        phase_order=record.phase_order,
        status=record.status,
        attempt_count=record.attempt_count,
        input_refs=record.input_refs,
        output_refs=record.output_refs,
        checkpoints=record.checkpoints,
        blockers=record.blockers,
        warnings=record.warnings,
        started_at=record.started_at,
        completed_at=record.completed_at,
        updated_at=record.updated_at,
    )


def serialize_run(record: ACPBuildRunRecord) -> ACPBuildRunResponse:
    return ACPBuildRunResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        session_id=record.session_id,
        blueprint_version_number=record.blueprint_version_number,
        status=record.status,
        current_phase_key=record.current_phase_key,
        progress_percent=record.progress_percent,
        phase_order=record.phase_order,
        checkpoints=record.checkpoints,
        artifacts=record.artifacts,
        blockers=record.blockers,
        warnings=record.warnings,
        created_at=record.created_at,
        updated_at=record.updated_at,
        completed_at=record.completed_at,
    )


def build_acp_workspace_response(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    readiness: ConstructionReadinessReport,
    access: CommercialAccessSnapshotV2,
) -> ACPWorkspaceResponse:
    run = _load_current_acp_run(db, record=record, snapshot=snapshot)
    phase_map = {item.phase_key: item for item in _phase_rows(db, run)} if run is not None else {}
    if run is None:
        run_response = ACPBuildRunResponse(
            workspace_id=record.workspace_id,
            session_id=record.id,
            blueprint_version_number=_blueprint_version_number(snapshot),
            current_phase_key=ACP_PHASES[0].key,
            phase_order=_phase_order_keys(),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        next_action = "Iniciar la validacion del Blueprint para crear la ejecucion ACP."
    else:
        run_response = serialize_run(run)
        next_action = "Ejecutar la siguiente subfase ACP."
        if run.status == ACPWorkflowRunStatus.waiting_user:
            next_action = "Responder preguntas de implementacion antes de continuar."
        if run.status == ACPWorkflowRunStatus.blocked:
            next_action = "Resolver GAPs bloqueantes antes de exportar el paquete."
        if run.status in {ACPWorkflowRunStatus.completed, ACPWorkflowRunStatus.completed_with_observations}:
            next_action = "Generar export durable o revisar launcher."
    journey_state = load_persisted_journey_state_machine(db, record=record)
    return ACPWorkspaceResponse(
        session_id=record.id,
        workspace_id=record.workspace_id,
        access=access,
        run=run_response,
        phases=[serialize_phase_run(phase_map.get(definition.key), definition) for definition in ACP_PHASES],
        phase_definitions=phase_definition_responses(),
        readiness=readiness,
        validation=preview.validation,
        journey_state_machine=journey_state.model_dump(mode="json") if journey_state is not None else {},
        next_action=next_action,
    )


def _load_current_acp_run(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
) -> ACPBuildRunRecord | None:
    return db.exec(
        select(ACPBuildRunRecord).where(
            ACPBuildRunRecord.workspace_id == record.workspace_id,
            ACPBuildRunRecord.session_id == record.id,
            ACPBuildRunRecord.idempotency_key == _run_idempotency_key(record, _blueprint_version_number(snapshot)),
        )
    ).first()
