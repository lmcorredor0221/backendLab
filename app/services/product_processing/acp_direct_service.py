from __future__ import annotations

from collections import Counter
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    CommercialTier,
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    SessionRecord,
    SessionSnapshot,
)
from app.services.deliverable_catalog.registry_service import list_registry_entries
from app.services.product_processing.backlog_service import list_uncertainty_backlog
from app.services.product_processing.contracts import (
    AcpDirectRouteResolution,
    AcpStageReadinessEntry,
    ProductProcessingMode,
    QuestionPolicyMode,
    UncertaintyDisposition,
)


ACP_REQUIRED_STAGE_KEYS: tuple[str, ...] = (
    "discover",
    "define",
    "design",
    "tools",
    "memory",
    "estimate",
    "validate",
)

ACP_STAGE_LABELS: dict[str, str] = {
    "discover": "Descubrir",
    "define": "Definir",
    "design": "Disenar",
    "tools": "Herramientas",
    "memory": "Memoria",
    "estimate": "Estimar",
    "validate": "Validar",
}

ACP_STAGE_ACTIONS: dict[str, str] = {
    "discover": "Completa y aprueba el contexto de negocio.",
    "define": "Consolida requisitos, reglas y criterios de aceptacion.",
    "design": "Aprueba arquitectura, comportamiento, patrones y guardrails.",
    "tools": "Aprueba el set minimo de herramientas y contratos.",
    "memory": "Aprueba la estrategia de memoria, conocimiento y RAG.",
    "estimate": "Genera estimacion y comparativa de valor.",
    "validate": "Valida escenarios, gaps, preguntas y readiness antes de Package.",
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


def _approved_journey_stages(db: Session, record: SessionRecord) -> set[str]:
    if record.workspace_id is None:
        return set()
    rows = db.exec(
        select(JourneyStageArtifactRecord).where(
            JourneyStageArtifactRecord.workspace_id == record.workspace_id,
            JourneyStageArtifactRecord.session_id == record.id,
            JourneyStageArtifactRecord.state.in_(
                (
                    JourneyArtifactState.approved,
                    JourneyArtifactState.approved_legacy,
                )
            ),
        )
    ).all()
    return {row.stage_key for row in rows}


def _legacy_completed_stages(snapshot: SessionSnapshot | None) -> set[str]:
    if snapshot is None:
        return set()

    completed: set[str] = set()
    if snapshot.discovery is not None:
        completed.add("discover")
    if snapshot.canvas is not None:
        completed.add("define")
    if snapshot.blueprint is not None:
        blueprint = snapshot.blueprint
        if blueprint.architecture or blueprint.reasoning_pattern or blueprint.narrative:
            completed.add("design")
        if blueprint.tools:
            completed.add("tools")
        if blueprint.memory_strategy or blueprint.memory_profile or blueprint.knowledge_profile:
            completed.add("memory")
    if snapshot.estimation_report is not None:
        completed.add("estimate")
    if (
        getattr(snapshot, "evaluation", None) is not None
        or getattr(snapshot, "evaluation_dataset", None) is not None
        or getattr(snapshot, "evaluation_rubric", None) is not None
    ):
        completed.add("validate")
    return completed


def _catalog_counts() -> tuple[dict[str, int], list[str]]:
    entries = list_registry_entries()
    product_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    portable_paths: list[str] = []
    for entry in entries:
        if "acp" not in entry.product_scope:
            continue
        product_counter["acp"] += 1
        type_counter[entry.deliverable_type.value] += 1
        portable_paths.extend(entry.portable_paths)
    counts = {
        "acp_deliverables": product_counter["acp"],
        **{f"type_{key}": value for key, value in sorted(type_counter.items())},
    }
    return counts, _dedupe(portable_paths)


def build_acp_direct_resolution(
    db: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot | None = None,
    stage_justifications: dict[str, str] | None = None,
) -> AcpDirectRouteResolution:
    if record.workspace_id is None:
        raise ValueError("ACP direct resolution requires a workspace-scoped session.")

    justifications = {
        str(key): str(value).strip()
        for key, value in (stage_justifications or {}).items()
        if str(key).strip() in ACP_REQUIRED_STAGE_KEYS and str(value).strip()
    }
    completed = _dedupe(
        [
            *_approved_journey_stages(db, record),
            *_legacy_completed_stages(snapshot),
        ]
    )
    completed_set = set(completed)
    justified = [stage for stage in ACP_REQUIRED_STAGE_KEYS if stage in justifications]
    missing = [stage for stage in ACP_REQUIRED_STAGE_KEYS if stage not in completed_set and stage not in justified]

    backlog = list_uncertainty_backlog(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        product_mode=ProductProcessingMode.acp_implementation,
        include_closed=False,
    )
    technical_by_stage: Counter[str] = Counter()
    blocking_by_stage: Counter[str] = Counter()
    for entry in backlog:
        stage = entry.source_stage if entry.source_stage in ACP_REQUIRED_STAGE_KEYS else entry.target_stage
        if stage not in ACP_REQUIRED_STAGE_KEYS:
            stage = "validate"
        technical_by_stage[stage] += 1
        if entry.disposition == UncertaintyDisposition.block:
            blocking_by_stage[stage] += 1

    stages = [
        AcpStageReadinessEntry(
            stage_key=stage,
            label=ACP_STAGE_LABELS[stage],
            completed=stage in completed_set,
            justified=stage in justifications,
            justification=justifications.get(stage, ""),
            technical_question_count=technical_by_stage[stage],
            blocking_question_count=blocking_by_stage[stage],
            next_action="" if stage in completed_set or stage in justifications else ACP_STAGE_ACTIONS[stage],
        )
        for stage in ACP_REQUIRED_STAGE_KEYS
    ]
    catalog_counts, portable_paths = _catalog_counts()
    total_blocking_questions = sum(blocking_by_stage.values())
    readiness_blockers = [
        *(f"missing_stage:{stage}" for stage in missing),
        *(f"blocking_questions:{stage}:{count}" for stage, count in sorted(blocking_by_stage.items()) if count),
    ]
    can_start_package = not missing and total_blocking_questions == 0
    route_kind = "acp_after_blueprint" if set(ACP_REQUIRED_STAGE_KEYS[:5]).issubset(completed_set) else "acp_direct"
    next_stage_key = next((stage for stage in ACP_REQUIRED_STAGE_KEYS if stage in missing), "package")

    return AcpDirectRouteResolution(
        workspace_id=record.workspace_id,
        session_id=record.id,
        current_tier=record.commercial_tier if record.commercial_tier is not None else CommercialTier.blueprint,
        route_kind=route_kind,
        product_mode=ProductProcessingMode.acp_implementation,
        question_policy=QuestionPolicyMode.full_readiness,
        required_stage_keys=list(ACP_REQUIRED_STAGE_KEYS),
        completed_stage_keys=[stage for stage in ACP_REQUIRED_STAGE_KEYS if stage in completed_set],
        missing_stage_keys=missing,
        justified_stage_keys=justified,
        stages=stages,
        can_start_package=can_start_package,
        can_export_package=can_start_package and record.commercial_tier == CommercialTier.acp,
        next_stage_key=next_stage_key,
        readiness_blockers=readiness_blockers,
        total_technical_questions=sum(technical_by_stage.values()),
        total_blocking_questions=total_blocking_questions,
        catalog_counts=catalog_counts,
        portable_catalog_paths=portable_paths[:80],
        processing_guidance=(
            "ACP directo usa full readiness desde la primera etapa: pregunta lo necesario, "
            "requiere etapas LEAN completas o justificacion explicita, y solo permite Package cuando no hay bloqueos."
        ),
    )


def acp_route_blocking_reasons(resolution: AcpDirectRouteResolution) -> list[str]:
    return list(resolution.readiness_blockers)
