from __future__ import annotations

import hashlib
import json
from time import perf_counter
from uuid import UUID

import app.services.skill_runtime as skill_runtime
from sqlmodel import Session, select

from app.models import (
    ArtifactStatus,
    ConfidenceBreakdown,
    EstimationAnalysisArtifact,
    EstimationAnalysisDecision,
    EstimationBenchmarkRef,
    EstimationComplexityDriver,
    EstimationConfidenceAdjustmentProposal,
    EstimationConfidenceLabel,
    EstimationDeterministicInputs,
    EstimationPackagePolicyState,
    EstimationQuestion,
    EstimationRecentCalibrationEntry,
    EstimationReportArtifact,
    EstimationRiskRegisterEntry,
    EstimationRunRecord,
    EstimationScenarioAdjustment,
    EstimationSavingsOpportunity,
    EstimationUncertaintyFactor,
    EvidenceItem,
    EvidenceSource,
    KnowledgeScope,
    RuntimeCatalogEntryRecord,
    SessionSnapshot,
    SessionStage,
    utc_now,
)
from app.services.estimation_calibration import build_estimation_calibration_dashboard
from app.services.knowledge_memory import KnowledgeMemoryService
from app.services.llm_runtime.builder_contracts import EstimationRiskAnalysisInput, EstimationRiskAnalysisOutput
from app.services.llm_runtime.stage_context_types import StageContextBundle


CATALOG_KEYS_USED = [
    "estimation_role_rates",
    "estimation_workstream_effort",
    "estimation_automation_matrix",
    "estimation_pricing_profiles",
    "estimation_confidence_bands",
    "estimation_confidence_weights",
]
DEFAULT_FORMULA_NOTES = [
    "Escenario tradicional = horas deterministicas por workstream x tarifa activa del catalogo.",
    "Escenario agentic = costo humano supervisado + runtime provider segun pricing snapshot activo.",
    "Los factores LLM solo ajustan confianza, sensibilidad y narrativas; no fijan tarifas ni costos finales.",
]


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
    return result


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _confidence_label_for_score(score: int) -> EstimationConfidenceLabel:
    if score >= 90:
        return EstimationConfidenceLabel.high
    if score >= 75:
        return EstimationConfidenceLabel.medium_high
    if score >= 60:
        return EstimationConfidenceLabel.medium
    if score >= 40:
        return EstimationConfidenceLabel.medium_low
    return EstimationConfidenceLabel.low


def build_estimation_pricing_catalog_signature(session: Session) -> str:
    rows = session.exec(
        select(RuntimeCatalogEntryRecord)
        .where(
            RuntimeCatalogEntryRecord.catalog_key == "estimation_pricing_profiles",
            RuntimeCatalogEntryRecord.is_active == True,  # noqa: E712
        )
        .order_by(RuntimeCatalogEntryRecord.order_index.asc(), RuntimeCatalogEntryRecord.id.asc())
    ).all()
    payload = [
        {
            "id": str(row.id),
            "order_index": row.order_index,
            "payload": row.payload,
            "updated_at": row.updated_at.isoformat(),
        }
        for row in rows
    ]
    return _stable_hash(payload)


def build_estimation_validation_fingerprint_from_state(
    *,
    latest_validate_artifact,
    evaluation_runs: list,
    simulation_runs: list,
) -> str:
    latest_evaluation_run = evaluation_runs[0] if evaluation_runs else None
    latest_simulation_run = simulation_runs[0] if simulation_runs else None
    payload = {
        "validate_artifact_id": str(latest_validate_artifact.id) if latest_validate_artifact is not None else "",
        "validate_version": latest_validate_artifact.version_number if latest_validate_artifact is not None else None,
        "validate_state": latest_validate_artifact.state if latest_validate_artifact is not None else "",
        "evaluation_run_id": str(latest_evaluation_run.id) if latest_evaluation_run is not None else "",
        "evaluation_score": latest_evaluation_run.overall_score if latest_evaluation_run is not None else None,
        "evaluation_status": latest_evaluation_run.status if latest_evaluation_run is not None else "",
        "simulation_run_id": str(latest_simulation_run.id) if latest_simulation_run is not None else "",
        "simulation_status": latest_simulation_run.status if latest_simulation_run is not None else "",
        "simulation_final_status": latest_simulation_run.final_status if latest_simulation_run is not None else "",
        "simulation_hard_gate_status": latest_simulation_run.hard_gate_status if latest_simulation_run is not None else "",
    }
    if not any(payload.values()):
        return "validate:missing"
    return _stable_hash(payload)


def build_estimation_validation_fingerprint(snapshot: SessionSnapshot) -> str:
    return build_estimation_validation_fingerprint_from_state(
        latest_validate_artifact=(snapshot.journey_latest_artifacts or {}).get("validate"),
        evaluation_runs=list(snapshot.evaluation_runs),
        simulation_runs=list(snapshot.simulation_runs),
    )


def build_estimation_benchmark_corpus_hash(
    session: Session,
    *,
    workspace_id: UUID | None,
    session_id: UUID | None,
) -> str:
    if workspace_id is None and session_id is None:
        return ""
    service = KnowledgeMemoryService()
    return service._combined_corpus_hash(session, workspace_id=workspace_id, session_id=session_id)


def build_estimation_deterministic_inputs(
    session: Session,
    *,
    snapshot: SessionSnapshot,
    report: EstimationReportArtifact,
    benchmark_ids: list[str],
    benchmark_corpus_hash: str,
) -> EstimationDeterministicInputs:
    dashboard = (
        build_estimation_calibration_dashboard(session, snapshot.session.workspace_id)
        if snapshot.session.workspace_id is not None
        else None
    )
    return EstimationDeterministicInputs(
        pricing_catalog_signature=build_estimation_pricing_catalog_signature(session),
        validation_fingerprint=build_estimation_validation_fingerprint(snapshot),
        benchmark_corpus_hash=benchmark_corpus_hash,
        catalogs_used=list(CATALOG_KEYS_USED),
        benchmark_ids=list(benchmark_ids),
        formula_notes=[
            *DEFAULT_FORMULA_NOTES,
            (
                f"Pricing snapshot: {report.agentic.pricing_policy or 'sin perfil'}"
                f" / {report.agentic.provider_model or 'sin modelo'}."
            ),
        ],
        calibration_sample_size=dashboard.calibrated_runs if dashboard is not None else 0,
    )


def _build_default_scenarios(*, preliminary: bool) -> list[EstimationScenarioAdjustment]:
    optimistic = (0.94, 0.94, 0.93) if preliminary else (0.96, 0.96, 0.95)
    conservative = (1.16, 1.14, 1.15) if preliminary else (1.1, 1.08, 1.1)
    return [
        EstimationScenarioAdjustment(
            scenario_key="optimistic",
            hours_multiplier=optimistic[0],
            duration_multiplier=optimistic[1],
            cost_multiplier=optimistic[2],
            rationale="Escenario optimista con menor retrabajo y aprobaciones mas fluidas.",
        ),
        EstimationScenarioAdjustment(
            scenario_key="base",
            hours_multiplier=1.0,
            duration_multiplier=1.0,
            cost_multiplier=1.0,
            rationale="Escenario base derivado del motor deterministico.",
        ),
        EstimationScenarioAdjustment(
            scenario_key="conservative",
            hours_multiplier=conservative[0],
            duration_multiplier=conservative[1],
            cost_multiplier=conservative[2],
            rationale="Escenario conservador por integraciones, aprobaciones o hardening adicional.",
        ),
    ]


def _build_workspace_benchmark_refs(
    recent_projects: list[EstimationRecentCalibrationEntry],
) -> list[EstimationBenchmarkRef]:
    refs: list[EstimationBenchmarkRef] = []
    for item in recent_projects[:3]:
        refs.append(
            EstimationBenchmarkRef(
                benchmark_key=f"workspace-actual:{item.estimation_run_id}",
                title=f"Actual calibrado: {item.session_title or item.session_id}",
                source_kind="workspace_actuals",
                source_ref=f"workspace://estimation-runs/{item.estimation_run_id}",
                sample_size=1,
                captured_at=item.updated_at.isoformat(),
                freshness="reciente",
                summary=(
                    f"Proyecto {item.session_title or item.session_id} con error costo "
                    f"{item.cost_absolute_percentage_error:.2f}% y band_hit={item.band_hit_overall}."
                ),
                workspace_scoped=True,
            )
        )
    return refs


def _build_context_benchmark_refs(stage_context: StageContextBundle | None) -> list[EstimationBenchmarkRef]:
    if stage_context is None:
        return []
    refs: list[EstimationBenchmarkRef] = []
    for item in stage_context.retrieved_hits[:4]:
        refs.append(
            EstimationBenchmarkRef(
                benchmark_key=item.key or item.relative_path,
                title=item.title or item.relative_path,
                source_kind="knowledge_document",
                source_ref=item.uri,
                sample_size=0,
                captured_at="",
                freshness=item.source_version or "",
                summary=item.summary or item.excerpt,
                workspace_scoped=False,
            )
        )
    return refs


def _build_validation_summary(snapshot: SessionSnapshot) -> list[str]:
    latest_validate_artifact = (snapshot.journey_latest_artifacts or {}).get("validate")
    latest_evaluation_run = snapshot.evaluation_runs[0] if snapshot.evaluation_runs else None
    latest_simulation_run = snapshot.simulation_runs[0] if snapshot.simulation_runs else None
    summary = []
    if latest_validate_artifact is not None:
        summary.append(
            f"Validate artifact: state={latest_validate_artifact.state} version={latest_validate_artifact.version_number}"
        )
    if latest_evaluation_run is not None:
        summary.append(
            f"Evaluation run: status={latest_evaluation_run.status} score={latest_evaluation_run.overall_score}"
        )
    if latest_simulation_run is not None:
        summary.append(
            f"Simulation run: status={latest_simulation_run.status} overall={latest_simulation_run.overall_assessment}"
        )
    if not summary:
        summary.append("No existe una aprobacion Validate vigente; la estimacion debe tratarse como preliminar.")
    return summary


def _build_workspace_calibration_summary(session: Session, snapshot: SessionSnapshot) -> tuple[list[str], list[EstimationBenchmarkRef], int]:
    workspace_id = snapshot.session.workspace_id
    if workspace_id is None:
        return ([], [], 0)
    dashboard = build_estimation_calibration_dashboard(session, workspace_id)
    summary = [
        f"Workspace calibration: {dashboard.calibrated_runs}/{dashboard.total_runs} runs calibrados.",
        f"Coverage={dashboard.coverage_percent:.2f}% band_hit_rate={dashboard.band_hit_rate:.2f}%.",
        f"MAPE costo={dashboard.mean_absolute_percentage_error_cost:.2f}%.",
    ]
    return (summary, _build_workspace_benchmark_refs(dashboard.recent_projects), dashboard.calibrated_runs)


def _build_pricing_summary(report: EstimationReportArtifact) -> list[str]:
    snapshot = report.agentic.pricing_snapshot
    if snapshot is None:
        return ["No existe pricing snapshot activo; la confianza debe degradarse y la banda ampliarse."]
    return [
        f"Provider={snapshot.provider} model={snapshot.model} pricing_mode={snapshot.pricing_mode}.",
        f"Profile={snapshot.profile_key} fx={snapshot.cop_per_usd}.",
        *snapshot.assumptions[:3],
    ]


def _build_fallback_analysis(
    report: EstimationReportArtifact,
    *,
    stage_context: StageContextBundle | None,
    workspace_benchmarks: list[EstimationBenchmarkRef],
    calibration_sample_size: int,
    reason: str,
) -> EstimationAnalysisArtifact:
    top_workstreams = sorted(
        report.agentic.workstream_breakdown,
        key=lambda item: item.estimated_hours,
        reverse=True,
    )[:3]
    preliminary = report.deterministic_inputs.validation_fingerprint == "validate:missing"
    benchmark_refs = workspace_benchmarks + _build_context_benchmark_refs(stage_context)
    adjustment = EstimationConfidenceAdjustmentProposal(
        proposed_score_delta=-8 if preliminary else (-6 if report.agentic.pricing_snapshot is None else 0),
        proposed_uncertainty_band_delta=12 if preliminary else (8 if report.agentic.pricing_snapshot is None else 0),
        rationale="Fallback deterministico aplicado porque el analisis LLM no estuvo disponible o no paso validacion.",
        evidence_refs=["estimate.fallback"],
    )
    return EstimationAnalysisArtifact(
        summary=reason,
        complexity_drivers=[
            EstimationComplexityDriver(
                driver_key=item.workstream_key,
                title=item.label,
                workstream_key=item.workstream_key,
                impact_level="high" if index == 0 else "medium",
                summary=item.notes[0] if item.notes else f"Workstream con {item.estimated_hours:.1f}h estimadas.",
                evidence_refs=[f"estimate.agentic.workstream_breakdown.{item.workstream_key}"],
            )
            for index, item in enumerate(top_workstreams)
        ],
        risk_register=[
            EstimationRiskRegisterEntry(
                risk_key=f"risk-{index + 1}",
                title=item,
                severity="high" if index == 0 else "medium",
                likelihood="medium",
                impact=item,
                mitigation="Resolver el gap o documentar una mitigacion verificable antes de comprometer fecha o costo.",
                evidence_refs=[f"estimate.risk_drivers.{index}"],
            )
            for index, item in enumerate(report.risk_drivers[:4])
        ],
        uncertainty_factors=[
            EstimationUncertaintyFactor(
                factor_key=f"uncertainty-{index + 1}",
                title=item,
                category="confidence",
                impact_area="confidence",
                summary=item,
                evidence_refs=[f"estimate.confidence.negative_signals.{index}"],
            )
            for index, item in enumerate(report.confidence.negative_signals[:4])
        ],
        benchmark_refs=benchmark_refs[:6],
        scenario_adjustments=_build_default_scenarios(preliminary=preliminary),
        savings_opportunities=[
            EstimationSavingsOpportunity(
                opportunity_key="reduce_manual_rework",
                title="Reducir retrabajo manual",
                summary="Cerrar contratos, validaciones y aprobaciones temprano evita retrabajo en build y QA.",
                expected_impact="Menor banda de incertidumbre y menos supervision correctiva.",
                prerequisites=["Aprobar Validate", "Cerrar pricing snapshot", "Resolver gaps del ACP"],
                evidence_refs=["estimate.confidence.recommended_next_actions"],
            )
        ],
        assumptions=list(report.assumptions[:5]),
        questions=[
            EstimationQuestion(
                question_key=f"question-{index + 1}",
                question=item,
                rationale="El motor deterministico todavia necesita evidencia adicional para estrechar la banda comercial.",
                blocking=index == 0 and preliminary,
            )
            for index, item in enumerate(report.confidence.recommended_next_actions[:4])
        ],
        evidence_refs=_dedupe(
            [*([item.key for item in stage_context.retrieved_hits[:4]] if stage_context is not None else []), "estimate.fallback"]
        ),
        confidence_adjustment_proposal=adjustment,
    )


def _normalize_scenarios(
    candidate: list[EstimationScenarioAdjustment],
    *,
    preliminary: bool,
) -> tuple[list[EstimationScenarioAdjustment], list[str]]:
    warnings: list[str] = []
    by_key = {item.scenario_key: item for item in candidate}
    defaults = {item.scenario_key: item for item in _build_default_scenarios(preliminary=preliminary)}
    normalized: list[EstimationScenarioAdjustment] = []
    for key in ("optimistic", "base", "conservative"):
        selected = by_key.get(key, defaults[key])
        normalized.append(selected)

    if not (
        normalized[0].hours_multiplier <= normalized[1].hours_multiplier <= normalized[2].hours_multiplier
        and normalized[0].duration_multiplier <= normalized[1].duration_multiplier <= normalized[2].duration_multiplier
        and normalized[0].cost_multiplier <= normalized[1].cost_multiplier <= normalized[2].cost_multiplier
    ):
        warnings.append("Los escenarios propuestos no respetaban el orden optimistic/base/conservative; se aplico fallback deterministico.")
        return (_build_default_scenarios(preliminary=preliminary), warnings)
    return (normalized, warnings)


def _sanitize_analysis_artifact(
    report: EstimationReportArtifact,
    *,
    artifact: EstimationAnalysisArtifact,
    stage_context: StageContextBundle | None,
    workspace_benchmarks: list[EstimationBenchmarkRef],
    calibration_sample_size: int,
) -> tuple[EstimationAnalysisArtifact, list[str]]:
    warnings: list[str] = []
    preliminary = report.deterministic_inputs.validation_fingerprint == "validate:missing"
    scenarios, scenario_warnings = _normalize_scenarios(artifact.scenario_adjustments, preliminary=preliminary)
    warnings.extend(scenario_warnings)

    benchmark_refs = []
    for item in [*artifact.benchmark_refs, *workspace_benchmarks, *_build_context_benchmark_refs(stage_context)]:
        if item.source_kind == "workspace_actuals":
            benchmark_refs.append(item.model_copy(update={"workspace_scoped": True}))
        else:
            benchmark_refs.append(item)

    benchmark_refs = list(
        {
            (item.benchmark_key or item.source_ref or item.title): item
            for item in benchmark_refs
        }.values()
    )

    proposal = artifact.confidence_adjustment_proposal
    if calibration_sample_size == 0 and proposal.proposed_score_delta > 0:
        proposal = proposal.model_copy(
            update={
                "proposed_score_delta": 0,
                "rationale": "Sin calibration historica suficiente no se permite aumentar la confianza.",
            }
        )
        warnings.append("La propuesta de confianza no podia subir el score sin calibration historica; se neutralizo.")

    return (
        artifact.model_copy(
            update={
                "benchmark_refs": benchmark_refs[:6],
                "scenario_adjustments": scenarios,
                "confidence_adjustment_proposal": proposal,
                "evidence_refs": _dedupe(artifact.evidence_refs),
                "assumptions": _dedupe(artifact.assumptions),
            }
        ),
        warnings,
    )


def run_estimation_analysis(
    session: Session,
    *,
    snapshot: SessionSnapshot,
    report: EstimationReportArtifact,
    stage_context: StageContextBundle | None = None,
) -> tuple[EstimationAnalysisArtifact, skill_runtime.SkillExecutionTrace]:
    workspace_summary, workspace_benchmarks, calibration_sample_size = _build_workspace_calibration_summary(session, snapshot)
    input_payload = EstimationRiskAnalysisInput(
        blueprint=snapshot.blueprint,
        estimation_report=report,
        pricing_summary=_build_pricing_summary(report),
        validation_summary=_build_validation_summary(snapshot),
        workspace_calibration_summary=workspace_summary,
        benchmark_hints=[
            *workspace_summary,
            *[item.summary for item in workspace_benchmarks[:3]],
        ],
        source_refs=["session.blueprint", "session.estimation_report", "workspace.calibration_dashboard"],
    )

    started = perf_counter()
    warnings: list[str] = []
    llm_result = None
    try:
        llm_service = skill_runtime._builder_service_for_stage("estimate")
        llm_result = llm_service.analyze_estimation_risks(input_payload, context_bundle=stage_context)
        artifact = (
            EstimationAnalysisArtifact.model_validate(llm_result.artifact.model_dump(mode="json"))
            if llm_result is not None and llm_result.artifact is not None
            else None
        )
    except Exception as exc:  # noqa: BLE001
        artifact = None
        warnings.append(f"El analisis LLM de estimacion no estuvo disponible: {exc}")

    if artifact is None:
        artifact = _build_fallback_analysis(
            report,
            stage_context=stage_context,
            workspace_benchmarks=workspace_benchmarks,
            calibration_sample_size=calibration_sample_size,
            reason="Analisis deterministico de respaldo aplicado porque el LLM no devolvio un artefacto usable.",
        )
        warnings.append("Se uso un analisis de estimacion deterministico de respaldo.")

    artifact, sanitization_warnings = _sanitize_analysis_artifact(
        report,
        artifact=artifact,
        stage_context=stage_context,
        workspace_benchmarks=workspace_benchmarks,
        calibration_sample_size=calibration_sample_size,
    )
    warnings.extend(sanitization_warnings)

    duration_ms = int((perf_counter() - started) * 1000)
    trace = skill_runtime.SkillExecutionTrace(
        skill_key="estimation_risk_analysis_skill",
        label="Estimation risk analysis skill",
        stage=SessionStage.post_validation,
        status=ArtifactStatus.ready,
        duration_ms=duration_ms,
        warnings=list(_dedupe(warnings)),
        evidence=[
            EvidenceItem(
                source=EvidenceSource.llm_inference if llm_result is not None and llm_result.artifact is not None else EvidenceSource.rule_engine,
                detail="Analisis de riesgos, benchmarks y sensibilidad separado del motor numerico.",
            ),
            EvidenceItem(
                source=EvidenceSource.rule_engine,
                detail="Los costos, tarifas y formulas finales permanecen bajo control deterministico.",
            ),
        ],
        llm_trace=skill_runtime._build_llm_trace(llm_result),
        result_summary=artifact.summary or "Analisis de estimacion generado.",
        input_kind="EstimationRiskAnalysisInput",
        input_payload=input_payload.model_dump(mode="json"),
        output_kind="EstimationAnalysisArtifact",
        output_payload=artifact.model_dump(mode="json"),
    )
    return (artifact, trace)


def apply_estimation_analysis(
    report: EstimationReportArtifact,
    *,
    analysis: EstimationAnalysisArtifact,
    decision: EstimationAnalysisDecision | None = None,
) -> EstimationReportArtifact:
    decision = decision or report.analysis_decision or EstimationAnalysisDecision()
    base_confidence = report.base_confidence or report.confidence
    score = base_confidence.score
    band = base_confidence.uncertainty_band_percent
    package_block_reasons: list[str] = []
    preliminary = report.deterministic_inputs.validation_fingerprint == "validate:missing"

    if preliminary:
        score -= 8
        band = max(band + 12, 30)
        package_block_reasons.append("Validate aun no tiene una aprobacion vigente; la estimacion permanece preliminar.")

    if report.agentic.pricing_snapshot is None:
        score -= 6
        band += 8
        package_block_reasons.append("No existe pricing snapshot vigente para sustentar el costo variable del provider.")

    if base_confidence.blocking_gaps > 0:
        package_block_reasons.append("Persisten blocking gaps en la continuidad constructiva.")

    proposal = analysis.confidence_adjustment_proposal
    if decision.decision == "pending" and (
        proposal.proposed_score_delta != 0 or proposal.proposed_uncertainty_band_delta != 0
    ):
        package_block_reasons.append("La propuesta LLM de ajuste de confianza sigue pendiente de aceptar o rechazar.")
    elif decision.decision == "accepted":
        score += proposal.proposed_score_delta
        band += proposal.proposed_uncertainty_band_delta

    score = _clamp(score, 5, 96)
    band = _clamp(band, 8, 80)
    if score < 60:
        package_block_reasons.append("La confianza efectiva sigue por debajo del minimo recomendado para Package.")

    next_confidence = base_confidence.model_copy(
        update={
            "score": score,
            "label": _confidence_label_for_score(score),
            "uncertainty_band_percent": band,
        }
    )

    package_policy = EstimationPackagePolicyState(
        preliminary=preliminary,
        can_continue_to_package=not package_block_reasons,
        package_block_reasons=_dedupe(package_block_reasons),
        commercial_blocked=preliminary,
    )

    deterministic_inputs = report.deterministic_inputs.model_copy(
        update={
            "benchmark_ids": [
                item.benchmark_key or item.source_ref
                for item in analysis.benchmark_refs
                if item.benchmark_key or item.source_ref
            ]
        }
    )

    notes = list(report.notes)
    if preliminary and "Estimate se muestra como preliminar hasta cerrar Validate." not in notes:
        notes.append("Estimate se muestra como preliminar hasta cerrar Validate.")
    if report.agentic.pricing_snapshot is None and "Sin pricing snapshot vigente el costo final requiere revision manual." not in notes:
        notes.append("Sin pricing snapshot vigente el costo final requiere revision manual.")

    return report.model_copy(
        update={
            "confidence": next_confidence,
            "base_confidence": base_confidence,
            "analysis": analysis,
            "analysis_decision": decision,
            "deterministic_inputs": deterministic_inputs,
            "package_policy": package_policy,
            "notes": _dedupe(notes),
        }
    )


def apply_estimation_analysis_decision(
    report: EstimationReportArtifact,
    *,
    decision: str,
    note: str = "",
) -> EstimationReportArtifact:
    next_decision = EstimationAnalysisDecision(
        decision=decision,
        note=note.strip(),
        decided_at=utc_now(),
    )
    analysis = report.analysis or EstimationAnalysisArtifact(summary="No existe analisis LLM persistido para esta corrida.")
    return apply_estimation_analysis(report, analysis=analysis, decision=next_decision)


def build_estimation_stale_reasons(
    report: EstimationReportArtifact,
    *,
    current_blueprint_version_number: int | None,
    current_validation_fingerprint: str,
    current_pricing_catalog_signature: str,
    current_benchmark_corpus_hash: str = "",
) -> list[str]:
    stale_reasons: list[str] = []
    bound_blueprint_version = report.blueprint_version_number
    if current_blueprint_version_number is not None:
        if bound_blueprint_version is None:
            stale_reasons.append("estimation_blueprint_version_missing")
        elif bound_blueprint_version != current_blueprint_version_number:
            stale_reasons.append("blueprint_version_changed")

    current_inputs = report.deterministic_inputs
    if current_validation_fingerprint:
        if not current_inputs.validation_fingerprint:
            stale_reasons.append("estimation_validation_fingerprint_missing")
        elif current_inputs.validation_fingerprint != current_validation_fingerprint:
            stale_reasons.append("validation_context_changed")

    if current_pricing_catalog_signature:
        if not current_inputs.pricing_catalog_signature:
            stale_reasons.append("estimation_pricing_signature_missing")
        elif current_inputs.pricing_catalog_signature != current_pricing_catalog_signature:
            stale_reasons.append("pricing_catalog_changed")

    if current_benchmark_corpus_hash:
        if current_inputs.benchmark_corpus_hash and current_inputs.benchmark_corpus_hash != current_benchmark_corpus_hash:
            stale_reasons.append("benchmark_corpus_changed")
    return stale_reasons


def latest_estimation_run(session: Session, session_id: UUID) -> EstimationRunRecord | None:
    return session.exec(
        select(EstimationRunRecord)
        .where(EstimationRunRecord.session_id == session_id)
        .order_by(EstimationRunRecord.created_at.desc())
    ).first()
