from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    EstimationActualsUpsertRequest,
    EstimationCalibrationDashboard,
    EstimationCalibrationStageSummary,
    EstimationErrorMetricEntry,
    EstimationErrorMetricRecord,
    EstimationMaturityStage,
    EstimationRecentCalibrationEntry,
    EstimationReportArtifact,
    EstimationRunEntry,
    EstimationRunRecord,
    EstimationScenarioType,
    LLMProviderKey,
    ProjectActualsEntry,
    ProjectActualsRecord,
    SessionRecord,
    utc_now,
)


def _round_metric(value: float) -> float:
    return round(value, 2)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _round_metric(sum(values) / len(values))


def _percentage_error(estimated: float, actual: float) -> float:
    if actual == 0:
        return 0.0 if estimated == 0 else 100.0
    return _round_metric(abs(estimated - actual) / abs(actual) * 100)


def _bias_percent(estimated: float, actual: float) -> float:
    if actual == 0:
        return 0.0 if estimated == 0 else 100.0
    return _round_metric((estimated - actual) / abs(actual) * 100)


def _band_hit(estimated: float, actual: float, uncertainty_band_percent: int) -> bool:
    if uncertainty_band_percent <= 0:
        return estimated == actual
    delta = abs(estimated) * (uncertainty_band_percent / 100)
    return (estimated - delta) <= actual <= (estimated + delta)


def _hydrate_estimation_report(record: EstimationRunRecord) -> EstimationReportArtifact:
    return EstimationReportArtifact.model_validate(record.estimation_payload)


def build_estimation_run_entry(record: EstimationRunRecord) -> EstimationRunEntry:
    return EstimationRunEntry(
        id=record.id,
        blueprint_version_number=record.blueprint_version_number,
        source_action=record.source_action,
        maturity_stage=record.maturity_stage,
        active_provider=record.active_provider,
        pricing_policy=record.pricing_policy,
        confidence_score=record.confidence_score,
        confidence_label=record.confidence_label,
        uncertainty_band_percent=record.uncertainty_band_percent,
        traditional_hours_total=record.traditional_hours_total,
        traditional_duration_weeks=record.traditional_duration_weeks,
        traditional_cost_total=record.traditional_cost_total,
        agentic_hours_total=record.agentic_hours_total,
        agentic_duration_weeks=record.agentic_duration_weeks,
        agentic_cost_total=record.agentic_cost_total,
        automation_coverage_percent=record.automation_coverage_percent,
        created_at=record.created_at,
    )


def build_project_actuals_entry(record: ProjectActualsRecord) -> ProjectActualsEntry:
    return ProjectActualsEntry(
        id=record.id,
        estimation_run_id=record.estimation_run_id,
        delivery_mode=record.delivery_mode,
        actual_provider=record.actual_provider,
        actual_hours_total=record.actual_hours_total,
        actual_duration_weeks=record.actual_duration_weeks,
        actual_cost_total=record.actual_cost_total,
        actual_automation_coverage_percent=record.actual_automation_coverage_percent,
        notes=record.notes,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def build_estimation_error_metric_entry(record: EstimationErrorMetricRecord) -> EstimationErrorMetricEntry:
    return EstimationErrorMetricEntry(
        id=record.id,
        estimation_run_id=record.estimation_run_id,
        maturity_stage=record.maturity_stage,
        scenario_type=record.scenario_type,
        active_provider=record.active_provider,
        absolute_percentage_error_hours=record.absolute_percentage_error_hours,
        absolute_percentage_error_duration=record.absolute_percentage_error_duration,
        absolute_percentage_error_cost=record.absolute_percentage_error_cost,
        absolute_percentage_error_automation=record.absolute_percentage_error_automation,
        bias_hours_percent=record.bias_hours_percent,
        bias_duration_percent=record.bias_duration_percent,
        bias_cost_percent=record.bias_cost_percent,
        bias_automation_percent=record.bias_automation_percent,
        band_hit_hours=record.band_hit_hours,
        band_hit_duration=record.band_hit_duration,
        band_hit_cost=record.band_hit_cost,
        band_hit_overall=record.band_hit_overall,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def list_estimation_runs(session: Session, session_id: UUID) -> list[EstimationRunRecord]:
    return session.exec(
        select(EstimationRunRecord)
        .where(EstimationRunRecord.session_id == session_id)
        .order_by(EstimationRunRecord.created_at.desc())
    ).all()


def list_project_actuals(session: Session, session_id: UUID) -> list[ProjectActualsRecord]:
    return session.exec(
        select(ProjectActualsRecord)
        .where(ProjectActualsRecord.session_id == session_id)
        .order_by(ProjectActualsRecord.updated_at.desc(), ProjectActualsRecord.created_at.desc())
    ).all()


def list_estimation_error_metrics(session: Session, session_id: UUID) -> list[EstimationErrorMetricRecord]:
    return session.exec(
        select(EstimationErrorMetricRecord)
        .where(EstimationErrorMetricRecord.session_id == session_id)
        .order_by(EstimationErrorMetricRecord.updated_at.desc(), EstimationErrorMetricRecord.created_at.desc())
    ).all()


def persist_estimation_run(
    session: Session,
    *,
    session_id: UUID,
    blueprint_version_number: int | None,
    source_action: str,
    estimation_report: EstimationReportArtifact,
) -> EstimationRunRecord:
    payload = estimation_report.model_dump(mode="json")
    record = EstimationRunRecord(
        session_id=session_id,
        blueprint_version_number=blueprint_version_number,
        source_action=source_action,
        maturity_stage=estimation_report.maturity_stage,
        active_provider=estimation_report.agentic.active_provider,
        pricing_policy=estimation_report.agentic.pricing_policy,
        confidence_score=estimation_report.confidence.score,
        confidence_label=estimation_report.confidence.label,
        uncertainty_band_percent=estimation_report.confidence.uncertainty_band_percent,
        traditional_hours_total=estimation_report.traditional.estimated_hours_total,
        traditional_duration_weeks=estimation_report.traditional.estimated_duration_weeks,
        traditional_cost_total=estimation_report.traditional.estimated_cost,
        agentic_hours_total=estimation_report.agentic.estimated_hours_total,
        agentic_duration_weeks=estimation_report.agentic.estimated_duration_weeks,
        agentic_cost_total=estimation_report.agentic.estimated_cost,
        automation_coverage_percent=estimation_report.agentic.automation_coverage_percent,
        estimation_payload=payload,
        created_at=estimation_report.generated_at or utc_now(),
    )
    session.add(record)
    session.flush()
    return record


def _build_error_metric_record(
    estimation_run: EstimationRunRecord,
    actuals: ProjectActualsRecord,
) -> EstimationErrorMetricRecord:
    report = _hydrate_estimation_report(estimation_run)
    expected_hours = (
        report.traditional.estimated_hours_total
        if actuals.delivery_mode == EstimationScenarioType.traditional
        else report.agentic.estimated_hours_total
    )
    expected_duration = (
        report.traditional.estimated_duration_weeks
        if actuals.delivery_mode == EstimationScenarioType.traditional
        else report.agentic.estimated_duration_weeks
    )
    expected_cost = (
        report.traditional.estimated_cost
        if actuals.delivery_mode == EstimationScenarioType.traditional
        else report.agentic.estimated_cost
    )
    expected_automation = 0 if actuals.delivery_mode == EstimationScenarioType.traditional else report.agentic.automation_coverage_percent
    uncertainty_band = report.confidence.uncertainty_band_percent

    return EstimationErrorMetricRecord(
        session_id=estimation_run.session_id,
        estimation_run_id=estimation_run.id,
        actuals_id=actuals.id,
        maturity_stage=estimation_run.maturity_stage,
        scenario_type=actuals.delivery_mode,
        active_provider=actuals.actual_provider,
        absolute_percentage_error_hours=_percentage_error(expected_hours, actuals.actual_hours_total),
        absolute_percentage_error_duration=_percentage_error(expected_duration, actuals.actual_duration_weeks),
        absolute_percentage_error_cost=_percentage_error(expected_cost, actuals.actual_cost_total),
        absolute_percentage_error_automation=_percentage_error(
            float(expected_automation),
            float(actuals.actual_automation_coverage_percent),
        ),
        bias_hours_percent=_bias_percent(expected_hours, actuals.actual_hours_total),
        bias_duration_percent=_bias_percent(expected_duration, actuals.actual_duration_weeks),
        bias_cost_percent=_bias_percent(expected_cost, actuals.actual_cost_total),
        bias_automation_percent=_bias_percent(
            float(expected_automation),
            float(actuals.actual_automation_coverage_percent),
        ),
        band_hit_hours=_band_hit(expected_hours, actuals.actual_hours_total, uncertainty_band),
        band_hit_duration=_band_hit(expected_duration, actuals.actual_duration_weeks, uncertainty_band),
        band_hit_cost=_band_hit(expected_cost, actuals.actual_cost_total, uncertainty_band),
        band_hit_overall=(
            _band_hit(expected_hours, actuals.actual_hours_total, uncertainty_band)
            and _band_hit(expected_duration, actuals.actual_duration_weeks, uncertainty_band)
            and _band_hit(expected_cost, actuals.actual_cost_total, uncertainty_band)
        ),
        created_at=utc_now(),
        updated_at=utc_now(),
    )


def upsert_project_actuals(
    session: Session,
    *,
    session_id: UUID,
    current_user_id: UUID,
    payload: EstimationActualsUpsertRequest,
) -> tuple[ProjectActualsRecord, EstimationErrorMetricRecord]:
    estimation_run = session.exec(
        select(EstimationRunRecord).where(
            EstimationRunRecord.id == payload.estimation_run_id,
            EstimationRunRecord.session_id == session_id,
        )
    ).first()
    if estimation_run is None:
        raise ValueError("Estimation run not found for this session")

    actual_provider: LLMProviderKey | None = payload.actual_provider
    actual_automation = payload.actual_automation_coverage_percent
    if payload.delivery_mode == EstimationScenarioType.traditional:
        actual_provider = None
        actual_automation = 0
    elif actual_provider is None:
        actual_provider = estimation_run.active_provider

    record = session.exec(
        select(ProjectActualsRecord).where(ProjectActualsRecord.estimation_run_id == payload.estimation_run_id)
    ).first()
    if record is None:
        record = ProjectActualsRecord(
            session_id=session_id,
            estimation_run_id=payload.estimation_run_id,
            delivery_mode=payload.delivery_mode,
            actual_provider=actual_provider,
            actual_hours_total=payload.actual_hours_total,
            actual_duration_weeks=payload.actual_duration_weeks,
            actual_cost_total=payload.actual_cost_total,
            actual_automation_coverage_percent=actual_automation,
            notes=payload.notes,
            captured_by_user_id=current_user_id,
        )
        session.add(record)
        session.flush()
    else:
        record.delivery_mode = payload.delivery_mode
        record.actual_provider = actual_provider
        record.actual_hours_total = payload.actual_hours_total
        record.actual_duration_weeks = payload.actual_duration_weeks
        record.actual_cost_total = payload.actual_cost_total
        record.actual_automation_coverage_percent = actual_automation
        record.notes = payload.notes
        record.captured_by_user_id = current_user_id
        record.updated_at = utc_now()
        session.add(record)
        session.flush()

    existing_metric = session.exec(
        select(EstimationErrorMetricRecord).where(EstimationErrorMetricRecord.estimation_run_id == payload.estimation_run_id)
    ).first()
    next_metric = _build_error_metric_record(estimation_run, record)
    if existing_metric is None:
        session.add(next_metric)
        session.flush()
        return record, next_metric

    existing_metric.actuals_id = next_metric.actuals_id
    existing_metric.maturity_stage = next_metric.maturity_stage
    existing_metric.scenario_type = next_metric.scenario_type
    existing_metric.active_provider = next_metric.active_provider
    existing_metric.absolute_percentage_error_hours = next_metric.absolute_percentage_error_hours
    existing_metric.absolute_percentage_error_duration = next_metric.absolute_percentage_error_duration
    existing_metric.absolute_percentage_error_cost = next_metric.absolute_percentage_error_cost
    existing_metric.absolute_percentage_error_automation = next_metric.absolute_percentage_error_automation
    existing_metric.bias_hours_percent = next_metric.bias_hours_percent
    existing_metric.bias_duration_percent = next_metric.bias_duration_percent
    existing_metric.bias_cost_percent = next_metric.bias_cost_percent
    existing_metric.bias_automation_percent = next_metric.bias_automation_percent
    existing_metric.band_hit_hours = next_metric.band_hit_hours
    existing_metric.band_hit_duration = next_metric.band_hit_duration
    existing_metric.band_hit_cost = next_metric.band_hit_cost
    existing_metric.band_hit_overall = next_metric.band_hit_overall
    existing_metric.updated_at = utc_now()
    session.add(existing_metric)
    session.flush()
    return record, existing_metric


def _build_stage_summary(
    maturity_stage: EstimationMaturityStage,
    total_runs: int,
    metrics: list[EstimationErrorMetricRecord],
) -> EstimationCalibrationStageSummary:
    return EstimationCalibrationStageSummary(
        maturity_stage=maturity_stage,
        total_runs=total_runs,
        calibrated_runs=len(metrics),
        mean_absolute_percentage_error_hours=_mean([item.absolute_percentage_error_hours for item in metrics]),
        mean_absolute_percentage_error_duration=_mean([item.absolute_percentage_error_duration for item in metrics]),
        mean_absolute_percentage_error_cost=_mean([item.absolute_percentage_error_cost for item in metrics]),
        mean_absolute_percentage_error_automation=_mean([item.absolute_percentage_error_automation for item in metrics]),
        mean_bias_hours_percent=_mean([item.bias_hours_percent for item in metrics]),
        mean_bias_duration_percent=_mean([item.bias_duration_percent for item in metrics]),
        mean_bias_cost_percent=_mean([item.bias_cost_percent for item in metrics]),
        mean_bias_automation_percent=_mean([item.bias_automation_percent for item in metrics]),
        band_hit_rate=_round_metric((sum(1 for item in metrics if item.band_hit_overall) / len(metrics)) * 100)
        if metrics
        else 0,
    )


def build_estimation_calibration_dashboard(session: Session, workspace_id: UUID) -> EstimationCalibrationDashboard:
    session_records = session.exec(
        select(SessionRecord).where(SessionRecord.workspace_id == workspace_id).order_by(SessionRecord.updated_at.desc())
    ).all()
    if not session_records:
        return EstimationCalibrationDashboard(generated_at=utc_now())

    session_ids = [item.id for item in session_records]
    session_titles = {item.id: item.title for item in session_records}
    estimation_runs = session.exec(
        select(EstimationRunRecord)
        .where(EstimationRunRecord.session_id.in_(session_ids))
        .order_by(EstimationRunRecord.created_at.desc())
    ).all()
    if not estimation_runs:
        return EstimationCalibrationDashboard(generated_at=utc_now())

    metrics = session.exec(
        select(EstimationErrorMetricRecord)
        .where(EstimationErrorMetricRecord.session_id.in_(session_ids))
        .order_by(EstimationErrorMetricRecord.updated_at.desc(), EstimationErrorMetricRecord.created_at.desc())
    ).all()
    actuals = session.exec(
        select(ProjectActualsRecord)
        .where(ProjectActualsRecord.session_id.in_(session_ids))
        .order_by(ProjectActualsRecord.updated_at.desc(), ProjectActualsRecord.created_at.desc())
    ).all()

    metrics_by_run = {item.estimation_run_id: item for item in metrics}
    actuals_by_run = {item.estimation_run_id: item for item in actuals}
    runs_by_stage: dict[EstimationMaturityStage, list[EstimationRunRecord]] = defaultdict(list)
    metrics_by_stage: dict[EstimationMaturityStage, list[EstimationErrorMetricRecord]] = defaultdict(list)
    for item in estimation_runs:
        runs_by_stage[item.maturity_stage].append(item)
    for item in metrics:
        metrics_by_stage[item.maturity_stage].append(item)

    stage_order = [
        EstimationMaturityStage.canvas,
        EstimationMaturityStage.blueprint,
        EstimationMaturityStage.ready_to_build,
    ]
    precision_by_stage = [
        _build_stage_summary(stage, len(runs_by_stage.get(stage, [])), metrics_by_stage.get(stage, []))
        for stage in stage_order
        if runs_by_stage.get(stage) or metrics_by_stage.get(stage)
    ]

    recent_projects: list[EstimationRecentCalibrationEntry] = []
    for actual in actuals[:6]:
        run = next((item for item in estimation_runs if item.id == actual.estimation_run_id), None)
        if run is None:
            continue
        metric = metrics_by_run.get(run.id)
        recent_projects.append(
            EstimationRecentCalibrationEntry(
                session_id=run.session_id,
                session_title=session_titles.get(run.session_id, "Proyecto local"),
                estimation_run_id=run.id,
                maturity_stage=run.maturity_stage,
                scenario_type=actual.delivery_mode,
                provider=actual.actual_provider if actual.delivery_mode == EstimationScenarioType.agentic else None,
                estimated_cost_total=run.agentic_cost_total if actual.delivery_mode == EstimationScenarioType.agentic else run.traditional_cost_total,
                actual_cost_total=actual.actual_cost_total,
                cost_absolute_percentage_error=metric.absolute_percentage_error_cost if metric is not None else 0,
                band_hit_overall=metric.band_hit_overall if metric is not None else False,
                updated_at=actual.updated_at,
            )
        )

    total_runs = len(estimation_runs)
    calibrated_runs = len(metrics)
    return EstimationCalibrationDashboard(
        generated_at=utc_now(),
        total_runs=total_runs,
        calibrated_runs=calibrated_runs,
        coverage_percent=_round_metric((calibrated_runs / total_runs) * 100) if total_runs else 0,
        mean_absolute_percentage_error_hours=_mean([item.absolute_percentage_error_hours for item in metrics]),
        mean_absolute_percentage_error_duration=_mean([item.absolute_percentage_error_duration for item in metrics]),
        mean_absolute_percentage_error_cost=_mean([item.absolute_percentage_error_cost for item in metrics]),
        mean_absolute_percentage_error_automation=_mean([item.absolute_percentage_error_automation for item in metrics]),
        mean_bias_cost_percent=_mean([item.bias_cost_percent for item in metrics]),
        band_hit_rate=_round_metric((sum(1 for item in metrics if item.band_hit_overall) / calibrated_runs) * 100)
        if calibrated_runs
        else 0,
        precision_by_stage=precision_by_stage,
        recent_projects=recent_projects,
    )
