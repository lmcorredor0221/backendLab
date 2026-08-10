from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.models import (
    ArtifactStatus,
    ExecutionLogRecord,
    JourneyArtifactState,
    MonitoringCapabilityObservabilityEntry,
    MonitoringContextBackendEntry,
    MonitoringProviderObservabilityEntry,
    MonitoringReleaseGateEntry,
    MonitoringReleaseObservability,
    MonitoringStageObservabilityEntry,
    SessionSnapshot,
    SkillRunEntry,
)
from app.services.journey_stage_contract import journey_stage_for_source_action, list_journey_stage_boundaries

_AUTH_ISOLATION_KEYWORDS = (
    "auth",
    "unauthor",
    "forbidden",
    "permiso",
    "permission",
    "workspace",
    "tenant",
    "isolation",
    "not available for the current user",
)
_LONG_TERM_SOURCE_KEYWORDS = ("knowledge", "catalog", "docs", "repo://", "::doc::")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _as_int(value: object) -> int:
    return int(round(_as_float(value)))


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _normalize_stage(stage: object) -> str:
    return str(getattr(stage, "value", stage) or "").strip().lower()


def _resolve_stage_for_run(run: SkillRunEntry) -> tuple[str, str]:
    mapped = journey_stage_for_source_action(run.source_action)
    if mapped is not None:
        return mapped
    stage_key = _normalize_stage(run.stage) or "unknown"
    return stage_key, stage_key.replace("_", " ")


def _normalized_sources(run: SkillRunEntry) -> list[dict[str, Any]]:
    trace = run.llm_trace
    if trace is None:
        return []
    return [item for item in trace.context_used_sources if isinstance(item, dict)]


def _has_long_term_hit(run: SkillRunEntry) -> bool:
    for source in _normalized_sources(run):
        uri = str(source.get("uri", "")).strip().lower()
        refs = [str(item).strip().lower() for item in source.get("source_refs", []) if isinstance(item, str)]
        lineages = [str(item).strip().lower() for item in source.get("source_lineage", []) if isinstance(item, str)]
        if uri.startswith("repo://") or any(token in uri for token in _LONG_TERM_SOURCE_KEYWORDS):
            return True
        if any(not ref.startswith("session.") for ref in refs if ref):
            return True
        if any(any(token in lineage for token in _LONG_TERM_SOURCE_KEYWORDS) for lineage in lineages):
            return True
    return False


def _compaction_ratio(run: SkillRunEntry) -> float | None:
    trace = run.llm_trace
    if trace is None:
        return None
    stats = trace.context_stats or {}
    baseline_tokens = _as_float(stats.get("baseline_estimated_tokens"))
    reduction_tokens = _as_float(stats.get("reduction_estimated_tokens"))
    if baseline_tokens <= 0:
        assembled_tokens = _as_float(stats.get("assembled_estimated_tokens"))
        if reduction_tokens > 0 or assembled_tokens > 0:
            baseline_tokens = reduction_tokens + assembled_tokens
    if baseline_tokens <= 0:
        return None
    return round((reduction_tokens / baseline_tokens) * 100, 2)


def _budget_exceeded(run: SkillRunEntry) -> bool:
    trace = run.llm_trace
    if trace is None:
        return False
    stats = trace.context_stats or {}
    budget_tokens = _as_float(stats.get("budget_tokens"))
    assembled_tokens = _as_float(stats.get("assembled_estimated_tokens"))
    return budget_tokens > 0 and assembled_tokens > budget_tokens


def _context_fingerprint_coverage(snapshot: SessionSnapshot) -> tuple[int, int]:
    artifacts = [item for item in snapshot.journey_artifacts if item.stage_key]
    with_fingerprint = sum(1 for item in artifacts if item.context_fingerprint.strip())
    return with_fingerprint, len(artifacts)


def _source_version_coverage(snapshot: SessionSnapshot) -> tuple[int, int]:
    artifacts = [
        item
        for item in snapshot.journey_artifacts
        if item.stage_key and item.stage_key != "discover" and item.state != JourneyArtifactState.approved_legacy
    ]
    with_versions = sum(1 for item in artifacts if bool(item.source_stage_versions))
    return with_versions, len(artifacts)


def _approval_audit_coverage(snapshot: SessionSnapshot) -> tuple[int, int]:
    approvals = list(snapshot.approvals)
    audited = 0
    for item in approvals:
        if item.status == "pending":
            continue
        if item.resolved_at and item.resolution_note.strip():
            audited += 1
    resolved = sum(1 for item in approvals if item.status != "pending")
    return audited, resolved


def _hard_gate_overrides(snapshot: SessionSnapshot) -> list[str]:
    violations: list[str] = []
    for run in snapshot.simulation_runs:
        if run.hard_gate_status != "pass" and run.final_status == "pass":
            violations.append(run.scenario_key or run.id)
    return violations


def _fallback_visibility_evidence(snapshot: SessionSnapshot, recent_error_records: list[ExecutionLogRecord]) -> int:
    texts: list[str] = []
    for artifact in snapshot.journey_artifacts:
        texts.extend(artifact.warnings)
        texts.extend(artifact.stale_reasons)
    for alert in snapshot.alert_events:
        texts.append(alert.title)
        texts.append(alert.message)
        texts.extend(alert.evidence)
    for record in recent_error_records:
        texts.append(record.message)
        if record.payload:
            texts.append(str(record.payload))
    normalized = " ".join(texts).lower()
    return int("fallback" in normalized or "degrad" in normalized or "provider_unavailable" in normalized)


def _is_auth_or_isolation_error(record: ExecutionLogRecord) -> bool:
    haystack = f"{record.message} {record.payload}".lower()
    return any(token in haystack for token in _AUTH_ISOLATION_KEYWORDS)


def _stage_order() -> list[tuple[str, str]]:
    return [(item.stage_key, item.label) for item in list_journey_stage_boundaries()]


def _stage_label(stage_key: str) -> str:
    for key, label in _stage_order():
        if key == stage_key:
            return label
    return stage_key.replace("_", " ")


def _gate_status(*, passed: bool, detail: str, evidence: list[str] | None = None) -> MonitoringReleaseGateEntry:
    return MonitoringReleaseGateEntry(
        status="pass" if passed else "fail",
        detail=detail,
        evidence=evidence or [],
    )


def build_release_observability(
    snapshot: SessionSnapshot,
    *,
    recent_error_records: list[ExecutionLogRecord],
    estimated_cost_usd: float = 0.0,
) -> MonitoringReleaseObservability:
    traced_runs = [item for item in snapshot.skill_runs if item.llm_trace is not None]
    total_llm_runs = len(traced_runs)
    fallback_runs = sum(1 for item in traced_runs if item.llm_trace is not None and item.llm_trace.fallback_used)
    degraded_runs = sum(1 for item in traced_runs if item.llm_trace is not None and item.llm_trace.degraded)
    real_llm_runs = max(total_llm_runs - fallback_runs, 0)
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    compaction_values: list[float] = []
    stage_order = _stage_order()
    stage_metrics: dict[str, dict[str, Any]] = {
        key: {
            "label": label,
            "run_count": 0,
            "success_count": 0,
            "needs_review_count": 0,
            "failure_count": 0,
            "approved_artifact_count": 0,
            "stale_artifact_count": 0,
            "rerun_count": 0,
            "long_term_hit_count": 0,
            "confidence_values": [],
            "simulation_run_count": 0,
            "simulation_pass_count": 0,
        }
        for key, label in stage_order
    }
    provider_metrics: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "run_count": 0,
            "fallback_count": 0,
            "degraded_count": 0,
            "long_term_hit_count": 0,
            "total_duration_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    capability_metrics: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "run_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "fallback_count": 0,
            "degraded_count": 0,
            "long_term_hit_count": 0,
        }
    )
    context_backend_counts: dict[str, int] = defaultdict(int)

    for run in traced_runs:
        trace = run.llm_trace
        if trace is None:
            continue
        stage_key, stage_label = _resolve_stage_for_run(run)
        if stage_key not in stage_metrics:
            stage_metrics[stage_key] = {
                "label": stage_label,
                "run_count": 0,
                "success_count": 0,
                "needs_review_count": 0,
                "failure_count": 0,
                "approved_artifact_count": 0,
                "stale_artifact_count": 0,
                "rerun_count": 0,
                "long_term_hit_count": 0,
                "confidence_values": [],
                "simulation_run_count": 0,
                "simulation_pass_count": 0,
            }
        stage_bucket = stage_metrics[stage_key]
        stage_bucket["run_count"] += 1
        if run.status == ArtifactStatus.ready:
            stage_bucket["success_count"] += 1
        elif run.status == ArtifactStatus.failed:
            stage_bucket["failure_count"] += 1
        else:
            stage_bucket["needs_review_count"] += 1
        if run.source_action.strip().lower().startswith("rerun:"):
            stage_bucket["rerun_count"] += 1
        if _has_long_term_hit(run):
            stage_bucket["long_term_hit_count"] += 1

        input_tokens = _as_int(trace.token_usage.get("input_tokens") or trace.token_usage.get("prompt_tokens"))
        output_tokens = _as_int(trace.token_usage.get("output_tokens") or trace.token_usage.get("completion_tokens"))
        trace_total_tokens = _as_int(trace.token_usage.get("total_tokens")) or (input_tokens + output_tokens)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_tokens += trace_total_tokens
        ratio = _compaction_ratio(run)
        if ratio is not None:
            compaction_values.append(ratio)

        provider_key = trace.provider_key or "unknown"
        model_name = trace.model_name or "unknown"
        execution_backend = trace.execution_backend or "unknown"
        context_backend = trace.effective_context_backend or "unknown"
        context_backend_counts[context_backend] += 1
        provider_bucket = provider_metrics[(provider_key, model_name, execution_backend, context_backend)]
        provider_bucket["run_count"] += 1
        provider_bucket["fallback_count"] += int(trace.fallback_used)
        provider_bucket["degraded_count"] += int(trace.degraded)
        provider_bucket["long_term_hit_count"] += int(_has_long_term_hit(run))
        provider_bucket["total_duration_ms"] += run.duration_ms
        provider_bucket["input_tokens"] += input_tokens
        provider_bucket["output_tokens"] += output_tokens
        provider_bucket["total_tokens"] += trace_total_tokens

        capability_key = trace.capability_key.strip() or run.skill_key or "unknown_capability"
        capability_label = run.label or capability_key.replace("_", " ")
        capability_bucket = capability_metrics[(capability_key, capability_label)]
        capability_bucket["run_count"] += 1
        capability_bucket["success_count"] += int(run.status == ArtifactStatus.ready)
        capability_bucket["failure_count"] += int(run.status == ArtifactStatus.failed)
        capability_bucket["fallback_count"] += int(trace.fallback_used)
        capability_bucket["degraded_count"] += int(trace.degraded)
        capability_bucket["long_term_hit_count"] += int(_has_long_term_hit(run))

    stale_artifact_count = 0
    rerun_count = sum(1 for item in snapshot.skill_runs if item.source_action.strip().lower().startswith("rerun:"))
    for artifact in snapshot.journey_artifacts:
        stage_key = artifact.stage_key
        if stage_key not in stage_metrics:
            stage_metrics[stage_key] = {
                "label": _stage_label(stage_key),
                "run_count": 0,
                "success_count": 0,
                "needs_review_count": 0,
                "failure_count": 0,
                "approved_artifact_count": 0,
                "stale_artifact_count": 0,
                "rerun_count": 0,
                "long_term_hit_count": 0,
                "confidence_values": [],
                "simulation_run_count": 0,
                "simulation_pass_count": 0,
            }
        stage_bucket = stage_metrics[stage_key]
        if artifact.state in {JourneyArtifactState.approved, JourneyArtifactState.approved_legacy}:
            stage_bucket["approved_artifact_count"] += 1
        if artifact.state == JourneyArtifactState.stale or artifact.stale_reasons:
            stage_bucket["stale_artifact_count"] += 1
            stale_artifact_count += 1
        if artifact.confidence is not None:
            stage_bucket["confidence_values"].append(float(artifact.confidence))

    simulation_run_count = len(snapshot.simulation_runs)
    simulation_pass_count = 0
    for run in snapshot.simulation_runs:
        validate_bucket = stage_metrics.setdefault(
            "validate",
            {
                "label": _stage_label("validate"),
                "run_count": 0,
                "success_count": 0,
                "needs_review_count": 0,
                "failure_count": 0,
                "approved_artifact_count": 0,
                "stale_artifact_count": 0,
                "rerun_count": 0,
                "long_term_hit_count": 0,
                "confidence_values": [],
                "simulation_run_count": 0,
                "simulation_pass_count": 0,
            },
        )
        validate_bucket["simulation_run_count"] += 1
        if run.final_status == "pass":
            validate_bucket["simulation_pass_count"] += 1
            simulation_pass_count += 1

    auth_or_isolation_error_count = sum(1 for item in recent_error_records if _is_auth_or_isolation_error(item))
    context_fingerprint_ok, context_fingerprint_total = _context_fingerprint_coverage(snapshot)
    source_versions_ok, source_versions_total = _source_version_coverage(snapshot)
    approvals_audited, approvals_resolved = _approval_audit_coverage(snapshot)
    hard_gate_violations = _hard_gate_overrides(snapshot)
    fallback_visibility_evidence = _fallback_visibility_evidence(snapshot, recent_error_records)
    budget_violations = [item.skill_key for item in traced_runs if _budget_exceeded(item)]

    providers = [
        MonitoringProviderObservabilityEntry(
            provider_key=provider_key,
            model_name=model_name,
            execution_backend=execution_backend,
            effective_context_backend=context_backend,
            run_count=data["run_count"],
            fallback_count=data["fallback_count"],
            degraded_count=data["degraded_count"],
            long_term_hit_count=data["long_term_hit_count"],
            total_duration_ms=data["total_duration_ms"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            total_tokens=data["total_tokens"],
        )
        for (provider_key, model_name, execution_backend, context_backend), data in sorted(
            provider_metrics.items(),
            key=lambda item: (-item[1]["run_count"], item[0][0], item[0][1]),
        )
    ]
    context_backends = [
        MonitoringContextBackendEntry(
            key=key,
            label=key.replace("_", " "),
            run_count=count,
            share_percent=_pct(count, total_llm_runs),
        )
        for key, count in sorted(context_backend_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    capabilities = [
        MonitoringCapabilityObservabilityEntry(
            capability_key=capability_key,
            label=capability_label,
            run_count=data["run_count"],
            success_count=data["success_count"],
            failure_count=data["failure_count"],
            fallback_count=data["fallback_count"],
            degraded_count=data["degraded_count"],
            long_term_hit_count=data["long_term_hit_count"],
        )
        for (capability_key, capability_label), data in sorted(
            capability_metrics.items(),
            key=lambda item: (-item[1]["run_count"], item[0][1]),
        )
    ]
    stages = [
        MonitoringStageObservabilityEntry(
            stage_key=stage_key,
            label=data["label"],
            run_count=data["run_count"],
            success_count=data["success_count"],
            needs_review_count=data["needs_review_count"],
            failure_count=data["failure_count"],
            approved_artifact_count=data["approved_artifact_count"],
            stale_artifact_count=data["stale_artifact_count"],
            rerun_count=data["rerun_count"],
            long_term_hit_count=data["long_term_hit_count"],
            average_confidence=_avg(data["confidence_values"]),
            simulation_run_count=data["simulation_run_count"],
            simulation_pass_rate=_pct(data["simulation_pass_count"], data["simulation_run_count"]),
        )
        for stage_key, data in sorted(
            stage_metrics.items(),
            key=lambda item: (
                next((index for index, pair in enumerate(stage_order) if pair[0] == item[0]), 999),
                item[0],
            ),
        )
    ]
    estimation_metric_count = len(snapshot.estimation_error_metrics)
    estimation_band_hit_rate = _pct(
        sum(1 for item in snapshot.estimation_error_metrics if item.band_hit_overall),
        estimation_metric_count,
    )

    release_gates = [
        MonitoringReleaseGateEntry(
            gate_key="context_fingerprint_coverage",
            label="Context fingerprint coverage",
            status="pass" if context_fingerprint_total > 0 and context_fingerprint_ok == context_fingerprint_total else "fail",
            detail=(
                f"{context_fingerprint_ok}/{context_fingerprint_total} artefactos de journey conservan context fingerprint."
                if context_fingerprint_total
                else "Aun no hay artefactos de journey para medir context fingerprint."
            ),
            evidence=[],
        ),
        MonitoringReleaseGateEntry(
            gate_key="source_version_coverage",
            label="Source versions coverage",
            status="pass" if source_versions_total > 0 and source_versions_ok == source_versions_total else "fail",
            detail=(
                f"{source_versions_ok}/{source_versions_total} propuestas conservan source_stage_versions."
                if source_versions_total
                else "Aun no hay propuestas de journey para medir source_stage_versions."
            ),
            evidence=[],
        ),
        MonitoringReleaseGateEntry(
            gate_key="approval_audit_coverage",
            label="Approval audit coverage",
            status="pass" if approvals_resolved == approvals_audited and approvals_resolved == len([item for item in snapshot.approvals if item.status != 'pending']) and not any(item.status == "pending" for item in snapshot.approvals) else "fail",
            detail=(
                f"{approvals_audited}/{approvals_resolved} approvals resueltos tienen resolved_at y resolution_note."
                if approvals_resolved
                else "No hay approvals resueltos todavia."
            ),
            evidence=[f"pending={sum(1 for item in snapshot.approvals if item.status == 'pending')}"],
        ),
        MonitoringReleaseGateEntry(
            gate_key="hard_gate_authority",
            label="Hard gate authority",
            status="pass" if not hard_gate_violations else "fail",
            detail=(
                "Ninguna simulacion permitio que el juicio LLM sobreescribiera un hard fail."
                if not hard_gate_violations
                else "Se detectaron simulaciones donde el estado final contradice el hard gate."
            ),
            evidence=hard_gate_violations[:3],
        ),
        MonitoringReleaseGateEntry(
            gate_key="fallback_visibility",
            label="Fallback visibility",
            status="pass" if fallback_runs == 0 or fallback_visibility_evidence > 0 else "fail",
            detail=(
                "No hubo fallbacks/degradaciones en la muestra actual."
                if fallback_runs == 0 and degraded_runs == 0
                else "Los fallbacks/degradaciones quedan trazados con evidencia visible."
                if fallback_visibility_evidence > 0
                else "Hubo fallbacks pero no se encontro evidencia visible en alertas, warnings o errores recientes."
            ),
            evidence=[f"fallback_runs={fallback_runs}", f"degraded_runs={degraded_runs}"],
        ),
        MonitoringReleaseGateEntry(
            gate_key="budget_respected",
            label="Context budget respected",
            status="pass" if not budget_violations else "fail",
            detail=(
                "No se detectaron corridas que superen el budget estimado de contexto."
                if not budget_violations
                else "Hay corridas que excedieron el budget estimado y requieren compaction/remediation."
            ),
            evidence=budget_violations[:3],
        ),
        MonitoringReleaseGateEntry(
            gate_key="auth_isolation_errors",
            label="Auth / isolation incidents",
            status="pass" if auth_or_isolation_error_count == 0 else "fail",
            detail=(
                "No se registraron errores recientes de auth o aislamiento."
                if auth_or_isolation_error_count == 0
                else f"Se registraron {auth_or_isolation_error_count} errores recientes de auth o aislamiento."
            ),
            evidence=[],
        ),
    ]

    return MonitoringReleaseObservability(
        total_llm_runs=total_llm_runs,
        real_llm_runs=real_llm_runs,
        fallback_runs=fallback_runs,
        fallback_rate=_pct(fallback_runs, total_llm_runs),
        degraded_runs=degraded_runs,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        average_latency_ms=_avg([float(item.duration_ms) for item in traced_runs]),
        estimated_cost_usd=round(estimated_cost_usd, 4),
        average_compaction_ratio=_avg(compaction_values),
        context_fingerprint_coverage=_pct(context_fingerprint_ok, context_fingerprint_total),
        source_version_coverage=_pct(source_versions_ok, source_versions_total),
        approval_resolution_rate=_pct(
            sum(1 for item in snapshot.approvals if item.status != "pending"),
            len(snapshot.approvals),
        ),
        stale_artifact_count=stale_artifact_count,
        rerun_count=rerun_count,
        long_term_hit_count=sum(1 for item in traced_runs if _has_long_term_hit(item)),
        simulation_run_count=simulation_run_count,
        simulation_pass_rate=_pct(simulation_pass_count, simulation_run_count),
        auth_or_isolation_error_count=auth_or_isolation_error_count,
        project_actuals_count=len(snapshot.project_actuals),
        estimation_error_metric_count=estimation_metric_count,
        estimation_band_hit_rate=estimation_band_hit_rate,
        context_backends=context_backends,
        providers=providers,
        stages=stages,
        capabilities=capabilities,
        release_gates=release_gates,
    )
