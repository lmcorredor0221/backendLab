from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models import (
    MemoryDashboardEntry,
    MemoryObservabilityMetric,
    MemoryObservabilityReport,
    MemoryValidationCheckEntry,
    ShortTermMemoryRuntimeState,
    SkillRunEntry,
)
from app.services.memory_rollout import journey_stage_for_source_action
from app.services.memory_traceability import build_repo_document_lineage


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


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _status_for_percentage(value: float, *, good_floor: float = 80.0, warn_floor: float = 50.0) -> str:
    if value >= good_floor:
        return "ok"
    if value >= warn_floor:
        return "warning"
    return "critical"


def _budget_status(value: float) -> str:
    if value <= 80:
        return "ok"
    if value <= 95:
        return "warning"
    return "critical"


def _normalize_sources(run: SkillRunEntry) -> list[dict[str, Any]]:
    trace = run.llm_trace
    if trace is None:
        return []
    return [item for item in trace.context_used_sources if isinstance(item, dict)]


def _extract_budget_metrics(run: SkillRunEntry) -> tuple[float | None, float | None]:
    trace = run.llm_trace
    if trace is None:
        return None, None
    stats = trace.context_stats or {}
    budget_tokens = _as_float(stats.get("budget_tokens"))
    assembled_tokens = _as_float(stats.get("assembled_estimated_tokens"))
    reduction_tokens = _as_float(stats.get("reduction_estimated_tokens"))
    baseline_tokens = _as_float(stats.get("baseline_estimated_tokens")) or (assembled_tokens + reduction_tokens)
    budget_utilization = round((assembled_tokens / budget_tokens) * 100, 2) if budget_tokens > 0 else None
    compression_gain = round((reduction_tokens / baseline_tokens) * 100, 2) if baseline_tokens > 0 else None
    return budget_utilization, compression_gain


def _is_grounded_source(source: dict[str, Any]) -> bool:
    lineages = [item for item in source.get("source_lineage", []) if isinstance(item, str) and item.strip()]
    refs = [item for item in source.get("source_refs", []) if isinstance(item, str) and item.strip()]
    uri = str(source.get("uri", "")).strip()
    return bool(lineages or refs or uri.startswith("repo://") or uri.startswith("session://"))


def _has_citation_coverage(source: dict[str, Any]) -> bool:
    lineages = [item for item in source.get("source_lineage", []) if isinstance(item, str) and item.strip()]
    refs = [item for item in source.get("source_refs", []) if isinstance(item, str) and item.strip()]
    relative_path = str(source.get("relative_path", "")).strip()
    return bool(lineages or refs or relative_path)


def _repo_lineages_for_source(repo_root: Path, source: dict[str, Any]) -> tuple[list[str], list[str]]:
    refs = [item for item in source.get("source_refs", []) if isinstance(item, str) and item.strip()]
    repo_refs = [ref for ref in refs if not ref.startswith("session.")]
    current = [lineage for lineage in (build_repo_document_lineage(repo_root, ref) for ref in repo_refs) if lineage]
    stored = [
        item
        for item in source.get("source_lineage", [])
        if isinstance(item, str) and item.strip() and "::doc::" in item and item.split("::doc::", 1)[0] in repo_refs
    ]
    return stored, current


def _is_stale_source(repo_root: Path, source: dict[str, Any]) -> bool | None:
    stored, current = _repo_lineages_for_source(repo_root, source)
    if not stored or not current:
        return None
    return sorted(stored) != sorted(current)


def _is_contaminated_source(source: dict[str, Any]) -> bool:
    if bool(source.get("contaminated")):
        return True
    authority_level = str(source.get("authority_level", "")).strip().lower()
    if authority_level in {"untrusted", "expired", "shadow_only"}:
        return True
    return False


@dataclass
class _RunSummary:
    skill_key: str
    label: str
    stage: str
    journey_stage_key: str
    journey_stage_label: str
    grounded_sources: int
    traced_sources: int
    citation_sources: int
    stale_sources: int
    stale_eligible_sources: int
    contaminated_sources: int
    budget_utilization: float | None
    compression_gain: float | None
    has_required_source: bool


def _summarize_run(run: SkillRunEntry, *, repo_root: Path) -> _RunSummary:
    sources = _normalize_sources(run)
    grounded_sources = sum(1 for source in sources if _is_grounded_source(source))
    citation_sources = sum(1 for source in sources if _has_citation_coverage(source))
    stale_flags = [_is_stale_source(repo_root, source) for source in sources]
    stale_sources = sum(1 for flag in stale_flags if flag is True)
    stale_eligible_sources = sum(1 for flag in stale_flags if flag is not None)
    contaminated_sources = sum(1 for source in sources if _is_contaminated_source(source))
    has_required_source = any(bool(source.get("required")) for source in sources)
    budget_utilization, compression_gain = _extract_budget_metrics(run)
    stage_value = getattr(run.stage, "value", run.stage)
    journey_stage = journey_stage_for_source_action(run.source_action)
    journey_stage_key = journey_stage[0] if journey_stage is not None else str(stage_value)
    journey_stage_label = journey_stage[1] if journey_stage is not None else str(stage_value).replace("_", " ")
    return _RunSummary(
        skill_key=run.skill_key or "unknown_skill",
        label=run.label or run.skill_key or "Unknown agent",
        stage=str(stage_value),
        journey_stage_key=journey_stage_key,
        journey_stage_label=journey_stage_label,
        grounded_sources=grounded_sources,
        traced_sources=len(sources),
        citation_sources=citation_sources,
        stale_sources=stale_sources,
        stale_eligible_sources=stale_eligible_sources,
        contaminated_sources=contaminated_sources,
        budget_utilization=budget_utilization,
        compression_gain=compression_gain,
        has_required_source=has_required_source,
    )


def _build_dashboard(
    entries: list[_RunSummary],
    *,
    scope_kind: str,
    expected_entries: list[tuple[str, str]] | None = None,
) -> list[MemoryDashboardEntry]:
    grouped: dict[str, list[_RunSummary]] = defaultdict(list)
    for item in entries:
        key = item.skill_key if scope_kind == "agent" else item.journey_stage_key
        grouped[key].append(item)

    dashboard_by_key: dict[str, MemoryDashboardEntry] = {}
    for key, items in grouped.items():
        label = items[0].label if scope_kind == "agent" else items[0].journey_stage_label
        traced_sources = sum(item.traced_sources for item in items)
        stale_eligible = sum(item.stale_eligible_sources for item in items)
        dashboard_by_key[key] = MemoryDashboardEntry(
            scope_key=key,
            label=label,
            llm_runs=len(items),
            grounded_hit_rate=_pct(sum(1 for item in items if item.grounded_sources > 0), len(items)),
            citation_coverage=_pct(sum(item.citation_sources for item in items), traced_sources),
            stale_rate=_pct(sum(item.stale_sources for item in items), stale_eligible),
            average_budget_utilization=_avg(
                [item.budget_utilization for item in items if item.budget_utilization is not None]
            ),
            average_compression_gain=_avg(
                [item.compression_gain for item in items if item.compression_gain is not None]
            ),
        )
    if scope_kind == "stage" and expected_entries:
        for key, label in expected_entries:
            dashboard_by_key.setdefault(
                key,
                MemoryDashboardEntry(
                    scope_key=key,
                    label=label,
                    llm_runs=0,
                    grounded_hit_rate=0,
                    citation_coverage=0,
                    stale_rate=0,
                    average_budget_utilization=0,
                    average_compression_gain=0,
                ),
            )

    ordered_keys: list[str] = []
    if expected_entries:
        ordered_keys.extend(key for key, _ in expected_entries if key in dashboard_by_key)
    ordered_keys.extend(key for key in sorted(dashboard_by_key) if key not in ordered_keys)
    return [dashboard_by_key[key] for key in ordered_keys]


def _build_validation_checks(
    *,
    run_summaries: list[_RunSummary],
    recoverability_score: float,
    rollback_available: bool,
) -> list[MemoryValidationCheckEntry]:
    if not run_summaries:
        return [
            MemoryValidationCheckEntry(
                check_key="memory_runs_pending",
                label="Memory runs pending",
                status="not_applicable",
                summary="No existen corridas LLM suficientes para evaluar observabilidad de memoria.",
                evidence=["Sin skill runs con llm_trace persistido."],
            )
        ]

    needle_pass = next(
        (
            item
            for item in run_summaries
            if item.has_required_source and item.grounded_sources > 0 and (item.compression_gain or 0) > 0
        ),
        None,
    )
    long_context_pass = next(
        (
            item
            for item in run_summaries
            if (item.compression_gain or 0) >= 10 and item.grounded_sources > 0
        ),
        None,
    )
    contaminated_count = sum(item.contaminated_sources for item in run_summaries)
    stale_eligible = sum(item.stale_eligible_sources for item in run_summaries)
    stale_count = sum(item.stale_sources for item in run_summaries)

    checks = [
        MemoryValidationCheckEntry(
            check_key="needle_in_the_haystack_recovery",
            label="Needle in the haystack recovery",
            status="pass" if needle_pass is not None else "warning",
            summary=(
                f"La corrida `{needle_pass.skill_key}` recupero evidencia requerida bajo compresion."
                if needle_pass is not None
                else "No se encontro una corrida comprimida con evidencia requerida recuperada."
            ),
            evidence=(
                [
                    f"skill={needle_pass.skill_key}",
                    f"stage={needle_pass.stage}",
                    f"compression_gain={needle_pass.compression_gain or 0}%",
                ]
                if needle_pass is not None
                else ["Se necesitan corridas con `required=true` y trazas compactadas."]
            ),
        ),
        MemoryValidationCheckEntry(
            check_key="long_context_recovery",
            label="Long-context recovery",
            status="pass" if long_context_pass is not None else "warning",
            summary=(
                f"La corrida `{long_context_pass.skill_key}` mantuvo grounding con compresion >= 10%."
                if long_context_pass is not None
                else "No hay evidencia de recuperacion robusta en corridas con contexto largo."
            ),
            evidence=(
                [
                    f"skill={long_context_pass.skill_key}",
                    f"stage={long_context_pass.stage}",
                    f"compression_gain={long_context_pass.compression_gain or 0}%",
                ]
                if long_context_pass is not None
                else ["Validar corridas con baseline_estimated_tokens y reduction_estimated_tokens persistidos."]
            ),
        ),
        MemoryValidationCheckEntry(
            check_key="contaminated_memory_guard",
            label="Contaminated memory guard",
            status="pass" if contaminated_count == 0 else "fail",
            summary=(
                "No se detectaron fuentes contaminadas o sin autoridad permitida en las trazas."
                if contaminated_count == 0
                else f"Se detectaron {contaminated_count} fuentes contaminadas en el contexto usado."
            ),
            evidence=(
                ["authority_level distinto de `untrusted`, `expired` o `shadow_only` en todas las fuentes."]
                if contaminated_count == 0
                else [f"contaminated_source_count={contaminated_count}"]
            ),
        ),
        MemoryValidationCheckEntry(
            check_key="stale_source_invalidation",
            label="Stale source invalidation",
            status=(
                "not_applicable"
                if stale_eligible == 0
                else "pass" if stale_count == 0 else "fail"
            ),
            summary=(
                "Aun no hay suficientes fuentes repo-backed con lineage persistido para evaluar stale invalidation."
                if stale_eligible == 0
                else "No se detectaron fuentes obsoletas frente al estado actual del repo."
                if stale_count == 0
                else f"Se detectaron {stale_count} fuentes obsoletas frente al repo actual."
            ),
            evidence=(
                ["Persistir `source_lineage` y `source_refs` en corridas repo-backed para activar esta validacion."]
                if stale_eligible == 0
                else [f"stale_source_count={stale_count}", f"stale_eligible_sources={stale_eligible}"]
            ),
        ),
        MemoryValidationCheckEntry(
            check_key="short_term_recoverability",
            label="Short-term recoverability",
            status="pass" if recoverability_score >= 75 else "warning",
            summary=(
                "La memoria corta tiene checkpoints y rollback suficientes para recuperar el hilo operativo."
                if recoverability_score >= 75
                else "La memoria corta necesita mas señales de recuperacion consistente."
            ),
            evidence=[
                f"recoverability_score={recoverability_score}%",
                f"rollback_available={'true' if rollback_available else 'false'}",
            ],
        ),
    ]
    return checks


def build_memory_observability_report(
    *,
    skill_runs: list[SkillRunEntry],
    short_term_memory: ShortTermMemoryRuntimeState | None,
    repo_root: Path | None = None,
    expected_stages: list[tuple[str, str]] | None = None,
) -> MemoryObservabilityReport:
    resolved_repo_root = repo_root or Path(__file__).resolve().parents[3]
    traced_runs = [item for item in skill_runs if item.llm_trace is not None]
    run_summaries = [_summarize_run(item, repo_root=resolved_repo_root) for item in traced_runs]

    llm_run_count = len(run_summaries)
    traced_source_count = sum(item.traced_sources for item in run_summaries)
    grounded_hit_runs = sum(1 for item in run_summaries if item.grounded_sources > 0)
    citation_source_count = sum(item.citation_sources for item in run_summaries)
    stale_source_count = sum(item.stale_sources for item in run_summaries)
    stale_eligible_source_count = sum(item.stale_eligible_sources for item in run_summaries)
    budget_values = [item.budget_utilization for item in run_summaries if item.budget_utilization is not None]
    compression_values = [item.compression_gain for item in run_summaries if item.compression_gain is not None]

    recoverability_signals = [
        bool(short_term_memory and short_term_memory.resume_supported),
        bool(short_term_memory and short_term_memory.checkpoint_count > 0),
        bool(short_term_memory and short_term_memory.last_consistent_checkpoint_key),
        bool(short_term_memory and short_term_memory.branch_count > 0),
        bool(short_term_memory and short_term_memory.rollback_available),
    ]
    recoverability_score = _pct(sum(1 for flag in recoverability_signals if flag), len(recoverability_signals))

    metrics = [
        MemoryObservabilityMetric(
            key="token_budget_utilization",
            label="Token budget utilization",
            value=_avg(budget_values),
            numerator=int(round(sum(budget_values))),
            denominator=len(budget_values),
            status=_budget_status(_avg(budget_values)),
            detail="Promedio de uso del budget de contexto sobre corridas con `budget_tokens` persistido.",
        ),
        MemoryObservabilityMetric(
            key="hit_rate",
            label="Grounded hit rate",
            value=_pct(grounded_hit_runs, llm_run_count),
            numerator=grounded_hit_runs,
            denominator=llm_run_count,
            status=_status_for_percentage(_pct(grounded_hit_runs, llm_run_count)),
            detail="Porcentaje de corridas LLM con al menos una fuente grounded o trazable en el contexto usado.",
        ),
        MemoryObservabilityMetric(
            key="stale_rate",
            label="Stale source rate",
            value=_pct(stale_source_count, stale_eligible_source_count),
            numerator=stale_source_count,
            denominator=stale_eligible_source_count,
            status="ok" if stale_source_count == 0 else "critical",
            detail="Fuentes repo-backed cuyo lineage persistido ya no coincide con el contenido actual del repo.",
        ),
        MemoryObservabilityMetric(
            key="citation_coverage",
            label="Citation coverage",
            value=_pct(citation_source_count, traced_source_count),
            numerator=citation_source_count,
            denominator=traced_source_count,
            status=_status_for_percentage(_pct(citation_source_count, traced_source_count)),
            detail="Cobertura de `source_refs`, `source_lineage` o `relative_path` sobre las fuentes usadas por el runtime.",
        ),
        MemoryObservabilityMetric(
            key="recoverability",
            label="Recoverability",
            value=recoverability_score,
            numerator=sum(1 for flag in recoverability_signals if flag),
            denominator=len(recoverability_signals),
            status=_status_for_percentage(recoverability_score),
            detail="Senales de reanudacion, checkpoint consistente, ramas activas y rollback disponible en short-term memory.",
        ),
    ]

    recent_warnings: list[str] = []
    if stale_source_count > 0:
        recent_warnings.append(f"Se detectaron {stale_source_count} fuentes obsoletas en corridas repo-backed.")
    if traced_source_count > 0 and citation_source_count < traced_source_count:
        recent_warnings.append("Existen fuentes usadas sin cobertura completa de citas o lineage.")
    if recoverability_score < 75:
        recent_warnings.append("La memoria corta no cumple aun una recuperabilidad robusta en todos los checkpoints.")
    if llm_run_count > 0 and _avg(compression_values) == 0:
        recent_warnings.append("No hay reduccion de contexto observable en las corridas analizadas.")

    return MemoryObservabilityReport(
        llm_run_count=llm_run_count,
        traced_source_count=traced_source_count,
        grounded_hit_runs=grounded_hit_runs,
        stale_source_count=stale_source_count,
        recent_warnings=recent_warnings,
        metrics=metrics,
        by_agent=_build_dashboard(run_summaries, scope_kind="agent"),
        by_stage=_build_dashboard(
            run_summaries,
            scope_kind="stage",
            expected_entries=expected_stages,
        ),
        validations=_build_validation_checks(
            run_summaries=run_summaries,
            recoverability_score=recoverability_score,
            rollback_available=bool(short_term_memory and short_term_memory.rollback_available),
        ),
    )
