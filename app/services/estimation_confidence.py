from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.models import ConfidenceBreakdown, EstimationConfidenceLabel, EstimationMaturityStage, ReviewState, SessionSnapshot

if TYPE_CHECKING:
    from app.services.estimation_service import EstimationSignals


DEFAULT_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "base_canvas": 40,
    "base_blueprint": 58,
    "base_ready_to_build": 76,
    "scope_v1_scope_multiplier": 4,
    "scope_constraint_multiplier": 2,
    "scope_cap": 16,
    "technical_tool_multiplier": 3,
    "technical_workflow_multiplier": 2,
    "technical_safety_multiplier": 1,
    "technical_memory_multiplier": 2,
    "technical_cap": 18,
    "operations_evaluation_case_multiplier": 1,
    "operations_observability_multiplier": 1,
    "operations_ready_bonus": 4,
    "operations_cap": 12,
    "delivery_deliverable_multiplier": 1,
    "delivery_ready_bonus": 4,
    "delivery_cap": 10,
    "readiness_complete_delta": 6,
    "readiness_partial_delta": 0,
    "readiness_blocked_delta": -8,
    "maturity_complete_delta": 5,
    "maturity_partial_delta": 1,
    "maturity_missing_delta": -8,
    "maturity_blocked_delta": -12,
    "maturity_not_applicable_delta": 3,
    "blocking_gap_penalty": 4,
    "design_gap_penalty": 4,
    "implementation_gap_penalty": 2,
    "open_question_penalty": 1,
    "design_open_question_penalty": 2,
    "implementation_open_question_penalty": 1,
    "open_question_penalty_cap": 6,
    "assumption_penalty": 2,
    "assumption_penalty_cap": 12,
    "score_floor": 5,
    "score_ceiling": 96,
    "subscore_complete": 96,
    "subscore_partial": 62,
    "subscore_missing": 24,
    "subscore_blocked": 12,
    "subscore_not_applicable": 100,
}


def build_confidence_breakdown(
    *,
    signals: EstimationSignals,
    snapshot: SessionSnapshot,
    confidence_bands: list[dict[str, Any]],
    confidence_weights: dict[str, float] | None = None,
) -> ConfidenceBreakdown:
    weights = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    if confidence_weights:
        weights.update(confidence_weights)

    base_score = _base_score_for_stage(signals.maturity_stage, weights)
    scope_signal = min(
        int(_weight(weights, "scope_cap")),
        (
            _discovery_scope_count(snapshot) * _weight(weights, "scope_v1_scope_multiplier")
            + _discovery_constraint_count(snapshot) * _weight(weights, "scope_constraint_multiplier")
        ),
    )
    technical_signal = min(
        int(_weight(weights, "technical_cap")),
        (
            signals.tool_count * _weight(weights, "technical_tool_multiplier")
            + signals.workflow_steps * _weight(weights, "technical_workflow_multiplier")
            + signals.safety_checks * _weight(weights, "technical_safety_multiplier")
            + signals.memory_layers * _weight(weights, "technical_memory_multiplier")
        ),
    )
    operations_signal = min(
        int(_weight(weights, "operations_cap")),
        (
            signals.evaluation_cases * _weight(weights, "operations_evaluation_case_multiplier")
            + signals.observability_signals * _weight(weights, "operations_observability_multiplier")
            + (_weight(weights, "operations_ready_bonus") if signals.maturity_stage == EstimationMaturityStage.ready_to_build else 0)
        ),
    )
    delivery_signal = min(
        int(_weight(weights, "delivery_cap")),
        (
            signals.deliverables * _weight(weights, "delivery_deliverable_multiplier")
            + (_weight(weights, "delivery_ready_bonus") if signals.maturity_stage == EstimationMaturityStage.ready_to_build else 0)
        ),
    )
    readiness_adjustment = _readiness_delta(signals.readiness_state, weights)
    maturity_adjustment = sum(
        _maturity_delta(state, weights)
        for state in (
            signals.api_contract_maturity,
            signals.deployment_maturity,
            signals.knowledge_maturity,
        )
    )
    design_gap_count = _signal_int(signals, "design_gap_count", signals.blocking_gaps)
    implementation_gap_count = _signal_int(signals, "implementation_gap_count", 0)
    design_open_questions = _signal_int(signals, "design_open_questions", signals.open_questions)
    implementation_open_questions = _signal_int(signals, "implementation_open_questions", 0)
    residual_gap_penalty = (
        design_gap_count * _weight(weights, "design_gap_penalty")
        + implementation_gap_count * _weight(weights, "implementation_gap_penalty")
    )
    residual_question_penalty = min(
        (
            design_open_questions * _weight(weights, "design_open_question_penalty")
            + implementation_open_questions * _weight(weights, "implementation_open_question_penalty")
        ),
        _weight(weights, "open_question_penalty_cap"),
    )
    penalties = residual_gap_penalty + residual_question_penalty + min(
        signals.assumptions_count * _weight(weights, "assumption_penalty"),
        _weight(weights, "assumption_penalty_cap"),
    )

    score = int(
        max(
            _weight(weights, "score_floor"),
            min(
                _weight(weights, "score_ceiling"),
                round(base_score + scope_signal + technical_signal + operations_signal + delivery_signal + readiness_adjustment + maturity_adjustment - penalties),
            ),
        )
    )
    band = _resolve_confidence_band(score, confidence_bands)

    subscores = {
        "scope_definition": _clamp_score(base_score + scope_signal + readiness_adjustment * 2),
        "technical_definition": _clamp_score(24 + technical_signal * 4),
        "operational_readiness": _clamp_score(
            20 + operations_signal * 5 + readiness_adjustment * 2 - residual_gap_penalty
        ),
        "delivery_readiness": _clamp_score(
            18 + delivery_signal * 6 + readiness_adjustment * 2 - residual_question_penalty
        ),
        "api_contract_maturity": _maturity_subscore(signals.api_contract_maturity, weights),
        "deployment_maturity": _maturity_subscore(signals.deployment_maturity, weights),
        "knowledge_maturity": _maturity_subscore(signals.knowledge_maturity, weights),
        "build_readiness": _readiness_subscore(signals.readiness_state, weights),
        "design_uncertainty": _clamp_score(100 - design_gap_count * 4 - design_open_questions * 2),
        "implementation_uncertainty": _clamp_score(100 - implementation_gap_count * 2 - implementation_open_questions),
    }

    return ConfidenceBreakdown(
        score=score,
        label=EstimationConfidenceLabel(band["label_key"]),
        uncertainty_band_percent=band["uncertainty_band_max_percent"],
        blocking_gaps=signals.blocking_gaps,
        open_questions=signals.open_questions,
        design_gap_count=design_gap_count,
        implementation_gap_count=implementation_gap_count,
        design_open_questions=design_open_questions,
        implementation_open_questions=implementation_open_questions,
        assumptions_count=signals.assumptions_count,
        subscores=subscores,
        positive_signals=signals.positive_signals,
        negative_signals=signals.negative_signals,
        recommended_next_actions=_build_recommended_next_actions(signals),
    )


def _build_recommended_next_actions(signals: EstimationSignals) -> list[str]:
    actions: list[str] = []
    if signals.maturity_stage == EstimationMaturityStage.canvas:
        actions.append("Completar blueprint tecnico para subir confianza desde Canvas a Blueprint.")
    if signals.api_contract_maturity in {"missing", "partial", "blocked"}:
        actions.append("Cerrar contratos API, sandbox y side effects antes de fijar una fecha o costo comprometido.")
    if signals.deployment_maturity in {"missing", "partial", "blocked"}:
        actions.append("Definir deployment target, secretos y runbook operativo antes de cerrar la propuesta comercial.")
    if signals.knowledge_maturity in {"missing", "partial", "blocked"}:
        actions.append("Definir owner, fuentes, refresh policy y estrategia semantica de knowledge/retrieval.")
    if signals.evaluation_cases == 0:
        actions.append("Crear dataset y rubrica minima para aterrizar el esfuerzo QA y la cobertura automatizable.")
    if signals.blocking_gaps > 0 or signals.open_questions > 0:
        actions.append("Documentar las preguntas residuales en el ACP y resolver solo las que dependan del entorno durante implementacion.")
    return _dedupe(actions)[:5]


def _resolve_confidence_band(score: int, confidence_bands: list[dict[str, Any]]) -> dict[str, Any]:
    for band in confidence_bands:
        if band["min_score"] <= score <= band["max_score"]:
            return band
    return {
        "label_key": EstimationConfidenceLabel.low.value,
        "uncertainty_band_max_percent": 60,
    }


def _base_score_for_stage(stage: EstimationMaturityStage, weights: dict[str, float]) -> float:
    return _weight(weights, f"base_{stage.value}")


def _readiness_delta(state: ReviewState, weights: dict[str, float]) -> float:
    if state == ReviewState.complete:
        return _weight(weights, "readiness_complete_delta")
    if state == ReviewState.blocked:
        return _weight(weights, "readiness_blocked_delta")
    return _weight(weights, "readiness_partial_delta")


def _maturity_delta(state: str, weights: dict[str, float]) -> float:
    mapping = {
        "complete": "maturity_complete_delta",
        "partial": "maturity_partial_delta",
        "missing": "maturity_missing_delta",
        "blocked": "maturity_blocked_delta",
        "not_applicable": "maturity_not_applicable_delta",
    }
    return _weight(weights, mapping.get(state, "maturity_partial_delta"))


def _maturity_subscore(state: str, weights: dict[str, float]) -> int:
    mapping = {
        "complete": "subscore_complete",
        "partial": "subscore_partial",
        "missing": "subscore_missing",
        "blocked": "subscore_blocked",
        "not_applicable": "subscore_not_applicable",
    }
    return int(_weight(weights, mapping.get(state, "subscore_partial")))


def _readiness_subscore(state: ReviewState, weights: dict[str, float]) -> int:
    if state == ReviewState.complete:
        return int(_weight(weights, "subscore_complete"))
    if state == ReviewState.blocked:
        return int(_weight(weights, "subscore_blocked"))
    return int(_weight(weights, "subscore_partial"))


def _discovery_scope_count(snapshot: SessionSnapshot) -> int:
    if snapshot.discovery is None:
        return 0
    return len(snapshot.discovery.mvp_definition.v1_scope)


def _discovery_constraint_count(snapshot: SessionSnapshot) -> int:
    if snapshot.discovery is None:
        return 0
    return len(snapshot.discovery.constraints)


def _weight(weights: dict[str, float], key: str) -> float:
    return float(weights.get(key, DEFAULT_CONFIDENCE_WEIGHTS[key]))


def _signal_int(signals: "EstimationSignals", attr_name: str, fallback: int) -> int:
    value = getattr(signals, attr_name, fallback)
    return int(value or 0)


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        token = item.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered
