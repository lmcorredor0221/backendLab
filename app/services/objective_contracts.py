from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from app.models import (
    ConstructionQuestionResponseRecord,
    ObjectiveContract,
    ObjectiveContractBundle,
    ObjectiveSuccessCriterion,
    ObjectiveTerminationConditions,
    OperationalCapabilityProfile,
    SessionSnapshot,
)

OBJECTIVE_QUESTIONS_FLAG = "objective_questions_v1"
OBJECTIVE_GATE_FLAG = "objective_gate_v1"
OBJECTIVE_LOOP_FLAG = "objective_loop_v1"


AGENT_OBJECTIVE_TEMPLATES: dict[str, tuple[dict[str, Any], ...]] = {
    "supervisor_with_subagents": (
        {
            "agent_key": "supervisor",
            "role": "supervisor",
            "purpose": "Asignar especialistas, fusionar hallazgos y decidir cierre o remediacion.",
            "success_signals": ["Despacha solo especialistas necesarios y conserva trazabilidad de handoffs."],
            "output_contracts": ["orchestration_plan", "final_decision"],
            "failure_mode": "Merge ambiguo o falta de ownership claro para cerrar.",
            "timeout_policy": "SLA corto para despachar y SLA medio para consolidar findings.",
            "side_effect_policy": "El supervisor no ejecuta side effects directos; solo autoriza handoffs y merge final.",
        },
        {
            "agent_key": "evaluation_specialist",
            "role": "specialist",
            "purpose": "Revisar readiness, acceptance cases y gaps antes del cierre final.",
            "success_signals": ["Expone gaps, score y recomendacion de readiness sin tocar estado externo."],
            "output_contracts": ["evaluation_findings"],
            "failure_mode": "Readiness ambigua o dataset insuficiente.",
            "timeout_policy": "Timeout medio controlado por el supervisor.",
            "side_effect_policy": "Solo lectura sobre evaluation-pack y prompt-pack.",
        },
        {
            "agent_key": "risk_specialist",
            "role": "specialist",
            "purpose": "Validar tools, side effects, approvals y compensaciones antes de permitir promotion.",
            "success_signals": ["Cada tool queda clasificada por riesgo, approval y compensacion."],
            "output_contracts": ["risk_findings"],
            "failure_mode": "Tool sin contrato, timeout o compensacion declarada.",
            "timeout_policy": "Timeout corto por analisis de tool.",
            "side_effect_policy": "Solo inspeccion contractual; no puede ejecutar side effects sobre sistemas externos.",
        },
        {
            "agent_key": "artifact_specialist",
            "role": "specialist",
            "purpose": "Revisar coherencia de artefactos, prompts y handoffs tecnicos antes del export.",
            "success_signals": ["Confirma presencia y coherencia de artefactos, prompts y handoffs."],
            "output_contracts": ["artifact_findings"],
            "failure_mode": "Prompts faltantes o paquete inconsistente.",
            "timeout_policy": "Timeout corto de revision documental.",
            "side_effect_policy": "Solo lectura sobre artefactos canonicamente exportados.",
        },
    ),
    "router_parallel": (
        {
            "agent_key": "router",
            "role": "router",
            "purpose": "Clasificar el trabajo y derivarlo a una rama especializada sin ejecutar side effects.",
            "success_signals": ["Selecciona solo ramas justificadas por el contrato."],
            "output_contracts": ["route_plan"],
            "failure_mode": "Ruta ambigua o conflicto entre ramas.",
            "timeout_policy": "Timeout corto para clasificacion inicial.",
            "side_effect_policy": "El router no ejecuta side effects.",
        },
        {
            "agent_key": "retrieval_lane",
            "role": "specialist",
            "purpose": "Recuperar evidencia autorizada para la rama de knowledge o retrieval.",
            "success_signals": ["Entrega evidencia citada o ausencia explicita de evidencia."],
            "output_contracts": ["retrieval_findings"],
            "failure_mode": "Respuesta sin grounding o falta de fuentes aprobadas.",
            "timeout_policy": "Timeout medio para consultas paralelas.",
            "side_effect_policy": "Solo lectura con evidencia citada.",
        },
        {
            "agent_key": "tool_lane",
            "role": "specialist",
            "purpose": "Validar el contrato de una tool y devolver findings sin mezclarlo con retrieval.",
            "success_signals": ["Cada tool queda evaluada con permisos, retries y side effects."],
            "output_contracts": ["tool_findings"],
            "failure_mode": "Tool sin contrato suficiente o con side effects no aislados.",
            "timeout_policy": "Timeout medio por rama.",
            "side_effect_policy": "Solo inspeccion contractual mientras la topologia siga planned_only.",
        },
        {
            "agent_key": "aggregator",
            "role": "aggregator",
            "purpose": "Fusionar findings paralelos y preparar el cierre o remediacion.",
            "success_signals": ["La fusion final identifica contradicciones y propone cierre seguro."],
            "output_contracts": ["parallel_merge_report"],
            "failure_mode": "Merge ambiguo o ramas inconsistentes.",
            "timeout_policy": "Timeout corto al consolidar resultados.",
            "side_effect_policy": "Sin side effects directos.",
        },
    ),
}


def _normalize(value: object) -> str:
    return " ".join(str(value or "").split())


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = _normalize(value)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _slug(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized[:48] or fallback


def _agent_value(agent_contract: Any, key: str, default: Any = "") -> Any:
    if isinstance(agent_contract, dict):
        return agent_contract.get(key, default)
    return getattr(agent_contract, key, default)


def _agent_contracts_from_snapshot(snapshot: SessionSnapshot) -> list[dict[str, Any]]:
    architecture = ""
    if snapshot.blueprint is not None:
        architecture = _normalize(snapshot.blueprint.architecture)
    return [dict(item) for item in AGENT_OBJECTIVE_TEMPLATES.get(architecture, ())]


def _agent_label(agent_key: str) -> str:
    return agent_key.replace("_", " ").replace("-", " ").strip().title() or "Agente"


def _sentence_case_lower(value: str) -> str:
    if not value:
        return value
    return value[:1].lower() + value[1:]


def _feature_enabled(snapshot: SessionSnapshot, flag_key: str) -> bool:
    for item in snapshot.workspace_contract.feature_flags:
        if item.key == flag_key:
            return bool(item.enabled)
    return False


def objective_questions_enabled(snapshot: SessionSnapshot) -> bool:
    return _feature_enabled(snapshot, OBJECTIVE_QUESTIONS_FLAG)


def objective_gate_enabled(snapshot: SessionSnapshot) -> bool:
    return _feature_enabled(snapshot, OBJECTIVE_GATE_FLAG)


def objective_loop_enabled(snapshot: SessionSnapshot) -> bool:
    return _feature_enabled(snapshot, OBJECTIVE_LOOP_FLAG)


def _profile_from_snapshot(snapshot: SessionSnapshot) -> OperationalCapabilityProfile:
    if snapshot.blueprint is not None and snapshot.blueprint.operational_profile.archetype_key:
        return snapshot.blueprint.operational_profile
    if snapshot.discovery is not None and snapshot.discovery.operational_profile.archetype_key:
        return snapshot.discovery.operational_profile
    return OperationalCapabilityProfile()


def _objective_statement_and_sources(snapshot: SessionSnapshot) -> tuple[str, list[str], float]:
    if snapshot.canvas is not None and _normalize(snapshot.canvas.user_goal):
        refs = ["canvas.user_goal"]
        if snapshot.discovery is not None and _normalize(snapshot.discovery.desired_outcome):
            refs.append("discovery.desired_outcome")
        return _normalize(snapshot.canvas.user_goal), refs, 0.84
    if snapshot.discovery is not None and _normalize(snapshot.discovery.desired_outcome):
        return _normalize(snapshot.discovery.desired_outcome), ["discovery.desired_outcome"], 0.72
    if _normalize(snapshot.session.title):
        return _normalize(snapshot.session.title), ["session.title"], 0.48
    return "Definir y completar el objetivo operativo aprobado por el usuario.", ["derived"], 0.35


def _success_criteria(snapshot: SessionSnapshot, statement: str) -> list[ObjectiveSuccessCriterion]:
    criteria: list[ObjectiveSuccessCriterion] = []
    if snapshot.canvas is not None and _normalize(snapshot.canvas.success_metric):
        criteria.append(
            ObjectiveSuccessCriterion(
                criterion_id="canvas-success-metric",
                statement=_normalize(snapshot.canvas.success_metric),
                evidence_refs=["canvas.success_metric"],
                verification_method="business_metric",
            )
        )
        for index, metric in enumerate(snapshot.canvas.agent_profile.success_metrics, start=1):
            if _normalize(metric):
                criteria.append(
                    ObjectiveSuccessCriterion(
                        criterion_id=f"agent-profile-metric-{index}",
                        statement=_normalize(metric),
                        evidence_refs=["canvas.agent_profile.success_metrics"],
                        verification_method="business_metric",
                    )
                )
    if snapshot.evaluation_dataset is not None:
        for index, case in enumerate(snapshot.evaluation_dataset.cases[:3], start=1):
            if _normalize(case.expected_result):
                criteria.append(
                    ObjectiveSuccessCriterion(
                        criterion_id=f"evaluation-case-{index}",
                        statement=_normalize(case.expected_result),
                        evidence_refs=[case.case_key or f"evaluation_dataset.cases.{index}"],
                        verification_method="evaluation_case",
                    )
                )
    if not criteria:
        criteria.append(
            ObjectiveSuccessCriterion(
                criterion_id="verified-outcome",
                statement=f"Existe evidencia verificable de que se cumplio: {statement}",
                evidence_refs=["derived"],
                verification_method="evidence_required",
            )
        )
    return criteria


def _runtime_tracking(snapshot: SessionSnapshot, profile: OperationalCapabilityProfile) -> str:
    text = " ".join(
        [
            snapshot.blueprint.reasoning_pattern if snapshot.blueprint is not None else "",
            snapshot.blueprint.architecture if snapshot.blueprint is not None else "",
            " ".join(profile.required_capabilities),
            " ".join(profile.action_capabilities),
        ]
    ).lower()
    if any(token in text for token in ("browser_execute", "browser_observe", "supervisor", "checkpoint", "react")):
        return "required"
    if any(token in text for token in ("plan-and-execute", "plan_execute", "handoff", "tool")):
        return "recommended"
    return "not_required"


def _base_objective(snapshot: SessionSnapshot) -> ObjectiveContract:
    statement, source_refs, confidence = _objective_statement_and_sources(snapshot)
    profile = _profile_from_snapshot(snapshot)
    success_criteria = _success_criteria(snapshot, statement)
    runtime_tracking = _runtime_tracking(snapshot, profile)
    progress_signals = ["objective_loaded"]
    if "business_policy_evaluation" in profile.required_capabilities:
        progress_signals.append("policy_validated")
    if "action_verification" in profile.required_capabilities:
        progress_signals.append("action_verified")
    stop_conditions = ["insufficient_evidence", "constraint_violation"]
    if profile.archetype_key == "business_ui_operator" or "business_policy_evaluation" in profile.required_capabilities:
        stop_conditions.extend(["policy_denied", "approval_required"])
    if runtime_tracking == "required":
        stop_conditions.append("no_progress_limit_reached")
    return ObjectiveContract(
        objective_id=f"obj-{_slug(statement, fallback='operational-goal')}",
        level="operational",
        statement=statement,
        owner="business_owner",
        status="inferred",
        source_refs=source_refs,
        confidence=confidence,
        success_criteria=success_criteria,
        constraint_refs=_dedupe(
            [
                *(snapshot.discovery.constraints if snapshot.discovery is not None else []),
                *(snapshot.blueprint.guardrails if snapshot.blueprint is not None else []),
            ]
        ),
        termination_conditions=ObjectiveTerminationConditions(
            success=[f"{item.criterion_id} satisfied" for item in success_criteria],
            stop=_dedupe(stop_conditions),
        ),
        progress_signals=_dedupe(progress_signals),
        mutation_policy="human_approval_required" if profile.archetype_key == "business_ui_operator" else "bounded_replanning",
        runtime_tracking=runtime_tracking,  # type: ignore[arg-type]
        version=1,
    )


def _success_criteria_from_agent(
    *,
    agent_key: str,
    success_signals: Sequence[Any],
    statement: str,
) -> list[ObjectiveSuccessCriterion]:
    criteria = [
        ObjectiveSuccessCriterion(
            criterion_id=f"{_slug(agent_key, fallback='agent')}-signal-{index}",
            statement=_normalize(signal),
            evidence_refs=[f"behavior_spec.multi_agent_topology.agent_contracts.{agent_key}.success_signals"],
            verification_method="agent_success_signal",
        )
        for index, signal in enumerate(success_signals, start=1)
        if _normalize(signal)
    ]
    if criteria:
        return criteria
    return [
        ObjectiveSuccessCriterion(
            criterion_id=f"{_slug(agent_key, fallback='agent')}-verified-output",
            statement=f"Existe evidencia verificable de que el subobjetivo se cumplio: {statement}",
            evidence_refs=[f"behavior_spec.multi_agent_topology.agent_contracts.{agent_key}"],
            verification_method="agent_output_evidence",
        )
    ]


def _delegated_objectives_from_agent_contracts(
    root_objective: ObjectiveContract,
    agent_contracts: Sequence[Any],
) -> list[ObjectiveContract]:
    objectives: list[ObjectiveContract] = []
    seen_ids: set[str] = {root_objective.objective_id}
    for index, agent_contract in enumerate(agent_contracts, start=1):
        agent_key = _normalize(_agent_value(agent_contract, "agent_key")) or f"agent_{index}"
        purpose = _normalize(_agent_value(agent_contract, "purpose"))
        if not purpose:
            continue
        role = _normalize(_agent_value(agent_contract, "role")) or "agent"
        objective_id = f"{root_objective.objective_id}-{_slug(agent_key, fallback=f'agent-{index}')}"
        if objective_id in seen_ids:
            continue
        seen_ids.add(objective_id)
        success_signals = [
            _normalize(item)
            for item in (_agent_value(agent_contract, "success_signals", []) or [])
            if _normalize(item)
        ]
        output_contracts = [
            _normalize(item)
            for item in (_agent_value(agent_contract, "output_contracts", []) or [])
            if _normalize(item)
        ]
        failure_mode = _normalize(_agent_value(agent_contract, "failure_mode"))
        timeout_policy = _normalize(_agent_value(agent_contract, "timeout_policy"))
        side_effect_policy = _normalize(_agent_value(agent_contract, "side_effect_policy"))
        statement = f"{_agent_label(agent_key)} debe {_sentence_case_lower(purpose.rstrip('.'))}."
        criteria = _success_criteria_from_agent(
            agent_key=agent_key,
            success_signals=success_signals,
            statement=statement,
        )
        objectives.append(
            ObjectiveContract(
                objective_id=objective_id,
                level="delegated",
                parent_objective_id=root_objective.objective_id,
                statement=statement,
                owner=agent_key,
                status="inferred",
                source_refs=_dedupe(
                    [
                        "blueprint.architecture",
                        f"behavior_spec.multi_agent_topology.agent_contracts.{agent_key}",
                    ]
                ),
                confidence=min(max(root_objective.confidence - 0.04, 0.45), 0.82),
                success_criteria=criteria,
                constraint_refs=_dedupe(
                    [
                        *root_objective.constraint_refs,
                        side_effect_policy,
                    ]
                ),
                termination_conditions=ObjectiveTerminationConditions(
                    success=[f"{item.criterion_id} satisfied" for item in criteria],
                    stop=_dedupe(
                        [
                            "parent_objective_rejected",
                            "handoff_conflict",
                            failure_mode,
                            timeout_policy,
                        ]
                    ),
                ),
                progress_signals=_dedupe([*success_signals, *[f"emits:{item}" for item in output_contracts]]),
                mutation_policy=(
                    "human_approval_required"
                    if "approval" in side_effect_policy.lower() or "aproba" in side_effect_policy.lower()
                    else "bounded_replanning"
                ),
                runtime_tracking=(
                    "required" if root_objective.runtime_tracking == "required" else "recommended"
                ),
                version=1,
            )
        )
    return objectives


def _decision_context(record: ConstructionQuestionResponseRecord) -> dict:
    return dict(record.decision_context or {})


def _apply_objective_response(objective: ObjectiveContract, record: ConstructionQuestionResponseRecord) -> ObjectiveContract:
    if record.status == "open":
        return objective
    context = _decision_context(record)
    if context.get("question_kind") != "objective_validation":
        return objective
    subject_id = str(context.get("subject_id") or "")
    if subject_id and subject_id != objective.objective_id:
        return objective
    decision = str(context.get("domain_decision") or context.get("selected_option_key") or "").strip().lower()
    answer = _normalize(record.answer_text)
    if decision == "reject":
        return objective.model_copy(update={"status": "rejected", "version": max(objective.version + 1, int(context.get("contract_version", 1)))})
    if decision == "confirm" or record.status in {"answered", "resolved"} and not answer:
        return objective.model_copy(update={"status": "confirmed"})
    if answer and record.status in {"answered", "resolved"}:
        return objective.model_copy(
            update={
                "statement": answer,
                "status": "confirmed",
                "source_refs": _dedupe([*objective.source_refs, "acp.questions.objective_validation"]),
                "version": objective.version + 1,
            }
        )
    return objective


def build_objective_contract_bundle(
    snapshot: SessionSnapshot,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
    agent_contracts: Sequence[Any] | None = None,
) -> ObjectiveContractBundle:
    base = _base_objective(snapshot)
    objectives = [
        base,
        *_delegated_objectives_from_agent_contracts(
            base,
            agent_contracts if agent_contracts is not None else _agent_contracts_from_snapshot(snapshot),
        ),
    ]
    for record in response_records or []:
        objectives = [_apply_objective_response(objective, record) for objective in objectives]
    constraints = _dedupe(
        constraint
        for objective in objectives
        for constraint in objective.constraint_refs
    )
    source_refs = _dedupe(
        source_ref
        for objective in objectives
        for source_ref in objective.source_refs
    )
    return ObjectiveContractBundle(
        objectives=objectives,
        constraints=constraints,
        source_refs=source_refs,
        active_objective_id=base.objective_id,
    )


def active_objective(bundle: ObjectiveContractBundle) -> ObjectiveContract | None:
    for objective in bundle.objectives:
        if objective.objective_id == bundle.active_objective_id:
            return objective
    return bundle.objectives[0] if bundle.objectives else None


def objective_requires_runtime_loop(objective: ObjectiveContract | None) -> bool:
    return objective is not None and objective.runtime_tracking in {"recommended", "required"}
