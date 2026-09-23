from __future__ import annotations

import re
from collections.abc import Iterable

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
) -> ObjectiveContractBundle:
    base = _base_objective(snapshot)
    objective = base
    for record in response_records or []:
        objective = _apply_objective_response(objective, record)
    return ObjectiveContractBundle(
        objectives=[objective],
        constraints=list(objective.constraint_refs),
        source_refs=list(objective.source_refs),
        active_objective_id=objective.objective_id,
    )


def active_objective(bundle: ObjectiveContractBundle) -> ObjectiveContract | None:
    for objective in bundle.objectives:
        if objective.objective_id == bundle.active_objective_id:
            return objective
    return bundle.objectives[0] if bundle.objectives else None


def objective_requires_runtime_loop(objective: ObjectiveContract | None) -> bool:
    return objective is not None and objective.runtime_tracking in {"recommended", "required"}
