from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.models import EvidenceItem, EvidenceSource
from app.services.agentic_runtime.contracts import (
    BuilderActionRequest,
    BuilderActionResult,
    BuilderAgentRunResult,
    BuilderAgentState,
    BuilderEvaluation,
    BuilderIterationTrace,
    BuilderObservation,
)


def build_synthetic_react_run(trace: Any) -> BuilderAgentRunResult:
    """Project a legacy skill trace into ReAct vocabulary without exposing CoT."""
    status = getattr(getattr(trace, "status", None), "value", None) or str(getattr(trace, "status", "completed"))
    stage = getattr(getattr(trace, "stage", None), "value", None) or str(getattr(trace, "stage", ""))
    skill_key = str(getattr(trace, "skill_key", "skill"))
    warning_list = list(getattr(trace, "warnings", []) or [])
    output_payload = dict(getattr(trace, "output_payload", {}) or {})
    result_summary = str(getattr(trace, "result_summary", "") or "Resultado de capability sincronizado.")
    action = BuilderActionRequest(
        key="invoke_capability",
        stage=stage,
        capability=skill_key,
        arguments={"legacy_skill_key": skill_key},
        idempotency_key=f"legacy-skill:{skill_key}",
    )
    observation = BuilderObservation(
        action_key=action.key,
        status=status,
        summary=result_summary,
        output_refs=[skill_key],
        warnings=warning_list,
    )
    evaluation = BuilderEvaluation(
        status="finish" if status in {"ready", "completed", "draft", "needs_review"} else "fail",
        reason_summary="Capability ejecutada y evaluada por su estado y schema de salida.",
        confidence=1.0 if status not in {"failed", "error"} else 0.0,
        issues=warning_list,
        next_action="review" if status == "needs_review" else "finish_stage",
    )
    iteration = BuilderIterationTrace(
        iteration_id=f"legacy:{uuid4()}",
        iteration=1,
        reason_summary=f"Ejecutar la capability {skill_key} para la etapa {stage}.",
        action=action,
        observation=observation,
        evaluation=evaluation,
        duration_ms=int(getattr(trace, "duration_ms", 0) or 0),
    )
    llm_trace = getattr(trace, "llm_trace", None)
    llm_payload = (
        llm_trace.model_dump(mode="json")
        if hasattr(llm_trace, "model_dump")
        else llm_trace
        if isinstance(llm_trace, dict)
        else {}
    )
    context_stats = llm_payload.get("context_stats", {}) if isinstance(llm_payload, dict) else {}
    token_usage = int(context_stats.get("total_tokens", 0) or 0) if isinstance(context_stats, dict) else 0
    state = BuilderAgentState(
        run_id=uuid4(),
        stage=stage,
        capability=skill_key,
        status="completed" if evaluation.status == "finish" else "failed",
        iteration=1,
        llm_calls=1 if llm_trace is not None else 0,
        token_usage=token_usage,
        last_action=action.key,
        last_observation=observation.model_dump(mode="json"),
        last_evaluation=evaluation.model_dump(mode="json"),
    )
    return BuilderAgentRunResult(
        run_id=state.run_id,
        status="completed" if evaluation.status == "finish" else "failed",
        state=state,
        traces=[iteration],
        output=output_payload,
        message=result_summary,
    )


def build_react_evidence_manifest(result: BuilderAgentRunResult) -> list[EvidenceItem]:
    """Return safe operational evidence; never include model chain-of-thought."""
    return [
        EvidenceItem(
            source=EvidenceSource.react_runtime,
            detail=(
                f"run={result.run_id}; status={result.status}; "
                f"iterations={len(result.traces)}; checkpoint={result.checkpoint_id or 'none'}"
            ),
            metadata={
                "contract_version": result.contract_version,
                "status": result.status,
                "iterations": len(result.traces),
                "checkpoint_id": result.checkpoint_id,
                "actions": [item.action.key for item in result.traces],
            },
        )
    ]


def build_react_metrics(result: BuilderAgentRunResult) -> dict[str, Any]:
    """Return support metrics without persisting hidden model reasoning."""
    failures = sum(1 for item in result.traces if item.observation.status in {"failed", "retryable"})
    waiting_human = result.status == "waiting_human"
    return {
        "contract_version": "builder.react.metrics.v1",
        "run_id": str(result.run_id),
        "status": result.status,
        "iterations": len(result.traces),
        "llm_calls": result.state.llm_calls,
        "token_usage": result.state.token_usage,
        "failure_count": failures,
        "waiting_human": waiting_human,
        "checkpoint_id": result.checkpoint_id,
        "action_keys": [item.action.key for item in result.traces],
        "resume_scope": result.state.resume_scope,
    }
