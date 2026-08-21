from __future__ import annotations

from collections.abc import Callable
from time import monotonic
from typing import Any
from uuid import uuid4

from app.services.agentic_runtime.action_registry import BuilderActionRegistry
from app.services.agentic_runtime.contracts import (
    BuilderActionRequest,
    BuilderActionResult,
    BuilderAgentRunRequest,
    BuilderAgentRunResult,
    BuilderAgentState,
    BuilderEvaluation,
    BuilderIterationTrace,
    BuilderObservation,
)
from app.services.agentic_runtime.guards import (
    BuilderLoopGuardConfig,
    BuilderLoopGuardState,
    BuilderLoopGuardViolation,
    BuilderLoopGuards,
    build_idempotency_key,
)
from app.services.agentic_runtime.stage_policy import get_stage_agent_policy


ActionExecutor = Callable[[BuilderActionRequest, BuilderAgentState], BuilderActionResult]
Reasoner = Callable[
    [BuilderAgentRunRequest, BuilderAgentState, BuilderActionResult | None],
    BuilderActionRequest,
]


def _result_from_value(action: BuilderActionRequest, value: Any) -> BuilderActionResult:
    if isinstance(value, BuilderActionResult):
        return value
    if isinstance(value, dict):
        return BuilderActionResult.model_validate({"key": action.key, **value})
    return BuilderActionResult(key=action.key, output={"value": value}, summary="Accion completada.")


def _default_next_action(
    request: BuilderAgentRunRequest,
    state: BuilderAgentState,
    previous: BuilderActionResult | None,
) -> BuilderActionRequest:
    previous_key = previous.key if previous else ""
    output = previous.output if previous else {}
    issues = [str(item) for item in output.get("issues", []) if str(item).strip()]
    blocking = bool(output.get("blocking", False))

    if not previous_key:
        key = "retrieve_context"
    elif previous is not None and previous.status == "retryable":
        if previous_key == "invoke_capability" and bool(output.get("can_auto_retry", False)):
            key = "invoke_capability"
        elif bool(output.get("repairable", False)):
            key = "repair_structured_output"
        else:
            key = "create_attention_decision"
    elif previous is not None and previous.status == "failed":
        key = "create_attention_decision"
    elif previous_key == "retrieve_context":
        key = "invoke_capability"
    elif previous_key == "invoke_capability":
        key = "run_validator"
    elif previous_key == "run_validator":
        if blocking or issues:
            key = "create_attention_decision"
        else:
            key = "persist_stage_artifact"
    elif previous_key == "repair_structured_output":
        key = "run_validator"
    elif previous_key == "persist_stage_artifact":
        key = "finish_stage"
    elif previous_key == "create_attention_decision":
        key = "checkpoint"
    elif previous_key == "checkpoint":
        key = "checkpoint"
    elif previous_key == "finish_stage":
        key = "finish_stage"
    else:
        key = "finish_stage"

    return BuilderActionRequest(
        key=key,
        stage=request.stage,
        capability=request.capability if key == "invoke_capability" else "",
        arguments={"phase": "retry"} if key == "invoke_capability" and previous_key == "invoke_capability" else {},
    )


class BuilderReActController:
    """Framework-neutral ReAct controller for internal builder capabilities.

    The controller records operational summaries only. It deliberately does not
    capture chain-of-thought or expose hidden model reasoning to callers.
    """

    def __init__(
        self,
        *,
        registry: BuilderActionRegistry | None = None,
        guards: BuilderLoopGuards | None = None,
    ):
        self.registry = registry or BuilderActionRegistry()
        self.guards = guards

    def run(
        self,
        request: BuilderAgentRunRequest,
        executor: ActionExecutor,
        *,
        reasoner: Reasoner | None = None,
        initial_state: BuilderAgentState | None = None,
    ) -> BuilderAgentRunResult:
        policy = get_stage_agent_policy(request.stage)
        guards = self.guards or BuilderLoopGuards(
            BuilderLoopGuardConfig(
                max_iterations=policy.max_iterations,
                max_llm_calls=policy.max_llm_calls,
            )
        )
        runtime = BuilderLoopGuardState()
        state = initial_state or BuilderAgentState(
            run_id=request.run_id,
            session_id=request.session_id,
            workspace_id=request.workspace_id,
            stage=request.stage,
            capability=request.capability,
            context_refs=list(request.context_refs),
        )
        state = state.model_copy(
            update={
                "status": "running",
                "updated_at": state.updated_at,
                "checkpoint_id": request.checkpoint_id or state.checkpoint_id,
            }
        )
        traces: list[BuilderIterationTrace] = []
        output: dict[str, Any] = {}
        previous: BuilderActionResult | None = (
            BuilderActionResult(
                key="retrieve_context",
                status="success",
                summary="Contexto aprobado ya estaba disponible antes de la pausa HITL.",
            )
            if request.mode == "resume" and request.checkpoint_id
            else None
        )

        if request.mode == "resume" and request.checkpoint_id:
            state = state.model_copy(update={"resume_action": request.capability, "resume_scope": request.stage})

        for _ in range(policy.max_iterations + 1):
            action = (reasoner or _default_next_action)(request, state, previous)
            if not action.idempotency_key and action.key in {
                "persist_stage_artifact",
                "finish_stage",
                "checkpoint",
                "resume_from_checkpoint",
            }:
                action = action.model_copy(
                    update={
                        "idempotency_key": build_idempotency_key(
                            request.run_id, action.key, state.iteration
                        ),
                        "side_effect": True,
                    }
                )

            started = monotonic()
            try:
                self.registry.assert_allowed(action)
                guards.before_action(state=state, action=action, runtime=runtime)
                guards.record_action(runtime, action)
                if request.mode == "dry_run" and action.key in {
                    "persist_stage_artifact",
                    "finish_stage",
                    "checkpoint",
                    "resume_from_checkpoint",
                }:
                    result = BuilderActionResult(
                        key=action.key,
                        summary="Dry run: side effect omitido.",
                        side_effect_applied=False,
                    )
                else:
                    result = _result_from_value(action, executor(action, state))
            except (BuilderLoopGuardViolation, ValueError) as exc:
                result = BuilderActionResult(
                    key=action.key,
                    status="failed",
                    summary="La accion fue detenida por una guardia de ejecucion.",
                    error_kind=getattr(exc, "code", "action_rejected"),
                    warnings=[str(exc)],
                )
            except Exception as exc:  # noqa: BLE001
                result = BuilderActionResult(
                    key=action.key,
                    status="failed",
                    summary="La accion fallo y se deriva a una decision de atencion.",
                    error_kind=type(exc).__name__,
                    warnings=["El detalle tecnico se conserva en trazabilidad; no se expone como razonamiento del agente."],
                    output={"issues": [str(exc)], "blocking": True},
                )

            duration_ms = int((monotonic() - started) * 1000)
            observation = BuilderObservation(
                action_key=action.key,
                status=result.status,
                summary=result.summary,
                output_refs=[str(item) for item in result.output.get("output_refs", [])],
                warnings=list(result.warnings),
                error_kind=result.error_kind,
                token_usage=result.token_usage,
            )
            evaluation = self._evaluate(action, result)
            state = state.model_copy(
                update={
                    "status": "running",
                    "iteration": state.iteration + 1,
                    "llm_calls": state.llm_calls + (1 if action.key in {"invoke_capability", "invoke_critique"} else 0),
                    "token_usage": state.token_usage + result.token_usage,
                    "last_action": action.key,
                    "last_observation": observation.model_dump(mode="json"),
                    "last_evaluation": evaluation.model_dump(mode="json"),
                }
            )
            traces.append(
                BuilderIterationTrace(
                    iteration_id=f"{request.run_id}:{state.iteration}:{uuid4().hex[:8]}",
                    iteration=state.iteration,
                    reason_summary=f"Evaluar la salida de {request.capability} y decidir el siguiente paso.",
                    action=action,
                    observation=observation,
                    evaluation=evaluation,
                    duration_ms=duration_ms,
                )
            )
            output.update(result.output)
            previous = result

            if evaluation.status == "waiting_human":
                checkpoint_id = state.checkpoint_id or f"react:{request.run_id}:checkpoint:{state.iteration}"
                state = state.model_copy(
                    update={
                        "status": "waiting_human",
                        "checkpoint_id": checkpoint_id,
                        "resume_action": request.capability,
                        "resume_scope": request.stage,
                    }
                )
                return BuilderAgentRunResult(
                    run_id=request.run_id,
                    status="waiting_human",
                    state=state,
                    traces=traces,
                    output=output,
                    checkpoint_id=checkpoint_id,
                    message="La ejecucion quedo pausada hasta resolver una decision de atencion.",
                )
            if evaluation.status == "fail":
                state = state.model_copy(update={"status": "failed"})
                return BuilderAgentRunResult(
                    run_id=request.run_id,
                    status="failed",
                    state=state,
                    traces=traces,
                    output=output,
                    message=evaluation.reason_summary,
                )
            if evaluation.status == "finish":
                state = state.model_copy(update={"status": "completed"})
                return BuilderAgentRunResult(
                    run_id=request.run_id,
                    status="completed",
                    state=state,
                    traces=traces,
                    output=output,
                    message="La ejecucion ReAct termino correctamente.",
                )

        state = state.model_copy(update={"status": "failed"})
        return BuilderAgentRunResult(
            run_id=request.run_id,
            status="failed",
            state=state,
            traces=traces,
            output=output,
            message="El loop ReAct alcanzo el maximo de iteraciones.",
        )

    @staticmethod
    def _evaluate(action: BuilderActionRequest, result: BuilderActionResult) -> BuilderEvaluation:
        issues = [str(item) for item in result.output.get("issues", []) if str(item).strip()]
        if action.key in {"create_attention_decision", "raise_cross_stage_remediation"}:
            return BuilderEvaluation(
                status="continue",
                reason_summary=(
                    "La remediacion entre etapas fue registrada; se persiste el checkpoint antes de pausar."
                    if action.key == "raise_cross_stage_remediation"
                    else "La decision HITL fue registrada; se persiste el checkpoint antes de pausar."
                ),
                confidence=0.0,
                issues=issues,
                next_action="checkpoint",
            )
        if action.key == "checkpoint" or result.status == "waiting_human":
            return BuilderEvaluation(
                status="waiting_human",
                reason_summary="La salida requiere una decision humana antes de continuar.",
                confidence=0.0,
                issues=issues,
                next_action="checkpoint",
            )
        if result.status == "failed":
            return BuilderEvaluation(
                status="fail",
                reason_summary=result.summary or "La ejecucion fallo.",
                confidence=0.0,
                issues=issues,
            )
        if action.key == "finish_stage":
            return BuilderEvaluation(
                status="finish",
                reason_summary="La salida paso las validaciones de la etapa.",
                confidence=1.0,
            )
        return BuilderEvaluation(
            status="continue",
            reason_summary="La salida es procesable; continua el flujo gobernado.",
            confidence=0.8,
            issues=issues,
            next_action="continue",
        )
