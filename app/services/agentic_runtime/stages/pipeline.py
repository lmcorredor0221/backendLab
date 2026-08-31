from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID

from app.services.agent_i18n import detect_user_visible_language_status, get_effective_language
from app.services.agentic_runtime.contracts import (
    BuilderActionRequest,
    BuilderActionResult,
    BuilderAgentRunRequest,
    BuilderAgentRunResult,
    BuilderAgentState,
    BuilderQualityGateResult,
)
from app.services.agentic_runtime.controller import BuilderReActController
from app.services.agentic_runtime.stage_policy import StageAgentPolicy, get_stage_agent_policy


@dataclass
class ReactCapabilityOutput:
    value: Any = None
    traces: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""
    token_usage: int = 0


@dataclass
class ReactStageExecution:
    value: Any = None
    traces: list[Any] = field(default_factory=list)
    react_run: BuilderAgentRunResult | None = None
    warnings: list[str] = field(default_factory=list)


CapabilityRunner = Callable[[], ReactCapabilityOutput]
SecondaryRunner = Callable[[Any], ReactCapabilityOutput]
Validator = Callable[[Any], Any]


def _clamp_confidence(value: Any, *, fallback: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    try:
        return len(value)
    except TypeError:
        return 1


def _extract_confidence(value: Any) -> float | None:
    confidence = _field(value, "confidence")
    if isinstance(confidence, (int, float)):
        return _clamp_confidence(confidence)
    if confidence is not None:
        overall = _field(confidence, "overall")
        if overall is not None:
            return _clamp_confidence(overall)
    confidence_score = _field(value, "confidence_score")
    if confidence_score is not None:
        return _clamp_confidence(confidence_score)
    return None


def _count_pending_resolution(value: Any) -> int:
    pending = 0
    for name in ("open_questions", "guided_questions", "missing_information", "needs_information", "coverage_gaps"):
        pending += _safe_len(_field(value, name))
    for gap in _field(value, "dependency_gaps", []) or []:
        status = str(_field(gap, "status", "open") or "open").strip().lower()
        if status in {"", "open", "pending"}:
            pending += 1
    return pending


def _coerce_validation_result(validation: Any) -> tuple[list[str], bool, str, BuilderQualityGateResult | None]:
    if isinstance(validation, BuilderQualityGateResult):
        return list(validation.issues), bool(validation.blocking), validation.reason_summary, validation
    if isinstance(validation, tuple) and len(validation) >= 3:
        issues, blocking, summary = validation[:3]
        return [str(item) for item in issues or [] if str(item).strip()], bool(blocking), str(summary or ""), None
    if isinstance(validation, list):
        return [str(item) for item in validation if str(item).strip()], bool(validation), "", None
    return [], False, str(validation or ""), None


def _build_quality_gate(
    *,
    stage: str,
    capability: str,
    value: Any,
    validation: Any,
    policy: StageAgentPolicy,
    state: BuilderAgentState,
    cross_stage_remediation: bool,
    effective_language: str,
    runtime_warnings: list[str] | None = None,
) -> BuilderQualityGateResult:
    issues, blocking, summary, explicit_gate = _coerce_validation_result(validation)
    if explicit_gate is not None:
        return explicit_gate

    base_confidence = _extract_confidence(value)
    if base_confidence is None:
        base_confidence = 0.65 if issues or blocking else 0.9

    pending_resolution = _count_pending_resolution(value)
    evidence_confidence = _clamp_confidence(base_confidence - min(0.3, pending_resolution * 0.04))
    quality_confidence = base_confidence
    free_delegation_stage = stage in {"discover", "define", "design"}
    repair_policy = "none"
    language_status = (
        detect_user_visible_language_status(value, effective_language)
        if policy.language_gate
        else "not_checked"
    )
    if language_status == "mismatch":
        quality_confidence = min(quality_confidence, 0.74)
        issues.append("La salida visible no respeta el idioma configurado por el usuario.")

    normalized_warnings = [str(item) for item in runtime_warnings or [] if str(item).strip()]
    context_warnings = [
        item
        for item in normalized_warnings
        if any(
            marker in item.lower()
            for marker in (
                "context_budget",
                "knowledge_access_backend",
                "required_truncated",
                "truncated",
            )
        )
    ]
    has_required_truncation = any(
        marker in item.lower()
        for item in context_warnings
        for marker in ("required_truncated", "required context truncated")
    )
    if context_warnings:
        evidence_confidence = _clamp_confidence(evidence_confidence - 0.1)
    if has_required_truncation:
        quality_confidence = min(quality_confidence, 0.74)
        issues.append("El contexto requerido para la etapa fue truncado antes del LLM.")

    if (
        policy.allow_free_delegation_without_quality_penalty
        and free_delegation_stage
        and pending_resolution > 0
        and not blocking
    ):
        quality_confidence = max(quality_confidence, policy.quality_threshold)
        repair_policy = "document_and_delegate"

    quality_shortfall = quality_confidence < policy.quality_threshold
    if quality_shortfall:
        issues.append(
            f"La confianza de calidad ({quality_confidence:.2f}) esta por debajo del umbral "
            f"{policy.quality_threshold:.2f}."
        )
    minimum_repair_not_met = (
        policy.quality_gate_enabled
        and state.quality_repair_cycles > 0
        and state.quality_repair_cycles < policy.min_repair_cycles
    )
    can_repair_quality = (
        policy.quality_gate_enabled
        and not cross_stage_remediation
        and state.quality_repair_cycles < policy.max_quality_repair_cycles
        and (
            blocking
            or quality_shortfall
            or language_status == "mismatch"
            or has_required_truncation
            or minimum_repair_not_met
        )
    )
    if can_repair_quality:
        repair_policy = "language_repair" if language_status == "mismatch" else "react_repair"
    elif blocking:
        repair_policy = "attention_required"
    elif policy.quality_gate_enabled and (
        quality_shortfall or language_status == "mismatch" or has_required_truncation
    ):
        blocking = True
        repair_policy = "attention_required"

    flow_readiness = not blocking or repair_policy == "document_and_delegate"
    return BuilderQualityGateResult(
        stage=stage,
        capability=capability,
        quality_confidence=_clamp_confidence(quality_confidence),
        evidence_confidence=evidence_confidence,
        pending_resolution=pending_resolution,
        flow_readiness=flow_readiness,
        issues=list(dict.fromkeys(issues)),
        warnings=list(dict.fromkeys(normalized_warnings)),
        repair_policy=repair_policy,
        language_status=language_status,
        schema_status="invalid" if value is None else "valid",
        reason_summary=summary
        or (
            "La salida requiere completar el minimo de ciclos ReAct antes de continuar."
            if can_repair_quality and minimum_repair_not_met
            else
            "La salida requiere reparacion ReAct antes de continuar."
            if can_repair_quality
            else "La salida paso el gate de calidad provider-neutral."
        ),
        should_repair=can_repair_quality,
        blocking=blocking,
    )


def _model_payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": value}


def run_react_stage(
    *,
    stage: str,
    capability: str,
    session_id: UUID,
    workspace_id: UUID,
    context_refs: list[str],
    primary_runner: CapabilityRunner,
    validator: Validator,
    secondary_capability: str = "",
    secondary_runner: SecondaryRunner | None = None,
    remediation_action: str = "",
    initial_state: BuilderAgentState | None = None,
    effective_language: str = "es",
) -> ReactStageExecution:
    """Run one bounded ReAct loop around an existing stage capability.

    The skill runtime remains the source of truth for generation. This adapter
    only governs sequencing, validation, HITL and checkpoint boundaries.
    """

    policy = get_stage_agent_policy(stage)
    language = get_effective_language(effective_language)
    current_value: Any = None
    collected_traces: list[Any] = []
    warnings: list[str] = []
    last_runtime_warnings: list[str] = []

    def reasoner(request: BuilderAgentRunRequest, state: BuilderAgentState, previous: BuilderActionResult | None) -> BuilderActionRequest:
        previous_key = previous.key if previous else ""
        previous_output = previous.output if previous else {}
        if not previous_key:
            return BuilderActionRequest(key="retrieve_context", stage=stage)
        if previous is not None and previous.status == "retryable":
            if previous_key == "invoke_capability" and bool(previous_output.get("can_auto_retry", False)):
                return BuilderActionRequest(
                    key="invoke_capability",
                    stage=stage,
                    capability=capability,
                    arguments={"phase": "retry"},
                )
            if previous_key == "invoke_critique" and secondary_capability and bool(previous_output.get("can_auto_retry", False)):
                return BuilderActionRequest(
                    key="invoke_critique",
                    stage=stage,
                    capability=secondary_capability,
                    arguments={"phase": "retry"},
                )
            if previous_key == "run_validator" and bool(previous_output.get("quality_repair_allowed", False)):
                return BuilderActionRequest(
                    key="invoke_capability",
                    stage=stage,
                    capability=capability,
                    arguments={"phase": "quality_repair"},
                )
            if bool(previous_output.get("repairable", False)):
                return BuilderActionRequest(key="repair_structured_output", stage=stage)
            return BuilderActionRequest(
                key=remediation_action or "create_attention_decision",
                stage=stage,
                arguments={
                    "issues": [str(item) for item in previous_output.get("issues", []) if str(item).strip()],
                    "blocking": bool(previous_output.get("blocking", True)),
                },
            )
        if previous_key == "retrieve_context":
            return BuilderActionRequest(
                key="invoke_capability",
                stage=stage,
                capability=capability,
                arguments={"phase": "propose"},
            )
        if previous_key == "invoke_capability" and secondary_capability and secondary_runner is not None:
            return BuilderActionRequest(
                key="invoke_critique",
                stage=stage,
                capability=secondary_capability,
                arguments={"phase": "critique"},
            )
        if previous_key in {"invoke_capability", "invoke_critique", "repair_structured_output"}:
            return BuilderActionRequest(key="run_validator", stage=stage)
        if previous_key == "run_validator":
            issues = [str(item) for item in (previous.output if previous else {}).get("issues", []) if str(item).strip()]
            blocking = bool((previous.output if previous else {}).get("blocking", False))
            if blocking or issues:
                return BuilderActionRequest(
                    key=remediation_action or "create_attention_decision",
                    stage=stage,
                    arguments={"issues": issues, "blocking": blocking},
                )
            return BuilderActionRequest(key="persist_stage_artifact", stage=stage)
        if previous_key == "repair_structured_output":
            return BuilderActionRequest(key="run_validator", stage=stage)
        if previous_key == "persist_stage_artifact":
            return BuilderActionRequest(key="finish_stage", stage=stage)
        if previous_key in {"create_attention_decision", "raise_cross_stage_remediation"}:
            return BuilderActionRequest(key="checkpoint", stage=stage)
        return BuilderActionRequest(key="finish_stage", stage=stage)

    def execute(action: BuilderActionRequest, _state: BuilderAgentState) -> BuilderActionResult:
        nonlocal current_value, collected_traces, warnings, last_runtime_warnings
        if action.key == "retrieve_context":
            return BuilderActionResult(
                key=action.key,
                output={"output_refs": list(context_refs)},
                summary="Contexto aprobado y memoria compacta recuperados.",
            )
        if action.key == "invoke_capability":
            try:
                result = primary_runner()
            except Exception as exc:  # noqa: BLE001
                can_auto_retry = _state.llm_calls < 1
                return BuilderActionResult(
                    key=action.key,
                    status="retryable",
                    output={
                        "issues": [f"No se pudo ejecutar {capability}: {type(exc).__name__}"],
                        "blocking": True,
                        "can_auto_retry": can_auto_retry,
                        "retry_attempt": _state.llm_calls + 1,
                    },
                    summary=(
                        f"La capability {capability} fallo; se intentara un reintento gobernado."
                        if can_auto_retry
                        else f"La capability {capability} fallo despues del reintento y requiere recuperacion guiada."
                    ),
                    error_kind="provider_or_schema_failure",
                )
            current_value = result.value
            collected_traces.extend(result.traces)
            warnings.extend(result.warnings)
            last_runtime_warnings = list(result.warnings)
            return BuilderActionResult(
                key=action.key,
                output={"artifact": _model_payload(current_value), "output_refs": list(context_refs)},
                summary=result.summary or f"Capability {capability} ejecutada.",
                warnings=list(result.warnings),
                token_usage=result.token_usage,
            )
        if action.key == "invoke_critique":
            if current_value is None:
                return BuilderActionResult(
                    key=action.key,
                    status="failed",
                    summary="No existe una propuesta para criticar.",
                    error_kind="missing_stage_output",
                )
            try:
                result = secondary_runner(current_value) if secondary_runner is not None else ReactCapabilityOutput(value=current_value)
            except Exception as exc:  # noqa: BLE001
                can_auto_retry = _state.llm_calls < 2
                return BuilderActionResult(
                    key=action.key,
                    status="retryable",
                    output={
                        "issues": [f"No se pudo ejecutar {secondary_capability}: {type(exc).__name__}"],
                        "blocking": True,
                        "can_auto_retry": can_auto_retry,
                        "retry_attempt": _state.llm_calls + 1,
                    },
                    summary=(
                        f"La critica {secondary_capability} fallo; se intentara un reintento gobernado."
                        if can_auto_retry
                        else f"La critica {secondary_capability} fallo despues del reintento y requiere recuperacion guiada."
                    ),
                    error_kind="provider_or_schema_failure",
                )
            current_value = result.value
            collected_traces.extend(result.traces)
            warnings.extend(result.warnings)
            last_runtime_warnings = list(result.warnings)
            return BuilderActionResult(
                key=action.key,
                output={"artifact": _model_payload(current_value), "output_refs": list(context_refs)},
                summary=result.summary or f"Critica {secondary_capability} completada.",
                warnings=list(result.warnings),
                token_usage=result.token_usage,
            )
        if action.key == "run_validator":
            validation = validator(current_value)
            quality_gate = _build_quality_gate(
                stage=stage,
                capability=capability,
                value=current_value,
                validation=validation,
                policy=policy,
                state=_state,
                cross_stage_remediation=False,
                effective_language=language,
                runtime_warnings=last_runtime_warnings,
            )
            return BuilderActionResult(
                key=action.key,
                status="retryable" if quality_gate.should_repair else "success",
                output={
                    "issues": list(quality_gate.issues),
                    "blocking": quality_gate.blocking,
                    "quality_gate": quality_gate.model_dump(mode="json"),
                    "quality_repair_allowed": quality_gate.should_repair,
                    "quality_repair_requested": quality_gate.should_repair,
                    "output_refs": list(context_refs),
                },
                summary=quality_gate.reason_summary,
            )
        if action.key == "repair_structured_output":
            if current_value is None:
                return BuilderActionResult(
                    key=action.key,
                    status="failed",
                    output={"issues": ["No existe una salida estructurada reparable."], "blocking": True},
                    summary="La salida requiere una decision humana.",
                    error_kind="missing_stage_output",
                )
            return BuilderActionResult(
                key=action.key,
                output={"artifact": _model_payload(current_value)},
                summary="La salida estructurada fue normalizada antes de revalidar.",
            )
        if action.key in {"create_attention_decision", "raise_cross_stage_remediation"}:
            validation = validator(current_value)
            quality_gate = _build_quality_gate(
                stage=stage,
                capability=capability,
                value=current_value,
                validation=validation,
                policy=policy,
                state=_state,
                cross_stage_remediation=True,
                effective_language=language,
                runtime_warnings=last_runtime_warnings,
            )
            return BuilderActionResult(
                key=action.key,
                output={
                    "issues": list(quality_gate.issues),
                    "blocking": quality_gate.blocking,
                    "quality_gate": quality_gate.model_dump(mode="json"),
                    "output_refs": [f"attention.{stage}"],
                },
                summary=(
                    "La incompatibilidad entre etapas fue derivada a una remediacion guiada."
                    if action.key == "raise_cross_stage_remediation"
                    else "La salida fue derivada a una decision guiada de Atencion."
                ),
            )
        if action.key in {"persist_stage_artifact", "finish_stage"}:
            return BuilderActionResult(
                key=action.key,
                summary="La persistencia transaccional queda a cargo del endpoint de la etapa.",
                side_effect_applied=False,
            )
        if action.key == "checkpoint":
            return BuilderActionResult(
                key=action.key,
                status="waiting_human",
                summary="Checkpoint ReAct preparado para resolver la decision de Atencion.",
                output={"issues": [f"La etapa {stage} requiere una decision humana."], "blocking": True},
            )
        return BuilderActionResult(key=action.key, status="failed", summary="Accion no soportada por el adaptador ReAct.", error_kind="unsupported_stage_action")

    request = BuilderAgentRunRequest(
        session_id=session_id,
        workspace_id=workspace_id,
        stage=stage,
        capability=capability,
        mode="resume" if initial_state is not None else "run",
        checkpoint_id=initial_state.checkpoint_id if initial_state is not None else "",
        context_refs=list(context_refs),
    )
    react_run = BuilderReActController().run(
        request,
        execute,
        reasoner=reasoner,
        initial_state=initial_state,
    )
    return ReactStageExecution(
        value=current_value,
        traces=collected_traces,
        react_run=react_run,
        warnings=list(dict.fromkeys(warnings)),
    )
