from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from app.contracts.canonical_v1 import (
    BehaviorSpecV1,
    BehaviorState,
    CanonicalProvenanceEntry,
    EvaluationCaseV1,
    HeuristicDecisionFact,
    HeuristicDecisionV1,
    KnowledgeContractV1,
    LLMFunctionPolicy,
    LLMPolicyV1,
    MemoryPolicyV1,
    MultiAgentBenchmarkMetricV1,
    MultiAgentBenchmarkV1,
    MultiAgentExecutionBudgetV1,
    MultiAgentHandoffContractV1,
    MultiAgentMessageContractV1,
    MultiAgentPermissionBoundaryV1,
    MultiAgentRoleContractV1,
    MultiAgentSharedStateContractV1,
    MultiAgentTopologyV1,
    PromptArtifactV1,
    PromptPackOrigin,
    PromptPackV1,
    PromptVariable,
    SuccessCriterion,
    ToolContractV1,
)
from app.models import PatternCatalogEntry, SessionSnapshot

_PROMPT_ARTIFACT_CACHE: dict[tuple[str, str], PromptArtifactV1] = {}
_SUPPORTED_REASONING_PATTERNS = {"Plan-and-Execute", "ReAct"}
_PLANNED_REASONING_PATTERNS = {"HTN", "Reflexion", "ToT"}
_SUPPORTED_MULTI_AGENT_ARCHITECTURES = {"supervisor_with_subagents"}
_PLANNED_MULTI_AGENT_ARCHITECTURES = {"router_parallel"}
_ROLE_MEMORY_TASK_KINDS = {
    "planner": "planning_runtime",
    "executor": "execution_runtime",
    "evaluator": "evaluation_runtime",
    "tool_use": "tool_runtime",
    "memory": "memory_runtime",
    "retrieval": "retrieval_runtime",
    "recovery": "recovery_runtime",
}
_ROLE_CONTRACT_CONTEXT_SOURCES = {
    "planner": ["blueprint-core.v1", "behavior-spec.v1", "heuristic-decision.v1", "short-term-memory.v1"],
    "executor": ["behavior-spec.v1", "memory-policy.v1", "short-term-memory.v1"],
    "evaluator": ["evaluation-pack.v1", "behavior-spec.v1", "heuristic-decision.v1", "short-term-memory.v1"],
    "tool_use": ["tool-contract.v1", "llm-policy.v1", "short-term-memory.v1"],
    "memory": ["memory-policy.v1", "behavior-spec.v1", "short-term-memory.v1", "knowledge-manifest.v1"],
    "retrieval": ["knowledge-contract.v1", "memory-policy.v1", "knowledge-manifest.v1", "short-term-memory.v1"],
    "recovery": ["behavior-spec.v1", "heuristic-decision.v1", "evaluation-pack.v1", "short-term-memory.v1"],
}


@dataclass(frozen=True)
class RoleAssembledContext:
    role: str
    task_kind: str
    knowledge_access_backend: str
    context_sources: list[str]
    required_source_keys: list[str]
    candidate_source_keys: list[str]
    source_digests: list[str]
    prompt_digest_lines: list[str]


def latest_blueprint_version(snapshot: SessionSnapshot) -> int | None:
    versions = snapshot.blueprint_versions or []
    if not versions:
        return None
    return max(item.version_number for item in versions)


def base_metadata(snapshot: SessionSnapshot, generated_at: Any) -> dict[str, Any]:
    return {
        "source_session_id": snapshot.session.id,
        "generated_at": generated_at,
        "source_blueprint_version": latest_blueprint_version(snapshot),
    }


def provenance(*entries: tuple[str, list[str], str]) -> list[CanonicalProvenanceEntry]:
    return [
        CanonicalProvenanceEntry(target_path=target_path, source_paths=source_paths, note=note)
        for target_path, source_paths, note in entries
    ]


def normalized_items(items: Iterable[str] | None) -> list[str]:
    return [item.strip() for item in items or [] if isinstance(item, str) and item.strip()]


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for raw in items:
        token = str(raw or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        values.append(token)
    return values


def coalesce(*values: str | None, fallback: str = "") -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def stable_hash_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: stable_hash_payload(item)
            for key, item in value.items()
            if key not in {"source_session_id", "generated_at"}
        }
    if isinstance(value, list):
        return [stable_hash_payload(item) for item in value]
    return value


def evaluation_case_key(case: Any, index: int) -> str:
    raw = getattr(case, "case_key", None) or getattr(case, "name", None) or getattr(case, "title", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return f"case_{index + 1}"


def evaluation_cases(snapshot: SessionSnapshot) -> list[EvaluationCaseV1]:
    if snapshot.evaluation_dataset is not None and snapshot.evaluation_dataset.cases:
        return [
            EvaluationCaseV1(
                key=evaluation_case_key(case, index),
                title=case.title,
                category=case.category,
                scenario=case.scenario,
                expected_result=case.expected_result,
            )
            for index, case in enumerate(snapshot.evaluation_dataset.cases)
        ]

    if snapshot.evaluation is not None and snapshot.evaluation.cases:
        return [
            EvaluationCaseV1(
                key=evaluation_case_key(case, index),
                title=case.name,
                category=case.category,
                scenario=case.scenario,
                expected_result=case.expected_result,
            )
            for index, case in enumerate(snapshot.evaluation.cases)
        ]

    return []


def prompt_variables(tool_contracts: Sequence[ToolContractV1], success_criteria: Sequence[SuccessCriterion]) -> list[PromptVariable]:
    variables = [
        PromptVariable(
            name="goal",
            description="Objetivo principal del blueprint en la ejecucion actual.",
            source_paths=["discovery.desired_outcome", "canvas.user_goal"],
        ),
        PromptVariable(
            name="constraints",
            description="Restricciones obligatorias del proyecto.",
            source_paths=["discovery.constraints", "blueprint.guardrails"],
        ),
        PromptVariable(
            name="workflow_states",
            description="Estados o nodos visibles que el compilador debe respetar.",
            source_paths=["behavior-spec.v1.states"],
        ),
    ]
    if tool_contracts:
        variables.append(
            PromptVariable(
                name="approved_tools",
                description="Herramientas aprobadas y sujetas a contratos versionados.",
                source_paths=["tool-contract.v1"],
            )
        )
    if success_criteria:
        variables.append(
            PromptVariable(
                name="success_criteria",
                description="Criterios medibles de exito y cierre.",
                source_paths=["canvas.success_metric", "evaluation-pack.v1.cases"],
            )
        )
    return variables


def prompt_stop_conditions(snapshot: SessionSnapshot) -> list[str]:
    conditions = [
        "Detenerse si falta evidencia estructural para responder o decidir.",
        "Detenerse si una accion contradice constraints, approvals o guardrails aprobados.",
    ]
    if any(item.status == "pending" for item in snapshot.approvals):
        conditions.append("Detenerse cuando una accion con side effects requiera approval pendiente.")
    return conditions


def deliverable_map(snapshot: SessionSnapshot) -> dict[str, str]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return {}
    return {
        item.key: item.content_markdown.strip()
        for item in blueprint.delivery_package.deliverables
        if item.key and item.content_markdown.strip()
    }


def derive_success_criteria(snapshot: SessionSnapshot) -> list[SuccessCriterion]:
    criteria: list[SuccessCriterion] = []
    if snapshot.canvas is not None and snapshot.canvas.success_metric.strip():
        criteria.append(
            SuccessCriterion(
                key="success_metric",
                description=snapshot.canvas.success_metric.strip(),
                source="canvas.success_metric",
            )
        )
    if snapshot.discovery is not None and snapshot.discovery.mvp_definition.north_star_metric.strip():
        criteria.append(
            SuccessCriterion(
                key="north_star_metric",
                description=snapshot.discovery.mvp_definition.north_star_metric.strip(),
                source="discovery.mvp_definition.north_star_metric",
            )
        )
    if snapshot.canvas is not None:
        for index, metric in enumerate(normalized_items(snapshot.canvas.agent_profile.success_metrics), start=1):
            criteria.append(
                SuccessCriterion(
                    key=f"agent_profile_metric_{index}",
                    description=metric,
                    source="canvas.agent_profile.success_metrics",
                )
            )
    return criteria or [
        SuccessCriterion(
            key="readiness",
            description="Mantener un blueprint coherente, trazable y listo para handoff controlado.",
            source="derived",
        )
    ]


def render_system_prompt(snapshot: SessionSnapshot, success_criteria: Sequence[SuccessCriterion], compiler_label: str) -> str:
    deliverables = deliverable_map(snapshot)
    if deliverables.get("system_prompt"):
        return deliverables["system_prompt"]

    criteria = "\n".join(f"- {item.description}" for item in success_criteria) or "- Mantener trazabilidad del blueprint."
    return "\n".join(
        [
            "Actua como constructor Lean de blueprints agenticos.",
            "Trabaja solo con contratos versionados, evidencia aprobada y restricciones trazables.",
            f"Modo de comportamiento compilado: {compiler_label}.",
            "No inventes tools, memoria, prompts ni politicas fuera del contrato.",
            "Criterios de exito:",
            criteria,
        ]
    )


def format_states(states: Sequence[BehaviorState]) -> str:
    return "\n".join(
        f"- {state.name}: {state.objective} (actor={state.actor}; outputs={', '.join(state.outputs) or 'ninguno'})"
        for state in states
    ) or "- Sin estados definidos."


def unique_outputs(states: Sequence[BehaviorState]) -> list[str]:
    return sorted({output for state in states for output in state.outputs if output})


def build_role_assembled_context(
    snapshot: SessionSnapshot,
    *,
    memory_policy: MemoryPolicyV1,
    role: str,
) -> RoleAssembledContext:
    from app.services.llm_runtime.codex_cli.context_assembler import CodexContextAssembler, CodexContextRequest

    normalized_role = role.strip()
    knowledge_access_backend = "hybrid" if "knowledge.approved_sources" in memory_policy.retrieval_scopes else "inline_context"
    task_kind = _ROLE_MEMORY_TASK_KINDS.get(normalized_role, f"{normalized_role}_runtime")
    assembly = CodexContextAssembler().assemble(
        task_kind=task_kind,
        request=CodexContextRequest(
            role=normalized_role,
            knowledge_access_backend=knowledge_access_backend,
            session_snapshot=snapshot,
            memory_policy=memory_policy,
        ),
    )
    required_source_keys = [item.key for item in assembly.required_sources]
    candidate_source_keys = [item.key for item in assembly.candidate_sources]
    source_digests = [
        f"{item.key}: {item.summary}"
        for item in assembly.used_sources[:4]
    ]
    prompt_digest_lines = [
        "Assembled context:",
        f"- knowledge_access_backend: {knowledge_access_backend}",
        f"- required staged sources: {', '.join(required_source_keys) or 'none'}",
        f"- candidate staged sources: {', '.join(candidate_source_keys) or 'none'}",
        f"- staged source counts: required={len(required_source_keys)}, candidate={len(candidate_source_keys)}",
        (
            "- source digest: "
            + (" | ".join(source_digests) if source_digests else "sin fuentes compactadas")
        ),
    ]
    context_sources = dedupe_preserve_order(
        [
            *_ROLE_CONTRACT_CONTEXT_SOURCES.get(normalized_role, []),
            *required_source_keys,
            *candidate_source_keys,
        ]
    )
    return RoleAssembledContext(
        role=normalized_role,
        task_kind=task_kind,
        knowledge_access_backend=knowledge_access_backend,
        context_sources=context_sources,
        required_source_keys=required_source_keys,
        candidate_source_keys=candidate_source_keys,
        source_digests=source_digests,
        prompt_digest_lines=prompt_digest_lines,
    )


def default_states() -> list[BehaviorState]:
    return [
        BehaviorState(
            name="capture_context",
            actor="builder",
            objective="Consolidar contexto aprobado, restricciones y evidencia minima antes de actuar.",
            outputs=["context_snapshot"],
            fallback="Escalar a revision humana si el contexto aprobado no alcanza.",
            requires_approval=False,
        ),
        BehaviorState(
            name="produce_artifact",
            actor="builder",
            objective="Generar o actualizar el artefacto gobernado que sigue en el workflow.",
            outputs=["artifact_update"],
            fallback="Responder needs-resolution si faltan contratos o approvals.",
            requires_approval=False,
        ),
        BehaviorState(
            name="handoff_or_close",
            actor="evaluator",
            objective="Confirmar readiness, remediation minima o handoff controlado.",
            outputs=["readiness_decision"],
            fallback="Mantener blocked hasta nueva evidencia.",
            requires_approval=False,
        ),
    ]


def workflow_states_from_snapshot(snapshot: SessionSnapshot) -> list[BehaviorState]:
    blueprint = snapshot.blueprint
    workflow = blueprint.delivery_package.workflow_profile if blueprint is not None else None
    if workflow is None or not workflow.steps:
        return default_states()
    return [
        BehaviorState(
            name=step.name or f"step_{index + 1}",
            actor=step.actor or "builder",
            objective=step.objective or "Completar el paso definido.",
            outputs=normalized_items(step.outputs),
            fallback=step.fallback,
            requires_approval=step.requires_approval,
        )
        for index, step in enumerate(workflow.steps)
    ]


def _count_side_effect_tools(tool_contracts: Sequence[ToolContractV1]) -> int:
    return sum(1 for tool in tool_contracts if tool.side_effects)


def _benchmark_delta(direction: str, baseline: float, projected: float) -> float:
    if direction == "higher_is_better":
        return round(projected - baseline, 2)
    return round(baseline - projected, 2)


def _benchmark_metric(
    *,
    metric_key: str,
    label: str,
    direction: str,
    unit: str,
    baseline: float,
    projected: float,
    rationale: str,
) -> MultiAgentBenchmarkMetricV1:
    return MultiAgentBenchmarkMetricV1(
        metric_key=metric_key,
        label=label,
        direction=direction,  # type: ignore[arg-type]
        unit=unit,
        baseline_single_agent=round(baseline, 2),
        projected_multi_agent=round(projected, 2),
        improvement_delta=_benchmark_delta(direction, baseline, projected),
        rationale=rationale,
    )


def _multi_agent_architecture_selected(context: "Stage4CompilerContext") -> bool:
    return context.selected_architecture in (_SUPPORTED_MULTI_AGENT_ARCHITECTURES | _PLANNED_MULTI_AGENT_ARCHITECTURES)


def _build_multi_agent_topology(context: "Stage4CompilerContext", behavior: "BehaviorCompilation") -> MultiAgentTopologyV1 | None:
    architecture = context.selected_architecture
    if architecture not in (_SUPPORTED_MULTI_AGENT_ARCHITECTURES | _PLANNED_MULTI_AGENT_ARCHITECTURES):
        return None

    tool_names = [tool.name for tool in context.tool_contracts]
    side_effect_tools = [tool.name for tool in context.tool_contracts if tool.side_effects]
    workflow_state_count = len(context.workflow_states)
    tool_count = len(context.tool_contracts)
    actor_count = len({state.actor for state in context.workflow_states if state.actor})
    acceptance_case_count = len(context.evaluation_cases)
    pending_approvals = len(context.pending_approvals)
    side_effect_count = len(side_effect_tools)
    support_state = "supported" if architecture in _SUPPORTED_MULTI_AGENT_ARCHITECTURES else "planned_only"

    limitation_signals = [
        f"La arquitectura declarada `{architecture}` exige ownership, handoffs y permisos separados por rol.",
    ]
    if workflow_state_count >= 3:
        limitation_signals.append(
            f"El workflow visible ya tiene {workflow_state_count} estados y concentra demasiada coordinacion en un solo agente."
        )
    if tool_count >= 2:
        limitation_signals.append(
            f"El blueprint ya combina {tool_count} tools versionadas y pide especializacion para validarlas sin mezclar contextos."
        )
    if actor_count >= 3:
        limitation_signals.append(
            f"Existen {actor_count} actores logicos en el workflow y un supervisor reduce cambios de contexto no trazados."
        )
    if pending_approvals > 0 or side_effect_count > 0:
        limitation_signals.append(
            "Hay approvals o side effects que conviene aislar para reducir blast radius por especialista."
        )
    if acceptance_case_count >= 2:
        limitation_signals.append(
            "La calidad se beneficia de una revision cruzada entre especialistas antes del cierre final."
        )

    measurable_limitation = (
        f"El baseline single-agent concentra {workflow_state_count} estados, {tool_count} tools, "
        f"{pending_approvals} approvals pendientes y {actor_count} actores logicos bajo una sola unidad de control."
    )
    complexity_score = max(workflow_state_count, 1) + max(tool_count, 1) + actor_count + pending_approvals + side_effect_count
    go_decision = "go" if support_state == "supported" and len(limitation_signals) >= 2 else "hold"
    baseline_latency = float(complexity_score * 12)
    projected_latency = baseline_latency - 18 if go_decision == "go" else baseline_latency + 8
    baseline_quality = float(min(95, 54 + acceptance_case_count * 4 + tool_count * 6 + side_effect_count * 8))
    projected_quality = float(min(99, baseline_quality + (14 if go_decision == "go" else 4)))
    baseline_blast_radius = float(max(1, tool_count + side_effect_count + pending_approvals))
    projected_blast_radius = float(1 if go_decision == "go" else max(1, baseline_blast_radius - 1))
    baseline_coordination_cost = float(complexity_score * 7)
    projected_coordination_cost = float(baseline_coordination_cost + (12 if go_decision == "go" else 6))
    benchmark = MultiAgentBenchmarkV1(
        go_decision=go_decision,
        explicit_go_reason=(
            "La topologia supervisor-especialistas quedo aprobada y supera el baseline single-agent en calidad y aislamiento."
            if go_decision == "go"
            else "La topologia declarada aun no justifica o no soporta una activacion MAS completa."
        ),
        measurable_single_agent_limitation=measurable_limitation,
        limitation_signals=limitation_signals,
        metrics=[
            _benchmark_metric(
                metric_key="quality_coverage",
                label="Cobertura de calidad",
                direction="higher_is_better",
                unit="score",
                baseline=baseline_quality,
                projected=projected_quality,
                rationale="La revision cruzada mejora cobertura de hallazgos antes del cierre.",
            ),
            _benchmark_metric(
                metric_key="failure_blast_radius",
                label="Blast radius de falla",
                direction="lower_is_better",
                unit="relative_score",
                baseline=baseline_blast_radius,
                projected=projected_blast_radius,
                rationale="El aislamiento por especialista reduce el impacto de una falla puntual sobre el estado compartido.",
            ),
            _benchmark_metric(
                metric_key="latency_budget",
                label="Latencia operativa",
                direction="lower_is_better",
                unit="seconds",
                baseline=baseline_latency,
                projected=projected_latency,
                rationale="El supervisor evita retries globales y concentra el merge solo cuando existen findings utiles.",
            ),
            _benchmark_metric(
                metric_key="coordination_cost",
                label="Costo de coordinacion",
                direction="lower_is_better",
                unit="relative_score",
                baseline=baseline_coordination_cost,
                projected=projected_coordination_cost,
                rationale="La coordinacion multiagente agrega overhead y debe mantenerse dentro del presupuesto declarado.",
            ),
        ],
        success_gate=(
            "La topologia multiagente debe mejorar calidad y reducir blast radius sin exceder 25% de overhead de coordinacion."
        ),
        latency_budget="Maximo dos handoffs seriales por ciclo y cierre bajo SLA del supervisor.",
        cost_budget="Hasta tres especialistas activos por corrida y sin retries ciegos sobre side effects.",
    )

    if architecture == "router_parallel":
        agent_contracts = [
            MultiAgentRoleContractV1(
                agent_key="router",
                role="router",
                purpose="Clasificar el trabajo y derivarlo a una rama especializada sin ejecutar side effects.",
                runtime_mode="planned_only",
                permissions=MultiAgentPermissionBoundaryV1(
                    allowed_tools=[],
                    required_approvals=[],
                    side_effect_policy="El router no ejecuta side effects.",
                    escalation_policy="Escalar a supervisor si dos ramas compiten por el mismo estado.",
                ),
                input_contracts=["behavior-spec.v1", "tool-contract.v1"],
                output_contracts=["route_plan"],
                success_signals=["Selecciona solo ramas justificadas por el contrato."],
                failure_mode="Ruta ambigua o conflicto entre ramas.",
                retry_strategy="No reintentar en paralelo sin deduplicacion explicita.",
                timeout_policy="Timeout corto para clasificacion inicial.",
                isolation_boundary="No modifica el estado compartido; solo emite routing intent.",
            ),
            MultiAgentRoleContractV1(
                agent_key="retrieval_lane",
                role="specialist",
                purpose="Recuperar evidencia autorizada para la rama de knowledge o retrieval.",
                runtime_mode="planned_only",
                permissions=MultiAgentPermissionBoundaryV1(
                    allowed_tools=[],
                    required_approvals=[],
                    side_effect_policy="Solo lectura con evidencia citada.",
                    escalation_policy="Escalar si no existe evidencia suficiente.",
                ),
                input_contracts=["knowledge-contract.v1", "memory-policy.v1"],
                output_contracts=["retrieval_findings"],
                success_signals=["Entrega evidencia citada o ausencia explicita de evidencia."],
                failure_mode="Respuesta sin grounding o falta de fuentes aprobadas.",
                retry_strategy="Reintentar solo con filtros diferentes y sin repetir la misma fuente.",
                timeout_policy="Timeout medio para consultas paralelas.",
                isolation_boundary="No escribe estado fuera del buffer de findings.",
            ),
            MultiAgentRoleContractV1(
                agent_key="tool_lane",
                role="specialist",
                purpose="Validar el contrato de una tool y devolver findings sin mezclarlo con retrieval.",
                runtime_mode="planned_only",
                permissions=MultiAgentPermissionBoundaryV1(
                    allowed_tools=tool_names,
                    required_approvals=context.pending_approvals,
                    side_effect_policy="Solo inspeccion contractual mientras la topologia siga planned_only.",
                    escalation_policy="Escalar si una tool requiere approval o compensacion no declarada.",
                ),
                input_contracts=["tool-contract.v1", "behavior-spec.v1"],
                output_contracts=["tool_findings"],
                success_signals=["Cada tool queda evaluada con permisos, retries y side effects."],
                failure_mode="Tool sin contrato suficiente o con side effects no aislados.",
                retry_strategy="No ejecutar la misma tool dos veces sin idempotencia declarada.",
                timeout_policy="Timeout medio por rama.",
                isolation_boundary="No dispara side effects; solo registra findings estructurados.",
            ),
            MultiAgentRoleContractV1(
                agent_key="aggregator",
                role="aggregator",
                purpose="Fusionar findings paralelos y preparar el cierre o remediation.",
                runtime_mode="planned_only",
                permissions=MultiAgentPermissionBoundaryV1(
                    allowed_tools=[],
                    required_approvals=[],
                    side_effect_policy="Sin side effects directos.",
                    escalation_policy="Escalar si las ramas se contradicen o faltan pruebas.",
                ),
                input_contracts=["retrieval_findings", "tool_findings"],
                output_contracts=["parallel_merge_report"],
                success_signals=["La fusion final identifica contradicciones y propone cierre seguro."],
                failure_mode="Merge ambiguo o ramas inconsistentes.",
                retry_strategy="Reintentar solo el merge cuando existan findings nuevos.",
                timeout_policy="Timeout corto al consolidar resultados.",
                isolation_boundary="Solo escribe el reporte de merge final.",
            ),
        ]
        message_contracts = [
            MultiAgentMessageContractV1(
                message_key="router_to_retrieval_lane",
                from_agent="router",
                to_agent="retrieval_lane",
                purpose="Delegar consultas de conocimiento con alcance acotado.",
                payload_schema={"type": "object", "properties": {"intent": {"type": "string"}, "filters": {"type": "array"}}},
                required_fields=["intent"],
                idempotency_strategy="route_id + branch_key",
                timeout_policy="Cancelar la rama si no responde dentro del SLA paralelo.",
                retry_strategy="Un retry maximo por rama y sin duplicar queries.",
                failure_behavior="Cerrar la rama y notificar al aggregator sin corromper otras ramas.",
            ),
            MultiAgentMessageContractV1(
                message_key="router_to_tool_lane",
                from_agent="router",
                to_agent="tool_lane",
                purpose="Delegar validaciones de tools sobre una rama independiente.",
                payload_schema={"type": "object", "properties": {"tool_names": {"type": "array"}, "risk_focus": {"type": "string"}}},
                required_fields=["tool_names"],
                idempotency_strategy="route_id + branch_key",
                timeout_policy="Cancelar la rama si no responde dentro del SLA paralelo.",
                retry_strategy="Un retry maximo por rama y sin repetir tools no idempotentes.",
                failure_behavior="Mantener findings parciales y marcar la rama como aislada.",
            ),
            MultiAgentMessageContractV1(
                message_key="branches_to_aggregator",
                from_agent="retrieval_lane",
                to_agent="aggregator",
                purpose="Entregar findings paralelos para consolidacion.",
                payload_schema={"type": "object", "properties": {"finding_key": {"type": "string"}, "summary": {"type": "string"}}},
                required_fields=["finding_key", "summary"],
                idempotency_strategy="route_id + finding_key",
                timeout_policy="Ignorar mensajes duplicados fuera de ventana activa.",
                retry_strategy="Reintentar solo si el ack del aggregator no existe.",
                failure_behavior="Mantener el buffer de findings y detener el merge final.",
            ),
        ]
        handoff_contracts = [
            MultiAgentHandoffContractV1(
                handoff_key="router_parallel_merge",
                from_agent="aggregator",
                to_agent="supervisor",
                trigger="Todas las ramas paralelas cerraron o quedaron aisladas.",
                ownership_transfer="El supervisor retoma la decision final solo despues del merge.",
                required_artifacts=["parallel_merge_report"],
                success_criteria=["No hay conflictos sin explicar entre ramas paralelas."],
                failure_behavior="Mantener planned_only y pedir remediation del owner.",
                audit_trail=["route_plan", "branch_statuses", "parallel_merge_report"],
            )
        ]
        shared_state_contracts = [
            MultiAgentSharedStateContractV1(
                state_key="parallel_branch_board",
                purpose="Registrar el estado de cada rama paralela y sus findings.",
                owner_agent="router",
                readers=["router", "retrieval_lane", "tool_lane", "aggregator"],
                writers=["router", "retrieval_lane", "tool_lane"],
                payload_schema={"type": "object", "properties": {"branches": {"type": "array"}}},
                update_policy="Cada rama solo puede escribir su propio slot y el router marca aperturas o cierres.",
                consistency_policy="Deduplicacion por route_id y branch_key antes de merge final.",
                rollback_strategy="Descartar solo la rama fallida sin borrar findings de otras ramas.",
            )
        ]
        execution_budget = MultiAgentExecutionBudgetV1(
            latency_budget_ms=int(max(45000, baseline_latency * 1000)),
            max_parallel_agents=2,
            max_retries_per_handoff=1,
            max_tool_calls_per_agent=1 if tool_names else 0,
            cost_budget="Overhead maximo del 20% mientras la topologia siga planned_only.",
        )
        failure_isolation_rules = [
            "Una rama paralela fallida no invalida automaticamente las otras ramas.",
            "Toda escritura compartida se deduplica por route_id antes del merge.",
            "El router no puede ejecutar side effects ni aprobar tools por cuenta propia.",
        ]
        return MultiAgentTopologyV1(
            declared_pattern=architecture,
            runtime_pattern="router_parallel_planned",
            support_state=support_state,
            activation_mode="contract_preview_only",
            benchmark=benchmark,
            agent_contracts=agent_contracts,
            message_contracts=message_contracts,
            handoff_contracts=handoff_contracts,
            shared_state_contracts=shared_state_contracts,
            execution_budget=execution_budget,
            failure_isolation_rules=failure_isolation_rules,
        )

    agent_contracts = [
        MultiAgentRoleContractV1(
            agent_key="supervisor",
            role="supervisor",
            purpose="Asignar especialistas, fusionar findings y decidir cierre o remediation.",
            runtime_mode="supported",
            permissions=MultiAgentPermissionBoundaryV1(
                allowed_tools=[],
                required_approvals=context.pending_approvals,
                side_effect_policy="El supervisor no ejecuta side effects directos; solo autoriza handoffs y merge final.",
                escalation_policy="Escalar a humano si dos especialistas contradicen un contrato bloqueante.",
            ),
            input_contracts=["behavior-spec.v1", "llm-policy.v1", "evaluation-pack.v1"],
            output_contracts=["orchestration_plan", "final_decision"],
            success_signals=["Despacha solo especialistas necesarios y conserva trazabilidad de handoffs."],
            failure_mode="Merge ambiguo o falta de ownership claro para cerrar.",
            retry_strategy="Reintentar solo especialistas idempotentes y nunca todo el flujo completo a ciegas.",
            timeout_policy="SLA corto para despachar y SLA medio para consolidar findings.",
            isolation_boundary="No escribe findings tecnicos; solo resume y decide con contratos versionados.",
        ),
        MultiAgentRoleContractV1(
            agent_key="evaluation_specialist",
            role="specialist",
            purpose="Revisar readiness, acceptance cases y gaps antes del cierre final.",
            runtime_mode="supported",
            permissions=MultiAgentPermissionBoundaryV1(
                allowed_tools=[],
                required_approvals=[],
                side_effect_policy="Solo lectura sobre evaluation-pack y prompt-pack.",
                escalation_policy="Escalar si los casos de aceptacion no cubren el cierre propuesto.",
            ),
            input_contracts=["evaluation-pack.v1", "prompt-pack.v1"],
            output_contracts=["evaluation_findings"],
            success_signals=["Expone gaps, score y recomendacion de readiness sin tocar estado externo."],
            failure_mode="Readiness ambigua o dataset insuficiente.",
            retry_strategy="Un retry maximo si aparece evidencia nueva del dataset o la rubrica.",
            timeout_policy="Timeout medio controlado por el supervisor.",
            isolation_boundary="Solo puede escribir findings bajo su propio namespace.",
        ),
        MultiAgentRoleContractV1(
            agent_key="risk_specialist",
            role="specialist",
            purpose="Validar tools, side effects, approvals y compensaciones antes de permitir promotion.",
            runtime_mode="supported",
            permissions=MultiAgentPermissionBoundaryV1(
                allowed_tools=side_effect_tools or tool_names,
                required_approvals=context.pending_approvals,
                side_effect_policy="Solo inspeccion contractual; no puede ejecutar side effects sobre sistemas externos.",
                escalation_policy="Escalar si una tool carece de approval, rollback o timeout contractual.",
            ),
            input_contracts=["tool-contract.v1", "behavior-spec.v1"],
            output_contracts=["risk_findings"],
            success_signals=["Cada tool queda clasificada por riesgo, approval y compensacion."],
            failure_mode="Tool sin contrato, timeout o compensacion declarada.",
            retry_strategy="Reintentar solo verificaciones de lectura y nunca side effects reales.",
            timeout_policy="Timeout corto por analisis de tool.",
            isolation_boundary="No actualiza tools ni permisos; solo produce findings versionados.",
        ),
        MultiAgentRoleContractV1(
            agent_key="artifact_specialist",
            role="specialist",
            purpose="Revisar coherencia de artefactos, prompts y handoffs tecnicos antes del export.",
            runtime_mode="supported",
            permissions=MultiAgentPermissionBoundaryV1(
                allowed_tools=[],
                required_approvals=[],
                side_effect_policy="Solo lectura sobre artefactos canonicamente exportados.",
                escalation_policy="Escalar si faltan artefactos o el prompt pack no cubre el flujo completo.",
            ),
            input_contracts=["construction-pack.v1", "prompt-pack.v1"],
            output_contracts=["artifact_findings"],
            success_signals=["Confirma presencia y coherencia de artefactos, prompts y handoffs."],
            failure_mode="Prompts faltantes o paquete inconsistente.",
            retry_strategy="Reintentar solo despues de regenerar el paquete canonico.",
            timeout_policy="Timeout corto de revision documental.",
            isolation_boundary="Solo escribe findings de consistencia y nunca modifica el paquete fuente.",
        ),
    ]
    message_contracts = [
        MultiAgentMessageContractV1(
            message_key="supervisor_to_evaluation_specialist",
            from_agent="supervisor",
            to_agent="evaluation_specialist",
            purpose="Delegar la revision de readiness y acceptance cases.",
            payload_schema={"type": "object", "properties": {"focus": {"type": "string"}, "acceptance_cases": {"type": "array"}}},
            required_fields=["focus"],
            idempotency_strategy="run_id + agent_key + focus",
            timeout_policy="Cancelar el intento si no responde dentro del SLA del especialista.",
            retry_strategy="Un retry maximo si no hay side effects ni cambios de dataset en curso.",
            failure_behavior="Marcar la rama como aislada y devolver el control al supervisor.",
        ),
        MultiAgentMessageContractV1(
            message_key="supervisor_to_risk_specialist",
            from_agent="supervisor",
            to_agent="risk_specialist",
            purpose="Delegar la revision de riesgo, approvals y compensaciones.",
            payload_schema={"type": "object", "properties": {"tool_scope": {"type": "array"}, "approval_scope": {"type": "array"}}},
            required_fields=["tool_scope"],
            idempotency_strategy="run_id + agent_key + tool_scope_hash",
            timeout_policy="Cancelar el intento si una tool no puede analizarse dentro del SLA contractual.",
            retry_strategy="No repetir herramientas no idempotentes sin evidencia nueva.",
            failure_behavior="Persistir findings parciales y bloquear promotion hasta nueva decision.",
        ),
        MultiAgentMessageContractV1(
            message_key="supervisor_to_artifact_specialist",
            from_agent="supervisor",
            to_agent="artifact_specialist",
            purpose="Delegar la revision del paquete, prompts y handoffs listos para consumo tecnico.",
            payload_schema={"type": "object", "properties": {"artifact_scope": {"type": "array"}, "prompt_scope": {"type": "array"}}},
            required_fields=["artifact_scope"],
            idempotency_strategy="run_id + agent_key + artifact_scope_hash",
            timeout_policy="Cancelar el intento si el paquete no puede inspeccionarse completo en el SLA previsto.",
            retry_strategy="Un retry maximo tras regenerar el paquete si corresponde.",
            failure_behavior="Devolver findings sin afectar otras ramas de especialista.",
        ),
        MultiAgentMessageContractV1(
            message_key="specialists_to_supervisor",
            from_agent="evaluation_specialist",
            to_agent="supervisor",
            purpose="Entregar findings especializados para consolidacion final.",
            payload_schema={"type": "object", "properties": {"finding_key": {"type": "string"}, "severity": {"type": "string"}, "summary": {"type": "string"}}},
            required_fields=["finding_key", "severity", "summary"],
            idempotency_strategy="run_id + finding_key",
            timeout_policy="Ignorar duplicados fuera de la ventana activa de merge.",
            retry_strategy="Reintentar solo si falta ack del supervisor.",
            failure_behavior="Mantener la finding board intacta y cerrar el handoff como aislado.",
        ),
    ]
    handoff_contracts = [
        MultiAgentHandoffContractV1(
            handoff_key="supervisor_to_evaluation_review",
            from_agent="supervisor",
            to_agent="evaluation_specialist",
            trigger="El supervisor detecta necesidad de validar readiness antes del cierre.",
            ownership_transfer="La titularidad de la revision pasa temporalmente al especialista hasta devolver findings.",
            required_artifacts=["evaluation-pack.v1", "prompt-pack.v1"],
            success_criteria=["Se devuelve score o gap claro para el merge final."],
            failure_behavior="Aislar la rama y pedir remediation del owner si la evaluacion queda inconclusa.",
            audit_trail=["orchestration_plan", "evaluation_findings"],
        ),
        MultiAgentHandoffContractV1(
            handoff_key="supervisor_to_risk_review",
            from_agent="supervisor",
            to_agent="risk_specialist",
            trigger="Existe al menos una tool con side effects o approval pendiente.",
            ownership_transfer="La revision de riesgo queda a cargo del especialista hasta devolver findings auditables.",
            required_artifacts=["tool-contract.v1", "behavior-spec.v1"],
            success_criteria=["Cada tool de riesgo queda clasificada con approval, timeout y rollback."],
            failure_behavior="Bloquear promotion si falta contrato o compensacion.",
            audit_trail=["orchestration_plan", "risk_findings"],
        ),
        MultiAgentHandoffContractV1(
            handoff_key="supervisor_to_artifact_review",
            from_agent="supervisor",
            to_agent="artifact_specialist",
            trigger="El paquete necesita validacion final de artefactos y prompts.",
            ownership_transfer="La revision documental pasa al especialista hasta devolver findings o readiness.",
            required_artifacts=["construction-pack.v1", "prompt-pack.v1"],
            success_criteria=["No faltan prompts ni artefactos bloqueantes para el handoff tecnico."],
            failure_behavior="Marcar el paquete como needs_review hasta regenerar o completar artefactos.",
            audit_trail=["orchestration_plan", "artifact_findings"],
        ),
        MultiAgentHandoffContractV1(
            handoff_key="specialist_merge_to_supervisor",
            from_agent="artifact_specialist",
            to_agent="supervisor",
            trigger="Todos los especialistas cerraron o quedaron aislados.",
            ownership_transfer="El supervisor recupera el ownership para emitir la decision final.",
            required_artifacts=["evaluation_findings", "risk_findings", "artifact_findings"],
            success_criteria=["El merge final explica conflictos, readiness y decision de cierre."],
            failure_behavior="Mantener blocked hasta aclarar hallazgos contradictorios.",
            audit_trail=["finding_board", "final_decision"],
        ),
    ]
    shared_state_contracts = [
        MultiAgentSharedStateContractV1(
            state_key="execution_brief",
            purpose="Contexto compartido minimo y estable que el supervisor entrega a cada especialista.",
            owner_agent="supervisor",
            readers=["supervisor", "evaluation_specialist", "risk_specialist", "artifact_specialist"],
            writers=["supervisor"],
            payload_schema={"type": "object", "properties": {"goal": {"type": "string"}, "workflow_states": {"type": "array"}}},
            update_policy="Solo el supervisor actualiza el brief entre handoffs.",
            consistency_policy="Cada especialista lee la misma version del brief por run_id.",
            rollback_strategy="Volver a la ultima version aprobada del brief si un merge falla.",
        ),
        MultiAgentSharedStateContractV1(
            state_key="finding_board",
            purpose="Tablero de findings especializados aislado por agente.",
            owner_agent="supervisor",
            readers=["supervisor", "evaluation_specialist", "risk_specialist", "artifact_specialist"],
            writers=["evaluation_specialist", "risk_specialist", "artifact_specialist"],
            payload_schema={"type": "object", "properties": {"findings": {"type": "array"}}},
            update_policy="Cada especialista escribe solo bajo su namespace y el supervisor consolida.",
            consistency_policy="Deduplicar findings por finding_key antes del merge final.",
            rollback_strategy="Eliminar solo la finding corrupta sin tocar otras ramas.",
        ),
        MultiAgentSharedStateContractV1(
            state_key="final_decision_record",
            purpose="Registrar la decision consolidada del supervisor y el motivo del cierre.",
            owner_agent="supervisor",
            readers=["supervisor", "evaluation_specialist", "risk_specialist", "artifact_specialist"],
            writers=["supervisor"],
            payload_schema={"type": "object", "properties": {"decision": {"type": "string"}, "summary": {"type": "string"}}},
            update_policy="Solo el supervisor escribe el cierre despues del merge.",
            consistency_policy="Una sola decision final por run_id.",
            rollback_strategy="Mantener la ultima decision valida si la nueva fusion falla validacion.",
        ),
    ]
    execution_budget = MultiAgentExecutionBudgetV1(
        latency_budget_ms=int(max(45000, baseline_latency * 1000)),
        max_parallel_agents=3,
        max_retries_per_handoff=1,
        max_tool_calls_per_agent=1 if tool_names else 0,
        cost_budget="Overhead maximo de 25% frente al baseline single-agent y sin duplicar side effects.",
    )
    failure_isolation_rules = [
        "Una falla de especialista no puede sobrescribir el execution_brief compartido.",
        "Cada especialista escribe findings solo en su namespace y el supervisor hace el merge final.",
        "Los side effects permanecen bloqueados hasta que riesgo y supervisor cierren el handoff correspondiente.",
    ]
    return MultiAgentTopologyV1(
        declared_pattern=architecture,
        runtime_pattern="supervisor_specialist_runtime",
        support_state=support_state,
        activation_mode="feature_flag_controlled_runtime",
        benchmark=benchmark,
        agent_contracts=agent_contracts,
        message_contracts=message_contracts,
        handoff_contracts=handoff_contracts,
        shared_state_contracts=shared_state_contracts,
        execution_budget=execution_budget,
        failure_isolation_rules=failure_isolation_rules,
    )


@dataclass(frozen=True)
class Stage4CompilerContext:
    snapshot: SessionSnapshot
    generated_at: Any
    tool_contracts: list[ToolContractV1]
    memory_policy: MemoryPolicyV1
    knowledge_contract: KnowledgeContractV1
    assembled_role_contexts: dict[str, RoleAssembledContext]
    success_criteria: list[SuccessCriterion]
    selected_architecture: str
    selected_reasoning_pattern: str
    selected_workflow_template_key: str
    workflow_states: list[BehaviorState]
    pending_approvals: list[str]
    evaluation_cases: list[EvaluationCaseV1]
    facts: list[HeuristicDecisionFact]
    prompt_variables: list[PromptVariable]
    stop_conditions: list[str]
    blueprint_guardrails: list[str]
    candidate_catalog: list[PatternCatalogEntry]
    decision_summary: str
    decision_trace: list[Any]
    goal: str

    @property
    def has_tools(self) -> bool:
        return bool(self.tool_contracts)

    @property
    def has_memory(self) -> bool:
        return self.memory_policy.strategy != "no_memory" or bool(self.memory_policy.storage_layers)

    @property
    def has_knowledge(self) -> bool:
        return self.knowledge_contract.enabled

    @property
    def evaluation_gap_count(self) -> int:
        evaluation = self.snapshot.evaluation
        return len(normalized_items(evaluation.gaps if evaluation is not None else []))


@dataclass(frozen=True)
class PromptPlan:
    roles: list[str]
    rationale_by_role: dict[str, str]


@dataclass(frozen=True)
class BehaviorCompilation:
    compiler_key: str
    compiler_label: str
    supported: bool
    support_note: str
    behavior_spec: BehaviorSpecV1
    prompt_hints: dict[str, str]
    llm_reasoning_effort: dict[str, str]
    llm_max_tokens: dict[str, int]
    restrictions: list[str]


@dataclass(frozen=True)
class Stage4CompilationResult:
    context: Stage4CompilerContext
    behavior: BehaviorCompilation
    heuristic_decision: HeuristicDecisionV1
    llm_policy: LLMPolicyV1
    prompt_pack: PromptPackV1
    prompt_plan: PromptPlan

    @property
    def behavior_spec(self) -> BehaviorSpecV1:
        return self.behavior.behavior_spec


class ContextNormalizer:
    def normalize(
        self,
        snapshot: SessionSnapshot,
        *,
        generated_at: Any,
        tool_contracts: Sequence[ToolContractV1],
        memory_policy: MemoryPolicyV1,
        knowledge_contract: KnowledgeContractV1,
        success_criteria: Sequence[SuccessCriterion],
    ) -> Stage4CompilerContext:
        discovery = snapshot.discovery
        canvas = snapshot.canvas
        blueprint = snapshot.blueprint

        goal = coalesce(
            discovery.desired_outcome if discovery is not None else None,
            canvas.user_goal if canvas is not None else None,
            snapshot.session.title,
            fallback="Objetivo no documentado.",
        )
        selected_reasoning_pattern = coalesce(
            blueprint.reasoning_pattern if blueprint is not None else None,
            fallback="Plan-and-Execute",
        )
        selected_architecture = coalesce(
            blueprint.architecture if blueprint is not None else None,
            fallback="single_agent",
        )
        pending_approvals = [approval.gate_key for approval in snapshot.approvals if approval.status == "pending"]
        facts = [
            HeuristicDecisionFact(
                key="case_type",
                value=discovery.case_type if discovery is not None else "unknown",
                source="discovery.case_type",
            ),
            HeuristicDecisionFact(
                key="autonomy_level",
                value=discovery.autonomy_level if discovery is not None else "unknown",
                source="discovery.autonomy_level",
            ),
            HeuristicDecisionFact(
                key="reasoning_pattern",
                value=selected_reasoning_pattern,
                source="blueprint.reasoning_pattern",
            ),
            HeuristicDecisionFact(
                key="architecture",
                value=selected_architecture,
                source="blueprint.architecture",
            ),
            HeuristicDecisionFact(
                key="workflow_template",
                value=snapshot.selected_workflow_template_key or "none",
                source="selected_workflow_template_key",
            ),
            HeuristicDecisionFact(
                key="tool_count",
                value=str(len(tool_contracts)),
                source="tool-contract.v1",
            ),
            HeuristicDecisionFact(
                key="knowledge_enabled",
                value="true" if knowledge_contract.enabled else "false",
                source="knowledge-contract.v1",
            ),
            HeuristicDecisionFact(
                key="memory_strategy",
                value=memory_policy.strategy,
                source="memory-policy.v1",
            ),
            HeuristicDecisionFact(
                key="pending_approvals",
                value=str(len(pending_approvals)),
                source="approvals",
            ),
            HeuristicDecisionFact(
                key="evaluation_gaps",
                value=str(len(normalized_items(snapshot.evaluation.gaps if snapshot.evaluation is not None else []))),
                source="evaluation.gaps",
            ),
        ]
        delivery = blueprint.delivery_package if blueprint is not None else None
        assembled_role_contexts = {
            role: build_role_assembled_context(snapshot, memory_policy=memory_policy, role=role)
            for role in _ROLE_CONTRACT_CONTEXT_SOURCES
        }
        return Stage4CompilerContext(
            snapshot=snapshot,
            generated_at=generated_at,
            tool_contracts=list(tool_contracts),
            memory_policy=memory_policy,
            knowledge_contract=knowledge_contract,
            assembled_role_contexts=assembled_role_contexts,
            success_criteria=list(success_criteria),
            selected_architecture=selected_architecture,
            selected_reasoning_pattern=selected_reasoning_pattern,
            selected_workflow_template_key=snapshot.selected_workflow_template_key or "",
            workflow_states=workflow_states_from_snapshot(snapshot),
            pending_approvals=pending_approvals,
            evaluation_cases=evaluation_cases(snapshot),
            facts=facts,
            prompt_variables=prompt_variables(tool_contracts, success_criteria),
            stop_conditions=prompt_stop_conditions(snapshot),
            blueprint_guardrails=normalized_items(blueprint.guardrails if blueprint is not None else []),
            candidate_catalog=list(delivery.pattern_catalog if delivery is not None else []),
            decision_summary=coalesce(
                delivery.decision_summary if delivery is not None else None,
                fallback="Decision trace derivada del blueprint actual con foco en arquitectura, razonamiento y memoria.",
            ),
            decision_trace=list(delivery.decision_trace if delivery is not None else []),
            goal=goal,
        )


class BehaviorCompiler(ABC):
    key: str
    label: str

    @abstractmethod
    def compile(self, context: Stage4CompilerContext) -> BehaviorCompilation:
        raise NotImplementedError

    def _build_behavior_spec(
        self,
        context: Stage4CompilerContext,
        *,
        execution_pattern: str,
        reasoning_pattern: str,
        checkpoint_policy: str,
        retry_strategy: str,
        compensation_strategy: str,
        approval_pause: str,
        timeout_policy: str,
        states: Sequence[BehaviorState],
        termination_criteria: Sequence[str],
    ) -> BehaviorSpecV1:
        return BehaviorSpecV1(
            **base_metadata(context.snapshot, context.generated_at),
            execution_pattern=execution_pattern,
            reasoning_pattern=reasoning_pattern,
            selected_workflow_template_key=context.selected_workflow_template_key,
            checkpoint_policy=checkpoint_policy,
            retry_strategy=retry_strategy,
            compensation_strategy=compensation_strategy,
            approval_pause=approval_pause,
            timeout_policy=timeout_policy,
            states=list(states),
            termination_criteria=list(termination_criteria),
            outputs=unique_outputs(states),
            required_approvals=list(context.pending_approvals),
            provenance=provenance(
                (
                    "states",
                    ["blueprint.delivery_package.workflow_profile.steps", "selected_workflow_template_key"],
                    "El compilador S4 proyecta el workflow visible en un comportamiento ejecutable y trazable.",
                ),
                (
                    "reasoning_pattern",
                    ["blueprint.reasoning_pattern"],
                    "La politica de razonamiento preserva la seleccion aprobada sin degradar silenciosamente.",
                ),
                (
                    "termination_criteria",
                    ["evaluation.gaps", "approvals"],
                    "Los criterios de cierre amarran termination, approvals y evaluacion del snapshot.",
                ),
            ),
        )

    def _termination_criteria(self, context: Stage4CompilerContext, *extra_items: str) -> list[str]:
        criteria = [
            "Todos los outputs obligatorios del workflow existen y son trazables.",
            "No quedan approvals pendientes para tools con side effects antes del cierre final.",
            "La evaluacion y los acceptance cases no reportan gaps bloqueantes sin remediation.",
        ]
        criteria.extend(item for item in extra_items if item)
        return criteria


class DeterministicWorkflowCompiler(BehaviorCompiler):
    key = "deterministic"
    label = "Deterministic workflow"

    def compile(self, context: Stage4CompilerContext) -> BehaviorCompilation:
        states = list(context.workflow_states)
        states.append(
            BehaviorState(
                name="verify_deterministic_path",
                actor="evaluator",
                objective="Confirmar que el workflow cerro sin ramas, loops no aprobados ni improvisacion.",
                outputs=["deterministic_trace"],
                fallback="Bloquear el cierre y pedir remediation si aparece una rama no autorizada.",
                requires_approval=False,
            )
        )
        behavior_spec = self._build_behavior_spec(
            context,
            execution_pattern="deterministic_workflow",
            reasoning_pattern=context.selected_reasoning_pattern or "deterministic",
            checkpoint_policy="Persistir el snapshot al cerrar cada estado fijo del workflow.",
            retry_strategy="Reintentar solo pasos idempotentes y mantener el orden exacto del workflow.",
            compensation_strategy="Volver al ultimo checkpoint valido y reemitir el estado fijo pendiente.",
            approval_pause="Pausar antes de cualquier side effect o desviacion del camino aprobado.",
            timeout_policy="Escalar a humano si un estado fijo excede el SLA sin salida trazable.",
            states=states,
            termination_criteria=self._termination_criteria(
                context,
                "No se abrieron ramas alternativas fuera del camino determinista compilado.",
            ),
        )
        return BehaviorCompilation(
            compiler_key=self.key,
            compiler_label=self.label,
            supported=True,
            support_note="S4 soporta este modo para workflows lineales, sin bifurcaciones no aprobadas.",
            behavior_spec=behavior_spec,
            prompt_hints={
                "planner": "Convierte el objetivo en un checklist cerrado y fijo. No explores alternativas ni reordenes estados.",
                "executor": "Ejecuta exactamente un estado fijo por turno y no improvises acciones fuera del camino compilado.",
                "evaluator": "Valida cumplimiento exacto de orden, outputs y termination criteria del camino determinista.",
            },
            llm_reasoning_effort={"planner": "low", "executor": "low", "evaluator": "low", "recovery": "low"},
            llm_max_tokens={"planner": 1200, "executor": 1600, "evaluator": 1400, "recovery": 1200},
            restrictions=[
                "No se permiten ramas alternativas sin recompilar el workflow.",
                "Toda desviacion debe terminar en blocked o needs-resolution.",
            ],
        )


class ToolCallingBehaviorCompiler(BehaviorCompiler):
    key = "tool-calling"
    label = "Tool calling"

    def compile(self, context: Stage4CompilerContext) -> BehaviorCompilation:
        needs_approval = any(tool.requires_approval for tool in context.tool_contracts)
        states = [
            BehaviorState(
                name="inspect_request",
                actor="planner",
                objective="Determinar la intencion aprobada y si requiere tool calling.",
                outputs=["action_intent"],
                fallback="Pedir needs-resolution cuando el objetivo no este suficientemente acotado.",
                requires_approval=False,
            ),
            BehaviorState(
                name="select_tool_contract",
                actor="executor",
                objective="Elegir la tool aprobada y validar permisos, schema y side effects antes de usarla.",
                outputs=["selected_tool", "tool_input_draft"],
                fallback="No invocar ninguna tool si el contrato requerido esta incompleto.",
                requires_approval=needs_approval,
            ),
            BehaviorState(
                name="run_tool",
                actor="executor",
                objective="Ejecutar una sola tool aprobada y capturar su recibo trazable.",
                outputs=["tool_receipt", "tool_output"],
                fallback="Escalar a recovery si la tool falla o devuelve un schema invalido.",
                requires_approval=needs_approval,
            ),
            BehaviorState(
                name="validate_tool_result",
                actor="evaluator",
                objective="Validar output, side effects y evidencia devuelta por la tool antes de continuar.",
                outputs=["validated_output"],
                fallback="Bloquear el flujo hasta aclarar el resultado de la tool.",
                requires_approval=False,
            ),
            BehaviorState(
                name="finalize_or_escalate",
                actor="evaluator",
                objective="Cerrar el turno con respuesta, approval gate o remediation minima.",
                outputs=["tool_call_decision"],
                fallback="Mantener blocked si la evidencia sigue siendo insuficiente.",
                requires_approval=False,
            ),
        ]
        behavior_spec = self._build_behavior_spec(
            context,
            execution_pattern="tool_calling_loop",
            reasoning_pattern=context.selected_reasoning_pattern or "tool-calling",
            checkpoint_policy="Persistir input, output y validacion de cada tool call como checkpoint independiente.",
            retry_strategy="Reintentar solo tools idempotentes y nunca repetir side effects sin policy de compensacion.",
            compensation_strategy="Aplicar rollback contractual o escalar a humano si la tool toca estado externo.",
            approval_pause="Pausar antes de cualquier tool con side effects o approval requerido.",
            timeout_policy="Abortar la invocacion y escalar si una tool supera su timeout contractual.",
            states=states,
            termination_criteria=self._termination_criteria(
                context,
                "Cada tool usada queda validada contra schema, permisos y strategy de rollback antes del cierre.",
            ),
        )
        return BehaviorCompilation(
            compiler_key=self.key,
            compiler_label=self.label,
            supported=True,
            support_note="S4 soporta tool calling con contratos, approvals y validacion de schema por invocacion.",
            behavior_spec=behavior_spec,
            prompt_hints={
                "planner": "Define la secuencia minima de tool calls y sus puntos de validacion antes de ejecutar.",
                "executor": "Evalua si hace falta una tool y solo ejecuta una accion permitida por iteracion.",
                "evaluator": "Verifica schema, permisos, retries y resultado observable de cada tool call.",
                "tool_use": "Antes de invocar una tool, comprueba contrato, permisos, retries y compensacion aplicable.",
            },
            llm_reasoning_effort={"planner": "medium", "executor": "medium", "evaluator": "low", "tool_use": "low", "recovery": "low"},
            llm_max_tokens={"planner": 2000, "executor": 2400, "evaluator": 1500, "tool_use": 1400, "recovery": 1200},
            restrictions=[
                "Nunca llamar una tool sin contrato versionado y approval correspondiente.",
                "No reintentar side effects sin estrategia de compensacion declarada.",
            ],
        )


class PlanAndExecuteBehaviorCompiler(BehaviorCompiler):
    key = "plan-and-execute"
    label = "Plan-and-Execute"

    def compile(self, context: Stage4CompilerContext) -> BehaviorCompilation:
        states = [
            BehaviorState(
                name="plan_workflow",
                actor="planner",
                objective="Descomponer el objetivo en una secuencia acotada, trazable y verificable antes de ejecutar.",
                outputs=["approved_plan", "risk_register"],
                fallback="Devolver needs-resolution si faltan inputs estructurales para cerrar el plan.",
                requires_approval=False,
            ),
            *context.workflow_states,
            BehaviorState(
                name="evaluate_readiness",
                actor="evaluator",
                objective="Confirmar cobertura, termination y readiness del plan ejecutado.",
                outputs=["readiness_report"],
                fallback="Marcar blocked y proponer la remediation minima necesaria.",
                requires_approval=False,
            ),
        ]
        behavior_spec = self._build_behavior_spec(
            context,
            execution_pattern=coalesce(
                context.snapshot.blueprint.delivery_package.workflow_profile.execution_pattern
                if context.snapshot.blueprint is not None
                else None,
                fallback="durable_linear_workflow",
            ),
            reasoning_pattern="Plan-and-Execute",
            checkpoint_policy="Persistir plan, ejecucion por estado y evaluacion final como checkpoints separados.",
            retry_strategy="Permitir retries por estado solo despues de revisar el plan y su estado previo.",
            compensation_strategy="Volver al ultimo checkpoint consistente y abrir replan explicito si cambia el contexto.",
            approval_pause="Pausar antes de side effects y antes de continuar si el plan queda invalidado por nueva evidencia.",
            timeout_policy="Escalar a humano si un estado del plan excede su SLA o requiere replan no autorizado.",
            states=states,
            termination_criteria=self._termination_criteria(
                context,
                "Cada paso del plan fue ejecutado o cerrado con replan explicito y trazable.",
            ),
        )
        return BehaviorCompilation(
            compiler_key=self.key,
            compiler_label=self.label,
            supported=True,
            support_note="S4 soporta Plan-and-Execute para workflows visibles, largos o con checkpoints claros.",
            behavior_spec=behavior_spec,
            prompt_hints={
                "planner": "Construye primero el plan completo, con pasos, riesgos, checkpoints y criterios de replan.",
                "executor": "Ejecuta solo el paso actual del plan aprobado y no cambies el orden sin replan explicito.",
                "evaluator": "Compara resultados, plan, outputs y termination criteria antes de cerrar o replanificar.",
            },
            llm_reasoning_effort={"planner": "high", "executor": "medium", "evaluator": "low", "tool_use": "low", "memory": "low", "retrieval": "low", "recovery": "low"},
            llm_max_tokens={"planner": 2600, "executor": 2400, "evaluator": 1600, "tool_use": 1400, "memory": 1200, "retrieval": 1400, "recovery": 1200},
            restrictions=[
                "No reordenar pasos sin abrir replan explicito y trazable.",
                "Si el plan pierde vigencia, el flujo debe detenerse y pedir remediation.",
            ],
        )


class SupervisorSpecialistBehaviorCompiler(BehaviorCompiler):
    key = "supervisor-specialist"
    label = "Supervisor + Specialists"

    def compile(self, context: Stage4CompilerContext) -> BehaviorCompilation:
        states = [
            BehaviorState(
                name="supervisor_intake",
                actor="supervisor",
                objective="Sintetizar el objetivo, decidir si hace falta delegacion y preparar el brief compartido.",
                outputs=["orchestration_plan", "execution_brief"],
                fallback="Escalar a humano si el objetivo o el ownership siguen ambiguos.",
                requires_approval=False,
            ),
            BehaviorState(
                name="dispatch_specialists",
                actor="supervisor",
                objective="Seleccionar los especialistas minimos necesarios y emitir handoffs trazables por rama.",
                outputs=["specialist_handoffs"],
                fallback="Bloquear si la delegacion no puede justificarse con contratos o permisos claros.",
                requires_approval=bool(context.pending_approvals),
            ),
            BehaviorState(
                name="collect_specialist_findings",
                actor="specialist",
                objective="Ejecutar revisiones aisladas por dominio y devolver findings versionados al supervisor.",
                outputs=["evaluation_findings", "risk_findings", "artifact_findings"],
                fallback="Aislar la rama fallida y preservar las findings validas del resto.",
                requires_approval=False,
            ),
            BehaviorState(
                name="merge_supervisor_decision",
                actor="supervisor",
                objective="Consolidar findings, resolver contradicciones y decidir cierre, remediation o retorno controlado.",
                outputs=["final_decision", "merge_report"],
                fallback="Mantener blocked si el merge no puede justificar una decision segura.",
                requires_approval=bool(context.pending_approvals),
            ),
            BehaviorState(
                name="verify_handoff_isolation",
                actor="evaluator",
                objective="Validar que los handoffs fueron trazables, idempotentes y sin corrupcion del estado compartido.",
                outputs=["isolation_report"],
                fallback="Marcar el flujo como failed si un especialista contamino el estado comun.",
                requires_approval=False,
            ),
        ]
        behavior_spec = self._build_behavior_spec(
            context,
            execution_pattern="supervisor_specialist_runtime",
            reasoning_pattern=context.selected_reasoning_pattern or "Plan-and-Execute",
            checkpoint_policy="Persistir brief, handoffs, findings y merge final como checkpoints independientes por agente.",
            retry_strategy="Reintentar solo la rama aislada e idempotente; nunca reiniciar todo el supervisor loop por una falla local.",
            compensation_strategy="Descartar findings corruptas, restaurar el ultimo shared state valido y escalar si existe side effect dudoso.",
            approval_pause="Pausar cuando un handoff requiera approval o un especialista detecte side effects sin rollback claro.",
            timeout_policy="Cerrar la rama vencida, preservar el estado compartido consistente y devolver el control al supervisor.",
            states=states,
            termination_criteria=self._termination_criteria(
                context,
                "Cada handoff deja ownership explicito, artifacts requeridos y merge final auditado.",
                "Una falla de especialista no contamina el estado compartido ni invalida findings ya aprobadas.",
            ),
        )
        return BehaviorCompilation(
            compiler_key=self.key,
            compiler_label=self.label,
            supported=True,
            support_note="S4 soporta supervisor + specialists con handoffs aislados, shared state controlado y merge final trazable.",
            behavior_spec=behavior_spec,
            prompt_hints={
                "planner": "Actua como supervisor. Decide primero si la delegacion es necesaria, minimiza especialistas y deja handoffs auditables.",
                "executor": "Ejecuta solo el paso del supervisor o del especialista activo sin contaminar shared state ajeno.",
                "evaluator": "Valida aislamiento, idempotencia de handoffs, merge final y cumplimiento del benchmark declarado.",
                "recovery": "Aisla la rama fallida, conserva las findings sanas y propone remediation puntual antes de repetir la orquestacion.",
            },
            llm_reasoning_effort={"planner": "high", "executor": "medium", "evaluator": "medium", "recovery": "medium"},
            llm_max_tokens={"planner": 2800, "executor": 2400, "evaluator": 1800, "recovery": 1800},
            restrictions=[
                "El supervisor no puede degradar silenciosamente a single-agent si la topologia declarada exige delegacion.",
                "Cada especialista debe operar con permisos, retry y namespace de findings independientes.",
            ],
        )


class ReActBehaviorCompiler(BehaviorCompiler):
    key = "react"
    label = "ReAct"

    def compile(self, context: Stage4CompilerContext) -> BehaviorCompilation:
        loop_states = [
            BehaviorState(
                name="observe_context",
                actor="planner",
                objective="Leer el estado actual, detectar la mejor siguiente accion y preparar la observacion esperada.",
                outputs=["context_observation"],
                fallback="Pedir needs-resolution si la observacion requerida no esta disponible.",
                requires_approval=False,
            )
        ]
        loop_states.extend(
            BehaviorState(
                name=f"react_{state.name}",
                actor=state.actor or "executor",
                objective=f"Razonar y actuar sobre el estado '{state.name}' con observacion inmediata despues de cada accion.",
                outputs=unique_outputs([state]) + ["observation_log"],
                fallback=state.fallback or "Escalar a recovery si la observacion posterior no permite decidir el siguiente paso.",
                requires_approval=state.requires_approval,
            )
            for state in context.workflow_states
        )
        loop_states.append(
            BehaviorState(
                name="evaluate_progress",
                actor="evaluator",
                objective="Decidir si continuar, cerrar o pedir approval segun la ultima observacion util.",
                outputs=["progress_decision"],
                fallback="Escalar a humano si las observaciones siguen siendo ambiguas o contradictorias.",
                requires_approval=False,
            )
        )
        behavior_spec = self._build_behavior_spec(
            context,
            execution_pattern="react_observe_act_loop",
            reasoning_pattern="ReAct",
            checkpoint_policy="Persistir observacion, accion y artifact despues de cada iteracion util del loop.",
            retry_strategy="Permitir una nueva iteracion solo si la ultima observacion aporta evidencia nueva y recuperable.",
            compensation_strategy="Deshacer side effects segun contrato o pausar hasta nueva decision humana.",
            approval_pause="Pausar el loop cuando una accion requiera approval o cambie el riesgo del caso.",
            timeout_policy="Escalar si el loop supera el numero esperado de iteraciones o deja de producir observaciones nuevas.",
            states=loop_states,
            termination_criteria=self._termination_criteria(
                context,
                "La ultima observacion confirma que no quedan acciones utiles ni approvals abiertos para el objetivo actual.",
            ),
        )
        return BehaviorCompilation(
            compiler_key=self.key,
            compiler_label=self.label,
            supported=True,
            support_note="S4 soporta ReAct para iteraciones cortas, observacion local y uso controlado de tools.",
            behavior_spec=behavior_spec,
            prompt_hints={
                "planner": "No construyas un plan largo. Propone solo la siguiente microsecuencia de 1 a 3 movimientos utiles.",
                "executor": "Alterna razonamiento breve, accion permitida y observacion. Reevalua despues de cada resultado.",
                "evaluator": "Valida si la ultima observacion mejora el estado, exige approval o debe cortar el loop.",
            },
            llm_reasoning_effort={"planner": "low", "executor": "medium", "evaluator": "low", "tool_use": "low", "memory": "low", "retrieval": "low", "recovery": "low"},
            llm_max_tokens={"planner": 1400, "executor": 2600, "evaluator": 1500, "tool_use": 1400, "memory": 1200, "retrieval": 1400, "recovery": 1200},
            restrictions=[
                "No extender el loop si la observacion no agrega evidencia accionable.",
                "Cada iteracion debe producir una observacion trazable antes de la siguiente accion.",
            ],
        )


def _decorate_behavior_with_multi_agent_topology(
    context: Stage4CompilerContext,
    behavior: BehaviorCompilation,
) -> BehaviorCompilation:
    topology = _build_multi_agent_topology(context, behavior)
    if topology is None:
        return behavior
    restrictions = list(behavior.restrictions)
    if topology.support_state != "supported":
        restrictions.append(
            f"La topologia `{topology.declared_pattern}` queda declarada pero el runtime soportado sigue en estado `{topology.support_state}`."
        )
    updated_behavior_spec = behavior.behavior_spec.model_copy(update={"multi_agent_topology": topology})
    return replace(
        behavior,
        behavior_spec=updated_behavior_spec,
        restrictions=restrictions,
    )


class UnsupportedBehaviorCompiler(BehaviorCompiler):
    def __init__(self, unsupported_pattern: str) -> None:
        self.key = f"unsupported::{unsupported_pattern}"
        self.label = unsupported_pattern
        self.unsupported_pattern = unsupported_pattern

    def compile(self, context: Stage4CompilerContext) -> BehaviorCompilation:
        states = [
            BehaviorState(
                name="needs_resolution",
                actor="planner",
                objective=f"Reportar que el patron {self.unsupported_pattern} aun no esta soportado por S4 y pedir remediation.",
                outputs=["compiler_gap_report"],
                fallback="Mantener blocked hasta que el owner seleccione un patron soportado o implemente el compilador faltante.",
                requires_approval=False,
            )
        ]
        behavior_spec = self._build_behavior_spec(
            context,
            execution_pattern=f"unsupported::{self.unsupported_pattern.lower().replace(' ', '_')}",
            reasoning_pattern=self.unsupported_pattern,
            checkpoint_policy="Persistir el diagnostico del gap del compilador y no abrir ejecucion real.",
            retry_strategy="No reintentar automaticamente; esperar decision del owner.",
            compensation_strategy="No aplica porque no se debe ejecutar comportamiento productivo.",
            approval_pause="Pausar de inmediato y solicitar cambio de patron o implementacion del soporte faltante.",
            timeout_policy="Mantener blocked hasta nueva decision estructural.",
            states=states,
            termination_criteria=self._termination_criteria(
                context,
                f"Cerrar solo cuando el owner cambie {self.unsupported_pattern} por un patron soportado o habilite su compilador.",
            ),
        )
        return BehaviorCompilation(
            compiler_key=self.key,
            compiler_label=self.label,
            supported=False,
            support_note=f"S4 no soporta todavia {self.unsupported_pattern}; se debe pedir remediation y nunca degradar silenciosamente a otro patron.",
            behavior_spec=behavior_spec,
            prompt_hints={
                "planner": f"No intentes compilar {self.unsupported_pattern}. Devuelve needs-resolution con el gap y la minima remediation.",
                "executor": "No ejecutes acciones productivas. Devuelve blocked o needs-resolution con evidencia del gap.",
                "evaluator": "Confirma que el comportamiento se bloqueo por soporte incompleto y no por una falla transitoria.",
                "recovery": "Propone cambiar a ReAct o Plan-and-Execute, o implementar el compilador faltante antes de continuar.",
            },
            llm_reasoning_effort={"planner": "low", "executor": "low", "evaluator": "low", "recovery": "low"},
            llm_max_tokens={"planner": 1200, "executor": 1000, "evaluator": 1200, "recovery": 1200},
            restrictions=[
                "Nunca degradar silenciosamente un patron no soportado a otro comportamiento.",
                "No publicar un prompt pack que parezca ejecutable si el patron sigue fuera de soporte.",
            ],
        )


class BehaviorCompilerRegistry:
    def __init__(self, compilers: Sequence[BehaviorCompiler]) -> None:
        self._compilers = {compiler.key: compiler for compiler in compilers}

    def _normalize_key(self, key: str) -> str:
        aliases = {
            "Plan-and-Execute": "plan-and-execute",
            "ReAct": "react",
            "deterministic_workflow": "deterministic",
            "tool_calling": "tool-calling",
            "tool-calling": "tool-calling",
        }
        return aliases.get(key, key)

    def compile_named(self, key: str, context: Stage4CompilerContext) -> BehaviorCompilation:
        normalized = self._normalize_key(key)
        if normalized in self._compilers:
            return self._compilers[normalized].compile(context)
        if key in _PLANNED_REASONING_PATTERNS:
            return UnsupportedBehaviorCompiler(key).compile(context)
        raise KeyError(f"Unsupported compiler key: {key}")

    def select(self, context: Stage4CompilerContext) -> BehaviorCompilation:
        if context.selected_architecture == "supervisor_with_subagents":
            return self.compile_named("supervisor-specialist", context)
        pattern = context.selected_reasoning_pattern
        if pattern in _SUPPORTED_REASONING_PATTERNS:
            return self.compile_named(pattern, context)
        if pattern in _PLANNED_REASONING_PATTERNS:
            return UnsupportedBehaviorCompiler(pattern).compile(context)
        if context.has_tools:
            return self.compile_named("tool-calling", context)
        return self.compile_named("deterministic", context)


DEFAULT_BEHAVIOR_COMPILER_REGISTRY = BehaviorCompilerRegistry(
    (
        DeterministicWorkflowCompiler(),
        ToolCallingBehaviorCompiler(),
        PlanAndExecuteBehaviorCompiler(),
        SupervisorSpecialistBehaviorCompiler(),
        ReActBehaviorCompiler(),
    )
)


class PromptPlanner:
    def plan(self, context: Stage4CompilerContext, behavior: BehaviorCompilation) -> PromptPlan:
        roles = ["system", "planner", "executor", "evaluator"]
        rationale = {
            "system": "Siempre fija el marco contractual del compilador.",
            "planner": "Siempre define la unidad minima de plan soportada por el patron compilado.",
            "executor": "Siempre ejecuta el siguiente paso permitido por el behavior spec.",
            "evaluator": "Siempre valida readiness, termination y remediation.",
        }
        if context.has_tools:
            roles.append("tool_use")
            rationale["tool_use"] = "Se requiere porque existen tools aprobadas y contratos versionados."
        if context.has_memory:
            roles.append("memory")
            rationale["memory"] = "Se requiere porque la estrategia de memoria permite lectura o escritura contextual."
        if context.has_knowledge:
            roles.append("retrieval")
            rationale["retrieval"] = "Se requiere porque el caso habilita knowledge o RAG con grounding."
        topology = behavior.behavior_spec.multi_agent_topology
        if (
            context.pending_approvals
            or context.evaluation_gap_count > 0
            or not behavior.supported
            or (topology is not None and topology.support_state != "supported")
        ):
            roles.append("recovery")
            rationale["recovery"] = (
                "Se requiere para bloquear, escalar o remediar gaps, approvals, topologias planned_only o patrones fuera de soporte."
            )
        return PromptPlan(roles=roles, rationale_by_role=rationale)


class HeuristicEngine:
    def compile(
        self,
        context: Stage4CompilerContext,
        behavior: BehaviorCompilation,
        prompt_plan: PromptPlan,
    ) -> HeuristicDecisionV1:
        facts = [
            *context.facts,
            HeuristicDecisionFact(
                key="compiler_key",
                value=behavior.compiler_key,
                source="stage4.behavior_compiler_registry",
            ),
            HeuristicDecisionFact(
                key="compiler_support",
                value="supported" if behavior.supported else "planned",
                source="stage4.behavior_compiler_registry",
            ),
            HeuristicDecisionFact(
                key="prompt_role_count",
                value=str(len(prompt_plan.roles)),
                source="stage4.prompt_planner",
            ),
        ]
        review_notes = [
            "La heuristica sigue siendo rule-first; el texto libre solo complementa la narrativa.",
            "El LLM no puede modificar tools, permisos, autonomia, arquitectura ni guardrails aprobados.",
            behavior.support_note,
            f"Cobertura de prompts requerida: {', '.join(prompt_plan.roles)}.",
            *behavior.restrictions,
        ]
        decision_summary = (
            f"{context.decision_summary} "
            f"Compiler S4 seleccionado: {behavior.compiler_label} ({behavior.compiler_key}). "
            f"Support={'supported' if behavior.supported else 'planned_only'}."
        )
        return HeuristicDecisionV1(
            **base_metadata(context.snapshot, context.generated_at),
            decision_summary=decision_summary,
            decision_trace=list(context.decision_trace),
            candidate_catalog=list(context.candidate_catalog),
            facts=facts,
            recommended_prompts=list(prompt_plan.roles),
            review_notes=review_notes,
            provenance=provenance(
                (
                    "decision_trace",
                    ["blueprint.delivery_package.decision_trace"],
                    "Las decisiones heuristicas siguen siendo la evidencia principal del contrato rule-first.",
                ),
                (
                    "candidate_catalog",
                    ["blueprint.delivery_package.pattern_catalog"],
                    "El catalogo preserva alternativas consideradas y su fit relativo antes de compilar prompts.",
                ),
                (
                    "recommended_prompts",
                    ["stage4.prompt_planner", "tool-contract.v1", "memory-policy.v1", "knowledge-contract.v1"],
                    "La cobertura de prompts se deriva del patron compilado y de las capacidades aprobadas.",
                ),
            ),
        )


class LLMPolicyCompiler:
    def compile(
        self,
        context: Stage4CompilerContext,
        behavior: BehaviorCompilation,
        prompt_plan: PromptPlan,
    ) -> LLMPolicyV1:
        estimation = context.snapshot.estimation_report
        blueprint_policy = context.snapshot.blueprint.llm_policy if context.snapshot.blueprint is not None else None
        provider = coalesce(
            blueprint_policy.provider if blueprint_policy is not None else None,
            estimation.agentic.active_provider if estimation is not None else None,
            fallback="openai",
        )
        fast_model = coalesce(
            blueprint_policy.fast_model if blueprint_policy is not None else None,
            estimation.agentic.provider_model if estimation is not None else None,
            fallback="gpt-5-mini",
        )
        reasoning_model = coalesce(
            blueprint_policy.reasoning_model if blueprint_policy is not None else None,
            estimation.agentic.provider_model if estimation is not None else None,
            fallback="gpt-5.5",
        )
        fallback_model = coalesce(
            blueprint_policy.fallback_model if blueprint_policy is not None else None,
            fallback="manual_review_gate",
        )
        role_overrides = {
            item.role: item
            for item in (blueprint_policy.functions if blueprint_policy is not None else [])
            if coalesce(item.role, fallback="")
        }
        tool_names = [tool.name for tool in context.tool_contracts]
        role_context_sources = {
            role: assembled.context_sources
            for role, assembled in context.assembled_role_contexts.items()
        }

        def function_policy(function_key: str, role: str, intent: str, context_sources: list[str], tool_availability: list[str]) -> LLMFunctionPolicy:
            role_override = role_overrides.get(role)
            default_model = reasoning_model if role in {"planner", "evaluator", "recovery"} else fast_model
            return LLMFunctionPolicy(
                function_key=function_key,
                role=role,
                provider=coalesce(role_override.provider if role_override is not None else None, fallback=str(provider)),
                model=coalesce(role_override.model if role_override is not None else None, fallback=default_model),
                intent=intent,
                reasoning_effort=coalesce(
                    role_override.reasoning_effort if role_override is not None else None,
                    behavior.llm_reasoning_effort.get(role, "low"),
                    fallback="low",
                ),
                context_sources=role_context_sources.get(role, context_sources),
                tool_availability=(
                    [
                        item.strip()
                        for item in (role_override.tool_availability if role_override is not None else [])
                        if isinstance(item, str) and item.strip()
                    ]
                    or tool_availability
                ),
                fallback_model=coalesce(
                    role_override.fallback_model if role_override is not None else None,
                    fallback=fallback_model,
                ),
                max_tokens=(
                    role_override.max_tokens
                    if role_override is not None and role_override.max_tokens > 0
                    else behavior.llm_max_tokens.get(role, 1400)
                ),
            )

        functions = [
            function_policy(
                "planner",
                "planner",
                behavior.prompt_hints["planner"],
                ["blueprint-core.v1", "behavior-spec.v1", "heuristic-decision.v1"],
                [],
            ),
            function_policy(
                "executor",
                "executor",
                behavior.prompt_hints["executor"],
                ["behavior-spec.v1", "memory-policy.v1"],
                tool_names,
            ),
            function_policy(
                "evaluator",
                "evaluator",
                behavior.prompt_hints["evaluator"],
                ["evaluation-pack.v1", "behavior-spec.v1", "heuristic-decision.v1"],
                [],
            ),
        ]
        if "tool_use" in prompt_plan.roles:
            functions.append(
                function_policy(
                    "tool_use",
                    "tool_use",
                    behavior.prompt_hints.get("tool_use", "Validar contrato, permisos y compensacion antes de invocar una tool."),
                    ["tool-contract.v1", "llm-policy.v1"],
                    tool_names,
                )
            )
        if "memory" in prompt_plan.roles:
            functions.append(
                function_policy(
                    "memory",
                    "memory",
                    "Leer y escribir memoria solo cuando agregue continuidad trazable al objetivo actual.",
                    ["memory-policy.v1", "behavior-spec.v1"],
                    [],
                )
            )
        if "retrieval" in prompt_plan.roles:
            functions.append(
                function_policy(
                    "retrieval",
                    "retrieval",
                    "Recuperar evidencia autorizada, citar fuente y devolver needs-resolution cuando falte grounding.",
                    ["knowledge-contract.v1", "memory-policy.v1"],
                    [tool.name for tool in context.tool_contracts],
                )
            )
        if "recovery" in prompt_plan.roles:
            functions.append(
                function_policy(
                    "recovery",
                    "recovery",
                    behavior.prompt_hints.get("recovery", "Diagnosticar el bloqueo y proponer la remediation minima trazable."),
                    ["behavior-spec.v1", "heuristic-decision.v1", "evaluation-pack.v1"],
                    [],
                )
            )

        sampling_policy_by_compiler = {
            "deterministic": "Temperatura minima y salidas cerradas para planner, executor y evaluator.",
            "tool-calling": "Temperatura baja y validacion estricta antes de cada tool call.",
            "plan-and-execute": "Planner estructural con razonamiento alto; executor estable y sin improvisacion.",
            "react": "Ajuste local controlado para executor y razonamiento breve para planner.",
        }
        context_policy = "Trabajar solo con contratos versionados, datos aprobados y evidencia trazable del snapshot."
        if not behavior.supported:
            context_policy += " Si el patron sigue fuera de soporte, devolver needs-resolution y no abrir ejecucion real."

        fallback_policy = (
            "Bloquear o escalar a revision humana cuando falte contexto, falle una tool, se invalide el plan "
            "o el patron compilado no este soportado. Nunca degradar silenciosamente a otro patron."
        )

        return LLMPolicyV1(
            **base_metadata(context.snapshot, context.generated_at),
            provider=str(provider),
            fast_model=fast_model,
            reasoning_model=reasoning_model,
            fallback_model=fallback_model,
            functions=functions,
            context_policy=coalesce(
                blueprint_policy.context_policy if blueprint_policy is not None else None,
                fallback=context_policy,
            ),
            sampling_policy=coalesce(
                blueprint_policy.sampling_policy if blueprint_policy is not None else None,
                sampling_policy_by_compiler.get(behavior.compiler_key, "Temperatura baja y salidas estructuradas."),
                fallback="Temperatura baja y salidas estructuradas.",
            ),
            fallback_policy=coalesce(
                blueprint_policy.fallback_policy if blueprint_policy is not None else None,
                fallback=fallback_policy,
            ),
            circuit_breaker_policy=coalesce(
                blueprint_policy.circuit_breaker_policy if blueprint_policy is not None else None,
                fallback="Abrir circuit breaker por fallos consecutivos del proveedor o de la tool y escalar a review.",
            ),
            budget_policy=(
                coalesce(
                    blueprint_policy.budget_policy if blueprint_policy is not None else None,
                    estimation.agentic.pricing_policy if estimation is not None else None,
                    fallback="Priorizar costo acotado en planning/evaluation y reservar razonamiento para ejecucion compleja.",
                )
            ),
            output_validation_policy=coalesce(
                blueprint_policy.output_validation_policy if blueprint_policy is not None else None,
                fallback="Toda salida estructural debe validar contra schemas versionados antes de considerarse lista.",
            ),
            log_redaction_policy=coalesce(
                blueprint_policy.log_redaction_policy if blueprint_policy is not None else None,
                fallback="Redactar secretos, tokens y configuracion privada; conservar solo ids, contratos y trazas auditables.",
            ),
            provenance=provenance(
                (
                    "functions",
                    ["estimation_report.agentic", "tool-contract.v1", "behavior-spec.v1"],
                    "La politica por funcion se deriva del compilador, del costo estimado y del set aprobado de capacidades.",
                ),
                (
                    "sampling_policy",
                    ["behavior-spec.v1", "heuristic-decision.v1"],
                    "La politica de muestreo cambia segun el patron compilado y su nivel de control esperado.",
                ),
            ),
        )


class LLMPromptCompiler:
    def compile(
        self,
        context: Stage4CompilerContext,
        behavior: BehaviorCompilation,
        heuristic_decision: HeuristicDecisionV1,
        llm_policy: LLMPolicyV1,
        prompt_plan: PromptPlan,
    ) -> PromptPackV1:
        input_hash = self._input_hash(
            behavior.behavior_spec,
            heuristic_decision,
            llm_policy,
            context.memory_policy,
            context.knowledge_contract,
            context.tool_contracts,
        )
        role_contexts = context.assembled_role_contexts
        system_prompt = self._prompt_artifact(
            role="system",
            prompt_key="system",
            title=f"System prompt ({behavior.compiler_label})",
            content=render_system_prompt(context.snapshot, context.success_criteria, behavior.compiler_label),
            variables=context.prompt_variables,
            context_sources=["blueprint-core.v1", "heuristic-decision.v1"],
            output_schema=self._output_schema("system", behavior.compiler_key),
            guardrails=context.blueprint_guardrails,
            stop_conditions=context.stop_conditions,
            fallback="Escalar a revision humana si el contrato no cubre la decision requerida.",
            evaluation_case_keys=[case.key for case in context.evaluation_cases],
            input_contracts=["blueprint-core.v1", "heuristic-decision.v1"],
            dependency_payload={
                "compiler_key": behavior.compiler_key,
                "guardrails": context.blueprint_guardrails,
                "success_criteria": [item.model_dump(mode="json") for item in context.success_criteria],
                "decision_summary": heuristic_decision.decision_summary,
            },
            note="Prompt base del sistema gobernado por el compilador y los contratos aprobados.",
        )
        planner_prompt = self._prompt_artifact(
            role="planner",
            prompt_key="planner",
            title=f"Planner prompt ({behavior.compiler_label})",
            content="\n".join(
                [
                    behavior.prompt_hints["planner"],
                    f"Objetivo actual: {context.goal}",
                    "Estados compilados:",
                    format_states(behavior.behavior_spec.states),
                    f"Criteria de cierre: {', '.join(behavior.behavior_spec.termination_criteria)}",
                ]
            ),
            variables=context.prompt_variables,
            context_sources=role_contexts["planner"].context_sources,
            output_schema=self._output_schema("planner", behavior.compiler_key),
            guardrails=context.blueprint_guardrails,
            stop_conditions=context.stop_conditions,
            fallback="Si el contexto no alcanza, devolver needs-resolution con la decision faltante.",
            evaluation_case_keys=[case.key for case in context.evaluation_cases],
            input_contracts=["behavior-spec.v1", "heuristic-decision.v1"],
            dependency_payload={
                "compiler_key": behavior.compiler_key,
                "states": [state.model_dump(mode="json") for state in behavior.behavior_spec.states],
                "termination": behavior.behavior_spec.termination_criteria,
                "reasoning_pattern": behavior.behavior_spec.reasoning_pattern,
            },
            note="El planner conserva el patron compilado, los estados visibles y sus criterios de cierre.",
        )
        executor_prompt = self._prompt_artifact(
            role="executor",
            prompt_key="executor",
            title=f"Executor prompt ({behavior.compiler_label})",
            content="\n".join(
                [
                    behavior.prompt_hints["executor"],
                    "Solo puedes avanzar un estado permitido por turno.",
                    "Estados compilados:",
                    format_states(behavior.behavior_spec.states),
                    f"Checkpoint policy: {behavior.behavior_spec.checkpoint_policy}",
                    f"Timeout policy: {behavior.behavior_spec.timeout_policy}",
                ]
            ),
            variables=context.prompt_variables,
            context_sources=role_contexts["executor"].context_sources,
            output_schema=self._output_schema("executor", behavior.compiler_key),
            guardrails=context.blueprint_guardrails,
            stop_conditions=context.stop_conditions,
            fallback="Solicitar remediation o approval antes de continuar cuando un estado quede bloqueado.",
            evaluation_case_keys=[case.key for case in context.evaluation_cases],
            input_contracts=["behavior-spec.v1", "memory-policy.v1"],
            dependency_payload={
                "compiler_key": behavior.compiler_key,
                "states": [state.model_dump(mode="json") for state in behavior.behavior_spec.states],
                "checkpoint_policy": behavior.behavior_spec.checkpoint_policy,
                "timeout_policy": behavior.behavior_spec.timeout_policy,
                "memory_strategy": context.memory_policy.strategy,
            },
            note="La ejecucion deriva solo del behavior spec compilado y de la policy de memoria aprobada.",
        )
        evaluator_prompt = self._prompt_artifact(
            role="evaluator",
            prompt_key="evaluator",
            title=f"Evaluator prompt ({behavior.compiler_label})",
            content="\n".join(
                [
                    behavior.prompt_hints["evaluator"],
                    "Valida readiness, termination, retries, fallbacks y trazabilidad antes de cerrar.",
                    f"Acceptance cases: {', '.join(case.key for case in context.evaluation_cases) or 'sin casos declarados'}",
                    f"Termination criteria: {', '.join(behavior.behavior_spec.termination_criteria)}",
                ]
            ),
            variables=context.prompt_variables,
            context_sources=role_contexts["evaluator"].context_sources,
            output_schema=self._output_schema("evaluator", behavior.compiler_key),
            guardrails=context.blueprint_guardrails,
            stop_conditions=context.stop_conditions,
            fallback="Si la evaluacion no es concluyente, marcar blocked y describir la evidencia faltante.",
            evaluation_case_keys=[case.key for case in context.evaluation_cases],
            input_contracts=["evaluation-pack.v1", "behavior-spec.v1", "heuristic-decision.v1"],
            dependency_payload={
                "compiler_key": behavior.compiler_key,
                "evaluation_cases": [case.model_dump(mode="json") for case in context.evaluation_cases],
                "termination": behavior.behavior_spec.termination_criteria,
            },
            note="El evaluator se alimenta del contrato de evaluacion y del comportamiento compilado.",
        )

        tool_use_prompt = None
        if "tool_use" in prompt_plan.roles:
            tool_use_prompt = self._prompt_artifact(
                role="tool_use",
                prompt_key="tool_use",
                title=f"Tool-use prompt ({behavior.compiler_label})",
                content="\n".join(
                    [
                        behavior.prompt_hints.get("tool_use", "Valida el contrato de la tool antes de usarla."),
                        f"Tools aprobadas: {', '.join(tool.name for tool in context.tool_contracts) or 'ninguna'}",
                        "Nunca invoques una tool sin schema, permisos y rollback claros.",
                    ]
                ),
                variables=context.prompt_variables,
                context_sources=role_contexts["tool_use"].context_sources,
                output_schema=self._output_schema("tool_use", behavior.compiler_key),
                guardrails=context.blueprint_guardrails,
                stop_conditions=context.stop_conditions,
                fallback="No llamar ninguna tool si el contrato requerido no esta completo.",
                evaluation_case_keys=[case.key for case in context.evaluation_cases],
                input_contracts=["tool-contract.v1", "llm-policy.v1"],
                dependency_payload={
                    "compiler_key": behavior.compiler_key,
                    "tools": [tool.model_dump(mode="json") for tool in context.tool_contracts],
                },
                note="Prompt especializado para el set aprobado de tools y sus contratos versionados.",
            )

        memory_prompt = None
        if "memory" in prompt_plan.roles:
            memory_runtime_context = role_contexts["memory"]
            memory_prompt = self._prompt_artifact(
                role="memory",
                prompt_key="memory",
                title=f"Memory prompt ({behavior.compiler_label})",
                content="\n".join(
                    [
                        "Escribe y recupera memoria solo cuando ayude al objetivo actual y siempre con trazabilidad de origen.",
                        f"Strategy: {context.memory_policy.strategy}",
                        f"Write policy: {context.memory_policy.write_policy}",
                        f"Retrieval policy: {context.memory_policy.retrieval_policy}",
                        f"Retrieval scopes: {', '.join(context.memory_policy.retrieval_scopes) or 'sin scopes'}",
                        f"Summary policy: {context.memory_policy.summary_policy}",
                        f"Invalidation policy: {context.memory_policy.invalidation_policy}",
                        "Context budgets: "
                        + (
                            ", ".join(
                                f"{item.role}={item.max_tokens}t/{item.max_items}i/{item.max_chars}c"
                                for item in context.memory_policy.context_budgets
                            )
                            or "sin budgets"
                        ),
                        f"Retention policy: {context.memory_policy.retention_policy}",
                        f"TTL policy: {context.memory_policy.ttl_policy}",
                        f"Sensitivity rules: {', '.join(context.memory_policy.sensitivity_rules) or 'sin reglas extra'}",
                        f"No evidence behavior: {context.memory_policy.grounding_policy.get('no_evidence_behavior', '')}",
                        *memory_runtime_context.prompt_digest_lines,
                        "No expandas libremente la policy: usa solo los contratos y fuentes staged listados.",
                    ]
                ),
                variables=context.prompt_variables,
                context_sources=memory_runtime_context.context_sources,
                output_schema=self._output_schema("memory", behavior.compiler_key),
                guardrails=context.blueprint_guardrails,
                stop_conditions=context.stop_conditions,
                fallback="Si la memoria no es confiable o no existe, continuar con el estado explicito del workflow.",
                evaluation_case_keys=[case.key for case in context.evaluation_cases],
                input_contracts=["memory-policy.v1", "short-term-memory.v1", "knowledge-manifest.v1"],
                dependency_payload={
                    "strategy": context.memory_policy.strategy,
                    "storage_layers": context.memory_policy.storage_layers,
                    "write_policy": context.memory_policy.write_policy,
                    "retrieval_policy": context.memory_policy.retrieval_policy,
                    "retrieval_scopes": context.memory_policy.retrieval_scopes,
                    "summary_policy": context.memory_policy.summary_policy,
                    "invalidation_policy": context.memory_policy.invalidation_policy,
                    "context_budgets": [item.model_dump(mode="json") for item in context.memory_policy.context_budgets],
                    "retention_policy": context.memory_policy.retention_policy,
                    "ttl_policy": context.memory_policy.ttl_policy,
                    "grounding_policy": context.memory_policy.grounding_policy,
                    "sensitivity_rules": context.memory_policy.sensitivity_rules,
                    "assembled_context_sources": memory_runtime_context.context_sources,
                },
                note="El prompt de memoria sigue la policy aprobada y no la expande.",
            )

        retrieval_prompt = None
        if "retrieval" in prompt_plan.roles:
            retrieval_runtime_context = role_contexts["retrieval"]
            retrieval_prompt = self._prompt_artifact(
                role="retrieval",
                prompt_key="retrieval",
                title=f"Retrieval prompt ({behavior.compiler_label})",
                content="\n".join(
                    [
                        "Recupera solo evidencia autorizada, cita la fuente usada y declara ausencia de evidencia cuando aplique.",
                        f"Knowledge mode: {context.knowledge_contract.mode}",
                        f"Sources: {', '.join(source.title for source in context.knowledge_contract.sources) or 'sin fuentes'}",
                        f"Source lineage: {', '.join(context.knowledge_contract.source_lineage) or 'sin lineage'}",
                        f"Fallback: {context.knowledge_contract.grounding_policy.get('no_evidence_behavior', '')}",
                        f"Sensitivity rules: {', '.join(context.knowledge_contract.sensitivity_rules) or 'sin reglas extra'}",
                        *retrieval_runtime_context.prompt_digest_lines,
                        "No expandas libremente el retrieval: usa solo las fuentes staged y la policy aprobada.",
                    ]
                ),
                variables=context.prompt_variables,
                context_sources=retrieval_runtime_context.context_sources,
                output_schema=self._output_schema("retrieval", behavior.compiler_key),
                guardrails=context.blueprint_guardrails,
                stop_conditions=context.stop_conditions,
                fallback="Si no hay evidencia suficiente, devolver needs-resolution sin improvisar una fuente.",
                evaluation_case_keys=[case.key for case in context.evaluation_cases],
                input_contracts=["knowledge-contract.v1", "memory-policy.v1", "knowledge-manifest.v1", "short-term-memory.v1"],
                dependency_payload={
                    "knowledge_contract": context.knowledge_contract.model_dump(mode="json"),
                    "assembled_context_sources": retrieval_runtime_context.context_sources,
                },
                note="El retrieval se habilita solo por evidencia contractual del caso.",
            )

        recovery_prompt = None
        if "recovery" in prompt_plan.roles:
            recovery_runtime_context = role_contexts["recovery"]
            recovery_prompt = self._prompt_artifact(
                role="recovery",
                prompt_key="recovery",
                title=f"Recovery prompt ({behavior.compiler_label})",
                content="\n".join(
                    [
                        behavior.prompt_hints.get("recovery", "Describe el gap, la evidencia faltante y la minima remediation necesaria."),
                        f"Pending approvals: {', '.join(context.pending_approvals) or 'ninguno'}",
                        f"Evaluation gaps: {context.evaluation_gap_count}",
                        *recovery_runtime_context.prompt_digest_lines,
                        "No expandas la remediation fuera del contexto staged y de los blockers aprobados.",
                    ]
                ),
                variables=context.prompt_variables,
                context_sources=recovery_runtime_context.context_sources,
                output_schema=self._output_schema("recovery", behavior.compiler_key),
                guardrails=context.blueprint_guardrails,
                stop_conditions=context.stop_conditions,
                fallback="Mantener el estado blocked si la remediation depende de un owner externo.",
                evaluation_case_keys=[case.key for case in context.evaluation_cases],
                input_contracts=["behavior-spec.v1", "heuristic-decision.v1", "evaluation-pack.v1", "short-term-memory.v1"],
                dependency_payload={
                    "compiler_key": behavior.compiler_key,
                    "pending_approvals": context.pending_approvals,
                    "evaluation_gap_count": context.evaluation_gap_count,
                    "supported": behavior.supported,
                    "assembled_context_sources": recovery_runtime_context.context_sources,
                },
                note="El recovery responde solo a gaps, approvals o patrones fuera de soporte ya detectados.",
            )

        topology = behavior.behavior_spec.multi_agent_topology
        agent_role_prompts: list[PromptArtifactV1] = []
        handoff_prompts: list[PromptArtifactV1] = []
        if topology is not None:
            for contract in topology.agent_contracts:
                allowed_tools = ", ".join(contract.permissions.allowed_tools) or "ninguna"
                required_approvals = ", ".join(contract.permissions.required_approvals) or "ninguno"
                agent_role_prompts.append(
                    self._prompt_artifact(
                        role="agent_role",
                        prompt_key=f"agent_role_{contract.agent_key}",
                        title=f"Agent role prompt ({contract.agent_key})",
                        content="\n".join(
                            [
                                f"Rol objetivo: {contract.role}",
                                f"Proposito: {contract.purpose}",
                                f"Modo runtime: {contract.runtime_mode}",
                                f"Allowed tools: {allowed_tools}",
                                f"Required approvals: {required_approvals}",
                                f"Input contracts: {', '.join(contract.input_contracts) or 'ninguno'}",
                                f"Output contracts: {', '.join(contract.output_contracts) or 'ninguno'}",
                                f"Success signals: {', '.join(contract.success_signals) or 'sin senales extra'}",
                                f"Isolation boundary: {contract.isolation_boundary}",
                                f"Retry strategy: {contract.retry_strategy}",
                                f"Timeout policy: {contract.timeout_policy}",
                                f"Failure mode: {contract.failure_mode}",
                                f"Escalation policy: {contract.permissions.escalation_policy}",
                            ]
                        ),
                        variables=context.prompt_variables,
                        context_sources=["behavior-spec.v1", "tool-contract.v1", "llm-policy.v1"],
                        output_schema=self._output_schema("agent_role", behavior.compiler_key),
                        guardrails=context.blueprint_guardrails,
                        stop_conditions=context.stop_conditions,
                        fallback="Escalar al supervisor si el rol requiere capacidades o permisos fuera de su boundary.",
                        evaluation_case_keys=[case.key for case in context.evaluation_cases],
                        input_contracts=["behavior-spec.v1", *contract.input_contracts],
                        dependency_payload={
                            "compiler_key": behavior.compiler_key,
                            "multi_agent_contract": contract.model_dump(mode="json"),
                            "support_state": topology.support_state,
                        },
                        note="Prompt adicional por agente derivado del contrato de rol multiagente y sus boundaries.",
                    )
                )
            for handoff in topology.handoff_contracts:
                handoff_prompts.append(
                    self._prompt_artifact(
                        role="handoff",
                        prompt_key=f"handoff_{handoff.handoff_key}",
                        title=f"Handoff prompt ({handoff.handoff_key})",
                        content="\n".join(
                            [
                                f"Handoff key: {handoff.handoff_key}",
                                f"From: {handoff.from_agent}",
                                f"To: {handoff.to_agent}",
                                f"Trigger: {handoff.trigger}",
                                f"Ownership transfer: {handoff.ownership_transfer}",
                                f"Required artifacts: {', '.join(handoff.required_artifacts) or 'ninguno'}",
                                f"Success criteria: {', '.join(handoff.success_criteria) or 'sin criterios extra'}",
                                f"Failure behavior: {handoff.failure_behavior}",
                                f"Audit trail: {', '.join(handoff.audit_trail) or 'sin trail adicional'}",
                            ]
                        ),
                        variables=context.prompt_variables,
                        context_sources=["behavior-spec.v1", "evaluation-pack.v1"],
                        output_schema=self._output_schema("handoff", behavior.compiler_key),
                        guardrails=context.blueprint_guardrails,
                        stop_conditions=context.stop_conditions,
                        fallback="Bloquear el handoff si falta un artefacto requerido o el ownership no quedo explicito.",
                        evaluation_case_keys=[case.key for case in context.evaluation_cases],
                        input_contracts=["behavior-spec.v1", *handoff.required_artifacts],
                        dependency_payload={
                            "compiler_key": behavior.compiler_key,
                            "handoff_contract": handoff.model_dump(mode="json"),
                            "support_state": topology.support_state,
                        },
                        note="Prompt adicional para asegurar que el handoff preserve ownership, criteria y audit trail.",
                    )
                )

        return PromptPackV1(
            **base_metadata(context.snapshot, context.generated_at),
            origin=PromptPackOrigin(
                blueprint_core_version="blueprint-core.v1",
                behavior_spec_version=behavior.behavior_spec.schema_version,
                llm_policy_version=llm_policy.schema_version,
                heuristic_decision_version=heuristic_decision.schema_version,
                input_hash=input_hash,
            ),
            system_prompt=system_prompt,
            planner_prompt=planner_prompt,
            executor_prompt=executor_prompt,
            evaluator_prompt=evaluator_prompt,
            tool_use_prompt=tool_use_prompt,
            memory_prompt=memory_prompt,
            retrieval_prompt=retrieval_prompt,
            recovery_prompt=recovery_prompt,
            agent_role_prompts=agent_role_prompts,
            handoff_prompts=handoff_prompts,
            provenance=provenance(
                (
                    "origin",
                    ["behavior-spec.v1", "heuristic-decision.v1", "llm-policy.v1"],
                    "El prompt pack queda amarrado al compilador, a sus contratos fuente y al input hash resultante.",
                ),
            ),
        )

    def _prompt_artifact(
        self,
        *,
        role: str,
        prompt_key: str,
        title: str,
        content: str,
        variables: Sequence[PromptVariable],
        context_sources: list[str],
        output_schema: dict[str, Any],
        guardrails: list[str],
        stop_conditions: list[str],
        fallback: str,
        evaluation_case_keys: list[str],
        input_contracts: list[str],
        dependency_payload: dict[str, Any],
        note: str,
    ) -> PromptArtifactV1:
        payload_hash = hashlib.sha256(
            json.dumps(
                stable_hash_payload(
                    {
                        "role": role,
                        "prompt_key": prompt_key,
                        "title": title,
                        "content": content,
                        "variables": [item.model_dump(mode="json") for item in variables],
                        "context_sources": context_sources,
                        "output_schema": output_schema,
                        "guardrails": guardrails,
                        "stop_conditions": stop_conditions,
                        "fallback": fallback,
                        "evaluation_case_keys": evaluation_case_keys,
                        "input_contracts": input_contracts,
                        **dependency_payload,
                    }
                ),
                ensure_ascii=True,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache_key = (role, payload_hash)
        cached = _PROMPT_ARTIFACT_CACHE.get(cache_key)
        if cached is not None:
            return cached.model_copy(deep=True)
        artifact = PromptArtifactV1(
            prompt_key=prompt_key,
            role=role,
            title=title,
            content=content,
            variables=list(variables),
            context_sources=context_sources,
            output_schema=output_schema,
            guardrails=guardrails,
            stop_conditions=stop_conditions,
            fallback=fallback,
            evaluation_case_keys=evaluation_case_keys,
            input_contracts=input_contracts,
            provenance=provenance(
                ("content", input_contracts or context_sources, note),
            ),
        )
        _PROMPT_ARTIFACT_CACHE[cache_key] = artifact
        return artifact.model_copy(deep=True)

    def _input_hash(
        self,
        behavior_spec: BehaviorSpecV1,
        heuristic_decision: HeuristicDecisionV1,
        llm_policy: LLMPolicyV1,
        memory_policy: MemoryPolicyV1,
        knowledge_contract: KnowledgeContractV1,
        tool_contracts: Sequence[ToolContractV1],
    ) -> str:
        payload = {
            "behavior_spec": behavior_spec.model_dump(mode="json"),
            "heuristic_decision": heuristic_decision.model_dump(mode="json"),
            "llm_policy": llm_policy.model_dump(mode="json"),
            "memory_policy": memory_policy.model_dump(mode="json"),
            "knowledge_contract": knowledge_contract.model_dump(mode="json"),
            "tool_contracts": [tool.model_dump(mode="json") for tool in tool_contracts],
        }
        return hashlib.sha256(
            json.dumps(stable_hash_payload(payload), ensure_ascii=True, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _output_schema(self, role: str, compiler_key: str) -> dict[str, Any]:
        if role == "planner":
            if compiler_key == "plan-and-execute":
                return {
                    "type": "object",
                    "properties": {
                        "plan": {"type": "array", "items": {"type": "string"}},
                        "risks": {"type": "array", "items": {"type": "string"}},
                        "checkpoints": {"type": "array", "items": {"type": "string"}},
                        "replan_needed": {"type": "boolean"},
                    },
                    "required": ["plan", "replan_needed"],
                }
            if compiler_key == "react":
                return {
                    "type": "object",
                    "properties": {
                        "next_actions": {"type": "array", "items": {"type": "string"}},
                        "observation_targets": {"type": "array", "items": {"type": "string"}},
                        "tool_intent": {"type": "string"},
                    },
                    "required": ["next_actions"],
                }
            if compiler_key == "tool-calling":
                return {
                    "type": "object",
                    "properties": {
                        "tool_sequence": {"type": "array", "items": {"type": "string"}},
                        "validation_points": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["tool_sequence"],
                }
            if compiler_key.startswith("unsupported::"):
                return {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "missing_capability": {"type": "string"},
                        "remediation": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["status", "missing_capability"],
                }
            return {
                "type": "object",
                "properties": {
                    "checklist": {"type": "array", "items": {"type": "string"}},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["checklist"],
            }
        if role == "executor":
            if compiler_key == "react":
                return {
                    "type": "object",
                    "properties": {
                        "thought_summary": {"type": "string"},
                        "action": {"type": "string"},
                        "observation": {"type": "string"},
                        "needs_approval": {"type": "boolean"},
                    },
                    "required": ["action", "observation", "needs_approval"],
                }
            if compiler_key == "tool-calling":
                return {
                    "type": "object",
                    "properties": {
                        "selected_tool": {"type": "string"},
                        "tool_input": {"type": "object"},
                        "validation_result": {"type": "string"},
                        "needs_approval": {"type": "boolean"},
                    },
                    "required": ["selected_tool", "validation_result", "needs_approval"],
                }
            if compiler_key.startswith("unsupported::"):
                return {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "remediation": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["status"],
                }
            return {
                "type": "object",
                "properties": {
                    "state": {"type": "string"},
                    "action": {"type": "string"},
                    "artifacts": {"type": "array", "items": {"type": "string"}},
                    "needs_approval": {"type": "boolean"},
                },
                "required": ["state", "action", "needs_approval"],
            }
        if role in {"evaluator", "recovery"}:
            return {
                "type": "object",
                "properties": {
                    "readiness": {"type": "string"},
                    "blocking_issues": {"type": "array", "items": {"type": "string"}},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                    "termination_reached": {"type": "boolean"},
                },
                "required": ["readiness", "termination_reached"],
            }
        if role == "agent_role":
            return {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "findings": {"type": "array", "items": {"type": "string"}},
                    "escalation_needed": {"type": "boolean"},
                },
                "required": ["status", "findings", "escalation_needed"],
            }
        if role == "handoff":
            return {
                "type": "object",
                "properties": {
                    "handoff_ready": {"type": "boolean"},
                    "missing_artifacts": {"type": "array", "items": {"type": "string"}},
                    "ownership_confirmed": {"type": "boolean"},
                },
                "required": ["handoff_ready", "ownership_confirmed"],
            }
        return {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
            },
            "required": ["response"],
        }


class PromptValidator:
    def validate(
        self,
        context: Stage4CompilerContext,
        behavior: BehaviorCompilation,
        prompt_plan: PromptPlan,
        prompt_pack: PromptPackV1,
    ) -> None:
        issues: list[str] = []
        prompts_by_role = {
            "system": prompt_pack.system_prompt,
            "planner": prompt_pack.planner_prompt,
            "executor": prompt_pack.executor_prompt,
            "evaluator": prompt_pack.evaluator_prompt,
            "tool_use": prompt_pack.tool_use_prompt,
            "memory": prompt_pack.memory_prompt,
            "retrieval": prompt_pack.retrieval_prompt,
            "recovery": prompt_pack.recovery_prompt,
        }
        for role in prompt_plan.roles:
            if prompts_by_role.get(role) is None:
                issues.append(f"missing_prompt:{role}")
        if not context.has_tools and prompt_pack.tool_use_prompt is not None:
            issues.append("tool_use_without_tools")
        if not context.has_knowledge and prompt_pack.retrieval_prompt is not None:
            issues.append("retrieval_without_knowledge")
        if not behavior.supported and "needs-resolution" not in prompt_pack.executor_prompt.content:
            issues.append("unsupported_pattern_without_needs_resolution")
        topology = behavior.behavior_spec.multi_agent_topology
        if topology is not None:
            if len(prompt_pack.agent_role_prompts) != len(topology.agent_contracts):
                issues.append("agent_role_prompt_count_mismatch")
            if len(prompt_pack.handoff_prompts) != len(topology.handoff_contracts):
                issues.append("handoff_prompt_count_mismatch")
        for role in prompt_plan.roles:
            artifact = prompts_by_role[role]
            if artifact is None or not artifact.output_schema:
                issues.append(f"missing_output_schema:{role}")
            if artifact is not None and not artifact.context_sources:
                issues.append(f"missing_context_sources:{role}")
            if role in {"memory", "retrieval", "recovery"} and artifact is not None and "Assembled context:" not in artifact.content:
                issues.append(f"missing_assembled_context_digest:{role}")
        if issues:
            raise ValueError(f"Prompt validation failed: {', '.join(sorted(issues))}")


class PromptEvaluator:
    def evaluate(
        self,
        context: Stage4CompilerContext,
        behavior: BehaviorCompilation,
        prompt_pack: PromptPackV1,
    ) -> None:
        prompts = [
            prompt_pack.system_prompt,
            prompt_pack.planner_prompt,
            prompt_pack.executor_prompt,
            prompt_pack.evaluator_prompt,
            prompt_pack.tool_use_prompt,
            prompt_pack.memory_prompt,
            prompt_pack.retrieval_prompt,
            prompt_pack.recovery_prompt,
            *prompt_pack.agent_role_prompts,
            *prompt_pack.handoff_prompts,
        ]
        expected_case_keys = [case.key for case in context.evaluation_cases]
        for prompt in [item for item in prompts if item is not None]:
            if expected_case_keys and prompt.evaluation_case_keys != expected_case_keys:
                raise ValueError(f"Prompt evaluation failed: coverage mismatch for role={prompt.role}")
        if not behavior.behavior_spec.states:
            raise ValueError("Prompt evaluation failed: behavior spec without states")


def build_stage4_context(
    snapshot: SessionSnapshot,
    *,
    generated_at: Any,
    tool_contracts: Sequence[ToolContractV1],
    memory_policy: MemoryPolicyV1,
    knowledge_contract: KnowledgeContractV1,
    success_criteria: Sequence[SuccessCriterion] | None = None,
) -> Stage4CompilerContext:
    criteria = list(success_criteria or derive_success_criteria(snapshot))
    return ContextNormalizer().normalize(
        snapshot,
        generated_at=generated_at,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=criteria,
    )


def compile_stage4_artifacts(
    snapshot: SessionSnapshot,
    *,
    generated_at: Any,
    tool_contracts: Sequence[ToolContractV1],
    memory_policy: MemoryPolicyV1,
    knowledge_contract: KnowledgeContractV1,
    success_criteria: Sequence[SuccessCriterion] | None = None,
    compiler_key_override: str | None = None,
) -> Stage4CompilationResult:
    context = build_stage4_context(
        snapshot,
        generated_at=generated_at,
        tool_contracts=tool_contracts,
        memory_policy=memory_policy,
        knowledge_contract=knowledge_contract,
        success_criteria=success_criteria,
    )
    registry = DEFAULT_BEHAVIOR_COMPILER_REGISTRY
    behavior = registry.compile_named(compiler_key_override, context) if compiler_key_override else registry.select(context)
    behavior = _decorate_behavior_with_multi_agent_topology(context, behavior)
    prompt_plan = PromptPlanner().plan(context, behavior)
    heuristic_decision = HeuristicEngine().compile(context, behavior, prompt_plan)
    llm_policy = LLMPolicyCompiler().compile(context, behavior, prompt_plan)
    prompt_pack = LLMPromptCompiler().compile(context, behavior, heuristic_decision, llm_policy, prompt_plan)
    PromptValidator().validate(context, behavior, prompt_plan, prompt_pack)
    PromptEvaluator().evaluate(context, behavior, prompt_pack)
    return Stage4CompilationResult(
        context=context,
        behavior=behavior,
        heuristic_decision=heuristic_decision,
        llm_policy=llm_policy,
        prompt_pack=prompt_pack,
        prompt_plan=prompt_plan,
    )
