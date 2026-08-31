from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    ApprovalGateRecord,
    ApprovalStatus,
    ArtifactStatus,
    BlueprintArtifact,
    GeneratedDeliverable,
    GovernancePolicyEntry,
    GovernancePolicyRecord,
    HandoffRecord,
    HandoffRecordEntry,
    ReviewState,
    RuntimeFeatureFlagRecord,
    SessionRecord,
    SessionSnapshot,
    SessionStage,
    SubagentRunEntry,
    SubagentRunRecord,
    WorkflowProfile,
    WorkflowStep,
    WorkflowTemplateEntry,
    WorkflowTemplateRecord,
    utc_now,
)


HANDOFF_STATUS_PENDING = "pending"
HANDOFF_STATUS_COMPLETED = "completed"
HANDOFF_STATUS_RETURNED = "returned"

FEATURE_FLAG_WORKFLOWS = "workflow_templates_v1"
FEATURE_FLAG_GOVERNANCE = "governance_console_v1"
FEATURE_FLAG_SUBAGENTS = "specialized_subagents_v1"
FEATURE_FLAG_MULTI_AGENT_RUNTIME = "multi_agent_runtime_v1"
FEATURE_FLAG_ESTIMATION = "estimation_comparative_v1"
FEATURE_FLAG_TOOL_RECOMMENDATION = "tool_recommendation_llm_v1"
FEATURE_FLAG_REACT_RUNTIME = "react_runtime_v1"
FEATURE_FLAG_DESIGN_INTELLIGENCE = "design_intelligence_v2"
FEATURE_FLAG_BLUEPRINT_TIER_POLICY = "blueprint_tier_policy_enabled"
FEATURE_FLAG_DELIVERABLE_CATALOG = "deliverable_catalog_enabled"
FEATURE_FLAG_DELIVERABLE_GOVERNANCE_ADMIN = "deliverable_governance_admin_enabled"

SPECIALIST_RUN_KINDS = ("evaluation_specialist", "risk_specialist", "artifact_specialist")


def _workflow_profile_payload(
    *,
    execution_pattern: str,
    checkpoint_policy: str,
    approval_pause: str,
    steps: list[dict[str, Any]],
    inbox_strategy: str = "Input estructurado desde discovery y decisiones persistidas",
    outbox_strategy: str = "Artefactos versionados, handoff y exportes trazables",
    retry_strategy: str = "Retry controlado por etapa y rerun parcial por skill",
    compensation_strategy: str = "Rollback a blueprint previo y retorno a revision humana",
    timeout_policy: str = "Timeout corto por etapa con escalamiento visible",
) -> dict[str, Any]:
    return WorkflowProfile(
        execution_pattern=execution_pattern,
        inbox_strategy=inbox_strategy,
        outbox_strategy=outbox_strategy,
        checkpoint_policy=checkpoint_policy,
        retry_strategy=retry_strategy,
        compensation_strategy=compensation_strategy,
        approval_pause=approval_pause,
        timeout_policy=timeout_policy,
        steps=[WorkflowStep.model_validate(item) for item in steps],
    ).model_dump(mode="json")


DEFAULT_WORKFLOW_TEMPLATES = [
    {
        "template_key": "single_agent_linear",
        "label": "Single agent linear",
        "summary": "Flujo corto con checkpoints ligeros y handoff final simple.",
        "architecture_scope": ["single_agent", "single_agent_with_skills"],
        "supports_approvals": False,
        "supports_handoffs": True,
        "governance_hints": [
            "Usar cuando el MVP tiene bajo riesgo operativo.",
            "Mantener aprobaciones como excepcion y no como regla.",
        ],
        "workflow_profile": _workflow_profile_payload(
            execution_pattern="Flujo lineal con validacion ligera y handoff final.",
            checkpoint_policy="Checkpoint despues de discovery, blueprint y export final.",
            approval_pause="Solo pausar si aparece una tool con side effects o un riesgo alto.",
            steps=[
                {
                    "name": "discovery_capture",
                    "objective": "Normalizar el problema y el alcance minimo.",
                    "actor": "builder",
                    "outputs": ["structured_discovery"],
                    "fallback": "Pedir ajustes de discovery si hay huecos fuertes.",
                    "requires_approval": False,
                },
                {
                    "name": "blueprint_build",
                    "objective": "Construir el blueprint con memoria, tools y guardrails.",
                    "actor": "builder",
                    "outputs": ["blueprint"],
                    "fallback": "Volver a definir canvas o riesgo si queda parcial.",
                    "requires_approval": False,
                },
                {
                    "name": "implementation_handoff",
                    "objective": "Preparar exportes y handoff tecnico.",
                    "actor": "human_reviewer",
                    "outputs": ["implementation_handoff"],
                    "fallback": "Regresar al blueprint si el paquete no es implementable.",
                    "requires_approval": False,
                },
            ],
        ),
    },
    {
        "template_key": "approval_gate_workflow",
        "label": "Approval gate workflow",
        "summary": "Flujo durable con pausa formal de aprobacion antes de promover el paquete.",
        "architecture_scope": ["single_agent", "single_agent_with_skills", "handoffs"],
        "supports_approvals": True,
        "supports_handoffs": True,
        "governance_hints": [
            "Usar cuando existen tools con side effects.",
            "Asegurar gate explicito antes del handoff a implementacion.",
        ],
        "workflow_profile": _workflow_profile_payload(
            execution_pattern="Flujo con gate humano obligatorio antes de promotion.",
            checkpoint_policy="Checkpoint en blueprint, evaluacion y gate de promotion.",
            approval_pause="Pausar antes de activar side effects o abrir handoff a implementacion.",
            steps=[
                {
                    "name": "discovery_capture",
                    "objective": "Normalizar discovery y decisiones no delegables.",
                    "actor": "builder",
                    "outputs": ["structured_discovery"],
                    "fallback": "Completar faltantes antes del canvas.",
                    "requires_approval": False,
                },
                {
                    "name": "blueprint_build",
                    "objective": "Generar blueprint y matriz de riesgo.",
                    "actor": "builder",
                    "outputs": ["blueprint", "risk_matrix"],
                    "fallback": "Volver a diseno si la cobertura queda parcial.",
                    "requires_approval": False,
                },
                {
                    "name": "approval_gate",
                    "objective": "Validar riesgos, side effects y readiness del paquete.",
                    "actor": "local_admin",
                    "outputs": ["approved_blueprint_package"],
                    "fallback": "Retornar el handoff a blueprint si el gate falla.",
                    "requires_approval": True,
                },
                {
                    "name": "implementation_handoff",
                    "objective": "Transferir el paquete a implementacion con trazabilidad.",
                    "actor": "human_reviewer",
                    "outputs": ["implementation_handoff"],
                    "fallback": "Pedir ajustes y reabrir el gate.",
                    "requires_approval": True,
                },
            ],
        ),
    },
    {
        "template_key": "handoff_governed_workflow",
        "label": "Handoff governed workflow",
        "summary": "Flujo secuencial con checkpoints, retorno y ownership por fase.",
        "architecture_scope": ["handoffs", "supervisor_with_subagents"],
        "supports_approvals": True,
        "supports_handoffs": True,
        "governance_hints": [
            "Usar cuando el proceso necesita ownership por etapa.",
            "Registrar retorno explicito al blueprint si el paquete falla el review.",
        ],
        "workflow_profile": _workflow_profile_payload(
            execution_pattern="Handoffs secuenciales con retorno controlado y ownership por fase.",
            checkpoint_policy="Checkpoint formal en discovery, blueprint, evaluacion y handoff final.",
            approval_pause="Pausar en cada handoff que cambie de owner o exponga side effects.",
            steps=[
                {
                    "name": "builder_design",
                    "objective": "Consolidar discovery, canvas y blueprint inicial.",
                    "actor": "builder",
                    "outputs": ["blueprint_candidate"],
                    "fallback": "Volver a discovery si el alcance sigue ambiguo.",
                    "requires_approval": False,
                },
                {
                    "name": "governance_review",
                    "objective": "Revisar gates, riesgos y readiness del paquete.",
                    "actor": "local_admin",
                    "outputs": ["governed_blueprint"],
                    "fallback": "Retornar a blueprint con nota de gobierno.",
                    "requires_approval": True,
                },
                {
                    "name": "implementation_handoff",
                    "objective": "Transferir el paquete al owner de implementacion.",
                    "actor": "implementation_owner",
                    "outputs": ["implementation_handoff"],
                    "fallback": "Retornar al builder si faltan artefactos o evidencia.",
                    "requires_approval": True,
                },
            ],
        ),
    },
    {
        "template_key": "subagent_escalation_workflow",
        "label": "Subagent escalation workflow",
        "summary": "Escala a procesos especializados solo cuando el caso ya justifico complejidad adicional.",
        "architecture_scope": ["supervisor_with_subagents", "router_parallel"],
        "supports_approvals": True,
        "supports_handoffs": True,
        "governance_hints": [
            "Usar solo bajo feature flag y con evidencia de necesidad.",
            "Mantener subagentes como procesos especializados y no como base por defecto.",
        ],
        "workflow_profile": _workflow_profile_payload(
            execution_pattern="Flujo gobernado con escalamiento opcional a subprocesos especializados.",
            checkpoint_policy="Checkpoint antes y despues de cualquier escalamiento a subagente.",
            approval_pause="Pausar antes de abrir subagentes de riesgo, evaluacion o artefactos complejos.",
            steps=[
                {
                    "name": "builder_core",
                    "objective": "Consolidar el blueprint principal y sus exportes base.",
                    "actor": "builder",
                    "outputs": ["core_blueprint"],
                    "fallback": "Permanecer en modo simple si no hay evidencia para escalar.",
                    "requires_approval": False,
                },
                {
                    "name": "specialized_subprocess",
                    "objective": "Ejecutar un subproceso especializado de evaluacion, riesgo o artefactos.",
                    "actor": "specialized_subagent",
                    "outputs": ["specialized_report"],
                    "fallback": "Cerrar el subproceso y volver al blueprint principal.",
                    "requires_approval": True,
                },
                {
                    "name": "implementation_handoff",
                    "objective": "Entregar el paquete consolidado con lineage del subproceso.",
                    "actor": "implementation_owner",
                    "outputs": ["implementation_handoff"],
                    "fallback": "Retornar al builder si el subproceso no agrega evidencia util.",
                    "requires_approval": True,
                },
            ],
        ),
    },
]


DEFAULT_GOVERNANCE_POLICIES = [
    {
        "policy_key": "local_admin_approver",
        "label": "Aprobador local controlado",
        "summary": "Solo el usuario administrador local puede aprobar gates y resolver handoffs gobernados.",
        "scope": "approvals_and_handoffs",
        "is_active": True,
        "policy_payload": {
            "allowed_approver_emails": [get_settings().local_admin_email],
            "protected_actions": ["approval_resolve", "handoff_resolve", "feature_flag_update"],
        },
    },
    {
        "policy_key": "promotion_blockers",
        "label": "Bloqueadores de promotion",
        "summary": "La promotion se bloquea si quedan approvals pendientes, la evaluacion cae o el readiness esta bloqueado.",
        "scope": "promotion",
        "is_active": True,
        "policy_payload": {
            "blocked_when": ["pending_approvals", "evaluation_score_below_70", "blueprint_readiness_blocked"],
        },
    },
    {
        "policy_key": "mandatory_gate_for_side_effects",
        "label": "Gate obligatorio para side effects",
        "summary": "Toda tool con side effects debe declarar approval gate y rationale explicita.",
        "scope": "tools",
        "is_active": True,
        "policy_payload": {
            "required_fields": ["requires_approval", "approval_reason"],
            "enforce_on": ["has_side_effects"],
        },
    },
]


def build_workflow_template_entry(record: WorkflowTemplateRecord) -> WorkflowTemplateEntry:
    return WorkflowTemplateEntry(
        id=record.id,
        template_key=record.template_key,
        label=record.label,
        summary=record.summary,
        architecture_scope=record.architecture_scope,
        supports_approvals=record.supports_approvals,
        supports_handoffs=record.supports_handoffs,
        workflow_profile=WorkflowProfile.model_validate(record.workflow_profile),
        governance_hints=record.governance_hints,
        is_active=record.is_active,
        updated_at=record.updated_at,
    )


def build_handoff_record_entry(record: HandoffRecord) -> HandoffRecordEntry:
    return HandoffRecordEntry(
        id=record.id,
        blueprint_version_number=record.blueprint_version_number,
        handoff_key=record.handoff_key,
        title=record.title,
        from_stage=record.from_stage,
        to_stage=record.to_stage,
        status=record.status,
        owner_role=record.owner_role,
        triggered_by=record.triggered_by,
        summary=record.summary,
        resolution_note=record.resolution_note,
        payload=record.payload,
        created_at=record.created_at,
        updated_at=record.updated_at,
        resolved_at=record.resolved_at,
    )


def build_subagent_run_entry(record: SubagentRunRecord) -> SubagentRunEntry:
    return SubagentRunEntry(
        id=record.id,
        blueprint_version_number=record.blueprint_version_number,
        run_kind=record.run_kind,
        title=record.title,
        status=record.status,
        feature_flag_key=record.feature_flag_key,
        summary=record.summary,
        input_payload=record.input_payload,
        output_payload=record.output_payload,
        created_at=record.created_at,
    )


def seed_workflow_templates(session: Session, *, workspace_id: UUID) -> None:
    existing = {
        item.template_key: item
        for item in session.exec(
            select(WorkflowTemplateRecord).where(WorkflowTemplateRecord.workspace_id == workspace_id)
        ).all()
    }
    for payload in DEFAULT_WORKFLOW_TEMPLATES:
        record = existing.get(payload["template_key"])
        if record is None:
            session.add(WorkflowTemplateRecord(workspace_id=workspace_id, **payload))
            continue
        record.label = str(payload["label"])
        record.summary = str(payload["summary"])
        record.architecture_scope = list(payload["architecture_scope"])
        record.supports_approvals = bool(payload["supports_approvals"])
        record.supports_handoffs = bool(payload["supports_handoffs"])
        record.workflow_profile = dict(payload["workflow_profile"])
        record.governance_hints = list(payload["governance_hints"])
        record.is_active = True
        record.updated_at = utc_now()
        session.add(record)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()


def seed_governance_policies(session: Session, *, workspace_id: UUID) -> None:
    existing = {
        item.policy_key: item
        for item in session.exec(
            select(GovernancePolicyRecord).where(GovernancePolicyRecord.workspace_id == workspace_id)
        ).all()
    }
    for payload in DEFAULT_GOVERNANCE_POLICIES:
        record = existing.get(payload["policy_key"])
        if record is None:
            session.add(GovernancePolicyRecord(workspace_id=workspace_id, **payload))
            continue
        record.label = str(payload["label"])
        record.summary = str(payload["summary"])
        record.scope = str(payload["scope"])
        record.is_active = bool(payload["is_active"])
        record.policy_payload = dict(payload["policy_payload"])
        record.updated_at = utc_now()
        session.add(record)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()


def is_feature_flag_enabled(session: Session, flag_key: str, *, workspace_id: UUID, default_if_missing: bool = False) -> bool:
    record = session.exec(
        select(RuntimeFeatureFlagRecord).where(
            RuntimeFeatureFlagRecord.workspace_id == workspace_id,
            RuntimeFeatureFlagRecord.flag_key == flag_key,
        )
    ).first()
    return bool(record.enabled) if record is not None else bool(default_if_missing)


def update_feature_flag(session: Session, *, workspace_id: UUID, flag_key: str, enabled: bool) -> RuntimeFeatureFlagRecord:
    record = session.exec(
        select(RuntimeFeatureFlagRecord).where(
            RuntimeFeatureFlagRecord.workspace_id == workspace_id,
            RuntimeFeatureFlagRecord.flag_key == flag_key,
        )
    ).first()
    if record is None:
        raise ValueError(f"Unknown feature flag: {flag_key}")
    record.enabled = enabled
    record.updated_at = utc_now()
    session.add(record)
    session.flush()
    return record


def list_workflow_templates(session: Session, *, workspace_id: UUID) -> list[WorkflowTemplateRecord]:
    return session.exec(
        select(WorkflowTemplateRecord)
        .where(WorkflowTemplateRecord.workspace_id == workspace_id, WorkflowTemplateRecord.is_active == True)  # noqa: E712
        .order_by(WorkflowTemplateRecord.label.asc())
    ).all()


def feature_flag_for_subagent_run(run_kind: str) -> str:
    if run_kind == "supervisor_orchestrator":
        return FEATURE_FLAG_MULTI_AGENT_RUNTIME
    return FEATURE_FLAG_SUBAGENTS


def _message_contract_for(topology: Any, from_agent: str, to_agent: str) -> dict[str, Any] | None:
    for contract in topology.message_contracts:
        if contract.from_agent == from_agent and contract.to_agent == to_agent:
            return contract.model_dump(mode="json")
    return None


def _handoff_contract_for(topology: Any, from_agent: str, to_agent: str) -> dict[str, Any] | None:
    for contract in topology.handoff_contracts:
        if contract.from_agent == from_agent and contract.to_agent == to_agent:
            return contract.model_dump(mode="json")
    return None


def _shared_state_contract_for(topology: Any, state_key: str) -> dict[str, Any] | None:
    for contract in topology.shared_state_contracts:
        if contract.state_key == state_key:
            return contract.model_dump(mode="json")
    return None


def _execution_brief_payload(snapshot: SessionSnapshot, topology: Any) -> dict[str, Any]:
    behavior_spec = topology and snapshot.blueprint is not None
    return {
        "goal": snapshot.discovery.desired_outcome if snapshot.discovery is not None else snapshot.session.title,
        "architecture": snapshot.blueprint.architecture if snapshot.blueprint is not None else "",
        "reasoning_pattern": snapshot.blueprint.reasoning_pattern if snapshot.blueprint is not None else "",
        "workflow_template": snapshot.selected_workflow_template_key or "",
        "workflow_states": [
            {
                "name": step.name,
                "actor": step.actor,
                "outputs": list(step.outputs),
                "requires_approval": bool(step.requires_approval),
            }
            for step in snapshot.blueprint.delivery_package.workflow_profile.steps
        ]
        if behavior_spec
        else [],
        "pending_approvals": [approval.gate_key for approval in snapshot.approvals if approval.status == ApprovalStatus.pending],
        "evaluation_cases": [case.case_key for case in snapshot.evaluation_dataset.cases]
        if snapshot.evaluation_dataset is not None
        else [],
        "support_state": topology.support_state,
    }


def _specialist_run_specs(topology: Any) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for contract in topology.agent_contracts:
        if contract.agent_key in SPECIALIST_RUN_KINDS:
            specs.append(
                {
                    "agent_key": contract.agent_key,
                    "run_kind": contract.agent_key,
                    "role": contract.role,
                    "purpose": contract.purpose,
                }
            )
    return specs


def _dispatch_payload(
    *,
    orchestration_id: str,
    agent_key: str,
    purpose: str,
    message_contract: dict[str, Any] | None,
    handoff_contract: dict[str, Any] | None,
    shared_state_namespace: str,
) -> dict[str, Any]:
    return {
        "trace_key": f"{orchestration_id}:{agent_key}:dispatch",
        "phase": "dispatch",
        "agent_key": agent_key,
        "purpose": purpose,
        "message_contract_key": message_contract.get("message_key", "") if message_contract else "",
        "handoff_key": handoff_contract.get("handoff_key", "") if handoff_contract else "",
        "shared_state_namespace": shared_state_namespace,
    }


def _finding_summary(run_kind: str, output_payload: dict[str, Any]) -> list[str]:
    if run_kind == "evaluation_specialist":
        return [
            f"score_reference={output_payload.get('score_reference', 'unknown')}",
            f"recommended_actions={len(output_payload.get('recommended_actions', []))}",
        ]
    if run_kind == "risk_specialist":
        return [
            f"high_risk_tools={len(output_payload.get('high_risk_tools', []))}",
            f"recommended_actions={len(output_payload.get('recommended_actions', []))}",
        ]
    if run_kind == "artifact_specialist":
        return [
            f"missing_deliverables={len(output_payload.get('missing_deliverables', []))}",
            f"recommended_actions={len(output_payload.get('recommended_actions', []))}",
        ]
    return [f"recommended_actions={len(output_payload.get('recommended_actions', []))}"]


def _orchestrate_supervisor_runtime(
    session: Session,
    *,
    session_record: SessionRecord,
    blueprint_version_number: int | None,
    blueprint: BlueprintArtifact,
    approvals: list[ApprovalGateRecord],
    latest_evaluation_score: int | None,
    snapshot: SessionSnapshot,
) -> tuple[ArtifactStatus, dict[str, Any], str]:
    from app.services.canonical_exports import build_construction_pack

    construction_pack = build_construction_pack(snapshot)
    topology = construction_pack.behavior_spec.multi_agent_topology
    benchmark = construction_pack.multi_agent_benchmark
    if topology is None or topology.support_state != "supported":
        raise ValueError("Multi-agent topology is not supported for this blueprint yet")

    orchestration_id = str(uuid4())
    specialist_specs = _specialist_run_specs(topology)
    message_trace: list[dict[str, Any]] = []
    handoff_trace: list[dict[str, Any]] = []
    specialist_runs: list[dict[str, Any]] = []
    specialist_run_ids: list[str] = []
    isolated_namespaces: list[str] = []
    preserved_namespaces: list[str] = []
    execution_brief = _execution_brief_payload(snapshot, topology)
    shared_state = {
        "execution_brief": execution_brief,
        "finding_board": {},
        "final_decision_record": {},
        "contracts": {
            "execution_brief": _shared_state_contract_for(topology, "execution_brief"),
            "finding_board": _shared_state_contract_for(topology, "finding_board"),
            "final_decision_record": _shared_state_contract_for(topology, "final_decision_record"),
        },
    }

    for spec in specialist_specs:
        agent_key = spec["agent_key"]
        shared_state_namespace = f"finding_board.{agent_key}"
        message_contract = _message_contract_for(topology, "supervisor", agent_key)
        completion_contract = _message_contract_for(topology, agent_key, "supervisor") or _message_contract_for(
            topology, "evaluation_specialist", "supervisor"
        )
        handoff_contract = _handoff_contract_for(topology, "supervisor", agent_key)
        dispatch_payload = _dispatch_payload(
            orchestration_id=orchestration_id,
            agent_key=agent_key,
            purpose=spec["purpose"],
            message_contract=message_contract,
            handoff_contract=handoff_contract,
            shared_state_namespace=shared_state_namespace,
        )
        message_trace.append(
            {
                "trace_key": dispatch_payload["trace_key"],
                "status": "sent",
                "from_agent": "supervisor",
                "to_agent": agent_key,
                "contract_key": dispatch_payload["message_contract_key"],
                "purpose": spec["purpose"],
                "payload": {
                    "focus": spec["purpose"],
                    "goal": execution_brief["goal"],
                    "shared_state_namespace": shared_state_namespace,
                },
            }
        )
        handoff_trace.append(
            {
                "trace_key": f"{orchestration_id}:{agent_key}:handoff",
                "status": "dispatched",
                "from_agent": "supervisor",
                "to_agent": agent_key,
                "handoff_key": dispatch_payload["handoff_key"],
                "required_artifacts": list(handoff_contract.get("required_artifacts", [])) if handoff_contract else [],
                "ownership_transfer": handoff_contract.get("ownership_transfer", "") if handoff_contract else "",
            }
        )
        child_run = create_subagent_run(
            session,
            session_record=session_record,
            blueprint_version_number=blueprint_version_number,
            run_kind=spec["run_kind"],
            blueprint=blueprint,
            approvals=approvals,
            latest_evaluation_score=latest_evaluation_score,
        )
        child_output = dict(child_run.output_payload)
        child_input = dict(child_run.input_payload)
        child_input.update(
            {
                "orchestration_id": orchestration_id,
                "parent_runtime": "supervisor_orchestrator",
                "assigned_agent_key": agent_key,
                "message_contract_key": dispatch_payload["message_contract_key"],
                "handoff_key": dispatch_payload["handoff_key"],
                "shared_state_namespace": shared_state_namespace,
            }
        )
        finding_entry = {
            "agent_key": agent_key,
            "run_id": str(child_run.id),
            "status": child_run.status.value,
            "summary": child_run.summary,
            "focus": child_output.get("focus", ""),
            "recommended_actions": list(child_output.get("recommended_actions", [])),
            "signals": _finding_summary(spec["run_kind"], child_output),
        }
        shared_state["finding_board"][agent_key] = finding_entry
        if child_run.status == ArtifactStatus.ready:
            preserved_namespaces.append(shared_state_namespace)
        else:
            isolated_namespaces.append(shared_state_namespace)
            preserved_namespaces.extend(
                item
                for item in [f"finding_board.{other}" for other in shared_state["finding_board"].keys() if other != agent_key]
                if item not in preserved_namespaces
            )
        completion_trace_key = f"{orchestration_id}:{agent_key}:complete"
        message_trace.append(
            {
                "trace_key": completion_trace_key,
                "status": "received",
                "from_agent": agent_key,
                "to_agent": "supervisor",
                "contract_key": completion_contract.get("message_key", "") if completion_contract else "",
                "purpose": "Return specialist findings for merge",
                "payload": finding_entry,
            }
        )
        handoff_trace.append(
            {
                "trace_key": f"{orchestration_id}:{agent_key}:return",
                "status": "completed" if child_run.status == ArtifactStatus.ready else "isolated",
                "from_agent": agent_key,
                "to_agent": "supervisor",
                "handoff_key": dispatch_payload["handoff_key"],
                "required_artifacts": list(handoff_contract.get("required_artifacts", [])) if handoff_contract else [],
                "ownership_transfer": "El supervisor recupera el ownership tras recibir findings.",
            }
        )
        child_output.update(
            {
                "orchestration_id": orchestration_id,
                "orchestrated_by": "supervisor_orchestrator",
                "dispatch_trace_key": dispatch_payload["trace_key"],
                "completion_trace_key": completion_trace_key,
                "shared_state_namespace": shared_state_namespace,
                "finding_entry": finding_entry,
            }
        )
        child_run.input_payload = child_input
        child_run.output_payload = child_output
        session.add(child_run)
        session.flush()
        specialist_run_ids.append(str(child_run.id))
        specialist_runs.append(
            {
                "run_id": str(child_run.id),
                "run_kind": spec["run_kind"],
                "agent_key": agent_key,
                "status": child_run.status.value,
                "summary": child_run.summary,
                "shared_state_namespace": shared_state_namespace,
            }
        )

    blocking_specialists = [
        item["agent_key"] for item in specialist_runs if item["status"] != ArtifactStatus.ready
    ]
    merge_ready = benchmark is not None and benchmark.go_decision == "go" and not blocking_specialists
    final_status = ArtifactStatus.ready if merge_ready else ArtifactStatus.needs_review
    merge_summary = (
        "El supervisor consolido findings sin ramas bloqueadas y el benchmark multiagente justifica el cierre."
        if merge_ready
        else "El supervisor consolido findings, pero existen ramas aisladas o condiciones que mantienen el runtime en needs_review."
    )
    final_decision_record = {
        "status": final_status.value,
        "benchmark_go_decision": benchmark.go_decision if benchmark is not None else "hold",
        "blocking_specialists": blocking_specialists,
        "merge_summary": merge_summary,
        "recommended_actions": (
            ["Continuar con promotion o medicion runtime del flujo multiagente."]
            if merge_ready
            else ["Resolver los findings bloqueantes antes de promover el flujo multiagente."]
        ),
    }
    shared_state["final_decision_record"] = final_decision_record
    failure_isolation_result = {
        "passed": bool(shared_state["finding_board"]) and len(shared_state["finding_board"]) == len(specialist_specs),
        "isolated_namespaces": isolated_namespaces,
        "preserved_namespaces": preserved_namespaces,
        "notes": (
            ["Ningun especialista sobreescribio namespaces ajenos durante la corrida."]
            if not isolated_namespaces
            else ["Las ramas con findings no listos quedaron aisladas sin borrar findings previas del resto."]
        ),
    }
    runtime_metrics = {
        "orchestration_id": orchestration_id,
        "active_specialist_count": len(specialist_runs),
        "message_count": len(message_trace),
        "handoff_count": len(handoff_trace),
        "isolated_branch_count": len(isolated_namespaces),
        "preserved_branch_count": len(preserved_namespaces),
        "baseline_go_decision": benchmark.go_decision if benchmark is not None else "hold",
        "quality_delta": next(
            (metric.improvement_delta for metric in benchmark.metrics if metric.metric_key == "quality_coverage"),
            0,
        )
        if benchmark is not None
        else 0,
        "latency_projection_seconds": next(
            (metric.projected_multi_agent for metric in benchmark.metrics if metric.metric_key == "latency_budget"),
            0,
        )
        if benchmark is not None
        else 0,
        "coordination_cost_projection": next(
            (metric.projected_multi_agent for metric in benchmark.metrics if metric.metric_key == "coordination_cost"),
            0,
        )
        if benchmark is not None
        else 0,
    }
    output_payload = {
        "classification": "multi_agent_orchestration_run",
        "focus": "Ejecutar orquestacion supervisor + especialistas con contratos aislados y merge trazable.",
        "runtime_pattern": topology.runtime_pattern,
        "support_state": topology.support_state,
        "benchmark": benchmark.model_dump(mode="json") if benchmark is not None else {},
        "agent_contracts": [item.model_dump(mode="json") for item in topology.agent_contracts],
        "message_contracts": [item.model_dump(mode="json") for item in topology.message_contracts],
        "handoff_contracts": [item.model_dump(mode="json") for item in topology.handoff_contracts],
        "shared_state_contracts": [item.model_dump(mode="json") for item in topology.shared_state_contracts],
        "failure_isolation_rules": list(topology.failure_isolation_rules),
        "orchestration_id": orchestration_id,
        "specialist_run_ids": specialist_run_ids,
        "specialist_runs": specialist_runs,
        "message_trace": message_trace,
        "handoff_trace": handoff_trace,
        "shared_state": shared_state,
        "merge_summary": merge_summary,
        "failure_isolation_result": failure_isolation_result,
        "runtime_metrics": runtime_metrics,
        "recommended_actions": final_decision_record["recommended_actions"],
    }
    return final_status, output_payload, "Orquestacion supervisor + especialistas"


def recommend_workflow_template_key(blueprint: BlueprintArtifact | None) -> str:
    if blueprint is None:
        return "single_agent_linear"
    if blueprint.architecture in {"supervisor_with_subagents", "router_parallel"}:
        return "subagent_escalation_workflow"
    if blueprint.architecture == "handoffs":
        return "handoff_governed_workflow"
    if any(tool.requires_approval or tool.has_side_effects for tool in blueprint.tools):
        return "approval_gate_workflow"
    return "single_agent_linear"


def render_workflow_markdown(profile: WorkflowProfile) -> str:
    lines = [
        f"# {profile.execution_pattern or 'Workflow profile'}",
        "",
        f"- inbox_strategy: {profile.inbox_strategy}",
        f"- outbox_strategy: {profile.outbox_strategy}",
        f"- checkpoint_policy: {profile.checkpoint_policy}",
        f"- retry_strategy: {profile.retry_strategy}",
        f"- compensation_strategy: {profile.compensation_strategy}",
        f"- approval_pause: {profile.approval_pause}",
        f"- timeout_policy: {profile.timeout_policy}",
        "",
        "## Steps",
    ]
    for step in profile.steps:
        lines.extend(
            [
                f"- {step.name}",
                f"  - actor: {step.actor}",
                f"  - objective: {step.objective}",
                f"  - outputs: {', '.join(step.outputs)}",
                f"  - fallback: {step.fallback}",
                f"  - requires_approval: {step.requires_approval}",
            ]
        )
    return "\n".join(lines)


def apply_workflow_template(blueprint: BlueprintArtifact, template: WorkflowTemplateRecord) -> BlueprintArtifact:
    updated_profile = WorkflowProfile.model_validate(template.workflow_profile)
    updated_deliverables: list[GeneratedDeliverable] = []
    replaced_state_flow = False
    for item in blueprint.delivery_package.deliverables:
        if item.key != "state_flow":
            updated_deliverables.append(item)
            continue
        updated_deliverables.append(
            item.model_copy(
                update={
                    "summary": f"{item.summary} Plantilla aplicada: {template.label}.",
                    "content_markdown": render_workflow_markdown(updated_profile),
                }
            )
        )
        replaced_state_flow = True
    if not replaced_state_flow:
        updated_deliverables.append(
            GeneratedDeliverable(
                key="state_flow",
                title="State flow",
                summary=f"Workflow durable derivado de la plantilla {template.label}.",
                content_markdown=render_workflow_markdown(updated_profile),
            )
        )

    updated_delivery_package = blueprint.delivery_package.model_copy(
        update={
            "workflow_profile": updated_profile,
            "deliverables": updated_deliverables,
            "decision_summary": (
                f"{blueprint.delivery_package.decision_summary}\nPlantilla de workflow aplicada: {template.label}."
            ).strip(),
        }
    )
    return blueprint.model_copy(update={"delivery_package": updated_delivery_package})


def upsert_handoff_record(
    session: Session,
    *,
    session_id: UUID,
    handoff_key: str,
    blueprint_version_number: int | None,
    title: str,
    from_stage: SessionStage,
    to_stage: SessionStage,
    status: str,
    owner_role: str,
    triggered_by: str,
    summary: str,
    payload: dict[str, Any],
    resolution_note: str = "",
) -> HandoffRecord:
    record = session.exec(
        select(HandoffRecord).where(
            HandoffRecord.session_id == session_id,
            HandoffRecord.handoff_key == handoff_key,
            HandoffRecord.status == HANDOFF_STATUS_PENDING,
        )
    ).first()
    timestamp = utc_now()
    if record is None:
        record = HandoffRecord(
            session_id=session_id,
            blueprint_version_number=blueprint_version_number,
            handoff_key=handoff_key,
            title=title,
            from_stage=from_stage,
            to_stage=to_stage,
            status=status,
            owner_role=owner_role,
            triggered_by=triggered_by,
            summary=summary,
            resolution_note=resolution_note,
            payload=payload,
        )
    else:
        record.blueprint_version_number = blueprint_version_number
        record.title = title
        record.from_stage = from_stage
        record.to_stage = to_stage
        record.status = status
        record.owner_role = owner_role
        record.triggered_by = triggered_by
        record.summary = summary
        record.resolution_note = resolution_note
        record.payload = payload
        record.updated_at = timestamp
        if status in {HANDOFF_STATUS_COMPLETED, HANDOFF_STATUS_RETURNED}:
            record.resolved_at = timestamp
    session.add(record)
    session.flush()
    return record


def resolve_handoff_record(
    session: Session,
    *,
    handoff_record: HandoffRecord,
    decision: str,
    resolution_note: str,
) -> HandoffRecord:
    if decision not in {HANDOFF_STATUS_COMPLETED, HANDOFF_STATUS_RETURNED}:
        raise ValueError("Unsupported handoff decision")
    handoff_record.status = decision
    handoff_record.resolution_note = resolution_note
    handoff_record.updated_at = utc_now()
    handoff_record.resolved_at = handoff_record.updated_at
    session.add(handoff_record)
    session.flush()
    return handoff_record


def sync_governance_handoff(
    session: Session,
    *,
    session_record: SessionRecord,
    blueprint_version_number: int | None,
    blueprint: BlueprintArtifact,
    source_action: str,
    pending_approvals: int,
) -> HandoffRecord:
    return upsert_handoff_record(
        session,
        session_id=session_record.id,
        handoff_key="governance_review",
        blueprint_version_number=blueprint_version_number,
        title="Revision de gobierno del blueprint",
        from_stage=SessionStage.build_blueprint,
        to_stage=SessionStage.post_validation,
        status=HANDOFF_STATUS_PENDING if pending_approvals > 0 else HANDOFF_STATUS_COMPLETED,
        owner_role="local_admin",
        triggered_by=source_action,
        summary=(
            "El blueprint requiere gate humano antes de promotion."
            if pending_approvals > 0
            else "El blueprint quedo listo para avanzar sin gates pendientes."
        ),
        payload={
            "architecture": blueprint.architecture,
            "reasoning_pattern": blueprint.reasoning_pattern,
            "pending_approvals": pending_approvals,
        },
        resolution_note="" if pending_approvals > 0 else "Sin gates pendientes en esta revision.",
    )


def sync_evaluation_handoff(
    session: Session,
    *,
    session_record: SessionRecord,
    blueprint_version_number: int | None,
    source_action: str,
    overall_score: int,
    status: ArtifactStatus,
) -> HandoffRecord:
    return upsert_handoff_record(
        session,
        session_id=session_record.id,
        handoff_key="evaluation_review",
        blueprint_version_number=blueprint_version_number,
        title="Revision de evaluacion antes de promotion",
        from_stage=SessionStage.post_validation,
        to_stage=SessionStage.ready_for_export,
        status=HANDOFF_STATUS_PENDING if status != ArtifactStatus.ready or overall_score < 70 else HANDOFF_STATUS_COMPLETED,
        owner_role="quality_reviewer",
        triggered_by=source_action,
        summary=(
            "La evaluacion quedo por debajo del umbral y pide retorno controlado."
            if status != ArtifactStatus.ready or overall_score < 70
            else "La evaluacion quedo lista para promotion."
        ),
        payload={"overall_score": overall_score, "evaluation_status": status},
        resolution_note="" if status != ArtifactStatus.ready or overall_score < 70 else "Evaluacion aprobada.",
    )


def create_export_handoff(
    session: Session,
    *,
    session_record: SessionRecord,
    blueprint_version_number: int | None,
    source_action: str,
    artifact_key: str,
) -> HandoffRecord:
    return upsert_handoff_record(
        session,
        session_id=session_record.id,
        handoff_key=f"implementation_handoff:{artifact_key}",
        blueprint_version_number=blueprint_version_number,
        title="Handoff tecnico a implementacion",
        from_stage=SessionStage.ready_for_export,
        to_stage=SessionStage.ready_for_export,
        status=HANDOFF_STATUS_COMPLETED,
        owner_role="implementation_owner",
        triggered_by=source_action,
        summary="Se entrego un export gobernado para consumo tecnico.",
        payload={"artifact_key": artifact_key},
        resolution_note="Export listo y trazado.",
    )


def evaluate_governance_policies(
    session: Session,
    *,
    session_record: SessionRecord,
    blueprint: BlueprintArtifact | None,
    approvals: list[ApprovalGateRecord],
    latest_evaluation_score: int | None,
    latest_evaluation_status: str,
) -> list[GovernancePolicyEntry]:
    policies = session.exec(
        select(GovernancePolicyRecord)
        .where(GovernancePolicyRecord.workspace_id == session_record.workspace_id)
        .order_by(GovernancePolicyRecord.label.asc())
    ).all()
    settings = get_settings()
    entries: list[GovernancePolicyEntry] = []
    for policy in policies:
        compliance_status = "compliant"
        evidence: list[str] = []
        if policy.policy_key == "local_admin_approver":
            allowed = list(policy.policy_payload.get("allowed_approver_emails", []))
            compliance_status = "compliant" if settings.local_admin_email in allowed else "violated"
            evidence.append(f"allowed_approver={settings.local_admin_email}")
        elif policy.policy_key == "promotion_blockers":
            pending_approvals = sum(1 for item in approvals if item.status == ApprovalStatus.pending)
            if pending_approvals > 0:
                compliance_status = "blocked"
                evidence.append(f"pending_approvals={pending_approvals}")
            elif latest_evaluation_score is not None and (
                latest_evaluation_score < 70 or latest_evaluation_status == ArtifactStatus.failed
            ):
                compliance_status = "warning"
                evidence.append(f"latest_evaluation_score={latest_evaluation_score}")
            elif blueprint is not None and blueprint.readiness_state == ReviewState.blocked:
                compliance_status = "blocked"
                evidence.append("blueprint_readiness=blocked")
            else:
                evidence.append("promotion_ready=true")
        elif policy.policy_key == "mandatory_gate_for_side_effects":
            if blueprint is None:
                compliance_status = "unknown"
                evidence.append("blueprint_missing=true")
            else:
                offenders = [
                    tool.name
                    for tool in blueprint.tools
                    if tool.has_side_effects and (not tool.requires_approval or not tool.approval_reason)
                ]
                compliance_status = "violated" if offenders else "compliant"
                evidence.append(f"side_effect_offenders={','.join(offenders) if offenders else 'none'}")
        entries.append(
            GovernancePolicyEntry(
                id=policy.id,
                policy_key=policy.policy_key,
                label=policy.label,
                summary=policy.summary,
                scope=policy.scope,
                is_active=policy.is_active,
                policy_payload=policy.policy_payload,
                compliance_status=compliance_status,
                evidence=evidence,
                updated_at=policy.updated_at,
            )
        )
    return entries


def ensure_local_admin_can_govern(current_user_email: str) -> None:
    if current_user_email != get_settings().local_admin_email:
        raise PermissionError("Solo el admin local puede ejecutar esta accion de gobierno")


def create_subagent_run(
    session: Session,
    *,
    session_record: SessionRecord,
    blueprint_version_number: int | None,
    run_kind: str,
    blueprint: BlueprintArtifact,
    approvals: list[ApprovalGateRecord],
    latest_evaluation_score: int | None,
    snapshot: SessionSnapshot | None = None,
) -> SubagentRunRecord:
    pending_approvals = sum(1 for item in approvals if item.status == ApprovalStatus.pending)
    input_payload = {
        "architecture": blueprint.architecture,
        "reasoning_pattern": blueprint.reasoning_pattern,
        "blueprint_version_number": blueprint_version_number,
        "pending_approvals": pending_approvals,
        "latest_evaluation_score": latest_evaluation_score,
        "deliverable_count": len(blueprint.delivery_package.deliverables),
    }
    if run_kind == "evaluation_specialist":
        status = ArtifactStatus.ready if (latest_evaluation_score or 0) >= 70 else ArtifactStatus.needs_review
        output_payload = {
            "classification": "specialist_review_run",
            "focus": "Validar gaps y bloqueos antes de promotion.",
            "recommended_actions": (
                ["Avanzar a export controlado."]
                if status == ArtifactStatus.ready
                else ["Reforzar dataset, rubrica y corridas antes del handoff final."]
            ),
            "score_reference": latest_evaluation_score,
        }
        title = "Revision especializada de evaluacion"
    elif run_kind == "risk_specialist":
        high_risk_tools = [
            tool.name for tool in blueprint.tools if tool.risk_level.lower() in {"high", "critical"} or tool.has_side_effects
        ]
        status = ArtifactStatus.needs_review if high_risk_tools else ArtifactStatus.ready
        output_payload = {
            "classification": "specialist_review_run",
            "focus": "Analizar side effects, riesgo y gates obligatorios.",
            "high_risk_tools": high_risk_tools,
            "recommended_actions": (
                ["Cerrar approvals y documentar compensaciones."]
                if high_risk_tools
                else ["Mantener controles actuales y continuar."]
            ),
        }
        title = "Revision especializada de riesgo"
    elif run_kind == "artifact_specialist":
        missing_deliverables = [
            key for key in ["prd", "technical_spec", "system_prompt"] if key not in {item.key for item in blueprint.delivery_package.deliverables}
        ]
        status = ArtifactStatus.needs_review if missing_deliverables else ArtifactStatus.ready
        output_payload = {
            "classification": "specialist_review_run",
            "focus": "Revisar consistencia del paquete y artefactos de implementacion.",
            "missing_deliverables": missing_deliverables,
            "recommended_actions": (
                ["Completar artefactos base antes del export final."]
                if missing_deliverables
                else ["El paquete principal ya cubre los artefactos criticos."]
            ),
        }
        title = "Revision especializada de artefactos"
    elif run_kind == "supervisor_orchestrator":
        if snapshot is None:
            raise ValueError("Session snapshot is required for supervisor_orchestrator")
        if blueprint.architecture != "supervisor_with_subagents":
            raise ValueError("The blueprint architecture must be supervisor_with_subagents before running the orchestrator")
        status, output_payload, title = _orchestrate_supervisor_runtime(
            session,
            session_record=session_record,
            blueprint_version_number=blueprint_version_number,
            blueprint=blueprint,
            approvals=approvals,
            latest_evaluation_score=latest_evaluation_score,
            snapshot=snapshot,
        )
    else:
        raise ValueError("Unsupported subagent kind")

    record = SubagentRunRecord(
        session_id=session_record.id,
        blueprint_version_number=blueprint_version_number,
        run_kind=run_kind,
        title=title,
        status=status,
        feature_flag_key=feature_flag_for_subagent_run(run_kind),
        summary=str(output_payload.get("focus", "")),
        input_payload=input_payload,
        output_payload=output_payload,
    )
    session.add(record)
    session.flush()
    return record
