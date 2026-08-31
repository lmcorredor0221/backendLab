from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlmodel import Session, select

from app.models import (
    ACPBuildRunRecord,
    ACPPhaseRunRecord,
    ACPWorkflowRunStatus,
    CommercialAccessRequestRecord,
    CommercialAccessRequestStatus,
    CommercialTier,
    JourneyStateRecord,
    JourneyStateTransitionRecord,
    SessionRecord,
    UserRecord,
    utc_now,
)
from app.services.commercial_access import build_commercial_access_snapshot_v2
from app.services.product_processing.contracts import (
    JourneyStateKey,
    JourneyStateMachine,
    JourneyStateMachineStage,
    JourneyStateSubstate,
    JourneyStateTransition,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductJourneyOverview,
    ProductJourneyProductSummary,
)


ACTIVE_LIFECYCLES = {
    ProductBuildLifecycle.queued,
    ProductBuildLifecycle.preparing,
    ProductBuildLifecycle.running,
}

WORK_STAGE_KEYS = {
    JourneyStateKey.discover,
    JourneyStateKey.define,
    JourneyStateKey.design,
    JourneyStateKey.tools,
    JourneyStateKey.memory,
    JourneyStateKey.estimate,
}

STATE_LABELS: dict[JourneyStateKey, str] = {
    JourneyStateKey.discover: "Descubrir",
    JourneyStateKey.define: "Definir",
    JourneyStateKey.design: "Disenar",
    JourneyStateKey.tools: "Herramientas",
    JourneyStateKey.memory: "Memoria",
    JourneyStateKey.estimate: "Blueprint Free",
    JourneyStateKey.blueprint_free_ready: "Blueprint Free listo",
    JourneyStateKey.blueprint_pro_access_requested: "Solicitud Blueprint Pro",
    JourneyStateKey.blueprint_pro_access_pending: "Activacion Blueprint Pro",
    JourneyStateKey.blueprint_pro_active: "Blueprint Pro",
    JourneyStateKey.acp_access_requested: "Solicitud ACP",
    JourneyStateKey.acp_access_pending: "Activacion ACP",
    JourneyStateKey.acp_prep: "ACP",
    JourneyStateKey.validate: "Validar",
    JourneyStateKey.package: "Package",
    JourneyStateKey.completed: "Completado",
}

STATE_DETAILS: dict[JourneyStateKey, str] = {
    JourneyStateKey.discover: "Captura y entendimiento inicial del problema.",
    JourneyStateKey.define: "Objetivos, alcance y requerimientos en consolidacion.",
    JourneyStateKey.design: "Arquitectura y comportamiento del agente en definicion.",
    JourneyStateKey.tools: "Capacidades, contratos e integraciones en definicion.",
    JourneyStateKey.memory: "Memoria, conocimiento y contexto en definicion.",
    JourneyStateKey.estimate: "Blueprint Free en preparacion antes de pasar al producto premium.",
    JourneyStateKey.blueprint_free_ready: "Blueprint Free disponible para revisar antes de solicitar Blueprint Pro.",
    JourneyStateKey.blueprint_pro_access_requested: "La solicitud de Blueprint Pro ya fue creada y espera aprobacion.",
    JourneyStateKey.blueprint_pro_access_pending: "Blueprint Pro tiene una activacion comercial pendiente antes de habilitarse.",
    JourneyStateKey.blueprint_pro_active: "Blueprint Pro esta habilitado para enriquecer, generar y descargar entregables premium.",
    JourneyStateKey.acp_access_requested: "La solicitud de ACP ya fue creada y espera aprobacion.",
    JourneyStateKey.acp_access_pending: "ACP tiene una activacion comercial pendiente antes de habilitarse.",
    JourneyStateKey.acp_prep: "ACP esta habilitado y listo para preparar respuestas, gaps, decisiones y dependencias.",
    JourneyStateKey.validate: "El ACP se encuentra en validacion funcional, tecnica y de gobernanza.",
    JourneyStateKey.package: "El ACP se encuentra en empaquetado final y export.",
    JourneyStateKey.completed: "El paquete final premium se encuentra completo y listo para su descarga.",
}


def build_journey_state_machine(
    db: Session,
    *,
    record: SessionRecord,
    overview: ProductJourneyOverview,
    current_user: UserRecord | None = None,
) -> JourneyStateMachine:
    persisted = load_persisted_journey_state_machine(db, record=record)
    if persisted is not None:
        return persisted

    access = build_commercial_access_snapshot_v2(db, record, current_user=current_user)
    pending_requests = _pending_request_products(db, record=record)
    products_by_key = {product.product_key: product for product in overview.products}
    current_state_key = _resolve_current_state_key(
        record=record,
        overview=overview,
        access_tier=access.tier,
        checkout_state=access.checkout_state,
        pending_requests=pending_requests,
        products_by_key=products_by_key,
    )
    current = _build_stage(
        record=record,
        overview=overview,
        products_by_key=products_by_key,
        state_key=current_state_key,
    )
    source_contracts = sorted(
        {
            "journey-state-machine.v1",
            "commercial-access.v2",
            "commercial-access-requests.v1",
            *overview.source_contracts,
        }
    )
    return JourneyStateMachine(
        workspace_id=record.workspace_id,
        session_id=record.id,
        current=current,
        state_source="legacy_projection",
        source_contracts=source_contracts,
    )


def initialize_journey_state(
    db: Session,
    *,
    record: SessionRecord,
    state_key: JourneyStateKey = JourneyStateKey.discover,
    substate: JourneyStateSubstate = JourneyStateSubstate.idle,
    actor_type: str = "system",
    actor_user_id: UUID | None = None,
    reason: str = "",
    correlation_id: str = "",
    metadata: dict[str, Any] | None = None,
    progress_percent: int | None = None,
    blocking: bool = False,
) -> JourneyStateMachine:
    """Create the canonical snapshot once; repeat calls are intentionally no-ops."""
    existing = _load_current_state(db, session_id=record.id)
    if existing is not None:
        return _serialize_persisted_state(db, record=record, current=existing)

    now = utc_now()
    stage = _stage_for_persisted_values(
        record=record,
        state_key=state_key,
        substate=substate,
        progress_percent=progress_percent,
        blocking=blocking,
    )
    current = JourneyStateRecord(
        workspace_id=record.workspace_id,
        session_id=record.id,
        state_key=stage.state_key.value,
        substate=stage.substate.value,
        product_key=stage.product_key.value,
        stage_key=stage.stage_key,
        progress_percent=stage.progress_percent,
        blocking=stage.blocking,
        revision=1,
        source_contracts=["journey-state-machine.v1", "journey-state-persistence.v1"],
        state_payload=_stage_payload(stage),
        last_transition_at=now,
        created_at=now,
        updated_at=now,
    )
    db.add(current)
    db.flush()
    db.add(
        JourneyStateTransitionRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            sequence=1,
            event_key="journey_initialized",
            to_state_key=stage.state_key.value,
            to_substate=stage.substate.value,
            actor_type=actor_type,
            actor_user_id=actor_user_id,
            reason=reason or "Estado canonico inicializado.",
            correlation_id=correlation_id or f"journey-init:{record.id}",
            transition_payload=_stage_payload(stage, metadata=metadata),
            occurred_at=now,
        )
    )
    db.flush()
    return _serialize_persisted_state(db, record=record, current=current)


def transition_journey_state(
    db: Session,
    *,
    record: SessionRecord,
    event_key: str,
    target_state_key: JourneyStateKey,
    target_substate: JourneyStateSubstate,
    actor_type: str = "system",
    actor_user_id: UUID | None = None,
    reason: str = "",
    correlation_id: str = "",
    metadata: dict[str, Any] | None = None,
    initial_state_key: JourneyStateKey | None = None,
    initial_substate: JourneyStateSubstate = JourneyStateSubstate.idle,
    progress_percent: int | None = None,
    blocking: bool = False,
) -> JourneyStateMachine:
    """Append an idempotent transition and update the one canonical current-state row."""
    normalized_correlation = correlation_id.strip() or f"{event_key}:{uuid4()}"
    existing_transition = db.exec(
        select(JourneyStateTransitionRecord).where(
            JourneyStateTransitionRecord.session_id == record.id,
            JourneyStateTransitionRecord.correlation_id == normalized_correlation,
        )
    ).first()
    current = _load_current_state(db, session_id=record.id)
    if existing_transition is not None and current is not None:
        return _serialize_persisted_state(db, record=record, current=current)

    if current is None:
        initialize_journey_state(
            db,
            record=record,
            state_key=initial_state_key or target_state_key,
            substate=initial_substate,
            actor_type="migration",
            actor_user_id=actor_user_id,
            reason="Estado legacy proyectado antes de registrar una transicion canonica.",
            correlation_id=f"legacy-import:{record.id}",
        )
        current = _load_current_state(db, session_id=record.id)
        if current is None:  # pragma: no cover - defensive guard for persistence failures.
            raise RuntimeError("No se pudo inicializar el estado canonico del journey.")

    now = utc_now()
    previous_state_key = current.state_key
    previous_substate = current.substate
    stage = _stage_for_persisted_values(
        record=record,
        state_key=target_state_key,
        substate=target_substate,
        progress_percent=progress_percent,
        blocking=blocking,
    )
    next_revision = current.revision + 1
    current.state_key = stage.state_key.value
    current.substate = stage.substate.value
    current.product_key = stage.product_key.value
    current.stage_key = stage.stage_key
    current.progress_percent = stage.progress_percent
    current.blocking = stage.blocking
    current.revision = next_revision
    current.state_payload = _stage_payload(stage)
    current.last_transition_at = now
    current.updated_at = now
    db.add(current)
    db.add(
        JourneyStateTransitionRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            sequence=next_revision,
            event_key=event_key,
            from_state_key=previous_state_key,
            from_substate=previous_substate,
            to_state_key=stage.state_key.value,
            to_substate=stage.substate.value,
            actor_type=actor_type,
            actor_user_id=actor_user_id,
            reason=reason,
            correlation_id=normalized_correlation,
            transition_payload=_stage_payload(stage, metadata=metadata),
            occurred_at=now,
        )
    )
    db.flush()
    return _serialize_persisted_state(db, record=record, current=current)


def load_persisted_journey_state_machine(
    db: Session,
    *,
    record: SessionRecord,
) -> JourneyStateMachine | None:
    current = _load_current_state(db, session_id=record.id)
    if current is None:
        return None
    return _serialize_persisted_state(db, record=record, current=current)


def transition_for_commercial_access_request(
    db: Session,
    *,
    record: SessionRecord,
    request: CommercialAccessRequestRecord,
    event_key: str,
    actor_user_id: UUID | None,
    reason: str = "",
) -> JourneyStateMachine:
    product_key = str(request.product_key or "").strip()
    is_acp = product_key == "acp"
    target_state = _commercial_target_state(product_key=product_key, event_key=event_key)
    target_substate = (
        JourneyStateSubstate.waiting_dependency
        if event_key.startswith("request_")
        else JourneyStateSubstate.idle
    )
    if event_key.startswith("deny_"):
        target_substate = JourneyStateSubstate.idle
    return transition_journey_state(
        db,
        record=record,
        event_key=event_key,
        target_state_key=target_state,
        target_substate=target_substate,
        actor_type="user" if actor_user_id is not None else "system",
        actor_user_id=actor_user_id,
        reason=reason,
        correlation_id=f"access-request:{request.id}:{event_key}",
        metadata={"access_request_id": str(request.id), "product_key": product_key, "status": request.status.value},
        initial_state_key=JourneyStateKey.blueprint_pro_active if is_acp else JourneyStateKey.blueprint_free_ready,
    )


def transition_for_stage_approval(
    db: Session,
    *,
    record: SessionRecord,
    approved_stage_key: str,
    actor_user_id: UUID | None,
    correlation_id: str,
) -> JourneyStateMachine:
    """Record the next actionable journey state after an approved work-stage proposal."""
    normalized_stage = _normalize_stage_key(approved_stage_key)
    target_state = _state_after_stage_approval(normalized_stage)
    target_substate = (
        JourneyStateSubstate.completed
        if target_state == JourneyStateKey.blueprint_free_ready
        else JourneyStateSubstate.idle
    )
    return transition_journey_state(
        db,
        record=record,
        event_key=f"stage_{normalized_stage.value}_approved",
        target_state_key=target_state,
        target_substate=target_substate,
        actor_type="user" if actor_user_id is not None else "system",
        actor_user_id=actor_user_id,
        reason=f"Etapa {normalized_stage.value} aprobada; el flujo continua en {target_state.value}.",
        correlation_id=correlation_id,
        metadata={"approved_stage_key": normalized_stage.value},
        initial_state_key=normalized_stage,
    )


def transition_for_acp_workspace_phase(
    db: Session,
    *,
    record: SessionRecord,
    run: ACPBuildRunRecord,
    phase: ACPPhaseRunRecord,
    actor_user_id: UUID | None,
) -> JourneyStateMachine:
    """Project the durable ACP phase result into the cross-product journey state."""
    phase_status = phase.status
    target_state = _state_for_acp_phase(phase.phase_key)
    target_substate = _substate_for_acp_phase_status(phase_status)
    return transition_journey_state(
        db,
        record=record,
        event_key=f"acp_phase_{phase.phase_key}_{phase_status.value}",
        target_state_key=target_state,
        target_substate=target_substate,
        actor_type="user" if actor_user_id is not None else "system",
        actor_user_id=actor_user_id,
        reason=f"Subfase ACP {phase.phase_key} finalizo con estado {phase_status.value}.",
        correlation_id=f"acp-phase:{run.id}:{phase.phase_key}:{phase.attempt_count}",
        metadata={
            "acp_run_id": str(run.id),
            "phase_run_id": str(phase.id),
            "phase_key": phase.phase_key,
            "phase_status": phase_status.value,
            "attempt_count": phase.attempt_count,
        },
        initial_state_key=JourneyStateKey.acp_prep,
        progress_percent=run.progress_percent,
        blocking=phase_status == ACPWorkflowRunStatus.blocked,
    )


def transition_for_export_ready(
    db: Session,
    *,
    record: SessionRecord,
    export_job_id: UUID,
    product_key: str,
    artifact_kind: str,
    actor_user_id: UUID | None,
) -> JourneyStateMachine | None:
    """Only final downloadable packages advance the cross-product journey."""
    if product_key == "blueprint_pro" and artifact_kind == "blueprint_professional":
        target_state = JourneyStateKey.blueprint_pro_active
        initial_state = JourneyStateKey.blueprint_pro_active
    elif product_key == "acp" and artifact_kind == "acp_portable_zip":
        target_state = JourneyStateKey.completed
        initial_state = JourneyStateKey.package
    else:
        return None
    return transition_journey_state(
        db,
        record=record,
        event_key=f"{product_key}_export_ready",
        target_state_key=target_state,
        target_substate=JourneyStateSubstate.completed,
        actor_type="user" if actor_user_id is not None else "system",
        actor_user_id=actor_user_id,
        reason=f"Paquete descargable {artifact_kind} disponible.",
        correlation_id=f"export-ready:{export_job_id}",
        metadata={"export_job_id": str(export_job_id), "product_key": product_key, "artifact_kind": artifact_kind},
        initial_state_key=initial_state,
        progress_percent=100,
    )


def transition_for_paid_product_activation(
    db: Session,
    *,
    record: SessionRecord,
    order_id: UUID,
    product_key: str,
    actor_user_id: UUID | None,
    source: str,
) -> JourneyStateMachine | None:
    """Project one paid entitlement activation, regardless of its payment provider."""
    if product_key == "blueprint_pro":
        target_state = JourneyStateKey.blueprint_pro_active
        initial_state = JourneyStateKey.blueprint_free_ready
    elif product_key == "acp":
        target_state = JourneyStateKey.acp_prep
        initial_state = JourneyStateKey.blueprint_pro_active
    else:
        return None
    return transition_journey_state(
        db,
        record=record,
        event_key=f"paid_{product_key}_activated",
        target_state_key=target_state,
        target_substate=JourneyStateSubstate.idle,
        actor_type="user" if actor_user_id is not None else "system",
        actor_user_id=actor_user_id,
        reason=f"Acceso {product_key} activado por orden pagada ({source}).",
        correlation_id=f"paid-order:{order_id}:{product_key}",
        metadata={"order_id": str(order_id), "product_key": product_key, "source": source},
        initial_state_key=initial_state,
    )


def _commercial_target_state(*, product_key: str, event_key: str) -> JourneyStateKey:
    if product_key == "acp":
        if event_key.startswith("request_"):
            return JourneyStateKey.acp_access_requested
        if event_key.startswith("deny_"):
            return JourneyStateKey.blueprint_pro_active
        return JourneyStateKey.acp_prep
    if event_key.startswith("request_"):
        return JourneyStateKey.blueprint_pro_access_requested
    if event_key.startswith("deny_"):
        return JourneyStateKey.blueprint_free_ready
    return JourneyStateKey.blueprint_pro_active


def _state_after_stage_approval(state_key: JourneyStateKey) -> JourneyStateKey:
    next_states = {
        JourneyStateKey.discover: JourneyStateKey.define,
        JourneyStateKey.define: JourneyStateKey.design,
        JourneyStateKey.design: JourneyStateKey.tools,
        JourneyStateKey.tools: JourneyStateKey.memory,
        JourneyStateKey.memory: JourneyStateKey.estimate,
        JourneyStateKey.estimate: JourneyStateKey.blueprint_free_ready,
        JourneyStateKey.validate: JourneyStateKey.package,
        # Package completion is recorded by the durable export event, not merely
        # by entering the packaging stage.
        JourneyStateKey.package: JourneyStateKey.package,
    }
    return next_states.get(state_key, state_key)


def _state_for_acp_phase(phase_key: str) -> JourneyStateKey:
    if phase_key == "blueprint_validation":
        return JourneyStateKey.validate
    if phase_key in {"package_build", "conformance_export"}:
        return JourneyStateKey.package
    return JourneyStateKey.acp_prep


def _substate_for_acp_phase_status(status: ACPWorkflowRunStatus) -> JourneyStateSubstate:
    if status == ACPWorkflowRunStatus.blocked:
        return JourneyStateSubstate.blocked
    if status == ACPWorkflowRunStatus.waiting_user:
        return JourneyStateSubstate.waiting_user
    if status == ACPWorkflowRunStatus.failed:
        return JourneyStateSubstate.failed
    if status in {ACPWorkflowRunStatus.running, ACPWorkflowRunStatus.not_started, ACPWorkflowRunStatus.stale}:
        return JourneyStateSubstate.running
    if status in {ACPWorkflowRunStatus.completed, ACPWorkflowRunStatus.completed_with_observations}:
        return JourneyStateSubstate.completed
    return JourneyStateSubstate.idle


def _load_current_state(db: Session, *, session_id: UUID) -> JourneyStateRecord | None:
    return db.exec(select(JourneyStateRecord).where(JourneyStateRecord.session_id == session_id)).first()


def _serialize_persisted_state(
    db: Session,
    *,
    record: SessionRecord,
    current: JourneyStateRecord,
) -> JourneyStateMachine:
    transitions = db.exec(
        select(JourneyStateTransitionRecord)
        .where(JourneyStateTransitionRecord.session_id == record.id)
        .order_by(JourneyStateTransitionRecord.sequence.desc())
        .limit(50)
    ).all()
    history = [_serialize_transition(record, item) for item in reversed(transitions)]
    return JourneyStateMachine(
        workspace_id=record.workspace_id,
        session_id=record.id,
        current=_stage_for_persisted_values(
            record=record,
            state_key=_as_state_key(current.state_key),
            substate=_as_substate(current.substate),
            progress_percent=current.progress_percent,
            blocking=current.blocking,
            payload=current.state_payload,
        ),
        revision=current.revision,
        state_source="canonical",
        history=history,
        source_contracts=sorted({"journey-state-machine.v1", "journey-state-persistence.v1", *(current.source_contracts or [])}),
    )


def _serialize_transition(record: SessionRecord, transition: JourneyStateTransitionRecord) -> JourneyStateTransition:
    payload = transition.transition_payload or {}
    return JourneyStateTransition(
        sequence=transition.sequence,
        event_key=transition.event_key,
        from_state_key=_as_optional_state_key(transition.from_state_key),
        from_substate=_as_optional_substate(transition.from_substate),
        to_state=_stage_for_persisted_values(
            record=record,
            state_key=_as_state_key(transition.to_state_key),
            substate=_as_substate(transition.to_substate),
            progress_percent=payload.get("progress_percent"),
            blocking=bool(payload.get("blocking")),
            payload=payload,
        ),
        actor_type=transition.actor_type,
        actor_user_id=transition.actor_user_id,
        reason=transition.reason,
        correlation_id=transition.correlation_id,
        occurred_at=transition.occurred_at.isoformat(),
        metadata=dict(payload.get("metadata") or {}),
    )


def _as_state_key(value: str) -> JourneyStateKey:
    try:
        return JourneyStateKey(str(value))
    except ValueError:
        return JourneyStateKey.discover


def _as_optional_state_key(value: str) -> JourneyStateKey | None:
    return _as_state_key(value) if str(value).strip() else None


def _as_substate(value: str) -> JourneyStateSubstate:
    try:
        return JourneyStateSubstate(str(value))
    except ValueError:
        return JourneyStateSubstate.idle


def _as_optional_substate(value: str) -> JourneyStateSubstate | None:
    return _as_substate(value) if str(value).strip() else None


def _stage_for_persisted_values(
    *,
    record: SessionRecord,
    state_key: JourneyStateKey,
    substate: JourneyStateSubstate,
    progress_percent: int | None,
    blocking: bool,
    payload: dict[str, Any] | None = None,
) -> JourneyStateMachineStage:
    saved = payload or {}
    product_key = _product_for_state(state_key)
    stage_key = str(saved.get("stage_key") or _persisted_stage_key(state_key))
    normalized_progress = 100 if state_key in {JourneyStateKey.blueprint_free_ready, JourneyStateKey.completed} else int(
        progress_percent if progress_percent is not None else saved.get("progress_percent") or 0
    )
    return JourneyStateMachineStage(
        state_key=state_key,
        substate=substate,
        label=str(saved.get("label") or STATE_LABELS[state_key]),
        detail=str(saved.get("detail") or STATE_DETAILS[state_key]),
        product_key=product_key,
        stage_key=stage_key,
        href=str(saved.get("href") or _href_for_state(record_id=str(record.id), state_key=state_key)),
        progress_percent=normalized_progress,
        blocking=blocking,
    )


def _stage_payload(stage: JourneyStateMachineStage, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "label": stage.label,
        "detail": stage.detail,
        "product_key": stage.product_key.value,
        "stage_key": stage.stage_key,
        "href": stage.href,
        "progress_percent": stage.progress_percent,
        "blocking": stage.blocking,
        "metadata": dict(metadata or {}),
    }


def _persisted_stage_key(state_key: JourneyStateKey) -> str:
    if state_key in WORK_STAGE_KEYS:
        return state_key.value
    if state_key == JourneyStateKey.blueprint_free_ready:
        return "estimate"
    if state_key in {
        JourneyStateKey.blueprint_pro_access_requested,
        JourneyStateKey.blueprint_pro_access_pending,
        JourneyStateKey.blueprint_pro_active,
    }:
        return "blueprint_pro"
    if state_key in {JourneyStateKey.validate, JourneyStateKey.package}:
        return state_key.value
    if state_key == JourneyStateKey.completed:
        return "package"
    return "acp"


def _pending_request_products(db: Session, *, record: SessionRecord) -> set[str]:
    rows = db.exec(
        select(CommercialAccessRequestRecord.product_key).where(
            CommercialAccessRequestRecord.workspace_id == record.workspace_id,
            CommercialAccessRequestRecord.session_id == record.id,
            CommercialAccessRequestRecord.status == CommercialAccessRequestStatus.pending,
        )
    ).all()
    return {str(product_key or "").strip() for product_key in rows if str(product_key or "").strip()}


def _resolve_current_state_key(
    *,
    record: SessionRecord,
    overview: ProductJourneyOverview,
    access_tier: CommercialTier,
    checkout_state: str,
    pending_requests: set[str],
    products_by_key: dict[ProductBuildProductKey, ProductJourneyProductSummary],
) -> JourneyStateKey:
    current_stage_key = _normalize_stage_key(overview.current_stage.stage_key)
    record_stage_key = _normalize_stage_key(str(getattr(record.current_stage, "value", record.current_stage or "")))
    blueprint = products_by_key.get(ProductBuildProductKey.blueprint_basic)
    pro = products_by_key.get(ProductBuildProductKey.blueprint_pro)
    acp = products_by_key.get(ProductBuildProductKey.acp)

    if access_tier == CommercialTier.acp:
        if _is_completed_state(overview, acp):
            return JourneyStateKey.completed
        if current_stage_key == JourneyStateKey.package or record_stage_key == JourneyStateKey.package:
            return JourneyStateKey.package
        if current_stage_key == JourneyStateKey.validate or record_stage_key == JourneyStateKey.validate:
            return JourneyStateKey.validate
        return JourneyStateKey.acp_prep

    if "acp" in pending_requests:
        return JourneyStateKey.acp_access_requested

    if access_tier == CommercialTier.blueprint_pro and checkout_state == "pending":
        return JourneyStateKey.acp_access_pending

    if access_tier == CommercialTier.blueprint_pro:
        return JourneyStateKey.blueprint_pro_active

    if "blueprint_pro" in pending_requests:
        return JourneyStateKey.blueprint_pro_access_requested

    if checkout_state == "pending":
        return JourneyStateKey.blueprint_pro_access_pending

    if _is_blueprint_ready(record=record, overview=overview, blueprint=blueprint, pro=pro):
        return JourneyStateKey.blueprint_free_ready

    return current_stage_key


def _is_blueprint_ready(
    *,
    record: SessionRecord,
    overview: ProductJourneyOverview,
    blueprint: ProductJourneyProductSummary | None,
    pro: ProductJourneyProductSummary | None,
) -> bool:
    stage_value = str(getattr(record.current_stage, "value", record.current_stage or "")).strip().lower()
    if stage_value in {"post_validation", "ready_for_export"}:
        return True
    if overview.current_stage.stage_key in {"estimate", "validate", "package"}:
        return True
    if blueprint is not None and (
        blueprint.progress_percent > 0
        or blueprint.available_deliverable_count > 0
        or blueprint.lifecycle in {ProductBuildLifecycle.partial, ProductBuildLifecycle.completed}
    ):
        return True
    return pro is not None and pro.progress_percent > 0


def _is_completed_state(overview: ProductJourneyOverview, acp: ProductJourneyProductSummary | None) -> bool:
    if overview.blocking_attention_count or overview.technical_error_count:
        return False
    if acp is None:
        return False
    return acp.lifecycle == ProductBuildLifecycle.completed


def _build_stage(
    *,
    record: SessionRecord,
    overview: ProductJourneyOverview,
    products_by_key: dict[ProductBuildProductKey, ProductJourneyProductSummary],
    state_key: JourneyStateKey,
) -> JourneyStateMachineStage:
    product_key = _product_for_state(state_key)
    target_product = products_by_key.get(product_key)
    stage_key = _stage_key_for_state(state_key, overview)
    return JourneyStateMachineStage(
        state_key=state_key,
        substate=_substate_for_state(state_key, overview=overview, target_product=target_product),
        label=STATE_LABELS[state_key],
        detail=STATE_DETAILS[state_key],
        product_key=product_key,
        stage_key=stage_key,
        href=_href_for_state(record_id=str(record.id), state_key=state_key),
        progress_percent=_progress_for_state(state_key, overview=overview, target_product=target_product),
        blocking=overview.blocking_attention_count > 0,
    )


def _product_for_state(state_key: JourneyStateKey) -> ProductBuildProductKey:
    if state_key in {
        JourneyStateKey.blueprint_pro_access_requested,
        JourneyStateKey.blueprint_pro_access_pending,
        JourneyStateKey.blueprint_pro_active,
    }:
        return ProductBuildProductKey.blueprint_pro
    if state_key in {
        JourneyStateKey.acp_access_requested,
        JourneyStateKey.acp_access_pending,
        JourneyStateKey.acp_prep,
        JourneyStateKey.validate,
        JourneyStateKey.package,
        JourneyStateKey.completed,
    }:
        return ProductBuildProductKey.acp
    return ProductBuildProductKey.blueprint_basic


def _substate_for_state(
    state_key: JourneyStateKey,
    *,
    overview: ProductJourneyOverview,
    target_product: ProductJourneyProductSummary | None,
) -> JourneyStateSubstate:
    if state_key in {
        JourneyStateKey.blueprint_pro_access_requested,
        JourneyStateKey.blueprint_pro_access_pending,
        JourneyStateKey.acp_access_requested,
        JourneyStateKey.acp_access_pending,
    }:
        return JourneyStateSubstate.waiting_dependency
    if state_key in {JourneyStateKey.blueprint_free_ready, JourneyStateKey.completed}:
        return JourneyStateSubstate.completed
    if overview.blocking_attention_count > 0:
        return JourneyStateSubstate.blocked
    if overview.technical_error_count > 0 or (target_product is not None and target_product.lifecycle == ProductBuildLifecycle.error):
        return JourneyStateSubstate.failed
    if overview.active_operation is not None or (target_product is not None and target_product.active_operation is not None):
        return JourneyStateSubstate.running
    if target_product is not None and target_product.lifecycle in ACTIVE_LIFECYCLES:
        return JourneyStateSubstate.running
    if target_product is not None and target_product.lifecycle == ProductBuildLifecycle.completed:
        return JourneyStateSubstate.completed
    return JourneyStateSubstate.idle


def _progress_for_state(
    state_key: JourneyStateKey,
    *,
    overview: ProductJourneyOverview,
    target_product: ProductJourneyProductSummary | None,
) -> int:
    if state_key in WORK_STAGE_KEYS:
        return overview.current_stage.progress_percent
    if state_key in {JourneyStateKey.blueprint_free_ready, JourneyStateKey.completed}:
        return 100
    if target_product is not None:
        return target_product.progress_percent
    return 0


def _normalize_stage_key(value: str) -> JourneyStateKey:
    normalized = str(value or "").strip().lower()
    legacy_map = {
        "draft_capture": JourneyStateKey.discover,
        "input_validation": JourneyStateKey.discover,
        "normalize_discovery": JourneyStateKey.discover,
        "build_canvas": JourneyStateKey.define,
        "build_blueprint": JourneyStateKey.design,
        "post_validation": JourneyStateKey.validate,
        "ready_for_export": JourneyStateKey.package,
    }
    if normalized in {item.value for item in WORK_STAGE_KEYS}:
        return JourneyStateKey(normalized)
    if normalized == "validate":
        return JourneyStateKey.validate
    if normalized == "package":
        return JourneyStateKey.package
    return legacy_map.get(normalized, JourneyStateKey.discover)


def _stage_key_for_state(state_key: JourneyStateKey, overview: ProductJourneyOverview) -> str:
    if state_key in WORK_STAGE_KEYS:
        return state_key.value
    if state_key in {JourneyStateKey.validate, JourneyStateKey.package}:
        return state_key.value
    if state_key == JourneyStateKey.completed:
        return "package"
    if state_key == JourneyStateKey.blueprint_free_ready:
        return "estimate"
    if state_key in {
        JourneyStateKey.blueprint_pro_access_requested,
        JourneyStateKey.blueprint_pro_access_pending,
        JourneyStateKey.blueprint_pro_active,
    }:
        return "blueprint_pro"
    if state_key in {
        JourneyStateKey.acp_access_requested,
        JourneyStateKey.acp_access_pending,
        JourneyStateKey.acp_prep,
    }:
        return overview.current_stage.stage_key if overview.current_stage.stage_key in {"validate", "package"} else "acp"
    return overview.current_stage.stage_key


def _href_for_state(*, record_id: str, state_key: JourneyStateKey) -> str:
    if state_key in WORK_STAGE_KEYS:
        return f"/projects/{record_id}/work/{state_key.value}"
    if state_key == JourneyStateKey.blueprint_free_ready:
        return f"/projects/{record_id}/blueprint"
    if state_key in {
        JourneyStateKey.blueprint_pro_access_requested,
        JourneyStateKey.blueprint_pro_access_pending,
    }:
        return f"/projects/{record_id}/blueprint/pro/overview"
    if state_key == JourneyStateKey.blueprint_pro_active:
        return f"/projects/{record_id}/blueprint/pro"
    if state_key in {
        JourneyStateKey.acp_access_requested,
        JourneyStateKey.acp_access_pending,
    }:
        return f"/projects/{record_id}/acp/overview"
    if state_key == JourneyStateKey.validate:
        return f"/projects/{record_id}/acp?acp_tab=validate"
    if state_key in {JourneyStateKey.package, JourneyStateKey.completed}:
        return f"/projects/{record_id}/acp?acp_tab=package"
    return f"/projects/{record_id}/acp"
