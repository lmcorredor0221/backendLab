from app.models import CommercialEventRecord, CommercialTier, JourneyArtifactState, JourneyStageArtifactRecord, SessionRecord, StageOperationRecord
from app.services.diagram_center.persistence import DiagramGenerationJobRecord  # noqa: F401
from app.services.product_processing import (
    ProductProcessingMode,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    PremiumUncertaintyResolutionRequest,
    UncertaintyBacklogStatus,
    UncertaintyDisposition,
    acp_route_blocking_reasons,
    build_acp_direct_resolution,
    build_premium_enrichment_workspace,
    classify_inference_permission,
    classify_uncertainty_for_profile,
    defer_premium_uncertainty_to_acp,
    get_product_processing_profile,
    list_product_build_runs,
    list_product_build_steps,
    list_uncertainty_backlog,
    prioritize_uncertainty_backlog,
    resolve_premium_uncertainty,
    resolve_uncertainty,
    resolve_product_processing_mode,
    upsert_uncertainty_backlog,
)
from app.services.deliverable_catalog.persistence import (  # noqa: F401
    DeliverableGenerationJobRecord,
    DeliverablePromptVersionRecord,
    DeliverableQualitySnapshotRecord,
)
from app.services.product_processing.persistence import (  # noqa: F401
    ProductBuildRunRecord,
    ProductBuildStepRecord,
    UncertaintyBacklogRecord,
)
from sqlmodel import SQLModel, Session, create_engine, select
from sqlalchemy.pool import StaticPool
from uuid import UUID, uuid4


def _count_records(db: Session, model: type) -> int:
    return len(db.exec(select(model)).all())


def test_resolve_product_processing_mode_by_tier_and_direct_acp() -> None:
    assert resolve_product_processing_mode(CommercialTier.blueprint) == ProductProcessingMode.basic_free
    assert (
        resolve_product_processing_mode(CommercialTier.blueprint_pro)
        == ProductProcessingMode.premium_enrichment
    )
    assert resolve_product_processing_mode(CommercialTier.acp) == ProductProcessingMode.acp_implementation
    assert (
        resolve_product_processing_mode(CommercialTier.blueprint, acp_direct=True)
        == ProductProcessingMode.acp_implementation
    )


def test_basic_free_infers_or_defers_without_user_attention() -> None:
    profile = get_product_processing_profile(ProductProcessingMode.basic_free)
    classification = classify_uncertainty_for_profile(
        "discover",
        {
            "key": "q1",
            "question": "Que base de datos debe usarse en implementacion?",
            "confidence": 0.4,
            "priority": "high",
            "blocking": True,
        },
        profile,
    )

    assert classification.disposition == UncertaintyDisposition.defer
    assert classification.should_continue_processing is True
    assert classification.should_surface_to_user is False
    assert classification.should_create_attention is False


def test_inference_permission_respects_free_vs_acp_boundary() -> None:
    assert (
        classify_inference_permission(
            "define",
            {
                "key": "define_runtime",
                "question": "Que runtime, credenciales y despliegue final se usaran?",
                "suggested_answer": "Usar contenedor administrado y vault del cliente.",
            },
            ProductProcessingMode.basic_free,
            inferred_answer="Usar contenedor administrado y vault del cliente.",
            confidence=0.93,
        )
        == "defer_to_acp"
    )
    assert (
        classify_inference_permission(
            "design",
            {
                "key": "design_pattern",
                "question": "Que patron de coordinacion encaja mejor con el flujo principal?",
                "suggested_answer": "Supervisor con handoffs trazables.",
            },
            ProductProcessingMode.basic_free,
            inferred_answer="Supervisor con handoffs trazables.",
            confidence=0.91,
        )
        == "apply_now"
    )


def test_basic_free_delegation_does_not_touch_build_state_or_stale_artifacts() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()

    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="Delegacion gobernada en Free",
                commercial_tier=CommercialTier.blueprint,
            )
        )
        artifact = JourneyStageArtifactRecord(
            workspace_id=workspace_id,
            session_id=session_id,
            artifact_kind="design_recommendation",
            stage_key="design",
            version_number=1,
            state=JourneyArtifactState.approved,
            stale_reasons=[],
        )
        db.add(artifact)
        classification = classify_uncertainty_for_profile(
            "design",
            {
                "key": "implementation_database",
                "question": "Que base de datos y credenciales se usaran durante la implementacion?",
                "confidence": 0.34,
                "priority": "high",
                "blocking": True,
            },
            ProductProcessingMode.basic_free,
        )

        entry = upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=classification,
            dependency_keys=["design.selected_alternative"],
            created_from="test",
        )
        db.refresh(artifact)

        assert entry.status == UncertaintyBacklogStatus.deferred
        assert entry.target_stage == "acp"
        assert classification.should_continue_processing is True
        assert classification.should_create_attention is False
        assert _count_records(db, ProductBuildRunRecord) == 0
        assert _count_records(db, ProductBuildStepRecord) == 0
        assert _count_records(db, DeliverableGenerationJobRecord) == 0
        assert _count_records(db, DiagramGenerationJobRecord) == 0
        assert _count_records(db, DeliverableQualitySnapshotRecord) == 0
        assert _count_records(db, StageOperationRecord) == 0
        assert artifact.state == JourneyArtifactState.approved
        assert artifact.stale_reasons == []


def test_premium_prioritizes_high_value_business_questions() -> None:
    profile = get_product_processing_profile(ProductProcessingMode.premium_enrichment)
    classification = classify_uncertainty_for_profile(
        "define",
        {
            "key": "q2",
            "question": "Que excepciones criticas debe contemplar el flujo objetivo?",
            "confidence": 0.82,
            "priority": "high",
            "suggested_answer": "Escalar excepciones ambiguas a revision humana.",
            "answer_options": ["Escalar a humano", "Registrar para analisis posterior"],
        },
        profile,
    )

    assert classification.disposition == UncertaintyDisposition.resolve_now
    assert classification.should_surface_to_user is True
    assert classification.should_create_attention is True


def test_opening_premium_workspace_does_not_create_hidden_generation_jobs() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()

    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="Premium no ejecuta sin accion explicita",
                commercial_tier=CommercialTier.blueprint_pro,
            )
        )
        artifact = JourneyStageArtifactRecord(
            workspace_id=workspace_id,
            session_id=session_id,
            artifact_kind="design_recommendation",
            stage_key="design",
            version_number=1,
            state=JourneyArtifactState.approved,
            stale_reasons=[],
        )
        db.add(artifact)
        deferred = classify_uncertainty_for_profile(
            "design",
            {
                "key": "implementation_runtime",
                "question": "Que runtime, secrets y estrategia de despliegue se usaran?",
                "confidence": 0.38,
                "priority": "high",
            },
            ProductProcessingMode.basic_free,
        )
        upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=deferred,
            dependency_keys=["design.selected_alternative"],
            created_from="test",
        )

        workspace = build_premium_enrichment_workspace(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            current_tier=CommercialTier.blueprint_pro,
        )
        db.refresh(artifact)
        active_steps = db.exec(
            select(ProductBuildStepRecord).where(
                ProductBuildStepRecord.status.in_(["queued", "running", "generating"])
            )
        ).all()

        assert workspace.deferred_count == 1
        assert _count_records(db, ProductBuildRunRecord) == 1
        assert _count_records(db, DeliverableGenerationJobRecord) == 0
        assert _count_records(db, DiagramGenerationJobRecord) == 0
        assert _count_records(db, DeliverableQualitySnapshotRecord) == 0
        assert _count_records(db, StageOperationRecord) == 0
        assert active_steps == []
        assert artifact.state == JourneyArtifactState.approved
        assert artifact.stale_reasons == []


def test_acp_implementation_does_not_silence_required_questions() -> None:
    profile = get_product_processing_profile(ProductProcessingMode.acp_implementation)
    classification = classify_uncertainty_for_profile(
        "discover",
        {
            "key": "q3",
            "question": "Que credenciales de integracion se usaran durante la implementacion?",
            "priority": "critical",
            "required_for_implementation": True,
        },
        profile,
    )

    assert classification.disposition == UncertaintyDisposition.block
    assert classification.should_surface_to_user is True
    assert classification.should_create_attention is True
    assert classification.should_continue_processing is False


def test_delegating_premium_uncertainty_to_acp_does_not_start_jobs_or_stale_artifacts() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()

    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="Premium delega al ACP sin reconciliar",
                commercial_tier=CommercialTier.blueprint_pro,
            )
        )
        artifact = JourneyStageArtifactRecord(
            workspace_id=workspace_id,
            session_id=session_id,
            artifact_kind="design_recommendation",
            stage_key="design",
            version_number=1,
            state=JourneyArtifactState.approved,
            stale_reasons=[],
        )
        db.add(artifact)
        classification = classify_uncertainty_for_profile(
            "design",
            {
                "key": "approval_exception_policy",
                "question": "Que excepciones criticas requieren aprobacion humana?",
                "confidence": 0.83,
                "priority": "high",
                "blocking": True,
                "answer_options": [{"key": "always_escalate", "label": "Escalar excepciones ambiguas"}],
            },
            ProductProcessingMode.premium_enrichment,
        )
        entry = upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=classification,
            dependency_keys=["design.approval_points"],
            created_from="test",
        )

        build_premium_enrichment_workspace(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            current_tier=CommercialTier.blueprint_pro,
        )
        deferred = defer_premium_uncertainty_to_acp(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            backlog_id=UUID(entry.id),
            actor_user_id=actor_id,
        )
        db.refresh(artifact)
        active_steps = db.exec(
            select(ProductBuildStepRecord).where(
                ProductBuildStepRecord.status.in_(["queued", "running", "generating"])
            )
        ).all()
        decision_events = db.exec(
            select(CommercialEventRecord).where(
                CommercialEventRecord.session_id == session_id,
                CommercialEventRecord.event_key == "attention_action_v2",
            )
        ).all()

        assert deferred.status == UncertaintyBacklogStatus.deferred.value
        assert deferred.disposition == UncertaintyDisposition.defer.value
        assert deferred.target_stage == "acp"
        assert len(decision_events) == 1
        event_metadata = decision_events[0].metadata_payload
        assert event_metadata["decision_contract_version"] == "decision-observability.v1"
        assert event_metadata["action_kind"] == "defer"
        assert event_metadata["target_stage"] == "acp"
        assert event_metadata["automatic_job_creation"] is False
        assert event_metadata["reconciliation_policy"].endswith("delegation_never_generates_jobs")
        assert _count_records(db, DeliverableGenerationJobRecord) == 0
        assert _count_records(db, DiagramGenerationJobRecord) == 0
        assert _count_records(db, DeliverableQualitySnapshotRecord) == 0
        assert _count_records(db, StageOperationRecord) == 0
        assert active_steps == []
        assert artifact.state == JourneyArtifactState.approved
        assert artifact.stale_reasons == []


def test_uncertainty_backlog_persists_and_prioritizes_items() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()

    with Session(engine) as db:
        blocked = classify_uncertainty_for_profile(
            "tools",
            {
                "key": "tool_contract",
                "question": "Confirmar contrato minimo de la API de soporte.",
                "priority": "critical",
                "blocking": True,
                "required_for_implementation": True,
                "answer_options": [{"key": "rest", "label": "REST", "recommended": True}],
            },
            ProductProcessingMode.acp_implementation,
        )
        deferred = classify_uncertainty_for_profile(
            "discover",
            {"key": "db_choice", "question": "Que base de datos se usara?", "confidence": 0.3},
            ProductProcessingMode.basic_free,
        )

        blocked_entry = upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=blocked,
            dependency_keys=["tools.minimum_set"],
        )
        upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=deferred,
        )

        entries = list_uncertainty_backlog(db, workspace_id=workspace_id, session_id=session_id)
        prioritized = prioritize_uncertainty_backlog(entries)

        assert len(entries) == 2
        assert prioritized[0].uncertainty_key == "tool_contract"
        assert prioritized[0].status == UncertaintyBacklogStatus.open
        assert prioritized[1].status == UncertaintyBacklogStatus.deferred

        resolved = resolve_uncertainty(db, backlog_id=UUID(blocked_entry.id), resolved_answer="Usar REST")
        assert resolved.status == UncertaintyBacklogStatus.resolved


def test_premium_enrichment_resolves_with_impact_analysis_before_reconciliation() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()

    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="Blueprint Pro selectivo",
                commercial_tier=CommercialTier.blueprint_pro,
            )
        )
        classification = classify_uncertainty_for_profile(
            "define",
            {
                "key": "critical_exception_flow",
                "question": "Que excepciones criticas debe contemplar el flujo objetivo?",
                "confidence": 0.86,
                "priority": "high",
                "suggested_answer": "Escalar excepciones ambiguas a revision humana.",
                "answer_options": [{"key": "human_escalation", "label": "Escalar a humano", "recommended": True}],
            },
            ProductProcessingMode.premium_enrichment,
        )
        entry = upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=classification,
            dependency_keys=["definition.requirements"],
            created_from="test",
        )

        workspace = build_premium_enrichment_workspace(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            current_tier=CommercialTier.blueprint_pro,
        )
        runs_before = list_product_build_runs(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )
        lifecycle_before = runs_before[0].lifecycle
        step_statuses_before = [
            (step.step_key, step.status)
            for step in list_product_build_steps(db, run_id=runs_before[0].id)
        ]
        result = resolve_premium_uncertainty(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            backlog_id=UUID(entry.id),
            actor_user_id=actor_id,
            payload=PremiumUncertaintyResolutionRequest(selected_option_key="human_escalation", max_deliverables=2),
        )
        runs_after = list_product_build_runs(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            product_key=ProductBuildProductKey.blueprint_pro,
        )
        steps_after = list_product_build_steps(db, run_id=runs_after[0].id)
        decision_events = db.exec(
            select(CommercialEventRecord).where(
                CommercialEventRecord.session_id == session_id,
                CommercialEventRecord.event_key == "attention_action_v2",
            )
        ).all()

    assert workspace.items
    assert workspace.items[0].ordered_regeneration_keys
    assert lifecycle_before == ProductBuildLifecycle.requires_attention.value
    assert any(key.startswith("premium_backlog:") and status == "requires_attention" for key, status in step_statuses_before)
    assert "diagram.c4_context" in result.affected_deliverable_keys
    assert result.stale_deliverable_keys == []
    assert result.material_impact is True
    assert result.reconciliation_decision == "structural_reconciliation"
    assert result.reconciliation_status == "pending_user_confirmation"
    assert result.reprocess_decision == "structural_reprocess"
    assert result.regenerated_deliverable_keys == []
    assert result.reconciled_deliverable_keys == []
    assert result.resolved_entry.status == UncertaintyBacklogStatus.resolved
    assert result.preserved_deliverable_keys
    assert result.generation_job_ids == []
    assert result.superseded_uncertainty_count == 0
    assert result.queue_total == min(2, len(result.ordered_regeneration_keys))
    assert result.queue_completed == 0
    assert result.queue_status == "pending_user_confirmation"
    assert len(decision_events) == 1
    event_metadata = decision_events[0].metadata_payload
    assert event_metadata["decision_contract_version"] == "decision-observability.v1"
    assert event_metadata["action_kind"] == "answer"
    assert event_metadata["material_impact"] is True
    assert event_metadata["reconciliation_status"] == "pending_user_confirmation"
    assert event_metadata["execution_mode"] == "analyze_only"
    assert event_metadata["automatic_job_creation"] is False
    assert any(step.step_key.startswith("premium_backlog:") and step.status == "completed" for step in steps_after)
    assert runs_after[0].lifecycle != ProductBuildLifecycle.requires_attention.value


def test_premium_enrichment_reconciles_only_when_explicitly_requested() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()

    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="Blueprint Pro reconciliacion explicita",
                commercial_tier=CommercialTier.blueprint_pro,
            )
        )
        classification = classify_uncertainty_for_profile(
            "define",
            {
                "key": "critical_exception_flow",
                "question": "Que excepciones criticas debe contemplar el flujo objetivo?",
                "confidence": 0.86,
                "priority": "high",
                "suggested_answer": "Escalar excepciones ambiguas a revision humana.",
                "answer_options": [{"key": "human_escalation", "label": "Escalar a humano", "recommended": True}],
            },
            ProductProcessingMode.premium_enrichment,
        )
        entry = upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=classification,
            dependency_keys=["definition.requirements"],
            created_from="test",
        )

        result = resolve_premium_uncertainty(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            backlog_id=UUID(entry.id),
            actor_user_id=actor_id,
            payload=PremiumUncertaintyResolutionRequest(
                execution_mode="apply_reprocess",
                max_deliverables=2,
                regenerate=True,
                selected_option_key="human_escalation",
            ),
        )
        decision_events = db.exec(
            select(CommercialEventRecord).where(
                CommercialEventRecord.session_id == session_id,
                CommercialEventRecord.event_key == "attention_action_v2",
            )
        ).all()

    assert result.material_impact is True
    assert result.execution_mode == "apply_reconciliation"
    assert result.legacy_execution_mode == "apply_reprocess"
    assert result.reconciliation_decision == "structural_reconciliation"
    assert result.reconciliation_status == "completed"
    assert result.reprocess_decision == "structural_reprocess"
    assert len(result.reconciled_deliverable_keys) <= 2
    assert len(result.regenerated_deliverable_keys) <= 2
    assert result.generation_job_ids
    assert result.queue_total == min(2, len(result.ordered_regeneration_keys))
    assert result.queue_completed == len(result.regenerated_deliverable_keys)
    assert result.queue_status == "completed"
    assert len(decision_events) == 1
    event_metadata = decision_events[0].metadata_payload
    assert event_metadata["decision_contract_version"] == "decision-observability.v1"
    assert event_metadata["execution_mode"] == "apply_reconciliation"
    assert event_metadata["legacy_execution_mode"] == "apply_reprocess"
    assert event_metadata["automatic_job_creation"] is True
    assert event_metadata["generation_job_ids"]


def test_premium_enrichment_accepts_canonical_reconciliation_execution_mode() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()

    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="Blueprint Pro canonical reconciliation",
                commercial_tier=CommercialTier.blueprint_pro,
            )
        )
        classification = classify_uncertainty_for_profile(
            "tools",
            {
                "key": "integration_contracts",
                "question": "Que contratos de integracion deben quedar versionados?",
                "confidence": 0.84,
                "priority": "high",
                "suggested_answer": "Versionar contratos externos y esquemas de errores.",
                "answer_options": [{"key": "versioned_contracts", "label": "Versionar contratos", "recommended": True}],
            },
            ProductProcessingMode.premium_enrichment,
        )
        entry = upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=classification,
            dependency_keys=["tools.minimum_set"],
            created_from="test",
        )

        result = resolve_premium_uncertainty(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            backlog_id=UUID(entry.id),
            actor_user_id=actor_id,
            payload=PremiumUncertaintyResolutionRequest(
                execution_mode="apply_reconciliation",
                max_deliverables=1,
                selected_option_key="versioned_contracts",
            ),
        )

    assert result.execution_mode == "apply_reconciliation"
    assert result.legacy_execution_mode is None
    assert result.reconciliation_status == "completed"
    assert len(result.reconciled_deliverable_keys) == 1
    assert result.generation_job_ids


def test_premium_workspace_does_not_duplicate_resolved_deferred_items() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()

    with Session(engine) as db:
        db.add(
            SessionRecord(
                id=session_id,
                user_id=actor_id,
                workspace_id=workspace_id,
                title="Deferred item resolved in Premium",
                commercial_tier=CommercialTier.blueprint_pro,
            )
        )
        deferred = classify_uncertainty_for_profile(
            "discover",
            {
                "key": "db_choice",
                "question": "Que base de datos se usara?",
                "confidence": 0.35,
            },
            ProductProcessingMode.basic_free,
        )
        entry = upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=deferred,
            dependency_keys=["session.discovery"],
            created_from="test",
        )
        resolve_uncertainty(db, backlog_id=UUID(entry.id), resolved_answer="PostgreSQL gestionado.")

        workspace = build_premium_enrichment_workspace(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            current_tier=CommercialTier.blueprint_pro,
        )

    assert workspace.deferred_count == 0
    assert workspace.resolved_count == 1
    assert len(workspace.items) == 1
    assert [item.entry.id for item in workspace.items] == [entry.id]
    assert workspace.items[0].entry.status == UncertaintyBacklogStatus.resolved


def test_acp_direct_resolution_requires_all_stages_and_open_blockers_closed() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    workspace_id = uuid4()
    session_id = uuid4()
    actor_id = uuid4()
    required_stages = ("discover", "define", "design", "tools", "memory", "estimate", "validate")

    with Session(engine) as db:
        record = SessionRecord(
            id=session_id,
            user_id=actor_id,
            workspace_id=workspace_id,
            title="ACP directo",
            commercial_tier=CommercialTier.acp,
        )
        db.add(record)
        for index, stage in enumerate(("discover", "define"), start=1):
            db.add(
                JourneyStageArtifactRecord(
                    workspace_id=workspace_id,
                    session_id=session_id,
                    artifact_kind=f"{stage}_artifact",
                    stage_key=stage,
                    version_number=index,
                    state=JourneyArtifactState.approved,
                )
            )
        classification = classify_uncertainty_for_profile(
            "tools",
            {
                "key": "tool_contract_owner",
                "question": "Quien aprobara los contratos de herramientas durante la implementacion?",
                "priority": "critical",
                "required_for_implementation": True,
            },
            ProductProcessingMode.acp_implementation,
        )
        entry = upsert_uncertainty_backlog(
            db,
            workspace_id=workspace_id,
            session_id=session_id,
            classification=classification,
            dependency_keys=["tools.minimum_set"],
            created_from="test",
        )

        blocked = build_acp_direct_resolution(db, record=record)
        assert blocked.route_kind == "acp_direct"
        assert blocked.can_start_package is False
        assert "design" in blocked.missing_stage_keys
        assert "missing_stage:design" in acp_route_blocking_reasons(blocked)
        assert blocked.total_blocking_questions == 1

        for index, stage in enumerate(required_stages[2:], start=3):
            db.add(
                JourneyStageArtifactRecord(
                    workspace_id=workspace_id,
                    session_id=session_id,
                    artifact_kind=f"{stage}_artifact",
                    stage_key=stage,
                    version_number=index,
                    state=JourneyArtifactState.approved,
                )
            )
        resolve_uncertainty(db, backlog_id=UUID(entry.id), resolved_answer="Owner tecnico asignado.")

        ready = build_acp_direct_resolution(db, record=record)

    assert ready.can_start_package is True
    assert ready.can_export_package is True
    assert ready.missing_stage_keys == []
    assert set(ready.completed_stage_keys) == set(required_stages)
