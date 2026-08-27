from app.models import CommercialTier, JourneyArtifactState, JourneyStageArtifactRecord, SessionRecord
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
    classify_uncertainty_for_profile,
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
from app.services.product_processing.persistence import UncertaintyBacklogRecord  # noqa: F401
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool
from uuid import UUID, uuid4


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


def test_premium_enrichment_resolves_with_impact_analysis_before_reprocessing() -> None:
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

    assert workspace.items
    assert workspace.items[0].ordered_regeneration_keys
    assert lifecycle_before == ProductBuildLifecycle.requires_attention.value
    assert any(key.startswith("premium_backlog:") and status == "requires_attention" for key, status in step_statuses_before)
    assert "diagram.c4_context" in result.stale_deliverable_keys
    assert result.material_impact is True
    assert result.reprocess_decision == "structural_reprocess"
    assert result.regenerated_deliverable_keys == []
    assert result.resolved_entry.status == UncertaintyBacklogStatus.resolved
    assert result.preserved_deliverable_keys
    assert result.generation_job_ids == []
    assert result.queue_total == 0
    assert result.queue_completed == 0
    assert result.queue_status == "not_requested"
    assert any(step.step_key.startswith("premium_backlog:") and step.status == "completed" for step in steps_after)
    assert runs_after[0].lifecycle != ProductBuildLifecycle.requires_attention.value


def test_premium_enrichment_reprocesses_only_when_explicitly_requested() -> None:
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
                title="Blueprint Pro reprocessamiento explicito",
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

    assert result.material_impact is True
    assert result.reprocess_decision == "structural_reprocess"
    assert len(result.regenerated_deliverable_keys) <= 2
    assert result.generation_job_ids
    assert result.queue_total == min(2, len(result.ordered_regeneration_keys))
    assert result.queue_completed == len(result.regenerated_deliverable_keys)
    assert result.queue_status == "completed"


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
