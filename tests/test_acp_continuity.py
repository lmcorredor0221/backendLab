from __future__ import annotations

from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    ACPPreview,
    ACPValidationIssue,
    ACPValidationReport,
    ConstructionGapEntry,
    ConstructionQuestionAnswerRequest,
    ConstructionQuestionEntry,
    ConstructionQuestionResponseRecord,
    ConstructionReadinessReport,
    UserRecord,
)
from app.services.acp_continuity import (
    append_construction_readiness_gaps,
    apply_uncertainty_backlog_acp_answer,
    build_construction_gaps_from_uncertainty_backlog,
    build_deferred_construction_decision_backlog,
    build_construction_question_views,
    build_continuity_answer_map,
    load_uncertainty_backlog_records,
    merge_construction_question_records_with_uncertainty_backlog,
    overlay_construction_readiness,
    uncertainty_backlog_question_key,
)
from app.services.product_processing.persistence import UncertaintyBacklogRecord


def _memory_db() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_overlay_construction_readiness_preserves_validation_block_even_after_answers() -> None:
    session_id = uuid4()
    preview = ACPPreview(
        session_id=session_id,
        validation=ACPValidationReport(
            overall_status="incomplete",
            can_export_zip=False,
            issues=[
                ACPValidationIssue(
                    code="missing_runtime_contract",
                    severity="error",
                    blocking=True,
                    message="Falta contrato runtime.",
                )
            ],
        ),
        construction_readiness=ConstructionReadinessReport(
            overall_status="blocked",
            can_start_build=False,
            blocking_gaps=0,
            open_questions=1,
            assumptions_count=0,
            gaps=[
                ConstructionGapEntry(
                    gap_key="runtime_contract_incomplete",
                    title="Runtime incompleto",
                    severity="warning",
                    status="open",
                    blocking_stage="runtime_configuration",
                    summary="Faltan detalles del runtime.",
                    questions=[
                        ConstructionQuestionEntry(
                            question_key="runtime_stack",
                            question_text="Define stack runtime",
                            rationale="Necesario para preparar el ACP.",
                            target_owner="implementation_owner",
                            blocking=False,
                        )
                    ],
                )
            ],
            next_recommended_action="answer_open_questions",
        ),
    )
    records = [
        ConstructionQuestionResponseRecord(
            session_id=session_id,
            question_key="runtime_stack",
            gap_key="runtime_contract_incomplete",
            gap_title="Runtime incompleto",
            domain="runtime",
            question_text="Define stack runtime",
            rationale="Necesario para preparar el ACP.",
            target_owner="implementation_owner",
            status="answered",
            answer_text="python: FastAPI; database: PostgreSQL",
        )
    ]

    readiness = overlay_construction_readiness(preview, records)

    assert readiness.open_questions == 0
    assert readiness.can_start_build is False
    assert readiness.overall_status == "blocked"
    assert readiness.next_recommended_action == "resolve_blocking_construction_gaps"


def test_build_continuity_answer_map_excludes_deferred_answers() -> None:
    session_id = uuid4()
    records = [
        ConstructionQuestionResponseRecord(
            session_id=session_id,
            question_key="deployment_target",
            status="deferred",
            answer_text="Delegado a implementacion. Resolver durante la construccion con trazabilidad ACP.",
        ),
        ConstructionQuestionResponseRecord(
            session_id=session_id,
            question_key="knowledge_sources",
            status="answered",
            answer_text="name=Confluence; type=wiki; owner=ops",
        ),
    ]

    answers = build_continuity_answer_map(records)

    assert answers == {"knowledge_sources": "name=Confluence; type=wiki; owner=ops"}


def test_answered_and_deferred_questions_surface_impact_analysis() -> None:
    session_id = uuid4()
    preview = ACPPreview(
        session_id=session_id,
        validation=ACPValidationReport(
            overall_status="incomplete",
            can_export_zip=False,
            issues=[],
        ),
        construction_readiness=ConstructionReadinessReport(
            overall_status="needs_questions",
            can_start_build=False,
            blocking_gaps=0,
            open_questions=2,
            assumptions_count=0,
            gaps=[
                ConstructionGapEntry(
                    gap_key="deployment_target_unknown",
                    title="Falta decidir despliegue",
                    domain="deployment",
                    severity="warning",
                    status="open",
                    blocking_stage="package_build",
                    summary="Falta cerrar infraestructura objetivo.",
                    evidence_paths=["ACP/deployment/env.template", "ACP/runtime/config.yaml"],
                    questions=[
                        ConstructionQuestionEntry(
                            question_key="deployment_target",
                            question_text="Define la infraestructura objetivo.",
                            rationale="Necesario para empaquetar.",
                            target_owner="platform_owner",
                            blocking=True,
                        )
                    ],
                ),
                ConstructionGapEntry(
                    gap_key="knowledge_sources_missing",
                    title="Faltan fuentes",
                    domain="knowledge",
                    severity="warning",
                    status="open",
                    blocking_stage="implementation_questions",
                    summary="Faltan fuentes de conocimiento.",
                    evidence_paths=["ACP/knowledge/sources.yaml"],
                    questions=[
                        ConstructionQuestionEntry(
                            question_key="knowledge_sources",
                            question_text="Que fuentes consultara?",
                            rationale="Necesario para RAG.",
                            target_owner="domain_owner",
                            blocking=False,
                        )
                    ],
                ),
            ],
            next_recommended_action="answer_open_questions",
        ),
    )
    records = [
        ConstructionQuestionResponseRecord(
            session_id=session_id,
            question_key="deployment_target",
            gap_key="deployment_target_unknown",
            gap_title="Falta decidir despliegue",
            domain="deployment",
            question_text="Define la infraestructura objetivo.",
            rationale="Necesario para empaquetar.",
            target_owner="platform_owner",
            blocking=True,
            status="deferred",
            answer_text="Delegado a implementacion. Resolver durante la construccion con trazabilidad ACP.",
            impacted_artifacts=["ACP/deployment/env.template", "ACP/runtime/config.yaml"],
        ),
        ConstructionQuestionResponseRecord(
            session_id=session_id,
            question_key="knowledge_sources",
            gap_key="knowledge_sources_missing",
            gap_title="Faltan fuentes",
            domain="knowledge",
            question_text="Que fuentes consultara?",
            rationale="Necesario para RAG.",
            target_owner="domain_owner",
            blocking=False,
            status="answered",
            answer_text="name=Confluence; type=wiki; owner=ops",
            impacted_artifacts=["ACP/knowledge/sources.yaml"],
        ),
    ]

    questions = build_construction_question_views(preview, records)
    delegated = next(item for item in questions if item.question_key == "deployment_target")
    answered = next(item for item in questions if item.question_key == "knowledge_sources")

    assert delegated.impact_analysis is not None
    assert delegated.impact_analysis.impact_kind == "delegated_to_implementation"
    assert delegated.impact_analysis.reprocess_decision == "delegated_to_implementation"
    assert delegated.impact_analysis.reconciliation_decision == "delegated_to_implementation"
    assert "package" in delegated.impact_analysis.affected_stage_keys

    assert answered.impact_analysis is not None
    assert answered.impact_analysis.impact_kind == "localized_impact"
    assert answered.impact_analysis.reprocess_decision == "localized_reconciliation"
    assert answered.impact_analysis.reconciliation_decision == "localized_reconciliation"
    assert "implementation_questions" in answered.impact_analysis.affected_phase_keys


def test_deferred_uncertainty_backlog_travels_to_acp_without_blocking_package() -> None:
    workspace_id = uuid4()
    session_id = uuid4()
    with _memory_db() as db:
        backlog = UncertaintyBacklogRecord(
            workspace_id=workspace_id,
            session_id=session_id,
            uncertainty_key="document_provider_decision",
            product_mode="basic_free",
            source_stage="design",
            target_stage="acp",
            kind="question",
            disposition="defer",
            status="deferred",
            title="Confirmar proveedor documental",
            description="Confirmar proveedor documental y credenciales durante la implementacion.",
            reason="Depende del entorno real de implementacion.",
            impact="Afecta memoria, RAG y permisos.",
            assumed_answer="Documentos aprobados disponibles por API o export controlado.",
            suggested_answer="Usar repositorio documental centralizado con permisos por rol.",
            affected_deliverable_keys=["ACP/knowledge/sources.yaml", "ACP/memory/strategy.yaml"],
            dependency_keys=["memory.knowledge_sources"],
        )
        db.add(backlog)
        backlog_question_key = uncertainty_backlog_question_key(backlog)
        db.commit()

        records = load_uncertainty_backlog_records(db, session_id)
        gaps = build_construction_gaps_from_uncertainty_backlog(records)
        preview_records = merge_construction_question_records_with_uncertainty_backlog([], records)
        preview = append_construction_readiness_gaps(
            ACPPreview(
                session_id=session_id,
                validation=ACPValidationReport(overall_status="complete", can_export_zip=True),
                construction_readiness=ConstructionReadinessReport(
                    overall_status="ready_to_build",
                    can_start_build=True,
                    blocking_gaps=0,
                    open_questions=0,
                    assumptions_count=0,
                    gaps=[],
                    next_recommended_action="start_agentic_build",
                ),
            ),
            gaps,
        )

        questions = build_construction_question_views(preview, preview_records)
        deferred = build_deferred_construction_decision_backlog(preview, preview_records)
        continuity_answers = build_continuity_answer_map(preview_records)

    assert preview.construction_readiness.can_start_build is True
    assert preview.construction_readiness.open_questions == 0
    assert questions[0].question_key == backlog_question_key
    assert questions[0].status == "deferred"
    assert questions[0].answer_text.startswith("Documentos aprobados")
    assert deferred[0]["question_key"] == backlog_question_key
    assert continuity_answers == {}


def test_blocking_uncertainty_backlog_blocks_acp_until_answer_updates_original_record() -> None:
    workspace_id = uuid4()
    session_id = uuid4()
    user = UserRecord(email="acp-owner@example.com", full_name="ACP Owner")
    with _memory_db() as db:
        backlog = UncertaintyBacklogRecord(
            workspace_id=workspace_id,
            session_id=session_id,
            uncertainty_key="side_effect_approval_gate",
            product_mode="acp_implementation",
            source_stage="tools",
            target_stage="acp",
            kind="question",
            disposition="block",
            status="open",
            title="Definir aprobacion para side effects",
            description="Toda tool con side effects debe pausar y justificar aprobacion.",
            reason="Bloquea Package porque afecta seguridad y auditoria.",
            impact="Afecta contratos, permisos y validacion.",
            affected_deliverable_keys=["ACP/tools/external/tool-create-ticket.yaml"],
        )
        db.add(backlog)
        db.commit()

        records = load_uncertainty_backlog_records(db, session_id)
        open_preview = append_construction_readiness_gaps(
            ACPPreview(
                session_id=session_id,
                validation=ACPValidationReport(overall_status="complete", can_export_zip=True),
                construction_readiness=ConstructionReadinessReport(
                    overall_status="ready_to_build",
                    can_start_build=True,
                    blocking_gaps=0,
                    open_questions=0,
                    assumptions_count=0,
                    gaps=[],
                    next_recommended_action="start_agentic_build",
                ),
            ),
            build_construction_gaps_from_uncertainty_backlog(records),
        )

        assert open_preview.construction_readiness.can_start_build is False
        assert open_preview.construction_readiness.blocking_gaps == 1
        assert open_preview.construction_readiness.open_questions == 1
        assert build_construction_question_views(open_preview, [])[0].status == "open"

        apply_uncertainty_backlog_acp_answer(
            db,
            session_id=session_id,
            backlog_id=backlog.id,
            payload=ConstructionQuestionAnswerRequest(
                answer_text="Toda accion con side effects requiere aprobacion humana y audit log.",
                owner_role="security_owner",
            ),
            current_user=user,
        )
        db.commit()

        refreshed_backlog = db.get(UncertaintyBacklogRecord, backlog.id)
        assert refreshed_backlog is not None
        assert refreshed_backlog.status == "resolved"
        assert refreshed_backlog.assumed_answer.startswith("Toda accion con side effects")
        assert db.exec(select(ConstructionQuestionResponseRecord)).all() == []

        refreshed_records = load_uncertainty_backlog_records(db, session_id)
        preview_records = merge_construction_question_records_with_uncertainty_backlog([], refreshed_records)
        ready_preview = append_construction_readiness_gaps(
            ACPPreview(
                session_id=session_id,
                validation=ACPValidationReport(overall_status="complete", can_export_zip=True),
                construction_readiness=ConstructionReadinessReport(
                    overall_status="ready_to_build",
                    can_start_build=True,
                    blocking_gaps=0,
                    open_questions=0,
                    assumptions_count=0,
                    gaps=[],
                    next_recommended_action="start_agentic_build",
                ),
            ),
            build_construction_gaps_from_uncertainty_backlog(refreshed_records),
        )
        answered = build_construction_question_views(ready_preview, preview_records)

    assert ready_preview.construction_readiness.can_start_build is True
    assert ready_preview.construction_readiness.blocking_gaps == 0
    assert ready_preview.construction_readiness.open_questions == 0
    assert answered[0].status == "resolved"
    assert answered[0].answer_text.startswith("Toda accion con side effects")
