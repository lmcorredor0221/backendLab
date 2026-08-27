from __future__ import annotations

from uuid import uuid4

from app.models import (
    ACPPreview,
    ACPValidationIssue,
    ACPValidationReport,
    ConstructionGapEntry,
    ConstructionQuestionEntry,
    ConstructionQuestionResponseRecord,
    ConstructionReadinessReport,
)
from app.services.acp_continuity import overlay_construction_readiness


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

