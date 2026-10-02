from uuid import uuid4

from app.models import (
    ApprovedToolsDigest,
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintTool,
    ReviewState,
    SessionCreateResponse,
    SessionSnapshot,
    SessionStage,
    ToolRecommendationArtifact,
    ToolRequirementCoverageEntry,
    utc_now,
)
from app.services.blueprint_consistency_service import build_blueprint_consistency_report


def test_policy_scoring_gap_is_ignored_when_approved_digest_covers_requirement() -> None:
    now = utc_now()
    snapshot = SessionSnapshot(
        session=SessionCreateResponse(
            id=uuid4(),
            title="Caso scoring ICP",
            status=ArtifactStatus.ready,
            current_stage=SessionStage.ready_for_export,
            created_at=now,
            updated_at=now,
        ),
        blueprint=BlueprintArtifact(
            tools=[
                BlueprintTool(
                    name="business_policy_evaluation",
                    purpose="Evaluar reglas de negocio, scoring y priorizacion contra ICP.",
                )
            ],
        ),
        latest_tool_recommendation=ToolRecommendationArtifact(
            review_state=ReviewState.complete,
            requirements_coverage=[
                ToolRequirementCoverageEntry(
                    requirement_key="RF-002",
                    requirement_title="Scoring de priorizacion de cuentas contra ICP",
                    category="functional",
                    priority="high",
                    coverage_status="gap",
                    covered_by_tool_keys=[],
                    rationale="No se detecto una tool del shortlist que cubra explicitamente el scoring ICP.",
                    source_refs=["tool_recommendation.requirements_coverage"],
                )
            ],
            approved_tools_digest=ApprovedToolsDigest(
                tool_count=1,
                approved_tool_keys=["business_policy_evaluation"],
                mandatory_tool_keys=["business_policy_evaluation"],
                selected_blueprint_tool_names=["business_policy_evaluation"],
            ),
        ),
    )

    report = build_blueprint_consistency_report(snapshot)

    assert "RF-002" not in report.uncovered_requirement_keys
    assert all(item.issue_key != "tools_requirement_gap:RF-002" for item in report.issues)
    assert report.overall_status != ReviewState.blocked

