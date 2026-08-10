from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import (
    ACPPreview,
    ArtifactStatus,
    ConstructionReadinessReport,
    EstimationActualsUpsertRequest,
    EstimationMaturityStage,
    LLMProviderKey,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceRecord,
)
from app.services.auth_service import hash_password
from app.services.estimation_calibration import (
    build_estimation_calibration_dashboard,
    persist_estimation_run,
    upsert_project_actuals,
)
from app.services.estimation_service import build_estimation_report
from app.services.workspace_bootstrap import apply_workspace_bootstrap
from tests.test_acp_generator import build_ready_snapshot


def test_estimation_calibration_builds_stage_cohorts_and_error_metrics() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        workspace = WorkspaceRecord(
            name="Calibration Workspace",
            slug=f"calibration-{uuid4().hex[:10]}",
        )
        session.add(workspace)
        session.flush()
        apply_workspace_bootstrap(session, workspace.id)
        user = UserRecord(
            email="calibration@test.local",
            full_name="Calibration Tester",
            password_hash=hash_password("LeanBuilder123!"),
            default_workspace_id=workspace.id,
        )
        session.add(user)
        session.flush()

        blueprint_session = SessionRecord(
            user_id=user.id,
            workspace_id=workspace.id,
            title="Blueprint calibration",
            status=ArtifactStatus.ready,
            current_stage=SessionStage.post_validation,
        )
        ready_session = SessionRecord(
            user_id=user.id,
            workspace_id=workspace.id,
            title="Ready calibration",
            status=ArtifactStatus.ready,
            current_stage=SessionStage.ready_for_export,
        )
        session.add(blueprint_session)
        session.add(ready_session)
        session.flush()

        blueprint_snapshot = build_ready_snapshot(session_id=blueprint_session.id)
        ready_snapshot = build_ready_snapshot(session_id=ready_session.id)
        blueprint_snapshot.session.workspace_id = workspace.id
        ready_snapshot.session.workspace_id = workspace.id

        blueprint_report = build_estimation_report(session, snapshot=blueprint_snapshot, acp_preview=None)
        ready_report = build_estimation_report(
            session,
            snapshot=ready_snapshot,
            acp_preview=ACPPreview(
                session_id=ready_snapshot.session.id,
                blueprint_version_number=1,
                construction_readiness=ConstructionReadinessReport(
                    overall_status="ready_to_build",
                    can_start_build=True,
                    blocking_gaps=0,
                    open_questions=0,
                    assumptions_count=0,
                ),
            ),
        )

        assert blueprint_report.maturity_stage == EstimationMaturityStage.blueprint
        assert ready_report.maturity_stage == EstimationMaturityStage.ready_to_build

        blueprint_run = persist_estimation_run(
            session,
            session_id=blueprint_session.id,
            blueprint_version_number=3,
            source_action="generate_estimation_report",
            estimation_report=blueprint_report,
        )
        ready_run = persist_estimation_run(
            session,
            session_id=ready_session.id,
            blueprint_version_number=4,
            source_action="generate_acp_preview",
            estimation_report=ready_report,
        )

        _, blueprint_metric = upsert_project_actuals(
            session,
            session_id=blueprint_session.id,
            current_user_id=user.id,
            payload=EstimationActualsUpsertRequest(
                estimation_run_id=blueprint_run.id,
                delivery_mode="agentic",
                actual_provider=LLMProviderKey.openai,
                actual_hours_total=blueprint_report.agentic.estimated_hours_total * 1.08,
                actual_duration_weeks=blueprint_report.agentic.estimated_duration_weeks * 1.05,
                actual_cost_total=blueprint_report.agentic.estimated_cost * 1.1,
                actual_automation_coverage_percent=max(0, blueprint_report.agentic.automation_coverage_percent - 6),
                notes="Primer corte con leves desviaciones.",
            ),
        )
        _, ready_metric = upsert_project_actuals(
            session,
            session_id=ready_session.id,
            current_user_id=user.id,
            payload=EstimationActualsUpsertRequest(
                estimation_run_id=ready_run.id,
                delivery_mode="agentic",
                actual_provider=LLMProviderKey.openai,
                actual_hours_total=ready_report.agentic.estimated_hours_total * 0.96,
                actual_duration_weeks=ready_report.agentic.estimated_duration_weeks * 1.02,
                actual_cost_total=ready_report.agentic.estimated_cost * 1.03,
                actual_automation_coverage_percent=max(0, ready_report.agentic.automation_coverage_percent - 2),
                notes="Proyecto casi dentro de banda.",
            ),
        )

        dashboard = build_estimation_calibration_dashboard(session, workspace.id)

    assert blueprint_metric.absolute_percentage_error_cost > 0
    assert ready_metric.absolute_percentage_error_cost > 0
    assert dashboard.total_runs == 2
    assert dashboard.calibrated_runs == 2
    assert dashboard.coverage_percent == 100
    assert len(dashboard.precision_by_stage) == 2
    assert any(
        item.maturity_stage == EstimationMaturityStage.blueprint and item.calibrated_runs == 1
        for item in dashboard.precision_by_stage
    )
    assert any(
        item.maturity_stage == EstimationMaturityStage.ready_to_build and item.calibrated_runs == 1
        for item in dashboard.precision_by_stage
    )
    assert dashboard.recent_projects
    assert dashboard.recent_projects[0].session_title in {"Blueprint calibration", "Ready calibration"}
