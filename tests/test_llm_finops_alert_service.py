from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    LLMBudgetPeriodType,
    LLMBudgetPolicyRecord,
    LLMBudgetScopeType,
    LLMFinOpsAlertRecord,
    LLMUsageLedgerRecord,
)
from app.services.llm_finops.alert_service import LLMFinOpsAlertService
from app.services.llm_finops.budget_service import LLMBudgetService


def build_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def add_usage_record(
    db: Session,
    *,
    workspace_id: UUID,
    provider_key: str = "openai",
    model_name: str = "gpt-5.5",
    stage: str = "define",
    cost_total: float,
    fallback_used: bool = False,
    started_at: datetime = datetime(2026, 8, 13, 10, 0, 0),
) -> LLMUsageLedgerRecord:
    record = LLMUsageLedgerRecord(
        workspace_id=workspace_id,
        user_id=uuid4(),
        session_id=uuid4(),
        project_id=uuid4(),
        initiative_id=uuid4(),
        stage=stage,
        agent_key="builder",
        capability_key="define_requirements",
        provider_key=provider_key,
        model_name=model_name,
        request_id=f"req-{uuid4()}",
        status="succeeded",
        started_at=started_at,
        duration_ms=100,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cost_total=cost_total,
        currency="USD",
        fallback_used=fallback_used,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def test_budget_threshold_alerts_are_created_once_per_period_and_scope() -> None:
    db = build_session()
    workspace_id = uuid4()
    add_usage_record(db, workspace_id=workspace_id, cost_total=60)
    add_usage_record(db, workspace_id=workspace_id, cost_total=25)
    add_usage_record(db, workspace_id=uuid4(), cost_total=99)
    policy = LLMBudgetService().create_policy(
        db,
        LLMBudgetPolicyRecord(
            workspace_id=workspace_id,
            policy_key="workspace-monthly",
            scope_type=LLMBudgetScopeType.workspace,
            period_type=LLMBudgetPeriodType.monthly,
            limit_amount=100,
            threshold_percentages=[50, 80, 95, 100],
        ),
    )
    service = LLMFinOpsAlertService()

    first = service.sync_budget_threshold_alerts(db, workspace_id=workspace_id, now=datetime(2026, 8, 13, 12, 0, 0))
    second = service.sync_budget_threshold_alerts(db, workspace_id=workspace_id, now=datetime(2026, 8, 13, 12, 0, 0))
    persisted = db.exec(select(LLMFinOpsAlertRecord)).all()

    assert [item.threshold_percent for item in first] == [50.0, 80.0]
    assert [item.id for item in second] == [item.id for item in first]
    assert len(persisted) == 2
    assert all(item.workspace_id == workspace_id for item in persisted)
    assert all(item.budget_policy_id == policy.id for item in persisted)
    assert all(item.scope_type == LLMBudgetScopeType.workspace.value for item in persisted)
    assert all(item.scope_value == str(workspace_id) for item in persisted)


def test_budget_threshold_alerts_mark_hard_limit_as_critical() -> None:
    db = build_session()
    workspace_id = uuid4()
    add_usage_record(db, workspace_id=workspace_id, cost_total=120)
    LLMBudgetService().create_policy(
        db,
        LLMBudgetPolicyRecord(
            workspace_id=workspace_id,
            policy_key="workspace-hard-limit",
            scope_type=LLMBudgetScopeType.workspace,
            period_type=LLMBudgetPeriodType.monthly,
            limit_amount=100,
            threshold_percentages=[50, 80, 95, 100],
            hard_limit_percent=100,
        ),
    )

    alerts = LLMFinOpsAlertService().sync_budget_threshold_alerts(
        db,
        workspace_id=workspace_id,
        now=datetime(2026, 8, 13, 12, 0, 0),
    )

    critical = [item for item in alerts if item.threshold_percent == 100.0]
    assert len(critical) == 1
    assert critical[0].severity == "critical"
    assert critical[0].consumed_amount == 120
    assert critical[0].limit_amount == 100


def test_operational_alerts_cover_fallback_model_policy_and_high_cost_call() -> None:
    db = build_session()
    workspace_id = uuid4()
    for _ in range(3):
        add_usage_record(
            db,
            workspace_id=workspace_id,
            provider_key="openai",
            model_name="gpt-5.5",
            stage="define",
            cost_total=1,
            fallback_used=True,
        )
    expensive = add_usage_record(
        db,
        workspace_id=workspace_id,
        provider_key="deepseek",
        model_name="deepseek-v4-pro",
        stage="design",
        cost_total=15,
    )
    add_usage_record(
        db,
        workspace_id=workspace_id,
        provider_key="openai",
        model_name="experimental-model",
        stage="define",
        cost_total=0.5,
    )
    service = LLMFinOpsAlertService(fallback_count_threshold=3, high_cost_call_threshold=10)

    first = service.sync_operational_alerts(
        db,
        workspace_id=workspace_id,
        now=datetime(2026, 8, 13, 12, 0, 0),
        allowed_models={"openai:gpt-5.5", "deepseek-v4-pro"},
    )
    second = service.sync_operational_alerts(
        db,
        workspace_id=workspace_id,
        now=datetime(2026, 8, 13, 12, 0, 0),
        allowed_models={"openai:gpt-5.5", "deepseek-v4-pro"},
    )
    persisted = db.exec(select(LLMFinOpsAlertRecord)).all()

    assert {item.alert_type for item in first} == {
        "fallback_repeated",
        "high_cost_call",
        "model_outside_policy",
    }
    assert len(second) == 3
    assert len(persisted) == 3
    assert any(item.usage_record_id == expensive.id for item in persisted)
    assert any(item.scope_value == "experimental-model" for item in persisted)
    assert all(item.workspace_id == workspace_id for item in persisted)
