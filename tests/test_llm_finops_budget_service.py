from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlmodel import SQLModel, Session, create_engine

from app.models import LLMBudgetPeriodType, LLMBudgetPolicyRecord, LLMBudgetScopeType, LLMUsageLedgerRecord
from app.services.llm_finops.budget_service import LLMBudgetService, resolve_budget_period


def build_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def add_usage_record(
    db: Session,
    *,
    workspace_id: UUID,
    user_id: UUID,
    project_id: UUID,
    initiative_id: UUID,
    stage: str,
    provider_key: str,
    model_name: str,
    cost_total: float,
    started_at: datetime,
) -> None:
    db.add(
        LLMUsageLedgerRecord(
            workspace_id=workspace_id,
            user_id=user_id,
            session_id=uuid4(),
            project_id=project_id,
            initiative_id=initiative_id,
            stage=stage,
            agent_key="builder",
            capability_key="define_requirements",
            provider_key=provider_key,
            model_name=model_name,
            status="succeeded",
            started_at=started_at,
            duration_ms=100,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_total=cost_total,
            currency="USD",
        )
    )
    db.commit()


def seed_scope_records(db: Session) -> dict[str, UUID]:
    ids = {
        "workspace": uuid4(),
        "other_workspace": uuid4(),
        "user": uuid4(),
        "other_user": uuid4(),
        "project": uuid4(),
        "other_project": uuid4(),
        "initiative": uuid4(),
        "other_initiative": uuid4(),
    }
    add_usage_record(
        db,
        workspace_id=ids["workspace"],
        user_id=ids["user"],
        project_id=ids["project"],
        initiative_id=ids["initiative"],
        stage="define",
        provider_key="openai",
        model_name="gpt-5.5",
        cost_total=4.0,
        started_at=datetime(2026, 8, 13, 10, 0, 0),
    )
    add_usage_record(
        db,
        workspace_id=ids["workspace"],
        user_id=ids["other_user"],
        project_id=ids["other_project"],
        initiative_id=ids["other_initiative"],
        stage="design",
        provider_key="deepseek",
        model_name="deepseek-v4-pro",
        cost_total=7.0,
        started_at=datetime(2026, 8, 13, 11, 0, 0),
    )
    add_usage_record(
        db,
        workspace_id=ids["other_workspace"],
        user_id=ids["user"],
        project_id=ids["project"],
        initiative_id=ids["initiative"],
        stage="define",
        provider_key="openai",
        model_name="gpt-5.5",
        cost_total=99.0,
        started_at=datetime(2026, 8, 13, 12, 0, 0),
    )
    return ids


@pytest.mark.parametrize(
    ("scope_type", "policy_kwargs", "expected_cost"),
    [
        (LLMBudgetScopeType.workspace, {}, 11.0),
        (LLMBudgetScopeType.user, {"user_id": "user"}, 4.0),
        (LLMBudgetScopeType.project, {"project_id": "project"}, 4.0),
        (LLMBudgetScopeType.initiative, {"initiative_id": "initiative"}, 4.0),
        (LLMBudgetScopeType.stage, {"stage": "define"}, 4.0),
        (LLMBudgetScopeType.provider, {"provider_key": "openai"}, 4.0),
        (LLMBudgetScopeType.model, {"model_name": "gpt-5.5"}, 4.0),
    ],
)
def test_budget_service_evaluates_supported_scopes(scope_type, policy_kwargs, expected_cost) -> None:
    db = build_session()
    ids = seed_scope_records(db)
    resolved_kwargs = {
        key: ids[value] if value in ids else value
        for key, value in policy_kwargs.items()
    }
    policy = LLMBudgetPolicyRecord(
        workspace_id=ids["workspace"],
        policy_key=f"policy-{scope_type.value}",
        scope_type=scope_type,
        period_type=LLMBudgetPeriodType.monthly,
        limit_amount=20,
        threshold_percentages=[50, 80, 95, 100],
        **resolved_kwargs,
    )

    evaluation = LLMBudgetService().evaluate_policy(db, policy, now=datetime(2026, 8, 13, 15, 0, 0))

    assert evaluation.consumed_amount == expected_cost
    assert evaluation.workspace_id == ids["workspace"]
    assert evaluation.period_start == datetime(2026, 8, 1, 0, 0, 0)
    assert evaluation.period_end == datetime(2026, 9, 1, 0, 0, 0)


def test_budget_policy_can_be_created_and_reports_warning_threshold() -> None:
    db = build_session()
    ids = seed_scope_records(db)
    service = LLMBudgetService()

    policy = service.create_policy(
        db,
        LLMBudgetPolicyRecord(
            workspace_id=ids["workspace"],
            scope_type=LLMBudgetScopeType.workspace,
            period_type=LLMBudgetPeriodType.monthly,
            limit_amount=20,
            threshold_percentages=[50, 80, 100],
        ),
    )
    evaluation = service.evaluate_policy(db, policy, now=datetime(2026, 8, 13, 15, 0, 0))

    assert policy.scope_value == str(ids["workspace"])
    assert policy.currency == "USD"
    assert evaluation.status == "warning"
    assert evaluation.percent_consumed == 55.0
    assert evaluation.thresholds_reached == [50.0]
    assert evaluation.next_threshold_percent == 80.0
    assert evaluation.remaining_amount == 9.0


def test_budget_service_identifies_hard_limit() -> None:
    db = build_session()
    ids = seed_scope_records(db)
    policy = LLMBudgetPolicyRecord(
        workspace_id=ids["workspace"],
        policy_key="openai-hard-limit",
        scope_type=LLMBudgetScopeType.provider,
        provider_key="openai",
        period_type=LLMBudgetPeriodType.monthly,
        limit_amount=3,
        threshold_percentages=[50, 100],
        hard_limit_percent=100,
    )

    evaluation = LLMBudgetService().evaluate_policy(db, policy, now=datetime(2026, 8, 13, 15, 0, 0))

    assert evaluation.status == "hard_limit"
    assert evaluation.is_hard_limit is True
    assert evaluation.percent_consumed == 133.3333
    assert evaluation.thresholds_reached == [50.0, 100.0]
    assert evaluation.remaining_amount == 0


def test_budget_periods_exclude_usage_outside_window() -> None:
    db = build_session()
    ids = seed_scope_records(db)
    add_usage_record(
        db,
        workspace_id=ids["workspace"],
        user_id=ids["user"],
        project_id=ids["project"],
        initiative_id=ids["initiative"],
        stage="define",
        provider_key="openai",
        model_name="gpt-5.5",
        cost_total=12.0,
        started_at=datetime(2026, 7, 31, 23, 0, 0),
    )
    policy = LLMBudgetPolicyRecord(
        workspace_id=ids["workspace"],
        policy_key="daily-budget",
        scope_type=LLMBudgetScopeType.workspace,
        period_type=LLMBudgetPeriodType.daily,
        limit_amount=100,
    )

    evaluation = LLMBudgetService().evaluate_policy(db, policy, now=datetime(2026, 8, 13, 15, 0, 0))

    assert evaluation.consumed_amount == 11.0
    assert resolve_budget_period(policy, now=datetime(2026, 8, 13, 15, 0, 0)) == (
        datetime(2026, 8, 13, 0, 0, 0),
        datetime(2026, 8, 14, 0, 0, 0),
    )


def test_custom_budget_period_uses_explicit_bounds() -> None:
    db = build_session()
    ids = seed_scope_records(db)
    policy = LLMBudgetPolicyRecord(
        workspace_id=ids["workspace"],
        policy_key="custom-budget",
        scope_type=LLMBudgetScopeType.workspace,
        period_type=LLMBudgetPeriodType.custom,
        custom_period_start=datetime(2026, 8, 13, 10, 30, 0),
        custom_period_end=datetime(2026, 8, 13, 11, 30, 0),
        limit_amount=10,
    )

    evaluation = LLMBudgetService().evaluate_policy(db, policy, now=datetime(2026, 8, 13, 15, 0, 0))

    assert evaluation.consumed_amount == 7.0
    assert evaluation.call_count == 1
