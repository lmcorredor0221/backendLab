from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import SQLModel, Session, create_engine

from app.models import LLMUsageLedgerRecord
from app.services.llm_finops.analytics_service import LLMUsageAnalyticsFilters, LLMUsageAnalyticsService


def build_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def add_record(
    db: Session,
    *,
    user_id,
    project_id,
    stage: str,
    agent_key: str,
    capability_key: str,
    provider_key: str,
    model_name: str,
    cost_total: float,
    total_tokens: int,
    status: str = "succeeded",
    duration_ms: int = 100,
    retry_count: int = 0,
    fallback_used: bool = False,
    workspace_id=None,
    started_at: datetime | None = None,
    currency: str = "USD",
    usage_is_estimated: bool = False,
) -> LLMUsageLedgerRecord:
    record = LLMUsageLedgerRecord(
        workspace_id=workspace_id or uuid4(),
        user_id=user_id,
        session_id=uuid4(),
        project_id=project_id,
        stage=stage,
        agent_key=agent_key,
        capability_key=capability_key,
        provider_key=provider_key,
        model_name=model_name,
        status=status,
        started_at=started_at or datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc),
        duration_ms=duration_ms,
        input_tokens=total_tokens // 2,
        output_tokens=total_tokens - (total_tokens // 2),
        total_tokens=total_tokens,
        cost_total=cost_total,
        currency=currency,
        retry_count=retry_count,
        fallback_used=fallback_used,
        other_token_metrics={"usage_is_estimated": usage_is_estimated},
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def seed_records(db: Session):
    user_a = uuid4()
    user_b = uuid4()
    project_a = uuid4()
    project_b = uuid4()
    add_record(
        db,
        user_id=user_a,
        project_id=project_a,
        stage="define",
        agent_key="builder",
        capability_key="define_requirements",
        provider_key="openai",
        model_name="gpt-5.5",
        cost_total=1.5,
        total_tokens=1000,
        duration_ms=100,
    )
    add_record(
        db,
        user_id=user_a,
        project_id=project_b,
        stage="design",
        agent_key="critic",
        capability_key="critique_agent_design",
        provider_key="deepseek",
        model_name="deepseek-v4-pro",
        cost_total=0.5,
        total_tokens=500,
        status="failed",
        duration_ms=300,
        retry_count=1,
        fallback_used=True,
    )
    add_record(
        db,
        user_id=user_b,
        project_id=project_a,
        stage="define",
        agent_key="builder",
        capability_key="define_requirements",
        provider_key="openai",
        model_name="gpt-5.5",
        cost_total=2.0,
        total_tokens=2000,
        duration_ms=200,
    )
    return user_a, user_b, project_a, project_b


def test_analytics_summary_returns_core_totals_and_efficiency_metrics() -> None:
    db = build_session()
    seed_records(db)

    summary = LLMUsageAnalyticsService().summarize(db)

    assert summary["call_count"] == 3
    assert summary["cost_total"] == 4.0
    assert summary["total_tokens"] == 3500
    assert summary["cost_per_call"] == 1.33333333
    assert summary["avg_latency_ms"] == 200
    assert summary["p95_latency_ms"] == 300
    assert summary["error_count"] == 1
    assert summary["retry_count"] == 1
    assert summary["fallback_count"] == 1


def test_analytics_filters_can_be_combined() -> None:
    db = build_session()
    user_a, _, _, _ = seed_records(db)
    service = LLMUsageAnalyticsService()

    user_summary = service.summarize(db, LLMUsageAnalyticsFilters(user_id=user_a))
    define_openai_summary = service.summarize(
        db,
        LLMUsageAnalyticsFilters(stage="define", provider_key="openai"),
    )

    assert user_summary["call_count"] == 2
    assert user_summary["cost_total"] == 2.0
    assert user_summary["total_tokens"] == 1500
    assert define_openai_summary["call_count"] == 2
    assert define_openai_summary["cost_total"] == 3.5


def test_analytics_top_consumers_and_provider_breakdown_are_ranked() -> None:
    db = build_session()
    user_a, user_b, _, _ = seed_records(db)
    service = LLMUsageAnalyticsService()

    top_users = service.top_consumers(db, dimension="user_id")
    provider_breakdown = service.provider_breakdown(db)

    assert top_users[0]["key"] == str(user_b)
    assert top_users[0]["cost_total"] == 2.0
    assert top_users[1]["key"] == str(user_a)
    assert provider_breakdown[0]["provider_key"] == "openai"
    assert provider_breakdown[0]["model_name"] == "gpt-5.5"
    assert provider_breakdown[0]["cost_total"] == 3.5
    assert provider_breakdown[1]["provider_key"] == "deepseek"


def test_analytics_timeseries_groups_usage_by_day_with_operational_metrics() -> None:
    db = build_session()
    workspace_id = uuid4()
    user_id = uuid4()
    project_id = uuid4()
    add_record(
        db,
        workspace_id=workspace_id,
        user_id=user_id,
        project_id=project_id,
        stage="define",
        agent_key="builder",
        capability_key="define_requirements",
        provider_key="openai",
        model_name="gpt-5.5",
        cost_total=1.0,
        total_tokens=1000,
        started_at=datetime(2026, 8, 13, 10, 0, 0, tzinfo=timezone.utc),
        retry_count=1,
        usage_is_estimated=True,
    )
    add_record(
        db,
        workspace_id=workspace_id,
        user_id=user_id,
        project_id=project_id,
        stage="define",
        agent_key="builder",
        capability_key="define_requirements",
        provider_key="openai",
        model_name="gpt-5.5",
        cost_total=2.0,
        total_tokens=1500,
        status="failed",
        started_at=datetime(2026, 8, 13, 15, 0, 0, tzinfo=timezone.utc),
        fallback_used=True,
    )
    add_record(
        db,
        workspace_id=workspace_id,
        user_id=user_id,
        project_id=project_id,
        stage="design",
        agent_key="critic",
        capability_key="critique_agent_design",
        provider_key="deepseek",
        model_name="deepseek-v4-pro",
        cost_total=0.5,
        total_tokens=500,
        started_at=datetime(2026, 8, 14, 9, 0, 0, tzinfo=timezone.utc),
    )

    rows = LLMUsageAnalyticsService().timeseries(
        db,
        LLMUsageAnalyticsFilters(workspace_id=workspace_id),
        granularity="day",
    )

    assert len(rows) == 2
    assert rows[0]["call_count"] == 2
    assert rows[0]["cost_total"] == 3.0
    assert rows[0]["total_tokens"] == 2500
    assert rows[0]["error_count"] == 1
    assert rows[0]["retry_count"] == 1
    assert rows[0]["fallback_count"] == 1
    assert rows[0]["estimated_count"] == 1
    assert rows[0]["currency"] == "USD"
    assert rows[1]["call_count"] == 1
