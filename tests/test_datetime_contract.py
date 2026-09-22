from __future__ import annotations

from datetime import datetime

from app.models import GovernancePolicyRecord, RuntimeFeatureFlagRecord, WorkflowTemplateRecord, utc_now


def test_utc_now_uses_naive_utc_for_persisted_model_timestamps() -> None:
    timestamp = utc_now()

    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is None


def test_workspace_bootstrap_timestamp_columns_remain_naive() -> None:
    models = (
        RuntimeFeatureFlagRecord,
        WorkflowTemplateRecord,
        GovernancePolicyRecord,
    )

    for model in models:
        updated_at = model.__table__.c.updated_at
        assert getattr(updated_at.type, "timezone", None) is False
