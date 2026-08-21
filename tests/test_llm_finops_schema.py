from sqlalchemy import JSON

from app.models import LLMUsageLedgerRecord


def test_llm_usage_ledger_table_contains_finops_columns() -> None:
    table = LLMUsageLedgerRecord.__table__

    assert table.name == "llm_usage_ledger"
    for column_name in {
        "workspace_id",
        "user_id",
        "session_id",
        "project_id",
        "initiative_id",
        "stage",
        "capability_key",
        "provider_key",
        "model_name",
        "request_id",
        "attempt_number",
        "status",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "cost_input",
        "cost_output",
        "cost_other",
        "cost_total",
        "currency",
        "pricing_profile_key",
        "pricing_snapshot",
        "usage_raw_redacted",
    }:
        assert column_name in table.c


def test_llm_usage_ledger_json_columns_are_structured() -> None:
    table = LLMUsageLedgerRecord.__table__

    assert isinstance(table.c.other_token_metrics.type, JSON)
    assert isinstance(table.c.provider_metrics.type, JSON)
    assert isinstance(table.c.pricing_snapshot.type, JSON)
    assert isinstance(table.c.usage_raw_redacted.type, JSON)
    assert isinstance(table.c.metadata.type, JSON)


def test_llm_usage_ledger_declares_query_indexes() -> None:
    index_names = {index.name for index in LLMUsageLedgerRecord.__table__.indexes}

    assert "ix_llm_usage_ledger_workspace_started" in index_names
    assert "ix_llm_usage_ledger_user_started" in index_names
    assert "ix_llm_usage_ledger_session_started" in index_names
    assert "ix_llm_usage_ledger_stage_capability_started" in index_names
    assert "ix_llm_usage_ledger_provider_model_started" in index_names
    assert "ix_llm_usage_ledger_request_attempt" in index_names
