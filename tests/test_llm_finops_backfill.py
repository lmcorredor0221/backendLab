from __future__ import annotations

import json

from sqlmodel import SQLModel, Session, create_engine, select

from app.models import LLMUsageLedgerRecord
from scripts.backfill_llm_finops_from_runtime_audits import backfill_llm_finops_from_runtime_audits


def build_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def build_cli_audit(**overrides):
    payload = {
        "run_id": "codex-run-1",
        "task_kind": "define_requirements",
        "status": "succeeded",
        "requested_model": "gpt-5.5",
        "selected_model": "gpt-5.4-mini",
        "attempted_models": ["gpt-5.5", "gpt-5.4-mini"],
        "fallback_used": True,
        "started_at": "2026-08-13T10:00:00+00:00",
        "finished_at": "2026-08-13T10:00:03+00:00",
        "returncode": 0,
        "prompt_text": "no debe persistirse completo",
        "response_text": "tampoco debe persistirse completo",
        "attempts": [{"attempt_number": 1}, {"attempt_number": 2}],
        "metrics": {
            "duration_ms": 3000,
            "queue_wait_ms": 25,
            "output_estimated_tokens": 32,
            "output_size_bytes": 128,
            "prompt_estimated_tokens": 64,
            "stderr_bytes": 0,
            "stdout_bytes": 256,
            "exit_code": 0,
        },
        "metadata": {
            "agent_key": "requirements_builder",
            "stage": "define",
        },
    }
    payload.update(overrides)
    return payload


def write_jsonl(path, *payloads) -> None:
    path.write_text("\n".join(json.dumps(payload, ensure_ascii=True) for payload in payloads) + "\n", encoding="utf-8")


def test_backfill_noops_when_audit_logs_are_missing(tmp_path) -> None:
    db = build_session()

    summary = backfill_llm_finops_from_runtime_audits(
        db,
        codex_audit_path=tmp_path / "runtime-audit.jsonl",
        antigravity_audit_path=tmp_path / "runtime-audit-agy.jsonl",
    )
    records = db.exec(select(LLMUsageLedgerRecord)).all()

    assert summary.files_missing == 2
    assert summary.records_created == 0
    assert summary.errors == []
    assert records == []


def test_backfill_imports_cli_audits_and_is_idempotent_by_run_id(tmp_path) -> None:
    db = build_session()
    codex_path = tmp_path / "runtime-audit.jsonl"
    agy_path = tmp_path / "runtime-audit-agy.jsonl"
    write_jsonl(codex_path, build_cli_audit())
    write_jsonl(
        agy_path,
        build_cli_audit(
            run_id="agy-run-1",
            requested_model="gemini-3.6-flash",
            selected_model="gemini-3.6-pro",
            attempted_models=["gemini-3.6-flash", "gemini-3.6-pro"],
        ),
    )

    first = backfill_llm_finops_from_runtime_audits(
        db,
        codex_audit_path=codex_path,
        antigravity_audit_path=agy_path,
    )
    second = backfill_llm_finops_from_runtime_audits(
        db,
        codex_audit_path=codex_path,
        antigravity_audit_path=agy_path,
    )
    records = db.exec(select(LLMUsageLedgerRecord).order_by(LLMUsageLedgerRecord.provider_key)).all()

    assert first.files_seen == 2
    assert first.records_created == 2
    assert first.by_provider == {"antigravity_cli": 1, "codex_local": 1}
    assert second.records_created == 0
    assert second.duplicates == 2
    assert len(records) == 2
    assert {record.request_id for record in records} == {"agy-run-1", "codex-run-1"}
    assert all(record.metadata_payload["backfilled"] is True for record in records)
    assert all(record.metadata_payload["cost_is_estimated"] is True for record in records)
    assert all(record.other_token_metrics["usage_is_estimated"] is True for record in records)
    assert all("prompt_text" not in record.usage_raw_redacted for record in records)
    assert all("response_text" not in record.usage_raw_redacted for record in records)
    assert records[0].provider_key == "antigravity_cli"
    assert records[0].model_name == "gemini-3.6-pro"
    assert records[1].provider_key == "codex_local"
    assert records[1].retry_count == 1
    assert records[1].total_tokens == 96


def test_backfill_records_failed_audit_and_skips_invalid_lines(tmp_path) -> None:
    db = build_session()
    codex_path = tmp_path / "runtime-audit.jsonl"
    codex_path.write_text(
        "{bad json}\n"
        + json.dumps(
            build_cli_audit(
                run_id="codex-run-failed",
                status="failed",
                returncode=1,
                error_code="auth_error",
                error="provider failed with api_key=sk-super-secret-value",
                selected_model="gpt-5.5",
                attempted_models=["gpt-5.5"],
                fallback_used=False,
                attempts=[{"attempt_number": 1, "error_message": "auth failed"}],
            ),
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = backfill_llm_finops_from_runtime_audits(
        db,
        codex_audit_path=codex_path,
        antigravity_audit_path=tmp_path / "runtime-audit-agy.jsonl",
    )
    record = db.exec(select(LLMUsageLedgerRecord)).one()

    assert summary.invalid_lines == 1
    assert summary.records_created == 1
    assert record.status == "failed"
    assert record.failure_kind == "auth_error"
    assert "sk-super-secret-value" not in record.failure_detail_redacted
    assert "[REDACTED]" in record.failure_detail_redacted
