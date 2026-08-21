from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import (
    AgentExecutionBackend,
    LLMPricingProfile,
    LLMProviderKey,
    LLMUsageLedgerRecord,
    RuntimeCatalogEntryRecord,
)
from app.services.llm_finops.contracts import LLMCallContext, LLMCallStatus, LLMUsageRecordInput
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_finops.usage_normalization import normalize_cli_usage


SENSITIVE_AUDIT_KEYS = {
    "content",
    "input",
    "last_message",
    "messages",
    "output",
    "prompt",
    "prompt_text",
    "response",
    "response_text",
    "stderr",
    "stdout",
    "structured_output",
}


@dataclass(frozen=True)
class RuntimeAuditSource:
    provider_key: LLMProviderKey
    execution_backend: str
    runtime_name: str
    path: Path


@dataclass
class RuntimeAuditBackfillSummary:
    files_seen: int = 0
    files_missing: int = 0
    lines_seen: int = 0
    invalid_lines: int = 0
    records_created: int = 0
    duplicates: int = 0
    records_failed: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_seen": self.files_seen,
            "files_missing": self.files_missing,
            "lines_seen": self.lines_seen,
            "invalid_lines": self.invalid_lines,
            "records_created": self.records_created,
            "duplicates": self.duplicates,
            "records_failed": self.records_failed,
            "by_provider": dict(sorted(self.by_provider.items())),
            "errors": list(self.errors),
        }


def backfill_llm_finops_from_runtime_audits(
    session: Session,
    *,
    audit_root: Path | str | None = None,
    codex_audit_path: Path | str | None = None,
    antigravity_audit_path: Path | str | None = None,
    ledger_service: LLMUsageLedgerService | None = None,
) -> RuntimeAuditBackfillSummary:
    sources = resolve_runtime_audit_sources(
        audit_root=audit_root,
        codex_audit_path=codex_audit_path,
        antigravity_audit_path=antigravity_audit_path,
    )
    service = ledger_service or LLMUsageLedgerService(pricing_profiles=load_pricing_profiles(session))
    summary = RuntimeAuditBackfillSummary()

    for source in sources:
        _backfill_source(session, source=source, ledger_service=service, summary=summary)

    return summary


def resolve_runtime_audit_sources(
    *,
    audit_root: Path | str | None = None,
    codex_audit_path: Path | str | None = None,
    antigravity_audit_path: Path | str | None = None,
) -> list[RuntimeAuditSource]:
    root = Path(audit_root).resolve() if audit_root is not None else _default_audit_root()
    codex_path = Path(codex_audit_path).resolve() if codex_audit_path is not None else _resolve_first_candidate(
        root / "codex-workspaces" / "runtime-audit.jsonl",
        root / "runtime-audit.jsonl",
    )
    agy_path = Path(antigravity_audit_path).resolve() if antigravity_audit_path is not None else _resolve_first_candidate(
        root / "agy-workspaces" / "runtime-audit-agy.jsonl",
        root / "runtime-audit-agy.jsonl",
    )
    return [
        RuntimeAuditSource(
            provider_key=LLMProviderKey.codex_local,
            execution_backend=AgentExecutionBackend.codex_cli.value,
            runtime_name="codex_cli",
            path=codex_path,
        ),
        RuntimeAuditSource(
            provider_key=LLMProviderKey.antigravity_cli,
            execution_backend=AgentExecutionBackend.antigravity_cli.value,
            runtime_name="antigravity_cli",
            path=agy_path,
        ),
    ]


def load_pricing_profiles(session: Session) -> dict[LLMProviderKey, list[LLMPricingProfile]]:
    rows = session.exec(
        select(RuntimeCatalogEntryRecord)
        .where(
            RuntimeCatalogEntryRecord.catalog_key == "estimation_pricing_profiles",
            RuntimeCatalogEntryRecord.is_active == True,  # noqa: E712
        )
        .order_by(RuntimeCatalogEntryRecord.order_index.asc())
    ).all()
    grouped: dict[LLMProviderKey, list[LLMPricingProfile]] = {}
    for row in rows:
        try:
            payload = {key: value for key, value in row.payload.items() if key in LLMPricingProfile.model_fields}
            profile = LLMPricingProfile.model_validate(payload)
        except Exception:
            continue
        grouped.setdefault(profile.provider, []).append(profile)
    return grouped


def _backfill_source(
    session: Session,
    *,
    source: RuntimeAuditSource,
    ledger_service: LLMUsageLedgerService,
    summary: RuntimeAuditBackfillSummary,
) -> None:
    if not source.path.exists():
        summary.files_missing += 1
        return
    summary.files_seen += 1

    for line_number, line in enumerate(source.path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        summary.lines_seen += 1
        try:
            audit = json.loads(stripped)
        except json.JSONDecodeError as exc:
            summary.invalid_lines += 1
            _append_error(summary, f"{source.path}:{line_number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(audit, dict):
            summary.invalid_lines += 1
            _append_error(summary, f"{source.path}:{line_number}: JSON line is not an object")
            continue

        try:
            result = _record_audit(session, source=source, audit=audit, line_number=line_number, ledger_service=ledger_service)
        except Exception as exc:
            summary.records_failed += 1
            _append_error(summary, f"{source.path}:{line_number}: {type(exc).__name__}: {exc}")
            continue

        provider = source.provider_key.value
        if result == "duplicate":
            summary.duplicates += 1
        else:
            summary.records_created += 1
            summary.by_provider[provider] = summary.by_provider.get(provider, 0) + 1


def _record_audit(
    session: Session,
    *,
    source: RuntimeAuditSource,
    audit: dict[str, Any],
    line_number: int,
    ledger_service: LLMUsageLedgerService,
) -> str:
    run_id = str(audit.get("run_id") or audit.get("request_id") or "").strip()
    if not run_id:
        raise ValueError("runtime audit line does not include run_id")
    if _existing_usage_record(session, provider_key=source.provider_key.value, request_id=run_id) is not None:
        return "duplicate"

    sanitized_audit = _sanitize_audit_payload(audit)
    usage = normalize_cli_usage(sanitized_audit)
    metrics = sanitized_audit.get("metrics", {}) if isinstance(sanitized_audit.get("metrics"), dict) else {}
    task_kind = _string_field(sanitized_audit, "task_kind")
    selected_model = _string_field(sanitized_audit, "selected_model")
    requested_model = _string_field(sanitized_audit, "requested_model")
    attempts = sanitized_audit.get("attempts", [])
    attempted_models = sanitized_audit.get("attempted_models", [])
    retry_count = _retry_count(attempts=attempts, attempted_models=attempted_models)

    record_input = LLMUsageRecordInput(
        context=LLMCallContext(
            workspace_id=_uuid_field(sanitized_audit, "workspace_id"),
            user_id=_uuid_field(sanitized_audit, "user_id"),
            session_id=_uuid_field(sanitized_audit, "session_id"),
            project_id=_uuid_field(sanitized_audit, "project_id"),
            initiative_id=_uuid_field(sanitized_audit, "initiative_id"),
            stage=_metadata_string(sanitized_audit, "stage") or _stage_from_task_kind(task_kind),
            substage=_metadata_string(sanitized_audit, "substage"),
            agent_key=_metadata_string(sanitized_audit, "agent_key"),
            capability_key=_metadata_string(sanitized_audit, "capability_key") or task_kind,
            action_key=_metadata_string(sanitized_audit, "action_key") or task_kind,
            operation_id=_uuid_field(sanitized_audit, "operation_id"),
            parent_run_id=_string_field(sanitized_audit, "parent_run_id"),
            execution_mode=_metadata_string(sanitized_audit, "execution_mode") or "backfill",
            correlation_id=_string_field(sanitized_audit, "correlation_id") or run_id,
            source="runtime_audit_backfill",
            metadata={
                "audit_line_number": line_number,
                "audit_runtime": source.runtime_name,
                "backfilled": True,
                "cost_is_estimated": usage.usage_is_estimated,
                "source_audit_path": str(source.path),
                "task_kind": task_kind,
                "workspace_root": _string_field(sanitized_audit, "workspace_root"),
            },
        ),
        provider_key=source.provider_key.value,
        model_name=selected_model or requested_model,
        requested_model=requested_model or selected_model,
        execution_backend=source.execution_backend,
        execution_mode="backfill",
        request_id=run_id,
        attempt_number=1,
        retry_count=retry_count,
        fallback_used=bool(sanitized_audit.get("fallback_used", False)),
        status=_status_from_audit(sanitized_audit),
        failure_kind=_failure_kind_from_audit(sanitized_audit),
        failure_detail=_failure_detail_from_audit(sanitized_audit),
        started_at=_datetime_field(sanitized_audit, "started_at"),
        finished_at=_datetime_field(sanitized_audit, "finished_at"),
        duration_ms=_int_field(metrics, "duration_ms") or _int_field(sanitized_audit, "duration_ms"),
        queue_wait_ms=_int_field(metrics, "queue_wait_ms") or _int_field(sanitized_audit, "queue_wait_ms"),
        usage=usage,
        finish_reason=_string_field(sanitized_audit, "status"),
        metadata={
            "attempted_models": attempted_models if isinstance(attempted_models, list) else [],
            "backfilled": True,
            "cost_is_estimated": usage.usage_is_estimated,
            "error_code": _string_field(sanitized_audit, "error_code"),
            "recoverable": bool(sanitized_audit.get("recoverable", False)),
            "runtime": source.runtime_name,
        },
    )
    record_result = ledger_service.record_call(session, record_input)
    return "duplicate" if record_result.duplicate else "created"


def _existing_usage_record(session: Session, *, provider_key: str, request_id: str) -> LLMUsageLedgerRecord | None:
    return session.exec(
        select(LLMUsageLedgerRecord).where(
            LLMUsageLedgerRecord.provider_key == provider_key,
            LLMUsageLedgerRecord.request_id == request_id,
            LLMUsageLedgerRecord.attempt_number == 1,
        )
    ).first()


def _sanitize_audit_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_audit_payload(item)
            for key, item in value.items()
            if str(key) not in SENSITIVE_AUDIT_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_audit_payload(item) for item in value]
    return value


def _status_from_audit(audit: dict[str, Any]) -> LLMCallStatus:
    status = _string_field(audit, "status").lower()
    error_code = _string_field(audit, "error_code").lower()
    returncode = audit.get("returncode")
    if status in {"succeeded", "success", "completed", "ok"}:
        return LLMCallStatus.succeeded
    if status in {"cancelled", "canceled"}:
        return LLMCallStatus.cancelled
    if status in {"provider_unavailable", "unavailable"}:
        return LLMCallStatus.provider_unavailable
    if status in {"schema_invalid", "invalid_schema"} or error_code == "invalid_schema":
        return LLMCallStatus.schema_invalid
    if status == "timeout" or error_code == "timeout":
        return LLMCallStatus.timeout
    if status == "retry":
        return LLMCallStatus.retry
    if status == "failed" or audit.get("error") or (returncode not in (None, 0, "0")):
        return LLMCallStatus.failed
    return LLMCallStatus.succeeded


def _failure_kind_from_audit(audit: dict[str, Any]) -> str:
    status = _status_from_audit(audit)
    if status == LLMCallStatus.succeeded:
        return ""
    return _string_field(audit, "error_code") or status.value


def _failure_detail_from_audit(audit: dict[str, Any]) -> str:
    if _status_from_audit(audit) == LLMCallStatus.succeeded:
        return ""
    direct_error = _string_field(audit, "error")
    if direct_error:
        return direct_error
    attempts = audit.get("attempts", [])
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if isinstance(attempt, dict) and attempt.get("error_message"):
                return str(attempt["error_message"])
    return ""


def _retry_count(*, attempts: Any, attempted_models: Any) -> int:
    if isinstance(attempts, list) and attempts:
        return max(0, len(attempts) - 1)
    if isinstance(attempted_models, list) and attempted_models:
        return max(0, len(attempted_models) - 1)
    return 0


def _stage_from_task_kind(task_kind: str) -> str:
    normalized = task_kind.strip().lower()
    if normalized.startswith("stage"):
        return normalized.split("_", 1)[0]
    if normalized in {"discovery_normalization", "define_requirements"}:
        return "define" if normalized == "define_requirements" else "discover"
    return ""


def _metadata_string(audit: dict[str, Any], key: str) -> str:
    value = audit.get(key)
    if value not in (None, ""):
        return str(value).strip()
    metadata = audit.get("metadata", {})
    if isinstance(metadata, dict):
        return _string_field(metadata, key)
    return ""


def _string_field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _int_field(payload: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(payload.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _uuid_field(payload: dict[str, Any], key: str) -> UUID | None:
    raw_value = _metadata_string(payload, key) or _string_field(payload, key)
    if not raw_value:
        return None
    try:
        return UUID(raw_value)
    except ValueError:
        return None


def _datetime_field(payload: dict[str, Any], key: str) -> datetime | None:
    value = payload.get(key)
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _append_error(summary: RuntimeAuditBackfillSummary, message: str) -> None:
    if len(summary.errors) < 20:
        summary.errors.append(message)


def _resolve_first_candidate(primary: Path, fallback: Path) -> Path:
    if primary.exists() or not fallback.exists():
        return primary.resolve()
    return fallback.resolve()


def _default_audit_root() -> Path:
    from app.core.config import get_settings

    return get_settings().llm_config_path.parent.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill del ledger FinOps LLM desde auditorias JSONL de Codex local y Antigravity CLI.",
    )
    parser.add_argument("--audit-root", default=None, help="Raiz donde buscar codex-workspaces/ y agy-workspaces/.")
    parser.add_argument("--codex-audit-path", default=None, help="Ruta explicita a runtime-audit.jsonl.")
    parser.add_argument("--antigravity-audit-path", default=None, help="Ruta explicita a runtime-audit-agy.jsonl.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from app.db import create_db_and_tables, engine

    create_db_and_tables()
    with Session(engine) as session:
        summary = backfill_llm_finops_from_runtime_audits(
            session,
            audit_root=args.audit_root,
            codex_audit_path=args.codex_audit_path,
            antigravity_audit_path=args.antigravity_audit_path,
        )
    print(json.dumps({"ok": True, "summary": summary.to_dict()}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
