from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from time import perf_counter
from typing import Any, Callable, ContextManager

from sqlmodel import Session

from app.models import LLMProviderKey, utc_now
from app.services.llm_finops.contracts import (
    LLMCallContext,
    LLMCallStatus,
    LLMUsageRecordInput,
    NormalizedLLMUsage,
)
from app.services.llm_finops.ledger_service import LLMUsageLedgerService


FinOpsSessionFactory = Callable[[], ContextManager[Session]]


def default_finops_session_factory() -> ContextManager[Session]:
    from app.db import engine

    @contextmanager
    def _session_scope() -> Any:
        with Session(engine) as session:
            yield session

    return _session_scope()


def record_provider_call(
    *,
    call: Callable[[], Any],
    call_context: LLMCallContext,
    provider_key: LLMProviderKey | str,
    model_name: str,
    requested_model: str = "",
    execution_backend: str = "",
    execution_mode: str = "",
    ledger_service: LLMUsageLedgerService | None = None,
    session_factory: FinOpsSessionFactory | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    started_at = utc_now()
    perf_started = perf_counter()
    try:
        result = call()
    except Exception as exc:
        finished_at = utc_now()
        duration_ms = _elapsed_ms(perf_started)
        _record_usage_safely(
            ledger_service=ledger_service,
            session_factory=session_factory,
            record_input=LLMUsageRecordInput(
                context=call_context,
                provider_key=_provider_value(provider_key),
                model_name=model_name,
                requested_model=requested_model or model_name,
                execution_backend=execution_backend,
                execution_mode=execution_mode or call_context.execution_mode,
                status=LLMCallStatus.failed,
                failure_kind="provider_error",
                failure_detail=str(exc),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                usage=NormalizedLLMUsage(),
                finish_reason="exception",
                metadata=metadata or {},
            ),
        )
        raise

    finished_at = utc_now()
    duration_ms = _elapsed_ms(perf_started)
    return record_provider_result(
        result,
        call_context=call_context,
        provider_key=provider_key,
        model_name=model_name,
        requested_model=requested_model,
        execution_backend=execution_backend,
        execution_mode=execution_mode,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        ledger_service=ledger_service,
        session_factory=session_factory,
        metadata=metadata,
    )


def record_provider_result(
    result: Any,
    *,
    call_context: LLMCallContext,
    provider_key: LLMProviderKey | str,
    model_name: str,
    requested_model: str = "",
    execution_backend: str = "",
    execution_mode: str = "",
    started_at: Any,
    finished_at: Any,
    duration_ms: int,
    ledger_service: LLMUsageLedgerService | None = None,
    session_factory: FinOpsSessionFactory | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    normalized_usage = _normalized_usage_from_result(result)
    result_context = getattr(result, "finops_context", None) or call_context
    context_used_sources = _safe_context_sources(getattr(result, "context_used_sources", []) or [])
    context_stats = dict(getattr(result, "context_stats", {}) or {})
    context_prompt_truncated_keys = [
        str(item.get("key", ""))
        for item in context_used_sources
        if isinstance(item, dict) and item.get("prompt_truncated")
    ]
    context_required_truncated_keys = [
        str(item.get("key", ""))
        for item in context_used_sources
        if isinstance(item, dict) and item.get("required") and (item.get("truncated") or item.get("prompt_truncated"))
    ]
    record_input = LLMUsageRecordInput(
        context=result_context,
        provider_key=str(getattr(result, "provider_key", "") or _provider_value(provider_key)),
        model_name=str(getattr(result, "model_name", "") or model_name),
        requested_model=requested_model or model_name,
        execution_backend=str(getattr(result, "execution_backend", "") or execution_backend),
        execution_mode=str(getattr(result, "execution_mode", "") or execution_mode or result_context.execution_mode),
        request_id=str(getattr(result, "request_id", "") or ""),
        retry_count=int(getattr(result, "retry_count", 0) or 0),
        fallback_used=bool(getattr(result, "fallback_used", False)),
        shadow_provider_key=str(getattr(result, "shadow_provider_key", "") or ""),
        status=_status_from_result(result),
        failure_kind=str(getattr(result, "failure_kind", "") or ""),
        failure_detail=str(getattr(result, "failure_detail", "") or ""),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        queue_wait_ms=int(getattr(result, "queue_wait_ms", 0) or 0),
        usage=normalized_usage,
        prompt_hash=_prompt_hash_from_context(context_stats, context_used_sources),
        response_hash=_response_hash_from_result(result),
        schema_validation_status=str(getattr(result, "schema_validation_status", "") or ""),
        finish_reason=str(getattr(result, "finish_reason", "") or ""),
        metadata={
            **(metadata or {}),
            "route_reason": str(getattr(result, "route_reason", "") or ""),
            "capability_policy": dict(getattr(result, "capability_policy", {}) or {}),
            "knowledge_access_backend": str(getattr(result, "knowledge_access_backend", "") or ""),
            "effective_context_backend": str(getattr(result, "effective_context_backend", "") or ""),
            "context_stats": context_stats,
            "context_used_sources": context_used_sources,
            "context_prompt_truncated_keys": [item for item in context_prompt_truncated_keys if item],
            "context_required_truncated_keys": [item for item in context_required_truncated_keys if item],
            "context_required_truncated_count": len([item for item in context_required_truncated_keys if item]),
        },
    )
    record_result = _record_usage_safely(
        ledger_service=ledger_service,
        session_factory=session_factory,
        record_input=record_input,
    )
    update_payload: dict[str, Any] = {
        "duration_ms": duration_ms,
        "finops_context": result_context,
        "normalized_usage": normalized_usage,
    }
    if record_result is not None:
        update_payload.update(
            {
                "usage_record_id": record_result.usage_record_id,
                "cost_total": record_result.cost.cost_total,
                "currency": record_result.cost.currency,
            }
        )
    return replace(result, **update_payload)


def _safe_context_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    safe_sources: list[dict[str, Any]] = []
    allowed_fields = {
        "key",
        "title",
        "source_type",
        "uri",
        "authority_level",
        "required",
        "summary",
        "relative_path",
        "baseline_chars",
        "assembled_chars",
        "prompt_chars",
        "staged_file_chars",
        "token_estimate",
        "truncated",
        "prompt_truncated",
        "staged_file_truncated",
        "delivery_mode",
        "source_refs",
        "source_lineage",
        "source_version",
        "stage_affinity",
        "agent_affinity",
        "metadata",
    }
    for item in value:
        if not isinstance(item, dict):
            continue
        safe_item = {key: item.get(key) for key in allowed_fields if key in item}
        metadata = safe_item.get("metadata")
        if isinstance(metadata, dict):
            safe_item["metadata"] = {
                key: metadata.get(key)
                for key in (
                    "context_quality_version",
                    "input_payload_chars",
                    "compact_payload_chars",
                    "compact_payload_tokens_est",
                    "compact_retention_pct",
                    "payload_model",
                    "source_key",
                    "api_compaction_applied",
                )
                if key in metadata
            }
        safe_sources.append(safe_item)
    return safe_sources


def _prompt_hash_from_context(context_stats: dict[str, Any], context_used_sources: list[dict[str, Any]]) -> str:
    explicit_hash = str(context_stats.get("context_user_payload_sha256") or "").strip()
    if explicit_hash:
        return explicit_hash
    return _stable_payload_hash(
        {
            "context_fingerprint": context_stats.get("context_fingerprint", ""),
            "corpus_hash": context_stats.get("corpus_hash", ""),
            "used_sources": context_used_sources,
        }
    )


def _response_hash_from_result(result: Any) -> str:
    artifact = getattr(result, "artifact", None)
    if artifact is not None and hasattr(artifact, "model_dump"):
        return _stable_payload_hash(artifact.model_dump(mode="json"))
    if artifact is not None:
        return _stable_payload_hash(artifact)
    return _stable_payload_hash(
        {
            "warning": str(getattr(result, "warning", "") or ""),
            "failure_kind": str(getattr(result, "failure_kind", "") or ""),
            "failure_detail": str(getattr(result, "failure_detail", "") or ""),
            "finish_reason": str(getattr(result, "finish_reason", "") or ""),
            "schema_validation_status": str(getattr(result, "schema_validation_status", "") or ""),
        }
    )


def _stable_payload_hash(value: Any) -> str:
    try:
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        payload = str(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_usage_safely(
    *,
    ledger_service: LLMUsageLedgerService | None,
    session_factory: FinOpsSessionFactory | None,
    record_input: LLMUsageRecordInput,
) -> Any | None:
    if ledger_service is None or session_factory is None:
        return None
    try:
        with session_factory() as session:
            return ledger_service.record_call(session, record_input)
    except Exception:
        return None


def _normalized_usage_from_result(result: Any) -> NormalizedLLMUsage:
    normalized_usage = getattr(result, "normalized_usage", None)
    if isinstance(normalized_usage, NormalizedLLMUsage):
        return normalized_usage
    token_usage = getattr(result, "token_usage", {}) or {}
    return NormalizedLLMUsage(
        input_tokens=token_usage.get("input_tokens") or token_usage.get("prompt_tokens") or 0,
        output_tokens=token_usage.get("output_tokens") or token_usage.get("completion_tokens") or 0,
        total_tokens=token_usage.get("total_tokens") or 0,
    )


def _status_from_result(result: Any) -> LLMCallStatus:
    failure_kind = str(getattr(result, "failure_kind", "") or "")
    finish_reason = str(getattr(result, "finish_reason", "") or "")
    schema_status = str(getattr(result, "schema_validation_status", "") or "")
    artifact = getattr(result, "artifact", None)
    if failure_kind == "provider_unavailable" or finish_reason == "provider_unavailable":
        return LLMCallStatus.provider_unavailable
    if failure_kind == "timeout" or finish_reason == "timeout":
        return LLMCallStatus.timeout
    if failure_kind == "provider_error":
        return LLMCallStatus.failed
    if schema_status in {"invalid", "missing_output"} and artifact is None:
        return LLMCallStatus.schema_invalid
    if failure_kind or artifact is None:
        return LLMCallStatus.failed
    return LLMCallStatus.succeeded


def _provider_value(provider_key: LLMProviderKey | str) -> str:
    return provider_key.value if isinstance(provider_key, LLMProviderKey) else str(provider_key)


def _elapsed_ms(perf_started: float) -> int:
    return max(0, round((perf_counter() - perf_started) * 1000))
