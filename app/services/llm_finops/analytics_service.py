from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import LLMUsageLedgerRecord


@dataclass(frozen=True)
class LLMUsageAnalyticsFilters:
    workspace_id: UUID | None = None
    started_from: datetime | None = None
    started_to: datetime | None = None
    user_id: UUID | None = None
    session_id: UUID | None = None
    project_id: UUID | None = None
    initiative_id: UUID | None = None
    stage: str = ""
    agent_key: str = ""
    capability_key: str = ""
    provider_key: str = ""
    model_name: str = ""


class LLMUsageAnalyticsService:
    def list_usage(
        self,
        session: Session,
        filters: LLMUsageAnalyticsFilters | None = None,
        *,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[LLMUsageLedgerRecord]:
        statement = _apply_filters(select(LLMUsageLedgerRecord), filters).order_by(
            LLMUsageLedgerRecord.started_at.desc()
        )
        return list(session.exec(statement.offset(max(0, offset)).limit(max(1, limit))).all())

    def summarize(
        self,
        session: Session,
        filters: LLMUsageAnalyticsFilters | None = None,
    ) -> dict[str, Any]:
        records = _records(session, filters)
        call_count = len(records)
        cost_total = round(sum(item.cost_total for item in records), 8)
        total_tokens = sum(item.total_tokens for item in records)
        duration_values = [item.duration_ms for item in records if item.duration_ms > 0]
        error_count = sum(1 for item in records if item.status != "succeeded")
        retry_count = sum(max(0, item.retry_count) for item in records)
        fallback_count = sum(1 for item in records if item.fallback_used)
        currency_breakdown = _currency_breakdown(records)
        return {
            "call_count": call_count,
            "cost_total": cost_total,
            "currency": currency_breakdown[0]["currency"] if len(currency_breakdown) == 1 else "MIXED",
            "currency_breakdown": currency_breakdown,
            "total_tokens": total_tokens,
            "input_tokens": sum(item.input_tokens for item in records),
            "output_tokens": sum(item.output_tokens for item in records),
            "cost_per_call": round(cost_total / call_count, 8) if call_count else 0,
            "avg_latency_ms": round(sum(duration_values) / len(duration_values), 2) if duration_values else 0,
            "p95_latency_ms": _percentile(duration_values, 0.95),
            "error_count": error_count,
            "error_rate": round(error_count / call_count, 4) if call_count else 0,
            "retry_count": retry_count,
            "fallback_count": fallback_count,
            "estimated_count": sum(1 for item in records if _is_estimated_usage(item)),
            "estimated_availability": {
                "status": "available" if any(item.other_token_metrics for item in records) else "partial",
                "source": "llm_usage_ledger.other_token_metrics.usage_is_estimated",
                "reason": "El indicador se persiste en other_token_metrics para llamadas instrumentadas recientes.",
            },
        }

    def top_consumers(
        self,
        session: Session,
        filters: LLMUsageAnalyticsFilters | None = None,
        *,
        dimension: str = "user_id",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        records = _records(session, filters)
        allowed_dimensions = {
            "user_id",
            "project_id",
            "initiative_id",
            "stage",
            "agent_key",
            "capability_key",
            "provider_key",
            "model_name",
        }
        group_key = dimension if dimension in allowed_dimensions else "user_id"
        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            value = str(getattr(record, group_key) or "unassigned")
            bucket = grouped.setdefault(
                value,
                {
                    "dimension": group_key,
                    "key": value,
                    "call_count": 0,
                    "cost_total": 0.0,
                    "total_tokens": 0,
                    "error_count": 0,
                },
            )
            _accumulate_bucket(bucket, record)
        return _ranked_buckets(grouped, limit=limit)

    def provider_breakdown(
        self,
        session: Session,
        filters: LLMUsageAnalyticsFilters | None = None,
    ) -> list[dict[str, Any]]:
        records = _records(session, filters)
        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            key = f"{record.provider_key}:{record.model_name}"
            bucket = grouped.setdefault(
                key,
                {
                    "provider_key": record.provider_key,
                    "model_name": record.model_name,
                    "call_count": 0,
                    "cost_total": 0.0,
                    "total_tokens": 0,
                    "error_count": 0,
                },
            )
            _accumulate_bucket(bucket, record)
        return _ranked_buckets(grouped, limit=len(grouped) or 1)

    def timeseries(
        self,
        session: Session,
        filters: LLMUsageAnalyticsFilters | None = None,
        *,
        granularity: str = "day",
    ) -> list[dict[str, Any]]:
        records = _records(session, filters)
        safe_granularity = granularity if granularity in {"day", "week", "month"} else "day"
        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            bucket_key = _bucket_start(record.started_at, granularity=safe_granularity).isoformat()
            bucket = grouped.setdefault(
                bucket_key,
                {
                    "bucket": bucket_key,
                    "period_start": bucket_key,
                    "granularity": safe_granularity,
                    "call_count": 0,
                    "cost_total": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "error_count": 0,
                    "retry_count": 0,
                    "fallback_count": 0,
                    "estimated_count": 0,
                    "currency_breakdown": {},
                },
            )
            bucket["call_count"] += 1
            bucket["cost_total"] = round(bucket["cost_total"] + record.cost_total, 8)
            bucket["input_tokens"] += record.input_tokens
            bucket["output_tokens"] += record.output_tokens
            bucket["total_tokens"] += record.total_tokens
            bucket["retry_count"] += max(0, record.retry_count)
            if record.status != "succeeded":
                bucket["error_count"] += 1
            if record.fallback_used:
                bucket["fallback_count"] += 1
            if _is_estimated_usage(record):
                bucket["estimated_count"] += 1
            currency = (record.currency or "USD").upper()
            currency_bucket = bucket["currency_breakdown"].setdefault(
                currency,
                {"currency": currency, "cost_total": 0.0, "call_count": 0},
            )
            currency_bucket["cost_total"] = round(currency_bucket["cost_total"] + record.cost_total, 8)
            currency_bucket["call_count"] += 1

        rows = []
        for bucket in grouped.values():
            currency_items = sorted(
                bucket["currency_breakdown"].values(),
                key=lambda item: (-float(item["cost_total"]), str(item["currency"])),
            )
            bucket["currency_breakdown"] = currency_items
            bucket["currency"] = currency_items[0]["currency"] if len(currency_items) == 1 else "MIXED"
            bucket["error_rate"] = round(bucket["error_count"] / bucket["call_count"], 4) if bucket["call_count"] else 0
            rows.append(bucket)
        return sorted(rows, key=lambda item: item["bucket"])


def _records(session: Session, filters: LLMUsageAnalyticsFilters | None) -> list[LLMUsageLedgerRecord]:
    return list(session.exec(_apply_filters(select(LLMUsageLedgerRecord), filters)).all())


def _apply_filters(statement: Any, filters: LLMUsageAnalyticsFilters | None) -> Any:
    if filters is None:
        return statement
    if filters.started_from is not None:
        statement = statement.where(LLMUsageLedgerRecord.started_at >= filters.started_from)
    if filters.started_to is not None:
        statement = statement.where(LLMUsageLedgerRecord.started_at <= filters.started_to)
    for field_name in (
        "user_id",
        "workspace_id",
        "session_id",
        "project_id",
        "initiative_id",
        "stage",
        "agent_key",
        "capability_key",
        "provider_key",
        "model_name",
    ):
        value = getattr(filters, field_name)
        if value in (None, ""):
            continue
        statement = statement.where(getattr(LLMUsageLedgerRecord, field_name) == value)
    return statement


def _accumulate_bucket(bucket: dict[str, Any], record: LLMUsageLedgerRecord) -> None:
    bucket["call_count"] += 1
    bucket["cost_total"] = round(bucket["cost_total"] + record.cost_total, 8)
    bucket["total_tokens"] += record.total_tokens
    if record.status != "succeeded":
        bucket["error_count"] += 1


def _ranked_buckets(grouped: dict[str, dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ranked = sorted(
        grouped.values(),
        key=lambda item: (item["cost_total"], item["total_tokens"], item["call_count"]),
        reverse=True,
    )
    return ranked[: max(1, limit)]


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def _currency_breakdown(records: list[LLMUsageLedgerRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        currency = (record.currency or "USD").upper()
        bucket = grouped.setdefault(currency, {"currency": currency, "cost_total": 0.0, "call_count": 0})
        bucket["cost_total"] = round(bucket["cost_total"] + record.cost_total, 8)
        bucket["call_count"] += 1
    return sorted(grouped.values(), key=lambda item: (-float(item["cost_total"]), str(item["currency"])))


def _is_estimated_usage(record: LLMUsageLedgerRecord) -> bool:
    metrics = record.other_token_metrics if isinstance(record.other_token_metrics, dict) else {}
    pricing_snapshot = record.pricing_snapshot if isinstance(record.pricing_snapshot, dict) else {}
    raw_usage = record.usage_raw_redacted if isinstance(record.usage_raw_redacted, dict) else {}
    metadata = record.metadata_payload if isinstance(record.metadata_payload, dict) else {}
    return bool(
        metrics.get("usage_is_estimated")
        or pricing_snapshot.get("usage_is_estimated")
        or raw_usage.get("usage_is_estimated")
        or metadata.get("usage_is_estimated")
    )


def _bucket_start(value: datetime, *, granularity: str) -> datetime:
    normalized = _to_naive_utc(value)
    if granularity == "month":
        return normalized.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if granularity == "week":
        start = normalized - timedelta(days=normalized.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0)
    return normalized.replace(hour=0, minute=0, second=0, microsecond=0)


def _to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
