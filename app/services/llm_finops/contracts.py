from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FinOpsContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class LLMCallStatus(str, Enum):
    succeeded = "succeeded"
    failed = "failed"
    retry = "retry"
    cancelled = "cancelled"
    timeout = "timeout"
    provider_unavailable = "provider_unavailable"
    schema_invalid = "schema_invalid"


class LLMCallContext(FinOpsContractModel):
    workspace_id: UUID | None = None
    user_id: UUID | None = None
    session_id: UUID | None = None
    project_id: UUID | None = None
    initiative_id: UUID | None = None
    stage: str = ""
    substage: str = ""
    agent_key: str = ""
    capability_key: str = ""
    action_key: str = ""
    operation_id: UUID | None = None
    parent_run_id: str = ""
    execution_mode: str = "primary"
    correlation_id: str = ""
    source: str = "builder_runtime"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "stage",
        "substage",
        "agent_key",
        "capability_key",
        "action_key",
        "parent_run_id",
        "execution_mode",
        "correlation_id",
        "source",
        mode="before",
    )
    @classmethod
    def normalize_string_fields(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


class NormalizedLLMUsage(FinOpsContractModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0
    tool_call_count: int = 0
    provider_metrics: dict[str, Any] = Field(default_factory=dict)
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    usage_is_estimated: bool = False
    normalization_version: str = "llm-usage-normalization.v1"

    @field_validator(
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "accepted_prediction_tokens",
        "rejected_prediction_tokens",
        "tool_call_count",
        mode="before",
    )
    @classmethod
    def coerce_non_negative_int(cls, value: Any) -> int:
        try:
            parsed = int(value or 0)
        except (TypeError, ValueError):
            parsed = 0
        return max(0, parsed)

    @model_validator(mode="after")
    def fill_total_tokens(self) -> "NormalizedLLMUsage":
        if self.total_tokens <= 0:
            object.__setattr__(self, "total_tokens", self.input_tokens + self.output_tokens)
        return self

    def compatibility_token_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class LLMUsageCostBreakdown(FinOpsContractModel):
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_other: float = 0.0
    cost_total: float = 0.0
    currency: str = "USD"
    fx_rate: float = 1.0
    pricing_profile_key: str = ""
    pricing_snapshot: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: Any) -> str:
        return (str(value or "USD").strip() or "USD").upper()

    @model_validator(mode="after")
    def fill_cost_total(self) -> "LLMUsageCostBreakdown":
        if self.cost_total <= 0:
            object.__setattr__(
                self,
                "cost_total",
                round(self.cost_input + self.cost_output + self.cost_other, 8),
            )
        return self


class LLMUsageRecordInput(FinOpsContractModel):
    context: LLMCallContext = Field(default_factory=LLMCallContext)
    provider_key: str = ""
    model_name: str = ""
    requested_model: str = ""
    execution_backend: str = ""
    execution_mode: str = "primary"
    request_id: str = ""
    provider_request_id: str = ""
    attempt_number: int = 1
    retry_count: int = 0
    fallback_used: bool = False
    shadow_provider_key: str = ""
    status: LLMCallStatus = LLMCallStatus.succeeded
    failure_kind: str = ""
    failure_detail: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int = 0
    queue_wait_ms: int = 0
    usage: NormalizedLLMUsage = Field(default_factory=NormalizedLLMUsage)
    cost: LLMUsageCostBreakdown = Field(default_factory=LLMUsageCostBreakdown)
    prompt_hash: str = ""
    response_hash: str = ""
    schema_validation_status: str = ""
    finish_reason: str = ""
    value_signal: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("attempt_number", mode="before")
    @classmethod
    def coerce_attempt_number(cls, value: Any) -> int:
        try:
            parsed = int(value or 1)
        except (TypeError, ValueError):
            parsed = 1
        return max(1, parsed)

    @field_validator("retry_count", "duration_ms", "queue_wait_ms", mode="before")
    @classmethod
    def coerce_non_negative_metrics(cls, value: Any) -> int:
        try:
            parsed = int(value or 0)
        except (TypeError, ValueError):
            parsed = 0
        return max(0, parsed)


class LLMUsageRecordResult(FinOpsContractModel):
    usage_record_id: UUID | None = None
    created: bool = False
    duplicate: bool = False
    cost: LLMUsageCostBreakdown = Field(default_factory=LLMUsageCostBreakdown)
    warnings: list[str] = Field(default_factory=list)
