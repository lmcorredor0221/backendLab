from __future__ import annotations

from uuid import UUID

from app.models import SessionSnapshot
from app.services.acp_prompt_synthesis import (
    ACPPromptSectionSynthesisRequest,
    ACPPromptSectionSynthesizer,
    PromptSectionSynthesis,
)
from app.services.llm_runtime.provider_router import BuilderProviderFacade
from app.services.llm_runtime.stage_context_types import StageContextBundle


class LLMBackedACPPromptSectionSynthesizer:
    def __init__(
        self,
        builder_service: BuilderProviderFacade,
        *,
        workspace_id: UUID | None,
        session_id: UUID | None,
        snapshot: SessionSnapshot | None,
        effective_language: str = "",
    ) -> None:
        self._builder_service = builder_service
        self._workspace_id = workspace_id
        self._session_id = session_id
        self._snapshot = snapshot
        self._effective_language = effective_language
        self._events: list[dict[str, object]] = []

    def __call__(self, request: ACPPromptSectionSynthesisRequest) -> PromptSectionSynthesis | None:
        context_bundle = StageContextBundle(
            capability="synthesize_acp_prompt_section",
            role="builder",
            stage="package",
            workspace_id=self._workspace_id,
            session_id=self._session_id,
            session_snapshot=self._snapshot,
            effective_language=self._effective_language,
            knowledge_manifest=None,
            memory_policy=None,
            short_term_memory=None,
            approved_refs=[],
            retrieved_hits=[],
            context_fingerprint=request.context_fingerprint,
            absence_reason="ACP prompt synthesis uses approved ProjectGenerationContext only.",
            finops_metadata={
                "acp_prompt_path": request.path,
                "acp_prompt_section_id": request.section_id,
                "context_version": request.context_version,
            },
        )
        result = self._builder_service.synthesize_acp_prompt_section(request, context_bundle=context_bundle)
        self._events.append(
            {
                "section_id": request.section_id,
                "path": request.path,
                "provider_key": result.provider_key or "",
                "execution_backend": result.execution_backend or "",
                "execution_mode": result.execution_mode or "",
                "model_name": result.model_name or "",
                "prompt_version": result.prompt_version or "",
                "request_id": result.request_id or "",
                "schema_validation_status": result.schema_validation_status or "",
                "finish_reason": result.finish_reason or "",
                "failure_kind": result.failure_kind or "",
                "warning": result.warning or "",
                "duration_ms": result.duration_ms,
                "queue_wait_ms": result.queue_wait_ms,
                "retry_count": result.retry_count,
                "cost_total": result.cost_total,
                "currency": result.currency,
                "token_usage": dict(result.token_usage),
                "provider_artifact_returned": result.artifact is not None,
                "validation_status": "pending",
                "validation_reason": "",
            }
        )
        if not isinstance(result.artifact, PromptSectionSynthesis):
            return None
        return result.artifact

    def record_validation_outcome(self, *, section_id: str, path: str, status: str, reason: str = "") -> None:
        for event in reversed(self._events):
            if event.get("section_id") == section_id and event.get("path") == path:
                event["validation_status"] = status
                event["validation_reason"] = reason
                return
        self._events.append(
            {
                "section_id": section_id,
                "path": path,
                "provider_key": "",
                "execution_backend": "",
                "execution_mode": "",
                "model_name": "",
                "prompt_version": "",
                "request_id": "",
                "schema_validation_status": "",
                "finish_reason": "",
                "failure_kind": "",
                "warning": "",
                "duration_ms": 0,
                "queue_wait_ms": 0,
                "retry_count": 0,
                "cost_total": 0.0,
                "currency": "USD",
                "token_usage": {},
                "provider_artifact_returned": False,
                "validation_status": status,
                "validation_reason": reason,
            }
        )

    def build_report_payload(self) -> dict[str, object]:
        attempted = len(self._events)
        applied = sum(1 for event in self._events if event.get("validation_status") == "applied")
        rejected = sum(1 for event in self._events if event.get("validation_status") == "rejected")
        fallback = sum(1 for event in self._events if event.get("validation_status") == "fallback")
        total_duration_ms = sum(int(event.get("duration_ms") or 0) for event in self._events)
        total_cost = sum(float(event.get("cost_total") or 0.0) for event in self._events)
        return {
            "schema_version": "acp-prompt-synthesis-report.v1",
            "enabled": True,
            "attempted_sections": attempted,
            "applied_sections": applied,
            "rejected_sections": rejected,
            "fallback_sections": fallback,
            "rejection_rate": rejected / attempted if attempted else 0.0,
            "total_duration_ms": total_duration_ms,
            "total_cost": total_cost,
            "currency": "USD",
            "events": list(self._events),
        }


def build_llm_acp_prompt_synthesizer(
    builder_service: BuilderProviderFacade,
    *,
    workspace_id: UUID | None,
    session_id: UUID | None,
    snapshot: SessionSnapshot | None,
    effective_language: str = "",
) -> ACPPromptSectionSynthesizer:
    return LLMBackedACPPromptSectionSynthesizer(
        builder_service,
        workspace_id=workspace_id,
        session_id=session_id,
        snapshot=snapshot,
        effective_language=effective_language,
    )
