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
        if not isinstance(result.artifact, PromptSectionSynthesis):
            return None
        return result.artifact


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
