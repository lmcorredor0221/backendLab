from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

from app.models import ContractModel, PydanticField

if TYPE_CHECKING:
    from app.services.deliverable_catalog.project_generation_context import ProjectGenerationContext


class ACPPromptSectionSynthesisRequest(ContractModel):
    section_id: str
    path: str
    title: str
    deterministic_markdown: str
    context_version: str = ""
    context_fingerprint: str = ""
    source_refs: list[str] = PydanticField(default_factory=list)
    specificity_anchors: list[str] = PydanticField(default_factory=list)
    allowed_terms: list[str] = PydanticField(default_factory=list)
    forbidden_terms: list[str] = PydanticField(default_factory=list)
    first_actionable_question: str = ""


class PromptSectionSynthesis(ContractModel):
    section_id: str
    section_markdown: str
    rationale: str = ""
    warnings: list[str] = PydanticField(default_factory=list)
    cited_source_refs: list[str] = PydanticField(default_factory=list)
    introduced_terms: list[str] = PydanticField(default_factory=list)


class ACPPromptSectionSynthesizer(Protocol):
    def __call__(self, request: ACPPromptSectionSynthesisRequest) -> PromptSectionSynthesis | None:
        ...


class PromptSectionSynthesisRejected(ValueError):
    pass


_ALWAYS_FORBIDDEN_TERMS = (
    "http://",
    "https://",
    "bearer ",
    "api_key",
    "client_secret",
    "cookie",
    "cookies",
)
_CONDITIONAL_FORBIDDEN_TERMS = (
    "endpoint",
    "oauth app",
    "oauth_app",
    "vector store",
    "vector_store",
    "embedding",
    "embeddings",
    "sla",
)
_GENERIC_ALLOWED_TERMS = (
    "actor",
    "architecture",
    "arquitectura",
    "criterio",
    "decision",
    "evaluator",
    "evaluador",
    "fuente",
    "gap",
    "guardrail",
    "hitl",
    "memoria",
    "needs_review",
    "planner",
    "pregunta",
    "problema",
    "rag",
    "rol",
    "source_ref",
    "tool",
    "tools",
    "workflow",
)


def build_prompt_section_synthesis_request(
    *,
    section_id: str,
    path: str,
    title: str,
    deterministic_markdown: str,
    context: ProjectGenerationContext,
    first_actionable_question: str,
) -> ACPPromptSectionSynthesisRequest:
    allowed_terms = _allowed_terms_from_context(context)
    return ACPPromptSectionSynthesisRequest(
        section_id=section_id,
        path=path,
        title=title,
        deterministic_markdown=deterministic_markdown,
        context_version=context.context_version,
        context_fingerprint=context.input_fingerprint,
        source_refs=[item.ref for item in context.source_refs if item.ref],
        specificity_anchors=[item.value for item in context.specificity_anchors if item.value],
        allowed_terms=sorted({*allowed_terms, *_GENERIC_ALLOWED_TERMS}),
        forbidden_terms=[*_ALWAYS_FORBIDDEN_TERMS, *_CONDITIONAL_FORBIDDEN_TERMS],
        first_actionable_question=first_actionable_question,
    )


def validate_prompt_section_synthesis(
    synthesis: PromptSectionSynthesis,
    request: ACPPromptSectionSynthesisRequest,
) -> PromptSectionSynthesis:
    if synthesis.section_id != request.section_id:
        raise PromptSectionSynthesisRejected("section_id mismatch")

    markdown = synthesis.section_markdown.strip()
    if len(markdown) < 80:
        raise PromptSectionSynthesisRejected("section_markdown too short")

    allowed_refs = set(request.source_refs)
    if allowed_refs and not synthesis.cited_source_refs:
        raise PromptSectionSynthesisRejected("missing cited_source_refs")
    unknown_refs = [item for item in synthesis.cited_source_refs if item not in allowed_refs]
    if unknown_refs:
        raise PromptSectionSynthesisRejected("unknown cited_source_refs")

    markdown_lower = markdown.lower()
    allowed_blob = " ".join(
        [
            request.deterministic_markdown,
            *request.allowed_terms,
            *request.specificity_anchors,
            *request.source_refs,
        ]
    ).lower()
    for term in _ALWAYS_FORBIDDEN_TERMS:
        if term in markdown_lower:
            raise PromptSectionSynthesisRejected(f"forbidden term: {term}")
    for term in request.forbidden_terms:
        if term in _ALWAYS_FORBIDDEN_TERMS:
            continue
        if term in markdown_lower and term not in allowed_blob:
            raise PromptSectionSynthesisRejected(f"unapproved operational term: {term}")

    allowed_normalized = {_normalize_term(item) for item in request.allowed_terms}
    invalid_terms = [
        term
        for term in synthesis.introduced_terms
        if _requires_explicit_allowlist(term) and _normalize_term(term) not in allowed_normalized
    ]
    if invalid_terms:
        raise PromptSectionSynthesisRejected("introduced_terms not allowed")

    return synthesis.model_copy(update={"section_markdown": markdown})


def _allowed_terms_from_context(context: Any) -> set[str]:
    terms = {
        context.context_version,
        context.input_fingerprint,
        context.project_title or "",
        context.problem_statement or "",
        context.current_process or "",
        context.current_user or "",
        context.desired_outcome or "",
        context.architecture or "",
        context.reasoning_pattern or "",
        context.coordination_model or "",
        context.memory_strategy or "",
        "approved_retrieval_design" if context.rag_required else "pending_knowledge_foundation",
    }
    terms.update(context.objectives)
    terms.update(context.mvp_scope)
    terms.update(context.constraints)
    terms.update(context.nondelegable_decisions)
    terms.update(context.guardrails)
    terms.update(context.risks)
    terms.update(context.acceptance_criteria)
    terms.update(item.value for item in context.roles)
    terms.update(item.name for item in context.tools)
    terms.update(item.purpose for item in context.tools)
    terms.update(item.name for item in context.tool_contracts)
    terms.update(item.category for item in context.tool_contracts)
    terms.update(item.name for item in context.knowledge_sources)
    terms.update(item.source_type for item in context.knowledge_sources)
    terms.update(item.question for item in context.open_questions)
    terms.update(item.question_text for item in context.construction_questions)
    terms.update(item.value for item in context.specificity_anchors)
    terms.update(item.ref for item in context.source_refs)
    return {item.strip() for item in terms if item and item.strip()}


def _normalize_term(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def _requires_explicit_allowlist(value: str) -> bool:
    normalized = _normalize_term(value)
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            ".",
            "api",
            "binding",
            "cookie",
            "credential",
            "embedding",
            "endpoint",
            "oauth",
            "provider",
            "secret",
            "sla",
            "token",
            "tool",
            "vector",
        )
    )
