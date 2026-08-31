from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from app.models import CommercialTier
from app.services.lean_question_policy import classify_stage_question
from app.services.product_processing.contracts import (
    BlueprintTierPolicy,
    BlueprintUncertainty,
    ProductProcessingMode,
    ProductProcessingProfile,
    QuestionPolicyMode,
    UncertaintyClassification,
    UncertaintyDisposition,
    UncertaintyKind,
    UncertaintyOption,
)

InferencePermissionStatus = Literal[
    "apply_now",
    "record_as_hypothesis",
    "defer_to_next_stage",
    "defer_to_blueprint_pro",
    "defer_to_acp",
    "requires_human",
    "reject_as_noise",
    "not_inferable",
]


BLUEPRINT_TIER_POLICY = BlueprintTierPolicy(
    profiles={
        ProductProcessingMode.basic_free: ProductProcessingProfile(
            mode=ProductProcessingMode.basic_free,
            commercial_tier=CommercialTier.blueprint,
            label="Blueprint Basico",
            question_policy=QuestionPolicyMode.infer_defer,
            default_disposition=UncertaintyDisposition.defer,
            max_questions_per_stage=2,
            max_llm_iterations_per_stage=3,
            max_llm_calls_per_stage=2,
            cost_budget_units_per_stage=1,
            deliverable_generation_budget="low",
            reprocess_strategy="none",
            allow_inferred_assumptions=True,
            allow_nonblocking_continuation=True,
            require_stage_readiness=False,
            surface_deferred_questions=False,
            surface_technical_questions=False,
            create_attention_for_nonblocking=False,
        ),
        ProductProcessingMode.premium_enrichment: ProductProcessingProfile(
            mode=ProductProcessingMode.premium_enrichment,
            commercial_tier=CommercialTier.blueprint_pro,
            label="Blueprint Premium",
            question_policy=QuestionPolicyMode.prioritized_enrichment,
            default_disposition=UncertaintyDisposition.resolve_now,
            max_questions_per_stage=6,
            max_llm_iterations_per_stage=5,
            max_llm_calls_per_stage=4,
            cost_budget_units_per_stage=3,
            deliverable_generation_budget="balanced",
            reprocess_strategy="selective",
            allow_inferred_assumptions=True,
            allow_nonblocking_continuation=True,
            require_stage_readiness=False,
            surface_deferred_questions=True,
            surface_technical_questions=False,
            create_attention_for_nonblocking=True,
        ),
        ProductProcessingMode.acp_implementation: ProductProcessingProfile(
            mode=ProductProcessingMode.acp_implementation,
            commercial_tier=CommercialTier.acp,
            label="Agent Construction Package",
            question_policy=QuestionPolicyMode.full_readiness,
            default_disposition=UncertaintyDisposition.resolve_now,
            max_questions_per_stage=12,
            max_llm_iterations_per_stage=8,
            max_llm_calls_per_stage=6,
            cost_budget_units_per_stage=6,
            deliverable_generation_budget="full",
            reprocess_strategy="selective",
            allow_inferred_assumptions=True,
            allow_nonblocking_continuation=False,
            require_stage_readiness=True,
            require_all_readiness_blockers=True,
            surface_deferred_questions=True,
            surface_technical_questions=True,
            create_attention_for_nonblocking=True,
        ),
    }
)


def get_product_processing_profile(
    mode: ProductProcessingMode | str,
    *,
    policy: BlueprintTierPolicy = BLUEPRINT_TIER_POLICY,
) -> ProductProcessingProfile:
    normalized = mode if isinstance(mode, ProductProcessingMode) else ProductProcessingMode(str(mode))
    return policy.profiles[normalized]


def resolve_product_processing_mode(
    tier: CommercialTier | str,
    *,
    premium_enrichment: bool = False,
    acp_direct: bool = False,
) -> ProductProcessingMode:
    if acp_direct:
        return ProductProcessingMode.acp_implementation
    commercial_tier = tier if isinstance(tier, CommercialTier) else CommercialTier(str(tier))
    if commercial_tier == CommercialTier.acp:
        return ProductProcessingMode.acp_implementation
    if premium_enrichment or commercial_tier == CommercialTier.blueprint_pro:
        return ProductProcessingMode.premium_enrichment
    return ProductProcessingMode.basic_free


def _field(question: Any, key: str, default: Any = "") -> Any:
    if isinstance(question, Mapping):
        return question.get(key, default)
    return getattr(question, key, default)


def _text(question: Any) -> str:
    return str(
        _field(question, "question", "")
        or _field(question, "question_text", "")
        or _field(question, "title", "")
        or _field(question, "key", "")
        or question
    ).strip()


def _confidence(question: Any) -> float:
    raw = _field(question, "confidence", 0.0)
    try:
        return min(max(float(raw or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def _is_blocking(question: Any) -> bool:
    raw = _field(question, "blocking", False)
    priority = str(_field(question, "priority", "") or "").strip().lower()
    return bool(raw) or priority in {"high", "critical", "blocking"}


def _options(question: Any) -> list[UncertaintyOption]:
    raw_options = _field(question, "answer_options", []) or _field(question, "options", []) or []
    options: list[UncertaintyOption] = []
    if not isinstance(raw_options, list):
        return options
    for index, raw in enumerate(raw_options, start=1):
        if isinstance(raw, str):
            options.append(UncertaintyOption(key=f"option_{index}", label=raw))
        elif isinstance(raw, Mapping):
            label = str(raw.get("label") or raw.get("title") or raw.get("value") or "").strip()
            if not label:
                continue
            options.append(
                UncertaintyOption(
                    key=str(raw.get("key") or f"option_{index}"),
                    label=label,
                    description=str(raw.get("description") or ""),
                    impact=str(raw.get("impact") or ""),
                    recommended=bool(raw.get("recommended", False)),
                    confidence=float(raw.get("confidence") or 0.0),
                )
            )
    return options


def _to_uncertainty(stage: str, question: Any, profile: ProductProcessingProfile) -> BlueprintUncertainty:
    title = _text(question) or "Incertidumbre por resolver"
    key = str(_field(question, "key", "") or _field(question, "id", "") or title[:80]).strip()
    suggested = str(_field(question, "suggested_answer", "") or "").strip()
    assumed = str(_field(question, "assumed_answer", "") or "").strip()
    return BlueprintUncertainty(
        key=key,
        kind=UncertaintyKind(str(_field(question, "kind", "question") or "question")),
        stage=stage,
        title=title,
        description=str(_field(question, "description", "") or _field(question, "rationale", "") or ""),
        reason=str(_field(question, "reason", "") or ""),
        impact=str(_field(question, "impact", "") or ""),
        confidence=_confidence(question),
        source_refs=list(_field(question, "source_refs", []) or []),
        affected_deliverable_keys=list(_field(question, "affected_deliverable_keys", []) or []),
        product_targets=[profile.mode],
        assumed_answer=assumed,
        suggested_answer=suggested,
        answer_options=_options(question),
        blocking=_is_blocking(question),
        required_for_implementation=bool(_field(question, "required_for_implementation", False)),
    )


def classify_uncertainty_for_profile(
    stage: str,
    question: Any,
    profile: ProductProcessingProfile | ProductProcessingMode | str,
    *,
    policy: BlueprintTierPolicy = BLUEPRINT_TIER_POLICY,
) -> UncertaintyClassification:
    resolved_profile = (
        get_product_processing_profile(profile, policy=policy)
        if not isinstance(profile, ProductProcessingProfile)
        else profile
    )
    uncertainty = _to_uncertainty(stage, question, resolved_profile)
    stage_decision = classify_stage_question(stage, question)
    target_stage = stage_decision.deferral_target_stage or ""

    if resolved_profile.mode == ProductProcessingMode.basic_free:
        if stage_decision.status == "reject_as_noise":
            disposition = UncertaintyDisposition.defer
            reason = stage_decision.reason or "Pregunta sin valor suficiente para bloquear Basico."
        elif uncertainty.blocking and uncertainty.confidence < policy.infer_confidence_threshold:
            disposition = UncertaintyDisposition.defer
            reason = "Basico registra la brecha como oportunidad de enriquecimiento sin detener el flujo."
        elif uncertainty.confidence >= policy.infer_confidence_threshold or uncertainty.assumed_answer:
            disposition = UncertaintyDisposition.infer
            reason = "Basico puede inferir y continuar registrando el supuesto."
        else:
            disposition = UncertaintyDisposition.defer
            reason = stage_decision.reason or "Basico difiere preguntas no indispensables."
        return UncertaintyClassification(
            uncertainty=uncertainty.model_copy(update={"disposition": disposition, "deferral_target_stage": target_stage}),
            profile_mode=resolved_profile.mode,
            disposition=disposition,
            reason=reason,
            target_stage=target_stage,
            should_surface_to_user=False,
            should_create_attention=False,
            should_continue_processing=True,
        )

    if resolved_profile.mode == ProductProcessingMode.premium_enrichment:
        if stage_decision.status == "defer_to_acp":
            disposition = UncertaintyDisposition.defer
            reason = stage_decision.reason or "Premium conserva la decision para ACP si es tecnica de implementacion."
            surface = resolved_profile.surface_deferred_questions
        elif uncertainty.confidence >= policy.premium_priority_threshold or uncertainty.blocking:
            disposition = UncertaintyDisposition.resolve_now
            reason = "Premium prioriza resolver la incertidumbre para enriquecer entregables afectados."
            surface = True
        else:
            disposition = UncertaintyDisposition.infer
            reason = "Premium conserva inferencia cuando el valor de preguntar es bajo."
            surface = False
        return UncertaintyClassification(
            uncertainty=uncertainty.model_copy(update={"disposition": disposition, "deferral_target_stage": target_stage}),
            profile_mode=resolved_profile.mode,
            disposition=disposition,
            reason=reason,
            target_stage=target_stage,
            should_surface_to_user=surface,
            should_create_attention=surface and resolved_profile.create_attention_for_nonblocking,
            should_continue_processing=True,
        )

    disposition = UncertaintyDisposition.resolve_now
    if stage_decision.status == "reject_as_noise":
        disposition = UncertaintyDisposition.defer
        reason = stage_decision.reason or "ACP registra ruido como no bloqueante."
    elif uncertainty.blocking or uncertainty.required_for_implementation:
        disposition = UncertaintyDisposition.block
        reason = "ACP requiere cerrar esta decision antes de construir el paquete."
    else:
        reason = stage_decision.reason or "ACP mantiene la pregunta visible para readiness de implementacion."

    should_continue = disposition != UncertaintyDisposition.block
    return UncertaintyClassification(
        uncertainty=uncertainty.model_copy(update={"disposition": disposition, "deferral_target_stage": target_stage}),
        profile_mode=resolved_profile.mode,
        disposition=disposition,
        reason=reason,
        target_stage=target_stage,
        should_surface_to_user=True,
        should_create_attention=True,
        should_continue_processing=should_continue,
    )


def classify_inference_permission(
    stage: str,
    question: Any,
    profile: ProductProcessingProfile | ProductProcessingMode | str,
    *,
    inferred_answer: str = "",
    confidence: float = 0.0,
    policy: BlueprintTierPolicy = BLUEPRINT_TIER_POLICY,
) -> InferencePermissionStatus:
    resolved_profile = (
        get_product_processing_profile(profile, policy=policy)
        if not isinstance(profile, ProductProcessingProfile)
        else profile
    )
    classification = classify_uncertainty_for_profile(stage, question, resolved_profile, policy=policy)
    stage_decision = classify_stage_question(stage, question)
    answer = str(
        inferred_answer
        or _field(question, "assumed_answer", "")
        or _field(question, "suggested_answer", "")
    ).strip()
    score = min(max(float(confidence or 0.0), 0.0), 1.0)
    tentative = score >= policy.infer_confidence_threshold
    high_confidence = score >= 0.85

    if stage_decision.status == "reject_as_noise":
        return "reject_as_noise"
    if classification.disposition == UncertaintyDisposition.block:
        return "requires_human"
    if not answer:
        if stage_decision.status == "defer_to_acp":
            return "defer_to_acp"
        if stage_decision.status == "defer_to_next_stage":
            return "defer_to_next_stage"
        return "not_inferable"

    if stage_decision.status == "defer_to_acp" or classification.target_stage == "acp":
        return "defer_to_acp"

    if resolved_profile.mode == ProductProcessingMode.basic_free:
        if stage_decision.status == "defer_to_next_stage":
            return "record_as_hypothesis" if tentative else "defer_to_next_stage"
        if high_confidence:
            return "apply_now"
        if tentative:
            return "record_as_hypothesis"
        return "not_inferable"

    if resolved_profile.mode == ProductProcessingMode.premium_enrichment:
        if stage_decision.status == "defer_to_next_stage" and classification.target_stage:
            return "defer_to_blueprint_pro" if classification.target_stage != "acp" else "defer_to_acp"
        if high_confidence:
            return "apply_now"
        if tentative:
            return "record_as_hypothesis"
        return "not_inferable"

    if high_confidence or tentative:
        return "apply_now"
    return "requires_human"
