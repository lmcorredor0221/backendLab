from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from app.services.agentic_runtime.contracts import (
    BuilderInferenceResolution,
    BuilderInferenceTrace,
)
from app.services.lean_question_policy import classify_stage_question
from app.services.product_processing.contracts import ProductProcessingMode, ProductProcessingProfile
from app.services.product_processing.policy import (
    BLUEPRINT_TIER_POLICY,
    classify_inference_permission,
    get_product_processing_profile,
)


PENDING_SOURCE_FIELDS = (
    "open_questions",
    "guided_questions",
    "missing_information",
    "needs_information",
    "coverage_gaps",
)


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").strip().lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.split())


def _slug(value: str) -> str:
    normalized = _normalize_text(value)
    compact = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return compact or "item"


def _item_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json"))
    return {"question": str(value or "").strip()}


def _item_text(payload: dict[str, Any]) -> str:
    for key in ("question", "question_text", "title", "detail", "summary", "key", "gap_key"):
        text = str(payload.get(key) or "").strip()
        if text:
            return text
    return ""


def _iter_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def build_pending_item_key(stage: str, source_ref: str, index: int, item: Any) -> str:
    payload = _item_dump(item)
    explicit = str(payload.get("key") or payload.get("id") or payload.get("gap_key") or "").strip()
    if explicit:
        return explicit
    text = _item_text(payload)
    digest = hashlib.sha1(f"{stage}:{source_ref}:{text}".encode("utf-8")).hexdigest()[:10]
    return f"{stage}:{source_ref}:{index}:{_slug(text)[:48]}:{digest}"


def iter_pending_items(value: Any) -> list[tuple[str, int, Any]]:
    items: list[tuple[str, int, Any]] = []
    for source_ref in PENDING_SOURCE_FIELDS:
        for index, item in enumerate(_iter_values(_field(value, source_ref, [])), start=1):
            items.append((source_ref, index, item))
    for index, item in enumerate(_iter_values(_field(value, "dependency_gaps", [])), start=1):
        items.append(("dependency_gaps", index, item))
    validation = _field(value, "validation")
    if validation is not None:
        for index, item in enumerate(_iter_values(_field(validation, "blocking_open_questions", [])), start=1):
            items.append(("validation.blocking_open_questions", index, item))
        for index, item in enumerate(_iter_values(_field(validation, "blocking_issues", [])), start=1):
            items.append(("validation.blocking_issues", index, item))
    preflight = _field(value, "preflight")
    if preflight is not None:
        for index, item in enumerate(_iter_values(_field(preflight, "missing_information", [])), start=1):
            items.append(("preflight.missing_information", index, item))
    dry_compile_status = _field(value, "dry_compile_status")
    if dry_compile_status is not None:
        for index, item in enumerate(_iter_values(_field(dry_compile_status, "blocking_issues", [])), start=1):
            items.append(("dry_compile.blocking_issues", index, item))
    return items


def build_inference_resolution_map(trace: BuilderInferenceTrace | Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if trace is None:
        return {}
    payload = trace.model_dump(mode="json") if hasattr(trace, "model_dump") else dict(trace)
    resolutions = payload.get("resolutions")
    if not isinstance(resolutions, list):
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for item in resolutions:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("question_key") or "").strip()
        if key:
            mapping[key] = dict(item)
    return mapping


def _option_answer(payload: dict[str, Any]) -> str:
    options = payload.get("answer_options")
    if not isinstance(options, list):
        return ""
    recommended = next(
        (
            item
            for item in options
            if isinstance(item, Mapping) and bool(item.get("recommended")) and str(item.get("label") or "").strip()
        ),
        None,
    )
    if recommended is not None:
        return str(recommended.get("label") or "").strip()
    first = next(
        (
            item
            for item in options
            if isinstance(item, Mapping) and str(item.get("label") or "").strip()
        ),
        None,
    )
    if first is not None:
        return str(first.get("label") or "").strip()
    first_text = next((item for item in options if isinstance(item, str) and item.strip()), "")
    return str(first_text or "").strip()


def _infer_candidate(payload: dict[str, Any]) -> tuple[str, float, str]:
    assumed = str(payload.get("assumed_answer") or "").strip()
    suggested = str(payload.get("suggested_answer") or "").strip()
    option_answer = _option_answer(payload)
    raw_confidence = payload.get("confidence")
    try:
        base_confidence = min(max(float(raw_confidence or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        base_confidence = 0.0

    if assumed:
        confidence = max(base_confidence, 0.88)
        return assumed, confidence, "Supuesto ya presente en la etapa y respaldado por el contexto aprobado."
    if suggested:
        confidence = max(base_confidence, 0.78)
        return suggested, confidence, "Respuesta sugerida ya presente en la salida de la etapa."
    if option_answer:
        confidence = max(base_confidence, 0.74)
        return option_answer, confidence, "Respuesta inferida desde la opcion recomendada por la etapa."
    return "", base_confidence, ""


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.85:
        return "high_confidence"
    if confidence >= BLUEPRINT_TIER_POLICY.infer_confidence_threshold:
        return "tentative"
    return "low_confidence"


def _final_disposition(permission_status: str) -> str:
    if permission_status in {"apply_now", "record_as_hypothesis"}:
        return permission_status
    if permission_status in {"defer_to_next_stage", "defer_to_blueprint_pro", "defer_to_acp"}:
        return "defer"
    if permission_status == "requires_human":
        return "requires_human"
    if permission_status == "reject_as_noise":
        return "reject_as_noise"
    return "not_inferable"


class StageAnswerInferenceService:
    def run(
        self,
        *,
        stage: str,
        value: Any,
        effective_language: str = "es",
        product_mode: ProductProcessingProfile | ProductProcessingMode | str = ProductProcessingMode.basic_free,
        provider_key: str = "",
        model_name: str = "",
        context_refs: Iterable[str] | None = None,
    ) -> BuilderInferenceTrace:
        profile = (
            get_product_processing_profile(product_mode)
            if not isinstance(product_mode, ProductProcessingProfile)
            else product_mode
        )
        context_evidence = [str(item) for item in context_refs or [] if str(item).strip()]
        resolutions: list[BuilderInferenceResolution] = []

        for source_ref, index, item in iter_pending_items(value):
            payload = _item_dump(item)
            question_key = build_pending_item_key(stage, source_ref, index, item)
            question_text = _item_text(payload)
            inferred_answer, confidence, evidence_summary = _infer_candidate(payload)
            permission_status = classify_inference_permission(
                stage,
                payload,
                profile,
                inferred_answer=inferred_answer,
                confidence=confidence,
            )
            decision = classify_stage_question(stage, payload)
            source_refs = [str(item) for item in payload.get("source_refs", []) or [] if str(item).strip()]
            evidence_refs = list(dict.fromkeys([source_ref, *source_refs, *context_evidence[:4]]))
            applied_to_stage = permission_status in {"apply_now", "record_as_hypothesis"}
            resolutions.append(
                BuilderInferenceResolution(
                    question_key=question_key,
                    question_text=question_text,
                    source_stage=stage,
                    target_stage=decision.deferral_target_stage or "",
                    inferred_answer=inferred_answer,
                    confidence=confidence,
                    confidence_bucket=_confidence_bucket(confidence),
                    evidence_refs=evidence_refs,
                    evidence_summary=evidence_summary,
                    model_name=model_name,
                    provider_key=provider_key,
                    applied_to_stage=applied_to_stage,
                    permission_status=permission_status,
                    final_disposition=_final_disposition(permission_status),
                )
            )

        applied_count = sum(1 for item in resolutions if item.permission_status == "apply_now")
        hypothesis_count = sum(1 for item in resolutions if item.permission_status == "record_as_hypothesis")
        deferred_count = sum(
            1
            for item in resolutions
            if item.permission_status in {"defer_to_next_stage", "defer_to_blueprint_pro", "defer_to_acp"}
        )
        unresolved_count = sum(
            1
            for item in resolutions
            if item.permission_status in {"requires_human", "reject_as_noise", "not_inferable"}
        )
        return BuilderInferenceTrace(
            stage=stage,
            effective_language=effective_language,
            product_mode=profile.mode.value,
            resolution_count=len(resolutions),
            applied_count=applied_count,
            hypothesis_count=hypothesis_count,
            deferred_count=deferred_count,
            unresolved_count=unresolved_count,
            resolutions=resolutions,
        )
