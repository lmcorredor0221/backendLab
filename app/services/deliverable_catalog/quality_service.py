from __future__ import annotations

import json
from hashlib import sha256
from uuid import UUID

from sqlmodel import Session

from app.services.deliverable_catalog.contracts import (
    DeliverableQualityEvaluation,
    DeliverableRegistryEntry,
)
from app.services.deliverable_catalog.persistence import DeliverableQualitySnapshotRecord


def fingerprint_payload(payload: object) -> str:
    return sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _is_non_empty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def evaluate_deliverable_quality(
    entry: DeliverableRegistryEntry,
    payload: object,
) -> DeliverableQualityEvaluation:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {}
    schema_contract = entry.quality_policy.schema_contract
    validator_key = entry.quality_policy.validator_key

    if payload is None:
        errors.append("output_missing")
    elif schema_contract == "diagram-model.v1":
        if not isinstance(payload, dict):
            errors.append("diagram_payload_must_be_object")
        else:
            nodes = payload.get("nodes")
            edges = payload.get("edges")
            checks["has_nodes"] = isinstance(nodes, list) and bool(nodes)
            checks["has_edges"] = isinstance(edges, list)
            if not checks["has_nodes"]:
                errors.append("diagram_nodes_missing")
            if not isinstance(edges, list):
                warnings.append("diagram_edges_missing")
            if isinstance(nodes, list):
                node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
                if any(not _is_non_empty_text(node_id) for node_id in node_ids):
                    errors.append("diagram_node_id_missing")
    elif schema_contract == "deliverable-artifact.v1":
        if not isinstance(payload, dict):
            errors.append("artifact_payload_must_be_object")
        else:
            title = payload.get("title")
            content = payload.get("content") or payload.get("markdown") or payload.get("summary")
            checks["has_title"] = _is_non_empty_text(title)
            checks["has_content"] = _is_non_empty_text(content)
            if not checks["has_title"]:
                errors.append("artifact_title_missing")
            if not checks["has_content"]:
                errors.append("artifact_content_missing")
            if validator_key == "artifact.commercial_consistency.v1":
                text = json.dumps(payload, ensure_ascii=False).lower()
                checks["declares_estimate_source"] = "estimate" in text or "estimar" in text or "traceability" in text
                if not checks["declares_estimate_source"]:
                    warnings.append("commercial_artifact_should_reference_estimate_sources")
    elif schema_contract == "professional-document.v1":
        if not isinstance(payload, dict):
            errors.append("professional_document_payload_must_be_object")
        else:
            checks["has_title"] = _is_non_empty_text(payload.get("title"))
            checks["has_summary"] = isinstance(payload.get("summary"), list) and bool(payload.get("summary"))
            checks["has_sections"] = isinstance(payload.get("sections"), list) and bool(payload.get("sections"))
            checks["has_traceability"] = isinstance(payload.get("traceability_refs"), list) and bool(payload.get("traceability_refs"))
            if not checks["has_title"]:
                errors.append("professional_document_title_missing")
            if not checks["has_summary"]:
                errors.append("professional_document_summary_missing")
            if not checks["has_sections"]:
                errors.append("professional_document_sections_missing")
            if not checks["has_traceability"]:
                warnings.append("professional_document_traceability_missing")
    elif schema_contract in {"plantuml-source.v1", "mermaid-source.v1", "bpmn-source.v1", "c4-source.v1", "diagram-presentation.v1"}:
        if not isinstance(payload, dict):
            errors.append("diagram_source_payload_must_be_object")
        else:
            checks["has_diagram_key"] = _is_non_empty_text(payload.get("diagram_key"))
            source = payload.get("source") or payload.get("xml") or payload.get("protected_view")
            checks["has_source_or_presentation"] = isinstance(source, bool) or _is_non_empty_text(source)
            checks["has_traceability"] = isinstance(payload.get("traceability_refs"), list)
            if not checks["has_diagram_key"]:
                errors.append("diagram_source_key_missing")
            if not checks["has_source_or_presentation"]:
                errors.append("diagram_source_content_missing")
    else:
        if not isinstance(payload, dict):
            errors.append("payload_must_be_object")
        elif not payload:
            warnings.append("payload_empty")

    base_score = 100
    score = max(0, base_score - (35 * len(errors)) - (10 * len(warnings)))
    state = "failed" if errors or score < entry.quality_policy.minimum_score else "warning" if warnings else "passed"
    return DeliverableQualityEvaluation(
        deliverable_key=entry.deliverable_key,
        schema_contract=schema_contract,
        validator_key=validator_key,
        state=state,
        score=score,
        errors=errors,
        warnings=warnings,
        checks=checks,
    )


def record_deliverable_quality_snapshot(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID,
    entry: DeliverableRegistryEntry,
    version_ref: str,
    payload: object,
    source_fingerprint: str = "",
) -> DeliverableQualitySnapshotRecord:
    evaluation = evaluate_deliverable_quality(entry, payload)
    record = DeliverableQualitySnapshotRecord(
        workspace_id=workspace_id,
        session_id=session_id,
        deliverable_key=entry.deliverable_key,
        version_ref=version_ref,
        state=evaluation.state,
        score=evaluation.score,
        errors=evaluation.errors,
        warnings=evaluation.warnings,
        checks=evaluation.checks,
        source_fingerprint=source_fingerprint or fingerprint_payload(payload),
    )
    db.add(record)
    db.flush()
    return record
