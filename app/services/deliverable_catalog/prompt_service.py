from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from app.models import utc_now
from app.services.deliverable_catalog.contracts import (
    DeliverableGenerationMode,
    DeliverablePromptResponse,
    DeliverablePromptUpdate,
    DeliverablePromptValidationRequest,
    DeliverablePromptValidationResponse,
    DeliverablePromptVersionEntry,
    DeliverableRegistryEntry,
    DeliverableType,
)
from app.services.deliverable_catalog.persistence import (
    DeliverableGovernanceRecord,
    DeliverablePromptAuditRecord,
    DeliverablePromptVersionRecord,
)
from app.services.deliverable_catalog.policy_service import scope_key_for_workspace


def _default_prompt_body(entry: DeliverableRegistryEntry) -> str:
    source_contract = entry.prompt_policy.schema_contract or entry.quality_policy.schema_contract
    validator_key = entry.prompt_policy.validator_key or entry.quality_policy.validator_key
    if entry.deliverable_type == DeliverableType.diagram:
        return (
            f"Eres un modelador y arquitecto de sistemas experto. Genera el diagrama formal '{entry.title}' respetando su estándar semántico y de visualización.\n"
            f"Contrato fuente obligatorio: {source_contract}.\n"
            f"Validador objetivo: {validator_key}.\n"
            f"Fuentes obligatorias: {', '.join(entry.context_policy.short_term_refs or entry.dependency_policy.depends_on) or 'artefactos aprobados disponibles'}.\n"
            "Usa solo información aprobada, conserva source_refs y registra supuestos si la información no es suficiente.\n"
            "No mezcles notaciones, no expongas datos sensibles y no inventes tecnología, costos, endpoints ni owners."
        )
    if entry.deliverable_type in {DeliverableType.artifact, DeliverableType.document}:
        return (
            f"Crea el entregable profesional de nivel directivo y técnico '{entry.title}' como documento editorial completamente trazable.\n"
            f"Contrato fuente: {source_contract}. Contrato de presentación esperado: professional-document.v1.\n"
            f"Debes heredar y referenciar: {', '.join(entry.context_policy.short_term_refs or entry.dependency_policy.depends_on) or 'snapshots aprobados'}.\n"
            "Debes estructurar el documento con: Resumen ejecutivo, Diagnóstico/Alcance, Arquitectura de razonamiento, Estrategia de memoria/conocimiento, Catálogo de herramientas y Métricas de estimación/ROI cuando aplique.\n"
            "Está estrictamente prohibido inventar costos, horas, porcentajes, proveedores, tecnologías o decisiones no aprobadas.\n"
            "Cada número, decisión técnica o afirmación crítica debe conservar traceability_refs hacia la etapa o artefacto origen."
        )
    return (
        f"Genera el entregable técnico '{entry.title}' usando únicamente contexto aprobado y evidencia trazable.\n"
        f"Responde estrictamente con el contrato {source_contract}.\n"
        "Incluye supuestos, fuentes, modelo de inferencia, herramientas gobernadas y advertencias cuando la confianza no sea suficiente."
    )


def _scope_keys(workspace_id: UUID | None) -> list[str]:
    keys = ["platform"]
    if workspace_id is not None:
        keys.append(scope_key_for_workspace(workspace_id))
    return keys


def _latest_prompt_version(
    db: Session,
    entry: DeliverableRegistryEntry,
    *,
    workspace_id: UUID | None,
) -> DeliverablePromptVersionRecord | None:
    rows = db.exec(
        select(DeliverablePromptVersionRecord)
        .where(
            DeliverablePromptVersionRecord.deliverable_key == entry.deliverable_key,
            DeliverablePromptVersionRecord.scope_key.in_(_scope_keys(workspace_id)),
        )
        .order_by(DeliverablePromptVersionRecord.created_at.desc())
    ).all()
    workspace_scope = scope_key_for_workspace(workspace_id)
    return next((row for row in rows if row.scope_key == workspace_scope), None) or next(
        (row for row in rows if row.scope_key == "platform"),
        None,
    )


def _version_entry(row: DeliverablePromptVersionRecord) -> DeliverablePromptVersionEntry:
    return DeliverablePromptVersionEntry(
        id=row.id,
        version=row.version,
        status=row.status,
        prompt_template_key=row.prompt_template_key,
        schema_contract=row.schema_contract,
        validator_key=row.validator_key,
        fallback_policy=row.fallback_policy,
        created_by_user_id=row.created_by_user_id,
        created_at=row.created_at,
    )


def _list_prompt_versions(
    db: Session,
    entry: DeliverableRegistryEntry,
    *,
    workspace_id: UUID | None,
) -> list[DeliverablePromptVersionEntry]:
    rows = db.exec(
        select(DeliverablePromptVersionRecord)
        .where(
            DeliverablePromptVersionRecord.deliverable_key == entry.deliverable_key,
            DeliverablePromptVersionRecord.scope_key.in_(_scope_keys(workspace_id)),
        )
        .order_by(DeliverablePromptVersionRecord.created_at.desc())
        .limit(20)
    ).all()
    return [_version_entry(row) for row in rows]


def get_deliverable_prompt(
    db: Session,
    entry: DeliverableRegistryEntry,
    *,
    workspace_id: UUID | None,
) -> DeliverablePromptResponse:
    latest = _latest_prompt_version(db, entry, workspace_id=workspace_id)
    governance = db.exec(
        select(DeliverableGovernanceRecord).where(
            DeliverableGovernanceRecord.scope_key == scope_key_for_workspace(workspace_id),
            DeliverableGovernanceRecord.deliverable_key == entry.deliverable_key,
        )
    ).first()
    prompt_override = governance.prompt_override if governance is not None else {}
    prompt_body = str(prompt_override.get("prompt_body") or (latest.prompt_body if latest else "") or _default_prompt_body(entry))
    schema_contract = str(
        prompt_override.get("schema_contract")
        or (latest.schema_contract if latest else "")
        or entry.prompt_policy.schema_contract
        or entry.quality_policy.schema_contract
    )
    validator_key = str(prompt_override.get("validator_key") or (latest.validator_key if latest else "") or entry.prompt_policy.validator_key)
    fallback_policy = str(
        prompt_override.get("fallback_policy") or (latest.fallback_policy if latest else "") or entry.prompt_policy.fallback_policy
    )
    return DeliverablePromptResponse(
        deliverable_key=entry.deliverable_key,
        scope_key=scope_key_for_workspace(workspace_id),
        workspace_id=workspace_id,
        prompt_template_key=entry.prompt_policy.prompt_template_key,
        prompt_status=governance.prompt_status if governance is not None else entry.prompt_policy.prompt_status,
        prompt_version=(latest.version if latest else entry.prompt_policy.prompt_version),
        prompt_body=prompt_body,
        schema_contract=schema_contract,
        validator_key=validator_key,
        fallback_policy=fallback_policy,
        max_iterations=entry.prompt_policy.max_iterations,
        prompt_override=prompt_override,
        versions=_list_prompt_versions(db, entry, workspace_id=workspace_id),
    )


def validate_deliverable_prompt(
    entry: DeliverableRegistryEntry,
    payload: DeliverablePromptValidationRequest,
) -> DeliverablePromptValidationResponse:
    required_schema = entry.prompt_policy.schema_contract or entry.quality_policy.schema_contract
    required_validator = entry.prompt_policy.validator_key or entry.quality_policy.validator_key
    schema = (payload.schema_contract or required_schema).strip()
    validator = (payload.validator_key or required_validator).strip()
    fallback = (payload.fallback_policy or entry.prompt_policy.fallback_policy).strip()
    prompt_body = payload.prompt_body.strip()
    errors: list[str] = []
    warnings: list[str] = []

    if not prompt_body:
        errors.append("prompt_body_required")
    if entry.generation_mode in {
        DeliverableGenerationMode.llm_supported,
        DeliverableGenerationMode.llm_required,
        DeliverableGenerationMode.llm_with_deterministic_fallback,
    }:
        if schema != required_schema:
            errors.append("schema_contract_mismatch")
        if validator != required_validator:
            errors.append("validator_key_mismatch")
        if not fallback:
            errors.append("fallback_policy_required")
    if required_schema and required_schema not in prompt_body and "schema_contract" not in prompt_body.lower():
        warnings.append("prompt_does_not_reference_schema_contract")
    if "evidence" not in prompt_body.lower() and "evidencia" not in prompt_body.lower():
        warnings.append("prompt_does_not_reference_traceable_evidence")

    return DeliverablePromptValidationResponse(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        required_schema_contract=required_schema,
        required_validator_key=required_validator,
    )


def update_deliverable_prompt(
    db: Session,
    entry: DeliverableRegistryEntry,
    payload: DeliverablePromptUpdate,
    *,
    workspace_id: UUID | None,
    actor_user_id: UUID | None,
) -> DeliverablePromptResponse:
    validation = validate_deliverable_prompt(
        entry,
        DeliverablePromptValidationRequest(
            prompt_body=payload.prompt_body,
            schema_contract=payload.schema_contract,
            validator_key=payload.validator_key,
            fallback_policy=payload.fallback_policy,
        ),
    )
    if not validation.valid:
        raise ValueError(",".join(validation.errors))

    scope_key = scope_key_for_workspace(workspace_id)
    governance = db.exec(
        select(DeliverableGovernanceRecord).where(
            DeliverableGovernanceRecord.scope_key == scope_key,
            DeliverableGovernanceRecord.deliverable_key == entry.deliverable_key,
        )
    ).first()
    if governance is None:
        governance = DeliverableGovernanceRecord(
            scope_key=scope_key,
            workspace_id=workspace_id,
            deliverable_key=entry.deliverable_key,
        )
    before_payload = {
        "prompt_status": governance.prompt_status,
        "prompt_override": governance.prompt_override,
    }
    schema_contract = payload.schema_contract or entry.prompt_policy.schema_contract or entry.quality_policy.schema_contract
    validator_key = payload.validator_key or entry.prompt_policy.validator_key
    fallback_policy = payload.fallback_policy or entry.prompt_policy.fallback_policy
    prompt_override = {
        **governance.prompt_override,
        "prompt_body": payload.prompt_body,
        "schema_contract": schema_contract,
        "validator_key": validator_key,
        "fallback_policy": fallback_policy,
        "metadata": payload.metadata,
    }
    governance.prompt_status = payload.prompt_status
    governance.prompt_override = prompt_override
    governance.updated_by_user_id = actor_user_id
    governance.updated_at = utc_now()
    db.add(governance)
    db.flush()

    version = payload.version.strip() or f"{entry.prompt_policy.prompt_version}.{len(_list_prompt_versions(db, entry, workspace_id=workspace_id)) + 1}"
    existing_version = db.exec(
        select(DeliverablePromptVersionRecord).where(
            DeliverablePromptVersionRecord.scope_key == scope_key,
            DeliverablePromptVersionRecord.deliverable_key == entry.deliverable_key,
            DeliverablePromptVersionRecord.version == version,
        )
    ).first()
    if existing_version is not None:
        raise ValueError("prompt_version_already_exists")

    version_record = DeliverablePromptVersionRecord(
        scope_key=scope_key,
        workspace_id=workspace_id,
        deliverable_key=entry.deliverable_key,
        version=version,
        status=payload.prompt_status,
        prompt_template_key=entry.prompt_policy.prompt_template_key,
        prompt_body=payload.prompt_body,
        schema_contract=schema_contract,
        validator_key=validator_key,
        fallback_policy=fallback_policy,
        metadata_payload=payload.metadata,
        created_by_user_id=actor_user_id,
    )
    db.add(version_record)
    db.flush()
    after_payload = {
        "prompt_status": governance.prompt_status,
        "prompt_override": governance.prompt_override,
        "prompt_version_id": str(version_record.id),
    }
    changed_fields = [key for key, value in after_payload.items() if before_payload.get(key) != value]
    db.add(
        DeliverablePromptAuditRecord(
            scope_key=scope_key,
            workspace_id=workspace_id,
            deliverable_key=entry.deliverable_key,
            prompt_version_id=version_record.id,
            changed_fields=changed_fields,
            before_payload=before_payload,
            after_payload=after_payload,
            actor_user_id=actor_user_id,
            reason=payload.change_reason,
        )
    )
    db.flush()
    return get_deliverable_prompt(db, entry, workspace_id=workspace_id)
