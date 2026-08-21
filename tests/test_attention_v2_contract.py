from __future__ import annotations

import json
from uuid import uuid4

from app.models import (
    AttentionActionRequestV2,
    AttentionActionResultV2,
    AttentionItemResponse,
    AttentionResponse,
    AttentionResponseV2,
    CommercialAccessSnapshotV2,
    CommercialTier,
    SessionRecord,
)
from app.services.attention.adapters.lean_stage import items_from_stage_payload
from app.services.attention.adapters.runtime import items_from_runtime_operation
from app.services.attention.contract import build_attention_response_v2
from app.services.attention.decision_contract import (
    ATTENTION_DECISION_CONTRACT_VERSION_V3,
    AttentionDecisionActionV3,
    AttentionDecisionOptionV3,
    AttentionDecisionSourceV3,
    AttentionDecisionV3,
    decision_to_attention_item_v2,
)
from app.services.attention.governor import govern_attention_items
from app.services.shared_specs import resolve_shared_specs_dir
from app.services.attention.validation_issue_normalizer import validation_issue_to_attention_item

SHARED_SPECS_ROOT = resolve_shared_specs_dir()


def test_attention_v2_schema_files_match_pydantic_model() -> None:
    model_schema = AttentionResponseV2.model_json_schema()
    top_level_schema = json.loads((SHARED_SPECS_ROOT / "attention.v2.schema.json").read_text(encoding="utf-8"))
    canonical_schema = json.loads(
        (SHARED_SPECS_ROOT / "schemas" / "attention.v2.schema.json").read_text(encoding="utf-8")
    )

    assert top_level_schema == model_schema
    assert canonical_schema == model_schema


def test_attention_action_v2_schema_files_match_pydantic_models() -> None:
    for name, model in {
        "attention-action-request.v2": AttentionActionRequestV2,
        "attention-action-result.v2": AttentionActionResultV2,
    }.items():
        model_schema = model.model_json_schema()
        top_level_schema = json.loads((SHARED_SPECS_ROOT / f"{name}.schema.json").read_text(encoding="utf-8"))
        canonical_schema = json.loads(
            (SHARED_SPECS_ROOT / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8")
        )
        assert top_level_schema == model_schema
        assert canonical_schema == model_schema


def test_attention_v2_keys_are_stable_between_reads_of_same_snapshot() -> None:
    payload = {
        "product": "blueprint",
        "stage": "design",
        "source": "design_recommendation",
        "artifact_id": "design-artifact-001",
        "artifact_version": 7,
        "href": "/projects/session-001/design",
        "return_href": "/projects/session-001/design",
        "open_questions": ["Confirma quien aprueba escalamiento por evidencia contradictoria."],
        "gaps": ["Falta definir owner de la decision no delegable."],
    }

    first = items_from_stage_payload(**payload)
    second = items_from_stage_payload(**payload)

    assert [item.key for item in first] == [item.key for item in second]


def test_attention_v2_preserves_stage_source_and_artifact_version() -> None:
    item = items_from_stage_payload(
        product="blueprint",
        stage="tools",
        source="tool_recommendation",
        artifact_id="tools-artifact-002",
        artifact_version=12,
        href="/projects/session-001/tools",
        return_href="/projects/session-001/tools",
        decisions=[
            {
                "key": "document_ingestion_scope",
                "title": "Confirmar alcance de document_ingestion para RAG",
                "reason": "Memoria requiere ingesta de fuentes aprobadas.",
                "impact": "Sin esta herramienta, la estrategia RAG queda incompleta.",
                "severity": "blocking",
                "owner_role": "solution_architect",
            }
        ],
    )[0]

    response = build_attention_response_v2(session_id=uuid4(), workspace_id=uuid4(), current_stage="tools", items=[item])

    assert response.primary_item is not None
    assert response.primary_item.stage == "tools"
    assert response.primary_item.source == "tool_recommendation"
    assert response.primary_item.source_ref.artifact_version == 12
    assert response.counts_by_stage == {"tools": 1}
    assert response.counts_by_product == {"blueprint": 1}


def test_attention_v2_ignores_runtime_fallback_warnings_from_stage_payload() -> None:
    items = items_from_stage_payload(
        product="blueprint",
        stage="define",
        source="journey.definition_artifact",
        artifact_id="define-artifact-001",
        artifact_version=3,
        href="/projects/session-001/define",
        return_href="/projects/session-001/define",
        warnings=[
            "Codex local no pudo ejecutar define_requirements; policy=needs_review_on_provider_or_schema_failure.",
            "OpenAI no pudo normalizar discovery; se uso fallback deterministico.",
            "DeepSeek no esta disponible para recomendar tools minimas; se mantiene el preflight heuristico.",
        ],
    )

    assert items == []


def test_attention_v2_keeps_actionable_stage_warnings_visible() -> None:
    items = items_from_stage_payload(
        product="blueprint",
        stage="define",
        source="journey.definition_artifact",
        artifact_id="define-artifact-002",
        artifact_version=4,
        href="/projects/session-001/define",
        return_href="/projects/session-001/define",
        warnings=[
            "La cobertura funcional aprobada no coincide con los criterios de aceptacion declarados.",
        ],
    )

    assert len(items) == 1
    assert items[0].type == "inconsistency"
    assert items[0].severity == "warning"
    assert items[0].title == "La cobertura funcional aprobada no coincide con los criterios de aceptacion declarados."


def test_attention_decision_v3_serializes_to_attention_v2_item() -> None:
    decision = AttentionDecisionV3(
        item_type="question",
        severity="warning",
        title="Confirmar owner funcional",
        reason="Define necesita ownership para cerrar responsabilidades.",
        consequence_if_unresolved="El diseno podria quedar sin responsable claro.",
        suggested_answer="Lider de soporte",
        source=AttentionDecisionSourceV3(
            product="blueprint",
            stage="define",
            source="journey.definition",
            artifact_id="artifact-1",
            artifact_version=2,
            entity_id="owner",
            field_path="open_questions",
            href="/projects/demo/define",
        ),
        options=[
            AttentionDecisionOptionV3(
                key="support_lead",
                label="Lider de soporte",
                recommended=True,
                confidence=0.8,
            )
        ],
        action=AttentionDecisionActionV3(primary_kind="answer", can_resolve_inline=True),
    )

    item = decision_to_attention_item_v2(decision)

    assert decision.contract_version == ATTENTION_DECISION_CONTRACT_VERSION_V3
    assert item.type == "question"
    assert item.stage == "define"
    assert item.action.kind == "answer"
    assert item.action.can_resolve_inline is True
    assert item.options[0].key == "support_lead"


def test_runtime_errors_are_normalized_without_provider_policy_leak() -> None:
    items = items_from_runtime_operation(
        {
            "id": "run-1",
            "state": "error",
            "stage": "define",
            "product": "blueprint",
            "title": "Codex local no pudo ejecutar define_requirements; policy=needs_review_on_provider_or_schema_failure.",
            "message": "Codex local no pudo ejecutar define_requirements; policy=needs_review_on_provider_or_schema_failure.",
        },
        href="/projects/demo/attention",
        return_href="/projects/demo/define",
    )

    assert len(items) == 1
    item = items[0]
    assert item.type == "runtime_error"
    assert item.action.kind == "retry"
    assert "policy=" not in item.title.lower()
    assert "policy=" not in item.reason.lower()
    assert "codex local no pudo" not in item.title.lower()
    assert item.options[0].key == "retry_generation"
    assert item.options[0].recommended is True
    assert item.action.label == "Reintentar recuperacion"
    assert item.diagnostics is not None
    assert item.diagnostics.error_kind == "provider_or_schema"
    assert item.diagnostics.capability == "define_requirements"
    assert "policy=needs_review_on_provider_or_schema_failure" in item.diagnostics.technical_message
    assert "define_requirements" in item.diagnostics.technical_message
    assert item.diagnostics.repair_hint


def test_runtime_error_diagnostics_redact_secrets_from_technical_message() -> None:
    items = items_from_runtime_operation(
        {
            "id": "run-secret",
            "state": "error",
            "stage": "define",
            "product": "blueprint",
            "title": "Codex local no pudo ejecutar define_requirements",
            "message": "provider failed with api_key=sk-super-secret-value and schema mismatch",
        },
        href="/projects/demo/attention",
        return_href="/projects/demo/define",
    )

    item = items[0]

    assert item.diagnostics is not None
    assert "sk-super-secret-value" not in item.diagnostics.technical_message
    assert "<redacted>" in item.diagnostics.technical_message


def test_successful_skill_summary_does_not_become_runtime_attention_item() -> None:
    items = items_from_runtime_operation(
        {
            "id": "run-summary",
            "state": "blocked",
            "stage": "define",
            "product": "blueprint",
            "title": "requirements_definition_skill",
            "message": "Definition consolidada con 6 FR, 3 NFR y 1 preguntas.",
        },
        href="/projects/demo/attention",
        return_href="/projects/demo/define",
    )

    assert items == []


def test_internal_blocked_runtime_without_human_or_error_signal_is_not_attention() -> None:
    items = items_from_runtime_operation(
        {
            "id": "checkpoint-1",
            "state": "blocked",
            "stage": "define",
            "product": "blueprint",
            "title": "requirements_definition_skill",
            "message": "Resultado de capability sincronizado.",
        },
        href="/projects/demo/attention",
        return_href="/projects/demo/define",
    )

    assert items == []


def test_needs_review_success_summary_is_not_runtime_error_attention() -> None:
    items = items_from_runtime_operation(
        {
            "id": "run-needs-review",
            "state": "blocked",
            "stage": "define",
            "product": "blueprint",
            "title": "requirements_definition_skill",
            "message": "Definition consolidada con 6 FR, 3 NFR y 1 preguntas.",
            "summary": "Capability ejecutada y evaluada por su estado y schema de salida.",
        },
        href="/projects/demo/attention",
        return_href="/projects/demo/define",
    )

    assert items == []


def test_explicit_human_runtime_operation_still_becomes_hitl_attention() -> None:
    items = items_from_runtime_operation(
        {
            "id": "approval-1",
            "state": "blocked",
            "stage": "memory",
            "product": "blueprint",
            "title": "Aprobacion runtime pendiente: memory_profile",
            "message": "La memoria de corto plazo conserva una aprobacion pendiente.",
            "requires_user_action": True,
        },
        href="/projects/demo/attention",
        return_href="/projects/demo/memory",
    )

    assert len(items) == 1
    assert items[0].type == "hitl"
    assert items[0].severity == "blocking"


def test_attention_v2_deduplicates_equivalent_runtime_errors_from_distinct_runs() -> None:
    first = items_from_runtime_operation(
        {
            "id": "run-1",
            "state": "error",
            "stage": "define",
            "product": "blueprint",
            "title": "Codex local no pudo ejecutar define_requirements; policy=needs_review_on_provider_or_schema_failure.",
            "message": "Codex local no pudo ejecutar define_requirements; policy=needs_review_on_provider_or_schema_failure.",
        },
        href="/projects/demo/attention",
        return_href="/projects/demo/define",
    )
    second = items_from_runtime_operation(
        {
            "id": "run-2",
            "state": "blocked",
            "stage": "define",
            "product": "blueprint",
            "title": "requirements_definition_skill",
            "message": "Codex local no pudo ejecutar define_requirements; policy=needs_review_on_provider_or_schema_failure.",
        },
        href="/projects/demo/attention",
        return_href="/projects/demo/define",
    )

    response = build_attention_response_v2(
        session_id=uuid4(),
        workspace_id=uuid4(),
        items=[*first, *second],
        current_stage="define",
    )

    assert response.total_count == 1
    assert response.blocking_count == 1
    assert response.items[0].title == "No se pudo generar Definir automaticamente"


def test_runtime_blocked_recovery_is_normalized_with_decision_options() -> None:
    items = items_from_runtime_operation(
        {
            "id": "run-2",
            "state": "blocked",
            "stage": "define",
            "product": "blueprint",
            "title": "requirements_definition_skill",
            "message": "Codex local no pudo ejecutar define_requirements; policy=needs_review_on_provider_or_schema_failure.",
        },
        href="/projects/demo/attention",
        return_href="/projects/demo/define",
    )

    assert len(items) == 1
    item = items[0]
    assert item.type == "runtime_error"
    assert item.action.kind == "retry"
    assert item.action.can_resolve_inline is True
    assert item.options
    assert "requirements_definition_skill" not in item.title
    assert "policy=" not in item.title.lower()


def test_validation_issue_codes_become_guided_attention_items() -> None:
    item = validation_issue_to_attention_item(
        "untraced_item:FR-001",
        product="blueprint",
        stage="define",
        source="journey.definition",
        artifact_id="artifact-2",
        artifact_version=3,
        href="/projects/demo/define",
        return_href="/projects/demo/define",
    )

    assert item.type == "validation"
    assert item.source_ref.entity_id == "untraced_item:FR-001"
    assert item.source_ref.field_path == "missing_information"
    assert "untraced_item" not in item.title
    assert item.options[0].key == "link_existing_evidence"
    assert item.action.can_resolve_inline is True


def test_basic_blueprint_governor_defers_non_runtime_questions_and_validations() -> None:
    session_id = uuid4()
    workspace_id = uuid4()
    user_id = uuid4()
    record = SessionRecord(
        id=session_id,
        user_id=user_id,
        workspace_id=workspace_id,
        title="Basic project",
        commercial_tier=CommercialTier.blueprint,
    )
    access = CommercialAccessSnapshotV2(
        workspace_id=workspace_id,
        session_id=session_id,
        user_id=user_id,
        tier=CommercialTier.blueprint,
    )
    items = [
        validation_issue_to_attention_item(
            "untraced_item:FR-001",
            product="blueprint",
            stage="define",
            source="journey.definition",
            artifact_id=str(uuid4()),
            artifact_version=3,
            href="/projects/demo/define",
            return_href="/projects/demo/define",
        ),
        *items_from_stage_payload(
            product="blueprint",
            stage="define",
            source="journey.definition",
            artifact_id=str(uuid4()),
            artifact_version=3,
            href="/projects/demo/define",
            return_href="/projects/demo/define",
            open_questions=[
                {
                    "key": "baseline_metric",
                    "question": "Que linea base funcional se usara para medir la reduccion?",
                    "suggested_answer": "Usar linea base manual actual.",
                }
            ],
            warnings=["La definicion contiene blockers de trazabilidad, criterios, NFR o preguntas abiertas."],
        ),
        *items_from_runtime_operation(
            {
                "id": "run-3",
                "state": "blocked",
                "stage": "define",
                "product": "blueprint",
                "title": "requirements_definition_skill",
                "message": "policy=needs_review_on_provider_or_schema_failure.",
            },
            href="/projects/demo/attention",
            return_href="/projects/demo/define",
        ),
    ]

    governed = govern_attention_items(items, record=record, access=access)

    assert [item.type for item in governed] == ["runtime_error"]
    assert governed[0].options


def test_premium_governor_groups_repetitive_validation_findings() -> None:
    session_id = uuid4()
    workspace_id = uuid4()
    user_id = uuid4()
    artifact_id = str(uuid4())
    record = SessionRecord(
        id=session_id,
        user_id=user_id,
        workspace_id=workspace_id,
        title="Premium project",
        commercial_tier=CommercialTier.blueprint_pro,
    )
    access = CommercialAccessSnapshotV2(
        workspace_id=workspace_id,
        session_id=session_id,
        user_id=user_id,
        tier=CommercialTier.blueprint_pro,
    )
    validation_items = [
        validation_issue_to_attention_item(
            issue,
            product="blueprint",
            stage="define",
            source="journey.definition",
            artifact_id=artifact_id,
            artifact_version=7,
            href="/projects/demo/define",
            return_href="/projects/demo/define",
        )
        for issue in [
            "untraced_item:FR-001",
            "untraced_item:FR-002",
            "missing_acceptance_criteria:FR-003",
            "missing_business_rule:BR-001",
        ]
    ]
    generic_warning = items_from_stage_payload(
        product="blueprint",
        stage="define",
        source="journey.definition",
        artifact_id=artifact_id,
        artifact_version=7,
        href="/projects/demo/define",
        return_href="/projects/demo/define",
        warnings=["La definicion contiene blockers de trazabilidad, criterios, NFR o preguntas abiertas."],
    )

    governed = govern_attention_items([*validation_items, *generic_warning], record=record, access=access)

    assert len(governed) == 1
    assert governed[0].type == "validation"
    assert "4 hallazgo" in governed[0].title
    assert governed[0].action.can_resolve_inline is True
    assert [option.key for option in governed[0].options] == ["register_as_enrichment_work", "review_in_stage"]


def test_stage_payload_creates_fallback_options_from_suggested_answer() -> None:
    items = items_from_stage_payload(
        product="blueprint",
        stage="define",
        source="definition",
        artifact_id="definition",
        artifact_version=5,
        href="/projects/demo/define",
        return_href="/projects/demo/attention",
        open_questions=[
            {
                "key": "exception_flow",
                "question": "Que excepciones debe contemplar el flujo?",
                "suggested_answer": "Escalar excepciones a un owner humano con trazabilidad.",
                "impact": "Reduce riesgo operativo.",
            }
        ],
    )

    assert len(items) == 1
    assert items[0].suggested_answer == "Escalar excepciones a un owner humano con trazabilidad."
    assert [option.key for option in items[0].options] == ["accept_suggested_answer", "provide_custom_answer"]


def test_attention_v1_contract_remains_available() -> None:
    response = AttentionResponse(session_id=uuid4(), workspace_id=uuid4())
    item = AttentionItemResponse()

    assert response.contract_version == "attention.v1"
    assert item.type == "info"
    assert "action_label" in item.model_dump()


def test_attention_v2_derives_unblocks_and_resume_action() -> None:
    define_item = items_from_stage_payload(
        product="blueprint",
        stage="define",
        source="journey.definition_artifact",
        artifact_id="define-artifact-001",
        artifact_version=2,
        href="/projects/demo/define",
        return_href="/projects/demo/attention",
        gaps=["Falta especificacion de arquitectura."],
    )[0]

    assert "Desbloquea definicion de requisitos" in define_item.unblocks
    assert define_item.resume_action == "define_requirements"

    tools_item = items_from_stage_payload(
        product="blueprint",
        stage="tools",
        source="tool_recommendation",
        artifact_id="tools-artifact-001",
        artifact_version=1,
        href="/projects/demo/tools",
        return_href="/projects/demo/attention",
        open_questions=["Seleccionar conector de base de datos"],
    )[0]

    assert "Tools" in tools_item.unblocks or "herramientas" in tools_item.unblocks.lower()


def test_attention_v2_deduplication_merges_provenance_and_preserves_distinct_questions() -> None:
    # Two identical questions from different sources
    item_source_a = items_from_stage_payload(
        product="blueprint",
        stage="define",
        source="journey.source_a",
        artifact_id="artifact_a",
        artifact_version=1,
        href="/projects/demo/define",
        return_href="/projects/demo/attention",
        open_questions=["Cual es el SLA requerido para la respuesta del agente?"],
    )[0].model_copy(update={"affected_artifact_refs": ["doc-a.md"]})

    item_source_b = items_from_stage_payload(
        product="blueprint",
        stage="define",
        source="journey.source_b",
        artifact_id="artifact_b",
        artifact_version=1,
        href="/projects/demo/define",
        return_href="/projects/demo/attention",
        open_questions=["Cual es el SLA requerido para la respuesta del agente?"],
    )[0].model_copy(update={"affected_artifact_refs": ["doc-b.md"]})

    # A distinct question that must NOT be deduplicated
    distinct_question = items_from_stage_payload(
        product="blueprint",
        stage="define",
        source="journey.source_a",
        artifact_id="artifact_a",
        artifact_version=1,
        href="/projects/demo/define",
        return_href="/projects/demo/attention",
        open_questions=["Que modelo LLM de razonamiento se debe utilizar?"],
    )[0].model_copy(update={"affected_artifact_refs": ["doc-c.md"]})

    response = build_attention_response_v2(
        session_id=uuid4(),
        workspace_id=uuid4(),
        current_stage="define",
        items=[item_source_a, item_source_b, distinct_question],
    )

    # Must produce 2 unique items (1 merged SLA question + 1 distinct LLM question)
    assert response.total_count == 2
    titles = [item.title for item in response.items]
    assert "Cual es el SLA requerido para la respuesta del agente?" in titles
    assert "Que modelo LLM de razonamiento se debe utilizar?" in titles

    # The merged item must have combined provenance from both sources
    sla_item = next(item for item in response.items if "SLA" in item.title)
    assert set(sla_item.affected_artifact_refs) == {"doc-a.md", "doc-b.md"}
