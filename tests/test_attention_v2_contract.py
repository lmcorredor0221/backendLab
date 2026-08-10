from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from app.models import AttentionActionRequestV2, AttentionActionResultV2, AttentionItemResponse, AttentionResponse, AttentionResponseV2
from app.services.attention.adapters.lean_stage import items_from_stage_payload
from app.services.attention.contract import build_attention_response_v2

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_attention_v2_schema_files_match_pydantic_model() -> None:
    model_schema = AttentionResponseV2.model_json_schema()
    top_level_schema = json.loads((REPO_ROOT / "shared_specs" / "attention.v2.schema.json").read_text(encoding="utf-8"))
    canonical_schema = json.loads(
        (REPO_ROOT / "shared_specs" / "schemas" / "attention.v2.schema.json").read_text(encoding="utf-8")
    )

    assert top_level_schema == model_schema
    assert canonical_schema == model_schema


def test_attention_action_v2_schema_files_match_pydantic_models() -> None:
    for name, model in {
        "attention-action-request.v2": AttentionActionRequestV2,
        "attention-action-result.v2": AttentionActionResultV2,
    }.items():
        model_schema = model.model_json_schema()
        top_level_schema = json.loads((REPO_ROOT / "shared_specs" / f"{name}.schema.json").read_text(encoding="utf-8"))
        canonical_schema = json.loads(
            (REPO_ROOT / "shared_specs" / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8")
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


def test_attention_v1_contract_remains_available() -> None:
    response = AttentionResponse(session_id=uuid4(), workspace_id=uuid4())
    item = AttentionItemResponse()

    assert response.contract_version == "attention.v1"
    assert item.type == "info"
    assert "action_label" in item.model_dump()
