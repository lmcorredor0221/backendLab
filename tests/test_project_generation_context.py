from __future__ import annotations

from app.services.deliverable_catalog.contracts import DeliverableContextPolicy
from app.services.deliverable_catalog.project_generation_context import (
    ProjectGenerationContext,
    merge_generation_contexts,
)


def _policy(*refs: str) -> DeliverableContextPolicy:
    return DeliverableContextPolicy(short_term_refs=list(refs), max_context_tokens=5000)


def test_formal_approved_context_extracts_core_project_fields() -> None:
    payload = {
        "project_title": "Mesa de ayudas clinica",
        "approved_context_refs": ["journey:discover:v1", "journey:define:v2", "journey:design:v3"],
        "approved_context": {
            "stages": {
                "discover": {
                    "problem_statement": "Las solicitudes de soporte clinico se clasifican tarde.",
                    "current_process": "Radicacion manual en correo.",
                    "current_user": "Coordinador de soporte clinico",
                    "desired_outcome": "Priorizar casos urgentes en menos de cinco minutos.",
                },
                "define": {
                    "mvp_scope": ["Clasificar solicitudes por urgencia", "Escalar casos criticos"],
                    "out_of_scope": ["Autorizar procedimientos medicos"],
                    "nondelegable_decisions": ["Aprobar cambio de prioridad clinica"],
                },
                "design": {
                    "architecture": "router_triage_with_human_gate",
                    "reasoning_pattern": "structured_triage",
                    "guardrails": ["No emitir diagnosticos medicos"],
                },
                "tools": {
                    "tools": [{"name": "Zendesk", "purpose": "Leer tickets aprobados"}],
                },
                "memory": {
                    "memory_strategy": "case_summary_checkpoints",
                    "rag_required": True,
                    "knowledge_sources": [{"name": "Manual interno de triage", "permissions": "solo lectura"}],
                },
            }
        },
    }

    context = ProjectGenerationContext.from_approved_payload(
        payload,
        deliverable_key="blueprint.architecture_spec",
        policy=_policy("stage.discover", "stage.define", "stage.design", "stage.tools", "stage.memory"),
    )

    assert context.context_version == "project-generation-context.v1"
    assert context.problem_statement == "Las solicitudes de soporte clinico se clasifican tarde."
    assert context.current_user == "Coordinador de soporte clinico"
    assert context.architecture == "router_triage_with_human_gate"
    assert context.tools[0].name == "Zendesk"
    assert context.memory_strategy == "case_summary_checkpoints"
    assert context.rag_required is True
    assert {anchor.anchor_type for anchor in context.specificity_anchors} >= {"actor", "problem", "tool"}
    assert "architecture" not in {field.field for field in context.missing_fields}


def test_flat_snapshot_gives_equivalent_fields_without_fake_defaults() -> None:
    context = ProjectGenerationContext.from_snapshot(
        {
            "summary": "Automatizar aprobaciones de compras con evidencia trazable.",
            "current_user": "Analista de compras",
            "desired_outcome": "Reducir retrabajo en aprobaciones.",
            "mvp_scope": ["Validar solicitud contra presupuesto aprobado"],
            "architecture": "approval_router",
            "tools": [{"name": "ERP compras", "purpose": "Consultar ordenes"}],
            "memory_strategy": "session_audit_log",
        },
        deliverable_key="definition.requirements",
    )

    assert context.problem_statement == "Automatizar aprobaciones de compras con evidencia trazable."
    assert context.current_user == "Analista de compras"
    assert context.mvp_scope == ["Validar solicitud contra presupuesto aprobado"]
    assert context.tools[0].name == "ERP compras"
    assert all("Usuario Operativo" not in anchor.value for anchor in context.specificity_anchors)


def test_unknown_field_stays_missing_and_acp_question_is_not_promoted_to_fact() -> None:
    context = ProjectGenerationContext.from_acp_inputs(
        {"project_title": "Asistente de cartera"},
        [
            {
                "question_key": "erp_binding",
                "question_text": "Que ERP autorizado debe usarse para consultar facturas?",
                "answer_text": "",
                "status": "open",
                "blocking": True,
            }
        ],
        {},
        deliverable_key="acp.tools",
    )

    assert context.tools == []
    assert "tools" in {field.field for field in context.missing_fields}
    assert context.construction_questions[0].question_text.startswith("Que ERP autorizado")
    assert context.open_questions[0].question == context.construction_questions[0].question_text


def test_input_fingerprint_changes_with_source_version_but_not_ref_order() -> None:
    base_payload = {
        "approved_context_refs": ["journey:discover:v1", "journey:define:v1"],
        "approved_context": {
            "stages": {
                "discover": {"problem_statement": "Clasificar leads B2B por fit."},
                "define": {"mvp_scope": ["Asignar score comercial"]},
            }
        },
    }
    reordered_payload = {
        **base_payload,
        "approved_context_refs": ["journey:define:v1", "journey:discover:v1"],
    }
    changed_payload = {
        **base_payload,
        "approved_context_refs": ["journey:discover:v2", "journey:define:v1"],
    }

    first = ProjectGenerationContext.from_approved_payload(
        base_payload,
        deliverable_key="definition.requirements",
        policy=_policy("stage.discover", "stage.define"),
    )
    reordered = ProjectGenerationContext.from_approved_payload(
        reordered_payload,
        deliverable_key="definition.requirements",
        policy=_policy("stage.discover", "stage.define"),
    )
    changed = ProjectGenerationContext.from_approved_payload(
        changed_payload,
        deliverable_key="definition.requirements",
        policy=_policy("stage.discover", "stage.define"),
    )

    assert first.input_fingerprint == reordered.input_fingerprint
    assert first.input_fingerprint != changed.input_fingerprint


def test_context_policy_blocks_unrequested_stages() -> None:
    context = ProjectGenerationContext.from_approved_payload(
        {
            "approved_context": {
                "stages": {
                    "discover": {"problem_statement": "Resolver tickets repetidos."},
                    "design": {"architecture": "specialized_workers"},
                }
            }
        },
        deliverable_key="discovery.analysis",
        policy=_policy("stage.discover"),
    )

    assert context.problem_statement == "Resolver tickets repetidos."
    assert context.architecture is None
    assert "architecture" in {field.field for field in context.missing_fields}


def test_merge_generation_contexts_only_fills_missing_fields() -> None:
    approved = ProjectGenerationContext.from_snapshot(
        {"problem_statement": "Aprobar reembolsos con control humano."},
        deliverable_key="discovery.analysis",
    )
    fallback = ProjectGenerationContext.from_snapshot(
        {
            "problem_statement": "No debe sobrescribir",
            "architecture": "approval_router",
            "tools": [{"name": "ERP financiero"}],
        },
        deliverable_key="discovery.analysis",
    )

    merged = merge_generation_contexts(approved, fallback)

    assert merged.problem_statement == "Aprobar reembolsos con control humano."
    assert merged.architecture == "approval_router"
    assert merged.tools[0].name == "ERP financiero"
