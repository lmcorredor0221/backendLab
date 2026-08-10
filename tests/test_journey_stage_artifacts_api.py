from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.db import get_session
from app.models import WorkspaceRole
from tests.api_testkit import TEST_PASSWORD, build_test_client
from tests.test_sessions_api import (
    auth_headers,
    auth_headers_for_credentials,
    build_session_flow,
    build_session_flow_for_headers,
    create_workspace_for_user,
    seed_user,
)


SECONDARY_EMAIL = "workspace-b@leanbuilder.local"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def _minimal_tool_payload() -> dict:
    return {
        "approved_tools": [
            {
                "name": "approval_gate",
                "purpose": "Capturar la decision humana antes de ejecutar acciones sensibles.",
                "owner": "business_owner_pending",
                "archetype": "governance_gate",
                "integration_kind": "governed_handoff",
                "requires_approval": True,
                "inputs": ["proposed_action", "evidence_bundle"],
                "outputs": ["approval_decision"],
                "validations": ["approval_payload_validation"],
                "permissions": ["request_approval"],
                "scopes": ["workspace", "human_in_the_loop"],
                "audit_rules": ["Registrar approver y rationale."],
                "approval_policy": "Requiere decision humana explicita.",
                "contract_review_state": "needs-review",
            }
        ]
    }


def _minimal_memory_payload() -> dict:
    return {
        "schema_version": "memory-recommendation.v1",
        "summary": "Memoria aprobada para validar invalidacion downstream.",
        "memory_need_decision": {
            "mode": "hybrid_memory",
            "required": True,
            "summary": "Usar memoria hibrida con RAG gobernado.",
            "rationale": "Permite grounding sin inflar la ventana de contexto.",
            "source_refs": ["journey.tools"],
        },
        "short_term_design": {
            "layer_key": "short_term",
            "label": "Short-term memory",
            "owner": "orchestrator",
            "summary": "Resumen operativo de la sesion.",
            "stores": ["working_memory"],
            "write_triggers": ["message_received"],
            "read_paths": ["response_generation"],
            "compaction_policy": "rolling_summary",
            "retention_policy": "session",
        },
        "working_memory_design": {
            "layer_key": "working_memory",
            "label": "Working memory",
            "owner": "orchestrator",
            "summary": "Decisiones y contexto activo del flujo.",
            "stores": ["session_store"],
            "write_triggers": ["decision_approved"],
            "read_paths": ["tool_selection", "memory_planning"],
            "compaction_policy": "checkpoint_based",
            "retention_policy": "session",
        },
        "long_term_design": {
            "layer_key": "long_term",
            "label": "Long-term memory",
            "owner": "workspace_memory_owner",
            "summary": "Artefactos aprobados y conocimiento reutilizable.",
            "stores": ["knowledge_store"],
            "write_triggers": ["artifact_approved"],
            "read_paths": ["rag_retrieval"],
            "compaction_policy": "curated_artifacts_only",
            "retention_policy": "workspace",
        },
        "knowledge_design": {
            "mode": "rag",
            "rag_required": True,
            "summary": "RAG gobernado con fuentes aprobadas.",
            "source_scope": "workspace",
            "approved_sources": [],
            "notes": ["Usar solo artefactos aprobados y corpus gobernado."],
        },
        "context_budget_plan": [
            {
                "role": "orchestrator",
                "task_kind": "response_generation",
                "max_context_tokens": 8000,
                "max_short_term_items": 10,
                "max_retrieved_sources": 4,
                "strategy": "summary_plus_rag",
                "source_refs": ["journey.memory"],
            }
        ],
        "write_read_matrix": [
            {
                "scope": "session",
                "owner": "orchestrator",
                "write_when": "decision_approved",
                "do_not_write_when": "insufficient_evidence",
                "read_when": "before_responding",
                "compact_when": "token_pressure",
            }
        ],
        "retention_and_deletion": [
            {
                "scope": "session",
                "retention_policy": "retain_approved_decisions",
                "ttl_policy": "90d",
                "deletion_policy": "soft_delete",
                "residency": "workspace",
                "source_refs": ["journey.memory"],
            }
        ],
        "sensitivity_and_isolation": [
            {
                "scope": "workspace",
                "isolation_mode": "tenant_isolation",
                "data_classes": ["operational"],
                "restrictions": ["no_cross_workspace_access"],
                "source_refs": ["journey.memory"],
            }
        ],
        "tool_dependencies": [
            {
                "tool_key": "knowledge_retrieval",
                "required": True,
                "status": "approved",
                "reason": "RAG requiere retrieval aprobado.",
                "capabilities": ["retrieval"],
            },
            {
                "tool_key": "document_ingestion",
                "required": True,
                "status": "approved",
                "reason": "RAG requiere ingesta aprobada.",
                "capabilities": ["ingestion", "refresh"],
            },
        ],
        "confidence": {
            "overall": 0.8,
            "band": "high",
            "rationale": "Cobertura suficiente para aprobar estrategia.",
        },
        "dry_compile_status": {
            "status": "ready",
            "summary": "Compilacion seca consistente.",
            "generated_contracts": ["memory_profile", "knowledge_profile"],
            "blocking_issues": [],
        },
        "review_state": "complete",
    }


def _minimal_design_payload() -> dict:
    alternative = {
        "alternative_key": "single-agent",
        "label": "Single agent",
        "architecture": "single_agent_with_skills",
        "reasoning_pattern": "ReAct",
        "coordination_model": "single-agent",
        "summary": "Un solo agente con skills especializados y approval gate.",
        "topology": "Agente unico con skills y handoff controlado.",
        "roles": [
            {
                "key": "orchestrator",
                "title": "Orchestrator",
                "responsibility": "Consulta sistemas, aplica reglas y coordina aprobaciones.",
                "limits": ["No ejecuta side effects sin approval"],
            }
        ],
        "handoffs": [],
        "approval_points": ["Promocion a implementacion"],
        "decision_policy": "Mantener el MVP simple y trazable.",
        "blueprint_projection": {
            "architecture": "single_agent_with_skills",
            "reasoning_pattern": "ReAct",
            "guardrails": ["Toda escritura requiere aprobacion humana y audit trail."],
            "narrative": "Arquitectura simple con control humano.",
        },
    }
    return {
        "schema_version": "design-recommendation.v1",
        "alternatives": [alternative],
        "recommended_alternative_key": alternative["alternative_key"],
        "selected_design": alternative,
        "requirements_coverage": [],
        "confidence": {"overall": 0.82, "band": "high", "rationale": "Cobertura suficiente."},
        "review_state": "complete",
        "summary": "Design aprobado para Herramientas.",
    }


def _prepare_design_tools_memory_chain(
    client: TestClient,
    *,
    headers: dict[str, str],
    session_id: str,
) -> tuple[dict, dict, dict]:
    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    create_design = client.post(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts",
        headers=headers,
        json={
            "artifact_kind": "design_recommendation_artifact",
            "source_action": "manual_design_ci1",
            "proposal_payload": _minimal_design_payload(),
        },
    )
    assert create_design.status_code == 200
    design_artifact = create_design.json()

    approve_design = client.post(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts/{design_artifact['id']}/approve",
        headers=headers,
        json={"note": "Promover design valido antes de tools."},
    )
    assert approve_design.status_code == 200
    assert approve_design.json()["state"] == "approved"

    create_tools = client.post(
        f"/api/v1/sessions/{session_id}/journey/tools/artifacts",
        headers=headers,
        json={
            "artifact_kind": "tool_recommendation_artifact",
            "source_action": "manual_tools_ci1",
            "proposal_payload": _minimal_tool_payload(),
        },
    )
    assert create_tools.status_code == 200
    tools_artifact = create_tools.json()

    approve_tools = client.post(
        f"/api/v1/sessions/{session_id}/journey/tools/artifacts/{tools_artifact['id']}/approve",
        headers=headers,
        json={"note": "Aprobar set minimo de tools para continuar."},
    )
    assert approve_tools.status_code == 200
    assert approve_tools.json()["state"] == "approved"

    create_memory = client.post(
        f"/api/v1/sessions/{session_id}/journey/memory/artifacts",
        headers=headers,
        json={
            "artifact_kind": "memory_recommendation_artifact",
            "source_action": "manual_memory_ci1",
            "proposal_payload": _minimal_memory_payload(),
        },
    )
    assert create_memory.status_code == 200
    memory_artifact = create_memory.json()

    approve_memory = client.post(
        f"/api/v1/sessions/{session_id}/journey/memory/artifacts/{memory_artifact['id']}/approve",
        headers=headers,
        json={"note": "Aprobar memoria para validar downstream invalidation."},
    )
    assert approve_memory.status_code == 200
    assert approve_memory.json()["state"] == "approved"

    return snapshot, design_artifact, memory_artifact


def test_snapshot_backfills_journey_artifacts_idempotently(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    first_snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert first_snapshot.status_code == 200
    second_snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert second_snapshot.status_code == 200

    first_payload = first_snapshot.json()
    second_payload = second_snapshot.json()

    first_keys = {(item["stage_key"], item["version_number"]) for item in first_payload["journey_artifacts"]}
    second_keys = {(item["stage_key"], item["version_number"]) for item in second_payload["journey_artifacts"]}

    assert first_keys == second_keys
    assert {"discover", "define", "design", "memory"}.issubset(first_payload["journey_latest_artifacts"].keys())


def test_journey_artifact_routes_invalidate_downstream_after_upstream_approval(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    snapshot, design_legacy, _ = _prepare_design_tools_memory_chain(
        client,
        headers=headers,
        session_id=str(session_id),
    )

    revised_design = client.patch(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts/{design_legacy['id']}",
        headers=headers,
        json={
            "proposal_payload": dict(design_legacy["proposal_payload"]),
            "note": "Nuevo diseno derivado del review humano.",
        },
    )
    assert revised_design.status_code == 200
    revised_payload = revised_design.json()
    assert revised_payload["version_number"] > design_legacy["version_number"]

    approve_design = client.post(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts/{revised_payload['id']}/approve",
        headers=headers,
        json={"note": "Aprobar revision de arquitectura."},
    )
    assert approve_design.status_code == 200

    latest_tools = client.get(
        f"/api/v1/sessions/{session_id}/journey/tools/artifacts/latest",
        headers=headers,
    )
    latest_memory = client.get(
        f"/api/v1/sessions/{session_id}/journey/memory/artifacts/latest",
        headers=headers,
    )
    assert latest_tools.status_code == 200
    assert latest_memory.status_code == 200
    assert latest_tools.json()["state"] == "stale"
    assert latest_memory.json()["state"] == "stale"
    assert any("upstream_design_artifact" in item for item in latest_tools.json()["stale_reasons"])
    assert any("upstream_design_artifact" in item for item in latest_memory.json()["stale_reasons"])


def test_journey_artifact_routes_invalidate_downstream_when_upstream_reprocess_starts(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    snapshot, design_legacy, _ = _prepare_design_tools_memory_chain(
        client,
        headers=headers,
        session_id=str(session_id),
    )

    revised_design = client.patch(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts/{design_legacy['id']}",
        headers=headers,
        json={
            "proposal_payload": dict(design_legacy["proposal_payload"]),
            "note": "Nuevo diseno reprocesado.",
        },
    )
    assert revised_design.status_code == 200
    revised_payload = revised_design.json()
    assert revised_payload["state"] == "reviewed"
    assert revised_payload["version_number"] > design_legacy["version_number"]

    latest_tools = client.get(
        f"/api/v1/sessions/{session_id}/journey/tools/artifacts/latest",
        headers=headers,
    )
    latest_memory = client.get(
        f"/api/v1/sessions/{session_id}/journey/memory/artifacts/latest",
        headers=headers,
    )
    assert latest_tools.status_code == 200
    assert latest_memory.status_code == 200
    assert latest_tools.json()["state"] == "stale"
    assert latest_memory.json()["state"] == "stale"
    assert any(reason.endswith("_regenerated") for reason in latest_tools.json()["stale_reasons"])
    assert any(reason.endswith("_regenerated") for reason in latest_memory.json()["stale_reasons"])


def test_journey_artifact_approval_rejects_non_latest_version(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    design_legacy = snapshot["journey_latest_artifacts"]["design"]

    first_revision = client.patch(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts/{design_legacy['id']}",
        headers=headers,
        json={
            "proposal_payload": {
                "architecture": "supervisor_first",
                "reasoning_pattern": "planner_critic",
                "safety_checks": snapshot["blueprint"]["safety_checks"],
                "guardrails": snapshot["blueprint"]["guardrails"],
                "narrative": "Revision 1",
            },
        },
    )
    second_revision = client.patch(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts/{design_legacy['id']}",
        headers=headers,
        json={
            "proposal_payload": {
                "architecture": "supervisor_second",
                "reasoning_pattern": "planner_critic",
                "safety_checks": snapshot["blueprint"]["safety_checks"],
                "guardrails": snapshot["blueprint"]["guardrails"],
                "narrative": "Revision 2",
            },
        },
    )
    assert first_revision.status_code == 200
    assert second_revision.status_code == 200
    first_revision_id = first_revision.json()["id"]

    stale_approval = client.post(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts/{first_revision_id}/approve",
        headers=headers,
        json={"note": "Intento de aprobar una version no vigente."},
    )
    assert stale_approval.status_code == 409
    assert "latest version" in stale_approval.json()["detail"]


def test_journey_artifact_routes_respect_workspace_isolation(client: TestClient) -> None:
    owner_headers = auth_headers(client)
    session_id = build_session_flow_for_headers(client, owner_headers)
    client.get(f"/api/v1/sessions/{session_id}", headers=owner_headers)

    seed_user(
        client,
        email=SECONDARY_EMAIL,
        password=TEST_PASSWORD,
        full_name="Workspace B",
    )
    create_workspace_for_user(client, email=SECONDARY_EMAIL, name="Workspace B", role=WorkspaceRole.editor)
    other_headers = auth_headers_for_credentials(client, email=SECONDARY_EMAIL, password=TEST_PASSWORD)

    forbidden = client.get(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts",
        headers=other_headers,
    )
    assert forbidden.status_code == 404
