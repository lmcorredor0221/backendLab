from __future__ import annotations

from uuid import UUID
from fastapi.testclient import TestClient
import pytest
from sqlmodel import select

from app.db import get_session
from app.models import SessionRecord, SessionStage, UserRecord
from app.services.diagram_center.contracts import DiagramEdge, DiagramModel, DiagramNode
from app.services.diagram_center.persistence import DiagramGenerationJobRecord
from app.services.diagram_center.generation_service import run_generation_job
from app.services.openai_builder import BuilderProviderFacade, LLMArtifactResult, OpenAIBuilderService
from app.services.product_processing.contracts import ProductBuildLifecycle
from app.services.product_processing.persistence import ProductBuildRunRecord, ProductBuildStepRecord
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


def _fake_generate_diagram_model(self, payload, context_bundle=None) -> LLMArtifactResult:
    if payload.diagram_key == "agent_orchestration":
        model = DiagramModel(
            diagram_key=payload.diagram_key,
            title=payload.title or "Orquestacion agentiva",
            notation=payload.notation.value if hasattr(payload.notation, "value") else str(payload.notation),
            nodes=[
                DiagramNode(id="orchestrator", label="Orquestador principal", kind="orchestrator", agent_kind="orchestrator", source_refs=["session.baseline"]),
                DiagramNode(id="analysis_agent", label="Agente de analisis", kind="agent", agent_kind="worker", source_refs=["session.baseline"]),
                DiagramNode(id="design_agent", label="Agente de diseno", kind="agent", agent_kind="worker", source_refs=["session.baseline"]),
                DiagramNode(id="memory", label="Memoria RAG y checkpoints", kind="memory", memory_kind="vector_store", source_refs=["session.baseline"]),
                DiagramNode(id="tools", label="Herramientas MCP y APIs", kind="tool", tool_kind="mcp_server", source_refs=["session.baseline"]),
                DiagramNode(id="guardrail", label="Guardrails y aprobacion HITL", kind="guardrail_gate", tool_kind="guardrail_gate", source_refs=["session.baseline"]),
                DiagramNode(id="fallback", label="Fallback y escalamiento", kind="fallback", source_refs=["session.baseline"]),
                DiagramNode(id="output", label="Resultado entregable", kind="output", source_refs=["session.baseline"]),
            ],
            edges=[
                DiagramEdge(id="e1", source="orchestrator", target="analysis_agent", kind="handoff", label="delega analisis", order=1, source_refs=["session.baseline"]),
                DiagramEdge(id="e2", source="analysis_agent", target="design_agent", kind="handoff", label="transfiere contexto", order=2, source_refs=["session.baseline"]),
                DiagramEdge(id="e3", source="design_agent", target="memory", kind="checkpoint_resume", label="actualiza memoria", order=3, source_refs=["session.baseline"]),
                DiagramEdge(id="e4", source="design_agent", target="tools", kind="tool_call", label="usa herramientas", order=4, source_refs=["session.baseline"]),
                DiagramEdge(id="e5", source="tools", target="guardrail", kind="decision", label="control HITL", order=5, source_refs=["session.baseline"]),
                DiagramEdge(id="e6", source="guardrail", target="fallback", kind="escalation", label="escala error", order=6, source_refs=["session.baseline"]),
                DiagramEdge(id="e7", source="fallback", target="output", kind="retry", label="reintento controlado", order=7, source_refs=["session.baseline"]),
            ],
            source_refs=["session.baseline"],
        )
        return LLMArtifactResult(
            artifact=model,
            provider_key="mock",
            model_name="mock-model",
            prompt_version="1.0.0",
        )

    model = DiagramModel(
        diagram_key=payload.diagram_key,
        title=payload.title or "Diagrama",
        notation=payload.notation.value if hasattr(payload.notation, "value") else str(payload.notation),
        nodes=[
            DiagramNode(id="node_1", label="Componente 1", kind="service", source_refs=["session.baseline"]),
            DiagramNode(id="node_2", label="Componente 2", kind="database", source_refs=["session.baseline"]),
        ],
        edges=[
            DiagramEdge(id="edge_1", source="node_1", target="node_2", label="conecta", order=1),
        ],
        source_refs=["session.baseline"],
    )
    return LLMArtifactResult(
        artifact=model,
        provider_key="mock",
        model_name="mock-model",
        prompt_version="1.0.0",
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(OpenAIBuilderService, "generate_diagram_model", _fake_generate_diagram_model)
    monkeypatch.setattr(BuilderProviderFacade, "generate_diagram_model", _fake_generate_diagram_model)
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _db_session_from_client(client: TestClient):
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    return next(session_generator)


def test_blueprint_basic_preparation_and_actions_close_deterministically(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    # 1) Execute blueprint commercial result preparation
    prep_response = client.post(f"/api/v1/sessions/{session_id}/blueprint/commercial-result", headers=headers)
    assert prep_response.status_code == 200
    snapshot = prep_response.json()
    assert snapshot["session"]["id"] == session_id

    # 2) agent_orchestration must be the first automatic diagram job in Blueprint Free.
    db = _db_session_from_client(client)
    try:
        all_diagram_jobs = db.exec(
            select(DiagramGenerationJobRecord)
            .where(DiagramGenerationJobRecord.session_id == UUID(session_id))
            .order_by(DiagramGenerationJobRecord.requested_at.asc())
        ).all()
        assert all_diagram_jobs
        assert all_diagram_jobs[0].diagram_key == "agent_orchestration"

        diagram_jobs = db.exec(
            select(DiagramGenerationJobRecord).where(
                DiagramGenerationJobRecord.session_id == UUID(session_id),
                DiagramGenerationJobRecord.status == "queued",
            )
        ).all()
        for job in diagram_jobs:
            run_generation_job(job.id, db.get_bind())
    finally:
        db.close()

    # 3) Check product build status on GET: must automatically reconcile to completed and 100%
    status_response = client.get(f"/api/v1/sessions/{session_id}/product-builds/blueprint_basic", headers=headers)
    assert status_response.status_code == 200
    status_data = status_response.json()

    assert status_data["product_key"] == "blueprint_basic"
    assert status_data["lifecycle"] == "completed"
    assert status_data["progress"]["percent"] == 100
    assert status_data["progress"]["completed_units"] == status_data["progress"]["total_units"]

    # 4) Test action endpoint POST /actions with resume and retry
    action_response = client.post(
        f"/api/v1/sessions/{session_id}/product-builds/blueprint_basic/actions",
        headers=headers,
        json={"action": "resume", "idempotency_key": "resume-test-1"},
    )
    if action_response.status_code != 200:
        print(f"DEBUG action_response 422: {action_response.json()}")
    assert action_response.status_code == 200
    action_data = action_response.json()
    assert action_data["lifecycle"] == "completed"
    assert action_data["progress"]["percent"] == 100

    retry_response = client.post(
        f"/api/v1/sessions/{session_id}/product-builds/blueprint_basic/actions",
        headers=headers,
        json={"action": "retry", "idempotency_key": "retry-test-1"},
    )
    assert retry_response.status_code == 200
    retry_data = retry_response.json()
    assert retry_data["lifecycle"] == "completed"
    assert retry_data["progress"]["percent"] == 100
