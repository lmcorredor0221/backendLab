from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from app.services.diagram_center.contracts import DiagramEdge, DiagramModel, DiagramNode
from app.services.diagram_center.quality_service import evaluate_diagram_quality
from app.services.diagram_center.registry_service import build_prompt_spec, load_diagram_registry
from app.services.diagram_center.renderer_service import render_diagram
from app.services.llm_runtime.capability_registry import BuilderCapability, get_builder_capability_spec
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_registry_covers_standardized_diagram_families_and_prompt_specs() -> None:
    registry = load_diagram_registry()
    keys = {entry.key for entry in registry.entries}

    assert len(keys) == 33
    assert {
        "solution_architecture",
        "logical_architecture",
        "physical_architecture",
        "component_diagram",
        "deployment_diagram",
        "class_diagram",
        "entity_relationship",
        "bpmn_process",
        "sequence_diagram",
        "state_diagram",
        "ux_navigation_flow",
        "user_journey",
        "capability_map",
        "context_map",
        "c4_context",
        "c4_container",
        "c4_component",
        "c4_code",
    } <= keys
    prompt = build_prompt_spec(next(entry for entry in registry.entries if entry.key == "sequence_diagram"))
    assert prompt["output_contract"] == "diagram-model.v1"
    assert prompt["notation"] == "sequence"
    assert prompt["semantic_rules"]
    assert prompt["quality_gates"]
    for entry in registry.entries:
        governed_prompt = build_prompt_spec(entry)
        assert governed_prompt["diagram_key"] == entry.key
        assert governed_prompt["version"] == registry.prompt_spec_version
        assert governed_prompt["objective"].strip()
        assert governed_prompt["required_inputs"]
        assert governed_prompt["semantic_rules"]
        assert governed_prompt["output_contract"] == "diagram-model.v1"


def test_diagram_model_quality_and_renderers_use_canonical_graph() -> None:
    model = DiagramModel(
        diagram_key="sequence_diagram",
        title="Secuencia de aprobación",
        notation="sequence",
        nodes=[
            DiagramNode(id="user", label="Usuario", kind="actor", source_refs=["requirement:1"]),
            DiagramNode(id="agent", label="Agente", kind="service", source_refs=["architecture:1"]),
        ],
        edges=[DiagramEdge(id="request", source="user", target="agent", label="Solicita aprobación", order=1)],
        source_refs=["requirement:1", "architecture:1"],
    )

    report = evaluate_diagram_quality(model)
    renderings = render_diagram(model)

    assert report.valid is True
    assert report.score == 100
    assert renderings["mermaid"].startswith("sequenceDiagram")
    assert "<svg" in renderings["svg"]
    assert '"schema_version": "diagram-model.v1"' in renderings["json"]


def test_provider_registry_exposes_diagram_generation_as_governed_capability() -> None:
    spec = get_builder_capability_spec(BuilderCapability.generate_diagram_model)

    assert spec.output_model is DiagramModel
    assert spec.llm_required is True
    assert spec.fallback_policy == "fail_visible_without_synthetic_diagram"
    assert spec.prompt_version == "diagram-prompts.v1.0.0"


def test_v3_catalog_replaces_legacy_local_catalog_and_explains_access(client: TestClient) -> None:
    headers = _auth_headers(client)
    created = client.post("/api/v1/sessions", headers=headers)
    assert created.status_code == 201
    project_id = created.json()["id"]

    response = client.get(f"/api/v3/projects/{project_id}/diagrams", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "diagram-catalog.v3"
    assert payload["total_count"] == 33
    assert payload["provider_key"] in {"openai", "deepseek", "codex_local"}
    assert all("access" in item and item["access"]["reason"] for item in payload["entries"])
    assert any(item["access"]["access_state"] == "available" for item in payload["entries"])
    assert any(item["access"]["access_state"] == "stage_locked" for item in payload["entries"])


def test_generation_job_fails_visibly_when_approved_context_is_missing(client: TestClient) -> None:
    headers = _auth_headers(client)
    created = client.post("/api/v1/sessions", headers=headers)
    project_id = created.json()["id"]

    response = client.post(
        f"/api/v3/projects/{project_id}/diagrams/user_journey/generate",
        headers=headers,
        json={"detail_level": "standard", "reason": "user_request", "idempotency_key": "test-no-context"},
    )

    assert response.status_code == 202
    job_id = response.json()["id"]
    job_response = client.get(f"/api/v3/projects/{project_id}/diagram-jobs/{job_id}", headers=headers)
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "error"
    assert job_response.json()["error_code"] == "approved_context_missing"


def test_governance_reuses_runtime_provider_and_audits_policy_changes(client: TestClient) -> None:
    headers = _auth_headers(client)

    overview = client.get("/api/v3/admin/diagram-governance/overview", headers=headers)
    assert overview.status_code == 200
    assert overview.json()["active_provider"] in {"openai", "deepseek", "codex_local"}
    assert overview.json()["prompt_spec_version"] == "diagram-prompts.v1.0.0"

    updated = client.patch(
        "/api/v3/admin/diagram-governance/activity_diagram",
        headers=headers,
        json={
            "enabled": True,
            "generation_enabled": True,
            "required_tier_override": "blueprint",
            "preview_mode_override": "full",
            "prompt_status": "active",
            "prompt_override": {"objective": "Objetivo de prueba gobernado."},
            "notes": "Cambio de certificación",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["prompt_spec"]["objective"] == "Objetivo de prueba gobernado."

    refreshed = client.get("/api/v3/admin/diagram-governance/overview", headers=headers)
    assert refreshed.status_code == 200
    audit = refreshed.json()["recent_audit"]
    assert audit[0]["diagram_key"] == "activity_diagram"
    assert "prompt_override_hash" in audit[0]["changed_fields"]
