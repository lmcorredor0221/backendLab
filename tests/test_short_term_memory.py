from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def complete_discovery_payload() -> dict:
    return {
        "problem_statement": "Disenar agentes de soporte con metodologia Lean y bajo riesgo operativo.",
        "current_user": "Arquitecto de soluciones",
        "current_process": "Recoge discovery en documentos, decide arquitectura y luego redacta artefactos manualmente.",
        "desired_outcome": "Generar un blueprint implementable con tools, memoria, evaluacion y seguridad.",
        "autonomy_level": "high",
        "constraints": [
            "Sin microservicios en MVP",
            "No ejecutar side effects irreversibles sin aprobacion humana",
        ],
        "operational_baseline": {
            "current_time_spent": "6 horas por caso",
            "current_cost": "Retrabajo tecnico y validaciones tardias",
            "frequent_errors": [
                "Se pierde contexto entre discovery y blueprint",
                "No se recorta el alcance del MVP",
            ],
            "automation_opportunities": [
                "Normalizar discovery en estructura",
                "Generar artefactos base sin rehacer documentos",
            ],
        },
        "mvp_definition": {
            "v1_scope": [
                "Capturar discovery estructurado",
                "Construir canvas y blueprint inicial",
            ],
            "out_of_scope": [
                "Subagentes operativos",
                "Provisioning automatico",
            ],
            "north_star_metric": "Paquete de implementacion util en una sola sesion",
            "non_delegable_decisions": [
                "Aprobar el handoff a implementacion",
            ],
        },
    }


def build_session_until_canvas(client: TestClient) -> tuple[dict[str, str], str]:
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert normalize_response.status_code == 200
    canvas_response = client.post(f"/api/v1/sessions/{session_id}/build-canvas", headers=headers)
    assert canvas_response.status_code == 200
    return headers, session_id


def test_short_term_memory_supports_resume_rollback_and_reload(client: TestClient) -> None:
    headers, session_id = build_session_until_canvas(client)

    define_runtime_response = client.get(f"/api/v1/sessions/{session_id}/short-term-memory", headers=headers)
    assert define_runtime_response.status_code == 200
    define_runtime = define_runtime_response.json()

    assert define_runtime["active_branch_key"] == "main"
    assert define_runtime["memory"]["active_stage"] == "define"
    assert define_runtime["checkpoint_count"] >= 3
    define_checkpoint_key = define_runtime["active_checkpoint_key"]
    assert define_checkpoint_key

    blueprint_response = client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers)
    assert blueprint_response.status_code == 200

    design_runtime_response = client.get(f"/api/v1/sessions/{session_id}/short-term-memory", headers=headers)
    assert design_runtime_response.status_code == 200
    design_runtime = design_runtime_response.json()
    assert design_runtime["memory"]["active_stage"] == "design"
    assert design_runtime["active_checkpoint_key"] != define_checkpoint_key
    assert any(item["namespace"] == "session.branch_board" for item in design_runtime["memory"]["namespaces"])

    rollback_response = client.post(
        f"/api/v1/sessions/{session_id}/short-term-memory/rollback",
        headers=headers,
        json={"checkpoint_key": define_checkpoint_key, "reason": "Volver al ultimo canvas consistente."},
    )
    assert rollback_response.status_code == 200
    rolled_back_runtime = rollback_response.json()

    assert rolled_back_runtime["active_checkpoint_key"] == define_checkpoint_key
    assert rolled_back_runtime["memory"]["active_stage"] == "define"
    assert rolled_back_runtime["memory"]["active_goal"]

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot_payload = snapshot_response.json()
    assert snapshot_payload["short_term_memory"]["active_checkpoint_key"] == define_checkpoint_key
    assert snapshot_payload["short_term_memory"]["memory"]["active_stage"] == "define"

    reload_response = client.post(f"/api/v1/sessions/{session_id}/short-term-memory/reload", headers=headers)
    assert reload_response.status_code == 200
    reloaded_runtime = reload_response.json()

    assert reloaded_runtime["memory"]["active_stage"] == "design"
    assert reloaded_runtime["active_checkpoint_key"] != define_checkpoint_key
    assert reloaded_runtime["checkpoint_count"] > rolled_back_runtime["checkpoint_count"]


def test_short_term_memory_branch_board_isolates_subagent_runs(client: TestClient) -> None:
    headers, session_id = build_session_until_canvas(client)

    blueprint_response = client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers)
    assert blueprint_response.status_code == 200

    flag_response = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/specialized_subagents_v1",
        headers=headers,
        json={"enabled": True},
    )
    assert flag_response.status_code == 200

    subagent_response = client.post(
        f"/api/v1/sessions/{session_id}/subagents/risk_specialist/run",
        headers=headers,
    )
    assert subagent_response.status_code == 200

    runtime_response = client.get(f"/api/v1/sessions/{session_id}/short-term-memory", headers=headers)
    assert runtime_response.status_code == 200
    runtime_state = runtime_response.json()

    assert runtime_state["active_branch_key"].startswith("subagent_run:")
    assert runtime_state["memory"]["active_stage"] == "design"
    assert runtime_state["branch_count"] >= 2

    branches = {item["branch_key"]: item for item in runtime_state["branch_board"]}
    assert "main" in branches
    assert runtime_state["active_branch_key"] in branches

    active_branch = branches[runtime_state["active_branch_key"]]
    main_branch = branches["main"]

    assert active_branch["topology"] == "subagent"
    assert active_branch["isolation_mode"] == "isolated_namespace"
    assert active_branch["active_checkpoint_key"] != main_branch["active_checkpoint_key"]
    assert set(active_branch["namespace_keys"]) != set(main_branch["namespace_keys"])
    assert any(item["namespace"].startswith("session.branch.") for item in runtime_state["memory"]["namespaces"])
