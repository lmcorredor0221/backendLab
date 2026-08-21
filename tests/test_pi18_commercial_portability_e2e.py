from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from app.services.shared_specs import resolve_shared_specs_dir
from tests.api_testkit import build_test_client
from tests.test_sessions_api import (
    approve_design_for_session,
    approve_memory_for_session,
    approve_validate_for_session,
    auth_headers,
    auth_headers_for_credentials,
    complete_discovery_payload,
    seed_user,
    upgrade_session_tier,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONSUMER_PATH = resolve_shared_specs_dir() / "reference_consumers" / "python"
sys.path.insert(0, str(REFERENCE_CONSUMER_PATH))

from acp_zip_reference_consumer import validate_acp_zip  # noqa: E402


ANSWER_MAP = {
    "knowledge_sources": "name=Confluence; type=wiki; owner=ops; frequency=diaria",
    "knowledge_ingestion": "strategy=sync_incremental; frequency=diaria; mechanism=cron; owner=ops",
    "knowledge_embedding_strategy": "provider=text-embedding-3-small; chunking=800_tokens_overlap_120; notes=openai",
    "runtime_fallback_model": "model=gpt-4.1-mini; condition=cuando falle el modelo primario",
    "runtime_vector_store": "vector_store=pgvector; notes=misma base local",
    "runtime_secret_source": "source=.env local protegido; owner=platform_owner; environment=desarrollo",
    "deployment_target": "target=local_vm; restrictions=solo red interna y acceso por VPN",
    "deployment_image_strategy": "strategy=local_windows_service; registry=no_aplica",
    "deployment_network_constraints": "network=solo red interna; secrets=.env local; dependencies=postgres local",
}


def close_definition_review_gaps(definition_payload: dict) -> dict:
    traceability = {
        item.get("requirement_key"): item
        for item in definition_payload.get("traceability", [])
        if item.get("requirement_key")
    }

    def enrich_item(item: dict, *, trace: bool, default_source: str) -> None:
        item["status"] = "accepted" if item.get("status") in {"", "proposed", "needs_input"} else item.get("status", "accepted")
        if not item.get("source_refs"):
            item["source_refs"] = [default_source, "pi18.review"]
        if trace and item.get("key") and item["key"] not in traceability:
            traceability[item["key"]] = {
                "key": f"pi18-trace-{len(traceability) + 1}-{item['key']}",
                "requirement_key": item["key"],
                "source_ref": item["source_refs"][0],
                "rationale": "PI18 cerro trazabilidad explicita antes de aprobar Define.",
                "coverage_status": "covered",
            }
        if trace and not item.get("acceptance"):
            item["acceptance"] = [
                "Existe evidencia trazable hacia Discovery y el canvas aprobado.",
                "El comportamiento puede validarse mediante escenarios representativos antes de construir.",
            ]

    for requirement in definition_payload.get("functional_requirements", []):
        enrich_item(requirement, trace=True, default_source="canvas.mvp_scope")

    for requirement in definition_payload.get("non_functional_requirements", []):
        if not requirement.get("metric"):
            requirement["metric"] = "cumplimiento_operativo"
        if not requirement.get("target"):
            requirement["target"] = ">= 95% de ejecuciones dentro del flujo definido"
        enrich_item(requirement, trace=True, default_source="discovery.constraints")
        requirement["acceptance"] = [
            *requirement.get("acceptance", []),
            f"Medicion definida: {requirement['metric']} con objetivo {requirement['target']}.",
        ]

    for rule in definition_payload.get("business_rules", []):
        enrich_item(rule, trace=True, default_source="discovery.business_rules")

    for criterion in definition_payload.get("acceptance_criteria", []):
        enrich_item(criterion, trace=True, default_source="discovery.mvp_definition")

    for dependency in definition_payload.get("dependencies", []):
        enrich_item(dependency, trace=True, default_source="discovery.integrations")

    for assumption in definition_payload.get("assumptions", []):
        enrich_item(assumption, trace=True, default_source="discovery.assumptions")

    for question in definition_payload.get("open_questions", []):
        enrich_item(question, trace=False, default_source="pi18.implementation_decisions")
        question["blocking"] = False
        question["status"] = "accepted"
        if not question.get("suggested_answer"):
            question["suggested_answer"] = "Cerrar durante implementacion si depende del entorno tecnico real."

    definition_payload["traceability"] = list(traceability.values())
    definition_payload["validation"] = {
        **definition_payload.get("validation", {}),
        "duplicate_keys": [],
        "duplicate_signals": [],
        "contradictions": [],
        "vague_nfrs": [],
        "missing_acceptance": [],
        "untraced_items": [],
        "blocking_open_questions": [],
        "blocking_issues": [],
        "coverage_ratio": 1.0,
    }
    return definition_payload


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def answer_acp_questions(client: TestClient, headers: dict[str, str], session_id: str) -> None:
    response = client.get(f"/api/v1/sessions/{session_id}/acp/questions", headers=headers)
    assert response.status_code == 200
    for question in response.json():
        question_key = str(question["question_key"])
        update_response = client.patch(
            f"/api/v1/sessions/{session_id}/acp/questions/{question_key}",
            headers=headers,
            json={
                "answer_text": ANSWER_MAP.get(
                    question_key,
                    f"resolved={question_key}; source=pi18_e2e; owner=platform_owner",
                ),
                "owner_role": "platform_owner",
                "impacted_artifacts": question.get("impacted_artifacts", []),
            },
        )
        assert update_response.status_code == 200


def complete_agent_design_flow(client: TestClient, headers: dict[str, str], session_id: str) -> None:
    approve_design_for_session(client, headers, session_id)
    approve_tools_for_pi18(client, headers, session_id)
    approve_memory_for_session(client, headers, session_id)
    approve_validate_for_session(client, headers, session_id)
    bootstrap_evaluation_response = client.post(
        f"/api/v1/sessions/{session_id}/evaluation/bootstrap",
        headers=headers,
    )
    assert bootstrap_evaluation_response.status_code == 200, bootstrap_evaluation_response.text
    evaluation_response = client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers)
    assert evaluation_response.status_code == 200, evaluation_response.text
    estimate_response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert estimate_response.status_code == 200


def approve_tools_for_pi18(client: TestClient, headers: dict[str, str], session_id: str) -> None:
    recommend_response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)
    assert recommend_response.status_code == 200, recommend_response.text
    recommendation_payload = recommend_response.json()
    optional_keys = [item["tool_key"] for item in recommendation_payload["data"]["optional_tools"]]
    preferred_optional = [
        item for item in optional_keys if item in {"document_ingestion", "knowledge_retrieval", "audit_trail"}
    ]

    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/approve-tools-selection",
        headers=headers,
        json={"include_optional_tool_keys": preferred_optional},
    )
    assert approve_response.status_code == 200, approve_response.text


def build_blueprint_flow_for_pi18(client: TestClient, headers: dict[str, str]) -> str:
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    discovery_payload = complete_discovery_payload()
    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=discovery_payload,
    )
    assert normalize_response.status_code == 200
    assert normalize_response.json()["status"] == "ready"

    analyze_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=discovery_payload,
    )
    assert analyze_response.status_code == 200
    discover_artifact = analyze_response.json()

    approve_discover_response = client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{discover_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Discover aprobado en PI18.",
            "decision_payload": {
                "approval_reason": "Discovery trazable listo para Define.",
            },
        },
    )
    assert approve_discover_response.status_code == 200, approve_discover_response.text

    define_response = client.post(f"/api/v1/sessions/{session_id}/define-requirements", headers=headers)
    assert define_response.status_code == 200
    define_artifact = define_response.json()
    definition_payload = define_artifact["proposal_payload"]
    if definition_payload.get("validation", {}).get("blocking_issues"):
        definition_payload = close_definition_review_gaps(definition_payload)
        patch_define_response = client.patch(
            f"/api/v1/sessions/{session_id}/journey/define/artifacts/{define_artifact['id']}",
            headers=headers,
            json={
                "proposal_payload": definition_payload,
                "missing_information": [],
                "warnings": [
                    "Blocking issues de Define resueltos en PI18 antes de aprobar.",
                ],
                "note": "PI18 cierra blockers de Define para validar el flujo completo.",
            },
        )
        assert patch_define_response.status_code == 200, patch_define_response.text
        define_artifact = patch_define_response.json()

    approve_define_response = client.post(
        f"/api/v1/sessions/{session_id}/journey/define/artifacts/{define_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Define aprobado en PI18.",
            "decision_payload": {
                "approval_reason": "Definition listo para construir Blueprint.",
            },
        },
    )
    assert approve_define_response.status_code == 200, approve_define_response.text

    blueprint_response = client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers)
    assert blueprint_response.status_code == 200
    return session_id


def assert_zip_is_portable_and_launcher_guides_without_lean(zip_path: Path, extract_root: Path) -> None:
    assert validate_acp_zip(zip_path, "acp-portable") == []
    with ZipFile(zip_path) as archive:
        archive.extractall(extract_root)

    launcher_path = extract_root / "ACP" / "launcher" / "acp-launcher.py"
    result = subprocess.run(
        [sys.executable, str(launcher_path), "--dry-run", "--no-open"],
        cwd=extract_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((extract_root / "ACP" / "launcher" / "launch-report.json").read_text(encoding="utf-8"))
    assert report["safety"]["requires_lean_backend"] is False
    assert report["safety"]["runs_build"] is False
    assert report["safety"]["runs_destructive_commands"] is False
    assert report["package_state"]["missing_files"] == []
    assert report["next_steps"]


def test_pi18_commercial_technical_and_portability_e2e(client: TestClient, tmp_path: Path) -> None:
    headers = auth_headers(client)
    session_id = build_blueprint_flow_for_pi18(client, headers)

    free_snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert free_snapshot.status_code == 200
    assert free_snapshot.json()["commercial_access"]["tier"] == "blueprint"

    diagram_catalog = client.get(f"/api/v1/sessions/{session_id}/diagrams/catalog", headers=headers)
    assert diagram_catalog.status_code == 200
    diagram_payload = diagram_catalog.json()
    assert diagram_payload["total_count"] >= 24
    assert diagram_payload["sample_count"] >= 1
    assert diagram_payload["locked_count"] >= 1
    assert all("content" not in item for item in diagram_payload["entries"])

    assert client.get(f"/api/v1/sessions/{session_id}/export/markdown", headers=headers).status_code == 403
    assert client.get(f"/api/v1/sessions/{session_id}/export/json", headers=headers).status_code == 403
    assert client.get(f"/api/v1/sessions/{session_id}/acp/preview", headers=headers).status_code == 403
    assert client.get(f"/api/v1/sessions/{session_id}/acp/export.zip", headers=headers).status_code == 403

    blocked_event = client.post(
        f"/api/v1/sessions/{session_id}/commercial-events",
        headers=headers,
        json={
            "event_key": "blueprint_results_viewed",
            "metadata": {"source": "pi18_e2e"},
            "product": "blueprint",
            "source": "pi18_contract_test",
        },
    )
    assert blocked_event.status_code == 200

    upgrade_session_tier(client, headers, session_id, tier="blueprint_pro")
    blueprint_export = client.get(f"/api/v1/sessions/{session_id}/export/markdown", headers=headers)
    assert blueprint_export.status_code == 200
    assert "Blueprint" in blueprint_export.text

    blueprint_diagrams = client.get(f"/api/v1/sessions/{session_id}/diagrams/catalog", headers=headers).json()
    assert blueprint_diagrams["unlocked_count"] >= diagram_payload["sample_count"]
    assert client.get(f"/api/v1/sessions/{session_id}/acp/preview", headers=headers).status_code == 403

    complete_agent_design_flow(client, headers, session_id)
    upgrade_session_tier(client, headers, session_id, tier="acp")

    first_preview = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    assert first_preview.status_code == 200
    answer_acp_questions(client, headers, session_id)
    final_preview = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    assert final_preview.status_code == 200
    preview_payload = final_preview.json()
    assert preview_payload["validation"]["completeness_percent"] >= 90

    validation = client.get(f"/api/v1/sessions/{session_id}/acp/validate", headers=headers)
    assert validation.status_code == 200
    assert validation.json()["completeness_percent"] >= 90
    portable_readiness = client.get(
        f"/api/v1/sessions/{session_id}/acp/construction-readiness?profile=acp-portable",
        headers=headers,
    )
    assert portable_readiness.status_code == 200
    portable_readiness_payload = portable_readiness.json()
    assert portable_readiness_payload["can_start_build"] is True, json.dumps(
        portable_readiness_payload,
        ensure_ascii=False,
        indent=2,
    )[:2000]

    test_pack = client.get(f"/api/v1/sessions/{session_id}/export/test-pack", headers=headers)
    assert test_pack.status_code in {200, 409}
    if test_pack.status_code == 200:
        assert test_pack.json()["schema_version"] == "test-pack.v1"

    zip_response = client.get(f"/api/v1/sessions/{session_id}/acp/export.zip?profile=acp-portable", headers=headers)
    assert zip_response.status_code == 200, zip_response.text[:500]
    assert zip_response.headers["content-type"].startswith("application/zip")
    assert zip_response.headers["x-acp-export-checksum-sha256"]
    zip_path = tmp_path / "pi18-acp-portable.zip"
    zip_path.write_bytes(zip_response.content)
    assert_zip_is_portable_and_launcher_guides_without_lean(zip_path, tmp_path / "portable-consumer")

    audit_report = client.get(f"/api/v1/sessions/{session_id}/commercial-audit", headers=headers)
    assert audit_report.status_code == 200
    audit_payload = audit_report.json()
    assert audit_payload["current_tier"] == "acp"
    assert any(item["key"] == "exports" and item["value"] >= 2 for item in audit_payload["metrics"])
    assert any(item["product"] == "acp" and item["exports"] >= 1 for item in audit_payload["product_summary"])

    seed_user(
        client,
        email="pi18-outsider@leanbuilder.local",
        password="Outsider123!",
        full_name="PI18 Outsider",
    )
    outsider_headers = auth_headers_for_credentials(
        client,
        email="pi18-outsider@leanbuilder.local",
        password="Outsider123!",
    )
    for path in (
        f"/api/v1/sessions/{session_id}",
        f"/api/v1/sessions/{session_id}/diagrams/catalog",
        f"/api/v1/sessions/{session_id}/commercial-audit",
        f"/api/v1/sessions/{session_id}/acp/export.zip?profile=acp-portable",
    ):
        response = client.get(path, headers=outsider_headers)
        assert response.status_code == 404
