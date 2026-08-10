from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from fastapi.testclient import TestClient

from app.models import SessionSnapshot

from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE0_FIXTURES_ROOT = REPO_ROOT / "Docs" / "reingenieria-plataforma-2026-07-15" / "stage-0" / "fixtures"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z)?")

ANSWER_MAP = {
    "knowledge_sources": "name=Confluence; type=wiki; owner=ops; frequency=diaria",
    "knowledge_ingestion": "strategy=sync_incremental; frequency=diaria; mechanism=cron; owner=ops",
    "knowledge_embedding_strategy": "provider=text-embedding-3-small; chunking=800_tokens_overlap_120; notes=openai",
    "runtime_fallback_model": "model=gpt-4.1-mini; condition=cuando falle el modelo primario",
    "runtime_vector_store": "vector_store=pgvector; notes=misma base local",
    "runtime_secret_source": "source=.env local protegido; owner=platform_owner; environment=desarrollo",
    "deployment_target": "target=local_vm; restrictions=solo red interna y acceso por VPN",
    "deployment_image_strategy": "strategy=docker_compose_local; registry=no_aplica",
    "deployment_network_constraints": "network=solo red interna; secrets=.env local; dependencies=postgres local",
}

FIXTURE_CASES = [
    {
        "key": "01-copilot-simple",
        "title": "Caso copiloto simple",
    },
    {
        "key": "02-agent-with-tools",
        "title": "Caso agente con tools",
    },
    {
        "key": "03-agent-with-knowledge-rag",
        "title": "Caso agente con knowledge rag",
    },
]


def load_fixture_json(case_key: str, suffix: str) -> dict:
    path = STAGE0_FIXTURES_ROOT / f"{case_key}.{suffix}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def alias_uuid(raw: str, registry: dict[str, str]) -> str:
    registry.setdefault(raw, "<uuid>")
    return registry[raw]


def sanitize_dynamic_contract_value(value: Any, registry: dict[str, str], parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_dynamic_contract_value(item, registry, key) for key, item in value.items()}

    if isinstance(value, list):
        return [sanitize_dynamic_contract_value(item, registry, parent_key) for item in value]

    if isinstance(value, str):
        if parent_key == "id" or (parent_key and parent_key.endswith("_id")):
            return alias_uuid(value, registry) if UUID_RE.fullmatch(value) else value
        if parent_key and parent_key.endswith("_at"):
            return "<timestamp>"
        if UUID_RE.fullmatch(value):
            return alias_uuid(value, registry)
        if TIMESTAMP_RE.fullmatch(value):
            return "<timestamp>"
        value = UUID_RE.sub(lambda match: alias_uuid(match.group(0), registry), value)
        return TIMESTAMP_RE.sub("<timestamp>", value)

    return value


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def answer_questions(client: TestClient, headers: dict[str, str], session_id: str) -> None:
    response = client.get(f"/api/v1/sessions/{session_id}/acp/questions", headers=headers)
    response.raise_for_status()

    for question in response.json():
        question_key = str(question["question_key"])
        answer_text = ANSWER_MAP.get(
            question_key,
            f"resolved={question_key}; source=stage1_contracts; owner=platform_owner",
        )
        patch_response = client.patch(
            f"/api/v1/sessions/{session_id}/acp/questions/{question_key}",
            headers=headers,
            json={
                "answer_text": answer_text,
                "owner_role": "platform_owner",
                "impacted_artifacts": question.get("impacted_artifacts", []),
            },
        )
        patch_response.raise_for_status()


def build_full_session_snapshot(client: TestClient, case_key: str, case_title: str) -> SessionSnapshot:
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    create_response.raise_for_status()
    session_id = create_response.json()["id"]

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=load_fixture_json(case_key, "discovery"),
    )
    normalize_response.raise_for_status()

    client.post(f"/api/v1/sessions/{session_id}/build-canvas", headers=headers).raise_for_status()
    client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers).raise_for_status()

    patch_path = STAGE0_FIXTURES_ROOT / f"{case_key}.blueprint-patch.json"
    if patch_path.exists():
        patch_response = client.patch(
            f"/api/v1/sessions/{session_id}/blueprint",
            headers=headers,
            json=json.loads(patch_path.read_text(encoding="utf-8")),
        )
        patch_response.raise_for_status()

    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    bootstrap_response.raise_for_status()
    bootstrap_snapshot = bootstrap_response.json()

    edited_cases = bootstrap_snapshot["evaluation_dataset"]["cases"]
    if edited_cases:
        edited_cases[0]["title"] = case_title
        edited_cases[0]["source"] = "manual"
        dataset_response = client.patch(
            f"/api/v1/sessions/{session_id}/evaluation/dataset",
            headers=headers,
            json={"cases": edited_cases},
        )
        dataset_response.raise_for_status()

    client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers).raise_for_status()
    client.patch(
        f"/api/v1/sessions/{session_id}/commercial-tier",
        headers=headers,
        json={"tier": "acp"},
    ).raise_for_status()

    preview_response = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    preview_response.raise_for_status()
    preview_payload = preview_response.json()
    if preview_payload.get("construction_readiness", {}).get("open_questions", 0) > 0:
        answer_questions(client, headers, session_id)
        client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers).raise_for_status()

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    snapshot_response.raise_for_status()
    return SessionSnapshot.model_validate(snapshot_response.json())
