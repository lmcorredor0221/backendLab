from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


STAGE0_ROOT = REPO_ROOT / "Docs" / "reingenieria-plataforma-2026-07-15" / "stage-0"
FIXTURES_ROOT = STAGE0_ROOT / "fixtures"
GOLDEN_ROOT = STAGE0_ROOT / "golden"

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def alias_uuid(raw: str, registry: dict[str, str]) -> str:
    if raw not in registry:
        registry[raw] = f"<uuid-{len(registry) + 1}>"
    return registry[raw]


def sanitize_value(value: Any, registry: dict[str, str], parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item, registry, key) for key, item in value.items()}

    if isinstance(value, list):
        return [sanitize_value(item, registry, parent_key) for item in value]

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


def build_preview_index(preview: dict[str, Any], validation: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    files = []
    for item in preview.get("files", []):
        files.append(
            {
                "path": item.get("path", ""),
                "status": item.get("status", ""),
                "summary": item.get("summary", ""),
            }
        )

    files.sort(key=lambda entry: entry["path"])

    return {
        "manifest_path": preview.get("manifest_path", ""),
        "file_count": len(files),
        "files": files,
        "validation": {
            "can_export_zip": validation.get("can_export_zip", False),
            "completeness_percent": validation.get("completeness_percent", 0),
            "error_count": validation.get("error_count", 0),
            "warning_count": validation.get("warning_count", 0),
        },
        "construction_readiness": {
            "overall_status": readiness.get("overall_status", ""),
            "can_start_build": readiness.get("can_start_build", False),
            "blocking_gaps": readiness.get("blocking_gaps", 0),
            "open_questions": readiness.get("open_questions", 0),
        },
    }


def answer_questions(client: TestClient, headers: dict[str, str], session_id: str) -> None:
    response = client.get(f"/api/v1/sessions/{session_id}/acp/questions", headers=headers)
    response.raise_for_status()
    questions = response.json()

    for question in questions:
        question_key = str(question["question_key"])
        answer_text = ANSWER_MAP.get(
            question_key,
            f"resolved={question_key}; source=stage0_baseline; owner=platform_owner",
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


def build_case(client: TestClient, case_key: str, case_title: str) -> dict[str, Any]:
    discovery_path = FIXTURES_ROOT / f"{case_key}.discovery.json"
    patch_path = FIXTURES_ROOT / f"{case_key}.blueprint-patch.json"

    headers = auth_headers(client)

    create_response = client.post("/api/v1/sessions", headers=headers)
    create_response.raise_for_status()
    session_id = create_response.json()["id"]

    discovery_payload = load_json(discovery_path)
    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=discovery_payload,
    )
    normalize_response.raise_for_status()

    canvas_response = client.post(f"/api/v1/sessions/{session_id}/build-canvas", headers=headers)
    canvas_response.raise_for_status()

    blueprint_response = client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers)
    blueprint_response.raise_for_status()

    if patch_path.exists():
        patch_payload = load_json(patch_path)
        patch_response = client.patch(
            f"/api/v1/sessions/{session_id}/blueprint",
            headers=headers,
            json=patch_payload,
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

    evaluate_response = client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers)
    evaluate_response.raise_for_status()

    preview_response = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    preview_response.raise_for_status()
    preview = preview_response.json()

    if preview.get("construction_readiness", {}).get("open_questions", 0) > 0:
        answer_questions(client, headers, session_id)
        preview_response = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
        preview_response.raise_for_status()
        preview = preview_response.json()

    validation_response = client.get(f"/api/v1/sessions/{session_id}/acp/validate", headers=headers)
    validation_response.raise_for_status()
    validation = validation_response.json()

    readiness_response = client.get(f"/api/v1/sessions/{session_id}/acp/construction-readiness", headers=headers)
    readiness_response.raise_for_status()
    readiness = readiness_response.json()

    manifest_response = client.get(f"/api/v1/sessions/{session_id}/acp/files/ACP/manifest.yaml", headers=headers)
    manifest_response.raise_for_status()
    manifest_text = manifest_response.json()["content_text"]

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    snapshot_response.raise_for_status()
    snapshot = snapshot_response.json()

    export_response = client.get(f"/api/v1/sessions/{session_id}/export/json", headers=headers)
    export_response.raise_for_status()
    export_payload = export_response.json()

    if validation.get("can_export_zip") is not True:
        raise RuntimeError(f"{case_key} no quedo exportable tras responder preguntas.")

    uuid_registry: dict[str, str] = {}
    sanitized_snapshot = sanitize_value(snapshot, uuid_registry)
    sanitized_export = sanitize_value(export_payload, uuid_registry)
    sanitized_preview_index = sanitize_value(build_preview_index(preview, validation, readiness), uuid_registry)

    return {
        "case_key": case_key,
        "session_snapshot": sanitized_snapshot,
        "export_json": sanitized_export,
        "acp_preview_index": sanitized_preview_index,
        "acp_manifest": manifest_text,
    }


def compare_or_write_json(path: Path, payload: Any, mode: str) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, indent=2)
    if mode == "verify":
        expected = path.read_text(encoding="utf-8")
        if expected != serialized:
            raise AssertionError(f"Golden JSON desactualizado: {path}")
        return
    write_text(path, serialized)


def compare_or_write_text(path: Path, content: str, mode: str) -> None:
    if mode == "verify":
        expected = path.read_text(encoding="utf-8")
        if expected != content:
            raise AssertionError(f"Golden text desactualizado: {path}")
        return
    write_text(path, content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["refresh", "verify"], default="refresh")
    args = parser.parse_args()

    stage0_manifest: list[dict[str, Any]] = []

    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            for case in FIXTURE_CASES:
                payload = build_case(client, case["key"], case["title"])
                case_root = GOLDEN_ROOT / case["key"]
                compare_or_write_json(case_root / "session-snapshot.v1.json", payload["session_snapshot"], args.mode)
                compare_or_write_json(case_root / "export.json", payload["export_json"], args.mode)
                compare_or_write_json(case_root / "acp-preview-index.json", payload["acp_preview_index"], args.mode)
                compare_or_write_text(case_root / "acp-manifest.yaml", payload["acp_manifest"], args.mode)

                stage0_manifest.append(
                    {
                        "case_key": case["key"],
                        "files": [
                            f"golden/{case['key']}/session-snapshot.v1.json",
                            f"golden/{case['key']}/export.json",
                            f"golden/{case['key']}/acp-preview-index.json",
                            f"golden/{case['key']}/acp-manifest.yaml",
                        ],
                    }
                )

    compare_or_write_json(GOLDEN_ROOT / "fixture-manifest.json", stage0_manifest, args.mode)
    print(json.dumps(stage0_manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
