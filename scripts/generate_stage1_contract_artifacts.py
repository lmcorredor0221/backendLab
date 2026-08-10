from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.contracts import CANONICAL_CONTRACT_MODELS, CANONICAL_CONTRACT_ORDER, get_schema_file_name
from app.services.canonical_exports import build_contract_bundle
from tests.api_testkit import build_test_client
from tests.canonical_fixture_builder import FIXTURE_CASES, build_full_session_snapshot, sanitize_dynamic_contract_value

STAGE1_ROOT = REPO_ROOT / "Docs" / "reingenieria-plataforma-2026-07-15" / "stage-1"
GOLDEN_ROOT = STAGE1_ROOT / "golden"
SCHEMAS_ROOT = REPO_ROOT / "shared_specs" / "schemas"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def compare_or_write_json(path: Path, payload: Any, mode: str) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    if mode == "verify":
        expected = path.read_text(encoding="utf-8")
        if expected != serialized:
            raise AssertionError(f"Golden JSON desactualizado: {path}")
        return
    write_json(path, payload)


def build_registry_payload() -> dict[str, Any]:
    contracts = []
    for schema_version in CANONICAL_CONTRACT_ORDER:
        model = CANONICAL_CONTRACT_MODELS[schema_version]
        schema = model.model_json_schema()
        contracts.append(
            {
                "schema_version": schema_version,
                "schema_file": get_schema_file_name(schema_version),
                "title": schema.get("title", ""),
                "required": sorted(schema.get("required", [])),
            }
        )

    return {
        "registry_version": "canonical-contract-registry.v1",
        "contracts": contracts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["refresh", "verify"], default="refresh")
    args = parser.parse_args()

    registry_payload = build_registry_payload()
    compare_or_write_json(SCHEMAS_ROOT / "contract-registry.v1.json", registry_payload, args.mode)

    for schema_version in CANONICAL_CONTRACT_ORDER:
        model = CANONICAL_CONTRACT_MODELS[schema_version]
        compare_or_write_json(
            SCHEMAS_ROOT / get_schema_file_name(schema_version),
            model.model_json_schema(),
            args.mode,
        )

    stage_manifest: list[dict[str, Any]] = []
    contract_keys = (
        "blueprint-core.v1",
        "construction-pack.v1",
        "prompt-pack.v1",
        "estimation-pack.v1",
        "evaluation-pack.v1",
        "test-pack.v1",
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            for case in FIXTURE_CASES:
                snapshot = build_full_session_snapshot(client, case["key"], case["title"])
                bundle = build_contract_bundle(snapshot)
                case_root = GOLDEN_ROOT / case["key"]
                files = []
                uuid_registry: dict[str, str] = {}
                for contract_key in contract_keys:
                    compare_or_write_json(
                        case_root / f"{contract_key}.json",
                        sanitize_dynamic_contract_value(bundle[contract_key].model_dump(mode="json"), uuid_registry),
                        args.mode,
                    )
                    files.append(f"golden/{case['key']}/{contract_key}.json")
                stage_manifest.append(
                    {
                        "case_key": case["key"],
                        "files": files,
                    }
                )

    compare_or_write_json(GOLDEN_ROOT / "fixture-manifest.json", stage_manifest, args.mode)
    print(json.dumps(stage_manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
