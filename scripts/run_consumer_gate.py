from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest

from app.contracts import CANONICAL_CONTRACT_MODELS
from app.services.canonical_exports import build_contract_bundle
from tests.api_testkit import build_test_client
from tests.canonical_fixture_builder import FIXTURE_CASES, build_full_session_snapshot


REPO_ROOT = BACKEND_ROOT.parent
OUTPUT_ROOT = BACKEND_ROOT / "runtime" / "stage9-consumer-gate"
REFERENCE_CONSUMER = REPO_ROOT / "shared_specs" / "reference_consumers" / "python" / "reference_consumer.py"
CONTRACT_KEYS = (
    "blueprint-core.v1",
    "construction-pack.v1",
    "prompt-pack.v1",
    "estimation-pack.v1",
    "evaluation-pack.v1",
    "test-pack.v1",
)


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def checksum(payload: Any) -> str:
    return __import__("hashlib").sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def remove_nested_field(payload: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    current = payload
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            return
        current = next_value
    current.pop(parts[-1], None)


def invalid_fixture_relative_path(contract_key: str, dotted_path: str) -> str:
    normalized_path = dotted_path.replace(".", "_").replace("-", "_")
    return f"contracts/invalid/{contract_key}.{normalized_path}.missing.json"


def materialize_clean_pack(root: Path, bundle: dict[str, Any]) -> Path:
    contracts_dir = root / "contracts"
    tests_dir = root / "tests"
    for contract_key in CONTRACT_KEYS:
        write_json(contracts_dir / f"{contract_key}.json", bundle[contract_key].model_dump(mode="json"))

    test_pack_payload = bundle["test-pack.v1"].model_dump(mode="json")
    valid_payloads = {
        contract_key: bundle[contract_key].model_dump(mode="json")
        for contract_key in CONTRACT_KEYS
    }
    for case in test_pack_payload["mutation_cases"]:
        mutated_payload = json.loads(json.dumps(valid_payloads[case["contract_key"]]))
        remove_nested_field(mutated_payload, case["path"])
        write_json(root / invalid_fixture_relative_path(case["contract_key"], case["path"]), mutated_payload)

    write_json(
        tests_dir / "acceptance-cases.json",
        bundle["evaluation-pack.v1"].model_dump(mode="json")["acceptance_cases"],
    )
    write_json(tests_dir / "mutation-cases.json", test_pack_payload["mutation_cases"])
    write_json(tests_dir / "prompt-evaluation.json", test_pack_payload["prompt_evaluation_cases"])

    consumer_target = root / test_pack_payload["external_consumer"]["relative_path"]
    consumer_target.parent.mkdir(parents=True, exist_ok=True)
    consumer_target.write_text(REFERENCE_CONSUMER.read_text(encoding="utf-8"), encoding="utf-8")
    return consumer_target


def validate_bundle(bundle: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for contract_key in CONTRACT_KEYS:
        payload = bundle[contract_key].model_dump(mode="json")
        CANONICAL_CONTRACT_MODELS[contract_key].model_validate(payload)
        result[contract_key] = checksum(payload)
    return result


def run_reference_consumer(clean_root: Path, consumer_script: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(consumer_script),
            "--pack",
            "contracts/test-pack.v1.json",
            "--contracts",
            "contracts",
            "--mode",
            "full",
        ],
        capture_output=True,
        check=False,
        cwd=clean_root,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)


def collect_bundles() -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            for case in FIXTURE_CASES:
                snapshot = build_full_session_snapshot(client, case["key"], case["title"])
                bundles[case["key"]] = build_contract_bundle(snapshot)
    return bundles


def main() -> int:
    run_dir = OUTPUT_ROOT / now_slug()
    run_dir.mkdir(parents=True, exist_ok=True)

    case_summaries: list[dict[str, Any]] = []
    bundles = collect_bundles()
    for case in FIXTURE_CASES:
        case_key = case["key"]
        bundle = bundles[case_key]
        case_root = run_dir / case_key
        consumer_script = materialize_clean_pack(case_root, bundle)
        checksums = validate_bundle(bundle)
        consumer_summary = run_reference_consumer(case_root, consumer_script)

        construction_payload = bundle["construction-pack.v1"].model_dump(mode="json")
        prompt_payload = bundle["prompt-pack.v1"].model_dump(mode="json")
        snapshot_free = "SessionSnapshot" not in json.dumps(construction_payload, ensure_ascii=True)
        snapshot_free = snapshot_free and "SessionSnapshot" not in json.dumps(prompt_payload, ensure_ascii=True)

        summary = {
            "case_key": case_key,
            "title": case["title"],
            "checksums": checksums,
            "consumer_summary": consumer_summary,
            "external_consumer_script": str(consumer_script.relative_to(case_root)),
            "snapshot_free": snapshot_free,
        }
        write_json(case_root / "summary.json", summary)
        case_summaries.append(summary)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture_count": len(case_summaries),
        "status": "passed",
        "cases": case_summaries,
    }
    write_json(run_dir / "summary.json", payload)
    write_json(OUTPUT_ROOT / "latest" / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
