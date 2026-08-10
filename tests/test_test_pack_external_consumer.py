from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from app.services.canonical_exports import build_contract_bundle
from tests.api_testkit import build_test_client
from tests.canonical_fixture_builder import FIXTURE_CASES, build_full_session_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONSUMER = REPO_ROOT / "shared_specs" / "reference_consumers" / "python" / "reference_consumer.py"
CONTRACT_KEYS = (
    "blueprint-core.v1",
    "construction-pack.v1",
    "prompt-pack.v1",
    "estimation-pack.v1",
    "evaluation-pack.v1",
    "test-pack.v1",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


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


@pytest.fixture(scope="module")
def canonical_bundles() -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        with build_test_client(monkeypatch) as client:
            for case in FIXTURE_CASES:
                snapshot = build_full_session_snapshot(client, case["key"], case["title"])
                bundles[case["key"]] = build_contract_bundle(snapshot)
    return bundles


def test_reference_consumer_script_is_builder_agnostic() -> None:
    content = REFERENCE_CONSUMER.read_text(encoding="utf-8")
    assert "from app" not in content
    assert "import app" not in content
    assert "SessionSnapshot" not in content


def test_reference_consumer_runs_full_suite_from_clean_directory(
    tmp_path: Path,
    canonical_bundles: dict[str, dict[str, Any]],
) -> None:
    for case_key, bundle in canonical_bundles.items():
        clean_root = tmp_path / case_key
        consumer_script = materialize_clean_pack(clean_root, bundle)
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

        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads(result.stdout)
        assert summary["status"] == "passed"
        assert summary["mode"] == "full"


def test_reference_consumer_blocks_when_a_mutation_fixture_stops_failing(
    tmp_path: Path,
    canonical_bundles: dict[str, dict[str, Any]],
) -> None:
    bundle = canonical_bundles["01-copilot-simple"]
    clean_root = tmp_path / "broken-mutation"
    consumer_script = materialize_clean_pack(clean_root, bundle)
    test_pack_payload = bundle["test-pack.v1"].model_dump(mode="json")
    first_case = test_pack_payload["mutation_cases"][0]
    valid_payload = bundle[first_case["contract_key"]].model_dump(mode="json")
    broken_invalid_fixture = clean_root / invalid_fixture_relative_path(first_case["contract_key"], first_case["path"])
    write_json(broken_invalid_fixture, valid_payload)

    result = subprocess.run(
        [
            sys.executable,
            str(consumer_script),
            "--pack",
            "contracts/test-pack.v1.json",
            "--contracts",
            "contracts",
            "--mode",
            "mutations",
        ],
        capture_output=True,
        check=False,
        cwd=clean_root,
        text=True,
    )

    assert result.returncode != 0
    summary = json.loads(result.stdout)
    assert summary["status"] == "failed"
    assert any(first_case["key"] in item for item in summary["failures"])
