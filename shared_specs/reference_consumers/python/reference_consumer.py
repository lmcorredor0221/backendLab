from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROMPT_FIELDS = (
    "system_prompt",
    "planner_prompt",
    "executor_prompt",
    "evaluator_prompt",
    "tool_use_prompt",
    "memory_prompt",
    "retrieval_prompt",
    "recovery_prompt",
)
CRITICAL_PROMPTS = ("system", "planner", "executor", "evaluator", "recovery")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def get_nested_value(payload: Any, dotted_path: str) -> tuple[bool, Any]:
    current = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def ensure(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def invalid_fixture_relative_path(contract_key: str, dotted_path: str) -> str:
    normalized_path = dotted_path.replace(".", "_").replace("-", "_")
    return f"contracts/invalid/{contract_key}.{normalized_path}.missing.json"


def prompt_artifact_map(prompt_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for field_name in PROMPT_FIELDS:
        artifact = prompt_pack.get(field_name)
        if isinstance(artifact, dict):
            prompt_key = str(artifact.get("prompt_key") or field_name.removesuffix("_prompt"))
            artifacts[prompt_key] = artifact
    return artifacts


def required_fields_by_contract(test_pack: dict[str, Any]) -> dict[str, set[str]]:
    fields: dict[str, set[str]] = {}
    for item in test_pack.get("mutation_cases", []):
        contract_key = str(item.get("contract_key", ""))
        field_path = str(item.get("path", ""))
        if not contract_key or not field_path:
            continue
        fields.setdefault(contract_key, set()).add(field_path)
    return fields


def fixture_map(test_pack: dict[str, Any], *, valid: bool) -> dict[str, dict[str, Any]]:
    fixtures = test_pack.get("fixtures" if valid else "invalid_fixtures", [])
    result: dict[str, dict[str, Any]] = {}
    for item in fixtures:
        contract_key = str(item.get("contract_key", ""))
        relative_path = str(item.get("relative_path", ""))
        if contract_key and relative_path:
            result[f"{contract_key}::{relative_path}"] = item
    return result


def first_valid_fixture_for_contract(test_pack: dict[str, Any], contract_key: str) -> dict[str, Any] | None:
    for item in test_pack.get("fixtures", []):
        if item.get("contract_key") == contract_key and item.get("valid", True):
            return item
    return None


def run_schema_mode(test_pack: dict[str, Any], contracts_dir: Path) -> list[str]:
    failures: list[str] = []
    required_pack_fields = (
        "schema_version",
        "framework_target",
        "fixtures",
        "invalid_fixtures",
        "commands",
        "mutation_cases",
        "prompt_evaluation_cases",
        "recovery_cases",
        "acceptance_journeys",
        "stable_issue_catalog",
        "external_consumer",
    )
    for field_name in required_pack_fields:
        ensure(field_name in test_pack, f"test-pack missing required field `{field_name}`", failures)

    ensure(test_pack.get("schema_version") == "test-pack.v1", "test-pack schema_version must be `test-pack.v1`", failures)
    ensure(bool(str(test_pack.get("framework_target", "")).strip()), "framework_target cannot be empty", failures)

    command_kinds = {str(item.get("kind", "")) for item in test_pack.get("commands", [])}
    ensure("schema_validation" in command_kinds, "test-pack missing schema_validation command", failures)
    ensure("mutation" in command_kinds, "test-pack missing mutation command", failures)
    ensure("prompt_recovery" in command_kinds, "test-pack missing prompt_recovery command", failures)
    ensure("external_consumer" in command_kinds, "test-pack missing external_consumer command", failures)

    consumer = test_pack.get("external_consumer", {})
    consumer_relative_path = str(consumer.get("relative_path", ""))
    consumer_path = resolve_path(consumer_relative_path) if consumer_relative_path else None
    ensure(bool(consumer_relative_path), "external_consumer.relative_path cannot be empty", failures)
    ensure(bool(str(consumer.get("entry_command", "")).strip()), "external_consumer.entry_command cannot be empty", failures)
    ensure(consumer_path is not None and consumer_path.exists(), "external consumer script must exist in the clean directory", failures)
    if consumer_path is not None and consumer_path.exists():
        consumer_text = consumer_path.read_text(encoding="utf-8")
        builder_module_name = "".join(("a", "pp"))
        ensure(
            f"from {builder_module_name}" not in consumer_text,
            "external consumer must not import `app` modules",
            failures,
        )
        ensure(
            f"import {builder_module_name}" not in consumer_text,
            "external consumer must not import builder runtime",
            failures,
        )

    invalid_fixtures = fixture_map(test_pack, valid=False)
    ensure(bool(invalid_fixtures), "test-pack must include invalid fixtures", failures)

    required_fields = required_fields_by_contract(test_pack)
    for fixture in test_pack.get("fixtures", []):
        relative_path = str(fixture.get("relative_path", ""))
        contract_key = str(fixture.get("contract_key", ""))
        fixture_path = resolve_path(relative_path)
        ensure(fixture_path.exists(), f"missing fixture file `{relative_path}`", failures)
        if not fixture_path.exists():
            continue
        payload = load_json(fixture_path)
        ensure(payload.get("schema_version") == contract_key, f"{relative_path} has mismatched schema_version", failures)
        for field_name in sorted(required_fields.get(contract_key, set())):
            exists, _ = get_nested_value(payload, field_name)
            ensure(exists, f"{relative_path} missing required field `{field_name}`", failures)

    for issue in test_pack.get("stable_issue_catalog", []):
        ensure(bool(str(issue.get("code", "")).strip()), "stable_issue_catalog entry missing code", failures)
        ensure(bool(str(issue.get("kind", "")).strip()), "stable_issue_catalog entry missing kind", failures)
        ensure(bool(str(issue.get("severity", "")).strip()), "stable_issue_catalog entry missing severity", failures)
        ensure(bool(str(issue.get("remediation", "")).strip()), "stable_issue_catalog entry missing remediation", failures)

    for journey in test_pack.get("acceptance_journeys", []):
        ensure(bool(str(journey.get("key", "")).strip()), "acceptance_journey missing key", failures)
        ensure(bool(str(journey.get("input_reference", "")).strip()), "acceptance_journey missing input_reference", failures)
        ensure(bool(str(journey.get("expected_behavior", "")).strip()), "acceptance_journey missing expected_behavior", failures)
        ensure(bool(str(journey.get("measurable_criterion", "")).strip()), "acceptance_journey missing measurable_criterion", failures)

    return failures


def run_mutation_mode(test_pack: dict[str, Any], contracts_dir: Path) -> list[str]:
    failures: list[str] = []
    for case in test_pack.get("mutation_cases", []):
        contract_key = str(case.get("contract_key", ""))
        field_path = str(case.get("path", ""))
        case_key = str(case.get("key", ""))
        valid_fixture = first_valid_fixture_for_contract(test_pack, contract_key)
        ensure(valid_fixture is not None, f"{case_key}: missing valid fixture for `{contract_key}`", failures)
        if valid_fixture is None:
            continue

        valid_path = resolve_path(str(valid_fixture.get("relative_path", "")))
        invalid_path = resolve_path(invalid_fixture_relative_path(contract_key, field_path))
        ensure(valid_path.exists(), f"{case_key}: valid fixture `{valid_path}` does not exist", failures)
        ensure(invalid_path.exists(), f"{case_key}: invalid fixture `{invalid_path}` does not exist", failures)
        if not valid_path.exists() or not invalid_path.exists():
            continue

        valid_payload = load_json(valid_path)
        invalid_payload = load_json(invalid_path)

        valid_exists, _ = get_nested_value(valid_payload, field_path)
        invalid_exists, _ = get_nested_value(invalid_payload, field_path)
        ensure(valid_exists, f"{case_key}: valid fixture must keep `{field_path}`", failures)
        ensure(not invalid_exists, f"{case_key}: invalid fixture must remove `{field_path}`", failures)
        ensure(bool(str(case.get("expected_issue_code", "")).strip()), f"{case_key}: expected_issue_code cannot be empty", failures)
        ensure(bool(str(case.get("expected_issue_path", "")).strip()), f"{case_key}: expected_issue_path cannot be empty", failures)
    return failures


def run_prompt_mode(test_pack: dict[str, Any], contracts_dir: Path) -> list[str]:
    failures: list[str] = []
    prompt_pack = load_json(contracts_dir / "prompt-pack.v1.json")
    construction_pack = load_json(contracts_dir / "construction-pack.v1.json")
    evaluation_pack = load_json(contracts_dir / "evaluation-pack.v1.json")
    artifacts = prompt_artifact_map(prompt_pack)

    cases_by_prompt: dict[str, dict[str, int]] = {}
    for case in test_pack.get("prompt_evaluation_cases", []):
        prompt_key = str(case.get("prompt_key", ""))
        mode = str(case.get("mode", ""))
        artifact = artifacts.get(prompt_key)
        ensure(artifact is not None, f"prompt case `{case.get('key', '')}` references unknown prompt `{prompt_key}`", failures)
        if artifact is None:
            continue

        content = str(artifact.get("content", ""))
        for expected_substring in case.get("expected_substrings", []):
            ensure(expected_substring in content, f"prompt `{prompt_key}` lost expected signal `{expected_substring}`", failures)
        normalized_content = content.lower()
        for forbidden_substring in case.get("forbidden_substrings", []):
            ensure(
                str(forbidden_substring).lower() not in normalized_content,
                f"prompt `{prompt_key}` contains forbidden fragment `{forbidden_substring}`",
                failures,
            )
        cases_by_prompt.setdefault(prompt_key, {}).setdefault(mode, 0)
        cases_by_prompt[prompt_key][mode] += 1

    for prompt_key in CRITICAL_PROMPTS:
        if prompt_key not in artifacts:
            continue
        ensure(cases_by_prompt.get(prompt_key, {}).get("positive", 0) >= 1, f"critical prompt `{prompt_key}` lacks a positive case", failures)
        ensure(cases_by_prompt.get(prompt_key, {}).get("failure", 0) >= 1, f"critical prompt `{prompt_key}` lacks a failure case", failures)

    recovery_cases = {str(item.get("trigger", "")): item for item in test_pack.get("recovery_cases", [])}
    ensure("tool_failure" in recovery_cases, "missing recovery case for tool_failure", failures)
    ensure("llm_timeout" in recovery_cases, "missing recovery case for llm_timeout", failures)

    knowledge_contract = construction_pack.get("knowledge_contract", {})
    llm_policy = construction_pack.get("llm_policy", {})
    knowledge_mode = str(knowledge_contract.get("mode", ""))
    retrieval_artifact = artifacts.get("retrieval")
    if retrieval_artifact is not None or knowledge_mode == "rag":
        ensure("retrieval_no_evidence" in recovery_cases, "missing recovery case for retrieval_no_evidence", failures)
        ensure(
            bool(str(knowledge_contract.get("grounding_policy", {}).get("no_evidence_behavior", "")).strip()),
            "knowledge_contract must define no_evidence_behavior",
            failures,
        )

    for trigger, case in recovery_cases.items():
        expected_prompt_key = str(case.get("expected_prompt_key", ""))
        ensure(expected_prompt_key in artifacts, f"recovery case `{trigger}` references missing prompt `{expected_prompt_key}`", failures)
        ensure(bool(str(case.get("measurable_criterion", "")).strip()), f"recovery case `{trigger}` missing measurable_criterion", failures)
        if trigger == "llm_timeout":
            ensure(bool(str(llm_policy.get("fallback_model", "")).strip()), "llm_timeout recovery requires llm_policy.fallback_model", failures)

    evaluation_case_keys = {str(item.get("key", "")) for item in evaluation_pack.get("acceptance_cases", [])}
    journey_keys = set()
    for journey in test_pack.get("acceptance_journeys", []):
        journey_key = str(journey.get("key", ""))
        journey_keys.add(journey_key)
        ensure(journey_key in evaluation_case_keys, f"acceptance journey `{journey_key}` is missing from evaluation-pack", failures)
        ensure(bool(str(journey.get("expected_behavior", "")).strip()), f"acceptance journey `{journey_key}` missing expected_behavior", failures)
        ensure(bool(str(journey.get("measurable_criterion", "")).strip()), f"acceptance journey `{journey_key}` missing measurable_criterion", failures)

    ensure(journey_keys == evaluation_case_keys, "acceptance journeys must cover all evaluation acceptance cases", failures)
    return failures


def build_summary(mode: str, failures: list[str]) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": "passed" if not failures else "failed",
        "failure_count": len(failures),
        "failures": failures,
    }


def run_mode(mode: str, test_pack: dict[str, Any], contracts_dir: Path) -> list[str]:
    if mode == "schema":
        return run_schema_mode(test_pack, contracts_dir)
    if mode == "mutations":
        return run_mutation_mode(test_pack, contracts_dir)
    if mode == "prompts":
        return run_prompt_mode(test_pack, contracts_dir)

    failures: list[str] = []
    failures.extend(run_schema_mode(test_pack, contracts_dir))
    failures.extend(run_mutation_mode(test_pack, contracts_dir))
    failures.extend(run_prompt_mode(test_pack, contracts_dir))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True)
    parser.add_argument("--contracts", required=True)
    parser.add_argument("--mode", choices=("schema", "mutations", "prompts", "full"), default="full")
    args = parser.parse_args(argv)

    pack_path = resolve_path(args.pack)
    contracts_dir = resolve_path(args.contracts)
    if not pack_path.exists():
        sys.stdout.write(json.dumps(build_summary(args.mode, [f"pack file not found: {pack_path}"]), ensure_ascii=True, indent=2) + "\n")
        return 1
    if not contracts_dir.exists():
        sys.stdout.write(json.dumps(build_summary(args.mode, [f"contracts directory not found: {contracts_dir}"]), ensure_ascii=True, indent=2) + "\n")
        return 1

    test_pack = load_json(pack_path)
    failures = run_mode(args.mode, test_pack, contracts_dir)
    sys.stdout.write(json.dumps(build_summary(args.mode, failures), ensure_ascii=True, indent=2) + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
