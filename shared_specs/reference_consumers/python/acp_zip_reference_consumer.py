from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile


PROFILE_ALIASES = {
    "design-only": "acp-portable",
    "extended": "acp-full",
}
SUPPORTED_PROFILES = {"blueprint-professional", "acp-portable", "acp-full", "design-only", "extended"}
COMMON_REQUIRED_FILES = {
    "ACP/manifest.yaml",
    "ACP/README.md",
    "ACP/conformance/file-index.json",
    "ACP/conformance/checksums.sha256",
    "ACP/conformance/portability-report.json",
    "ACP/conformance/portability-report.md",
    "ACP/conformance/external-consumer-readme.md",
}
PROFILE_RULES = {
    "blueprint-professional": {
        "required": {
            "ACP/architecture/topology.yaml",
            "ACP/cognition/reasoning.yaml",
            "ACP/memory/strategy.yaml",
            "ACP/tools/permissions.yaml",
            "ACP/workflows/state-machine.yaml",
        },
        "forbidden_prefixes": (
            "ACP/launcher/",
            "ACP/adapters/",
            "ACP/construction-readiness/",
            "ACP/prompts/",
            "ACP/runtime/",
            "ACP/deployment/",
            "ACP/observability/",
            "ACP/evaluation/",
        ),
    },
    "acp-portable": {
        "required": {
            "ACP/launcher/acp-launcher.py",
            "ACP/launcher/start-acp.ps1",
            "ACP/launcher/start-acp.bat",
            "ACP/launcher/start-acp.sh",
            "ACP/adapters/adapter-registry.json",
            "ACP/construction-readiness/overview.yaml",
            "ACP/prompts/builder-handoff.md",
            "ACP/runtime/config.yaml",
            "ACP/evaluation/benchmarks.yaml",
        },
        "forbidden_prefixes": (
            "ACP/deployment/",
            "ACP/observability/",
        ),
    },
    "acp-full": {
        "required": {
            "ACP/launcher/acp-launcher.py",
            "ACP/adapters/adapter-registry.json",
            "ACP/construction-readiness/overview.yaml",
            "ACP/prompts/builder-handoff.md",
            "ACP/runtime/config.yaml",
            "ACP/evaluation/benchmarks.yaml",
            "ACP/deployment/env.template",
            "ACP/observability/telemetry.yaml",
        },
        "forbidden_prefixes": (),
    },
}
INTERNAL_RUNTIME_MARKERS = (
    "/api/v1/sessions",
    "SessionSnapshot",
    "journey_stage_artifact_id",
    "skill_run_id",
    "workspace_internal_id",
    "C:/Users/",
    "C:\\Users\\",
)


def _effective_profile(profile: str) -> str:
    normalized = profile.strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported ACP ZIP profile: {profile}")
    return PROFILE_ALIASES.get(normalized, normalized)


def _decode_text(archive: ZipFile, name: str) -> str:
    return archive.read(name).decode("utf-8")


def _sha256_text(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    normalized = "\n".join(lines).strip("\n")
    payload = f"{normalized}\n" if normalized else ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest() if payload else ""


def _validate_file_index(archive: ZipFile, names: set[str], issues: list[str]) -> None:
    try:
        file_index = json.loads(_decode_text(archive, "ACP/conformance/file-index.json"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        issues.append(f"cannot read ACP/conformance/file-index.json: {exc}")
        return
    if file_index.get("schema_version") != "acp-file-index.v1":
        issues.append("file-index schema_version must be acp-file-index.v1")
    entries = file_index.get("files")
    if not isinstance(entries, list) or not entries:
        issues.append("file-index files must not be empty")
        return
    indexed_paths = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            issues.append(f"file-index files[{index}] must be an object")
            continue
        path = str(entry.get("path") or "")
        checksum = str(entry.get("content_hash") or "")
        indexed_paths.add(path)
        if path not in names:
            issues.append(f"file-index references missing file: {path}")
            continue
        if path.startswith("ACP/conformance/"):
            issues.append(f"file-index must not index conformance self-file: {path}")
        if len(checksum) != 64:
            issues.append(f"file-index checksum must be sha256 for {path}")
            continue
        actual_checksum = _sha256_text(_decode_text(archive, path))
        if actual_checksum != checksum:
            issues.append(f"file-index checksum mismatch for {path}")
    non_conformance_names = {name for name in names if not name.startswith("ACP/conformance/")}
    missing_from_index = sorted(non_conformance_names - indexed_paths)
    issues.extend(f"file-index missing package file: {path}" for path in missing_from_index)


def _validate_checksums(archive: ZipFile, names: set[str], issues: list[str]) -> None:
    try:
        checksums_text = _decode_text(archive, "ACP/conformance/checksums.sha256")
    except (KeyError, UnicodeDecodeError) as exc:
        issues.append(f"cannot read ACP/conformance/checksums.sha256: {exc}")
        return
    if not checksums_text.strip():
        issues.append("checksums.sha256 must not be empty")
        return
    for line_number, line in enumerate(checksums_text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            issues.append(f"invalid checksum line {line_number}")
            continue
        expected, path = parts[0], parts[1].strip()
        if path not in names:
            issues.append(f"checksum references missing file: {path}")
            continue
        actual = _sha256_text(_decode_text(archive, path))
        if actual != expected:
            issues.append(f"checksum mismatch for {path}")


def _validate_portability_report(archive: ZipFile, issues: list[str], *, expected_profile: str) -> None:
    try:
        report = json.loads(_decode_text(archive, "ACP/conformance/portability-report.json"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        issues.append(f"cannot read ACP/conformance/portability-report.json: {exc}")
        return
    if report.get("schema_version") != "acp-portability-report.v1":
        issues.append("portability report schema_version must be acp-portability-report.v1")
    if report.get("profile") != expected_profile:
        issues.append(f"portability report profile must be {expected_profile}")
    if report.get("requires_lean_backend") is not False:
        issues.append("portability report must declare requires_lean_backend=false")
    integrity = report.get("reference_integrity") if isinstance(report.get("reference_integrity"), dict) else {}
    if integrity.get("broken_references"):
        issues.append("portability report contains broken references")
    if integrity.get("internal_markers"):
        issues.append("portability report contains internal markers")
    if report.get("ready_for_external_consumer") is not True:
        issues.append("portability report must be ready_for_external_consumer=true")


def _scan_internal_markers(archive: ZipFile, names: set[str], issues: list[str]) -> None:
    for name in sorted(names):
        if name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            continue
        try:
            content = _decode_text(archive, name)
        except UnicodeDecodeError:
            continue
        normalized = content.replace("\\", "/")
        for marker in INTERNAL_RUNTIME_MARKERS:
            if marker.replace("\\", "/") in normalized:
                issues.append(f"internal marker {marker} present in {name}")


def validate_acp_zip(zip_path: Path, profile: str = "acp-full") -> list[str]:
    effective_profile = _effective_profile(profile)
    issues: list[str] = []
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        missing_common = sorted(COMMON_REQUIRED_FILES - names)
        issues.extend(f"missing required file: {path}" for path in missing_common)
        rules = PROFILE_RULES[effective_profile]
        missing_required = sorted(rules["required"] - names)
        issues.extend(f"missing {effective_profile} file: {path}" for path in missing_required)
        for prefix in rules["forbidden_prefixes"]:
            forbidden = sorted(name for name in names if name.startswith(prefix))
            issues.extend(f"{effective_profile} must not include {name}" for name in forbidden)
        if missing_common:
            return issues
        _validate_file_index(archive, names, issues)
        _validate_checksums(archive, names, issues)
        _validate_portability_report(archive, issues, expected_profile=effective_profile)
        _scan_internal_markers(archive, names, issues)
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a portable ACP ZIP without Lean Agent Builder backend.")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--profile", default="acp-full")
    args = parser.parse_args()
    try:
        issues = validate_acp_zip(args.zip_path, args.profile)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"ACP ZIP rejected: {exc}", file=sys.stderr)
        return 1
    if issues:
        print("ACP ZIP rejected", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print("ACP ZIP accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
