from __future__ import annotations

import json
from typing import Any

from sqlmodel import Session

from app.models import (
    ACPLaunchReportRecord,
    ACPPreview,
    LauncherMetadataResponse,
    LauncherReportResponse,
    LauncherReportSubmitRequest,
    LauncherScriptResponse,
    SessionRecord,
    UserRecord,
)
from app.services.commerce_service import record_commercial_event


COMMANDS_BY_PLATFORM = {
    "windows_powershell": "powershell -ExecutionPolicy Bypass -File .\\ACP\\launcher\\start-acp.ps1",
    "windows_cmd": "ACP\\launcher\\start-acp.bat",
    "posix_shell": "sh ACP/launcher/start-acp.sh",
    "python": "python ACP/launcher/acp-launcher.py --dry-run --no-open",
}


def _file_map(preview: ACPPreview):
    return {item.path: item for item in preview.files}


def _parse_manifest(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = "\n".join(line for line in stripped.splitlines() if not line.strip().startswith("```"))
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_launcher_metadata(*, record: SessionRecord, preview: ACPPreview) -> LauncherMetadataResponse:
    files = _file_map(preview)
    manifest_path = "ACP/launcher/launch-manifest.json"
    manifest = _parse_manifest(files.get(manifest_path).content_text if manifest_path in files else "")
    entrypoints = manifest.get("entrypoints") if isinstance(manifest.get("entrypoints"), dict) else {}
    package = manifest.get("package") if isinstance(manifest.get("package"), dict) else {}
    scripts = [
        LauncherScriptResponse(
            platform=str(platform),
            path=str(path),
            command=COMMANDS_BY_PLATFORM.get(str(platform), str(path)),
            available=str(path) in files,
        )
        for platform, path in sorted(entrypoints.items())
    ]
    safe_defaults = manifest.get("safe_defaults") if isinstance(manifest.get("safe_defaults"), dict) else {}
    restrictions = [
        "No instala dependencias.",
        "No ejecuta builds, migraciones ni despliegues.",
        "No requiere servicios internos ni endpoints de Lean Agent Builder.",
        "Solo prepara diagnostico local y abre el workspace cuando el usuario lo permite.",
    ]
    return LauncherMetadataResponse(
        session_id=record.id,
        workspace_id=record.workspace_id,
        manifest_path=manifest_path,
        launcher_version=str(manifest.get("launcher_version") or ""),
        package_name=str(package.get("name") or record.title),
        requires_lean_backend=bool(package.get("requires_lean_backend", False)),
        report_output=str(manifest.get("report_output") or "ACP/launcher/launch-report.json"),
        scripts=scripts,
        restrictions=restrictions,
        safe_defaults=safe_defaults,
    )


def submit_launcher_report(
    db: Session,
    *,
    record: SessionRecord,
    current_user: UserRecord,
    payload: LauncherReportSubmitRequest,
) -> LauncherReportResponse:
    report = ACPLaunchReportRecord(
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=current_user.id,
        report_path=payload.report_path.strip() or "ACP/launcher/launch-report.json",
        launcher_version=payload.launcher_version.strip(),
        detected_tool=payload.detected_tool.strip(),
        detected_ide=payload.detected_ide.strip(),
        status=payload.status.strip() or "received",
        summary=payload.summary.strip()[:1000],
        report_payload=payload.report,
    )
    db.add(report)
    db.flush()
    record_commercial_event(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=current_user.id,
        event_key="acp_launcher_report_received",
        product_key="acp",
        source="acp_launcher",
        metadata={
            "report_id": str(report.id),
            "report_path": report.report_path,
            "launcher_version": report.launcher_version,
            "detected_tool": report.detected_tool,
            "detected_ide": report.detected_ide,
            "status": report.status,
        },
    )
    return LauncherReportResponse(
        id=report.id,
        workspace_id=report.workspace_id,
        session_id=report.session_id,
        report_path=report.report_path,
        launcher_version=report.launcher_version,
        detected_tool=report.detected_tool,
        detected_ide=report.detected_ide,
        status=report.status,
        summary=report.summary,
        created_at=report.created_at,
    )
