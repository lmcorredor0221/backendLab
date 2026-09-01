from __future__ import annotations

import re
from typing import Any

from app.models import ACPFileEntry, ACPPreview
from app.services.acp_serialization import serialize_json_document, serialize_markdown_document
from app.services.acp_validation import build_acp_file_entry


ACP_REFERENCE_PATTERN = re.compile(r"ACP/[A-Za-z0-9._/\-]+")
ACP_DYNAMIC_REFERENCES = {
    "ACP/launcher/launch-report.json",
}
ACP_CONFORMANCE_PREFIX = "ACP/conformance/"
INTERNAL_RUNTIME_MARKERS = (
    "/api/v1/sessions",
    "SessionSnapshot",
    "journey_stage_artifact_id",
    "skill_run_id",
    "workspace_internal_id",
    "C:/Users/",
    "C:\\Users\\",
)
PROFILE_EXCLUDED_PREFIXES: dict[str, tuple[str, ...]] = {
    "blueprint-professional": (
        "ACP/launcher/",
        "ACP/adapters/",
        "ACP/construction-readiness/",
        "ACP/prompts/",
        "ACP/runtime/",
        "ACP/deployment/",
        "ACP/observability/",
        "ACP/evaluation/",
    ),
    "acp-portable": (
        "ACP/deployment/",
        "ACP/observability/",
    ),
    "acp-full": (),
}


def profile_excluded_prefixes(profile: str) -> tuple[str, ...]:
    return PROFILE_EXCLUDED_PREFIXES.get(profile, PROFILE_EXCLUDED_PREFIXES["acp-full"])


def _extract_acp_references(content: str) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()
    for match in ACP_REFERENCE_PATTERN.findall(content):
        reference = match.rstrip("`'\"),.;:]}>")
        if reference and reference not in seen:
            seen.add(reference)
            references.append(reference)
    return references


def _reference_exists(reference: str, paths: set[str], *, excluded_prefixes: tuple[str, ...]) -> bool:
    if reference in ACP_DYNAMIC_REFERENCES:
        return True
    if reference.startswith(ACP_CONFORMANCE_PREFIX):
        return True
    if any(reference.startswith(prefix) for prefix in excluded_prefixes):
        return True
    if reference in paths:
        return True
    normalized = reference.rstrip("/")
    return any(path.startswith(f"{normalized}/") for path in paths)


def _find_reference_integrity(
    files: list[ACPFileEntry],
    *,
    profile: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    paths = {item.path for item in files}
    excluded_prefixes = profile_excluded_prefixes(profile)
    broken_references: list[dict[str, str]] = []
    filtered_references: list[dict[str, str]] = []
    for item in files:
        for reference in _extract_acp_references(item.content_text):
            if any(reference.startswith(prefix) for prefix in excluded_prefixes):
                filtered_references.append({"source_path": item.path, "reference": reference})
                continue
            if not _reference_exists(reference, paths, excluded_prefixes=excluded_prefixes):
                broken_references.append({"source_path": item.path, "reference": reference})
    return broken_references, filtered_references


def _find_internal_markers(files: list[ACPFileEntry]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for item in files:
        normalized_content = item.content_text.replace("\\", "/")
        for marker in INTERNAL_RUNTIME_MARKERS:
            normalized_marker = marker.replace("\\", "/")
            if normalized_marker in normalized_content:
                findings.append({"path": item.path, "marker": marker})
    return findings


def _file_index_payload(files: list[ACPFileEntry], *, profile: str) -> dict[str, Any]:
    return {
        "schema_version": "acp-file-index.v1",
        "profile": profile,
        "file_count": len(files),
        "files": [
            {
                "path": item.path,
                "domain": item.domain,
                "title": item.title,
                "format": item.format,
                "status": item.status,
                "content_hash": item.content_hash,
                "source_sections": item.source_sections,
            }
            for item in sorted(files, key=lambda entry: entry.path)
        ],
    }


def _checksums_text(files: list[ACPFileEntry]) -> str:
    return "\n".join(
        f"{item.content_hash}  {item.path}"
        for item in sorted(files, key=lambda entry: entry.path)
        if item.content_hash
    )


def _signal_payload(files: list[ACPFileEntry], prefix: str) -> dict[str, Any]:
    paths = sorted(item.path for item in files if item.path.startswith(prefix))
    return {"present": bool(paths), "paths": paths}


def _implementation_actions(preview: ACPPreview, *, profile: str) -> list[str]:
    actions: list[str] = []
    readiness = preview.construction_readiness
    if readiness.open_questions:
        actions.append(
            "Revisar preguntas abiertas y resolverlas o conservarlas como decisiones delegadas antes de tocar el artefacto afectado."
        )
    if readiness.blocking_gaps:
        actions.append(
            "Cerrar bloqueos reales de integridad; no reabrir el Blueprint por deuda operativa interna cerrada en el handoff."
        )
    if profile == "blueprint-professional":
        actions.append("Adquirir ACP para obtener launcher, adapters, readiness operativo, prompts y runtime package.")
    elif profile == "acp-portable":
        actions.append("Seleccionar tecnologia, infraestructura y observabilidad durante la implementacion guiada.")
    else:
        actions.append("Usar el launcher y adapters para iniciar la implementacion en la herramienta agentica elegida.")
    return actions


def _portability_report_payload(
    preview: ACPPreview,
    files: list[ACPFileEntry],
    *,
    profile: str,
) -> dict[str, Any]:
    broken_references, filtered_references = _find_reference_integrity(files, profile=profile)
    internal_markers = _find_internal_markers(files)
    deferred_gaps = [
        {
            "gap_key": gap.gap_key,
            "domain": gap.domain,
            "severity": gap.severity,
            "status": gap.status,
            "summary": gap.summary,
            "questions": [question.question_key for question in gap.questions],
        }
        for gap in preview.construction_readiness.gaps
        if gap.status == "open" or gap.questions
    ]
    return {
        "schema_version": "acp-portability-report.v1",
        "profile": profile,
        "requires_lean_backend": False,
        "manifest_path": preview.manifest_path,
        "file_count": len(files),
        "package_version": preview.package_version,
        "profiles": [
            {
                "profile_key": "blueprint-professional",
                "product": "Blueprint Professional",
                "purpose": "Documento profesional de diseno, visualizacion y estimacion sin artefactos premium ACP.",
                "excluded_prefixes": list(profile_excluded_prefixes("blueprint-professional")),
            },
            {
                "profile_key": "acp-portable",
                "product": "Agent Construction Package Portable",
                "purpose": "Paquete portable para iniciar construccion agentica sin decisiones cerradas de deployment.",
                "excluded_prefixes": list(profile_excluded_prefixes("acp-portable")),
            },
            {
                "profile_key": "acp-full",
                "product": "Agent Construction Package Full",
                "purpose": "Paquete completo con guias de deployment, observabilidad y todos los artefactos tecnicos.",
                "excluded_prefixes": list(profile_excluded_prefixes("acp-full")),
            },
        ],
        "validation": {
            "overall_status": preview.validation.overall_status,
            "can_export_zip": preview.validation.can_export_zip,
            "completeness_percent": preview.validation.completeness_percent,
            "issues": [issue.model_dump(mode="json") for issue in preview.validation.issues],
        },
        "readiness": {
            "overall_status": preview.construction_readiness.overall_status,
            "can_start_build": preview.construction_readiness.can_start_build,
            "blocking_gaps": preview.construction_readiness.blocking_gaps,
            "open_questions": preview.construction_readiness.open_questions,
            "next_recommended_action": preview.construction_readiness.next_recommended_action,
        },
        "deferred_gaps": deferred_gaps,
        "signals": {
            "estimation": _signal_payload(files, "ACP/estimation/"),
            "evaluation": _signal_payload(files, "ACP/evaluation/"),
            "benchmarks": {
                "present": any(item.path == "ACP/evaluation/benchmarks.yaml" for item in files),
                "paths": [item.path for item in files if item.path == "ACP/evaluation/benchmarks.yaml"],
            },
            "conformance": {
                "present": True,
                "paths": [
                    "ACP/conformance/file-index.json",
                    "ACP/conformance/checksums.sha256",
                    "ACP/conformance/portability-report.json",
                    "ACP/conformance/portability-report.md",
                    "ACP/conformance/external-consumer-readme.md",
                ],
            },
        },
        "reference_integrity": {
            "broken_references": broken_references,
            "filtered_profile_references": filtered_references,
            "internal_markers": internal_markers,
        },
        "implementation_actions": _implementation_actions(preview, profile=profile),
        "ready_for_external_consumer": not broken_references and not internal_markers,
    }


def _portability_report_markdown(payload: dict[str, Any]) -> str:
    readiness = payload["readiness"]
    lines = [
        "# Reporte de portabilidad ACP",
        "",
        f"- Perfil: `{payload['profile']}`",
        f"- Requiere backend Lean Agent Builder: `{payload['requires_lean_backend']}`",
        f"- Archivos incluidos: `{payload['file_count']}`",
        f"- Estado de validacion: `{payload['validation']['overall_status']}`",
        f"- Readiness: `{readiness['overall_status']}`",
        f"- Preguntas abiertas: `{readiness['open_questions']}`",
        f"- Gaps bloqueantes: `{readiness['blocking_gaps']}`",
        "",
        "## Integridad",
        "",
        f"- Referencias rotas: `{len(payload['reference_integrity']['broken_references'])}`",
        f"- Marcadores internos detectados: `{len(payload['reference_integrity']['internal_markers'])}`",
        f"- Referencias filtradas por perfil: `{len(payload['reference_integrity']['filtered_profile_references'])}`",
        "",
        "## Senales sincronizadas",
        "",
        f"- Estimacion: `{payload['signals']['estimation']['present']}`",
        f"- Evaluacion: `{payload['signals']['evaluation']['present']}`",
        f"- Benchmarks: `{payload['signals']['benchmarks']['present']}`",
        "",
        "## Acciones de implementacion",
        "",
    ]
    lines.extend(f"- {action}" for action in payload["implementation_actions"])
    if payload["deferred_gaps"]:
        lines.extend(["", "## Gaps diferidos", ""])
        lines.extend(
            f"- `{gap['gap_key']}` ({gap['domain']}): {gap['summary']}"
            for gap in payload["deferred_gaps"]
        )
    return "\n".join(lines)


def build_acp_conformance_files(
    preview: ACPPreview,
    files: list[ACPFileEntry],
    *,
    profile: str = "acp-full",
) -> list[ACPFileEntry]:
    profile_key = profile if profile in PROFILE_EXCLUDED_PREFIXES else "acp-full"
    indexed_files = [item for item in files if not item.path.startswith(ACP_CONFORMANCE_PREFIX)]
    file_index = _file_index_payload(indexed_files, profile=profile_key)
    checksums = _checksums_text(indexed_files)
    portability_report = _portability_report_payload(preview, indexed_files, profile=profile_key)
    consumer_readme = "\n".join(
        [
            "# Consumidor externo del ACP",
            "",
            "Este paquete puede validarse sin backend Lean Agent Builder.",
            "",
            "1. Extrae el ZIP en un workspace local.",
            "2. Revisa `ACP/conformance/portability-report.md`.",
            "3. Valida `ACP/conformance/checksums.sha256` contra los archivos incluidos.",
            "4. Si el perfil es ACP, ejecuta el launcher o abre el paquete con una herramienta agentica compatible.",
            "",
            "El reporte separa gaps de diseno, decisiones diferidas de implementacion y referencias filtradas por perfil comercial.",
        ]
    )
    return [
        build_acp_file_entry(
            path="ACP/conformance/file-index.json",
            domain="conformance",
            title="File index",
            format="json",
            source_sections=["acp_files", "export_profile"],
            content_text=serialize_json_document(file_index),
        ),
        build_acp_file_entry(
            path="ACP/conformance/checksums.sha256",
            domain="conformance",
            title="Checksums",
            format="sha256",
            source_sections=["acp_files"],
            content_text=checksums,
        ),
        build_acp_file_entry(
            path="ACP/conformance/portability-report.json",
            domain="conformance",
            title="Portability report",
            format="json",
            source_sections=["validation", "construction_readiness", "export_profile"],
            content_text=serialize_json_document(portability_report),
        ),
        build_acp_file_entry(
            path="ACP/conformance/portability-report.md",
            domain="conformance",
            title="Portability report",
            format="markdown",
            source_sections=["validation", "construction_readiness", "export_profile"],
            content_text=serialize_markdown_document(_portability_report_markdown(portability_report)),
        ),
        build_acp_file_entry(
            path="ACP/conformance/external-consumer-readme.md",
            domain="conformance",
            title="External consumer README",
            format="markdown",
            source_sections=["conformance"],
            content_text=serialize_markdown_document(consumer_readme),
        ),
    ]
