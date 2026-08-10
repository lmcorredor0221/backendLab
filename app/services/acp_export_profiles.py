from __future__ import annotations

from typing import Literal

from app.models import ACPPreview, ACPValidationIssue, ACPValidationReport, ConstructionGapEntry, ConstructionReadinessReport
from app.services.acp_conformance import build_acp_conformance_files, profile_excluded_prefixes
from app.services.acp_serialization import serialize_markdown_document
from app.services.acp_validation import build_acp_file_entry

ACPExportProfile = Literal["blueprint-professional", "acp-portable", "acp-full", "design-only", "extended"]
EffectiveACPExportProfile = Literal["blueprint-professional", "acp-portable", "acp-full"]

PROFILE_ALIASES: dict[str, EffectiveACPExportProfile] = {
    "design-only": "acp-portable",
    "extended": "acp-full",
}
SUPPORTED_EXPORT_PROFILES = {
    "blueprint-professional",
    "acp-portable",
    "acp-full",
    "design-only",
    "extended",
}
PROFILE_EXCLUDED_GAP_DOMAINS: dict[EffectiveACPExportProfile, set[str]] = {
    "blueprint-professional": {
        "construction-readiness",
        "deployment",
        "evaluation",
        "integrations",
        "observability",
        "package",
        "prompts",
        "runtime",
    },
    "acp-portable": {"deployment"},
    "acp-full": set(),
}


def normalize_acp_export_profile(value: str | None) -> ACPExportProfile:
    normalized = (value or "extended").strip().lower()
    if normalized not in SUPPORTED_EXPORT_PROFILES:
        raise ValueError("Unsupported ACP export profile")
    return normalized  # type: ignore[return-value]


def effective_acp_export_profile(profile: ACPExportProfile | str) -> EffectiveACPExportProfile:
    normalized = normalize_acp_export_profile(profile)
    return PROFILE_ALIASES.get(normalized, normalized)  # type: ignore[return-value]


def _is_excluded_path(path: str, profile: EffectiveACPExportProfile) -> bool:
    return any(path.startswith(prefix) for prefix in profile_excluded_prefixes(profile))


def _flatten_assumptions(gaps: list[ConstructionGapEntry]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for gap in gaps:
        for assumption in gap.current_assumptions:
            normalized = assumption.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                items.append(normalized)
    return items


def _filter_validation_issues(
    issues: list[ACPValidationIssue],
    profile: EffectiveACPExportProfile,
) -> list[ACPValidationIssue]:
    return [
        issue
        for issue in issues
        if not issue.path or not _is_excluded_path(issue.path, profile)
    ]


def _rebuild_validation(
    preview: ACPPreview,
    *,
    files,
    profile: EffectiveACPExportProfile,
) -> ACPValidationReport:
    issues = _filter_validation_issues(preview.validation.issues, profile)
    complete_files = sum(1 for item in files if item.status == "complete")
    completeness_percent = round((complete_files / len(files)) * 100) if files else 0
    has_blocking_error = any(issue.severity == "error" and issue.blocking for issue in issues)

    overall_status = "complete"
    if has_blocking_error:
        overall_status = "incomplete"
    elif issues or any(item.status != "complete" for item in files):
        overall_status = "needs_review"

    return ACPValidationReport(
        overall_status=overall_status,
        completeness_percent=completeness_percent,
        can_export_zip=not has_blocking_error,
        issues=issues,
    )


def _filter_gap(
    gap: ConstructionGapEntry,
    *,
    allowed_paths: set[str],
    profile: EffectiveACPExportProfile,
) -> ConstructionGapEntry | None:
    if gap.domain in PROFILE_EXCLUDED_GAP_DOMAINS[profile]:
        return None

    evidence_paths = [path for path in gap.evidence_paths if path in allowed_paths]
    return gap.model_copy(
        update={
            "evidence_paths": evidence_paths,
        }
    )


def _rebuild_readiness(
    validation: ACPValidationReport,
    gaps: list[ConstructionGapEntry],
) -> ConstructionReadinessReport:
    blocking_gaps = sum(1 for item in gaps if item.severity == "blocking")
    open_questions = sum(len(item.questions) for item in gaps if item.status == "open")
    can_start_build = validation.can_export_zip and blocking_gaps == 0 and open_questions == 0

    if not validation.can_export_zip or blocking_gaps > 0:
        overall_status = "blocked"
        next_action = "resolve_blocking_construction_gaps"
    elif open_questions > 0:
        overall_status = "needs_questions"
        next_action = "answer_open_questions"
    else:
        overall_status = "ready_to_build"
        next_action = "start_agentic_build"

    return ConstructionReadinessReport(
        overall_status=overall_status,
        can_start_build=can_start_build,
        blocking_gaps=blocking_gaps,
        open_questions=open_questions,
        assumptions_count=len(_flatten_assumptions(gaps)),
        gaps=gaps,
        next_recommended_action=next_action,
    )


def _preserve_answered_gap_status(
    preview: ACPPreview,
    gaps: list[ConstructionGapEntry],
) -> list[ConstructionGapEntry]:
    if preview.construction_readiness.open_questions != 0:
        return gaps
    return [
        gap.model_copy(update={"status": "answered"})
        if gap.status == "open" and gap.questions
        else gap
        for gap in gaps
    ]


def _rewrite_readme_for_profile(item, *, profile: EffectiveACPExportProfile):
    if item.path != "ACP/README.md" or profile != "blueprint-professional":
        return item

    content = "\n".join(
        [
            "# Blueprint Professional",
            "",
            "Este paquete contiene el diseno profesional del sistema agentico y los artefactos necesarios para entender, vender y aprobar la solucion.",
            "",
            "## Contenido incluido",
            "",
            "- Arquitectura propuesta y topologia.",
            "- Objetivos, alcance, reglas de negocio y diseno funcional.",
            "- Patrones agentivos, razonamiento, herramientas, memoria y conocimiento.",
            "- Diagramas, contratos de alto nivel, estimacion, roadmap y evidencia de conformance.",
            "",
            "## Contenido premium no incluido",
            "",
            "- Launcher, adapters y handoff para herramientas agenticas.",
            "- Readiness operativo, preguntas de implementacion, runtime package, pruebas ejecutables y guias de despliegue.",
            "",
            "Para iniciar construccion portable adquiere el Agent Construction Package.",
        ]
    )
    return build_acp_file_entry(
        path=item.path,
        domain=item.domain,
        title=item.title,
        format=item.format,
        source_sections=item.source_sections,
        content_text=serialize_markdown_document(content),
    )


def _without_conformance(files):
    return [item for item in files if not item.path.startswith("ACP/conformance/")]


def apply_acp_export_profile(preview: ACPPreview, profile: ACPExportProfile | str) -> ACPPreview:
    normalized_profile = normalize_acp_export_profile(profile)
    effective_profile = effective_acp_export_profile(normalized_profile)

    files = [
        _rewrite_readme_for_profile(item, profile=effective_profile)
        for item in _without_conformance(preview.files)
        if not _is_excluded_path(item.path, effective_profile)
    ]
    files = sorted(files, key=lambda item: item.path)
    validation = _rebuild_validation(preview, files=files, profile=effective_profile)
    allowed_paths = {item.path for item in files}
    gaps = [
        filtered_gap
        for gap in preview.construction_readiness.gaps
        if (filtered_gap := _filter_gap(gap, allowed_paths=allowed_paths, profile=effective_profile)) is not None
    ]
    gaps = _preserve_answered_gap_status(preview, gaps)
    readiness = _rebuild_readiness(validation, gaps)
    profile_package_version = f"{preview.package_version}.{normalized_profile}"
    interim_preview = preview.model_copy(
        update={
            "files": files,
            "validation": validation,
            "construction_readiness": readiness,
            "package_version": profile_package_version,
        }
    )
    conformance_files = build_acp_conformance_files(interim_preview, files, profile=effective_profile)
    files = sorted(files + conformance_files, key=lambda item: item.path)
    validation = _rebuild_validation(preview, files=files, profile=effective_profile)
    readiness = _rebuild_readiness(validation, gaps)

    return preview.model_copy(
        update={
            "files": files,
            "validation": validation,
            "construction_readiness": readiness,
            "package_version": profile_package_version,
        }
    )


def rebuild_profile_conformance_with_readiness(
    preview: ACPPreview,
    *,
    profile: ACPExportProfile | str,
    readiness: ConstructionReadinessReport,
) -> ACPPreview:
    normalized_profile = normalize_acp_export_profile(profile)
    effective_profile = effective_acp_export_profile(normalized_profile)
    files_without_conformance = sorted(_without_conformance(preview.files), key=lambda item: item.path)
    effective_preview = preview.model_copy(update={"construction_readiness": readiness})
    conformance_files = build_acp_conformance_files(
        effective_preview,
        files_without_conformance,
        profile=effective_profile,
    )
    files = sorted(files_without_conformance + conformance_files, key=lambda item: item.path)
    validation = _rebuild_validation(effective_preview, files=files, profile=effective_profile)
    return effective_preview.model_copy(
        update={
            "files": files,
            "validation": validation,
        }
    )
