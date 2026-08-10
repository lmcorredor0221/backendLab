from __future__ import annotations

from typing import Iterable

from app.models import (
    ACPFileEntry,
    ACPPreview,
    ACPValidationIssue,
    ACPValidationReport,
    ArtifactStatus,
    SessionSnapshot,
)
from app.services.acp_construction_readiness import build_initial_construction_readiness
from app.services.acp_paths import build_tool_contract_path_for_tool
from app.services.acp_serialization import build_content_hash, normalize_text_document

VALIDATION_ISSUE_CATALOG: dict[str, dict[str, str]] = {
    "missing_agent_name": {
        "severity": "error",
        "remediation": "Asignar un titulo explicito a la sesion antes de exportar el ACP.",
    },
    "default_agent_name": {
        "severity": "warning",
        "remediation": "Renombrar la sesion con un nombre de agente estable y significativo.",
    },
    "missing_discovery": {
        "severity": "error",
        "remediation": "Persistir discovery antes de construir o exportar el ACP.",
    },
    "missing_business_objective": {
        "severity": "error",
        "remediation": "Completar `desired_outcome` para cerrar el objetivo de negocio.",
    },
    "missing_target_user": {
        "severity": "error",
        "remediation": "Declarar el usuario objetivo en discovery o canvas.",
    },
    "missing_autonomy_level": {
        "severity": "error",
        "remediation": "Definir el nivel de autonomia del agente antes de empaquetar.",
    },
    "missing_blueprint": {
        "severity": "error",
        "remediation": "Construir y persistir el blueprint antes del export canónico.",
    },
    "missing_architecture": {
        "severity": "error",
        "remediation": "Seleccionar una arquitectura gobernada en el blueprint.",
    },
    "missing_reasoning_pattern": {
        "severity": "error",
        "remediation": "Seleccionar un patron de razonamiento antes del export.",
    },
    "missing_memory_strategy": {
        "severity": "error",
        "remediation": "Definir la estrategia de memoria y sus capas minimas.",
    },
    "missing_guardrails": {
        "severity": "error",
        "remediation": "Agregar guardrails o safety checks visibles al blueprint.",
    },
    "blueprint_not_ready": {
        "severity": "warning",
        "remediation": "Cerrar los gaps del blueprint o aceptar explicitamente el riesgo antes del export final.",
    },
    "missing_evaluation_base": {
        "severity": "error",
        "remediation": "Generar dataset, rubrica o casos base de evaluacion antes de validar el ACP.",
    },
    "runtime_signals_missing": {
        "severity": "warning",
        "remediation": "Persistir health checks de integraciones para reducir placeholders de runtime.",
    },
    "tool_missing_name": {
        "severity": "error",
        "remediation": "Asignar nombre estable a la tool para poder generar su contrato.",
    },
    "tool_missing_description": {
        "severity": "error",
        "remediation": "Completar el `purpose` de la tool con una descripcion accionable.",
    },
    "tool_missing_inputs": {
        "severity": "error",
        "remediation": "Declarar al menos un input para la tool.",
    },
    "tool_missing_outputs": {
        "severity": "error",
        "remediation": "Declarar al menos un output verificable para la tool.",
    },
    "acp_file_incomplete": {
        "severity": "error",
        "remediation": "Completar los campos faltantes del archivo ACP antes de exportar.",
    },
    "acp_file_needs_review": {
        "severity": "warning",
        "remediation": "Revisar manualmente el archivo ACP antes de su uso automatizado.",
    },
}


def determine_acp_file_status(
    missing_fields: list[str] | None = None,
    warnings: list[str] | None = None,
) -> str:
    missing = [item for item in (missing_fields or []) if item.strip()]
    warning_items = [item for item in (warnings or []) if item.strip()]
    if missing:
        return "incomplete"
    if warning_items:
        return "needs_review"
    return "complete"


def build_acp_file_entry(
    *,
    path: str,
    domain: str,
    title: str,
    format: str,
    source_sections: list[str],
    content_text: str = "",
    missing_fields: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ACPFileEntry:
    normalized_text = normalize_text_document(content_text)
    clean_missing = [item.strip() for item in (missing_fields or []) if item.strip()]
    clean_warnings = [item.strip() for item in (warnings or []) if item.strip()]
    return ACPFileEntry(
        path=path,
        domain=domain,
        title=title,
        format=format,
        status=determine_acp_file_status(clean_missing, clean_warnings),
        source_sections=[item.strip() for item in source_sections if item.strip()],
        missing_fields=clean_missing,
        warnings=clean_warnings,
        content_text=normalized_text,
        content_hash=build_content_hash(normalized_text),
    )


def _normalize_file_entry(entry: ACPFileEntry) -> ACPFileEntry:
    return build_acp_file_entry(
        path=entry.path,
        domain=entry.domain,
        title=entry.title,
        format=entry.format,
        source_sections=entry.source_sections,
        content_text=entry.content_text,
        missing_fields=entry.missing_fields,
        warnings=entry.warnings,
    )


def _issue(
    code: str,
    message: str,
    *,
    severity: str = "warning",
    path: str = "",
    remediation: str = "",
    source_sections: list[str] | None = None,
    blocking: bool = False,
) -> ACPValidationIssue:
    catalog_entry = VALIDATION_ISSUE_CATALOG.get(code, {})
    return ACPValidationIssue(
        code=code,
        severity=catalog_entry.get("severity", severity),
        path=path,
        message=message,
        remediation=remediation or catalog_entry.get("remediation", ""),
        source_sections=source_sections or [],
        blocking=blocking,
    )


def _has_evaluation_base(snapshot: SessionSnapshot) -> bool:
    if snapshot.evaluation_dataset is not None and snapshot.evaluation_dataset.cases:
        return True
    if snapshot.evaluation_rubric is not None and snapshot.evaluation_rubric.dimensions:
        return True
    if snapshot.evaluation is not None and snapshot.evaluation.cases:
        return True
    blueprint = snapshot.blueprint
    if blueprint is None:
        return False
    return any(item.key == "test_cases" and bool(item.content_markdown.strip()) for item in blueprint.delivery_package.deliverables)


def _latest_blueprint_version_number(snapshot: SessionSnapshot) -> int | None:
    if not snapshot.blueprint_versions:
        return None
    return max(item.version_number for item in snapshot.blueprint_versions)


def _validation_checks(snapshot: SessionSnapshot) -> list[bool]:
    discovery = snapshot.discovery
    blueprint = snapshot.blueprint
    canvas = snapshot.canvas
    has_guardrails = bool(blueprint and (blueprint.guardrails or blueprint.safety_checks))
    has_runtime_signal = bool(snapshot.integration_statuses)
    return [
        bool(snapshot.session.title.strip()),
        bool(discovery and discovery.desired_outcome.strip()),
        bool((discovery and discovery.current_user.strip()) or (canvas and canvas.agent_profile.primary_user.strip())),
        bool(blueprint and blueprint.architecture.strip()),
        bool(blueprint and blueprint.reasoning_pattern.strip()),
        bool(discovery and discovery.autonomy_level.strip()),
        bool(blueprint and (blueprint.memory_strategy.strip() or blueprint.memory_profile.strategy.strip())),
        _has_evaluation_base(snapshot),
        has_guardrails,
        has_runtime_signal,
    ]


def _iter_tool_issues(snapshot: SessionSnapshot) -> Iterable[ACPValidationIssue]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []

    issues: list[ACPValidationIssue] = []
    for index, tool in enumerate(blueprint.tools, start=1):
        tool_path = build_tool_contract_path_for_tool(tool, index)
        if not tool.name.strip():
            issues.append(
                _issue(
                    "tool_missing_name",
                    "Una tool del blueprint no tiene nombre y no puede convertirse en contrato ACP.",
                    severity="error",
                    path=tool_path,
                    source_sections=["blueprint.tools"],
                    blocking=True,
                )
            )
        if not tool.purpose.strip():
            issues.append(
                _issue(
                    "tool_missing_description",
                    f"La tool '{tool.name or f'#{index}'}' no tiene descripcion/purpose.",
                    severity="error",
                    path=tool_path,
                    source_sections=["blueprint.tools"],
                    blocking=True,
                )
            )
        if not tool.inputs:
            issues.append(
                _issue(
                    "tool_missing_inputs",
                    f"La tool '{tool.name or f'#{index}'}' no define inputs.",
                    severity="error",
                    path=tool_path,
                    source_sections=["blueprint.tools"],
                    blocking=True,
                )
            )
        if not tool.outputs:
            issues.append(
                _issue(
                    "tool_missing_outputs",
                    f"La tool '{tool.name or f'#{index}'}' no define outputs.",
                    severity="error",
                    path=tool_path,
                    source_sections=["blueprint.tools"],
                    blocking=True,
                )
            )
    return issues


def build_acp_validation_report(
    snapshot: SessionSnapshot,
    files: list[ACPFileEntry] | None = None,
) -> ACPValidationReport:
    issues: list[ACPValidationIssue] = []
    discovery = snapshot.discovery
    blueprint = snapshot.blueprint
    canvas = snapshot.canvas

    if not snapshot.session.title.strip():
        issues.append(
            _issue(
                "missing_agent_name",
                "La sesion no tiene titulo y no puede derivarse un nombre base para el agente.",
                severity="error",
                path="ACP/manifest.yaml",
                source_sections=["session.title"],
                blocking=True,
            )
        )
    elif snapshot.session.title.strip().lower() == "nueva sesion":
        issues.append(
            _issue(
                "default_agent_name",
                "La sesion aun conserva el titulo por defecto; conviene renombrarla antes del ACP final.",
                path="ACP/manifest.yaml",
                source_sections=["session.title"],
            )
        )

    if discovery is None:
        issues.append(
            _issue(
                "missing_discovery",
                "No existe discovery persistido para construir el ACP.",
                severity="error",
                path="ACP/business/lean-canvas.yaml",
                source_sections=["discovery"],
                blocking=True,
            )
        )
    else:
        if not discovery.desired_outcome.strip():
            issues.append(
                _issue(
                    "missing_business_objective",
                    "Falta el objetivo de negocio o desired_outcome.",
                    severity="error",
                    path="ACP/manifest.yaml",
                    source_sections=["discovery.desired_outcome"],
                    blocking=True,
                )
            )
        if not (discovery.current_user.strip() or (canvas and canvas.agent_profile.primary_user.strip())):
            issues.append(
                _issue(
                    "missing_target_user",
                    "Falta el usuario objetivo del agente.",
                    severity="error",
                    path="ACP/manifest.yaml",
                    source_sections=["discovery.current_user", "canvas.agent_profile.primary_user"],
                    blocking=True,
                )
            )
        if not discovery.autonomy_level.strip():
            issues.append(
                _issue(
                    "missing_autonomy_level",
                    "Falta el nivel de autonomia del agente.",
                    severity="error",
                    path="ACP/manifest.yaml",
                    source_sections=["discovery.autonomy_level"],
                    blocking=True,
                )
            )

    if blueprint is None:
        issues.append(
            _issue(
                "missing_blueprint",
                "No existe blueprint persistido para construir el ACP.",
                severity="error",
                path="ACP/architecture/topology.yaml",
                source_sections=["blueprint"],
                blocking=True,
            )
        )
    else:
        if not blueprint.architecture.strip():
            issues.append(
                _issue(
                    "missing_architecture",
                    "Falta la arquitectura seleccionada del blueprint.",
                    severity="error",
                    path="ACP/manifest.yaml",
                    source_sections=["blueprint.architecture"],
                    blocking=True,
                )
            )
        if not blueprint.reasoning_pattern.strip():
            issues.append(
                _issue(
                    "missing_reasoning_pattern",
                    "Falta el patron de razonamiento del blueprint.",
                    severity="error",
                    path="ACP/cognition/reasoning.yaml",
                    source_sections=["blueprint.reasoning_pattern"],
                    blocking=True,
                )
            )
        if not (blueprint.memory_strategy.strip() or blueprint.memory_profile.strategy.strip()):
            issues.append(
                _issue(
                    "missing_memory_strategy",
                    "Falta la estrategia de memoria del agente.",
                    severity="error",
                    path="ACP/memory/strategy.yaml",
                    source_sections=["blueprint.memory_strategy", "blueprint.memory_profile.strategy"],
                    blocking=True,
                )
            )
        if not (blueprint.guardrails or blueprint.safety_checks):
            issues.append(
                _issue(
                    "missing_guardrails",
                    "El ACP requiere guardrails o safety checks base.",
                    severity="error",
                    path="ACP/cognition/guardrails.yaml",
                    source_sections=["blueprint.guardrails", "blueprint.safety_checks"],
                    blocking=True,
                )
            )
        elif blueprint.readiness_state != "complete":
            issues.append(
                _issue(
                    "blueprint_not_ready",
                    "El blueprint aun no esta completamente listo; el ACP deberia revisarse antes del export final.",
                    path="ACP/manifest.yaml",
                    source_sections=["blueprint.readiness_state"],
                )
            )

    if not _has_evaluation_base(snapshot):
        issues.append(
            _issue(
                "missing_evaluation_base",
                "Falta dataset, rubrica o casos de evaluacion base para el ACP.",
                severity="error",
                path="ACP/evaluation/rubrics.yaml",
                source_sections=["evaluation_dataset", "evaluation_rubric", "evaluation", "delivery_package.deliverables"],
                blocking=True,
            )
        )

    if not snapshot.integration_statuses:
        issues.append(
            _issue(
                "runtime_signals_missing",
                "No hay health checks persistidos de integraciones; runtime saldra con placeholders controlados.",
                path="ACP/runtime/config.yaml",
                source_sections=["integration_statuses"],
            )
        )

    issues.extend(_iter_tool_issues(snapshot))

    normalized_files = [_normalize_file_entry(item) for item in files] if files else []
    for item in normalized_files:
        if item.status == "incomplete":
            issues.append(
                _issue(
                    "acp_file_incomplete",
                    f"El archivo '{item.path}' tiene campos faltantes y no deberia exportarse aun.",
                    severity="error",
                    path=item.path,
                    source_sections=item.source_sections,
                    blocking=True,
                )
            )
        elif item.status == "needs_review" and item.warnings:
            issues.append(
                _issue(
                    "acp_file_needs_review",
                    f"El archivo '{item.path}' requiere revision antes del uso automatizado.",
                    path=item.path,
                    source_sections=item.source_sections,
                )
            )

    blocking_error = any(item.blocking and item.severity == "error" for item in issues)
    if normalized_files:
        complete_files = sum(1 for item in normalized_files if item.status == "complete")
        completeness_percent = round((complete_files / len(normalized_files)) * 100)
    else:
        checks = _validation_checks(snapshot)
        completeness_percent = round((sum(1 for item in checks if item) / len(checks)) * 100)

    overall_status = "complete"
    if blocking_error:
        overall_status = "incomplete"
    elif issues or any(item.status != "complete" for item in normalized_files):
        overall_status = "needs_review"

    return ACPValidationReport(
        overall_status=overall_status,
        completeness_percent=completeness_percent,
        can_export_zip=not blocking_error,
        issues=issues,
    )


def build_acp_preview(
    snapshot: SessionSnapshot,
    files: list[ACPFileEntry] | None = None,
) -> ACPPreview:
    normalized_files = [_normalize_file_entry(item) for item in (files or [])]
    validation = build_acp_validation_report(snapshot, normalized_files)
    construction_readiness = build_initial_construction_readiness(snapshot, normalized_files, validation)
    return ACPPreview(
        session_id=snapshot.session.id,
        blueprint_version_number=_latest_blueprint_version_number(snapshot),
        files=normalized_files,
        validation=validation,
        construction_readiness=construction_readiness,
    )


def should_block_acp_export(report: ACPValidationReport) -> bool:
    return not report.can_export_zip or any(
        item.severity == "error" and item.blocking
        for item in report.issues
    )


def derive_acp_export_status(report: ACPValidationReport) -> ArtifactStatus:
    if should_block_acp_export(report):
        return ArtifactStatus.failed
    if report.overall_status == "complete":
        return ArtifactStatus.ready
    return ArtifactStatus.needs_review
