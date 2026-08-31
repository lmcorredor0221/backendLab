from __future__ import annotations

from typing import Any

from app.models import (
    ACPFileEntry,
    ACPPreview,
    ConstructionGapEntry,
    ConstructionQuestionResponseRecord,
    EstimationReportArtifact,
    SessionSnapshot,
)
from app.services.acp_conformance import build_acp_conformance_files
from app.services.acp_continuity import (
    append_construction_readiness_gaps,
    build_construction_decision_log,
    build_deferred_construction_decision_backlog,
    is_no_applicable_answer,
    parse_answer_list,
    parse_answer_pairs,
    parse_contract_answer_entries,
)
from app.services.blueprint_consistency_service import (
    ensure_blueprint_consistency_report,
    render_blueprint_consistency_markdown,
)
from app.services.acp_construction_readiness import INTERNAL_BUILDER_TOOL_NAMES
from app.services.acp_paths import (
    ACP_CANONICAL_ENV_TEMPLATE_PATH,
    build_tool_contract_path,
    build_tool_contract_path_for_tool,
    slugify_acp_token,
)
from app.services.acp_serialization import (
    serialize_json_document,
    serialize_markdown_document,
    serialize_yaml_document,
)
from app.services.acp_visualization import build_acp_visualization_files
from app.services.acp_validation import build_acp_file_entry, build_acp_preview
from app.services.deliverable_catalog.registry_service import list_registry_entries


def _slugify(value: str, default: str = "item") -> str:
    return slugify_acp_token(value, default=default)


def _title_from_path(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _find_deliverable(snapshot: SessionSnapshot, key: str) -> str:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return ""
    for item in blueprint.delivery_package.deliverables:
        if item.key == key:
            return item.content_markdown
    return ""


def _find_integration_detail(snapshot: SessionSnapshot, integration_key: str) -> str:
    for item in snapshot.integration_statuses:
        if item.integration_key == integration_key:
            return item.detail
    return ""


def _parse_detail_tokens(detail: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for chunk in detail.split():
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        tokens[key.strip()] = value.strip()
    return tokens


def _runtime_defaults(snapshot: SessionSnapshot) -> dict[str, str]:
    active_detail = _parse_detail_tokens(_find_integration_detail(snapshot, "llm_runtime"))
    if not active_detail:
        active_detail = _parse_detail_tokens(_find_integration_detail(snapshot, "openai"))
    model_name = active_detail.get("reasoning") or active_detail.get("fast") or active_detail.get("model") or "needs_review"
    return {
        "framework": "custom_workflow",
        "llm_provider": active_detail.get("provider", "openai"),
        "model": model_name,
        "vector_db": "needs_review",
    }


def _continuity_answer_text(
    continuity_answers: dict[str, str] | None,
    question_key: str,
) -> str:
    if not continuity_answers:
        return ""
    return continuity_answers.get(question_key, "").strip()


def _continuity_answer_pairs(
    continuity_answers: dict[str, str] | None,
    question_key: str,
    *,
    aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    return parse_answer_pairs(_continuity_answer_text(continuity_answers, question_key), aliases=aliases)


def _continuity_answer_list(
    continuity_answers: dict[str, str] | None,
    question_key: str,
) -> list[str]:
    return parse_answer_list(_continuity_answer_text(continuity_answers, question_key))


def _is_placeholder_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return value <= 0

    normalized = str(value).strip().lower()
    if not normalized:
        return True
    return any(
        token in normalized
        for token in (
            "needs_review",
            "pending",
            "captured_from_owner",
            "placeholder",
        )
    )


def _source_entries_from_answer(answer_text: str) -> list[dict[str, str]]:
    aliases = {
        "source": "name",
        "name": "name",
        "type": "type",
        "owner": "owner",
        "frequency": "frequency",
        "actualizacion": "frequency",
    }
    entries: list[dict[str, str]] = []
    for raw_line in answer_text.splitlines():
        item = raw_line.strip().lstrip("-").strip()
        if not item:
            continue
        parsed = parse_answer_pairs(item, aliases=aliases)
        if parsed:
            entries.append(parsed)
            continue
        entries.append({"name": item})
    if entries:
        return entries

    parsed = parse_answer_pairs(answer_text, aliases=aliases)
    if parsed:
        return [parsed]
    return entries


def _find_contract_answer_for_tool(tool_name: str, answer_text: str) -> dict[str, str] | None:
    entries = parse_contract_answer_entries(answer_text)
    if not entries:
        return None
    tool_slug = slugify_acp_token(tool_name, default=tool_name)
    for entry in entries:
        tool_value = entry.get("tool", "")
        if not tool_value:
            continue
        if slugify_acp_token(tool_value, default=tool_value) == tool_slug:
            return entry
    if len(entries) == 1 and not entries[0].get("tool"):
        return entries[0]
    return None


def _knowledge_sources_from_owner_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized_entries: list[dict[str, str]] = []
    for index, item in enumerate(entries, start=1):
        source_name = item.get("name", "").strip() or f"knowledge-source-{index}"
        source_key = _slugify(source_name, default=f"knowledge-source-{index}")
        source_type = item.get("type", "").strip() or "captured_from_owner"
        owner = item.get("owner", "").strip() or "captured_from_owner"
        frequency = item.get("frequency", "").strip()
        description = "Fuente capturada desde respuestas del owner."
        if frequency:
            description = f"{description} Actualizacion: {frequency}."
        normalized_entries.append(
            {
                "description": description,
                "key": source_key,
                "lineage_key": f"{source_key}::owner-captured",
                "license": "captured_from_owner",
                "owner": owner,
                "sensitivity": "internal",
                "source_type": source_type,
                "source_version": "owner-captured",
                "title": source_name,
                "uri": f"captured://knowledge/{source_key}",
            }
        )
    return normalized_entries


def _format_cop(value: float) -> str:
    return f"COP {value:,.0f}"


def _format_usd(value: float) -> str:
    return f"USD {value:,.2f}"


def _format_hours(value: float) -> str:
    return f"{value:,.0f}h"


def _build_estimation_markdown(report: EstimationReportArtifact) -> str:
    lines = [
        "# Estimacion comparativa",
        "",
        "## Resumen ejecutivo",
        (
            f"- Escenario tradicional: {_format_cop(report.traditional.estimated_cost)} | "
            f"{_format_hours(report.traditional.estimated_hours_total)} | "
            f"{report.traditional.estimated_duration_weeks:.1f} semanas"
        ),
        (
            f"- Escenario agentic: {_format_cop(report.agentic.estimated_cost)} | "
            f"{_format_hours(report.agentic.estimated_hours_total)} | "
            f"{report.agentic.estimated_duration_weeks:.1f} semanas"
        ),
        f"- Ahorro potencial: {_format_cop(report.agentic.net_savings_vs_traditional)}",
        f"- Automatizable estimado: {report.agentic.automation_coverage_percent}%",
        (
            f"- Confianza comercial: {report.confidence.label} "
            f"({report.confidence.score}/100, +/-{report.confidence.uncertainty_band_percent}%)"
        ),
        (
            f"- Proveedor activo: {report.agentic.active_provider.value} | "
            f"modelo economico: {report.agentic.economic_model or 'standard'} | "
            f"modelo runtime: {report.agentic.provider_model or 'n/d'}"
        ),
        "",
        "## Workstreams con mayor diferencia",
    ]

    workstream_deltas = sorted(
        [
            {
                "label": item.label,
                "traditional_hours": item.estimated_hours,
                "agentic_hours": next(
                    (
                        candidate.estimated_hours
                        for candidate in report.agentic.workstream_breakdown
                        if candidate.workstream_key == item.workstream_key
                    ),
                    0,
                ),
                "coverage": report.agentic.automation_coverage_by_workstream.get(item.workstream_key, 0),
            }
            for item in report.traditional.workstream_breakdown
        ],
        key=lambda item: item["traditional_hours"] - item["agentic_hours"],
        reverse=True,
    )
    for item in workstream_deltas[:5]:
        saved_hours = max(0, item["traditional_hours"] - item["agentic_hours"])
        lines.append(
            f"- {item['label']}: {_format_hours(item['traditional_hours'])} trad. vs {_format_hours(item['agentic_hours'])} agentic | ahorro {_format_hours(saved_hours)} | auto {item['coverage']}%"
        )

    lines.extend(
        [
            "",
            "## Supuestos principales",
            *[f"- {item}" for item in report.assumptions[:6]],
            "",
            "## Drivers de sensibilidad",
            *[f"- {item}" for item in report.risk_drivers[:6]],
            "",
            "## Siguientes acciones",
            *[f"- {item}" for item in report.confidence.recommended_next_actions[:6]],
        ]
    )
    return "\n".join(lines)


def _build_estimation_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    report = snapshot.estimation_report
    if report is None:
        return []
    report_dump = report.model_dump(mode="json")
    confidence_dump = report_dump["confidence"]
    traditional_dump = report_dump["traditional"]
    agentic_dump = report_dump["agentic"]

    workstream_deltas = []
    for item in report.traditional.workstream_breakdown:
        agentic_item = next(
            (candidate for candidate in report.agentic.workstream_breakdown if candidate.workstream_key == item.workstream_key),
            None,
        )
        agentic_hours = agentic_item.estimated_hours if agentic_item is not None else 0
        workstream_deltas.append(
            {
                "workstream_key": item.workstream_key,
                "label": item.label,
                "traditional_hours": item.estimated_hours,
                "agentic_hours": agentic_hours,
                "saved_hours": round(max(0, item.estimated_hours - agentic_hours), 2),
                "automation_percent": report.agentic.automation_coverage_by_workstream.get(item.workstream_key, 0),
            }
        )

    sensitivity_payload = {
        "confidence": {
            "score": confidence_dump["score"],
            "label": confidence_dump["label"],
            "uncertainty_band_percent": confidence_dump["uncertainty_band_percent"],
            "blocking_gaps": confidence_dump["blocking_gaps"],
            "open_questions": confidence_dump["open_questions"],
            "assumptions_count": confidence_dump["assumptions_count"],
            "subscores": confidence_dump["subscores"],
        },
        "risk_drivers": report_dump["risk_drivers"],
        "recommended_next_actions": confidence_dump["recommended_next_actions"],
        "positive_signals": confidence_dump["positive_signals"],
        "negative_signals": confidence_dump["negative_signals"],
        "workstream_deltas": sorted(workstream_deltas, key=lambda item: item["saved_hours"], reverse=True),
        "automation_floor_families": [
            {
                "family_key": item["family_key"],
                "label": item["label"],
                "coverage_percent": item["coverage_percent"],
                "risk_tier": item["risk_tier"],
                "mandatory_human_review": item["mandatory_human_review"],
                "non_automatable_reasons": item["non_automatable_reasons"],
            }
            for item in sorted(agentic_dump["automation_assessments"], key=lambda entry: entry["coverage_percent"])[:6]
        ],
    }
    assumptions_payload = {
        "maturity_stage": report_dump["maturity_stage"],
        "assumptions": report_dump["assumptions"],
        "traditional_warnings": traditional_dump["warnings"],
        "agentic_warnings": agentic_dump["warnings"],
        "pricing_assumptions": agentic_dump["pricing_assumptions"],
        "notes": report_dump["notes"],
    }
    warnings = (
        ["La estimacion integrada al ACP sigue teniendo banda amplia; leer como rango y no como cifra cerrada."]
        if report.confidence.label in {"low", "medium_low"}
        else []
    )

    return [
        build_acp_file_entry(
            path="ACP/estimation/estimation-report.json",
            domain="estimation",
            title="Estimation report JSON",
            format="json",
            source_sections=["estimation_report", "construction_readiness", "evaluation_runs"],
            content_text=serialize_json_document(report.model_dump(mode="json")),
            warnings=warnings,
        ),
        build_acp_file_entry(
            path="ACP/estimation/estimation-report.md",
            domain="estimation",
            title="Estimation report",
            format="markdown",
            source_sections=["estimation_report", "construction_readiness", "evaluation_runs"],
            content_text=serialize_markdown_document(_build_estimation_markdown(report)),
            warnings=warnings,
        ),
        build_acp_file_entry(
            path="ACP/estimation/assumptions.yaml",
            domain="estimation",
            title="Estimation assumptions",
            format="yaml",
            source_sections=["estimation_report", "construction_readiness.gaps.current_assumptions"],
            content_text=serialize_yaml_document(assumptions_payload),
            warnings=warnings,
        ),
        build_acp_file_entry(
            path="ACP/estimation/sensitivity-drivers.yaml",
            domain="estimation",
            title="Estimation sensitivity drivers",
            format="yaml",
            source_sections=["estimation_report", "construction_readiness.gaps", "construction_readiness.gaps.questions"],
            content_text=serialize_yaml_document(sensitivity_payload),
            warnings=warnings,
        ),
    ]


def _tool_contract_file(tool: Any, index: int) -> ACPFileEntry:
    tool_type = getattr(tool, "tool_type", "external") or "external"
    payload = {
        "name": tool.name or f"tool_{index}",
        "purpose": tool.purpose,
        "tool_type": tool_type,
        "execution_stage": getattr(tool, "execution_stage", "tools") or "tools",
        "when_to_use": getattr(tool, "when_to_use", "") or "",
        "type": "write" if tool.has_side_effects else "read",
        "risk_level": tool.risk_level or "needs_review",
        "inputs": getattr(tool, "request_schema", {}) or {item: {"type": "string", "required": False} for item in tool.inputs},
        "outputs": getattr(tool, "response_schema", {}) or {item: {"type": "string"} for item in tool.outputs},
        "usage_examples": getattr(tool, "usage_examples", []),
        "security_config": getattr(tool, "security_config", {}),
        "registered_api_ref": getattr(tool, "registered_api_ref", ""),
        "permissions": {
            "requires_approval": tool.requires_approval,
            "approval_reason": tool.approval_reason,
            "allowed_roles": getattr(tool, "permissions", []),
        },
        "execution": {
            "retry_policy": tool.retry_strategy or "needs_review",
            "timeout_policy": getattr(tool, "timeout_policy", "30s"),
            "side_effects": tool.has_side_effects,
            "compensation_required": bool(tool.compensation_strategy.strip()),
            "compensation_strategy": tool.compensation_strategy,
            "failure_mode": tool.failure_mode,
        },
    }
    warnings: list[str] = []
    if not tool.inputs and not payload["inputs"]:
        warnings.append("La tool no incluye un schema estructurado de inputs; se usa representacion minima.")
    if not tool.outputs and not payload["outputs"]:
        warnings.append("La tool no incluye un schema estructurado de outputs; se usa representacion minima.")
    path = build_tool_contract_path(tool.name, index, tool_type=tool_type)
    return build_acp_file_entry(
        path=path,
        domain="tools",
        title=tool.name or f"Tool contract {index}",
        format="yaml",
        source_sections=["blueprint.tools", "approvals", "risk_summary"],
        content_text=serialize_yaml_document(payload),
        missing_fields=[],
        warnings=warnings,
    )


def _build_manifest_file(snapshot: SessionSnapshot) -> ACPFileEntry:
    discovery = snapshot.discovery
    blueprint = snapshot.blueprint
    runtime = _runtime_defaults(snapshot)
    payload = {
        "metadata": {
            "id": _slugify(snapshot.session.title, default="agent"),
            "name": snapshot.session.title.strip() or "needs_review",
            "version": "1.0.0",
            "maturity": "MVP",
            "generated_by": "Lean Agent Builder",
        },
        "business": {
            "objective": discovery.desired_outcome if discovery else "",
            "users": [discovery.current_user] if discovery and discovery.current_user else [],
            "kpis": [discovery.mvp_definition.north_star_metric] if discovery and discovery.mvp_definition.north_star_metric else [],
        },
        "architecture": {
            "topology": blueprint.architecture if blueprint else "",
            "reasoning_pattern": blueprint.reasoning_pattern if blueprint else "",
            "autonomy_level": discovery.autonomy_level if discovery else "",
        },
        "memory": {
            "strategy": blueprint.memory_profile.strategy if blueprint else "",
            "short_term": "session",
            "long_term": "needs_review",
        },
        "runtime": runtime,
        "delivery": {
            "target": "ai-builder-agent",
            "format": "agent-construction-package",
        },
    }
    return build_acp_file_entry(
        path="ACP/manifest.yaml",
        domain="manifest",
        title="Manifest",
        format="yaml",
        source_sections=["session.title", "discovery", "blueprint", "integration_statuses"],
        content_text=serialize_yaml_document(payload),
    )


def _build_readme_file(snapshot: SessionSnapshot) -> ACPFileEntry:
    discovery = snapshot.discovery
    blueprint = snapshot.blueprint
    canvas = snapshot.canvas
    title = snapshot.session.title or "Agent Construction Package"

    desired_outcome = discovery.desired_outcome if discovery and discovery.desired_outcome else "Automatizar flujos operativos con garantías de trazabilidad y gobernanza."
    problem_statement = discovery.problem_statement if discovery and discovery.problem_statement else "Optimización operativa mediante agentes de IA."
    primary_user = (canvas.agent_profile.primary_user if canvas and canvas.agent_profile else None) or (discovery.current_user if discovery else "Usuario operativo")
    architecture = blueprint.architecture if blueprint and blueprint.architecture else "Arquitectura agéntica estructurada"
    reasoning_pattern = blueprint.reasoning_pattern if blueprint and blueprint.reasoning_pattern else "Plan-and-Execute"
    memory_strategy = blueprint.memory_strategy if blueprint and blueprint.memory_strategy else "Memoria dual (sesión + RAG)"
    autonomy_level = discovery.autonomy_level if discovery and discovery.autonomy_level else "Supervisada (HITL)"
    tool_count = len(blueprint.tools) if blueprint and blueprint.tools else 0

    sections = [
        f"# {title} — Agent Construction Package (ACP v2)",
        "",
        "> **Paquete de Construcción Portable y Ejecutable para Desarrolladores e IDEs Agénticos**",
        "> Este paquete contiene la especificación formal, contratos tipados, prompts de ingeniería, suite de pruebas y guías de ensamblaje para construir y desplegar el agente en producción sin ambigüedades.",
        "",
        "## 1. Resumen Ejecutivo del Agente",
        f"- **Objetivo de Negocio:** {desired_outcome}",
        f"- **Problema Operativo:** {problem_statement}",
        f"- **Usuario / Actor Primario:** {primary_user}",
        f"- **Topología de Arquitectura:** `{architecture}`",
        f"- **Modelo de Razonamiento:** `{reasoning_pattern}`",
        f"- **Estrategia de Memoria:** `{memory_strategy}`",
        f"- **Nivel de Autonomía:** `{autonomy_level}`",
        f"- **Herramientas Gobernadas:** `{tool_count}` herramientas con contratos de interfaz.",
        "",
        "## 2. Estructura de Directorios del Paquete",
        "```text",
        "ACP/",
        "├── manifest.yaml             # Declaración formal de dependencias y versiones",
        "├── business/                 # Lean canvas, KPIs y restricciones de negocio",
        "├── architecture/             # Topología, C4 context y traza de decisiones",
        "├── cognition/                # Patrones de razonamiento y perfiles de workflow",
        "├── memory/                   # Estrategia de memoria dual y context budgets",
        "├── knowledge/                # Fuentes documentales, embeddings e ingestión",
        "├── tools/                    # Contratos tipados y permisos de herramientas",
        "├── workflows/                # Máquinas de estado y grafos ejecutables",
        "├── prompts/                  # Prompts de roles (planner, evaluator, system, skills)",
        "├── adapters/                 # Guías de configuración para Cursor, Codex y Claude Code",
        "├── conformance/              # Reglas de validación y linters de construcción",
        "├── evaluation/               # Datasets de evaluación y casos de prueba",
        "├── deployment/               # Especificaciones de infraestructura y runtime",
        "└── launcher/                 # Scripts de inicialización multiplataforma",
        "```",
        "",
        "## 3. Instrucciones de Arranque y Asistencia con IDEs",
        "### Aceleración con Herramientas Agénticas:",
        "- **Cursor IDE:** Abre la raíz del proyecto en Cursor. Las directivas de contexto se encuentran preconfiguradas.",
        "- **Claude Code:** Ejecuta `claude` en el directorio para que interprete automáticamente `ACP/manifest.yaml` y los prompts de `ACP/prompts/`.",
        "- **Codex CLI:** Ejecuta `codex` para inicializar el asistente de construcción.",
        "",
        "### Script de Inicialización Automática:",
        "- **Windows (PowerShell):** `ACP/launcher/start-acp.ps1`",
        "- **Windows (CMD):** `ACP\\launcher\\start-acp.bat`",
        "- **macOS / Linux:** `sh ACP/launcher/start-acp.sh`",
        "",
        "## 4. Gobernanza y Fuente de Verdad",
        "Este paquete ha sido generado y validado contra el baseline del **Lean Agent Builder**. Todo cambio en los contratos debe registrarse en los archivos de conformance correspondientes.",
    ]
    return build_acp_file_entry(
        path="ACP/README.md",
        domain="manifest",
        title="README",
        format="markdown",
        source_sections=["session.title", "discovery", "blueprint", "evaluation"],
        content_text=serialize_markdown_document("\n".join(sections)),
    )


def _deliverable_catalog_entry_payload(entry) -> dict[str, Any]:
    def path_hint(path: str) -> str:
        normalized = str(path or "").strip()
        if normalized.startswith("ACP/"):
            return normalized.replace("ACP/", "${ACP_ROOT}/", 1)
        return normalized

    return {
        "key": entry.deliverable_key,
        "title": entry.title,
        "description": entry.description,
        "type": entry.deliverable_type.value,
        "category": entry.category,
        "stage": entry.stage,
        "enabled_from_stage": entry.enabled_from_stage,
        "product_scope": list(entry.product_scope),
        "required_tier": entry.required_tier.value,
        "access_level": entry.access_level,
        "formats": entry.formats.model_dump(mode="json"),
        "generation_mode": entry.generation_mode.value,
        "prompt_policy": entry.prompt_policy.model_dump(mode="json"),
        "context_policy": entry.context_policy.model_dump(mode="json"),
        "quality_policy": entry.quality_policy.model_dump(mode="json"),
        "dependency_policy": entry.dependency_policy.model_dump(mode="json"),
        "path_reference_policy": "Path hints use ${ACP_ROOT}; they are not active ZIP references unless materialized in the package.",
        "canonical_path_hints": [path_hint(path) for path in entry.canonical_paths],
        "portable_path_hints": [path_hint(path) for path in entry.portable_paths],
        "exportable": entry.exportable,
        "blueprint_download": entry.blueprint_download,
        "acp_download": entry.acp_download,
    }


def _build_deliverable_catalog_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    entries = [entry for entry in list_registry_entries() if entry.active]
    acp_entries = [entry for entry in entries if "acp" in entry.product_scope]
    blueprint_entries = [entry for entry in entries if entry.blueprint_download]
    acp_payload = {
        "schema_version": "acp-deliverable-catalog.v1",
        "package_portability": {
            "portable": True,
            "requires_origin_platform": False,
            "contains_internal_session_ids": False,
        },
        "source_policy": "Deliverable Catalog governance resolved at export time.",
        "session_title": snapshot.session.title,
        "entries": [_deliverable_catalog_entry_payload(entry) for entry in acp_entries],
        "counts": {
            "total": len(acp_entries),
            "diagrams": sum(1 for entry in acp_entries if entry.deliverable_type.value == "diagram"),
            "prompts": sum(1 for entry in acp_entries if entry.deliverable_type.value == "prompt"),
            "contracts": sum(1 for entry in acp_entries if entry.deliverable_type.value == "contract"),
            "tests": sum(1 for entry in acp_entries if entry.deliverable_type.value == "test"),
            "packages": sum(1 for entry in acp_entries if entry.deliverable_type.value == "package"),
        },
    }
    blueprint_payload = {
        "schema_version": "blueprint-export-scope.v1",
        "product": "blueprint_pro",
        "downloadable_entries": [_deliverable_catalog_entry_payload(entry) for entry in blueprint_entries],
        "restricted_from_blueprint": [
            entry.deliverable_key
            for entry in acp_entries
            if entry.required_tier.value == "acp" or entry.acp_download
        ],
        "separation_policy": "Blueprint Pro exports design documentation; ACP exports construction-ready implementation artifacts.",
    }
    return [
        build_acp_file_entry(
            path="ACP/governance/deliverable-catalog.acp.json",
            domain="governance",
            title="ACP deliverable catalog",
            format="json",
            source_sections=["deliverable_catalog", "commercial_access", "governance"],
            content_text=serialize_json_document(acp_payload),
        ),
        build_acp_file_entry(
            path="ACP/governance/blueprint-export-scope.json",
            domain="governance",
            title="Blueprint export scope",
            format="json",
            source_sections=["deliverable_catalog", "commercial_access", "governance"],
            content_text=serialize_json_document(blueprint_payload),
        ),
    ]


def _build_launcher_python_script() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAUNCHER_VERSION = "acp-launcher.v1"

AGENTIC_TOOLS = [
    {
        "key": "codex-cli",
        "label": "Codex CLI",
        "commands": ["codex"],
        "type": "agentic_cli",
        "priority": 100,
        "open_policy": "recommend_command_only",
        "suggested_command": "codex",
        "adapter_path": "ACP/adapters/codex-cli.md",
    },
    {
        "key": "claude-code",
        "label": "Claude Code",
        "commands": ["claude"],
        "type": "agentic_cli",
        "priority": 90,
        "open_policy": "recommend_command_only",
        "suggested_command": "claude",
        "adapter_path": "ACP/adapters/claude-code.md",
    },
    {
        "key": "cursor",
        "label": "Cursor",
        "commands": ["cursor"],
        "type": "agentic_ide",
        "priority": 80,
        "open_policy": "open_workspace",
        "suggested_command": "cursor .",
        "adapter_path": "ACP/adapters/cursor.md",
    },
    {
        "key": "github-copilot-vscode",
        "label": "GitHub Copilot via VS Code",
        "commands": ["code"],
        "type": "ide_assistant",
        "priority": 70,
        "open_policy": "open_workspace",
        "suggested_command": "code .",
        "adapter_path": "ACP/adapters/github-copilot.md",
    },
]

PREREQUISITES = [
    {"key": "python", "commands": ["python3", "python"], "required": True},
    {"key": "git", "commands": ["git"], "required": False},
    {"key": "node", "commands": ["node"], "required": False},
]

EXPECTED_FILES = [
    "ACP/manifest.yaml",
    "ACP/README.md",
    "ACP/prompts/builder-handoff.md",
    "ACP/construction-readiness/overview.yaml",
    "ACP/runtime/config.yaml",
    "ACP/runtime/providers.yaml",
    "ACP/workflows/durable-workflow.yaml",
]


def _command_path(candidates: list[str]) -> tuple[str, str] | None:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return candidate, resolved
    return None


def _detect_commands(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detected: list[dict[str, Any]] = []
    for entry in entries:
        match = _command_path(list(entry["commands"]))
        detected.append(
            {
                **entry,
                "available": match is not None,
                "command": match[0] if match else "",
                "path": match[1] if match else "",
            }
        )
    return detected


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _inspect_package(workspace_root: Path) -> dict[str, Any]:
    files = []
    missing = []
    for item in EXPECTED_FILES:
        exists = (workspace_root / item).exists()
        files.append({"path": item, "exists": exists})
        if not exists:
            missing.append(item)
    acp_root = workspace_root / "ACP"
    total_files = sum(1 for path in acp_root.rglob("*") if path.is_file()) if acp_root.exists() else 0
    return {
        "acp_root": _relative(acp_root, workspace_root),
        "exists": acp_root.exists(),
        "total_files": total_files,
        "expected_files": files,
        "missing_files": missing,
    }


def _choose_recommendation(tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    available = [tool for tool in tools if tool["available"]]
    if not available:
        return None
    return sorted(available, key=lambda item: item["priority"], reverse=True)[0]


def _next_steps(package_state: dict[str, Any], recommendation: dict[str, Any] | None) -> list[str]:
    steps = [
        "Revisar ACP/README.md y ACP/construction-readiness/overview.yaml.",
        "Resolver preguntas abiertas antes de construir codigo dependiente de entorno, secretos o infraestructura.",
        "Usar ACP/adapters/adapter-registry.json para mapear el paquete a la herramienta elegida.",
    ]
    if package_state["missing_files"]:
        steps.insert(0, "El paquete no tiene todos los archivos esperados; validar integridad del ZIP antes de continuar.")
    if recommendation:
        steps.append(f"Herramienta recomendada detectada: {recommendation['label']}. Ver {recommendation['adapter_path']}.")
        if recommendation["open_policy"] == "recommend_command_only":
            steps.append(
                f"Comando sugerido: abrir una terminal en la raiz del paquete y ejecutar `{recommendation['suggested_command']}`."
            )
    else:
        steps.append("No se detecto herramienta agentica/IDE compatible; usar el ACP como guia manual o instalar una herramienta de preferencia.")
    return steps


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _open_workspace(recommendation: dict[str, Any] | None, workspace_root: Path, *, no_open: bool, dry_run: bool) -> dict[str, Any]:
    if no_open or dry_run or recommendation is None:
        return {"attempted": False, "reason": "dry_run_or_no_open_or_no_recommendation"}
    if recommendation.get("open_policy") != "open_workspace":
        return {"attempted": False, "reason": "selected_tool_requires_manual_command"}
    command = recommendation.get("command")
    if not command:
        return {"attempted": False, "reason": "command_not_available"}
    try:
        subprocess.Popen([command, str(workspace_root)], cwd=str(workspace_root))
    except OSError as exc:
        return {"attempted": True, "success": False, "error": str(exc)}
    return {"attempted": True, "success": True, "command": command, "cwd": str(workspace_root)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="ACP portable launcher")
    parser.add_argument("--workspace", default="", help="Raiz del paquete extraido. Por defecto usa el padre de ACP/.")
    parser.add_argument("--report", default="", help="Ruta destino del launch-report.json.")
    parser.add_argument("--dry-run", action="store_true", help="Solo genera reporte; no abre IDE.")
    parser.add_argument("--no-open", action="store_true", help="No abre IDE aunque exista.")
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    inferred_workspace = script_path.parents[2] if script_path.parent.name == "launcher" else Path.cwd()
    workspace_root = Path(args.workspace).resolve() if args.workspace else inferred_workspace
    report_path = Path(args.report).resolve() if args.report else workspace_root / "ACP" / "launcher" / "launch-report.json"

    detected_tools = _detect_commands(AGENTIC_TOOLS)
    detected_prerequisites = _detect_commands(PREREQUISITES)
    recommendation = _choose_recommendation(detected_tools)
    package_state = _inspect_package(workspace_root)
    open_result = _open_workspace(recommendation, workspace_root, no_open=args.no_open, dry_run=args.dry_run)

    report = {
        "launcher_version": LAUNCHER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
        },
        "workspace_root": str(workspace_root),
        "package_state": package_state,
        "detected_tools": detected_tools,
        "detected_prerequisites": detected_prerequisites,
        "recommendation": recommendation,
        "open_result": open_result,
        "safety": {
            "installs_dependencies": False,
            "runs_build": False,
            "runs_destructive_commands": False,
            "requires_lean_backend": False,
        },
        "next_steps": _next_steps(package_state, recommendation),
    }
    _write_report(report_path, report)
    print(f"ACP launch report written: {report_path}")
    if recommendation:
        print(f"Recommended tool: {recommendation['label']}")
    else:
        print("No compatible agentic tool or IDE detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''


def _build_launcher_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    package_name = _slugify(snapshot.session.title, default="agent")
    launch_manifest = {
        "launcher_version": "acp-launcher.v1",
        "package": {
            "name": snapshot.session.title.strip() or package_name,
            "portable": True,
            "requires_lean_backend": False,
        },
        "entrypoints": {
            "windows_powershell": "ACP/launcher/start-acp.ps1",
            "windows_cmd": "ACP/launcher/start-acp.bat",
            "posix_shell": "ACP/launcher/start-acp.sh",
            "python": "ACP/launcher/acp-launcher.py",
        },
        "report_output": "ACP/launcher/launch-report.json",
        "safe_defaults": {
            "installs_dependencies": False,
            "runs_build": False,
            "runs_destructive_commands": False,
            "opens_workspace_only": True,
        },
        "expected_inputs": [
            "ACP/manifest.yaml",
            "ACP/construction-readiness/overview.yaml",
            "ACP/prompts/builder-handoff.md",
            "ACP/runtime/config.yaml",
            "ACP/adapters/adapter-registry.json",
        ],
    }
    powershell = r'''$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "acp-launcher.py"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  $python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $python) {
  Write-Error "Python no esta disponible. Instala Python 3 o ejecuta manualmente la guia ACP/launcher/README.md."
  exit 1
}
& $python.Source $script @args
exit $LASTEXITCODE
'''
    batch = r'''@echo off
setlocal
set SCRIPT_DIR=%~dp0
where python >nul 2>nul
if %errorlevel%==0 (
  python "%SCRIPT_DIR%acp-launcher.py" %*
  exit /b %errorlevel%
)
where py >nul 2>nul
if %errorlevel%==0 (
  py "%SCRIPT_DIR%acp-launcher.py" %*
  exit /b %errorlevel%
)
echo Python no esta disponible. Instala Python 3 o ejecuta manualmente la guia ACP\launcher\README.md.
exit /b 1
'''
    shell = r'''#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$SCRIPT_DIR/acp-launcher.py" "$@"
fi
if command -v python >/dev/null 2>&1; then
  exec python "$SCRIPT_DIR/acp-launcher.py" "$@"
fi
echo "Python no esta disponible. Instala Python 3 o ejecuta manualmente la guia ACP/launcher/README.md." >&2
exit 1
'''
    readme = "\n".join(
        [
            "# ACP Launcher",
            "",
            "Este launcher ayuda a iniciar el Agent Construction Package en un entorno local sin depender de Lean Agent Builder.",
            "",
            "## Comandos",
            "",
            "- Windows PowerShell: `ACP/launcher/start-acp.ps1`",
            "- Windows CMD: `ACP\\launcher\\start-acp.bat`",
            "- macOS/Linux: `sh ACP/launcher/start-acp.sh`",
            "- Solo reporte: `python ACP/launcher/acp-launcher.py --dry-run --no-open`",
            "",
            "## Que hace",
            "",
            "- Detecta Codex CLI, Claude Code, Cursor, VS Code/Copilot y prerequisitos basicos.",
            "- Genera `ACP/launcher/launch-report.json` con hallazgos y siguientes pasos.",
            "- Abre el workspace en Cursor o VS Code si estan disponibles y no se usa `--no-open`.",
            "",
            "## Que no hace",
            "",
            "- No instala dependencias.",
            "- No ejecuta builds, migraciones ni despliegues.",
            "- No lee servicios internos de Lean Agent Builder.",
            "- No modifica credenciales ni archivos fuera del paquete extraido.",
        ]
    )
    return [
        build_acp_file_entry(
            path="ACP/launcher/launch-manifest.json",
            domain="launcher",
            title="Launcher manifest",
            format="json",
            source_sections=["session.title", "acp_launcher"],
            content_text=serialize_json_document(launch_manifest),
        ),
        build_acp_file_entry(
            path="ACP/launcher/acp-launcher.py",
            domain="launcher",
            title="ACP launcher",
            format="python",
            source_sections=["acp_launcher"],
            content_text=serialize_markdown_document(_build_launcher_python_script()),
        ),
        build_acp_file_entry(
            path="ACP/launcher/start-acp.ps1",
            domain="launcher",
            title="Start ACP PowerShell",
            format="powershell",
            source_sections=["acp_launcher"],
            content_text=serialize_markdown_document(powershell),
        ),
        build_acp_file_entry(
            path="ACP/launcher/start-acp.bat",
            domain="launcher",
            title="Start ACP CMD",
            format="batch",
            source_sections=["acp_launcher"],
            content_text=serialize_markdown_document(batch),
        ),
        build_acp_file_entry(
            path="ACP/launcher/start-acp.sh",
            domain="launcher",
            title="Start ACP shell",
            format="shell",
            source_sections=["acp_launcher"],
            content_text=serialize_markdown_document(shell),
        ),
        build_acp_file_entry(
            path="ACP/launcher/README.md",
            domain="launcher",
            title="Launcher README",
            format="markdown",
            source_sections=["acp_launcher"],
            content_text=serialize_markdown_document(readme),
        ),
    ]


def _build_adapter_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    registry = {
        "schema_version": "acp-adapter-registry.v1",
        "framework_neutral": True,
        "requires_lean_backend": False,
        "source_artifacts": {
            "manifest": "ACP/manifest.yaml",
            "readiness": "ACP/construction-readiness/overview.yaml",
            "runtime": "ACP/runtime/config.yaml",
            "providers": "ACP/runtime/providers.yaml",
            "prompts": "ACP/prompts/",
            "workflows": "ACP/workflows/",
            "memory": "ACP/memory/",
            "knowledge": "ACP/knowledge/",
        },
        "adapters": [
            {
                "target_key": "codex-cli",
                "label": "Codex CLI",
                "adapter_doc": "ACP/adapters/codex-cli.md",
                "launch_preference": "terminal_command",
                "command_candidates": ["codex"],
                "recommended_when": [
                    "El equipo quiere ejecutar el ACP por pasos sobre un repositorio local.",
                    "Se requiere trazabilidad de cambios y revision humana antes de aplicar acciones.",
                ],
            },
            {
                "target_key": "claude-code",
                "label": "Claude Code",
                "adapter_doc": "ACP/adapters/claude-code.md",
                "launch_preference": "terminal_command",
                "command_candidates": ["claude"],
                "recommended_when": [
                    "El equipo quiere una sesion agentica interactiva en terminal.",
                    "Se prioriza lectura amplia de artefactos antes de editar codigo.",
                ],
            },
            {
                "target_key": "cursor",
                "label": "Cursor",
                "adapter_doc": "ACP/adapters/cursor.md",
                "launch_preference": "open_workspace",
                "command_candidates": ["cursor"],
                "recommended_when": [
                    "El equipo quiere IDE agentico con contexto del paquete completo.",
                    "Se requiere navegar artefactos, prompts y contratos visualmente.",
                ],
            },
            {
                "target_key": "github-copilot-vscode",
                "label": "GitHub Copilot via VS Code",
                "adapter_doc": "ACP/adapters/github-copilot.md",
                "launch_preference": "open_workspace",
                "command_candidates": ["code"],
                "recommended_when": [
                    "El equipo usa VS Code y Copilot como asistente de implementacion.",
                    "Se requiere aplicar el ACP como referencia estructurada, no como runtime automatico.",
                ],
            },
            {
                "target_key": "openai-agents-sdk",
                "label": "OpenAI Agents SDK",
                "adapter_doc": "ACP/adapters/openai-agents-sdk.md",
                "launch_preference": "implementation_mapping",
                "command_candidates": [],
                "recommended_when": [
                    "El equipo decide materializar el agente en un SDK programatico.",
                    "Se requiere mapear herramientas, handoffs y evaluaciones a codigo productivo.",
                ],
            },
        ],
    }
    neutral_plan = "\n".join(
        [
            "# Framework-neutral build plan",
            "",
            f"Arquitectura objetivo: {blueprint.architecture if blueprint else 'needs_review'}",
            f"Patron de razonamiento: {blueprint.reasoning_pattern if blueprint else 'needs_review'}",
            "",
            "## Orden sugerido",
            "",
            "1. Leer `ACP/manifest.yaml` y `ACP/README.md`.",
            "2. Resolver `ACP/construction-readiness/open-questions.yaml` si existen preguntas bloqueantes.",
            "3. Elegir stack/runtime usando `ACP/runtime/config.yaml`, `ACP/runtime/providers.yaml` y `ACP/deployment/`.",
            "4. Mapear prompts desde `ACP/prompts/` al asistente o framework elegido.",
            "5. Mapear herramientas desde `ACP/tools/` sin asumir proveedores internos.",
            "6. Implementar memoria y knowledge desde `ACP/memory/` y `ACP/knowledge/`.",
            "7. Ejecutar pruebas desde `ACP/evaluation/` antes de promover a produccion.",
        ]
    )
    codex = "\n".join(
        [
            "# Adapter: Codex CLI",
            "",
            "## Uso sugerido",
            "",
            "1. Extrae el ZIP en un workspace limpio.",
            "2. Ejecuta `python ACP/launcher/acp-launcher.py --dry-run --no-open` para generar el reporte.",
            "3. Abre una terminal en la raiz del paquete.",
            "4. Ejecuta `codex` y usa `ACP/prompts/builder-handoff.md` como instruccion inicial.",
            "",
            "## Contexto minimo",
            "",
            "- `ACP/manifest.yaml`",
            "- `ACP/construction-readiness/overview.yaml`",
            "- `ACP/runtime/config.yaml`",
            "- `ACP/workflows/durable-workflow.yaml`",
            "- `ACP/prompts/builder-handoff.md`",
            "",
            "No ejecutes despliegues, migraciones ni integraciones reales sin cerrar primero las preguntas del ACP.",
        ]
    )
    claude = "\n".join(
        [
            "# Adapter: Claude Code",
            "",
            "Usa el ACP como carpeta de especificacion. Carga primero `ACP/README.md`, `ACP/construction-readiness/overview.yaml` y `ACP/prompts/builder-handoff.md`.",
            "",
            "Prioriza resolver preguntas humanas y gaps antes de crear codigo dependiente de entorno.",
        ]
    )
    cursor = "\n".join(
        [
            "# Adapter: Cursor",
            "",
            "Abre el workspace extraido con `cursor .` o desde el launcher si Cursor esta disponible.",
            "",
            "Archivos recomendados para fijar como contexto:",
            "",
            "- `ACP/manifest.yaml`",
            "- `ACP/adapters/framework-neutral-build-plan.md`",
            "- `ACP/prompts/builder-handoff.md`",
            "- `ACP/tools/`",
            "- `ACP/memory/`",
            "- `ACP/knowledge/`",
        ]
    )
    copilot = "\n".join(
        [
            "# Adapter: GitHub Copilot / VS Code",
            "",
            "Abre el workspace con `code .`. Usa Copilot Chat con referencias explicitas a los archivos ACP.",
            "",
            "Prompt inicial sugerido:",
            "",
            "Implementa este sistema agentico siguiendo `ACP/adapters/framework-neutral-build-plan.md`; antes de generar codigo, lista las preguntas pendientes de `ACP/construction-readiness/open-questions.yaml`.",
        ]
    )
    agents_sdk = "\n".join(
        [
            "# Adapter: OpenAI Agents SDK",
            "",
            "Mapea el ACP a un runtime programatico solo despues de seleccionar lenguaje, framework, provider, secretos y estrategia de deployment.",
            "",
            "## Mapeo",
            "",
            "- Agentes: derivar desde arquitectura y prompts en `ACP/prompts/`.",
            "- Tools: implementar contratos desde `ACP/tools/`.",
            "- Memoria/RAG: usar `ACP/memory/` y `ACP/knowledge/` como especificacion.",
            "- Evaluacion: convertir `ACP/evaluation/` en pruebas automatizadas.",
        ]
    )
    return [
        build_acp_file_entry(
            path="ACP/adapters/adapter-registry.json",
            domain="adapters",
            title="Adapter registry",
            format="json",
            source_sections=["runtime", "workflows", "prompts", "memory", "knowledge"],
            content_text=serialize_json_document(registry),
        ),
        build_acp_file_entry(
            path="ACP/adapters/framework-neutral-build-plan.md",
            domain="adapters",
            title="Framework-neutral build plan",
            format="markdown",
            source_sections=["blueprint", "runtime", "construction_readiness"],
            content_text=serialize_markdown_document(neutral_plan),
        ),
        build_acp_file_entry(
            path="ACP/adapters/codex-cli.md",
            domain="adapters",
            title="Codex CLI adapter",
            format="markdown",
            source_sections=["runtime_targets", "prompts"],
            content_text=serialize_markdown_document(codex),
        ),
        build_acp_file_entry(
            path="ACP/adapters/claude-code.md",
            domain="adapters",
            title="Claude Code adapter",
            format="markdown",
            source_sections=["runtime_targets", "prompts"],
            content_text=serialize_markdown_document(claude),
        ),
        build_acp_file_entry(
            path="ACP/adapters/cursor.md",
            domain="adapters",
            title="Cursor adapter",
            format="markdown",
            source_sections=["runtime_targets", "prompts"],
            content_text=serialize_markdown_document(cursor),
        ),
        build_acp_file_entry(
            path="ACP/adapters/github-copilot.md",
            domain="adapters",
            title="GitHub Copilot adapter",
            format="markdown",
            source_sections=["runtime_targets", "prompts"],
            content_text=serialize_markdown_document(copilot),
        ),
        build_acp_file_entry(
            path="ACP/adapters/openai-agents-sdk.md",
            domain="adapters",
            title="OpenAI Agents SDK adapter",
            format="markdown",
            source_sections=["runtime_targets", "tools", "memory", "knowledge"],
            content_text=serialize_markdown_document(agents_sdk),
        ),
    ]


def _build_business_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    if discovery is None or canvas is None:
        return [
            build_acp_file_entry(
                path="ACP/business/lean-canvas.yaml",
                domain="business",
                title="Lean canvas",
                format="yaml",
                source_sections=["discovery", "canvas"],
                missing_fields=["discovery", "canvas"],
            ),
            build_acp_file_entry(
                path="ACP/business/kpis.yaml",
                domain="business",
                title="KPIs",
                format="yaml",
                source_sections=["discovery", "canvas"],
                missing_fields=["discovery", "canvas"],
            ),
            build_acp_file_entry(
                path="ACP/business/constraints.yaml",
                domain="business",
                title="Constraints",
                format="yaml",
                source_sections=["discovery", "canvas"],
                missing_fields=["discovery", "canvas"],
            ),
        ]

    canvas_payload = {
        "problem_statement": discovery.problem_statement,
        "current_user": discovery.current_user,
        "current_process": discovery.current_process,
        "desired_outcome": discovery.desired_outcome,
        "value_statement": discovery.value_statement,
        "mvp_scope": canvas.mvp_scope,
        "out_of_scope": canvas.out_of_scope,
        "primary_risk": canvas.primary_risk,
        "allowed_decisions": canvas.agent_profile.allowed_decisions,
        "prohibited_decisions": canvas.agent_profile.prohibited_decisions,
    }
    kpi_payload = {
        "north_star_metric": discovery.mvp_definition.north_star_metric,
        "success_metrics": canvas.agent_profile.success_metrics or [canvas.success_metric],
        "success_metric": canvas.success_metric,
    }
    constraints_payload = {
        "constraints": discovery.constraints,
        "non_delegable_decisions": discovery.mvp_definition.non_delegable_decisions,
        "human_approvals": canvas.agent_profile.human_approvals,
    }
    return [
        build_acp_file_entry(
            path="ACP/business/lean-canvas.yaml",
            domain="business",
            title="Lean canvas",
            format="yaml",
            source_sections=["discovery", "canvas"],
            content_text=serialize_yaml_document(canvas_payload),
        ),
        build_acp_file_entry(
            path="ACP/business/kpis.yaml",
            domain="business",
            title="KPIs",
            format="yaml",
            source_sections=["discovery.mvp_definition", "canvas"],
            content_text=serialize_yaml_document(kpi_payload),
        ),
        build_acp_file_entry(
            path="ACP/business/constraints.yaml",
            domain="business",
            title="Constraints",
            format="yaml",
            source_sections=["discovery.constraints", "discovery.mvp_definition", "canvas.agent_profile"],
            content_text=serialize_yaml_document(constraints_payload),
        ),
    ]


def _build_architecture_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    discovery = snapshot.discovery
    if blueprint is None:
        return [
            build_acp_file_entry(
                path="ACP/architecture/topology.yaml",
                domain="architecture",
                title="Topology",
                format="yaml",
                source_sections=["blueprint"],
                missing_fields=["blueprint"],
            ),
            build_acp_file_entry(
                path="ACP/architecture/decisions.yaml",
                domain="architecture",
                title="Decisions",
                format="yaml",
                source_sections=["blueprint.delivery_package.decision_trace"],
                missing_fields=["blueprint"],
            ),
            build_acp_file_entry(
                path="ACP/architecture/c4-context.md",
                domain="architecture",
                title="C4 Context",
                format="markdown",
                source_sections=["blueprint"],
                missing_fields=["blueprint"],
            ),
        ]

    topology_payload = {
        "architecture": blueprint.architecture,
        "case_type": discovery.case_type if discovery else "",
        "reasoning_pattern": blueprint.reasoning_pattern,
        "workflow_template": snapshot.selected_workflow_template_key,
        "components": [
            {"name": "llm_core", "role": "reasoning"},
            {"name": "tooling_layer", "role": "actuation"},
            {"name": "memory_layer", "role": "state"},
            {"name": "governance_layer", "role": "controls"},
        ],
    }
    decisions_payload = {
        "decision_summary": blueprint.delivery_package.decision_summary,
        "decision_trace": [item.model_dump(mode="json") for item in blueprint.delivery_package.decision_trace],
        "pattern_catalog": [item.model_dump(mode="json") for item in blueprint.delivery_package.pattern_catalog],
    }
    c4_context = "\n".join(
        [
            f"# C4 Context: {snapshot.session.title}",
            "",
            "> **Diagrama y Especificación de Contexto C4 del Sistema Agéntico**",
            "",
            "## 1. Sistema Agéntico Principal",
            f"- **Nombre del Sistema:** `{snapshot.session.title}`",
            f"- **Misión y Propósito:** {discovery.desired_outcome if discovery and discovery.desired_outcome else 'Automatización agéntica de procesos de negocio.'}",
            f"- **Usuario / Actor Primario:** {discovery.current_user if discovery and discovery.current_user else 'Usuario operativo'}",
            f"- **Narrativa de Diseño:** {blueprint.narrative or 'El sistema agéntico actúa como un copiloto autónomo con supervisión humana por diseño.'}",
            "",
            "## 2. Límites del Sistema y Relaciones Externas",
            "- **Capa de Inferencia (LLM Core):** Motor de razonamiento cognitivo encargado de la interpretación y toma de decisiones.",
            "- **Capa de Herramientas (Actuation Layer):** Adaptadores y contratos de integración con APIs externas y microservicios.",
            "- **Capa de Memoria y Conocimiento (State Layer):** Almacenamiento de sesiones cortas y vector store para recuperación RAG.",
            "- **Capa de Gobernanza (Control Layer):** Guardrails de seguridad, mitigación de alucinaciones y protocolo Human-in-the-Loop.",
            "",
            "## 3. Interfaces de Comunicación",
            f"- **Patrón de Interacción:** `{blueprint.reasoning_pattern}`",
            f"- **Topología de Ejecución:** `{blueprint.architecture}`",
            f"- **Herramientas Registradas:** {len(blueprint.tools)} herramienta(s) con esquemas JSON/OpenAPI tipados.",
        ]
    )
    return [
        build_acp_file_entry(
            path="ACP/architecture/topology.yaml",
            domain="architecture",
            title="Topology",
            format="yaml",
            source_sections=["blueprint.architecture", "discovery.case_type", "selected_workflow_template_key"],
            content_text=serialize_yaml_document(topology_payload),
        ),
        build_acp_file_entry(
            path="ACP/architecture/decisions.yaml",
            domain="architecture",
            title="Decisions",
            format="yaml",
            source_sections=["blueprint.delivery_package.decision_trace", "blueprint.delivery_package.pattern_catalog"],
            content_text=serialize_yaml_document(decisions_payload),
        ),
        build_acp_file_entry(
            path="ACP/architecture/c4-context.md",
            domain="architecture",
            title="C4 Context",
            format="markdown",
            source_sections=["session.title", "discovery", "blueprint.narrative"],
            content_text=serialize_markdown_document(c4_context),
        ),
    ]


def _build_cognition_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    reasoning_payload = {
        "selected_pattern": blueprint.reasoning_pattern,
        "available_patterns": [item.model_dump(mode="json") for item in blueprint.delivery_package.pattern_catalog if item.family == "reasoning"],
        "plan_summary_policy": blueprint.delivery_package.observability_plan.plan_summary_policy,
    }
    planner_payload = {
        "execution_pattern": blueprint.delivery_package.workflow_profile.execution_pattern,
        "steps": [item.model_dump(mode="json") for item in blueprint.delivery_package.workflow_profile.steps],
        "checkpoint_policy": blueprint.delivery_package.workflow_profile.checkpoint_policy,
    }
    reflection_payload = {
        "review_trigger": blueprint.memory_profile.review_trigger,
        "goal_drift_guard": blueprint.memory_profile.goal_drift_guard,
        "decision_logging": blueprint.delivery_package.observability_plan.decision_logging,
    }
    guardrails_payload = {
        "guardrails": blueprint.guardrails,
        "safety_checks": [item.model_dump(mode="json") for item in blueprint.safety_checks],
        "risk_summary": blueprint.delivery_package.risk_summary.model_dump(mode="json"),
    }
    return [
        build_acp_file_entry(
            path="ACP/cognition/reasoning.yaml",
            domain="cognition",
            title="Reasoning",
            format="yaml",
            source_sections=["blueprint.reasoning_pattern", "blueprint.delivery_package.pattern_catalog"],
            content_text=serialize_yaml_document(reasoning_payload),
        ),
        build_acp_file_entry(
            path="ACP/cognition/planner.yaml",
            domain="cognition",
            title="Planner",
            format="yaml",
            source_sections=["blueprint.delivery_package.workflow_profile"],
            content_text=serialize_yaml_document(planner_payload),
        ),
        build_acp_file_entry(
            path="ACP/cognition/reflection.yaml",
            domain="cognition",
            title="Reflection",
            format="yaml",
            source_sections=["blueprint.memory_profile", "blueprint.delivery_package.observability_plan"],
            content_text=serialize_yaml_document(reflection_payload),
        ),
        build_acp_file_entry(
            path="ACP/cognition/guardrails.yaml",
            domain="cognition",
            title="Guardrails",
            format="yaml",
            source_sections=["blueprint.guardrails", "blueprint.safety_checks", "blueprint.delivery_package.risk_summary"],
            content_text=serialize_yaml_document(guardrails_payload),
        ),
    ]


def _build_memory_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    memory_profile = blueprint.memory_profile
    grounding_payload = memory_profile.grounding_policy.model_dump(mode="json")
    strategy_payload = {
        "strategy": memory_profile.strategy or blueprint.memory_strategy,
        "short_term": {
            "enabled": True,
            "type": "session_state",
            "workspace_scope": memory_profile.workspace_scope,
        },
        "long_term": {
            "enabled": bool(memory_profile.storage_layers),
            "layers": memory_profile.storage_layers,
            "agent_scope": memory_profile.agent_scope,
            "retention_policy": memory_profile.retention_policy,
        },
        "goal_drift_control": {
            "enabled": bool(memory_profile.goal_drift_guard),
            "anchor_fields": ["business.objective", "constraints", "agent.mission"],
        },
    }
    retrieval_payload = {
        "retrieval_policy": memory_profile.retrieval_policy,
        "storage_layers": memory_profile.storage_layers,
        "grounding_policy": grounding_payload,
        "sensitivity_rules": memory_profile.sensitivity_rules,
        "workspace_scope": memory_profile.workspace_scope,
        "top_k": 5,
    }
    lifecycle_payload = {
        "write_policy": memory_profile.write_policy,
        "review_trigger": memory_profile.review_trigger,
        "retention_policy": memory_profile.retention_policy,
        "ttl_policy": memory_profile.ttl_policy,
        "workspace_scope": memory_profile.workspace_scope,
        "agent_scope": memory_profile.agent_scope,
        "sensitivity_rules": memory_profile.sensitivity_rules,
        "approval_pause": blueprint.delivery_package.workflow_profile.approval_pause,
    }
    return [
        build_acp_file_entry(
            path="ACP/memory/strategy.yaml",
            domain="memory",
            title="Memory strategy",
            format="yaml",
            source_sections=["blueprint.memory_profile", "blueprint.memory_strategy"],
            content_text=serialize_yaml_document(strategy_payload),
        ),
        build_acp_file_entry(
            path="ACP/memory/retrieval.yaml",
            domain="memory",
            title="Memory retrieval",
            format="yaml",
            source_sections=["blueprint.memory_profile"],
            content_text=serialize_yaml_document(retrieval_payload),
        ),
        build_acp_file_entry(
            path="ACP/memory/lifecycle.yaml",
            domain="memory",
            title="Memory lifecycle",
            format="yaml",
            source_sections=["blueprint.memory_profile", "blueprint.delivery_package.workflow_profile"],
            content_text=serialize_yaml_document(lifecycle_payload),
        ),
    ]


def _build_knowledge_files(
    snapshot: SessionSnapshot,
    continuity_answers: dict[str, str] | None = None,
) -> list[ACPFileEntry]:
    discovery = snapshot.discovery
    blueprint = snapshot.blueprint
    knowledge_profile = blueprint.knowledge_profile if blueprint is not None else None
    current_process = discovery.current_process if discovery else ""
    sources_answer = _continuity_answer_text(continuity_answers, "knowledge_sources")
    ingestion_answer = _continuity_answer_text(continuity_answers, "knowledge_ingestion")
    embeddings_answer = _continuity_answer_text(continuity_answers, "knowledge_embedding_strategy")
    runtime_vector_answer = _continuity_answer_text(continuity_answers, "runtime_vector_store")

    source_entries = _source_entries_from_answer(sources_answer) if sources_answer else []
    ingestion_pairs = _continuity_answer_pairs(
        continuity_answers,
        "knowledge_ingestion",
        aliases={
            "strategy": "strategy",
            "flow": "strategy",
            "frequency": "frequency",
            "owner": "owner",
            "mechanism": "mechanism",
        },
    )
    embeddings_pairs = _continuity_answer_pairs(
        continuity_answers,
        "knowledge_embedding_strategy",
        aliases={
            "provider": "provider",
            "modelo": "provider",
            "model": "provider",
            "chunking": "chunking",
            "chunks": "chunking",
            "policy": "chunking",
            "notes": "notes",
        },
    )
    runtime_vector_pairs = _continuity_answer_pairs(
        continuity_answers,
        "runtime_vector_store",
        aliases={
            "vector_store": "vector_store",
            "vector_db": "vector_store",
            "provider": "vector_store",
            "store": "vector_store",
        },
    )

    knowledge_mode = knowledge_profile.mode if knowledge_profile is not None else ""
    owner_sources = _knowledge_sources_from_owner_entries(source_entries) if source_entries else []
    explicit_sources = []
    if knowledge_profile is not None and knowledge_profile.sources:
        explicit_sources = [
            {
                "description": item.description,
                "key": item.key,
                "lineage_key": f"{(item.key or item.title or 'knowledge-source').strip()}::{item.source_version}",
                "license": item.license,
                "owner": item.owner,
                "sensitivity": item.sensitivity,
                "source_type": item.source_type,
                "source_version": item.source_version,
                "title": item.title,
                "uri": item.uri,
            }
            for item in knowledge_profile.sources
        ]
    owner_sources_are_authoritative = bool(owner_sources)
    source_payload_entries = owner_sources if owner_sources_are_authoritative else explicit_sources
    explicit_lineage = [item["lineage_key"] for item in source_payload_entries if item.get("lineage_key")]

    sources_payload = {
        "known_sources": source_payload_entries,
        "current_process_context": current_process,
        "mode": knowledge_mode or "none",
        "source_lineage": explicit_lineage,
    }

    if knowledge_profile is not None and knowledge_profile.mode == "none":
        disabled_sources_payload = {
            **sources_payload,
            "known_sources": [],
            "notes": knowledge_profile.notes or "Knowledge deshabilitado para este caso.",
        }
        disabled_ingestion_payload = {
            "strategy": "not_required",
            "frequency": "not_required",
            "owner": "not_required",
            "mechanism": "not_required",
            "notes": knowledge_profile.notes or "No hay ingestion porque el caso no usa retrieval documental.",
        }
        disabled_embeddings_payload = {
            "provider": "not_required",
            "vector_store": "not_required",
            "chunking_policy": "not_required",
            "configuration_summary": knowledge_profile.notes or "Sin RAG ni retrieval semantico.",
            "search_mode": "not_required",
            "top_k": 0,
            "reranking_policy": "not_required",
            "fallback_behavior": knowledge_profile.grounding_policy.no_evidence_behavior,
        }
        return [
            build_acp_file_entry(
                path="ACP/knowledge/sources.yaml",
                domain="knowledge",
                title="Knowledge sources",
                format="yaml",
                source_sections=["blueprint.knowledge_profile", "discovery.current_process"],
                content_text=serialize_yaml_document(disabled_sources_payload),
            ),
            build_acp_file_entry(
                path="ACP/knowledge/ingestion.yaml",
                domain="knowledge",
                title="Knowledge ingestion",
                format="yaml",
                source_sections=["blueprint.knowledge_profile", "discovery.current_process"],
                content_text=serialize_yaml_document(disabled_ingestion_payload),
            ),
            build_acp_file_entry(
                path="ACP/knowledge/embeddings.yaml",
                domain="knowledge",
                title="Knowledge embeddings",
                format="yaml",
                source_sections=["blueprint.knowledge_profile", "integration_statuses"],
                content_text=serialize_yaml_document(disabled_embeddings_payload),
            ),
        ]

    ingestion_payload = {
        "strategy": ingestion_pairs.get("strategy")
        or (knowledge_profile.ingestion_policy.parser if knowledge_profile is not None else "")
        or ("captured_from_owner" if ingestion_answer else "needs_review"),
        "frequency": ingestion_pairs.get("frequency") or (knowledge_profile.refresh_policy.frequency if knowledge_profile is not None else ""),
        "owner": ingestion_pairs.get("owner") or (knowledge_profile.sources[0].owner if knowledge_profile is not None and knowledge_profile.sources else ""),
        "mechanism": ingestion_pairs.get("mechanism") or (knowledge_profile.ingestion_policy.chunking_policy if knowledge_profile is not None else ""),
        "notes": ingestion_answer or (knowledge_profile.notes if knowledge_profile is not None else "") or "Definir fuentes, frecuencia y ownership de ingestion.",
    }

    if knowledge_profile is not None and knowledge_profile.mode == "rag":
        embedding_provider = knowledge_profile.embedding_policy.provider
        if embeddings_pairs.get("provider") and _is_placeholder_value(embedding_provider):
            embedding_provider = embeddings_pairs["provider"]
        embedding_dimensions = knowledge_profile.embedding_policy.dimensions
        if embeddings_answer and embedding_dimensions <= 0:
            embedding_dimensions = 1536 if embedding_provider == "text-embedding-3-small" else 1
        embedding_version = knowledge_profile.embedding_policy.version
        if embeddings_answer and _is_placeholder_value(embedding_version):
            embedding_version = "owner-captured"
        chunking_policy = knowledge_profile.ingestion_policy.chunking_policy
        if embeddings_pairs.get("chunking") and _is_placeholder_value(chunking_policy):
            chunking_policy = embeddings_pairs["chunking"]
        vector_store_value = runtime_vector_pairs.get("vector_store", "")
        if runtime_vector_answer and is_no_applicable_answer(runtime_vector_answer):
            vector_store_value = "not_required"
        embeddings_payload = {
            "provider": embedding_provider,
            "vector_store": vector_store_value or ("captured_from_owner" if runtime_vector_answer else "pending_review"),
            "chunking_policy": chunking_policy,
            "dimensions": embedding_dimensions,
            "version": embedding_version,
            "configuration_summary": embeddings_answer or knowledge_profile.notes,
            "search_mode": knowledge_profile.retrieval_policy.search_mode,
            "top_k": knowledge_profile.retrieval_policy.top_k,
            "reranking_policy": knowledge_profile.retrieval_policy.reranking_policy,
            "fallback_behavior": knowledge_profile.retrieval_policy.fallback_behavior,
        }
        sources_warning = "" if source_payload_entries else "Completar fuentes aprobadas antes de construir retrieval real."
        ingestion_warning = "" if knowledge_profile.ingestion_policy.parser and knowledge_profile.ingestion_policy.chunking_policy else "Completar parser y chunking antes de construir retrieval real."
        embeddings_warnings = (
            []
            if (
                embeddings_answer
                or (
                    not _is_placeholder_value(embedding_provider)
                    and embedding_dimensions > 0
                )
            )
            else ["Completar provider, dimensions o version de embeddings antes de construir retrieval real."]
        )
    else:
        embeddings_provider = "needs_review"
        embeddings_chunking = "needs_review"
        embeddings_warning = "No existe modelado explicito de knowledge sources en el builder actual; completar manualmente."
        if embeddings_answer:
            if is_no_applicable_answer(embeddings_answer):
                embeddings_provider = "not_required"
                embeddings_chunking = "not_required"
            else:
                embeddings_provider = embeddings_pairs.get("provider", "captured_from_owner")
                embeddings_chunking = embeddings_pairs.get("chunking", "captured_from_owner")
                if embeddings_pairs.get("provider") and embeddings_pairs.get("chunking"):
                    embeddings_warning = ""

        vector_store_value = runtime_vector_pairs.get("vector_store", "")
        if runtime_vector_answer and is_no_applicable_answer(runtime_vector_answer):
            vector_store_value = "not_required"

        embeddings_payload = {
            "provider": embeddings_provider,
            "vector_store": vector_store_value or ("captured_from_owner" if runtime_vector_answer else "needs_review"),
            "chunking_policy": embeddings_chunking,
            "configuration_summary": embeddings_answer or "",
        }
        sources_warning = "" if source_payload_entries else "No existe modelado explicito de knowledge sources en el builder actual; completar manualmente."
        ingestion_warning = "" if ingestion_answer else "No existe modelado explicito de knowledge sources en el builder actual; completar manualmente."
        embeddings_warnings = [embeddings_warning] if embeddings_warning else []
    return [
        build_acp_file_entry(
            path="ACP/knowledge/sources.yaml",
            domain="knowledge",
            title="Knowledge sources",
            format="yaml",
            source_sections=["blueprint.knowledge_profile", "discovery.current_process"],
            content_text=serialize_yaml_document(sources_payload),
            warnings=[sources_warning] if sources_warning else [],
        ),
        build_acp_file_entry(
            path="ACP/knowledge/ingestion.yaml",
            domain="knowledge",
            title="Knowledge ingestion",
            format="yaml",
            source_sections=["blueprint.knowledge_profile", "discovery.current_process"],
            content_text=serialize_yaml_document(ingestion_payload),
            warnings=[ingestion_warning] if ingestion_warning else [],
        ),
        build_acp_file_entry(
            path="ACP/knowledge/embeddings.yaml",
            domain="knowledge",
            title="Knowledge embeddings",
            format="yaml",
            source_sections=["blueprint.knowledge_profile", "integration_statuses"],
            content_text=serialize_yaml_document(embeddings_payload),
            warnings=embeddings_warnings,
        ),
    ]


def _build_tools_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    permissions_payload = {
        "tools": [
            {
                "name": item.name,
                "requires_approval": item.requires_approval,
                "approval_reason": item.approval_reason,
                "risk_level": item.risk_level,
                "side_effects": item.has_side_effects,
            }
            for item in blueprint.tools
        ]
    }
    files = [
        build_acp_file_entry(
            path="ACP/tools/permissions.yaml",
            domain="tools",
            title="Tool permissions",
            format="yaml",
            source_sections=["blueprint.tools", "approvals", "risk_summary"],
            content_text=serialize_yaml_document(permissions_payload),
        )
    ]
    files.extend(_tool_contract_file(tool, index) for index, tool in enumerate(blueprint.tools, start=1))
    return files


def _build_workflow_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    steps = blueprint.delivery_package.workflow_profile.steps
    state_machine_payload = {
        "execution_pattern": blueprint.delivery_package.workflow_profile.execution_pattern,
        "states": [
            {
                "name": item.name,
                "objective": item.objective,
                "actor": item.actor,
                "requires_approval": item.requires_approval,
                "fallback": item.fallback,
            }
            for item in steps
        ],
    }
    durable_payload = blueprint.delivery_package.workflow_profile.model_dump(mode="json")
    langgraph_payload = {
        "nodes": [{"id": item.name, "type": "workflow_step"} for item in steps],
        "edges": [
            {"source": steps[index].name, "target": steps[index + 1].name}
            for index in range(len(steps) - 1)
        ],
        "metadata": {
            "approval_pause": blueprint.delivery_package.workflow_profile.approval_pause,
            "retry_strategy": blueprint.delivery_package.workflow_profile.retry_strategy,
        },
    }
    return [
        build_acp_file_entry(
            path="ACP/workflows/state-machine.yaml",
            domain="workflows",
            title="State machine",
            format="yaml",
            source_sections=["blueprint.delivery_package.workflow_profile"],
            content_text=serialize_yaml_document(state_machine_payload),
        ),
        build_acp_file_entry(
            path="ACP/workflows/langgraph.json",
            domain="workflows",
            title="LangGraph compatible graph",
            format="json",
            source_sections=["blueprint.delivery_package.workflow_profile"],
            content_text=serialize_json_document(langgraph_payload),
            warnings=["Representacion de interoperabilidad para agentes constructores; revisar antes de ejecucion real."],
        ),
        build_acp_file_entry(
            path="ACP/workflows/durable-workflow.yaml",
            domain="workflows",
            title="Durable workflow",
            format="yaml",
            source_sections=["blueprint.delivery_package.workflow_profile"],
            content_text=serialize_yaml_document(durable_payload),
        ),
    ]


def _build_prompt_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    discovery = snapshot.discovery
    blueprint = snapshot.blueprint
    canvas = snapshot.canvas
    title = snapshot.session.title or "Agent System"

    desired_outcome = discovery.desired_outcome if discovery and discovery.desired_outcome else "Automatizar y optimizar el flujo operativo según el diseño aprobado."
    problem_statement = discovery.problem_statement if discovery and discovery.problem_statement else "Resolver fricciones operativas mediante asistencia agéntica."
    primary_user = (canvas.agent_profile.primary_user if canvas and canvas.agent_profile else None) or (discovery.current_user if discovery else "Usuario operativo")
    current_process = discovery.current_process if discovery and discovery.current_process else "Proceso operativo manual susceptible de automatización."
    architecture = blueprint.architecture if blueprint and blueprint.architecture else "Arquitectura agéntica estructurada"
    reasoning_pattern = blueprint.reasoning_pattern if blueprint and blueprint.reasoning_pattern else "Plan-and-Execute"
    autonomy_level = discovery.autonomy_level if discovery and discovery.autonomy_level else "Supervisada (HITL)"
    
    guardrails_list = blueprint.guardrails if blueprint and blueprint.guardrails else [
        "Validación estricta de esquemas de entrada y salida",
        "Mitigación de alucinaciones con recuperación grounded",
        "Límites de tokens y presupuesto de inferencia por llamada",
        "Supervisión humana obligatoria para acciones con efectos secundarios",
    ]
    guardrails_text = "\n".join([f"- {g}" for g in guardrails_list])

    system_prompt = _find_deliverable(snapshot, "system_prompt")
    if not system_prompt or system_prompt.strip() == "# System Prompt\nPendiente de revision.":
        system_prompt = "\n".join(
            [
                f"# System Prompt: {title}",
                "",
                "## 1. Identidad y Propósito",
                f"Eres un agente de inteligencia artificial especializado en {title}.",
                f"Tu objetivo principal es: {desired_outcome}",
                "",
                "## 2. Directrices de Comportamiento y Operación",
                f"- **Usuario Principal:** {primary_user}",
                f"- **Modelo de Razonamiento:** {reasoning_pattern}",
                f"- **Topología:** {architecture}",
                f"- **Nivel de Autonomía:** {autonomy_level}",
                "",
                "## 3. Guardrails y Políticas de Seguridad",
                guardrails_text,
                "",
                "## 4. Manejo de Errores y Excepciones",
                "- Si la información provista es insuficiente o ambigua, solicita aclaración de manera concisa.",
                "- Si una herramienta falla o devuelve error, registra el fallo y utiliza el mecanismo de fallback sin exponer detalles internos sensibles.",
            ]
        )

    skill_spec = _find_deliverable(snapshot, "skill_spec")
    
    planner_prompt = "\n".join(
        [
            f"# Planner Role Prompt: {title}",
            "",
            "> **Módulo de Planificación Cognitiva y Descomposición de Tareas**",
            "",
            "## 1. Misión del Planificador",
            f"Eres el planificador cognitivo del agente `{title}`. Tu responsabilidad es analizar la solicitud del usuario, descomponerla en una secuencia lógica y estructurada de pasos atómicos, identificar qué herramientas o memorias consultar y definir checkpoints de validación antes de ejecutar.",
            "",
            "## 2. Contexto y Objetivos Aprobados",
            f"- **Objetivo Principal:** {desired_outcome}",
            f"- **Problema Operativo:** {problem_statement}",
            f"- **Arquitectura:** {architecture}",
            f"- **Patrón Cognitivo:** {reasoning_pattern}",
            f"- **Nivel de Autonomía:** {autonomy_level}",
            "",
            "## 3. Reglas de Planificación y Ejecución",
            "1. **Atomicidad:** Cada paso debe tener un único objetivo observable y verificable.",
            "2. **Dependencias:** Modela las dependencias entre pasos para evitar llamadas a herramientas sin los parámetros requeridos.",
            "3. **Idempotencia y Riesgo:** Cualquier acción que altere estado externo debe marcarse explícitamente.",
            "4. **Presupuesto:** Minimiza el consumo de tokens y llamadas redundantes a APIs.",
            "",
            "## 4. Guardrails y Parada (Stop Conditions)",
            guardrails_text,
            "- Si faltan datos críticos para planificar, emite un estado `needs_resolution` indicando el campo faltante.",
        ]
    )

    evaluator_prompt = "\n".join(
        [
            f"# Evaluator Role Prompt: {title}",
            "",
            "> **Módulo de Control de Calidad, Grounding y Auditoría de Seguridad**",
            "",
            "## 1. Misión del Evaluador",
            f"Eres el evaluador de calidad del agente `{title}`. Tu misión es auditar cada resultado generado, verificar el cumplimiento de los contratos de herramientas, comprobar que no existan alucinaciones y validar que se cumplan las políticas de seguridad antes de entregar la respuesta final.",
            "",
            "## 2. Criterios de Evaluación Obligatorios",
            "1. **Completitud:** ¿La respuesta cubre todos los puntos solicitados por el usuario?",
            "2. **Grounding y Evidencia:** ¿Cada dato o afirmación crítica proviene de fuentes autorizadas o resultados de herramientas? (Prohibido alucinar datos).",
            "3. **Seguridad y Guardrails:**",
            guardrails_text,
            "4. **Formato y Esquema:** ¿La salida cumple con la estructura y tipos de datos requeridos?",
            "",
            "## 3. Rúbrica de Decisión",
            "- **PASS:** Cumple el 100% de los criterios y guardrails.",
            "- **FAIL:** Identifica el gap específico y devuelve feedback estructurado para replanificación.",
        ]
    )

    discovery_skill_prompt = "\n".join(
        [
            f"# Discovery Skill: Diagnóstico y Captura de Contexto",
            "",
            "## Propósito",
            "Especialista en extracción estructurada de problemas de negocio, mapeo de procesos y requerimientos operativos.",
            "",
            "## Contexto Operativo",
            f"- **Iniciativa:** {title}",
            f"- **Usuario Objetivo:** {primary_user}",
            f"- **Problema Diagnosticado:** {problem_statement}",
            f"- **Proceso Actual:** {current_process}",
            "",
            "## Directrices de Ejecución",
            "- Estructurar siempre las necesidades en: Hechos observables, Supuestos por validar y Restricciones.",
            "- Identificar métricas cuantitativas de éxito y puntos de fricción.",
        ]
    )

    architecture_skill_prompt = "\n".join(
        [
            f"# Architecture Skill: Diseño y Orquestación Agéntica",
            "",
            "## Propósito",
            "Especialista en modelado de topologías de agentes, máquinas de estado, contratos de interfaces y patrones de razonamiento.",
            "",
            "## Especificación Técnica",
            f"- **Topología Seleccionada:** {architecture}",
            f"- **Patrón Cognitivo:** {reasoning_pattern}",
            f"- **Autonomía:** {autonomy_level}",
            "",
            "## Directrices de Ejecución",
            "- Asegurar que cada agente/herramienta tenga fronteras de aislamiento y contratos de entrada/salida tipados.",
            "- Garantizar que las transiciones de estado sean deterministas y auditables.",
        ]
    )

    evaluation_skill_prompt = "\n".join(
        [
            f"# Evaluation Skill: Testing Automatizado y Aseguramiento de Calidad",
            "",
            "## Propósito",
            "Especialista en ejecución de suites de evaluación, pruebas de mutación y auditoría de conformance para agentes de IA.",
            "",
            "## Directrices de Ejecución",
            "- Validar datasets de prueba con casos de éxito, casos límite (edge cases) y casos de fallo controlado.",
            "- Verificar el cumplimiento de guardrails de seguridad y mitigación de alucinaciones antes del despliegue.",
        ]
    )

    files = [
        build_acp_file_entry(
            path="ACP/prompts/system.md",
            domain="prompts",
            title="System prompt",
            format="markdown",
            source_sections=["delivery_package.deliverables.system_prompt"],
            content_text=serialize_markdown_document(system_prompt),
        ),
        build_acp_file_entry(
            path="ACP/prompts/planner.md",
            domain="prompts",
            title="Planner prompt",
            format="markdown",
            source_sections=["discovery", "blueprint"],
            content_text=serialize_markdown_document(planner_prompt),
        ),
        build_acp_file_entry(
            path="ACP/prompts/evaluator.md",
            domain="prompts",
            title="Evaluator prompt",
            format="markdown",
            source_sections=["blueprint.guardrails", "evaluation_rubric"],
            content_text=serialize_markdown_document(evaluator_prompt),
        ),
        build_acp_file_entry(
            path="ACP/prompts/skills/discovery.md",
            domain="prompts",
            title="Discovery skill prompt",
            format="markdown",
            source_sections=["discovery"],
            content_text=serialize_markdown_document(discovery_skill_prompt),
        ),
        build_acp_file_entry(
            path="ACP/prompts/skills/architecture.md",
            domain="prompts",
            title="Architecture skill prompt",
            format="markdown",
            source_sections=["blueprint.delivery_package.decision_trace"],
            content_text=serialize_markdown_document(architecture_skill_prompt),
        ),
        build_acp_file_entry(
            path="ACP/prompts/skills/evaluation.md",
            domain="prompts",
            title="Evaluation skill prompt",
            format="markdown",
            source_sections=["evaluation_dataset", "evaluation_rubric"],
            content_text=serialize_markdown_document(evaluation_skill_prompt),
        ),
    ]
    if skill_spec:
        files.append(
            build_acp_file_entry(
                path="ACP/prompts/skills/catalog.md",
                domain="prompts",
                title="Skill catalog prompt",
                format="markdown",
                source_sections=["delivery_package.deliverables.skill_spec"],
                content_text=serialize_markdown_document(skill_spec),
            )
        )
    return files


def _suggested_owners(gap: ConstructionGapEntry) -> list[str]:
    owners: list[str] = []
    seen: set[str] = set()
    for question in gap.questions:
        owner = question.target_owner.strip()
        if owner and owner not in seen:
            seen.add(owner)
            owners.append(owner)
    return owners


def _flatten_open_questions(preview: ACPPreview) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for gap in preview.construction_readiness.gaps:
        for question in gap.questions:
            options_list: list[dict[str, Any]] = []
            suggested_answer = ""
            if question.options:
                for opt in question.options:
                    if opt.recommended and not suggested_answer:
                        suggested_answer = opt.label
                    options_list.append(
                        {
                            "key": opt.key,
                            "label": opt.label,
                            "description": opt.description,
                            "impact": opt.impact,
                            "example": opt.example,
                            "recommended": opt.recommended,
                            "confidence": opt.confidence,
                            "source_refs": list(opt.source_refs),
                        }
                    )
            questions.append(
                {
                    "question_key": question.question_key,
                    "gap_key": gap.gap_key,
                    "domain": gap.domain,
                    "question_text": question.question_text,
                    "rationale": question.rationale,
                    "purpose": question.purpose,
                    "suggested_answer": suggested_answer,
                    "target_owner": question.target_owner,
                    "expected_answer_format": question.expected_answer_format,
                    "blocking": question.blocking,
                    "impacted_artifacts": gap.evidence_paths,
                    "options": options_list,
                }
            )
    return questions


def _flatten_assumption_entries(preview: ACPPreview) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gap in preview.construction_readiness.gaps:
        for assumption in gap.current_assumptions:
            normalized = assumption.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            entries.append(
                {
                    "assumption": normalized,
                    "domain": gap.domain,
                    "source_gap_key": gap.gap_key,
                    "safe_temporarily": gap.severity != "blocking",
                    "requires_confirmation": True,
                    "invalidates_production": gap.severity == "blocking",
                    "impacted_artifacts": gap.evidence_paths,
                }
            )
    return entries


def _external_dependency_entries(preview: ACPPreview) -> list[dict[str, Any]]:
    category_by_domain = {
        "integrations": "external_api_contract",
        "deployment": "deployment_environment",
        "runtime": "runtime_or_secrets",
        "knowledge": "knowledge_source",
        "package": "package_validation",
    }
    entries: list[dict[str, Any]] = []
    for gap in preview.construction_readiness.gaps:
        if gap.domain not in category_by_domain:
            continue
        entries.append(
            {
                "dependency_key": gap.gap_key,
                "category": category_by_domain[gap.domain],
                "blocking": gap.severity == "blocking",
                "summary": gap.summary,
                "suggested_owners": _suggested_owners(gap),
                "required_inputs": [question.question_key for question in gap.questions],
                "evidence_paths": gap.evidence_paths,
                "closure_criteria": gap.closure_criteria,
            }
        )
    return entries


def _iter_external_tools(snapshot: SessionSnapshot) -> list[tuple[int, Any]]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    return [
        (index, tool)
        for index, tool in enumerate(blueprint.tools, start=1)
        if tool.name not in INTERNAL_BUILDER_TOOL_NAMES and getattr(tool, "tool_type", "external") != "internal"
    ]


def _build_construction_readiness_files(
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    continuity_answers: dict[str, str] | None = None,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
) -> list[ACPFileEntry]:
    readiness = preview.construction_readiness
    validation = preview.validation
    response_records = response_records or []
    blocking_gaps = [gap for gap in readiness.gaps if gap.severity == "blocking"]
    open_questions = _flatten_open_questions(preview)
    assumptions = _flatten_assumption_entries(preview)
    external_dependencies = _external_dependency_entries(preview)
    decision_log = build_construction_decision_log(preview, response_records)
    deferred_decisions = build_deferred_construction_decision_backlog(preview, response_records)
    impact_outcomes = {
        "answered_count": sum(1 for item in decision_log if item["status"] == "answered"),
        "resolved_count": sum(1 for item in decision_log if item["status"] == "resolved"),
        "deferred_count": len(deferred_decisions),
        "no_material_impact_count": sum(
            1
            for item in decision_log
            if (item.get("impact_analysis") or {}).get("impact_kind") == "no_material_impact"
        ),
        "localized_impact_count": sum(
            1
            for item in decision_log
            if (item.get("impact_analysis") or {}).get("impact_kind") == "localized_impact"
        ),
        "structural_impact_count": sum(
            1
            for item in decision_log
            if (item.get("impact_analysis") or {}).get("impact_kind") == "structural_impact"
        ),
    }
    deployment_questions = [
        question
        for question in open_questions
        if question["domain"] in {"deployment", "runtime"}
    ]
    external_tools = _iter_external_tools(snapshot)
    external_contract_answer = _continuity_answer_text(continuity_answers, "external_api_contracts")
    required_api_contracts: list[dict[str, Any]] = []
    required_api_contracts_warning = ""
    for index, tool in external_tools:
        contract_entry = _find_contract_answer_for_tool(tool.name, external_contract_answer)
        if contract_entry is None:
            required_api_contracts.append(
                {
                    "system_name": tool.name,
                    "purpose": tool.purpose,
                    "required_endpoints_or_actions": ["needs_review"],
                    "expected_authentication": "needs_review",
                    "unknown_payloads": ["request_schema", "response_schema", "error_schema"],
                    "examples_required": True,
                    "impact_if_missing": "Bloquea implementacion y pruebas de integracion reales.",
                    "contract_path": build_tool_contract_path_for_tool(tool, index),
                }
            )
            required_api_contracts_warning = (
                "Persisten tools externas sin contrato operativo suficiente para construccion automatizada."
            )
            continue

        endpoints = [
            value
            for value in [contract_entry.get("endpoint", ""), contract_entry.get("action", "")]
            if value
        ]
        unknown_payloads = [
            field_name
            for field_name, key_name in (
                ("request_schema", "request"),
                ("response_schema", "response"),
                ("error_schema", "errors"),
            )
            if not contract_entry.get(key_name)
        ]
        if not endpoints or not contract_entry.get("auth") or unknown_payloads:
            required_api_contracts_warning = (
                "Persisten tools externas sin contrato operativo suficiente para construccion automatizada."
            )
        required_api_contracts.append(
            {
                "system_name": contract_entry.get("system") or tool.name,
                "tool_name": tool.name,
                "purpose": tool.purpose,
                "required_endpoints_or_actions": endpoints or ["captured_in_owner_answer"],
                "expected_authentication": contract_entry.get("auth", "captured_in_owner_answer"),
                "request_summary": contract_entry.get("request", ""),
                "response_summary": contract_entry.get("response", ""),
                "error_summary": contract_entry.get("errors", ""),
                "unknown_payloads": unknown_payloads,
                "examples_required": bool(unknown_payloads),
                "impact_if_missing": "Bloquea implementacion y pruebas de integracion reales.",
                "contract_path": build_tool_contract_path_for_tool(tool, index),
                "owner_notes": contract_entry.get("notes", ""),
            }
        )

    overview_payload = {
        "package_validation": {
            "overall_status": validation.overall_status,
            "can_export_zip": validation.can_export_zip,
            "completeness_percent": validation.completeness_percent,
        },
        "construction_readiness": {
            "overall_status": readiness.overall_status,
            "can_start_build": readiness.can_start_build,
            "blocking_gaps": readiness.blocking_gaps,
            "open_questions": readiness.open_questions,
            "assumptions_count": readiness.assumptions_count,
            "next_recommended_action": readiness.next_recommended_action,
        },
        "key_paths": {
            "manifest": preview.manifest_path,
            "canonical_env_template": ACP_CANONICAL_ENV_TEMPLATE_PATH,
            "builder_handoff_prompt": "ACP/prompts/builder-handoff.md",
            "gap_closure_prompt": "ACP/prompts/gap-closure.md",
            "question_impact_log": "ACP/construction-readiness/question-impact-log.yaml",
            "deferred_decisions": "ACP/construction-readiness/deferred-decisions.yaml",
        },
        "question_outcomes": impact_outcomes,
    }
    blocking_gaps_payload = {
        "blocking_gaps": [
            {
                "gap_key": gap.gap_key,
                "title": gap.title,
                "domain": gap.domain,
                "blocking_stage": gap.blocking_stage,
                "summary": gap.summary,
                "suggested_owners": _suggested_owners(gap),
                "evidence_paths": gap.evidence_paths,
                "source_sections": gap.source_sections,
                "closure_criteria": gap.closure_criteria,
                "impacted_artifacts": gap.evidence_paths,
            }
            for gap in blocking_gaps
        ]
    }
    open_questions_payload = {"open_questions": open_questions}
    impact_log_payload = {"question_decisions": decision_log}
    deferred_decisions_payload = {"deferred_decisions": deferred_decisions}
    assumptions_payload = {"assumptions": assumptions}
    external_dependencies_payload = {"external_dependencies": external_dependencies}
    required_api_contracts_payload = {"required_api_contracts": required_api_contracts}
    deployment_decisions_payload = {
        "deployment_decisions_needed": [
            {
                "decision_key": question["question_key"],
                "domain": question["domain"],
                "question_text": question["question_text"],
                "rationale": question["rationale"],
                "target_owner": question["target_owner"],
                "expected_answer_format": question["expected_answer_format"],
                "blocking": question["blocking"],
                "impacted_artifacts": question["impacted_artifacts"],
                "options": question.get("options", []),
            }
            for question in deployment_questions
        ]
    }
    resolution_workflow_payload = {
        "steps": [
            {"order": 1, "action": "read_overview", "path": "ACP/construction-readiness/overview.yaml"},
            {"order": 2, "action": "review_blocking_gaps", "path": "ACP/construction-readiness/blocking-gaps.yaml"},
            {"order": 3, "action": "ask_open_questions", "path": "ACP/construction-readiness/open-questions.yaml"},
            {"order": 4, "action": "review_answer_impact", "path": "ACP/construction-readiness/question-impact-log.yaml"},
            {"order": 5, "action": "register_answers", "path": ACP_CANONICAL_ENV_TEMPLATE_PATH},
            {"order": 6, "action": "review_deferred_decisions", "path": "ACP/construction-readiness/deferred-decisions.yaml"},
            {"order": 7, "action": "recalculate_readiness", "path": "ACP/construction-readiness/overview.yaml"},
            {"order": 8, "action": "continue_to_implementation", "condition": "only_if_can_start_build_true"},
        ]
    }

    return [
        build_acp_file_entry(
            path="ACP/construction-readiness/overview.yaml",
            domain="construction-readiness",
            title="Construction readiness overview",
            format="yaml",
            source_sections=["construction_readiness", "validation"],
            content_text=serialize_yaml_document(overview_payload),
        ),
        build_acp_file_entry(
            path="ACP/construction-readiness/blocking-gaps.yaml",
            domain="construction-readiness",
            title="Blocking gaps",
            format="yaml",
            source_sections=["construction_readiness.gaps"],
            content_text=serialize_yaml_document(blocking_gaps_payload),
        ),
        build_acp_file_entry(
            path="ACP/construction-readiness/open-questions.yaml",
            domain="construction-readiness",
            title="Open questions",
            format="yaml",
            source_sections=["construction_readiness.gaps.questions"],
            content_text=serialize_yaml_document(open_questions_payload),
        ),
        build_acp_file_entry(
            path="ACP/construction-readiness/question-impact-log.yaml",
            domain="construction-readiness",
            title="Question impact log",
            format="yaml",
            source_sections=["construction_readiness.gaps.questions"],
            content_text=serialize_yaml_document(impact_log_payload),
        ),
        build_acp_file_entry(
            path="ACP/construction-readiness/deferred-decisions.yaml",
            domain="construction-readiness",
            title="Deferred decisions",
            format="yaml",
            source_sections=["construction_readiness.gaps.questions"],
            content_text=serialize_yaml_document(deferred_decisions_payload),
        ),
        build_acp_file_entry(
            path="ACP/construction-readiness/assumptions.yaml",
            domain="construction-readiness",
            title="Assumptions",
            format="yaml",
            source_sections=["construction_readiness.gaps.current_assumptions"],
            content_text=serialize_yaml_document(assumptions_payload),
        ),
        build_acp_file_entry(
            path="ACP/construction-readiness/external-dependencies.yaml",
            domain="construction-readiness",
            title="External dependencies",
            format="yaml",
            source_sections=["construction_readiness.gaps", "integration_statuses"],
            content_text=serialize_yaml_document(external_dependencies_payload),
        ),
        build_acp_file_entry(
            path="ACP/construction-readiness/required-api-contracts.yaml",
            domain="construction-readiness",
            title="Required API contracts",
            format="yaml",
            source_sections=["blueprint.tools", "construction_readiness.gaps"],
            content_text=serialize_yaml_document(required_api_contracts_payload),
            warnings=[required_api_contracts_warning] if required_api_contracts_warning else [],
        ),
        build_acp_file_entry(
            path="ACP/construction-readiness/deployment-decisions-needed.yaml",
            domain="construction-readiness",
            title="Deployment decisions needed",
            format="yaml",
            source_sections=["construction_readiness.gaps.questions"],
            content_text=serialize_yaml_document(deployment_decisions_payload),
        ),
        build_acp_file_entry(
            path="ACP/construction-readiness/resolution-workflow.yaml",
            domain="construction-readiness",
            title="Resolution workflow",
            format="yaml",
            source_sections=["construction_readiness", "validation"],
            content_text=serialize_yaml_document(resolution_workflow_payload),
        ),
    ]


def _build_continuity_prompt_files(preview: ACPPreview) -> list[ACPFileEntry]:
    readiness = preview.construction_readiness
    builder_handoff = "\n".join(
        [
            "# Builder Handoff",
            "",
            "Continua la construccion del agente usando este ACP sin inventar datos criticos del entorno.",
            "",
            "## Estado actual",
            f"- package_validation: {preview.validation.overall_status}",
            f"- construction_readiness: {readiness.overall_status}",
            f"- blocking_gaps: {readiness.blocking_gaps}",
            f"- open_questions: {readiness.open_questions}",
            f"- can_start_build: {str(readiness.can_start_build).lower()}",
            "",
            "## Reglas obligatorias",
            "- Lee primero `ACP/construction-readiness/overview.yaml`.",
            "- Usa `ACP/blueprint.graph.json` y `ACP/diagrams/Architecture.md` como mapa vivo antes de tocar runtime, tools o deployment.",
            "- Revisa `blocking-gaps.yaml`, `open-questions.yaml` y `deployment-decisions-needed.yaml` antes de construir.",
            "- No asumas detalles de deployment, secretos ni contratos API externos cuando aparezcan como gaps abiertos.",
            "- Manten trazabilidad entre `gap_key`, `question_key`, respuesta recibida y artefactos ACP impactados.",
            "- Reanuda construccion solo despues de cerrar gaps bloqueantes y recalcular readiness.",
        ]
    )
    gap_closure = "\n".join(
        [
            "# Gap Closure",
            "",
            "Usa este modo operativo para cerrar vacios del ACP sin alucinar.",
            "",
            "## Flujo",
            "1. Identifica el `gap_key` y revisa su evidencia.",
            "2. Emite una pregunta concreta y una sola decision por vez.",
            "3. Propone el formato esperado de respuesta antes de continuar.",
            "4. Registra la evidencia recibida y los archivos ACP impactados.",
            "5. Solicita confirmacion si la respuesta cambia runtime, deployment o integraciones externas.",
            "6. Actualiza el estado del gap y recalcula readiness.",
            "",
            "## Salida minima",
            "```yaml",
            "gap_key: <id>",
            "question_key: <id>",
            "answer_summary: <texto breve>",
            "evidence_source: <owner o documento>",
            "affected_files:",
            "  - ACP/...",
            "status_after_update: <open|resolved>",
            "```",
        ]
    )
    return [
        build_acp_file_entry(
            path="ACP/prompts/builder-handoff.md",
            domain="prompts",
            title="Builder handoff prompt",
            format="markdown",
            source_sections=["construction_readiness", "validation"],
            content_text=serialize_markdown_document(builder_handoff),
        ),
        build_acp_file_entry(
            path="ACP/prompts/gap-closure.md",
            domain="prompts",
            title="Gap closure prompt",
            format="markdown",
            source_sections=["construction_readiness.gaps.questions"],
            content_text=serialize_markdown_document(gap_closure),
        ),
    ]


def _build_runtime_files(
    snapshot: SessionSnapshot,
    continuity_answers: dict[str, str] | None = None,
) -> list[ACPFileEntry]:
    runtime = _runtime_defaults(snapshot)
    integration_statuses = [item.model_dump(mode="json") for item in snapshot.integration_statuses]
    fallback_answer = _continuity_answer_text(continuity_answers, "runtime_fallback_model")
    fallback_pairs = _continuity_answer_pairs(
        continuity_answers,
        "runtime_fallback_model",
        aliases={
            "model": "model",
            "fallback_model": "model",
            "condition": "condition",
            "rule": "condition",
            "trigger": "condition",
        },
    )
    vector_store_answer = _continuity_answer_text(continuity_answers, "runtime_vector_store")
    vector_store_pairs = _continuity_answer_pairs(
        continuity_answers,
        "runtime_vector_store",
        aliases={
            "vector_store": "vector_store",
            "vector_db": "vector_store",
            "provider": "vector_store",
            "store": "vector_store",
            "notes": "notes",
        },
    )
    secret_answer = _continuity_answer_text(continuity_answers, "runtime_secret_source")
    secret_pairs = _continuity_answer_pairs(
        continuity_answers,
        "runtime_secret_source",
        aliases={
            "source": "source",
            "owner": "owner",
            "environment": "environment",
            "env": "environment",
            "notes": "notes",
        },
    )

    config_payload = {
        "framework": runtime["framework"],
        "execution_mode": "local-first",
        "provider": runtime["llm_provider"],
        "integrations": integration_statuses,
    }
    fallback_value = "needs_review"
    fallback_policy = ""
    if fallback_answer:
        if is_no_applicable_answer(fallback_answer):
            fallback_value = "not_required"
            fallback_policy = "No se requiere fallback segun el owner."
        else:
            fallback_value = fallback_pairs.get("model", "captured_from_owner")
            fallback_policy = fallback_pairs.get("condition", fallback_answer)
    models_payload = {
        "primary_model": runtime["model"],
        "fallback_model": fallback_value,
        "fallback_policy": fallback_policy,
    }
    secret_source = secret_pairs.get("source", "")
    if secret_answer and is_no_applicable_answer(secret_answer):
        secret_source = "not_required"
    vector_store_value = vector_store_pairs.get("vector_store", "")
    if vector_store_answer and is_no_applicable_answer(vector_store_answer):
        vector_store_value = "not_required"
    providers_payload = {
        "llm_provider": runtime["llm_provider"],
        "database": "postgresql",
        "auth": "local_auth",
        "vector_store": vector_store_value or ("captured_from_owner" if vector_store_answer else runtime["vector_db"]),
        "secret_source": secret_source or ("captured_from_owner" if secret_answer else "needs_review"),
        "secret_owner": secret_pairs.get("owner", ""),
        "target_environment": secret_pairs.get("environment", ""),
    }
    warnings: list[str] = []
    if providers_payload["vector_store"] in {"needs_review", "captured_from_owner"}:
        warnings.append("El proveedor de vector DB no esta modelado en el builder actual.")
    if providers_payload["secret_source"] in {"needs_review", "captured_from_owner"}:
        warnings.append("La fuente de secretos del runtime todavia requiere mayor precision operativa.")
    return [
        build_acp_file_entry(
            path="ACP/runtime/config.yaml",
            domain="runtime",
            title="Runtime config",
            format="yaml",
            source_sections=["integration_statuses"],
            content_text=serialize_yaml_document(config_payload),
        ),
        build_acp_file_entry(
            path="ACP/runtime/models.yaml",
            domain="runtime",
            title="Runtime models",
            format="yaml",
            source_sections=["integration_statuses"],
            content_text=serialize_yaml_document(models_payload),
            warnings=[] if fallback_value not in {"needs_review", "captured_from_owner"} else warnings[:1],
        ),
        build_acp_file_entry(
            path="ACP/runtime/providers.yaml",
            domain="runtime",
            title="Runtime providers",
            format="yaml",
            source_sections=["integration_statuses"],
            content_text=serialize_yaml_document(providers_payload),
            warnings=warnings,
        ),
    ]


def _build_evaluation_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    dataset = snapshot.evaluation_dataset
    rubric = snapshot.evaluation_rubric
    if dataset is None or rubric is None:
        return [
            build_acp_file_entry(
                path="ACP/evaluation/golden-dataset.json",
                domain="evaluation",
                title="Golden dataset",
                format="json",
                source_sections=["evaluation_dataset"],
                missing_fields=["evaluation_dataset"],
            ),
            build_acp_file_entry(
                path="ACP/evaluation/rubrics.yaml",
                domain="evaluation",
                title="Rubrics",
                format="yaml",
                source_sections=["evaluation_rubric"],
                missing_fields=["evaluation_rubric"],
            ),
            build_acp_file_entry(
                path="ACP/evaluation/benchmarks.yaml",
                domain="evaluation",
                title="Benchmarks",
                format="yaml",
                source_sections=["evaluation_runs"],
                warnings=["No existen corridas persistidas; benchmark inicial requiere revision."],
                content_text=serialize_yaml_document({"benchmarks": []}),
            ),
            build_acp_file_entry(
                path="ACP/evaluation/test-cases.feature",
                domain="evaluation",
                title="Test cases",
                format="gherkin",
                source_sections=["evaluation_dataset"],
                missing_fields=["evaluation_dataset"],
            ),
        ]

    benchmarks_payload = {
        "latest_runs": [item.model_dump(mode="json") for item in snapshot.evaluation_runs[:3]],
        "expected_min_score": 70,
    }
    gherkin_lines = ["Feature: ACP validation package", ""]
    for item in dataset.cases[:8]:
        gherkin_lines.extend(
            [
                f"Scenario: {item.title}",
                f"  Given el agente recibe el contexto '{item.scenario}'",
                f"  Then el resultado esperado es '{item.expected_result}'",
                "",
            ]
        )
    return [
        build_acp_file_entry(
            path="ACP/evaluation/golden-dataset.json",
            domain="evaluation",
            title="Golden dataset",
            format="json",
            source_sections=["evaluation_dataset"],
            content_text=serialize_json_document(dataset.model_dump(mode="json")),
        ),
        build_acp_file_entry(
            path="ACP/evaluation/rubrics.yaml",
            domain="evaluation",
            title="Rubrics",
            format="yaml",
            source_sections=["evaluation_rubric"],
            content_text=serialize_yaml_document(rubric.model_dump(mode="json")),
        ),
        build_acp_file_entry(
            path="ACP/evaluation/benchmarks.yaml",
            domain="evaluation",
            title="Benchmarks",
            format="yaml",
            source_sections=["evaluation_runs"],
            content_text=serialize_yaml_document(benchmarks_payload),
            warnings=[] if snapshot.evaluation_runs else ["No existen corridas persistidas; benchmark inicial requiere revision."],
        ),
        build_acp_file_entry(
            path="ACP/evaluation/test-cases.feature",
            domain="evaluation",
            title="Test cases",
            format="gherkin",
            source_sections=["evaluation_dataset"],
            content_text=serialize_markdown_document("\n".join(gherkin_lines)),
        ),
    ]


def _build_deployment_files(
    snapshot: SessionSnapshot,
    continuity_answers: dict[str, str] | None = None,
) -> list[ACPFileEntry]:
    target_answer = _continuity_answer_text(continuity_answers, "deployment_target")
    target_pairs = _continuity_answer_pairs(
        continuity_answers,
        "deployment_target",
        aliases={
            "target": "target",
            "environment": "target",
            "restrictions": "restrictions",
            "constraints": "restrictions",
        },
    )
    image_answer = _continuity_answer_text(continuity_answers, "deployment_image_strategy")
    image_pairs = _continuity_answer_pairs(
        continuity_answers,
        "deployment_image_strategy",
        aliases={
            "strategy": "strategy",
            "build": "strategy",
            "mode": "strategy",
            "image": "image",
            "registry": "registry",
        },
    )
    network_answer = _continuity_answer_text(continuity_answers, "deployment_network_constraints")
    network_pairs = _continuity_answer_pairs(
        continuity_answers,
        "deployment_network_constraints",
        aliases={
            "network": "network",
            "constraints": "network",
            "secrets": "secrets",
            "dependencies": "dependencies",
            "notes": "notes",
        },
    )

    env_template = "\n".join(
        [
            "OPENAI_API_KEY=",
            "DATABASE_URL=",
            "APP_ENV=development",
            f"# deployment_target={target_pairs.get('target', '')}",
            f"# secrets_source={network_pairs.get('secrets', '')}",
            "",
        ]
    )
    agent_service: dict[str, Any] = {
        "environment": ["OPENAI_API_KEY", "DATABASE_URL"],
        "ports": ["8000:8000"],
    }
    image_name = image_pairs.get("image", "")
    strategy_value = image_pairs.get("strategy", "")
    if image_name:
        agent_service["image"] = image_name
    elif strategy_value and any(token in strategy_value.lower() for token in ["docker", "contenedor", "container", "compose"]):
        agent_service["build"] = {"context": ".", "dockerfile": "Dockerfile"}
    else:
        agent_service["delivery_strategy"] = strategy_value or "needs_review"

    payload = {
        "services": {
            "agent-app": agent_service,
            "database": {
                "image": "postgres:16",
                "ports": ["5432:5432"],
            },
        },
        "deployment_target": target_pairs.get("target", target_answer),
        "deployment_restrictions": target_pairs.get("restrictions", ""),
        "network_constraints": network_pairs.get("network", network_answer),
        "secret_constraints": network_pairs.get("secrets", ""),
        "dependencies": network_pairs.get("dependencies", ""),
    }
    deployment_warning = ""
    if not target_answer or not image_answer or not network_answer:
        deployment_warning = "Los artefactos de deployment son plantillas para agentes constructores y requieren ajuste humano."
    elif "needs_review" in serialize_yaml_document(payload):
        deployment_warning = "Persisten campos de deployment que requieren mayor precision antes de construir."

    kubernetes_readme = "\n".join(
        [
            "# Kubernetes",
            "",
            f"- target_capturado: {target_pairs.get('target', target_answer) or 'sin_definir'}",
            f"- estrategia_imagen: {image_pairs.get('strategy', image_answer) or 'sin_definir'}",
            f"- restricciones_red: {network_pairs.get('network', network_answer) or 'sin_definir'}",
        ]
    )
    cicd_readme = "\n".join(
        [
            "# CI/CD",
            "",
            f"- delivery_strategy: {image_pairs.get('strategy', image_answer) or 'sin_definir'}",
            f"- registry: {image_pairs.get('registry', '') or 'sin_definir'}",
            f"- notas_operativas: {network_pairs.get('notes', network_answer) or 'sin_definir'}",
        ]
    )
    return [
        build_acp_file_entry(
            path="ACP/deployment/docker-compose.yaml",
            domain="deployment",
            title="Docker Compose",
            format="yaml",
            source_sections=["integration_statuses", "runtime"],
            content_text=serialize_yaml_document(payload),
            warnings=[deployment_warning] if deployment_warning else [],
        ),
        build_acp_file_entry(
            path=ACP_CANONICAL_ENV_TEMPLATE_PATH,
            domain="deployment",
            title="Environment template",
            format="dotenv",
            source_sections=["integration_statuses"],
            content_text=serialize_markdown_document(env_template),
            warnings=[deployment_warning] if deployment_warning else [],
        ),
        build_acp_file_entry(
            path="ACP/deployment/kubernetes/README.md",
            domain="deployment",
            title="Kubernetes placeholder",
            format="markdown",
            source_sections=["integration_statuses"],
            content_text=serialize_markdown_document(kubernetes_readme),
            warnings=[deployment_warning] if deployment_warning else [],
        ),
        build_acp_file_entry(
            path="ACP/deployment/cicd/README.md",
            domain="deployment",
            title="CI/CD placeholder",
            format="markdown",
            source_sections=["integration_statuses"],
            content_text=serialize_markdown_document(cicd_readme),
            warnings=[deployment_warning] if deployment_warning else [],
        ),
    ]


def _build_observability_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    observability_plan = blueprint.delivery_package.observability_plan if blueprint else None
    telemetry_payload = {
        "captured_signals": observability_plan.captured_signals if observability_plan else [],
        "plan_summary_policy": observability_plan.plan_summary_policy if observability_plan else "",
    }
    tracing_payload = {
        "tool_response_logging": observability_plan.tool_response_logging if observability_plan else "",
        "decision_logging": observability_plan.decision_logging if observability_plan else "",
        "result_tracking": observability_plan.result_tracking if observability_plan else "",
    }
    metrics_payload = {
        "latest_metric": snapshot.metric_snapshots[0].model_dump(mode="json") if snapshot.metric_snapshots else {},
        "cost_tracking": observability_plan.cost_tracking if observability_plan else "",
        "duration_tracking": observability_plan.duration_tracking if observability_plan else "",
    }
    alerts_payload = {
        "configured_triggers": observability_plan.alert_triggers if observability_plan else [],
        "active_alerts": [item.model_dump(mode="json") for item in snapshot.alert_events],
    }
    return [
        build_acp_file_entry(
            path="ACP/observability/telemetry.yaml",
            domain="observability",
            title="Telemetry",
            format="yaml",
            source_sections=["blueprint.delivery_package.observability_plan"],
            content_text=serialize_yaml_document(telemetry_payload),
        ),
        build_acp_file_entry(
            path="ACP/observability/tracing.yaml",
            domain="observability",
            title="Tracing",
            format="yaml",
            source_sections=["blueprint.delivery_package.observability_plan"],
            content_text=serialize_yaml_document(tracing_payload),
        ),
        build_acp_file_entry(
            path="ACP/observability/metrics.yaml",
            domain="observability",
            title="Metrics",
            format="yaml",
            source_sections=["metric_snapshots", "blueprint.delivery_package.observability_plan"],
            content_text=serialize_yaml_document(metrics_payload),
        ),
        build_acp_file_entry(
            path="ACP/observability/alerts.yaml",
            domain="observability",
            title="Alerts",
            format="yaml",
            source_sections=["alert_events", "blueprint.delivery_package.observability_plan"],
            content_text=serialize_yaml_document(alerts_payload),
        ),
    ]


def _build_governance_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    report = ensure_blueprint_consistency_report(snapshot)
    lineage_payload = {
        "generated_from_blueprint_version": report.generated_from_blueprint_version,
        "overall_status": str(report.overall_status),
        "approved_stage_lineage": [item.model_dump(mode="json") for item in report.approved_stage_lineage],
        "exportable_lineage": report.exportable_lineage,
        "restricted_lineage": report.restricted_lineage,
    }
    decisions_payload = {
        "summary": report.summary,
        "decision_history": report.decision_history,
    }
    consistency_payload = report.model_dump(mode="json")
    return [
        build_acp_file_entry(
            path="ACP/governance/consistency-report.json",
            domain="governance",
            title="Consistency report",
            format="json",
            source_sections=["blueprint_consistency", "journey_artifacts", "estimation_report"],
            content_text=serialize_json_document(consistency_payload),
            warnings=report.warnings[:4],
        ),
        build_acp_file_entry(
            path="ACP/governance/consistency-report.md",
            domain="governance",
            title="Consistency report",
            format="markdown",
            source_sections=["blueprint_consistency", "journey_artifacts", "estimation_report"],
            content_text=serialize_markdown_document(render_blueprint_consistency_markdown(report)),
            warnings=report.warnings[:4],
        ),
        build_acp_file_entry(
            path="ACP/governance/approved-stage-lineage.yaml",
            domain="governance",
            title="Approved stage lineage",
            format="yaml",
            source_sections=["blueprint_consistency", "journey_artifacts"],
            content_text=serialize_yaml_document(lineage_payload),
        ),
        build_acp_file_entry(
            path="ACP/governance/journey-decisions.json",
            domain="governance",
            title="Journey decisions",
            format="json",
            source_sections=["blueprint_consistency", "journey_artifacts"],
            content_text=serialize_json_document(decisions_payload),
        ),
    ]


def generate_acp_files(
    snapshot: SessionSnapshot,
    continuity_answers: dict[str, str] | None = None,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
    extra_readiness_gaps: list[ConstructionGapEntry] | None = None,
) -> list[ACPFileEntry]:
    files: list[ACPFileEntry] = []
    files.append(_build_manifest_file(snapshot))
    files.append(_build_readme_file(snapshot))
    files.extend(_build_deliverable_catalog_files(snapshot))
    files.extend(_build_launcher_files(snapshot))
    files.extend(_build_adapter_files(snapshot))
    files.extend(_build_business_files(snapshot))
    files.extend(_build_architecture_files(snapshot))
    files.extend(_build_cognition_files(snapshot))
    files.extend(_build_memory_files(snapshot))
    files.extend(_build_knowledge_files(snapshot, continuity_answers))
    files.extend(_build_tools_files(snapshot))
    files.extend(_build_workflow_files(snapshot))
    files.extend(_build_prompt_files(snapshot))
    files.extend(_build_runtime_files(snapshot, continuity_answers))
    files.extend(_build_evaluation_files(snapshot))
    files.extend(_build_deployment_files(snapshot, continuity_answers))
    files.extend(_build_observability_files(snapshot))
    files.extend(_build_governance_files(snapshot))
    files.extend(_build_estimation_files(snapshot))
    base_files = sorted(files, key=lambda item: item.path)
    base_preview = build_acp_preview(snapshot, base_files)
    base_preview = append_construction_readiness_gaps(base_preview, extra_readiness_gaps)
    continuity_files = _build_construction_readiness_files(
        snapshot,
        base_preview,
        continuity_answers,
        response_records,
    )
    continuity_files.extend(_build_continuity_prompt_files(base_preview))
    acp_without_diagrams = sorted(base_files + continuity_files, key=lambda item: item.path)
    visualization_files = build_acp_visualization_files(snapshot, acp_without_diagrams)
    acp_without_conformance = sorted(acp_without_diagrams + visualization_files, key=lambda item: item.path)
    conformance_preview = build_acp_preview(snapshot, acp_without_conformance)
    conformance_preview = append_construction_readiness_gaps(conformance_preview, extra_readiness_gaps)
    conformance_files = build_acp_conformance_files(
        conformance_preview,
        acp_without_conformance,
        profile="acp-full",
    )
    return sorted(acp_without_conformance + conformance_files, key=lambda item: item.path)


def generate_acp_preview(
    snapshot: SessionSnapshot,
    continuity_answers: dict[str, str] | None = None,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
    extra_readiness_gaps: list[ConstructionGapEntry] | None = None,
) -> ACPPreview:
    preview = build_acp_preview(
        snapshot,
        generate_acp_files(snapshot, continuity_answers, response_records, extra_readiness_gaps),
    )
    return append_construction_readiness_gaps(preview, extra_readiness_gaps)
