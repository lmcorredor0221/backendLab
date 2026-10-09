from __future__ import annotations

from html import escape
from types import SimpleNamespace
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
    build_construction_question_views,
    is_no_applicable_answer,
    overlay_construction_readiness,
    parse_answer_list,
    parse_answer_pairs,
    parse_contract_answer_entries,
)
from app.services.blueprint_consistency_service import (
    ensure_blueprint_consistency_report,
    render_blueprint_consistency_markdown,
)
from app.services.acp_construction_readiness import (
    INTERNAL_BUILDER_TOOL_NAMES,
    is_blueprint_handoff_process_debt_issue,
)
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
from app.services.acp_prompt_synthesis import (
    ACPPromptSectionSynthesizer,
    PromptSectionSynthesisRejected,
    build_prompt_section_synthesis_request,
    validate_prompt_section_synthesis,
)
from app.services.acp_visualization import build_acp_visualization_files
from app.services.acp_validation import build_acp_file_entry, build_acp_preview
from app.services.deliverable_catalog.project_generation_context import ProjectGenerationContext
from app.services.deliverable_catalog.registry_service import list_registry_entries
from app.services.objective_contracts import active_objective, build_objective_contract_bundle, objective_requires_runtime_loop
from app.services.tool_family_projection import (
    odoo_connector_keys_for_snapshot as projected_odoo_connector_keys_for_snapshot,
    project_blueprint_tools_for_construction,
    resolve_google_workspace_connector_key,
    resolve_odoo_connector_key,
    resolve_whatsapp_connector_key,
    snapshot_has_odoo_quote_signal,
)


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


def _build_acp_generation_context(
    snapshot: SessionSnapshot,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
    extra_readiness_gaps: list[ConstructionGapEntry] | None = None,
) -> ProjectGenerationContext:
    raw_snapshot = snapshot.model_dump(mode="json")
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    blueprint = snapshot.blueprint
    knowledge_profile = blueprint.knowledge_profile if blueprint is not None else None
    snapshot_payload: dict[str, Any] = {
        "raw_snapshot": raw_snapshot,
        "session_id": str(snapshot.session.id),
        "workspace_id": str(getattr(snapshot.session, "workspace_id", "") or ""),
        "project_title": snapshot.session.title,
        "problem_statement": discovery.problem_statement if discovery else "",
        "current_process": discovery.current_process if discovery else "",
        "current_user": discovery.current_user if discovery else "",
        "desired_outcome": discovery.desired_outcome if discovery else "",
        "objectives": [canvas.user_goal] if canvas and canvas.user_goal else [],
        "mvp_scope": canvas.mvp_scope if canvas else [],
        "out_of_scope": canvas.out_of_scope if canvas else [],
        "constraints": discovery.constraints if discovery else [],
        "nondelegable_decisions": discovery.mvp_definition.non_delegable_decisions if discovery else [],
        "architecture": blueprint.architecture if blueprint else "",
        "reasoning_pattern": blueprint.reasoning_pattern if blueprint else "",
        "tools": [tool.model_dump(mode="json") for tool in blueprint.tools] if blueprint else [],
        "memory_strategy": blueprint.memory_strategy if blueprint else "",
        "rag_required": bool(knowledge_profile and knowledge_profile.mode == "rag"),
        "knowledge_sources": [source.model_dump(mode="json") for source in knowledge_profile.sources] if knowledge_profile else [],
        "guardrails": blueprint.guardrails if blueprint else [],
        "risks": [canvas.primary_risk] if canvas and canvas.primary_risk else [],
        "acceptance_criteria": [canvas.success_metric] if canvas and canvas.success_metric else [],
    }
    questions = [item.model_dump(mode="json") for item in response_records or []]
    artifacts = {
        "extra_readiness_gaps": [item.model_dump(mode="json") for item in extra_readiness_gaps or []],
        "journey_latest_artifacts": raw_snapshot.get("journey_latest_artifacts", {}),
    }
    return ProjectGenerationContext.from_acp_inputs(
        snapshot_payload,
        questions,
        artifacts,
        deliverable_key="acp.preview",
    )


def _context_source_refs(context: ProjectGenerationContext | None) -> list[str]:
    if context is None:
        return []
    return [source.ref for source in context.source_refs[:12]]


def _context_anchor_values(context: ProjectGenerationContext | None) -> list[str]:
    if context is None:
        return []
    return [anchor.value for anchor in context.specificity_anchors[:8] if anchor.value]


def _context_trace_payload(context: ProjectGenerationContext | None) -> dict[str, Any]:
    if context is None:
        return {}
    return {
        "context_version": context.context_version,
        "context_fingerprint": context.input_fingerprint,
        "context_source_refs": _context_source_refs(context),
        "specificity_anchors": _context_anchor_values(context),
    }


def _tool_binding_category(tool: Any) -> str:
    registered_ref = str(getattr(tool, "registered_api_ref", "") or "").strip()
    if registered_ref and not _is_placeholder_value(registered_ref):
        return "design_contract"
    if getattr(tool, "inputs", None) or getattr(tool, "outputs", None) or getattr(tool, "request_schema", None) or getattr(tool, "response_schema", None):
        return "design_contract"
    return "pending_binding"


def _retrieval_design_category(snapshot: SessionSnapshot, context: ProjectGenerationContext | None) -> str:
    blueprint = snapshot.blueprint
    knowledge_profile = blueprint.knowledge_profile if blueprint is not None else None
    source_count = len(knowledge_profile.sources) if knowledge_profile is not None else 0
    mode = str(getattr(knowledge_profile, "mode", "") or "").strip().lower() if knowledge_profile is not None else ""
    memory_strategy = (context.memory_strategy if context is not None else "") or (blueprint.memory_strategy if blueprint else "")
    mentions_retrieval = any(token in memory_strategy.lower() for token in ("rag", "retrieval", "vector", "knowledge", "conocimiento"))
    if mode == "none" or context is not None and context.rag_required is False:
        return "not_required"
    if mode == "rag" or (context is not None and context.rag_required is True) or mentions_retrieval:
        return "approved_retrieval_design" if source_count or (context is not None and context.knowledge_sources) else "pending_knowledge_foundation"
    return "not_required"


def _first_actionable_construction_question(context: ProjectGenerationContext | None) -> str:
    if context is None:
        return "Revisar `ACP/construction-readiness/open-questions.yaml` antes de activar integraciones."
    for question in context.construction_questions:
        if question.answer_text.strip():
            continue
        if question.question_text.strip():
            impacted = ", ".join(question.impacted_artifacts[:3])
            suffix = f" Impacta: {impacted}." if impacted else ""
            return f"{question.question_text.strip()}{suffix}"
    if context.missing_fields:
        return f"Completar dato pendiente: {context.missing_fields[0].field}."
    return "Revisar `ACP/construction-readiness/open-questions.yaml` antes de activar integraciones."


def _tool_state_summary(snapshot: SessionSnapshot) -> str:
    blueprint = snapshot.blueprint
    if blueprint is None or not blueprint.tools:
        return "Sin herramientas externas confirmadas; mantener categoria pending_binding hasta definir contratos."
    categories = [_tool_binding_category(tool) for tool in project_blueprint_tools_for_construction(snapshot)]
    design_count = sum(1 for category in categories if category == "design_contract")
    pending_count = sum(1 for category in categories if category == "pending_binding")
    return f"{design_count} design_contract, {pending_count} pending_binding, 0 operational_binding."


def _retrieval_state_summary(snapshot: SessionSnapshot, context: ProjectGenerationContext | None) -> str:
    category = _retrieval_design_category(snapshot, context)
    return {
        "approved_retrieval_design": "approved_retrieval_design: usar fuentes y politicas aprobadas antes de implementar retrieval.",
        "pending_knowledge_foundation": "pending_knowledge_foundation: fijar fuentes, ingestion y vector store antes de activar retrieval.",
        "not_required": "not_required: el diseno aprobado no exige retrieval documental.",
    }.get(category, category)


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
    binding_category = _tool_binding_category(tool)
    payload = {
        "name": tool.name or f"tool_{index}",
        "purpose": tool.purpose,
        "tool_type": tool_type,
        "binding_category": binding_category,
        "binding_guidance": {
            "category": binding_category,
            "expected_next_step": (
                "Completar provider, endpoint permitido y secretos como referencias de entorno antes de activar la tool."
                if binding_category == "pending_binding"
                else "Implementar contra este contrato de diseno; no asumir binding operacional hasta confirmarlo."
            ),
            "forbidden_in_acp": ["plain_secret", "real_token", "cookie_value", "invented_endpoint"],
        },
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


def _build_manifest_file(snapshot: SessionSnapshot, context: ProjectGenerationContext | None = None) -> ACPFileEntry:
    discovery = snapshot.discovery
    blueprint = snapshot.blueprint
    runtime = _runtime_defaults(snapshot)
    retrieval_category = _retrieval_design_category(snapshot, context)
    payload = {
        "metadata": {
            "id": _slugify(snapshot.session.title, default="agent"),
            "name": snapshot.session.title.strip() or "needs_review",
            "version": "1.0.0",
            "maturity": "MVP",
            "generated_by": "Lean Agent Builder",
            "context_version": context.context_version if context is not None else "",
            "context_fingerprint": context.input_fingerprint if context is not None else "",
            "context_source_refs": _context_source_refs(context),
        },
        "business": {
            "objective": (context.desired_outcome if context and context.desired_outcome else discovery.desired_outcome if discovery else ""),
            "users": [context.current_user] if context and context.current_user else [discovery.current_user] if discovery and discovery.current_user else [],
            "kpis": [discovery.mvp_definition.north_star_metric] if discovery and discovery.mvp_definition.north_star_metric else [],
        },
        "architecture": {
            "topology": (context.architecture if context and context.architecture else blueprint.architecture if blueprint else ""),
            "reasoning_pattern": (context.reasoning_pattern if context and context.reasoning_pattern else blueprint.reasoning_pattern if blueprint else ""),
            "autonomy_level": discovery.autonomy_level if discovery else "",
        },
        "memory": {
            "strategy": (context.memory_strategy if context and context.memory_strategy else blueprint.memory_profile.strategy if blueprint else ""),
            "retrieval_design_category": retrieval_category,
            "short_term": "session",
            "long_term": "approved_sources" if retrieval_category == "approved_retrieval_design" else retrieval_category,
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


def _build_readme_file(snapshot: SessionSnapshot, context: ProjectGenerationContext | None = None) -> ACPFileEntry:
    discovery = snapshot.discovery
    blueprint = snapshot.blueprint
    canvas = snapshot.canvas
    title = snapshot.session.title or "Agent Construction Package"

    desired_outcome = (context.desired_outcome if context and context.desired_outcome else discovery.desired_outcome if discovery and discovery.desired_outcome else "needs_review")
    problem_statement = (context.problem_statement if context and context.problem_statement else discovery.problem_statement if discovery and discovery.problem_statement else "needs_review")
    primary_user = (
        context.current_user
        if context and context.current_user
        else (canvas.agent_profile.primary_user if canvas and canvas.agent_profile else None)
        or (discovery.current_user if discovery and discovery.current_user else "needs_review")
    )
    architecture = (context.architecture if context and context.architecture else blueprint.architecture if blueprint and blueprint.architecture else "needs_review")
    reasoning_pattern = (
        context.reasoning_pattern
        if context and context.reasoning_pattern
        else blueprint.reasoning_pattern if blueprint and blueprint.reasoning_pattern else "needs_review"
    )
    memory_strategy = (
        context.memory_strategy
        if context and context.memory_strategy
        else blueprint.memory_strategy if blueprint and blueprint.memory_strategy else "needs_review"
    )
    autonomy_level = discovery.autonomy_level if discovery and discovery.autonomy_level else "Supervisada (HITL)"
    tool_count = len(blueprint.tools) if blueprint and blueprint.tools else 0
    anchors = _context_anchor_values(context)
    first_question = _first_actionable_construction_question(context)

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
        f"- **Estado de Tools:** {_tool_state_summary(snapshot)}",
        f"- **Estado de Memoria/RAG:** {_retrieval_state_summary(snapshot, context)}",
        f"- **Primera Pregunta Accionable:** {first_question}",
        f"- **Fuentes de Contexto:** `{len(_context_source_refs(context))}` referencias trazadas.",
        f"- **Anclas de Especificidad:** {', '.join(anchors) if anchors else 'needs_review'}",
        "",
        "## 2. Estructura de Directorios del Paquete",
        "```text",
        "ACP/",
        "├── manifest.yaml             # Declaración formal de dependencias y versiones",
        "├── business/                 # Lean canvas, KPIs y restricciones de negocio",
        "├── architecture/             # Topología, C4 context y traza de decisiones",
        "├── cognition/                # Patrones de razonamiento y perfiles de workflow",
        "├── objectives/               # Objective Contract, criterios, progreso y terminacion",
        "├── memory/                   # Estrategia de memoria dual y context budgets",
        "├── knowledge/                # Fuentes documentales, embeddings e ingestión",
        "├── tools/                    # Contratos tipados y permisos de herramientas",
        "├── workflows/                # Máquinas de estado y grafos ejecutables",
        "├── construction-readiness/   # Guía paso a paso, preguntas, gaps y decisiones",
        "├── prompts/                  # Prompts de roles (planner, evaluator, system, skills)",
        "├── adapters/                 # Guías de configuración para Cursor, Codex y Claude Code",
        "├── conformance/              # Reglas de validación y linters de construcción",
        "├── costs/                    # Estimación de costo operativo del agente",
        "├── evaluation/               # Datasets de evaluación y casos de prueba",
        "├── deployment/               # Especificaciones de infraestructura y runtime",
        "├── index.html                # Viewer portable de navegación y storytelling",
        "└── launcher/                 # Scripts de inicialización multiplataforma",
        "```",
        "",
        "## 3. Recorrido recomendado",
        "1. Abre `ACP/index.html` para recorrer la historia de implementación.",
        "2. Lee `ACP/construction-readiness/construction-guide.md` como guía paso a paso antes de construir.",
        "3. Usa `ACP/IMPLEMENTATION_GUIDE.md` como contrato rector para Codex, Cursor, Antigravity, Claude Code u otra herramienta agentica.",
        "4. Revisa `ACP/construction-readiness/open-questions.yaml` y `ACP/construction-readiness/deferred-decisions.yaml` antes de modificar artefactos.",
        "5. Consulta `ACP/costs/operational-cost-estimate.md` para entender supuestos de operación y consumo.",
        "",
        "## 4. Instrucciones de Arranque y Asistencia con IDEs",
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
        "## 5. Gobernanza y Fuente de Verdad",
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
            {
                "target_key": "langgraph",
                "label": "LangGraph",
                "adapter_doc": "ACP/adapters/langgraph.md",
                "launch_preference": "implementation_mapping",
                "command_candidates": [],
                "recommended_when": [
                    "El ACP requiere flujo con estado, aprobaciones humanas, retries o memoria entre pasos.",
                    "Existe `ACP/workflows/langgraph.json` como mapa inicial que debe revisarse antes de codificar.",
                ],
            },
            {
                "target_key": "pure-code",
                "label": "Codigo puro",
                "adapter_doc": "ACP/adapters/pure-code.md",
                "launch_preference": "implementation_mapping",
                "command_candidates": [],
                "recommended_when": [
                    "El equipo necesita control completo sobre runtime, seguridad, deployment y contratos API.",
                    "Existen side effects, integraciones custom o politicas de aprobacion que no conviene ocultar en low-code.",
                ],
            },
            {
                "target_key": "n8n",
                "label": "n8n",
                "adapter_doc": "ACP/adapters/n8n.md",
                "launch_preference": "orientation_only",
                "command_candidates": [],
                "recommended_when": [
                    "El flujo puede expresarse como automatizacion por nodos con APIs HTTP y aprobaciones humanas claras.",
                    "El ACP no requiere memoria compleja, estado durable fino ni politicas de runtime avanzadas.",
                ],
            },
            {
                "target_key": "make",
                "label": "Make",
                "adapter_doc": "ACP/adapters/make.md",
                "launch_preference": "orientation_only",
                "command_candidates": [],
                "recommended_when": [
                    "El caso se parece a una automatizacion SaaS lineal con triggers, acciones y pocas ramas.",
                    "Las credenciales, endpoints y payloads estan confirmados antes de crear escenarios importables.",
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
    langgraph = "\n".join(
        [
            "# Adapter: LangGraph",
            "",
            "Usa `ACP/workflows/langgraph.json` como mapa inicial, no como codigo final.",
            "",
            "Antes de implementar:",
            "",
            "1. Confirma `ACP/construction-readiness/overview.yaml`.",
            "2. Cierra decisiones de memoria, runtime y side effects.",
            "3. Revisa aprobaciones humanas en `ACP/tools/permissions.yaml`.",
            "4. Traduce cada nodo a una funcion con estado, retries y compensacion verificables.",
        ]
    )
    pure_code = "\n".join(
        [
            "# Adapter: Codigo puro",
            "",
            "Usa este target cuando el ACP requiera control completo de runtime, secretos, seguridad, integraciones o deployment.",
            "",
            "No empieces desde cero: implementa contra los contratos de `ACP/tools/`, `ACP/workflows/`, `ACP/runtime/`, `ACP/memory/` y `ACP/evaluation/`.",
        ]
    )
    n8n = "\n".join(
        [
            "# Adapter: n8n",
            "",
            "Este adapter es orientativo. No crea workflows ni conecta cuentas reales.",
            "",
            "n8n puede ser viable cuando el flujo del ACP se expresa como nodos HTTP/SaaS, aprobaciones humanas claras y bajo requerimiento de memoria compleja.",
            "",
            "Antes de crear un workflow importable, confirma endpoints, auth, payloads, secretos, politica HITL y limites de runtime.",
        ]
    )
    make = "\n".join(
        [
            "# Adapter: Make",
            "",
            "Este adapter es orientativo. No crea escenarios ni conecta cuentas reales.",
            "",
            "Make puede ser viable para automatizaciones lineales con triggers, acciones SaaS y pocas ramas de estado.",
            "",
            "Si el ACP requiere reasoning, memoria persistente, retries complejos o side effects gobernados, usa este target solo para subflujos acotados.",
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
        build_acp_file_entry(
            path="ACP/adapters/langgraph.md",
            domain="adapters",
            title="LangGraph adapter",
            format="markdown",
            source_sections=["runtime_targets", "workflows", "memory"],
            content_text=serialize_markdown_document(langgraph),
        ),
        build_acp_file_entry(
            path="ACP/adapters/pure-code.md",
            domain="adapters",
            title="Pure code adapter",
            format="markdown",
            source_sections=["runtime_targets", "tools", "deployment"],
            content_text=serialize_markdown_document(pure_code),
        ),
        build_acp_file_entry(
            path="ACP/adapters/n8n.md",
            domain="adapters",
            title="n8n adapter",
            format="markdown",
            source_sections=["runtime_targets", "tools", "workflows"],
            content_text=serialize_markdown_document(n8n),
        ),
        build_acp_file_entry(
            path="ACP/adapters/make.md",
            domain="adapters",
            title="Make adapter",
            format="markdown",
            source_sections=["runtime_targets", "tools", "workflows"],
            content_text=serialize_markdown_document(make),
        ),
    ]


def _build_business_files(snapshot: SessionSnapshot, context: ProjectGenerationContext | None = None) -> list[ACPFileEntry]:
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
        **_context_trace_payload(context),
        "problem_statement": context.problem_statement if context and context.problem_statement else discovery.problem_statement,
        "current_user": context.current_user if context and context.current_user else discovery.current_user,
        "current_process": context.current_process if context and context.current_process else discovery.current_process,
        "desired_outcome": context.desired_outcome if context and context.desired_outcome else discovery.desired_outcome,
        "value_statement": discovery.value_statement,
        "mvp_scope": context.mvp_scope if context and context.mvp_scope else canvas.mvp_scope,
        "out_of_scope": context.out_of_scope if context and context.out_of_scope else canvas.out_of_scope,
        "primary_risk": context.risks[0] if context and context.risks else canvas.primary_risk,
        "allowed_decisions": canvas.agent_profile.allowed_decisions,
        "prohibited_decisions": canvas.agent_profile.prohibited_decisions,
    }
    kpi_payload = {
        **_context_trace_payload(context),
        "north_star_metric": discovery.mvp_definition.north_star_metric,
        "success_metrics": canvas.agent_profile.success_metrics or [canvas.success_metric],
        "success_metric": canvas.success_metric,
    }
    constraints_payload = {
        **_context_trace_payload(context),
        "constraints": context.constraints if context and context.constraints else discovery.constraints,
        "non_delegable_decisions": context.nondelegable_decisions if context and context.nondelegable_decisions else discovery.mvp_definition.non_delegable_decisions,
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


def _build_architecture_files(snapshot: SessionSnapshot, context: ProjectGenerationContext | None = None) -> list[ACPFileEntry]:
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
        **_context_trace_payload(context),
        "architecture": context.architecture if context and context.architecture else blueprint.architecture,
        "case_type": discovery.case_type if discovery else "",
        "reasoning_pattern": context.reasoning_pattern if context and context.reasoning_pattern else blueprint.reasoning_pattern,
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


def _cognition_default_patterns(selected_pattern: str, has_tools: bool) -> list[dict[str, Any]]:
    normalized = selected_pattern.strip().lower()
    return [
        {
            "family": "reasoning",
            "key": "plan_and_execute",
            "label": "Plan-and-Execute",
            "summary": "Descomponer la solicitud en pasos verificables antes de ejecutar herramientas o responder.",
            "use_when": [
                "El usuario pide una accion con multiples dependencias.",
                "La respuesta depende de fuentes externas, memoria o aprobaciones.",
            ],
            "tradeoffs": ["Mas trazabilidad a cambio de mayor latencia."],
            "fit_score": 90,
            "selected": "plan" in normalized or "execute" in normalized,
        },
        {
            "family": "reasoning",
            "key": "react_tool_loop",
            "label": "ReAct tool loop",
            "summary": "Alternar observacion, seleccion de herramienta, accion, verificacion y respuesta trazable.",
            "use_when": [
                "El agente necesita consultar herramientas antes de decidir.",
                "La solicitud requiere evidencia actualizada o efectos externos gobernados.",
            ],
            "tradeoffs": ["Requiere contratos de tools y manejo estricto de errores."],
            "fit_score": 88 if has_tools else 70,
            "selected": bool(has_tools),
        },
        {
            "family": "reasoning",
            "key": "human_in_the_loop",
            "label": "Human-in-the-loop gated reasoning",
            "summary": "Pausar decisiones sensibles, side effects o incertidumbre alta para aprobacion humana.",
            "use_when": [
                "La accion modifica sistemas externos.",
                "Falta evidencia, hay ambiguedad o la confianza no alcanza el umbral definido.",
            ],
            "tradeoffs": ["Reduce automatizacion pero aumenta control operacional."],
            "fit_score": 82,
            "selected": True,
        },
    ]


def _cognition_tool_entries(snapshot: SessionSnapshot) -> list[dict[str, Any]]:
    tools = project_blueprint_tools_for_construction(snapshot) if snapshot.blueprint is not None else []
    entries: list[dict[str, Any]] = []
    for index, tool in enumerate(tools, start=1):
        entries.append(
            {
                "tool_name": getattr(tool, "name", ""),
                "connector_key": getattr(tool, "connector_key", None) or getattr(tool, "registered_api_ref", "") or "",
                "purpose": getattr(tool, "purpose", ""),
                "contract_ref": build_tool_contract_path_for_tool(tool, index),
                "permission_mode": _tool_permission_mode(tool),
                "risk_level": getattr(tool, "risk_level", "") or "medium",
                "side_effects": bool(getattr(tool, "has_side_effects", False)),
                "requires_approval": bool(getattr(tool, "requires_approval", False)),
                "when_to_use": getattr(tool, "when_to_use", "") or getattr(tool, "purpose", ""),
                "failure_mode": getattr(tool, "failure_mode", "") or "fail_closed_and_ask_for_review",
            }
        )
    return entries


def _cognition_connector_policies(snapshot: SessionSnapshot) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    tools = project_blueprint_tools_for_construction(snapshot) if snapshot.blueprint is not None else []
    connector_keys = {
        str(
            resolve_whatsapp_connector_key(tool)
            or resolve_google_workspace_connector_key(tool)
            or resolve_odoo_connector_key(tool)
            or getattr(tool, "connector_key", "")
            or getattr(tool, "registered_api_ref", "")
            or getattr(tool, "name", "")
        )
        .strip()
        .lower()
        .replace("-", "_")
        for tool in tools
    }
    if "whatsapp_cloud_api" in connector_keys or "whatsapp_business_messaging" in connector_keys:
        policies.append(
            {
                "connector_family": "whatsapp",
                "reasoning_policy": "Tratar WhatsApp como canal de entrada/salida gobernado; normalizar inbound antes de decidir.",
                "must_escalate_when": ["message_type=image", "payload_ambiguous", "delivery_failure_after_retry"],
                "evidence_required": ["wa_id", "message_id", "timestamp", "normalized_message_type"],
                "side_effect_guard": "No enviar template, confirmacion de compra o instruccion de pago sin politica de aprobacion aplicable.",
            }
        )
    if "google_sheets_read_table" in connector_keys:
        policies.append(
            {
                "connector_family": "google_sheets",
                "reasoning_policy": "Usar Sheets como fuente tabular para catalogo, stock, precios o datos operativos autorizados.",
                "must_verify_before_decision": ["spreadsheet_id", "range", "header_row", "source_revision"],
                "evidence_required": ["row_id_or_range", "columns_used", "fetched_at"],
                "fallback": "Si falta fila, precio, stock o revision, responder needs_review antes de cotizar.",
            }
        )
    if "google_drive_file_picker" in connector_keys:
        policies.append(
            {
                "connector_family": "google_drive",
                "reasoning_policy": "Usar Drive solo para archivos seleccionados o permitidos, con referencias de fuente trazables.",
                "must_verify_before_decision": ["file_id", "mime_type", "export_format", "source_ref"],
                "evidence_required": ["file_id", "file_name", "source_ref"],
                "fallback": "Si la ficha, imagen o documento no existe o no esta autorizado, no inventar atributos del producto.",
            }
        )
    if any(key.startswith("odoo_") for key in connector_keys) or _odoo_connector_keys_for_snapshot(snapshot):
        policies.append(
            {
                "connector_family": "odoo",
                "reasoning_policy": "Usar Odoo mediante allowlist de modelos/campos y approval gate para escrituras comerciales.",
                "must_verify_before_decision": ["odoo_version", "model_allowlist", "technical_user_permissions"],
                "evidence_required": ["model", "record_id", "domain_or_payload_hash"],
                "side_effect_guard": "No crear o actualizar registros sin idempotency_key, owner y aprobacion cuando aplique.",
            }
        )
    return policies


def _default_planner_steps(snapshot: SessionSnapshot) -> list[dict[str, Any]]:
    objective_statement = ""
    if snapshot.blueprint is not None:
        objective = active_objective(snapshot.blueprint.objective_contract)
        objective_statement = objective.statement if objective is not None else ""
    if not objective_statement and snapshot.canvas is not None:
        objective_statement = snapshot.canvas.user_goal

    steps = [
        {
            "step_key": "intake",
            "objective": "Recibir solicitud, identificar canal, usuario, intencion y datos minimos.",
            "required_inputs": ["user_message", "channel_context"],
            "outputs": ["normalized_request", "intent", "missing_fields"],
            "tool_policy": "no_tool_call",
            "approval_required": False,
        },
        {
            "step_key": "plan",
            "objective": "Crear plan corto contra el objetivo activo sin exponer cadena de pensamiento privada.",
            "required_inputs": ["normalized_request", "objective_contract", "guardrails"],
            "outputs": ["plan_summary", "tool_sequence", "stop_conditions"],
            "tool_policy": "select_tools_by_contract",
            "approval_required": False,
        },
        {
            "step_key": "gather_evidence",
            "objective": "Consultar solo herramientas necesarias para obtener evidencia actualizada.",
            "required_inputs": ["tool_sequence", "tool_contracts", "allowed_resources"],
            "outputs": ["evidence_refs", "tool_results", "confidence_signals"],
            "tool_policy": "read_only_tools_first",
            "approval_required": False,
        },
        {
            "step_key": "decide",
            "objective": "Comparar evidencia contra reglas de negocio, restricciones y criterios del objetivo.",
            "required_inputs": ["evidence_refs", "business_rules", "memory_policy"],
            "outputs": ["decision_summary", "risk_flags", "next_action"],
            "tool_policy": "no_side_effects_during_decision",
            "approval_required": False,
        },
        {
            "step_key": "act_or_escalate",
            "objective": "Ejecutar accion permitida, pedir aprobacion o escalar a humano segun riesgo y evidencia.",
            "required_inputs": ["next_action", "approval_policy", "side_effect_policy"],
            "outputs": ["action_result", "handoff_ref", "user_response"],
            "tool_policy": "side_effects_require_gate",
            "approval_required": True,
        },
        {
            "step_key": "verify_and_log",
            "objective": "Verificar resultado, registrar trazabilidad y decidir cierre o replanificacion acotada.",
            "required_inputs": ["action_result", "evidence_refs", "objective_success_criteria"],
            "outputs": ["verification_status", "decision_log", "final_or_replan_signal"],
            "tool_policy": "audit_only",
            "approval_required": False,
        },
    ]
    if objective_statement:
        steps[1]["active_objective"] = objective_statement
    return steps


def _build_cognition_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    tools = _cognition_tool_entries(snapshot)
    pattern_catalog = [
        item.model_dump(mode="json")
        for item in blueprint.delivery_package.pattern_catalog
        if item.family == "reasoning"
    ]
    if not pattern_catalog:
        pattern_catalog = _cognition_default_patterns(blueprint.reasoning_pattern, bool(tools))
    workflow_steps = [
        item.model_dump(mode="json")
        for item in blueprint.delivery_package.workflow_profile.steps
        if item.name or item.objective
    ]
    if not workflow_steps:
        workflow_steps = _default_planner_steps(snapshot)
    objective = active_objective(blueprint.objective_contract)
    reasoning_payload = {
        "schema_version": "acp-cognition-reasoning.v1",
        "selected_pattern": blueprint.reasoning_pattern,
        "available_patterns": pattern_catalog,
        "pattern_stack": [
            item["label"]
            for item in pattern_catalog
            if item.get("selected") or item.get("key") in {"plan_and_execute", "react_tool_loop"}
        ],
        "active_objective": objective.statement if objective is not None else (snapshot.canvas.user_goal if snapshot.canvas else ""),
        "plan_summary_policy": blueprint.delivery_package.observability_plan.plan_summary_policy
        or "Exponer un resumen breve del plan, no la cadena de pensamiento privada.",
        "private_reasoning_policy": {
            "do_not_export_chain_of_thought": True,
            "expose_only": ["plan_summary", "tool_calls", "evidence_refs", "decision_summary", "approval_reason"],
            "redact": ["secret_values", "tokens", "raw_private_reasoning"],
        },
        "operating_loop": [
            {"step": "observe", "purpose": "Normalizar solicitud, canal, objetivo y restricciones."},
            {"step": "plan", "purpose": "Definir pasos verificables y herramientas necesarias."},
            {"step": "act", "purpose": "Ejecutar solo herramientas permitidas por contrato."},
            {"step": "verify", "purpose": "Contrastar resultado con evidencia, guardrails y criterios de exito."},
            {"step": "respond_or_escalate", "purpose": "Responder, pedir aprobacion o transferir a humano."},
        ],
        "decision_rules": [
            "Si falta un dato critico, preguntar o marcar needs_review antes de actuar.",
            "Si la accion tiene side effects, aplicar approval_gate o owner explicito antes de ejecutar.",
            "Si la evidencia contradice el objetivo o los datos fuente, detener y escalar.",
            "Si una herramienta falla, registrar fallo, aplicar retry policy y evitar duplicados.",
        ],
        "tool_reasoning_contracts": tools,
        "connector_policies": _cognition_connector_policies(snapshot),
    }
    planner_payload = {
        "schema_version": "acp-cognition-planner.v1",
        "execution_pattern": blueprint.delivery_package.workflow_profile.execution_pattern
        or "plan -> gather_evidence -> decide -> act_or_escalate -> verify_and_log",
        "steps": workflow_steps,
        "checkpoint_policy": blueprint.delivery_package.workflow_profile.checkpoint_policy
        or "Persistir checkpoint antes de tool calls, antes de side effects y despues de verificacion.",
        "retry_strategy": blueprint.delivery_package.workflow_profile.retry_strategy
        or "Un reintento gobernado para fallas transitorias; despues escalar o marcar needs_review.",
        "approval_pause": blueprint.delivery_package.workflow_profile.approval_pause
        or "Pausar antes de escrituras externas, confirmaciones comerciales o acciones irreversibles.",
        "stop_conditions": [
            "insufficient_evidence",
            "missing_required_field",
            "constraint_violation",
            "tool_contract_missing",
            "human_handoff_required",
        ],
        "checkpoint_schema": {
            "required_fields": [
                "session_id",
                "step_key",
                "objective_ref",
                "tool_refs",
                "evidence_refs",
                "decision_summary",
                "approval_state",
                "next_action",
            ]
        },
    }
    reflection_payload = {
        "schema_version": "acp-cognition-reflection.v1",
        "review_trigger": blueprint.memory_profile.review_trigger,
        "goal_drift_guard": blueprint.memory_profile.goal_drift_guard,
        "decision_logging": blueprint.delivery_package.observability_plan.decision_logging
        or "Registrar decision_summary, evidence_refs, tool_result_refs, approval_state y next_action.",
        "self_check_rubric": [
            "La respuesta esta alineada con el objetivo activo.",
            "Cada recomendacion o cotizacion tiene evidencia o fuente autorizada.",
            "No se inventaron campos, precios, stock, politicas ni credenciales.",
            "Los side effects fueron aprobados o escalados segun politica.",
            "El usuario recibio una respuesta accionable o una pregunta concreta.",
        ],
        "reflection_triggers": [
            "tool_error",
            "low_confidence",
            "contradictory_evidence",
            "before_side_effect",
            "before_final_answer",
            "after_human_handoff",
        ],
        "escalation_policy": {
            "escalate_when": [
                "unsupported_media_or_image",
                "ambiguous_business_decision",
                "payment_or_purchase_confirmation_unclear",
                "policy_or_price_missing",
                "security_or_secret_issue",
            ],
            "handoff_payload": ["reason", "conversation_ref", "evidence_refs", "last_safe_state", "recommended_next_step"],
        },
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


def _build_memory_files(snapshot: SessionSnapshot, context: ProjectGenerationContext | None = None) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    memory_profile = blueprint.memory_profile
    grounding_payload = memory_profile.grounding_policy.model_dump(mode="json")
    retrieval_category = _retrieval_design_category(snapshot, context)
    strategy_payload = {
        "strategy": memory_profile.strategy or blueprint.memory_strategy,
        "retrieval_design_category": retrieval_category,
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
        "retrieval_design_category": retrieval_category,
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
    context: ProjectGenerationContext | None = None,
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
    retrieval_category = _retrieval_design_category(snapshot, context)
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
        "retrieval_design_category": retrieval_category,
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
            "retrieval_design_category": "not_required",
            "strategy": "not_required",
            "frequency": "not_required",
            "owner": "not_required",
            "mechanism": "not_required",
            "notes": knowledge_profile.notes or "No hay ingestion porque el caso no usa retrieval documental.",
        }
        disabled_embeddings_payload = {
            "retrieval_design_category": "not_required",
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
        "retrieval_design_category": retrieval_category,
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
            "retrieval_design_category": retrieval_category,
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
            "retrieval_design_category": retrieval_category,
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


def _build_tools_files(snapshot: SessionSnapshot, context: ProjectGenerationContext | None = None) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    tools: list[Any] = project_blueprint_tools_for_construction(snapshot)
    existing_odoo_keys = {
        key
        for key in (_odoo_connector_key(tool) for tool in tools)
        if key
    }
    inferred_odoo_keys = _odoo_connector_keys_for_snapshot(snapshot) - existing_odoo_keys
    base_tool_count = len(tools)
    for offset, key in enumerate(sorted(inferred_odoo_keys), start=1):
        tools.append(_synthetic_odoo_tool(key, index=base_tool_count + offset))
    permissions_payload = {
        "context_version": context.context_version if context is not None else "",
        "context_source_refs": _context_source_refs(context),
        "tools": [
            {
                "name": item.name,
                "binding_category": _tool_binding_category(item),
                "requires_approval": item.requires_approval,
                "approval_reason": item.approval_reason,
                "risk_level": item.risk_level,
                "side_effects": item.has_side_effects,
            }
            for item in tools
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
    files.extend(_tool_contract_file(tool, index) for index, tool in enumerate(tools, start=1))
    return files


def _tool_connector_slug(tool: Any, index: int) -> str:
    if _is_whatsapp_cloud_tool(tool):
        return "whatsapp-cloud-api"
    google_key = _google_workspace_connector_key(tool)
    if google_key:
        return google_key.replace("_", "-")
    odoo_key = _odoo_connector_key(tool)
    if odoo_key:
        return odoo_key.replace("_", "-")
    registered_ref = str(getattr(tool, "registered_api_ref", "") or "").strip()
    source = registered_ref.rsplit("/", 1)[-1] if registered_ref else str(getattr(tool, "name", "") or "")
    return _slugify(source, default=f"tool-{index}")


def _tool_contract_ref(tool: Any, index: int) -> str:
    tool_type = getattr(tool, "tool_type", "external") or "external"
    return build_tool_contract_path(str(getattr(tool, "name", "") or ""), index, tool_type=tool_type)


def _tool_secret_key(tool: Any, index: int) -> str:
    return _tool_connector_slug(tool, index).replace("-", "_").upper()


def _tool_permission_mode(tool: Any) -> str:
    return "write" if getattr(tool, "has_side_effects", False) else "read"


def _is_whatsapp_cloud_tool(tool: Any) -> bool:
    return bool(resolve_whatsapp_connector_key(tool))


GOOGLE_WORKSPACE_CONNECTOR_PROFILES: dict[str, dict[str, Any]] = {
    "google_drive_file_picker": {
        "label": "Google Drive Picker - selected files",
        "service": "drive",
        "binding_type": "oauth2_rest_api",
        "actions": ["picker_select_file", "read_file_metadata", "download_or_export_selected_file"],
        "required_fields": ["oauth_client_id_ref", "oauth_client_secret_ref", "redirect_uri_ref", "allowed_scopes", "selected_file_policy"],
        "required_env_refs": ["GOOGLE_OAUTH_REDIRECT_URI", "GOOGLE_ALLOWED_SCOPES"],
        "required_secret_refs": ["GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN_REF"],
        "contract_refs": ["ACP/integrations/google-drive/file-picker.contract.yaml", "ACP/integrations/google-drive/selected-file-reader.contract.yaml"],
        "minimum_scope_policy": "Preferir drive.file con Google Picker; evitar drive.readonly amplio salvo decision explicita.",
        "side_effects": False,
    },
    "google_sheets_read_table": {
        "label": "Google Sheets API - read table",
        "service": "sheets",
        "binding_type": "oauth2_rest_api",
        "actions": ["read_values", "read_spreadsheet_metadata"],
        "required_fields": ["oauth_client_id_ref", "oauth_client_secret_ref", "spreadsheet_id", "range", "header_row", "cache_policy"],
        "required_env_refs": ["GOOGLE_OAUTH_REDIRECT_URI", "GOOGLE_ALLOWED_SCOPES", "GOOGLE_SHEETS_SPREADSHEET_ID"],
        "required_secret_refs": ["GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN_REF"],
        "contract_refs": ["ACP/integrations/google-sheets/read-table.contract.yaml", "ACP/integrations/google-sheets/schema-mapping.yaml"],
        "minimum_scope_policy": "Lectura sobre spreadsheet seleccionado; no habilitar escritura por defecto.",
        "side_effects": False,
    },
    "google_calendar_availability_reader": {
        "label": "Google Calendar API - availability",
        "service": "calendar",
        "binding_type": "oauth2_rest_api",
        "actions": ["query_freebusy", "list_events_readonly"],
        "required_fields": ["oauth_client_id_ref", "oauth_client_secret_ref", "calendar_id", "timezone", "availability_window"],
        "required_env_refs": ["GOOGLE_OAUTH_REDIRECT_URI", "GOOGLE_ALLOWED_SCOPES", "GOOGLE_CALENDAR_DEFAULT_ID"],
        "required_secret_refs": ["GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN_REF"],
        "contract_refs": ["ACP/integrations/google-calendar/availability.contract.yaml"],
        "minimum_scope_policy": "Usar disponibilidad/lectura minima; evitar lectura amplia de eventos cuando no sea necesaria.",
        "side_effects": False,
    },
    "google_calendar_event_creator": {
        "label": "Google Calendar API - create event",
        "service": "calendar",
        "binding_type": "oauth2_rest_api",
        "actions": ["create_event", "update_event_if_approved"],
        "required_fields": ["oauth_client_id_ref", "oauth_client_secret_ref", "calendar_id", "timezone", "attendee_policy", "idempotency_key"],
        "required_env_refs": ["GOOGLE_OAUTH_REDIRECT_URI", "GOOGLE_ALLOWED_SCOPES", "GOOGLE_CALENDAR_DEFAULT_ID"],
        "required_secret_refs": ["GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN_REF"],
        "contract_refs": ["ACP/integrations/google-calendar/create-event.contract.yaml", "ACP/integrations/google-calendar/approval-policy.yaml"],
        "minimum_scope_policy": "Crear eventos solo con approval policy o regla de negocio explicita.",
        "side_effects": True,
    },
    "gmail_draft_creator": {
        "label": "Gmail API - create draft",
        "service": "gmail",
        "binding_type": "oauth2_rest_api",
        "actions": ["create_draft"],
        "required_fields": ["oauth_client_id_ref", "oauth_client_secret_ref", "sender_account", "recipient_policy", "draft_review_policy"],
        "required_env_refs": ["GOOGLE_OAUTH_REDIRECT_URI", "GOOGLE_ALLOWED_SCOPES"],
        "required_secret_refs": ["GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN_REF"],
        "contract_refs": ["ACP/integrations/gmail/create-draft.contract.yaml", "ACP/integrations/gmail/restricted-scope-warning.yaml"],
        "minimum_scope_policy": "Preferir borradores para revision humana antes de enviar.",
        "side_effects": True,
    },
    "gmail_send_message": {
        "label": "Gmail API - send message",
        "service": "gmail",
        "binding_type": "oauth2_rest_api",
        "actions": ["send_message"],
        "required_fields": ["oauth_client_id_ref", "oauth_client_secret_ref", "sender_account", "recipient_policy", "approval_policy", "idempotency_key"],
        "required_env_refs": ["GOOGLE_OAUTH_REDIRECT_URI", "GOOGLE_ALLOWED_SCOPES"],
        "required_secret_refs": ["GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN_REF"],
        "contract_refs": ["ACP/integrations/gmail/send-message.contract.yaml", "ACP/integrations/gmail/restricted-scope-warning.yaml"],
        "minimum_scope_policy": "gmail.send requiere approval_gate o politica explicita; no habilitar lectura amplia del inbox en MVP.",
        "side_effects": True,
    },
}


def _google_workspace_connector_key(tool: Any) -> str:
    return resolve_google_workspace_connector_key(tool)


def _is_google_workspace_tool(tool: Any) -> bool:
    return bool(_google_workspace_connector_key(tool))


ODOO_CONNECTOR_PROFILES: dict[str, dict[str, Any]] = {
    "odoo_partner_read": {
        "label": "Odoo - leer clientes/contactos",
        "model": "res.partner",
        "binding_type": "versioned_rpc_api",
        "actions": ["search_read", "read"],
        "required_fields": ["api_mode", "base_url_ref", "database_ref", "username_ref", "auth_secret_ref", "allowed_models"],
        "required_env_refs": ["ODOO_API_MODE", "ODOO_BASE_URL", "ODOO_DATABASE", "ODOO_ALLOWED_MODELS"],
        "required_secret_refs": ["ODOO_USERNAME", "ODOO_PASSWORD", "ODOO_API_KEY"],
        "contract_refs": ["ACP/integrations/odoo/rpc-17-18.contract.yaml", "ACP/integrations/odoo/json2-19.contract.yaml", "ACP/integrations/odoo/models-scope.yaml"],
        "side_effects": False,
    },
    "odoo_crm_lead_read": {
        "label": "Odoo CRM - leer leads/oportunidades",
        "model": "crm.lead",
        "binding_type": "versioned_rpc_api",
        "actions": ["search_read", "read"],
        "required_fields": ["api_mode", "base_url_ref", "database_ref", "username_ref", "auth_secret_ref", "allowed_models"],
        "required_env_refs": ["ODOO_API_MODE", "ODOO_BASE_URL", "ODOO_DATABASE", "ODOO_ALLOWED_MODELS"],
        "required_secret_refs": ["ODOO_USERNAME", "ODOO_PASSWORD", "ODOO_API_KEY"],
        "contract_refs": ["ACP/integrations/odoo/rpc-17-18.contract.yaml", "ACP/integrations/odoo/json2-19.contract.yaml", "ACP/integrations/odoo/models-scope.yaml"],
        "side_effects": False,
    },
    "odoo_sale_order_read": {
        "label": "Odoo Sales - leer cotizaciones/pedidos",
        "model": "sale.order",
        "binding_type": "versioned_rpc_api",
        "actions": ["search_read", "read"],
        "required_fields": ["api_mode", "base_url_ref", "database_ref", "username_ref", "auth_secret_ref", "allowed_models"],
        "required_env_refs": ["ODOO_API_MODE", "ODOO_BASE_URL", "ODOO_DATABASE", "ODOO_ALLOWED_MODELS"],
        "required_secret_refs": ["ODOO_USERNAME", "ODOO_PASSWORD", "ODOO_API_KEY"],
        "contract_refs": ["ACP/integrations/odoo/rpc-17-18.contract.yaml", "ACP/integrations/odoo/json2-19.contract.yaml", "ACP/integrations/odoo/models-scope.yaml"],
        "side_effects": False,
    },
    "odoo_sale_quote_create": {
        "label": "Odoo Sales - crear cotizacion",
        "model": "sale.order",
        "binding_type": "versioned_rpc_api",
        "actions": ["create_quote"],
        "required_fields": ["api_mode", "base_url_ref", "database_ref", "username_ref", "auth_secret_ref", "allowed_models", "allowed_write_actions"],
        "required_env_refs": ["ODOO_API_MODE", "ODOO_BASE_URL", "ODOO_DATABASE", "ODOO_ALLOWED_MODELS", "ODOO_ALLOWED_WRITE_ACTIONS"],
        "required_secret_refs": ["ODOO_USERNAME", "ODOO_PASSWORD", "ODOO_API_KEY"],
        "contract_refs": ["ACP/integrations/odoo/rpc-17-18.contract.yaml", "ACP/integrations/odoo/json2-19.contract.yaml", "ACP/integrations/odoo/models-scope.yaml", "ACP/integrations/odoo/quote-policy.yaml", "ACP/integrations/odoo/write-approval-policy.yaml"],
        "side_effects": True,
    },
    "odoo_activity_create": {
        "label": "Odoo - crear actividad de seguimiento",
        "model": "mail.activity",
        "binding_type": "versioned_rpc_api",
        "actions": ["create_activity"],
        "required_fields": ["api_mode", "base_url_ref", "database_ref", "username_ref", "auth_secret_ref", "allowed_models", "allowed_write_actions"],
        "required_env_refs": ["ODOO_API_MODE", "ODOO_BASE_URL", "ODOO_DATABASE", "ODOO_ALLOWED_MODELS", "ODOO_ALLOWED_WRITE_ACTIONS"],
        "required_secret_refs": ["ODOO_USERNAME", "ODOO_PASSWORD", "ODOO_API_KEY"],
        "contract_refs": ["ACP/integrations/odoo/rpc-17-18.contract.yaml", "ACP/integrations/odoo/json2-19.contract.yaml", "ACP/integrations/odoo/models-scope.yaml", "ACP/integrations/odoo/write-approval-policy.yaml"],
        "side_effects": True,
    },
    "odoo_crm_lead_update": {
        "label": "Odoo CRM - actualizar lead/oportunidad",
        "model": "crm.lead",
        "binding_type": "versioned_rpc_api",
        "actions": ["update_lead"],
        "required_fields": ["api_mode", "base_url_ref", "database_ref", "username_ref", "auth_secret_ref", "allowed_models", "allowed_write_actions"],
        "required_env_refs": ["ODOO_API_MODE", "ODOO_BASE_URL", "ODOO_DATABASE", "ODOO_ALLOWED_MODELS", "ODOO_ALLOWED_WRITE_ACTIONS"],
        "required_secret_refs": ["ODOO_USERNAME", "ODOO_PASSWORD", "ODOO_API_KEY"],
        "contract_refs": ["ACP/integrations/odoo/rpc-17-18.contract.yaml", "ACP/integrations/odoo/json2-19.contract.yaml", "ACP/integrations/odoo/models-scope.yaml", "ACP/integrations/odoo/write-approval-policy.yaml"],
        "side_effects": True,
    },
}

def _odoo_connector_key(tool: Any) -> str:
    return resolve_odoo_connector_key(tool)


def _is_odoo_tool(tool: Any) -> bool:
    return bool(_odoo_connector_key(tool))


def _snapshot_has_odoo_quote_signal(snapshot: SessionSnapshot) -> bool:
    return snapshot_has_odoo_quote_signal(snapshot)


def _odoo_connector_keys_for_snapshot(snapshot: SessionSnapshot) -> set[str]:
    return projected_odoo_connector_keys_for_snapshot(snapshot)


def _synthetic_odoo_tool(key: str, *, index: int) -> SimpleNamespace:
    profile = ODOO_CONNECTOR_PROFILES[key]
    return SimpleNamespace(
        name=key,
        purpose=profile["label"],
        archetype="transactional_write" if profile["side_effects"] else "read_only_lookup",
        integration_kind="versioned_rpc_api",
        tool_type="external",
        execution_stage="execution",
        when_to_use=f"Usar cuando el flujo aprobado requiera {profile['label']} contra Odoo.",
        connector_key=key,
        registered_api_ref=key,
        risk_level="high" if profile["side_effects"] else "medium",
        requires_approval=bool(profile["side_effects"]),
        has_side_effects=bool(profile["side_effects"]),
        inputs=[],
        outputs=[],
        request_schema={"type": "object", "properties": {"odoo_model": {"type": "string", "enum": [profile["model"]]}}},
        response_schema={"type": "object", "properties": {"status": {"type": "string"}, "odoo_record_id": {"type": "integer"}}},
        usage_examples=[],
        security_config={},
        permissions=profile["actions"],
        scopes=["workspace", "odoo"],
        typed_errors=["ODOO_AUTH_FAILED", "ODOO_ACCESS_DENIED", "ODOO_MODEL_UNAVAILABLE", "ODOO_VALIDATION_ERROR"],
        audit_rules=["Registrar request_id, modelo, accion, payload_hash y record_id devuelto por Odoo."],
        retry_strategy="Retry corto solo para fallas transitorias sin mutacion confirmada.",
        timeout_policy="10s",
        failure_mode="Fallar cerrado y escalar si version, permisos o modelo no estan confirmados.",
        compensation_strategy="No repetir escrituras sin idempotency_key y recibo auditable.",
        approval_reason="Requiere aprobacion humana para escrituras comerciales en Odoo." if profile["side_effects"] else "",
        contract_review_state="connector-detected",
    )


def _tool_connector_profile_payload(tool: Any, index: int) -> dict[str, Any]:
    slug = _tool_connector_slug(tool, index)
    if _is_whatsapp_cloud_tool(tool):
        return {
            "schema_version": "tool-connector-profile.v1",
            "connector_key": "whatsapp_cloud_api",
            "label": "WhatsApp Business Cloud API",
            "source_tool_contract": _tool_contract_ref(tool, index),
            "tool_type": "external",
            "binding_category": "pending_binding",
            "provider_model": {
                "known_provider_required": True,
                "provider": "meta_whatsapp_cloud_api",
                "custom_provider_supported": True,
                "registered_api_ref": "whatsapp_cloud_api",
                "supported_binding_types": ["webhook_plus_rest_api", "rest_api", "webhook"],
            },
            "contract_surface": {
                "purpose": getattr(tool, "purpose", "") or "Canal conversacional inbound/outbound por WhatsApp Business.",
                "permission_mode": "write",
                "actions": [
                    "receive_inbound_message",
                    "send_session_message",
                    "send_template_message",
                    "receive_delivery_status",
                    "handoff_to_human",
                ],
                "inputs": getattr(tool, "request_schema", {}) or {},
                "outputs": getattr(tool, "response_schema", {}) or {},
                "when_to_use": getattr(tool, "when_to_use", "") or "",
            },
            "configuration_schema": {
                "required_fields": [
                    "binding_type",
                    "environment",
                    "phone_number_id_ref",
                    "business_account_id_ref",
                    "auth_secret_ref",
                    "webhook_verify_token_ref",
                    "app_secret_ref",
                    "callback_url_ref",
                    "enabled_actions",
                ],
                "expected_auth_schemes": ["bearer_token", "webhook_verify_token", "app_secret_signature"],
                "secret_policy": "references_only_no_plaintext_values",
                "environment_specific_bindings": True,
            },
            "governance": {
                "risk_level": getattr(tool, "risk_level", "") or "medium",
                "requires_approval": bool(getattr(tool, "requires_approval", False)),
                "approval_reason": getattr(tool, "approval_reason", "") or "Mensajes sensibles o iniciados por negocio requieren politica aprobada.",
                "allowed_roles": list(getattr(tool, "permissions", []) or ["send_whatsapp_message"]),
                "side_effects": True,
                "retry_policy": getattr(tool, "retry_strategy", "") or "Retry asincrono con circuit breaker por canal.",
                "timeout_policy": getattr(tool, "timeout_policy", "5000ms") or "5000ms",
                "failure_mode": getattr(tool, "failure_mode", "") or "Escalar a owner si el canal falla o falta binding.",
                "compensation_strategy": getattr(tool, "compensation_strategy", "") or "Evitar reenvios duplicados y registrar fallo.",
            },
        }
    google_key = _google_workspace_connector_key(tool)
    if google_key:
        profile = GOOGLE_WORKSPACE_CONNECTOR_PROFILES[google_key]
        return {
            "schema_version": "tool-connector-profile.v1",
            "connector_key": google_key,
            "label": profile["label"],
            "source_tool_contract": _tool_contract_ref(tool, index),
            "tool_type": "external",
            "binding_category": "pending_binding",
            "provider_model": {
                "known_provider_required": True,
                "provider": "google_workspace",
                "custom_provider_supported": True,
                "registered_api_ref": google_key,
                "supported_binding_types": ["oauth2_rest_api"],
                "lab_runtime_boundary": "LAB entrega contrato de construccion; no opera OAuth ni Google APIs por el usuario final.",
            },
            "contract_surface": {
                "purpose": getattr(tool, "purpose", "") or f"Conectar {profile['label']} con scopes minimos.",
                "permission_mode": "write" if profile["side_effects"] else "read",
                "actions": list(profile["actions"]),
                "inputs": getattr(tool, "request_schema", {}) or {},
                "outputs": getattr(tool, "response_schema", {}) or {},
                "when_to_use": getattr(tool, "when_to_use", "") or "",
                "contract_refs": list(profile["contract_refs"]),
            },
            "configuration_schema": {
                "required_fields": list(profile["required_fields"]),
                "expected_auth_schemes": ["oauth2"],
                "required_env_refs": list(profile["required_env_refs"]),
                "required_secret_refs": list(profile["required_secret_refs"]),
                "minimum_scope_policy": profile["minimum_scope_policy"],
                "secret_policy": "references_only_no_plaintext_values",
                "environment_specific_bindings": True,
            },
            "governance": {
                "risk_level": getattr(tool, "risk_level", "") or ("medium" if profile["side_effects"] else "low"),
                "requires_approval": bool(getattr(tool, "requires_approval", False) or profile["side_effects"]),
                "approval_reason": getattr(tool, "approval_reason", "") or profile["minimum_scope_policy"],
                "allowed_roles": list(getattr(tool, "permissions", []) or profile["actions"]),
                "side_effects": bool(getattr(tool, "has_side_effects", False) or profile["side_effects"]),
                "retry_policy": getattr(tool, "retry_strategy", "") or "Retry corto para 429/5xx con backoff y limite por workspace.",
                "timeout_policy": getattr(tool, "timeout_policy", "10s") or "10s",
                "failure_mode": getattr(tool, "failure_mode", "") or "Fallar cerrado cuando falten scopes, recurso permitido o token OAuth.",
                "compensation_strategy": getattr(tool, "compensation_strategy", "") or "No repetir side effects sin idempotency_key y evidencia.",
            },
        }
    odoo_key = _odoo_connector_key(tool)
    if odoo_key:
        profile = ODOO_CONNECTOR_PROFILES[odoo_key]
        return {
            "schema_version": "tool-connector-profile.v1",
            "connector_key": odoo_key,
            "label": profile["label"],
            "source_tool_contract": _tool_contract_ref(tool, index),
            "tool_type": "external",
            "binding_category": "pending_binding",
            "provider_model": {
                "known_provider_required": True,
                "provider": "odoo",
                "custom_provider_supported": True,
                "registered_api_ref": odoo_key,
                "supported_binding_types": ["xmlrpc_17_18", "json2_19"],
                "lab_runtime_boundary": "LAB entrega contrato de construccion; no opera Odoo ni almacena credenciales del usuario final.",
            },
            "contract_surface": {
                "purpose": getattr(tool, "purpose", "") or f"Conectar {profile['label']} con Odoo.",
                "permission_mode": "write" if profile["side_effects"] else "read",
                "actions": list(profile["actions"]),
                "model": profile["model"],
                "inputs": getattr(tool, "request_schema", {}) or {},
                "outputs": getattr(tool, "response_schema", {}) or {},
                "when_to_use": getattr(tool, "when_to_use", "") or "",
                "contract_refs": list(profile["contract_refs"]),
            },
            "configuration_schema": {
                "required_fields": list(profile["required_fields"]),
                "expected_auth_schemes": ["odoo_api_key_or_password"],
                "required_env_refs": list(profile["required_env_refs"]),
                "required_secret_refs": list(profile["required_secret_refs"]),
                "secret_policy": "references_only_no_plaintext_values",
                "environment_specific_bindings": True,
                "version_policy_ref": "ACP/integrations/odoo/version-policy.yaml",
            },
            "governance": {
                "risk_level": getattr(tool, "risk_level", "") or ("high" if profile["side_effects"] else "low"),
                "requires_approval": bool(getattr(tool, "requires_approval", False) or profile["side_effects"]),
                "approval_reason": getattr(tool, "approval_reason", "") or "Odoo writes require explicit owner approval, allowlisted model/action and idempotency.",
                "allowed_roles": list(getattr(tool, "permissions", []) or profile["actions"]),
                "side_effects": bool(getattr(tool, "has_side_effects", False) or profile["side_effects"]),
                "retry_policy": getattr(tool, "retry_strategy", "") or "Retry corto solo en fallas transitorias confirmadas sin duplicar writes.",
                "timeout_policy": getattr(tool, "timeout_policy", "10s") or "10s",
                "failure_mode": getattr(tool, "failure_mode", "") or "Fallar cerrado cuando falte version, modelo permitido, permiso Odoo o secret ref.",
                "compensation_strategy": getattr(tool, "compensation_strategy", "") or "No repetir writes sin idempotency_key; escalar fallo parcial al owner.",
            },
        }
    security_config = getattr(tool, "security_config", {}) or {}
    auth_scheme = str(security_config.get("auth_scheme") or security_config.get("type") or "").strip()
    expected_auth_schemes = [auth_scheme] if auth_scheme else ["bearer", "api_key", "oauth2", "basic", "service_account", "custom"]
    return {
        "schema_version": "tool-connector-profile.v1",
        "connector_key": slug,
        "label": getattr(tool, "name", "") or f"Tool {index}",
        "source_tool_contract": _tool_contract_ref(tool, index),
        "tool_type": getattr(tool, "tool_type", "external") or "external",
        "binding_category": _tool_binding_category(tool),
        "provider_model": {
            "known_provider_required": False,
            "custom_provider_supported": True,
            "registered_api_ref": getattr(tool, "registered_api_ref", "") or "",
            "supported_binding_types": ["rest_api", "webhook", "database", "low_code_adapter", "custom"],
        },
        "contract_surface": {
            "purpose": getattr(tool, "purpose", "") or "",
            "permission_mode": _tool_permission_mode(tool),
            "inputs": getattr(tool, "request_schema", {}) or {item: {"type": "string", "required": False} for item in getattr(tool, "inputs", [])},
            "outputs": getattr(tool, "response_schema", {}) or {item: {"type": "string"} for item in getattr(tool, "outputs", [])},
            "when_to_use": getattr(tool, "when_to_use", "") or "",
        },
        "configuration_schema": {
            "required_fields": ["binding_type", "environment", "base_url_ref", "auth_secret_ref", "enabled_actions"],
            "expected_auth_schemes": expected_auth_schemes,
            "secret_policy": "references_only_no_plaintext_values",
            "environment_specific_bindings": True,
        },
        "governance": {
            "risk_level": getattr(tool, "risk_level", "") or "needs_review",
            "requires_approval": bool(getattr(tool, "requires_approval", False)),
            "approval_reason": getattr(tool, "approval_reason", "") or "",
            "allowed_roles": list(getattr(tool, "permissions", []) or []),
            "side_effects": bool(getattr(tool, "has_side_effects", False)),
            "retry_policy": getattr(tool, "retry_strategy", "") or "needs_review",
            "timeout_policy": getattr(tool, "timeout_policy", "30s") or "30s",
            "failure_mode": getattr(tool, "failure_mode", "") or "",
            "compensation_strategy": getattr(tool, "compensation_strategy", "") or "",
        },
    }


def _tool_binding_payload(tool: Any, index: int, environment: str) -> dict[str, Any]:
    slug = _tool_connector_slug(tool, index)
    secret_key = _tool_secret_key(tool, index)
    tool_name = getattr(tool, "name", "") or f"tool_{index}"
    if _is_whatsapp_cloud_tool(tool):
        env_prefix = environment.upper()
        return {
            "schema_version": "tool-environment-binding.v1",
            "environment": environment,
            "connector_key": "whatsapp_cloud_api",
            "tool_name": tool_name,
            "source_tool_contract": _tool_contract_ref(tool, index),
            "binding_category": "pending_binding",
            "binding": {
                "binding_type": "webhook_plus_rest_api",
                "provider": "meta_whatsapp_cloud_api",
                "base_url_ref": "env:WHATSAPP_GRAPH_API_BASE_URL",
                "graph_api_version_ref": "env:WHATSAPP_GRAPH_API_VERSION",
                "phone_number_id_ref": "env:WHATSAPP_PHONE_NUMBER_ID",
                "business_account_id_ref": "env:WHATSAPP_BUSINESS_ACCOUNT_ID",
                "auth_secret_ref": "secret:WHATSAPP_ACCESS_TOKEN",
                "webhook_verify_token_ref": "secret:WHATSAPP_WEBHOOK_VERIFY_TOKEN",
                "app_secret_ref": "secret:WHATSAPP_APP_SECRET",
                "callback_url_ref": f"env:WHATSAPP_{env_prefix}_WEBHOOK_CALLBACK_URL",
                "enabled_actions": [
                    "receive_inbound_message",
                    "send_session_message",
                    "send_template_message",
                    "receive_delivery_status",
                ],
                "client_owned_contract": True,
            },
            "approval_overrides": {
                "requires_approval": bool(getattr(tool, "requires_approval", False)),
                "side_effects": True,
                "approval_reason": getattr(tool, "approval_reason", "") or "Usar templates aprobados y opt-in antes de mensajes iniciados por negocio.",
                "production_write_actions_require_named_owner": True,
            },
            "validation": {
                "webhook_contract_ref": "ACP/webhooks/whatsapp-business-webhook.yaml",
                "smoke_test_ref": "ACP/tools/tests/whatsapp-cloud-api-smoke-test.yaml",
                "contract_must_match_blueprint_tool": True,
                "fail_closed_when_binding_missing": environment == "production",
            },
            "notes": [
                "LAB entrega el contrato de construccion del webhook; no inventa URL publica, WABA, phone number ni secretos.",
                "No almacenar secretos planos en el ACP; usar referencias de entorno o vault.",
            ],
        }
    google_key = _google_workspace_connector_key(tool)
    if google_key:
        profile = GOOGLE_WORKSPACE_CONNECTOR_PROFILES[google_key]
        env_prefix = environment.upper()
        return {
            "schema_version": "tool-environment-binding.v1",
            "environment": environment,
            "connector_key": google_key,
            "tool_name": tool_name,
            "source_tool_contract": _tool_contract_ref(tool, index),
            "binding_category": "pending_binding",
            "binding": {
                "binding_type": "oauth2_rest_api",
                "provider": "google_workspace",
                "oauth_client_id_ref": "secret:GOOGLE_OAUTH_CLIENT_ID",
                "oauth_client_secret_ref": "secret:GOOGLE_OAUTH_CLIENT_SECRET",
                "refresh_token_ref": f"secret:GOOGLE_{env_prefix}_REFRESH_TOKEN_REF",
                "redirect_uri_ref": f"env:GOOGLE_{env_prefix}_OAUTH_REDIRECT_URI",
                "allowed_scopes_ref": "env:GOOGLE_ALLOWED_SCOPES",
                "enabled_actions": list(profile["actions"]),
                "resource_refs": {
                    "drive_file_policy_ref": "ACP/integrations/google-drive/file-picker.contract.yaml" if profile["service"] == "drive" else "",
                    "spreadsheet_id_ref": "env:GOOGLE_SHEETS_SPREADSHEET_ID" if google_key == "google_sheets_read_table" else "",
                    "calendar_id_ref": "env:GOOGLE_CALENDAR_DEFAULT_ID" if profile["service"] == "calendar" else "",
                    "sender_account_ref": "env:GMAIL_SENDER_ACCOUNT" if profile["service"] == "gmail" else "",
                },
                "client_owned_contract": True,
            },
            "approval_overrides": {
                "requires_approval": bool(getattr(tool, "requires_approval", False) or profile["side_effects"]),
                "side_effects": bool(profile["side_effects"]),
                "approval_reason": getattr(tool, "approval_reason", "") or profile["minimum_scope_policy"],
                "production_write_actions_require_named_owner": bool(profile["side_effects"]),
            },
            "validation": {
                "contract_refs": list(profile["contract_refs"]),
                "smoke_test_ref": f"ACP/tools/tests/{slug}-smoke-test.yaml",
                "contract_must_match_blueprint_tool": True,
                "fail_closed_when_binding_missing": environment == "production",
            },
            "notes": [
                "LAB entrega instrucciones y contratos; el builder debe crear/configurar OAuth en el proyecto destino.",
                "No almacenar client_secret, refresh_token ni otros secretos planos en el ACP.",
                "Usar scopes minimos y pedir decision si el caso requiere scopes sensibles o restringidos.",
            ],
        }
    odoo_key = _odoo_connector_key(tool)
    if odoo_key:
        profile = ODOO_CONNECTOR_PROFILES[odoo_key]
        return {
            "schema_version": "tool-environment-binding.v1",
            "environment": environment,
            "connector_key": odoo_key,
            "tool_name": tool_name,
            "source_tool_contract": _tool_contract_ref(tool, index),
            "binding_category": "pending_binding",
            "binding": {
                "binding_type": "versioned_rpc_api",
                "provider": "odoo",
                "api_mode_ref": "env:ODOO_API_MODE",
                "base_url_ref": "env:ODOO_BASE_URL",
                "database_ref": "env:ODOO_DATABASE",
                "username_ref": "secret:ODOO_USERNAME",
                "password_ref": "secret:ODOO_PASSWORD",
                "api_key_ref": "secret:ODOO_API_KEY",
                "allowed_models_ref": "env:ODOO_ALLOWED_MODELS",
                "allowed_write_actions_ref": "env:ODOO_ALLOWED_WRITE_ACTIONS",
                "model": profile["model"],
                "enabled_actions": list(profile["actions"]),
                "client_owned_contract": True,
            },
            "approval_overrides": {
                "requires_approval": bool(getattr(tool, "requires_approval", False) or profile["side_effects"]),
                "side_effects": bool(profile["side_effects"]),
                "approval_reason": getattr(tool, "approval_reason", "") or "Odoo writes require approval_gate, allowlist and idempotency.",
                "production_write_actions_require_named_owner": bool(profile["side_effects"]),
            },
            "validation": {
                "contract_refs": list(profile["contract_refs"]),
                "version_policy_ref": "ACP/integrations/odoo/version-policy.yaml",
                "smoke_test_ref": f"ACP/tools/tests/{slug}-smoke-test.yaml",
                "contract_must_match_blueprint_tool": True,
                "fail_closed_when_binding_missing": environment == "production",
            },
            "notes": [
                "LAB entrega instrucciones y contratos; el builder debe configurar Odoo en el proyecto destino.",
                "No almacenar base URL privada, usuario, password ni API key como valores planos en el ACP.",
                "Confirmar si el Odoo destino usa XML-RPC/JSON-RPC en 17/18 o JSON-2 en 19 antes de construir.",
            ],
        }
    return {
        "schema_version": "tool-environment-binding.v1",
        "environment": environment,
        "connector_key": slug,
        "tool_name": tool_name,
        "source_tool_contract": _tool_contract_ref(tool, index),
        "binding_category": "pending_binding",
        "binding": {
            "binding_type": "needs_review",
            "provider": "custom_or_client_specific",
            "base_url_ref": f"env:{secret_key}_{environment.upper()}_BASE_URL",
            "auth_secret_ref": f"secret:{secret_key}_{environment.upper()}_AUTH",
            "webhook_secret_ref": f"secret:{secret_key}_{environment.upper()}_WEBHOOK_SECRET",
            "enabled_actions": [tool_name],
            "client_owned_contract": True,
        },
        "approval_overrides": {
            "requires_approval": bool(getattr(tool, "requires_approval", False)),
            "side_effects": bool(getattr(tool, "has_side_effects", False)),
            "approval_reason": getattr(tool, "approval_reason", "") or "",
            "production_write_actions_require_named_owner": bool(getattr(tool, "has_side_effects", False)),
        },
        "validation": {
            "smoke_test_ref": f"ACP/tools/tests/{slug}-smoke-test.yaml",
            "contract_must_match_blueprint_tool": True,
            "fail_closed_when_binding_missing": environment == "production",
        },
        "notes": [
            "Completar este binding con el contrato real del cliente antes de activar la tool.",
            "No almacenar secretos planos en el ACP; usar referencias de entorno o vault.",
        ],
    }


def _tool_smoke_test_payload(tool: Any, index: int) -> dict[str, Any]:
    slug = _tool_connector_slug(tool, index)
    if _is_whatsapp_cloud_tool(tool):
        return {
            "schema_version": "tool-smoke-test.v1",
            "connector_key": "whatsapp_cloud_api",
            "tool_name": getattr(tool, "name", "") or "whatsapp_business_messaging",
            "source_tool_contract": _tool_contract_ref(tool, index),
            "checks": [
                {"check": "binding_resolves", "description": "El binding sandbox/production existe y usa referencias, no secretos planos."},
                {"check": "webhook_verify_accepts_valid_challenge", "description": "GET /webhooks/whatsapp retorna hub.challenge cuando verify token coincide."},
                {"check": "webhook_verify_rejects_invalid_token", "description": "GET /webhooks/whatsapp retorna 403 cuando verify token no coincide."},
                {"check": "webhook_signature_validation", "description": "POST /webhooks/whatsapp valida X-Hub-Signature-256 cuando existe app secret."},
                {"check": "inbound_payload_normalizes", "description": "El payload inbound se transforma a InboundMessage normalizado."},
                {"check": "message_id_idempotency", "description": "Un provider message_id duplicado no dispara dos veces el agente."},
                {"check": "template_send_uses_approved_template", "description": "Los mensajes iniciados por negocio usan templates aprobados."},
                {"check": "typed_errors_map_provider_failures", "description": "Errores de Meta/proveedor se mapean a typed_errors del contrato."},
            ],
            "expected_result": "ready_for_sandbox_activation_after_all_checks_pass",
        }
    google_key = _google_workspace_connector_key(tool)
    if google_key:
        profile = GOOGLE_WORKSPACE_CONNECTOR_PROFILES[google_key]
        checks = [
            {"check": "binding_resolves", "description": "El binding sandbox/production existe y usa referencias, no secretos planos."},
            {"check": "oauth_client_resolves", "description": "Client ID, client secret y redirect URI estan referenciados desde vault/env."},
            {"check": "scopes_match_minimum_policy", "description": "Los scopes solicitados coinciden con la matriz aprobada para esta tool."},
            {"check": "resource_allowlist_enforced", "description": "Solo se accede a archivos, hojas, calendarios o cuentas aprobadas."},
            {"check": "typed_errors_map_google_failures", "description": "401/403/404/429/5xx se mapean a typed_errors del contrato."},
        ]
        if profile["side_effects"]:
            checks.extend(
                [
                    {"check": "approval_gate_enforced", "description": "La accion con side effect exige approval_gate o politica aprobada."},
                    {"check": "idempotency_enforced", "description": "Reintentos no crean eventos/correos duplicados."},
                ]
            )
        return {
            "schema_version": "tool-smoke-test.v1",
            "connector_key": google_key,
            "tool_name": getattr(tool, "name", "") or google_key,
            "source_tool_contract": _tool_contract_ref(tool, index),
            "checks": checks,
            "expected_result": "ready_for_sandbox_activation_after_oauth_and_scope_checks_pass",
        }
    odoo_key = _odoo_connector_key(tool)
    if odoo_key:
        profile = ODOO_CONNECTOR_PROFILES[odoo_key]
        checks = [
            {"check": "binding_resolves", "description": "El binding sandbox/production existe y usa referencias, no secretos planos."},
            {"check": "version_policy_selected", "description": "ODOO_API_MODE coincide con la version real: XML-RPC/JSON-RPC para 17/18 o JSON-2 para 19."},
            {"check": "auth_and_database_resolve", "description": "Base URL, database, usuario y secret ref existen en el vault/runtime seleccionado."},
            {"check": "allowed_model_enforced", "description": f"Solo se permite operar el modelo {profile['model']} u otros modelos aprobados."},
            {"check": "access_rights_validated", "description": "El usuario tecnico tiene permisos Odoo minimos para la accion."},
            {"check": "typed_errors_map_odoo_failures", "description": "401/403/404/429/5xx y errores de modelo se mapean a typed_errors del contrato."},
        ]
        if profile["side_effects"]:
            checks.extend(
                [
                    {"check": "approval_gate_enforced", "description": "La accion con side effect exige approval_gate o politica aprobada."},
                    {"check": "write_action_allowlisted", "description": "La accion aparece en ODOO_ALLOWED_WRITE_ACTIONS antes de ejecutar."},
                    {"check": "idempotency_enforced", "description": "Reintentos no crean cotizaciones, actividades ni updates duplicados."},
                ]
            )
        return {
            "schema_version": "tool-smoke-test.v1",
            "connector_key": odoo_key,
            "tool_name": getattr(tool, "name", "") or odoo_key,
            "source_tool_contract": _tool_contract_ref(tool, index),
            "checks": checks,
            "expected_result": "ready_for_sandbox_activation_after_version_auth_model_and_approval_checks_pass",
        }
    return {
        "schema_version": "tool-smoke-test.v1",
        "connector_key": slug,
        "tool_name": getattr(tool, "name", "") or f"tool_{index}",
        "source_tool_contract": _tool_contract_ref(tool, index),
        "checks": [
            {"check": "binding_resolves", "description": "El binding del entorno existe y apunta a referencias, no secretos planos."},
            {"check": "auth_reference_resolves", "description": "La referencia de autenticacion existe en el runtime o vault seleccionado."},
            {"check": "endpoint_reachable", "description": "El endpoint o adaptador responde dentro del timeout definido."},
            {"check": "input_schema_validates", "description": "Los inputs enviados cumplen el contrato de la tool."},
            {"check": "output_schema_validates", "description": "La respuesta puede mapearse al contrato esperado."},
            {"check": "approval_gate_enforced", "description": "Las acciones con side effects no corren sin aprobacion cuando aplica."},
            {"check": "fallback_path_available", "description": "Existe salida de error, compensacion o escalamiento humano."},
        ],
        "expected_result": "ready_for_environment_activation_after_all_checks_pass",
    }


def _build_tool_connector_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []

    catalog_items: list[dict[str, Any]] = []
    files: list[ACPFileEntry] = []

    def append_connector_files(tool: Any, index: int, *, warnings: list[str] | None = None) -> None:
        slug = _tool_connector_slug(tool, index)
        catalog_items.append(
            {
                "connector_key": slug,
                "tool_name": getattr(tool, "name", "") or f"tool_{index}",
                "profile_ref": f"ACP/tools/connectors/{slug}.yaml",
                "contract_ref": _tool_contract_ref(tool, index),
                "sandbox_binding_ref": f"ACP/tools/bindings/{slug}.sandbox.yaml",
                "production_binding_ref": f"ACP/tools/bindings/{slug}.production.yaml",
                "smoke_test_ref": f"ACP/tools/tests/{slug}-smoke-test.yaml",
                "custom_provider_supported": True,
                "requires_approval": bool(getattr(tool, "requires_approval", False)),
                "side_effects": bool(getattr(tool, "has_side_effects", False)),
            }
        )
        files.append(
            build_acp_file_entry(
                path=f"ACP/tools/connectors/{slug}.yaml",
                domain="tools",
                title=f"Connector profile: {getattr(tool, 'name', '') or slug}",
                format="yaml",
                source_sections=["blueprint.tools", "tool_contracts", "client_integrations"],
                content_text=serialize_yaml_document(_tool_connector_profile_payload(tool, index)),
                warnings=warnings or ["Perfil generico: completar proveedor, autenticacion y acciones segun la herramienta real del cliente."],
            )
        )
        for environment in ("sandbox", "production"):
            files.append(
                build_acp_file_entry(
                    path=f"ACP/tools/bindings/{slug}.{environment}.yaml",
                    domain="tools",
                    title=f"{environment.title()} binding: {getattr(tool, 'name', '') or slug}",
                    format="yaml",
                    source_sections=["blueprint.tools", "runtime_secrets", "deployment_environment"],
                    content_text=serialize_yaml_document(_tool_binding_payload(tool, index, environment)),
                    warnings=["Binding pendiente de completar con referencias reales del entorno antes de activar la tool."],
                )
            )
        files.append(
            build_acp_file_entry(
                path=f"ACP/tools/tests/{slug}-smoke-test.yaml",
                domain="tools",
                title=f"Smoke test: {getattr(tool, 'name', '') or slug}",
                format="yaml",
                source_sections=["blueprint.tools", "evaluation", "release_readiness"],
                content_text=serialize_yaml_document(_tool_smoke_test_payload(tool, index)),
            )
        )

    tools = project_blueprint_tools_for_construction(snapshot)
    for index, tool in enumerate(tools, start=1):
        append_connector_files(tool, index)

    existing_odoo_keys = {
        key
        for key in (_odoo_connector_key(tool) for tool in tools)
        if key
    }
    inferred_odoo_keys = _odoo_connector_keys_for_snapshot(snapshot) - existing_odoo_keys
    base_tool_count = len(tools)
    for offset, key in enumerate(sorted(inferred_odoo_keys), start=1):
        append_connector_files(
            _synthetic_odoo_tool(key, index=base_tool_count + offset),
            base_tool_count + offset,
            warnings=[
                "Connector Odoo inferido desde senales legacy del Blueprint; confirmar version, modelos, permisos y alcance antes de construir."
            ],
        )

    catalog_payload = {
        "schema_version": "tool-connector-catalog.v1",
        "custom_client_tools_supported": True,
        "known_connectors_are_examples_not_limits": True,
        "items": catalog_items,
    }
    files.insert(
        0,
        build_acp_file_entry(
            path="ACP/tools/connectors/catalog.yaml",
            domain="tools",
            title="Tool connector catalog",
            format="yaml",
            source_sections=["blueprint.tools", "tool_contracts"],
            content_text=serialize_yaml_document(catalog_payload),
        ),
    )
    return files


def _build_whatsapp_connector_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None or not blueprint.tools:
        return []
    tools = project_blueprint_tools_for_construction(snapshot)
    whatsapp_tools = [(index, tool) for index, tool in enumerate(tools, start=1) if _is_whatsapp_cloud_tool(tool)]
    if not whatsapp_tools:
        return []
    index, tool = whatsapp_tools[0]
    tool_contract_ref = _tool_contract_ref(tool, index)
    webhook_contract = {
        "schema_version": "webhook-contract.v1",
        "webhook_key": "whatsapp_business_webhook",
        "provider": "meta_whatsapp_cloud_api",
        "callback_route": "/webhooks/whatsapp",
        "lab_boundary": "LAB entrega este contrato; el builder implementa el endpoint en el proyecto destino.",
        "methods": {
            "verify": {
                "method": "GET",
                "purpose": "Meta webhook verification challenge",
                "query_params": ["hub.mode", "hub.verify_token", "hub.challenge"],
                "expected_behavior": [
                    "compare hub.verify_token against WHATSAPP_WEBHOOK_VERIFY_TOKEN_REF",
                    "return hub.challenge when valid",
                    "return 403 when invalid",
                ],
            },
            "receive": {
                "method": "POST",
                "purpose": "Receive inbound messages and status events",
                "required_headers": ["X-Hub-Signature-256"],
                "validation": [
                    "verify signature with WHATSAPP_APP_SECRET_REF when available",
                    "reject malformed payloads",
                    "dedupe by provider message id",
                ],
                "response": {"success_status": 200, "failure_statuses": [400, 401, 403, 500]},
            },
        },
        "normalized_event": {
            "type": "InboundMessage",
            "fields": [
                "provider",
                "workspace_id",
                "wa_id",
                "phone_number_id",
                "message_id",
                "timestamp",
                "message_type",
                "text",
                "media_ref",
                "raw_payload_ref",
            ],
        },
        "idempotency": {"key": "message_id", "duplicate_behavior": "ignore_and_ack"},
        "source_tool_contract": tool_contract_ref,
    }
    send_message_contract = {
        "schema_version": "whatsapp-send-message-contract.v1",
        "connector_key": "whatsapp_cloud_api",
        "provider": "meta_whatsapp_cloud_api",
        "endpoint": "https://graph.facebook.com/{version}/{phone_number_id}/messages",
        "actions": {
            "send_session_message": {
                "requires": ["wa_id", "text", "WHATSAPP_ACCESS_TOKEN_REF", "WHATSAPP_PHONE_NUMBER_ID"],
                "policy": "Usar solo cuando exista ventana conversacional o regla aprobada por el owner.",
            },
            "send_template_message": {
                "requires": ["wa_id", "template_name", "template_language", "template_variables", "WHATSAPP_ACCESS_TOKEN_REF"],
                "policy": "Usar templates aprobados para mensajes iniciados por negocio.",
            },
        },
        "typed_errors": [
            "WHATSAPP_AUTH_EXPIRED",
            "WHATSAPP_TEMPLATE_NOT_APPROVED",
            "WHATSAPP_DELIVERY_FAILED",
            "WHATSAPP_RATE_LIMITED",
        ],
        "audit": ["provider_message_id", "wa_id_hash", "template_name", "delivery_status"],
        "source_tool_contract": tool_contract_ref,
    }
    templates_contract = {
        "schema_version": "whatsapp-template-contract.v1",
        "connector_key": "whatsapp_cloud_api",
        "required_owner_input": True,
        "templates": [
            {
                "template_name": "needs_client_answer",
                "category": "needs_client_answer",
                "language": "needs_client_answer",
                "variables": [],
                "approval_owner": "marketing_or_ops_owner",
            }
        ],
        "rules": [
            "No inventar templates en el ACP.",
            "Usar solo nombres, idiomas y variables aprobadas por el cliente/proveedor.",
            "Mensajes sensibles o comerciales requieren politica de opt-in y aprobacion cuando aplique.",
        ],
    }
    return [
        build_acp_file_entry(
            path="ACP/webhooks/whatsapp-business-webhook.yaml",
            domain="integrations",
            title="WhatsApp Business webhook contract",
            format="yaml",
            source_sections=["blueprint.tools", "client_integrations"],
            content_text=serialize_yaml_document(webhook_contract),
        ),
        build_acp_file_entry(
            path="ACP/integrations/whatsapp/send-message.contract.yaml",
            domain="integrations",
            title="WhatsApp send message contract",
            format="yaml",
            source_sections=["blueprint.tools", "tool_contracts"],
            content_text=serialize_yaml_document(send_message_contract),
        ),
        build_acp_file_entry(
            path="ACP/integrations/whatsapp/templates.yaml",
            domain="integrations",
            title="WhatsApp template contract",
            format="yaml",
            source_sections=["blueprint.tools", "construction_readiness.gaps.questions"],
            content_text=serialize_yaml_document(templates_contract),
            warnings=["Completar templates aprobados antes de activar mensajes iniciados por negocio."],
        ),
    ]


def _build_google_workspace_connector_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None or not blueprint.tools:
        return []
    google_tools = [
        (index, tool, _google_workspace_connector_key(tool))
        for index, tool in enumerate(project_blueprint_tools_for_construction(snapshot), start=1)
        if _is_google_workspace_tool(tool)
    ]
    if not google_tools:
        return []

    keys = {key for _, _, key in google_tools}
    files: list[ACPFileEntry] = []
    oauth_policy = {
        "schema_version": "google-workspace-oauth-policy.v1",
        "lab_boundary": "LAB entrega contrato ACP; el builder configura OAuth y Google APIs en el proyecto destino.",
        "secret_policy": "references_only_no_plaintext_values",
        "required_common_refs": [
            "GOOGLE_OAUTH_CLIENT_ID",
            "GOOGLE_OAUTH_CLIENT_SECRET",
            "GOOGLE_REFRESH_TOKEN_REF",
            "GOOGLE_OAUTH_REDIRECT_URI",
            "GOOGLE_ALLOWED_SCOPES",
        ],
        "lifecycle": {
            "pending": "falta respuesta del cliente o owner tecnico",
            "answered": "dato capturado como referencia, no valor secreto plano",
            "delegated": "decision transferida al builder antes de construir",
            "resolved": "binding validado en entorno sandbox/production",
            "reopened": "scope/recurso cambió y requiere nueva aprobación",
        },
        "rules": [
            "Usar scopes minimos por tool.",
            "Pedir decision explicita antes de scopes sensibles/restringidos.",
            "No mezclar OAuth de LAB con OAuth del producto construido para el cliente.",
            "Fallar cerrado cuando falte recurso permitido, refresh token o scope aprobado.",
        ],
    }
    scopes_matrix = {
        "schema_version": "google-workspace-scopes-matrix.v1",
        "tools": [
            {
                "connector_key": key,
                "label": GOOGLE_WORKSPACE_CONNECTOR_PROFILES[key]["label"],
                "minimum_scope_policy": GOOGLE_WORKSPACE_CONNECTOR_PROFILES[key]["minimum_scope_policy"],
                "side_effects": GOOGLE_WORKSPACE_CONNECTOR_PROFILES[key]["side_effects"],
                "approval_required": GOOGLE_WORKSPACE_CONNECTOR_PROFILES[key]["side_effects"],
                "scope_values": "to_be_selected_by_builder_from_google_docs_and_client_policy",
            }
            for key in sorted(keys)
        ],
    }
    risk_notes = {
        "schema_version": "google-workspace-connector-risk-notes.v1",
        "cost_note": "Uso estandar de APIs publicas Google Workspace suele no requerir costo adicional bajo cuotas; controlar volumen, cache y retries.",
        "verification_note": "Algunos scopes sensibles/restringidos pueden requerir verificacion OAuth; el ACP debe dejarlo como decision visible.",
        "not_in_scope": [
            "LAB no ejecuta OAuth ni almacena tokens del cliente final.",
            "LAB no lee Drive, Gmail, Calendar ni Sheets por cuenta propia.",
            "El builder debe implementar consent screen, redirect URI, vault y revocacion.",
        ],
    }
    files.extend(
        [
            build_acp_file_entry(
                path="ACP/integrations/google-workspace/oauth-policy.yaml",
                domain="integrations",
                title="Google Workspace OAuth policy",
                format="yaml",
                source_sections=["blueprint.tools", "construction_readiness.gaps.questions"],
                content_text=serialize_yaml_document(oauth_policy),
            ),
            build_acp_file_entry(
                path="ACP/integrations/google-workspace/scopes-matrix.yaml",
                domain="integrations",
                title="Google Workspace scopes matrix",
                format="yaml",
                source_sections=["blueprint.tools", "tool_contracts"],
                content_text=serialize_yaml_document(scopes_matrix),
                warnings=["Validar scopes finales contra documentacion de Google y politica del cliente antes de construir."],
            ),
            build_acp_file_entry(
                path="ACP/integrations/google-workspace/connector-risk-notes.yaml",
                domain="integrations",
                title="Google Workspace connector risk notes",
                format="yaml",
                source_sections=["blueprint.tools", "risk_summary"],
                content_text=serialize_yaml_document(risk_notes),
            ),
        ]
    )

    if "google_drive_file_picker" in keys:
        file_picker_contract = {
            "schema_version": "google-drive-file-picker-contract.v1",
            "connector_key": "google_drive_file_picker",
            "purpose": "Permitir que el usuario seleccione archivos concretos de Drive para lectura/ingesta.",
            "preferred_scope_policy": "drive.file via Google Picker",
            "actions": {
                "picker_select_file": {"requires": ["oauth_client", "picker_api_key_or_app_config", "allowed_mime_types"]},
                "read_file_metadata": {"endpoint": "GET /drive/v3/files/{fileId}", "requires": ["file_id_allowlist"]},
                "download_or_export_selected_file": {"requires": ["file_id", "mime_type", "export_format_if_google_native"]},
            },
            "unknowns_to_ask": ["allowed_mime_types", "max_file_size_mb", "refresh_mode", "owner_email_domain_policy"],
            "source_tool_contracts": [
                _tool_contract_ref(tool, index)
                for index, tool, key in google_tools
                if key == "google_drive_file_picker"
            ],
        }
        files.extend(
            [
                build_acp_file_entry(
                    path="ACP/integrations/google-drive/file-picker.contract.yaml",
                    domain="integrations",
                    title="Google Drive file picker contract",
                    format="yaml",
                    source_sections=["blueprint.tools", "tool_contracts"],
                    content_text=serialize_yaml_document(file_picker_contract),
                ),
                build_acp_file_entry(
                    path="ACP/integrations/google-drive/selected-file-reader.contract.yaml",
                    domain="integrations",
                    title="Google Drive selected file reader contract",
                    format="yaml",
                    source_sections=["blueprint.tools", "knowledge"],
                    content_text=serialize_yaml_document(
                        {
                            "schema_version": "google-drive-selected-file-reader.v1",
                            "connector_key": "google_drive_file_picker",
                            "read_policy": "Solo archivos seleccionados/permitidos; no crawler general de Drive en MVP.",
                            "normalization": ["metadata", "download_or_export", "parse", "chunk_if_rag_required", "audit_source_revision"],
                            "typed_errors": ["DRIVE_FILE_NOT_FOUND", "DRIVE_SCOPE_NOT_GRANTED", "FILE_TOO_LARGE", "PARSER_FAILURE"],
                        }
                    ),
                ),
            ]
        )

    if "google_sheets_read_table" in keys:
        files.extend(
            [
                build_acp_file_entry(
                    path="ACP/integrations/google-sheets/read-table.contract.yaml",
                    domain="integrations",
                    title="Google Sheets read table contract",
                    format="yaml",
                    source_sections=["blueprint.tools", "tool_contracts"],
                    content_text=serialize_yaml_document(
                        {
                            "schema_version": "google-sheets-read-table-contract.v1",
                            "connector_key": "google_sheets_read_table",
                            "endpoint": "GET https://sheets.googleapis.com/v4/spreadsheets/{spreadsheetId}/values/{range}",
                            "required_configuration": ["spreadsheet_id", "range", "header_row", "column_schema", "cache_policy"],
                            "read_policy": "Solo lectura tabular sobre rangos permitidos.",
                            "unknowns_to_ask": ["spreadsheet_id", "range", "column_schema", "owner", "refresh_frequency"],
                        }
                    ),
                ),
                build_acp_file_entry(
                    path="ACP/integrations/google-sheets/schema-mapping.yaml",
                    domain="integrations",
                    title="Google Sheets schema mapping",
                    format="yaml",
                    source_sections=["blueprint.tools", "construction_readiness.gaps.questions"],
                    content_text=serialize_yaml_document(
                        {
                            "schema_version": "google-sheets-schema-mapping.v1",
                            "columns": "needs_client_answer",
                            "primary_key": "needs_client_answer",
                            "required_columns": [],
                            "normalization_rules": [],
                        }
                    ),
                    warnings=["Completar columnas y llave primaria antes de construir lookup real sobre Sheets."],
                ),
                build_acp_file_entry(
                    path="ACP/integrations/google-sheets/data-quality-rules.yaml",
                    domain="integrations",
                    title="Google Sheets data quality rules",
                    format="yaml",
                    source_sections=["blueprint.tools", "validation"],
                    content_text=serialize_yaml_document(
                        {
                            "schema_version": "google-sheets-data-quality.v1",
                            "rules": ["reject_empty_header", "dedupe_by_primary_key_when_defined", "limit_rows_per_request", "log_schema_drift"],
                        }
                    ),
                ),
            ]
        )

    if {"google_calendar_availability_reader", "google_calendar_event_creator"} & keys:
        calendar_files = []
        if "google_calendar_availability_reader" in keys:
            calendar_files.append(
                build_acp_file_entry(
                    path="ACP/integrations/google-calendar/availability.contract.yaml",
                    domain="integrations",
                    title="Google Calendar availability contract",
                    format="yaml",
                    source_sections=["blueprint.tools", "tool_contracts"],
                    content_text=serialize_yaml_document(
                        {
                            "schema_version": "google-calendar-availability-contract.v1",
                            "connector_key": "google_calendar_availability_reader",
                            "endpoint": "POST https://www.googleapis.com/calendar/v3/freeBusy",
                            "required_configuration": ["calendar_ids", "timezone", "availability_window", "slot_duration_minutes"],
                            "privacy_policy": "No exponer detalles de eventos privados; devolver busy/available slots.",
                        }
                    ),
                )
            )
        if "google_calendar_event_creator" in keys:
            calendar_files.extend(
                [
                    build_acp_file_entry(
                        path="ACP/integrations/google-calendar/create-event.contract.yaml",
                        domain="integrations",
                        title="Google Calendar create event contract",
                        format="yaml",
                        source_sections=["blueprint.tools", "tool_contracts"],
                        content_text=serialize_yaml_document(
                            {
                                "schema_version": "google-calendar-create-event-contract.v1",
                                "connector_key": "google_calendar_event_creator",
                                "endpoint": "POST https://www.googleapis.com/calendar/v3/calendars/{calendarId}/events",
                                "required_configuration": ["calendar_id", "timezone", "attendee_policy", "idempotency_key"],
                                "side_effects": True,
                                "unknowns_to_ask": ["calendar_id", "timezone", "attendee_policy", "approval_policy", "conflict_policy"],
                            }
                        ),
                    ),
                    build_acp_file_entry(
                        path="ACP/integrations/google-calendar/approval-policy.yaml",
                        domain="integrations",
                        title="Google Calendar approval policy",
                        format="yaml",
                        source_sections=["blueprint.tools", "approvals"],
                        content_text=serialize_yaml_document(
                            {
                                "schema_version": "google-calendar-approval-policy.v1",
                                "create_event_requires": ["valid_availability_check", "idempotency_key", "approved_attendee_policy"],
                                "human_approval_required_when": ["external_attendees", "paid_service", "sensitive_context", "policy_unknown"],
                            }
                        ),
                    ),
                ]
            )
        files.extend(calendar_files)

    if {"gmail_draft_creator", "gmail_send_message"} & keys:
        if "gmail_draft_creator" in keys:
            files.append(
                build_acp_file_entry(
                    path="ACP/integrations/gmail/create-draft.contract.yaml",
                    domain="integrations",
                    title="Gmail create draft contract",
                    format="yaml",
                    source_sections=["blueprint.tools", "tool_contracts"],
                    content_text=serialize_yaml_document(
                        {
                            "schema_version": "gmail-create-draft-contract.v1",
                            "connector_key": "gmail_draft_creator",
                            "endpoint": "POST https://gmail.googleapis.com/gmail/v1/users/{userId}/drafts",
                            "required_configuration": ["sender_account", "recipient_policy", "draft_review_policy"],
                            "preferred_policy": "Crear borrador y dejar revision humana antes de envio.",
                        }
                    ),
                )
            )
        if "gmail_send_message" in keys:
            files.append(
                build_acp_file_entry(
                    path="ACP/integrations/gmail/send-message.contract.yaml",
                    domain="integrations",
                    title="Gmail send message contract",
                    format="yaml",
                    source_sections=["blueprint.tools", "tool_contracts"],
                    content_text=serialize_yaml_document(
                        {
                            "schema_version": "gmail-send-message-contract.v1",
                            "connector_key": "gmail_send_message",
                            "endpoint": "POST https://gmail.googleapis.com/gmail/v1/users/{userId}/messages/send",
                            "required_configuration": ["sender_account", "recipient_policy", "approval_policy", "idempotency_key"],
                            "side_effects": True,
                            "human_approval_required_when": ["message_initiated_by_business", "recipient_policy_unknown", "sensitive_content"],
                        }
                    ),
                    warnings=["Gmail send requiere politica explicita; preferir borrador si no hay aprobacion."],
                )
            )
        files.append(
            build_acp_file_entry(
                path="ACP/integrations/gmail/restricted-scope-warning.yaml",
                domain="integrations",
                title="Gmail scope warning",
                format="yaml",
                source_sections=["blueprint.tools", "risk_summary"],
                content_text=serialize_yaml_document(
                    {
                        "schema_version": "gmail-scope-warning.v1",
                        "rule": "Evitar lectura amplia de inbox en MVP. Usar gmail.compose/gmail.send solo si el cliente aprueba el alcance.",
                        "requires_review": ["oauth_consent_screen", "scope_sensitivity", "recipient_policy", "data_retention_policy"],
                    }
                ),
            )
        )

    return files


def _build_odoo_connector_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None or not blueprint.tools:
        return []
    keys = _odoo_connector_keys_for_snapshot(snapshot)
    if not keys:
        return []

    models = sorted({ODOO_CONNECTOR_PROFILES[key]["model"] for key in keys})
    has_write = any(ODOO_CONNECTOR_PROFILES[key]["side_effects"] for key in keys)
    files: list[ACPFileEntry] = []
    api_profile = {
        "schema_version": "odoo-api-profile.v1",
        "provider": "odoo",
        "lab_boundary": "LAB entrega contrato ACP; el builder configura Odoo, permisos y credenciales en el proyecto destino.",
        "supported_versions": {
            "odoo_17_18": "External API via XML-RPC/JSON-RPC execute_kw.",
            "odoo_19": "External JSON-2 API where available; legacy XML-RPC/JSON-RPC is planned for removal.",
        },
        "required_common_refs": [
            "ODOO_API_MODE",
            "ODOO_BASE_URL",
            "ODOO_DATABASE",
            "ODOO_USERNAME",
            "ODOO_PASSWORD",
            "ODOO_API_KEY",
            "ODOO_ALLOWED_MODELS",
            "ODOO_ALLOWED_WRITE_ACTIONS",
        ],
        "secret_policy": "references_only_no_plaintext_values",
        "not_in_scope": [
            "LAB no inicia sesion en Odoo del cliente.",
            "LAB no descubre modelos custom en vivo.",
            "LAB no confirma permisos ni modulos instalados sin respuesta del owner tecnico.",
        ],
    }
    version_policy = {
        "schema_version": "odoo-version-policy.v1",
        "selection_required": True,
        "allowed_api_modes": ["xmlrpc_17_18", "json2_19"],
        "rules": [
            "Si version es Odoo 17 u 18, construir adaptador con execute_kw sobre /xmlrpc/2/common y /xmlrpc/2/object, o equivalente JSON-RPC si el proyecto lo decide.",
            "Si version es Odoo 19, preferir JSON-2 y validar endpoints por modelo antes de escribir codigo.",
            "No mezclar modos en una misma tool sin decision explicita.",
            "Si la version es desconocida, dejar binding en pending y preguntar odoo_version_context.",
        ],
        "source_docs": [
            "https://www.odoo.com/documentation/17.0/developer/reference/external_api.html",
            "https://www.odoo.com/documentation/18.0/developer/reference/external_api.html",
            "https://www.odoo.com/documentation/19.0/developer/reference/external_api.html",
        ],
    }
    rpc_contract = {
        "schema_version": "odoo-rpc-17-18-contract.v1",
        "api_mode": "xmlrpc_17_18",
        "common_endpoint": "{ODOO_BASE_URL}/xmlrpc/2/common",
        "object_endpoint": "{ODOO_BASE_URL}/xmlrpc/2/object",
        "auth_flow": ["authenticate database, username and password/api_key", "execute_kw only on allowed models/actions"],
        "read_actions": ["search_read", "read"],
        "write_actions": ["create", "write"] if has_write else [],
        "required_guards": ["allowed_model_validation", "field_allowlist_validation", "access_rights_check", "typed_error_mapping"],
    }
    json2_contract = {
        "schema_version": "odoo-json2-19-contract.v1",
        "api_mode": "json2_19",
        "base_endpoint_pattern": "{ODOO_BASE_URL}/json/2/{model}/{method}",
        "read_actions": ["search_read", "read"],
        "write_actions": ["create", "write"] if has_write else [],
        "required_guards": ["allowed_model_validation", "field_allowlist_validation", "access_rights_check", "typed_error_mapping"],
        "activation_note": "Validar disponibilidad real de JSON-2 en el Odoo destino antes de construir.",
    }
    models_scope = {
        "schema_version": "odoo-models-scope.v1",
        "allowed_models": [
            {
                "model": model,
                "tools": [
                    key
                    for key in sorted(keys)
                    if ODOO_CONNECTOR_PROFILES[key]["model"] == model
                ],
                "allowed_fields": "needs_client_answer",
                "allowed_domains": "needs_client_answer",
            }
            for model in models
        ],
        "custom_models_policy": "No usar modelos custom sin respuesta explicita del owner tecnico y evidencia del modulo instalado.",
    }
    files.extend(
        [
            build_acp_file_entry(
                path="ACP/integrations/odoo/api-profile.yaml",
                domain="integrations",
                title="Odoo API profile",
                format="yaml",
                source_sections=["blueprint.tools", "construction_readiness.gaps.questions"],
                content_text=serialize_yaml_document(api_profile),
            ),
            build_acp_file_entry(
                path="ACP/integrations/odoo/version-policy.yaml",
                domain="integrations",
                title="Odoo version policy",
                format="yaml",
                source_sections=["blueprint.tools", "tool_contracts"],
                content_text=serialize_yaml_document(version_policy),
                warnings=["Confirmar version Odoo y API mode antes de construir adaptadores."],
            ),
            build_acp_file_entry(
                path="ACP/integrations/odoo/rpc-17-18.contract.yaml",
                domain="integrations",
                title="Odoo RPC 17/18 contract",
                format="yaml",
                source_sections=["blueprint.tools", "tool_contracts"],
                content_text=serialize_yaml_document(rpc_contract),
            ),
            build_acp_file_entry(
                path="ACP/integrations/odoo/json2-19.contract.yaml",
                domain="integrations",
                title="Odoo JSON-2 19 contract",
                format="yaml",
                source_sections=["blueprint.tools", "tool_contracts"],
                content_text=serialize_yaml_document(json2_contract),
            ),
            build_acp_file_entry(
                path="ACP/integrations/odoo/models-scope.yaml",
                domain="integrations",
                title="Odoo models scope",
                format="yaml",
                source_sections=["blueprint.tools", "construction_readiness.gaps.questions"],
                content_text=serialize_yaml_document(models_scope),
                warnings=["Completar campos, dominios y permisos por modelo antes de construir calls reales."],
            ),
        ]
    )
    if "odoo_sale_quote_create" in keys:
        files.append(
            build_acp_file_entry(
                path="ACP/integrations/odoo/quote-policy.yaml",
                domain="integrations",
                title="Odoo quote policy",
                format="yaml",
                source_sections=["blueprint.tools", "approvals", "risk_summary"],
                content_text=serialize_yaml_document(
                    {
                        "schema_version": "odoo-quote-policy.v1",
                        "connector_key": "odoo_sale_quote_create",
                        "model": "sale.order",
                        "requires": ["valid_partner_id", "valid_product_ids", "pricelist_policy", "tax_policy", "approval_token", "idempotency_key"],
                        "unknowns_to_ask": ["pricelist", "currency", "taxes", "discount_policy", "quote_expiration", "confirmation_policy"],
                        "rules": [
                            "Crear cotizacion en borrador salvo aprobacion explicita para confirmar pedido.",
                            "No inventar productos, precios, impuestos ni descuentos.",
                            "Registrar payload_hash y record_id para evitar duplicados.",
                        ],
                    }
                ),
                warnings=["Crear cotizaciones es side effect comercial; exigir approval y politica de precios."],
            )
        )
    if has_write:
        files.append(
            build_acp_file_entry(
                path="ACP/integrations/odoo/write-approval-policy.yaml",
                domain="integrations",
                title="Odoo write approval policy",
                format="yaml",
                source_sections=["blueprint.tools", "approvals", "governance"],
                content_text=serialize_yaml_document(
                    {
                        "schema_version": "odoo-write-approval-policy.v1",
                        "write_requires": ["approval_gate", "allowed_model", "allowed_write_action", "payload_schema", "idempotency_key"],
                        "allowed_write_actions": sorted(
                            action
                            for key in keys
                            for action in ODOO_CONNECTOR_PROFILES[key]["actions"]
                            if ODOO_CONNECTOR_PROFILES[key]["side_effects"]
                        ),
                        "human_approval_required_when": ["production_environment", "price_or_discount_change", "stage_change", "customer_visible_document"],
                        "failure_policy": "fail_closed_and_escalate_to_owner",
                    }
                ),
            )
        )
    return files


def _flow_node(
    node_id: str,
    *,
    label: str,
    node_type: str,
    layer: str,
    state: str,
    x: int,
    y: int,
    description: str,
    source_files: list[str],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "type": node_type,
        "layer": layer,
        "state": state,
        "x": x,
        "y": y,
        "description": description,
        "source_files": source_files,
        "metrics": metrics or {},
    }


def _flow_edge(
    source: str,
    target: str,
    relation: str,
    *,
    label: str,
    mode: str = "normal",
) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "label": label,
        "mode": mode,
    }


def _agent_flow_map_payload(snapshot: SessionSnapshot) -> dict[str, Any]:
    blueprint = snapshot.blueprint
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    tool_items = list(enumerate(project_blueprint_tools_for_construction(snapshot), start=1)) if blueprint is not None else []
    workflow_steps = blueprint.delivery_package.workflow_profile.steps if blueprint is not None else []
    has_knowledge = bool(
        blueprint is not None
        and (
            blueprint.knowledge_profile.mode
            or blueprint.knowledge_profile.sources
            or blueprint.knowledge_profile.ingestion_policy.allowed_file_types
        )
    )
    has_memory = bool(blueprint is not None and (blueprint.memory_profile.storage_layers or blueprint.memory_strategy))
    requires_approval = any(bool(getattr(tool, "requires_approval", False) or getattr(tool, "has_side_effects", False)) for _, tool in tool_items)

    nodes.append(
        _flow_node(
            "user_channel",
            label=(discovery.current_user if discovery and discovery.current_user else "Usuario / canal"),
            node_type="entry",
            layer="experience",
            state="defined",
            x=6,
            y=48,
            description="Punto de entrada de la solicitud o evento del usuario.",
            source_files=["ACP/business/lean-canvas.yaml", "ACP/manifest.yaml"],
        )
    )
    nodes.append(
        _flow_node(
            "agent_core",
            label=snapshot.session.title or "Agent core",
            node_type="agent",
            layer="runtime",
            state="defined",
            x=24,
            y=48,
            description="Nucleo del agente definido por el Blueprint aprobado.",
            source_files=["ACP/manifest.yaml", "ACP/architecture/topology.yaml"],
            metrics={"workflow_steps": len(workflow_steps), "tools": len(tool_items)},
        )
    )
    nodes.append(
        _flow_node(
            "planner",
            label=(blueprint.reasoning_pattern if blueprint and blueprint.reasoning_pattern else "Planner"),
            node_type="reasoning",
            layer="cognition",
            state="defined",
            x=42,
            y=25,
            description="Planifica la siguiente accion respetando guardrails, objetivo y workflow.",
            source_files=["ACP/cognition/reasoning.yaml", "ACP/cognition/planner.yaml", "ACP/workflows/state-machine.yaml"],
        )
    )
    if has_knowledge:
        nodes.append(
            _flow_node(
                "rag",
                label="RAG / conocimiento",
                node_type="rag",
                layer="knowledge",
                state="defined",
                x=60,
                y=18,
                description="Recupera evidencia y contexto antes de responder o llamar tools.",
                source_files=["ACP/knowledge/sources.yaml", "ACP/knowledge/ingestion.yaml", "ACP/knowledge/embeddings.yaml"],
            )
        )
    if has_memory:
        nodes.append(
            _flow_node(
                "memory",
                label=blueprint.memory_profile.strategy or blueprint.memory_strategy or "Memoria",
                node_type="memory",
                layer="memory",
                state="defined",
                x=60,
                y=48,
                description="Conserva estado, checkpoints y recuperacion segun politica aprobada.",
                source_files=["ACP/memory/strategy.yaml", "ACP/memory/retrieval.yaml", "ACP/memory/lifecycle.yaml"],
                metrics={"layers": len(blueprint.memory_profile.storage_layers) if blueprint else 0},
            )
        )

    tool_y_positions = [30, 50, 70, 20, 82]
    for offset, (index, tool) in enumerate(tool_items[:5]):
        slug = _tool_connector_slug(tool, index)
        side_effects = bool(getattr(tool, "has_side_effects", False))
        needs_approval = bool(getattr(tool, "requires_approval", False) or side_effects)
        nodes.append(
            _flow_node(
                f"tool_{slug}",
                label=getattr(tool, "name", "") or f"Tool {index}",
                node_type="tool",
                layer="tools",
                state="approval_required" if needs_approval else "needs_binding",
                x=76,
                y=tool_y_positions[offset % len(tool_y_positions)],
                description=getattr(tool, "purpose", "") or "Tool definida en el Blueprint.",
                source_files=[
                    _tool_contract_ref(tool, index),
                    f"ACP/tools/connectors/{slug}.yaml",
                    f"ACP/tools/bindings/{slug}.production.yaml",
                    f"ACP/tools/tests/{slug}-smoke-test.yaml",
                ],
                metrics={
                    "risk_level": getattr(tool, "risk_level", "") or "needs_review",
                    "side_effects": side_effects,
                    "requires_approval": needs_approval,
                },
            )
        )

    if requires_approval:
        nodes.append(
            _flow_node(
                "human_approval",
                label="Handoff / approval",
                node_type="approval",
                layer="governance",
                state="approval_required",
                x=76,
                y=88,
                description="Pausa humana para side effects, decisiones delegadas o aprobaciones de produccion.",
                source_files=["ACP/governance/approval-matrix.yaml", "ACP/governance/decision-policy.yaml", "ACP/tools/permissions.yaml"],
            )
        )
    nodes.append(
        _flow_node(
            "fallbacks",
            label="Fallbacks",
            node_type="fallback",
            layer="governance",
            state="fallback_available" if blueprint is not None else "defined",
            x=91,
            y=78,
            description="Rutas alternativas, pausa de agente o escalamiento ante fallo.",
            source_files=["ACP/governance/control-plane.yaml", "ACP/ops/runbooks/fallback-and-incident-response.md"],
        )
    )
    nodes.append(
        _flow_node(
            "finops",
            label="FinOps",
            node_type="finops",
            layer="costs",
            state="defined",
            x=42,
            y=82,
            description="Controla presupuesto, umbrales y sensibilidad de costo operativo.",
            source_files=["ACP/finops/budget-policy.yaml", "ACP/costs/operational-cost-estimate.json"],
        )
    )
    nodes.append(
        _flow_node(
            "observability",
            label="Observabilidad",
            node_type="observability",
            layer="observability",
            state="defined",
            x=60,
            y=82,
            description="Emite eventos, metricas, alertas y trazas para auditoria.",
            source_files=["ACP/observability/event-model.yaml", "ACP/observability/metrics.yaml", "ACP/observability/alerts.yaml"],
        )
    )
    nodes.append(
        _flow_node(
            "output",
            label=(canvas.success_metric if canvas and canvas.success_metric else "Respuesta / accion final"),
            node_type="output",
            layer="experience",
            state="defined",
            x=93,
            y=48,
            description="Salida esperada del agente segun objetivo, workflow y deliverables.",
            source_files=["ACP/README.md", "ACP/workflows/durable-workflow.yaml", "ACP/evaluation/golden-dataset.json"],
        )
    )

    node_ids = {node["id"] for node in nodes}
    edges.append(_flow_edge("user_channel", "agent_core", "receives_request", label="solicitud"))
    edges.append(_flow_edge("agent_core", "planner", "plans", label="plan"))
    if "rag" in node_ids:
        edges.append(_flow_edge("planner", "rag", "retrieves_context", label="evidencia"))
        edges.append(_flow_edge("rag", "planner", "grounds", label="contexto"))
    if "memory" in node_ids:
        edges.append(_flow_edge("planner", "memory", "reads_memory", label="estado"))
        edges.append(_flow_edge("memory", "planner", "returns_memory", label="checkpoint"))
    for index, tool in tool_items[:5]:
        slug = _tool_connector_slug(tool, index)
        tool_id = f"tool_{slug}"
        if tool_id not in node_ids:
            continue
        edges.append(_flow_edge("planner", tool_id, "calls_tool", label="tool call"))
        if bool(getattr(tool, "requires_approval", False) or getattr(tool, "has_side_effects", False)) and "human_approval" in node_ids:
            edges.append(_flow_edge(tool_id, "human_approval", "requires_approval", label="approval", mode="approval"))
            edges.append(_flow_edge("human_approval", tool_id, "approves", label="ok", mode="approval"))
        edges.append(_flow_edge(tool_id, "observability", "emits_event", label="eventos"))
        edges.append(_flow_edge(tool_id, "fallbacks", "falls_back_to", label="fallo", mode="fallback"))
        edges.append(_flow_edge(tool_id, "output", "returns_output", label="resultado"))
    edges.append(_flow_edge("planner", "finops", "tracks_cost", label="costo"))
    edges.append(_flow_edge("planner", "observability", "emits_event", label="traza"))
    edges.append(_flow_edge("planner", "output", "returns_output", label="respuesta"))
    edges.append(_flow_edge("fallbacks", "output", "returns_output", label="handoff"))

    return {
        "schema_version": "agent-flow-map.v1",
        "title": "Mapa vivo del agente",
        "source_files": [
            "ACP/blueprint.graph.json",
            "ACP/workflows/langgraph.json",
            "ACP/tools/connectors/catalog.yaml",
            "ACP/governance/control-plane.yaml",
            "ACP/observability/event-model.yaml",
        ],
        "modes": ["design", "simulation", "operations"],
        "layers": ["experience", "runtime", "cognition", "knowledge", "memory", "tools", "governance", "costs", "observability"],
        "states": ["defined", "needs_binding", "sandbox_ready", "production_ready", "live", "degraded", "blocked", "fallback_available", "approval_required"],
        "nodes": nodes,
        "edges": edges,
    }


def _agent_flow_map_scenarios(payload: dict[str, Any]) -> dict[str, Any]:
    node_ids = {str(node.get("id")) for node in payload.get("nodes", [])}
    steps = [
        {"step": "request_received", "node_id": "user_channel", "event": "agent.started", "caption": "Entra una solicitud o evento."},
        {"step": "plan_next_action", "node_id": "planner", "event": "decision.requested", "caption": "El planner decide que contexto y acciones necesita."},
    ]
    if "rag" in node_ids:
        steps.append({"step": "retrieve_context", "node_id": "rag", "event": "knowledge.retrieved", "caption": "RAG aporta evidencia antes de actuar."})
    if "memory" in node_ids:
        steps.append({"step": "read_memory", "node_id": "memory", "event": "memory.read", "caption": "La memoria recupera estado y checkpoints."})
    first_tool = next((node_id for node_id in node_ids if node_id.startswith("tool_")), "")
    if first_tool:
        steps.append({"step": "call_tool", "node_id": first_tool, "event": "tool.call.started", "caption": "La tool se evalua contra contrato, binding y policy."})
    if "human_approval" in node_ids:
        steps.append({"step": "approval_gate", "node_id": "human_approval", "event": "approval.requested", "caption": "La accion sensible espera aprobacion humana."})
    steps.extend(
        [
            {"step": "track_cost", "node_id": "finops", "event": "budget.threshold_checked", "caption": "FinOps controla costo y limites."},
            {"step": "emit_observability", "node_id": "observability", "event": "tool.call.completed", "caption": "La consola registra eventos y trazas."},
            {"step": "return_output", "node_id": "output", "event": "agent.completed", "caption": "El agente devuelve respuesta, accion o handoff."},
        ]
    )
    return {
        "schema_version": "agent-flow-simulation-scenarios.v1",
        "scenarios": [
            {
                "scenario_key": "happy_path_with_governance",
                "label": "Flujo gobernado",
                "offline_simulation_only": True,
                "steps": steps,
            },
            {
                "scenario_key": "fallback_path",
                "label": "Fallo con fallback",
                "offline_simulation_only": True,
                "steps": [
                    {"step": "tool_fails", "node_id": first_tool or "planner", "event": "tool.call.failed", "caption": "Una tool falla o no tiene binding valido."},
                    {"step": "activate_fallback", "node_id": "fallbacks", "event": "fallback.activated", "caption": "Se activa fallback, pausa o escalamiento humano."},
                    {"step": "return_safe_output", "node_id": "output", "event": "agent.completed", "caption": "El flujo termina sin ejecutar side effects no autorizados."},
                ],
            },
        ],
    }


def _build_agent_flow_map_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    payload = _agent_flow_map_payload(snapshot)
    scenarios = _agent_flow_map_scenarios(payload)
    manifest = {
        "schema_version": "agent-flow-map-manifest.v1",
        "viewer_chapter": "flow-map",
        "offline_first": True,
        "runtime_execution": False,
        "data_ref": "ACP/ops/agent-flow-map/flow-map-data.json",
        "scenario_ref": "ACP/ops/agent-flow-map/simulation-scenarios.json",
        "source_files": payload["source_files"],
        "modes": payload["modes"],
    }
    state_policy = {
        "schema_version": "agent-flow-node-state-policy.v1",
        "states": {
            "defined": "Existe en el ACP.",
            "needs_binding": "Requiere binding de entorno antes de produccion.",
            "sandbox_ready": "Listo para prueba controlada.",
            "production_ready": "Listo para aprobacion final.",
            "live": "Activo en runtime.",
            "degraded": "Activo con alerta o incidente.",
            "blocked": "No debe activarse.",
            "fallback_available": "Tiene ruta alternativa definida.",
            "approval_required": "Requiere aprobacion humana.",
        },
        "hard_rules": [
            "El mapa no ejecuta tools reales.",
            "El mapa no almacena secretos.",
            "Cada nodo debe enlazar a fuentes ACP.",
            "La simulacion debe ser offline y no modificar arquitectura aprobada.",
        ],
    }
    interaction_model = {
        "schema_version": "agent-flow-interaction-model.v1",
        "controls": ["mode_switch", "play_pause", "speed", "layer_filter", "node_detail"],
        "node_click_behavior": "open_detail_panel_with_source_links",
        "mode_behavior": {
            "design": "mostrar arquitectura y responsabilidades",
            "simulation": "animar flujo de informacion sin ejecucion real",
            "operations": "resaltar readiness, bindings, approvals, fallbacks y eventos",
        },
    }
    readme = "\n".join(
        [
            "# Interactive Agent Flow Map",
            "",
            "Este mapa vivo forma parte del ACP descargable. Es una simulacion visual offline derivada de contratos reales.",
            "",
            "## No ejecuta runtime",
            "",
            "La animacion no llama herramientas, no lee secretos y no activa produccion. Solo explica flujo, estados y evidencia.",
            "",
            "## Fuentes principales",
            "",
            "- `ACP/blueprint.graph.json`",
            "- `ACP/workflows/langgraph.json`",
            "- `ACP/tools/connectors/catalog.yaml`",
            "- `ACP/governance/control-plane.yaml`",
            "- `ACP/observability/event-model.yaml`",
        ]
    )
    return [
        build_acp_file_entry(
            path="ACP/ops/agent-flow-map/flow-map-manifest.json",
            domain="governance",
            title="Agent flow map manifest",
            format="json",
            source_sections=["blueprint", "tool_contracts", "runtime", "governance", "observability"],
            content_text=serialize_json_document(manifest),
        ),
        build_acp_file_entry(
            path="ACP/ops/agent-flow-map/flow-map-data.json",
            domain="governance",
            title="Agent flow map data",
            format="json",
            source_sections=["blueprint", "workflows", "tools", "memory", "knowledge", "governance"],
            content_text=serialize_json_document(payload),
        ),
        build_acp_file_entry(
            path="ACP/ops/agent-flow-map/simulation-scenarios.json",
            domain="governance",
            title="Agent flow map simulation scenarios",
            format="json",
            source_sections=["workflows", "tools", "observability", "governance"],
            content_text=serialize_json_document(scenarios),
        ),
        build_acp_file_entry(
            path="ACP/ops/agent-flow-map/node-state-policy.yaml",
            domain="governance",
            title="Agent flow node state policy",
            format="yaml",
            source_sections=["governance", "tool_contracts", "runtime"],
            content_text=serialize_yaml_document(state_policy),
        ),
        build_acp_file_entry(
            path="ACP/ops/agent-flow-map/interaction-model.yaml",
            domain="governance",
            title="Agent flow interaction model",
            format="yaml",
            source_sections=["governance", "viewer", "operations"],
            content_text=serialize_yaml_document(interaction_model),
        ),
        build_acp_file_entry(
            path="ACP/ops/agent-flow-map/README.md",
            domain="governance",
            title="Interactive Agent Flow Map README",
            format="markdown",
            source_sections=["governance", "viewer"],
            content_text=serialize_markdown_document(readme),
        ),
    ]


def _build_workflow_files(snapshot: SessionSnapshot, context: ProjectGenerationContext | None = None) -> list[ACPFileEntry]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    steps = blueprint.delivery_package.workflow_profile.steps
    state_machine_payload = {
        **_context_trace_payload(context),
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
    durable_payload = {**_context_trace_payload(context), **durable_payload}
    langgraph_payload = {
        **_context_trace_payload(context),
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


def _build_objective_files(
    snapshot: SessionSnapshot,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
) -> list[ACPFileEntry]:
    bundle = build_objective_contract_bundle(snapshot, response_records)
    objective = active_objective(bundle)
    objective_payload = objective.model_dump(mode="json") if objective is not None else {}
    traceability_payload = {
        "contract_version": bundle.contract_version,
        "active_objective_id": bundle.active_objective_id,
        "policy_version": bundle.policy_version,
        "source_refs": list(bundle.source_refs),
        "constraints": list(bundle.constraints),
        "status": objective.status if objective is not None else "missing",
        "runtime_tracking": objective.runtime_tracking if objective is not None else "not_required",
        "success_criteria": objective_payload.get("success_criteria", []),
        "termination_conditions": objective_payload.get("termination_conditions", {}),
        "progress_signals": objective_payload.get("progress_signals", []),
        "acp_validation": {
            "uses_responder_preguntas": True,
            "question_kind": "objective_validation",
            "reopen_supported": True,
            "new_interaction_mechanism": False,
        },
    }
    files = [
        build_acp_file_entry(
            path="ACP/objectives/objective-contract.yaml",
            domain="objectives",
            title="Objective contract",
            format="yaml",
            source_sections=["canvas.user_goal", "discovery.desired_outcome", "evaluation_dataset", "acp.questions"],
            content_text=serialize_yaml_document(bundle.model_dump(mode="json")),
            warnings=[] if objective is not None and objective.status != "rejected" else ["El objetivo activo fue rechazado; corregir antes de activar runtime operacional."],
        ),
        build_acp_file_entry(
            path="ACP/objectives/objective-traceability.yaml",
            domain="objectives",
            title="Objective traceability",
            format="yaml",
            source_sections=["objective-contract.v1", "prompt-pack.v1", "construction-readiness"],
            content_text=serialize_yaml_document(traceability_payload),
        ),
    ]
    if objective is not None and objective.status != "rejected" and objective_requires_runtime_loop(objective):
        loop_payload = {
            "workflow_key": "objective_loop",
            "activation": "design_contract_only",
            "runtime_tracking": objective.runtime_tracking,
            "objective_id": objective.objective_id,
            "statement": objective.statement,
            "loop_steps": [
                {
                    "step": "load_objective",
                    "purpose": "Cargar objetivo activo, restricciones, criterios y condiciones de terminacion.",
                    "required_inputs": ["objective-contract.v1"],
                },
                {
                    "step": "plan_next_action",
                    "purpose": "Seleccionar la siguiente accion que acerque al objetivo sin violar restricciones.",
                    "required_inputs": ["behavior-spec.v1", "memory-policy.v1", "tool-contract.v1"],
                },
                {
                    "step": "execute_or_request_approval",
                    "purpose": "Ejecutar solo acciones permitidas o pedir aprobacion humana cuando aplique.",
                    "required_inputs": ["approved_action", "approval_policy"],
                },
                {
                    "step": "evaluate_progress",
                    "purpose": "Comparar evidencia y resultado contra criterios de exito y senales de progreso.",
                    "required_inputs": ["evidence_refs", "progress_signals"],
                },
                {
                    "step": "terminate_or_replan",
                    "purpose": "Cerrar si hay exito, detener por stop condition o replanificar de forma acotada.",
                    "required_inputs": ["termination_conditions", "mutation_policy"],
                },
            ],
            "guardrails": {
                "mutation_policy": objective.mutation_policy,
                "stop_conditions": list(objective.termination_conditions.stop),
                "progress_signals": list(objective.progress_signals),
                "human_validation_source": "ACP Responder preguntas",
            },
        }
        files.append(
            build_acp_file_entry(
                path="ACP/workflows/objective-loop.yaml",
                domain="workflows",
                title="Objective Loop",
                format="yaml",
                source_sections=["objective-contract.v1", "behavior-spec.v1", "memory-policy.v1"],
                content_text=serialize_yaml_document(loop_payload),
                warnings=["Especificacion de diseno portable; no activa ejecucion productiva dentro de LAB."],
            )
        )
    return files


_PROMPT_SYNTHESIS_SECTION_IDS = {
    "ACP/prompts/system.md": "system",
    "ACP/prompts/planner.md": "planner",
    "ACP/prompts/evaluator.md": "evaluator",
    "ACP/prompts/skills/discovery.md": "skill.discovery",
    "ACP/prompts/skills/architecture.md": "skill.architecture",
    "ACP/prompts/skills/evaluation.md": "skill.evaluation",
    "ACP/prompts/skills/catalog.md": "skill.catalog",
}


def _build_prompt_files(
    snapshot: SessionSnapshot,
    context: ProjectGenerationContext | None = None,
    prompt_synthesizer: ACPPromptSectionSynthesizer | None = None,
) -> list[ACPFileEntry]:
    discovery = snapshot.discovery
    blueprint = snapshot.blueprint
    canvas = snapshot.canvas
    title = snapshot.session.title or "Agent System"

    desired_outcome = context.desired_outcome if context and context.desired_outcome else discovery.desired_outcome if discovery and discovery.desired_outcome else "needs_review"
    problem_statement = context.problem_statement if context and context.problem_statement else discovery.problem_statement if discovery and discovery.problem_statement else "needs_review"
    primary_user = (
        context.current_user
        if context and context.current_user
        else (canvas.agent_profile.primary_user if canvas and canvas.agent_profile else None)
        or (discovery.current_user if discovery and discovery.current_user else "needs_review")
    )
    current_process = context.current_process if context and context.current_process else discovery.current_process if discovery and discovery.current_process else "needs_review"
    architecture = context.architecture if context and context.architecture else blueprint.architecture if blueprint and blueprint.architecture else "needs_review"
    reasoning_pattern = context.reasoning_pattern if context and context.reasoning_pattern else blueprint.reasoning_pattern if blueprint and blueprint.reasoning_pattern else "needs_review"
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
                f"> Contexto: `{context.context_version if context is not None else 'needs_review'}` / `{context.input_fingerprint if context is not None else 'needs_review'}`",
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
    elif context is not None and "project-generation-context.v1" not in system_prompt:
        system_prompt = "\n".join(
            [
                f"> Contexto: `{context.context_version}` / `{context.input_fingerprint}`",
                "",
                system_prompt,
            ]
        )

    skill_spec = _find_deliverable(snapshot, "skill_spec")
    
    planner_prompt = "\n".join(
        [
            f"# Planner Role Prompt: {title}",
            "",
            f"> Contexto: `{context.context_version if context is not None else 'needs_review'}` / `{context.input_fingerprint if context is not None else 'needs_review'}`",
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
            f"> Contexto: `{context.context_version if context is not None else 'needs_review'}` / `{context.input_fingerprint if context is not None else 'needs_review'}`",
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
    return _apply_prompt_section_synthesis(files, context, prompt_synthesizer)


def _apply_prompt_section_synthesis(
    files: list[ACPFileEntry],
    context: ProjectGenerationContext | None,
    prompt_synthesizer: ACPPromptSectionSynthesizer | None,
) -> list[ACPFileEntry]:
    if context is None or prompt_synthesizer is None:
        return files

    synthesized_files: list[ACPFileEntry] = []
    first_question = _first_actionable_construction_question(context)
    for entry in files:
        section_id = _PROMPT_SYNTHESIS_SECTION_IDS.get(entry.path)
        if not section_id:
            synthesized_files.append(entry)
            continue

        request = build_prompt_section_synthesis_request(
            section_id=section_id,
            path=entry.path,
            title=entry.title,
            deterministic_markdown=entry.content_text,
            context=context,
            first_actionable_question=first_question,
        )
        try:
            synthesis = prompt_synthesizer(request)
            if synthesis is None:
                _record_prompt_synthesis_outcome(
                    prompt_synthesizer,
                    section_id=section_id,
                    path=entry.path,
                    status="fallback",
                    reason="provider_returned_no_artifact",
                )
                synthesized_files.append(entry)
                continue
            validated = validate_prompt_section_synthesis(synthesis, request)
        except (PromptSectionSynthesisRejected, ValueError, TypeError) as exc:
            _record_prompt_synthesis_outcome(
                prompt_synthesizer,
                section_id=section_id,
                path=entry.path,
                status="rejected",
                reason=str(exc)[:240],
            )
            synthesized_files.append(entry)
            continue

        _record_prompt_synthesis_outcome(
            prompt_synthesizer,
            section_id=section_id,
            path=entry.path,
            status="applied",
            reason="validated",
        )
        synthesized_files.append(
            build_acp_file_entry(
                path=entry.path,
                domain=entry.domain,
                title=entry.title,
                format=entry.format,
                source_sections=[*entry.source_sections, "llm_prompt_synthesis"],
                content_text=serialize_markdown_document(validated.section_markdown),
                missing_fields=entry.missing_fields,
                warnings=entry.warnings,
            )
        )
    report_payload = _prompt_synthesis_report_payload(prompt_synthesizer)
    if report_payload:
        synthesized_files.append(
            build_acp_file_entry(
                path="ACP/prompts/synthesis-report.json",
                domain="prompts",
                title="Prompt synthesis report",
                format="json",
                source_sections=["llm_prompt_synthesis"],
                content_text=serialize_json_document(report_payload),
            )
        )
    return synthesized_files


def _record_prompt_synthesis_outcome(
    prompt_synthesizer: ACPPromptSectionSynthesizer,
    *,
    section_id: str,
    path: str,
    status: str,
    reason: str = "",
) -> None:
    recorder = getattr(prompt_synthesizer, "record_validation_outcome", None)
    if callable(recorder):
        recorder(section_id=section_id, path=path, status=status, reason=reason)


def _prompt_synthesis_report_payload(prompt_synthesizer: ACPPromptSectionSynthesizer) -> dict[str, object] | None:
    report_builder = getattr(prompt_synthesizer, "build_report_payload", None)
    if not callable(report_builder):
        return None
    payload = report_builder()
    return payload if isinstance(payload, dict) else None


def _suggested_owners(gap: ConstructionGapEntry) -> list[str]:
    owners: list[str] = []
    seen: set[str] = set()
    for question in gap.questions:
        owner = question.target_owner.strip()
        if owner and owner not in seen:
            seen.add(owner)
            owners.append(owner)
    return owners


def _flatten_open_questions(
    preview: ACPPreview,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for question in build_construction_question_views(preview, response_records or []):
        if question.status != "open":
            continue
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
                "gap_key": question.gap_key,
                "domain": question.domain,
                "question_text": question.question_text,
                "rationale": question.rationale,
                "purpose": question.purpose,
                "suggested_answer": suggested_answer,
                "target_owner": question.target_owner,
                "expected_answer_format": question.expected_answer_format,
                "blocking": question.blocking,
                "impacted_artifacts": question.impacted_artifacts,
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
        "objectives": "objective_contract",
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
    tools = project_blueprint_tools_for_construction(snapshot)
    return [
        (index, tool)
        for index, tool in enumerate(tools, start=1)
        if tool.name not in INTERNAL_BUILDER_TOOL_NAMES and getattr(tool, "tool_type", "external") != "internal"
    ]


def _whatsapp_required_api_contract(tool: Any, index: int) -> dict[str, Any]:
    return {
        "system_name": "WhatsApp Business Cloud API",
        "connector_key": "whatsapp_cloud_api",
        "tool_name": getattr(tool, "name", "") or "whatsapp_business_messaging",
        "purpose": "Canal conversacional inbound/outbound para el agente.",
        "required_endpoints_or_actions": [
            "GET /webhooks/whatsapp",
            "POST /webhooks/whatsapp",
            "POST /{phone_number_id}/messages",
        ],
        "expected_authentication": {
            "outbound": "Bearer token via WHATSAPP_ACCESS_TOKEN_REF",
            "inbound_verify": "WHATSAPP_WEBHOOK_VERIFY_TOKEN_REF",
            "inbound_signature": "WHATSAPP_APP_SECRET_REF",
        },
        "unknown_payloads": [
            "approved_templates",
            "sandbox_callback_url",
            "production_callback_url",
            "deployment_target",
            "contact_opt_in_source",
        ],
        "examples_required": True,
        "impact_if_missing": "Bloquea construccion real del canal WhatsApp.",
        "contract_path": build_tool_contract_path_for_tool(tool, index),
        "webhook_contract_path": "ACP/webhooks/whatsapp-business-webhook.yaml",
        "send_contract_path": "ACP/integrations/whatsapp/send-message.contract.yaml",
        "template_contract_path": "ACP/integrations/whatsapp/templates.yaml",
    }


def _google_workspace_required_api_contract(tool: Any, index: int) -> dict[str, Any]:
    google_key = _google_workspace_connector_key(tool)
    profile = GOOGLE_WORKSPACE_CONNECTOR_PROFILES.get(google_key, {})
    service = str(profile.get("service") or "google_workspace")
    unknowns_by_service = {
        "drive": ["oauth_client", "redirect_uri", "selected_files_or_picker_policy", "allowed_mime_types", "refresh_mode"],
        "sheets": ["oauth_client", "spreadsheet_id", "range", "column_schema", "primary_key", "cache_policy"],
        "calendar": ["oauth_client", "calendar_id", "timezone", "availability_window", "attendee_policy", "approval_policy"],
        "gmail": ["oauth_client", "sender_account", "recipient_policy", "draft_or_send_policy", "approval_policy", "data_retention_policy"],
    }
    return {
        "system_name": "Google Workspace",
        "connector_key": google_key or "google_workspace",
        "tool_name": getattr(tool, "name", "") or google_key or f"tool_{index}",
        "purpose": getattr(tool, "purpose", "") or str(profile.get("label") or "Integracion Google Workspace"),
        "required_endpoints_or_actions": list(profile.get("actions") or ["oauth2_rest_api"]),
        "expected_authentication": {
            "auth_type": "OAuth 2.0",
            "client_id_ref": "GOOGLE_OAUTH_CLIENT_ID",
            "client_secret_ref": "GOOGLE_OAUTH_CLIENT_SECRET",
            "refresh_token_ref": "GOOGLE_REFRESH_TOKEN_REF",
            "allowed_scopes_ref": "GOOGLE_ALLOWED_SCOPES",
        },
        "unknown_payloads": unknowns_by_service.get(service, ["oauth_client", "allowed_scopes", "resource_policy"]),
        "examples_required": True,
        "impact_if_missing": "Permite generar contrato, pero bloquea activacion sandbox/produccion de la tool Google Workspace.",
        "contract_path": build_tool_contract_path_for_tool(tool, index),
        "integration_policy_path": "ACP/integrations/google-workspace/oauth-policy.yaml",
        "scopes_matrix_path": "ACP/integrations/google-workspace/scopes-matrix.yaml",
        "service_contract_paths": list(profile.get("contract_refs") or []),
    }


def _odoo_required_api_contract(tool: Any, index: int) -> dict[str, Any]:
    odoo_key = _odoo_connector_key(tool)
    profile = ODOO_CONNECTOR_PROFILES.get(odoo_key, {})
    model = str(profile.get("model") or "needs_review")
    side_effects = bool(profile.get("side_effects"))
    unknowns = ["odoo_version", "api_mode", "base_url", "database", "technical_user", "allowed_models", "allowed_fields", "access_rights"]
    if side_effects:
        unknowns.extend(["allowed_write_actions", "approval_policy", "idempotency_key_policy"])
    if odoo_key == "odoo_sale_quote_create":
        unknowns.extend(["pricelist", "tax_policy", "discount_policy", "quote_expiration"])
    return {
        "system_name": "Odoo",
        "connector_key": odoo_key or "odoo",
        "tool_name": getattr(tool, "name", "") or odoo_key or f"tool_{index}",
        "purpose": getattr(tool, "purpose", "") or str(profile.get("label") or "Integracion Odoo"),
        "required_endpoints_or_actions": list(profile.get("actions") or ["search_read"]),
        "expected_authentication": {
            "auth_type": "Odoo API key/password",
            "api_mode_ref": "ODOO_API_MODE",
            "base_url_ref": "ODOO_BASE_URL",
            "database_ref": "ODOO_DATABASE",
            "username_ref": "ODOO_USERNAME",
            "password_or_api_key_ref": "ODOO_PASSWORD or ODOO_API_KEY",
        },
        "target_model": model,
        "unknown_payloads": unknowns,
        "examples_required": True,
        "impact_if_missing": "Permite generar contrato, pero bloquea activacion sandbox/produccion de la tool Odoo.",
        "contract_path": build_tool_contract_path_for_tool(tool, index),
        "api_profile_path": "ACP/integrations/odoo/api-profile.yaml",
        "version_policy_path": "ACP/integrations/odoo/version-policy.yaml",
        "models_scope_path": "ACP/integrations/odoo/models-scope.yaml",
        "service_contract_paths": list(profile.get("contract_refs") or []),
    }


def _build_construction_step_guide_markdown(
    *,
    validation: Any,
    readiness: Any,
    blocking_gaps: list[ConstructionGapEntry],
    open_questions: list[dict[str, Any]],
    structured_deferred_decisions: list[dict[str, Any]],
    external_dependencies: list[dict[str, Any]],
    required_api_contracts: list[dict[str, Any]],
    deployment_questions: list[dict[str, Any]],
    context: ProjectGenerationContext | None = None,
) -> str:
    def append_artifacts(lines: list[str], artifacts: list[str]) -> None:
        lines.append("- Artefactos impactados:")
        if artifacts:
            for artifact in artifacts[:8]:
                lines.append(f"  - `{artifact}`")
        else:
            lines.append("  - `needs_review`")

    def append_options(lines: list[str], options: list[dict[str, Any]]) -> None:
        if not options:
            return
        lines.append("- Alternativas sugeridas:")
        for option in options[:5]:
            label = str(option.get("label") or option.get("key") or "opcion").strip()
            description = str(option.get("description") or option.get("impact") or "").strip()
            recommended = " recomendada" if option.get("recommended") else ""
            suffix = f": {description}" if description else ""
            lines.append(f"  - `{label}`{recommended}{suffix}")

    lines: list[str] = [
        "# Guia paso a paso de construccion ACP",
        "",
        "Este archivo convierte el ACP en una guia operativa para construir el agente sin inventar informacion faltante.",
        "Usalo como primera lectura despues de abrir el paquete y antes de modificar runtime, tools, memoria, conocimiento, deployment o codigo.",
        "",
        "## Regla de avance",
        f"- `package_validation.can_export_zip`: `{str(validation.can_export_zip).lower()}`.",
        f"- `construction_readiness.can_start_build`: `{str(readiness.can_start_build).lower()}`.",
        "- Poder descargar el ZIP no significa que todas las integraciones, secrets, fuentes RAG o bindings reales existan.",
        "- Si una decision falta, pregunta al usuario en el momento indicado y registra la respuesta antes de editar el artefacto afectado.",
        "- Mantén la trazabilidad `gap_key` -> `question_key` -> respuesta -> artefactos actualizados.",
        "",
        "## Reglas de no-asuncion",
        "- Las tools externas del diseno son contratos hasta que exista un binding operativo confirmado.",
        "- No agregues endpoints, OAuth apps, tokens, secrets, URLs reales, payloads o bases de datos si no estan definidos en el ACP o por el usuario.",
        "- Si una integracion se opera por navegador, el usuario debe operar la sesion autorizada y entregar capturas, texto, HTML o JSON curado.",
        "- No solicites cookies, credenciales directas, scraping no autorizado ni acceso directo a sesiones privadas.",
        "- Si RAG usa fuentes, portafolio, documentos, vector store o embeddings como placeholders, primero guia al usuario para construir esa base.",
        "",
        "## Paso 1 - Confirmar estado inicial",
        f"- Contexto normalizado: `{context.context_version if context is not None else 'needs_review'}`.",
        f"- Fingerprint de contexto: `{context.input_fingerprint if context is not None else 'needs_review'}`.",
        f"- Primera pregunta accionable: {_first_actionable_construction_question(context)}",
        f"- Estado del paquete: `{validation.overall_status}`.",
        f"- Estado de construccion: `{readiness.overall_status}`.",
        f"- Gaps bloqueantes: `{readiness.blocking_gaps}`.",
        f"- Preguntas abiertas: `{readiness.open_questions}`.",
        f"- Supuestos pendientes: `{readiness.assumptions_count}`.",
        "- Lee en paralelo `ACP/construction-readiness/overview.yaml` y `ACP/conformance/portability-report.md`.",
        "",
        "## Paso 2 - Resolver bloqueos reales",
    ]
    if blocking_gaps:
        for index, gap in enumerate(blocking_gaps, start=1):
            lines.extend(
                [
                    f"### 2.{index} `{gap.gap_key}`",
                    f"- Dominio: `{gap.domain}`.",
                    f"- Resumen: {gap.summary}",
                    f"- Criterio de cierre: {'; '.join(gap.closure_criteria) if gap.closure_criteria else 'needs_review'}.",
                ]
            )
            append_artifacts(lines, list(gap.evidence_paths or []))
    else:
        lines.append("- No hay gaps bloqueantes actuales; conserva esta verificacion antes de activar integraciones reales.")

    lines.extend(
        [
            "",
            "## Paso 3 - Preguntar lo abierto",
        ]
    )
    if open_questions:
        for index, question in enumerate(open_questions, start=1):
            lines.extend(
                [
                    f"### 3.{index} `{question['question_key']}`",
                    f"- Gap: `{question['gap_key']}`.",
                    f"- Dominio: `{question['domain']}`.",
                    f"- Pregunta para el usuario: {question['question_text']}",
                    f"- Por que importa: {question['rationale']}",
                    f"- Owner sugerido: `{question['target_owner']}`.",
                    f"- Formato esperado: `{question['expected_answer_format']}`.",
                    f"- Bloqueante: `{str(question['blocking']).lower()}`.",
                ]
            )
            append_options(lines, list(question.get("options") or []))
            append_artifacts(lines, list(question.get("impacted_artifacts") or []))
    else:
        lines.append("- No hay preguntas abiertas activas. Revisa de todos modos las decisiones delegadas antes de construir cada componente.")

    lines.extend(
        [
            "",
            "## Paso 4 - Ejecutar decisiones delegadas durante implementacion",
        ]
    )
    if structured_deferred_decisions:
        for index, decision in enumerate(structured_deferred_decisions, start=1):
            lines.extend(
                [
                    f"### 4.{index} `{decision.get('question_key')}`",
                    f"- Gap: `{decision.get('gap_key') or 'needs_review'}`.",
                    f"- Dominio: `{decision.get('domain') or 'general'}`.",
                    f"- Pregunta a formular: {decision.get('question_text') or 'needs_review'}",
                    f"- Owner sugerido: `{decision.get('target_owner') or 'developer'}`.",
                    "- Politica: `DO_NOT_ASSUME_SILENTLY`; pregunta al usuario antes de tocar el componente afectado.",
                ]
            )
            append_options(lines, list(decision.get("options") or []))
            append_artifacts(lines, list(decision.get("impacted_artifacts") or []))
    else:
        lines.append("- No hay decisiones delegadas actuales.")

    lines.extend(
        [
            "",
            "## Paso 5 - Cerrar contratos de integracion",
        ]
    )
    if required_api_contracts:
        for index, contract in enumerate(required_api_contracts, start=1):
            unknowns = list(contract.get("unknown_payloads") or [])
            lines.extend(
                [
                    f"### 5.{index} `{contract.get('system_name') or contract.get('tool_name') or 'external_system'}`",
                    f"- Tipo: contrato de diseno; no binding operativo hasta confirmar credenciales, permisos y payloads.",
                    f"- Proposito: {contract.get('purpose') or 'needs_review'}",
                    f"- Acciones/endpoints requeridos: `{', '.join(contract.get('required_endpoints_or_actions') or ['needs_review'])}`.",
                    f"- Autenticacion esperada: `{contract.get('expected_authentication') or 'needs_review'}`.",
                    f"- Payloads por definir: `{', '.join(unknowns) if unknowns else 'cerrado'}`.",
                    f"- Contrato afectado: `{contract.get('contract_path') or 'needs_review'}`.",
                    "- Si el acceso se hace por navegador, detenerse en preparacion guiada y pedir al usuario evidencias curadas.",
                ]
            )
    else:
        lines.append("- No hay contratos API externos pendientes en el ACP actual.")

    lines.extend(
        [
            "",
            "## Paso 6 - Preparar conocimiento y RAG",
        ]
    )
    knowledge_dependencies = [
        dependency for dependency in external_dependencies if dependency.get("category") == "knowledge_source"
    ]
    if knowledge_dependencies:
        for index, dependency in enumerate(knowledge_dependencies, start=1):
            lines.extend(
                [
                    f"### 6.{index} `{dependency.get('dependency_key')}`",
                    f"- Resumen: {dependency.get('summary') or 'needs_review'}",
                    "- Antes de activar retrieval real, define fuentes, permisos, estructura documental, estrategia de ingesta, embeddings y vector store.",
                    f"- Inputs requeridos: `{', '.join(dependency.get('required_inputs') or ['needs_review'])}`.",
                ]
            )
            append_artifacts(lines, list(dependency.get("evidence_paths") or []))
    else:
        lines.append("- No hay dependencia RAG abierta, pero valida que fuentes y vector store no sean placeholders antes de produccion.")

    lines.extend(
        [
            "",
            "## Paso 7 - Confirmar runtime, deployment y aprobaciones",
        ]
    )
    runtime_dependencies = [
        dependency
        for dependency in external_dependencies
        if dependency.get("category") in {"runtime_or_secrets", "deployment_environment"}
    ]
    if runtime_dependencies or deployment_questions:
        for dependency in runtime_dependencies:
            lines.extend(
                [
                    f"- `{dependency.get('dependency_key')}`: {dependency.get('summary') or 'needs_review'}",
                    f"  - Inputs: `{', '.join(dependency.get('required_inputs') or ['needs_review'])}`.",
                ]
            )
        for question in deployment_questions:
            lines.append(f"- Preguntar `{question['question_key']}` antes de fijar deployment o secrets.")
    else:
        lines.append("- No hay decisiones de runtime/deployment abiertas segun readiness actual.")

    lines.extend(
        [
            "",
            "## Paso 8 - Actualizar solo lo impactado",
            "- Cuando el usuario responda, actualiza unicamente los artefactos listados como impactados.",
            "- Registra evidencia en `ACP/construction-readiness/question-impact-log.yaml` o en el mecanismo equivalente de la implementacion.",
            "- Si una respuesta contradice el Blueprint aprobado, crea reconciliacion granular del artefacto; no reinicies fases completas.",
            "- Al terminar, ejecuta validaciones de conformance y revisa `ACP/release-readiness-checklist.md`.",
        ]
    )
    return "\n".join(lines)


def _build_construction_readiness_files(
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    continuity_answers: dict[str, str] | None = None,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
    context: ProjectGenerationContext | None = None,
) -> list[ACPFileEntry]:
    readiness = preview.construction_readiness
    validation = preview.validation
    response_records = response_records or []
    blocking_gaps = [
        gap
        for gap in readiness.gaps
        if gap.severity == "blocking" and gap.status not in {"answered", "resolved"}
    ]
    open_questions = _flatten_open_questions(preview, response_records)
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
            if _is_whatsapp_cloud_tool(tool):
                required_api_contracts.append(_whatsapp_required_api_contract(tool, index))
                required_api_contracts_warning = (
                    "WhatsApp Business Cloud API requiere datos de activacion del cliente antes de conectar sandbox o produccion."
                )
                continue
            if _is_google_workspace_tool(tool):
                required_api_contracts.append(_google_workspace_required_api_contract(tool, index))
                required_api_contracts_warning = (
                    "Google Workspace requiere OAuth, scopes minimos y recursos autorizados antes de conectar sandbox o produccion."
                )
                continue
            if _is_odoo_tool(tool):
                required_api_contracts.append(_odoo_required_api_contract(tool, index))
                required_api_contracts_warning = (
                    "Odoo requiere version/API mode, modelos permitidos, permisos y credenciales referenciadas antes de conectar sandbox o produccion."
                )
                continue
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

    represented_odoo_keys = {
        key
        for key in (_odoo_connector_key(tool) for _, tool in external_tools)
        if key
    }
    inferred_odoo_keys = _odoo_connector_keys_for_snapshot(snapshot) - represented_odoo_keys
    if inferred_odoo_keys:
        required_api_contracts_warning = (
            "Odoo requiere version/API mode, modelos permitidos, permisos y credenciales referenciadas antes de conectar sandbox o produccion."
        )
        next_index = len(external_tools) + 1
        for offset, key in enumerate(sorted(inferred_odoo_keys)):
            required_api_contracts.append(
                _odoo_required_api_contract(
                    _synthetic_odoo_tool(key, index=next_index + offset),
                    next_index + offset,
                )
            )

    overview_payload = {
        **_context_trace_payload(context),
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
            "construction_guide": "ACP/construction-readiness/construction-guide.md",
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
    open_questions_payload = {
        "open_questions": [
            {
                "question_key": question["question_key"],
                "gap_key": question["gap_key"],
                "domain": question["domain"],
                "question_text": question["question_text"],
                "rationale": question["rationale"],
                "target_owner": question["target_owner"],
                "expected_answer_format": question["expected_answer_format"],
                "blocking": question["blocking"],
                "impacted_artifacts": question["impacted_artifacts"],
                "options": question.get("options", []),
            }
            for question in open_questions
        ]
    }
    impact_log_payload = {
        "question_outcomes": impact_outcomes,
        "question_impacts": decision_log,
    }
    structured_deferred_decisions = []
    for item in deferred_decisions:
        question_text = str(item.get("question_text") or "").strip()
        domain = str(item.get("domain") or "general")
        options = item.get("options") or []
        structured_deferred_decisions.append(
            {
                "question_key": item.get("question_key"),
                "domain": domain,
                "gap_key": item.get("gap_key"),
                "status": "deferred_to_implementation",
                "impact_analysis": item.get("impact_analysis"),
                "question_text": question_text,
                "rationale": item.get("rationale") or "",
                "target_owner": item.get("target_owner") or "developer",
                "expected_answer_format": item.get("expected_answer_format") or "decision_with_rationale",
                "impacted_artifacts": list(item.get("impacted_artifacts") or []),
                "options": options,
                "agentic_instruction": {
                    "policy": "DO_NOT_ASSUME_SILENTLY",
                    "behavior": "MUST_PROMPT_USER_DURING_IMPLEMENTATION",
                    "action_required": (
                        "Formular la pregunta durante la implementacion explicando las alternativas "
                        "disponibles y solicitando confirmacion del desarrollador antes de asumir una respuesta."
                    ),
                    "notes": item.get("answer_text") or "Decision delegada formalmente desde la etapa ACP.",
                },
            }
        )
    deferred_decisions_payload = {
        "contract_version": "acp-agentic-deferred.v1",
        "description": "Decisiones pendientes de resolucion transferidas formalmente a la herramienta agentica de implementacion.",
        "execution_policy": "DO_NOT_ASSUME_SILENTLY",
        "deferred_decisions": structured_deferred_decisions,
    }
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
    construction_guide_markdown = _build_construction_step_guide_markdown(
        validation=validation,
        readiness=readiness,
        blocking_gaps=blocking_gaps,
        open_questions=open_questions,
        structured_deferred_decisions=structured_deferred_decisions,
        external_dependencies=external_dependencies,
        required_api_contracts=required_api_contracts,
        deployment_questions=deployment_questions,
        context=context,
    )
    resolution_workflow_payload = {
        "steps": [
            {"order": 1, "action": "read_step_by_step_construction_guide", "path": "ACP/construction-readiness/construction-guide.md"},
            {"order": 2, "action": "read_overview", "path": "ACP/construction-readiness/overview.yaml"},
            {"order": 3, "action": "review_blocking_gaps", "path": "ACP/construction-readiness/blocking-gaps.yaml"},
            {"order": 4, "action": "ask_open_questions", "path": "ACP/construction-readiness/open-questions.yaml"},
            {"order": 5, "action": "review_answer_impact", "path": "ACP/construction-readiness/question-impact-log.yaml"},
            {"order": 6, "action": "register_answers", "path": ACP_CANONICAL_ENV_TEMPLATE_PATH},
            {"order": 7, "action": "review_deferred_decisions", "path": "ACP/construction-readiness/deferred-decisions.yaml"},
            {"order": 8, "action": "recalculate_readiness", "path": "ACP/construction-readiness/overview.yaml"},
            {"order": 9, "action": "continue_to_implementation", "condition": "only_if_can_start_build_true"},
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
            path="ACP/construction-readiness/construction-guide.md",
            domain="construction-readiness",
            title="Step-by-step construction guide",
            format="markdown",
            source_sections=[
                "construction_readiness",
                "construction_readiness.gaps.questions",
                "blueprint.tools",
                "runtime",
                "knowledge",
            ],
            content_text=serialize_markdown_document(construction_guide_markdown),
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


def _apply_question_readiness_overlay(
    preview: ACPPreview,
    response_records: list[ConstructionQuestionResponseRecord] | None,
) -> ACPPreview:
    if not response_records:
        return preview
    readiness = overlay_construction_readiness(preview, response_records)
    return preview.model_copy(update={"construction_readiness": readiness})


def _build_continuity_prompt_files(preview: ACPPreview) -> list[ACPFileEntry]:
    readiness = preview.construction_readiness
    has_whatsapp = any(item.path == "ACP/webhooks/whatsapp-business-webhook.yaml" for item in preview.files)
    has_google_workspace = any(item.path == "ACP/integrations/google-workspace/oauth-policy.yaml" for item in preview.files)
    has_odoo = any(item.path == "ACP/integrations/odoo/api-profile.yaml" for item in preview.files)
    builder_handoff_lines = [
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
            "- Lee primero `ACP/construction-readiness/construction-guide.md` y luego `ACP/construction-readiness/overview.yaml`.",
            "- Usa `ACP/blueprint.graph.json` y `ACP/diagrams/Architecture.md` como mapa vivo antes de tocar runtime, tools o deployment.",
            "- Revisa `blocking-gaps.yaml`, `open-questions.yaml` y `deployment-decisions-needed.yaml` antes de construir.",
            "- Trata tools externas, RAG, runtime y deployment como contratos de diseno hasta que el usuario confirme bindings reales.",
            "- No asumas detalles de deployment, secretos ni contratos API externos cuando aparezcan como gaps abiertos.",
            "- Manten trazabilidad entre `gap_key`, `question_key`, respuesta recibida y artefactos ACP impactados.",
            "- No reabras fases estables del Blueprint por flags stale o deuda operativa interna ya cerrada en el handoff.",
            "- Resuelve o delega cada decision implementable justo antes de modificar el artefacto afectado.",
            "- Si una respuesta contradice el Blueprint aprobado, crea una reconciliacion granular del artefacto afectado; no reinicies fases completas.",
    ]
    if has_whatsapp:
        builder_handoff_lines.extend(
            [
                "",
                "## WhatsApp Business Cloud API",
                "- LAB ya entrega el contrato tecnico del webhook; no preguntes como disenar el webhook.",
                "- Implementa primero `ACP/webhooks/whatsapp-business-webhook.yaml` con GET verification y POST inbound.",
                "- Luego implementa `ACP/integrations/whatsapp/send-message.contract.yaml` y `ACP/integrations/whatsapp/templates.yaml`.",
                "- Pide solo datos de activacion: WABA, phone number, callback URL publica, referencias de secretos, templates aprobados y opt-in.",
                "- No actives produccion hasta pasar `ACP/tools/tests/whatsapp-cloud-api-smoke-test.yaml`.",
            ]
        )
    if has_google_workspace:
        builder_handoff_lines.extend(
            [
                "",
                "## Google Workspace public APIs",
                "- LAB ya entrega contratos de OAuth, scopes y recursos; no conviertas LAB en runtime Google.",
                "- Revisa `ACP/integrations/google-workspace/oauth-policy.yaml` y `scopes-matrix.yaml` antes de implementar.",
                "- Resuelve preguntas de Drive/Sheets/Calendar/Gmail desde `open-questions.yaml` antes de pedir scopes o recursos amplios.",
                "- No guardes client secrets, refresh tokens ni contenido de Drive/Gmail/Calendar en texto plano.",
            ]
        )
    if has_odoo:
        builder_handoff_lines.extend(
            [
                "",
                "## Odoo public/external APIs",
                "- LAB ya entrega contratos de version, modelos y approval; no conviertas LAB en runtime Odoo.",
                "- Revisa `ACP/integrations/odoo/version-policy.yaml` antes de escoger XML-RPC/JSON-RPC o JSON-2.",
                "- Resuelve version, hosting, database, usuario tecnico, modelos/campos y permisos desde `open-questions.yaml`.",
                "- No crees cotizaciones, actividades ni updates sin approval_gate, allowlist e idempotency_key.",
            ]
        )
    builder_handoff = "\n".join(builder_handoff_lines)
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


def _build_implementation_guidance_files(preview: ACPPreview) -> list[ACPFileEntry]:
    readiness = preview.construction_readiness
    has_whatsapp = any(item.path == "ACP/webhooks/whatsapp-business-webhook.yaml" for item in preview.files)
    has_google_workspace = any(item.path == "ACP/integrations/google-workspace/oauth-policy.yaml" for item in preview.files)
    has_odoo = any(item.path == "ACP/integrations/odoo/api-profile.yaml" for item in preview.files)
    lines = [
        "# Implementation Guide",
        "",
        "Este ACP es el paquete de construccion para una herramienta agentica externa.",
        "El Blueprint aprobado se considera la fuente estable de diseno; no debe reabrirse por deuda operativa interna.",
        "",
        "## Politica Blueprint -> ACP",
        "- Deuda de proceso: flags stale, warnings de sincronizacion, gates transitorios y recomendaciones internas quedan cerrados en el handoff.",
        "- Deuda real: preguntas, restricciones o decisiones de negocio/arquitectura/implementacion viajan con trazabilidad.",
        "- Bloqueo critico: si compromete la integridad del Blueprint aprobado, debe tratarse antes de modificar artefactos.",
        "- Reconciliacion granular: una respuesta nueva actualiza solo diagramas, contratos, documentos o artefactos afectados.",
        "",
        "## Como usar el paquete",
        "1. Lee `ACP/README.md` y `ACP/construction-readiness/construction-guide.md`.",
        "2. Usa la guia para recorrer bloqueos, preguntas abiertas, decisiones delegadas, contratos externos, RAG y runtime.",
        "3. Revisa `ACP/construction-readiness/open-questions.yaml` y `deferred-decisions.yaml` como contratos fuente.",
        "4. Antes de implementar un componente, resuelve o conserva como delegada la pregunta asociada a ese componente.",
        "5. Si una decision cambia un entregable, actualiza solo los archivos impactados y registra la razon.",
        "6. Usa `ACP/conformance/portability-report.md` para verificar que el paquete sigue siendo portable.",
        "",
        "## Estado actual de readiness",
        f"- status: {readiness.overall_status}",
        f"- blocking_gaps: {readiness.blocking_gaps}",
        f"- open_questions: {readiness.open_questions}",
        f"- can_start_build: {str(readiness.can_start_build).lower()}",
    ]
    if has_whatsapp:
        lines.extend(
            [
                "",
                "## WhatsApp Business Cloud API",
                "- Construir `GET /webhooks/whatsapp` segun `ACP/webhooks/whatsapp-business-webhook.yaml`.",
                "- Construir `POST /webhooks/whatsapp` con validacion de firma, normalizacion e idempotencia.",
                "- Construir sender REST desacoplado para mensajes de sesion y templates.",
                "- Resolver solo datos de activacion desde `open-questions.yaml`; no pedir al usuario que disene el webhook.",
            ]
        )
    if has_google_workspace:
        lines.extend(
            [
                "",
                "## Google Workspace public APIs",
                "- Construir OAuth, redirect URI, vault y refresh-token flow en el proyecto destino, no dentro de LAB.",
                "- Usar `ACP/integrations/google-workspace/scopes-matrix.yaml` como contrato de scopes minimos.",
                "- Para Drive/Sheets/Calendar/Gmail, implementar solo los contratos presentes bajo `ACP/integrations/google-*` o `ACP/integrations/gmail`.",
                "- Antes de activar sandbox, cerrar las preguntas sobre recurso permitido, owner, scopes y politica de aprobacion.",
            ]
        )
    if has_odoo:
        lines.extend(
            [
                "",
                "## Odoo public/external APIs",
                "- Construir adaptador segun `ACP/integrations/odoo/version-policy.yaml`: XML-RPC/JSON-RPC para 17/18 o JSON-2 para 19 cuando aplique.",
                "- Usar `ACP/integrations/odoo/models-scope.yaml` como allowlist de modelos, campos y dominios.",
                "- Para cotizaciones, aplicar `ACP/integrations/odoo/quote-policy.yaml` antes de tocar `sale.order`.",
                "- Antes de activar sandbox, cerrar version, base URL, database, usuario tecnico, permisos, modelos y write actions permitidas.",
            ]
        )
    checklist = [
        "# Release Readiness Checklist",
        "",
        "- [ ] El paquete fue abierto desde `ACP/README.md` o el viewer generado.",
        "- [ ] Se siguio `ACP/construction-readiness/construction-guide.md` como guia paso a paso.",
        "- [ ] Se revisaron preguntas abiertas y decisiones delegadas.",
        "- [ ] Se confirmo que no viajan flags stale como deuda de implementacion.",
        "- [ ] Los contratos de tools requeridos tienen owner o decision delegada.",
        "- [ ] La estrategia de memoria tiene politica de lectura, escritura y retencion.",
        "- [ ] Las pruebas o rubricas ausentes quedaron documentadas como trabajo de implementacion.",
        "- [ ] Cualquier cambio posterior se aplica mediante reconciliacion granular, no reproceso de fase completa.",
    ]
    return [
        build_acp_file_entry(
            path="ACP/IMPLEMENTATION_GUIDE.md",
            domain="governance",
            title="Implementation guide",
            format="markdown",
            source_sections=["construction_readiness", "blueprint_handoff"],
            content_text=serialize_markdown_document("\n".join(lines)),
        ),
        build_acp_file_entry(
            path="ACP/release-readiness-checklist.md",
            domain="governance",
            title="Release readiness checklist",
            format="markdown",
            source_sections=["construction_readiness", "blueprint_handoff"],
            content_text=serialize_markdown_document("\n".join(checklist)),
        ),
    ]


def _build_operational_cost_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    estimation = snapshot.estimation_report
    agentic = getattr(estimation, "agentic", None)
    provider = str(getattr(getattr(agentic, "active_provider", None), "value", "") or "needs_review")
    model = str(getattr(agentic, "provider_model", "") or "needs_review")
    base_cost_cop = float(getattr(agentic, "provider_runtime_cost_total_cop", 0) or 0)
    base_cost_usd = float(getattr(agentic, "provider_runtime_cost_total_usd", 0) or 0)
    llm_cost_usd = float(getattr(agentic, "llm_runtime_cost_usd", 0) or 0)
    tool_cost_usd = float(getattr(agentic, "tool_runtime_cost_usd", 0) or 0)
    platform_cost_usd = float(getattr(agentic, "platform_overhead_cost_usd", 0) or 0)
    warnings = list(getattr(agentic, "warnings", []) or [])
    if not estimation or not agentic:
        warnings.append("No existe estimacion agentic persistida; costos operativos quedan como plantilla needs_review.")

    def scenario(multiplier: float) -> dict[str, Any]:
        return {
            "per_execution_usd": round(base_cost_usd * multiplier, 4),
            "per_execution_cop": round(base_cost_cop * multiplier, 2),
            "per_100_executions_usd": round(base_cost_usd * multiplier * 100, 4),
            "per_1000_executions_usd": round(base_cost_usd * multiplier * 1000, 4),
            "monthly_1000_executions_usd": round(base_cost_usd * multiplier * 1000, 4),
        }

    payload = {
        "schema_version": "acp-operational-cost-estimate.v1",
        "source_policy": "Derivado de Estimate y telemetria disponible; no ejecuta LLM durante Package.",
        "provider": provider,
        "model": model,
        "currency": "USD",
        "base_components": {
            "llm_runtime_cost_usd": round(llm_cost_usd, 4),
            "tool_runtime_cost_usd": round(tool_cost_usd, 4),
            "platform_overhead_cost_usd": round(platform_cost_usd, 4),
            "provider_runtime_cost_total_usd": round(base_cost_usd, 4),
            "provider_runtime_cost_total_cop": round(base_cost_cop, 2),
        },
        "scenarios": {
            "conservative": scenario(0.75),
            "expected": scenario(1.0),
            "high_consumption": scenario(1.5),
        },
        "assumptions": [
            "Los escenarios escalan linealmente desde la estimacion agentic disponible.",
            "Antes de produccion, reemplazar supuestos por metricas reales de tokens, llamadas a herramientas y latencia.",
            "Mantener separado costo operativo de costo de construccion/desarrollo.",
        ],
        "warnings": warnings,
    }
    markdown_lines = [
        "# Operational Cost Estimate",
        "",
        "Este archivo resume el costo esperado de operar el agente construido desde este ACP.",
        "",
        f"- Proveedor: `{provider}`",
        f"- Modelo: `{model}`",
        f"- Costo esperado por ejecucion: `{_format_usd(base_cost_usd)}`",
        f"- Costo esperado por 100 ejecuciones: `{_format_usd(base_cost_usd * 100)}`",
        f"- Costo esperado por 1000 ejecuciones: `{_format_usd(base_cost_usd * 1000)}`",
        "",
        "## Componentes",
        f"- LLM runtime: `{_format_usd(llm_cost_usd)}`",
        f"- Tools runtime: `{_format_usd(tool_cost_usd)}`",
        f"- Overhead plataforma: `{_format_usd(platform_cost_usd)}`",
        "",
        "## Politica",
        "Package no recalcula costos con LLM. Esta estimacion deriva de Estimate, pricing/telemetria disponible y supuestos explicitados.",
    ]
    if warnings:
        markdown_lines.extend(["", "## Warnings"])
        markdown_lines.extend(f"- {item}" for item in warnings[:8])

    return [
        build_acp_file_entry(
            path="ACP/costs/operational-cost-estimate.json",
            domain="costs",
            title="Operational cost estimate",
            format="json",
            source_sections=["estimation_report", "metric_snapshots", "product_build_telemetry"],
            content_text=serialize_json_document(payload),
            warnings=warnings[:4],
        ),
        build_acp_file_entry(
            path="ACP/costs/operational-cost-estimate.md",
            domain="costs",
            title="Operational cost estimate",
            format="markdown",
            source_sections=["estimation_report", "metric_snapshots", "product_build_telemetry"],
            content_text=serialize_markdown_document("\n".join(markdown_lines)),
            warnings=warnings[:4],
        ),
    ]


def _acp_viewer_stage_for_path(path: str, domain: str) -> str:
    normalized = path.lower()
    if "/ops/agent-flow-map/" in normalized:
        return "flow-map"
    if "/construction-readiness/" in normalized or "readiness" in domain:
        return "readiness"
    if "/evaluation/" in normalized or domain == "evaluation":
        return "validate"
    if "/costs/" in normalized or domain == "costs":
        return "costs"
    if "/governance/" in normalized or "question" in normalized or "decision" in normalized:
        return "decisions"
    if "/ops/" in normalized or "/launcher/" in normalized or "/adapters/" in normalized:
        return "implement"
    if "/conformance/" in normalized or "/manifest" in normalized:
        return "package"
    if domain in {"architecture", "tools", "memory", "runtime", "deployment", "prompts", "workflows"}:
        return "specification"
    return "entry"


def _implementation_prompt_moment(domain: str, blocking: bool, status: str) -> str:
    if blocking:
        return "before_build"
    if status == "deferred":
        return "during_implementation"
    if domain in {"runtime", "deployment", "integrations", "tools", "knowledge", "memory"}:
        return "before_target"
    return "opening_review"


def _implementation_urgency_label(prompt_moment: str) -> str:
    labels = {
        "before_build": "Bloquea construccion",
        "before_target": "Resolver antes de elegir target",
        "during_implementation": "Debe saltar durante implementacion",
        "opening_review": "Revisar al abrir ACP",
    }
    return labels.get(prompt_moment, "Revisar")


def _clamp_score(value: int) -> int:
    return max(0, min(100, value))


def _confidence_label(score: int) -> str:
    if score >= 75:
        return "alta"
    if score >= 50:
        return "media"
    return "baja"


def _target_candidate(
    *,
    target_key: str,
    label: str,
    score: int,
    evidence: list[str],
    warnings: list[str],
    source_files: list[str],
    adapter_doc: str,
    orientation_only: bool = False,
) -> dict[str, Any]:
    normalized_score = _clamp_score(score)
    return {
        "target_key": target_key,
        "label": label,
        "confidence": _confidence_label(normalized_score),
        "score": normalized_score,
        "orientation_only": orientation_only,
        "adapter_doc": adapter_doc,
        "source_files": source_files,
        "evidence": evidence[:5],
        "warnings": warnings[:5],
    }


def _implementation_target_selector(snapshot: SessionSnapshot, preview: ACPPreview) -> dict[str, Any]:
    blueprint = snapshot.blueprint
    readiness = preview.construction_readiness
    external_tools = _iter_external_tools(snapshot)
    tool_count = len(project_blueprint_tools_for_construction(snapshot)) if blueprint is not None else 0
    external_tool_count = len(external_tools)
    workflow_steps = len(blueprint.delivery_package.workflow_profile.steps) if blueprint is not None else 0
    workflow_profile = blueprint.delivery_package.workflow_profile if blueprint is not None else None
    approval_pause = bool(getattr(workflow_profile, "approval_pause", False))
    retry_strategy = str(getattr(workflow_profile, "retry_strategy", "") or "")
    memory_strategy = str(getattr(blueprint, "memory_strategy", "") or "").lower() if blueprint is not None else ""
    reasoning_pattern = str(getattr(blueprint, "reasoning_pattern", "") or "").lower() if blueprint is not None else ""
    architecture = str(getattr(blueprint, "architecture", "") or "").lower() if blueprint is not None else ""
    has_memory = any(token in memory_strategy for token in ["memory", "persistent", "blackboard", "vector"])
    has_complex_reasoning = any(token in reasoning_pattern for token in ["plan", "execute", "reflection", "supervisor"])
    has_supervisor = "supervisor" in architecture
    has_retries = bool(retry_strategy and retry_strategy.lower() not in {"none", "sin_definir", "needs_review"})
    has_evaluation = any(item.path.startswith("ACP/evaluation/") for item in preview.files)
    has_runtime_questions = any(gap.domain in {"runtime", "deployment", "integrations"} for gap in readiness.gaps)
    has_open_questions = readiness.open_questions > 0 or readiness.blocking_gaps > 0
    side_effect_like_tools = [
        tool.name
        for _, tool in external_tools
        if any(token in tool.name.lower() for token in ["write", "create", "update", "delete", "notify", "send", "ingest"])
    ]

    shared_sources = [
        "ACP/adapters/adapter-registry.json",
        "ACP/construction-readiness/overview.yaml",
        "ACP/tools/permissions.yaml",
        "ACP/workflows/durable-workflow.yaml",
        "ACP/runtime/config.yaml",
    ]
    general_warning = (
        "Hay preguntas o decisiones pendientes; bajar confianza y resolver antes de bloquear target final."
        if has_open_questions
        else ""
    )

    candidates = [
        _target_candidate(
            target_key="defer-target",
            label="Diferir decision final",
            score=88 if has_open_questions else 25,
            evidence=[
                f"can_start_build={str(readiness.can_start_build).lower()}",
                f"open_questions={readiness.open_questions}",
                f"blocking_gaps={readiness.blocking_gaps}",
            ],
            warnings=[] if has_open_questions else ["El ACP ya permite seleccionar target con mayor confianza."],
            source_files=[
                "ACP/construction-readiness/construction-guide.md",
                "ACP/construction-readiness/overview.yaml",
                "ACP/construction-readiness/open-questions.yaml",
                "ACP/construction-readiness/deferred-decisions.yaml",
            ],
            adapter_doc="ACP/construction-readiness/resolution-workflow.yaml",
            orientation_only=True,
        ),
        _target_candidate(
            target_key="pure-code",
            label="Codigo puro",
            score=62
            + (12 if external_tool_count else 0)
            + (10 if side_effect_like_tools else 0)
            + (8 if has_runtime_questions else 0)
            + (8 if has_memory else 0)
            - (10 if has_open_questions else 0),
            evidence=[
                f"{tool_count} herramientas definidas",
                f"{external_tool_count} herramientas externas",
                "Mayor control sobre runtime, secretos, side effects y deployment.",
            ],
            warnings=[item for item in [general_warning, "Requiere equipo tecnico e infraestructura."] if item],
            source_files=shared_sources + ["ACP/adapters/pure-code.md"],
            adapter_doc="ACP/adapters/pure-code.md",
        ),
        _target_candidate(
            target_key="openai-agents-sdk",
            label="OpenAI Agents SDK",
            score=60
            + (12 if tool_count else 0)
            + (8 if has_evaluation else 0)
            + (8 if has_complex_reasoning else 0)
            - (10 if has_open_questions else 0),
            evidence=[
                "El ACP contiene prompts, tools y evaluacion mapeables a SDK.",
                f"reasoning_pattern={reasoning_pattern or 'needs_review'}",
                f"evaluation_present={str(has_evaluation).lower()}",
            ],
            warnings=[item for item in [general_warning, "Confirmar provider, modelos, secretos y deployment antes de codificar."] if item],
            source_files=shared_sources + ["ACP/adapters/openai-agents-sdk.md"],
            adapter_doc="ACP/adapters/openai-agents-sdk.md",
        ),
        _target_candidate(
            target_key="langgraph",
            label="LangGraph",
            score=55
            + (12 if workflow_steps >= 3 else 0)
            + (10 if approval_pause else 0)
            + (8 if has_retries else 0)
            + (8 if has_memory else 0)
            - (10 if has_open_questions else 0),
            evidence=[
                f"workflow_steps={workflow_steps}",
                f"approval_pause={str(approval_pause).lower()}",
                f"retry_strategy={retry_strategy or 'needs_review'}",
            ],
            warnings=[item for item in [general_warning, "Requiere modelar estado, retries y handoffs con disciplina tecnica."] if item],
            source_files=shared_sources + ["ACP/workflows/langgraph.json", "ACP/adapters/langgraph.md"],
            adapter_doc="ACP/adapters/langgraph.md",
        ),
        _target_candidate(
            target_key="n8n",
            label="n8n",
            score=46
            + (12 if external_tool_count else 0)
            + (8 if approval_pause else 0)
            - (12 if has_memory else 0)
            - (10 if has_supervisor or has_complex_reasoning else 0)
            - (12 if has_open_questions else 0),
            evidence=[
                "Viable para automatizaciones por nodos si APIs y aprobaciones estan claras.",
                f"external_tools={external_tool_count}",
                f"approval_pause={str(approval_pause).lower()}",
            ],
            warnings=[
                item
                for item in [
                    general_warning,
                    "No crear workflow real sin credenciales, endpoints, payloads y politica HITL confirmados.",
                    "No ideal para memoria avanzada, estado fino o reasoning complejo.",
                ]
                if item
            ],
            source_files=shared_sources + ["ACP/adapters/n8n.md"],
            adapter_doc="ACP/adapters/n8n.md",
            orientation_only=True,
        ),
        _target_candidate(
            target_key="make",
            label="Make",
            score=42
            + (10 if external_tool_count else 0)
            + (8 if workflow_steps <= 4 else 0)
            - (12 if has_memory else 0)
            - (10 if has_supervisor or has_complex_reasoning else 0)
            - (12 if has_open_questions else 0),
            evidence=[
                "Viable para escenarios SaaS lineales y subflujos acotados.",
                f"workflow_steps={workflow_steps}",
                f"external_tools={external_tool_count}",
            ],
            warnings=[
                item
                for item in [
                    general_warning,
                    "Menos adecuado para agentes con retries complejos, memoria persistente o gobernanza avanzada.",
                ]
                if item
            ],
            source_files=shared_sources + ["ACP/adapters/make.md"],
            adapter_doc="ACP/adapters/make.md",
            orientation_only=True,
        ),
    ]
    candidates = sorted(candidates, key=lambda item: int(item["score"]), reverse=True)
    recommended = candidates[0] if candidates else {}
    return {
        "schema_version": "acp-implementation-target-selector.v1",
        "source_files": shared_sources,
        "recommended_target_key": recommended.get("target_key", ""),
        "recommended_label": recommended.get("label", ""),
        "confidence": recommended.get("confidence", "baja"),
        "reason": (
            "Diferir el target final hasta cerrar readiness."
            if recommended.get("target_key") == "defer-target"
            else "Target recomendado segun arquitectura, tools, workflow y readiness del ACP."
        ),
        "candidates": candidates,
        "rules": [
            "No generar proyectos reales en n8n o Make sin credenciales, permisos y confirmacion explicita.",
            "No cambiar la arquitectura aprobada; el target materializa el ACP existente.",
            "Si can_start_build=false, resolver o delegar explicitamente antes de bloquear target final.",
        ],
    }


def _implementation_cockpit(
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
) -> dict[str, Any]:
    records = response_records or []
    readiness = preview.construction_readiness
    question_views = build_construction_question_views(preview, records)
    deferred_decisions = build_deferred_construction_decision_backlog(preview, records)

    decision_cards: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for question in question_views:
        if question.status != "open":
            continue
        moment = _implementation_prompt_moment(question.domain, question.blocking, question.status)
        key = f"open:{question.question_key}"
        seen_keys.add(key)
        decision_cards.append(
            {
                "key": key,
                "question_key": question.question_key,
                "status": "open",
                "domain": question.domain or "general",
                "question_text": question.question_text,
                "target_owner": question.target_owner or "builder",
                "source_file": "ACP/construction-readiness/open-questions.yaml",
                "prompt_moment": moment,
                "urgency_label": _implementation_urgency_label(moment),
                "blocking": question.blocking,
                "must_prompt": question.blocking or moment in {"before_target", "before_build"},
                "impact_summary": (
                    question.impact_analysis.impact_summary
                    if question.impact_analysis is not None
                    else question.rationale
                ),
                "impacted_artifacts": list(question.impacted_artifacts or [])[:6],
                "option_count": len(question.options or []),
            }
        )

    for item in deferred_decisions:
        question_key = str(item.get("question_key") or "")
        key = f"deferred:{question_key}"
        if key in seen_keys:
            continue
        domain = str(item.get("domain") or "general")
        moment = _implementation_prompt_moment(domain, False, "deferred")
        decision_cards.append(
            {
                "key": key,
                "question_key": question_key,
                "status": "deferred_to_implementation",
                "domain": domain,
                "question_text": str(item.get("question_text") or ""),
                "target_owner": str(item.get("target_owner") or "builder"),
                "source_file": "ACP/construction-readiness/deferred-decisions.yaml",
                "prompt_moment": moment,
                "urgency_label": _implementation_urgency_label(moment),
                "blocking": False,
                "must_prompt": True,
                "impact_summary": str(item.get("rationale") or "Decision delegada formalmente a implementacion."),
                "impacted_artifacts": list((item.get("impacted_artifacts") or []))[:6],
                "option_count": len(item.get("options") or []),
            }
        )

    prompt_order = {
        "before_build": 0,
        "before_target": 1,
        "during_implementation": 2,
        "opening_review": 3,
    }
    decision_cards = sorted(
        decision_cards,
        key=lambda item: (
            prompt_order.get(str(item.get("prompt_moment")), 9),
            str(item.get("domain") or ""),
            str(item.get("question_key") or ""),
        ),
    )

    if readiness.can_start_build:
        headline = "Listo para iniciar construccion"
        recommended_action = "start_agentic_build"
    elif readiness.blocking_gaps > 0:
        headline = "Resolver bloqueos antes de construir"
        recommended_action = "resolve_blocking_construction_gaps"
    elif decision_cards:
        headline = "Resolver decisiones antes de construir con confianza"
        recommended_action = "answer_open_questions"
    else:
        headline = "Revisar supuestos antes de avanzar"
        recommended_action = readiness.next_recommended_action

    return {
        "schema_version": "acp-implementation-cockpit.v1",
        "source_files": [
            "ACP/construction-readiness/construction-guide.md",
            "ACP/construction-readiness/overview.yaml",
            "ACP/construction-readiness/open-questions.yaml",
            "ACP/construction-readiness/deferred-decisions.yaml",
            "ACP/construction-readiness/question-impact-log.yaml",
            "ACP/construction-readiness/resolution-workflow.yaml",
        ],
        "readiness": {
            "headline": headline,
            "overall_status": readiness.overall_status,
            "can_export_zip": preview.validation.can_export_zip,
            "can_start_build": readiness.can_start_build,
            "blocking_gaps": readiness.blocking_gaps,
            "open_questions": readiness.open_questions,
            "assumptions_count": readiness.assumptions_count,
            "next_recommended_action": recommended_action,
        },
        "prompt_policy": {
            "policy": "DO_NOT_ASSUME_SILENTLY",
            "behavior": "MUST_PROMPT_USER_DURING_IMPLEMENTATION",
            "summary": (
                "Las preguntas abiertas o delegadas deben mostrarse antes de elegir target, "
                "antes de construir o durante implementacion segun el dominio afectado."
            ),
        },
        "decision_queue": decision_cards[:18],
        "decision_queue_total": len(decision_cards),
        "target_selector": _implementation_target_selector(snapshot, preview),
        "interaction_rules": [
            {
                "moment": "opening_review",
                "label": "Al abrir el ACP",
                "rule": "Mostrar estado general, preguntas abiertas y decisiones delegadas sin bloquear la lectura inicial.",
            },
            {
                "moment": "before_target",
                "label": "Antes de elegir target",
                "rule": "Pedir decisiones que afecten runtime, deployment, tools, memoria, conocimiento o integraciones.",
            },
            {
                "moment": "before_build",
                "label": "Antes de construir",
                "rule": "Bloquear avance si la decision es marcada como bloqueante o impide can_start_build=true.",
            },
            {
                "moment": "during_implementation",
                "label": "Durante implementacion",
                "rule": "Hacer saltar las decisiones delegadas justo cuando el builder toque el dominio afectado.",
            },
        ],
    }


def _build_acp_navigation_manifest(
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for file in sorted(preview.files, key=lambda item: item.path):
        if file.path in {"ACP/index.html", "ACP/navigation-manifest.v1.json"} or file.path.startswith("ACP/assets/"):
            continue
        stage = _acp_viewer_stage_for_path(file.path, file.domain)
        items.append(
            {
                "id": _slugify(file.path, default="file"),
                "path": file.path,
                "title": file.title or _title_from_path(file.path),
                "domain": file.domain,
                "stage": stage,
                "format": file.format,
                "status": file.status,
                "description": "; ".join(file.warnings[:2]) or f"Archivo ACP de dominio {file.domain or 'general'}.",
                "source_sections": list(file.source_sections or []),
            }
        )
    chapters = [
        {
            "id": "entry",
            "title": "Entrada aprobada",
            "narrative": "El ACP parte del Blueprint aprobado como verdad estable. No reabre etapas anteriores; convierte lo generado en insumos implementables.",
            "why_it_matters": "Evita que la implementacion pierda contexto o reactive deuda operativa ya cerrada.",
        },
        {
            "id": "validate",
            "title": "Validar",
            "narrative": "Agrupa pruebas, rubricas y simulacion para confirmar el comportamiento esperado antes de construir.",
            "why_it_matters": "Convierte preguntas y riesgos en evidencia verificable.",
        },
        {
            "id": "decisions",
            "title": "Decisiones y gaps",
            "narrative": "Muestra decisiones respondidas, delegadas o pendientes con impacto y momento recomendado de cierre.",
            "why_it_matters": "Permite avanzar sin ocultar deuda real ni regalar decisiones de implementacion.",
        },
        {
            "id": "specification",
            "title": "Especificacion implementable",
            "narrative": "Reune arquitectura, herramientas, memoria, prompts, runtime, workflows y deployment como contrato tecnico.",
            "why_it_matters": "Le da a la herramienta agentica los limites, componentes y responsabilidades necesarios para construir.",
        },
        {
            "id": "flow-map",
            "title": "Mapa vivo del agente",
            "narrative": "Visualiza como se mueve la informacion entre usuario, planner, RAG, memoria, tools, handoff, fallbacks, costos y observabilidad.",
            "why_it_matters": "Hace entendible y vendible el ACP sin desconectarse de sus contratos fuente.",
        },
        {
            "id": "costs",
            "title": "Costos operativos",
            "narrative": "Separa el costo de operar el agente del costo de construirlo, con escenarios trazables.",
            "why_it_matters": "Ayuda a tomar decisiones de modelo, volumen y presupuesto antes de produccion.",
        },
        {
            "id": "readiness",
            "title": "Readiness",
            "narrative": "Resume si el paquete es consumible por una persona o herramienta agentica y que falta tratar.",
            "why_it_matters": "Bloquea solo faltantes estructurales; las brechas holisticas viajan como preguntas/delegaciones.",
        },
        {
            "id": "package",
            "title": "Package",
            "narrative": "Organiza archivos, manifest, conformance y viewer sin llamar LLM ni redisenar.",
            "why_it_matters": "Garantiza portabilidad offline y una unica fuente de verdad basada en los archivos generados.",
        },
        {
            "id": "implement",
            "title": "Implementar",
            "narrative": "Entrega guia, launcher y adapters para iniciar la construccion con Codex, Cursor, Antigravity u otra herramienta compatible.",
            "why_it_matters": "Permite que un tercero arranque sin reconstruir contexto desde LAB.",
        },
    ]
    for index, chapter in enumerate(chapters):
        chapter_items = [item for item in items if item["stage"] == chapter["id"]]
        chapter["key_takeaways"] = [
            f"{len(chapter_items)} archivo(s) relacionados.",
            f"Readiness actual: {preview.construction_readiness.overall_status}.",
        ]
        chapter["related_files"] = [item["id"] for item in chapter_items[:12]]
        chapter["next_chapter_id"] = chapters[index + 1]["id"] if index + 1 < len(chapters) else ""
    return {
        "contract_version": "acp-navigation-manifest.v1",
        "package_type": "agent_construction_package",
        "session_id": str(snapshot.session.id),
        "title": snapshot.session.title or "Agent Construction Package",
        "blueprint_version_number": preview.blueprint_version_number,
        "validation_status": preview.validation.overall_status,
        "can_export_zip": preview.validation.can_export_zip,
        "construction_readiness": preview.construction_readiness.model_dump(mode="json"),
        "implementation_cockpit": _implementation_cockpit(snapshot, preview, response_records),
        "agent_flow_map": _agent_flow_map_payload(snapshot),
        "storyline": chapters,
        "items": items,
    }


def _acp_viewer_css() -> str:
    return """
:root { color-scheme: light; --ink:#111827; --muted:#566174; --line:#d7deeb; --panel:#ffffff; --soft:#f4f7fb; --brand:#2f43bd; --accent:#2f7d52; --warn:#9a5a00; --danger:#b42318; }
* { box-sizing:border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:linear-gradient(135deg,#fbfcff,#eef4ff); }
a { color:inherit; }
.shell { display:grid; grid-template-columns:300px minmax(0,1fr); min-height:100vh; }
.sidebar { position:sticky; top:0; height:100vh; overflow:auto; padding:24px; border-right:1px solid var(--line); background:rgba(255,255,255,.88); backdrop-filter:blur(10px); }
.brand { font-size:12px; letter-spacing:.18em; text-transform:uppercase; font-weight:900; color:var(--brand); }
.title { margin:10px 0 8px; font-size:25px; line-height:1.08; }
.meta { color:var(--muted); font-size:12px; line-height:1.5; }
.search { width:100%; margin:18px 0 14px; border:1px solid var(--line); border-radius:14px; padding:10px 12px; }
.nav { display:grid; gap:8px; }
.nav button { border:1px solid var(--line); background:white; border-radius:14px; padding:10px 12px; text-align:left; font-weight:850; cursor:pointer; }
.nav button.active { border-color:var(--brand); color:var(--brand); box-shadow:0 10px 24px rgba(47,67,189,.13); }
.main { padding:34px; }
.hero,.chapter,.card { border:1px solid var(--line); border-radius:24px; background:rgba(255,255,255,.94); box-shadow:0 16px 35px rgba(17,24,39,.07); }
.hero { padding:28px; margin-bottom:22px; }
.hero h1 { margin:0; font-size:34px; }
.hero p,.chapter p { color:var(--muted); line-height:1.7; }
.cockpit { margin-top:22px; border:1px solid var(--line); border-radius:8px; background:#fbfcff; overflow:hidden; }
.cockpit-head { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(240px,.8fr); gap:16px; padding:18px; border-bottom:1px solid var(--line); }
.cockpit h2 { margin:4px 0 8px; font-size:21px; line-height:1.2; }
.cockpit p { margin:0; color:var(--muted); font-size:13px; line-height:1.55; }
.cockpit-status { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
.metric { border:1px solid var(--line); border-radius:8px; padding:10px; background:white; min-height:66px; }
.metric strong { display:block; font-size:20px; line-height:1; }
.metric span { display:block; margin-top:6px; color:var(--muted); font-size:11px; font-weight:850; text-transform:uppercase; letter-spacing:.08em; }
.gate { display:inline-flex; width:max-content; max-width:100%; border-radius:999px; padding:6px 9px; font-size:12px; font-weight:900; background:#eaf7ef; color:var(--accent); }
.gate.warn { background:#fff3df; color:var(--warn); }
.gate.danger { background:#fff0f0; color:var(--danger); }
.decision-queue { display:grid; gap:10px; padding:14px 18px 18px; }
.decision-card { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:12px; border:1px solid var(--line); border-radius:8px; background:white; padding:12px; }
.decision-card h3 { margin:4px 0 6px; font-size:14px; line-height:1.35; }
.decision-card p { font-size:12px; }
.target-selector { padding:0 18px 18px; }
.target-head { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }
.target-head h3 { margin:0; font-size:15px; }
.target-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }
.target-card { border:1px solid var(--line); border-radius:8px; background:white; padding:12px; }
.target-card.recommended { border-color:var(--brand); box-shadow:0 10px 22px rgba(47,67,189,.10); }
.target-score { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }
.target-score strong { font-size:14px; }
.score { border-radius:999px; padding:5px 8px; background:#eaf0ff; color:var(--brand); font-size:12px; font-weight:900; }
.target-card p { margin:0 0 8px; font-size:12px; color:var(--muted); line-height:1.45; }
.target-card ul { margin:8px 0 0; padding-left:18px; color:var(--muted); font-size:12px; line-height:1.45; }
.target-card a { display:inline-flex; margin-top:10px; border-radius:8px; background:var(--soft); color:var(--brand); padding:7px 9px; text-decoration:none; font-weight:850; font-size:12px; }
.decision-meta { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.tag { border-radius:999px; background:var(--soft); color:var(--muted); padding:5px 8px; font-size:11px; font-weight:850; }
.tag.urgent { background:#fff3df; color:var(--warn); }
.tag.blocking { background:#fff0f0; color:var(--danger); }
.source-link { align-self:start; white-space:nowrap; border-radius:8px; background:var(--soft); color:var(--brand); padding:8px 10px; text-decoration:none; font-weight:850; font-size:12px; }
.rules { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:8px; padding:0 18px 18px; }
.rule { border:1px solid var(--line); border-radius:8px; background:white; padding:10px; }
.rule strong { display:block; font-size:12px; margin-bottom:4px; }
.rule span { color:var(--muted); font-size:12px; line-height:1.45; }
.chapter { padding:28px; }
.eyebrow { color:var(--brand); font-size:12px; font-weight:900; letter-spacing:.18em; text-transform:uppercase; }
.chapter h2 { margin:8px 0 12px; font-size:30px; }
.pills { display:flex; flex-wrap:wrap; gap:8px; margin:16px 0; }
.pill { border-radius:999px; background:#eaf0ff; color:var(--brand); padding:7px 10px; font-size:12px; font-weight:850; }
.section-group { margin-top:24px; padding-top:18px; border-top:1px dashed var(--line); }
.section-group:first-of-type { border-top:none; padding-top:0; margin-top:14px; }
.section-header { font-size:14px; font-weight:850; color:var(--ink); margin-bottom:12px; display:flex; align-items:center; gap:8px; text-transform:uppercase; letter-spacing:.06em; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; margin-top:10px; }
.card { padding:16px; }
.card small { color:var(--muted); text-transform:uppercase; letter-spacing:.12em; font-weight:850; }
.card h3 { margin:8px 0; font-size:16px; }
.card p { margin:0 0 12px; font-size:13px; color:var(--muted); }
.card a { display:inline-flex; padding:8px 10px; border-radius:12px; background:var(--soft); color:var(--brand); text-decoration:none; font-weight:850; font-size:13px; }
.status-needs_review { border-left:4px solid var(--warn); }
.status-incomplete { border-left:4px solid #b42318; }
.flow-shell { margin:18px 0 22px; border:1px solid #162033; border-radius:8px; overflow:hidden; background:#07111f; color:#eef6ff; box-shadow:0 22px 55px rgba(7,17,31,.22); }
.flow-toolbar { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px; padding:14px; border-bottom:1px solid rgba(218,226,240,.18); background:linear-gradient(90deg,#07111f,#0d2730); }
.flow-toolbar h3 { margin:0; font-size:15px; color:white; }
.flow-tools { display:flex; flex-wrap:wrap; gap:8px; }
.flow-tools button,.flow-tools select { border:1px solid rgba(238,246,255,.24); border-radius:8px; background:rgba(255,255,255,.08); color:#eef6ff; padding:8px 10px; font-weight:850; cursor:pointer; }
.flow-tools button.active { background:#20c997; color:#06251c; border-color:#20c997; }
.flow-stage { display:grid; grid-template-columns:minmax(0,1fr) 280px; min-height:520px; }
.flow-canvas { position:relative; min-height:520px; overflow:hidden; background:radial-gradient(circle at 20% 20%,rgba(32,201,151,.18),transparent 28%),radial-gradient(circle at 75% 35%,rgba(255,193,7,.16),transparent 24%),linear-gradient(135deg,#07111f,#102131); }
.flow-canvas::before { content:""; position:absolute; inset:0; background-image:linear-gradient(rgba(255,255,255,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.045) 1px,transparent 1px); background-size:42px 42px; mask-image:linear-gradient(to bottom,rgba(0,0,0,.85),rgba(0,0,0,.25)); }
.flow-wires { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.flow-edge { stroke:#8be9d2; stroke-width:.32; stroke-linecap:round; opacity:.68; stroke-dasharray:1.4 1.2; animation:flowDash 2.2s linear infinite; }
.flow-edge.approval { stroke:#ffd166; }
.flow-edge.fallback { stroke:#ff7b7b; }
.flow-shell.paused .flow-edge { animation-play-state:paused; }
.flow-node { position:absolute; width:138px; min-height:74px; transform:translate(-50%,-50%); border:1px solid rgba(238,246,255,.24); border-radius:8px; padding:10px; background:rgba(9,21,34,.88); color:#eef6ff; box-shadow:0 16px 35px rgba(0,0,0,.22); cursor:pointer; backdrop-filter:blur(10px); transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease,opacity .18s ease; z-index:2; }
.flow-node:hover,.flow-node.selected { transform:translate(-50%,-50%) scale(1.04); border-color:#20c997; box-shadow:0 18px 38px rgba(32,201,151,.22); }
.flow-node.dimmed { opacity:.28; }
.flow-node strong { display:block; font-size:13px; line-height:1.25; }
.flow-node span { display:block; margin-top:6px; color:#a9bacd; font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:850; }
.flow-state { display:inline-flex; margin-top:8px; border-radius:999px; padding:4px 7px; font-size:10px; font-weight:900; background:rgba(32,201,151,.15); color:#8be9d2; }
.state-needs_binding,.state-approval_required { background:rgba(255,209,102,.16); color:#ffd166; }
.state-blocked,.state-degraded { background:rgba(255,123,123,.16); color:#ff9b9b; }
.state-fallback_available { background:rgba(125,211,252,.16); color:#7dd3fc; }
.flow-panel { border-left:1px solid rgba(218,226,240,.18); padding:16px; background:#091522; color:#dbeafe; }
.flow-panel h4 { margin:0 0 8px; color:white; font-size:16px; }
.flow-panel p { color:#a9bacd; font-size:13px; line-height:1.55; }
.flow-panel a { display:block; margin-top:8px; color:#8be9d2; text-decoration:none; font-size:12px; font-weight:850; overflow-wrap:anywhere; }
.flow-caption { position:absolute; left:18px; bottom:18px; max-width:440px; border:1px solid rgba(238,246,255,.18); border-radius:8px; padding:12px; background:rgba(7,17,31,.82); color:#dbeafe; font-size:13px; line-height:1.45; z-index:3; }
@keyframes flowDash { to { stroke-dashoffset:-8; } }
@media (prefers-reduced-motion:reduce){ .flow-edge{animation:none;} .flow-node{transition:none;} }
.controls { display:flex; justify-content:space-between; gap:12px; margin-top:24px; }
.controls button { border:0; border-radius:14px; padding:12px 16px; background:var(--brand); color:white; font-weight:900; cursor:pointer; }
.controls button.secondary { background:white; color:var(--brand); border:1px solid var(--line); }
@media (max-width:860px){ .shell{grid-template-columns:1fr;} .sidebar{position:relative;height:auto;} .main{padding:18px;} .hero h1{font-size:28px;} .cockpit-head{grid-template-columns:1fr;} .decision-card{grid-template-columns:1fr;} .source-link{width:max-content;} .flow-stage{grid-template-columns:1fr;} .flow-panel{border-left:0;border-top:1px solid rgba(218,226,240,.18);} .flow-node{width:118px;} }
""".strip()


def _acp_viewer_js() -> str:
    return """
(function(){
  const data = window.ACP_NAVIGATION_MANIFEST || {storyline:[], items:[]};
  const nav = document.querySelector('[data-nav]');
  const chapter = document.querySelector('[data-chapter]');
  const cockpitNode = document.querySelector('[data-cockpit]');
  const search = document.querySelector('[data-search]');
  let current = 0;
  let flowMode = 'design';
  let flowPlaying = true;
  let flowLayer = 'all';
  let selectedFlowNode = '';
  function esc(value){ return String(value || '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
  function shortPath(path){ return String(path || '').replace(/^ACP\\//, ''); }
  function gateClass(readiness){
    if(readiness.can_start_build){ return 'gate'; }
    if((readiness.blocking_gaps || 0) > 0){ return 'gate danger'; }
    return 'gate warn';
  }
  function renderCockpit(){
    if(!cockpitNode){ return; }
    const cockpit = data.implementation_cockpit || {};
    const readiness = cockpit.readiness || {};
    const queue = cockpit.decision_queue || [];
    const targetSelector = cockpit.target_selector || {};
    const rules = cockpit.interaction_rules || [];
    const queueTotal = cockpit.decision_queue_total || queue.length;
    const sourceFiles = cockpit.source_files || [];
    const visibleQueue = queue.slice(0, 6);
    const statusText = readiness.can_start_build ? 'Construccion habilitada' : 'Decisiones pendientes';
    const queueHtml = visibleQueue.length ? visibleQueue.map(item => {
      const source = shortPath(item.source_file || 'ACP/construction-readiness/open-questions.yaml');
      const tags = [
        `<span class="tag urgent">${esc(item.urgency_label || 'Revisar')}</span>`,
        `<span class="tag">${esc(item.domain || 'general')}</span>`,
        `<span class="tag">${esc(item.target_owner || 'builder')}</span>`,
      ];
      if(item.blocking){ tags.unshift('<span class="tag blocking">Bloqueante</span>'); }
      if(item.must_prompt){ tags.push('<span class="tag blocking">No asumir</span>'); }
      return `<article class="decision-card">
        <div>
          <small>${esc(item.status || 'open')} · ${esc(item.prompt_moment || 'review')}</small>
          <h3>${esc(item.question_text || item.question_key || 'Decision pendiente')}</h3>
          <p>${esc(item.impact_summary || 'Revisar impacto en los artefactos fuente antes de implementar.')}</p>
          <div class="decision-meta">${tags.join('')}</div>
        </div>
        <a class="source-link" href="${esc(source)}" target="_blank" rel="noreferrer">Fuente</a>
      </article>`;
    }).join('') : '<p>No hay preguntas abiertas ni decisiones delegadas en la cola de implementacion.</p>';
    const rulesHtml = rules.map(rule => `<div class="rule"><strong>${esc(rule.label)}</strong><span>${esc(rule.rule)}</span></div>`).join('');
    const targetCandidates = (targetSelector.candidates || []).slice(0, 6);
    const targetHtml = targetCandidates.length ? `<div class="target-selector">
      <div class="target-head">
        <h3>Implementation Target Selector</h3>
        <span class="gate ${targetSelector.recommended_target_key === 'defer-target' ? 'warn' : ''}">${esc(targetSelector.recommended_label || 'Sin recomendacion')} · ${esc(targetSelector.confidence || 'baja')}</span>
      </div>
      <p>${esc(targetSelector.reason || 'Recomendacion derivada de adapter registry, readiness, tools y workflows.')}</p>
      <div class="target-grid">${targetCandidates.map(item => {
        const doc = shortPath(item.adapter_doc || '');
        const evidence = (item.evidence || []).slice(0, 3).map(value => `<li>${esc(value)}</li>`).join('');
        const warnings = (item.warnings || []).slice(0, 2).map(value => `<li>${esc(value)}</li>`).join('');
        const recommended = item.target_key === targetSelector.recommended_target_key;
        return `<article class="target-card ${recommended ? 'recommended' : ''}">
          <div class="target-score"><strong>${esc(item.label)}</strong><span class="score">${esc(item.score)} · ${esc(item.confidence)}</span></div>
          <p>${esc(item.orientation_only ? 'Orientacion, no despliegue automatico.' : 'Target tecnico de implementacion.')}</p>
          ${evidence ? `<ul>${evidence}</ul>` : ''}
          ${warnings ? `<ul>${warnings}</ul>` : ''}
          ${doc ? `<a href="${esc(doc)}" target="_blank" rel="noreferrer">Ver adapter</a>` : ''}
        </article>`;
      }).join('')}</div>
    </div>` : '';
    cockpitNode.innerHTML = `<section class="cockpit" aria-label="Implementation cockpit">
      <div class="cockpit-head">
        <div>
          <span class="${gateClass(readiness)}">${esc(statusText)}</span>
          <h2>${esc(readiness.headline || 'Estado de implementacion')}</h2>
          <p>${esc((cockpit.prompt_policy || {}).summary || 'Las decisiones de implementacion se muestran desde los artefactos de readiness del ACP.')}</p>
          <div class="decision-meta">${sourceFiles.slice(0, 4).map(path => `<span class="tag">${esc(shortPath(path))}</span>`).join('')}</div>
        </div>
        <div class="cockpit-status">
          <div class="metric"><strong>${esc(readiness.can_start_build ? 'Si' : 'No')}</strong><span>Can start build</span></div>
          <div class="metric"><strong>${esc(readiness.open_questions || 0)}</strong><span>Preguntas abiertas</span></div>
          <div class="metric"><strong>${esc(readiness.blocking_gaps || 0)}</strong><span>Bloqueos</span></div>
          <div class="metric"><strong>${esc(queueTotal)}</strong><span>Cards en cola</span></div>
        </div>
      </div>
      <div class="decision-queue">${queueHtml}</div>
      ${targetHtml}
      <div class="rules">${rulesHtml}</div>
    </section>`;
  }
  function fileCard(item){
    const path = (item.path || '').replace(/^ACP\\//, '');
    const isSvg = path.endsWith('.svg');
    const isMd = path.endsWith('.md');
    const isJson = path.endsWith('.json');
    const isMmd = path.endsWith('.mmd') || path.endsWith('.mermaid');
    const isXml = path.endsWith('.xml');
    let badge = esc(item.domain || 'archivo');
    let actionLabel = 'Abrir archivo';
    if (isSvg) { badge = 'DIAGRAMA SVG'; actionLabel = 'Ver diagrama SVG'; }
    else if (isMd) { badge = 'DOCUMENTO'; actionLabel = 'Leer documento'; }
    else if (isJson) { badge = 'CONTRATO JSON'; actionLabel = 'Ver contrato JSON'; }
    else if (isMmd) { badge = 'MERMAID'; actionLabel = 'Ver Mermaid'; }
    else if (isXml) { badge = 'BPMN XML'; actionLabel = 'Ver XML BPMN'; }

    return `<article class="card status-${esc(item.status)}"><small>${badge} · ${esc(item.status)}</small><h3>${esc(item.title)}</h3><p>${esc(item.description)}</p><a href="${esc(path)}" target="_blank" rel="noreferrer">${esc(actionLabel)}</a></article>`;
  }
  function renderSegmentedFiles(filesList){
    const svgItems = [];
    const mmdItems = [];
    const jsonItems = [];
    const mdItems = [];
    const otherItems = [];

    (filesList || []).forEach(item => {
      const path = (item.path || '').toLowerCase();
      const title = (item.title || '').toLowerCase();
      if (path.endsWith('.svg')) {
        svgItems.push(item);
      } else if (path.endsWith('.mmd') || path.endsWith('.mermaid') || path.endsWith('.xml') || title.includes('mermaid') || title.includes('bpmn')) {
        mmdItems.push(item);
      } else if (path.endsWith('.json')) {
        jsonItems.push(item);
      } else if (path.endsWith('.md')) {
        mdItems.push(item);
      } else {
        otherItems.push(item);
      }
    });

    let html = '';

    if (svgItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">🎨 Diagramas Visuales Vectoriales (SVG) (${svgItems.length})</div>
        <div class="grid">${svgItems.map(fileCard).join('')}</div>
      </div>`;
    }

    if (mmdItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">📐 Código & Especificaciones de Diagrama (Mermaid / BPMN) (${mmdItems.length})</div>
        <div class="grid">${mmdItems.map(fileCard).join('')}</div>
      </div>`;
    }

    if (jsonItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">📜 Contratos Semánticos, Modelos & Calidad (.json) (${jsonItems.length})</div>
        <div class="grid">${jsonItems.map(fileCard).join('')}</div>
      </div>`;
    }

    if (mdItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">📄 Documentación & Entregables (.md) (${mdItems.length})</div>
        <div class="grid">${mdItems.map(fileCard).join('')}</div>
      </div>`;
    }

    if (otherItems.length > 0) {
      html += `<div class="section-group">
        <div class="section-header">📦 Otros Artefactos (${otherItems.length})</div>
        <div class="grid">${otherItems.map(fileCard).join('')}</div>
      </div>`;
    }

    return html || '<p>No hay archivos en este capítulo.</p>';
  }
  function stateLabel(state){
    return String(state || 'defined').replace(/_/g, ' ');
  }
  function flowNodeById(flow, id){
    return (flow.nodes || []).find(node => node.id === id) || null;
  }
  function renderFlowMap(){
    const flow = data.agent_flow_map || {};
    const nodes = flow.nodes || [];
    if(!nodes.length){ return ''; }
    if(!selectedFlowNode || !flowNodeById(flow, selectedFlowNode)){ selectedFlowNode = nodes[0].id; }
    const selected = flowNodeById(flow, selectedFlowNode) || nodes[0];
    const visibleNodes = nodes.filter(node => flowLayer === 'all' || node.layer === flowLayer);
    const visibleIds = new Set(visibleNodes.map(node => node.id));
    const edges = (flow.edges || []).filter(edge => visibleIds.has(edge.source) && visibleIds.has(edge.target));
    const wires = edges.map(edge => {
      const source = flowNodeById(flow, edge.source);
      const target = flowNodeById(flow, edge.target);
      if(!source || !target){ return ''; }
      return `<line class="flow-edge ${esc(edge.mode || '')}" x1="${esc(source.x)}" y1="${esc(source.y)}" x2="${esc(target.x)}" y2="${esc(target.y)}"><title>${esc(edge.label || edge.relation)}</title></line>`;
    }).join('');
    const nodeHtml = nodes.map(node => {
      const hidden = flowLayer !== 'all' && node.layer !== flowLayer;
      const selectedClass = node.id === selectedFlowNode ? 'selected' : '';
      const dimmed = hidden ? 'dimmed' : '';
      return `<button type="button" class="flow-node ${selectedClass} ${dimmed}" data-flow-node="${esc(node.id)}" style="left:${esc(node.x)}%;top:${esc(node.y)}%" aria-label="${esc(node.label)}">
        <strong>${esc(node.label)}</strong>
        <span>${esc(node.type)} · ${esc(node.layer)}</span>
        <em class="flow-state state-${esc(node.state)}">${esc(stateLabel(node.state))}</em>
      </button>`;
    }).join('');
    const layers = ['all'].concat(flow.layers || []);
    const sourceLinks = (selected.source_files || []).slice(0, 6).map(path => `<a href="${esc(shortPath(path))}" target="_blank" rel="noreferrer">${esc(shortPath(path))}</a>`).join('');
    const metrics = Object.entries(selected.metrics || {}).slice(0, 4).map(([key, value]) => `<span class="tag">${esc(key)}=${esc(value)}</span>`).join('');
    const modeButtons = (flow.modes || ['design','simulation','operations']).map(mode => `<button type="button" class="${mode===flowMode?'active':''}" data-flow-mode="${esc(mode)}">${esc(mode)}</button>`).join('');
    const layerOptions = layers.map(layer => `<option value="${esc(layer)}" ${layer===flowLayer?'selected':''}>${esc(layer)}</option>`).join('');
    const caption = flowMode === 'simulation'
      ? 'Simulacion offline: la informacion se mueve por el flujo sin ejecutar herramientas reales.'
      : flowMode === 'operations'
        ? 'Operations View: revisa bindings, approvals, fallbacks, costos y observabilidad.'
        : 'Design View: entiende la arquitectura aprobada antes de construir.';
    return `<section class="flow-shell ${flowPlaying ? '' : 'paused'}" aria-label="Mapa vivo del agente">
      <div class="flow-toolbar">
        <h3>${esc(flow.title || 'Mapa vivo del agente')}</h3>
        <div class="flow-tools">
          ${modeButtons}
          <button type="button" data-flow-play>${flowPlaying ? 'Pausar' : 'Reproducir'}</button>
          <select data-flow-layer aria-label="Filtrar capa">${layerOptions}</select>
        </div>
      </div>
      <div class="flow-stage">
        <div class="flow-canvas">
          <svg class="flow-wires" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${wires}</svg>
          ${nodeHtml}
          <div class="flow-caption">${esc(caption)}</div>
        </div>
        <aside class="flow-panel">
          <h4>${esc(selected.label)}</h4>
          <p>${esc(selected.description || 'Nodo derivado de artefactos ACP.')}</p>
          <div class="decision-meta">${metrics}</div>
          <p><strong>Estado:</strong> ${esc(stateLabel(selected.state))}</p>
          <p><strong>Fuentes ACP</strong></p>
          ${sourceLinks || '<p>No hay fuentes declaradas para este nodo.</p>'}
        </aside>
      </div>
    </section>`;
  }
  function bindFlowMap(){
    chapter.querySelectorAll('[data-flow-mode]').forEach(btn => btn.addEventListener('click', () => { flowMode = btn.dataset.flowMode || 'design'; render(); }));
    const play = chapter.querySelector('[data-flow-play]');
    if(play){ play.addEventListener('click', () => { flowPlaying = !flowPlaying; render(); }); }
    const layer = chapter.querySelector('[data-flow-layer]');
    if(layer){ layer.addEventListener('change', () => { flowLayer = layer.value || 'all'; render(); }); }
    chapter.querySelectorAll('[data-flow-node]').forEach(btn => btn.addEventListener('click', () => { selectedFlowNode = btn.dataset.flowNode || ''; render(); }));
  }
  function renderNav(){
    nav.innerHTML = data.storyline.map((item, index) => `<button type="button" class="${index===current?'active':''}" data-index="${index}">${esc(index+1)}. ${esc(item.title)}</button>`).join('');
    nav.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => { current = Number(btn.dataset.index || 0); render(); }));
  }
  function render(){
    const item = data.storyline[current] || data.storyline[0];
    if(!item){ chapter.innerHTML = '<p>No hay capitulos disponibles en este ACP.</p>'; return; }
    const files = (data.items || []).filter(entry => (item.related_files || []).includes(entry.id));
    chapter.innerHTML = `
      <div class="eyebrow">${esc(item.id)}</div>
      <h2>${esc(item.title)}</h2>
      <p>${esc(item.narrative)}</p>
      <p><strong>Por que importa:</strong> ${esc(item.why_it_matters)}</p>
      <div class="pills">${(item.key_takeaways || []).map(t => `<span class="pill">${esc(t)}</span>`).join('')}</div>
      ${item.id === 'flow-map' ? renderFlowMap() : ''}
      ${renderSegmentedFiles(files)}
      <div class="controls"><button class="secondary" type="button" data-prev>Anterior</button><button type="button" data-next>Siguiente</button></div>
    `;
    chapter.querySelector('[data-prev]').addEventListener('click', () => { current = Math.max(0, current - 1); render(); });
    chapter.querySelector('[data-next]').addEventListener('click', () => { current = Math.min(data.storyline.length - 1, current + 1); render(); });
    bindFlowMap();
    renderNav();
  }
  if(search){
    search.addEventListener('input', () => {
      const query = search.value.toLowerCase();
      nav.querySelectorAll('button').forEach(btn => { btn.hidden = query && !btn.textContent.toLowerCase().includes(query); });
    });
  }
  renderCockpit();
  render();
})();
""".strip()


def _build_acp_viewer_html(manifest: dict[str, Any]) -> str:
    manifest_json = serialize_json_document(manifest).replace("</", "<\\/")
    chapters_count = len(manifest.get("storyline", []))
    items_count = len(manifest.get("items", []))
    title = str(manifest.get("title") or "Agent Construction Package")
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="assets/acp-viewer.css">
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">Lean Agent Builder</div>
      <h1 class="title">{escape(title)}</h1>
      <p class="meta">Agent Construction Package · {chapters_count} capitulos · {items_count} archivos · readiness {escape(str(manifest.get('validation_status') or 'needs_review'))}</p>
      <input class="search" data-search type="search" placeholder="Buscar capitulo">
      <nav class="nav" data-nav aria-label="Mapa ACP"></nav>
    </aside>
    <main class="main">
      <section class="hero">
        <div class="eyebrow">ACP Viewer</div>
        <h1>De Blueprint aprobado a paquete implementable</h1>
        <p>Este viewer recorre el ACP como historia de implementacion: entrada aprobada, validacion, decisiones, especificacion, costos, readiness, package e implementacion.</p>
        <div data-cockpit></div>
      </section>
      <section class="chapter" data-chapter aria-live="polite"></section>
    </main>
  </div>
  <script>window.ACP_NAVIGATION_MANIFEST = {manifest_json};</script>
  <script src="assets/acp-viewer.js"></script>
</body>
</html>"""


def _build_acp_viewer_files(
    snapshot: SessionSnapshot,
    preview: ACPPreview,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
) -> list[ACPFileEntry]:
    manifest = _build_acp_navigation_manifest(snapshot, preview, response_records)
    return [
        build_acp_file_entry(
            path="ACP/navigation-manifest.v1.json",
            domain="manifest",
            title="ACP navigation manifest",
            format="json",
            source_sections=["acp_preview.files", "construction_readiness"],
            content_text=serialize_json_document(manifest),
        ),
        build_acp_file_entry(
            path="ACP/assets/acp-viewer.css",
            domain="manifest",
            title="ACP viewer CSS",
            format="css",
            source_sections=["acp_preview.files"],
            content_text=_acp_viewer_css(),
        ),
        build_acp_file_entry(
            path="ACP/assets/acp-viewer.js",
            domain="manifest",
            title="ACP viewer JS",
            format="javascript",
            source_sections=["acp_preview.files"],
            content_text=_acp_viewer_js(),
        ),
        build_acp_file_entry(
            path="ACP/index.html",
            domain="manifest",
            title="ACP viewer",
            format="html",
            source_sections=["acp_preview.files", "construction_readiness"],
            content_text=_build_acp_viewer_html(manifest),
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


def _build_evaluation_files(snapshot: SessionSnapshot, context: ProjectGenerationContext | None = None) -> list[ACPFileEntry]:
    dataset = snapshot.evaluation_dataset
    rubric = snapshot.evaluation_rubric
    if dataset is None or rubric is None:
        delegated_evaluation_payload = {
            **_context_trace_payload(context),
            "schema_version": "acp-evaluation-delegation.v1",
            "status": "delegated_to_implementation",
            "reason": "El Blueprint aprobado no incluye dataset/rubrica de evaluacion cerrados.",
            "policy": "No reabrir fases estables; construir pruebas durante implementacion usando el ACP como evidencia.",
            "required_traceability": [
                "pregunta_o_gap_origen",
                "respuesta_o_decision",
                "artefactos_afectados",
                "criterio_de_aceptacion",
            ],
            "starter_cases": [],
        }
        delegated_rubric_payload = {
            **_context_trace_payload(context),
            "schema_version": "acp-rubric-delegation.v1",
            "status": "delegated_to_implementation",
            "dimensions": [
                {
                    "key": "business_alignment",
                    "label": "Alineacion con objetivo de negocio",
                    "minimum_score": "needs_definition",
                },
                {
                    "key": "safety_and_governance",
                    "label": "Seguridad, permisos y gobernanza",
                    "minimum_score": "needs_definition",
                },
                {
                    "key": "runtime_quality",
                    "label": "Calidad operativa del agente",
                    "minimum_score": "needs_definition",
                },
            ],
        }
        test_cases_text = "\n".join(
            [
                "Feature: ACP implementation evaluation",
                "",
                "  # Delegado a implementacion: completar escenarios concretos antes del primer release.",
                "  Scenario: Definir caso de evaluacion desde una decision delegada",
                "    Given el builder agent revisa ACP/construction-readiness/open-questions.yaml",
                "    When una decision impacta un componente, herramienta, memoria o integracion",
                "    Then debe crear un caso de prueba trazable antes de implementar el cambio",
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
                warnings=["Dataset de evaluacion delegado a implementacion; no bloquea el paquete ACP."],
                content_text=serialize_json_document(delegated_evaluation_payload),
            ),
            build_acp_file_entry(
                path="ACP/evaluation/rubrics.yaml",
                domain="evaluation",
                title="Rubrics",
                format="yaml",
                source_sections=["evaluation_rubric"],
                warnings=["Rubrica base delegada a implementacion; usar metodologia ACP para cerrarla."],
                content_text=serialize_yaml_document(delegated_rubric_payload),
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
                warnings=["Casos E2E delegados a implementacion; construirlos antes del primer release."],
                content_text=serialize_markdown_document(test_cases_text),
            ),
        ]

    benchmarks_payload = {
        **_context_trace_payload(context),
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
            content_text=serialize_json_document({**_context_trace_payload(context), "dataset": dataset.model_dump(mode="json")}),
        ),
        build_acp_file_entry(
            path="ACP/evaluation/rubrics.yaml",
            domain="evaluation",
            title="Rubrics",
            format="yaml",
            source_sections=["evaluation_rubric"],
            content_text=serialize_yaml_document({**_context_trace_payload(context), "rubric": rubric.model_dump(mode="json")}),
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

    env_lines = [
        "OPENAI_API_KEY=",
        "DATABASE_URL=",
        "APP_ENV=development",
        f"# deployment_target={target_pairs.get('target', '')}",
        f"# secrets_source={network_pairs.get('secrets', '')}",
    ]
    environment_refs = ["OPENAI_API_KEY", "DATABASE_URL"]
    blueprint = snapshot.blueprint
    projected_tools = project_blueprint_tools_for_construction(snapshot) if blueprint else []
    has_whatsapp = bool(projected_tools and any(_is_whatsapp_cloud_tool(tool) for tool in projected_tools))
    google_keys = {
        _google_workspace_connector_key(tool)
        for tool in projected_tools
        if _is_google_workspace_tool(tool)
    } if projected_tools else set()
    odoo_keys = _odoo_connector_keys_for_snapshot(snapshot) if blueprint else set()
    if has_whatsapp:
        env_lines.extend(
            [
                "",
                "# WhatsApp Business Cloud API (referencias, no secretos planos)",
                "WHATSAPP_GRAPH_API_VERSION=",
                "WHATSAPP_GRAPH_API_BASE_URL=",
                "WHATSAPP_BUSINESS_ACCOUNT_ID=",
                "WHATSAPP_PHONE_NUMBER_ID=",
                "WHATSAPP_SANDBOX_WEBHOOK_CALLBACK_URL=",
                "WHATSAPP_PRODUCTION_WEBHOOK_CALLBACK_URL=",
                "WHATSAPP_ACCESS_TOKEN_REF=",
                "WHATSAPP_WEBHOOK_VERIFY_TOKEN_REF=",
                "WHATSAPP_APP_SECRET_REF=",
            ]
        )
        environment_refs.extend(
            [
                "WHATSAPP_GRAPH_API_VERSION",
                "WHATSAPP_GRAPH_API_BASE_URL",
                "WHATSAPP_BUSINESS_ACCOUNT_ID",
                "WHATSAPP_PHONE_NUMBER_ID",
                "WHATSAPP_SANDBOX_WEBHOOK_CALLBACK_URL",
                "WHATSAPP_PRODUCTION_WEBHOOK_CALLBACK_URL",
            ]
        )
    if google_keys:
        env_lines.extend(
            [
                "",
                "# Google Workspace public APIs (referencias, no secretos planos)",
                "GOOGLE_ALLOWED_SCOPES=",
                "GOOGLE_SANDBOX_OAUTH_REDIRECT_URI=",
                "GOOGLE_PRODUCTION_OAUTH_REDIRECT_URI=",
                "GOOGLE_OAUTH_CLIENT_ID_REF=",
                "GOOGLE_OAUTH_CLIENT_SECRET_REF=",
                "GOOGLE_SANDBOX_REFRESH_TOKEN_REF=",
                "GOOGLE_PRODUCTION_REFRESH_TOKEN_REF=",
            ]
        )
        environment_refs.extend(
            [
                "GOOGLE_ALLOWED_SCOPES",
                "GOOGLE_SANDBOX_OAUTH_REDIRECT_URI",
                "GOOGLE_PRODUCTION_OAUTH_REDIRECT_URI",
            ]
        )
        if "google_sheets_read_table" in google_keys:
            env_lines.append("GOOGLE_SHEETS_SPREADSHEET_ID=")
            environment_refs.append("GOOGLE_SHEETS_SPREADSHEET_ID")
        if {"google_calendar_availability_reader", "google_calendar_event_creator"} & google_keys:
            env_lines.append("GOOGLE_CALENDAR_DEFAULT_ID=")
            environment_refs.append("GOOGLE_CALENDAR_DEFAULT_ID")
        if {"gmail_draft_creator", "gmail_send_message"} & google_keys:
            env_lines.append("GMAIL_SENDER_ACCOUNT=")
            environment_refs.append("GMAIL_SENDER_ACCOUNT")
    if odoo_keys:
        env_lines.extend(
            [
                "",
                "# Odoo public/external APIs (referencias, no secretos planos)",
                "ODOO_API_MODE=",
                "ODOO_BASE_URL=",
                "ODOO_DATABASE=",
                "ODOO_ALLOWED_MODELS=",
                "ODOO_ALLOWED_WRITE_ACTIONS=",
                "ODOO_USERNAME_REF=",
                "ODOO_PASSWORD_REF=",
                "ODOO_API_KEY_REF=",
            ]
        )
        environment_refs.extend(
            [
                "ODOO_API_MODE",
                "ODOO_BASE_URL",
                "ODOO_DATABASE",
                "ODOO_ALLOWED_MODELS",
                "ODOO_ALLOWED_WRITE_ACTIONS",
            ]
        )
    env_lines.append("")
    env_template = "\n".join(env_lines)
    agent_service: dict[str, Any] = {
        "environment": environment_refs,
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


def _governance_tool_items(snapshot: SessionSnapshot) -> list[dict[str, Any]]:
    blueprint = snapshot.blueprint
    if blueprint is None:
        return []
    items: list[dict[str, Any]] = []
    for index, tool in enumerate(project_blueprint_tools_for_construction(snapshot), start=1):
        slug = _tool_connector_slug(tool, index)
        items.append(
            {
                "tool_name": getattr(tool, "name", "") or f"tool_{index}",
                "connector_key": slug,
                "contract_ref": _tool_contract_ref(tool, index),
                "connector_profile_ref": f"ACP/tools/connectors/{slug}.yaml",
                "sandbox_binding_ref": f"ACP/tools/bindings/{slug}.sandbox.yaml",
                "production_binding_ref": f"ACP/tools/bindings/{slug}.production.yaml",
                "smoke_test_ref": f"ACP/tools/tests/{slug}-smoke-test.yaml",
                "risk_level": getattr(tool, "risk_level", "") or "needs_review",
                "permission_mode": _tool_permission_mode(tool),
                "requires_approval": bool(getattr(tool, "requires_approval", False)),
                "side_effects": bool(getattr(tool, "has_side_effects", False)),
                "failure_mode": getattr(tool, "failure_mode", "") or "",
            }
        )
    return items


def _build_agent_governance_console_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    tools = _governance_tool_items(snapshot)
    source_files = [
        "ACP/governance/control-plane.yaml",
        "ACP/governance/tool-governance-policy.yaml",
        "ACP/governance/decision-policy.yaml",
        "ACP/governance/approval-matrix.yaml",
        "ACP/tools/connectors/catalog.yaml",
        "ACP/costs/operational-cost-estimate.json",
        "ACP/observability/event-model.yaml",
    ]
    console_manifest = {
        "schema_version": "agent-governance-console.v1",
        "console_type": "web_control_plane_definition",
        "purpose": "Interfaz web para configurar, operar y auditar el agente construido desde el ACP.",
        "custom_client_tools_supported": True,
        "known_connectors_are_examples_not_limits": True,
        "source_files": source_files,
        "modules": [
            {
                "module_key": "overview",
                "label": "Estado del agente",
                "responsibility": "Mostrar estado runtime, readiness, version activa y alertas principales.",
                "source_refs": ["ACP/runtime/config.yaml", "ACP/construction-readiness/overview.yaml"],
            },
            {
                "module_key": "tool_contracts",
                "label": "Contratos de tools",
                "responsibility": "Configurar conectores, bindings, secretos referenciados y smoke tests por herramienta.",
                "source_refs": ["ACP/tools/connectors/catalog.yaml", "ACP/tools/bindings/"],
            },
            {
                "module_key": "decisions",
                "label": "Decisiones y aprobaciones",
                "responsibility": "Resolver preguntas delegadas, aprobaciones humanas y overrides por entorno.",
                "source_refs": ["ACP/governance/decision-policy.yaml", "ACP/governance/approval-matrix.yaml"],
            },
            {
                "module_key": "finops",
                "label": "FinOps",
                "responsibility": "Definir presupuestos, umbrales, limites de consumo y reglas de degradacion.",
                "source_refs": ["ACP/finops/budget-policy.yaml", "ACP/costs/operational-cost-estimate.json"],
            },
            {
                "module_key": "fallbacks",
                "label": "Fallbacks e incidentes",
                "responsibility": "Pausar herramientas, activar rutas alternativas, escalar a humano y cerrar incidentes.",
                "source_refs": ["ACP/governance/control-plane.yaml", "ACP/ops/runbooks/fallback-and-incident-response.md"],
            },
            {
                "module_key": "observability",
                "label": "Observabilidad",
                "responsibility": "Monitorear eventos, metricas, trazas, alertas y auditoria operacional.",
                "source_refs": ["ACP/observability/event-model.yaml", "ACP/observability/metrics.yaml", "ACP/observability/alerts.yaml"],
            },
        ],
        "minimum_ui_states": ["draft", "sandbox_ready", "production_ready", "live", "paused", "degraded", "incident"],
    }
    ui_map = {
        "schema_version": "agent-governance-console-ui-map.v1",
        "navigation": [
            {"route": "/agent", "module_key": "overview", "primary_action": "review_current_state"},
            {"route": "/agent/tools", "module_key": "tool_contracts", "primary_action": "configure_tool_binding"},
            {"route": "/agent/decisions", "module_key": "decisions", "primary_action": "resolve_required_decision"},
            {"route": "/agent/finops", "module_key": "finops", "primary_action": "set_budget_guardrails"},
            {"route": "/agent/fallbacks", "module_key": "fallbacks", "primary_action": "manage_fallback_route"},
            {"route": "/agent/observability", "module_key": "observability", "primary_action": "inspect_events"},
        ],
        "tool_configuration_flow": [
            "select_tool",
            "choose_binding_type",
            "map_client_contract_fields",
            "attach_secret_references",
            "run_smoke_test",
            "request_approval_if_needed",
            "activate_environment_binding",
        ],
        "must_surface_when_present": [
            "blocking_construction_gaps",
            "delegated_implementation_decisions",
            "missing_tool_bindings",
            "plain_secret_values",
            "production_write_tool_without_owner",
            "budget_threshold_breach",
            "fallback_route_missing",
        ],
    }
    role_permissions = {
        "schema_version": "agent-governance-console-roles.v1",
        "roles": {
            "owner": ["view_all", "approve_release", "approve_side_effects", "manage_budget", "pause_agent"],
            "admin": ["view_all", "configure_tools", "manage_fallbacks", "run_smoke_tests"],
            "implementer": ["view_specs", "configure_sandbox_tools", "run_smoke_tests", "propose_production_binding"],
            "operator": ["view_runtime", "pause_tool", "open_incident", "activate_fallback"],
            "auditor": ["view_audit", "export_logs", "view_decisions"],
        },
        "rules": [
            "Production bindings with write or side effects require owner or admin approval.",
            "Implementers can prepare contracts but cannot silently bypass approval gates.",
            "Auditors can inspect decisions and logs without changing runtime state.",
        ],
    }
    control_plane = {
        "schema_version": "agent-control-plane.v1",
        "states": ["draft", "sandbox_ready", "production_ready", "live", "paused", "degraded", "incident", "retired"],
        "actions": [
            "activate_sandbox",
            "promote_to_production",
            "pause_agent",
            "resume_agent",
            "disable_tool",
            "enable_tool",
            "activate_fallback",
            "request_human_approval",
            "open_incident",
            "close_incident",
            "export_audit_report",
        ],
        "hard_rules": [
            "no_plain_secrets",
            "no_production_write_without_named_owner",
            "no_unmapped_custom_client_tool",
            "no_silent_resolution_of_delegated_decisions",
            "fail_closed_when_tool_binding_missing",
        ],
        "tool_refs": tools,
    }
    tool_policy = {
        "schema_version": "tool-governance-policy.v1",
        "custom_client_tools_supported": True,
        "scope": "all_blueprint_tools",
        "default_policy": {
            "read_tools": "allowed_after_contract_and_binding_validation",
            "write_tools": "approval_required_before_production_activation",
            "unknown_or_custom_tools": "allowed_only_with_connector_profile_binding_and_smoke_test",
        },
        "tools": tools,
    }
    decision_policy = {
        "schema_version": "agent-decision-policy.v1",
        "policy": "DO_NOT_ASSUME_SILENTLY",
        "decision_sources": [
            "ACP/construction-readiness/open-questions.yaml",
            "ACP/construction-readiness/deferred-decisions.yaml",
            "ACP/governance/journey-decisions.json",
        ],
        "must_prompt_when": [
            "decision_is_blocking",
            "decision_affects_runtime_or_deployment",
            "decision_affects_tool_binding",
            "decision_affects_budget_or_sla",
            "decision_affects_side_effect_or_fallback",
        ],
    }
    approval_matrix = {
        "schema_version": "agent-approval-matrix.v1",
        "approval_triggers": [
            {"trigger": "production_release", "required_role": "owner"},
            {"trigger": "write_tool_activation", "required_role": "owner_or_admin"},
            {"trigger": "budget_limit_change", "required_role": "owner"},
            {"trigger": "fallback_policy_change", "required_role": "admin"},
            {"trigger": "custom_client_tool_binding", "required_role": "admin"},
        ],
        "tool_overrides": [
            {
                "tool_name": item["tool_name"],
                "requires_approval": item["requires_approval"] or item["side_effects"],
                "reason": "Blueprint approval policy or side effects.",
            }
            for item in tools
        ],
    }
    budget_policy = {
        "schema_version": "agent-finops-budget-policy.v1",
        "source_estimate": "ACP/costs/operational-cost-estimate.json",
        "budget_controls": {
            "monthly_budget_limit": "needs_review",
            "per_run_budget_limit": "needs_review",
            "alert_thresholds_percent": [50, 80, 95],
            "degradation_strategy": "switch_model_reduce_context_or_require_human_approval",
        },
        "must_track": ["llm_tokens", "tool_calls", "external_api_costs", "storage_costs", "human_review_time"],
    }
    event_model = {
        "schema_version": "agent-observability-event-model.v1",
        "required_events": [
            "agent.started",
            "agent.completed",
            "agent.failed",
            "decision.requested",
            "decision.resolved",
            "tool.binding.changed",
            "tool.call.started",
            "tool.call.completed",
            "tool.call.failed",
            "approval.requested",
            "approval.granted",
            "approval.denied",
            "fallback.activated",
            "budget.threshold_reached",
            "incident.opened",
            "incident.closed",
        ],
        "tool_event_namespace": [{"tool_name": item["tool_name"], "prefix": f"tool.{item['connector_key']}"} for item in tools],
        "privacy_policy": "log_references_and_metadata_first; avoid_payload_logging_unless_explicitly_allowed",
    }
    runbook = "\n".join(
        [
            "# Fallback and incident response",
            "",
            "## Activation criteria",
            "- Tool binding missing or failing smoke tests.",
            "- Budget threshold breached.",
            "- Approval gate unavailable for a side-effect action.",
            "- External provider incident or unexpected response schema.",
            "",
            "## Operator flow",
            "1. Open the incident in the governance console.",
            "2. Disable only the affected tool or route when possible.",
            "3. Activate the configured fallback or human handoff.",
            "4. Capture decision, owner, timestamp and evidence.",
            "5. Run the smoke test again before resuming production traffic.",
            "",
            "## Non-negotiables",
            "- Do not paste plaintext secrets into the console or ACP.",
            "- Do not activate production write actions without the required approval.",
            "- Do not treat known connector examples as the full integration catalog.",
        ]
    )
    return [
        build_acp_file_entry(
            path="ACP/ops/agent-governance-console/console-manifest.yaml",
            domain="governance",
            title="Agent governance console manifest",
            format="yaml",
            source_sections=["blueprint", "construction_readiness", "tool_contracts", "runtime", "observability"],
            content_text=serialize_yaml_document(console_manifest),
        ),
        build_acp_file_entry(
            path="ACP/ops/agent-governance-console/ui-map.json",
            domain="governance",
            title="Agent governance console UI map",
            format="json",
            source_sections=["tool_contracts", "runtime", "governance"],
            content_text=serialize_json_document(ui_map),
        ),
        build_acp_file_entry(
            path="ACP/ops/agent-governance-console/role-permissions.yaml",
            domain="governance",
            title="Agent governance console role permissions",
            format="yaml",
            source_sections=["approvals", "risk_summary", "tool_contracts"],
            content_text=serialize_yaml_document(role_permissions),
        ),
        build_acp_file_entry(
            path="ACP/governance/control-plane.yaml",
            domain="governance",
            title="Agent control plane",
            format="yaml",
            source_sections=["runtime", "tool_contracts", "governance"],
            content_text=serialize_yaml_document(control_plane),
        ),
        build_acp_file_entry(
            path="ACP/governance/tool-governance-policy.yaml",
            domain="governance",
            title="Tool governance policy",
            format="yaml",
            source_sections=["blueprint.tools", "approvals", "risk_summary"],
            content_text=serialize_yaml_document(tool_policy),
        ),
        build_acp_file_entry(
            path="ACP/governance/decision-policy.yaml",
            domain="governance",
            title="Agent decision policy",
            format="yaml",
            source_sections=["construction_readiness", "blueprint_consistency"],
            content_text=serialize_yaml_document(decision_policy),
        ),
        build_acp_file_entry(
            path="ACP/governance/approval-matrix.yaml",
            domain="governance",
            title="Agent approval matrix",
            format="yaml",
            source_sections=["approvals", "blueprint.tools"],
            content_text=serialize_yaml_document(approval_matrix),
        ),
        build_acp_file_entry(
            path="ACP/finops/budget-policy.yaml",
            domain="costs",
            title="Agent FinOps budget policy",
            format="yaml",
            source_sections=["estimation_report", "operational_costs", "runtime"],
            content_text=serialize_yaml_document(budget_policy),
        ),
        build_acp_file_entry(
            path="ACP/observability/event-model.yaml",
            domain="observability",
            title="Agent observability event model",
            format="yaml",
            source_sections=["observability", "tool_contracts", "runtime"],
            content_text=serialize_yaml_document(event_model),
        ),
        build_acp_file_entry(
            path="ACP/ops/runbooks/fallback-and-incident-response.md",
            domain="governance",
            title="Fallback and incident response runbook",
            format="markdown",
            source_sections=["governance", "observability", "tool_contracts"],
            content_text=serialize_markdown_document(runbook),
        ),
    ]


def _build_governance_files(snapshot: SessionSnapshot) -> list[ACPFileEntry]:
    report = ensure_blueprint_consistency_report(snapshot)
    process_debt = [
        issue.model_dump(mode="json")
        for issue in report.issues
        if is_blueprint_handoff_process_debt_issue(issue.issue_key)
    ]
    real_debt = [
        issue.model_dump(mode="json")
        for issue in report.issues
        if not is_blueprint_handoff_process_debt_issue(issue.issue_key)
        and issue.severity in {"blocking", "warning"}
    ]
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
    handoff_closure_payload = {
        "schema_version": "blueprint-acp-handoff-closure.v1",
        "policy": {
            "blueprint_approval_closes_operational_cycle": True,
            "process_debt_does_not_travel_as_acp_debt": True,
            "real_debt_keeps_traceability": True,
            "critical_integrity_issues_must_block_or_be_resolved": True,
            "reconciliation_scope": "granular_artifacts_only",
        },
        "classification": {
            "process_debt_closed_or_archived": process_debt,
            "real_debt_preserved_for_acp": real_debt,
        },
        "notes": [
            "Flags stale, warnings de sincronizacion y drift transitorio del Blueprint son evidencia operativa, no deuda de implementacion.",
            "Preguntas, restricciones o decisiones delegadas deben resolverse en el momento de implementar el artefacto afectado.",
            "Si aparece una contradiccion real contra el Blueprint aprobado, usar reconciliacion granular; no reiniciar fases completas.",
        ],
    }
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
        build_acp_file_entry(
            path="ACP/governance/blueprint-handoff-closure.yaml",
            domain="governance",
            title="Blueprint ACP handoff closure",
            format="yaml",
            source_sections=["blueprint_consistency", "blueprint_handoff"],
            content_text=serialize_yaml_document(handoff_closure_payload),
            warnings=[
                "Contiene clasificacion del handoff; no usar process_debt_closed_or_archived como backlog de implementacion."
            ]
            if process_debt
            else [],
        ),
    ]


def generate_acp_files(
    snapshot: SessionSnapshot,
    continuity_answers: dict[str, str] | None = None,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
    extra_readiness_gaps: list[ConstructionGapEntry] | None = None,
    context: ProjectGenerationContext | None = None,
    prompt_synthesizer: ACPPromptSectionSynthesizer | None = None,
) -> list[ACPFileEntry]:
    acp_context = context or _build_acp_generation_context(snapshot, response_records, extra_readiness_gaps)
    files: list[ACPFileEntry] = []
    files.append(_build_manifest_file(snapshot, acp_context))
    files.append(_build_readme_file(snapshot, acp_context))
    files.extend(_build_deliverable_catalog_files(snapshot))
    files.extend(_build_launcher_files(snapshot))
    files.extend(_build_adapter_files(snapshot))
    files.extend(_build_business_files(snapshot, acp_context))
    files.extend(_build_architecture_files(snapshot, acp_context))
    files.extend(_build_cognition_files(snapshot))
    files.extend(_build_memory_files(snapshot, acp_context))
    files.extend(_build_knowledge_files(snapshot, continuity_answers, acp_context))
    files.extend(_build_tools_files(snapshot, acp_context))
    files.extend(_build_tool_connector_files(snapshot))
    files.extend(_build_whatsapp_connector_files(snapshot))
    files.extend(_build_google_workspace_connector_files(snapshot))
    files.extend(_build_odoo_connector_files(snapshot))
    files.extend(_build_objective_files(snapshot, response_records))
    files.extend(_build_workflow_files(snapshot, acp_context))
    files.extend(_build_prompt_files(snapshot, acp_context, prompt_synthesizer))
    files.extend(_build_runtime_files(snapshot, continuity_answers))
    files.extend(_build_evaluation_files(snapshot, acp_context))
    files.extend(_build_deployment_files(snapshot, continuity_answers))
    files.extend(_build_operational_cost_files(snapshot))
    files.extend(_build_observability_files(snapshot))
    files.extend(_build_governance_files(snapshot))
    files.extend(_build_agent_governance_console_files(snapshot))
    files.extend(_build_agent_flow_map_files(snapshot))
    files.extend(_build_estimation_files(snapshot))
    base_files = sorted(files, key=lambda item: item.path)
    base_preview = build_acp_preview(snapshot, base_files)
    base_preview = append_construction_readiness_gaps(base_preview, extra_readiness_gaps)
    base_preview = _apply_question_readiness_overlay(base_preview, response_records)
    continuity_files = _build_construction_readiness_files(
        snapshot,
        base_preview,
        continuity_answers,
        response_records,
        acp_context,
    )
    continuity_files.extend(_build_continuity_prompt_files(base_preview))
    continuity_files.extend(_build_implementation_guidance_files(base_preview))
    acp_without_diagrams = sorted(base_files + continuity_files, key=lambda item: item.path)
    visualization_files = build_acp_visualization_files(snapshot, acp_without_diagrams)
    acp_without_conformance = sorted(acp_without_diagrams + visualization_files, key=lambda item: item.path)
    conformance_preview = build_acp_preview(snapshot, acp_without_conformance)
    conformance_preview = append_construction_readiness_gaps(conformance_preview, extra_readiness_gaps)
    conformance_preview = _apply_question_readiness_overlay(conformance_preview, response_records)
    conformance_files = build_acp_conformance_files(
        conformance_preview,
        acp_without_conformance,
        profile="acp-full",
    )
    acp_without_viewer = sorted(acp_without_conformance + conformance_files, key=lambda item: item.path)
    viewer_preview = build_acp_preview(snapshot, acp_without_viewer)
    viewer_preview = append_construction_readiness_gaps(viewer_preview, extra_readiness_gaps)
    viewer_preview = _apply_question_readiness_overlay(viewer_preview, response_records)
    viewer_files = _build_acp_viewer_files(snapshot, viewer_preview, response_records)
    return sorted(acp_without_viewer + viewer_files, key=lambda item: item.path)


def generate_acp_preview(
    snapshot: SessionSnapshot,
    continuity_answers: dict[str, str] | None = None,
    response_records: list[ConstructionQuestionResponseRecord] | None = None,
    extra_readiness_gaps: list[ConstructionGapEntry] | None = None,
    prompt_synthesizer: ACPPromptSectionSynthesizer | None = None,
) -> ACPPreview:
    acp_context = _build_acp_generation_context(snapshot, response_records, extra_readiness_gaps)
    preview = build_acp_preview(
        snapshot,
        generate_acp_files(
            snapshot,
            continuity_answers,
            response_records,
            extra_readiness_gaps,
            acp_context,
            prompt_synthesizer,
        ),
    )
    preview = append_construction_readiness_gaps(preview, extra_readiness_gaps)
    return _apply_question_readiness_overlay(preview, response_records)
