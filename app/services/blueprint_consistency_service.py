from __future__ import annotations

from datetime import datetime
from typing import Any

from app.models import (
    ApprovedStageLineageEntry,
    BlueprintConsistencyIssue,
    BlueprintConsistencyReport,
    DesignRecommendationArtifact,
    JourneyArtifactState,
    JourneyDecisionType,
    JourneyStageArtifactEntry,
    MemoryRecommendationArtifact,
    ReviewState,
    SessionSnapshot,
    SimulationSpecificationArtifact,
)
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionOutput

_APPROVED_STATES = {JourneyArtifactState.approved, JourneyArtifactState.approved_legacy}
_PRIVATE_LINEAGE_FLAGS = (
    "repo://",
    "docs/system-analysis",
    "repo_memory_manifest",
    "reingenieria_core_canonical",
    "system_analysis_operational",
    "taxonomy",
    "global_corpus",
    "platform_memory",
)
_STAGES_IN_ORDER = ("discover", "define", "design", "tools", "memory", "validate")


def _normalized(value: str | None) -> str:
    return (value or "").strip()


def _normalized_list(items: list[str] | None) -> list[str]:
    return [item.strip() for item in (items or []) if isinstance(item, str) and item.strip()]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _artifact_sort_key(artifact: JourneyStageArtifactEntry) -> tuple[int, datetime | str]:
    return artifact.version_number, artifact.updated_at or artifact.created_at


def _latest_approved_artifact(snapshot: SessionSnapshot, stage_key: str) -> JourneyStageArtifactEntry | None:
    approved = [
        item
        for item in snapshot.journey_artifacts
        if item.stage_key == stage_key and item.state in _APPROVED_STATES
    ]
    if not approved:
        return None
    return max(approved, key=_artifact_sort_key)


def _resolve_design_artifact(artifact: JourneyStageArtifactEntry | None) -> DesignRecommendationArtifact | None:
    if artifact is None:
        return None
    schema_version = artifact.schema_version or artifact.proposal_payload.get("schema_version", "")
    if schema_version == "design-recommendation.v1" or "alternatives" in artifact.proposal_payload:
        return DesignRecommendationArtifact.model_validate(artifact.proposal_payload)
    return None


def _resolve_memory_artifact(artifact: JourneyStageArtifactEntry | None) -> MemoryRecommendationArtifact | None:
    if artifact is None:
        return None
    schema_version = artifact.schema_version or artifact.proposal_payload.get("schema_version", "")
    if schema_version == "memory-recommendation.v1" or "proposed_memory_profile" in artifact.proposal_payload:
        return MemoryRecommendationArtifact.model_validate(artifact.proposal_payload)
    return None


def _resolve_validation_artifact(artifact: JourneyStageArtifactEntry | None) -> SimulationSpecificationArtifact | None:
    if artifact is None:
        return None
    schema_version = artifact.schema_version or artifact.proposal_payload.get("schema_version", "")
    if schema_version == "validation-simulation-spec.v1" or "scenarios" in artifact.proposal_payload:
        return SimulationSpecificationArtifact.model_validate(artifact.proposal_payload)
    return None


def _artifact_citations(artifact: JourneyStageArtifactEntry) -> list[str]:
    citations: list[str] = []
    for item in artifact.evidence_manifest:
        label = item.citation_label or item.artifact_ref or item.source_id
        if label:
            citations.append(label)
    return _dedupe(citations)


def _artifact_lineage_refs(artifact: JourneyStageArtifactEntry) -> list[str]:
    refs: list[str] = []
    for item in artifact.evidence_manifest:
        refs.extend(_normalized_list(item.source_lineage))
    return _dedupe(refs)


def _is_exportable_lineage(lineage_ref: str) -> bool:
    normalized = lineage_ref.strip().lower()
    if not normalized:
        return False
    return not any(flag in normalized for flag in _PRIVATE_LINEAGE_FLAGS)


def _current_blueprint_version(snapshot: SessionSnapshot) -> int | None:
    versions = snapshot.blueprint_versions or []
    if not versions:
        return None
    return max(item.version_number for item in versions)


def _issue(
    *,
    issue_key: str,
    severity: str,
    category: str,
    title: str,
    detail: str,
    affected_stage_keys: list[str] | None = None,
    source_refs: list[str] | None = None,
    citations: list[str] | None = None,
) -> BlueprintConsistencyIssue:
    return BlueprintConsistencyIssue(
        issue_key=issue_key,
        severity=severity,  # type: ignore[arg-type]
        category=category,
        title=title,
        detail=detail,
        affected_stage_keys=affected_stage_keys or [],
        source_refs=source_refs or [],
        citations=citations or [],
    )


def _build_stage_lineage(snapshot: SessionSnapshot) -> tuple[list[ApprovedStageLineageEntry], list[str], list[str]]:
    entries: list[ApprovedStageLineageEntry] = []
    exportable_lineage: list[str] = []
    restricted_lineage: list[str] = []
    for stage_key in _STAGES_IN_ORDER:
        artifact = _latest_approved_artifact(snapshot, stage_key)
        if artifact is None:
            continue
        lineage_refs = _artifact_lineage_refs(artifact)
        exportable_lineage.extend(item for item in lineage_refs if _is_exportable_lineage(item))
        restricted_lineage.extend(item for item in lineage_refs if not _is_exportable_lineage(item))
        entries.append(
            ApprovedStageLineageEntry(
                stage_key=artifact.stage_key,
                artifact_id=artifact.id,
                artifact_kind=artifact.artifact_kind,
                source_action=artifact.source_action,
                version_number=artifact.version_number,
                state=str(artifact.state),
                approved_at=artifact.approved_at,
                decision_count=len(artifact.decisions),
                rejection_count=sum(1 for item in artifact.decisions if item.decision_type == JourneyDecisionType.reject),
                citation_labels=_artifact_citations(artifact),
                lineage_refs=lineage_refs,
            )
        )

    blueprint = snapshot.blueprint
    if blueprint is not None and blueprint.knowledge_profile.sources:
        for source in blueprint.knowledge_profile.sources:
            lineage_ref = f"{(source.key or source.title or 'knowledge-source').strip()}::{source.source_version}"
            if _is_exportable_lineage(lineage_ref):
                exportable_lineage.append(lineage_ref)
            else:
                restricted_lineage.append(lineage_ref)

    return entries, _dedupe(exportable_lineage), _dedupe(restricted_lineage)


def _build_decision_history(snapshot: SessionSnapshot) -> list[dict[str, Any]]:
    ordered = sorted(
        snapshot.journey_artifacts,
        key=lambda item: (item.stage_key, item.version_number, item.created_at),
    )
    history: list[dict[str, Any]] = []
    for artifact in ordered:
        history.append(
            {
                "stage_key": artifact.stage_key,
                "artifact_id": artifact.id,
                "artifact_kind": artifact.artifact_kind,
                "version_number": artifact.version_number,
                "state": artifact.state,
                "source_action": artifact.source_action,
                "approved_at": artifact.approved_at,
                "rejected_at": artifact.rejected_at,
                "reviewed_at": artifact.reviewed_at,
                "stale_at": artifact.stale_at,
                "citation_labels": _artifact_citations(artifact),
                "decisions": [
                    {
                        "decision_type": item.decision_type,
                        "previous_state": item.previous_state,
                        "next_state": item.next_state,
                        "note": item.note,
                        "payload": item.payload,
                        "created_at": item.created_at,
                    }
                    for item in artifact.decisions
                ],
            }
        )
    return history


def build_blueprint_consistency_report(snapshot: SessionSnapshot) -> BlueprintConsistencyReport:
    issues: list[BlueprintConsistencyIssue] = []
    uncovered_requirement_keys: list[str] = []
    orphan_design_role_keys: list[str] = []
    orphan_tool_keys: list[str] = []
    orphan_memory_dependency_keys: list[str] = []
    stale_stage_keys: list[str] = []

    stage_lineage, exportable_lineage, restricted_lineage = _build_stage_lineage(snapshot)
    decision_history = _build_decision_history(snapshot)

    blueprint = snapshot.blueprint
    approved_define = _latest_approved_artifact(snapshot, "define")
    approved_design = _latest_approved_artifact(snapshot, "design")
    approved_validate = _latest_approved_artifact(snapshot, "validate")
    approved_memory = _latest_approved_artifact(snapshot, "memory")
    tool_recommendation = snapshot.latest_tool_recommendation

    requirement_priorities: dict[str, str] = {}
    if approved_define is not None:
        definition = RequirementsDefinitionOutput.model_validate(approved_define.proposal_payload)
        for item in definition.functional_requirements:
            requirement_priorities[item.key] = item.priority
        for item in definition.non_functional_requirements:
            requirement_priorities[item.key] = item.priority
        for item in definition.business_rules:
            requirement_priorities[item.key] = item.priority

        design = _resolve_design_artifact(approved_design)
        if design is not None and approved_design is not None:
            coverage_by_requirement = {
                item.requirement_key: item for item in design.requirements_coverage if item.requirement_key
            }
            for requirement_key, priority in requirement_priorities.items():
                coverage = coverage_by_requirement.get(requirement_key)
                if coverage is None:
                    uncovered_requirement_keys.append(requirement_key)
                    issues.append(
                        _issue(
                            issue_key=f"design_requirement_missing:{requirement_key}",
                            severity="blocking" if priority == "high" else "warning",
                            category="requirement_to_design",
                            title="Requirement sin cobertura de diseño",
                            detail=f"El requirement `{requirement_key}` no aparece cubierto en la alternativa aprobada de Design.",
                            affected_stage_keys=["define", "design"],
                            source_refs=["journey.define", "journey.design"],
                            citations=_artifact_citations(approved_design),
                        )
                    )
                    continue
                if coverage.coverage_status == "gap":
                    uncovered_requirement_keys.append(requirement_key)
                    issues.append(
                        _issue(
                            issue_key=f"design_requirement_gap:{requirement_key}",
                            severity="blocking" if priority == "high" else "warning",
                            category="requirement_to_design",
                            title="Requirement con gap en diseño",
                            detail=coverage.rationale
                            or f"El requirement `{requirement_key}` quedo marcado con coverage_status=gap en Design.",
                            affected_stage_keys=["define", "design"],
                            source_refs=coverage.source_refs or ["journey.design"],
                            citations=_artifact_citations(approved_design),
                        )
                    )

            selected_design = design.selected_design
            if blueprint is not None and selected_design is not None:
                expected_architecture = _normalized(
                    selected_design.blueprint_projection.architecture or selected_design.architecture
                )
                expected_reasoning = _normalized(
                    selected_design.blueprint_projection.reasoning_pattern or selected_design.reasoning_pattern
                )
                if expected_architecture and _normalized(blueprint.architecture) != expected_architecture:
                    issues.append(
                        _issue(
                            issue_key="design_blueprint_projection_drift:architecture",
                            severity="blocking",
                            category="design_to_blueprint",
                            title="Arquitectura actual desalineada del Design aprobado",
                            detail=(
                                f"Blueprint tiene `{blueprint.architecture}` pero Design aprobado proyecto "
                                f"`{expected_architecture}`."
                            ),
                            affected_stage_keys=["design"],
                            source_refs=["blueprint.architecture", "journey.design.selected_design"],
                            citations=_artifact_citations(approved_design),
                        )
                    )
                if expected_reasoning and _normalized(blueprint.reasoning_pattern) != expected_reasoning:
                    issues.append(
                        _issue(
                            issue_key="design_blueprint_projection_drift:reasoning",
                            severity="blocking",
                            category="design_to_blueprint",
                            title="Patron de razonamiento desalineado del Design aprobado",
                            detail=(
                                f"Blueprint tiene `{blueprint.reasoning_pattern}` pero Design aprobado proyecto "
                                f"`{expected_reasoning}`."
                            ),
                            affected_stage_keys=["design"],
                            source_refs=["blueprint.reasoning_pattern", "journey.design.selected_design"],
                            citations=_artifact_citations(approved_design),
                        )
                    )
                projected_guardrails = set(_normalized_list(selected_design.blueprint_projection.guardrails))
                current_guardrails = set(_normalized_list(blueprint.guardrails))
                missing_guardrails = sorted(projected_guardrails - current_guardrails)
                if missing_guardrails:
                    issues.append(
                        _issue(
                            issue_key="design_blueprint_projection_drift:guardrails",
                            severity="warning",
                            category="design_to_blueprint",
                            title="Guardrails proyectados por Design faltan en el blueprint actual",
                            detail="Faltan guardrails: " + ", ".join(missing_guardrails),
                            affected_stage_keys=["design"],
                            source_refs=["blueprint.guardrails", "journey.design.selected_design"],
                            citations=_artifact_citations(approved_design),
                        )
                    )

                role_keys = {item.key for item in selected_design.roles if _normalized(item.key)}
                covered_role_keys = {
                    item.role_key
                    for item in (tool_recommendation.design_role_coverage if tool_recommendation is not None else [])
                    if _normalized(item.role_key)
                }
                orphan_design_role_keys.extend(sorted(role_keys - covered_role_keys))

    if tool_recommendation is not None:
        if tool_recommendation.is_stale:
            stale_stage_keys.append("tools")
            issues.append(
                _issue(
                    issue_key="tools_recommendation_stale",
                    severity="blocking",
                    category="design_to_tools",
                    title="La recomendacion de Tools esta stale",
                    detail=" | ".join(_normalized_list(tool_recommendation.stale_reasons))
                    or "La recomendacion aprobada de tools ya no corresponde al blueprint vigente.",
                    affected_stage_keys=["tools"],
                    source_refs=["tool_recommendation"],
                )
            )

        for item in tool_recommendation.requirements_coverage:
            if item.coverage_status != "gap":
                continue
            uncovered_requirement_keys.append(item.requirement_key)
            issues.append(
                _issue(
                    issue_key=f"tools_requirement_gap:{item.requirement_key}",
                    severity="blocking" if item.priority == "high" else "warning",
                    category="design_to_tools",
                    title="Requirement sin cobertura de herramientas",
                    detail=item.rationale or f"El requirement `{item.requirement_key}` quedo sin cobertura en Tools.",
                    affected_stage_keys=["tools"],
                    source_refs=item.source_refs or ["tool_recommendation.requirements_coverage"],
                )
            )

        if blueprint is not None and tool_recommendation.approved_tools_digest is not None:
            current_tool_names = {
                item.name.strip()
                for item in blueprint.tools
                if _normalized(item.name)
            }
            approved_tool_names = {
                item.strip()
                for item in tool_recommendation.approved_tools_digest.selected_blueprint_tool_names
                if item.strip()
            }
            unexpected_tools = sorted(current_tool_names - approved_tool_names)
            missing_tools = sorted(approved_tool_names - current_tool_names)
            orphan_tool_keys.extend(unexpected_tools)
            if unexpected_tools or missing_tools:
                issues.append(
                    _issue(
                        issue_key="approved_tools_digest_drift",
                        severity="blocking",
                        category="tools_to_blueprint",
                        title="El set de tools actual no coincide con la seleccion aprobada",
                        detail=(
                            f"Extras actuales: {', '.join(unexpected_tools) or 'ninguno'} | "
                            f"faltantes aprobados: {', '.join(missing_tools) or 'ninguno'}."
                        ),
                        affected_stage_keys=["tools"],
                        source_refs=["blueprint.tools", "tool_recommendation.approved_tools_digest"],
                    )
                )
        elif blueprint is not None and blueprint.tools:
            issues.append(
                _issue(
                    issue_key="approved_tools_digest_missing",
                    severity="warning",
                    category="tools_to_blueprint",
                    title="Blueprint con tools pero sin digest aprobado",
                    detail="Existe configuracion de tools en el blueprint actual sin approved_tools_digest activo.",
                    affected_stage_keys=["tools"],
                    source_refs=["blueprint.tools", "tool_recommendation"],
                )
            )

    memory_artifact = _resolve_memory_artifact(approved_memory)
    if memory_artifact is not None and approved_memory is not None and blueprint is not None:
        current_memory_profile = blueprint.memory_profile.model_dump(mode="json")
        current_knowledge_profile = blueprint.knowledge_profile.model_dump(mode="json")
        approved_memory_profile = memory_artifact.proposed_memory_profile.model_dump(mode="json")
        approved_knowledge_profile = memory_artifact.proposed_knowledge_profile.model_dump(mode="json")
        if current_memory_profile != approved_memory_profile or current_knowledge_profile != approved_knowledge_profile:
            issues.append(
                _issue(
                    issue_key="memory_blueprint_projection_drift",
                    severity="blocking",
                    category="tools_to_memory",
                    title="La memoria actual no coincide con la propuesta aprobada",
                    detail=(
                        "El blueprint tiene cambios en memory_profile o knowledge_profile posteriores a la aprobacion "
                        "de Memory."
                    ),
                    affected_stage_keys=["memory"],
                    source_refs=["blueprint.memory_profile", "blueprint.knowledge_profile", "journey.memory"],
                    citations=_artifact_citations(approved_memory),
                )
            )

        approved_tool_keys = {
            item
            for item in (
                tool_recommendation.approved_tools_digest.approved_tool_keys
                if tool_recommendation is not None and tool_recommendation.approved_tools_digest is not None
                else []
            )
            if _normalized(item)
        }
        required_memory_deps = {
            item.tool_key
            for item in memory_artifact.tool_dependencies
            if item.required and _normalized(item.tool_key)
        }
        missing_memory_deps = sorted(required_memory_deps - approved_tool_keys)
        orphan_memory_dependency_keys.extend(missing_memory_deps)
        if missing_memory_deps:
            issues.append(
                _issue(
                    issue_key="memory_required_tool_dependency_missing",
                    severity="blocking",
                    category="tools_to_memory",
                    title="Memory depende de tools no aprobadas",
                    detail="Dependencias faltantes: " + ", ".join(missing_memory_deps),
                    affected_stage_keys=["tools", "memory"],
                    source_refs=["journey.memory.tool_dependencies", "tool_recommendation.approved_tools_digest"],
                    citations=_artifact_citations(approved_memory),
                )
            )

        if memory_artifact.is_stale:
            stale_stage_keys.append("memory")
            issues.append(
                _issue(
                    issue_key="memory_recommendation_stale",
                    severity="blocking",
                    category="tools_to_memory",
                    title="La recomendacion de Memory esta stale",
                    detail=" | ".join(_normalized_list(memory_artifact.stale_reasons))
                    or "La propuesta aprobada de memoria ya no corresponde al blueprint vigente.",
                    affected_stage_keys=["memory"],
                    source_refs=["journey.memory"],
                    citations=_artifact_citations(approved_memory),
                )
            )

    validate_artifact = _resolve_validation_artifact(approved_validate)
    if validate_artifact is not None:
        latest_versions = {
            stage_key: next(
                (
                    item.version_number
                    for item in stage_lineage
                    if item.stage_key == stage_key and item.version_number is not None
                ),
                None,
            )
            for stage_key in ("discover", "define", "design", "tools", "memory")
        }
        for stage_key, current_version in latest_versions.items():
            expected_version = validate_artifact.source_stage_versions.get(stage_key)
            if expected_version is None or current_version is None:
                continue
            if expected_version != current_version:
                stale_stage_keys.append("validate")
                issues.append(
                    _issue(
                        issue_key=f"validate_source_stage_drift:{stage_key}",
                        severity="blocking",
                        category="memory_to_validate",
                        title="Validate referencia versiones aprobadas desactualizadas",
                        detail=(
                            f"Validate quedo ligado a {stage_key}=v{expected_version}, pero la version aprobada vigente es "
                            f"v{current_version}."
                        ),
                        affected_stage_keys=[stage_key, "validate"],
                        source_refs=["journey.validate.source_stage_versions"],
                        citations=_artifact_citations(approved_validate),
                    )
                )

        for gap in _normalized_list(validate_artifact.coverage_gaps):
            issues.append(
                _issue(
                    issue_key=f"validate_coverage_gap:{gap}",
                    severity="warning",
                    category="memory_to_validate",
                    title="Validate mantiene coverage gap abierto",
                    detail=gap,
                    affected_stage_keys=["validate"],
                    source_refs=["journey.validate.coverage_gaps"],
                    citations=_artifact_citations(approved_validate),
                )
            )

    estimation_report = snapshot.estimation_report
    if estimation_report is not None and estimation_report.is_stale:
        stale_stage_keys.append("estimate")
        issues.append(
            _issue(
                issue_key="estimate_stale",
                severity="warning",
                category="validate_to_estimate",
                title="Estimate ya no esta alineado con el blueprint vigente",
                detail=" | ".join(_normalized_list(estimation_report.stale_reasons))
                or "La estimacion actual necesita recalculo antes de usarse como referencia vigente.",
                affected_stage_keys=["estimate"],
                source_refs=["estimation_report"],
            )
        )

    if restricted_lineage:
        issues.append(
            _issue(
                issue_key="restricted_lineage_excluded_from_exports",
                severity="info",
                category="export_governance",
                title="Lineage restringido excluido del export",
                detail=(
                    "Se detectaron referencias de memoria/plataforma no exportables y quedaron fuera del lineage "
                    "publicable del blueprint profesional/ACP."
                ),
                affected_stage_keys=["memory", "package"],
                source_refs=["journey.evidence_manifest", "knowledge_manifest"],
                citations=restricted_lineage[:6],
            )
        )

    blocking_issues = [item.detail for item in issues if item.severity == "blocking"]
    warnings = [item.detail for item in issues if item.severity == "warning"]
    if blocking_issues:
        overall_status = ReviewState.blocked
    elif warnings:
        overall_status = ReviewState.partial
    else:
        overall_status = ReviewState.complete

    if blocking_issues:
        summary = (
            f"Se detectaron {len(blocking_issues)} bloqueos de coherencia entre artefactos aprobados y "
            "el blueprint vigente."
        )
    elif warnings:
        summary = (
            f"La cadena aprobada es utilizable pero mantiene {len(warnings)} advertencias de cobertura o vigencia."
        )
    else:
        summary = "La cadena aprobada conserva coherencia transversal y el package puede derivarse de fuentes vigentes."

    return BlueprintConsistencyReport(
        overall_status=overall_status,
        summary=summary,
        generated_from_blueprint_version=_current_blueprint_version(snapshot),
        approved_stage_lineage=stage_lineage,
        issues=issues,
        blocking_issues=_dedupe(blocking_issues),
        warnings=_dedupe(warnings),
        uncovered_requirement_keys=_dedupe(uncovered_requirement_keys),
        orphan_design_role_keys=_dedupe(orphan_design_role_keys),
        orphan_tool_keys=_dedupe(orphan_tool_keys),
        orphan_memory_dependency_keys=_dedupe(orphan_memory_dependency_keys),
        stale_stage_keys=_dedupe(stale_stage_keys),
        exportable_lineage=exportable_lineage,
        restricted_lineage=restricted_lineage,
        decision_history=decision_history,
    )


def ensure_blueprint_consistency_report(snapshot: SessionSnapshot) -> BlueprintConsistencyReport:
    report = getattr(snapshot, "blueprint_consistency", None)
    if isinstance(report, BlueprintConsistencyReport) and (
        report.summary
        or report.approved_stage_lineage
        or report.issues
        or report.exportable_lineage
        or report.restricted_lineage
    ):
        return report
    return build_blueprint_consistency_report(snapshot)


def render_blueprint_consistency_markdown(report: BlueprintConsistencyReport) -> str:
    sections = [
        "# Blueprint consistency",
        "",
        f"- overall_status: {report.overall_status}",
        f"- generated_from_blueprint_version: {report.generated_from_blueprint_version or 'n/a'}",
        f"- summary: {report.summary}",
        "",
        "## Approved stage lineage",
    ]
    if report.approved_stage_lineage:
        for item in report.approved_stage_lineage:
            sections.append(
                (
                    f"- {item.stage_key}: v{item.version_number or 0} / {item.state} / action={item.source_action} / "
                    f"citations={', '.join(item.citation_labels) or 'none'}"
                )
            )
    else:
        sections.append("- No hay stages aprobados registrados.")

    sections.extend(["", "## Blocking issues"])
    if report.blocking_issues:
        sections.extend(f"- {item}" for item in report.blocking_issues)
    else:
        sections.append("- Ninguno.")

    sections.extend(["", "## Warnings"])
    if report.warnings:
        sections.extend(f"- {item}" for item in report.warnings)
    else:
        sections.append("- Ninguna.")

    sections.extend(
        [
            "",
            "## Coverage and lineage",
            f"- uncovered_requirement_keys: {', '.join(report.uncovered_requirement_keys) or 'none'}",
            f"- orphan_design_role_keys: {', '.join(report.orphan_design_role_keys) or 'none'}",
            f"- orphan_tool_keys: {', '.join(report.orphan_tool_keys) or 'none'}",
            f"- orphan_memory_dependency_keys: {', '.join(report.orphan_memory_dependency_keys) or 'none'}",
            f"- stale_stage_keys: {', '.join(report.stale_stage_keys) or 'none'}",
            f"- exportable_lineage: {', '.join(report.exportable_lineage) or 'none'}",
            f"- restricted_lineage: {', '.join(report.restricted_lineage) or 'none'}",
        ]
    )
    return "\n".join(sections)
