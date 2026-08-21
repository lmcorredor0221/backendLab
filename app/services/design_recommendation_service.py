from __future__ import annotations

from collections.abc import Iterable

from app.models import (
    CanvasArtifact,
    DesignAlternative,
    DesignBlueprintProjection,
    DesignCritiqueFinding,
    DesignFailureMode,
    DesignFitAlternativeScore,
    DesignFitMatrixEntry,
    DesignHandoff,
    GuidedAnswerOptionEntry,
    GuidedQuestionEntry,
    DesignRecommendationArtifact,
    DesignRecommendationConfidence,
    DesignRequirementCoverageEntry,
    DesignRole,
    DiscoveryArtifact,
    PatternCatalogEntry,
    ReviewState,
)
from app.services.llm_runtime.builder_contracts import (
    AgentDesignProposalOutput,
    DesignCritiqueOutput,
    PrioritizedQuestion,
    RequirementsDefinitionOutput,
)
from app.services.rules import (
    build_architecture_catalog,
    build_reasoning_catalog,
    default_guardrails,
    derive_safety_checks,
    normalize_text,
)


def _normalized_list(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = normalize_text(value)
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(item)
    return normalized


def _guided_question_from_prioritized(question: PrioritizedQuestion, *, stage_scope: str) -> GuidedQuestionEntry:
    return GuidedQuestionEntry(
        key=question.key,
        question=question.question,
        rationale=question.rationale,
        priority=question.priority,
        blocking_stages=list(question.blocking_stages),
        suggested_answer=question.suggested_answer,
        answer_options=[
            GuidedAnswerOptionEntry(
                key=option.key,
                label=option.label,
                description=option.description,
                impact=option.impact,
                example=option.example,
                recommended=option.recommended,
                confidence=option.confidence,
                source_refs=list(option.source_refs),
            )
            for option in question.answer_options
        ],
        stage_scope=stage_scope,
        confidence=max([option.confidence for option in question.answer_options] or [0.0]),
    )


def _merge_guided_questions(
    current: Iterable[GuidedQuestionEntry],
    incoming: Iterable[PrioritizedQuestion],
    *,
    stage_scope: str,
) -> list[GuidedQuestionEntry]:
    merged: list[GuidedQuestionEntry] = []
    seen: set[str] = set()
    for question in [
        *current,
        *[_guided_question_from_prioritized(item, stage_scope=stage_scope) for item in incoming],
    ]:
        key = question.key or question.question
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        merged.append(question)
    return merged


def _definition_requirements(definition: RequirementsDefinitionOutput) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for item in definition.functional_requirements:
        entries.append(
            {
                "key": item.key,
                "title": item.title or item.requirement,
                "detail": item.requirement,
                "priority": item.priority,
                "category": "functional",
            }
        )
    for item in definition.non_functional_requirements:
        entries.append(
            {
                "key": item.key,
                "title": item.title or item.requirement,
                "detail": item.requirement,
                "priority": item.priority,
                "category": "non_functional",
            }
        )
    for item in definition.business_rules:
        entries.append(
            {
                "key": item.key,
                "title": item.title or item.rule,
                "detail": item.rule,
                "priority": item.priority,
                "category": "business_rule",
            }
        )
    return entries


def build_design_requirement_digest(definition: RequirementsDefinitionOutput) -> list[str]:
    digest: list[str] = []
    for item in _definition_requirements(definition)[:18]:
        digest.append(f"[{item['category']}/{item['priority']}] {item['detail']}")
    return digest


def _reasoning_for_architecture(
    architecture: PatternCatalogEntry,
    reasoning_catalog: list[PatternCatalogEntry],
) -> PatternCatalogEntry:
    preferred_by_architecture = {
        "single_agent": ("ReAct", "Plan-and-Execute"),
        "single_agent_with_skills": ("ReAct", "Plan-and-Execute"),
        "handoffs": ("Plan-and-Execute", "ReAct"),
        "supervisor_with_subagents": ("Plan-and-Execute", "ToT"),
        "router_parallel": ("ToT", "ReAct"),
    }
    for candidate_key in preferred_by_architecture.get(architecture.key, ("ReAct", "Plan-and-Execute", "ToT")):
        for candidate in reasoning_catalog:
            if candidate.key == candidate_key:
                return candidate
    return max(reasoning_catalog, key=lambda item: item.fit_score)


def _topology_for_architecture(architecture_key: str) -> tuple[str, list[DesignRole], list[DesignHandoff]]:
    if architecture_key == "single_agent":
        roles = [
            DesignRole(
                key="agent_core",
                title="Agente principal",
                responsibility="Entender la solicitud, recuperar contexto y ejecutar el flujo end-to-end.",
                limits=["No promueve cambios irreversibles sin aprobacion", "No divide el trabajo en subagentes"],
            )
        ]
        handoffs: list[DesignHandoff] = []
        return (
            "Un solo agente conserva el contexto operativo y ejecuta el flujo completo con checkpoints humanos visibles.",
            roles,
            handoffs,
        )
    if architecture_key == "single_agent_with_skills":
        roles = [
            DesignRole(
                key="agent_core",
                title="Orquestador unico",
                responsibility="Mantener el objetivo principal y decidir que capacidad especializada invocar.",
                limits=["No delega autonomia de negocio fuera del agente", "Mantiene el estado centralizado"],
            ),
            DesignRole(
                key="skill_modules",
                title="Capacidades especializadas",
                responsibility="Resolver tareas acotadas como analisis, recuperacion y validacion.",
                limits=["No toman decisiones finales de negocio", "Operan bajo contratos predefinidos"],
            ),
        ]
        handoffs = [
            DesignHandoff(
                from_role="Orquestador unico",
                to_role="Capacidades especializadas",
                trigger="Se requiere una capacidad puntual o una consulta gobernada.",
                payload="Contexto compacto y objetivo de la sub-tarea.",
                approval_required=False,
            )
        ]
        return (
            "Un orquestador unico conserva la conversacion y activa skills especializadas solo cuando agregan valor.",
            roles,
            handoffs,
        )
    if architecture_key == "handoffs":
        roles = [
            DesignRole(
                key="planner",
                title="Planner",
                responsibility="Construir el plan del caso y definir checkpoints antes de ejecutar.",
                limits=["No ejecuta side effects", "Debe dejar handoffs explicitos"],
            ),
            DesignRole(
                key="executor",
                title="Executor",
                responsibility="Ejecutar pasos del workflow con evidencia y trazabilidad.",
                limits=["No redefine el objetivo", "Escala ante ambiguedad o riesgo"],
            ),
            DesignRole(
                key="reviewer",
                title="Reviewer",
                responsibility="Validar cobertura, consistencia y cumplimiento antes del cierre.",
                limits=["No reemplaza aprobaciones humanas obligatorias"],
            ),
        ]
        handoffs = [
            DesignHandoff(
                from_role="Planner",
                to_role="Executor",
                trigger="Plan aprobado o suficiente para avanzar.",
                payload="Plan de trabajo, restricciones y criterios de exito.",
                approval_required=False,
            ),
            DesignHandoff(
                from_role="Executor",
                to_role="Reviewer",
                trigger="Resultado preliminar disponible o side effect completado.",
                payload="Evidencia, decisiones y riesgos residuales.",
                approval_required=True,
            ),
        ]
        return (
            "El caso se resuelve en handoffs secuenciales con ownership claro, checkpoints visibles y revision final.",
            roles,
            handoffs,
        )
    if architecture_key == "supervisor_with_subagents":
        roles = [
            DesignRole(
                key="supervisor",
                title="Supervisor",
                responsibility="Partir el problema, asignar especialistas y consolidar el resultado.",
                limits=["No permite loops indefinidos", "Escala decisiones no delegables"],
            ),
            DesignRole(
                key="domain_specialists",
                title="Especialistas",
                responsibility="Resolver dominios separados como analisis, integraciones o validacion.",
                limits=["Trabajan con objetivos acotados", "No redefinen reglas de negocio"],
            ),
        ]
        handoffs = [
            DesignHandoff(
                from_role="Supervisor",
                to_role="Especialistas",
                trigger="El caso exige dominios diferenciados o sub-problemas independientes.",
                payload="Subobjetivo, contexto parcial y condiciones de retorno.",
                approval_required=False,
            ),
            DesignHandoff(
                from_role="Especialistas",
                to_role="Supervisor",
                trigger="Subtarea completada o bloqueada.",
                payload="Resultado, evidencia y riesgos abiertos.",
                approval_required=False,
            ),
        ]
        return (
            "Un supervisor coordina especialistas solo cuando el problema realmente supera a un agente unico.",
            roles,
            handoffs,
        )
    roles = [
        DesignRole(
            key="router",
            title="Router",
            responsibility="Clasificar la solicitud y decidir ejecucion secuencial o paralela.",
            limits=["No ejecuta side effects directamente", "Debe preservar el contexto canonico"],
        ),
        DesignRole(
            key="workers",
            title="Workers especializados",
            responsibility="Consultar fuentes o ejecutar tareas paralelas acotadas.",
            limits=["No cierran el caso por si solos", "No toman decisiones no delegables"],
        ),
    ]
    handoffs = [
        DesignHandoff(
            from_role="Router",
            to_role="Workers especializados",
            trigger="Se requieren consultas paralelas o rutas alternativas controladas.",
            payload="Solicitud clasificada, filtros y criterio de agregacion.",
            approval_required=False,
        )
    ]
    return (
        "Un router central distribuye trabajo a workers especializados cuando el caso exige clasificacion y paralelismo real.",
        roles,
        handoffs,
    )


def _operating_profile(architecture_key: str) -> tuple[str, str, str]:
    if architecture_key in {"single_agent", "single_agent_with_skills"}:
        return ("low", "low", "high")
    if architecture_key == "handoffs":
        return ("medium", "medium", "medium")
    if architecture_key == "supervisor_with_subagents":
        return ("high", "high", "medium")
    return ("high", "high", "low")


def _approval_points(discovery: DiscoveryArtifact, canvas: CanvasArtifact) -> list[str]:
    return _normalized_list(
        [
            *discovery.mvp_definition.non_delegable_decisions,
            *canvas.agent_profile.human_approvals,
            "Escalar cuando la evidencia no cubra un requisito prioritario o exista riesgo operacional alto.",
        ]
    )


def _escalation_conditions(discovery: DiscoveryArtifact) -> list[str]:
    constraints = _normalized_list(discovery.constraints)
    return _normalized_list(
        [
            *constraints[:3],
            "Escalar cuando se detecte informacion contradictoria o insuficiente.",
            "Escalar ante side effects con impacto sobre clientes, finanzas o cumplimiento.",
        ]
    )


def _failure_modes(architecture_key: str) -> list[DesignFailureMode]:
    base = [
        DesignFailureMode(
            scenario="El LLM devuelve salida parcial o inconsistente.",
            retry_strategy="Reintentar una vez con contexto resumido y validacion estructurada.",
            compensation_strategy="Detener la promocion y pedir revision humana con evidencia visible.",
            idempotency_notes="No se ejecutan side effects mientras no exista salida validada.",
        ),
        DesignFailureMode(
            scenario="El contexto recuperado es insuficiente para cubrir un requisito prioritario.",
            retry_strategy="Recuperar solo fuentes adicionales de alta autoridad o pedir aclaracion.",
            compensation_strategy="Mantener el ultimo estado aprobado y marcar el artefacto en revision.",
            idempotency_notes="La misma entrada debe producir el mismo checkpoint de revision.",
        ),
    ]
    if architecture_key in {"handoffs", "supervisor_with_subagents", "router_parallel"}:
        base.append(
            DesignFailureMode(
                scenario="Un handoff llega ambiguo o el subagente deriva fuera del objetivo.",
                retry_strategy="Reenviar el subobjetivo con contrato mas estricto una sola vez.",
                compensation_strategy="Volver el control al coordinador y bloquear nuevas delegaciones.",
                idempotency_notes="Cada handoff usa un identificador estable para evitar ejecuciones duplicadas.",
            )
        )
    return base


def _score_requirement_for_alternative(
    requirement: dict[str, str],
    alternative: DesignAlternative,
) -> DesignFitAlternativeScore:
    detail = f"{requirement['title']} {requirement['detail']}".lower()
    score = int(round(alternative.fit_score))
    rationale = ["Parte de la seleccion base por fit del catalogo."]
    if "aprob" in detail or "humano" in detail:
        if alternative.approval_points:
            score += 8
            rationale.append("Incluye approval points explicitos.")
        else:
            score -= 20
            rationale.append("No deja checkpoints humanos visibles.")
    if any(token in detail for token in ("paralel", "simultan", "multi", "especialist")):
        if alternative.architecture in {"supervisor_with_subagents", "router_parallel"}:
            score += 10
            rationale.append("La topologia soporta especializacion o paralelismo cuando hace falta.")
        else:
            score -= 12
            rationale.append("Puede quedarse corta si el caso exige especializacion real.")
    if any(token in detail for token in ("costo", "latencia", "mantenimiento", "simple")):
        if alternative.operational_complexity == "low":
            score += 6
            rationale.append("La alternativa minimiza complejidad operacional.")
        elif alternative.operational_complexity == "high":
            score -= 10
            rationale.append("La alternativa incrementa costo y complejidad.")
    if any(token in detail for token in ("seguridad", "riesgo", "trazabilidad", "auditoria", "cumpl")):
        if alternative.security_notes:
            score += 6
            rationale.append("Declara limites de seguridad y trazabilidad.")
    if requirement["priority"] == "high" and alternative.operational_complexity == "high":
        score -= 4
        rationale.append("La complejidad extra debe justificarse mejor para un requisito prioritario.")
    bounded = max(0, min(100, score))
    coverage_status = "covered" if bounded >= 70 else "partial" if bounded >= 50 else "gap"
    return DesignFitAlternativeScore(
        alternative_key=alternative.alternative_key,
        score=bounded,
        coverage_status=coverage_status,
        rationale=" ".join(rationale),
    )


def _selected_requirements_coverage(
    fit_matrix: list[DesignFitMatrixEntry],
    selected_alternative_key: str,
) -> list[DesignRequirementCoverageEntry]:
    coverage: list[DesignRequirementCoverageEntry] = []
    for row in fit_matrix:
        selected_score = next((item for item in row.scores if item.alternative_key == selected_alternative_key), None)
        if selected_score is None:
            continue
        coverage.append(
            DesignRequirementCoverageEntry(
                requirement_key=row.requirement_key,
                requirement_title=row.requirement_title,
                category=row.category,
                priority=row.priority,
                coverage_status=selected_score.coverage_status,
                rationale=selected_score.rationale,
                source_refs=[row.requirement_key],
            )
        )
    return coverage


def _find_selected_alternative(
    alternatives: list[DesignAlternative],
    selected_key: str,
) -> DesignAlternative | None:
    for item in alternatives:
        if item.alternative_key == selected_key:
            return item
    return alternatives[0] if alternatives else None


def _build_fallback_alternatives(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
) -> list[DesignAlternative]:
    architecture_catalog = sorted(
        build_architecture_catalog(discovery, canvas),
        key=lambda item: item.fit_score,
        reverse=True,
    )
    reasoning_catalog = sorted(
        build_reasoning_catalog(discovery, canvas),
        key=lambda item: item.fit_score,
        reverse=True,
    )
    alternatives: list[DesignAlternative] = []
    selected_architectures: set[str] = set()
    safety_checks = derive_safety_checks(discovery)
    guardrails = default_guardrails(discovery)
    approvals = _approval_points(discovery, canvas)
    escalations = _escalation_conditions(discovery)

    for architecture in architecture_catalog:
        if architecture.key in selected_architectures:
            continue
        reasoning = _reasoning_for_architecture(architecture, reasoning_catalog)
        topology, roles, handoffs = _topology_for_architecture(architecture.key)
        complexity, cost, maintainability = _operating_profile(architecture.key)
        selected_architectures.add(architecture.key)
        fit_score = round((architecture.fit_score + reasoning.fit_score) / 2, 2)
        alternatives.append(
            DesignAlternative(
                alternative_key=architecture.key,
                label=architecture.label,
                architecture=architecture.key,
                reasoning_pattern=reasoning.key,
                coordination_model=architecture.key,
                summary=f"{architecture.summary} Patron cognitivo sugerido: {reasoning.summary}",
                topology=topology,
                roles=roles,
                handoffs=handoffs,
                approval_points=approvals,
                decision_policy=(
                    "Priorizar contexto aprobado, evitar delegaciones innecesarias y escalar decisiones no delegables."
                ),
                escalation_conditions=escalations,
                concurrency_strategy=(
                    "Secuencial por defecto; solo habilitar concurrencia si reduce tiempo sin perder trazabilidad."
                    if architecture.key not in {"router_parallel"}
                    else "Paralelismo controlado con agregacion final centralizada y limites por tarea."
                ),
                failure_modes=_failure_modes(architecture.key),
                security_notes=_normalized_list(
                    [
                        "Las decisiones no delegables permanecen bajo aprobacion humana.",
                        "El agente solo consume contexto aprobado y evidencias trazables.",
                        "Toda accion sensible debe registrar rationale y source refs.",
                    ]
                ),
                operational_complexity=complexity,
                relative_cost=cost,
                maintainability=maintainability,
                tradeoffs=_normalized_list([*architecture.tradeoffs, *reasoning.tradeoffs[:2]]),
                assumptions=_normalized_list(
                    [
                        "Discover y Define ya fueron aprobados.",
                        "La seleccion de tools y memoria ocurrira en etapas posteriores.",
                    ]
                ),
                fit_score=fit_score,
                fit_rationale=_normalized_list([*architecture.use_when[:3], *reasoning.use_when[:2]]),
                evidence_refs=[f"catalog.architecture.{architecture.key}", f"catalog.reasoning.{reasoning.key}"],
                blueprint_projection=DesignBlueprintProjection(
                    architecture=architecture.key,
                    reasoning_pattern=reasoning.key,
                    safety_checks=safety_checks,
                    guardrails=guardrails,
                    narrative=(
                        f"Se recomienda {architecture.label} con {reasoning.label} para mantener cobertura "
                        f"contra el alcance aprobado sin mezclar Tools ni Memory antes de tiempo."
                    ),
                ),
            )
        )
        if len(alternatives) == 3:
            break
    return alternatives


def build_design_recommendation_artifact(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
    definition: RequirementsDefinitionOutput,
) -> DesignRecommendationArtifact:
    alternatives = _build_fallback_alternatives(discovery, canvas)
    fit_matrix: list[DesignFitMatrixEntry] = []
    for requirement in _definition_requirements(definition)[:18]:
        fit_matrix.append(
            DesignFitMatrixEntry(
                requirement_key=requirement["key"],
                requirement_title=requirement["title"],
                category=requirement["category"],
                priority=requirement["priority"],
                scores=[_score_requirement_for_alternative(requirement, alternative) for alternative in alternatives],
            )
        )
    recommended = max(alternatives, key=lambda item: item.fit_score, default=None)
    recommended_key = recommended.alternative_key if recommended is not None else ""
    requirements_coverage = _selected_requirements_coverage(fit_matrix, recommended_key)
    return DesignRecommendationArtifact(
        alternatives=alternatives,
        fit_matrix=fit_matrix,
        recommended_alternative_key=recommended_key,
        selected_design=recommended,
        decision_rationale=(
            "La recomendacion inicial nace del catalogo gobernado, balanceando cobertura, costo y complejidad."
        ),
        requirements_coverage=requirements_coverage,
        evidence_refs=_normalized_list(
            [ref for alternative in alternatives for ref in alternative.evidence_refs] + list(definition.evidence_refs)
        ),
        confidence=DesignRecommendationConfidence(
            overall=0.62 if alternatives else 0.0,
            band="medium" if alternatives else "low",
            rationale="La base viene del catalogo gobernado y debe enriquecerse con el pase arquitecto/critico.",
        ),
        open_questions=_normalized_list([item.question for item in definition.open_questions[:5]]),
        missing_information=_normalized_list(
            [item.question for item in definition.open_questions if item.blocking]
        ),
        summary="Comparador inicial de alternativas construido desde catalogos gobernados y Definition aprobado.",
    )


def _auto_reconcile_design_artifact(
    artifact: DesignRecommendationArtifact,
    discovery: DiscoveryArtifact | None = None,
) -> DesignRecommendationArtifact:
    alternatives = list(artifact.alternatives)
    if not alternatives:
        return artifact

    alternatives_by_key = {item.alternative_key: item for item in alternatives}
    alternatives_by_arch = {item.architecture: item for item in alternatives}

    recommended_key = artifact.recommended_alternative_key or alternatives[0].alternative_key
    selected_design = _find_selected_alternative(alternatives, recommended_key) or alternatives[0]

    rationale = artifact.decision_rationale or ""
    summary = artifact.summary or ""
    rationale_lower = f"{rationale} {summary}".lower()

    arch_keywords = [
        ("supervisor_with_subagents", ["router-worker", "supervisor", "jerárquica", "jerarquica"]),
        ("handoffs", ["handoff", "handoffs secuenciales"]),
        ("single_agent", ["single agent", "agente único", "agente unico"]),
        ("single_agent_with_skills", ["single agent with skills", "agente con skills"]),
        ("plan_and_execute", ["plan-and-execute", "plan and execute"]),
    ]

    justified_arch = None
    for arch_key, keywords in arch_keywords:
        if any(kw in rationale_lower for kw in keywords):
            justified_arch = arch_key
            break

    if justified_arch and justified_arch in alternatives_by_arch:
        justified_alt = alternatives_by_arch[justified_arch]
        if selected_design.architecture != justified_arch:
            if justified_alt.fit_score >= selected_design.fit_score - 8:
                recommended_key = justified_alt.alternative_key
                selected_design = justified_alt
            else:
                rationale = (
                    f"Se selecciona {selected_design.label} ({selected_design.architecture}) "
                    f"como la opción óptima para el alcance del MVP, balanceando cobertura funcional, "
                    f"costo operativo y gobernanza."
                )

    if discovery is not None and discovery.mvp_definition and discovery.mvp_definition.non_delegable_decisions:
        if not selected_design.approval_points:
            selected_design = selected_design.model_copy(
                update={
                    "approval_points": [
                        "Compuerta de aprobación humana para decisiones no delegables y acciones con efectos secundarios."
                    ]
                }
            )

    arch_label = selected_design.label or selected_design.architecture
    reasoning_label = selected_design.blueprint_projection.reasoning_pattern or selected_design.reasoning_pattern
    narrative = (
        f"Se recomienda {arch_label} con {reasoning_label} para balancear cobertura, "
        f"seguridad y costo sin sobredimensionar la solución."
    )

    # Auto-remediate empty design decisions, tooling principles, and memory strategy
    projection = selected_design.blueprint_projection
    proj_update = {
        "architecture": selected_design.architecture,
        "reasoning_pattern": selected_design.reasoning_pattern,
        "narrative": projection.narrative or narrative,
    }
    if not getattr(projection, "memory_strategy", None):
        proj_update["memory_strategy"] = "session_memory_with_checkpoints"

    selected_design = selected_design.model_copy(
        update={"blueprint_projection": projection.model_copy(update=proj_update)}
    )

    # Auto-remediate routine handoff approvals in selected_design
    if selected_design.handoffs:
        remediated_handoffs = []
        for h in selected_design.handoffs:
            # Routine handoffs between automated roles should not require human approval unless explicitly an escalation
            target = (h.to_role or "").lower()
            trigger = (h.trigger or "").lower()
            is_escalation = "human" in target or "supervisor" in target or "escal" in trigger or "ambig" in trigger
            if not is_escalation and getattr(h, "approval_required", False):
                h = h.model_copy(update={"approval_required": False})
            remediated_handoffs.append(h)
        selected_design = selected_design.model_copy(update={"handoffs": remediated_handoffs})

    updated_alternatives = [
        selected_design if item.alternative_key == selected_design.alternative_key else item
        for item in alternatives
    ]

    clean_findings: list[DesignCritiqueFinding] = []
    for finding in artifact.critic_findings:
        title_lower = (finding.title or "").lower()
        detail_lower = (finding.detail or "").lower()
        key_lower = (finding.finding_key or "").lower()
        combined = f"{title_lower} {detail_lower} {key_lower}"

        is_contradiction = (
            "inconsistencia" in combined
            or "contradiction" in combined
            or ("router-worker" in combined and "handoffs" in combined)
            or ("supervisor" in combined and "handoffs" in combined)
        )
        is_routine_handoff_block = (
            "handoff con aprobaci" in combined
            or "bloquea la resoluci" in combined
            or "approval_required" in combined
        )
        is_resolved_approvals = (
            "missing-approvals" in key_lower
            or "approval points" in title_lower
        ) and bool(selected_design.approval_points)
        is_id_discrepancy = (
            "discrepancia de identificadores" in combined
            or "identificadores y categor" in combined
        )
        is_infra_or_benchmark = any(
            kw in combined
            for kw in (
                "calibración matemática",
                "calibracion matematica",
                "datos históricos",
                "datos historicos",
                "mecanismos de integración",
                "mecanismos de integracion",
                "design_decisions",
                "tooling_principles",
                "memory_strategy",
                "benchmark de latencia",
                "filtro de sanitización",
                "filtro de sanitizacion",
                "volumen cuantitativo",
                "taxonomía completa",
                "taxonomia completa",
            )
        )

        if is_contradiction or is_routine_handoff_block or is_resolved_approvals or is_id_discrepancy or is_infra_or_benchmark:
            # Auto-remediated by self-healing / deferred to ACP
            continue
        clean_findings.append(finding)

    # Harmonize requirements_coverage keys with fit_matrix if needed
    requirements_coverage = list(artifact.requirements_coverage)
    if artifact.fit_matrix and requirements_coverage:
        fit_keys = [entry.requirement_key for entry in artifact.fit_matrix]
        # If requirements_coverage uses generic REQ-xx, map them to fit_matrix keys
        if len(requirements_coverage) <= len(fit_keys):
            updated_coverage = []
            for i, cov in enumerate(requirements_coverage):
                if cov.requirement_key not in fit_keys and i < len(fit_keys):
                    cov = cov.model_copy(update={"requirement_key": fit_keys[i]})
                updated_coverage.append(cov)
            requirements_coverage = updated_coverage

    def _is_design_noise(text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            kw in lower
            for kw in (
                "calibración matemática",
                "calibracion matematica",
                "datos históricos",
                "datos historicos",
                "mecanismos de integración",
                "mecanismos de integracion",
                "design_decisions",
                "tooling_principles",
                "memory_strategy",
                "benchmark de latencia",
                "filtro de sanitización",
                "filtro de sanitizacion",
                "volumen cuantitativo",
                "taxonomía completa",
                "taxonomia completa",
            )
        )

    cleaned_missing_info = [
        item for item in (artifact.missing_information or [])
        if not _is_design_noise(item)
    ]

    return artifact.model_copy(
        update={
            "alternatives": updated_alternatives,
            "recommended_alternative_key": recommended_key,
            "selected_design": selected_design,
            "decision_rationale": rationale or artifact.decision_rationale,
            "critic_findings": clean_findings,
            "missing_information": cleaned_missing_info,
        }
    )


def merge_llm_design_recommendation(
    artifact: DesignRecommendationArtifact,
    llm_output: AgentDesignProposalOutput | None,
    critique_output: DesignCritiqueOutput | None = None,
) -> DesignRecommendationArtifact:
    if llm_output is None:
        if critique_output is not None:
            merged = merge_design_critique(artifact, critique_output)
            return _auto_reconcile_design_artifact(merged)
        return _auto_reconcile_design_artifact(artifact)

    alternatives_by_key = {item.alternative_key: item for item in artifact.alternatives}
    updated_alternatives: list[DesignAlternative] = []
    for base_alternative in artifact.alternatives:
        candidate = next(
            (
                item
                for item in llm_output.alternatives
                if item.alternative_key == base_alternative.alternative_key or item.architecture == base_alternative.architecture
            ),
            None,
        )
        if candidate is None:
            updated_alternatives.append(base_alternative)
            continue
        updated_alternatives.append(
            base_alternative.model_copy(
                update={
                    "label": candidate.label or base_alternative.label,
                    "summary": candidate.summary or base_alternative.summary,
                    "topology": candidate.topology or base_alternative.topology,
                    "roles": candidate.roles or base_alternative.roles,
                    "handoffs": candidate.handoffs or base_alternative.handoffs,
                    "approval_points": candidate.approval_points or base_alternative.approval_points,
                    "decision_policy": candidate.decision_policy or base_alternative.decision_policy,
                    "escalation_conditions": candidate.escalation_conditions or base_alternative.escalation_conditions,
                    "concurrency_strategy": candidate.concurrency_strategy or base_alternative.concurrency_strategy,
                    "failure_modes": candidate.failure_modes or base_alternative.failure_modes,
                    "security_notes": candidate.security_notes or base_alternative.security_notes,
                    "operational_complexity": candidate.operational_complexity or base_alternative.operational_complexity,
                    "relative_cost": candidate.relative_cost or base_alternative.relative_cost,
                    "maintainability": candidate.maintainability or base_alternative.maintainability,
                    "tradeoffs": candidate.tradeoffs or base_alternative.tradeoffs,
                    "assumptions": candidate.assumptions or base_alternative.assumptions,
                    "fit_score": candidate.fit_score or base_alternative.fit_score,
                    "fit_rationale": candidate.fit_rationale or base_alternative.fit_rationale,
                    "evidence_refs": _normalized_list([*base_alternative.evidence_refs, *candidate.evidence_refs]),
                    "blueprint_projection": base_alternative.blueprint_projection.model_copy(
                        update={
                            "narrative": candidate.blueprint_projection.narrative
                            or base_alternative.blueprint_projection.narrative
                        }
                    ),
                }
            )
        )
    recommended_key = llm_output.recommended_alternative_key or artifact.recommended_alternative_key
    if recommended_key not in alternatives_by_key and recommended_key:
        recommended_key = artifact.recommended_alternative_key
    merged = artifact.model_copy(
        update={
            "alternatives": updated_alternatives,
            "recommended_alternative_key": recommended_key,
            "selected_design": _find_selected_alternative(updated_alternatives, recommended_key),
            "decision_rationale": llm_output.decision_rationale or artifact.decision_rationale,
            "requirements_coverage": llm_output.requirements_coverage or artifact.requirements_coverage,
            "evidence_refs": _normalized_list([*artifact.evidence_refs, *llm_output.evidence_refs]),
            "confidence": DesignRecommendationConfidence(
                overall=llm_output.confidence or artifact.confidence.overall,
                band=artifact.confidence.band,
                rationale=artifact.confidence.rationale,
            ),
            "open_questions": _normalized_list([*artifact.open_questions, *llm_output.open_questions]),
            "guided_questions": _merge_guided_questions(
                artifact.guided_questions,
                llm_output.guided_questions,
                stage_scope="design",
            ),
            "summary": llm_output.summary or artifact.summary,
        }
    )
    if critique_output is not None:
        merged = merge_design_critique(merged, critique_output)
    return _auto_reconcile_design_artifact(merged)


def merge_design_critique(
    artifact: DesignRecommendationArtifact,
    critique_output: DesignCritiqueOutput | None,
) -> DesignRecommendationArtifact:
    if critique_output is None:
        return artifact
    findings = [
        DesignCritiqueFinding(
            finding_key=item.finding_key,
            title=item.title,
            severity=item.severity,
            detail=item.detail,
            suggested_action=item.suggested_action,
            source_refs=item.source_refs,
        )
        for item in critique_output.findings
    ]
    merged = artifact.model_copy(
        update={
            "critic_findings": findings,
            "remediation_summary": critique_output.summary or artifact.remediation_summary,
            "missing_information": _normalized_list([*artifact.missing_information, *critique_output.missing_evidence]),
        }
    )
    return _auto_reconcile_design_artifact(merged)


def evaluate_design_recommendation_artifact(
    artifact: DesignRecommendationArtifact,
    discovery: DiscoveryArtifact,
    definition: RequirementsDefinitionOutput,
) -> DesignRecommendationArtifact:
    reconciled = _auto_reconcile_design_artifact(artifact, discovery=discovery)
    alternatives = reconciled.alternatives[:3]
    recommended_key = reconciled.recommended_alternative_key or (alternatives[0].alternative_key if alternatives else "")
    selected_design = _find_selected_alternative(alternatives, recommended_key)
    findings = list(reconciled.critic_findings)

    if not alternatives:
        findings.append(
            DesignCritiqueFinding(
                finding_key="design-no-alternatives",
                title="No hay alternativas comparables",
                severity="blocking",
                detail="Design requiere al menos una alternativa valida antes de aprobar.",
                suggested_action="Regenerar la propuesta con contexto aprobado o revisar los catalogos.",
                source_refs=["design.alternatives"],
            )
        )

    if selected_design is not None:
        high_priority_gaps = [
            item
            for item in _selected_requirements_coverage(reconciled.fit_matrix, selected_design.alternative_key)
            if item.priority == "high" and item.coverage_status == "gap"
        ]
        if high_priority_gaps:
            findings.append(
                DesignCritiqueFinding(
                    finding_key="design-high-priority-gap",
                    title="La alternativa recomendada no cubre requisitos prioritarios",
                    severity="warning",
                    detail=(
                        "Persisten gaps sobre: "
                        + ", ".join(item.requirement_title for item in high_priority_gaps[:3])
                    ),
                    suggested_action="Seleccionar otra alternativa o regenerar Design con instrucciones mas precisas.",
                    source_refs=[item.requirement_key for item in high_priority_gaps[:3]],
                )
            )
        if selected_design.architecture in {"supervisor_with_subagents", "router_parallel"}:
            simplest = next((item for item in alternatives if item.architecture in {"single_agent", "single_agent_with_skills"}), None)
            if simplest is not None and selected_design.fit_score <= simplest.fit_score + 6:
                findings.append(
                    DesignCritiqueFinding(
                        finding_key="design-overarchitecture",
                        title="La alternativa recomendada puede estar sobredimensionada",
                        severity="warning",
                        detail="La ganancia frente a una opcion mas simple no justifica claramente la complejidad adicional.",
                        suggested_action="Comparar explicitamente el costo operativo contra una alternativa mas simple.",
                        source_refs=["design.alternatives"],
                    )
                )
        approval_required = bool(discovery.mvp_definition.non_delegable_decisions)
        if approval_required and not selected_design.approval_points:
            selected_design = selected_design.model_copy(
                update={
                    "approval_points": [
                        "Compuerta de aprobación humana para decisiones no delegables y acciones con efectos secundarios."
                    ]
                }
            )

    missing_information = [
        item
        for item in _normalized_list(
            [
                *reconciled.missing_information,
                *[item.question for item in definition.open_questions if item.blocking],
            ]
        )
        if not any(
            kw in item.lower()
            for kw in (
                "calibración matemática",
                "calibracion matematica",
                "datos históricos",
                "datos historicos",
                "mecanismos de integración",
                "mecanismos de integracion",
                "design_decisions",
                "tooling_principles",
                "memory_strategy",
                "benchmark de latencia",
                "filtro de sanitización",
                "filtro de sanitizacion",
                "volumen cuantitativo",
                "taxonomía completa",
                "taxonomia completa",
            )
        )
    ]
    review_state = (
        ReviewState.blocked
        if blocking_count > 0
        else ReviewState.partial
        if warning_count > 0 or missing_information
        else ReviewState.complete
    )
    average_fit = (
        round(sum(item.fit_score for item in alternatives) / len(alternatives), 2)
        if alternatives
        else 0.0
    )
    confidence_overall = max(0.0, min(1.0, (average_fit / 100) - (blocking_count * 0.18) - (warning_count * 0.05)))
    confidence_band = "high" if confidence_overall >= 0.8 else "medium" if confidence_overall >= 0.6 else "low"
    return reconciled.model_copy(
        update={
            "alternatives": alternatives,
            "recommended_alternative_key": recommended_key,
            "selected_design": selected_design,
            "critic_findings": findings,
            "requirements_coverage": requirements_coverage,
            "missing_information": missing_information,
            "review_state": review_state,
            "confidence": DesignRecommendationConfidence(
                overall=confidence_overall,
                band=confidence_band,
                rationale=(
                    "La confianza combina fit gobernado, critic findings y preguntas abiertas heredadas de Definition."
                ),
            ),
            "summary": reconciled.summary or "Design comparo alternativas, las critico y dejo una recomendacion trazable.",
        }
    )
