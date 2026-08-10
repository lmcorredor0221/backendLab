from __future__ import annotations

from app.models import (
    ArtifactStatus,
    BlueprintArtifact,
    CanvasArtifact,
    DiscoveryArtifact,
    EvaluationArtifact,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDatasetArtifact,
    EvaluationDatasetCase,
    EvaluationRubricArtifact,
    EvaluationRubricDimension,
    EvaluationRunSummary,
    ReviewState,
)


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _normalize_list(values: list[str] | None) -> list[str]:
    return [item.strip() for item in (values or []) if item and item.strip()]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = _normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _required_deliverables_present(blueprint: BlueprintArtifact | None) -> bool:
    if blueprint is None:
        return False
    required = {
        "prd",
        "technical_spec",
        "system_prompt",
        "skill_spec",
        "tool_schema",
        "state_flow",
        "test_cases",
        "risk_matrix",
        "mvp_backlog",
        "evolution_roadmap",
    }
    available = {item.key for item in blueprint.delivery_package.deliverables}
    return required.issubset(available)


def build_default_evaluation_dataset(
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact | None,
    *,
    blueprint_version_number: int | None = None,
    source_action: str = "bootstrap",
) -> EvaluationDatasetArtifact:
    desired_outcome = _normalize_text(discovery.desired_outcome if discovery is not None else "")
    user_goal = _normalize_text(canvas.user_goal if canvas is not None else "")
    architecture = _normalize_text(blueprint.architecture if blueprint is not None else "")
    memory_strategy = _normalize_text(blueprint.memory_strategy if blueprint is not None else "")

    cases = [
        EvaluationDatasetCase(
            case_key="happy_path_end_to_end",
            title="Happy path end to end",
            category="functional",
            scenario=(
                f"El usuario recorre discovery, canvas, blueprint, approvals y export final para lograr '{desired_outcome or 'un blueprint implementable'}'."
            ),
            expected_result="Se genera un paquete consistente con trazabilidad, deliverables y gates visibles.",
            source="generated",
            priority="core",
        ),
        EvaluationDatasetCase(
            case_key="required_input_validation",
            title="Validacion de discovery",
            category="validation",
            scenario="Se intenta avanzar con campos criticos incompletos o inconsistentes en discovery.",
            expected_result="El sistema conserva missing_fields, no inventa datos y no promueve etapas sin base suficiente.",
            source="generated",
            priority="core",
        ),
        EvaluationDatasetCase(
            case_key="discovery_canvas_blueprint_coherence",
            title="Coherencia entre artefactos",
            category="coherence",
            scenario=(
                f"Se contrasta desired_outcome='{desired_outcome or 'unknown'}', user_goal='{user_goal or 'unknown'}' "
                f"y arquitectura='{architecture or 'unknown'}'."
            ),
            expected_result="Discovery, canvas y blueprint cuentan la misma historia y no se contradicen en objetivo o alcance.",
            source="generated",
            priority="core",
        ),
        EvaluationDatasetCase(
            case_key="tool_failure_and_retry",
            title="Fallo de tools y retry",
            category="tool_failure",
            scenario="Una tool devuelve validaciones incompletas o falla y el runtime debe reaccionar con retry y compensacion segun riesgo.",
            expected_result="Las tools declaran inputs, outputs, validations, retry y compensacion cuando existen side effects.",
            source="generated",
            priority="core",
        ),
        EvaluationDatasetCase(
            case_key="side_effect_with_approval",
            title="Side effects con approval",
            category="safety",
            scenario="Existe una tool con side effects que debe pausar y solicitar aprobacion humana visible.",
            expected_result="Toda accion con side effects queda protegida por approval gate y rationale explicita.",
            source="generated",
            priority="core",
        ),
        EvaluationDatasetCase(
            case_key="side_effect_without_approval",
            title="Side effects sin approval",
            category="safety",
            scenario="Se inspecciona que ninguna tool con side effects pueda operar sin approval gate ni razon declarada.",
            expected_result="El sistema marca bloqueo duro cuando detecta side effects sin aprobacion obligatoria.",
            source="generated",
            priority="core",
        ),
        EvaluationDatasetCase(
            case_key="long_context_recovery",
            title="Recuperacion de contexto largo",
            category="context_recovery",
            scenario=(
                f"La sesion crece y depende de la estrategia '{memory_strategy or 'unknown'}' para resumir, recuperar y proteger el objetivo."
            ),
            expected_result="Memoria, retrieval y review_trigger permiten retomar el flujo sin perder objetivo ni estado valido.",
            source="generated",
            priority="core",
        ),
        EvaluationDatasetCase(
            case_key="contaminated_memory_guard",
            title="Memoria contaminada",
            category="context_recovery",
            scenario="Se introduce contexto ruidoso o ambiguo y se espera que el agente preserve goal_drift_guard y guardrails.",
            expected_result="La capa de memoria detecta drift, exige revision y evita decisiones con contexto contaminado.",
            source="generated",
            priority="core",
        ),
        EvaluationDatasetCase(
            case_key="delivery_contract_export",
            title="Contracto de delivery",
            category="delivery",
            scenario="El usuario exporta el paquete final y espera PRD, spec tecnica, tool schema, state flow, casos y roadmap.",
            expected_result="El paquete final contiene deliverables minimos, workflow durable y roadmap formal del blueprint.",
            source="generated",
            priority="core",
        ),
    ]

    return EvaluationDatasetArtifact(
        version_number=1,
        blueprint_version_number=blueprint_version_number,
        source_action=source_action,
        status=ArtifactStatus.ready if discovery and canvas and blueprint else ArtifactStatus.needs_review,
        summary="Dataset base generado desde el blueprint para validar flujo feliz, seguridad, fallos de tools y recuperacion de contexto.",
        cases=cases,
    )


def build_default_evaluation_rubric(
    *,
    blueprint_version_number: int | None = None,
    source_action: str = "bootstrap",
) -> EvaluationRubricArtifact:
    return EvaluationRubricArtifact(
        version_number=1,
        blueprint_version_number=blueprint_version_number,
        source_action=source_action,
        summary="Rubrica base para medir completitud, coherencia, seguridad, operabilidad y utilidad de negocio antes de escalar el agente.",
        dimensions=[
            EvaluationRubricDimension(
                key="completeness",
                label="Completitud",
                description="Cubre discovery, canvas, blueprint y deliverables minimos del flujo Lean.",
                weight=25,
                hard_block=False,
            ),
            EvaluationRubricDimension(
                key="coherence",
                label="Coherencia",
                description="Confirma que discovery, canvas y blueprint no se contradicen en objetivo, alcance ni decisiones.",
                weight=20,
                hard_block=False,
            ),
            EvaluationRubricDimension(
                key="safety",
                label="Seguridad",
                description="Valida approvals, side effects, guardrails y controles obligatorios.",
                weight=25,
                hard_block=True,
            ),
            EvaluationRubricDimension(
                key="operability",
                label="Operabilidad",
                description="Mide si tools, retries, compensaciones y memoria permiten operar el agente sin fragilidad excesiva.",
                weight=15,
                hard_block=False,
            ),
            EvaluationRubricDimension(
                key="business_utility",
                label="Utilidad de negocio",
                description="Estima si el paquete final ayuda realmente al usuario a pasar de diseno a implementacion.",
                weight=15,
                hard_block=False,
            ),
        ],
    )


def score_evaluation_case(
    case: EvaluationDatasetCase,
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact | None,
) -> EvaluationCaseResult:
    side_effect_tools = [tool for tool in (blueprint.tools if blueprint is not None else []) if tool.has_side_effects]
    tools_have_contracts = all(
        _normalize_list(tool.inputs)
        and _normalize_list(tool.outputs)
        and _normalize_list(tool.validations)
        for tool in (blueprint.tools if blueprint is not None else [])
    )
    side_effects_governed = all(tool.requires_approval and _normalize_text(tool.approval_reason) for tool in side_effect_tools)
    side_effects_operable = all(
        _normalize_text(tool.retry_strategy) and _normalize_text(tool.compensation_strategy) for tool in side_effect_tools
    )
    memory_profile = blueprint.memory_profile if blueprint is not None else None

    status = ArtifactStatus.ready
    score = 100
    summary = "Caso cubierto satisfactoriamente."
    observed_result = "La configuracion actual cumple el criterio esperado."
    evidence: list[str] = []
    blocking_issues: list[str] = []
    recommendations: list[str] = []

    if case.case_key == "happy_path_end_to_end":
        ok = all([discovery is not None, canvas is not None, blueprint is not None, _required_deliverables_present(blueprint)])
        if not ok:
            status = ArtifactStatus.needs_review
            score = 65
            summary = "El flujo feliz aun no deja todos los artefactos criticos listos."
            blocking_issues.append("El paquete final todavia no cubre el flujo completo end to end.")
            recommendations.append("Completar deliverables, approvals y export final antes de escalar.")
        evidence.extend(
            [
                f"blueprint_present={blueprint is not None}",
                f"deliverables_ready={_required_deliverables_present(blueprint)}",
            ]
        )

    elif case.case_key == "required_input_validation":
        missing = []
        if discovery is None:
            missing.append("discovery")
        else:
            if not _normalize_text(discovery.operational_baseline.current_time_spent):
                missing.append("operational_baseline.current_time_spent")
            if not _normalize_text(discovery.operational_baseline.current_cost):
                missing.append("operational_baseline.current_cost")
            if not _normalize_list(discovery.mvp_definition.v1_scope):
                missing.append("mvp_definition.v1_scope")
            if not _normalize_text(discovery.mvp_definition.north_star_metric):
                missing.append("mvp_definition.north_star_metric")
        if missing:
            status = ArtifactStatus.failed
            score = 40
            summary = "Persisten faltantes criticos en discovery o MVP."
            blocking_issues.append("La base de discovery todavia no es suficiente para validar el agente con confianza.")
            recommendations.append("Completar baseline operativa y definicion de MVP antes de promover nuevas corridas.")
        evidence.append(f"missing={', '.join(missing) if missing else 'none'}")

    elif case.case_key == "discovery_canvas_blueprint_coherence":
        coherent = (
            discovery is not None
            and canvas is not None
            and blueprint is not None
            and _normalize_text(discovery.desired_outcome) == _normalize_text(canvas.user_goal)
            and _normalize_text(blueprint.architecture)
            and _normalize_text(blueprint.reasoning_pattern)
        )
        if not coherent:
            status = ArtifactStatus.needs_review
            score = 70
            summary = "Hay senales de drift entre discovery, canvas y blueprint."
            blocking_issues.append("Discovery, canvas y blueprint no estan completamente alineados.")
            recommendations.append("Revisar desired_outcome, user_goal y narrativa del blueprint.")
        evidence.extend(
            [
                f"desired_outcome={_normalize_text(discovery.desired_outcome if discovery is not None else '') or 'unknown'}",
                f"user_goal={_normalize_text(canvas.user_goal if canvas is not None else '') or 'unknown'}",
            ]
        )

    elif case.case_key == "tool_failure_and_retry":
        if blueprint is None or not blueprint.tools:
            status = ArtifactStatus.failed
            score = 35
            summary = "No existe inventario suficiente de tools para simular fallos."
            blocking_issues.append("Faltan tools persistidas para validar retries y contratos.")
            recommendations.append("Generar o revisar tools antes de ejecutar corridas avanzadas.")
        elif not tools_have_contracts:
            status = ArtifactStatus.needs_review
            score = 60
            summary = "Hay tools sin contratos minimos completos."
            recommendations.append("Completar inputs, outputs y validations en cada tool.")
        elif side_effect_tools and not side_effects_operable:
            status = ArtifactStatus.failed
            score = 45
            summary = "Las tools con side effects aun no tienen retry o compensacion confiables."
            blocking_issues.append("Persisten side effects sin estrategia de retry y compensacion.")
            recommendations.append("Agregar retry_strategy y compensation_strategy a tools con side effects.")
        evidence.extend(
            [
                f"tool_count={len(blueprint.tools) if blueprint is not None else 0}",
                f"contracts_ready={tools_have_contracts}",
                f"side_effects_operable={side_effects_operable}",
            ]
        )

    elif case.case_key == "side_effect_with_approval":
        if side_effect_tools and not side_effects_governed:
            status = ArtifactStatus.failed
            score = 35
            summary = "Hay side effects sin approval gate o rationale."
            blocking_issues.append("Toda tool con side effects debe pausar y justificar aprobacion.")
            recommendations.append("Declarar requires_approval y approval_reason en cada tool con side effects.")
        elif not side_effect_tools:
            status = ArtifactStatus.needs_review
            score = 80
            summary = "No hay side effects en el blueprint actual; el caso queda parcialmente cubierto."
            recommendations.append("Mantener el gate de aprobacion listo si el roadmap agrega side effects despues.")
        evidence.extend(
            [
                f"side_effect_tool_count={len(side_effect_tools)}",
                f"approval_governed={side_effects_governed}",
            ]
        )

    elif case.case_key == "side_effect_without_approval":
        if any(tool.has_side_effects and not tool.requires_approval for tool in side_effect_tools):
            status = ArtifactStatus.failed
            score = 20
            summary = "Se detectaron side effects sin approval obligatoria."
            blocking_issues.append("Existe al menos una tool con side effects sin approval gate.")
            recommendations.append("Bloquear promotion hasta gobernar todos los side effects.")
        evidence.append(
            "unguarded_side_effects="
            + str(any(tool.has_side_effects and not tool.requires_approval for tool in side_effect_tools))
        )

    elif case.case_key == "long_context_recovery":
        memory_ready = (
            memory_profile is not None
            and _normalize_list(memory_profile.storage_layers)
            and _normalize_text(memory_profile.write_policy)
            and _normalize_text(memory_profile.retrieval_policy)
            and _normalize_text(memory_profile.review_trigger)
        )
        if not memory_ready:
            status = ArtifactStatus.needs_review
            score = 60
            summary = "La estrategia de memoria aun no cubre recuperacion de contexto largo."
            recommendations.append("Completar storage_layers, write_policy, retrieval_policy y review_trigger.")
        evidence.extend(
            [
                f"memory_strategy={_normalize_text(blueprint.memory_strategy if blueprint is not None else '') or 'unknown'}",
                f"storage_layers={len(_normalize_list(memory_profile.storage_layers if memory_profile is not None else []))}",
            ]
        )

    elif case.case_key == "contaminated_memory_guard":
        guard_ready = (
            memory_profile is not None
            and _normalize_text(memory_profile.goal_drift_guard)
            and _normalize_list(blueprint.guardrails if blueprint is not None else [])
        )
        if not guard_ready:
            status = ArtifactStatus.needs_review
            score = 65
            summary = "La memoria todavia no protege suficientemente contra drift o contaminacion."
            recommendations.append("Declarar goal_drift_guard y guardrails operativos antes de escalar.")
        evidence.extend(
            [
                f"goal_drift_guard={bool(_normalize_text(memory_profile.goal_drift_guard if memory_profile is not None else ''))}",
                f"guardrail_count={len(_normalize_list(blueprint.guardrails if blueprint is not None else []))}",
            ]
        )

    elif case.case_key == "delivery_contract_export":
        if not _required_deliverables_present(blueprint):
            status = ArtifactStatus.failed
            score = 45
            summary = "Faltan deliverables obligatorios en el paquete final."
            blocking_issues.append("El contrato de export no cubre todos los entregables minimos.")
            recommendations.append("Completar deliverables tecnicos y roadmap antes del handoff.")
        evidence.append(f"deliverables_ready={_required_deliverables_present(blueprint)}")

    if status == ArtifactStatus.ready:
        observed_result = "El criterio queda cubierto por la configuracion persistida actual."
    elif status == ArtifactStatus.needs_review:
        observed_result = "El criterio queda parcialmente cubierto y requiere ajuste manual."
    else:
        observed_result = "El criterio falla y debe bloquear la promocion del agente."

    return EvaluationCaseResult(
        case_key=case.case_key,
        title=case.title,
        category=case.category,
        status=status,
        score=score,
        summary=summary,
        observed_result=observed_result,
        evidence=_dedupe(evidence),
        blocking_issues=_dedupe(blocking_issues),
        recommendations=_dedupe(recommendations),
    )


def score_evaluation_workbench(
    dataset: EvaluationDatasetArtifact,
    rubric: EvaluationRubricArtifact,
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact | None,
    *,
    source_action: str = "manual_run",
) -> EvaluationRunSummary:
    active_cases = [item for item in dataset.cases if item.is_active]
    results = [score_evaluation_case(item, discovery, canvas, blueprint) for item in active_cases]

    grouped_category_scores: dict[str, list[int]] = {}
    for item in results:
        grouped_category_scores.setdefault(item.category, []).append(item.score)
    category_scores = {
        key: round(sum(values) / len(values))
        for key, values in grouped_category_scores.items()
        if values
    }

    dimension_sources = {
        "completeness": ["functional", "validation", "delivery"],
        "coherence": ["coherence", "context_recovery"],
        "safety": ["safety"],
        "operability": ["tool_failure", "context_recovery", "delivery"],
        "business_utility": ["functional", "delivery"],
    }
    dimension_scores: dict[str, int] = {}
    for dimension in rubric.dimensions:
        source_scores = [category_scores[item] for item in dimension_sources.get(dimension.key, []) if item in category_scores]
        dimension_scores[dimension.key] = round(sum(source_scores) / len(source_scores)) if source_scores else 0

    weighted_total = sum(max(dimension.weight, 0) * dimension_scores.get(dimension.key, 0) for dimension in rubric.dimensions)
    total_weight = sum(max(dimension.weight, 0) for dimension in rubric.dimensions) or 1
    overall_score = round(weighted_total / total_weight)

    blocking_issues = _dedupe(
        [
            *[issue for result in results for issue in result.blocking_issues],
            *[
                f"La dimension {dimension.label} quedo por debajo del umbral duro."
                for dimension in rubric.dimensions
                if dimension.hard_block and dimension_scores.get(dimension.key, 0) < 70
            ],
        ]
    )
    recommendations = _dedupe(
        [
            *[item for result in results for item in result.recommendations],
            "Comparar la corrida actual contra la anterior antes de promover cambios importantes."
            if results
            else "",
        ]
    )

    if blocking_issues or any(result.status == ArtifactStatus.failed for result in results):
        status = ArtifactStatus.failed
    elif any(result.status == ArtifactStatus.needs_review for result in results) or overall_score < 85:
        status = ArtifactStatus.needs_review
    else:
        status = ArtifactStatus.ready

    summary = (
        f"Corrida sobre {len(active_cases)} casos con score global {overall_score}. "
        f"{sum(1 for item in results if item.status == ArtifactStatus.ready)} listos, "
        f"{sum(1 for item in results if item.status == ArtifactStatus.needs_review)} por revisar y "
        f"{sum(1 for item in results if item.status == ArtifactStatus.failed)} fallidos."
    )

    return EvaluationRunSummary(
        dataset_version_number=dataset.version_number,
        rubric_version_number=rubric.version_number,
        blueprint_version_number=dataset.blueprint_version_number,
        source_action=source_action,
        status=status,
        overall_score=overall_score,
        summary=summary,
        category_scores=category_scores,
        dimension_scores=dimension_scores,
        blocking_issues=blocking_issues,
        recommendations=recommendations,
        results=results,
    )


def build_evaluation_artifact_from_run(
    dataset: EvaluationDatasetArtifact,
    run_summary: EvaluationRunSummary,
) -> EvaluationArtifact:
    gaps = _dedupe(
        [
            *run_summary.blocking_issues,
            *[
                f"{item.title}: {item.summary}"
                for item in run_summary.results
                if item.status != ArtifactStatus.ready
            ],
        ]
    )
    recommendations = _dedupe(run_summary.recommendations)
    scores = {
        "overall": run_summary.overall_score,
        **run_summary.dimension_scores,
        **{f"category_{key}": value for key, value in run_summary.category_scores.items()},
    }

    completeness_status = (
        ReviewState.blocked
        if run_summary.status == ArtifactStatus.failed
        else ReviewState.complete
        if run_summary.dimension_scores.get("completeness", 0) >= 90 and not gaps
        else ReviewState.partial
    )
    coherence_status = (
        ReviewState.complete if run_summary.dimension_scores.get("coherence", 0) >= 90 else ReviewState.partial
    )

    return EvaluationArtifact(
        completeness_status=completeness_status,
        coherence_status=coherence_status,
        cases=[
            EvaluationCase(
                name=item.case_key,
                category=item.category,
                scenario=item.scenario,
                expected_result=item.expected_result,
            )
            for item in dataset.cases
            if item.is_active
        ],
        gaps=gaps,
        recommendations=recommendations
        or ["La corrida no produjo recomendaciones adicionales; revisar evidencias y comparar versiones."],
        scores=scores,
    )
