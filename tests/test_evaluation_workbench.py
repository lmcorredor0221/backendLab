from app.models import BlueprintArtifact, CanvasArtifact, DiscoveryArtifact, DiscoveryInput
from app.services.evaluation_workbench import (
    build_default_evaluation_dataset,
    build_default_evaluation_rubric,
    score_evaluation_workbench,
)
from app.services.skill_runtime import run_blueprint_stage, run_canvas_stage, run_discovery_stage


def complete_discovery_input() -> DiscoveryInput:
    return DiscoveryInput(
        problem_statement="Disenar agentes de soporte con metodologia Lean y bajo riesgo operativo.",
        current_user="Arquitecto de soluciones",
        current_process="Recoge discovery en documentos y luego redacta artefactos manualmente.",
        desired_outcome="Generar un blueprint implementable con tools, memoria y evaluacion.",
        autonomy_level="high",
        constraints=["Sin side effects irreversibles", "Mantener un MVP simple"],
        operational_baseline={
            "current_time_spent": "6 horas por caso",
            "current_cost": "Retrabajo tecnico y validaciones tardias",
            "frequent_errors": ["Se pierde contexto", "No se recorta el MVP"],
            "automation_opportunities": ["Normalizar discovery", "Generar artefactos base"],
        },
        mvp_definition={
            "v1_scope": ["Capturar discovery", "Construir canvas", "Construir blueprint"],
            "out_of_scope": ["Provisioning automatico"],
            "north_star_metric": "Blueprint util en una sola sesion",
            "non_delegable_decisions": ["Aprobar el handoff a implementacion"],
        },
    )


def build_artifacts() -> tuple[DiscoveryArtifact, CanvasArtifact, BlueprintArtifact]:
    discovery_envelope, _ = run_discovery_stage(complete_discovery_input())
    canvas_envelope, _ = run_canvas_stage(discovery_envelope.data)
    blueprint_envelope, _ = run_blueprint_stage(discovery_envelope.data, canvas_envelope.data)
    return discovery_envelope.data, canvas_envelope.data, blueprint_envelope.data


def test_default_evaluation_dataset_covers_stage_three_categories() -> None:
    discovery, canvas, blueprint = build_artifacts()

    dataset = build_default_evaluation_dataset(
        discovery,
        canvas,
        blueprint,
        blueprint_version_number=3,
    )

    assert dataset.version_number == 1
    assert dataset.blueprint_version_number == 3
    categories = {item.category for item in dataset.cases}
    assert {"tool_failure", "context_recovery", "safety", "delivery"}.issubset(categories)


def test_default_evaluation_rubric_contains_hard_block_safety_dimension() -> None:
    rubric = build_default_evaluation_rubric(blueprint_version_number=4)

    assert len(rubric.dimensions) == 5
    safety_dimension = next(item for item in rubric.dimensions if item.key == "safety")
    assert safety_dimension.hard_block is True


def test_score_evaluation_workbench_returns_results_for_all_active_cases() -> None:
    discovery, canvas, blueprint = build_artifacts()
    dataset = build_default_evaluation_dataset(discovery, canvas, blueprint, blueprint_version_number=5)
    rubric = build_default_evaluation_rubric(blueprint_version_number=5)

    run_summary = score_evaluation_workbench(
        dataset,
        rubric,
        discovery,
        canvas,
        blueprint,
        source_action="test_run",
    )

    assert run_summary.source_action == "test_run"
    assert run_summary.overall_score >= 0
    assert set(run_summary.dimension_scores).issuperset(
        {"completeness", "coherence", "safety", "operability", "business_utility"}
    )
    assert len(run_summary.results) == len([item for item in dataset.cases if item.is_active])
