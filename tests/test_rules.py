from app.models import CanvasArtifact, DiscoveryArtifact
from app.services.rules import build_reasoning_catalog, infer_case_type, select_architecture, select_reasoning_pattern


def test_infer_case_type_prefers_automation_for_high_autonomy() -> None:
    result = infer_case_type(
        problem_statement="Automatizar la construccion de agentes",
        desired_outcome="Reducir pasos manuales",
        autonomy_level="high",
    )
    assert result == "automatizacion"


def test_select_architecture_returns_single_agent_for_simple_case() -> None:
    discovery = DiscoveryArtifact(
        problem_statement="Definir un blueprint simple",
        current_user="Product manager",
        current_process="Recolecta requisitos por formulario",
        desired_outcome="Generar una propuesta util",
        autonomy_level="medium",
        constraints=[],
        case_type="copiloto",
        value_statement="Reducir ambiguedad",
    )
    assert select_architecture(discovery) == "single_agent"


def test_reasoning_catalog_exposes_tot_and_can_recommend_it_for_exploratory_cases() -> None:
    discovery = DiscoveryArtifact(
        problem_statement="Comparar multiples fuentes simultaneas para elegir la mejor estrategia de agente.",
        current_user="Arquitecto de soluciones",
        current_process="Analiza multiples fuentes, clasifica alternativas y decide la mejor ruta.",
        desired_outcome="Definir una opcion robusta despues de explorar varios caminos posibles.",
        autonomy_level="high",
        constraints=[],
        operational_baseline={
            "current_time_spent": "5 horas",
            "current_cost": "Retrabajo por decisiones ambiguas",
            "frequent_errors": ["Se cierran rutas muy pronto", "Se pierde contexto entre alternativas", "Se repite el analisis"],
            "automation_opportunities": ["Comparar alternativas", "Clasificar entradas"],
        },
        mvp_definition={
            "v1_scope": ["Discovery", "Canvas", "Blueprint", "Evaluacion", "Exportes"],
            "out_of_scope": ["Subagentes"],
            "north_star_metric": "Elegir la mejor arquitectura con evidencia",
            "non_delegable_decisions": ["Elegir la opcion final"],
        },
        case_type="copiloto",
        value_statement="Reducir ambiguedad en decisiones complejas",
    )
    canvas = CanvasArtifact(
        user_goal="Elegir la mejor arquitectura con evidencia",
        mvp_scope=["Discovery", "Canvas", "Blueprint", "Evaluacion", "Exportes"],
        out_of_scope=["Subagentes"],
        success_metric="Elegir la mejor ruta",
        primary_risk="Cerrar una opcion sin explorar alternativas",
        agent_profile={
            "mission": "Explorar alternativas antes de decidir.",
            "primary_user": "Arquitecto de soluciones",
            "agent_task": "Comparar rutas para el agente",
            "allowed_decisions": ["Recomendar alternativas"],
            "prohibited_decisions": ["Ejecutar cambios reales"],
            "key_inputs": ["Discovery"],
            "expected_outputs": ["Comparativo"],
            "human_approvals": ["Revision de la opcion final"],
            "success_metrics": ["Elegir la mejor ruta"],
        },
    )

    catalog = build_reasoning_catalog(discovery, canvas)

    assert any(item.key == "ToT" for item in catalog)
    assert select_reasoning_pattern(discovery, canvas) == "ToT"
