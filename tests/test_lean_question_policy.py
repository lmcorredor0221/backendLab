from __future__ import annotations

from app.services.lean_question_policy import (
    classify_stage_question,
    deferred_stage_questions,
    filter_stage_question_texts,
    sanitize_discovery_analysis_output,
)
from app.services.llm_runtime.builder_contracts import DiscoveryAnalysisOutput, PrioritizedQuestion


def test_discovery_policy_defers_later_stage_technical_questions() -> None:
    analysis = DiscoveryAnalysisOutput(
        open_questions=[
            PrioritizedQuestion(
                key="erp_integration",
                question="Que ERP especifico y metodo de integracion debe usar el agente?",
                rationale="Impacta Tools y contratos de implementacion.",
                priority="high",
                blocking_stages=["tools"],
            ),
            PrioritizedQuestion(
                key="current_owner",
                question="Quien vive hoy el problema y debe validar que el proceso actual es correcto?",
                rationale="Ayuda a cerrar el entendimiento del problema.",
                priority="medium",
                blocking_stages=["define"],
            ),
        ],
        missing_information=[
            "Herramienta de ticketing o bandeja humana para handoff.",
            "desired_outcome",
        ],
    )

    sanitized = sanitize_discovery_analysis_output(analysis)

    assert [question.key for question in sanitized.open_questions] == ["current_owner"]
    assert sanitized.missing_information == ["desired_outcome"]


def test_discovery_policy_preserves_required_form_field_questions() -> None:
    analysis = DiscoveryAnalysisOutput(
        open_questions=[
            PrioritizedQuestion(
                key="question:operational_baseline.current_cost",
                question="Confirma el dato faltante para operational baseline > current cost.",
                rationale="La omision afecta la calidad del discovery aprobado y las etapas posteriores.",
                priority="high",
                blocking_stages=["estimate"],
            )
        ],
        missing_information=["operational_baseline.current_cost"],
    )

    sanitized = sanitize_discovery_analysis_output(analysis)

    assert len(sanitized.open_questions) == 1
    assert sanitized.missing_information == ["operational_baseline.current_cost"]


def test_attention_filter_hides_historical_discovery_questions_from_future_stages() -> None:
    questions = filter_stage_question_texts(
        "discover",
        [
            "Falta informacion: ERP especifico, metodo de integracion y datos disponibles para consulta de pedidos.",
            "Cuales son las politicas de proteccion de datos aplicables para conversaciones y datos personales?",
            "Falta informacion: operational_baseline.current_time_spent",
            "Quien valida que el proceso actual representa el problema real?",
        ],
    )

    assert questions == [
        "Falta informacion: operational_baseline.current_time_spent",
        "Quien valida que el proceso actual representa el problema real?",
    ]


def test_discovery_policy_defers_business_clarifications_without_treating_them_as_missing_now() -> None:
    analysis = DiscoveryAnalysisOutput(
        open_questions=[
            PrioritizedQuestion(
                key="mvp_scope_precision",
                question="Cual es el alcance exacto del modo limitado inicial?",
                rationale="Ayuda a precisar el MVP, pero no bloquea el entendimiento actual del problema.",
                priority="medium",
                blocking_stages=["define"],
                suggested_answer="Empezar con clasificacion y propuesta de respuesta para casos repetitivos.",
            )
        ],
        missing_information=["Grupo de usuarios prioritario para la primera version."],
    )

    sanitized = sanitize_discovery_analysis_output(analysis)

    assert sanitized.open_questions == []
    assert sanitized.missing_information == []
    assert len(sanitized.deferred_resolution_items) == 2
    assert sanitized.deferred_resolution_items[0].target_stage == "define"
    assert sanitized.deferred_resolution_items[1].target_stage == "define"
    assert all(item.kind for item in sanitized.deferred_resolution_items)


def test_discovery_policy_routes_tools_memory_and_estimation_clarifications_to_the_right_stage() -> None:
    tools = classify_stage_question("discover", "Que ERP y API debe consultar el agente para resolver tickets?")
    memory = classify_stage_question("discover", "Que fuentes documentales y estrategia RAG necesita el agente?")
    estimate = classify_stage_question("discover", "Cuales son las metricas actuales de tiempo y costo por caso?")

    assert tools.status == "defer_to_next_stage"
    assert tools.deferral_target_stage == "tools"
    assert memory.status == "defer_to_next_stage"
    assert memory.deferral_target_stage == "memory"
    assert estimate.status == "defer_to_next_stage"
    assert estimate.deferral_target_stage == "estimate"


def test_define_policy_defers_implementation_stack_questions_to_acp() -> None:
    decision = classify_stage_question(
        "define",
        {
            "key": "database_stack",
            "question": "Que base de datos, framework y estrategia de despliegue se usaran en produccion?",
            "priority": "high",
        },
    )

    assert decision.status == "defer_to_acp"
    assert decision.deferral_target_stage == "acp"


def test_tools_policy_allows_tool_selection_but_defers_credentials() -> None:
    allowed = classify_stage_question(
        "tools",
        "Que herramienta obligatoria cubre la consulta del sistema de ticketing?",
    )
    deferred = classify_stage_question(
        "tools",
        "Cuales son las credenciales y endpoints finales para el despliegue?",
    )

    assert allowed.status == "allowed_now"
    assert deferred.status == "defer_to_acp"


def test_define_policy_keeps_questions_that_reference_sections_not_stages() -> None:
    question = {
        "key": "question:error-scenarios",
        "question": "Que errores o excepciones criticas debe contemplar el flujo objetivo?",
        "priority": "medium",
        "blocking": False,
        "impacted_sections": ["functional_requirements", "business_rules"],
    }

    decision = classify_stage_question("define", question)
    allowed = filter_stage_question_texts("define", [question])

    assert decision.status == "allowed_now"
    assert allowed == [question]


def test_deferred_questions_are_exportable_for_acp_follow_up() -> None:
    deferred = deferred_stage_questions(
        "design",
        [
            {
                "key": "deploy_runtime",
                "question": "Definir cloud, runtime y secrets de despliegue.",
            }
        ],
    )

    assert len(deferred) == 1
    assert "deploy_runtime" in deferred[0]["question"]
    assert deferred[0]["source_stage"] == "design"
    assert deferred[0]["target_stage"] == "acp"
    assert deferred[0]["status"] == "defer_to_acp"


def test_question_policy_rejects_unmanaged_deferral_targets() -> None:
    decision = classify_stage_question(
        "design",
        {
            "key": "phantom_stage",
            "question": "Resolver esta duda en la fase nebulosa externa.",
            "deferral_target_stage": "nebulosa_externa",
        },
    )

    assert decision.status == "reject_as_noise"
    assert decision.deferral_target_stage == ""
    assert "no gobernado" in decision.reason


def test_deferred_questions_do_not_export_unmanaged_targets() -> None:
    deferred = deferred_stage_questions(
        "design",
        [
            {
                "key": "valid_acp_question",
                "question": "Que credenciales se usaran durante la implementacion?",
                "deferral_target_stage": "ACP",
            },
            {
                "key": "invalid_stage_question",
                "question": "Enviar decision a una fase externa no definida.",
                "deferral_target_stage": "fase_magica",
            },
        ],
    )

    assert len(deferred) == 1
    assert deferred[0]["target_stage"] == "acp"
    assert "valid_acp_question" in deferred[0]["question"]
