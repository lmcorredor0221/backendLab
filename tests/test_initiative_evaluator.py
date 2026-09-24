from __future__ import annotations

import pytest

from sqlmodel import SQLModel, Session, create_engine, select

from app.models import InitiativeEvaluationAttemptRecord, InitiativeEvaluationRequest
from app.services.initiative_evaluator import evaluate_initiative_service


def test_evaluate_viable_agent_initiative_spanish():
    req = InitiativeEvaluationRequest(
        initiative_text="Queremos un agente autónomo que audite facturas y contratos en PDF, verifique precios contra el ERP SAP mediante APIs y notifique inconsistencias para aprobación humana.",
        language="es",
    )
    res = evaluate_initiative_service(req)
    assert res.is_viable is True
    assert res.readiness_score >= 60
    assert res.verdict_badge == "viable"
    assert "Candidato Óptimo" in res.verdict_title
    assert res.suggested_archetype is not None
    assert len(res.dimensions) == 5
    assert res.alternative is None
    assert "Agente:" in res.prefilled_project_data.get("title", "")


def test_evaluate_non_viable_initiative_recommends_alternative():
    req = InitiativeEvaluationRequest(
        initiative_text="Una calculadora simple para sumar y restar valores fijos de una tabla de base de datos fija y exportar un CSV.",
        language="es",
    )
    res = evaluate_initiative_service(req)
    assert res.is_viable is False
    assert res.verdict_badge == "not_recommended"
    assert res.alternative is not None
    assert "Script" in res.alternative.recommended_technology or "Webhook" in res.alternative.recommended_technology
    assert res.alternative.why_not_agent != ""


def test_support_case_with_average_response_time_is_not_rejected_as_calculator():
    req = InitiativeEvaluationRequest(
        initiative_text=(
            "Recibimos muchas solicitudes repetidas de clientes por chat y email. "
            "Un agente debe leer mensajes, buscar politicas internas, detectar datos faltantes, "
            "proponer respuestas seguras, escalar reclamos sensibles a un humano y conservar "
            "evidencia de la fuente usada. Queremos reducir en 30% el tiempo promedio de primera "
            "respuesta sin aprobar reembolsos ni modificar datos de cliente automaticamente."
        ),
        language="es",
    )

    res = evaluate_initiative_service(req)

    assert res.is_viable is True
    assert res.readiness_score >= 60
    assert res.alternative is None
    assert "promedio" not in " ".join(res.key_risks_or_gaps).lower()


def test_research_case_does_not_infer_business_ui_from_erp_substring():
    req = InitiativeEvaluationRequest(
        initiative_text=(
            "Queremos un agente de investigacion que lea documentos internos y fuentes publicas, "
            "separe evidencia de inferencias, compare hallazgos contradictorios, cite fuentes y "
            "prepare un brief para que direccion valide la interpretacion antes de tomar decisiones."
        ),
        language="es",
    )

    res = evaluate_initiative_service(req)

    assert res.is_viable is True
    assert res.operational_profile is None
    assert res.suggested_archetype != "Agente operador de aplicaciones"


@pytest.mark.parametrize(
    "idea",
    [
        "Un agente de soporte que lea chats y emails de clientes, busque politicas internas, pida datos faltantes, proponga respuestas con fuente y escale reclamos sensibles a humanos.",
        "Un agente de investigacion que revise documentos internos y fuentes publicas, compare evidencia contradictoria, cite fuentes y prepare briefs para decision humana.",
        "Un agente para solicitudes internas que llegan por formulario y email, valide datos, consulte sistemas, prepare aprobaciones, actualice estados y deje evidencia auditable.",
        "Un agente de conocimiento interno que responda preguntas sobre PDFs y procedimientos aprobados, cite fuente y version, y escale cuando haya conflicto entre documentos.",
        "Un agente de BI que responda preguntas recurrentes con datos trazables, use definiciones aprobadas, explique variaciones y marque baja confianza.",
        "Un agente de marketing que convierta briefs en outlines y borradores, revise guia de marca, sugiera SEO y deje claims sensibles para aprobacion humana.",
        "Un agente de ventas que investigue cuentas, sugiera prioridad, prepare mensajes, detecte follow-ups y ayude a completar CRM sin enviar outreach automaticamente.",
        "Un agente financiero que extraiga datos de facturas, detecte inconsistencias, prepare aprobaciones, sugiera categorias y conserve evidencia sin ejecutar pagos.",
        "Un agente legal que lea contratos, extraiga clausulas, compare contra playbooks, marque riesgos y prepare resumen para revision legal humana.",
        "Un agente de RRHH que responda FAQs con fuente, prepare onboarding, resuma CVs o feedback y deje decisiones laborales finales en humanos.",
    ],
)
def test_lab_fit_use_cases_remain_viable(idea: str):
    res = evaluate_initiative_service(
        InitiativeEvaluationRequest(initiative_text=idea, language="es")
    )

    assert res.is_viable is True
    assert res.readiness_score >= 60
    assert res.alternative is None


def test_evaluate_multilingual_support_english():
    req = InitiativeEvaluationRequest(
        initiative_text="An autonomous agent to parse support tickets from Zendesk, query customer order database, make refund decisions and escalate to human if risk is high.",
        language="en",
    )
    res = evaluate_initiative_service(req)
    assert res.is_viable is True
    assert "Prime Candidate" in res.verdict_title
    assert res.dimensions[0].dimension_name == "Ambiguity & Unstructured Reasoning"


def test_evaluate_multilingual_support_portuguese():
    req = InitiativeEvaluationRequest(
        initiative_text="Um agente inteligente para analisar contratos em PDF, validar cláusulas com a API do sistema jurídico e solicitar revisão humana quando houver risco.",
        language="pt",
    )
    res = evaluate_initiative_service(req)
    assert res.is_viable is True
    assert "Candidato" in res.verdict_title
    assert res.dimensions[0].dimension_name == "Ambiguidade e Raciocínio Não Estruturado"


def test_evaluate_persists_examples_and_deduplicates_repeated_ideas():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    idea = (
        "Queremos que un agente use un portal interno para crear ordenes, consultar manuales, "
        "validar descuentos y pedir aprobacion humana cuando supere el limite."
    )

    with Session(engine) as db:
        first = evaluate_initiative_service(
            InitiativeEvaluationRequest(
                initiative_text=idea,
                language="es",
                input_type="example",
                example_id="business-portal",
            ),
            db=db,
        )
        second = evaluate_initiative_service(
            InitiativeEvaluationRequest(
                initiative_text=f"  {idea}  ",
                language="es",
                input_type="custom",
            ),
            db=db,
        )

        rows = db.exec(select(InitiativeEvaluationAttemptRecord)).all()

    assert first.is_repeat is False
    assert first.repeat_count == 1
    assert second.is_repeat is True
    assert second.repeat_count == 2
    assert second.evaluation_id == first.evaluation_id
    assert len(rows) == 1
    assert rows[0].submission_count == 2
    assert rows[0].example_submission_count == 1
    assert rows[0].custom_submission_count == 1
    assert rows[0].example_id == "business-portal"
