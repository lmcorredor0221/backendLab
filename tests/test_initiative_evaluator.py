from __future__ import annotations

import pytest
from app.models import InitiativeEvaluationRequest
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
