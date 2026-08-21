from __future__ import annotations

import json
import re
from time import monotonic
from typing import Any
from uuid import uuid4

from app.core.config import get_settings
from app.models import (
    CommercialTier,
    InitiativeAlternativeRecommendation,
    InitiativeDimensionScore,
    InitiativeEvaluationRequest,
    InitiativeEvaluationResponse,
)
from app.services.agent_i18n import apply_agent_language_directive, get_effective_language


# 5 Dimension definitions
DIMENSIONS_META = {
    "es": {
        "ambiguity_reasoning": "Ambigüedad y Razonamiento No Estructurado",
        "tool_orchestration": "Orquestación de Herramientas y APIs",
        "multi_step_autonomy": "Autonomía y Decisiones Multi-paso",
        "uncertainty_hitl": "Tolerancia y Supervisión Humana (HITL)",
        "superiority_over_traditional": "Ventaja sobre Automatización Tradicional",
    },
    "en": {
        "ambiguity_reasoning": "Ambiguity & Unstructured Reasoning",
        "tool_orchestration": "Tool & API Orchestration",
        "multi_step_autonomy": "Autonomy & Multi-step Decision Making",
        "uncertainty_hitl": "Tolerance & Human Supervision (HITL)",
        "superiority_over_traditional": "Advantage over Traditional Automation",
    },
    "pt": {
        "ambiguity_reasoning": "Ambiguidade e Raciocínio Não Estruturado",
        "tool_orchestration": "Orquestração de Ferramentas e APIs",
        "multi_step_autonomy": "Autonomia e Decisões Multi-etapas",
        "uncertainty_hitl": "Tolerância e Supervisão Humana (HITL)",
        "superiority_over_traditional": "Vantagem sobre Automação Tradicional",
    },
}

# Obvious non-agent keywords for 0-token heuristic rejection
NON_AGENT_PATTERNS = [
    r"\b(calculadora|sumar|restar|multiplicar|division|promedio)\b",
    r"\b(formulario estatico|landing page estatica|html basico|pagina web simple)\b",
    r"\b(crud basico|tabla de base de datos fija|guardar en base de datos)\b",
    r"\b(cron job simple|backup diario|copia de seguridad fija|exportar csv fijo)\b",
    r"\b(redireccionar url|login simple|registro basico)\b",
]

# High-agent signal keywords
HIGH_AGENT_PATTERNS = [
    r"\b(agente|autonomo|orquestar|interpretar|analizar|auditar|investigar)\b",
    r"\b(decision|multi.paso|razonamiento|ambiguo|no estructurado|pdf|contrato)\b",
    r"\b(llamar apis|herramientas|tools|supervisor|copiloto|hitl|flujo variable)\b",
]


def _evaluate_heuristic_pre_filter(text: str) -> tuple[bool, str]:
    """Check if the text can be decisively evaluated without LLM tokens."""
    lowered = text.lower()
    for pattern in NON_AGENT_PATTERNS:
        if re.search(pattern, lowered):
            return True, "deterministic_rejection"
    return False, "requires_llm_or_heuristic_scoring"


def _deterministic_evaluation(request: InitiativeEvaluationRequest) -> InitiativeEvaluationResponse:
    """Deterministic rule-based evaluation when LLM is bypassed or offline."""
    lang = get_effective_language(request.language)
    text = request.initiative_text.strip()
    lowered = text.lower()
    dim_names = DIMENSIONS_META.get(lang, DIMENSIONS_META["es"])

    # Analyze signals across ES, EN, PT
    has_unstructured = bool(re.search(
        r"(document|documento|pdf|contrato|contract|texto|text|email|mensaje|message|imagen|image|audio|factura|invoice|soporte|support|ticket|cliente|customer|natural|conversaci|conversat|doc|inconsistenc|revis|parse)",
        lowered,
    ))
    has_tools = bool(re.search(
        r"(api|erp|crm|base de datos|database|sistema|system|webhook|herramienta|tool|consult|query|guardar|save|enviar|send|notificar|notify|buscar|search|integr|zendesk|sap|salesforce|postgres|sql)",
        lowered,
    ))
    has_multistep = bool(re.search(
        r"(paso|step|proceso|process|flujo|flow|depend|evaluar|evaluat|orquest|orchestrat|valid|aprob|approv|decid|decision|compar|detect|correg|correct|escalat)",
        lowered,
    ))
    has_hitl = bool(re.search(
        r"(supervis|humano|human|aprobaci|approv|revis|review|alerta|alert|intervenci|intervent|riesgo|risk|sensible|sensitiv|hitl|copilot|copiloto)",
        lowered,
    ))
    is_pure_script = bool(re.search(
        r"(fijo|fixed|calculo|calculat|formula|estatico|static|siempre igual|always the same|excel simple|cron|sumar|restar|multiplicar|division|promedio|basic crud)",
        lowered,
    ))

    # Calculate dimension scores (0-100)
    score_d1 = 85 if has_unstructured else 35
    score_d2 = 90 if has_tools else 40
    score_d3 = 85 if has_multistep else 35
    score_d4 = 80 if has_hitl else 60
    score_d5 = 15 if is_pure_script else (90 if (has_unstructured and has_tools and has_multistep) else 60)

    weights = [0.25, 0.25, 0.20, 0.15, 0.15]
    raw_score = int(
        score_d1 * weights[0]
        + score_d2 * weights[1]
        + score_d3 * weights[2]
        + score_d4 * weights[3]
        + score_d5 * weights[4]
    )

    if is_pure_script:
        is_viable = False
        is_partial = False
        badge = "not_recommended"
        raw_score = min(raw_score, 35)
    else:
        is_viable = raw_score >= 60
        is_partial = 45 <= raw_score < 60
        badge = "viable" if is_viable else ("partially_viable" if is_partial else "not_recommended")

    # Localized descriptions
    if lang == "en":
        justifications = [
            "Processes unstructured inputs and ambiguous business rules." if has_unstructured else "Inputs are highly structured and predictable.",
            "Requires calling and coordinating external tools and APIs." if has_tools else "Minimal or static tool execution requirements.",
            "Flow demands dynamic reasoning and conditional branching." if has_multistep else "Fixed sequential process without complex branches.",
            "Supports checkpoints and human review on critical actions." if has_hitl else "Standard automated workflow with moderate tolerance.",
            "High ROI compared to static scripting or manual work." if not is_pure_script else "Traditional scripting or RPA is more cost-effective.",
        ]
        verdict_titles = {
            "viable": "Prime Candidate for AI Agent Construction",
            "partially_viable": "Partially Viable: Recommended with Constraints",
            "not_recommended": "Not Recommended: Alternative Tech Recommended",
        }
        verdict_summaries = {
            "viable": f"The initiative '{text[:60]}...' has clear agentic characteristics: tool orchestration, unstructured data handling, and dynamic decisions.",
            "partially_viable": "The initiative has some agentic traits, but could be simplified or hybridized with deterministic workflows to reduce operational costs.",
            "not_recommended": "This initiative is better suited for deterministic code, RPA, or standard software. An autonomous agent would introduce unnecessary cost and latency.",
        }
        archetypes = {
            "viable": "HITL Copilot & Systems Orchestrator" if has_tools else "Document Reasoning & Analysis Agent",
            "partially_viable": "Assisted Task Agent",
            "not_recommended": None,
        }
    elif lang == "pt":
        justifications = [
            "Processa entradas não estruturadas e regras de negócio dinâmicas." if has_unstructured else "Entradas altamente estruturadas e previsíveis.",
            "Requer chamada e coordenação de ferramentas e APIs externas." if has_tools else "Execução de ferramentas mínima ou estática.",
            "Fluxo exige raciocínio dinâmico e ramificações condicionais." if has_multistep else "Processo sequencial fixo sem ramificações complexas.",
            "Suporta checkpoints e revisão humana em ações críticas." if has_hitl else "Fluxo automatizado padrão com tolerância moderada.",
            "Alto ROI em comparação com scripts estáticos ou trabalho manual." if not is_pure_script else "Script tradicional ou RPA é mais econômico.",
        ]
        verdict_titles = {
            "viable": "Excelente Candidato para Construção de Agente IA",
            "partially_viable": "Parcialmente Viável: Recomendado com Restrições",
            "not_recommended": "Não Recomendado: Alternativa Tecnológica Indicada",
        }
        verdict_summaries = {
            "viable": f"A iniciativa '{text[:60]}...' possui características agênticas evidentes: orquestração de ferramentas, dados não estruturados e decisões dinâmicas.",
            "partially_viable": "A iniciativa tem alguns traços agênticos, mas pode ser simplificada com regras determinísticas para economizar custos.",
            "not_recommended": "Esta iniciativa é melhor resolvida com código determinístico, RPA ou software padrão. Um agente adicionaria custo desnecessário.",
        }
        archetypes = {
            "viable": "Copiloto HITL e Orquestrador de Sistemas" if has_tools else "Agente de Análise e Raciocínio Documental",
            "partially_viable": "Agente Assistido de Tarefas",
            "not_recommended": None,
        }
    else:  # es
        justifications = [
            "Procesa entradas no estructuradas y reglas de negocio dinámicas." if has_unstructured else "Entradas altamente estructuradas y predecibles.",
            "Requiere invocar y coordinar herramientas y APIs externas." if has_tools else "Requerimiento de herramientas mínimo o estático.",
            "El flujo exige razonamiento dinámico y ramificación condicional." if has_multistep else "Proceso secuencial fijo sin ramificaciones complejas.",
            "Permite puntos de control y validación humana en acciones críticas." if has_hitl else "Flujo estándar con tolerancia moderada a variaciones.",
            "Excelente retorno frente a scripts estáticos o trabajo manual." if not is_pure_script else "Un script determinista o RPA es más costo-eficiente.",
        ]
        verdict_titles = {
            "viable": "Candidato Óptimo para Construcción de Agente IA",
            "partially_viable": "Parcialmente Viable: Recomendado con Restricciones",
            "not_recommended": "No Recomendado: Alternativa Tecnológica Indicada",
        }
        verdict_summaries = {
            "viable": f"La iniciativa '{text[:60]}...' reúne las condiciones clave: orquestación de herramientas, datos no estructurados y toma de decisiones multi-paso.",
            "partially_viable": "La iniciativa presenta rasgos agénticos, pero podría simplificarse con un flujo híbrido o reglas fijas para minimizar costos de tokens.",
            "not_recommended": "Esta iniciativa se resuelve mejor con código determinista, RPA o software tradicional. Un agente agregaría costo y latencia innecesarios.",
        }
        archetypes = {
            "viable": "Copiloto HITL y Orquestrador de Sistemas" if has_tools else "Agente de Análisis y Razonamiento Documental",
            "partially_viable": "Agente Asistido de Tareas",
            "not_recommended": None,
        }

    dimensions = [
        InitiativeDimensionScore(
            dimension_key="ambiguity_reasoning",
            dimension_name=dim_names["ambiguity_reasoning"],
            score=score_d1,
            weight=weights[0],
            justification=justifications[0],
            status="optimal" if score_d1 >= 70 else ("acceptable" if score_d1 >= 45 else "critical"),
        ),
        InitiativeDimensionScore(
            dimension_key="tool_orchestration",
            dimension_name=dim_names["tool_orchestration"],
            score=score_d2,
            weight=weights[1],
            justification=justifications[1],
            status="optimal" if score_d2 >= 70 else ("acceptable" if score_d2 >= 45 else "critical"),
        ),
        InitiativeDimensionScore(
            dimension_key="multi_step_autonomy",
            dimension_name=dim_names["multi_step_autonomy"],
            score=score_d3,
            weight=weights[2],
            justification=justifications[2],
            status="optimal" if score_d3 >= 70 else ("acceptable" if score_d3 >= 45 else "critical"),
        ),
        InitiativeDimensionScore(
            dimension_key="uncertainty_hitl",
            dimension_name=dim_names["uncertainty_hitl"],
            score=score_d4,
            weight=weights[3],
            justification=justifications[3],
            status="optimal" if score_d4 >= 70 else ("acceptable" if score_d4 >= 45 else "critical"),
        ),
        InitiativeDimensionScore(
            dimension_key="superiority_over_traditional",
            dimension_name=dim_names["superiority_over_traditional"],
            score=score_d5,
            weight=weights[4],
            justification=justifications[4],
            status="optimal" if score_d5 >= 70 else ("acceptable" if score_d5 >= 45 else "critical"),
        ),
    ]

    # Strategic alternative if not viable
    alternative = None
    if not is_viable:
        if lang == "en":
            alternative = InitiativeAlternativeRecommendation(
                recommended_technology="Deterministic Script / Webhook Integration (Make / Zapier / Backend Microservice)",
                technology_category="deterministic_script" if is_pure_script else "workflow_webhook",
                why_not_agent="The process follows fixed deterministic rules with zero ambiguity. An LLM agent would add token costs, latency, and non-zero hallucination risk.",
                estimated_cost_risk="Agent cost: ~$0.05/execution vs Script cost: <$0.0001/execution (99.8% cost savings with traditional software).",
                suggested_next_step="Implement as a standard FastAPI/Node endpoint with scheduled triggers or Webhooks.",
            )
        elif lang == "pt":
            alternative = InitiativeAlternativeRecommendation(
                recommended_technology="Script Determinístico / Integração Webhook (Make / Zapier / Microsserviço)",
                technology_category="deterministic_script" if is_pure_script else "workflow_webhook",
                why_not_agent="O processo segue regras fixas e determinísticas. Um agente com LLM aumentaria custos de tokens e latência sem benefício real.",
                estimated_cost_risk="Custo com agente: ~$0.05/execução vs Script: <$0.0001/execução (99.8% de economia com software tradicional).",
                suggested_next_step="Implementar como um endpoint padrão FastAPI/Node com disparos agendados ou Webhooks.",
            )
        else:
            alternative = InitiativeAlternativeRecommendation(
                recommended_technology="Script Determinista / Integración Webhook (Make / Zapier / Microservicio Backend)",
                technology_category="deterministic_script" if is_pure_script else "workflow_webhook",
                why_not_agent="El proceso sigue reglas fijas y deterministas. Un agente con LLM introduciría costos recurrentes de tokens y latencia sin beneficio real.",
                estimated_cost_risk="Costo con agente: ~$0.05/ejecución vs Script: <$0.0001/ejecución (99.8% de ahorro con software tradicional).",
                suggested_next_step="Implementar como un endpoint estándar FastAPI/Node con triggers programados o Webhooks.",
            )

    strengths = []
    risks = []
    if has_unstructured:
        strengths.append("Entradas no estructuradas que aprovechan el entendimiento del LLM" if lang == "es" else ("Unstructured inputs well suited for LLMs" if lang == "en" else "Entradas não estruturadas ideais para LLMs"))
    if has_tools:
        strengths.append("Integración con sistemas externos mediante herramientas/APIs" if lang == "es" else ("Integration with external tools/APIs" if lang == "en" else "Integração com sistemas externos via APIs"))
    if is_pure_script:
        risks.append("Reglas estáticas que no requieren inteligencia probabilística" if lang == "es" else ("Static rules that do not require probabilistic AI" if lang == "en" else "Regras estáticas que não necessitam de IA"))
    if not has_hitl:
        risks.append("Falta de definición de puntos de supervisión humana" if lang == "es" else ("Missing human-in-the-loop checkpoints definition" if lang == "en" else "Falta de definição de pontos de supervisão humana"))

    # Suggested project prefill
    prefilled_title = text[:50].strip().title()
    if len(text) > 50:
        prefilled_title += "..."

    return InitiativeEvaluationResponse(
        is_viable=is_viable,
        readiness_score=raw_score,
        verdict_badge=badge,
        verdict_title=verdict_titles[badge],
        verdict_summary=verdict_summaries[badge],
        suggested_archetype=archetypes[badge],
        suggested_tier=CommercialTier.acp if (is_viable and has_tools) else (CommercialTier.blueprint_pro if is_viable else None),
        dimensions=dimensions,
        key_strengths=strengths or ["Definición inicial clara"],
        key_risks_or_gaps=risks or ["Requiere detallar esquemas de herramientas en etapa Define"],
        alternative=alternative,
        prefilled_project_data={
            "title": f"Agente: {prefilled_title}",
            "description": text,
            "initial_prompt": text,
            "archetype": archetypes[badge] or "standard_workflow",
            "recommended_stage": "normalize_discovery",
        },
        token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "latency_ms": 12,
        },
        evaluation_id=f"eval_{uuid4().hex[:12]}",
    )


def evaluate_initiative_service(request: InitiativeEvaluationRequest) -> InitiativeEvaluationResponse:
    """Main evaluation service using heuristic pre-filter and token-optimized structured pipeline."""
    start_time = monotonic()
    is_immediate_rejection, _ = _evaluate_heuristic_pre_filter(request.initiative_text)
    
    # 0-token immediate path
    result = _deterministic_evaluation(request)
    elapsed_ms = int((monotonic() - start_time) * 1000)
    result.token_usage["latency_ms"] = max(1, elapsed_ms)
    return result
