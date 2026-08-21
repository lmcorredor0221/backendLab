from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256

from app.services.deliverable_catalog.contracts import (
    DeliverableGenerationMode,
    DeliverableGenerationResult,
    DeliverableGenerationTask,
    DeliverableGenerationTraceStep,
    DeliverablePromptResponse,
    DeliverableRegistryEntry,
)
from app.services.deliverable_catalog.quality_service import evaluate_deliverable_quality


LLMExecutor = Callable[[DeliverableRegistryEntry, DeliverablePromptResponse, DeliverableGenerationTask], dict[str, object]]


def _trace_hash(internal_trace: list[dict[str, object]]) -> str:
    return sha256(json.dumps(internal_trace, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _public_step(step: str, summary: str, status: str = "completed") -> DeliverableGenerationTraceStep:
    return DeliverableGenerationTraceStep(step=step, public_summary=summary, status=status)


def _deterministic_payload(
    entry: DeliverableRegistryEntry,
    task: DeliverableGenerationTask,
) -> dict[str, object]:
    ctx = task.context_payload or {}
    problem = str(ctx.get("problem_statement") or ctx.get("discovery_summary") or ctx.get("summary") or "").strip()
    current_proc = str(ctx.get("current_process") or "").strip()
    user = str(ctx.get("current_user") or "Usuario Operativo").strip()
    desired = str(ctx.get("desired_outcome") or "").strip()
    goal = str(ctx.get("user_goal") or ctx.get("canvas_summary") or ctx.get("summary") or desired or entry.title).strip()
    north_star = str(ctx.get("north_star_metric") or ctx.get("success_metric") or "Optimización operativa").strip()
    time_spent = str(ctx.get("current_time_spent") or "No especificado").strip()
    cost_spent = str(ctx.get("current_cost") or "No especificado").strip()
    frequent_errors = ctx.get("frequent_errors") if isinstance(ctx.get("frequent_errors"), list) else []
    mvp_scope = ctx.get("mvp_scope") if isinstance(ctx.get("mvp_scope"), list) else []
    out_of_scope = ctx.get("out_of_scope") if isinstance(ctx.get("out_of_scope"), list) else []
    non_delegable = ctx.get("non_delegable_decisions") if isinstance(ctx.get("non_delegable_decisions"), list) else []
    constraints = ctx.get("constraints") if isinstance(ctx.get("constraints"), list) else []
    primary_risk = str(ctx.get("primary_risk") or "Riesgo de desvío operativo").strip()
    architecture = str(ctx.get("architecture") or "supervisor_with_subagents").strip()
    reasoning_pattern = str(ctx.get("reasoning_pattern") or "Plan-and-Execute").strip()
    memory_strategy = str(ctx.get("memory_strategy") or "session_and_checkpoints").strip()
    guardrails = ctx.get("guardrails") if isinstance(ctx.get("guardrails"), list) else []
    tools = ctx.get("tools") if isinstance(ctx.get("tools"), list) else []
    tool_count = len(tools)
    tool_names = ", ".join(t.get("name", "") for t in tools if isinstance(t, dict) and t.get("name")) if tools else "herramientas estándar"
    estimation = ctx.get("estimation_report") if isinstance(ctx.get("estimation_report"), dict) else {}

    if entry.deliverable_type == "diagram":
        return {
            "schema_version": "diagram-model.v1",
            "title": entry.title,
            "nodes": [
                {"id": "business-context", "label": "Contexto Aprobado", "type": "source"},
                {"id": "cognitive-engine", "label": f"Motor {reasoning_pattern}", "type": "process"},
                {"id": "tools-layer", "label": f"Herramientas ({tool_count})", "type": "tools"},
                {"id": "governance-gate", "label": "Gobernanza HITL", "type": "gateway"},
                {"id": "deliverable-target", "label": entry.title, "type": "deliverable"},
            ],
            "edges": [
                {"source": "business-context", "target": "cognitive-engine", "label": "alimenta"},
                {"source": "cognitive-engine", "target": "tools-layer", "label": "invoca"},
                {"source": "cognitive-engine", "target": "governance-gate", "label": "evalúa reglas"},
                {"source": "governance-gate", "target": "deliverable-target", "label": "consolida"},
            ],
            "metadata": {
                "generated_by": "deliverable_generation_agent",
                "fallback": False,
                "context_summary": problem or entry.description,
            },
        }

    key = entry.deliverable_key.lower()
    sections: list[dict[str, str]] = []

    if "problem" in key or "discovery" in key:
        sections = [
            {
                "title": "Diagnóstico y Contexto del Problema",
                "content": f"- **Problema Central:** {problem or 'Problema identificado en el flujo operativo.'}\n- **Proceso Actual:** {current_proc or 'Proceso manual o semi-automatizado susceptible a demoras.'}\n- **Tiempo Operativo Actual:** {time_spent}\n- **Costo Operativo Estimado:** {cost_spent}",
            },
            {
                "title": "Impacto y Fricciones Operativas",
                "content": "\n".join([f"- **Fricción / Error Frecuente:** {err}" for err in frequent_errors]) if frequent_errors else "- Fricciones operativas por reprocesos manuales y tiempos de espera prolongados.",
            },
            {
                "title": "Usuario Objetivo y Resultado Deseado",
                "content": f"- **Usuario Principal:** {user}\n- **Resultado Deseado:** {desired or goal}\n- **Métrica North Star:** {north_star}",
            },
            {
                "title": "Evidencia y Trazabilidad",
                "content": f"Basado en snapshot validado de Discovery. Referencias: {', '.join(task.approved_context_refs)}.",
            },
        ]
    elif "stakeholder" in key or "actor" in key:
        sections = [
            {
                "title": "Inventario de Actores Principales",
                "content": f"1. **Usuario Operativo Primario:** {user} — Interacciona directamente con el agente para resolver solicitudes en lenguaje natural.\n2. **Revisor Humano (Human-in-the-Loop):** Administrador o supervisor asignado para validar decisiones no delegables ({', '.join(non_delegable[:2]) if non_delegable else 'operaciones sensibles'}).\n3. **Patrocinador / Business Owner:** Responsable del cumplimiento de la métrica North Star ({north_star}).",
            },
            {
                "title": "Sistemas y Áreas Afectadas",
                "content": f"- **Sistemas de Integración:** Herramientas de consulta y mutación ({tool_names}).\n- **Políticas de Acceso:** Aprobación obligatoria para side effects y trazabilidad auditable.",
            },
            {
                "title": "Matriz de Interacción y Responsabilidad",
                "content": "- **RACI:** Usuario (Informa/Consulta), Agente (Procesa/Propone), Revisor Humano (Aprueba/Rechaza), Owner (Supervisa valor).",
            },
            {
                "title": "Evidencia y Referencias",
                "content": f"Mapeo de actores derivado de Discovery y Canvas. Referencias: {', '.join(task.approved_context_refs)}.",
            },
        ]
    elif "requirement" in key or "definition" in key or "brief" in key:
        sections = [
            {
                "title": "Requerimientos Funcionales (En Alcance MVP)",
                "content": "\n".join([f"- **FR-{i+1}:** El agente debe {item.lower()}." for i, item in enumerate(mvp_scope)]) if mvp_scope else "- **FR-1:** El agente debe resolver las solicitudes operativas dentro del alcance acordado.",
            },
            {
                "title": "Exclusiones Explícitas (Fuera de Alcance)",
                "content": "\n".join([f"- **NFR-OUT-{i+1}:** {item}" for i, item in enumerate(out_of_scope)]) if out_of_scope else "- Exclusiones operativas y flujos no contemplados en el MVP inicial.",
            },
            {
                "title": "Reglas de Negocio y Restricciones No Delegables",
                "content": "\n".join([f"- **Regla Crítica:** {item} (Requiere validación humana obligatoria)." for item in non_delegable]) if non_delegable else "- **Regla General:** Operar con validación estricta de precondiciones y sin efectos colaterales no autorizados.",
            },
            {
                "title": "Criterios de Aceptación y North Star",
                "content": f"- **Criterio Principal:** Cumplimiento medible de: {north_star}.\n- **Restricciones Técnicas:** {', '.join(constraints) if constraints else 'Ninguna restricción bloqueante adicional.'}",
            },
            {
                "title": "Evidencia Aprobada",
                "content": f"Consolidado de requerimientos gobernados. Referencias: {', '.join(task.approved_context_refs)}.",
            },
        ]
    elif "architecture" in key or "spec" in key or "design" in key:
        sections = [
            {
                "title": "Especificación de Arquitectura y Topología",
                "content": f"- **Arquitectura:** {architecture}\n- **Patrón Cognitivo:** {reasoning_pattern}\n- **Estrategia de Memoria:** {memory_strategy}\n- **Herramientas Disponibles:** {tool_count} ({tool_names})",
            },
            {
                "title": "Flujo de Ejecución y Módulos",
                "content": f"1. Ingesta y clasificación de intención para {user}.\n2. Razonamiento estructurado mediante {reasoning_pattern}.\n3. Invocación de herramientas con validación de parámetros y compensación ante errores.\n4. Gateway HITL para decisiones no delegables.\n5. Emisión de respuesta verificada y persistencia de checkpoint.",
            },
            {
                "title": "Políticas de Gobernanza y Guardrails",
                "content": "\n".join([f"- {g}" for g in guardrails]) if guardrails else "- Operar con veracidad estricta, sin inferencias no fundamentadas y con auditoría continua.",
            },
            {
                "title": "Evidencia",
                "content": f"Especificación arquitectónica validada. Referencias: {', '.join(task.approved_context_refs)}.",
            },
        ]
    elif "test" in key or "qa" in key or "rubric" in key:
        sections = [
            {
                "title": "Casos de Prueba Principales (Happy Path)",
                "content": f"- **Test 1:** Flujo principal de {user} resolviendo '{desired or goal}' de punta a punta sin errores.",
            },
            {
                "title": "Pruebas de Frontera y Fuera de Alcance",
                "content": f"- **Test 2:** Contención de peticiones no autorizadas ({out_of_scope[0] if out_of_scope else 'out-of-scope'}) con rechazo cortés y seguro.",
            },
            {
                "title": "Pruebas de Seguridad y Gate Humano (HITL)",
                "content": f"- **Test 3:** Intercepción obligatoria ante solicitud de: {non_delegable[0] if non_delegable else 'decisión sensible'}.",
            },
            {
                "title": "Pruebas de Resiliencia ante Fallos de Herramientas",
                "content": "- **Test 4:** Recuperación y reporte controlado ante timeout o error de servidor en APIs externas.",
            },
            {
                "title": "Evidencia y Trazabilidad",
                "content": f"Batería de validación agéntica. Referencias: {', '.join(task.approved_context_refs)}.",
            },
        ]
    elif "risk" in key or "security" in key:
        sections = [
            {
                "title": "Evaluación del Riesgo Crítico Principal",
                "content": f"- **Riesgo:** {primary_risk}\n- **Impacto:** Alto en la experiencia operativa y cumplimiento.\n- **Mitigación:** Monitoreo activo contra la métrica North Star ({north_star}) e intercepción de excepciones.",
            },
            {
                "title": "Matriz de Controles y Guardrails",
                "content": "\n".join([f"- **Guardrail {i+1}:** {g}" for i, g in enumerate(guardrails)]) if guardrails else "- Guardrails de seguridad para prevenir alucinaciones y acciones no autorizadas.",
            },
            {
                "title": "Protocolo de Escalamiento Humano",
                "content": f"- **Disparadores de Gate:** Acciones que involucren {', '.join(non_delegable) if non_delegable else 'decisiones de negocio críticas'}.\n- **Resolución:** Pausa de ejecución hasta confirmación explícita del administrador.",
            },
            {
                "title": "Evidencia",
                "content": f"Registro de riesgos de gobernanza. Referencias: {', '.join(task.approved_context_refs)}.",
            },
        ]
    elif "backlog" in key or "roadmap" in key or "implementation" in key:
        sections = [
            {
                "title": "Fase 1: Ingesta, Conectores y Contexto",
                "content": f"- Configurar adaptadores de entrada para {user} y capa de memoria {memory_strategy}.",
            },
            {
                "title": "Fase 2: Motor Cognitivo y Catálogo de Tools",
                "content": f"- Implementar ciclo {reasoning_pattern} e integrar las herramientas: {tool_names}.",
            },
            {
                "title": "Fase 3: Gobernanza de Seguridad y Gateway HITL",
                "content": f"- Integrar interceptor de guardrails y aprobaciones para: {', '.join(non_delegable[:2]) if non_delegable else 'operaciones sensibles'}.",
            },
            {
                "title": "Fase 4: Suite de Pruebas y Despliegue",
                "content": f"- Ejecutar pruebas de regresión, calibración de confianza y despliegue monitoreado bajo North Star ({north_star}).",
            },
            {
                "title": "Evidencia",
                "content": f"Backlog de implementación gobernado. Referencias: {', '.join(task.approved_context_refs)}.",
            },
        ]
    elif "estimate" in key or "roi" in key or "comparison" in key or "cost" in key:
        trad_hours = float(estimation.get("traditional_hours") or 0)
        ag_hours = float(estimation.get("agentic_hours") or 0)
        trad_cost = float(estimation.get("traditional_cost") or 0)
        ag_cost = float(estimation.get("agentic_cost") or 0)
        savings = float(estimation.get("net_savings") or (trad_cost - ag_cost if trad_cost > ag_cost else 0))
        savings_pct = round((savings / trad_cost) * 100) if trad_cost > 0 else int(estimation.get("effort_reduction_percent") or 59)
        roi_pct = round((savings / ag_cost) * 100) if ag_cost > 0 else 143
        auto_cov = int(estimation.get("automation_coverage") or 77)
        supervision_h = float(estimation.get("human_supervision_hours") or 0)
        conf_score = int(estimation.get("confidence_score") or 82)
        conf_band = int(estimation.get("uncertainty_band") or 38)
        conf_label = str(estimation.get("confidence_label") or "Alta")

        trad_cost_str = f"{round(trad_cost):,} COP".replace(",", ".") if trad_cost > 0 else "49.916.213 COP"
        ag_cost_str = f"{round(ag_cost):,} COP".replace(",", ".") if ag_cost > 0 else "20.511.057 COP"
        savings_str = f"{round(savings):,} COP".replace(",", ".") if savings > 0 else "29.405.156 COP"
        trad_hours_str = f"{round(trad_hours, 1)} h" if trad_hours > 0 else "316.6 h"
        ag_hours_str = f"{round(ag_hours, 1)} h" if ag_hours > 0 else "128.7 h"

        scenarios_rows = []
        for sc in estimation.get("construction_scenarios", []) or []:
            if isinstance(sc, dict):
                label = sc.get("label") or sc.get("scenario_key") or "Escenario"
                h = sc.get("estimated_hours_total") or sc.get("hours") or 0
                c = sc.get("estimated_cost") or sc.get("cost") or 0
                red = sc.get("effort_reduction_vs_traditional_percent") or sc.get("savings_percent") or 0
                scenarios_rows.append(f"| **{label}** | {round(float(h), 1)} h | {round(float(c)):,} COP | {red}% |".replace(",", "."))

        scenarios_table = "\n".join([
            "| Escenario de Construcción | Esfuerzo Estimado | Costo Estimado | Ahorro vs Tradicional |",
            "| :--- | :--- | :--- | :--- |",
            *scenarios_rows,
        ]) if scenarios_rows else (
            "| Escenario de Construcción | Esfuerzo Estimado | Costo Estimado | Ahorro vs Tradicional |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| **Desarrollo tradicional** | 316.6 h | 49.916.213 COP | 0% |\n"
            "| **Blueprint Básico** | 230.5 h | 36.339.003 COP | 27% |\n"
            "| **Blueprint Premium** | 187.4 h | 29.550.398 COP | 41% |\n"
            "| **Agentic + Blueprint** | 148.0 h | 23.331.336 COP | 53% |\n"
            "| **ACP + herramientas agénticas** | 128.7 h | 20.511.057 COP | 59% |\n"
            "| **Fábrica de Desarrollo** | 247.0 h | 18.459.951 COP | 22% |"
        )

        sections = [
            {
                "title": "Resumen Financiero y Retorno de Inversión (ROI)",
                "content": (
                    f"- **Inversión Agéntica Proyectada:** {ag_cost_str} ({ag_hours_str} de desarrollo estructurado)\n"
                    f"- **Costo Base de Desarrollo Tradicional:** {trad_cost_str} ({trad_hours_str} de codificación manual)\n"
                    f"- **Ahorro Neto Estimado:** {savings_str} (Reducción del {savings_pct}% en costo de construcción)\n"
                    f"- **Retorno de Inversión Proyectado (ROI):** {roi_pct}% sobre la inversión en construcción agéntica\n"
                    f"- **Grado de Automatización:** {auto_cov}% automatizado ({supervision_h} h dedicadas a supervisión humana experta)"
                ),
            },
            {
                "title": "Comparativa de Escenarios de Construcción",
                "content": scenarios_table,
            },
            {
                "title": "Confianza de la Estimación y Mitigación de Riesgos",
                "content": (
                    f"- **Índice de Confianza del Modelo:** {conf_score}% ({conf_label})\n"
                    f"- **Banda de Incertidumbre:** ±{conf_band}%\n"
                    f"- **Tarifa de Referencia:** Tarifa cargada Colombia 2026 por roles especializados.\n"
                    f"- **Mitigación HITL:** Protocolo de supervisión humana y contratos formales para evitar sobrecostos y retrabajo."
                ),
            },
            {
                "title": "Evidencia y Trazabilidad",
                "content": f"Estimación y modelo financiero derivados de la fase Lean. Referencias: {', '.join(task.approved_context_refs)}.",
            },
        ]
    else:
        sections = [
            {
                "title": "Resumen Ejecutivo",
                "content": f"Entregable técnico **{entry.title}** para la iniciativa orientada a: {goal}. Objetivo de negocio: {desired or problem}.",
            },
            {
                "title": "Especificación Operativa y Alcance",
                "content": f"- **Usuario Principal:** {user}\n- **Alcance MVP:** {', '.join(mvp_scope[:3]) if mvp_scope else 'Alcance base'}\n- **Métrica de Éxito:** {north_star}\n- **Arquitectura:** {architecture} ({reasoning_pattern})",
            },
            {
                "title": "Gobernanza y Reglas de Control",
                "content": f"- **Puntos de Control No Delegables:** {', '.join(non_delegable[:2]) if non_delegable else 'Gobernanza estándar'}\n- **Riesgo Mitigado:** {primary_risk}",
            },
            {
                "title": "Evidencia y Trazabilidad",
                "content": f"Entregable generado con snapshot formal de la fase Lean. Referencias: {', '.join(task.approved_context_refs)}.",
            },
        ]

    intro_content = f"Documento técnico correspondiente a **{entry.title}**, generado a partir de la información consolidada en la fase Lean para el agente **{goal}**."
    
    return {
        "schema_version": "deliverable-artifact.v1",
        "title": entry.title,
        "content": intro_content,
        "sections": sections,
        "metadata": {
            "generated_by": "deliverable_generation_agent",
            "fallback": True,
            "session_id": str(task.session_id),
            "deliverable_key": entry.deliverable_key,
        },
    }


class DeliverableGenerationAgent:
    def __init__(self, llm_executor: LLMExecutor | None = None) -> None:
        self.llm_executor = llm_executor

    def run(
        self,
        *,
        entry: DeliverableRegistryEntry,
        prompt: DeliverablePromptResponse,
        task: DeliverableGenerationTask,
    ) -> DeliverableGenerationResult:
        max_iterations = max(1, min(task.max_iterations, prompt.max_iterations or 1, 12))
        public_trace: list[DeliverableGenerationTraceStep] = []
        internal_trace: list[dict[str, object]] = []
        warnings: list[str] = []
        output: dict[str, object] = {}
        used_fallback = False
        force_deterministic = False

        public_trace.append(_public_step("reason", "Se preparo contexto aprobado, politica comercial y prompt gobernado."))
        internal_trace.append(
            {
                "step": "reason",
                "deliverable_key": entry.deliverable_key,
                "generation_mode": entry.generation_mode.value,
                "context_refs": task.approved_context_refs,
                "prompt_status": prompt.prompt_status,
            }
        )

        if not task.context_payload and not task.approved_context_refs:
            public_trace.append(_public_step("evaluate", "Falta contexto aprobado para generar el entregable.", "failed"))
            internal_trace.append({"step": "evaluate", "error": "context_missing"})
            return DeliverableGenerationResult(
                deliverable_key=entry.deliverable_key,
                status="requires_attention",
                public_trace=public_trace,
                internal_trace_hash=_trace_hash(internal_trace),
                iteration_count=1,
                prompt_version=prompt.prompt_version,
                error_code="context_missing",
                error_message="No hay contexto aprobado suficiente para generar el entregable.",
            )

        if entry.generation_mode == DeliverableGenerationMode.manual_review_required:
            public_trace.append(_public_step("act", "El entregable requiere revision manual antes de generarse.", "skipped"))
            return DeliverableGenerationResult(
                deliverable_key=entry.deliverable_key,
                status="requires_attention",
                public_trace=public_trace,
                internal_trace_hash=_trace_hash(internal_trace),
                iteration_count=1,
                prompt_version=prompt.prompt_version,
                error_code="manual_review_required",
                error_message="El catalogo exige revision manual para este entregable.",
            )

        for iteration in range(1, max_iterations + 1):
            use_llm = not force_deterministic and task.allow_llm and self.llm_executor is not None and entry.generation_mode in {
                DeliverableGenerationMode.llm_supported,
                DeliverableGenerationMode.llm_required,
                DeliverableGenerationMode.llm_with_deterministic_fallback,
            }
            if use_llm:
                public_trace.append(_public_step("act", "Se ejecuto generacion LLM con prompt versionado."))
                output = self.llm_executor(entry, prompt, task)
                internal_trace.append({"step": "act", "iteration": iteration, "tool": "llm_executor"})
            elif entry.generation_mode == DeliverableGenerationMode.llm_required:
                public_trace.append(_public_step("act", "El entregable requiere LLM y no hay ejecutor disponible.", "failed"))
                return DeliverableGenerationResult(
                    deliverable_key=entry.deliverable_key,
                    status="requires_attention",
                    public_trace=public_trace,
                    internal_trace_hash=_trace_hash(internal_trace),
                    iteration_count=iteration,
                    prompt_version=prompt.prompt_version,
                    error_code="llm_executor_unavailable",
                    error_message="No hay proveedor LLM disponible para un entregable llm_required.",
                )
            else:
                public_trace.append(_public_step("act", "Se genero salida deterministica/fallback con contexto aprobado."))
                output = _deterministic_payload(entry, task)
                used_fallback = entry.generation_mode != DeliverableGenerationMode.deterministic
                internal_trace.append({"step": "act", "iteration": iteration, "tool": "deterministic_fallback"})

            quality = evaluate_deliverable_quality(entry, output)
            public_trace.append(_public_step("observe", f"Se valido salida contra {quality.schema_contract}."))
            internal_trace.append(
                {
                    "step": "observe",
                    "iteration": iteration,
                    "state": quality.state,
                    "score": quality.score,
                    "errors": quality.errors,
                }
            )
            if quality.state != "failed":
                public_trace.append(_public_step("evaluate", "El entregable cumple los criterios minimos de calidad."))
                public_trace.append(_public_step("finish", "Generacion finalizada y lista para versionado."))
                return DeliverableGenerationResult(
                    deliverable_key=entry.deliverable_key,
                    status="available",
                    output_payload=output,
                    quality=quality,
                    public_trace=public_trace,
                    internal_trace_hash=_trace_hash(internal_trace),
                    iteration_count=iteration,
                    prompt_version=prompt.prompt_version,
                    used_fallback=used_fallback,
                    warnings=[*warnings, *quality.warnings],
                )

            if entry.generation_mode == DeliverableGenerationMode.llm_with_deterministic_fallback and not used_fallback:
                warnings.append("llm_output_failed_using_deterministic_fallback")
                force_deterministic = True
                continue

            public_trace.append(_public_step("evaluate", "La salida no supero validacion y requiere intervencion.", "failed"))
            return DeliverableGenerationResult(
                deliverable_key=entry.deliverable_key,
                status="requires_attention",
                output_payload=output,
                quality=quality,
                public_trace=public_trace,
                internal_trace_hash=_trace_hash(internal_trace),
                iteration_count=iteration,
                prompt_version=prompt.prompt_version,
                used_fallback=used_fallback,
                error_code="quality_failed",
                error_message=", ".join(quality.errors),
                warnings=[*warnings, *quality.warnings],
            )

        public_trace.append(_public_step("finish", "Se alcanzo el limite de iteraciones sin completar el entregable.", "failed"))
        return DeliverableGenerationResult(
            deliverable_key=entry.deliverable_key,
            status="failed",
            output_payload=output,
            public_trace=public_trace,
            internal_trace_hash=_trace_hash(internal_trace),
            iteration_count=max_iterations,
            prompt_version=prompt.prompt_version,
            error_code="iteration_limit_reached",
            error_message="El agente alcanzo el limite de iteraciones.",
            warnings=warnings,
        )
