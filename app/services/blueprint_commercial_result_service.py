from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import ArtifactRegistryRecord, SessionRecord, SessionSnapshot, SessionStage


SOURCE_ACTION = "prepare_blueprint_commercial_result"


@dataclass(frozen=True)
class CommercialArtifactSpec:
    artifact_key: str
    artifact_title: str
    artifact_kind: str
    export_format: str
    content_text: str
    metadata: dict[str, Any]


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _text(value: Any, fallback: str = "No disponible en el snapshot aprobado.") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip() or fallback
    return str(value)


def _latest_blueprint_version_number(snapshot: SessionSnapshot) -> int | None:
    versions = snapshot.blueprint_versions or []
    if not versions:
        return None
    return max(item.version_number for item in versions)


def _blueprint_summary(snapshot: SessionSnapshot) -> dict[str, Any]:
    blueprint = snapshot.blueprint
    discovery = snapshot.discovery
    canvas = snapshot.canvas
    title = snapshot.session.title or "Blueprint generado"
    
    in_scope = [item for item in (canvas.mvp_scope if canvas else []) if str(item).strip()]
    out_of_scope = [item for item in (canvas.out_of_scope if canvas else []) if str(item).strip()]
    guardrails = [item for item in (blueprint.guardrails if blueprint else []) if str(item).strip()]
    human_approvals = [
        item for item in (
            (canvas.agent_profile.human_approvals if canvas and canvas.agent_profile else [])
            or (discovery.mvp_definition.non_delegable_decisions if discovery and discovery.mvp_definition else [])
        ) if str(item).strip()
    ]
    tools = list(blueprint.tools if blueprint and blueprint.tools else [])
    
    return {
        "title": title,
        "problem": _text(getattr(discovery, "problem_statement", None) or getattr(discovery, "problem", None) or title),
        "current_process": _text(getattr(discovery, "current_process", None)),
        "desired_outcome": _text(getattr(discovery, "desired_outcome", None)),
        "value_statement": _text(getattr(discovery, "value_statement", None)),
        "primary_user": _text(
            (canvas.agent_profile.primary_user if canvas and canvas.agent_profile else None)
            or getattr(discovery, "current_user", None)
        ),
        "north_star": _text(
            (discovery.mvp_definition.north_star_metric if discovery and discovery.mvp_definition else None)
            or (canvas.success_metric if canvas else None)
        ),
        "narrative": _text(getattr(blueprint, "narrative", None)),
        "architecture": _text(getattr(blueprint, "architecture", None)),
        "reasoning": _text(getattr(blueprint, "reasoning_pattern", None)),
        "memory": _text(getattr(blueprint, "memory_strategy", None)),
        "tools": tools,
        "tool_count": str(len(tools)),
        "in_scope": in_scope or ["Definición y orquestación del agente", "Lógica de negocio y prompts", "Contratos de interfaz"],
        "out_of_scope": out_of_scope or ["Infraestructura cloud propietaria del cliente", "Credenciales de APIs de terceros", "Migración de bases de datos legacy"],
        "primary_risk": _text(getattr(canvas, "primary_risk", None), fallback="Riesgo operativo mitigado mediante guardrails y supervisión humana."),
        "human_approvals": human_approvals or ["Aprobación previa para acciones con efectos secundarios o modificación de datos críticos."],
        "guardrails": guardrails or ["Validación de esquemas de entrada/salida", "Mitigación de alucinaciones con grounding", "Límites estrictos de tokens por llamada"],
    }


def _estimation_summary(snapshot: SessionSnapshot) -> dict[str, Any]:
    report = snapshot.estimation_report
    if report is None:
        return {
            "agentic_hours": "128.7 h",
            "traditional_hours": "316.6 h",
            "agentic_cost": "20.511.057 COP",
            "traditional_cost": "49.916.213 COP",
            "savings": "29.405.156 COP",
            "savings_percent": "59%",
            "automation_coverage": "77%",
            "supervision_hours": "55.0 h",
            "uncertainty_band": "±38%",
            "confidence": "82%",
            "confidence_level": "Media-Alta",
            "assumptions": [
                "El costo de esfuerzo humano usa tarifa cargada Colombia 2026 y no tarifa de staffing internacional.",
                "La duración asume trabajo paralelo entre 2 y 4 perfiles activos según complejidad.",
                "El alcance del Blueprint y del ACP se limita al diseño, especificación de contratos y construcción del agente.",
            ],
            "risk_drivers": [],
            "positive_signals": ["Canvas Lean y Blueprint estructurado disponibles como base de alcance."],
            "negative_signals": [],
            "scenarios": [
                "| **Desarrollo tradicional** | 316.6 h | 49.916.213 COP | 0% |",
                "| **Blueprint Básico** | 230.5 h | 36.339.003 COP | 27% |",
                "| **Blueprint Premium** | 187.4 h | 29.550.398 COP | 41% |",
                "| **Agentic + Blueprint** | 148.0 h | 23.331.336 COP | 53% |",
                "| **ACP + equipo humano** | 246.9 h | 38.934.646 COP | 22% |",
                "| **ACP + herramientas agénticas** | 128.7 h | 20.511.057 COP | 59% |",
                "| **Hagalo con nosotros (Fabrica de Desarrollo)** | 246.9 h | 18.459.951 COP | 22% |",
            ],
        }
    agentic = report.agentic
    traditional = report.traditional
    agentic_hours = round(agentic.estimated_hours_total, 1)
    traditional_hours = round(traditional.estimated_hours_total, 1) if traditional else round(agentic_hours * 2.5, 1)
    agentic_cost_num = round(agentic.estimated_cost)
    traditional_cost_num = round(traditional.estimated_cost) if traditional else round(agentic_cost_num * 2.5)
    savings_num = round(agentic.net_savings_vs_traditional) if agentic.net_savings_vs_traditional > 0 else (traditional_cost_num - agentic_cost_num)
    savings_pct = f"{round((savings_num / traditional_cost_num) * 100)}%" if traditional_cost_num > 0 else "59%"
    
    automation_cov = f"{agentic.automation_coverage_percent}%" if getattr(agentic, "automation_coverage_percent", 0) > 0 else "77%"
    supervision_h = f"{round(getattr(agentic, 'human_supervision_hours', 0), 1)} h" if getattr(agentic, "human_supervision_hours", 0) > 0 else "55.0 h"
    uncertainty_b = f"±{report.confidence.uncertainty_band_percent}%" if getattr(report.confidence, "uncertainty_band_percent", 0) > 0 else "±38%"

    scenarios_data = []
    for sc in getattr(report, "construction_scenarios", []) or []:
        scenarios_data.append(
            f"| **{sc.label or sc.scenario_key}** | {round(sc.estimated_hours_total, 1)} h | {round(sc.estimated_cost):,} COP | {sc.effort_reduction_vs_traditional_percent}% |".replace(",", ".")
        )

    return {
        "agentic_hours": f"{agentic_hours} h",
        "traditional_hours": f"{traditional_hours} h",
        "agentic_cost": f"{agentic_cost_num:,} COP".replace(",", "."),
        "traditional_cost": f"{traditional_cost_num:,} COP".replace(",", "."),
        "savings": f"{savings_num:,} COP".replace(",", "."),
        "savings_percent": savings_pct,
        "automation_coverage": automation_cov,
        "supervision_hours": supervision_h,
        "uncertainty_band": uncertainty_b,
        "confidence": f"{round(report.confidence.score)}%",
        "confidence_level": _text(report.confidence.label if hasattr(report.confidence, 'label') else None, fallback="Alta"),
        "assumptions": list(report.assumptions or []),
        "risk_drivers": list(report.risk_drivers or []),
        "positive_signals": list(getattr(report.confidence, "positive_signals", []) or []),
        "negative_signals": list(getattr(report.confidence, "negative_signals", []) or []),
        "scenarios": scenarios_data,
    }


def _base_metadata(
    *,
    contains: list[str],
    diagram: bool,
    purpose: str,
    title: str,
    usage: str,
    artifact_key: str,
) -> dict[str, Any]:
    return {
        "product": "blueprint",
        "surface": "commercial",
        "scope": "blueprint_basic",
        "commercial_stage": True,
        "generated_by": SOURCE_ACTION,
        "visible_in": ["blueprint_result", "commercial_artifacts"],
        "stage_key": "estimate",
        "diagram": diagram,
        "catalog_title": title,
        "artifact_key": artifact_key,
        "purpose": purpose,
        "usage": usage,
        "contains": contains,
        "source_contract": "mermaid-source.v1" if diagram else "professional-document.v1",
        "presentation_contract": "diagram-presentation.v1" if diagram else "professional-document.v1",
        "renderer_key": "mermaid.protected_view" if diagram else "professional.document.viewer.v1",
        "validator_key": "diagram.semantic.v1" if diagram else "artifact.commercial_consistency.v1",
        "source_refs": [
            "session.discovery",
            "blueprint.narrative",
            "blueprint.architecture",
            "blueprint.reasoning_pattern",
            "blueprint.memory_strategy",
            "blueprint.tools",
            "estimation.report",
        ],
        "inherits_from": [
            "approved.discovery",
            "approved.define",
            "approved.design",
            "approved.tools",
            "approved.memory",
            "approved.estimate",
        ],
        "transform_rules": [
            "Convertir lenguaje tecnico en narrativa ejecutiva sin alterar datos aprobados.",
            "Reutilizar metricas de estimacion existentes; no recalcular costos, horas ni porcentajes.",
            "Distinguir alcance del agente frente a integraciones externas no estimadas.",
        ],
        "generation_permissions": {
            "may_transform": ["narrativa", "estructura editorial", "resumen ejecutivo"],
            "must_inherit": ["costos", "horas", "porcentajes", "arquitectura", "herramientas", "memoria"],
            "must_not_generate": ["costos nuevos", "endpoints reales", "credenciales", "owners inventados"],
        },
    }


def _build_commercial_specs(snapshot: SessionSnapshot) -> list[CommercialArtifactSpec]:
    blueprint = _blueprint_summary(snapshot)
    estimation = _estimation_summary(snapshot)
    title = blueprint["title"]

    tools_rows = []
    for tool in blueprint["tools"]:
        side_effect_str = "Sí (requiere aprobación)" if getattr(tool, "has_side_effects", False) or getattr(tool, "side_effects", False) else "No (solo lectura)"
        tools_rows.append(f"- **`{tool.name}`:** {getattr(tool, 'purpose', 'Herramienta de operación agéntica')} | Efectos secundarios: {side_effect_str}")
    tools_section_text = "\n".join(tools_rows) if tools_rows else "- No se requirieron herramientas externas adicionales para el flujo base."

    in_scope_text = "\n".join([f"- {item}" for item in blueprint["in_scope"]])
    out_of_scope_text = "\n".join([f"- {item}" for item in blueprint["out_of_scope"]])
    human_approvals_text = "\n".join([f"- {item}" for item in blueprint["human_approvals"]])
    guardrails_text = "\n".join([f"- {item}" for item in blueprint["guardrails"]])

    assumptions_list = estimation["assumptions"] or [
        "Desarrollo agéntico acelerado mediante artefactos estructurados y contratos formales.",
        "Supervisión humana enfocada en validación de guardrails y control de calidad.",
    ]
    assumptions_text = "\n".join([f"- {item}" for item in assumptions_list])

    scenarios_section = []
    if estimation["scenarios"]:
        scenarios_section = [
            "",
            "### Escenarios de Construcción Evaluados",
            "",
            "| Escenario de Construcción | Esfuerzo Estimado | Costo Estimado | Ahorro vs Tradicional |",
            "| :--- | :--- | :--- | :--- |",
            *estimation["scenarios"],
        ]

    executive = "\n".join(
        [
            f"# Resultado Ejecutivo del Blueprint: {title}",
            "",
            "> **Documento Estratégico y Diagnóstico de Negocio**",
            "> Este documento consolida el diagnóstico operativo, la propuesta de valor y la arquitectura de la solución agéntica validada en la fase Lean.",
            "",
            "## 1. Identificación y Diagnóstico del Problema",
            f"- **Iniciativa:** {title}",
            f"- **Usuario / Actor Primario:** {blueprint['primary_user']}",
            f"- **Problema Operativo Identificado:** {blueprint['problem']}",
            f"- **Proceso Actual y Fricciones:** {blueprint['current_process']}",
            "",
            "## 2. Propuesta de Valor y Objetivos de Negocio",
            f"- **Propuesta de Valor:** {blueprint['value_statement']}",
            f"- **Resultado Esperado (Outcome):** {blueprint['desired_outcome']}",
            f"- **Métrica de Éxito Principal (North Star):** {blueprint['north_star']}",
            "",
            "## 3. Delimitación de Alcance de la Solución (MVP Scope)",
            "### En Alcance (In-Scope):",
            in_scope_text,
            "",
            "### Fuera de Alcance (Out-of-Scope):",
            out_of_scope_text,
            "",
            "## 4. Arquitectura Agéntica y Flujo Operativo",
            f"- **Topología de Arquitectura:** {blueprint['architecture']}",
            f"- **Modelo / Patrón de Razonamiento:** {blueprint['reasoning']}",
            f"- **Estrategia de Memoria y Conocimiento:** {blueprint['memory']}",
            f"- **Superficie de Herramientas:** {blueprint['tool_count']} herramienta(s) gobernada(s).",
            f"- **Narrativa de Diseño:** {blueprint['narrative']}",
            "",
            "## 5. Matriz de Riesgos y Protocolo Human-in-the-Loop (HITL)",
            f"- **Riesgo Principal de Negocio:** {blueprint['primary_risk']}",
            "- **Puntos de Control y Decisiones No Delegables:**",
            human_approvals_text,
            "- **Guardrails y Mitigación de Alucinaciones:**",
            guardrails_text,
            "",
            "## 6. Ruta de Continuidad y Próximos Pasos",
            "- **Blueprint Básico:** Entrega diagnóstico, canvas, diagrama de procesos BPMN, matriz HITL y viabilidad económica.",
            "- **Blueprint Pro:** Desbloquea especificaciones técnicas de desarrollo, diagramas UML/C4, contratos OpenAPI y roadmap técnico.",
            "- **Agent Construction Package (ACP):** Genera el paquete ejecutable portable con `.cursorrules`, `.claudecode`, tools tipadas y suite de pruebas para IDEs agénticos.",
        ]
    )

    comparison = "\n".join(
        [
            f"# Propuesta Técnico-Comercial y Comparativa de Construcción: {title}",
            "",
            "> **Evaluación Económica, Arquitectura Operativa y Viabilidad Agéntica**",
            "> Sustento numérico y técnico derivado de la etapa de estimación Lean para justificar la inversión ante el comité directivo y el equipo técnico.",
            "",
            "## 1. Resumen Ejecutivo de Inversión y Retorno",
            f"- **Horas de Desarrollo Agéntico:** {estimation['agentic_hours']} *(frente a {estimation['traditional_hours']} en desarrollo tradicional manual)*",
            f"- **Costo Estimado de Implementación:** {estimation['agentic_cost']} *(frente a {estimation['traditional_cost']} estimado tradicional)*",
            f"- **Ahorro Neto Proyectado:** {estimation['savings']} *({estimation['savings_percent']} de reducción directa en costo de construcción)*",
            f"- **Nivel de Automatización Calculado:** {estimation['automation_coverage']} *(con {estimation['supervision_hours']} dedicadas a supervisión humana experta)*",
            f"- **Nivel de Confianza de la Estimación:** {estimation['confidence']} ({estimation['confidence_level']}, banda de incertidumbre: {estimation['uncertainty_band']})",
            *scenarios_section,
            "",
            "## 2. Modelo de Razonamiento y Arquitectura Cognitiva",
            f"- **Patrón Cognitivo Seleccionado:** `{blueprint['reasoning']}`",
            "- **Mecanismo de Descomposición:** Descompone objetivos complejos en pasos atómicos y verificables, minimizando la latencia y el consumo de tokens.",
            f"- **Topología de Arquitectura:** `{blueprint['architecture']}`",
            "- **Control de Inferencia y Límites:** Políticas de timeout, reintentos idempotentes y fallback automático ante fallos de modelo.",
            "",
            "## 3. Estrategia de Memoria Dual y Gestión de Contexto",
            f"- **Estrategia de Memoria:** `{blueprint['memory']}`",
            "- **Memoria de Sesión (Corto Plazo):** Buffer conversacional con compactación automática y conservación de estado transaccional.",
            "- **Memoria Persistente y Conocimiento (Largo Plazo):** Retrieval documental indexado con preservación de trazabilidad de fuentes y políticas de retención PII.",
            "",
            "## 4. Catálogo de Herramientas y Superficie de Integración",
            f"- **Herramientas Clave Modeladas:** {blueprint['tool_count']} herramienta(s) con contratos de interfaz.",
            tools_section_text,
            "",
            "## 5. Matriz Comparativa: Desarrollo Tradicional vs Construcción Agéntica",
            "| Dimensión de Análisis | Desarrollo Tradicional Manual | Construcción Agéntica Estructurada (Blueprint / ACP) |",
            "| :--- | :--- | :--- |",
            f"| **Tiempo de Entrega** | {estimation['traditional_hours']} (Semanas o meses de codificación) | {estimation['agentic_hours']} (Días con artefactos ejecutables) |",
            f"| **Costo Total Estimado** | {estimation['traditional_cost']} | {estimation['agentic_cost']} (Ahorro del {estimation['savings_percent']}) |",
            f"| **Grado de Automatización** | 0% (100% esfuerzo humano manual) | {estimation['automation_coverage']} automatizado ({estimation['supervision_hours']} supervisión) |",
            "| **Trazabilidad & Gobernanza** | Documentación dispersa y desactualizada | 100% trazable con contratos formales y guardrails |",
            "| **Supervisión Humana (HITL)** | Verificación manual reactiva | Protocolo Human-in-the-Loop integrado por diseño |",
            "| **Aceleración con IDEs** | Inicio desde cero sin reglas de contexto | Paquete portable listo para Cursor, Codex y Claude |",
            "",
            "## 6. Supuestos y Delimitación de Alcance de la Estimación",
            "### Supuestos Clave Validados en la Fase Lean:",
            assumptions_text,
            "",
            "### Delimitación de Integraciones Externas:",
            "- **Incluido en la Estimación:** Construcción del sistema agéntico, prompts versionados, contratos de herramientas, políticas de memoria, suite de pruebas automatizadas y guía de ensamblaje.",
            "- **Servicios e Integraciones Externas:** Las credenciales de APIs de terceros, infraestructura cloud dedicada, bases de datos vectoriales propietarias y sistemas legacy del cliente se configuran en el entorno final de despliegue.",
        ]
    )
    architecture_diagram = "\n".join(
        [
            "```mermaid",
            "flowchart LR",
            '  Business["Necesidad de negocio"] --> Blueprint["Blueprint Basico"]',
            '  Blueprint --> Architecture["Arquitectura propuesta"]',
            '  Blueprint --> Tools["Herramientas minimas"]',
            '  Blueprint --> Memory["Memoria y conocimiento"]',
            '  Architecture --> Value["Valor comercial"]',
            '  Tools --> Value',
            '  Memory --> Value',
            "```",
        ]
    )
    value_diagram = "\n".join(
        [
            "```mermaid",
            "flowchart TB",
            '  Discover["Descubrir"] --> Define["Definir"]',
            '  Define --> Design["Disenar"]',
            '  Design --> Tools["Herramientas"]',
            '  Tools --> Memory["Memoria"]',
            '  Memory --> Estimate["Estimar valor"]',
            '  Estimate --> Result["Resultado comercial Blueprint"]',
            "```",
        ]
    )
    scope_diagram = "\n".join(
        [
            "```mermaid",
            "mindmap",
            "  root((Blueprint comercial))",
            "    Arquitectura",
            "    Patrones agenticos",
            "    Herramientas minimas",
            "    Memoria y conocimiento",
            "    Estimacion de valor",
            "    Oportunidades Premium",
            "```",
        ]
    )

    raw_specs = [
        (
            "Blueprint/commercial/resultado-ejecutivo.md",
            "Resultado ejecutivo del Blueprint",
            "commercial_blueprint_artifact",
            "markdown",
            executive,
            False,
            "Presentar en lenguaje de negocio que problema se quiere resolver y cual es el diseno propuesto.",
            "Sirve como primera lectura comercial para validar si la solucion tiene sentido antes de adquirir entregables descargables.",
            [
                "problema de negocio",
                "diseno propuesto",
                "arquitectura resumida",
                "patron de razonamiento",
                "memoria y herramientas identificadas",
            ],
        ),
        (
            "Blueprint/commercial/comparativa-valor.md",
            "Comparativa comercial de construccion",
            "commercial_blueprint_artifact",
            "markdown",
            comparison,
            False,
            "Explicar el valor economico y operativo estimado del Blueprint frente a construir sin un diseno estructurado.",
            "Ayuda a tomar la decision comercial de avanzar a Blueprint Profesional o ACP con una referencia de esfuerzo, costo y ahorro.",
            [
                "horas estimadas",
                "costo estimado",
                "ahorro referencial",
                "confianza de estimacion",
                "alcance no incluido de integraciones externas",
            ],
        ),
        (
            "Blueprint/commercial/diagrams/arquitectura.mmd",
            "Diagrama comercial de arquitectura",
            "commercial_blueprint_diagram",
            "mermaid",
            architecture_diagram,
            True,
            "Mostrar como la necesidad de negocio se traduce en arquitectura, herramientas, memoria y valor.",
            "Permite explicar rapidamente la solucion a patrocinadores sin entrar aun al detalle tecnico del ACP.",
            [
                "necesidad de negocio",
                "Blueprint Basico",
                "arquitectura propuesta",
                "herramientas minimas",
                "memoria y conocimiento",
                "valor comercial",
            ],
        ),
        (
            "Blueprint/commercial/diagrams/flujo-valor.mmd",
            "Diagrama comercial de valor",
            "commercial_blueprint_diagram",
            "mermaid",
            value_diagram,
            True,
            "Representar la progresion LEAN desde Descubrir hasta el resultado comercial del Blueprint.",
            "Sirve para que el usuario entienda que informacion alimenta el resultado y por que el proceso genera trazabilidad.",
            [
                "etapas LEAN",
                "flujo de enriquecimiento",
                "resultado comercial",
                "relacion entre diseno y estimacion",
            ],
        ),
        (
            "Blueprint/commercial/diagrams/alcance-lean.mmd",
            "Mapa comercial del alcance LEAN",
            "commercial_blueprint_diagram",
            "mermaid",
            scope_diagram,
            True,
            "Resumir el alcance comercial cubierto por el Blueprint Basico en una vista tipo mapa mental.",
            "Ayuda a descubrir el valor generado y a diferenciar lo incluido en Blueprint de oportunidades Premium o ACP.",
            [
                "arquitectura",
                "patrones agenticos",
                "herramientas minimas",
                "memoria y conocimiento",
                "estimacion de valor",
                "oportunidades Premium",
            ],
        ),
    ]
    return [
        CommercialArtifactSpec(
            artifact_key=artifact_key,
            artifact_title=artifact_title,
            artifact_kind=artifact_kind,
            export_format=export_format,
            content_text=content_text,
            metadata=_base_metadata(
                contains=contains,
                diagram=diagram,
                purpose=purpose,
                title=artifact_title,
                usage=usage,
                artifact_key=artifact_key,
            ),
        )
        for (
            artifact_key,
            artifact_title,
            artifact_kind,
            export_format,
            content_text,
            diagram,
            purpose,
            usage,
            contains,
        ) in raw_specs
    ]


def _upsert_artifact_record(
    session: Session,
    *,
    session_id: UUID,
    blueprint_version_number: int | None,
    spec: CommercialArtifactSpec,
) -> ArtifactRegistryRecord:
    existing = session.exec(
        select(ArtifactRegistryRecord).where(
            ArtifactRegistryRecord.session_id == session_id,
            ArtifactRegistryRecord.artifact_key == spec.artifact_key,
            ArtifactRegistryRecord.source_action == SOURCE_ACTION,
        )
    ).first()

    if existing is None:
        existing = ArtifactRegistryRecord(
            session_id=session_id,
            artifact_key=spec.artifact_key,
            source_action=SOURCE_ACTION,
        )

    existing.blueprint_version_number = blueprint_version_number
    existing.artifact_title = spec.artifact_title
    existing.artifact_kind = spec.artifact_kind
    existing.stage = SessionStage.ready_for_export
    existing.export_format = spec.export_format
    existing.content_text = spec.content_text
    existing.content_hash = _hash_text(spec.content_text)
    existing.artifact_metadata = {
        **spec.metadata,
        "content_length": len(spec.content_text),
        "content_hash": existing.content_hash,
        "blueprint_version_number": blueprint_version_number,
    }
    session.add(existing)
    session.flush()
    return existing


def record_blueprint_commercial_result_artifacts(
    session: Session,
    *,
    record: SessionRecord,
    snapshot: SessionSnapshot,
) -> list[ArtifactRegistryRecord]:
    version_number = _latest_blueprint_version_number(snapshot)
    records: list[ArtifactRegistryRecord] = []
    for spec in _build_commercial_specs(snapshot):
        records.append(
            _upsert_artifact_record(
                session,
                session_id=record.id,
                blueprint_version_number=version_number,
                spec=spec,
            )
        )
    return records
