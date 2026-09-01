from __future__ import annotations

from uuid import UUID

from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CatalogItemSummary,
    CatalogSummaryEntry,
    CommercialQuotaWorkspaceOverrideRecord,
    FeatureFlagEntry,
    GovernancePolicyRecord,
    PlatformRole,
    PlatformRoleAssignmentRecord,
    RuntimeCatalogEntryRecord,
    RuntimeFeatureFlagRecord,
    RuntimeGovernanceScopeType,
    RuntimeSettingsAuditRecord,
    SchemaMigrationRecord,
    UserRecord,
    WorkspaceContract,
    WorkspaceRecord,
    WorkspaceRuntimeSettingsRecord,
    WorkspaceSectionEntry,
    WorkflowTemplateRecord,
    utc_now,
)
from app.services.deliverable_catalog.governance_bootstrap import seed_deliverable_governance_defaults
from app.services.stage5_service import (
    DEFAULT_GOVERNANCE_POLICIES,
    DEFAULT_WORKFLOW_TEMPLATES,
    FEATURE_FLAG_BLUEPRINT_TIER_POLICY,
    FEATURE_FLAG_DELIVERABLE_CATALOG,
    FEATURE_FLAG_DELIVERABLE_GOVERNANCE_ADMIN,
    FEATURE_FLAG_DESIGN_INTELLIGENCE,
    FEATURE_FLAG_STAGE_ANSWER_INFERENCE,
    seed_governance_policies,
    seed_workflow_templates,
)
from app.services.skill_runtime import sync_skill_catalog


MIGRATION_KEY_STAGE0 = "2026-07-06-stage0-workspace-contract"
MIGRATION_KEY_STAGE1 = "2026-07-06-stage1-blueprint-parity"
MIGRATION_KEY_STAGE2 = "2026-07-06-stage2-skill-runtime"
MIGRATION_KEY_STAGE3 = "2026-07-06-stage3-evaluation-workbench"
MIGRATION_KEY_STAGE4 = "2026-07-06-stage4-operational-modules"
MIGRATION_KEY_STAGE5 = "2026-07-06-stage5-controlled-mvp3"
MIGRATION_KEY_STAGE6 = "2026-07-08-stage6-estimation-contracts"
CATALOG_VERSION_CURRENT = "stage6.v3"
FEATURE_FLAG_PRODUCT_EXPERIENCE_V2 = "product_experience_v2"
FEATURE_FLAG_ATTENTION_V2 = "attention_v2"
FEATURE_FLAG_REACT_RUNTIME = "react_runtime_v1"
WORKSPACE_BASE_INHERITANCE_EVENT = "workspace_base_configuration_inherited"


DEFAULT_FEATURE_FLAGS = [
    {
        "key": "workspace_contract_v1",
        "enabled": True,
        "description": "Expone el contrato de workspace en snapshot y exportes.",
        "stage_hint": "stage_0",
    },
    {
        "key": FEATURE_FLAG_PRODUCT_EXPERIENCE_V2,
        "enabled": True,
        "description": (
            "Selecciona la nueva experiencia frontend completa a nivel workspace. "
            "No habilita pantallas, etapas ni componentes visuales por separado."
        ),
        "stage_hint": "workspace",
    },
    {
        "key": FEATURE_FLAG_ATTENTION_V2,
        "enabled": True,
        "description": "Expone el contrato attention.v2 en paralelo a attention.v1 durante validacion.",
        "stage_hint": "workspace",
    },
    {
        "key": FEATURE_FLAG_REACT_RUNTIME,
        "enabled": False,
        "description": "Activa el controlador ReAct nativo por workspace para pilotos controlados.",
        "stage_hint": "runtime",
    },
    {
        "key": FEATURE_FLAG_STAGE_ANSWER_INFERENCE,
        "enabled": False,
        "description": "Activa la inferencia de respuestas por etapa antes del quality gate dentro de ReAct.",
        "stage_hint": "iai148",
    },
    {
        "key": FEATURE_FLAG_BLUEPRINT_TIER_POLICY,
        "enabled": True,
        "description": "Activa la politica Basic/Premium/ACP para inferir, diferir, enriquecer y validar por producto.",
        "stage_hint": "bdg17",
    },
    {
        "key": FEATURE_FLAG_DELIVERABLE_CATALOG,
        "enabled": True,
        "description": "Activa el catalogo canonico de entregables, diagramas, artefactos y reglas de acceso.",
        "stage_hint": "bdg17",
    },
    {
        "key": FEATURE_FLAG_DELIVERABLE_GOVERNANCE_ADMIN,
        "enabled": True,
        "description": "Activa la consola admin de gobernanza de entregables, prompts, auditoria y overrides.",
        "stage_hint": "bdg17",
    },
    {
        "key": "blueprint_evolution_roadmap",
        "enabled": True,
        "description": "Activa roadmap de evolucion formal del blueprint y su cobertura estructurada.",
        "stage_hint": "stage_1",
    },
    {
        "key": "skill_runtime_v1",
        "enabled": True,
        "description": "Activa runtime real de skills del builder y trazas por skill.",
        "stage_hint": "stage_2",
    },
    {
        "key": "evaluation_datasets_v1",
        "enabled": True,
        "description": "Activa datasets, rubricas y corridas persistidas de evaluacion.",
        "stage_hint": "stage_3",
    },
    {
        "key": "active_monitoring_v1",
        "enabled": True,
        "description": "Activa metricas agregadas, alertas y monitoreo operativo real.",
        "stage_hint": "stage_4",
    },
    {
        "key": "workflow_templates_v1",
        "enabled": True,
        "description": "Activa plantillas reutilizables de workflow y su selector persistido por sesion.",
        "stage_hint": "stage_5",
    },
    {
        "key": "governance_console_v1",
        "enabled": True,
        "description": "Activa la consola minima de gobierno, handoffs y politicas de promotion.",
        "stage_hint": "stage_5",
    },
    {
        "key": "specialized_subagents_v1",
        "enabled": False,
        "description": "Activa subprocesos especializados opcionales solo bajo control explicito.",
        "stage_hint": "stage_5",
    },
    {
        "key": "multi_agent_runtime_v1",
        "enabled": False,
        "description": "Activa la orquestacion supervisor-especialistas con handoffs y aislamiento por agente.",
        "stage_hint": "stage_10",
    },
    {
        "key": "estimation_comparative_v1",
        "enabled": True,
        "description": "Habilita el motor v1 de estimacion comparativa tradicional vs agentic sobre el flujo actual.",
        "stage_hint": "stage_6",
    },
    {
        "key": "memory_hybrid_define_design_v1",
        "enabled": True,
        "description": "Activa memoria hibrida gobernada en Define y Design con contexto staged por defecto.",
        "stage_hint": "stage_m8",
    },
    {
        "key": "memory_hybrid_extended_journey_v1",
        "enabled": True,
        "description": "Extiende la memoria hibrida gobernada a Tools, Memory, Evaluate y Build.",
        "stage_hint": "stage_m8",
    },
    {
        "key": FEATURE_FLAG_DESIGN_INTELLIGENCE,
        "enabled": True,
        "description": (
            "Activa Design Intelligence v2: arquetipos, patrones, implicaciones hacia Tools/Memory "
            "y learning loop gobernado sin escritura global automatica."
        ),
        "stage_hint": "di141",
    },
    {
        "key": "tool_recommendation_llm_v1",
        "enabled": True,
        "description": "Activa la inferencia LLM de Herramientas y su promocion controlada hacia blueprint.tools.",
        "stage_hint": "stage_ht7",
    },
]


DEFAULT_CATALOGS = {
    "architectures": [
        {
            "item_key": "single_agent",
            "label": "Agente unico",
            "status": "active",
            "summary": "Topologia minima recomendada para el MVP.",
        },
        {
            "item_key": "single_agent_with_skills",
            "label": "Agente con skills",
            "status": "active",
            "summary": "Una sola interfaz con especializaciones delimitadas.",
        },
        {
            "item_key": "handoffs",
            "label": "Handoffs",
            "status": "active",
            "summary": "Etapas secuenciales con checkpoints y cambios de contexto.",
        },
        {
            "item_key": "supervisor_with_subagents",
            "label": "Orquestador + subagentes",
            "status": "active",
            "summary": "Control central para dominios separados y mayor complejidad.",
        },
        {
            "item_key": "router_parallel",
            "label": "Router paralelo",
            "status": "active",
            "summary": "Desvia el trabajo cuando hay fuentes o tareas simultaneas reales.",
        },
    ],
    "reasoning_patterns": [
        {
            "item_key": "ReAct",
            "label": "ReAct",
            "status": "active",
            "summary": "Observa, decide y ejecuta con herramientas.",
        },
        {
            "item_key": "Plan-and-Execute",
            "label": "Plan & Execute",
            "status": "active",
            "summary": "Divide flujos largos antes de actuar.",
        },
        {
            "item_key": "Reflexion",
            "label": "Reflexion",
            "status": "active",
            "summary": "Itera cuando el caso requiere autoevaluacion.",
        },
        {
            "item_key": "HTN",
            "label": "HTN",
            "status": "active",
            "summary": "Descompone procesos con una jerarquia mas formal.",
        },
        {
            "item_key": "ToT",
            "label": "Tree of Thoughts",
            "status": "active",
            "summary": "Explora varias rutas cuando el caso es ambiguo o requiere comparar alternativas.",
        },
    ],
    "builder_skills": [
        {
            "item_key": "discovery_skill",
            "label": "Discovery skill",
            "status": "active",
            "summary": "Normaliza discovery y deja evidencia estructurada lista para las siguientes etapas.",
        },
        {
            "item_key": "lean_scope_skill",
            "label": "Lean scope skill",
            "status": "active",
            "summary": "Recorta alcance MVP y construye el canvas operativo del agente.",
        },
        {
            "item_key": "architecture_selection_skill",
            "label": "Architecture selection skill",
            "status": "active",
            "summary": "Recomienda topologia y tradeoffs con trazabilidad operativa.",
        },
        {
            "item_key": "reasoning_pattern_skill",
            "label": "Reasoning pattern skill",
            "status": "active",
            "summary": "Selecciona el patron cognitivo apropiado y deja traza de fit.",
        },
        {
            "item_key": "tool_design_skill",
            "label": "Tool design skill",
            "status": "active",
            "summary": "Define contratos, validaciones y approval gates reales.",
        },
        {
            "item_key": "memory_design_skill",
            "label": "Memory design skill",
            "status": "active",
            "summary": "Protege contexto, checkpoints y goal drift con runtime real.",
        },
        {
            "item_key": "safety_skill",
            "label": "Safety skill",
            "status": "active",
            "summary": "Revisa riesgos y decisiones no delegables con evidencia persistida.",
        },
        {
            "item_key": "blueprint_generation_skill",
            "label": "Blueprint generation skill",
            "status": "active",
            "summary": "Empaqueta la salida final para implementacion como skill ejecutable.",
        },
        {
            "item_key": "evaluation_skill",
            "label": "Evaluation skill",
            "status": "active",
            "summary": "Genera la evaluacion inicial con trazas persistidas por skill.",
        },
    ],
    "workflow_templates": [
        {
            "item_key": "single_agent_linear",
            "label": "Single agent linear",
            "status": "active",
            "summary": "Flujo lineal con aprobacion opcional y handoff final.",
        },
        {
            "item_key": "approval_gate_workflow",
            "label": "Approval gate workflow",
            "status": "active",
            "summary": "Workflow durable con pausa para aprobacion humana.",
        },
        {
            "item_key": "handoff_template",
            "label": "Handoff template",
            "status": "active",
            "summary": "Plantilla activa para flujos con ownership por fase y retorno controlado.",
        },
        {
            "item_key": "subagent_escalation",
            "label": "Subagent escalation",
            "status": "active",
            "summary": "Activa subprocesos especializados solo cuando hay evidencia de necesidad.",
        },
    ],
    "roadmap_templates": [
        {
            "item_key": "single_agent_growth",
            "label": "Single agent growth",
            "status": "active",
            "summary": "Escala desde agente unico hacia skills y validacion con evidencia.",
        },
        {
            "item_key": "skills_first_growth",
            "label": "Skills first growth",
            "status": "active",
            "summary": "Prioriza runtime de skills y despues handoffs o subagentes.",
        },
        {
            "item_key": "handoff_governed_growth",
            "label": "Handoff governed growth",
            "status": "active",
            "summary": "Escala mediante checkpoints y gobierno antes de paralelizar.",
        },
    ],
    "evaluation_types": [
        {
            "item_key": "functional",
            "label": "Functional",
            "status": "active",
            "summary": "Valida el caso feliz y la utilidad primaria.",
        },
        {
            "item_key": "validation",
            "label": "Validation",
            "status": "active",
            "summary": "Revisa faltantes, schemas y bloqueos duros.",
        },
        {
            "item_key": "consistency",
            "label": "Consistency",
            "status": "active",
            "summary": "Contrasta discovery, canvas y blueprint.",
        },
        {
            "item_key": "safety",
            "label": "Safety",
            "status": "active",
            "summary": "Valida aprobaciones, riesgos y side effects.",
        },
        {
            "item_key": "delivery",
            "label": "Delivery",
            "status": "active",
            "summary": "Asegura exportes y artefactos tecnicos minimos.",
        },
        {
            "item_key": "tool_failure",
            "label": "Tool failure",
            "status": "active",
            "summary": "Simula fallos, retries y compensaciones sobre tools del agente.",
        },
        {
            "item_key": "context_recovery",
            "label": "Context recovery",
            "status": "active",
            "summary": "Valida memoria, perdida de contexto y recuperacion controlada.",
        },
    ],
    "estimation_maturity_stages": [
        {
            "item_key": "canvas",
            "label": "Canvas",
            "status": "active",
            "summary": "Primer corte comercial temprano con rango amplio y supuestos explicitos.",
            "stage_order": 1,
        },
        {
            "item_key": "blueprint",
            "label": "Blueprint",
            "status": "active",
            "summary": "Primer corte con confianza aceptable y desglose por workstream.",
            "stage_order": 2,
        },
        {
            "item_key": "ready_to_build",
            "label": "ACP ready_to_build",
            "status": "active",
            "summary": "Corte mas solido para propuesta economica formal y continuidad constructiva.",
            "stage_order": 3,
        },
    ],
    "estimation_role_rates": [
        {
            "item_key": "backend_engineer",
            "label": "Backend engineer mid",
            "status": "active",
            "summary": "Rol base para APIs, reglas, contratos y persistencia.",
            "role_key": "backend_engineer",
            "seniority": "mid_colombia_2026",
            "currency": "COP",
            "hourly_rate": 145000,
            "source_note": "Derivado de 1.16M COP/dia interno recomendado para Colombia 2026.",
        },
        {
            "item_key": "frontend_engineer",
            "label": "Frontend engineer mid",
            "status": "active",
            "summary": "Rol base para UI, UX aplicada al builder y paneles operativos.",
            "role_key": "frontend_engineer",
            "seniority": "mid_colombia_2026",
            "currency": "COP",
            "hourly_rate": 158750,
            "source_note": "Derivado de 1.27M COP/dia interno recomendado para Colombia 2026.",
        },
        {
            "item_key": "integration_engineer",
            "label": "Integration engineer mid",
            "status": "active",
            "summary": "Rol base para contratos API, side effects y compatibilidad de terceros.",
            "role_key": "integration_engineer",
            "seniority": "mid_colombia_2026",
            "currency": "COP",
            "hourly_rate": 160000,
            "source_note": "Tarifa v1 interpolada para integraciones entre backend y data engineering con foco enterprise.",
        },
        {
            "item_key": "data_engineer",
            "label": "Data engineer mid",
            "status": "active",
            "summary": "Rol base para fuentes, calidad, knowledge y pipelines de datos.",
            "role_key": "data_engineer",
            "seniority": "mid_colombia_2026",
            "currency": "COP",
            "hourly_rate": 163750,
            "source_note": "Derivado de 1.31M COP/dia interno recomendado para Colombia 2026.",
        },
        {
            "item_key": "qa_engineer",
            "label": "QA engineer mid",
            "status": "active",
            "summary": "Rol base para evaluacion, datasets, pruebas funcionales y regresion.",
            "role_key": "qa_engineer",
            "seniority": "mid_colombia_2026",
            "currency": "COP",
            "hourly_rate": 133750,
            "source_note": "Promedio v1 entre QA functional y QA automation para un flujo mixto de builder y ACP.",
        },
        {
            "item_key": "devops_engineer",
            "label": "DevOps engineer mid",
            "status": "active",
            "summary": "Rol base para deployment, runtime, observabilidad y operacion.",
            "role_key": "devops_engineer",
            "seniority": "mid_colombia_2026",
            "currency": "COP",
            "hourly_rate": 190000,
            "source_note": "Derivado de 1.52M COP/dia interno recomendado para Colombia 2026.",
        },
    ],
    "estimation_workstream_effort": [
        {
            "item_key": "backend",
            "label": "Backend",
            "status": "active",
            "summary": "Workstream base para APIs, reglas, sesiones, runtime y exportes.",
            "workstream_key": "backend",
            "default_role_keys": ["backend_engineer"],
            "bands": [
                {"complexity": "simple", "relative_weight": 0.9, "base_hours_min": 100, "base_hours_max": 140},
                {"complexity": "moderate", "relative_weight": 1.0, "base_hours_min": 140, "base_hours_max": 200},
                {"complexity": "complex", "relative_weight": 1.15, "base_hours_min": 200, "base_hours_max": 280},
                {"complexity": "critical", "relative_weight": 1.3, "base_hours_min": 260, "base_hours_max": 360},
            ],
            "notes": ["Bandas v1 calibradas con investigacion Colombia 2026 para apps transaccionales y agentic MVP."],
        },
        {
            "item_key": "frontend",
            "label": "Frontend",
            "status": "active",
            "summary": "Workstream base para dashboard, builder, ACP Preview y paneles de operacion.",
            "workstream_key": "frontend",
            "default_role_keys": ["frontend_engineer"],
            "bands": [
                {"complexity": "simple", "relative_weight": 0.85, "base_hours_min": 90, "base_hours_max": 130},
                {"complexity": "moderate", "relative_weight": 1.0, "base_hours_min": 130, "base_hours_max": 190},
                {"complexity": "complex", "relative_weight": 1.15, "base_hours_min": 180, "base_hours_max": 240},
                {"complexity": "critical", "relative_weight": 1.28, "base_hours_min": 220, "base_hours_max": 320},
            ],
            "notes": ["La UI operativa del builder tiende a crecer con blueprint, preview ACP y monitoreo."],
        },
        {
            "item_key": "integrations",
            "label": "Integrations",
            "status": "active",
            "summary": "Workstream base para APIs externas, herramientas y contratos operativos.",
            "workstream_key": "integrations",
            "default_role_keys": ["integration_engineer"],
            "bands": [
                {"complexity": "simple", "relative_weight": 0.9, "base_hours_min": 50, "base_hours_max": 90},
                {"complexity": "moderate", "relative_weight": 1.0, "base_hours_min": 70, "base_hours_max": 120},
                {"complexity": "complex", "relative_weight": 1.2, "base_hours_min": 110, "base_hours_max": 160},
                {"complexity": "critical", "relative_weight": 1.4, "base_hours_min": 150, "base_hours_max": 220},
            ],
            "notes": ["Las integraciones externas siguen siendo el principal multiplicador de incertidumbre en etapa temprana."],
        },
        {
            "item_key": "data",
            "label": "Data",
            "status": "active",
            "summary": "Workstream base para knowledge, fuentes, calidad y retrieval.",
            "workstream_key": "data",
            "default_role_keys": ["data_engineer"],
            "bands": [
                {"complexity": "simple", "relative_weight": 0.82, "base_hours_min": 40, "base_hours_max": 70},
                {"complexity": "moderate", "relative_weight": 1.0, "base_hours_min": 60, "base_hours_max": 100},
                {"complexity": "complex", "relative_weight": 1.18, "base_hours_min": 95, "base_hours_max": 145},
                {"complexity": "critical", "relative_weight": 1.35, "base_hours_min": 130, "base_hours_max": 190},
            ],
            "notes": ["Knowledge, retrieval y ownership pesan mas en soluciones con memoria o RAG."],
        },
        {
            "item_key": "qa",
            "label": "QA",
            "status": "active",
            "summary": "Workstream base para evaluacion, casos de prueba y regresion.",
            "workstream_key": "qa",
            "default_role_keys": ["qa_engineer"],
            "bands": [
                {"complexity": "simple", "relative_weight": 0.88, "base_hours_min": 70, "base_hours_max": 90},
                {"complexity": "moderate", "relative_weight": 1.0, "base_hours_min": 90, "base_hours_max": 120},
                {"complexity": "complex", "relative_weight": 1.16, "base_hours_min": 120, "base_hours_max": 150},
                {"complexity": "critical", "relative_weight": 1.32, "base_hours_min": 170, "base_hours_max": 210},
            ],
            "notes": ["Incluye un piso minimo para datasets, rubricas, regresion y validacion manual."],
        },
        {
            "item_key": "devops",
            "label": "DevOps",
            "status": "active",
            "summary": "Workstream base para deployment, observabilidad y hardening operativo.",
            "workstream_key": "devops",
            "default_role_keys": ["devops_engineer"],
            "bands": [
                {"complexity": "simple", "relative_weight": 0.86, "base_hours_min": 50, "base_hours_max": 70},
                {"complexity": "moderate", "relative_weight": 1.0, "base_hours_min": 70, "base_hours_max": 100},
                {"complexity": "complex", "relative_weight": 1.18, "base_hours_min": 90, "base_hours_max": 120},
                {"complexity": "critical", "relative_weight": 1.35, "base_hours_min": 120, "base_hours_max": 150},
            ],
            "notes": ["Deployment target, secret management y observabilidad siguen siendo drivers fuertes de incertidumbre."],
        },
    ],
    "estimation_automation_matrix": [
        {
            "item_key": "discovery_canvas",
            "label": "Discovery / Canvas",
            "status": "active",
            "summary": "Entregables estructurados y narrativos de descubrimiento temprano.",
            "family_key": "discovery_canvas",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 85, "automation_ceiling_percent": 90, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "moderate", "base_automation_percent": 75, "automation_ceiling_percent": 80, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "complex", "base_automation_percent": 60, "automation_ceiling_percent": 65, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "critical", "base_automation_percent": 45, "automation_ceiling_percent": 50, "mandatory_human_review": True, "risk_tier": "medium"},
            ],
            "blocking_conditions": [],
            "penalty_rules": [],
            "bonus_rules": [],
            "notes": ["La captura es alta siempre que el contexto de negocio este disponible."],
        },
        {
            "item_key": "prd_narrative",
            "label": "PRD / narrativa funcional",
            "status": "active",
            "summary": "Definicion funcional y narrativa de alcance.",
            "family_key": "prd_narrative",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 75, "automation_ceiling_percent": 80, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "moderate", "base_automation_percent": 65, "automation_ceiling_percent": 70, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "complex", "base_automation_percent": 50, "automation_ceiling_percent": 55, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "critical", "base_automation_percent": 35, "automation_ceiling_percent": 40, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": [],
            "penalty_rules": [],
            "bonus_rules": [],
            "notes": ["Siempre requiere validacion humana de negocio antes de comprometer alcance."],
        },
        {
            "item_key": "architecture_spec",
            "label": "Arquitectura / technical spec",
            "status": "active",
            "summary": "Especificaciones tecnicas, topologia y decisiones de arquitectura.",
            "family_key": "architecture_spec",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 70, "automation_ceiling_percent": 75, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "moderate", "base_automation_percent": 60, "automation_ceiling_percent": 65, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "complex", "base_automation_percent": 45, "automation_ceiling_percent": 50, "mandatory_human_review": True, "risk_tier": "high"},
                {"complexity": "critical", "base_automation_percent": 30, "automation_ceiling_percent": 35, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": [],
            "penalty_rules": [],
            "bonus_rules": [
                {
                    "rule_key": "acp_ready_to_build",
                    "label": "Decisiones operativas cerradas en ACP",
                    "delta_percent": 4,
                    "rationale": "Una arquitectura con continuity package listo reduce ambiguedad tecnica.",
                }
            ],
            "notes": ["La cobertura automatizable baja cuando faltan restricciones operativas del entorno."],
        },
        {
            "item_key": "prompts_playbooks",
            "label": "Prompts / playbooks",
            "status": "active",
            "summary": "Prompts estructurados, playbooks operativos y skills base.",
            "family_key": "prompts_playbooks",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 85, "automation_ceiling_percent": 90, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "moderate", "base_automation_percent": 75, "automation_ceiling_percent": 80, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "complex", "base_automation_percent": 60, "automation_ceiling_percent": 65, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "critical", "base_automation_percent": 45, "automation_ceiling_percent": 50, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": [],
            "penalty_rules": [],
            "bonus_rules": [
                {
                    "rule_key": "evaluation_complete",
                    "label": "Playbooks ya contrastados con evaluacion",
                    "delta_percent": 4,
                    "rationale": "Los prompts y playbooks mejoran cuando ya existe evidencia de evaluacion.",
                }
            ],
            "notes": ["Alta automatizacion mientras el comportamiento esperado ya este definido."],
        },
        {
            "item_key": "tool_schemas",
            "label": "Tool schemas / contratos internos",
            "status": "active",
            "summary": "Contratos internos de herramientas, entradas, salidas y validaciones.",
            "family_key": "tool_schemas",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 80, "automation_ceiling_percent": 85, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "moderate", "base_automation_percent": 70, "automation_ceiling_percent": 75, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "complex", "base_automation_percent": 55, "automation_ceiling_percent": 60, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "critical", "base_automation_percent": 40, "automation_ceiling_percent": 45, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": ["tool_side_effects_not_governed"],
            "penalty_rules": [
                {
                    "rule_key": "tool_side_effects_not_governed",
                    "label": "Hay side effects todavia no gobernados",
                    "delta_percent": 6,
                    "rationale": "Los tools con efectos irreversibles reducen la cobertura sin aprobaciones claras.",
                }
            ],
            "bonus_rules": [
                {
                    "rule_key": "schema_complete",
                    "label": "Todos los tools tienen esquema validable",
                    "delta_percent": 5,
                    "rationale": "Los contratos con schema completo aumentan la capacidad de automatizacion confiable.",
                }
            ],
            "notes": ["La cobertura cae con tools irreversibles o de alto impacto operacional."],
        },
        {
            "item_key": "external_api_contracts",
            "label": "Contratos API externos",
            "status": "active",
            "summary": "Contratos de terceros aun no cerrados o con documentacion parcial.",
            "family_key": "external_api_contracts",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 45, "automation_ceiling_percent": 50, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "moderate", "base_automation_percent": 35, "automation_ceiling_percent": 40, "mandatory_human_review": True, "risk_tier": "high"},
                {"complexity": "complex", "base_automation_percent": 25, "automation_ceiling_percent": 30, "mandatory_human_review": True, "risk_tier": "high"},
                {"complexity": "critical", "base_automation_percent": 15, "automation_ceiling_percent": 20, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": ["api_contract_missing", "sandbox_unknown"],
            "penalty_rules": [
                {
                    "rule_key": "api_contract_missing",
                    "label": "Contratos API de terceros aun abiertos",
                    "delta_percent": 10,
                    "rationale": "Sin contratos verificables no conviene automatizar integraciones complejas.",
                },
                {
                    "rule_key": "sandbox_unknown",
                    "label": "Sandbox o ambiente de prueba aun no confirmado",
                    "delta_percent": 6,
                    "rationale": "La falta de sandbox eleva el riesgo de integracion y retrabajo.",
                },
                {
                    "rule_key": "production_side_effects",
                    "label": "Integraciones con side effects productivos",
                    "delta_percent": 5,
                    "rationale": "Las integraciones activas requieren mas control humano y menor delegacion.",
                },
            ],
            "bonus_rules": [
                {
                    "rule_key": "schema_complete",
                    "label": "Payloads internos ya estan estructurados",
                    "delta_percent": 4,
                    "rationale": "Los contratos internos completos facilitan traducirlos a integraciones externas.",
                }
            ],
            "notes": ["La madurez contractual externa es uno de los mayores frenos de automatizacion."],
        },
        {
            "item_key": "evaluation_assets",
            "label": "Evaluation assets",
            "status": "active",
            "summary": "Datasets, rubricas, casos de prueba y scaffolds de evaluacion.",
            "family_key": "evaluation_assets",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 80, "automation_ceiling_percent": 85, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "moderate", "base_automation_percent": 70, "automation_ceiling_percent": 75, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "complex", "base_automation_percent": 55, "automation_ceiling_percent": 60, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "critical", "base_automation_percent": 40, "automation_ceiling_percent": 45, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": [],
            "penalty_rules": [],
            "bonus_rules": [
                {
                    "rule_key": "evaluation_complete",
                    "label": "Dataset y corrida de evaluacion completos",
                    "delta_percent": 6,
                    "rationale": "QA automatizable mejora cuando ya existe evidencia formal de calidad.",
                }
            ],
            "notes": ["El review humano aumenta cuando el dominio tiene impacto regulatorio o contractual."],
        },
        {
            "item_key": "runtime_config",
            "label": "Runtime config / providers",
            "status": "active",
            "summary": "Configuracion del runtime, proveedores, modelos y dependencias base.",
            "family_key": "runtime_config",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 75, "automation_ceiling_percent": 80, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "moderate", "base_automation_percent": 65, "automation_ceiling_percent": 70, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "complex", "base_automation_percent": 50, "automation_ceiling_percent": 55, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "critical", "base_automation_percent": 35, "automation_ceiling_percent": 40, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": ["provider_or_secret_source_unknown"],
            "penalty_rules": [
                {
                    "rule_key": "provider_or_secret_source_unknown",
                    "label": "Provider, modelo o secretos aun indefinidos",
                    "delta_percent": 8,
                    "rationale": "Sin cierre del runtime real el setup automatizable sigue limitado.",
                }
            ],
            "bonus_rules": [
                {
                    "rule_key": "schema_complete",
                    "label": "Schemas internos ya normalizados",
                    "delta_percent": 4,
                    "rationale": "Los providers y dependencias se configuran mejor cuando los contratos base ya son estables.",
                },
                {
                    "rule_key": "acp_ready_to_build",
                    "label": "ACP listo para handoff tecnico",
                    "delta_percent": 4,
                    "rationale": "Un ACP listo reduce ambiguedad de runtime y dependencias.",
                },
            ],
            "notes": ["La precision depende de tener claros provider, modelo y politica de secretos."],
        },
        {
            "item_key": "deployment_infra",
            "label": "Deployment / infra",
            "status": "active",
            "summary": "Infraestructura, despliegue, ambientes, secrets y operacion.",
            "family_key": "deployment_infra",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 55, "automation_ceiling_percent": 60, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "moderate", "base_automation_percent": 45, "automation_ceiling_percent": 50, "mandatory_human_review": True, "risk_tier": "high"},
                {"complexity": "complex", "base_automation_percent": 30, "automation_ceiling_percent": 35, "mandatory_human_review": True, "risk_tier": "high"},
                {"complexity": "critical", "base_automation_percent": 15, "automation_ceiling_percent": 20, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": ["target_environment_unknown", "secret_owner_missing"],
            "penalty_rules": [
                {
                    "rule_key": "target_environment_unknown",
                    "label": "Target de deployment aun no definido",
                    "delta_percent": 10,
                    "rationale": "El despliegue no debe automatizarse sin cerrar ambiente objetivo.",
                },
                {
                    "rule_key": "secret_owner_missing",
                    "label": "Ownership de secretos sigue abierto",
                    "delta_percent": 6,
                    "rationale": "Sin dueno de secretos y runtime la automatizacion es fragil.",
                },
            ],
            "bonus_rules": [
                {
                    "rule_key": "acp_ready_to_build",
                    "label": "ACP con readiness operativo cerrado",
                    "delta_percent": 6,
                    "rationale": "El readiness operativo mejora la capacidad de automatizar deployment e infra.",
                }
            ],
            "notes": ["Deployment y seguridad operativa no deben automatizarse a ciegas."],
        },
        {
            "item_key": "knowledge_retrieval",
            "label": "Knowledge / retrieval",
            "status": "active",
            "summary": "Fuentes de conocimiento, retrieval, refresh y ownership.",
            "family_key": "knowledge_retrieval",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 60, "automation_ceiling_percent": 65, "mandatory_human_review": False, "risk_tier": "medium"},
                {"complexity": "moderate", "base_automation_percent": 50, "automation_ceiling_percent": 55, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "complex", "base_automation_percent": 35, "automation_ceiling_percent": 40, "mandatory_human_review": True, "risk_tier": "high"},
                {"complexity": "critical", "base_automation_percent": 20, "automation_ceiling_percent": 25, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": ["knowledge_owner_missing", "refresh_policy_missing"],
            "penalty_rules": [
                {
                    "rule_key": "knowledge_owner_missing",
                    "label": "Knowledge owner aun no definido",
                    "delta_percent": 8,
                    "rationale": "La base de conocimiento pierde confiabilidad sin owner claro.",
                },
                {
                    "rule_key": "refresh_policy_missing",
                    "label": "Refresh policy todavia no aterrizada",
                    "delta_percent": 6,
                    "rationale": "Sin refresh policy no conviene delegar retrieval y sync automaticos.",
                },
            ],
            "bonus_rules": [
                {
                    "rule_key": "acp_ready_to_build",
                    "label": "ACP ya bajo control operativo",
                    "delta_percent": 4,
                    "rationale": "Un ACP listo suele venir con ownership y continuidad mejor definidos.",
                }
            ],
            "notes": ["Knowledge sin owner estable degrada mucho la calidad de la estimacion."],
        },
        {
            "item_key": "observability",
            "label": "Observabilidad / alertas / tracing",
            "status": "active",
            "summary": "Senales, logs, alertas y trazabilidad operativa.",
            "family_key": "observability",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 70, "automation_ceiling_percent": 75, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "moderate", "base_automation_percent": 60, "automation_ceiling_percent": 65, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "complex", "base_automation_percent": 45, "automation_ceiling_percent": 50, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "critical", "base_automation_percent": 30, "automation_ceiling_percent": 35, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": [],
            "penalty_rules": [],
            "bonus_rules": [
                {
                    "rule_key": "evaluation_complete",
                    "label": "Observabilidad ya contrastada con evaluacion",
                    "delta_percent": 4,
                    "rationale": "Las senales de monitoreo ganan precision cuando ya existe una corrida valida de evaluacion.",
                },
                {
                    "rule_key": "acp_ready_to_build",
                    "label": "Runbook operativo listo para build",
                    "delta_percent": 4,
                    "rationale": "La cobertura observability aumenta al cerrar readiness y handoff.",
                },
            ],
            "notes": ["La observabilidad suele automatizarse bien cuando ya existe el target operativo."],
        },
        {
            "item_key": "acp_packaging",
            "label": "ACP packaging",
            "status": "active",
            "summary": "Empaquetado estructurado del ACP y entregables asociados.",
            "family_key": "acp_packaging",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 90, "automation_ceiling_percent": 95, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "moderate", "base_automation_percent": 85, "automation_ceiling_percent": 90, "mandatory_human_review": False, "risk_tier": "low"},
                {"complexity": "complex", "base_automation_percent": 75, "automation_ceiling_percent": 80, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "critical", "base_automation_percent": 60, "automation_ceiling_percent": 65, "mandatory_human_review": True, "risk_tier": "medium"},
            ],
            "blocking_conditions": [],
            "penalty_rules": [],
            "bonus_rules": [
                {
                    "rule_key": "acp_ready_to_build",
                    "label": "ACP completo y listo para continuidad",
                    "delta_percent": 5,
                    "rationale": "El packaging gana cobertura cuando la estructura ya esta validada para build.",
                }
            ],
            "notes": ["Es el entregable mas estructurado y por eso tiene la mayor cobertura potencial."],
        },
        {
            "item_key": "implementation_code",
            "label": "Codigo implementable core",
            "status": "active",
            "summary": "Implementacion ejecutable del sistema objetivo y sus cambios productivos.",
            "family_key": "implementation_code",
            "bands": [
                {"complexity": "simple", "base_automation_percent": 60, "automation_ceiling_percent": 65, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "moderate", "base_automation_percent": 50, "automation_ceiling_percent": 55, "mandatory_human_review": True, "risk_tier": "medium"},
                {"complexity": "complex", "base_automation_percent": 35, "automation_ceiling_percent": 40, "mandatory_human_review": True, "risk_tier": "high"},
                {"complexity": "critical", "base_automation_percent": 20, "automation_ceiling_percent": 25, "mandatory_human_review": True, "risk_tier": "high"},
            ],
            "blocking_conditions": ["production_side_effects", "regulated_domain_requirements"],
            "penalty_rules": [
                {
                    "rule_key": "production_side_effects",
                    "label": "Codigo con side effects productivos",
                    "delta_percent": 8,
                    "rationale": "El codigo ejecutable con impacto real requiere hardening y aprobacion humana.",
                },
                {
                    "rule_key": "regulated_domain_requirements",
                    "label": "Controles regulatorios o de alto riesgo",
                    "delta_percent": 6,
                    "rationale": "Los dominios sensibles reducen la delegacion automatica incluso con buenos artefactos.",
                },
            ],
            "bonus_rules": [
                {
                    "rule_key": "schema_complete",
                    "label": "Contratos internos ya estan formalizados",
                    "delta_percent": 4,
                    "rationale": "El codigo core se automatiza mejor si las interfaces ya estan normalizadas.",
                },
                {
                    "rule_key": "evaluation_complete",
                    "label": "QA inicial ya esta listo para regresion",
                    "delta_percent": 5,
                    "rationale": "La implementacion automatizada mejora cuando ya hay dataset y corrida confiable.",
                },
            ],
            "notes": ["El porcentaje real siempre debera contrastarse con evaluacion y controles humanos."],
        },
    ],
    "estimation_pricing_profiles": [
        {
            "item_key": "openai_structured_output",
            "label": "OpenAI structured output",
            "status": "active",
            "summary": "Perfil base para costos OpenAI fast/reasoning con cache y output.",
            "profile_key": "openai_structured_output",
            "provider": "openai",
            "model": "fast=gpt-5.4-mini | reasoning=gpt-5.5",
            "mode": "standard",
            "is_local_inference": False,
            "effective_from": "2026-07-08",
            "cop_per_usd": 4000,
            "rates": [
                {"metric_key": "fast_input_tokens_m", "label": "OpenAI fast input", "unit": "1M tokens", "amount_usd": 0.375},
                {"metric_key": "fast_cached_input_tokens_m", "label": "OpenAI fast cached input", "unit": "1M tokens", "amount_usd": 0.0375},
                {"metric_key": "fast_output_tokens_m", "label": "OpenAI fast output", "unit": "1M tokens", "amount_usd": 2.25},
                {"metric_key": "reasoning_input_tokens_m", "label": "OpenAI reasoning input", "unit": "1M tokens", "amount_usd": 5.0},
                {"metric_key": "reasoning_cached_input_tokens_m", "label": "OpenAI reasoning cached input", "unit": "1M tokens", "amount_usd": 0.5},
                {"metric_key": "reasoning_output_tokens_m", "label": "OpenAI reasoning output", "unit": "1M tokens", "amount_usd": 22.5},
                {"metric_key": "tool_calls_k", "label": "OpenAI tool call fee", "unit": "1K calls", "amount_usd": 0},
                {"metric_key": "provider_session", "label": "OpenAI provider session", "unit": "session", "amount_usd": 0},
            ],
            "notes": [
                "Semilla v1 basada en pricing oficial OpenAI API para gpt-5.4-mini y gpt-5.5.",
                "FX de reporte asumido para planeacion: 4000 COP por 1 USD.",
            ],
        },
        {
            "item_key": "deepseek_api_profile",
            "label": "DeepSeek API profile",
            "status": "active",
            "summary": "Perfil base para costos DeepSeek flash/pro con cache hit, cache miss y output.",
            "profile_key": "deepseek_api_profile",
            "provider": "deepseek",
            "model": "fast=deepseek-v4-flash | reasoning=deepseek-v4-pro",
            "mode": "api",
            "is_local_inference": False,
            "effective_from": "2026-07-08",
            "cop_per_usd": 4000,
            "rates": [
                {"metric_key": "fast_input_cache_hit_tokens_m", "label": "DeepSeek flash cache hit", "unit": "1M tokens", "amount_usd": 0.0028},
                {"metric_key": "fast_input_cache_miss_tokens_m", "label": "DeepSeek flash cache miss", "unit": "1M tokens", "amount_usd": 0.14},
                {"metric_key": "fast_output_tokens_m", "label": "DeepSeek flash output", "unit": "1M tokens", "amount_usd": 0.28},
                {"metric_key": "reasoning_input_cache_hit_tokens_m", "label": "DeepSeek pro cache hit", "unit": "1M tokens", "amount_usd": 0.003625},
                {"metric_key": "reasoning_input_cache_miss_tokens_m", "label": "DeepSeek pro cache miss", "unit": "1M tokens", "amount_usd": 0.435},
                {"metric_key": "reasoning_output_tokens_m", "label": "DeepSeek pro output", "unit": "1M tokens", "amount_usd": 0.87},
                {"metric_key": "tool_calls_k", "label": "DeepSeek tool call fee", "unit": "1K calls", "amount_usd": 0},
                {"metric_key": "provider_session", "label": "DeepSeek provider session", "unit": "session", "amount_usd": 0},
            ],
            "notes": [
                "Semilla v1 basada en pricing oficial DeepSeek API para deepseek-v4-flash y deepseek-v4-pro.",
                "FX de reporte asumido para planeacion: 4000 COP por 1 USD.",
            ],
        },
        {
            "item_key": "codex_local_hybrid",
            "label": "Codex local hybrid",
            "status": "active",
            "summary": "Perfil base para costos locales por compute, orchestration y overhead del workstation.",
            "profile_key": "codex_local_hybrid",
            "provider": "codex_local",
            "model": "command=codex | tier=local",
            "mode": "local",
            "is_local_inference": True,
            "local_cost_policy": "hybrid",
            "effective_from": "2026-07-08",
            "cop_per_usd": 4000,
            "rates": [
                {"metric_key": "compute_hour_core", "label": "Local compute core", "unit": "hour", "amount_usd": 0.85},
                {"metric_key": "tool_calls_k", "label": "Local orchestration overhead", "unit": "1K calls", "amount_usd": 0.03},
                {"metric_key": "local_session", "label": "Local session overhead", "unit": "session", "amount_usd": 0.35},
                {"metric_key": "workstation_hour_hybrid", "label": "Hybrid workstation overhead", "unit": "hour", "amount_usd": 1.25},
                {"metric_key": "workstation_hour_fully_loaded", "label": "Fully loaded workstation overhead", "unit": "hour", "amount_usd": 3.25},
            ],
            "notes": [
                "Semilla v1 para Codex local basada en heuristica interna de compute y overhead de workstation.",
                "La politica marginal_only usa solo compute core mas session minima.",
                "FX de reporte asumido para planeacion: 4000 COP por 1 USD.",
            ],
        },
    ],
    "estimation_confidence_bands": [
        {
            "item_key": "low",
            "label": "Low",
            "status": "active",
            "summary": "Score 0-39 con banda sugerida entre 45% y 60%.",
            "label_key": "low",
            "min_score": 0,
            "max_score": 39,
            "uncertainty_band_min_percent": 45,
            "uncertainty_band_max_percent": 60,
        },
        {
            "item_key": "medium_low",
            "label": "Medium low",
            "status": "active",
            "summary": "Score 40-59 con banda sugerida entre 30% y 45%.",
            "label_key": "medium_low",
            "min_score": 40,
            "max_score": 59,
            "uncertainty_band_min_percent": 30,
            "uncertainty_band_max_percent": 45,
        },
        {
            "item_key": "medium",
            "label": "Medium",
            "status": "active",
            "summary": "Score 60-74 con banda sugerida entre 20% y 30%.",
            "label_key": "medium",
            "min_score": 60,
            "max_score": 74,
            "uncertainty_band_min_percent": 20,
            "uncertainty_band_max_percent": 30,
        },
        {
            "item_key": "medium_high",
            "label": "Medium high",
            "status": "active",
            "summary": "Score 75-89 con banda sugerida entre 12% y 20%.",
            "label_key": "medium_high",
            "min_score": 75,
            "max_score": 89,
            "uncertainty_band_min_percent": 12,
            "uncertainty_band_max_percent": 20,
        },
        {
            "item_key": "high",
            "label": "High",
            "status": "active",
            "summary": "Score 90-100 con banda sugerida entre 8% y 12%.",
            "label_key": "high",
            "min_score": 90,
            "max_score": 100,
            "uncertainty_band_min_percent": 8,
            "uncertainty_band_max_percent": 12,
        },
    ],
    "estimation_confidence_weights": [
        {
            "item_key": "base_canvas",
            "label": "Base score Canvas",
            "status": "active",
            "summary": "Piso de confianza para el primer corte comercial en Canvas.",
            "metric_key": "base_canvas",
            "amount": 40,
        },
        {
            "item_key": "base_blueprint",
            "label": "Base score Blueprint",
            "status": "active",
            "summary": "Piso de confianza cuando ya existe blueprint estructurado.",
            "metric_key": "base_blueprint",
            "amount": 58,
        },
        {
            "item_key": "base_ready_to_build",
            "label": "Base score Ready to build",
            "status": "active",
            "summary": "Piso de confianza cuando el ACP ya esta listo para continuidad constructiva.",
            "metric_key": "base_ready_to_build",
            "amount": 76,
        },
        {
            "item_key": "scope_v1_scope_multiplier",
            "label": "Scope V1 multiplier",
            "status": "active",
            "summary": "Peso por item de alcance V1 declarado.",
            "metric_key": "scope_v1_scope_multiplier",
            "amount": 4,
        },
        {
            "item_key": "scope_constraint_multiplier",
            "label": "Scope constraint multiplier",
            "status": "active",
            "summary": "Peso por restriccion explicita de discovery.",
            "metric_key": "scope_constraint_multiplier",
            "amount": 2,
        },
        {
            "item_key": "scope_cap",
            "label": "Scope cap",
            "status": "active",
            "summary": "Tope de aporte del alcance al score.",
            "metric_key": "scope_cap",
            "amount": 16,
        },
        {
            "item_key": "technical_tool_multiplier",
            "label": "Technical tool multiplier",
            "status": "active",
            "summary": "Peso por tool definida en el blueprint.",
            "metric_key": "technical_tool_multiplier",
            "amount": 3,
        },
        {
            "item_key": "technical_workflow_multiplier",
            "label": "Technical workflow multiplier",
            "status": "active",
            "summary": "Peso por paso de workflow estructurado.",
            "metric_key": "technical_workflow_multiplier",
            "amount": 2,
        },
        {
            "item_key": "technical_safety_multiplier",
            "label": "Technical safety multiplier",
            "status": "active",
            "summary": "Peso por safety check activo.",
            "metric_key": "technical_safety_multiplier",
            "amount": 1,
        },
        {
            "item_key": "technical_memory_multiplier",
            "label": "Technical memory multiplier",
            "status": "active",
            "summary": "Peso por capa de memoria o retrieval definida.",
            "metric_key": "technical_memory_multiplier",
            "amount": 2,
        },
        {
            "item_key": "technical_cap",
            "label": "Technical cap",
            "status": "active",
            "summary": "Tope del bloque tecnico.",
            "metric_key": "technical_cap",
            "amount": 18,
        },
        {
            "item_key": "operations_evaluation_case_multiplier",
            "label": "Operations evaluation multiplier",
            "status": "active",
            "summary": "Peso por caso de evaluacion disponible.",
            "metric_key": "operations_evaluation_case_multiplier",
            "amount": 1,
        },
        {
            "item_key": "operations_observability_multiplier",
            "label": "Operations observability multiplier",
            "status": "active",
            "summary": "Peso por senal observability capturada.",
            "metric_key": "operations_observability_multiplier",
            "amount": 1,
        },
        {
            "item_key": "operations_ready_bonus",
            "label": "Operations ready bonus",
            "status": "active",
            "summary": "Bonus por ACP ready_to_build.",
            "metric_key": "operations_ready_bonus",
            "amount": 4,
        },
        {
            "item_key": "operations_cap",
            "label": "Operations cap",
            "status": "active",
            "summary": "Tope del bloque operativo.",
            "metric_key": "operations_cap",
            "amount": 12,
        },
        {
            "item_key": "delivery_deliverable_multiplier",
            "label": "Delivery deliverable multiplier",
            "status": "active",
            "summary": "Peso por entregable estructurado del paquete tecnico.",
            "metric_key": "delivery_deliverable_multiplier",
            "amount": 1,
        },
        {
            "item_key": "delivery_ready_bonus",
            "label": "Delivery ready bonus",
            "status": "active",
            "summary": "Bonus por readiness listo para construir.",
            "metric_key": "delivery_ready_bonus",
            "amount": 4,
        },
        {
            "item_key": "delivery_cap",
            "label": "Delivery cap",
            "status": "active",
            "summary": "Tope del bloque de entrega.",
            "metric_key": "delivery_cap",
            "amount": 10,
        },
        {
            "item_key": "readiness_complete_delta",
            "label": "Readiness complete delta",
            "status": "active",
            "summary": "Ajuste cuando el blueprint ya esta marcado como completo.",
            "metric_key": "readiness_complete_delta",
            "amount": 6,
        },
        {
            "item_key": "readiness_partial_delta",
            "label": "Readiness partial delta",
            "status": "active",
            "summary": "Ajuste neutro para readiness parcial.",
            "metric_key": "readiness_partial_delta",
            "amount": 0,
        },
        {
            "item_key": "readiness_blocked_delta",
            "label": "Readiness blocked delta",
            "status": "active",
            "summary": "Penalidad por readiness bloqueado.",
            "metric_key": "readiness_blocked_delta",
            "amount": -8,
        },
        {
            "item_key": "maturity_complete_delta",
            "label": "Maturity complete delta",
            "status": "active",
            "summary": "Bonus cuando una dimension critica ya esta cerrada.",
            "metric_key": "maturity_complete_delta",
            "amount": 5,
        },
        {
            "item_key": "maturity_partial_delta",
            "label": "Maturity partial delta",
            "status": "active",
            "summary": "Bonus marginal para una dimension parcialmente definida.",
            "metric_key": "maturity_partial_delta",
            "amount": 1,
        },
        {
            "item_key": "maturity_missing_delta",
            "label": "Maturity missing delta",
            "status": "active",
            "summary": "Penalidad para dimensiones faltantes.",
            "metric_key": "maturity_missing_delta",
            "amount": -8,
        },
        {
            "item_key": "maturity_blocked_delta",
            "label": "Maturity blocked delta",
            "status": "active",
            "summary": "Penalidad fuerte para dimensiones bloqueadas.",
            "metric_key": "maturity_blocked_delta",
            "amount": -12,
        },
        {
            "item_key": "maturity_not_applicable_delta",
            "label": "Maturity N/A delta",
            "status": "active",
            "summary": "Ajuste para dimensiones que no aplican al caso actual.",
            "metric_key": "maturity_not_applicable_delta",
            "amount": 3,
        },
        {
            "item_key": "blocking_gap_penalty",
            "label": "Blocking gap penalty",
            "status": "active",
            "summary": "Penalidad legacy por gap blocking del ACP; se conserva por compatibilidad.",
            "metric_key": "blocking_gap_penalty",
            "amount": 4,
        },
        {
            "item_key": "design_gap_penalty",
            "label": "Design gap penalty",
            "status": "active",
            "summary": "Penalidad maxima por gap de diseno vivo del Blueprint/ACP.",
            "metric_key": "design_gap_penalty",
            "amount": 4,
        },
        {
            "item_key": "implementation_gap_penalty",
            "label": "Implementation gap penalty",
            "status": "active",
            "summary": "Penalidad maxima por gap que solo se cierra durante implementacion.",
            "metric_key": "implementation_gap_penalty",
            "amount": 2,
        },
        {
            "item_key": "open_question_penalty",
            "label": "Open question penalty",
            "status": "active",
            "summary": "Penalidad legacy por pregunta abierta que afecta build o costo.",
            "metric_key": "open_question_penalty",
            "amount": 1,
        },
        {
            "item_key": "design_open_question_penalty",
            "label": "Design open question penalty",
            "status": "active",
            "summary": "Penalidad por pregunta abierta de diseno pendiente.",
            "metric_key": "design_open_question_penalty",
            "amount": 2,
        },
        {
            "item_key": "implementation_open_question_penalty",
            "label": "Implementation open question penalty",
            "status": "active",
            "summary": "Penalidad por pregunta que depende del entorno real de implementacion.",
            "metric_key": "implementation_open_question_penalty",
            "amount": 1,
        },
        {
            "item_key": "open_question_penalty_cap",
            "label": "Open question cap",
            "status": "active",
            "summary": "Tope de penalidad combinada por preguntas abiertas residuales.",
            "metric_key": "open_question_penalty_cap",
            "amount": 6,
        },
        {
            "item_key": "assumption_penalty",
            "label": "Assumption penalty",
            "status": "active",
            "summary": "Penalidad por supuestos vigentes del ACP.",
            "metric_key": "assumption_penalty",
            "amount": 2,
        },
        {
            "item_key": "assumption_penalty_cap",
            "label": "Assumption cap",
            "status": "active",
            "summary": "Tope de penalidad por supuestos activos.",
            "metric_key": "assumption_penalty_cap",
            "amount": 12,
        },
        {
            "item_key": "score_floor",
            "label": "Score floor",
            "status": "active",
            "summary": "Valor minimo del score final.",
            "metric_key": "score_floor",
            "amount": 5,
        },
        {
            "item_key": "score_ceiling",
            "label": "Score ceiling",
            "status": "active",
            "summary": "Valor maximo del score final.",
            "metric_key": "score_ceiling",
            "amount": 96,
        },
        {
            "item_key": "subscore_complete",
            "label": "Subscore complete",
            "status": "active",
            "summary": "Subscore visual para dimensiones completas.",
            "metric_key": "subscore_complete",
            "amount": 96,
        },
        {
            "item_key": "subscore_partial",
            "label": "Subscore partial",
            "status": "active",
            "summary": "Subscore visual para dimensiones parciales.",
            "metric_key": "subscore_partial",
            "amount": 62,
        },
        {
            "item_key": "subscore_missing",
            "label": "Subscore missing",
            "status": "active",
            "summary": "Subscore visual para dimensiones faltantes.",
            "metric_key": "subscore_missing",
            "amount": 24,
        },
        {
            "item_key": "subscore_blocked",
            "label": "Subscore blocked",
            "status": "active",
            "summary": "Subscore visual para dimensiones bloqueadas.",
            "metric_key": "subscore_blocked",
            "amount": 12,
        },
        {
            "item_key": "subscore_not_applicable",
            "label": "Subscore N/A",
            "status": "active",
            "summary": "Subscore visual para dimensiones no aplicables.",
            "metric_key": "subscore_not_applicable",
            "amount": 100,
        },
    ],
}


WORKSPACE_SECTIONS = [
    {
        "key": "inicio",
        "label": "Inicio",
        "view_kind": "informational",
        "capability_status": "active",
        "source_of_truth": "session summaries, activity and validations",
        "read_only": True,
        "summary": "Resumen ejecutivo del workspace y accesos directos.",
    },
    {
        "key": "proyectos",
        "label": "Proyectos",
        "view_kind": "operational",
        "capability_status": "active",
        "source_of_truth": "session snapshot and step actions",
        "read_only": False,
        "summary": "Workspace principal para ejecutar discovery, canvas, blueprint y exportes.",
    },
    {
        "key": "agentes",
        "label": "Agentes",
        "view_kind": "operational",
        "capability_status": "active",
        "source_of_truth": "blueprint and delivery package",
        "read_only": False,
        "summary": "Centro del blueprint con arquitectura, memoria y guardrails.",
    },
    {
        "key": "plantillas",
        "label": "Plantillas",
        "view_kind": "generated",
        "capability_status": "active",
        "source_of_truth": "delivery_package and exports",
        "read_only": False,
        "summary": "Browser de artefactos versionados, exportes reales y comparativos del paquete tecnico.",
    },
    {
        "key": "evaluaciones",
        "label": "Evaluaciones",
        "view_kind": "operational",
        "capability_status": "active",
        "source_of_truth": "evaluation artifact",
        "read_only": False,
        "summary": "Workbench de evaluacion con dataset, rubrica, corridas y scoring persistido.",
    },
    {
        "key": "monitoreo",
        "label": "Monitoreo",
        "view_kind": "operational",
        "capability_status": "active",
        "source_of_truth": "metric snapshots, alerts, approvals and execution logs",
        "read_only": False,
        "summary": "Monitoreo operativo con metricas agregadas, timeline de errores y alertas activas.",
    },
    {
        "key": "biblioteca",
        "label": "Biblioteca",
        "view_kind": "operational",
        "capability_status": "active",
        "source_of_truth": "artifact registry, blueprint lineage and persisted traces",
        "read_only": False,
        "summary": "Busqueda estructurada de artefactos, lineage del blueprint y trazas persistidas.",
    },
    {
        "key": "integraciones",
        "label": "Integraciones",
        "view_kind": "operational",
        "capability_status": "active",
        "source_of_truth": "integration status snapshots and runtime health checks",
        "read_only": False,
        "summary": "Estado real de OpenAI, PostgreSQL y auth local con ultimo check persistido.",
    },
    {
        "key": "configuracion",
        "label": "Configuracion",
        "view_kind": "operational",
        "capability_status": "active",
        "source_of_truth": "auth state and session controls",
        "read_only": False,
        "summary": "Control local del workspace, acceso y sesiones.",
    },
    {
        "key": "roadmap",
        "label": "Roadmap",
        "view_kind": "generated",
        "capability_status": "active",
        "source_of_truth": "delivery package",
        "read_only": True,
        "summary": "Roadmap formal del blueprint para escalar de MVP 1 a MVP 3 sin sobredisenar.",
    },
    {
        "key": "skill_runtime",
        "label": "Skill runtime",
        "view_kind": "operational",
        "capability_status": "active",
        "source_of_truth": "skill runs and traces",
        "read_only": False,
        "summary": "Runtime real de skills del builder con trazas, diffs y reejecucion parcial.",
    },
    {
        "key": "evaluation_dataset",
        "label": "Evaluation datasets",
        "view_kind": "operational",
        "capability_status": "active",
        "source_of_truth": "evaluation datasets and rubric runs",
        "read_only": False,
        "summary": "Datasets, rubricas y corridas persistidas para validar agentes con evidencia reusable.",
    },
]


def apply_workspace_bootstrap(session: Session, workspace_id: UUID) -> None:
    seed_runtime_feature_flags(session, workspace_id=workspace_id)
    seed_runtime_catalogs(session)
    seed_workflow_templates(session, workspace_id=workspace_id)
    seed_governance_policies(session, workspace_id=workspace_id)
    _seed_deliverable_governance_defaults_if_available(session)
    sync_skill_catalog(session)
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE0,
        description="Base de migracion, feature flags y catalogos del workspace contract v1.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE1,
        description="Paridad del blueprint: ToT, roadmap de evolucion y catalogos activos de etapa 1.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE2,
        description="Runtime real de skills, catalogo persistido y trazas operativas por skill.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE3,
        description="Evaluation workbench con datasets, rubricas, corridas persistidas y comparables.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE4,
        description="Operacion activa con metricas, artefactos versionados, integraciones y alertas persistidas.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE5,
        description="Workflow templates, handoffs, gobierno y subprocesos especializados opcionales de MVP 3.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE6,
        description="Contrato base de estimacion comparativa, catalogos gobernados y feature flag de scaffolding.",
    )


def initialize_new_workspace_configuration(
    session: Session,
    *,
    workspace_id: UUID,
    actor_user_id: UUID | None = None,
) -> None:
    """Seed a newly-created workspace from the Platform Admin workspace when available."""
    source_workspace_id = _resolve_platform_admin_template_workspace_id(session, exclude_workspace_id=workspace_id)
    if source_workspace_id is None:
        apply_workspace_bootstrap(session, workspace_id)
        return

    seed_runtime_catalogs(session)
    _seed_deliverable_governance_defaults_if_available(session)
    sync_skill_catalog(session)
    _inherit_runtime_feature_flags(session, source_workspace_id=source_workspace_id, target_workspace_id=workspace_id)
    _inherit_workflow_templates(session, source_workspace_id=source_workspace_id, target_workspace_id=workspace_id)
    _inherit_governance_policies(session, source_workspace_id=source_workspace_id, target_workspace_id=workspace_id)
    _inherit_commercial_quota_overrides(
        session,
        source_workspace_id=source_workspace_id,
        target_workspace_id=workspace_id,
        actor_user_id=actor_user_id,
    )
    _inherit_workspace_runtime_settings(
        session,
        source_workspace_id=source_workspace_id,
        target_workspace_id=workspace_id,
        actor_user_id=actor_user_id,
    )
    _record_workspace_inheritance_audit(
        session,
        source_workspace_id=source_workspace_id,
        target_workspace_id=workspace_id,
        actor_user_id=actor_user_id,
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE0,
        description="Base de migracion, feature flags y catalogos del workspace contract v1.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE1,
        description="Paridad del blueprint: ToT, roadmap de evolucion y catalogos activos de etapa 1.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE2,
        description="Runtime real de skills, catalogo persistido y trazas operativas por skill.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE3,
        description="Evaluation workbench con datasets, rubricas, corridas persistidas y comparables.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE4,
        description="Operacion activa con metricas, artefactos versionados, integraciones y alertas persistidas.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE5,
        description="Workflow templates, handoffs, gobierno y subprocesos especializados opcionales de MVP 3.",
    )
    _ensure_migration(
        session,
        migration_key=MIGRATION_KEY_STAGE6,
        description="Contrato base de estimacion comparativa, catalogos gobernados y feature flag de scaffolding.",
    )


def _resolve_platform_admin_template_workspace_id(
    session: Session,
    *,
    exclude_workspace_id: UUID,
) -> UUID | None:
    configured_email = get_settings().local_admin_email.strip().lower()
    query = (
        select(UserRecord, PlatformRoleAssignmentRecord)
        .join(PlatformRoleAssignmentRecord, PlatformRoleAssignmentRecord.user_id == UserRecord.id)
        .where(
            PlatformRoleAssignmentRecord.role == PlatformRole.platform_admin,
            PlatformRoleAssignmentRecord.is_active == True,  # noqa: E712
            UserRecord.is_active == True,  # noqa: E712
        )
    )
    if configured_email:
        query = query.where(UserRecord.email == configured_email)
    row = session.exec(query.order_by(PlatformRoleAssignmentRecord.created_at.asc())).first()
    if row is None and configured_email:
        row = session.exec(
            select(UserRecord, PlatformRoleAssignmentRecord)
            .join(PlatformRoleAssignmentRecord, PlatformRoleAssignmentRecord.user_id == UserRecord.id)
            .where(
                PlatformRoleAssignmentRecord.role == PlatformRole.platform_admin,
                PlatformRoleAssignmentRecord.is_active == True,  # noqa: E712
                UserRecord.is_active == True,  # noqa: E712
            )
            .order_by(PlatformRoleAssignmentRecord.created_at.asc())
        ).first()
    if row is None:
        return None

    admin_user = row[0]
    candidate_id = admin_user.default_workspace_id
    if candidate_id is None or candidate_id == exclude_workspace_id:
        return None
    workspace = session.get(WorkspaceRecord, candidate_id)
    if workspace is None or not workspace.is_active:
        return None
    return workspace.id


def _inherit_runtime_feature_flags(
    session: Session,
    *,
    source_workspace_id: UUID,
    target_workspace_id: UUID,
) -> None:
    existing = {
        item.flag_key
        for item in session.exec(
            select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.workspace_id == target_workspace_id)
        ).all()
    }
    source_rows = session.exec(
        select(RuntimeFeatureFlagRecord)
        .where(RuntimeFeatureFlagRecord.workspace_id == source_workspace_id)
        .order_by(RuntimeFeatureFlagRecord.flag_key.asc())
    ).all()
    for row in source_rows:
        if row.flag_key in existing:
            continue
        session.add(
            RuntimeFeatureFlagRecord(
                workspace_id=target_workspace_id,
                flag_key=row.flag_key,
                enabled=row.enabled,
                description=row.description,
                stage_hint=row.stage_hint,
            )
        )
        existing.add(row.flag_key)

    for payload in DEFAULT_FEATURE_FLAGS:
        if payload["key"] in existing:
            continue
        session.add(
            RuntimeFeatureFlagRecord(
                workspace_id=target_workspace_id,
                flag_key=payload["key"],
                enabled=payload["enabled"],
                description=payload["description"],
                stage_hint=payload["stage_hint"],
            )
        )
        existing.add(payload["key"])
    session.flush()


def _seed_deliverable_governance_defaults_if_available(session: Session) -> None:
    try:
        seed_deliverable_governance_defaults(session)
    except OperationalError as exc:
        session.rollback()
        if "deliverable_governance_v1" not in str(exc):
            raise


def _inherit_workflow_templates(
    session: Session,
    *,
    source_workspace_id: UUID,
    target_workspace_id: UUID,
) -> None:
    existing = {
        item.template_key
        for item in session.exec(
            select(WorkflowTemplateRecord).where(WorkflowTemplateRecord.workspace_id == target_workspace_id)
        ).all()
    }
    source_rows = session.exec(
        select(WorkflowTemplateRecord)
        .where(WorkflowTemplateRecord.workspace_id == source_workspace_id)
        .order_by(WorkflowTemplateRecord.template_key.asc())
    ).all()
    for row in source_rows:
        if row.template_key in existing:
            continue
        session.add(
            WorkflowTemplateRecord(
                workspace_id=target_workspace_id,
                template_key=row.template_key,
                label=row.label,
                summary=row.summary,
                architecture_scope=list(row.architecture_scope or []),
                supports_approvals=row.supports_approvals,
                supports_handoffs=row.supports_handoffs,
                workflow_profile=dict(row.workflow_profile or {}),
                governance_hints=list(row.governance_hints or []),
                is_active=row.is_active,
            )
        )
        existing.add(row.template_key)

    for payload in DEFAULT_WORKFLOW_TEMPLATES:
        if payload["template_key"] in existing:
            continue
        session.add(WorkflowTemplateRecord(workspace_id=target_workspace_id, **payload))
        existing.add(str(payload["template_key"]))
    session.flush()


def _inherit_governance_policies(
    session: Session,
    *,
    source_workspace_id: UUID,
    target_workspace_id: UUID,
) -> None:
    existing = {
        item.policy_key
        for item in session.exec(
            select(GovernancePolicyRecord).where(GovernancePolicyRecord.workspace_id == target_workspace_id)
        ).all()
    }
    source_rows = session.exec(
        select(GovernancePolicyRecord)
        .where(GovernancePolicyRecord.workspace_id == source_workspace_id)
        .order_by(GovernancePolicyRecord.policy_key.asc())
    ).all()
    for row in source_rows:
        if row.policy_key in existing:
            continue
        session.add(
            GovernancePolicyRecord(
                workspace_id=target_workspace_id,
                policy_key=row.policy_key,
                label=row.label,
                summary=row.summary,
                scope=row.scope,
                is_active=row.is_active,
                policy_payload=dict(row.policy_payload or {}),
            )
        )
        existing.add(row.policy_key)

    for payload in DEFAULT_GOVERNANCE_POLICIES:
        if payload["policy_key"] in existing:
            continue
        session.add(GovernancePolicyRecord(workspace_id=target_workspace_id, **payload))
        existing.add(str(payload["policy_key"]))
    session.flush()


def _active_workspace_runtime_settings(
    session: Session,
    *,
    workspace_id: UUID,
) -> WorkspaceRuntimeSettingsRecord | None:
    return session.exec(
        select(WorkspaceRuntimeSettingsRecord)
        .where(
            WorkspaceRuntimeSettingsRecord.workspace_id == workspace_id,
            WorkspaceRuntimeSettingsRecord.is_active == True,  # noqa: E712
        )
        .order_by(WorkspaceRuntimeSettingsRecord.version.desc())
    ).first()


def _inherit_workspace_runtime_settings(
    session: Session,
    *,
    source_workspace_id: UUID,
    target_workspace_id: UUID,
    actor_user_id: UUID | None,
) -> None:
    existing = _active_workspace_runtime_settings(session, workspace_id=target_workspace_id)
    if existing is not None:
        return
    source = _active_workspace_runtime_settings(session, workspace_id=source_workspace_id)
    if source is None:
        return
    now = utc_now()
    session.add(
        WorkspaceRuntimeSettingsRecord(
            workspace_id=target_workspace_id,
            active_provider=source.active_provider,
            agent_execution_backend=source.agent_execution_backend,
            knowledge_access_backend=source.knowledge_access_backend,
            provider_overrides=dict(source.provider_overrides or {}),
            uses_platform_credentials=source.uses_platform_credentials,
            is_active=True,
            version=1,
            updated_by_user_id=actor_user_id,
            created_at=now,
            updated_at=now,
        )
    )
    session.flush()


def _record_workspace_inheritance_audit(
    session: Session,
    *,
    source_workspace_id: UUID,
    target_workspace_id: UUID,
    actor_user_id: UUID | None,
) -> None:
    session.add(
        RuntimeSettingsAuditRecord(
            scope_type=RuntimeGovernanceScopeType.workspace,
            scope_id=str(target_workspace_id),
            change_type=WORKSPACE_BASE_INHERITANCE_EVENT,
            before_payload_redacted={},
            after_payload_redacted={
                "source_workspace_id": str(source_workspace_id),
                "target_workspace_id": str(target_workspace_id),
                "copied": [
                    "runtime_feature_flags",
                    "workflow_templates",
                    "governance_policies",
                    "commercial_quota_workspace_overrides",
                    "workspace_runtime_settings_if_present",
                ],
                "secret_policy": "provider secrets are not copied between workspaces",
            },
            actor_user_id=actor_user_id,
        )
    )
    session.flush()


def _inherit_commercial_quota_overrides(
    session: Session,
    *,
    source_workspace_id: UUID,
    target_workspace_id: UUID,
    actor_user_id: UUID | None,
) -> None:
    existing = {
        item.product_key
        for item in session.exec(
            select(CommercialQuotaWorkspaceOverrideRecord).where(
                CommercialQuotaWorkspaceOverrideRecord.workspace_id == target_workspace_id
            )
        ).all()
    }
    source_rows = session.exec(
        select(CommercialQuotaWorkspaceOverrideRecord)
        .where(CommercialQuotaWorkspaceOverrideRecord.workspace_id == source_workspace_id)
        .order_by(CommercialQuotaWorkspaceOverrideRecord.product_key.asc())
    ).all()
    now = utc_now()
    for row in source_rows:
        if row.product_key in existing:
            continue
        session.add(
            CommercialQuotaWorkspaceOverrideRecord(
                workspace_id=target_workspace_id,
                product_key=row.product_key,
                is_active=row.is_active,
                enabled_override=row.enabled_override,
                free_units_override=row.free_units_override,
                consumption_priority_override=list(row.consumption_priority_override or []),
                checkout_required_on_zero_balance_override=row.checkout_required_on_zero_balance_override,
                fifo_auto_approval_enabled_override=row.fifo_auto_approval_enabled_override,
                default_blocked_request_ttl_hours_override=row.default_blocked_request_ttl_hours_override,
                default_checkout_ttl_minutes_override=row.default_checkout_ttl_minutes_override,
                debt_enabled_override=row.debt_enabled_override,
                effective_from=row.effective_from,
                effective_to=row.effective_to,
                notes=row.notes,
                updated_by_user_id=actor_user_id,
                metadata_payload=dict(row.metadata_payload or {}),
                created_at=now,
                updated_at=now,
            )
        )
        existing.add(row.product_key)
    session.flush()


def _ensure_migration(session: Session, *, migration_key: str, description: str) -> None:
    existing = session.exec(
        select(SchemaMigrationRecord).where(SchemaMigrationRecord.migration_key == migration_key)
    ).first()
    if existing is None:
        session.add(
            SchemaMigrationRecord(
                migration_key=migration_key,
                description=description,
            )
        )
        session.commit()


def _build_catalog_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"item_key", "label"}
    }


def _seed_runtime_feature_flags_once(session: Session, workspace_id: UUID) -> None:
    existing = {
        item.flag_key: item
        for item in session.exec(
            select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.workspace_id == workspace_id)
        ).all()
    }
    for payload in DEFAULT_FEATURE_FLAGS:
        record = existing.get(payload["key"])
        if record is None:
            session.add(
                RuntimeFeatureFlagRecord(
                    workspace_id=workspace_id,
                    flag_key=payload["key"],
                    enabled=payload["enabled"],
                    description=payload["description"],
                    stage_hint=payload["stage_hint"],
                )
            )
            continue
        record.description = payload["description"]
        record.stage_hint = payload["stage_hint"]
        session.add(record)
    session.commit()


def seed_runtime_feature_flags(session: Session, workspace_id: UUID) -> None:
    for attempt in range(3):
        try:
            _seed_runtime_feature_flags_once(session, workspace_id)
            return
        except IntegrityError:
            session.rollback()
            if attempt == 2:
                raise


def seed_runtime_catalogs(session: Session) -> None:
    existing = {
        (item.catalog_key, item.item_key): item
        for item in session.exec(select(RuntimeCatalogEntryRecord)).all()
    }
    for catalog_key, items in DEFAULT_CATALOGS.items():
        for index, payload in enumerate(items, start=1):
            record = existing.get((catalog_key, payload["item_key"]))
            if record is None:
                session.add(
                    RuntimeCatalogEntryRecord(
                        catalog_key=catalog_key,
                        item_key=payload["item_key"],
                        label=payload["label"],
                        version=CATALOG_VERSION_CURRENT,
                        order_index=index,
                        is_active=payload["status"] == "active",
                        payload=_build_catalog_payload(payload),
                    )
                )
                continue
            record.label = payload["label"]
            record.version = CATALOG_VERSION_CURRENT
            record.order_index = index
            record.is_active = payload["status"] == "active"
            record.payload = _build_catalog_payload(payload)
            session.add(record)
    session.commit()


def build_workspace_contract(session: Session, workspace_id: UUID) -> WorkspaceContract:
    flags = session.exec(
        select(RuntimeFeatureFlagRecord)
        .where(RuntimeFeatureFlagRecord.workspace_id == workspace_id)
        .order_by(RuntimeFeatureFlagRecord.flag_key.asc())
    ).all()
    catalog_rows = session.exec(
        select(RuntimeCatalogEntryRecord)
        .order_by(RuntimeCatalogEntryRecord.catalog_key.asc(), RuntimeCatalogEntryRecord.order_index.asc())
    ).all()

    grouped_catalogs: dict[str, list[RuntimeCatalogEntryRecord]] = {}
    for row in catalog_rows:
        grouped_catalogs.setdefault(row.catalog_key, []).append(row)

    return WorkspaceContract(
        sections=[WorkspaceSectionEntry(**item) for item in WORKSPACE_SECTIONS],
        feature_flags=(
            [
                FeatureFlagEntry(
                    key=item.flag_key,
                    enabled=item.enabled,
                    description=item.description,
                    stage_hint=item.stage_hint,
                )
                for item in flags
            ]
            if flags
            else [
                FeatureFlagEntry(
                    key=item["key"],
                    enabled=item["enabled"],
                    description=item["description"],
                    stage_hint=item["stage_hint"],
                )
                for item in DEFAULT_FEATURE_FLAGS
            ]
        ),
        catalogs=(
            [
                CatalogSummaryEntry(
                    catalog_key=catalog_key,
                    version=items[0].version if items else CATALOG_VERSION_CURRENT,
                    item_count=len(items),
                    active_count=sum(1 for item in items if item.is_active),
                    items=[
                        CatalogItemSummary(
                            item_key=item.item_key,
                            label=item.label,
                            status=str(item.payload.get("status", "active")),
                            summary=str(item.payload.get("summary", "")),
                        )
                        for item in items
                    ],
                )
                for catalog_key, items in grouped_catalogs.items()
            ]
            if grouped_catalogs
            else [
                CatalogSummaryEntry(
                    catalog_key=catalog_key,
                    version=CATALOG_VERSION_CURRENT,
                    item_count=len(items),
                    active_count=sum(1 for item in items if item["status"] == "active"),
                    items=[
                        CatalogItemSummary(
                            item_key=item["item_key"],
                            label=item["label"],
                            status=item["status"],
                            summary=item["summary"],
                        )
                        for item in items
                    ],
                )
                for catalog_key, items in DEFAULT_CATALOGS.items()
            ]
        ),
    )
