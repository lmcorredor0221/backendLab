from __future__ import annotations

from collections.abc import Iterable

from app.models import (
    ACPFileEntry,
    ACPValidationReport,
    ConstructionGapEntry,
    ConstructionQuestionEntry,
    ConstructionQuestionOption,
    ConstructionReadinessReport,
    MemoryDependencyGap,
    MemoryRecommendationArtifact,
    SessionSnapshot,
)
from app.services.blueprint_consistency_service import ensure_blueprint_consistency_report
from app.services.acp_paths import ACP_CANONICAL_ENV_TEMPLATE_PATH, build_tool_contract_path_for_tool
from app.services.objective_contracts import (
    active_objective,
    build_objective_contract_bundle,
    objective_gate_enabled,
    objective_questions_enabled,
)
from app.services.tool_family_projection import (
    ODOO_CANONICAL_CONNECTOR_KEYS,
    ODOO_LEGACY_CONNECTOR_ALIASES,
    project_blueprint_tools_for_construction,
    resolve_google_workspace_connector_key,
    resolve_odoo_connector_key,
    resolve_whatsapp_connector_key,
    snapshot_has_odoo_quote_signal,
)


INTERNAL_BUILDER_TOOL_NAMES = {
    "normalize_discovery",
    "build_canvas",
    "build_blueprint",
    "promote_blueprint_for_implementation",
}

def _snapshot_has_odoo_quote_signal(snapshot: SessionSnapshot) -> bool:
    return snapshot_has_odoo_quote_signal(snapshot)

CONSTRUCTION_GAP_CATALOG: dict[str, dict[str, str]] = {
    "acp_package_validation_blocked": {
        "severity": "blocking",
        "remediation": "Resolver todos los errores bloqueantes del ACP antes de continuar con construccion.",
    },
    "knowledge_sources_missing": {
        "severity": "warning",
        "remediation": "Definir fuentes, ownership, ingestion y estrategia semantica de knowledge antes del retrieval real.",
    },
    "runtime_contract_incomplete": {
        "severity": "warning",
        "remediation": "Cerrar fallback model, vector store y origen de secretos para el runtime objetivo.",
    },
    "deployment_target_unknown": {
        "severity": "warning",
        "remediation": "Definir entorno objetivo, estrategia de imagen y restricciones operativas del despliegue.",
    },
    "external_api_contracts_missing": {
        "severity": "warning",
        "remediation": "Publicar contratos API abstractos y reglas de sandbox antes de construir integraciones externas.",
    },
    "objective_contract_validation": {
        "severity": "warning",
        "remediation": "Confirmar, corregir o rechazar el objetivo inferido antes de activar Objective Loop o runtime operacional.",
    },
    "memory_dependency_questions": {
        "severity": "warning",
        "remediation": "Resolver, diferir o excluir dependencias de memoria desde preguntas ACP; no bloquear aprobacion LEAN por decisiones de construccion.",
    },
}

BLUEPRINT_HANDOFF_PROCESS_DEBT_ISSUE_KEYS = {
    "memory_required_tool_dependency_missing",
    "tools_recommendation_stale",
    "memory_recommendation_stale",
    "estimate_stale",
}
BLUEPRINT_HANDOFF_PROCESS_DEBT_PREFIXES = (
    "design_blueprint_projection_drift:",
    "validate_source_stage_drift:",
)


def is_blueprint_handoff_process_debt_issue(issue_key: str) -> bool:
    normalized = issue_key.strip()
    return normalized in BLUEPRINT_HANDOFF_PROCESS_DEBT_ISSUE_KEYS or any(
        normalized.startswith(prefix) for prefix in BLUEPRINT_HANDOFF_PROCESS_DEBT_PREFIXES
    )


def _file_map(files: list[ACPFileEntry]) -> dict[str, ACPFileEntry]:
    return {item.path: item for item in files}


def _contains_needs_review(entry: ACPFileEntry | None) -> bool:
    if entry is None:
        return False
    normalized = entry.content_text.lower()
    return "needs_review" in normalized or "pendiente" in normalized


def _question(
    *,
    question_key: str,
    question_text: str,
    rationale: str,
    purpose: str = "",
    expected_answer_format: str = "",
    target_owner: str = "",
    blocking: bool = False,
    options: list[ConstructionQuestionOption] | None = None,
    question_kind: str = "general",
    subject_type: str = "",
    subject_id: str = "",
    allowed_decisions: list[str] | None = None,
    answer_semantics: str = "",
    contract_version: int = 1,
) -> ConstructionQuestionEntry:
    return ConstructionQuestionEntry(
        question_key=question_key,
        question_text=question_text,
        rationale=rationale,
        purpose=purpose,
        expected_answer_format=expected_answer_format,
        target_owner=target_owner,
        blocking=blocking,
        options=options or [],
        question_kind=question_kind,
        subject_type=subject_type,
        subject_id=subject_id,
        allowed_decisions=allowed_decisions or [],
        answer_semantics=answer_semantics,
        contract_version=contract_version,
    )


def _gap(
    *,
    gap_key: str,
    title: str,
    domain: str,
    severity: str,
    blocking_stage: str,
    summary: str,
    evidence_paths: list[str],
    source_sections: list[str],
    current_assumptions: list[str],
    closure_criteria: list[str],
    questions: list[ConstructionQuestionEntry],
) -> ConstructionGapEntry:
    catalog_entry = CONSTRUCTION_GAP_CATALOG.get(gap_key, {})
    return ConstructionGapEntry(
        gap_key=gap_key,
        title=title,
        domain=domain,
        severity=catalog_entry.get("severity", severity),
        status="open",
        blocking_stage=blocking_stage,
        summary=summary,
        remediation=catalog_entry.get("remediation", ""),
        evidence_paths=evidence_paths,
        source_sections=source_sections,
        current_assumptions=current_assumptions,
        closure_criteria=closure_criteria,
        questions=questions,
    )


def _collect_validation_gap(report: ACPValidationReport) -> ConstructionGapEntry | None:
    blocking_issues = [item for item in report.issues if item.blocking and item.severity == "error"]
    if not blocking_issues:
        return None
    evidence_paths = sorted({item.path for item in blocking_issues if item.path})
    source_sections = sorted({section for item in blocking_issues for section in item.source_sections})
    return _gap(
        gap_key="acp_package_validation_blocked",
        title="El ACP aun no pasa sus validaciones base",
        domain="package",
        severity="blocking",
        blocking_stage="package_validation",
        summary="Antes de continuar con construccion, el ACP debe cerrar los errores bloqueantes detectados en la validacion base.",
        evidence_paths=evidence_paths,
        source_sections=source_sections,
        current_assumptions=[],
        closure_criteria=[
            "No deben quedar errores bloqueantes en ACPValidationReport.",
            "El ACP debe poder exportarse sin campos criticos faltantes.",
        ],
        questions=[],
    )


def _objective_validation_question(objective, *, is_active: bool, is_blocking: bool) -> ConstructionQuestionEntry:
    subject_label = "objetivo operativo" if is_active else "subobjetivo operativo"
    confirm_label = "Confirmar objetivo" if is_active else "Confirmar subobjetivo"
    reject_label = "Rechazar objetivo" if is_active else "Rechazar subobjetivo"
    reject_impact = (
        "El runtime no debe activar Objective Loop hasta que exista un objetivo corregido o aprobado."
        if is_active
        else "El agente delegado no debe usar este subobjetivo hasta que exista una version corregida o aprobada."
    )
    return _question(
        question_key=f"objective_validation:{objective.objective_id}:v{objective.version}",
        question_text=f"Confirma o corrige el {subject_label} inferido para {objective.owner}: {objective.statement}",
        rationale=(
            "El objetivo se usara para alinear prompts, criterios de exito, condiciones de terminacion y seguimiento "
            "de progreso del agente."
        ),
        purpose="Validar el Objective Contract dentro del ACP sin crear un mecanismo nuevo de interaccion.",
        expected_answer_format="Selecciona confirmar/rechazar o escribe la version corregida en una frase verificable.",
        target_owner=objective.owner or "business_owner",
        blocking=is_blocking,
        options=[
            ConstructionQuestionOption(
                key="confirm",
                label=confirm_label,
                description="Mantener la formulacion inferida como contrato activo.",
                impact="El ACP puede usar este contrato para prompts, criterios y seguimiento de progreso.",
                example=objective.statement,
                recommended=True,
                confidence=objective.confidence,
                source_refs=list(objective.source_refs),
            ),
            ConstructionQuestionOption(
                key="reject",
                label=reject_label,
                description="Marcar la formulacion inferida como no valida para este alcance.",
                impact=reject_impact,
                example="El objetivo no corresponde al proceso real que queremos construir.",
            ),
        ],
        question_kind="objective_validation",
        subject_type="objective" if is_active else "subobjective",
        subject_id=objective.objective_id,
        allowed_decisions=["confirm", "correct", "reject"],
        answer_semantics="update_objective_contract",
        contract_version=objective.version,
    )


def _collect_objective_validation_gap(snapshot: SessionSnapshot) -> ConstructionGapEntry | None:
    if not objective_questions_enabled(snapshot):
        return None
    bundle = build_objective_contract_bundle(snapshot)
    active = active_objective(bundle)
    inferred_objectives = [objective for objective in bundle.objectives if objective.status == "inferred"]
    if active is None or not inferred_objectives:
        return None
    active_objective_id = active.objective_id
    is_blocking = objective_gate_enabled(snapshot)
    summary = (
        "LAB infirio el objetivo operativo y los subobjetivos de los agentes propuestos para alimentar prompts, "
        "criterios de cierre y posible Objective Loop. Antes de usarlos como contrato de ejecucion, el usuario "
        "debe validarlos o corregirlos desde Responder preguntas."
    )
    questions = [
        _objective_validation_question(
            objective,
            is_active=objective.objective_id == active_objective_id,
            is_blocking=is_blocking and objective.objective_id == active_objective_id,
        )
        for objective in inferred_objectives
    ]
    return _gap(
        gap_key="objective_contract_validation",
        title="Validar objetivos y subobjetivos operativos",
        domain="objectives",
        severity="blocking" if is_blocking else "warning",
        blocking_stage="acp_questions_resolution",
        summary=summary,
        evidence_paths=["contracts/objective-contract.v1.json", "ACP/objectives/objective-contract.yaml"],
        source_sections=sorted({source for objective in inferred_objectives for source in objective.source_refs}),
        current_assumptions=[
            *[
                f"Objetivo inferido: {objective.statement}"
                for objective in inferred_objectives
                if objective.objective_id == active_objective_id
            ],
            *[
                f"Subobjetivo inferido para {objective.owner}: {objective.statement}"
                for objective in inferred_objectives
                if objective.objective_id != active_objective_id
            ],
            f"Contratos pendientes de validacion: {len(inferred_objectives)}",
        ],
        closure_criteria=[
            "Confirmar que el objetivo representa lo que el agente debe perseguir.",
            "Confirmar o corregir los subobjetivos de cada agente delegado.",
            "Rechazar cualquier objetivo o subobjetivo que no deba alimentar prompts, Objective Loop o runtime operacional.",
        ],
        questions=questions,
    )


def _collect_knowledge_gap(files: dict[str, ACPFileEntry]) -> ConstructionGapEntry | None:
    sources = files.get("ACP/knowledge/sources.yaml")
    ingestion = files.get("ACP/knowledge/ingestion.yaml")
    embeddings = files.get("ACP/knowledge/embeddings.yaml")
    return _gap(
        gap_key="knowledge_sources_missing",
        title="La capa de conocimiento aun no esta especificada",
        domain="knowledge",
        severity="warning",
        blocking_stage="knowledge_integration",
        summary="El ACP conserva placeholders en sources, ingestion o embeddings. Un builder agent debe cerrar estas definiciones antes de automatizar retrieval real.",
        evidence_paths=[item.path for item in [sources, ingestion, embeddings] if item is not None],
        source_sections=sorted(
            {
                section
                for item in [sources, ingestion, embeddings]
                if item is not None
                for section in item.source_sections
            }
        ),
        current_assumptions=["No existe una base de conocimiento externa confirmada en la captura actual."],
        closure_criteria=[
            "Definir fuentes de conocimiento concretas.",
            "Definir estrategia de ingestion y ownership.",
            "Definir proveedor de embeddings o justificar que no aplica.",
        ],
        questions=[
            _question(
                question_key="knowledge_sources",
                question_text="¿De qué lugares o fuentes de información debe obtener respuestas el asistente y quién administra cada una?",
                rationale="Para responder con precisión, el asistente requiere consultar fuentes oficiales autorizadas por tu organización.",
                purpose="Identificar los orígenes de datos oficiales para que el asistente entregue información verídica y confiable.",
                expected_answer_format="una línea por fuente con name=<fuente>; type=<tipo>; owner=<propietario>; frequency=<frecuencia>",
                target_owner="domain_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="internal_docs",
                        label="Documentos e información subida manualmente (PDFs, Word)",
                        description="Cargar archivos operativos e instructivos directamente en la plataforma.",
                        impact="El asistente responderá basándose en los manuales e instructivos que adjuntes.",
                        example="Ejemplo: Manual de atención al cliente en PDF o reglamento interno en Word."
                    ),
                    ConstructionQuestionOption(
                        key="database_api",
                        label="Base de datos o sistema existente de la empresa (API)",
                        description="Conectar directamente con sistemas donde ya vive la información viva.",
                        impact="El asistente consultará datos actualizados automáticamente desde los sistemas actuales de la empresa.",
                        example="Ejemplo: Conexión con el sistema de inventarios o CRM corporativo."
                    ),
                    ConstructionQuestionOption(
                        key="none",
                        label="Sin fuentes externas (Conocimiento general)",
                        description="El asistente responderá con su entrenamiento básico sin consultar archivos privados.",
                        impact="No requiere configuración de archivos, pero no conocerá detalles específicos de tu negocio.",
                        example="Ejemplo: Asistente para redacción de correos o apoyo en lluvias de ideas."
                    )
                ]
            ),
            _question(
                question_key="knowledge_ingestion",
                question_text="¿Con qué frecuencia debe actualizarse la información que utiliza el asistente?",
                rationale="Conocer la frecuencia de cambio permite programar la sincronización de datos de manera eficiente.",
                purpose="Establecer cada cuánto se revisan y leen los datos para garantizar respuestas al día.",
                expected_answer_format="strategy=<mecanismo>; frequency=<frecuencia>; owner=<propietario>",
                target_owner="knowledge_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="realtime",
                        label="Actualización continua (En tiempo real)",
                        description="Cualquier cambio en tus sistemas se refleja de inmediato en las respuestas.",
                        impact="Garantiza respuestas al segundo, ideal para productos con precio o stock cambiante.",
                        example="Ejemplo: Cambios de disponibilidad de habitaciones de hotel o inventario."
                    ),
                    ConstructionQuestionOption(
                        key="periodic",
                        label="Actualización programada (Diaria o Semanal)",
                        description="Sincronización automática periódica en horarios de bajo tráfico.",
                        impact="Mantiene la información relevante al día reduciendo la carga en los servidores.",
                        example="Ejemplo: Sincronización nocturna de políticas o catálogos de productos."
                    ),
                    ConstructionQuestionOption(
                        key="manual",
                        label="Carga manual al modificar un documento",
                        description="Solo se actualiza cuando un usuario publica intencionalmente un nuevo archivo.",
                        impact="Control total sobre cuándo se actualizan las respuestas del asistente.",
                        example="Ejemplo: Publicación anual del manual de beneficios para colaboradores."
                    )
                ]
            ),
            _question(
                question_key="knowledge_embedding_strategy",
                question_text="¿Qué nivel de búsqueda inteligente por significado requiere el asistente?",
                rationale="Permite entender el sentido de las preguntas aunque los usuarios utilicen palabras o sinónimos diferentes.",
                purpose="Definir cómo el asistente interpreta las intenciones de búsqueda de los usuarios.",
                expected_answer_format="provider=<proveedor>; notes=<detalle>",
                target_owner="ai_architect",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="semantic_standard",
                        label="Búsqueda por significado (Recomendado)",
                        description="Encuentra respuestas interpretando la intención, aunque no coincidan las palabras exactas.",
                        impact="Brinda una experiencia fluida y natural para los usuarios sin requerir palabras clave exactas.",
                        example="Ejemplo: Preguntar '¿dónde pido vacaciones?' y encontrar 'Procedimiento de licencias'."
                    ),
                    ConstructionQuestionOption(
                        key="exact_keyword",
                        label="Búsqueda por palabras exactas",
                        description="Busca coincidencias textuales directas de las palabras escritas.",
                        impact="Respuestas muy rápidas ideales para búsquedas por códigos o identificadores numéricos.",
                        example="Ejemplo: Buscar por código de error 'ERR-504' o código de producto."
                    ),
                    ConstructionQuestionOption(
                        key="none",
                        label="Sin búsqueda avanzada de documentos",
                        description="No requiere procesar grandes volúmenes de texto o manuales.",
                        impact="Simplifica la configuración inicial para asistentes de tareas simples.",
                        example="Ejemplo: Asistentes que siguen guiones de conversación fijos."
                    )
                ]
            ),
        ],
    )


def _collect_runtime_gap(snapshot: SessionSnapshot, files: dict[str, ACPFileEntry]) -> ConstructionGapEntry | None:
    runtime_models = files.get("ACP/runtime/models.yaml")
    runtime_providers = files.get("ACP/runtime/providers.yaml")
    runtime_config = files.get("ACP/runtime/config.yaml")
    env_template = files.get(ACP_CANONICAL_ENV_TEMPLATE_PATH)
    knowledge_mode = (
        snapshot.blueprint.knowledge_profile.mode.strip().lower()
        if snapshot.blueprint is not None and snapshot.blueprint.knowledge_profile is not None
        else ""
    )
    requires_vector_store = bool(
        snapshot.blueprint
        and (
            knowledge_mode == "rag"
            or any("vector" in layer.lower() for layer in snapshot.blueprint.memory_profile.storage_layers)
        )
    )
    has_runtime_placeholder = any(
        item is not None and (item.warnings or _contains_needs_review(item))
        for item in [runtime_models, runtime_providers, runtime_config]
    )
    severity = "warning"
    return _gap(
        gap_key="runtime_contract_incomplete",
        title="El runtime aun no tiene todos los parametros de construccion",
        domain="runtime",
        severity=severity,
        blocking_stage="runtime_configuration",
        summary="El ACP aun no cierra todos los detalles de runtime, especialmente fallback model, vector store o fuentes de secretos.",
        evidence_paths=[item.path for item in [runtime_config, runtime_models, runtime_providers, env_template] if item is not None],
        source_sections=sorted(
            {
                section
                for item in [runtime_config, runtime_models, runtime_providers, env_template]
                if item is not None
                for section in item.source_sections
            }
        ),
        current_assumptions=[
            "El provider LLM base se deriva de las integraciones activas del builder.",
            "PostgreSQL y auth local se toman como baseline del entorno actual.",
        ],
        closure_criteria=[
            "Definir fallback model si se requiere resiliencia del runtime.",
            "Definir vector DB o declarar que no aplica al caso.",
            "Definir fuente y owner de variables sensibles del entorno.",
        ],
        questions=[
            _question(
                question_key="runtime_fallback_model",
                question_text="¿Deseas activar un modelo de respaldo de Inteligencia Artificial por si el principal presenta fallas?",
                rationale="Un modelo de respaldo garantiza respuestas ininterrumpidas si el proveedor principal se satura.",
                purpose="Mantener la alta disponibilidad del servicio ante problemas temporales del proveedor primario.",
                expected_answer_format="model=<modelo>; condition=<regla> o 'no aplica'",
                target_owner="ai_architect",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="auto_fallback",
                        label="Activar respaldo automático (Recomendado)",
                        description="Conmuta automáticamente a un segundo proveedor si el principal se ralentiza o cae.",
                        impact="Los usuarios nunca experimentarán caídas en el servicio.",
                        example="Ejemplo: Usar Anthropic Claude como respaldo secundario si OpenAI no responde."
                    ),
                    ConstructionQuestionOption(
                        key="single_model",
                        label="Modelo único sin respaldo",
                        description="Operar únicamente con un proveedor principal.",
                        impact="Reduce costos y complejidad inicial, aceptando breves pausas si el proveedor principal falla.",
                        example="Ejemplo: Operación estándar suficiente para asistentes internos."
                    )
                ]
            ),
            _question(
                question_key="runtime_vector_store",
                question_text="¿Dónde prefieres almacenar la memoria de búsqueda avanzada del asistente?",
                rationale="Determina el motor de base de datos donde se guardan los datos para consultas rápidas por significado.",
                purpose="Elegir la tecnología para el almacenamiento de memoria y documentos del asistente.",
                expected_answer_format="vector_store=<proveedor>; notes=<detalle> o 'no aplica'",
                target_owner="platform_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="cloud_managed",
                        label="Servicio en la nube administrado",
                        description="La plataforma gestiona el almacenamiento sin requerir mantenimiento técnico.",
                        impact="Cero esfuerzo de administración con alta disponibilidad desde el inicio.",
                        example="Ejemplo: Uso de Qdrant o Pinecone completamente gestionado."
                    ),
                    ConstructionQuestionOption(
                        key="local_database",
                        label="Base de datos integrada en el proyecto",
                        description="Guardar los índices directamente en la base de datos principal de la empresa.",
                        impact="Toda la información permanece dentro de tu infraestructura actual.",
                        example="Ejemplo: PostgreSQL con extensión pgvector en servidores propios."
                    ),
                    ConstructionQuestionOption(
                        key="none",
                        label="Sin almacenamiento de memoria persistente",
                        description="Para asistentes que no requieren recordar documentos largos.",
                        impact="No requiere contratar ni configurar bases de datos adicionales.",
                        example="Ejemplo: Asistentes de cálculo o procesadores de texto en línea."
                    )
                ]
            ),
            _question(
                question_key="runtime_secret_source",
                question_text="¿Cómo se administrarán las contraseñas y claves secretas necesarias para operar?",
                rationale="Las claves de acceso permiten al asistente comunicarse de forma segura con proveedores de IA.",
                purpose="Garantizar la protección y administración segura de credenciales sensibles.",
                expected_answer_format="source=<mecanismo>; owner=<propietario>",
                target_owner="platform_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="env_variables",
                        label="Archivo de variables de entorno (.env)",
                        description="Guardar las claves en un archivo protegido en el servidor de aplicación.",
                        impact="Método estándar, seguro y de fácil administración para el equipo de sistemas.",
                        example="Ejemplo: Guardar la clave OPENAI_API_KEY en el servidor."
                    ),
                    ConstructionQuestionOption(
                        key="vault_managed",
                        label="Bóveda de secretos empresarial (Key Vault)",
                        description="Uso de un administrador de claves cifradas corporativo.",
                        impact="Cumple con las normativas más exigentes de ciberseguridad corporativa.",
                        example="Ejemplo: AWS Secrets Manager o Azure Key Vault."
                    )
                ]
            ),
        ],
    )


def _collect_deployment_gap(files: dict[str, ACPFileEntry]) -> ConstructionGapEntry | None:
    docker_compose = files.get("ACP/deployment/docker-compose.yaml")
    kubernetes = files.get("ACP/deployment/kubernetes/README.md")
    cicd = files.get("ACP/deployment/cicd/README.md")
    env_template = files.get(ACP_CANONICAL_ENV_TEMPLATE_PATH)
    return _gap(
        gap_key="deployment_target_unknown",
        title="El entorno de despliegue aun no esta decidido",
        domain="deployment",
        severity="warning",
        blocking_stage="deployment_design",
        summary="El ACP contiene deployment base, pero aun no define el entorno objetivo, la estrategia de imagen ni la operacion final.",
        evidence_paths=[item.path for item in [docker_compose, env_template, kubernetes, cicd] if item is not None],
        source_sections=sorted(
            {
                section
                for item in [docker_compose, env_template, kubernetes, cicd]
                if item is not None
                for section in item.source_sections
            }
        ),
        current_assumptions=["Se conserva un baseline local-first hasta definir el entorno real de despliegue."],
        closure_criteria=[
            "Definir entorno objetivo de despliegue.",
            "Definir estrategia de build/publicacion de imagenes o justificar otra via.",
            "Definir CI/CD o proceso operativo equivalente.",
        ],
        questions=[
            _question(
                question_key="deployment_target",
                question_text="¿En qué tipo de infraestructura se instalará y ejecutará el asistente?",
                rationale="Identificar el servidor o la nube donde vivirá la solución para preparar los paquetes de instalación.",
                purpose="Seleccionar el entorno de hospedaje óptimo para la operación continua.",
                expected_answer_format="target=<entorno>; restrictions=<restricciones>",
                target_owner="platform_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="cloud_container",
                        label="Servidor en la Nube o Contenedor (Cloud / Docker)",
                        description="Despliegue en servidores en la nube listos para escalar según la demanda.",
                        impact="Capacidad de atender a múltiples usuarios simultáneos sin caídas.",
                        example="Ejemplo: Servidores en AWS, Azure, Google Cloud o Render."
                    ),
                    ConstructionQuestionOption(
                        key="on_premise",
                        label="Servidores propios de la empresa (On-Premise)",
                        description="Instalación en la red privada o centro de datos de la organización.",
                        impact="Garantiza que toda la información permanezca dentro de la red corporativa.",
                        example="Ejemplo: Servidores físicos en las oficinas de la empresa."
                    ),
                    ConstructionQuestionOption(
                        key="local_desktop",
                        label="Equipo de escritorio local",
                        description="Instalación para uso personal o de pruebas en una computadora.",
                        impact="Ideal para validar y realizar pruebas antes del lanzamiento masivo.",
                        example="Ejemplo: Ejecución en laptop mediante Docker Desktop."
                    )
                ]
            ),
            _question(
                question_key="deployment_image_strategy",
                question_text="¿De qué manera prefieres empaquetar el asistente para su instalación?",
                rationale="El paquete de instalación determina la facilidad de despliegue y actualización.",
                purpose="Elegir la forma de distribución del código y sus componentes.",
                expected_answer_format="strategy=<mecanismo>",
                target_owner="devops_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="docker_image",
                        label="Imagen estandarizada lista para usar (Docker)",
                        description="Empaqueta la aplicación con todas sus dependencias incluidas.",
                        impact="Garantiza que funcione exactamente igual en cualquier servidor.",
                        example="Ejemplo: Contenedor desplegado con un solo clic."
                    ),
                    ConstructionQuestionOption(
                        key="python_package",
                        label="Código Python ejecutable",
                        description="Entrega del código estructurado listo para ejecutar en el servidor.",
                        impact="Permite inspeccionar y personalizar scripts directamente en el servidor.",
                        example="Ejemplo: Instalación en entorno virtual de Python."
                    )
                ]
            ),
            _question(
                question_key="deployment_network_constraints",
                question_text="¿Existen restricciones de seguridad o conexión a internet en el servidor?",
                rationale="Identificar bloqueos de red para solicitar los permisos necesarios antes de instalar.",
                purpose="Asegurar que el asistente pueda comunicarse con los servicios requeridos.",
                expected_answer_format="network=<restricciones>",
                target_owner="security_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="standard_internet",
                        label="Acceso a internet libre",
                        description="El servidor se comunica sin restricciones con servicios externos.",
                        impact="Permite conectar cualquier API o proveedor de IA inmediatamente.",
                        example="Ejemplo: Servidor web convencional conectado a la nube."
                    ),
                    ConstructionQuestionOption(
                        key="restricted_firewall",
                        label="Red protegida con Firewall / Proxy corporativo",
                        description="Solo se permite tráfico hacia sitios o puertos explícitamente aprobados.",
                        impact="Requerirá que el equipo de seguridad habilite los dominios requeridos.",
                        example="Ejemplo: Servidor en red bancaria o gubernamental."
                    )
                ]
            ),
        ],
    )


def _collect_external_api_gap(snapshot: SessionSnapshot, files: dict[str, ACPFileEntry]) -> ConstructionGapEntry | None:
    blueprint = snapshot.blueprint
    if blueprint is None or not blueprint.tools:
        return None

    tools = project_blueprint_tools_for_construction(snapshot)
    whatsapp_tools = [
        tool
        for tool in tools
        if resolve_whatsapp_connector_key(tool)
    ]
    google_workspace_tools = [
        tool
        for tool in tools
        if resolve_google_workspace_connector_key(tool)
    ]
    odoo_tools = [
        tool
        for tool in tools
        if resolve_odoo_connector_key(tool)
    ]
    google_keys = set().union(
        *[
            {resolve_google_workspace_connector_key(tool)}
            for tool in google_workspace_tools
        ]
    ) if google_workspace_tools else set()
    odoo_keys = set().union(
        *[
            {resolve_odoo_connector_key(tool)}
            for tool in odoo_tools
        ]
    ) if odoo_tools else set()
    if odoo_keys:
        odoo_keys = {
            ODOO_LEGACY_CONNECTOR_ALIASES.get(key, key)
            for key in odoo_keys
            if key in ODOO_CANONICAL_CONNECTOR_KEYS or key in ODOO_LEGACY_CONNECTOR_ALIASES
        }
        odoo_keys.update({"odoo_partner_read", "odoo_crm_lead_read"})
        if _snapshot_has_odoo_quote_signal(snapshot):
            odoo_keys.update({"odoo_sale_order_read", "odoo_sale_quote_create"})
    external_tool_paths = [
        build_tool_contract_path_for_tool(tool, index)
        for index, tool in enumerate(tools, start=1)
        if tool.name not in INTERNAL_BUILDER_TOOL_NAMES and getattr(tool, "tool_type", "external") != "internal"
    ]
    required_contracts = files.get("ACP/construction-readiness/required-api-contracts.yaml")
    if not external_tool_paths:
        return None
    unresolved_contracts = required_contracts is None or bool(required_contracts.warnings) or _contains_needs_review(required_contracts)
    if not unresolved_contracts:
        return None
    evidence_paths = external_tool_paths[:]
    if required_contracts is not None:
        evidence_paths.insert(0, required_contracts.path)
    questions = [
        _question(
            question_key="external_api_contracts",
            question_text="¿Qué otros sistemas o herramientas de tu empresa debe conectar el asistente?",
            rationale="Detalla qué aplicaciones externas consultará o modificará el asistente.",
            purpose="Establecer los enlaces seguros entre el asistente y tus herramientas actuales.",
            expected_answer_format="una línea por herramienta con tool=<herramienta>; system=<sistema>; endpoint=<ruta>",
            target_owner="integration_owner",
            blocking=False,
            options=[
                ConstructionQuestionOption(
                    key="standard_rest_api",
                    label="Servicios web estándar (API REST)",
                    description="Conexión limpia a través de servicios web con clave de API.",
                    impact="Integración ágil con software moderno como CRMs, ERPs o mensajería.",
                    example="Ejemplo: Enviar un mensaje por WhatsApp o crear un ticket en Jira."
                ),
                ConstructionQuestionOption(
                    key="custom_database",
                    label="Conexión directa a base de datos de la empresa",
                    description="Acceso a tablas específicas para leer o guardar registros.",
                    impact="Acceso a datos históricos en tiempo real sin requerir APIs adicionales.",
                    example="Ejemplo: Consultar la tabla de clientes en SQL Server."
                ),
                ConstructionQuestionOption(
                    key="none",
                    label="Sin conexiones externas por el momento",
                    description="El asistente funcionará de forma independiente sin conectarse a otros sistemas.",
                    impact="Despliegue inmediato sin requerir permisos de integración.",
                    example="Ejemplo: Asistente independiente para consultas de manuales."
                )
            ]
        )
    ]
    if whatsapp_tools:
        questions.append(
            _question(
                question_key="whatsapp_activation_context",
                question_text=(
                    "¿Cuáles son los datos de activación de WhatsApp Business Cloud API para sandbox y producción?"
                ),
                rationale=(
                    "LAB entrega el diseño del webhook; solo faltan datos que dependen del entorno real del cliente."
                ),
                purpose="Permitir que el builder implemente y pruebe WhatsApp sin inventar credenciales ni URLs.",
                expected_answer_format=(
                    "provider=<meta_cloud_api|partner>; waba_id=<id>; phone_number_id=<id>; "
                    "callback_url_sandbox=<url>; callback_url_production=<url>; deployment_target=<target>; "
                    "access_token_ref=<secret_ref>; verify_token_ref=<secret_ref>; app_secret_ref=<secret_ref>; "
                    "templates=<lista>; opt_in_source=<fuente>; human_handoff=<equipo/canal>"
                ),
                target_owner="integration_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="meta_cloud_api_direct",
                        label="Meta Cloud API directa",
                        description="El equipo configurará app, WABA, número, callback URL y secretos en Meta.",
                        impact="Permite implementar el webhook y sender con el contrato ACP.",
                        example="provider=meta_cloud_api; deployment_target=render; callback_url_production=https://api.example.com/webhooks/whatsapp"
                    ),
                    ConstructionQuestionOption(
                        key="business_solution_provider",
                        label="Proveedor intermediario",
                        description="Twilio, 360dialog, Zenvia, WATI u otro proveedor gestionará parte de la integración.",
                        impact="El builder adapta el contrato ACP al proveedor seleccionado.",
                        example="provider=twilio; callback_url_production=https://api.example.com/webhooks/whatsapp"
                    ),
                    ConstructionQuestionOption(
                        key="unknown",
                        label="Pendiente por definir",
                        description="Aún no se conoce el proveedor, URL pública o secret store.",
                        impact="Se puede construir el contrato, pero no activar sandbox ni producción.",
                        example="provider=unknown; deployment_target=unknown"
                    ),
                ],
            )
        )
    if google_workspace_tools:
        questions.append(
            _question(
                question_key="google_workspace_activation_context",
                question_text="¿Cuál será la configuración OAuth y política de scopes para Google Workspace en sandbox y producción?",
                rationale="LAB solo entrega contratos ACP; OAuth, consent screen, redirect URI y vault pertenecen al proyecto destino.",
                purpose="Evitar que el builder invente client IDs, scopes o tokens y conservar el ciclo de vida de configuración.",
                expected_answer_format=(
                    "oauth_project=<id/nombre>; consent_screen=<internal|external|pending>; "
                    "redirect_uri_sandbox=<url>; redirect_uri_production=<url>; "
                    "client_id_ref=<secret_ref>; client_secret_ref=<secret_ref>; refresh_token_ref=<secret_ref>; "
                    "allowed_scopes=<lista>; secret_store=<vault/env>; owner=<persona/equipo>"
                ),
                target_owner="integration_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="client_google_project",
                        label="Proyecto Google del cliente",
                        description="El cliente administra OAuth, consent screen y secretos.",
                        impact="Mantiene separación clara entre LAB y el runtime construido.",
                        example="oauth_project=cliente-prod; consent_screen=external; secret_store=render_env"
                    ),
                    ConstructionQuestionOption(
                        key="builder_creates_project",
                        label="Builder crea configuración",
                        description="El equipo constructor guía la creación de OAuth en una cuenta controlada por el cliente.",
                        impact="Requiere instrucciones paso a paso y validación manual del owner.",
                        example="oauth_project=pending; owner=cliente_admin_google"
                    ),
                    ConstructionQuestionOption(
                        key="deferred",
                        label="Pendiente por definir",
                        description="Se conserva como decisión delegada antes de construir el binding.",
                        impact="El ACP puede exportarse, pero no se activa Google Workspace.",
                        example="oauth_project=pending; allowed_scopes=pending"
                    ),
                ],
            )
        )
    if "google_drive_file_picker" in google_keys:
        questions.append(
            _question(
                question_key="google_drive_resource_scope",
                question_text="¿Qué archivos, carpetas, tipos MIME y modo de refresco de Google Drive puede usar el agente?",
                rationale="Drive debe limitarse a archivos seleccionados o permitidos; no conviene asumir lectura amplia.",
                purpose="Definir recursos permitidos y evitar acceso excesivo a Drive.",
                expected_answer_format="files=<ids/lista>; folders=<ids/lista>; mime_types=<lista>; max_size_mb=<n>; refresh=<manual|on_demand|scheduled>; owner=<persona>",
                target_owner="knowledge_owner",
                blocking=False,
            )
        )
    if "google_sheets_read_table" in google_keys:
        questions.append(
            _question(
                question_key="google_sheets_table_contract",
                question_text="¿Cuál es el spreadsheet, rango, columnas y llave primaria que debe leer el agente en Google Sheets?",
                rationale="Una hoja puede ser útil como fuente inicial, pero sin esquema produce lecturas frágiles.",
                purpose="Convertir la hoja en contrato tabular determinístico antes de construir.",
                expected_answer_format="spreadsheet_id=<id>; range=<A1>; header_row=<n>; primary_key=<columna>; columns=<lista>; cache=<politica>; owner=<persona>",
                target_owner="data_owner",
                blocking=False,
            )
        )
    if {"google_calendar_availability_reader", "google_calendar_event_creator"} & google_keys:
        questions.append(
            _question(
                question_key="google_calendar_booking_policy",
                question_text="¿Qué calendario, zona horaria, ventanas y reglas de aprobación aplican para consultar o crear eventos?",
                rationale="Consultar disponibilidad es lectura; crear eventos tiene side effects y requiere política clara.",
                purpose="Separar disponibilidad, creación de citas, aprobación e idempotencia.",
                expected_answer_format="calendar_id=<id>; timezone=<tz>; availability_window=<regla>; slot_minutes=<n>; attendee_policy=<regla>; approval_policy=<regla>; conflict_policy=<regla>",
                target_owner="ops_owner",
                blocking=False,
            )
        )
    if {"gmail_draft_creator", "gmail_send_message"} & google_keys:
        questions.append(
            _question(
                question_key="gmail_message_policy",
                question_text="¿Desde qué cuenta Gmail se crearán borradores o envíos, y qué política de destinatarios/aprobación aplica?",
                rationale="Gmail puede exponer datos sensibles y enviar comunicaciones; el MVP debe preferir borradores si no hay política explícita.",
                purpose="Definir sender, destinatarios permitidos, retención y aprobación antes de activar Gmail.",
                expected_answer_format="sender_account=<email>; mode=<draft|send>; recipient_policy=<regla>; approval_policy=<regla>; retention=<regla>; owner=<persona>",
                target_owner="communications_owner",
                blocking=False,
            )
        )
    if odoo_tools:
        questions.append(
            _question(
                question_key="odoo_version_context",
                question_text="¿Qué versión de Odoo se integrará y qué modo API debe usar el builder?",
                rationale="Odoo 17/18 suelen construirse con External API XML-RPC/JSON-RPC; Odoo 19 puede requerir JSON-2. LAB no debe inferirlo.",
                purpose="Seleccionar el contrato tecnico correcto antes de construir adaptadores Odoo.",
                expected_answer_format="version=<17|18|19|otra>; edition=<community|enterprise|online|sh>; api_mode=<xmlrpc_17_18|json2_19>; hosting=<url/entorno>; owner=<persona>",
                target_owner="integration_owner",
                blocking=False,
                options=[
                    ConstructionQuestionOption(
                        key="odoo_17_18_rpc",
                        label="Odoo 17/18 RPC",
                        description="Construir con execute_kw sobre la External API tradicional.",
                        impact="El ACP usará el contrato RPC 17/18 y sus pruebas de autenticación/modelos.",
                        example="version=17; edition=community; api_mode=xmlrpc_17_18; hosting=https://odoo.example.com"
                    ),
                    ConstructionQuestionOption(
                        key="odoo_19_json2",
                        label="Odoo 19 JSON-2",
                        description="Construir con JSON-2 si el Odoo destino lo expone para los modelos requeridos.",
                        impact="El ACP usará política de versión Odoo 19 y validará endpoints por modelo.",
                        example="version=19; api_mode=json2_19; hosting=https://odoo.example.com"
                    ),
                    ConstructionQuestionOption(
                        key="unknown",
                        label="Pendiente por definir",
                        description="Se conserva como decisión delegada antes de activar sandbox.",
                        impact="El builder no debe construir llamadas reales hasta confirmar versión y modo API.",
                        example="version=pending; api_mode=pending"
                    ),
                ],
            )
        )
        questions.append(
            _question(
                question_key="odoo_api_access_context",
                question_text="¿Cuáles son las referencias de acceso Odoo para sandbox y producción?",
                rationale="La integración necesita base URL, database y usuario técnico, pero el ACP no debe almacenar secretos planos.",
                purpose="Evitar credenciales inventadas y mantener secret refs dentro del ciclo de vida de configuración.",
                expected_answer_format="base_url=<url>; database=<db>; username_ref=<secret_ref>; password_ref=<secret_ref>; api_key_ref=<secret_ref>; secret_store=<vault/env>; sandbox=<si/no>; owner=<persona>",
                target_owner="integration_owner",
                blocking=False,
            )
        )
        questions.append(
            _question(
                question_key="odoo_module_scope",
                question_text="¿Qué módulos, modelos, campos y dominios de Odoo puede usar el agente?",
                rationale="Odoo suele tener campos custom y permisos por módulo; leer o escribir sin allowlist puede romper el proceso comercial.",
                purpose="Definir allowlist de modelos/campos/dominios antes de construir consultas.",
                expected_answer_format="models=<lista>; fields_by_model=<mapa>; domains=<reglas>; custom_modules=<lista>; owner=<persona>",
                target_owner="process_owner",
                blocking=False,
            )
        )
    if {"odoo_sale_quote_create", "odoo_activity_create", "odoo_crm_lead_update"} & odoo_keys:
        questions.append(
            _question(
                question_key="odoo_write_policy",
                question_text="¿Qué escrituras en Odoo están permitidas y qué aprobación exige cada una?",
                rationale="Crear o modificar registros Odoo tiene side effects comerciales; debe pasar por approval, allowlist e idempotencia.",
                purpose="Separar lecturas determinísticas de acciones con impacto operativo.",
                expected_answer_format="allowed_write_actions=<lista>; approval_policy=<regla>; idempotency=<regla>; rollback_or_escalation=<regla>; owner=<persona>",
                target_owner="business_owner",
                blocking=False,
            )
        )
    if "odoo_sale_quote_create" in odoo_keys:
        questions.append(
            _question(
                question_key="odoo_quote_policy",
                question_text="¿Cuál es la política comercial para crear cotizaciones en Odoo?",
                rationale="Una cotización depende de cliente, productos, lista de precios, impuestos, descuentos y vencimiento; el agente no debe inventarlos.",
                purpose="Hacer construible la creación de cotizaciones sin asumir reglas comerciales.",
                expected_answer_format="pricelist=<id/regla>; currency=<moneda>; taxes=<regla>; discount_policy=<regla>; expiration=<regla>; confirm_order=<si/no>; owner=<persona>",
                target_owner="sales_owner",
                blocking=False,
            )
        )
    return _gap(
        gap_key="external_api_contracts_missing",
        title="Faltan contratos operativos de APIs o sistemas externos",
        domain="integrations",
        severity="warning",
        blocking_stage="external_integration",
        summary="Existen tools que parecen depender de sistemas externos y el ACP aun no describe sus contratos de integracion.",
        evidence_paths=evidence_paths,
        source_sections=["blueprint.tools", "integration_statuses"],
        current_assumptions=["Las tools externas requieren contratos adicionales antes de implementarse contra sistemas reales."],
        closure_criteria=[
            "Definir endpoint o accion requerida por cada sistema externo.",
            "Definir autenticacion, payloads y errores esperados.",
            "Definir limites operativos o retries si aplican.",
        ],
        questions=questions,
    )


def _latest_memory_artifact(snapshot: SessionSnapshot) -> MemoryRecommendationArtifact | None:
    latest = snapshot.journey_latest_artifacts.get("memory") if snapshot.journey_latest_artifacts else None
    if latest is None or not latest.proposal_payload:
        return None
    try:
        return MemoryRecommendationArtifact.model_validate(latest.proposal_payload)
    except Exception:  # noqa: BLE001
        return None


def _memory_dependency_question(gap: MemoryDependencyGap) -> ConstructionQuestionEntry:
    capability = gap.capability_key.strip() or gap.gap_key.split(":", 1)[-1]
    reason = gap.reason.strip() or "Memoria declaro una dependencia de herramienta no aprobada en Tools."
    return _question(
        question_key=gap.gap_key or f"memory_dependency:{capability}",
        question_text=(
            f"Como quieres tratar la dependencia de memoria `{capability}` durante la construccion ACP?"
        ),
        rationale=reason,
        purpose=(
            "Mantener la aprobacion LEAN separada de decisiones tecnicas de implementacion y conservar "
            "trazabilidad hasta el ACP."
        ),
        expected_answer_format=(
            "Elige defer_to_acp o exclude_from_mvp y agrega una nota con owner, contrato afectado o criterio de cierre."
        ),
        target_owner="solution_owner",
        blocking=False,
        options=[
            ConstructionQuestionOption(
                key="defer_to_acp",
                label="Diferir al ACP",
                description="Conservar la dependencia como decision de implementacion para que el builder la cierre.",
                impact="El ACP mantiene la pregunta abierta sin bloquear la aprobacion de Memoria.",
                example=f"defer_to_acp; owner=integration_owner; dependency={capability}",
                recommended=True,
                source_refs=list(gap.source_refs),
            ),
            ConstructionQuestionOption(
                key="exclude_from_mvp",
                label="Excluir del MVP",
                description="Marcar que esta dependencia no se construira en el alcance inicial.",
                impact="El builder debe ajustar memoria, prompts o flujo para operar sin esa dependencia.",
                example=f"exclude_from_mvp; dependency={capability}; reason=fuera_de_alcance",
                source_refs=list(gap.source_refs),
            ),
        ],
        question_kind="memory_dependency_resolution",
        subject_type="memory_dependency",
        subject_id=gap.gap_key or capability,
        allowed_decisions=["defer_to_acp", "exclude_from_mvp"],
        answer_semantics="resolve_memory_dependency_gap",
        contract_version=1,
    )


def _collect_memory_dependency_gap(snapshot: SessionSnapshot) -> ConstructionGapEntry | None:
    memory = _latest_memory_artifact(snapshot)
    if memory is None:
        return None
    open_gaps = [
        gap
        for gap in memory.dependency_gaps
        if gap.status == "open" and (gap.required or gap.remediation_policy in {"human_review", "implementation_pending"})
    ]
    if not open_gaps:
        return None
    questions = [_memory_dependency_question(gap) for gap in open_gaps]
    return _gap(
        gap_key="memory_dependency_questions",
        title="Dependencias de memoria pendientes para construccion ACP",
        domain="memory",
        severity="warning",
        blocking_stage="acp_questions_resolution",
        summary=(
            "Memoria detecto dependencias requeridas que no deben bloquear LEAN; deben resolverse, diferirse "
            "o excluirse como preguntas ACP con trazabilidad."
        ),
        evidence_paths=[
            "ACP/memory/strategy.yaml",
            "ACP/memory/lifecycle.yaml",
            "ACP/construction-readiness/open-questions.yaml",
        ],
        source_sections=["memory.dependency_gaps", "memory.tool_dependencies"],
        current_assumptions=[gap.reason for gap in open_gaps if gap.reason],
        closure_criteria=[
            "Cada dependencia abierta debe quedar diferida al ACP o excluida del MVP.",
            "La decision debe conservar `question_key`, owner y artefactos ACP impactados.",
            "El builder no debe inventar integraciones, credenciales ni herramientas faltantes.",
        ],
        questions=questions,
    )


def _collect_consistency_gap(snapshot: SessionSnapshot) -> ConstructionGapEntry | None:
    report = ensure_blueprint_consistency_report(snapshot)
    actionable_issues = [
        issue
        for issue in report.issues
        if not is_blueprint_handoff_process_debt_issue(issue.issue_key)
        and issue.severity in {"blocking", "warning"}
    ]
    if not actionable_issues:
        return None

    blocking_issues = [issue for issue in actionable_issues if issue.severity == "blocking"]
    warning_issues = [issue for issue in actionable_issues if issue.severity == "warning"]
    severity = "blocking" if blocking_issues else "warning"
    summary = (
        "El package detecto deuda real de coherencia entre Requirement, Design, Tools, Memory, Validate y Estimate."
    )
    remediation = (
        "Resolver bloqueos reales o delegar decisiones implementables; no reabrir fases estables por deuda operativa interna."
    )
    closure_criteria = [
        "Confirmar que los issues bloqueantes restantes comprometen la integridad del Blueprint.",
        "Registrar como decision delegada aquello que pueda resolverse durante implementacion.",
        "Mantener fuera del ACP los flags stale, warnings de sincronizacion y estados transitorios del Blueprint.",
    ]
    assumptions = [issue.detail for issue in (blocking_issues + warning_issues)[:3]]
    return ConstructionGapEntry(
        gap_key="cross_stage_consistency_drift",
        title="La cadena aprobada no esta coherente extremo a extremo",
        domain="consistency",
        severity=severity,  # type: ignore[arg-type]
        status="open",
        blocking_stage="package",
        summary=summary,
        remediation=remediation,
        evidence_paths=[
            "ACP/governance/consistency-report.json",
            "ACP/governance/approved-stage-lineage.yaml",
            "ACP/governance/journey-decisions.json",
        ],
        source_sections=["blueprint_consistency", "journey_artifacts", "estimation_report"],
        current_assumptions=assumptions,
        closure_criteria=closure_criteria,
        questions=[
            _question(
                question_key="cross_stage_consistency_drift_resolution",
                question_text=(
                    "Como quieres cerrar la deuda de coherencia detectada entre etapas antes de empaquetar el ACP?"
                ),
                rationale=(
                    "El ACP no debe bloquearse sin una accion clara. Esta decision permite confirmar si el drift "
                    "requiere regeneracion, puede delegarse a implementacion o debe mantenerse como bloqueo real."
                ),
                purpose="Dar una salida accionable al bloqueo de consistencia sin reabrir fases estables automaticamente.",
                expected_answer_format=(
                    "Elige una opcion y agrega una nota breve con la decision: regenerar, delegar a implementacion o mantener bloqueo."
                ),
                target_owner="solution_owner",
                blocking=severity == "blocking",
                options=[
                    ConstructionQuestionOption(
                        key="delegate_to_implementation",
                        label="Delegar a implementacion",
                        description="Registrar la deuda como decision implementable dentro del ACP.",
                        impact="Permite continuar si el issue no compromete la integridad del Blueprint aprobado.",
                        example="Delegar la validacion final del scoring ICP al builder durante implementacion.",
                        recommended=True,
                        source_refs=["blueprint_consistency.issues"],
                    ),
                    ConstructionQuestionOption(
                        key="regenerate_affected_artifacts",
                        label="Regenerar artefactos afectados",
                        description="Reprocesar las piezas impactadas antes de empaquetar.",
                        impact="Mantiene el bloqueo hasta que los artefactos afectados queden sincronizados.",
                        example="Regenerar Tools y readiness ACP con el digest aprobado actualizado.",
                        source_refs=["journey_artifacts", "blueprint_consistency.issues"],
                    ),
                    ConstructionQuestionOption(
                        key="keep_blocking",
                        label="Mantener bloqueo",
                        description="Confirmar que la inconsistencia impide construir el agente.",
                        impact="El ACP debe detenerse hasta resolver la coherencia extremo a extremo.",
                        example="No continuar porque falta una tool obligatoria para un requisito high.",
                        source_refs=["blueprint_consistency.issues"],
                    ),
                ],
                question_kind="consistency_resolution",
                subject_type="blueprint_consistency",
                subject_id="cross_stage_consistency_drift",
                allowed_decisions=["answer", "choose_option", "delegate", "dismiss"],
                answer_semantics="resolve_or_delegate_consistency_gap",
            )
        ],
    )


def _flatten_assumptions(gaps: Iterable[ConstructionGapEntry]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for gap in gaps:
        for item in gap.current_assumptions:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
    return ordered


def build_initial_construction_readiness(
    snapshot: SessionSnapshot,
    files: list[ACPFileEntry],
    validation: ACPValidationReport,
) -> ConstructionReadinessReport:
    if not files:
        return ConstructionReadinessReport(
            overall_status="not_started",
            can_start_build=False,
            blocking_gaps=0,
            open_questions=0,
            assumptions_count=0,
            gaps=[],
            next_recommended_action="generate_acp_preview",
        )

    mapped_files = _file_map(files)
    gaps: list[ConstructionGapEntry] = []

    for candidate in [
        _collect_validation_gap(validation),
        _collect_objective_validation_gap(snapshot),
        _collect_knowledge_gap(mapped_files),
        _collect_runtime_gap(snapshot, mapped_files),
        _collect_deployment_gap(mapped_files),
        _collect_external_api_gap(snapshot, mapped_files),
        _collect_memory_dependency_gap(snapshot),
        _collect_consistency_gap(snapshot),
    ]:
        if candidate is not None:
            gaps.append(candidate)

    blocking_gaps = sum(1 for item in gaps if item.severity == "blocking" and item.status not in {"answered", "resolved"})
    open_questions = sum(len(item.questions) for item in gaps if item.status == "open")
    assumptions = _flatten_assumptions(gaps)
    can_start_build = validation.can_export_zip and blocking_gaps == 0

    if can_start_build:
        overall_status = "ready_to_build"
        next_action = "start_agentic_build"
    elif blocking_gaps > 0:
        overall_status = "blocked"
        next_action = "resolve_blocking_construction_gaps"
    else:
        overall_status = "needs_questions"
        next_action = "answer_open_questions"

    return ConstructionReadinessReport(
        overall_status=overall_status,
        can_start_build=can_start_build,
        blocking_gaps=blocking_gaps,
        open_questions=open_questions,
        assumptions_count=len(assumptions),
        gaps=gaps,
        next_recommended_action=next_action,
    )
