from __future__ import annotations

from collections.abc import Iterable

from app.models import (
    ACPFileEntry,
    ACPValidationReport,
    ConstructionGapEntry,
    ConstructionQuestionEntry,
    ConstructionQuestionOption,
    ConstructionReadinessReport,
    SessionSnapshot,
)
from app.services.blueprint_consistency_service import ensure_blueprint_consistency_report
from app.services.acp_paths import ACP_CANONICAL_ENV_TEMPLATE_PATH, build_tool_contract_path_for_tool


INTERNAL_BUILDER_TOOL_NAMES = {
    "normalize_discovery",
    "build_canvas",
    "build_blueprint",
    "promote_blueprint_for_implementation",
}

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
}

BLUEPRINT_HANDOFF_PROCESS_DEBT_ISSUE_KEYS = {
    "tools_recommendation_stale",
    "memory_recommendation_stale",
    "estimate_stale",
}
BLUEPRINT_HANDOFF_PROCESS_DEBT_PREFIXES = (
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
    external_tool_paths = [
        build_tool_contract_path_for_tool(tool, index)
        for index, tool in enumerate(blueprint.tools, start=1)
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
        questions=[
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
        ],
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
        questions=[],
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
        _collect_knowledge_gap(mapped_files),
        _collect_runtime_gap(snapshot, mapped_files),
        _collect_deployment_gap(mapped_files),
        _collect_external_api_gap(snapshot, mapped_files),
        _collect_consistency_gap(snapshot),
    ]:
        if candidate is not None:
            gaps.append(candidate)

    blocking_gaps = sum(1 for item in gaps if item.severity == "blocking" and item.status not in {"answered", "resolved"})
    open_questions = sum(len(item.questions) for item in gaps if item.status == "open")
    assumptions = _flatten_assumptions(gaps)
    can_start_build = validation.can_export_zip and blocking_gaps == 0 and open_questions == 0

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
