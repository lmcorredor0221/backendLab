from __future__ import annotations

import re
from typing import Any

from app.models import AttentionItemV2
from app.services.attention.decision_contract import (
    AttentionDecisionActionV3,
    AttentionDecisionOptionV3,
    AttentionDecisionSourceV3,
    AttentionDecisionV3,
    decision_to_attention_item_v2,
)


_CAPABILITY_LABELS = {
    "normalize_discovery": "normalizar Descubrir",
    "analyze_discovery": "analizar Descubrir",
    "build_canvas": "construir el canvas",
    "define_requirements": "generar Definir",
    "requirements_definition_skill": "generar Definir",
    "synthesize_blueprint_narrative": "generar el Blueprint",
    "propose_agent_design": "generar Disenar",
    "critique_agent_design": "evaluar Disenar",
    "recommend_minimal_tools": "recomendar Herramientas",
    "recommend_memory_architecture": "generar Memoria",
    "critique_memory_architecture": "evaluar Memoria",
    "generate_validation_scenarios": "generar Validar",
    "analyze_estimation_risks": "generar Estimar",
}

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)\s*[:=]\s*([^\s;,\"]+)"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _get(source: Any, key: str, default: Any = "") -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _capability_from_text(text: str) -> str:
    normalized = text.strip()
    candidates = [
        match.group(1)
        for pattern in (
            r"ejecutar\s+([a-z0-9_]+)",
            r"for\s+([a-z0-9_]+)",
            r"/([a-z0-9-]+)\s+timed\s+out",
        )
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE)
    ]
    for candidate in candidates:
        key = candidate.replace("-", "_").lower()
        if key in _CAPABILITY_LABELS:
            return key
    for key in _CAPABILITY_LABELS:
        if key in normalized.lower():
            return key
    return ""


def _issue_kind(text: str) -> str:
    normalized = text.lower()
    if "timed out" in normalized or "timeout" in normalized or "time out" in normalized:
        return "timeout"
    if "rate limit" in normalized or "too many requests" in normalized:
        return "rate_limit"
    if "policy=needs_review_on_provider_or_schema_failure" in normalized:
        return "provider_or_schema"
    if "schema" in normalized or "provider" in normalized:
        return "provider_or_schema"
    if "codex local" in normalized and "no pudo" in normalized:
        return "local_runtime"
    return "runtime"


def _sanitize_error_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return ""
    for pattern in _SECRET_PATTERNS:
        normalized = pattern.sub(lambda match: f"{match.group(1) if match.lastindex and match.lastindex > 1 else 'secret'}=<redacted>", normalized)
    return normalized[:900]


def _runtime_title(stage: str, capability: str, issue_kind: str) -> str:
    action = _CAPABILITY_LABELS.get(capability)
    if issue_kind == "timeout":
        return f"La generacion de {stage.title()} tardo mas de lo esperado"
    if action:
        return f"No se pudo {action} automaticamente"
    if stage and stage != "runtime":
        return f"La etapa {stage.title()} necesita una recuperacion guiada"
    return "La operacion inteligente necesita una recuperacion guiada"


def _runtime_reason(issue_kind: str) -> str:
    if issue_kind == "timeout":
        return (
            "La ejecucion supero el tiempo esperado antes de entregar una respuesta util. "
            "El contexto aprobado se conserva y puedes reintentar sin perder informacion."
        )
    if issue_kind == "rate_limit":
        return (
            "El proveedor limito temporalmente la solicitud. Conviene reintentar o cambiar a una ruta de respaldo "
            "sin pedir informacion tecnica al usuario final."
        )
    if issue_kind == "provider_or_schema":
        return (
            "El proveedor LLM o el esquema de respuesta no entrego una salida valida para la etapa. "
            "Esto es un asunto operativo del runtime, no una pregunta funcional para el usuario."
        )
    if issue_kind == "local_runtime":
        return (
            "El runtime local no pudo completar la tarea. La plataforma debe ofrecer reintento, respaldo seguro "
            "o revision de configuracion sin exponer detalles tecnicos como accion principal."
        )
    return "El runtime necesita una accion controlada para continuar con trazabilidad."


def _runtime_repair_hint(issue_kind: str, capability_label: str) -> str:
    target = capability_label or "la capacidad de la etapa"
    if issue_kind == "timeout":
        return (
            f"Reintentar {target} con el mismo contexto aprobado. Si se repite, aumentar timeout "
            "o reducir el contexto enviado al proveedor."
        )
    if issue_kind == "rate_limit":
        return (
            "Esperar unos minutos, validar cuota/rate limit del proveedor o usar el modelo de respaldo configurado."
        )
    if issue_kind == "provider_or_schema":
        return (
            f"Validar proveedor/modelo activo, contrato JSON de salida y schema de {target}. "
            "Si el proveedor respondio parcialmente, revisar la traza antes de reintentar."
        )
    if issue_kind == "local_runtime":
        return (
            "Confirmar que Codex/local runtime este disponible, que el proceso tenga permisos y que el proveedor "
            "configurado pueda ejecutar la capability solicitada."
        )
    return "Revisar la traza operativa, corregir la causa tecnica y reintentar la etapa."


def _runtime_summary(issue_kind: str, stage: str, capability_label: str) -> str:
    target = capability_label or f"la etapa {stage.title()}"
    if issue_kind == "timeout":
        return f"{target} no entrego respuesta antes del limite de tiempo."
    if issue_kind == "rate_limit":
        return f"El proveedor limito temporalmente la ejecucion de {target}."
    if issue_kind == "provider_or_schema":
        return f"{target} fallo porque el proveedor o el esquema no entrego una salida valida."
    if issue_kind == "local_runtime":
        return f"El runtime local no pudo ejecutar {target}."
    return f"{target} fallo en el runtime y requiere recuperacion controlada."


def runtime_operation_to_attention_item(
    operation: Any,
    *,
    href: str,
    return_href: str,
) -> AttentionItemV2:
    operation_id = _value(_get(operation, "id", "")) or _value(_get(operation, "operation_id", "")) or "operation"
    stage = _value(_get(operation, "stage", "")) or "runtime"
    product = _value(_get(operation, "product", "")) or "blueprint"
    raw_title = _value(_get(operation, "title", ""))
    raw_message = _value(_get(operation, "message", ""))
    raw_text = " ".join(part for part in (raw_title, raw_message) if part)
    capability = _capability_from_text(raw_text)
    issue_kind = _issue_kind(raw_text)
    capability_label = _CAPABILITY_LABELS.get(capability, "")
    title = _runtime_title(stage, capability, issue_kind)
    source_refs = [f"runtime.operation:{operation_id}"]
    if capability:
        source_refs.append(f"runtime.capability:{capability}")
    options = [
        AttentionDecisionOptionV3(
            key="retry_generation",
            label="Reintentar generacion",
            description="Ejecutar nuevamente la etapa con el mismo contexto aprobado.",
            impact="Puede recuperar la salida sin pedir datos adicionales al usuario.",
            example="Reintentar Definir luego de un fallo temporal del proveedor.",
            recommended=True,
            confidence=0.78,
            source_refs=source_refs,
        ),
        AttentionDecisionOptionV3(
            key="use_safe_fallback",
            label="Usar respaldo seguro",
            description="Continuar con una propuesta deterministica marcada como evidencia de respaldo.",
            impact="Evita bloquear el flujo, pero debe revisarse antes de aprobar.",
            example="Generar una base minima de requisitos desde el canvas aprobado.",
            recommended=False,
            confidence=0.62,
            source_refs=source_refs,
        ),
    ]
    if issue_kind in {"provider_or_schema", "local_runtime", "rate_limit"}:
        options.append(
            AttentionDecisionOptionV3(
                key="review_runtime_configuration",
                label="Revisar configuracion LLM",
                description="Enviar la incidencia a configuracion del proveedor, modelo o esquema.",
                impact="Corrige la causa raiz sin convertirla en pregunta del flujo LEAN.",
                example="Validar proveedor, modelo activo, timeout o contrato de salida.",
                recommended=False,
                confidence=0.66,
                source_refs=source_refs,
            )
        )
    decision = AttentionDecisionV3(
        decision_key="",
        item_type="runtime_error",
        severity="blocking",
        title=title,
        reason=_runtime_reason(issue_kind),
        impact="La etapa no deberia avanzar hasta recuperar o aceptar una salida trazable.",
        consequence_if_unresolved="El flujo permanecera detenido o con salida desactualizada hasta ejecutar una recuperacion.",
        required_decision="Seleccionar una ruta de recuperacion operacional.",
        suggested_answer="Reintentar generacion con el mismo contexto aprobado.",
        diagnostics={
            "summary": _runtime_summary(issue_kind, stage, capability_label),
            "technical_message": _sanitize_error_text(raw_text),
            "error_kind": issue_kind,
            "capability": capability,
            "capability_label": capability_label,
            "operation_id": operation_id,
            "retry_policy": (
                "El runtime debe intentar recuperacion automatica para fallos recuperables. "
                "Si el error llega a Atencion, el reintento queda trazado y puede reanudar desde checkpoint cuando exista."
            ),
            "repair_hint": _runtime_repair_hint(issue_kind, capability_label),
            "trace_refs": source_refs,
        },
        source=AttentionDecisionSourceV3(
            product=product,  # type: ignore[arg-type]
            stage=stage,
            source="runtime_operation",
            artifact_id="operation",
            entity_id=operation_id,
            field_path="runtime_issue",
            href=href,
            return_href=return_href,
            owner_role=_value(_get(operation, "owner_role", "")) or "operator",
            affected_artifact_refs=source_refs,
        ),
        options=options,
        action=AttentionDecisionActionV3(
            primary_kind="retry",
            primary_label="Reintentar recuperacion",
            can_resolve_inline=True,
            allowed_kinds=["retry", "answer", "defer"],
        ),
    )
    return decision_to_attention_item_v2(decision)
