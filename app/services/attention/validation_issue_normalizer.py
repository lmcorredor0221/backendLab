from __future__ import annotations

from collections.abc import Iterable

from app.models import AttentionItemV2
from app.services.attention.decision_contract import (
    AttentionDecisionActionV3,
    AttentionDecisionOptionV3,
    AttentionDecisionSourceV3,
    AttentionDecisionV3,
    decision_to_attention_item_v2,
)

_VALIDATION_PREFIXES = {
    "missing_acceptance",
    "untraced_item",
    "vague_nfr",
    "blocking_question",
    "duplicate_key",
    "canvas_scope_conflict",
}


def _split_issue(issue: str) -> tuple[str, str]:
    normalized = str(issue or "").strip()
    if ":" not in normalized:
        return normalized, ""
    prefix, detail = normalized.split(":", 1)
    return prefix.strip(), detail.strip()


def is_validation_issue_code(issue: str) -> bool:
    prefix, _ = _split_issue(issue)
    return prefix in _VALIDATION_PREFIXES


def split_validation_issue_codes(issues: Iterable[str]) -> tuple[list[str], list[str]]:
    validation_codes: list[str] = []
    plain_items: list[str] = []
    for issue in issues:
        text = str(issue or "").strip()
        if not text:
            continue
        if is_validation_issue_code(text):
            validation_codes.append(text)
        else:
            plain_items.append(text)
    return validation_codes, plain_items


def _label(target: str) -> str:
    return target or "el item detectado"


def _issue_copy(prefix: str, detail: str) -> dict[str, str]:
    target = _label(detail)
    if prefix == "missing_acceptance":
        return {
            "type": "question",
            "severity": "warning",
            "title": f"Criterio de aceptacion para {target}",
            "reason": f"El requerimiento {target} no tiene un criterio verificable para validar su cumplimiento.",
            "impact": "Sin criterio medible, la aprobacion del Blueprint puede ser subjetiva.",
            "consequence": "El Blueprint conservara incertidumbre sobre como validar o aceptar este componente.",
            "suggested": f"Definir un criterio medible para {target} con condicion de exito y evidencia esperada.",
        }
    if prefix == "untraced_item":
        return {
            "type": "validation",
            "severity": "warning",
            "title": f"Trazabilidad de requerimiento {target}",
            "reason": f"El item {target} no esta conectado con una fuente o decision aprobada de Discover o Canvas.",
            "impact": "Reduce auditabilidad del Blueprint y puede introducir decisiones no justificadas.",
            "consequence": "El artefacto quedara menos confiable para aprobacion o construccion posterior.",
            "suggested": f"Vincular {target} con objetivos del Canvas o registrarlo como supuesto trazable.",
        }
    if prefix == "vague_nfr":
        return {
            "type": "question",
            "severity": "warning",
            "title": f"Metrica observable para {target}",
            "reason": f"El requisito no funcional {target} requiere una metrica cuantificable (latencia, disponibilidad o volumen).",
            "impact": "Sin metrica, estimacion, validacion y monitoreo pierden precision.",
            "consequence": "El NFR podria pasar a implementacion sin umbral de cumplimiento.",
            "suggested": f"Convertir {target} en un NFR medible con umbral, ventana de medicion y owner.",
        }
    if prefix == "blocking_question":
        return {
            "type": "question",
            "severity": "blocking",
            "title": f"Definicion pendiente para {target}",
            "reason": f"La pregunta funcional {target} sigue abierta y requiere confirmacion para cerrar la etapa.",
            "impact": "Impide aprobar con trazabilidad suficiente.",
            "consequence": "La siguiente etapa podria heredar una decision incompleta.",
            "suggested": f"Responder {target} con una decision funcional o diferirla al ACP.",
        }
    if prefix == "duplicate_key":
        return {
            "type": "validation",
            "severity": "blocking",
            "title": "Hay identificadores duplicados",
            "reason": f"{target} se repite y puede romper trazabilidad entre requisitos, decisiones o artefactos.",
            "impact": "Puede provocar sobrescritura o confusion al reconciliar cambios.",
            "consequence": "El paquete podria exportar referencias ambiguas.",
            "suggested": f"Renombrar o unificar los duplicados asociados con {target}.",
        }
    if prefix == "canvas_scope_conflict":
        return {
            "type": "decision",
            "severity": "blocking",
            "title": "Existe conflicto entre alcance y fuera de alcance",
            "reason": f"{target} aparece con senales contradictorias en el canvas.",
            "impact": "Puede cambiar el diseno, las herramientas y la estimacion.",
            "consequence": "La solucion podria construir capacidades que el negocio no espera.",
            "suggested": f"Decidir si {target} queda dentro del alcance, fuera del alcance o diferido al ACP.",
        }
    return {
        "type": "validation",
        "severity": "warning",
        "title": "Validacion requiere revision",
        "reason": target,
        "impact": "Puede afectar calidad o trazabilidad del artefacto.",
        "consequence": "La observacion quedara abierta hasta revisarla.",
        "suggested": "Resolver o documentar la validacion con una justificacion.",
    }


def _issue_options(prefix: str, detail: str) -> list[AttentionDecisionOptionV3]:
    target = _label(detail)
    common_custom = AttentionDecisionOptionV3(
        key="manual_adjustment",
        label="Ajustar manualmente",
        description="Editar la etapa con una respuesta o correccion propia.",
        impact="Mantiene control humano cuando la inferencia no es suficiente.",
        example=f"Actualizar {target} con el criterio decidido por el owner.",
        recommended=False,
        confidence=0.68,
        source_refs=["validation.issue"],
    )
    if prefix == "missing_acceptance":
        return [
            AttentionDecisionOptionV3(
                key="accept_suggested_criterion",
                label="Usar criterio sugerido",
                description="Adoptar un criterio medible basado en la evidencia disponible.",
                impact="Cierra la incertidumbre sin sobrecargar al usuario.",
                example=f"{target}: cumplimiento validado con evidencia observable y owner asignado.",
                recommended=True,
                confidence=0.74,
                source_refs=["validation.missing_acceptance"],
            ),
            common_custom,
        ]
    if prefix == "untraced_item":
        return [
            AttentionDecisionOptionV3(
                key="link_existing_evidence",
                label="Vincular evidencia existente",
                description="Conectar el item con una fuente, decision o artefacto ya aprobado.",
                impact="Mejora trazabilidad sin cambiar el diseno.",
                example=f"Relacionar {target} con Discovery, Canvas o Definir.",
                recommended=True,
                confidence=0.72,
                source_refs=["validation.untraced_item"],
            ),
            AttentionDecisionOptionV3(
                key="record_as_assumption",
                label="Registrar como supuesto",
                description="Mantener el item con justificacion cuando no exista evidencia directa.",
                impact="Evita ocultar incertidumbre residual.",
                example=f"{target} queda como supuesto trazable para revision posterior.",
                recommended=False,
                confidence=0.58,
                source_refs=["validation.untraced_item"],
            ),
        ]
    if prefix == "vague_nfr":
        return [
            AttentionDecisionOptionV3(
                key="make_measurable",
                label="Convertir a metrica",
                description="Definir umbral, ventana de medicion y fuente de evidencia.",
                impact="Mejora validacion, estimacion y monitoreo.",
                example=f"{target}: responder en menos de 5 segundos para el 95% de solicitudes.",
                recommended=True,
                confidence=0.78,
                source_refs=["validation.vague_nfr"],
            ),
            common_custom,
        ]
    if prefix == "blocking_question":
        return [
            AttentionDecisionOptionV3(
                key="answer_now",
                label="Responder ahora",
                description="Cerrar la pregunta con una decision funcional dentro de la etapa actual.",
                impact="Permite avanzar sin heredar bloqueo.",
                example=f"Responder {target} con alcance, regla o criterio concreto.",
                recommended=True,
                confidence=0.8,
                source_refs=["validation.blocking_question"],
            ),
            AttentionDecisionOptionV3(
                key="defer_with_reason",
                label="Diferir con justificacion",
                description="Solo usar si la decision corresponde a implementacion o ACP.",
                impact="Evita adelantar decisiones tecnicas fuera de etapa.",
                example=f"{target} se difiere al ACP con impacto documentado.",
                recommended=False,
                confidence=0.6,
                source_refs=["validation.blocking_question"],
            ),
        ]
    if prefix == "duplicate_key":
        return [
            AttentionDecisionOptionV3(
                key="rename_duplicates",
                label="Renombrar duplicados",
                description="Asignar identificadores unicos conservando el significado.",
                impact="Recupera trazabilidad y evita referencias ambiguas.",
                example=f"{target}-1 y {target}-2.",
                recommended=True,
                confidence=0.76,
                source_refs=["validation.duplicate_key"],
            ),
            AttentionDecisionOptionV3(
                key="merge_duplicates",
                label="Unificar duplicados",
                description="Combinar items equivalentes y conservar una sola referencia canonica.",
                impact="Reduce ruido y evita doble conteo.",
                example=f"Fusionar definiciones repetidas de {target}.",
                recommended=False,
                confidence=0.68,
                source_refs=["validation.duplicate_key"],
            ),
        ]
    if prefix == "canvas_scope_conflict":
        return [
            AttentionDecisionOptionV3(
                key="keep_in_scope",
                label="Mantener dentro del alcance",
                description="Confirmar que la capacidad hace parte del Blueprint.",
                impact="Afecta diseno, herramientas, memoria y estimacion.",
                example=f"{target} queda como capacidad incluida.",
                recommended=False,
                confidence=0.62,
                source_refs=["validation.canvas_scope_conflict"],
            ),
            AttentionDecisionOptionV3(
                key="move_out_of_scope",
                label="Mover fuera del alcance",
                description="Excluir la capacidad y mantenerla como restriccion o no objetivo.",
                impact="Reduce complejidad y costo estimado.",
                example=f"{target} queda fuera de alcance del Blueprint.",
                recommended=True,
                confidence=0.7,
                source_refs=["validation.canvas_scope_conflict"],
            ),
            AttentionDecisionOptionV3(
                key="defer_to_acp",
                label="Diferir al ACP",
                description="Usar si la decision depende del entorno de implementacion.",
                impact="Mantiene el Blueprint limpio y deja la pregunta para construccion.",
                example=f"{target} se documenta como decision de implementacion.",
                recommended=False,
                confidence=0.58,
                source_refs=["validation.canvas_scope_conflict"],
            ),
        ]
    return [common_custom]


def validation_issue_to_attention_item(
    issue: str,
    *,
    product: str,
    stage: str,
    source: str,
    artifact_id: str,
    artifact_version: int | None,
    href: str,
    return_href: str,
) -> AttentionItemV2:
    prefix, detail = _split_issue(issue)
    copy = _issue_copy(prefix, detail)
    decision = AttentionDecisionV3(
        decision_key="",
        item_type=copy["type"],  # type: ignore[arg-type]
        severity=copy["severity"],  # type: ignore[arg-type]
        title=copy["title"],
        reason=copy["reason"],
        impact=copy["impact"],
        consequence_if_unresolved=copy["consequence"],
        required_decision=copy["title"],
        suggested_answer=copy["suggested"],
        source=AttentionDecisionSourceV3(
            product=product,  # type: ignore[arg-type]
            stage=stage,
            source=source,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            entity_id=issue,
            field_path="missing_information",
            href=href,
            return_href=return_href,
            owner_role="business_owner",
            affected_artifact_refs=[f"validation:{prefix}", f"artifact:{artifact_id}"],
        ),
        options=_issue_options(prefix, detail),
        action=AttentionDecisionActionV3(
            primary_kind="answer",
            primary_label="Resolver",
            can_resolve_inline=True,
            allowed_kinds=["answer", "confirm", "defer"],
        ),
    )
    return decision_to_attention_item_v2(decision)


def items_from_validation_issues(
    issues: Iterable[str],
    *,
    product: str,
    stage: str,
    source: str,
    artifact_id: str,
    artifact_version: int | None,
    href: str,
    return_href: str,
) -> list[AttentionItemV2]:
    return [
        validation_issue_to_attention_item(
            issue,
            product=product,
            stage=stage,
            source=source,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            href=href,
            return_href=return_href,
        )
        for issue in issues
    ]
