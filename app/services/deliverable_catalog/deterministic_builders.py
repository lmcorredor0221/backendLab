from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.services.deliverable_catalog.contracts import DeliverableGenerationTask, DeliverableRegistryEntry
from app.services.deliverable_catalog.project_generation_context import ProjectGenerationContext


DETERMINISTIC_DELIVERABLE_KEYS = frozenset({"definition.acceptance_trace"})


@dataclass(frozen=True)
class DeterministicBuilderContextError(ValueError):
    code: str
    message: str
    missing_refs: tuple[str, ...] = ()


def supports_deterministic_deliverable(key: str) -> bool:
    return str(key or "").strip() in DETERMINISTIC_DELIVERABLE_KEYS


def _walk(value: object) -> list[object]:
    items = [value]
    if isinstance(value, dict):
        for child in value.values():
            items.extend(_walk(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(_walk(child))
    return items


def _text(value: object, *, limit: int = 500) -> str:
    if isinstance(value, str):
        normalized = " ".join(value.split()).strip()
    else:
        try:
            normalized = " ".join(json.dumps(value, ensure_ascii=False, default=str).split()).strip()
        except Exception:
            normalized = " ".join(str(value or "").split()).strip()
    return normalized[:limit]


def _collect_named_lists(context: dict[str, object], names: set[str]) -> list[object]:
    collected: list[object] = []
    for item in _walk(context):
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            normalized = str(key or "").strip().lower()
            if normalized not in names:
                continue
            if isinstance(value, list):
                collected.extend(value)
            elif value not in (None, "", {}, []):
                collected.append(value)
    return collected


def _statement(item: object, fallback: str) -> str:
    if isinstance(item, dict):
        for key in (
            "statement",
            "description",
            "requirement",
            "title",
            "name",
            "label",
            "content",
            "text",
        ):
            value = _text(item.get(key))
            if value:
                return value
    value = _text(item)
    return value or fallback


def _source_refs(task: DeliverableGenerationTask) -> list[str]:
    refs = [str(ref or "").strip() for ref in task.approved_context_refs if str(ref or "").strip()]
    if refs:
        return refs
    policy = task.context_payload.get("context_policy") if isinstance(task.context_payload, dict) else {}
    requested = policy.get("requested_refs", []) if isinstance(policy, dict) else []
    return [str(ref or "").strip() for ref in requested if str(ref or "").strip()]


def _acceptance_trace(
    entry: DeliverableRegistryEntry,
    task: DeliverableGenerationTask,
    generation_context: ProjectGenerationContext | None = None,
) -> dict[str, object]:
    context = task.context_payload or {}
    refs = _source_refs(task)
    if generation_context is not None:
        refs = [source.ref for source in generation_context.source_refs] or refs
        requirements = [
            *generation_context.mvp_scope,
            *generation_context.objectives,
        ]
        criteria = list(generation_context.acceptance_criteria)
        rules = [
            *generation_context.nondelegable_decisions,
            *generation_context.constraints,
        ]
        summary = _text(
            generation_context.problem_statement
            or generation_context.desired_outcome
            or generation_context.project_title
            or "",
            limit=700,
        )
    else:
        requirements = _collect_named_lists(
            context,
            {
                "functional_requirements",
                "non_functional_requirements",
                "requirements",
                "mvp_scope",
                "scope",
                "v1_scope",
            },
        )
        criteria = _collect_named_lists(
            context,
            {
                "acceptance_criteria",
                "criteria",
                "success_criteria",
                "validation_criteria",
            },
        )
        rules = _collect_named_lists(
            context,
            {
                "business_rules",
                "rules",
                "non_delegable_decisions",
                "constraints",
            },
        )
        summary = _text(context.get("summary") or context.get("project_title") or "", limit=700)
    if not requirements and not criteria and not summary:
        raise DeterministicBuilderContextError(
            code="approved_requirements_missing",
            message="No hay requisitos, criterios ni resumen aprobado suficientes para construir la trazabilidad.",
            missing_refs=("definition.requirements", "session.define"),
        )
    if not refs:
        raise DeterministicBuilderContextError(
            code="source_refs_missing",
            message="No hay referencias aprobadas para respaldar la trazabilidad.",
            missing_refs=("approved_context_refs",),
        )

    if not requirements:
        requirements = [summary]
    if not criteria:
        criteria = [f"El usuario confirma que el requisito queda satisfecho: {_statement(requirements[0], 'requisito aprobado')}"]

    rows: list[dict[str, object]] = []
    for index, requirement in enumerate(requirements[:12], start=1):
        criterion = criteria[min(index - 1, len(criteria) - 1)]
        linked_rule = rules[min(index - 1, len(rules) - 1)] if rules else ""
        rows.append(
            {
                "requirement_id": f"REQ-{index:02d}",
                "requirement": _statement(requirement, f"Requisito aprobado {index}"),
                "acceptance_criterion_id": f"AC-{index:02d}",
                "acceptance_criterion": _statement(criterion, f"Criterio de aceptacion {index}"),
                "linked_rule": _statement(linked_rule, "") if linked_rule else "",
                "evidence_refs": refs[:5],
                "evidence_type": "explicit_approved_context",
                "confidence": "high" if task.approved_context_refs else "medium",
            }
        )

    markdown_lines = [
        f"# {entry.title}",
        "",
        "Matriz deterministica construida solo con contexto aprobado. No agrega requisitos ni criterios no evidenciados.",
        "",
        "| Requisito | Criterio de aceptacion | Regla vinculada | Evidencia |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["requirement"]),
                    str(row["acceptance_criterion"]),
                    str(row["linked_rule"] or "No declarada"),
                    ", ".join(str(ref) for ref in row["evidence_refs"]),
                ]
            )
            + " |"
        )

    return {
        "schema_version": "deliverable-artifact.v1",
        "title": entry.title,
        "content": "\n".join(markdown_lines),
        "trace_matrix": rows,
        "source_refs": refs,
        "metadata": {
            "generated_by": "deterministic_python",
            "deliverable_key": entry.deliverable_key,
            "input_summary": summary,
            "requirement_count": len(rows),
        },
    }


def build_deterministic_deliverable(
    entry: DeliverableRegistryEntry,
    task: DeliverableGenerationTask,
    generation_context: ProjectGenerationContext | None = None,
) -> dict[str, object]:
    key = str(entry.deliverable_key or "").strip()
    if key == "definition.acceptance_trace":
        return _acceptance_trace(entry, task, generation_context)
    raise DeterministicBuilderContextError(
        code="deterministic_builder_not_supported",
        message=f"No deterministic builder is registered for {key}.",
    )
