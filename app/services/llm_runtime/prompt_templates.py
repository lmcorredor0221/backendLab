from __future__ import annotations


TOOL_RECOMMENDATION_CATALOG_GUARDRAIL = (
    "Usa exclusivamente tools del catalogo permitido. Nunca inventes tool keys fuera del catalogo."
)

TOOL_RECOMMENDATION_DESIGN_CONTEXT_RULE = (
    "Usa design_tool_implications y design_memory_implications para respetar la arquitectura seleccionada "
    "sin sobregenerar tools ni bloquear por capacidades blandas no gobernadas."
)

TOOL_RECOMMENDATION_SELECTION_RULE = (
    "Manten toda tool mandatory si la evidencia la sostiene. "
    "Usa requirements_coverage, design_role_coverage y design_tool_implications para justificar cobertura real. "
    "Usa design_memory_implications para anticipar dependencias que Memory necesitara despues. "
    "Marca como unnecessary cualquier tool candidata que no aporte capacidad unica. "
    "Si falta informacion, devuelve gaps estructurados en lugar de inventar tools."
)


def build_tool_recommendation_registry_task_instruction(
    *,
    tools_scope_instruction: str = "",
    guided_question_instruction: str = "",
) -> str:
    suffix = " ".join(
        item.strip()
        for item in (tools_scope_instruction, guided_question_instruction)
        if item.strip()
    )
    return " ".join(
        item
        for item in (
            "Clasifica tools como mandatory, optional o unnecessary.",
            TOOL_RECOMMENDATION_CATALOG_GUARDRAIL,
            TOOL_RECOMMENDATION_DESIGN_CONTEXT_RULE,
            suffix,
        )
        if item
    )


def build_tool_recommendation_context_task_instruction() -> str:
    return " ".join(
        [
            "Selecciona el conjunto minimo de herramientas usando solo `tool_recommendation_case` y "
            "`tool_recommendation_catalog`.",
            TOOL_RECOMMENDATION_CATALOG_GUARDRAIL,
            TOOL_RECOMMENDATION_SELECTION_RULE,
        ]
    )


def build_tool_recommendation_system_instruction() -> str:
    return " ".join(
        [
            "Selecciona el conjunto minimo de herramientas para un agente Lean.",
            "Usa solo el contexto aprobado y el catalogo permitido.",
            TOOL_RECOMMENDATION_CATALOG_GUARDRAIL,
            TOOL_RECOMMENDATION_SELECTION_RULE,
        ]
    )


def build_tool_recommendation_staged_prompt() -> str:
    return " ".join(
        [
            "Devuelve exclusivamente JSON valido segun el schema provisto.",
            "Selecciona el conjunto minimo de herramientas usando solo las fuentes staged "
            "`tool_recommendation_case` y `tool_recommendation_catalog`.",
            TOOL_RECOMMENDATION_CATALOG_GUARDRAIL,
            TOOL_RECOMMENDATION_SELECTION_RULE,
        ]
    )


def build_tool_recommendation_inline_prompt(*, case_json: str, catalog_json: str) -> str:
    return (
        " ".join(
            [
                "Devuelve exclusivamente JSON valido segun el schema provisto.",
                "Selecciona el conjunto minimo de herramientas para un agente Lean usando solo el contexto aprobado "
                "y el catalogo permitido.",
                TOOL_RECOMMENDATION_CATALOG_GUARDRAIL,
                TOOL_RECOMMENDATION_SELECTION_RULE,
            ]
        )
        + f"\n\nCASE:\n{case_json}\n\nCATALOG:\n{catalog_json}"
    )


def build_tool_recommendation_schema_prompt(
    *,
    schema_json: str,
    case_json: str,
    catalog_json: str,
) -> str:
    return (
        "Devuelve exclusivamente un JSON valido que cumpla con el siguiente schema JSON:\n"
        f"{schema_json}\n\n"
        + build_tool_recommendation_inline_prompt(case_json=case_json, catalog_json=catalog_json)
    )
