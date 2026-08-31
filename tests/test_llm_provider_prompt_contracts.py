from pathlib import Path

from app.services.llm_runtime.capability_registry import BuilderCapability, get_builder_capability_spec
from app.services.llm_runtime.prompt_templates import (
    build_tool_recommendation_context_task_instruction,
    build_tool_recommendation_inline_prompt,
    build_tool_recommendation_schema_prompt,
    build_tool_recommendation_staged_prompt,
    build_tool_recommendation_system_instruction,
)


def test_tool_recommendation_registry_requires_design_implication_context() -> None:
    spec = get_builder_capability_spec(BuilderCapability.recommend_minimal_tools)

    assert "design_tool_implications" in spec.task_instruction
    assert "design_memory_implications" in spec.task_instruction
    assert "design_tool_implications" in spec.system_instruction
    assert "design_memory_implications" in spec.system_instruction


def test_design_registry_requires_business_pattern_and_dependency_contract() -> None:
    spec = get_builder_capability_spec(BuilderCapability.propose_agent_design)

    for required_fragment in (
        "arquetipo de agente",
        "familia de patron",
        "ajuste con el negocio",
        "metricas de negocio",
        "implicaciones para Tools",
        "implicaciones para Memory",
    ):
        assert required_fragment in spec.task_instruction
    assert "sin definir Tools ni Memory de forma canonica" in spec.system_instruction


def test_memory_registry_declares_governed_tool_dependency_requests() -> None:
    spec = get_builder_capability_spec(BuilderCapability.recommend_memory_architecture)

    assert "tool_dependency_requests" in spec.task_instruction
    assert "No inventes tool keys" in spec.task_instruction
    assert "outbound_notification" in spec.task_instruction


def test_tool_recommendation_shared_template_carries_design_implication_contract() -> None:
    template_outputs = [
        build_tool_recommendation_context_task_instruction(),
        build_tool_recommendation_system_instruction(),
        build_tool_recommendation_staged_prompt(),
        build_tool_recommendation_inline_prompt(case_json="{}", catalog_json="{}"),
        build_tool_recommendation_schema_prompt(schema_json="{}", case_json="{}", catalog_json="{}"),
    ]

    for output in template_outputs:
        assert "design_tool_implications" in output
        assert "design_memory_implications" in output
        assert "Nunca inventes tool keys fuera del catalogo" in output


def test_provider_specific_tool_prompts_preserve_design_implication_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    provider_files = {
        root / "app" / "services" / "openai_builder.py": [
            "build_tool_recommendation_context_task_instruction",
            "build_tool_recommendation_system_instruction",
        ],
        root / "app" / "services" / "llm_runtime" / "codex_cli" / "provider_facade.py": [
            "build_tool_recommendation_staged_prompt",
            "build_tool_recommendation_inline_prompt",
        ],
        root / "app" / "services" / "llm_runtime" / "antigravity_cli" / "provider_facade.py": [
            "build_tool_recommendation_schema_prompt",
        ],
    }

    for provider_file, required_helpers in provider_files.items():
        source = provider_file.read_text(encoding="utf-8")
        for helper in required_helpers:
            assert helper in source


def test_provider_specific_design_prompts_route_through_capability_registry() -> None:
    root = Path(__file__).resolve().parents[1]
    provider_files = [
        root / "app" / "services" / "openai_builder.py",
        root / "app" / "services" / "llm_runtime" / "codex_cli" / "provider_facade.py",
        root / "app" / "services" / "llm_runtime" / "antigravity_cli" / "provider_facade.py",
    ]

    for provider_file in provider_files:
        source = provider_file.read_text(encoding="utf-8")
        assert "BuilderCapability.propose_agent_design" in source
        assert "get_builder_capability_spec" in source
