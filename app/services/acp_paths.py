from __future__ import annotations

import re

ACP_CANONICAL_ENV_TEMPLATE_PATH = "ACP/deployment/env.template"


def slugify_acp_token(value: str, default: str = "item") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or default


def build_tool_contract_path(tool_name: str, index: int, tool_type: str = "external") -> str:
    category = "internal" if tool_type == "internal" else "external"
    return f"ACP/tools/{category}/tool-{slugify_acp_token(tool_name, default=str(index))}.yaml"


def build_tool_contract_path_for_tool(tool, index: int) -> str:
    tool_name = getattr(tool, "name", "") or f"tool_{index}"
    tool_type = getattr(tool, "tool_type", "external") or "external"
    return build_tool_contract_path(tool_name, index, tool_type=tool_type)
