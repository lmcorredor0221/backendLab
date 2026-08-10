from __future__ import annotations

from dataclasses import dataclass, field
from html import escape as html_escape
import math
from typing import Any

from app.models import ACPFileEntry, SessionSnapshot
from app.services.acp_paths import build_tool_contract_path_for_tool, slugify_acp_token
from app.services.acp_serialization import serialize_json_document, serialize_markdown_document
from app.services.acp_validation import build_acp_file_entry


@dataclass
class DiagramNode:
    key: str
    label: str


@dataclass
class DiagramEdge:
    source: str
    target: str
    label: str = ""
    style: str = "solid"


@dataclass
class DiagramSpec:
    key: str
    title: str
    summary: str
    source_artifacts: list[str]
    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
    notes: list[str] = field(default_factory=list)
    mermaid_kind: str = "flowchart LR"


class _GraphBuilder:
    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_node(
        self,
        node_id: str,
        *,
        label: str,
        node_type: str,
        description: str = "",
        source_artifacts: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        entry = self._nodes.setdefault(
            node_id,
            {
                "id": node_id,
                "label": label,
                "type": node_type,
                "description": description,
                "source_artifacts": [],
                "properties": {},
            },
        )
        if label:
            entry["label"] = label
        if node_type:
            entry["type"] = node_type
        if description and not entry["description"]:
            entry["description"] = description
        if source_artifacts:
            entry["source_artifacts"] = sorted({*entry["source_artifacts"], *source_artifacts})
        if properties:
            merged = dict(entry["properties"])
            for key, value in properties.items():
                if value not in ("", None, [], {}):
                    merged[key] = value
            entry["properties"] = merged

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        *,
        description: str = "",
        source_artifacts: list[str] | None = None,
    ) -> None:
        key = (source, target, relation)
        entry = self._edges.setdefault(
            key,
            {
                "source": source,
                "target": target,
                "type": relation,
                "description": description,
                "source_artifacts": [],
            },
        )
        if description and not entry["description"]:
            entry["description"] = description
        if source_artifacts:
            entry["source_artifacts"] = sorted({*entry["source_artifacts"], *source_artifacts})

    def export(self) -> dict[str, Any]:
        return {
            "nodes": sorted(self._nodes.values(), key=lambda item: item["id"]),
            "edges": sorted(
                self._edges.values(),
                key=lambda item: (item["source"], item["target"], item["type"]),
            ),
        }


def _artifact_paths(files: list[ACPFileEntry], *, prefix: str = "", domain: str = "") -> list[str]:
    paths = []
    for item in files:
        if prefix and not item.path.startswith(prefix):
            continue
        if domain and item.domain != domain:
            continue
        paths.append(item.path)
    return sorted(paths)


def _first_path(files: list[ACPFileEntry], path: str) -> str | None:
    for item in files:
        if item.path == path:
            return item.path
    return None


def _sanitize_node_key(value: str) -> str:
    slug = slugify_acp_token(value, default="node").replace("-", "_")
    return f"n_{slug}"


def _short_label(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _runtime_tokens(snapshot: SessionSnapshot) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for item in snapshot.integration_statuses:
        for chunk in item.detail.split():
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            tokens[key.strip()] = value.strip()
    return tokens


def _detect_capabilities(snapshot: SessionSnapshot, files: list[ACPFileEntry]) -> list[tuple[str, list[str]]]:
    blueprint = snapshot.blueprint
    detected: list[tuple[str, list[str]]] = []
    if blueprint is not None and blueprint.reasoning_pattern.strip():
        detected.append(("Planning", ["ACP/cognition/reasoning.yaml", "ACP/cognition/planner.yaml"]))
        detected.append(("Reasoning", ["ACP/cognition/reasoning.yaml"]))
    if blueprint is not None and blueprint.tools:
        detected.append(("Tool Calling", ["ACP/tools/permissions.yaml"]))
    if _artifact_paths(files, prefix="ACP/memory/"):
        detected.append(("Memory", _artifact_paths(files, prefix="ACP/memory/")))
    if _artifact_paths(files, prefix="ACP/knowledge/"):
        detected.append(("Retrieval", _artifact_paths(files, prefix="ACP/knowledge/")))
    if _artifact_paths(files, prefix="ACP/prompts/"):
        detected.append(("Generation", _artifact_paths(files, prefix="ACP/prompts/")))
    if _artifact_paths(files, prefix="ACP/evaluation/"):
        detected.append(("Validation", _artifact_paths(files, prefix="ACP/evaluation/")))
    if _artifact_paths(files, prefix="ACP/deployment/"):
        detected.append(("Deployment", _artifact_paths(files, prefix="ACP/deployment/")))
    if _artifact_paths(files, prefix="ACP/observability/"):
        detected.append(("Observability", _artifact_paths(files, prefix="ACP/observability/")))
    if snapshot.discovery is not None:
        detected.append(("Analysis", ["ACP/business/lean-canvas.yaml"]))
    unique: dict[str, list[str]] = {}
    for label, sources in detected:
        merged = unique.setdefault(label, [])
        merged.extend(sources)
    return sorted((label, sorted(set(paths))) for label, paths in unique.items())


def _build_graph(snapshot: SessionSnapshot, files: list[ACPFileEntry]) -> dict[str, Any]:
    builder = _GraphBuilder()
    file_map = {item.path: item for item in files}
    session_title = snapshot.session.title.strip() or "Lean Agent Builder"
    agent_sources = [path for path in ["ACP/manifest.yaml", "ACP/README.md"] if path in file_map]
    builder.add_node(
        "agent_main",
        label=session_title,
        node_type="Agent",
        description="Primary ACP target agent.",
        source_artifacts=agent_sources,
        properties={"stage": snapshot.session.current_stage, "status": snapshot.session.status},
    )

    for domain in sorted({item.domain for item in files}):
        domain_id = f"domain_{_sanitize_node_key(domain)}"
        builder.add_node(
            domain_id,
            label=domain.replace("-", " ").title(),
            node_type="Domain",
            source_artifacts=_artifact_paths(files, domain=domain),
            properties={"domain": domain},
        )
        builder.add_edge(domain_id, "agent_main", "DESCRIBES", source_artifacts=_artifact_paths(files, domain=domain))

    for item in files:
        artifact_id = f"artifact_{_sanitize_node_key(item.path)}"
        builder.add_node(
            artifact_id,
            label=item.title or _short_label(item.path),
            node_type="Artifact",
            description=item.path,
            source_artifacts=[item.path],
            properties={
                "path": item.path,
                "domain": item.domain,
                "format": item.format,
                "status": item.status,
                "warnings_count": len(item.warnings),
                "missing_fields_count": len(item.missing_fields),
            },
        )
        builder.add_edge(artifact_id, f"domain_{_sanitize_node_key(item.domain)}", "BELONGS_TO", source_artifacts=[item.path])

    discovery = snapshot.discovery
    if discovery is not None:
        builder.add_node(
            "problem_main",
            label=discovery.problem_statement or "Problem statement",
            node_type="Problem",
            source_artifacts=["ACP/business/lean-canvas.yaml"],
        )
        builder.add_node(
            "goal_main",
            label=discovery.desired_outcome or "Desired outcome",
            node_type="Goal",
            source_artifacts=["ACP/business/lean-canvas.yaml", "ACP/manifest.yaml"],
        )
        builder.add_node(
            "persona_primary",
            label=discovery.current_user or "Primary user",
            node_type="Persona",
            source_artifacts=["ACP/business/lean-canvas.yaml", "ACP/architecture/c4-context.md"],
        )
        builder.add_edge("problem_main", "agent_main", "MOTIVATES", source_artifacts=["ACP/business/lean-canvas.yaml"])
        builder.add_edge("agent_main", "goal_main", "PURSUES", source_artifacts=["ACP/business/lean-canvas.yaml", "ACP/manifest.yaml"])
        builder.add_edge("persona_primary", "agent_main", "USES_AGENT", source_artifacts=["ACP/business/lean-canvas.yaml"])
        if discovery.mvp_definition.north_star_metric.strip():
            builder.add_node(
                "metric_north_star",
                label=discovery.mvp_definition.north_star_metric,
                node_type="Metric",
                source_artifacts=["ACP/business/kpis.yaml", "ACP/manifest.yaml"],
            )
            builder.add_edge("agent_main", "metric_north_star", "MEASURES", source_artifacts=["ACP/business/kpis.yaml"])
        for index, constraint in enumerate(discovery.constraints, start=1):
            if not constraint.strip():
                continue
            node_id = f"constraint_{index}"
            builder.add_node(
                node_id,
                label=constraint,
                node_type="Constraint",
                source_artifacts=["ACP/business/constraints.yaml"],
            )
            builder.add_edge(node_id, "agent_main", "RESTRICTS", source_artifacts=["ACP/business/constraints.yaml"])

    canvas = snapshot.canvas
    if canvas is not None and canvas.primary_risk.strip():
        builder.add_node(
            "risk_primary",
            label=canvas.primary_risk,
            node_type="Risk",
            source_artifacts=["ACP/business/lean-canvas.yaml", "ACP/cognition/guardrails.yaml"],
        )
        builder.add_edge("risk_primary", "agent_main", "THREATENS", source_artifacts=["ACP/business/lean-canvas.yaml"])

    blueprint = snapshot.blueprint
    if blueprint is not None:
        builder.add_node(
            "architecture_main",
            label=blueprint.architecture or "Architecture",
            node_type="Architecture",
            source_artifacts=["ACP/architecture/topology.yaml", "ACP/architecture/decisions.yaml"],
        )
        builder.add_node(
            "reasoning_main",
            label=blueprint.reasoning_pattern or "Reasoning pattern",
            node_type="Reasoning",
            source_artifacts=["ACP/cognition/reasoning.yaml"],
        )
        builder.add_node(
            "memory_main",
            label=blueprint.memory_profile.strategy or blueprint.memory_strategy or "Memory strategy",
            node_type="Memory",
            source_artifacts=["ACP/memory/strategy.yaml", "ACP/memory/retrieval.yaml"],
        )
        builder.add_edge("agent_main", "architecture_main", "IMPLEMENTS", source_artifacts=["ACP/architecture/topology.yaml"])
        builder.add_edge("agent_main", "reasoning_main", "USES_PATTERN", source_artifacts=["ACP/cognition/reasoning.yaml"])
        builder.add_edge("agent_main", "memory_main", "PERSISTS", source_artifacts=["ACP/memory/strategy.yaml"])

        for index, layer in enumerate(blueprint.memory_profile.storage_layers, start=1):
            if not layer.strip():
                continue
            node_id = f"memory_layer_{index}"
            builder.add_node(
                node_id,
                label=layer,
                node_type="MemoryLayer",
                source_artifacts=["ACP/memory/retrieval.yaml"],
            )
            builder.add_edge("memory_main", node_id, "STORES_IN", source_artifacts=["ACP/memory/retrieval.yaml"])

        for index, tool in enumerate(blueprint.tools, start=1):
            tool_label = tool.name or f"tool_{index}"
            tool_id = f"tool_{_sanitize_node_key(tool_label)}"
            contract_path = build_tool_contract_path_for_tool(tool, index)
            builder.add_node(
                tool_id,
                label=tool_label,
                node_type="Tool",
                description=tool.purpose,
                source_artifacts=[contract_path, "ACP/tools/permissions.yaml"],
                properties={
                    "risk_level": tool.risk_level,
                    "requires_approval": tool.requires_approval,
                    "side_effects": tool.has_side_effects,
                },
            )
            builder.add_edge("agent_main", tool_id, "USES_TOOL", source_artifacts=[contract_path, "ACP/tools/permissions.yaml"])

        steps = blueprint.delivery_package.workflow_profile.steps
        for index, step in enumerate(steps, start=1):
            if not step.name.strip():
                continue
            step_id = f"workflow_step_{index}"
            builder.add_node(
                step_id,
                label=step.name,
                node_type="Workflow",
                description=step.objective,
                source_artifacts=["ACP/workflows/durable-workflow.yaml", "ACP/workflows/state-machine.yaml"],
                properties={"actor": step.actor, "requires_approval": step.requires_approval},
            )
            builder.add_edge("agent_main", step_id, "EXECUTES", source_artifacts=["ACP/workflows/durable-workflow.yaml"])
            if index > 1:
                builder.add_edge(
                    f"workflow_step_{index - 1}",
                    step_id,
                    "NEXT",
                    source_artifacts=["ACP/workflows/state-machine.yaml"],
                )

        for index, deliverable in enumerate(blueprint.delivery_package.deliverables, start=1):
            if not deliverable.key.strip():
                continue
            deliverable_id = f"deliverable_{index}"
            builder.add_node(
                deliverable_id,
                label=deliverable.title or deliverable.key,
                node_type="Deliverable",
                description=deliverable.summary,
                source_artifacts=["ACP/README.md"],
                properties={"deliverable_key": deliverable.key},
            )
            builder.add_edge("agent_main", deliverable_id, "PRODUCES", source_artifacts=["ACP/README.md"])

        for index, safety_check in enumerate(blueprint.safety_checks, start=1):
            if not safety_check.risk.strip():
                continue
            risk_id = f"safety_risk_{index}"
            builder.add_node(
                risk_id,
                label=safety_check.risk,
                node_type="Risk",
                description=safety_check.mitigation,
                source_artifacts=["ACP/cognition/guardrails.yaml"],
                properties={"severity": safety_check.severity, "status": safety_check.status},
            )
            builder.add_edge(risk_id, "agent_main", "REQUIRES", source_artifacts=["ACP/cognition/guardrails.yaml"])

    runtime_tokens = _runtime_tokens(snapshot)
    llm_provider = runtime_tokens.get("provider", "")
    llm_model = runtime_tokens.get("reasoning") or runtime_tokens.get("fast") or ""
    if llm_provider or llm_model:
        builder.add_node(
            "llm_primary",
            label=f"{llm_provider or 'provider'}:{llm_model or 'model'}",
            node_type="LLM",
            source_artifacts=["ACP/runtime/models.yaml", "ACP/runtime/providers.yaml"],
            properties={"provider": llm_provider, "model": llm_model},
        )
        builder.add_edge("agent_main", "llm_primary", "USES_MODEL", source_artifacts=["ACP/runtime/models.yaml"])

    for item in snapshot.integration_statuses:
        node_type = "ExternalSystem"
        if "postgres" in item.integration_key.lower():
            node_type = "Database"
        elif "openai" in item.integration_key.lower():
            node_type = "LLM"
        node_id = f"integration_{_sanitize_node_key(item.integration_key)}"
        builder.add_node(
            node_id,
            label=item.label or item.integration_key,
            node_type=node_type,
            description=item.detail,
            source_artifacts=["ACP/runtime/config.yaml", "ACP/deployment/env.template"],
            properties={"status": item.status, "configured": item.configured, "reachable": item.reachable},
        )
        builder.add_edge("agent_main", node_id, "DEPENDS_ON", source_artifacts=["ACP/runtime/config.yaml"])

    for path in _artifact_paths(files, prefix="ACP/prompts/"):
        prompt_id = f"prompt_{_sanitize_node_key(path)}"
        builder.add_node(
            prompt_id,
            label=_short_label(path),
            node_type="Prompt",
            description=path,
            source_artifacts=[path],
        )
        builder.add_edge("agent_main", prompt_id, "REFERENCES", source_artifacts=[path])

    capabilities = _detect_capabilities(snapshot, files)
    for index, (label, sources) in enumerate(capabilities, start=1):
        node_id = f"capability_{index}"
        builder.add_node(
            node_id,
            label=label,
            node_type="Capability",
            source_artifacts=sources,
        )
        builder.add_edge("agent_main", node_id, "ENABLES", source_artifacts=sources)

    return {
        "graph_version": "blueprint-graph.v1",
        "generated_from_session_id": str(snapshot.session.id),
        "generated_at": snapshot.session.updated_at.isoformat(),
        **builder.export(),
    }


def _select_nodes(keys: list[str], labels: dict[str, str]) -> list[DiagramNode]:
    return [DiagramNode(key=key, label=labels[key]) for key in keys if key in labels]


def _normalize_spec(spec: DiagramSpec) -> DiagramSpec:
    node_map: dict[str, DiagramNode] = {}
    for node in spec.nodes:
        if node.key not in node_map:
            node_map[node.key] = node
    node_keys = set(node_map)
    edges = [edge for edge in spec.edges if edge.source in node_keys and edge.target in node_keys]
    return DiagramSpec(
        key=spec.key,
        title=spec.title,
        summary=spec.summary,
        source_artifacts=sorted(set(spec.source_artifacts)),
        nodes=list(node_map.values()),
        edges=edges,
        notes=list(dict.fromkeys(spec.notes)),
        mermaid_kind=spec.mermaid_kind,
    )


def _mermaid_escape(value: str) -> str:
    return value.replace('"', '\\"')


def _render_mermaid(spec: DiagramSpec) -> str:
    if spec.mermaid_kind == "stateDiagram-v2":
        lines = ["stateDiagram-v2"]
        for node in spec.nodes:
            lines.append(f'  state "{node.label}" as {node.key}')
        if spec.nodes:
            lines.append(f"  [*] --> {spec.nodes[0].key}")
        for edge in spec.edges:
            label = f" : {edge.label}" if edge.label else ""
            lines.append(f"  {edge.source} --> {edge.target}{label}")
        if spec.nodes:
            lines.append(f"  {spec.nodes[-1].key} --> [*]")
        return serialize_markdown_document("\n".join(lines))

    lines = [spec.mermaid_kind]
    for node in spec.nodes:
        lines.append(f'  {node.key}["{_mermaid_escape(node.label)}"]')
    for edge in spec.edges:
        arrow = "-.->" if edge.style == "dashed" else "-->"
        label = f"|{edge.label.replace('|', '/')}|" if edge.label else ""
        lines.append(f"  {edge.source} {arrow}{label} {edge.target}")
    return serialize_markdown_document("\n".join(lines))


def _render_d2(spec: DiagramSpec) -> str:
    direction = "right"
    if "TD" in spec.mermaid_kind:
        direction = "down"
    if spec.mermaid_kind == "stateDiagram-v2":
        direction = "right"
    lines = [f"direction: {direction}"]
    for node in spec.nodes:
        lines.append(f'{node.key}: "{node.label}"')
    for edge in spec.edges:
        label = f': "{edge.label}"' if edge.label else ""
        connector = "->"
        if edge.style == "dashed":
            connector = "-->"
        lines.append(f"{edge.source} {connector} {edge.target}{label}")
    return serialize_markdown_document("\n".join(lines))


def _render_plantuml(spec: DiagramSpec) -> str:
    lines = ["@startuml"]
    if spec.mermaid_kind == "stateDiagram-v2":
        for node in spec.nodes:
            lines.append(f'state "{node.label}" as {node.key}')
        if spec.nodes:
            lines.append(f"[*] --> {spec.nodes[0].key}")
        for edge in spec.edges:
            label = f" : {edge.label}" if edge.label else ""
            lines.append(f"{edge.source} --> {edge.target}{label}")
        if spec.nodes:
            lines.append(f"{spec.nodes[-1].key} --> [*]")
    else:
        lines.append("left to right direction")
        lines.append("skinparam shadowing false")
        for node in spec.nodes:
            lines.append(f'rectangle "{node.label}" as {node.key}')
        for edge in spec.edges:
            arrow = "..>" if edge.style == "dashed" else "-->"
            label = f" : {edge.label}" if edge.label else ""
            lines.append(f"{edge.source} {arrow} {edge.target}{label}")
    lines.append("@enduml")
    return serialize_markdown_document("\n".join(lines))


def _render_markdown(spec: DiagramSpec, mermaid_content: str) -> str:
    lines = [
        f"# {spec.title}",
        "",
        spec.summary,
        "",
        "## Source Artifacts",
    ]
    if spec.source_artifacts:
        lines.extend(f"- `{path}`" for path in spec.source_artifacts)
    else:
        lines.append("- No explicit ACP sources detected.")
    if spec.notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in spec.notes)
    lines.extend(["", "## Mermaid", "```mermaid", mermaid_content.rstrip(), "```"])
    return serialize_markdown_document("\n".join(lines))


def _wrap_svg_text(value: str, *, max_chars: int = 20, max_lines: int = 3) -> list[str]:
    words = value.strip().split()
    if not words:
        return ["Untitled"]

    lines: list[str] = []
    current = ""
    for word in words:
        proposal = f"{current} {word}".strip()
        if len(proposal) <= max_chars:
            current = proposal
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines - 1:
            break

    if current and len(lines) < max_lines:
        lines.append(current)

    remaining = " ".join(words)
    rendered = " ".join(lines)
    if len(lines) == max_lines and len(remaining) > len(rendered):
        lines[-1] = f"{lines[-1][: max(0, max_chars - 3)].rstrip()}..."
    return lines[:max_lines]


def _truncate_svg_text(value: str, *, max_chars: int = 88) -> str:
    clean = value.strip()
    if len(clean) <= max_chars:
        return clean
    return f"{clean[: max_chars - 3].rstrip()}..."


def _build_svg_layout(spec: DiagramSpec) -> tuple[dict[str, tuple[float, float]], float, float]:
    node_count = max(len(spec.nodes), 1)
    if spec.mermaid_kind == "stateDiagram-v2":
        columns = min(max(node_count, 1), 5)
    else:
        columns = min(max(2, math.ceil(math.sqrt(node_count))), 4) if node_count > 1 else 1
    rows = math.ceil(node_count / columns)

    node_width = 208
    node_height = 88
    column_gap = 64
    row_gap = 34
    padding_x = 46
    padding_top = 92
    padding_bottom = 42

    positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(spec.nodes):
        column = index % columns
        row = index // columns
        positions[node.key] = (
            padding_x + column * (node_width + column_gap),
            padding_top + row * (node_height + row_gap),
        )

    width = padding_x * 2 + columns * node_width + max(0, columns - 1) * column_gap
    height = padding_top + rows * node_height + max(0, rows - 1) * row_gap + padding_bottom
    return positions, float(width), float(height)


def _edge_anchor_points(
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    node_width: float,
    node_height: float,
) -> tuple[float, float, float, float]:
    source_center_x = source[0] + node_width / 2
    source_center_y = source[1] + node_height / 2
    target_center_x = target[0] + node_width / 2
    target_center_y = target[1] + node_height / 2
    delta_x = target_center_x - source_center_x
    delta_y = target_center_y - source_center_y

    if abs(delta_x) >= abs(delta_y):
        start_x = source[0] + (node_width if delta_x >= 0 else 0)
        start_y = source_center_y
        end_x = target[0] + (0 if delta_x >= 0 else node_width)
        end_y = target_center_y
    else:
        start_x = source_center_x
        start_y = source[1] + (node_height if delta_y >= 0 else 0)
        end_x = target_center_x
        end_y = target[1] + (0 if delta_y >= 0 else node_height)
    return start_x, start_y, end_x, end_y


def _render_svg(spec: DiagramSpec) -> str:
    if not spec.nodes:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 220" width="640" height="220">'
            '<rect width="640" height="220" fill="#f8fafc" rx="24"/>'
            f'<text x="36" y="64" font-family="Georgia, Cambria, \'Times New Roman\', serif" font-size="28" '
            f'font-weight="700" fill="#14213d">{html_escape(spec.title)}</text>'
            '<text x="36" y="108" font-family="Georgia, Cambria, \'Times New Roman\', serif" font-size="16" '
            'fill="#52606d">No nodes were available for this ACP visualization.</text>'
            "</svg>"
        )

    node_width = 208.0
    node_height = 88.0
    positions, width, height = _build_svg_layout(spec)
    title_id = f"{slugify_acp_token(spec.key, default='diagram')}-title"
    desc_id = f"{slugify_acp_token(spec.key, default='diagram')}-desc"
    summary = _truncate_svg_text(spec.summary, max_chars=110)
    source_count = len(spec.source_artifacts)
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(width)} {int(height)}" width="{int(width)}" height="{int(height)}" role="img" aria-labelledby="{title_id} {desc_id}">',
        f"<title id=\"{title_id}\">{html_escape(spec.title)}</title>",
        f"<desc id=\"{desc_id}\">{html_escape(summary)}</desc>",
        "<defs>",
        '  <linearGradient id="bg-gradient" x1="0%" y1="0%" x2="100%" y2="100%">',
        '    <stop offset="0%" stop-color="#f8fbff"/>',
        '    <stop offset="58%" stop-color="#eef2ff"/>',
        '    <stop offset="100%" stop-color="#fff8ef"/>',
        "  </linearGradient>",
        '  <marker id="diagram-arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">',
        '    <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/>',
        "  </marker>",
        "</defs>",
        f'<rect x="0" y="0" width="{int(width)}" height="{int(height)}" rx="28" fill="url(#bg-gradient)"/>',
        f'<text x="38" y="46" font-family="Georgia, Cambria, \'Times New Roman\', serif" font-size="30" font-weight="700" fill="#14213d">{html_escape(spec.title)}</text>',
        f'<text x="38" y="72" font-family="Georgia, Cambria, \'Times New Roman\', serif" font-size="14" fill="#52606d">{html_escape(summary)}</text>',
        f'<text x="{int(width - 38)}" y="46" text-anchor="end" font-family="Georgia, Cambria, \'Times New Roman\', serif" font-size="12" font-weight="700" letter-spacing="2" fill="#7c8aa5">ACP SVG</text>',
        f'<text x="{int(width - 38)}" y="68" text-anchor="end" font-family="Georgia, Cambria, \'Times New Roman\', serif" font-size="13" fill="#52606d">{source_count} source artifacts</text>',
    ]

    for edge in spec.edges:
        source = positions.get(edge.source)
        target = positions.get(edge.target)
        if source is None or target is None:
            continue
        start_x, start_y, end_x, end_y = _edge_anchor_points(source, target, node_width=node_width, node_height=node_height)
        horizontal = abs(end_x - start_x) >= abs(end_y - start_y)
        delta = max(abs(end_x - start_x), abs(end_y - start_y))
        control = max(42.0, delta * 0.34)
        if horizontal:
            control_ax = start_x + (control if end_x >= start_x else -control)
            control_ay = start_y
            control_bx = end_x - (control if end_x >= start_x else -control)
            control_by = end_y
        else:
            control_ax = start_x
            control_ay = start_y + (control if end_y >= start_y else -control)
            control_bx = end_x
            control_by = end_y - (control if end_y >= start_y else -control)
        dash_attr = ' stroke-dasharray="10 7"' if edge.style == "dashed" else ""
        svg_parts.append(
            f'<path d="M {start_x:.1f} {start_y:.1f} C {control_ax:.1f} {control_ay:.1f}, {control_bx:.1f} {control_by:.1f}, {end_x:.1f} {end_y:.1f}" '
            f'fill="none" stroke="#94a3b8" stroke-width="2.1" marker-end="url(#diagram-arrow)"{dash_attr}/>'
        )
        if edge.label:
            label_x = (start_x + end_x) / 2
            label_y = (start_y + end_y) / 2 - 8
            svg_parts.append(
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-family="Georgia, Cambria, \'Times New Roman\', serif" '
                f'font-size="12" font-weight="700" fill="#4f46e5">{html_escape(edge.label)}</text>'
            )

    node_palettes = [
        ("#ffffff", "#5b63ff", "#14213d"),
        ("#fff8ef", "#d97706", "#7c5b14"),
        ("#eefaf4", "#0f9f6e", "#14532d"),
    ]
    for index, node in enumerate(spec.nodes):
        x, y = positions[node.key]
        fill, stroke, accent = node_palettes[index % len(node_palettes)]
        label_lines = _wrap_svg_text(node.label)
        svg_parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_width:.1f}" height="{node_height:.1f}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
                f'<rect x="{x + 14:.1f}" y="{y + 14:.1f}" width="56" height="6" rx="3" fill="{stroke}" opacity="0.72"/>',
            ]
        )
        for line_index, line in enumerate(label_lines):
            svg_parts.append(
                f'<text x="{x + 16:.1f}" y="{y + 42 + line_index * 18:.1f}" font-family="Georgia, Cambria, \'Times New Roman\', serif" '
                f'font-size="16" font-weight="700" fill="{accent}">{html_escape(line)}</text>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _graph_nodes_by_type(graph: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    return [item for item in graph["nodes"] if item["type"] == node_type]


def _build_diagram_specs(snapshot: SessionSnapshot, files: list[ACPFileEntry], graph: dict[str, Any]) -> list[DiagramSpec]:
    labels = {item["id"]: item["label"] for item in graph["nodes"]}
    tools = _graph_nodes_by_type(graph, "Tool")
    integrations = [
        item for item in graph["nodes"] if item["type"] in {"Database", "ExternalSystem", "LLM"}
    ]
    prompts = _graph_nodes_by_type(graph, "Prompt")
    capabilities = _graph_nodes_by_type(graph, "Capability")
    workflow_steps = _graph_nodes_by_type(graph, "Workflow")
    constraints = _graph_nodes_by_type(graph, "Constraint")
    risks = _graph_nodes_by_type(graph, "Risk")
    metrics = _graph_nodes_by_type(graph, "Metric")
    memory_layers = _graph_nodes_by_type(graph, "MemoryLayer")

    architecture_component_nodes = [DiagramNode("agent_main", labels["agent_main"])]
    architecture_component_nodes.extend(_select_nodes(["architecture_main", "reasoning_main", "memory_main", "llm_primary"], labels))
    architecture_component_nodes.extend(
        DiagramNode(item["id"], item["label"])
        for item in integrations
        if item["id"] != "llm_primary"
    )
    architecture_component_nodes.extend(
        [
            DiagramNode("knowledge_layer", "Knowledge"),
            DiagramNode("tool_layer", "Tools"),
            DiagramNode("workflow_layer", "Workflow"),
            DiagramNode("deployment_layer", "Deployment"),
            DiagramNode("observability_layer", "Observability"),
            DiagramNode("security_layer", "Security"),
        ]
    )
    architecture_component_labels = {item.key for item in architecture_component_nodes}
    architecture_edges = [
        DiagramEdge("agent_main", "architecture_main", "topology"),
        DiagramEdge("agent_main", "reasoning_main", "reasoning"),
        DiagramEdge("agent_main", "memory_main", "memory"),
        DiagramEdge("agent_main", "tool_layer", "tool calling"),
        DiagramEdge("agent_main", "workflow_layer", "workflow"),
        DiagramEdge("agent_main", "deployment_layer", "deployment"),
        DiagramEdge("agent_main", "observability_layer", "signals"),
        DiagramEdge("agent_main", "security_layer", "guardrails"),
        DiagramEdge("tool_layer", "knowledge_layer", "retrieval"),
    ]
    if "llm_primary" in architecture_component_labels:
        architecture_edges.append(DiagramEdge("agent_main", "llm_primary", "llm"))
    for item in integrations:
        if item["id"] in architecture_component_labels and item["id"] != "llm_primary":
            architecture_edges.append(DiagramEdge("tool_layer", item["id"], "integration"))

    executive_nodes = [DiagramNode("agent_main", labels["agent_main"])]
    executive_nodes.extend(_select_nodes(["problem_main", "goal_main", "persona_primary", "memory_main"], labels))
    executive_nodes.extend(DiagramNode(item["id"], item["label"]) for item in metrics[:2])
    executive_nodes.extend(DiagramNode(item["id"], item["label"]) for item in constraints[:3])
    executive_nodes.extend(DiagramNode(item["id"], item["label"]) for item in risks[:2])
    executive_nodes.append(DiagramNode("tool_summary", f"Tools ({len(tools)})"))
    executive_nodes.append(DiagramNode("integration_summary", f"Integrations ({len(integrations)})"))
    executive_edges = [
        DiagramEdge("problem_main", "agent_main", "problem"),
        DiagramEdge("agent_main", "goal_main", "objective"),
        DiagramEdge("persona_primary", "agent_main", "user"),
        DiagramEdge("agent_main", "memory_main", "memory"),
        DiagramEdge("agent_main", "tool_summary", "tools"),
        DiagramEdge("tool_summary", "integration_summary", "integrates"),
    ]
    executive_edges.extend(DiagramEdge(item["id"], "agent_main", "constraint", "dashed") for item in constraints[:3])
    executive_edges.extend(DiagramEdge("agent_main", item["id"], "kpi") for item in metrics[:2])
    executive_edges.extend(DiagramEdge(item["id"], "agent_main", "risk", "dashed") for item in risks[:2])

    capability_nodes = [DiagramNode("agent_main", labels["agent_main"])]
    capability_nodes.extend(DiagramNode(item["id"], item["label"]) for item in capabilities)
    capability_edges = [DiagramEdge("agent_main", item["id"], "capability") for item in capabilities]

    workflow_nodes = [DiagramNode(item["id"], item["label"]) for item in workflow_steps]
    workflow_edges = []
    for index in range(len(workflow_steps) - 1):
        workflow_edges.append(DiagramEdge(workflow_steps[index]["id"], workflow_steps[index + 1]["id"], "next"))

    state_nodes = workflow_nodes or [
        DiagramNode("draft_capture", "Draft Capture"),
        DiagramNode("discover", "Discover"),
        DiagramNode("design", "Blueprint"),
        DiagramNode("evaluate", "Evaluate"),
        DiagramNode("ready", "Ready for Export"),
    ]
    state_edges = workflow_edges or [
        DiagramEdge("draft_capture", "discover", "normalize"),
        DiagramEdge("discover", "design", "build"),
        DiagramEdge("design", "evaluate", "validate"),
        DiagramEdge("evaluate", "ready", "approve"),
    ]
    state_edges.append(DiagramEdge(state_nodes[-1].key, state_nodes[0].key, "rework"))

    memory_nodes = _select_nodes(["memory_main"], labels)
    memory_nodes.extend(DiagramNode(item["id"], item["label"]) for item in memory_layers[:4])
    if _artifact_paths(files, prefix="ACP/knowledge/"):
        memory_nodes.append(DiagramNode("knowledge_base", "Knowledge Base"))
    if _first_path(files, "ACP/runtime/providers.yaml"):
        memory_nodes.append(DiagramNode("vector_store_pending", "Vector Store (needs review)"))
    memory_edges = [DiagramEdge("memory_main", item["id"], "layer") for item in memory_layers[:4]]
    if any(item.key == "knowledge_base" for item in memory_nodes):
        memory_edges.append(DiagramEdge("memory_main", "knowledge_base", "retrieves"))
    if any(item.key == "vector_store_pending" for item in memory_nodes):
        memory_edges.append(DiagramEdge("knowledge_base", "vector_store_pending", "indexes", "dashed"))

    tool_nodes = [DiagramNode("agent_main", labels["agent_main"])]
    tool_nodes.extend(DiagramNode(item["id"], item["label"]) for item in tools)
    tool_nodes.extend(DiagramNode(item["id"], item["label"]) for item in integrations)
    tool_edges = [DiagramEdge("agent_main", item["id"], "uses") for item in tools]
    for tool in tools:
        for integration in integrations[:3]:
            tool_edges.append(DiagramEdge(tool["id"], integration["id"], "depends_on", "dashed"))

    integration_nodes = [DiagramNode("agent_main", labels["agent_main"])]
    integration_nodes.extend(DiagramNode(item["id"], item["label"]) for item in integrations)
    integration_edges = [DiagramEdge("agent_main", item["id"], "integration") for item in integrations]

    dependency_nodes = _select_nodes(
        ["agent_main", "architecture_main", "reasoning_main", "memory_main", "llm_primary"],
        labels,
    )
    dependency_nodes.extend(DiagramNode(item["id"], item["label"]) for item in tools[:4])
    dependency_nodes.extend(DiagramNode(item["id"], item["label"]) for item in prompts[:4])
    dependency_edges = [
        DiagramEdge("agent_main", "architecture_main", "depends_on"),
        DiagramEdge("agent_main", "reasoning_main", "depends_on"),
        DiagramEdge("agent_main", "memory_main", "depends_on"),
    ]
    if any(item.key == "llm_primary" for item in dependency_nodes):
        dependency_edges.append(DiagramEdge("agent_main", "llm_primary", "depends_on"))
    dependency_edges.extend(DiagramEdge("agent_main", item["id"], "uses") for item in tools[:4])
    dependency_edges.extend(DiagramEdge("reasoning_main", item["id"], "references", "dashed") for item in prompts[:4])

    trace_nodes = _select_nodes(["goal_main", "architecture_main", "reasoning_main"], labels)
    trace_nodes.extend(DiagramNode(item["id"], item["label"]) for item in tools[:3])
    trace_nodes.extend(DiagramNode(item["id"], item["label"]) for item in prompts[:2])
    trace_nodes.extend(_select_nodes(["llm_primary"], labels))
    trace_nodes.append(DiagramNode("deployment_layer", "Deployment"))
    if metrics:
        trace_nodes.append(DiagramNode(metrics[0]["id"], metrics[0]["label"]))
    trace_edges = [
        DiagramEdge("goal_main", "architecture_main", "drives"),
        DiagramEdge("architecture_main", "reasoning_main", "selects"),
    ]
    if tools:
        trace_edges.append(DiagramEdge("reasoning_main", tools[0]["id"], "activates"))
    if tools and prompts:
        trace_edges.append(DiagramEdge(tools[0]["id"], prompts[0]["id"], "guided_by"))
    elif prompts:
        trace_edges.append(DiagramEdge("reasoning_main", prompts[0]["id"], "guided_by"))
    if any(item.key == "llm_primary" for item in trace_nodes):
        llm_source = prompts[0]["id"] if prompts else (tools[0]["id"] if tools else "reasoning_main")
        trace_edges.append(DiagramEdge(llm_source, "llm_primary", "calls"))
        trace_edges.append(DiagramEdge("llm_primary", "deployment_layer", "deployed_with"))
    else:
        trace_edges.append(DiagramEdge("reasoning_main", "deployment_layer", "targets", "dashed"))
    if metrics:
        trace_edges.append(DiagramEdge("deployment_layer", metrics[0]["id"], "validated_by"))

    loop_nodes = [
        DiagramNode("loop_objective", "Objective"),
        DiagramNode("loop_plan", "Planning"),
        DiagramNode("loop_context", "Context"),
        DiagramNode("loop_memory", "Memory"),
        DiagramNode("loop_tools", "Tool Calling"),
        DiagramNode("loop_reflect", "Reflection"),
        DiagramNode("loop_evaluate", "Evaluation"),
        DiagramNode("loop_replan", "Replan"),
        DiagramNode("loop_response", "Response"),
    ]
    loop_edges = [
        DiagramEdge("loop_objective", "loop_plan"),
        DiagramEdge("loop_plan", "loop_context"),
        DiagramEdge("loop_context", "loop_memory"),
        DiagramEdge("loop_memory", "loop_tools"),
        DiagramEdge("loop_tools", "loop_reflect"),
        DiagramEdge("loop_reflect", "loop_evaluate"),
        DiagramEdge("loop_evaluate", "loop_replan"),
        DiagramEdge("loop_replan", "loop_response"),
        DiagramEdge("loop_evaluate", "loop_plan", "feedback", "dashed"),
    ]

    construction_nodes = [
        DiagramNode("flow_idea", "Idea"),
        DiagramNode("flow_discovery", "Discovery"),
        DiagramNode("flow_requirements", "Requirements"),
        DiagramNode("flow_architecture", "Architecture"),
        DiagramNode("flow_blueprint", "Blueprint"),
        DiagramNode("flow_validation", "Validation"),
        DiagramNode("flow_acp", "Construction Package"),
        DiagramNode("flow_codegen", "Code Generation"),
        DiagramNode("flow_deploy", "Deployment"),
        DiagramNode("flow_test", "Testing"),
        DiagramNode("flow_monitor", "Monitoring"),
    ]
    construction_edges = [
        DiagramEdge("flow_idea", "flow_discovery"),
        DiagramEdge("flow_discovery", "flow_requirements"),
        DiagramEdge("flow_requirements", "flow_architecture"),
        DiagramEdge("flow_architecture", "flow_blueprint"),
        DiagramEdge("flow_blueprint", "flow_validation"),
        DiagramEdge("flow_validation", "flow_acp"),
        DiagramEdge("flow_acp", "flow_codegen", "builder handoff"),
        DiagramEdge("flow_codegen", "flow_deploy", "target", "dashed"),
        DiagramEdge("flow_deploy", "flow_test", "verify"),
        DiagramEdge("flow_test", "flow_monitor", "operate"),
    ]

    data_model_nodes = [
        DiagramNode("model_session", "Session"),
        DiagramNode("model_discovery", "Discovery"),
        DiagramNode("model_canvas", "Canvas"),
        DiagramNode("model_blueprint", "Blueprint"),
        DiagramNode("model_tool_contract", "Tool Contract"),
        DiagramNode("model_prompt", "Prompt"),
        DiagramNode("model_runtime", "Runtime Config"),
        DiagramNode("model_deployment", "Deployment Config"),
        DiagramNode("model_evaluation", "Evaluation Dataset"),
    ]
    data_model_edges = [
        DiagramEdge("model_session", "model_discovery", "contains"),
        DiagramEdge("model_discovery", "model_canvas", "shapes"),
        DiagramEdge("model_canvas", "model_blueprint", "feeds"),
        DiagramEdge("model_blueprint", "model_tool_contract", "defines"),
        DiagramEdge("model_blueprint", "model_prompt", "references"),
        DiagramEdge("model_blueprint", "model_runtime", "requires"),
        DiagramEdge("model_runtime", "model_deployment", "deploys"),
        DiagramEdge("model_blueprint", "model_evaluation", "validated_by"),
    ]

    infrastructure_nodes = [
        DiagramNode("infra_agent", "Agent Service"),
        DiagramNode("infra_env", "env.template"),
        DiagramNode("infra_compose", "docker-compose"),
        DiagramNode("infra_db", "PostgreSQL"),
        DiagramNode("infra_llm", labels.get("llm_primary", "OpenAI Runtime")),
        DiagramNode("infra_obs", "Observability"),
    ]
    infrastructure_edges = [
        DiagramEdge("infra_env", "infra_agent", "configures"),
        DiagramEdge("infra_compose", "infra_agent", "runs"),
        DiagramEdge("infra_compose", "infra_db", "runs"),
        DiagramEdge("infra_agent", "infra_llm", "calls"),
        DiagramEdge("infra_agent", "infra_db", "persists"),
        DiagramEdge("infra_agent", "infra_obs", "emits"),
    ]

    knowledge_nodes = [DiagramNode("agent_main", labels["agent_main"])]
    knowledge_nodes.extend(DiagramNode(item["id"], item["label"]) for item in tools[:4])
    knowledge_nodes.extend(DiagramNode(item["id"], item["label"]) for item in integrations[:4])
    knowledge_nodes.extend(DiagramNode(item["id"], item["label"]) for item in prompts[:4])
    knowledge_nodes.extend(DiagramNode(item["id"], item["label"]) for item in capabilities[:4])
    knowledge_edges = [DiagramEdge("agent_main", item["id"], "relates") for item in tools[:4]]
    knowledge_edges.extend(DiagramEdge("agent_main", item["id"], "relates") for item in prompts[:4])
    for tool in tools[:3]:
        for integration in integrations[:2]:
            knowledge_edges.append(DiagramEdge(tool["id"], integration["id"], "depends_on"))

    notes = {
        "architecture": [
            "Only components with direct ACP evidence are visualized automatically.",
            "Missing layers such as frontend, queues or cloud targets stay out until the ACP defines them.",
        ],
        "construction-flow": [
            "Code Generation, Deployment and Monitoring are shown as target continuation stages of the ACP handoff.",
        ],
        "traceability": [
            "If a layer is not captured yet, the chain intentionally skips it instead of inventing it.",
        ],
        "knowledge-graph": [
            "The Mermaid view is a curated slice. Use `ACP/blueprint.graph.json` for the full graph.",
        ],
    }

    return [
        _normalize_spec(
        DiagramSpec(
            key="executive-canvas",
            title="ExecutiveCanvas",
            summary="Executive view derived from business, memory, tools, integrations, metrics, risks and constraints already present in the ACP.",
            source_artifacts=[
                "ACP/business/lean-canvas.yaml",
                "ACP/business/kpis.yaml",
                "ACP/business/constraints.yaml",
                "ACP/memory/strategy.yaml",
                "ACP/tools/permissions.yaml",
            ],
            nodes=executive_nodes,
            edges=executive_edges,
        )),
        _normalize_spec(
        DiagramSpec(
            key="architecture",
            title="Architecture",
            summary="Detected architecture components inferred from topology, cognition, runtime, tools, deployment and observability artifacts.",
            source_artifacts=[
                "ACP/architecture/topology.yaml",
                "ACP/cognition/reasoning.yaml",
                "ACP/runtime/config.yaml",
                "ACP/deployment/docker-compose.yaml",
                "ACP/observability/telemetry.yaml",
            ],
            nodes=architecture_component_nodes,
            edges=architecture_edges,
            notes=notes["architecture"],
        )),
        _normalize_spec(
        DiagramSpec(
            key="agent-loop",
            title="AgentLoop",
            summary="Reasoning loop inferred from reasoning, planner, memory, tool and evaluation artifacts.",
            source_artifacts=[
                "ACP/cognition/reasoning.yaml",
                "ACP/cognition/planner.yaml",
                "ACP/cognition/reflection.yaml",
                "ACP/evaluation/rubrics.yaml",
            ],
            nodes=loop_nodes,
            edges=loop_edges,
        )),
        _normalize_spec(
        DiagramSpec(
            key="memory",
            title="Memory",
            summary="Memory architecture derived from memory, runtime and knowledge artifacts.",
            source_artifacts=[
                "ACP/memory/strategy.yaml",
                "ACP/memory/retrieval.yaml",
                "ACP/knowledge/embeddings.yaml",
                "ACP/runtime/providers.yaml",
            ],
            nodes=memory_nodes,
            edges=memory_edges,
        )),
        _normalize_spec(
        DiagramSpec(
            key="tool-map",
            title="ToolMap",
            summary="Tool and integration map inferred from tool contracts, runtime and deployment artifacts.",
            source_artifacts=[
                "ACP/tools/permissions.yaml",
                "ACP/runtime/config.yaml",
                "ACP/deployment/env.template",
            ],
            nodes=tool_nodes,
            edges=tool_edges,
        )),
        _normalize_spec(
        DiagramSpec(
            key="construction-flow",
            title="ConstructionFlow",
            summary="End-to-end ACP construction flow showing current completed stages and target continuation stages for the builder agent.",
            source_artifacts=[
                "ACP/business/lean-canvas.yaml",
                "ACP/architecture/topology.yaml",
                "ACP/evaluation/rubrics.yaml",
                "ACP/construction-readiness/overview.yaml",
            ],
            nodes=construction_nodes,
            edges=construction_edges,
            notes=notes["construction-flow"],
            mermaid_kind="flowchart TD",
        )),
        _normalize_spec(
        DiagramSpec(
            key="capability-map",
            title="CapabilityMap",
            summary="Capability map inferred automatically from the blueprint, tools, memory, evaluation, deployment and observability evidence.",
            source_artifacts=sorted({path for _, paths in _detect_capabilities(snapshot, files) for path in paths}),
            nodes=capability_nodes,
            edges=capability_edges,
        )),
        _normalize_spec(
        DiagramSpec(
            key="workflow",
            title="Workflow",
            summary="Functional workflow derived from the durable workflow profile stored in the ACP.",
            source_artifacts=[
                "ACP/workflows/durable-workflow.yaml",
                "ACP/workflows/state-machine.yaml",
                "ACP/workflows/langgraph.json",
            ],
            nodes=workflow_nodes or state_nodes,
            edges=workflow_edges or state_edges,
            mermaid_kind="flowchart TD",
        )),
        _normalize_spec(
        DiagramSpec(
            key="state-machine",
            title="StateMachine",
            summary="State machine synthesized from ACP workflow states and lifecycle checkpoints.",
            source_artifacts=[
                "ACP/workflows/state-machine.yaml",
                "ACP/workflows/durable-workflow.yaml",
            ],
            nodes=state_nodes,
            edges=state_edges,
            mermaid_kind="stateDiagram-v2",
        )),
        _normalize_spec(
        DiagramSpec(
            key="integrations",
            title="Integrations",
            summary="All known integrations represented from runtime and deployment evidence already present in the ACP.",
            source_artifacts=[
                "ACP/runtime/config.yaml",
                "ACP/runtime/providers.yaml",
                "ACP/deployment/env.template",
            ],
            nodes=integration_nodes,
            edges=integration_edges,
        )),
        _normalize_spec(
        DiagramSpec(
            key="data-model",
            title="DataModel",
            summary="Package-level data model showing how the main ACP entities relate to each other.",
            source_artifacts=[
                "ACP/manifest.yaml",
                "ACP/README.md",
                "ACP/tools/permissions.yaml",
                "ACP/evaluation/golden-dataset.json",
            ],
            nodes=data_model_nodes,
            edges=data_model_edges,
            mermaid_kind="flowchart TD",
        )),
        _normalize_spec(
        DiagramSpec(
            key="infrastructure",
            title="Infrastructure",
            summary="Physical infrastructure view inferred from deployment, runtime and observability artifacts without inventing a cloud target.",
            source_artifacts=[
                "ACP/deployment/docker-compose.yaml",
                "ACP/deployment/env.template",
                "ACP/runtime/config.yaml",
                "ACP/observability/telemetry.yaml",
            ],
            nodes=infrastructure_nodes,
            edges=infrastructure_edges,
        )),
        _normalize_spec(
        DiagramSpec(
            key="dependency-graph",
            title="DependencyGraph",
            summary="Dependency graph showing which major ACP components depend on others.",
            source_artifacts=[
                "ACP/architecture/topology.yaml",
                "ACP/cognition/reasoning.yaml",
                "ACP/memory/strategy.yaml",
                "ACP/prompts/system.md",
                "ACP/tools/permissions.yaml",
            ],
            nodes=dependency_nodes,
            edges=dependency_edges,
        )),
        _normalize_spec(
        DiagramSpec(
            key="traceability",
            title="Traceability",
            summary="Visual traceability chain from goal to architecture, tools, prompts, runtime, deployment and validation evidence.",
            source_artifacts=[
                "ACP/business/lean-canvas.yaml",
                "ACP/architecture/decisions.yaml",
                "ACP/prompts/planner.md",
                "ACP/runtime/models.yaml",
                "ACP/deployment/env.template",
                "ACP/evaluation/rubrics.yaml",
            ],
            nodes=trace_nodes,
            edges=trace_edges,
            notes=notes["traceability"],
            mermaid_kind="flowchart TD",
        )),
        _normalize_spec(
        DiagramSpec(
            key="knowledge-graph",
            title="KnowledgeGraph",
            summary="Curated view of the ACP knowledge graph. The complete graph is exported separately as JSON, GraphML and Cypher.",
            source_artifacts=[
                "ACP/blueprint.graph.json",
                "ACP/blueprint.graphml",
                "ACP/blueprint.cypher",
            ],
            nodes=knowledge_nodes,
            edges=knowledge_edges,
            notes=notes["knowledge-graph"],
        )),
    ]


def _render_graphml(graph: dict[str, Any]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="description" for="node" attr.name="description" attr.type="string"/>',
        '  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>',
        '  <graph id="blueprint" edgedefault="directed">',
    ]
    for node in graph["nodes"]:
        lines.extend(
            [
                f'    <node id="{html_escape(node["id"])}">',
                f'      <data key="label">{html_escape(node["label"])}</data>',
                f'      <data key="type">{html_escape(node["type"])}</data>',
                f'      <data key="description">{html_escape(node["description"])}</data>',
                "    </node>",
            ]
        )
    for index, edge in enumerate(graph["edges"], start=1):
        lines.extend(
            [
                f'    <edge id="e{index}" source="{html_escape(edge["source"])}" target="{html_escape(edge["target"])}">',
                f'      <data key="relation">{html_escape(edge["type"])}</data>',
                "    </edge>",
            ]
        )
    lines.extend(["  </graph>", "</graphml>"])
    return serialize_markdown_document("\n".join(lines))


def _cypher_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _render_cypher(graph: dict[str, Any]) -> str:
    lines: list[str] = []
    for node in graph["nodes"]:
        lines.extend(
            [
                f"MERGE (n:{slugify_acp_token(node['type'], default='node').upper()} {{id: '{_cypher_escape(node['id'])}'}})",
                f"SET n.label = '{_cypher_escape(node['label'])}',",
                f"    n.type = '{_cypher_escape(node['type'])}',",
                f"    n.description = '{_cypher_escape(node['description'])}';",
                "",
            ]
        )
    for edge in graph["edges"]:
        relation = slugify_acp_token(edge["type"], default="rel").upper().replace("-", "_")
        lines.extend(
            [
                f"MATCH (a {{id: '{_cypher_escape(edge['source'])}'}}), (b {{id: '{_cypher_escape(edge['target'])}'}})",
                f"MERGE (a)-[:{relation}]->(b);",
                "",
            ]
        )
    return serialize_markdown_document("\n".join(lines))


def _build_manifest_payload(graph: dict[str, Any], specs: list[DiagramSpec]) -> dict[str, Any]:
    return {
        "manifest_version": "blueprint-visualization.v1",
        "generated_at": graph["generated_at"],
        "graph": {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "exports": [
                "ACP/blueprint.graph.json",
                "ACP/blueprint.graphml",
                "ACP/blueprint.cypher",
            ],
        },
        "diagram_count": len(specs),
        "diagram_paths": [f"ACP/diagrams/{spec.title}.md" for spec in specs],
        "diagrams": [
            {
                "spec_key": spec.key,
                "title": spec.title,
                "summary": spec.summary,
                "source_artifacts": spec.source_artifacts,
                "paths": {
                    "markdown": f"ACP/diagrams/{spec.title}.md",
                    "mermaid": f"ACP/mermaid/{spec.title}.mmd",
                    "plantuml": f"ACP/plantuml/{spec.title}.puml",
                    "d2": f"ACP/d2/{spec.title}.d2",
                    "svg": f"ACP/svg/{spec.title}.svg",
                },
            }
            for spec in specs
        ],
        "formats": {
            "generated": ["markdown", "mermaid", "plantuml", "d2", "svg"],
            "pending": ["png"],
        },
    }


def build_acp_visualization_files(snapshot: SessionSnapshot, files: list[ACPFileEntry]) -> list[ACPFileEntry]:
    graph = _build_graph(snapshot, files)
    specs = _build_diagram_specs(snapshot, files, graph)
    generated: list[ACPFileEntry] = []

    for spec in specs:
        mermaid = _render_mermaid(spec)
        generated.append(
            build_acp_file_entry(
                path=f"ACP/diagrams/{spec.title}.md",
                domain="diagrams",
                title=spec.title,
                format="markdown",
                source_sections=["acp_visualization", "knowledge_graph"],
                content_text=_render_markdown(spec, mermaid),
            )
        )
        generated.append(
            build_acp_file_entry(
                path=f"ACP/mermaid/{spec.title}.mmd",
                domain="diagrams",
                title=f"{spec.title} Mermaid",
                format="mermaid",
                source_sections=["acp_visualization", "knowledge_graph"],
                content_text=mermaid,
            )
        )
        generated.append(
            build_acp_file_entry(
                path=f"ACP/plantuml/{spec.title}.puml",
                domain="diagrams",
                title=f"{spec.title} PlantUML",
                format="plantuml",
                source_sections=["acp_visualization", "knowledge_graph"],
                content_text=_render_plantuml(spec),
            )
        )
        generated.append(
            build_acp_file_entry(
                path=f"ACP/d2/{spec.title}.d2",
                domain="diagrams",
                title=f"{spec.title} D2",
                format="d2",
                source_sections=["acp_visualization", "knowledge_graph"],
                content_text=_render_d2(spec),
            )
        )
        generated.append(
            build_acp_file_entry(
                path=f"ACP/svg/{spec.title}.svg",
                domain="diagrams",
                title=f"{spec.title} SVG",
                format="svg",
                source_sections=["acp_visualization", "knowledge_graph"],
                content_text=_render_svg(spec),
            )
        )

    generated.extend(
        [
            build_acp_file_entry(
                path="ACP/png/README.md",
                domain="diagrams",
                title="PNG exports",
                format="markdown",
                source_sections=["acp_visualization"],
                content_text=serialize_markdown_document(
                    "# PNG exports\n\nPNG rendering is not emitted by the current local builder runtime yet. Use the generated SVG files as the canonical rendered output and keep Mermaid, PlantUML or D2 as source formats.\n"
                ),
                warnings=["PNG rendering is not available in the local ACP generator runtime yet."],
            ),
            build_acp_file_entry(
                path="ACP/blueprint.graph.json",
                domain="diagrams",
                title="Blueprint knowledge graph JSON",
                format="json",
                source_sections=["acp_visualization", "knowledge_graph"],
                content_text=serialize_json_document(graph),
            ),
            build_acp_file_entry(
                path="ACP/blueprint.graphml",
                domain="diagrams",
                title="Blueprint knowledge graph GraphML",
                format="graphml",
                source_sections=["acp_visualization", "knowledge_graph"],
                content_text=_render_graphml(graph),
            ),
            build_acp_file_entry(
                path="ACP/blueprint.cypher",
                domain="diagrams",
                title="Blueprint knowledge graph Cypher",
                format="cypher",
                source_sections=["acp_visualization", "knowledge_graph"],
                content_text=_render_cypher(graph),
            ),
            build_acp_file_entry(
                path="ACP/blueprint.manifest.json",
                domain="diagrams",
                title="Blueprint visualization manifest",
                format="json",
                source_sections=["acp_visualization", "knowledge_graph"],
                content_text=serialize_json_document(_build_manifest_payload(graph, specs)),
            ),
        ]
    )
    return sorted(generated, key=lambda item: item.path)
