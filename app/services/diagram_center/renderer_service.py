from __future__ import annotations

from html import escape
import json
import math
import re

from app.services.diagram_center.contracts import DiagramModel, DiagramNotation


def _safe_mermaid_text(value: str) -> str:
    return re.sub(r"[\[\]{}()<>\"`|]", " ", value).replace("\n", " ").strip()


def _render_flowchart(model: DiagramModel) -> str:
    lines = [f"flowchart {model.direction}"]
    for group in model.groups:
        lines.append(f"  subgraph {group.id}[\"{_safe_mermaid_text(group.label)}\"]")
        for node in [item for item in model.nodes if item.group_id == group.id]:
            lines.append(f"    {node.id}[\"{_safe_mermaid_text(node.label)}\"]")
        lines.append("  end")
    grouped_node_ids = {node.id for node in model.nodes if node.group_id}
    for node in model.nodes:
        if node.id not in grouped_node_ids:
            lines.append(f"  {node.id}[\"{_safe_mermaid_text(node.label)}\"]")
    for edge in model.edges:
        label = f"|{_safe_mermaid_text(edge.label)}|" if edge.label else ""
        lines.append(f"  {edge.source} -->{label} {edge.target}")
    return "\n".join(lines)


def _render_sequence(model: DiagramModel) -> str:
    lines = ["sequenceDiagram"]
    for node in model.nodes:
        lines.append(f"  participant {node.id} as {_safe_mermaid_text(node.label)}")
    for edge in sorted(model.edges, key=lambda item: item.order if item.order is not None else 9999):
        lines.append(f"  {edge.source}->>{edge.target}: {_safe_mermaid_text(edge.label or edge.kind)}")
    return "\n".join(lines)


def _render_class(model: DiagramModel) -> str:
    lines = ["classDiagram"]
    for node in model.nodes:
        lines.append(f"  class {node.id}[\"{_safe_mermaid_text(node.label)}\"]")
    for edge in model.edges:
        lines.append(f"  {edge.source} --> {edge.target} : {_safe_mermaid_text(edge.label or edge.kind)}")
    return "\n".join(lines)


def _render_er(model: DiagramModel) -> str:
    lines = ["erDiagram"]
    for node in model.nodes:
        lines.append(f"  {node.id} {{")
        attributes = node.metadata.get("attributes", [])
        if isinstance(attributes, list):
            for index, attribute in enumerate(attributes[:12]):
                lines.append(f"    string field_{index + 1} \"{_safe_mermaid_text(str(attribute))}\"")
        lines.append("  }")
    for edge in model.edges:
        lines.append(f"  {edge.source} ||--o{{ {edge.target} : \"{_safe_mermaid_text(edge.label or edge.kind)}\"")
    return "\n".join(lines)


def _render_state(model: DiagramModel) -> str:
    lines = ["stateDiagram-v2"]
    for node in model.nodes:
        lines.append(f"  state \"{_safe_mermaid_text(node.label)}\" as {node.id}")
    for edge in model.edges:
        lines.append(f"  {edge.source} --> {edge.target}: {_safe_mermaid_text(edge.label or edge.kind)}")
    return "\n".join(lines)


def render_mermaid(model: DiagramModel) -> str:
    if model.notation == DiagramNotation.sequence:
        return _render_sequence(model)
    if model.notation == DiagramNotation.class_diagram:
        return _render_class(model)
    if model.notation == DiagramNotation.entity_relationship:
        return _render_er(model)
    if model.notation == DiagramNotation.state:
        return _render_state(model)
    return _render_flowchart(model)


def render_svg(model: DiagramModel) -> str:
    width = 1120
    columns = 3 if len(model.nodes) > 6 else 2
    node_width, node_height = 280, 76
    horizontal_gap, vertical_gap = 72, 62
    rows = max(1, math.ceil(max(len(model.nodes), 1) / columns))
    height = max(360, 120 + rows * (node_height + vertical_gap))
    positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(model.nodes):
        column = index % columns
        row = index // columns
        total_width = columns * node_width + (columns - 1) * horizontal_gap
        start_x = (width - total_width) / 2
        positions[node.id] = (start_x + column * (node_width + horizontal_gap), 92 + row * (node_height + vertical_gap))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(model.title)}</title>',
        f'<desc id="desc">{escape(model.description)}</desc>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#60708a"/></marker><filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#172033" flood-opacity="0.12"/></filter></defs>',
        '<rect width="100%" height="100%" rx="20" fill="#f8f9fc"/>',
    ]
    for edge in model.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        sx, sy = positions[edge.source]
        tx, ty = positions[edge.target]
        x1, y1 = sx + node_width / 2, sy + node_height / 2
        x2, y2 = tx + node_width / 2, ty + node_height / 2
        parts.append(f'<path d="M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}" stroke="#60708a" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
        if edge.label:
            parts.append(f'<text x="{(x1+x2)/2:.1f}" y="{(y1+y2)/2-8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#44506a">{escape(edge.label[:48])}</text>')
    for node in model.nodes:
        x, y = positions[node.id]
        parts.append(f'<g filter="url(#shadow)"><rect x="{x:.1f}" y="{y:.1f}" width="{node_width}" height="{node_height}" rx="10" fill="#ffffff" stroke="#cbd3e1"/></g>')
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="5" height="{node_height}" rx="2.5" fill="#3047b8"/>')
        parts.append(f'<text x="{x+22:.1f}" y="{y+31:.1f}" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="700" fill="#10172a">{escape(node.label[:38])}</text>')
        parts.append(f'<text x="{x+22:.1f}" y="{y+52:.1f}" font-family="Inter,Arial,sans-serif" font-size="11" fill="#69748b">{escape(node.kind[:42])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_diagram(model: DiagramModel) -> dict[str, str]:
    return {
        "json": json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2),
        "mermaid": render_mermaid(model),
        "svg": render_svg(model),
    }

