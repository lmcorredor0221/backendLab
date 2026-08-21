from __future__ import annotations

from html import escape
import json
import math
import re

from app.services.diagram_center.contracts import DiagramLane, DiagramModel, DiagramNotation, DiagramPool
from app.services.diagram_center.layout_engine import compute_layered_layout, route_layered_edges
from app.services.diagram_center.layout_sizing import measure_generic_node


RENDERER_REVISION = "diagram-renderer.v1.3.0"


def _safe_mermaid_text(value: str) -> str:
    return re.sub(r"[\[\]{}()<>\"`|]", " ", value).replace("\n", " ").strip()


def _safe_puml_text(value: str) -> str:
    return value.replace("\n", " ").replace('"', "'").strip()


def _kind(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").lower()).strip("_")


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


def _plantuml_use_case(model: DiagramModel) -> str:
    lines = ["@startuml", "left to right direction"]
    actors = [node for node in model.nodes if "actor" in _kind(node.kind)]
    use_cases = [node for node in model.nodes if node not in actors]
    for actor in actors:
        lines.append(f'actor "{_safe_puml_text(actor.label)}" as {actor.id}')
    lines.append(f'rectangle "{_safe_puml_text(model.title)}" {{')
    for node in use_cases:
        lines.append(f'  usecase "{_safe_puml_text(node.label)}" as {node.id}')
    lines.append("}")
    for edge in model.edges:
        arrow = "..>" if edge.kind in {"include", "extend"} else "--"
        label = f" : {_safe_puml_text(edge.label or edge.kind)}" if edge.label or edge.kind else ""
        lines.append(f"{edge.source} {arrow} {edge.target}{label}")
    lines.append("@enduml")
    return "\n".join(lines)


def _plantuml_activity(model: DiagramModel) -> str:
    lines = ["@startuml", "start"]
    emitted: set[str] = set()
    for edge in sorted(model.edges, key=lambda item: item.order if item.order is not None else 9999):
        source = next((node for node in model.nodes if node.id == edge.source), None)
        target = next((node for node in model.nodes if node.id == edge.target), None)
        if source and "decision" in _kind(source.kind):
            lines.append(f'if ({_safe_puml_text(source.label)}?) then ({_safe_puml_text(edge.label or "si")})')
            emitted.add(source.id)
            if target:
                lines.append(f'  :{_safe_puml_text(target.label)};')
                emitted.add(target.id)
            lines.append("endif")
        elif source:
            if "start" not in _kind(source.kind) and "final" not in _kind(source.kind) and "end" not in _kind(source.kind):
                lines.append(f':{_safe_puml_text(source.label)};')
                emitted.add(source.id)
            if target and "decision" in _kind(target.kind) and target.id not in emitted:
                lines.append(f'if ({_safe_puml_text(target.label)}?) then ({_safe_puml_text(edge.label or "si")})')
                lines.append("endif")
                emitted.add(target.id)
    if model.nodes and not model.edges:
        for node in model.nodes:
            if "start" not in _kind(node.kind) and "final" not in _kind(node.kind):
                lines.append(f':{_safe_puml_text(node.label)};')
    else:
        for node in model.nodes:
            if node.id in emitted:
                continue
            node_kind = _kind(node.kind)
            if "decision" in node_kind:
                lines.append(f'if ({_safe_puml_text(node.label)}?) then (si)')
                lines.append("endif")
            elif "start" not in node_kind and "final" not in node_kind and "end" not in node_kind:
                lines.append(f':{_safe_puml_text(node.label)};')
    lines.append("stop")
    lines.append("@enduml")
    return "\n".join(lines)


def _plantuml_component(model: DiagramModel) -> str:
    lines = ["@startuml", "skinparam componentStyle rectangle"]
    for group in model.groups:
        lines.append(f'package "{_safe_puml_text(group.label)}" {{')
        for node in [item for item in model.nodes if item.group_id == group.id]:
            lines.append(f'  component "{_safe_puml_text(node.label)}" as {node.id}')
        lines.append("}")
    grouped = {node.id for node in model.nodes if node.group_id}
    for node in model.nodes:
        if node.id not in grouped:
            lines.append(f'component "{_safe_puml_text(node.label)}" as {node.id}')
    for edge in model.edges:
        lines.append(f'{edge.source} ..> {edge.target} : {_safe_puml_text(edge.label or edge.kind)}')
    lines.append("@enduml")
    return "\n".join(lines)


def _plantuml_deployment(model: DiagramModel) -> str:
    lines = ["@startuml"]
    for node in model.nodes:
        node_kind = _kind(node.kind)
        if "artifact" in node_kind:
            lines.append(f'artifact "{_safe_puml_text(node.label)}" as {node.id}')
        else:
            lines.append(f'node "{_safe_puml_text(node.label)}" as {node.id}')
    for edge in model.edges:
        lines.append(f'{edge.source} --> {edge.target} : {_safe_puml_text(edge.label or edge.kind)}')
    lines.append("@enduml")
    return "\n".join(lines)


def _plantuml_package(model: DiagramModel) -> str:
    lines = ["@startuml"]
    for node in model.nodes:
        lines.append(f'package "{_safe_puml_text(node.label)}" as {node.id}')
    for edge in model.edges:
        lines.append(f"{edge.source} ..> {edge.target} : {_safe_puml_text(edge.label or edge.kind)}")
    lines.append("@enduml")
    return "\n".join(lines)


def _plantuml_sequence(model: DiagramModel) -> str:
    lines = ["@startuml"]
    for node in model.nodes:
        node_kind = _kind(node.kind)
        keyword = "actor" if "actor" in node_kind or "user" in node_kind else "participant"
        lines.append(f'{keyword} "{_safe_puml_text(node.label)}" as {node.id}')
    for edge in sorted(model.edges, key=lambda item: item.order or 0):
        lines.append(f"{edge.source} -> {edge.target} : {_safe_puml_text(edge.label or edge.kind)}")
    lines.append("@enduml")
    return "\n".join(lines)


def _plantuml_class(model: DiagramModel) -> str:
    lines = ["@startuml"]
    for node in model.nodes:
        keyword = "interface" if "interface" in _kind(node.kind) else "class"
        lines.append(f'{keyword} "{_safe_puml_text(node.label)}" as {node.id}')
    for edge in model.edges:
        arrow = "<|--" if "inherit" in _kind(edge.kind) else "-->"
        lines.append(f"{edge.source} {arrow} {edge.target} : {_safe_puml_text(edge.label or edge.kind)}")
    lines.append("@enduml")
    return "\n".join(lines)


def _plantuml_state(model: DiagramModel) -> str:
    node_kinds = {node.id: _kind(node.kind) for node in model.nodes}
    lines = ["@startuml"]
    for node in model.nodes:
        if "start" in node_kinds[node.id] or "end" in node_kinds[node.id] or "final" in node_kinds[node.id]:
            continue
        lines.append(f'state "{_safe_puml_text(node.label)}" as {node.id}')
    for edge in model.edges:
        source = "[*]" if "start" in node_kinds.get(edge.source, "") else edge.source
        target = "[*]" if "end" in node_kinds.get(edge.target, "") or "final" in node_kinds.get(edge.target, "") else edge.target
        lines.append(f"{source} --> {target} : {_safe_puml_text(edge.label or edge.kind)}")
    lines.append("@enduml")
    return "\n".join(lines)


def _c4_source(model: DiagramModel) -> str:
    lines = ["@startuml", "!include <C4/C4_Context>"]
    for node in model.nodes:
        kind = _kind(node.kind)
        label = _safe_puml_text(node.label)
        if "actor" in kind or "person" in kind:
            lines.append(f'Person({node.id}, "{label}")')
        elif "external" in kind:
            lines.append(f'System_Ext({node.id}, "{label}")')
        else:
            lines.append(f'System({node.id}, "{label}")')
    for edge in model.edges:
        lines.append(f'Rel({edge.source}, {edge.target}, "{_safe_puml_text(edge.label or edge.kind)}")')
    lines.append("@enduml")
    return "\n".join(lines)


def _bpmn_xml_id(prefix: str, value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "")).strip("_")
    if not normalized or not normalized[0].isalpha():
        normalized = f"{prefix}_{normalized or 'item'}"
    return normalized[:120]


def _bpmn_xml_tag(kind: str) -> str:
    normalized = _bpmn_kind(kind)
    if normalized == "start_event":
        return "startEvent"
    if normalized == "end_event":
        return "endEvent"
    if normalized == "exclusive_gateway":
        return "exclusiveGateway"
    if normalized == "parallel_gateway":
        return "parallelGateway"
    if normalized == "subprocess":
        return "subProcess"
    if normalized == "intermediate_event":
        return "intermediateThrowEvent"
    return "task"


def _bpmn_xml(model: DiagramModel) -> str:
    pools = _bpmn_effective_pools(model)
    assignments = _bpmn_node_assignments(model, pools)
    pool_by_id = {pool.id: pool for pool in pools}
    nodes_by_pool: dict[str, list[DiagramNode]] = {pool.id: [] for pool in pools}
    for node in model.nodes:
        pool_id, _lane_id = assignments.get(node.id, (pools[0].id, pools[0].lanes[0].id))
        nodes_by_pool.setdefault(pool_id, []).append(node)

    sequence_edges = [
        edge
        for edge in model.edges
        if assignments.get(edge.source, ("", ""))[0] == assignments.get(edge.target, ("", ""))[0]
        and _kind(edge.kind) != "message_flow"
    ]
    message_edges = [edge for edge in model.edges if edge not in sequence_edges]
    process_id = _bpmn_xml_id("process", model.diagram_key)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" id="definitions">',
    ]
    if len(pools) > 1 or any(pool.lanes for pool in pools):
        lines.append(f'  <bpmn:collaboration id="{escape(_bpmn_xml_id("collaboration", model.diagram_key))}">')
        for pool in pools:
            participant_id = _bpmn_xml_id("participant", pool.id)
            pool_process_id = _bpmn_xml_id("process", pool.id)
            lines.append(
                f'    <bpmn:participant id="{escape(participant_id)}" name="{escape(pool.label)}" processRef="{escape(pool_process_id)}" />'
            )
        for edge in message_edges:
            lines.append(
                f'    <bpmn:messageFlow id="{escape(edge.id)}" sourceRef="{escape(edge.source)}" targetRef="{escape(edge.target)}" name="{escape(edge.label or edge.kind)}" />'
            )
        lines.append("  </bpmn:collaboration>")

    for pool in pools:
        pool_process_id = _bpmn_xml_id("process", pool.id) if len(pools) > 1 else process_id
        lines.append(f'  <bpmn:process id="{escape(pool_process_id)}" name="{escape(pool.label)}" isExecutable="false">')
        if pool.lanes:
            lines.append(f'    <bpmn:laneSet id="{escape(_bpmn_xml_id("lane_set", pool.id))}">')
            for lane in pool.lanes:
                lane_id = _bpmn_xml_id("lane", lane.id)
                lines.append(f'      <bpmn:lane id="{escape(lane_id)}" name="{escape(lane.label)}">')
                for node in nodes_by_pool.get(pool.id, []):
                    _pool_id, node_lane_id = assignments.get(node.id, ("", ""))
                    if node_lane_id == lane.id:
                        lines.append(f"        <bpmn:flowNodeRef>{escape(node.id)}</bpmn:flowNodeRef>")
                lines.append("      </bpmn:lane>")
            lines.append("    </bpmn:laneSet>")
        for node in nodes_by_pool.get(pool.id, []):
            tag = _bpmn_xml_tag(node.kind)
            lines.append(f'    <bpmn:{tag} id="{escape(node.id)}" name="{escape(node.label)}" />')
        for edge in sequence_edges:
            source_pool = assignments.get(edge.source, ("", ""))[0]
            if source_pool != pool.id:
                continue
            lines.append(
                f'    <bpmn:sequenceFlow id="{escape(edge.id)}" sourceRef="{escape(edge.source)}" targetRef="{escape(edge.target)}" name="{escape(edge.label or edge.kind)}" />'
            )
        lines.append("  </bpmn:process>")
    lines.append("</bpmn:definitions>")
    return "\n".join(lines)


def render_source(model: DiagramModel) -> dict[str, str]:
    if model.notation == DiagramNotation.uml_use_case:
        return {"plantuml": _plantuml_use_case(model)}
    if model.notation == DiagramNotation.uml_activity:
        return {"plantuml": _plantuml_activity(model)}
    if model.notation == DiagramNotation.uml_component:
        return {"plantuml": _plantuml_component(model)}
    if model.notation == DiagramNotation.sequence:
        return {"plantuml": _plantuml_sequence(model)}
    if model.notation == DiagramNotation.class_diagram:
        return {"plantuml": _plantuml_class(model)}
    if model.notation == DiagramNotation.state:
        return {"plantuml": _plantuml_state(model)}
    if model.notation == DiagramNotation.deployment:
        return {"plantuml": _plantuml_deployment(model)}
    if model.notation == DiagramNotation.package:
        return {"plantuml": _plantuml_package(model)}
    if model.notation == DiagramNotation.c4:
        return {"plantuml": _c4_source(model)}
    if model.notation == DiagramNotation.bpmn:
        return {"bpmn_xml": _bpmn_xml(model)}
    return {"mermaid": render_mermaid(model)}


def _node_positions(model: DiagramModel, *, width: int, node_width: int, node_height: int) -> dict[str, tuple[float, float]]:
    columns = 3 if len(model.nodes) > 6 else 2
    horizontal_gap, vertical_gap = 72, 62
    positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(model.nodes):
        column = index % columns
        row = index // columns
        total_width = columns * node_width + (columns - 1) * horizontal_gap
        start_x = (width - total_width) / 2
        positions[node.id] = (start_x + column * (node_width + horizontal_gap), 92 + row * (node_height + vertical_gap))
    return positions


def _svg_root_attrs(model: DiagramModel) -> str:
    renderer_key = str(model.metadata.get("renderer_key") or "renderer.svg.generic.v1")
    return (
        f'data-diagram-key="{escape(model.diagram_key)}" '
        f'data-diagram-notation="{escape(model.notation.value)}" '
        f'data-renderer-key="{escape(renderer_key)}" '
        f'data-renderer-revision="{escape(RENDERER_REVISION)}"'
    )


def _text_lines(value: str, *, max_chars: int = 28, max_lines: int = 2) -> list[str]:
    words = str(value or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word[:max_chars]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if not lines:
        lines.append(str(value or "")[:max_chars])
    if len(words) and " ".join(lines) != str(value or "").strip() and lines:
        lines[-1] = f"{lines[-1][: max_chars - 1]}..."
    return lines


def _svg_multiline_text(
    lines: list[str],
    *,
    x: float,
    y: float,
    anchor: str = "middle",
    size: int = 14,
    weight: int = 700,
    fill: str = "#10172a",
) -> str:
    parts: list[str] = []
    for index, line in enumerate(lines):
        parts.append(
            f'<text x="{x:.1f}" y="{y + index * (size + 5):.1f}" text-anchor="{anchor}" '
            f'font-family="Inter,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
        )
    return "".join(parts)


def _actor_center(x: float, y: float) -> tuple[float, float]:
    return (x, y + 48)


def _svg_actor(node_id: str, label: str, *, x: float, y: float) -> str:
    label_lines = _text_lines(label, max_chars=20, max_lines=2)
    label_svg = _svg_multiline_text(label_lines, x=x, y=y + 98, size=13, weight=700)
    return (
        f'<g data-node-id="{escape(node_id)}" data-node-kind="actor" filter="url(#shadow)">'
        f'<circle cx="{x:.1f}" cy="{y+18:.1f}" r="13" fill="#fff" stroke="#3047b8" stroke-width="2"/>'
        f'<path d="M {x:.1f} {y+31:.1f} L {x:.1f} {y+58:.1f} '
        f'M {x-24:.1f} {y+42:.1f} L {x+24:.1f} {y+42:.1f} '
        f'M {x:.1f} {y+58:.1f} L {x-22:.1f} {y+78:.1f} '
        f'M {x:.1f} {y+58:.1f} L {x+22:.1f} {y+78:.1f}" '
        f'stroke="#3047b8" stroke-width="2.2" stroke-linecap="round" fill="none"/>'
        f"{label_svg}</g>"
    )


def _svg_use_case(node_id: str, label: str, *, x: float, y: float, width: int, height: int) -> str:
    lines = _text_lines(label, max_chars=28, max_lines=2)
    text_y = y + height / 2 - ((len(lines) - 1) * 9) + 5
    return (
        f'<g data-node-id="{escape(node_id)}" data-node-kind="use_case" filter="url(#shadow)">'
        f'<ellipse cx="{x+width/2:.1f}" cy="{y+height/2:.1f}" rx="{width/2:.1f}" ry="{height/2:.1f}" '
        f'fill="#ffffff" stroke="#3047b8" stroke-width="2"/>'
        f'{_svg_multiline_text(lines, x=x + width / 2, y=text_y, size=14, weight=700)}'
        f"</g>"
    )


def _render_uml_use_case_svg(model: DiagramModel) -> str:
    width = 1120
    actors = [node for node in model.nodes if "actor" in _kind(node.kind) or "person" in _kind(node.kind)]
    use_cases = [node for node in model.nodes if node not in actors]
    if not use_cases:
        use_cases = model.nodes
        actors = []

    use_case_width = 260
    use_case_height = 76
    columns = 2 if len(use_cases) > 3 else 1
    rows = max(1, math.ceil(max(len(use_cases), 1) / columns))
    boundary_x = 290
    boundary_y = 76
    boundary_width = 760
    boundary_height = max(360, 118 + rows * 118)
    height = max(460, boundary_y + boundary_height + 52)

    use_case_positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(use_cases):
        column = index % columns
        row = index // columns
        gap_x = 70
        total_width = columns * use_case_width + (columns - 1) * gap_x
        start_x = boundary_x + (boundary_width - total_width) / 2
        use_case_positions[node.id] = (
            start_x + column * (use_case_width + gap_x),
            boundary_y + 78 + row * 118,
        )

    left_actors = actors[: math.ceil(len(actors) / 2)]
    right_actors = actors[math.ceil(len(actors) / 2) :]
    actor_positions: dict[str, tuple[float, float]] = {}

    def assign_actor_positions(items: list[DiagramNode], *, side: str) -> None:
        if not items:
            return
        lane_x = 118 if side == "left" else width - 118
        usable_top = boundary_y + 52
        usable_height = max(240, boundary_height - 120)
        gap_y = usable_height / max(len(items), 1)
        for index, node in enumerate(items):
            actor_positions[node.id] = (lane_x, usable_top + gap_y * index + 12)

    assign_actor_positions(left_actors, side="left")
    assign_actor_positions(right_actors, side="right")

    centers: dict[str, tuple[float, float]] = {
        node.id: (x + use_case_width / 2, y + use_case_height / 2)
        for node, (x, y) in ((node, use_case_positions[node.id]) for node in use_cases if node.id in use_case_positions)
    }
    centers.update({node_id: _actor_center(x, y) for node_id, (x, y) in actor_positions.items()})

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" {_svg_root_attrs(model)}>',
        f'<title id="title">{escape(model.title)}</title>',
        f'<desc id="desc">{escape(model.description or "UML Use Case Diagram")}</desc>',
        '<defs><marker id="uml-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#60708a"/></marker><filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#172033" flood-opacity="0.12"/></filter></defs>',
        '<rect width="100%" height="100%" rx="20" fill="#f8f9fc"/>',
        f'<text x="40" y="42" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="800" letter-spacing="3" fill="#3047b8">UML USE CASE</text>',
        f'<rect data-node-kind="system_boundary" x="{boundary_x}" y="{boundary_y}" width="{boundary_width}" height="{boundary_height}" rx="14" fill="#ffffff" stroke="#cbd3e1" stroke-width="1.5" stroke-dasharray="8 7"/>',
        f'<text x="{boundary_x+22}" y="{boundary_y+34}" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="800" fill="#10172a">{escape(model.title)}</text>',
    ]

    for edge in model.edges:
        if edge.source not in centers or edge.target not in centers:
            continue
        x1, y1 = centers[edge.source]
        x2, y2 = centers[edge.target]
        edge_kind = _kind(edge.kind)
        dashed = ' stroke-dasharray="7 7"' if edge_kind in {"include", "extend", "dependency"} else ""
        marker = ' marker-end="url(#uml-arrow)"' if edge_kind in {"include", "extend", "dependency"} else ""
        parts.append(
            f'<path data-edge-id="{escape(edge.id)}" d="M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}" '
            f'stroke="#60708a" stroke-width="1.8" fill="none"{dashed}{marker}/>'
        )
        label = edge.label or ("<<include>>" if edge_kind == "include" else "<<extend>>" if edge_kind == "extend" else "")
        if label:
            parts.append(
                f'<text x="{(x1+x2)/2:.1f}" y="{(y1+y2)/2-8:.1f}" text-anchor="middle" '
                f'font-family="Inter,Arial,sans-serif" font-size="11" fill="#44506a">{escape(label[:42])}</text>'
            )

    for node in use_cases:
        x, y = use_case_positions[node.id]
        parts.append(_svg_use_case(node.id, node.label, x=x, y=y, width=use_case_width, height=use_case_height))
    for node in actors:
        x, y = actor_positions[node.id]
        parts.append(_svg_actor(node.id, node.label, x=x, y=y))

    parts.append("</svg>")
    return "".join(parts)


def _render_uml_activity_svg(model: DiagramModel) -> str:
    measurements = {node.id: measure_generic_node(node, model.notation) for node in model.nodes}
    layout = compute_layered_layout(model, measurements, min_width=1180, gap_x=122, gap_y=72)
    routes = route_layered_edges(model, layout.positions, measurements)
    width = layout.width
    height = max(layout.height, 520)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" {_svg_root_attrs(model)}>',
        f'<title id="title">{escape(model.title)}</title>',
        f'<desc id="desc">{escape(model.description or "UML Activity Diagram")}</desc>',
        '<defs><marker id="uml-activity-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#526176"/></marker><filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#172033" flood-opacity="0.12"/></filter></defs>',
        '<rect width="100%" height="100%" rx="20" fill="#f8f9fc"/>',
        '<text x="40" y="44" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="900" letter-spacing="3" fill="#3047b8">UML ACTIVITY</text>',
    ]
    for edge in model.edges:
        route = routes.get(edge.id)
        if not route:
            continue
        points = route.points
        path = f"M {points[0][0]:.1f} {points[0][1]:.1f} " + " ".join(
            f"L {point[0]:.1f} {point[1]:.1f}" for point in points[1:]
        )
        parts.append(
            f'<path data-edge-id="{escape(edge.id)}" data-edge-kind="{escape(edge.kind)}" d="{path}" stroke="#526176" stroke-width="2" fill="none" marker-end="url(#uml-activity-arrow)"/>'
        )
        if edge.label:
            label_x, label_y = route.label_position
            label = escape(edge.label[:44])
            label_width = max(42, min(220, len(edge.label[:44]) * 7 + 18))
            parts.append(
                f'<rect x="{label_x-label_width/2:.1f}" y="{label_y-15:.1f}" width="{label_width:.1f}" height="22" rx="11" fill="#f8f9fc" stroke="#d8deea" stroke-width="1"/>'
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="11" fill="#44506a">{label}</text>'
            )
    for node in model.nodes:
        x, y = layout.positions[node.id]
        size = measurements[node.id]
        parts.append(
            _svg_node(
                model,
                node.id,
                node.label,
                node.kind,
                x,
                y,
                size.width,
                size.height,
                label_lines=list(size.label_lines),
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _svg_node(
    model: DiagramModel,
    node_id: str,
    label: str,
    kind: str,
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    label_lines: list[str] | None = None,
) -> str:
    normalized_kind = _kind(kind)
    label_lines = label_lines or _text_lines(label, max_chars=max(16, int((width - 44) / 7)), max_lines=3)
    label_text = escape(label[:42])
    kind_text = escape(kind[:42])
    if model.notation == DiagramNotation.uml_use_case:
        if "actor" in normalized_kind:
            cx = x + width / 2
            return (
                f'<g filter="url(#shadow)"><circle cx="{cx:.1f}" cy="{y+18:.1f}" r="13" fill="#fff" stroke="#3047b8" stroke-width="2"/>'
                f'<path d="M {cx:.1f} {y+31:.1f} L {cx:.1f} {y+58:.1f} M {cx-22:.1f} {y+41:.1f} L {cx+22:.1f} {y+41:.1f} M {cx:.1f} {y+58:.1f} L {cx-20:.1f} {y+76:.1f} M {cx:.1f} {y+58:.1f} L {cx+20:.1f} {y+76:.1f}" stroke="#3047b8" stroke-width="2" fill="none"/>'
                f'<text x="{cx:.1f}" y="{y+100:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" fill="#10172a">{label_text}</text></g>'
            )
        return (
            f'<g filter="url(#shadow)"><ellipse cx="{x+width/2:.1f}" cy="{y+height/2:.1f}" rx="{width/2:.1f}" ry="{height/2:.1f}" fill="#fff" stroke="#3047b8" stroke-width="2"/>'
            f'<text x="{x+width/2:.1f}" y="{y+height/2-2:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="700" fill="#10172a">{label_text}</text>'
            f'<text x="{x+width/2:.1f}" y="{y+height/2+20:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="11" fill="#69748b">use case</text></g>'
        )
    if model.notation in {DiagramNotation.uml_activity, DiagramNotation.bpmn}:
        if "decision" in normalized_kind or "gateway" in normalized_kind:
            points = f"{x+width/2:.1f},{y:.1f} {x+width:.1f},{y+height/2:.1f} {x+width/2:.1f},{y+height:.1f} {x:.1f},{y+height/2:.1f}"
            text_y = y + height / 2 - ((len(label_lines) - 1) * 8) + 4
            return (
                f'<g filter="url(#shadow)"><polygon points="{points}" fill="#fff9eb" stroke="#c47a1c" stroke-width="2"/>'
                f'{_svg_multiline_text(label_lines, x=x + width / 2, y=text_y, size=12, weight=800, fill="#10172a")}</g>'
            )
        if "start" in normalized_kind or "event" in normalized_kind or "final" in normalized_kind or "end" in normalized_kind:
            return (
                f'<g filter="url(#shadow)"><circle cx="{x+width/2:.1f}" cy="{y+height/2:.1f}" r="{min(width,height)/2-4:.1f}" fill="#eff8f0" stroke="#2d7d46" stroke-width="2"/>'
                f'{_svg_multiline_text(label_lines[:2], x=x + width / 2, y=y + height / 2 + 5, size=12, weight=800, fill="#10172a")}</g>'
            )
    if model.notation == DiagramNotation.uml_component:
        return (
            f'<g filter="url(#shadow)"><rect x="{x:.1f}" y="{y:.1f}" width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#3047b8" stroke-width="1.8"/>'
            f'<rect x="{x+width-36:.1f}" y="{y+12:.1f}" width="20" height="12" fill="#eef2ff" stroke="#3047b8"/>'
            f'<rect x="{x+width-36:.1f}" y="{y+30:.1f}" width="20" height="12" fill="#eef2ff" stroke="#3047b8"/>'
            f'<text x="{x+22:.1f}" y="{y+34:.1f}" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="700" fill="#10172a">{label_text}</text>'
            f'<text x="{x+22:.1f}" y="{y+56:.1f}" font-family="Inter,Arial,sans-serif" font-size="11" fill="#69748b">{kind_text}</text></g>'
        )
    if model.notation == DiagramNotation.deployment:
        return (
            f'<g filter="url(#shadow)"><path d="M {x:.1f} {y+12:.1f} L {x+20:.1f} {y:.1f} H {x+width:.1f} V {y+height-12:.1f} L {x+width-20:.1f} {y+height:.1f} H {x:.1f} Z" fill="#ffffff" stroke="#3047b8" stroke-width="1.8"/>'
            f'<text x="{x+24:.1f}" y="{y+36:.1f}" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="700" fill="#10172a">{label_text}</text>'
            f'<text x="{x+24:.1f}" y="{y+58:.1f}" font-family="Inter,Arial,sans-serif" font-size="11" fill="#69748b">deployment node</text></g>'
        )
    if model.notation == DiagramNotation.package:
        return (
            f'<g filter="url(#shadow)"><path d="M {x:.1f} {y+14:.1f} H {x+86:.1f} L {x+100:.1f} {y:.1f} H {x+width:.1f} V {y+height:.1f} H {x:.1f} Z" fill="#ffffff" stroke="#3047b8" stroke-width="1.8"/>'
            f'<text x="{x+22:.1f}" y="{y+40:.1f}" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="700" fill="#10172a">{label_text}</text></g>'
        )
    return (
        f'<g filter="url(#shadow)"><rect x="{x:.1f}" y="{y:.1f}" width="{width}" height="{height}" rx="10" fill="#ffffff" stroke="#cbd3e1"/></g>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="5" height="{height}" rx="2.5" fill="#3047b8"/>'
        f'{_svg_multiline_text(label_lines, x=x + 22, y=y + 28, anchor="start", size=14, weight=800, fill="#10172a")}'
        f'<text x="{x+22:.1f}" y="{y+height-15:.1f}" font-family="Inter,Arial,sans-serif" font-size="11" fill="#69748b">{kind_text}</text>'
    )


def _bpmn_kind(kind: str) -> str:
    normalized = _kind(kind)
    if "start" in normalized:
        return "start_event"
    if "end" in normalized or "final" in normalized:
        return "end_event"
    if "parallel" in normalized:
        return "parallel_gateway"
    if "gateway" in normalized or "decision" in normalized:
        return "exclusive_gateway"
    if "subprocess" in normalized or "sub_process" in normalized:
        return "subprocess"
    if "event" in normalized:
        return "intermediate_event"
    return "task"


def _topological_bpmn_order(model: DiagramModel) -> list[DiagramNode]:
    nodes_by_id = {node.id: node for node in model.nodes}
    incoming: dict[str, int] = {node.id: 0 for node in model.nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in model.nodes}
    for edge in model.edges:
        if edge.source not in nodes_by_id or edge.target not in nodes_by_id:
            continue
        incoming[edge.target] += 1
        outgoing[edge.source].append(edge.target)

    start_ids = [
        node.id
        for node in model.nodes
        if _bpmn_kind(node.kind) == "start_event" or incoming[node.id] == 0
    ]
    queue = list(dict.fromkeys(start_ids))
    visited: set[str] = set()
    ordered: list[DiagramNode] = []
    while queue:
        node_id = queue.pop(0)
        if node_id in visited or node_id not in nodes_by_id:
            continue
        visited.add(node_id)
        ordered.append(nodes_by_id[node_id])
        for target in outgoing[node_id]:
            if target not in visited:
                queue.append(target)

    ordered.extend(node for node in model.nodes if node.id not in visited)
    return ordered


def _bpmn_node_size(kind: str) -> tuple[int, int]:
    bpmn_kind = _bpmn_kind(kind)
    if bpmn_kind in {"start_event", "end_event", "intermediate_event"}:
        return (68, 68)
    if bpmn_kind in {"exclusive_gateway", "parallel_gateway"}:
        return (86, 86)
    return (230, 82)


def _bpmn_node_size_for_label(kind: str, label: str) -> tuple[int, int]:
    bpmn_kind = _bpmn_kind(kind)
    if bpmn_kind in {"start_event", "end_event", "intermediate_event"}:
        return _bpmn_node_size(kind)
    if bpmn_kind in {"exclusive_gateway", "parallel_gateway"}:
        lines = _text_lines(label, max_chars=26, max_lines=2)
        max_line = max((len(line) for line in lines), default=16)
        width = max(104, min(190, 86 + max(0, max_line - 14) * 8))
        return (width, max(width, 92))
    lines = _text_lines(label, max_chars=30, max_lines=3)
    max_line = max((len(line) for line in lines), default=16)
    width = max(240, min(340, 224 + max(0, max_line - 22) * 7))
    height = max(86, min(124, 74 + (len(lines) - 1) * 18))
    return (width, height)


def _bpmn_effective_pools(model: DiagramModel) -> list[DiagramPool]:
    if model.pools:
        return [
            pool if pool.lanes else pool.model_copy(update={"lanes": [DiagramLane(id=f"{pool.id}_lane", label=pool.label)]})
            for pool in model.pools
        ]
    return [DiagramPool(id="process", label="Proceso", lanes=[DiagramLane(id="process_lane", label="Proceso")])]


def _bpmn_lane_lookup(pools: list[DiagramPool]) -> dict[str, tuple[DiagramPool, DiagramLane]]:
    lookup: dict[str, tuple[DiagramPool, DiagramLane]] = {}
    for pool in pools:
        for lane in pool.lanes:
            lookup[lane.id] = (pool, lane)
    return lookup


def _bpmn_node_assignments(model: DiagramModel, pools: list[DiagramPool]) -> dict[str, tuple[str, str]]:
    lane_lookup = _bpmn_lane_lookup(pools)
    pool_lookup = {pool.id: pool for pool in pools}
    default_pool = pools[0]
    default_lane = default_pool.lanes[0]
    assignments: dict[str, tuple[str, str]] = {}
    for node in model.nodes:
        pool_id = str(node.metadata.get("pool_id") or "").strip()
        lane_id = str(node.metadata.get("lane_id") or "").strip()
        if lane_id in lane_lookup:
            lane_pool, _lane = lane_lookup[lane_id]
            assignments[node.id] = (lane_pool.id, lane_id)
            continue
        if pool_id in pool_lookup and pool_lookup[pool_id].lanes:
            assignments[node.id] = (pool_id, pool_lookup[pool_id].lanes[0].id)
            continue
        assignments[node.id] = (default_pool.id, default_lane.id)
    return assignments


def _bpmn_positions(
    model: DiagramModel,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, tuple[int, int]],
    list[dict[str, object]],
    dict[str, tuple[str, str]],
    int,
    int,
]:
    pools = _bpmn_effective_pools(model)
    assignments = _bpmn_node_assignments(model, pools)
    ordered = _topological_bpmn_order(model)
    lane_nodes: dict[str, list[DiagramNode]] = {lane.id: [] for pool in pools for lane in pool.lanes}
    for node in ordered:
        _pool_id, lane_id = assignments.get(node.id, (pools[0].id, pools[0].lanes[0].id))
        lane_nodes.setdefault(lane_id, []).append(node)

    pool_x = 44
    pool_y = 88
    pool_gap = 30
    lane_label_width = 142
    lane_min_height = 170
    pool_header_height = 46
    widest_node = max((_bpmn_node_size_for_label(node.kind, node.label)[0] for node in model.nodes), default=230)
    slot_width = max(340, widest_node + 116)
    content_x = pool_x + lane_label_width + 34
    max_lane_items = max((len(items) for items in lane_nodes.values()), default=1)
    canvas_width = max(1180, int(content_x + max_lane_items * slot_width + 76))
    positions: dict[str, tuple[float, float]] = {}
    node_sizes: dict[str, tuple[int, int]] = {}
    pool_layouts: list[dict[str, object]] = []
    cursor_y = pool_y
    for pool in pools:
        lane_layouts: list[dict[str, object]] = []
        lane_item_counts = [len(lane_nodes.get(lane.id, [])) for lane in pool.lanes]
        lane_height = max(lane_min_height, 130 + min(3, max(lane_item_counts, default=1)) * 18)
        pool_height = pool_header_height + max(1, len(pool.lanes)) * lane_height
        for lane_index, lane in enumerate(pool.lanes):
            lane_y = cursor_y + pool_header_height + lane_index * lane_height
            lane_layouts.append({"id": lane.id, "label": lane.label, "x": pool_x, "y": lane_y, "height": lane_height})
            for node_index, node in enumerate(lane_nodes.get(lane.id, [])):
                node_width, node_height = _bpmn_node_size_for_label(node.kind, node.label)
                node_sizes[node.id] = (node_width, node_height)
                x = content_x + node_index * slot_width + max(0, (widest_node - node_width) / 2)
                y = lane_y + lane_height / 2 - node_height / 2 - (8 if node_height <= 68 else 0)
                positions[node.id] = (x, y)
        pool_layouts.append(
            {
                "id": pool.id,
                "label": pool.label,
                "x": pool_x,
                "y": cursor_y,
                "width": canvas_width - (pool_x * 2),
                "height": pool_height,
                "lanes": lane_layouts,
            }
        )
        cursor_y += pool_height + pool_gap
    return positions, node_sizes, pool_layouts, assignments, canvas_width, max(420, int(cursor_y + 44))


def _bpmn_connection_point(
    positions: dict[str, tuple[float, float]],
    node_sizes: dict[str, tuple[int, int]],
    node_id: str,
    *,
    side: str,
) -> tuple[float, float]:
    x, y = positions[node_id]
    width, height = node_sizes.get(node_id, (230, 82))
    if side == "left":
        return (x, y + height / 2)
    return (x + width, y + height / 2)


def _svg_bpmn_node(node_id: str, label: str, kind: str, x: float, y: float, width: int, height: int) -> str:
    bpmn_kind = _bpmn_kind(kind)
    kind_text = escape(bpmn_kind.replace("_", " "))
    cx = x + width / 2
    cy = y + height / 2
    label_lines = _text_lines(label, max_chars=22, max_lines=2)
    if bpmn_kind == "start_event":
        return (
            f'<g data-node-id="{escape(node_id)}" data-node-kind="start_event" filter="url(#shadow)">'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="26" fill="#f0fbf3" stroke="#2f7d46" stroke-width="2.2"/>'
            f'{_svg_multiline_text(label_lines, x=cx, y=y + height + 22, size=12, weight=800, fill="#10172a")}'
            "</g>"
        )
    if bpmn_kind == "end_event":
        return (
            f'<g data-node-id="{escape(node_id)}" data-node-kind="end_event" filter="url(#shadow)">'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="27" fill="#fff4f2" stroke="#b9332c" stroke-width="3.2"/>'
            f'{_svg_multiline_text(label_lines, x=cx, y=y + height + 22, size=12, weight=800, fill="#10172a")}'
            "</g>"
        )
    if bpmn_kind == "intermediate_event":
        return (
            f'<g data-node-id="{escape(node_id)}" data-node-kind="intermediate_event" filter="url(#shadow)">'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="27" fill="#f8fafc" stroke="#60708a" stroke-width="2"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="22" fill="none" stroke="#60708a" stroke-width="1.4"/>'
            f'{_svg_multiline_text(label_lines, x=cx, y=y + height + 22, size=12, weight=800, fill="#10172a")}'
            "</g>"
        )
    if bpmn_kind == "exclusive_gateway":
        points = f"{cx:.1f},{y:.1f} {x+width:.1f},{cy:.1f} {cx:.1f},{y+height:.1f} {x:.1f},{cy:.1f}"
        return (
            f'<g data-node-id="{escape(node_id)}" data-node-kind="exclusive_gateway" filter="url(#shadow)">'
            f'<polygon points="{points}" fill="#fff9eb" stroke="#b86b12" stroke-width="2.1"/>'
            f'<path d="M {cx-10:.1f} {cy-10:.1f} L {cx+10:.1f} {cy+10:.1f} M {cx+10:.1f} {cy-10:.1f} L {cx-10:.1f} {cy+10:.1f}" stroke="#b86b12" stroke-width="2.5" stroke-linecap="round"/>'
            f'{_svg_multiline_text(label_lines, x=cx, y=y + height + 22, size=12, weight=800, fill="#10172a")}'
            "</g>"
        )
    marker = (
        f'<rect x="{cx-8:.1f}" y="{y+height-16:.1f}" width="16" height="10" rx="2" fill="#eef2ff" stroke="#3047b8" stroke-width="1.2"/>'
        f'<text x="{cx:.1f}" y="{y+height-8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="10" font-weight="900" fill="#3047b8">+</text>'
        if bpmn_kind == "subprocess"
        else ""
    )
    return (
        f'<g data-node-id="{escape(node_id)}" data-node-kind="{escape(bpmn_kind)}" filter="url(#shadow)">'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width}" height="{height}" rx="14" fill="#ffffff" stroke="#3047b8" stroke-width="1.8"/>'
        f'{_svg_multiline_text(label_lines, x=x + 18, y=y + 31, anchor="start", size=14, weight=800, fill="#10172a")}'
        f'<text x="{x+18:.1f}" y="{y+53:.1f}" font-family="Inter,Arial,sans-serif" font-size="11" fill="#69748b">{kind_text}</text>'
        f"{marker}</g>"
    )


def _render_bpmn_svg(model: DiagramModel) -> str:
    positions, node_sizes, pool_layouts, assignments, width, height = _bpmn_positions(model)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" {_svg_root_attrs(model)}>',
        f'<title id="title">{escape(model.title)}</title>',
        f'<desc id="desc">{escape(model.description or "BPMN 2.0 process diagram")}</desc>',
        '<defs><marker id="bpmn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#526176"/></marker><marker id="bpmn-message-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 1 1 L 9 5 L 1 9 Z" fill="#ffffff" stroke="#526176" stroke-width="1.4"/></marker><filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#172033" flood-opacity="0.12"/></filter></defs>',
        '<rect width="100%" height="100%" rx="20" fill="#f8f9fc"/>',
        '<text x="40" y="44" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="900" letter-spacing="3" fill="#3047b8">BPMN 2.0</text>',
    ]
    for pool_layout in pool_layouts:
        pool_x = float(pool_layout["x"])
        pool_y = float(pool_layout["y"])
        pool_width = float(pool_layout["width"])
        pool_height = float(pool_layout["height"])
        pool_label = escape(str(pool_layout["label"]))
        pool_id = escape(str(pool_layout["id"]))
        parts.append(
            f'<rect data-bpmn-kind="pool" data-pool-id="{pool_id}" x="{pool_x:.1f}" y="{pool_y:.1f}" width="{pool_width:.1f}" height="{pool_height:.1f}" rx="18" fill="#ffffff" stroke="#cbd3e1" stroke-width="1.5"/>'
        )
        parts.append(
            f'<rect x="{pool_x:.1f}" y="{pool_y:.1f}" width="{pool_width:.1f}" height="46" rx="18" fill="#eef2ff" stroke="#cbd3e1" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pool_x+22:.1f}" y="{pool_y+29:.1f}" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="900" letter-spacing="2.3" fill="#3047b8">{pool_label}</text>'
        )
        for lane_layout in pool_layout["lanes"]:  # type: ignore[index]
            lane_x = float(lane_layout["x"])
            lane_y = float(lane_layout["y"])
            lane_height = float(lane_layout["height"])
            lane_label = escape(str(lane_layout["label"]))
            lane_id = escape(str(lane_layout["id"]))
            parts.append(
                f'<rect data-bpmn-kind="lane" data-pool-id="{pool_id}" data-lane-id="{lane_id}" x="{lane_x:.1f}" y="{lane_y:.1f}" width="{pool_width:.1f}" height="{lane_height:.1f}" fill="#fbfcff" stroke="#e4e8f2" stroke-width="1"/>'
            )
            parts.append(
                f'<rect data-bpmn-kind="lane-label" data-pool-id="{pool_id}" data-lane-id="{lane_id}" x="{lane_x:.1f}" y="{lane_y:.1f}" width="142" height="{lane_height:.1f}" fill="#f3f6ff" stroke="#d8dff0" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{lane_x+72:.1f}" y="{lane_y+lane_height/2:.1f}" transform="rotate(-90 {lane_x+72:.1f} {lane_y+lane_height/2:.1f})" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="11" font-weight="900" letter-spacing="2.2" fill="#3047b8">{lane_label}</text>'
            )
    nodes_by_id = {node.id: node for node in model.nodes}
    for edge in model.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        source = nodes_by_id[edge.source]
        target = nodes_by_id[edge.target]
        x1, y1 = _bpmn_connection_point(positions, node_sizes, edge.source, side="right")
        x2, y2 = _bpmn_connection_point(positions, node_sizes, edge.target, side="left")
        mid_x = (x1 + x2) / 2
        source_pool = assignments.get(edge.source, ("", ""))[0]
        target_pool = assignments.get(edge.target, ("", ""))[0]
        is_message_flow = _kind(edge.kind) == "message_flow" or source_pool != target_pool
        edge_kind = "message_flow" if is_message_flow else "sequence_flow"
        dash_attr = ' stroke-dasharray="7 7"' if is_message_flow else ""
        marker_id = "bpmn-message-arrow" if is_message_flow else "bpmn-arrow"
        parts.append(
            f'<path data-edge-id="{escape(edge.id)}" data-edge-kind="{edge_kind}" d="M {x1:.1f} {y1:.1f} C {mid_x:.1f} {y1:.1f}, {mid_x:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}" '
            f'stroke="#526176" stroke-width="2" fill="none"{dash_attr} marker-end="url(#{marker_id})"/>'
        )
        if edge.label:
            parts.append(
                f'<text x="{mid_x:.1f}" y="{(y1+y2)/2-10:.1f}" text-anchor="middle" '
                f'font-family="Inter,Arial,sans-serif" font-size="11" fill="#44506a">{escape(edge.label[:46])}</text>'
            )
    for node in model.nodes:
        x, y = positions[node.id]
        node_width, node_height = node_sizes.get(node.id, _bpmn_node_size(node.kind))
        parts.append(_svg_bpmn_node(node.id, node.label, node.kind, x, y, node_width, node_height))
    parts.append("</svg>")
    return "".join(parts)


def render_svg(model: DiagramModel) -> str:
    if model.notation == DiagramNotation.bpmn:
        return _render_bpmn_svg(model)
    if model.notation == DiagramNotation.uml_use_case:
        return _render_uml_use_case_svg(model)
    if model.notation == DiagramNotation.uml_activity:
        return _render_uml_activity_svg(model)

    node_measurements = {node.id: measure_generic_node(node, model.notation) for node in model.nodes}
    layout = compute_layered_layout(model, node_measurements)
    width = layout.width
    height = layout.height
    positions = layout.positions
    routes = route_layered_edges(model, positions, node_measurements)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" {_svg_root_attrs(model)}>',
        f'<title id="title">{escape(model.title)}</title>',
        f'<desc id="desc">{escape(model.description)}</desc>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#60708a"/></marker><filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#172033" flood-opacity="0.12"/></filter></defs>',
        '<rect width="100%" height="100%" rx="20" fill="#f8f9fc"/>',
        f'<text x="40" y="42" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="800" letter-spacing="3" fill="#3047b8">{escape(model.notation.value.upper())}</text>',
    ]
    for edge in model.edges:
        route = routes.get(edge.id)
        if not route:
            continue
        points = route.points
        path = f"M {points[0][0]:.1f} {points[0][1]:.1f} " + " ".join(
            f"L {point[0]:.1f} {point[1]:.1f}" for point in points[1:]
        )
        parts.append(
            f'<path data-edge-id="{escape(edge.id)}" d="{path}" stroke="#60708a" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'
        )
        if edge.label:
            label_x, label_y = route.label_position
            label = escape(edge.label[:48])
            label_width = max(56, min(260, len(edge.label[:48]) * 7 + 18))
            parts.append(
                f'<rect x="{label_x-label_width/2:.1f}" y="{label_y-15:.1f}" width="{label_width:.1f}" height="22" rx="11" fill="#f8f9fc" stroke="#d8deea" stroke-width="1"/>'
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="11" fill="#44506a">{label}</text>'
            )
    for node in model.nodes:
        x, y = positions[node.id]
        measurement = node_measurements.get(node.id)
        width_for_node = measurement.width if measurement else node_width
        height_for_node = measurement.height if measurement else node_height
        label_lines = list(measurement.label_lines) if measurement else None
        parts.append(
            _svg_node(
                model,
                node.id,
                node.label,
                node.kind,
                x,
                y,
                width_for_node,
                height_for_node,
                label_lines=label_lines,
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def render_diagram(model: DiagramModel) -> dict[str, str]:
    source_renderings = render_source(model)
    presentation = {
        "schema_version": "diagram-presentation.v1",
        "diagram_key": model.diagram_key,
        "title": model.title,
        "standard": model.metadata.get("standard") or model.notation.value,
        "rendering_format": "svg",
        "protected_view": True,
        "summary": model.description,
        "legend": [item.model_dump(mode="json") for item in model.legend],
        "traceability_refs": model.source_refs,
    }
    return {
        "json": json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2),
        "mermaid": render_mermaid(model),
        "svg": render_svg(model),
        "presentation": json.dumps(presentation, ensure_ascii=False, indent=2),
        **source_renderings,
    }
