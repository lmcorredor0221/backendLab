from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.layout_sizing import DiagramNodeSize


@dataclass(frozen=True)
class LayeredLayout:
    positions: dict[str, tuple[float, float]]
    width: int
    height: int
    layers: dict[str, int]


@dataclass(frozen=True)
class EdgeRoute:
    edge_id: str
    points: tuple[tuple[float, float], ...]
    label_position: tuple[float, float]


def _node_layers(model: DiagramModel) -> dict[str, int]:
    node_ids = {node.id for node in model.nodes}
    incoming: dict[str, int] = {node.id: 0 for node in model.nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in model.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        outgoing[edge.source].append(edge.target)
        incoming[edge.target] += 1

    queue = deque([node.id for node in model.nodes if incoming[node.id] == 0])
    layers: dict[str, int] = {node.id: 0 for node in model.nodes}
    visited: set[str] = set()
    while queue:
        node_id = queue.popleft()
        visited.add(node_id)
        for target in outgoing.get(node_id, []):
            layers[target] = max(layers.get(target, 0), layers[node_id] + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)

    if len(visited) != len(node_ids):
        for index, node in enumerate(model.nodes):
            if node.id not in visited:
                predecessor_layers = [
                    layers.get(edge.source, 0)
                    for edge in model.edges
                    if edge.target == node.id and edge.source in layers
                ]
                layers[node.id] = max(predecessor_layers, default=index % 4) + (1 if predecessor_layers else 0)
    return layers


def compute_layered_layout(
    model: DiagramModel,
    node_sizes: dict[str, DiagramNodeSize],
    *,
    min_width: int = 1400,
    margin_x: int = 72,
    margin_y: int = 100,
    gap_x: int = 140,
    gap_y: int = 80,
) -> LayeredLayout:
    if not model.nodes:
        return LayeredLayout(positions={}, width=min_width, height=360, layers={})

    direction = str(model.direction or "LR").upper()
    is_vertical = direction in {"TB", "TD", "BT"}
    layers = _node_layers(model)
    nodes_by_layer: dict[int, list[str]] = defaultdict(list)
    for node in model.nodes:
        nodes_by_layer[layers.get(node.id, 0)].append(node.id)

    sorted_layers = sorted(nodes_by_layer, reverse=direction in {"BT", "RL"})
    layer_widths: dict[int, int] = {}
    layer_heights: dict[int, int] = {}
    for layer in sorted_layers:
        ids = nodes_by_layer[layer]
        if is_vertical:
            layer_widths[layer] = sum(node_sizes[node_id].width for node_id in ids) + max(0, len(ids) - 1) * gap_x
            layer_heights[layer] = max((node_sizes[node_id].height for node_id in ids), default=96)
        else:
            layer_widths[layer] = max((node_sizes[node_id].width for node_id in ids), default=260)
            layer_heights[layer] = sum(node_sizes[node_id].height for node_id in ids) + max(0, len(ids) - 1) * gap_y

    positions: dict[str, tuple[float, float]] = {}

    if is_vertical:
        total_height = (
            margin_y * 2
            + sum(layer_heights[layer] for layer in sorted_layers)
            + max(0, len(sorted_layers) - 1) * gap_y
        )
        canvas_width = max(min_width, int(margin_x * 2 + max(layer_widths.values(), default=0)))
        canvas_height = max(420, int(total_height))

        y = margin_y
        for layer in sorted_layers:
            ids = nodes_by_layer[layer]
            layer_width = layer_widths[layer]
            layer_height = layer_heights[layer]
            x = margin_x + max(0, (canvas_width - margin_x * 2 - layer_width) / 2)
            for node_id in ids:
                size = node_sizes[node_id]
                positions[node_id] = (x, y + (layer_height - size.height) / 2)
                x += size.width + gap_x
            y += layer_height + gap_y
    else:
        total_width = (
            margin_x * 2
            + sum(layer_widths[layer] for layer in sorted_layers)
            + max(0, len(sorted_layers) - 1) * gap_x
        )
        canvas_width = max(min_width, int(total_width))
        canvas_height = max(420, int(margin_y * 2 + max(layer_heights.values(), default=0)))

        x = margin_x
        for layer in sorted_layers:
            ids = nodes_by_layer[layer]
            layer_height = layer_heights[layer]
            y = margin_y + max(0, (canvas_height - margin_y * 2 - layer_height) / 2)
            for node_id in ids:
                size = node_sizes[node_id]
                layer_width = layer_widths[layer]
                positions[node_id] = (x + (layer_width - size.width) / 2, y)
                y += size.height + gap_y
            x += layer_widths[layer] + gap_x

    return LayeredLayout(positions=positions, width=canvas_width, height=canvas_height, layers=layers)


def route_layered_edges(
    model: DiagramModel,
    positions: dict[str, tuple[float, float]],
    node_sizes: dict[str, DiagramNodeSize],
) -> dict[str, EdgeRoute]:
    routes: dict[str, EdgeRoute] = {}
    sibling_offsets: dict[tuple[str, str], int] = defaultdict(int)
    direction = str(model.direction or "LR").upper()
    is_vertical = direction in {"TB", "TD", "BT"}
    for edge in model.edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        source_size = node_sizes[edge.source]
        target_size = node_sizes[edge.target]
        sx, sy = positions[edge.source]
        tx, ty = positions[edge.target]
        source_right = (sx + source_size.width, sy + source_size.height / 2)
        source_left = (sx, sy + source_size.height / 2)
        target_left = (tx, ty + target_size.height / 2)
        target_right = (tx + target_size.width, ty + target_size.height / 2)
        source_bottom = (sx + source_size.width / 2, sy + source_size.height)
        source_top = (sx + source_size.width / 2, sy)
        target_top = (tx + target_size.width / 2, ty)
        target_bottom = (tx + target_size.width / 2, ty + target_size.height)

        key = tuple(sorted((edge.source, edge.target)))
        offset_index = sibling_offsets[key]
        sibling_offsets[key] += 1
        offset = (offset_index % 3) * 18
        if offset_index % 2:
            offset *= -1

        if is_vertical:
            forward = ty >= sy
            start = source_bottom if forward else source_top
            end = target_top if forward else target_bottom
            mid_y = start[1] + (end[1] - start[1]) / 2
            points = (start, (start[0] + offset, mid_y), (end[0] + offset, mid_y), end)
            label_position = ((points[1][0] + points[2][0]) / 2, points[1][1] - 8)
        else:
            forward = tx >= sx
            start = source_right if forward else source_left
            end = target_left if forward else target_right
            mid_x = start[0] + (end[0] - start[0]) / 2
            points = (start, (mid_x, start[1] + offset), (mid_x, end[1] + offset), end)
            label_position = (points[1][0], (points[1][1] + points[2][1]) / 2 - 8)
        routes[edge.id] = EdgeRoute(edge_id=edge.id, points=points, label_position=label_position)
    return routes
