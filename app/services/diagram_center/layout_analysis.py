from __future__ import annotations

from collections import defaultdict, deque

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.layout_contracts import DiagramLayoutMetrics, DiagramLayoutRisk


def _max_degree(model: DiagramModel) -> int:
    degrees: dict[str, int] = {node.id: 0 for node in model.nodes}
    for edge in model.edges:
        if edge.source in degrees:
            degrees[edge.source] += 1
        if edge.target in degrees:
            degrees[edge.target] += 1
    return max(degrees.values(), default=0)


def _has_cycle(model: DiagramModel) -> bool:
    graph: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {node.id: 0 for node in model.nodes}
    for edge in model.edges:
        if edge.source not in indegree or edge.target not in indegree:
            continue
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1

    queue = deque([node_id for node_id, degree in indegree.items() if degree == 0])
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for target in graph.get(node_id, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return bool(indegree) and visited != len(indegree)


def _risk_from_complexity(
    *,
    node_count: int,
    edge_count: int,
    edge_density: float,
    max_degree: int,
    label_avg_chars: float,
    lane_count: int,
    has_cycle: bool,
) -> DiagramLayoutRisk:
    score = 0
    if node_count >= 16:
        score += 3
    elif node_count >= 11:
        score += 2
    elif node_count >= 8:
        score += 1

    if edge_density >= 1.6:
        score += 3
    elif edge_density >= 1.25:
        score += 2
    elif edge_density >= 0.95:
        score += 1

    if max_degree >= 6:
        score += 2
    elif max_degree >= 4:
        score += 1

    if label_avg_chars >= 38:
        score += 2
    elif label_avg_chars >= 26:
        score += 1

    if has_cycle:
        score += 1

    if lane_count >= 3 and node_count >= 12:
        score += 2

    if score >= 8:
        return DiagramLayoutRisk.critical
    if score >= 6:
        return DiagramLayoutRisk.high
    if score >= 3:
        return DiagramLayoutRisk.medium
    return DiagramLayoutRisk.low


def analyze_diagram_complexity(model: DiagramModel) -> DiagramLayoutMetrics:
    node_count = len(model.nodes)
    edge_count = len(model.edges)
    edge_density = round(edge_count / max(node_count, 1), 3)
    label_avg_chars = round(
        sum(len(node.label or "") for node in model.nodes) / max(node_count, 1),
        2,
    )
    max_degree = _max_degree(model)
    has_cycle = _has_cycle(model)
    lane_count = sum(len(pool.lanes) for pool in model.pools)
    pool_count = len(model.pools)
    risk = _risk_from_complexity(
        node_count=node_count,
        edge_count=edge_count,
        edge_density=edge_density,
        max_degree=max_degree,
        label_avg_chars=label_avg_chars,
        lane_count=lane_count,
        has_cycle=has_cycle,
    )
    split_recommended = risk == DiagramLayoutRisk.critical and (node_count >= 18 or edge_density >= 1.7)

    return DiagramLayoutMetrics(
        node_count=node_count,
        edge_count=edge_count,
        edge_density=edge_density,
        max_degree=max_degree,
        label_avg_chars=label_avg_chars,
        lane_count=lane_count,
        pool_count=pool_count,
        estimated_crossing_risk=risk,
        split_recommended=split_recommended,
    )
