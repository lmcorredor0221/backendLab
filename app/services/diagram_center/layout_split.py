from __future__ import annotations

from collections import Counter, defaultdict

from app.services.diagram_center.contracts import DiagramModel
from app.services.diagram_center.layout_analysis import analyze_diagram_complexity
from app.services.diagram_center.layout_contracts import DiagramLayoutRisk, DiagramSplitRecommendation


def recommend_diagram_split(model: DiagramModel) -> DiagramSplitRecommendation:
    metrics = analyze_diagram_complexity(model)
    if metrics.estimated_crossing_risk not in {DiagramLayoutRisk.high, DiagramLayoutRisk.critical}:
        return DiagramSplitRecommendation(recommended=False)

    kind_counts = Counter(str(node.kind or "component").lower() for node in model.nodes)
    lane_counts: dict[str, int] = defaultdict(int)
    for node in model.nodes:
        lane_id = str(node.metadata.get("lane_id") or "").strip()
        if lane_id:
            lane_counts[lane_id] += 1

    suggested_chunks: list[str] = []
    if lane_counts:
        suggested_chunks.extend(f"lane:{lane_id}" for lane_id, count in lane_counts.items() if count >= 3)
    if kind_counts:
        suggested_chunks.extend(f"kind:{kind}" for kind, count in kind_counts.items() if count >= 3)
    if not suggested_chunks:
        suggested_chunks = ["overview", "details"]

    reason = (
        f"Riesgo visual {metrics.estimated_crossing_risk.value} con {metrics.node_count} nodos, "
        f"{metrics.edge_count} relaciones y densidad {metrics.edge_density}."
    )
    return DiagramSplitRecommendation(
        recommended=metrics.estimated_crossing_risk in {DiagramLayoutRisk.high, DiagramLayoutRisk.critical},
        reason=reason,
        suggested_chunks=list(dict.fromkeys(suggested_chunks))[:8],
    )
