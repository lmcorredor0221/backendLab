from __future__ import annotations

from collections import Counter, deque

from app.services.diagram_center.contracts import DiagramModel, DiagramQualityReport


FORBIDDEN_CONTENT = ("api_key", "password", "secret=", "sk-proj-", "BEGIN PRIVATE KEY")


def evaluate_diagram_quality(model: DiagramModel) -> DiagramQualityReport:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    checks["has_nodes"] = bool(model.nodes)
    if not model.nodes:
        errors.append("El diagrama no contiene nodos.")

    checks["has_source_refs"] = bool(model.source_refs or any(node.source_refs for node in model.nodes))
    if not checks["has_source_refs"]:
        warnings.append("El diagrama no declara referencias de origen.")

    normalized_labels = [node.label.strip().lower() for node in model.nodes]
    duplicate_labels = [label for label, count in Counter(normalized_labels).items() if count > 1]
    checks["labels_are_distinct"] = not duplicate_labels
    if duplicate_labels:
        warnings.append("Hay etiquetas de nodo duplicadas que reducen la legibilidad.")

    serialized = model.model_dump_json().lower()
    leaked_tokens = [token for token in FORBIDDEN_CONTENT if token.lower() in serialized]
    checks["contains_no_secrets"] = not leaked_tokens
    if leaked_tokens:
        errors.append("El contenido contiene patrones que podrían corresponder a secretos.")

    node_ids = {node.id for node in model.nodes}
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in model.edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    visited: set[str] = set()
    if node_ids:
        queue: deque[str] = deque([next(iter(node_ids))])
        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            queue.extend(adjacency[node_id] - visited)
    checks["graph_is_connected"] = not node_ids or visited == node_ids
    if not checks["graph_is_connected"]:
        warnings.append("Existen nodos desconectados; valide si son intencionales.")

    checks["reasonable_density"] = len(model.nodes) <= 40 and len(model.edges) <= 80
    if not checks["reasonable_density"]:
        warnings.append("El diagrama supera la densidad recomendada y debería dividirse por nivel.")

    score = 100 - (30 * len(errors)) - (8 * len(warnings))
    return DiagramQualityReport(valid=not errors, score=max(0, score), errors=errors, warnings=warnings, checks=checks)

