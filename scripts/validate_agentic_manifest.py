from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "Docs" / "reingenieria-plataforma-2026-07-15" / "10-agentic-execution-manifest.yaml"
OUTPUT_PATH = REPO_ROOT / "Docs" / "reingenieria-plataforma-2026-07-15" / "stage-0" / "manifest-validation.json"


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def topological_order(nodes: list[str], dependencies: dict[str, list[str]]) -> list[str]:
    indegree = {node: 0 for node in nodes}
    reverse_edges: dict[str, list[str]] = defaultdict(list)

    for node in nodes:
        for dependency in dependencies.get(node, []):
            ensure(dependency in indegree, f"Dependencia desconocida: {dependency} -> {node}")
            indegree[node] += 1
            reverse_edges[dependency].append(node)

    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    ordered: list[str] = []

    while queue:
        current = queue.popleft()
        ordered.append(current)
        for follower in sorted(reverse_edges.get(current, [])):
            indegree[follower] -= 1
            if indegree[follower] == 0:
                queue.append(follower)

    ensure(len(ordered) == len(nodes), "El grafo contiene al menos un ciclo.")
    return ordered


def validate_stage_work_packets(stage: dict[str, Any]) -> dict[str, Any]:
    work_packets = stage.get("work_packages", [])
    work_packet_ids = [item["id"] for item in work_packets]
    ensure(len(work_packet_ids) == len(set(work_packet_ids)), f"IDs duplicados en {stage['id']}")

    dependencies = {item["id"]: list(item.get("depends_on", [])) for item in work_packets}
    order = topological_order(work_packet_ids, dependencies)

    return {
        "stage_id": stage["id"],
        "work_packet_count": len(work_packet_ids),
        "work_packet_order": order,
    }


def main() -> int:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    ensure(isinstance(payload, dict), "El manifiesto no es un objeto YAML valido.")

    stages = payload.get("stages", [])
    ensure(isinstance(stages, list) and stages, "El manifiesto no declara etapas.")

    stage_ids = [stage["id"] for stage in stages]
    ensure(len(stage_ids) == len(set(stage_ids)), "Hay IDs de etapa duplicados.")

    stage_dependencies = {stage["id"]: list(stage.get("depends_on", [])) for stage in stages}
    stage_order = topological_order(stage_ids, stage_dependencies)

    stage_results = [validate_stage_work_packets(stage) for stage in stages]

    close_gate_coverage: dict[str, bool] = {}
    for stage in stages:
        has_close_gate = bool(stage.get("close_gate"))
        has_start_gate = bool(stage.get("start_gate"))
        close_gate_coverage[stage["id"]] = has_close_gate or has_start_gate
        ensure(close_gate_coverage[stage["id"]], f"La etapa {stage['id']} no declara gate de cierre o arranque.")

    validation_summary = {
        "schema_version": payload.get("schema_version", ""),
        "plan_id": payload.get("plan_id", ""),
        "validated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "stage_count": len(stage_ids),
        "work_packet_count": sum(item["work_packet_count"] for item in stage_results),
        "stage_order": stage_order,
        "close_gate_coverage": close_gate_coverage,
        "stage_results": stage_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(validation_summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(validation_summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
