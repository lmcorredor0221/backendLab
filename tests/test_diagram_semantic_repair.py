from __future__ import annotations

from app.services.diagram_center.contracts import DiagramNotation, StructuredDiagramModel
from app.services.diagram_center.semantic_repair import (
    finalize_structured_diagram_artifact,
    repair_structured_diagram_model,
)


def _sample_bpmn_without_terminals() -> StructuredDiagramModel:
    return StructuredDiagramModel(
        diagram_key="current_process_map",
        title="Proceso actual",
        notation=DiagramNotation.bpmn,
        nodes=[
            {
                "id": "identify_context",
                "label": "Identificar contexto",
                "kind": "task",
                "metadata": {"pool_id": "pool_ops", "lane_id": "lane_analyst", "attributes": []},
                "source_refs": ["journey:discover:v1"],
            },
            {
                "id": "review_code",
                "label": "Revisar codigo fuente",
                "kind": "task",
                "metadata": {"pool_id": "pool_ops", "lane_id": "lane_analyst", "attributes": []},
                "source_refs": ["journey:discover:v1"],
            },
        ],
        edges=[
            {
                "id": "flow_1",
                "source": "identify_context",
                "target": "review_code",
                "kind": "sequence_flow",
                "source_refs": ["journey:discover:v1"],
            }
        ],
        pools=[
            {
                "id": "pool_ops",
                "label": "Operacion",
                "lanes": [
                    {
                        "id": "lane_analyst",
                        "label": "Analista",
                        "source_refs": ["journey:discover:v1"],
                    }
                ],
                "source_refs": ["journey:discover:v1"],
            }
        ],
        source_refs=["journey:discover:v1"],
    )


def _sample_bpmn_with_generic_inicio_fin() -> StructuredDiagramModel:
    return StructuredDiagramModel(
        diagram_key="current_process_map",
        title="Proceso actual",
        notation=DiagramNotation.bpmn,
        nodes=[
            {
                "id": "inicio",
                "label": "Inicio",
                "kind": "task",
                "metadata": {"pool_id": "pool_ops", "lane_id": "lane_analyst", "attributes": []},
                "source_refs": ["journey:discover:v1"],
            },
            {
                "id": "documentar_conocimiento",
                "label": "Documentar conocimiento",
                "kind": "task",
                "metadata": {"pool_id": "pool_ops", "lane_id": "lane_analyst", "attributes": []},
                "source_refs": ["journey:discover:v1"],
            },
            {
                "id": "fin",
                "label": "Fin",
                "kind": "task",
                "metadata": {"pool_id": "pool_ops", "lane_id": "lane_analyst", "attributes": []},
                "source_refs": ["journey:discover:v1"],
            },
        ],
        edges=[
            {
                "id": "flow_1",
                "source": "inicio",
                "target": "documentar_conocimiento",
                "kind": "sequence_flow",
                "source_refs": ["journey:discover:v1"],
            },
            {
                "id": "flow_2",
                "source": "documentar_conocimiento",
                "target": "fin",
                "kind": "sequence_flow",
                "source_refs": ["journey:discover:v1"],
            },
        ],
        pools=[
            {
                "id": "pool_ops",
                "label": "Operacion",
                "lanes": [
                    {
                        "id": "lane_analyst",
                        "label": "Analista",
                        "source_refs": ["journey:discover:v1"],
                    }
                ],
                "source_refs": ["journey:discover:v1"],
            }
        ],
        source_refs=["journey:discover:v1"],
    )


def test_repair_structured_bpmn_adds_missing_terminal_events() -> None:
    repaired, repairs = repair_structured_diagram_model(_sample_bpmn_without_terminals())

    assert len(repairs) == 2
    assert any(node.kind == "start_event" for node in repaired.nodes)
    assert any(node.kind == "end_event" for node in repaired.nodes)
    assert any(edge.kind == "sequence_flow" and edge.source.startswith("bpmn_start_event") for edge in repaired.edges)
    assert any(edge.kind == "sequence_flow" and edge.target.startswith("bpmn_end_event") for edge in repaired.edges)
    start_node = next(node for node in repaired.nodes if node.kind == "start_event")
    end_node = next(node for node in repaired.nodes if node.kind == "end_event")
    assert start_node.metadata.pool_id == "pool_ops"
    assert start_node.metadata.lane_id == "lane_analyst"
    assert end_node.metadata.pool_id == "pool_ops"
    assert end_node.metadata.lane_id == "lane_analyst"
    assert any("start event BPMN" in note for note in repaired.assumptions)
    assert any("end event BPMN" in note for note in repaired.assumptions)


def test_repair_structured_bpmn_promotes_existing_inicio_fin_nodes() -> None:
    repaired, repairs = repair_structured_diagram_model(_sample_bpmn_with_generic_inicio_fin())

    assert len(repairs) == 2
    assert len(repaired.nodes) == 3
    assert len([node for node in repaired.nodes if node.label == "Inicio"]) == 1
    assert len([node for node in repaired.nodes if node.label == "Fin"]) == 1
    start_node = next(node for node in repaired.nodes if node.label == "Inicio")
    end_node = next(node for node in repaired.nodes if node.label == "Fin")
    assert start_node.kind == "start_event"
    assert end_node.kind == "end_event"
    assert not any(edge.source.startswith("bpmn_start_event") for edge in repaired.edges)
    assert not any(edge.target.startswith("bpmn_end_event") for edge in repaired.edges)
    assert any("promovio un nodo existente a start_event" in note for note in repaired.assumptions)
    assert any("promovio un nodo existente a end_event" in note for note in repaired.assumptions)


def test_finalize_structured_diagram_artifact_marks_repaired_status() -> None:
    artifact, schema_status = finalize_structured_diagram_artifact(
        _sample_bpmn_without_terminals(),
        schema_status="valid",
    )

    assert isinstance(artifact, StructuredDiagramModel)
    assert schema_status == "repaired_bpmn_terminals"
