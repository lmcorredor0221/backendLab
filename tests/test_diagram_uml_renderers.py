from __future__ import annotations

from pathlib import Path
import re

from app.services.diagram_center.contracts import DiagramEdge, DiagramModel, DiagramNode
from app.services.diagram_center.renderer_service import render_svg


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagram_layout"


def _load_model(name: str) -> DiagramModel:
    return DiagramModel.model_validate_json((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _viewbox_size(svg: str) -> tuple[int, int]:
    match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    assert match is not None
    return int(match.group(1)), int(match.group(2))


def test_dlg7_activity_renderer_uses_uml_activity_semantics() -> None:
    model = _load_model("dense_uml_activity.json")

    svg = render_svg(model)

    assert "UML ACTIVITY" in svg
    assert 'data-diagram-notation="uml_activity"' in svg
    assert "<polygon" in svg
    assert 'data-edge-kind="relationship"' in svg
    assert "uml-activity-arrow" in svg


def test_dlg7_activity_renderer_honors_top_bottom_direction() -> None:
    model = DiagramModel(
        diagram_key="activity_diagram",
        title="Actividad vertical",
        notation="uml_activity",
        direction="TB",
        nodes=[
            DiagramNode(id="start", label="Inicio", kind="start"),
            DiagramNode(id="question", label="Usuario pregunta", kind="activity"),
            DiagramNode(id="retrieve", label="Recuperar respuesta", kind="activity"),
            DiagramNode(id="evaluate", label="Respuesta suficiente", kind="decision"),
            DiagramNode(id="compose", label="Preparar respuesta", kind="activity"),
            DiagramNode(id="validate", label="Validar trazabilidad", kind="activity"),
            DiagramNode(id="respond", label="Responder al usuario", kind="activity"),
            DiagramNode(id="end", label="Fin", kind="end"),
        ],
        edges=[
            DiagramEdge(id="e1", source="start", target="question"),
            DiagramEdge(id="e2", source="question", target="retrieve"),
            DiagramEdge(id="e3", source="retrieve", target="evaluate"),
            DiagramEdge(id="e4", source="evaluate", target="compose", label="Si"),
            DiagramEdge(id="e5", source="compose", target="validate"),
            DiagramEdge(id="e6", source="validate", target="respond"),
            DiagramEdge(id="e7", source="respond", target="end"),
        ],
        source_refs=["test:activity-vertical"],
    )

    svg = render_svg(model)
    width, height = _viewbox_size(svg)

    assert height > width


def test_dlg7_use_case_renderer_keeps_actor_and_use_case_semantics() -> None:
    model = _load_model("dense_uml_use_case.json")

    svg = render_svg(model)

    assert 'data-node-kind="actor"' in svg
    assert 'data-node-kind="use_case"' in svg
    assert "UML USE CASE" in svg


def test_dlg7_sequence_renderer_uses_lifelines_and_message_arrows() -> None:
    model = DiagramModel(
        diagram_key="sequence_diagram",
        title="Consulta documental",
        notation="sequence",
        nodes=[
            DiagramNode(id="user", label="Usuario", kind="actor", source_refs=["discover:1"]),
            DiagramNode(id="assistant", label="Asistente Conversacional", kind="participant", source_refs=["design:1"]),
            DiagramNode(id="docs", label="Fuente de Documentacion", kind="external_system", source_refs=["tools:1"]),
        ],
        edges=[
            DiagramEdge(id="m1", source="user", target="assistant", label="enviar_pregunta", kind="sync_message", order=1),
            DiagramEdge(id="m2", source="assistant", target="docs", label="consultar_documentacion", kind="sync_message", order=2),
            DiagramEdge(id="m3", source="docs", target="assistant", label="respuesta_documentacion", kind="return_message", order=3),
        ],
        source_refs=["discover:1", "design:1", "tools:1"],
    )

    svg = render_svg(model)

    assert "UML SEQUENCE" in svg
    assert 'data-diagram-notation="sequence"' in svg
    assert 'data-sequence-kind="lifeline"' in svg
    assert 'data-sequence-kind="participant-head"' in svg
    assert 'data-sequence-kind="message"' in svg
    assert 'data-sequence-kind="activation"' in svg
    assert 'data-node-kind="actor"' in svg
    assert 'data-node-kind="participant"' in svg
    assert 'data-edge-kind="sync_message"' in svg
    assert 'data-edge-kind="return_message"' in svg
    assert "uml-sequence-return-arrow" in svg
