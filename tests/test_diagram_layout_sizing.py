from __future__ import annotations

from app.services.diagram_center.contracts import DiagramNode, DiagramNotation
from app.services.diagram_center.layout_sizing import measure_generic_node, wrap_label


def test_dlg3_wrap_label_preserves_short_text() -> None:
    assert wrap_label("Clasificar intencion", max_chars=24, max_lines=2) == ("Clasificar intencion",)


def test_dlg3_wrap_label_truncates_with_ellipsis() -> None:
    lines = wrap_label(
        "Clasificar intencion y prioridad segun evidencia corporativa autorizada",
        max_chars=24,
        max_lines=2,
    )

    assert len(lines) == 2
    assert lines[-1].endswith("...")


def test_dlg3_measurement_expands_long_labels_without_exceeding_bounds() -> None:
    short_node = DiagramNode(id="a", label="Clasificar", kind="process")
    long_node = DiagramNode(
        id="b",
        label="Normalizacion y enriquecimiento de solicitud con evidencia corporativa autorizada",
        kind="process",
    )

    short_size = measure_generic_node(short_node, DiagramNotation.flowchart)
    long_size = measure_generic_node(long_node, DiagramNotation.flowchart)

    assert long_size.width >= short_size.width
    assert long_size.height > short_size.height
    assert long_size.width <= 380
    assert len(long_size.label_lines) >= 2


def test_dlg3_activity_decision_uses_diamond_compatible_size() -> None:
    node = DiagramNode(id="decision", label="Evidencia suficiente y consistente", kind="decision")

    size = measure_generic_node(node, DiagramNotation.uml_activity)

    assert size.width >= 220
    assert size.height >= 86
    assert len(size.label_lines) <= 2
