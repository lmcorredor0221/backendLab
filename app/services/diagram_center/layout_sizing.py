from __future__ import annotations

from dataclasses import dataclass

from app.services.diagram_center.contracts import DiagramNode, DiagramNotation


@dataclass(frozen=True)
class DiagramNodeSize:
    width: int
    height: int
    label_lines: tuple[str, ...]


def wrap_label(value: str, *, max_chars: int = 30, max_lines: int = 3) -> tuple[str, ...]:
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
    original = str(value or "").strip()
    if original and " ".join(lines) != original:
        lines[-1] = f"{lines[-1][: max_chars - 1]}..."
    return tuple(lines)


def _clamp(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def measure_generic_node(node: DiagramNode, notation: DiagramNotation) -> DiagramNodeSize:
    kind = str(node.kind or "").lower()
    if notation == DiagramNotation.uml_activity and ("decision" in kind or "gateway" in kind):
        lines = wrap_label(node.label, max_chars=28, max_lines=2)
        max_line = max((len(line) for line in lines), default=10)
        width = _clamp(210 + max(0, max_line - 18) * 8, minimum=220, maximum=360)
        height = _clamp(78 + (len(lines) - 1) * 18, minimum=86, maximum=124)
        return DiagramNodeSize(width=width, height=height, label_lines=lines)

    if notation == DiagramNotation.uml_activity and any(token in kind for token in ["start", "event", "final", "end"]):
        lines = wrap_label(node.label, max_chars=16, max_lines=2)
        diameter = _clamp(70 + (len(lines) - 1) * 14, minimum=70, maximum=92)
        return DiagramNodeSize(width=diameter, height=diameter, label_lines=lines)

    max_chars = 34 if notation in {DiagramNotation.flowchart, DiagramNotation.capability, DiagramNotation.c4} else 30
    lines = wrap_label(node.label, max_chars=max_chars, max_lines=3)
    max_line = max((len(line) for line in lines), default=10)
    width = _clamp(230 + max(0, max_line - 22) * 7, minimum=240, maximum=380)
    height = _clamp(70 + (len(lines) - 1) * 18, minimum=76, maximum=124)
    return DiagramNodeSize(width=width, height=height, label_lines=lines)
