import re
import tempfile
import zipfile
from io import BytesIO
import pytest

from app.services.blueprint_zip_export import (
    _navigation_item,
    _validate_blueprint_package_integrity,
    _viewer_js,
    _viewer_css,
    _build_blueprint_viewer_html,
    _build_blueprint_readme,
)


def test_navigation_item_paths_and_labels():
    # Diagram SVG
    item_svg = _navigation_item(
        path="Blueprint/diagrams/system_architecture/diagram.svg",
        item_type="diagram",
    )
    assert item_svg["path"] == "Blueprint/diagrams/system_architecture/diagram.svg"
    assert item_svg["relative_path"] == "diagrams/system_architecture/diagram.svg"
    assert item_svg["stage"] == "design"
    assert "(SVG)" in item_svg["title"]

    # Deliverable markdown with explicit stage or fallback
    item_doc = _navigation_item(
        path="Blueprint/deliverables/brief_ejecutivo.md",
        item_type="artifact",
        stage="define",
    )
    assert item_doc["path"] == "Blueprint/deliverables/brief_ejecutivo.md"
    assert item_doc["relative_path"] == "deliverables/brief_ejecutivo.md"
    assert item_doc["stage"] == "define"

    # Core contract JSON
    item_contract = _navigation_item(
        path="Blueprint/contracts/blueprint-core.v1.json",
        item_type="artifact",
    )
    assert item_contract["path"] == "Blueprint/contracts/blueprint-core.v1.json"
    assert item_contract["relative_path"] == "contracts/blueprint-core.v1.json"
    assert item_contract["stage"] == "design"


def test_validate_blueprint_package_integrity_clean():
    files = {
        "Blueprint/index.html": b"""<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="assets/blueprint-viewer.css">
</head>
<body>
  <script src="assets/blueprint-viewer.js"></script>
  <noscript>
    <a href="deliverables/doc.md">Doc</a>
  </noscript>
</body>
</html>""",
        "Blueprint/README.md": b"""# Blueprint
Explora [index.html](index.html) (Blueprint/index.html).
Revisa [Doc](deliverables/doc.md).
""",
        "Blueprint/manifest.json": b"{}",
        "Blueprint/navigation-manifest.v1.json": b"{}",
        "Blueprint/assets/blueprint-viewer.css": b"/* css */",
        "Blueprint/assets/blueprint-viewer.js": b"/* js */",
        "Blueprint/deliverables/doc.md": b"# Doc Content",
    }
    manifest = {
        "items": [
            {
                "id": "deliverables:doc",
                "title": "Doc",
                "type": "artifact",
                "stage": "define",
                "path": "Blueprint/deliverables/doc.md",
                "relative_path": "deliverables/doc.md",
            }
        ],
        "storyline": [],
    }

    # Should pass without raising any exception
    _validate_blueprint_package_integrity(files, manifest)


def test_validate_blueprint_package_integrity_catches_broken_links():
    # 1. Missing required file
    incomplete_files = {
        "Blueprint/index.html": b"<html></html>",
        "Blueprint/manifest.json": b"{}",
    }
    with pytest.raises(ValueError, match="Archivo estructural obligatorio ausente o vacio"):
        _validate_blueprint_package_integrity(incomplete_files, {"items": []})

    # 2. Broken relative link in index.html
    broken_html_files = {
        "Blueprint/index.html": b'<a href="missing_file.md">Broken</a>',
        "Blueprint/README.md": b"[index](index.html)",
        "Blueprint/manifest.json": b"{}",
        "Blueprint/navigation-manifest.v1.json": b"{}",
        "Blueprint/assets/blueprint-viewer.css": b"",
        "Blueprint/assets/blueprint-viewer.js": b"",
    }
    with pytest.raises(ValueError, match="enlace roto 'missing_file.md'"):
        _validate_blueprint_package_integrity(broken_html_files, {"items": []})

    # 3. External absolute URL in index.html (violates offline portability)
    external_link_files = {
        "Blueprint/index.html": b'<script src="https://cdn.example.com/lib.js"></script>',
        "Blueprint/README.md": b"[index](index.html)",
        "Blueprint/manifest.json": b"{}",
        "Blueprint/navigation-manifest.v1.json": b"{}",
        "Blueprint/assets/blueprint-viewer.css": b"",
        "Blueprint/assets/blueprint-viewer.js": b"",
    }
    with pytest.raises(ValueError, match="enlace absoluto o externo no portable"):
        _validate_blueprint_package_integrity(external_link_files, {"items": []})

    # 4. Redundant 'Blueprint/' prefix in link target
    redundant_prefix_files = {
        "Blueprint/index.html": b'<a href="Blueprint/deliverables/doc.md">Doc</a>',
        "Blueprint/README.md": b"[index](index.html)",
        "Blueprint/manifest.json": b"{}",
        "Blueprint/navigation-manifest.v1.json": b"{}",
        "Blueprint/assets/blueprint-viewer.css": b"",
        "Blueprint/assets/blueprint-viewer.js": b"",
    }
    with pytest.raises(ValueError, match="prefijo redundante Blueprint/"):
        _validate_blueprint_package_integrity(redundant_prefix_files, {"items": []})


def test_e2e_extraction_and_local_relative_traversal():
    """Simula un paquete completo descomprimido en disco local (file:///)
    y verifica que todos los enlaces son estrictamente relativos y resuelven a archivos existentes."""
    manifest = {
        "contract_version": "blueprint-navigation-manifest.v1",
        "title": "Test Agent Pro",
        "blueprint_version_number": 1,
        "executive_summary": "Resumen ejecutivo de prueba",
        "items": [
            {
                "id": "diagrams:system_architecture:diagram.svg",
                "title": "Arquitectura del Sistema (SVG)",
                "type": "diagram",
                "stage": "design",
                "path": "Blueprint/diagrams/system_architecture/diagram.svg",
                "relative_path": "diagrams/system_architecture/diagram.svg",
            },
            {
                "id": "deliverables:brief_ejecutivo.md",
                "title": "Brief Ejecutivo",
                "type": "artifact",
                "stage": "define",
                "path": "Blueprint/deliverables/brief_ejecutivo.md",
                "relative_path": "deliverables/brief_ejecutivo.md",
            },
            {
                "id": "contracts:blueprint-core.v1.json",
                "title": "Blueprint Core (JSON)",
                "type": "artifact",
                "stage": "design",
                "path": "Blueprint/contracts/blueprint-core.v1.json",
                "relative_path": "contracts/blueprint-core.v1.json",
            },
        ],
        "storyline": [
            {
                "id": "define",
                "title": "Definicion",
                "stage": "define",
                "narrative": "Definicion del problema y objetivos.",
                "why_it_matters": "Claridad en el alcance.",
                "business_value": "Alineacion estrategica.",
                "key_takeaways": ["1 entregable"],
                "evidence_refs": ["Blueprint/deliverables/brief_ejecutivo.md"],
                "related_artifacts": ["deliverables:brief_ejecutivo.md"],
                "related_diagrams": [],
                "next_chapter_id": "design",
                "next_question": "Como se diseña la arquitectura?",
            },
            {
                "id": "design",
                "title": "Diseño",
                "stage": "design",
                "narrative": "Diseño arquitectonico y contratos.",
                "why_it_matters": "Estructura tecnica.",
                "business_value": "Viabilidad de ejecucion.",
                "key_takeaways": ["1 diagrama", "1 contrato"],
                "evidence_refs": ["Blueprint/diagrams/system_architecture/diagram.svg"],
                "related_artifacts": ["contracts:blueprint-core.v1.json"],
                "related_diagrams": ["diagrams:system_architecture:diagram.svg"],
                "next_chapter_id": "",
                "next_question": "",
            },
        ],
        "decisions": [
            {
                "id": "uncertainty:1",
                "title": "Uso de modelo local",
                "stage": "design",
                "status": "Asumido por LAB",
                "disposition": "assume",
                "impact_level": "alto",
                "recommendation": "Usar modelo local",
                "why_later": "Por definir infraestructura",
                "affected_items": ["contracts:blueprint-core.v1.json"],
                "resolution_moment": "Antes de produccion",
            }
        ],
    }

    files = {
        "Blueprint/index.html": _build_blueprint_viewer_html(manifest).encode("utf-8"),
        "Blueprint/README.md": _build_blueprint_readme(manifest, overview_markdown="## Resumen").encode("utf-8"),
        "Blueprint/manifest.json": b"{}",
        "Blueprint/navigation-manifest.v1.json": b"{}",
        "Blueprint/assets/blueprint-viewer.css": _viewer_css().encode("utf-8"),
        "Blueprint/assets/blueprint-viewer.js": _viewer_js().encode("utf-8"),
        "Blueprint/diagrams/system_architecture/diagram.svg": b"<svg></svg>",
        "Blueprint/deliverables/brief_ejecutivo.md": b"# Brief Ejecutivo",
        "Blueprint/contracts/blueprint-core.v1.json": b"{}",
        "Blueprint/governance/decisiones-delegadas-y-supuestos.md": b"# Decisiones",
    }

    # Validate package integrity
    _validate_blueprint_package_integrity(files, manifest)

    # Build zip buffer
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p, content in files.items():
            zf.writestr(p, content)

    # Extract to temp directory to simulate local user extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(BytesIO(buf.getvalue()), "r") as zf:
            zf.extractall(tmpdir)

        from pathlib import Path

        blueprint_root = Path(tmpdir) / "Blueprint"
        assert blueprint_root.is_dir()
        index_html_path = blueprint_root / "index.html"
        assert index_html_path.is_file()

        index_html_text = index_html_path.read_text(encoding="utf-8")

        # Verify no external protocols
        assert "http://" not in index_html_text
        assert "https://" not in index_html_text
        assert "//" not in [m.group(1) for m in re.finditer(r'href="([^"]+)"', index_html_text) if m.group(1).startswith("//")]

        # Verify all links inside index.html resolve to real files
        for match in re.finditer(r'(?:href|src)=["\']([^"\']+)["\']', index_html_text):
            target = match.group(1).split("#")[0].strip()
            if not target or target.startswith(("javascript:", "data:")):
                continue
            assert not target.startswith("Blueprint/"), f"Link in index.html must not start with 'Blueprint/': {target}"
            resolved = (blueprint_root / target).resolve()
            assert resolved.is_file(), f"Target file does not exist on disk: {target} (resolved: {resolved})"

        # Verify all markdown links in README.md resolve to real files
        readme_text = (blueprint_root / "README.md").read_text(encoding="utf-8")
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', readme_text):
            target = match.group(2).split("#")[0].strip()
            if not target or target.startswith(("http://", "https:", "mailto:")):
                continue
            assert not target.startswith("Blueprint/"), f"Link in README.md must not start with 'Blueprint/': {target}"
            resolved = (blueprint_root / target).resolve()
            assert resolved.is_file(), f"Target file in README does not exist on disk: {target} (resolved: {resolved})"

        # Verify that JavaScript viewer code constructs relative paths properly
        viewer_js_text = (blueprint_root / "assets" / "blueprint-viewer.js").read_text(encoding="utf-8")
        assert "item.relative_path" in viewer_js_text
        assert "replace(/^Blueprint\\//, '')" in viewer_js_text
