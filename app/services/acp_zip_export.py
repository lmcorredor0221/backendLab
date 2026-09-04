from __future__ import annotations

import json
from io import BytesIO
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from app.models import ACPPreview
from app.services.acp_serialization import normalize_text_document
from app.services.blueprint_zip_export import build_blueprint_files

if TYPE_CHECKING:
    from sqlmodel import Session
    from app.models import SessionSnapshot

ZIP_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _zip_info(filename: str) -> ZipInfo:
    info = ZipInfo(filename=filename, date_time=ZIP_FIXED_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    return info


def _build_acp_incremental_readme(preview: ACPPreview) -> str:
    return f"""# ACP Incremental Agentic Package (Blueprint Pro + ACP Portable)

Este paquete es un contenedor incremental completo que integra el diseño arquitectónico (**Blueprint Pro**) y el paquete ejecutable para agentes de IA (**ACP Portable**).

## 📁 Estructura del Paquete Incremental

### 1. `/Blueprint/` (Artefactos y Gobernanza de Diseño)
- **contracts/**: Especificación canónica `blueprint-core.v1.json`.
- **deliverables/**: Entregables detallados de arquitectura, herramientas, workflows y memoria.
- **diagrams/**: Modelos de diagramas, reportes de calidad y representaciones visuales.
- **governance/**: Decisiones delegadas, supuestos y matriz de riesgos.
- **index.html & assets/**: Visor interactivo standalone de la arquitectura Blueprint.

### 2. `/ACP/` (Paquete Portable de Construcción y Ejecución Agéntica)
- **runtime/**: Especificación de ejecución, prompts de sistema e instrucciones.
- **tools/**: Contratos OpenAPI / esquemas de herramientas y guardrails.
- **memory/**: Perfil de memoria contextual y patrones de razonamiento.
- **workflows/**: Grafos de ejecución y secuencias agénticas.
- **adapters/**: Adaptadores de integración y conectores de datos.
- **questions/**: Preguntas abiertas e incertidumbres de construcción.

---
*Paquete incremental generado por Lean Agent Builder v{preview.package_version}*
"""


def build_acp_zip(
    preview: ACPPreview,
    *,
    db: Session | None = None,
    snapshot: SessionSnapshot | None = None,
    overview_markdown: str = "",
) -> bytes:
    files: dict[str, bytes] = {}

    # 1. Incluir el paquete base de Blueprint Pro si snapshot y db están disponibles
    if db is not None and snapshot is not None:
        try:
            blueprint_files = build_blueprint_files(
                db,
                snapshot=snapshot,
                preview=preview,
                overview_markdown=overview_markdown,
            )
            files.update(blueprint_files)
        except Exception:
            # Fallback en caso de error parcial en blueprint: mantener ACP intacto
            pass

    # 2. Incluir todos los archivos del ACP Portable (dentro de ACP/...)
    for item in sorted(preview.files, key=lambda entry: entry.path):
        files[item.path] = normalize_text_document(item.content_text).encode("utf-8")

    # 3. Manifiesto y README raíz del paquete incremental
    generated_at = (
        snapshot.session.updated_at or snapshot.session.created_at
    ).isoformat() if snapshot and snapshot.session else "2026-09-04T00:00:00Z"

    manifest_payload = {
        "contract_version": "acp-incremental-package.v1",
        "session_id": str(snapshot.session.id) if snapshot and snapshot.session else "",
        "blueprint_version_number": preview.blueprint_version_number,
        "generated_at": generated_at,
        "contains": ["Blueprint", "ACP"],
        "file_count": len(files),
        "files": sorted(files),
    }
    files["manifest.json"] = json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    files["README.md"] = normalize_text_document(_build_acp_incremental_readme(preview)).encode("utf-8")

    # 4. Empaquetar todo en el ZIP
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path and not path.endswith("/"):
                archive.writestr(_zip_info(path), files[path])
    return buffer.getvalue()
