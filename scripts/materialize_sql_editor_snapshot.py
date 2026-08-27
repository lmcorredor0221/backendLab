from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "snapshot" in raw[0]:
        snapshot = raw[0]["snapshot"]
        if isinstance(snapshot, dict):
            return snapshot
    if isinstance(raw, dict) and "snapshot" in raw and isinstance(raw["snapshot"], dict):
        return raw["snapshot"]
    if isinstance(raw, dict):
        return raw
    raise SystemExit(f"Formato no soportado en {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materializa un snapshot JSON copiado desde Supabase SQL Editor.")
    parser.add_argument("--input", type=Path, required=True, help="Archivo JSON crudo copiado desde SQL Editor.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directorio destino para los archivos por tabla.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = _load_payload(input_path)
    counts: dict[str, int] = {}

    for table_name, rows in sorted(snapshot.items()):
        filename = f"{table_name}.json"
        if isinstance(rows, list):
            payload = rows
            counts[filename] = len(rows)
        elif rows is None:
            payload = []
            counts[filename] = 0
        else:
            payload = [rows]
            counts[filename] = 1
        _write_json(output_dir / filename, payload)

    session_rows = snapshot.get("sessions") if isinstance(snapshot.get("sessions"), list) else []
    workspace_rows = snapshot.get("workspaces") if isinstance(snapshot.get("workspaces"), list) else []
    manifest = {
        "project_id": str(session_rows[0].get("id")) if session_rows else "",
        "workspace_id": str(workspace_rows[0].get("id")) if workspace_rows else "",
        "source_payload": str(input_path),
        "snapshot_dir": str(output_dir),
        "counts": counts,
        "notes": [
            "Snapshot materializado desde un copy JSON del SQL Editor de Supabase.",
            "No incluye secretos cifrados de workspace_provider_secrets ni hotmart_integration_secrets.",
        ],
    }
    _write_json(output_dir / "snapshot_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
