from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.services.llm_runtime.settings_migration import (
    inspect_runtime_settings_migration,
    migrate_runtime_settings_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normaliza backend/runtime/llm_settings.json al schema final de Codex Runtime Bridge.",
    )
    parser.add_argument(
        "--config",
        default=str(get_settings().llm_config_path),
        help="Ruta del llm_settings.json a inspeccionar o migrar.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Aplica la migracion y escribe el payload normalizado si hay cambios.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    result = (
        migrate_runtime_settings_file(config_path)
        if args.write
        else inspect_runtime_settings_migration(config_path)
    )
    payload = {
        "ok": True,
        "config_path": str(config_path),
        "changed": bool(result["changed"]),
        "written": bool(result.get("written", False)),
        "backup_path": result.get("backup_path"),
        "normalized_payload": result["written_payload"] if result.get("written") else result["normalized_payload"],
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
