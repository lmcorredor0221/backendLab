from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlmodel import Session

from app.core.config import get_settings
from app.db import create_db_and_tables, engine
from app.services.llm_runtime.settings_migration import (
    apply_runtime_llm_multitenant_migration,
    migrate_runtime_settings_file,
)

DEFAULT_REPORT_PATH = BACKEND_ROOT / "runtime" / "governance" / "rt7-runtime-llm-multitenant-report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el backfill multitenant del runtime LLM y genera un reporte de migracion RT7.",
    )
    parser.add_argument(
        "--config",
        default=str(get_settings().llm_config_path),
        help="Ruta del llm_settings.json legado a usar como insumo de migracion.",
    )
    parser.add_argument(
        "--workspace-mode",
        choices=["inherit_defaults", "seed_overrides"],
        default="inherit_defaults",
        help="Controla si los workspaces heredan defaults de plataforma o si se siembran overrides explicitos.",
    )
    parser.add_argument(
        "--normalize-file",
        action="store_true",
        help="Normaliza el llm_settings.json legado antes de ejecutar el backfill.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Ruta del reporte JSON generado por la migracion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    report_path = Path(args.report).resolve()

    normalization = None
    if args.normalize_file:
        normalization = migrate_runtime_settings_file(config_path)

    create_db_and_tables()

    with Session(engine) as session:
        summary = apply_runtime_llm_multitenant_migration(
            session,
            config_path=config_path,
            workspace_mode=args.workspace_mode,
            report_path=report_path,
        )

    payload = {
        "ok": True,
        "config_path": str(config_path),
        "report_path": str(report_path),
        "normalization": normalization,
        "summary": summary.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
