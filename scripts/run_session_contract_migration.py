from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlmodel import SQLModel, Session

from app.db import engine, ensure_runtime_schema
from app.services.auth_service import seed_default_user
from app.services.session_migration import apply_session_contract_migration
from app.services.workspace_bootstrap import apply_workspace_bootstrap


DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "runtime" / "stage9-migration"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    SQLModel.metadata.create_all(engine)
    ensure_runtime_schema()
    seed_default_user()

    with Session(engine) as session:
        apply_workspace_bootstrap(session)
        summary = apply_session_contract_migration(session)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        **summary.to_dict(),
    }

    run_dir = Path(args.output_root) / now_slug()
    write_json(run_dir / "summary.json", payload)
    write_json(Path(args.output_root) / "latest" / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
