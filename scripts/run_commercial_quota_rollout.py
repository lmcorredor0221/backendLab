from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlmodel import Session

from app.db import create_db_and_tables, engine
from app.services.commercial_rollout_service import apply_commercial_quota_rollout


def main() -> int:
    create_db_and_tables()
    with Session(engine) as session:
        summary = apply_commercial_quota_rollout(session)
    print(json.dumps({"ok": True, "summary": summary.to_dict()}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
