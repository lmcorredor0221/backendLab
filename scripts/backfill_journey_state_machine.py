"""Backfill canonical journey state without making read routes write to the database."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

# Allow direct execution from the repository root without requiring installation.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db import engine
from app.models import JourneyStateRecord, SessionRecord
from app.services.product_processing.journey_state_machine_service import initialize_journey_state
from app.services.product_processing.product_journey_overview_service import build_product_journey_overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inicializa el estado canonico para sesiones legacy.")
    parser.add_argument("--session-id", type=UUID, help="Limita el backfill a un proyecto.")
    parser.add_argument("--limit", type=int, default=200, help="Maximo de sesiones por ejecucion.")
    parser.add_argument("--apply", action="store_true", help="Persiste el backfill. Sin esta opcion solo reporta.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Session(engine) as db:
        query = (
            select(SessionRecord)
            .outerjoin(JourneyStateRecord, JourneyStateRecord.session_id == SessionRecord.id)
            .where(JourneyStateRecord.id.is_(None), SessionRecord.deleted_at.is_(None))
            .order_by(SessionRecord.created_at.asc())
            .limit(max(1, args.limit))
        )
        if args.session_id is not None:
            query = query.where(SessionRecord.id == args.session_id)
        records = db.exec(query).all()

        initialized: list[str] = []
        skipped: list[dict[str, str]] = []
        for record in records:
            if record.workspace_id is None:
                skipped.append({"session_id": str(record.id), "reason": "La sesion no pertenece a un workspace."})
                continue
            overview = build_product_journey_overview(db, record=record, current_user=None)
            state_machine = overview.journey_state_machine
            if state_machine is None:
                skipped.append({"session_id": str(record.id), "reason": "No se pudo proyectar un estado legacy."})
                continue
            if args.apply:
                initialize_journey_state(
                    db,
                    record=record,
                    state_key=state_machine.current.state_key,
                    substate=state_machine.current.substate,
                    actor_type="migration",
                    reason="Backfill explicito desde la proyeccion legacy del journey.",
                    correlation_id=f"legacy-backfill:{record.id}",
                    progress_percent=state_machine.current.progress_percent,
                    blocking=state_machine.current.blocking,
                    metadata={"source": "backfill_journey_state_machine"},
                )
            initialized.append(str(record.id))

        if args.apply:
            db.commit()

    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry_run",
                "candidate_count": len(records),
                "initialized_session_ids": initialized,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
