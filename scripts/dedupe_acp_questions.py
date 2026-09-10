from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlmodel import Session, create_engine, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import ConstructionQuestionResponseRecord, utc_now  # noqa: E402
from app.services.product_processing.persistence import UncertaintyBacklogRecord  # noqa: E402
from app.services.question_identity import merge_unique_strings, question_dedupe_signature  # noqa: E402


CLOSED_BACKLOG_STATUSES = {"dismissed", "superseded"}
BACKLOG_RESOLUTION_PRIORITY = {
    "open": 0,
    "in_progress": 0,
    "deferred": 1,
    "dismissed": 2,
    "superseded": 2,
    "resolved": 3,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate ACP uncertainty backlog questions by semantic question text."
    )
    parser.add_argument("--session-id", required=True, help="Production project/session UUID.")
    parser.add_argument("--product-mode", default="", help="Optional product mode filter, for example acp_implementation.")
    parser.add_argument("--database-url", default="", help="Optional database URL. Defaults to DATABASE_URL/app settings.")
    parser.add_argument("--apply", action="store_true", help="Persist the dedupe. Without this flag the script is dry-run.")
    return parser.parse_args()


def _engine(database_url: str):
    if database_url.strip():
        os.environ["DATABASE_URL"] = database_url.strip()
        return create_engine(database_url.strip(), echo=False, pool_pre_ping=True)
    from app.db import engine

    return engine


def _backlog_signature(record: UncertaintyBacklogRecord) -> str:
    return question_dedupe_signature(record.description or record.title, fallback_key=record.uncertainty_key)


def _response_signature(record: ConstructionQuestionResponseRecord) -> str:
    return question_dedupe_signature(record.question_text, fallback_key=record.question_key)


def _backlog_rank(record: UncertaintyBacklogRecord) -> tuple[int, int, int, float, object]:
    answer = record.assumed_answer or record.suggested_answer
    impacted_count = len(record.affected_deliverable_keys or []) + len(record.dependency_keys or [])
    return (
        BACKLOG_RESOLUTION_PRIORITY.get(str(record.status or "open").strip().lower(), 0),
        1 if str(answer or "").strip() else 0,
        impacted_count,
        float(record.confidence or 0),
        record.updated_at or record.created_at,
    )


def _load_backlog_records(
    session: Session,
    *,
    session_id: UUID,
    product_mode: str,
) -> list[UncertaintyBacklogRecord]:
    statement = select(UncertaintyBacklogRecord).where(
        UncertaintyBacklogRecord.session_id == session_id,
        UncertaintyBacklogRecord.status.notin_(list(CLOSED_BACKLOG_STATUSES)),
    )
    if product_mode.strip():
        statement = statement.where(UncertaintyBacklogRecord.product_mode == product_mode.strip())
    return session.exec(statement.order_by(UncertaintyBacklogRecord.updated_at.desc())).all()


def _load_response_records(
    session: Session,
    *,
    session_id: UUID,
) -> list[ConstructionQuestionResponseRecord]:
    return session.exec(
        select(ConstructionQuestionResponseRecord)
        .where(ConstructionQuestionResponseRecord.session_id == session_id)
        .order_by(ConstructionQuestionResponseRecord.updated_at.desc())
    ).all()


def _group_by_signature(records: list[Any], signature_fn) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[signature_fn(record)].append(record)
    return dict(grouped)


def _backlog_group_summary(signature: str, records: list[UncertaintyBacklogRecord]) -> dict[str, Any]:
    winner = max(records, key=_backlog_rank)
    return {
        "signature": signature,
        "keep_id": str(winner.id),
        "supersede_ids": [str(record.id) for record in records if record.id != winner.id],
        "question": winner.description or winner.title or winner.uncertainty_key,
        "statuses": sorted({record.status for record in records}),
        "uncertainty_keys": [record.uncertainty_key for record in records],
        "affected_deliverable_keys": merge_unique_strings(
            key
            for record in records
            for key in record.affected_deliverable_keys or []
        ),
        "dependency_keys": merge_unique_strings(
            key
            for record in records
            for key in record.dependency_keys or []
        ),
    }


def _response_group_summary(signature: str, records: list[ConstructionQuestionResponseRecord]) -> dict[str, Any]:
    return {
        "signature": signature,
        "question": records[0].question_text or records[0].question_key,
        "record_ids": [str(record.id) for record in records],
        "question_keys": [record.question_key for record in records],
        "statuses": sorted({record.status for record in records}),
    }


def _apply_backlog_group(
    session: Session,
    *,
    signature: str,
    records: list[UncertaintyBacklogRecord],
) -> int:
    winner = max(records, key=_backlog_rank)
    duplicates = [record for record in records if record.id != winner.id]
    now = utc_now()
    payload = dict(winner.payload or {})
    payload["dedupe_signature"] = signature
    payload["merged_uncertainty_ids"] = [str(record.id) for record in duplicates]
    winner.source_refs = merge_unique_strings(
        source_ref
        for record in records
        for source_ref in record.source_refs or []
    )
    winner.affected_deliverable_keys = merge_unique_strings(
        key
        for record in records
        for key in record.affected_deliverable_keys or []
    )
    winner.dependency_keys = merge_unique_strings(
        key
        for record in records
        for key in record.dependency_keys or []
    )
    winner.payload = payload
    winner.updated_at = now
    session.add(winner)

    for record in duplicates:
        duplicate_payload = dict(record.payload or {})
        duplicate_payload["dedupe_signature"] = signature
        duplicate_payload["superseded_by"] = str(winner.id)
        duplicate_payload["superseded_reason"] = "duplicate_acp_question"
        record.status = "superseded"
        record.target_stage = "closed"
        record.payload = duplicate_payload
        record.superseded_at = now
        record.updated_at = now
        session.add(record)
    return len(duplicates)


def main() -> int:
    args = _parse_args()
    session_id = UUID(args.session_id)
    engine = _engine(args.database_url)
    with Session(engine) as session:
        backlog_records = _load_backlog_records(
            session,
            session_id=session_id,
            product_mode=args.product_mode,
        )
        backlog_groups = _group_by_signature(backlog_records, _backlog_signature)
        duplicate_backlog_groups = {
            signature: records
            for signature, records in backlog_groups.items()
            if len(records) > 1
        }
        response_records = _load_response_records(session, session_id=session_id)
        response_groups = _group_by_signature(response_records, _response_signature)
        duplicate_response_groups = {
            signature: records
            for signature, records in response_groups.items()
            if len(records) > 1
        }

        superseded_count = 0
        if args.apply:
            for signature, records in duplicate_backlog_groups.items():
                superseded_count += _apply_backlog_group(session, signature=signature, records=records)
            session.commit()

        report = {
            "ok": True,
            "mode": "apply" if args.apply else "dry_run",
            "session_id": str(session_id),
            "product_mode": args.product_mode.strip() or "all",
            "active_backlog_count_before": len(backlog_records),
            "duplicate_backlog_group_count": len(duplicate_backlog_groups),
            "backlog_superseded_count": superseded_count,
            "duplicate_response_group_count": len(duplicate_response_groups),
            "duplicate_backlog_groups": [
                _backlog_group_summary(signature, records)
                for signature, records in duplicate_backlog_groups.items()
            ],
            "duplicate_response_groups": [
                _response_group_summary(signature, records)
                for signature, records in duplicate_response_groups.items()
            ],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
