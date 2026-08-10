from __future__ import annotations

from dataclasses import asdict, dataclass, field

from sqlmodel import Session, select

from app.diagnostics import normalize_autonomy_level, normalize_case_type
from app.models import OpportunityRecord, SchemaMigrationRecord


MIGRATION_KEY_STAGE9 = "2026-07-16-stage9-session-compatibility"


@dataclass
class SessionMigrationSummary:
    migration_key: str = MIGRATION_KEY_STAGE9
    already_recorded: bool = False
    normalized_autonomy_rows: int = 0
    normalized_case_type_rows: int = 0
    touched_sessions: int = 0
    session_snapshot_contract_version: str = "session-snapshot.v1"
    notes: list[str] = field(
        default_factory=lambda: [
            "session-snapshot.v1 permanece como contrato interno legible durante S9.",
            "Los aliases legacy siguen permitidos mientras se mide su uso antes del retiro.",
            "El perfil ACP extended permanece soportado como compatibilidad controlada.",
        ]
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_session_contract_migration(session: Session) -> SessionMigrationSummary:
    summary = SessionMigrationSummary()
    touched_session_ids: set[str] = set()
    dirty = False

    opportunities = session.exec(select(OpportunityRecord)).all()
    for record in opportunities:
        record_changed = False
        normalized_autonomy = normalize_autonomy_level(record.autonomy_level)
        if record.autonomy_level != normalized_autonomy:
            record.autonomy_level = normalized_autonomy
            summary.normalized_autonomy_rows += 1
            touched_session_ids.add(str(record.session_id))
            record_changed = True

        normalized_case_type = normalize_case_type(record.case_type, default=record.case_type)
        if normalized_case_type and record.case_type != normalized_case_type:
            record.case_type = normalized_case_type
            summary.normalized_case_type_rows += 1
            touched_session_ids.add(str(record.session_id))
            record_changed = True

        if record_changed:
            session.add(record)
            dirty = True

    migration_record = session.exec(
        select(SchemaMigrationRecord).where(SchemaMigrationRecord.migration_key == MIGRATION_KEY_STAGE9)
    ).first()
    if migration_record is None:
        session.add(
            SchemaMigrationRecord(
                migration_key=MIGRATION_KEY_STAGE9,
                description="Normaliza oportunidades legacy, conserva session-snapshot.v1 y documenta compatibilidad S9.",
            )
        )
        dirty = True
    else:
        summary.already_recorded = True

    if dirty:
        session.commit()

    summary.touched_sessions = len(touched_session_ids)
    return summary
