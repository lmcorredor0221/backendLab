from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models import OpportunityRecord, SchemaMigrationRecord, SessionRecord, UserRecord
from app.services.auth_service import hash_password
from app.services.session_migration import MIGRATION_KEY_STAGE9, apply_session_contract_migration


def build_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


def seed_legacy_opportunity(session: Session) -> OpportunityRecord:
    user = UserRecord(
        email="migration@leanbuilder.local",
        full_name="Migration Test",
        password_hash=hash_password("LeanBuilder123!"),
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    session_record = SessionRecord(user_id=user.id)
    session.add(session_record)
    session.commit()
    session.refresh(session_record)

    opportunity = OpportunityRecord(
        session_id=session_record.id,
        problem_statement="Normalizar sesion historica",
        current_user="Operaciones",
        current_process="Usa nombres legacy",
        desired_outcome="Mantener compatibilidad sin romper exportes",
        autonomy_level="autonomous",
        constraints=["Sin side effects"],
        operational_baseline={
            "current_time_spent": "2h",
            "current_cost": "Retrabajo",
            "frequent_errors": ["Aliases legacy"],
            "automation_opportunities": ["Canonizar taxonomia"],
        },
        mvp_definition={
            "v1_scope": ["Normalizar discovery"],
            "out_of_scope": ["Eliminar contratos legacy"],
            "north_star_metric": "Sesion legible",
            "non_delegable_decisions": ["Aprobar retiro legacy"],
        },
        case_type="informational_assistant",
    )
    session.add(opportunity)
    session.commit()
    session.refresh(opportunity)
    return opportunity


def test_session_contract_migration_normalizes_legacy_values() -> None:
    engine = build_engine()
    with Session(engine) as session:
        seeded = seed_legacy_opportunity(session)
        summary = apply_session_contract_migration(session)

        migrated = session.exec(select(OpportunityRecord).where(OpportunityRecord.id == seeded.id)).one()
        migration_record = session.exec(
            select(SchemaMigrationRecord).where(SchemaMigrationRecord.migration_key == MIGRATION_KEY_STAGE9)
        ).one()

    assert summary.already_recorded is False
    assert summary.normalized_autonomy_rows == 1
    assert summary.normalized_case_type_rows == 1
    assert summary.touched_sessions == 1
    assert migrated.autonomy_level == "high"
    assert migrated.case_type == "informacion"
    assert migration_record.description


def test_session_contract_migration_is_idempotent() -> None:
    engine = build_engine()
    with Session(engine) as session:
        seed_legacy_opportunity(session)
        first = apply_session_contract_migration(session)
        second = apply_session_contract_migration(session)
        migration_records = session.exec(
            select(SchemaMigrationRecord).where(SchemaMigrationRecord.migration_key == MIGRATION_KEY_STAGE9)
        ).all()

    assert first.already_recorded is False
    assert first.normalized_autonomy_rows == 1
    assert first.normalized_case_type_rows == 1
    assert second.already_recorded is True
    assert second.normalized_autonomy_rows == 0
    assert second.normalized_case_type_rows == 0
    assert len(migration_records) == 1
