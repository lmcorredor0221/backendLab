from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    AuthTokenRecord,
    UserLegalAcceptanceRecord,
    UserRegisterRequest,
    WorkspaceMembershipRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password, hash_token, register_user, verify_password


def test_password_hash_roundtrip() -> None:
    password_hash = hash_password("s3cret-pass")

    assert verify_password("s3cret-pass", password_hash) is True
    assert verify_password("wrong-pass", password_hash) is False


def test_token_hash_is_stable() -> None:
    token = "demo-token"

    assert hash_token(token) == hash_token(token)
    assert hash_token(token) != token


def test_register_user_creates_workspace_token_and_legal_acceptance_records() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        user, raw_token, expires_at = register_user(
            session,
            UserRegisterRequest(
                accept_data_treatment=True,
                accept_privacy=True,
                accept_terms=True,
                confirm_password="ValidPass1!",
                email="Founder@Example.COM",
                full_name="Jane Founder",
                password="ValidPass1!",
                workspace_name="Founder Workspace",
            ),
            ip_address="127.0.0.1",
            user_agent="pytest",
        )

        session.refresh(user)
        token_record = session.exec(
            select(AuthTokenRecord).where(AuthTokenRecord.token_hash == hash_token(raw_token))
        ).first()
        memberships = session.exec(
            select(WorkspaceMembershipRecord).where(WorkspaceMembershipRecord.user_id == user.id)
        ).all()
        acceptances = session.exec(
            select(UserLegalAcceptanceRecord).where(UserLegalAcceptanceRecord.user_id == user.id)
        ).all()

    assert user.email == "founder@example.com"
    assert user.default_workspace_id is not None
    assert token_record is not None
    assert token_record.user_id == user.id
    assert expires_at == token_record.expires_at
    assert len(memberships) == 1
    assert memberships[0].workspace_id == user.default_workspace_id
    assert memberships[0].role == WorkspaceRole.owner
    assert {item.document_type for item in acceptances} == {
        "data_treatment_policy",
        "privacy_policy",
        "terms_and_conditions",
    }
    assert all(item.accepted for item in acceptances)
    assert all(item.ip_address == "127.0.0.1" for item in acceptances)
