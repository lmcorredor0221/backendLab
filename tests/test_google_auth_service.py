import pytest
from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    GoogleAuthRequest,
    UserAuthIdentityRecord,
    UserLegalAcceptanceRecord,
    UserRecord,
    UserRegisterRequest,
    WorkspaceMembershipRecord,
)
from app.services import google_auth_service
from app.services.auth_service import register_user


def build_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def google_claims(*, subject: str = "google-sub-1", email: str = "founder@example.com"):
    return {
        "subject": subject,
        "email": email,
        "full_name": "Jane Founder",
    }


def test_google_registration_requires_legal_acceptance_and_creates_internal_account(monkeypatch) -> None:
    engine = build_engine()
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(google_auth_service, "_verify_google_credential", lambda _: google_claims())

    with Session(engine) as session:
        pending = google_auth_service.authenticate_with_google(
            session,
            GoogleAuthRequest(credential="verified-google-token"),
        )
        assert pending.status == "registration_required"
        assert session.exec(select(UserRecord)).all() == []

        completed = google_auth_service.authenticate_with_google(
            session,
            GoogleAuthRequest(
                credential="verified-google-token",
                accept_terms=True,
                accept_privacy=True,
                accept_data_treatment=True,
                workspace_name="AI Contracts",
            ),
            ip_address="127.0.0.1",
            user_agent="pytest",
        )

        users = session.exec(select(UserRecord)).all()
        identities = session.exec(select(UserAuthIdentityRecord)).all()
        memberships = session.exec(select(WorkspaceMembershipRecord)).all()
        acceptances = session.exec(select(UserLegalAcceptanceRecord)).all()

    assert completed.status == "authenticated"
    assert completed.is_new_user is True
    assert completed.access_token
    assert len(users) == 1
    assert users[0].password_hash is None
    assert len(identities) == 1
    assert identities[0].provider_subject == "google-sub-1"
    assert len(memberships) == 1
    assert len(acceptances) == 3


def test_returning_google_identity_reuses_the_same_user(monkeypatch) -> None:
    engine = build_engine()
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(google_auth_service, "_verify_google_credential", lambda _: google_claims())

    with Session(engine) as session:
        first = google_auth_service.authenticate_with_google(
            session,
            GoogleAuthRequest(
                credential="verified-google-token",
                accept_terms=True,
                accept_privacy=True,
                accept_data_treatment=True,
            ),
        )
        second = google_auth_service.authenticate_with_google(
            session,
            GoogleAuthRequest(credential="verified-google-token"),
        )
        users = session.exec(select(UserRecord)).all()

    assert first.user is not None
    assert second.user is not None
    assert first.user.id == second.user.id
    assert second.is_new_user is False
    assert len(users) == 1


def test_existing_password_account_requires_password_before_google_link(monkeypatch) -> None:
    engine = build_engine()
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(google_auth_service, "_verify_google_credential", lambda _: google_claims())

    with Session(engine) as session:
        user, _, _ = register_user(
            session,
            UserRegisterRequest(
                full_name="Jane Founder",
                email="founder@example.com",
                password="ValidPass1!",
                confirm_password="ValidPass1!",
                accept_terms=True,
                accept_privacy=True,
                accept_data_treatment=True,
            ),
        )

        pending = google_auth_service.authenticate_with_google(
            session,
            GoogleAuthRequest(credential="verified-google-token"),
        )
        assert pending.status == "link_required"

        with pytest.raises(HTTPException) as wrong_password:
            google_auth_service.authenticate_with_google(
                session,
                GoogleAuthRequest(credential="verified-google-token", password="WrongPass1!"),
            )
        assert wrong_password.value.status_code == 401

        linked = google_auth_service.authenticate_with_google(
            session,
            GoogleAuthRequest(credential="verified-google-token", password="ValidPass1!"),
        )
        identity = session.exec(select(UserAuthIdentityRecord)).one()

    assert linked.status == "authenticated"
    assert linked.user is not None
    assert linked.user.id == user.id
    assert identity.user_id == user.id
