from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import inspect
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from app.models import SessionRecord
import app.db as db_module


def test_build_engine_kwargs_keeps_local_databases_unpooled_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        db_module,
        "get_settings",
        lambda: SimpleNamespace(
            app_debug=False,
            database_url="postgresql+psycopg://lean_builder:lean_builder@127.0.0.1:5432/lab",
            database_pool_size=None,
            database_max_overflow=None,
            database_pool_timeout_seconds=30,
            database_pool_recycle_seconds=1800,
        ),
    )

    kwargs = db_module._build_engine_kwargs()

    assert kwargs == {
        "echo": False,
        "pool_pre_ping": True,
    }


def test_build_engine_kwargs_applies_explicit_pool_overrides_for_local_databases(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        db_module,
        "get_settings",
        lambda: SimpleNamespace(
            app_debug=False,
            database_url="postgresql+psycopg://lean_builder:lean_builder@127.0.0.1:5432/lab",
            database_pool_size=2,
            database_max_overflow=1,
            database_pool_timeout_seconds=15,
            database_pool_recycle_seconds=600,
        ),
    )

    kwargs = db_module._build_engine_kwargs()

    assert kwargs == {
        "echo": False,
        "pool_pre_ping": True,
        "pool_size": 2,
        "max_overflow": 1,
        "pool_timeout": 15,
        "pool_recycle": 600,
        "pool_use_lifo": True,
    }


def test_build_engine_kwargs_uses_interactive_pool_defaults_for_remote_databases(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        db_module,
        "get_settings",
        lambda: SimpleNamespace(
            app_debug=False,
            database_url="postgresql+psycopg://user:pass@aws-0-us-east-2.pooler.supabase.com:5432/postgres",
            database_pool_size=None,
            database_max_overflow=None,
            database_pool_timeout_seconds=45,
            database_pool_recycle_seconds=900,
        ),
    )

    kwargs = db_module._build_engine_kwargs()

    assert kwargs == {
        "echo": False,
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 5,
        "pool_timeout": 45,
        "pool_recycle": 900,
        "pool_use_lifo": True,
    }


def test_commit_without_expiring_preserves_loaded_state_and_session_policy() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        record = SessionRecord(user_id=uuid4(), title="Proyecto sin recarga implicita")
        session.add(record)

        assert session.expire_on_commit is True
        db_module.commit_without_expiring(session)

        state = inspect(record)
        assert not state.expired_attributes
        assert record.title == "Proyecto sin recarga implicita"
        assert session.expire_on_commit is True
