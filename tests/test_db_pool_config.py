from types import SimpleNamespace

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
