from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

import app.main as main_module
from app.db import get_session
from app.main import app
from app.models import UserRecord
from app.core.config import get_settings
from app.services.auth_service import hash_password

TEST_EMAIL = get_settings().local_admin_email
TEST_PASSWORD = get_settings().local_admin_password


@contextmanager
def build_test_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """
    Single seam for API tests so the suite can migrate away from TestClient
    with minimal churn when the runtime stack changes.
    """

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(
            UserRecord(
                email=TEST_EMAIL,
                full_name="Lean Builder Test",
                password_hash=hash_password(TEST_PASSWORD),
            )
        )
        session.commit()

    def override_get_session() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    runtime_dir = TemporaryDirectory(prefix="lean-builder-tests-")
    settings = get_settings()
    original_llm_config_path = settings.llm_config_path
    settings.llm_config_path = Path(runtime_dir.name) / "llm_settings.json"
    runtime_knowledge_root = settings.llm_config_path.parent / "knowledge-memory"
    runtime_knowledge_root.mkdir(parents=True, exist_ok=True)
    (runtime_knowledge_root / "knowledge-corpus-manifest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-20T00:00:00Z",
                "source_root": "Docs",
                "corpus_hash": "test-corpus-hash",
                "document_count": 1,
                "section_count": 1,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(main_module, "run_startup_tasks", lambda: None)
    app.dependency_overrides[get_session] = override_get_session

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        settings.llm_config_path = original_llm_config_path
        runtime_dir.cleanup()
        app.dependency_overrides.clear()
