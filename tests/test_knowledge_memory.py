from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.core.config import get_settings
from app.models import (
    KnowledgeDocumentGovernancePatchRequest,
    KnowledgeDocumentRecord,
    KnowledgeManagedDocumentUpsertRequest,
    KnowledgeScope,
    KnowledgeSectionRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.knowledge_memory import KnowledgeMemoryService
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client
from tests.test_sessions_api import (
    assign_platform_role,
    auth_headers_for_credentials,
    create_workspace_for_user,
    seed_user,
)


def _auth_headers(client) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _write_doc(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_memory_session(*, enforce_foreign_keys: bool = False) -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    if enforce_foreign_keys:
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _workspace_headers(headers: dict[str, str], workspace_id: str) -> dict[str, str]:
    return {**headers, "x-workspace-id": workspace_id}


def test_docs_ingestion_is_scope_aware_and_reindexes_incrementally(tmp_path: Path) -> None:
    docs_root = tmp_path / "Docs"
    runtime_root = tmp_path / "runtime"
    platform_doc = docs_root / "product" / "agent-builder.md"
    _write_doc(
        platform_doc,
        "# Agent Builder\n\n## Objetivo\nCrear un agente con memoria gobernada.\n\n## Riesgos\nControlar approvals y side effects.\n",
    )

    service = KnowledgeMemoryService(docs_root=docs_root, runtime_root=runtime_root)
    with _build_memory_session() as session:
        first = service.sync_docs_corpus(session, force=True)
        first_sections = session.exec(
            select(KnowledgeSectionRecord).order_by(KnowledgeSectionRecord.sort_order)
        ).all()
        first_hash = first.corpus_hash
        first_lineages = [item.source_lineage for item in first_sections]

        second = service.sync_docs_corpus(session, force=False)
        second_sections = session.exec(
            select(KnowledgeSectionRecord).order_by(KnowledgeSectionRecord.sort_order)
        ).all()

        assert first_hash == second.corpus_hash
        assert second.changed_document_count == 0
        assert [item.section_key for item in first_sections] == [item.section_key for item in second_sections]
        assert [item.source_lineage for item in second_sections] == first_lineages
        assert runtime_root.joinpath("knowledge-corpus-manifest.json").exists()
        assert runtime_root.joinpath("lexical-index.json").exists()
        assert runtime_root.joinpath("vector-index.json").exists()

        _write_doc(
            docs_root / "product" / "runbook.md",
            "# Runbook\n\n## Recuperacion\nUsar checkpoints y evidencia versionada.\n",
        )
        added = service.sync_docs_corpus(session, force=False)
        assert added.changed_document_count == 1
        assert "Docs/product/runbook.md" in added.changed_paths

        _write_doc(
            platform_doc,
            "# Agent Builder\n\n## Objetivo\nCrear un agente con memoria gobernada.\n\n## Riesgos\nControlar approvals, side effects y rollback.\n",
        )
        (docs_root / "product" / "runbook.md").unlink()

        third = service.sync_docs_corpus(session, force=False)
        current_docs = session.exec(select(KnowledgeDocumentRecord)).all()
        third_sections = session.exec(
            select(KnowledgeSectionRecord).order_by(KnowledgeSectionRecord.relative_path, KnowledgeSectionRecord.sort_order)
        ).all()

        assert third.corpus_hash != first_hash
        assert third.changed_document_count == 2
        assert any(path.startswith("deleted::Docs/product/runbook.md") for path in third.changed_paths)
        assert len(current_docs) == 1
        assert current_docs[0].version_number == 2
        assert [item.source_lineage for item in third_sections] != first_lineages


def test_docs_ingestion_deletes_stale_sections_before_documents(tmp_path: Path) -> None:
    docs_root = tmp_path / "Docs"
    runtime_root = tmp_path / "runtime"
    _write_doc(docs_root / "old" / "first.md", "# First\n\n## One\nContenido inicial.\n")
    _write_doc(docs_root / "old" / "second.md", "# Second\n\n## Two\nContenido inicial.\n")

    service = KnowledgeMemoryService(docs_root=docs_root, runtime_root=runtime_root)
    with _build_memory_session(enforce_foreign_keys=True) as session:
        service.sync_docs_corpus(session, force=True)
        assert len(session.exec(select(KnowledgeDocumentRecord)).all()) == 2
        assert session.exec(select(KnowledgeSectionRecord)).all()

        (docs_root / "old" / "first.md").unlink()
        (docs_root / "old" / "second.md").unlink()

        report = service.sync_docs_corpus(session, force=False)

        assert report.document_count == 0
        assert len(session.exec(select(KnowledgeDocumentRecord)).all()) == 0
        assert len(session.exec(select(KnowledgeSectionRecord)).all()) == 0
        assert "deleted::Docs/old/first.md" in report.changed_paths
        assert "deleted::Docs/old/second.md" in report.changed_paths


def test_docs_ingestion_redacts_sensitive_content_before_indexing(tmp_path: Path) -> None:
    docs_root = tmp_path / "Docs"
    runtime_root = tmp_path / "runtime"
    secret = "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    _write_doc(
        docs_root / "security" / "secrets.md",
        (
            "# Secrets\n\n"
            f"api_key={secret}\n"
            "password = super-secret-password\n"
            "Authorization: Bearer this-should-not-leak\n"
        ),
    )

    service = KnowledgeMemoryService(docs_root=docs_root, runtime_root=runtime_root)
    parsed = service._parse_document(docs_root / "security" / "secrets.md")

    assert secret not in parsed.content_text
    assert "super-secret-password" not in parsed.content_text
    assert "this-should-not-leak" not in parsed.content_text

    with _build_memory_session() as session:
        report = service.sync_docs_corpus(session, force=True)
        stored_sections = session.exec(select(KnowledgeSectionRecord)).all()
        manifest = runtime_root.joinpath("knowledge-corpus-manifest.json").read_text(encoding="utf-8")

        assert report.document_count == 1
        assert stored_sections
        assert all(secret not in item.content_text for item in stored_sections)
        assert all("super-secret-password" not in item.content_text for item in stored_sections)
        assert "this-should-not-leak" not in manifest


def test_ensure_repo_docs_ingested_reuses_latest_run_when_runtime_artifacts_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "Docs"
    runtime_root = tmp_path / "runtime"
    _write_doc(
        docs_root / "ops" / "playbook.md",
        "# Playbook\n\n## Recovery\nReusar el corpus persistido si el runtime local es efimero.\n",
    )

    service = KnowledgeMemoryService(docs_root=docs_root, runtime_root=runtime_root)
    with _build_memory_session() as session:
        first = service.sync_docs_corpus(session, force=True)

        runtime_root.joinpath("knowledge-corpus-manifest.json").unlink()
        runtime_root.joinpath("lexical-index.json").unlink()
        runtime_root.joinpath("vector-index.json").unlink()

        def _unexpected_sync(*_args, **_kwargs):
            raise AssertionError("ensure_repo_docs_ingested no debe reconstruir el corpus cuando ya existe un run persistido.")

        monkeypatch.setattr(service, "sync_docs_corpus", _unexpected_sync)

        second = service.ensure_repo_docs_ingested(session)

        assert second.run_id == first.run_id
        assert second.corpus_hash == first.corpus_hash
        assert second.changed_document_count == 0
        assert second.document_count == first.document_count
        assert second.section_count == first.section_count
        assert not runtime_root.joinpath("knowledge-corpus-manifest.json").exists()
        assert not runtime_root.joinpath("lexical-index.json").exists()
        assert not runtime_root.joinpath("vector-index.json").exists()


def test_ensure_repo_docs_ingested_skips_repo_autosync_for_remote_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "Docs"
    runtime_root = tmp_path / "runtime"
    _write_doc(
        docs_root / "ops" / "playbook.md",
        "# Playbook\n\n## Recovery\nLa resincronizacion del repo se ejecuta de forma explicita fuera de produccion.\n",
    )

    settings = get_settings()
    original_database_url = settings.database_url
    original_autosync_override = settings.knowledge_repo_autosync_enabled
    settings.database_url = "postgresql+psycopg://user:secret@aws-0-us-east-2.pooler.supabase.com:5432/postgres"
    settings.knowledge_repo_autosync_enabled = None

    try:
        service = KnowledgeMemoryService(docs_root=docs_root, runtime_root=runtime_root)
        with _build_memory_session() as session:
            def _unexpected_sync(*_args, **_kwargs):
                raise AssertionError("El autosync del repo no debe ejecutarse cuando la base configurada es remota.")

            monkeypatch.setattr(service, "sync_docs_corpus", _unexpected_sync)
            report = service.ensure_repo_docs_ingested(session)

            assert report.status == "skipped"
            assert report.run_id is None
            assert report.document_count == 0
            assert report.changed_document_count == 0
    finally:
        settings.database_url = original_database_url
        settings.knowledge_repo_autosync_enabled = original_autosync_override


def test_governed_search_isolated_by_workspace_and_session_with_filters_and_cursor(tmp_path: Path) -> None:
    docs_root = tmp_path / "Docs"
    runtime_root = tmp_path / "runtime"
    _write_doc(
        docs_root / "governance" / "approval-flow.md",
        (
            "# Approval Flow\n\n"
            "## Design Policy\n"
            "Runtime design policy for agent orchestration and approval gates.\n"
        ),
    )

    service = KnowledgeMemoryService(docs_root=docs_root, runtime_root=runtime_root)
    workspace_a = uuid4()
    workspace_b = uuid4()
    session_a = uuid4()

    with _build_memory_session() as session:
        service.sync_docs_corpus(session, force=True)
        platform_doc = session.exec(select(KnowledgeDocumentRecord)).one()
        service.update_document_governance(
            session,
            document_id=platform_doc.id,
            payload=KnowledgeDocumentGovernancePatchRequest(
                authority_level="canonical",
                memory_usage="required_retrieval",
                stage_affinity=["design"],
            ),
            actor_user_id=None,
        )
        service.upsert_managed_document(
            session,
            payload=KnowledgeManagedDocumentUpsertRequest(
                scope=KnowledgeScope.workspace,
                relative_path="private/alpha-runtime.md",
                content_text="# Alpha Runtime\n\nAgent runtime for workspace alpha with design evidence.\n",
                authority_level="operational",
                memory_usage="candidate_retrieval",
                stage_affinity=["design"],
            ),
            workspace_id=workspace_a,
            actor_user_id=None,
        )
        service.upsert_managed_document(
            session,
            payload=KnowledgeManagedDocumentUpsertRequest(
                scope=KnowledgeScope.workspace,
                relative_path="private/alpha-checklist.md",
                content_text="# Alpha Checklist\n\nAgent runtime checklist for workspace alpha design operations.\n",
                authority_level="approved_artifact",
                memory_usage="candidate_retrieval",
                stage_affinity=["design"],
            ),
            workspace_id=workspace_a,
            actor_user_id=None,
        )
        service.upsert_managed_document(
            session,
            payload=KnowledgeManagedDocumentUpsertRequest(
                scope=KnowledgeScope.session,
                session_id=session_a,
                relative_path="private/session-only.md",
                content_text="# Session Only\n\nCalibration notes only for this session design thread.\n",
                authority_level="operational",
                memory_usage="candidate_retrieval",
                stage_affinity=["design"],
            ),
            workspace_id=workspace_a,
            actor_user_id=None,
        )
        service.upsert_managed_document(
            session,
            payload=KnowledgeManagedDocumentUpsertRequest(
                scope=KnowledgeScope.workspace,
                relative_path="private/beta-runtime.md",
                content_text="# Beta Runtime\n\nAgent runtime for workspace beta with isolated evidence.\n",
                authority_level="operational",
                memory_usage="candidate_retrieval",
                stage_affinity=["design"],
            ),
            workspace_id=workspace_b,
            actor_user_id=None,
        )

        alpha_result = service.search_governed(
            session,
            query="workspace alpha runtime design",
            role="planner",
            workspace_id=workspace_a,
            stage="design",
            limit=5,
        )
        beta_result = service.search_governed(
            session,
            query="workspace beta runtime design",
            role="planner",
            workspace_id=workspace_b,
            stage="design",
            limit=5,
        )
        session_hidden = service.search_governed(
            session,
            query="calibration notes session",
            role="planner",
            workspace_id=workspace_a,
            stage="design",
            limit=5,
        )
        session_visible = service.search_governed(
            session,
            query="calibration notes session",
            role="planner",
            workspace_id=workspace_a,
            session_id=session_a,
            stage="design",
            limit=5,
        )
        authority_only = service.search_governed(
            session,
            query="runtime design agent",
            role="planner",
            workspace_id=workspace_a,
            stage="design",
            authority_allowlist=["canonical"],
            limit=5,
        )
        paged = service.search_governed(
            session,
            query="runtime design agent",
            role="planner",
            workspace_id=workspace_a,
            stage="design",
            limit=1,
        )
        paged_next = service.search_governed(
            session,
            query="runtime design agent",
            role="planner",
            workspace_id=workspace_a,
            stage="design",
            limit=1,
            cursor=paged.next_cursor,
        )

        assert alpha_result.evidence_status == "grounded"
        assert all(not item.relative_path.startswith(f"Workspace/{workspace_b}") for item in alpha_result.items)
        assert any(item.relative_path.endswith("alpha-runtime.md") for item in alpha_result.items)
        assert beta_result.evidence_status == "grounded"
        assert all(not item.relative_path.startswith(f"Workspace/{workspace_a}") for item in beta_result.items)
        assert any(item.relative_path.endswith("beta-runtime.md") for item in beta_result.items)
        assert session_hidden.items == []
        assert session_hidden.absence_reason == "no_grounded_evidence_after_filters"
        assert any(item.relative_path.endswith("session-only.md") for item in session_visible.items)
        assert authority_only.items
        assert all(item.authority_level == "canonical" for item in authority_only.items)
        assert paged.next_cursor
        assert paged.items
        assert paged_next.items
        assert paged_next.items[0].section_key != paged.items[0].section_key


def test_governed_search_applies_precedence_and_excludes_expired_docs(tmp_path: Path) -> None:
    docs_root = tmp_path / "Docs"
    runtime_root = tmp_path / "runtime"
    _write_doc(
        docs_root / "governance" / "fallback-policy.md",
        (
            "# Fallback Policy\n\n"
            "## Design\n"
            "Fallback approval orchestration for build pipelines.\n"
        ),
    )
    service = KnowledgeMemoryService(docs_root=docs_root, runtime_root=runtime_root)
    workspace_id = uuid4()

    with _build_memory_session() as session:
        service.sync_docs_corpus(session, force=True)
        platform_doc = session.exec(select(KnowledgeDocumentRecord)).one()
        service.update_document_governance(
            session,
            document_id=platform_doc.id,
            payload=KnowledgeDocumentGovernancePatchRequest(
                authority_level="canonical",
                memory_usage="required_retrieval",
                stage_affinity=["design"],
            ),
            actor_user_id=None,
        )
        workspace_doc = service.upsert_managed_document(
            session,
            payload=KnowledgeManagedDocumentUpsertRequest(
                scope=KnowledgeScope.workspace,
                relative_path="private/fallback-policy.md",
                content_text="# Fallback Policy\n\nFallback approval orchestration for build pipelines.\n",
                authority_level="canonical",
                memory_usage="required_retrieval",
                stage_affinity=["design"],
            ),
            workspace_id=workspace_id,
            actor_user_id=None,
        )
        expired_doc = service.upsert_managed_document(
            session,
            payload=KnowledgeManagedDocumentUpsertRequest(
                scope=KnowledgeScope.workspace,
                relative_path="private/expired-runbook.md",
                content_text="# Expired Runbook\n\nExclusive expired manual runbook for legacy escalations.\n",
                authority_level="operational",
                memory_usage="candidate_retrieval",
                stage_affinity=["design"],
            ),
            workspace_id=workspace_id,
            actor_user_id=None,
        )
        service.update_document_governance(
            session,
            document_id=workspace_doc.id,
            payload=KnowledgeDocumentGovernancePatchRequest(
                authority_level="canonical",
                memory_usage="required_retrieval",
                stage_affinity=["design"],
            ),
            actor_user_id=None,
        )
        service.update_document_governance(
            session,
            document_id=expired_doc.id,
            payload=KnowledgeDocumentGovernancePatchRequest(
                expires_at=utc_now() - timedelta(days=1),
            ),
            actor_user_id=None,
        )

        precedence = service.search_governed(
            session,
            query="fallback approval orchestration",
            role="planner",
            workspace_id=workspace_id,
            stage="design",
            limit=5,
        )
        expired = service.search_governed(
            session,
            query="exclusive expired manual runbook",
            role="planner",
            workspace_id=workspace_id,
            stage="design",
            limit=5,
        )

        assert precedence.items
        assert precedence.items[0].scope == KnowledgeScope.platform
        assert expired.items == []
        assert expired.absence_reason == "no_grounded_evidence_after_filters"


def test_knowledge_routes_enforce_scope_permissions_and_workspace_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "Docs"
    _write_doc(
        docs_root / "system-analysis" / "memory-strategy.md",
        (
            "# Memory Strategy\n\n"
            "## Memory\n"
            "Usar checkpoints, resumentes y referencias versionadas.\n"
        ),
    )
    settings = get_settings()
    original_docs_root = settings.knowledge_docs_root
    settings.knowledge_docs_root = docs_root

    try:
        with build_test_client(monkeypatch) as client:
            headers = _auth_headers(client)
            member_email = "member@leanbuilder.local"
            member_password = "LeanBuilderMember123!"
            seed_user(
                client,
                email=member_email,
                password=member_password,
                full_name="Knowledge Member",
            )
            member_workspace_id = create_workspace_for_user(
                client,
                email=member_email,
                name="Knowledge Member Workspace",
                role=WorkspaceRole.admin,
            )
            member_headers = _workspace_headers(
                auth_headers_for_credentials(
                    client,
                    email=member_email,
                    password=member_password,
                ),
                member_workspace_id,
            )
            workspace_id = create_workspace_for_user(
                client,
                email=TEST_EMAIL,
                name="Knowledge Admin Workspace",
                role=WorkspaceRole.admin,
            )
            scoped_headers = _workspace_headers(headers, workspace_id)

            forbidden = client.post(
                "/api/v1/knowledge/docs/reingest",
                params={"scope": "platform", "force": "true"},
                headers=member_headers,
            )
            assert forbidden.status_code == 403

            assign_platform_role(client, email=TEST_EMAIL)

            reingest = client.post(
                "/api/v1/knowledge/docs/reingest",
                params={"scope": "platform", "force": "true"},
                headers=scoped_headers,
            )
            assert reingest.status_code == 200
            reingest_payload = reingest.json()
            assert reingest_payload["scope"] == "platform"
            assert reingest_payload["document_count"] == 1

            created = client.post(
                "/api/v1/knowledge/docs/entries",
                json={
                    "scope": "workspace",
                    "relative_path": "private/runtime-tools.md",
                    "content_text": "# Runtime Tools\n\nAnalisis operativo de herramientas minimas para memoria.\n",
                    "authority_level": "operational",
                    "memory_usage": "candidate_retrieval",
                    "stage_affinity": ["memory"],
                },
                headers=scoped_headers,
            )
            assert created.status_code == 200
            created_payload = created.json()
            assert created_payload["scope"] == "workspace"
            assert created_payload["relative_path"].endswith("private/runtime-tools.md")

            search = client.get(
                "/api/v1/knowledge/docs/search",
                params={
                    "q": "herramientas minimas memoria",
                    "role": "planner",
                    "stage": "memory",
                    "limit": "5",
                },
                headers=scoped_headers,
            )
            assert search.status_code == 200
            search_payload = search.json()
            assert search_payload["authorized_scopes"] == ["platform", "workspace"]
            assert any(item["relative_path"].endswith("private/runtime-tools.md") for item in search_payload["items"])

            expired = client.patch(
                f"/api/v1/knowledge/docs/entries/{created_payload['id']}",
                json={"expires_at": (utc_now() - timedelta(days=1)).isoformat()},
                headers=scoped_headers,
            )
            assert expired.status_code == 200
            assert expired.json()["status"] == "expired"

            after_expiry = client.get(
                "/api/v1/knowledge/docs/search",
                params={
                    "q": "herramientas minimas memoria",
                    "role": "planner",
                    "stage": "memory",
                    "limit": "5",
                },
                headers=scoped_headers,
            )
            assert after_expiry.status_code == 200
            after_expiry_payload = after_expiry.json()
            assert all(
                not item["relative_path"].endswith("private/runtime-tools.md")
                for item in after_expiry_payload["items"]
            )
    finally:
        settings.knowledge_docs_root = original_docs_root
