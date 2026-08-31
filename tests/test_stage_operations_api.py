from __future__ import annotations

from collections.abc import Generator
from datetime import timedelta
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.routes import sessions as sessions_routes
from app.db import get_session
from app.models import (
    JourneyStageArtifactEntry,
    JourneyStageArtifactRecord,
    SessionRecord,
    StageOperationRecord,
    StageOperationStatus,
    utc_now,
)
from app.services.product_processing.persistence import UncertaintyBacklogRecord
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


REAL_RUN_GENERATE_ESTIMATION_REPORT_OPERATION = sessions_routes._run_generate_estimation_report_operation


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setattr(sessions_routes, "_run_analyze_discovery_operation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sessions_routes, "_run_define_requirements_operation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sessions_routes, "_run_propose_design_operation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sessions_routes, "_run_recommend_tools_operation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sessions_routes, "_run_recommend_memory_operation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sessions_routes, "_run_generate_estimation_report_operation", lambda *_args, **_kwargs: None)
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def create_session(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post("/api/v1/sessions", headers=headers)
    assert response.status_code == 201
    return response.json()["id"]


def discovery_payload() -> dict[str, object]:
    return {
        "autonomy_level": "medium",
        "constraints": ["No publicar sin aprobacion"],
        "current_process": "El equipo responde tickets repetitivos manualmente.",
        "current_user": "Analista de soporte",
        "desired_outcome": "Reducir tiempos de respuesta con trazabilidad.",
        "mvp_definition": {
            "non_delegable_decisions": ["Aprobar excepciones"],
            "north_star_metric": "Tiempo promedio de resolucion",
            "out_of_scope": ["Ejecutar pagos"],
            "v1_scope": ["Clasificar tickets", "Proponer respuestas"],
        },
        "operational_baseline": {
            "automation_opportunities": ["Recuperar conocimiento"],
            "current_cost": "Impacto moderado en tiempo y calidad",
            "current_time_spent": "Entre 2 y 8 horas por semana",
            "frequent_errors": ["Respuestas inconsistentes"],
        },
        "problem_statement": "Soporte recibe solicitudes repetitivas.",
    }


def db_session_from_client(client: TestClient):
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    return session, session_generator


def test_stage_operation_start_is_idempotent_and_can_be_cancelled_then_retried(client: TestClient) -> None:
    headers = {**auth_headers(client), "x-idempotency-key": "design-once"}
    session_id = create_session(client, headers)

    first = client.post(
        f"/api/v1/sessions/{session_id}/propose-design/start",
        headers=headers,
        json={"instructions": "Explorar arquitectura gobernada."},
    )
    second = client.post(
        f"/api/v1/sessions/{session_id}/propose-design/start",
        headers=headers,
        json={"instructions": "Explorar arquitectura gobernada."},
    )

    assert first.status_code == 202
    assert second.status_code == 202
    first_payload = first.json()
    second_payload = second.json()
    assert second_payload["id"] == first_payload["id"]
    assert second_payload["attempt_count"] == 1
    assert second_payload["idempotency_key"] == "design-once"
    assert second_payload["status"] == "queued"
    assert second_payload["can_cancel"] is True

    current = client.get(
        f"/api/v1/sessions/{session_id}/stage-operations/current?stage_key=design&action=propose_design",
        headers=headers,
    )
    assert current.status_code == 200
    assert current.json()["id"] == first_payload["id"]

    cancel = client.post(
        f"/api/v1/sessions/{session_id}/stage-operations/{first_payload['id']}/cancel",
        headers=headers,
    )
    assert cancel.status_code == 200
    cancelled = cancel.json()
    assert cancelled["status"] == "cancelled"
    assert cancelled["can_cancel"] is False
    assert cancelled["can_retry"] is True
    assert cancelled["cancel_requested_at"]

    retry = client.post(
        f"/api/v1/sessions/{session_id}/stage-operations/{first_payload['id']}/retry",
        headers=headers,
    )
    assert retry.status_code == 202
    retried = retry.json()
    assert retried["id"] == first_payload["id"]
    assert retried["status"] == "queued"
    assert retried["attempt_count"] == 2
    assert retried["can_cancel"] is True
    assert retried["can_retry"] is False


def test_discover_and_define_stage_operation_starts_are_idempotent(client: TestClient) -> None:
    headers = {**auth_headers(client), "x-idempotency-key": "discover-once"}
    session_id = create_session(client, headers)

    discover_first = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery/start",
        headers=headers,
        json=discovery_payload(),
    )
    discover_second = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery/start",
        headers=headers,
        json=discovery_payload(),
    )

    assert discover_first.status_code == 202
    assert discover_second.status_code == 202
    discover_payload = discover_second.json()
    assert discover_payload["id"] == discover_first.json()["id"]
    assert discover_payload["stage_key"] == "discover"
    assert discover_payload["action"] == "analyze_discovery"
    assert discover_payload["idempotency_key"] == "discover-once"
    assert [step["key"] for step in discover_payload["steps"]] == [
        "queued",
        "normalize",
        "analysis",
        "questions",
        "persist",
    ]

    define_headers = {**auth_headers(client), "x-idempotency-key": "define-once"}
    define_first = client.post(
        f"/api/v1/sessions/{session_id}/define-requirements/start",
        headers=define_headers,
    )
    define_second = client.post(
        f"/api/v1/sessions/{session_id}/define-requirements/start",
        headers=define_headers,
    )

    assert define_first.status_code == 202
    assert define_second.status_code == 202
    define_payload = define_second.json()
    assert define_payload["id"] == define_first.json()["id"]
    assert define_payload["stage_key"] == "define"
    assert define_payload["action"] == "define_requirements"
    assert define_payload["idempotency_key"] == "define-once"
    assert [step["key"] for step in define_payload["steps"]] == [
        "queued",
        "canvas",
        "requirements",
        "questions",
        "persist",
    ]


def test_tools_memory_and_estimate_stage_operation_starts_are_idempotent(client: TestClient) -> None:
    headers = auth_headers(client)
    session_id = create_session(client, headers)

    tools_headers = {**headers, "x-idempotency-key": "tools-once"}
    tools_first = client.post(
        f"/api/v1/sessions/{session_id}/recommend-tools/start",
        headers=tools_headers,
        json={"instructions": "Solo herramientas minimas."},
    )
    tools_second = client.post(
        f"/api/v1/sessions/{session_id}/recommend-tools/start",
        headers=tools_headers,
        json={"instructions": "Solo herramientas minimas."},
    )

    assert tools_first.status_code == 202
    assert tools_second.status_code == 202
    tools_payload = tools_second.json()
    assert tools_payload["id"] == tools_first.json()["id"]
    assert tools_payload["stage_key"] == "tools"
    assert tools_payload["action"] == "recommend_tools"
    assert tools_payload["idempotency_key"] == "tools-once"
    assert [step["key"] for step in tools_payload["steps"]] == [
        "queued",
        "context",
        "recommendation",
        "questions",
        "persist",
    ]

    memory_headers = {**headers, "x-idempotency-key": "memory-once"}
    memory_first = client.post(
        f"/api/v1/sessions/{session_id}/recommend-memory/start",
        headers=memory_headers,
        json={"instructions": "RAG gobernado con fuentes aprobadas."},
    )
    memory_second = client.post(
        f"/api/v1/sessions/{session_id}/recommend-memory/start",
        headers=memory_headers,
        json={"instructions": "RAG gobernado con fuentes aprobadas."},
    )

    assert memory_first.status_code == 202
    assert memory_second.status_code == 202
    memory_payload = memory_second.json()
    assert memory_payload["id"] == memory_first.json()["id"]
    assert memory_payload["stage_key"] == "memory"
    assert memory_payload["action"] == "recommend_memory"
    assert memory_payload["idempotency_key"] == "memory-once"
    assert [step["key"] for step in memory_payload["steps"]] == [
        "queued",
        "context",
        "profile",
        "critique",
        "questions",
        "persist",
    ]

    estimate_headers = {**headers, "x-idempotency-key": "estimate-once"}
    estimate_first = client.post(
        f"/api/v1/sessions/{session_id}/estimate/start",
        headers=estimate_headers,
    )
    estimate_second = client.post(
        f"/api/v1/sessions/{session_id}/estimate/start",
        headers=estimate_headers,
    )

    assert estimate_first.status_code == 202
    assert estimate_second.status_code == 202
    estimate_payload = estimate_second.json()
    assert estimate_payload["id"] == estimate_first.json()["id"]
    assert estimate_payload["stage_key"] == "estimate"
    assert estimate_payload["action"] == "generate_estimation_report"
    assert estimate_payload["idempotency_key"] == "estimate-once"
    assert [step["key"] for step in estimate_payload["steps"]] == [
        "queued",
        "inputs",
        "analysis",
        "persist",
    ]


def test_generate_estimation_report_stage_operation_runner_executes_background_tasks(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = auth_headers(client)
    session_id = create_session(client, headers)

    db, session_generator = db_session_from_client(client)
    try:
        record = db.get(SessionRecord, UUID(session_id))
        assert record is not None
        now = utc_now()
        operation = StageOperationRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=record.user_id,
            stage_key="estimate",
            action="generate_estimation_report",
            idempotency_key="estimate-worker-test",
            attempt_count=1,
            status=StageOperationStatus.queued,
            current_step="queued",
            detail="Solicitud recibida.",
            request_payload={},
            steps=[],
            heartbeat_at=now,
            expires_at=now + timedelta(minutes=30),
        )
        db.add(operation)
        db.commit()
        db.refresh(operation)
        operation_id = operation.id
        bind = db.get_bind()
    finally:
        session_generator.close()

    captured: dict[str, object] = {}

    def fake_generate_estimation_report_route(session_id_arg, background_tasks, db, current_user):  # noqa: ANN001
        del db, current_user
        captured["session_id"] = str(session_id_arg)
        captured["background_tasks_type"] = type(background_tasks).__name__

        def mark_task() -> None:
            captured["background_task_executed"] = True

        background_tasks.add_task(mark_task)

    monkeypatch.setattr(sessions_routes, "generate_estimation_report_route", fake_generate_estimation_report_route)
    monkeypatch.setattr(
        sessions_routes,
        "load_latest_persisted_estimation_report",
        lambda db, session_id, current_blueprint_version_number: {
            "session_id": str(session_id),
            "blueprint_version": current_blueprint_version_number,
        },
    )
    monkeypatch.setattr(sessions_routes, "latest_blueprint_version_number", lambda db, session_id: 1)

    REAL_RUN_GENERATE_ESTIMATION_REPORT_OPERATION(operation_id, bind)

    db, session_generator = db_session_from_client(client)
    try:
        refreshed = db.get(StageOperationRecord, operation_id)
        assert refreshed is not None
        assert refreshed.status == StageOperationStatus.completed
        assert refreshed.current_step == "persist"
        assert refreshed.error_message == ""
    finally:
        session_generator.close()

    assert captured["session_id"] == session_id
    assert captured["background_tasks_type"] == "BackgroundTasks"
    assert captured["background_task_executed"] is True


def test_stage_operation_current_recovers_stale_active_operation_with_retry(client: TestClient) -> None:
    headers = auth_headers(client)
    session_id = create_session(client, headers)
    db, session_generator = db_session_from_client(client)
    try:
        record = db.get(SessionRecord, UUID(session_id))
        assert record is not None
        now = utc_now()
        operation = StageOperationRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=record.user_id,
            stage_key="design",
            action="propose_design",
            idempotency_key="stale-design",
            attempt_count=1,
            status=StageOperationStatus.running,
            current_step="proposal",
            detail="Generando propuesta.",
            request_payload={"instructions": "stale"},
            steps=[],
            heartbeat_at=now - timedelta(hours=1),
            expires_at=now - timedelta(minutes=1),
        )
        db.add(operation)
        db.commit()
        db.refresh(operation)
        operation_id = str(operation.id)
    finally:
        session_generator.close()

    current = client.get(
        f"/api/v1/sessions/{session_id}/stage-operations/current?stage_key=design&action=propose_design",
        headers=headers,
    )

    assert current.status_code == 200
    payload = current.json()
    assert payload["id"] == operation_id
    assert payload["status"] == "failed"
    assert payload["can_retry"] is True
    assert payload["can_cancel"] is False
    assert payload["technical_detail"] == "stage_operation_stale"
    assert payload["error_message"] == "Stage operation heartbeat expired before completion."


def test_stage_operation_with_missing_information_waits_for_user(client: TestClient) -> None:
    headers = auth_headers(client)
    session_id = create_session(client, headers)
    db, session_generator = db_session_from_client(client)
    try:
        record = db.get(SessionRecord, UUID(session_id))
        assert record is not None
        now = utc_now()
        operation = StageOperationRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=record.user_id,
            stage_key="discover",
            action="analyze_discovery",
            idempotency_key="discover-waiting",
            attempt_count=1,
            status=StageOperationStatus.running,
            current_step="analysis",
            detail="Analizando Discover.",
            request_payload=discovery_payload(),
            steps=[],
            heartbeat_at=now,
            expires_at=now + timedelta(minutes=30),
        )
        db.add(operation)
        db.commit()
        db.refresh(operation)

        artifact = JourneyStageArtifactEntry(
            id=operation.id,
            workspace_id=record.workspace_id,
            session_id=record.id,
            artifact_kind="discovery_analysis_artifact",
            stage_key="discover",
            source_action="analyze_discovery",
            missing_information=["Confirmar volumen mensual de tickets."],
            created_at=now,
            updated_at=now,
        )
        sessions_routes._complete_stage_operation_with_artifact(
            db,
            operation,
            artifact=artifact,
            completed_detail="Discover listo.",
            waiting_detail="Discover requiere una respuesta accionable.",
        )

        db.refresh(operation)
        assert operation.status == StageOperationStatus.waiting_for_user
        assert operation.current_step == "questions"
        assert operation.result_artifact_id == artifact.id
        assert "respuesta accionable" in operation.detail
        assert operation.expires_at is None
    finally:
        session_generator.close()


def test_waiting_memory_stage_operation_does_not_block_a_new_start_request(client: TestClient) -> None:
    headers = {**auth_headers(client), "x-idempotency-key": "memory-waiting"}
    session_id = create_session(client, headers)
    db, session_generator = db_session_from_client(client)
    try:
        record = db.get(SessionRecord, UUID(session_id))
        assert record is not None
        now = utc_now()
        operation = StageOperationRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=record.user_id,
            stage_key="memory",
            action="recommend_memory",
            idempotency_key="memory-waiting",
            attempt_count=1,
            status=StageOperationStatus.waiting_for_user,
            current_step="questions",
            detail="Memoria espera resolver una dependencia antes de continuar.",
            request_payload={"instructions": "RAG gobernado con fuentes aprobadas."},
            steps=[],
            heartbeat_at=now,
            expires_at=now + timedelta(minutes=30),
        )
        db.add(operation)
        db.commit()
        db.refresh(operation)
        waiting_operation_id = str(operation.id)
    finally:
        session_generator.close()

    response = client.post(
        f"/api/v1/sessions/{session_id}/recommend-memory/start",
        headers=headers,
        json={"instructions": "RAG gobernado con fuentes aprobadas."},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["id"] == waiting_operation_id
    assert payload["status"] == "queued"
    assert payload["action"] == "recommend_memory"
    assert payload["attempt_count"] == 2


def test_complete_stage_operation_does_not_pause_for_deferred_policy_backlog(client: TestClient) -> None:
    headers = auth_headers(client)
    session_id = create_session(client, headers)
    db, session_generator = db_session_from_client(client)
    try:
        record = db.get(SessionRecord, UUID(session_id))
        assert record is not None
        now = utc_now()
        operation = StageOperationRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=record.user_id,
            stage_key="memory",
            action="recommend_memory",
            idempotency_key="memory-deferred-backlog",
            attempt_count=1,
            status=StageOperationStatus.running,
            current_step="profile",
            detail="Generando Memoria.",
            request_payload={},
            steps=[],
            heartbeat_at=now,
            expires_at=now + timedelta(minutes=30),
        )
        db.add(operation)
        db.add(
            UncertaintyBacklogRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                uncertainty_key="memory:missing_information:1",
                product_mode="basic_free",
                source_stage="memory",
                target_stage="acp",
                kind="gap",
                disposition="defer",
                status="deferred",
                title="Confirmar fuentes historicas de memoria.",
                created_from="recommend_memory",
            )
        )
        db.commit()
        db.refresh(operation)

        artifact = JourneyStageArtifactEntry(
            id=operation.id,
            workspace_id=record.workspace_id,
            session_id=record.id,
            artifact_kind="memory_recommendation_artifact",
            stage_key="memory",
            source_action="recommend_memory",
            missing_information=["Confirmar fuentes historicas de memoria."],
            created_at=now,
            updated_at=now,
        )
        sessions_routes._complete_stage_operation_with_artifact(
            db,
            operation,
            artifact=artifact,
            completed_detail="Memoria generada con pendientes diferidos.",
            waiting_detail="Memoria genero preguntas accionables.",
        )

        db.refresh(operation)
        assert operation.status == StageOperationStatus.completed
        assert operation.current_step == "persist"
        assert "pendientes diferidos" in operation.detail
    finally:
        session_generator.close()


def test_current_stage_operation_completes_paused_operation_when_only_deferred_backlog_remains(
    client: TestClient,
) -> None:
    headers = auth_headers(client)
    session_id = create_session(client, headers)
    db, session_generator = db_session_from_client(client)
    try:
        record = db.get(SessionRecord, UUID(session_id))
        assert record is not None
        now = utc_now()
        artifact = JourneyStageArtifactRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            artifact_kind="memory_recommendation_artifact",
            stage_key="memory",
            source_action="recommend_memory",
            proposal_payload={"schema_version": "memory-recommendation.v1"},
            schema_version="memory-recommendation.v1",
            missing_information=["Confirmar fuentes historicas de memoria."],
            created_at=now,
            updated_at=now,
        )
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        operation = StageOperationRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=record.user_id,
            stage_key="memory",
            action="recommend_memory",
            idempotency_key="memory-paused-deferred",
            attempt_count=1,
            status=StageOperationStatus.waiting_for_user,
            current_step="questions",
            detail="Memoria genero preguntas accionables.",
            request_payload={},
            steps=[],
            result_artifact_id=artifact.id,
            heartbeat_at=now,
            expires_at=None,
        )
        db.add(operation)
        db.add(
            UncertaintyBacklogRecord(
                workspace_id=record.workspace_id,
                session_id=record.id,
                uncertainty_key="memory:missing_information:1",
                product_mode="basic_free",
                source_stage="memory",
                target_stage="acp",
                kind="gap",
                disposition="defer",
                status="deferred",
                title="Confirmar fuentes historicas de memoria.",
                created_from="recommend_memory",
            )
        )
        db.commit()
        operation_id = str(operation.id)
    finally:
        session_generator.close()

    current = client.get(
        f"/api/v1/sessions/{session_id}/stage-operations/current?stage_key=memory&action=recommend_memory",
        headers=headers,
    )

    assert current.status_code == 200
    payload = current.json()
    assert payload["id"] == operation_id
    assert payload["status"] == "completed"
    assert payload["current_step"] == "persist"
    assert payload["can_retry"] is False
    assert payload["can_cancel"] is False
