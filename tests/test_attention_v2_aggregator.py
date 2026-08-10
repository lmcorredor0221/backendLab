from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.db import get_session
from app.models import (
    ApprovalGateRecord,
    ApprovalStatus,
    AttentionActionRequestV2,
    CommercialAccessRequestRecord,
    CommercialAccessRequestStatus,
    CommercialAccessSnapshotV2,
    CommercialEventRecord,
    CommercialTier,
    ConstructionGapEntry,
    ConstructionQuestionEntry,
    ConstructionQuestionResponseRecord,
    ConstructionReadinessReport,
    JourneyArtifactState,
    JourneyStageArtifactEntry,
    JourneyStageArtifactRecord,
    SessionCreateResponse,
    SessionRecord,
    SessionSnapshot,
    SessionStage,
    ArtifactStatus,
    UserRecord,
    WorkspaceRecord,
    WorkspaceMembershipRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.attention.adapters import (
    items_from_approval_gates,
    items_from_commercial_access,
    items_from_governance_policies,
    items_from_handoffs,
    items_from_runtime_operation,
    items_from_stage_artifact_state,
    items_from_stage_payload,
)
from app.services.attention_service import apply_attention_action_v2, build_attention_metrics_v2, build_attention_response_v2
from app.services.auth_service import hash_password
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _db_session_from_client(client: TestClient) -> Session:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    return next(session_generator)


def _create_memory_engine_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _seed_minimal_records(session: Session) -> tuple[UserRecord, WorkspaceRecord, SessionRecord]:
    user = UserRecord(email="uxa2@leanbuilder.local", full_name="UXA2 Tester", password_hash=hash_password("Secret123!"))
    session.add(user)
    session.flush()
    workspace = WorkspaceRecord(name="UXA2 Workspace", slug=f"uxa2-{str(user.id)[:8]}", created_by_user_id=user.id)
    session.add(workspace)
    session.flush()
    session.add(WorkspaceMembershipRecord(workspace_id=workspace.id, user_id=user.id, role=WorkspaceRole.owner))
    record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="UXA2 Project")
    session.add(record)
    session.commit()
    return user, workspace, record


def _snapshot(record: SessionRecord) -> SessionSnapshot:
    now = utc_now()
    return SessionSnapshot(
        session=SessionCreateResponse(
            id=record.id,
            workspace_id=record.workspace_id,
            title=record.title,
            status=record.status,
            current_stage=record.current_stage,
            commercial_tier=record.commercial_tier,
            created_at=now,
            updated_at=now,
        )
    )


def test_discovery_attention_filters_questions_that_belong_to_future_stages() -> None:
    session = _create_memory_engine_session()
    try:
        _, workspace, record = _seed_minimal_records(session)
        now = utc_now()
        artifact_id = uuid4()
        snapshot = _snapshot(record)
        snapshot.journey_latest_artifacts = {
            "discover": JourneyStageArtifactEntry(
                id=artifact_id,
                workspace_id=workspace.id,
                session_id=record.id,
                artifact_kind="discovery_analysis_artifact",
                stage_key="discover",
                version_number=1,
                proposal_payload={
                    "open_questions": [
                        {
                            "key": "erp_integration",
                            "question": "Que ERP especifico y metodo de integracion debe usar el agente?",
                            "blocking_stages": ["tools"],
                        },
                        {
                            "key": "problem_owner",
                            "question": "Quien valida que el proceso actual representa el problema real?",
                            "blocking_stages": ["define"],
                        },
                    ]
                },
                missing_information=[
                    "Herramienta de ticketing o bandeja humana para handoff.",
                    "operational_baseline.current_cost",
                ],
                created_at=now,
                updated_at=now,
            )
        }

        response = build_attention_response_v2(
            session,
            record=record,
            snapshot=snapshot,
            readiness=ConstructionReadinessReport(),
            access=CommercialAccessSnapshotV2(
                workspace_id=record.workspace_id,
                session_id=record.id,
                user_id=record.user_id,
                tier=CommercialTier.blueprint,
            ),
        )
        titles = [item.title for item in response.items]

        assert "Que ERP especifico y metodo de integracion debe usar el agente?" not in titles
        assert "Falta informacion: Herramienta de ticketing o bandeja humana para handoff." not in titles
        assert "Quien valida que el proceso actual representa el problema real?" in titles
        assert "Falta informacion: operational_baseline.current_cost" in titles
    finally:
        session.close()


def test_stage_payload_maps_guided_questions_to_attention_options() -> None:
    items = items_from_stage_payload(
        product="blueprint",
        stage="define",
        source="definition",
        artifact_id="definition",
        artifact_version=4,
        href="/projects/demo/define",
        return_href="/projects/demo/attention",
        open_questions=[
            {
                "key": "owner_policy",
                "question": "Quien aprueba excepciones de politica?",
                "rationale": "Define necesita ownership funcional antes de diseno.",
                "priority": "high",
                "suggested_answer": "Lider de soporte",
                "answer_options": [
                    {
                        "key": "support_lead",
                        "label": "Lider de soporte",
                        "description": "Owner natural del proceso actual.",
                        "impact": "Cierra ownership funcional sin esperar implementacion.",
                        "example": "Lider de soporte valida excepciones semanalmente.",
                        "recommended": True,
                        "confidence": 0.84,
                        "source_refs": ["define.business_rules"],
                    }
                ],
            }
        ],
    )

    assert len(items) == 1
    assert items[0].suggested_answer == "Lider de soporte"
    assert items[0].options[0].key == "support_lead"
    assert items[0].options[0].recommended is True
    assert items[0].source_ref.entity_id == "owner_policy"


def test_attention_keeps_define_question_when_impacted_sections_are_artifact_sections() -> None:
    session = _create_memory_engine_session()
    try:
        user, workspace, record = _seed_minimal_records(session)
        now = utc_now()
        snapshot = _snapshot(record)
        snapshot.journey_latest_artifacts = {
            "define": JourneyStageArtifactEntry(
                id=uuid4(),
                workspace_id=workspace.id,
                session_id=record.id,
                artifact_kind="definition",
                stage_key="define",
                version_number=4,
                proposal_payload={
                    "open_questions": [
                        {
                            "key": "question:error-scenarios",
                            "question": "Que errores o excepciones criticas debe contemplar el flujo objetivo?",
                            "rationale": "Las excepciones reales deben alimentar Define antes de pasar a Design.",
                            "impacted_sections": ["functional_requirements", "business_rules"],
                            "blocking": False,
                            "answer_options": [
                                {
                                    "key": "human_escalation",
                                    "label": "Escalar excepciones a humano",
                                    "recommended": True,
                                    "confidence": 0.78,
                                }
                            ],
                        }
                    ]
                },
                created_at=now,
                updated_at=now,
            )
        }
        access = CommercialAccessSnapshotV2(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=user.id,
            tier=CommercialTier.blueprint,
        )

        response = build_attention_response_v2(
            session,
            record=record,
            snapshot=snapshot,
            readiness=ConstructionReadinessReport(),
            access=access,
            current_stage="define",
        )

        item = next(
            entry
            for entry in response.items
            if entry.source_ref.entity_id == "question:error-scenarios"
        )
        assert item.stage == "define"
        assert item.type == "question"
        assert item.title == "Que errores o excepciones criticas debe contemplar el flujo objetivo?"
    finally:
        session.close()


def test_attention_does_not_surface_payload_items_from_stale_artifacts() -> None:
    session = _create_memory_engine_session()
    try:
        user, workspace, record = _seed_minimal_records(session)
        now = utc_now()
        snapshot = _snapshot(record)
        snapshot.journey_latest_artifacts = {
            "design": JourneyStageArtifactEntry(
                id=uuid4(),
                workspace_id=workspace.id,
                session_id=record.id,
                artifact_kind="design_recommendation_artifact",
                stage_key="design",
                version_number=3,
                state=JourneyArtifactState.stale,
                stale_reasons=["upstream_define_artifact_v4_regenerated"],
                proposal_payload={
                    "open_questions": [
                        "Confirmar arquitectura objetivo.",
                    ],
                    "critic_findings": [
                        {
                            "title": "Falta cubrir seguridad operacional",
                            "severity": "blocking",
                            "suggested_action": "Agregar guardrails antes de aprobar.",
                        }
                    ],
                    "warnings": ["Warning vieja que ya no debe heredarse."],
                },
                created_at=now,
                updated_at=now,
            )
        }
        access = CommercialAccessSnapshotV2(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=user.id,
            tier=CommercialTier.blueprint,
        )

        response = build_attention_response_v2(
            session,
            record=record,
            snapshot=snapshot,
            readiness=ConstructionReadinessReport(),
            access=access,
            current_stage="design",
        )
        titles = [item.title for item in response.items]

        assert "design_recommendation_artifact requiere revision" in titles
        assert "Confirmar arquitectura objetivo." not in titles
        assert "Falta cubrir seguridad operacional" not in titles
        assert "Warning vieja que ya no debe heredarse." not in titles
    finally:
        session.close()


def test_generic_attention_answer_is_traced_and_hidden_after_refresh() -> None:
    session = _create_memory_engine_session()
    try:
        user, workspace, record = _seed_minimal_records(session)
        now = utc_now()
        snapshot = _snapshot(record)
        snapshot.journey_latest_artifacts = {
            "define": JourneyStageArtifactEntry(
                id=uuid4(),
                workspace_id=workspace.id,
                session_id=record.id,
                artifact_kind="definition",
                stage_key="define",
                version_number=2,
                proposal_payload={
                    "open_questions": [
                        {
                            "key": "owner_policy",
                            "question": "Quien aprueba excepciones de politica?",
                            "rationale": "Define necesita ownership funcional antes de diseno.",
                            "suggested_answer": "Lider de soporte",
                            "answer_options": [
                                {
                                    "key": "support_lead",
                                    "label": "Lider de soporte",
                                    "recommended": True,
                                    "confidence": 0.84,
                                }
                            ],
                        }
                    ]
                },
                created_at=now,
                updated_at=now,
            )
        }
        access = CommercialAccessSnapshotV2(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=user.id,
            tier=CommercialTier.blueprint,
        )
        before = build_attention_response_v2(session, record=record, snapshot=snapshot, readiness=ConstructionReadinessReport(), access=access)
        item = next(entry for entry in before.items if entry.source.startswith("journey."))

        result = apply_attention_action_v2(
            session,
            record=record,
            snapshot=snapshot,
            readiness=ConstructionReadinessReport(),
            access=access,
            current_user=user,
            item_key=item.key,
            payload=AttentionActionRequestV2(
                action_kind="answer",
                answer_text="Lider de soporte",
                selected_option_key="support_lead",
                was_suggested_answer_used=True,
                idempotency_key="answer-owner-policy",
                source_artifact_version=2,
            ),
        )
        session.commit()
        after = build_attention_response_v2(session, record=record, snapshot=snapshot, readiness=ConstructionReadinessReport(), access=access)
        metrics = build_attention_metrics_v2(
            session,
            record=record,
            snapshot=snapshot,
            readiness=ConstructionReadinessReport(),
            access=access,
            current_stage="define",
        )
        event = session.exec(select(CommercialEventRecord).where(CommercialEventRecord.event_key == "attention_action_v2")).first()

        assert result.status == "applied"
        assert item.key not in {entry.key for entry in after.items}
        assert metrics["answered_questions"] == 1
        assert metrics["suggested_answer_acceptances"] == 1
        assert metrics["selected_option_answers"] == 1
        assert event is not None
        assert event.metadata_payload["selected_option_key"] == "support_lead"
        assert event.metadata_payload["was_suggested_answer_used"] is True
    finally:
        session.close()


def test_uxa2_fixture_has_at_least_one_item_per_attention_type() -> None:
    base = "/projects/demo"
    approval_id = uuid4()
    request_id = uuid4()
    types = {
        item.type
        for item in [
            *items_from_stage_payload(
                product="blueprint",
                stage="define",
                source="definition",
                artifact_id="definition",
                artifact_version=1,
                href=f"{base}/define",
                return_href=f"{base}/attention",
                open_questions=["Pregunta abierta"],
                gaps=["GAP bloqueante"],
                decisions=[{"key": "decision", "title": "Decision requerida", "reason": "Razon", "severity": "warning"}],
                warnings=["Inconsistencia detectable"],
            ),
            *items_from_approval_gates(
                [
                    SimpleNamespace(
                        id=approval_id,
                        gate_key="gate",
                        title="Approval pendiente",
                        status=ApprovalStatus.pending,
                        requested_in_stage=SessionStage.post_validation,
                        rationale="Requiere aprobacion",
                        instructions="Aprobar o rechazar",
                    )
                ],
                base_href=base,
                return_href=f"{base}/attention",
            ),
            *items_from_commercial_access(
                SimpleNamespace(checkout_state="pending"),
                [
                    SimpleNamespace(
                        id=request_id,
                        product_key="acp",
                        capability="acp.build",
                        reason="Necesita ACP",
                    )
                ],
                base_href=base,
                return_href=f"{base}/attention",
            ),
            *items_from_stage_artifact_state(
                stage="memory",
                artifact_id="memory-artifact",
                artifact_version=2,
                artifact_kind="memory",
                state="stale",
                reason="Version anterior",
                href=f"{base}/memory",
                return_href=f"{base}/attention",
            ),
            *items_from_runtime_operation(
                {"id": "runtime-1", "state": "error", "stage": "validate", "product": "acp", "title": "Fallo runtime"},
                href=f"{base}/attention",
                return_href=f"{base}/validate",
            ),
            *items_from_runtime_operation(
                {"id": "hitl-1", "state": "waiting_for_user", "stage": "design", "product": "blueprint", "title": "HITL"},
                href=f"{base}/attention",
                return_href=f"{base}/design",
            ),
            *items_from_handoffs(
                [
                    SimpleNamespace(
                        id=uuid4(),
                        status="pending",
                        title="Handoff gobierno",
                        from_stage=SessionStage.build_blueprint,
                        summary="Pendiente",
                        owner_role="local_admin",
                    )
                ],
                base_href=base,
                return_href=f"{base}/attention",
            ),
            *items_from_governance_policies(
                [
                    SimpleNamespace(
                        id=uuid4(),
                        compliance_status="fail",
                        label="Politica fallida",
                        summary="No cumple",
                        scope="validate",
                        evidence=[],
                    )
                ],
                base_href=base,
                return_href=f"{base}/attention",
            ),
        ]
    }

    assert {
        "question",
        "gap",
        "decision",
        "approval",
        "confirmation",
        "validation",
        "hitl",
        "inconsistency",
        "stale",
        "runtime_error",
        "access_request",
    }.issubset(types)


def test_uxa2_answer_resolver_updates_acp_question_source_and_removes_item() -> None:
    session = _create_memory_engine_session()
    try:
        user, _, record = _seed_minimal_records(session)
        readiness = ConstructionReadinessReport(
            gaps=[
                ConstructionGapEntry(
                    gap_key="runtime_contract",
                    title="Runtime incompleto",
                    severity="blocking",
                    blocking_stage="package",
                    summary="Falta stack runtime.",
                    questions=[
                        ConstructionQuestionEntry(
                            question_key="runtime_stack",
                            question_text="Define stack runtime",
                            rationale="Necesario para ACP",
                            target_owner="implementation_owner",
                            blocking=True,
                        )
                    ],
                )
            ]
        )
        access = CommercialAccessSnapshotV2(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=user.id,
            tier=CommercialTier.acp,
        )
        snapshot = _snapshot(record)
        before = build_attention_response_v2(session, record=record, snapshot=snapshot, readiness=readiness, access=access)
        question = next(item for item in before.items if item.source == "acp_questions")

        result = apply_attention_action_v2(
            session,
            record=record,
            snapshot=snapshot,
            readiness=readiness,
            access=access,
            current_user=user,
            item_key=question.key,
            payload=AttentionActionRequestV2(
                action_kind="answer",
                answer_text="python: FastAPI; database: PostgreSQL",
                idempotency_key="answer-runtime-stack",
            ),
        )
        session.commit()
        responses = session.exec(select(ConstructionQuestionResponseRecord)).all()
        after = build_attention_response_v2(session, record=record, snapshot=snapshot, readiness=readiness, access=access)

        assert result.status == "applied"
        assert responses[0].question_key == "runtime_stack"
        assert question.key not in {item.key for item in after.items}
    finally:
        session.close()


def test_uxa2_blueprint_tier_does_not_surface_acp_readiness_as_active_blocker() -> None:
    session = _create_memory_engine_session()
    try:
        user, _, record = _seed_minimal_records(session)
        readiness = ConstructionReadinessReport(
            gaps=[
                ConstructionGapEntry(
                    gap_key="acp_package_validation_blocked",
                    title="El ACP aun no pasa sus validaciones base",
                    severity="blocking",
                    blocking_stage="package_validation",
                    summary="El ACP debe cerrar validaciones antes de construccion.",
                )
            ]
        )
        access = CommercialAccessSnapshotV2(
            workspace_id=record.workspace_id,
            session_id=record.id,
            user_id=user.id,
            tier=CommercialTier.blueprint,
        )

        response = build_attention_response_v2(
            session,
            record=record,
            snapshot=_snapshot(record),
            readiness=readiness,
            access=access,
        )

        assert response.blocking_count == 0
        assert all(item.product != "acp" for item in response.items)
    finally:
        session.close()


def test_uxa2_endpoint_filters_paginates_and_resolves_with_idempotency(client: TestClient) -> None:
    headers = _auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]
    workspace_id = create_response.json()["workspace_id"]

    db_session = _db_session_from_client(client)
    try:
        record = db_session.get(SessionRecord, UUID(session_id))
        user = db_session.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).first()
        assert record is not None
        assert user is not None
        approval = ApprovalGateRecord(
            session_id=record.id,
            gate_key="uxa2_gate",
            title="Aprobar gate UXA2",
            requested_in_stage=SessionStage.post_validation,
            status=ApprovalStatus.pending,
        )
        db_session.add(approval)
        access_request = CommercialAccessRequestRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            requester_user_id=user.id,
            capability="acp.build",
            product_key="acp",
            status=CommercialAccessRequestStatus.pending,
            reason="Validar attention access request",
        )
        db_session.add(access_request)
        stale = JourneyStageArtifactRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            stage_key="tools",
            artifact_kind="tool_recommendation",
            version_number=3,
            state=JourneyArtifactState.stale,
            proposal_payload={
                "open_questions": ["Confirmar owner de ticketing."],
                "critic_findings": [
                    {
                        "finding_key": "tool_conflict",
                        "title": "Herramientas incompatibles",
                        "detail": "Dos herramientas cubren la misma capacidad.",
                        "severity": "warning",
                    }
                ],
            },
            stale_reasons=["Design cambio despues de Tools."],
        )
        db_session.add(stale)
        db_session.commit()
    finally:
        db_session.close()

    response = client.get(f"/api/v1/sessions/{session_id}/attention-v2?current_stage=tools&limit=1", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "attention.v2"
    assert payload["workspace_id"] == workspace_id
    assert payload["total_count"] >= 3
    assert len(payload["items"]) == 1
    assert payload["cursor"]

    stale_response = client.get(f"/api/v1/sessions/{session_id}/attention-v2?type=stale", headers=headers)
    stale_item = stale_response.json()["items"][0]
    stale_conflict = client.post(
        f"/api/v1/sessions/{session_id}/attention-v2/{stale_item['key']}/actions",
        headers=headers,
        json={
            "action_kind": "regenerate",
            "idempotency_key": "stale-old-version",
            "source_artifact_version": stale_item["source_ref"]["artifact_version"] - 1,
        },
    )
    assert stale_conflict.status_code == 200
    assert stale_conflict.json()["status"] == "conflict"

    approval_response = client.get(f"/api/v1/sessions/{session_id}/attention-v2?type=approval", headers=headers)
    approval_item = approval_response.json()["items"][0]
    apply_response = client.post(
        f"/api/v1/sessions/{session_id}/attention-v2/{approval_item['key']}/actions",
        headers=headers,
        json={"action_kind": "approve", "idempotency_key": "approve-uxa2-gate", "resolution_note": "ok"},
    )
    assert apply_response.status_code == 200
    assert apply_response.json()["status"] == "applied"
    duplicate_response = client.post(
        f"/api/v1/sessions/{session_id}/attention-v2/{approval_item['key']}/actions",
        headers=headers,
        json={"action_kind": "approve", "idempotency_key": "approve-uxa2-gate", "resolution_note": "ok"},
    )
    assert duplicate_response.status_code == 200
    assert duplicate_response.json()["status"] == "duplicate"

    remaining_approvals = client.get(f"/api/v1/sessions/{session_id}/attention-v2?type=approval", headers=headers)
    assert remaining_approvals.status_code == 200
    assert remaining_approvals.json()["total_count"] == 0


def test_uxa2_attention_v2_respects_workspace_isolation(client: TestClient) -> None:
    headers = _auth_headers(client)
    first = client.post("/api/v1/sessions", headers=headers).json()
    second = client.post("/api/v1/sessions", headers=headers).json()

    db_session = _db_session_from_client(client)
    try:
        first_record = db_session.get(SessionRecord, UUID(first["id"]))
        second_record = db_session.get(SessionRecord, UUID(second["id"]))
        assert first_record is not None
        assert second_record is not None
        db_session.add(
            CommercialAccessRequestRecord(
                workspace_id=first_record.workspace_id,
                session_id=first_record.id,
                requester_user_id=first_record.user_id,
                capability="acp.build",
                product_key="acp",
                status=CommercialAccessRequestStatus.pending,
                reason="Solo primer proyecto",
            )
        )
        db_session.commit()
    finally:
        db_session.close()

    first_attention = client.get(f"/api/v1/sessions/{first['id']}/attention-v2?type=access_request", headers=headers)
    second_attention = client.get(f"/api/v1/sessions/{second['id']}/attention-v2?type=access_request", headers=headers)

    assert first_attention.status_code == 200
    assert second_attention.status_code == 200
    assert first_attention.json()["total_count"] == 1
    assert second_attention.json()["total_count"] == 0
