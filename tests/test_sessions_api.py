import json
from io import BytesIO
from collections.abc import Generator
from pathlib import Path
from uuid import UUID
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlmodel import select

from app.core.config import get_settings
from app.db import get_session
from app.models import (
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintRecord,
    CanvasArtifact,
    CanvasRecord,
    DiscoveryArtifact,
    EstimationBenchmarkRef,
    EstimationComplexityDriver,
    EstimationConfidenceAdjustmentProposal,
    EstimationRiskRegisterEntry,
    EstimationScenarioAdjustment,
    EstimationSavingsOpportunity,
    EstimationUncertaintyFactor,
    PlatformRole,
    PlatformRoleAssignmentRecord,
    RuntimeCatalogEntryRecord,
    OpportunityRecord,
    SessionRecord,
    JourneyStageArtifactRecord,
    SkillRunArtifactRecord,
    SkillRunRecord,
    SessionStage,
    ToolRecommendationArtifact,
    ToolRecommendationConfidence,
    ToolRecommendationEnvelope,
    ToolRecommendationLLMDecision,
    ToolRecommendationLLMOutput,
    MemoryRecommendationArtifact,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.auth_service import hash_password
from app.services.llm_runtime.builder_contracts import (
    AcceptanceCriterion,
    AgentDesignProposalOutput,
    BusinessRule,
    CritiqueFinding,
    DesignCritiqueOutput,
    Dependency,
    DiscoveryAnalysisOutput,
    EstimationRiskAnalysisOutput,
    FunctionalRequirement,
    LLMArtifactResult,
    MemoryArchitectureCritiqueOutput,
    MemoryArchitectureRecommendationOutput,
    NonFunctionalRequirement,
    OpenQuestion,
    PrioritizedQuestion,
    RequirementTraceEntry,
    RequirementsDefinitionOutput,
    StructuredInsight,
    ValidationRunJudgmentOutput,
    ValidationScenarioGenerationOutput,
    ValidationScenarioItem,
    ValidationSimulationOutput,
)
from app.services.openai_builder import BlueprintNarrativeOutput
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def auth_headers(client: TestClient) -> dict[str, str]:
    return auth_headers_for_credentials(client, email=TEST_EMAIL, password=TEST_PASSWORD)


def auth_headers_for_credentials(client: TestClient, *, email: str, password: str) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def seed_user(client: TestClient, *, email: str, password: str, full_name: str) -> None:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        existing = session.exec(select(UserRecord).where(UserRecord.email == email)).first()
        if existing is not None:
            return
        session.add(
            UserRecord(
                email=email,
                full_name=full_name,
                password_hash=hash_password(password),
            )
        )
        session.commit()
    finally:
        session_generator.close()


def create_workspace_for_user(client: TestClient, *, email: str, name: str, role: WorkspaceRole = WorkspaceRole.editor) -> str:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        user = session.exec(select(UserRecord).where(UserRecord.email == email)).first()
        assert user is not None
        workspace = WorkspaceRecord(
            name=name,
            slug=f"{name.lower().replace(' ', '-')}-{str(user.id)[:8]}",
            created_by_user_id=user.id,
        )
        session.add(workspace)
        session.flush()
        session.add(
            WorkspaceMembershipRecord(
                workspace_id=workspace.id,
                user_id=user.id,
                role=role,
            )
        )
        session.commit()
        return str(workspace.id)
    finally:
        session_generator.close()


def assign_platform_role(
    client: TestClient,
    *,
    email: str,
    role: PlatformRole = PlatformRole.platform_admin,
) -> None:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        user = session.exec(select(UserRecord).where(UserRecord.email == email)).first()
        assert user is not None
        assignment = session.exec(
            select(PlatformRoleAssignmentRecord).where(
                PlatformRoleAssignmentRecord.user_id == user.id,
                PlatformRoleAssignmentRecord.role == role,
            )
        ).first()
        if assignment is None:
            session.add(
                PlatformRoleAssignmentRecord(
                    user_id=user.id,
                    role=role,
                )
            )
        else:
            assignment.is_active = True
            assignment.updated_at = utc_now()
            session.add(assignment)
        session.commit()
    finally:
        session_generator.close()


def create_session_for_workspace(client: TestClient, *, email: str, workspace_id: str, title: str = "Proyecto externo") -> str:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        user = session.exec(select(UserRecord).where(UserRecord.email == email)).first()
        assert user is not None
        record = SessionRecord(
            user_id=user.id,
            workspace_id=UUID(workspace_id),
            title=title,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return str(record.id)
    finally:
        session_generator.close()


def complete_discovery_payload() -> dict:
    return {
        "problem_statement": "Disenar agentes de soporte con metodologia Lean y bajo riesgo operativo.",
        "current_user": "Arquitecto de soluciones",
        "current_process": "Recoge discovery en documentos, decide arquitectura y luego redacta artefactos manualmente.",
        "desired_outcome": "Generar un blueprint implementable con tools, memoria, evaluacion y seguridad.",
        "autonomy_level": "high",
        "constraints": [
            "Sin microservicios en MVP",
            "No ejecutar side effects irreversibles sin aprobacion humana",
        ],
        "operational_baseline": {
            "current_time_spent": "6 horas por caso",
            "current_cost": "Retrabajo tecnico y validaciones tardias",
            "frequent_errors": [
                "Se pierde contexto entre discovery y blueprint",
                "No se recorta el alcance del MVP",
            ],
            "automation_opportunities": [
                "Normalizar discovery en estructura",
                "Generar artefactos base sin rehacer documentos",
            ],
        },
        "mvp_definition": {
            "v1_scope": [
                "Capturar discovery estructurado",
                "Construir canvas y blueprint inicial",
            ],
            "out_of_scope": [
                "Subagentes operativos",
                "Provisioning automatico",
            ],
            "north_star_metric": "Paquete de implementacion util en una sola sesion",
            "non_delegable_decisions": [
                "Aprobar el handoff a implementacion",
            ],
        },
    }


def unwrap_discovery_payload(payload):
    return payload.discovery_capture if hasattr(payload, "discovery_capture") else payload


def build_discovery_artifact_from_payload(payload) -> DiscoveryArtifact:
    payload = unwrap_discovery_payload(payload)
    return DiscoveryArtifact(
        problem_statement=payload.problem_statement,
        current_user=payload.current_user,
        current_process=payload.current_process,
        desired_outcome=payload.desired_outcome,
        autonomy_level=payload.autonomy_level,
        constraints=payload.constraints,
        operational_baseline=payload.operational_baseline,
        mvp_definition=payload.mvp_definition,
        case_type="automatizacion",
        value_statement="Reducir retrabajo con contexto compacto y trazable.",
    )


def build_discovery_analysis_artifact_from_payload(payload) -> DiscoveryAnalysisOutput:
    payload = unwrap_discovery_payload(payload)
    discovery = build_discovery_artifact_from_payload(payload)
    return DiscoveryAnalysisOutput(
        summary="El discovery tiene suficiente contexto para construir una propuesta trazable y detectar preguntas antes de Define.",
        facts=[
            StructuredInsight(
                key="current_process",
                statement=payload.current_process,
                source_refs=["discovery.current_process"],
                confidence=0.95,
            ),
            StructuredInsight(
                key="desired_outcome",
                statement=payload.desired_outcome,
                source_refs=["discovery.desired_outcome"],
                confidence=0.93,
            ),
        ],
        inferred_needs=[
            StructuredInsight(
                key="traceable_handoff",
                statement="Se necesita una version aprobable de Discover antes de construir Define.",
                source_refs=["discovery.problem_statement", "discovery.mvp_definition.north_star_metric"],
                confidence=0.86,
            )
        ],
        assumptions=[
            StructuredInsight(
                key="source_availability",
                statement="Las fuentes y restricciones declaradas son suficientes para un primer canvas Lean.",
                source_refs=["discovery.constraints"],
                confidence=0.64,
            )
        ],
        ambiguities=[
            StructuredInsight(
                key="document_sources",
                statement="No se explicitan fuentes documentales ni ownership de conocimiento por workspace.",
                source_refs=["discovery.constraints"],
                confidence=0.58,
            )
        ],
        open_questions=[
            PrioritizedQuestion(
                key="knowledge_sources",
                question="Que fuentes documentales por workspace se usaran para enriquecer etapas posteriores?",
                rationale="Impacta Define, Design y Memory por recuperacion de contexto y trazabilidad.",
                priority="high",
                blocking_stages=["define", "design", "memory"],
                suggested_answer="Listar fuentes aprobadas, owner y frecuencia de actualizacion.",
            )
        ],
        domain_signals=[
            StructuredInsight(
                key="lean_builder",
                statement="El caso corresponde a un builder Lean orientado a arquitectura, memoria y gobernanza.",
                source_refs=["discovery.problem_statement", "discovery.current_process"],
                confidence=0.82,
            )
        ],
        risk_signals=[
            StructuredInsight(
                key="stage_drift",
                statement="Existe riesgo de drift entre discovery, canvas y blueprint si no se aprueba una version trazable.",
                source_refs=["discovery.desired_outcome", "discovery.mvp_definition.non_delegable_decisions"],
                confidence=0.9,
            )
        ],
        sensitive_data_signals=[],
        missing_information=[],
        evidence_refs=["session.discovery_capture", "taxonomy.lean_discovery"],
        confidence=0.82,
        normalized_discovery_candidate=discovery,
    )


def build_definition_artifact_from_discovery_canvas(
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
) -> RequirementsDefinitionOutput:
    return RequirementsDefinitionOutput(
        summary="Define consolida requisitos, reglas y criterios trazables antes de pasar a Design.",
        measurable_objectives=[
            discovery.desired_outcome,
            discovery.mvp_definition.north_star_metric,
        ],
        functional_requirements=[
            FunctionalRequirement(
                key="fr-approved-artifacts",
                title="Consumo de artefactos aprobados",
                priority="high",
                status="proposed",
                source_refs=["discovery.desired_outcome", "canvas.agent_profile.human_approvals"],
                rationale="Evitar drift entre etapas.",
                acceptance=["La siguiente etapa solo lee la ultima version aprobada."],
                requirement="Cada etapa debe consumir solo artefactos aprobados.",
                actor=discovery.current_user,
                trigger=discovery.current_process,
                happy_path="El flujo promueve solo versiones aprobadas y bloquea stale.",
                exceptions=["Si cambia upstream, se invalida la salida downstream."],
            )
        ],
        non_functional_requirements=[
            NonFunctionalRequirement(
                key="nfr-traceability",
                title="Trazabilidad completa",
                priority="high",
                status="proposed",
                source_refs=["session.skill_runs"],
                rationale="La ejecucion debe ser auditable.",
                acceptance=["La corrida persiste provider, prompt version y context fingerprint."],
                requirement="La solucion debe conservar trazabilidad completa de las corridas.",
                category="governance",
                metric="trace_fields_coverage",
                target="100%",
            )
        ],
        business_rules=[
            BusinessRule(
                key="rule-human-approval",
                title="Aprobacion humana obligatoria",
                priority="high",
                status="proposed",
                source_refs=["canvas.agent_profile.human_approvals", "discovery.mvp_definition.non_delegable_decisions"],
                rationale="Hay decisiones que no pueden delegarse.",
                acceptance=["Debe existir aprobacion humana antes de promotion."],
                rule="No promover sin aprobacion humana visible.",
                owner="business_owner",
            )
        ],
        acceptance_criteria=[
            AcceptanceCriterion(
                key="ac-define-trace",
                title="Define trazable",
                priority="high",
                status="proposed",
                source_refs=["session.skill_runs"],
                rationale="La definicion debe auditarse sin ambiguedad.",
                acceptance=["La corrida deja skill runs con metadata completa."],
                criterion="El runtime persiste provider, prompt version y fingerprint de contexto.",
                requirement_keys=["fr-approved-artifacts", "nfr-traceability"],
            )
        ],
        dependencies=[
            Dependency(
                key="dep-knowledge",
                title="Knowledge memory indexado",
                priority="medium",
                status="proposed",
                source_refs=["discovery.constraints"],
                rationale="Design y Memory reutilizan conocimiento gobernado.",
                acceptance=["Existe owner y politica de acceso para la fuente."],
                dependency="Knowledge memory indexado",
                dependency_type="knowledge",
                owner="platform_owner",
            )
        ],
        assumptions=[],
        open_questions=[
            OpenQuestion(
                key="question-workspace-sources",
                title="Fuentes privadas por workspace",
                priority="medium",
                status="proposed",
                source_refs=["discovery.constraints"],
                rationale="Impacta memoria y seguridad.",
                acceptance=["La respuesta define owner y filtro por workspace."],
                question="Que fuentes privadas por workspace deben recuperarse?",
                blocking=False,
                impacted_sections=["dependencies", "memory", "design"],
                suggested_answer="Listar owners y sensibilidad de cada fuente.",
            )
        ],
        traceability=[
            RequirementTraceEntry(
                key="trace-fr-approved-artifacts",
                requirement_key="fr-approved-artifacts",
                source_ref="discovery.desired_outcome",
                rationale="El objetivo exige continuidad entre etapas.",
                coverage_status="covered",
            ),
            RequirementTraceEntry(
                key="trace-nfr-traceability",
                requirement_key="nfr-traceability",
                source_ref="session.skill_runs",
                rationale="La trazabilidad tecnica nace de los skill runs persistidos.",
                coverage_status="covered",
            ),
            RequirementTraceEntry(
                key="trace-rule-human-approval",
                requirement_key="rule-human-approval",
                source_ref="canvas.agent_profile.human_approvals",
                rationale="La aprobacion humana visible gobierna la promocion entre etapas.",
                coverage_status="covered",
            ),
            RequirementTraceEntry(
                key="trace-ac-define-trace",
                requirement_key="ac-define-trace",
                source_ref="session.skill_runs",
                rationale="El criterio se valida directamente sobre la metadata persistida.",
                coverage_status="covered",
            ),
            RequirementTraceEntry(
                key="trace-dep-knowledge",
                requirement_key="dep-knowledge",
                source_ref="discovery.constraints",
                rationale="La dependencia se justifica por las restricciones declaradas.",
                coverage_status="covered",
            ),
        ],
        evidence_refs=["session.discovery", "session.canvas", "knowledge.requirements"],
        confidence=0.84,
        canvas_projection=canvas,
    )


def runtime_settings_payload(
    *,
    active_provider: str,
    runner_id: str,
    agent_execution_backend: str | None = None,
    knowledge_access_backend: str = "workspace_staged",
) -> dict:
    return {
        "active_provider": active_provider,
        "agent_execution_backend": (
            agent_execution_backend
            or ("codex_cli" if active_provider == "codex_local" else "provider_native")
        ),
        "knowledge_access_backend": knowledge_access_backend,
        "openai": {
            "fast_model": "gpt-5.4-mini",
            "reasoning_model": "gpt-5.5",
            "reasoning_effort": "low",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "fast_model": "deepseek-v4-flash",
            "reasoning_model": "deepseek-v4-pro",
            "reasoning_effort": "high",
        },
        "codex_local": {
            "command": "codex",
            "model": "gpt-5.5",
            "profile": f"profile-{runner_id}" if active_provider == "codex_local" else "",
            "cost_policy": "hybrid",
            "timeout_ms": 150000,
            "max_concurrency": 1,
            "runner_id": runner_id,
            "auth_mode": "auto",
            "fallback_models": [],
            "primary_agents": [],
            "shadow_agents": [],
            "staged_agents": [],
        },
    }


def patch_workspace_runtime(
    client: TestClient,
    headers: dict[str, str],
    *,
    active_provider: str,
    runner_id: str,
    agent_execution_backend: str | None = None,
    knowledge_access_backend: str = "workspace_staged",
) -> dict:
    response = client.patch(
        "/api/v1/runtime/llm",
        headers=headers,
        json=runtime_settings_payload(
            active_provider=active_provider,
            runner_id=runner_id,
            agent_execution_backend=agent_execution_backend,
            knowledge_access_backend=knowledge_access_backend,
        ),
    )
    assert response.status_code == 200
    return response.json()


def build_session_flow_for_headers(client: TestClient, headers: dict[str, str]) -> str:
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert normalize_response.status_code == 200
    assert normalize_response.json()["status"] == "ready"

    analyze_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert analyze_response.status_code == 200
    discover_artifact = analyze_response.json()

    approve_discover_response = client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{discover_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Discover aprobado en flujo helper.",
            "decision_payload": {
                "approval_reason": "Discovery trazable listo para Define.",
            },
        },
    )
    assert approve_discover_response.status_code == 200

    canvas_response = client.post(f"/api/v1/sessions/{session_id}/build-canvas", headers=headers)
    assert canvas_response.status_code == 200
    assert canvas_response.json()["status"] == "ready"

    define_response = client.post(f"/api/v1/sessions/{session_id}/define-requirements", headers=headers)
    assert define_response.status_code == 200
    define_artifact = define_response.json()

    approve_define_response = client.post(
        f"/api/v1/sessions/{session_id}/journey/define/artifacts/{define_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Define aprobado en flujo helper.",
            "decision_payload": {
                "approval_reason": "Definition listo para blueprint.",
            },
        },
    )
    assert approve_define_response.status_code == 200

    blueprint_response = client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers)
    assert blueprint_response.status_code == 200
    return session_id


def build_session_flow(client: TestClient) -> tuple[dict[str, str], str]:
    headers = auth_headers(client)
    return headers, build_session_flow_for_headers(client, headers)


def approve_design_for_session(client: TestClient, headers: dict[str, str], session_id: str) -> None:
    propose_response = client.post(f"/api/v1/sessions/{session_id}/propose-design", headers=headers)
    assert propose_response.status_code == 200
    artifact = propose_response.json()
    selected_key = (
        artifact.get("proposal_payload", {}).get("selected_design", {}).get("alternative_key")
        or artifact.get("proposal_payload", {}).get("recommended_alternative_key")
        or ""
    )
    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts/{artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Design aprobado en flujo helper.",
            "decision_payload": {
                "selected_alternative_key": selected_key,
            },
        },
    )
    assert approve_response.status_code == 200


def approve_tools_for_session(client: TestClient, headers: dict[str, str], session_id: str) -> tuple[dict, dict]:
    recommend_response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)
    assert recommend_response.status_code == 200
    recommendation_payload = recommend_response.json()
    optional_keys = [item["tool_key"] for item in recommendation_payload["data"]["optional_tools"]]

    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/approve-tools-selection",
        headers=headers,
        json={"include_optional_tool_keys": optional_keys[:1]},
    )
    assert approve_response.status_code == 200
    return recommendation_payload, approve_response.json()


def delete_canonical_session_rows(client: TestClient, *, session_id: str) -> None:
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        session_id_uuid = UUID(session_id)
        for model in (OpportunityRecord, CanvasRecord, BlueprintRecord):
            record = session.exec(select(model).where(model.session_id == session_id_uuid)).first()
            if record is not None:
                session.delete(record)
        session.commit()
    finally:
        session_generator.close()


def approve_memory_for_session(client: TestClient, headers: dict[str, str], session_id: str) -> dict:
    recommendation = client.post(f"/api/v1/sessions/{session_id}/recommend-memory", headers=headers)
    assert recommendation.status_code == 200
    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/approve-memory-profile",
        headers=headers,
        json={
            "note": "Memoria aprobada en helper.",
            "decision_payload": {
                "approval_reason": "La propuesta de memoria es consistente para Validate.",
            },
        },
    )
    assert approve_response.status_code == 200
    return approve_response.json()


def approve_validate_for_session(client: TestClient, headers: dict[str, str], session_id: str) -> dict:
    generate_response = client.post(
        f"/api/v1/sessions/{session_id}/generate-validation-scenarios",
        headers=headers,
        json={"instructions": "Aprobar la simulacion base con foco en trazabilidad y coherencia."},
    )
    assert generate_response.status_code == 200
    validate_artifact = generate_response.json()
    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/journey/validate/artifacts/{validate_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Validate aprobado en helper.",
            "decision_payload": {
                "approval_reason": "La simulacion base ya representa la conducta esperada antes de Package.",
            },
        },
    )
    assert approve_response.status_code == 200
    return approve_response.json()


def upgrade_session_tier(client: TestClient, headers: dict[str, str], session_id: str, tier: str = "acp") -> None:
    response = client.patch(
        f"/api/v1/sessions/{session_id}/commercial-tier",
        headers=headers,
        json={"tier": tier},
    )
    assert response.status_code == 200
    assert response.json()["commercial_access"]["tier"] == tier


class FakeLLMTraceBuilderService:
    def normalize_discovery(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        artifact = build_discovery_artifact_from_payload(payload)
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="hybrid",
            effective_context_backend="hybrid_inline_compact",
            context_used_sources=[
                {
                    "key": "discovery_capture",
                    "uri": "session://discovery-capture",
                    "required": True,
                    "source_refs": ["session.discovery"],
                    "source_lineage": ["session://discovery-capture::state::1111111111111111"],
                    "source_version": "lineage::1111111111111111",
                }
            ],
            context_stats={
                "budget_tokens": 700,
                "assembled_estimated_tokens": 220,
                "baseline_estimated_tokens": 321,
                "reduction_estimated_tokens": 101,
            },
        )

    def analyze_discovery(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        artifact = build_discovery_analysis_artifact_from_payload(payload)
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="hybrid",
            effective_context_backend="hybrid_inline_compact",
            context_used_sources=[
                {
                    "key": "discovery_analysis_input",
                    "uri": "session://discovery-analysis-input",
                    "required": True,
                    "source_refs": ["session.discovery", "session.opportunity"],
                    "source_lineage": ["session://discovery-analysis-input::state::1212121212121212"],
                    "source_version": "lineage::1212121212121212",
                }
            ],
            context_stats={
                "budget_tokens": 720,
                "assembled_estimated_tokens": 244,
                "baseline_estimated_tokens": 358,
                "reduction_estimated_tokens": 114,
            },
        )

    def build_canvas(self, discovery, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        artifact = CanvasArtifact(
            user_goal=discovery.desired_outcome,
            mvp_scope=list(discovery.mvp_definition.v1_scope),
            out_of_scope=list(discovery.mvp_definition.out_of_scope),
            success_metric=discovery.mvp_definition.north_star_metric,
            primary_risk="Perder continuidad entre providers",
            agent_profile={
                "mission": "Convertir discovery en canvas consistente.",
                "primary_user": discovery.current_user,
                "agent_task": "Construir alcance Lean",
                "allowed_decisions": ["Proponer MVP"],
                "prohibited_decisions": ["Promover a implementacion sin aprobacion"],
                "key_inputs": ["Discovery aprobado"],
                "expected_outputs": ["Canvas listo para blueprint"],
                "human_approvals": ["Promocion"],
                "success_metrics": [discovery.mvp_definition.north_star_metric],
            },
        )
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="deepseek",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="inline_context",
            effective_context_backend="inline_context_compact",
            context_used_sources=[
                {
                    "key": "normalized_discovery",
                    "uri": "session://normalized-discovery",
                    "required": True,
                    "source_refs": ["session.discovery"],
                    "source_lineage": ["session://normalized-discovery::state::2222222222222222"],
                    "source_version": "lineage::2222222222222222",
                }
            ],
            context_stats={
                "budget_tokens": 640,
                "assembled_estimated_tokens": 180,
                "baseline_estimated_tokens": 247,
                "reduction_estimated_tokens": 67,
            },
        )

    def define_requirements(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        artifact = build_definition_artifact_from_discovery_canvas(payload.discovery, payload.canvas)
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_compact",
            context_used_sources=[
                {
                    "key": "requirements_definition_input",
                    "uri": "session://requirements-definition-input",
                    "required": True,
                    "source_refs": ["session.discovery", "session.canvas", "knowledge.requirements"],
                    "source_lineage": ["session://requirements-definition-input::state::2323232323232323"],
                    "source_version": "lineage::2323232323232323",
                }
            ],
            context_stats={
                "budget_tokens": 980,
                "assembled_estimated_tokens": 312,
                "baseline_estimated_tokens": 488,
                "reduction_estimated_tokens": 176,
            },
        )

    def propose_agent_design(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return LLMArtifactResult(
            artifact=AgentDesignProposalOutput(
                summary="El provider fake propone una opcion simple como baseline.",
                recommended_alternative_key="single_agent_with_skills",
                architecture="single_agent_with_skills",
                reasoning_pattern="Plan-and-Execute",
                coordination_model="single_agent_with_skills",
                decision_rationale="La propuesta fake privilegia simplicidad con checkpoints visibles.",
                open_questions=["Confirmar si el caso requiere especialistas reales o solo skills."],
                confidence=0.78,
                narrative="Un agente unico con skills cubre el alcance aprobado sin sobredimensionar el MVP.",
            ),
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_compact",
            context_used_sources=[
                {
                    "key": "agent_design_input",
                    "uri": "session://agent-design-input",
                    "required": True,
                    "source_refs": ["session.discovery", "session.canvas", "session.journey_latest_artifacts.define"],
                    "source_lineage": ["session://agent-design-input::state::8787878787878787"],
                    "source_version": "lineage::8787878787878787",
                }
            ],
            context_stats={
                "budget_tokens": 1240,
                "assembled_estimated_tokens": 386,
                "baseline_estimated_tokens": 612,
                "reduction_estimated_tokens": 226,
            },
        )

    def critique_agent_design(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        return LLMArtifactResult(
            artifact=DesignCritiqueOutput(
                overall_status="needs_revision",
                summary="La propuesta es viable y mantiene un hallazgo menor de sobrearquitectura potencial.",
                findings=[
                    CritiqueFinding(
                        finding_key="design-cost-watch",
                        title="Vigilar crecimiento innecesario del diseño",
                        severity="warning",
                        detail="Si el alcance sigue simple, evita evolucionar a multiagente antes de HT8.",
                        suggested_action="Mantener esta etapa en single agent with skills hasta validar Tools y Memory.",
                        source_refs=["design.alternatives"],
                    )
                ],
                contradictions=[],
                missing_evidence=[],
            ),
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_compact",
            context_used_sources=[
                {
                    "key": "agent_design_critique_input",
                    "uri": "session://agent-design-critique-input",
                    "required": True,
                    "source_refs": ["session.discovery", "session.canvas", "session.journey_latest_artifacts.design"],
                    "source_lineage": ["session://agent-design-critique-input::state::8989898989898989"],
                    "source_version": "lineage::8989898989898989",
                }
            ],
            context_stats={
                "budget_tokens": 980,
                "assembled_estimated_tokens": 304,
                "baseline_estimated_tokens": 522,
                "reduction_estimated_tokens": 218,
            },
        )

    def synthesize_blueprint_narrative(
        self,
        discovery: DiscoveryArtifact,
        canvas: CanvasArtifact,
        blueprint: BlueprintArtifact,
        *,
        context_bundle=None,
    ) -> LLMArtifactResult:
        del discovery, canvas, blueprint, context_bundle
        return LLMArtifactResult(
            artifact=BlueprintNarrativeOutput(
                narrative="Narrativa sintetizada con continuidad entre routing, contexto y memoria."
            ),
            provider_key="codex_local",
            execution_backend="shadow_codex_cli",
            execution_mode="shadow",
            shadow_provider_key="codex_local",
            route_reason="La capacidad esta marcada en codex_local.shadow_agents para corrida paralela. Shadow promovido por indisponibilidad del provider activo.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_filesystem",
            context_used_sources=[
                {
                    "key": "narrative_discovery",
                    "uri": "session://narrative-discovery",
                    "required": True,
                    "source_refs": ["session.discovery"],
                    "source_lineage": ["session://narrative-discovery::state::3333333333333333"],
                    "source_version": "lineage::3333333333333333",
                },
                {
                    "key": "narrative_canvas",
                    "uri": "session://narrative-canvas",
                    "required": True,
                    "source_refs": ["session.canvas"],
                    "source_lineage": ["session://narrative-canvas::state::4444444444444444"],
                    "source_version": "lineage::4444444444444444",
                },
                {
                    "key": "narrative_blueprint",
                    "uri": "session://narrative-blueprint",
                    "required": True,
                    "source_refs": ["session.blueprint"],
                    "source_lineage": ["session://narrative-blueprint::state::5555555555555555"],
                    "source_version": "lineage::5555555555555555",
                },
            ],
            context_stats={
                "budget_tokens": 1200,
                "assembled_estimated_tokens": 320,
                "baseline_estimated_tokens": 500,
                "reduction_estimated_tokens": 180,
            },
        )

    def recommend_minimal_tools(self, prompt_input, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        decisions = [
            ToolRecommendationLLMDecision(
                tool_key=item,
                classification="mandatory",
                decision_reason="El provider fake conserva el shortlist mandatory del preflight.",
                source_evidence=["preflight.mandatory_capabilities"],
                confidence=0.82,
            )
            for item in prompt_input.mandatory_tool_keys
        ]
        return LLMArtifactResult(
            artifact=ToolRecommendationLLMOutput(
                summary="Fake provider selecciono tools minimas con contexto compacto.",
                confidence=ToolRecommendationConfidence(
                    overall=0.76,
                    band="medium",
                    rationale="La seleccion usa solo tools permitidas por el preflight.",
                ),
                tool_decisions=decisions,
            ),
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="hybrid",
            effective_context_backend="hybrid_inline_compact",
            context_used_sources=[
                {
                    "key": "tool_recommendation_case",
                    "uri": "session://tool-recommendation-case",
                    "required": True,
                    "source_refs": ["session.discovery", "session.canvas", "session.blueprint"],
                    "source_lineage": ["session://tool-recommendation-case::state::6666666666666666"],
                    "source_version": "lineage::6666666666666666",
                },
                {
                    "key": "tool_recommendation_catalog",
                    "uri": "session://tool-recommendation-catalog",
                    "required": True,
                    "source_refs": ["session.tool_recommendation.preflight"],
                    "source_lineage": ["session://tool-recommendation-catalog::state::7777777777777777"],
                    "source_version": "lineage::7777777777777777",
                },
            ],
            context_stats={
                "budget_tokens": 840,
                "assembled_estimated_tokens": 248,
                "baseline_estimated_tokens": 472,
                "reduction_estimated_tokens": 224,
            },
        )

    def recommend_memory_architecture(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        artifact = MemoryArchitectureRecommendationOutput(
            memory_strategy=payload.blueprint.memory_strategy or "session_memory",
            short_term_strategy="Mantener contexto minimo por etapa y checkpoints compactos.",
            long_term_strategy="Persistir solo decisiones aprobadas y artefactos trazables por workspace.",
            retrieval_strategy="Recuperar contexto solo bajo demanda y con evidencia aprobada.",
            storage_layers=list(payload.blueprint.memory_profile.storage_layers),
            write_policy=payload.blueprint.memory_profile.write_policy
            or "Persistir solo decisiones aprobadas y resumenes compactos.",
            pruning_policy=payload.blueprint.memory_profile.retention_policy or "Aplicar TTL por etapa y limpieza segura.",
            security_notes=["Aislamiento por workspace", "Solo artefactos aprobados pasan a memoria durable"],
            open_questions=[],
            rationale="La propuesta usa el blueprint vigente y el digest aprobado de Herramientas.",
        )
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_compact",
            context_used_sources=[
                {
                    "key": "memory_architecture_input",
                    "uri": "session://memory-architecture-input",
                    "required": True,
                    "source_refs": [
                        "session.discovery",
                        "session.canvas",
                        "session.journey_latest_artifacts.define",
                        "session.journey_latest_artifacts.design",
                        "session.journey_latest_artifacts.tools",
                    ],
                    "source_lineage": ["session://memory-architecture-input::state::8181818181818181"],
                    "source_version": "lineage::8181818181818181",
                }
            ],
            context_stats={
                "budget_tokens": 900,
                "assembled_estimated_tokens": 266,
                "baseline_estimated_tokens": 501,
                "reduction_estimated_tokens": 235,
            },
        )

    def critique_memory_architecture(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        artifact = MemoryArchitectureCritiqueOutput(
            overall_status="accepted",
            summary="La memoria propuesta es minima, gobernada y consistente con el digest aprobado.",
            findings=[],
            contradictions=[],
            missing_evidence=[],
        )
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_compact",
            context_used_sources=[
                {
                    "key": "memory_architecture_critique_input",
                    "uri": "session://memory-architecture-critique-input",
                    "required": True,
                    "source_refs": [
                        "session.discovery",
                        "session.canvas",
                        "session.journey_latest_artifacts.define",
                        "session.journey_latest_artifacts.design",
                        "session.journey_latest_artifacts.tools",
                        "session.short_term_memory",
                    ],
                    "source_lineage": ["session://memory-architecture-critique-input::state::9191919191919191"],
                    "source_version": "lineage::9191919191919191",
                }
            ],
            context_stats={
                "budget_tokens": 860,
                "assembled_estimated_tokens": 241,
                "baseline_estimated_tokens": 438,
                "reduction_estimated_tokens": 197,
            },
        )

    def generate_validation_scenarios(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        artifact = ValidationScenarioGenerationOutput(
            summary="Se generan escenarios de happy path, timeout de herramientas y no-evidence con escalacion.",
            scenarios=[
                ValidationScenarioItem(
                    scenario_key="happy_path_high_value",
                    title="Happy path con cierre aprobable",
                    objective="Validar el flujo principal del agente con evidencia y gate visible.",
                    steps=[
                        "Recuperar contexto aprobado",
                        "Invocar la tool principal",
                        "Decidir y cerrar con evidencia",
                    ],
                    expected_outcomes=[
                        "Lectura y escritura de memoria visibles",
                        "Decision con criterio trazable",
                    ],
                    failure_signals=["Sin approval gate", "Sin evidencia de memoria"],
                    priority="high",
                ),
                ValidationScenarioItem(
                    scenario_key="tool_timeout_compensation",
                    title="Timeout con compensacion",
                    objective="Comprobar recovery y escalacion segura cuando una tool excede el SLA.",
                    steps=[
                        "Invocar la tool",
                        "Detectar timeout",
                        "Aplicar compensacion y escalar",
                    ],
                    expected_outcomes=[
                        "Timeout visible",
                        "Compensacion explicita",
                    ],
                    failure_signals=["Timeout silencioso", "Sin escalacion"],
                    priority="high",
                ),
                ValidationScenarioItem(
                    scenario_key="no_evidence_human_escalation",
                    title="No-evidence con escalacion",
                    objective="Evitar respuestas inventadas cuando la evidencia no alcanza.",
                    steps=[
                        "Consultar memoria y retrieval",
                        "Detectar ausencia de evidencia",
                        "Escalar al humano",
                    ],
                    expected_outcomes=[
                        "Politica no-evidence visible",
                        "Escalacion humana explicita",
                    ],
                    failure_signals=["Hallucination", "Sin escalacion"],
                    priority="high",
                ),
            ],
            coverage_gaps=[],
        )
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_compact",
            context_used_sources=[
                {
                    "key": "validation_scenario_generation_input",
                    "uri": "session://validation-scenario-generation-input",
                    "required": True,
                    "source_refs": [
                        "session.discovery",
                        "session.canvas",
                        "session.blueprint",
                        "session.journey_latest_artifacts.memory",
                    ],
                    "source_lineage": ["session://validation-scenario-generation-input::state::9292929292929292"],
                    "source_version": "lineage::9292929292929292",
                }
            ],
            context_stats={
                "budget_tokens": 1040,
                "assembled_estimated_tokens": 312,
                "baseline_estimated_tokens": 566,
                "reduction_estimated_tokens": 254,
            },
        )

    def simulate_validation_scenario(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        artifact = ValidationSimulationOutput(
            scenario_key=payload.scenario.scenario_key,
            result_status="needs_revision",
            simulated_transcript=[
                "El agente explica el objetivo del escenario.",
                "El agente justifica la decision principal con evidencia recuperada.",
            ],
            observed_decisions=["El agente decide escalar cuando la evidencia no es concluyente."],
            tool_interactions=["La tool principal responde con trazabilidad suficiente para el escenario."],
            issues=["El transcript deberia reforzar la explicacion del fallback cuando no hay evidencia."],
        )
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_compact",
            context_used_sources=[
                {
                    "key": "validation_scenario_simulation_input",
                    "uri": "session://validation-scenario-simulation-input",
                    "required": True,
                    "source_refs": ["session.blueprint", "session.journey_latest_artifacts.validate"],
                    "source_lineage": ["session://validation-scenario-simulation-input::state::9393939393939393"],
                    "source_version": "lineage::9393939393939393",
                }
            ],
            context_stats={
                "budget_tokens": 980,
                "assembled_estimated_tokens": 288,
                "baseline_estimated_tokens": 471,
                "reduction_estimated_tokens": 183,
            },
        )

    def judge_validation_run(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        artifact = ValidationRunJudgmentOutput(
            scenario_key=payload.simulation.scenario_key,
            judgment="needs_revision",
            summary="La corrida es consistente, pero el judge pide reforzar la explicacion del fallback y de la escalacion.",
            findings=[
                CritiqueFinding(
                    finding_key="validate-fallback-clarity",
                    title="Fallback poco explicado",
                    severity="warning",
                    detail="La corrida deberia verbalizar mejor el motivo del fallback y la politica no-evidence.",
                    suggested_action="Agregar una respuesta estandar para gaps de evidencia y recovery.",
                    source_refs=["validate.timeline", "memory.grounding_policy"],
                )
            ],
            score=82,
        )
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="agent_execution_backend=provider_native mantiene el provider activo sin override.",
            knowledge_access_backend="workspace_staged",
            effective_context_backend="workspace_staged_compact",
            context_used_sources=[
                {
                    "key": "validation_run_judgment_input",
                    "uri": "session://validation-run-judgment-input",
                    "required": True,
                    "source_refs": ["session.blueprint", "session.simulation_runs"],
                    "source_lineage": ["session://validation-run-judgment-input::state::9494949494949494"],
                    "source_version": "lineage::9494949494949494",
                }
            ],
            context_stats={
                "budget_tokens": 900,
                "assembled_estimated_tokens": 274,
                "baseline_estimated_tokens": 452,
                "reduction_estimated_tokens": 178,
            },
        )

    def analyze_estimation_risks(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        artifact = EstimationRiskAnalysisOutput(
            summary="El estimate queda listo para revision, pero debe mantener banda amplia mientras falten pricing o aprobaciones de Validate.",
            complexity_drivers=[
                EstimationComplexityDriver(
                    driver_key="integrations",
                    title="Integraciones y side effects",
                    workstream_key="integrations",
                    impact_level="high",
                    summary="Las integraciones con aprobacion siguen siendo el principal driver de riesgo operacional.",
                    evidence_refs=["session.blueprint.tools"],
                )
            ],
            risk_register=[
                EstimationRiskRegisterEntry(
                    risk_key="validate-or-pricing-gap",
                    title="Validate o pricing incompletos",
                    severity="high",
                    likelihood="medium",
                    impact="La confianza comercial no debe cerrarse mientras falten esas evidencias.",
                    mitigation="Completar Validate y confirmar pricing snapshot antes de continuar a Package.",
                    evidence_refs=["session.estimation_report", "session.journey_latest_artifacts.validate"],
                )
            ],
            uncertainty_factors=[
                EstimationUncertaintyFactor(
                    factor_key="validate-coverage",
                    title="Cobertura de Validate",
                    category="governance",
                    impact_area="confidence",
                    summary="Sin validate aprobado la banda debe ampliarse y tratarse como preliminar.",
                    evidence_refs=["session.simulation_runs"],
                )
            ],
            benchmark_refs=[
                EstimationBenchmarkRef(
                    benchmark_key="workspace-actual-sample",
                    title="Workspace calibrated sample",
                    source_kind="workspace_actuals",
                    source_ref="workspace://estimation-runs/sample",
                    sample_size=1,
                    captured_at="2026-07-10T09:00:00",
                    freshness="reciente",
                    summary="Referencia calibrada del mismo workspace.",
                    workspace_scoped=True,
                )
            ],
            scenario_adjustments=[
                EstimationScenarioAdjustment(
                    scenario_key="optimistic",
                    hours_multiplier=0.96,
                    duration_multiplier=0.96,
                    cost_multiplier=0.95,
                    rationale="Menor retrabajo y menos aprobaciones.",
                ),
                EstimationScenarioAdjustment(
                    scenario_key="base",
                    hours_multiplier=1.0,
                    duration_multiplier=1.0,
                    cost_multiplier=1.0,
                    rationale="Escenario base deterministico.",
                ),
                EstimationScenarioAdjustment(
                    scenario_key="conservative",
                    hours_multiplier=1.1,
                    duration_multiplier=1.08,
                    cost_multiplier=1.1,
                    rationale="Mayor hardening y supervision.",
                ),
            ],
            savings_opportunities=[
                EstimationSavingsOpportunity(
                    opportunity_key="close-validate-fast",
                    title="Cerrar Validate antes de comprometer Package",
                    summary="Reduce la banda comercial y evita reprocesar supuestos.",
                    expected_impact="Menor incertidumbre y menos retrabajo.",
                    prerequisites=["Simulation aprobada", "Pricing vigente"],
                    evidence_refs=["session.simulation_runs"],
                )
            ],
            assumptions=["Se reutilizan los catalogos activos del workspace y no se inventan tarifas."],
            questions=[],
            evidence_refs=["workspace.calibration_dashboard"],
            confidence_adjustment_proposal=EstimationConfidenceAdjustmentProposal(
                proposed_score_delta=-4,
                proposed_uncertainty_band_delta=6,
                rationale="Mientras Validate no este completamente cerrada conviene ampliar la banda.",
                evidence_refs=["session.simulation_runs"],
            ),
        )
        return LLMArtifactResult(
            artifact=artifact,
            provider_key="openai",
            execution_backend="provider_native",
            execution_mode="primary",
            route_reason="estimate usa provider activo con contexto controlado.",
            knowledge_access_backend="hybrid",
            effective_context_backend="hybrid_inline_compact",
            context_used_sources=[
                {
                    "key": "estimation_risk_analysis_input",
                    "uri": "session://estimation-risk-analysis-input",
                    "required": True,
                    "source_refs": ["session.estimation_report", "session.blueprint"],
                    "source_lineage": ["session://estimation-risk-analysis-input::state::estimate-ci11"],
                    "source_version": "lineage::estimate-ci11",
                }
            ],
            context_stats={
                "budget_tokens": 1800,
                "assembled_estimated_tokens": 690,
                "baseline_estimated_tokens": 1120,
                "reduction_estimated_tokens": 430,
            },
        )


class RuntimeBoundBuilderService:
    def __init__(self, provider_key: str, sink: list[tuple[str, str]]) -> None:
        self.provider_key = provider_key
        self.sink = sink

    def normalize_discovery(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        self.sink.append(("discover", self.provider_key))
        artifact = build_discovery_artifact_from_payload(payload).model_copy(
            update={"value_statement": f"runtime={self.provider_key}"}
        )
        return LLMArtifactResult(artifact=artifact, provider_key=self.provider_key)

    def analyze_discovery(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        self.sink.append(("discover_analysis", self.provider_key))
        artifact = build_discovery_analysis_artifact_from_payload(payload)
        return LLMArtifactResult(artifact=artifact, provider_key=self.provider_key)

    def build_canvas(self, discovery, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        self.sink.append(("define", self.provider_key))
        artifact = CanvasArtifact(
            user_goal=discovery.desired_outcome,
            mvp_scope=list(discovery.mvp_definition.v1_scope),
            out_of_scope=list(discovery.mvp_definition.out_of_scope),
            success_metric=discovery.mvp_definition.north_star_metric,
            primary_risk=f"runtime={self.provider_key}",
            agent_profile={
                "mission": "Mantener el runtime por sesion.",
                "primary_user": discovery.current_user,
                "agent_task": "Convertir discovery en canvas",
                "allowed_decisions": ["Proponer MVP"],
                "prohibited_decisions": ["Promover sin aprobacion"],
                "key_inputs": ["Discovery"],
                "expected_outputs": ["Canvas"],
                "human_approvals": ["Promocion"],
                "success_metrics": [discovery.mvp_definition.north_star_metric],
            },
        )
        return LLMArtifactResult(artifact=artifact, provider_key=self.provider_key)

    def define_requirements(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        self.sink.append(("define_requirements", self.provider_key))
        artifact = build_definition_artifact_from_discovery_canvas(payload.discovery, payload.canvas)
        return LLMArtifactResult(artifact=artifact, provider_key=self.provider_key)

    def propose_agent_design(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        self.sink.append(("design_proposal", self.provider_key))
        return LLMArtifactResult(
            artifact=AgentDesignProposalOutput(
                summary=f"Runtime {self.provider_key} propuso una opcion simple.",
                recommended_alternative_key="single_agent_with_skills",
                architecture="single_agent_with_skills",
                reasoning_pattern="Plan-and-Execute",
                coordination_model="single_agent_with_skills",
                decision_rationale=f"Runtime {self.provider_key} privilegia simplicidad gobernada.",
                confidence=0.74,
            ),
            provider_key=self.provider_key,
        )

    def critique_agent_design(self, payload, *, context_bundle=None) -> LLMArtifactResult:
        del payload, context_bundle
        self.sink.append(("design_critique", self.provider_key))
        return LLMArtifactResult(
            artifact=DesignCritiqueOutput(
                overall_status="accepted",
                summary=f"Runtime {self.provider_key} no encontro blockers en Design.",
                findings=[],
                contradictions=[],
                missing_evidence=[],
            ),
            provider_key=self.provider_key,
        )

    def synthesize_blueprint_narrative(
        self,
        discovery: DiscoveryArtifact,
        canvas: CanvasArtifact,
        blueprint: BlueprintArtifact,
        *,
        context_bundle=None,
    ) -> LLMArtifactResult:
        del discovery, canvas, blueprint, context_bundle
        self.sink.append(("design", self.provider_key))
        return LLMArtifactResult(
            artifact=BlueprintNarrativeOutput(
                narrative=f"Narrativa sintetizada con runtime {self.provider_key}."
            ),
            provider_key=self.provider_key,
        )

    def recommend_minimal_tools(self, prompt_input, *, context_bundle=None) -> LLMArtifactResult:
        del context_bundle
        self.sink.append(("tools", self.provider_key))
        return LLMArtifactResult(
            artifact=ToolRecommendationLLMOutput(
                summary=f"Runtime {self.provider_key} resolvio tools minimas.",
                confidence=ToolRecommendationConfidence(
                    overall=0.7,
                    band="medium",
                    rationale="La corrida de runtime por sesion uso el shortlist permitido.",
                ),
                tool_decisions=[
                    ToolRecommendationLLMDecision(
                        tool_key=item,
                        classification="mandatory",
                        decision_reason=f"Runtime {self.provider_key} mantiene la tool mandatory.",
                        source_evidence=["preflight.mandatory_capabilities"],
                        confidence=0.8,
                    )
                    for item in prompt_input.mandatory_tool_keys
                ],
            ),
            provider_key=self.provider_key,
        )


def test_backend_smoke_flow_covers_login_discovery_blueprint_evaluation_and_acp_export(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    upgrade_session_tier(client, headers, session_id)

    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    assert bootstrap_response.status_code == 200

    evaluation_response = client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers)
    assert evaluation_response.status_code == 200
    assert evaluation_response.json()["status"] in {"needs_review", "ready"}

    preview_response = client.get(f"/api/v1/sessions/{session_id}/acp/preview", headers=headers)
    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["files"]
    assert preview["manifest_path"] == "ACP/manifest.yaml"

    generate_response = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    assert generate_response.status_code == 200
    generated_preview = generate_response.json()
    assert generated_preview["validation"]["can_export_zip"] is True

    zip_response = client.get(f"/api/v1/sessions/{session_id}/acp/export.zip", headers=headers)
    assert zip_response.status_code == 200
    assert zip_response.content[:2] == b"PK"


def test_session_routes_persist_llm_trace_in_responses_and_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    discovery_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert discovery_response.status_code == 200
    discovery_payload = discovery_response.json()
    assert discovery_payload["llm_trace"]["provider_key"] == "openai"
    assert discovery_payload["llm_trace"]["knowledge_access_backend"] == "hybrid"
    assert discovery_payload["llm_trace"]["context_used_sources"][0]["key"] == "discovery_capture"

    analyze_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert analyze_response.status_code == 200
    discover_artifact = analyze_response.json()

    approve_discover = client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{discover_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Discover aprobado para trazas.",
            "decision_payload": {
                "approval_reason": "Discover listo para construir Define.",
            },
        },
    )
    assert approve_discover.status_code == 200

    canvas_response = client.post(f"/api/v1/sessions/{session_id}/build-canvas", headers=headers)
    assert canvas_response.status_code == 200
    canvas_payload = canvas_response.json()
    assert canvas_payload["llm_trace"]["provider_key"] == "deepseek"
    assert canvas_payload["llm_trace"]["effective_context_backend"] == "inline_context_compact"

    define_response = client.post(f"/api/v1/sessions/{session_id}/define-requirements", headers=headers)
    assert define_response.status_code == 200
    define_payload = define_response.json()
    assert define_payload["provider_key"] == "openai"

    approve_define = client.post(
        f"/api/v1/sessions/{session_id}/journey/define/artifacts/{define_payload['id']}/approve",
        headers=headers,
        json={
            "note": "Define aprobado para blueprint.",
            "decision_payload": {
                "approval_reason": "Definition y trazabilidad aceptadas.",
            },
        },
    )
    assert approve_define.status_code == 200

    blueprint_response = client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers)
    assert blueprint_response.status_code == 200
    blueprint_payload = blueprint_response.json()
    assert blueprint_payload["llm_trace"]["provider_key"] == "codex_local"
    assert blueprint_payload["llm_trace"]["execution_mode"] == "shadow"
    assert blueprint_payload["llm_trace"]["effective_context_backend"] == "workspace_staged_filesystem"

    approve_design_for_session(client, headers, session_id)
    tools_response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)
    assert tools_response.status_code == 200
    tools_payload = tools_response.json()
    assert tools_payload["llm_trace"]["provider_key"] == "openai"
    assert tools_payload["llm_trace"]["context_used_sources"][0]["key"] == "tool_recommendation_case"

    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    skill_runs_by_key = {item["skill_key"]: item for item in snapshot["skill_runs"]}

    assert skill_runs_by_key["discovery_skill"]["llm_trace"]["provider_key"] == "openai"
    assert skill_runs_by_key["lean_scope_skill"]["llm_trace"]["provider_key"] == "deepseek"
    assert skill_runs_by_key["requirements_definition_skill"]["llm_trace"]["provider_key"] == "openai"
    assert skill_runs_by_key["blueprint_generation_skill"]["llm_trace"]["provider_key"] == "codex_local"
    assert skill_runs_by_key["blueprint_generation_skill"]["llm_trace"]["context_used_sources"][2]["key"] == (
        "narrative_blueprint"
    )
    assert skill_runs_by_key["tool_recommendation_skill"]["llm_trace"]["provider_key"] == "openai"


def test_recommend_memory_route_persists_llm_trace_in_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.skill_runtime as skill_runtime

    original_builder_factory = skill_runtime._builder_service_for_stage
    monkeypatch.setattr(
        skill_runtime,
        "_builder_service_for_stage",
        lambda stage_key, runtime_settings=None: (
            FakeLLMTraceBuilderService()
            if stage_key == "memory"
            else original_builder_factory(stage_key, runtime_settings)
        ),
    )

    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)
    approve_tools_for_session(client, headers, session_id)

    memory_response = client.post(f"/api/v1/sessions/{session_id}/recommend-memory", headers=headers)
    assert memory_response.status_code == 200

    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    skill_runs_by_key = {item["skill_key"]: item for item in snapshot["skill_runs"]}

    assert skill_runs_by_key["memory_recommendation_skill"]["llm_trace"]["provider_key"] == "openai"
    assert (
        skill_runs_by_key["memory_recommendation_skill"]["llm_trace"]["context_used_sources"][0]["key"]
        == "memory_architecture_input"
    )
    assert skill_runs_by_key["memory_critique_skill"]["llm_trace"]["provider_key"] == "openai"
    assert (
        skill_runs_by_key["memory_critique_skill"]["llm_trace"]["context_used_sources"][0]["key"]
        == "memory_architecture_critique_input"
    )


def test_session_routes_and_exports_are_isolated_by_owner(client: TestClient) -> None:
    owner_headers, session_id = build_session_flow(client)
    assert owner_headers["Authorization"].startswith("Bearer ")

    seed_user(
        client,
        email="reviewer@leanbuilder.local",
        password="Reviewer123!",
        full_name="Reviewer Local",
    )
    other_headers = auth_headers_for_credentials(
        client,
        email="reviewer@leanbuilder.local",
        password="Reviewer123!",
    )

    protected_paths = (
        f"/api/v1/sessions/{session_id}",
        f"/api/v1/sessions/{session_id}/export/json",
        f"/api/v1/sessions/{session_id}/export/markdown",
        f"/api/v1/sessions/{session_id}/export/construction-pack?preview=true",
        f"/api/v1/sessions/{session_id}/acp/export.zip?profile=design-only",
    )

    for path in protected_paths:
        response = client.get(path, headers=other_headers)
        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"


def test_auth_and_sessions_can_switch_between_user_workspaces(client: TestClient) -> None:
    headers = auth_headers(client)
    me_response = client.get("/api/v1/auth/me", headers=headers)
    assert me_response.status_code == 200
    current_user = me_response.json()
    default_workspace_id = current_user["active_workspace_id"]
    assert default_workspace_id

    secondary_workspace_id = create_workspace_for_user(
        client,
        email=TEST_EMAIL,
        name="Expansion Workspace",
        role=WorkspaceRole.admin,
    )

    switched_response = client.post(
        "/api/v1/auth/workspaces/select",
        headers=headers,
        json={"workspace_id": secondary_workspace_id},
    )
    assert switched_response.status_code == 200
    switched_user = switched_response.json()
    assert switched_user["active_workspace_id"] == secondary_workspace_id
    assert len(switched_user["workspaces"]) >= 2

    default_headers = {**headers, "x-workspace-id": default_workspace_id}
    default_session_response = client.post("/api/v1/sessions", headers=default_headers)
    assert default_session_response.status_code == 201
    default_session_id = default_session_response.json()["id"]
    assert default_session_response.json()["workspace_id"] == default_workspace_id

    secondary_headers = {**headers, "x-workspace-id": secondary_workspace_id}
    secondary_session_response = client.post("/api/v1/sessions", headers=secondary_headers)
    assert secondary_session_response.status_code == 201
    secondary_session_id = secondary_session_response.json()["id"]
    assert secondary_session_response.json()["workspace_id"] == secondary_workspace_id

    default_list_response = client.get("/api/v1/sessions", headers=default_headers)
    assert default_list_response.status_code == 200
    default_session_ids = {item["id"] for item in default_list_response.json()["items"]}
    assert default_session_id in default_session_ids
    assert secondary_session_id not in default_session_ids

    secondary_list_response = client.get("/api/v1/sessions", headers=secondary_headers)
    assert secondary_list_response.status_code == 200
    secondary_session_ids = {item["id"] for item in secondary_list_response.json()["items"]}
    assert secondary_session_id in secondary_session_ids
    assert default_session_id not in secondary_session_ids


def test_platform_admin_can_access_sessions_from_external_workspace(client: TestClient) -> None:
    seed_user(
        client,
        email="external-owner@leanbuilder.local",
        password="ExternalOwner123!",
        full_name="External Owner",
    )
    external_workspace_id = create_workspace_for_user(
        client,
        email="external-owner@leanbuilder.local",
        name="External Workspace",
        role=WorkspaceRole.owner,
    )
    external_session_id = create_session_for_workspace(
        client,
        email="external-owner@leanbuilder.local",
        workspace_id=external_workspace_id,
        title="Proyecto externo administrable",
    )

    headers = auth_headers(client)
    me_response = client.get(
        "/api/v1/auth/me",
        headers={**headers, "x-workspace-id": external_workspace_id},
    )
    assert me_response.status_code == 200
    assert me_response.json()["active_workspace_id"] == external_workspace_id

    snapshot_response = client.get(f"/api/v1/sessions/{external_session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    assert snapshot_response.json()["session"]["workspace_id"] == external_workspace_id


def test_project_portfolio_rename_lifecycle_and_facets(client: TestClient) -> None:
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    created = create_response.json()
    session_id = created["id"]

    rename_response = client.patch(
        f"/api/v1/sessions/{session_id}",
        headers=headers,
        json={"title": "Asistente de beneficios RRHH", "expected_version": created["row_version"]},
    )
    assert rename_response.status_code == 200
    renamed = rename_response.json()
    assert renamed["title"] == "Asistente de beneficios RRHH"
    assert renamed["title_source"] == "manual"
    assert renamed["row_version"] == created["row_version"] + 1
    assert renamed["capabilities"]["can_archive"] is True

    conflict_response = client.patch(
        f"/api/v1/sessions/{session_id}",
        headers=headers,
        json={"title": "Nombre obsoleto", "expected_version": created["row_version"]},
    )
    assert conflict_response.status_code == 409

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json={
            **complete_discovery_payload(),
            "problem_statement": "Titulo sugerido por Discovery que no debe reemplazar el manual.",
        },
    )
    assert normalize_response.status_code == 200
    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot_session = snapshot_response.json()["session"]
    assert snapshot_session["title"] == "Asistente de beneficios RRHH"
    assert snapshot_session["suggested_title"].startswith("Titulo sugerido por Discovery")

    active_list_response = client.get("/api/v1/sessions?q=beneficios", headers=headers)
    assert active_list_response.status_code == 200
    active_payload = active_list_response.json()
    assert active_payload["items"][0]["id"] == session_id
    assert active_payload["page"]["total"] >= 1
    assert active_payload["facets"]["active"] >= 1

    delete_before_archive = client.request(
        "DELETE",
        f"/api/v1/sessions/{session_id}",
        headers=headers,
        json={"confirm_title": "Asistente de beneficios RRHH"},
    )
    assert delete_before_archive.status_code == 409

    archive_response = client.post(f"/api/v1/sessions/{session_id}/archive", headers=headers)
    assert archive_response.status_code == 200
    assert archive_response.json()["archived_at"] is not None

    active_after_archive = client.get("/api/v1/sessions", headers=headers)
    assert session_id not in {item["id"] for item in active_after_archive.json()["items"]}

    archived_list = client.get("/api/v1/sessions?lifecycle=archived", headers=headers)
    assert archived_list.status_code == 200
    archived_item = next(item for item in archived_list.json()["items"] if item["id"] == session_id)
    assert archived_item["capabilities"]["can_restore"] is True
    assert archived_item["capabilities"]["can_delete"] is True

    delete_response = client.request(
        "DELETE",
        f"/api/v1/sessions/{session_id}",
        headers=headers,
        json={"confirm_title": "Asistente de beneficios RRHH"},
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted_at"] is not None

    trash_list = client.get("/api/v1/sessions?lifecycle=trash", headers=headers)
    assert trash_list.status_code == 200
    assert session_id in {item["id"] for item in trash_list.json()["items"]}

    restore_response = client.post(f"/api/v1/sessions/{session_id}/restore", headers=headers)
    assert restore_response.status_code == 200
    assert restore_response.json()["archived_at"] is None
    assert restore_response.json()["deleted_at"] is None


def test_auth_login_succeeds_when_default_workspace_id_is_stored_as_text(client: TestClient) -> None:
    workspace_id = create_workspace_for_user(
        client,
        email=TEST_EMAIL,
        name="Text Workspace Id Regression",
        role=WorkspaceRole.owner,
    )
    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        user = session.exec(select(UserRecord).where(UserRecord.email == TEST_EMAIL)).first()
        assert user is not None
        session.execute(
            text("UPDATE users SET default_workspace_id = :workspace_id WHERE id = :user_id"),
            {
                "workspace_id": workspace_id,
                "user_id": str(user.id),
            },
        )
        session.commit()
    finally:
        session_generator.close()

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
        },
    )

    assert response.status_code == 200
    assert response.json()["user"]["active_workspace_id"] == workspace_id


def test_session_builder_flow_stays_bound_to_session_workspace_runtime_after_workspace_switch(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = auth_headers(client)
    current_user = client.get("/api/v1/auth/me", headers=headers).json()
    default_workspace_id = current_user["active_workspace_id"]
    secondary_workspace_id = create_workspace_for_user(
        client,
        email=TEST_EMAIL,
        name="Runtime Session Isolation",
        role=WorkspaceRole.admin,
    )

    default_headers = {**headers, "x-workspace-id": default_workspace_id}
    secondary_headers = {**headers, "x-workspace-id": secondary_workspace_id}
    patch_workspace_runtime(client, default_headers, active_provider="deepseek", runner_id="workspace-a")
    patch_workspace_runtime(client, secondary_headers, active_provider="codex_local", runner_id="workspace-b")

    create_response = client.post("/api/v1/sessions", headers=default_headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]
    switch_response = client.post(
        "/api/v1/auth/workspaces/select",
        headers=headers,
        json={"workspace_id": secondary_workspace_id},
    )
    assert switch_response.status_code == 200
    assert switch_response.json()["active_workspace_id"] == secondary_workspace_id

    runtime_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: RuntimeBoundBuilderService(
            runtime_settings.active_provider.value if runtime_settings is not None else "global",
            runtime_calls,
        ),
    )

    discovery_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert discovery_response.status_code == 200
    analyze_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert analyze_response.status_code == 200
    discover_artifact = analyze_response.json()
    approve_discover = client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{discover_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Discover aprobado en runtime isolation.",
            "decision_payload": {
                "approval_reason": "Discover listo para continuar.",
            },
        },
    )
    assert approve_discover.status_code == 200
    canvas_response = client.post(
        f"/api/v1/sessions/{session_id}/build-canvas",
        headers=headers,
    )
    assert canvas_response.status_code == 200
    define_response = client.post(
        f"/api/v1/sessions/{session_id}/define-requirements",
        headers=headers,
    )
    assert define_response.status_code == 200
    define_artifact = define_response.json()
    approve_define = client.post(
        f"/api/v1/sessions/{session_id}/journey/define/artifacts/{define_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Define aprobado en runtime isolation.",
            "decision_payload": {
                "approval_reason": "Define listo para blueprint.",
            },
        },
    )
    assert approve_define.status_code == 200
    blueprint_response = client.post(
        f"/api/v1/sessions/{session_id}/build-blueprint",
        headers=headers,
    )
    assert blueprint_response.status_code == 200

    assert runtime_calls == [
        ("discover", "deepseek"),
        ("discover_analysis", "deepseek"),
        ("define", "deepseek"),
        ("define_requirements", "deepseek"),
        ("design", "deepseek"),
    ]


def test_normalize_discovery_accepts_legacy_autonomy_aliases(client: TestClient) -> None:
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    payload = complete_discovery_payload()
    payload["autonomy_level"] = "autonomous"

    response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=payload,
    )
    assert response.status_code == 200

    discovery = response.json()["data"]
    assert discovery["autonomy_level"] == "high"
    assert discovery["case_type"] in {
        "informacion",
        "automatizacion",
        "copiloto",
        "operador_autonomo",
        "sistema_multiagente",
    }


def test_analyze_discovery_partial_draft_returns_questions_and_candidate(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoopAnalysisBuilderService:
        def analyze_discovery(self, payload, *, context_bundle=None):
            del payload, context_bundle
            return None

    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: NoopAnalysisBuilderService(),
    )

    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    partial_payload = {
        "problem_statement": "Necesitamos entender si el agente debe responder o solo recomendar.",
        "current_user": "Equipo de operaciones",
        "current_process": "Analiza solicitudes entrantes y deriva casos manualmente.",
        "desired_outcome": "",
        "autonomy_level": "medium",
        "constraints": [],
        "operational_baseline": {
            "current_time_spent": "",
            "current_cost": "",
            "frequent_errors": [],
            "automation_opportunities": [],
        },
        "mvp_definition": {
            "v1_scope": [],
            "out_of_scope": [],
            "north_star_metric": "",
            "non_delegable_decisions": [],
        },
    }

    response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=partial_payload,
    )
    assert response.status_code == 200
    artifact = response.json()

    assert artifact["stage_key"] == "discover"
    assert artifact["source_action"] == "analyze_discovery"
    assert artifact["schema_version"] == "discovery-analysis.v1"
    assert artifact["proposal_payload"]["open_questions"]
    assert artifact["proposal_payload"]["normalized_discovery_candidate"]["problem_statement"] == partial_payload["problem_statement"]
    assert artifact["missing_information"]


def test_analyze_discovery_requires_approval_before_canvas(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert normalize_response.status_code == 200

    analyze_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert analyze_response.status_code == 200
    artifact = analyze_response.json()
    assert artifact["provider_key"] == "openai"
    assert artifact["evidence_manifest"]
    assert artifact["proposal_payload"]["facts"]

    blocked_canvas = client.post(f"/api/v1/sessions/{session_id}/build-canvas", headers=headers)
    assert blocked_canvas.status_code == 409
    assert blocked_canvas.json()["detail"] == "Discover must be approved before canvas"


def test_approving_discover_analysis_projects_candidate_and_unblocks_canvas(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert normalize_response.status_code == 200

    analyze_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert analyze_response.status_code == 200
    artifact = analyze_response.json()

    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Discover aprobado en CI5",
            "decision_payload": {
                "review_decisions": {
                    "fact:current_process:0": "accepted",
                    "question:knowledge_sources:0": "accepted",
                }
            },
        },
    )
    assert approve_response.status_code == 200
    approved_artifact = approve_response.json()
    assert approved_artifact["state"] == "approved"
    assert approved_artifact["proposal_payload"]["normalized_discovery_candidate"]["desired_outcome"] == complete_discovery_payload()["desired_outcome"]

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()
    assert snapshot["journey_latest_artifacts"]["discover"]["state"] == "approved"
    assert snapshot["discovery"]["desired_outcome"] == complete_discovery_payload()["desired_outcome"]

    canvas_response = client.post(f"/api/v1/sessions/{session_id}/build-canvas", headers=headers)
    assert canvas_response.status_code == 200
    assert canvas_response.json()["status"] == "ready"


def test_define_requirements_route_persists_definition_artifact_and_canvas_projection(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert normalize_response.status_code == 200

    analyze_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert analyze_response.status_code == 200
    discover_artifact = analyze_response.json()

    approve_discover = client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{discover_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Discover aprobado para CI6.",
            "decision_payload": {
                "approval_reason": "Discover listo para construir Definition.",
            },
        },
    )
    assert approve_discover.status_code == 200

    define_response = client.post(f"/api/v1/sessions/{session_id}/define-requirements", headers=headers)
    assert define_response.status_code == 200
    artifact = define_response.json()

    assert artifact["stage_key"] == "define"
    assert artifact["schema_version"] == "definition-artifact.v1"
    assert artifact["provider_key"] == "openai"
    assert artifact["proposal_payload"]["functional_requirements"]
    assert artifact["proposal_payload"]["canvas_projection"]["success_metric"] == complete_discovery_payload()["mvp_definition"]["north_star_metric"]
    assert any(item["source_id"] == "session://requirements-definition-input" for item in artifact["evidence_manifest"])


def test_define_requirements_requires_approval_before_blueprint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert normalize_response.status_code == 200

    analyze_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert analyze_response.status_code == 200
    discover_artifact = analyze_response.json()

    approve_discover = client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{discover_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Discover aprobado para blueprint gate.",
            "decision_payload": {
                "approval_reason": "Discover listo para Define.",
            },
        },
    )
    assert approve_discover.status_code == 200

    define_response = client.post(f"/api/v1/sessions/{session_id}/define-requirements", headers=headers)
    assert define_response.status_code == 200

    blocked_blueprint = client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers)
    assert blocked_blueprint.status_code == 409
    assert blocked_blueprint.json()["detail"] == "Define must be approved before blueprint"


def test_approving_define_projects_canvas_and_unblocks_blueprint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert normalize_response.status_code == 200

    analyze_response = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    assert analyze_response.status_code == 200
    discover_artifact = analyze_response.json()

    approve_discover = client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{discover_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Discover aprobado para Define approval test.",
            "decision_payload": {
                "approval_reason": "Discover listo para continuar.",
            },
        },
    )
    assert approve_discover.status_code == 200

    define_response = client.post(f"/api/v1/sessions/{session_id}/define-requirements", headers=headers)
    assert define_response.status_code == 200
    artifact = define_response.json()

    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/journey/define/artifacts/{artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Define aprobado en CI6",
            "decision_payload": {
                "approval_reason": "La definicion ya no tiene blockers."
            },
        },
    )
    assert approve_response.status_code == 200
    approved_artifact = approve_response.json()
    assert approved_artifact["state"] == "approved"

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()
    assert snapshot["journey_latest_artifacts"]["define"]["state"] == "approved"
    assert snapshot["canvas"]["success_metric"] == complete_discovery_payload()["mvp_definition"]["north_star_metric"]

    blueprint_response = client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers)
    assert blueprint_response.status_code == 200
    assert blueprint_response.json()["status"] in {"ready", "needs_review"}


def test_propose_design_route_persists_design_artifact_after_define_approval(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers = auth_headers(client)
    session_id = client.post("/api/v1/sessions", headers=headers).json()["id"]

    client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    discover_artifact = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    ).json()
    approve_discover = client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{discover_artifact['id']}/approve",
        headers=headers,
        json={"note": "Discover aprobado para CI7.", "decision_payload": {"approval_reason": "Listo para Design."}},
    )
    assert approve_discover.status_code == 200

    define_artifact = client.post(f"/api/v1/sessions/{session_id}/define-requirements", headers=headers).json()
    approve_define = client.post(
        f"/api/v1/sessions/{session_id}/journey/define/artifacts/{define_artifact['id']}/approve",
        headers=headers,
        json={"note": "Define aprobado para CI7.", "decision_payload": {"approval_reason": "Canvas trazable listo."}},
    )
    assert approve_define.status_code == 200

    response = client.post(
        f"/api/v1/sessions/{session_id}/propose-design",
        headers=headers,
        json={"instructions": "Prioriza la opcion mas simple posible."},
    )
    assert response.status_code == 200
    artifact = response.json()

    assert artifact["stage_key"] == "design"
    assert artifact["schema_version"] == "design-recommendation.v1"
    assert artifact["provider_key"] == "openai"
    assert artifact["proposal_payload"]["alternatives"]
    assert artifact["proposal_payload"]["recommended_alternative_key"]
    assert artifact["proposal_payload"]["critic_findings"]
    assert any(item["source_id"] == "session://agent-design-input" for item in artifact["evidence_manifest"])


def test_approving_design_projects_only_design_owned_fields_to_blueprint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers = auth_headers(client)
    session_id = client.post("/api/v1/sessions", headers=headers).json()["id"]

    client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    )
    discover_artifact = client.post(
        f"/api/v1/sessions/{session_id}/analyze-discovery",
        headers=headers,
        json=complete_discovery_payload(),
    ).json()
    client.post(
        f"/api/v1/sessions/{session_id}/journey/discover/artifacts/{discover_artifact['id']}/approve",
        headers=headers,
        json={"note": "Discover aprobado para CI7 projection.", "decision_payload": {"approval_reason": "OK"}},
    )
    define_artifact = client.post(f"/api/v1/sessions/{session_id}/define-requirements", headers=headers).json()
    client.post(
        f"/api/v1/sessions/{session_id}/journey/define/artifacts/{define_artifact['id']}/approve",
        headers=headers,
        json={"note": "Define aprobado para CI7 projection.", "decision_payload": {"approval_reason": "OK"}},
    )
    design_artifact = client.post(f"/api/v1/sessions/{session_id}/propose-design", headers=headers).json()

    approve_design = client.post(
        f"/api/v1/sessions/{session_id}/journey/design/artifacts/{design_artifact['id']}/approve",
        headers=headers,
        json={
            "note": "Design aprobado.",
            "decision_payload": {"selected_alternative_key": design_artifact["proposal_payload"]["recommended_alternative_key"]},
        },
    )
    assert approve_design.status_code == 200
    approved_design = approve_design.json()
    assert approved_design["state"] == "approved"

    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    selected_design = snapshot["journey_latest_artifacts"]["design"]["proposal_payload"]["selected_design"]
    assert snapshot["blueprint"]["architecture"] == selected_design["blueprint_projection"]["architecture"]
    assert snapshot["blueprint"]["reasoning_pattern"] == selected_design["blueprint_projection"]["reasoning_pattern"]
    assert snapshot["blueprint"]["memory_strategy"] == ""
    assert snapshot["blueprint"]["tools"] == []


def resolve_first_approval(client: TestClient, headers: dict[str, str], session_id: str) -> None:
    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    approval = snapshot["approvals"][0]

    resolve_response = client.post(
        f"/api/v1/sessions/{session_id}/approvals/{approval['id']}/resolve",
        headers=headers,
        json={
            "decision": "approved",
            "resolution_note": "Blueprint autorizado para handoff tecnico.",
        },
    )
    assert resolve_response.status_code == 200
    assert resolve_response.json()["status"] == "approved"


def test_session_flow_exposes_enriched_artifacts_and_pending_approvals(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    assert snapshot["discovery"]["operational_baseline"]["current_time_spent"]
    assert snapshot["discovery"]["mvp_definition"]["north_star_metric"]
    assert snapshot["canvas"]["agent_profile"]["mission"]
    assert snapshot["canvas"]["agent_profile"]["human_approvals"]
    assert snapshot["blueprint"]["delivery_package"]["workflow_profile"]["steps"]
    assert snapshot["blueprint"]["delivery_package"]["observability_plan"]["captured_signals"]
    assert snapshot["blueprint"]["delivery_package"]["decision_summary"]
    assert snapshot["blueprint"]["delivery_package"]["decision_trace"]
    assert snapshot["blueprint"]["delivery_package"]["pattern_catalog"]
    assert snapshot["blueprint"]["delivery_package"]["roadmap_evolution"]["milestones"]
    assert snapshot["blueprint"]["delivery_package"]["blueprint_coverage"]["total_sections"] >= 14
    assert any(item["key"] == "evolution_roadmap" for item in snapshot["blueprint"]["delivery_package"]["deliverables"])
    assert len(snapshot["blueprint"]["delivery_package"]["deliverables"]) >= 12
    assert snapshot["contract_version"] == "session-snapshot.v1"
    assert snapshot["estimation_report"] is None
    assert snapshot["workspace_contract"]["contract_version"] == "workspace-contract.v1"
    assert snapshot["workspace_contract"]["sections"]
    assert snapshot["workspace_contract"]["feature_flags"]
    assert snapshot["workspace_contract"]["catalogs"]
    assert any(item["catalog_key"] == "roadmap_templates" for item in snapshot["workspace_contract"]["catalogs"])
    assert any(item["catalog_key"] == "estimation_automation_matrix" for item in snapshot["workspace_contract"]["catalogs"])
    assert any(item["catalog_key"] == "estimation_pricing_profiles" for item in snapshot["workspace_contract"]["catalogs"])
    assert any(
        item["item_key"] == "ToT"
        for catalog in snapshot["workspace_contract"]["catalogs"]
        if catalog["catalog_key"] == "reasoning_patterns"
        for item in catalog["items"]
    )
    assert snapshot["skill_catalog"]
    assert any(item["skill_key"] == "blueprint_generation_skill" for item in snapshot["skill_catalog"])
    assert snapshot["skill_runs"]
    assert any(item["skill_key"] == "discovery_skill" for item in snapshot["skill_runs"])
    assert any(item["skill_key"] == "blueprint_generation_skill" for item in snapshot["skill_runs"])
    assert all(item["artifacts"] for item in snapshot["skill_runs"])
    assert snapshot["approvals"]
    assert snapshot["approvals"][0]["status"] == "pending"
    assert snapshot["session"]["status"] == "needs_review"

    evaluation_response = client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers)
    assert evaluation_response.status_code == 200
    evaluation = evaluation_response.json()

    assert evaluation["status"] == "needs_review"
    assert evaluation["next_action"] == "resolve_approvals"
    assert any("approval gates pendientes" in item.lower() for item in evaluation["data"]["gaps"])


def test_recommend_tools_route_persists_placeholder_contract_and_snapshot_view(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)

    response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)
    assert response.status_code == 200
    payload = response.json()

    assert payload["data"]["schema_version"] == "tool-recommendation.v1"
    assert payload["data"]["source_session_id"] == session_id
    assert payload["data"]["preflight"]["case_classification"]
    assert payload["data"]["preflight"]["candidate_tool_families"]
    assert payload["data"]["evaluation"]["summary"]
    assert payload["data"]["evaluation"]["promotion_blocked"] is False
    assert payload["status"] == "needs_review"
    assert payload["next_action"] == "review_tool_recommendation"

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    assert snapshot["latest_tool_recommendation"]["schema_version"] == "tool-recommendation.v1"
    assert snapshot["latest_tool_recommendation"]["source_blueprint_version"] >= 1
    assert snapshot["latest_tool_recommendation"]["preflight"]["mandatory_capabilities"]
    assert snapshot["latest_tool_recommendation"]["requirements_coverage"]
    assert "tools" in snapshot["journey_latest_artifacts"]
    assert snapshot["latest_tool_recommendation"]["evaluation"]["overall_status"] == "complete"
    assert any(item["artifact_kind"] == "tool_recommendation" for item in snapshot["artifact_records"])
    assert any(item["skill_key"] == "tool_recommendation_skill" for item in snapshot["skill_runs"])


def test_session_snapshot_accepts_persisted_react_skill_run_evidence(client: TestClient) -> None:
    headers = auth_headers(client)
    me_response = client.get("/api/v1/auth/me", headers=headers)
    assert me_response.status_code == 200
    workspace_id = me_response.json()["active_workspace_id"]
    assert workspace_id

    session_id = create_session_for_workspace(
        client,
        email=TEST_EMAIL,
        workspace_id=workspace_id,
        title="Validacion react runtime",
    )
    headers = {**headers, "x-workspace-id": workspace_id}

    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        record = SkillRunRecord(
            session_id=UUID(session_id),
            skill_key="react:tools:tool_recommendation",
            stage=SessionStage.build_blueprint,
            source_action="recommend_tools",
            status=ArtifactStatus.ready,
            duration_ms=12,
            result_summary="ReAct tools ejecutado.",
            warnings=[],
            evidence=[
                {
                    "source": "react_runtime",
                    "detail": "run=react-run-1; status=completed; iterations=1; checkpoint=react:tools:1",
                    "metadata": {
                        "contract_version": "builder.react.trace.v1",
                        "status": "completed",
                        "iterations": 1,
                        "checkpoint_id": "react:tools:1",
                        "actions": ["recommend_tools"],
                    },
                }
            ],
        )
        session.add(record)
        session.flush()
        session.add(
            SkillRunArtifactRecord(
                skill_run_id=record.id,
                artifact_role="react_trace",
                artifact_kind="builder.react.trace.v1",
                payload={
                    "run_id": "react-run-1",
                    "status": "completed",
                },
            )
        )
        session.commit()
    finally:
        session_generator.close()

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    react_run = next(item for item in snapshot["skill_runs"] if item["skill_key"] == "react:tools:tool_recommendation")
    assert react_run["evidence"][0]["source"] == "react_runtime"
    assert react_run["evidence"][0]["metadata"]["contract_version"] == "builder.react.trace.v1"
    assert react_run["evidence"][0]["metadata"]["actions"] == ["recommend_tools"]


def test_recommend_tools_route_respects_workspace_feature_flag(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)

    disable_response = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/tool_recommendation_llm_v1",
        headers=headers,
        json={"enabled": False},
    )
    assert disable_response.status_code == 200

    response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)
    assert response.status_code == 409
    assert "feature flag is disabled" in response.json()["detail"].lower()


def test_recommend_tools_route_uses_stage_artifacts_when_canonical_rows_are_missing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes.sessions as sessions_routes

    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    monkeypatch.setattr(sessions_routes, "run_tools_react", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("react-disabled-test")))

    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)
    delete_canonical_session_rows(client, session_id=session_id)

    response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["schema_version"] == "tool-recommendation.v1"
    assert payload["data"]["recommended_tools"]


def test_recommend_tools_route_attempts_react_even_when_workspace_flag_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes.sessions as sessions_routes

    react_attempted = {"value": False}
    session_id = UUID("11111111-1111-1111-1111-111111111111")

    def _fail_react(**kwargs):
        react_attempted["value"] = True
        raise RuntimeError("react-disabled-test")

    monkeypatch.setattr(sessions_routes, "run_tools_react", _fail_react)
    monkeypatch.setattr(
        sessions_routes,
        "run_tool_recommendation_stage",
        lambda *args, **kwargs: (
            ToolRecommendationEnvelope(
                status=ArtifactStatus.needs_review,
                stage=SessionStage.build_blueprint,
                data=ToolRecommendationArtifact(source_session_id=session_id),
                missing_fields=[],
                assumptions=[],
                warnings=[],
                evidence=[],
                llm_trace=None,
                next_action="review_tool_recommendation",
            ),
            [],
        ),
    )

    payload, _, _, warnings = sessions_routes._execute_tools_runtime(
        session_id=session_id,
        workspace_id=UUID("22222222-2222-2222-2222-222222222222"),
        discovery=None,
        canvas=None,
        blueprint=None,
        definition_artifact=None,
        design_artifact=None,
        instructions="",
        blueprint_version_number=1,
        runtime_settings=None,
        stage_context=None,
    )

    assert react_attempted["value"] is True
    assert "react_runtime_fallback:RuntimeError" in warnings
    assert payload.next_action == "review_tool_recommendation"


def test_approve_tools_selection_promotes_blueprint_tools_and_memory_uses_digest(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)

    recommend_response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)
    assert recommend_response.status_code == 200
    recommendation_payload = recommend_response.json()
    optional_keys = [item["tool_key"] for item in recommendation_payload["data"]["optional_tools"]]

    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/approve-tools-selection",
        headers=headers,
        json={"include_optional_tool_keys": optional_keys[:1]},
    )
    assert approve_response.status_code == 200
    snapshot = approve_response.json()

    approved_digest = snapshot["latest_tool_recommendation"]["approved_tools_digest"]
    assert approved_digest is not None
    assert approved_digest["tool_count"] == len(snapshot["blueprint"]["tools"])
    assert approved_digest["promoted_blueprint_version"] >= 1
    assert snapshot["latest_tool_recommendation"]["review_state"] == "complete"
    assert snapshot["blueprint"]["tools"]
    assert snapshot["blueprint_versions"]

    rerun_response = client.post(
        f"/api/v1/sessions/{session_id}/skills/memory_design_skill/rerun",
        headers=headers,
    )
    assert rerun_response.status_code == 200
    rerun_snapshot = rerun_response.json()["snapshot"]
    assert "approved_tools_digest" in rerun_snapshot["blueprint"]["memory_profile"]["retrieval_policy"]
    assert "tools aprobadas" in rerun_snapshot["blueprint"]["memory_profile"]["write_policy"]


def test_approve_tools_selection_uses_stage_artifact_when_schema_version_only_exists_in_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes.sessions as sessions_routes

    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    monkeypatch.setattr(sessions_routes, "run_tools_react", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("react-disabled-test")))

    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)

    recommend_response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)
    assert recommend_response.status_code == 200
    recommendation_payload = recommend_response.json()
    optional_keys = [item["tool_key"] for item in recommendation_payload["data"]["optional_tools"]]

    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        artifact = session.exec(
            select(JourneyStageArtifactRecord)
            .where(
                JourneyStageArtifactRecord.session_id == UUID(session_id),
                JourneyStageArtifactRecord.stage_key == "tools",
            )
            .order_by(JourneyStageArtifactRecord.version_number.desc(), JourneyStageArtifactRecord.created_at.desc())
        ).first()
        assert artifact is not None
        artifact.schema_version = ""
        session.add(artifact)
        session.commit()
    finally:
        session_generator.close()

    delete_canonical_session_rows(client, session_id=session_id)

    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/approve-tools-selection",
        headers=headers,
        json={"include_optional_tool_keys": optional_keys[:1]},
    )

    assert approve_response.status_code == 200
    snapshot = approve_response.json()
    assert snapshot["blueprint"]["tools"]
    assert snapshot["latest_tool_recommendation"]["approved_tools_digest"] is not None
    assert snapshot["latest_tool_recommendation"]["review_state"] == "complete"


def test_recommend_memory_route_persists_artifact_and_skill_runs(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.skill_runtime as skill_runtime

    original_builder_factory = skill_runtime._builder_service_for_stage
    monkeypatch.setattr(
        skill_runtime,
        "_builder_service_for_stage",
        lambda stage_key, runtime_settings=None: (
            FakeLLMTraceBuilderService()
            if stage_key == "memory"
            else original_builder_factory(stage_key, runtime_settings)
        ),
    )

    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)
    _, tools_snapshot = approve_tools_for_session(client, headers, session_id)

    response = client.post(
        f"/api/v1/sessions/{session_id}/recommend-memory",
        headers=headers,
        json={"instructions": "Priorizar memoria minima y solo herramientas ya aprobadas."},
    )
    assert response.status_code == 200
    artifact = response.json()

    assert artifact["stage_key"] == "memory"
    assert artifact["schema_version"] == "memory-recommendation.v1"
    assert artifact["source_action"] == "recommend_memory"
    proposal = artifact["proposal_payload"]
    assert proposal["schema_version"] == "memory-recommendation.v1"
    assert proposal["source_session_id"] == session_id
    assert proposal["source_stage_versions"]["tools"] == tools_snapshot["journey_latest_artifacts"]["tools"]["version_number"]
    assert isinstance(proposal["tool_dependencies"], list)
    assert proposal["dry_compile_status"]["summary"]

    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    assert snapshot["journey_latest_artifacts"]["memory"]["id"] == artifact["id"]
    skill_runs_by_key = {item["skill_key"]: item for item in snapshot["skill_runs"]}
    assert "memory_recommendation_skill" in skill_runs_by_key
    assert "memory_critique_skill" in skill_runs_by_key


def test_recommend_memory_route_attempts_react_even_when_workspace_flag_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes.sessions as sessions_routes

    react_attempted = {"value": False}
    session_id = UUID("33333333-3333-3333-3333-333333333333")

    def _fail_react(**kwargs):
        react_attempted["value"] = True
        raise RuntimeError("react-disabled-test")

    monkeypatch.setattr(sessions_routes, "run_memory_react", _fail_react)
    monkeypatch.setattr(
        sessions_routes,
        "run_memory_recommendation_stage",
        lambda **kwargs: (
            MemoryRecommendationArtifact(source_session_id=session_id),
            [],
        ),
    )

    payload, _, _, warnings = sessions_routes._execute_memory_runtime(
        session_id=session_id,
        workspace_id=UUID("44444444-4444-4444-4444-444444444444"),
        discovery=None,
        canvas=None,
        blueprint=None,
        definition_artifact=None,
        design_artifact=None,
        approved_tools_digest=None,
        tools_artifact=None,
        session_snapshot=None,
        instructions="",
        blueprint_version_number=1,
        source_stage_versions=None,
        runtime_settings=None,
        proposal_stage_context=None,
        critique_stage_context=None,
    )

    assert react_attempted["value"] is True
    assert "react_runtime_fallback:RuntimeError" in warnings
    assert payload.schema_version == "memory-recommendation.v1"


def test_approve_memory_profile_refreshes_user_edits_and_promotes_blueprint_sections(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.skill_runtime as skill_runtime

    original_builder_factory = skill_runtime._builder_service_for_stage
    monkeypatch.setattr(
        skill_runtime,
        "_builder_service_for_stage",
        lambda stage_key, runtime_settings=None: (
            FakeLLMTraceBuilderService()
            if stage_key == "memory"
            else original_builder_factory(stage_key, runtime_settings)
        ),
    )

    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)
    approve_tools_for_session(client, headers, session_id)

    recommendation = client.post(f"/api/v1/sessions/{session_id}/recommend-memory", headers=headers)
    assert recommendation.status_code == 200
    memory_artifact = recommendation.json()
    edited_payload = memory_artifact["proposal_payload"]
    edited_payload["summary"] = "Resumen revisado por el usuario antes de aprobar Memoria."
    edited_payload["open_questions"] = ["Confirmar retencion exacta durante implementacion."]
    edited_payload["critic_findings"] = [
        {
            "finding_key": "deferred-retention-owner",
            "title": "Owner de retencion pendiente",
            "detail": "La decision puede diferirse al ACP sin bloquear Blueprint Basico.",
            "severity": "warning",
            "category": "governance",
            "suggested_action": "Documentar como decision diferida.",
            "source_refs": ["memory.review"],
        }
    ]
    edited_payload["proposed_memory_profile"]["write_policy"] = (
        "Persistir decisiones aprobadas y resumenes compactos por 30 dias."
    )
    edited_payload["proposed_memory_profile"]["ttl_policy"] = "30_dias"

    patch_response = client.patch(
        f"/api/v1/sessions/{session_id}/journey/memory/artifacts/{memory_artifact['id']}",
        headers=headers,
        json={
            "note": "Ajuste manual del write policy antes de aprobar.",
            "proposal_payload": edited_payload,
        },
    )
    assert patch_response.status_code == 200

    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/approve-memory-profile",
        headers=headers,
        json={
            "note": "Memoria aprobada despues de la revision humana.",
            "decision_payload": {
                "approval_reason": "La propuesta final mantiene minimalidad y trazabilidad.",
            },
        },
    )
    assert approve_response.status_code == 200
    snapshot = approve_response.json()

    assert snapshot["blueprint"]["memory_profile"]["write_policy"] == (
        "Persistir decisiones aprobadas y resumenes compactos por 30 dias."
    )
    assert snapshot["blueprint"]["memory_profile"]["ttl_policy"] == "30_dias"
    assert snapshot["blueprint"]["memory_strategy"] == snapshot["blueprint"]["memory_profile"]["strategy"]
    assert snapshot["journey_latest_artifacts"]["memory"]["state"] == "approved"
    assert (
        snapshot["journey_latest_artifacts"]["memory"]["proposal_payload"]["proposed_memory_profile"]["write_policy"]
        == "Persistir decisiones aprobadas y resumenes compactos por 30 dias."
    )
    approved_memory_payload = snapshot["journey_latest_artifacts"]["memory"]["proposal_payload"]
    assert approved_memory_payload["summary"] == "Resumen revisado por el usuario antes de aprobar Memoria."
    assert approved_memory_payload["open_questions"] == ["Confirmar retencion exacta durante implementacion."]
    assert any(
        finding["finding_key"] == "deferred-retention-owner"
        for finding in approved_memory_payload["critic_findings"]
    )


def test_validate_simulation_flow_generates_runs_judgement_and_preserves_hard_fail_authority(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.skill_runtime as skill_runtime
    from app.models import ValidationSimulationRunStateRecord

    class ValidateJudgePassService(FakeLLMTraceBuilderService):
        def judge_validation_run(self, payload, *, context_bundle=None) -> LLMArtifactResult:
            del payload, context_bundle
            return LLMArtifactResult(
                artifact=ValidationRunJudgmentOutput(
                    scenario_key="happy_path_high_value",
                    judgment="pass",
                    summary="El judge considera que la corrida es aprobable.",
                    findings=[],
                    score=96,
                ),
                provider_key="openai",
                execution_backend="provider_native",
                execution_mode="primary",
                knowledge_access_backend="workspace_staged",
                effective_context_backend="workspace_staged_compact",
            )

    original_builder_factory = skill_runtime._builder_service_for_stage
    monkeypatch.setattr(
        skill_runtime,
        "_builder_service_for_stage",
        lambda stage_key, runtime_settings=None: (
            ValidateJudgePassService()
            if stage_key == "validate"
            else (
                FakeLLMTraceBuilderService()
                if stage_key == "memory"
                else original_builder_factory(stage_key, runtime_settings)
            )
        ),
    )

    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)
    approve_tools_for_session(client, headers, session_id)
    approve_memory_for_session(client, headers, session_id)

    generate_response = client.post(
        f"/api/v1/sessions/{session_id}/generate-validation-scenarios",
        headers=headers,
        json={"instructions": "Enfatiza recovery y no-evidence con trazabilidad visible."},
    )
    assert generate_response.status_code == 200
    validate_artifact = generate_response.json()
    assert validate_artifact["stage_key"] == "validate"
    assert validate_artifact["schema_version"] == "validation-simulation-spec.v1"
    assert len(validate_artifact["proposal_payload"]["scenarios"]) >= 3
    assert any(item["source_id"] == "session://validation-scenario-generation-input" for item in validate_artifact["evidence_manifest"])

    run_response = client.post(
        f"/api/v1/sessions/{session_id}/run-validation-simulation",
        headers=headers,
        json={
            "scenario_key": "happy_path_high_value",
            "initial_input_override": "Ejecuta el caso principal y muestra memoria, decision y gate.",
            "injected_conditions": [],
        },
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["scenario_key"] == "happy_path_high_value"
    assert run_payload["hard_gate_status"] == "pass"
    assert any(item["event_type"] == "memory_read" for item in run_payload["events"])
    assert any(item["event_type"] == "approval_gate" for item in run_payload["events"])

    injected_response = client.post(
        f"/api/v1/sessions/{session_id}/inject-validation-event",
        headers=headers,
        json={
            "run_id": run_payload["id"],
            "injection_type": "no_evidence",
            "note": "Forzar ausencia de evidencia para validar escalacion.",
        },
    )
    assert injected_response.status_code == 200
    injected_payload = injected_response.json()
    assert "no_evidence" in injected_payload["injected_conditions"]
    assert any(item["event_type"] == "fault_injected" for item in injected_payload["events"])

    judge_response = client.post(
        f"/api/v1/sessions/{session_id}/judge-validation-run",
        headers=headers,
        json={"run_id": injected_payload["id"]},
    )
    assert judge_response.status_code == 200
    judged_payload = judge_response.json()
    assert judged_payload["judgement"]["llm_judgment"] == "pass"
    assert judged_payload["final_status"] == "pass"

    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        persisted_run = session.exec(
            select(ValidationSimulationRunStateRecord).where(
                ValidationSimulationRunStateRecord.id == UUID(injected_payload["id"])
            )
        ).first()
        assert persisted_run is not None
        persisted_run.hard_gate_status = "fail"
        persisted_run.final_status = "fail"
        persisted_run.status = "needs_review"
        persisted_run.judgement = {
            "scenario_key": persisted_run.scenario_key,
            "hard_gate_status": "fail",
            "llm_judgment": "fail",
            "final_status": "fail",
            "score": 40,
            "summary": "Hard gate forzado a fail para validar autoridad determinista.",
            "hard_gate_findings": ["Se rompio intencionalmente el hard gate para la prueba."],
            "findings": [],
        }
        session.add(persisted_run)
        session.commit()
    finally:
        session_generator.close()

    hard_fail_judge_response = client.post(
        f"/api/v1/sessions/{session_id}/judge-validation-run",
        headers=headers,
        json={"run_id": injected_payload["id"]},
    )
    assert hard_fail_judge_response.status_code == 200
    hard_fail_payload = hard_fail_judge_response.json()
    assert hard_fail_payload["hard_gate_status"] == "fail"
    assert hard_fail_payload["judgement"]["llm_judgment"] == "pass"
    assert hard_fail_payload["final_status"] == "fail"

    approve_validate = client.post(
        f"/api/v1/sessions/{session_id}/approve-validation-scenarios",
        headers=headers,
        json={
            "note": "Validate aprobado despues de revisar corridas.",
            "decision_payload": {"approval_reason": "Escenarios y simulaciones listos para Estimate."},
        },
    )
    assert approve_validate.status_code == 200
    snapshot = approve_validate.json()
    assert snapshot["journey_latest_artifacts"]["validate"]["state"] == "approved"
    assert snapshot["simulation_runs"]


def test_tool_recommendation_stales_after_design_change_and_blocks_memory_rerun(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)

    recommend_response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)
    assert recommend_response.status_code == 200
    optional_keys = [item["tool_key"] for item in recommend_response.json()["data"]["optional_tools"]]

    approve_response = client.post(
        f"/api/v1/sessions/{session_id}/approve-tools-selection",
        headers=headers,
        json={"include_optional_tool_keys": optional_keys[:1]},
    )
    assert approve_response.status_code == 200

    patch_response = client.patch(
        f"/api/v1/sessions/{session_id}/blueprint",
        headers=headers,
        json={
            "guardrails": [
                "Toda escritura requiere aprobacion humana y audit trail",
                "Notificar al owner cuando cambie el estado de una solicitud",
            ],
        },
    )
    assert patch_response.status_code == 200

    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    assert snapshot["latest_tool_recommendation"]["is_stale"] is True
    assert "tool_recommendation_context_changed" in snapshot["latest_tool_recommendation"]["stale_reasons"]
    assert any(item["alert_key"] == "tool_recommendation_stale" for item in snapshot["alert_events"])

    rerun_response = client.post(
        f"/api/v1/sessions/{session_id}/skills/memory_design_skill/rerun",
        headers=headers,
    )
    assert rerun_response.status_code == 409
    assert "regenerating tools" in rerun_response.json()["detail"].lower()


def test_tool_recommendation_inherits_journey_stale_reason_after_reapproving_design(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)

    recommend_response = client.post(f"/api/v1/sessions/{session_id}/recommend-tools", headers=headers)
    assert recommend_response.status_code == 200

    approve_design_for_session(client, headers, session_id)

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    assert snapshot["journey_latest_artifacts"]["tools"]["state"] == "stale"
    assert any(
        reason.startswith("upstream_design_artifact_v")
        for reason in snapshot["journey_latest_artifacts"]["tools"]["stale_reasons"]
    )
    assert snapshot["latest_tool_recommendation"]["is_stale"] is True
    assert any(
        reason.startswith("upstream_design_artifact_v")
        for reason in snapshot["latest_tool_recommendation"]["stale_reasons"]
    )


def test_generate_estimation_report_persists_snapshot_and_artifact_registry(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert response.status_code == 200
    payload = response.json()

    assert payload["data"]["maturity_stage"] == "blueprint"
    assert payload["data"]["traditional"]["estimated_hours_total"] > 0
    assert payload["data"]["agentic"]["estimated_hours_total"] > 0
    assert payload["data"]["agentic"]["estimated_hours_total"] < payload["data"]["traditional"]["estimated_hours_total"]
    assert payload["data"]["agentic"]["automation_assessments"]
    assert payload["data"]["agentic"]["pricing_snapshot"] is not None
    assert payload["data"]["agentic"]["provider_model"]
    assert any(item["family_key"] == "tool_schemas" for item in payload["data"]["agentic"]["automation_assessments"])
    assert payload["data"]["confidence"]["score"] > 0
    assert payload["data"]["analysis"] is not None
    assert payload["data"]["base_confidence"] is not None
    assert payload["data"]["deterministic_inputs"]["pricing_catalog_signature"]
    assert payload["data"]["package_policy"]["can_continue_to_package"] is False
    assert payload["data"]["analysis_decision"]["decision"] == "pending"

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    assert snapshot["estimation_report"] is not None
    assert snapshot["estimation_runs"]
    assert snapshot["estimation_report"]["maturity_stage"] == "blueprint"
    assert snapshot["estimation_report"]["blueprint_version_number"] == snapshot["blueprint_versions"][0]["version_number"]
    assert snapshot["estimation_report"]["current_blueprint_version_number"] == snapshot["blueprint_versions"][0]["version_number"]
    assert snapshot["estimation_report"]["is_stale"] is False
    assert snapshot["estimation_report"]["stale_reasons"] == []
    assert snapshot["estimation_report"]["agentic"]["automation_assessments"]
    assert snapshot["estimation_report"]["agentic"]["pricing_snapshot"] is not None
    assert snapshot["estimation_report"]["analysis"]["scenario_adjustments"]
    assert snapshot["estimation_report"]["package_policy"]["package_block_reasons"]
    assert any(item["artifact_kind"] == "estimation_report" for item in snapshot["artifact_records"])


def test_estimation_analysis_decision_persists_without_mutating_historical_run(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers, session_id = build_session_flow(client)

    estimate_response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert estimate_response.status_code == 200
    estimate_payload = estimate_response.json()["data"]
    initial_score = estimate_payload["confidence"]["score"]

    snapshot_before = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    initial_run_id = snapshot_before["estimation_runs"][0]["id"]
    initial_run_score = snapshot_before["estimation_runs"][0]["confidence_score"]

    decision_response = client.post(
        f"/api/v1/sessions/{session_id}/estimate/analysis-decision",
        headers=headers,
        json={"decision": "accepted", "note": ""},
    )
    assert decision_response.status_code == 200
    decision_payload = decision_response.json()["data"]

    assert decision_payload["analysis_decision"]["decision"] == "accepted"
    assert decision_payload["confidence"]["score"] == initial_score - 4

    snapshot_after = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    assert snapshot_after["estimation_report"]["analysis_decision"]["decision"] == "accepted"
    assert snapshot_after["estimation_report"]["confidence"]["score"] == initial_score - 4
    assert snapshot_after["estimation_runs"][0]["id"] == initial_run_id
    assert snapshot_after["estimation_runs"][0]["confidence_score"] == initial_run_score


def test_estimation_snapshot_and_export_mark_stale_after_blueprint_changes(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    estimate_response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert estimate_response.status_code == 200
    initial_blueprint_version = estimate_response.json()["data"]["blueprint_version_number"]
    assert initial_blueprint_version is not None

    patch_response = client.patch(
        f"/api/v1/sessions/{session_id}/blueprint",
        headers=headers,
        json={
            "narrative": "Blueprint actualizado para incluir una politica de aprobacion adicional antes del package.",
        },
    )
    assert patch_response.status_code == 200

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    assert snapshot["estimation_report"] is not None
    assert snapshot["estimation_report"]["blueprint_version_number"] == initial_blueprint_version
    assert snapshot["estimation_report"]["current_blueprint_version_number"] == snapshot["blueprint_versions"][0]["version_number"]
    assert snapshot["estimation_report"]["current_blueprint_version_number"] > initial_blueprint_version
    assert snapshot["estimation_report"]["is_stale"] is True
    assert "blueprint_version_changed" in snapshot["estimation_report"]["stale_reasons"]

    preview_response = client.get(
        f"/api/v1/sessions/{session_id}/export/estimation-pack?preview=true",
        headers=headers,
    )
    assert preview_response.status_code == 200
    assert preview_response.headers["x-canonical-export-readiness"] == "needs_review"
    assert preview_response.headers["x-canonical-source-blueprint-version"] == str(initial_blueprint_version)
    estimation_pack = preview_response.json()
    assert estimation_pack["blueprint_ref"]["source_blueprint_version"] == initial_blueprint_version
    assert estimation_pack["source_blueprint_version"] == initial_blueprint_version


def test_estimation_snapshot_marks_stale_after_pricing_catalog_changes(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    estimate_response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert estimate_response.status_code == 200

    session_override = client.app.dependency_overrides[get_session]
    session_generator = session_override()
    session = next(session_generator)
    try:
        pricing_row = session.exec(
            select(RuntimeCatalogEntryRecord)
            .where(RuntimeCatalogEntryRecord.catalog_key == "estimation_pricing_profiles")
            .order_by(RuntimeCatalogEntryRecord.order_index.asc())
        ).first()
        assert pricing_row is not None
        updated_payload = dict(pricing_row.payload)
        updated_payload["model"] = f"{updated_payload.get('model', 'pricing-model')}-rev-july-23-2026"
        pricing_row.payload = updated_payload
        pricing_row.updated_at = utc_now()
        session.add(pricing_row)
        session.commit()
    finally:
        session_generator.close()

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    assert snapshot["estimation_report"] is not None
    assert snapshot["estimation_report"]["is_stale"] is True
    assert "pricing_catalog_changed" in snapshot["estimation_report"]["stale_reasons"]


def test_estimation_actuals_route_persists_metrics_and_updates_dashboard(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    estimate_response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert estimate_response.status_code == 200

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()
    estimation_run = snapshot["estimation_runs"][0]
    estimate_payload = estimate_response.json()["data"]

    actuals_response = client.post(
        f"/api/v1/sessions/{session_id}/estimate/actuals",
        headers=headers,
        json={
            "estimation_run_id": estimation_run["id"],
            "delivery_mode": "agentic",
            "actual_provider": "openai",
            "actual_hours_total": estimate_payload["agentic"]["estimated_hours_total"] * 1.07,
            "actual_duration_weeks": estimate_payload["agentic"]["estimated_duration_weeks"] * 1.04,
            "actual_cost_total": estimate_payload["agentic"]["estimated_cost"] * 1.05,
            "actual_automation_coverage_percent": max(
                0,
                estimate_payload["agentic"]["automation_coverage_percent"] - 5,
            ),
            "notes": "Proyecto piloto con pequenas desviaciones.",
        },
    )
    assert actuals_response.status_code == 200
    actuals_payload = actuals_response.json()
    assert actuals_payload["estimation_run_id"] == estimation_run["id"]
    assert actuals_payload["delivery_mode"] == "agentic"

    refreshed_snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    assert refreshed_snapshot["project_actuals"]
    assert refreshed_snapshot["estimation_error_metrics"]
    assert refreshed_snapshot["estimation_error_metrics"][0]["absolute_percentage_error_cost"] > 0

    dashboard_response = client.get("/api/v1/estimation/calibration", headers=headers)
    assert dashboard_response.status_code == 200
    dashboard = dashboard_response.json()

    assert dashboard["total_runs"] >= 1
    assert dashboard["calibrated_runs"] >= 1
    assert any(item["maturity_stage"] == "blueprint" for item in dashboard["precision_by_stage"])
    assert any(item["session_id"] == session_id for item in dashboard["recent_projects"])


def test_estimation_routes_respect_feature_flag_rollout(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    disable_response = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/estimation_comparative_v1",
        headers=headers,
        json={"enabled": False},
    )
    assert disable_response.status_code == 200

    estimate_response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert estimate_response.status_code == 409
    assert "feature flag is disabled" in estimate_response.json()["detail"].lower()

    dashboard_response = client.get("/api/v1/estimation/calibration", headers=headers)
    assert dashboard_response.status_code == 409
    assert "feature flag is disabled" in dashboard_response.json()["detail"].lower()

    upgrade_session_tier(client, headers, session_id)
    acp_response = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    assert acp_response.status_code == 200
    assert all(
        not item["path"].startswith("ACP/estimation/")
        for item in acp_response.json()["files"]
    )


def test_estimation_actuals_route_rejects_zero_metrics(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    estimate_response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert estimate_response.status_code == 200

    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    estimation_run = snapshot["estimation_runs"][0]

    actuals_response = client.post(
        f"/api/v1/sessions/{session_id}/estimate/actuals",
        headers=headers,
        json={
            "estimation_run_id": estimation_run["id"],
            "delivery_mode": "agentic",
            "actual_provider": "openai",
            "actual_hours_total": 0,
            "actual_duration_weeks": 6,
            "actual_cost_total": 25000000,
            "actual_automation_coverage_percent": 40,
            "notes": "Datos incompletos.",
        },
    )

    assert actuals_response.status_code == 422


def test_resolving_approval_unlocks_export_markdown(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    resolve_first_approval(client, headers, session_id)

    updated_snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert updated_snapshot_response.status_code == 200
    updated_snapshot = updated_snapshot_response.json()

    assert all(item["status"] == "approved" for item in updated_snapshot["approvals"])
    assert updated_snapshot["session"]["status"] == "ready"

    bootstrap_evaluation_response = client.post(
        f"/api/v1/sessions/{session_id}/evaluation/bootstrap",
        headers=headers,
    )
    assert bootstrap_evaluation_response.status_code == 200

    evaluation_response = client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers)
    assert evaluation_response.status_code == 200

    export_response = client.get(f"/api/v1/sessions/{session_id}/export/markdown", headers=headers)
    assert export_response.status_code == 200
    markdown = export_response.text

    assert "## Workspace Contract" in markdown
    assert "### Feature Flags" in markdown
    assert "### Catalog Summary" in markdown
    assert "### Operational Baseline" in markdown
    assert "### MVP Definition" in markdown
    assert "### Workflow Profile" in markdown
    assert "### Observability Plan" in markdown
    assert "### Blueprint Coverage" in markdown
    assert "### Roadmap Evolution" in markdown
    assert "### Decision Trace" in markdown
    assert "## Evaluation Workbench" in markdown
    assert "### Dataset activo" in markdown
    assert "### Rubrica activa" in markdown
    assert "### Corridas persistidas" in markdown
    assert "## Approval Gates" in markdown
    assert "## Skill Runtime" in markdown
    assert "PRD del agente" in markdown

    json_export_response = client.get(f"/api/v1/sessions/{session_id}/export/json", headers=headers)
    assert json_export_response.status_code == 200
    json_export = json_export_response.json()
    assert json_export["generated_at"]
    assert json_export["workspace_contract"]["sections"]
    assert json_export["skill_runs"]
    assert json_export["evaluation_dataset"]["cases"]
    assert json_export["evaluation_rubric"]["dimensions"]
    assert json_export["evaluation_runs"]
    assert json_export["blueprint"]["delivery_package"]["decision_summary"]
    assert json_export["blueprint"]["delivery_package"]["decision_trace"]
    assert json_export["blueprint"]["delivery_package"]["roadmap_evolution"]["milestones"]
    assert json_export["blueprint"]["delivery_package"]["blueprint_coverage"]["total_sections"] >= 14


def test_operational_snapshot_and_monitoring_workspace_expose_stage4_state(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    assert snapshot["artifact_records"]
    assert snapshot["metric_snapshots"]
    assert snapshot["integration_statuses"]
    assert any(item["alert_key"] == "approvals_unresolved" for item in snapshot["alert_events"])

    monitoring_response = client.get(f"/api/v1/sessions/{session_id}/monitoring", headers=headers)
    assert monitoring_response.status_code == 200
    monitoring = monitoring_response.json()

    assert monitoring["current_metrics"]["artifact_count"] >= 1
    assert monitoring["current_metrics"]["approvals_pending"] >= 1
    assert monitoring["history"]
    assert monitoring["recent_errors"]
    assert any(item["alert_key"] == "approvals_unresolved" for item in monitoring["alerts"])
    assert {"openai", "postgresql", "local_auth"}.issubset(
        {item["integration_key"] for item in monitoring["integrations"]}
    )
    assert monitoring["memory_observability"] is not None
    assert monitoring["memory_observability"]["metrics"]
    assert monitoring["memory_observability"]["by_stage"]
    assert monitoring["memory_observability"]["validations"]


def test_monitoring_workspace_exposes_memory_observability_from_llm_traces(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.skill_runtime._builder_service_for_stage",
        lambda stage_key, runtime_settings=None: FakeLLMTraceBuilderService(),
    )
    headers, session_id = build_session_flow(client)

    monitoring_response = client.get(f"/api/v1/sessions/{session_id}/monitoring", headers=headers)
    assert monitoring_response.status_code == 200
    monitoring = monitoring_response.json()

    assert monitoring["memory_observability"]["llm_run_count"] >= 3
    metrics_by_key = {item["key"]: item for item in monitoring["memory_observability"]["metrics"]}
    validations_by_key = {item["check_key"]: item for item in monitoring["memory_observability"]["validations"]}
    stages_by_key = {
        item["scope_key"]: item for item in monitoring["memory_observability"]["by_stage"]
    }

    assert metrics_by_key["hit_rate"]["value"] == 100.0
    assert metrics_by_key["citation_coverage"]["value"] == 100.0
    assert metrics_by_key["recoverability"]["value"] >= 80.0
    assert validations_by_key["needle_in_the_haystack_recovery"]["status"] == "pass"
    assert {"define", "design", "tools", "memory", "evaluate", "build"}.issubset(
        stages_by_key
    )
    assert stages_by_key["tools"]["llm_runs"] == 0
    assert stages_by_key["build"]["llm_runs"] == 0
    assert monitoring["release_observability"] is not None
    assert monitoring["release_observability"]["total_llm_runs"] >= 3
    assert monitoring["release_observability"]["context_fingerprint_coverage"] == 100.0
    assert monitoring["release_observability"]["source_version_coverage"] == 100.0
    assert monitoring["release_observability"]["providers"]
    assert {
        item["provider_key"] for item in monitoring["release_observability"]["providers"]
    } >= {"openai", "deepseek", "codex_local"}
    release_gates_by_key = {
        item["gate_key"]: item for item in monitoring["release_observability"]["release_gates"]
    }
    assert release_gates_by_key["context_fingerprint_coverage"]["status"] == "pass"
    assert release_gates_by_key["hard_gate_authority"]["status"] == "pass"


def test_artifact_browser_library_filters_and_exports_persist_records(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    artifacts_response = client.get(f"/api/v1/sessions/{session_id}/artifacts", headers=headers)
    assert artifacts_response.status_code == 200
    artifacts = artifacts_response.json()["items"]

    assert any(item["artifact_key"] == "prd" for item in artifacts)
    assert any(item["artifact_kind"] == "delivery_package" for item in artifacts)

    library_response = client.get(
        f"/api/v1/sessions/{session_id}/library?q=prd&artifact_kind=delivery_package&stage=build_blueprint",
        headers=headers,
    )
    assert library_response.status_code == 200
    library_items = library_response.json()["items"]
    assert library_items
    assert any(item["artifact_key"] == "prd" for item in library_items)

    resolve_first_approval(client, headers, session_id)

    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    assert bootstrap_response.status_code == 200

    evaluation_response = client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers)
    assert evaluation_response.status_code == 200

    markdown_export_response = client.get(f"/api/v1/sessions/{session_id}/export/markdown", headers=headers)
    assert markdown_export_response.status_code == 200

    json_export_response = client.get(f"/api/v1/sessions/{session_id}/export/json", headers=headers)
    assert json_export_response.status_code == 200

    refreshed_artifacts_response = client.get(f"/api/v1/sessions/{session_id}/artifacts", headers=headers)
    assert refreshed_artifacts_response.status_code == 200
    refreshed_artifacts = refreshed_artifacts_response.json()["items"]

    export_records = [item for item in refreshed_artifacts if item["artifact_kind"] == "export"]
    export_keys = {item["artifact_key"] for item in export_records}
    assert {"markdown_export", "json_export"}.issubset(export_keys)

    export_library_response = client.get(
        f"/api/v1/sessions/{session_id}/library?artifact_kind=export&q=markdown_export",
        headers=headers,
    )
    assert export_library_response.status_code == 200
    export_library_items = export_library_response.json()["items"]
    assert export_library_items
    assert any(item["artifact_key"] == "markdown_export" for item in export_library_items)


def test_integrations_routes_refresh_health_and_append_operational_trace(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    initial_snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    initial_metric_count = len(initial_snapshot["metric_snapshots"])

    integrations_response = client.get(f"/api/v1/sessions/{session_id}/integrations", headers=headers)
    assert integrations_response.status_code == 200
    integrations = integrations_response.json()
    assert {"openai", "deepseek", "postgresql", "local_auth"}.issubset(
        {item["integration_key"] for item in integrations}
    )

    check_response = client.post(f"/api/v1/sessions/{session_id}/integrations/check", headers=headers)
    assert check_response.status_code == 200
    checked_snapshot = check_response.json()

    assert len(checked_snapshot["metric_snapshots"]) >= initial_metric_count + 1
    assert checked_snapshot["metric_snapshots"][0]["source_action"] == "check_integrations"
    assert {"openai", "deepseek", "postgresql", "local_auth"}.issubset(
        {item["integration_key"] for item in checked_snapshot["integration_statuses"]}
    )
    assert any(item["message"] == "Integraciones verificadas" for item in checked_snapshot["activity"])


def test_runtime_llm_settings_can_switch_active_provider_without_breaking_health(client: TestClient) -> None:
    headers = auth_headers(client)

    initial_response = client.get("/api/v1/runtime/llm", headers=headers)
    assert initial_response.status_code == 200
    initial_payload = initial_response.json()
    assert initial_payload["active_provider"] == "openai"
    assert initial_payload["memory_rollout"]["manifest_ready"] is True
    assert initial_payload["memory_rollout"]["effective_default_backend"] == "workspace_staged"
    assert {"define", "design", "tools", "memory", "evaluate", "build"} == {
        item["stage_key"] for item in initial_payload["memory_rollout"]["stages"]
    }
    assert {"openai", "deepseek", "codex_local", "antigravity_cli"} == {
        item["key"] for item in initial_payload["provider_options"]
    }

    update_response = client.patch(
        "/api/v1/runtime/llm",
        headers=headers,
        json={
            "active_provider": "deepseek",
            "agent_execution_backend": "codex_cli",
            "openai": {
                "fast_model": "gpt-5.4-mini",
                "reasoning_model": "gpt-5.5",
                "reasoning_effort": "low",
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "fast_model": "deepseek-v4-flash",
                "reasoning_model": "deepseek-v4-pro",
                "reasoning_effort": "max",
            },
            "codex_local": {
                "command": "codex",
                "model": "gpt-5.5",
                "profile": "deep-review",
                "cost_policy": "fully_loaded",
                "timeout_ms": 180000,
                "max_concurrency": 3,
                "runner_id": "local-shadow",
                "auth_mode": "chatgpt_session",
                "fallback_models": ["gpt-5.5-mini", "gpt-5.4-mini"],
                "primary_agents": ["normalize_discovery", "build_canvas"],
                "shadow_agents": ["synthesize_blueprint_narrative"],
                "staged_agents": ["evaluate_readiness"],
            },
            "knowledge_access_backend": "workspace_staged",
        },
    )
    assert update_response.status_code == 200
    updated_payload = update_response.json()
    assert updated_payload["active_provider"] == "deepseek"
    assert updated_payload["agent_execution_backend"] == "codex_cli"
    assert updated_payload["deepseek"]["base_url"] == "https://api.deepseek.com"
    assert updated_payload["deepseek"]["fast_model"] == "deepseek-v4-flash"
    assert updated_payload["deepseek"]["reasoning_model"] == "deepseek-v4-pro"
    assert updated_payload["deepseek"]["reasoning_effort"] == "max"
    assert updated_payload["codex_local"]["cost_policy"] == "fully_loaded"
    assert updated_payload["codex_local"]["timeout_ms"] == 180000
    assert updated_payload["codex_local"]["max_concurrency"] == 3
    assert updated_payload["codex_local"]["runner_id"] == "local-shadow"
    assert updated_payload["codex_local"]["auth_mode"] == "chatgpt_session"
    assert updated_payload["codex_local"]["fallback_models"] == ["gpt-5.5-mini", "gpt-5.4-mini"]
    assert updated_payload["knowledge_access_backend"] == "workspace_staged"
    assert updated_payload["memory_rollout"]["status"] == "ready"
    assert all(item["enabled"] for item in updated_payload["memory_rollout"]["phases"])

    health_response = client.get("/health")
    assert health_response.status_code == 200
    health_payload = health_response.json()
    assert health_payload["status"] == "ok"
    assert health_payload["runtime"]["scope"] == "platform_default"
    assert "workspace" in health_payload["runtime"]["scope_detail"]
    assert health_payload["runtime"]["active_provider"] in {"openai", "deepseek", "codex_local"}
    assert "llm_runtime_settings" not in health_payload
    assert "command" not in json.dumps(health_payload)
    assert "fallback_models" not in json.dumps(health_payload)

    workspace_health_response = client.get("/api/v1/runtime/llm/health", headers=headers)
    assert workspace_health_response.status_code == 200
    workspace_health_payload = workspace_health_response.json()
    assert workspace_health_payload["provider_key"] == "deepseek"
    assert workspace_health_payload["agent_execution_backend"] == "codex_cli"
    assert workspace_health_payload["knowledge_access_backend"] == "workspace_staged"

    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]
    integrations_response = client.get(f"/api/v1/sessions/{session_id}/integrations", headers=headers)
    assert integrations_response.status_code == 200
    integration_keys = {item["integration_key"] for item in integrations_response.json()}
    assert {"llm_runtime", "openai", "deepseek", "codex_local", "postgresql", "local_auth"}.issubset(
        integration_keys
    )


def test_runtime_llm_settings_are_scoped_by_workspace(client: TestClient) -> None:
    headers = auth_headers(client)
    me_response = client.get("/api/v1/auth/me", headers=headers)
    assert me_response.status_code == 200
    default_workspace_id = me_response.json()["active_workspace_id"]
    assert default_workspace_id

    secondary_workspace_id = create_workspace_for_user(
        client,
        email=TEST_EMAIL,
        name="Runtime Isolation Workspace",
        role=WorkspaceRole.admin,
    )

    default_headers = {**headers, "x-workspace-id": default_workspace_id}
    secondary_headers = {**headers, "x-workspace-id": secondary_workspace_id}

    default_initial = client.get("/api/v1/runtime/llm", headers=default_headers)
    assert default_initial.status_code == 200
    assert default_initial.json()["active_provider"] == "openai"

    update_response = client.patch(
        "/api/v1/runtime/llm",
        headers=default_headers,
        json={
            "active_provider": "deepseek",
            "agent_execution_backend": "provider_native",
            "openai": {
                "fast_model": "gpt-5.4-mini",
                "reasoning_model": "gpt-5.5",
                "reasoning_effort": "low",
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "fast_model": "deepseek-v4-flash",
                "reasoning_model": "deepseek-v4-pro",
                "reasoning_effort": "high",
            },
            "codex_local": {
                "command": "codex",
                "model": "gpt-5.5",
                "profile": "isolated-profile",
                "cost_policy": "hybrid",
                "timeout_ms": 150000,
                "max_concurrency": 1,
                "runner_id": "isolated-runner",
                "auth_mode": "auto",
                "fallback_models": [],
                "primary_agents": [],
                "shadow_agents": [],
                "staged_agents": [],
            },
            "knowledge_access_backend": "workspace_staged",
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["active_provider"] == "deepseek"

    default_after = client.get("/api/v1/runtime/llm", headers=default_headers)
    assert default_after.status_code == 200
    assert default_after.json()["active_provider"] == "deepseek"
    assert default_after.json()["codex_local"]["profile"] == "isolated-profile"

    secondary_after = client.get("/api/v1/runtime/llm", headers=secondary_headers)
    assert secondary_after.status_code == 200
    assert secondary_after.json()["active_provider"] == "openai"
    assert secondary_after.json()["codex_local"]["profile"] != "isolated-profile"


def test_monitoring_and_estimation_routes_use_session_workspace_runtime_after_workspace_switch(
    client: TestClient,
) -> None:
    headers = auth_headers(client)
    current_user = client.get("/api/v1/auth/me", headers=headers).json()
    default_workspace_id = current_user["active_workspace_id"]
    secondary_workspace_id = create_workspace_for_user(
        client,
        email=TEST_EMAIL,
        name="Runtime Monitoring Isolation",
        role=WorkspaceRole.admin,
    )

    default_headers = {**headers, "x-workspace-id": default_workspace_id}
    secondary_headers = {**headers, "x-workspace-id": secondary_workspace_id}
    patch_workspace_runtime(client, default_headers, active_provider="deepseek", runner_id="workspace-a")
    patch_workspace_runtime(client, secondary_headers, active_provider="codex_local", runner_id="workspace-b")

    session_id = build_session_flow_for_headers(client, default_headers)
    switch_response = client.post(
        "/api/v1/auth/workspaces/select",
        headers=headers,
        json={"workspace_id": secondary_workspace_id},
    )
    assert switch_response.status_code == 200
    assert switch_response.json()["active_workspace_id"] == secondary_workspace_id

    integrations_response = client.get(f"/api/v1/sessions/{session_id}/integrations", headers=headers)
    assert integrations_response.status_code == 200
    integrations_by_key = {item["integration_key"]: item for item in integrations_response.json()}
    assert integrations_by_key["llm_runtime"]["label"] == "LLM activo (deepseek)"
    assert "provider=deepseek" in integrations_by_key["llm_runtime"]["detail"]

    monitoring_response = client.get(f"/api/v1/sessions/{session_id}/monitoring", headers=headers)
    assert monitoring_response.status_code == 200
    monitoring_integrations = {
        item["integration_key"]: item for item in monitoring_response.json()["integrations"]
    }
    assert monitoring_integrations["llm_runtime"]["label"] == "LLM activo (deepseek)"

    estimate_response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert estimate_response.status_code == 200
    assert estimate_response.json()["data"]["agentic"]["active_provider"] == "deepseek"


def test_runtime_secret_routes_redact_plaintext_and_isolate_workspaces(client: TestClient) -> None:
    headers = auth_headers(client)
    me_response = client.get("/api/v1/auth/me", headers=headers)
    assert me_response.status_code == 200
    default_workspace_id = me_response.json()["active_workspace_id"]
    assert default_workspace_id

    secondary_workspace_id = create_workspace_for_user(
        client,
        email=TEST_EMAIL,
        name="Runtime Secrets Workspace",
        role=WorkspaceRole.admin,
    )

    default_headers = {**headers, "x-workspace-id": default_workspace_id}
    secondary_headers = {**headers, "x-workspace-id": secondary_workspace_id}

    create_secret_response = client.post(
        "/api/v1/runtime/llm/secrets/openai",
        headers=default_headers,
        json={
            "secret_value": "sk-runtime-workspace-alpha",
            "activate_for_runtime": True,
        },
    )
    assert create_secret_response.status_code == 200
    create_secret_payload = create_secret_response.json()
    assert "secret_value" not in json.dumps(create_secret_payload)
    assert create_secret_payload["secret_source"] == "workspace_managed"
    assert create_secret_payload["configured"] is True
    assert create_secret_payload["uses_platform_credentials"] is False
    assert create_secret_payload["storage_mode"] == "ciphertext"

    default_runtime_response = client.get("/api/v1/runtime/llm", headers=default_headers)
    assert default_runtime_response.status_code == 200
    default_runtime = default_runtime_response.json()
    assert default_runtime["uses_platform_credentials"] is False
    assert default_runtime["openai"]["secret_source"] == "workspace_managed"
    assert default_runtime["openai"]["api_key_configured"] is True
    assert default_runtime["openai"]["health_status"] == "workspace_ready"
    assert "sk-runtime-workspace-alpha" not in json.dumps(default_runtime)

    secondary_runtime_response = client.get("/api/v1/runtime/llm", headers=secondary_headers)
    assert secondary_runtime_response.status_code == 200
    secondary_runtime = secondary_runtime_response.json()
    assert secondary_runtime["uses_platform_credentials"] is True
    assert secondary_runtime["openai"]["secret_source"] == "platform_managed"
    assert secondary_runtime["openai"]["health_status"].startswith("platform_")

    rotate_secret_response = client.post(
        "/api/v1/runtime/llm/secrets/openai/rotate",
        headers=default_headers,
        json={
            "secret_ref": "vault://runtime/openai/default",
            "activate_for_runtime": True,
        },
    )
    assert rotate_secret_response.status_code == 200
    rotate_secret_payload = rotate_secret_response.json()
    assert rotate_secret_payload["storage_mode"] == "reference"
    assert rotate_secret_payload["last_rotated_at"]

    delete_secret_response = client.delete(
        "/api/v1/runtime/llm/secrets/openai",
        headers=default_headers,
    )
    assert delete_secret_response.status_code == 200
    delete_secret_payload = delete_secret_response.json()
    assert delete_secret_payload["secret_source"] == "platform_managed"
    assert delete_secret_payload["uses_platform_credentials"] is True
    assert delete_secret_payload["health_status"].startswith("platform_")

    runtime_after_delete = client.get("/api/v1/runtime/llm", headers=default_headers)
    assert runtime_after_delete.status_code == 200
    assert runtime_after_delete.json()["uses_platform_credentials"] is True
    assert runtime_after_delete.json()["openai"]["secret_source"] == "platform_managed"


def test_workspace_runtime_routes_require_platform_admin_for_non_platform_users(client: TestClient) -> None:
    seed_user(
        client,
        email="viewer@leanbuilder.local",
        password="Viewer123!",
        full_name="Viewer Runtime",
    )
    viewer_workspace_id = create_workspace_for_user(
        client,
        email="viewer@leanbuilder.local",
        name="Viewer Runtime Workspace",
        role=WorkspaceRole.viewer,
    )
    viewer_headers = {
        **auth_headers_for_credentials(
            client,
            email="viewer@leanbuilder.local",
            password="Viewer123!",
        ),
        "x-workspace-id": viewer_workspace_id,
    }

    read_response = client.get("/api/v1/runtime/llm", headers=viewer_headers)
    assert read_response.status_code == 403

    update_response = client.patch(
        "/api/v1/runtime/llm",
        headers=viewer_headers,
        json={
            "active_provider": "openai",
            "agent_execution_backend": "provider_native",
            "knowledge_access_backend": "workspace_staged",
            "openai": {
                "fast_model": "gpt-5.4-mini",
                "reasoning_model": "gpt-5.5",
                "reasoning_effort": "low",
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "fast_model": "deepseek-v4-flash",
                "reasoning_model": "deepseek-v4-pro",
                "reasoning_effort": "high",
            },
            "codex_local": {
                "command": "codex",
                "model": "gpt-5.5",
                "profile": "",
                "cost_policy": "hybrid",
                "timeout_ms": 150000,
                "max_concurrency": 1,
                "runner_id": "viewer",
                "auth_mode": "auto",
                "fallback_models": [],
                "primary_agents": [],
                "shadow_agents": [],
                "staged_agents": [],
            },
        },
    )
    assert update_response.status_code == 403

    health_response = client.get("/api/v1/runtime/llm/health", headers=viewer_headers)
    assert health_response.status_code == 403

    test_response = client.post("/api/v1/runtime/llm/test", headers=viewer_headers)
    assert test_response.status_code == 403


def test_workspace_runtime_routes_forbid_workspace_admin_without_platform_role(client: TestClient) -> None:
    seed_user(
        client,
        email="workspace-admin@leanbuilder.local",
        password="WorkspaceAdmin123!",
        full_name="Workspace Admin Runtime",
    )
    admin_workspace_id = create_workspace_for_user(
        client,
        email="workspace-admin@leanbuilder.local",
        name="Workspace Admin Runtime",
        role=WorkspaceRole.admin,
    )
    admin_headers = {
        **auth_headers_for_credentials(
            client,
            email="workspace-admin@leanbuilder.local",
            password="WorkspaceAdmin123!",
        ),
        "x-workspace-id": admin_workspace_id,
    }

    update_response = client.patch(
        "/api/v1/runtime/llm",
        headers=admin_headers,
        json={
            "active_provider": "openai",
            "agent_execution_backend": "provider_native",
            "knowledge_access_backend": "workspace_staged",
            "openai": {
                "fast_model": "gpt-5.4-mini",
                "reasoning_model": "gpt-5.5",
                "reasoning_effort": "low",
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "fast_model": "deepseek-v4-flash",
                "reasoning_model": "deepseek-v4-pro",
                "reasoning_effort": "high",
            },
            "codex_local": {
                "command": "codex",
                "model": "gpt-5.5",
                "profile": "",
                "cost_policy": "hybrid",
                "timeout_ms": 150000,
                "max_concurrency": 1,
                "runner_id": "workspace-admin",
                "auth_mode": "auto",
                "fallback_models": [],
                "primary_agents": [],
                "shadow_agents": [],
                "staged_agents": [],
            },
        },
    )
    assert update_response.status_code == 403

    health_response = client.get("/api/v1/runtime/llm/health", headers=admin_headers)
    assert health_response.status_code == 403

    test_response = client.post("/api/v1/runtime/llm/test", headers=admin_headers)
    assert test_response.status_code == 403


def test_platform_runtime_routes_require_platform_admin(client: TestClient) -> None:
    seed_user(
        client,
        email="runtime-reviewer@leanbuilder.local",
        password="RuntimeReviewer123!",
        full_name="Runtime Reviewer",
    )
    reviewer_headers = auth_headers_for_credentials(
        client,
        email="runtime-reviewer@leanbuilder.local",
        password="RuntimeReviewer123!",
    )

    assert client.get("/api/v1/platform/runtime/providers", headers=reviewer_headers).status_code == 403
    assert client.get("/api/v1/platform/runtime/defaults", headers=reviewer_headers).status_code == 403
    assert client.get("/api/v1/platform/runtime/audit", headers=reviewer_headers).status_code == 403
    assert client.get("/api/v1/runtime/status", headers=reviewer_headers).status_code == 403


def test_platform_runtime_routes_allow_platform_admin_and_govern_workspace_runtime(client: TestClient) -> None:
    headers = auth_headers(client)

    providers_response = client.get("/api/v1/platform/runtime/providers", headers=headers)
    assert providers_response.status_code == 200
    provider_keys = {item["provider_key"] for item in providers_response.json()}
    assert provider_keys == {"openai", "deepseek", "codex_local", "antigravity_cli"}

    update_provider_response = client.patch(
        "/api/v1/platform/runtime/providers/deepseek",
        headers=headers,
        json={"is_enabled": False},
    )
    assert update_provider_response.status_code == 200
    assert update_provider_response.json()["provider_key"] == "deepseek"
    assert update_provider_response.json()["is_enabled"] is False

    defaults_response = client.get("/api/v1/platform/runtime/defaults", headers=headers)
    assert defaults_response.status_code == 200
    assert defaults_response.json()["active_provider"] in {"openai", "deepseek", "codex_local"}

    update_defaults_response = client.patch(
        "/api/v1/platform/runtime/defaults",
        headers=headers,
        json={
            "active_provider": "openai",
            "agent_execution_backend": "provider_native",
            "knowledge_access_backend": "workspace_staged",
            "openai": {
                "fast_model": "gpt-5.4-mini",
                "reasoning_model": "gpt-5.5",
                "reasoning_effort": "low",
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "fast_model": "deepseek-v4-flash",
                "reasoning_model": "deepseek-v4-pro",
                "reasoning_effort": "high",
            },
            "codex_local": {
                "command": "codex",
                "model": "gpt-5.5",
                "profile": "platform-default",
                "cost_policy": "hybrid",
                "timeout_ms": 150000,
                "max_concurrency": 1,
                "runner_id": "platform-default",
                "auth_mode": "auto",
                "fallback_models": [],
                "primary_agents": [],
                "shadow_agents": [],
                "staged_agents": [],
            },
        },
    )
    assert update_defaults_response.status_code == 200
    assert update_defaults_response.json()["active_provider"] == "openai"

    audit_response = client.get("/api/v1/platform/runtime/audit", headers=headers)
    assert audit_response.status_code == 200
    audit_change_types = {item["change_type"] for item in audit_response.json()["items"]}
    assert "platform_runtime_provider_updated" in audit_change_types
    assert "platform_runtime_defaults_updated" in audit_change_types

    secondary_workspace_id = create_workspace_for_user(
        client,
        email=TEST_EMAIL,
        name="Governed Workspace",
        role=WorkspaceRole.admin,
    )
    workspace_headers = {**headers, "x-workspace-id": secondary_workspace_id}
    blocked_runtime_response = client.patch(
        "/api/v1/runtime/llm",
        headers=workspace_headers,
        json={
            "active_provider": "deepseek",
            "agent_execution_backend": "provider_native",
            "knowledge_access_backend": "workspace_staged",
            "openai": {
                "fast_model": "gpt-5.4-mini",
                "reasoning_model": "gpt-5.5",
                "reasoning_effort": "low",
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "fast_model": "deepseek-v4-flash",
                "reasoning_model": "deepseek-v4-pro",
                "reasoning_effort": "high",
            },
            "codex_local": {
                "command": "codex",
                "model": "gpt-5.5",
                "profile": "",
                "cost_policy": "hybrid",
                "timeout_ms": 150000,
                "max_concurrency": 1,
                "runner_id": "blocked-provider",
                "auth_mode": "auto",
                "fallback_models": [],
                "primary_agents": [],
                "shadow_agents": [],
                "staged_agents": [],
            },
        },
    )
    assert blocked_runtime_response.status_code == 400
    assert "deshabilitado" in blocked_runtime_response.json()["detail"]


def test_runtime_status_route_requires_platform_admin(client: TestClient) -> None:
    seed_user(
        client,
        email="runtime-observer@leanbuilder.local",
        password="RuntimeObserver123!",
        full_name="Runtime Observer",
    )
    observer_headers = auth_headers_for_credentials(
        client,
        email="runtime-observer@leanbuilder.local",
        password="RuntimeObserver123!",
    )
    response = client.get("/api/v1/runtime/status", headers=observer_headers)
    assert response.status_code == 403


def test_runtime_status_route_exposes_governed_codex_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = auth_headers(client)
    settings = get_settings()
    runtime_base = settings.llm_config_path.parent
    fake_executable = runtime_base / "fake-codex.cmd"
    fake_executable.write_text(
        "@echo off\r\nif \"%1\"==\"--version\" echo codex-cli 0.0-test\r\n",
        encoding="utf-8",
    )
    codex_home = runtime_base / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text('{"mode":"session"}', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    update_response = client.patch(
        "/api/v1/runtime/llm",
        headers=headers,
        json={
            "active_provider": "openai",
            "agent_execution_backend": "codex_cli",
            "openai": {
                "fast_model": "gpt-5.4-mini",
                "reasoning_model": "gpt-5.5",
                "reasoning_effort": "low",
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "fast_model": "deepseek-v4-flash",
                "reasoning_model": "deepseek-v4-pro",
                "reasoning_effort": "high",
            },
            "codex_local": {
                "command": str(fake_executable),
                "model": "gpt-5.5",
                "profile": "",
                "cost_policy": "hybrid",
                "timeout_ms": 210000,
                "max_concurrency": 2,
                "runner_id": "runtime-status-test",
                "auth_mode": "auto",
                "fallback_models": ["gpt-5.5-mini", "gpt-5.4-mini"],
                "primary_agents": [],
                "shadow_agents": [],
                "staged_agents": [],
            },
            "knowledge_access_backend": "workspace_staged",
        },
    )
    assert update_response.status_code == 200

    audit_root = runtime_base / "codex-workspaces"
    audit_root.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        "run_id": "run-123",
        "task_kind": "runtime_smoke",
        "status": "succeeded",
        "selected_model": "gpt-5.5",
        "attempted_models": ["gpt-5.5"],
        "fallback_used": False,
        "error_code": None,
        "recoverable": False,
        "returncode": 0,
        "workspace_root": str(audit_root / "run-123"),
        "finished_at": "2026-07-17T15:50:00Z",
        "metrics": {
            "duration_ms": 1234,
            "queue_wait_ms": 12,
        },
    }
    (audit_root / "runtime-audit.jsonl").write_text(
        json.dumps(audit_payload, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    response = client.get("/api/v1/runtime/status", headers=headers)
    assert response.status_code == 200
    payload = response.json()

    assert payload["provider"] == "codex_local"
    assert payload["implementation_backend"] == "codex_exec_wrapper"
    assert payload["executable"] == str(fake_executable)
    assert payload["version"] == "codex-cli 0.0-test"
    assert payload["auth_mode"] == "chatgpt_session"
    assert payload["auth_detected"] is True
    assert payload["smoke_ready"] is True
    assert payload["runner_id"] == "runtime-status-test"
    assert payload["timeout_ms"] == 210000
    assert payload["max_concurrency"] == 2
    assert payload["configured_models"]["default"] == "gpt-5.5"
    assert payload["configured_fallback_models"]["default"] == ["gpt-5.5-mini", "gpt-5.4-mini"]
    assert "run_codex_runtime_smoke.py" in payload["smoke_command"]
    assert payload["last_known_result"]["run_id"] == "run-123"
    assert payload["last_known_result"]["duration_ms"] == 1234
    assert payload["last_known_result"]["queue_wait_ms"] == 12
    assert "access_token" not in json.dumps(payload).lower()


def test_stage5_workflow_templates_and_governance_are_exposed_and_exported(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()

    assert snapshot["workflow_templates"]
    assert snapshot["governance_policies"]
    assert snapshot["selected_workflow_template_key"]
    assert any(item["handoff_key"] == "governance_review" for item in snapshot["handoff_records"])

    selected_key = snapshot["selected_workflow_template_key"]
    target_template = next(
        item["template_key"] for item in snapshot["workflow_templates"] if item["template_key"] != selected_key
    )

    apply_response = client.post(
        f"/api/v1/sessions/{session_id}/workflow-template/apply",
        headers=headers,
        json={"template_key": target_template},
    )
    assert apply_response.status_code == 200
    updated_snapshot = apply_response.json()

    assert updated_snapshot["selected_workflow_template_key"] == target_template
    assert updated_snapshot["blueprint_versions"][0]["source_action"] == "apply_workflow_template"
    assert any(item["handoff_key"] == "governance_review" for item in updated_snapshot["handoff_records"])
    assert any(item["policy_key"] == "promotion_blockers" for item in updated_snapshot["governance_policies"])

    markdown_export_response = client.get(f"/api/v1/sessions/{session_id}/export/markdown", headers=headers)
    assert markdown_export_response.status_code == 200
    markdown = markdown_export_response.text
    assert "## MVP 3 Governance" in markdown
    assert "### Workflow Templates" in markdown
    assert "### Governance Policies" in markdown


def test_stage5_handoffs_and_feature_flags_support_controlled_return_and_toggle(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    governance_handoff = next(item for item in snapshot["handoff_records"] if item["handoff_key"] == "governance_review")

    handoff_response = client.post(
        f"/api/v1/sessions/{session_id}/handoffs/{governance_handoff['id']}/resolve",
        headers=headers,
        json={"decision": "returned", "resolution_note": "Regresar a blueprint para ajustar gates."},
    )
    assert handoff_response.status_code == 200
    returned_snapshot = handoff_response.json()

    assert returned_snapshot["session"]["status"] == "needs_review"
    assert any(
        item["handoff_key"] == "governance_review" and item["status"] == "returned"
        for item in returned_snapshot["handoff_records"]
    )

    enable_flag_response = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/specialized_subagents_v1",
        headers=headers,
        json={"enabled": True},
    )
    assert enable_flag_response.status_code == 200
    enabled_snapshot = enable_flag_response.json()
    assert any(
        item["key"] == "specialized_subagents_v1" and item["enabled"]
        for item in enabled_snapshot["workspace_contract"]["feature_flags"]
    )

    disable_flag_response = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/specialized_subagents_v1",
        headers=headers,
        json={"enabled": False},
    )
    assert disable_flag_response.status_code == 200
    disabled_snapshot = disable_flag_response.json()
    assert any(
        item["key"] == "specialized_subagents_v1" and not item["enabled"]
        for item in disabled_snapshot["workspace_contract"]["feature_flags"]
    )


def test_stage5_subagents_require_flag_and_leave_specialized_trace(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    disabled_response = client.post(
        f"/api/v1/sessions/{session_id}/subagents/risk_specialist/run",
        headers=headers,
    )
    assert disabled_response.status_code == 409

    enable_flag_response = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/specialized_subagents_v1",
        headers=headers,
        json={"enabled": True},
    )
    assert enable_flag_response.status_code == 200

    run_response = client.post(
        f"/api/v1/sessions/{session_id}/subagents/risk_specialist/run",
        headers=headers,
    )
    assert run_response.status_code == 200
    run_snapshot = run_response.json()
    assert any(item["run_kind"] == "risk_specialist" for item in run_snapshot["subagent_runs"])
    assert any(item["message"] == "Subproceso especializado ejecutado" for item in run_snapshot["activity"])

    disable_flag_response = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/specialized_subagents_v1",
        headers=headers,
        json={"enabled": False},
    )
    assert disable_flag_response.status_code == 200

    rollback_response = client.post(
        f"/api/v1/sessions/{session_id}/subagents/risk_specialist/run",
        headers=headers,
    )
    assert rollback_response.status_code == 409


def test_stage10_multi_agent_runtime_exports_contracts_and_runs_supervisor_orchestrator(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    assert bootstrap_response.status_code == 200
    evaluate_response = client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers)
    assert evaluate_response.status_code == 200

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    blueprint = snapshot_response.json()["blueprint"]
    assert blueprint is not None

    patch_response = client.patch(
        f"/api/v1/sessions/{session_id}/blueprint",
        headers=headers,
        json={
            "architecture": "supervisor_with_subagents",
            "reasoning_pattern": "Plan-and-Execute",
        },
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["data"]["architecture"] == "supervisor_with_subagents"

    disabled_response = client.post(
        f"/api/v1/sessions/{session_id}/subagents/supervisor_orchestrator/run",
        headers=headers,
    )
    assert disabled_response.status_code == 409

    enable_flag_response = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/multi_agent_runtime_v1",
        headers=headers,
        json={"enabled": True},
    )
    assert enable_flag_response.status_code == 200

    run_response = client.post(
        f"/api/v1/sessions/{session_id}/subagents/supervisor_orchestrator/run",
        headers=headers,
    )
    assert run_response.status_code == 200
    run_snapshot = run_response.json()
    specialist_runs = {
        item["run_kind"]: item
        for item in run_snapshot["subagent_runs"]
        if item["run_kind"] in {"evaluation_specialist", "risk_specialist", "artifact_specialist"}
    }
    assert {"evaluation_specialist", "risk_specialist", "artifact_specialist"} <= set(specialist_runs)
    orchestrator_run = next(item for item in run_snapshot["subagent_runs"] if item["run_kind"] == "supervisor_orchestrator")
    assert orchestrator_run["feature_flag_key"] == "multi_agent_runtime_v1"
    assert orchestrator_run["output_payload"]["classification"] == "multi_agent_orchestration_run"
    assert orchestrator_run["output_payload"]["runtime_pattern"] == "supervisor_specialist_runtime"
    assert orchestrator_run["output_payload"]["support_state"] == "supported"
    assert orchestrator_run["output_payload"]["benchmark"]["go_decision"] == "go"
    assert len(orchestrator_run["output_payload"]["agent_contracts"]) >= 4
    assert len(orchestrator_run["output_payload"]["message_contracts"]) >= 3
    assert len(orchestrator_run["output_payload"]["handoff_contracts"]) >= 3
    assert len(orchestrator_run["output_payload"]["shared_state_contracts"]) >= 2
    assert len(orchestrator_run["output_payload"]["specialist_run_ids"]) == 3
    assert len(orchestrator_run["output_payload"]["message_trace"]) == 6
    assert len(orchestrator_run["output_payload"]["handoff_trace"]) == 6
    assert orchestrator_run["output_payload"]["shared_state"]["finding_board"]["evaluation_specialist"]["run_id"]
    assert orchestrator_run["output_payload"]["shared_state"]["final_decision_record"]["benchmark_go_decision"] == "go"
    assert any(item["message"] == "Orquestacion multiagente ejecutada" for item in run_snapshot["activity"])

    construction_pack_response = client.get(
        f"/api/v1/sessions/{session_id}/export/construction-pack?preview=true",
        headers=headers,
    )
    assert construction_pack_response.status_code == 200
    construction_pack = construction_pack_response.json()
    assert construction_pack["multi_agent_benchmark"]["go_decision"] == "go"
    assert construction_pack["behavior_spec"]["multi_agent_topology"]["declared_pattern"] == "supervisor_with_subagents"
    assert construction_pack["behavior_spec"]["multi_agent_topology"]["runtime_pattern"] == "supervisor_specialist_runtime"
    assert construction_pack["behavior_spec"]["multi_agent_topology"]["support_state"] == "supported"
    manifest_paths = {item["path"] for item in construction_pack["file_manifest"]}
    assert any(path.startswith("prompts/agents/agent_role_supervisor") for path in manifest_paths)
    assert any(path.startswith("prompts/handoffs/handoff_supervisor_to_risk_review") for path in manifest_paths)

    prompt_pack_response = client.get(
        f"/api/v1/sessions/{session_id}/export/prompt-pack?preview=true",
        headers=headers,
    )
    assert prompt_pack_response.status_code == 200
    prompt_pack = prompt_pack_response.json()
    assert prompt_pack["agent_role_prompts"]
    assert prompt_pack["handoff_prompts"]
    assert any(item["prompt_key"] == "agent_role_supervisor" for item in prompt_pack["agent_role_prompts"])
    assert any(item["prompt_key"] == "handoff_supervisor_to_risk_review" for item in prompt_pack["handoff_prompts"])


def test_stage10_runtime_isolates_specialist_findings_when_one_branch_needs_review(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    assert bootstrap_response.status_code == 200
    evaluate_response = client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers)
    assert evaluate_response.status_code == 200

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    blueprint = snapshot_response.json()["blueprint"]
    assert blueprint is not None
    tools = [dict(item) for item in blueprint["tools"]]
    tools[0]["risk_level"] = "critical"

    patch_response = client.patch(
        f"/api/v1/sessions/{session_id}/blueprint",
        headers=headers,
        json={
            "architecture": "supervisor_with_subagents",
            "reasoning_pattern": "Plan-and-Execute",
            "tools": tools,
        },
    )
    assert patch_response.status_code == 200

    enable_flag_response = client.patch(
        f"/api/v1/sessions/{session_id}/feature-flags/multi_agent_runtime_v1",
        headers=headers,
        json={"enabled": True},
    )
    assert enable_flag_response.status_code == 200

    run_response = client.post(
        f"/api/v1/sessions/{session_id}/subagents/supervisor_orchestrator/run",
        headers=headers,
    )
    assert run_response.status_code == 200
    run_snapshot = run_response.json()
    orchestrator_run = next(item for item in run_snapshot["subagent_runs"] if item["run_kind"] == "supervisor_orchestrator")
    finding_board = orchestrator_run["output_payload"]["shared_state"]["finding_board"]
    isolation = orchestrator_run["output_payload"]["failure_isolation_result"]
    final_decision = orchestrator_run["output_payload"]["shared_state"]["final_decision_record"]

    assert finding_board["risk_specialist"]["status"] == "needs_review"
    assert finding_board["evaluation_specialist"]["run_id"]
    assert finding_board["artifact_specialist"]["run_id"]
    assert "finding_board.risk_specialist" in isolation["isolated_namespaces"]
    assert "finding_board.evaluation_specialist" in isolation["preserved_namespaces"]
    assert "finding_board.artifact_specialist" in isolation["preserved_namespaces"]
    assert isolation["passed"] is True
    assert final_decision["status"] == "needs_review"
    assert "risk_specialist" in final_decision["blocking_specialists"]


def test_stage5_tool_and_llm_policy_round_trip_updates_canonical_exports(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    assert bootstrap_response.status_code == 200

    initial_prompt_pack = client.get(
        f"/api/v1/sessions/{session_id}/export/prompt-pack?preview=true",
        headers=headers,
    )
    assert initial_prompt_pack.status_code == 200
    initial_hash = initial_prompt_pack.json()["origin"]["input_hash"]

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    blueprint = snapshot_response.json()["blueprint"]
    assert blueprint is not None

    first_tool = dict(blueprint["tools"][0])
    first_tool.update(
        {
            "approval_policy": "not_required",
            "archetype": "read_only",
            "audit_rules": ["Registrar request_id", "Versionar input hash"],
            "auth_reference": "none",
            "contract_review_state": "ready",
            "endpoint_reference": "internal://skill_runtime/normalize_discovery:v2",
            "idempotency_strategy": "Repetible sobre el mismo input estructurado.",
            "integration_kind": "local_runtime",
            "owner": "builder",
            "permissions": ["read_discovery_input"],
            "rate_limit_policy": "Sin limite externo; una ejecucion por accion del usuario.",
            "scopes": ["read"],
            "timeout_policy": "Timeout local corto de 12 segundos.",
            "typed_errors": ["validation_error", "missing_required_field"],
        }
    )

    llm_policy = dict(blueprint["llm_policy"])
    llm_policy.update(
        {
            "provider": "deepseek",
            "fast_model": "deepseek-chat",
            "reasoning_model": "deepseek-reasoner",
            "fallback_model": "manual_review_gate",
            "context_policy": "Usar solo contratos aprobados y el snapshot vigente.",
            "sampling_policy": "Temperatura baja y salidas estructuradas por rol.",
            "fallback_policy": "Escalar a review si falla el provider o el contrato.",
            "circuit_breaker_policy": "Abrir circuit breaker tras 3 fallos consecutivos.",
            "budget_policy": "Reservar reasoning para planner y evaluator.",
            "output_validation_policy": "Validar cada salida contra schemas versionados.",
            "log_redaction_policy": "Redactar secretos y datos sensibles.",
        }
    )
    llm_policy["functions"][0].update({"provider": "deepseek", "model": "deepseek-reasoner"})
    llm_policy["functions"][1].update({"provider": "deepseek", "model": "deepseek-chat"})

    patch_response = client.patch(
        f"/api/v1/sessions/{session_id}/blueprint",
        headers=headers,
        json={
            "llm_policy": llm_policy,
            "tools": [first_tool, *blueprint["tools"][1:]],
        },
    )
    assert patch_response.status_code == 200
    patched_blueprint = patch_response.json()["data"]
    assert patched_blueprint["llm_policy"]["provider"] == "deepseek"
    assert patched_blueprint["tools"][0]["endpoint_reference"] == "internal://skill_runtime/normalize_discovery:v2"
    assert patched_blueprint["tools"][0]["contract_review_state"] == "ready"

    construction_pack_response = client.get(
        f"/api/v1/sessions/{session_id}/export/construction-pack",
        headers=headers,
    )
    assert construction_pack_response.status_code == 200
    construction_pack = construction_pack_response.json()
    assert construction_pack["llm_policy"]["provider"] == "deepseek"
    assert construction_pack["llm_policy"]["fast_model"] == "deepseek-chat"
    assert construction_pack["llm_policy"]["reasoning_model"] == "deepseek-reasoner"
    first_contract = construction_pack["tool_contracts"][0]
    assert first_contract["owner"] == "builder"
    assert first_contract["endpoint_reference"] == "internal://skill_runtime/normalize_discovery:v2"
    assert first_contract["auth_reference"] == "none"
    assert first_contract["contract_review_state"] == "ready"
    serialized_contract = json.dumps(first_contract).lower()
    assert "api_key" not in serialized_contract
    assert "access_token" not in serialized_contract
    assert "sk-" not in serialized_contract

    updated_prompt_pack = client.get(
        f"/api/v1/sessions/{session_id}/export/prompt-pack?preview=true",
        headers=headers,
    )
    assert updated_prompt_pack.status_code == 200
    assert updated_prompt_pack.json()["origin"]["input_hash"] != initial_hash


def test_acp_routes_generate_preview_validate_file_and_zip(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    upgrade_session_tier(client, headers, session_id)

    preview_response = client.get(f"/api/v1/sessions/{session_id}/acp/preview", headers=headers)
    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["manifest_path"] == "ACP/manifest.yaml"
    assert any(item["path"] == "ACP/manifest.yaml" for item in preview["files"])
    assert any(item["path"] == "ACP/tools/contracts/tool-build-blueprint.yaml" for item in preview["files"])
    assert any(item["path"] == "ACP/construction-readiness/overview.yaml" for item in preview["files"])
    assert any(item["path"] == "ACP/prompts/builder-handoff.md" for item in preview["files"])
    assert any(item["path"] == "ACP/diagrams/KnowledgeGraph.md" for item in preview["files"])
    assert any(item["path"] == "ACP/svg/KnowledgeGraph.svg" for item in preview["files"])
    assert any(item["path"] == "ACP/blueprint.graph.json" for item in preview["files"])
    assert preview["validation"]["can_export_zip"] is False
    assert preview["construction_readiness"]["overall_status"] == "blocked"
    assert preview["construction_readiness"]["can_start_build"] is False
    assert preview["construction_readiness"]["blocking_gaps"] >= 1
    assert preview["construction_readiness"]["open_questions"] >= 1

    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    assert bootstrap_response.status_code == 200

    generate_response = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    assert generate_response.status_code == 200
    generated_preview = generate_response.json()
    assert generated_preview["validation"]["can_export_zip"] is True
    assert generated_preview["validation"]["overall_status"] == "needs_review"
    assert any(item["path"] == "ACP/estimation/estimation-report.json" for item in generated_preview["files"])
    assert any(item["path"] == "ACP/estimation/estimation-report.md" for item in generated_preview["files"])
    assert any(item["path"] == "ACP/estimation/assumptions.yaml" for item in generated_preview["files"])
    assert any(item["path"] == "ACP/estimation/sensitivity-drivers.yaml" for item in generated_preview["files"])
    assert generated_preview["construction_readiness"]["overall_status"] == "blocked"
    assert generated_preview["construction_readiness"]["can_start_build"] is False
    assert generated_preview["construction_readiness"]["blocking_gaps"] >= 1
    assert generated_preview["construction_readiness"]["open_questions"] >= 1

    validate_response = client.get(f"/api/v1/sessions/{session_id}/acp/validate", headers=headers)
    assert validate_response.status_code == 200
    validation = validate_response.json()
    assert validation["can_export_zip"] is True
    assert validation["completeness_percent"] == generated_preview["validation"]["completeness_percent"]

    readiness_response = client.get(f"/api/v1/sessions/{session_id}/acp/construction-readiness", headers=headers)
    assert readiness_response.status_code == 200
    readiness = readiness_response.json()
    assert readiness["overall_status"] == "blocked"
    assert readiness["blocking_gaps"] >= 1

    questions_response = client.get(f"/api/v1/sessions/{session_id}/acp/questions", headers=headers)
    assert questions_response.status_code == 200
    questions = questions_response.json()
    assert questions
    assert any(item["gap_key"] == "deployment_target_unknown" for item in questions)

    graph_response = client.get(f"/api/v1/sessions/{session_id}/acp/knowledge-graph", headers=headers)
    assert graph_response.status_code == 200
    graph = graph_response.json()
    assert graph["graph_version"] == "blueprint-graph.v1"
    assert graph["nodes"]
    assert graph["edges"]
    assert any(item["type"] == "Agent" for item in graph["nodes"])

    gap_response = client.get(
        f"/api/v1/sessions/{session_id}/acp/gaps/deployment_target_unknown",
        headers=headers,
    )
    assert gap_response.status_code == 200
    gap = gap_response.json()
    assert gap["gap_key"] == "deployment_target_unknown"
    assert gap["evidence_paths"]

    manifest_response = client.get(
        f"/api/v1/sessions/{session_id}/acp/files/ACP/manifest.yaml",
        headers=headers,
    )
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["path"] == "ACP/manifest.yaml"
    assert "generated_by: Lean Agent Builder" in manifest["content_text"]

    zip_response = client.get(f"/api/v1/sessions/{session_id}/acp/export.zip", headers=headers)
    assert zip_response.status_code == 200
    assert zip_response.headers["content-type"] == "application/zip"
    assert zip_response.content[:2] == b"PK"
    with ZipFile(BytesIO(zip_response.content)) as archive:
        names = sorted(archive.namelist())
    assert "ACP/construction-readiness/overview.yaml" in names
    assert "ACP/construction-readiness/blocking-gaps.yaml" in names
    assert "ACP/prompts/builder-handoff.md" in names
    assert "ACP/diagrams/KnowledgeGraph.md" in names
    assert "ACP/svg/KnowledgeGraph.svg" in names
    assert "ACP/blueprint.graph.json" in names
    assert "ACP/estimation/estimation-report.json" in names
    assert "ACP/estimation/estimation-report.md" in names
    assert names.count("ACP/deployment/env.template") == 1

    artifacts_response = client.get(f"/api/v1/sessions/{session_id}/artifacts", headers=headers)
    assert artifacts_response.status_code == 200
    artifacts = artifacts_response.json()["items"]
    assert any(item["artifact_kind"] == "acp_preview" for item in artifacts)
    assert any(item["artifact_kind"] == "acp_manifest" for item in artifacts)
    assert any(item["artifact_kind"] == "acp_file" for item in artifacts)
    assert any(item["artifact_kind"] == "estimation_report" for item in artifacts)
    assert any(item["artifact_key"] == "acp_zip_export" for item in artifacts)
    assert any(item["artifact_metadata"].get("acp_path") == "ACP/construction-readiness/overview.yaml" for item in artifacts)
    assert any(item["artifact_metadata"].get("acp_path") == "ACP/diagrams/Architecture.md" for item in artifacts)
    assert any(item["artifact_metadata"].get("acp_path") == "ACP/svg/Architecture.svg" for item in artifacts)
    assert any(item["artifact_metadata"].get("lineage_scope") == "construction_readiness" for item in artifacts if item["artifact_kind"] == "acp_file")
    assert any(item["artifact_metadata"].get("acp_path") == "ACP/estimation/estimation-report.md" for item in artifacts)
    assert any(item["artifact_metadata"].get("lineage_scope") == "estimation" for item in artifacts if item["artifact_kind"] == "acp_file")


def test_acp_questions_can_be_answered_and_reinjected_into_regeneration(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    upgrade_session_tier(client, headers, session_id)

    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    assert bootstrap_response.status_code == 200

    generate_response = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    assert generate_response.status_code == 200

    questions_response = client.get(f"/api/v1/sessions/{session_id}/acp/questions", headers=headers)
    assert questions_response.status_code == 200
    questions = questions_response.json()
    question_keys = {item["question_key"] for item in questions}
    assert "deployment_target" in question_keys

    answer_response = client.patch(
        f"/api/v1/sessions/{session_id}/acp/questions/deployment_target",
        headers=headers,
        json={
            "answer_text": "target=local_vm; restrictions=solo red interna y acceso por VPN",
            "owner_role": "platform_owner",
            "impacted_artifacts": ["ACP/deployment/env.template"],
        },
    )
    assert answer_response.status_code == 200
    answered_question = answer_response.json()
    assert answered_question["status"] == "answered"
    assert answered_question["owner_role"] == "platform_owner"

    refreshed_questions_response = client.get(f"/api/v1/sessions/{session_id}/acp/questions", headers=headers)
    assert refreshed_questions_response.status_code == 200
    refreshed_questions = refreshed_questions_response.json()
    deployment_question = next(item for item in refreshed_questions if item["question_key"] == "deployment_target")
    assert deployment_question["status"] == "answered"
    assert deployment_question["answer_text"] == "target=local_vm; restrictions=solo red interna y acceso por VPN"

    answers_by_key = {
        "knowledge_sources": "name=Confluence; type=wiki; owner=ops; frequency=diaria",
        "knowledge_ingestion": "strategy=sync_incremental; frequency=diaria; mechanism=cron; owner=ops",
        "knowledge_embedding_strategy": "provider=text-embedding-3-small; chunking=800_tokens_overlap_120; notes=openai",
        "runtime_fallback_model": "model=gpt-4.1-mini; condition=cuando falle el modelo primario",
        "runtime_vector_store": "vector_store=pgvector; notes=misma base local",
        "runtime_secret_source": "source=.env local protegido; owner=platform_owner; environment=desarrollo",
        "deployment_target": "target=local_vm; restrictions=solo red interna y acceso por VPN",
        "deployment_image_strategy": "strategy=docker_compose_local; registry=no_aplica",
        "deployment_network_constraints": "network=solo red interna; secrets=.env local; dependencies=postgres local",
    }

    for question_key in question_keys:
        response = client.patch(
            f"/api/v1/sessions/{session_id}/acp/questions/{question_key}",
            headers=headers,
            json={
                "answer_text": answers_by_key[question_key],
                "owner_role": "owner_resuelto",
                "impacted_artifacts": [],
            },
        )
        assert response.status_code == 200

    regenerated_response = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    assert regenerated_response.status_code == 200
    regenerated_preview = regenerated_response.json()
    assert regenerated_preview["construction_readiness"]["overall_status"] == "ready_to_build"
    assert regenerated_preview["construction_readiness"]["blocking_gaps"] == 0
    assert regenerated_preview["construction_readiness"]["open_questions"] == 0

    readiness_response = client.get(f"/api/v1/sessions/{session_id}/acp/construction-readiness", headers=headers)
    assert readiness_response.status_code == 200
    readiness = readiness_response.json()
    assert readiness["overall_status"] == "ready_to_build"
    assert readiness["can_start_build"] is True

    final_questions_response = client.get(f"/api/v1/sessions/{session_id}/acp/questions", headers=headers)
    assert final_questions_response.status_code == 200
    final_questions = final_questions_response.json()
    assert any(item["question_key"] == "deployment_target" and item["status"] == "resolved" for item in final_questions)


def test_canonical_export_routes_publish_metadata_and_enforce_preview_mode(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    preview_response = client.get(
        f"/api/v1/sessions/{session_id}/export/construction-pack?preview=true",
        headers=headers,
    )
    assert preview_response.status_code == 200
    preview_payload = preview_response.json()
    assert preview_payload["schema_version"] == "construction-pack.v1"
    assert preview_response.headers["x-canonical-contract-version"] == "construction-pack.v1"
    assert preview_response.headers["x-canonical-export-preview"] == "true"
    assert preview_response.headers["x-canonical-export-readiness"] == "blocked"
    assert len(preview_response.headers["x-canonical-checksum-sha256"]) == 64

    blocked_response = client.get(
        f"/api/v1/sessions/{session_id}/export/construction-pack",
        headers=headers,
    )
    assert blocked_response.status_code == 403
    assert "ACP Premium" in blocked_response.json()["detail"]

    blueprint_core_preview_response = client.get(
        f"/api/v1/sessions/{session_id}/export/blueprint-core?preview=true",
        headers=headers,
    )
    assert blueprint_core_preview_response.status_code == 200
    assert blueprint_core_preview_response.headers["x-canonical-contract-version"] == "blueprint-core.v1"
    assert blueprint_core_preview_response.headers["x-canonical-export-preview"] == "true"
    assert blueprint_core_preview_response.json()["schema_version"] == "blueprint-core.v1"

    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    assert bootstrap_response.status_code == 200
    estimate_response = client.post(f"/api/v1/sessions/{session_id}/estimate", headers=headers)
    assert estimate_response.status_code == 200

    upgrade_session_tier(client, headers, session_id)
    ready_response = client.get(
        f"/api/v1/sessions/{session_id}/export/construction-pack",
        headers=headers,
    )
    assert ready_response.status_code == 200
    assert ready_response.headers["x-canonical-export-preview"] == "false"
    assert ready_response.headers["x-canonical-export-readiness"] == "ready"

    prompt_pack_response = client.get(
        f"/api/v1/sessions/{session_id}/export/prompt-pack?preview=true",
        headers=headers,
    )
    assert prompt_pack_response.status_code == 200
    assert prompt_pack_response.json()["schema_version"] == "prompt-pack.v1"
    assert prompt_pack_response.json()["origin"]["blueprint_core_version"] == "blueprint-core.v1"

    test_pack_preview_response = client.get(
        f"/api/v1/sessions/{session_id}/export/test-pack?preview=true",
        headers=headers,
    )
    assert test_pack_preview_response.status_code == 200
    assert test_pack_preview_response.headers["x-canonical-contract-version"] == "test-pack.v1"
    assert test_pack_preview_response.json()["schema_version"] == "test-pack.v1"
    assert test_pack_preview_response.json()["framework_target"] == "python-stdlib-external-consumer"
    assert any(item["contract_key"] == "evaluation-pack.v1" for item in test_pack_preview_response.json()["fixtures"])

    test_pack_response = client.get(
        f"/api/v1/sessions/{session_id}/export/test-pack",
        headers=headers,
    )
    assert test_pack_response.status_code == 200
    assert test_pack_response.headers["x-canonical-export-preview"] == "false"
    assert test_pack_response.headers["x-canonical-export-readiness"] == "ready"
    assert any(item["kind"] == "mutation" for item in test_pack_response.json()["commands"])


def test_package_preview_and_canonical_exports_detect_consistency_drift_against_approved_chain(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)
    approve_tools_for_session(client, headers, session_id)
    approve_memory_for_session(client, headers, session_id)
    approve_validate_for_session(client, headers, session_id)

    drift_response = client.patch(
        f"/api/v1/sessions/{session_id}/blueprint",
        headers=headers,
        json={
            "architecture": "router_parallel",
            "narrative": "Cambio manual posterior a la aprobacion para validar drift transversal.",
        },
    )
    assert drift_response.status_code == 200

    snapshot_response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()
    assert snapshot["blueprint_consistency"]["overall_status"] == "blocked"
    assert any(
        item["issue_key"] == "design_blueprint_projection_drift:architecture"
        for item in snapshot["blueprint_consistency"]["issues"]
    )
    assert snapshot["blueprint_consistency"]["approved_stage_lineage"]

    preview_response = client.get(
        f"/api/v1/sessions/{session_id}/export/construction-pack?preview=true",
        headers=headers,
    )
    assert preview_response.status_code == 200
    assert preview_response.headers["x-canonical-export-readiness"] == "blocked"
    preview_payload = preview_response.json()
    assert any("Blueprint tiene `router_parallel`" in item for item in preview_payload["readiness"]["blocking_issues"])
    assert preview_payload["topology"]["consistency_summary"]["overall_status"] == "blocked"

    upgrade_session_tier(client, headers, session_id)
    acp_response = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    assert acp_response.status_code == 200
    acp_payload = acp_response.json()
    assert acp_payload["construction_readiness"]["overall_status"] == "blocked"
    assert any(item["gap_key"] == "cross_stage_consistency_drift" for item in acp_payload["construction_readiness"]["gaps"])
    generated_paths = {item["path"] for item in acp_payload["files"]}
    assert "ACP/governance/consistency-report.json" in generated_paths
    assert "ACP/governance/approved-stage-lineage.yaml" in generated_paths
    assert "ACP/governance/journey-decisions.json" in generated_paths


def test_blueprint_professional_markdown_export_includes_consistency_and_decision_history(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    approve_design_for_session(client, headers, session_id)
    approve_tools_for_session(client, headers, session_id)
    approve_memory_for_session(client, headers, session_id)
    approve_validate_for_session(client, headers, session_id)
    upgrade_session_tier(client, headers, session_id, tier="blueprint_pro")

    markdown_response = client.get(f"/api/v1/sessions/{session_id}/export/markdown", headers=headers)
    assert markdown_response.status_code == 200
    markdown = markdown_response.text
    assert "# Blueprint consistency" in markdown
    assert "## Approved stage lineage" in markdown
    assert "## Journey Decisions" in markdown


def test_acp_design_only_profile_closes_independently_from_extended_profile(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)
    upgrade_session_tier(client, headers, session_id)

    bootstrap_response = client.post(f"/api/v1/sessions/{session_id}/evaluation/bootstrap", headers=headers)
    assert bootstrap_response.status_code == 200

    initial_preview = client.post(f"/api/v1/sessions/{session_id}/acp/generate", headers=headers)
    assert initial_preview.status_code == 200

    design_only_questions_response = client.get(
        f"/api/v1/sessions/{session_id}/acp/questions?profile=design-only",
        headers=headers,
    )
    assert design_only_questions_response.status_code == 200
    questions = design_only_questions_response.json()
    question_keys = {item["question_key"] for item in questions}
    assert "deployment_target" not in question_keys
    assert "runtime_secret_source" in question_keys

    answers_by_key = {
        "knowledge_sources": "name=Confluence; type=wiki; owner=ops; frequency=diaria",
        "knowledge_ingestion": "strategy=sync_incremental; frequency=diaria; mechanism=cron; owner=ops",
        "knowledge_embedding_strategy": "provider=text-embedding-3-small; chunking=800_tokens_overlap_120; notes=openai",
        "runtime_fallback_model": "model=gpt-4.1-mini; condition=cuando falle el modelo primario",
        "runtime_vector_store": "vector_store=no aplica; notes=memoria de sesion sin retrieval persistente",
        "runtime_secret_source": "source=.env local protegido; owner=platform_owner; environment=desarrollo",
        "external_api_contracts": "tool=build_blueprint; system=builder; endpoint=/api/v1/build; action=compose; auth=session_token; request=normalized_discovery; response=blueprint; errors=validation",
    }

    for question in questions:
        response = client.patch(
            f"/api/v1/sessions/{session_id}/acp/questions/{question['question_key']}",
            headers=headers,
            json={
                "answer_text": answers_by_key[question["question_key"]],
                "owner_role": "platform_owner",
                "impacted_artifacts": [],
            },
        )
        assert response.status_code == 200

    profiled_preview_response = client.post(
        f"/api/v1/sessions/{session_id}/acp/generate?profile=design-only",
        headers=headers,
    )
    assert profiled_preview_response.status_code == 200
    profiled_preview = profiled_preview_response.json()
    assert profiled_preview["construction_readiness"]["overall_status"] == "ready_to_build"
    assert all(not item["path"].startswith("ACP/deployment/") for item in profiled_preview["files"])
    assert all(not item["path"].startswith("ACP/observability/") for item in profiled_preview["files"])

    design_only_zip_response = client.get(
        f"/api/v1/sessions/{session_id}/acp/export.zip?profile=design-only",
        headers=headers,
    )
    assert design_only_zip_response.status_code == 200
    assert design_only_zip_response.headers["x-acp-export-profile"] == "design-only"
    assert design_only_zip_response.headers["x-acp-export-readiness"] == "ready_to_build"
    assert len(design_only_zip_response.headers["x-acp-export-checksum-sha256"]) == 64
    with ZipFile(BytesIO(design_only_zip_response.content)) as archive:
        names = sorted(archive.namelist())
    assert all(not name.startswith("ACP/deployment/") for name in names)
    assert all(not name.startswith("ACP/observability/") for name in names)
    assert "ACP/runtime/config.yaml" in names

    extended_zip_response = client.get(
        f"/api/v1/sessions/{session_id}/acp/export.zip?profile=extended",
        headers=headers,
    )
    assert extended_zip_response.status_code == 409
    assert "extended" in extended_zip_response.json()["detail"]


def test_patch_blueprint_rejects_ungoverned_fields(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    response = client.patch(
        f"/api/v1/sessions/{session_id}/blueprint",
        headers=headers,
        json={
            "architecture": "single_agent",
            "rogue_field": "should_fail",
        },
    )

    assert response.status_code == 422


def test_build_blueprint_blocks_when_structured_mvp_definition_is_missing(client: TestClient) -> None:
    headers = auth_headers(client)
    create_response = client.post("/api/v1/sessions", headers=headers)
    assert create_response.status_code == 201
    session_id = create_response.json()["id"]

    incomplete_payload = {
        "problem_statement": "Disenar un agente de onboarding",
        "current_user": "Equipo de RRHH",
        "current_process": "Recolecta datos por correo y luego arma documentos manualmente.",
        "desired_outcome": "Tener un blueprint consistente del agente",
        "autonomy_level": "medium",
        "constraints": ["No aprobar accesos automaticamente"],
        "operational_baseline": {
            "current_time_spent": "3 horas por caso",
            "current_cost": "Tiempo analista",
            "frequent_errors": ["Se repiten preguntas al usuario"],
            "automation_opportunities": ["Normalizar discovery"],
        },
        "mvp_definition": {
            "v1_scope": [],
            "out_of_scope": ["Provisioning automatico"],
            "north_star_metric": "",
            "non_delegable_decisions": [],
        },
    }

    normalize_response = client.post(
        f"/api/v1/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json=incomplete_payload,
    )
    assert normalize_response.status_code == 200
    assert normalize_response.json()["status"] == "needs_review"

    canvas_response = client.post(f"/api/v1/sessions/{session_id}/build-canvas", headers=headers)
    assert canvas_response.status_code == 409
    assert canvas_response.json()["detail"] == "Discover must be approved before canvas"

    blueprint_response = client.post(f"/api/v1/sessions/{session_id}/build-blueprint", headers=headers)
    assert blueprint_response.status_code == 409
    assert "define must be approved before blueprint" in blueprint_response.text.lower()


def test_rerun_skill_persists_trace_and_diff(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    rerun_response = client.post(
        f"/api/v1/sessions/{session_id}/skills/tool_design_skill/rerun",
        headers=headers,
    )
    assert rerun_response.status_code == 200
    payload = rerun_response.json()

    assert payload["skill_run"]["skill_key"] == "tool_design_skill"
    assert payload["skill_run"]["source_action"] == "rerun:tool_design_skill"
    assert any(item["artifact_role"] == "input" for item in payload["skill_run"]["artifacts"])
    assert any(item["artifact_role"] == "output" for item in payload["skill_run"]["artifacts"])
    assert any(item["artifact_role"] == "diff" for item in payload["skill_run"]["artifacts"])
    assert payload["snapshot"]["skill_runs"][0]["id"] == payload["skill_run"]["id"]


def test_evaluation_workbench_bootstrap_updates_and_persists_runs(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    bootstrap_response = client.post(
        f"/api/v1/sessions/{session_id}/evaluation/bootstrap",
        headers=headers,
    )
    assert bootstrap_response.status_code == 200
    bootstrap_snapshot = bootstrap_response.json()

    assert bootstrap_snapshot["evaluation_dataset"]["version_number"] == 1
    assert bootstrap_snapshot["evaluation_rubric"]["version_number"] == 1
    assert any(item["category"] == "tool_failure" for item in bootstrap_snapshot["evaluation_dataset"]["cases"])
    assert any(item["key"] == "safety" and item["hard_block"] for item in bootstrap_snapshot["evaluation_rubric"]["dimensions"])

    edited_cases = bootstrap_snapshot["evaluation_dataset"]["cases"]
    edited_cases[0]["title"] = "Happy path afinado"
    edited_cases[0]["source"] = "manual"
    dataset_update_response = client.patch(
        f"/api/v1/sessions/{session_id}/evaluation/dataset",
        headers=headers,
        json={"cases": edited_cases},
    )
    assert dataset_update_response.status_code == 200
    dataset_snapshot = dataset_update_response.json()
    assert dataset_snapshot["evaluation_dataset"]["version_number"] == 2
    assert dataset_snapshot["evaluation_dataset"]["cases"][0]["title"] == "Happy path afinado"

    edited_dimensions = dataset_snapshot["evaluation_rubric"]["dimensions"]
    edited_dimensions[0]["weight"] = 30
    rubric_update_response = client.patch(
        f"/api/v1/sessions/{session_id}/evaluation/rubric",
        headers=headers,
        json={
            "summary": "Rubrica afinada para stage 3.",
            "dimensions": edited_dimensions,
        },
    )
    assert rubric_update_response.status_code == 200
    rubric_snapshot = rubric_update_response.json()
    assert rubric_snapshot["evaluation_rubric"]["version_number"] == 2
    assert rubric_snapshot["evaluation_rubric"]["summary"] == "Rubrica afinada para stage 3."
    assert rubric_snapshot["evaluation_rubric"]["dimensions"][0]["weight"] == 30

    evaluation_response = client.post(f"/api/v1/sessions/{session_id}/evaluate", headers=headers)
    assert evaluation_response.status_code == 200

    evaluated_snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()
    assert evaluated_snapshot["evaluation_dataset"]["version_number"] == 2
    assert evaluated_snapshot["evaluation_rubric"]["version_number"] == 2
    assert len(evaluated_snapshot["evaluation_runs"]) == 1
    assert evaluated_snapshot["evaluation_runs"][0]["source_action"] == "evaluate_blueprint"
    assert evaluated_snapshot["evaluation_runs"][0]["dataset_version_number"] == 2
    assert evaluated_snapshot["evaluation_runs"][0]["rubric_version_number"] == 2
    assert evaluated_snapshot["evaluation_runs"][0]["results"]

    rerun_response = client.post(
        f"/api/v1/sessions/{session_id}/skills/evaluation_skill/rerun",
        headers=headers,
    )
    assert rerun_response.status_code == 200
    rerun_payload = rerun_response.json()
    assert rerun_payload["skill_run"]["skill_key"] == "evaluation_skill"
    assert rerun_payload["snapshot"]["evaluation_runs"][0]["source_action"] == "rerun:evaluation_skill"
    assert len(rerun_payload["snapshot"]["evaluation_runs"]) == 2


def test_saas_tiers_gate_exports_and_library_access(client: TestClient) -> None:
    headers, session_id = build_session_flow(client)

    initial_snapshot = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert initial_snapshot.status_code == 200
    assert initial_snapshot.json()["commercial_access"]["tier"] == "blueprint"

    markdown_blocked = client.get(f"/api/v1/sessions/{session_id}/export/markdown", headers=headers)
    assert markdown_blocked.status_code == 403
    assert "Blueprint Profesional" in markdown_blocked.json()["detail"]

    blueprint_core_blocked = client.get(f"/api/v1/sessions/{session_id}/export/blueprint-core", headers=headers)
    assert blueprint_core_blocked.status_code == 403
    assert "Blueprint Profesional" in blueprint_core_blocked.json()["detail"]

    library_blocked = client.get(f"/api/v1/sessions/{session_id}/library", headers=headers)
    assert library_blocked.status_code == 403
    assert "ACP Premium" in library_blocked.json()["detail"]

    acp_preview_blocked = client.get(f"/api/v1/sessions/{session_id}/acp/preview", headers=headers)
    assert acp_preview_blocked.status_code == 403
    assert "ACP Premium" in acp_preview_blocked.json()["detail"]

    pro_upgrade = client.patch(
        f"/api/v1/sessions/{session_id}/commercial-tier",
        headers=headers,
        json={"tier": "blueprint_pro"},
    )
    assert pro_upgrade.status_code == 200
    assert pro_upgrade.json()["commercial_access"]["tier"] == "blueprint_pro"

    markdown_allowed = client.get(f"/api/v1/sessions/{session_id}/export/markdown", headers=headers)
    assert markdown_allowed.status_code == 200
    assert "## Discovery" in markdown_allowed.text

    json_allowed = client.get(f"/api/v1/sessions/{session_id}/export/json", headers=headers)
    assert json_allowed.status_code == 200

    blueprint_core_allowed = client.get(f"/api/v1/sessions/{session_id}/export/blueprint-core", headers=headers)
    assert blueprint_core_allowed.status_code == 200
    assert blueprint_core_allowed.headers["x-canonical-contract-version"] == "blueprint-core.v1"
    assert blueprint_core_allowed.json()["schema_version"] == "blueprint-core.v1"

    library_still_blocked = client.get(f"/api/v1/sessions/{session_id}/library", headers=headers)
    assert library_still_blocked.status_code == 403

    acp_upgrade = client.patch(
        f"/api/v1/sessions/{session_id}/commercial-tier",
        headers=headers,
        json={"tier": "acp"},
    )
    assert acp_upgrade.status_code == 200
    assert acp_upgrade.json()["commercial_access"]["tier"] == "acp"

    library_allowed = client.get(f"/api/v1/sessions/{session_id}/library", headers=headers)
    assert library_allowed.status_code == 200
