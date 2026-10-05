from __future__ import annotations

from uuid import uuid4

from app.models import (
    AgentCanvasProfile,
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintTool,
    CanvasArtifact,
    CommercialTier,
    DiscoveryArtifact,
    SessionCreateResponse,
    SessionSnapshot,
    SessionStage,
    utc_now,
)
from app.services.blueprint_commercial_result_service import _build_commercial_specs


def test_commercial_mermaid_diagrams_are_parameterized_from_snapshot() -> None:
    snapshot = SessionSnapshot(
        session=SessionCreateResponse(
            id=uuid4(),
            workspace_id=uuid4(),
            title="Triage clinico contextual",
            status=ArtifactStatus.ready,
            current_stage=SessionStage.ready_for_export,
            commercial_tier=CommercialTier.blueprint,
            created_at=utc_now(),
            updated_at=utc_now(),
        ),
        discovery=DiscoveryArtifact(
            problem_statement="Las solicitudes de soporte clinico se clasifican tarde.",
            current_user="Coordinador de soporte clinico",
            current_process="Radicacion manual en correo.",
            desired_outcome="Priorizar casos urgentes en menos de cinco minutos.",
            value_statement="Reducir espera clinica con trazabilidad.",
        ),
        canvas=CanvasArtifact(
            mvp_scope=["Clasificar solicitudes por urgencia", "Escalar casos criticos"],
            out_of_scope=["Autorizar procedimientos medicos"],
            success_metric="Casos urgentes priorizados en menos de cinco minutos",
            primary_risk="Riesgo de clasificacion clinica incorrecta",
            agent_profile=AgentCanvasProfile(
                primary_user="Coordinador de soporte clinico",
                human_approvals=["Aprobar cambio de prioridad clinica"],
            ),
        ),
        blueprint=BlueprintArtifact(
            architecture="router_triage_with_human_gate",
            reasoning_pattern="structured_triage",
            memory_strategy="case_summary_checkpoints",
            tools=[BlueprintTool(name="Zendesk Salud", purpose="Leer tickets aprobados")],
            guardrails=["No emitir diagnosticos medicos"],
            narrative="Arquitectura de triage con validacion humana.",
        ),
    )

    specs = {spec.artifact_key: spec for spec in _build_commercial_specs(snapshot)}
    architecture = specs["Blueprint/commercial/diagrams/arquitectura.mmd"].content_text
    value = specs["Blueprint/commercial/diagrams/flujo-valor.mmd"].content_text
    scope = specs["Blueprint/commercial/diagrams/alcance-lean.mmd"].content_text

    assert "router_triage_with_human_gate" in architecture
    assert "Zendesk Salud" in architecture
    assert "case_summary_checkpoints" in architecture
    assert "Coordinador de soporte clinico" in value
    assert "Priorizar casos urgentes" in value
    assert "Clasificar solicitudes por urgencia" in scope
    assert "Autorizar procedimientos medicos" in scope
    assert 'Business["Necesidad de negocio"]' not in architecture
    assert 'Discover["Descubrir"]' not in value
    assert "Patrones agenticos" not in scope
