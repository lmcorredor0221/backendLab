from app.diagnostics import normalize_autonomy_level, normalize_case_type
from app.models import DiscoveryArtifact, DiscoveryInput


def test_normalize_autonomy_level_accepts_legacy_aliases() -> None:
    assert normalize_autonomy_level("assist") == "low"
    assert normalize_autonomy_level("copilot") == "medium"
    assert normalize_autonomy_level("autonomous") == "high"


def test_normalize_case_type_maps_legacy_values_to_canonical() -> None:
    assert normalize_case_type("workflow_automation") == "automatizacion"
    assert normalize_case_type("informational_assistant") == "informacion"
    assert normalize_case_type("single_task_builder") == "copiloto"


def test_discovery_models_store_canonical_taxonomy() -> None:
    payload = DiscoveryInput(
        problem_statement="Responder preguntas frecuentes de clientes",
        current_user="Soporte",
        current_process="Revisa un documento y responde por correo",
        desired_outcome="Contestar consultas repetitivas sin perder contexto",
        autonomy_level="supervised",
    )
    artifact = DiscoveryArtifact(
        problem_statement=payload.problem_statement,
        current_user=payload.current_user,
        current_process=payload.current_process,
        desired_outcome=payload.desired_outcome,
        autonomy_level="assist",
        case_type="informational_assistant",
    )

    assert payload.autonomy_level == "medium"
    assert artifact.autonomy_level == "low"
    assert artifact.case_type == "informacion"
