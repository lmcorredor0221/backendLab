from app.models import (
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintEnvelope,
    BlueprintPatchRequest,
    CanvasArtifact,
    CanvasEnvelope,
    DiscoveryArtifact,
    DiscoveryEnvelope,
    DiscoveryInput,
    EvidenceItem,
    EvidenceSource,
    EvaluationEnvelope,
    ReviewState,
    SessionStage,
)
from app.services.rules import derive_readiness_state
from app.services.skill_runtime import (
    compose_blueprint_artifact,
    run_blueprint_stage,
    run_canvas_stage,
    run_discovery_stage,
    run_enrich_stage,
    run_evaluation_stage,
)


def normalize_discovery(payload: DiscoveryInput) -> DiscoveryEnvelope:
    envelope, _ = run_discovery_stage(payload)
    return envelope


def build_canvas(discovery: DiscoveryArtifact) -> CanvasEnvelope:
    envelope, _ = run_canvas_stage(discovery)
    return envelope


def build_blueprint(discovery: DiscoveryArtifact, canvas: CanvasArtifact) -> BlueprintEnvelope:
    envelope, _ = run_blueprint_stage(discovery, canvas)
    return envelope


def enrich_blueprint(
    blueprint: BlueprintArtifact,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
) -> BlueprintEnvelope:
    envelope, _ = run_enrich_stage(blueprint, discovery, canvas)
    return envelope


def patch_blueprint(
    current: BlueprintArtifact,
    patch: BlueprintPatchRequest,
    discovery: DiscoveryArtifact,
    canvas: CanvasArtifact,
) -> BlueprintEnvelope:
    updates = patch.model_dump(exclude_unset=True)
    current_payload = current.model_dump(mode="python")
    current_payload.update(updates)
    updated = BlueprintArtifact.model_validate(current_payload)
    rebuilt = compose_blueprint_artifact(
        discovery,
        canvas,
        architecture=updated.architecture,
        reasoning_pattern=updated.reasoning_pattern,
        memory_strategy=updated.memory_strategy,
        tools=updated.tools,
        llm_policy=updated.llm_policy,
        memory_profile=updated.memory_profile,
        knowledge_profile=updated.knowledge_profile,
        safety_checks=updated.safety_checks,
        guardrails=updated.guardrails,
        narrative=updated.narrative,
    )
    if patch.delivery_package is not None:
        rebuilt = rebuilt.model_copy(update={"delivery_package": patch.delivery_package})
        rebuilt = rebuilt.model_copy(update={"readiness_state": derive_readiness_state(rebuilt)})
    status = rebuilt.readiness_state
    return BlueprintEnvelope(
        status=ArtifactStatus.ready if status == ReviewState.complete else ArtifactStatus.needs_review,
        stage=SessionStage.post_validation,
        data=rebuilt,
        missing_fields=[],
        assumptions=[],
        warnings=[] if status == ReviewState.complete else ["Persisten elementos por revisar en el blueprint."],
        evidence=[
            EvidenceItem(
                source=EvidenceSource.form_input,
                detail="Blueprint ajustado manualmente por el usuario",
            )
        ],
        next_action="evaluate_blueprint" if status == ReviewState.complete else "review_blueprint_details",
    )


def evaluate_blueprint(
    discovery: DiscoveryArtifact | None,
    canvas: CanvasArtifact | None,
    blueprint: BlueprintArtifact | None,
) -> EvaluationEnvelope:
    envelope, _ = run_evaluation_stage(discovery, canvas, blueprint)
    return envelope
