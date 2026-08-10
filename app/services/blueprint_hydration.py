from __future__ import annotations

from app.models import BlueprintArtifact, BlueprintRecord, ReviewState


def hydrate_blueprint_record(record: BlueprintRecord) -> BlueprintArtifact:
    readiness_state = record.readiness_state
    if readiness_state is not None and not isinstance(readiness_state, ReviewState):
        readiness_state = ReviewState(str(readiness_state))
    return BlueprintArtifact.model_validate(
        {
            **record.model_dump(exclude={"id", "session_id", "updated_at"}, warnings=False),
            "tools": record.tools,
            "memory_profile": record.memory_profile,
            "knowledge_profile": record.knowledge_profile,
            "safety_checks": record.safety_checks,
            "delivery_package": record.delivery_package,
            "readiness_state": readiness_state,
        }
    )
