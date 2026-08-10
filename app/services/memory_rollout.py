from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlmodel import Session

from app.core.config import get_settings
from app.models import (
    KnowledgeAccessBackend,
    LLMRuntimeSettings,
    MemoryRolloutPhaseEntry,
    MemoryRolloutStageEntry,
    MemoryRolloutSummary,
)
from app.services.journey_stage_contract import journey_stage_for_source_action
from app.services.stage5_service import is_feature_flag_enabled


FEATURE_FLAG_MEMORY_HYBRID_DEFINE_DESIGN = "memory_hybrid_define_design_v1"
FEATURE_FLAG_MEMORY_HYBRID_EXTENDED_JOURNEY = "memory_hybrid_extended_journey_v1"

PHASE_DEFINE_DESIGN = "define_design"
PHASE_EXTENDED_JOURNEY = "extended_journey"


@dataclass(frozen=True)
class JourneyMemoryStageSpec:
    stage_key: str
    label: str
    phase_key: str
    expects_llm_call: bool


DISCOVER_STAGE = JourneyMemoryStageSpec(
    stage_key="discover",
    label="Discover",
    phase_key=PHASE_DEFINE_DESIGN,
    expects_llm_call=True,
)

M8_STAGE_SPECS: tuple[JourneyMemoryStageSpec, ...] = (
    JourneyMemoryStageSpec(
        stage_key="define",
        label="Define",
        phase_key=PHASE_DEFINE_DESIGN,
        expects_llm_call=True,
    ),
    JourneyMemoryStageSpec(
        stage_key="design",
        label="Design",
        phase_key=PHASE_DEFINE_DESIGN,
        expects_llm_call=True,
    ),
    JourneyMemoryStageSpec(
        stage_key="tools",
        label="Tools",
        phase_key=PHASE_EXTENDED_JOURNEY,
        expects_llm_call=True,
    ),
    JourneyMemoryStageSpec(
        stage_key="memory",
        label="Memory",
        phase_key=PHASE_EXTENDED_JOURNEY,
        expects_llm_call=True,
    ),
    JourneyMemoryStageSpec(
        stage_key="evaluate",
        label="Evaluate",
        phase_key=PHASE_EXTENDED_JOURNEY,
        expects_llm_call=False,
    ),
    JourneyMemoryStageSpec(
        stage_key="build",
        label="Build",
        phase_key=PHASE_EXTENDED_JOURNEY,
        expects_llm_call=False,
    ),
)

_STAGE_BY_KEY = {item.stage_key: item for item in (DISCOVER_STAGE, *M8_STAGE_SPECS)}


def _rollout_manifest_path(runtime_root: Path | None = None) -> Path:
    settings = get_settings()
    runtime_base = runtime_root or settings.llm_config_path.parent / "knowledge-memory"
    return runtime_base / "knowledge-corpus-manifest.json"


def is_rollout_manifest_ready(runtime_root: Path | None = None) -> bool:
    manifest_path = _rollout_manifest_path(runtime_root)
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and bool(payload.get("corpus_hash"))


def _phase_enabled(
    *,
    phase_key: str,
    define_design_enabled: bool,
    extended_enabled: bool,
) -> bool:
    if phase_key == PHASE_DEFINE_DESIGN:
        return define_design_enabled
    return extended_enabled


def resolve_effective_stage_backend(
    requested_backend: str,
    *,
    stage_key: str,
    manifest_ready: bool,
    define_design_enabled: bool = True,
    extended_enabled: bool = True,
) -> str:
    normalized = requested_backend.strip() or KnowledgeAccessBackend.workspace_staged.value
    if normalized not in {
        KnowledgeAccessBackend.inline_context.value,
        KnowledgeAccessBackend.workspace_staged.value,
        KnowledgeAccessBackend.hybrid.value,
    }:
        normalized = KnowledgeAccessBackend.inline_context.value
    if normalized == KnowledgeAccessBackend.inline_context.value:
        return normalized

    spec = _STAGE_BY_KEY.get(stage_key)
    if spec is None:
        return KnowledgeAccessBackend.inline_context.value

    if not manifest_ready:
        return KnowledgeAccessBackend.inline_context.value

    if not _phase_enabled(
        phase_key=spec.phase_key,
        define_design_enabled=define_design_enabled,
        extended_enabled=extended_enabled,
    ):
        return KnowledgeAccessBackend.inline_context.value
    return normalized


def resolve_runtime_settings_for_stage(
    runtime_settings: LLMRuntimeSettings,
    *,
    stage_key: str,
    manifest_ready: bool | None = None,
    define_design_enabled: bool = True,
    extended_enabled: bool = True,
) -> LLMRuntimeSettings:
    resolved_manifest_ready = (
        is_rollout_manifest_ready() if manifest_ready is None else manifest_ready
    )
    effective_backend = resolve_effective_stage_backend(
        runtime_settings.knowledge_access_backend.value,
        stage_key=stage_key,
        manifest_ready=resolved_manifest_ready,
        define_design_enabled=define_design_enabled,
        extended_enabled=extended_enabled,
    )
    return runtime_settings.model_copy(
        update={"knowledge_access_backend": KnowledgeAccessBackend(effective_backend)}
    )


def build_memory_rollout_summary(
    runtime_settings: LLMRuntimeSettings,
    *,
    session: Session | None = None,
    workspace_id: UUID | None = None,
    runtime_root: Path | None = None,
) -> MemoryRolloutSummary:
    manifest_ready = is_rollout_manifest_ready(runtime_root)
    requested_backend = runtime_settings.knowledge_access_backend.value
    define_design_enabled = (
        is_feature_flag_enabled(session, FEATURE_FLAG_MEMORY_HYBRID_DEFINE_DESIGN, workspace_id=workspace_id)
        if session is not None and workspace_id is not None
        else True
    )
    extended_enabled = (
        is_feature_flag_enabled(session, FEATURE_FLAG_MEMORY_HYBRID_EXTENDED_JOURNEY, workspace_id=workspace_id)
        if session is not None and workspace_id is not None
        else True
    )
    effective_default_backend = (
        requested_backend
        if manifest_ready
        else KnowledgeAccessBackend.inline_context.value
    )
    phases = [
        MemoryRolloutPhaseEntry(
            phase_key=PHASE_DEFINE_DESIGN,
            label="Define + Design",
            description="Promocion inicial del contexto gobernado en las etapas de definicion y blueprint.",
            enabled=define_design_enabled,
            stage_keys=["define", "design"],
        ),
        MemoryRolloutPhaseEntry(
            phase_key=PHASE_EXTENDED_JOURNEY,
            label="Tools + Memory + Evaluate + Build",
            description="Extiende la politica de memoria al journey posterior sin enviar contexto redundante.",
            enabled=extended_enabled,
            stage_keys=["tools", "memory", "evaluate", "build"],
        ),
    ]
    stages = [
        MemoryRolloutStageEntry(
            stage_key=spec.stage_key,
            label=spec.label,
            phase_key=spec.phase_key,
            enabled=_phase_enabled(
                phase_key=spec.phase_key,
                define_design_enabled=define_design_enabled,
                extended_enabled=extended_enabled,
            ),
            expects_llm_call=spec.expects_llm_call,
            requested_backend=requested_backend,
            effective_backend=resolve_effective_stage_backend(
                requested_backend,
                stage_key=spec.stage_key,
                manifest_ready=manifest_ready,
                define_design_enabled=define_design_enabled,
                extended_enabled=extended_enabled,
            ),
        )
        for spec in M8_STAGE_SPECS
    ]
    notes: list[str] = []
    if manifest_ready:
        notes.append("El corpus gobernado ya esta listo y permite usar workspace staged como backend por defecto.")
    else:
        notes.append("El corpus gobernado no esta listo; el runtime debe caer a inline_context para evitar staging roto.")
    if not extended_enabled:
        notes.append("La fase extendida sigue bajo rollout controlado y no promueve contexto staged en todo el journey.")

    enabled_stage_count = sum(1 for item in stages if item.enabled)
    if requested_backend == KnowledgeAccessBackend.inline_context.value:
        status = "inline_only"
    elif manifest_ready and enabled_stage_count == len(stages):
        status = "ready"
    elif manifest_ready and enabled_stage_count > 0:
        status = "partial"
    else:
        status = "blocked"

    return MemoryRolloutSummary(
        status=status,
        manifest_ready=manifest_ready,
        requested_backend=requested_backend,
        effective_default_backend=effective_default_backend,
        phases=phases,
        stages=stages,
        notes=notes,
    )
def expected_monitoring_stages(summary: MemoryRolloutSummary) -> list[tuple[str, str]]:
    return [(item.stage_key, item.label) for item in summary.stages if item.enabled]
