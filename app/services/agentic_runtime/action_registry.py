from __future__ import annotations

from dataclasses import dataclass

from app.services.agentic_runtime.contracts import BuilderActionRequest
from app.services.agentic_runtime.stage_policy import get_stage_agent_policy


class BuilderActionRejectedError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class BuilderActionDefinition:
    key: str
    description: str
    side_effect: bool = False
    requires_capability_stage_match: bool = False


CAPABILITY_STAGE_MAP = {
    "normalize_discovery": "discover",
    "discovery_analysis": "discover",
    "build_canvas": "define",
    "define_requirements": "define",
    "propose_agent_design": "design",
    "critique_agent_design": "design",
    "recommend_minimal_tools": "tools",
    "recommend_memory_architecture": "memory",
    "critique_memory_architecture": "memory",
    "generate_validation_scenarios": "validate",
    "simulate_validation_scenario": "validate",
    "judge_validation_run": "validate",
    "analyze_estimation_risks": "estimate",
    "generate_acp_preview": "package",
}


class BuilderActionRegistry:
    def __init__(self, definitions: list[BuilderActionDefinition] | None = None):
        self._definitions = {item.key: item for item in definitions or self.default_definitions()}

    @staticmethod
    def default_definitions() -> list[BuilderActionDefinition]:
        return [
            BuilderActionDefinition("retrieve_context", "Recuperar contexto aprobado y memoria compacta."),
            BuilderActionDefinition("invoke_capability", "Invocar una capability gobernada del builder.", requires_capability_stage_match=True),
            BuilderActionDefinition("invoke_critique", "Ejecutar la critica gobernada de una propuesta.", requires_capability_stage_match=True),
            BuilderActionDefinition("run_validator", "Validar estructura, consistencia y alcance de la salida."),
            BuilderActionDefinition("repair_structured_output", "Reparar un payload estructurado recuperable."),
            BuilderActionDefinition("create_attention_decision", "Crear una decision HITL guiada."),
            BuilderActionDefinition("raise_cross_stage_remediation", "Elevar una remediacion a la etapa que origino la dependencia."),
            BuilderActionDefinition("persist_stage_artifact", "Persistir un artefacto versionado de la etapa.", side_effect=True),
            BuilderActionDefinition("finish_stage", "Cerrar la ejecucion de la etapa.", side_effect=True),
            BuilderActionDefinition("checkpoint", "Persistir un checkpoint de ejecucion.", side_effect=True),
            BuilderActionDefinition("resume_from_checkpoint", "Reanudar una ejecucion desde un checkpoint.", side_effect=True),
        ]

    def get(self, key: str) -> BuilderActionDefinition:
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise BuilderActionRejectedError("unknown_action", f"Accion ReAct desconocida: {key}") from exc

    def assert_allowed(self, request: BuilderActionRequest) -> None:
        definition = self.get(request.key)
        policy = get_stage_agent_policy(request.stage)
        if not policy.allows(request.key):
            raise BuilderActionRejectedError(
                "stage_action_forbidden",
                f"La accion '{request.key}' no esta permitida en la etapa '{request.stage}'.",
            )
        if definition.requires_capability_stage_match:
            capability_stage = CAPABILITY_STAGE_MAP.get(request.capability)
            if capability_stage != request.stage:
                raise BuilderActionRejectedError(
                    "capability_stage_forbidden",
                    f"La capability '{request.capability}' no pertenece a la etapa '{request.stage}'.",
                )
        if definition.side_effect and not request.idempotency_key.strip():
            raise BuilderActionRejectedError(
                "idempotency_required",
                f"La accion con side effect '{request.key}' requiere idempotency_key.",
            )

    def list(self) -> list[BuilderActionDefinition]:
        return list(self._definitions.values())
