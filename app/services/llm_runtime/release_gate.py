from __future__ import annotations

from typing import Any

from app.models import LLMRuntimeSettings


REQUIRED_RELEASE_CHECKS = ("migration", "smoke", "queue", "fallback", "timeout", "walkthrough")


def build_rollout_counts(runtime_settings: LLMRuntimeSettings) -> dict[str, int]:
    primary = len(runtime_settings.codex_local.primary_agents)
    shadow = len(runtime_settings.codex_local.shadow_agents)
    staged = len(runtime_settings.codex_local.staged_agents)
    return {
        "primary": primary,
        "shadow": shadow,
        "staged": staged,
        "total": primary + shadow + staged,
    }


def detect_rollout_stage(runtime_settings: LLMRuntimeSettings) -> str:
    rollout = build_rollout_counts(runtime_settings)
    if (
        runtime_settings.active_provider.value == "codex_local"
        or runtime_settings.agent_execution_backend.value == "codex_cli"
        or rollout["primary"] > 0
    ):
        return "primary"
    if (
        runtime_settings.agent_execution_backend.value == "shadow_codex_cli"
        or rollout["shadow"] > 0
        or rollout["staged"] > 0
    ):
        return "shadow"
    return "disabled"


def _build_requirement(key: str, label: str, passed: bool) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "passed": passed,
    }


def _build_gate(stage: str, summary: str, requirements: list[dict[str, Any]], *, current_stage: str) -> dict[str, Any]:
    gate_passed = all(item["passed"] for item in requirements)
    status = "active" if stage == current_stage and gate_passed else "eligible" if gate_passed else "blocked"
    return {
        "stage": stage,
        "status": status,
        "summary": summary,
        "requirements": requirements,
    }


def evaluate_release_gate(
    runtime_settings: LLMRuntimeSettings,
    runtime_status: dict[str, Any],
    checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rollout = build_rollout_counts(runtime_settings)
    current_stage = detect_rollout_stage(runtime_settings)
    smoke_ready = bool(runtime_status.get("smoke_ready"))
    last_known_result = runtime_status.get("last_known_result") or {}
    last_known_status = str(last_known_result.get("status", "")).strip().lower()
    failed_checks = [
        name for name in REQUIRED_RELEASE_CHECKS if not bool((checks.get(name) or {}).get("ok"))
    ]
    release_checks_ok = not failed_checks

    disabled_gate = _build_gate(
        "disabled",
        "Codex queda fuera del trafico de builder; solo diagnostico, smoke y regresiones.",
        [
            _build_requirement("smoke_ready", "El runtime Codex pasa readiness.", smoke_ready),
            _build_requirement(
                "release_checks",
                "Migration, smoke, queue, fallback, timeout y walkthrough pasan.",
                release_checks_ok,
            ),
            _build_requirement(
                "provider_native_backend",
                "El backend agentico sigue en provider_native.",
                runtime_settings.agent_execution_backend.value == "provider_native",
            ),
            _build_requirement(
                "non_codex_active_provider",
                "El provider activo visible no es codex_local.",
                runtime_settings.active_provider.value != "codex_local",
            ),
            _build_requirement(
                "no_rollout_agents",
                "No hay capacidades promovidas a primary/shadow/staged.",
                rollout["total"] == 0,
            ),
        ],
        current_stage=current_stage,
    )
    shadow_gate = _build_gate(
        "shadow",
        "Codex observa o acompana capacidades sin recibir el path principal.",
        [
            _build_requirement("smoke_ready", "El runtime Codex pasa readiness.", smoke_ready),
            _build_requirement(
                "release_checks",
                "Migration, smoke, queue, fallback, timeout y walkthrough pasan.",
                release_checks_ok,
            ),
            _build_requirement(
                "shadow_backend",
                "El backend agentico esta en shadow_codex_cli.",
                runtime_settings.agent_execution_backend.value == "shadow_codex_cli",
            ),
            _build_requirement(
                "non_codex_active_provider",
                "El provider activo visible aun no es codex_local.",
                runtime_settings.active_provider.value != "codex_local",
            ),
            _build_requirement(
                "shadow_or_staged_rollout",
                "Existe al menos una capacidad en shadow o staged.",
                rollout["shadow"] > 0 or rollout["staged"] > 0,
            ),
            _build_requirement(
                "no_primary_rollout",
                "No existen capacidades primary mientras se esta en shadow.",
                rollout["primary"] == 0,
            ),
        ],
        current_stage=current_stage,
    )
    primary_gate = _build_gate(
        "primary",
        "Codex queda habilitado como backend principal o provider activo para capacidades reales.",
        [
            _build_requirement("smoke_ready", "El runtime Codex pasa readiness.", smoke_ready),
            _build_requirement(
                "release_checks",
                "Migration, smoke, queue, fallback, timeout y walkthrough pasan.",
                release_checks_ok,
            ),
            _build_requirement(
                "primary_backend_or_provider",
                "El stage primario usa codex_cli o el provider activo visible ya es codex_local.",
                runtime_settings.agent_execution_backend.value == "codex_cli"
                or runtime_settings.active_provider.value == "codex_local",
            ),
            _build_requirement(
                "primary_rollout_or_active_provider",
                "Existe capacidad primary o codex_local ya es el provider visible activo.",
                rollout["primary"] > 0 or runtime_settings.active_provider.value == "codex_local",
            ),
            _build_requirement(
                "last_known_success",
                "La ultima corrida conocida del runtime termino en succeeded.",
                last_known_status == "succeeded",
            ),
        ],
        current_stage=current_stage,
    )

    transition_gates = [
        {
            "from": "disabled",
            "to": "shadow",
            "status": "ready"
            if current_stage == "disabled" and smoke_ready and release_checks_ok
            else "blocked",
            "requirements": [
                _build_requirement(
                    "current_disabled",
                    "La configuracion actual esta en disabled.",
                    current_stage == "disabled",
                ),
                _build_requirement("smoke_ready", "El runtime Codex pasa readiness.", smoke_ready),
                _build_requirement(
                    "release_checks",
                    "Migration, smoke, queue, fallback, timeout y walkthrough pasan.",
                    release_checks_ok,
                ),
            ],
        },
        {
            "from": "shadow",
            "to": "primary",
            "status": "ready"
            if current_stage == "shadow" and smoke_ready and release_checks_ok and last_known_status == "succeeded"
            else "blocked",
            "requirements": [
                _build_requirement(
                    "current_shadow",
                    "La configuracion actual ya esta en shadow.",
                    current_stage == "shadow",
                ),
                _build_requirement("smoke_ready", "El runtime Codex pasa readiness.", smoke_ready),
                _build_requirement(
                    "release_checks",
                    "Migration, smoke, queue, fallback, timeout y walkthrough pasan.",
                    release_checks_ok,
                ),
                _build_requirement(
                    "last_known_success",
                    "La ultima corrida conocida termino en succeeded.",
                    last_known_status == "succeeded",
                ),
            ],
        },
    ]

    current_gate = {
        "disabled": disabled_gate,
        "shadow": shadow_gate,
        "primary": primary_gate,
    }[current_stage]

    return {
        "current_stage": current_stage,
        "failed_checks": failed_checks,
        "release_checks_ok": release_checks_ok,
        "rollout_counts": rollout,
        "stages": [disabled_gate, shadow_gate, primary_gate],
        "transitions": transition_gates,
        "overall_ok": current_gate["status"] == "active" and release_checks_ok,
    }
