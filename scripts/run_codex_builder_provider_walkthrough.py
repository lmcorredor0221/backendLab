from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


BASE_URL = os.environ.get("LEAN_AGENT_BUILDER_API_URL", "http://127.0.0.1:8000/api/v1")
DEFAULT_EMAIL = os.environ.get("LEAN_AGENT_BUILDER_EMAIL", "admin@leanbuilder.local")
DEFAULT_PASSWORD = os.environ.get("LEAN_AGENT_BUILDER_PASSWORD", "LeanBuilder123!")
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "codex-builder-walkthrough"
TIMEOUT = httpx.Timeout(180.0, connect=10.0)


DISCOVERY_PAYLOAD = {
    "problem_statement": "Coordinar discovery y blueprint con menor retrabajo y continuidad operativa visible.",
    "current_user": "Platform owner",
    "current_process": "El equipo captura contexto, define arquitectura y luego ajusta runtime y rollout a mano.",
    "desired_outcome": "Validar cambio de provider y ejecucion real del builder sin romper el journey.",
    "autonomy_level": "high",
    "constraints": [
        "Sin side effects irreversibles sin aprobacion humana",
        "Sin microservicios en el MVP",
        "Mantener continuidad entre discovery, canvas y blueprint",
    ],
    "operational_baseline": {
        "current_time_spent": "5 horas por ciclo",
        "current_cost": "Retrabajo tecnico y diagnostico manual del runtime",
        "frequent_errors": [
            "Se pierde contexto entre el cambio de provider y la corrida real",
            "No queda claro si Codex esta corriendo como shadow o primary",
        ],
        "automation_opportunities": [
            "Aplicar provider rollout controlado por settings",
            "Verificar ejecucion end-to-end del builder con evidencia",
        ],
    },
    "mvp_definition": {
        "v1_scope": [
            "Capturar discovery estructurado",
            "Construir canvas y blueprint",
            "Validar cambios de provider con evidencia",
        ],
        "out_of_scope": [
            "Subagentes permanentes",
            "Provisioning automatico",
        ],
        "north_star_metric": "Cambiar el path de ejecucion sin romper los artefactos del builder",
        "non_delegable_decisions": [
            "Promover Codex a primary",
        ],
    },
}


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def api_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json_payload: Any | None = None,
    expected_status: int | tuple[int, ...] = 200,
) -> dict[str, Any]:
    response = client.request(method, path, headers=headers, json=json_payload)
    expected = expected_status if isinstance(expected_status, tuple) else (expected_status,)
    if response.status_code not in expected:
        raise RuntimeError(f"{method} {path} fallo con {response.status_code}: {response.text[:1200]}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{method} {path} no devolvio un objeto JSON.")
    return payload


def login(client: httpx.Client) -> dict[str, Any]:
    return api_json(
        client,
        "POST",
        "/auth/login",
        json_payload={"email": DEFAULT_EMAIL, "password": DEFAULT_PASSWORD},
    )


def runtime_payload_from_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "active_provider": payload["active_provider"],
        "agent_execution_backend": payload["agent_execution_backend"],
        "knowledge_access_backend": payload["knowledge_access_backend"],
        "openai": {
            "fast_model": payload["openai"]["fast_model"],
            "reasoning_model": payload["openai"]["reasoning_model"],
            "reasoning_effort": payload["openai"]["reasoning_effort"],
        },
        "deepseek": {
            "base_url": payload["deepseek"]["base_url"],
            "fast_model": payload["deepseek"]["fast_model"],
            "reasoning_model": payload["deepseek"]["reasoning_model"],
            "reasoning_effort": payload["deepseek"]["reasoning_effort"],
        },
        "codex_local": {
            "command": payload["codex_local"]["command"],
            "model": payload["codex_local"]["model"],
            "profile": payload["codex_local"]["profile"],
            "cost_policy": payload["codex_local"]["cost_policy"],
            "timeout_ms": payload["codex_local"]["timeout_ms"],
            "max_concurrency": payload["codex_local"]["max_concurrency"],
            "runner_id": payload["codex_local"]["runner_id"],
            "auth_mode": payload["codex_local"]["auth_mode"],
            "fallback_models": list(payload["codex_local"]["fallback_models"]),
            "primary_agents": list(payload["codex_local"]["primary_agents"]),
            "shadow_agents": list(payload["codex_local"]["shadow_agents"]),
            "staged_agents": list(payload["codex_local"]["staged_agents"]),
        },
    }


def summarize_llm_trace(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("llm_trace")
    if not isinstance(trace, dict):
        return {}
    used_sources = trace.get("context_used_sources")
    source_keys = []
    if isinstance(used_sources, list):
        source_keys = [
            str(item.get("key", "")).strip()
            for item in used_sources
            if isinstance(item, dict) and str(item.get("key", "")).strip()
        ]
    return {
        "provider_key": trace.get("provider_key"),
        "execution_backend": trace.get("execution_backend"),
        "execution_mode": trace.get("execution_mode"),
        "shadow_provider_key": trace.get("shadow_provider_key"),
        "route_reason": trace.get("route_reason"),
        "knowledge_access_backend": trace.get("knowledge_access_backend"),
        "effective_context_backend": trace.get("effective_context_backend"),
        "context_source_keys": source_keys,
        "context_stats": trace.get("context_stats", {}),
    }


def build_mode_payload(base_payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    payload = deepcopy(base_payload)
    if mode == "openai_provider_native":
        payload["active_provider"] = "openai"
        payload["agent_execution_backend"] = "provider_native"
        payload["knowledge_access_backend"] = "inline_context"
        payload["codex_local"]["primary_agents"] = []
        payload["codex_local"]["shadow_agents"] = []
        payload["codex_local"]["staged_agents"] = []
        return payload
    if mode == "openai_shadow_codex":
        payload["active_provider"] = "openai"
        payload["agent_execution_backend"] = "shadow_codex_cli"
        payload["knowledge_access_backend"] = "workspace_staged"
        payload["codex_local"]["primary_agents"] = []
        payload["codex_local"]["shadow_agents"] = [
            "normalize_discovery",
            "build_canvas",
            "build_blueprint",
        ]
        payload["codex_local"]["staged_agents"] = ["synthesize_blueprint_narrative"]
        return payload
    if mode == "codex_primary":
        payload["active_provider"] = "codex_local"
        payload["agent_execution_backend"] = "codex_cli"
        payload["knowledge_access_backend"] = "workspace_staged"
        payload["codex_local"]["primary_agents"] = [
            "normalize_discovery",
            "build_canvas",
            "build_blueprint",
        ]
        payload["codex_local"]["shadow_agents"] = []
        payload["codex_local"]["staged_agents"] = []
        return payload
    raise ValueError(f"Modo no soportado: {mode}")


def run_builder_flow(
    client: httpx.Client,
    *,
    headers: dict[str, str],
    runtime_payload: dict[str, Any],
    output_dir: Path,
    mode: str,
) -> dict[str, Any]:
    applied_runtime = api_json(client, "PATCH", "/runtime/llm", headers=headers, json_payload=runtime_payload)
    runtime_status = api_json(client, "GET", "/runtime/status", headers=headers)
    created_session = api_json(client, "POST", "/sessions", headers=headers, expected_status=201)
    session_id = str(created_session["id"])

    discovery = api_json(
        client,
        "POST",
        f"/sessions/{session_id}/normalize-discovery",
        headers=headers,
        json_payload=DISCOVERY_PAYLOAD,
    )
    canvas = api_json(client, "POST", f"/sessions/{session_id}/build-canvas", headers=headers, json_payload={})
    blueprint = api_json(client, "POST", f"/sessions/{session_id}/build-blueprint", headers=headers, json_payload={})
    snapshot = api_json(client, "GET", f"/sessions/{session_id}", headers=headers)

    if not snapshot.get("discovery") or not snapshot.get("canvas") or not snapshot.get("blueprint"):
        raise RuntimeError(f"El walkthrough {mode} no dejo discovery/canvas/blueprint persistidos.")

    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    write_json(mode_dir / "runtime-settings.json", applied_runtime)
    write_json(mode_dir / "runtime-status.json", runtime_status)
    write_json(mode_dir / "session-created.json", created_session)
    write_json(mode_dir / "normalize-discovery.json", discovery)
    write_json(mode_dir / "build-canvas.json", canvas)
    write_json(mode_dir / "build-blueprint.json", blueprint)
    write_json(mode_dir / "session-final.json", snapshot)

    return {
        "mode": mode,
        "ok": True,
        "session_id": session_id,
        "stage_llm_traces": {
            "normalize_discovery": summarize_llm_trace(discovery),
            "build_canvas": summarize_llm_trace(canvas),
            "build_blueprint": summarize_llm_trace(blueprint),
        },
        "runtime_status": {
            "status": runtime_status.get("status"),
            "smoke_ready": runtime_status.get("smoke_ready"),
            "active_provider": runtime_status.get("active_provider"),
            "selected_as_active_provider": runtime_status.get("selected_as_active_provider"),
            "last_known_result": runtime_status.get("last_known_result"),
        },
    }


def main() -> int:
    output_dir = OUTPUT_ROOT / now_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        auth_payload = login(client)
        token = str(auth_payload["access_token"])
        headers = {"Authorization": f"Bearer {token}"}
        original_runtime = api_json(client, "GET", "/runtime/llm", headers=headers)
        base_payload = runtime_payload_from_response(original_runtime)

        results: list[dict[str, Any]] = []
        restored = False
        try:
            for mode in ("openai_provider_native", "openai_shadow_codex", "codex_primary"):
                mode_payload = build_mode_payload(base_payload, mode=mode)
                results.append(
                    run_builder_flow(
                        client,
                        headers=headers,
                        runtime_payload=mode_payload,
                        output_dir=output_dir,
                        mode=mode,
                    )
                )
        finally:
            api_json(client, "PATCH", "/runtime/llm", headers=headers, json_payload=base_payload)
            restored = True

    latest_dir = OUTPUT_ROOT / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": BASE_URL,
        "evidence_dir": str(output_dir),
        "results": results,
        "restored_original_runtime": restored,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(latest_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
