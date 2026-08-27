from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


BASE_URL = os.environ.get("LEAN_AGENT_BUILDER_API_URL", "http://127.0.0.1:8000/api/v1")
HEALTH_URL = BASE_URL.removesuffix("/api/v1") + "/health"
DEFAULT_EMAIL = os.environ.get("LEAN_AGENT_BUILDER_EMAIL", "admin@leanbuilder.local")
DEFAULT_PASSWORD = os.environ.get("LEAN_AGENT_BUILDER_PASSWORD", "LeanBuilder123!")
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "release-stage6"
TIMEOUT = httpx.Timeout(120.0, connect=10.0)


DISCOVERY_PAYLOAD = {
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


ANSWER_MAP = {
    "knowledge_sources": "name=Confluence; type=wiki; owner=ops; frequency=diaria",
    "knowledge_ingestion": "strategy=sync_incremental; frequency=diaria; mechanism=cron; owner=ops",
    "knowledge_embedding_strategy": "provider=text-embedding-3-small; chunking=800_tokens_overlap_120; notes=openai",
    "runtime_fallback_model": "model=gpt-4.1-mini; condition=cuando falle el modelo primario",
    "runtime_vector_store": "vector_store=pgvector; notes=misma base local",
    "runtime_secret_source": "source=.env local protegido; owner=platform_owner; environment=desarrollo",
    "deployment_target": "target=local_vm; restrictions=solo red interna y acceso por VPN",
    "deployment_image_strategy": "strategy=local_windows_service; registry=no_aplica",
    "deployment_network_constraints": "network=solo red interna; secrets=.env local; dependencies=postgres local",
}


def now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def log_step(message: str) -> None:
    print(f"[release-check] {message}", flush=True)


def sanitize_health_payload(payload: dict[str, Any]) -> dict[str, Any]:
    llm = payload.get("llm")
    if isinstance(llm, dict):
        payload = dict(payload)
        payload["llm"] = {
            key: value
            for key, value in llm.items()
            if key in {"provider", "mode", "configured", "sdk_ready", "fast_model", "reasoning_model"}
        }
    return payload


def collect_warning_strings(payload: dict[str, Any]) -> list[str]:
    warnings = payload.get("warnings", [])
    return [str(item) for item in warnings] if isinstance(warnings, list) else []


def collect_evidence_details(payload: dict[str, Any]) -> list[str]:
    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        return []
    details: list[str] = []
    for item in evidence:
        if isinstance(item, dict) and item.get("detail"):
            details.append(str(item["detail"]))
    return details


def assert_ok(response: httpx.Response, expected_status: int | tuple[int, ...], step: str) -> None:
    expected = expected_status if isinstance(expected_status, tuple) else (expected_status,)
    if response.status_code in expected:
        return
    snippet = response.text[:1000]
    raise RuntimeError(f"{step} fallo con {response.status_code}: {snippet}")


def api_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json_payload: Any | None = None,
    expected_status: int | tuple[int, ...] = 200,
) -> dict[str, Any] | list[Any]:
    response = client.request(method, path, headers=headers, json=json_payload)
    assert_ok(response, expected_status, f"{method} {path}")
    return response.json()


def api_text(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    expected_status: int | tuple[int, ...] = 200,
) -> str:
    response = client.request(method, path, headers=headers)
    assert_ok(response, expected_status, f"{method} {path}")
    return response.text


def api_binary(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    expected_status: int | tuple[int, ...] = 200,
) -> tuple[bytes, httpx.Headers]:
    response = client.request(method, path, headers=headers)
    assert_ok(response, expected_status, f"{method} {path}")
    return response.content, response.headers


def login(client: httpx.Client) -> dict[str, Any]:
    payload = api_json(
        client,
        "POST",
        "/auth/login",
        json_payload={"email": DEFAULT_EMAIL, "password": DEFAULT_PASSWORD},
    )
    if not isinstance(payload, dict) or "access_token" not in payload:
        raise RuntimeError("La autenticacion no devolvio access_token.")
    return payload


def resolve_first_approval(client: httpx.Client, headers: dict[str, str], session_id: str) -> dict[str, Any]:
    snapshot = api_json(client, "GET", f"/sessions/{session_id}", headers=headers)
    if not isinstance(snapshot, dict):
        raise RuntimeError("La sesion no devolvio snapshot JSON.")
    approvals = snapshot.get("approvals", [])
    if not isinstance(approvals, list) or not approvals:
        raise RuntimeError("La sesion no expone approval gates para resolver.")
    approval_id = approvals[0]["id"]
    return api_json(
        client,
        "POST",
        f"/sessions/{session_id}/approvals/{approval_id}/resolve",
        headers=headers,
        json_payload={
            "decision": "approved",
            "resolution_note": "Blueprint autorizado para release local verificable.",
        },
    )


def upgrade_session_tier(client: httpx.Client, headers: dict[str, str], session_id: str, *, tier: str = "acp") -> dict[str, Any]:
    payload = api_json(
        client,
        "PATCH",
        f"/sessions/{session_id}/commercial-tier",
        headers=headers,
        json_payload={"tier": tier},
    )
    if not isinstance(payload, dict):
        raise RuntimeError("La actualizacion comercial no devolvio un objeto JSON.")
    resolved_tier = str((((payload.get("commercial_access") or {}) if isinstance(payload, dict) else {}).get("tier")) or "")
    if resolved_tier != tier:
        raise RuntimeError(f"La sesion no actualizo al tier esperado '{tier}': {payload}")
    return payload


def answer_questions(client: httpx.Client, headers: dict[str, str], session_id: str, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answered: list[dict[str, Any]] = []
    for question in questions:
        question_key = str(question["question_key"])
        payload = {
            "answer_text": ANSWER_MAP.get(
                question_key,
                f"resolved={question_key}; source=release_local_validation; owner=platform_owner",
            ),
            "owner_role": "platform_owner",
            "impacted_artifacts": question.get("impacted_artifacts", []) if isinstance(question, dict) else [],
        }
        response = api_json(
            client,
            "PATCH",
            f"/sessions/{session_id}/acp/questions/{question_key}",
            headers=headers,
            json_payload=payload,
        )
        if not isinstance(response, dict):
            raise RuntimeError(f"La pregunta {question_key} no devolvio un objeto JSON.")
        answered.append(response)
    return answered


def wait_for_estimate_completion(
    client: httpx.Client,
    headers: dict[str, str],
    session_id: str,
    *,
    timeout_seconds: float = 420.0,
    poll_interval_seconds: float = 5.0,
) -> dict[str, Any]:
    started_operation = api_json(
        client,
        "POST",
        f"/sessions/{session_id}/estimate/start",
        headers=headers,
        json_payload={},
        expected_status=202,
    )
    if not isinstance(started_operation, dict):
        raise RuntimeError("Estimate start no devolvio una operacion JSON.")

    deadline = time.monotonic() + timeout_seconds
    last_operation: dict[str, Any] = started_operation
    while time.monotonic() < deadline:
        try:
            operation = api_json(
                client,
                "GET",
                f"/sessions/{session_id}/stage-operations/current?stage_key=estimate&action=generate_estimation_report",
                headers=headers,
            )
        except httpx.HTTPError:
            operation = last_operation
        session_snapshot = None
        try:
            snapshot_candidate = api_json(client, "GET", f"/sessions/{session_id}", headers=headers)
            if isinstance(snapshot_candidate, dict):
                session_snapshot = snapshot_candidate
        except httpx.HTTPError:
            session_snapshot = None
        if isinstance(operation, dict):
            last_operation = operation
            status = str(operation.get("status") or "")
            if session_snapshot is not None:
                estimation_report = session_snapshot.get("estimation_report")
                if isinstance(estimation_report, dict) and estimation_report:
                    return {
                        "operation": operation,
                        "estimation_report": estimation_report,
                        "session": session_snapshot.get("session"),
                    }
            if status == "completed":
                if session_snapshot is None:
                    raise RuntimeError("Estimate completo sin snapshot accesible para validar persistencia.")
                estimation_report = session_snapshot.get("estimation_report")
                if not isinstance(estimation_report, dict):
                    raise RuntimeError("Estimate completo sin estimation_report persistido.")
                return {
                    "operation": operation,
                    "estimation_report": estimation_report,
                    "session": session_snapshot.get("session"),
                }
            if status in {"failed", "cancelled", "expired"}:
                raise RuntimeError(f"Estimate termino en estado terminal inesperado: {operation}")
        time.sleep(poll_interval_seconds)

    raise RuntimeError(f"Estimate no completo dentro del timeout de {timeout_seconds} segundos: {last_operation}")


def build_markdown_summary(
    *,
    session_id: str,
    output_dir: Path,
    health_payload: dict[str, Any],
    openai_observations: list[str],
    readiness: dict[str, Any],
    validation: dict[str, Any],
    zip_filename: str,
) -> str:
    lines = [
        "# Release local verificable",
        "",
        f"- Fecha UTC: {datetime.now(UTC).isoformat()}",
        f"- Session ID: `{session_id}`",
        f"- API base: `{BASE_URL}`",
        f"- Carpeta de evidencia: `{output_dir}`",
        "",
        "## Stack confirmado",
        "",
        f"- Health: `{health_payload.get('status', 'unknown')}`",
        f"- LLM provider: `{health_payload.get('llm', {}).get('provider', 'unknown')}`",
        f"- LLM mode: `{health_payload.get('llm', {}).get('mode', 'unknown')}`",
        f"- OpenAI configured: `{health_payload.get('llm', {}).get('configured', False)}`",
        f"- OpenAI sdk_ready: `{health_payload.get('llm', {}).get('sdk_ready', False)}`",
        "",
        "## Validacion del flujo",
        "",
        f"- Construction readiness: `{readiness.get('overall_status', 'unknown')}`",
        f"- Can start build: `{readiness.get('can_start_build', False)}`",
        f"- Blocking gaps: `{readiness.get('blocking_gaps', 'unknown')}`",
        f"- Open questions: `{readiness.get('open_questions', 'unknown')}`",
        f"- ACP completeness: `{validation.get('completeness_percent', 'unknown')}`",
        f"- ACP exportable: `{validation.get('can_export_zip', False)}`",
        f"- ZIP export: `{zip_filename}`",
        "",
        "## Evidencia OpenAI",
        "",
    ]
    if openai_observations:
        lines.extend([f"- {item}" for item in openai_observations])
    else:
        lines.append("- No se detectaron trazas explicitas de OpenAI en los envelopes; revisar warnings/evidence JSON.")
    lines.extend(
        [
            "",
            "## Artefactos guardados",
            "",
            "- `health.json`",
            "- `session-final.json`",
            "- `integrations-check.json`",
            "- `discovery-normalized.json`",
            "- `canvas-built.json`",
            "- `blueprint-built.json`",
            "- `evaluation.json`",
            "- `estimate.json`",
            "- `estimate-operation.json`",
            "- `acp-preview-initial.json`",
            "- `acp-preview-final.json`",
            "- `acp-readiness-final.json`",
            "- `acp-questions-final.json`",
            "- `export.json`",
            "- `export.md`",
            f"- `{zip_filename}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    run_dir = OUTPUT_ROOT / now_slug()
    run_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        log_step("Validando salud del backend")
        health_response = client.get(HEALTH_URL)
        assert_ok(health_response, 200, f"GET {HEALTH_URL}")
        health_payload = sanitize_health_payload(health_response.json())
        write_json(run_dir / "health.json", health_payload)

        log_step("Autenticando usuario local")
        auth_payload = login(client)
        token = str(auth_payload["access_token"])
        headers = {"Authorization": f"Bearer {token}"}
        write_json(
            run_dir / "auth-context.json",
            {
                "user": auth_payload.get("user"),
                "session_hint": "access_token redacted",
            },
        )

        log_step("Creando sesion limpia para la validacion")
        created_session = api_json(client, "POST", "/sessions", headers=headers, expected_status=201)
        if not isinstance(created_session, dict) or "id" not in created_session:
            raise RuntimeError("La creacion de sesion no devolvio id.")
        session_id = str(created_session["id"])
        write_json(run_dir / "session-created.json", created_session)

        log_step("Construyendo discovery, canvas y blueprint")
        normalize_payload = api_json(
            client,
            "POST",
            f"/sessions/{session_id}/normalize-discovery",
            headers=headers,
            json_payload=DISCOVERY_PAYLOAD,
        )
        canvas_payload = api_json(client, "POST", f"/sessions/{session_id}/build-canvas", headers=headers, json_payload={})
        blueprint_payload = api_json(
            client,
            "POST",
            f"/sessions/{session_id}/build-blueprint",
            headers=headers,
            json_payload={},
        )
        write_json(run_dir / "discovery-normalized.json", normalize_payload)
        write_json(run_dir / "canvas-built.json", canvas_payload)
        write_json(run_dir / "blueprint-built.json", blueprint_payload)

        log_step("Resolviendo aprobacion inicial del blueprint")
        approval_resolution = resolve_first_approval(client, headers, session_id)
        write_json(run_dir / "approval-resolution.json", approval_resolution)

        log_step("Preparando evaluacion base")
        bootstrap_payload = api_json(
            client,
            "POST",
            f"/sessions/{session_id}/evaluation/bootstrap",
            headers=headers,
            json_payload={},
        )
        evaluation_payload = api_json(
            client,
            "POST",
            f"/sessions/{session_id}/evaluate",
            headers=headers,
            json_payload={},
        )
        log_step("Calculando estimate antes de ACP")
        estimate_payload = wait_for_estimate_completion(client, headers, session_id)
        log_step("Habilitando tier ACP despues de estimate")
        tier_upgrade = upgrade_session_tier(client, headers, session_id, tier="acp")
        write_json(run_dir / "evaluation-bootstrap.json", bootstrap_payload)
        write_json(run_dir / "evaluation.json", evaluation_payload)
        write_json(run_dir / "estimate.json", estimate_payload)
        write_json(run_dir / "estimate-operation.json", estimate_payload["operation"])
        write_json(run_dir / "tier-upgrade.json", tier_upgrade)

        log_step("Generando vista inicial del paquete ACP")
        preview_initial = api_json(client, "GET", f"/sessions/{session_id}/acp/preview", headers=headers)
        generated_initial = api_json(
            client,
            "POST",
            f"/sessions/{session_id}/acp/generate",
            headers=headers,
            json_payload={},
        )
        readiness_initial = api_json(client, "GET", f"/sessions/{session_id}/acp/construction-readiness", headers=headers)
        questions_initial = api_json(client, "GET", f"/sessions/{session_id}/acp/questions", headers=headers)
        graph_payload = api_json(client, "GET", f"/sessions/{session_id}/acp/knowledge-graph", headers=headers)
        write_json(run_dir / "acp-preview-initial.json", preview_initial)
        write_json(run_dir / "acp-generated-initial.json", generated_initial)
        write_json(run_dir / "acp-readiness-initial.json", readiness_initial)
        write_json(run_dir / "acp-questions-initial.json", questions_initial)
        write_json(run_dir / "acp-knowledge-graph.json", graph_payload)

        if not isinstance(questions_initial, list):
            raise RuntimeError("Las preguntas ACP no devolvieron una lista.")
        log_step(f"Resolviendo {len(questions_initial)} preguntas ACP")
        answered_questions = answer_questions(client, headers, session_id, questions_initial)
        write_json(run_dir / "acp-questions-answered.json", answered_questions)

        log_step("Regenerando ACP con respuestas aplicadas")
        preview_final = api_json(
            client,
            "POST",
            f"/sessions/{session_id}/acp/generate",
            headers=headers,
            json_payload={},
        )
        readiness_final = api_json(client, "GET", f"/sessions/{session_id}/acp/construction-readiness", headers=headers)
        questions_final = api_json(client, "GET", f"/sessions/{session_id}/acp/questions", headers=headers)
        validation_final = api_json(client, "GET", f"/sessions/{session_id}/acp/validate", headers=headers)
        artifacts_payload = api_json(client, "GET", f"/sessions/{session_id}/artifacts", headers=headers)
        integrations_payload = api_json(
            client,
            "POST",
            f"/sessions/{session_id}/integrations/check",
            headers=headers,
            json_payload={},
        )
        session_final = api_json(client, "GET", f"/sessions/{session_id}", headers=headers)
        write_json(run_dir / "acp-preview-final.json", preview_final)
        write_json(run_dir / "acp-readiness-final.json", readiness_final)
        write_json(run_dir / "acp-questions-final.json", questions_final)
        write_json(run_dir / "acp-validation-final.json", validation_final)
        write_json(run_dir / "artifacts.json", artifacts_payload)
        write_json(run_dir / "integrations-check.json", integrations_payload)
        write_json(run_dir / "session-final.json", session_final)

        log_step("Exportando markdown, JSON y ZIP finales")
        export_markdown = api_text(client, "GET", f"/sessions/{session_id}/export/markdown", headers=headers)
        export_json = api_json(client, "GET", f"/sessions/{session_id}/export/json", headers=headers)
        zip_bytes, zip_headers = api_binary(client, "GET", f"/sessions/{session_id}/acp/export.zip", headers=headers)
        zip_filename = "acp-export.zip"
        disposition = zip_headers.get("content-disposition", "")
        if "filename=" in disposition:
            zip_filename = disposition.split("filename=", maxsplit=1)[1].strip().strip('"')
        write_text(run_dir / "export.md", export_markdown)
        write_json(run_dir / "export.json", export_json)
        (run_dir / zip_filename).write_bytes(zip_bytes)

        if not isinstance(readiness_final, dict) or not isinstance(validation_final, dict):
            raise RuntimeError("La validacion final ACP no devolvio objetos JSON.")
        if readiness_final.get("overall_status") != "ready_to_build":
            raise RuntimeError(f"ACP final no quedo listo para build: {readiness_final}")
        if readiness_final.get("blocking_gaps") != 0 or readiness_final.get("open_questions") != 0:
            raise RuntimeError(f"ACP final mantiene gaps abiertos: {readiness_final}")
        if validation_final.get("can_export_zip") is not True:
            raise RuntimeError(f"ACP final no permite export ZIP: {validation_final}")
        if zip_bytes[:2] != b"PK":
            raise RuntimeError("El archivo ZIP exportado no tiene firma PK.")

        openai_observations: list[str] = []
        for artifact_name, payload in (
            ("discovery", normalize_payload),
            ("canvas", canvas_payload),
            ("blueprint", blueprint_payload),
        ):
            if not isinstance(payload, dict):
                continue
            warnings = collect_warning_strings(payload)
            evidence_details = collect_evidence_details(payload)
            used_openai = any("OpenAI" in item for item in evidence_details)
            fallback_warning = any("fallback" in item.lower() or "no pudo" in item.lower() for item in warnings)
            if used_openai and not fallback_warning:
                openai_observations.append(f"{artifact_name}: Structured Outputs activo sin fallback.")
            elif used_openai:
                openai_observations.append(f"{artifact_name}: OpenAI reportado con warning/fallback -> {warnings}.")
            elif warnings:
                openai_observations.append(f"{artifact_name}: sin evidencia OpenAI; warnings={warnings}.")

        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "base_url": BASE_URL,
            "output_dir": str(run_dir),
            "health": health_payload,
            "readiness": readiness_final,
            "validation": validation_final,
            "zip_filename": zip_filename,
            "openai_observations": openai_observations,
        }
        write_json(run_dir / "release-summary.json", summary)
        write_text(
            run_dir / "release-summary.md",
            build_markdown_summary(
                session_id=session_id,
                output_dir=run_dir,
                health_payload=health_payload,
                openai_observations=openai_observations,
                readiness=readiness_final,
                validation=validation_final,
                zip_filename=zip_filename,
            ),
        )

        latest_dir = OUTPUT_ROOT / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            latest_dir / "session-context.json",
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "session_id": session_id,
                "base_url": BASE_URL,
                "release_run_dir": str(run_dir),
            },
        )
        write_text(latest_dir / "latest-run.txt", str(run_dir))

        log_step(f"Release local verificado. Session ID: {session_id}")
        log_step(f"Evidencia: {run_dir}")


if __name__ == "__main__":
    main()
