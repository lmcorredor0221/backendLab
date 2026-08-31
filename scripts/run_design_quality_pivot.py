from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient
from sqlmodel import Session, select

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings
from app.db import engine
from app.main import app
from app.api.routes import sessions as sessions_routes
from app.models import (
    AgentExecutionBackend,
    CanvasRecord,
    CodexLocalProviderConfig,
    DesignRecommendationArtifact,
    EstimationRunRecord,
    JourneyStageArtifactRecord,
    KnowledgeAccessBackend,
    LLMProviderKey,
    OpportunityRecord,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRole,
)
from app.services.auth_service import hash_password
from app.services.design_recommendation_service import evaluate_design_recommendation_artifact
from app.services.llm_runtime.builder_contracts import RequirementsDefinitionOutput
from app.services.openai_builder import load_llm_runtime_settings
from app.services.product_processing import blueprint_basic_service
from app.services.skill_runtime import validate_definition_artifact

engine.echo = False


DEFAULT_SOURCE_SESSION_ID = UUID("f9454f9e-7ef8-4462-97cc-7fe9d060ead6")
EVIDENCE_ROOT = WORKSPACE_ROOT / "Docs" / "system-analysis" / "evidence" / "design-quality-pivot"
STAGE_ORDER = ("discover", "define", "design", "tools", "memory", "estimate")
STOPWORDS = {
    "actual",
    "agente",
    "antes",
    "arquitectura",
    "cada",
    "como",
    "con",
    "contexto",
    "debe",
    "desde",
    "donde",
    "entre",
    "esta",
    "este",
    "estos",
    "forma",
    "informacion",
    "información",
    "manual",
    "para",
    "permita",
    "proceso",
    "sobre",
    "toda",
    "trazabilidad",
    "trazable",
    "usuario",
    "usuarios",
}


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _opportunity_to_discovery_input(opportunity: OpportunityRecord) -> dict[str, Any]:
    return {
        "problem_statement": opportunity.problem_statement,
        "current_user": opportunity.current_user,
        "current_process": opportunity.current_process,
        "desired_outcome": opportunity.desired_outcome,
        "autonomy_level": opportunity.autonomy_level,
        "constraints": list(opportunity.constraints or []),
        "operational_baseline": dict(opportunity.operational_baseline or {}),
        "mvp_definition": dict(opportunity.mvp_definition or {}),
    }


def _load_source_payload(source_session_id: UUID) -> tuple[SessionRecord, dict[str, Any]]:
    with Session(engine) as db:
        source = db.get(SessionRecord, source_session_id)
        if source is None:
            raise RuntimeError(f"Source session not found: {source_session_id}")
        opportunity = db.exec(
            select(OpportunityRecord).where(OpportunityRecord.session_id == source_session_id)
        ).first()
        if opportunity is None:
            raise RuntimeError(f"Source session has no opportunity input: {source_session_id}")
        return source, _opportunity_to_discovery_input(opportunity)


def _deterministic_runtime_settings():
    runtime = load_llm_runtime_settings()
    return runtime.model_copy(
        update={
            "active_provider": LLMProviderKey.codex_local,
            "agent_execution_backend": AgentExecutionBackend.provider_native,
            "knowledge_access_backend": KnowledgeAccessBackend.inline_context,
            "uses_platform_credentials": False,
            "codex_local": CodexLocalProviderConfig(
                command="codex-disabled-for-quality-pivot",
                model="deterministic-contract-mode",
                available=False,
                executable_found=False,
                status_note="Provider deshabilitado intencionalmente por el pivote de calidad local.",
            ),
        }
    )


@contextmanager
def _runtime_mode(provider_mode: str):
    if provider_mode == "configured":
        yield
        return

    original_loader = sessions_routes.load_effective_runtime_settings
    deterministic = _deterministic_runtime_settings()

    def _load_deterministic_runtime_settings(*args, **kwargs):
        return deterministic

    sessions_routes.load_effective_runtime_settings = _load_deterministic_runtime_settings
    try:
        yield
    finally:
        sessions_routes.load_effective_runtime_settings = original_loader


@contextmanager
def _suppress_blueprint_basic_postprocessing():
    original_prepare = blueprint_basic_service.prepare_blueprint_basic_commercial_result

    def _skip_blueprint_basic_commercial_result(*args, **kwargs):
        return None, None

    blueprint_basic_service.prepare_blueprint_basic_commercial_result = _skip_blueprint_basic_commercial_result
    try:
        yield
    finally:
        blueprint_basic_service.prepare_blueprint_basic_commercial_result = original_prepare


def _ensure_pivot_actor(*, email: str, password: str, source: SessionRecord) -> None:
    with Session(engine) as db:
        user = db.exec(select(UserRecord).where(UserRecord.email == email)).first()
        if user is None:
            user = UserRecord(
                email=email,
                full_name="QA Design Pivot",
                password_hash=hash_password(password),
                default_workspace_id=source.workspace_id,
                preferred_language="es",
            )
            db.add(user)
            db.flush()
        elif email.endswith("@leanbuilder.local"):
            user.password_hash = hash_password(password)
            user.default_workspace_id = source.workspace_id
            user.is_active = True
            db.add(user)
            db.flush()

        membership = db.exec(
            select(WorkspaceMembershipRecord).where(
                WorkspaceMembershipRecord.workspace_id == source.workspace_id,
                WorkspaceMembershipRecord.user_id == user.id,
            )
        ).first()
        if membership is None:
            membership = WorkspaceMembershipRecord(
                workspace_id=source.workspace_id,
                user_id=user.id,
                role=WorkspaceRole.owner,
                is_active=True,
            )
        else:
            membership.role = WorkspaceRole.owner
            membership.is_active = True
        db.add(membership)
        db.commit()


def _artifact_payloads(session_id: UUID) -> dict[str, dict[str, Any]]:
    with Session(engine) as db:
        rows = db.exec(
            select(JourneyStageArtifactRecord)
            .where(JourneyStageArtifactRecord.session_id == session_id)
            .order_by(JourneyStageArtifactRecord.stage_key, JourneyStageArtifactRecord.version_number.desc())
        ).all()
        latest_estimation = db.exec(
            select(EstimationRunRecord)
            .where(EstimationRunRecord.session_id == session_id)
            .order_by(EstimationRunRecord.created_at.desc())
        ).first()
    latest: dict[str, JourneyStageArtifactRecord] = {}
    for row in rows:
        if row.stage_key not in latest:
            latest[row.stage_key] = row
    payloads = {
        stage: {
            "artifact_id": str(row.id),
            "artifact_kind": row.artifact_kind,
            "version_number": row.version_number,
            "state": str(row.state),
            "source_action": row.source_action,
            "provider_key": row.provider_key,
            "model": row.model,
            "confidence": row.confidence,
            "missing_information_count": len(row.missing_information or []),
            "warnings_count": len(row.warnings or []),
            "proposal_payload": row.proposal_payload,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
        for stage, row in latest.items()
    }
    if "estimate" not in payloads and latest_estimation is not None:
        payloads["estimate"] = {
            "artifact_id": str(latest_estimation.id),
            "artifact_kind": "estimation_report_artifact",
            "version_number": 1,
            "state": "persisted",
            "source_action": latest_estimation.source_action,
            "provider_key": latest_estimation.active_provider.value,
            "model": "",
            "confidence": round(float(latest_estimation.confidence_score or 0) / 100, 3),
            "missing_information_count": 0,
            "warnings_count": len((latest_estimation.estimation_payload or {}).get("notes") or []),
            "proposal_payload": latest_estimation.estimation_payload,
            "created_at": latest_estimation.created_at,
            "updated_at": latest_estimation.created_at,
        }
    return payloads


def _latest_artifact_record(db: Session, session_id: UUID, stage_key: str) -> JourneyStageArtifactRecord | None:
    return db.exec(
        select(JourneyStageArtifactRecord)
        .where(
            JourneyStageArtifactRecord.session_id == session_id,
            JourneyStageArtifactRecord.stage_key == stage_key,
        )
        .order_by(JourneyStageArtifactRecord.version_number.desc(), JourneyStageArtifactRecord.updated_at.desc())
    ).first()


def _design_summary(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload.get("selected_design") or {}
    projection = selected.get("blueprint_projection") or {}
    critic_findings = payload.get("critic_findings") or []
    missing_information = payload.get("missing_information") or []
    confidence = _confidence_from_payload("design", payload, None)
    return {
        "review_state": payload.get("review_state"),
        "confidence": confidence,
        "missing_information_count": len(missing_information),
        "critic_findings_count": len(critic_findings),
        "blocking_findings_count": sum(1 for item in critic_findings if item.get("severity") == "blocking"),
        "warning_findings_count": sum(1 for item in critic_findings if item.get("severity") == "warning"),
        "selected_alternative_key": selected.get("alternative_key") or payload.get("recommended_alternative_key"),
        "selected_fit_score": selected.get("fit_score"),
        "handoffs_count": len(selected.get("handoffs") or []),
        "failure_modes_count": len(selected.get("failure_modes") or []),
        "tool_implications_count": len(selected.get("tool_implications") or projection.get("tool_implications") or []),
        "memory_implications_count": len(selected.get("memory_implications") or projection.get("memory_implications") or []),
        "guardrails_count": len(projection.get("guardrails") or []),
        "remediation_summary": payload.get("remediation_summary") or "",
    }


def _source_design_repair_comparison(source_session_id: UUID) -> dict[str, Any]:
    with Session(engine) as db:
        opportunity = db.exec(
            select(OpportunityRecord).where(OpportunityRecord.session_id == source_session_id)
        ).first()
        canvas = db.exec(select(CanvasRecord).where(CanvasRecord.session_id == source_session_id)).first()
        define_record = _latest_artifact_record(db, source_session_id, "define")
        design_record = _latest_artifact_record(db, source_session_id, "design")
        if opportunity is None or canvas is None or define_record is None or design_record is None:
            return {
                "available": False,
                "reason": "Source session is missing opportunity, canvas, define artifact or design artifact.",
            }

        discovery = sessions_routes.hydrate_discovery(opportunity)
        definition = validate_definition_artifact(
            RequirementsDefinitionOutput.model_validate(define_record.proposal_payload)
        )
        before_payload = dict(design_record.proposal_payload or {})
        repaired = evaluate_design_recommendation_artifact(
            DesignRecommendationArtifact.model_validate(before_payload),
            discovery,
            definition,
        )
        after_payload = repaired.model_dump(mode="json")

    before = _design_summary(before_payload)
    after = _design_summary(after_payload)
    return {
        "available": True,
        "source_design_artifact_id": str(design_record.id),
        "before": before,
        "after": after,
        "delta": {
            "confidence": round((after.get("confidence") or 0) - (before.get("confidence") or 0), 3),
            "missing_information_count": int(after.get("missing_information_count") or 0)
            - int(before.get("missing_information_count") or 0),
            "critic_findings_count": int(after.get("critic_findings_count") or 0)
            - int(before.get("critic_findings_count") or 0),
            "blocking_findings_count": int(after.get("blocking_findings_count") or 0)
            - int(before.get("blocking_findings_count") or 0),
            "failure_modes_count": int(after.get("failure_modes_count") or 0)
            - int(before.get("failure_modes_count") or 0),
            "guardrails_count": int(after.get("guardrails_count") or 0)
            - int(before.get("guardrails_count") or 0),
        },
    }


def _payload_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=_json_default).lower()


def _source_terms(discovery_input: dict[str, Any], *, limit: int = 28) -> list[str]:
    seed = _payload_text(discovery_input)
    words = re.findall(r"[a-záéíóúñü]{5,}", seed)
    counts = Counter(word for word in words if word not in STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def _continuity_score(payload: Any, source_terms: list[str]) -> dict[str, Any]:
    text = _payload_text(payload)
    matched = [term for term in source_terms if term in text]
    total = len(source_terms) or 1
    return {
        "score": round(len(matched) / total, 3),
        "matched_terms": matched,
        "missing_terms": [term for term in source_terms if term not in matched],
    }


def _confidence_from_payload(stage: str, payload: dict[str, Any], fallback: float | None) -> float | None:
    confidence = payload.get("confidence")
    if isinstance(confidence, dict):
        value = confidence.get("overall")
        return float(value) if isinstance(value, (int, float)) else fallback
    if isinstance(confidence, (int, float)):
        return float(confidence)
    if stage == "estimate":
        score = payload.get("confidence_score")
        if isinstance(score, (int, float)):
            return round(float(score) / 100, 3)
    return fallback


def _stage_quality(stage: str, artifact: dict[str, Any] | None, source_terms: list[str]) -> dict[str, Any]:
    if artifact is None:
        return {
            "stage": stage,
            "available": False,
            "quality_score": 0.0,
            "confidence": None,
            "missing_information_count": None,
            "warnings_count": None,
            "continuity_score": 0.0,
        }
    payload = artifact["proposal_payload"]
    confidence = _confidence_from_payload(stage, payload, artifact.get("confidence"))
    missing_count = int(artifact.get("missing_information_count") or len(payload.get("missing_information") or []))
    warnings_count = int(artifact.get("warnings_count") or 0)
    continuity = _continuity_score(payload, source_terms)
    structural_bonus = 0.0
    if stage == "define":
        reqs = payload.get("functional_requirements") or []
        questions = payload.get("open_questions") or []
        structural_bonus = min(0.12, (len(reqs) * 0.015) + (len(questions) * 0.005))
    elif stage == "design":
        selected = payload.get("selected_design") or {}
        projection = selected.get("blueprint_projection") or {}
        signals = [
            bool(selected.get("business_fit")),
            bool(selected.get("why_recommended")),
            bool(selected.get("tool_implications") or projection.get("tool_implications")),
            bool(selected.get("memory_implications") or projection.get("memory_implications")),
            bool(selected.get("handoffs")),
            bool(selected.get("failure_modes")),
            bool(projection.get("guardrails")),
        ]
        structural_bonus = round(sum(1 for item in signals if item) / len(signals) * 0.15, 3)
    elif stage == "tools":
        structural_bonus = min(
            0.15,
            ((len(payload.get("recommended_tools") or []) + len(payload.get("optional_tools") or [])) * 0.025),
        )
    elif stage == "memory":
        profile = payload.get("proposed_memory_profile") or {}
        knowledge = payload.get("proposed_knowledge_profile") or {}
        structural_bonus = 0.05 * sum(1 for item in (profile, knowledge, payload.get("tool_dependencies")) if item)
    elif stage == "estimate":
        structural_bonus = 0.15 if payload.get("scenarios") or payload.get("complexity_drivers") else 0.08
    quality_score = (confidence if confidence is not None else 0.55) * 0.65
    quality_score += continuity["score"] * 0.25
    quality_score += structural_bonus
    quality_score -= min(0.25, missing_count * 0.04 + warnings_count * 0.02)
    quality_score = round(max(0.0, min(1.0, quality_score)), 3)
    return {
        "stage": stage,
        "available": True,
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["artifact_kind"],
        "version_number": artifact["version_number"],
        "state": artifact["state"],
        "provider_key": artifact["provider_key"],
        "model": artifact["model"],
        "confidence": confidence,
        "quality_score": quality_score,
        "missing_information_count": missing_count,
        "warnings_count": warnings_count,
        "continuity_score": continuity["score"],
        "continuity_missing_terms": continuity["missing_terms"][:8],
        "payload_chars": len(_payload_text(payload)),
    }


def _quality_report(session_id: UUID, discovery_input: dict[str, Any]) -> dict[str, Any]:
    source_terms = _source_terms(discovery_input)
    artifacts = _artifact_payloads(session_id)
    stages = [_stage_quality(stage, artifacts.get(stage), source_terms) for stage in STAGE_ORDER]
    available = [item for item in stages if item["available"]]
    aggregate = round(sum(item["quality_score"] for item in available) / len(available), 3) if available else 0.0
    return {
        "session_id": str(session_id),
        "source_terms": source_terms,
        "aggregate_quality_score": aggregate,
        "stages": stages,
    }


def _response_payload(response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def _request_json(client: TestClient, method: str, url: str, *, headers: dict[str, str] | None = None, json_body: Any = None) -> Any:
    print(json.dumps({"pivot_request": method, "url": url}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    response = client.request(method, url, headers=headers, json=json_body)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    payload = _response_payload(response)
    print(
        json.dumps(
            {
                "pivot_response": method,
                "url": url,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            json.dumps(
                {
                    "method": method,
                    "url": url,
                    "status_code": response.status_code,
                    "elapsed_ms": elapsed_ms,
                    "payload": payload,
                },
                ensure_ascii=False,
                default=_json_default,
            )
        )
    return payload


def _approve_artifact(
    client: TestClient,
    *,
    headers: dict[str, str],
    session_id: str,
    stage: str,
    artifact: dict[str, Any],
    note: str,
    decision_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _request_json(
        client,
        "POST",
        f"/api/v1/sessions/{session_id}/journey/{stage}/artifacts/{artifact['id']}/approve",
        headers=headers,
        json_body={"note": note, "decision_payload": decision_payload or {}},
    )


def _approve_optional_keys(tool_recommendation: dict[str, Any], mode: str) -> list[str]:
    optional_tools = tool_recommendation.get("data", {}).get("optional_tools") or []
    optional_keys = [str(item.get("tool_key") or "").strip() for item in optional_tools]
    optional_keys = [key for key in optional_keys if key]
    if mode == "all":
        return optional_keys
    if mode == "none":
        return []
    return optional_keys[:1]


def run_pivot(
    source_session_id: UUID,
    *,
    approve_optional: str,
    actor_email: str | None,
    provider_mode: str,
) -> dict[str, Any]:
    settings = get_settings()
    source, discovery_input = _load_source_payload(source_session_id)
    baseline = _quality_report(source_session_id, discovery_input)
    steps: list[dict[str, Any]] = []
    login_email = actor_email or settings.local_admin_email
    login_password = settings.local_admin_password

    with _runtime_mode(provider_mode), _suppress_blueprint_basic_postprocessing(), TestClient(app) as client:
        try:
            login = _request_json(
                client,
                "POST",
                "/api/v1/auth/login",
                json_body={"email": login_email, "password": login_password},
            )
            steps.append({"step": "login", "actor_email": login_email, "mode": "configured_user"})
        except RuntimeError:
            login_email = "qa.design.pivot@leanbuilder.local"
            _ensure_pivot_actor(email=login_email, password=login_password, source=source)
            login = _request_json(
                client,
                "POST",
                "/api/v1/auth/login",
                json_body={"email": login_email, "password": login_password},
            )
            steps.append({"step": "login", "actor_email": login_email, "mode": "qa_actor_created_or_refreshed"})
        headers = {"Authorization": f"Bearer {login['access_token']}"}

        created = _request_json(client, "POST", "/api/v1/sessions", headers=headers)
        pivot_id = created["id"]
        steps.append({"step": "create_session", "session_id": pivot_id})
        title = f"QA pivote Design calidad {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        _request_json(client, "PATCH", f"/api/v1/sessions/{pivot_id}", headers=headers, json_body={"title": title})

        normalize = _request_json(
            client,
            "POST",
            f"/api/v1/sessions/{pivot_id}/normalize-discovery",
            headers=headers,
            json_body=discovery_input,
        )
        steps.append({"step": "normalize_discovery", "status": normalize.get("status")})

        discover_artifact = _request_json(
            client,
            "POST",
            f"/api/v1/sessions/{pivot_id}/analyze-discovery",
            headers=headers,
            json_body=discovery_input,
        )
        steps.append({"step": "analyze_discovery", "artifact_id": discover_artifact["id"]})
        _approve_artifact(
            client,
            headers=headers,
            session_id=pivot_id,
            stage="discover",
            artifact=discover_artifact,
            note="Discover aprobado para pivote de calidad.",
            decision_payload={"approval_reason": "Input fuente replicado para medicion E2E."},
        )

        define_artifact = _request_json(
            client,
            "POST",
            f"/api/v1/sessions/{pivot_id}/define-requirements",
            headers=headers,
        )
        steps.append({"step": "define_requirements", "artifact_id": define_artifact["id"]})
        _approve_artifact(
            client,
            headers=headers,
            session_id=pivot_id,
            stage="define",
            artifact=define_artifact,
            note="Define aprobado para pivote de calidad.",
            decision_payload={"approval_reason": "Definition lista para evaluar Design."},
        )

        design_artifact = _request_json(
            client,
            "POST",
            f"/api/v1/sessions/{pivot_id}/propose-design",
            headers=headers,
        )
        selected_key = (
            design_artifact.get("proposal_payload", {}).get("selected_design", {}).get("alternative_key")
            or design_artifact.get("proposal_payload", {}).get("recommended_alternative_key")
            or ""
        )
        steps.append(
            {
                "step": "propose_design",
                "artifact_id": design_artifact["id"],
                "confidence": design_artifact.get("confidence"),
                "selected_alternative_key": selected_key,
                "missing_information_count": len(design_artifact.get("missing_information") or []),
                "warnings_count": len(design_artifact.get("warnings") or []),
            }
        )
        _approve_artifact(
            client,
            headers=headers,
            session_id=pivot_id,
            stage="design",
            artifact=design_artifact,
            note="Design aprobado para pivote de calidad.",
            decision_payload={"selected_alternative_key": selected_key},
        )

        tools = _request_json(client, "POST", f"/api/v1/sessions/{pivot_id}/recommend-tools", headers=headers)
        optional_keys = _approve_optional_keys(tools, approve_optional)
        steps.append(
            {
                "step": "recommend_tools",
                "recommended_count": len(tools.get("data", {}).get("recommended_tools") or []),
                "optional_count": len(tools.get("data", {}).get("optional_tools") or []),
                "selected_optional_count": len(optional_keys),
            }
        )
        _request_json(
            client,
            "POST",
            f"/api/v1/sessions/{pivot_id}/approve-tools-selection",
            headers=headers,
            json_body={"include_optional_tool_keys": optional_keys},
        )

        memory = _request_json(client, "POST", f"/api/v1/sessions/{pivot_id}/recommend-memory", headers=headers)
        steps.append(
            {
                "step": "recommend_memory",
                "artifact_id": memory["id"],
                "confidence": memory.get("confidence"),
                "missing_information_count": len(memory.get("missing_information") or []),
                "warnings_count": len(memory.get("warnings") or []),
            }
        )
        _request_json(
            client,
            "POST",
            f"/api/v1/sessions/{pivot_id}/approve-memory-profile",
            headers=headers,
            json_body={
                "note": "Memory aprobado para completar el pivote Discovery a Estimate.",
                "decision_payload": {"approval_reason": "Memoria suficiente para estimacion comparativa."},
            },
        )

        estimate = _request_json(client, "POST", f"/api/v1/sessions/{pivot_id}/estimate", headers=headers)
        steps.append(
            {
                "step": "estimate",
                "status": estimate.get("status"),
                "confidence_score": (estimate.get("data") or {}).get("confidence_score"),
            }
        )

    pivot = _quality_report(UUID(pivot_id), discovery_input)
    return {
        "contract_version": "design-quality-pivot-run.v1",
        "generated_at": datetime.now(timezone.utc),
        "source_session_id": str(source.id),
        "source_title": source.title,
        "actor_email": login_email,
        "clone_mode": "discovery_input_replay_to_independent_session",
        "provider_mode": provider_mode,
        "approve_optional_tools_mode": approve_optional,
        "pivot_session_id": pivot["session_id"],
        "steps": steps,
        "baseline": baseline,
        "pivot": pivot,
        "source_design_repair_comparison": _source_design_repair_comparison(source_session_id),
        "delta": {
            "aggregate_quality_score": round(
                pivot["aggregate_quality_score"] - baseline["aggregate_quality_score"],
                3,
            ),
            "design_quality_score": round(
                next(item["quality_score"] for item in pivot["stages"] if item["stage"] == "design")
                - next(item["quality_score"] for item in baseline["stages"] if item["stage"] == "design"),
                3,
            ),
            "design_confidence": round(
                (next(item["confidence"] or 0 for item in pivot["stages"] if item["stage"] == "design"))
                - (next(item["confidence"] or 0 for item in baseline["stages"] if item["stage"] == "design")),
                3,
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crea un proyecto pivote local y ejecuta Discovery -> Estimate para medir calidad de Design."
    )
    parser.add_argument("--source-session-id", type=UUID, default=DEFAULT_SOURCE_SESSION_ID)
    parser.add_argument("--approve-optional-tools", choices=("first", "all", "none"), default="first")
    parser.add_argument("--actor-email", default="")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument(
        "--provider-mode",
        choices=("deterministic", "configured"),
        default="deterministic",
        help=(
            "deterministic deshabilita proveedores LLM dentro del runner para probar contrato/continuidad sin tokens; "
            "configured usa la configuracion real del workspace."
        ),
    )
    args = parser.parse_args()

    result = run_pivot(
        args.source_session_id,
        approve_optional=args.approve_optional_tools,
        actor_email=args.actor_email.strip() or None,
        provider_mode=args.provider_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"run-{_now_stamp()}-{result['pivot_session_id']}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **result["delta"], "pivot_session_id": result["pivot_session_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
