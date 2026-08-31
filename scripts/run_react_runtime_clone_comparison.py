from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import MetaData, delete, func, select as sa_select
from sqlmodel import Session, select

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings
from app.db import engine
from app.main import app
from app.models import (
    ArtifactStatus,
    CanvasRecord,
    ExecutionLogRecord,
    JourneyStageArtifactRecord,
    LLMUsageLedgerRecord,
    OpportunityRecord,
    RuntimeFeatureFlagRecord,
    SessionRecord,
    ShortTermCheckpointRecord,
    SkillRunRecord,
    StageOperationRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceProviderSecretRecord,
    WorkspaceRecord,
    WorkspaceRole,
    WorkspaceRuntimeSettingsRecord,
    utc_now,
)
from app.services.auth_service import hash_password
from app.services.stage5_service import update_feature_flag
from app.services.workspace_bootstrap import apply_workspace_bootstrap, seed_runtime_feature_flags
from scripts.run_design_quality_pivot import (
    STAGE_ORDER,
    _approve_artifact,
    _approve_optional_keys,
    _artifact_payloads,
    _json_default,
    _opportunity_to_discovery_input,
    _quality_report,
    _request_json,
    _source_terms,
    _suppress_blueprint_basic_postprocessing,
)

engine.echo = False


DEFAULT_SOURCE_SESSION_ID = UUID("a0028176-36a7-44fe-870d-a6bd6af7ad52")
DEFAULT_REACT_OFF_SESSION_ID = UUID("11111111-2222-4630-8000-0000000146f0")
DEFAULT_REACT_ON_SESSION_ID = UUID("11111111-2222-4630-8000-0000000146f1")
DEFAULT_REACT_OFF_WORKSPACE_ID = UUID("22222222-2222-4630-8000-0000000146f0")
DEFAULT_REACT_ON_WORKSPACE_ID = UUID("22222222-2222-4630-8000-0000000146f1")
EVIDENCE_ROOT = WORKSPACE_ROOT / "Docs" / "system-analysis" / "evidence" / "react-runtime-comparison"
GLOBAL_TABLE_EXCLUSIONS = {"knowledge_documents", "knowledge_sections"}
PRESERVE_SESSION_TABLES = {"sessions", "opportunities"}


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_database_url(raw: str) -> str:
    if raw.startswith("postgresql+psycopg://"):
        return "postgresql://" + raw.removeprefix("postgresql+psycopg://")
    if raw.startswith("postgresql+asyncpg://"):
        return "postgresql://" + raw.removeprefix("postgresql+asyncpg://")
    return raw


def _assert_local_database() -> dict[str, Any]:
    settings = get_settings()
    parsed = urlparse(_normalize_database_url(settings.database_url))
    host = (parsed.hostname or "").strip().lower()
    is_local = host in {"localhost", "127.0.0.1", "::1"}
    if not is_local:
        raise RuntimeError(
            "Refusing to run comparison purge against a non-local database. "
            f"Resolved host: {host or 'unknown'}"
        )
    return {"database_host": host, "database_name": (parsed.path or "").lstrip("/") or "unknown"}


def _table_pk_id(table) -> Any | None:
    if "id" not in table.c:
        return None
    pk_columns = list(table.primary_key.columns)
    if len(pk_columns) == 1 and pk_columns[0].name == "id":
        return table.c.id
    return None


def _topological_order(metadata: MetaData, table_names: set[str]) -> list[str]:
    dependencies: dict[str, set[str]] = {name: set() for name in table_names}
    dependents: dict[str, set[str]] = {name: set() for name in table_names}
    for table_name in table_names:
        table = metadata.tables[table_name]
        for fk in table.foreign_keys:
            parent_table = fk.column.table.name
            if parent_table == table_name or parent_table not in table_names:
                continue
            dependencies[table_name].add(parent_table)
            dependents[parent_table].add(table_name)
    ready = deque(sorted(name for name, deps in dependencies.items() if not deps))
    ordered: list[str] = []
    while ready:
        name = ready.popleft()
        ordered.append(name)
        for child in sorted(dependents[name]):
            dependencies[child].discard(name)
            if not dependencies[child]:
                ready.append(child)
    if len(ordered) != len(table_names):
        ordered.extend(sorted(table_names.difference(ordered)))
    return ordered


def _collect_deletable_ids(connection: Any, metadata: MetaData, session_ids: list[UUID]) -> dict[str, set[Any]]:
    selected_ids: dict[str, set[Any]] = defaultdict(set)
    selected_ids["sessions"].update(session_ids)
    for table_name, table in metadata.tables.items():
        if table_name in PRESERVE_SESSION_TABLES or table_name in GLOBAL_TABLE_EXCLUSIONS:
            continue
        pk = _table_pk_id(table)
        if pk is None or "session_id" not in table.c:
            continue
        rows = connection.execute(sa_select(pk).where(table.c.session_id.in_(session_ids))).all()
        selected_ids[table_name].update(row[0] for row in rows if row[0] is not None)

    changed = True
    while changed:
        changed = False
        for table_name, table in metadata.tables.items():
            if table_name in PRESERVE_SESSION_TABLES or table_name in GLOBAL_TABLE_EXCLUSIONS:
                continue
            pk = _table_pk_id(table)
            if pk is None:
                continue
            if selected_ids.get(table_name):
                continue
            matched: set[Any] = set()
            for fk in table.foreign_keys:
                parent_ids = selected_ids.get(fk.column.table.name)
                if not parent_ids:
                    continue
                rows = connection.execute(sa_select(pk).where(fk.parent.in_(list(parent_ids)))).all()
                matched.update(row[0] for row in rows if row[0] is not None)
            if matched:
                selected_ids[table_name].update(matched)
                changed = True
    return selected_ids


def _purge_generated_state(session_ids: list[UUID]) -> dict[str, int]:
    metadata = MetaData()
    metadata.reflect(bind=engine)
    deleted: dict[str, int] = {}
    with engine.begin() as connection:
        selected_ids = _collect_deletable_ids(connection, metadata, session_ids)
        ordered = _topological_order(metadata, set(metadata.tables))
        for table_name in reversed(ordered):
            if table_name in PRESERVE_SESSION_TABLES or table_name in GLOBAL_TABLE_EXCLUSIONS:
                continue
            table = metadata.tables[table_name]
            ids = selected_ids.get(table_name)
            if ids and "id" in table.c:
                result = connection.execute(delete(table).where(table.c.id.in_(list(ids))))
                if result.rowcount:
                    deleted[table_name] = int(result.rowcount)
                continue
            if "session_id" in table.c:
                result = connection.execute(delete(table).where(table.c.session_id.in_(session_ids)))
                if result.rowcount:
                    deleted[table_name] = int(result.rowcount)
        sessions = metadata.tables["sessions"]
        connection.execute(
            sessions.update()
            .where(sessions.c.id.in_(session_ids))
            .values(current_stage="normalize_discovery", status="needs_review", updated_at=utc_now())
        )
    return dict(sorted(deleted.items()))


def _ensure_user(db: Session, *, email: str, password: str) -> UserRecord:
    user = db.exec(select(UserRecord).where(UserRecord.email == email)).first()
    if user is None:
        user = UserRecord(
            email=email,
            full_name="LAB QA Runtime Comparison",
            password_hash=hash_password(password),
            preferred_language="es",
            is_active=True,
        )
    else:
        user.password_hash = hash_password(password)
        user.preferred_language = "es"
        user.is_active = True
        user.updated_at = utc_now()
    db.add(user)
    db.flush()
    return user


def _ensure_workspace(db: Session, *, workspace_id: UUID, name: str, slug: str) -> WorkspaceRecord:
    workspace = db.get(WorkspaceRecord, workspace_id)
    if workspace is None:
        workspace = WorkspaceRecord(id=workspace_id, name=name, slug=slug, is_active=True)
    else:
        workspace.name = name
        workspace.slug = slug
        workspace.is_active = True
        workspace.updated_at = utc_now()
    db.add(workspace)
    db.flush()
    apply_workspace_bootstrap(db, workspace_id)
    return workspace


def _ensure_membership(db: Session, *, workspace_id: UUID, user_id: UUID) -> None:
    membership = db.exec(
        select(WorkspaceMembershipRecord).where(
            WorkspaceMembershipRecord.workspace_id == workspace_id,
            WorkspaceMembershipRecord.user_id == user_id,
        )
    ).first()
    if membership is None:
        membership = WorkspaceMembershipRecord(
            workspace_id=workspace_id,
            user_id=user_id,
            role=WorkspaceRole.owner,
            is_active=True,
        )
    else:
        membership.role = WorkspaceRole.owner
        membership.is_active = True
        membership.updated_at = utc_now()
    db.add(membership)
    db.flush()


def _copy_workspace_runtime(db: Session, *, source_workspace_id: UUID, target_workspace_id: UUID) -> dict[str, Any]:
    source_runtime = db.exec(
        select(WorkspaceRuntimeSettingsRecord)
        .where(
            WorkspaceRuntimeSettingsRecord.workspace_id == source_workspace_id,
            WorkspaceRuntimeSettingsRecord.is_active == True,  # noqa: E712
        )
        .order_by(WorkspaceRuntimeSettingsRecord.version.desc())
    ).first()
    target_runtime = db.exec(
        select(WorkspaceRuntimeSettingsRecord)
        .where(
            WorkspaceRuntimeSettingsRecord.workspace_id == target_workspace_id,
            WorkspaceRuntimeSettingsRecord.is_active == True,  # noqa: E712
        )
        .order_by(WorkspaceRuntimeSettingsRecord.version.desc())
    ).first()
    if source_runtime is not None:
        if target_runtime is None:
            max_version = db.exec(
                select(func.max(WorkspaceRuntimeSettingsRecord.version)).where(
                    WorkspaceRuntimeSettingsRecord.workspace_id == target_workspace_id
                )
            ).one()
            target_runtime = WorkspaceRuntimeSettingsRecord(
                workspace_id=target_workspace_id,
                version=int(max_version or 0) + 1,
            )
        target_runtime.active_provider = source_runtime.active_provider
        target_runtime.agent_execution_backend = source_runtime.agent_execution_backend
        target_runtime.knowledge_access_backend = source_runtime.knowledge_access_backend
        target_runtime.provider_overrides = dict(source_runtime.provider_overrides or {})
        target_runtime.uses_platform_credentials = source_runtime.uses_platform_credentials
        target_runtime.is_active = True
        target_runtime.updated_by_user_id = source_runtime.updated_by_user_id
        target_runtime.updated_at = utc_now()
        db.add(target_runtime)
    secret_count = 0
    for source_secret in db.exec(
        select(WorkspaceProviderSecretRecord).where(WorkspaceProviderSecretRecord.workspace_id == source_workspace_id)
    ).all():
        target_secret = db.exec(
            select(WorkspaceProviderSecretRecord).where(
                WorkspaceProviderSecretRecord.workspace_id == target_workspace_id,
                WorkspaceProviderSecretRecord.provider_key == source_secret.provider_key,
                WorkspaceProviderSecretRecord.secret_kind == source_secret.secret_kind,
            )
        ).first()
        if target_secret is None:
            target_secret = WorkspaceProviderSecretRecord(
                workspace_id=target_workspace_id,
                provider_key=source_secret.provider_key,
                secret_kind=source_secret.secret_kind,
            )
        target_secret.secret_ciphertext = source_secret.secret_ciphertext
        target_secret.secret_ref = source_secret.secret_ref
        target_secret.status = source_secret.status
        target_secret.last_rotated_at = source_secret.last_rotated_at
        target_secret.updated_by_user_id = source_secret.updated_by_user_id
        target_secret.updated_at = utc_now()
        db.add(target_secret)
        secret_count += 1
    db.flush()
    return {
        "provider": source_runtime.active_provider.value if source_runtime is not None else "platform_default",
        "execution_backend": source_runtime.agent_execution_backend.value if source_runtime is not None else "",
        "knowledge_backend": source_runtime.knowledge_access_backend.value if source_runtime is not None else "",
        "workspace_secret_rows_copied": secret_count,
    }


def prepare_comparison_workspaces(
    *,
    source_session_id: UUID,
    react_off_session_id: UUID,
    react_on_session_id: UUID,
    react_off_workspace_id: UUID,
    react_on_workspace_id: UUID,
) -> dict[str, Any]:
    settings = get_settings()
    with Session(engine) as db:
        source = db.get(SessionRecord, source_session_id)
        react_off_session = db.get(SessionRecord, react_off_session_id)
        react_on_session = db.get(SessionRecord, react_on_session_id)
        if source is None:
            raise RuntimeError(f"Source session not found: {source_session_id}")
        if source.workspace_id is None:
            raise RuntimeError("Source session has no workspace_id; cannot copy runtime settings.")
        if react_off_session is None or react_on_session is None:
            raise RuntimeError("One or both comparison clones are missing.")

        actor = _ensure_user(db, email=settings.local_admin_email, password=settings.local_admin_password)
        source_user_ids = {source.user_id, react_off_session.user_id, react_on_session.user_id}
        _ensure_workspace(
            db,
            workspace_id=react_off_workspace_id,
            name="ACP146 QA ReAct OFF",
            slug="acp146-react-off",
        )
        _ensure_workspace(
            db,
            workspace_id=react_on_workspace_id,
            name="ACP146 QA ReAct ON",
            slug="acp146-react-on",
        )
        runtime_off = _copy_workspace_runtime(
            db,
            source_workspace_id=source.workspace_id,
            target_workspace_id=react_off_workspace_id,
        )
        runtime_on = _copy_workspace_runtime(
            db,
            source_workspace_id=source.workspace_id,
            target_workspace_id=react_on_workspace_id,
        )
        for workspace_id in (react_off_workspace_id, react_on_workspace_id):
            _ensure_membership(db, workspace_id=workspace_id, user_id=actor.id)
            for user_id in source_user_ids:
                _ensure_membership(db, workspace_id=workspace_id, user_id=user_id)
            seed_runtime_feature_flags(db, workspace_id=workspace_id)
        update_feature_flag(db, workspace_id=react_off_workspace_id, flag_key="react_runtime_v1", enabled=False)
        update_feature_flag(db, workspace_id=react_on_workspace_id, flag_key="react_runtime_v1", enabled=True)
        react_off_session.workspace_id = react_off_workspace_id
        react_on_session.workspace_id = react_on_workspace_id
        react_off_session.updated_at = utc_now()
        react_on_session.updated_at = utc_now()
        db.add(react_off_session)
        db.add(react_on_session)
        db.commit()
        return {
            "source_session_id": str(source_session_id),
            "source_workspace_id": str(source.workspace_id),
            "actor_email": settings.local_admin_email,
            "react_off": {
                "session_id": str(react_off_session_id),
                "workspace_id": str(react_off_workspace_id),
                "react_runtime_v1": False,
                "runtime": runtime_off,
            },
            "react_on": {
                "session_id": str(react_on_session_id),
                "workspace_id": str(react_on_workspace_id),
                "react_runtime_v1": True,
                "runtime": runtime_on,
            },
        }


def _load_discovery_input(session_id: UUID) -> dict[str, Any]:
    with Session(engine) as db:
        opportunity = db.exec(select(OpportunityRecord).where(OpportunityRecord.session_id == session_id)).first()
        if opportunity is None:
            raise RuntimeError(f"Session {session_id} has no opportunity input.")
        return _opportunity_to_discovery_input(opportunity)


def _login_headers(client: TestClient) -> tuple[dict[str, str], str]:
    settings = get_settings()
    login_email = settings.local_admin_email
    login_password = settings.local_admin_password
    login = _request_json(
        client,
        "POST",
        "/api/v1/auth/login",
        json_body={"email": login_email, "password": login_password},
    )
    return {"Authorization": f"Bearer {login['access_token']}"}, login_email


def _append_step(steps: list[dict[str, Any]], *, name: str, response: Any, started: float) -> None:
    item: dict[str, Any] = {
        "step": name,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }
    if isinstance(response, dict):
        for key in ("id", "status", "stage", "state", "confidence", "missing_information", "warnings"):
            if key not in response:
                continue
            value = response[key]
            if key in {"missing_information", "warnings"} and isinstance(value, list):
                item[f"{key}_count"] = len(value)
            else:
                item[key] = value
        if response.get("proposal_payload"):
            quality_gate = response["proposal_payload"].get("quality_gate")
            if isinstance(quality_gate, dict):
                item["quality_gate"] = {
                    "quality_confidence": quality_gate.get("quality_confidence"),
                    "repair_policy": quality_gate.get("repair_policy"),
                    "should_repair": quality_gate.get("should_repair"),
                    "minimum_repair_cycles": quality_gate.get("minimum_repair_cycles"),
                    "quality_repair_cycles": quality_gate.get("quality_repair_cycles"),
                    "blocking": quality_gate.get("blocking"),
                }
    steps.append(item)


def _timed_request(
    client: TestClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_body: Any = None,
    step_name: str,
    steps: list[dict[str, Any]],
) -> Any:
    started = time.perf_counter()
    response = _request_json(client, method, url, headers=headers, json_body=json_body)
    _append_step(steps, name=step_name, response=response, started=started)
    return response


@contextmanager
def _comparison_client():
    with _suppress_blueprint_basic_postprocessing(), TestClient(app) as client:
        yield client


def run_clone_flow(session_id: UUID, *, variant: str, approve_optional_tools: str) -> dict[str, Any]:
    discovery_input = _load_discovery_input(session_id)
    steps: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc)
    run_status = "completed"
    failure: dict[str, Any] | None = None
    with _comparison_client() as client:
        headers, actor_email = _login_headers(client)
        try:
            _timed_request(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/normalize-discovery",
                headers=headers,
                json_body=discovery_input,
                step_name="normalize_discovery",
                steps=steps,
            )
            discover_artifact = _timed_request(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/analyze-discovery",
                headers=headers,
                json_body=discovery_input,
                step_name="analyze_discovery",
                steps=steps,
            )
            started = time.perf_counter()
            _approve_artifact(
                client,
                headers=headers,
                session_id=str(session_id),
                stage="discover",
                artifact=discover_artifact,
                note=f"Discover aprobado para comparacion {variant}.",
                decision_payload={"approval_reason": "Input fuente replicado para comparacion A/B."},
            )
            steps.append({"step": "approve_discover", "elapsed_ms": round((time.perf_counter() - started) * 1000)})
            _timed_request(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/build-canvas",
                headers=headers,
                step_name="build_canvas",
                steps=steps,
            )
            define_artifact = _timed_request(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/define-requirements",
                headers=headers,
                step_name="define_requirements",
                steps=steps,
            )
            started = time.perf_counter()
            _approve_artifact(
                client,
                headers=headers,
                session_id=str(session_id),
                stage="define",
                artifact=define_artifact,
                note=f"Define aprobado para comparacion {variant}.",
                decision_payload={"approval_reason": "Definition lista para evaluar continuidad hacia Design."},
            )
            steps.append({"step": "approve_define", "elapsed_ms": round((time.perf_counter() - started) * 1000)})
            _timed_request(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/build-blueprint",
                headers=headers,
                step_name="build_blueprint",
                steps=steps,
            )
            design_artifact = _timed_request(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/propose-design",
                headers=headers,
                step_name="propose_design",
                steps=steps,
            )
            selected_key = (
                design_artifact.get("proposal_payload", {}).get("selected_design", {}).get("alternative_key")
                or design_artifact.get("proposal_payload", {}).get("recommended_alternative_key")
                or ""
            )
            started = time.perf_counter()
            _approve_artifact(
                client,
                headers=headers,
                session_id=str(session_id),
                stage="design",
                artifact=design_artifact,
                note=f"Design aprobado para comparacion {variant}.",
                decision_payload={"selected_alternative_key": selected_key},
            )
            steps.append(
                {
                    "step": "approve_design",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "selected_alternative_key": selected_key,
                }
            )
            tools = _timed_request(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/recommend-tools",
                headers=headers,
                step_name="recommend_tools",
                steps=steps,
            )
            optional_keys = _approve_optional_keys(tools, approve_optional_tools)
            started = time.perf_counter()
            _request_json(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/approve-tools-selection",
                headers=headers,
                json_body={"include_optional_tool_keys": optional_keys},
            )
            steps.append(
                {
                    "step": "approve_tools_selection",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "selected_optional_count": len(optional_keys),
                }
            )
            memory = _timed_request(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/recommend-memory",
                headers=headers,
                step_name="recommend_memory",
                steps=steps,
            )
            started = time.perf_counter()
            _request_json(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/approve-memory-profile",
                headers=headers,
                json_body={
                    "note": f"Memory aprobado para comparacion {variant}.",
                    "decision_payload": {"approval_reason": "Memoria suficiente para estimacion comparativa."},
                },
            )
            steps.append({"step": "approve_memory_profile", "elapsed_ms": round((time.perf_counter() - started) * 1000)})
            _timed_request(
                client,
                "POST",
                f"/api/v1/sessions/{session_id}/estimate",
                headers=headers,
                step_name="estimate",
                steps=steps,
            )
        except Exception as exc:  # noqa: BLE001
            run_status = "failed"
            failure = {"type": type(exc).__name__, "message": str(exc)}
    finished_at = datetime.now(timezone.utc)
    return {
        "variant": variant,
        "session_id": str(session_id),
        "actor_email": actor_email,
        "run_status": run_status,
        "failure": failure,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": round((finished_at - started_at).total_seconds() * 1000),
        "steps": steps,
        "quality": _quality_report(session_id, discovery_input),
        "observability": collect_observability(session_id),
    }


def _safe_enum_value(value: Any) -> str:
    return getattr(value, "value", str(value))


def collect_observability(session_id: UUID) -> dict[str, Any]:
    with Session(engine) as db:
        record = db.get(SessionRecord, session_id)
        artifacts = _artifact_payloads(session_id)
        usage_rows = db.exec(select(LLMUsageLedgerRecord).where(LLMUsageLedgerRecord.session_id == session_id)).all()
        skill_rows = db.exec(
            select(SkillRunRecord).where(SkillRunRecord.session_id == session_id).order_by(SkillRunRecord.created_at.asc())
        ).all()
        log_rows = db.exec(
            select(ExecutionLogRecord)
            .where(ExecutionLogRecord.session_id == session_id)
            .order_by(ExecutionLogRecord.created_at.asc())
        ).all()
        operation_rows = db.exec(
            select(StageOperationRecord)
            .where(StageOperationRecord.session_id == session_id)
            .order_by(StageOperationRecord.created_at.asc())
        ).all()
        checkpoint_rows = db.exec(
            select(ShortTermCheckpointRecord)
            .where(ShortTermCheckpointRecord.session_id == session_id)
            .order_by(ShortTermCheckpointRecord.created_at.asc())
        ).all()
    provider_counts = Counter(row.provider_key for row in usage_rows)
    stage_usage: dict[str, dict[str, Any]] = {}
    for row in usage_rows:
        stage = row.stage or "unknown"
        bucket = stage_usage.setdefault(stage, {"calls": 0, "total_tokens": 0, "cost_total": 0.0, "fallbacks": 0})
        bucket["calls"] += 1
        bucket["total_tokens"] += int(row.total_tokens or 0)
        bucket["cost_total"] += float(row.cost_total or 0)
        bucket["fallbacks"] += 1 if row.fallback_used else 0
    for bucket in stage_usage.values():
        bucket["cost_total"] = round(bucket["cost_total"], 6)
    react_runs = [row for row in skill_rows if row.skill_key.startswith("react:")]
    return {
        "session": {
            "current_stage": _safe_enum_value(record.current_stage) if record else "",
            "status": _safe_enum_value(record.status) if record else "",
            "workspace_id": str(record.workspace_id) if record else "",
        },
        "artifacts": {
            stage: {
                "artifact_kind": payload.get("artifact_kind"),
                "state": payload.get("state"),
                "confidence": payload.get("confidence"),
                "provider_key": payload.get("provider_key"),
                "model": payload.get("model"),
                "missing_information_count": payload.get("missing_information_count"),
                "warnings_count": payload.get("warnings_count"),
                "payload_chars": len(json.dumps(payload.get("proposal_payload") or {}, ensure_ascii=False, default=_json_default)),
            }
            for stage, payload in artifacts.items()
        },
        "llm_usage": {
            "calls": len(usage_rows),
            "providers": dict(provider_counts),
            "total_tokens": sum(int(row.total_tokens or 0) for row in usage_rows),
            "input_tokens": sum(int(row.input_tokens or 0) for row in usage_rows),
            "output_tokens": sum(int(row.output_tokens or 0) for row in usage_rows),
            "cost_total": round(sum(float(row.cost_total or 0) for row in usage_rows), 6),
            "stage_usage": stage_usage,
        },
        "skill_runs": {
            "count": len(skill_rows),
            "react_count": len(react_runs),
            "react_runs": [
                {
                    "skill_key": row.skill_key,
                    "status": _safe_enum_value(row.status),
                    "duration_ms": row.duration_ms,
                    "summary": row.result_summary,
                    "evidence": row.evidence,
                }
                for row in react_runs
            ],
            "fallback_warning_count": sum(
                1
                for row in skill_rows
                for warning in row.warnings
                if "fallback" in str(warning).lower() or "determin" in str(warning).lower()
            ),
        },
        "execution_logs": {
            "count": len(log_rows),
            "failed": [
                {
                    "stage": _safe_enum_value(row.stage),
                    "status": _safe_enum_value(row.status),
                    "message": row.message,
                    "created_at": row.created_at,
                }
                for row in log_rows
                if row.status == ArtifactStatus.failed
            ],
            "tail": [
                {
                    "stage": _safe_enum_value(row.stage),
                    "status": _safe_enum_value(row.status),
                    "message": row.message,
                    "created_at": row.created_at,
                }
                for row in log_rows[-8:]
            ],
        },
        "stage_operations": {
            "count": len(operation_rows),
            "non_terminal": [
                {
                    "action": row.action,
                    "stage_key": row.stage_key,
                    "status": _safe_enum_value(row.status),
                    "detail": row.detail,
                    "updated_at": row.updated_at,
                }
                for row in operation_rows
                if _safe_enum_value(row.status) not in {"completed", "failed", "cancelled"}
            ],
        },
        "checkpoints": {
            "count": len(checkpoint_rows),
            "react_checkpoints": [
                {
                    "checkpoint_key": row.checkpoint_key,
                    "stage": row.stage,
                    "source_action": row.source_action,
                    "status": row.status,
                    "summary": row.summary,
                    "is_active": row.is_active,
                    "is_consistent": row.is_consistent,
                }
                for row in checkpoint_rows
                if row.source_action.startswith("react_") or "react" in row.checkpoint_key
            ],
        },
    }


def compare_results(react_off: dict[str, Any], react_on: dict[str, Any]) -> dict[str, Any]:
    by_stage_off = {item["stage"]: item for item in react_off["quality"]["stages"]}
    by_stage_on = {item["stage"]: item for item in react_on["quality"]["stages"]}
    stage_delta = []
    for stage in STAGE_ORDER:
        off = by_stage_off.get(stage, {})
        on = by_stage_on.get(stage, {})
        stage_delta.append(
            {
                "stage": stage,
                "off_available": off.get("available"),
                "on_available": on.get("available"),
                "quality_score_off": off.get("quality_score"),
                "quality_score_on": on.get("quality_score"),
                "quality_delta": round(float(on.get("quality_score") or 0) - float(off.get("quality_score") or 0), 3),
                "confidence_off": off.get("confidence"),
                "confidence_on": on.get("confidence"),
                "confidence_delta": round(float(on.get("confidence") or 0) - float(off.get("confidence") or 0), 3),
                "missing_off": off.get("missing_information_count"),
                "missing_on": on.get("missing_information_count"),
                "warnings_off": off.get("warnings_count"),
                "warnings_on": on.get("warnings_count"),
                "continuity_off": off.get("continuity_score"),
                "continuity_on": on.get("continuity_score"),
            }
        )
    usage_off = react_off["observability"]["llm_usage"]
    usage_on = react_on["observability"]["llm_usage"]
    return {
        "aggregate_quality_score_off": react_off["quality"]["aggregate_quality_score"],
        "aggregate_quality_score_on": react_on["quality"]["aggregate_quality_score"],
        "aggregate_quality_delta": round(
            react_on["quality"]["aggregate_quality_score"] - react_off["quality"]["aggregate_quality_score"],
            3,
        ),
        "llm_calls_off": usage_off["calls"],
        "llm_calls_on": usage_on["calls"],
        "llm_calls_delta": usage_on["calls"] - usage_off["calls"],
        "total_tokens_off": usage_off["total_tokens"],
        "total_tokens_on": usage_on["total_tokens"],
        "total_tokens_delta": usage_on["total_tokens"] - usage_off["total_tokens"],
        "cost_total_off": usage_off["cost_total"],
        "cost_total_on": usage_on["cost_total"],
        "cost_total_delta": round(usage_on["cost_total"] - usage_off["cost_total"], 6),
        "react_runs_off": react_off["observability"]["skill_runs"]["react_count"],
        "react_runs_on": react_on["observability"]["skill_runs"]["react_count"],
        "stage_delta": stage_delta,
    }


def _markdown_report(result: dict[str, Any]) -> str:
    comparison = result["comparison"]
    lines = [
        "# Comparacion ACP146 react_runtime_v1",
        "",
        f"- Generado: {result['generated_at']}",
        f"- DB local: {result['database']}",
        f"- Source: `{result['source_session_id']}`",
        f"- OFF clone: `{result['react_off']['session_id']}` workspace `{result['react_off']['observability']['session']['workspace_id']}`",
        f"- ON clone: `{result['react_on']['session_id']}` workspace `{result['react_on']['observability']['session']['workspace_id']}`",
        "",
        "## Resultado Ejecutivo",
        "",
        f"- Calidad agregada OFF: `{comparison['aggregate_quality_score_off']}`",
        f"- Calidad agregada ON: `{comparison['aggregate_quality_score_on']}`",
        f"- Delta calidad ON-OFF: `{comparison['aggregate_quality_delta']}`",
        f"- Llamadas LLM OFF/ON: `{comparison['llm_calls_off']}` / `{comparison['llm_calls_on']}`",
        f"- Tokens OFF/ON: `{comparison['total_tokens_off']}` / `{comparison['total_tokens_on']}`",
        f"- Costo OFF/ON: `{comparison['cost_total_off']}` / `{comparison['cost_total_on']}`",
        f"- ReAct runs OFF/ON: `{comparison['react_runs_off']}` / `{comparison['react_runs_on']}`",
        "",
        "## Delta Por Etapa",
        "",
        "| Etapa | Calidad OFF | Calidad ON | Delta | Conf OFF | Conf ON | Faltantes OFF/ON | Warnings OFF/ON | Continuidad OFF/ON |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparison["stage_delta"]:
        lines.append(
            "| {stage} | {quality_score_off} | {quality_score_on} | {quality_delta} | "
            "{confidence_off} | {confidence_on} | {missing_off}/{missing_on} | "
            "{warnings_off}/{warnings_on} | {continuity_off}/{continuity_on} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Estados",
            "",
            f"- OFF status: `{result['react_off']['run_status']}` failure: `{result['react_off']['failure']}`",
            f"- ON status: `{result['react_on']['run_status']}` failure: `{result['react_on']['failure']}`",
            "",
            "## Observabilidad",
            "",
            f"- OFF proveedores: `{result['react_off']['observability']['llm_usage']['providers']}`",
            f"- ON proveedores: `{result['react_on']['observability']['llm_usage']['providers']}`",
            f"- OFF operaciones no terminales: `{result['react_off']['observability']['stage_operations']['non_terminal']}`",
            f"- ON operaciones no terminales: `{result['react_on']['observability']['stage_operations']['non_terminal']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    database = _assert_local_database()
    session_ids = [args.react_off_session_id, args.react_on_session_id]
    purge_report = _purge_generated_state(session_ids) if args.purge_before_run else {}
    workspace_report = prepare_comparison_workspaces(
        source_session_id=args.source_session_id,
        react_off_session_id=args.react_off_session_id,
        react_on_session_id=args.react_on_session_id,
        react_off_workspace_id=args.react_off_workspace_id,
        react_on_workspace_id=args.react_on_workspace_id,
    )
    with Session(engine) as db:
        source = db.get(SessionRecord, args.source_session_id)
        source_opportunity = db.exec(
            select(OpportunityRecord).where(OpportunityRecord.session_id == args.source_session_id)
        ).first()
        source_discovery_input = _opportunity_to_discovery_input(source_opportunity) if source_opportunity else {}
    result_off = run_clone_flow(args.react_off_session_id, variant="react_runtime_v1=false", approve_optional_tools=args.approve_optional_tools)
    result_on = run_clone_flow(args.react_on_session_id, variant="react_runtime_v1=true", approve_optional_tools=args.approve_optional_tools)
    result = {
        "contract_version": "react-runtime-clone-comparison.v1",
        "generated_at": datetime.now(timezone.utc),
        "database": database,
        "source_session_id": str(args.source_session_id),
        "source_title": source.title if source is not None else "",
        "source_terms": _source_terms(source_discovery_input),
        "purge_before_run": args.purge_before_run,
        "purge_report": purge_report,
        "workspace_report": workspace_report,
        "react_off": result_off,
        "react_on": result_on,
        "comparison": compare_results(result_off, result_on),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara dos clones locales Discovery -> Estimate con react_runtime_v1 OFF vs ON."
    )
    parser.add_argument("--source-session-id", type=UUID, default=DEFAULT_SOURCE_SESSION_ID)
    parser.add_argument("--react-off-session-id", type=UUID, default=DEFAULT_REACT_OFF_SESSION_ID)
    parser.add_argument("--react-on-session-id", type=UUID, default=DEFAULT_REACT_ON_SESSION_ID)
    parser.add_argument("--react-off-workspace-id", type=UUID, default=DEFAULT_REACT_OFF_WORKSPACE_ID)
    parser.add_argument("--react-on-workspace-id", type=UUID, default=DEFAULT_REACT_ON_WORKSPACE_ID)
    parser.add_argument("--approve-optional-tools", choices=("first", "all", "none"), default="first")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--purge-before-run", action="store_true")
    args = parser.parse_args()

    result = run_comparison(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()
    json_path = args.output_dir / f"run-{stamp}.json"
    md_path = args.output_dir / f"run-{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    md_path.write_text(_markdown_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "comparison": result["comparison"],
                "react_off_status": result["react_off"]["run_status"],
                "react_on_status": result["react_on"]["run_status"],
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0 if result["react_off"]["run_status"] == result["react_on"]["run_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
