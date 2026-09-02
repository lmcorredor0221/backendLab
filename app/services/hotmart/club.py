from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import func
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    CommercialEntitlementRecord,
    CommercialEntitlementStatus,
    CommercialOrderRecord,
    HotmartClubModuleResponse,
    HotmartClubOverviewResponse,
    HotmartClubPageResponse,
    HotmartClubProgressResponse,
    HotmartClubStudentResponse,
    HotmartClubSyncRequest,
    HotmartIntegrationConfigRecord,
    HotmartReconciliationIssueRecord,
    HotmartSyncCursorRecord,
    HotmartSyncRunRecord,
    HotmartSyncRunResponse,
    UserRecord,
    utc_now,
)
from app.services.commerce_service import record_commercial_event
from app.services.hotmart.auth import (
    HotmartAuthClient,
    HotmartAuthError,
    default_hotmart_api_base_url,
    normalize_hotmart_environment,
)
from app.services.hotmart.redaction import redact_payload
from app.services.hotmart.secrets import build_hotmart_status, load_hotmart_credentials
from app.services.hotmart.sync import SyncIssueStats, _count_issue_result, _open_or_update_issue


CLUB_RESOURCE = "club"
CLUB_ISSUE_TYPES = {"club_student_without_internal_access", "internal_access_without_club_student"}


class HotmartClubError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.payload = redact_payload(payload or {})


@dataclass(frozen=True)
class HotmartClubPayloads:
    modules: list[dict[str, Any]]
    pages: list[dict[str, Any]]
    students: list[dict[str, Any]]
    progress: list[dict[str, Any]]

    @property
    def total_records(self) -> int:
        return len(self.modules) + len(self.pages) + len(self.students) + len(self.progress)


class HotmartClubApiClient:
    def __init__(
        self,
        *,
        api_base_url: str,
        timeout_seconds: int = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/") or default_hotmart_api_base_url("sandbox")
        self.timeout_seconds = max(1, timeout_seconds)
        self.transport = transport

    def _url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self.api_base_url}{normalized_path}"

    def fetch_path(
        self,
        *,
        access_token: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = client.get(self._url(path), headers=headers, params=_flatten_params(params or {}))
        payload = _read_response_payload(response)
        redacted = redact_payload(payload)
        if response.status_code == 429:
            raise HotmartClubError(
                "rate_limited",
                "Hotmart rate limited the Club request.",
                http_status=response.status_code,
                payload=redacted,
            )
        if response.status_code >= 400:
            raise HotmartClubError(
                "club_request_rejected",
                "Hotmart rejected the Club request.",
                http_status=response.status_code,
                payload=redacted,
            )
        return _extract_items(payload)


def _read_response_payload(response: httpx.Response) -> dict[str, Any]:
    if not response.text:
        return {}
    try:
        payload = response.json()
    except ValueError:
        return {"raw": response.text[:500]}
    return payload if isinstance(payload, dict) else {"items": payload}


def _flatten_params(values: dict[str, Any]) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []
    for key, value in values.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            params.append((key, str(value).lower()))
            continue
        if isinstance(value, list):
            params.extend((key, str(item)) for item in value if str(item).strip())
            continue
        params.append((key, str(value)))
    return params


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("items", "results", "data", "content", "modules", "pages", "users", "lessons", "students"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload] if payload else []


def _get_path(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_text(payload: dict[str, Any], *paths: tuple[str, ...] | str) -> str:
    for path in paths:
        if isinstance(path, str):
            value = payload.get(path)
        else:
            value = _get_path(payload, *path)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "sim", "si", "completed", "complete", "done"}


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _extract_module_id(item: dict[str, Any]) -> str:
    return _first_text(item, "module_id", "id", "ucode", "moduleId")


def _extract_page_id(item: dict[str, Any]) -> str:
    return _first_text(item, "page_id", "lesson_id", "id", "ucode", "pageId", "lessonId")


def _extract_student_id(item: dict[str, Any]) -> str:
    return _first_text(item, "user_id", "student_id", "id", "ucode", ("user", "id"), ("student", "id"))


def _extract_student_email(item: dict[str, Any]) -> str:
    return _first_text(item, "email", ("user", "email"), ("student", "email")).lower()


def _extract_student_name(item: dict[str, Any]) -> str:
    return _first_text(item, "name", "full_name", ("user", "name"), ("student", "name"), ("profile", "name"))


def _serialize_sync_run(record: HotmartSyncRunRecord) -> HotmartSyncRunResponse:
    return HotmartSyncRunResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        environment=record.environment,  # type: ignore[arg-type]
        resource=record.resource,
        status=record.status,
        started_by_user_id=record.started_by_user_id,
        started_at=record.started_at,
        finished_at=record.finished_at,
        cursor_before=record.cursor_before,
        cursor_after=record.cursor_after,
        records_read=record.records_read,
        records_created=record.records_created,
        records_updated=record.records_updated,
        records_skipped=record.records_skipped,
        error_summary=record.error_summary,
        issue_count=int(record.metadata_payload.get("issue_count") or 0),
    )


def _latest_club_run(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
) -> HotmartSyncRunRecord | None:
    return session.exec(
        select(HotmartSyncRunRecord)
        .where(
            HotmartSyncRunRecord.workspace_id == workspace_id,
            HotmartSyncRunRecord.environment == environment,
            HotmartSyncRunRecord.resource == CLUB_RESOURCE,
        )
        .order_by(HotmartSyncRunRecord.started_at.desc())
    ).first()


def _club_snapshot_list(record: HotmartSyncRunRecord | None, key: str) -> list[dict[str, Any]]:
    if record is None:
        return []
    value = record.metadata_payload.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _club_issue_query(
    *,
    workspace_id: UUID,
    environment: str,
    open_only: bool = True,
):
    query = select(HotmartReconciliationIssueRecord).where(
        HotmartReconciliationIssueRecord.workspace_id == workspace_id,
        HotmartReconciliationIssueRecord.environment == environment,
    )
    if open_only:
        query = query.where(HotmartReconciliationIssueRecord.status == "open")
    return query


def _update_cursor(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
) -> None:
    cursor = session.exec(
        select(HotmartSyncCursorRecord).where(
            HotmartSyncCursorRecord.workspace_id == workspace_id,
            HotmartSyncCursorRecord.environment == environment,
            HotmartSyncCursorRecord.resource == CLUB_RESOURCE,
        )
    ).first()
    if cursor is None:
        cursor = HotmartSyncCursorRecord(
            workspace_id=workspace_id,
            environment=environment,
            resource=CLUB_RESOURCE,
        )
    cursor.last_success_at = utc_now()
    cursor.updated_at = utc_now()
    session.add(cursor)


def _update_config_last_sync(session: Session, *, workspace_id: UUID, environment: str) -> None:
    config = session.exec(
        select(HotmartIntegrationConfigRecord).where(
            HotmartIntegrationConfigRecord.workspace_id == workspace_id,
            HotmartIntegrationConfigRecord.environment == environment,
        )
    ).first()
    if config is not None:
        config.last_sync_at = utc_now()
        config.updated_at = utc_now()
        session.add(config)


def _has_active_entitlement_for_user(session: Session, *, workspace_id: UUID, user_id: UUID) -> bool:
    orders = session.exec(
        select(CommercialOrderRecord).where(
            CommercialOrderRecord.workspace_id == workspace_id,
            CommercialOrderRecord.buyer_user_id == user_id,
        )
    ).all()
    order_ids = [order.id for order in orders]
    if not order_ids:
        return False
    now = utc_now()
    entitlement = session.exec(
        select(CommercialEntitlementRecord).where(
            CommercialEntitlementRecord.workspace_id == workspace_id,
            CommercialEntitlementRecord.status == CommercialEntitlementStatus.active,
            CommercialEntitlementRecord.order_id.in_(order_ids),
        )
    ).first()
    return bool(entitlement and entitlement.starts_at <= now and (entitlement.ends_at is None or entitlement.ends_at > now))


def _active_entitlements_with_user_email(
    session: Session,
    *,
    workspace_id: UUID,
) -> list[tuple[CommercialEntitlementRecord, CommercialOrderRecord, UserRecord]]:
    rows: list[tuple[CommercialEntitlementRecord, CommercialOrderRecord, UserRecord]] = []
    now = utc_now()
    entitlements = session.exec(
        select(CommercialEntitlementRecord).where(
            CommercialEntitlementRecord.workspace_id == workspace_id,
            CommercialEntitlementRecord.status == CommercialEntitlementStatus.active,
        )
    ).all()
    for entitlement in entitlements:
        if entitlement.starts_at > now or (entitlement.ends_at is not None and entitlement.ends_at <= now):
            continue
        if entitlement.order_id is None:
            continue
        order = session.get(CommercialOrderRecord, entitlement.order_id)
        if order is None or order.workspace_id != workspace_id:
            continue
        user = session.get(UserRecord, order.buyer_user_id)
        if user is None or not user.email.strip():
            continue
        rows.append((entitlement, order, user))
    return rows


def _fetch_club_payloads(
    *,
    client: HotmartClubApiClient,
    access_token: str,
    payload: HotmartClubSyncRequest,
) -> HotmartClubPayloads:
    settings = get_settings()
    subdomain = payload.subdomain.strip()
    modules: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    students: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []

    if payload.sync_modules:
        module_params: dict[str, Any] = {"subdomain": subdomain}
        if payload.is_extra is not None:
            module_params["is_extra"] = payload.is_extra
        modules = client.fetch_path(
            access_token=access_token,
            path=settings.hotmart_club_modules_path,
            params=module_params,
        )

    if payload.sync_pages:
        module_refs = [payload.module_id.strip()] if payload.module_id.strip() else []
        if not module_refs:
            for module in modules:
                module_id = _extract_module_id(module)
                if module_id and module_id not in module_refs:
                    module_refs.append(module_id)
        for module_id in module_refs:
            module_pages = client.fetch_path(
                access_token=access_token,
                path=settings.hotmart_club_pages_path_template.format(module_id=module_id),
                params={"subdomain": subdomain},
            )
            for page in module_pages:
                page.setdefault("module_id", module_id)
            pages.extend(module_pages)

    if payload.sync_students:
        students = client.fetch_path(
            access_token=access_token,
            path=settings.hotmart_club_students_path,
            params={"subdomain": subdomain},
        )

    if payload.sync_progress:
        user_refs = [payload.user_id.strip()] if payload.user_id.strip() else []
        if not user_refs:
            for student in students:
                student_id = _extract_student_id(student)
                if student_id and student_id not in user_refs:
                    user_refs.append(student_id)
        email_by_user_ref = {_extract_student_id(student): _extract_student_email(student) for student in students}
        for user_ref in user_refs:
            lessons = client.fetch_path(
                access_token=access_token,
                path=settings.hotmart_club_progress_path_template.format(user_id=user_ref),
                params={"subdomain": subdomain},
            )
            for lesson in lessons:
                lesson.setdefault("user_id", user_ref)
                lesson.setdefault("email", email_by_user_ref.get(user_ref, ""))
            progress.extend(lessons)

    return HotmartClubPayloads(modules=modules, pages=pages, students=students, progress=progress)


def _reconcile_club_students(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str,
    students: list[dict[str, Any]],
    actor_user_id: UUID | None,
) -> SyncIssueStats:
    stats = SyncIssueStats()
    remote_student_emails: set[str] = set()
    for student in students:
        email = _extract_student_email(student)
        student_id = _extract_student_id(student)
        provider_ref = student_id or email
        if not email:
            stats.skipped += 1
            continue
        remote_student_emails.add(email)
        user = session.exec(select(UserRecord).where(func.lower(UserRecord.email) == email)).first()
        if user is not None and _has_active_entitlement_for_user(session, workspace_id=workspace_id, user_id=user.id):
            stats.skipped += 1
            continue
        result = _open_or_update_issue(
            session,
            workspace_id=workspace_id,
            environment=environment,
            issue_type="club_student_without_internal_access",
            provider_ref=provider_ref,
            internal_ref=str(user.id) if user is not None else "",
            severity="high",
            summary=f"Hotmart Club student {email} has no active internal entitlement.",
            suggested_action="Validate the Hotmart purchase and grant/link/revoke access from the admin workflow.",
            metadata={"student": redact_payload(student), "email": email},
            actor_user_id=actor_user_id,
        )
        _count_issue_result(stats, result)

    for entitlement, order, user in _active_entitlements_with_user_email(session, workspace_id=workspace_id):
        email = user.email.strip().lower()
        if email in remote_student_emails:
            stats.skipped += 1
            continue
        result = _open_or_update_issue(
            session,
            workspace_id=workspace_id,
            environment=environment,
            issue_type="internal_access_without_club_student",
            provider_ref=email,
            internal_ref=str(entitlement.id),
            severity="medium",
            summary=f"Internal active entitlement for {email} was not found in Hotmart Club students.",
            suggested_action="Invite/sync the student in Hotmart Club or mark the entitlement as intentionally outside Club.",
            metadata={"entitlement_id": str(entitlement.id), "order_id": str(order.id), "buyer_user_id": str(user.id)},
            actor_user_id=actor_user_id,
        )
        _count_issue_result(stats, result)
    return stats


def _serialize_module(item: dict[str, Any]) -> HotmartClubModuleResponse:
    return HotmartClubModuleResponse(
        module_id=_extract_module_id(item),
        name=_first_text(item, "name", "title", "module_name"),
        sequence=_as_int(_first_text(item, "sequence", "order", "position")),
        is_public=_as_bool(item.get("is_public") or item.get("public")),
        is_extra=_as_bool(item.get("is_extra")),
        is_extra_paid=_as_bool(item.get("is_extra_paid") or item.get("extra_paid")),
        total_pages=_as_int(item.get("total_pages") or item.get("pages_count") or item.get("total_lessons")),
    )


def _serialize_page(item: dict[str, Any]) -> HotmartClubPageResponse:
    return HotmartClubPageResponse(
        page_id=_extract_page_id(item),
        module_id=_first_text(item, "module_id", "moduleId", ("module", "id")),
        name=_first_text(item, "name", "title", "page_name", "lesson_name"),
        page_order=_as_int(_first_text(item, "page_order", "order", "position", "sequence")),
        type=_first_text(item, "type", "page_type", "lesson_type"),
    )


def _serialize_student(item: dict[str, Any]) -> HotmartClubStudentResponse:
    progress = item.get("progress")
    return HotmartClubStudentResponse(
        user_id=_extract_student_id(item),
        name=_extract_student_name(item),
        email=_extract_student_email(item),
        status=_first_text(item, "status", "access_status", ("subscription", "status")),
        engagement=_first_text(item, "engagement", "engagement_status", "last_engagement"),
        progress=progress if isinstance(progress, dict) else {},
    )


def _serialize_progress(item: dict[str, Any]) -> HotmartClubProgressResponse:
    return HotmartClubProgressResponse(
        user_id=_first_text(item, "user_id", "student_id", ("user", "id")),
        email=_first_text(item, "email", ("user", "email")).lower(),
        page_id=_extract_page_id(item),
        page_name=_first_text(item, "page_name", "lesson_name", "name", "title"),
        completed=_as_bool(item.get("completed") or item.get("is_completed") or item.get("status")),
        completed_at=_parse_datetime(item.get("completed_at") or item.get("completion_date") or item.get("finished_at")),
        progress_payload=redact_payload(item),
    )


def sync_hotmart_club(
    session: Session,
    *,
    workspace_id: UUID,
    payload: HotmartClubSyncRequest,
    actor_user_id: UUID | None = None,
    transport: httpx.BaseTransport | None = None,
) -> HotmartSyncRunResponse:
    env = normalize_hotmart_environment(payload.environment)
    subdomain = payload.subdomain.strip()
    if not subdomain:
        raise ValueError("Hotmart Club subdomain is required.")

    run = HotmartSyncRunRecord(
        workspace_id=workspace_id,
        environment=env,
        resource=CLUB_RESOURCE,
        status="running",
        started_by_user_id=actor_user_id,
    )
    session.add(run)
    session.flush()
    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="hotmart_sync_started",
        product_key=CLUB_RESOURCE,
        source="hotmart_club",
        metadata={"run_id": str(run.id), "subdomain": subdomain},
        correlation_id=str(run.id),
    )

    status = build_hotmart_status(session, workspace_id=workspace_id, environment=env)
    credentials = load_hotmart_credentials(session, workspace_id=workspace_id, environment=env)
    if credentials is None:
        run.status = "failed"
        run.finished_at = utc_now()
        run.error_summary = "Hotmart OAuth credentials are required before syncing Club."
        session.add(run)
        session.flush()
        raise ValueError(run.error_summary)

    client = HotmartClubApiClient(
        api_base_url=status.api_base_url or default_hotmart_api_base_url(env),
        timeout_seconds=get_settings().hotmart_request_timeout_seconds,
        transport=transport,
    )
    try:
        token = HotmartAuthClient(
            environment=env,
            auth_base_url=status.auth_base_url,
            timeout_seconds=get_settings().hotmart_request_timeout_seconds,
            transport=transport,
        ).fetch_access_token(credentials)
        club_payloads = _fetch_club_payloads(client=client, access_token=token.access_token, payload=payload)
        issue_stats = (
            _reconcile_club_students(
                session,
                workspace_id=workspace_id,
                environment=env,
                students=club_payloads.students,
                actor_user_id=actor_user_id,
            )
            if payload.sync_students
            else SyncIssueStats(skipped=club_payloads.total_records)
        )
    except HotmartAuthError as exc:
        run.status = "failed"
        run.finished_at = utc_now()
        run.error_summary = "Hotmart OAuth failed while syncing Club."
        run.metadata_payload = {"error_code": exc.code, "error_payload_redacted": exc.payload}
        session.add(run)
        session.flush()
        raise HotmartClubError("club_auth_failed", run.error_summary, http_status=exc.http_status, payload=exc.payload) from exc
    except HotmartClubError as exc:
        run.status = "rate_limited" if exc.code == "rate_limited" else "failed"
        run.finished_at = utc_now()
        run.error_summary = str(exc)
        run.metadata_payload = {"error_code": exc.code, "error_payload_redacted": exc.payload}
        session.add(run)
        session.flush()
        raise

    _update_cursor(session, workspace_id=workspace_id, environment=env)
    _update_config_last_sync(session, workspace_id=workspace_id, environment=env)
    run.status = "succeeded"
    run.finished_at = utc_now()
    run.records_read = club_payloads.total_records
    run.records_created = issue_stats.created
    run.records_updated = issue_stats.updated
    run.records_skipped = issue_stats.skipped
    run.metadata_payload = {
        "issue_count": issue_stats.total,
        "subdomain": subdomain,
        "club_counts": {
            "modules": len(club_payloads.modules),
            "pages": len(club_payloads.pages),
            "students": len(club_payloads.students),
            "progress": len(club_payloads.progress),
        },
        "modules": redact_payload(club_payloads.modules),
        "pages": redact_payload(club_payloads.pages),
        "students": redact_payload(club_payloads.students),
        "progress": redact_payload(club_payloads.progress),
        "sync_options": {
            "sync_modules": payload.sync_modules,
            "sync_pages": payload.sync_pages,
            "sync_students": payload.sync_students,
            "sync_progress": payload.sync_progress,
            "module_id": payload.module_id,
            "user_id": payload.user_id,
            "is_extra": payload.is_extra,
        },
    }
    session.add(run)
    record_commercial_event(
        session,
        workspace_id=workspace_id,
        session_id=None,
        user_id=actor_user_id,
        event_key="hotmart_sync_completed",
        product_key=CLUB_RESOURCE,
        source="hotmart_club",
        metadata={
            "run_id": str(run.id),
            "records_read": run.records_read,
            "issues_created": run.records_created,
            "issues_updated": run.records_updated,
        },
        correlation_id=str(run.id),
    )
    session.flush()
    return _serialize_sync_run(run)


def get_hotmart_club_overview(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
) -> HotmartClubOverviewResponse:
    env = normalize_hotmart_environment(environment)
    latest_run = _latest_club_run(session, workspace_id=workspace_id, environment=env)
    issue_count = int(
        session.exec(
            select(func.count()).select_from(HotmartReconciliationIssueRecord).where(
                HotmartReconciliationIssueRecord.workspace_id == workspace_id,
                HotmartReconciliationIssueRecord.environment == env,
                HotmartReconciliationIssueRecord.status == "open",
                HotmartReconciliationIssueRecord.issue_type.in_(CLUB_ISSUE_TYPES),
            )
        ).one()
    )
    counts = latest_run.metadata_payload.get("club_counts", {}) if latest_run is not None else {}
    return HotmartClubOverviewResponse(
        workspace_id=workspace_id,
        environment=env,  # type: ignore[arg-type]
        subdomain=str(latest_run.metadata_payload.get("subdomain") or "") if latest_run is not None else "",
        modules_count=_as_int(counts.get("modules") if isinstance(counts, dict) else 0),
        pages_count=_as_int(counts.get("pages") if isinstance(counts, dict) else 0),
        students_count=_as_int(counts.get("students") if isinstance(counts, dict) else 0),
        progress_count=_as_int(counts.get("progress") if isinstance(counts, dict) else 0),
        open_issue_count=issue_count,
        last_sync_status=latest_run.status if latest_run is not None else "idle",
        last_sync_at=latest_run.finished_at or latest_run.started_at if latest_run is not None else None,
    )


def list_hotmart_club_modules(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
    limit: int = 100,
) -> list[HotmartClubModuleResponse]:
    env = normalize_hotmart_environment(environment)
    run = _latest_club_run(session, workspace_id=workspace_id, environment=env)
    return [_serialize_module(item) for item in _club_snapshot_list(run, "modules")[: max(1, min(limit, 200))]]


def list_hotmart_club_pages(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
    limit: int = 100,
) -> list[HotmartClubPageResponse]:
    env = normalize_hotmart_environment(environment)
    run = _latest_club_run(session, workspace_id=workspace_id, environment=env)
    return [_serialize_page(item) for item in _club_snapshot_list(run, "pages")[: max(1, min(limit, 200))]]


def list_hotmart_club_students(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
    limit: int = 100,
) -> list[HotmartClubStudentResponse]:
    env = normalize_hotmart_environment(environment)
    run = _latest_club_run(session, workspace_id=workspace_id, environment=env)
    return [_serialize_student(item) for item in _club_snapshot_list(run, "students")[: max(1, min(limit, 200))]]


def list_hotmart_club_progress(
    session: Session,
    *,
    workspace_id: UUID,
    environment: str = "sandbox",
    limit: int = 100,
) -> list[HotmartClubProgressResponse]:
    env = normalize_hotmart_environment(environment)
    run = _latest_club_run(session, workspace_id=workspace_id, environment=env)
    return [_serialize_progress(item) for item in _club_snapshot_list(run, "progress")[: max(1, min(limit, 200))]]
