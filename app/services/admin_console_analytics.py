from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    AdminUserInvitationRecord,
    ArtifactStatus,
    AuthTokenRecord,
    LLMUsageLedgerRecord,
    PlatformRole,
    PlatformRoleAssignmentRecord,
    RuntimeGovernanceScopeType,
    RuntimeSettingsAuditRecord,
    SessionRecord,
    SessionStage,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.llm_finops.analytics_service import LLMUsageAnalyticsFilters, LLMUsageAnalyticsService


DEFAULT_ADMIN_PERIOD_DAYS = 30


@dataclass(frozen=True)
class AdminAnalyticsFilters:
    workspace_id: UUID
    started_from: datetime | None = None
    started_to: datetime | None = None
    user_id: UUID | None = None
    project_id: UUID | None = None
    stage: str = ""
    provider_key: str = ""
    model_name: str = ""
    granularity: str = "day"


class AdminConsoleAnalyticsService:
    def overview(
        self,
        session: Session,
        *,
        workspace: WorkspaceRecord,
        filters: AdminAnalyticsFilters,
    ) -> dict[str, Any]:
        normalized_filters = _filters_with_default_period(filters)
        llm_filters = _llm_filters(normalized_filters)
        llm_service = LLMUsageAnalyticsService()
        llm_summary = llm_service.summarize(session, llm_filters)
        projects = self.project_analytics(session, filters=normalized_filters)
        users = self.users_summary(session, filters=normalized_filters)
        activity = self.activity_feed(session, filters=normalized_filters, limit=8)

        return {
            "workspace": {
                "id": str(workspace.id),
                "name": workspace.name,
                "slug": workspace.slug,
            },
            "period": _period_payload(normalized_filters),
            "filters": _filters_payload(normalized_filters),
            "availability": {
                "llm_usage": _availability(
                    "available" if llm_summary["call_count"] else "empty",
                    source="llm_usage_ledger",
                    reason="Ledger LLM filtrado por workspace y periodo.",
                ),
                "projects": _availability(
                    "available" if projects["total"] else "empty",
                    source="sessions",
                    reason="Sesiones/proyectos filtrados por workspace y periodo.",
                ),
                "users": _availability(
                    "available" if users["total"] else "empty",
                    source="workspace_memberships + users + auth_tokens",
                    reason="Usuarios por membresia del workspace; actividad aproximada por ultimo token/uso registrado.",
                ),
                "connected_users": _availability(
                    "not_instrumented",
                    source="auth_tokens",
                    reason=(
                        "No existe heartbeat/presencia en tiempo real. Se expone actividad reciente, "
                        "no conexion en vivo."
                    ),
                ),
                "project_finalized_at": _availability(
                    "not_instrumented",
                    source="sessions.current_stage",
                    reason="El modelo tiene estado actual, pero no timestamp historico de finalizacion.",
                ),
            },
            "llm": {
                "summary": llm_summary,
                "provider_breakdown": llm_service.provider_breakdown(session, llm_filters),
            },
            "projects": projects,
            "users": users,
            "activity": activity,
        }

    def project_analytics(self, session: Session, *, filters: AdminAnalyticsFilters) -> dict[str, Any]:
        records = _project_records(session, filters)
        total = len(records)
        active_records = [record for record in records if _is_active_project(record)]
        finalized_records = [record for record in records if _is_finalized_project(record)]
        archived_records = [record for record in records if record.archived_at is not None]
        deleted_records = [record for record in records if record.deleted_at is not None]

        by_stage: dict[str, dict[str, Any]] = {}
        by_status: dict[str, dict[str, Any]] = {}
        for record in records:
            stage = _enum_value(record.current_stage) or "unknown"
            status = _enum_value(record.status) or "unknown"
            by_stage.setdefault(stage, {"stage": stage, "count": 0, "percentage": 0.0})["count"] += 1
            by_status.setdefault(status, {"status": status, "count": 0, "percentage": 0.0})["count"] += 1
        for bucket in by_stage.values():
            bucket["percentage"] = round(bucket["count"] / total, 4) if total else 0
        for bucket in by_status.values():
            bucket["percentage"] = round(bucket["count"] / total, 4) if total else 0

        created_series = _datetime_series(
            [record.created_at for record in records],
            granularity=filters.granularity,
            metric_key="created_count",
        )

        return {
            "total": total,
            "active": len(active_records),
            "finalized": len(finalized_records),
            "archived": len(archived_records),
            "deleted": len(deleted_records),
            "distribution_by_stage": _sorted_buckets(by_stage, key_name="stage"),
            "distribution_by_status": _sorted_buckets(by_status, key_name="status"),
            "created_series": {
                "items": created_series,
                "availability": _availability(
                    "available" if created_series else "empty",
                    source="sessions.created_at",
                    reason="Serie calculada desde fecha de creacion de sesiones/proyectos.",
                ),
            },
            "finalized_series": {
                "items": [],
                "availability": _availability(
                    "not_instrumented",
                    source="sessions.current_stage",
                    reason="No existe finalized_at ni historial de transiciones para calcular una serie confiable.",
                ),
            },
            "definitions": {
                "active_project": "Proyecto no archivado, no eliminado y no finalizado.",
                "finalized_project": "Snapshot: current_stage=ready_for_export o status=ready.",
            },
            "period": _period_payload(filters),
        }

    def users_summary(self, session: Session, *, filters: AdminAnalyticsFilters) -> dict[str, Any]:
        directory = self.list_users(session, filters=filters, limit=500, offset=0)
        items = directory["items"]
        total = directory["count"]
        active = sum(1 for item in items if item["is_active"] and item["membership"]["is_active"])
        inactive = total - active
        recently_active = sum(1 for item in items if item["activity"]["is_recently_active"])
        new_users = sum(1 for item in items if _in_period(_parse_iso(item["created_at"]), filters))
        by_role: dict[str, dict[str, Any]] = {}
        for item in items:
            role = str(item["membership"]["role"] or "unassigned")
            by_role.setdefault(role, {"role": role, "count": 0, "percentage": 0.0})["count"] += 1
        for bucket in by_role.values():
            bucket["percentage"] = round(bucket["count"] / total, 4) if total else 0
        return {
            "total": total,
            "active": active,
            "inactive": inactive,
            "recently_active": recently_active,
            "new_users": new_users,
            "connected": None,
            "connected_availability": _availability(
                "not_instrumented",
                source="auth_tokens",
                reason="No existe presencia/heartbeat; connected queda nulo y no debe graficarse como real.",
            ),
            "distribution_by_role": _sorted_buckets(by_role, key_name="role"),
            "period": _period_payload(filters),
        }

    def list_users(
        self,
        session: Session,
        *,
        filters: AdminAnalyticsFilters,
        search: str = "",
        role: str = "",
        status_filter: str = "all",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        membership_statement = select(WorkspaceMembershipRecord).where(
            WorkspaceMembershipRecord.workspace_id == filters.workspace_id
        )
        if role.strip():
            try:
                membership_statement = membership_statement.where(WorkspaceMembershipRecord.role == WorkspaceRole(role))
            except ValueError:
                return {"items": [], "count": 0, "limit": limit, "offset": offset}

        memberships = list(session.exec(membership_statement).all())
        user_ids = [membership.user_id for membership in memberships]
        if not user_ids:
            return {"items": [], "count": 0, "limit": limit, "offset": offset}

        users = list(session.exec(select(UserRecord).where(UserRecord.id.in_(user_ids))).all())
        users_by_id = {user.id: user for user in users}
        activity_by_user = _last_activity_by_user(session, workspace_id=filters.workspace_id, user_ids=user_ids)
        period_start = _period_start_for_activity(filters)
        query = search.strip().lower()

        rows: list[dict[str, Any]] = []
        for membership in memberships:
            user = users_by_id.get(membership.user_id)
            if user is None:
                continue
            if query and query not in user.email.lower() and query not in user.full_name.lower():
                continue
            if status_filter == "active" and (not user.is_active or not membership.is_active):
                continue
            if status_filter == "inactive" and user.is_active and membership.is_active:
                continue

            last_activity_at = activity_by_user.get(user.id)
            rows.append(
                {
                    "id": str(user.id),
                    "email": user.email,
                    "full_name": user.full_name,
                    "is_active": user.is_active,
                    "email_verified": user.email_verified,
                    "preferred_language": user.preferred_language,
                    "preferred_currency": user.preferred_currency,
                    "created_at": _iso(user.created_at),
                    "updated_at": _iso(user.updated_at),
                    "membership": {
                        "id": str(membership.id),
                        "workspace_id": str(membership.workspace_id),
                        "role": _enum_value(membership.role),
                        "is_active": membership.is_active,
                        "created_at": _iso(membership.created_at),
                        "updated_at": _iso(membership.updated_at),
                    },
                    "activity": {
                        "last_activity_at": _iso(last_activity_at),
                        "is_recently_active": bool(last_activity_at and period_start and last_activity_at >= period_start),
                        "activity_definition": (
                            "Maximo entre ultimo token usado, ultimo uso LLM y ultima actualizacion de proyecto "
                            "dentro del workspace."
                        ),
                    },
                }
            )

        rows.sort(key=lambda item: item["activity"]["last_activity_at"] or item["updated_at"], reverse=True)
        count = len(rows)
        safe_offset = max(0, offset)
        safe_limit = max(1, min(limit, 500))
        return {
            "items": rows[safe_offset : safe_offset + safe_limit],
            "count": count,
            "limit": safe_limit,
            "offset": safe_offset,
        }

    def list_invitations(
        self,
        session: Session,
        *,
        filters: AdminAnalyticsFilters,
        status_filter: str = "pending",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        statement = select(AdminUserInvitationRecord).where(
            AdminUserInvitationRecord.workspace_id == filters.workspace_id
        )
        if status_filter.strip() and status_filter != "all":
            statement = statement.where(AdminUserInvitationRecord.status == status_filter.strip())
        rows = list(
            session.exec(
                statement.order_by(AdminUserInvitationRecord.created_at.desc())
                .offset(max(0, offset))
                .limit(max(1, min(limit, 500)))
            ).all()
        )
        return {
            "items": [_invitation_payload(row) for row in rows],
            "count": len(rows),
            "limit": max(1, min(limit, 500)),
            "offset": max(0, offset),
        }

    def roles_catalog(
        self,
        session: Session,
        *,
        workspace_id: UUID,
        current_user_id: UUID,
    ) -> dict[str, Any]:
        workspace_membership = session.exec(
            select(WorkspaceMembershipRecord).where(
                WorkspaceMembershipRecord.workspace_id == workspace_id,
                WorkspaceMembershipRecord.user_id == current_user_id,
                WorkspaceMembershipRecord.is_active == True,  # noqa: E712
            )
        ).first()
        platform_assignments = list(
            session.exec(
                select(PlatformRoleAssignmentRecord).where(
                    PlatformRoleAssignmentRecord.user_id == current_user_id,
                    PlatformRoleAssignmentRecord.is_active == True,  # noqa: E712
                )
            ).all()
        )
        return {
            "workspace_roles": [_workspace_role_payload(role) for role in WorkspaceRole],
            "platform_roles": [_platform_role_payload(role) for role in PlatformRole],
            "effective": {
                "workspace": _enum_value(workspace_membership.role) if workspace_membership else None,
                "platform": [_enum_value(item.role) for item in platform_assignments],
            },
            "definitions": {
                "permission_origin": "static_rbac_contract",
                "workspace_scope": str(workspace_id),
            },
        }

    def activity_feed(
        self,
        session: Session,
        *,
        filters: AdminAnalyticsFilters,
        limit: int = 50,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []

        audits = list(
            session.exec(
                select(RuntimeSettingsAuditRecord).where(
                    RuntimeSettingsAuditRecord.scope_id == str(filters.workspace_id)
                )
            ).all()
        )
        for audit in audits:
            if not _in_period(audit.created_at, filters):
                continue
            items.append(
                {
                    "id": str(audit.id),
                    "type": "audit",
                    "severity": "info",
                    "source": "runtime_settings_audit",
                    "title": audit.change_type,
                    "actor_user_id": str(audit.actor_user_id) if audit.actor_user_id else None,
                    "actor_email": audit.actor_email,
                    "created_at": _iso(audit.created_at),
                    "metadata": {
                        "before": audit.before_payload_redacted,
                        "after": audit.after_payload_redacted,
                    },
                }
            )

        failed_calls = list(
            session.exec(
                select(LLMUsageLedgerRecord).where(
                    LLMUsageLedgerRecord.workspace_id == filters.workspace_id,
                    LLMUsageLedgerRecord.status != "succeeded",
                )
            ).all()
        )
        for record in failed_calls:
            if not _in_period(record.started_at, filters):
                continue
            items.append(
                {
                    "id": str(record.id),
                    "type": "llm_usage",
                    "severity": "warning",
                    "source": "llm_usage_ledger",
                    "title": f"LLM {record.status}: {record.provider_key}/{record.model_name}",
                    "actor_user_id": str(record.user_id) if record.user_id else None,
                    "actor_email": "",
                    "created_at": _iso(record.started_at),
                    "metadata": {
                        "stage": record.stage,
                        "failure_kind": record.failure_kind,
                        "retry_count": record.retry_count,
                        "fallback_used": record.fallback_used,
                    },
                }
            )

        sessions = list(
            session.exec(select(SessionRecord).where(SessionRecord.workspace_id == filters.workspace_id)).all()
        )
        for record in sessions:
            if not _in_period(record.created_at, filters):
                continue
            items.append(
                {
                    "id": str(record.id),
                    "type": "project",
                    "severity": "info",
                    "source": "sessions",
                    "title": f"Proyecto creado: {record.title}",
                    "actor_user_id": str(record.user_id),
                    "actor_email": "",
                    "created_at": _iso(record.created_at),
                    "metadata": {
                        "stage": _enum_value(record.current_stage),
                        "status": _enum_value(record.status),
                    },
                }
            )

        invitations = list(
            session.exec(
                select(AdminUserInvitationRecord).where(
                    AdminUserInvitationRecord.workspace_id == filters.workspace_id
                )
            ).all()
        )
        for invitation in invitations:
            if not _in_period(invitation.created_at, filters):
                continue
            items.append(
                {
                    "id": str(invitation.id),
                    "type": "user_invitation",
                    "severity": "info",
                    "source": "admin_user_invitations",
                    "title": f"Invitacion {invitation.status}: {invitation.email}",
                    "actor_user_id": str(invitation.invited_by_user_id) if invitation.invited_by_user_id else None,
                    "actor_email": "",
                    "created_at": _iso(invitation.created_at),
                    "metadata": {
                        "role": _enum_value(invitation.role),
                        "expires_at": _iso(invitation.expires_at),
                    },
                }
            )

        items.sort(key=lambda item: item["created_at"] or "", reverse=True)
        return {
            "items": items[: max(1, min(limit, 200))],
            "count": len(items),
            "availability": _availability(
                "partial" if items else "empty",
                source="runtime_settings_audit + llm_usage_ledger + sessions + admin_user_invitations",
                reason="Feed consolidado inicial; no existe event bus unico de actividad administrativa.",
            ),
        }


def audit_admin_change(
    session: Session,
    *,
    workspace_id: UUID,
    actor: UserRecord,
    change_type: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    session.add(
        RuntimeSettingsAuditRecord(
            scope_type=RuntimeGovernanceScopeType.workspace,
            scope_id=str(workspace_id),
            change_type=change_type,
            before_payload_redacted=before,
            after_payload_redacted=after,
            actor_user_id=actor.id,
            actor_email=actor.email,
        )
    )


def _project_records(session: Session, filters: AdminAnalyticsFilters) -> list[SessionRecord]:
    statement = select(SessionRecord).where(SessionRecord.workspace_id == filters.workspace_id)
    if filters.user_id is not None:
        statement = statement.where(SessionRecord.user_id == filters.user_id)
    if filters.project_id is not None:
        statement = statement.where(SessionRecord.id == filters.project_id)
    if filters.stage.strip():
        stage_value = filters.stage.strip()
        try:
            statement = statement.where(SessionRecord.current_stage == SessionStage(stage_value))
        except ValueError:
            return []
    if filters.started_from is not None:
        statement = statement.where(SessionRecord.created_at >= _to_naive_utc(filters.started_from))
    if filters.started_to is not None:
        statement = statement.where(SessionRecord.created_at <= _to_naive_utc(filters.started_to))
    return list(session.exec(statement).all())


def _last_activity_by_user(
    session: Session,
    *,
    workspace_id: UUID,
    user_ids: list[UUID],
) -> dict[UUID, datetime]:
    last_activity: dict[UUID, datetime] = {}

    tokens = list(session.exec(select(AuthTokenRecord).where(AuthTokenRecord.user_id.in_(user_ids))).all())
    for token in tokens:
        _remember_latest(last_activity, token.user_id, token.last_used_at)

    llm_records = list(
        session.exec(
            select(LLMUsageLedgerRecord).where(
                LLMUsageLedgerRecord.workspace_id == workspace_id,
                LLMUsageLedgerRecord.user_id.in_(user_ids),
            )
        ).all()
    )
    for record in llm_records:
        if record.user_id is not None:
            _remember_latest(last_activity, record.user_id, record.started_at)

    project_records = list(
        session.exec(
            select(SessionRecord).where(
                SessionRecord.workspace_id == workspace_id,
                SessionRecord.user_id.in_(user_ids),
            )
        ).all()
    )
    for record in project_records:
        _remember_latest(last_activity, record.user_id, record.updated_at)

    return last_activity


def _remember_latest(target: dict[UUID, datetime], user_id: UUID, value: datetime | None) -> None:
    normalized = _to_naive_utc(value)
    if normalized is None:
        return
    current = target.get(user_id)
    if current is None or normalized > current:
        target[user_id] = normalized


def _period_start_for_activity(filters: AdminAnalyticsFilters) -> datetime | None:
    if filters.started_from is not None:
        return _to_naive_utc(filters.started_from)
    return utc_now() - timedelta(days=DEFAULT_ADMIN_PERIOD_DAYS)


def _filters_with_default_period(filters: AdminAnalyticsFilters) -> AdminAnalyticsFilters:
    if filters.started_from is not None or filters.started_to is not None:
        return filters
    ended_at = utc_now()
    return AdminAnalyticsFilters(
        workspace_id=filters.workspace_id,
        started_from=ended_at - timedelta(days=DEFAULT_ADMIN_PERIOD_DAYS),
        started_to=ended_at,
        user_id=filters.user_id,
        project_id=filters.project_id,
        stage=filters.stage,
        provider_key=filters.provider_key,
        model_name=filters.model_name,
        granularity=filters.granularity,
    )


def _llm_filters(filters: AdminAnalyticsFilters) -> LLMUsageAnalyticsFilters:
    return LLMUsageAnalyticsFilters(
        workspace_id=filters.workspace_id,
        started_from=filters.started_from,
        started_to=filters.started_to,
        user_id=filters.user_id,
        project_id=filters.project_id,
        stage=filters.stage.strip(),
        provider_key=filters.provider_key.strip(),
        model_name=filters.model_name.strip(),
    )


def _is_active_project(record: SessionRecord) -> bool:
    return record.archived_at is None and record.deleted_at is None and not _is_finalized_project(record)


def _is_finalized_project(record: SessionRecord) -> bool:
    return record.current_stage == SessionStage.ready_for_export or record.status == ArtifactStatus.ready


def _datetime_series(values: list[datetime], *, granularity: str, metric_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, int] = {}
    for value in values:
        bucket = _bucket_label(value, granularity=granularity)
        grouped[bucket] = grouped.get(bucket, 0) + 1
    return [{"bucket": key, metric_key: count} for key, count in sorted(grouped.items())]


def _bucket_label(value: datetime, *, granularity: str) -> str:
    normalized = _to_naive_utc(value) or utc_now()
    if granularity == "month":
        return normalized.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    if granularity == "week":
        start = normalized - timedelta(days=normalized.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    return normalized.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _sorted_buckets(grouped: dict[str, dict[str, Any]], *, key_name: str) -> list[dict[str, Any]]:
    return sorted(grouped.values(), key=lambda item: (-int(item["count"]), str(item[key_name])))


def _workspace_role_payload(role: WorkspaceRole) -> dict[str, Any]:
    permissions = {
        WorkspaceRole.owner: [
            "workspace.manage",
            "settings.manage",
            "runtime.manage",
            "users.manage",
            "roles.manage",
            "billing.manage",
            "projects.manage",
        ],
        WorkspaceRole.admin: [
            "settings.manage",
            "runtime.manage",
            "users.manage",
            "billing.manage",
            "projects.manage",
        ],
        WorkspaceRole.editor: ["projects.create", "projects.edit", "settings.read", "analytics.read"],
        WorkspaceRole.viewer: ["projects.read", "settings.read", "analytics.read"],
    }[role]
    return {
        "key": _enum_value(role),
        "label": role.value.replace("_", " ").title(),
        "scope": "workspace",
        "permissions": permissions,
        "permission_count": len(permissions),
        "is_system": True,
    }


def _platform_role_payload(role: PlatformRole) -> dict[str, Any]:
    permissions = {
        PlatformRole.platform_admin: [
            "platform.manage",
            "platform.runtime.manage",
            "platform.governance.manage",
            "workspace.impersonation.disabled",
        ],
        PlatformRole.platform_operator: ["platform.runtime.read", "platform.audit.read"],
    }[role]
    return {
        "key": _enum_value(role),
        "label": role.value.replace("_", " ").title(),
        "scope": "platform",
        "permissions": permissions,
        "permission_count": len(permissions),
        "is_system": True,
    }


def _invitation_payload(record: AdminUserInvitationRecord) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "workspace_id": str(record.workspace_id),
        "email": record.email,
        "full_name": record.full_name,
        "role": _enum_value(record.role),
        "status": record.status,
        "invited_by_user_id": str(record.invited_by_user_id) if record.invited_by_user_id else None,
        "accepted_user_id": str(record.accepted_user_id) if record.accepted_user_id else None,
        "expires_at": _iso(record.expires_at),
        "message": record.message,
        "metadata": record.metadata_payload,
        "created_at": _iso(record.created_at),
        "updated_at": _iso(record.updated_at),
        "delivery_status": "manual_delivery_required",
    }


def _availability(status: str, *, source: str, reason: str) -> dict[str, str]:
    return {"status": status, "source": source, "reason": reason}


def _period_payload(filters: AdminAnalyticsFilters) -> dict[str, Any]:
    return {
        "started_from": _iso(filters.started_from),
        "started_to": _iso(filters.started_to),
        "granularity": filters.granularity,
        "timezone": "UTC",
    }


def _filters_payload(filters: AdminAnalyticsFilters) -> dict[str, Any]:
    return {
        "workspace_id": str(filters.workspace_id),
        "user_id": str(filters.user_id) if filters.user_id else None,
        "project_id": str(filters.project_id) if filters.project_id else None,
        "stage": filters.stage.strip(),
        "provider_key": filters.provider_key.strip(),
        "model_name": filters.model_name.strip(),
    }


def _in_period(value: datetime | None, filters: AdminAnalyticsFilters) -> bool:
    normalized = _to_naive_utc(value)
    if normalized is None:
        return False
    started_from = _to_naive_utc(filters.started_from)
    started_to = _to_naive_utc(filters.started_to)
    if started_from is not None and normalized < started_from:
        return False
    if started_to is not None and normalized > started_to:
        return False
    return True


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _to_naive_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _iso(value: datetime | None) -> str | None:
    normalized = _to_naive_utc(value)
    return normalized.isoformat() if normalized is not None else None


def _enum_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "")
