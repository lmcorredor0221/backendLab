from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from app.models import (
    SessionSnapshot,
    ShortTermBranchBoardEntry,
    ShortTermBranchRecord,
    ShortTermCheckpointEntry,
    ShortTermCheckpointRecord,
    ShortTermMemoryRollbackRequest,
    ShortTermMemoryRuntimeState,
    ShortTermMemoryState,
    ShortTermMemoryStateNamespace,
    ShortTermSessionStateRecord,
    utc_now,
)
from app.services.canonical_exports import build_short_term_memory


MAIN_BRANCH_KEY = "main"
ACTIVE_STATUS = "active"
SUPERSEDED_STATUS = "superseded"
ROLLED_BACK_STATUS = "rolled_back"
INACTIVE_STATUS = "inactive"


def _stable_hash_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable_hash_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable_hash_payload(item) for item in value]
    return value


def _state_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_stable_hash_payload(payload), ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _branch_namespace(branch_key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", ".", branch_key.lower()).strip(".")
    return f"session.branch.{normalized or 'branch'}"


def _infer_topology(branch_key: str) -> str:
    if branch_key.startswith("handoff:"):
        return "handoff"
    if branch_key.startswith("subagent_run:"):
        return "subagent"
    return "mainline"


def _infer_isolation_mode(branch_key: str) -> str:
    if branch_key.startswith("subagent_run:"):
        return "isolated_namespace"
    if branch_key.startswith("handoff:"):
        return "shared_contract"
    return "shared"


class ShortTermMemoryService:
    def resume_session_state(
        self,
        session: Session,
        *,
        session_id: UUID,
        snapshot: SessionSnapshot | None = None,
        source_action: str = "load_short_term_memory",
    ) -> ShortTermMemoryRuntimeState:
        session_state = self._load_session_state(session, session_id=session_id)
        if session_state is None:
            if snapshot is None:
                raise ValueError("Session snapshot is required to initialize short-term memory")
            return self.capture_session_state(
                session,
                session_id=session_id,
                snapshot=snapshot,
                source_action=source_action,
            )
        if snapshot is not None and source_action not in {"load_session_snapshot", "load_short_term_memory"}:
            self._sync_branch_board(session, session_id=session_id, snapshot=snapshot, memory_state=None)
        return self.build_runtime_state(session, session_id=session_id)

    def reload_from_snapshot(
        self,
        session: Session,
        *,
        session_id: UUID,
        snapshot: SessionSnapshot,
        source_action: str = "reload_short_term_memory",
        branch_key: str = MAIN_BRANCH_KEY,
    ) -> ShortTermMemoryRuntimeState:
        return self.capture_session_state(
            session,
            session_id=session_id,
            snapshot=snapshot,
            source_action=source_action,
            branch_key=branch_key,
        )

    def capture_session_state(
        self,
        session: Session,
        *,
        session_id: UUID,
        snapshot: SessionSnapshot,
        source_action: str,
        branch_key: str = MAIN_BRANCH_KEY,
    ) -> ShortTermMemoryRuntimeState:
        canonical_state = build_short_term_memory(snapshot)
        memory_state = self._project_runtime_state(
            self._runtime_state_from_canonical_payload(canonical_state.model_dump(mode="json")),
            branch_key=branch_key,
        )
        payload = memory_state.model_dump(mode="json")
        state_hash = _state_hash(payload)
        now = utc_now()

        self._sync_branch_board(
            session,
            session_id=session_id,
            snapshot=snapshot,
            memory_state=memory_state,
            active_branch_key=branch_key,
        )
        branch = self._ensure_branch(
            session,
            session_id=session_id,
            branch_key=branch_key,
            title=self._branch_title(branch_key, snapshot),
            topology=_infer_topology(branch_key),
            stage=memory_state.active_stage,
            status=ACTIVE_STATUS,
            isolation_mode=_infer_isolation_mode(branch_key),
            summary=memory_state.current_focus or memory_state.active_goal,
            namespace_keys=[item.namespace for item in memory_state.namespaces],
        )

        session_state = self._load_session_state(session, session_id=session_id)
        current_checkpoint = self._load_active_checkpoint(session, session_id=session_id, branch_key=branch_key)
        if (
            session_state is not None
            and session_state.active_branch_key == branch_key
            and session_state.state_hash == state_hash
            and session_state.active_checkpoint_key
            and current_checkpoint is not None
        ):
            branch.last_activity_at = now
            branch.updated_at = now
            session.add(branch)
            return self.build_runtime_state(session, session_id=session_id)

        if current_checkpoint is not None:
            current_checkpoint.is_active = False
            current_checkpoint.updated_at = now
            if current_checkpoint.status == ACTIVE_STATUS:
                current_checkpoint.status = SUPERSEDED_STATUS
            session.add(current_checkpoint)

        checkpoint_number = branch.checkpoint_count + 1
        checkpoint_key = f"{branch_key}:cp{checkpoint_number}"
        checkpoint = ShortTermCheckpointRecord(
            session_id=session_id,
            branch_key=branch_key,
            checkpoint_key=checkpoint_key,
            parent_checkpoint_key=branch.active_checkpoint_key,
            checkpoint_number=checkpoint_number,
            stage=memory_state.active_stage,
            source_action=source_action,
            status=ACTIVE_STATUS,
            summary=memory_state.current_focus or memory_state.active_goal,
            state_hash=state_hash,
            state_payload=payload,
            is_consistent=True,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        session.add(checkpoint)

        branch.checkpoint_count = checkpoint_number
        branch.active_checkpoint_key = checkpoint_key
        branch.last_consistent_checkpoint_key = checkpoint_key
        branch.stage = memory_state.active_stage
        branch.status = ACTIVE_STATUS
        branch.summary = memory_state.current_focus or memory_state.active_goal
        branch.namespace_keys = [item.namespace for item in memory_state.namespaces]
        branch.last_activity_at = now
        branch.updated_at = now
        session.add(branch)

        if session_state is None:
            session_state = ShortTermSessionStateRecord(
                session_id=session_id,
                active_branch_key=branch_key,
                active_checkpoint_key=checkpoint_key,
                last_consistent_checkpoint_key=checkpoint_key,
                source_action=source_action,
                state_hash=state_hash,
                state_payload=payload,
                updated_at=now,
            )
        else:
            session_state.active_branch_key = branch_key
            session_state.active_checkpoint_key = checkpoint_key
            session_state.last_consistent_checkpoint_key = checkpoint_key
            session_state.source_action = source_action
            session_state.state_hash = state_hash
            session_state.state_payload = payload
            session_state.updated_at = now
        session.add(session_state)
        session.flush()
        return self.build_runtime_state(session, session_id=session_id)

    def rollback_session_state(
        self,
        session: Session,
        *,
        session_id: UUID,
        payload: ShortTermMemoryRollbackRequest,
    ) -> ShortTermMemoryRuntimeState:
        session_state = self._load_session_state(session, session_id=session_id)
        if session_state is None:
            raise ValueError("Short-term memory has not been initialized for this session")

        branch_key = (payload.branch_key or session_state.active_branch_key or MAIN_BRANCH_KEY).strip() or MAIN_BRANCH_KEY
        active_checkpoint = self._load_active_checkpoint(session, session_id=session_id, branch_key=branch_key)
        target = self._resolve_rollback_target(
            session,
            session_id=session_id,
            branch_key=branch_key,
            checkpoint_key=payload.checkpoint_key,
            active_checkpoint_key=active_checkpoint.checkpoint_key if active_checkpoint is not None else "",
        )
        if target is None:
            raise ValueError("No consistent checkpoint is available for rollback")

        now = utc_now()
        if active_checkpoint is not None and active_checkpoint.id != target.id:
            active_checkpoint.is_active = False
            active_checkpoint.status = ROLLED_BACK_STATUS
            active_checkpoint.rollback_note = payload.reason.strip()
            active_checkpoint.updated_at = now
            session.add(active_checkpoint)

        target.is_active = True
        target.status = ACTIVE_STATUS
        target.updated_at = now
        session.add(target)

        branch = self._load_branch(session, session_id=session_id, branch_key=branch_key)
        if branch is None:
            raise ValueError("Short-term memory branch not found")
        branch.active_checkpoint_key = target.checkpoint_key
        branch.last_consistent_checkpoint_key = target.checkpoint_key
        branch.stage = target.stage
        branch.status = ACTIVE_STATUS
        branch.last_activity_at = now
        branch.updated_at = now
        session.add(branch)

        session_state.active_branch_key = branch_key
        session_state.active_checkpoint_key = target.checkpoint_key
        session_state.last_consistent_checkpoint_key = target.checkpoint_key
        session_state.source_action = "rollback_short_term_memory"
        session_state.state_hash = target.state_hash
        session_state.state_payload = target.state_payload
        session_state.updated_at = now
        session.add(session_state)
        session.flush()
        return self.build_runtime_state(session, session_id=session_id)

    def build_runtime_state(self, session: Session, *, session_id: UUID) -> ShortTermMemoryRuntimeState:
        session_state = self._load_session_state(session, session_id=session_id)
        if session_state is None:
            raise ValueError("Short-term memory has not been initialized for this session")

        memory_state = ShortTermMemoryState.model_validate(session_state.state_payload)
        branches = session.exec(
            select(ShortTermBranchRecord)
            .where(ShortTermBranchRecord.session_id == session_id)
            .order_by(ShortTermBranchRecord.last_activity_at.desc(), ShortTermBranchRecord.created_at.desc())
        ).all()
        checkpoints = session.exec(
            select(ShortTermCheckpointRecord)
            .where(ShortTermCheckpointRecord.session_id == session_id)
            .order_by(ShortTermCheckpointRecord.updated_at.desc(), ShortTermCheckpointRecord.created_at.desc())
        ).all()

        branch_board = [
            ShortTermBranchBoardEntry(
                branch_key=item.branch_key,
                parent_branch_key=item.parent_branch_key,
                title=item.title,
                topology=item.topology,
                stage=item.stage,
                status=item.status,
                isolation_mode=item.isolation_mode,
                summary=item.summary,
                namespace_keys=item.namespace_keys,
                checkpoint_count=item.checkpoint_count,
                active_checkpoint_key=item.active_checkpoint_key,
                last_consistent_checkpoint_key=item.last_consistent_checkpoint_key,
                last_activity_at=item.last_activity_at,
            )
            for item in branches
        ]
        checkpoint_history = [
            ShortTermCheckpointEntry(
                checkpoint_key=item.checkpoint_key,
                branch_key=item.branch_key,
                checkpoint_number=item.checkpoint_number,
                parent_checkpoint_key=item.parent_checkpoint_key,
                stage=item.stage,
                source_action=item.source_action,
                status=item.status,
                summary=item.summary,
                state_hash=item.state_hash,
                is_consistent=item.is_consistent,
                is_active=item.is_active,
                rollback_note=item.rollback_note,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in checkpoints[:20]
        ]
        rollback_available = any(
            item.branch_key == session_state.active_branch_key
            and item.is_consistent
            and item.checkpoint_key != session_state.active_checkpoint_key
            for item in checkpoints
        )

        return ShortTermMemoryRuntimeState(
            session_id=session_id,
            source_action=session_state.source_action,
            active_branch_key=session_state.active_branch_key,
            active_checkpoint_key=session_state.active_checkpoint_key,
            last_consistent_checkpoint_key=session_state.last_consistent_checkpoint_key,
            rollback_available=rollback_available,
            branch_count=len(branch_board),
            checkpoint_count=len(checkpoints),
            memory=memory_state,
            branch_board=branch_board,
            checkpoint_history=checkpoint_history,
            updated_at=session_state.updated_at,
        )

    def _resolve_rollback_target(
        self,
        session: Session,
        *,
        session_id: UUID,
        branch_key: str,
        checkpoint_key: str | None,
        active_checkpoint_key: str,
    ) -> ShortTermCheckpointRecord | None:
        if checkpoint_key:
            return session.exec(
                select(ShortTermCheckpointRecord).where(
                    ShortTermCheckpointRecord.session_id == session_id,
                    ShortTermCheckpointRecord.checkpoint_key == checkpoint_key,
                )
            ).first()

        checkpoints = session.exec(
            select(ShortTermCheckpointRecord)
            .where(
                ShortTermCheckpointRecord.session_id == session_id,
                ShortTermCheckpointRecord.branch_key == branch_key,
                ShortTermCheckpointRecord.is_consistent == True,  # noqa: E712
            )
            .order_by(ShortTermCheckpointRecord.checkpoint_number.desc())
        ).all()
        for item in checkpoints:
            if item.checkpoint_key != active_checkpoint_key:
                return item
        return checkpoints[0] if checkpoints else None

    def _project_runtime_state(self, state: ShortTermMemoryState, *, branch_key: str) -> ShortTermMemoryState:
        projected = state.model_copy(deep=True)
        if projected.branch_refs and not any(item.namespace == "session.branch_board" for item in projected.namespaces):
            projected.namespaces.append(
                ShortTermMemoryStateNamespace(
                    namespace="session.branch_board",
                    summary="Board operativo de ramas, handoffs y ejecuciones paralelas.",
                    ref_keys=[item.key for item in projected.branch_refs],
                    freshness="refresh_on_branch_activity",
                    read_roles=["planner", "executor", "recovery", "memory"],
                    write_roles=["supervisor", "subagent", "memory"],
                )
            )

        if branch_key == MAIN_BRANCH_KEY:
            return projected

        projected.branch_refs = [item for item in projected.branch_refs if item.key == branch_key]
        branch_namespace = _branch_namespace(branch_key)
        projected.namespaces = [
            item
            for item in projected.namespaces
            if item.namespace not in {"session.branch_board"} and not item.namespace.startswith("session.branch.")
        ]
        projected.namespaces.append(
            ShortTermMemoryStateNamespace(
                namespace=branch_namespace,
                summary="Namespace aislado para el checkpoint operativo de la rama activa.",
                ref_keys=[item.key for item in projected.branch_refs],
                freshness="refresh_on_branch_activity",
                read_roles=["supervisor", "specialist", "recovery", "memory"],
                write_roles=["supervisor", "specialist", "memory"],
            )
        )
        if branch_key.startswith("subagent_run:"):
            projected.current_focus = projected.current_focus or "Resolver la rama especializada sin contaminar otras ramas."
        return projected

    def _runtime_state_from_canonical_payload(self, payload: dict[str, Any]) -> ShortTermMemoryState:
        allowed_fields = set(ShortTermMemoryState.model_fields.keys())
        return ShortTermMemoryState.model_validate({key: value for key, value in payload.items() if key in allowed_fields})

    def _sync_branch_board(
        self,
        session: Session,
        *,
        session_id: UUID,
        snapshot: SessionSnapshot,
        memory_state: ShortTermMemoryState | None,
        active_branch_key: str = MAIN_BRANCH_KEY,
    ) -> None:
        main_namespace_keys = (
            [item.namespace for item in memory_state.namespaces]
            if memory_state is not None and active_branch_key == MAIN_BRANCH_KEY
            else []
        )
        self._ensure_branch(
            session,
            session_id=session_id,
            branch_key=MAIN_BRANCH_KEY,
            title="Rama principal",
            topology="mainline",
            stage=memory_state.active_stage if memory_state is not None else "",
            status=ACTIVE_STATUS,
            isolation_mode="shared",
            summary=(memory_state.current_focus or memory_state.active_goal) if memory_state is not None else "",
            namespace_keys=main_namespace_keys,
        )

        seen = {MAIN_BRANCH_KEY}
        for item in snapshot.handoff_records:
            branch_key = f"handoff:{item.handoff_key or item.id}"
            seen.add(branch_key)
            self._ensure_branch(
                session,
                session_id=session_id,
                branch_key=branch_key,
                title=item.title or item.handoff_key or "Handoff",
                topology="handoff",
                stage=(
                    f"{item.from_stage.value if hasattr(item.from_stage, 'value') else item.from_stage}"
                    f"->{item.to_stage.value if hasattr(item.to_stage, 'value') else item.to_stage}"
                ),
                status=item.status,
                isolation_mode="shared_contract",
                summary=item.summary,
                namespace_keys=["session.recovery.current", _branch_namespace(branch_key)],
            )

        for item in snapshot.subagent_runs:
            branch_key = f"subagent_run:{item.id}"
            seen.add(branch_key)
            self._ensure_branch(
                session,
                session_id=session_id,
                branch_key=branch_key,
                title=item.title or item.run_kind or "Subagente",
                topology="subagent",
                stage="branch_execution",
                status=item.status.value if hasattr(item.status, "value") else str(item.status),
                isolation_mode="isolated_namespace",
                summary=item.summary,
                namespace_keys=["session.short_term.execution", _branch_namespace(branch_key)],
            )

        existing = session.exec(select(ShortTermBranchRecord).where(ShortTermBranchRecord.session_id == session_id)).all()
        for item in existing:
            if item.branch_key in seen:
                continue
            if item.branch_key == MAIN_BRANCH_KEY:
                continue
            if item.status != INACTIVE_STATUS:
                item.status = INACTIVE_STATUS
                item.updated_at = utc_now()
                session.add(item)

    def _ensure_branch(
        self,
        session: Session,
        *,
        session_id: UUID,
        branch_key: str,
        title: str,
        topology: str,
        stage: str,
        status: str,
        isolation_mode: str,
        summary: str,
        namespace_keys: list[str],
    ) -> ShortTermBranchRecord:
        record = self._load_branch(session, session_id=session_id, branch_key=branch_key)
        now = utc_now()
        if record is None:
            record = ShortTermBranchRecord(
                session_id=session_id,
                branch_key=branch_key,
                title=title,
                topology=topology,
                stage=stage,
                status=status,
                isolation_mode=isolation_mode,
                summary=summary,
                namespace_keys=namespace_keys,
                created_at=now,
                updated_at=now,
                last_activity_at=now,
            )
        else:
            record.title = title or record.title
            record.topology = topology or record.topology
            record.stage = stage or record.stage
            record.status = status or record.status
            record.isolation_mode = isolation_mode or record.isolation_mode
            record.summary = summary or record.summary
            record.namespace_keys = namespace_keys or record.namespace_keys
            record.updated_at = now
            record.last_activity_at = now
        session.add(record)
        session.flush()
        return record

    def _branch_title(self, branch_key: str, snapshot: SessionSnapshot) -> str:
        if branch_key == MAIN_BRANCH_KEY:
            return "Rama principal"
        if branch_key.startswith("handoff:"):
            return next(
                (
                    item.title
                    for item in snapshot.handoff_records
                    if f"handoff:{item.handoff_key or item.id}" == branch_key
                ),
                "Handoff",
            )
        if branch_key.startswith("subagent_run:"):
            return next(
                (
                    item.title
                    for item in snapshot.subagent_runs
                    if f"subagent_run:{item.id}" == branch_key
                ),
                "Subagente",
            )
        return branch_key

    def _load_session_state(self, session: Session, *, session_id: UUID) -> ShortTermSessionStateRecord | None:
        return session.exec(
            select(ShortTermSessionStateRecord).where(ShortTermSessionStateRecord.session_id == session_id)
        ).first()

    def _load_branch(self, session: Session, *, session_id: UUID, branch_key: str) -> ShortTermBranchRecord | None:
        return session.exec(
            select(ShortTermBranchRecord).where(
                ShortTermBranchRecord.session_id == session_id,
                ShortTermBranchRecord.branch_key == branch_key,
            )
        ).first()

    def _load_active_checkpoint(
        self,
        session: Session,
        *,
        session_id: UUID,
        branch_key: str,
    ) -> ShortTermCheckpointRecord | None:
        return session.exec(
            select(ShortTermCheckpointRecord).where(
                ShortTermCheckpointRecord.session_id == session_id,
                ShortTermCheckpointRecord.branch_key == branch_key,
                ShortTermCheckpointRecord.is_active == True,  # noqa: E712
            )
        ).first()
