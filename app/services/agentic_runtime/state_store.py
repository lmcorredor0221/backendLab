from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import desc
from sqlmodel import Session, select

from app.models import ShortTermCheckpointRecord, utc_now
from app.services.agentic_runtime.contracts import BuilderAgentState


def _state_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _react_branch_key(stage: str) -> str:
    normalized_stage = str(stage or "").strip().lower() or "runtime"
    return f"react:{normalized_stage}"


class BuilderReActCheckpointStore:
    """Persists ReAct checkpoints through the existing checkpoint table only.

    ReAct state shares checkpoint persistence with short-term memory history, but it
    must not overwrite the active short-term session payload that powers the UI
    snapshot and branch runtime state.
    """

    @staticmethod
    def save(
        session: Session,
        state: BuilderAgentState,
        *,
        summary: str,
        source_action: str,
    ) -> str:
        if state.session_id is None:
            raise ValueError("No se puede persistir un checkpoint ReAct sin session_id.")
        checkpoint_key = state.checkpoint_id or f"react:{state.run_id}:checkpoint:{state.iteration}"
        stage_key = str(state.stage or "").strip().lower() or "runtime"
        branch_key = _react_branch_key(stage_key)
        payload = state.model_dump(mode="json")
        digest = _state_hash(payload)
        previous = session.exec(
            select(ShortTermCheckpointRecord)
            .where(ShortTermCheckpointRecord.session_id == state.session_id)
            .order_by(desc(ShortTermCheckpointRecord.created_at))
        ).first()
        active_records = session.exec(
            select(ShortTermCheckpointRecord).where(
                ShortTermCheckpointRecord.session_id == state.session_id,
                ShortTermCheckpointRecord.is_active == True,  # noqa: E712
                ShortTermCheckpointRecord.checkpoint_key.startswith("react:"),
                ShortTermCheckpointRecord.stage == stage_key,
            )
        ).all()
        for record in active_records:
            record.is_active = False
            record.updated_at = utc_now()
            session.add(record)

        checkpoint = session.exec(
            select(ShortTermCheckpointRecord).where(
                ShortTermCheckpointRecord.session_id == state.session_id,
                ShortTermCheckpointRecord.checkpoint_key == checkpoint_key,
            )
        ).first()
        if checkpoint is None:
            checkpoint = ShortTermCheckpointRecord(
                session_id=state.session_id,
                branch_key=branch_key,
                checkpoint_key=checkpoint_key,
                parent_checkpoint_key=previous.checkpoint_key if previous else "",
                checkpoint_number=(previous.checkpoint_number + 1) if previous else 1,
            )
        checkpoint.branch_key = branch_key
        checkpoint.stage = stage_key
        checkpoint.source_action = source_action
        checkpoint.status = state.status
        checkpoint.summary = summary
        checkpoint.state_hash = digest
        checkpoint.state_payload = payload
        checkpoint.is_consistent = True
        checkpoint.is_active = True
        checkpoint.updated_at = utc_now()
        session.add(checkpoint)
        session.flush()
        return checkpoint_key

    @staticmethod
    def load(
        session: Session,
        *,
        session_id: Any,
        checkpoint_id: str = "",
        stage: str = "",
        include_inactive: bool = False,
    ) -> BuilderAgentState | None:
        if checkpoint_id:
            record = session.exec(
                select(ShortTermCheckpointRecord).where(
                    ShortTermCheckpointRecord.session_id == session_id,
                    ShortTermCheckpointRecord.checkpoint_key == checkpoint_id,
                )
            ).first()
        else:
            stage_key = str(stage or "").strip().lower()
            statement = select(ShortTermCheckpointRecord).where(
                ShortTermCheckpointRecord.session_id == session_id,
                ShortTermCheckpointRecord.checkpoint_key.startswith("react:"),
            )
            if stage_key:
                statement = statement.where(ShortTermCheckpointRecord.stage == stage_key)
            if not include_inactive:
                statement = statement.where(ShortTermCheckpointRecord.is_active == True)  # noqa: E712
            records = session.exec(
                statement.order_by(desc(ShortTermCheckpointRecord.updated_at), desc(ShortTermCheckpointRecord.created_at))
            ).all()
            record = None
            for candidate in records:
                state = BuilderAgentState.model_validate(candidate.state_payload)
                if include_inactive and state.status in {"completed", "failed", "cancelled"}:
                    continue
                record = candidate
                break
        return BuilderAgentState.model_validate(record.state_payload) if record else None

    @staticmethod
    def mark_resume_requested(
        session: Session,
        *,
        session_id: Any,
        action: str,
        scope: str,
        expected_stage: str = "",
        source_action: str = "attention_resume_requested",
        summary: str = "Decision humana resuelta; checkpoint listo para reanudar.",
    ) -> BuilderAgentState | None:
        stage_key = str(expected_stage or "").strip().lower()
        state = BuilderReActCheckpointStore.load(
            session,
            session_id=session_id,
            stage=stage_key,
        ) if stage_key else BuilderReActCheckpointStore.load(session, session_id=session_id)
        if state is None and stage_key:
            state = BuilderReActCheckpointStore.load(
                session,
                session_id=session_id,
                stage=stage_key,
                include_inactive=True,
            )
        if state is None:
            return None
        if stage_key and str(state.stage or "").strip().lower() != stage_key:
            return None
        state = state.model_copy(
            update={
                "status": "running",
                "resume_action": action,
                "resume_scope": scope,
                "updated_at": utc_now(),
            }
        )
        BuilderReActCheckpointStore.save(
            session,
            state,
            summary=summary,
            source_action=source_action,
        )
        return state

    @staticmethod
    def mark_completed(session: Session, *, session_id: Any, checkpoint_id: str = "") -> None:
        record = (
            session.exec(
                select(ShortTermCheckpointRecord).where(
                    ShortTermCheckpointRecord.session_id == session_id,
                    ShortTermCheckpointRecord.checkpoint_key == checkpoint_id,
                )
            ).first()
            if checkpoint_id
            else session.exec(
                select(ShortTermCheckpointRecord)
                .where(
                    ShortTermCheckpointRecord.session_id == session_id,
                    ShortTermCheckpointRecord.checkpoint_key.startswith("react:"),
                    ShortTermCheckpointRecord.is_active == True,  # noqa: E712
                )
                .order_by(desc(ShortTermCheckpointRecord.created_at))
            ).first()
        )
        if record is None:
            return
        record.status = "completed"
        record.is_active = False
        record.updated_at = utc_now()
        session.add(record)
        session.flush()
