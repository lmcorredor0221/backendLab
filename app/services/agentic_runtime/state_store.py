from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import desc
from sqlmodel import Session, select

from app.models import ShortTermCheckpointRecord, ShortTermSessionStateRecord, utc_now
from app.services.agentic_runtime.contracts import BuilderAgentState


def _state_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class BuilderReActCheckpointStore:
    """Persists ReAct checkpoints through the existing short-term memory tables."""

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
                branch_key="main",
                checkpoint_key=checkpoint_key,
                parent_checkpoint_key=previous.checkpoint_key if previous else "",
                checkpoint_number=(previous.checkpoint_number + 1) if previous else 1,
            )
        checkpoint.stage = state.stage
        checkpoint.source_action = source_action
        checkpoint.status = state.status
        checkpoint.summary = summary
        checkpoint.state_hash = digest
        checkpoint.state_payload = payload
        checkpoint.is_consistent = True
        checkpoint.is_active = True
        checkpoint.updated_at = utc_now()
        session.add(checkpoint)

        session_state = session.exec(
            select(ShortTermSessionStateRecord).where(
                ShortTermSessionStateRecord.session_id == state.session_id
            )
        ).first()
        if session_state is None:
            session_state = ShortTermSessionStateRecord(session_id=state.session_id)
        session_state.active_branch_key = "main"
        session_state.active_checkpoint_key = checkpoint_key
        session_state.last_consistent_checkpoint_key = checkpoint_key
        session_state.source_action = source_action
        session_state.state_hash = digest
        session_state.state_payload = payload
        session_state.updated_at = utc_now()
        session.add(session_state)
        session.flush()
        return checkpoint_key

    @staticmethod
    def load(
        session: Session,
        *,
        session_id: Any,
        checkpoint_id: str = "",
    ) -> BuilderAgentState | None:
        if checkpoint_id:
            record = session.exec(
                select(ShortTermCheckpointRecord).where(
                    ShortTermCheckpointRecord.session_id == session_id,
                    ShortTermCheckpointRecord.checkpoint_key == checkpoint_id,
                )
            ).first()
        else:
            record = session.exec(
                select(ShortTermCheckpointRecord)
                .where(
                    ShortTermCheckpointRecord.session_id == session_id,
                    ShortTermCheckpointRecord.is_active == True,  # noqa: E712
                    ShortTermCheckpointRecord.checkpoint_key.startswith("react:"),
                )
                .order_by(desc(ShortTermCheckpointRecord.created_at))
            ).first()
        return BuilderAgentState.model_validate(record.state_payload) if record else None

    @staticmethod
    def mark_resume_requested(
        session: Session,
        *,
        session_id: Any,
        action: str,
        scope: str,
    ) -> BuilderAgentState | None:
        state = BuilderReActCheckpointStore.load(session, session_id=session_id)
        if state is None:
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
            summary="Decision humana resuelta; checkpoint listo para reanudar.",
            source_action="attention_resume_requested",
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
        session_state = session.exec(
            select(ShortTermSessionStateRecord).where(
                ShortTermSessionStateRecord.session_id == session_id
            )
        ).first()
        if session_state is not None and session_state.active_checkpoint_key == record.checkpoint_key:
            session_state.active_checkpoint_key = ""
            session_state.source_action = "react_completed"
            session_state.updated_at = utc_now()
            session.add(session_state)
        session.flush()
