from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlmodel import Session, select

from app.db import engine
from app.models import RuntimeFeatureFlagRecord
from app.services.stage5_service import update_feature_flag
from app.services.workspace_bootstrap import FEATURE_FLAG_REACT_RUNTIME, seed_runtime_feature_flags


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or update one workspace-scoped Lean Agent Builder feature flag.",
    )
    parser.add_argument("--workspace-id", required=True, help="Workspace UUID to inspect or update.")
    parser.add_argument(
        "--flag-key",
        default=FEATURE_FLAG_REACT_RUNTIME,
        help=f"Feature flag key. Default: {FEATURE_FLAG_REACT_RUNTIME}",
    )
    parser.add_argument(
        "--set",
        choices=("enabled", "disabled"),
        help="Optional desired state. Omit to inspect only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    workspace_id = UUID(args.workspace_id)
    with Session(engine) as session:
        seed_runtime_feature_flags(session, workspace_id=workspace_id)
        if args.set is not None:
            record = update_feature_flag(
                session,
                workspace_id=workspace_id,
                flag_key=args.flag_key,
                enabled=args.set == "enabled",
            )
            session.commit()
        else:
            record = session.exec(
                select(RuntimeFeatureFlagRecord).where(
                    RuntimeFeatureFlagRecord.workspace_id == workspace_id,
                    RuntimeFeatureFlagRecord.flag_key == args.flag_key,
                )
            ).first()
        if record is None:
            print(f"{args.flag_key}: missing for workspace {workspace_id}")
            return
        print(
            f"{record.flag_key}: {'enabled' if record.enabled else 'disabled'} "
            f"workspace_id={record.workspace_id}"
        )


if __name__ == "__main__":
    main()
