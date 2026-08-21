from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from sqlmodel import Session  # noqa: E402

from app.api.routes.productization import _context  # noqa: E402
from app.db import engine  # noqa: E402
from app.models import SessionRecord, UserRecord, utc_now  # noqa: E402
from app.services.attention_service import build_attention_metrics_v2, build_attention_response_v2  # noqa: E402

TECHNICAL_LEAK_MARKERS = (
    "policy=",
    "codex local no pudo",
    "needs_review_on_provider_or_schema_failure",
    "falta informacion: missing_acceptance:",
    "falta informacion: untraced_item:",
    "falta informacion: vague_nfr:",
    "falta informacion: blocking_question:",
    "falta informacion: duplicate_key:",
    "falta informacion: canvas_scope_conflict:",
)


def _contains_technical_leak(value: str) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in TECHNICAL_LEAK_MARKERS)


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "").strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _build_report(session_id: UUID) -> tuple[dict, str]:
    with Session(engine) as db:
        record = db.get(SessionRecord, session_id)
        if record is None:
            raise SystemExit(f"Session not found: {session_id}")
        current_user = db.get(UserRecord, record.user_id)
        if current_user is None:
            raise SystemExit(f"Session user not found: {record.user_id}")
        snapshot, _, readiness, access = _context(db, record, current_user)
        response = build_attention_response_v2(
            db,
            record=record,
            snapshot=snapshot,
            readiness=readiness,
            access=access,
            current_stage="",
            limit=100,
        )
        metrics = build_attention_metrics_v2(
            db,
            record=record,
            snapshot=snapshot,
            readiness=readiness,
            access=access,
            current_stage="",
        )
    items = [item.model_dump(mode="json") for item in response.items]
    technical_leaks = [
        item
        for item in items
        if _contains_technical_leak(" ".join([item.get("title", ""), item.get("reason", ""), item.get("impact", "")]))
    ]
    report = {
        "contract": "attention.audit.iah0",
        "generated_at": utc_now().isoformat(),
        "session_id": str(session_id),
        "workspace_id": str(response.workspace_id),
        "total_count": response.total_count,
        "actionable_count": response.actionable_count,
        "blocking_count": response.blocking_count,
        "warning_count": response.warning_count,
        "info_count": response.info_count,
        "counts_by_stage": response.counts_by_stage,
        "counts_by_type": response.counts_by_type,
        "counts_by_product": response.counts_by_product,
        "counts_by_source": _count_by(items, "source"),
        "technical_leak_count": len(technical_leaks),
        "technical_leaks": [
            {
                "key": item.get("key"),
                "stage": item.get("stage"),
                "type": item.get("type"),
                "source": item.get("source"),
                "title": item.get("title"),
                "reason": item.get("reason"),
            }
            for item in technical_leaks
        ],
        "metrics": metrics,
    }
    stage_lines = [f"- `{key}`: {value}" for key, value in response.counts_by_stage.items()] or ["- None"]
    type_lines = [f"- `{key}`: {value}" for key, value in response.counts_by_type.items()] or ["- None"]
    leak_lines = [
        f"- `{item.get('key')}` [{item.get('stage')}/{item.get('type')}]: {item.get('title')}"
        for item in technical_leaks
    ] or ["- None"]
    md = "\n".join(
        [
            "# IAH0 Attention Baseline",
            "",
            f"- Session: `{session_id}`",
            f"- Generated at: `{report['generated_at']}`",
            f"- Total items: `{response.total_count}`",
            f"- Actionable items: `{response.actionable_count}`",
            f"- Blocking items: `{response.blocking_count}`",
            f"- Technical leak count: `{len(technical_leaks)}`",
            "",
            "## Counts By Stage",
            "",
            *stage_lines,
            "",
            "## Counts By Type",
            "",
            *type_lines,
            "",
            "## Technical Leaks",
            "",
            *leak_lines,
            "",
        ]
    )
    return report, md


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Attention v2 without mutating project data.")
    parser.add_argument("--session-id", required=True, help="Project/session UUID to audit.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "Docs" / "system-analysis" / "evidence" / "attention-hitl"),
        help="Directory where JSON and Markdown evidence will be written.",
    )
    args = parser.parse_args()
    session_id = UUID(args.session_id)
    report, md = _build_report(session_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"iah0-attention-baseline-{session_id}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "technical_leak_count": report["technical_leak_count"]}))


if __name__ == "__main__":
    main()
