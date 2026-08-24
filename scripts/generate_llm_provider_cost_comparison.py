from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlmodel import Session, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.routes.sessions import build_snapshot, load_latest_persisted_acp_preview
from app.db import engine
from app.models import LLMProviderKey, SessionRecord
from app.services.estimation_service import (
    _build_agentic_estimate,
    _build_construction_scenarios,
    _build_signals,
    _build_traditional_estimate,
    _load_automation_profiles,
    _load_pricing_profiles,
    _load_role_rates,
    _load_workstream_profiles,
)
from app.services.llm_runtime.runtime_settings_service import (
    load_effective_runtime_settings,
    load_platform_runtime_defaults,
)


PROVIDER_ORDER: list[LLMProviderKey] = [
    LLMProviderKey.openai,
    LLMProviderKey.deepseek,
    LLMProviderKey.codex_local,
]

COMMERCIAL_SCENARIO_MAP: list[tuple[str, str, str]] = [
    ("lean_process", "Lean", "traditional_blueprint"),
    ("blueprint_free", "Blueprint free", "blueprint_basic"),
    ("blueprint_pro", "Blueprint pro", "blueprint_premium"),
    ("acp_premium", "ACP premium", "acp_agentic"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un comparativo local de costos del flujo SaaS por proveedor LLM."
    )
    parser.add_argument("--session-id", default="", help="UUID de sesion a evaluar.")
    parser.add_argument(
        "--title-contains",
        default="",
        help="Filtro opcional por titulo para elegir la sesion candidata mas reciente.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BACKEND_ROOT.parent / "outputs" / "llm-provider-cost-comparison"),
        help="Directorio donde se escriben el JSON y el Markdown.",
    )
    return parser.parse_args()


def _pick_session(
    session: Session,
    *,
    session_id: str,
    title_contains: str,
) -> tuple[SessionRecord, Any]:
    if session_id.strip():
        record = session.get(SessionRecord, UUID(session_id.strip()))
        if record is None:
            raise RuntimeError(f"No existe la sesion solicitada: {session_id}")
        if record.workspace_id is None:
            raise RuntimeError("La sesion solicitada no tiene workspace asociado.")
        snapshot = build_snapshot(session, record)
        return record, snapshot

    candidates = session.exec(select(SessionRecord).order_by(SessionRecord.updated_at.desc())).all()
    lowered_filter = title_contains.strip().lower()
    fallback_record: SessionRecord | None = None
    fallback_snapshot: Any | None = None

    for record in candidates:
        if record.workspace_id is None:
            continue
        if lowered_filter and lowered_filter not in (record.title or "").lower():
            continue
        snapshot = build_snapshot(session, record)
        if fallback_record is None:
            fallback_record = record
            fallback_snapshot = snapshot
        if snapshot.canvas is not None and snapshot.blueprint is not None:
            return record, snapshot
        if snapshot.canvas is not None:
            return record, snapshot

    if fallback_record is not None and fallback_snapshot is not None:
        return fallback_record, fallback_snapshot
    raise RuntimeError("No se encontro una sesion local elegible para generar el comparativo.")


def _provider_label(provider: str) -> str:
    return {
        "openai": "OpenAI",
        "deepseek": "DeepSeek",
        "codex_local": "Codex local",
        "antigravity_cli": "Antigravity CLI",
    }.get(provider, provider)


def _runtime_settings_for_provider(base_settings: Any, provider: LLMProviderKey) -> Any:
    runtime_settings = base_settings.model_copy(deep=True)
    runtime_settings.active_provider = provider
    return runtime_settings


def _build_provider_rows(
    session: Session,
    *,
    snapshot: Any,
    acp_preview: Any,
    runtime_settings: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    role_rates = _load_role_rates(session)
    workstream_profiles = _load_workstream_profiles(session)
    automation_profiles = _load_automation_profiles(session)
    pricing_profiles = _load_pricing_profiles(session)

    baseline_signals = _build_signals(
        snapshot=snapshot,
        acp_preview=acp_preview,
        runtime_settings=runtime_settings,
    )
    traditional = _build_traditional_estimate(
        baseline_signals,
        role_rates,
        workstream_profiles,
    )

    rows: list[dict[str, Any]] = []
    for provider in PROVIDER_ORDER:
        provider_settings = _runtime_settings_for_provider(runtime_settings, provider)
        provider_signals = _build_signals(
            snapshot=snapshot,
            acp_preview=acp_preview,
            runtime_settings=provider_settings,
        )
        agentic = _build_agentic_estimate(
            provider_signals,
            traditional,
            runtime_settings=provider_settings,
            role_rates=role_rates,
            workstream_profiles=workstream_profiles,
            automation_profiles=automation_profiles,
            pricing_profiles=pricing_profiles,
        )
        construction_scenarios = _build_construction_scenarios(
            traditional=traditional,
            agentic=agentic,
            signals=provider_signals,
        )
        rows.append(
            {
                "provider_key": provider.value,
                "provider_label": _provider_label(provider.value),
                "status": "estimated",
                "estimation_source": "native_estimator",
                "maturity_stage": provider_signals.maturity_stage.value,
                "complexity": provider_signals.complexity.value,
                "scope_points": provider_signals.scope_points,
                "report": agentic.model_dump(mode="json"),
                "construction_scenarios": [item.model_dump(mode="json") for item in construction_scenarios],
            }
        )

    rows.append(
        _build_antigravity_row(
            snapshot=snapshot,
            acp_preview=acp_preview,
            runtime_settings=runtime_settings,
            traditional=traditional,
            role_rates=role_rates,
            workstream_profiles=workstream_profiles,
            automation_profiles=automation_profiles,
            pricing_profiles=pricing_profiles,
        )
    )
    return traditional.model_dump(mode="json"), rows


def _build_antigravity_row(
    *,
    snapshot: Any,
    acp_preview: Any,
    runtime_settings: Any,
    traditional: Any,
    role_rates: dict[str, Any],
    workstream_profiles: dict[str, Any],
    automation_profiles: dict[str, Any],
    pricing_profiles: dict[Any, Any],
) -> dict[str, Any]:
    simulated = runtime_settings.model_copy(deep=True)
    simulated.active_provider = LLMProviderKey.codex_local
    simulated.codex_local.command = runtime_settings.antigravity.executable or "agy"
    simulated.codex_local.model = (
        runtime_settings.antigravity.model
        or (runtime_settings.antigravity.fallback_models[0] if runtime_settings.antigravity.fallback_models else "")
    )
    simulated.codex_local.profile = runtime_settings.antigravity.runner_id or "local-antigravity-cli"

    signals = _build_signals(
        snapshot=snapshot,
        acp_preview=acp_preview,
        runtime_settings=simulated,
    )
    agentic = _build_agentic_estimate(
        signals,
        traditional,
        runtime_settings=simulated,
        role_rates=role_rates,
        workstream_profiles=workstream_profiles,
        automation_profiles=automation_profiles,
        pricing_profiles=pricing_profiles,
    )
    payload = agentic.model_dump(mode="json")
    payload["active_provider"] = LLMProviderKey.antigravity_cli.value
    payload["provider_model"] = (
        f"executable={runtime_settings.antigravity.executable or 'agy'} | "
        f"model={(runtime_settings.antigravity.model or 'not_configured')} | "
        f"effort={runtime_settings.antigravity.effort or 'high'}"
    )
    payload["pricing_policy"] = "antigravity_cli_extrapolated_from_codex_local_hybrid"
    payload["warnings"] = [
        "No existe un pricing profile activo dedicado para antigravity_cli; este renglón se extrapola desde la curva local de codex_local.",
        *payload.get("warnings", []),
    ]
    payload["pricing_assumptions"] = [
        "Extrapolacion heuristica: antigravity_cli reutiliza la economia local de codex_local hasta que exista un perfil propio.",
        *payload.get("pricing_assumptions", []),
    ]
    pricing_snapshot = payload.get("pricing_snapshot")
    if isinstance(pricing_snapshot, dict):
        pricing_snapshot["provider"] = LLMProviderKey.antigravity_cli.value
        pricing_snapshot["profile_key"] = "antigravity_cli_extrapolated_from_codex_local_hybrid"
        pricing_snapshot["label"] = "Antigravity CLI extrapolated from codex_local_hybrid"
        pricing_snapshot["model"] = payload["provider_model"]
    construction_scenarios = _build_construction_scenarios(
        traditional=traditional,
        agentic=agentic,
        signals=signals,
    )
    return {
        "provider_key": LLMProviderKey.antigravity_cli.value,
        "provider_label": _provider_label(LLMProviderKey.antigravity_cli.value),
        "status": "estimated_extrapolated",
        "estimation_source": "codex_local_local_runtime_curve",
        "maturity_stage": signals.maturity_stage.value,
        "complexity": signals.complexity.value,
        "scope_points": signals.scope_points,
        "report": payload,
        "construction_scenarios": [item.model_dump(mode="json") for item in construction_scenarios],
    }


def _load_runtime_settings(session: Session, *, workspace_id: UUID | None) -> Any:
    if workspace_id is not None:
        return load_effective_runtime_settings(session, workspace_id)
    return load_platform_runtime_defaults(session)


def _load_observed_usage(session: Session) -> dict[str, Any]:
    rows = session.exec(
        text(
            """
            select
                provider_key,
                count(*) as call_count,
                coalesce(sum(total_tokens), 0) as total_tokens,
                coalesce(sum(cost_total), 0) as total_cost,
                min(started_at) as first_seen,
                max(started_at) as last_seen
            from llm_usage_ledger
            group by provider_key
            order by call_count desc, provider_key
            """
        )
    ).all()
    ledger = [
        {
            "provider_key": row[0],
            "provider_label": _provider_label(row[0]),
            "call_count": int(row[1] or 0),
            "total_tokens": int(row[2] or 0),
            "total_cost": float(row[3] or 0),
            "first_seen": row[4].isoformat() if row[4] is not None else None,
            "last_seen": row[5].isoformat() if row[5] is not None else None,
        }
        for row in rows
    ]
    return {
        "ledger": ledger,
        "codex_runtime_audit_runs": _count_jsonl_lines(BACKEND_ROOT / "runtime" / "codex-workspaces" / "runtime-audit.jsonl"),
        "antigravity_runtime_audit_runs": _count_jsonl_lines(BACKEND_ROOT / "runtime" / "agy-workspaces" / "runtime-audit-agy.jsonl"),
    }


def _build_commercial_matrix(provider_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for item in provider_rows:
        scenarios_by_key = {
            scenario["scenario_key"]: scenario
            for scenario in item.get("construction_scenarios", [])
        }
        row = {
            "provider_key": item["provider_key"],
            "provider_label": item["provider_label"],
            "status": item["status"],
        }
        for matrix_key, _label, scenario_key in COMMERCIAL_SCENARIO_MAP:
            scenario = scenarios_by_key.get(scenario_key, {})
            row[matrix_key] = {
                "scenario_key": scenario_key,
                "estimated_cost": float(scenario.get("estimated_cost") or 0),
                "estimated_hours_total": float(scenario.get("estimated_hours_total") or 0),
                "estimated_duration_weeks": float(scenario.get("estimated_duration_weeks") or 0),
                "cost_savings_vs_traditional": float(scenario.get("cost_savings_vs_traditional") or 0),
            }
        acp_manual = scenarios_by_key.get("acp_manual", {})
        row["acp_premium_manual_reference"] = {
            "scenario_key": "acp_manual",
            "estimated_cost": float(acp_manual.get("estimated_cost") or 0),
            "estimated_hours_total": float(acp_manual.get("estimated_hours_total") or 0),
            "estimated_duration_weeks": float(acp_manual.get("estimated_duration_weeks") or 0),
            "cost_savings_vs_traditional": float(acp_manual.get("cost_savings_vs_traditional") or 0),
        }
        matrix.append(row)
    return matrix


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _format_money(value: float) -> str:
    return f"{value:,.2f}"


def _table_cell(value: Any) -> str:
    return str(value if value is not None else "n/a").replace("|", "<br>")


def _build_markdown_report(payload: dict[str, Any]) -> str:
    traditional = payload["traditional"]
    session_meta = payload["session"]
    rows = payload["providers"]
    commercial_matrix = payload["commercial_matrix"]
    observed = payload["observed_usage"]
    lines = [
        "# LLM provider cost comparison",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Session: {session_meta['title']} ({session_meta['id']})",
        f"- Workspace: {session_meta['workspace_id']}",
        f"- Current local active provider: {payload['runtime']['active_provider']}",
        f"- ACP preview available: {str(payload['inputs']['has_acp_preview']).lower()}",
        f"- Project actuals count: {payload['inputs']['project_actuals_count']}",
        "",
        "## Traditional baseline",
        "",
        f"- Estimated hours: {_format_money(float(traditional['estimated_hours_total']))}",
        f"- Estimated weeks: {_format_money(float(traditional['estimated_duration_weeks']))}",
        f"- Estimated cost COP: {_format_money(float(traditional['estimated_cost']))}",
        "",
        "## Provider comparison",
        "",
        "| Provider | Status | Model | Total COP | Provider COP | Runtime USD | Hours | Weeks | Savings vs traditional COP | Pricing |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        report = row["report"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["provider_label"],
                    row["status"],
                    _table_cell(report.get("provider_model") or "n/a"),
                    _format_money(float(report.get("estimated_cost") or 0)),
                    _format_money(float(report.get("provider_runtime_cost_total_cop") or 0)),
                    _format_money(float(report.get("provider_runtime_cost_total_usd") or 0)),
                    _format_money(float(report.get("estimated_hours_total") or 0)),
                    _format_money(float(report.get("estimated_duration_weeks") or 0)),
                    _format_money(float(report.get("net_savings_vs_traditional") or 0)),
                    _table_cell(report.get("pricing_policy") or "n/a"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Offer comparison",
            "",
            "- Mapping used: `Lean` = `traditional_blueprint`, `Blueprint free` = `blueprint_basic`, `Blueprint pro` = `blueprint_premium`, `ACP premium` = `acp_agentic`.",
            "- Optional reference: `ACP + equipo humano` remains available in the JSON as `acp_manual`.",
            "",
            "| Provider | Status | Lean COP | Blueprint free COP | Blueprint pro COP | ACP premium COP |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in commercial_matrix:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["provider_label"],
                    item["status"],
                    _format_money(float(item["lean_process"]["estimated_cost"])),
                    _format_money(float(item["blueprint_free"]["estimated_cost"])),
                    _format_money(float(item["blueprint_pro"]["estimated_cost"])),
                    _format_money(float(item["acp_premium"]["estimated_cost"])),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Observed local telemetry",
            "",
            f"- Codex runtime audits: {observed['codex_runtime_audit_runs']}",
            f"- Antigravity runtime audits: {observed['antigravity_runtime_audit_runs']}",
        ]
    )
    if observed["ledger"]:
        lines.extend(
            [
                "",
                "| Ledger provider | Calls | Tokens | Cost | First seen | Last seen |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for item in observed["ledger"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        item["provider_label"],
                        str(item["call_count"]),
                        f"{item['total_tokens']:,}",
                        _format_money(float(item["total_cost"])),
                        str(item["first_seen"] or "n/a"),
                        str(item["last_seen"] or "n/a"),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- OpenAI, DeepSeek y Codex local salen del motor de estimacion actual del backend.",
            "- Antigravity CLI se reporta como extrapolacion local mientras no exista pricing profile seed dedicado.",
            "- Los costos historicos observados y los costos estimados no representan la misma cosa: el ledger mide ejecuciones reales, mientras este reporte estima el flujo SaaS completo.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.echo = False

    with Session(engine) as session:
        record, snapshot = _pick_session(
            session,
            session_id=args.session_id,
            title_contains=args.title_contains,
        )
        acp_preview = load_latest_persisted_acp_preview(session, record.id)
        runtime_settings = _load_runtime_settings(session, workspace_id=record.workspace_id)
        traditional, provider_rows = _build_provider_rows(
            session,
            snapshot=snapshot,
            acp_preview=acp_preview,
            runtime_settings=runtime_settings,
        )
        observed_usage = _load_observed_usage(session)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"llm-provider-cost-comparison-{record.id}-{timestamp}"
    json_path = output_dir / f"{base_name}.json"
    md_path = output_dir / f"{base_name}.md"

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "session": {
            "id": str(record.id),
            "title": record.title,
            "workspace_id": str(record.workspace_id) if record.workspace_id is not None else None,
            "updated_at": record.updated_at.isoformat() if record.updated_at is not None else None,
        },
        "runtime": {
            "active_provider": runtime_settings.active_provider.value,
            "openai_fast_model": runtime_settings.openai.fast_model,
            "openai_reasoning_model": runtime_settings.openai.reasoning_model,
            "deepseek_fast_model": runtime_settings.deepseek.fast_model,
            "deepseek_reasoning_model": runtime_settings.deepseek.reasoning_model,
            "codex_command": runtime_settings.codex_local.command,
            "codex_model": runtime_settings.codex_local.model,
            "antigravity_executable": runtime_settings.antigravity.executable,
            "antigravity_model": runtime_settings.antigravity.model,
        },
        "inputs": {
            "has_discovery": snapshot.discovery is not None,
            "has_canvas": snapshot.canvas is not None,
            "has_blueprint": snapshot.blueprint is not None,
            "has_evaluation": snapshot.evaluation is not None,
            "has_acp_preview": acp_preview is not None,
            "project_actuals_count": len(snapshot.project_actuals),
        },
        "traditional": traditional,
        "providers": provider_rows,
        "commercial_matrix": _build_commercial_matrix(provider_rows),
        "observed_usage": observed_usage,
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown_report(payload), encoding="utf-8")

    print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
