from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlmodel import Session, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db import engine
from app.models import LLMProviderKey, LLMUsageLedgerRecord, SessionRecord
from app.services.estimation_service import _load_pricing_profiles
from app.services.llm_finops.contracts import LLMUsageCostBreakdown, NormalizedLLMUsage
from app.services.llm_finops.pricing_resolver import PricingResolver
from app.services.llm_finops.usage_normalization import normalize_cli_usage
from app.services.llm_runtime.runtime_settings_service import (
    load_effective_runtime_settings,
    load_platform_runtime_defaults,
)


PROVIDER_ORDER: list[LLMProviderKey] = [
    LLMProviderKey.openai,
    LLMProviderKey.deepseek,
    LLMProviderKey.codex_local,
    LLMProviderKey.antigravity_cli,
]

ZERO_COST_DIAGRAM_PROVIDERS = {"", "reference_suite_manual_sync"}
AUDIO_PATHS = {
    LLMProviderKey.antigravity_cli: BACKEND_ROOT / "runtime" / "agy-workspaces" / "runtime-audit-agy.jsonl",
    LLMProviderKey.codex_local: BACKEND_ROOT / "runtime" / "codex-workspaces" / "runtime-audit.jsonl",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un estimado del costo operativo para producir la informacion del flujo actual."
    )
    parser.add_argument("--session-id", default="", help="UUID de sesion a evaluar.")
    parser.add_argument(
        "--title-contains",
        default="",
        help="Filtro opcional por titulo para elegir la sesion candidata mas reciente.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BACKEND_ROOT.parent / "outputs" / "current-flow-runtime-cost"),
        help="Directorio donde se escriben el JSON y el Markdown.",
    )
    return parser.parse_args()


def _pick_session(
    session: Session,
    *,
    session_id: str,
    title_contains: str,
) -> SessionRecord:
    if session_id.strip():
        record = session.get(SessionRecord, UUID(session_id.strip()))
        if record is None:
            raise RuntimeError(f"No existe la sesion solicitada: {session_id}")
        return record

    candidates = session.exec(select(SessionRecord).order_by(SessionRecord.updated_at.desc())).all()
    lowered_filter = title_contains.strip().lower()
    for record in candidates:
        if lowered_filter and lowered_filter not in (record.title or "").lower():
            continue
        if record.workspace_id is not None:
            return record
    raise RuntimeError("No se encontro una sesion local elegible para estimar el flujo actual.")


def _provider_label(provider: str) -> str:
    return {
        "openai": "OpenAI",
        "deepseek": "DeepSeek",
        "codex_local": "Codex local",
        "antigravity_cli": "Antigravity CLI",
    }.get(provider, provider)


def _read_text(path_value: str | None) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _average_usage(usages: list[NormalizedLLMUsage]) -> NormalizedLLMUsage:
    if not usages:
        return NormalizedLLMUsage(usage_is_estimated=True)
    return NormalizedLLMUsage(
        input_tokens=round(mean(item.input_tokens for item in usages)),
        output_tokens=round(mean(item.output_tokens for item in usages)),
        total_tokens=round(mean(item.total_tokens for item in usages)),
        cached_input_tokens=round(mean(item.cached_input_tokens for item in usages)),
        reasoning_tokens=round(mean(item.reasoning_tokens for item in usages)),
        tool_call_count=round(mean(item.tool_call_count for item in usages)),
        provider_metrics={
            "duration_ms": round(
                mean(float(item.provider_metrics.get("duration_ms", 0) or 0) for item in usages)
            ),
            "queue_wait_ms": round(
                mean(float(item.provider_metrics.get("queue_wait_ms", 0) or 0) for item in usages)
            ),
            "output_size_bytes": round(
                mean(float(item.provider_metrics.get("output_size_bytes", 0) or 0) for item in usages)
            ),
        },
        usage_is_estimated=True,
    )


def _load_pricing_profiles_with_antigravity(session: Session) -> dict[LLMProviderKey, list[Any]]:
    pricing_profiles = _load_pricing_profiles(session)
    if LLMProviderKey.antigravity_cli not in pricing_profiles:
        clones = []
        for profile in pricing_profiles.get(LLMProviderKey.codex_local, []):
            clone = profile.model_copy(deep=True)
            clone.provider = LLMProviderKey.antigravity_cli
            clone.profile_key = f"antigravity::{profile.profile_key}"
            clone.label = f"Antigravity extrapolated from {profile.label}"
            clones.append(clone)
        pricing_profiles[LLMProviderKey.antigravity_cli] = clones
    return pricing_profiles


def _runtime_settings_for(record: SessionRecord, session: Session) -> Any:
    if record.workspace_id is not None:
        return load_effective_runtime_settings(session, record.workspace_id)
    return load_platform_runtime_defaults(session)


def _model_for_provider(runtime_settings: Any, provider: LLMProviderKey) -> str:
    if provider == LLMProviderKey.openai:
        return runtime_settings.openai.reasoning_model
    if provider == LLMProviderKey.deepseek:
        return runtime_settings.deepseek.reasoning_model
    if provider == LLMProviderKey.codex_local:
        return runtime_settings.codex_local.model
    if provider == LLMProviderKey.antigravity_cli:
        return (
            runtime_settings.antigravity.model
            or (runtime_settings.antigravity.fallback_models[0] if runtime_settings.antigravity.fallback_models else "")
            or runtime_settings.codex_local.model
        )
    return ""


def _cop_per_usd(
    pricing_profiles: dict[LLMProviderKey, list[Any]],
    provider: LLMProviderKey,
) -> float:
    profiles = pricing_profiles.get(provider, [])
    if not profiles:
        return 0.0
    return float(profiles[0].cop_per_usd or 0.0)


def _cost_to_payload(cost: LLMUsageCostBreakdown) -> dict[str, Any]:
    return {
        "cost_input_usd": float(cost.cost_input or 0),
        "cost_output_usd": float(cost.cost_output or 0),
        "cost_other_usd": float(cost.cost_other or 0),
        "cost_total_usd": float(cost.cost_total or 0),
        "currency": cost.currency,
        "fx_rate": float(cost.fx_rate or 0),
        "pricing_profile_key": cost.pricing_profile_key,
        "pricing_snapshot": cost.pricing_snapshot,
        "warnings": list(cost.warnings),
    }


def _resolve_cost(
    resolver: PricingResolver,
    *,
    provider: LLMProviderKey,
    model_name: str,
    usage: NormalizedLLMUsage,
    pricing_profiles: dict[LLMProviderKey, list[Any]],
) -> LLMUsageCostBreakdown:
    return resolver.resolve_call_cost(
        provider_key=provider,
        model_name=model_name,
        usage=usage,
        pricing_profiles=pricing_profiles,
    )


def _build_ledger_workloads(session: Session, *, session_id: UUID) -> tuple[list[dict[str, Any]], list[str]]:
    rows = session.exec(
        select(LLMUsageLedgerRecord)
        .where(LLMUsageLedgerRecord.session_id == session_id)
        .order_by(LLMUsageLedgerRecord.started_at.asc())
    ).all()

    workloads: list[dict[str, Any]] = []
    zero_usage_capabilities: list[str] = []
    for row in rows:
        usage = NormalizedLLMUsage(
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            total_tokens=row.total_tokens,
            cached_input_tokens=row.cached_input_tokens,
            reasoning_tokens=row.reasoning_tokens,
            tool_call_count=int((row.other_token_metrics or {}).get("tool_call_count") or 0),
            provider_metrics={},
            raw_usage={},
            usage_is_estimated=bool((row.other_token_metrics or {}).get("usage_is_estimated")),
        )
        if usage.total_tokens <= 0:
            zero_usage_capabilities.append(row.capability_key or row.action_key or "unknown")
            continue
        workloads.append(
            {
                "kind": "ledger_capability",
                "label": row.capability_key or row.action_key or "unknown",
                "source_provider": row.provider_key,
                "source_model": row.model_name,
                "count": 1,
                "usage": usage,
                "usage_mode": "actual",
            }
        )
    return workloads, zero_usage_capabilities


def _build_tool_recommendation_workload(*, session_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    workloads: list[dict[str, Any]] = []
    warnings: list[str] = []
    for provider, audit_path in AUDIO_PATHS.items():
        for payload in _iter_jsonl(audit_path):
            raw_line = json.dumps(payload, ensure_ascii=False)
            if session_id not in raw_line:
                continue
            if payload.get("task_kind") != "tool_recommendation_minimal":
                continue
            if payload.get("status") != "succeeded":
                continue
            usage = normalize_cli_usage(
                payload,
                prompt_text=_read_text(str(payload.get("prompt_path") or "")),
                output_text=_read_text(
                    str(
                        payload.get("output_path")
                        or payload.get("structured_output_path")
                        or payload.get("last_message_path")
                        or ""
                    )
                ),
            )
            workloads.append(
                {
                    "kind": "runtime_audit",
                    "label": "tool_recommendation_minimal",
                    "source_provider": provider.value,
                    "source_model": str(payload.get("selected_model") or ""),
                    "count": 1,
                    "usage": usage,
                    "usage_mode": "estimated_from_runtime_audit",
                }
            )
    if not workloads:
        warnings.append("No se encontro una corrida exitosa de tool recommendation vinculada a la sesion.")
    return workloads, warnings


def _build_diagram_workloads(session: Session, *, session_id: UUID) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    rows = session.execute(
        text(
            """
            select provider_key, count(*) as diagram_count
            from diagram_versions_v3
            where session_id = :sid
            group by provider_key
            order by provider_key
            """
        ),
        {"sid": session_id},
    ).fetchall()

    counts_by_provider = {str(provider or ""): int(count or 0) for provider, count in rows}
    provider_generated_count = sum(
        count for provider, count in counts_by_provider.items() if provider not in ZERO_COST_DIAGRAM_PROVIDERS
    )
    manual_sync_count = counts_by_provider.get("reference_suite_manual_sync", 0)

    successful_diagram_usages: dict[LLMProviderKey, list[NormalizedLLMUsage]] = defaultdict(list)
    for provider, audit_path in AUDIO_PATHS.items():
        for payload in _iter_jsonl(audit_path):
            if payload.get("task_kind") != "diagram_model_generation":
                continue
            if payload.get("status") != "succeeded":
                continue
            usage = normalize_cli_usage(
                payload,
                prompt_text=_read_text(str(payload.get("prompt_path") or "")),
                output_text=_read_text(
                    str(
                        payload.get("output_path")
                        or payload.get("structured_output_path")
                        or payload.get("last_message_path")
                        or ""
                    )
                ),
            )
            successful_diagram_usages[provider].append(usage)

    warnings: list[str] = []
    api_proxy_template = _average_usage(successful_diagram_usages.get(LLMProviderKey.antigravity_cli, []))
    if api_proxy_template.total_tokens <= 0:
        api_proxy_template = _average_usage(
            [item for values in successful_diagram_usages.values() for item in values]
        )
    if api_proxy_template.total_tokens <= 0:
        warnings.append("No se pudo construir una plantilla de costo para diagramas desde runtime audits.")

    workloads: list[dict[str, Any]] = []
    for provider_key, count in counts_by_provider.items():
        if provider_key in ZERO_COST_DIAGRAM_PROVIDERS or count <= 0:
            continue
        workloads.append(
            {
                "kind": "diagram_suite",
                "label": f"provider_generated_diagrams:{provider_key}",
                "source_provider": provider_key,
                "source_model": "",
                "count": count,
                "usage": None,
                "usage_mode": "template_from_successful_runtime_audits",
            }
        )

    template_summary = {
        "counts_by_provider": counts_by_provider,
        "provider_generated_count": provider_generated_count,
        "manual_sync_count": manual_sync_count,
        "diagram_success_samples": {
            provider.value: len(successful_diagram_usages.get(provider, []))
            for provider in AUDIO_PATHS
        },
        "api_proxy_template": {
            "input_tokens": api_proxy_template.input_tokens,
            "output_tokens": api_proxy_template.output_tokens,
            "duration_ms": float(api_proxy_template.provider_metrics.get("duration_ms", 0) or 0),
        },
        "local_runtime_templates": {
            provider.value: {
                "input_tokens": _average_usage(usages).input_tokens,
                "output_tokens": _average_usage(usages).output_tokens,
                "duration_ms": float(_average_usage(usages).provider_metrics.get("duration_ms", 0) or 0),
            }
            for provider, usages in successful_diagram_usages.items()
        },
    }
    return workloads, template_summary, warnings


def _diagram_usage_for_provider(
    *,
    provider: LLMProviderKey,
    template_summary: dict[str, Any],
) -> NormalizedLLMUsage:
    if provider in {LLMProviderKey.codex_local, LLMProviderKey.antigravity_cli}:
        local_template = template_summary["local_runtime_templates"].get(provider.value)
        if isinstance(local_template, dict) and float(local_template.get("duration_ms") or 0) > 0:
            return NormalizedLLMUsage(
                input_tokens=int(local_template.get("input_tokens") or 0),
                output_tokens=int(local_template.get("output_tokens") or 0),
                total_tokens=int(local_template.get("input_tokens") or 0) + int(local_template.get("output_tokens") or 0),
                provider_metrics={"duration_ms": float(local_template.get("duration_ms") or 0)},
                usage_is_estimated=True,
            )

    api_proxy = template_summary["api_proxy_template"]
    return NormalizedLLMUsage(
        input_tokens=int(api_proxy.get("input_tokens") or 0),
        output_tokens=int(api_proxy.get("output_tokens") or 0),
        total_tokens=int(api_proxy.get("input_tokens") or 0) + int(api_proxy.get("output_tokens") or 0),
        provider_metrics={"duration_ms": float(api_proxy.get("duration_ms") or 0)},
        usage_is_estimated=True,
    )


def _build_runtime_cost_matrix(
    *,
    pricing_profiles: dict[LLMProviderKey, list[Any]],
    runtime_settings: Any,
    ledger_workloads: list[dict[str, Any]],
    tool_workloads: list[dict[str, Any]],
    diagram_workloads: list[dict[str, Any]],
    diagram_templates: dict[str, Any],
) -> list[dict[str, Any]]:
    resolver = PricingResolver()
    matrix: list[dict[str, Any]] = []

    for provider in PROVIDER_ORDER:
        model_name = _model_for_provider(runtime_settings, provider)
        total_usd = 0.0
        components: list[dict[str, Any]] = []

        for item in ledger_workloads:
            usage = item["usage"]
            cost = _resolve_cost(
                resolver,
                provider=provider,
                model_name=model_name,
                usage=usage,
                pricing_profiles=pricing_profiles,
            )
            total_usd += cost.cost_total
            components.append(
                {
                    "label": item["label"],
                    "count": item["count"],
                    "usage_mode": item["usage_mode"],
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "reasoning_tokens": usage.reasoning_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    "cost": _cost_to_payload(cost),
                }
            )

        for item in tool_workloads:
            usage = item["usage"]
            cost = _resolve_cost(
                resolver,
                provider=provider,
                model_name=model_name,
                usage=usage,
                pricing_profiles=pricing_profiles,
            )
            total_usd += cost.cost_total
            components.append(
                {
                    "label": item["label"],
                    "count": item["count"],
                    "usage_mode": item["usage_mode"],
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "duration_ms": float(usage.provider_metrics.get("duration_ms", 0) or 0),
                    },
                    "cost": _cost_to_payload(cost),
                }
            )

        for item in diagram_workloads:
            count = int(item["count"] or 0)
            unit_usage = _diagram_usage_for_provider(provider=provider, template_summary=diagram_templates)
            unit_cost = _resolve_cost(
                resolver,
                provider=provider,
                model_name=model_name,
                usage=unit_usage,
                pricing_profiles=pricing_profiles,
            )
            total_usd += unit_cost.cost_total * count
            components.append(
                {
                    "label": item["label"],
                    "count": count,
                    "usage_mode": item["usage_mode"],
                    "usage": {
                        "input_tokens": unit_usage.input_tokens,
                        "output_tokens": unit_usage.output_tokens,
                        "total_tokens": unit_usage.total_tokens,
                        "duration_ms": float(unit_usage.provider_metrics.get("duration_ms", 0) or 0),
                    },
                    "cost_per_unit": _cost_to_payload(unit_cost),
                    "cost_total_usd": round(unit_cost.cost_total * count, 8),
                }
            )

        fx_rate = _cop_per_usd(pricing_profiles, provider)
        matrix.append(
            {
                "provider_key": provider.value,
                "provider_label": _provider_label(provider.value),
                "model_name": model_name,
                "estimated_total_usd": round(total_usd, 8),
                "estimated_total_cop": round(total_usd * fx_rate, 2),
                "fx_rate": fx_rate,
                "pricing_mode": "token_and_local_runtime_repricing",
                "components": components,
            }
        )

    return matrix


def _format_money(value: float) -> str:
    return f"{value:,.2f}"


def _build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Current flow runtime cost",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Session: {payload['session']['title']} ({payload['session']['id']})",
        f"- Workspace: {payload['session']['workspace_id']}",
        f"- Current stage: {payload['session']['current_stage']}",
        f"- Commercial tier: {payload['session']['commercial_tier']}",
        "",
        "## Scope measured",
        "",
        f"- Provider-generated diagrams available: {payload['scope']['provider_generated_diagrams']}",
        f"- Reference/manual-sync diagrams: {payload['scope']['manual_sync_diagrams']}",
        f"- Deliverables generated with zero runtime LLM cost in current flow: {payload['scope']['zero_cost_deliverables']}",
        f"- Ledger workloads with measurable usage: {payload['scope']['metered_ledger_workloads']}",
        f"- Tool recommendation runs linked to session: {payload['scope']['tool_recommendation_runs']}",
        "",
        "## Provider totals",
        "",
        "| Provider | Model | Estimated USD | Estimated COP |",
        "| --- | --- | ---: | ---: |",
    ]

    for item in payload["providers"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["provider_label"],
                    str(item["model_name"] or "n/a").replace("|", "<br>"),
                    _format_money(float(item["estimated_total_usd"])),
                    _format_money(float(item["estimated_total_cop"])),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Method notes",
            "",
            "- Los deliverables gobernados del flujo actual quedaron en costo variable cero porque se generaron de forma deterministica o fallback sin LLM (`allow_llm=false` o proveedor no invocado).",
            "- Los diagramas se tarifaron con plantillas derivadas de corridas exitosas en `runtime-audit*.jsonl`.",
            "- Para OpenAI y DeepSeek se uso la plantilla de diagramas mas cercana al modo API (exitos de Antigravity).",
            "- Para Codex local y Antigravity CLI se uso su propia curva local de duracion exitosa, reprecificada con el perfil local vigente.",
            "- Los capabilities con telemetry de tokens en cero se mantuvieron en cero y quedan como posible subestimacion.",
        ]
    )

    warnings = payload.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        engine.echo = False
        record = _pick_session(
            session,
            session_id=args.session_id,
            title_contains=args.title_contains,
        )
        runtime_settings = _runtime_settings_for(record, session)
        pricing_profiles = _load_pricing_profiles_with_antigravity(session)

        ledger_workloads, zero_usage_capabilities = _build_ledger_workloads(session, session_id=record.id)
        tool_workloads, tool_warnings = _build_tool_recommendation_workload(session_id=str(record.id))
        diagram_workloads, diagram_templates, diagram_warnings = _build_diagram_workloads(
            session,
            session_id=record.id,
        )

        zero_cost_deliverables = session.execute(
            text(
                """
                select count(*)
                from (
                    select distinct on (deliverable_key) deliverable_key
                    from deliverable_generation_jobs_v1
                    where session_id = :sid
                      and status = 'available'
                      and coalesce(provider_key, '') = ''
                      and coalesce(tokens_input, 0) = 0
                      and coalesce(tokens_output, 0) = 0
                      and coalesce(estimated_cost_usd, 0) = 0
                    order by deliverable_key, updated_at desc
                ) t
                """
            ),
            {"sid": record.id},
        ).scalar_one()

        provider_matrix = _build_runtime_cost_matrix(
            pricing_profiles=pricing_profiles,
            runtime_settings=runtime_settings,
            ledger_workloads=ledger_workloads,
            tool_workloads=tool_workloads,
            diagram_workloads=diagram_workloads,
            diagram_templates=diagram_templates,
        )

    warnings = [
        *tool_warnings,
        *diagram_warnings,
    ]
    if zero_usage_capabilities:
        warnings.append(
            "Existen capabilities con telemetry de uso en cero y se mantuvieron sin costo: "
            + ", ".join(sorted(set(zero_usage_capabilities)))
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "session": {
            "id": str(record.id),
            "title": record.title,
            "workspace_id": str(record.workspace_id) if record.workspace_id is not None else "",
            "current_stage": getattr(record.current_stage, "value", str(record.current_stage or "")),
            "commercial_tier": getattr(record.commercial_tier, "value", str(record.commercial_tier or "")),
            "updated_at": record.updated_at.isoformat(),
        },
        "scope": {
            "provider_generated_diagrams": int(diagram_templates["provider_generated_count"]),
            "manual_sync_diagrams": int(diagram_templates["manual_sync_count"]),
            "zero_cost_deliverables": int(zero_cost_deliverables or 0),
            "metered_ledger_workloads": len(ledger_workloads),
            "tool_recommendation_runs": len(tool_workloads),
            "zero_usage_capabilities": zero_usage_capabilities,
        },
        "diagram_templates": diagram_templates,
        "providers": provider_matrix,
        "warnings": warnings,
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = f"current-flow-runtime-cost-{record.id}-{timestamp}"
    json_path = output_dir / f"{slug}.json"
    md_path = output_dir / f"{slug}.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown_report(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "json_path": str(json_path),
                "markdown_path": str(md_path),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
