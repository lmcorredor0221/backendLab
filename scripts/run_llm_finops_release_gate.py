from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from sqlmodel import SQLModel, Session, create_engine, select

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.routes import llm_finops
from app.models import (
    CodexLocalCostPolicy,
    LLMBudgetPolicyRecord,
    LLMFinOpsAlertRecord,
    LLMPricingProfile,
    LLMPricingRateEntry,
    LLMProviderKey,
    LLMUsageLedgerRecord,
    LLMValueAnnotationRecord,
)
from app.services.llm_finops import (
    LLMCallContext,
    LLMUsageCostBreakdown,
    LLMUsageRecordInput,
    NormalizedLLMUsage,
)
from app.services.llm_finops.analytics_service import LLMUsageAnalyticsFilters, LLMUsageAnalyticsService
from app.services.llm_finops.ledger_service import LLMUsageLedgerService
from app.services.llm_finops.pricing_resolver import PricingResolver
from app.services.llm_finops.usage_normalization import normalize_cli_usage, normalize_openai_usage


EXPECTED_MIGRATIONS = {
    "20260813_0008_llm_finops_ledger.py": "20260813_0008",
    "20260813_0010_llm_finops_budgets.py": "20260813_0010",
    "20260813_0011_llm_finops_alerts.py": "20260813_0011",
    "20260813_0012_llm_value_annotations.py": "20260813_0012",
}


@dataclass
class GateCheck:
    name: str
    ok: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def run_llm_finops_release_gate(*, force_fail: str = "") -> dict[str, Any]:
    checks = [
        _run_check("migrations", _check_migrations, force_fail=force_fail),
        _run_check("normalization", _check_normalization, force_fail=force_fail),
        _run_check("pricing", _check_pricing, force_fail=force_fail),
        _run_check("ledger", _check_ledger, force_fail=force_fail),
        _run_check("api_summary", _check_api_summary, force_fail=force_fail),
        _run_check("prompt_response_storage", _check_prompt_response_storage, force_fail=force_fail),
    ]
    ok = all(check.ok for check in checks)
    return {
        "ok": ok,
        "generated_at": datetime.now(UTC).isoformat(),
        "checks": [check.to_dict() for check in checks],
    }


def _run_check(name: str, check: Callable[[], GateCheck], *, force_fail: str) -> GateCheck:
    if force_fail == name:
        return GateCheck(name=name, ok=False, detail=f"Forced failure for release gate check: {name}")
    try:
        return check()
    except Exception as exc:
        return GateCheck(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}")


def _check_migrations() -> GateCheck:
    versions_dir = BACKEND_ROOT / "alembic" / "versions"
    loaded_revisions: dict[str, Any] = {}
    for file_name, expected_revision in EXPECTED_MIGRATIONS.items():
        path = versions_dir / file_name
        if not path.exists():
            raise RuntimeError(f"Missing migration file {file_name}")
        module = _load_migration_module(path)
        if getattr(module, "revision", "") != expected_revision:
            raise RuntimeError(f"Migration {file_name} exposes unexpected revision")
        loaded_revisions[file_name] = {
            "revision": getattr(module, "revision", ""),
            "down_revision": getattr(module, "down_revision", None),
        }
    if loaded_revisions["20260813_0012_llm_value_annotations.py"]["down_revision"] != "20260813_0011":
        raise RuntimeError("Value annotations migration is not chained after alerts")

    table_names = set(SQLModel.metadata.tables)
    expected_tables = {
        LLMUsageLedgerRecord.__tablename__,
        LLMBudgetPolicyRecord.__tablename__,
        LLMFinOpsAlertRecord.__tablename__,
        LLMValueAnnotationRecord.__tablename__,
    }
    missing_tables = sorted(expected_tables - table_names)
    if missing_tables:
        raise RuntimeError(f"Missing SQLModel tables: {', '.join(missing_tables)}")
    return GateCheck(
        name="migrations",
        ok=True,
        detail="FinOps migrations and SQLModel tables are present.",
        evidence={"migrations": loaded_revisions, "tables": sorted(expected_tables)},
    )


def _check_normalization() -> GateCheck:
    openai_usage = normalize_openai_usage(
        {
            "input_tokens": 120,
            "output_tokens": 40,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 8},
        }
    )
    cli_usage = normalize_cli_usage(
        {
            "run_id": "gate-cli",
            "status": "succeeded",
            "selected_model": "gpt-test",
            "metrics": {"prompt_estimated_tokens": 50, "output_estimated_tokens": 20, "duration_ms": 1000},
        }
    )
    if openai_usage.total_tokens != 160 or openai_usage.cached_input_tokens != 20:
        raise RuntimeError("OpenAI usage normalization returned unexpected token totals")
    if cli_usage.total_tokens != 70 or cli_usage.usage_is_estimated is not True:
        raise RuntimeError("CLI usage normalization did not mark estimated usage")
    return GateCheck(
        name="normalization",
        ok=True,
        detail="Provider and CLI usage normalization are stable.",
        evidence={
            "openai_total_tokens": openai_usage.total_tokens,
            "cli_total_tokens": cli_usage.total_tokens,
            "cli_estimated": cli_usage.usage_is_estimated,
        },
    )


def _check_pricing() -> GateCheck:
    resolver = PricingResolver()
    profiles = _pricing_profiles()
    token_cost = resolver.resolve_call_cost(
        provider_key=LLMProviderKey.openai,
        model_name="gpt-test",
        usage=NormalizedLLMUsage(input_tokens=1_000_000, output_tokens=1_000_000, cached_input_tokens=100_000),
        pricing_profiles=profiles,
        occurred_at=datetime(2026, 8, 13, tzinfo=UTC),
    )
    local_cost = resolver.resolve_call_cost(
        provider_key=LLMProviderKey.codex_local,
        model_name="gpt-test",
        usage=normalize_cli_usage({"run_id": "gate-local", "metrics": {"duration_ms": 3_600_000}}),
        pricing_profiles=profiles,
        occurred_at=datetime(2026, 8, 13, tzinfo=UTC),
        local_cost_policy=CodexLocalCostPolicy.hybrid,
    )
    if token_cost.cost_total <= 0 or local_cost.cost_total <= 0:
        raise RuntimeError("Pricing resolver returned zero cost for priced usage")
    return GateCheck(
        name="pricing",
        ok=True,
        detail="Token and local-runtime pricing produce reproducible snapshots.",
        evidence={
            "token_cost_total": token_cost.cost_total,
            "token_profile": token_cost.pricing_profile_key,
            "local_cost_total": local_cost.cost_total,
            "local_profile": local_cost.pricing_profile_key,
        },
    )


def _check_ledger() -> GateCheck:
    db = _build_session()
    payload = _usage_payload(request_id="gate-ledger")
    service = LLMUsageLedgerService()
    first = service.record_call(db, payload)
    duplicate = service.record_call(db, payload)
    records = db.exec(select(LLMUsageLedgerRecord)).all()
    if not first.created or not duplicate.duplicate or len(records) != 1:
        raise RuntimeError("Ledger idempotency failed")
    record = records[0]
    if record.cost_total != 0.25 or record.total_tokens != 150:
        raise RuntimeError("Ledger persisted unexpected cost or token totals")
    return GateCheck(
        name="ledger",
        ok=True,
        detail="Ledger persists costed calls and rejects duplicate request attempts.",
        evidence={"usage_record_id": str(record.id), "cost_total": record.cost_total, "total_tokens": record.total_tokens},
    )


def _check_api_summary() -> GateCheck:
    paths = {getattr(route, "path", "") for route in llm_finops.router.routes}
    if "/summary" not in paths and "/finops/llm/summary" not in paths:
        raise RuntimeError("FinOps summary route is not registered")

    db = _build_session()
    workspace_id = uuid4()
    LLMUsageLedgerService().record_call(db, _usage_payload(workspace_id=workspace_id, request_id="gate-summary", cost_total=0.4))
    summary = LLMUsageAnalyticsService().summarize(db, LLMUsageAnalyticsFilters(workspace_id=workspace_id))
    if summary["call_count"] != 1 or summary["cost_total"] != 0.4:
        raise RuntimeError("FinOps summary aggregation returned unexpected totals")
    return GateCheck(
        name="api_summary",
        ok=True,
        detail="Summary endpoint is registered and summary aggregation returns scoped totals.",
        evidence={"route_present": True, "summary": summary},
    )


def _check_prompt_response_storage() -> GateCheck:
    columns = {column.name for column in LLMUsageLedgerRecord.__table__.columns}
    forbidden_columns = {"prompt", "prompt_text", "response", "response_text", "completion_text", "messages"}
    leaked_columns = sorted(columns & forbidden_columns)
    if leaked_columns:
        raise RuntimeError(f"Ledger schema stores forbidden prompt/response columns: {', '.join(leaked_columns)}")

    db = _build_session()
    result = LLMUsageLedgerService().record_call(
        db,
        _usage_payload(request_id="gate-no-content").model_copy(
            update={
                "prompt_hash": "sha256:prompt",
                "response_hash": "sha256:response",
                "metadata": {"trace_note": "hashes only"},
            }
        ),
    )
    record = db.get(LLMUsageLedgerRecord, result.usage_record_id)
    if record is None:
        raise RuntimeError("Ledger did not persist no-content usage record")
    serialized = json.dumps(
        {
            "metadata": record.metadata_payload,
            "usage_raw": record.usage_raw_redacted,
            "prompt_hash": record.prompt_hash,
            "response_hash": record.response_hash,
        },
        ensure_ascii=True,
    )
    if "prompt completo" in serialized or "respuesta completa" in serialized:
        raise RuntimeError("Prompt or response content leaked into ledger payload")
    return GateCheck(
        name="prompt_response_storage",
        ok=True,
        detail="Ledger schema stores hashes and redacted metrics, not prompt/response text fields.",
        evidence={"forbidden_columns_checked": sorted(forbidden_columns), "stored_hashes": True},
    )


def _build_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _usage_payload(
    *,
    workspace_id=None,
    request_id: str,
    cost_total: float = 0.25,
) -> LLMUsageRecordInput:
    return LLMUsageRecordInput(
        context=LLMCallContext(
            workspace_id=workspace_id or uuid4(),
            stage="define",
            agent_key="release_gate",
            capability_key="finops_release_gate",
            action_key="verify",
        ),
        provider_key="openai",
        model_name="gpt-test",
        requested_model="gpt-test",
        execution_backend="provider_native",
        request_id=request_id,
        usage=NormalizedLLMUsage(input_tokens=100, output_tokens=50),
        cost=LLMUsageCostBreakdown(cost_total=cost_total, currency="USD"),
    )


def _pricing_profiles() -> dict[LLMProviderKey, list[LLMPricingProfile]]:
    return {
        LLMProviderKey.openai: [
            LLMPricingProfile(
                profile_key="gate-openai",
                label="Gate OpenAI",
                provider=LLMProviderKey.openai,
                model="gpt-test",
                mode="standard",
                effective_from="2026-01-01",
                cop_per_usd=4000,
                rates=[
                    LLMPricingRateEntry(metric_key="reasoning_input_tokens_m", amount_usd=1.0),
                    LLMPricingRateEntry(metric_key="reasoning_cached_input_tokens_m", amount_usd=0.1),
                    LLMPricingRateEntry(metric_key="reasoning_output_tokens_m", amount_usd=2.0),
                ],
            )
        ],
        LLMProviderKey.codex_local: [
            LLMPricingProfile(
                profile_key="gate-codex-local",
                label="Gate Codex local",
                provider=LLMProviderKey.codex_local,
                model="gpt-test",
                mode="local",
                is_local_inference=True,
                local_cost_policy=CodexLocalCostPolicy.hybrid,
                effective_from="2026-01-01",
                cop_per_usd=4000,
                rates=[
                    LLMPricingRateEntry(metric_key="compute_hour_core", amount_usd=0.5),
                    LLMPricingRateEntry(metric_key="local_session", amount_usd=0.1),
                    LLMPricingRateEntry(metric_key="workstation_hour_hybrid", amount_usd=1.0),
                ],
            )
        ],
    }


def _load_migration_module(path: Path) -> Any:
    module_name = f"finops_migration_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load migration module {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Release gate integral para FinOps IA LLM.")
    parser.add_argument(
        "--force-fail",
        choices=["", "migrations", "normalization", "pricing", "ledger", "api_summary", "prompt_response_storage"],
        default="",
        help="Solo para pruebas: fuerza el fallo de un check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_llm_finops_release_gate(force_fail=args.force_fail)
    print(json.dumps(summary, ensure_ascii=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
