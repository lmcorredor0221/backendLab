from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import ValidationError

from app.contracts import CANONICAL_CONTRACT_MODELS, collect_validation_issues
from app.models import SessionSnapshot
from app.services.blueprint_consistency_service import ensure_blueprint_consistency_report
from app.services.canonical_exports import (
    build_agent_construction_package_v2,
    build_blueprint_core,
    build_construction_pack,
    build_estimation_pack,
    build_test_pack,
)

CanonicalExportKind = Literal[
    "blueprint-core.v1",
    "construction-pack.v1",
    "agent-construction-package.v2",
    "prompt-pack.v1",
    "estimation-pack.v1",
    "test-pack.v1",
]

CANONICAL_EXPORT_ARTIFACTS: dict[CanonicalExportKind, tuple[str, str]] = {
    "blueprint-core.v1": ("blueprint_core_export", "Blueprint core export"),
    "construction-pack.v1": ("construction_pack_export", "Construction pack export"),
    "agent-construction-package.v2": ("agent_construction_package_v2_export", "Agent Construction Package v2 export"),
    "prompt-pack.v1": ("prompt_pack_export", "Prompt pack export"),
    "estimation-pack.v1": ("estimation_pack_export", "Estimation pack export"),
    "test-pack.v1": ("test_pack_export", "Test pack export"),
}


@dataclass(frozen=True)
class CanonicalExportDocument:
    blocking_reasons: list[str]
    contract_key: CanonicalExportKind
    generated_at: str
    payload: dict[str, Any]
    payload_text: str
    readiness: str
    source_blueprint_version: int | None
    validation_issues: list[dict[str, str]]

    @property
    def checksum_sha256(self) -> str:
        return hashlib.sha256(self.payload_text.encode("utf-8")).hexdigest()


def serialize_contract_payload(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _requires_evaluation_assets(snapshot: SessionSnapshot) -> bool:
    if snapshot.evaluation_dataset is not None and snapshot.evaluation_dataset.cases:
        return True
    if snapshot.evaluation_rubric is not None and snapshot.evaluation_rubric.dimensions:
        return True
    if snapshot.evaluation is not None and snapshot.evaluation.cases:
        return True
    return False


def _missing_dependencies(snapshot: SessionSnapshot, contract_key: CanonicalExportKind) -> list[str]:
    missing: list[str] = []
    if snapshot.discovery is None:
        missing.append("discovery")
    if snapshot.canvas is None:
        missing.append("canvas")
    if snapshot.blueprint is None:
        missing.append("blueprint")

    if contract_key in {
        "construction-pack.v1",
        "agent-construction-package.v2",
        "prompt-pack.v1",
        "test-pack.v1",
    } and not _requires_evaluation_assets(snapshot):
        missing.append("evaluation")
    if contract_key in {"estimation-pack.v1", "test-pack.v1"} and snapshot.estimation_report is None:
        missing.append("estimation_report")
    return missing


def _test_pack_blocking_reasons(bundle: dict[str, Any]) -> list[str]:
    test_pack = bundle.get("test-pack.v1")
    if test_pack is None:
        return ["test_pack:missing_bundle"]

    reasons: list[str] = []
    if not test_pack.mutation_cases:
        reasons.append("test_pack:missing_mutation_cases")
    if not test_pack.prompt_evaluation_cases:
        reasons.append("test_pack:missing_prompt_evaluation_cases")
    if not test_pack.recovery_cases:
        reasons.append("test_pack:missing_recovery_cases")
    if not test_pack.acceptance_journeys:
        reasons.append("test_pack:missing_acceptance_journeys")
    if not test_pack.stable_issue_catalog:
        reasons.append("test_pack:missing_stable_issue_catalog")
    if not test_pack.external_consumer.relative_path or not test_pack.external_consumer.entry_command:
        reasons.append("test_pack:external_consumer_incomplete")
    return reasons


def _resolve_readiness(
    snapshot: SessionSnapshot,
    contract_key: CanonicalExportKind,
    bundle: dict[str, Any],
) -> str:
    if _missing_dependencies(snapshot, contract_key):
        return "blocked"
    consistency_report = ensure_blueprint_consistency_report(snapshot)
    if consistency_report.overall_status == "blocked":
        return "blocked"

    construction_pack = bundle["construction-pack.v1"]
    if contract_key == "test-pack.v1":
        test_pack_reasons = _test_pack_blocking_reasons(bundle)
        return "ready" if construction_pack.readiness.can_build and not test_pack_reasons else "blocked"

    if contract_key in {"construction-pack.v1", "agent-construction-package.v2", "prompt-pack.v1"}:
        return "ready" if construction_pack.readiness.can_build else "blocked"

    if contract_key == "blueprint-core.v1":
        return "ready" if construction_pack.readiness.can_build else "needs_review"

    if contract_key in {"estimation-pack.v1", "test-pack.v1"} and snapshot.estimation_report is not None:
        return "needs_review" if snapshot.estimation_report.is_stale else "ready"

    return "ready"


def _blocking_reasons(
    snapshot: SessionSnapshot,
    contract_key: CanonicalExportKind,
    bundle: dict[str, Any],
) -> list[str]:
    reasons = [f"missing:{item}" for item in _missing_dependencies(snapshot, contract_key)]
    consistency_report = ensure_blueprint_consistency_report(snapshot)
    reasons.extend(f"consistency:{item}" for item in consistency_report.blocking_issues)
    construction_pack = bundle["construction-pack.v1"]
    if contract_key in {"construction-pack.v1", "agent-construction-package.v2", "prompt-pack.v1", "test-pack.v1"}:
        reasons.extend(f"readiness:{item}" for item in construction_pack.readiness.blocking_issues)
    if contract_key == "test-pack.v1":
        reasons.extend(_test_pack_blocking_reasons(bundle))
    if contract_key in {"estimation-pack.v1", "test-pack.v1"} and snapshot.estimation_report is not None:
        reasons.extend(f"review:{item}" for item in snapshot.estimation_report.stale_reasons)
    return reasons


def should_block_canonical_export(document: CanonicalExportDocument, *, preview: bool) -> bool:
    if preview:
        return False
    return document.readiness == "blocked"


def build_canonical_export_headers(document: CanonicalExportDocument, *, preview: bool) -> dict[str, str]:
    return {
        "X-Canonical-Checksum-SHA256": document.checksum_sha256,
        "X-Canonical-Contract-Version": document.contract_key,
        "X-Canonical-Export-Preview": "true" if preview else "false",
        "X-Canonical-Export-Readiness": document.readiness,
        "X-Canonical-Generated-At": document.generated_at,
        "X-Canonical-Source-Blueprint-Version": (
            str(document.source_blueprint_version) if document.source_blueprint_version is not None else ""
        ),
    }


def build_canonical_export_document(
    snapshot: SessionSnapshot,
    contract_key: CanonicalExportKind,
    *,
    generated_at: datetime | None = None,
) -> CanonicalExportDocument:
    construction_pack = build_construction_pack(snapshot, generated_at=generated_at)
    bundle: dict[str, Any] = {
        "construction-pack.v1": construction_pack,
        "agent-construction-package.v2": build_agent_construction_package_v2(
            snapshot,
            construction_pack=construction_pack,
            generated_at=generated_at,
        ),
        "prompt-pack.v1": construction_pack.prompt_pack,
        "evaluation-pack.v1": construction_pack.evaluation_pack,
    }
    if contract_key == "blueprint-core.v1":
        bundle["blueprint-core.v1"] = build_blueprint_core(snapshot, generated_at=generated_at)
    elif contract_key in {"estimation-pack.v1", "test-pack.v1"}:
        bundle["estimation-pack.v1"] = build_estimation_pack(snapshot, generated_at=generated_at)
        if contract_key == "test-pack.v1":
            bundle["test-pack.v1"] = build_test_pack(
                snapshot,
                construction_pack=construction_pack,
                estimation_pack=bundle["estimation-pack.v1"],
                generated_at=generated_at,
            )

    if contract_key == "blueprint-core.v1":
        contract = bundle["blueprint-core.v1"]
    elif contract_key == "construction-pack.v1":
        contract = construction_pack
    elif contract_key == "agent-construction-package.v2":
        contract = bundle["agent-construction-package.v2"]
    elif contract_key == "prompt-pack.v1":
        contract = construction_pack.prompt_pack
    elif contract_key == "test-pack.v1":
        contract = bundle["test-pack.v1"]
    else:
        contract = bundle["estimation-pack.v1"]
    payload = contract.model_dump(mode="json")
    model_cls = CANONICAL_CONTRACT_MODELS[contract_key]

    validation_issues: list[dict[str, str]] = []
    try:
        model_cls.model_validate(payload)
    except ValidationError as exc:
        validation_issues = [
            issue.model_dump(mode="json")
            for issue in collect_validation_issues(exc)
        ]

    return CanonicalExportDocument(
        blocking_reasons=_blocking_reasons(snapshot, contract_key, bundle),
        contract_key=contract_key,
        generated_at=str(payload["generated_at"]),
        payload=payload,
        payload_text=serialize_contract_payload(payload),
        readiness=_resolve_readiness(snapshot, contract_key, bundle),
        source_blueprint_version=payload.get("source_blueprint_version"),
        validation_issues=validation_issues,
    )
