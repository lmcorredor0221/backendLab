from __future__ import annotations

from uuid import uuid4

from app.models import CommercialTier
from app.services.acp_generator import generate_acp_preview
from app.services.commercial_access import build_entitlement_context
from app.services.diagram_catalog_service import build_diagram_catalog, build_diagram_content
from tests.test_acp_generator import build_ready_snapshot


def test_diagram_catalog_lists_all_entries_and_marks_free_sample() -> None:
    snapshot = build_ready_snapshot()
    preview = generate_acp_preview(snapshot)
    workspace_id = uuid4()
    context = build_entitlement_context(tier=CommercialTier.blueprint, workspace_id=workspace_id, user_id=uuid4())

    catalog = build_diagram_catalog(snapshot=snapshot, preview=preview, context=context, workspace_id=workspace_id)

    assert catalog.total_count == 24
    assert catalog.current_stage == "package"
    assert catalog.sample_count == 1
    assert catalog.locked_count > 0
    architecture = next(item for item in catalog.entries if item.diagram_key == "architecture_overview")
    assert architecture.access_state == "sample"
    assert "svg" in architecture.available_content_formats
    assert architecture.source_paths == []
    assert architecture.upsell is not None
    c4_context = next(item for item in catalog.entries if item.diagram_key == "c4_context")
    assert c4_context.access_state == "locked_blueprint"
    acp_only = next(item for item in catalog.entries if item.diagram_key == "tool_contract_sequence")
    assert acp_only.access_state == "locked_acp"
    rag_pipeline = next(item for item in catalog.entries if item.diagram_key == "rag_ingestion_pipeline")
    assert rag_pipeline.generation_state == "not_generated"


def test_diagram_content_never_returns_locked_payload() -> None:
    snapshot = build_ready_snapshot()
    preview = generate_acp_preview(snapshot)
    workspace_id = uuid4()
    context = build_entitlement_context(tier=CommercialTier.blueprint, workspace_id=workspace_id, user_id=uuid4())

    sample = build_diagram_content(
        diagram_key="architecture_overview",
        snapshot=snapshot,
        preview=preview,
        context=context,
        workspace_id=workspace_id,
        requested_format="svg",
    )
    locked = build_diagram_content(
        diagram_key="c4_context",
        snapshot=snapshot,
        preview=preview,
        context=context,
        workspace_id=workspace_id,
        requested_format="mermaid",
    )

    assert sample is not None
    assert sample.access_state == "sample"
    assert sample.content is not None
    assert sample.content.startswith("<svg")
    assert locked is not None
    assert locked.access_state == "locked_blueprint"
    assert locked.content is None
    assert locked.upsell is not None
    assert locked.upsell.target_tier == CommercialTier.blueprint_pro


def test_diagram_content_unlocks_by_blueprint_and_acp_tier() -> None:
    snapshot = build_ready_snapshot()
    preview = generate_acp_preview(snapshot)
    workspace_id = uuid4()

    blueprint_pro_context = build_entitlement_context(
        tier=CommercialTier.blueprint_pro,
        workspace_id=workspace_id,
        user_id=uuid4(),
    )
    acp_context = build_entitlement_context(tier=CommercialTier.acp, workspace_id=workspace_id, user_id=uuid4())

    blueprint_content = build_diagram_content(
        diagram_key="c4_context",
        snapshot=snapshot,
        preview=preview,
        context=blueprint_pro_context,
        workspace_id=workspace_id,
        requested_format="mermaid",
    )
    acp_content = build_diagram_content(
        diagram_key="tool_contract_sequence",
        snapshot=snapshot,
        preview=preview,
        context=acp_context,
        workspace_id=workspace_id,
        requested_format="mermaid",
    )

    assert blueprint_content is not None
    assert blueprint_content.access_state == "unlocked"
    assert blueprint_content.content is not None
    assert "flowchart" in blueprint_content.content
    assert acp_content is not None
    assert acp_content.access_state == "unlocked"
    assert acp_content.content is not None
    assert "flowchart" in acp_content.content
