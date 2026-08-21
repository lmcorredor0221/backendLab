from __future__ import annotations

from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.models import CommercialTier, WorkspaceRole
from app.models import RuntimeFeatureFlagRecord
from app.services.deliverable_catalog import (
    DeliverableGovernanceUpdate,
    DeliverablePolicyContext,
    deliverable_governance_entry,
    get_registry_entry,
    resolve_deliverable_policy,
    upsert_deliverable_governance,
)
from app.services.deliverable_catalog.persistence import DeliverableGovernanceAuditRecord
from app.services.deliverable_catalog.persistence import DeliverableGovernanceRecord
from app.services.stage5_service import (
    FEATURE_FLAG_BLUEPRINT_TIER_POLICY,
    FEATURE_FLAG_DELIVERABLE_CATALOG,
    FEATURE_FLAG_DELIVERABLE_GOVERNANCE_ADMIN,
)
from app.services.workspace_bootstrap import apply_workspace_bootstrap


def _session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _entry():
    entry = get_registry_entry("discovery.problem_context_brief")
    assert entry is not None
    return entry


def test_policy_stage_locks_deliverable_before_enabled_stage() -> None:
    with _session() as db:
        entry = get_registry_entry("diagram.architecture_overview")
        assert entry is not None

        decision = resolve_deliverable_policy(
            db,
            entry,
            DeliverablePolicyContext(
                role=WorkspaceRole.editor,
                tier=CommercialTier.acp,
                current_stage="discover",
            ),
        )

    assert decision.visible is True
    assert decision.access_state == "stage_locked"
    assert decision.can_view is False
    assert decision.reason_code == "stage_not_reached"


def test_policy_normalizes_legacy_session_stages_for_catalog_access() -> None:
    with _session() as db:
        entry = get_registry_entry("definition.requirements_brief")
        assert entry is not None

        decision = resolve_deliverable_policy(
            db,
            entry,
            DeliverablePolicyContext(
                role=WorkspaceRole.editor,
                tier=CommercialTier.blueprint_pro,
                current_stage="build_canvas",
            ),
        )

    assert decision.access_state == "not_generated"
    assert decision.can_generate is True


def test_policy_exposes_preview_when_tier_is_below_required_product() -> None:
    with _session() as db:
        entry = _entry()
        upsert_deliverable_governance(
            db,
            entry,
            DeliverableGovernanceUpdate(required_tier_override=CommercialTier.blueprint_pro.value),
        )
        db.commit()

        decision = resolve_deliverable_policy(
            db,
            entry,
            DeliverablePolicyContext(
                role=WorkspaceRole.editor,
                tier=CommercialTier.blueprint,
                current_stage="discover",
                has_current_version=True,
                quality_state="passed",
            ),
        )

    assert decision.access_state == "preview"
    assert decision.can_view is True
    assert decision.can_download is False
    assert decision.required_tier == CommercialTier.blueprint_pro


def test_workspace_override_takes_precedence_over_platform_governance() -> None:
    workspace_id = uuid4()
    with _session() as db:
        entry = _entry()
        upsert_deliverable_governance(
            db,
            entry,
            DeliverableGovernanceUpdate(enabled=False, notes="platform pause"),
        )
        upsert_deliverable_governance(
            db,
            entry,
            DeliverableGovernanceUpdate(enabled=True, notes="workspace enable"),
            workspace_id=workspace_id,
        )
        db.commit()

        platform_decision = resolve_deliverable_policy(
            db,
            entry,
            DeliverablePolicyContext(
                role=WorkspaceRole.editor,
                tier=CommercialTier.acp,
                current_stage="discover",
            ),
        )
        workspace_decision = resolve_deliverable_policy(
            db,
            entry,
            DeliverablePolicyContext(
                workspace_id=workspace_id,
                role=WorkspaceRole.editor,
                tier=CommercialTier.acp,
                current_stage="discover",
            ),
        )

    assert platform_decision.access_state == "disabled"
    assert workspace_decision.access_state == "not_generated"


def test_policy_blocks_generation_when_prompt_is_paused() -> None:
    with _session() as db:
        entry = _entry()
        upsert_deliverable_governance(
            db,
            entry,
            DeliverableGovernanceUpdate(prompt_status="paused"),
        )
        db.commit()

        decision = resolve_deliverable_policy(
            db,
            entry,
            DeliverablePolicyContext(
                role=WorkspaceRole.editor,
                tier=CommercialTier.acp,
                current_stage="discover",
            ),
        )

    assert decision.access_state == "not_generated"
    assert decision.can_generate is False
    assert decision.effective_prompt_status == "paused"


def test_policy_resolves_role_permissions_and_audit_entries() -> None:
    with _session() as db:
        entry = _entry()
        upsert_deliverable_governance(
            db,
            entry,
            DeliverableGovernanceUpdate(notes="initial governance"),
            actor_user_id=uuid4(),
        )
        db.commit()

        viewer_decision = resolve_deliverable_policy(
            db,
            entry,
            DeliverablePolicyContext(
                role=WorkspaceRole.viewer,
                tier=CommercialTier.acp,
                current_stage="discover",
                has_current_version=True,
                quality_state="passed",
            ),
        )
        admin_decision = resolve_deliverable_policy(
            db,
            entry,
            DeliverablePolicyContext(
                role=WorkspaceRole.admin,
                tier=CommercialTier.acp,
                current_stage="discover",
                has_current_version=True,
                quality_state="passed",
            ),
        )
        audits = db.exec(select(DeliverableGovernanceAuditRecord)).all()

    assert viewer_decision.can_view is True
    assert viewer_decision.can_download is False
    assert viewer_decision.can_edit_prompt is False
    assert admin_decision.can_download is True
    assert admin_decision.can_edit_prompt is True
    assert admin_decision.can_regenerate is False
    assert len(audits) == 1
    assert audits[0].reason == "initial governance"


def test_workspace_bootstrap_seeds_bdg_flags_and_governance_defaults_idempotently() -> None:
    workspace_id = uuid4()
    with _session() as db:
        apply_workspace_bootstrap(db, workspace_id)
        flags = {
            item.flag_key: item.enabled
            for item in db.exec(
                select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.workspace_id == workspace_id)
            ).all()
        }
        governance_defaults = db.exec(select(DeliverableGovernanceRecord)).all()
        initial_count = len(governance_defaults)

        apply_workspace_bootstrap(db, workspace_id)
        after_count = len(db.exec(select(DeliverableGovernanceRecord)).all())

    assert flags[FEATURE_FLAG_BLUEPRINT_TIER_POLICY] is True
    assert flags[FEATURE_FLAG_DELIVERABLE_CATALOG] is True
    assert flags[FEATURE_FLAG_DELIVERABLE_GOVERNANCE_ADMIN] is True
    assert any(item.deliverable_key == "discovery.problem_context_brief" for item in governance_defaults)
    assert initial_count > 0
    assert after_count == initial_count


def test_deliverable_governance_entry_exposes_catalog_metadata_for_admin_tables() -> None:
    with _session() as db:
        entry = _entry()
        governance = deliverable_governance_entry(db, entry)

    assert governance.deliverable_key == entry.deliverable_key
    assert governance.description == entry.description
    assert governance.deliverable_type == entry.deliverable_type
    assert governance.stage == entry.stage
    assert governance.enabled_from_stage == entry.enabled_from_stage
    assert governance.product_scope == list(entry.product_scope)
    assert governance.access_level == entry.access_level
    assert governance.formats.preferred in governance.formats.available
    assert governance.generation_mode == entry.generation_mode
    assert governance.prompt_policy.prompt_template_key == entry.prompt_policy.prompt_template_key
    assert governance.quality_policy.validator_key == entry.quality_policy.validator_key
    assert governance.dependency_policy.depends_on == entry.dependency_policy.depends_on
    assert governance.exportable == entry.exportable
    assert governance.active == entry.active
