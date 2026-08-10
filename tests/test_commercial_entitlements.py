from uuid import uuid4

from app.models import CommercialTier, WorkspaceRole
from app.services.commercial_access import (
    build_entitlement_context,
    build_entitlement_matrix,
    build_session_commercial_access,
    resolve_capability_access,
    validate_capability,
)


def test_blueprint_free_tier_only_allows_in_app_value_and_samples() -> None:
    context = build_entitlement_context(
        tier=CommercialTier.blueprint,
        workspace_id=uuid4(),
        user_id=uuid4(),
        role=WorkspaceRole.viewer,
    )

    decisions = {item.capability: item for item in build_entitlement_matrix(context)}

    assert decisions["blueprint.view"].allowed is True
    assert decisions["diagram.view.sample"].allowed is True
    assert decisions["acp.invite"].allowed is True
    assert decisions["blueprint.download"].allowed is False
    assert decisions["diagram.view.blueprint"].allowed is False
    assert decisions["acp.build"].allowed is False
    assert decisions["acp.download"].allowed is False


def test_blueprint_pro_unlocks_downloads_and_blueprint_diagrams_for_editor() -> None:
    context = build_entitlement_context(
        tier=CommercialTier.blueprint_pro,
        workspace_id=uuid4(),
        user_id=uuid4(),
        role=WorkspaceRole.editor,
        purchase_refs=("purchase:blueprint-pro:001",),
    )

    assert resolve_capability_access(context, "blueprint.download").allowed is True
    assert resolve_capability_access(context, "diagram.view.blueprint").allowed is True
    assert resolve_capability_access(context, "acp.build").allowed is False

    access = build_session_commercial_access(CommercialTier.blueprint_pro, role=WorkspaceRole.editor)
    assert access.can_download_blueprint is True
    assert access.can_view_diagram_blueprint is True
    assert access.can_build_acp is False


def test_acp_viewer_can_see_premium_diagrams_but_cannot_build_or_download() -> None:
    context = build_entitlement_context(
        tier=CommercialTier.acp,
        workspace_id=uuid4(),
        user_id=uuid4(),
        role=WorkspaceRole.viewer,
        purchase_refs=("purchase:acp:001",),
    )

    assert resolve_capability_access(context, "diagram.view.acp").allowed is True
    assert resolve_capability_access(context, "acp.build").allowed is False
    assert resolve_capability_access(context, "acp.download").allowed is False

    violation = validate_capability(CommercialTier.acp, "acp.build", context=context)
    assert violation is not None
    assert violation.reason == "role"
    assert "rol actual" in violation.message


def test_entitlement_matrix_preserves_workspace_scope() -> None:
    workspace_a = uuid4()
    workspace_b = uuid4()
    context_a = build_entitlement_context(
        tier=CommercialTier.acp,
        workspace_id=workspace_a,
        user_id=uuid4(),
        role=WorkspaceRole.owner,
        purchase_refs=("purchase:workspace-a:acp",),
    )
    context_b = build_entitlement_context(
        tier=CommercialTier.blueprint,
        workspace_id=workspace_b,
        user_id=uuid4(),
        role=WorkspaceRole.owner,
    )

    acp_decision_a = resolve_capability_access(context_a, "acp.build")
    acp_decision_b = resolve_capability_access(context_b, "acp.build")

    assert acp_decision_a.allowed is True
    assert acp_decision_a.workspace_id == workspace_a
    assert acp_decision_a.purchase_refs == ("purchase:workspace-a:acp",)
    assert acp_decision_b.allowed is False
    assert acp_decision_b.workspace_id == workspace_b
    assert acp_decision_b.purchase_refs == ()

