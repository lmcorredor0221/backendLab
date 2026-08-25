from app.services.product_processing.contracts import (
    BlueprintTierPolicy,
    BlueprintUncertainty,
    AcpDirectRouteResolution,
    AcpStageReadinessEntry,
    PremiumEnrichmentItem,
    PremiumEnrichmentWorkspace,
    PremiumSelectiveReprocessResult,
    PremiumUncertaintyResolutionRequest,
    ProductBuildAction,
    ProductBuildActionState,
    ProductBuildAttentionItem,
    ProductBuildAttentionSeverity,
    ProductBuildAttentionSummary,
    ProductBuildCommandRequest,
    ProductBuildCurrentActivity,
    ProductBuildDeliverableState,
    ProductBuildDeliverableStatus,
    ProductBuildEntitlement,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductBuildProgress,
    ProductBuildRecoverableError,
    ProductBuildStageStatus,
    ProductBuildStatus,
    ProductBuildTelemetryEvent,
    ProductBuildTelemetryProductSummary,
    ProductBuildTelemetryReport,
    ProductBuildTelemetryTotals,
    ProductJourneyCurrentStage,
    ProductJourneyDeliverableSummary,
    ProductJourneyOutcome,
    ProductJourneyOverview,
    ProductJourneyProductSummary,
    ProductJourneyRecommendedAction,
    UncertaintyBacklogEntry,
    UncertaintyBacklogStatus,
    ProductProcessingMode,
    ProductProcessingProfile,
    QuestionPolicyMode,
    UncertaintyClassification,
    UncertaintyDisposition,
    UncertaintyKind,
    UncertaintyOption,
    calculate_product_build_percent,
)
from app.services.product_processing.policy import (
    BLUEPRINT_TIER_POLICY,
    classify_uncertainty_for_profile,
    get_product_processing_profile,
    resolve_product_processing_mode,
)
from app.services.product_processing.backlog_service import (
    backlog_entry_from_record,
    list_uncertainty_backlog,
    prioritize_uncertainty_backlog,
    resolve_uncertainty,
    supersede_unresolved_uncertainties,
    upsert_uncertainty_backlog,
)
from app.services.product_processing.product_build_run_service import (
    ensure_product_build_run,
    list_product_build_runs,
    list_product_build_steps,
    update_product_build_run_state,
    upsert_product_build_step,
)
from app.services.product_processing.product_build_status_service import (
    build_all_product_build_statuses,
    build_product_build_status,
)
from app.services.product_processing.product_journey_overview_service import build_product_journey_overview
from app.services.product_processing.product_build_telemetry_service import build_product_build_telemetry_report
from app.services.product_processing.product_build_orchestrator import (
    ProductBuildOrchestrationOptions,
    ensure_product_build_orchestration,
)
from app.services.product_processing.product_build_activation_service import (
    activate_product_builds_for_paid_order,
)
__all__ = [
    "ACP_REQUIRED_STAGE_KEYS",
    "BLUEPRINT_TIER_POLICY",
    "AcpDirectRouteResolution",
    "AcpStageReadinessEntry",
    "BlueprintTierPolicy",
    "BlueprintUncertainty",
    "PremiumEnrichmentItem",
    "PremiumEnrichmentWorkspace",
    "PremiumSelectiveReprocessResult",
    "PremiumUncertaintyResolutionRequest",
    "ProductBuildAction",
    "ProductBuildActionState",
    "ProductBuildAttentionItem",
    "ProductBuildAttentionSeverity",
    "ProductBuildAttentionSummary",
    "ProductBuildCurrentActivity",
    "ProductBuildDeliverableState",
    "ProductBuildDeliverableStatus",
    "ProductBuildEntitlement",
    "ProductBuildLifecycle",
    "ProductBuildProductKey",
    "ProductBuildProgress",
    "ProductBuildRecoverableError",
    "ProductBuildStageStatus",
    "ProductBuildStatus",
    "ProductBuildTelemetryEvent",
    "ProductBuildTelemetryProductSummary",
    "ProductBuildTelemetryReport",
    "ProductBuildTelemetryTotals",
    "ProductJourneyCurrentStage",
    "ProductJourneyDeliverableSummary",
    "ProductJourneyOutcome",
    "ProductJourneyOverview",
    "ProductJourneyProductSummary",
    "ProductJourneyRecommendedAction",
    "ProductBuildOrchestrationOptions",
    "UncertaintyBacklogEntry",
    "UncertaintyBacklogStatus",
    "ProductProcessingMode",
    "ProductProcessingProfile",
    "QuestionPolicyMode",
    "UncertaintyClassification",
    "UncertaintyDisposition",
    "UncertaintyKind",
    "UncertaintyOption",
    "classify_uncertainty_for_profile",
    "get_product_processing_profile",
    "backlog_entry_from_record",
    "ensure_product_build_run",
    "list_uncertainty_backlog",
    "list_product_build_runs",
    "list_product_build_steps",
    "prioritize_uncertainty_backlog",
    "resolve_uncertainty",
    "build_premium_enrichment_workspace",
    "build_all_product_build_statuses",
    "build_product_build_status",
    "build_product_journey_overview",
    "build_product_build_telemetry_report",
    "ensure_product_build_orchestration",
    "ensure_acp_product_orchestration",
    "activate_product_builds_for_paid_order",
    "build_acp_direct_resolution",
    "acp_route_blocking_reasons",
    "calculate_product_build_percent",
    "defer_premium_uncertainty_to_acp",
    "dismiss_premium_uncertainty",
    "resolve_premium_uncertainty",
    "resolve_product_processing_mode",
    "sync_premium_enrichment_product_run",
    "sync_product_builds_after_attention_action",
    "sync_product_builds_after_stage_approval",
    "supersede_unresolved_uncertainties",
    "update_product_build_run_state",
    "upsert_uncertainty_backlog",
    "upsert_product_build_step",
]


def __getattr__(name: str):
    if name in {
        "ACP_REQUIRED_STAGE_KEYS",
        "acp_route_blocking_reasons",
        "build_acp_direct_resolution",
    }:
        from app.services.product_processing import acp_direct_service

        return getattr(acp_direct_service, name)
    if name == "ensure_acp_product_orchestration":
        from app.services.product_processing import acp_product_orchestration_service

        return getattr(acp_product_orchestration_service, name)
    if name in {
        "build_premium_enrichment_workspace",
        "defer_premium_uncertainty_to_acp",
        "dismiss_premium_uncertainty",
        "resolve_premium_uncertainty",
        "sync_premium_enrichment_product_run",
    }:
        from app.services.product_processing import premium_enrichment_service

        return getattr(premium_enrichment_service, name)
    if name == "sync_product_builds_after_attention_action":
        from app.services.product_processing import product_build_attention_sync_service

        return getattr(product_build_attention_sync_service, name)
    if name == "sync_product_builds_after_stage_approval":
        from app.services.product_processing import product_build_progress_sync_service

        return getattr(product_build_progress_sync_service, name)
    raise AttributeError(f"module 'app.services.product_processing' has no attribute {name!r}")
