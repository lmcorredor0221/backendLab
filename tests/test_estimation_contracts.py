from uuid import uuid4

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import EstimationReportArtifact, LLMProviderKey, RuntimeCatalogEntryRecord, RuntimeFeatureFlagRecord
from app.services.workspace_bootstrap import apply_workspace_bootstrap


def test_estimation_report_contract_defaults_are_stable() -> None:
    artifact = EstimationReportArtifact()

    assert artifact.contract_version == "estimation-report.v1"
    assert artifact.maturity_stage == "canvas"
    assert artifact.blueprint_version_number is None
    assert artifact.current_blueprint_version_number is None
    assert artifact.is_stale is False
    assert artifact.stale_reasons == []
    assert artifact.traditional.scenario_type == "traditional"
    assert artifact.agentic.scenario_type == "agentic"
    assert artifact.agentic.active_provider == LLMProviderKey.openai
    assert artifact.agentic.blueprint_design_coverage_percent == 0
    assert artifact.agentic.acp_package_readiness_percent == 0
    assert artifact.agentic.implementation_scope_coverage_percent == 0
    assert artifact.agentic.automation_assessments == []
    assert artifact.construction_scenarios == []
    assert artifact.agentic.provider_model == ""
    assert artifact.agentic.pricing_snapshot is None
    assert artifact.confidence.label == "low"


def test_workspace_bootstrap_seeds_structured_estimation_catalogs() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        apply_workspace_bootstrap(session, uuid4())

        automation_row = session.exec(
            select(RuntimeCatalogEntryRecord).where(
                RuntimeCatalogEntryRecord.catalog_key == "estimation_automation_matrix",
                RuntimeCatalogEntryRecord.item_key == "discovery_canvas",
            )
        ).first()
        assert automation_row is not None
        assert automation_row.payload["family_key"] == "discovery_canvas"
        assert automation_row.payload["bands"][0]["complexity"] == "simple"
        assert automation_row.payload["bands"][0]["base_automation_percent"] == 85
        assert automation_row.payload["penalty_rules"] == []
        assert automation_row.payload["bonus_rules"] == []

        deployment_row = session.exec(
            select(RuntimeCatalogEntryRecord).where(
                RuntimeCatalogEntryRecord.catalog_key == "estimation_automation_matrix",
                RuntimeCatalogEntryRecord.item_key == "deployment_infra",
            )
        ).first()
        assert deployment_row is not None
        assert deployment_row.payload["penalty_rules"][0]["rule_key"] == "target_environment_unknown"
        assert deployment_row.payload["bonus_rules"][0]["rule_key"] == "acp_ready_to_build"

        pricing_row = session.exec(
            select(RuntimeCatalogEntryRecord).where(
                RuntimeCatalogEntryRecord.catalog_key == "estimation_pricing_profiles",
                RuntimeCatalogEntryRecord.item_key == "codex_local_hybrid",
            )
        ).first()
        assert pricing_row is not None
        assert pricing_row.payload["provider"] == "codex_local"
        assert pricing_row.payload["local_cost_policy"] == "hybrid"
        assert pricing_row.payload["cop_per_usd"] == 4000
        assert pricing_row.payload["rates"][0]["metric_key"] == "compute_hour_core"

        feature_flag = session.exec(
            select(RuntimeFeatureFlagRecord).where(RuntimeFeatureFlagRecord.flag_key == "estimation_comparative_v1")
        ).first()
        assert feature_flag is not None
        assert feature_flag.enabled is True

        rate_row = session.exec(
            select(RuntimeCatalogEntryRecord).where(
                RuntimeCatalogEntryRecord.catalog_key == "estimation_role_rates",
                RuntimeCatalogEntryRecord.item_key == "backend_engineer",
            )
        ).first()
        assert rate_row is not None
        assert rate_row.payload["currency"] == "COP"
        assert rate_row.payload["hourly_rate"] == 145000

        confidence_weight_row = session.exec(
            select(RuntimeCatalogEntryRecord).where(
                RuntimeCatalogEntryRecord.catalog_key == "estimation_confidence_weights",
                RuntimeCatalogEntryRecord.item_key == "blocking_gap_penalty",
            )
        ).first()
        assert confidence_weight_row is not None
        assert confidence_weight_row.payload["metric_key"] == "blocking_gap_penalty"
        assert confidence_weight_row.payload["amount"] == 4

        implementation_gap_weight_row = session.exec(
            select(RuntimeCatalogEntryRecord).where(
                RuntimeCatalogEntryRecord.catalog_key == "estimation_confidence_weights",
                RuntimeCatalogEntryRecord.item_key == "implementation_gap_penalty",
            )
        ).first()
        assert implementation_gap_weight_row is not None
        assert implementation_gap_weight_row.payload["metric_key"] == "implementation_gap_penalty"
        assert implementation_gap_weight_row.payload["amount"] == 2
