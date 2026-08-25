from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app.models import (
    JourneyArtifactState,
    JourneyStageArtifactRecord,
    SessionRecord,
    UserRecord,
    WorkspaceRecord,
    WorkspaceRole,
)
from app.api.routes.diagram_center import _governance_entry
from app.services.diagram_center.contracts import (
    DiagramEdge,
    DiagramGenerationInput,
    DiagramLane,
    DiagramModel,
    DiagramNode,
    DiagramPool,
    StructuredDiagramModel,
    StructuredDiagramNode,
    StructuredDiagramNodeMetadata,
    StructuredDiagramPool,
    StructuredDiagramLane,
)
from app.services.diagram_center.catalog_service import _renderings_need_refresh, build_catalog_v3, build_diagram_detail_v3
from app.services.diagram_center.generation_service import run_generation_job
from app.services.diagram_center.persistence import DiagramGovernanceRecord, DiagramVersionRecord
from app.services.diagram_center.quality_service import evaluate_diagram_quality
from app.services.diagram_center.registry_service import build_prompt_spec, get_registry_entry, list_registry_entries, load_diagram_registry
from app.services.diagram_center.renderer_service import render_diagram
from app.services.llm_runtime.capability_registry import BuilderCapability, get_builder_capability_spec
from app.services.llm_runtime.builder_contracts import LLMArtifactResult
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    with build_test_client(monkeypatch) as test_client:
        yield test_client


def _auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_diagram_governance_entry_exposes_taxonomy_metadata_for_admin_tables() -> None:
    with _session() as db:
        entry = _governance_entry(db, "architecture_overview")

    assert entry.diagram_key == "architecture_overview"
    assert entry.description
    assert entry.category == "architecture"
    assert entry.diagram_surface == "agent_design"
    assert entry.product_scope == ["blueprint"]
    assert entry.enabled_from_stage == "design"
    assert entry.access_level == "sample"
    assert entry.default_generation_state == "generated"
    assert entry.formats["preferred"] == "svg"
    assert "svg" in entry.formats["available"]
    assert "blueprint.architecture_spec" in entry.source_artifact_keys
    assert entry.portable_paths
    assert entry.active is True


def test_registry_covers_standardized_diagram_families_and_prompt_specs() -> None:
    registry = load_diagram_registry()
    keys = {entry.key for entry in registry.entries}
    effective_keys = {entry.key for entry in list_registry_entries()}

    assert len(keys) == 33
    assert {"problem_context_map", "stakeholder_map", "current_process_map", "traceability_matrix"} <= effective_keys
    assert {
        "solution_architecture",
        "logical_architecture",
        "physical_architecture",
        "component_diagram",
        "deployment_diagram",
        "class_diagram",
        "entity_relationship",
        "bpmn_process",
        "sequence_diagram",
        "state_diagram",
        "ux_navigation_flow",
        "user_journey",
        "capability_map",
        "context_map",
        "c4_context",
        "c4_container",
        "c4_component",
        "c4_code",
    } <= keys
    prompt = build_prompt_spec(next(entry for entry in list_registry_entries() if entry.key == "sequence_diagram"))
    assert prompt["output_contract"] == "plantuml-source.v1"
    assert prompt["notation"] == "sequence"
    assert prompt["standard"] == "UML Sequence Diagram"
    assert prompt["renderer_key"] == "renderer.plantuml.v1"
    assert prompt["semantic_rules"]
    assert prompt["quality_gates"]
    supported_contracts = {
        "diagram-model.v1",
        "diagram-presentation.v1",
        "plantuml-source.v1",
        "bpmn-source.v1",
        "c4-source.v1",
        "mermaid-source.v1",
    }
    effective_entries = list_registry_entries()
    use_case_prompt = build_prompt_spec(next(entry for entry in effective_entries if entry.key == "use_case_diagram"))
    activity_prompt = build_prompt_spec(next(entry for entry in effective_entries if entry.key == "activity_diagram"))
    bpmn_prompt = build_prompt_spec(next(entry for entry in effective_entries if entry.key == "bpmn_process"))
    current_process_prompt = build_prompt_spec(next(entry for entry in effective_entries if entry.key == "current_process_map"))
    assert use_case_prompt["standard"] == "UML Use Case Diagram"
    assert use_case_prompt["notation"] == "uml_use_case"
    assert activity_prompt["standard"] == "UML Activity Diagram"
    assert activity_prompt["notation"] == "uml_activity"
    assert bpmn_prompt["standard"] == "BPMN 2.0"
    assert bpmn_prompt["output_contract"] == "bpmn-source.v1"
    assert current_process_prompt["standard"] == "BPMN 2.0"
    assert current_process_prompt["notation"] == "bpmn"
    assert current_process_prompt["output_contract"] == "bpmn-source.v1"
    assert current_process_prompt["renderer_key"] == "renderer.bpmn_js.v1"
    assert current_process_prompt["validator_key"] == "bpmn.2_0.schema_semantic.v1"
    assert current_process_prompt["layout_guidance"]["preferred_strategy"] == "bpmn_swimlane"
    assert current_process_prompt["layout_guidance"]["must_split_when_dense"] is True
    for entry in effective_entries:
        governed_prompt = build_prompt_spec(entry)
        assert governed_prompt["diagram_key"] == entry.key
        assert governed_prompt["version"] == registry.prompt_spec_version
        assert governed_prompt["objective"].strip()
        assert governed_prompt["required_inputs"]
        assert governed_prompt["semantic_rules"]
        assert governed_prompt["output_contract"] in supported_contracts
        assert governed_prompt["standard"]
        assert governed_prompt["renderer_key"]
        assert governed_prompt["validator_key"]
        assert governed_prompt["layout_guidance"]["schema_version"] == "diagram-layout-guidance.v1"
        assert governed_prompt["layout_guidance"]["preferred_strategy"]


def test_structured_diagram_output_model_is_openai_compatible() -> None:
    spec = get_builder_capability_spec(BuilderCapability.generate_diagram_model)
    schema = spec.output_model.model_json_schema()
    metadata_schema = schema["$defs"]["StructuredDiagramMetadata"]
    node_metadata_schema = schema["$defs"]["StructuredDiagramNodeMetadata"]

    assert spec.output_model is StructuredDiagramModel
    assert metadata_schema["additionalProperties"] is False
    assert node_metadata_schema["additionalProperties"] is False

    structured = StructuredDiagramModel(
        diagram_key="current_process_map",
        title="Proceso actual",
        notation="bpmn",
        pools=[
            StructuredDiagramPool(
                id="operations_pool",
                label="Operacion",
                lanes=[StructuredDiagramLane(id="support_lane", label="Soporte")],
            )
        ],
        nodes=[
            StructuredDiagramNode(
                id="triage",
                label="Clasificar",
                kind="task",
                metadata=StructuredDiagramNodeMetadata(pool_id="operations_pool", lane_id="support_lane"),
                source_refs=["discover:1"],
            )
        ],
        source_refs=["discover:1"],
    )

    canonical = DiagramModel.model_validate(structured.model_dump(mode="json"))

    assert canonical.nodes[0].metadata["pool_id"] == "operations_pool"
    assert canonical.nodes[0].metadata["lane_id"] == "support_lane"


def test_diagram_model_quality_and_renderers_use_canonical_graph() -> None:
    model = DiagramModel(
        diagram_key="sequence_diagram",
        title="Secuencia de aprobación",
        notation="sequence",
        nodes=[
            DiagramNode(id="user", label="Usuario", kind="actor", source_refs=["requirement:1"]),
            DiagramNode(id="agent", label="Agente", kind="service", source_refs=["architecture:1"]),
        ],
        edges=[DiagramEdge(id="request", source="user", target="agent", label="Solicita aprobación", order=1)],
        source_refs=["requirement:1", "architecture:1"],
    )

    report = evaluate_diagram_quality(model)
    renderings = render_diagram(model)

    assert report.valid is True
    assert report.score == 100
    assert renderings["mermaid"].startswith("sequenceDiagram")
    assert "<svg" in renderings["svg"]
    assert "UML SEQUENCE" in renderings["svg"]
    assert 'data-diagram-notation="sequence"' in renderings["svg"]
    assert 'data-sequence-kind="lifeline"' in renderings["svg"]
    assert 'data-sequence-kind="message"' in renderings["svg"]
    assert "@startuml" in renderings["plantuml"]
    assert '"diagram_key": "sequence_diagram"' in renderings["presentation"]
    assert '"schema_version": "diagram-model.v1"' in renderings["json"]


def test_standard_specific_renderers_and_quality_warnings() -> None:
    use_case_model = DiagramModel(
        diagram_key="use_case_diagram",
        title="Casos de uso",
        notation="uml_use_case",
        nodes=[
            DiagramNode(id="customer", label="Cliente", kind="actor", source_refs=["define:1"]),
            DiagramNode(id="consult", label="Consultar estado", kind="use_case", source_refs=["define:1"]),
        ],
        edges=[DiagramEdge(id="customer_consult", source="customer", target="consult", label="solicita")],
        source_refs=["define:1"],
    )
    activity_model = DiagramModel(
        diagram_key="activity_diagram",
        title="Actividad",
        notation="uml_activity",
        nodes=[
            DiagramNode(id="start", label="Inicio", kind="start", source_refs=["discover:1"]),
            DiagramNode(id="triage", label="Clasificar solicitud", kind="activity", source_refs=["define:1"]),
            DiagramNode(id="route", label="Ruta valida", kind="decision", source_refs=["design:1"]),
        ],
        edges=[
            DiagramEdge(id="e1", source="start", target="triage"),
            DiagramEdge(id="e2", source="triage", target="route"),
        ],
        source_refs=["discover:1", "define:1", "design:1"],
    )
    bpmn_model = DiagramModel(
        diagram_key="current_process_map",
        title="Proceso actual",
        notation="bpmn",
        pools=[
            DiagramPool(
                id="customer_pool",
                label="Cliente",
                lanes=[DiagramLane(id="customer_lane", label="Solicitante")],
            ),
            DiagramPool(
                id="operations_pool",
                label="Operacion",
                lanes=[DiagramLane(id="support_lane", label="Soporte")],
            ),
        ],
        nodes=[
            DiagramNode(
                id="start",
                label="Solicitud",
                kind="start_event",
                metadata={"pool_id": "customer_pool", "lane_id": "customer_lane"},
                source_refs=["discover:1"],
            ),
            DiagramNode(
                id="triage",
                label="Clasificar",
                kind="task",
                metadata={"pool_id": "operations_pool", "lane_id": "support_lane"},
                source_refs=["discover:1"],
            ),
            DiagramNode(
                id="route",
                label="Decision",
                kind="exclusive_gateway",
                metadata={"pool_id": "operations_pool", "lane_id": "support_lane"},
                source_refs=["discover:1"],
            ),
            DiagramNode(
                id="end",
                label="Cierre",
                kind="end_event",
                metadata={"pool_id": "customer_pool", "lane_id": "customer_lane"},
                source_refs=["discover:1"],
            ),
        ],
        edges=[
            DiagramEdge(id="b1", source="start", target="triage", label="recibe", kind="message_flow"),
            DiagramEdge(id="b2", source="triage", target="route", label="evalua"),
            DiagramEdge(id="b3", source="route", target="end", label="resuelve", kind="message_flow"),
        ],
        source_refs=["discover:1"],
    )

    use_case_quality = evaluate_diagram_quality(use_case_model)
    activity_quality = evaluate_diagram_quality(activity_model)
    bpmn_quality = evaluate_diagram_quality(bpmn_model)
    use_case_renderings = render_diagram(use_case_model)
    activity_renderings = render_diagram(activity_model)
    bpmn_renderings = render_diagram(bpmn_model)

    assert use_case_quality.valid is True
    assert activity_quality.valid is True
    assert bpmn_quality.valid is True
    assert "@startuml" in use_case_renderings["plantuml"]
    assert 'actor "Cliente"' in use_case_renderings["plantuml"]
    assert 'data-diagram-notation="uml_use_case"' in use_case_renderings["svg"]
    assert 'data-node-kind="system_boundary"' in use_case_renderings["svg"]
    assert 'data-node-kind="actor"' in use_case_renderings["svg"]
    assert "<ellipse" in use_case_renderings["svg"]
    assert "@startuml" in activity_renderings["plantuml"]
    assert "if (Ruta valida?) then" in activity_renderings["plantuml"]
    assert "<bpmn:startEvent" in bpmn_renderings["bpmn_xml"]
    assert "<bpmn:exclusiveGateway" in bpmn_renderings["bpmn_xml"]
    assert "<bpmn:participant" in bpmn_renderings["bpmn_xml"]
    assert "<bpmn:lane" in bpmn_renderings["bpmn_xml"]
    assert "<bpmn:messageFlow" in bpmn_renderings["bpmn_xml"]
    assert 'data-diagram-notation="bpmn"' in bpmn_renderings["svg"]
    assert 'data-renderer-revision="diagram-renderer.v1.3.0"' in bpmn_renderings["svg"]
    assert 'data-bpmn-kind="pool"' in bpmn_renderings["svg"]
    assert 'data-bpmn-kind="lane-label"' in bpmn_renderings["svg"]
    assert 'data-pool-id="customer_pool"' in bpmn_renderings["svg"]
    assert 'data-lane-id="support_lane"' in bpmn_renderings["svg"]
    assert 'data-edge-kind="message_flow"' in bpmn_renderings["svg"]
    assert "BPMN 2.0" in bpmn_renderings["svg"]
    assert 'data-node-kind="exclusive_gateway"' in bpmn_renderings["svg"]


def test_detail_refreshes_legacy_generic_svg_for_specialized_notation() -> None:
    entry = get_registry_entry("use_case_diagram")
    assert entry is not None
    model = DiagramModel(
        diagram_key="use_case_diagram",
        title="Casos de uso",
        notation="uml_use_case",
        metadata={"renderer_key": entry.renderer_key},
        nodes=[
            DiagramNode(id="customer", label="Cliente", kind="actor_external", source_refs=["define:1"]),
            DiagramNode(id="consult", label="Consultar estado", kind="use_case", source_refs=["define:1"]),
        ],
        edges=[DiagramEdge(id="customer_consult", source="customer", target="consult", label="solicita")],
        source_refs=["define:1"],
    )
    legacy_renderings = {
        "svg": '<svg xmlns="http://www.w3.org/2000/svg"><rect x="1" y="1" width="10" height="10"/></svg>',
        "mermaid": "flowchart LR\ncustomer-->consult",
        "presentation": "{}",
    }
    fresh_renderings = render_diagram(model)

    assert _renderings_need_refresh(legacy_renderings, model, entry) is True
    assert _renderings_need_refresh(fresh_renderings, model, entry) is False


def test_detail_refreshes_legacy_generic_svg_for_sequence_notation() -> None:
    entry = get_registry_entry("sequence_diagram")
    assert entry is not None
    model = DiagramModel(
        diagram_key="sequence_diagram",
        title="Consulta documental",
        notation="sequence",
        metadata={"renderer_key": entry.renderer_key},
        nodes=[
            DiagramNode(id="user", label="Usuario", kind="actor", source_refs=["discover:1"]),
            DiagramNode(id="assistant", label="Asistente", kind="participant", source_refs=["design:1"]),
        ],
        edges=[DiagramEdge(id="m1", source="user", target="assistant", label="consulta", kind="sync_message", order=1)],
        source_refs=["discover:1", "design:1"],
    )
    legacy_renderings = {
        "svg": (
            '<svg xmlns="http://www.w3.org/2000/svg" data-diagram-notation="sequence" '
            'data-renderer-revision="diagram-renderer.v1.3.0"><rect x="1" y="1" width="10" height="10"/></svg>'
        ),
        "plantuml": "@startuml\nactor Usuario\n@enduml",
        "presentation": "{}",
    }
    fresh_renderings = render_diagram(model)

    assert _renderings_need_refresh(legacy_renderings, model, entry) is True
    assert 'data-sequence-kind="lifeline"' in fresh_renderings["svg"]
    assert _renderings_need_refresh(fresh_renderings, model, entry) is False


def test_detail_rehydrates_and_persists_legacy_current_process_when_policy_changes_to_bpmn(
) -> None:
    local_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(local_engine)
    legacy_model = DiagramModel(
        diagram_key="current_process_map",
        title="Diagrama del proceso actual",
        notation="flowchart",
        metadata={
            "standard": "Generic directed graph",
            "source_contract": "diagram-model.v1",
            "presentation_contract": "diagram-presentation.v1",
            "renderer_key": "renderer.svg.generic.v1",
            "validator_key": "diagram.graph_integrity.v1",
        },
        nodes=[
            DiagramNode(id="start", label="Solicitud del cliente", kind="start_event", source_refs=["discover:1"]),
            DiagramNode(id="triage", label="Asesor recibe solicitud", kind="task", source_refs=["discover:1"]),
            DiagramNode(id="decision", label="Determinar procedimiento", kind="exclusive_gateway", source_refs=["discover:1"]),
            DiagramNode(id="end", label="Cierre", kind="end_event", source_refs=["discover:1"]),
        ],
        edges=[
            DiagramEdge(id="e1", source="start", target="triage", label="recibe"),
            DiagramEdge(id="e2", source="triage", target="decision", label="evalua"),
            DiagramEdge(id="e3", source="decision", target="end", label="resuelve"),
        ],
        source_refs=["discover:1"],
    )
    legacy_renderings = render_diagram(legacy_model)
    assert 'data-diagram-notation="flowchart"' in legacy_renderings["svg"]
    with Session(local_engine) as db:
        user = UserRecord(email="diagram-policy@test.local", full_name="Diagram Policy", password_hash="test")
        db.add(user)
        db.commit()
        db.refresh(user)
        workspace = WorkspaceRecord(name="Diagram Policy Workspace", slug="diagram-policy-workspace", created_by_user_id=user.id)
        db.add(workspace)
        db.commit()
        db.refresh(workspace)
        record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Policy rehydration")
        db.add(record)
        db.add(
            DiagramGovernanceRecord(
                diagram_key="current_process_map",
                enabled=True,
                generation_enabled=True,
                required_tier_override="blueprint",
                preview_mode_override="full",
                prompt_status="active",
                prompt_override={
                    "notation": "bpmn",
                    "source_contract": "bpmn-source.v1",
                    "renderer_key": "renderer.bpmn_js.v1",
                    "validator_key": "bpmn.2_0.schema_semantic.v1",
                },
                notes="BPMN debe gobernar versiones legacy existentes.",
            )
        )
        db.commit()
        db.refresh(record)
        version = DiagramVersionRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            diagram_key="current_process_map",
            version_number=1,
            diagram_model=legacy_model.model_dump(mode="json"),
            renderings=legacy_renderings,
            quality_report=evaluate_diagram_quality(legacy_model).model_dump(mode="json"),
            source_fingerprint="legacy-policy",
            source_refs=["discover:1"],
            provider_key="codex_local",
            model_name="test",
            prompt_spec_version="diagram-prompts.v1.0.0",
            request_id="legacy",
        )
        db.add(version)
        db.commit()

        detail = build_diagram_detail_v3(
            db,
            record=record,
            role=None,
            diagram_key="current_process_map",
        )
        assert detail is not None
        assert detail.item.notation == "bpmn"
        assert detail.item.standard == "BPMN 2.0"
        assert detail.model.notation == "bpmn"
        assert detail.model.metadata["standard"] == "BPMN 2.0"
        assert detail.model.metadata["source_contract"] == "bpmn-source.v1"
        assert detail.model.metadata["renderer_key"] == "renderer.bpmn_js.v1"
        assert detail.model.metadata["validator_key"] == "bpmn.2_0.schema_semantic.v1"
        assert "bpmn_xml" in detail.renderings
        assert "<bpmn:startEvent" in detail.renderings["bpmn_xml"]
        assert 'data-diagram-notation="bpmn"' in detail.renderings["svg"]
        assert 'data-renderer-key="renderer.bpmn_js.v1"' in detail.renderings["svg"]
        assert 'data-renderer-revision="diagram-renderer.v1.3.0"' in detail.renderings["svg"]
        assert 'data-bpmn-kind="pool"' in detail.renderings["svg"]
        assert "BPMN 2.0" in detail.renderings["svg"]

        persisted = db.exec(
            select(DiagramVersionRecord).where(
                DiagramVersionRecord.session_id == record.id,
                DiagramVersionRecord.diagram_key == "current_process_map",
            )
        ).one()
        assert persisted.diagram_model["notation"] == "bpmn"
        assert persisted.diagram_model["metadata"]["source_contract"] == "bpmn-source.v1"
        assert persisted.renderings["svg"] == detail.renderings["svg"]
        assert "bpmn_xml" in persisted.renderings


def test_catalog_flags_legacy_renderer_revision_for_layout_upgrade() -> None:
    local_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(local_engine)
    entry = get_registry_entry("current_process_map")
    assert entry is not None
    model = DiagramModel(
        diagram_key="current_process_map",
        title="Diagrama del proceso actual",
        notation="bpmn",
        metadata={
            "standard": "BPMN 2.0",
            "source_contract": "bpmn-source.v1",
            "presentation_contract": "diagram-presentation.v1",
            "renderer_key": "renderer.bpmn_js.v1",
            "validator_key": "bpmn.2_0.schema_semantic.v1",
        },
        pools=[DiagramPool(id="ops", label="Operacion", lanes=[DiagramLane(id="support", label="Soporte")])],
        nodes=[
            DiagramNode(id="start", label="Solicitud", kind="start_event", metadata={"pool_id": "ops", "lane_id": "support"}, source_refs=["discover:1"]),
            DiagramNode(id="task", label="Atender", kind="task", metadata={"pool_id": "ops", "lane_id": "support"}, source_refs=["discover:1"]),
        ],
        edges=[DiagramEdge(id="e1", source="start", target="task", label="recibe")],
        source_refs=["discover:1"],
    )
    old_renderings = render_diagram(model)
    old_renderings["svg"] = old_renderings["svg"].replace(
        'data-renderer-revision="diagram-renderer.v1.3.0"',
        'data-renderer-revision="diagram-renderer.v1.2.0"',
    )
    assert 'data-renderer-revision="diagram-renderer.v1.2.0"' in old_renderings["svg"]

    with Session(local_engine) as db:
        user = UserRecord(email="diagram-upgrade@test.local", full_name="Diagram Upgrade", password_hash="test")
        db.add(user)
        db.commit()
        db.refresh(user)
        workspace = WorkspaceRecord(name="Diagram Upgrade Workspace", slug="diagram-upgrade-workspace", created_by_user_id=user.id)
        db.add(workspace)
        db.commit()
        db.refresh(workspace)
        record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Layout upgrade")
        db.add(record)
        db.commit()
        db.refresh(record)
        version = DiagramVersionRecord(
            workspace_id=record.workspace_id,
            session_id=record.id,
            diagram_key=entry.key,
            version_number=1,
            diagram_model=model.model_dump(mode="json"),
            renderings=old_renderings,
            quality_report=evaluate_diagram_quality(model).model_dump(mode="json"),
            source_fingerprint="legacy-renderer",
            source_refs=["discover:1"],
            provider_key="codex_local",
            model_name="test",
            prompt_spec_version="diagram-prompts.v1.0.0",
            request_id="legacy-renderer",
        )
        db.add(version)
        db.commit()

        catalog = build_catalog_v3(db, record=record, role=WorkspaceRole.owner)
        item = next(item for item in catalog.entries if item.key == entry.key)

        assert item.needs_layout_upgrade is True
        assert "diagram-renderer.v1.2.0" in item.layout_upgrade_reason
        assert "diagram-renderer.v1.3.0" in item.layout_upgrade_reason
        assert "layout_upgrade" in item.available_actions

        detail = build_diagram_detail_v3(db, record=record, role=WorkspaceRole.owner, diagram_key=entry.key)
        assert detail is not None
        assert detail.item.needs_layout_upgrade is True
        assert 'data-renderer-revision="diagram-renderer.v1.2.0"' in detail.renderings["svg"]

        persisted = db.exec(select(DiagramVersionRecord).where(DiagramVersionRecord.id == version.id)).one()
        assert persisted.renderings["svg"] == old_renderings["svg"]


def test_provider_registry_exposes_diagram_generation_as_governed_capability() -> None:
    spec = get_builder_capability_spec(BuilderCapability.generate_diagram_model)

    assert spec.output_model is StructuredDiagramModel
    assert spec.llm_required is True
    assert spec.fallback_policy == "fail_visible_without_synthetic_diagram"
    assert spec.prompt_version == "diagram-prompts.v1.0.0"
    assert "pools y lanes" in spec.system_instruction


def test_bpmn_prompt_spec_requires_dynamic_pools_lanes_and_message_flows() -> None:
    entry = get_registry_entry("current_process_map")
    assert entry is not None

    prompt_spec = build_prompt_spec(entry, override={"notation": "bpmn"})
    rules = "\n".join(prompt_spec["semantic_rules"])
    gates = "\n".join(prompt_spec["quality_gates"])

    assert prompt_spec["notation"] == "bpmn"
    assert prompt_spec["source_contract"] == "bpmn-source.v1"
    assert prompt_spec["renderer_key"] == "renderer.bpmn_js.v1"
    assert "Inferir dinamicamente pools" in rules
    assert "metadata.pool_id" in rules
    assert "message_flow" in rules
    assert "Cada nodo debe quedar asignado a una lane" in gates


def test_v3_catalog_replaces_legacy_local_catalog_and_explains_access(client: TestClient) -> None:
    headers = _auth_headers(client)
    created = client.post("/api/v1/sessions", headers=headers)
    assert created.status_code == 201
    project_id = created.json()["id"]

    response = client.get(f"/api/v3/projects/{project_id}/diagrams", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "diagram-catalog.v3"
    assert payload["total_count"] >= 33
    assert "problem_context_map" in {item["key"] for item in payload["entries"]}
    assert payload["provider_key"] in {"openai", "deepseek", "codex_local"}
    assert all("access" in item and item["access"]["reason"] for item in payload["entries"])
    assert any(item["access"]["access_state"] == "available" for item in payload["entries"])
    assert any(item["access"]["access_state"] == "stage_locked" for item in payload["entries"])


def test_generation_job_fails_visibly_when_approved_context_is_missing(client: TestClient) -> None:
    headers = _auth_headers(client)
    created = client.post("/api/v1/sessions", headers=headers)
    project_id = created.json()["id"]

    response = client.post(
        f"/api/v3/projects/{project_id}/diagrams/user_journey/generate",
        headers=headers,
        json={"detail_level": "standard", "reason": "user_request", "idempotency_key": "test-no-context"},
    )

    assert response.status_code == 202
    job_id = response.json()["id"]
    job_response = client.get(f"/api/v3/projects/{project_id}/diagram-jobs/{job_id}", headers=headers)
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "error"
    assert job_response.json()["error_code"] == "approved_context_missing"


def test_run_generation_job_passes_workspace_context_to_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    local_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(local_engine)
    captured: dict[str, object] = {}

    class FakeProvider:
        def generate_diagram_model(
            self,
            payload: DiagramGenerationInput,
            *,
            context_bundle=None,
        ) -> LLMArtifactResult:
            captured["payload"] = payload
            captured["context_bundle"] = context_bundle
            artifact = DiagramModel(
                diagram_key=payload.diagram_key,
                title=payload.title,
                notation=payload.notation,
                nodes=[
                    DiagramNode(
                        id="approved_context",
                        label="Approved context",
                        kind="task",
                        source_refs=list(payload.source_refs),
                    )
                ],
                edges=[],
                source_refs=list(payload.source_refs),
            )
            return LLMArtifactResult(
                artifact=artifact,
                provider_key="deepseek",
                model_name="deepseek-v4-pro",
                prompt_version="diagram-prompts.v1.0.0",
            )

    monkeypatch.setattr(
        "app.services.diagram_center.generation_service.build_builder_service",
        lambda runtime_settings: FakeProvider(),
    )

    with Session(local_engine) as db:
        user = UserRecord(email="diagram-runtime@test.local", full_name="Diagram Runtime", password_hash="test")
        db.add(user)
        db.commit()
        db.refresh(user)

        workspace = WorkspaceRecord(
            name="Diagram Runtime Workspace",
            slug="diagram-runtime-workspace",
            created_by_user_id=user.id,
        )
        db.add(workspace)
        db.commit()
        db.refresh(workspace)

        record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Diagram runtime propagation")
        db.add(record)
        db.commit()
        db.refresh(record)

        db.add(
            JourneyStageArtifactRecord(
                workspace_id=workspace.id,
                session_id=record.id,
                artifact_kind="discover",
                stage_key="discover",
                version_number=1,
                state=JourneyArtifactState.approved,
                proposal_payload={"summary": "Contexto aprobado para el diagrama."},
                input_fingerprint="discover-in",
                context_fingerprint="discover-ctx",
                output_fingerprint="discover-out",
                provider_key="deepseek",
                model="deepseek-v4-pro",
                execution_backend="provider_native",
                prompt_version="discover-prompts.v1.0.0",
                schema_version="discovery.v1",
                approved_by_user_id=user.id,
            )
        )
        db.flush()

        from app.services.diagram_center.persistence import DiagramGenerationJobRecord

        job = DiagramGenerationJobRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            diagram_key="sequence_diagram",
            requested_by_user_id=user.id,
            status="queued",
            detail_level="standard",
            reason="user_request",
            idempotency_key="ctx-propagation",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        workspace_id = workspace.id
        record_id = record.id

    run_generation_job(job.id, local_engine)

    context_bundle = captured.get("context_bundle")
    assert context_bundle is not None
    assert context_bundle.workspace_id == workspace_id
    assert context_bundle.session_id == record_id
    assert context_bundle.capability == BuilderCapability.generate_diagram_model.value


def test_run_generation_job_resolves_required_inputs_for_architecture_diagram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(local_engine)
    captured: dict[str, object] = {}

    class FakeProvider:
        def generate_diagram_model(
            self,
            payload: DiagramGenerationInput,
            *,
            context_bundle=None,
        ) -> LLMArtifactResult:
            captured["payload"] = payload
            artifact = DiagramModel(
                diagram_key=payload.diagram_key,
                title=payload.title,
                notation=payload.notation,
                nodes=[
                    DiagramNode(
                        id="architecture_overview",
                        label="Architecture overview",
                        kind="service",
                        source_refs=list(payload.source_refs),
                    )
                ],
                edges=[],
                source_refs=list(payload.source_refs),
            )
            return LLMArtifactResult(
                artifact=artifact,
                provider_key="deepseek",
                model_name="deepseek-v4-pro",
                prompt_version="diagram-prompts.v1.0.0",
            )

    monkeypatch.setattr(
        "app.services.diagram_center.generation_service.build_builder_service",
        lambda runtime_settings: FakeProvider(),
    )

    with Session(local_engine) as db:
        user = UserRecord(email="diagram-required-inputs@test.local", full_name="Diagram Inputs", password_hash="test")
        db.add(user)
        db.commit()
        db.refresh(user)

        workspace = WorkspaceRecord(
            name="Diagram Inputs Workspace",
            slug="diagram-inputs-workspace",
            created_by_user_id=user.id,
        )
        db.add(workspace)
        db.commit()
        db.refresh(workspace)

        record = SessionRecord(user_id=user.id, workspace_id=workspace.id, title="Architecture diagram input resolution")
        db.add(record)
        db.commit()
        db.refresh(record)

        db.add(
            JourneyStageArtifactRecord(
                workspace_id=workspace.id,
                session_id=record.id,
                artifact_kind="design_recommendation_artifact",
                stage_key="design",
                version_number=1,
                state=JourneyArtifactState.approved,
                proposal_payload={
                    "summary": "Arquitectura aprobada con handoffs y supervision.",
                    "selected_design": {
                        "alternative_key": "handoffs",
                        "architecture_pattern": "supervisor_with_specialists",
                        "reasoning_pattern": "route_then_execute",
                    },
                    "decision_rationale": "Separar consulta, grounding y validacion reduce riesgo y mejora trazabilidad.",
                },
                input_fingerprint="design-in",
                context_fingerprint="design-ctx",
                output_fingerprint="design-out",
                provider_key="deepseek",
                model="deepseek-v4-pro",
                execution_backend="provider_native",
                prompt_version="design-prompts.v1.0.0",
                schema_version="design-recommendation.v1",
                approved_by_user_id=user.id,
            )
        )
        db.flush()

        from app.services.diagram_center.persistence import DiagramGenerationJobRecord

        job = DiagramGenerationJobRecord(
            workspace_id=workspace.id,
            session_id=record.id,
            diagram_key="architecture_overview",
            requested_by_user_id=user.id,
            status="queued",
            detail_level="standard",
            reason="user_request",
            idempotency_key="architecture-required-inputs",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

    run_generation_job(job.id, local_engine)

    payload = captured.get("payload")
    assert isinstance(payload, DiagramGenerationInput)
    assert payload.required_inputs == ["blueprint.architecture_spec", "blueprint.patterns"]
    assert payload.missing_required_inputs == []
    assert {item["input_key"] for item in payload.resolved_inputs} == {
        "blueprint.architecture_spec",
        "blueprint.patterns",
    }
    assert "pattern=supervisor_with_specialists" in payload.context_brief
    assert payload.source_context["coverage_summary"]["resolved_input_count"] == 2
    assert payload.source_context["coverage_summary"]["missing_input_count"] == 0
    assert payload.source_context["approved_artifact_keys"][0] == "design_recommendation_artifact"
    assert payload.resolved_inputs[0]["matched_artifact_keys"] == ["design_recommendation_artifact"]


def test_failed_generation_idempotency_key_can_be_retried(client: TestClient) -> None:
    headers = _auth_headers(client)
    created = client.post("/api/v1/sessions", headers=headers)
    project_id = created.json()["id"]
    payload = {"detail_level": "standard", "reason": "user_request", "idempotency_key": "retry-after-error"}

    first = client.post(f"/api/v3/projects/{project_id}/diagrams/user_journey/generate", headers=headers, json=payload)
    assert first.status_code == 202
    first_job_id = first.json()["id"]
    first_job = client.get(f"/api/v3/projects/{project_id}/diagram-jobs/{first_job_id}", headers=headers)
    assert first_job.status_code == 200
    assert first_job.json()["status"] == "error"

    second = client.post(f"/api/v3/projects/{project_id}/diagrams/user_journey/generate", headers=headers, json=payload)
    assert second.status_code == 202
    second_job_id = second.json()["id"]
    second_job = client.get(f"/api/v3/projects/{project_id}/diagram-jobs/{second_job_id}", headers=headers)

    assert second_job_id != first_job_id
    assert second_job.status_code == 200
    assert second_job.json()["status"] == "error"
    assert second_job.json()["error_code"] == "approved_context_missing"


def test_governance_reuses_runtime_provider_and_audits_policy_changes(client: TestClient) -> None:
    headers = _auth_headers(client)

    overview = client.get("/api/v3/admin/diagram-governance/overview", headers=headers)
    assert overview.status_code == 200
    assert overview.json()["active_provider"] in {"openai", "deepseek", "codex_local"}
    assert overview.json()["prompt_spec_version"] == "diagram-prompts.v1.0.0"

    updated = client.patch(
        "/api/v3/admin/diagram-governance/activity_diagram",
        headers=headers,
        json={
            "enabled": True,
            "generation_enabled": True,
            "required_tier_override": "blueprint",
            "preview_mode_override": "full",
            "prompt_status": "active",
            "prompt_override": {"objective": "Objetivo de prueba gobernado."},
            "notes": "Cambio de certificación",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["prompt_spec"]["objective"] == "Objetivo de prueba gobernado."

    bpmn_override = client.patch(
        "/api/v3/admin/diagram-governance/current_process_map",
        headers=headers,
        json={
            "enabled": True,
            "generation_enabled": True,
            "required_tier_override": "blueprint",
            "preview_mode_override": "limited",
            "prompt_status": "active",
            "prompt_override": {"notation": "bpmn"},
            "notes": "Cambio a BPMN por gobierno de notacion",
        },
    )
    assert bpmn_override.status_code == 200
    assert bpmn_override.json()["prompt_spec"]["notation"] == "bpmn"
    assert bpmn_override.json()["prompt_spec"]["source_contract"] == "bpmn-source.v1"
    assert bpmn_override.json()["prompt_spec"]["renderer_key"] == "renderer.bpmn_js.v1"
    assert bpmn_override.json()["prompt_spec"]["validator_key"] == "bpmn.2_0.schema_semantic.v1"

    refreshed = client.get("/api/v3/admin/diagram-governance/overview", headers=headers)
    assert refreshed.status_code == 200
    audit = refreshed.json()["recent_audit"]
    assert audit[0]["diagram_key"] == "current_process_map"
    assert "prompt_override_hash" in audit[0]["changed_fields"]
    assert any(event["diagram_key"] == "activity_diagram" for event in audit)
