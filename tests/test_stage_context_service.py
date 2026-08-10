from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from sqlmodel import Session

from app.core.config import get_settings
from app.models import (
    AgentExecutionBackend,
    ArtifactStatus,
    BlueprintArtifact,
    BlueprintTool,
    BlueprintVersionEntry,
    CanvasArtifact,
    CodexLocalProviderConfig,
    DeepSeekProviderConfig,
    DiscoveryArtifact,
    DiscoveryInput,
    EmbeddingPolicy,
    GroundingPolicy,
    IngestionPolicy,
    KnowledgeAccessBackend,
    KnowledgeDocumentStatus,
    KnowledgeProfile,
    KnowledgeScope,
    KnowledgeSearchHit,
    KnowledgeSearchResponse,
    KnowledgeSource,
    KnowledgeVisibility,
    LLMProviderKey,
    LLMRuntimeSettings,
    MemoryProfile,
    OpenAIProviderConfig,
    RefreshPolicy,
    RetrievalPolicyProfile,
    ReviewState,
    SessionCreateResponse,
    SessionSnapshot,
    SessionStage,
    utc_now,
)
from app.services.llm_runtime.api_context_adapter import APIProviderContextAdapter
from app.services.llm_runtime.codex_cli.context_assembler import CodexContextInlineSource
from app.services.llm_runtime.codex_cli.provider_facade import CodexLocalBuilderService
from app.services.llm_runtime.stage_context_service import StageContextService, StageKnowledgePlanner
from tests.api_testkit import TEST_EMAIL, TEST_PASSWORD, build_test_client


def _write_doc(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _runtime_settings(*, backend: KnowledgeAccessBackend) -> LLMRuntimeSettings:
    return LLMRuntimeSettings(
        active_provider=LLMProviderKey.codex_local,
        agent_execution_backend=AgentExecutionBackend.codex_cli,
        knowledge_access_backend=backend,
        openai=OpenAIProviderConfig(
            fast_model="gpt-5.4-mini",
            reasoning_model="gpt-5.5",
            reasoning_effort="low",
            api_key_configured=True,
            available=True,
            status_note="ready",
        ),
        deepseek=DeepSeekProviderConfig(
            base_url="https://api.deepseek.com",
            fast_model="deepseek-v4-flash",
            reasoning_model="deepseek-v4-pro",
            reasoning_effort="high",
            api_key_configured=True,
            available=True,
            status_note="ready",
        ),
        codex_local=CodexLocalProviderConfig(
            command="codex",
            model="gpt-5.5",
            profile="ci3-stage-context",
            executable_found=True,
            available=True,
        ),
    )


def _auth_headers(client) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _complete_discovery_payload() -> dict:
    return DiscoveryInput(
        problem_statement="Disenar agentes de soporte con metodologia Lean y bajo riesgo operativo.",
        current_user="Arquitecto de soluciones",
        current_process="Recoge discovery, decide arquitectura y redacta artefactos manualmente.",
        desired_outcome="Generar un blueprint implementable con tools, memoria y evaluacion.",
        autonomy_level="high",
        constraints=["Sin microservicios en MVP", "No side effects irreversibles sin aprobacion humana"],
        operational_baseline={
            "current_time_spent": "6 horas por caso",
            "current_cost": "Retrabajo tecnico y validaciones tardias",
            "frequent_errors": ["Se pierde contexto entre discovery y blueprint"],
            "automation_opportunities": ["Normalizar discovery", "Generar artefactos base"],
        },
        mvp_definition={
            "v1_scope": ["Capturar discovery estructurado", "Construir canvas y blueprint inicial"],
            "out_of_scope": ["Provisioning automatico"],
            "north_star_metric": "Paquete de implementacion util en una sola sesion",
            "non_delegable_decisions": ["Aprobar el handoff a implementacion"],
        },
    ).model_dump(mode="json")


def _manual_snapshot(*, knowledge_version: str = "2026.07", narrative_suffix: str = "") -> SessionSnapshot:
    now = utc_now()
    workspace_id = uuid4()
    session_id = uuid4()
    return SessionSnapshot(
        session=SessionCreateResponse(
            id=session_id,
            workspace_id=workspace_id,
            title="Caso CI3",
            status=ArtifactStatus.ready,
            current_stage=SessionStage.build_blueprint,
            created_at=now,
            updated_at=now,
        ),
        discovery=DiscoveryArtifact(
            problem_statement="Construir agentes Lean con continuidad de contexto entre etapas.",
            current_user="Arquitecto de soluciones",
            current_process="Analiza discovery y traduce arquitectura en sesiones separadas.",
            desired_outcome="Generar un blueprint implementable con memoria y trazabilidad.",
            autonomy_level="high",
            constraints=["Sin side effects irreversibles", "Preservar aprobaciones humanas"],
            case_type="sistema_multiagente",
            value_statement="Reducir retrabajo y perdida de contexto.",
            operational_baseline={
                "current_time_spent": "6 horas",
                "current_cost": "Retrabajo tecnico",
                "frequent_errors": ["Se pierde contexto entre etapas"],
                "automation_opportunities": ["Context packing", "Blueprint generation"],
            },
            mvp_definition={
                "v1_scope": ["Discovery", "Canvas", "Blueprint"],
                "out_of_scope": ["Provisioning automatico"],
                "north_star_metric": "Blueprint util en una sola sesion",
                "non_delegable_decisions": ["Aprobar promocion"],
            },
        ),
        canvas=CanvasArtifact(
            user_goal="Construir un blueprint coherente usando memoria corta y larga.",
            mvp_scope=["Discovery", "Canvas", "Blueprint", "Tools"],
            out_of_scope=["Provisioning automatico"],
            success_metric="Blueprint aprobado sin retrabajo mayor",
            primary_risk="Drift del contexto entre etapas",
            agent_profile={
                "mission": "Conectar etapas Lean con contexto gobernado.",
                "primary_user": "Arquitecto de soluciones",
                "agent_task": "Sintetizar discovery, canvas y blueprint sin perder trazabilidad.",
                "allowed_decisions": ["Proponer arquitectura", "Compactar contexto"],
                "prohibited_decisions": ["Ejecutar side effects irreversibles"],
                "key_inputs": ["Discovery aprobado", "Canvas aprobado", "Knowledge base"],
                "expected_outputs": ["Blueprint trazable", "Tooling minimo"],
                "human_approvals": ["Promocion a implementacion"],
                "success_metrics": ["Blueprint aprobado sin drift"],
            },
        ),
        blueprint=BlueprintArtifact(
            architecture="single_agent_with_skills",
            reasoning_pattern="Plan-and-Execute",
            memory_strategy="session_memory_with_checkpoints",
            tools=[
                BlueprintTool(
                    name="knowledge_retrieval",
                    purpose="Consultar runbooks y lineamientos aprobados.",
                    integration_kind="rag",
                    inputs=["query"],
                    outputs=["grounded_answer"],
                )
            ],
            memory_profile=MemoryProfile(
                strategy="session_memory_with_checkpoints",
                storage_layers=["session_state", "vector_store"],
                write_policy="Persist validated checkpoints",
                retrieval_policy="Recover only by session_id and stage references",
                review_trigger="Missing evidence or stale blueprint decisions",
                goal_drift_guard="Compare every proposal against approved canvas and discovery",
                grounding_policy=GroundingPolicy(
                    citations_policy="Responder solo con evidencia citada.",
                    confidence_policy="Priorizar fuentes aprobadas.",
                    no_evidence_behavior="Declarar insuficiencia y escalar.",
                    contradictory_evidence_behavior="Detener y pedir revision.",
                ),
                sensitivity_rules=["No exponer secretos ni prompts internos."],
            ),
            knowledge_profile=KnowledgeProfile(
                mode="rag",
                sources=[
                    KnowledgeSource(
                        key="system_analysis",
                        title="System analysis knowledge",
                        source_type="document_repository",
                        uri="repo://Docs/system-analysis",
                        owner="platform",
                        sensitivity="internal",
                        license="internal",
                        description="Lineamientos operativos y arquitectura aprobada.",
                        source_version=knowledge_version,
                    )
                ],
                ingestion_policy=IngestionPolicy(
                    parser="markdown",
                    chunking_policy="800_tokens_overlap_120",
                    metadata_fields=["stage_affinity", "authority_level"],
                    include_filters=["Docs/system-analysis/**/*.md"],
                ),
                embedding_policy=EmbeddingPolicy(
                    provider="openai",
                    model="text-embedding-3-small",
                    dimensions=1536,
                    version="1",
                ),
                retrieval_policy=RetrievalPolicyProfile(
                    top_k=4,
                    filters=["stage_affinity=design"],
                    search_mode="hybrid",
                    reranking_policy="bm25_plus_vector",
                    fallback_behavior="return_needs_resolution",
                ),
                refresh_policy=RefreshPolicy(
                    frequency="daily",
                    triggers=["manual_publish", "document_change"],
                    expiration_policy="quarterly_review",
                    deletion_policy="soft_delete_with_lineage",
                ),
                grounding_policy=GroundingPolicy(
                    citations_policy="Responder solo con evidencia citada.",
                    confidence_policy="Priorizar fuentes aprobadas.",
                    no_evidence_behavior="Declarar insuficiencia y escalar.",
                    contradictory_evidence_behavior="Detener y pedir revision.",
                ),
                sensitivity_rules=["No exponer secretos ni prompts internos."],
                notes="Base de conocimiento larga para design, tools y memory.",
            ),
            guardrails=["No inventar datos", "No promover cambios sin aprobacion"],
            readiness_state=ReviewState.partial,
            narrative=f"Blueprint base con memoria hibrida. {narrative_suffix}".strip(),
        ),
        blueprint_versions=[
            BlueprintVersionEntry(
                version_number=1,
                source_action="bootstrap",
                status=ArtifactStatus.ready,
                readiness_state=ReviewState.partial,
                architecture="single_agent_with_skills",
                reasoning_pattern="Plan-and-Execute",
                created_at=now,
            )
        ],
    )


def _search_hit(
    *,
    key: str,
    preview: str,
    relative_path: str,
    title: str,
    score: float,
    source_lineage: str,
) -> KnowledgeSearchHit:
    return KnowledgeSearchHit(
        document_id=uuid4(),
        scope=KnowledgeScope.platform,
        relative_path=relative_path,
        section_key=key,
        title=title,
        visibility=KnowledgeVisibility.platform,
        status=KnowledgeDocumentStatus.approved,
        authority_level="canonical",
        memory_usage="required_retrieval",
        stage_affinity=["design", "tools", "memory"],
        agent_affinity=["builder", "planner"],
        source_lineage=source_lineage,
        preview=preview,
        score=score,
        lexical_score=score,
        vector_score=0.41,
        version_number=1,
    )


def _search_response(
    *,
    items: list[KnowledgeSearchHit],
    corpus_hash: str,
    next_cursor: str = "",
    query: str = "etapa design arquitectura memoria tools",
) -> KnowledgeSearchResponse:
    return KnowledgeSearchResponse(
        query=query,
        role="planner",
        total_hits=len(items) + (1 if next_cursor else 0),
        grounded_hits=len(items),
        corpus_hash=corpus_hash,
        evidence_status="grounded" if items else "no_evidence",
        absence_reason="" if items else "no_evidence_for_stage",
        applied_filters=["scope=platform", "stage=design"],
        authorized_scopes=["platform"],
        citations=[item.source_lineage for item in items if item.source_lineage],
        next_cursor=next_cursor,
        discarded_hits=0,
        items=items,
    )


class _FakeKnowledgeService:
    def __init__(self, responses: list[KnowledgeSearchResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def search_governed(self, session: Session, **kwargs) -> KnowledgeSearchResponse:
        self.calls.append(kwargs)
        cursor = str(kwargs.get("cursor", ""))
        if cursor and len(self.responses) > 1:
            return self.responses[1]
        return self.responses[0]


def test_stage_knowledge_planner_supports_explicit_second_page() -> None:
    snapshot = _manual_snapshot()
    fake_service = _FakeKnowledgeService(
        [
            _search_response(
                items=[
                    _search_hit(
                        key="design-core",
                        preview="Arquitectura base single_agent_with_skills para etapa design.",
                        relative_path="Docs/system-analysis/design-core.md",
                        title="Design Core",
                        score=0.92,
                        source_lineage="Docs/system-analysis/design-core.md#design-core",
                    )
                ],
                corpus_hash="corp-001",
                next_cursor="cursor-page-2",
            ),
            _search_response(
                items=[
                    _search_hit(
                        key="design-memory",
                        preview="session_memory_with_checkpoints protege continuidad y aprobaciones.",
                        relative_path="Docs/system-analysis/design-memory.md",
                        title="Design Memory",
                        score=0.87,
                        source_lineage="Docs/system-analysis/design-memory.md#design-memory",
                    )
                ],
                corpus_hash="corp-001",
            ),
        ]
    )
    planner = StageKnowledgePlanner(knowledge_service=fake_service)
    approved_refs = StageContextService().approved_artifact_resolver.resolve(snapshot, stage="design")

    response, evidence, pages = planner.plan(
        cast(Session, object()),
        snapshot=snapshot,
        workspace_id=snapshot.session.workspace_id,
        session_id=snapshot.session.id,
        stage="design",
        role="builder",
        approved_refs=approved_refs,
        allow_second_page=True,
        page_size=1,
    )

    assert response is not None
    assert pages == 2
    assert len(evidence) == 2
    assert fake_service.calls[1]["cursor"] == "cursor-page-2"
    assert response.corpus_hash == "corp-001"


def test_stage_context_bundle_is_deterministic_and_changes_when_evidence_changes() -> None:
    snapshot = _manual_snapshot()
    fake_service = _FakeKnowledgeService(
        [
            _search_response(
                items=[
                    _search_hit(
                        key="design-core",
                        preview="Arquitectura base single_agent_with_skills con Plan-and-Execute.",
                        relative_path="Docs/system-analysis/design-core.md",
                        title="Design Core",
                        score=0.92,
                        source_lineage="Docs/system-analysis/design-core.md#design-core",
                    )
                ],
                corpus_hash="corp-001",
            )
        ]
    )
    service = StageContextService(
        knowledge_planner=StageKnowledgePlanner(knowledge_service=fake_service)
    )

    bundle_one = service.build(
        cast(Session, object()),
        workspace_id=snapshot.session.workspace_id,
        session_id=snapshot.session.id,
        session_snapshot=snapshot,
        capability="synthesize_blueprint_narrative",
        role="builder",
        stage="design",
        task_source_keys=["narrative_discovery", "narrative_canvas", "narrative_blueprint"],
        allow_second_page=False,
    )
    bundle_two = service.build(
        cast(Session, object()),
        workspace_id=snapshot.session.workspace_id,
        session_id=snapshot.session.id,
        session_snapshot=snapshot,
        capability="synthesize_blueprint_narrative",
        role="builder",
        stage="design",
        task_source_keys=["narrative_discovery", "narrative_canvas", "narrative_blueprint"],
        allow_second_page=False,
    )

    fake_service.responses[0] = _search_response(
        items=[
            _search_hit(
                key="design-core",
                preview="Arquitectura con Tree-of-Thought y handoffs visibles por etapa.",
                relative_path="Docs/system-analysis/design-core.md",
                title="Design Core",
                score=0.95,
                source_lineage="Docs/system-analysis/design-core.md#design-core",
            )
        ],
        corpus_hash="corp-002",
    )
    bundle_three = service.build(
        cast(Session, object()),
        workspace_id=snapshot.session.workspace_id,
        session_id=snapshot.session.id,
        session_snapshot=snapshot,
        capability="synthesize_blueprint_narrative",
        role="builder",
        stage="design",
        task_source_keys=["narrative_discovery", "narrative_canvas", "narrative_blueprint"],
        allow_second_page=False,
    )

    assert bundle_one.context_fingerprint == bundle_two.context_fingerprint
    assert bundle_one.strict_budget is not None
    assert bundle_one.strict_budget.max_tokens == 2200
    assert bundle_one.corpus_hash == "corp-001"
    assert bundle_three.context_fingerprint != bundle_one.context_fingerprint


def test_api_and_codex_builders_share_same_bundle_semantics() -> None:
    snapshot = _manual_snapshot()
    fake_service = _FakeKnowledgeService(
        [
            _search_response(
                items=[
                    _search_hit(
                        key="design-core",
                        preview="Arquitectura single_agent_with_skills y memoria por checkpoints.",
                        relative_path="Docs/system-analysis/design-core.md",
                        title="Design Core",
                        score=0.91,
                        source_lineage="Docs/system-analysis/design-core.md#design-core",
                    )
                ],
                corpus_hash="corp-010",
            )
        ]
    )
    bundle = StageContextService(
        knowledge_planner=StageKnowledgePlanner(knowledge_service=fake_service)
    ).build(
        cast(Session, object()),
        workspace_id=snapshot.session.workspace_id,
        session_id=snapshot.session.id,
        session_snapshot=snapshot,
        capability="synthesize_blueprint_narrative",
        role="builder",
        stage="design",
        task_source_keys=["narrative_discovery", "narrative_canvas", "narrative_blueprint"],
        allow_second_page=False,
    )

    inline_sources = [
        CodexContextInlineSource(
            key="narrative_discovery",
            title="Discovery for blueprint narrative",
            content=json.dumps(snapshot.discovery.model_dump(mode="json"), ensure_ascii=True, indent=2),
            required=True,
            summary="Discovery aprobado para la narrativa tecnica.",
        )
    ]
    api_envelope = APIProviderContextAdapter().build(
        role="builder",
        task_kind="openai_blueprint_narrative",
        knowledge_access_backend="hybrid",
        task_instruction="Sintetiza la narrativa sin alterar el contrato del blueprint.",
        inline_sources=inline_sources,
        workspace_id=bundle.workspace_id,
        session_id=bundle.session_id,
        session_snapshot=bundle.session_snapshot,
        knowledge_manifest=bundle.knowledge_manifest,
        memory_policy=bundle.memory_policy,
        short_term_memory=bundle.short_term_memory,
        approved_refs=bundle.approved_refs,
        retrieved_hits=bundle.retrieved_hits,
        strict_budget=bundle.strict_budget,
        stage_hint=bundle.stage,
        context_fingerprint=bundle.context_fingerprint,
        corpus_hash=bundle.corpus_hash,
        retrieval_pages=bundle.retrieval_pages,
        absence_reason=bundle.absence_reason,
    )
    codex_service = CodexLocalBuilderService(_runtime_settings(backend=KnowledgeAccessBackend.hybrid))
    codex_envelope = codex_service._build_context_envelope(
        task_kind="blueprint_narrative",
        role="builder",
        sources=inline_sources,
        context_bundle=bundle,
    )

    assert [item["key"] for item in api_envelope.used_sources] == [
        item["key"] for item in codex_envelope.used_sources
    ]
    assert api_envelope.context_stats["context_fingerprint"] == codex_envelope.context_stats["context_fingerprint"]
    assert api_envelope.context_stats["corpus_hash"] == codex_envelope.context_stats["corpus_hash"]
    assert api_envelope.context_stats["assembled_estimated_tokens"] <= api_envelope.context_stats["budget_tokens"] + 5


def test_session_routes_emit_context_fingerprint_for_llm_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "Docs"
    _write_doc(
        docs_root / "system-analysis" / "design-core.md",
        "# Design Core\n\nArquitectura single_agent_with_skills para etapa design y tools.\n",
    )

    settings = get_settings()
    original_docs_root = settings.knowledge_docs_root
    settings.knowledge_docs_root = docs_root
    try:
        with build_test_client(monkeypatch) as client:
            headers = _auth_headers(client)
            create_response = client.post("/api/v1/sessions", headers=headers)
            assert create_response.status_code == 201
            session_id = create_response.json()["id"]

            discovery_response = client.post(
                f"/api/v1/sessions/{session_id}/normalize-discovery",
                headers=headers,
                json=_complete_discovery_payload(),
            )
            assert discovery_response.status_code == 200
            discovery_trace = discovery_response.json()["llm_trace"]
            assert discovery_trace["context_stats"]["context_fingerprint"]
            assert discovery_trace["context_stats"]["stage_hint"] == "discover"

            canvas_response = client.post(
                f"/api/v1/sessions/{session_id}/build-canvas",
                headers=headers,
            )
            assert canvas_response.status_code == 200
            canvas_trace = canvas_response.json()["llm_trace"]
            assert canvas_trace["context_stats"]["context_fingerprint"]
            assert canvas_trace["context_stats"]["stage_hint"] == "define"

            blueprint_response = client.post(
                f"/api/v1/sessions/{session_id}/build-blueprint",
                headers=headers,
            )
            assert blueprint_response.status_code == 200
            blueprint_trace = blueprint_response.json()["llm_trace"]
            assert blueprint_trace["context_stats"]["context_fingerprint"]
            assert blueprint_trace["context_stats"]["corpus_hash"]
            assert blueprint_trace["context_stats"]["stage_hint"] == "design"
    finally:
        settings.knowledge_docs_root = original_docs_root
