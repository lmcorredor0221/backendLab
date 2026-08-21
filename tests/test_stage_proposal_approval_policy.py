from app.models import (
    CommercialTier,
    DesignCritiqueFinding,
    DesignRecommendationArtifact,
    MemoryRecommendationArtifact,
    MemoryRecommendationFinding,
    MemoryToolDependency,
    ReviewState,
)
from app.services.llm_runtime.builder_contracts import DefinitionValidationSummary, RequirementsDefinitionOutput
from app.services.stage_proposal_service import (
    _definition_approval_blocking_issues,
    _design_approval_blocking_issues,
    _memory_approval_blocking_issues,
)


def test_basic_blueprint_defers_define_quality_blockers_at_approval() -> None:
    definition = RequirementsDefinitionOutput(
        validation=DefinitionValidationSummary(
            blocking_issues=[
                "untraced_item:NFR-001",
                "vague_nfr:NFR-002",
                "Resolver pregunta bloqueante: OQ-001",
            ],
        )
    )

    assert _definition_approval_blocking_issues(definition, CommercialTier.blueprint) == []


def test_premium_blueprint_keeps_define_quality_blockers_at_approval() -> None:
    issues = ["untraced_item:NFR-001", "vague_nfr:NFR-002"]
    definition = RequirementsDefinitionOutput(
        validation=DefinitionValidationSummary(blocking_issues=issues)
    )

    assert _definition_approval_blocking_issues(definition, CommercialTier.blueprint_pro) == issues


def test_acp_keeps_define_quality_blockers_at_approval() -> None:
    issues = ["untraced_item:NFR-001"]
    definition = RequirementsDefinitionOutput(
        validation=DefinitionValidationSummary(blocking_issues=issues)
    )

    assert _definition_approval_blocking_issues(definition, CommercialTier.acp) == issues


def test_basic_blueprint_defers_design_open_questions_at_approval() -> None:
    design = DesignRecommendationArtifact(
        critic_findings=[
            DesignCritiqueFinding(
                finding_key="DMC-001",
                title="Confirmar ownership de arquitectura",
                severity="blocking",
            )
        ],
        missing_information=["Owner tecnico del flujo"],
        open_questions=["Que meta numerica debe usarse para evaluar disminucion de tiempo?"],
        review_state=ReviewState.blocked,
    )

    assert _design_approval_blocking_issues(design, CommercialTier.blueprint) == []


def test_premium_blueprint_keeps_design_questions_as_approval_blockers() -> None:
    design = DesignRecommendationArtifact(
        critic_findings=[
            DesignCritiqueFinding(
                finding_key="DMC-001",
                title="Confirmar ownership de arquitectura",
                severity="blocking",
            )
        ],
        missing_information=["Owner tecnico del flujo"],
        open_questions=["Que meta numerica debe usarse para evaluar disminucion de tiempo?"],
        review_state=ReviewState.blocked,
    )

    issues = _design_approval_blocking_issues(design, CommercialTier.blueprint_pro)

    assert "design_review_state:blocked" in issues
    assert "blocking_finding:DMC-001" in issues
    assert "missing_information:Owner tecnico del flujo" in issues
    assert "open_question:Que meta numerica debe usarse para evaluar disminucion de tiempo?" in issues


def test_basic_blueprint_defers_memory_quality_blockers_at_approval() -> None:
    memory = MemoryRecommendationArtifact(
        critic_findings=[
            MemoryRecommendationFinding(
                finding_key="MEM-001",
                title="Cerrar TTL exacto de memoria",
                severity="blocking",
            )
        ],
        missing_information=["Owner de knowledge base"],
        open_questions=["Que proveedor de embeddings se usara en implementacion?"],
        review_state=ReviewState.blocked,
        tool_dependencies=[
            MemoryToolDependency(tool_key="knowledge_retrieval", required=True, status="approved"),
            MemoryToolDependency(tool_key="document_ingestion", required=True, status="approved"),
        ],
    )

    assert _memory_approval_blocking_issues(memory, CommercialTier.blueprint) == []


def test_premium_blueprint_keeps_memory_quality_blockers_at_approval() -> None:
    memory = MemoryRecommendationArtifact(
        critic_findings=[
            MemoryRecommendationFinding(
                finding_key="MEM-001",
                title="Cerrar TTL exacto de memoria",
                severity="blocking",
            )
        ],
        missing_information=["Owner de knowledge base"],
        open_questions=["Que proveedor de embeddings se usara en implementacion?"],
        review_state=ReviewState.blocked,
    )

    issues = _memory_approval_blocking_issues(memory, CommercialTier.blueprint_pro)

    assert "memory_review_state:blocked" in issues
    assert "blocking_finding:MEM-001" in issues
    assert "missing_information:Owner de knowledge base" in issues
    assert "open_question:Que proveedor de embeddings se usara en implementacion?" in issues


def test_basic_blueprint_still_blocks_memory_when_required_tool_is_missing() -> None:
    memory = MemoryRecommendationArtifact(
        tool_dependencies=[
            MemoryToolDependency(tool_key="document_ingestion", required=True, status="missing"),
        ],
    )

    assert _memory_approval_blocking_issues(memory, CommercialTier.blueprint) == [
        "missing_required_tool:document_ingestion"
    ]
