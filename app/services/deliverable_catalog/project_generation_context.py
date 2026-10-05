from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, field_validator

from app.models import ContractModel, PydanticField
from app.services.deliverable_catalog.contracts import DeliverableContextPolicy, LEAN_STAGE_ORDER


CONTEXT_VERSION: Literal["project-generation-context.v1"] = "project-generation-context.v1"
BUILDER_VERSION = "contextual-deterministic.v1"
_STAGE_KEYS = set(LEAN_STAGE_ORDER)
_SESSION_STAGE_ALIASES = {
    "discovery": "discover",
    "canvas": "define",
    "blueprint": "design",
    "latest_tool_recommendation": "tools",
    "estimation_report": "estimate",
}


class SourceReference(ContractModel):
    ref: str
    source_type: Literal["stage", "artifact", "snapshot", "acp_question", "task_ref", "unknown"] = "unknown"
    stage: str | None = None
    artifact_key: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"


class SourceVersion(ContractModel):
    source_ref: str
    version: str = ""


class MissingField(ContractModel):
    field: str
    reason: str
    expected_stage: str | None = None


class SpecificityAnchor(ContractModel):
    anchor_type: str
    value: str
    source_ref: str
    stage: str | None = None


class ProjectRole(ContractModel):
    value: str
    source_ref: str = ""
    stage: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence_kind: Literal["explicit", "approved_inference", "unknown"] = "explicit"


class ProjectTool(ContractModel):
    name: str
    purpose: str = ""
    source_ref: str = ""
    stage: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence_kind: Literal["explicit", "approved_inference", "unknown"] = "explicit"
    requires_approval: bool | None = None
    has_side_effects: bool | None = None


class ToolContractSummary(ContractModel):
    name: str
    purpose: str = ""
    category: Literal["design_contract", "operational_binding", "pending_binding"] = "design_contract"
    source_ref: str = ""


class KnowledgeSourceSummary(ContractModel):
    name: str
    source_type: str = ""
    permissions: str = ""
    source_ref: str = ""


class EstimationSummary(ContractModel):
    traditional_hours: float | None = None
    agentic_hours: float | None = None
    traditional_cost: float | None = None
    agentic_cost: float | None = None
    net_savings: float | None = None
    effort_reduction_percent: float | None = None
    confidence_label: str = ""
    source_ref: str = ""


class OpenQuestionSummary(ContractModel):
    question: str
    source_ref: str = ""
    stage: str | None = None
    blocking: bool | None = None


class ConstructionQuestionSummary(ContractModel):
    question_key: str = ""
    question_text: str
    answer_text: str = ""
    status: str = ""
    impacted_artifacts: list[str] = PydanticField(default_factory=list)
    source_ref: str = ""
    blocking: bool | None = None


class _ValueWithSource(ContractModel):
    model_config = ConfigDict(extra="forbid")

    value: Any
    source_ref: str
    stage: str | None = None
    artifact_key: str | None = None


class ProjectGenerationContext(ContractModel):
    context_version: Literal["project-generation-context.v1"] = CONTEXT_VERSION
    session_id: UUID | None = None
    workspace_id: UUID | None = None
    project_title: str | None = None
    problem_statement: str | None = None
    current_process: str | None = None
    current_user: str | None = None
    desired_outcome: str | None = None
    objectives: list[str] = PydanticField(default_factory=list)
    mvp_scope: list[str] = PydanticField(default_factory=list)
    out_of_scope: list[str] = PydanticField(default_factory=list)
    constraints: list[str] = PydanticField(default_factory=list)
    nondelegable_decisions: list[str] = PydanticField(default_factory=list)
    architecture: str | None = None
    reasoning_pattern: str | None = None
    coordination_model: str | None = None
    roles: list[ProjectRole] = PydanticField(default_factory=list)
    tools: list[ProjectTool] = PydanticField(default_factory=list)
    tool_contracts: list[ToolContractSummary] = PydanticField(default_factory=list)
    memory_strategy: str | None = None
    rag_required: bool | None = None
    knowledge_sources: list[KnowledgeSourceSummary] = PydanticField(default_factory=list)
    guardrails: list[str] = PydanticField(default_factory=list)
    risks: list[str] = PydanticField(default_factory=list)
    acceptance_criteria: list[str] = PydanticField(default_factory=list)
    validation_scenarios: list[str] = PydanticField(default_factory=list)
    estimation_summary: EstimationSummary | None = None
    open_questions: list[OpenQuestionSummary] = PydanticField(default_factory=list)
    construction_questions: list[ConstructionQuestionSummary] = PydanticField(default_factory=list)
    source_refs: list[SourceReference] = PydanticField(default_factory=list)
    source_versions: list[SourceVersion] = PydanticField(default_factory=list)
    missing_fields: list[MissingField] = PydanticField(default_factory=list)
    specificity_anchors: list[SpecificityAnchor] = PydanticField(default_factory=list)
    input_fingerprint: str = ""
    estimated_input_tokens: int = 0

    @field_validator(
        "objectives",
        "mvp_scope",
        "out_of_scope",
        "constraints",
        "nondelegable_decisions",
        "guardrails",
        "risks",
        "acceptance_criteria",
        "validation_scenarios",
        mode="before",
    )
    @classmethod
    def _normalize_string_list(cls, value: object) -> list[str]:
        return _string_list(value)

    @classmethod
    def from_approved_payload(
        cls,
        payload: dict[str, object],
        *,
        deliverable_key: str,
        policy: DeliverableContextPolicy | dict[str, object] | None,
    ) -> "ProjectGenerationContext":
        payload = payload or {}
        approved = payload.get("approved_context") if isinstance(payload.get("approved_context"), dict) else {}
        approved = approved if isinstance(approved, dict) else {}
        effective_policy = policy or payload.get("context_policy")
        allowed_stages = _allowed_stage_keys(effective_policy)
        stages = _filter_stages(approved.get("stages"), allowed_stages)
        artifacts = approved.get("artifacts") if isinstance(approved.get("artifacts"), dict) else {}
        snapshot = approved.get("snapshot") if isinstance(approved.get("snapshot"), dict) else {}
        if not snapshot and not approved:
            snapshot = payload
        construction_questions = approved.get("construction_questions") if isinstance(approved.get("construction_questions"), list) else []
        return cls._from_parts(
            payload=payload,
            stages=stages,
            artifacts=artifacts if isinstance(artifacts, dict) else {},
            snapshot=snapshot if isinstance(snapshot, dict) else {},
            construction_questions=construction_questions,
            deliverable_key=deliverable_key,
            policy=effective_policy,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, object],
        *,
        deliverable_key: str | None = None,
        policy: DeliverableContextPolicy | dict[str, object] | None = None,
    ) -> "ProjectGenerationContext":
        return cls._from_parts(
            payload=snapshot or {},
            stages={},
            artifacts={},
            snapshot=snapshot or {},
            construction_questions=[],
            deliverable_key=deliverable_key or "",
            policy=policy,
        )

    @classmethod
    def from_acp_inputs(
        cls,
        snapshot: dict[str, object],
        questions: list[object],
        artifacts: dict[str, object] | list[object],
        *,
        deliverable_key: str,
    ) -> "ProjectGenerationContext":
        artifact_map = _artifact_map(artifacts)
        return cls._from_parts(
            payload=snapshot or {},
            stages={},
            artifacts=artifact_map,
            snapshot=snapshot or {},
            construction_questions=questions or [],
            deliverable_key=deliverable_key,
            policy=None,
        )

    @classmethod
    def _from_parts(
        cls,
        *,
        payload: dict[str, object],
        stages: dict[str, object],
        artifacts: dict[str, object],
        snapshot: dict[str, object],
        construction_questions: list[object],
        deliverable_key: str,
        policy: DeliverableContextPolicy | dict[str, object] | None,
    ) -> "ProjectGenerationContext":
        sources: list[SourceReference] = _source_refs_from_payload(payload, stages, artifacts, bool(snapshot), construction_questions)
        source_versions = [SourceVersion(source_ref=source.ref, version=_version_from_ref(source.ref)) for source in sources]

        problem = _first_text(stages, ["discover"], ("problem_statement", "description", "problem"))
        desired = _first_text(stages, ["discover"], ("desired_outcome", "outcome", "target_outcome"))
        current_process = _first_text(stages, ["discover"], ("current_process", "process", "as_is_process"))
        current_user = _first_text(stages, ["discover"], ("current_user", "primary_user", "user", "actor"))
        objectives = _first_list(stages, ["define"], ("objectives", "goals", "business_objectives", "success_goals"))
        mvp_scope = _first_list(stages, ["define"], ("mvp_scope", "scope", "v1_scope", "functional_requirements", "requirements"))
        out_of_scope = _first_list(stages, ["define"], ("out_of_scope", "excluded_scope", "exclusions"))
        constraints = _first_list(stages, ["discover", "define"], ("constraints", "technical_constraints", "business_constraints"))
        nondelegable = _first_list(stages, ["define", "design"], ("nondelegable_decisions", "non_delegable_decisions", "human_approval_required", "hitl_rules"))
        architecture = _first_text(stages, ["design"], ("architecture", "architecture_pattern", "topology"))
        reasoning = _first_text(stages, ["design"], ("reasoning_pattern", "cognitive_pattern", "reasoning"))
        coordination = _first_text(stages, ["design"], ("coordination_model", "coordination", "orchestration_model"))
        roles = _project_roles(_first_list(stages, ["design", "define", "discover"], ("roles", "actors", "stakeholders")), "stage")
        tools = _project_tools(_first_list(stages, ["tools", "design"], ("tools", "tool_inventory", "recommended_tools")), "stage")
        tool_contracts = _tool_contracts(_first_list(stages, ["tools"], ("tool_contracts", "contracts", "recommended_tools")), "stage")
        memory = _first_text(stages, ["memory", "design"], ("memory_strategy", "memory", "state_strategy"))
        rag_required = _first_bool(stages, ["memory"], ("rag_required", "retrieval_required", "requires_rag"))
        knowledge_sources = _knowledge_sources(_first_list(stages, ["memory"], ("knowledge_sources", "sources", "retrieval_sources")), "stage")
        guardrails = _first_list(stages, ["design", "validate"], ("guardrails", "safety_rules", "policies"))
        risks = _first_list(stages, ["define", "estimate", "validate"], ("risks", "primary_risk", "risk_register"))
        acceptance = _first_list(stages, ["validate", "define"], ("acceptance_criteria", "criteria", "success_criteria", "validation_criteria"))
        scenarios = _first_list(stages, ["validate"], ("validation_scenarios", "test_scenarios", "tests"))
        estimation = _estimation_summary(_first_dict(stages, ["estimate"], ("estimation_report", "estimate", "estimation")))

        # Artifact context is approved but not necessarily stage-shaped. It only fills gaps.
        problem = problem or _first_text(artifacts, [], ("problem_statement", "description", "problem"))
        desired = desired or _first_text(artifacts, [], ("desired_outcome", "outcome", "target_outcome"))
        architecture = architecture or _first_text(artifacts, [], ("architecture", "architecture_pattern", "topology"))
        reasoning = reasoning or _first_text(artifacts, [], ("reasoning_pattern", "cognitive_pattern", "reasoning"))
        memory = memory or _first_text(artifacts, [], ("memory_strategy", "memory", "state_strategy"))
        if not tools:
            tools = _project_tools(_first_list(artifacts, [], ("tools", "tool_inventory", "recommended_tools")), "artifact")

        # Snapshot fallback comes after approved stages and artifacts.
        problem = problem or _text(
            snapshot.get("problem_statement")
            or snapshot.get("description")
            or snapshot.get("problem")
            or snapshot.get("summary")
        )
        current_process = current_process or _text(snapshot.get("current_process"))
        current_user = current_user or _text(snapshot.get("current_user") or snapshot.get("primary_user"))
        desired = desired or _text(snapshot.get("desired_outcome") or snapshot.get("outcome") or snapshot.get("target_outcome"))
        objectives = objectives or _string_list(snapshot.get("objectives") or snapshot.get("goals") or snapshot.get("user_goal"))
        mvp_scope = mvp_scope or _string_list(snapshot.get("mvp_scope") or snapshot.get("scope") or snapshot.get("functional_requirements"))
        out_of_scope = out_of_scope or _string_list(snapshot.get("out_of_scope") or snapshot.get("exclusions"))
        constraints = constraints or _string_list(snapshot.get("constraints"))
        nondelegable = nondelegable or _string_list(snapshot.get("nondelegable_decisions") or snapshot.get("non_delegable_decisions"))
        architecture = architecture or _text(snapshot.get("architecture"))
        reasoning = reasoning or _text(snapshot.get("reasoning_pattern"))
        coordination = coordination or _text(snapshot.get("coordination_model"))
        memory = memory or _text(snapshot.get("memory_strategy"))
        if rag_required is None:
            rag_required = _bool_or_none(snapshot.get("rag_required") or snapshot.get("retrieval_required"))
        guardrails = guardrails or _string_list(snapshot.get("guardrails"))
        risks = risks or _string_list(snapshot.get("risks") or snapshot.get("primary_risk"))
        acceptance = acceptance or _string_list(snapshot.get("acceptance_criteria") or snapshot.get("success_metric"))
        scenarios = scenarios or _string_list(snapshot.get("validation_scenarios"))
        if not tools:
            tools = _project_tools(_string_list_or_objects(snapshot.get("tools")), "snapshot")
        if estimation is None:
            estimation = _estimation_summary(snapshot.get("estimation_report") if isinstance(snapshot.get("estimation_report"), dict) else None)

        construction = _construction_questions(construction_questions, sources)
        open_questions = [
            OpenQuestionSummary(
                question=item.question_text,
                source_ref=item.source_ref,
                blocking=item.blocking,
            )
            for item in construction
            if item.status not in {"answered", "resolved"} or not item.answer_text
        ]

        missing = _missing_fields(
            {
                "problem_statement": problem,
                "desired_outcome": desired,
                "architecture": architecture,
                "tools": tools,
                "memory_strategy": memory,
                "acceptance_criteria": acceptance,
            }
        )
        project_title = _text(payload.get("project_title") or payload.get("session_title") or snapshot.get("project_title") or snapshot.get("session_title"))
        source_ref = sorted(source.ref for source in sources)[0] if sources else "unknown"
        anchors = _specificity_anchors(
            [
                ("actor", current_user),
                ("problem", problem),
                ("outcome", desired),
                ("architecture", architecture),
                ("reasoning", reasoning),
                ("memory", memory),
                ("tool", tools[0].name if tools else ""),
                ("guardrail", guardrails[0] if guardrails else ""),
                ("criterion", acceptance[0] if acceptance else ""),
                ("risk", risks[0] if risks else ""),
            ],
            source_ref=source_ref,
        )

        context = cls(
            session_id=_uuid_or_none(payload.get("session_id") or snapshot.get("session_id")),
            workspace_id=_uuid_or_none(payload.get("workspace_id") or snapshot.get("workspace_id")),
            project_title=project_title or None,
            problem_statement=problem or None,
            current_process=current_process or None,
            current_user=current_user or None,
            desired_outcome=desired or None,
            objectives=objectives,
            mvp_scope=mvp_scope,
            out_of_scope=out_of_scope,
            constraints=constraints,
            nondelegable_decisions=nondelegable,
            architecture=architecture or None,
            reasoning_pattern=reasoning or None,
            coordination_model=coordination or None,
            roles=roles,
            tools=tools,
            tool_contracts=tool_contracts,
            memory_strategy=memory or None,
            rag_required=rag_required,
            knowledge_sources=knowledge_sources,
            guardrails=guardrails,
            risks=risks,
            acceptance_criteria=acceptance,
            validation_scenarios=scenarios,
            estimation_summary=estimation,
            open_questions=open_questions,
            construction_questions=construction,
            source_refs=sources,
            source_versions=source_versions,
            missing_fields=missing,
            specificity_anchors=anchors,
            estimated_input_tokens=_estimate_tokens(payload, stages, artifacts, snapshot, construction_questions),
        )
        return context.model_copy(
            update={
                "input_fingerprint": _input_fingerprint(
                    context,
                    deliverable_key=deliverable_key,
                    policy=policy,
                )
            }
        )


def merge_generation_contexts(primary: ProjectGenerationContext, fallback: ProjectGenerationContext) -> ProjectGenerationContext:
    updates: dict[str, object] = {}
    for field_name in (
        "session_id",
        "workspace_id",
        "project_title",
        "problem_statement",
        "current_process",
        "current_user",
        "desired_outcome",
        "architecture",
        "reasoning_pattern",
        "coordination_model",
        "memory_strategy",
        "rag_required",
        "estimation_summary",
    ):
        if getattr(primary, field_name) in (None, "", []):
            updates[field_name] = getattr(fallback, field_name)
    for field_name in (
        "objectives",
        "mvp_scope",
        "out_of_scope",
        "constraints",
        "nondelegable_decisions",
        "roles",
        "tools",
        "tool_contracts",
        "knowledge_sources",
        "guardrails",
        "risks",
        "acceptance_criteria",
        "validation_scenarios",
        "open_questions",
        "construction_questions",
        "specificity_anchors",
    ):
        if not getattr(primary, field_name):
            updates[field_name] = getattr(fallback, field_name)
    merged_refs = _unique_models([*primary.source_refs, *fallback.source_refs], "ref")
    merged_versions = _unique_models([*primary.source_versions, *fallback.source_versions], "source_ref")
    updates["source_refs"] = merged_refs
    updates["source_versions"] = merged_versions
    merged = primary.model_copy(update=updates)
    present = {
        "problem_statement": merged.problem_statement,
        "desired_outcome": merged.desired_outcome,
        "architecture": merged.architecture,
        "tools": merged.tools,
        "memory_strategy": merged.memory_strategy,
        "acceptance_criteria": merged.acceptance_criteria,
    }
    return merged.model_copy(
        update={
            "missing_fields": _missing_fields(present),
            "input_fingerprint": _input_fingerprint(merged, deliverable_key="", policy=None),
        }
    )


def _allowed_stage_keys(policy: DeliverableContextPolicy | dict[str, object] | None) -> set[str] | None:
    if policy is None:
        return None
    if isinstance(policy, DeliverableContextPolicy):
        refs = list(policy.short_term_refs)
    elif isinstance(policy, dict):
        refs = list(policy.get("short_term_refs") or policy.get("requested_refs") or [])
    else:
        refs = []
    allowed = {_stage_from_ref(ref) for ref in refs}
    allowed = {stage for stage in allowed if stage}
    return allowed or None


def _stage_from_ref(ref: object) -> str | None:
    normalized = str(ref or "").strip()
    if normalized.startswith("stage."):
        candidate = normalized.removeprefix("stage.")
    elif normalized.startswith("session."):
        candidate = normalized.removeprefix("session.")
    else:
        return None
    candidate = _SESSION_STAGE_ALIASES.get(candidate, candidate)
    return candidate if candidate in _STAGE_KEYS else None


def _filter_stages(value: object, allowed: set[str] | None) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {}
    for key, item in value.items():
        stage = _SESSION_STAGE_ALIASES.get(str(key), str(key))
        if stage in _STAGE_KEYS and (allowed is None or stage in allowed):
            result[stage] = item
    return result


def _first_text(container: dict[str, object], stage_order: list[str], aliases: tuple[str, ...]) -> str:
    found = _first_value(container, stage_order, aliases)
    return _text(found.value if found else None)


def _first_list(container: dict[str, object], stage_order: list[str], aliases: tuple[str, ...]) -> list[str]:
    found = _first_value(container, stage_order, aliases)
    return _string_list(found.value if found else None)


def _first_bool(container: dict[str, object], stage_order: list[str], aliases: tuple[str, ...]) -> bool | None:
    found = _first_value(container, stage_order, aliases)
    return _bool_or_none(found.value if found else None)


def _first_dict(container: dict[str, object], stage_order: list[str], aliases: tuple[str, ...]) -> dict[str, object] | None:
    found = _first_value(container, stage_order, aliases)
    return found.value if found and isinstance(found.value, dict) else None


def _first_value(container: dict[str, object], stage_order: list[str], aliases: tuple[str, ...]) -> _ValueWithSource | None:
    keys = stage_order or sorted(container)
    for stage in keys:
        value = container.get(stage) if stage_order else container.get(stage)
        found = _find_alias(value, aliases)
        if found not in (None, "", [], {}):
            return _ValueWithSource(value=found, source_ref=f"stage.{stage}" if stage in _STAGE_KEYS else f"artifact.{stage}", stage=stage if stage in _STAGE_KEYS else None)
    return None


def _find_alias(value: object, aliases: tuple[str, ...]) -> object | None:
    if isinstance(value, dict):
        for alias in aliases:
            if alias in value:
                return value[alias]
        for child in value.values():
            found = _find_alias(child, aliases)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_alias(child, aliases)
            if found not in (None, "", [], {}):
                return found
    return None


def _text(value: object, *, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    elif isinstance(value, dict):
        for key in ("value", "statement", "description", "title", "name", "label", "content", "text", "question_text"):
            if key in value:
                nested = _text(value.get(key), limit=limit)
                if nested:
                    return nested
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    return " ".join(text.split()).strip()[:limit]


def _string_list(value: object) -> list[str]:
    items = _string_list_or_objects(value)
    return [_text(item, limit=350) for item in items if _text(item, limit=350)]


def _string_list_or_objects(value: object) -> list[object]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "si", "sí", "required", "requerido"}:
            return True
        if normalized in {"false", "no", "not_required", "no_requerido"}:
            return False
    return None


def _project_roles(items: list[object], source: str) -> list[ProjectRole]:
    roles: list[ProjectRole] = []
    for item in items[:12]:
        text = _text(item)
        if text:
            roles.append(ProjectRole(value=text, source_ref=source, confidence="medium"))
    return roles


def _project_tools(items: list[object], source: str) -> list[ProjectTool]:
    tools: list[ProjectTool] = []
    for item in items[:20]:
        if isinstance(item, dict):
            name = _text(item.get("name") or item.get("tool_name") or item.get("label"))
            purpose = _text(item.get("purpose") or item.get("description") or item.get("rationale"))
            requires_approval = _bool_or_none(item.get("requires_approval"))
            has_side_effects = _bool_or_none(item.get("has_side_effects"))
        else:
            name = _text(item)
            purpose = ""
            requires_approval = None
            has_side_effects = None
        if name:
            tools.append(
                ProjectTool(
                    name=name,
                    purpose=purpose,
                    source_ref=source,
                    requires_approval=requires_approval,
                    has_side_effects=has_side_effects,
                )
            )
    return tools


def _tool_contracts(items: list[object], source: str) -> list[ToolContractSummary]:
    contracts: list[ToolContractSummary] = []
    for item in items[:20]:
        if isinstance(item, dict):
            name = _text(item.get("name") or item.get("tool_name") or item.get("label"))
            purpose = _text(item.get("purpose") or item.get("description") or item.get("rationale"))
            category = str(item.get("category") or "design_contract")
            if category not in {"design_contract", "operational_binding", "pending_binding"}:
                category = "design_contract"
        else:
            name = _text(item)
            purpose = ""
            category = "design_contract"
        if name:
            contracts.append(ToolContractSummary(name=name, purpose=purpose, category=category, source_ref=source))
    return contracts


def _knowledge_sources(items: list[object], source: str) -> list[KnowledgeSourceSummary]:
    sources: list[KnowledgeSourceSummary] = []
    for item in items[:20]:
        if isinstance(item, dict):
            name = _text(item.get("name") or item.get("title") or item.get("source"))
            source_type = _text(item.get("source_type") or item.get("type"))
            permissions = _text(item.get("permissions") or item.get("access_policy"))
        else:
            name = _text(item)
            source_type = ""
            permissions = ""
        if name:
            sources.append(KnowledgeSourceSummary(name=name, source_type=source_type, permissions=permissions, source_ref=source))
    return sources


def _estimation_summary(value: dict[str, object] | None) -> EstimationSummary | None:
    if not value:
        return None
    return EstimationSummary(
        traditional_hours=_float_or_none(value.get("traditional_hours")),
        agentic_hours=_float_or_none(value.get("agentic_hours")),
        traditional_cost=_float_or_none(value.get("traditional_cost")),
        agentic_cost=_float_or_none(value.get("agentic_cost")),
        net_savings=_float_or_none(value.get("net_savings")),
        effort_reduction_percent=_float_or_none(value.get("effort_reduction_percent")),
        confidence_label=_text(value.get("confidence_label")),
        source_ref="stage.estimate",
    )


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _construction_questions(items: list[object], sources: list[SourceReference]) -> list[ConstructionQuestionSummary]:
    result: list[ConstructionQuestionSummary] = []
    refs = [source.ref for source in sources if source.source_type == "acp_question"]
    for index, item in enumerate(items[:30]):
        if isinstance(item, dict):
            question = _text(item.get("question_text") or item.get("question") or item.get("gap_title"))
            answer = _text(item.get("answer_text") or item.get("answer"))
            result.append(
                ConstructionQuestionSummary(
                    question_key=_text(item.get("question_key") or item.get("gap_key")),
                    question_text=question,
                    answer_text=answer,
                    status=_text(item.get("status")),
                    impacted_artifacts=_string_list(item.get("impacted_artifacts")),
                    source_ref=refs[index] if index < len(refs) else f"acp-question:{index + 1}",
                    blocking=_bool_or_none(item.get("blocking")),
                )
            )
    return [item for item in result if item.question_text]


def _source_refs_from_payload(
    payload: dict[str, object],
    stages: dict[str, object],
    artifacts: dict[str, object],
    has_snapshot: bool,
    questions: list[object],
) -> list[SourceReference]:
    refs: list[SourceReference] = []
    raw_refs = payload.get("approved_context_refs") or payload.get("source_refs") or []
    if isinstance(raw_refs, list):
        for ref in raw_refs:
            text = str(ref or "").strip()
            if text:
                refs.append(SourceReference(ref=text, source_type=_source_type_for_ref(text), stage=_stage_from_ref(text), confidence="high"))
    for stage in stages:
        refs.append(SourceReference(ref=f"stage.{stage}", source_type="stage", stage=stage, confidence="medium"))
    for key in artifacts:
        refs.append(SourceReference(ref=f"artifact.{key}", source_type="artifact", artifact_key=str(key), confidence="medium"))
    if has_snapshot:
        refs.append(SourceReference(ref="snapshot", source_type="snapshot", confidence="medium"))
    for index, _question in enumerate(questions):
        refs.append(SourceReference(ref=f"acp-question:{index + 1}", source_type="acp_question", confidence="medium"))
    return _unique_models(refs, "ref")


def _source_type_for_ref(ref: str) -> Literal["stage", "artifact", "snapshot", "acp_question", "task_ref", "unknown"]:
    if ref.startswith("journey:") or ref.startswith("stage.") or ref.startswith("session."):
        return "stage"
    if ref.startswith("artifact:") or ref.startswith("artifact."):
        return "artifact"
    if ref.startswith("acp-question:"):
        return "acp_question"
    if ref == "snapshot":
        return "snapshot"
    return "task_ref"


def _version_from_ref(ref: str) -> str:
    if ":v" in ref:
        return ref.rsplit(":v", 1)[-1]
    return ""


def _missing_fields(values: dict[str, object]) -> list[MissingField]:
    expectations = {
        "problem_statement": "discover",
        "desired_outcome": "discover",
        "architecture": "design",
        "tools": "tools",
        "memory_strategy": "memory",
        "acceptance_criteria": "validate",
    }
    missing: list[MissingField] = []
    for field, value in values.items():
        empty = value in (None, "", [], {})
        if empty:
            missing.append(MissingField(field=field, reason="approved_source_not_available", expected_stage=expectations.get(field)))
    return missing


def _specificity_anchors(items: list[tuple[str, object]], *, source_ref: str) -> list[SpecificityAnchor]:
    banned = {"usuario", "herramienta", "gobernanza", "ia", "agente", "proceso"}
    anchors: list[SpecificityAnchor] = []
    seen: set[str] = set()
    for anchor_type, raw in items:
        value = _text(raw, limit=140)
        normalized = value.lower()
        if not value or normalized in banned or normalized in seen:
            continue
        seen.add(normalized)
        anchors.append(SpecificityAnchor(anchor_type=anchor_type, value=value, source_ref=source_ref))
    return anchors[:12]


def _estimate_tokens(*values: object) -> int:
    chars = len(json.dumps(values, ensure_ascii=True, sort_keys=True, default=str))
    return max(1, chars // 4)


def _uuid_or_none(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value)) if value else None
    except (TypeError, ValueError):
        return None


def _artifact_map(value: dict[str, object] | list[object]) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    result: dict[str, object] = {}
    for index, item in enumerate(value or []):
        if isinstance(item, dict):
            key = _text(item.get("deliverable_key") or item.get("artifact_key") or item.get("key")) or f"artifact_{index + 1}"
            result[key] = item
    return result


def _unique_models(items: list[Any], field_name: str) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for item in items:
        value = str(getattr(item, field_name, "") or "")
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(item)
    return result


def _canonical(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])) if item not in (None, "", [], {})}
    if isinstance(value, list):
        canonical_items = [_canonical(item) for item in value if item not in (None, "", [], {})]
        return sorted(canonical_items, key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True, default=str))
    return value


def _policy_basis(policy: DeliverableContextPolicy | dict[str, object] | None) -> object:
    if isinstance(policy, DeliverableContextPolicy):
        return policy.model_dump(mode="json")
    if isinstance(policy, dict):
        return policy
    return {}


def _input_fingerprint(
    context: ProjectGenerationContext,
    *,
    deliverable_key: str,
    policy: DeliverableContextPolicy | dict[str, object] | None,
) -> str:
    payload = context.model_dump(mode="json", exclude={"input_fingerprint", "estimated_input_tokens"})
    basis = {
        "context_version": CONTEXT_VERSION,
        "builder_version": BUILDER_VERSION,
        "deliverable_key": deliverable_key,
        "policy": _policy_basis(policy),
        "context": payload,
    }
    return sha256(json.dumps(_canonical(basis), ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()
