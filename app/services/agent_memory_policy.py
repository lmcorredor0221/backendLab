from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.contracts.canonical_v1 import KnowledgeManifestSourceV1


_SPECIALIST_ROLE_SUFFIX = "_specialist"
_BRANCH_NAMESPACE_RE = re.compile(r"[^a-z0-9]+")
_KNOWLEDGE_NAMESPACE_WRITERS = {
    "knowledge.canonical_docs": frozenset({"ingestion_pipeline"}),
    "knowledge.operational_docs": frozenset({"ingestion_pipeline"}),
    "knowledge.research_refs": frozenset({"ingestion_pipeline"}),
    "knowledge.golden_fixtures": frozenset({"ingestion_pipeline"}),
    "knowledge.visual_refs": frozenset({"ingestion_pipeline"}),
    "knowledge.search_cache": frozenset({"retrieval_service"}),
}
_ROLE_TOKEN_ALIASES = {
    "retrieval_lane": {"retrieval"},
    "tool_lane": {"tool_use"},
    "evaluation_specialist": {"evaluator", "specialist"},
    "risk_specialist": {"specialist"},
    "artifact_specialist": {"specialist"},
    "subagent": {"specialist"},
}


@dataclass(frozen=True)
class AgentShortTermMemoryView:
    role: str
    normalized_roles: tuple[str, ...]
    visible_namespaces: list[Any]
    hidden_namespace_keys: tuple[str, ...]


@dataclass(frozen=True)
class MemoryNamespaceAccessDecision:
    role: str
    normalized_roles: tuple[str, ...]
    namespace: str
    action: str
    allowed: bool
    reason: str
    branch_key: str = ""


def branch_namespace_for_key(branch_key: str) -> str:
    normalized = _BRANCH_NAMESPACE_RE.sub(".", branch_key.strip().lower()).strip(".")
    return f"session.branch.{normalized or 'branch'}"


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


class AgentMemoryPolicyService:
    def resolve_role_tokens(self, role: str) -> tuple[str, ...]:
        normalized = _normalize_token(role)
        tokens = {normalized} if normalized else set()
        tokens.update(_ROLE_TOKEN_ALIASES.get(normalized, set()))
        if normalized.endswith(_SPECIALIST_ROLE_SUFFIX):
            tokens.add("specialist")
        return tuple(sorted(token for token in tokens if token))

    def short_term_memory_view(self, memory_payload: Any, *, role: str) -> AgentShortTermMemoryView:
        tokens = set(self.resolve_role_tokens(role))
        visible_namespaces: list[Any] = []
        hidden_namespace_keys: list[str] = []
        for namespace in self._iter_short_term_namespaces(memory_payload):
            read_roles = {_normalize_token(item) for item in getattr(namespace, "read_roles", []) if str(item).strip()}
            namespace_key = str(getattr(namespace, "namespace", "")).strip()
            if not namespace_key:
                continue
            if not read_roles or tokens & read_roles:
                visible_namespaces.append(namespace)
            else:
                hidden_namespace_keys.append(namespace_key)
        return AgentShortTermMemoryView(
            role=role,
            normalized_roles=tuple(sorted(tokens)),
            visible_namespaces=visible_namespaces,
            hidden_namespace_keys=tuple(hidden_namespace_keys),
        )

    def can_use_knowledge_source(self, *, role: str, source: KnowledgeManifestSourceV1) -> bool:
        allowed_usages = set(self.allowed_knowledge_memory_usages(role))
        memory_usage = _normalize_token(source.memory_usage)
        authority_level = _normalize_token(source.authority_level)
        if not allowed_usages or not memory_usage:
            return False
        if authority_level == "visual_reference" or memory_usage == "visual_only":
            return False
        return memory_usage in allowed_usages

    def allowed_knowledge_memory_usages(self, role: str) -> tuple[str, ...]:
        tokens = set(self.resolve_role_tokens(role))
        allowed: set[str] = set()

        if "planner" in tokens or "retrieval" in tokens or "memory" in tokens:
            allowed.update({"required_retrieval", "candidate_retrieval"})
        if "executor" in tokens:
            allowed.add("required_retrieval")
        if "evaluator" in tokens:
            allowed.update({"required_retrieval", "candidate_retrieval", "validation_only"})
        if "recovery" in tokens:
            allowed.update({"candidate_retrieval", "validation_only"})
        if "artifact_specialist" in tokens:
            allowed.update({"candidate_retrieval", "validation_only"})
        if "risk_specialist" in tokens:
            allowed.update({"required_retrieval", "candidate_retrieval"})

        return tuple(sorted(allowed))

    def evaluate_write_access(
        self,
        memory_payload: Any,
        *,
        role: str,
        namespace: str,
        branch_key: str = "",
    ) -> MemoryNamespaceAccessDecision:
        namespace_key = str(namespace or "").strip()
        tokens = set(self.resolve_role_tokens(role))
        normalized_roles = tuple(sorted(tokens))

        if not namespace_key:
            return MemoryNamespaceAccessDecision(
                role=role,
                normalized_roles=normalized_roles,
                namespace="",
                action="write",
                allowed=False,
                reason="namespace_empty",
                branch_key=branch_key,
            )

        namespace_entry = self._find_short_term_namespace(memory_payload, namespace_key)
        if namespace_entry is not None:
            allowed_roles = {
                _normalize_token(item)
                for item in getattr(namespace_entry, "write_roles", [])
                if str(item).strip()
            }
            if not allowed_roles or not tokens & allowed_roles:
                return MemoryNamespaceAccessDecision(
                    role=role,
                    normalized_roles=normalized_roles,
                    namespace=namespace_key,
                    action="write",
                    allowed=False,
                    reason="role_not_authorized_for_namespace",
                    branch_key=branch_key,
                )
            if namespace_key.startswith("session.branch.") and "specialist" in tokens and not (tokens & {"memory", "supervisor"}):
                expected_namespace = branch_namespace_for_key(branch_key) if branch_key.strip() else ""
                if not expected_namespace:
                    return MemoryNamespaceAccessDecision(
                        role=role,
                        normalized_roles=normalized_roles,
                        namespace=namespace_key,
                        action="write",
                        allowed=False,
                        reason="branch_key_required_for_specialist_write",
                        branch_key=branch_key,
                    )
                if expected_namespace != namespace_key:
                    return MemoryNamespaceAccessDecision(
                        role=role,
                        normalized_roles=normalized_roles,
                        namespace=namespace_key,
                        action="write",
                        allowed=False,
                        reason="specialist_branch_slot_mismatch",
                        branch_key=branch_key,
                    )
            return MemoryNamespaceAccessDecision(
                role=role,
                normalized_roles=normalized_roles,
                namespace=namespace_key,
                action="write",
                allowed=True,
                reason="role_authorized_by_namespace_contract",
                branch_key=branch_key,
            )

        if namespace_key.startswith("knowledge."):
            allowed_roles = _KNOWLEDGE_NAMESPACE_WRITERS.get(namespace_key, frozenset())
            if tokens & allowed_roles:
                return MemoryNamespaceAccessDecision(
                    role=role,
                    normalized_roles=normalized_roles,
                    namespace=namespace_key,
                    action="write",
                    allowed=True,
                    reason="role_authorized_for_knowledge_namespace",
                    branch_key=branch_key,
                )
            return MemoryNamespaceAccessDecision(
                role=role,
                normalized_roles=normalized_roles,
                namespace=namespace_key,
                action="write",
                allowed=False,
                reason="knowledge_namespaces_are_ingestion_only",
                branch_key=branch_key,
            )

        return MemoryNamespaceAccessDecision(
            role=role,
            normalized_roles=normalized_roles,
            namespace=namespace_key,
            action="write",
            allowed=False,
            reason="namespace_not_registered_in_policy",
            branch_key=branch_key,
        )

    def _find_short_term_namespace(self, memory_payload: Any, namespace: str) -> Any | None:
        for item in self._iter_short_term_namespaces(memory_payload):
            if str(getattr(item, "namespace", "")).strip() == namespace:
                return item
        return None

    def _iter_short_term_namespaces(self, memory_payload: Any) -> list[Any]:
        if memory_payload is None:
            return []
        if hasattr(memory_payload, "memory") and hasattr(memory_payload.memory, "namespaces"):
            return list(memory_payload.memory.namespaces)
        if hasattr(memory_payload, "namespaces"):
            return list(memory_payload.namespaces)
        return []
