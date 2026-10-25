from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import networkx as nx

from src.memory_baselines.base import BaseMemoryBackend, MemoryRecord
from src.utils.model_clients import EmbeddingClient, load_model_config


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)


class AccessControlledMemoryBackend(BaseMemoryBackend):
    """B4 collaborative memory baseline with private/shared tiers and ACL reads.

    This is a minimal reproduction of the collaborative-memory baseline:
    memory fragments are written into private or shared tiers, every read is
    filtered by a dynamic user-agent-resource access graph, and provenance is
    carried as metadata. It intentionally does not implement MAP-Graph
    path-based trust propagation or action-time blocking.
    """

    method_name = "B4"

    def __init__(self, top_k: int | None = None):
        config = load_model_config()
        self.top_k = top_k or int(config.get("retrieval", {}).get("top_k", 5))
        self.embedder = EmbeddingClient()
        self._client = None
        self._collections: dict[str, Any] = {}
        self._records: dict[str, MemoryRecord] = {}
        self._records_by_tier: dict[str, dict[str, MemoryRecord]] = {"private": {}, "shared": {}}
        self._access_graph = nx.DiGraph()
        self._source_policies: dict[str, dict[str, Any]] = {}
        self._task_agents: list[str] = []
        self._observed_source_keys: set[tuple[str, str]] = set()
        self._observed_tool_output_keys: set[tuple[str, str]] = set()
        self._seeded_task_id: str | None = None
        self._counter = 0
        self._last_retrieval: dict[str, Any] = {}

    def reset_task(self, task_id: str) -> None:
        super().reset_task(task_id)
        self._collections = {}
        self._records = {}
        self._records_by_tier = {"private": {}, "shared": {}}
        self._access_graph = nx.DiGraph()
        self._source_policies = {}
        self._task_agents = []
        self._observed_source_keys = set()
        self._observed_tool_output_keys = set()
        self._seeded_task_id = None
        self._counter = 0
        self._last_retrieval = {}

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "B4 access_controlled_memory requires Chroma. Install it with:\n"
                "  conda activate mapgraph\n"
                "  pip install chromadb\n"
            ) from exc

        self._client = chromadb.EphemeralClient()

    def observe_sources(self, agent_id: str, sources: list[dict], task: dict, metadata: dict) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        written = []
        for source in sources:
            source_id = source.get("source_id")
            if not source_id:
                continue
            self._register_source_policy(source, task)

            source_key = (agent_id, source_id)
            if source_key in self._observed_source_keys:
                continue
            self._observed_source_keys.add(source_key)

            policy = self._policy_from_source(source, task)
            tier = self._write_tier_for_source(source, task)
            record = self._make_record(
                owner_agent=agent_id,
                content=str(source.get("content", "")),
                visibility=tier,
                source_id=str(source_id),
                derived_from=[str(source_id)],
                trust_label=self._trust_label_from_payload(source),
                metadata={
                    **metadata,
                    "origin": "source_document",
                    "tier": tier,
                    "write_policy": "source_private_or_restricted_to_private_else_shared",
                    "read_policy": "filter_by_current_acl_before_similarity_return",
                    "provenance": self._provenance(
                        creator_agent=agent_id,
                        owner_agent=policy["owner_agent"],
                        source_id=str(source_id),
                        derived_from=[str(source_id)],
                        source_payload=source,
                    ),
                    **policy,
                },
            )
            self._add_record(record, tier=tier, permissions=policy["permission_scope"], owner=policy["owner_agent"])
            written.append(record)
        return written

    def observe_tool_outputs(
        self,
        agent_id: str,
        tool_outputs: list[dict],
        task: dict,
        metadata: dict,
    ) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        written = []
        for tool_output in tool_outputs:
            output_id = tool_output.get("tool_output_id") or tool_output.get("output_id") or tool_output.get("id")
            if not output_id:
                continue
            output_key = (agent_id, str(output_id))
            if output_key in self._observed_tool_output_keys:
                continue
            self._observed_tool_output_keys.add(output_key)

            content = tool_output.get("content")
            if content is None:
                content = tool_output.get("output")
            if content is None:
                content = tool_output.get("result")
            if content is None:
                content = json.dumps(tool_output, ensure_ascii=False)

            tier = self._write_tier_for_payload(tool_output, task, default_owner=agent_id)
            permission_scope = self._permission_scope(tool_output, task, default_owner=agent_id)
            owner = str(tool_output.get("owner") or tool_output.get("owner_agent") or agent_id)
            record = self._make_record(
                owner_agent=owner,
                content=str(content),
                visibility=tier,
                source_id=None,
                derived_from=[str(output_id)],
                trust_label=self._trust_label_from_payload(tool_output),
                metadata={
                    **metadata,
                    "origin": "tool_output",
                    "tool_output_id": str(output_id),
                    "tier": tier,
                    "permission_scope": permission_scope,
                    "owner_agent": owner,
                    "write_policy": "tool_output_private_or_restricted_to_private_else_shared",
                    "read_policy": "filter_by_current_acl_before_similarity_return",
                    "provenance": self._provenance(
                        creator_agent=agent_id,
                        owner_agent=owner,
                        source_id=None,
                        derived_from=[str(output_id)],
                        source_payload=tool_output,
                    ),
                },
            )
            self._add_record(record, tier=tier, permissions=permission_scope, owner=owner)
            written.append(record)
        return written

    def retrieve(self, agent_id: str, query: str, task: dict) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        if not self._records:
            return []

        query_embedding = self.embedder.embed(query)[0]
        candidates: list[tuple[float, MemoryRecord]] = []
        candidate_ids: list[str] = []
        denied_ids: list[str] = []
        for tier in ("private", "shared"):
            records = self._records_by_tier.get(tier, {})
            if not records:
                continue
            collection = self._collections.get(tier)
            if collection is None:
                continue
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=len(records),
                include=["distances"],
            )
            ids = result.get("ids", [[]])[0]
            distances = result.get("distances", [[]])[0]
            for memory_id, distance in zip(ids, distances):
                record = records.get(memory_id)
                if record is None:
                    continue
                candidate_ids.append(record.memory_id)
                if not self._can_read(agent_id, record):
                    denied_ids.append(record.memory_id)
                    continue
                score = 1.0 - float(distance)
                candidates.append((score, record))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [record for _, record in candidates[: self.top_k]]
        self._last_retrieval = {
            "agent_id": agent_id,
            "candidate_memory_ids": candidate_ids,
            "permission_denied_memory_ids": denied_ids,
            "selected_memory_ids": [record.memory_id for record in selected],
        }
        return selected

    def write(self, agent_id: str, content: str, metadata: dict) -> MemoryRecord | None:
        if not content or not content.strip():
            return None

        tier, permission_scope, derived_from, inherited = self._write_policy_for_agent_message(agent_id, metadata)
        record = self._make_record(
            owner_agent=agent_id,
            content=content,
            visibility=tier,
            source_id=None,
            derived_from=derived_from,
            trust_label=inherited.get("trust_label"),
            metadata={
                **metadata,
                "origin": "agent_message",
                "tier": tier,
                "permission_scope": permission_scope,
                "owner_agent": agent_id,
                "write_policy": "inherit_private_or_restricted_inputs_else_shared",
                "read_policy": "filter_by_current_acl_before_similarity_return",
                "provenance": self._provenance(
                    creator_agent=agent_id,
                    owner_agent=agent_id,
                    source_id=None,
                    derived_from=derived_from,
                    source_payload=inherited,
                ),
            },
        )
        self._add_record(record, tier=tier, permissions=permission_scope, owner=agent_id)
        return record

    def all_memories(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def last_retrieval_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_retrieval)

    def apply_timeline_event(self, event: dict, task: dict) -> None:
        if event.get("event_type") != "revoke_permission":
            return
        source_id = str(event.get("source_id") or "")
        if not source_id:
            return
        policy = self._source_policies.setdefault(source_id, {})
        policy.update({"is_revoked": True, "source_visibility": "revoked", "permission_scope": []})
        affected_ids = set(str(item) for item in event.get("derived_from") or [])
        affected_ids.add(source_id)
        for record in self._records.values():
            if source_id not in set(record.derived_from) and record.memory_id not in affected_ids:
                continue
            record.metadata["is_revoked"] = True
            record.metadata["source_visibility"] = "revoked"
            record.metadata["permission_scope"] = []
            resource = self._resource_node(record.memory_id)
            agent_nodes = [node for node in self._access_graph.predecessors(resource) if str(node).startswith("agent:")]
            for agent_node in agent_nodes:
                if self._access_graph.has_edge(agent_node, resource):
                    self._access_graph.remove_edge(agent_node, resource)

    def export_graph(self) -> dict[str, Any]:
        return {
            "task_id": getattr(self, "task_id", "unknown_task"),
            "method": self.method_name,
            "graph_type": "access_control_graph",
            "node_counts": {
                "access_control": self._access_graph.number_of_nodes(),
            },
            "edge_counts": {
                "access_control": self._access_graph.number_of_edges(),
            },
            "nodes": [
                {"id": node_id, **data}
                for node_id, data in self._access_graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target, **data}
                for source, target, data in self._access_graph.edges(data=True)
            ],
        }

    def _ensure_task_seeded(self, task: dict) -> None:
        task_id = task.get("task_id", getattr(self, "task_id", "unknown_task"))
        if self._seeded_task_id == task_id:
            return

        self._task_agents = [str(agent) for agent in task.get("agent_roles", [])]
        for source in task.get("source_documents", []):
            self._register_source_policy(source, task)

        for event in task.get("initial_memory_events", task.get("memory_events", [])):
            self._seed_memory_event(event, task)
        self._seeded_task_id = task_id

    def _seed_memory_event(self, event: dict, task: dict) -> None:
        memory_id = event.get("memory_id")
        if not memory_id or memory_id in self._records:
            return

        owner = str(event.get("owner_agent") or event.get("agent_id") or event.get("owner") or "unknown_owner")
        permission_scope = self._permission_scope(event, task, default_owner=owner)
        tier = self._write_tier_for_payload(event, task, default_owner=owner)
        source_id = event.get("source_id")
        derived_from = [str(item) for item in event.get("derived_from") or ([source_id] if source_id else [])]
        record = MemoryRecord(
            memory_id=_safe_id(str(memory_id)),
            content=str(event.get("content", "")),
            owner_agent=owner,
            visibility=tier,
            source_id=str(source_id) if source_id else None,
            derived_from=derived_from,
            trust_label=event.get("trust_label"),
            metadata={
                "task_id": task.get("task_id"),
                "domain": task.get("domain"),
                "event_type": event.get("event_type"),
                "origin": "seed_memory_event",
                "event_id": event.get("event_id"),
                "tier": tier,
                "permission_scope": permission_scope,
                "owner_agent": owner,
                "write_policy": "seeded_event_private_or_restricted_to_private_else_shared",
                "read_policy": "filter_by_current_acl_before_similarity_return",
                "source_visibility": event.get("visibility"),
                "is_private": bool(event.get("visibility") == "private"),
                "is_revoked": bool(event.get("visibility") == "revoked" or event.get("trust_label") == "revoked"),
                "provenance": self._provenance(
                    creator_agent=str(event.get("agent_id") or owner),
                    owner_agent=owner,
                    source_id=str(source_id) if source_id else None,
                    derived_from=derived_from,
                    source_payload=event,
                ),
            },
        )
        self._add_record(record, tier=tier, permissions=permission_scope, owner=owner)

    def _register_source_policy(self, source: dict, task: dict) -> None:
        source_id = source.get("source_id")
        if not source_id:
            return
        self._source_policies[str(source_id)] = self._policy_from_source(source, task)

    def _policy_from_source(self, source: dict, task: dict) -> dict[str, Any]:
        owner = str(source.get("owner") or source.get("owner_agent") or "unknown_owner")
        return {
            "permission_scope": self._permission_scope(source, task, default_owner=owner),
            "owner_agent": owner,
            "source_visibility": source.get("visibility", "shared"),
            "is_private": bool(source.get("is_private")),
            "is_revoked": bool(source.get("is_revoked")) or source.get("visibility") == "revoked",
            "risk_tags": source.get("risk_tags", []),
        }

    def _permission_scope(self, payload: dict, task: dict, default_owner: str) -> list[str]:
        if payload.get("visibility") == "revoked" or payload.get("is_revoked") or payload.get("trust_label") == "revoked":
            return []
        scope = payload.get("permission_scope")
        if scope:
            return [str(item) for item in scope]
        if payload.get("visibility") == "shared":
            return [str(agent) for agent in task.get("agent_roles", [])]
        return [str(default_owner)]

    def _write_tier_for_source(self, source: dict, task: dict) -> str:
        owner = str(source.get("owner") or source.get("owner_agent") or "unknown_owner")
        return self._write_tier_for_payload(source, task, default_owner=owner)

    def _write_tier_for_payload(self, payload: dict, task: dict, default_owner: str) -> str:
        visibility = payload.get("visibility", "shared")
        permission_scope = set(self._permission_scope(payload, task, default_owner=default_owner))
        all_agents = set(str(agent) for agent in task.get("agent_roles", []))
        if payload.get("is_private") or visibility in {"private", "isolated"}:
            return "private"
        if visibility == "shared" and (not all_agents or permission_scope >= all_agents):
            return "shared"
        return "private"

    def _write_policy_for_agent_message(self, agent_id: str, metadata: dict) -> tuple[str, list[str], list[str], dict]:
        retrieved_ids = [str(item) for item in metadata.get("retrieved_memory_ids", []) if item]
        visible_source_ids = [str(item) for item in metadata.get("visible_source_ids", []) if item]
        derived_from = retrieved_ids + visible_source_ids
        inherited: dict[str, Any] = {
            "derived_from_retrieved_memory_ids": retrieved_ids,
            "derived_from_visible_source_ids": visible_source_ids,
        }

        private_inputs: list[MemoryRecord] = []
        permission_sets: list[set[str]] = []
        for memory_id in retrieved_ids:
            record = self._records.get(memory_id)
            if record is None:
                continue
            permissions = set(record.metadata.get("permission_scope") or [record.owner_agent])
            permission_sets.append(permissions)
            if record.visibility == "private" or record.metadata.get("tier") == "private":
                private_inputs.append(record)

        for source_id in visible_source_ids:
            policy = self._source_policies.get(source_id)
            if policy is None:
                continue
            permissions = set(policy.get("permission_scope") or [agent_id])
            permission_sets.append(permissions)
            if policy.get("is_private") or policy.get("source_visibility") != "shared":
                inherited["source_policy"] = policy
                private_inputs.append(
                    MemoryRecord(
                        memory_id=f"source:{source_id}",
                        content="",
                        owner_agent=str(policy.get("owner_agent") or agent_id),
                        visibility="private",
                    )
                )

        if private_inputs:
            allowed = set.intersection(*permission_sets) if permission_sets else {agent_id}
            allowed.add(agent_id)
            inherited["inherited_private_inputs"] = [memory.memory_id for memory in private_inputs]
            return "private", sorted(allowed), derived_from, inherited

        return "shared", sorted(set(self._task_agents or [agent_id])), derived_from, inherited

    def _add_record(self, record: MemoryRecord, tier: str, permissions: list[str], owner: str) -> None:
        collection = self._collection_for_tier(tier)
        embedding = self.embedder.embed(record.content)[0]
        collection.add(
            ids=[record.memory_id],
            embeddings=[embedding],
            documents=[record.content],
            metadatas=[self._metadata_for_chroma(record)],
        )
        record.metadata["tier"] = tier
        record.metadata["permission_scope"] = permissions
        record.metadata["owner_agent"] = owner
        self._records[record.memory_id] = record
        self._records_by_tier.setdefault(tier, {})[record.memory_id] = record
        self._update_access_graph(record, tier=tier, permissions=permissions, owner=owner)

    def _collection_for_tier(self, tier: str):
        if self._client is None:
            raise RuntimeError("Chroma client has not been initialized. Call reset_task first.")
        if tier not in self._collections:
            task_id = getattr(self, "task_id", "task")
            collection_hash = hashlib.sha1(f"{task_id}:{tier}".encode("utf-8")).hexdigest()[:12]
            collection_name = _safe_id(f"access_controlled_{tier}_{collection_hash}")
            self._collections[tier] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[tier]

    def _update_access_graph(self, record: MemoryRecord, tier: str, permissions: list[str], owner: str) -> None:
        owner_node = self._user_node(owner)
        resource_node = self._resource_node(record.memory_id)
        self._access_graph.add_node(owner_node, node_type="user", user_id=owner)
        self._access_graph.add_node(resource_node, node_type="resource", memory_id=record.memory_id, tier=tier)
        self._access_graph.add_edge(owner_node, resource_node, relation="owns")

        if tier == "shared":
            shared_node = self._user_node("shared_workspace")
            self._access_graph.add_node(shared_node, node_type="user", user_id="shared_workspace")
            self._access_graph.add_edge(shared_node, resource_node, relation="shared_tier")

        for agent_id in permissions:
            agent_node = self._agent_node(agent_id)
            self._access_graph.add_node(agent_node, node_type="agent", agent_id=agent_id)
            self._access_graph.add_edge(owner_node, agent_node, relation="grants_read")
            self._access_graph.add_edge(agent_node, resource_node, relation="can_read")
            self._access_graph.add_edge(agent_node, resource_node, relation="can_write_derived")

    def _can_read(self, agent_id: str, record: MemoryRecord) -> bool:
        if record.metadata.get("is_revoked") or record.metadata.get("source_visibility") == "revoked":
            return False
        return self._access_graph.has_edge(self._agent_node(agent_id), self._resource_node(record.memory_id))

    def _make_record(
        self,
        owner_agent: str,
        content: str,
        visibility: str,
        source_id: str | None,
        derived_from: list[str],
        trust_label: str | None,
        metadata: dict[str, Any],
    ) -> MemoryRecord:
        self._counter += 1
        task_id = getattr(self, "task_id", "task")
        prefix = str(metadata.get("origin") or "memory")
        memory_id = _safe_id(f"{prefix}_{task_id}_{owner_agent}_{self._counter:04d}")
        return MemoryRecord(
            memory_id=memory_id,
            content=content,
            owner_agent=owner_agent,
            visibility=visibility,
            source_id=source_id,
            derived_from=derived_from,
            trust_label=trust_label,
            metadata=metadata,
        )

    def _metadata_for_chroma(self, record: MemoryRecord) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "memory_id": record.memory_id,
            "owner_agent": record.owner_agent,
            "visibility": record.visibility,
        }
        for key, value in record.metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = json.dumps(value, ensure_ascii=True)
        return metadata

    def _provenance(
        self,
        creator_agent: str,
        owner_agent: str,
        source_id: str | None,
        derived_from: list[str],
        source_payload: dict,
    ) -> dict[str, Any]:
        return {
            "creator_agent": creator_agent,
            "owner_agent": owner_agent,
            "source_id": source_id,
            "derived_from": derived_from,
            "source_visibility": source_payload.get("visibility"),
            "source_permission_scope": source_payload.get("permission_scope"),
            "source_trust_label": source_payload.get("trust_label"),
            "source_trust": source_payload.get("trust"),
            "source_risk_tags": source_payload.get("risk_tags"),
            "source_is_private": source_payload.get("is_private"),
            "source_is_revoked": source_payload.get("is_revoked"),
        }

    def _trust_label_from_payload(self, payload: dict) -> str | None:
        if payload.get("trust_label"):
            return str(payload.get("trust_label"))
        risk_tags = payload.get("risk_tags") or []
        if "private" in risk_tags:
            return "private"
        if "revoked" in risk_tags or payload.get("is_revoked"):
            return "revoked"
        if "poisoned" in risk_tags or payload.get("is_poisoned"):
            return "poisoned"
        if "trusted" in risk_tags:
            return "trusted"
        return None

    def _agent_node(self, agent_id: str) -> str:
        return f"agent:{agent_id}"

    def _user_node(self, user_id: str) -> str:
        return f"user:{user_id}"

    def _resource_node(self, memory_id: str) -> str:
        return f"resource:{memory_id}"
